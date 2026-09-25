"""
blackbox_pipeline.models.rf_router.router.base

Shared router machinery: input hygiene, the cascade cut, the probability
transform, and the fit report.

Everything both arms do identically lives here. ``SingleRFRouter`` and
``RFRouter`` differ only in how they produce scores and solve thresholds; the
contract they expose, the validation they apply and the way a score pair becomes
a decision are all defined once, in this module.

Train/test protocol
-------------------
Nothing here accepts test data, and no subclass may.

``fit`` takes the training split and nothing else. Forests are fitted on
training rows, probabilities for threshold selection are out-of-fold over those
same rows, and the operating point is solved from those probabilities. The test
split is touched exactly once, after ``fit`` has returned and both thresholds
are frozen, by passing it to ``predict``.

``resolve_at`` is the one method that changes a frozen operating point. It
re-solves from the STORED training OOF arrays and never refits, so it cannot
reach test data either — but a threshold re-solved after test metrics have been
seen is threshold shopping regardless of what the code permits. Re-solve before
you look, or report both.

``predict_train_oof`` exists so train-side routing decisions can be obtained
honestly, each from a model that did not train on that row. Stage 4 stacking
features must come from there rather than from ``predict(X_train)``, which is
in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..bands import BandConfidence
from ..config import RFRouterConfig
from ..threshhold import ThresholdPair, ThresholdSolver
from .decisions import (
    ABSTAIN,
    DECISION_DTYPE,
    NOT_SUBSCRIBE,
    PASS1,
    PASS2,
    SUBSCRIBE,
    UNCERTAIN,
)

__all__ = ["_FitReport", "_BaseRouter", "apply_cuts"]


def apply_cuts(p1: np.ndarray, p2: np.ndarray, t1: float, t2: float):
    """
    Turn a pair of score arrays into ``(preds, decisions)``.

    Parameters
    ----------
    p1 : numpy.ndarray
        Pass 1 scores for every row.
    p2 : numpy.ndarray
        Pass 2 scores, same length. May be NaN where Pass 2 did not score a row
        (outside the remainder); those rows can never be flagged. The single-RF
        ablation passes the same array for both.
    t1, t2 : float
        The frozen operating point.

    Returns
    -------
    preds : numpy.ndarray of int
        In {0, 1, -1}.
    decisions : numpy.ndarray of object
        In {"pass1", "pass2", "uncertain"}.

    Notes
    -----
    ``p1 < t1`` routes NOT_SUBSCRIBE; whatever Pass 1 leaves is flagged
    SUBSCRIBE when ``p2 > t2``; everything else abstains. Pass 1 claims first,
    mirroring the ``mask & (decisions == "uncertain")`` cascade in
    ``GlassRouterPipeline.predict``. Both comparisons are strict, matching the
    conventions the solver selected the thresholds under.
    """
    n = len(p1)
    preds = np.full(n, ABSTAIN, dtype=int)
    decisions = np.array([UNCERTAIN] * n, dtype=DECISION_DTYPE)

    routed = p1 < t1
    preds[routed] = NOT_SUBSCRIBE
    decisions[routed] = PASS1

    flagged = (~routed) & ~np.isnan(p2) & (p2 > t2)
    preds[flagged] = SUBSCRIBE
    decisions[flagged] = PASS2

    return preds, decisions


@dataclass
class _FitReport:
    """
    Everything the fit learned, for printing, saving and auditing.

    Attributes
    ----------
    mode : str
        Which experiment produced this — ablation or cascade.
    n_train : int
        Training rows seen.
    training_base_rate : float
        Positive rate over those rows. Used by ``predict_proba``.
    split_fingerprint : str or None
        Identifies the train/test split this operating point was solved under.
    thresholds, bands : dict
        The frozen operating point and the band statistics justifying it.
    oof_summary : dict
        Fold counts, seeds and scored-row counts for the OOF runs.
    remainder : dict
        Two-pass only: size, fraction, positives and base rate of the population
        Pass 2 was fitted on. Empty for the ablation.
    nested_check : dict or None
        Present when ``validate_nested`` ran. Carries the per-fold thresholds
        and the agreement verdict.
    """

    mode: str
    n_train: int
    training_base_rate: float
    split_fingerprint: Optional[str]
    thresholds: dict
    bands: dict
    oof_summary: dict
    remainder: dict = field(default_factory=dict)
    nested_check: Optional[dict] = None


class _BaseRouter:
    """
    Shared contract, validation and probability transform.

    Not instantiated directly. Subclasses supply ``fit``, ``predict``,
    ``predict_train_oof`` and ``_oof_for_solving``; everything else is inherited.

    Attributes
    ----------
    config : RFRouterConfig
        The validated configuration.
    solver : ThresholdSolver
        Bound to ``config.constraints`` at construction.
    is_fitted : bool
        Whether ``fit`` has completed.
    feature_names_ : list of str or None
        Column order seen at fit time. ``predict`` reorders against it.
    training_base_rate : float or None
        Positive rate over the training split.
    thresholds_ : ThresholdPair or None
        The frozen operating point.
    bands_ : BandConfidence or None
        Empirical band confidences, from training OOF.
    fit_report_ : _FitReport or None
        The audit record.
    split_fingerprint_ : str or None
        Recorded at fit, checked when reloading an artifact.
    last_scores_ : dict
        ``p1``/``p2`` from the most recent ``predict``, for diagnostics. Always
        reflects the last call and is not part of the contract.
    """

    def __init__(self, config: RFRouterConfig):
        self.config = config
        self.solver = ThresholdSolver(config.constraints, verbose=config.verbose)

        self.is_fitted = False
        self.feature_names_: Optional[list[str]] = None
        self.training_base_rate: Optional[float] = None
        self.thresholds_: Optional[ThresholdPair] = None
        self.bands_: Optional[BandConfidence] = None
        self.fit_report_: Optional[_FitReport] = None
        self.split_fingerprint_: Optional[str] = None
        self.last_scores_: dict = {}
        self._y_train_ = None

    # ------------------------------------------------------------------
    # Input hygiene
    # ------------------------------------------------------------------
    def _check_fit_inputs(
        self, X: pd.DataFrame, y
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate a training pair and coerce ``y`` to an aligned Series.

        Raises
        ------
        TypeError
            If ``X`` is not a DataFrame.
        ValueError
            On a length or index mismatch, NaN in ``X``, non-binary ``y``, or a
            non-binary ``X``.

        Notes
        -----
        The index check matters most: misaligned labels pass every length check
        and train on the wrong targets. The binary-``X`` check enforces that the
        router is being fed the 29 ``RF_FEATURES_BINARY`` bins rather than raw
        or engineered continuous columns — the same representation GLASS
        receives, which is what makes the arms comparable.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")

        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y), index=X.index)
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if not X.index.equals(y.index):
            raise ValueError(
                "X.index and y.index differ — reindex y to X before fitting "
                "(y = y.loc[X.index])."
            )
        if X.isna().any().any():
            raise ValueError("X contains NaN")
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        if not X.isin([0, 1]).all().all():
            raise ValueError(
                "X is not binary. The router consumes the 29 RF_FEATURES_BINARY "
                "bins — the same representation BankSegmentBuilder hands GLASS."
            )
        return X, y

    def _align_predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder / verify prediction columns against the fitted frame.

        sklearn is positional once fitted, so a column-order difference between
        fit and predict would silently permute features. Extra columns are
        dropped; missing ones raise.
        """
        if self.feature_names_ is None:
            raise ValueError("Call fit() first")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing fitted features: {missing}")
        return X[self.feature_names_]

    def _require_fitted(self) -> None:
        """Raise ``ValueError`` if ``fit`` has not completed."""
        if not self.is_fitted:
            raise ValueError("Call fit() first")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _oof_for_solving(self):
        """Return ``(p1, y1, p2, y2, n_positives_global)`` from stored OOF."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        """
        Route rows. Returns the ``(preds, conf, decisions)`` triple.

        Implemented by each subclass; see the package docstring for the contract.
        """
        raise NotImplementedError

    def predict_train_oof(self):
        """
        Honest train-side routing: every decision from a model that did not
        train on that row. Stage 4 stacking features must come from here.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------
    def predict_proba(
        self, X: pd.DataFrame, base_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Probabilistic output, using the SAME transform as
        ``GlassRouterPipeline.predict_proba``::

            pass1   -> P(1) = base_rate * (1 - conf)
            pass2   -> P(1) = base_rate + (1 - base_rate) * conf
            abstain -> the flat training base rate

        Parameters
        ----------
        X : pandas.DataFrame
            Rows to score, carrying the fitted feature columns.
        base_rate : float, optional
            Override the prior. Defaults to ``training_base_rate``.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            ``[P(NOT_SUBSCRIBE), P(SUBSCRIBE)]``.

        Notes
        -----
        A heuristic mapping of band precision onto a probability, not a
        calibration. Reproduced verbatim so the two arms' outputs are comparable
        objects; Platt or isotonic scaling on the RF arm alone would make them
        incomparable. Do not report these as calibrated probabilities.
        """
        self._require_fitted()
        preds, conf, decisions = self.predict(X)
        n = len(preds)

        if base_rate is None:
            base_rate = self.training_base_rate
        base_rate = float(base_rate)

        probas = np.full((n, 2), [1.0 - base_rate, base_rate], dtype=float)

        p1_mask = decisions == PASS1
        probas[p1_mask, 1] = base_rate * (1.0 - conf[p1_mask])
        probas[p1_mask, 0] = 1.0 - probas[p1_mask, 1]

        p2_mask = decisions == PASS2
        probas[p2_mask, 1] = base_rate + (1.0 - base_rate) * conf[p2_mask]
        probas[p2_mask, 0] = 1.0 - probas[p2_mask, 1]

        return probas

    # ------------------------------------------------------------------
    def resolve_at(self, constraints, apply: bool = False) -> ThresholdPair:
        """
        Re-solve ``t1``/``t2`` from the STORED OOF arrays. No refit.

        Parameters
        ----------
        constraints : OperatingConstraints
            The constraint set to solve against.
        apply : bool, default False
            When True, replace the frozen operating point, rebuild the band
            confidences and adopt ``constraints`` on the config. When False,
            return the pair and change nothing.

        Returns
        -------
        ThresholdPair

        Raises
        ------
        ValueError
            If either pass is infeasible under ``constraints``.

        Notes
        -----
        Cheap enough to sweep a constraint set without refitting forests. It
        reads training OOF only and cannot see test data — but re-solving after
        test metrics have been seen is threshold shopping whatever the code
        allows. Sweep before you look.
        """
        self._require_fitted()
        solver = ThresholdSolver(constraints, verbose=False)
        p1, y1, p2, y2, npos = self._oof_for_solving()

        r1 = solver.solve_pass1(p1, y1)
        r1.raise_if_infeasible(context="re-solve Pass 1")
        r2 = solver.solve_pass2(p2, y2, n_positives_global=npos)
        r2.raise_if_infeasible(context="re-solve Pass 2")

        pair = ThresholdPair(
            t1=float(r1.threshold), t2=float(r2.threshold), pass1=r1, pass2=r2
        )
        if apply:
            self.thresholds_ = pair
            self.bands_ = BandConfidence.from_oof(
                pass1_proba=p1, pass1_y=y1, t1=pair.t1,
                pass2_proba=p2, pass2_y=y2, t2=pair.t2,
                min_support=min(constraints.min_band_support_pass1,
                                constraints.min_band_support_pass2),
            )
            self.config.constraints = constraints
        return pair

    # ------------------------------------------------------------------
    def _make_bands(self, p1, y1, p2, y2) -> BandConfidence:
        """
        Build band confidences from training OOF at the current thresholds.

        Requires ``thresholds_`` to be set, so it runs after the solve. The
        support floor is the stricter of the two per-pass minimums.
        """
        c = self.config.constraints
        return BandConfidence.from_oof(
            pass1_proba=p1, pass1_y=y1, t1=self.thresholds_.t1,
            pass2_proba=p2, pass2_y=y2, t2=self.thresholds_.t2,
            min_support=min(c.min_band_support_pass1, c.min_band_support_pass2),
        )

    # ------------------------------------------------------------------
    def describe(self) -> None:
        """Print the frozen operating point."""
        self._require_fitted()
        r = self.fit_report_
        print("=" * 78)
        print(f"  RF ROUTER — {r.mode}")
        print("=" * 78)
        print(f"  train rows          : {r.n_train:,}")
        print(f"  training base rate  : {r.training_base_rate:.4f}")
        print(f"  split fingerprint   : {r.split_fingerprint}")
        print(f"  t1 (route < t1)     : {self.thresholds_.t1:.6f}")
        print(f"  t2 (flag  > t2)     : {self.thresholds_.t2:.6f}")
        print(f"  pass1 band conf     : {self.bands_.pass1_confidence:.4f} "
              f"(n={self.bands_.pass1_support:,})")
        print(f"  pass2 band conf     : {self.bands_.pass2_confidence:.4f} "
              f"(n={self.bands_.pass2_support:,})")
        if r.remainder:
            print(f"  remainder rows      : {r.remainder.get('n_rows'):,} "
                  f"({r.remainder.get('fraction', 0):.1%})")
        if r.nested_check:
            print(f"  nested check        : {r.nested_check.get('verdict')}")
        print("=" * 78)
