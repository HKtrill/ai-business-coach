"""
blackbox_pipeline.models.rf_router.router
==========================================
Orchestration and the Stage 2 output contract.

Two routers live here:

``SingleRFRouter``  — the ablation. One forest, one score, two thresholds.
``RFRouter``        — the GLASS counterpart. Pass 1 forest routes
                      NOT_SUBSCRIBE; the remainder, defined from OOF Pass 1
                      decisions, trains the Pass 2 forest; the rest abstains.

Both expose exactly the contract ``GLASSBRWPipeline`` exposes, so
``evaluation.router_metrics.evaluate_router`` consumes either arm unchanged:

    predict(X)       -> (preds, conf, decisions)
                        preds     in {0, 1, -1}          int
                        conf      in [0, 1]              float
                        decisions in {"pass1", "pass2", "uncertain"}  object
    predict_proba(X) -> (n, 2) array [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]

Train/test protocol
-------------------
``fit`` receives the training split and nothing else. Every threshold and every
band precision is solved from out-of-fold predictions on that split. The test
split is touched exactly once, by the evaluation module, after the operating
point is frozen. There is no API on these classes that accepts test data.

The one residual dependence
---------------------------
In ``RFRouter``, each Pass 1 probability is honestly out-of-fold, but the cut
point ``t1`` that turns those probabilities into a remainder was solved using
every training label. So the remainder membership of row *i* depends, weakly and
through one scalar, on row *i*'s own label. This is the same order of dependence
GLASS carries — its rules are selected against the labels of the very set it is
scored on — so the arms are matched rather than one being handicapped.

``config.validate_nested = True`` removes even that: ``oof.nested_cascade_oof``
recomputes the whole cascade with per-fold thresholds, so nothing derived from a
scored row reaches its own prediction. Run it once; if the operating point and
headline metrics agree with the cheap path, report the cheap path and cite the
check. If they diverge, report the nested numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .bands import BandConfidence
from .config import RFRouterConfig
from .forest import ForestTrainer
from .oof import OOFScorer, nested_cascade_oof
from .thresholds import ThresholdPair, ThresholdSolver
from .evaluation.router_metrics import (  # noqa: F401
    compare_routers,
    evaluate_router,
    metrics_to_frame,
)

__all__ = [
    "compare_routers",
    "evaluate_router",
    "metrics_to_frame",
    "RFRouter",
    "SingleRFRouter",
    "SUBSCRIBE",
    "NOT_SUBSCRIBE",
    "ABSTAIN",
    "split_fingerprint",
    "assert_disjoint_split",
    "unpack_global_split",
]

# Mirrors glass_brw.core.rule — duplicated rather than imported so the
# black-box package carries no dependency on the GLASS package.
SUBSCRIBE = 1
NOT_SUBSCRIBE = 0
ABSTAIN = -1

_DECISION_DTYPE = object


# ======================================================================
# Global-split guards
# ======================================================================

def split_fingerprint(X_train: pd.DataFrame, X_test: pd.DataFrame) -> str:
    """
    Stable hash of the train/test row indices.

    Recorded at fit time and re-checked at evaluation time so that a router
    fitted under one split can never be scored against another. A silently
    regenerated split is the single most damaging form of contamination
    available here: it would put training rows into the test set, and every
    metric downstream would look excellent and mean nothing.
    """
    h = hashlib.sha256()
    for name, idx in (("train", X_train.index), ("test", X_test.index)):
        h.update(name.encode())
        h.update(np.asarray(idx).astype(str).tobytes())
        h.update(str(len(idx)).encode())
    return h.hexdigest()[:16]


def assert_disjoint_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Hard stop if any row index appears in both splits."""
    overlap = X_train.index.intersection(X_test.index)
    if len(overlap) > 0:
        raise AssertionError(
            f"Train/test indices overlap in {len(overlap)} rows "
            f"(first few: {list(overlap[:5])}). The global split has been "
            "regenerated or the frames were rebuilt from different sources."
        )


def unpack_global_split(
    global_split: dict,
    engineered: Optional[dict] = None,
) -> dict:
    """
    Read the shared split, verify it, and return the frames Stage 2 uses.

    Does NOT create a split, re-run the preprocessor, or re-engineer features.
    The RF arm must consume the same ``GLOBAL_SPLIT`` and the same
    ``engineer_features`` output the GLASS arm consumes; an RF-specific source
    of information would make the comparison meaningless even if no test data
    were involved.

    Parameters
    ----------
    global_split
        ``GLOBAL_SPLIT`` from ``GlobalSplitManager.create_split`` — expects keys
        ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    engineered
        Optional dict carrying the already-binned 29-column frames, e.g.
        ``BRW_DATA`` with ``X_eng_train`` / ``X_eng_test``. When omitted, the
        raw split frames are returned and the caller is responsible for feeding
        the router the engineered ones.
    """
    required = ("X_train", "X_test", "y_train", "y_test")
    missing = [k for k in required if k not in global_split]
    if missing:
        raise KeyError(f"GLOBAL_SPLIT missing keys: {missing}")

    X_train = global_split["X_train"]
    X_test = global_split["X_test"]
    y_train = global_split["y_train"]
    y_test = global_split["y_test"]

    assert_disjoint_split(X_train, X_test)
    fingerprint = split_fingerprint(X_train, X_test)

    out = {
        "y_train": y_train,
        "y_test": y_test,
        "split_fingerprint": fingerprint,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    if engineered is not None:
        for src, dst in (("X_eng_train", "X_train"), ("X_eng_test", "X_test")):
            if src not in engineered:
                raise KeyError(f"engineered dict missing key '{src}'")
        Xtr, Xte = engineered["X_eng_train"], engineered["X_eng_test"]

        # The engineered frames must describe the same rows, in the same order,
        # as the split they came from. A reset_index upstream would pass a
        # length check and silently pair features with the wrong labels.
        if len(Xtr) != len(X_train) or len(Xte) != len(X_test):
            raise ValueError(
                "Engineered frames do not match the split row counts: "
                f"train {len(Xtr)} vs {len(X_train)}, test {len(Xte)} vs {len(X_test)}"
            )
        if not Xtr.index.equals(X_train.index) or not Xte.index.equals(X_test.index):
            raise ValueError(
                "Engineered frames carry a different index from GLOBAL_SPLIT. "
                "Feature rows and label rows must line up positionally."
            )
        assert_disjoint_split(Xtr, Xte)
        out["X_train"], out["X_test"] = Xtr, Xte
    else:
        out["X_train"], out["X_test"] = X_train, X_test

    return out


# ======================================================================
# Shared base
# ======================================================================

@dataclass
class _FitReport:
    """Everything the fit learned, for printing, saving and auditing."""

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
    """Shared contract, validation and probability transform."""

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
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")

        y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y
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
        fit and predict silently permutes features. Reindexing here makes that
        impossible and raises on any genuinely missing column.
        """
        if self.feature_names_ is None:
            raise ValueError("Call fit() first")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing fitted features: {missing}")
        return X[self.feature_names_]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Call fit() first")

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_outputs(n: int):
        preds = np.full(n, ABSTAIN, dtype=int)
        decisions = np.array(["uncertain"] * n, dtype=_DECISION_DTYPE)
        return preds, decisions

    def predict_proba(
        self, X: pd.DataFrame, base_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Probabilistic output, using the SAME transform as
        ``GLASSBRWPipeline.predict_proba``:

            pass1     -> P(1) = base_rate * (1 - conf)
            pass2     -> P(1) = base_rate + (1 - base_rate) * conf
            abstain   -> the flat training base rate

        This is a heuristic mapping of band precision onto a probability, not a
        calibration. It is reproduced verbatim so the two arms' ``predict_proba``
        outputs are comparable objects; adding Platt or isotonic scaling to the
        RF arm alone would make them incomparable.
        """
        self._require_fitted()
        preds, conf, decisions = self.predict(X)
        n = len(preds)

        if base_rate is None:
            base_rate = self.training_base_rate
        base_rate = float(base_rate)

        probas = np.full((n, 2), [1.0 - base_rate, base_rate], dtype=float)

        p1_mask = decisions == "pass1"
        probas[p1_mask, 1] = base_rate * (1.0 - conf[p1_mask])
        probas[p1_mask, 0] = 1.0 - probas[p1_mask, 1]

        p2_mask = decisions == "pass2"
        probas[p2_mask, 1] = base_rate + (1.0 - base_rate) * conf[p2_mask]
        probas[p2_mask, 0] = 1.0 - probas[p2_mask, 1]

        return probas

    # ------------------------------------------------------------------
    def resolve_at(self, constraints, apply: bool = False) -> ThresholdPair:
        # Re-solve t1/t2 from the STORED OOF arrays. No refit.
        self._require_fitted()
        solver = ThresholdSolver(constraints, verbose=False)
        p1, y1, p2, y2, npos = self._oof_for_solving()

        r1 = solver.solve_pass1(p1, y1)
        r1.raise_if_infeasible(context="re-solve Pass 1")
        r2 = solver.solve_pass2(p2, y2, n_positives_global=npos)
        r2.raise_if_infeasible(context="re-solve Pass 2")

        pair = ThresholdPair(t1=float(r1.threshold), t2=float(r2.threshold),
                             pass1=r1, pass2=r2)
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


# ======================================================================
# Ablation: one forest, two thresholds
# ======================================================================

class SingleRFRouter(_BaseRouter):
    """
    Single-RF two-threshold router — the ablation.

    One forest produces one score; two thresholds cut it into three regions.
    This is NOT the final architecture: it exists to show what the RF does
    naturally, before the operating point is pushed toward GLASS. Collapsing the
    two-pass router into this because one model can produce all three decisions
    would discard the structural comparison the experiment is for.
    """

    def __init__(self, config: RFRouterConfig):
        if config.mode != "single_rf":
            raise ValueError(
                f"SingleRFRouter requires mode='single_rf', got {config.mode!r}"
            )
        super().__init__(config)
        self.model_: Optional[Any] = None
        self.oof_proba_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        split_fingerprint_value: Optional[str] = None,
    ) -> "SingleRFRouter":
        cfg = self.config
        X, y = self._check_fit_inputs(X_train, y_train)
        self.feature_names_ = list(X.columns)
        self.training_base_rate = float(np.asarray(y).mean())
        self._y_train_ = np.asarray(y).astype(int)
        self.split_fingerprint_ = split_fingerprint_value

        trainer = ForestTrainer(cfg.rf_params, cfg.random_state)

        if cfg.verbose:
            print("\n── single-RF ablation ──────────────────────────────────")
            print("  out-of-fold scoring")
        scorer = OOFScorer(
            trainer, cfg.n_oof_folds, cfg.random_state,
            cfg.stratify_oof, verbose=cfg.verbose,
        )
        oof = scorer.score(X, y, label="single-RF OOF")
        p_oof = oof.scored_proba()
        y_arr = np.asarray(y)

        if cfg.verbose:
            print("  solving thresholds (train OOF only)")
        t1_res = self.solver.solve_pass1(p_oof, y_arr)
        t1_res.raise_if_infeasible()
        t2_res = self.solver.solve_pass2(
            p_oof, y_arr, n_positives_global=int(y_arr.sum())
        )
        t2_res.raise_if_infeasible()

        # Both thresholds cut the SAME score here, so overlap is possible and
        # the ordering check is meaningful.
        self.thresholds_ = self.solver.make_pair(t1_res, t2_res)

        self.bands_ = BandConfidence.from_oof(
            pass1_proba=p_oof, pass1_y=y_arr, t1=self.thresholds_.t1,
            pass2_proba=p_oof, pass2_y=y_arr, t2=self.thresholds_.t2,
            min_support=min(
                cfg.constraints.min_band_support_pass1,
                cfg.constraints.min_band_support_pass2,
            ),
        )

        if cfg.verbose:
            print("  refitting on the full training split")
        self.model_ = trainer.fit(X, y)
        self.oof_proba_ = p_oof

        self.fit_report_ = _FitReport(
            mode="single_rf (ablation)",
            n_train=len(X),
            training_base_rate=self.training_base_rate,
            split_fingerprint=self.split_fingerprint_,
            thresholds=self.thresholds_.to_dict(),
            bands=self.bands_.to_dict(),
            oof_summary={
                "n_folds": oof.n_folds,
                "n_scored": oof.n_scored,
                "random_state": oof.random_state,
                "stratified": oof.stratified,
            },
        )
        self.is_fitted = True
        if cfg.verbose:
            self.describe()
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame):
        self._require_fitted()
        Xa = self._align_predict_frame(X)
        n = len(Xa)
        preds, decisions = self._empty_outputs(n)

        p = ForestTrainer.positive_proba(self.model_, Xa)
        t1, t2 = self.thresholds_.t1, self.thresholds_.t2

        routed = p < t1
        preds[routed] = NOT_SUBSCRIBE
        decisions[routed] = "pass1"

        flagged = (~routed) & (p > t2)
        preds[flagged] = SUBSCRIBE
        decisions[flagged] = "pass2"

        conf = self.bands_.assign(decisions)
        self.last_scores_ = {"p1": p, "p2": p}
        return preds, conf, decisions

    # ------------------------------------------------------------------
    def _oof_for_solving(self):
        y = np.asarray(self._y_train_).astype(int)
        return self.oof_proba_, y, self.oof_proba_, y, int(y.sum())

    def predict_train_oof(self):
        # Honest train-side routing: every decision from a model that did not
        # train on that row. Stage 4 stacking features must come from here.
        self._require_fitted()
        p = self.oof_proba_
        preds, decisions = self._empty_outputs(len(p))
        t1, t2 = self.thresholds_.t1, self.thresholds_.t2

        routed = p < t1
        preds[routed] = NOT_SUBSCRIBE
        decisions[routed] = "pass1"

        flagged = (~routed) & (p > t2)
        preds[flagged] = SUBSCRIBE
        decisions[flagged] = "pass2"

        return preds, self.bands_.assign(decisions), decisions


# ======================================================================
# Final: two-pass cascade
# ======================================================================

class RFRouter(_BaseRouter):
    """
    Two-pass RF router — the structural counterpart of GLASS Stage 2.

    Fit protocol
    ------------
    1. Pass 1 OOF over the full training split.
    2. Solve ``t1`` from those OOF probabilities.
    3. Remainder := rows whose OOF Pass 1 probability is ``>= t1``.
       This is the leakage-safe construction: a row's remainder membership is
       decided by a model that never trained on it. Using in-sample Pass 1
       predictions here would let a row's own contribution to the Pass 1 fit
       decide whether it joins the Pass 2 training population, and a
       near-separating forest would hand Pass 2 a population selected by
       memorised labels.
    4. Pass 2 OOF WITHIN the remainder. Each fold's Pass 2 model sees remainder
       rows outside that fold only.
    5. Solve ``t2`` from the remainder's OOF probabilities.
    6. Measure both band precisions on OOF.
    7. Refit: Pass 1 on the full split, Pass 2 on the full remainder.

    Predict protocol
    ----------------
    Pass 1 claims first; Pass 2 sees only what Pass 1 left. This mirrors the
    ``mask & (decisions == "uncertain")`` cascade in ``GLASSBRWPipeline.predict``
    — in GLASS the remainder is likewise a runtime artefact, never a
    training-time subset.
    """

    def __init__(self, config: RFRouterConfig):
        if config.mode != "two_pass":
            raise ValueError(
                f"RFRouter requires mode='two_pass', got {config.mode!r}"
            )
        super().__init__(config)
        self.pass1_model_: Optional[Any] = None
        self.pass2_model_: Optional[Any] = None
        self.remainder_mask_: Optional[np.ndarray] = None
        # Retained for auditing: the arrays the operating point was solved from.
        self.oof_pass1_proba_: Optional[np.ndarray] = None
        self.oof_pass2_proba_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        split_fingerprint_value: Optional[str] = None,
    ) -> "RFRouter":
        cfg = self.config
        X, y = self._check_fit_inputs(X_train, y_train)
        y_arr = np.asarray(y).astype(int)

        self.feature_names_ = list(X.columns)
        self.training_base_rate = float(y_arr.mean())
        self._y_train_ = y_arr
        self.split_fingerprint_ = split_fingerprint_value

        p1_trainer = ForestTrainer(cfg.rf_params, cfg.random_state)
        p2_trainer = ForestTrainer(cfg.effective_pass2_params, cfg.random_state)

        # ---- 1. Pass 1 OOF ------------------------------------------------
        if cfg.verbose:
            print("\n── two-pass RF router ──────────────────────────────────")
            print("  [1/6] Pass 1 out-of-fold scoring")
        scorer1 = OOFScorer(
            p1_trainer, cfg.n_oof_folds, cfg.random_state,
            cfg.stratify_oof, verbose=cfg.verbose,
        )
        oof1 = scorer1.score(X, y, label="pass1 OOF")
        p1_oof = oof1.scored_proba()

        # ---- 2. t1 --------------------------------------------------------
        if cfg.verbose:
            print("  [2/6] solving t1 (train OOF only)")
        t1_res = self.solver.solve_pass1(p1_oof, y_arr)
        t1_res.raise_if_infeasible(context="Pass 1")
        t1 = t1_res.threshold

        # ---- 3. remainder from OOF Pass 1 decisions -----------------------
        remainder = p1_oof >= t1
        n_rem = int(remainder.sum())
        if cfg.verbose:
            print(f"  [3/6] remainder: {n_rem:,} rows "
                  f"({n_rem / len(X):.1%}) survive Pass 1")

        if n_rem < cfg.remainder_min_size:
            raise ValueError(
                f"Pass 1 leaves only {n_rem} training rows "
                f"(remainder_min_size={cfg.remainder_min_size}). t1 is routing "
                "nearly everything — loosen the Pass 1 constraints or check "
                "that the leakage budget is not vacuously satisfied."
            )
        rem_pos = int(y_arr[remainder].sum())
        if rem_pos == 0 or rem_pos == n_rem:
            raise ValueError(
                f"Remainder is single-class ({rem_pos} positives of {n_rem}); "
                "Pass 2 cannot be fitted."
            )

        # ---- 4. Pass 2 OOF within the remainder ---------------------------
        if cfg.verbose:
            print("  [4/6] Pass 2 out-of-fold scoring within the remainder")
        scorer2 = OOFScorer(
            p2_trainer, cfg.n_oof_folds, cfg.random_state,
            cfg.stratify_oof, verbose=cfg.verbose,
        )
        oof2 = scorer2.score(X, y, subset_mask=remainder, label="pass2 OOF")
        p2_oof = oof2.scored_proba()
        y2 = oof2.scored_labels(y)

        # ---- 5. t2 --------------------------------------------------------
        if cfg.verbose:
            print("  [5/6] solving t2 (remainder OOF only)")
        t2_res = self.solver.solve_pass2(
            p2_oof, y2, n_positives_global=int(y_arr.sum())
        )
        t2_res.raise_if_infeasible(context="Pass 2")

        # No ordering check: t1 and t2 cut DIFFERENT scores over DIFFERENT
        # populations, so they cannot produce overlapping bands. Comparing them
        # numerically would be meaningless.
        self.thresholds_ = ThresholdPair(
            t1=float(t1), t2=float(t2_res.threshold),
            pass1=t1_res, pass2=t2_res,
        )

        # ---- 6. band confidences + refit ----------------------------------
        self.bands_ = BandConfidence.from_oof(
            pass1_proba=p1_oof, pass1_y=y_arr, t1=self.thresholds_.t1,
            pass2_proba=p2_oof, pass2_y=y2, t2=self.thresholds_.t2,
            min_support=min(
                cfg.constraints.min_band_support_pass1,
                cfg.constraints.min_band_support_pass2,
            ),
        )

        if cfg.verbose:
            print("  [6/6] refitting both forests")
        rem_pos_idx = np.flatnonzero(remainder)
        self.pass1_model_ = p1_trainer.fit(X, y)
        self.pass2_model_ = p2_trainer.fit(
            X.iloc[rem_pos_idx], pd.Series(y_arr[rem_pos_idx])
        )
        self.remainder_mask_ = remainder
        self.oof_pass1_proba_ = p1_oof
        self.oof_pass2_proba_ = oof2.proba  # NaN outside the remainder

        self.fit_report_ = _FitReport(
            mode="two_pass (GLASS counterpart)",
            n_train=len(X),
            training_base_rate=self.training_base_rate,
            split_fingerprint=self.split_fingerprint_,
            thresholds=self.thresholds_.to_dict(),
            bands=self.bands_.to_dict(),
            oof_summary={
                "pass1": {
                    "n_folds": oof1.n_folds, "n_scored": oof1.n_scored,
                    "random_state": oof1.random_state,
                },
                "pass2": {
                    "n_folds": oof2.n_folds, "n_scored": oof2.n_scored,
                    "random_state": oof2.random_state,
                },
            },
            remainder={
                "n_rows": n_rem,
                "fraction": float(n_rem / len(X)),
                "n_positives": rem_pos,
                "base_rate": float(rem_pos / n_rem),
                "source": "OOF pass1 probabilities >= t1",
            },
        )
        self.is_fitted = True

        # ---- optional nested diagnostic -----------------------------------
        if cfg.validate_nested:
            if cfg.verbose:
                print("\n  nested cascade OOF check (per-fold thresholds)")
            self.fit_report_.nested_check = self._run_nested_check(
                X, y, p1_trainer, p2_trainer
            )

        if cfg.verbose:
            self.describe()
        return self

    # ------------------------------------------------------------------
    def _run_nested_check(self, X, y, p1_trainer, p2_trainer) -> dict:
        cfg = self.config
        nested = nested_cascade_oof(
            X, y,
            pass1_trainer=p1_trainer,
            pass2_trainer=p2_trainer,
            solver=self.solver,
            n_outer_folds=cfg.n_oof_folds,
            n_inner_folds=cfg.n_inner_folds,
            random_state=cfg.random_state,
            stratify=cfg.stratify_oof,
            remainder_min_size=max(50, cfg.remainder_min_size // cfg.n_oof_folds),
            verbose=cfg.verbose,
        )
        t1_mean = float(np.mean(nested.fold_t1))
        t2_mean = float(np.mean(nested.fold_t2))
        drift_t1 = abs(t1_mean - self.thresholds_.t1)
        drift_t2 = abs(t2_mean - self.thresholds_.t2)
        rem_frac = float(nested.remainder_mask.mean())
        simple_rem = self.fit_report_.remainder["fraction"]

        verdict = (
            "consistent"
            if drift_t1 < 0.02 and drift_t2 < 0.02
            and abs(rem_frac - simple_rem) < 0.05
            else "DIVERGENT — report the nested numbers"
        )
        return {
            "fold_t1": [float(v) for v in nested.fold_t1],
            "fold_t2": [float(v) for v in nested.fold_t2],
            "t1_mean": t1_mean,
            "t2_mean": t2_mean,
            "t1_drift_vs_simple": drift_t1,
            "t2_drift_vs_simple": drift_t2,
            "remainder_fraction_nested": rem_frac,
            "remainder_fraction_simple": simple_rem,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame):
        self._require_fitted()
        Xa = self._align_predict_frame(X)
        n = len(Xa)
        preds, decisions = self._empty_outputs(n)

        t1, t2 = self.thresholds_.t1, self.thresholds_.t2

        p1 = ForestTrainer.positive_proba(self.pass1_model_, Xa)
        routed = p1 < t1
        preds[routed] = NOT_SUBSCRIBE
        decisions[routed] = "pass1"

        p2 = np.full(n, np.nan, dtype=float)
        rem_idx = np.flatnonzero(~routed)
        if rem_idx.size:
            p2_vals = ForestTrainer.positive_proba(
                self.pass2_model_, Xa.iloc[rem_idx]
            )
            p2[rem_idx] = p2_vals
            flagged = rem_idx[p2_vals > t2]
            preds[flagged] = SUBSCRIBE
            decisions[flagged] = "pass2"

        conf = self.bands_.assign(decisions)
        self.last_scores_ = {"p1": p1, "p2": p2}
        return preds, conf, decisions

    # ------------------------------------------------------------------
    def _oof_for_solving(self):
        y = np.asarray(self._y_train_).astype(int)
        rem = self.remainder_mask_
        return (self.oof_pass1_proba_, y,
                self.oof_pass2_proba_[rem], y[rem], int(y.sum()))

    def predict_train_oof(self):
        # Honest train-side routing: every decision from a model that did not
        # train on that row. Stage 4 stacking features must come from here.
        self._require_fitted()
        p1 = self.oof_pass1_proba_
        p2 = self.oof_pass2_proba_
        preds, decisions = self._empty_outputs(len(p1))
        t1, t2 = self.thresholds_.t1, self.thresholds_.t2

        routed = p1 < t1
        preds[routed] = NOT_SUBSCRIBE
        decisions[routed] = "pass1"

        flagged = (~routed) & ~np.isnan(p2) & (p2 > t2)
        preds[flagged] = SUBSCRIBE
        decisions[flagged] = "pass2"

        return preds, self.bands_.assign(decisions), decisions