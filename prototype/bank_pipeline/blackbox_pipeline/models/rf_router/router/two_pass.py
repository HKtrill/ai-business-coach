"""
Two-pass RF router — the structural counterpart of GLASS Stage 2.

Fit
---
1. Pass 1 OOF over the full training split.
2. Solve ``t1`` from those OOF probabilities.
3. Remainder := rows whose OOF Pass 1 probability is ``>= t1``.
4. Pass 2 OOF WITHIN the remainder.
5. Solve ``t2`` from the remainder's OOF probabilities.
6. Measure both band precisions on OOF, then refit Pass 1 on the full split and
   Pass 2 on the full remainder.

Step 3 is the leakage-safe construction: a row's remainder membership is decided
by a model that never trained on it. In-sample Pass 1 predictions would let a
row's own contribution to the Pass 1 fit decide whether it joins the Pass 2
training population, and a near-separating forest would hand Pass 2 a population
selected by memorised labels.

The one residual dependence
---------------------------
Each Pass 1 probability is honestly out-of-fold, but the cut point ``t1`` that
turns those probabilities into a remainder was solved using every training
label. So row *i*'s remainder membership depends, weakly and through one scalar,
on row *i*'s own label. GLASS carries the same order of dependence — its rules
are selected against the labels of the set it is scored on — so the arms are
matched rather than one being handicapped.

``config.validate_nested = True`` removes even that: ``oof.nested_cascade_oof``
recomputes the whole cascade with per-fold thresholds. Run it once; if the
operating point and headline metrics agree with the cheap path, report the cheap
path and cite the check. If they diverge, report the nested numbers.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import RFRouterConfig
from ..forest import ForestTrainer
from ..oof import OOFScorer, nested_cascade_oof
from ..threshhold import ThresholdPair
from .base import _BaseRouter, _FitReport, apply_cuts

__all__ = ["RFRouter"]

# Tolerances for the nested-vs-cheap agreement verdict.
_NESTED_THRESHOLD_TOL = 0.02
_NESTED_REMAINDER_TOL = 0.05


class RFRouter(_BaseRouter):
    """Pass 1 routes NOT_SUBSCRIBE, Pass 2 flags SUBSCRIBE, the rest abstains."""

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
        n_rem, rem_pos = self._check_remainder(remainder, y_arr, len(X))

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
        self.bands_ = self._make_bands(p1_oof, y_arr, p2_oof, y2)

        if cfg.verbose:
            print("  [6/6] refitting both forests")
        rem_idx = np.flatnonzero(remainder)
        self.pass1_model_ = p1_trainer.fit(X, y)
        self.pass2_model_ = p2_trainer.fit(
            X.iloc[rem_idx], pd.Series(y_arr[rem_idx])
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
    def _check_remainder(self, remainder, y_arr, n_train):
        """Guard the population Pass 2 will be fitted on."""
        n_rem = int(remainder.sum())
        if self.config.verbose:
            print(f"  [3/6] remainder: {n_rem:,} rows "
                  f"({n_rem / n_train:.1%}) survive Pass 1")

        if n_rem < self.config.remainder_min_size:
            raise ValueError(
                f"Pass 1 leaves only {n_rem} training rows "
                f"(remainder_min_size={self.config.remainder_min_size}). t1 is "
                "routing nearly everything — loosen the Pass 1 constraints or "
                "check that the leakage budget is not vacuously satisfied."
            )
        rem_pos = int(y_arr[remainder].sum())
        if rem_pos == 0 or rem_pos == n_rem:
            raise ValueError(
                f"Remainder is single-class ({rem_pos} positives of {n_rem}); "
                "Pass 2 cannot be fitted."
            )
        return n_rem, rem_pos

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

        consistent = (
            drift_t1 < _NESTED_THRESHOLD_TOL
            and drift_t2 < _NESTED_THRESHOLD_TOL
            and abs(rem_frac - simple_rem) < _NESTED_REMAINDER_TOL
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
            "verdict": ("consistent" if consistent
                        else "DIVERGENT — report the nested numbers"),
        }

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame):
        self._require_fitted()
        Xa = self._align_predict_frame(X)
        n = len(Xa)
        t1, t2 = self.thresholds_.t1, self.thresholds_.t2

        p1 = ForestTrainer.positive_proba(self.pass1_model_, Xa)

        # Pass 2 only scores what Pass 1 left — in GLASS the remainder is
        # likewise a runtime artefact, never a training-time subset.
        p2 = np.full(n, np.nan, dtype=float)
        rem_idx = np.flatnonzero(~(p1 < t1))
        if rem_idx.size:
            p2[rem_idx] = ForestTrainer.positive_proba(
                self.pass2_model_, Xa.iloc[rem_idx]
            )

        preds, decisions = apply_cuts(p1, p2, t1, t2)
        self.last_scores_ = {"p1": p1, "p2": p2}
        return preds, self.bands_.assign(decisions), decisions

    # ------------------------------------------------------------------
    def predict_train_oof(self):
        self._require_fitted()
        preds, decisions = apply_cuts(
            self.oof_pass1_proba_, self.oof_pass2_proba_,
            self.thresholds_.t1, self.thresholds_.t2,
        )
        return preds, self.bands_.assign(decisions), decisions

    # ------------------------------------------------------------------
    def _oof_for_solving(self):
        y = np.asarray(self._y_train_).astype(int)
        rem = self.remainder_mask_
        return (self.oof_pass1_proba_, y,
                self.oof_pass2_proba_[rem], y[rem], int(y.sum()))