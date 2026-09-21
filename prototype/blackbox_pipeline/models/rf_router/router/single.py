"""
Single-RF two-threshold router — the ablation.

One forest produces one score; two thresholds cut it into three regions. This is
NOT the final architecture: it exists to show what the RF does naturally, before
the operating point is pushed toward GLASS. Collapsing the two-pass router into
this because one model can produce all three decisions would discard the
structural comparison the experiment is for.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import RFRouterConfig
from ..forest import ForestTrainer
from ..oof import OOFScorer
from .base import _BaseRouter, _FitReport, apply_cuts

__all__ = ["SingleRFRouter"]


class SingleRFRouter(_BaseRouter):
    """One forest, one score, two thresholds."""

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
        y_arr = np.asarray(y).astype(int)

        self.feature_names_ = list(X.columns)
        self.training_base_rate = float(y_arr.mean())
        self._y_train_ = y_arr
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
        self.bands_ = self._make_bands(p_oof, y_arr, p_oof, y_arr)

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

        p = ForestTrainer.positive_proba(self.model_, Xa)
        preds, decisions = apply_cuts(
            p, p, self.thresholds_.t1, self.thresholds_.t2
        )

        self.last_scores_ = {"p1": p, "p2": p}
        return preds, self.bands_.assign(decisions), decisions

    # ------------------------------------------------------------------
    def predict_train_oof(self):
        self._require_fitted()
        p = self.oof_proba_
        preds, decisions = apply_cuts(
            p, p, self.thresholds_.t1, self.thresholds_.t2
        )
        return preds, self.bands_.assign(decisions), decisions

    # ------------------------------------------------------------------
    def _oof_for_solving(self):
        y = np.asarray(self._y_train_).astype(int)
        return self.oof_proba_, y, self.oof_proba_, y, int(y.sum())