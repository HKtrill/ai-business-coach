"""
blackbox_pipeline.models.mlp.stage
===================================
Stage 1 orchestration — the calibrated MLP stage.

Mirrors the GLASS ``CalibratedLRStage`` protocol step for step:

  0. StandardScaler fitted on the full training split.
  1. Optuna tuning, objective = mean CV ROC-AUC (training only).
  2. Final MLP fitted on the full training split with the best parameters.
  3. Probability calibration via GLASS ``lr.calibration.fit_calibration``.
  4. Decision threshold from OUT-OF-FOLD calibrated probabilities, F2 sweep
     over the same grid GLASS uses.
  5. Evaluation via GLASS ``lr.evaluation.compute_metrics`` (+ PR-AUC).

This module composes; it does not compute. Every numerical step lives in
``estimator``, ``tuning``, ``calibration``, ``thresholds`` or ``metrics``.

Train/test protocol
-------------------
``fit`` takes the training split and nothing else. The scaler, the
hyperparameters, the calibrator and the threshold are all derived from it.
There is no method here that accepts test data.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .calibration import fit_stage1_calibration
from .config import Stage1MLPConfig
from .estimator import Stage1MLPClassifier
from .features import STAGE1_FEATURES, select_features
from .evaluation.metrics import metrics_table as _metrics_table
from .thresholds import oof_probabilities, optimize_threshold_cv
from .tuning import params_to_kwargs, tune_stage1_mlp_auc

__all__ = ["CalibratedStage1MLP"]


class CalibratedStage1MLP:
    """
    Optuna (ROC-AUC) → final fit → calibration → CV F2 threshold.

    Constructor arguments are kept as loose keywords, matching the original
    class, so existing notebook cells run unchanged. Passing ``config=`` instead
    gives the validated dataclass; the two are mutually exclusive.
    """

    def __init__(
        self,
        calibration_method: str = "auto",
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
        max_epochs: int = 200,
        patience: int = 15,
        n_jobs: int = -1,
        *,
        config: Optional[Stage1MLPConfig] = None,
        **config_overrides,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = Stage1MLPConfig(
                calibration_method=calibration_method,
                cv_folds=cv_folds,
                n_trials=n_trials,
                random_state=random_state,
                max_epochs=max_epochs,
                patience=patience,
                n_jobs=n_jobs,
                **config_overrides,
            )

        # Mirrored onto the instance for backward compatibility: existing cells
        # read stage1.cv_folds, stage1.calibration_method and so on directly.
        self.calibration_method = self.config.calibration_method
        self.cv_folds = self.config.cv_folds
        self.n_trials = self.config.n_trials
        self.random_state = self.config.random_state
        self.max_epochs = self.config.max_epochs
        self.patience = self.config.patience
        self.n_jobs = self.config.n_jobs

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Stage1MLPClassifier] = None
        self.calibrated_model = None
        self.best_params: Dict = {}
        self.best_cv_roc_auc: Optional[float] = None
        self.tuning_trials: Optional[pd.DataFrame] = None
        self.calibration_metrics: Dict = {}
        self.optimal_threshold: float = 0.5
        self.cv_f2: Optional[float] = None
        self.threshold_sweep: Optional[pd.DataFrame] = None
        self.proba_train_oof: Optional[pd.Series] = None
        self.oof_provenance_: Optional[str] = None
        self.feature_names_ = list(STAGE1_FEATURES)
        self.fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CalibratedStage1MLP":
        cfg = self.config
        X = select_features(X_train)

        y_train = self._check_labels(X, y_train)

        # 0. Scaler — training split only.
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        cv = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )

        # 1. Hyperparameter tuning.
        if cfg.verbose:
            print("🧠 Tuning MLP — Optuna (ROC-AUC)")
        self.best_params, self.best_cv_roc_auc, self.tuning_trials = (
            tune_stage1_mlp_auc(
                X_scaled, y_train, cv, cfg.n_trials,
                cfg.estimator_fixed_kwargs, cfg.random_state,
                verbose=cfg.verbose,
            )
        )
        if cfg.verbose:
            print(f"   best CV ROC-AUC: {self.best_cv_roc_auc:.4f}")

        # 2. Final model on the full training split.
        self.model = Stage1MLPClassifier(
            **params_to_kwargs(self.best_params, **cfg.estimator_fixed_kwargs)
        )
        self.model.fit(X_scaled, y_train)

        # 3. Calibration (GLASS).
        if cfg.verbose:
            print("🔧 Calibrating")
        self.calibrated_model, self.calibration_method, self.calibration_metrics = (
            fit_stage1_calibration(
                self.model, X_scaled, y_train,
                cfg.calibration_method, cfg.cv_folds,
            )
        )

        # 4. Threshold from out-of-fold calibrated probabilities.
        if cfg.verbose:
            print("🎯 Optimizing threshold via cross-validation")
        self.proba_train_oof, self.oof_provenance_ = oof_probabilities(
            self.calibrated_model, X_scaled, y_train, X_train.index,
            cv_folds=cfg.cv_folds, random_state=cfg.random_state,
            n_jobs=cfg.n_jobs, strict=cfg.strict_oof,
        )
        self.optimal_threshold, self.cv_f2, self.threshold_sweep = (
            optimize_threshold_cv(
                y_train, self.proba_train_oof,
                beta=cfg.threshold_beta, grid_spec=cfg.threshold_grid,
            )
        )
        if cfg.verbose:
            print(f"   Optimal threshold (CV): {self.optimal_threshold:.4f}")
            print(f"   CV F2-score:            {self.cv_f2:.4f}")
            print(f"   OOF provenance:         {self.oof_provenance_}")

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Calibrated ``P(y = 1)``, indexed like ``X``."""
        self._check_fitted()
        X_sel = select_features(X)
        proba = self.calibrated_model.predict_proba(self.scaler.transform(X_sel))[:, 1]
        return pd.Series(proba, index=X.index, name="stage1_proba")

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        t = self.optimal_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype("int8").rename("stage1_pred")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def metrics_table(self, y_true, proba, label: str) -> pd.DataFrame:
        """Metrics at 0.50 and at the CV threshold. Row labels unchanged."""
        self._check_fitted()
        return _metrics_table(
            y_true, proba, label,
            thresholds=(0.5, self.optimal_threshold),
            calibration_method=self.calibration_method,
        )

    def describe(self) -> Dict:
        self._check_fitted()
        sizes = [len(STAGE1_FEATURES), *self.model.hidden_layer_sizes, 1]
        return {
            "layers": " → ".join(map(str, sizes)),
            "trainable_params": self.model.n_trainable_params_,
            **self.best_params,
            "best_epoch": self.model.best_epoch_,
            "best_cv_roc_auc": round(self.best_cv_roc_auc, 4),
            "calibration_method": self.calibration_method,
            "optimal_threshold": self.optimal_threshold,
            "cv_f2": round(self.cv_f2, 4),
            "oof_provenance": self.oof_provenance_,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _check_labels(X: pd.DataFrame, y) -> pd.Series:
        y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if not X.index.equals(y.index):
            raise ValueError(
                "X.index and y.index differ — reindex y to X before fitting "
                "(y = y.loc[X.index]). Misaligned labels pass every length "
                "check and silently train on the wrong targets."
            )
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        return y

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted else "not fitted"
        auc = f", cv_auc={self.best_cv_roc_auc:.4f}" if self.fitted else ""
        return (
            f"CalibratedStage1MLP({state}, "
            f"calibration='{self.calibration_method}', "
            f"threshold={self.optimal_threshold:.3f}{auc})"
        )
