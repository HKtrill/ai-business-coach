"""
lr.lr_stage_calibrated
======================
Production LR stage — Optuna tuning → calibration → CV threshold optimisation.

Extends BaseLRStage; only defines what is genuinely different:
  - calibrated_model, best_params, calibration_metrics, optimal_threshold
  - fit()  — Optuna search + CalibratedClassifierCV + CV threshold sweep
  - predict_proba() override — routes through calibrated_model, not base model
  - predict() override — uses optimal_threshold instead of 0.5
  - evaluate() override — richer metrics + 2x2 plot + caches X/y for save()
  - save() — full calibrated artifact via artifacts.save_artifact
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import fbeta_score, confusion_matrix
from typing import Dict, Optional, Tuple

from .base_stage  import BaseLRStage
from .tuning      import run_optuna_tuning
from .calibration import fit_calibration
from .evaluation  import compute_metrics, plot_evaluation
from .artifacts   import save_artifact

warnings.filterwarnings("ignore")


class CalibratedLRStage(BaseLRStage):
    """
    Production LR stage: Optuna tuning + probability calibration + CV threshold.

    predict_proba() routes through calibrated_model.
    predict() uses optimal_threshold (CV-optimised for F2), not 0.5.
    save() writes the full calibrated artifact consumed by Meta-EBM loader.
    """

    def __init__(
        self,
        tune_hyperparameters: bool = True,
        calibration_method: str = "auto",
        optimize_threshold: bool = True,
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)

        # Config
        self.tune_hyperparameters = tune_hyperparameters
        self.calibration_method   = calibration_method   # mutated to winner in fit()
        self.optimize_threshold   = optimize_threshold
        self.cv_folds             = cv_folds
        self.n_trials             = n_trials

        # Fitted state unique to this stage
        self.calibrated_model    = None
        self.best_params: Dict   = {}
        self.calibration_metrics: Dict = {}
        self.optimal_threshold   = 0.5

        # Cached splits for save() — populated in fit() / evaluate()
        self._X_train_full = None
        self._y_train_full = None
        self._X_test_full  = None
        self._y_test_full  = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CalibratedLRStage":
        print("\n" + "=" * 80)
        print("🔧 CALIBRATED LR STAGE: Optuna Tuning (balanced) + Calibration")
        print("=" * 80)

        self._X_train_full = X_train.copy()
        self._y_train_full = y_train.copy()
        self.feature_names = list(X_train.columns)

        X_scaled = self._scale_fit_transform(X_train)

        # 1. Hyperparameter tuning
        if self.tune_hyperparameters:
            print("\n" + "=" * 60)
            print("🧠 HYPERPARAMETER TUNING — Optuna (ROC-AUC, balanced)")
            print("=" * 60)
            params, _, _ = run_optuna_tuning(
                X_scaled, y_train, self.n_trials, self.random_state
            )
            self.best_params = params
            lr_kwargs = dict(
                C=params["C"], penalty=params["penalty"], solver="saga",
                max_iter=5000, random_state=self.random_state,
                class_weight="balanced",
            )
            if params.get("penalty") == "elasticnet":
                lr_kwargs["l1_ratio"] = params["l1_ratio"]
        else:
            lr_kwargs = dict(
                C=1.0, penalty="l2", solver="saga", max_iter=5000,
                random_state=self.random_state, class_weight="balanced",
            )
            self.best_params = {"C": 1.0, "penalty": "l2", "class_weight": "balanced"}
            print("\n✅ Skipping tuning — using default parameters")

        self.model = LogisticRegression(**lr_kwargs)
        self.model.fit(X_scaled, y_train)

        # 2. Probability calibration
        print("\n" + "=" * 60)
        print("🔧 PROBABILITY CALIBRATION")
        print("=" * 60)
        self.calibrated_model, self.calibration_method, self.calibration_metrics = (
            fit_calibration(
                self.model, X_scaled, y_train,
                self.calibration_method, self.cv_folds,
            )
        )

        # 3. CV threshold optimisation (F2)
        if self.optimize_threshold:
            print("\n🎯 Optimizing threshold via cross-validation...")
            self._optimize_threshold_cv(X_scaled, y_train)

        self.fitted = True
        return self

    def _optimize_threshold_cv(
        self, X_scaled: np.ndarray, y: pd.Series
    ) -> None:
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        y_proba_oof = cross_val_predict(
            self.calibrated_model, X_scaled, y,
            cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]

        best_f2, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.05, 0.50, 0.01):
            y_pred = (y_proba_oof >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            f2 = fbeta_score(y, y_pred, beta=2)
            if f2 > best_f2:
                best_f2, best_thresh = f2, thresh

        self.optimal_threshold = best_thresh
        print(f"   Optimal threshold (CV): {best_thresh:.4f}")
        print(f"   CV F2-score:            {best_f2:.4f}")

    # ── Predict overrides ─────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities — routes through calibrated_model."""
        self._check_fitted()
        return self.calibrated_model.predict_proba(
            self.scaler.transform(X)
        )[:, 1]

    def predict(
        self, X: pd.DataFrame, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Hard predictions using optimal_threshold (override base default of 0.5)."""
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X) >= t).astype(int)

    # ── Evaluate override ─────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot: bool = True,
    ) -> Dict:
        """
        Evaluate with richer metrics (F2, Brier, ECE) + 2x2 plot.
        Also caches X_test / y_test so save() can call predict_proba().
        """
        self._check_fitted()

        self._X_test_full = X_test.copy()
        self._y_test_full = y_test.copy()

        print("\n" + "=" * 80)
        print(f"📈 EVALUATING CALIBRATED LR (threshold={self.optimal_threshold:.4f})")
        print("=" * 80)

        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.metrics = compute_metrics(
            y_test, y_pred, y_proba,
            self.optimal_threshold, self.calibration_method,
        )

        print("\n📊 Test Performance:")
        for k, v in self.metrics.items():
            if k not in ("threshold", "calibration"):
                print(f"   {k.upper()}: {v:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"   FN: {cm[1, 0]}, TP: {cm[1, 1]}")

        if plot:
            plot_evaluation(
                y_test, y_proba, y_pred,
                self.metrics, self.optimal_threshold, self.calibration_method,
            )

        return self.metrics

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, output_dir: str = "./models/lr") -> dict:
        """Persist the full calibrated artifact via artifacts.save_artifact."""
        self._check_fitted()
        if self._X_train_full is None or self._X_test_full is None:
            raise ValueError(
                "No cached data — call fit() then evaluate() before save()."
            )
        return save_artifact(self, output_dir)


# ── Public entry point ────────────────────────────────────────────────────────

def train_lr_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr",
) -> Tuple["CalibratedLRStage", Dict]:
    """
    Train the production calibrated LR stage on GLOBAL_SPLIT data.

    No internal train/test splitting — receives the global splits directly.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    save            : whether to persist the calibrated artifact
    output_dir      : where to write the joblib file

    Returns
    -------
    stage         : fitted CalibratedLRStage
    artifact_path : path to saved calibrated LR artifact, or None if save=False
    """
    print("\n" + "=" * 80)
    print("🚀 TRAINING LR STAGE (FULL GLOBAL_SPLIT, NO INTERNAL SPLIT)")
    print("=" * 80)
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")

    stage = CalibratedLRStage(
        tune_hyperparameters=True,
        calibration_method="auto",
        optimize_threshold=True,
        cv_folds=10,
    )
    stage.fit(X_train, y_train)
    metrics = stage.evaluate(X_test, y_test, plot=True)

    print("\n📊 Feature Importance (Top 10):")
    print(stage.get_feature_importance().head(10).to_string(index=False))

    save_info = None
    artifact_path = None

    if save:
        save_info = stage.save(output_dir)
        artifact_path = save_info["path"]

    print("\n🎯 LR Stage complete")
    return stage, artifact_path


# Backward-compatibility alias (notebooks using old name still work)
train_calibrated_lr_stage = train_lr_stage