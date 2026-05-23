"""
lr.lr_stage_calibrated
======================
Calibrated LR orchestrator — wires tuning, calibration, evaluation, and artifact modules.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import fbeta_score, confusion_matrix
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore")

from .tuning     import run_optuna_tuning
from .calibration import fit_calibration, calculate_ece
from .evaluation  import compute_metrics, plot_evaluation
from .artifacts   import save_artifact


class CalibratedLRStage:
    """
    LR stage: Optuna balanced tuning → probability calibration → CV threshold optimisation.
    """

    def __init__(
        self,
        tune_hyperparameters: bool = True,
        calibration_method: str = "auto",
        optimize_threshold: bool = True,
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
    ):
        self.tune_hyperparameters = tune_hyperparameters
        self.calibration_method   = calibration_method
        self.optimize_threshold   = optimize_threshold
        self.cv_folds             = cv_folds
        self.n_trials             = n_trials
        self.random_state         = random_state

        self.model            = None
        self.calibrated_model = None
        self.scaler           = None
        self.feature_names    = None
        self.best_params      = {}
        self.optimal_threshold = 0.5
        self.metrics          = {}
        self.calibration_metrics = {}
        self.fitted           = False

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

        print("\n📏 Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 1. Tuning
        if self.tune_hyperparameters:
            print("\n" + "=" * 60)
            print("🧠 HYPERPARAMETER TUNING — Optuna (ROC-AUC, balanced)")
            print("=" * 60)
            params, _, _ = run_optuna_tuning(X_scaled, y_train, self.n_trials, self.random_state)
            self.best_params = params

            lr_kwargs = dict(
                C=params["C"], penalty=params["penalty"], solver="saga",
                max_iter=5000, random_state=self.random_state, class_weight="balanced",
            )
            if params.get("penalty") == "elasticnet":
                lr_kwargs["l1_ratio"] = params["l1_ratio"]
            self.model = LogisticRegression(**lr_kwargs)
            self.model.fit(X_scaled, y_train)
        else:
            self.model = LogisticRegression(
                C=1.0, penalty="l2", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=self.random_state,
            )
            self.model.fit(X_scaled, y_train)
            self.best_params = {"C": 1.0, "penalty": "l2", "class_weight": "balanced"}
            print("\n✅ Model trained with default parameters")

        # 2. Calibration
        print("\n" + "=" * 60)
        print("🔧 PROBABILITY CALIBRATION")
        print("=" * 60)
        self.calibrated_model, self.calibration_method, self.calibration_metrics = fit_calibration(
            self.model, X_scaled, y_train, self.calibration_method, self.cv_folds
        )

        # 3. Threshold
        if self.optimize_threshold:
            print("\n🎯 Optimizing threshold via cross-validation...")
            self._optimize_threshold_cv(X_scaled, y_train)

        self.fitted = True
        return self

    def _optimize_threshold_cv(self, X_scaled: np.ndarray, y: pd.Series):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
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
        print(f"   CV F2-score: {best_f2:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        if not self.fitted: raise ValueError("Model not fitted.")
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba(X) >= t).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted: raise ValueError("Model not fitted.")
        return self.calibrated_model.predict_proba(self.scaler.transform(X))[:, 1]

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, plot: bool = True) -> dict:
        if not self.fitted: raise ValueError("Model not fitted.")

        self._X_test_full = X_test.copy()
        self._y_test_full = y_test.copy()

        print("\n" + "=" * 80)
        print(f"📈 EVALUATING CALIBRATED LR (threshold={self.optimal_threshold:.4f})")
        print("=" * 80)

        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.metrics = compute_metrics(
            y_test, y_pred, y_proba, self.optimal_threshold, self.calibration_method
        )

        print("\n📊 Test Performance:")
        for k, v in self.metrics.items():
            if k not in ("threshold", "calibration"):
                print(f"   {k.upper()}: {v:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")

        if plot:
            plot_evaluation(
                y_test, y_proba, y_pred,
                self.metrics, self.optimal_threshold, self.calibration_method,
            )

        return self.metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.fitted: raise ValueError("Model not fitted.")
        return pd.DataFrame({
            "feature":         self.feature_names,
            "coefficient":     self.model.coef_[0],
            "abs_coefficient": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)

    def save(self, output_dir: str = "./models/lr") -> dict:
        if not self.fitted: raise ValueError("Model not fitted.")
        if self._X_train_full is None or self._X_test_full is None:
            raise ValueError("No cached data. Call fit() and evaluate() first.")
        return save_artifact(self, output_dir)


# ── Public entry point ────────────────────────────────────────────────────────

def train_lr_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr",
) -> Tuple[CalibratedLRStage, Dict]:
    """Train calibrated LR stage on GLOBAL_SPLIT data. No internal splitting."""
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

    if save:
        stage.save(output_dir)

    print("\n🎯 LR Stage complete")
    return stage, metrics


# Backward compatibility alias
train_calibrated_lr_stage = train_lr_stage