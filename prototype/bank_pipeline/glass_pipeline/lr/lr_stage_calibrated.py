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

PR 33 changes
-------------
* ``proba_train_oof`` is now KEPT. It was already being computed inside the
  threshold sweep and thrown away; what got persisted as ``train_predictions``
  was ``predict_proba(X_train)``, which routes through a calibrator fitted on
  all of X_train, so every training row scored itself. Stage 4 would have seen
  a GLASS column that is sharper on train than it can ever be at test, next to
  an honestly-degraded MLP column, and concluded LR was the more reliable arm.
* ``tuning_trials``, ``best_cv_roc_auc``, ``cv_f2`` and ``threshold_sweep`` are
  kept rather than printed and dropped.
* ``calibration_requested`` preserves the request, which used to be destroyed
  by writing the winner back over ``self.calibration_method``.
* Tuning now matches the MLP arm exactly: 10 folds, MedianPruner(5, 2)
  (previously 5 folds, MedianPruner(10, 1)). Same folds, same pruner, same
  100-trial budget, so ``best_cv_roc_auc`` is comparable across arms. The
  settings are explicit arguments (``tune_cv_folds``,
  ``pruner_startup_trials``, ``pruner_warmup_steps``) and recorded in the
  artifact config. Pass 5 / 10 / 1 to reproduce pre-PR-33 runs.
* ``to_stage_output()`` emits the shared cascade contract.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from typing import Dict, Optional, Tuple

from .base_stage  import BaseLRStage
from .tuning      import run_optuna_tuning
from .calibration import fit_calibration
from .evaluation  import compute_metrics, plot_evaluation
from .artifacts   import save_artifact

from shared.stage_io import StageOutput, fold_assignment
from shared.thresholds import optimize_threshold_cv
from shared.calibration_guard import assert_refittable
from shared.metrics import metrics_table as _metrics_table

warnings.filterwarnings("ignore")


class CalibratedLRStage(BaseLRStage):
    """
    Production LR stage: Optuna tuning + probability calibration + CV threshold.

    predict_proba() routes through calibrated_model.
    predict() uses optimal_threshold (CV-optimised for F2), not 0.5.
    save() writes the full calibrated artifact consumed by Meta-EBM loader.

    Attributes
    ----------
    proba_train_oof : pandas.Series or None
        Training OUT-OF-FOLD calibrated P(y=1). This — not
        ``predict_proba(X_train)`` — is what Stage 4 must consume.
    train_fold_id : pandas.Series or None
        Which fold each training row was held out in.
    calibration_method : str
        The family actually selected.
    calibration_requested : str
        What was asked for, preserved across the fit.
    """

    #: How ``optimal_threshold`` was chosen. See shared.stage_io.THRESHOLD_SOURCES.
    THRESHOLD_SOURCE = "in_stage_f2_cv"

    def __init__(
        self,
        tune_hyperparameters: bool = True,
        calibration_method: str = "auto",
        optimize_threshold: bool = True,
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
        tune_cv_folds: int = 10,          # matches Stage1MLPConfig.cv_folds
        pruner_startup_trials: int = 5,   # matches MLP MedianPruner(5, 2)
        pruner_warmup_steps: int = 2,
        threshold_beta: float = 2.0,
        threshold_grid: Tuple[float, float, float] = (0.05, 0.50, 0.01),
    ) -> None:
        super().__init__(random_state=random_state)

        # Config
        self.tune_hyperparameters  = tune_hyperparameters
        self.calibration_requested = calibration_method
        self.calibration_method    = calibration_method   # mutated to winner in fit()
        self.optimize_threshold    = optimize_threshold
        self.cv_folds              = cv_folds
        self.n_trials              = n_trials
        self.tune_cv_folds         = tune_cv_folds
        self.pruner_startup_trials = pruner_startup_trials
        self.pruner_warmup_steps   = pruner_warmup_steps
        self.threshold_beta        = threshold_beta
        self.threshold_grid        = threshold_grid

        # Fitted state unique to this stage
        self.calibrated_model    = None
        self.best_params: Dict   = {}
        self.calibration_metrics: Dict = {}
        self.optimal_threshold   = 0.5

        # Kept for Stage 4 and the write-up (previously computed and dropped)
        self.proba_train_oof: Optional[pd.Series]     = None
        self.train_fold_id:   Optional[pd.Series]     = None
        self.threshold_sweep: Optional[pd.DataFrame]  = None
        self.tuning_trials:   Optional[pd.DataFrame]  = None
        self.best_cv_roc_auc: Optional[float]         = None
        self.cv_f2:           Optional[float]         = None
        self.oof_provenance_: Optional[str]           = None

        # Cached splits for save() — populated in fit() / evaluate()
        self._X_train_full = None
        self._y_train_full = None
        self._X_test_full  = None
        self._y_test_full  = None

    def config_dict(self) -> Dict:
        """Everything needed to reproduce this fit."""
        return {
            "tune_hyperparameters":  self.tune_hyperparameters,
            "calibration_requested": self.calibration_requested,
            "optimize_threshold":    self.optimize_threshold,
            "cv_folds":              self.cv_folds,
            "tune_cv_folds":         self.tune_cv_folds,
            "n_trials":              self.n_trials,
            "pruner_startup_trials": self.pruner_startup_trials,
            "pruner_warmup_steps":   self.pruner_warmup_steps,
            "threshold_beta":        self.threshold_beta,
            "threshold_grid":        tuple(self.threshold_grid),
            "random_state":          self.random_state,
        }

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CalibratedLRStage":
        print("\n" + "=" * 80)
        print("🔧 CALIBRATED LR STAGE: Optuna Tuning (balanced) + Calibration")
        print("=" * 80)

        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(np.asarray(y_train), index=X_train.index)
        if not X_train.index.equals(y_train.index):
            raise ValueError(
                "X_train.index and y_train.index differ — reindex y to X before "
                "fitting (y = y.loc[X.index]). Misaligned labels pass every "
                "length check and silently train on the wrong targets."
            )

        self._X_train_full = X_train.copy()
        self._y_train_full = y_train.copy()
        self.feature_names = list(X_train.columns)

        X_scaled = self._scale_fit_transform(X_train)

        # 1. Hyperparameter tuning
        if self.tune_hyperparameters:
            print("\n" + "=" * 60)
            print("🧠 HYPERPARAMETER TUNING — Optuna (ROC-AUC, balanced)")
            print("=" * 60)
            params, study, _ = run_optuna_tuning(
                X_scaled, y_train, self.n_trials, self.random_state,
                cv_folds=self.tune_cv_folds,
                pruner_startup_trials=self.pruner_startup_trials,
                pruner_warmup_steps=self.pruner_warmup_steps,
            )
            self.best_params     = params
            self.best_cv_roc_auc = float(study.best_value)
            trials = study.trials_dataframe(
                attrs=("number", "value", "params", "state")
            )
            trials.columns = [c.replace("params_", "") for c in trials.columns]
            self.tuning_trials = trials

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
                self.calibration_requested, self.cv_folds,
            )
        )

        # 3. OOF probabilities + threshold optimisation (F2)
        #    The OOF probabilities are KEPT — they are the Stage 4 train feed.
        print("\n🎯 Computing OOF probabilities and optimizing threshold...")
        self._fit_threshold_cv(X_scaled, y_train, X_train.index)

        self.fitted = True
        return self

    def _fit_threshold_cv(
        self, X_scaled: np.ndarray, y: pd.Series, index: pd.Index
    ) -> None:
        """
        Cross-fit calibrated probabilities over the training split, then sweep F2.

        Notes
        -----
        The probabilities produced here are out-of-fold. The THRESHOLD chosen
        from them is not: it is one scalar selected by argmax over the grid
        using every training label, then applied back to those same rows. That
        distinction is carried forward in ``THRESHOLD_SOURCE`` rather than
        left implicit — see ``shared.stage_io.StageOutput.train_decisions_oof``.
        """
        self.oof_provenance_ = assert_refittable(self.calibrated_model, strict=True)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        proba_oof = cross_val_predict(
            self.calibrated_model, X_scaled, y,
            cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]

        if len(proba_oof) != len(index):
            raise AssertionError(
                f"cross_val_predict returned {len(proba_oof)} rows for "
                f"{len(index)} training rows — fold assignment and index are "
                "misaligned."
            )

        self.proba_train_oof = pd.Series(
            proba_oof, index=index, name="stage1_proba_oof"
        )
        self.train_fold_id = fold_assignment(
            y, cv_folds=self.cv_folds, random_state=self.random_state, index=index
        )

        if not self.optimize_threshold:
            print("   Threshold optimisation disabled — keeping 0.5")
            return

        self.optimal_threshold, self.cv_f2, self.threshold_sweep = (
            optimize_threshold_cv(
                y, self.proba_train_oof,
                beta=self.threshold_beta, grid_spec=self.threshold_grid,
            )
        )
        print(f"   Optimal threshold (CV): {self.optimal_threshold:.4f}")
        print(f"   CV F2-score:            {self.cv_f2:.4f}")
        print(f"   OOF provenance:         {self.oof_provenance_}")

    # ── Predict overrides ─────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calibrated probabilities — routes through calibrated_model.

        Warnings
        --------
        On the TRAINING split this is in-sample: the calibrator was fitted on
        those rows. Use ``proba_train_oof`` for anything downstream.
        """
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
        Evaluate with richer metrics (F2, Brier, ECE, PR-AUC) + 2x2 plot.
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

    def metrics_table(self, y_true, proba, label: str) -> pd.DataFrame:
        """
        Metrics at 0.50 and at the CV threshold, via the shared helper.

        Use with ``stage.proba_train_oof`` for the train row — never with
        ``predict_proba(X_train)``, which is in-sample.
        """
        self._check_fitted()
        return _metrics_table(
            y_true, proba, label,
            thresholds=(0.5, self.optimal_threshold),
            calibration_method=self.calibration_method,
        )

    # ── Stage 4 handoff ───────────────────────────────────────────────────────

    def to_stage_output(self) -> StageOutput:
        """
        Emit the shared cascade contract for Stage 4.

        Raises
        ------
        ValueError
            If ``evaluate()`` has not run — the test probabilities come from it.
        """
        self._check_fitted()
        if self._X_test_full is None:
            raise ValueError(
                "No cached test split — call evaluate() before to_stage_output()."
            )
        if self.proba_train_oof is None:
            raise ValueError("No OOF probabilities — refit with this version.")

        test_proba = pd.Series(
            self.predict_proba(self._X_test_full),
            index=self._X_test_full.index,
            name="stage1_proba",
        )

        return StageOutput(
            stage="stage1",
            arm="glass",
            model="calibrated_lr",
            feature_names=list(self.feature_names),
            train_proba_oof=self.proba_train_oof,
            test_proba=test_proba,
            y_train=self._y_train_full,
            y_test=self._y_test_full,
            train_fold_id=self.train_fold_id,
            threshold=float(self.optimal_threshold),
            threshold_source=self.THRESHOLD_SOURCE,
            calibration_method=self.calibration_method,
            calibration_requested=self.calibration_requested,
            oof_provenance=self.oof_provenance_ or "unknown",
            best_params=dict(self.best_params),
            best_cv_score=self.best_cv_roc_auc,
            cv_f2=self.cv_f2,
            threshold_sweep=self.threshold_sweep,
            calibration_metrics=dict(self.calibration_metrics),
            metrics_test=dict(self.metrics),
            config=self.config_dict(),
        )

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

    artifact_path = None
    if save:
        artifact_path = stage.save(output_dir)["path"]

    print("\n🎯 LR Stage complete")
    return stage, artifact_path


# Backward-compatibility alias (notebooks using old name still work)
train_calibrated_lr_stage = train_lr_stage
