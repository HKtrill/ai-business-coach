"""
Black-Box Stage 1 — Calibrated MLP
==================================
Mirrors the GLASS CalibratedLRStage protocol (lr/lr_stage_calibrated.py):

  0. StandardScaler fitted on the full training split (as GLASS).
  1. Optuna hyperparameter tuning, objective = mean CV ROC-AUC (training only).
  2. Final MLP fitted on the full training split with the best parameters.
  3. Probability calibration via GLASS lr.calibration.fit_calibration.
  4. CV threshold optimisation: out-of-fold calibrated probabilities
     (cross_val_predict), F2 sweep over 0.05–0.49 — same grid and rules as GLASS.
  5. Evaluation via GLASS lr.evaluation.compute_metrics (+ PR-AUC).

Imbalance handling: positive oversampling (tuned pos_ratio) inside each MLP fit.
Early stopping: inner stratified split of each fit's rows, monitored on ROC-AUC.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from lr.calibration import fit_calibration
from lr.evaluation import compute_metrics

STAGE1_FEATURES: Tuple[str, ...] = (
    "cellular_crisis",
    "euribor3m_local_rate",
    "dow_month_encoded",
)


# ---------------------------------------------------------------------------
# sklearn-compatible MLP (so GLASS calibration / cross_val_predict can clone it)
# ---------------------------------------------------------------------------

class Stage1MLPClassifier(ClassifierMixin, BaseEstimator):
    """Small MLP with positive oversampling and inner-split early stopping."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (16,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        batch_size: int = 256,
        pos_ratio: float = 0.5,
        max_epochs: int = 200,
        patience: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state,
        )
        X_fit, y_fit = _oversample_positives(X_fit, y_fit, self.pos_ratio, rng)

        net = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            shuffle=True,
            random_state=self.random_state,
        )

        best_score, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
        for epoch in range(1, self.max_epochs + 1):
            net.partial_fit(X_fit, y_fit, classes=self.classes_)
            val_auc = roc_auc_score(y_val, net.predict_proba(X_val)[:, 1])
            history.append({"epoch": epoch, "train_loss": float(net.loss_), "val_roc_auc": float(val_auc)})
            if val_auc > best_score + 1e-4:
                best_score, best_epoch, wait = val_auc, epoch, 0
                best_state = ([w.copy() for w in net.coefs_], [b.copy() for b in net.intercepts_])
            else:
                wait += 1
                if wait >= self.patience:
                    break

        net.coefs_, net.intercepts_ = best_state
        self.net_ = net
        self.best_epoch_ = best_epoch
        self.history_ = pd.DataFrame(history)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "net_")
        return self.net_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def _oversample_positives(X, y, pos_ratio, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = int(round(pos_ratio * len(neg))) - len(pos)
    if extra <= 0:
        return X, y
    idx = np.concatenate([np.arange(len(y)), rng.choice(pos, size=extra, replace=True)])
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Optuna tuning — ROC-AUC objective (training data only)
# ---------------------------------------------------------------------------

def _sample_params(trial) -> Dict:
    return {
        "n_layers":           trial.suggest_int("n_layers", 1, 2),
        "width":              trial.suggest_categorical("width", [4, 8, 16, 32]),
        "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 3e-4, 1e-2, log=True),
        "batch_size":         trial.suggest_categorical("batch_size", [128, 256, 512]),
        "pos_ratio":          trial.suggest_float("pos_ratio", 0.15, 1.0),
    }


def _params_to_kwargs(params: Dict, **fixed) -> Dict:
    width = int(params["width"])
    hidden = (width,) if int(params["n_layers"]) == 1 else (width, max(width // 2, 2))
    return dict(
        hidden_layer_sizes=hidden,
        activation=params["activation"],
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        batch_size=int(params["batch_size"]),
        pos_ratio=float(params["pos_ratio"]),
        **fixed,
    )


def tune_stage1_mlp_auc(
    X_scaled: np.ndarray,
    y: pd.Series,
    cv,
    n_trials: int,
    fixed: Dict,
    random_state: int,
) -> Tuple[Dict, float, pd.DataFrame]:
    """Maximise mean CV ROC-AUC. Returns best params, best score, trials table."""
    y_arr = np.asarray(y).astype(int)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    def objective(trial):
        kwargs = _params_to_kwargs(_sample_params(trial), **fixed)
        scores = []
        for k, (tr, va) in enumerate(cv.split(X_scaled, y_arr)):
            clf = Stage1MLPClassifier(**kwargs).fit(X_scaled[tr], y_arr[tr])
            scores.append(roc_auc_score(y_arr[va], clf.predict_proba(X_scaled[va])[:, 1]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    return study.best_params, float(study.best_value), trials


# ---------------------------------------------------------------------------
# Stage — same sequence as GLASS CalibratedLRStage
# ---------------------------------------------------------------------------

class CalibratedStage1MLP:
    """Optuna (ROC-AUC) → final fit → calibration → CV F2 threshold."""

    def __init__(
        self,
        calibration_method: str = "auto",
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
        max_epochs: int = 200,
        patience: int = 15,
        n_jobs: int = -1,
    ) -> None:
        self.calibration_method = calibration_method   # mutated to winner in fit()
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_jobs = n_jobs

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
        self.fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CalibratedStage1MLP":
        X = self._select(X_train)
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        fixed = dict(max_epochs=self.max_epochs, patience=self.patience, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # 1. Hyperparameter tuning (ROC-AUC)
        print("🧠 Tuning MLP — Optuna (ROC-AUC)")
        self.best_params, self.best_cv_roc_auc, self.tuning_trials = tune_stage1_mlp_auc(
            X_scaled, y_train, cv, self.n_trials, fixed, self.random_state,
        )
        print(f"   best CV ROC-AUC: {self.best_cv_roc_auc:.4f}")

        # 2. Final model on full training split
        self.model = Stage1MLPClassifier(**_params_to_kwargs(self.best_params, **fixed))
        self.model.fit(X_scaled, y_train)

        # 3. Probability calibration (GLASS)
        print("🔧 Calibrating")
        self.calibrated_model, self.calibration_method, self.calibration_metrics = fit_calibration(
            self.model, X_scaled, y_train, self.calibration_method, self.cv_folds,
        )

        # 4. CV threshold optimisation (F2)
        print("🎯 Optimizing threshold via cross-validation")
        self._optimize_threshold_cv(X_scaled, y_train, X_train.index)

        self.fitted = True
        return self

    def _optimize_threshold_cv(self, X_scaled: np.ndarray, y: pd.Series, index: pd.Index) -> None:
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        y_proba_oof = cross_val_predict(
            self.calibrated_model, X_scaled, y,
            cv=cv, method="predict_proba", n_jobs=self.n_jobs,
        )[:, 1]
        self.proba_train_oof = pd.Series(y_proba_oof, index=index, name="stage1_proba_oof")

        best_f2, best_thresh, rows = 0.0, 0.5, []
        for thresh in np.arange(0.05, 0.50, 0.01):
            y_pred = (y_proba_oof >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            f2 = fbeta_score(y, y_pred, beta=2)
            rows.append({"threshold": float(thresh), "f2": float(f2)})
            if f2 > best_f2:
                best_f2, best_thresh = f2, thresh

        self.optimal_threshold = float(best_thresh)
        self.cv_f2 = float(best_f2)
        self.threshold_sweep = pd.DataFrame(rows)
        print(f"   Optimal threshold (CV): {self.optimal_threshold:.4f}")
        print(f"   CV F2-score:            {self.cv_f2:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Calibrated P(y = 1), indexed like X."""
        self._check_fitted()
        proba = self.calibrated_model.predict_proba(self.scaler.transform(self._select(X)))[:, 1]
        return pd.Series(proba, index=X.index, name="stage1_proba")

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        t = self.optimal_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype("int8").rename("stage1_pred")

    # ── Metrics ───────────────────────────────────────────────────────────
    def metrics_table(self, y_true: pd.Series, proba: pd.Series, label: str) -> pd.DataFrame:
        """GLASS compute_metrics (+ PR-AUC, predicted positive rate) at 0.5 and the CV threshold."""
        p = np.asarray(proba, dtype=float)
        rows = {}
        for t in (0.5, self.optimal_threshold):
            pred = (p >= t).astype(int)
            m = compute_metrics(y_true, pred, p, t, self.calibration_method)
            m = {k: v for k, v in m.items() if k not in ("threshold", "calibration")}
            m["pr_auc"] = average_precision_score(y_true, p)
            m["pred_pos_rate"] = float(pred.mean())
            rows[f"{label} @ {t:.2f}"] = m
        return pd.DataFrame(rows).T

    def describe(self) -> Dict:
        self._check_fitted()
        sizes = [len(STAGE1_FEATURES), *self.model.hidden_layer_sizes, 1]
        net = self.model.net_
        return {
            "layers": " → ".join(map(str, sizes)),
            "trainable_params": int(sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_)),
            **self.best_params,
            "best_epoch": self.model.best_epoch_,
            "best_cv_roc_auc": round(self.best_cv_roc_auc, 4),
            "calibration_method": self.calibration_method,
            "optimal_threshold": self.optimal_threshold,
            "cv_f2": round(self.cv_f2, 4),
        }

    # ── Internal ──────────────────────────────────────────────────────────
    @staticmethod
    def _select(X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns)
        if set(cols) != set(STAGE1_FEATURES) or len(cols) != len(STAGE1_FEATURES):
            raise ValueError(f"Stage 1 expects exactly {list(STAGE1_FEATURES)}, got {cols}")
        return X[list(STAGE1_FEATURES)]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")