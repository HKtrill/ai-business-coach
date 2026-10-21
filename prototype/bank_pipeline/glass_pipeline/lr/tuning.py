"""
lr.tuning
=========
Optuna Bayesian hyperparameter search for LR.
ROC-AUC objective, class_weight='balanced' fixed.

PR 33: fold count and pruner settings are parameters, and the defaults now
match the MLP arm (``blackbox_pipeline.models.mlp.tuning``): 10 folds,
MedianPruner(n_startup_trials=5, n_warmup_steps=2). With the same folds, the
same pruner and the same trial count, ``study.best_value`` is comparable
across arms. Change one arm and you must change the other.

Pre-PR-33 behaviour was 5 folds, MedianPruner(10, 1).

PR 36 (audit F3/F4): pass ``folds`` (``shared.stage1.crossfit`` designs) and
each fold is scored with its target encodings and scaler refit on that fold's
training rows; ``X_scaled`` is then ignored. The penalty is passed through
``lr.penalty.penalty_kwargs`` (same model, no scikit-learn 1.8+ deprecation).
"""
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .penalty import penalty_kwargs

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_tuning(
    X_scaled: np.ndarray,
    y_train,
    n_trials: int = 100,
    random_state: int = 42,
    cv_folds: int = 10,
    pruner_startup_trials: int = 5,
    pruner_warmup_steps: int = 2,
    folds=None,
) -> tuple:
    """
    Bayesian search over C, penalty, l1_ratio.
    Returns (best_params, study, runtime_s).

    ``folds``: optional list of ``shared.stage1.crossfit.FoldDesign`` with
    per-fold preprocessing. Without it, ``X_scaled`` is split with
    ``StratifiedKFold(cv_folds, shuffle=True, random_state)``.
    """
    if folds is None:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        y_arr = np.asarray(y_train)
        folds = [(X_scaled[tr], y_arr[tr], X_scaled[va], y_arr[va])
                 for tr, va in cv.split(X_scaled, y_arr)]
    else:
        folds = [(d.X_train, d.y_train, d.X_val, d.y_val) for d in folds]

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e3, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        l1_ratio = (trial.suggest_float("l1_ratio", 0.0, 1.0)
                    if penalty == "elasticnet" else None)
        clf = LogisticRegression(
            C=C, solver="saga", max_iter=5000, random_state=random_state,
            class_weight="balanced", **penalty_kwargs(penalty, l1_ratio),
        )
        fold_scores = []
        for fold_idx, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
            try:
                clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
            except Exception:
                return float("-inf")
            fold_scores.append(auc)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        ),
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    runtime_s = time.perf_counter() - t0

    params = study.best_params.copy()
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    print(f"\n✅ Best Parameters (ROC-AUC={study.best_value:.6f}, {runtime_s:.1f}s):")
    print(f"   C            = {params['C']:.6f}")
    print(f"   penalty      = {params['penalty']}")
    if params.get("penalty") == "elasticnet":
        print(f"   l1_ratio     = {params['l1_ratio']:.4f}")
    print(f"   class_weight = balanced  |  solver = saga")
    print(f"   Trials: {n_complete} completed, {n_pruned} pruned")

    return params, study, runtime_s