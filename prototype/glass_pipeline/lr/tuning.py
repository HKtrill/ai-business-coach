"""
lr.tuning
=========
Optuna Bayesian hyperparameter search for LR.
ROC-AUC objective, class_weight='balanced' fixed.
"""
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_tuning(
    X_scaled: np.ndarray,
    y_train,
    n_trials: int = 100,
    random_state: int = 42,
) -> tuple:
    """
    Bayesian search over C, penalty, l1_ratio.
    Returns (best_params, study, runtime_s).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e3, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        lr_kwargs = dict(
            C=C, penalty=penalty, solver="saga", max_iter=5000,
            random_state=random_state, class_weight="balanced",
        )
        if penalty == "elasticnet":
            lr_kwargs["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        clf = LogisticRegression(**lr_kwargs)
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_scaled, y_train)):
            try:
                clf.fit(X_scaled[tr_idx], y_train.iloc[tr_idx])
                auc = roc_auc_score(
                    y_train.iloc[val_idx],
                    clf.predict_proba(X_scaled[val_idx])[:, 1],
                )
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
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
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