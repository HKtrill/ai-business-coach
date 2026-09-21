"""
blackbox_pipeline.models.mlp.tuning

Optuna hyperparameter search for Stage 1. Training data only.

The objective is mean cross-validated ROC-AUC, matching the GLASS LR stage.
Every fold fits a fresh ``Stage1MLPClassifier`` on the fold's training rows and
scores the held-out rows, so no trial ever sees a validation row during fitting.

Notes
-----
The search space is kept separate from the estimator on purpose:
:func:`params_to_kwargs` is re-used when rebuilding a saved model, and doing so
should not require importing Optuna.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from .estimator import Stage1MLPClassifier

__all__ = ["sample_params", "params_to_kwargs", "tune_stage1_mlp_auc"]


def sample_params(trial: "optuna.Trial") -> Dict:
    """
    Draw one point from the Stage 1 search space.

    Parameters
    ----------
    trial : optuna.Trial
        The trial to suggest from.

    Returns
    -------
    dict
        ``n_layers`` (1–2), ``width`` (4/8/16/32), ``activation``
        (relu/tanh), ``alpha`` (1e-6–1e-1, log), ``learning_rate_init``
        (3e-4–1e-2, log), ``batch_size`` (128/256/512), ``pos_ratio``
        (0.15–1.0).
    """
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "width": trial.suggest_categorical("width", [4, 8, 16, 32]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 3e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "pos_ratio": trial.suggest_float("pos_ratio", 0.15, 1.0),
    }


def params_to_kwargs(params: Dict, **fixed) -> Dict:
    """
    Translate flat Optuna params into ``Stage1MLPClassifier`` kwargs.

    Parameters
    ----------
    params : dict
        Output of :func:`sample_params`, or ``study.best_params``.
    **fixed
        Kwargs held constant across trials, merged in unchanged — see
        ``Stage1MLPConfig.estimator_fixed_kwargs``.

    Returns
    -------
    dict
        Ready to splat into ``Stage1MLPClassifier(**kwargs)``.

    Notes
    -----
    ``n_layers`` and ``width`` become ``hidden_layer_sizes``: a two-layer network
    halves the width at the second layer, floored at 2. Pure function of its
    inputs, so it is safe to call when rebuilding a saved model.
    """
    width = int(params["width"])
    n_layers = int(params["n_layers"])
    hidden = (width,) if n_layers == 1 else (width, max(width // 2, 2))
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
    y,
    cv,
    n_trials: int,
    fixed: Dict,
    random_state: int,
    verbose: bool = True,
) -> Tuple[Dict, float, pd.DataFrame]:
    """
    Maximise mean CV ROC-AUC over the Stage 1 search space.

    Parameters
    ----------
    X_scaled : numpy.ndarray of shape (n_samples, n_features)
        Scaled TRAINING features. The scaler must have been fitted on the
        training split only.
    y : array-like of shape (n_samples,)
        Binary training labels.
    cv : sklearn splitter
        The same ``StratifiedKFold`` the stage uses elsewhere.
    n_trials : int
        Number of Optuna trials.
    fixed : dict
        Constructor kwargs held constant across trials (epochs, patience,
        val_fraction, seed) — see ``Stage1MLPConfig.estimator_fixed_kwargs``.
    random_state : int
        Seed for the TPE sampler.
    verbose : bool, default True
        Show Optuna's progress bar.

    Returns
    -------
    best_params : dict
        ``study.best_params``, ready for :func:`params_to_kwargs`.
    best_value : float
        Mean CV ROC-AUC of the winning trial.
    trials : pandas.DataFrame
        One row per trial (``number``, ``value``, the parameters, ``state``),
        with the ``params_`` prefix stripped from the column names.

    Notes
    -----
    Pruning: each fold reports the RUNNING mean ROC-AUC so far, not that fold's
    own score, and the ``MedianPruner`` compares those running means across
    trials. It is inactive until 5 trials have completed
    (``n_startup_trials=5``) and, after that, can only fire from the third fold
    onward (``n_warmup_steps=2``). A pruned trial appears in ``trials`` with
    state ``PRUNED`` and a NaN ``value``, so filter on state before averaging
    that column.
    """
    y_arr = np.asarray(y).astype(int)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="stage1_mlp_auc",
    )

    def objective(trial: "optuna.Trial") -> float:
        """Mean ROC-AUC over the CV folds, reporting the running mean for pruning."""
        kwargs = params_to_kwargs(sample_params(trial), **fixed)
        scores: list[float] = []
        for k, (tr, va) in enumerate(cv.split(X_scaled, y_arr)):
            clf = Stage1MLPClassifier(**kwargs).fit(X_scaled[tr], y_arr[tr])
            scores.append(
                float(roc_auc_score(y_arr[va], clf.predict_proba(X_scaled[va])[:, 1]))
            )
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]

    return dict(study.best_params), float(study.best_value), trials
