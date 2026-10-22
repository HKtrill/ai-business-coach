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

import math
import numbers
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from .estimator import Stage1MLPClassifier, _binary_labels

__all__ = [
    "MLP_TUNED_PARAM_NAMES", "MLP_FIXED_PARAM_NAMES", "normalize_mlp_params",
    "sample_params", "params_to_kwargs", "tune_stage1_mlp_auc",
]

#: The tuned hyperparameters, in ``sample_params`` order.
MLP_TUNED_PARAM_NAMES: Tuple[str, ...] = (
    "n_layers", "width", "activation", "alpha", "learning_rate_init", "batch_size", "pos_ratio",
)
#: Constructor kwargs held constant across trials
#: (``Stage1MLPConfig.estimator_fixed_kwargs``); any subset may be passed.
MLP_FIXED_PARAM_NAMES: Tuple[str, ...] = ("max_epochs", "patience", "val_fraction", "random_state")
#: ``MLPClassifier``'s activations (the search space draws relu / tanh).
_ACTIVATIONS = ("identity", "logistic", "tanh", "relu")

# name -> (integer?, lower bound, lower inclusive, upper bound, upper inclusive)
_DOMAIN: Dict[str, tuple] = {
    "n_layers": (True, 1, True, 2, True),          # params_to_kwargs builds 1 or 2 layers
    "width": (True, 1, True, math.inf, False),
    "alpha": (False, 0.0, True, math.inf, False),
    "learning_rate_init": (False, 0.0, False, math.inf, False),
    "batch_size": (True, 1, True, math.inf, False),
    "pos_ratio": (False, 0.0, True, math.inf, False),
    "max_epochs": (True, 1, True, math.inf, False),
    "patience": (True, 1, True, math.inf, False),
    "val_fraction": (False, 0.0, False, 1.0, False),
    "random_state": (True, 0, True, math.inf, False),
}


def _number(name: str, value: Any):
    """``value`` as an int / float in ``_DOMAIN[name]``, or ``ValueError``."""
    integer, low, low_inc, high, high_inc = _DOMAIN[name]
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not math.isfinite(value)):
        raise ValueError(f"MLP {name} must be a finite number, got {value!r}")
    if integer:
        if not float(value).is_integer():
            raise ValueError(f"MLP {name} must be an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    if not ((value >= low) if low_inc else (value > low)) or \
            not ((value <= high) if high_inc else (value < high)):
        raise ValueError(f"MLP {name}={value!r} is outside "
                         f"{'[' if low_inc else '('}{low}, {high}{']' if high_inc else ')'}")
    return value


def normalize_mlp_params(params: Mapping[str, Any]) -> Dict:
    """
    Exactly the tuned keys, each a valid value, in ``sample_params`` order.

    PR 36 (audit F1): :func:`params_to_kwargs` read the seven names it knew
    and dropped every other key (``widht=8`` was recorded and never applied),
    truncated ``width`` / ``n_layers`` / ``batch_size`` with ``int()``
    (``width=32.9`` built 32 units, ``n_layers=True`` one layer) and mapped any
    ``n_layers`` other than 1 to two layers (0 and 3 included). Now a missing
    or unknown key raises ``KeyError``; a non-finite, boolean, non-integral or
    out-of-domain value, or an activation ``MLPClassifier`` does not offer,
    raises ``ValueError``.
    """
    missing = [k for k in MLP_TUNED_PARAM_NAMES if k not in params]
    if missing:
        raise KeyError(f"MLP params missing tuned keys: {missing}")
    unknown = [k for k in params if k not in MLP_TUNED_PARAM_NAMES]
    if unknown:
        raise KeyError(f"Unknown MLP hyperparameters {unknown}; expected exactly "
                       f"{list(MLP_TUNED_PARAM_NAMES)}.")
    activation = params["activation"]
    if not isinstance(activation, str) or activation not in _ACTIVATIONS:
        raise ValueError(f"MLP activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
    return {k: activation if k == "activation" else _number(k, params[k])
            for k in MLP_TUNED_PARAM_NAMES}


def _check_fixed(fixed: Mapping[str, Any]) -> Dict:
    """The fixed estimator kwargs, checked like the tuned ones (F1)."""
    unknown = [k for k in fixed if k not in MLP_FIXED_PARAM_NAMES]
    if unknown:
        raise KeyError(f"Unknown fixed MLP kwargs {unknown}; allowed: {list(MLP_FIXED_PARAM_NAMES)}.")
    return {k: _number(k, v) for k, v in fixed.items()}


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

    ``params`` must hold exactly the tuned keys and ``fixed`` only
    ``MLP_FIXED_PARAM_NAMES``, every value valid (:func:`normalize_mlp_params`,
    PR 36 audit F1); nothing is dropped or coerced, so the estimator runs with
    exactly the values recorded in ``best_params``.
    """
    p = normalize_mlp_params(params)
    fixed = _check_fixed(fixed)
    width = p["width"]
    hidden = (width,) if p["n_layers"] == 1 else (width, max(width // 2, 2))
    return dict(
        hidden_layer_sizes=hidden,
        activation=p["activation"],
        alpha=p["alpha"],
        learning_rate_init=p["learning_rate_init"],
        batch_size=p["batch_size"],
        pos_ratio=p["pos_ratio"],
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
    *,
    folds=None,
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
    folds : list of shared.stage1.crossfit.FoldDesign, optional
        Folds with encoders and scaler refit on each fold's training rows
        (audit F3/F4). The stage always passes them; ``X_scaled`` and ``cv``
        are then not used for splitting.

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

    Labels are checked to be binary 0/1, not truncated (audit F34), and the
    fixed kwargs before any trial is fitted (audit F1).
    """
    y_arr = _binary_labels(y)
    fixed = _check_fixed(fixed)
    if folds is None:
        folds = [(X_scaled[tr], y_arr[tr], X_scaled[va], y_arr[va])
                 for tr, va in cv.split(X_scaled, y_arr)]
    else:
        folds = [(d.X_train, _binary_labels(d.y_train), d.X_val, _binary_labels(d.y_val))
                 for d in folds]
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
        for k, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
            clf = Stage1MLPClassifier(**kwargs).fit(X_tr, y_tr)
            scores.append(float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]

    return dict(study.best_params), float(study.best_value), trials
