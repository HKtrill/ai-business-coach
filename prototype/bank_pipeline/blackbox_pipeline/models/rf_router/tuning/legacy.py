"""
blackbox_pipeline.models.rf_router.tuning.legacy
=================================================
The BASELINE hyperparameter path, moved out of the notebook unchanged.

Formerly notebook Cells 20-23:

* ``load_rf_params_from_artifact`` — Cell 21
* ``tune_rf_train_only``           — Cell 22 (ROC-AUC, 5-fold CV, X_train only)
* ``resolve_rf_params``            — Cell 23

Behaviour is preserved exactly so the existing RF Router baseline reproduces:
same search space and suggest order (``space.suggest_rf_params``), same CV
splitter, same estimator construction (``n_jobs=1`` inside folds), same
objective (mean per-fold ROC-AUC), same TPE seed. The only structural change is
that the per-configuration CV loop is now a named function,
``cv_auc_legacy``, so tests can cross-check it against the specialization
harness without running a study.

Nothing here is used by the specialization arms. Those re-tune inside
``tuning.search`` so every arm shares one harness and one budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..artifacts import find_artifact_for_split
from .space import REQUIRED_PARAM_KEYS, normalise_rf_params, suggest_rf_params

__all__ = [
    "cv_auc_legacy",
    "tune_rf_train_only",
    "load_rf_params_from_artifact",
    "resolve_rf_params",
]


def cv_auc_legacy(
    X: pd.DataFrame,
    y: pd.Series,
    params: Mapping,
    *,
    random_state: int,
    n_splits: int = 5,
) -> float:
    """
    Mean per-fold ROC-AUC for one normalised parameter dict — Cell 22's inner
    loop, verbatim.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mw = float(params["minority_weight"])
    scores = []
    for tr, va in cv.split(X, y):
        pipe = Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=params["max_depth"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=params["max_features"],
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            )
        ])
        pipe.fit(
            X.iloc[tr],
            y.iloc[tr],
            clf__sample_weight=y.iloc[tr].map({0: 1.0, 1: mw}).values,
        )
        scores.append(
            roc_auc_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1])
        )
    return float(np.mean(scores))


def tune_rf_train_only(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    random_state: int = 42,
    *,
    show_progress_bar: bool = True,
) -> dict:
    """Same search space as ``glass_router.rf.rf_training._tune_rf``. X_train only."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_rf_params(trial)
        return cv_auc_legacy(X, y, params, random_state=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    print(f"  best CV ROC-AUC: {study.best_value:.6f}")
    return normalise_rf_params(study.best_params)


def load_rf_params_from_artifact(
    dirs: Iterable[str | Path], expected_fp: str
) -> Tuple[dict, str]:
    """
    Params from the newest GLASS RF artifact fitted under this split.

    RF params were tuned with CV on X_train. An artifact from another split
    was tuned on rows that may now be test rows, so a fingerprint mismatch is
    skipped, not used.
    """
    dirs = list(dirs)
    obj, path = find_artifact_for_split(dirs, "rf_result_*.joblib", expected_fp)
    if obj is None:
        raise FileNotFoundError(
            "No RF artifact with split_fingerprint == "
            f"{expected_fp} in:\n"
            + "\n".join(f"  {d}" for d in dirs)
            + "\n\nRun the GLASS notebook's Cell 16 "
              "(it saves the RF artifact with its fingerprint), "
              "or set RF_PARAM_SOURCE = 'tune'."
        )
    if not isinstance(obj.get("params"), dict):
        raise KeyError(f"{path.name} has no 'params' dict")
    return normalise_rf_params(obj["params"]), str(path)


def resolve_rf_params(
    source: str,
    *,
    split_fp: str,
    artifact_dirs: Optional[Iterable[str | Path]] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    n_trials: Optional[int] = None,
    explicit: Optional[Mapping] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[dict, str]:
    """
    Resolve the baseline RF hyperparameters — Cell 23.

    ``source == "artifact"`` loads them from the GLASS RF artifact fitted under
    ``split_fp``; ``"tune"`` runs ``tune_rf_train_only`` on ``X_train``; any
    other value normalises ``explicit``.

    Returns
    -------
    (params, origin)
    """
    if source == "artifact":
        if artifact_dirs is None:
            raise ValueError("RF_PARAM_SOURCE='artifact' needs artifact_dirs")
        params, origin = load_rf_params_from_artifact(artifact_dirs, split_fp)
    elif source == "tune":
        if X_train is None or y_train is None or n_trials is None:
            raise ValueError("RF_PARAM_SOURCE='tune' needs X_train, y_train and n_trials")
        if verbose:
            print(f"Tuning on X_train only — {n_trials} trials, 5-fold CV")
        params = tune_rf_train_only(X_train, y_train, n_trials, random_state)
        origin = f"fresh train-only tune ({n_trials} trials)"
    else:
        if explicit is None:
            raise ValueError(
                f"RF_PARAM_SOURCE={source!r} needs RF_PARAMS_EXPLICIT"
            )
        params = normalise_rf_params(explicit)
        origin = "RF_PARAMS_EXPLICIT"

    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    assert not missing, f"RF_PARAMS missing keys: {missing}"

    if verbose:
        print(f"\nRF params ← {origin}")
        print(pd.Series(params).to_string())
    return params, origin
