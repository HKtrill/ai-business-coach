"""
blackbox_pipeline.models.rf_router.tuning.space
================================================
The RF hyperparameter search space and the parameter normaliser.

Single source of truth for both the baseline tuner (``legacy.py``, the former
notebook Cell 22) and the objective-specialization harness. Every arm of the
specialization ablation samples from exactly this space, which is what makes
their tuning budgets comparable.

``suggest_rf_params`` calls ``trial.suggest_*`` in the SAME ORDER as the former
Cell 22 / ``glass_pipeline.glass_router.rf.rf_training._tune_rf``. Optuna's
samplers draw from one RNG stream, so reordering the calls would change which
configurations a seeded study visits and the baseline would no longer reproduce.
``tests/test_rf_router_refactor.py`` pins the order.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..config import _REQUIRED_PARAM_KEYS, check_rf_param, validate_rf_params

__all__ = ["REQUIRED_PARAM_KEYS", "normalise_rf_params", "suggest_rf_params"]

#: Keys every normalised RF parameter dict carries. Same tuple ``RFRouterConfig``
#: validates against — re-exported, not redefined.
REQUIRED_PARAM_KEYS = _REQUIRED_PARAM_KEYS

#: The names ``suggest_rf_params`` gives Optuna (raw ``best_params``).
RAW_OPTUNA_KEYS = ("max_features_type", "use_max_depth", "n_estimators", "max_depth",
                   "min_samples_leaf", "max_features_fraction", "minority_weight")


def normalise_rf_params(raw: Mapping[str, Any]) -> dict:
    """
    Accept either normalised params or raw Optuna ``best_params``.

    Mirrors ``glass_pipeline.glass_router.rf.rf_training._normalize_params``
    (formerly ``_normalise_rf_params`` in notebook Cell 20), keeping its
    documented defaults (no ``use_max_depth`` -> no depth limit, no
    ``max_features_type`` -> ``'sqrt'``, no ``minority_weight`` -> 1.0).

    PR 36 (audit F1): a normalised dict with an extra key used to come back
    with the key silently dropped (``RF_PARAMS_EXPLICIT`` with ``max_dept``
    ran at the old depth), a raw dict with an unknown key was mapped as if it
    were not there, and ``max_features_type='fraction'`` without its fraction
    (or ``use_max_depth=True`` without ``max_depth``) became ``None``. All of
    these now raise ``ValueError``, and the result passes
    ``validate_rf_params``.
    """
    if all(k in raw for k in REQUIRED_PARAM_KEYS):
        return validate_rf_params(raw, "RF params")

    unknown = [k for k in raw if k not in RAW_OPTUNA_KEYS]
    if unknown:
        raise ValueError(f"RF params: unknown keys {unknown}; expected the normalised keys "
                         f"{list(REQUIRED_PARAM_KEYS)} or the raw Optuna keys {list(RAW_OPTUNA_KEYS)}")
    mf_type = raw.get("max_features_type", "sqrt")
    if mf_type == "fraction" and "max_features_fraction" not in raw:
        raise ValueError("RF params: max_features_type='fraction' needs max_features_fraction")
    if raw.get("use_max_depth") and "max_depth" not in raw:
        raise ValueError("RF params: use_max_depth=True needs max_depth")
    return validate_rf_params({
        "n_estimators": raw["n_estimators"],
        "max_depth": raw.get("max_depth") if raw.get("use_max_depth") else None,
        "min_samples_leaf": raw["min_samples_leaf"],
        "max_features": (
            raw.get("max_features_fraction") if mf_type == "fraction" else mf_type
        ),
        "minority_weight": raw.get("minority_weight", 1.0),
        "class_weight": "balanced",
    }, "RF params")


def suggest_rf_params(trial, fixed: Optional[Mapping[str, Any]] = None) -> dict:
    """
    Sample one normalised RF parameter dict from ``trial``.

    Parameters
    ----------
    trial
        An ``optuna.Trial``.
    fixed
        Normalised keys to hold constant instead of searching. A fixed key's
        ``suggest`` calls are skipped entirely (``max_features`` skips both
        ``max_features_type`` and ``max_features_fraction``; ``max_depth`` skips
        ``use_max_depth`` too). ``class_weight`` is always ``"balanced"``.

    Returns
    -------
    dict
        Keys exactly ``REQUIRED_PARAM_KEYS``.
    """
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(REQUIRED_PARAM_KEYS)
    if unknown:
        raise ValueError(f"fixed contains unknown keys: {sorted(unknown)}")
    if fixed.get("class_weight", "balanced") != "balanced":
        raise ValueError("class_weight is fixed to 'balanced' in this search space")
    # PR 36 (audit F1): fixed values are checked, not coerced
    # (n_estimators=500.5 used to run 500 trees).
    fixed = {k: check_rf_param(k, v, "fixed") for k, v in fixed.items()}

    # Order matters — see module docstring.
    mf_type = None
    if "max_features" not in fixed:
        mf_type = trial.suggest_categorical(
            "max_features_type", ["sqrt", "log2", "fraction"]
        )
    use_depth = None
    if "max_depth" not in fixed:
        use_depth = trial.suggest_categorical("use_max_depth", [True, False])

    if "n_estimators" in fixed:
        n_estimators = int(fixed["n_estimators"])
    else:
        n_estimators = trial.suggest_int("n_estimators", 200, 1200, step=50)

    if "max_depth" in fixed:
        max_depth = fixed["max_depth"]
    else:
        max_depth = trial.suggest_int("max_depth", 2, 12) if use_depth else None

    if "min_samples_leaf" in fixed:
        min_samples_leaf = int(fixed["min_samples_leaf"])
    else:
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50)

    if "max_features" in fixed:
        max_features = fixed["max_features"]
    elif mf_type == "fraction":
        max_features = trial.suggest_float("max_features_fraction", 0.3, 1.0)
    else:
        max_features = mf_type

    if "minority_weight" in fixed:
        minority_weight = float(fixed["minority_weight"])
    else:
        minority_weight = trial.suggest_float("minority_weight", 1.0, 3.0)

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "minority_weight": minority_weight,
        "class_weight": "balanced",
    }
