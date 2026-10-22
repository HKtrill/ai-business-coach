"""
blackbox_pipeline.models.rf_router.config
==========================================
Explicit configuration for the Stage 2 black-box RF router.

Follows the GlassRouterConfig discipline: the notebook is the single source of
truth, every knob is stated, nothing is silently defaulted where the choice
could change an experimental result.

Two experiments share this config:

``mode = "single_rf"``
    Ablation. One forest, two thresholds on one score:
    ``p < t1 -> NOT_SUBSCRIBE | t1 <= p <= t2 -> ABSTAIN | p > t2 -> SUBSCRIBE``.
    Observes the RF's natural routing behaviour before it is pushed toward the
    GLASS operating point.

``mode = "two_pass"``
    Structural counterpart of GLASS Stage 2. Pass 1 forest routes
    NOT_SUBSCRIBE; the remainder (defined from OOF Pass 1 decisions) trains a
    second forest that detects SUBSCRIBE; everything unresolved abstains.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional

import numpy as np

from .constraints import OperatingConstraints

__all__ = ["RFRouterConfig", "DEFAULT_RF_PARAMS", "validate_rf_params", "check_rf_param"]


# Mirrors glass_pipeline.glass_router.rf.rf_training._normalize_params output shape so that
# ``rf_result.params`` can be passed straight through.
DEFAULT_RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "minority_weight": 1.0,
    "class_weight": "balanced",
}

_REQUIRED_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "minority_weight",
    "class_weight",
)

_MAX_FEATURES_NAMES = ("sqrt", "log2")
_CLASS_WEIGHTS = ("balanced", "balanced_subsample", None)


def _integer(value: Any, where: str, low: int) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or not float(value).is_integer()):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < low:
        raise ValueError(f"{where} must be >= {low}, got {value!r}")
    return int(value)


def check_rf_param(key: str, value: Any, label: str = "rf_params") -> Any:
    """
    One RF hyperparameter in the domain ``forest.build_pipe`` applies it in.

    ``n_estimators`` / ``min_samples_leaf``: integers >= 1 (``build_pipe``
    casts with ``int()``, so a fraction would have been truncated);
    ``max_depth``: ``None`` or an integer >= 1; ``max_features``: ``'sqrt'``,
    ``'log2'``, ``None`` or a float fraction in (0, 1] (an int would mean a
    feature COUNT to scikit-learn); ``minority_weight``: a finite number > 0;
    ``class_weight``: ``'balanced'``, ``'balanced_subsample'`` or ``None``.
    Integral values become ``int``; everything else is returned unchanged.
    """
    where = f"{label}.{key}"
    if key in ("n_estimators", "min_samples_leaf"):
        return _integer(value, where, 1)
    if key == "max_depth":
        return None if value is None else _integer(value, where, 1)
    if key == "max_features":
        if value is None or (isinstance(value, str) and value in _MAX_FEATURES_NAMES):
            return value
        if (isinstance(value, (float, np.floating)) and math.isfinite(value)
                and 0.0 < value <= 1.0):
            return value
        raise ValueError(f"{where} must be 'sqrt', 'log2', None or a float fraction in "
                         f"(0, 1], got {value!r}")
    if key == "minority_weight":
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{where} must be a finite number > 0, got {value!r}")
        return value
    if key == "class_weight":
        if not (value is None or (isinstance(value, str) and value in _CLASS_WEIGHTS)):
            raise ValueError(f"{where} must be one of {list(_CLASS_WEIGHTS)}, got {value!r}")
        return value
    raise ValueError(f"{label} unknown key {key!r}")


def validate_rf_params(params: Any, label: str = "rf_params") -> dict:
    """
    Exactly ``_REQUIRED_PARAM_KEYS``, each value valid (``check_rf_param``),
    in that order.

    PR 36 (audit F1): only missing keys were refused. An unknown or misspelled
    key (``max_dept``) was kept in the config, recorded in the saved artifact
    and never applied (``build_pipe`` reads the six names it knows); a
    fractional ``n_estimators`` / ``min_samples_leaf`` was truncated; and
    out-of-domain values surfaced only when a forest was fitted, mid-run.
    Raises ``ValueError`` listing every problem.
    """
    if not isinstance(params, Mapping):
        raise ValueError(f"{label} must be a dict, got {type(params).__name__}")
    missing = [k for k in _REQUIRED_PARAM_KEYS if k not in params]
    unknown = [k for k in params if k not in _REQUIRED_PARAM_KEYS]
    problems = []
    if missing:
        problems.append(f"{label} missing keys: {missing}")
    if unknown:
        problems.append(f"{label} unknown keys: {unknown} (expected exactly "
                        f"{list(_REQUIRED_PARAM_KEYS)}; an unknown key is never applied)")
    out = {}
    for key in _REQUIRED_PARAM_KEYS:
        if key in params:
            try:
                out[key] = check_rf_param(key, params[key], label)
            except ValueError as exc:
                problems.append(str(exc))
    if problems:
        raise ValueError("; ".join(problems))
    return out


@dataclass
class RFRouterConfig:
    """
    Parameters
    ----------
    mode
        ``"single_rf"`` (ablation) or ``"two_pass"`` (GLASS counterpart).
    constraints
        Set-level operating requirements. See ``constraints.py``.
    rf_params
        Hyperparameters for every forest fitted by this router, in the shape
        produced by ``glass_pipeline.glass_router.rf.rf_training._normalize_params``. Reuse
        ``rf_result.params`` from Cell 12 — those were tuned with 5-fold CV on
        ``X_train`` only, so they carry no test information.
    pass2_rf_params
        Optional separate hyperparameters for the Pass 2 forest. ``None`` reuses
        ``rf_params``. Tuning these against the remainder is legitimate only if
        the search is run on the remainder's OOF folds; doing it by hand against
        reported numbers is threshold shopping.
    n_oof_folds
        Folds used to produce out-of-fold probabilities. 10 matches
        ``_evaluate_rf``'s evaluation CV.
    random_state
        Seed for forests and fold assignment. 42 matches the global split and
        the Optuna study.
    stratify_oof
        Stratified folds (default True). Necessary at this base rate; an
        unstratified fold can end up with too few positives to estimate band
        precision.
    remainder_min_size
        Hard floor on the number of training rows surviving Pass 1. Below this
        the Pass 2 forest is not identifiable and ``fit`` raises rather than
        producing a model trained on a handful of rows.
    validate_nested
        When True, ``RFRouter.fit`` additionally runs the fully nested cascade
        OOF diagnostic (``oof.nested_cascade_oof``) and stores the comparison in
        ``fit_report_``. Expensive (``n_oof_folds * (n_inner_folds + 2)`` fits)
        but it is the honest check on the one residual dependence in the simple
        protocol — see ``router.py`` for the discussion.
    n_inner_folds
        Inner folds for the nested diagnostic.
    verbose
        Print progress.
    """

    mode: Optional[str] = None
    constraints: Optional[OperatingConstraints] = None

    rf_params: Optional[dict] = None
    pass2_rf_params: Optional[dict] = None

    n_oof_folds: int = 10
    random_state: int = 42
    stratify_oof: bool = True

    remainder_min_size: int = 500

    validate_nested: bool = False
    n_inner_folds: int = 5

    verbose: bool = True

    # Populated in __post_init__; not user-set.
    _resolved: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.mode not in ("single_rf", "two_pass"):
            errors.append(
                f"mode must be 'single_rf' or 'two_pass', got {self.mode!r}"
            )

        if self.constraints is None:
            errors.append(
                "constraints is required — build it with "
                "OperatingConstraints.from_glass_config(GLASS_CONFIG)"
            )
        elif not isinstance(self.constraints, OperatingConstraints):
            errors.append(
                f"constraints must be an OperatingConstraints, "
                f"got {type(self.constraints).__name__}"
            )

        if self.rf_params is None:
            self.rf_params = dict(DEFAULT_RF_PARAMS)
        self.rf_params = self._check_params(self.rf_params, "rf_params", errors)

        if self.pass2_rf_params is not None:
            self.pass2_rf_params = self._check_params(
                self.pass2_rf_params, "pass2_rf_params", errors
            )
            if self.mode == "single_rf":
                # PR 36 (audit F1): recorded, never applied — the ablation
                # fits one forest with rf_params.
                errors.append(
                    "pass2_rf_params applies to the two_pass cascade only; the "
                    "single_rf ablation fits one forest with rf_params."
                )

        # PR 36 (audit F1): integers, not values that pass a comparison
        # (remainder_min_size=True was a floor of 1, n_oof_folds=5.5 failed
        # only inside the fold splitter).
        for name, low in (("n_oof_folds", 2), ("n_inner_folds", 2),
                          ("remainder_min_size", 1), ("random_state", 0)):
            try:
                _integer(getattr(self, name), name, low)
            except ValueError as exc:
                errors.append(str(exc))

        if self.mode == "single_rf" and self.validate_nested:
            errors.append(
                "validate_nested applies to the two_pass cascade only; "
                "the single_rf ablation has no remainder to nest."
            )

        if errors:
            raise ValueError(
                "RFRouterConfig validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        self._resolved = True

    # ------------------------------------------------------------------
    @staticmethod
    def _check_params(params: Any, label: str, errors: list) -> dict:
        """``validate_rf_params``, its problems collected into ``errors``."""
        try:
            return validate_rf_params(params, label)
        except ValueError as exc:
            errors.append(str(exc))
            return dict(params) if isinstance(params, Mapping) else {}

    # ------------------------------------------------------------------
    @property
    def effective_pass2_params(self) -> dict:
        """Hyperparameters actually used for the Pass 2 forest."""
        return dict(self.pass2_rf_params or self.rf_params)

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            out[f.name] = (
                val.to_dict() if isinstance(val, OperatingConstraints) else val
            )
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "RFRouterConfig":
        d = dict(d)
        if isinstance(d.get("constraints"), dict):
            d["constraints"] = OperatingConstraints.from_dict(d["constraints"])
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RFRouterConfig(mode='{self.mode}', "
            f"folds={self.n_oof_folds}, seed={self.random_state}, "
            f"{self.constraints!r})"
        )
