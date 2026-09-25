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

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .constraints import OperatingConstraints

__all__ = ["RFRouterConfig", "DEFAULT_RF_PARAMS"]


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

        if self.n_oof_folds < 2:
            errors.append("n_oof_folds must be >= 2")
        if self.n_inner_folds < 2:
            errors.append("n_inner_folds must be >= 2")
        if self.remainder_min_size < 1:
            errors.append("remainder_min_size must be >= 1")

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
        if not isinstance(params, dict):
            errors.append(f"{label} must be a dict, got {type(params).__name__}")
            return {}
        missing = [k for k in _REQUIRED_PARAM_KEYS if k not in params]
        if missing:
            errors.append(f"{label} missing keys: {missing}")
        return dict(params)

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
