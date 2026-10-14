"""
blackbox_pipeline.models.rf_router.tuning
==========================================
RF hyperparameter resolution and tuning for Stage 2.

Layout
------
``space``   the search space and the parameter normaliser — one source of truth
``legacy``  the baseline path (former notebook Cells 20-23), behaviour unchanged
"""

from __future__ import annotations

from .legacy import (
    cv_auc_legacy,
    load_rf_params_from_artifact,
    resolve_rf_params,
    tune_rf_train_only,
)
from .space import REQUIRED_PARAM_KEYS, normalise_rf_params, suggest_rf_params

__all__ = [
    "REQUIRED_PARAM_KEYS",
    "normalise_rf_params",
    "suggest_rf_params",
    "cv_auc_legacy",
    "tune_rf_train_only",
    "load_rf_params_from_artifact",
    "resolve_rf_params",
]
