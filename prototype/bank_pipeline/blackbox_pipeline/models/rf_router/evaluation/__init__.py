"""Shared Stage 2 evaluation utilities, used by both the GLASS and RF arms."""

from .paired import (
    BOOT_METRICS,
    CONSTRAINT_FIELDS,
    NON_OUTCOME_FIELDS,
    constraint_slippage,
    paired_bootstrap,
    slippage_table,
)
from .router_metrics import compare_routers, evaluate_router, metrics_to_frame

__all__ = [
    "evaluate_router",
    "metrics_to_frame",
    "compare_routers",
    "BOOT_METRICS",
    "CONSTRAINT_FIELDS",
    "NON_OUTCOME_FIELDS",
    "paired_bootstrap",
    "constraint_slippage",
    "slippage_table",
]
