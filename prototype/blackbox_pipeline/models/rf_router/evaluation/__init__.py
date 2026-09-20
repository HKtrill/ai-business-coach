"""Shared Stage 2 evaluation utilities, used by both the GLASS and RF arms."""

from .router_metrics import compare_routers, evaluate_router, metrics_to_frame

__all__ = ["evaluate_router", "metrics_to_frame", "compare_routers"]