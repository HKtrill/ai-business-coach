"""
blackbox_pipeline.models.mlp.evaluation.metrics

Re-exports ``shared.metrics``.

PR 33: the implementation moved to GLASS. It used to live here and import FROM
GLASS, which meant GLASS could not import it back — the MLP reported
``pr_auc``/``pred_pos_rate`` and metrics at two thresholds while GLASS reported
neither. One copy now, in the library arm, called by both.
"""

from shared.metrics import (  # noqa: F401
    binary_metrics,
    calculate_ece,
    compute_metrics,
    metrics_table,
)

__all__ = ["binary_metrics", "metrics_table", "compute_metrics", "calculate_ece"]
