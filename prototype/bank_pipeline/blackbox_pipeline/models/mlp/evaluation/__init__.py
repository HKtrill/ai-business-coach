
"""
blackbox_pipeline.models.mlp.evaluation

Shared classification metrics.

``binary_metrics`` and ``metrics_table`` are SHARED measurement: the Stage 1 MLP
and the GLASS Stage 1 LR are compared on these numbers, so both arms must
compute them with the same code. A metrics helper that lives inside one arm's
model module invites a second copy in the other, and then the comparison starts
arguing about denominators instead of models.

Notes
-----
Threshold TUNING is not evaluation and does not belong here. The F-beta threshold
sweeps read training out-of-fold probabilities to PRODUCE a
decision rule, which is fitting — they live in
``blackbox_pipeline.models.mlp.thresholds``.
"""
from .metrics import binary_metrics, metrics_table

__all__ = ["binary_metrics", "metrics_table"]