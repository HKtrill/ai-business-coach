"""
blackbox_pipeline.models.rf_router.thresholds
=============================================
Operating-point selection.

One rule governs this whole package: **thresholds are solved from training /
out-of-fold data only**. No function here accepts test probabilities or test
labels, and none should ever be given them.

Routing conventions (strict inequalities, matching ``router.predict``)::

    Pass 1 routes NOT_SUBSCRIBE when  p < t1
    Pass 2 flags  SUBSCRIBE     when  p > t2

``t1`` is the LARGEST threshold whose band still satisfies the NPV floor and
both leakage budgets; ``t2`` is the SMALLEST threshold whose band still
satisfies the precision floor (and the recall floor, when set). The reasoning
for each sits on the corresponding method in ``solver``.

Layout
------
``results``   ``ThresholdResult`` / ``ThresholdPair``
``bandscan``  which bands a score can realise, and their edges
``solver``    ``ThresholdSolver`` — the two scans and the ordering check
"""

from __future__ import annotations

from .results import ThresholdPair, ThresholdResult
from .solver import ThresholdSolver

__all__ = ["ThresholdResult", "ThresholdPair", "ThresholdSolver"]

# Back-compat: these were module-private helpers in the old thresholds.py.
# Safe to delete once nothing imports them by their old names.
from .bandscan import (  # noqa: F401
    lower_band_sizes as _lower_band_sizes,
    lower_edge as _lower_edge,
    prepare_scores as _prepare,
    upper_band_sizes as _upper_band_sizes,
    upper_edge as _upper_edge,
)