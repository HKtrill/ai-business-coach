"""
blackbox_pipeline.models.rf_router.threshhold
=============================================
Historical import path of ``blackbox_pipeline.models.rf_router.thresholds``
(the package's name before PR 41).

Kept so that historical callers and pickles resolve unchanged: anything that
imports ``rf_router.threshhold.<module>``, or was pickled under that path, gets
the ``rf_router.thresholds`` module of the same name (one object, not a copy),
so no code is duplicated and types compare equal. New code imports
``rf_router.thresholds``.
"""

from blackbox_pipeline.models.rf_router.thresholds import *          # noqa: F401,F403
from blackbox_pipeline.models.rf_router.thresholds import __all__    # noqa: F401
from blackbox_pipeline.models.rf_router.thresholds import (          # noqa: F401
    _lower_band_sizes,
    _lower_edge,
    _prepare,
    _upper_band_sizes,
    _upper_edge,
)
