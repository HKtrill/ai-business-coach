"""
blackbox_pipeline.models.rf_router
===================================
Black-box Random Forest router for Stage 2.

Replaces the GLASS symbolic two-pass router (beam search → ILP rule selection →
rule cascade) with forests and thresholds, while preserving the information
access, the global split, the routing role, the operating constraints, the
downstream contract and the train/test discipline.

Two experiments
---------------
``SingleRFRouter``  ablation — one forest, one score, two thresholds.
``RFRouter``        final    — two-pass cascade, the structural counterpart of
                              GLASS Stage 2.

Typical use
-----------
    from blackbox_pipeline.models.rf_router import (
        OperatingConstraints, RFRouterConfig, RFRouter, SingleRFRouter,
        unpack_global_split, save_rf_router,
    )
    from blackbox_pipeline.evaluation.router_metrics import (
        evaluate_router, compare_routers,
    )

    data = unpack_global_split(GLOBAL_SPLIT, engineered=BRW_DATA)
    constraints = OperatingConstraints.from_glass_config(GLASS_CONFIG)

    ablation = SingleRFRouter(
        RFRouterConfig(mode="single_rf", constraints=constraints,
                       rf_params=rf_result.params)
    ).fit(data["X_train"], data["y_train"], data["split_fingerprint"])

    router = RFRouter(
        RFRouterConfig(mode="two_pass", constraints=constraints,
                       rf_params=rf_result.params)
    ).fit(data["X_train"], data["y_train"], data["split_fingerprint"])

    # test split touched here, once, after both operating points are frozen
    m = evaluate_router(*router.predict(data["X_test"]), data["y_test"],
                        label="RF two-pass")
"""

from .bands import BandConfidence
from .config import DEFAULT_RF_PARAMS, RFRouterConfig
from .constraints import OperatingConstraints
from .forest import ForestTrainer
from .oof import OOFResult, OOFScorer, make_folds, nested_cascade_oof
from .router import (
    ABSTAIN,
    NOT_SUBSCRIBE,
    SUBSCRIBE,
    RFRouter,
    SingleRFRouter,
    assert_disjoint_split,
    split_fingerprint,
    unpack_global_split,
)
from .threshhold import ThresholdPair, ThresholdResult, ThresholdSolver
from .artifacts import load_rf_router, save_rf_router

__all__ = [
    "OperatingConstraints",
    "RFRouterConfig",
    "DEFAULT_RF_PARAMS",
    "ForestTrainer",
    "OOFScorer",
    "OOFResult",
    "make_folds",
    "nested_cascade_oof",
    "ThresholdSolver",
    "ThresholdResult",
    "ThresholdPair",
    "BandConfidence",
    "RFRouter",
    "SingleRFRouter",
    "split_fingerprint",
    "assert_disjoint_split",
    "unpack_global_split",
    "save_rf_router",
    "load_rf_router",
    "SUBSCRIBE",
    "NOT_SUBSCRIBE",
    "ABSTAIN",
]
