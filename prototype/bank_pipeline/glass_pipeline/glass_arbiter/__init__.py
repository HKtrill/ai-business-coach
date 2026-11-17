"""
glass_pipeline.glass_arbiter
============================
Stage 4 — the GLASS Arbiter, a weighted-confidence arbiter over
GLASS LR (Stage 1) + GLASS Router (Stage 2) + GLASS EBM (Stage 3).

    from glass_pipeline.glass_arbiter import train_glass_arbiter
    meta_artifact, meta_path = train_glass_arbiter(
        GLOBAL_SPLIT,
        lr_path=lr_calibrated_path,
        glass_path=glass_path,
        ebm_path=ebm_path,
        target_recall=0.70,
    )

``load_stage_outputs`` returns the verified, arm-agnostic ``Stage4Inputs``
(OOF train streams + refit test streams + split / index / fold provenance),
reusable by a later learned meta-model.

Before PR 39 (decision D5) this package was ``glass_pipeline.meta_ebm``
("Meta-EBM"); that import path still resolves here, and ``train_meta_stage``
and ``save_meta_ebm`` remain aliases. ``artifacts`` holds the current and the
historical artifact identifiers.
"""

from .meta_stage   import train_glass_arbiter, train_meta_stage, find_recall_threshold
from .arbiter      import meta_arbiter
from shared.stage4 import Stage4Inputs, StageStream, Stage4InputError, build_stage4_inputs
from .loader       import load_stage_outputs
from .calibration  import compute_calibration
from .weighting    import compute_hybrid_weights
from .evaluation   import analyze_disagreements, compute_metrics, evaluate_with_abstention
from .tuning       import tune_arbiter_threshold
from .artifacts    import (ARBITER, ARTIFACT_SCHEMA, HISTORICAL_ARBITER,
                           HISTORICAL_ARTIFACT_SCHEMA, READABLE_SCHEMAS, save_glass_arbiter,
                           save_meta_ebm)
from .tracer       import run_cascade_trace

__all__ = [
    "train_glass_arbiter",
    "find_recall_threshold",
    "meta_arbiter",
    "Stage4Inputs",
    "StageStream",
    "Stage4InputError",
    "build_stage4_inputs",
    "load_stage_outputs",
    "compute_calibration",
    "compute_hybrid_weights",
    "analyze_disagreements",
    "compute_metrics",
    "evaluate_with_abstention",
    "tune_arbiter_threshold",
    "save_glass_arbiter",
    "run_cascade_trace",
    "ARBITER",
    "ARTIFACT_SCHEMA",
    "HISTORICAL_ARBITER",
    "HISTORICAL_ARTIFACT_SCHEMA",
    "READABLE_SCHEMAS",
    # PR 36-38 aliases (D5), kept through PR 39
    "train_meta_stage",
    "save_meta_ebm",
]
