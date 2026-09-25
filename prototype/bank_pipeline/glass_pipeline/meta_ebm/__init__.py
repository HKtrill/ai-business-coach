"""
glass_pipeline.meta_ebm
=======================
Stage 4 — Meta-EBM weighted-confidence arbiter over
GLASS LR (Stage 1) + GLASS Router (Stage 2) + GLASS EBM (Stage 3).

    from glass_pipeline.meta_ebm.meta_stage import train_meta_stage
    meta_artifact, meta_path = train_meta_stage(
        GLOBAL_SPLIT,
        lr_path=lr_calibrated_path,
        glass_path=glass_path,
        ebm_path=ebm_path,
        target_recall=0.70,
    )

``load_stage_outputs`` returns the verified, arm-agnostic ``Stage4Inputs``
(OOF train streams + refit test streams + split / index / fold provenance),
reusable by a later learned meta-model.
"""

from .meta_stage   import train_meta_stage, find_recall_threshold
from .arbiter      import meta_arbiter
from shared.stage4 import Stage4Inputs, StageStream, Stage4InputError, build_stage4_inputs
from .loader       import load_stage_outputs
from .calibration  import compute_calibration
from .weighting    import compute_hybrid_weights
from .evaluation   import analyze_disagreements, compute_metrics, evaluate_with_abstention
from .tuning       import tune_arbiter_threshold
from .artifacts    import save_meta_ebm
from .tracer       import run_cascade_trace

__all__ = [
    "train_meta_stage",
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
    "save_meta_ebm",
    "run_cascade_trace",
]
