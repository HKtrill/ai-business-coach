"""
glass_pipeline.meta_ebm
========================
Stage 4 — Meta-EBM weighted confidence arbiter.

Ensemble: LR (Stage 1) + EBM (Stage 3) + GLASS-BRW (Stage 2).

Primary entry point (glass_cascade Cell 21):
    from meta_ebm.meta_stage import train_meta_stage
    meta_artifact, meta_path = train_meta_stage(
        GLOBAL_SPLIT,
        lr_path=lr_calibrated_path,
        glass_path=glass_path,
        ebm_path=ebm_path,
    )
"""

from .meta_stage   import train_meta_stage
from .arbiter      import meta_arbiter
from .loader       import load_stage_outputs
from .calibration  import compute_calibration
from .weighting    import compute_hybrid_weights
from .evaluation import analyze_disagreements, compute_metrics, evaluate_with_abstention
from .tuning       import tune_arbiter_threshold
from .artifacts    import save_meta_ebm
from .tracer       import run_cascade_trace

__all__ = [
    # primary entry point
    "train_meta_stage",
    # arbiter
    "meta_arbiter",
    # pipeline components
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