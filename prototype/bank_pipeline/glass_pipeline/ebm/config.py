"""
glass_pipeline.ebm.config
=========================
Stage 3 EBM configuration.

Every protocol field is inherited from ``shared.stage_runner.Stage3Config`` —
the same class ``XGBStage3Config`` inherits — so the two arms cannot drift.
Only the family name, artifact location and ``EBMSearchSpace`` are set here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from shared.stage_runner import Stage3Config

from .estimator import EBMSearchSpace

__all__ = ["EBMStage3Config"]


@dataclass
class EBMStage3Config(Stage3Config):
    model_family: str = "ebm"
    arm: str = "glass"
    counterpart: str = "blackbox_pipeline.models.xgb"
    column_prefix: str = "ebm_stage3_"
    study_name: str = "ebm_stage3_recall_biased"
    search_space: EBMSearchSpace = field(default_factory=EBMSearchSpace)
    artifact_dir: str = "models/ebm"
    artifact_stem: str = "ebm_stage3"

    _space_cls: ClassVar[Optional[type]] = EBMSearchSpace
