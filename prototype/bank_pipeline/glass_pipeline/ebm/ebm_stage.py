"""
glass_pipeline.ebm.ebm_stage
============================
Stage 3 orchestrator — EBM (Explainable Boosting Machine).

Invoked from glass_cascade Cell 20:

    block = Stage3Block.from_global_split(GLOBAL_SPLIT, EBMFeaturePipeline,
                                          EBM_STAGE3_CONFIG, split_fingerprint=fp)
    ebm_artifact, ebm_path = train_ebm_stage(block=block, config=EBM_STAGE3_CONFIG)

PR 33 changes (Stage 3 audit)
-----------------------------
Runs on ``shared.stage_runner`` — the same block, runner and artifact as the
black-box XGBoost arm. Relative to the pre-PR-33 stage:

* feature engineers are refitted inside every CV fold (no label leak into
  validation features);
* out-of-fold predictions are produced for the training split — Stage 4
  trains on ``artifact.stage4_frame()``, never on in-sample scores;
* the operating threshold is selected on OOF predictions, not ``y_test``
  (the test-fitted value survives only as ``threshold["test_oracle"]``);
* calibration is gated on OOF ECE and fitted as a fold-nested isotonic map —
  no refit of unweighted EBMs via ``CalibratedClassifierCV``;
* NaN/inf cleaning is learned on train and applied to every scored frame;
* the artifact carries indexes, split + block fingerprints, fold ids, tuning
  metadata, the fitted feature pipeline, cleaner and calibrator.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.stage_runner import Stage3Artifact
from shared.stage_runner import Stage3Block
from shared.stage_runner import Stage3Runner

from .config import EBMStage3Config
from .estimator import EBMFactory
from .feature_engineering import EBM_FEATURES, EBMFeaturePipeline
from .interactions import define_ebm_interactions

__all__ = ["train_ebm_stage", "make_ebm_factory"]


def make_ebm_factory(config: EBMStage3Config, block: Stage3Block) -> EBMFactory:
    return EBMFactory(
        interactions=define_ebm_interactions(block.X_train),
        search_space=config.search_space,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_ebm_stage(
    GLOBAL_SPLIT: Optional[dict] = None,
    *,
    block: Optional[Stage3Block] = None,
    config: Optional[EBMStage3Config] = None,
    params: Optional[dict[str, Any]] = None,
    params_source: Optional[str] = None,
    params_provenance: Optional[dict] = None,
    split_fingerprint: Optional[str] = None,
    save: bool = True,
) -> tuple[Stage3Artifact, Optional[str]]:
    """
    Train Stage 3 EBM under the shared protocol.

    Parameters
    ----------
    GLOBAL_SPLIT
        Raw split (X_train, X_test, y_train, y_test). Used only when ``block``
        is not given, to build one with ``EBMFeaturePipeline``.
    block
        The shared ``Stage3Block`` (preferred — build it once per notebook).
    config
        ``EBMStage3Config``; defaults enforce ``EBM_FEATURES``.
    params / params_source / params_provenance
        Skip Optuna (see ``shared.stage_runner.check_param_reuse``).
    split_fingerprint
        Recorded on a block built here; checked against a supplied block.

    Returns
    -------
    (Stage3Artifact, path or None)
    """
    config = config or EBMStage3Config(expected_features=list(EBM_FEATURES))

    if block is None:
        if GLOBAL_SPLIT is None:
            raise ValueError("pass block= (preferred) or GLOBAL_SPLIT")
        block = Stage3Block.from_global_split(
            GLOBAL_SPLIT, EBMFeaturePipeline, config,
            split_fingerprint=split_fingerprint,
        )
    elif split_fingerprint is not None and block.split_fingerprint != split_fingerprint:
        raise ValueError("block.split_fingerprint differs from split_fingerprint")

    estimator = make_ebm_factory(config, block)
    if config.verbose:
        print(f"  interaction pairs ({len(estimator.interactions)}): "
              f"{estimator.interactions or 'none (additive only)'}")

    runner = Stage3Runner(config, estimator)
    artifact = runner.fit(
        block, params=params, params_source=params_source,
        params_provenance=params_provenance,
        extras={"interactions": [list(p) for p in estimator.interactions]},
    )
    path = runner.save() if save else None
    return artifact, path
