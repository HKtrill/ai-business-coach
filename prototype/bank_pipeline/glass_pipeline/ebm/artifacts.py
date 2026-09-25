"""
glass_pipeline.ebm.artifacts
============================
Since PR 33 the EBM writes the shared ``shared.stage_runner.Stage3Artifact``
through ``Stage3ArtifactStore``: ``models/ebm/ebm_stage3_<ts>.joblib`` + a JSON
sidecar (split / block fingerprints, protocol, metrics) + an
``ebm_stage3_latest.joblib`` copy. Load by fingerprint with
``shared.stage_runner`` — not by modification time.
"""

from shared.stage_runner import (  # noqa: F401
    Stage3Artifact,
    Stage3ArtifactStore,
    find_stage3_artifact,
)


def save_ebm_artifacts(payload, base_path="models/ebm"):
    """Save a ``Stage3Artifact`` (the pre-PR-33 dict payload is not accepted)."""
    if not isinstance(payload, Stage3Artifact):
        raise TypeError(
            "save_ebm_artifacts expects a Stage3Artifact since PR 33; "
            "train_ebm_stage(save=True) saves it for you."
        )
    return Stage3ArtifactStore(base_path, "ebm_stage3").save(payload)


__all__ = ["Stage3Artifact", "Stage3ArtifactStore", "find_stage3_artifact",
           "save_ebm_artifacts"]
