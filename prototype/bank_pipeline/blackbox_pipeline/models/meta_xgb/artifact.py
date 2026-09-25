"""
blackbox_pipeline.models.meta_xgb.artifact
==========================================
``MetaXGBArtifact`` — the typed Stage 4 payload.

The learned part is a ``Stage3Artifact`` (``learner``) produced by the shared
runner, so tuning / meta-OOF / refit / calibration / threshold blocks have the
exact schema both Stage 3 artifacts use. Around it, the Stage 4 fields: input
provenance, feature contract, operating point, abstention, evaluations and the
comparison with Meta-EBM.

Four kinds of prediction, never mixed
-------------------------------------
``stage4_inputs``            upstream OOF features (train) / refit (test) — provenance
``learner.oof``              META-level OOF predictions + meta fold ids (train)
``learner.refit``            full-train Meta-XGB (model, calibrator, in-sample
                             train_proba — diagnostic only — and test_proba)
``test``                     held-out Stage 4 outputs used for evaluation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np

SCHEMA = "meta_xgb/1"


@dataclass
class MetaXGBArtifact:
    created_at: str
    split_fingerprint: str
    split_fingerprints: dict
    index_train: list
    index_test: list
    feature_names: list
    feature_contract: dict
    stage4_inputs: dict
    fold_provenance: dict
    learner: Any                       # shared Stage3Artifact (stage="stage4")
    operating: dict
    abstention: dict
    train_oof_eval: dict
    test: dict
    events: list
    validation: dict
    comparison: Optional[dict] = None

    stage: str = "stage4"
    model_family: str = "meta_xgb"
    arm: str = "blackbox"
    counterpart: str = "glass_pipeline.meta_ebm (Meta-EBM)"
    schema: str = SCHEMA

    # ---- convenience views --------------------------------------------
    @property
    def best_params(self) -> dict:
        return dict(self.learner.best_params)

    @property
    def cv_score(self) -> float:
        return float(self.learner.cv_score)

    @property
    def meta_oof_proba(self) -> np.ndarray:
        return np.asarray(self.learner.oof["proba"])

    @property
    def meta_oof_fold_id(self) -> np.ndarray:
        return np.asarray(self.learner.oof["fold_id"])

    @property
    def model(self):
        return self.learner.refit.get("model")

    @property
    def calibration(self) -> dict:
        return dict(self.learner.calibration)

    def summary(self) -> dict:
        """JSON-safe sidecar: no arrays, no models."""
        def clean(d):
            return {k: v for k, v in d.items()
                    if not isinstance(v, (np.ndarray, list)) or k in ("grid",)}
        L = self.learner
        return {
            "schema": self.schema, "stage": self.stage, "model_family": self.model_family,
            "arm": self.arm, "counterpart": self.counterpart, "created_at": self.created_at,
            "split_fingerprint": self.split_fingerprint,
            "split_fingerprints": self.split_fingerprints,
            "n_train": len(self.index_train), "n_test": len(self.index_test),
            "feature_names": self.feature_names, "feature_contract": self.feature_contract,
            "upstream": {n: {k: s.get(k) for k in ("artifact_path", "train_source", "test_source",
                                                    "probability_space", "fingerprint_scheme")}
                         for n, s in self.stage4_inputs["streams"].items()},
            "folds": self.fold_provenance,
            "tuning": {k: v for k, v in L.tuning.items() if k != "top_trials"},
            "best_params": L.best_params, "cv_score": L.cv_score,
            "calibration": L.calibration,
            "operating": self.operating,
            "abstention": clean(self.abstention),
            "train_oof_eval": self.train_oof_eval,
            "test": {"no_abstain": self.test["no_abstain"],
                     "with_abstain": self.test["with_abstain"]},
            "events": self.events,
            "validation_passed": {k: all(r["passed"] for r in v)
                                  for k, v in self.validation.items()},
            "comparison": None if self.comparison is None
            else self.comparison.get("summary"),
        }


def save_meta_xgb(artifact: MetaXGBArtifact, base_path: str = "models/meta_xgb",
                  stem: str = "meta_xgb_stage4") -> str:
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(base_path, f"{stem}_{ts}.joblib")
    joblib.dump(artifact, path)
    with open(path[:-len(".joblib")] + ".json", "w") as fh:
        json.dump(artifact.summary(), fh, indent=2, default=str)
    return path


def load_meta_xgb(path: str, expected_split_fingerprint: Optional[str] = None) -> MetaXGBArtifact:
    art = joblib.load(path)
    if getattr(art, "schema", None) != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} artifact")
    if expected_split_fingerprint is not None and \
            expected_split_fingerprint not in art.split_fingerprints.values():
        raise ValueError(f"{path}: built on another split {art.split_fingerprints}")
    return art
