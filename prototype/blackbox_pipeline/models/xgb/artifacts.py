"""
blackbox_pipeline.models.xgb.artifacts
=======================================
The Stage 3 payload, and the OOF / refit separation.

The single most important thing in this file is that two different sets of
predictions exist and they are never allowed to blur:

``oof``
    Out-of-fold predictions over the TRAINING split. Every row was scored by a
    booster that did not train on it. **This is what Stage 4 trains on.**

``refit``
    Predictions from the single model refitted on the full training split.
    Its ``test_*`` arrays are the held-out evaluation. Its ``train_*`` arrays
    are IN-SAMPLE and exist only for parity with the GLASS artifact, which
    stores exactly that and nothing else — they must not be used to fit
    anything downstream.

``Stage3Artifact.stage4_frame()`` returns the training-side table Stage 4
should consume; ``stage4_test_frame()`` returns its test-time counterpart from
the refit model. Using the wrong one is the failure this schema is shaped to
prevent, so the OOF frame is the one with the convenient accessor and the
in-sample columns are buried inside ``refit``.

Both frames' ``decision`` / ``margin`` / ``confidence`` use the RAW
probability. The test side uses the global OOF threshold. The train side uses
fold-nested thresholds (each fold's chosen without its own labels) when
``config.nested_oof_threshold`` is True, else the global one; the per-row
values are in ``oof["threshold_per_row"]``.

Loading a saved artifact requires this package to be importable (pickled
classes).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

__all__ = ["Stage3Artifact", "Stage3ArtifactStore"]


def _decision_block(proba: np.ndarray, threshold) -> dict[str, np.ndarray]:
    """
    Binary decision, signed margin and normalised confidence.

    ``threshold`` is a scalar or a per-row array (fold-nested thresholds).

    ``margin``      ``p - threshold``; sign is the decision, magnitude is the
                    distance from the boundary.
    ``confidence``  ``|margin|`` rescaled by the distance available on that
                    side of the boundary, so it lands in [0, 1] regardless of
                    where the threshold sits. A row exactly on the boundary has
                    confidence 0; a row at p=1 with threshold 0.3 has
                    confidence 1.
    """
    p = np.asarray(proba, dtype=float)
    t = np.broadcast_to(np.asarray(threshold, dtype=float), p.shape)
    decision = (p >= t).astype(int)
    margin = p - t
    upper = np.maximum(1.0 - t, 1e-12)
    lower = np.maximum(t, 1e-12)
    confidence = np.where(margin >= 0, margin / upper, -margin / lower)
    return {
        "decision": decision,
        "margin": margin,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "state": np.where(decision == 1, "flagged", "not_flagged").astype(object),
    }


@dataclass
class Stage3Artifact:
    """Canonical Stage 3 payload."""

    # ---- identity ---------------------------------------------------------
    stage: str
    model_family: str
    created_at: str
    config: dict
    mirror_report: dict

    # ---- input contract ---------------------------------------------------
    features: list[str]
    index_train: list
    index_test: list
    split_fingerprint: Optional[str] = None
    cleaning_report: dict = field(default_factory=dict)

    # ---- partition --------------------------------------------------------
    fold_plan: dict = field(default_factory=dict)
    class_weights: dict = field(default_factory=dict)

    # ---- search -----------------------------------------------------------
    tuning: dict = field(default_factory=dict)
    cv_report: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    cv_score: float = float("nan")

    # ---- OOF block — Stage 4 trains on THIS -------------------------------
    oof: dict = field(default_factory=dict)

    # ---- refit block — held-out evaluation --------------------------------
    refit: dict = field(default_factory=dict)

    # ---- operating point + calibration ------------------------------------
    threshold: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    # ---- metrics ----------------------------------------------------------
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Stage 4 interface
    # ------------------------------------------------------------------
    def stage4_frame(self) -> pd.DataFrame:
        """
        Training-side Stage 3 features for Stage 4. Leakage-safe.

        One row per training row, in the training split's own index and order.
        """
        block = self.oof
        return pd.DataFrame(
            {
                "xgb_stage3_proba": np.asarray(block["proba"]),
                "xgb_stage3_proba_calibrated": np.asarray(
                    block["proba_calibrated"]
                ),
                "xgb_stage3_decision": np.asarray(block["decision"]),
                "xgb_stage3_confidence": np.asarray(block["confidence"]),
                "xgb_stage3_margin": np.asarray(block["margin"]),
                "xgb_stage3_state": np.asarray(block["state"], dtype=object),
                "xgb_stage3_fold_id": np.asarray(block["fold_id"]),
            },
            index=pd.Index(self.index_train),
        )

    def stage4_test_frame(self) -> pd.DataFrame:
        """
        Test-side counterpart, from the refit model.

        Used when Stage 4 is EVALUATED, never when it is trained. Same columns
        as ``stage4_frame()``; ``xgb_stage3_fold_id`` is ``-1`` (no fold).
        """
        block = self.refit
        return pd.DataFrame(
            {
                "xgb_stage3_proba": np.asarray(block["test_proba"]),
                "xgb_stage3_proba_calibrated": np.asarray(
                    block["test_proba_calibrated"]
                ),
                "xgb_stage3_decision": np.asarray(block["test_decision"]),
                "xgb_stage3_confidence": np.asarray(block["test_confidence"]),
                "xgb_stage3_margin": np.asarray(block["test_margin"]),
                "xgb_stage3_state": np.asarray(block["test_state"], dtype=object),
                "xgb_stage3_fold_id": np.full(len(block["test_proba"]), -1),
            },
            index=pd.Index(self.index_test),
        )

    # ------------------------------------------------------------------
    def to_glass_schema(self) -> dict:
        """
        The GLASS Stage 3 key names, for code written against the EBM artifact.

        ``train_predictions`` maps to the IN-SAMPLE refit column, because that
        is what the EBM key means. Anything training on Stage 3 output should
        use ``stage4_frame()`` instead — this mapping exists for comparison and
        Venn-tracing code, not for model fitting.
        """
        return {
            "model": self.refit.get("model"),
            "calibrated_model": self.refit.get("calibrator"),
            "train_predictions": self.refit.get("train_proba"),
            "test_predictions": self.refit.get("test_proba"),
            "train_predictions_calibrated": self.refit.get(
                "train_proba_calibrated"
            ),
            "test_predictions_calibrated": self.refit.get(
                "test_proba_calibrated"
            ),
            "optimal_threshold": self.threshold.get("oof", {}).get("threshold"),
            "metrics": self.metrics.get("test_at_operating_threshold"),
            "metrics_at_half": self.metrics.get("test_at_half"),
            "ece": self.calibration.get("ece_before"),  # OOF ECE, not in-sample
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "features": self.features,
            "interactions": None,  # modelled implicitly — the experiment itself
        }

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Small, JSON-serialisable sidecar. No arrays, no models."""
        return {
            "stage": self.stage,
            "model_family": self.model_family,
            "created_at": self.created_at,
            "n_features": len(self.features),
            "features": self.features,
            "split_fingerprint": self.split_fingerprint,
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "tuning": {k: v for k, v in self.tuning.items()
                       if k not in ("top_trials",)},
            "cv_report": self.cv_report.get("aggregate", {}),
            # The sweep DataFrame stays in the joblib payload only.
            "threshold": {k: v for k, v in self.threshold.items()
                          if k != "sweep"},
            "calibration": self.calibration,
            "class_weights": self.class_weights,
            "fold_plan": {k: v for k, v in self.fold_plan.items()
                          if k != "fold_id"},
            "metrics": self.metrics,
            "mirror_report": self.mirror_report,
            "cleaning_report": self.cleaning_report,
        }

    def describe(self) -> None:
        s = self.summary()
        print("=" * 78)
        print(f"  STAGE 3 — {self.model_family.upper()}  ({self.created_at})")
        print("=" * 78)
        print(f"  features        : {s['n_features']}")
        print(f"  best CV F2      : {self.cv_score:.6f}")
        print(f"  operating thresh: {self.threshold['oof']['threshold']:.4f} "
              f"({self.threshold['oof']['selected_on']})")
        if "test_oracle" in self.threshold:
            print(f"  test-oracle th. : "
                  f"{self.threshold['test_oracle']['threshold']:.4f} "
                  f"(reference only)")
        print(f"  calibration     : "
              f"{'applied' if self.calibration.get('applied') else 'skipped'} "
              f"— {self.calibration.get('reason')}")
        for name in ("oof_at_operating_threshold",
                     "test_at_operating_threshold", "test_at_half"):
            m = self.metrics.get(name)
            if m:
                print(f"  {name:<28} F2={m['f2']:.4f}  rec={m['recall']:.4f}  "
                      f"prec={m['precision']:.4f}  AUC={m['roc_auc']:.4f}")
        print("=" * 78)


class Stage3ArtifactStore:
    """
    Save / load.

    ``save`` writes ``<stem>_<timestamp>.joblib`` plus a JSON summary sidecar,
    and (if timestamped) a full ``<stem>_latest.joblib`` copy — a duplicate
    file, not a symlink.
    """

    def __init__(
        self,
        base_path: str = "models/xgb",
        stem: str = "xgb_stage3",
        timestamped: bool = True,
        write_latest_pointer: bool = True,
    ):
        self.base_path = base_path
        self.stem = stem
        self.timestamped = bool(timestamped)
        self.write_latest_pointer = bool(write_latest_pointer)

    # ------------------------------------------------------------------
    def save(self, artifact: Stage3Artifact, verbose: bool = True) -> str:
        os.makedirs(self.base_path, exist_ok=True)
        if self.timestamped:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{self.stem}_{ts}.joblib"
        else:
            name = f"{self.stem}.joblib"
        path = os.path.join(self.base_path, name)
        joblib.dump(artifact, path)

        # JSON sidecar: readable without unpickling a booster.
        side = os.path.join(self.base_path, name.replace(".joblib", ".json"))
        with open(side, "w") as fh:
            json.dump(artifact.summary(), fh, indent=2, default=str)

        if self.write_latest_pointer and self.timestamped:
            latest = os.path.join(self.base_path, f"{self.stem}_latest.joblib")
            joblib.dump(artifact, latest)

        if verbose:
            print(f"\n  saved Stage 3 artifact → {path}")
            print(f"        summary sidecar  → {side}")
            if self.write_latest_pointer and self.timestamped:
                print(f"        latest pointer   → "
                      f"{self.stem}_latest.joblib")
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str) -> Stage3Artifact:
        return joblib.load(path)

    def load_latest(self) -> Stage3Artifact:
        latest = os.path.join(self.base_path, f"{self.stem}_latest.joblib")
        if os.path.exists(latest):
            return self.load(latest)
        candidates = sorted(
            f for f in os.listdir(self.base_path)
            if f.startswith(self.stem) and f.endswith(".joblib")
        )
        if not candidates:
            raise FileNotFoundError(
                f"no {self.stem}*.joblib under {self.base_path}"
            )
        return self.load(os.path.join(self.base_path, candidates[-1]))