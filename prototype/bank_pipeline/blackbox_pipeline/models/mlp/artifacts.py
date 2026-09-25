"""
blackbox_pipeline.models.mlp.artifacts

Persistence helpers for the fitted Stage 1 MLP.

Saves the model payload, metadata, threshold sweep, tuning trials, and training
history when available.

v2.0 (PR 33)
------------
v1.0 saved no probabilities at all, so Stage 4 could not load this arm's
predictions from disk. v2.0 always writes the training out-of-fold
probabilities, fold ids and labels (indexed Series), and — when a
``StageOutput`` is passed — the full shared contract including test
probabilities and the threshold's provenance.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

__all__ = ["save_stage1_mlp", "load_stage1_mlp"]

#: Bumped when the artifact layout changes.
_ARTIFACT_VERSION = "2.0"


def save_stage1_mlp(
    stage,
    output_dir: str | Path,
    *,
    name: str = "stage1_mlp",
    test_metrics: Optional[dict] = None,
    stage_output=None,
) -> Path:
    """
    Save a fitted Stage 1 MLP and its diagnostics.

    Parameters
    ----------
    stage
        Fitted ``CalibratedStage1MLP``.
    output_dir
        Destination directory.
    name
        Shared filename stem.
    test_metrics
        Optional held-out metrics to include.
    stage_output
        Optional ``StageOutput`` from ``stage.to_stage_output(X_test, y_test)``.
        Stored as a dict under ``stage_output``; this is what Stage 4 loads.

    Returns
    -------
    pathlib.Path
        Path to the saved ``.joblib`` artifact.

    Raises
    ------
    ValueError
        If the stage is not fitted.
    """
    if not getattr(stage, "fitted", False):
        raise ValueError("Stage is not fitted; nothing to save")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_version": _ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": stage.config.to_dict(),
        "feature_names": list(stage.feature_names_),
        "scaler": stage.scaler,
        "model": stage.model,
        "calibrated_model": stage.calibrated_model,
        "best_params": stage.best_params,
        "best_cv_roc_auc": stage.best_cv_roc_auc,
        "calibration_method": stage.calibration_method,
        "calibration_metrics": stage.calibration_metrics,
        "optimal_threshold": stage.optimal_threshold,
        "cv_f2": stage.cv_f2,
        "oof_provenance": stage.oof_provenance_,
        "threshold_source": (
            stage_output.threshold_source if stage_output is not None
            else stage.THRESHOLD_SOURCE
        ),
        "test_metrics": test_metrics,
        # ── Predictions (indexed; train side is OUT-OF-FOLD) ──────────────────
        "train_proba_oof": stage.proba_train_oof,
        "train_fold_id": stage.train_fold_id,
        "y_train": stage.y_train_,
        "stage_output": stage_output.to_dict() if stage_output is not None else None,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    path = out / f"{name}.joblib"
    joblib.dump(payload, path, compress=3)

    meta = {
        k: v for k, v in payload.items()
        if k not in ("scaler", "model", "calibrated_model", "train_proba_oof",
                     "train_fold_id", "y_train", "stage_output")
    }
    (out / f"{name}_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    if stage.threshold_sweep is not None:
        stage.threshold_sweep.to_csv(out / f"{name}_threshold_sweep.csv", index=False)

    if stage.tuning_trials is not None:
        stage.tuning_trials.to_csv(out / f"{name}_tuning_trials.csv", index=False)

    if getattr(stage.model, "history_", None) is not None:
        stage.model.history_.to_csv(out / f"{name}_training_history.csv", index=False)

    print(f"✅ saved Stage 1 MLP → {path}")
    return path


def load_stage1_mlp(path: str | Path) -> dict:
    """
    Load a saved Stage 1 MLP artifact.

    Parameters
    ----------
    path
        Path to the saved ``.joblib`` file.

    Returns
    -------
    dict
        Raw artifact payload.

    Notes
    -----
    A version mismatch produces a warning but does not prevent loading.
    """
    payload = joblib.load(Path(path))

    if payload.get("artifact_version") != _ARTIFACT_VERSION:
        print(
            f"⚠️  artifact version {payload.get('artifact_version')} "
            f"(expected {_ARTIFACT_VERSION}) — fields may have moved"
        )

    return payload

