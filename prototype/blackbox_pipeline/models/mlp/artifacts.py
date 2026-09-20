"""
blackbox_pipeline.models.mlp.artifacts
=======================================
Persistence for a fitted Stage 1 stage.

Saves the scaler, the MLP, the calibrator, the frozen threshold, the tuning
trials and the OOF provenance string — enough to reproduce a prediction and to
audit how the threshold was chosen.
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

_ARTIFACT_VERSION = "1.0"


def save_stage1_mlp(
    stage,
    output_dir: str | Path,
    *,
    name: str = "stage1_mlp",
    test_metrics: Optional[dict] = None,
) -> Path:
    """Write a fitted CalibratedStage1MLP to ``output_dir``."""
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
        "test_metrics": test_metrics,
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
        if k not in ("scaler", "model", "calibrated_model")
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
    """Load a saved Stage 1 artifact as a plain dict."""
    payload = joblib.load(Path(path))
    if payload.get("artifact_version") != _ARTIFACT_VERSION:
        print(
            f"⚠️  artifact version {payload.get('artifact_version')} "
            f"(expected {_ARTIFACT_VERSION}) — fields may have moved"
        )
    return payload
