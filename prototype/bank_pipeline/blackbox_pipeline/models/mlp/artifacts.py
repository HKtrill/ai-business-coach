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

Versions (PR 37)
----------------
Every save writes an immutable, read-only version
``<name>_<YYYYMMDD_HHMMSS_ffffff>.joblib`` (UTC) with its own sidecars and
returns its path, which is what downstream stages should record.
``<name>.joblib`` and its sidecars are a mutable latest alias, a copy of the
newest version (see ``shared.artifact_versions``). A fixed-name artifact saved
before PR 37 is kept as ``<name>_preserved_<sha256[:12]>.joblib`` before the
alias is replaced.
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

from shared.artifact_versions import save_versioned

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
        The immutable version ``<name>_<id>.joblib``; ``<name>.joblib`` is
        updated as the mutable latest alias (module docstring).

    Raises
    ------
    ValueError
        If the stage is not fitted.
    """
    if not getattr(stage, "fitted", False):
        raise ValueError("Stage is not fitted; nothing to save")

    if stage_output is None:
        print(
            "⚠️  no stage_output passed — this artifact will hold no test "
            "probabilities, and StageOutput.load() on it will fail. Pass "
            "stage_output=stage.to_stage_output(X_test, y_test) for Stage 4."
        )

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

    meta = {
        k: v for k, v in payload.items()
        if k not in ("scaler", "model", "calibrated_model", "train_proba_oof",
                     "train_fold_id", "y_train", "stage_output")
    }

    tables = {}
    if stage.threshold_sweep is not None:
        tables["_threshold_sweep.csv"] = (
            lambda p: stage.threshold_sweep.to_csv(p, index=False))

    if stage.tuning_trials is not None:
        tables["_tuning_trials.csv"] = lambda p: stage.tuning_trials.to_csv(p, index=False)

    if getattr(stage.model, "history_", None) is not None:
        tables["_training_history.csv"] = (
            lambda p: stage.model.history_.to_csv(p, index=False))

    saved = save_versioned(
        out, name,
        write_model=lambda p: joblib.dump(payload, p, compress=3),
        metadata=meta,
        dump_metadata=lambda m: json.dumps(m, indent=2, default=str),
        tables=tables,
        companions=("_threshold_sweep.csv", "_tuning_trials.csv", "_training_history.csv"),
    )

    print(f"✅ saved Stage 1 MLP → {saved.path} (immutable version)")
    print(f"   latest alias : {saved.alias} (mutable copy of the newest version)")
    for kept in saved.preserved:
        print(f"   kept earlier unversioned file as {kept.name}")
    return saved.path


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