"""
lr.artifacts
============
Save / load LR stage artifacts aligned with GLOBAL_SPLIT.

Artifact key contract (v3.0)
----------------------------
    train_proba_oof     OUT-OF-FOLD calibrated train probabilities  (Series)
    train_fold_id       fold each train row was held out in         (Series)
    test_proba          calibrated test probabilities               (Series)
    y_train, y_test     labels                                      (Series)
    optimal_threshold   CV-optimised F2 threshold
    threshold_source    HOW that threshold was chosen
    stage_output        the shared StageOutput contract, as a dict
    feature_contract    Stage 1 feature contract: id, version, ordered
                        features, fingerprint (v3.0)

v3.0 (PR 39): Stage 1 feature names
-----------------------------------
The three Stage 1 features were renamed (values unchanged; see
``shared.stage1.feature_engineering.STAGE1_FEATURE_RENAMES``). v3.0 artifacts
carry feature contract 2.0.0 and record it under ``feature_contract``. v2.0
artifacts keep the names they were saved with (contract 1.0.0) and still load
as they are; nothing in them is renamed. ``artifact_feature_contract`` reads
either.

What changed from v1.0, and why
-------------------------------
v1.0 wrote ``train_predictions = stage.predict_proba(X_train)``. That routes
through a calibrator fitted on all of X_train, so every training row scored
itself. Anything trained on it — Stage 4 above all — saw a GLASS column that
is sharper on train than it can be at test.

The honest out-of-fold probabilities were already being computed inside the
threshold sweep; they were simply discarded. v2.0 persists them under a
DIFFERENT key, so a loader written against v1.0 fails loudly on the missing
``train_predictions`` key instead of silently picking up in-sample numbers.

Everything is now an indexed ``pd.Series``. v1.0 wrote bare ndarrays, so
Stage 4 could only join the arms positionally.
"""

import os
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from shared.stage1.feature_engineering import (
    artifact_feature_contract,
    current_feature_contract_record,
)

#: Bumped when the artifact layout changes.
ARTIFACT_VERSION = "3.0"
#: Versions ``load_artifact`` reads without a warning: 2.0 holds the
#: historical feature names (contract 1.0.0), 3.0 the current ones (2.0.0).
READABLE_ARTIFACT_VERSIONS = ("2.0", "3.0")


def save_artifact(stage, output_dir: str = "./models/lr") -> dict:
    """Persist the fitted CalibratedLRStage to a timestamped joblib file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if stage.proba_train_oof is None:
        raise ValueError(
            "Stage has no OOF training probabilities. This artifact version "
            "will not persist in-sample train predictions — refit with the "
            "current CalibratedLRStage."
        )

    feature_contract = current_feature_contract_record(stage.feature_names, "LR artifact")
    out = stage.to_stage_output()

    print(f"\n💾 Saving LR artifact (v{ARTIFACT_VERSION})...")
    print(f"   train_proba_oof : {len(out.train_proba_oof)} samples (out-of-fold)")
    print(f"   test_proba      : {len(out.test_proba)} samples")
    print(f"   threshold       : {out.threshold:.4f} [{out.threshold_source}]")

    artifact = {
        # ── Provenance ────────────────────────────────────────────────────────
        "artifact_version": ARTIFACT_VERSION,
        "training_date":    timestamp,
        "config":           stage.config_dict(),
        "environment": {
            "python":  platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy":   np.__version__,
            "pandas":  pd.__version__,
        },

        # ── Models ────────────────────────────────────────────────────────────
        "base_model":       stage.model,
        "calibrated_model": stage.calibrated_model,
        "scaler":           stage.scaler,
        "feature_names":    stage.feature_names,
        "feature_contract": feature_contract,

        # ── Predictions (GLOBAL_SPLIT aligned, indexed) ───────────────────────
        # train side is OUT-OF-FOLD; test side is from the full-train model.
        "train_proba_oof": out.train_proba_oof,
        "train_fold_id":   out.train_fold_id,
        "test_proba":      out.test_proba,

        # ── Labels ────────────────────────────────────────────────────────────
        "y_train": out.y_train,
        "y_test":  out.y_test,

        # ── Operating point ───────────────────────────────────────────────────
        "optimal_threshold": stage.optimal_threshold,
        "threshold_source":  out.threshold_source,
        "threshold_sweep":   stage.threshold_sweep,
        "cv_f2":             stage.cv_f2,

        # ── Diagnostics ───────────────────────────────────────────────────────
        "best_params":           stage.best_params,
        "best_cv_roc_auc":       stage.best_cv_roc_auc,
        "tuning_trials":         stage.tuning_trials,
        "calibration_method":    stage.calibration_method,     # winner
        "calibration_requested": stage.calibration_requested,  # request
        "calibration_metrics":   stage.calibration_metrics,
        "oof_provenance":        stage.oof_provenance_,
        "performance_metrics":   stage.metrics,

        # ── Shared contract ───────────────────────────────────────────────────
        "stage_output": out.to_dict(),
    }

    path = os.path.join(output_dir, f"lr_calibrated_{timestamp}.joblib")
    joblib.dump(artifact, path, compress=3)
    print(f"   ✅ Saved → {path}")
    return {"path": path}


def load_artifact(path: str) -> dict:
    """
    Load a saved calibrated LR artifact as it was saved.

    v2.0 (historical feature names) and v3.0 load silently; a v3.0 artifact
    whose ``feature_contract`` disagrees with its feature names is refused.
    Any other version loads with a warning.
    """
    artifact = joblib.load(path)
    version = artifact.get("artifact_version")
    if version == ARTIFACT_VERSION:
        artifact_feature_contract(artifact)          # raises on a mismatched record
    elif version not in READABLE_ARTIFACT_VERSIONS:
        print(
            f"⚠️  artifact version {version} (expected one of {READABLE_ARTIFACT_VERSIONS}). "
            "v1.0 artifacts carry IN-SAMPLE train predictions under "
            "'train_predictions' — do not feed them to Stage 4."
        )
    return artifact
