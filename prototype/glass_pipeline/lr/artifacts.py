"""
lr.artifacts
============
Save / load LR stage artifacts aligned with GLOBAL_SPLIT.

Artifact key contract
---------------------
    train_predictions   calibrated train probabilities  (via calibrated_model)
    test_predictions    calibrated test probabilities
    y_train             training labels
    y_test              test labels
    optimal_threshold   CV-optimised F2 threshold

"""

import os
import numpy as np
import joblib
from datetime import datetime


def save_artifact(stage, output_dir: str = "./models/lr") -> dict:
    """Persist the fitted CalibratedLRStage to a timestamped joblib file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_probs = stage.predict_proba(stage._X_train_full)
    test_probs  = stage.predict_proba(stage._X_test_full)

    print(f"\n💾 Saving LR artifact...")
    print(f"   train_predictions : {len(train_probs)} samples")
    print(f"   test_predictions  : {len(test_probs)} samples")

    artifact = {
        # ── Models ────────────────────────────────────────────────────────────
        "base_model":       stage.model,
        "calibrated_model": stage.calibrated_model,
        "scaler":           stage.scaler,
        "feature_names":    stage.feature_names,

        # ── Predictions (GLOBAL_SPLIT aligned, always calibrated) ─────────────
        "train_predictions": train_probs,
        "test_predictions":  test_probs,

        # ── Labels ────────────────────────────────────────────────────────────
        "y_train": np.array(stage._y_train_full),
        "y_test":  np.array(stage._y_test_full),

        # ── Threshold ─────────────────────────────────────────────────────────
        "optimal_threshold": stage.optimal_threshold,

        # ── Diagnostics ───────────────────────────────────────────────────────
        "best_params":         stage.best_params,
        "performance_metrics": stage.metrics,
        "calibration_metrics": stage.calibration_metrics,

        # ── Metadata ──────────────────────────────────────────────────────────
        "training_date": timestamp,
    }

    path = os.path.join(output_dir, f"lr_calibrated_{timestamp}.joblib")
    joblib.dump(artifact, path)
    print(f"   ✅ Saved → {path}")
    return {"path": path}