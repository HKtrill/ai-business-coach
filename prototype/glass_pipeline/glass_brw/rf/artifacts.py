"""
glass_brw.rf.artifacts
=======================
Persist RF Stage 2 results to disk.

Saves rf_result (RFResult dataclass) and BRW_DATA (engineered feature dict)
as a single joblib artifact. GLASSBRWPipeline reads rf_result.model from
the loaded artifact.

Artifact key contract
---------------------
    pipe               Pipeline(clf=RandomForestClassifier) — full refit
    model              RandomForestClassifier — convenience accessor
    params             canonical hyperparams dict
    metrics_cv         10-fold CV metrics
    metrics_test       holdout test metrics
    feature_importance pd.DataFrame sorted by Gini importance
    brw_data           BRW_DATA dict (X_eng_train, y_eng_train, X_eng_test,
                       y_eng_test, feature_names)
    training_date      ISO timestamp string

Public API
----------
save_rf_artifact(rf_result, brw_data, output_dir) → dict
"""

from __future__ import annotations

import os
import joblib
from datetime import datetime

from .rf_training import RFResult

def save_rf_artifact(
    rf_result:  RFResult,
    brw_data:   dict,
    output_dir: str = "./models/rf",
) -> dict:
    """
    Persist rf_result and brw_data to a timestamped joblib file.

    Parameters
    ----------
    rf_result  : RFResult from train_rf_stage()
    brw_data   : BRW_DATA dict — {X_eng_train, y_eng_train,
                 X_eng_test, y_eng_test, feature_names}
    output_dir : directory to write the joblib file

    Returns
    -------
    dict with key 'path' pointing to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n💾 Saving RF artifact...")
    print(f"   CV AUC   : {rf_result.metrics_cv['auc_mean']:.4f} "
          f"± {rf_result.metrics_cv['auc_std']:.4f}")
    print(f"   Test AUC : {rf_result.metrics_test['auc']:.4f}")
    print(f"   Features : {brw_data['X_eng_train'].shape[1]} binary bins")
    print(f"   Train    : {brw_data['X_eng_train'].shape[0]:,} samples")
    print(f"   Test     : {brw_data['X_eng_test'].shape[0]:,} samples")

    artifact = {
        # ── Model ─────────────────────────────────────────────────────────
        "pipe":               rf_result.pipe,
        "model":              rf_result.model,
        "params":             rf_result.params,

        # ── Metrics ───────────────────────────────────────────────────────
        "metrics_cv":         rf_result.metrics_cv,
        "metrics_test":       rf_result.metrics_test,
        "feature_importance": rf_result.feature_importance,

        # ── Data contract for Glass-BRW ───────────────────────────────────
        "brw_data":           brw_data,

        # ── Metadata ──────────────────────────────────────────────────────
        "training_date":      timestamp,
    }

    path = os.path.join(output_dir, f"rf_result_{timestamp}.joblib")
    joblib.dump(artifact, path)
    print(f"   ✅ Saved → {path}")
    return {"path": path}