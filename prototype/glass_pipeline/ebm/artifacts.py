import os
import joblib
from datetime import datetime


def save_ebm_artifacts(payload, base_path="models/ebm"):
    """
    Save SINGLE canonical EBM artifact for Stage 3.
    """
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_path, f"ebm_stage3_{ts}.joblib")
    joblib.dump(payload, path)
    print(f"💾 Saved EBM artifact → {path}")
    return path

