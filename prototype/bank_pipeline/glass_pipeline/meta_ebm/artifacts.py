"""
glass_pipeline.meta_ebm.artifacts
=================================
Persist the Stage 4 Meta-EBM artifact as ``meta_ebm_<timestamp>.joblib``.
"""

import os
from datetime import datetime

import joblib


def save_meta_ebm(artifact, base_path="./models/meta_ebm"):
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(base_path, f"meta_ebm_{ts}.joblib")
    joblib.dump(artifact, path)
    return path
