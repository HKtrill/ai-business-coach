import os, joblib
from datetime import datetime

def save_meta_ebm(artifact, base_path="./models/meta_ebm"):
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{base_path}/meta_ebm_{ts}.joblib"
    joblib.dump(artifact, path)
    return path
