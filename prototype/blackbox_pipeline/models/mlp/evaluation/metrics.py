from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

__all__ = ["binary_metrics", "metrics_table"]


def binary_metrics(
    y_true,
    proba,
    threshold: float = 0.5,
    calibration_method: Optional[str] = None,
) -> pd.Series:
    from glass_pipeline.lr.evaluation import compute_metrics

    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)

    m = compute_metrics(y_true, pred, p, threshold, calibration_method)
    m = {k: v for k, v in m.items() if k not in ("threshold", "calibration")}
    m["pr_auc"] = float(average_precision_score(y_true, p))
    m["pred_pos_rate"] = float(pred.mean())
    return pd.Series(m)


def metrics_table(
    y_true,
    proba,
    label: str,
    thresholds: Sequence[float],
    calibration_method: Optional[str] = None,
) -> pd.DataFrame:
    rows: Dict[str, pd.Series] = {}
    for t in thresholds:
        rows[f"{label} @ {t:.2f}"] = binary_metrics(
            y_true, proba, t, calibration_method
        )
    return pd.DataFrame(rows).T