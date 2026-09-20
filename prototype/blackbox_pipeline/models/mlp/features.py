"""
blackbox_pipeline.models.mlp.features
======================================
The Stage 1 feature contract.

Stage 1 sees exactly the three engineered features the GLASS LR stage sees —
``glass_pipeline.lr.feature_engineering.LR_FEATURES``. Keeping the list in one
place means a drift between the two arms fails loudly at import rather than
quietly changing what "Stage 1" means.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

__all__ = ["STAGE1_FEATURES", "select_features", "assert_matches_glass"]

STAGE1_FEATURES: Tuple[str, ...] = (
    "cellular_crisis",
    "euribor3m_local_rate",
    "dow_month_encoded",
)


def select_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return the Stage 1 columns in canonical order.

    Strict on purpose: an extra or missing column means the caller handed Stage 1
    a different feature set than it was fitted on, and reordering silently would
    hide that. Column ORDER matters downstream because the scaler and the MLP are
    both positional once fitted.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")

    cols = list(X.columns)
    if set(cols) != set(STAGE1_FEATURES) or len(cols) != len(STAGE1_FEATURES):
        missing = [c for c in STAGE1_FEATURES if c not in cols]
        extra = [c for c in cols if c not in STAGE1_FEATURES]
        raise ValueError(
            f"Stage 1 expects exactly {list(STAGE1_FEATURES)}, got {cols}"
            + (f"\n  missing: {missing}" if missing else "")
            + (f"\n  extra:   {extra}" if extra else "")
        )
    return X[list(STAGE1_FEATURES)]


def assert_matches_glass(lr_features) -> None:
    """Fail loudly if the GLASS LR stage and Stage 1 have drifted apart."""
    if set(lr_features) != set(STAGE1_FEATURES):
        raise AssertionError(
            "Stage 1 features have drifted from GLASS LR_FEATURES.\n"
            f"  STAGE1_FEATURES : {sorted(STAGE1_FEATURES)}\n"
            f"  LR_FEATURES     : {sorted(lr_features)}"
        )
