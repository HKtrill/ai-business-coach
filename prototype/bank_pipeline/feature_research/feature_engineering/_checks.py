"""
feature_research.feature_engineering._checks
=============================================
Small fail-loud guards shared by the feature-engineering modules.

- check_xy_aligned : refuse to pair X and y positionally when their row
                     identity differs (prevents silent label misalignment
                     after an index reset / reorder).
- assert_finite    : raise if engineered columns contain NaN or inf, instead
                     of silently imputing with a statistic of whatever split
                     is being transformed.
"""

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["check_xy_aligned", "assert_finite"]


def check_xy_aligned(X: pd.DataFrame, y, where: str) -> None:
    """Raise if X and y cannot be safely aligned row-for-row."""
    if len(X) != len(y):
        raise ValueError(f"{where}: len(X)={len(X)} != len(y)={len(y)}.")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError(
            f"{where}: X and y have different indices. Refusing to align "
            "them by position — reindex y to X (or fix the upstream reset/"
            "reorder) before fitting."
        )


def assert_finite(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    """Raise if any of `cols` contains NaN or +/-inf."""
    cols = list(cols)
    values = df[cols].to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        counts = {c: int(n) for c, n in zip(cols, bad.sum(axis=0)) if n}
        raise ValueError(f"{where}: non-finite values in engineered columns: {counts}")
