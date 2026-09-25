"""
Input preparation: row alignment and positional fold construction.

Everything downstream of this module works in POSITIONS, never pandas labels.
Mixing the two is the fold-misalignment bug this package is built to avoid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

__all__ = ["make_folds", "check_aligned"]


def make_folds(
    y: pd.Series,
    n_splits: int,
    random_state: int,
    stratify: bool = True,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Positional ``(train_idx, val_idx)`` pairs over ``range(len(y))``.

    Consumers slice the result with ``.iloc`` / numpy fancy indexing only.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    dummy = np.zeros((n, 1))
    return [
        (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
        for tr, va in splitter.split(dummy, y_arr)
    ]


def check_aligned(X: pd.DataFrame, y) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Enforce that X and y describe the same rows in the same order.

    Length equality is not enough: a y whose index was re-sorted upstream would
    pass a length check and silently scramble every label. When both carry a
    pandas index, the indexes must be identical.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")

    if isinstance(y, pd.Series):
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if not X.index.equals(y.index):
            raise ValueError(
                "X.index and y.index differ. Row order must match positionally; "
                "reindex y to X before calling (y = y.loc[X.index])."
            )
        return X, y

    y_arr = np.asarray(y)
    if len(X) != len(y_arr):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y_arr)}")
    return X, pd.Series(y_arr, index=X.index)