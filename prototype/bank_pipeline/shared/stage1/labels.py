"""
shared.stage1.labels
====================
The Stage 1 label contract, applied by both arms before anything is fitted.

Moved from ``blackbox_pipeline.models.mlp.features`` (which re-exports it) so
the GLASS ``CalibratedLRStage`` validates labels exactly as the MLP does
(audit F18: the LR stage previously had no binary check and failed late, inside
calibration, with an unrelated message).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["check_labels"]


def check_labels(X: pd.DataFrame, y) -> pd.Series:
    """
    Coerce ``y`` to a Series and check it lines up with ``X``.

    Parameters
    ----------
    X : pandas.DataFrame
        The feature frame ``y`` must match, typically straight out of
        :func:`select_features`.
    y : array-like or pandas.Series
        Binary labels, 0/1. A non-Series is wrapped using ``X.index``, which
        assumes it is already in ``X``'s row order.

    Returns
    -------
    pandas.Series
        ``y`` as a Series indexed like ``X``.

    Raises
    ------
    ValueError
        On a length mismatch, an index mismatch, or non-binary labels.

    Notes
    -----
    The index check is the one that earns its keep: misaligned labels pass every
    length check and silently train on the wrong targets. Reindex with
    ``y = y.loc[X.index]`` before fitting rather than relying on positional luck.

    Wrapping a bare array is positional by necessity — there is no index to
    verify — so pass a Series whenever you have one and get the check for free.
    """
    y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if not X.index.equals(y.index):
        raise ValueError(
            "X.index and y.index differ — reindex y to X before fitting "
            "(y = y.loc[X.index]). Misaligned labels pass every length "
            "check and silently train on the wrong targets."
        )
    if not np.isin(np.asarray(y), (0, 1)).all():
        raise ValueError("y must be binary 0/1")
    return y
