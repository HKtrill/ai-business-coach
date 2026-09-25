"""
blackbox_pipeline.models.mlp.features

The Stage 1 input contract — which columns, and what a valid label vector is.

Stage 1 sees exactly the three engineered features the GLASS LR stage sees
(``glass_pipeline.lr.feature_engineering.LR_FEATURES``). Holding that list in
one place means drift between the two arms fails loudly instead of quietly
changing what "Stage 1" means.

Notes
-----
Everything here is a guard that runs before fitting: it validates inputs and
raises, and computes nothing. :func:`check_labels` lives beside
:func:`select_features` because they are the same kind of check — one on the
columns, one on the targets — and the stage applies them back to back.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

__all__ = [
    "STAGE1_FEATURES",
    "select_features",
    "check_labels",
    "assert_matches_glass",
]

#: Canonical Stage 1 feature names, in the order the scaler and MLP expect.
STAGE1_FEATURES: Tuple[str, ...] = (
    "cellular_crisis",
    "euribor3m_local_rate",
    "dow_month_encoded",
)


def select_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return the Stage 1 columns in canonical order.

    Parameters
    ----------
    X : pandas.DataFrame
        Frame holding exactly the ``STAGE1_FEATURES`` columns, in any order.

    Returns
    -------
    pandas.DataFrame
        ``X`` reordered to ``STAGE1_FEATURES``.

    Raises
    ------
    TypeError
        If ``X`` is not a DataFrame.
    ValueError
        If the columns are not exactly ``STAGE1_FEATURES``. The message lists
        what is missing and what is extra.

    Notes
    -----
    Strict on purpose. An extra or missing column means the caller handed Stage 1
    a different feature set than it was fitted on, and reordering silently would
    hide that. Column order matters downstream because the scaler and the MLP are
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


def assert_matches_glass(lr_features) -> None:
    """
    Check that Stage 1 and the GLASS LR stage still share a feature set.

    Parameters
    ----------
    lr_features : Iterable[str]
        ``glass_pipeline.lr.feature_engineering.LR_FEATURES``.

    Raises
    ------
    AssertionError
        If the two feature sets differ. The message shows both, sorted.

    Notes
    -----
    Compares as SETS, so it catches a feature being added, dropped or renamed
    but not a reordering. That is the right check here — each arm applies its
    own ``select_features`` to impose its own order, so only membership has to
    agree. Order still matters WITHIN an arm; see :func:`select_features`.
    """
    if set(lr_features) != set(STAGE1_FEATURES):
        raise AssertionError(
            "Stage 1 features have drifted from GLASS LR_FEATURES.\n"
            f"  STAGE1_FEATURES : {sorted(STAGE1_FEATURES)}\n"
            f"  LR_FEATURES     : {sorted(lr_features)}"
        )
