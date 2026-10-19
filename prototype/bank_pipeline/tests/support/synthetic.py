"""
Small, deterministic bank-like data for contract tests.

The columns, encodings and dtypes follow what ``BankPreprocessor(drop_leaky=True)``
produces (``data/preprocessing/sql/transform.sql`` + ``bank_preprocessing.py``),
so the frames can stand in for ``GLOBAL_SPLIT`` wherever a test needs one.

Rows carry a deliberately irregular index: unique, gapped, shuffled and never
equal to the row position. Code that silently mixes positional and label
indexing, or resets the index, therefore produces a visibly different result
instead of passing by accident.

Public API
----------
BANK_SCHEMA            feature column -> dtype, in preprocessed column order
TARGET                 name of the binary target column
irregular_index(n)     the index used by every builder
make_bank_frame(...)   features + target, with or without planted signal
split_frame(frame)     GLOBAL_SPLIT-style dict: X_train, X_test, y_train, y_test
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BANK_SCHEMA: Dict[str, str] = {
    "age": "int64",
    "job": "int8",
    "marital": "int8",
    "education": "int8",
    "default": "int8",
    "housing": "int8",
    "loan": "int8",
    "contact": "int8",
    "month": "int8",
    "day_of_week": "int8",
    "campaign": "int64",
    "previous": "int64",
    "emp.var.rate": "float64",
    "cons.price.idx": "float64",
    "cons.conf.idx": "float64",
    "euribor3m": "float64",
    "nr.employed": "float64",
}
TARGET = "y"


def irregular_index(n: int, seed: int = 0, start: int = 10_000, step: int = 7) -> pd.Index:
    """Unique, gapped, shuffled int64 labels, all far above any row position."""
    rng = np.random.default_rng(seed)
    return pd.Index(rng.permutation(start + step * np.arange(n)), dtype="int64")


def make_bank_frame(
    n: int = 400,
    *,
    seed: int = 0,
    positive_rate: float = 0.2,
    signal: bool = True,
) -> pd.DataFrame:
    """
    Bank-like frame: ``BANK_SCHEMA`` columns followed by an int8 ``y``.

    Parameters
    ----------
    n : int
        Number of rows.
    seed : int
        Fully determines the frame; the same seed gives an identical frame.
    positive_rate : float
        Exact share of positives (``round(n * positive_rate)`` rows).
    signal : bool
        True: ``y`` depends on ``euribor3m``, ``contact`` and ``previous``, so
        models learn something. False: ``y`` is independent of every feature,
        a null-signal control on which honest out-of-fold scores sit at chance.

    Notes
    -----
    The economic columns are continuous, so every row is unique; recorders
    rely on that to identify rows passed as plain arrays.
    """
    rng = np.random.default_rng(seed)
    cols = {
        "age": rng.integers(18, 90, n),
        "job": rng.integers(0, 11, n),
        "marital": rng.integers(0, 3, n),
        "education": rng.integers(-1, 7, n),
        "default": rng.choice([-1, 0, 1], n),
        "housing": rng.choice([-1, 0, 1], n),
        "loan": rng.choice([-1, 0, 1], n),
        "contact": rng.integers(0, 2, n),
        "month": rng.choice(np.arange(3, 13), n),
        "day_of_week": rng.integers(0, 5, n),
        "campaign": 1 + rng.poisson(1.5, n),
        "previous": rng.poisson(0.3, n),
        "emp.var.rate": rng.uniform(-3.4, 1.4, n),
        "cons.price.idx": rng.uniform(92.2, 94.8, n),
        "cons.conf.idx": rng.uniform(-50.8, -26.9, n),
        "euribor3m": rng.uniform(0.6, 5.0, n),
        "nr.employed": rng.uniform(4963.6, 5228.1, n),
    }
    frame = pd.DataFrame(cols).astype(BANK_SCHEMA)

    n_pos = int(round(n * positive_rate))
    if not 0 < n_pos < n:
        raise ValueError(f"positive_rate={positive_rate} leaves a single class for n={n}")
    if signal:
        eur = frame["euribor3m"].to_numpy()
        score = (
            -1.5 * (eur - eur.mean()) / eur.std()
            + 1.0 * (frame["contact"].to_numpy() == 0)
            + 0.8 * frame["previous"].to_numpy()
            + rng.logistic(size=n)
        )
        positives = np.argsort(-score, kind="stable")[:n_pos]
    else:
        positives = rng.permutation(n)[:n_pos]
    y = np.zeros(n, dtype="int8")
    y[positives] = 1
    frame[TARGET] = y

    frame.index = irregular_index(n, seed)
    return frame


def split_frame(frame: pd.DataFrame, test_size: float = 0.25, seed: int = 0) -> dict:
    """
    Stratified train/test split with the ``GLOBAL_SPLIT`` key names.

    Returns
    -------
    dict
        ``X_train``, ``X_test`` (DataFrames), ``y_train``, ``y_test`` (Series),
        all keeping the frame's irregular index.
    """
    X, y = frame.drop(columns=[TARGET]), frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
