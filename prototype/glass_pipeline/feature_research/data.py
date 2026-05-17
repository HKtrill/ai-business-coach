"""
feature_research/data.py
=========================
Data loading and preprocessing entry point for the Glass Cascade pipeline.

Wraps the raw CSV read and :class:`BankPreprocessor` call behind a single
typed function so notebook Cell 2 is a one-liner and the loading contract is
testable in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.preprocessing.bank_preprocessing import BankPreprocessor


#: Default path to the raw dataset relative to the project root.
RAW_DATA_PATH: Path = Path("data/raw/bank-additional-full.csv")


def load_and_preprocess(
    path: Path | str = RAW_DATA_PATH,
    *,
    drop_leaky: bool = True,
) -> pd.DataFrame:
    """Load the UCI bank-marketing CSV and apply standard preprocessing.

    Parameters
    ----------
    path:
        Path to ``bank-additional-full.csv``.  Defaults to
        ``data/raw/bank-additional-full.csv`` relative to the working
        directory.
    drop_leaky:
        Passed to :class:`BankPreprocessor`.  When ``True`` (default) the
        call-duration and prior-contact features are dropped before any model
        sees the data.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame with binary 0/1 target column ``'y'``
        and all features encoded as numeric types.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> df = load_and_preprocess()
    >>> df.shape
    (41188, ...)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path.absolute()}")

    df_raw = pd.read_csv(path, sep=";")
    print(f"📂 Loaded raw data   : {df_raw.shape}  ({path})")

    preprocessor = BankPreprocessor(drop_leaky=drop_leaky)
    df_processed = preprocessor.fit_transform(df_raw)
    print(f"✅ After preprocessing: {df_processed.shape}")

    return df_processed