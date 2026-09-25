"""
Bank Marketing Dataset Preprocessor
=====================================
Handles all preprocessing for the UCI Bank Marketing dataset:
  - Target encoding          (yes/no → 1/0)
  - Binary encoding          (yes/no/unknown → 1/0/-1)
  - Ordinal encoding         (education, month, day_of_week, poutcome, contact)
  - Nominal encoding         (job, marital)
  - Leaky feature removal    (duration, pdays, poutcome)
  - Final validation         (all-numeric, null-free)

Encodings run in SQL (sql/*.sql) against a raw_bank table. raw_bank is read
from the file-backed database (see data/bank_database.py), or staged in memory
when a DataFrame is passed.

Usage
-----
    preprocessor = BankPreprocessor(drop_leaky=True)
    df_processed  = preprocessor.fit_transform()        # reads raw_bank from the database
    df_processed  = preprocessor.fit_transform(df_raw)  # stages df_raw as raw_bank

"""

from __future__ import annotations

import sqlite3
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd

from ..bank_database import (
    DEFAULT_DB_PATH,
    RAW_TABLE,
    raw_bank_exists,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# SQL location and encoded columns
# ---------------------------------------------------------------------------

_SQL_DIR = Path(__file__).resolve().parent / "sql"

# Columns encoded by transform.sql, in the order they are validated
_ENCODED_COLUMNS: List[str] = [
    "y", "default", "housing", "loan", "contact", "month",
    "day_of_week", "poutcome", "education", "job", "marital",
]

# Assertion messages for columns that must not contain unmapped values
_UNMAPPED_MESSAGES: Dict[str, str] = {
    "y":           "Unexpected values in target 'y'.",
    "default":     "Unmapped values in 'default'.",
    "housing":     "Unmapped values in 'housing'.",
    "loan":        "Unmapped values in 'loan'.",
    "contact":     "Unmapped contact values.",
    "month":       "Unmapped month values.",
    "day_of_week": "Unmapped day_of_week values.",
    "poutcome":    "Unmapped poutcome values.",
    "education":   "Unmapped education values.",
    "marital":     "Unmapped marital values.",
}

# Columns that leak post-contact information — excluded by default
_LEAKY_FEATURES: List[str] = ["duration", "pdays", "poutcome"]

# Reference grouping of all features (used by downstream callers)
_FEATURE_GROUPS: Dict[str, List[str]] = {
    "target":           ["y"],
    "binary":           ["default", "housing", "loan"],
    "ordinal":          ["education", "poutcome", "contact", "month", "day_of_week"],
    "nominal":          ["job", "marital"],
    "numeric_campaign": ["age", "duration", "campaign", "pdays", "previous"],
    "numeric_economic": [
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed",
    ],
}


def _read_sql(name: str) -> str:
    return (_SQL_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# BankPreprocessor
# ---------------------------------------------------------------------------

class BankPreprocessor:
    """
    Sklearn-style fit/transform preprocessor for the UCI Bank Marketing dataset.

    Parameters
    ----------
    drop_leaky : bool, default True
        Drop features that are unavailable at prediction time
        (``duration``, ``pdays``, ``poutcome``).
    verbose : bool, default True
        Print a step-by-step encoding summary during transform.
    db_path : str or Path, default data/db/bank_marketing.db
        SQLite database containing raw_bank. Used when fit/transform
        are called without a DataFrame.
    """

    # Expose constants so callers can inspect without instantiating
    LEAKY_FEATURES:  List[str]              = _LEAKY_FEATURES
    FEATURE_GROUPS:  Dict[str, List[str]]   = _FEATURE_GROUPS

    def __init__(
        self,
        drop_leaky: bool = True,
        verbose: bool = True,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self.drop_leaky = drop_leaky
        self.verbose    = verbose
        self.db_path    = Path(db_path)
        self._job_map:  Dict[str, int] | None = None  # built from training data
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame | None = None) -> "BankPreprocessor":
        """
        Fit on training data.

        Currently learns: job category → integer mapping (sorted, stable).

        Parameters
        ----------
        df : raw training DataFrame, or None to read raw_bank from db_path
        """
        with self._open_raw_bank(df) as conn:
            rows = conn.execute(_read_sql("fit_job_map.sql")).fetchall()

        self._job_map   = {row[0]: i for i, row in enumerate(rows)}
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Apply all encodings to a (possibly held-out) dataset.

        Parameters
        ----------
        df : raw DataFrame, or None to read raw_bank from db_path

        Raises
        ------
        RuntimeError  if called before ``fit``.
        AssertionError on unexpected values or residual nulls.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        log = self._log  # shorthand

        log("\n" + "=" * 70)
        log("🔧  PREPROCESSING: Bank Marketing Dataset")
        log("=" * 70)

        # ── 1–9. Categorical encodings (SQLite, from raw_bank) ────────
        with self._open_raw_bank(df) as conn:
            conn.execute(
                "CREATE TEMP TABLE job_map (job TEXT PRIMARY KEY, code INTEGER NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO job_map (job, code) VALUES (?, ?)",
                list(self._job_map.items()),
            )
            out = pd.read_sql_query(_read_sql("transform.sql"), conn)

        if df is not None:
            out.index = df.index

        for col in _ENCODED_COLUMNS:
            if col in _UNMAPPED_MESSAGES:
                assert out[col].notna().all(), _UNMAPPED_MESSAGES[col]
            out[col] = out[col].astype("int8")

        log(f"✅  y  →  {out['y'].value_counts().to_dict()}")
        log("✅  binary [default, housing, loan]  (unknown=-1)")
        log("✅  contact")
        log(f"✅  month  →  {sorted(out['month'].unique())}")
        log("✅  day_of_week")
        log("✅  poutcome")
        log("✅  education  (ordinal 0-6, unknown=-1)")
        log(f"✅  job  →  {len(self._job_map)} categories  (unseen → -1)")
        log("✅  marital")

        # ── 10. Economic features (already numeric) ───────────────────
        for col in _FEATURE_GROUPS["numeric_economic"]:
            out[col] = out[col].astype("float64")
        log(f"✅  economic features  →  float64")

        # ── 11. Campaign features (already numeric) ───────────────────
        log(f"✅  campaign features  →  unchanged numeric")

        # ── 12. Drop leaky features ───────────────────────────────────
        if self.drop_leaky:
            to_drop = [c for c in _LEAKY_FEATURES if c in out.columns]
            if to_drop:
                out.drop(columns=to_drop, inplace=True)
                log(f"⚠️   dropped leaky features: {to_drop}")

        # ── Final validation ──────────────────────────────────────────
        non_numeric = out.select_dtypes(exclude=["number"]).columns.tolist()
        assert not non_numeric, f"Non-numeric columns remain: {non_numeric}"
        assert out.isnull().sum().sum() == 0, "Null values detected after transform."

        log(f"\n✅  df_processed ready — shape: {out.shape}")
        log(f"    memory: {out.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

        return out

    # ------------------------------------------------------------------
    # fit_transform  (sklearn convention)
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Fit on the dataset and return its transformation."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return a copy of the feature-group reference dict."""
        return _FEATURE_GROUPS.copy()

    @property
    def job_map(self) -> Dict[str, int] | None:
        """Learned job → integer mapping (None if not yet fitted)."""
        return self._job_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _open_raw_bank(self, df: pd.DataFrame | None) -> Iterator[sqlite3.Connection]:
        """Yield a connection exposing raw_bank: the database file, or df staged in memory."""
        if df is None:
            if not raw_bank_exists(self.db_path):
                raise FileNotFoundError(
                    f"raw_bank not found in {self.db_path}. Run build_database() first."
                )
            conn = sqlite3.connect(self.db_path)
        else:
            conn = sqlite3.connect(":memory:")
            df.to_sql(RAW_TABLE, conn, index=False)
        try:
            yield conn
        finally:
            conn.close()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Standalone inspection helper  (notebook / EDA use)
# ---------------------------------------------------------------------------

def inspect_raw_categoricals(df: pd.DataFrame) -> None:
    """
    Print value-count tables for every object-dtype column.

    Intended for exploratory use before fitting a preprocessor.

    Parameters
    ----------
    df : raw DataFrame
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("📌  RAW CATEGORICAL COLUMNS")
    print("-" * 60)
    for col in cat_cols:
        print(f"\n🔎  {col}")
        print(df[col].astype(str).value_counts(dropna=False).sort_index())