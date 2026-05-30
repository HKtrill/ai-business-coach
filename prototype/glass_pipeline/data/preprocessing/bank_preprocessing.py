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

Usage
-----
    preprocessor = BankPreprocessor(drop_leaky=True)
    df_processed  = preprocessor.fit_transform(df_raw)

Author: Glass Pipeline Team
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Encoding maps  (class-level constants — never mutated)
# ---------------------------------------------------------------------------

_BINARY_MAP: Dict[str, int] = {"no": 0, "yes": 1, "unknown": -1}
_CONTACT_MAP: Dict[str, int] = {"cellular": 0, "telephone": 1}
_MONTH_MAP: Dict[str, int] = {
    "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,
    "may": 5,  "jun": 6,  "jul": 7,  "aug": 8,
    "sep": 9,  "oct": 10, "nov": 11, "dec": 12,
}
_DAY_MAP: Dict[str, int] = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4,
}
_POUTCOME_MAP: Dict[str, int] = {
    "nonexistent": 0,  # never contacted before
    "failure":     1,  # previous campaign failed
    "success":     2,  # previous campaign succeeded
}
_EDUCATION_MAP: Dict[str, int] = {
    "illiterate":         0,
    "basic.4y":           1,
    "basic.6y":           2,
    "basic.9y":           3,
    "high.school":        4,
    "professional.course": 5,
    "university.degree":  6,
    "unknown":            -1,
}
_MARITAL_MAP: Dict[str, int] = {
    "divorced": 0, "married": 1, "single": 2, "unknown": -1,
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
    """

    # Expose constants so callers can inspect without instantiating
    LEAKY_FEATURES:  List[str]              = _LEAKY_FEATURES
    FEATURE_GROUPS:  Dict[str, List[str]]   = _FEATURE_GROUPS

    def __init__(self, drop_leaky: bool = True, verbose: bool = True) -> None:
        self.drop_leaky = drop_leaky
        self.verbose    = verbose
        self._job_map:  Dict[str, int] | None = None  # built from training data
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "BankPreprocessor":
        """
        Fit on training data.

        Currently learns: job category → integer mapping (sorted, stable).

        Parameters
        ----------
        df : raw training DataFrame (must contain 'job' column)
        """
        job_categories  = sorted(df["job"].dropna().unique())
        self._job_map   = {cat: i for i, cat in enumerate(job_categories)}
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all encodings to a (possibly held-out) DataFrame.

        Raises
        ------
        RuntimeError  if called before ``fit``.
        AssertionError on unexpected values or residual nulls.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        out = df.copy()
        log = self._log  # shorthand

        log("\n" + "=" * 70)
        log("🔧  PREPROCESSING: Bank Marketing Dataset")
        log("=" * 70)

        # ── 1. Target ──────────────────────────────────────────────────
        out["y"] = out["y"].map({"yes": 1, "no": 0}).astype("int8")
        assert out["y"].notna().all(), "Unexpected values in target 'y'."
        log(f"✅  y  →  {out['y'].value_counts().to_dict()}")

        # ── 2. Binary columns (unknown → -1) ──────────────────────────
        for col in ["default", "housing", "loan"]:
            out[col] = out[col].map(_BINARY_MAP).astype("int8")
            assert out[col].notna().all(), f"Unmapped values in '{col}'."
        log("✅  binary [default, housing, loan]  (unknown=-1)")

        # ── 3. Contact ────────────────────────────────────────────────
        out["contact"] = out["contact"].map(_CONTACT_MAP).astype("int8")
        assert out["contact"].notna().all(), "Unmapped contact values."
        log("✅  contact")

        # ── 4. Month (1–12) ───────────────────────────────────────────
        out["month"] = out["month"].map(_MONTH_MAP).astype("int8")
        assert out["month"].notna().all(), "Unmapped month values."
        log(f"✅  month  →  {sorted(out['month'].unique())}")

        # ── 5. Day of week (0–4) ──────────────────────────────────────
        out["day_of_week"] = out["day_of_week"].map(_DAY_MAP).astype("int8")
        assert out["day_of_week"].notna().all(), "Unmapped day_of_week values."
        log("✅  day_of_week")

        # ── 6. Poutcome ───────────────────────────────────────────────
        out["poutcome"] = out["poutcome"].map(_POUTCOME_MAP).astype("int8")
        assert out["poutcome"].notna().all(), "Unmapped poutcome values."
        log("✅  poutcome")

        # ── 7. Education (ordinal, unknown → -1) ──────────────────────
        out["education"] = out["education"].map(_EDUCATION_MAP).astype("int8")
        assert out["education"].notna().all(), "Unmapped education values."
        log("✅  education  (ordinal 0-6, unknown=-1)")

        # ── 8. Job (label-encoded from training vocab) ─────────────────
        out["job"] = out["job"].map(self._job_map).fillna(-1).astype("int8")
        log(f"✅  job  →  {len(self._job_map)} categories  (unseen → -1)")

        # ── 9. Marital (unknown → -1) ─────────────────────────────────
        out["marital"] = out["marital"].map(_MARITAL_MAP).astype("int8")
        assert out["marital"].notna().all(), "Unmapped marital values."
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

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return its transformation."""
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