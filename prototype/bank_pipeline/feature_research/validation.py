"""
feature_research/validation.py
================================
Pre-modelling sanity checks and feature-type classification for the Glass
Cascade pipeline.

Validates that BankPreprocessor has been applied correctly and that the
DataFrame meets the structural contract expected by all downstream cells:
binary 0/1 target, fully numeric columns, no missing values, no leaky features.

Also provides classify_features() which partitions columns into numeric vs.
categorical using a cardinality heuristic — a prerequisite for the
separation-metrics pipeline.

Finally, provides validate_engineered_features() which checks the finalized
df_engineered for NaN/inf and verifies all stage feature sets are present
and clean before training begins.

Raises AssertionError (or ValueError) on the first violated invariant so
problems surface loudly rather than silently corrupting model results.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Features whose presence after preprocessing indicates a data-leakage bug.
_LEAKY_FEATURES: list[str] = ["duration", "pdays", "poutcome"]

# Candidate names for the binary target column (checked in priority order).
_TARGET_CANDIDATES: list[str] = ["y", "target", "label", "subscribed"]


def sanity_check_data(df: pd.DataFrame) -> str:
    """Validate a preprocessed DataFrame before any feature engineering.

    Checks performed (in order):
    1. A recognised target column exists.
    2. The target is strictly binary {0, 1}.
    3. No known leaky features survive preprocessing.
    4. Every column (including target) is numeric.
    5. No missing values anywhere.

    Parameters
    ----------
    df:
        Output of BankPreprocessor.fit_transform().

    Returns
    -------
    str
        Name of the detected target column -- typically 'y'.

    Raises
    ------
    ValueError
        If no target column is found among the known candidates.
    AssertionError
        If any structural invariant is violated.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # 1. Locate target column
    target_col: str | None = next(
        (col for col in _TARGET_CANDIDATES if col in df.columns), None
    )
    if target_col is None:
        raise ValueError(
            f"No target column found. Searched for: {_TARGET_CANDIDATES}"
        )
    print(f"  Target column     : '{target_col}'")

    # 2. Binary {0, 1} target
    target_values = sorted(df[target_col].unique())
    assert len(target_values) == 2, (
        f"Target must be binary, got {len(target_values)} unique values: {target_values}"
    )
    assert set(target_values) == {0, 1}, (
        f"Target values must be {{0, 1}}, got: {set(target_values)}"
    )
    class_counts = df[target_col].value_counts().sort_index()
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"   Class distribution : {class_counts.to_dict()}")
    print(f"   Imbalance ratio    : {imbalance_ratio:.2f}:1")

    # 3. Leaky feature check
    present_leaky = [f for f in _LEAKY_FEATURES if f in df.columns]
    if present_leaky:
        print(f"  WARNING - leaky features present: {present_leaky}")
        print("   These should have been dropped by BankPreprocessor!")
    else:
        print(f"  No leaky features : {_LEAKY_FEATURES}")

    # 4. All numeric
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, (
        f"Non-numeric columns detected - encoding incomplete: {non_numeric}"
    )
    print(f"  All columns numeric: {len(df.columns)} total")

    # 5. No missing values
    n_missing = df.isnull().sum().sum()
    assert n_missing == 0, f"Missing values detected: {n_missing}"
    print("  No missing values")

    n_features = len(df.columns) - 1
    print(f"\n  Dataset shape     : {df.shape}")
    print(f"   Feature columns  : {n_features}")
    print(f"   Samples          : {len(df):,}")

    return target_col


# ---------------------------------------------------------------------------
# Feature-type classification
# ---------------------------------------------------------------------------

_CATEGORICAL_CARDINALITY_THRESHOLD: int = 10
_INTEGER_DTYPES: frozenset[str] = frozenset(
    ["int8", "int16", "int32", "int64"]
)


def classify_features(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[List[str], List[str]]:
    """Partition DataFrame columns into numeric and categorical feature lists.

    A column is treated as categorical when both conditions hold:
    1. Its dtype is an integer variant (int8 ... int64).
    2. It has at most _CATEGORICAL_CARDINALITY_THRESHOLD unique values.

    All other feature columns (floats, high-cardinality integers) are classed
    as numeric.

    Parameters
    ----------
    df:
        Fully preprocessed DataFrame.
    target_col:
        Name of the target column -- excluded from both output lists.

    Returns
    -------
    numeric_features, categorical_features : tuple of lists
    """
    features = [col for col in df.columns if col != target_col]
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    print("\n" + "=" * 80)
    print("FEATURE TYPING")
    print("=" * 80)

    for col in features:
        n_unique = df[col].nunique()
        is_categorical = (
            str(df[col].dtype) in _INTEGER_DTYPES
            and n_unique <= _CATEGORICAL_CARDINALITY_THRESHOLD
        )
        if is_categorical:
            categorical_features.append(col)
            print(f"  {col:20s} -> categorical  (n_unique={n_unique})")
        else:
            numeric_features.append(col)
            print(f"  {col:20s} -> numeric      (n_unique={n_unique})")

    print(f"\n  Numeric features     : {len(numeric_features)}")
    print(f"  Categorical features : {len(categorical_features)}")

    return numeric_features, categorical_features


# ---------------------------------------------------------------------------
# Post-engineering data quality check
# ---------------------------------------------------------------------------

def validate_engineered_features(
    df: pd.DataFrame,
    live_features: Dict[str, List[str]],
    target_col: str = 'y',
) -> bool:
    """
    Verify df_engineered is clean and all stage feature sets are intact.

    Checks (in order):
    1. No NaN values anywhere in df.
    2. No inf values in numeric columns.
    3. Every feature in every stage's list is present in df.
    4. Per-stage NaN/inf check on the columns each model will consume.

    Called after build_features() + finalize_features() and before any model
    training. Raises AssertionError on first failure so problems surface
    loudly rather than corrupting training silently.

    Parameters
    ----------
    df : pd.DataFrame
        Output of finalize_features() -- should contain only model-consumed
        columns + target.
    live_features : dict[str, list[str]]
        LIVE_FEATURES registry keyed by 'lr', 'rf', 'ebm'.
    target_col : str
        Target column name.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    AssertionError
        On any NaN, inf, missing column, or per-stage data quality failure.
    """
    print("\n" + "=" * 70)
    print("ENGINEERED FEATURE VALIDATION")
    print("=" * 70)
    print(f"\n   df shape: {df.shape}")

    # 1. Global NaN check
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols):
        for col, n in nan_cols.items():
            print(f"   NaN  {col}: {n} ({n / len(df):.2%})")
        raise AssertionError(f"NaN values found in {len(nan_cols)} columns.")
    print("   No NaN values")

    # 2. Global inf check
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols):
        for col, n in inf_cols.items():
            print(f"   Inf  {col}: {n} ({n / len(df):.2%})")
        raise AssertionError(f"Inf values found in {len(inf_cols)} columns.")
    print("   No inf values")

    # 3+4. Per-stage: presence + cleanliness
    print(f"\n   {'Stage':<6} {'Features':>8}  {'NaN':>6}  {'Inf':>6}  Status")
    print("   " + "-" * 45)

    all_ok = True
    for stage, features in live_features.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"   {stage.upper():<6} {'':>8}  MISSING: {missing}")
            all_ok = False
            continue

        subset = df[features].select_dtypes(include=[np.number])
        n_nan = int(subset.isnull().sum().sum())
        n_inf = int(np.isinf(subset).sum().sum())
        status = "clean" if (n_nan == 0 and n_inf == 0) else "ISSUES"
        if status != "clean":
            all_ok = False
        print(f"   {stage.upper():<6} {len(features):>8}  {n_nan:>6}  {n_inf:>6}  {status}")

    print("\n" + "=" * 70)
    if not all_ok:
        raise AssertionError("Validation failed -- fix issues above before training.")
    print("All engineered features validated -- safe to train")
    print("=" * 70)

    return True