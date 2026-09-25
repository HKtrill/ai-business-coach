"""
model_training.rf.binning
==========================
Lift-driven binary encoding of continuous RF features for GLASS-BRW input.

Thresholds were derived from Cell 13C lift analysis and are locked here as
the permanent research record. Each continuous feature is expanded into
mutually exclusive bins (cold / warm / hot / etc.) that form the 29-column
binary input space for the Random Forest stage.

Design rationale
----------------
GLASS-BRW operates on binary conjunctions — it requires binary inputs.
Multi-bin encoding (rather than simple above/below threshold) lets the RF
learn interactions between levels of the same feature (e.g. nsd_hot AND
jed_cold) without collapsing the signal into a single flag.

Public API
----------
create_binary_features(df, target_col) → (df_rf_binary, RF_FEATURES_BINARY)
validate_binary_features(df_binned, target_col, verbose)
RF_FEATURES_BINARY — canonical 29-feature list
BINNING_STRATEGY   — dict of threshold definitions (source of truth)
"""

from __future__ import annotations

import pandas as pd

# ── Canonical feature list (output of create_binary_features) ─────────────────
RF_FEATURES_BINARY: list[str] = [
    # neighborhood_subscription_density → 4 bins
    "nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot",
    # joint_economic_decay → 4 bins
    "jed_cold", "jed_transition", "jed_warm", "jed_hot",
    # cons.conf.idx → 5 bins (non-monotonic: two hot zones, dead valley)
    "cci_low", "cci_dead_mid", "cci_hot1", "cci_valley", "cci_hot2",
    # economic_curvature_intensity → 4 bins
    "eci_cold", "eci_mid", "eci_warm", "eci_hot",
    # dow_month_encoded → 4 bins (monotonic ramp)
    "dow_cold", "dow_low", "dow_mid", "dow_hot",
    # behavioral_favorability → 4 bins (step function)
    "behav_cold", "behav_baseline", "behav_warm", "behav_hot",
    # campaign → 3 bins (fresh ≤2 is positive signal; decays above)
    "campaign_fresh", "campaign_moderate", "campaign_heavy",
    # cpi_high_cellular → passthrough (already binary)
    "cpi_cellular",
]

# ── Mutual exclusivity groups (used for validation) ───────────────────────────
_FEATURE_GROUPS: dict[str, list[str]] = {
    "NSD":      ["nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot"],
    "JED":      ["jed_cold", "jed_transition", "jed_warm", "jed_hot"],
    "CCI":      ["cci_low", "cci_dead_mid", "cci_hot1", "cci_valley", "cci_hot2"],
    "ECI":      ["eci_cold", "eci_mid", "eci_warm", "eci_hot"],
    "DOW":      ["dow_cold", "dow_low", "dow_mid", "dow_hot"],
    "BEHAV":    ["behav_cold", "behav_baseline", "behav_warm", "behav_hot"],
    "CAMPAIGN": ["campaign_fresh", "campaign_moderate", "campaign_heavy"],
}

# ── Binning strategy — locked thresholds from 13C lift analysis ───────────────
# Format: source feature → list of (bin_name, lower_bound_exclusive, upper_bound_inclusive)
# None bounds mean -inf / +inf.
BINNING_STRATEGY: dict = {
    # ── neighborhood_subscription_density  [0.048, 0.350] ─────────────────────
    # Dead ≤0.052 (0.27–0.49x), transition to 0.19 (~0.9x),
    # warm 0.19–0.23 (1.04–1.43x), hot >0.23 (4.08x lift)
    "NSD": {
        "source": "neighborhood_subscription_density",
        "bins": [
            ("nsd_cold",     None,  0.0524),
            ("nsd_warm",     0.0524, 0.192),
            ("nsd_elevated", 0.192,  0.23),
            ("nsd_hot",      0.23,   None),
        ],
    },
    # ── joint_economic_decay  [0.0, 0.430] ────────────────────────────────────
    # Dead ≤0.00035 (0.27–0.49x), transition to 0.038,
    # warm 0.038–0.086 (1.04–1.43x), hot >0.086 (4.08x)
    # Cuts offset from NSD intentionally — breaks correlation, forces disagreement.
    "JED": {
        "source": "joint_economic_decay",
        "bins": [
            ("jed_cold",       None,    0.000352),
            ("jed_transition", 0.000352, 0.038),
            ("jed_warm",       0.038,    0.0856),
            ("jed_hot",        0.0856,   None),
        ],
    },
    # ── cons.conf.idx  [-50.8, -26.9]  NON-MONOTONIC ──────────────────────────
    # ≤-46.2: modest 1.24x | -46.2 to -41.8: dead (0.38–0.54x)
    # -41.8 to -40.0: HOT1 4.11x | -40.0 to -36.1: valley 0.46–0.64x
    # >-36.1: HOT2 3.78x
    "CCI": {
        "source": "cons.conf.idx",
        "bins": [
            ("cci_low",      None,   -46.2),
            ("cci_dead_mid", -46.2,  -41.8),
            ("cci_hot1",     -41.8,  -40.0),
            ("cci_valley",   -40.0,  -36.1),
            ("cci_hot2",     -36.1,   None),
        ],
    },
    # ── economic_curvature_intensity  [0.008, 0.153] ──────────────────────────
    # Dead ≤0.029 (0.44–0.52x), mid 0.029–0.076 (mixed),
    # warm 0.076–0.095 (1.49–1.67x), hot >0.095 (3.53x)
    "ECI": {
        "source": "economic_curvature_intensity",
        "bins": [
            ("eci_cold", None,   0.0291),
            ("eci_mid",  0.0291, 0.0758),
            ("eci_warm", 0.0758, 0.0948),
            ("eci_hot",  0.0948, None),
        ],
    },
    # ── dow_month_encoded  [0.055, 0.397]  MONOTONIC ──────────────────────────
    # Cold ≤0.065 (0.52–0.56x), low 0.065–0.090 (0.63–0.78x),
    # mid 0.090–0.127 (0.83–1.07x), hot >0.127 (3.55x)
    "DOW": {
        "source": "dow_month_encoded",
        "bins": [
            ("dow_cold", None,   0.0653),
            ("dow_low",  0.0653, 0.0895),
            ("dow_mid",  0.0895, 0.127),
            ("dow_hot",  0.127,  None),
        ],
    },
    # ── behavioral_favorability  [0.0, 1.0]  STEP FUNCTION ────────────────────
    # Cold ≤0.3 (0.36–0.58x), baseline 0.3–0.4 (0.95x),
    # warm 0.4–0.5 (1.20x), hot >0.5 (1.87–3.97x)
    "BEHAV": {
        "source": "behavioral_favorability",
        "bins": [
            ("behav_cold",     None, 0.3),
            ("behav_baseline", 0.3,  0.4),
            ("behav_warm",     0.4,  0.5),
            ("behav_hot",      0.5,  None),
        ],
    },
    # ── campaign  [1, 56] ─────────────────────────────────────────────────────
    # Fresh ≤2 (1.10x — only above-baseline bin), moderate 3–5 (0.77–0.95x),
    # heavy >5 (0.49x — sharply decaying)
    "CAMPAIGN": {
        "source": "campaign",
        "bins": [
            ("campaign_fresh",    None, 2),
            ("campaign_moderate", 2,    5),
            ("campaign_heavy",    5,    None),
        ],
    },
    # ── cpi_high_cellular — already binary, passthrough ───────────────────────
    "CPI": {
        "source": "cpi_high_cellular",
        "passthrough": "cpi_cellular",
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_group(df: pd.DataFrame, group_def: dict) -> pd.DataFrame:
    """Apply one feature group's binning rules to df in-place."""
    if "passthrough" in group_def:
        df[group_def["passthrough"]] = df[group_def["source"]].astype("int8")
        return df

    source = group_def["source"]
    col = df[source]

    for bin_name, lo, hi in group_def["bins"]:
        if lo is None and hi is not None:
            mask = col <= hi
        elif lo is not None and hi is None:
            mask = col > lo
        else:
            mask = (col > lo) & (col <= hi)
        df[bin_name] = mask.astype("int8")

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def create_binary_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform continuous RF features into 29 mutually exclusive binary bins.

    Reads source features from df (must be df_engineered or equivalent),
    applies the locked BINNING_STRATEGY thresholds, and returns an augmented
    copy with binary bin columns appended.

    Does NOT drop the original source columns — caller can slice with
    RF_FEATURES_BINARY to get the model-ready input.

    Parameters
    ----------
    df         : df_engineered (source features must be present).
    target_col : TARGET_COL — used for lift sanity check display only.

    Returns
    -------
    df_rf_binary : pd.DataFrame with binary bin columns appended.
    RF_FEATURES_BINARY : list[str] — canonical 29-bin feature list.
    """
    print(f"\n{'=' * 80}")
    print("🔀 FEATURE BINNING — LIFT-DRIVEN MULTI-BIN ENCODING")
    print(f"{'=' * 80}")
    print("   Converting continuous features → 29 binary bins")
    print(f"{'─' * 80}")

    df_binned = df.copy()

    for group_name, group_def in BINNING_STRATEGY.items():
        source = group_def["source"]
        assert source in df_binned.columns, (
            f"Source feature '{source}' not found in df. "
            f"Ensure df_engineered contains all RF source features."
        )
        _apply_group(df_binned, group_def)

    # ── Lift sanity report ────────────────────────────────────────────────────
    overall_rate = df_binned[target_col].mean()
    print(f"\n   Per-bin lift (base rate: {overall_rate:.4f}):")
    print(f"   {'Bin':25s} {'N':>7s} {'Conv':>8s} {'Lift':>7s}")
    print(f"   {'-' * 25} {'-' * 7} {'-' * 8} {'-' * 7}")
    for feat in RF_FEATURES_BINARY:
        n = int(df_binned[feat].sum())
        conv = (
            df_binned.loc[df_binned[feat] == 1, target_col].mean()
            if n > 0 else 0.0
        )
        lift = conv / overall_rate if overall_rate > 0 else 0.0
        tag = "🔥" if lift > 1.5 else "🔴" if lift < 0.7 else "  "
        print(f"   {feat:25s} {n:>7,} {conv:>8.4f} {lift:>6.2f}x {tag}")

    print(f"\n✅ {len(RF_FEATURES_BINARY)} binary bins ready → df_rf_binary")
    print(f"{'=' * 80}")

    return df_binned, RF_FEATURES_BINARY


def validate_binary_features(
    df_binned: pd.DataFrame,
    target_col: str,
    verbose: bool = True,
) -> bool:
    """
    Validate mutual exclusivity and completeness of binary bin groups.

    For each feature group (NSD, JED, CCI, etc.) checks that every row
    is assigned to exactly one bin (sum == 1). Raises AssertionError on
    first failure.

    Parameters
    ----------
    df_binned  : Output of create_binary_features().
    target_col : Not used directly; kept for signature consistency.
    verbose    : Print per-group results (default True).

    Returns
    -------
    True if all checks pass.
    """
    print(f"\n{'─' * 80}")
    print("🔍 BINARY FEATURE VALIDATION")
    print(f"{'─' * 80}")

    if verbose:
        print(f"\n   Shape: {df_binned[RF_FEATURES_BINARY].shape}")
        all_binary = df_binned[RF_FEATURES_BINARY].isin([0, 1]).all().all()
        n_patterns = df_binned[RF_FEATURES_BINARY].drop_duplicates().shape[0]
        print(f"   All binary: {all_binary}")
        print(f"   Unique patterns: {n_patterns} / {2 ** len(RF_FEATURES_BINARY)} possible")

    print(f"\n   Mutual exclusivity:")
    for grp, cols in _FEATURE_GROUPS.items():
        sums = df_binned[cols].sum(axis=1)
        ok = (sums == 1).all()
        icon = "✅" if ok else "⚠️ "
        if verbose:
            print(f"      {icon} {grp:10s}: {len(cols)} bins, sum range=[{sums.min()}, {sums.max()}]")
        assert ok, (
            f"Group '{grp}' has rows where sum ≠ 1. "
            f"Check binning thresholds in BINNING_STRATEGY."
        )

    if verbose:
        # Class-conditional activation rates
        overall_rate = df_binned[target_col].mean()
        print(f"\n   Class-conditional activation rates (base: {overall_rate:.4f}):")
        print(f"   {'Feature':25s} {'P(hot|y=0)':>12s} {'P(hot|y=1)':>12s} {'Ratio':>8s}")
        print(f"   {'-' * 65}")
        for feat in RF_FEATURES_BINARY:
            rate_neg = df_binned.loc[df_binned[target_col] == 0, feat].mean()
            rate_pos = df_binned.loc[df_binned[target_col] == 1, feat].mean()
            ratio = rate_pos / rate_neg if rate_neg > 0 else float("inf")
            marker = "🔥" if ratio > 3 else "🟢" if ratio > 1.5 else "  "
            print(f"   {feat:25s} {rate_neg:12.4f} {rate_pos:12.4f} {ratio:7.2f}x {marker}")

    print(f"\n✅ All binary feature groups valid")
    return True