"""
model_training.rf.binning
=========================
Lift-driven binary encoding of continuous RF features for the RF stage and
the GLASS Router.

Eight source features are expanded into 29 binary columns: seven groups of
mutually exclusive bins plus one passthrough flag. Thresholds are fixed
constants (no fitting), derived from the Cell 15A lift analysis and locked in
BINNING_STRATEGY. Bins are (lower, upper]; None means unbounded.

The lift figures in the BINNING_STRATEGY comments are from the original
derivation. On the current train-only frame NSD (nsd_elevated) and ECI
(eci_warm, eci_hot) have drifted; thresholds are re-derived at PR 39.

Design rationale
----------------
The GLASS Router is a two-pass symbolic rule router whose rules are
conjunctions of binary conditions, so it needs binary inputs. Multi-bin
encoding (rather than a single above/below flag) gives those rules and the RF
distinct levels of each feature to combine (e.g. nsd_hot AND jed_cold).

Public API
----------
bin_features(df)                        → df with the 29 bin columns added (pure)
BinaryFeaturePipeline(base_factory)     feature pipeline + binning, for per-fold refits
create_binary_features(df, target_col)  → (df_rf_binary, RF_FEATURES_BINARY), with lift summary
bin_lift_table(df_binned, target_col)   → per-bin rows, share, conversion rate, lift
validate_binary_features(df, target_col, verbose)
RF_FEATURES_BINARY                      canonical 29-column list
BINNING_STRATEGY                        threshold definitions (source of truth)
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

# ── Canonical feature list (output of bin_features) ───────────────────────────
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

# ── Mutual exclusivity groups (every row must fall in exactly one bin) ────────
_FEATURE_GROUPS: dict[str, list[str]] = {
    "NSD":      ["nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot"],
    "JED":      ["jed_cold", "jed_transition", "jed_warm", "jed_hot"],
    "CCI":      ["cci_low", "cci_dead_mid", "cci_hot1", "cci_valley", "cci_hot2"],
    "ECI":      ["eci_cold", "eci_mid", "eci_warm", "eci_hot"],
    "DOW":      ["dow_cold", "dow_low", "dow_mid", "dow_hot"],
    "BEHAV":    ["behav_cold", "behav_baseline", "behav_warm", "behav_hot"],
    "CAMPAIGN": ["campaign_fresh", "campaign_moderate", "campaign_heavy"],
}

# ── Binning strategy — locked thresholds ──────────────────────────────────────
# Format: group → {"source": column, "bins": [(bin_name, lower_exclusive, upper_inclusive)]}
#         or {"source": column, "passthrough": output_name} for already-binary sources.
# None bounds mean -inf / +inf. Lift figures are from the original derivation.
BINNING_STRATEGY: dict = {
    # ── neighborhood_subscription_density  [0.048, 0.350] ─────────────────────
    # Dead ≤0.052 (0.27–0.49x), transition to 0.19 (~0.9x),
    # elevated 0.19–0.23 (1.04–1.43x), hot >0.23 (4.08x lift)
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

_SOURCES: list[str] = [g["source"] for g in BINNING_STRATEGY.values()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_group(df: pd.DataFrame, group_def: dict) -> None:
    """Add one group's bin columns to df in place (int8, (lower, upper])."""
    if "passthrough" in group_def:
        df[group_def["passthrough"]] = df[group_def["source"]].astype("int8")
        return

    col = df[group_def["source"]]
    for bin_name, lo, hi in group_def["bins"]:
        if lo is None:
            mask = col <= hi
        elif hi is None:
            mask = col > lo
        else:
            mask = (col > lo) & (col <= hi)
        df[bin_name] = mask.astype("int8")


# ── Public API ────────────────────────────────────────────────────────────────

def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with the 29 RF_FEATURES_BINARY columns added.

    Pure transform: fixed thresholds, no target, no printing. Source columns
    are kept; slice with RF_FEATURES_BINARY for the model-ready matrix.
    Raises KeyError if any source feature is missing.
    """
    missing = [s for s in _SOURCES if s not in df.columns]
    if missing:
        raise KeyError(f"bin_features: missing source features {missing}")
    out = df.copy()
    for group_def in BINNING_STRATEGY.values():
        _apply_group(out, group_def)
    return out


class BinaryFeaturePipeline:
    """
    Feature pipeline followed by the locked RF binning.

    base_factory : zero-arg callable returning the feature pipeline,
                   e.g. make_feature_pipeline (Cell 11).

    fit_transform(X, y) fits a fresh base pipeline and bins its output;
    transform(X) applies the fitted base pipeline and bins its output.
    Used as a per-fold feature factory:
        partial(BinaryFeaturePipeline, make_feature_pipeline)
    """

    def __init__(self, base_factory: Callable[[], Any]):
        self.base_factory = base_factory

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.base_ = self.base_factory()
        return bin_features(self.base_.fit_transform(X, y))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return bin_features(self.base_.transform(X))


def _group_columns(group_def: dict) -> list[str]:
    return [b[0] for b in group_def["bins"]] if "bins" in group_def else [group_def["passthrough"]]


def bin_lift_table(df_binned: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Per-bin lift on df_binned.

    Returns DataFrame [group, bin, rows, share, conv_rate, lift], one row per
    RF_FEATURES_BINARY column, where lift = bin conversion rate / overall rate.
    """
    overall = df_binned[target_col].mean()
    records = []
    for group, group_def in BINNING_STRATEGY.items():
        for col in _group_columns(group_def):
            on = df_binned[col] == 1
            conv = float(df_binned.loc[on, target_col].mean()) if on.any() else 0.0
            records.append({
                "group": group, "bin": col, "rows": int(on.sum()), "share": float(on.mean()),
                "conv_rate": conv, "lift": conv / overall if overall > 0 else 0.0,
            })
    return pd.DataFrame(records)


def create_binary_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    bin_features(df) plus a one-line-per-group lift summary
    (full detail: bin_lift_table).

    Parameters
    ----------
    df         : df_engineered (source features + target_col, training rows).
    target_col : TARGET_COL — used only for the lift summary.

    Returns
    -------
    df_rf_binary       : copy of df with the 29 bin columns added.
    RF_FEATURES_BINARY : canonical 29-column list.
    """
    df_binned = bin_features(df)
    lift = bin_lift_table(df_binned, target_col)

    print(f"\nRF BINNING — {len(_SOURCES)} sources → {len(RF_FEATURES_BINARY)} binary columns "
          f"· base rate {df_binned[target_col].mean():.3f}")
    print("─" * 78)
    print("Lift per bin (share of rows); ▲ > 1.5×, ▼ < 0.7×")
    for group, rows in lift.groupby("group", sort=False):
        parts = [
            f"{r.bin.split('_', 1)[1]} {r.lift:.2f}×"
            f"{'▲' if r.lift > 1.5 else '▼' if r.lift < 0.7 else ''} ({r.share:.0%})"
            for r in rows.itertuples()
        ]
        print(f"  {group:<9} {' · '.join(parts)}")

    return df_binned, RF_FEATURES_BINARY


def validate_binary_features(
    df_binned: pd.DataFrame,
    target_col: str,
    verbose: bool = True,
) -> bool:
    """
    Hard stop on an invalid binary feature space.

    Raises ValueError unless every RF_FEATURES_BINARY column is 0/1 and every
    row falls in exactly one bin of each group. Prints a one-line confirmation;
    verbose=True adds the observed pattern count. target_col is unused (kept
    for call compatibility).

    Returns True when all checks pass.
    """
    X = df_binned[RF_FEATURES_BINARY]
    non_binary = [c for c in RF_FEATURES_BINARY if not X[c].isin([0, 1]).all()]
    if non_binary:
        raise ValueError(f"Non-binary (or NaN) values in {non_binary}.")

    for grp, cols in _FEATURE_GROUPS.items():
        sums = df_binned[cols].sum(axis=1)
        if not (sums == 1).all():
            raise ValueError(
                f"Group '{grp}' has rows where the bin sum ≠ 1 "
                f"(range [{sums.min()}, {sums.max()}]). Check the thresholds in BINNING_STRATEGY."
            )

    msg = (f"Validation ✓ {len(RF_FEATURES_BINARY)} columns are 0/1 · "
           f"{len(_FEATURE_GROUPS)} groups mutually exclusive and complete")
    if verbose:
        msg += f" · {X.drop_duplicates().shape[0]:,} observed patterns"
    print(msg)
    return True