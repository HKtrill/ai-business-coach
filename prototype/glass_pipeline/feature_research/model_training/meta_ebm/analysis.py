"""
model_training/meta_ebm/analysis.py
--------------------------------------
Missed sample analysis: statistical profiling of hard-floor samples
(targets no model catches) and Excel export for manual inspection.

Public API
----------
run_missed_analysis(
    test_df, y_test,
    lr_pred, rf_pred, ebm_pred,
    lr_pte,  rf_pte,  ebm_pte,
    lr_t,    rf_t,    ebm_t,
    lr_features, rf_features, ebm_features,
    output_path, verbose=True,
) -> (groups, profile_df, near_miss_stats)

Excel sheets produced
---------------------
summary              — high-level counts and thresholds
hard_floor           — target samples NO model catches, sorted by max_prob
lr_only_misses       — targets RF + EBM catch but LR misses
rf_only_misses       — targets LR + EBM catch but RF misses
ebm_only_misses      — targets LR + RF catch but EBM misses
statistical_profile  — feature-level Mann-Whitney + Cohen's d comparison
probability_heatmap  — all positive samples with per-model probs and catch flags
"""

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE GROUP MASKS
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_groups(
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    ebm_pred: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Build positive-class catch/miss masks for all analysis groups.

    Returns
    -------
    dict:
        pos        — all positive samples
        all_catch  — all 3 models catch (redundant)
        hard_floor — no model catches (the ceiling)
        lr_miss    — LR misses, RF + EBM catch
        rf_miss    — RF misses, LR + EBM catch
        ebm_miss   — EBM misses, LR + RF catch
    """
    pos   = y_test == 1
    lr_c  = (lr_pred  == 1) & pos
    rf_c  = (rf_pred  == 1) & pos
    ebm_c = (ebm_pred == 1) & pos
    return {
        "pos":        pos,
        "all_catch":  lr_c & rf_c & ebm_c,
        "hard_floor": pos & ~lr_c & ~rf_c & ~ebm_c,
        "lr_miss":    pos & ~lr_c &  rf_c &  ebm_c,
        "rf_miss":    pos &  lr_c & ~rf_c &  ebm_c,
        "ebm_miss":   pos &  lr_c &  rf_c & ~ebm_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistical_profile(
    caught_df: pd.DataFrame,
    floor_df: pd.DataFrame,
    features: list[str],
    lr_features: list[str],
    rf_features: list[str],
    ebm_features: list[str],
) -> pd.DataFrame:
    """
    Feature-level statistical comparison: caught (all-3-correct) vs hard-floor samples.

    Computes mean/std/median for each group, delta, Mann-Whitney U p-value,
    and Cohen's d effect size. Sorted by p-value ascending (most discriminating first).

    Returns
    -------
    pd.DataFrame sorted by mannwhitney_p
    """
    rows = []
    for feat in features:
        caught_vals = caught_df[feat].values if feat in caught_df.columns else np.array([])
        floor_vals  = floor_df[feat].values  if feat in floor_df.columns  else np.array([])

        row: dict = {
            "feature":       feat,
            "caught_mean":   np.mean(caught_vals)   if len(caught_vals) > 0 else np.nan,
            "caught_std":    np.std(caught_vals)    if len(caught_vals) > 0 else np.nan,
            "caught_median": np.median(caught_vals) if len(caught_vals) > 0 else np.nan,
            "floor_mean":    np.mean(floor_vals)    if len(floor_vals)  > 0 else np.nan,
            "floor_std":     np.std(floor_vals)     if len(floor_vals)  > 0 else np.nan,
            "floor_median":  np.median(floor_vals)  if len(floor_vals)  > 0 else np.nan,
            "delta_mean": (
                np.mean(floor_vals) - np.mean(caught_vals)
                if len(floor_vals) > 0 and len(caught_vals) > 0
                else np.nan
            ),
            "in_LR":  feat in lr_features,
            "in_RF":  feat in rf_features,
            "in_EBM": feat in ebm_features,
        }

        if len(caught_vals) > 5 and len(floor_vals) > 5:
            try:
                _, p_val = stats.mannwhitneyu(
                    caught_vals, floor_vals, alternative="two-sided"
                )
                row["mannwhitney_p"] = float(p_val)
                row["significant"]   = "YES" if p_val < 0.05 else "no"
            except Exception:
                row["mannwhitney_p"] = np.nan
                row["significant"]   = "N/A"

            pooled_std = np.sqrt((np.var(caught_vals) + np.var(floor_vals)) / 2)
            row["cohens_d"] = (
                float((np.mean(floor_vals) - np.mean(caught_vals)) / pooled_std)
                if pooled_std > 0
                else 0.0
            )
        else:
            row["mannwhitney_p"] = np.nan
            row["significant"]   = "N/A"
            row["cohens_d"]      = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mannwhitney_p", ascending=True)


def print_top_discriminating_features(profile_df: pd.DataFrame, top_n: int = 15) -> None:
    print(
        f"\n   {'Feature':<40} {'Caught μ':>10} {'Floor μ':>10} "
        f"{'Δ':>10} {'Cohen d':>10} {'p-val':>10}"
    )
    print(f"   {'-' * 90}")
    for _, row in profile_df.head(top_n).iterrows():
        p = row.get("mannwhitney_p", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(
            f"   {row['feature']:<40} {row['caught_mean']:>10.4f} {row['floor_mean']:>10.4f} "
            f"{row['delta_mean']:>+10.4f} {row.get('cohens_d', 0):>10.3f} "
            f"{p:>9.2e} {sig}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# NEAR-MISS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_near_misses(
    floor_lr_prob: np.ndarray,
    floor_rf_prob: np.ndarray,
    floor_ebm_prob: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    near_miss_pct: float = 0.85,
) -> dict:
    """
    Classify hard-floor samples as near-miss or truly invisible.

    A sample is a near-miss if any model scored >= near_miss_pct of its threshold.
    Truly invisible = no model came close — need new features or models.

    Returns
    -------
    dict: near_miss_count, truly_invisible_count, near_miss_pct_of_floor
    """
    near_any = (
        (floor_lr_prob  >= lr_t  * near_miss_pct)
        | (floor_rf_prob  >= rf_t  * near_miss_pct)
        | (floor_ebm_prob >= ebm_t * near_miss_pct)
    )
    n = len(floor_lr_prob)
    return {
        "near_miss_count":          int(near_any.sum()),
        "truly_invisible_count":    int(n - near_any.sum()),
        "near_miss_pct_of_floor":   float(near_any.mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_to_excel(
    test_df: pd.DataFrame,
    groups: dict,
    profile_df: pd.DataFrame,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    output_path: str,
    feature_cols: list[str],
) -> str:
    """
    Write all analysis sheets to Excel.

    Parameters
    ----------
    test_df     : test-set DataFrame with lr_pred, rf_pred, ebm_pred,
                  lr_prob, rf_prob, ebm_prob, y_true columns already attached
    groups      : output of build_sample_groups()
    profile_df  : output of compute_statistical_profile()
    feature_cols: list of feature column names to include in sample sheets

    Returns
    -------
    output_path
    """
    pred_cols   = ["y_true", "lr_pred", "rf_pred", "ebm_pred",
                   "lr_prob", "rf_prob", "ebm_prob"]
    export_cols = pred_cols + [c for c in feature_cols if c in test_df.columns]
    pos         = groups["pos"]

    # Summary sheet data
    summary = {
        "Metric": [
            "Total test samples",
            "Total positive (targets)",
            "Hard floor (none catch)",
            "Hard floor % of targets",
            "All 3 catch (redundant)",
            "LR only misses",
            "RF only misses",
            "EBM only misses",
            "LR threshold",
            "RF threshold",
            "EBM threshold",
        ],
        "Value": [
            len(test_df),
            int(pos.sum()),
            int(groups["hard_floor"].sum()),
            f"{groups['hard_floor'].sum() / pos.sum():.1%}",
            int(groups["all_catch"].sum()),
            int(groups["lr_miss"].sum()),
            int(groups["rf_miss"].sum()),
            int(groups["ebm_miss"].sum()),
            f"{lr_t:.4f}",
            f"{rf_t:.4f}",
            f"{ebm_t:.4f}",
        ],
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1 — Summary
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)

        # Sheet 2 — Hard floor (sorted by max_prob descending — near-misses first)
        floor_export = test_df[groups["hard_floor"]][export_cols].copy()
        floor_export["max_prob"] = floor_export[
            ["lr_prob", "rf_prob", "ebm_prob"]
        ].max(axis=1)
        floor_export["closest_model"] = floor_export[
            ["lr_prob", "rf_prob", "ebm_prob"]
        ].idxmax(axis=1)
        floor_export.sort_values("max_prob", ascending=False).to_excel(
            writer, sheet_name="hard_floor", index=False
        )

        # Sheets 3-5 — Per-model misses
        for sheet, mask_key in [
            ("lr_only_misses",  "lr_miss"),
            ("rf_only_misses",  "rf_miss"),
            ("ebm_only_misses", "ebm_miss"),
        ]:
            if groups[mask_key].sum() > 0:
                test_df[groups[mask_key]][export_cols].to_excel(
                    writer, sheet_name=sheet, index=False
                )

        # Sheet 6 — Statistical profile
        profile_df.to_excel(writer, sheet_name="statistical_profile", index=False)

        # Sheet 7 — Probability heatmap (all positive samples)
        prob_sheet = test_df[pos][
            ["y_true", "lr_prob", "rf_prob", "ebm_prob",
             "lr_pred", "rf_pred", "ebm_pred"]
        ].copy()
        prob_sheet["n_models_catch"] = (
            prob_sheet[["lr_pred", "rf_pred", "ebm_pred"]]
            .apply(lambda r: (r == 1).sum(), axis=1)
        )
        prob_sheet["category"] = prob_sheet["n_models_catch"].map(
            {0: "hard_floor", 1: "one_catches", 2: "two_catch", 3: "all_catch"}
        )
        prob_sheet.sort_values("n_models_catch", ascending=True).to_excel(
            writer, sheet_name="probability_heatmap", index=False
        )

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_missed_analysis(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    ebm_pred: np.ndarray,
    lr_pte: np.ndarray,
    rf_pte: np.ndarray,
    ebm_pte: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    lr_features: list[str],
    rf_features: list[str],
    ebm_features: list[str],
    output_path: str,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Full missed sample analysis pipeline.

    Attaches prediction columns to test_df, builds sample groups,
    computes statistical profile, near-miss breakdown, and exports to Excel.

    Parameters
    ----------
    test_df : df_engineered.iloc[test_idx].reset_index(drop=True)
              (pure features, no pred columns yet)
    y_test  : true labels for the test split

    Returns
    -------
    groups       : dict of boolean masks (build_sample_groups output)
    profile_df   : feature comparison DataFrame
    near_miss    : dict(near_miss_count, truly_invisible_count, ...)
    """
    # Attach prediction columns
    export_df = test_df.copy().reset_index(drop=True)
    export_df["y_true"]   = y_test
    export_df["lr_pred"]  = lr_pred
    export_df["rf_pred"]  = rf_pred
    export_df["ebm_pred"] = ebm_pred
    export_df["lr_prob"]  = lr_pte
    export_df["rf_prob"]  = rf_pte
    export_df["ebm_prob"] = ebm_pte

    groups = build_sample_groups(lr_pred, rf_pred, ebm_pred, y_test)

    if verbose:
        pos = groups["pos"]
        print(f"   Total positive samples : {pos.sum():,}")
        print(f"   Hard floor (none catch): {groups['hard_floor'].sum():,} "
              f"({groups['hard_floor'].sum() / pos.sum():.1%})")
        print(f"   All 3 catch (redundant): {groups['all_catch'].sum():,} "
              f"({groups['all_catch'].sum() / pos.sum():.1%})")

    # Features to profile
    all_features = sorted(set(lr_features + rf_features + ebm_features))
    all_features = [f for f in all_features if f in export_df.columns]

    caught_df = export_df[groups["all_catch"]].copy()
    floor_df  = export_df[groups["hard_floor"]].copy()

    if verbose:
        print(f"\n   Computing statistical profile over {len(all_features)} features...")
    profile_df = compute_statistical_profile(
        caught_df, floor_df, all_features, lr_features, rf_features, ebm_features
    )

    if verbose:
        print_top_discriminating_features(profile_df)

    # Near-miss analysis
    hf = groups["hard_floor"]
    near_miss = analyze_near_misses(
        lr_pte[hf], rf_pte[hf], ebm_pte[hf], lr_t, rf_t, ebm_t
    )

    if verbose:
        print(f"\n   Hard floor breakdown:")
        print(f"      Near-miss (any model >85% of thresh): {near_miss['near_miss_count']:,} "
              f"— recoverable with tuning")
        print(f"      Truly invisible (all <85%):           {near_miss['truly_invisible_count']:,} "
              f"— need new features or models")

    # Export
    export_to_excel(
        export_df, groups, profile_df,
        lr_t, rf_t, ebm_t,
        output_path, all_features,
    )

    if verbose:
        print(f"\n   ✅ Exported to: {output_path}")
        print(f"   Sheets: summary, hard_floor, lr_only_misses, rf_only_misses, "
              f"ebm_only_misses, statistical_profile, probability_heatmap")

    return groups, profile_df, near_miss