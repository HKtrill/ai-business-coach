"""
feature_research/model_training/glass_arbiter/analysis.py
=========================================================
Missed-positive analysis on the cross-fitted stage decisions: which targets
no stage catches (the hard floor), how they differ from targets every stage
catches, and how close the stages came. Optional Excel export.

Public API
----------
build_sample_groups(model_preds, y)              → dict of boolean masks
compute_statistical_profile(caught, floor, ...)  → DataFrame (Mann-Whitney + Cohen's d)
analyze_near_misses(P, row_thresholds, mask, pct) → dict
run_missed_analysis(features, y, model_preds, P, row_thresholds, ...) → (groups, profile_df, near_miss)

Excel sheets (when excel_path is given)
---------------------------------------
summary · hard_floor · lr_only_misses · rf_only_misses · ebm_only_misses ·
statistical_profile · probability_heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from feature_research.model_training.glass_arbiter.arbiter import MODELS
from feature_research.model_training.shared.report import show_table, title

_PROFILE_COLUMNS = ["feature", "caught_mean", "floor_mean", "delta_mean", "cohens_d", "mannwhitney_p",
                    "in_LR", "in_RF", "in_EBM"]


def build_sample_groups(model_preds: pd.DataFrame, y: pd.Series) -> dict:
    """Positive-class masks: pos, all_catch, hard_floor, and each single-stage miss."""
    pos = (y == 1).to_numpy()
    c = {m: (model_preds[m].to_numpy() == 1) & pos for m in MODELS}
    return {
        "pos": pos,
        "all_catch": c["lr"] & c["rf"] & c["ebm"],
        "hard_floor": pos & ~c["lr"] & ~c["rf"] & ~c["ebm"],
        "lr_miss": pos & ~c["lr"] & c["rf"] & c["ebm"],
        "rf_miss": pos & c["lr"] & ~c["rf"] & c["ebm"],
        "ebm_miss": pos & c["lr"] & c["rf"] & ~c["ebm"],
    }


def compute_statistical_profile(
    caught_df: pd.DataFrame,
    floor_df: pd.DataFrame,
    features: list[str],
    lr_features: list[str],
    rf_features: list[str],
    ebm_features: list[str],
) -> pd.DataFrame:
    """Per feature: caught vs hard-floor means/medians, Mann-Whitney p, Cohen's d; sorted by p."""
    rows = []
    for feat in features:
        a = caught_df[feat].to_numpy(dtype=float)
        b = floor_df[feat].to_numpy(dtype=float)
        row = {
            "feature": feat,
            "caught_mean": a.mean() if a.size else np.nan,
            "caught_std": a.std() if a.size else np.nan,
            "caught_median": np.median(a) if a.size else np.nan,
            "floor_mean": b.mean() if b.size else np.nan,
            "floor_std": b.std() if b.size else np.nan,
            "floor_median": np.median(b) if b.size else np.nan,
            "delta_mean": b.mean() - a.mean() if a.size and b.size else np.nan,
            "in_LR": feat in lr_features,
            "in_RF": feat in rf_features,
            "in_EBM": feat in ebm_features,
            "mannwhitney_p": np.nan,
            "cohens_d": np.nan,
        }
        if a.size > 5 and b.size > 5:
            try:
                row["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:  # all values identical
                pass
            pooled = np.sqrt((a.var() + b.var()) / 2)
            row["cohens_d"] = float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mannwhitney_p", kind="mergesort").reset_index(drop=True)


def analyze_near_misses(
    P: pd.DataFrame,
    row_thresholds: pd.DataFrame,
    mask: np.ndarray,
    near_miss_pct: float = 0.85,
) -> dict:
    """Among `mask` rows: near-miss if any stage scored ≥ near_miss_pct × its threshold."""
    X = P[list(MODELS)].to_numpy(dtype=float)[mask]
    T = row_thresholds[list(MODELS)].to_numpy(dtype=float)[mask]
    near = (X >= T * near_miss_pct).any(axis=1)
    return {
        "near_miss_count": int(near.sum()),
        "truly_invisible_count": int((~near).sum()),
        "near_miss_pct_of_floor": float(near.mean()) if near.size else 0.0,
    }


def _export_excel(export_df: pd.DataFrame, groups: dict, profile_df: pd.DataFrame,
                  feature_cols: list[str], path: Path) -> None:
    pred_cols = ["y_true"] + [f"{m}_pred" for m in MODELS] + [f"{m}_prob" for m in MODELS]
    cols = pred_cols + feature_cols
    prob_cols = [f"{m}_prob" for m in MODELS]
    pos = groups["pos"]
    summary = pd.DataFrame({
        "Metric": ["Rows", "Positive (targets)", "Hard floor (none catch)", "Hard floor % of targets",
                   "All 3 catch", "LR-only misses", "RF-only misses", "EBM-only misses"],
        "Value": [len(export_df), int(pos.sum()), int(groups["hard_floor"].sum()),
                  f"{groups['hard_floor'].sum() / pos.sum():.1%}", int(groups["all_catch"].sum()),
                  int(groups["lr_miss"].sum()), int(groups["rf_miss"].sum()), int(groups["ebm_miss"].sum())],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        floor = export_df.loc[groups["hard_floor"], cols].copy()
        floor["max_prob"] = floor[prob_cols].max(axis=1)
        floor["closest_model"] = floor[prob_cols].idxmax(axis=1)
        floor.sort_values("max_prob", ascending=False).to_excel(writer, sheet_name="hard_floor")
        for m in MODELS:
            if groups[f"{m}_miss"].any():
                export_df.loc[groups[f"{m}_miss"], cols].to_excel(writer, sheet_name=f"{m}_only_misses")
        profile_df.to_excel(writer, sheet_name="statistical_profile", index=False)
        heat = export_df.loc[pos, pred_cols].copy()
        heat["n_models_catch"] = heat[[f"{m}_pred" for m in MODELS]].sum(axis=1)
        heat["category"] = heat["n_models_catch"].map({0: "hard_floor", 1: "one_catches", 2: "two_catch", 3: "all_catch"})
        heat.sort_values("n_models_catch", kind="mergesort").to_excel(writer, sheet_name="probability_heatmap")


def run_missed_analysis(
    features: pd.DataFrame,
    y: pd.Series,
    model_preds: pd.DataFrame,
    P: pd.DataFrame,
    row_thresholds: pd.DataFrame,
    lr_features: list[str],
    rf_features: list[str],
    ebm_features: list[str],
    *,
    near_miss_pct: float = 0.85,
    excel_path: Optional[Path] = None,
    top_n: int = 15,
) -> tuple[dict, pd.DataFrame, dict]:
    """
    features        feature frame for the training rows (engineered + RF binary columns)
    model_preds, P, row_thresholds   from cross_fit_arbiter / stage OOF scores
    excel_path      write the workbook here; None → skip
    """
    for frame in (features, model_preds, P, row_thresholds):
        if not frame.index.equals(y.index):
            raise ValueError("analysis inputs are not aligned with y")

    groups = build_sample_groups(model_preds, y)
    profiled = [f for f in dict.fromkeys(list(lr_features) + list(rf_features) + list(ebm_features))
                if f in features.columns]
    profile_df = compute_statistical_profile(
        features.loc[groups["all_catch"], profiled], features.loc[groups["hard_floor"], profiled],
        profiled, lr_features, rf_features, ebm_features,
    )
    near = analyze_near_misses(P, row_thresholds, groups["hard_floor"], near_miss_pct)

    n_pos = int(groups["pos"].sum())
    hf, ac = int(groups["hard_floor"].sum()), int(groups["all_catch"].sum())
    title("GLASS Arbiter · missed positives (cross-fitted)")
    print(f"Positives  {n_pos:,} · hard floor {hf:,} ({hf / n_pos:.1%}) · all 3 catch {ac:,} ({ac / n_pos:.1%})")
    print("           single-stage misses: " + " · ".join(
        f"{m.upper()} {int(groups[f'{m}_miss'].sum()):,}" for m in MODELS))
    print(f"Floor      near-miss (any stage ≥ {near_miss_pct:.0%} of its threshold) {near['near_miss_count']:,} · "
          f"invisible {near['truly_invisible_count']:,}")
    print(f"Profile    {len(profiled)} features · caught (all 3) vs hard floor · top {top_n} by Mann-Whitney p")
    show_table(profile_df[_PROFILE_COLUMNS].head(top_n).set_index("feature").round(4))

    if excel_path is not None:
        export_df = features[profiled].copy()
        export_df.insert(0, "y_true", y.to_numpy())
        for k, m in enumerate(MODELS):
            export_df.insert(1 + k, f"{m}_pred", model_preds[m].to_numpy())
            export_df.insert(1 + len(MODELS) + k, f"{m}_prob", P[m].to_numpy())
        Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
        _export_excel(export_df, groups, profile_df, profiled, Path(excel_path))
        print(f"Saved      {excel_path}")

    return groups, profile_df, near
