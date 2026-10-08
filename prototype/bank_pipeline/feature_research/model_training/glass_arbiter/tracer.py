"""
feature_research/model_training/glass_arbiter/tracer.py
=======================================================
Agreement trace for the three stages: is their disagreement structured
(signal the arbiter can use) or noise? Works on the cross-fitted decisions
and per-row thresholds from cross_fit_arbiter, so every row is out-of-fold.

State encoding (per stage, per row)
-----------------------------------
00 Silent     score < threshold − margin
10 Candidate  threshold − margin ≤ score < threshold
11 Confirmed  score ≥ threshold

Public API
----------
bitmask_states(P, row_thresholds, margin)   → DataFrame of state labels
run_bitmask_trace(model_preds, P, row_thresholds, y, ...) → dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3

from feature_research.model_training.glass_arbiter.arbiter import MODELS
from feature_research.model_training.shared.report import show_table, title

_LABELS = tuple(m.upper() for m in MODELS)


def bitmask_states(P: pd.DataFrame, row_thresholds: pd.DataFrame, margin: float = 0.07) -> pd.DataFrame:
    hi = row_thresholds[list(MODELS)].to_numpy(dtype=float)
    lo = np.maximum(0.01, hi - margin)
    X = P[list(MODELS)].to_numpy(dtype=float)
    labels = np.where(X >= hi, "11 Confirmed", np.where(X >= lo, "10 Candidate", "00 Silent"))
    return pd.DataFrame(labels, index=P.index, columns=[f"{m.upper()} state" for m in MODELS])


def _group_table(keys: pd.DataFrame, y: pd.Series, top_n: Optional[int] = None) -> pd.DataFrame:
    """Count, share and target rate per unique key combination, largest first."""
    df = keys.assign(_y=y.to_numpy())
    out = (
        df.groupby(list(keys.columns), sort=False)["_y"]
        .agg(count="size", target_rate="mean")
        .sort_values("count", ascending=False, kind="mergesort")
    )
    out.insert(1, "share", out["count"] / len(df))
    return out.head(top_n) if top_n else out


def _verdict(agreement: float) -> str:
    if agreement > 0.85:
        return "⚠️ HIGH AGREEMENT — diversity tuning needed"
    if agreement > 0.75:
        return "🟡 MODERATE AGREEMENT — arbiter viable, diversity tuning will help"
    return "✅ HEALTHY DISAGREEMENT — proceed with the arbiter"


def _venn(correct: pd.DataFrame, catches: pd.DataFrame, save_path: Optional[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, frame, name in ((axes[0], correct, "All rows\n(correct decisions)"),
                            (axes[1], catches, "Positives only\n(caught targets)")):
        venn3([set(np.flatnonzero(frame[m].to_numpy())) for m in MODELS], set_labels=_LABELS, ax=ax)
        ax.set_title(name, fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_bitmask_trace(
    model_preds: pd.DataFrame,
    P: pd.DataFrame,
    row_thresholds: pd.DataFrame,
    y: pd.Series,
    *,
    margin: float = 0.07,
    save_dir: Optional[Path] = None,
    top_n: int = 15,
) -> dict:
    """
    model_preds     cross-fitted 0/1 per stage (ArbiterResult.model_preds)
    P               stage OOF scores
    row_thresholds  threshold applied to each row (ArbiterResult.row_thresholds)
    y               training labels
    save_dir        folder for the Venn PNG; None → shown inline only
    """
    for frame in (model_preds, P, row_thresholds):
        if not frame.index.equals(y.index):
            raise ValueError("trace inputs are not aligned with y")

    pos = (y == 1).to_numpy()
    correct = model_preds[list(MODELS)].eq(y, axis=0)
    catches = correct & pos[:, None]
    all_agree = correct.all(axis=1) | ~correct.any(axis=1)
    all_correct, all_wrong = correct.all(axis=1), ~correct.any(axis=1)
    alone = {m: correct[m] & ~correct.drop(columns=m).any(axis=1) for m in MODELS}

    marks = pd.DataFrame(np.where(correct, "✓", "✗"), index=correct.index, columns=list(_LABELS))
    patterns = _group_table(marks, y)
    states = bitmask_states(P, row_thresholds, margin)
    state_table = _group_table(states, y, top_n)

    unique_targets = {m: catches[m] & ~catches.drop(columns=m).any(axis=1) for m in MODELS}
    pairwise = {f"{a}_{b}": float((model_preds[a] != model_preds[b]).mean())
                for a, b in (("lr", "rf"), ("lr", "ebm"), ("rf", "ebm"))}

    n_pos = int(pos.sum())
    title("GLASS Arbiter · agreement trace (cross-fitted, all training rows)")
    print(f"Agreement  all 3 agree {all_agree.mean():.1%} (all correct {all_correct.mean():.1%} · "
          f"all wrong {all_wrong.mean():.1%}) · any disagreement {1 - all_agree.mean():.1%} · target 65–75%")
    print("Pairwise   " + " · ".join(f"{k.replace('_', ' vs ').upper()} {v:.1%}" for k, v in pairwise.items())
          + " (decisions differ)")
    print(f"Catches    " + " · ".join(f"{m.upper()} {catches[m].sum() / n_pos:.1%}" for m in MODELS)
          + f" of {n_pos:,} positives · all 3 {catches.all(axis=1).sum() / n_pos:.1%} · "
            f"none (hard floor) {(~catches.any(axis=1) & pos).sum() / n_pos:.1%}")
    print("Unique     " + " · ".join(
        f"{m.upper()} {int(u.sum()):,} targets ({u.sum() / n_pos:.1%})" for m, u in unique_targets.items())
        + " caught by one stage only")
    print("Alone      " + " · ".join(
        f"{m.upper()} only correct {int(a.sum()):,} (target rate {y[a].mean():.1%})" if a.any()
        else f"{m.upper()} only correct 0" for m, a in alone.items()))
    print(f"Verdict    {_verdict(all_agree.mean())}")

    print("\nCorrectness patterns")
    show_table(patterns.round(4))
    print(f"\nScore states (margin {margin}) — top {top_n}")
    show_table(state_table.round(4))

    venn_path = None
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        venn_path = Path(save_dir) / "venn_diagram_trace.png"
    _venn(correct, catches, venn_path)
    if venn_path is not None:
        print(f"Saved      {venn_path}")

    return {
        "correct": correct,
        "catches": catches,
        "patterns": patterns,
        "states": states,
        "state_table": state_table,
        "unique_targets": {m: int(u.sum()) for m, u in unique_targets.items()},
        "pairwise_disagreement": pairwise,
        "agreement": float(all_agree.mean()),
        "verdict": _verdict(all_agree.mean()),
        "venn_path": venn_path,
    }