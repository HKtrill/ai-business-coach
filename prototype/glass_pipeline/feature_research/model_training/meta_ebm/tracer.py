"""
model_training/meta_ebm/tracer.py
------------------------------------
Bitmask agreement analysis and Venn diagram trace for the meta-arbiter.

Diagnoses whether model disagreement is structured (exploitable signal)
or noise. Run after every retuning cycle to track diversity changes.

Public API
----------
run_bitmask_trace(lr_pred, rf_pred, ebm_pred,
                  lr_pte, rf_pte, ebm_pte,
                  lr_t, rf_t, ebm_t,
                  y_test, figures_dir, ...) -> dict
"""

import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTNESS & AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_correctness_masks(
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    ebm_pred: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-model binary correctness arrays on the test set."""
    return (
        (lr_pred  == y_test).astype(int),
        (rf_pred  == y_test).astype(int),
        (ebm_pred == y_test).astype(int),
    )


def compute_agreement_stats(
    lr_c: np.ndarray,
    rf_c: np.ndarray,
    ebm_c: np.ndarray,
) -> dict:
    """
    Compute agreement and disagreement masks.

    Returns
    -------
    dict: all_agree, all_correct, all_wrong, any_disagree  (boolean arrays)
    """
    all_agree   = (lr_c == rf_c) & (rf_c == ebm_c)
    all_correct = (lr_c == 1) & (rf_c == 1) & (ebm_c == 1)
    all_wrong   = (lr_c == 0) & (rf_c == 0) & (ebm_c == 0)
    return {
        "all_agree":    all_agree.astype(bool),
        "all_correct":  all_correct.astype(bool),
        "all_wrong":    all_wrong.astype(bool),
        "any_disagree": (~all_agree).astype(bool),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CATCH / MISS ANALYSIS (positive class only)
# ══════════════════════════════════════════════════════════════════════════════

def compute_catch_stats(
    lr_c: np.ndarray,
    rf_c: np.ndarray,
    ebm_c: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Positive-class catch/miss masks.

    Returns
    -------
    dict: pos_mask, lr_catches, rf_catches, ebm_catches,
          all_catch, none_catch (hard floor),
          lr_unique, rf_unique, ebm_unique (gold samples)
    """
    pos   = y_test == 1
    lr_c  = lr_c.astype(bool)  & pos
    rf_c  = rf_c.astype(bool)  & pos
    ebm_c = ebm_c.astype(bool) & pos
    return {
        "pos_mask":    pos,
        "lr_catches":  lr_c,
        "rf_catches":  rf_c,
        "ebm_catches": ebm_c,
        "all_catch":   lr_c & rf_c & ebm_c,
        "none_catch":  pos & ~lr_c & ~rf_c & ~ebm_c,   # hard floor
        "lr_unique":   lr_c & ~rf_c & ~ebm_c,           # LR alone correct
        "rf_unique":   rf_c & ~lr_c & ~ebm_c,           # RF alone correct
        "ebm_unique":  ebm_c & ~lr_c & ~rf_c,           # EBM alone correct
    }


# ══════════════════════════════════════════════════════════════════════════════
# BITMASK STATE ENCODING  (2-bit per model)
# ══════════════════════════════════════════════════════════════════════════════
# Bit1 = triggered at low threshold (candidate zone)
# Bit0 = triggered at high threshold (confirmed zone)
# States: 00=Silent, 10=Candidate, 11=Confirmed

_STATE_LABELS = {(0, 0): "00 Silent", (1, 0): "10 Candidate", (1, 1): "11 Confirmed"}


def _get_state(prob: float, t_high: float, t_low: float) -> tuple:
    return (int(prob >= t_low), int(prob >= t_high))


def compute_bitmask_states(
    lr_pte: np.ndarray,
    rf_pte: np.ndarray,
    ebm_pte: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    margin: float = 0.07,
) -> tuple:
    """
    Compute 2-bit bitmask states for all test samples.

    Parameters
    ----------
    margin : gap below the decision threshold that defines the 'candidate' zone

    Returns
    -------
    lr_states, rf_states, ebm_states : lists of (bit1, bit0) tuples
    thresholds : dict(lr=(t_high, t_low), rf=..., ebm=...)
    """
    thresholds = {
        "lr":  (lr_t,  max(0.01, lr_t  - margin)),
        "rf":  (rf_t,  max(0.01, rf_t  - margin)),
        "ebm": (ebm_t, max(0.01, ebm_t - margin)),
    }
    lr_states  = [_get_state(p, *thresholds["lr"])  for p in lr_pte]
    rf_states  = [_get_state(p, *thresholds["rf"])  for p in rf_pte]
    ebm_states = [_get_state(p, *thresholds["ebm"]) for p in ebm_pte]
    return lr_states, rf_states, ebm_states, thresholds


# ══════════════════════════════════════════════════════════════════════════════
# BITMASK DISTRIBUTION PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def _bitmask_label(lr: int, rf: int, ebm: int) -> str:
    return f"LR={'✓' if lr else '✗'} RF={'✓' if rf else '✗'} EBM={'✓' if ebm else '✗'}"


def print_bitmask_distribution(
    lr_c: np.ndarray,
    rf_c: np.ndarray,
    ebm_c: np.ndarray,
    y_test: np.ndarray,
) -> None:
    total = len(y_test)
    counts = Counter(
        _bitmask_label(l, r, e) for l, r, e in zip(lr_c, rf_c, ebm_c)
    )
    print(f"\n   {'Pattern':<30} {'Count':>8} {'%':>8}  Class Balance")
    print(f"   {'-' * 65}")
    for pattern, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct  = count / total * 100
        mask = np.array([
            _bitmask_label(l, r, e) == pattern
            for l, r, e in zip(lr_c, rf_c, ebm_c)
        ])
        pos_rate = y_test[mask].mean()
        bar = "█" * int(pct / 2)
        print(f"   {pattern:<30} {count:>8,} {pct:>7.1f}%  pos={pos_rate:.1%}  {bar}")


def print_state_table(
    lr_states: list,
    rf_states: list,
    ebm_states: list,
    y_test: np.ndarray,
    top_n: int = 15,
) -> None:
    total = len(y_test)
    combos = Counter(
        (
            _STATE_LABELS[tuple(l)],
            _STATE_LABELS[tuple(r)],
            _STATE_LABELS[tuple(e)],
        )
        for l, r, e in zip(lr_states, rf_states, ebm_states)
    )
    print(
        f"\n   {'LR State':<15} {'RF State':<15} {'EBM State':<15} "
        f"{'Count':>8} {'%':>7}  Target Rate"
    )
    print(f"   {'-' * 70}")
    for (ls, rs, es), cnt in sorted(combos.items(), key=lambda x: -x[1])[:top_n]:
        pct  = cnt / total * 100
        smask = np.array([
            _STATE_LABELS[tuple(l)] == ls
            and _STATE_LABELS[tuple(r)] == rs
            and _STATE_LABELS[tuple(e)] == es
            for l, r, e in zip(lr_states, rf_states, ebm_states)
        ])
        tr   = y_test[smask].mean() if smask.sum() > 0 else 0.0
        flag = (
            "🎯" if ls == rs == es == "11 Confirmed"
            else "⚡" if tr > 0.5
            else "🔥" if tr > 0.3
            else ""
        )
        print(
            f"   {ls:<15} {rs:<15} {es:<15} "
            f"{cnt:>8,} {pct:>6.1f}%  {tr:.1%} {flag}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# VENN DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def plot_venn(
    lr_c: np.ndarray,
    rf_c: np.ndarray,
    ebm_c: np.ndarray,
    catch_stats: dict,
    save_path: str,
    model_labels: tuple = ("LR", "RF", "EBM"),
    title_suffix: str = "",
) -> str:
    """
    Save two-panel Venn diagram (all samples + positives only).

    Returns
    -------
    save_path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12, fontweight="bold")

    # Panel 1 — all samples
    plt.sca(axes[0])
    venn3(
        [set(np.where(lr_c)[0]), set(np.where(rf_c)[0]), set(np.where(ebm_c)[0])],
        set_labels=model_labels,
        ax=axes[0],
    )
    axes[0].set_title("All Samples\n(correct predictions overlap)", fontsize=11)

    # Panel 2 — positive class only
    plt.sca(axes[1])
    venn3(
        [
            set(np.where(catch_stats["lr_catches"])[0]),
            set(np.where(catch_stats["rf_catches"])[0]),
            set(np.where(catch_stats["ebm_catches"])[0]),
        ],
        set_labels=model_labels,
        ax=axes[1],
    )
    axes[1].set_title("Target Class Only\n(correctly caught positives overlap)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════════════

def get_verdict(agreement_pct: float) -> str:
    if agreement_pct > 85:
        return (
            "⚠️  HIGH AGREEMENT — models too correlated.\n"
            "   → Retune with stronger diversity pressure + feature subsetting\n"
            "   → Re-run tracer after retuning to check improvement"
        )
    elif agreement_pct > 75:
        return (
            "🟡 MODERATE AGREEMENT — some diversity present.\n"
            "   → Bitmask Arbiter viable; diversity tuning will improve further"
        )
    else:
        return (
            "✅ HEALTHY DISAGREEMENT — Bitmask Arbiter is well motivated.\n"
            "   → Proceed to Stage 4 with agreement/disagreement features"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRACE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_bitmask_trace(
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    ebm_pred: np.ndarray,
    lr_pte: np.ndarray,
    rf_pte: np.ndarray,
    ebm_pte: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    y_test: np.ndarray,
    figures_dir: str,
    model_labels: tuple = ("LR", "RF", "EBM"),
    venn_title: str = "",
    verbose: bool = True,
) -> dict:
    """
    Full bitmask trace: agreement → gold samples → target analysis →
    state encoding → Venn diagram → verdict.

    Parameters
    ----------
    lr_pred, rf_pred, ebm_pred : binary prediction arrays on test set
    lr_pte, rf_pte, ebm_pte   : probability arrays on test set
    lr_t, rf_t, ebm_t         : decision thresholds (from recall targeting)
    y_test                     : true labels
    figures_dir                : directory to save Venn diagram PNG
    model_labels               : display names for the three models
    venn_title                 : optional suptitle for Venn figure

    Returns
    -------
    dict:
        correctness    : (lr_c, rf_c, ebm_c) — binary correctness arrays
        agree_stats    : agreement/disagreement masks
        catch_stats    : positive-class catch/miss masks
        bitmask_states : (lr_states, rf_states, ebm_states)
        thresholds     : dict of (t_high, t_low) per model
        venn_path      : path to saved Venn diagram PNG
        verdict        : verdict string
    """
    lr_c, rf_c, ebm_c = build_correctness_masks(lr_pred, rf_pred, ebm_pred, y_test)
    agree_stats  = compute_agreement_stats(lr_c, rf_c, ebm_c)
    catch_stats  = compute_catch_stats(lr_c, rf_c, ebm_c, y_test)
    lr_st, rf_st, ebm_st, thresholds = compute_bitmask_states(
        lr_pte, rf_pte, ebm_pte, lr_t, rf_t, ebm_t
    )

    if verbose:
        n_pos  = catch_stats["pos_mask"].sum()
        total  = len(y_test)
        ag     = agree_stats["all_agree"]

        print(f"\n{'─' * 80}")
        print("📊 BITMASK DISTRIBUTION")
        print(f"{'─' * 80}")
        print_bitmask_distribution(lr_c, rf_c, ebm_c, y_test)

        print(f"\n{'─' * 80}")
        print("📈 AGREEMENT SUMMARY")
        print(f"{'─' * 80}")
        print(f"   All 3 agree      : {ag.mean():.1%}  ({ag.sum():,} samples)")
        print(f"     → All correct  : {agree_stats['all_correct'].mean():.1%}")
        print(f"     → All wrong    : {agree_stats['all_wrong'].mean():.1%}")
        print(f"   Any disagreement : {agree_stats['any_disagree'].mean():.1%}")

        print(f"\n{'─' * 80}")
        print("🥇 GOLD SAMPLES — Unique Winners")
        print(f"{'─' * 80}")
        for name, key in [("LR only correct", "lr_unique"),
                           ("RF only correct", "rf_unique"),
                           ("EBM only correct", "ebm_unique")]:
            mask = catch_stats[key]
            if mask.sum() == 0:
                print(f"   {name:<22}: 0 samples")
                continue
            pos_rate = y_test[mask].mean()
            print(
                f"   {name:<22}: {mask.sum():,} ({mask.mean():.1%})  "
                f"target rate={pos_rate:.1%}  "
                f"{'🔥 HIGH SIGNAL' if pos_rate > 0.3 else '⚪ low signal'}"
            )

        print(f"\n{'─' * 80}")
        print("🎯 TARGET CLASS ANALYSIS")
        print(f"{'─' * 80}")
        print(f"   Total positive samples: {n_pos:,}")
        for name, key in [("LR", "lr_catches"), ("RF", "rf_catches"), ("EBM", "ebm_catches")]:
            c = catch_stats[key]
            print(f"   {name} catches : {c.sum():,} / {n_pos:,} ({c.sum() / n_pos:.1%})")
        ac = catch_stats["all_catch"]
        nc = catch_stats["none_catch"]
        print(f"\n   All 3 catch (redundant): {ac.sum():,} ({ac.sum() / n_pos:.1%} of targets)")
        print(f"   Hard floor (none catch): {nc.sum():,} ({nc.sum() / n_pos:.1%} of targets)")

        print(f"\n{'─' * 80}")
        print("🔢 BITMASK STATE ENCODING")
        print(f"{'─' * 80}")
        print_state_table(lr_st, rf_st, ebm_st, y_test)

    # Venn diagram
    os.makedirs(figures_dir, exist_ok=True)
    venn_path = os.path.join(figures_dir, "venn_diagram_trace.png")
    plot_venn(lr_c, rf_c, ebm_c, catch_stats, venn_path, model_labels, venn_title)
    if verbose:
        print(f"\n   ✅ Venn diagram saved → {venn_path}")

    # Verdict
    verdict = get_verdict(agree_stats["all_agree"].mean() * 100)
    if verbose:
        print(f"\n{'=' * 80}")
        print("📋 FEASIBILITY VERDICT")
        print(f"{'=' * 80}")
        print(f"\n   {verdict}")

    return {
        "correctness":    (lr_c, rf_c, ebm_c),
        "agree_stats":    agree_stats,
        "catch_stats":    catch_stats,
        "bitmask_states": (lr_st, rf_st, ebm_st),
        "thresholds":     thresholds,
        "venn_path":      venn_path,
        "verdict":        verdict,
    }