"""
glass_pipeline.meta_ebm.tracer
================================
Bitmask agreement analysis and Venn diagram trace for the Glass Cascade.

Ports Cell 22 (LR + EBM + GLASS-BRW) into a clean module.
All inputs are read from meta_artifact (produced by train_meta_stage)
and GLOBAL_SPLIT — no notebook-scope dependencies.

Public API
----------
run_cascade_trace(GLOBAL_SPLIT, meta_artifact, figures_dir) -> dict

Cell 22 replacement in glass_cascade:
    from meta_ebm.tracer import run_cascade_trace
    trace = run_cascade_trace(GLOBAL_SPLIT, meta_artifact, figures_dir="research_logs/figures")
"""

import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cascade_trace(
    GLOBAL_SPLIT:   dict,
    meta_artifact:  dict,
    figures_dir:    str  = "research_logs/figures",
    verbose:        bool = True,
) -> dict:
    """
    Full bitmask agreement and Venn trace for the three-model cascade.

    Parameters
    ----------
    GLOBAL_SPLIT   : cascade split dict — y_test read from here
    meta_artifact  : output of train_meta_stage() — contains test_probs,
                     test_preds, thresholds, glass_decisions_test
    figures_dir    : directory to save Venn diagram PNG

    Returns
    -------
    dict:
        bitmask_counts   : Counter of correctness patterns
        agreement        : dict of agreement/disagreement scalars
        gold_samples     : dict of unique-winner counts
        catch_stats      : positive-class catch/miss counts
        venn_path        : path to saved Venn diagram PNG
        verdict          : verdict string
    """
    if verbose:
        print("=" * 80)
        print("🔬 VENN DIAGRAM TRACE — CASCADE AGREEMENT ANALYSIS")
        print("=" * 80)
        print("   LR + EBM + Glass-BRW — real model agreement on test set")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    y_test     = np.array(GLOBAL_SPLIT['y_test'])
    lr_pred    = meta_artifact['test_preds']['lr']
    ebm_pred   = meta_artifact['test_preds']['ebm']
    glass_pred = meta_artifact['test_preds']['glass']

    lr_correct    = (lr_pred    == y_test).astype(int)
    ebm_correct   = (ebm_pred   == y_test).astype(int)
    glass_correct = (glass_pred == y_test).astype(int)

    # ------------------------------------------------------------------
    # 1. Bitmask distribution
    # ------------------------------------------------------------------
    def _label(l, e, g):
        return f"LR={'✓' if l else '✗'} EBM={'✓' if e else '✗'} GLASS={'✓' if g else '✗'}"

    bitmask_counts = Counter(
        _label(l, e, g)
        for l, e, g in zip(lr_correct, ebm_correct, glass_correct)
    )

    if verbose:
        total = len(y_test)
        print(f"\n{'─' * 80}")
        print("📊 BITMASK DISTRIBUTION (all samples)")
        print(f"{'─' * 80}")
        print(f"   {'Pattern':<35} {'Count':>8} {'%':>8}  {'Class Balance'}")
        print(f"   {'-' * 70}")
        for pattern, count in sorted(bitmask_counts.items(), key=lambda x: -x[1]):
            pct  = count / total * 100
            mask = np.array([
                _label(l, e, g) == pattern
                for l, e, g in zip(lr_correct, ebm_correct, glass_correct)
            ])
            pos_rate = y_test[mask].mean()
            bar = "█" * int(pct / 2)
            print(f"   {pattern:<35} {count:>8,} {pct:>7.1f}%  pos={pos_rate:.1%}  {bar}")

    # ------------------------------------------------------------------
    # 2. Agreement summary
    # ------------------------------------------------------------------
    all_agree   = (lr_correct == ebm_correct) & (ebm_correct == glass_correct)
    all_correct = (lr_correct == 1) & (ebm_correct == 1) & (glass_correct == 1)
    all_wrong   = (lr_correct == 0) & (ebm_correct == 0) & (glass_correct == 0)

    agreement = {
        'all_agree_rate':    float(all_agree.mean()),
        'all_correct_rate':  float(all_correct.mean()),
        'all_wrong_rate':    float(all_wrong.mean()),
        'any_disagree_rate': float((~all_agree).mean()),
        'lr_ebm_disagree':   float((lr_pred  != ebm_pred).mean()),
        'lr_glass_disagree': float((lr_pred  != glass_pred).mean()),
        'ebm_glass_disagree':float((ebm_pred != glass_pred).mean()),
    }

    if verbose:
        print(f"\n{'─' * 80}")
        print("📈 AGREEMENT SUMMARY")
        print(f"{'─' * 80}")
        print(f"   All 3 agree      : {all_agree.mean():.1%}  ({all_agree.sum():,} samples)")
        print(f"     → All correct  : {all_correct.mean():.1%}  ({all_correct.sum():,} samples)")
        print(f"     → All wrong    : {all_wrong.mean():.1%}  ({all_wrong.sum():,} samples)")
        print(f"   Any disagreement : {(~all_agree).mean():.1%}  ({(~all_agree).sum():,} samples)")
        print(f"\n   Pairwise disagreement:")
        print(f"   LR  vs EBM  : {agreement['lr_ebm_disagree']:.1%}")
        print(f"   LR  vs GLASS: {agreement['lr_glass_disagree']:.1%}")
        print(f"   EBM vs GLASS: {agreement['ebm_glass_disagree']:.1%}")

    # ------------------------------------------------------------------
    # 3. Gold samples (unique winners)
    # ------------------------------------------------------------------
    gold_samples = {}
    if verbose:
        print(f"\n{'─' * 80}")
        print("🥇 GOLD SAMPLES — Unique Winners")
        print(f"{'─' * 80}")

    for name, correct, others in [
        ("LR only correct",    lr_correct,    [ebm_correct,  glass_correct]),
        ("EBM only correct",   ebm_correct,   [lr_correct,   glass_correct]),
        ("GLASS only correct", glass_correct, [lr_correct,   ebm_correct]),
    ]:
        mask     = (correct == 1) & np.array([o == 0 for o in others]).all(axis=0)
        pos_rate = float(y_test[mask].mean()) if mask.sum() > 0 else 0.0
        gold_samples[name] = {'count': int(mask.sum()), 'pos_rate': pos_rate}

        if verbose:
            if mask.sum() == 0:
                print(f"   {name:<22}: 0 samples")
            else:
                print(
                    f"   {name:<22}: {mask.sum():,} ({mask.mean():.1%})  "
                    f"target rate={pos_rate:.1%}  "
                    f"{'🔥 HIGH SIGNAL' if pos_rate > 0.3 else '⚪ low signal'}"
                )

    # ------------------------------------------------------------------
    # 4. Target class analysis
    # ------------------------------------------------------------------
    pos_mask      = y_test == 1
    lr_catches    = (lr_correct    == 1) & pos_mask
    ebm_catches   = (ebm_correct   == 1) & pos_mask
    glass_catches = (glass_correct == 1) & pos_mask
    all_catch     = lr_catches & ebm_catches & glass_catches
    none_catch    = pos_mask & ~lr_catches & ~ebm_catches & ~glass_catches

    catch_stats = {
        'n_pos':          int(pos_mask.sum()),
        'lr_catch':       int(lr_catches.sum()),
        'ebm_catch':      int(ebm_catches.sum()),
        'glass_catch':    int(glass_catches.sum()),
        'all_catch':      int(all_catch.sum()),
        'none_catch':     int(none_catch.sum()),
        'hard_floor_pct': float(none_catch.sum() / pos_mask.sum()),
    }

    if verbose:
        print(f"\n{'─' * 80}")
        print("🎯 TARGET CLASS ANALYSIS (positive samples only)")
        print(f"{'─' * 80}")
        n_pos = catch_stats['n_pos']
        print(f"   Total positive samples in test: {n_pos:,}\n")
        for name, catches in [("LR", lr_catches), ("EBM", ebm_catches), ("GLASS", glass_catches)]:
            print(f"   {name:<6} catches: {catches.sum():,} / {n_pos:,} ({catches.sum()/n_pos:.1%})")

        for name, catches, others in [
            ("LR unique targets",    lr_catches,    [ebm_catches,   glass_catches]),
            ("EBM unique targets",   ebm_catches,   [lr_catches,    glass_catches]),
            ("GLASS unique targets", glass_catches, [lr_catches,    ebm_catches]),
        ]:
            unique = catches & ~others[0] & ~others[1]
            pct    = unique.sum() / n_pos * 100
            print(
                f"   {name:<24}: {unique.sum():,} ({pct:.1f}% of targets) "
                f"← {'✅ contributing' if unique.sum() > 0 else '❌ redundant'}"
            )

        print(
            f"\n   All 3 catch (redundant): {all_catch.sum():,} "
            f"({all_catch.sum()/n_pos:.1%} of targets)"
        )
        print(
            f"   None catch (hard floor): {none_catch.sum():,} "
            f"({none_catch.sum()/n_pos:.1%} of targets) ← ceiling"
        )

    # ------------------------------------------------------------------
    # 5. Venn diagram — save to file
    # ------------------------------------------------------------------
    os.makedirs(figures_dir, exist_ok=True)
    venn_path = os.path.join(figures_dir, "venn_cascade_trace.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Venn Diagram Trace — LR + EBM + Glass-BRW\n(UCI Bank Marketing — Real Cascade)",
        fontsize=13, fontweight='bold',
    )

    # Panel 1 — all samples
    plt.sca(axes[0])
    venn3(
        [set(np.where(lr_correct)[0]),
         set(np.where(ebm_correct)[0]),
         set(np.where(glass_correct)[0])],
        set_labels=('LR', 'EBM', 'GLASS'),
        ax=axes[0],
    )
    axes[0].set_title("All Samples\n(correct predictions overlap)", fontsize=11)

    # Panel 2 — positive class only
    plt.sca(axes[1])
    venn3(
        [set(np.where(lr_catches)[0]),
         set(np.where(ebm_catches)[0]),
         set(np.where(glass_catches)[0])],
        set_labels=('LR', 'EBM', 'GLASS'),
        ax=axes[1],
    )
    axes[1].set_title("Target Class Only\n(correctly caught positives overlap)", fontsize=11)

    plt.tight_layout()
    plt.savefig(venn_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"\n   ✅ Venn diagram saved → {venn_path}")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    agreement_pct = all_agree.mean() * 100
    unique_lr    = gold_samples["LR only correct"]["count"]
    unique_ebm   = gold_samples["EBM only correct"]["count"]
    unique_glass = gold_samples["GLASS only correct"]["count"]

    if agreement_pct > 85:
        verdict = "⚠️  HIGH AGREEMENT — diversity tuning needed"
    elif agreement_pct > 75:
        verdict = "🟡 MODERATE AGREEMENT — Bitmask Arbiter viable, diversity tuning will help"
    else:
        verdict = "✅ HEALTHY DISAGREEMENT — Proceed with Bitmask Arbiter"

    if verbose:
        print(f"\n{'=' * 80}")
        print("📋 BITMASK ARBITER FEASIBILITY VERDICT — REAL CASCADE")
        print(f"{'=' * 80}")
        print(f"\n   Agreement rate   : {agreement_pct:.1f}%  (target: 65–75%)")
        print(f"   Unique LR wins   : {unique_lr:,}")
        print(f"   Unique EBM wins  : {unique_ebm:,}")
        print(f"   Unique GLASS wins: {unique_glass:,}")
        print(f"   Hard floor (none): {none_catch.sum():,} targets no model catches")
        print(f"\n   {verdict}")
        print(f"\n{'=' * 80}")
        print("🎉 CASCADE VENN TRACE COMPLETE")
        print(f"{'=' * 80}")

    return {
        'bitmask_counts': bitmask_counts,
        'agreement':      agreement,
        'gold_samples':   gold_samples,
        'catch_stats':    catch_stats,
        'venn_path':      venn_path,
        'verdict':        verdict,
    }