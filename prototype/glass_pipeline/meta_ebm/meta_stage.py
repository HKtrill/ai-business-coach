"""
glass_pipeline.meta_ebm.meta_stage
=====================================
Stage 4 orchestrator — Meta-EBM weighted confidence arbiter.

Invoked from glass_cascade Cell 21:
    meta_artifact, meta_path = train_meta_stage(
        GLOBAL_SPLIT,
        lr_path=lr_calibrated_path,
        glass_path=glass_path,
        ebm_path=ebm_path,
    )

Pipeline
--------
1. Load stage artifacts + assemble arbiter inputs
   1b. Recompute arbiter thresholds via find_recall_threshold(0.70)
       Artifact thresholds are logged only — not used downstream.
2. Calibration + hybrid trust weights (Brier + accuracy)
3. Disagreement analysis (train set)
4. Arbiter threshold tuning (train set, F2 objective)
5. Evaluate arbiter on test set (no-abstain + with-abstain)
6. Save canonical meta artifact

Threshold strategy
------------------
Artifact thresholds (LR=0.10, EBM=0.58) are on incomparable scales.
Recompute at target_recall=0.70 on training probabilities so the
confidence comparisons operate on a shared reference — matching the
original exploratory arbiter convention (LR≈0.44, EBM≈0.08).
GLASS-BRW uses a fixed threshold of 0.5 (routing model).

Artifact thresholds are stored under 'artifact_thresholds' for
standalone eval; arbiter thresholds are stored under 'thresholds'.
"""

import numpy as np
from datetime import datetime

from .loader      import load_stage_outputs
from .arbiter     import meta_arbiter
from .calibration import compute_calibration
from .weighting   import compute_hybrid_weights
from .evaluation  import analyze_disagreements, compute_metrics, evaluate_with_abstention
from .tuning      import tune_arbiter_threshold
from .artifacts   import save_meta_ebm

_DEFAULT_LR_PATH    = r"prototype\glass_pipeline\models\lr\lr_calibrated_20260519_135147.joblib"
_DEFAULT_GLASS_PATH = r"prototype\glass_pipeline\models\glass_brw\glass_brw_20260519_143450.joblib"
_DEFAULT_EBM_PATH   = r"prototype\glass_pipeline\models\ebm\ebm_stage3_20260519_194440.joblib"

_GLASS_ARBITER_THRESH = 0.5


def find_recall_threshold(
    y_true:        np.ndarray,
    y_prob:        np.ndarray,
    target_recall: float = 0.70,
) -> float:
    """
    Lowest threshold achieving at least target_recall on y_prob.
    Falls back to the minimum probability (recall = 1.0) if none qualify.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    pos_mask   = y_true == 1
    n_pos      = pos_mask.sum()

    if n_pos == 0:
        return 0.5

    for t in thresholds:
        if ((y_prob >= t) & pos_mask).sum() / n_pos >= target_recall:
            return float(t)

    return float(thresholds[-1])


def train_meta_stage(
    GLOBAL_SPLIT:  dict,
    lr_path:       str   = _DEFAULT_LR_PATH,
    glass_path:    str   = _DEFAULT_GLASS_PATH,
    ebm_path:      str   = _DEFAULT_EBM_PATH,
    target_recall: float = 0.70,
) -> tuple[dict, str]:
    """
    Train Stage 4 Meta-EBM weighted confidence arbiter.

    Parameters
    ----------
    GLOBAL_SPLIT   : dict — keys X_train, X_test, y_train, y_test
    lr_path        : path to lr_calibrated_*.joblib
    glass_path     : path to glass_brw_*.joblib
    ebm_path       : path to ebm_stage3_*.joblib
    target_recall  : recall target for arbiter threshold recomputation (default 0.70)

    Returns
    -------
    artifact : dict — canonical Stage 4 payload
    path     : str  — path to saved .joblib file
    """
    print("\n" + "=" * 80)
    print("🎯 META-EBM — MODULAR ARBITER")
    print("=" * 80)

    y_train = np.array(GLOBAL_SPLIT['y_train'])
    y_test  = np.array(GLOBAL_SPLIT['y_test'])

    # ==========================================================
    # 1. LOAD STAGE OUTPUTS
    # ==========================================================
    print("\n📂 1. Loading stage outputs...")
    data = load_stage_outputs(lr_path, glass_path, ebm_path, y_train, y_test)

    pass1_train = data['glass_decisions_train'] == "pass1"
    pass2_train = data['glass_decisions_train'] == "pass2"
    pass1_test  = data['glass_decisions_test']  == "pass1"
    pass2_test  = data['glass_decisions_test']  == "pass2"
    glass_covered_train = pass1_train | pass2_train
    glass_covered_test  = pass1_test  | pass2_test

    print(f"   Artifact thresholds (standalone eval only):")
    print(f"     LR  = {data['lr_thresh']:.4f}")
    print(f"     EBM = {data['ebm_thresh']:.4f}")

    # ==========================================================
    # 1b. RECOMPUTE ARBITER THRESHOLDS (recall-targeted)
    # ==========================================================
    print(f"\n   Recomputing arbiter thresholds "
          f"(find_recall_threshold, target={target_recall:.0%})...")

    lr_arb_thresh    = find_recall_threshold(y_train, data['lr_prob_train'],  target_recall)
    ebm_arb_thresh   = find_recall_threshold(y_train, data['ebm_prob_train'], target_recall)
    glass_arb_thresh = _GLASS_ARBITER_THRESH

    print(f"   Arbiter thresholds (used everywhere below):")
    print(f"     LR    = {lr_arb_thresh:.4f}")
    print(f"     EBM   = {ebm_arb_thresh:.4f}")
    print(f"     GLASS = {glass_arb_thresh:.4f}  (fixed)")
    print(f"   Train samples:     {len(y_train):,}")
    print(f"   Test samples:      {len(y_test):,}")
    print(f"   GLASS train cover: {glass_covered_train.mean():.1%}")
    print(f"   GLASS test  cover: {glass_covered_test.mean():.1%}")

    # ==========================================================
    # 2. CALIBRATION + TRUST WEIGHTS
    # ==========================================================
    print("\n📊 2. Computing calibration + trust weights...")

    lr_cal    = compute_calibration(y_train, data['lr_prob_train'])
    ebm_cal   = compute_calibration(y_train, data['ebm_prob_train'])
    glass_cal = compute_calibration(
        y_train[glass_covered_train],
        data['glass_prob_train'][glass_covered_train],
    )

    print(f"   LR    Brier={lr_cal['brier']:.4f}  ECE={lr_cal['ece']:.4f}")
    print(f"   EBM   Brier={ebm_cal['brier']:.4f}  ECE={ebm_cal['ece']:.4f}")
    print(f"   GLASS Brier={glass_cal['brier']:.4f}  ECE={glass_cal['ece']:.4f}")

    glass_pred_train = (data['glass_decisions_train'] == "pass2").astype(int)

    weights = compute_hybrid_weights(
        lr_cal, ebm_cal, glass_cal,
        y_train,
        data['lr_prob_train'],
        data['ebm_prob_train'],
        glass_pred_train,
        lr_arb_thresh,
        ebm_arb_thresh,
        alpha=0.5,
    )
    MODEL_WEIGHTS = {k: weights[k] for k in ['lr', 'ebm', 'glass']}
    print(f"\n   Hybrid weights → "
          f"LR={MODEL_WEIGHTS['lr']:.3f}  "
          f"EBM={MODEL_WEIGHTS['ebm']:.3f}  "
          f"GLASS={MODEL_WEIGHTS['glass']:.3f}")

    # ==========================================================
    # 3. DISAGREEMENT ANALYSIS (TRAIN)
    # ==========================================================
    print("\n📊 3. Disagreement analysis (train set)...")

    lr_pred_train  = (data['lr_prob_train']  >= lr_arb_thresh).astype(int)
    ebm_pred_train = (data['ebm_prob_train'] >= ebm_arb_thresh).astype(int)

    disagreement_report = analyze_disagreements(
        y_true=y_train,
        lr_pred=lr_pred_train,
        ebm_pred=ebm_pred_train,
        glass_pred=glass_pred_train,
        glass_covered_mask=glass_covered_train,
    )
    for k, v in disagreement_report.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"   {k}: {v}")

    # ==========================================================
    # 4. ARBITER THRESHOLD TUNING (TRAIN)
    # ==========================================================
    print("\n🔧 4. Tuning min_weighted_confidence (F2 objective)...")

    best_cfg = tune_arbiter_threshold(
        meta_arbiter,
        probs=(data['lr_prob_train'], data['ebm_prob_train'], data['glass_prob_train']),
        masks=(pass1_train, pass2_train),
        y_true=y_train,
        lr_thresh=lr_arb_thresh,
        ebm_thresh=ebm_arb_thresh,
        weights=MODEL_WEIGHTS,
    )

    if best_cfg is None:
        print("   ⚠️  No config met coverage floor — using default")
        best_cfg = {'min_weighted_confidence': 0.07, 'train_f2': None, 'train_coverage': None}

    MIN_WEIGHTED_CONF = best_cfg['min_weighted_confidence']
    print(f"   Selected → min_weighted_confidence={MIN_WEIGHTED_CONF:.2f}  "
          f"train_f2={best_cfg['train_f2']:.4f}  "
          f"train_coverage={best_cfg['train_coverage']:.1%}")

    # ==========================================================
    # 5. FINAL EVALUATION (TEST)
    # ==========================================================
    print("\n📊 5. Evaluating Meta-EBM (test set)...")

    pred_na, prob_na, _ = meta_arbiter(
        data['lr_prob_test'],
        data['ebm_prob_test'],
        data['glass_prob_test'],
        pass1_test, pass2_test,
        lr_arb_thresh,
        ebm_arb_thresh,
        MODEL_WEIGHTS,
        allow_abstain=False,
    )
    metrics_na = compute_metrics(y_test, pred_na, prob_na)

    pred_a, prob_a, explain_a = meta_arbiter(
        data['lr_prob_test'],
        data['ebm_prob_test'],
        data['glass_prob_test'],
        pass1_test, pass2_test,
        lr_arb_thresh,
        ebm_arb_thresh,
        MODEL_WEIGHTS,
        allow_abstain=True,
        min_weighted_confidence=MIN_WEIGHTED_CONF,
    )
    eval_a = evaluate_with_abstention(y_test, pred_a, prob_a)

    print("\n   No-abstention metrics:")
    for k, v in metrics_na.items():
        print(f"      {k:<12} {v:.4f}")
    print(f"\n   With-abstention  (coverage={eval_a['coverage']:.1%}):")
    if eval_a['metrics']:
        for k, v in eval_a['metrics'].items():
            print(f"      {k:<12} {v:.4f}")

    # ==========================================================
    # 6. SAVE ARTIFACT
    # ==========================================================
    print("\n💾 6. Saving Meta-EBM artifact...")

    artifact = {
        'weights':     MODEL_WEIGHTS,
        'calibration': {'lr': lr_cal, 'ebm': ebm_cal, 'glass': glass_cal},
        'disagreement': disagreement_report,

        'thresholds': {
            'lr':                      lr_arb_thresh,
            'ebm':                     ebm_arb_thresh,
            'glass':                   glass_arb_thresh,
            'target_recall':           target_recall,
            'min_weighted_confidence': MIN_WEIGHTED_CONF,
        },
        'artifact_thresholds': {
            'lr':  data['lr_thresh'],
            'ebm': data['ebm_thresh'],
        },

        'no_abstain':   metrics_na,
        'with_abstain': eval_a,
        'explanations': explain_a,

        'train_probs': {
            'lr':    data['lr_prob_train'],
            'ebm':   data['ebm_prob_train'],
            'glass': data['glass_prob_train'],
        },
        'test_probs': {
            'lr':    data['lr_prob_test'],
            'ebm':   data['ebm_prob_test'],
            'glass': data['glass_prob_test'],
            'meta':  prob_na,
        },
        'test_preds': {
            'lr':           (data['lr_prob_test']  >= lr_arb_thresh).astype(int),
            'ebm':          (data['ebm_prob_test'] >= ebm_arb_thresh).astype(int),
            'glass':        (data['glass_decisions_test'] == "pass2").astype(int),
            'no_abstain':   pred_na,
            'with_abstain': pred_a,
        },
        'glass_decisions_test': data['glass_decisions_test'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = save_meta_ebm(artifact)
    print(f"   ✅ Saved → {path}")
    print("\n🎉 META-EBM COMPLETE")
    print("=" * 80)

    return artifact, path