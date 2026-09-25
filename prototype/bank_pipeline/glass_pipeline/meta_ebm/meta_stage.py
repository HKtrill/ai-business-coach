"""
glass_pipeline.meta_ebm.meta_stage
==================================
Stage 4 orchestrator — Meta-EBM weighted-confidence arbiter over
GLASS LR (Stage 1) + GLASS Router (Stage 2) + GLASS EBM (Stage 3).

    meta_artifact, meta_path = train_meta_stage(
        GLOBAL_SPLIT,
        lr_path=lr_calibrated_path,
        glass_path=glass_path,        # GLASS Router artifact
        ebm_path=ebm_path,            # Stage3Artifact
        target_recall=0.70,
    )

Pipeline
--------
1. Load + verify Stage 1–3 inputs (``loader.load_stage_outputs``): split
   fingerprint, row counts, index, labels, fold provenance, OOF honesty,
   router Pass 1 / Pass 2 semantics. Fails loudly on any violation.
   ── everything in 2–5 reads TRAIN-side OOF arrays and y_train only ──
2. Stage 4 thresholds: LR / EBM recomputed by find_recall_threshold at
   ``target_recall``; GLASS fixed at 0.5.
3. Calibration diagnostics + hybrid trust weights (Brier + accuracy).
4. Disagreement analysis.
5. F2 sweep of ``min_weighted_confidence`` (coverage ≥ 50%).
   ── configuration frozen here ──
6. Evaluate once on test: without and with abstention.
7. Save the artifact with full input provenance.

Three kinds of threshold, never interchanged
--------------------------------------------
``artifact_thresholds``  upstream standalone operating points (Stage 1
                         ``stage_output.threshold``; Stage 3 OOF threshold in
                         the consumed probability space). Reported only.
``thresholds``           what the arbiter actually uses: Stage 4's
                         recall-targeted LR / EBM cuts and the fixed GLASS 0.5.
``abstention``           ``min_weighted_confidence`` — the abstain cut on the
                         arbiter's weighted confidence, not a probability cut.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .arbiter import meta_arbiter
from .artifacts import save_meta_ebm
from .calibration import compute_calibration
from .evaluation import analyze_disagreements, compute_metrics, evaluate_with_abstention
from .loader import PASS1, PASS2, load_stage_outputs
from .tuning import tune_arbiter_threshold
from .weighting import compute_hybrid_weights

ARTIFACT_SCHEMA = "meta_ebm/2"

_GLASS_ARBITER_THRESH = 0.5
_MIN_CONF_FALLBACK = 0.07
_MIN_COVERAGE = 0.50


def find_recall_threshold(
    y_true:        np.ndarray,
    y_prob:        np.ndarray,
    target_recall: float = 0.70,
) -> float:
    """
    HIGHEST threshold t (among the observed probability values) at which
    recall(y_prob >= t) >= target_recall — i.e. the tightest cut that still
    meets the recall target. Falls back to the minimum probability
    (recall = 1.0) if none qualify; 0.5 if there are no positives.

    Vectorised; returns exactly what the original descending linear scan did.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    pos    = np.sort(y_prob[y_true == 1])
    n_pos  = len(pos)
    if n_pos == 0:
        return 0.5

    thresholds = np.unique(y_prob)[::-1]                 # descending
    # positives with p >= t, for every candidate t
    n_hit  = n_pos - np.searchsorted(pos, thresholds, side="left")
    ok     = n_hit / n_pos >= target_recall
    if ok.any():
        return float(thresholds[np.argmax(ok)])
    return float(thresholds[-1])


def train_meta_stage(
    GLOBAL_SPLIT:  dict,
    lr_path:       str,
    glass_path:    str,
    ebm_path:      str,
    target_recall: float = 0.70,
    artifact_dir:  str   = "./models/meta_ebm",
    verbose:       bool  = True,
) -> tuple[dict, str]:
    """
    Train the Stage 4 Meta-EBM weighted-confidence arbiter.

    Parameters
    ----------
    GLOBAL_SPLIT  : dict — X_train, X_test, y_train, y_test
    lr_path       : GLASS LR StageOutput (file or dict with ``stage_output``)
    glass_path    : GLASS Router artifact
    ebm_path      : GLASS EBM ``Stage3Artifact``
    target_recall : recall target for the Stage 4 LR / EBM cuts (0.70)
    artifact_dir  : where the Meta-EBM artifact is written

    Returns
    -------
    artifact : dict — canonical Stage 4 payload
    path     : str  — saved .joblib path
    """
    say = print if verbose else (lambda *a, **k: None)
    say("\n" + "=" * 80)
    say("🎯 META-EBM — STAGE 4 WEIGHTED-CONFIDENCE ARBITER")
    say("=" * 80)

    # ==========================================================
    # 1. LOAD + VERIFY STAGE 1–3 INPUTS
    # ==========================================================
    say("\n📂 1. Loading and verifying Stage 1–3 outputs...")
    inputs = load_stage_outputs(GLOBAL_SPLIT, lr_path, glass_path, ebm_path,
                                verbose=verbose)
    lr, glass, ebm = inputs["lr"], inputs["glass"], inputs["ebm"]
    y_train, y_test = inputs.y_train, inputs.y_test

    say(f"   split fingerprint : {inputs.split_fingerprint}")
    for s in (lr, glass, ebm):
        say(f"   {s.model:<28} train ← {s.train_source}")
        say(f"   {'':<28} test  ← {s.test_source}   [{s.probability_space}]")
    say("   Upstream standalone thresholds (reported only, not used by Stage 4):")
    say(f"     LR  = {lr.standalone_threshold:.4f}   {lr.standalone_threshold_source}")
    say(f"     EBM = {ebm.standalone_threshold:.4f}   {ebm.standalone_threshold_source}")

    # ----------------------------------------------------------
    # TRAIN-SIDE VIEW. Steps 2–5 may read only these names.
    # ----------------------------------------------------------
    lr_tr, ebm_tr, gl_tr = lr.train_proba, ebm.train_proba, glass.train_proba
    pass1_train = glass.train_decisions == PASS1
    pass2_train = glass.train_decisions == PASS2
    glass_covered_train = pass1_train | pass2_train
    glass_pred_train = pass2_train.astype(int)

    # ==========================================================
    # 2. STAGE 4 THRESHOLDS (recall-targeted, train OOF)
    # ==========================================================
    say(f"\n   Stage 4 thresholds (find_recall_threshold on train OOF, "
        f"target={target_recall:.0%}):")
    lr_arb_thresh    = find_recall_threshold(y_train, lr_tr,  target_recall)
    ebm_arb_thresh   = find_recall_threshold(y_train, ebm_tr, target_recall)
    glass_arb_thresh = _GLASS_ARBITER_THRESH
    say(f"     LR    = {lr_arb_thresh:.4f}")
    say(f"     EBM   = {ebm_arb_thresh:.4f}")
    say(f"     GLASS = {glass_arb_thresh:.4f}  (fixed)")
    say(f"   Train rows: {len(y_train):,}   Test rows: {len(y_test):,}")
    say(f"   GLASS Router train coverage: {glass_covered_train.mean():.1%}")

    # ==========================================================
    # 3. CALIBRATION + TRUST WEIGHTS (train OOF)
    # ==========================================================
    say("\n📊 2. Calibration diagnostics + trust weights (train OOF)...")
    lr_cal    = compute_calibration(y_train, lr_tr)
    ebm_cal   = compute_calibration(y_train, ebm_tr)
    glass_cal = compute_calibration(y_train[glass_covered_train],
                                    gl_tr[glass_covered_train])
    say(f"   LR    Brier={lr_cal['brier']:.4f}  ECE={lr_cal['ece']:.4f}")
    say(f"   EBM   Brier={ebm_cal['brier']:.4f}  ECE={ebm_cal['ece']:.4f}")
    say(f"   GLASS Brier={glass_cal['brier']:.4f}  ECE={glass_cal['ece']:.4f}  (covered rows)")

    weight_args = (lr_cal, ebm_cal, glass_cal, y_train, lr_tr, ebm_tr,
                   glass_pred_train, lr_arb_thresh, ebm_arb_thresh)
    weights = compute_hybrid_weights(*weight_args, alpha=0.5,
                                     glass_covered_mask=glass_covered_train)
    weights_pre_audit = compute_hybrid_weights(*weight_args, alpha=0.5,
                                               glass_covered_mask=None)
    MODEL_WEIGHTS = {k: weights[k] for k in ('lr', 'ebm', 'glass')}
    say(f"   Hybrid weights → LR={MODEL_WEIGHTS['lr']:.3f}  "
        f"EBM={MODEL_WEIGHTS['ebm']:.3f}  GLASS={MODEL_WEIGHTS['glass']:.3f}")
    say(f"   (pre-audit all-rows GLASS accuracy would give "
        f"LR={weights_pre_audit['lr']:.3f}  EBM={weights_pre_audit['ebm']:.3f}  "
        f"GLASS={weights_pre_audit['glass']:.3f})")

    # ==========================================================
    # 4. DISAGREEMENT ANALYSIS (train OOF)
    # ==========================================================
    say("\n📊 3. Disagreement analysis (train OOF)...")
    disagreement_report = analyze_disagreements(
        y_true=y_train,
        lr_pred=(lr_tr >= lr_arb_thresh).astype(int),
        ebm_pred=(ebm_tr >= ebm_arb_thresh).astype(int),
        glass_pred=glass_pred_train,
        glass_covered_mask=glass_covered_train,
    )
    for k, v in disagreement_report.items():
        say(f"   {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"   {k}: {v}")

    # ==========================================================
    # 5. ABSTENTION TUNING (train OOF, F2)
    # ==========================================================
    say("\n🔧 4. Tuning min_weighted_confidence (F2 on train OOF)...")
    best_cfg = tune_arbiter_threshold(
        meta_arbiter,
        probs=(lr_tr, ebm_tr, gl_tr),
        masks=(pass1_train, pass2_train),
        y_true=y_train,
        lr_thresh=lr_arb_thresh,
        ebm_thresh=ebm_arb_thresh,
        weights=MODEL_WEIGHTS,
        min_coverage=_MIN_COVERAGE,
    )
    if best_cfg is None:
        say(f"   ⚠️  No value met the {_MIN_COVERAGE:.0%} coverage floor — "
            f"using fallback {_MIN_CONF_FALLBACK}")
        best_cfg = {'min_weighted_confidence': _MIN_CONF_FALLBACK,
                    'train_f2': None, 'train_coverage': None,
                    'selected_by': 'fallback default (no sweep value met coverage floor)'}
    else:
        best_cfg['selected_by'] = (f"F2 sweep on train OOF, coverage ≥ {_MIN_COVERAGE:.0%}")
    MIN_WEIGHTED_CONF = best_cfg['min_weighted_confidence']

    def _fmt(x, spec):
        return "n/a" if x is None else format(x, spec)
    say(f"   Selected → min_weighted_confidence={MIN_WEIGHTED_CONF:.2f}  "
        f"train_f2={_fmt(best_cfg['train_f2'], '.4f')}  "
        f"train_coverage={_fmt(best_cfg['train_coverage'], '.1%')}")

    # Configuration is frozen from here on. Nothing below feeds back into it.
    frozen = {
        'lr': lr_arb_thresh, 'ebm': ebm_arb_thresh, 'glass': glass_arb_thresh,
        'weights': dict(MODEL_WEIGHTS), 'min_weighted_confidence': MIN_WEIGHTED_CONF,
    }

    # ==========================================================
    # 6. FINAL EVALUATION (test — evaluation only)
    # ==========================================================
    say("\n📊 5. Evaluating Meta-EBM (test set, once)...")
    pass1_test = glass.test_decisions == PASS1
    pass2_test = glass.test_decisions == PASS2
    arb_args = (lr.test_proba, ebm.test_proba, glass.test_proba, pass1_test, pass2_test,
                frozen['lr'], frozen['ebm'], frozen['weights'])

    pred_na, prob_na, _ = meta_arbiter(*arb_args, allow_abstain=False,
                                       glass_thresh=frozen['glass'])
    metrics_na = compute_metrics(y_test, pred_na, prob_na)

    pred_a, prob_a, explain_a = meta_arbiter(
        *arb_args, allow_abstain=True,
        min_weighted_confidence=frozen['min_weighted_confidence'],
        glass_thresh=frozen['glass'])
    eval_a = evaluate_with_abstention(y_test, pred_a, prob_a)

    say("\n   No-abstention metrics:")
    for k, v in metrics_na.items():
        say(f"      {k:<12} {v:.4f}")
    say(f"\n   With-abstention  (coverage={eval_a['coverage']:.1%}):")
    if eval_a['metrics']:
        for k, v in eval_a['metrics'].items():
            say(f"      {k:<12} {v:.4f}")

    # ==========================================================
    # 7. SAVE
    # ==========================================================
    say("\n💾 6. Saving Meta-EBM artifact...")
    artifact = {
        'schema':            ARTIFACT_SCHEMA,
        'split_fingerprint': inputs.split_fingerprint,
        'split_fingerprints': inputs.split_fingerprints,
        # Exactly what was consumed: paths, fields, probability spaces, folds,
        # and the full validation ledger.
        'stage4_inputs':     inputs.provenance(),

        'weights':         MODEL_WEIGHTS,
        'weights_details': {**weights['details'],
                            'pre_audit_all_rows_glass_accuracy': {
                                k: weights_pre_audit[k] for k in ('lr', 'ebm', 'glass')}},
        'calibration':     {'lr': lr_cal, 'ebm': ebm_cal, 'glass': glass_cal,
                            'computed_on': 'train OOF (GLASS: covered rows)'},
        'disagreement':    disagreement_report,

        # What the arbiter operates at (Stage 4's own).
        'thresholds': {
            'lr':                      lr_arb_thresh,
            'ebm':                     ebm_arb_thresh,
            'glass':                   glass_arb_thresh,
            'target_recall':           target_recall,
            'min_weighted_confidence': MIN_WEIGHTED_CONF,
            'sources': {
                'lr':    f'find_recall_threshold(y_train, LR OOF, {target_recall})',
                'ebm':   f'find_recall_threshold(y_train, EBM OOF, {target_recall})',
                'glass': 'fixed',
                'min_weighted_confidence': best_cfg['selected_by'],
            },
        },
        # Abstention cut on weighted confidence — not a probability threshold.
        'abstention': {
            'min_weighted_confidence': MIN_WEIGHTED_CONF,
            'selected_by':    best_cfg['selected_by'],
            'train_f2':       best_cfg['train_f2'],
            'train_coverage': best_cfg['train_coverage'],
            'min_coverage':   _MIN_COVERAGE,
        },
        # Upstream standalone operating points — informational only.
        'artifact_thresholds': {
            'lr':  lr.standalone_threshold,
            'ebm': ebm.standalone_threshold,
            'sources': {'lr': lr.standalone_threshold_source,
                        'ebm': ebm.standalone_threshold_source},
            'used_by_stage4': False,
        },

        'no_abstain':   metrics_na,
        'with_abstain': eval_a,
        'explanations': explain_a,

        'train_probs': {'lr': lr_tr, 'ebm': ebm_tr, 'glass': gl_tr},
        'train_probs_are_oof': True,
        'train_fold_ids': {n: s.train_fold_id for n, s in inputs.streams.items()},
        'glass_decisions_train': glass.train_decisions,
        'test_probs': {
            'lr':    lr.test_proba,
            'ebm':   ebm.test_proba,
            'glass': glass.test_proba,
            'meta':  prob_na,
        },
        'test_preds': {
            'lr':           (lr.test_proba  >= lr_arb_thresh).astype(int),
            'ebm':          (ebm.test_proba >= ebm_arb_thresh).astype(int),
            'glass':        pass2_test.astype(int),
            'no_abstain':   pred_na,
            'with_abstain': pred_a,
        },
        'glass_decisions_test': glass.test_decisions,
        'index_train': list(inputs.index_train),
        'index_test':  list(inputs.index_test),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = save_meta_ebm(artifact, base_path=artifact_dir)
    say(f"   ✅ Saved → {path}")
    say("\n🎉 META-EBM COMPLETE")
    say("=" * 80)
    return artifact, path
