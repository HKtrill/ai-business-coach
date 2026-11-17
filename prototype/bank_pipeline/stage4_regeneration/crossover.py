"""
stage4_regeneration.crossover
=============================
PR 39 Commit 5, Step 5B: the registered 2x2 crossover and the pre-presentation
cascade baseline (execution contract 1.0.0, sections 6.4, 6.5, 7.3 and 7.5;
gates A5.4 and A5.5; roadmap, Commit 5, "Stage 4 abstention metric
accounting"). Run from ``prototype/bank_pipeline`` on the Step 5A run:

    python -m stage4_regeneration.crossover record --run-id <Step 5A run id> --sums <SHA256SUMS.txt>
    python -m stage4_regeneration.crossover check  --run-id <Step 5A run id>

Nothing is fitted, tuned or rescored. Every number comes from the four arbiter
artifacts Step 5A pinned, read as saved, and from ``GLOBAL_SPLIT``'s test
labels. The test split is previously exposed, so every result is descriptive
(contract L1, section 12).

Registered comparisons (section 6.5), by the unchanged
``compare_with_glass_arbiter``: the same metrics, the exact McNemar test,
retention overlap, and the paired bootstrap of every difference (2,000
resamples of the test rows, seed 42).

* The two columns: the arbiter effect with the upstream held fixed
  (``require_same_upstream=True``), black-box Cell 38's comparison.
* Cascade against cascade, native against native, reported separately. It
  must reproduce the comparison black-box Cell 36 saved in the native Meta-XGB.

Not registered in PR 39 (section 6.5, PR 43): the rows (one arbiter on both
upstreams), an interaction estimate, coverage grids and confidence tests in
calibrated space. The four cells are reported side by side, without inference.

Abstention accounting, per cell. One decision per test row without
abstention, and the rows the arbiter retains with it. Three metric blocks,
each naming its population, with every ratio given as numerator and
denominator:

* ``no_abstention``: every test row, the decision without abstention.
* ``retained_only``: the retained rows only. This is what the existing
  "with abstention" rows report (``abstention_metrics``,
  ``evaluate_with_abstention``).
* ``whole_system``: every test row, an abstention issuing no decision. It
  holds coverage, retention and abstention per class, effective recall
  (TP / all positives), and the precision of the positive decisions issued.
  Accuracy, F1, F2, ROC-AUC and PR-AUC are undefined here without an
  abstention policy, and none is adopted.

The identities N = retained + abstained (overall and per class) and
retained = TP + FP + TN + FN are checked. So is effective recall = retained
recall x positive retention. The accuracy change from all rows to the
retained rows is split exactly into a prevalence-mix term, a recall term and
a specificity term (``decompose``). A ratio with a zero denominator, or a
ranking metric on one class, is ``None`` with the reason, never 0 or 0.5.

Calibration. Meta-XGB's block is reported as recorded, and its test Brier and
ECE are recounted with the same functions. The GLASS Arbiter's weighted score
is not a calibrated probability (section 6.5), so no calibration metric is
computed for it.

Stage 2's role in the final cascade (section 6.4) is descriptive. For each
upstream it reports the router's test routes, their positives, and the
obligations' quantities. It also reports Pass 1 suppression and Pass 2 unique
positives against the Stage 1 and Stage 3 decisions at the Stage 4 recall
cuts, and how each arbiter decided on each route. This is not the PR 38
comparative protocol (F10); the frozen selector is not touched.

Provenance. The Step 5A records are read by sidecar. Step 5A's code identity
is re-derived with this step's own files omitted, so the arbiters were fitted
by the code in this tree. Each arbiter is read by size and SHA-256. Its
identity is checked (``artifact_audit.check_arbiter_identity``,
``check_crossover``), and so is every Stage 1-3 file it consumed against the
Commit 4 manifest. The Commit 4 acceptance and the section 7.5 checks are
re-run. The record ``<run>/crossover_baseline.json`` is written once with a
``.sha256`` sidecar. ``check`` recomputes every result from the artifacts and
compares them value for value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from upstream_regeneration import checks as C
from upstream_regeneration import inventory as I
from upstream_regeneration import records as Rec

from . import runner as R
from . import stage4_copy as S

__all__ = ["STEP", "STEP5B_FILES", "COLUMNS", "ROWS", "N_BOOT", "SEED", "accounting",
           "decompose", "findings", "cell_arrays", "stored_metric_problems", "calibration",
           "stage2_role", "evaluate", "step5a_code_identity", "record", "check", "main"]

STEP = "crossover_baseline"
#: This step's own files. Omitted, they re-derive the code identity Step 5A ran under.
STEP5B_FILES = ("stage4_regeneration/crossover.py", "tests/stage4_regeneration/test_step5b.py")
STEP5A_RECORDS = ("preflight", "execution_glass", "execution_blackbox", "run_record")
#: Per upstream: (GLASS Arbiter key, Meta-XGB key). The registered columns (section 6.5).
COLUMNS = {"glass": ("glass_arbiter", "meta_xgb_on_glass"),
           "blackbox": ("glass_arbiter_on_blackbox", "meta_xgb")}
#: Per arbiter: (on the GLASS upstream, on the black-box upstream). Not registered (PR 43).
ROWS = {"glass_arbiter": ("glass_arbiter", "glass_arbiter_on_blackbox"),
        "meta_xgb": ("meta_xgb_on_glass", "meta_xgb")}
ROUTERS = {"glass": "GLASS Router", "blackbox": "RF Router"}
#: The registered bootstrap (``meta_xgb.comparison``): resamples and seed.
N_BOOT, SEED = 2000, 42
TOL = 1e-12
CONTRACT = {"record": Rec.CONTRACT["record"], "version": Rec.CONTRACT["version"],
            "sections": ["6.3", "6.4", "6.5", "7.1", "7.3", "7.5", "12"],
            "gates": ["A5.4", "A5.5"],
            "roadmap": "documentation/research_roadmap.md, Commit 5, Step 5B and Stage 4 abstention "
                       "metric accounting"}
UNDEFINED_WITHOUT_POLICY = ("accuracy", "f1", "f2", "roc_auc", "pr_auc")
NOT_REGISTERED = {
    "rows": "one arbiter on both upstreams (the upstream effect): not registered in PR 39 "
            "(contract section 6.5; PR 43). The four cells are reported side by side, with no "
            "paired comparison",
    "interaction": "an upstream x arbiter interaction estimate: not registered",
    "coverage_grids": "coverage grids and confidence tests in calibrated space: PR 43",
    "whole_system_uncertainty": "no interval is registered for the whole-system quantities; "
                                "they are point counts",
    "whole_system_metrics": "accuracy, F1, F2, ROC-AUC and PR-AUC over the whole population need "
                            "a policy for abstained rows; none is registered or adopted",
}
LIMITATIONS = (
    "The test split is previously exposed (PR 38 protocol section 15; contract L1, section 12). "
    "Every result is descriptive and informs no calibration, bin, selector, threshold or Stage 4 "
    "setting.",
    "Retained-only metrics are conditional on each arbiter's own confidence rule. Two arbiters "
    "retain different rows, so the registered with-abstention differences compare different "
    "populations.",
    "The registered abstention cut maximises F2 on retained training rows subject to retained "
    "coverage >= 0.50 (shared.stage4.tune_min_confidence). F2 = 5TP / (5TP + 4FN + FP) has no TN "
    "term, so retaining a true negative never raises it. Both confidences are threshold-relative "
    "(|p - t|, or the GLASS Arbiter's weighted |p - t|), so with t below 0.5 a negative decision's "
    "confidence is bounded by t and a positive one's by 1 - t. Abstention can therefore fall mostly "
    "on correct negatives. Retained recall and precision then rise while retained accuracy falls, "
    "and retained recall overstates the share of all positives found (effective recall). The "
    "accounting block and its decomposition report this per cell. The protocol is not changed.",
    "The coverage floor binds on training rows only. Test coverage is reported against it per "
    "cell and can fall below it.",
    "The GLASS Arbiter's score is a weighted average of upstream probabilities, not a calibrated "
    "probability: it supports ROC-AUC and PR-AUC as a ranking, not calibration metrics. Meta-XGB "
    "decides and abstains in its raw, class-weighted space; its calibrated probability is reported "
    "separately.",
    "The existing metric code reports a zero-denominator precision, recall or F-score as 0 "
    "(zero_division=0, the bootstrap's _prf). The GLASS Arbiter's own stored metrics "
    "(compute_metrics) store ROC-AUC 0.5 on a one-class population. The accounting block reports "
    "both as undefined, and the record checks that no stored value of the four arbiters relies on "
    "either convention.",
    "The bootstrap resamples test rows with the models fixed: it does not reflect training "
    "variability.",
    "Stage 2's role is descriptive (F10). Test-side route rates are not checks of the obligations, "
    "which bind at fit on training OOF. The PR 38 Tier 2 complementarity results remain tied to "
    "the historical upstream and are not compared with these.",
    "Stage 4 stacking keeps an indirect upstream dependence (contract L7, F39).",
)
HISTORICAL = ("The PR 36-38 Stage 4 comparisons are the committed outputs of black-box Cells "
              "36-38 (arbiters of 2026-11-06, meta_ebm/2). They are neither re-read nor replaced; "
              "their upstream (the legacy notebook Stage 2, isotonic Stage 1 LR) differs from "
              "PR 39's, so no number is compared across the two.")


# ======================================================================
# metric blocks with explicit populations and denominators
# ======================================================================
def _ratio(num, den) -> Dict[str, Any]:
    num, den = int(num), int(den)
    if den == 0:
        return {"value": None, "num": num, "den": den, "undefined": "zero denominator"}
    return {"value": num / den, "num": num, "den": den}


def _ranking(y: np.ndarray, s: np.ndarray):
    if len(np.unique(y)) < 2:
        why = {"value": None, "undefined": "one class only in this population"}
        return why, dict(why)
    from sklearn.metrics import average_precision_score, roc_auc_score
    return ({"value": float(roc_auc_score(y, s))},
            {"value": float(average_precision_score(y, s))})


def _block(y: np.ndarray, d: np.ndarray, s: np.ndarray, population: str) -> Dict[str, Any]:
    tp = int(((y == 1) & (d == 1)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    tn = int(((y == 0) & (d == 0)).sum())
    fn = int(((y == 1) & (d == 0)).sum())
    n, p, ng = int(len(y)), tp + fn, tn + fp
    roc, pr = _ranking(y, s)
    return {"population": population, "n": n, "positives": p, "negatives": ng,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "prevalence": _ratio(p, n), "positive_prediction_rate": _ratio(tp + fp, n),
            "accuracy": _ratio(tp + tn, n), "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, p), "specificity": _ratio(tn, ng), "npv": _ratio(tn, tn + fn),
            "f1": _ratio(2 * tp, 2 * tp + fn + fp), "f2": _ratio(5 * tp, 5 * tp + 4 * fn + fp),
            "roc_auc": roc, "pr_auc": pr}


def _v(m) -> Optional[float]:
    return None if m is None else m.get("value")


def decompose(base: Mapping[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Accuracy from ``base`` to ``other`` (two ``_block`` results), split exactly:
    acc = pi R + (1 - pi) S, so

        acc_o - acc_b = (pi_o - pi_b)(R_b - S_b) + pi_o (R_o - R_b) + (1 - pi_o)(S_o - S_b)

    with pi the prevalence, R recall and S specificity of each population.
    """
    names = ("prevalence", "recall", "specificity", "accuracy", "precision")
    b, o = ({k: _v(x[k]) for k in names} for x in (base, other))
    if any(v is None for v in (*b.values(), *o.values())):
        return {"value": None, "undefined": "a prevalence, recall, specificity, accuracy or "
                                            "precision is undefined in one population"}
    mix = (o["prevalence"] - b["prevalence"]) * (b["recall"] - b["specificity"])
    rec = o["prevalence"] * (o["recall"] - b["recall"])
    spec = (1 - o["prevalence"]) * (o["specificity"] - b["specificity"])
    delta = o["accuracy"] - b["accuracy"]
    return {"from": base["population"], "to": other["population"],
            "accuracy_change": delta, "prevalence_mix_term": mix, "recall_term": rec,
            "specificity_term": spec, "residual": delta - (mix + rec + spec),
            "changes": {k: o[k] - b[k] for k in ("prevalence", "recall", "specificity",
                                                  "precision")},
            "formula": "acc_o - acc_b = (pi_o - pi_b)(R_b - S_b) + pi_o (R_o - R_b) "
                       "+ (1 - pi_o)(S_o - S_b)"}


def accounting(y, pred, score, retained) -> Dict[str, Any]:
    """
    The abstention accounting of one arbiter on one population (module
    docstring). ``pred``: the decision without abstention, 0 or 1 on every row;
    ``retained``: the rows the arbiter classifies with abstention; ``score``:
    its ranking score on every row.
    """
    y = np.asarray(y).astype(int)
    d = np.asarray(pred).astype(int)
    s = np.asarray(score, dtype=float)
    r = np.asarray(retained, dtype=bool)
    if not (len(y) == len(d) == len(s) == len(r)):
        raise ValueError("y / pred / score / retained lengths differ")
    if not np.isin(d, (0, 1)).all():
        raise ValueError("pred must be the 0/1 decision without abstention on every row")
    a = ~r
    every = _block(y, d, s, "every test row: the decision without abstention")
    kept = _block(y[r], d[r], s[r], "retained rows only: the rows classified with abstention "
                                    "(what the existing 'with abstention' metrics report)")
    n, pos, neg = int(len(y)), int(y.sum()), int((1 - y).sum())
    p_a, n_a = int(y[a].sum()), int((1 - y[a]).sum())
    c = kept["confusion"]
    whole = {
        "population": "every test row; an abstention issues neither a positive nor a negative "
                      "decision",
        "coverage": _ratio(r.sum(), n), "abstention_rate": _ratio(a.sum(), n),
        "positive_retention": _ratio(kept["positives"], pos),
        "positive_abstention": _ratio(p_a, pos),
        "negative_retention": _ratio(kept["negatives"], neg),
        "negative_abstention": _ratio(n_a, neg),
        "retained_prevalence": kept["prevalence"],
        "abstained_prevalence": _ratio(p_a, a.sum()),
        "effective_recall": _ratio(c["tp"], pos),
        "issued_positive_precision": dict(kept["precision"], note=(
            "TP / (TP + FP) over the positive decisions issued; an abstention issues none, so "
            "this is the retained precision")),
        "correct_positives_share": _ratio(c["tp"], n),
        "undefined_without_an_abstention_policy": {
            m: "needs a policy for abstained rows (counted wrong, right or excluded); none is "
               "registered" for m in UNDEFINED_WITHOUT_POLICY},
        "roc_auc_populations": "no_abstention.roc_auc ranks the score of every test row (the "
                               "score exists whether or not the row is retained); "
                               "retained_only.roc_auc ranks the retained rows only, which the "
                               "confidence rule selected from the score",
    }
    eff, rec_r, keep_p = (_v(whole["effective_recall"]), _v(kept["recall"]),
                          _v(whole["positive_retention"]))
    identities = {
        "N_total = N_retained + N_abstained": n == int(r.sum()) + int(a.sum()),
        "positives = retained + abstained positives": pos == kept["positives"] + p_a,
        "negatives = retained + abstained negatives": neg == kept["negatives"] + n_a,
        "N_retained = TP + FP + TN + FN (retained)": int(r.sum()) == sum(c.values()),
        "retained positives = TP + FN; retained negatives = TN + FP":
            kept["positives"] == c["tp"] + c["fn"] and kept["negatives"] == c["tn"] + c["fp"],
        "N_total = TP + FP + TN + FN (no abstention)": n == sum(every["confusion"].values()),
        "effective recall = retained recall x positive retention": _product_holds(eff, rec_r,
                                                                                   keep_p),
    }
    dec = decompose(every, kept)
    if dec.get("residual") is not None:
        identities["accuracy decomposition is exact"] = abs(dec["residual"]) < 1e-9
    return {"no_abstention": every, "retained_only": kept, "whole_system": whole,
            "decomposition_all_rows_to_retained": dec, "identities": identities}


def _product_holds(eff, rec_r, keep_p) -> bool:
    """TP/P = TP/P_r x P_r/P; with no positive retained both sides are 0 (P_r/P = 0)."""
    if keep_p is None:                                   # no positives in the population
        return eff is None
    if keep_p == 0:
        return eff == 0
    return rec_r is not None and eff is not None and abs(eff - rec_r * keep_p) < TOL


def _gt(a, b) -> Optional[bool]:
    return None if a is None or b is None else bool(a > b)


def findings(acc: Mapping[str, Any], train_coverage, min_coverage) -> Dict[str, Any]:
    """Comparisons that need no interpretation: each True, False or None (undefined)."""
    every, kept, whole = acc["no_abstention"], acc["retained_only"], acc["whole_system"]
    return {
        "retained_accuracy_below_all_rows": _gt(_v(every["accuracy"]), _v(kept["accuracy"])),
        "retained_precision_above_all_rows": _gt(_v(kept["precision"]), _v(every["precision"])),
        "retained_recall_above_all_rows": _gt(_v(kept["recall"]), _v(every["recall"])),
        "retained_roc_auc_above_all_rows": _gt(_v(kept["roc_auc"]), _v(every["roc_auc"])),
        "retained_specificity_below_all_rows": _gt(_v(every["specificity"]),
                                                   _v(kept["specificity"])),
        "retained_prevalence_above_overall": _gt(_v(kept["prevalence"]), _v(every["prevalence"])),
        "effective_recall_below_no_abstention_recall": _gt(_v(every["recall"]),
                                                           _v(whole["effective_recall"])),
        "abstention_falls_more_on_negatives": _gt(_v(whole["negative_abstention"]),
                                                  _v(whole["positive_abstention"])),
        "test_coverage_below_training_floor": (None if min_coverage is None else
                                               bool(_v(whole["coverage"]) < min_coverage)),
        "train_coverage": train_coverage, "min_coverage": min_coverage,
    }


# ======================================================================
# one arbiter
# ======================================================================
def _glass(key: str) -> bool:
    return key.startswith("glass_arbiter")


def cell_arrays(key: str, art) -> Dict[str, Any]:
    """An arbiter's saved test arrays and configuration, as saved."""
    if _glass(key):
        pred = np.asarray(art["test_preds"]["no_abstain"]).astype(int)
        with_abstain = np.asarray(art["test_preds"]["with_abstain"]).astype(int)
        retained = with_abstain != -1
        ab = art.get("abstention") or {}
        return {"pred": pred, "retained": retained,
                "score": np.asarray(art["test_probs"]["meta"], dtype=float),
                "retained_decisions_unchanged": bool(np.array_equal(with_abstain[retained],
                                                                    pred[retained])),
                "train_coverage": ab.get("train_coverage"), "min_coverage": ab.get("min_coverage"),
                "configuration": {
                    "thresholds": {k: (art.get("thresholds") or {}).get(k) for k in
                                   ("lr", "ebm", "glass", "target_recall",
                                    "min_weighted_confidence")},
                    "weights": art.get("weights"),
                    "abstention": {k: ab.get(k) for k in ("min_weighted_confidence", "train_f2",
                                                          "train_coverage", "min_coverage",
                                                          "selected_by", "search")},
                    "score": "weighted probability (a ranking score, not a calibrated "
                             "probability)"}}
    t = art.test
    pred = np.asarray(t["pred"]).astype(int)
    retained = np.asarray(t["retained"], dtype=bool)
    ab = art.abstention or {}
    return {"pred": pred, "retained": retained, "score": np.asarray(t["proba"], dtype=float),
            "retained_decisions_unchanged": bool(np.array_equal(
                np.asarray(t["pred_with_abstain"]).astype(int), np.where(retained, pred, -1))),
            "train_coverage": ab.get("train_coverage"), "min_coverage": ab.get("min_coverage"),
            "configuration": {
                "operating": dict(art.operating),
                "abstention": {k: ab.get(k) for k in ("min_confidence", "train_f2",
                                                      "train_coverage", "min_coverage",
                                                      "selected_by", "confidence", "search")},
                "score": f"Meta-XGB probability in its operating space "
                         f"({art.operating.get('space')})"}}


def _close(a, b) -> bool:
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < TOL
    except (TypeError, ValueError):
        return False


def stored_metric_problems(key: str, art, acc: Mapping[str, Any]) -> List[str]:
    """Where the arbiter's own saved test metrics differ from the recount (empty: none)."""
    every, kept = acc["no_abstention"], acc["retained_only"]
    coverage = _v(acc["whole_system"]["coverage"])
    if _glass(key):
        na, wa = art.get("no_abstain") or {}, art.get("with_abstain") or {}
        names = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc")
    else:
        na, wa = art.test.get("no_abstain") or {}, art.test.get("with_abstain") or {}
        names = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc")
    pairs = [(f"no_abstain.{m}", na.get(m), every[m]) for m in names]
    pairs += [(f"with_abstain.{m}", (wa.get("metrics") or {}).get(m), kept[m]) for m in names]
    problems = []
    for name, stored, recount in pairs:
        if _v(recount) is None:
            problems.append(f"{name}: stored {stored!r} for a metric undefined on its population "
                            f"({recount.get('undefined')})")
        elif not _close(stored, _v(recount)):
            problems.append(f"{name}: stored {stored!r}, recount {_v(recount)!r}")
    if not _close(wa.get("coverage"), coverage):
        problems.append(f"with_abstain.coverage: stored {wa.get('coverage')!r}, recount {coverage!r}")
    return problems


def calibration(key: str, art, y) -> Dict[str, Any]:
    """Meta-XGB's block as recorded, with its test Brier and ECE recounted; none for the GLASS score."""
    if _glass(key):
        return {"applicable": False,
                "reason": "the GLASS Arbiter's weighted score is not a calibrated probability "
                          "(contract section 6.5)",
                "upstream_train_oof_as_recorded": art.get("calibration")}
    from sklearn.metrics import brier_score_loss
    from shared.metrics import calculate_ece
    cal = dict(art.learner.calibration or {})
    keep = ("applied", "method", "reason", "ece_threshold", "ece_bins", "ece_before", "ece_after",
            "brier_before", "brier_after", "test_ece_raw", "test_ece_calibrated",
            "test_brier_raw", "test_brier_calibrated", "calibrator_fitted_on")
    bins = int(cal.get("ece_bins") or 10)
    y = np.asarray(y).astype(int)
    test = {}
    for name, field in (("raw", "proba_raw"), ("calibrated", "proba_calibrated")):
        p = np.asarray(art.test[field], dtype=float)
        test[name] = {"brier": float(brier_score_loss(y, p)),
                      "ece": float(calculate_ece(y, p, bins)), "ece_bins": bins,
                      "recorded_brier": cal.get(f"test_brier_{name}"),
                      "recorded_ece": cal.get(f"test_ece_{name}")}
    return {"applicable": True, "operating_space": art.operating.get("space"),
            "as_recorded": {k: cal.get(k) for k in keep if k in cal}, "test_recount": test}


def _calibration_problems(cal: Mapping[str, Any]) -> List[str]:
    out = []
    for name, t in (cal.get("test_recount") or {}).items():
        for m in ("brier", "ece"):
            rec = t.get(f"recorded_{m}")
            if rec is not None and not _close(rec, t[m]):
                out.append(f"{name} {m}: recorded {rec!r}, recount {t[m]!r}")
    return out


# ======================================================================
# Stage 2's role in the final cascade (descriptive, section 6.4)
# ======================================================================
def _states(dec, y, rows) -> Dict[str, Any]:
    from shared.stage4 import ABSTAIN, PASS1, PASS2
    return {st: {"n": int((rows & (dec == st)).sum()),
                 "positives": int((rows & (dec == st) & (y == 1)).sum()),
                 "negatives": int((rows & (dec == st) & (y == 0)).sum())}
            for st in (PASS1, PASS2, ABSTAIN)}


def stage2_role(upstream: str, glass_art: Mapping[str, Any], cells: Mapping[str, Mapping],
                y, contract) -> Dict[str, Any]:
    """
    The router's test routes in one upstream, and how each arbiter of that
    column decided on them. ``glass_art``: the column's GLASS Arbiter, which
    saved the routes and the Stage 1 / Stage 3 decisions at its Stage 4 cuts.
    ``cells``: ``{key: cell_arrays(...)}`` for both arbiters of the column. The
    Meta-XGB of the column read the same routes (identical content
    fingerprints, which the registered comparison requires).
    """
    from shared.stage4 import ABSTAIN, PASS1, PASS2
    y = np.asarray(y).astype(int)
    dec = np.asarray(glass_art["glass_decisions_test"]).astype(str)
    s1 = np.asarray(glass_art["test_preds"]["lr"]).astype(int)
    s3 = np.asarray(glass_art["test_preds"]["ebm"]).astype(int)
    if not (len(dec) == len(s1) == len(s3) == len(y)) or \
            not np.isin(dec, (PASS1, PASS2, ABSTAIN)).all():
        raise ValueError(f"{upstream}: routes are not one of {PASS1}/{PASS2}/{ABSTAIN} on every row")
    every = np.ones(len(y), dtype=bool)
    p1, p2 = dec == PASS1, dec == PASS2
    flagged = (s1 == 1) | (s3 == 1)
    pos = int(y.sum())
    th = glass_art.get("thresholds") or {}
    out = {
        "router": ROUTERS[upstream],
        "upstream_flags": (f"Stage 1 and Stage 3 decisions at the Stage 4 recall cuts the GLASS "
                           f"Arbiter recorded (find_recall_threshold on training OOF, target "
                           f"recall {th.get('target_recall')}): stage 1 >= {th.get('lr')}, "
                           f"stage 3 >= {th.get('ebm')}"),
        "routes": _states(dec, y, every),
        "obligation_quantities_on_test": {
            "note": "descriptive: the obligations bind at fit, on training OOF (frozen contract "
                    f"{contract.contract_id} {contract.contract_version}); not a gate",
            "pass1_npv": {**_ratio(((y == 0) & p1).sum(), p1.sum()),
                          "contract_floor": contract.min_pass1_npv},
            "pass1_leakage_rate": {**_ratio(((y == 1) & p1).sum(), pos),
                                   "contract_cap": contract.max_pass1_leakage_rate},
            "pass2_precision": {**_ratio(((y == 1) & p2).sum(), p2.sum()),
                                "contract_floor": contract.min_pass2_precision},
            "pass1_rows": int(p1.sum()), "pass2_rows": int(p2.sum()),
            "training_support_floors": {"pass1": contract.min_pass1_support,
                                        "pass2": contract.min_pass2_support}},
        "pass1_suppression": {
            "pass1_rows_flagged_by_stage1_or_stage3": int((p1 & flagged).sum()),
            "intercepted_negatives": int((p1 & flagged & (y == 0)).sum()),
            "suppressed_positives": int((p1 & flagged & (y == 1)).sum()),
            "pass1_positives_flagged_by_neither": int((p1 & ~flagged & (y == 1)).sum())},
        "pass2_contribution": {
            "pass2_positives": int((p2 & (y == 1)).sum()),
            "unique_positives": int((p2 & ~flagged & (y == 1)).sum()),
            "redundant_positives": int((p2 & flagged & (y == 1)).sum()),
            "unique_negatives_added": int((p2 & ~flagged & (y == 0)).sum())},
        "arbiters": {},
    }
    for key, c in cells.items():
        r, d = np.asarray(c["retained"], dtype=bool), np.asarray(c["pred"]).astype(int)
        by = {}
        for st in (PASS1, PASS2, ABSTAIN):
            m = dec == st
            k = m & r
            by[st] = {"n": int(m.sum()), "retained": int(k.sum()), "abstained": int((m & ~r).sum()),
                      "tp": int((k & (d == 1) & (y == 1)).sum()),
                      "fp": int((k & (d == 1) & (y == 0)).sum()),
                      "tn": int((k & (d == 0) & (y == 0)).sum()),
                      "fn": int((k & (d == 0) & (y == 1)).sum())}
        out["arbiters"][key] = {
            "by_route": by,
            "unique_pass2_positives_retained_as_positive":
                int((p2 & ~flagged & (y == 1) & r & (d == 1)).sum()),
            "suppressed_positives_retained_as_positive":
                int((p1 & flagged & (y == 1) & r & (d == 1)).sum()),
            "pass1_positives_retained_as_negative": int((p1 & (y == 1) & r & (d == 0)).sum())}
    return out


# ======================================================================
# the four cells, the registered comparisons, Stage 2
# ======================================================================
def evaluate(GLOBAL_SPLIT, arts: Mapping[str, Any], *, n_boot: int = N_BOOT, seed: int = SEED,
             extra_fingerprints: Optional[dict] = None) -> Dict[str, Any]:
    """
    Every Step 5B result from the four loaded arbiters (keyed as
    ``artifact_audit.ROOTS``). Deterministic. ``extra_fingerprints`` is for
    synthetic splits only (tests).
    """
    from blackbox_pipeline.models.meta_xgb.comparison import compare_with_glass_arbiter
    from shared.stage2.contract import load_frozen_contract
    from shared.stage4 import split_reference
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    y = np.asarray(ref.y_test).astype(int)
    arrays = {key: cell_arrays(key, arts[key]) for key in S.ROOT_KEYS}
    cells = {}
    for key, c in arrays.items():
        art = arts[key]
        up = ((art if isinstance(art, dict) else vars(art)).get("stage4_inputs") or {}).get(
            "upstream") or {}
        acc = accounting(y, c["pred"], c["score"], c["retained"])
        cells[key] = {"upstream": up.get("arm"), "arbiter": up.get("arbiter"),
                      "crossover": up.get("crossover"), "combination": up.get("combination"),
                      "configuration": c["configuration"],
                      "retained_decisions_unchanged": c["retained_decisions_unchanged"],
                      **acc,
                      "findings": findings(acc, c["train_coverage"], c["min_coverage"]),
                      "stored_metric_problems": stored_metric_problems(key, art, acc),
                      "calibration": calibration(key, art, y)}

    def compare(glass_key, xgb_key, same):
        res = compare_with_glass_arbiter(GLOBAL_SPLIT, arts[glass_key], arts[xgb_key],
                                         n_boot=n_boot, seed=seed, verbose=False,
                                         extra_fingerprints=extra_fingerprints,
                                         require_same_upstream=same)
        return {"glass_arbiter": glass_key, "meta_xgb": xgb_key, "protocol": res["protocol"],
                **res["summary"]}

    comparisons = {}
    for arm, (g, x) in COLUMNS.items():
        comparisons[f"{arm}_column"] = {
            "question": "the arbiter effect with the upstream held fixed (registered; black-box "
                        "Cell 38)", **compare(g, x, True),
            "accuracy_decomposition": {
                "no_abstention": decompose(cells[g]["no_abstention"], cells[x]["no_abstention"]),
                "retained_only": decompose(cells[g]["retained_only"], cells[x]["retained_only"])}}
    comparisons["cascade"] = {
        "question": "cascade against cascade, native against native (registered, reported "
                    "separately; black-box Cell 36). The upstreams differ by design",
        **compare("glass_arbiter", "meta_xgb", False)}
    contract = load_frozen_contract()
    stage2 = {arm: stage2_role(arm, arts[g], {g: arrays[g], x: arrays[x]}, y, contract)
              for arm, (g, x) in COLUMNS.items()}
    return {"population": {"rows": "the test rows of GLOBAL_SPLIT, each scored once by every "
                                   "arbiter (previously exposed; descriptive)",
                           "n": int(len(y)), "positives": int(y.sum()),
                           "negatives": int((1 - y).sum()),
                           "split_fingerprints": dict(ref.fingerprints)},
            "cells": cells, "comparisons": comparisons, "stage2": stage2,
            "not_registered": NOT_REGISTERED}


# ======================================================================
# provenance
# ======================================================================
def step5a_code_identity() -> Dict[str, Any]:
    """``runner.code_identity`` of this tree with ``STEP5B_FILES`` left out."""
    bank = R.BANK_ROOT
    base = Rec.code_identity(bank, overrides={f: None for f in STEP5B_FILES
                                              if f.startswith("tests/")})
    h, n = hashlib.sha256(), 0
    for p in sorted((bank / "stage4_regeneration").rglob("*.py")):
        rel = p.relative_to(bank).as_posix()
        if "__pycache__" in p.parts or rel in STEP5B_FILES:
            continue
        h.update(rel.encode("utf-8") + b"\0")
        h.update(bytes.fromhex(Rec.file_sha256_lf(p)))
        n += 1
    return {**base, "stage4_regeneration_sha256": h.hexdigest(), "stage4_regeneration_files": n}


_CODE_KEYS = ("head", "code_sha256", "stage4_regeneration_sha256")


def _step5a(run_id: str) -> Dict[str, Any]:
    """The Step 5A records, each passed and sidecar-checked, and the three files they name."""
    recs = {name: R.require_passed(run_id, name) for name in STEP5A_RECORDS}
    run = recs["run_record"]
    files = {}
    for name, want in ((R.PINS_FILE, run["pins_file"]["sha256"]),
                       (f"{R.ARBITER_MANIFEST}.json", run["arbiter_manifest"]["sha256"]),
                       (R.AUDIT_FILE, (run.get("audit") or {}).get("sha256"))):
        data = R._verified_bytes(R.run_dir(run_id) / name)
        if hashlib.sha256(data).hexdigest() != want:
            raise Rec.RecordError(f"{name} is not the file the Step 5A run record names")
        files[name] = hashlib.sha256(data).hexdigest()
    recs["files"] = files
    recs["sidecars"] = {f"{name}.json": R._sidecar(R.run_dir(run_id) / f"{name}.json")
                        for name in STEP5A_RECORDS}
    return recs


def _load_arbiters(run_record: Mapping[str, Any], split, gate=None):
    """``({key: artifact}, {key: path})`` by the run record's path, size and SHA-256."""
    from shared.pinned_artifacts import PinnedFile, verify_pin
    from tests.support import artifact_audit as audit
    arts, paths = {}, {}
    for key in S.ROOT_KEYS:
        a = run_record["arbiters"][key]
        pin = PinnedFile(key, a["path"], a["sha256"], int(a["bytes"]))

        def load(key=key, pin=pin):
            paths[key] = verify_pin(pin, R.BANK_ROOT)
            arts[key] = audit.load_arbiter(key, paths[key], split)
            return True

        if gate is None:
            load()
        else:
            gate.check(f"A5.4 {key}: {pin.path} is the version Step 5A produced (size, SHA-256) "
                       "and loads on this split (schema, split identity, test index)", load,
                       pin.sha256[:16])
    return arts, paths


def _first_difference(a, b, path="results") -> Optional[str]:
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present on one side only"
            d = _first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} items"
        for i, (x, z) in enumerate(zip(a, b)):
            d = _first_difference(x, z, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


# ======================================================================
# the record
# ======================================================================
def record(run_id: str, sums_path) -> Dict[str, Any]:
    """Gates A5.4 and A5.5 and the baseline record ``<run>/crossover_baseline.json``."""
    from shared.pinned_artifacts import read_pins
    from tests.support import artifact_audit as audit
    R.check_run_id(run_id)
    if (R.run_dir(run_id) / f"{STEP}.json").exists():
        raise Rec.RecordError(f"run {run_id} already has a {STEP} record: records are written once")
    rec = {"format": f"pr39.commit5b.{STEP}/1", "run_id": run_id, "step": STEP,
           "contract": CONTRACT, "upstream_run": R.UPSTREAM_RUN, "started_utc": Rec.now(),
           "status": "the registered 2x2 crossover and the pre-presentation cascade baseline "
                     "(Step 5B): descriptive results on a previously exposed test split; the "
                     "owner's acceptance is the Commit 5 commit"}
    g4, g5 = C.Gate("A5.4"), C.Gate("A5.5")
    gates = {"A5.4": g4, "A5.5": g5}

    def stop():
        rec.update(gates={g: x.record() for g, x in gates.items()})
        return R._finish(run_id, STEP, rec, False)

    held: Dict[str, Any] = {}

    def step5a_records():
        held["step5a"] = _step5a(run_id)
        return True

    g4.check("A5.4 the Step 5A records (preflight, both executions, run record) passed and match "
             "their sidecars; the pins file, arbiter manifest and audit report are the files the "
             "run record names", step5a_records)
    if not g4.passed:
        return stop()
    s5a = held["step5a"]
    run = s5a["run_record"]
    derived = step5a_code_identity()
    g4.check("A5.4 Step 5A's code identity is re-derived with this step's files omitted: the "
             "arbiters were fitted by the code in this tree",
             lambda: all(derived.get(k) == run["code"].get(k) for k in _CODE_KEYS),
             lambda: {k: (run["code"].get(k), derived.get(k)) for k in _CODE_KEYS})
    upstream = R._accepted_upstream(g4)
    manifest = read_pins(R.manifest_path())
    entries = json.loads((R.run_dir(run_id) / f"{R.ARBITER_MANIFEST}.json").read_text(
        encoding="utf-8"))
    g4.check("A5.4 the arbiter manifest names the four versions the run record names",
             lambda: entries == {k: run["arbiters"][k]["path"] for k in S.ROOT_KEYS})
    split = C.global_split()
    arts, paths = _load_arbiters(run, split, g4)
    if sorted(arts) != sorted(S.ROOT_KEYS):
        return stop()
    for key in S.ROOT_KEYS:
        g4.check(f"A5.4 {key}: it is the {S.ROOT_UPSTREAM[key]}-upstream cell it records "
                 "(artifact_audit.check_arbiter_identity, PR 39 generation)",
                 lambda key=key: audit.check_arbiter_identity(key, arts[key],
                                                              generation=audit.GENERATION) is None)
        g4.check(f"A5.4 {key}: every Stage 1-3 file it consumed is the Commit 4 manifest's pin",
                 lambda key=key: bool(R.upstream_resolution(arts[key], S.ROOT_UPSTREAM[key],
                                                            manifest)))
    g4.check("A5.4 the 2x2 holds each upstream fixed within a column and keeps the arms apart "
             "within a row (artifact_audit.check_crossover)",
             lambda: audit.check_crossover(arts, {k: str(p) for k, p in paths.items()},
                                           split) is not None)
    results = evaluate(split, arts)
    for key, cell in results["cells"].items():
        g4.check(f"A5.4 {key}: population accounting identities hold",
                 lambda cell=cell: all(cell["identities"].values()),
                 lambda cell=cell: [k for k, v in cell["identities"].items() if not v] or "all")
        g4.check(f"A5.4 {key}: abstention never changes a retained row's decision",
                 lambda cell=cell: cell["retained_decisions_unchanged"])
        g4.check(f"A5.4 {key}: its saved test metrics are the recount's (same populations and "
                 "denominators; no substituted value)",
                 lambda cell=cell: not cell["stored_metric_problems"],
                 lambda cell=cell: cell["stored_metric_problems"][:6] or "all equal")
        if cell["calibration"]["applicable"]:
            g4.check(f"A5.4 {key}: test Brier and ECE recounted (shared.metrics.calculate_ece) "
                     "equal the recorded values",
                     lambda cell=cell: not _calibration_problems(cell["calibration"]),
                     lambda cell=cell: _calibration_problems(cell["calibration"]) or "equal")
    for arm in COLUMNS:
        g4.check(f"A5.4 the {arm} column is compared with the upstream held fixed "
                 "(require_same_upstream=True)",
                 lambda arm=arm: results["comparisons"][f"{arm}_column"]["protocol"]["upstream"][
                     "held_fixed"] is True)
    saved = (getattr(arts["meta_xgb"], "comparison", None) or {}).get("summary")
    redone = {k: results["comparisons"]["cascade"][k] for k in ("table", "paired", "bootstrap")}
    g4.check("A5.4 the cascade comparison reproduces the one black-box Cell 36 saved in the native "
             "Meta-XGB, value for value",
             lambda: saved is not None and Rec.canonical(saved) == Rec.canonical(redone),
             lambda: _first_difference(Rec.canonical(saved), Rec.canonical(redone), "cascade")
             if saved is not None else "no saved comparison")
    # A5.5: the baseline record, and nothing else changed
    evidence = {"preservation": C.preservation(g5, sums_path)}
    after = I.snapshot(R.BANK_ROOT)
    g5.check("A5.5 no model file changed since the Step 5A black-box execution",
             lambda: I.diff(s5a["execution_blackbox"]["inventory_after"], after) ==
             {"added": [], "modified": [], "deleted": []},
             lambda: {k: v[:5] for k, v in I.diff(s5a["execution_blackbox"]["inventory_after"],
                                                   after).items()})
    status = Rec.git_status(R.BANK_ROOT)
    before = s5a["preflight"]["git_status"] or []
    new = sorted(set(status or []) - set(before))
    gone = sorted(set(before) - set(status or []))

    def allowed(line: str) -> bool:
        p = line[3:].strip().strip('"')
        return f"research_logs/pr39/{run_id}/" in p or any(p == f or p.endswith("/" + f)
                                                          for f in STEP5B_FILES)
    unexpected = [ln for ln in new if not allowed(ln)] + gone
    g5.check("A5.5 git status since the Step 5A preflight: only this run's records and this step's "
             "files are new", lambda: status is not None and not unexpected,
             lambda: unexpected[:8])
    g5.check("A5.5 the record holds all four cells, both columns, the cascade and Stage 2 for both "
             "upstreams", lambda: sorted(results["cells"]) == sorted(S.ROOT_KEYS)
             and sorted(results["comparisons"]) == ["blackbox_column", "cascade", "glass_column"]
             and sorted(results["stage2"]) == sorted(COLUMNS))
    rec.update(
        gates={g: x.record() for g, x in gates.items()},
        references={
            "commit4": {"run_id": R.UPSTREAM_RUN,
                        "artifact_manifest": {"path": R._rel(R.manifest_path()),
                                              "sha256": upstream.get("manifest_sha256")},
                        "acceptance_sha256": upstream.get("acceptance_sha256"),
                        "roles": upstream.get("roles")},
            "step5a": {"run_id": run_id, "records": s5a["sidecars"], "files": s5a["files"]},
            "pr38_stage2_run": C.PR38_RUN_ID},
        arbiters={k: {**run["arbiters"][k], "upstream_files": {
            stage: manifest["files"][role] | {"role": role}
            for stage, role in R.UPSTREAM_ROLE_OF[S.ROOT_UPSTREAM[k]].items()}}
            for k in S.ROOT_KEYS},
        results=results, limitations=list(LIMITATIONS), historical=HISTORICAL,
        evidence=evidence, git_status_delta={"new": new, "gone": gone, "unexpected": unexpected},
        code={"step5b": R.code_identity(), "step5a_rederived": derived,
              "step5a_recorded": {k: run["code"].get(k) for k in _CODE_KEYS},
              "step5b_files": list(STEP5B_FILES)},
        environment=Rec.environment())
    return R._finish(run_id, STEP, rec, g4.passed and g5.passed)


def check(run_id: str) -> Dict[str, Any]:
    """Re-verify the record: sidecars, the Step 5A records, the arbiters, and every result."""
    rec = R.require_passed(run_id, STEP)
    s5a = _step5a(run_id)
    problems = []
    if s5a["sidecars"] != rec["references"]["step5a"]["records"] or \
            s5a["files"] != rec["references"]["step5a"]["files"]:
        problems.append("the Step 5A records are not the ones the baseline names")
    split = C.global_split()
    arts, _ = _load_arbiters(s5a["run_record"], split)
    diff = _first_difference(Rec.canonical(evaluate(split, arts)), rec["results"])
    if diff:
        problems.append(f"a result does not reproduce: {diff}")
    if problems:
        raise Rec.RecordError(f"run {run_id} {STEP}: " + "; ".join(problems))
    return {"run_id": run_id, "record_sha256": R._sidecar(R.run_dir(run_id) / f"{STEP}.json"),
            "reproduced": True}


# ======================================================================
# command line
# ======================================================================
def _pct(m) -> str:
    v = _v(m)
    return "  n/a " if v is None else f"{v:6.3f}"


def summary_lines(results: Mapping[str, Any]) -> List[str]:
    """One line per cell: the accounting quantities that differ in population."""
    head = (f"  {'cell':<26}{'cover':>7}{'pos.ret':>8}{'neg.ret':>8}{'rec.all':>8}"
            f"{'rec.ret':>8}{'rec.eff':>8}{'prec.ret':>9}{'acc.all':>8}{'acc.ret':>8}")
    out = [head]
    for key, c in results["cells"].items():
        a, r, w = c["no_abstention"], c["retained_only"], c["whole_system"]
        out.append(f"  {key:<26}{_pct(w['coverage']):>7}{_pct(w['positive_retention']):>8}"
                   f"{_pct(w['negative_retention']):>8}{_pct(a['recall']):>8}"
                   f"{_pct(r['recall']):>8}{_pct(w['effective_recall']):>8}"
                   f"{_pct(r['precision']):>9}{_pct(a['accuracy']):>8}{_pct(r['accuracy']):>8}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stage4_regeneration.crossover",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record", help="A5.4-A5.5: the registered 2x2 and the baseline record")
    p.add_argument("--run-id", required=True, help="the Step 5A run id (c5_<YYYYMMDD_HHMMSS>)")
    p.add_argument("--sums", required=True, help="the baseline snapshot's SHA256SUMS.txt")
    p = sub.add_parser("check", help="re-verify the record and recompute every result")
    p.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "record":
            rec = record(args.run_id, args.sums)
            print(f"OK {STEP} for run {args.run_id}: wrote {rec.get('path')}")
            for name, g in rec["gates"].items():
                print(f"  {name}: {g['evidence']}")
            print("\n".join(summary_lines(rec["results"])))
        else:
            out = check(args.run_id)
            print(f"OK {STEP} for run {args.run_id}: every result reproduced")
            print(f"  {STEP}.json sha256 {out['record_sha256']}")
        return 0
    except R.StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                        # noqa: BLE001 - a refusal, printed
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
