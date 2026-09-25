"""
blackbox_pipeline.models.meta_xgb.comparison
============================================
GLASS Meta-EBM vs black-box Meta-XGB on the SAME held-out rows, each at its own
pre-selected (train-only) configuration. Nothing here feeds back into either
model: both artifacts are frozen before this runs.

Reported
--------
* metrics without abstention and on retained rows with abstention
  (coverage, accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC);
* paired behaviour: prediction agreement, both correct / both wrong /
  Meta-EBM only / Meta-XGB only, exact McNemar test on the discordant pairs;
* with abstention: retention overlap, and the 2×2 on rows both retain;
* paired bootstrap (2000 resamples, seed 42 — the Stage 3 comparison
  convention) of every Δ = Meta-XGB − Meta-EBM with a 95% CI.

Meta-EBM's ranking score is its arbiter's weighted probability
(``test_probs["meta"]``): a legitimate score for ROC/PR-AUC, but a weighted
average of base probabilities, not a calibrated probability — noted in output.
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import average_precision_score

from shared.stage4 import Stage4InputError, index_alignment, split_reference

from .evaluation import abstention_metrics, classification_metrics

N_BOOT, SEED = 2000, 42


# ======================================================================
def load_meta_ebm_for_comparison(GLOBAL_SPLIT, meta_ebm, extra_fingerprints=None) -> dict:
    art = joblib.load(meta_ebm) if isinstance(meta_ebm, (str, bytes)) or \
        hasattr(meta_ebm, "__fspath__") else meta_ebm
    if not isinstance(art, dict) or art.get("schema") != "meta_ebm/2":
        raise Stage4InputError(
            "Meta-EBM artifact is not schema 'meta_ebm/2' (the audited Stage 4). "
            "Re-run GLASS Cell 19 so its test rows and provenance are recorded.")
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    fps = set((art.get("split_fingerprints") or {}).values()) | {art.get("split_fingerprint")}
    if not fps & set(ref.fingerprints.values()):
        raise Stage4InputError(f"Meta-EBM artifact was built on another split: {fps}")
    index_alignment(art["index_test"], ref.index_test, "Meta-EBM index_test")
    return art


def find_meta_ebm_artifact(dirs, GLOBAL_SPLIT, extra_fingerprints=None, verbose=True):
    """Newest meta_ebm_*.joblib that passes ``load_meta_ebm_for_comparison``."""
    from pathlib import Path
    files = sorted((p for d in dirs if Path(d).is_dir() for p in Path(d).glob("meta_ebm_*.joblib")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            art = load_meta_ebm_for_comparison(GLOBAL_SPLIT, str(f), extra_fingerprints)
        except Exception as exc:            # noqa: BLE001 — try the next file
            if verbose:
                print(f"  (skip {f.name}: {exc})")
            continue
        return art, str(f)
    raise FileNotFoundError("No audited Meta-EBM artifact (meta_ebm/2) for this split in "
                            f"{[str(d) for d in dirs]}")


# ======================================================================
def _auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _prf(y, d):
    tp = int((y & d).sum()); fp = int((d & (1 - y)).sum()); fn = int((y & (1 - d)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    return prec, rec, f2, float((y == d).mean()) if len(y) else np.nan


def _stats(y, d, s, keep):
    out = {}
    p, r, f2, acc = _prf(y, d)
    out.update({"no_abstain.precision": p, "no_abstain.recall": r, "no_abstain.f2": f2,
                "no_abstain.accuracy": acc, "no_abstain.roc_auc": _auc(y, s),
                "no_abstain.pr_auc": (average_precision_score(y, s)
                                      if 0 < y.sum() < len(y) else np.nan)})
    yk, dk = y[keep], d[keep]
    p, r, f2, acc = _prf(yk, dk)
    out.update({"with_abstain.coverage": float(keep.mean()), "with_abstain.precision": p,
                "with_abstain.recall": r, "with_abstain.f2": f2, "with_abstain.accuracy": acc})
    return out


def _paired_table(y, da, db, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    ca, cb = (da == y)[m], (db == y)[m]
    both, a_only, b_only, neither = (ca & cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum(), (~ca & ~cb).sum()
    n_disc = int(a_only + b_only)
    p = binomtest(int(b_only), n_disc, 0.5).pvalue if n_disc else 1.0
    return {"n": int(m.sum()), "agreement": float((da[m] == db[m]).mean()) if m.any() else np.nan,
            "both_correct": int(both), "meta_ebm_only_correct": int(a_only),
            "meta_xgb_only_correct": int(b_only), "both_wrong": int(neither),
            "mcnemar_exact_p": float(p)}


def compare_with_meta_ebm(GLOBAL_SPLIT, meta_ebm, meta_xgb, n_boot: int = N_BOOT,
                          seed: int = SEED, verbose: bool = True,
                          extra_fingerprints: Optional[dict] = None) -> dict:
    """``meta_ebm``: path or dict (schema meta_ebm/2); ``meta_xgb``: MetaXGBArtifact."""
    ebm = load_meta_ebm_for_comparison(GLOBAL_SPLIT, meta_ebm, extra_fingerprints)
    ref = split_reference(GLOBAL_SPLIT, extra_fingerprints)
    index_alignment(meta_xgb.index_test, ref.index_test, "Meta-XGB index_test")
    y = ref.y_test

    e_pred = np.asarray(ebm["test_preds"]["no_abstain"]).astype(int)
    e_abst = np.asarray(ebm["test_preds"]["with_abstain"]).astype(int)
    e_score = np.asarray(ebm["test_probs"]["meta"], dtype=float)
    e_keep = e_abst != -1
    x_pred = np.asarray(meta_xgb.test["pred"]).astype(int)
    x_score = np.asarray(meta_xgb.test["proba"], dtype=float)
    x_keep = np.asarray(meta_xgb.test["retained"], dtype=bool)
    # Retained rows use Meta-EBM's own with-abstain predictions as stored.
    e_pred_keep = np.where(e_keep, e_abst, e_pred)

    metrics = {
        "Meta-EBM": {"no_abstain": classification_metrics(y, e_pred, e_score),
                     "with_abstain": abstention_metrics(y, e_pred_keep, e_score, e_keep)},
        "Meta-XGB": {"no_abstain": classification_metrics(y, x_pred, x_score),
                     "with_abstain": abstention_metrics(y, x_pred, x_score, x_keep)},
    }
    cols = ["coverage", "accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    rows = {}
    for name, m in metrics.items():
        rows[(name, "no abstention")] = {"coverage": 1.0, **{c: m["no_abstain"][c] for c in cols[1:]}}
        wa = m["with_abstain"]
        rows[(name, "with abstention")] = {"coverage": wa["coverage"],
                                           **{c: (wa["metrics"] or {}).get(c, np.nan)
                                              for c in cols[1:]}}
    table = pd.DataFrame(rows).T[cols]

    paired = {
        "no_abstain": _paired_table(y, e_pred, x_pred),
        "with_abstain_both_retained": _paired_table(y, e_pred_keep, x_pred, e_keep & x_keep),
        "retention": {"both": int((e_keep & x_keep).sum()),
                      "meta_ebm_only": int((e_keep & ~x_keep).sum()),
                      "meta_xgb_only": int((~e_keep & x_keep).sum()),
                      "neither": int((~e_keep & ~x_keep).sum())},
    }

    # ---- paired bootstrap ------------------------------------------------
    rng = np.random.default_rng(seed)
    point_e = _stats(y, e_pred, e_score, e_keep)
    point_x = _stats(y, x_pred, x_score, x_keep)
    # with-abstain precision/recall/f2 use each model's stored retained preds
    point_e.update({k: v for k, v in _stats(y, e_pred_keep, e_score, e_keep).items()
                    if k.startswith("with_abstain")})
    deltas = {k: [] for k in point_e}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yb = y[i]
        if yb.min() == yb.max():
            continue
        se = _stats(yb, e_pred[i], e_score[i], e_keep[i])
        se.update({k: v for k, v in _stats(yb, e_pred_keep[i], e_score[i], e_keep[i]).items()
                   if k.startswith("with_abstain")})
        sx = _stats(yb, x_pred[i], x_score[i], x_keep[i])
        for k in deltas:
            deltas[k].append(sx[k] - se[k])
    brows = []
    for k in point_e:
        d = np.asarray(deltas[k], dtype=float)
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        brows.append({"metric": k, "Meta-EBM": point_e[k], "Meta-XGB": point_x[k],
                      "Δ (XGB−EBM)": point_x[k] - point_e[k], "95% CI low": lo,
                      "95% CI high": hi, "P(Δ > 0)": float((d > 0).mean()),
                      "CI excludes 0": not (lo <= 0 <= hi)})
    boot = pd.DataFrame(brows)

    out = {
        "table": table, "paired": paired, "bootstrap": boot,
        "meta_ebm_config": {"thresholds": ebm.get("thresholds"),
                            "abstention": ebm.get("abstention"),
                            "timestamp": ebm.get("timestamp")},
        "meta_xgb_config": {"operating": meta_xgb.operating,
                            "min_confidence": meta_xgb.abstention["min_confidence"]},
        "protocol": {"n_boot": n_boot, "seed": seed, "n_test": int(len(y)),
                     "selection": "each model at its own train-selected configuration; "
                                  "no test-based tuning of either",
                     "meta_ebm_score": "arbiter weighted probability (ranking score, "
                                       "not calibrated)"},
    }
    out["summary"] = {"table": table.round(6).reset_index().to_dict("records"),
                      "paired": paired,
                      "bootstrap": boot.round(6).to_dict("records")}
    if verbose:
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
            print("held-out metrics (each at its own train-selected configuration)")
            print(table.to_string())
            print("\npaired behaviour")
            print(pd.DataFrame({k: v for k, v in paired.items() if k != "retention"}).T.to_string())
            print(f"retention overlap: {paired['retention']}")
            print(f"\npaired bootstrap Δ = Meta-XGB − Meta-EBM ({n_boot} resamples, seed {seed})")
            print(boot.to_string(index=False))
        print("\nNote: a difference is only reported as real where its 95% CI excludes 0.")
    return out
