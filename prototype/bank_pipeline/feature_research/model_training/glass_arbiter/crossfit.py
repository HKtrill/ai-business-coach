"""
feature_research/model_training/glass_arbiter/crossfit.py
=========================================================
Cross-fitted GLASS Arbiter on the research stage out-of-fold scores.

Inputs are lr_oof / rf_oof / ebm_oof: one out-of-fold score per training row
from each stage trainer. The research notebook has no test split, so the
cascade's "configure on train OOF, evaluate once on test" becomes a cross-fit:
thresholds, weights and min_weighted_confidence are fit on K−1 folds and
frozen, then applied to the held-out fold. Every reported metric is
out-of-fold. A config fit on all rows is also returned; it is what the
cascade would freeze.

Public API
----------
ArbiterResult
cross_fit_arbiter(P, y, cv, rule, target_recall, ...) → ArbiterResult
compare_variants({name: ArbiterResult})              → DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from feature_research.model_training.glass_arbiter.arbiter import (
    ABSTENTION_GRID,
    ABSTENTION_MIN_COVERAGE,
    MODELS,
    ArbiterConfig,
    analyze_disagreements,
    apply_arbiter,
    check_scores,
    compute_calibration,
    compute_metrics,
    evaluate_with_abstention,
    explain,
    fit_arbiter,
)
from feature_research.model_training.shared.folds import recall_at_fpr
from feature_research.model_training.shared.report import show_table, title

METRIC_COLUMNS = ["coverage", "accuracy", "precision", "recall", "f1", "f2", "roc_auc", "recall_at_10fpr"]


@dataclass
class ArbiterResult:
    """
    rule            threshold rule ("recall" / "youden" / "f2")
    config          arbiter fit on all training rows (what the cascade would freeze)
    fold_configs    one frozen config per cross-fit fold
    proba           cross-fitted weighted score per row
    pred_full       cross-fitted decision, no abstention (0/1)
    pred            cross-fitted decision with abstention (−1/0/1)
    explain         per-row decision text
    model_preds     cross-fitted per-stage 0/1 at that fold's thresholds
    row_thresholds  threshold applied to each row, per stage
    calibration     Brier / ECE per stage on its OOF scores
    disagreement    pairwise disagreement report (cross-fitted decisions)
    metrics         one row per stage / arbiter variant, all out-of-fold
    """
    rule: str
    config: ArbiterConfig
    fold_configs: list[ArbiterConfig]
    proba: pd.Series
    pred_full: pd.Series
    pred: pd.Series
    explain: pd.Series
    model_preds: pd.DataFrame
    row_thresholds: pd.DataFrame
    calibration: dict
    disagreement: dict
    metrics: pd.DataFrame


def _metrics_table(y, P, model_preds, pred_full, pred, proba) -> pd.DataFrame:
    rows = {}
    for m in MODELS:
        rows[m.upper()] = {"coverage": 1.0, **compute_metrics(y, model_preds[m], P[m]),
                           "recall_at_10fpr": recall_at_fpr(y, P[m])}
    rows["Arbiter (no abstain)"] = {"coverage": 1.0, **compute_metrics(y, pred_full, proba),
                                    "recall_at_10fpr": recall_at_fpr(y, proba)}
    ev = evaluate_with_abstention(y, pred, proba)
    rows["Arbiter (abstain)"] = {"coverage": ev["coverage"], **(ev["metrics"] or {}), "recall_at_10fpr": np.nan}
    return pd.DataFrame(rows).T.reindex(columns=METRIC_COLUMNS).astype(float)


def _with_range(final: dict, folds: list[dict], fmt: str = ".4f") -> str:
    parts = []
    for m in MODELS:
        vals = [f[m] for f in folds]
        parts.append(f"{m.upper()} {final[m]:{fmt}} [{min(vals):{fmt}}–{max(vals):{fmt}}]")
    return " · ".join(parts)


def _report(r: ArbiterResult, y: pd.Series, n_folds: int) -> None:
    cfg, folds = r.config, r.fold_configs
    n_pos = int(y.sum())
    rule = r.rule + (f" (target recall {cfg.target_recall:.2f})" if r.rule == "recall" else "")

    title(f"GLASS Arbiter · {rule} thresholds")
    print(f"Inputs     {len(y):,} rows · {n_pos:,} positive ({n_pos / len(y):.1%}) · "
          f"stage OOF scores {', '.join(MODELS)} · {n_folds}-fold cross-fit")
    print("           values: all-rows fit [range across cross-fit folds]")
    print(f"Thresholds {_with_range(cfg.thresholds, [f.thresholds for f in folds])}")
    print(f"Weights    {_with_range(cfg.weights, [f.weights for f in folds], '.3f')}")
    fit_f2 = "n/a" if cfg.fit_f2 is None else f"{cfg.fit_f2:.4f}"
    fit_cov = "n/a" if cfg.fit_coverage is None else f"{cfg.fit_coverage:.1%}"
    print(f"Abstain    min_weighted_confidence {cfg.min_conf:.2f} ({cfg.selected_by}; "
          f"fit F2 {fit_f2}, coverage {fit_cov}) · folds "
          + ", ".join(f"{f.min_conf:.2f}" for f in folds))
    print("Calibrated " + " · ".join(
        f"{m.upper()} Brier {c['brier']:.4f} ECE {c['ece']:.4f}" for m, c in r.calibration.items()))
    d = r.disagreement
    print("Disagree   " + " · ".join(
        f"{a.upper()}≠{b.upper()} {d[f'{a}_{b}_disagree_rate']:.1%} "
        f"(right: {a.upper()} {d[f'{a}_correct_on_{a}_{b}_disagree']:.0%} / {b.upper()} {d[f'{b}_correct_on_{a}_{b}_disagree']:.0%})"
        for a, b in (("lr", "rf"), ("lr", "ebm"), ("rf", "ebm"))))
    show_table(r.metrics.round(4))


def cross_fit_arbiter(
    P: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    *,
    rule: str = "recall",
    target_recall: float = 0.70,
    grid: np.ndarray = ABSTENTION_GRID,
    min_coverage: float = ABSTENTION_MIN_COVERAGE,
    verbose: bool = True,
) -> ArbiterResult:
    """
    P   stage OOF scores, columns ("lr", "rf", "ebm"), indexed like y
    y   training labels (y_train)
    cv  splitter for the cross-fit (CV from Cell 5)
    """
    check_scores(P, y)
    fit_kw = dict(rule=rule, target_recall=target_recall, grid=grid, min_coverage=min_coverage)
    n = len(P)
    proba = np.full(n, np.nan)
    pred_full = np.full(n, -2)
    pred = np.full(n, -2)
    model_preds = np.full((n, len(MODELS)), -1)
    row_T = np.full((n, len(MODELS)), np.nan)

    fold_configs = []
    for tr, va in cv.split(P, y):
        cfg = fit_arbiter(P.iloc[tr], y.iloc[tr], **fit_kw)   # frozen before va is touched
        P_va = P.iloc[va]
        pred_full[va], proba[va] = apply_arbiter(P_va, cfg, abstain=False)
        pred[va], _ = apply_arbiter(P_va, cfg, abstain=True)
        T = np.array([cfg.thresholds[m] for m in MODELS])
        model_preds[va] = (P_va.to_numpy() >= T).astype(int)
        row_T[va] = T
        fold_configs.append(cfg)
    if np.isnan(proba).any():
        raise RuntimeError("cross-fit left rows without a score; cv must cover every row once")

    idx = P.index
    model_preds_df = pd.DataFrame(model_preds, index=idx, columns=list(MODELS))
    result = ArbiterResult(
        rule=rule,
        config=fit_arbiter(P, y, **fit_kw),
        fold_configs=fold_configs,
        proba=pd.Series(proba, index=idx, name="arbiter_proba"),
        pred_full=pd.Series(pred_full, index=idx, name="arbiter_pred_full"),
        pred=pd.Series(pred, index=idx, name="arbiter_pred"),
        explain=pd.Series(explain(pred), index=idx, name="arbiter_explain"),
        model_preds=model_preds_df,
        row_thresholds=pd.DataFrame(row_T, index=idx, columns=list(MODELS)),
        calibration={m: compute_calibration(y, P[m]) for m in MODELS},
        disagreement=analyze_disagreements(y, model_preds_df),
        metrics=_metrics_table(y, P, model_preds_df, pred_full, pred, proba),
    )
    if verbose:
        _report(result, y, len(fold_configs))
    return result


def compare_variants(results: dict[str, ArbiterResult]) -> pd.DataFrame:
    """Stack the metrics tables of several cross-fitted variants (e.g. recall / youden / f2)."""
    return pd.concat({name: r.metrics for name, r in results.items()}, names=["thresholds", "model"])
