"""
model_training.report
=====================
Compact output shared by the stage trainers and diagnostics. Display only:
nothing here changes a result.

Public API
----------
in_notebook()                      True inside a Jupyter kernel
show_table(df)                     HTML table in a notebook, plain text elsewhere
title(text)                        section heading with a rule
fmt_list(items, limit)             "a, b, c, … (+n)" or "none"
stability(auc_std)                 "stable" / "moderate variance" / "high variance"
baseline_verdict(delta, tol)       "✅ HOLD" / "⚠️ BELOW TARGET" / "❌ REGRESSED"
training_report(...)               the end-of-training summary used by train_lr / train_rf / train_ebm
"""

from __future__ import annotations

from typing import Iterable, Optional

import optuna
import pandas as pd

RULE = "─" * 78


def in_notebook() -> bool:
    """True inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    return ip is not None and getattr(ip, "kernel", None) is not None


def show_table(df: pd.DataFrame) -> None:
    """Render df as an HTML table inside a Jupyter kernel; otherwise print it as text."""
    if in_notebook():
        from IPython.display import display
        display(df)
    else:
        print(df.to_string())


def title(text: str) -> None:
    print(f"\n{text}\n{RULE}")


def fmt_list(items: Iterable, limit: int = 10) -> str:
    items = list(items)
    if not items:
        return "none"
    head = ", ".join(map(str, items[:limit]))
    return head + (f", … (+{len(items) - limit})" if len(items) > limit else "")


def stability(auc_std: float) -> str:
    return (
        "stable" if auc_std < 0.015
        else "moderate variance" if auc_std < 0.030
        else "high variance"
    )


def baseline_verdict(delta: float, tolerance: float) -> str:
    """Δ ≥ −tol → HOLD; Δ ≥ −2·tol → BELOW TARGET; otherwise REGRESSED."""
    if delta >= -tolerance:
        return "✅ HOLD"
    if delta >= -2 * tolerance:
        return "⚠️ BELOW TARGET"
    return "❌ REGRESSED"


def training_report(
    model_name: str,
    *,
    y: pd.Series,
    n_features: int,
    feature_step: str,
    study: Optional[optuna.Study],
    runtime_s: float,
    tune_fold_scores: list[float],
    params_line: str,
    metrics: dict,
    n_eval_folds: int,
    baseline_auc: Optional[float] = None,
    tolerance: float = 0.005,
    tune_metric: str = "AUC",
) -> None:
    """
    One block summarising a training run:
    data, feature step, tuning, chosen params, out-of-fold metrics, baseline.
    study=None → tuning skipped (fixed params). tune_metric labels study.best_value.
    """
    n_pos = int(y.sum())

    title(model_name)
    print(f"Data       {len(y):,} rows · {n_features} features · {n_pos:,} positive ({n_pos / len(y):.1%})")
    print(f"Features   {feature_step}")
    if study is None:
        print("Tuning     skipped — fixed params")
    else:
        n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        folds = " ".join(f"{s:.3f}" for s in tune_fold_scores)
        print(f"Tuning     {len(study.trials)} trials ({n_complete} complete, {n_pruned} pruned) · {runtime_s:.1f}s")
        print(f"           best {len(tune_fold_scores)}-fold {tune_metric} {study.best_value:.4f} (reproduced ✓) · folds {folds}")
    print(f"Params     {params_line}")
    print(
        f"OOF        {n_eval_folds}-fold AUC {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f} "
        f"({stability(metrics['auc_std'])}) · Recall@10%FPR {metrics['recall_at_10fpr']:.4f}"
    )
    f2 = f" · F2 {metrics['f2_mean']:.4f}" if "f2_mean" in metrics else ""
    print(
        f"           at 0.5: recall {metrics['recall_mean']:.4f} · "
        f"precision {metrics['precision_mean']:.4f} · F1 {metrics['f1_mean']:.4f}{f2}"
    )
    if baseline_auc is not None:
        delta = metrics["auc_mean"] - baseline_auc
        print(f"Baseline   {baseline_auc:.4f} · Δ {delta:+.4f} → {baseline_verdict(delta, tolerance)}")
