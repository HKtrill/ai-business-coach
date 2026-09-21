"""
blackbox_pipeline.models.xgb.evaluation.cv
===========================================
The k-fold reporting block.

Mirrors the 10-fold evaluation the EBM runs after Optuna finishes: same fold
count, same seed, same balanced weights, same five headline metrics measured at
the 0.5 cut. It is a report, not a selection step — nothing downstream depends
on its output, and it never sees the test split.

Note this partition is deliberately NOT the ``FoldPlan`` used for tuning and
OOF. The EBM reports on 10 folds while tuning on 5, and the black-box arm keeps
that asymmetry so the two reports are the same statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..estimator import XGBFactory
from ..folds import FoldPlan
from ..weights import BalancedWeights

__all__ = ["CVReport", "CVEvaluator"]


@dataclass
class CVReport:
    """Per-fold and aggregate scores."""

    folds: list[dict] = field(default_factory=list)
    n_folds: int = 0
    random_state: int = 42
    decision_threshold: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.folds)

    def aggregate(self) -> dict[str, dict[str, float]]:
        df = self.frame()
        metrics = [c for c in df.columns if c not in ("fold", "n_train", "n_val")]
        return {
            m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=0))}
            for m in metrics
        }

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "params": self.params,
            "folds": self.folds,
            "aggregate": self.aggregate(),
        }

    def describe(self) -> str:
        agg = self.aggregate()
        order = ["roc_auc", "recall", "precision", "f1", "f2"]
        lines = [f"{self.n_folds}-fold cross-validation (at the "
                 f"{self.decision_threshold} cut):"]
        for m in order:
            if m in agg:
                lines.append(
                    f"   {m:<10} {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}"
                )
        return "\n".join(lines)


class CVEvaluator:
    """Refit per fold and score the held-out rows."""

    def __init__(
        self,
        factory: XGBFactory,
        weights: BalancedWeights,
        n_folds: int = 10,
        random_state: int = 42,
        stratify: bool = True,
        decision_threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.factory = factory
        self.weights = weights
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.stratify = bool(stratify)
        self.decision_threshold = float(decision_threshold)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    def run(
        self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]
    ) -> CVReport:
        self.weights.assert_matches(len(X), "cv evaluation")
        plan = FoldPlan.build(
            y, n_splits=self.n_folds, random_state=self.random_state,
            stratify=self.stratify, purpose="reporting",
        )

        if self.verbose:
            print(f"\n  {self.n_folds}-fold cross-validation "
                  f"(final evaluation, seed {self.random_state})")

        rows: list[dict] = []
        for k, (train_idx, val_idx) in enumerate(plan, start=1):
            model = self.factory.fit(
                params, X.iloc[train_idx], y.iloc[train_idx],
                sample_weight=self.weights.for_rows(train_idx),
            )
            y_val = np.asarray(y.iloc[val_idx]).astype(int)
            proba = XGBFactory.positive_proba(model, X.iloc[val_idx])
            pred = (proba >= self.decision_threshold).astype(int)

            row = {
                "fold": k,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "roc_auc": float(roc_auc_score(y_val, proba))
                           if len(np.unique(y_val)) > 1 else float("nan"),
                "recall": float(recall_score(y_val, pred, zero_division=0)),
                "precision": float(precision_score(y_val, pred, zero_division=0)),
                "f1": float(f1_score(y_val, pred, zero_division=0)),
                "f2": float(fbeta_score(y_val, pred, beta=2, zero_division=0)),
            }
            rows.append(row)
            del model

            if self.verbose:
                print(f"     fold {k:2d}: AUC={row['roc_auc']:.4f}  "
                      f"Recall={row['recall']:.4f}  "
                      f"Precision={row['precision']:.4f}  "
                      f"F1={row['f1']:.4f}  F2={row['f2']:.4f}")

        report = CVReport(
            folds=rows,
            n_folds=self.n_folds,
            random_state=self.random_state,
            decision_threshold=self.decision_threshold,
            params=dict(params),
        )
        if self.verbose:
            print("\n" + report.describe())
        return report