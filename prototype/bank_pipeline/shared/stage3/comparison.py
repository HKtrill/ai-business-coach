"""
shared.stage3.comparison
========================
EBM-vs-XGBoost comparison on the test split: ``ArmScores`` (one arm's
output) and ``Stage3Comparison`` (ranking, matched budget, as-operated,
served-population and agreement reports).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .calibration import calculate_ece
from .evaluation import Stage3Metrics


@dataclass
class ArmScores:
    """One arm's test-split output."""

    name: str
    proba: np.ndarray
    threshold: float
    threshold_source: str
    proba_calibrated: Optional[np.ndarray] = None
    index_test: Optional[list] = None

    @classmethod
    def from_artifact(cls, artifact, name: Optional[str] = None,
                      source: str = "refit") -> "ArmScores":
        """Any ``Stage3Artifact`` (EBM or XGBoost)."""
        b = artifact.refit if source == "refit" else artifact.fold_mean
        return cls(
            name=name or artifact.model_family,
            proba=np.asarray(b["test_proba"], dtype=float),
            threshold=float(artifact.threshold["oof"]["threshold"]),
            threshold_source="fitted on out-of-fold TRAIN predictions",
            proba_calibrated=np.asarray(b["test_proba_calibrated"], dtype=float),
            index_test=list(artifact.index_test),
        )

    # Backwards-compatible alias.
    from_xgb_artifact = from_artifact

    @classmethod
    def from_glass_artifact(cls, artifact, name: str = "EBM (GLASS)",
                            threshold: Optional[float] = None,
                            threshold_source: Optional[str] = None) -> "ArmScores":
        """
        A new ``Stage3Artifact`` → ``from_artifact``. A pre-PR-33 GLASS dict is
        refused unless an OOF-derived ``threshold`` is supplied, because its
        stored threshold was fitted on the test labels.
        """
        if not isinstance(artifact, dict):
            return cls.from_artifact(artifact, name=name)
        if threshold is None:
            raise ValueError(
                "Pre-PR-33 GLASS artifact: its optimal_threshold was fitted on "
                "y_test. Re-run GLASS Stage 3, or pass threshold= explicitly."
            )
        return cls(
            name=name,
            proba=np.asarray(artifact["test_predictions"], dtype=float),
            threshold=float(threshold),
            threshold_source=threshold_source or "supplied explicitly",
            proba_calibrated=np.asarray(artifact["test_predictions_calibrated"],
                                        dtype=float),
        )


class Stage3Comparison:
    def __init__(self, y_test, arms: list[ArmScores], ece_bins: int = 10):
        self.y = np.asarray(y_test).astype(int)
        self.arms = list(arms)
        self.ece_bins = int(ece_bins)
        idx = [a.index_test for a in self.arms if a.index_test is not None]
        for i in idx[1:]:
            if list(i) != list(idx[0]):
                raise ValueError("arms were scored on different test indexes")
        for arm in self.arms:
            if len(arm.proba) != len(self.y):
                raise ValueError(f"{arm.name}: {len(arm.proba)} probabilities "
                                 f"for {len(self.y)} test labels")

    # ------------------------------------------------------------------
    def ranking(self) -> pd.DataFrame:
        rows = []
        for a in self.arms:
            row = {
                "arm": a.name,
                "roc_auc": float(roc_auc_score(self.y, a.proba)),
                "pr_auc": float(average_precision_score(self.y, a.proba)),
                "brier_calibrated": float("nan"),
                "ece_calibrated": float("nan"),
                "brier_raw (weighting-distorted)":
                    float(brier_score_loss(self.y, a.proba)),
                "ece_raw (weighting-distorted)":
                    calculate_ece(self.y, a.proba, self.ece_bins),
            }
            if a.proba_calibrated is not None:
                pc = a.proba_calibrated
                row["brier_calibrated"] = float(brier_score_loss(self.y, pc))
                row["ece_calibrated"] = calculate_ece(self.y, pc, self.ece_bins)
            rows.append(row)
        return pd.DataFrame(rows)

    def matched_budget(self, rates: Optional[list[float]] = None) -> pd.DataFrame:
        rates = rates or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        rows = []
        for rate in rates:
            for a in self.arms:
                cut = float(np.quantile(a.proba, 1.0 - rate))
                pred = (a.proba >= cut).astype(int)
                tp = int(((pred == 1) & (self.y == 1)).sum())
                flagged = int(pred.sum())
                rows.append({
                    "budget_rate": rate, "arm": a.name, "threshold": cut,
                    "n_flagged": flagged,
                    "recall": float(tp / self.y.sum()) if self.y.sum() else 0.0,
                    "precision": float(tp / flagged) if flagged else 0.0,
                    "lift": float((tp / flagged) / self.y.mean())
                            if flagged and self.y.mean() else float("nan"),
                })
        return pd.DataFrame(rows)

    def as_operated(self) -> pd.DataFrame:
        rows = []
        for a in self.arms:
            m = Stage3Metrics.compute(self.y, a.proba, a.threshold,
                                      split="test", ece_bins=self.ece_bins)
            rows.append({"arm": a.name, "threshold_source": a.threshold_source,
                         **m.to_dict()})
        return pd.DataFrame(rows)

    def served(self, masks: dict[str, Any]) -> pd.DataFrame:
        """
        ``masks``: {arm name: boolean test mask of rows that arm's Stage 2
        defers}. Rows: each arm on its own served set, then every arm on the
        intersection of all masks.
        """
        n = len(self.y)
        clean = {k: np.asarray(v, dtype=bool) for k, v in masks.items()}
        for k, v in clean.items():
            if len(v) != n:
                raise ValueError(f"mask {k!r}: {len(v)} rows, test has {n}")
        rows = []
        for a in self.arms:
            if a.name not in clean:
                continue
            m = clean[a.name]
            rows.append(self._served_row(a, m, "own Stage 2 deferrals"))
        inter = np.logical_and.reduce(list(clean.values())) if clean else None
        if inter is not None and inter.any():
            for a in self.arms:
                rows.append(self._served_row(a, inter, "intersection of deferrals"))
        return pd.DataFrame(rows)

    def _served_row(self, a: ArmScores, mask: np.ndarray, label: str) -> dict:
        m = Stage3Metrics.compute(self.y[mask], a.proba[mask], a.threshold,
                                  split=label, ece_bins=self.ece_bins,
                                  n_population=len(self.y))
        return {"arm": a.name, "population": label, "n": m.n,
                "coverage": m.coverage, "base_rate": m.base_rate,
                "recall": m.recall, "precision": m.precision, "f2": m.f2,
                "roc_auc": m.roc_auc}

    def agreement(self) -> dict:
        if len(self.arms) != 2:
            raise ValueError("agreement() compares exactly two arms")
        a, b = self.arms
        pa = (a.proba >= a.threshold).astype(int)
        pb = (b.proba >= b.threshold).astype(int)

        def block(mask):
            n = int(mask.sum())
            return {"n": n, "n_positive": int(self.y[mask].sum()),
                    "positive_rate": float(self.y[mask].mean()) if n else float("nan")}

        return {
            "arms": [a.name, b.name],
            "agreement_rate": float((pa == pb).mean()),
            "both_flag": block((pa == 1) & (pb == 1)),
            f"only_{a.name}": block((pa == 1) & (pb == 0)),
            f"only_{b.name}": block((pa == 0) & (pb == 1)),
            "neither_flags": block((pa == 0) & (pb == 0)),
        }

    def report(self) -> dict[str, Any]:
        out = {"ranking": self.ranking(), "matched_budget": self.matched_budget(),
               "as_operated": self.as_operated()}
        if len(self.arms) == 2:
            out["agreement"] = self.agreement()
        return out

    def print_report(self) -> None:
        r = self.report()
        print("=" * 78)
        print("  STAGE 3 — EBM vs XGBOOST (shared protocol)")
        print("=" * 78)
        print("\n  threshold-free (AUCs on raw; Brier/ECE on calibrated)")
        print(r["ranking"].to_string(index=False))
        print("\n  matched action budget (both arms flag the same fraction)")
        print(r["matched_budget"].to_string(index=False))
        print("\n  as operated (both thresholds selected on OOF train predictions)")
        print(r["as_operated"][["arm", "threshold", "threshold_source",
                                "recall", "precision", "f2"]].to_string(index=False))
        if "agreement" in r:
            print("\n  decision agreement")
            for k, v in r["agreement"].items():
                print(f"     {k}: {v}")
        print("=" * 78)
