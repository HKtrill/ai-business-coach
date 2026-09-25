"""
blackbox_pipeline.models.xgb.evaluation.compare
================================================
EBM vs XGBoost at the same Stage 3 operating role.

Generic classifier metrics answer "which model scores better". They do not
answer the question PR 3 asks, because the two arms currently choose their
operating points by different rules — the EBM fits its threshold on the test
labels, the black-box arm on out-of-fold training predictions. Comparing
recall at each arm's own threshold would partly measure that protocol
difference rather than the model family.

``Stage3Comparison`` therefore reports three views:

``ranking``
    Threshold-free: ROC-AUC, Brier, ECE (GLASS-parity ``ece`` plus every-row
    ``ece_full``). Unaffected by any threshold rule, so
    this is the cleanest read on whether removing the additivity constraint
    improved the underlying score.

``matched_budget``
    Both arms' thresholds are set so they flag the SAME fraction of the
    population, then recall and precision are compared. This is the operating
    role Stage 3 actually plays — "given we will act on N customers, which
    model finds more subscribers" — and it is immune to threshold protocol.

``as_operated``
    Each arm at its own stored threshold, which is what the cascade would
    really do. Reported last and with the protocol difference stated, so a
    reader cannot mistake it for a like-for-like number.

Plus ``agreement``: where the two arms disagree, and who is right when they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..calibration import calculate_ece
from .metrics import Stage3Metrics

__all__ = ["ArmScores", "Stage3Comparison"]


@dataclass
class ArmScores:
    """One arm's test-split output, normalised across the two artifact schemas."""

    name: str
    proba: np.ndarray
    threshold: float
    threshold_source: str

    @classmethod
    def from_glass_artifact(
        cls,
        artifact: dict,
        name: str = "EBM (GLASS)",
        calibrated: bool = False,
        threshold: Optional[float] = None,
        threshold_source: Optional[str] = None,
    ) -> "ArmScores":
        """
        GLASS artifact dict → arm.

        The GLASS artifact stores one ``optimal_threshold`` without saying
        which probability space it was fitted in. So:

        * ``calibrated=False`` uses it on raw test probabilities (default).
        * ``calibrated=True`` REQUIRES an explicit ``threshold`` — the stored
          one is not silently applied to calibrated probabilities.

        ``threshold`` also overrides the raw case (e.g. a threshold derived
        under a different protocol). Threshold-free views are unaffected.
        """
        key = "test_predictions_calibrated" if calibrated else "test_predictions"
        if key not in artifact:
            raise KeyError(f"GLASS artifact has no '{key}'")
        if threshold is None:
            if calibrated:
                raise ValueError(
                    "calibrated=True: the GLASS artifact's optimal_threshold "
                    "has no recorded probability space. Pass threshold= "
                    "explicitly (fitted on calibrated probabilities)."
                )
            threshold = float(artifact["optimal_threshold"])
            source = "fitted on the TEST split (GLASS protocol)"
        else:
            source = threshold_source or "supplied explicitly"
        return cls(
            name=name,
            proba=np.asarray(artifact[key], dtype=float),
            threshold=float(threshold),
            threshold_source=source,
        )

    @classmethod
    def from_xgb_artifact(
        cls, artifact, name: str = "XGBoost", calibrated: bool = False
    ) -> "ArmScores":
        """
        ``Stage3Artifact`` (or equivalent mapping) → arm.

        Uses the refit model's test probabilities with the matching OOF
        threshold: raw ↔ ``threshold["oof"]``, calibrated ↔
        ``threshold["oof_calibrated_space"]``.
        """
        refit = (artifact["refit"] if isinstance(artifact, dict)
                 else artifact.refit)
        thresholds = (artifact["threshold"] if isinstance(artifact, dict)
                      else artifact.threshold)
        key = "test_proba_calibrated" if calibrated else "test_proba"
        band = "oof_calibrated_space" if calibrated else "oof"
        return cls(
            name=name,
            proba=np.asarray(refit[key], dtype=float),
            threshold=float(thresholds[band]["threshold"]),
            threshold_source="fitted on out-of-fold TRAIN predictions",
        )


class Stage3Comparison:
    """Builds the three comparison views plus the agreement breakdown."""

    def __init__(self, y_test, arms: list[ArmScores], ece_bins: int = 10):
        self.y = np.asarray(y_test).astype(int)
        self.arms = list(arms)
        self.ece_bins = int(ece_bins)
        for arm in self.arms:
            if len(arm.proba) != len(self.y):
                raise ValueError(
                    f"{arm.name}: {len(arm.proba)} probabilities for "
                    f"{len(self.y)} test labels"
                )

    # ------------------------------------------------------------------
    def ranking(self) -> pd.DataFrame:
        """Threshold-free quality."""
        from sklearn.metrics import brier_score_loss, roc_auc_score
        rows = []
        for arm in self.arms:
            rows.append({
                "arm": arm.name,
                "roc_auc": float(roc_auc_score(self.y, arm.proba)),
                "brier": float(brier_score_loss(self.y, arm.proba)),
                "ece": calculate_ece(self.y, arm.proba, self.ece_bins),
                "ece_full": calculate_ece(self.y, arm.proba, self.ece_bins,
                                          include_zero=True),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def matched_budget(
        self, rates: Optional[list[float]] = None
    ) -> pd.DataFrame:
        """
        Both arms flag the same fraction of the population.

        The threshold for each arm is its own empirical ``1 - rate`` quantile,
        so the comparison holds the action budget fixed and lets recall vary —
        the question a campaign owner actually asks. Ties at the cut can flag
        slightly more than ``rate``; ``n_flagged`` reports the actual count.
        """
        if rates is None:
            rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        rows = []
        for rate in rates:
            for arm in self.arms:
                cut = float(np.quantile(arm.proba, 1.0 - rate))
                pred = (arm.proba >= cut).astype(int)
                tp = int(((pred == 1) & (self.y == 1)).sum())
                flagged = int(pred.sum())
                rows.append({
                    "budget_rate": rate,
                    "arm": arm.name,
                    "threshold": cut,
                    "n_flagged": flagged,
                    "recall": float(tp / self.y.sum()) if self.y.sum() else 0.0,
                    "precision": float(tp / flagged) if flagged else 0.0,
                    "lift": float((tp / flagged) / self.y.mean())
                            if flagged and self.y.mean() else float("nan"),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def as_operated(self) -> pd.DataFrame:
        """Each arm at its own stored threshold. Protocols differ — see column."""
        rows = []
        for arm in self.arms:
            m = Stage3Metrics.compute(
                self.y, arm.proba, arm.threshold,
                split="test", ece_bins=self.ece_bins,
            )
            row = {"arm": arm.name, "threshold_source": arm.threshold_source}
            row.update(m.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def agreement(self) -> dict:
        """Pairwise decision agreement at each arm's own threshold."""
        if len(self.arms) != 2:
            raise ValueError("agreement() compares exactly two arms")
        a, b = self.arms
        pa = (a.proba >= a.threshold).astype(int)
        pb = (b.proba >= b.threshold).astype(int)

        both = (pa == 1) & (pb == 1)
        only_a = (pa == 1) & (pb == 0)
        only_b = (pa == 0) & (pb == 1)
        neither = (pa == 0) & (pb == 0)

        def block(mask):
            n = int(mask.sum())
            return {
                "n": n,
                "n_positive": int(self.y[mask].sum()),
                "positive_rate": float(self.y[mask].mean()) if n else float("nan"),
            }

        return {
            "arms": [a.name, b.name],
            "agreement_rate": float((pa == pb).mean()),
            "both_flag": block(both),
            f"only_{a.name}": block(only_a),
            f"only_{b.name}": block(only_b),
            "neither_flags": block(neither),
        }

    # ------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        out = {
            "ranking": self.ranking(),
            "matched_budget": self.matched_budget(),
            "as_operated": self.as_operated(),
        }
        if len(self.arms) == 2:
            out["agreement"] = self.agreement()
        return out

    def print_report(self) -> None:
        r = self.report()
        print("=" * 78)
        print("  STAGE 3 — EBM vs XGBOOST")
        print("=" * 78)
        print("\n  threshold-free (unaffected by either threshold protocol)")
        print(r["ranking"].to_string(index=False))
        print("\n  matched action budget (both arms flag the same fraction)")
        print(r["matched_budget"].to_string(index=False))
        print("\n  as operated (NOTE: the arms select thresholds differently)")
        print(r["as_operated"][
            ["arm", "threshold", "threshold_source", "recall", "precision", "f2"]
        ].to_string(index=False))
        if "agreement" in r:
            print("\n  decision agreement")
            for k, v in r["agreement"].items():
                print(f"     {k}: {v}")
        print("=" * 78)