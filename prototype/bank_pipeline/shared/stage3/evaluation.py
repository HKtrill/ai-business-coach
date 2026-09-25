"""
shared.stage3.evaluation
========================
Scoring the tuned configuration:

* ``Stage3Metrics``           scalar metrics at one operating point
                              (values from ``shared.metrics.compute_metrics``)
* ``CVReport`` / ``CVEvaluator``         the k-fold reporting block
* ``OOFPredictions`` / ``OOFGenerator``  out-of-fold predictions over train
* ``stage3_metric_suite``     the fixed set of metric blocks stored in an artifact
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shared.metrics import compute_metrics

from .data import FoldFrames, FoldPlan


@dataclass
class Stage3Metrics:
    """Scalar metrics at one operating point. Probabilities are not stored."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    tn: int
    fp: int
    fn: int
    tp: int
    n: int
    base_rate: float
    positive_prediction_rate: float
    coverage: float
    split: str = ""
    probability_space: str = "raw"

    # ------------------------------------------------------------------
    @classmethod
    def compute(
        cls,
        y_true,
        proba: np.ndarray,
        threshold: float,
        *,
        split: str = "",
        probability_space: str = "raw",
        ece_bins: int = 10,
        n_population: Optional[int] = None,
        decision: Optional[np.ndarray] = None,
    ) -> "Stage3Metrics":
        """
        Parameters
        ----------
        n_population
            Rows that existed before Stage 3 selected its population. Stage 3
            currently runs on the full split, so coverage is 1.0; the parameter
            exists so the number stays meaningful if Stage 3 is ever moved
            behind the Stage 2 router.
        decision
            Precomputed 0/1 decisions (e.g. from per-fold thresholds). When
            given, it replaces ``proba >= threshold`` and ``threshold`` is
            recorded as a summary value only.
        """
        y = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        if len(y) != len(p):
            raise ValueError(f"length mismatch: y {len(y)}, proba {len(p)}")
        if len(y) == 0:
            raise ValueError("empty population")

        if decision is None:
            pred = (p >= float(threshold)).astype(int)
        else:
            pred = np.asarray(decision).astype(int)
            if len(pred) != len(y):
                raise ValueError(
                    f"length mismatch: y {len(y)}, decision {len(pred)}"
                )
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n = len(y)

        # The metric values are shared.metrics.compute_metrics — the same
        # function every stage and both arms report through. ROC/PR-AUC are
        # undefined on a single-class slice (possible on a small served
        # subset), so that case is filled with NaN instead of raising.
        if len(np.unique(y)) > 1:
            m = compute_metrics(y, pred, p, float(threshold), probability_space)
        else:
            from shared.metrics import calculate_ece
            m = {"accuracy": float((pred == y).mean()), "precision": float("nan"),
                 "recall": float("nan"), "f1": float("nan"), "f2": float("nan"),
                 "roc_auc": float("nan"), "pr_auc": float("nan"),
                 "brier": float(np.mean((p - y) ** 2)),
                 "ece": calculate_ece(y, p, ece_bins)}

        return cls(
            threshold=float(threshold),
            accuracy=float(m["accuracy"]),
            precision=float(m["precision"]),
            recall=float(m["recall"]),
            f1=float(m["f1"]),
            f2=float(m["f2"]),
            roc_auc=float(m["roc_auc"]),
            pr_auc=float(m["pr_auc"]),
            brier=float(m["brier"]),
            ece=float(m["ece"]),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            n=int(n),
            base_rate=float(y.mean()),
            positive_prediction_rate=float(pred.mean()),
            coverage=float(n / n_population) if n_population else 1.0,
            split=split,
            probability_space=probability_space,
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_glass_keys(self) -> dict:
        """The EBM's ``evaluate_ebm`` dict, key for key."""
        return {
            "Threshold": self.threshold,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "F2": self.f2,
            "ROC-AUC": self.roc_auc,
        }

    def confusion(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.tn, self.fp], [self.fn, self.tp]],
            index=["actual 0", "actual 1"],
            columns=["pred 0", "pred 1"],
        )

    def describe(self) -> str:
        label = f"{self.split} " if self.split else ""
        return (
            f"{label}@ {self.threshold:.4f} ({self.probability_space})  "
            f"acc={self.accuracy:.4f}  prec={self.precision:.4f}  "
            f"rec={self.recall:.4f}  F1={self.f1:.4f}  F2={self.f2:.4f}  "
            f"AUC={self.roc_auc:.4f}  Brier={self.brier:.4f}  "
            f"ECE={self.ece:.4f}  PPR={self.positive_prediction_rate:.4f}"
        )

    @staticmethod
    def to_frame(metrics: list["Stage3Metrics"]) -> pd.DataFrame:
        return pd.DataFrame([m.to_dict() for m in metrics])


@dataclass
class CVReport:
    folds: list[dict] = field(default_factory=list)
    n_folds: int = 0
    random_state: int = 42
    decision_threshold: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)
    feature_fit_scope: str = "per_fold"

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.folds)

    def aggregate(self) -> dict[str, dict[str, float]]:
        df = self.frame()
        metrics = [c for c in df.columns if c not in ("fold", "n_train", "n_val")]
        return {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=0))}
                for m in metrics}

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "feature_fit_scope": self.feature_fit_scope,
            "params": self.params,
            "folds": self.folds,
            "aggregate": self.aggregate(),
        }

    def describe(self) -> str:
        agg = self.aggregate()
        lines = [f"{self.n_folds}-fold cross-validation (at the "
                 f"{self.decision_threshold} cut, features fitted "
                 f"{self.feature_fit_scope}):"]
        for m in ("roc_auc", "recall", "precision", "f1", "f2"):
            if m in agg:
                lines.append(f"   {m:<10} {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")
        return "\n".join(lines)


class CVEvaluator:
    """Refit per reporting fold and score the held-out rows."""

    def __init__(self, estimator, decision_threshold: float = 0.5,
                 random_state: int = 42, feature_fit_scope: str = "per_fold",
                 verbose: bool = True):
        self.estimator = estimator
        self.decision_threshold = float(decision_threshold)
        self.random_state = int(random_state)
        self.feature_fit_scope = feature_fit_scope
        self.verbose = bool(verbose)

    def run(self, folds: list[FoldFrames], params: dict[str, Any]) -> CVReport:
        if self.verbose:
            print(f"\n  {len(folds)}-fold cross-validation "
                  f"(final evaluation, seed {self.random_state})")
        rows: list[dict] = []
        for f in folds:
            model = self.estimator.fit(params, f.X_tr, f.y_tr, sample_weight=f.w_tr)
            y_val = np.asarray(f.y_val).astype(int)
            proba = self.estimator.positive_proba(model, f.X_val)
            pred = (proba >= self.decision_threshold).astype(int)
            row = {
                "fold": f.k + 1,
                "n_train": int(len(f.train_idx)),
                "n_val": int(len(f.val_idx)),
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
                print(f"     fold {row['fold']:2d}: AUC={row['roc_auc']:.4f}  "
                      f"Recall={row['recall']:.4f}  "
                      f"Precision={row['precision']:.4f}  "
                      f"F1={row['f1']:.4f}  F2={row['f2']:.4f}")
        report = CVReport(folds=rows, n_folds=len(folds),
                          random_state=self.random_state,
                          decision_threshold=self.decision_threshold,
                          params=dict(params),
                          feature_fit_scope=self.feature_fit_scope)
        if self.verbose:
            print("\n" + report.describe())
        return report


@dataclass
class OOFPredictions:
    proba: np.ndarray
    fold_id: np.ndarray
    n_folds: int
    random_state: int
    params: dict[str, Any]
    fit_sizes: list[int] = field(default_factory=list)
    test_proba_by_fold: Optional[np.ndarray] = None   # (n_folds, n_test)

    @property
    def n_rows(self) -> int:
        return len(self.proba)

    @property
    def test_proba_fold_mean(self) -> Optional[np.ndarray]:
        if self.test_proba_by_fold is None:
            return None
        return self.test_proba_by_fold.mean(axis=0)

    def assert_valid(self) -> None:
        n = self.n_rows
        if len(self.fold_id) != n:
            raise AssertionError(
                f"OOF length mismatch: proba={n}, fold_id={len(self.fold_id)}")
        if np.isnan(self.proba).any():
            raise AssertionError(
                f"{int(np.isnan(self.proba).sum())} training rows have no "
                "out-of-fold prediction.")
        if ((self.proba < 0.0) | (self.proba > 1.0)).any():
            raise AssertionError("out-of-fold probabilities outside [0, 1].")
        counts = np.bincount(self.fold_id, minlength=self.n_folds)
        if (self.fold_id < 0).any() or len(counts) != self.n_folds \
                or (counts == 0).any() or int(counts.sum()) != n:
            raise AssertionError(f"Fold assignments invalid: {counts.tolist()}")

    def summary(self) -> dict:
        out = {
            "n_rows": self.n_rows,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "fit_sizes": list(self.fit_sizes),
            "proba_mean": float(self.proba.mean()),
            "proba_std": float(self.proba.std()),
            "proba_min": float(self.proba.min()),
            "proba_max": float(self.proba.max()),
        }
        if self.test_proba_by_fold is not None:
            sd = self.test_proba_by_fold.std(axis=0)
            out["test_fold_spread_mean_sd"] = float(sd.mean())
        return out


class OOFGenerator:
    """Refit the tuned configuration once per tuning fold."""

    def __init__(self, estimator, plan: FoldPlan, verbose: bool = False):
        self.estimator = estimator
        self.plan = plan
        self.verbose = bool(verbose)

    def generate(self, folds: list[FoldFrames], params: dict[str, Any],
                 score_test: bool = True) -> OOFPredictions:
        n = self.plan.n_rows
        if len(folds) != self.plan.n_splits:
            raise ValueError("fold frames do not match the fold plan")
        proba = np.full(n, np.nan, dtype=float)
        test_rows: list[np.ndarray] = []
        fit_sizes: list[int] = []
        if self.verbose:
            print(f"\n  out-of-fold predictions: {n:,} rows, {len(folds)} folds")

        for f in folds:
            if np.intersect1d(f.train_idx, f.val_idx).size:
                raise AssertionError(f"fold {f.k}: train/val positions overlap")
            if not np.array_equal(f.val_idx, self.plan.holdout(f.k)):
                raise AssertionError(f"fold {f.k}: frames disagree with plan")
            model = self.estimator.fit(params, f.X_tr, f.y_tr, sample_weight=f.w_tr)
            proba[f.val_idx] = self.estimator.positive_proba(model, f.X_val)
            if score_test:
                if f.X_test is None:
                    raise ValueError(f"fold {f.k} has no X_test frame")
                test_rows.append(self.estimator.positive_proba(model, f.X_test))
            fit_sizes.append(int(len(f.train_idx)))
            if self.verbose:
                print(f"     fold {f.k + 1}/{len(folds)}: fit "
                      f"{len(f.train_idx):,} → scored {len(f.val_idx):,}")
            del model

        result = OOFPredictions(
            proba=proba,
            fold_id=self.plan.fold_id.copy(),
            n_folds=self.plan.n_splits,
            random_state=self.plan.random_state,
            params=dict(params),
            fit_sizes=fit_sizes,
            test_proba_by_fold=np.vstack(test_rows) if score_test else None,
        )
        result.assert_valid()
        return result


def stage3_metric_suite(
    *,
    y_train,
    y_test,
    oof_proba: np.ndarray,
    oof_proba_calibrated: np.ndarray,
    threshold: float,
    threshold_calibrated: float,
    nested: dict,
    test_proba: np.ndarray,
    test_proba_calibrated: np.ndarray,
    fold_mean_proba: np.ndarray,
    fold_mean_proba_calibrated: np.ndarray,
    reference_threshold: float,
    oracle_threshold: float,
    ece_bins: int,
) -> dict:
    """
    The fixed set of metric blocks stored in ``Stage3Artifact.metrics``.

    ``threshold`` / ``threshold_calibrated`` are the OOF-selected operating
    points (raw / calibrated space); ``nested`` is
    ``F2ThresholdSelector.select_nested`` output. Key names and order are part
    of the artifact schema and JSON sidecar.
    """
    y_tr, y_te, t = y_train, y_test, threshold

    def M(y, p, thr, split, space="raw", decision=None):
        return Stage3Metrics.compute(y, p, thr, split=split,
                                     probability_space=space,
                                     ece_bins=ece_bins,
                                     decision=decision).to_dict()

    return {
        "oof_at_operating_threshold":
            M(y_tr, oof_proba, t, "train (OOF, in-selection)"),
        "oof_at_nested_thresholds":
            M(y_tr, oof_proba, nested["mean"],
              "train (OOF, fold-nested thresholds)",
              decision=(oof_proba >= nested["per_row"]).astype(int)),
        "oof_calibrated_at_operating_threshold":
            M(y_tr, oof_proba_calibrated, threshold_calibrated,
              "train (OOF, calibrated)", "calibrated"),
        "test_at_operating_threshold":
            M(y_te, test_proba, t, "test (refit)"),
        "test_at_half":
            M(y_te, test_proba, reference_threshold, "test (refit)"),
        "test_calibrated_at_operating_threshold":
            M(y_te, test_proba_calibrated, threshold_calibrated,
              "test (refit, calibrated)", "calibrated"),
        "test_fold_mean_at_operating_threshold":
            M(y_te, fold_mean_proba, t, "test (fold mean)"),
        "test_fold_mean_calibrated_at_operating_threshold":
            M(y_te, fold_mean_proba_calibrated, threshold_calibrated,
              "test (fold mean, calibrated)", "calibrated"),
        "test_at_oracle_threshold_REFERENCE_ONLY":
            M(y_te, test_proba, oracle_threshold, "test (oracle)"),
    }
