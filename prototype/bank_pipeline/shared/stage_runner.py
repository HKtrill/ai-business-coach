"""
shared.stage_runner
===================
The Stage 3 pipeline both arms run (GLASS EBM and black-box XGBoost), in one
file. Each arm supplies only its estimator factory, search space and config
subclass; everything else below is shared, so the arms are comparable by
construction.

Sections, in dependency order: config, folds, weights, features (contract +
cleaner), calibration, metrics, estimator interface, per-fold feature block,
thresholds, tuning, CV report, OOF, artifact, runner, protocol checks, arm
comparison.

Uses the existing shared modules rather than re-implementing them:
``shared.stage_io.fold_assignment`` / ``StageOutput``,
``shared.thresholds.sweep_f_beta``, ``shared.metrics.compute_metrics`` /
``calculate_ece``. Nothing here imports from glass_pipeline or
blackbox_pipeline.

Not loaded by ``import shared`` (it needs optuna); import it explicitly::

    from shared.stage_runner import Stage3Block, Stage3Runner
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import (Any, Callable, ClassVar, Iterable, Iterator, Optional,
                    Protocol, Tuple, runtime_checkable)

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from shared.metrics import calculate_ece as _shared_ece
from shared.metrics import compute_metrics
from shared.stage_io import fold_assignment
from shared.thresholds import sweep_f_beta


# ==========================================================================
# config
# ==========================================================================

FEATURE_FIT_SCOPES = ("per_fold", "global")

PROTOCOL_FIELDS: tuple[str, ...] = (
    "stage",
    "population",
    "random_state",
    "n_jobs",
    "n_tune_folds",
    "n_eval_folds",
    "stratify",
    "feature_fit_scope",
    "n_trials",
    "pruner_startup_trials",
    "pruner_warmup_steps",
    "beta",
    "tuning_decision_threshold",
    "threshold_low",
    "threshold_high",
    "threshold_steps",
    "reference_threshold",
    "nested_oof_threshold",
    "calibration_method",
    "ece_threshold",
    "ece_bins",
    "force_calibration",
    "class_weight",
    "clean_inf",
    "impute_missing",
    "expected_features",
)

@dataclass
class Stage3Config:
    """
    Shared Stage 3 protocol. Subclasses set the family-specific fields.

    Feature-fit scope
    -----------------
    ``feature_fit_scope="per_fold"`` (default) refits the target-dependent
    feature engineers inside every fold, so a validation row's engineered
    features never depend on its own label. This makes the OOF predictions
    Stage 4 trains on leakage-safe at the FEATURE level as well as the model
    level. ``"global"`` reproduces the pre-PR-33 behaviour (engineers fitted
    once on all of train) and exists only for before/after comparison.

    Population
    ----------
    ``population="full_train"``: Stage 3 trains on every training row. The rows
    Stage 3 actually SERVES in the cascade (those Stage 2 defers) are a subset;
    they are evaluated with ``Stage3Artifact.evaluate_served``.
    """

    # ---- experiment identity (family-specific; overridden) ---------------
    stage: str = "stage3"
    model_family: str = "unset"
    arm: str = "unset"                   # StageOutput arm: "glass" / "blackbox"
    counterpart: str = ""
    column_prefix: str = "stage3_"
    population: str = "full_train"

    # ---- reproducibility --------------------------------------------------
    random_state: int = 42
    n_jobs: int = 1

    # ---- cross-validation -------------------------------------------------
    # One 5-fold partition drives tuning, OOF, calibration and threshold
    # selection; a separate 10-fold partition drives the reporting block.
    n_tune_folds: int = 5
    n_eval_folds: int = 10
    stratify: bool = True
    feature_fit_scope: str = "per_fold"

    # ---- tuning -----------------------------------------------------------
    n_trials: int = 150
    pruner_startup_trials: int = 20
    pruner_warmup_steps: int = 2
    study_name: str = "stage3_recall_biased"
    search_space: Any = None            # family-specific dataclass

    # ---- objective --------------------------------------------------------
    # F2 on the 0.5 cut (what EBM ``model.predict`` scores).
    beta: float = 2.0
    tuning_decision_threshold: float = 0.5

    # ---- threshold selection ---------------------------------------------
    # F2 grid, scored on OUT-OF-FOLD training predictions — never on test.
    threshold_low: float = 0.10
    threshold_high: float = 0.90
    threshold_steps: int = 81
    reference_threshold: float = 0.5
    nested_oof_threshold: bool = True

    # ---- calibration ------------------------------------------------------
    # Isotonic, gated at OOF ECE > 0.05, fold-nested on the training side.
    calibration_method: str = "isotonic"
    ece_threshold: float = 0.05
    ece_bins: int = 10
    force_calibration: bool = False

    # ---- class imbalance --------------------------------------------------
    class_weight: str = "balanced"

    # ---- data hygiene -----------------------------------------------------
    clean_inf: bool = True
    impute_missing: bool = True

    # ---- artifacts --------------------------------------------------------
    artifact_dir: str = "models/stage3"
    artifact_stem: str = "stage3"
    timestamped: bool = True
    write_latest_pointer: bool = True

    # ---- reporting --------------------------------------------------------
    verbose: bool = True
    show_progress_bar: bool = True

    # ---- feature contract -------------------------------------------------
    expected_features: Optional[list[str]] = None

    # Subclasses set this so from_dict can rebuild the search space.
    _space_cls: ClassVar[Optional[type]] = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.n_tune_folds < 2:
            errors.append("n_tune_folds must be >= 2")
        if self.n_eval_folds < 2:
            errors.append("n_eval_folds must be >= 2")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        if not 0.0 < self.threshold_low < self.threshold_high < 1.0:
            errors.append(
                "require 0 < threshold_low < threshold_high < 1, got "
                f"{self.threshold_low} / {self.threshold_high}"
            )
        if self.threshold_steps < 2:
            errors.append("threshold_steps must be >= 2")
        if self.beta <= 0:
            errors.append("beta must be > 0")
        if not 0.0 <= self.ece_threshold <= 1.0:
            errors.append("ece_threshold must be in [0, 1]")
        if self.ece_bins < 2:
            errors.append("ece_bins must be >= 2")
        if self.calibration_method not in ("isotonic", "sigmoid"):
            errors.append(
                f"calibration_method must be 'isotonic' or 'sigmoid', "
                f"got {self.calibration_method!r}"
            )
        if self.feature_fit_scope not in FEATURE_FIT_SCOPES:
            errors.append(
                f"feature_fit_scope must be one of {FEATURE_FIT_SCOPES}, "
                f"got {self.feature_fit_scope!r}"
            )
        if self.population != "full_train":
            errors.append(
                "population: only 'full_train' is implemented; the served "
                "subset is evaluated via Stage3Artifact.evaluate_served"
            )
        if errors:
            raise ValueError(
                f"{type(self).__name__} validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        space = out.get("search_space")
        out["search_space"] = space.to_dict() if hasattr(space, "to_dict") else space
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Stage3Config":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in d.items() if k in known}
        space = payload.get("search_space")
        if isinstance(space, dict) and cls._space_cls is not None:
            payload["search_space"] = cls._space_cls(**space)
        return cls(**payload)

    def protocol(self) -> dict[str, Any]:
        """The fields that must match across arms, plus the search space."""
        out = {k: getattr(self, k) for k in PROTOCOL_FIELDS}
        out["search_space"] = (
            self.search_space.to_dict()
            if hasattr(self.search_space, "to_dict") else self.search_space
        )
        return out

    def mirror_report(self) -> dict[str, Any]:
        """Human-readable protocol statement embedded in every artifact."""
        return {
            "model_family": self.model_family,
            "counterpart": self.counterpart,
            "population": self.population,
            "random_state": self.random_state,
            "n_tune_folds": self.n_tune_folds,
            "n_eval_folds": self.n_eval_folds,
            "feature_fit_scope": self.feature_fit_scope,
            "n_trials": self.n_trials,
            "objective": f"F{self.beta:g} at decision threshold "
                         f"{self.tuning_decision_threshold}",
            "class_weight": self.class_weight,
            "calibration": f"{self.calibration_method}, gated at OOF ECE > "
                           f"{self.ece_threshold}, fold-nested on train",
            "threshold_rule": (
                f"F{self.beta:g}-maximising over "
                f"[{self.threshold_low}, {self.threshold_high}] in "
                f"{self.threshold_steps} steps, scored on OUT-OF-FOLD train "
                f"predictions"
                + ("; train-side decisions use fold-nested thresholds"
                   if self.nested_oof_threshold else "")
            ),
            "shared_implementation": "shared.stage_runner (both arms)",
            "family_specific": [
                "estimator",
                "hyperparameter search space",
            ],
        }


# ==========================================================================
# folds
# ==========================================================================

@dataclass(frozen=True)
class FoldPlan:
    """A frozen fold assignment over ``range(n)``."""

    fold_id: np.ndarray
    n_splits: int
    random_state: int
    stratified: bool
    purpose: str = "tuning+oof"

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        y,
        n_splits: int,
        random_state: int,
        stratify: bool = True,
        purpose: str = "tuning+oof",
    ) -> "FoldPlan":
        y_arr = np.asarray(y)
        n = len(y_arr)
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_splits > n:
            raise ValueError(
                f"n_splits={n_splits} exceeds the population size {n}"
            )
        if stratify:
            counts = np.bincount(y_arr.astype(int))
            smallest = int(counts[counts > 0].min())
            if smallest < n_splits:
                raise ValueError(
                    f"Stratified {n_splits}-fold needs at least {n_splits} "
                    f"samples in every class; the smallest has {smallest}."
                )

        if not stratify:
            raise ValueError(
                "Stage 3 folds come from shared.stage_io.fold_assignment, "
                "which is stratified (as in Stages 1–2). stratify=False is "
                "not supported."
            )
        # One fold splitter for the whole cascade: the same function Stages
        # 1–2 use, so fold ids are identical whenever n_splits and seed match.
        fold_id = fold_assignment(
            y_arr, cv_folds=n_splits, random_state=random_state
        ).to_numpy(dtype=int)

        plan = cls(
            fold_id=fold_id, n_splits=n_splits, random_state=random_state,
            stratified=bool(stratify), purpose=purpose,
        )
        plan.assert_valid()
        return plan

    # ------------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.fold_id)

    def splits(self) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Positional ``(train_idx, val_idx)`` pairs, derived from fold_id."""
        return list(self)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for k in range(self.n_splits):
            val = np.flatnonzero(self.fold_id == k)
            train = np.flatnonzero(self.fold_id != k)
            yield train, val

    def holdout(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.fold_id == k)

    # ------------------------------------------------------------------
    def assert_valid(self) -> None:
        if (self.fold_id < 0).any():
            raise AssertionError(
                f"{int((self.fold_id < 0).sum())} rows carry no fold assignment."
            )
        counts = np.bincount(self.fold_id, minlength=self.n_splits)
        if len(counts) != self.n_splits or (counts == 0).any():
            raise AssertionError(
                f"Expected {self.n_splits} non-empty folds, got {counts.tolist()}"
            )
        if int(counts.sum()) != self.n_rows:
            raise AssertionError("Folds do not partition the population.")

    def assert_matches(self, n_rows: int, context: str = "") -> None:
        """Guard against a fold plan being reused against the wrong frame."""
        if n_rows != self.n_rows:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"FoldPlan covers {self.n_rows} rows but was handed "
                f"{n_rows}{where}. The plan is built from the training split "
                "and must not be reused across populations."
            )

    # ------------------------------------------------------------------
    def class_balance(self, y) -> list[dict]:
        """Per-fold positive rate — printed at fit time, stored in the artifact."""
        y_arr = np.asarray(y).astype(int)
        rows = []
        for k, (train, val) in enumerate(self):
            rows.append({
                "fold": k,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "val_positive_rate": float(y_arr[val].mean()),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "stratified": self.stratified,
            "purpose": self.purpose,
            "n_rows": self.n_rows,
            "fold_id": self.fold_id.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldPlan":
        return cls(
            fold_id=np.asarray(d["fold_id"], dtype=int),
            n_splits=int(d["n_splits"]),
            random_state=int(d["random_state"]),
            stratified=bool(d["stratified"]),
            purpose=d.get("purpose", "tuning+oof"),
        )


# ==========================================================================
# weights
# ==========================================================================

@dataclass
class BalancedWeights:
    """Per-row sample weights, computed once and sliced positionally."""

    vector: np.ndarray
    strategy: str
    positive_weight: float
    negative_weight: float
    ratio: float
    n_rows: int

    # ------------------------------------------------------------------
    @classmethod
    def balanced(cls, y, strategy: str = "balanced") -> "BalancedWeights":
        y_arr = np.asarray(y).astype(int)
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        if y_arr.min() == y_arr.max():
            raise ValueError("y is single-class; cannot balance")

        vector = np.asarray(compute_sample_weight(strategy, y_arr), dtype=float)
        pos = float(vector[y_arr == 1].mean())
        neg = float(vector[y_arr == 0].mean())
        return cls(
            vector=vector,
            strategy=strategy,
            positive_weight=pos,
            negative_weight=neg,
            ratio=float(pos / neg) if neg else float("nan"),
            n_rows=len(y_arr),
        )

    # ------------------------------------------------------------------
    def for_rows(self, idx: np.ndarray) -> np.ndarray:
        """Slice by positional index — the EBM's ``sample_weights[tr_idx]``."""
        return self.vector[np.asarray(idx, dtype=int)]

    def assert_matches(self, n_rows: int, context: str = "") -> None:
        if n_rows != self.n_rows:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"Weights cover {self.n_rows} rows but were handed "
                f"{n_rows}{where}."
            )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "ratio": self.ratio,
            "n_rows": self.n_rows,
        }

    def describe(self) -> str:
        return (
            f"balanced sample weights: class 0 → {self.negative_weight:.4f}, "
            f"class 1 → {self.positive_weight:.4f} "
            f"(positive class upweighted {self.ratio:.1f}×)"
        )


# ==========================================================================
# features
# ==========================================================================

class Stage3FeatureContract:
    """
    Asserts that the Stage 3 input matches the EBM's feature block.

    Parameters
    ----------
    expected_features
        The column list the EBM consumed — pass
        ``glass_pipeline.ebm.feature_engineering.EBM_FEATURES``. When ``None``
        the contract only checks train/test agreement, which is weaker; the
        stage warns in that case.
    """

    def __init__(self, expected_features: Optional[list[str]] = None):
        self.expected_features = (
            None if expected_features is None else list(expected_features)
        )

    # ------------------------------------------------------------------
    def validate(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
        """
        Return the agreed column order, or raise explaining the mismatch.

        Order is ``expected_features`` when set, else ``X_train``'s order.
        Raises on non-DataFrame input, duplicate columns, train/test set
        mismatch, or any missing / extra column vs. the contract.
        """
        for name, frame in (("X_train", X_train), ("X_test", X_test)):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"{name} must be a DataFrame, got {type(frame).__name__}"
                )

        train_cols, test_cols = list(X_train.columns), list(X_test.columns)

        for name, cols in (("X_train", train_cols), ("X_test", test_cols)):
            dupes = sorted({c for c in cols if cols.count(c) > 1})
            if dupes:
                raise ValueError(f"{name} has duplicate columns: {dupes}")

        if set(train_cols) != set(test_cols):
            only_train = sorted(set(train_cols) - set(test_cols))
            only_test = sorted(set(test_cols) - set(train_cols))
            raise ValueError(
                "Stage 3 train/test columns differ.\n"
                f"  only in train: {only_train}\n"
                f"  only in test : {only_test}"
            )

        if self.expected_features is None:
            return train_cols

        expected = self.expected_features
        missing = [c for c in expected if c not in train_cols]
        extra = [c for c in train_cols if c not in expected]
        if missing or extra:
            raise ValueError(
                "Stage 3 input does not match the EBM feature contract — the "
                "two arms would not be seeing the same information.\n"
                f"  expected {len(expected)} columns from EBM_FEATURES\n"
                f"  missing  : {missing}\n"
                f"  unexpected: {extra}\n"
                "Re-run the GLASS engineering DAG (drop_leaky_features -> "
                "engineer_ebm_features -> select_ebm_features) and pass its "
                "output, or update config.expected_features deliberately."
            )
        return list(expected)

    # ------------------------------------------------------------------
    def align(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Validate, then return both frames in the contracted column order."""
        order = self.validate(X_train, X_test)
        return X_train[order], X_test[order], order

@dataclass
class FeatureCleaner:
    """
    inf -> NaN -> median impute, with medians learned on the training split.

    Used by both arms (PR 33, audit item 6). Replaces the old guard inside
    ``tune_ebm``, which cleaned only the tuning copy of ``X_train`` and never
    ``X_test``. Medians are fitted on the rows a model trains on (the full
    train split, or the fold's training rows) and reused on every frame that
    model scores; the fitted cleaner is saved in the artifact.

    Assumes numeric columns. Raises if inf remains (e.g. ``clean_inf=False``)
    or if NaN remains after imputation (e.g. a column all-NaN in train).
    ``report_`` lists only columns with NaN in TRAIN.
    """

    clean_inf: bool = True
    impute_missing: bool = True
    medians_: Optional[pd.Series] = field(default=None, repr=False)
    columns_: Optional[list[str]] = field(default=None, repr=False)
    report_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "FeatureCleaner":
        frame = self._replace_inf(X)
        self.columns_ = list(frame.columns)
        self.medians_ = frame.median(numeric_only=False)
        n_inf = self._count_inf(X)
        nan_cols = frame.columns[frame.isna().any()].tolist()
        self.report_ = {
            "n_inf_replaced": n_inf,
            "columns_imputed": nan_cols,
            "medians": {c: float(self.medians_[c]) for c in nan_cols},
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise ValueError("Call fit() first")
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Frame is missing fitted columns: {missing}")

        frame = self._replace_inf(X[self.columns_])
        if self.impute_missing and frame.isna().any().any():
            frame = frame.fillna(self.medians_)

        if self._count_inf(frame):
            raise AssertionError("Infinity values remain after cleaning")
        if self.impute_missing and frame.isna().any().any():
            raise AssertionError("NaN values remain after cleaning")
        return frame

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.clean_inf:
            return X.copy()
        return X.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _count_inf(X: pd.DataFrame) -> int:
        """Count ±inf across numeric columns."""
        numeric = X.select_dtypes(include=["number"])
        if numeric.empty:
            return 0
        return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


# ==========================================================================
# calibration
# ==========================================================================

def calculate_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected calibration error — ``shared.metrics.calculate_ece``, the one
    definition every stage and both arms report. Its first bin is closed on
    the left, so isotonic outputs of exactly 0.0 are counted.
    """
    return _shared_ece(y_true, y_prob, n_bins)

@dataclass
class CalibrationReport:
    """Diagnostics for the artifact and the Stage 3 write-up."""

    method: str
    applied: bool
    reason: str
    ece_threshold: float
    ece_bins: int
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    gate_measured_on: str = "out-of-fold training predictions"
    n_rows: int = 0
    n_zero_after: int = 0  # calibrated rows at exactly 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        head = (f"{self.method} applied" if self.applied
                else f"not applied ({self.reason})")
        return (
            f"calibration: {head}\n"
            f"   ECE   {self.ece_before:.4f} → {self.ece_after:.4f} "
            f"(gate {self.ece_threshold}, {self.ece_bins} bins, OOF)\n"
            f"   [{self.n_zero_after} calibrated rows at p=0]\n"
            f"   Brier {self.brier_before:.4f} → {self.brier_after:.4f}"
        )

class Stage3Calibrator:
    """
    Gate on OOF ECE, then fit isotonic (or Platt) leakage-safely.

    After ``fit``:
      ``oof_calibrated_``  out-of-fold calibrated training probabilities —
                           what Stage 4 consumes.
      ``transform(p)``     maps the refit model's probabilities (i.e. test)
                           through the calibrator fitted on all OOF rows.

    When the gate is not breached (and ``force=False``) the calibrator is an
    identity map and both of the above pass probabilities through unchanged,
    mirroring the EBM's behaviour of storing raw probabilities in the
    calibrated slots.

    ``method="sigmoid"`` is Platt scaling: an (effectively unregularised)
    logistic regression on the probability itself, not its log-odds.
    """

    def __init__(
        self,
        method: str = "isotonic",
        ece_threshold: float = 0.05,
        ece_bins: int = 10,
        force: bool = False,
        random_state: int = 42,
    ):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.ece_threshold = float(ece_threshold)
        self.ece_bins = int(ece_bins)
        self.force = bool(force)
        self.random_state = int(random_state)

        self.applied_: bool = False
        self.final_calibrator_: Optional[Any] = None
        self.oof_calibrated_: Optional[np.ndarray] = None
        self.report_: Optional[CalibrationReport] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        oof_proba: np.ndarray,
        y,
        folds: FoldPlan,
    ) -> "Stage3Calibrator":
        p = np.asarray(oof_proba, dtype=float)
        y_arr = np.asarray(y).astype(int)
        if len(p) != len(y_arr):
            raise ValueError(f"length mismatch: proba {len(p)}, y {len(y_arr)}")
        if np.isnan(p).any():
            raise ValueError(
                "OOF probabilities contain NaN — calibrate on the scored array."
            )
        folds.assert_matches(len(p), "calibration")

        ece_before = calculate_ece(y_arr, p, self.ece_bins)
        brier_before = float(brier_score_loss(y_arr, p))

        needs = self.force or ece_before > self.ece_threshold
        if not needs:
            self.applied_ = False
            self.final_calibrator_ = None
            self.oof_calibrated_ = p.copy()
            self.report_ = CalibrationReport(
                method=self.method,
                applied=False,
                reason=f"OOF ECE {ece_before:.4f} <= gate {self.ece_threshold}",
                ece_threshold=self.ece_threshold,
                ece_bins=self.ece_bins,
                ece_before=ece_before,
                ece_after=ece_before,
                brier_before=brier_before,
                brier_after=brier_before,
                n_rows=len(p),
                n_zero_after=int((p == 0.0).sum()),
            )
            return self

        # --- leakage-safe calibrated OOF column ---------------------------
        # Fold k's mapping is learned from the other folds' OOF rows only.
        calibrated = np.full(len(p), np.nan, dtype=float)
        for _, val_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(p)), val_idx,
                                     assume_unique=False)
            mapper = self._fit_mapper(p[train_idx], y_arr[train_idx])
            calibrated[val_idx] = self._apply(mapper, p[val_idx])

        if np.isnan(calibrated).any():
            raise AssertionError(
                "Nested calibration left rows unmapped — check the fold plan."
            )

        # --- calibrator for the refit model's outputs (test time) ---------
        self.final_calibrator_ = self._fit_mapper(p, y_arr)
        self.applied_ = True
        self.oof_calibrated_ = calibrated

        self.report_ = CalibrationReport(
            method=self.method,
            applied=True,
            reason=f"OOF ECE {ece_before:.4f} > gate {self.ece_threshold}",
            ece_threshold=self.ece_threshold,
            ece_bins=self.ece_bins,
            ece_before=ece_before,
            ece_after=calculate_ece(y_arr, calibrated, self.ece_bins),
            brier_before=brier_before,
            brier_after=float(brier_score_loss(y_arr, calibrated)),
            n_rows=len(p),
            n_zero_after=int((calibrated == 0.0).sum()),
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Map refit-model probabilities (test split) into calibrated space."""
        p = np.asarray(proba, dtype=float)
        if not self.applied_ or self.final_calibrator_ is None:
            return p.copy()
        return self._apply(self.final_calibrator_, p)

    # ------------------------------------------------------------------
    def _fit_mapper(self, p: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            return IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(p, y)
        # Platt scaling, fitted on the probability (not the logit).
        return LogisticRegression(
            C=1e10, solver="lbfgs", random_state=self.random_state
        ).fit(p.reshape(-1, 1), y)

    @staticmethod
    def _apply(mapper, p: np.ndarray) -> np.ndarray:
        if isinstance(mapper, IsotonicRegression):
            out = mapper.predict(p)
        else:
            out = mapper.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


# ==========================================================================
# metrics
# ==========================================================================

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


# ==========================================================================
# estimator
# ==========================================================================

@runtime_checkable
class Stage3Estimator(Protocol):
    """What an arm must provide."""

    family: str
    as_array: bool          # True → models are fitted on float ndarrays

    def suggest(self, trial) -> dict[str, Any]: ...
    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def fit(self, params: dict[str, Any], X: pd.DataFrame, y,
            sample_weight: Optional[np.ndarray] = None): ...
    def positive_proba(self, model, X: pd.DataFrame) -> np.ndarray: ...
    def search_space_dict(self) -> dict: ...
    def describe_space(self) -> dict[str, str]: ...
    def to_dict(self) -> dict: ...

def positive_proba(model, X, as_array: bool) -> np.ndarray:
    """``P(y = 1 | x)``, robust to a single-class fold."""
    data = np.asarray(X, dtype=float) if as_array else X
    proba = model.predict_proba(data)
    classes = list(getattr(model, "classes_", [0, 1]))
    if proba.shape[1] == 1:
        return np.full(len(X), float(classes[0]), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)

def _apply_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return p.copy()
    return Stage3Calibrator._apply(calibrator, p)

class CalibratedModel:
    """Refit model + OOF-fitted calibrator, called on ENGINEERED features."""

    def __init__(self, model, calibrator, as_array: bool, features: list[str]):
        self.model = model
        self.calibrator = calibrator
        self.as_array = bool(as_array)
        self.features = list(features)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features] if isinstance(X, pd.DataFrame) else X
        p = _apply_calibrator(
            self.calibrator, positive_proba(self.model, X, self.as_array)
        )
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

class Stage3ServingModel:
    """Raw GLOBAL_SPLIT-shaped rows → Stage 3 probabilities."""

    def __init__(self, pipeline, cleaner, model, calibrator,
                 as_array: bool, features: list[str]):
        self.pipeline = pipeline
        self.cleaner = cleaner
        self.model = model
        self.calibrator = calibrator
        self.as_array = bool(as_array)
        self.features = list(features)

    def engineer(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = self.pipeline.transform(X_raw.copy())
        return self.cleaner.transform(X[self.features])

    def predict_proba_raw(self, X_raw: pd.DataFrame) -> np.ndarray:
        return positive_proba(self.model, self.engineer(X_raw), self.as_array)

    def predict_proba_calibrated(self, X_raw: pd.DataFrame) -> np.ndarray:
        return _apply_calibrator(self.calibrator, self.predict_proba_raw(X_raw))


# ==========================================================================
# block
# ==========================================================================

class PassthroughPipeline:
    """Identity 'pipeline' for frames that are already engineered."""

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

@dataclass
class FoldFrames:
    """One fold's model-ready frames (engineered + cleaned)."""

    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_tr: pd.DataFrame
    y_tr: pd.Series
    w_tr: np.ndarray
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: Optional[pd.DataFrame] = None
    fitted_on_n_rows: int = 0

def _hash_frame(h, frame: pd.DataFrame) -> None:
    h.update("|".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())

@dataclass
class Stage3Block:
    """Everything both Stage 3 arms consume. Build with ``Stage3Block.build``."""

    features: list[str]
    feature_fit_scope: str
    X_train: pd.DataFrame           # full-train pipeline, cleaned
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Any                   # fitted on all of train (serving)
    cleaner: FeatureCleaner         # fitted on all of train (serving)
    weights: BalancedWeights
    tune_plan: FoldPlan
    report_plan: FoldPlan
    tune_folds: list[FoldFrames]
    report_folds: list[FoldFrames]
    split_fingerprint: Optional[str] = None
    block_fingerprint: str = ""
    settings: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def index_train(self) -> pd.Index:
        return self.X_train.index

    @property
    def index_test(self) -> pd.Index:
        return self.X_test.index

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        X_train_raw: pd.DataFrame,
        y_train,
        X_test_raw: pd.DataFrame,
        y_test,
        pipeline_factory: Callable[[], Any],
        *,
        feature_fit_scope: str = "per_fold",
        n_tune_folds: int = 5,
        n_eval_folds: int = 10,
        random_state: int = 42,
        stratify: bool = True,
        class_weight: str = "balanced",
        clean_inf: bool = True,
        impute_missing: bool = True,
        expected_features: Optional[list[str]] = None,
        split_fingerprint: Optional[str] = None,
        verbose: bool = True,
    ) -> "Stage3Block":
        say = print if verbose else (lambda *a, **k: None)
        if feature_fit_scope not in ("per_fold", "global"):
            raise ValueError(f"unknown feature_fit_scope {feature_fit_scope!r}")

        y_tr = _as_label_series(y_train, X_train_raw, "y_train")
        y_te = _as_label_series(y_test, X_test_raw, "y_test")
        contract = Stage3FeatureContract(expected_features)

        # ---- full-train fit: refit model + test split ------------------
        say(f"  Stage 3 block: fitting feature pipeline on all "
            f"{len(X_train_raw):,} training rows")
        pipe = pipeline_factory()
        Xtr = pipe.fit_transform(X_train_raw.copy(), y_tr)
        Xte = pipe.transform(X_test_raw.copy())
        _assert_index(Xtr, X_train_raw, "full-train fit_transform")
        _assert_index(Xte, X_test_raw, "full-train transform(test)")
        Xtr, Xte, order = contract.align(Xtr, Xte)
        cleaner = FeatureCleaner(clean_inf, impute_missing)
        Xtr = cleaner.fit_transform(Xtr)
        Xte = cleaner.transform(Xte)

        # ---- weights + folds (depend only on y, n and the seed) --------
        weights = BalancedWeights.balanced(y_tr, class_weight)
        tune_plan = FoldPlan.build(
            y_tr, n_tune_folds, random_state, stratify,
            purpose="tuning+oof+calibration+threshold",
        )
        report_plan = FoldPlan.build(
            y_tr, n_eval_folds, random_state, stratify, purpose="reporting",
        )

        # ---- per-fold frames --------------------------------------------
        def fold_frames(plan: FoldPlan, with_test: bool) -> list[FoldFrames]:
            out = []
            for k, (tr, va) in enumerate(plan):
                if feature_fit_scope == "per_fold":
                    p = pipeline_factory()
                    Xa_raw = X_train_raw.iloc[tr]
                    Xb_raw = X_train_raw.iloc[va]
                    Xa = p.fit_transform(Xa_raw.copy(), y_tr.iloc[tr])
                    Xb = p.transform(Xb_raw.copy())
                    _assert_index(Xa, Xa_raw, f"fold {k} fit_transform")
                    _assert_index(Xb, Xb_raw, f"fold {k} transform(val)")
                    Xt = None
                    if with_test:
                        Xt = p.transform(X_test_raw.copy())
                        _assert_index(Xt, X_test_raw, f"fold {k} transform(test)")
                    Xa, Xb, fold_order = contract.align(Xa, Xb)
                    if set(fold_order) != set(order):
                        raise ValueError(
                            f"fold {k}: pipeline produced columns "
                            f"{sorted(set(fold_order) ^ set(order))} that "
                            "differ from the full-train fit"
                        )
                    Xa, Xb = Xa[order], Xb[order]
                    c = FeatureCleaner(clean_inf, impute_missing)
                    Xa = c.fit_transform(Xa)
                    Xb = c.transform(Xb)
                    if Xt is not None:
                        Xt = c.transform(Xt[order])
                else:
                    Xa, Xb = Xtr.iloc[tr], Xtr.iloc[va]
                    Xt = Xte if with_test else None
                out.append(FoldFrames(
                    k=k, train_idx=tr, val_idx=va,
                    X_tr=Xa, y_tr=y_tr.iloc[tr], w_tr=weights.for_rows(tr),
                    X_val=Xb, y_val=y_tr.iloc[va], X_test=Xt,
                    fitted_on_n_rows=int(len(tr)),
                ))
            return out

        say(f"  Stage 3 block: feature_fit_scope={feature_fit_scope!r} — "
            + ("refitting the feature pipeline inside each of "
               f"{n_tune_folds} tuning + {n_eval_folds} reporting folds"
               if feature_fit_scope == "per_fold"
               else "slicing the full-train features (LEAKY — comparison only)"))
        tune_folds = fold_frames(tune_plan, with_test=True)
        report_folds = fold_frames(report_plan, with_test=False)

        settings = {
            "feature_fit_scope": feature_fit_scope,
            "n_tune_folds": int(n_tune_folds),
            "n_eval_folds": int(n_eval_folds),
            "random_state": int(random_state),
            "stratify": bool(stratify),
            "class_weight": class_weight,
            "clean_inf": bool(clean_inf),
            "impute_missing": bool(impute_missing),
            "pipeline": type(pipe).__module__ + "." + type(pipe).__qualname__,
        }
        block = cls(
            features=list(order), feature_fit_scope=feature_fit_scope,
            X_train=Xtr, X_test=Xte, y_train=y_tr, y_test=y_te,
            pipeline=pipe, cleaner=cleaner, weights=weights,
            tune_plan=tune_plan, report_plan=report_plan,
            tune_folds=tune_folds, report_folds=report_folds,
            split_fingerprint=split_fingerprint, settings=settings,
        )
        block.block_fingerprint = block.compute_fingerprint()
        say(f"  Stage 3 block: {len(order)} features, train {Xtr.shape}, "
            f"test {Xte.shape}, fingerprint {block.block_fingerprint[:16]}…")
        return block

    # ------------------------------------------------------------------
    @classmethod
    def from_global_split(
        cls,
        GLOBAL_SPLIT: dict,
        pipeline_factory: Callable[[], Any],
        config,
        split_fingerprint: Optional[str] = None,
    ) -> "Stage3Block":
        """Build with the fold / weight / hygiene settings of a Stage3Config."""
        missing = [k for k in ("X_train", "X_test", "y_train", "y_test")
                   if k not in GLOBAL_SPLIT]
        if missing:
            raise KeyError(f"GLOBAL_SPLIT missing keys: {missing}")
        return cls.build(
            GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["y_train"],
            GLOBAL_SPLIT["X_test"], GLOBAL_SPLIT["y_test"],
            pipeline_factory,
            feature_fit_scope=config.feature_fit_scope,
            n_tune_folds=config.n_tune_folds,
            n_eval_folds=config.n_eval_folds,
            random_state=config.random_state,
            stratify=config.stratify,
            class_weight=config.class_weight,
            clean_inf=config.clean_inf,
            impute_missing=config.impute_missing,
            expected_features=config.expected_features,
            split_fingerprint=split_fingerprint,
            verbose=config.verbose,
        )

    # ------------------------------------------------------------------
    def compute_fingerprint(self) -> str:
        """SHA-256 over every frame, label vector, fold assignment and weight."""
        h = hashlib.sha256()
        h.update(self.feature_fit_scope.encode())
        _hash_frame(h, self.X_train)
        _hash_frame(h, self.X_test)
        h.update(np.asarray(self.y_train, dtype=np.int64).tobytes())
        h.update(np.asarray(self.y_test, dtype=np.int64).tobytes())
        h.update(self.weights.vector.tobytes())
        h.update(self.tune_plan.fold_id.astype(np.int64).tobytes())
        h.update(self.report_plan.fold_id.astype(np.int64).tobytes())
        for f in self.tune_folds + self.report_folds:
            _hash_frame(h, f.X_tr)
            _hash_frame(h, f.X_val)
            if f.X_test is not None:
                _hash_frame(h, f.X_test)
        return h.hexdigest()

    def check_config(self, config) -> None:
        """Raise if a Stage3Config disagrees with how this block was built."""
        pairs = {
            "feature_fit_scope": config.feature_fit_scope,
            "n_tune_folds": config.n_tune_folds,
            "n_eval_folds": config.n_eval_folds,
            "random_state": config.random_state,
            "stratify": config.stratify,
            "class_weight": config.class_weight,
            "clean_inf": config.clean_inf,
            "impute_missing": config.impute_missing,
        }
        bad = {k: (self.settings.get(k), v) for k, v in pairs.items()
               if self.settings.get(k) != v}
        if config.expected_features is not None and \
                list(config.expected_features) != list(self.features):
            bad["expected_features"] = (self.features, config.expected_features)
        if bad:
            raise ValueError(
                "Stage3Block was built with different settings from the "
                "config:\n" + "\n".join(
                    f"  {k}: block={b!r} config={c!r}" for k, (b, c) in bad.items())
            )

    def describe(self) -> dict:
        return {
            **self.settings,
            "features": list(self.features),
            "n_train": int(len(self.X_train)),
            "n_test": int(len(self.X_test)),
            "split_fingerprint": self.split_fingerprint,
            "block_fingerprint": self.block_fingerprint,
        }

def _as_label_series(y, X: pd.DataFrame, name: str) -> pd.Series:
    if isinstance(y, pd.Series):
        if len(y) != len(X) or not y.index.equals(X.index):
            raise ValueError(
                f"{name}.index differs from its feature frame. Reindex "
                f"before calling ({name} = {name}.loc[X.index])."
            )
        out = y
    else:
        arr = np.asarray(y)
        if len(arr) != len(X):
            raise ValueError(f"{name} length {len(arr)} != {len(X)} rows")
        out = pd.Series(arr, index=X.index)
    if not np.isin(np.asarray(out), (0, 1)).all():
        raise ValueError(f"{name} must be binary 0/1")
    return out.astype(int)

def _assert_index(out: pd.DataFrame, ref: pd.DataFrame, where: str) -> None:
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"{where}: pipeline must return a DataFrame")
    if not out.index.equals(ref.index):
        raise ValueError(
            f"{where}: feature pipeline changed the row index/order. Stage 3 "
            "requires index-preserving transforms."
        )


# ==========================================================================
# thresholds
# ==========================================================================

@dataclass
class ThresholdChoice:
    """A selected threshold plus the sweep that justified it."""

    threshold: float
    score: float
    beta: float
    selected_on: str
    n_rows: int
    grid_low: float
    grid_high: float
    grid_steps: int
    sweep: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "sweep"}

    def describe(self) -> str:
        return (
            f"threshold {self.threshold:.4f} "
            f"(F{self.beta:g}={self.score:.4f}, selected on {self.selected_on}, "
            f"n={self.n_rows:,})"
        )

class F2ThresholdSelector:
    """
    Grid search for the F-beta-maximising decision threshold.

    Mirrors the EBM's linear scan rather than, say, a PR-curve argmax: the two
    arms must pick their operating point by the same procedure for the
    resulting metrics to be comparable.
    """

    def __init__(
        self,
        beta: float = 2.0,
        low: float = 0.10,
        high: float = 0.90,
        steps: int = 81,
    ):
        if not 0.0 < low < high < 1.0:
            raise ValueError(f"require 0 < low < high < 1, got {low} / {high}")
        if steps < 2:
            raise ValueError("steps must be >= 2")
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)
        self.steps = int(steps)

    # ------------------------------------------------------------------
    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.low, self.high, self.steps)

    def select(
        self,
        y_true,
        proba: np.ndarray,
        selected_on: str = "out-of-fold training predictions",
        keep_sweep: bool = True,
    ) -> ThresholdChoice:
        y_arr = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        if len(y_arr) != len(p):
            raise ValueError(f"length mismatch: y {len(y_arr)}, proba {len(p)}")
        if len(p) == 0:
            raise ValueError("empty population")
        if np.isnan(p).any():
            raise ValueError(
                "probabilities contain NaN — pass only scored rows."
            )

        # The sweep itself is shared.thresholds.sweep_f_beta (Stages 1–2 use
        # it too); Stage 3 passes its own grid. Ties → lowest threshold.
        sweep = sweep_f_beta(y_arr, p, beta=self.beta, grid=self.grid)
        if sweep.empty:
            best_t, best_score = 0.5, 0.0      # same fallback as shared
        else:
            best = sweep.loc[sweep["f_beta"].idxmax()]
            best_t, best_score = float(best["threshold"]), float(best["f_beta"])
        if not keep_sweep:
            sweep = None

        return ThresholdChoice(
            threshold=best_t,
            score=best_score,
            beta=self.beta,
            selected_on=selected_on,
            n_rows=len(p),
            grid_low=self.low,
            grid_high=self.high,
            grid_steps=self.steps,
            sweep=sweep,
        )

    # ------------------------------------------------------------------
    def select_nested(self, y_true, proba: np.ndarray, folds) -> dict:
        """
        Fold-nested thresholds for the training-side decision columns.

        For each fold *k*, the grid is scored on the OTHER folds' OOF rows
        only, and that threshold is applied to fold *k*. No row's label
        influences the threshold used to make its own decision — the same
        pattern as the nested calibration.

        Returns ``{"per_row": ndarray, "by_fold": [...], "mean", "std",
        "min", "max"}``. The spread across folds is a stability diagnostic
        for the operating point.
        """
        y_arr = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        folds.assert_matches(len(p), "nested threshold selection")
        per_row = np.full(len(p), np.nan, dtype=float)
        by_fold: list[float] = []
        for train_idx, val_idx in folds:
            t = self.select(y_arr[train_idx], p[train_idx],
                            keep_sweep=False).threshold
            per_row[val_idx] = t
            by_fold.append(float(t))
        if np.isnan(per_row).any():
            raise AssertionError(
                "Nested threshold selection left rows unassigned."
            )
        arr = np.asarray(by_fold)
        return {
            "per_row": per_row,
            "by_fold": by_fold,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    # ------------------------------------------------------------------
    def oracle_on_test(self, y_test, test_proba: np.ndarray) -> ThresholdChoice:
        """
        The value the GLASS rule would produce on the test split.

        Diagnostic only. Nothing downstream may consume it — it is recorded so
        the write-up can quantify how much the GLASS protocol's test-fitted
        threshold flatters its own metrics.
        """
        return self.select(
            y_test, test_proba,
            selected_on="TEST split (oracle — reference only)",
            keep_sweep=False,
        )


# ==========================================================================
# tuning
# ==========================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    objective: str
    n_trials_requested: int
    n_complete: int
    n_pruned: int
    n_failed: int
    runtime_s: float
    random_state: int
    study_name: str
    search_space: dict[str, str]
    fold_plan: dict
    feature_fit_scope: str = "per_fold"
    tuned: bool = True
    top_trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"best CV F2 : {self.best_value:.6f}",
            f"trials     : {self.n_complete} completed, {self.n_pruned} pruned, "
            f"{self.n_failed} failed of {self.n_trials_requested}",
            f"runtime    : {self.runtime_s:.1f}s",
            f"features   : fitted {self.feature_fit_scope}",
            "best params:",
        ]
        for k, v in self.best_params.items():
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"   {k:>20s} = {fmt}")
        return "\n".join(lines)

class Stage3Tuner:
    """Recall-biased Optuna search over the estimator's declared space."""

    def __init__(self, config, estimator, folds: list[FoldFrames],
                 fold_plan_dict: dict, feature_fit_scope: str):
        self.config = config
        self.estimator = estimator
        self.folds = folds
        self.fold_plan_dict = fold_plan_dict
        self.feature_fit_scope = feature_fit_scope

    # ------------------------------------------------------------------
    def _fold_score(self, params, f: FoldFrames) -> float:
        cfg = self.config
        model = self.estimator.fit(params, f.X_tr, f.y_tr, sample_weight=f.w_tr)
        p = self.estimator.positive_proba(model, f.X_val)
        pred = (p >= cfg.tuning_decision_threshold).astype(int)
        return float(fbeta_score(np.asarray(f.y_val).astype(int), pred,
                                 beta=cfg.beta, zero_division=0))

    def _objective(self, trial: optuna.Trial) -> float:
        params = self.estimator.suggest(trial)
        scores: list[float] = []
        for fold_idx, f in enumerate(self.folds):
            try:
                scores.append(self._fold_score(params, f))
            except optuna.TrialPruned:
                raise
            except Exception:
                return float("-inf")
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    def tune(self) -> TuningResult:
        cfg = self.config
        n = sum(len(f.val_idx) for f in self.folds)
        if cfg.verbose:
            print("\n" + "=" * 78)
            print(f"  {cfg.model_family.upper()} STAGE 3 TUNING — RECALL-BIASED "
                  f"(F{cfg.beta:g} at the {cfg.tuning_decision_threshold} cut)")
            print("=" * 78)
            print(f"  rows         : {n:,}  |  features fitted "
                  f"{self.feature_fit_scope}")
            print(f"  folds        : {len(self.folds)}-fold, seed "
                  f"{cfg.random_state}")
            print(f"  budget       : {cfg.n_trials} trials, TPE(seed="
                  f"{cfg.random_state}), MedianPruner("
                  f"{cfg.pruner_startup_trials}/{cfg.pruner_warmup_steps})")
            print("  search space :")
            for k, v in self.estimator.describe_space().items():
                print(f"     {k:>20s}  {v}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=cfg.random_state),
            pruner=MedianPruner(n_startup_trials=cfg.pruner_startup_trials,
                                n_warmup_steps=cfg.pruner_warmup_steps),
            study_name=cfg.study_name,
        )
        t0 = time.perf_counter()
        study.optimize(self._objective, n_trials=cfg.n_trials,
                       show_progress_bar=cfg.show_progress_bar and cfg.verbose)
        runtime_s = time.perf_counter() - t0

        TS = optuna.trial.TrialState
        finite = [t for t in study.trials
                  if t.state == TS.COMPLETE and t.value is not None
                  and np.isfinite(t.value)]
        if not finite:
            raise RuntimeError(
                f"No Optuna trial produced a finite score in {cfg.n_trials} "
                "trials — every configuration failed."
            )
        n_errored = sum(t.state == TS.COMPLETE for t in study.trials) - len(finite)
        best = max(finite, key=lambda t: t.value)

        result = TuningResult(
            best_params=self.estimator.normalize_params(best.params),
            best_value=float(best.value),
            objective=f"F{cfg.beta:g} @ {cfg.tuning_decision_threshold} cut, "
                      f"{len(self.folds)}-fold CV",
            n_trials_requested=cfg.n_trials,
            n_complete=len(finite),
            n_pruned=sum(t.state == TS.PRUNED for t in study.trials),
            n_failed=sum(t.state == TS.FAIL for t in study.trials) + n_errored,
            runtime_s=float(runtime_s),
            random_state=cfg.random_state,
            study_name=cfg.study_name,
            search_space=self.estimator.describe_space(),
            fold_plan=self.fold_plan_dict,
            feature_fit_scope=self.feature_fit_scope,
            top_trials=[
                {"number": t.number, "value": float(t.value),
                 "params": dict(t.params)}
                for t in sorted(finite, key=lambda t: t.value, reverse=True)[:5]
            ],
        )
        if cfg.verbose:
            print("\n" + "-" * 78)
            print(result.describe())
            print("-" * 78)
        return result


# ==========================================================================
# cv
# ==========================================================================

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


# ==========================================================================
# oof
# ==========================================================================

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


# ==========================================================================
# artifacts
# ==========================================================================

STAGE4_FEATURE_SUFFIXES = ("proba", "proba_calibrated", "decision",
                           "confidence", "margin")

TEST_SOURCES = ("refit", "fold_mean")

def decision_block(proba: np.ndarray, threshold) -> dict[str, np.ndarray]:
    """
    Binary decision, signed margin and normalised confidence.

    ``threshold`` is a scalar or per-row array. ``margin = p - t``;
    ``confidence`` rescales |margin| by the room on that side of the boundary
    so it lands in [0, 1] wherever the threshold sits.
    """
    p = np.asarray(proba, dtype=float)
    t = np.broadcast_to(np.asarray(threshold, dtype=float), p.shape)
    decision = (p >= t).astype(int)
    margin = p - t
    upper = np.maximum(1.0 - t, 1e-12)
    lower = np.maximum(t, 1e-12)
    confidence = np.where(margin >= 0, margin / upper, -margin / lower)
    return {
        "decision": decision,
        "margin": margin,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "state": np.where(decision == 1, "flagged", "not_flagged").astype(object),
    }

@dataclass
class Stage3Artifact:
    """Canonical Stage 3 payload (EBM and XGBoost alike)."""

    # ---- identity ---------------------------------------------------------
    stage: str
    model_family: str
    created_at: str
    config: dict
    mirror_report: dict
    column_prefix: str = "stage3_"
    population: str = "full_train"

    # ---- input contract ---------------------------------------------------
    features: list[str] = field(default_factory=list)
    index_train: list = field(default_factory=list)
    index_test: list = field(default_factory=list)
    y_train: list = field(default_factory=list)
    y_test: list = field(default_factory=list)
    split_fingerprint: Optional[str] = None
    block_fingerprint: Optional[str] = None
    feature_fit_scope: str = "per_fold"
    block_settings: dict = field(default_factory=dict)
    cleaning_report: dict = field(default_factory=dict)

    # ---- partition --------------------------------------------------------
    fold_plan: dict = field(default_factory=dict)
    report_fold_plan: dict = field(default_factory=dict)
    class_weights: dict = field(default_factory=dict)

    # ---- search -----------------------------------------------------------
    tuning: dict = field(default_factory=dict)
    cv_report: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    cv_score: float = float("nan")

    # ---- predictions ------------------------------------------------------
    oof: dict = field(default_factory=dict)
    refit: dict = field(default_factory=dict)
    fold_mean: dict = field(default_factory=dict)

    # ---- operating point + calibration ------------------------------------
    threshold: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    # ---- metrics + family extras -----------------------------------------
    metrics: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    # ==================================================================
    # Stage 4 interface
    # ==================================================================
    def _cols(self, block: dict, key_map: dict) -> dict:
        p = self.column_prefix
        return {f"{p}{suffix}": np.asarray(block[key])
                for suffix, key in key_map.items()}

    def stage4_frame(self) -> pd.DataFrame:
        """Training-side Stage 4 FEATURES. OOF, leakage-safe."""
        keys = {s: s for s in STAGE4_FEATURE_SUFFIXES}
        return pd.DataFrame(self._cols(self.oof, keys),
                            index=pd.Index(self.index_train))

    def stage4_metadata(self) -> pd.DataFrame:
        """Training-side bookkeeping. NOT features."""
        p = self.column_prefix
        return pd.DataFrame(
            {
                f"{p}fold_id": np.asarray(self.oof["fold_id"]),
                f"{p}threshold": np.asarray(self.oof["threshold_per_row"]),
                f"{p}state": np.asarray(self.oof["state"], dtype=object),
            },
            index=pd.Index(self.index_train),
        )

    def _test_block(self, source: str) -> dict:
        if source not in TEST_SOURCES:
            raise ValueError(f"source must be one of {TEST_SOURCES}")
        return self.refit if source == "refit" else self.fold_mean

    def stage4_test_frame(self, source: str = "refit") -> pd.DataFrame:
        """
        Test-side counterpart of ``stage4_frame()``; used to EVALUATE Stage 4.

        ``source="refit"``     the full-train model (default)
        ``source="fold_mean"`` mean of the 5 OOF fold models
        """
        b = self._test_block(source)
        keys = {
            "proba": "test_proba",
            "proba_calibrated": "test_proba_calibrated",
            "decision": "test_decision",
            "confidence": "test_confidence",
            "margin": "test_margin",
        }
        return pd.DataFrame(self._cols(b, keys), index=pd.Index(self.index_test))

    def stage4_test_metadata(self, source: str = "refit") -> pd.DataFrame:
        b = self._test_block(source)
        p = self.column_prefix
        n = len(b["test_proba"])
        return pd.DataFrame(
            {
                f"{p}fold_id": np.full(n, -1),
                f"{p}threshold": np.full(n, float(self.threshold["oof"]["threshold"])),
                f"{p}state": np.asarray(b["test_state"], dtype=object),
                f"{p}source": np.full(n, source, dtype=object),
            },
            index=pd.Index(self.index_test),
        )

    # ==================================================================
    # Cascade population
    # ==================================================================
    def evaluate_served(
        self,
        y_test,
        served_mask,
        source: str = "refit",
        space: str = "raw",
        label: str = "served",
    ) -> dict:
        """
        Stage 3 metrics on the test rows Stage 3 actually serves.

        ``served_mask``: boolean, aligned to ``index_test`` (a Series is
        reindexed by label and must cover every test row).
        ``space``: ``"raw"`` uses the OOF threshold, ``"calibrated"`` the OOF
        calibrated-space threshold. ``coverage`` = served / all test rows.
        """

        idx = pd.Index(self.index_test)
        if isinstance(served_mask, pd.Series):
            if not served_mask.index.isin(idx).all() or len(served_mask) != len(idx):
                raise ValueError("served_mask index must match index_test exactly")
            mask = served_mask.reindex(idx).to_numpy(dtype=bool)
        else:
            mask = np.asarray(served_mask, dtype=bool)
            if len(mask) != len(idx):
                raise ValueError(f"served_mask has {len(mask)} rows, test has {len(idx)}")
        y = pd.Series(np.asarray(y_test), index=idx) if not isinstance(y_test, pd.Series) \
            else y_test.reindex(idx)
        b = self._test_block(source)
        if space == "raw":
            p, t = np.asarray(b["test_proba"]), self.threshold["oof"]["threshold"]
        elif space == "calibrated":
            p = np.asarray(b["test_proba_calibrated"])
            t = self.threshold["oof_calibrated_space"]["threshold"]
        else:
            raise ValueError("space must be 'raw' or 'calibrated'")
        if not mask.any():
            raise ValueError("served_mask selects no rows")
        m = Stage3Metrics.compute(
            y.to_numpy()[mask], p[mask], t,
            split=f"test ({label}, {source})", probability_space=space,
            n_population=len(idx),
        )
        return m.to_dict()

    # ==================================================================
    # Cross-stage contract
    # ==================================================================
    def to_stage_output(self):
        """
        The ``shared.stage_io.StageOutput`` Stages 1–2 already hand to Stage 4.

        Carries the CALIBRATED probabilities (the contract's definition):
        train side = fold-nested calibrated OOF, test side = refit model
        through the all-OOF calibrator, with the calibrated-space OOF
        threshold. ``threshold_source='in_stage_f2_cv'``: that scalar was
        chosen on every training row's OOF labels, so ``StageOutput`` will
        refuse train-side decisions from it — use ``stage4_frame()``'s
        fold-nested decision columns for those.
        """
        from shared.stage_io import StageOutput

        itr, ite = pd.Index(self.index_train), pd.Index(self.index_test)
        c = self.calibration
        cfg = self.config
        cal_thr = self.threshold["oof_calibrated_space"]
        return StageOutput(
            stage=self.stage,
            arm=cfg.get("arm", self.model_family),
            model=self.model_family,
            feature_names=list(self.features),
            train_proba_oof=pd.Series(np.asarray(self.oof["proba_calibrated"], float),
                                      index=itr, name=f"{self.stage}_proba_oof"),
            test_proba=pd.Series(np.asarray(self.refit["test_proba_calibrated"], float),
                                 index=ite, name=f"{self.stage}_proba"),
            y_train=pd.Series(np.asarray(self.y_train, int), index=itr, name="y"),
            y_test=pd.Series(np.asarray(self.y_test, int), index=ite, name="y"),
            train_fold_id=pd.Series(np.asarray(self.oof["fold_id"], int),
                                    index=itr, name="fold_id"),
            threshold=float(cal_thr["threshold"]),
            threshold_source="in_stage_f2_cv",
            calibration_method=cfg.get("calibration_method") if c.get("applied") else "none",
            calibration_requested=cfg.get("calibration_method", "isotonic"),
            oof_provenance=(
                f"refittable (per-fold refit: features {self.feature_fit_scope}, "
                f"model, calibration map; {self.fold_plan.get('n_splits')}-fold "
                f"stratified, seed {self.fold_plan.get('random_state')})"
            ),
            best_params=dict(self.best_params),
            best_cv_score=None if not np.isfinite(self.cv_score) else float(self.cv_score),
            cv_f2=float(cal_thr["score"]),
            threshold_sweep=None,
            calibration_metrics=dict(c),
            metrics_test=dict(self.metrics.get("test_calibrated_at_operating_threshold", {})),
            config={
                **cfg,
                "threshold_grid": [cfg.get("threshold_low"), cfg.get("threshold_high"),
                                   cfg.get("threshold_steps")],
                "block_fingerprint": self.block_fingerprint,
                "stage3_split_fingerprint": self.split_fingerprint,
            },
        )

    # ==================================================================
    # Serving + legacy schema
    # ==================================================================
    @property
    def serving_model(self):
        """Raw rows → probabilities (fitted pipeline + cleaner + model + calibrator)."""
        return self.refit.get("serving_model")

    def to_glass_schema(self) -> dict:
        """
        Pre-PR-33 GLASS key names, for comparison / Venn-tracing code.

        ``train_predictions`` is the IN-SAMPLE refit column, because that is
        what the old key meant. Anything that FITS on Stage 3 output must use
        ``stage4_frame()`` instead. ``calibrated_model`` is a model callable on
        engineered features. ``ece`` is the OOF ECE before calibration.
        """
        cal_model = CalibratedModel(
            self.refit.get("model"), self.refit.get("calibrator"),
            self.refit.get("as_array", False), self.features,
        )
        return {
            "model": self.refit.get("model"),
            "calibrated_model": cal_model,
            "train_predictions": self.refit.get("train_proba"),
            "test_predictions": self.refit.get("test_proba"),
            "train_predictions_calibrated": self.refit.get("train_proba_calibrated"),
            "test_predictions_calibrated": self.refit.get("test_proba_calibrated"),
            "optimal_threshold": self.threshold.get("oof", {}).get("threshold"),
            "optimal_threshold_space": "raw",
            "metrics": self.metrics.get("test_at_operating_threshold"),
            "metrics_at_half": self.metrics.get("test_at_half"),
            "ece": self.calibration.get("ece_before"),
            "ece_semantics": "OOF ECE before calibration",
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "features": self.features,
            "interactions": self.extras.get("interactions"),
            "split_fingerprint": self.split_fingerprint,
            "index_test": list(self.index_test),
        }

    # ==================================================================
    def summary(self) -> dict:
        """Small, JSON-serialisable sidecar. No arrays, no models."""
        return {
            "stage": self.stage,
            "model_family": self.model_family,
            "created_at": self.created_at,
            "population": self.population,
            "n_features": len(self.features),
            "features": self.features,
            "split_fingerprint": self.split_fingerprint,
            "block_fingerprint": self.block_fingerprint,
            "feature_fit_scope": self.feature_fit_scope,
            "protocol": {k: v for k, v in self.config.items()
                         if k not in ("verbose", "show_progress_bar")},
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "tuning": {k: v for k, v in self.tuning.items() if k != "top_trials"},
            "cv_report": self.cv_report.get("aggregate", {}),
            "threshold": {k: v for k, v in self.threshold.items() if k != "sweep"},
            "calibration": self.calibration,
            "class_weights": self.class_weights,
            "fold_plan": {k: v for k, v in self.fold_plan.items() if k != "fold_id"},
            "metrics": self.metrics,
            "mirror_report": self.mirror_report,
            "cleaning_report": self.cleaning_report,
            "extras": self.extras,
        }

    def describe(self) -> None:
        print("=" * 78)
        print(f"  STAGE 3 — {self.model_family.upper()}  ({self.created_at})")
        print("=" * 78)
        print(f"  features        : {len(self.features)} "
              f"(fitted {self.feature_fit_scope})")
        print(f"  best CV F2      : {self.cv_score:.6f}")
        print(f"  operating thresh: {self.threshold['oof']['threshold']:.4f} "
              f"({self.threshold['oof']['selected_on']})")
        if "test_oracle" in self.threshold:
            print(f"  test-oracle th. : {self.threshold['test_oracle']['threshold']:.4f} "
                  f"(reference only)")
        c = self.calibration
        print(f"  calibration     : {'applied' if c.get('applied') else 'skipped'} "
              f"— {c.get('reason')}")
        print(f"  ECE  OOF {c.get('ece_before', float('nan')):.4f} → "
              f"{c.get('ece_after', float('nan')):.4f}  |  test "
              f"{c.get('test_ece_raw', float('nan')):.4f} → "
              f"{c.get('test_ece_calibrated', float('nan')):.4f}")
        for name in ("oof_at_nested_thresholds", "test_at_operating_threshold",
                     "test_fold_mean_at_operating_threshold", "test_at_half"):
            m = self.metrics.get(name)
            if m:
                print(f"  {name:<38} F2={m['f2']:.4f}  rec={m['recall']:.4f}  "
                      f"prec={m['precision']:.4f}  AUC={m['roc_auc']:.4f}")
        print("=" * 78)

class Stage3ArtifactStore:
    """
    ``save`` writes ``<stem>_<timestamp>.joblib`` + a JSON sidecar, and (if
    timestamped) a full ``<stem>_latest.joblib`` copy.
    """

    def __init__(self, base_path: str = "models/stage3", stem: str = "stage3",
                 timestamped: bool = True, write_latest_pointer: bool = True):
        self.base_path = base_path
        self.stem = stem
        self.timestamped = bool(timestamped)
        self.write_latest_pointer = bool(write_latest_pointer)

    def save(self, artifact: Stage3Artifact, verbose: bool = True) -> str:
        os.makedirs(self.base_path, exist_ok=True)
        if self.timestamped:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"{self.stem}_{ts}.joblib"
        else:
            name = f"{self.stem}.joblib"
        path = os.path.join(self.base_path, name)
        joblib.dump(artifact, path)
        side = path[:-len(".joblib")] + ".json"
        with open(side, "w") as fh:
            json.dump(artifact.summary(), fh, indent=2, default=str)
        if self.write_latest_pointer and self.timestamped:
            joblib.dump(artifact, os.path.join(self.base_path,
                                               f"{self.stem}_latest.joblib"))
        if verbose:
            print(f"\n  saved Stage 3 artifact → {path}")
            print(f"        summary sidecar  → {side}")
        return path

    @staticmethod
    def load(path: str) -> Stage3Artifact:
        return joblib.load(path)

    def load_latest(self) -> Stage3Artifact:
        latest = os.path.join(self.base_path, f"{self.stem}_latest.joblib")
        if os.path.exists(latest):
            return self.load(latest)
        candidates = sorted(
            f for f in os.listdir(self.base_path)
            if f.startswith(self.stem) and f.endswith(".joblib")
        )
        if not candidates:
            raise FileNotFoundError(f"no {self.stem}*.joblib under {self.base_path}")
        return self.load(os.path.join(self.base_path, candidates[-1]))

def find_stage3_artifact(
    dirs: Iterable,
    stem: str,
    split_fingerprint: str,
    block_fingerprint: Optional[str] = None,
    features: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[Stage3Artifact, str]:
    """
    The newest ``<stem>_*.joblib`` whose sidecar matches the current split
    (and, if given, block fingerprint and feature list). Raises when nothing
    matches — never falls back to modification time or row counts
    (audit item 11).
    """
    seen: list[str] = []
    matches: list[tuple[str, str]] = []
    for d in dirs:
        for side in glob.glob(os.path.join(str(d), f"{stem}_*.json")):
            try:
                with open(side) as fh:
                    s = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            why = []
            if s.get("split_fingerprint") != split_fingerprint:
                why.append("split")
            if block_fingerprint is not None and s.get("block_fingerprint") != block_fingerprint:
                why.append("block")
            if features is not None and list(s.get("features", [])) != list(features):
                why.append("features")
            seen.append(f"{os.path.basename(side)}: "
                        f"{'match' if not why else 'mismatch ' + '/'.join(why)}")
            jl = side[:-len(".json")] + ".joblib"
            if not why and os.path.exists(jl):
                matches.append((s.get("created_at", ""), jl))
    if not matches:
        raise FileNotFoundError(
            f"No {stem} artifact matches split {split_fingerprint!r}"
            + (f" / block {block_fingerprint[:16]!r}…" if block_fingerprint else "")
            + ". Re-run that arm's Stage 3 on this split.\n  candidates:\n    "
            + ("\n    ".join(seen) if seen else "(none with a JSON sidecar — "
               "pre-PR-33 artifacts are not loadable here)")
        )
    matches.sort()
    path = matches[-1][1]
    art = joblib.load(path)
    if art.split_fingerprint != split_fingerprint:
        raise ValueError(f"{path}: payload fingerprint differs from its sidecar")
    if verbose:
        print(f"  {stem} ← {path}  (matched by split"
              + (" + block" if block_fingerprint else "") + " fingerprint)")
    return art, path


# ==========================================================================
# runner
# ==========================================================================

class Stage3Runner:
    def __init__(self, config, estimator):
        self.config = config
        self.estimator = estimator
        self.selector = F2ThresholdSelector(
            beta=config.beta, low=config.threshold_low,
            high=config.threshold_high, steps=config.threshold_steps,
        )
        self.artifact_: Optional[Stage3Artifact] = None
        self.model_ = None
        self.calibrator_: Optional[Stage3Calibrator] = None

    # ==================================================================
    def fit(
        self,
        block: Stage3Block,
        params: Optional[dict[str, Any]] = None,
        params_source: Optional[str] = None,
        params_provenance: Optional[dict] = None,
        extras: Optional[dict] = None,
    ) -> Stage3Artifact:
        cfg = self.config
        est = self.estimator
        say = print if cfg.verbose else (lambda *a, **k: None)

        # ---- 1. check -------------------------------------------------
        block.check_config(cfg)
        if cfg.expected_features is None:
            say("  ⚠️  config.expected_features is None — the shared feature "
                "contract is NOT being enforced. Pass EBM_FEATURES.")
        y_tr, y_te = block.y_train, block.y_test
        say("\n" + "=" * 78)
        say(f"  STAGE 3 — {cfg.model_family.upper()}  (shared protocol; "
            f"counterpart {cfg.counterpart})")
        say("=" * 78)
        say(f"  train {len(block.X_train):,} | test {len(block.X_test):,} | "
            f"{len(block.features)} features | fitted {block.feature_fit_scope}")
        say(f"  {block.weights.describe()}")
        say(f"  block fingerprint {block.block_fingerprint[:16]}…")

        # ---- 2. tuning ------------------------------------------------
        fold_plan_meta = {k: v for k, v in block.tune_plan.to_dict().items()
                          if k != "fold_id"}
        if params is None:
            tuning = Stage3Tuner(cfg, est, block.tune_folds, fold_plan_meta,
                                 block.feature_fit_scope).tune()
            best_params, cv_score = tuning.best_params, tuning.best_value
            tuning_dict = tuning.to_dict()
        else:
            best_params = est.normalize_params(params)
            cv_score = float("nan")
            tuning_dict = {
                "tuned": False,
                "best_params": best_params,
                "source": "supplied — Optuna not run"
                          + (f" ({params_source})" if params_source else ""),
                "provenance": params_provenance,
            }
            say(f"\n  using supplied hyperparameters (no search): {best_params}")

        # ---- 3. k-fold report -----------------------------------------
        cv_report = CVEvaluator(
            est, cfg.tuning_decision_threshold, cfg.random_state,
            block.feature_fit_scope, cfg.verbose,
        ).run(block.report_folds, best_params)

        # ---- 4. refit -------------------------------------------------
        say("\n  refitting on the full training split")
        self.model_ = est.fit(best_params, block.X_train, y_tr,
                              sample_weight=block.weights.vector)

        # ---- 5. OOF ---------------------------------------------------
        oof = OOFGenerator(est, block.tune_plan, cfg.verbose).generate(
            block.tune_folds, best_params, score_test=True)

        # ---- 6. calibration -------------------------------------------
        self.calibrator_ = Stage3Calibrator(
            cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins,
            cfg.force_calibration, cfg.random_state,
        ).fit(oof.proba, y_tr, block.tune_plan)
        say("\n  " + self.calibrator_.report_.describe().replace("\n", "\n  "))

        # ---- 7. operating point ---------------------------------------
        choice = self.selector.select(
            y_tr, oof.proba, selected_on="out-of-fold training predictions")
        choice_cal = self.selector.select(
            y_tr, self.calibrator_.oof_calibrated_,
            selected_on="out-of-fold training predictions (calibrated space)",
            keep_sweep=False)
        nested = self.selector.select_nested(y_tr, oof.proba, block.tune_plan)
        say(f"\n  operating point: {choice.describe()}")
        say(f"  fold-nested thresholds: {nested['by_fold']} (sd {nested['std']:.4f})")

        # ---- 8. test split --------------------------------------------
        train_proba_ins = est.positive_proba(self.model_, block.X_train)
        test_proba = est.positive_proba(self.model_, block.X_test)
        test_proba_cal = self.calibrator_.transform(test_proba)
        train_proba_cal_ins = self.calibrator_.transform(train_proba_ins)
        fm_proba = oof.test_proba_fold_mean
        fm_proba_cal = self.calibrator_.transform(fm_proba)

        t = choice.threshold
        t_train = nested["per_row"] if cfg.nested_oof_threshold else t
        oof_dec = decision_block(oof.proba, t_train)
        test_dec = decision_block(test_proba, t)
        fm_dec = decision_block(fm_proba, t)
        oracle = self.selector.oracle_on_test(y_te, test_proba)

        def M(y, p, thr, split, space="raw", decision=None):
            return Stage3Metrics.compute(y, p, thr, split=split,
                                         probability_space=space,
                                         ece_bins=cfg.ece_bins,
                                         decision=decision).to_dict()

        metrics = {
            "oof_at_operating_threshold":
                M(y_tr, oof.proba, t, "train (OOF, in-selection)"),
            "oof_at_nested_thresholds":
                M(y_tr, oof.proba, nested["mean"],
                  "train (OOF, fold-nested thresholds)",
                  decision=(oof.proba >= nested["per_row"]).astype(int)),
            "oof_calibrated_at_operating_threshold":
                M(y_tr, self.calibrator_.oof_calibrated_, choice_cal.threshold,
                  "train (OOF, calibrated)", "calibrated"),
            "test_at_operating_threshold":
                M(y_te, test_proba, t, "test (refit)"),
            "test_at_half":
                M(y_te, test_proba, cfg.reference_threshold, "test (refit)"),
            "test_calibrated_at_operating_threshold":
                M(y_te, test_proba_cal, choice_cal.threshold,
                  "test (refit, calibrated)", "calibrated"),
            "test_fold_mean_at_operating_threshold":
                M(y_te, fm_proba, t, "test (fold mean)"),
            "test_fold_mean_calibrated_at_operating_threshold":
                M(y_te, fm_proba_cal, choice_cal.threshold,
                  "test (fold mean, calibrated)", "calibrated"),
            "test_at_oracle_threshold_REFERENCE_ONLY":
                M(y_te, test_proba, oracle.threshold, "test (oracle)"),
        }

        # Calibration diagnostics on test (reported, never used to select).
        yte = np.asarray(y_te).astype(int)
        calib = self.calibrator_.report_.to_dict()
        calib.update({
            "test_ece_raw": calculate_ece(yte, test_proba, cfg.ece_bins),
            "test_ece_calibrated": calculate_ece(yte, test_proba_cal, cfg.ece_bins),
            "test_brier_raw": float(brier_score_loss(yte, test_proba)),
            "test_brier_calibrated": float(brier_score_loss(yte, test_proba_cal)),
            "calibrator_fitted_on": "all OOF rows (test); other folds' OOF rows (train)",
        })

        serving = Stage3ServingModel(
            block.pipeline, block.cleaner, self.model_,
            self.calibrator_.final_calibrator_, est.as_array, block.features,
        )

        # ---- 9. assemble ----------------------------------------------
        artifact = Stage3Artifact(
            stage=cfg.stage,
            model_family=cfg.model_family,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config=cfg.to_dict(),
            mirror_report=cfg.mirror_report(),
            column_prefix=cfg.column_prefix,
            population=cfg.population,
            features=list(block.features),
            index_train=list(block.index_train),
            index_test=list(block.index_test),
            y_train=[int(v) for v in y_tr],
            y_test=[int(v) for v in y_te],
            split_fingerprint=block.split_fingerprint,
            block_fingerprint=block.block_fingerprint,
            feature_fit_scope=block.feature_fit_scope,
            block_settings=dict(block.settings),
            cleaning_report=block.cleaner.report_,
            fold_plan=block.tune_plan.to_dict(),
            report_fold_plan={k: v for k, v in block.report_plan.to_dict().items()
                              if k != "fold_id"},
            class_weights=block.weights.to_dict(),
            tuning=tuning_dict,
            cv_report=cv_report.to_dict(),
            best_params=best_params,
            cv_score=cv_score,
            oof={
                "description": "Out-of-fold over the TRAINING split; per-fold "
                               "features and models. Stage 4 trains on this.",
                "proba": oof.proba,
                "proba_calibrated": self.calibrator_.oof_calibrated_,
                "fold_id": oof.fold_id,
                "threshold": t,
                "threshold_per_row": np.broadcast_to(
                    np.asarray(t_train, dtype=float), oof.proba.shape).copy(),
                "summary": oof.summary(),
                **oof_dec,
            },
            refit={
                "description": "Full-train refit. test_* is held-out; train_* "
                               "is IN-SAMPLE — never fit anything on it.",
                "model": self.model_,
                "calibrator": self.calibrator_.final_calibrator_,
                "cleaner": block.cleaner,
                "feature_pipeline": block.pipeline,
                "serving_model": serving,
                "as_array": bool(est.as_array),
                "estimator": est.to_dict(),
                "train_proba": train_proba_ins,
                "train_proba_calibrated": train_proba_cal_ins,
                "test_proba": test_proba,
                "test_proba_calibrated": test_proba_cal,
                "test_decision": test_dec["decision"],
                "test_margin": test_dec["margin"],
                "test_confidence": test_dec["confidence"],
                "test_state": test_dec["state"],
            },
            fold_mean={
                "description": "Test split scored by each OOF fold model "
                               "through its own fold pipeline, averaged.",
                "test_proba": fm_proba,
                "test_proba_calibrated": fm_proba_cal,
                "test_proba_by_fold": oof.test_proba_by_fold,
                "test_decision": fm_dec["decision"],
                "test_margin": fm_dec["margin"],
                "test_confidence": fm_dec["confidence"],
                "test_state": fm_dec["state"],
            },
            threshold={
                "oof": choice.to_dict(),
                "oof_calibrated_space": choice_cal.to_dict(),
                "oof_nested": {
                    **{k: v for k, v in nested.items() if k != "per_row"},
                    "drives_train_decisions": bool(cfg.nested_oof_threshold),
                },
                "test_oracle": {
                    **oracle.to_dict(),
                    "warning": "Reference only — the pre-PR-33 GLASS rule of "
                               "fitting the threshold on y_test. Nothing "
                               "downstream consumes this value.",
                },
                "sweep": choice.sweep,
            },
            calibration=calib,
            metrics=metrics,
            extras=dict(extras or {}),
        )
        self.artifact_ = artifact
        if cfg.verbose:
            artifact.describe()
        return artifact

    # ==================================================================
    def save(self, verbose: bool = True) -> str:
        if self.artifact_ is None:
            raise ValueError("Call fit() first")
        cfg = self.config
        return Stage3ArtifactStore(
            cfg.artifact_dir, cfg.artifact_stem, cfg.timestamped,
            cfg.write_latest_pointer,
        ).save(self.artifact_, verbose=verbose)


# ==========================================================================
# protocol
# ==========================================================================

FAMILY_SPECIFIC = {"search_space"}

def effective_tuning(artifact) -> dict:
    """
    The tuning run that produced ``artifact.best_params``.

    For a run that reused parameters, that is the provenance record of the
    original search; ``{}`` if no search is on record (explicit params).
    """
    t = artifact.tuning or {}
    if t.get("tuned", "n_trials_requested" in t):
        return t
    prov = t.get("provenance") or {}
    return prov if prov.get("n_trials_requested") is not None else {}

def protocol_diff(a, b) -> list[tuple[str, Any, Any]]:
    """(field, a_value, b_value) for every protocol-relevant difference."""
    diffs: list[tuple[str, Any, Any]] = []
    for k in PROTOCOL_FIELDS:
        va, vb = a.config.get(k), b.config.get(k)
        if va != vb:
            diffs.append((k, va, vb))

    for k in ("split_fingerprint", "block_fingerprint", "feature_fit_scope",
              "population"):
        va, vb = getattr(a, k, None), getattr(b, k, None)
        if va != vb or va is None:
            diffs.append((k, va, vb))

    if list(a.features) != list(b.features):
        diffs.append(("features", a.features, b.features))
    if list(a.index_train) != list(b.index_train):
        diffs.append(("index_train", "…", "…"))
    if list(a.index_test) != list(b.index_test):
        diffs.append(("index_test", "…", "…"))
    fa, fb = a.fold_plan.get("fold_id"), b.fold_plan.get("fold_id")
    if fa is None or fb is None or not np.array_equal(fa, fb):
        diffs.append(("fold_id", "…", "…"))
    if a.class_weights != b.class_weights:
        diffs.append(("class_weights", a.class_weights, b.class_weights))

    ta, tb = effective_tuning(a), effective_tuning(b)
    for k in ("n_trials_requested", "random_state", "feature_fit_scope"):
        va, vb = ta.get(k), tb.get(k)
        if va != vb or va is None:
            diffs.append((f"tuning.{k}", va, vb))

    sa = (a.config.get("search_space") or {})
    sb = (b.config.get("search_space") or {})
    if sa != sb:
        diffs.append(("search_space", "family-specific", "family-specific"))
    return diffs

def assert_matched_protocol(a, b, allow: Iterable[str] = (),
                            verbose: bool = True) -> list[tuple[str, Any, Any]]:
    """Raise unless ``a`` and ``b`` differ only in family-specific fields."""
    allowed = set(allow) | FAMILY_SPECIFIC
    diffs = protocol_diff(a, b)
    blocking = [d for d in diffs if d[0] not in allowed]
    if verbose:
        print(f"  protocol check {a.model_family} vs {b.model_family}: "
              f"{len(diffs)} difference(s), {len(blocking)} blocking")
        for k, va, vb in diffs:
            tag = "  (allowed)" if k in allowed else ""
            print(f"     {k}: {va!r} vs {vb!r}{tag}")
    if blocking:
        raise AssertionError(
            "Stage 3 arms did not run the same protocol:\n"
            + "\n".join(f"  {k}: {va!r} vs {vb!r}" for k, va, vb in blocking)
        )
    return diffs

def check_param_reuse(
    prev,
    config,
    split_fingerprint: str,
    block_fingerprint: Optional[str],
    features: list[str],
) -> dict:
    """
    Validate a previous artifact as a hyperparameter source.

    Returns ``{"params", "source", "provenance"}`` for the runner, or raises.
    """
    problems: list[str] = []
    if prev.split_fingerprint != split_fingerprint:
        problems.append(f"split {prev.split_fingerprint!r} != {split_fingerprint!r}")
    if block_fingerprint is not None and \
            getattr(prev, "block_fingerprint", None) != block_fingerprint:
        problems.append("block fingerprint differs (features or folds changed)")
    if list(prev.features) != list(features):
        problems.append("feature contract differs")
    tuning = effective_tuning(prev)
    if not tuning:
        problems.append("the source artifact has no Optuna search on record")
    else:
        if tuning.get("n_trials_requested") != config.n_trials:
            problems.append(f"n_trials {tuning.get('n_trials_requested')} != "
                            f"{config.n_trials}")
        if tuning.get("feature_fit_scope") != config.feature_fit_scope:
            problems.append(f"feature_fit_scope {tuning.get('feature_fit_scope')!r}"
                            f" != {config.feature_fit_scope!r}")
    prev_cfg = prev.config or {}
    for k in ("n_tune_folds", "random_state", "beta", "tuning_decision_threshold",
              "pruner_startup_trials", "pruner_warmup_steps", "class_weight"):
        if prev_cfg.get(k) != getattr(config, k):
            problems.append(f"{k} {prev_cfg.get(k)!r} != {getattr(config, k)!r}")
    space = config.search_space.to_dict() if hasattr(config.search_space, "to_dict") \
        else config.search_space
    if prev_cfg.get("search_space") != space:
        problems.append("search space differs")
    if problems:
        raise ValueError(
            "Cannot reuse hyperparameters from this artifact:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-tune with PARAM_SOURCE='tune'."
        )
    prov = {k: v for k, v in tuning.items() if k != "top_trials"}
    return {
        "params": dict(prev.best_params),
        "source": f"artifact created {prev.created_at} "
                  f"(tuned CV F2 {tuning.get('best_value', float('nan')):.6f})",
        "provenance": prov,
    }


# ==========================================================================
# compare
# ==========================================================================

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