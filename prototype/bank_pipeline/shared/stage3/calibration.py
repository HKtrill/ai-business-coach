"""
shared.stage3.calibration
=========================
OOF-ECE-gated probability calibration (isotonic or Platt), strictly nested on
the training side (audit F22), and the calibration diagnostics recorded in the
artifact. ECE itself is ``shared.metrics.calculate_ece``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from shared.metrics import calculate_ece as _shared_ece

from .data import FoldPlan, check_inner_scores


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
    # training side (F22): one gate and map per outer fold, from inner OOF
    fold_applied: Optional[list] = None
    fold_ece_inner: Optional[list] = None
    fold_gate_measured_on: str = ("each outer fold: inner out-of-fold scores of its "
                                  "training rows")

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        head = (f"{self.method} applied" if self.applied
                else f"not applied ({self.reason})")
        folds = ("" if self.fold_applied is None else
                 f"\n   per-fold gates (inner OOF): "
                 f"{sum(self.fold_applied)}/{len(self.fold_applied)} applied, ECE "
                 + ", ".join(f"{e:.4f}" for e in self.fold_ece_inner))
        return (
            f"calibration: {head}\n"
            f"   ECE   {self.ece_before:.4f} → {self.ece_after:.4f} "
            f"(gate {self.ece_threshold}, {self.ece_bins} bins, OOF)\n"
            f"   [{self.n_zero_after} calibrated rows at p=0]\n"
            f"   Brier {self.brier_before:.4f} → {self.brier_after:.4f}"
            + folds
        )


class Stage3Calibrator:
    """
    Gate on ECE, then fit isotonic (or Platt) leakage-safely.

    Test side (the final model): the gate reads the ECE of all OOF rows and
    the map is fitted on all OOF rows, as in Stage 1. ``applied_`` and
    ``transform(p)`` report and apply that decision.

    Training side (audit F22), what Stage 4 consumes: for each outer fold k
    the whole procedure is replayed on the inner out-of-fold scores of fold
    k's TRAINING rows (``OOFGenerator.generate_inner``): fold k's own gate on
    their ECE, then its own map fitted on them, applied to fold k's held-out
    OOF scores. Neither fold k's labels nor any score from a model trained on
    them (the other folds' outer OOF) reaches its calibrated output, so the
    output may mix calibrated and identity folds, as Stage 1's may mix
    families.

    After ``fit``:
      ``oof_calibrated_``     strictly nested calibrated OOF training probabilities
      ``inner_calibrated_``   (n_folds, n_rows): fold k's map applied to its
                              inner scores; fold k's calibrated-space threshold
                              is selected on these
      ``fold_applied_``       the per-fold gate decisions
      ``transform(p)``        refit-model probabilities (i.e. test) through the
                              all-OOF map (identity when the gate is not breached)

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
        self.inner_calibrated_: Optional[np.ndarray] = None
        self.fold_applied_: Optional[list] = None
        self.report_: Optional[CalibrationReport] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        oof_proba: np.ndarray,
        y,
        folds: FoldPlan,
        inner_proba: np.ndarray,
    ) -> "Stage3Calibrator":
        """
        ``oof_proba``: outer out-of-fold scores of every training row.
        ``inner_proba``: ``(n_folds, n_rows)`` inner out-of-fold scores, row k
        covering fold k's training rows only (``check_inner_scores``).
        """
        p = np.asarray(oof_proba, dtype=float)
        y_arr = np.asarray(y).astype(int)
        if len(p) != len(y_arr):
            raise ValueError(f"length mismatch: proba {len(p)}, y {len(y_arr)}")
        if np.isnan(p).any():
            raise ValueError(
                "OOF probabilities contain NaN — calibrate on the scored array."
            )
        folds.assert_matches(len(p), "calibration")
        inner = check_inner_scores(inner_proba, folds, len(p))

        # --- test side: gate and map from all OOF rows ----------------------
        ece_before = calculate_ece(y_arr, p, self.ece_bins)
        brier_before = float(brier_score_loss(y_arr, p))
        self.applied_ = bool(self.force or ece_before > self.ece_threshold)
        self.final_calibrator_ = self._fit_mapper(p, y_arr) if self.applied_ else None

        # --- training side: gate and map per outer fold, from inner OOF ------
        calibrated = np.full(len(p), np.nan, dtype=float)
        inner_cal = np.full(inner.shape, np.nan, dtype=float)
        fold_applied, fold_ece = [], []
        for k, (train_idx, val_idx) in enumerate(folds):
            s, y_k = inner[k, train_idx], y_arr[train_idx]
            ece_k = calculate_ece(y_k, s, self.ece_bins)
            gate_k = bool(self.force or ece_k > self.ece_threshold)
            if gate_k:
                mapper = self._fit_mapper(s, y_k)
                calibrated[val_idx] = self._apply(mapper, p[val_idx])
                inner_cal[k, train_idx] = self._apply(mapper, s)
            else:
                calibrated[val_idx] = p[val_idx]
                inner_cal[k, train_idx] = s
            fold_applied.append(gate_k)
            fold_ece.append(float(ece_k))
        if np.isnan(calibrated).any():
            raise AssertionError(
                "Nested calibration left rows unmapped — check the fold plan."
            )
        self.oof_calibrated_ = calibrated
        self.inner_calibrated_ = inner_cal
        self.fold_applied_ = fold_applied

        rel = ">" if self.applied_ else "<="
        reason = f"OOF ECE {ece_before:.4f} {rel} gate {self.ece_threshold}"
        if self.force:
            reason = f"forced ({reason})"
        self.report_ = CalibrationReport(
            method=self.method,
            applied=self.applied_,
            reason=reason,
            ece_threshold=self.ece_threshold,
            ece_bins=self.ece_bins,
            ece_before=ece_before,
            ece_after=calculate_ece(y_arr, calibrated, self.ece_bins),
            brier_before=brier_before,
            brier_after=float(brier_score_loss(y_arr, calibrated)),
            n_rows=len(p),
            n_zero_after=int((calibrated == 0.0).sum()),
            fold_applied=list(fold_applied),
            fold_ece_inner=list(fold_ece),
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


def calibration_diagnostics(
    report: CalibrationReport,
    y_test,
    test_proba: np.ndarray,
    test_proba_calibrated: np.ndarray,
    ece_bins: int,
) -> dict:
    """
    The artifact's ``calibration`` block: the OOF ``CalibrationReport`` plus
    test-split ECE / Brier before and after calibration. Reported only —
    never used to select anything.
    """
    yte = np.asarray(y_test).astype(int)
    calib = report.to_dict()
    calib.update({
        "test_ece_raw": calculate_ece(yte, test_proba, ece_bins),
        "test_ece_calibrated": calculate_ece(yte, test_proba_calibrated, ece_bins),
        "test_brier_raw": float(brier_score_loss(yte, test_proba)),
        "test_brier_calibrated": float(brier_score_loss(yte, test_proba_calibrated)),
        "calibrator_fitted_on": ("all OOF rows (test); per outer fold, the inner OOF scores "
                                 "of its training rows (train, F22)"),
    })
    return calib
