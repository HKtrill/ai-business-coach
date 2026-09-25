"""
shared.stage3.calibration
=========================
OOF-ECE-gated, fold-nested probability calibration (isotonic or Platt), and
the calibration diagnostics recorded in the artifact. ECE itself is
``shared.metrics.calculate_ece``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from shared.metrics import calculate_ece as _shared_ece

from .data import FoldPlan


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
        "calibrator_fitted_on": "all OOF rows (test); other folds' OOF rows (train)",
    })
    return calib
