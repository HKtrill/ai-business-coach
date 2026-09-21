"""
blackbox_pipeline.models.xgb.calibration
=========================================
Probability calibration, gated and fitted without touching the test split.

Mirrored from GLASS
-------------------
* isotonic regression as the calibration family
* a gate: calibrate only when ECE exceeds 0.05
* the ECE estimator itself, reproduced bin-for-bin from
  ``glass_pipeline.ebm.calibration.calculate_ece`` so the two arms' ECE numbers
  are the same statistic and can be put in one table

Two departures, both recorded in the artifact
---------------------------------------------
1. **The gate is measured out-of-fold.** GLASS computes
   ``calculate_ece(y_train, model.predict_proba(X_train))`` — the model scoring
   its own training rows. An in-sample reliability curve is optimistic close to
   by construction, so that gate tends to report "already calibrated" and skip
   isotonic even when held-out data disagrees. Here the gate reads the OOF
   probabilities, which is the honest estimate of the same quantity.

2. **The calibrated training column is itself out-of-fold.** GLASS calls
   ``CalibratedClassifierCV(model, cv=3)``, which clones and refits the
   estimator three times and averages them — so its "calibrated" probabilities
   come from three models that are not the model it reports, and its calibrated
   train column is partly in-sample. Here, fold *k*'s calibrated values come
   from a mapper fitted on the OTHER folds' OOF probabilities and labels, so a
   row's own label never enters its raw score or its calibration map. A
   separate calibrator, fitted on all OOF probabilities, maps the refit
   model's test (and in-sample train) probabilities.

Caveat: the refit model is trained on 100% of train while the calibrator is
fitted on OOF scores from ~(k-1)/k-sized models — the standard assumption
that their score distributions are close.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from .folds import FoldPlan

__all__ = ["calculate_ece", "CalibrationReport", "Stage3Calibrator"]


def calculate_ece(
    y_true, y_prob, n_bins: int = 10, *, include_zero: bool = False
) -> float:
    """
    Expected calibration error.

    Sum over bins of ``|mean(y) - mean(p)| * (rows in bin / n)``.

    Default (``include_zero=False``) reproduces the GLASS implementation
    exactly, including its half-open binning ``(bins[i], bins[i+1]]`` — a
    probability of exactly 0.0 falls in no bin. Use this for any number put
    beside the EBM's.

    ``include_zero=True`` closes the first bin, ``[0, bins[1]]``, so every
    row counts. Identical to the default unless some ``p == 0.0`` — which
    isotonic output often has, where the default can read low. The stage
    records both (``*_full`` fields).
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if include_zero and i == 0:
            mask |= y_prob == bins[0]
        if mask.any():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(ece)


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
    # Every-row ECE (``include_zero=True``); the gate uses ``ece_before``.
    ece_before_full: float = float("nan")
    ece_after_full: float = float("nan")
    n_zero_after: int = 0  # calibrated rows at exactly 0.0 (GLASS ECE skips them)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        head = (f"{self.method} applied" if self.applied
                else f"not applied ({self.reason})")
        return (
            f"calibration: {head}\n"
            f"   ECE   {self.ece_before:.4f} → {self.ece_after:.4f} "
            f"(gate {self.ece_threshold}, {self.ece_bins} bins, OOF)\n"
            f"   ECE (all rows) {self.ece_before_full:.4f} → "
            f"{self.ece_after_full:.4f}  [{self.n_zero_after} rows at p=0]\n"
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
        ece_before_full = calculate_ece(y_arr, p, self.ece_bins,
                                        include_zero=True)
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
                ece_before_full=ece_before_full,
                ece_after_full=ece_before_full,
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
            ece_before_full=ece_before_full,
            ece_after_full=calculate_ece(y_arr, calibrated, self.ece_bins,
                                         include_zero=True),
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