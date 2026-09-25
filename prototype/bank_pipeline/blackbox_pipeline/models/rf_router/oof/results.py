"""
The container the scorer returns, and the structural invariants it asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["OOFResult"]


@dataclass
class OOFResult:
    """
    Out-of-fold probabilities over a scored population.

    Attributes
    ----------
    proba
        ``P(y = 1 | x)``, length ``n_rows`` of the FULL population handed to the
        scorer. Positions outside ``scored_mask`` hold ``np.nan``.
    scored_mask
        Boolean over the full population: which rows were scored. All True for a
        full-population run; the remainder for a remainder run.
    fold_id
        Which fold held out each scored row. ``-1`` where unscored.
    n_folds, random_state, stratified
        Provenance of the fold assignment.
    """

    proba: np.ndarray
    scored_mask: np.ndarray
    fold_id: np.ndarray
    n_folds: int
    random_state: int
    stratified: bool
    fit_sizes: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def n_scored(self) -> int:
        return int(self.scored_mask.sum())

    def scored_proba(self) -> np.ndarray:
        """Probabilities for scored rows only, in population order."""
        return self.proba[self.scored_mask]

    def scored_labels(self, y: pd.Series) -> np.ndarray:
        """Labels for scored rows only, in population order."""
        return np.asarray(y)[self.scored_mask]

    # ------------------------------------------------------------------
    def assert_valid(self) -> None:
        """
        Structural invariants. Cheap; called on every construction.

        Does not and cannot prove that a fold model never saw a row — that is
        enforced by construction in ``OOFScorer`` and verified independently by
        ``tests/test_rf_router_leakage.py``.
        """
        n = len(self.proba)
        if not (len(self.scored_mask) == len(self.fold_id) == n):
            raise AssertionError(
                "OOFResult array length mismatch: "
                f"proba={n}, mask={len(self.scored_mask)}, fold={len(self.fold_id)}"
            )

        scored = self.scored_mask
        if np.isnan(self.proba[scored]).any():
            raise AssertionError(
                "OOF probability is NaN for a row marked as scored — a fold "
                "failed to write its predictions."
            )
        if not np.isnan(self.proba[~scored]).all():
            raise AssertionError(
                "OOF probability present for a row marked as unscored — the "
                "scored mask and the probability array disagree."
            )
        if (self.fold_id[scored] < 0).any():
            raise AssertionError("Scored row has no fold assignment.")
        if (self.fold_id[~scored] != -1).any():
            raise AssertionError("Unscored row carries a fold assignment.")

        bad = (self.proba[scored] < 0.0) | (self.proba[scored] > 1.0)
        if bad.any():
            raise AssertionError(
                f"{int(bad.sum())} OOF probabilities outside [0, 1]."
            )

        counts = np.bincount(self.fold_id[scored], minlength=self.n_folds)
        if len(counts) != self.n_folds or (counts == 0).any():
            raise AssertionError(
                f"Expected {self.n_folds} non-empty folds, "
                f"got counts {counts.tolist()}"
            )
        if int(counts.sum()) != int(scored.sum()):
            raise AssertionError(
                "Fold assignments do not partition the scored rows: "
                f"{int(counts.sum())} assignments for {int(scored.sum())} rows."
            )