"""
blackbox_pipeline.models.xgb.folds
===================================
The canonical cross-validation partition.

One ``FoldPlan`` is built per Stage 3 run and used by everything that needs
folds: the Optuna objective, the OOF generation that feeds Stage 4, the
calibration fit and the threshold search. Because it is one object with one
persisted ``fold_id`` array, "the same fixed folds were used throughout" is a
property of the artifact rather than a claim in a docstring.

The partition is ``StratifiedKFold(shuffle=True, random_state=...)``, which is
what the EBM's tuning loop constructs. That splitter depends only on ``y``, the
fold count and the seed, so given the same training split in the same row order
the EBM and XGBoost arms receive bit-identical folds.

Indices are POSITIONAL, never pandas labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

__all__ = ["FoldPlan"]


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

        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=random_state)
            if stratify
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )
        fold_id = np.full(n, -1, dtype=int)
        for k, (_, val_idx) in enumerate(splitter.split(np.zeros((n, 1)), y_arr)):
            fold_id[val_idx] = k

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