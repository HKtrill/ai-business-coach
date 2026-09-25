"""
blackbox_pipeline.models.rf_router.forest
==========================================
Random Forest fitting. Nothing else.

This module knows how to build and fit one forest with the same recipe the
GLASS RF stage uses (``glass_pipeline.glass_router.rf.rf_training._build_pipe`` /
``_make_sample_weights``): ``class_weight="balanced"`` on the estimator plus an
explicit per-sample weight for the minority class.

It holds no thresholds, no routing logic, and no knowledge of passes. Anything
that decides *what a probability means* lives in ``thresholds.py`` / ``bands.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

__all__ = ["ForestTrainer", "make_sample_weights", "build_pipe"]


def make_sample_weights(y: pd.Series, minority_weight: float) -> np.ndarray:
    """Per-sample weights, mirroring ``rf_training._make_sample_weights``."""
    y = pd.Series(np.asarray(y))
    return y.map({0: 1.0, 1: float(minority_weight)}).to_numpy(dtype=float)


def build_pipe(params: dict, random_state: int, n_jobs: int = -1) -> Pipeline:
    """
    Construct an unfitted pipeline identical in shape to the GLASS RF stage.

    The single ``clf`` step is kept (rather than a bare estimator) so that
    ``pipe.named_steps["clf"]`` works the same way downstream and so a scaler
    could be inserted later without changing every call site. Binary inputs need
    no scaling, so there is nothing before ``clf`` today.

    Determinism note
    ----------------
    With ``n_jobs != 1`` sklearn accumulates per-tree probabilities under
    ``Parallel``, so the summation ORDER varies between runs and
    ``predict_proba`` can differ in the last ULP. The forest itself is fully
    seeded — the trees are identical run to run — but a solved threshold is read
    off a data point's probability, so ``t1`` / ``t2`` are reproducible to about
    1e-12 rather than bit-identical.

    This does not affect any reported metric: band MEMBERSHIP is stable, because
    the wobble is far smaller than the gap between adjacent distinct
    probabilities. It only means a saved threshold and a re-fitted one may
    differ in their last digits. Pass ``n_jobs=1`` if an experiment needs
    bit-exact reproduction and can afford the wall clock.
    """
    return Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=params["max_depth"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=params["max_features"],
                    class_weight=params.get("class_weight", "balanced"),
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            )
        ]
    )


class ForestTrainer:
    """
    Fits one forest on whatever population it is handed.

    Parameters
    ----------
    params
        Hyperparameters in ``rf_training._normalize_params`` shape.
    random_state
        Seed passed to the estimator.
    n_jobs
        Passed to the estimator. Use ``1`` inside a parallel fold loop.
    """

    def __init__(self, params: dict, random_state: int = 42, n_jobs: int = -1):
        self.params = dict(params)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    def new_pipe(self, random_state: Optional[int] = None) -> Pipeline:
        """An unfitted pipeline. Each fold must get its own."""
        return build_pipe(
            self.params,
            self.random_state if random_state is None else int(random_state),
            n_jobs=self.n_jobs,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: Optional[int] = None,
    ) -> Pipeline:
        """
        Fit a fresh pipeline on ``(X, y)`` and return it.

        A NEW pipeline is constructed on every call. Reusing a fitted estimator
        across folds is one of the classic ways OOF predictions silently become
        in-sample, so the only object that ever leaves this method is one that
        was created inside it.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )
        if len(X) == 0:
            raise ValueError("Cannot fit a forest on an empty population")

        y_arr = pd.Series(np.asarray(y))
        if y_arr.nunique() < 2:
            raise ValueError(
                "Cannot fit a forest on a single-class population "
                f"(classes present: {sorted(y_arr.unique().tolist())})"
            )

        pipe = self.new_pipe(random_state)
        sw = make_sample_weights(y_arr, self.params.get("minority_weight", 1.0))
        pipe.fit(X, np.asarray(y), clf__sample_weight=sw)
        return pipe

    # ------------------------------------------------------------------
    @staticmethod
    def positive_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
        """
        ``P(y = 1 | x)`` as a 1-D float array, robust to class ordering.

        ``predict_proba`` columns follow ``classes_``, which is sorted, so
        column 1 is the positive class whenever both classes were seen in
        training. ``ForestTrainer.fit`` refuses single-class populations, so the
        lookup below always finds class 1 — the explicit search is there so a
        future caller that bypasses ``fit`` fails loudly instead of silently
        returning ``P(y = 0)``.
        """
        clf = pipe.named_steps["clf"]
        classes = list(clf.classes_)
        if 1 not in classes:
            raise ValueError(
                f"Fitted forest never saw the positive class; classes_={classes}"
            )
        col = classes.index(1)
        return np.asarray(pipe.predict_proba(X)[:, col], dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ForestTrainer(n_estimators={self.params.get('n_estimators')}, "
            f"max_depth={self.params.get('max_depth')}, "
            f"min_samples_leaf={self.params.get('min_samples_leaf')}, "
            f"seed={self.random_state})"
        )
