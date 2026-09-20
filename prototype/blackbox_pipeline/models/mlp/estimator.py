"""
blackbox_pipeline.models.mlp.estimator
=======================================
The sklearn-compatible MLP itself. No tuning, no calibration, no thresholds.

``Stage1MLPClassifier`` is a thin wrapper around ``MLPClassifier`` that adds the
two things Stage 1 needs and sklearn does not give for free:

* **positive oversampling** inside each fit, at a tuned ``pos_ratio``;
* **early stopping on ROC-AUC**, measured on an inner stratified split of the
  fit's own rows, restoring the best epoch's weights.

Both happen inside ``fit``, which is what makes them safe under
``cross_val_predict`` and ``CalibratedClassifierCV``: every clone re-derives its
own inner split and its own oversampled copy from whatever rows it was handed,
so no information crosses a fold boundary.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted

__all__ = ["Stage1MLPClassifier", "oversample_positives"]


def oversample_positives(
    X: np.ndarray,
    y: np.ndarray,
    pos_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Duplicate positives until they reach ``pos_ratio`` of the negative count.

    Returns the inputs unchanged when the target ratio is already met, so a
    ``pos_ratio`` below the natural rate is a no-op rather than a downsample.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    extra = int(round(pos_ratio * len(neg))) - len(pos)
    if extra <= 0 or len(pos) == 0:
        return X, y
    idx = np.concatenate([np.arange(len(y)), rng.choice(pos, size=extra, replace=True)])
    return X[idx], y[idx]


class Stage1MLPClassifier(ClassifierMixin, BaseEstimator):
    """
    Small MLP with positive oversampling and inner-split early stopping.

    Every constructor argument is stored unmodified on ``self`` and nothing is
    computed in ``__init__`` — that is what lets ``sklearn.base.clone`` round-trip
    the estimator, which ``cross_val_predict`` and ``CalibratedClassifierCV``
    both rely on.

    Fitted attributes
    -----------------
    net_          the underlying MLPClassifier, weights restored to the best epoch
    best_epoch_   epoch that produced those weights
    history_      per-epoch train loss and inner-validation ROC-AUC
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (16,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        batch_size: int = 256,
        pos_ratio: float = 0.5,
        max_epochs: int = 200,
        patience: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "Stage1MLPClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Stage1MLPClassifier needs both classes; got {np.unique(y).tolist()}"
            )

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Inner split for early stopping — derived from THIS fit's rows only.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y,
            test_size=self.val_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        # Oversampling happens after the split, so duplicated positives can
        # never appear on both sides of it.
        X_fit, y_fit = oversample_positives(X_fit, y_fit, self.pos_ratio, rng)

        net = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            shuffle=True,
            random_state=self.random_state,
        )

        best_score, best_state, best_epoch, wait = -np.inf, None, 0, 0
        history = []

        for epoch in range(1, self.max_epochs + 1):
            net.partial_fit(X_fit, y_fit, classes=self.classes_)
            val_auc = float(roc_auc_score(y_val, net.predict_proba(X_val)[:, 1]))
            history.append({
                "epoch": epoch,
                "train_loss": float(net.loss_),
                "val_roc_auc": val_auc,
            })

            if val_auc > best_score + 1e-4:
                best_score, best_epoch, wait = val_auc, epoch, 0
                best_state = (
                    [w.copy() for w in net.coefs_],
                    [b.copy() for b in net.intercepts_],
                )
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Early stopping recorded no best epoch — max_epochs may be 0."
            )

        net.coefs_, net.intercepts_ = best_state
        self.net_ = net
        self.best_epoch_ = int(best_epoch)
        self.best_val_roc_auc_ = float(best_score)
        self.history_ = pd.DataFrame(history)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return self.net_.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    # ------------------------------------------------------------------
    @property
    def n_trainable_params_(self) -> int:
        check_is_fitted(self, "net_")
        return int(
            sum(w.size for w in self.net_.coefs_)
            + sum(b.size for b in self.net_.intercepts_)
        )
