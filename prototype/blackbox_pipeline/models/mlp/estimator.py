"""
blackbox_pipeline.models.mlp.estimator

The sklearn-compatible MLP itself. No tuning, no calibration, no thresholds.

``Stage1MLPClassifier`` wraps ``MLPClassifier`` with the two things Stage 1
needs and sklearn does not give for free:

* positive oversampling inside each fit, at a tuned ``pos_ratio``;
* early stopping on ROC-AUC, measured on an inner stratified split of the fit's
  own rows, restoring the best epoch's weights.

Notes
-----
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
    Duplicate positive rows until they reach ``pos_ratio`` of the negative count.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Binary labels, 0/1.
    pos_ratio : float
        Target positives-to-negatives ratio.
    rng : numpy.random.Generator
        Source of randomness for sampling the duplicates.

    Returns
    -------
    X_out, y_out : numpy.ndarray
        The inputs with duplicated positives appended, or the inputs unchanged.

    Notes
    -----
    Returns the inputs untouched when the target ratio is already met, so a
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

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default (16,)
        Units per hidden layer.
    activation : {'relu', 'tanh', 'logistic', 'identity'}, default 'relu'
        Hidden-layer activation. Passed straight to ``MLPClassifier``, so any
        value it accepts works here; the Optuna search space in :mod:`~.tuning`
        only ever draws ``'relu'`` or ``'tanh'``.
    alpha : float, default 1e-4
        L2 penalty.
    learning_rate_init : float, default 1e-3
        Initial Adam learning rate.
    batch_size : int, default 256
        Minibatch size, clipped to the row count. One epoch is one pass over
        the oversampled rows in minibatches of this size.
    pos_ratio : float, default 0.5
        Target positives-to-negatives ratio after oversampling. See
        :func:`oversample_positives`.
    max_epochs : int, default 200
        Epoch ceiling.
    patience : int, default 15
        Epochs without inner-validation improvement before stopping. An epoch
        counts as an improvement only if it beats the running best by more than
        1e-4, so noise below that margin still burns patience.
    val_fraction : float, default 0.15
        Share of the fit's rows held out for early stopping.
    random_state : int, default 42
        Seed for the inner split, oversampling and weight initialisation.

    Attributes
    ----------
    net_ : sklearn.neural_network.MLPClassifier
        The underlying network, weights restored to the best epoch.
    best_epoch_ : int
        Epoch that produced those weights.
    best_val_roc_auc_ : float
        Inner-validation ROC-AUC at that epoch.
    history_ : pandas.DataFrame
        Per-epoch ``epoch``, ``train_loss``, ``val_roc_auc``.
    classes_ : numpy.ndarray
        Always ``array([0, 1])``.
    n_features_in_ : int
        Feature count seen during ``fit``.

    Notes
    -----
    Every constructor argument is stored unmodified on ``self`` and nothing is
    computed in ``__init__``. That is what lets ``sklearn.base.clone``
    round-trip the estimator, which ``cross_val_predict`` and
    ``CalibratedClassifierCV`` both rely on.

    The epoch loop is driven by ``MLPClassifier.partial_fit``, where one call is
    one full pass over the rows in ``batch_size`` minibatches. Adam's optimizer
    state persists across calls, so the loop is a genuine multi-epoch fit rather
    than repeated cold starts. sklearn's own ``early_stopping`` is left off — it
    is ignored under ``partial_fit``, which is why the stopping logic is
    hand-rolled here.
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
        """
        Fit the network, oversampling positives and early-stopping on ROC-AUC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled features. Scaling is the caller's job.
        y : array-like of shape (n_samples,)
            Binary labels, 0/1. Both classes must be present.

        Returns
        -------
        Stage1MLPClassifier
            ``self``, with the fitted attributes set.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` differ in length, or ``y`` holds one class.
        RuntimeError
            If no epoch was ever recorded as best.
        """
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
        """
        Class probabilities from the best-epoch weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled features, same column count and order as ``fit`` saw.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Columns are ``P(y=0)`` and ``P(y=1)``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        ValueError
            If the feature count does not match ``n_features_in_``.
        """
        check_is_fitted(self, "net_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return self.net_.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """
        Hard labels at the fixed 0.5 cut.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled features.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Values drawn from ``classes_``.

        Notes
        -----
        0.5 is hard-coded because this estimator is not where the operating
        point is decided. The tuned Stage 1 threshold lives on
        ``CalibratedStage1MLP.optimal_threshold`` and is applied by
        ``CalibratedStage1MLP.predict``.
        """
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    # ------------------------------------------------------------------
    @property
    def n_trainable_params_(self) -> int:
        """
        Total weights plus biases across every layer.

        Returns
        -------
        int

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If accessed before ``fit``.
        """
        check_is_fitted(self, "net_")
        return int(
            sum(w.size for w in self.net_.coefs_)
            + sum(b.size for b in self.net_.intercepts_)
        )
