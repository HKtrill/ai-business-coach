"""
blackbox_pipeline.models.mlp.stage

Stage 1 orchestration — the calibrated MLP stage.

Mirrors the GLASS ``CalibratedLRStage`` protocol step for step:

0. ``StandardScaler`` fitted on the full training split.
1. Optuna tuning, objective = mean CV ROC-AUC. Training only.
2. Final MLP fitted on the full training split with the best parameters.
3. Probability calibration via GLASS ``glass_pipeline.lr.calibration.fit_calibration``.
4. Decision threshold from OUT-OF-FOLD calibrated probabilities, F2 sweep over
   the same grid GLASS uses.
5. Reporting via ``shared.metrics`` (``compute_metrics`` / ``metrics_table``),
   the same implementation GLASS uses.

Notes
-----
This module composes; it does not compute. Every numerical step lives in
:mod:`~.estimator`, :mod:`~.tuning`, :mod:`~.calibration`, :mod:`~.thresholds`
or :mod:`~.evaluation`.

Train/test protocol: ``fit`` takes the training split and nothing else. The
scaler, the hyperparameters, the calibrator and the threshold are all derived
from it. The only method that accepts test data is ``to_stage_output``, which
scores it and fits nothing.

PR 33: ``fit`` now also records ``train_fold_id`` and ``y_train_``, and
``to_stage_output`` emits the shared ``StageOutput`` contract — the same object
GLASS emits, so Stage 4 joins the arms on the index rather than on position.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from shared.stage_io import (
    THRESHOLD_SOURCES,
    StageOutput,
    fold_assignment,
)

from .calibration import fit_stage1_calibration
from .config import Stage1MLPConfig
from .estimator import Stage1MLPClassifier
from .features import STAGE1_FEATURES, check_labels, select_features
from .evaluation.metrics import metrics_table as _metrics_table
from .thresholds import oof_probabilities, optimize_threshold_cv
from .tuning import params_to_kwargs, tune_stage1_mlp_auc

__all__ = ["CalibratedStage1MLP"]


class CalibratedStage1MLP:
    """
    Optuna (ROC-AUC) → final fit → calibration → CV F2 threshold.

    Parameters
    ----------
    calibration_method : {'auto', 'sigmoid', 'isotonic'}, default 'auto'
        See :class:`~.config.Stage1MLPConfig`.
    cv_folds : int, default 10
        Folds for tuning, calibration and the out-of-fold probabilities.
    n_trials : int, default 100
        Optuna trials.
    random_state : int, default 42
        Seed for Optuna, every split and every fit.
    max_epochs : int, default 200
        Epoch ceiling for a single MLP fit.
    patience : int, default 15
        Early-stopping patience.
    n_jobs : int, default -1
        Parallelism for ``cross_val_predict``.
    config : Stage1MLPConfig, optional
        A ready-made config. When given it wins outright: every loose keyword
        above and every ``config_overrides`` entry is ignored, silently and
        without error. Pass one or the other, not both.
    **config_overrides
        Any other ``Stage1MLPConfig`` field (``val_fraction``,
        ``threshold_beta``, ``threshold_grid``, ``verbose``, ``strict_oof``).
        An unknown name raises ``TypeError`` from the dataclass.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted on the training split.
    model : Stage1MLPClassifier or None
        The final MLP.
    calibrated_model : estimator or None
        The calibrator wrapping ``model``.
    best_params : dict
        Winning Optuna parameters.
    best_cv_roc_auc : float or None
        Mean CV ROC-AUC of the winning trial.
    tuning_trials : pandas.DataFrame or None
        One row per Optuna trial.
    calibration_method : str
        The method actually used — GLASS's pick when ``'auto'`` was requested.
        ``self.config.calibration_method`` still holds the REQUEST, so after a
        fit with ``'auto'`` the two disagree on purpose. Report this one.
    calibration_metrics : dict
        GLASS's calibration diagnostics.
    optimal_threshold : float
        Operating point from the in-stage F2 sweep. 0.5 until fitted.
    cv_f2 : float or None
        F2 at that threshold, on out-of-fold probabilities.
    threshold_sweep : pandas.DataFrame or None
        The full in-stage sweep.
    proba_train_oof : pandas.Series or None
        Training out-of-fold calibrated ``P(y = 1)``.
    oof_provenance_ : str or None
        What the leakage guard found. See :mod:`~.calibration`.
    feature_names_ : list of str
        ``STAGE1_FEATURES``.
    fitted : bool
        Whether ``fit`` has completed.

    Notes
    -----
    Constructor arguments are kept as loose keywords, matching the original
    class, so existing notebook cells run unchanged. The config fields are also
    mirrored onto the instance for the same reason — cells read
    ``stage1.cv_folds`` and ``stage1.calibration_method`` directly.

    Examples
    --------
    >>> stage = CalibratedStage1MLP(cv_folds=10, n_trials=100).fit(X_train, y_train)
    >>> proba = stage.predict_proba(X_test)
    >>> pred = stage.predict(X_test)          # at stage.optimal_threshold
    """

    def __init__(
        self,
        calibration_method: str = "auto",
        cv_folds: int = 10,
        n_trials: int = 100,
        random_state: int = 42,
        max_epochs: int = 200,
        patience: int = 15,
        n_jobs: int = -1,
        *,
        config: Optional[Stage1MLPConfig] = None,
        **config_overrides,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = Stage1MLPConfig(
                calibration_method=calibration_method,
                cv_folds=cv_folds,
                n_trials=n_trials,
                random_state=random_state,
                max_epochs=max_epochs,
                patience=patience,
                n_jobs=n_jobs,
                **config_overrides,
            )

        # Mirrored onto the instance for backward compatibility: existing cells
        # read stage1.cv_folds, stage1.calibration_method and so on directly.
        self.calibration_method = self.config.calibration_method
        self.cv_folds = self.config.cv_folds
        self.n_trials = self.config.n_trials
        self.random_state = self.config.random_state
        self.max_epochs = self.config.max_epochs
        self.patience = self.config.patience
        self.n_jobs = self.config.n_jobs

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Stage1MLPClassifier] = None
        self.calibrated_model = None
        self.best_params: Dict = {}
        self.best_cv_roc_auc: Optional[float] = None
        self.tuning_trials: Optional[pd.DataFrame] = None
        self.calibration_metrics: Dict = {}
        self.optimal_threshold: float = 0.5
        self.cv_f2: Optional[float] = None
        self.threshold_sweep: Optional[pd.DataFrame] = None
        self.proba_train_oof: Optional[pd.Series] = None
        self.train_fold_id: Optional[pd.Series] = None
        self.y_train_: Optional[pd.Series] = None
        self.oof_provenance_: Optional[str] = None
        self.feature_names_ = list(STAGE1_FEATURES)
        self.fitted = False

    #: How ``optimal_threshold`` was chosen after a plain ``fit``.
    THRESHOLD_SOURCE = "in_stage_f2_cv"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CalibratedStage1MLP":
        """
        Run the five-step protocol on the training split.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Exactly the ``STAGE1_FEATURES`` columns. Unscaled — the scaler is
            fitted here.
        y_train : pandas.Series
            Binary labels, indexed identically to ``X_train``.

        Returns
        -------
        CalibratedStage1MLP
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the features are not exactly ``STAGE1_FEATURES``, if ``X`` and
            ``y`` differ in length or index, or if ``y`` is not binary 0/1.
        CalibrationLeakageError
            If the calibrator is prefit and ``strict_oof`` is True.
        """
        cfg = self.config
        X = select_features(X_train)

        y_train = check_labels(X, y_train)

        # 0. Scaler — training split only.
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        cv = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )

        # 1. Hyperparameter tuning.
        if cfg.verbose:
            print("🧠 Tuning MLP — Optuna (ROC-AUC)")
        self.best_params, self.best_cv_roc_auc, self.tuning_trials = (
            tune_stage1_mlp_auc(
                X_scaled, y_train, cv, cfg.n_trials,
                cfg.estimator_fixed_kwargs, cfg.random_state,
                verbose=cfg.verbose,
            )
        )
        if cfg.verbose:
            print(f"   best CV ROC-AUC: {self.best_cv_roc_auc:.4f}")

        # 2. Final model on the full training split.
        self.model = Stage1MLPClassifier(
            **params_to_kwargs(self.best_params, **cfg.estimator_fixed_kwargs)
        )
        self.model.fit(X_scaled, y_train)

        # 3. Calibration (GLASS).
        if cfg.verbose:
            print("🔧 Calibrating")
        self.calibrated_model, self.calibration_method, self.calibration_metrics = (
            fit_stage1_calibration(
                self.model, X_scaled, y_train,
                cfg.calibration_method, cfg.cv_folds,
            )
        )

        # 4. Threshold from out-of-fold calibrated probabilities.
        if cfg.verbose:
            print("🎯 Optimizing threshold via cross-validation")
        self.proba_train_oof, self.oof_provenance_ = oof_probabilities(
            self.calibrated_model, X_scaled, y_train, X_train.index,
            cv_folds=cfg.cv_folds, random_state=cfg.random_state,
            n_jobs=cfg.n_jobs, strict=cfg.strict_oof,
        )
        self.train_fold_id = fold_assignment(
            y_train, cv_folds=cfg.cv_folds, random_state=cfg.random_state,
            index=X_train.index,
        )
        self.y_train_ = y_train.copy()
        self.optimal_threshold, self.cv_f2, self.threshold_sweep = (
            optimize_threshold_cv(
                y_train, self.proba_train_oof,
                beta=cfg.threshold_beta, grid_spec=cfg.threshold_grid,
            )
        )
        if cfg.verbose:
            print(f"   Optimal threshold (CV): {self.optimal_threshold:.4f}")
            print(f"   CV F2-score:            {self.cv_f2:.4f}")
            print(f"   OOF provenance:         {self.oof_provenance_}")

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Calibrated ``P(y = 1)``.

        Parameters
        ----------
        X : pandas.DataFrame
            Exactly the ``STAGE1_FEATURES`` columns, unscaled.

        Returns
        -------
        pandas.Series
            Named ``stage1_proba``, indexed like ``X``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        self._check_fitted()
        X_sel = select_features(X)
        proba = self.calibrated_model.predict_proba(self.scaler.transform(X_sel))[:, 1]
        return pd.Series(proba, index=X.index, name="stage1_proba")

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Hard labels at the tuned threshold.

        Parameters
        ----------
        X : pandas.DataFrame
            Exactly the ``STAGE1_FEATURES`` columns, unscaled.
        threshold : float, optional
            Override the operating point. Defaults to ``optimal_threshold``.

        Returns
        -------
        pandas.Series of int8
            Named ``stage1_pred``, indexed like ``X``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        t = self.optimal_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype("int8").rename("stage1_pred")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def metrics_table(self, y_true, proba, label: str) -> pd.DataFrame:
        """
        Metrics at 0.50 and at the CV threshold.

        Parameters
        ----------
        y_true : array-like
            Binary labels.
        proba : array-like
            Predicted ``P(y = 1)``.
        label : str
            Row-label prefix, e.g. ``"test"``.

        Returns
        -------
        pandas.DataFrame
            Two rows, labelled ``"{label} @ 0.50"`` and
            ``"{label} @ {optimal_threshold:.2f}"``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.

        Notes
        -----
        The in-stage grid stops at 0.49, so the two thresholds never share a
        two-decimal row label.
        """
        self._check_fitted()
        return _metrics_table(
            y_true, proba, label,
            thresholds=(0.5, self.optimal_threshold),
            calibration_method=self.calibration_method,
        )

    def describe(self) -> Dict:
        """
        One-line summary of the fitted stage, for the notebook and write-up.

        Returns
        -------
        dict
            Architecture (``layers``, ``trainable_params``, ``best_epoch``), the
            winning Optuna parameters, ``best_cv_roc_auc``,
            ``calibration_method``, ``optimal_threshold``, ``cv_f2`` and
            ``oof_provenance``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        self._check_fitted()
        sizes = [len(STAGE1_FEATURES), *self.model.hidden_layer_sizes, 1]
        return {
            "layers": " → ".join(map(str, sizes)),
            "trainable_params": self.model.n_trainable_params_,
            **self.best_params,
            "best_epoch": self.model.best_epoch_,
            "best_cv_roc_auc": round(self.best_cv_roc_auc, 4),
            "calibration_method": self.calibration_method,
            "optimal_threshold": self.optimal_threshold,
            "cv_f2": round(self.cv_f2, 4),
            "oof_provenance": self.oof_provenance_,
        }

    # ------------------------------------------------------------------
    # Stage 4 handoff
    # ------------------------------------------------------------------
    def to_stage_output(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        threshold: Optional[float] = None,
        threshold_source: Optional[str] = None,
        metrics_test: Optional[Dict] = None,
    ) -> StageOutput:
        """
        Emit the shared cascade contract for Stage 4.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test split, exactly ``STAGE1_FEATURES``. Scored here, never fitted.
        y_test : pandas.Series
            Test labels, indexed like ``X_test``.
        threshold : float, optional
            Operating point to record. Defaults to ``optimal_threshold``.
        threshold_source : str, optional
            A key of ``THRESHOLD_SOURCES``. Required when ``threshold`` differs
            from ``optimal_threshold``. The contract will not let an
            overridden threshold be passed off as the in-stage one.
        metrics_test : dict, optional
            Held-out metrics to carry along.

        Returns
        -------
        StageOutput

        Notes
        -----
        This is the only method on the stage that accepts test data, and it
        uses it for scoring alone.
        """
        self._check_fitted()
        t = self.optimal_threshold if threshold is None else float(threshold)
        overridden = abs(t - self.optimal_threshold) > 1e-12
        if threshold_source is None:
            if overridden:
                raise ValueError(
                    f"threshold={t} differs from optimal_threshold="
                    f"{self.optimal_threshold}; say how it was chosen via "
                    f"threshold_source (one of {sorted(THRESHOLD_SOURCES)})."
                )
            threshold_source = self.THRESHOLD_SOURCE

        y_test = check_labels(select_features(X_test), y_test)

        return StageOutput(
            stage="stage1",
            arm="blackbox",
            model="calibrated_mlp",
            feature_names=list(self.feature_names_),
            train_proba_oof=self.proba_train_oof,
            test_proba=self.predict_proba(X_test),
            y_train=self.y_train_,
            y_test=y_test,
            train_fold_id=self.train_fold_id,
            threshold=t,
            threshold_source=threshold_source,
            calibration_method=self.calibration_method,
            calibration_requested=self.config.calibration_method,
            oof_provenance=self.oof_provenance_ or "unknown",
            best_params=dict(self.best_params),
            best_cv_score=self.best_cv_roc_auc,
            cv_f2=self.cv_f2,
            threshold_sweep=self.threshold_sweep,
            calibration_metrics=dict(self.calibration_metrics),
            metrics_test=dict(metrics_test or {}),
            config=self.config.to_dict(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if ``fit`` has not run."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted else "not fitted"
        auc = f", cv_auc={self.best_cv_roc_auc:.4f}" if self.fitted else ""
        return (
            f"CalibratedStage1MLP({state}, "
            f"calibration='{self.calibration_method}', "
            f"threshold={self.optimal_threshold:.3f}{auc})"
        )
