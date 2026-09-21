"""
blackbox_pipeline.models.xgb.tuning
====================================
Optuna hyperparameter search, mirroring the EBM's tuning protocol.

Mirrored exactly
----------------
* objective        F-beta(beta=2) on the 0.5 decision, i.e. the EBM's
                   ``fbeta_score(y_val, model.predict(X_val), beta=2)``
* sampler          ``TPESampler(seed=random_state)``
* pruner           ``MedianPruner(n_startup_trials=20, n_warmup_steps=2)``
* reporting        running mean of completed folds reported per fold, pruned
                   between folds
* failure policy   an exception inside a trial returns ``-inf`` rather than
                   aborting the study (counted in ``n_failed``; the search
                   raises if no trial scores finitely)
* budget           150 trials
* folds            5-fold stratified, seed 42 — here supplied as a ``FoldPlan``
                   so the same partition is reused for OOF, calibration and
                   threshold selection
* weights          balanced sample weights computed once and sliced per fold

Not mirrored
------------
The search space itself, which has no EBM counterpart beyond
``learning_rate``. It is declared in ``config.XGBSearchSpace``, recorded in the
artifact, and discussed there.

Nothing in this module sees the test split.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import fbeta_score

from .config import XGBSearchSpace, XGBStage3Config
from .estimator import XGBFactory
from .folds import FoldPlan
from .weights import BalancedWeights

optuna.logging.set_verbosity(optuna.logging.WARNING)

__all__ = ["TuningResult", "XGBTuner"]


@dataclass
class TuningResult:
    """Everything the search learned, for the artifact and the write-up."""

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
    top_trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"best CV F2 : {self.best_value:.6f}",
            f"trials     : {self.n_complete} completed, {self.n_pruned} pruned, "
            f"{self.n_failed} failed of {self.n_trials_requested}",
            f"runtime    : {self.runtime_s:.1f}s",
            "best params:",
        ]
        for k, v in self.best_params.items():
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"   {k:>18s} = {fmt}")
        return "\n".join(lines)


class XGBTuner:
    """
    Recall-biased Optuna search over ``XGBSearchSpace``.

    Parameters
    ----------
    config   Stage 3 configuration (budget, seeds, objective, space).
    factory  Builds the classifiers.
    weights  Balanced sample weights over the FULL training split.
    folds    The canonical partition; the same one later used for OOF.
    """

    def __init__(
        self,
        config: XGBStage3Config,
        factory: XGBFactory,
        weights: BalancedWeights,
        folds: FoldPlan,
    ):
        self.config = config
        self.factory = factory
        self.weights = weights
        self.folds = folds

    # ------------------------------------------------------------------
    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one point from the declared space."""
        s: XGBSearchSpace = self.config.search_space
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", s.n_estimators_low, s.n_estimators_high,
                step=s.n_estimators_step),
            "learning_rate": trial.suggest_float(
                "learning_rate", s.learning_rate_low, s.learning_rate_high,
                log=s.learning_rate_log),
            "max_depth": trial.suggest_int(
                "max_depth", s.max_depth_low, s.max_depth_high),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", s.min_child_weight_low,
                s.min_child_weight_high, log=s.min_child_weight_log),
            "subsample": trial.suggest_float(
                "subsample", s.subsample_low, s.subsample_high),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", s.colsample_bytree_low,
                s.colsample_bytree_high),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", s.reg_lambda_low, s.reg_lambda_high,
                log=s.reg_lambda_log),
        }

    # ------------------------------------------------------------------
    def _fold_score(self, params, X, y, train_idx, val_idx) -> float:
        cfg = self.config
        model = self.factory.fit(
            params, X.iloc[train_idx], y.iloc[train_idx],
            sample_weight=self.weights.for_rows(train_idx),
        )
        pred = XGBFactory.hard_predict(
            model, X.iloc[val_idx], cfg.tuning_decision_threshold
        )
        return float(
            fbeta_score(y.iloc[val_idx], pred, beta=cfg.beta, zero_division=0)
        )

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = self.suggest(trial)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            try:
                fold_scores.append(
                    self._fold_score(params, X, y, train_idx, val_idx)
                )
            except optuna.TrialPruned:
                raise
            except Exception:
                # Mirrors the EBM: a broken configuration scores -inf and the
                # study continues rather than dying mid-search.
                return float("-inf")

            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    # ------------------------------------------------------------------
    def tune(self, X: pd.DataFrame, y: pd.Series) -> TuningResult:
        cfg = self.config
        self.folds.assert_matches(len(X), "tuning")
        self.weights.assert_matches(len(X), "tuning")

        if cfg.verbose:
            print("\n" + "=" * 78)
            print(f"  XGBOOST STAGE 3 TUNING — RECALL-BIASED "
                  f"(F{cfg.beta:g} objective)")
            print("=" * 78)
            print(f"  dataset      : {X.shape[0]:,} rows, {X.shape[1]} features")
            print(f"  class balance: {y.value_counts().to_dict()}")
            print(f"  {self.weights.describe()}")
            print(f"  objective    : F{cfg.beta:g} at the "
                  f"{cfg.tuning_decision_threshold} cut "
                  f"(same as the EBM's model.predict)")
            print(f"  folds        : {self.folds.n_splits}-fold "
                  f"{self._fold_kind()}, "
                  f"seed {self.folds.random_state}")
            print(f"  budget       : {cfg.n_trials} trials, TPE(seed="
                  f"{cfg.random_state}), MedianPruner("
                  f"{cfg.pruner_startup_trials}/{cfg.pruner_warmup_steps})")
            print("  search space :")
            for k, v in cfg.search_space.describe().items():
                print(f"     {k:>18s}  {v}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=cfg.random_state),
            pruner=MedianPruner(
                n_startup_trials=cfg.pruner_startup_trials,
                n_warmup_steps=cfg.pruner_warmup_steps,
            ),
            study_name=cfg.study_name,
        )

        t0 = time.perf_counter()
        study.optimize(
            lambda t: self._objective(t, X, y),
            n_trials=cfg.n_trials,
            show_progress_bar=cfg.show_progress_bar and cfg.verbose,
        )
        runtime_s = time.perf_counter() - t0

        # A trial that raised returns -inf, which Optuna records as COMPLETE.
        # Count those as failures, and refuse to "select" a config if no
        # trial produced a finite score.
        TS = optuna.trial.TrialState
        finite = [t for t in study.trials
                  if t.state == TS.COMPLETE and t.value is not None
                  and np.isfinite(t.value)]
        if not finite:
            raise RuntimeError(
                f"No Optuna trial produced a finite score in {cfg.n_trials} "
                "trials — every configuration failed. Check the data and "
                "search space."
            )
        n_errored = sum(t.state == TS.COMPLETE for t in study.trials) - len(finite)

        result = TuningResult(
            best_params=XGBFactory.normalize_params(study.best_params),
            best_value=float(study.best_value),
            objective=f"F{cfg.beta:g} @ {cfg.tuning_decision_threshold} cut, "
                      f"{self.folds.n_splits}-fold {self._fold_kind()} CV",
            n_trials_requested=cfg.n_trials,
            n_complete=len(finite),
            n_pruned=sum(t.state == TS.PRUNED for t in study.trials),
            n_failed=sum(t.state == TS.FAIL for t in study.trials) + n_errored,
            runtime_s=float(runtime_s),
            random_state=cfg.random_state,
            study_name=cfg.study_name,
            search_space=cfg.search_space.describe(),
            fold_plan={k: v for k, v in self.folds.to_dict().items()
                       if k != "fold_id"},
            top_trials=self._top_trials(study),
        )

        if cfg.verbose:
            print("\n" + "-" * 78)
            print(result.describe())
            print("-" * 78)
        return result

    # ------------------------------------------------------------------
    def _fold_kind(self) -> str:
        return "stratified" if self.folds.stratified else "unstratified"

    @staticmethod
    def _top_trials(study: optuna.Study, n: int = 5) -> list[dict]:
        done = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and np.isfinite(t.value)
        ]
        done.sort(key=lambda t: t.value, reverse=True)
        return [
            {"number": t.number, "value": float(t.value), "params": dict(t.params)}
            for t in done[:n]
        ]