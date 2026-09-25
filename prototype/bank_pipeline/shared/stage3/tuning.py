"""
shared.stage3.tuning
====================
Recall-biased Optuna search (TPE + MedianPruner) over an estimator's declared
space, scored with F-beta at the tuning decision threshold across the tuning
folds. Optuna is imported only here (and configured to WARNING verbosity on
import, as before the split).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import fbeta_score

from .data import FoldFrames


optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
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
    feature_fit_scope: str = "per_fold"
    tuned: bool = True
    top_trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"best CV F2 : {self.best_value:.6f}",
            f"trials     : {self.n_complete} completed, {self.n_pruned} pruned, "
            f"{self.n_failed} failed of {self.n_trials_requested}",
            f"runtime    : {self.runtime_s:.1f}s",
            f"features   : fitted {self.feature_fit_scope}",
            "best params:",
        ]
        for k, v in self.best_params.items():
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"   {k:>20s} = {fmt}")
        return "\n".join(lines)


class Stage3Tuner:
    """Recall-biased Optuna search over the estimator's declared space."""

    def __init__(self, config, estimator, folds: list[FoldFrames],
                 fold_plan_dict: dict, feature_fit_scope: str):
        self.config = config
        self.estimator = estimator
        self.folds = folds
        self.fold_plan_dict = fold_plan_dict
        self.feature_fit_scope = feature_fit_scope

    # ------------------------------------------------------------------
    def _fold_score(self, params, f: FoldFrames) -> float:
        cfg = self.config
        model = self.estimator.fit(params, f.X_tr, f.y_tr, sample_weight=f.w_tr)
        p = self.estimator.positive_proba(model, f.X_val)
        pred = (p >= cfg.tuning_decision_threshold).astype(int)
        return float(fbeta_score(np.asarray(f.y_val).astype(int), pred,
                                 beta=cfg.beta, zero_division=0))

    def _objective(self, trial: optuna.Trial) -> float:
        params = self.estimator.suggest(trial)
        scores: list[float] = []
        for fold_idx, f in enumerate(self.folds):
            try:
                scores.append(self._fold_score(params, f))
            except optuna.TrialPruned:
                raise
            except Exception:
                return float("-inf")
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    def tune(self) -> TuningResult:
        cfg = self.config
        n = sum(len(f.val_idx) for f in self.folds)
        if cfg.verbose:
            print("\n" + "=" * 78)
            print(f"  {cfg.model_family.upper()} STAGE 3 TUNING — RECALL-BIASED "
                  f"(F{cfg.beta:g} at the {cfg.tuning_decision_threshold} cut)")
            print("=" * 78)
            print(f"  rows         : {n:,}  |  features fitted "
                  f"{self.feature_fit_scope}")
            print(f"  folds        : {len(self.folds)}-fold, seed "
                  f"{cfg.random_state}")
            print(f"  budget       : {cfg.n_trials} trials, TPE(seed="
                  f"{cfg.random_state}), MedianPruner("
                  f"{cfg.pruner_startup_trials}/{cfg.pruner_warmup_steps})")
            print("  search space :")
            for k, v in self.estimator.describe_space().items():
                print(f"     {k:>20s}  {v}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=cfg.random_state),
            pruner=MedianPruner(n_startup_trials=cfg.pruner_startup_trials,
                                n_warmup_steps=cfg.pruner_warmup_steps),
            study_name=cfg.study_name,
        )
        t0 = time.perf_counter()
        study.optimize(self._objective, n_trials=cfg.n_trials,
                       show_progress_bar=cfg.show_progress_bar and cfg.verbose)
        runtime_s = time.perf_counter() - t0

        TS = optuna.trial.TrialState
        finite = [t for t in study.trials
                  if t.state == TS.COMPLETE and t.value is not None
                  and np.isfinite(t.value)]
        if not finite:
            raise RuntimeError(
                f"No Optuna trial produced a finite score in {cfg.n_trials} "
                "trials — every configuration failed."
            )
        n_errored = sum(t.state == TS.COMPLETE for t in study.trials) - len(finite)
        best = max(finite, key=lambda t: t.value)

        result = TuningResult(
            best_params=self.estimator.normalize_params(best.params),
            best_value=float(best.value),
            objective=f"F{cfg.beta:g} @ {cfg.tuning_decision_threshold} cut, "
                      f"{len(self.folds)}-fold CV",
            n_trials_requested=cfg.n_trials,
            n_complete=len(finite),
            n_pruned=sum(t.state == TS.PRUNED for t in study.trials),
            n_failed=sum(t.state == TS.FAIL for t in study.trials) + n_errored,
            runtime_s=float(runtime_s),
            random_state=cfg.random_state,
            study_name=cfg.study_name,
            search_space=self.estimator.describe_space(),
            fold_plan=self.fold_plan_dict,
            feature_fit_scope=self.feature_fit_scope,
            top_trials=[
                {"number": t.number, "value": float(t.value),
                 "params": dict(t.params)}
                for t in sorted(finite, key=lambda t: t.value, reverse=True)[:5]
            ],
        )
        if cfg.verbose:
            print("\n" + "-" * 78)
            print(result.describe())
            print("-" * 78)
        return result
