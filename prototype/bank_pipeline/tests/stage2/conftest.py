"""
GLASS Router helpers for the Stage 2 suite.

legacy_config(**overrides)
    The GLASS notebook's active "legacy" profile (glass_cascade.ipynb, Cell 13).
selected_rule(rule_id, segment, pass_name, precision)
    A hand-built ``SelectedRule``.
router_with_rules(pass1, pass2, base_rate)
    A ``GlassRouterPipeline`` in the fitted state with hand-built rules, so
    routing is tested without rule search or the ILP.
run_crossfit(split, ...)
    ``crossfit_stage2`` with the real per-fold feature engineering and RF, a
    recording stub in place of the GLASS Router (a real fit takes ~30 s per
    fold), and every fit / score recorded. The stub routes each row by its
    index label, so the assembled OOF block can be checked row by row.

The real router fit is exercised by the ``slow`` test in
test_glass_router_fit.py; rule selection itself by
test_ilp_rule_selector_union.py.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
from glass_pipeline.glass_router.core.config import GlassRouterConfig
from glass_pipeline.glass_router.core.rule import ABSTAIN, NOT_SUBSCRIBE, SUBSCRIBE, SelectedRule
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from glass_pipeline.glass_router.rf.rf_training import _build_pipe
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from shared.stage2.feature_engineering import engineer_features
from tests.support.recorders import Call, CallLog, row_keys

# ----------------------------------------------------------------------
# configuration and hand-built routers
# ----------------------------------------------------------------------
_LEGACY = dict(
    mode="strict", max_complexity=3, diversity_weight=0.35, enable_novelty_constraints=True,
    max_leakage_rate_depth2=0.90, max_leakage_fraction_depth2=0.90,
    max_jaccard_overlap=0.5, max_high_overlap_rules=20,
    max_feature_reuse_pass1=None, max_feature_reuse_pass2=None,
    min_support_pass1=120, min_support_pass2=25,
    min_pass1_rules=2, max_pass1_rules=8,
    min_precision_not_subscribe=0.93, max_precision_not_subscribe=1.00,
    max_subscriber_leakage_rate=0.05, max_subscriber_leakage_absolute=800,
    max_union_leakage_rate_pass1=0.14,
    min_pass2_rules=2, max_pass2_rules=8,
    min_precision_subscribe=0.20, max_precision_subscribe=1.00,
    min_recall_subscribe=0.01, max_recall_subscribe=0.35,
    min_novelty_ratio_pass1=0.20, min_novelty_ratio_pass2=0.25,
    pass2_population="full",
)


def legacy_config(**overrides) -> GlassRouterConfig:
    return GlassRouterConfig(**{**_LEGACY, **overrides})


def selected_rule(rule_id, segment, pass_name: str, precision: Optional[float]) -> SelectedRule:
    return SelectedRule(
        rule_id=rule_id, segment=frozenset(segment),
        predicted_class=NOT_SUBSCRIBE if pass_name == "pass1" else SUBSCRIBE,
        complexity=len(segment), precision=precision, recall=0.0, coverage=0.0,
        pass_assignment=pass_name,
    )


def new_router(config: Optional[GlassRouterConfig] = None) -> GlassRouterPipeline:
    with contextlib.redirect_stdout(io.StringIO()):
        return GlassRouterPipeline(config or legacy_config(), rule_logger=RuleLogger(enabled=False))


def router_with_rules(pass1, pass2, base_rate: float = 0.2) -> GlassRouterPipeline:
    router = new_router()
    router.pass1_rules, router.pass2_rules = list(pass1), list(pass2)
    router.training_base_rate = base_rate
    router.is_fitted = True
    return router


# ----------------------------------------------------------------------
# cross-fit with recorders
# ----------------------------------------------------------------------
#: Small but valid ``rf_result.params`` (the keys crossfit_stage2 reads).
TINY_RF = {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 2,
           "max_features": "sqrt", "minority_weight": 3.0}

#: Stub routing by index label: label % 3 -> (prediction, confidence, decision).
STUB_ROUTES = {0: (NOT_SUBSCRIBE, 0.9, "pass1"), 1: (SUBSCRIBE, 0.6, "pass2"),
               2: (ABSTAIN, 0.0, "uncertain")}


def stub_route(index):
    codes = np.asarray(index, dtype=np.int64) % 3
    pred = np.array([STUB_ROUTES[c][0] for c in codes], dtype=int)
    conf = np.array([STUB_ROUTES[c][1] for c in codes], dtype=float)
    dec = np.array([STUB_ROUTES[c][2] for c in codes], dtype=object)
    return pred, conf, dec


def stub_proba(index, base_rate: float) -> np.ndarray:
    """The production predict_proba formula applied to the stub routes."""
    _, conf, dec = stub_route(index)
    p1 = np.where(dec == "pass1", base_rate * (1 - conf),
                  np.where(dec == "pass2", base_rate + (1 - base_rate) * conf, base_rate))
    return np.column_stack([1 - p1, p1])


@dataclass
class CrossfitRun:
    oof: "crossfit_mod.Stage2OOF"
    fe_log: CallLog
    rf_log: CallLog
    router_log: CallLog
    fe_inputs: List[dict] = field(default_factory=list)      # per fold
    pipes: list = field(default_factory=list)                # per fold
    routers: list = field(default_factory=list)              # per fold


class _RecordingPipe:
    """The fold RF pipeline, with its fit and scoring rows recorded."""

    def __init__(self, pipe, log: CallLog):
        self.pipe, self.log = pipe, log

    @property
    def named_steps(self):
        return self.pipe.named_steps

    def fit(self, X, y, **kwargs):
        if not X.index.equals(y.index):
            raise AssertionError("RF fitted on labels that are not aligned to X")
        self.fit_id = self.log.new_fit_id()
        self.log.append(Call("fit", self.fit_id, row_keys(X)))
        self.sample_weight = kwargs.get("clf__sample_weight")
        self.pipe.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        self.log.append(Call("predict_proba", self.fit_id, row_keys(X)))
        return self.pipe.predict_proba(X)


def _stub_router_class(log: CallLog, instances: list):
    class StubRouter:
        """Stands in for GlassRouterPipeline inside crossfit_stage2."""

        def __init__(self, config, rf_model=None, segment_builder=None, rule_logger=None):
            self.config, self.rf_model, self.rule_logger = config, rf_model, rule_logger
            self.pass1_rules, self.pass2_rules = ["p1"], ["p2a", "p2b"]
            self.seen = []
            instances.append(self)

        def fit(self, X, y, X_val=None, y_val=None):
            if not X.index.equals(y.index):
                raise AssertionError("router fitted on labels that are not aligned to X")
            self.fit_id = log.new_fit_id()
            log.append(Call("fit", self.fit_id, row_keys(X)))
            self.fit_X, self.training_base_rate = X.copy(), float(y.mean())
            return self

        def predict(self, X):
            log.append(Call("predict", self.fit_id, row_keys(X)))
            self.seen.append(X.copy())
            return stub_route(X.index)

        def predict_proba(self, X):
            log.append(Call("predict_proba", self.fit_id, row_keys(X)))
            return stub_proba(X.index, self.training_base_rate)
    return StubRouter


def run_crossfit(split: dict, *, y_train: Optional[pd.Series] = None, folds=None,
                 glass_rf_model: str = "fold_rf", config: Optional[GlassRouterConfig] = None,
                 n_splits: int = 5) -> CrossfitRun:
    """One recorded ``crossfit_stage2`` run on ``split["X_train"]`` (raw features)."""
    y = split["y_train"] if y_train is None else y_train
    run = CrossfitRun(None, CallLog(), CallLog(), CallLog())

    def recording_fe(X_tr, X_va, y_tr):
        if not X_tr.index.equals(y_tr.index):
            raise AssertionError("feature engineering fitted on labels not aligned to X")
        fid = run.fe_log.new_fit_id()
        run.fe_log.append(Call("fit_transform", fid, row_keys(X_tr)))
        run.fe_log.append(Call("transform", fid, row_keys(X_va)))
        Xtr_eng, Xva_eng = engineer_features(X_tr, X_va, y_tr)
        run.fe_inputs.append({"columns": list(X_tr.columns), "train": Xtr_eng, "val": Xva_eng})
        return Xtr_eng, Xva_eng

    def recording_build_pipe(params, random_state, n_jobs=-1):
        pipe = _RecordingPipe(_build_pipe(params, random_state, n_jobs=1), run.rf_log)
        run.pipes.append(pipe)
        return pipe

    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(crossfit_mod, "engineer_features", recording_fe)
        mp.setattr(crossfit_mod, "_build_pipe", recording_build_pipe)
        mp.setattr(crossfit_mod, "GlassRouterPipeline", _stub_router_class(run.router_log, run.routers))
        run.oof = crossfit_mod.crossfit_stage2(
            split["X_train"], y, config or legacy_config(), TINY_RF, folds=folds,
            n_splits=n_splits, random_state=42, glass_rf_model=glass_rf_model,
        )
    return run
