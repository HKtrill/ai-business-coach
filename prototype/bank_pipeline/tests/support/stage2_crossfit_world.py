"""
PR 38 runner fixtures: a synthetic train side, a loose test contract, small RF
settings and a GLASS Router whose rule search and ILP are scripted.

wired_glass_class(script=None)
    A ``GlassRouterPipeline`` subclass with the real ``fit`` (inner folds,
    remainder, nested rankers, selection reports) and scripted rule search:
    Pass 1 selects ``behav_cold``, Pass 2 ``jed_hot``; each selection writes a
    set-level report with its union counts. ``script`` maps a selection call
    number (1-based, per instance) to "infeasible" / "no_solution" / "unproven"
    / "greedy". ``predict`` refuses rows the fit saw only as held out is not
    knowable here, so instances count their ``predict`` calls instead.
wired(monkeypatch, cls)
    Install ``cls`` wherever the runner constructs the GLASS Router.
make_settings(...)
    ``ArmSettings`` (role "test") under ``LOOSE`` with the frozen protocol, so
    the required selection keys are the PR 38 ones (5 inner / 5 eligibility).
synthetic_train_side()
    ``runner.make_train_side`` on a synthetic bank frame (cached).
"""

from __future__ import annotations

import dataclasses
import functools
from types import SimpleNamespace

import numpy as np

import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
import stage2_crossfit.runner as runner_mod
import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import RFRouterConfig
from blackbox_pipeline.models.rf_router.contract import rf_constraints_from_contract_v2
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.core.rule import NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import SelectorInfeasibleError
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from tests.stage2.conftest import selected_rule
from tests.stage2.test_stage2_operating_contract_v2 import _v2
from tests.support.synthetic_rf_router import SMALL_PARAMS
from tests.support.synthetic_bank import make_bank_frame, split_frame

P1, P2 = "Pass 1 (NOT_SUBSCRIBE)", "Pass 2 (SUBSCRIBE)"
PASS1_SEGMENT, PASS2_SEGMENT = [("behav_cold", 1)], [("jed_hot", 1)]

#: Feasible for the scripted rules and the small RF on the synthetic frame.
LOOSE = _v2(min_pass1_npv=0.85, max_pass1_leakage_rate=0.5, min_pass2_precision=0.20,
            min_pass1_support=20, min_pass2_support=10)

TEST_RANKER = {"glass_rf_model": "fold_rf",
               "rf_params": {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 2,
                             "max_features": "sqrt", "minority_weight": 3.0,
                             "class_weight": "balanced"}}


class _Gen:
    def __init__(self, *args, **kwargs):
        pass

    def generate_candidates(self, X, y):
        return [SimpleNamespace(predicted_class=NOT_SUBSCRIBE),
                SimpleNamespace(predicted_class=SUBSCRIBE)]


class _Eval:
    def evaluate_candidates(self, candidates, X, y):
        return [SimpleNamespace(recall=0.5, coverage=0.5) for _ in candidates]


class _Sel:
    def __init__(self, owner):
        self.owner = owner
        self.selection_report_ = {}

    def _pass(self, scope, X, y):
        owner = self.owner
        owner.n_selections += 1
        action = owner.script.get(owner.n_selections)
        key = P1 if scope == "pass1" else P2
        rule = (selected_rule(1, PASS1_SEGMENT, "pass1", 0.9) if scope == "pass1"
                else selected_rule(2, PASS2_SEGMENT, "pass2", 0.3))
        y = np.asarray(y)
        if action in ("infeasible", "no_solution"):
            self.selection_report_[key] = {"source": "infeasible",
                                           "stage": "quality gates" if action == "infeasible"
                                           else None}
            raise SelectorInfeasibleError(f"scripted {action} ({scope})",
                                          proven=action == "infeasible")
        mask = GlassRouterPipeline._rules_mask([rule], X)
        good = (y == 0) if scope == "pass1" else (y == 1)
        rep = {"source": "ilp", "lexicographic_guarantee": "proven",
               "selected_rule_ids": [rule.rule_id], "union_rows": int(mask.sum()),
               "union_good": int(good[mask].sum())}
        if action == "unproven":
            rep["lexicographic_guarantee"] = "unproven"
        elif action == "greedy":
            rep.update(source="greedy_fallback", lexicographic_guarantee="not applicable")
        self.selection_report_[key] = rep
        return rule

    def select_rules(self, evaluated_rules, y_val, X_val, segment_builder,
                     passes=("pass1", "pass2")):
        self.selection_report_ = {}
        out = {"pass1_rules": [], "pass2_rules": []}
        for scope in passes:
            out[f"{scope}_rules"] = [self._pass(scope, X_val, y_val)]
        return out


def wired_glass_class(script=None):
    class WiredGlass(GlassRouterPipeline):
        def __init__(self, *args, **kwargs):
            self.script = dict(script or {})
            self.n_selections = 0
            self.n_predict = 0
            super().__init__(*args, **kwargs)

        def _new_generator(self, recall_scale=1.0, rf_model=None):
            return _Gen()

        def _new_evaluator(self):
            return _Eval()

        def _new_selector(self):
            return _Sel(self)

        def predict(self, X):
            self.n_predict += 1
            return super().predict(X)

    return WiredGlass


def wired(monkeypatch, cls):
    monkeypatch.setattr(crossfit_mod, "GlassRouterPipeline", cls)
    monkeypatch.setattr(runner_mod, "GlassRouterPipeline", cls)
    return cls


def rf_test_config(contract=LOOSE, **overrides) -> RFRouterConfig:
    kw = dict(mode="two_pass", constraints=rf_constraints_from_contract_v2(contract),
              rf_params=SMALL_PARAMS, n_oof_folds=5, eligibility="nested_t1",
              eligibility_folds=2, random_state=42, remainder_min_size=60, verbose=False)
    kw.update(overrides)
    return RFRouterConfig(**kw)


def make_settings(contract=LOOSE, *, glass_config=None, rf_factory=None, role="test"):
    glass = glass_config or dataclasses.replace(glass_profile.glass_primary_config(),
                                                remainder_min_size=50)
    return runner_mod.ArmSettings(role, glass, TEST_RANKER,
                                  rf_factory or (lambda: rf_test_config(contract)),
                                  contract, ep.load_frozen_protocol(), {"identities": {}})


@functools.lru_cache(maxsize=None)
def synthetic_train_side():
    gs = split_frame(make_bank_frame(n=2000, seed=3, positive_rate=0.15), test_size=0.2)
    return runner_mod.make_train_side(gs["X_train"], gs["y_train"].astype(int), gs["X_test"])
