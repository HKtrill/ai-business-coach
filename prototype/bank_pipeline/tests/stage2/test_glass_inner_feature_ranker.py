"""
GLASS ``oof_remainder``: the inner-fold Pass 1 feature ranker (PR 37 Commit 3, F17).

Contract
1. Default (``inner_feature_ranker="outer_rf"``) is historical: every inner-fold
   Pass 1 search is ordered by the pipeline's ``rf_model``. No factory is
   called, and a ``"full"`` fit report is unchanged.
2. ``"fold_nested_rf"``: each inner fold gets its own ranker, fitted on exactly
   that fold's training rows. It is never trained on the rows that fold's rules
   route (router-level isolation, conditional on the engineered features; PR 37
   F17 L1, PR 38 gate G13), and it is neither the outer ``rf_model`` nor another
   fold's ranker. The full-population Pass 1 and the Pass 2 search keep
   ``rf_model``. Provenance is recorded.
3. No silent fallback: a missing factory, a reused or invalid ranker, or an
   infeasible inner-fold selection raises. A failed ``oof_remainder`` fit never
   becomes a ``"full"`` fit.
4. ``make_rf_feature_ranker_factory`` fits the cross-fit's fold-RF recipe on
   exactly the rows given, deterministically. ``crossfit_stage2`` supplies it
   only on the opt-in path, and only with ``glass_rf_model="fold_rf"``.
5. The GLASS inner fold plan is the RF Router's: the same stratified, shuffled,
   seeded partition of the same rows.

Rule search, evaluation and the ILP are recording stubs, as in
test_glass_router_pass2_remainder.py; the fold loop, ranker handling and
remainder logic are production code.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
from blackbox_pipeline.models.rf_router import RFRouter, make_folds
from glass_pipeline.glass_router.core.rule import NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import SelectorInfeasibleError
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline, crossfit_stage2
from glass_pipeline.glass_router.pipeline.glass_router_pipeline import feature_order_fingerprint
from glass_pipeline.glass_router.rf.feature_ranker import make_rf_feature_ranker_factory
from glass_pipeline.glass_router.rf.rf_training import _build_pipe
from shared.stage_io import label_fingerprint
from tests.stage2.conftest import TINY_RF, legacy_config, new_router, selected_rule
from tests.support.synthetic_rf_router import make_data, router_config
from tests.support.synthetic_bank import irregular_index, make_bank_frame, split_frame

N, INNER, SEED, FULL_BUCKET = 600, 3, 42, 3


def _data(seed=0):
    rng = np.random.default_rng(seed)
    index = irregular_index(N, seed=seed)
    y = pd.Series((rng.random(N) < 0.25).astype(int), index=index)
    X = pd.DataFrame({"bucket": rng.integers(0, 4, N), "flag": rng.integers(0, 2, N)}, index=index)
    return X, y


def _inner_folds(y):
    skf = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=SEED)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), np.asarray(y))]


class _Ranker:
    """A fitted-ranker stand-in: importances plus the rows it was fitted on."""

    def __init__(self, X):
        self.rows = frozenset(X.index)
        self.feature_importances_ = np.linspace(1.0, 0.5, X.shape[1])
        self.random_state = SEED


def _stubbed(ranker="fold_nested_rf", population="oof_remainder", factory="recording"):
    router = new_router(legacy_config(pass2_population=population, pass1_oof_folds=INNER,
                                      random_state=SEED, remainder_min_size=50,
                                      inner_feature_ranker=ranker))
    rec = SimpleNamespace(pass1=[], generators=[], factory_rows=[])
    router.rf_model = SimpleNamespace(feature_importances_=np.array([0.6, 0.4]), name="outer")

    def recording_factory(X, y):
        assert X.index.equals(y.index)
        rec.factory_rows.append(frozenset(X.index))
        return _Ranker(X)

    if factory == "recording":
        router.feature_ranker_factory = recording_factory
    elif callable(factory):
        router.feature_ranker_factory = factory

    def fit_pass1_only(X, y, generator, evaluator, selector):
        rec.pass1.append((frozenset(X.index), generator))
        fid = len(rec.pass1)
        value = FULL_BUCKET if fid == 1 else fid - 2                 # call 1: full; then folds
        return [selected_rule(fid, [("bucket", value)], "pass1", 0.95)]

    class Generator:
        def __init__(self, recall_scale=1.0, rf_model="own"):
            self.recall_scale = recall_scale
            self.rf_model = router.rf_model if rf_model == "own" else rf_model
            rec.generators.append(self)

        def generate_candidates(self, X, y):
            return [SimpleNamespace(predicted_class=SUBSCRIBE, segment=frozenset({("flag", 1)}))]

    class Evaluator:
        def evaluate_candidates(self, candidates, X, y):
            return [SimpleNamespace(recall=0.5, coverage=0.5) for _ in candidates]

    class Selector:
        def select_rules(self, evaluated_rules, y_val, X_val, segment_builder, passes):
            return {"pass1_rules": [], "pass2_rules": [selected_rule(99, [("flag", 1)], "pass2", 0.5)]}

    router._fit_pass1_only = fit_pass1_only
    router._new_generator = Generator
    router._new_evaluator = Evaluator
    router._new_selector = Selector
    return router, rec


@pytest.fixture(scope="module")
def nested():
    X, y = _data()
    router, rec = _stubbed()
    router.fit(X, y)
    return SimpleNamespace(X=X, y=y, router=router, rec=rec)


# ----------------------------------------------------------------------
# 2. the opt-in nested ranker
# ----------------------------------------------------------------------
def test_each_inner_ranker_is_fitted_on_its_folds_training_rows_only(nested):
    X, rec = nested.X, nested.rec
    folds = _inner_folds(nested.y)
    assert len(rec.factory_rows) == INNER
    for (tr, va), rows in zip(folds, rec.factory_rows):
        assert rows == frozenset(X.index[tr])
        assert not rows & frozenset(X.index[va])                    # never the rows it routes


def test_each_inner_search_is_ordered_by_its_own_ranker_and_the_rest_by_rf_model(nested):
    router, rec = nested.router, nested.rec
    inner = [gen for _, gen in rec.pass1[1:]]
    rankers = [g.rf_model for g in inner]
    assert len({id(r) for r in rankers}) == INNER
    assert all(isinstance(r, _Ranker) and r is not router.rf_model for r in rankers)
    assert [r.rows for r in rankers] == rec.factory_rows
    assert rec.pass1[0][0] == frozenset(nested.X.index)             # full-population Pass 1
    assert rec.generators[-1].rf_model is router.rf_model           # the Pass 2 search
    assert router.inner_rankers_ == rankers


def test_the_nested_ranker_provenance_is_recorded(nested):
    X, y, rep = nested.X, nested.y, nested.router.fit_report_
    assert rep["inner_feature_ranker"] == "fold_nested_rf"
    prov = rep["inner_ranker_provenance"]
    assert [p["fold"] for p in prov] == list(range(INNER))
    for p, (tr, _) in zip(prov, _inner_folds(y)):
        assert p["n_train"] == len(tr)
        assert p["train_label_fingerprint"] == label_fingerprint(X.index[tr], y.iloc[tr])
        assert p["estimator"] == "_Ranker" and p["random_state"] == SEED
        assert len(p["feature_order_fingerprint"]) == 16
    assert all(not hasattr(v, "fit") for p in prov for v in p.values())   # metadata only


# ----------------------------------------------------------------------
# 1. the historical default
# ----------------------------------------------------------------------
def test_the_default_orders_inner_searches_by_rf_model_and_calls_no_factory():
    X, y = _data()
    router, rec = _stubbed(ranker="outer_rf")
    router.fit(X, y)
    assert rec.factory_rows == []
    assert all(gen.rf_model is router.rf_model for _, gen in rec.pass1[1:])
    assert router.fit_report_["inner_feature_ranker"] == "outer_rf"
    assert "inner_ranker_provenance" not in router.fit_report_


def test_a_full_fit_report_is_unchanged_whatever_the_ranker_setting():
    X, y = _data()
    router, _ = _stubbed(population="full", factory=None)
    router.rule_generator = SimpleNamespace(generate_candidates=lambda X, y: [])
    router.rule_evaluator = SimpleNamespace(evaluate_candidates=lambda c, X, y: [])
    router.ilp_selector = SimpleNamespace(select_rules=lambda **k: {"pass1_rules": [], "pass2_rules": []})
    router.fit(X, y)
    assert router.fit_report_ == {"pass2_population": "full"}


def test_the_config_default_and_validation():
    assert legacy_config().inner_feature_ranker == "outer_rf"
    with pytest.raises(ValueError, match="inner_feature_ranker must be one of"):
        legacy_config(inner_feature_ranker="nested")


# ----------------------------------------------------------------------
# 3. no silent fallback
# ----------------------------------------------------------------------
def test_a_missing_factory_is_refused_before_any_fit():
    X, y = _data()
    router, rec = _stubbed(factory=None)
    with pytest.raises(ValueError, match="needs a feature_ranker_factory"):
        router.fit(X, y)
    assert rec.pass1 == [] and not router.is_fitted


@pytest.mark.parametrize("case", ["outer rf_model", "same ranker twice", "wrong importances"])
def test_a_reused_or_invalid_ranker_is_refused(case):
    X, y = _data()
    router, _ = _stubbed(factory=None)
    shared = _Ranker(X)
    router.feature_ranker_factory = {
        "outer rf_model": lambda X_, y_: router.rf_model,           # 2 importances: passes the shape check
        "same ranker twice": lambda X_, y_: shared,
        "wrong importances": lambda X_, y_: SimpleNamespace(feature_importances_=np.ones(5)),
    }[case]
    with pytest.raises(ValueError, match="inner fold"):
        router.fit(X, y)
    assert not router.is_fitted


def test_an_infeasible_inner_selection_propagates_and_never_becomes_a_full_fit():
    X, y = _data()
    router, rec = _stubbed(ranker="outer_rf")
    calls = []

    def failing(X_, y_, generator, evaluator, selector):
        calls.append(len(X_))
        if len(calls) == 2:                                           # first inner fold
            raise SelectorInfeasibleError("inner Pass 1 infeasible", proven=True)
        return [selected_rule(1, [("bucket", FULL_BUCKET)], "pass1", 0.95)]

    router._fit_pass1_only = failing
    with pytest.raises(SelectorInfeasibleError):
        router.fit(X, y)
    assert not router.is_fitted and router.fit_report_ == {}


# ----------------------------------------------------------------------
# 4. the factory and the cross-fit wiring
# ----------------------------------------------------------------------
def test_the_factory_fits_the_fold_rf_recipe_on_exactly_the_rows_given():
    Xb, yb = make_data(n=600, seed=3)
    fitted = []

    def recording_build_pipe(params, random_state, n_jobs=-1):
        pipe = _build_pipe(params, random_state, n_jobs=1)
        real_fit = pipe.fit

        def fit(X, y, **kw):
            fitted.append((frozenset(X.index), kw.get("clf__sample_weight")))
            return real_fit(X, y, **kw)
        pipe.fit = fit
        return pipe

    factory = make_rf_feature_ranker_factory(TINY_RF, SEED, build_pipe=recording_build_pipe)
    sub = Xb.index[:400]
    ranker = factory(Xb.loc[sub], yb.loc[sub])
    assert isinstance(ranker, RandomForestClassifier) and ranker.random_state == SEED
    (rows, w), = fitted
    assert rows == frozenset(sub)
    np.testing.assert_array_equal(w, np.where(yb.loc[sub] == 1, TINY_RF["minority_weight"], 1.0))
    again = make_rf_feature_ranker_factory(TINY_RF, SEED)(Xb.loc[sub], yb.loc[sub])
    np.testing.assert_array_equal(again.feature_importances_, ranker.feature_importances_)
    assert (feature_order_fingerprint(again, list(Xb.columns))
            == feature_order_fingerprint(ranker, list(Xb.columns)))
    with pytest.raises(ValueError, match="index-aligned"):
        factory(Xb.loc[sub], yb.loc[sub].reset_index(drop=True))


def _crossfit_router_kwargs(config, glass_rf_model="fold_rf"):
    split = split_frame(make_bank_frame(n=400))
    seen = []

    class Capture:
        def __init__(self, config, **kwargs):
            seen.append(kwargs)
            self.pass1_rules, self.pass2_rules = [], []

        def fit(self, X, y, X_val=None, y_val=None):
            self.training_base_rate = float(y.mean())
            return self

        def predict(self, X):
            n = len(X)
            return np.full(n, -1), np.zeros(n), np.array(["uncertain"] * n, dtype=object)

        def predict_proba(self, X):
            return np.tile([1 - self.training_base_rate, self.training_base_rate], (len(X), 1))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(crossfit_mod, "GlassRouterPipeline", Capture)
        out = crossfit_stage2(split["X_train"], split["y_train"], config, TINY_RF,
                              random_state=SEED, glass_rf_model=glass_rf_model)
    return seen, out


def test_the_crossfit_supplies_a_fold_nested_factory_only_on_the_opt_in_path():
    seen, out = _crossfit_router_kwargs(legacy_config(pass2_population="oof_remainder",
                                                      inner_feature_ranker="fold_nested_rf"))
    assert len(seen) == 5 and out.protocol["glass_inner_feature_ranker"] == "fold_nested_rf"
    for kw in seen:
        assert kw["feature_ranker_factory"].recipe["random_state"] == SEED
        assert kw["feature_ranker_factory"].recipe["rf_params"] == TINY_RF
        assert isinstance(kw["rf_model"], RandomForestClassifier)    # the fold RF, unchanged
    assert len({id(kw["rf_model"]) for kw in seen}) == 5
    seen, out = _crossfit_router_kwargs(legacy_config(pass2_population="oof_remainder"))
    assert all("feature_ranker_factory" not in kw for kw in seen)
    assert out.protocol["glass_inner_feature_ranker"] == "outer_rf"


def test_the_nested_ranker_requires_the_fold_rf():
    with pytest.raises(ValueError, match="requires glass_rf_model='fold_rf'"):
        _crossfit_router_kwargs(legacy_config(pass2_population="oof_remainder",
                                              inner_feature_ranker="fold_nested_rf"),
                                glass_rf_model="none")


def test_glass_pipeline_accepts_the_factory_by_keyword_only_as_an_addition():
    params = inspect.signature(GlassRouterPipeline.__init__).parameters
    assert list(params)[:5] == ["self", "config", "rf_model", "segment_builder", "rule_logger"]
    assert params["feature_ranker_factory"].default is None


# ----------------------------------------------------------------------
# 5. one inner fold plan for both arms
# ----------------------------------------------------------------------
def test_glass_inner_folds_are_the_rf_routers_inner_folds_for_the_same_rows(nested):
    glass = [frozenset(nested.X.index) - rows for rows, _ in nested.rec.pass1[1:]]  # held-out rows
    rf = [frozenset(nested.X.index[va]) for _, va in make_folds(nested.y, INNER, SEED, stratify=True)]
    assert glass == rf


def test_the_rf_router_scores_pass1_on_that_same_plan():
    X, y = make_data(n=900, seed=5)
    r = RFRouter(router_config(n_oof_folds=INNER, random_state=SEED)).fit(X, y, "fp")
    want = np.full(len(y), -1)
    for k, (_, va) in enumerate(make_folds(y, INNER, SEED, stratify=True)):
        want[va] = k
    np.testing.assert_array_equal(r.oof_pass1_fold_id_, want)
