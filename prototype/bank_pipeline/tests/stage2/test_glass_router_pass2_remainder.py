"""
GLASS Router Pass 2 population (GlassRouterPipeline, pass2_population=
"oof_remainder").

Contract: Pass 1 rules are selected on the full training population; each
training row is then routed by Pass 1 rules fitted WITHOUT it (inner folds);
Pass 2 is generated, evaluated and selected only on the rows that OOF Pass 1
left, with recall and coverage rescaled to the global denominators; a
remainder that is too small or single-class is refused.

Rule search, evaluation and the ILP are replaced by recording stubs (the ILP
has its own suite); the fold loop, the routing masks and the remainder logic
are the production code. The notebook's active "legacy" profile uses
pass2_population="full" (F17), so this mode is not on the default path yet.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from glass_pipeline.glass_router.core.rule import NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from tests.stage2.conftest import legacy_config, new_router, selected_rule
from tests.support.checks import assert_oof_provenance
from tests.support.recorders import Call, CallLog, row_keys
from tests.support.synthetic import irregular_index

N, INNER, SEED = 600, 3, 42
FULL_BUCKET = 3          # the full-population Pass 1 rule: bucket == 3
                         # inner fold k's Pass 1 rule:          bucket == k


class _Rules(list):
    fit_id: int


def _data(seed=0):
    rng = np.random.default_rng(seed)
    index = irregular_index(N, seed=seed)
    y = pd.Series((rng.random(N) < 0.25).astype(int), index=index)
    X = pd.DataFrame({"bucket": rng.integers(0, 4, N), "flag": rng.integers(0, 2, N)}, index=index)
    return X, y


def _inner_fold(y):
    fold = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), np.asarray(y))):
        fold[va] = k
    return fold


def _stubbed_router(**config):
    """A real pipeline whose rule search / evaluation / ILP are recording stubs."""
    cfg = dict(pass2_population="oof_remainder", pass1_oof_folds=INNER, random_state=SEED,
               remainder_min_size=50)
    router = new_router(legacy_config(**{**cfg, **config}))
    rec = SimpleNamespace(log=CallLog(), generators=[], evaluated=None, selected=None)

    def fit_pass1_only(X, y, generator, evaluator, selector):
        fid = rec.log.new_fit_id()
        rec.log.append(Call("fit", fid, row_keys(X)))
        value = FULL_BUCKET if fid == 1 else fid - 2           # call 1: full; then folds 0..k
        rules = _Rules([selected_rule(fid, [("bucket", value)], "pass1", 0.95)])
        rules.fit_id = fid
        return rules

    def rules_mask(rules, X):
        rec.log.append(Call("predict", rules.fit_id, row_keys(X)))
        return GlassRouterPipeline._rules_mask(rules, X)

    class Generator:
        def __init__(self, recall_scale=1.0):
            self.recall_scale, self.X = recall_scale, None
            rec.generators.append(self)

        def generate_candidates(self, X, y):
            self.X, self.y = X, y
            return [SimpleNamespace(predicted_class=SUBSCRIBE, segment=frozenset({("flag", 1)})),
                    SimpleNamespace(predicted_class=NOT_SUBSCRIBE, segment=frozenset({("flag", 0)}))]

    class Evaluator:
        def evaluate_candidates(self, candidates, X, y):
            hit = (X["flag"] == 1).to_numpy()
            pos = (np.asarray(y) == 1)
            local = ((hit & pos).sum() / pos.sum(), hit.mean())
            return [SimpleNamespace(candidate=c, X_index=X.index, local=local,
                                    recall=local[0], coverage=local[1]) for c in candidates]

    class Selector:
        def select_rules(self, evaluated_rules, y_val, X_val, segment_builder, passes):
            rec.evaluated, rec.selected = list(evaluated_rules), (X_val.index, passes)
            return {"pass1_rules": [], "pass2_rules": [selected_rule(99, [("flag", 1)], "pass2", 0.5)]}

    router._fit_pass1_only = fit_pass1_only
    router._rules_mask = rules_mask
    router._new_generator = Generator
    router._new_evaluator = Evaluator
    router._new_selector = Selector
    return router, rec


@pytest.fixture(scope="module")
def fitted():
    X, y = _data()
    router, rec = _stubbed_router()
    router.fit(X, y)
    routed = X["bucket"].to_numpy() == _inner_fold(y)          # OOF Pass 1 routes
    return SimpleNamespace(X=X, y=y, router=router, rec=rec, remainder=~routed)


def test_inner_pass1_routing_is_out_of_fold(fitted):
    fits = fitted.rec.log.fits()
    assert fits[1] == frozenset(fitted.X.index)                 # Pass 1 rules: full population
    assert len(fits) == 1 + INNER
    assert_oof_provenance(fitted.rec.log, fitted.X.index, methods=("predict",))


def test_remainder_is_the_rows_their_oof_pass1_rules_did_not_route(fitted):
    X, rem = fitted.X, fitted.remainder
    np.testing.assert_array_equal(fitted.router.fit_report_["remainder_mask"], rem)
    gen = fitted.rec.generators[-1]
    assert list(gen.X.index) == list(X.index[rem])              # original order kept
    assert gen.y.index.equals(gen.X.index)
    in_sample = X["bucket"].to_numpy() != FULL_BUCKET           # what the full-train rules leave
    assert not np.array_equal(rem, in_sample)                   # the check can tell them apart


def test_pass2_is_selected_on_the_remainder_with_global_denominators(fitted):
    X, y, rem, rep = fitted.X, fitted.y, fitted.remainder, fitted.router.fit_report_
    rem_pos, total_pos = int(y[rem].sum()), int(y.sum())
    assert rep["remainder_rows"] == rem.sum() and rep["remainder_positives"] == rem_pos
    assert rep["recall_scale"] == pytest.approx(rem_pos / total_pos)
    assert rep["coverage_scale"] == pytest.approx(rem.sum() / N)
    assert fitted.rec.generators[-1].recall_scale == pytest.approx(rep["recall_scale"])
    (ev,) = fitted.rec.evaluated                                 # Pass 1 candidates dropped
    assert ev.candidate.predicted_class == SUBSCRIBE
    assert list(ev.X_index) == list(X.index[rem])
    assert ev.recall == pytest.approx(ev.local[0] * rep["recall_scale"])
    assert ev.coverage == pytest.approx(ev.local[1] * rep["coverage_scale"])
    sel_index, passes = fitted.rec.selected
    assert passes == ("pass2",) and list(sel_index) == list(X.index[rem])
    assert rep["recall_denominator"] == "global"
    assert rep["pass2_precision_measured_on"] == "remainder"


def test_final_rules_are_full_population_pass1_and_remainder_pass2(fitted):
    router = fitted.router
    assert [r.segment for r in router.pass1_rules] == [frozenset({("bucket", FULL_BUCKET)})]
    assert [r.rule_id for r in router.pass2_rules] == [99]
    assert router.training_base_rate == pytest.approx(fitted.y.mean())
    assert router.is_fitted


def test_a_too_small_remainder_is_refused():
    X, y = _data()
    router, _ = _stubbed_router(remainder_min_size=N)
    with pytest.raises(ValueError, match="leaves only"):
        router.fit(X, y)


def test_a_single_class_remainder_is_refused():
    X, y = _data()
    fold = _inner_fold(y)
    # inner-fold Pass 1 rules route every positive out of fold -> remainder has none
    X = X.assign(bucket=np.where(y.to_numpy() == 1, fold, FULL_BUCKET))
    router, _ = _stubbed_router()
    with pytest.raises(ValueError, match="single-class"):
        router.fit(X, y)


def test_a_separate_evaluation_population_is_refused():
    X, y = _data()
    router, _ = _stubbed_router()
    with pytest.raises(ValueError, match="oof_remainder"):
        router.fit(X, y, X_val=X.iloc[:100], y_val=y.iloc[:100])
