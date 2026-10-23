"""
GLASS Router inference semantics (GlassRouterPipeline.predict / predict_proba
and glass_router.model_outputs.build_glass_output), with hand-built rules.

Contract: every row lands in exactly one of three states - NOT_SUBSCRIBE by a
Pass 1 rule, SUBSCRIBE by a Pass 2 rule, or ABSTAIN - aligned to the input
rows by position; Pass 1 is applied first and its rows are never re-routed by
Pass 2; within a pass the first matching rule, in selection order, sets the
confidence; probabilities follow from the routes; the decision vocabulary is
the one Stage 4 accepts; and malformed rules or inputs fail loudly.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from glass_pipeline.glass_router.core.rule import ABSTAIN, NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.model_outputs import build_glass_output
from shared.stage4.adapters import normalize_router_decisions
from tests.stage2.conftest import new_router, router_with_rules, selected_rule
from tests.support import mutations as mut
from tests.support.synthetic import irregular_index

PASS1 = [selected_rule(1, [("a", 1)], "pass1", 0.95),
         selected_rule(2, [("b", 1)], "pass1", 0.90)]
PASS2 = [selected_rule(3, [("c", 1)], "pass2", 0.40),
         selected_rule(4, [("b", 0), ("c", 1)], "pass2", 0.30)]       # inside rule 3
STATE = {"pass1": NOT_SUBSCRIBE, "pass2": SUBSCRIBE, "uncertain": ABSTAIN}


@pytest.fixture
def X():
    """Every combination of three binary features, three times, shuffled, irregular index."""
    combos = list(itertools.product([0, 1], repeat=3)) * 3
    frame = pd.DataFrame(combos, columns=["a", "b", "c"], dtype="int8")
    frame.index = irregular_index(len(frame), seed=3)
    return mut.shuffle_rows(frame, seed=4)


def _reference(X, pass1, pass2):
    """Row-by-row restatement of the routing rule, independent of the vectorized code."""
    out = []
    for _, row in X.iterrows():
        route = (ABSTAIN, 0.0, "uncertain")
        for rules, pred, name in ((pass1, NOT_SUBSCRIBE, "pass1"), (pass2, SUBSCRIBE, "pass2")):
            hit = next((r for r in rules if all(row[f] == v for f, v in r.segment)), None)
            if hit is not None:
                route = (pred, hit.precision, name)
                break
        out.append(route)
    pred, conf, dec = zip(*out)
    return np.array(pred), np.array(conf), np.array(dec, dtype=object)


# ----------------------------------------------------------------------
# three-state routing
# ----------------------------------------------------------------------
def test_routes_match_the_reference_rule(X):
    pred, conf, dec = router_with_rules(PASS1, PASS2).predict(X)
    ref = _reference(X, PASS1, PASS2)
    np.testing.assert_array_equal(pred, ref[0])
    np.testing.assert_allclose(conf, ref[1])
    np.testing.assert_array_equal(dec, ref[2])
    assert set(dec) == {"pass1", "pass2", "uncertain"}                 # all states exercised


def test_states_are_exclusive_exhaustive_and_consistent(X):
    router = router_with_rules(PASS1, PASS2)
    pred, conf, dec = router.predict(X)
    assert len(pred) == len(conf) == len(dec) == len(X)
    np.testing.assert_array_equal(pred, [STATE[d] for d in dec])
    assert ((conf == 0) == (dec == "uncertain")).all()
    assert ((conf > 0) & (conf <= 1))[dec != "uncertain"].all()
    out = build_glass_output(router, X, abstain_value=ABSTAIN)
    assert (out["covered"] ^ out["abstained"]).all()
    np.testing.assert_array_equal(out["abstained"], dec == "uncertain")
    np.testing.assert_array_equal(out["pred"], pred)


def test_pass1_rows_are_never_rerouted_by_pass2(X):
    pred, _, dec = router_with_rules(PASS1, PASS2).predict(X)
    claimed_by_pass1 = (X["a"] == 1) | (X["b"] == 1)
    claimed_by_pass2 = X["c"] == 1
    both = (claimed_by_pass1 & claimed_by_pass2).to_numpy()
    assert both.any()
    assert (dec[both] == "pass1").all() and (pred[both] == NOT_SUBSCRIBE).all()
    only_pass2 = (~claimed_by_pass1 & claimed_by_pass2).to_numpy()
    assert (dec[only_pass2] == "pass2").all()
    # without Pass 1 the same rows go to Pass 2
    _, _, dec_no_p1 = router_with_rules([], PASS2).predict(X)
    assert (dec_no_p1[both] == "pass2").all()


def test_first_matching_rule_in_selection_order_sets_the_confidence(X):
    both = ((X["a"] == 1) & (X["b"] == 1)).to_numpy()
    _, conf, _ = router_with_rules(PASS1, PASS2).predict(X)
    _, conf_rev, _ = router_with_rules(PASS1[::-1], PASS2).predict(X)
    assert (conf[both] == 0.95).all() and (conf_rev[both] == 0.90).all()
    nested = ((X["a"] == 0) & (X["b"] == 0) & (X["c"] == 1)).to_numpy()
    _, conf2, _ = router_with_rules(PASS1, PASS2).predict(X)
    _, conf2_rev, _ = router_with_rules(PASS1, PASS2[::-1]).predict(X)
    assert (conf2[nested] == 0.40).all() and (conf2_rev[nested] == 0.30).all()


def test_routing_is_row_wise_and_follows_input_order(X):
    router = router_with_rules(PASS1, PASS2)
    base = router.predict(X)
    order = np.random.default_rng(0).permutation(len(X))
    moved = router.predict(X.iloc[order])
    for a, b in zip(base, moved):
        np.testing.assert_array_equal(a[order], b)
    reset = router.predict(mut.reset_index(X))                         # index labels play no part
    for a, b in zip(base, reset):
        np.testing.assert_array_equal(a, b)


def test_abstained_rows_are_forwarded_in_their_original_order(X):
    router = router_with_rules(PASS1, PASS2)
    out = build_glass_output(router, X, abstain_value=ABSTAIN)
    forwarded = X[out["abstained"]]
    assert list(forwarded.index) == [i for i, m in zip(X.index, out["abstained"]) if m]
    _, _, dec = router.predict(forwarded)
    assert (dec == "uncertain").all()                                  # re-routing is stable


def test_no_rules_means_every_row_abstains(X):
    pred, conf, dec = router_with_rules([], []).predict(X)
    assert (pred == ABSTAIN).all() and (conf == 0).all() and (dec == "uncertain").all()


def test_empty_input_gives_empty_outputs(X):
    router = router_with_rules(PASS1, PASS2)
    pred, conf, dec = router.predict(X.iloc[:0])
    assert len(pred) == len(conf) == len(dec) == 0
    assert router.predict_proba(X.iloc[:0]).shape == (0, 2)


def test_decision_vocabulary_is_what_stage4_accepts(X):
    _, _, dec = router_with_rules(PASS1, PASS2).predict(X)
    normalized = normalize_router_decisions(dec, "glass router")
    np.testing.assert_array_equal(
        normalized, np.where(dec == "uncertain", "abstain", dec).astype(normalized.dtype))


# ----------------------------------------------------------------------
# probabilities
# ----------------------------------------------------------------------
@pytest.mark.parametrize("base_rate", [None, 0.35])
def test_probabilities_follow_from_the_routes(X, base_rate):
    router = router_with_rules(PASS1, PASS2, base_rate=0.2)
    _, conf, dec = router.predict(X)
    proba = router.predict_proba(X, base_rate=base_rate)
    b = 0.2 if base_rate is None else base_rate
    expected = np.where(dec == "pass1", b * (1 - conf),
                        np.where(dec == "pass2", b + (1 - b) * conf, b))
    np.testing.assert_allclose(proba[:, 1], expected)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert (proba[dec == "pass1", 1] <= b).all() and (proba[dec == "pass2", 1] >= b).all()


# ----------------------------------------------------------------------
# refused states and inputs
# ----------------------------------------------------------------------
def test_an_unfitted_router_refuses_to_route(X):
    with pytest.raises(ValueError, match="fit"):
        new_router().predict(X)


@pytest.mark.parametrize("rules", ["pass1", "pass2"])
def test_a_missing_rule_feature_is_refused(X, rules):
    router = router_with_rules(PASS1, PASS2)
    column = "a" if rules == "pass1" else "c"
    with pytest.raises(ValueError, match="Missing required segment feature"):
        router.predict(X.drop(columns=[column]))


def test_a_rule_without_precision_is_refused(X):
    router = router_with_rules([selected_rule(9, [("a", 1)], "pass1", None)], [])
    with pytest.raises(ValueError, match="missing precision"):
        router.predict(X)


def test_probabilities_need_a_base_rate(X):
    router = router_with_rules(PASS1, PASS2)
    router.training_base_rate = None
    with pytest.raises(ValueError, match="training_base_rate"):
        router.predict_proba(X)
    assert router.predict_proba(X, base_rate=0.1).shape == (len(X), 2)


def test_selected_rules_validate_their_pass_and_freeze_their_segment():
    rule = selected_rule(1, [("a", 1), ("b", 0)], "pass1", 0.95)
    assert rule.segment == frozenset({("a", 1), ("b", 0)})
    with pytest.raises(ValueError, match="pass_assignment"):
        selected_rule(2, [("a", 1)], "pass3", 0.9)
