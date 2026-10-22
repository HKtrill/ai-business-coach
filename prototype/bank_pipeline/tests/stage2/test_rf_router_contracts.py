"""
Black-box Stage 2 (RF Router): configuration, threshold solving, routing
cuts, the Stage 2 output triple and input validation
(``blackbox_pipeline.models.rf_router``).

Contract: configurations and constraints refuse invalid values, all at once,
and survive their JSON form; the solver picks the LARGEST feasible Pass 1 band
(``p < t1``) and the LARGEST feasible Pass 2 band (``p > t2``), scanning every
realisable band, so it matches a brute-force reference even where feasibility
is not monotone; the cut is strict on both sides, Pass 1 claims first, and an
unscored Pass 2 row never flags; ``predict_proba`` is GLASS's own transform;
the routing triple is internally consistent and contradictions are refused;
training and scoring inputs are validated alike (F36: scoring used to accept
NaN and non-binary bins); fits are reproducible.

Out-of-fold integrity, the remainder and the cross-fit are in
test_rf_router_oof.py; the artifact and the Stage 4 handoff in
test_rf_router_handoff.py; the PR 35 tuning harness in
test_rf_router_specialization.py.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from blackbox_pipeline.models.rf_router import (
    DEFAULT_RF_PARAMS, OperatingConstraints, RFRouter, RFRouterConfig, SingleRFRouter,
    ThresholdSolver,
)
from blackbox_pipeline.models.rf_router.evaluation import evaluate_router
from blackbox_pipeline.models.rf_router.router.base import apply_cuts
from tests.stage2.conftest import new_router
from tests.support import mutations as mut
from tests.support.rf_router_world import serial_forests
from tests.support.stage2_synthetic import (
    CONSTRAINTS_DEGENERATE, CONSTRAINTS_OK, SMALL_PARAMS, make_data, router_config,
)


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def router(data):
    X, y = data
    with serial_forests():
        return RFRouter(router_config()).fit(X, y, "fp-test")


@pytest.fixture(scope="module")
def unseen():
    """Rows the routers never trained on (same generator, another seed)."""
    return make_data(n=600, seed=5)[0]


def _ablation(constraints, data):
    X, y = data
    with serial_forests():
        return SingleRFRouter(router_config(constraints=constraints, mode="single_rf")).fit(X, y, "fp")


@pytest.fixture(scope="module")
def ablation(data):
    return _ablation(CONSTRAINTS_OK, data)


# ----------------------------------------------------------------------
# configuration and constraints
# ----------------------------------------------------------------------
@pytest.mark.parametrize("override, message", [
    (dict(mode="three_pass"), "mode"),
    (dict(constraints=None), "constraints is required"),
    (dict(constraints={"min_precision_not_subscribe": 0.9}), "OperatingConstraints"),
    (dict(rf_params={"n_estimators": 10}), "rf_params missing keys"),
    (dict(pass2_rf_params={"n_estimators": 10}), "pass2_rf_params missing keys"),
    (dict(n_oof_folds=1), "n_oof_folds"),
    (dict(n_inner_folds=1), "n_inner_folds"),
    (dict(remainder_min_size=0), "remainder_min_size"),
    (dict(mode="single_rf", validate_nested=True), "validate_nested"),
])
def test_invalid_router_configs_are_refused(override, message):
    with pytest.raises(ValueError, match=message):
        router_config(**override)


def test_every_config_error_is_reported_at_once():
    with pytest.raises(ValueError) as err:
        router_config(mode="bogus", n_oof_folds=1, remainder_min_size=0)
    assert all(m in str(err.value) for m in ("mode", "n_oof_folds", "remainder_min_size"))


def test_each_router_requires_its_own_mode():
    with pytest.raises(ValueError, match="two_pass"):
        RFRouter(router_config(mode="single_rf"))
    with pytest.raises(ValueError, match="single_rf"):
        SingleRFRouter(router_config(mode="two_pass"))


def test_config_defaults_and_json_round_trip():
    cfg = RFRouterConfig(mode="two_pass", constraints=CONSTRAINTS_OK)
    assert cfg.rf_params == DEFAULT_RF_PARAMS and cfg.effective_pass2_params == DEFAULT_RF_PARAMS
    tuned = router_config(pass2_rf_params={**SMALL_PARAMS, "max_depth": 3})
    assert tuned.effective_pass2_params["max_depth"] == 3 and tuned.rf_params == SMALL_PARAMS
    restored = RFRouterConfig.from_dict(json.loads(json.dumps(tuned.to_dict())))
    assert restored.to_dict() == tuned.to_dict() and restored.constraints == tuned.constraints


@pytest.mark.parametrize("field, value", [
    ("min_precision_not_subscribe", 1.2), ("max_subscriber_leakage_rate", -0.1),
    ("min_precision_subscribe", 1.5), ("min_pass1_coverage", 2.0),
    ("min_recall_subscribe", -0.5), ("max_subscriber_leakage_absolute", -1),
    ("recall_denominator", "remainder_only"), ("min_band_support_pass1", 0),
    ("min_band_support_pass2", 0),
])
def test_invalid_constraints_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(CONSTRAINTS_OK, **{field: value})


def test_constraints_survive_their_json_form():
    c = dataclasses.replace(CONSTRAINTS_OK, min_recall_subscribe=0.1, recall_denominator="remainder")
    assert OperatingConstraints.from_dict(json.loads(json.dumps(c.to_dict()))) == c


# ----------------------------------------------------------------------
# threshold solver vs a brute-force reference
# ----------------------------------------------------------------------
def _scores(seed, n=400):
    """Scores with many ties (2 decimals) and a weak signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(0.3 + 0.25 * (y - 0.25) + rng.normal(0, 0.18, n), 0, 1).round(2)
    return p, y


def _brute_pass1(p, y, c):
    """Largest t with {p < t} feasible, over every realisable band."""
    best, n, total_pos = None, len(p), int(y.sum())
    for t in np.append(np.unique(p)[1:], np.nextafter(p.max(), np.inf)):
        band = p < t
        k, pos = int(band.sum()), int(y[band].sum())
        ok = (k >= c.min_band_support_pass1 and (k - pos) / k >= c.min_precision_not_subscribe
              and pos / total_pos <= c.max_subscriber_leakage_rate
              and pos <= c.max_subscriber_leakage_absolute
              and (c.min_pass1_coverage is None or k / n >= c.min_pass1_coverage))
        if ok:
            best = (float(t), k, pos)
    return best


def _brute_pass2(p, y, c, n_pos_global):
    """Smallest t with {p > t} feasible, over every realisable band."""
    denom = n_pos_global if c.recall_denominator == "global" else int(y.sum())
    for t in np.append(np.nextafter(p.min(), -np.inf), np.unique(p)[:-1]):
        band = p > t
        m, pos = int(band.sum()), int(y[band].sum())
        if (m >= c.min_band_support_pass2 and pos / m >= c.min_precision_subscribe
                and (c.min_recall_subscribe is None or pos / denom >= c.min_recall_subscribe)):
            return float(t), m, pos
    return None


PASS1_CASES = {
    "npv_leak": dict(min_precision_not_subscribe=0.85, max_subscriber_leakage_rate=0.35,
                     max_subscriber_leakage_absolute=10_000, min_band_support_pass1=30),
    "abs_cap_binds": dict(min_precision_not_subscribe=0.70, max_subscriber_leakage_rate=1.0,
                          max_subscriber_leakage_absolute=12, min_band_support_pass1=10),
    "coverage_floor": dict(min_precision_not_subscribe=0.80, max_subscriber_leakage_rate=0.5,
                           max_subscriber_leakage_absolute=10_000, min_pass1_coverage=0.5,
                           min_band_support_pass1=10),
}
PASS2_CASES = {
    "precision": dict(min_precision_subscribe=0.45, min_band_support_pass2=20),
    "recall_global": dict(min_precision_subscribe=0.35, min_recall_subscribe=0.3,
                          min_band_support_pass2=10),
    "recall_remainder": dict(min_precision_subscribe=0.35, min_recall_subscribe=0.5,
                             recall_denominator="remainder", min_band_support_pass2=10),
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("case", sorted(PASS1_CASES))
def test_pass1_threshold_is_the_largest_feasible_band(seed, case):
    p, y = _scores(seed)
    c = dataclasses.replace(CONSTRAINTS_OK, **PASS1_CASES[case])
    result = ThresholdSolver(c).solve_pass1(p, y)
    expected = _brute_pass1(p, y, c)
    if expected is None:
        assert not result.feasible and np.isnan(result.threshold)
        return
    t, k, pos = expected
    assert (result.threshold, result.band_size, result.positives_in_band) == (t, k, pos)
    assert result.band_precision == pytest.approx((k - pos) / k)
    assert result.leakage_rate == pytest.approx(pos / y.sum())
    assert ((p < result.threshold).sum(), y[p < result.threshold].sum()) == (k, pos)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("case", sorted(PASS2_CASES))
def test_pass2_threshold_is_the_largest_feasible_band(seed, case):
    p, y = _scores(seed)
    c = dataclasses.replace(CONSTRAINTS_OK, **PASS2_CASES[case])
    n_global = int(y.sum()) * 2                        # a remainder holds a share of all positives
    result = ThresholdSolver(c).solve_pass2(p, y, n_positives_global=n_global)
    expected = _brute_pass2(p, y, c, n_global)
    if expected is None:
        assert not result.feasible
        return
    t, m, pos = expected
    assert (result.threshold, result.band_size, result.positives_in_band) == (t, m, pos)
    denom = n_global if c.recall_denominator == "global" else y.sum()
    assert result.recall == pytest.approx(pos / denom)
    assert ((p > result.threshold).sum(), y[p > result.threshold].sum()) == (m, pos)


def test_the_solver_scans_past_an_infeasible_band():
    """Empirical precision is not monotone: a scan that stopped at the first
    infeasible band would route 50 rows instead of 120."""
    p = np.repeat([0.1, 0.2, 0.3, 0.9], [50, 10, 60, 20])
    y = np.concatenate([np.zeros(50), [1] * 6 + [0] * 4, np.zeros(60), np.ones(20)]).astype(int)
    c = dataclasses.replace(CONSTRAINTS_OK, min_precision_not_subscribe=0.93,
                            max_subscriber_leakage_rate=1.0, min_band_support_pass1=1)
    r1 = ThresholdSolver(c).solve_pass1(p, y)
    assert r1.sweep["feasible"].tolist() == [True, False, True, False]
    assert (r1.threshold, r1.band_size) == (0.9, 120)
    q = 1.0 - p                                         # the mirror image for Pass 2
    c2 = dataclasses.replace(CONSTRAINTS_OK, min_precision_subscribe=0.93, min_band_support_pass2=1)
    r2 = ThresholdSolver(c2).solve_pass2(q, 1 - y, n_positives_global=int((1 - y).sum()))
    assert r2.band_size == 120 and ((q > r2.threshold).sum(), (q > r2.threshold)[:50].all()) == (120, True)


def test_infeasible_operating_points_raise_with_the_sweep():
    p, y = _scores(0)
    c = dataclasses.replace(CONSTRAINTS_OK, min_precision_not_subscribe=0.999,
                            min_band_support_pass1=300, min_precision_subscribe=0.999,
                            min_band_support_pass2=200)       # ~75% of the positives must sit in a band
    solver = ThresholdSolver(c)
    for result in (solver.solve_pass1(p, y), solver.solve_pass2(p, y, n_positives_global=int(y.sum()))):
        assert not result.feasible and result.sweep is not None and not result.sweep["feasible"].any()
        with pytest.raises(ValueError, match="No feasible"):
            result.raise_if_infeasible()


@pytest.mark.parametrize("bad, message", [
    (dict(proba=[0.1, np.nan]), "NaN"), (dict(y=[0, 2]), "binary"),
    (dict(proba=[0.1]), "length mismatch"),
])
def test_the_solver_refuses_invalid_scores(bad, message):
    args = {"proba": [0.1, 0.2], "y": [0, 1], **bad}
    with pytest.raises(ValueError, match=message):
        ThresholdSolver(CONSTRAINTS_OK).solve_pass1(np.asarray(args["proba"]), np.asarray(args["y"]))


# ----------------------------------------------------------------------
# the cut and the output triple
# ----------------------------------------------------------------------
def test_cuts_are_strict_pass1_claims_first_and_unscored_rows_never_flag():
    p1 = np.array([0.1, 0.3, 0.3, 0.5, 0.7, 0.2])
    p2 = np.array([0.9, 0.6, 0.61, np.nan, 0.2, np.nan])
    preds, decisions = apply_cuts(p1, p2, t1=0.3, t2=0.6)
    assert decisions.tolist() == ["pass1", "uncertain", "pass2", "uncertain", "uncertain", "pass1"]
    assert preds.tolist() == [0, -1, 1, -1, -1, 0]


def test_the_triple_is_consistent_and_confidence_is_the_band_precision(router, unseen):
    preds, conf, decisions = router.predict(unseen)
    code = {"pass1": 0, "pass2": 1, "uncertain": -1}
    np.testing.assert_array_equal(preds, [code[d] for d in decisions])
    assert set(decisions) == {"pass1", "pass2", "uncertain"}             # all three routes occur
    b = router.bands_
    expected = {"pass1": b.pass1_confidence, "pass2": b.pass2_confidence, "uncertain": 0.0}
    np.testing.assert_array_equal(conf, [expected[d] for d in decisions])


def test_probabilities_are_glass_routers_own_transform(router, unseen):
    """``predict_proba`` must equal ``GlassRouterPipeline.predict_proba`` on the
    same triple and base rate — the arms are compared on these numbers."""
    X = unseen
    triple = router.predict(X)
    glass = new_router()
    glass.is_fitted, glass.training_base_rate = True, router.training_base_rate
    glass.predict = lambda _X: triple
    np.testing.assert_allclose(router.predict_proba(X), glass.predict_proba(X), rtol=0, atol=1e-15)
    proba, base = router.predict_proba(X), router.training_base_rate
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    by_route = {d: proba[triple[2] == d, 1] for d in ("pass1", "pass2", "uncertain")}
    assert (by_route["pass1"] < base).all() and (by_route["pass2"] > base).all()
    np.testing.assert_allclose(by_route["uncertain"], base)


@pytest.mark.parametrize("preds, decisions, message", [
    ([1], ["pass1"], "does not carry prediction 0"),
    ([0], ["pass2"], "does not carry prediction 1"),
    ([0], ["uncertain"], "does not carry prediction -1"),
    ([0], ["abstain"], "decisions must be"),
    ([2], ["pass1"], "preds must be"),
])
def test_contradictory_routing_states_are_refused(preds, decisions, message):
    with pytest.raises(ValueError, match=message):
        evaluate_router(np.array(preds), np.zeros(1), np.array(decisions, dtype=object), [0])


def test_the_ablation_lets_pass1_claim_overlapping_bands(data):
    """Degenerate constraints make the one-score bands overlap (t1 > t2): the
    ordered solve refuses, the relaxed one routes the overlap to Pass 1."""
    with pytest.raises(ValueError, match="Band ordering violated"):
        _ablation(CONSTRAINTS_DEGENERATE, data)
    relaxed = _ablation(dataclasses.replace(CONSTRAINTS_DEGENERATE, require_band_ordering=False), data)
    t1, t2, p = relaxed.thresholds_.t1, relaxed.thresholds_.t2, relaxed.oof_proba_
    overlap = (p > t2) & (p < t1)
    assert t1 > t2 and overlap.any()
    _, _, decisions = relaxed.predict_train_oof()
    assert (decisions[overlap] == "pass1").all()


# ----------------------------------------------------------------------
# training and scoring inputs
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case, error, message", [
    ("array", TypeError, "DataFrame"),
    ("shuffled_rows", ValueError, "index"), ("reset_index", ValueError, "index"),
    ("foreign_label", ValueError, "index"),
    ("nan_feature", ValueError, "NaN"), ("non_binary_feature", ValueError, "not binary"),
    ("non_binary_label", ValueError, "binary 0/1"),
])
def test_invalid_training_inputs_are_refused(data, case, error, message):
    X, y = data
    if case == "array":
        X = X.to_numpy()
    elif case in mut.INDEX_MUTATIONS:
        y = mut.INDEX_MUTATIONS[case](y)
    elif case == "nan_feature":
        X = X.astype(float).copy()
        X.iloc[0, 0] = np.nan
    elif case == "non_binary_feature":
        X = X.copy()
        X.iloc[0, 0] = 2
    else:
        y = y.copy()
        y.iloc[0] = 2
    with pytest.raises(error, match=message):
        RFRouter(router_config()).fit(X, y, "fp")


@pytest.mark.parametrize("override, message", [
    (dict(remainder_min_size=10_000), "remainder_min_size"),
    (dict(constraints=dataclasses.replace(CONSTRAINTS_OK, min_precision_not_subscribe=0.999,
                                          min_band_support_pass1=1000)), "No feasible Pass 1"),
    (dict(constraints=dataclasses.replace(CONSTRAINTS_OK, min_precision_subscribe=0.999,
                                          min_band_support_pass2=200)), "No feasible Pass 2"),
])
def test_an_unusable_operating_point_stops_the_fit(data, override, message):
    X, y = data
    with serial_forests(), pytest.raises(ValueError, match=message):
        RFRouter(router_config(**override)).fit(X, y, "fp")


@pytest.mark.parametrize("router_kind", ["two_pass", "single_rf"])
@pytest.mark.parametrize("value, message", [(np.nan, "NaN"), (2, "not binary"), (0.5, "not binary")])
def test_invalid_scoring_rows_are_refused_not_routed_f36(router, ablation, unseen, router_kind,
                                                         value, message):
    """Before F36 these rows were silently given a decision (0.5 changed one)."""
    model = router if router_kind == "two_pass" else ablation
    X = unseen.iloc[:20].astype(float)
    X.iloc[3, 2] = value
    with pytest.raises(ValueError, match=message):
        model.predict(X)
    with pytest.raises(ValueError, match=message):
        model.predict_with_proba(X)


def test_scoring_columns_are_matched_by_name(router, unseen):
    X = unseen
    base = router.predict(X)
    for frame in (mut.swap_columns(X, 0, 5), mut.add_column(X, "unused", 1)):
        for got, want in zip(router.predict(frame), base):
            np.testing.assert_array_equal(got, want)
    for name in ("missing_column", "renamed_column"):
        with pytest.raises(ValueError, match="missing fitted features"):
            router.predict(mut.COLUMN_MUTATIONS[name](X))


def test_unfitted_routers_refuse_every_output(data):
    X = data[0]
    for model in (RFRouter(router_config()), SingleRFRouter(router_config(mode="single_rf"))):
        for call in (model.predict, model.predict_proba, model.predict_with_proba):
            with pytest.raises(ValueError, match="fit"):
                call(X)


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------
def test_serial_fits_are_bit_identical_and_the_seed_matters(router, data):
    X, y = data
    with serial_forests():
        again = RFRouter(router_config()).fit(X, y, "fp-test")
        other = RFRouter(router_config(random_state=8)).fit(X, y, "fp-test")
    np.testing.assert_array_equal(again.oof_pass1_proba_, router.oof_pass1_proba_)
    np.testing.assert_array_equal(again.oof_pass2_proba_, router.oof_pass2_proba_)
    assert (again.thresholds_.t1, again.thresholds_.t2) == (router.thresholds_.t1, router.thresholds_.t2)
    assert not np.allclose(other.oof_pass1_proba_, router.oof_pass1_proba_)


def test_production_parallel_fits_agree_to_the_documented_precision(router, data, unseen):
    """``n_jobs=-1`` (production) sums tree probabilities in a varying order:
    thresholds agree to ~1e-12, decisions exactly (forest.build_pipe)."""
    X, y = data
    parallel = RFRouter(router_config()).fit(X, y, "fp-test")
    np.testing.assert_allclose(parallel.oof_pass1_proba_, router.oof_pass1_proba_, rtol=0, atol=1e-12)
    assert parallel.thresholds_.t1 == pytest.approx(router.thresholds_.t1, abs=1e-12)
    assert parallel.thresholds_.t2 == pytest.approx(router.thresholds_.t2, abs=1e-12)
    np.testing.assert_array_equal(parallel.remainder_mask_, router.remainder_mask_)
    np.testing.assert_array_equal(parallel.predict_train_oof()[2], router.predict_train_oof()[2])
    np.testing.assert_array_equal(parallel.predict(unseen)[2], router.predict(unseen)[2])
