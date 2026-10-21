"""
GLASS Stage 4 Meta-EBM: the weighted-confidence arbiter and the train-side
pieces that configure it (glass_pipeline.meta_ebm: arbiter, weighting,
calibration, tuning, evaluation).

Contract: the arbiter equals an independent vectorised statement of its rule
(LR and EBM vote on every row; the router votes only on the rows it routes,
Pass 2 -> 1, Pass 1 -> 0; weights renormalised over the voters; ties -> 0;
abstain when the winning side's weighted confidence is below the cut);
router probabilities on rows it does not route are inert; abstaining never
changes a retained row's decision; misaligned inputs are refused (F27). The
trust weights blend inverse Brier and accuracy as documented, scoring the
router on the rows it votes on; the calibration diagnostic is the shared ECE;
the abstention cut is the shared Stage 4 rule (``tune_min_confidence``,
an exact search since F30) applied to the arbiter's own confidence; disagreement with the router is
measured only where it votes.

Router evidence (F25): a routed row's confidence is its rule's positive rate
measured against the router's training base rate in the vote's direction,
clipped at zero (``router_evidence``). It never falls as a Pass 2 rule's
precision (P(y = 1 | rule)) or a Pass 1 rule's precision (P(y = 0 | rule))
rises, is zero for a rule at or beyond the base rate on the wrong side, is
finite and nonnegative, and is read back exactly from
``GlassRouterPipeline.predict_proba``. Before F25 it was |p - 0.5|, which
weakened Pass 2 votes as their precision rose (every selected Pass 2 rule
has P(subscribe) < 0.5).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, brier_score_loss, fbeta_score

from glass_pipeline.glass_router.pipeline.glass_router_pipeline import GlassRouterPipeline
from glass_pipeline.meta_ebm.arbiter import meta_arbiter
from glass_pipeline.meta_ebm.calibration import compute_calibration
from glass_pipeline.meta_ebm.evaluation import (
    analyze_disagreements, compute_metrics, evaluate_with_abstention,
)
from glass_pipeline.meta_ebm.router_evidence import (
    directional_confidence, fold_base_rates, glass_blend_mismatch, rule_positive_rate,
)
from glass_pipeline.meta_ebm.tuning import tune_arbiter_threshold
from glass_pipeline.meta_ebm.weighting import compute_hybrid_weights
from shared.metrics import calculate_ece
from shared.stage4 import ABSTENTION_GRID, ABSTENTION_SEARCH, tune_min_confidence

WEIGHTS = {"lr": 0.35, "ebm": 0.36, "glass": 0.29}          # the notebook's last run
T_LR, T_EBM = 0.08, 0.07


@pytest.fixture(scope="module")
def streams():
    """Train-side-like streams: labels, LR / EBM / router probabilities, routes."""
    rng = np.random.default_rng(3)
    n = 1500
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * z - 2.2)))).astype(int)
    lr = np.clip(1 / (1 + np.exp(-(1.8 * z - 2.0 + rng.normal(0, .6, n)))), 0, 1)
    ebm = np.clip(1 / (1 + np.exp(-(2.1 * z - 2.1 + rng.normal(0, .5, n)))), 0, 1)
    route = np.where(z < -0.3, 1, np.where(z > 1.0, 2, 0))       # 1 = Pass 1, 2 = Pass 2
    p1, p2 = route == 1, route == 2
    base = y.mean()
    # rule precision for the routed class; a wide range puts some rules on the
    # wrong side of the base rate (Pass 1 precision < 1 - base, Pass 2 < base)
    precision = np.where(route == 0, 0.0, rng.uniform(0.05, 0.99, n))
    gl = np.where(p1, base * (1 - precision), np.where(p2, base + (1 - base) * precision, np.nan))
    return {"y": y, "lr": lr, "ebm": ebm, "gl": gl, "p1": p1, "p2": p2, "b": base,
            "precision": precision}


def reference(s, weights=WEIGHTS, allow_abstain=True, min_conf=0.15):
    """The arbiter's rule, vectorised: (pred, weighted prob, max side confidence).
    Router confidence from the stored rule precision, not from its probability."""
    lr, ebm, gl, p1, p2, b = s["lr"], s["ebm"], s["gl"], s["p1"], s["p2"], s["b"]
    rate = np.where(p1, 1 - s["precision"], s["precision"])      # rule positive rate
    router_conf = np.where(p2, np.maximum(0, rate - b), np.where(p1, np.maximum(0, b - rate), 0))
    voting = p1 | p2
    W = np.column_stack([np.full(len(lr), weights["lr"]), np.full(len(lr), weights["ebm"]),
                         np.where(voting, weights["glass"], 0.0)])
    W = W / W.sum(axis=1, keepdims=True)
    g = np.where(voting, gl, 0.0)
    P = np.column_stack([lr, ebm, g])
    V = np.column_stack([lr >= T_LR, ebm >= T_EBM, p2]).astype(int)
    C = np.column_stack([np.abs(lr - T_LR), np.abs(ebm - T_EBM), router_conf])
    c1, c0 = (C * W * (V == 1)).sum(1), (C * W * (V == 0)).sum(1)
    pred = np.where(c1 > c0, 1, 0)
    conf = np.maximum(c1, c0)
    if allow_abstain:
        pred = np.where(conf < min_conf, -1, pred)
    return pred, (P * W).sum(1), conf


def arbiter(s, weights=WEIGHTS, **kw):
    return meta_arbiter(s["lr"], s["ebm"], s["gl"], s["p1"], s["p2"], T_LR, T_EBM, weights,
                        glass_base_rate=s["b"], **kw)


# ----------------------------------------------------------------------
# the arbiter rule
# ----------------------------------------------------------------------
@pytest.mark.parametrize("allow_abstain, min_conf", [(False, 0.15), (True, 0.05), (True, 0.15)])
def test_arbiter_matches_its_stated_rule(streams, allow_abstain, min_conf):
    pred, prob, explain = arbiter(streams, allow_abstain=allow_abstain,
                                  min_weighted_confidence=min_conf)
    ref_pred, ref_prob, _ = reference(streams, allow_abstain=allow_abstain, min_conf=min_conf)
    assert (streams["p2"] & (streams["precision"] < streams["b"])).any()      # wrong-side rules
    np.testing.assert_array_equal(pred, ref_pred)
    np.testing.assert_allclose(prob, ref_prob, rtol=0, atol=1e-12)
    assert set(np.unique(pred)) <= {-1, 0, 1}
    labels = {-1: "ABSTAIN", 0: "PREDICT 0", 1: "PREDICT 1"}
    assert all(e.startswith(labels[p]) for p, e in zip(pred, explain))


def test_weighted_probability_is_a_convex_combination_of_the_voters(streams):
    _, prob, _ = arbiter(streams, allow_abstain=False)
    voting = streams["p1"] | streams["p2"]
    g = np.where(voting, streams["gl"], np.nan)
    stack = np.column_stack([streams["lr"], streams["ebm"], g])
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    assert (prob >= np.nanmin(stack, axis=1) - 1e-12).all()
    assert (prob <= np.nanmax(stack, axis=1) + 1e-12).all()


def test_router_values_where_it_does_not_route_are_inert(streams):
    """Abstain rows may carry NaN (set to 0.5 by the loader) or anything else."""
    base = arbiter(streams, min_weighted_confidence=0.1)
    moved = dict(streams, gl=np.where(streams["p1"] | streams["p2"], streams["gl"], 0.99))
    other = arbiter(moved, min_weighted_confidence=0.1)
    np.testing.assert_array_equal(other[0], base[0])
    np.testing.assert_array_equal(other[1], base[1])


def test_abstaining_only_withholds_decisions(streams):
    """A retained row keeps its no-abstention decision; a higher cut abstains
    on a superset of rows; a zero cut abstains on none."""
    no_abstain, _, _ = arbiter(streams, allow_abstain=False)
    prev = np.zeros(len(no_abstain), dtype=bool)
    for mc in (0.0, 0.05, 0.1, 0.2, 0.4):
        pred, _, _ = arbiter(streams, allow_abstain=True, min_weighted_confidence=mc)
        abstained = pred == -1
        np.testing.assert_array_equal(pred[~abstained], no_abstain[~abstained])
        assert (abstained >= prev).all()
        prev = abstained
    assert not (arbiter(streams, min_weighted_confidence=0.0)[0] == -1).any()


def test_ties_go_to_the_negative_side():
    one = np.array([0.5])
    pred, _, _ = meta_arbiter(one, one, one, np.array([False]), np.array([False]),
                              0.5, 0.5, WEIGHTS, allow_abstain=False, glass_base_rate=0.1)
    assert pred.tolist() == [0]                          # both sides carry zero confidence


@pytest.mark.parametrize("case, error, message", [
    ("short_ebm", ValueError, "differ in length"),
    ("long_router", ValueError, "differ in length"),
    ("both_routes", ValueError, "both Pass 1 and Pass 2"),
    ("missing_weight", KeyError, "glass"),
    ("no_router_base_rate", ValueError, "glass_base_rate is required"),
])
def test_misaligned_or_inconsistent_inputs_are_refused_f27(streams, case, error, message):
    """F27: the loop indexed by len(lr_prob) used to drop extra rows silently,
    and Pass 2 silently won a row routed to both passes."""
    s, weights = dict(streams), dict(WEIGHTS)
    if case == "short_ebm":
        s["ebm"] = s["ebm"][:-1]
    elif case == "long_router":
        s["gl"] = np.append(s["gl"], 0.5)
    elif case == "both_routes":
        s["p1"] = s["p1"].copy()
        s["p1"][np.flatnonzero(s["p2"])[0]] = True
    elif case == "missing_weight":
        del weights["glass"]
    else:
        with pytest.raises(error, match=message):
            meta_arbiter(s["lr"], s["ebm"], s["gl"], s["p1"], s["p2"], T_LR, T_EBM, weights)
        return
    with pytest.raises(error, match=message):
        arbiter(s, weights=weights)


# ----------------------------------------------------------------------
# router evidence (F25)
# ----------------------------------------------------------------------
BASE = 0.113                                               # training base rate, bank data


def _router_votes(precision, route, lean):
    """Decisions for one routed row per precision; LR and EBM lean ``lean``
    (weakly) and the router's probability is GlassRouterPipeline's."""
    n = len(precision)
    p = BASE * (1 - precision) if route == 1 else BASE + (1 - BASE) * precision
    other = np.full(n, 0.02 if lean == 0 else 0.12)          # just below / above t = 0.07
    pred, _, _ = meta_arbiter(other, other, p, np.full(n, route == 1), np.full(n, route == 2),
                              0.07, 0.07, WEIGHTS, allow_abstain=False, glass_base_rate=BASE)
    return pred


def test_a_more_precise_pass2_rule_never_weakens_the_subscribe_vote_f25():
    """Before F25 (|p - 0.5|) these flipped 1, 0, 0, 0, 1 as precision rose."""
    precision = np.linspace(0.15, 0.95, 17)
    pred = _router_votes(precision, route=2, lean=0)
    assert (np.diff(pred) >= 0).all(), pred
    assert pred[0] == 0 and pred[-1] == 1                   # the router can carry the row


def test_a_more_precise_pass1_rule_never_weakens_the_not_subscribe_vote_f25():
    precision = np.linspace(0.80, 1.00, 21)                 # P(y = 0 | rule)
    pred = _router_votes(precision, route=1, lean=1)
    assert (np.diff(pred) <= 0).all(), pred


@pytest.mark.parametrize("route", [1, 2])
def test_router_confidence_is_monotone_directional_and_bounded_f25(route):
    precision = np.linspace(0.0, 1.0, 201)
    n = len(precision)
    p1, p2 = np.full(n, route == 1), np.full(n, route == 2)
    p = BASE * (1 - precision) if route == 1 else BASE + (1 - BASE) * precision
    conf = directional_confidence(rule_positive_rate(p, p1, p2, BASE), p1, p2, BASE)
    assert np.isfinite(conf).all() and (conf >= 0).all()
    assert (np.diff(conf) >= -1e-15).all()                  # never weaker with more precision
    rate = 1 - precision if route == 1 else precision
    wrong_side = rate >= BASE if route == 1 else rate <= BASE
    assert (conf[wrong_side] == 0).all()                    # no evidence, no confidence
    assert conf.max() == pytest.approx(BASE if route == 1 else 1 - BASE)


def test_unrouted_rows_carry_no_router_confidence():
    p = np.array([0.3, np.nan, 0.9])
    none = np.zeros(3, bool)
    rate = rule_positive_rate(p, none, none, BASE)
    assert np.isnan(rate).all()
    np.testing.assert_array_equal(directional_confidence(rate, none, none, BASE), np.zeros(3))


def test_rule_rate_is_read_back_exactly_from_the_glass_router():
    """Pins the inverse to ``GlassRouterPipeline.predict_proba`` itself."""
    rng = np.random.default_rng(0)
    n = 60
    decisions = rng.choice(np.array(["pass1", "pass2", "uncertain"], dtype=object), n)
    conf = np.where(decisions == "uncertain", 0.0, rng.uniform(0.05, 0.99, n))
    router = GlassRouterPipeline.__new__(GlassRouterPipeline)
    router.training_base_rate = BASE
    router.predict = lambda X: (np.zeros(n, int), conf, decisions)
    p = router.predict_proba(np.zeros((n, 1)))[:, 1]
    p1, p2 = decisions == "pass1", decisions == "pass2"
    rate = rule_positive_rate(p, p1, p2, BASE)
    np.testing.assert_allclose(rate[p1], 1 - conf[p1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rate[p2], conf[p2], rtol=0, atol=1e-12)
    assert glass_blend_mismatch(p, p1, p2, BASE, conf) is None
    assert glass_blend_mismatch(p, p1, p2, BASE, rng.permutation(conf)) is not None
    assert glass_blend_mismatch(p, p1, p2, BASE + 0.01, conf) is not None   # wrong base rate


def _stored_blend():
    """Routed rows as a GLASS bundle stores them: probability and rule precision."""
    dec = np.array(["pass1", "pass2", "uncertain"] * 8, dtype=object)
    prec = np.where(dec == "uncertain", 0.0, np.linspace(0.05, 0.99, len(dec)))
    p = np.where(dec == "pass1", BASE * (1 - prec), np.where(dec == "pass2", BASE + (1 - BASE) * prec, np.nan))
    return p, dec == "pass1", dec == "pass2", prec


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("route", ["pass1", "pass2"])
def test_routed_rows_with_non_finite_stored_precision_are_refused(route, bad):
    """F25 review: a NaN precision made the max-error comparison false
    (NaN > tol is False), so the row passed as a consistent blend."""
    p, p1, p2, prec = _stored_blend()
    assert glass_blend_mismatch(p, p1, p2, BASE, prec) is None
    prec[np.flatnonzero(p1 if route == "pass1" else p2)[0]] = bad
    why = glass_blend_mismatch(p, p1, p2, BASE, prec)
    assert why is not None and "non-finite stored rule precision" in why


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 7.0])
def test_unrouted_rows_keep_ignoring_their_stored_precision(bad):
    p, p1, p2, prec = _stored_blend()
    prec[~(p1 | p2)] = bad                                   # never read: no rule matched
    assert glass_blend_mismatch(p, p1, p2, BASE, prec) is None


def test_fold_base_rates_are_each_fold_models_training_rate():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
    fold = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    rates = fold_base_rates(y, fold)
    for k in range(3):
        assert (rates[fold == k] == y[fold != k].mean()).all()


@pytest.mark.parametrize("case, message", [
    ("probability_not_a_blend", "not GLASS rule blends"),
    ("nan_on_a_routed_row", "not GLASS rule blends"),
    ("base_rate_zero", "strictly between 0 and 1"),
])
def test_router_inputs_that_cannot_carry_rule_evidence_are_refused(case, message):
    p, p1, p2, base = np.array([0.05, 0.6]), np.array([True, False]), np.array([False, True]), BASE
    if case == "probability_not_a_blend":
        p = np.array([0.5, 0.6])                            # Pass 1 above its base rate
    elif case == "nan_on_a_routed_row":
        p = np.array([np.nan, 0.6])
    else:
        base = 0.0
    with pytest.raises(ValueError, match=message):
        rule_positive_rate(p, p1, p2, base)


def test_each_side_of_the_split_uses_its_own_router_base_rate():
    """Out-of-fold rows were scored by fold models trained at slightly
    different base rates; reading them at one shared rate would shift every
    recovered rule rate."""
    s_p, p1, p2 = np.array([0.35]), np.array([False]), np.array([True])
    at_fold = rule_positive_rate(s_p, p1, p2, np.array([0.110]))
    at_full = rule_positive_rate(s_p, p1, p2, np.array([0.113]))
    assert at_fold[0] != at_full[0]


# ----------------------------------------------------------------------
# what configures the arbiter (train side)
# ----------------------------------------------------------------------
def test_trust_weights_blend_inverse_brier_and_accuracy(streams):
    y, lr, ebm, p1, p2 = (streams[k] for k in ("y", "lr", "ebm", "p1", "p2"))
    covered = p1 | p2
    cal = {k: compute_calibration(y, v) for k, v in (("lr", lr), ("ebm", ebm))}
    cal["glass"] = compute_calibration(y[covered], streams["gl"][covered])
    args = (cal["lr"], cal["ebm"], cal["glass"], y, lr, ebm, p2.astype(int), T_LR, T_EBM)

    pure_brier = compute_hybrid_weights(*args, alpha=1.0, glass_covered_mask=covered)
    inv = 1 / (np.array([cal[k]["brier"] for k in ("lr", "ebm", "glass")]) + 1e-6)
    np.testing.assert_allclose([pure_brier[k] for k in ("lr", "ebm", "glass")], inv / inv.sum())

    pure_acc = compute_hybrid_weights(*args, alpha=0.0, glass_covered_mask=covered)
    acc = np.array([accuracy_score(y, lr >= T_LR), accuracy_score(y, ebm >= T_EBM),
                    accuracy_score(y[covered], p2[covered].astype(int))])
    np.testing.assert_allclose([pure_acc[k] for k in ("lr", "ebm", "glass")], acc / acc.sum())
    assert pure_acc["details"]["glass_accuracy_population"] == "covered_rows"

    hybrid = compute_hybrid_weights(*args, glass_covered_mask=covered)
    assert sum(hybrid[k] for k in ("lr", "ebm", "glass")) == pytest.approx(1.0)
    for k in ("lr", "ebm", "glass"):
        assert hybrid[k] == pytest.approx(0.5 * pure_brier[k] + 0.5 * pure_acc[k])

    pre_audit = compute_hybrid_weights(*args, glass_covered_mask=None)
    assert pre_audit["details"]["accuracy_raw"][2] == pytest.approx(accuracy_score(y, p2.astype(int)))


def test_calibration_diagnostic_is_the_shared_ece_and_brier(streams):
    y, p = streams["y"], streams["lr"].copy()
    p[:10], p[10:20] = 0.0, 1.0                            # isotonic outputs hit both ends
    out = compute_calibration(y, p)
    assert out["ece"] == pytest.approx(calculate_ece(y, p, 10))
    assert out["brier"] == pytest.approx(brier_score_loss(y, p))


def test_abstention_cut_is_the_shared_stage4_rule_on_the_arbiters_own_confidence(streams, monkeypatch):
    """``tune_arbiter_threshold`` (Meta-EBM) and ``tune_min_confidence``
    (shared, Meta-XGB) pick the same cut from the arbiter's own confidence,
    searched exactly over its distinct values (F30); the arbiter is run once,
    without abstention."""
    s = streams
    calls = []

    def counted(*a, **k):
        calls.append(k)
        return meta_arbiter(*a, **k)

    best = tune_arbiter_threshold(counted, probs=(s["lr"], s["ebm"], s["gl"]),
                                  masks=(s["p1"], s["p2"]), y_true=s["y"],
                                  lr_thresh=T_LR, ebm_thresh=T_EBM, weights=WEIGHTS,
                                  glass_base_rate=s["b"])
    assert len(calls) == 1 and calls[0]["allow_abstain"] is False
    pred, _, conf = reference(s, allow_abstain=False)
    shared = tune_min_confidence(s["y"], pred, conf)
    assert best["search"] == shared["search"] == ABSTENTION_SEARCH
    assert best["min_weighted_confidence"] == pytest.approx(shared["min_confidence"], abs=1e-12)
    assert best["train_f2"] == pytest.approx(shared["train_f2"])
    assert best["train_coverage"] == pytest.approx(shared["train_coverage"])
    assert best["train_coverage"] >= 0.5
    # an explicit grid follows the same rule
    on_grid = tune_arbiter_threshold(meta_arbiter, (s["lr"], s["ebm"], s["gl"]), (s["p1"], s["p2"]),
                                     s["y"], T_LR, T_EBM, WEIGHTS, min_confs=ABSTENTION_GRID,
                                     glass_base_rate=s["b"])
    assert on_grid["min_weighted_confidence"] == pytest.approx(
        tune_min_confidence(s["y"], pred, conf, ABSTENTION_GRID)["min_confidence"])
    assert tune_arbiter_threshold(meta_arbiter, (s["lr"], s["ebm"], s["gl"]), (s["p1"], s["p2"]),
                                  s["y"], T_LR, T_EBM, WEIGHTS, min_coverage=1.01,
                                  glass_base_rate=s["b"]) is None


def test_the_selected_cut_reproduces_its_coverage_and_decisions_in_the_arbiter(streams):
    """The cut is one of the arbiter's own confidence values, so applying it
    abstains on exactly the rows below it: training coverage as selected,
    retained decisions unchanged (F30)."""
    s = streams
    args = (s["lr"], s["ebm"], s["gl"], s["p1"], s["p2"], T_LR, T_EBM, WEIGHTS)
    pred, prob, _, conf = meta_arbiter(*args, allow_abstain=False, return_confidence=True,
                                       glass_base_rate=s["b"])
    _, _, ref_conf = reference(s, allow_abstain=False)
    np.testing.assert_allclose(conf, ref_conf, rtol=0, atol=1e-12)
    best = tune_arbiter_threshold(meta_arbiter, (s["lr"], s["ebm"], s["gl"]), (s["p1"], s["p2"]),
                                  s["y"], T_LR, T_EBM, WEIGHTS, glass_base_rate=s["b"])
    cut = best["min_weighted_confidence"]
    assert cut in conf
    pred_a, prob_a, _ = meta_arbiter(*args, min_weighted_confidence=cut, glass_base_rate=s["b"])
    kept = pred_a != -1
    np.testing.assert_array_equal(kept, conf >= cut)
    assert kept.mean() == best["train_coverage"]
    np.testing.assert_array_equal(pred_a[kept], pred[kept])
    np.testing.assert_array_equal(prob_a, prob)
    assert fbeta_score(s["y"][kept], pred_a[kept], beta=2) == pytest.approx(best["train_f2"])


def test_disagreement_with_the_router_is_measured_where_it_votes(streams):
    s = streams
    covered = s["p1"] | s["p2"]
    lr_pred, ebm_pred = (s["lr"] >= T_LR).astype(int), (s["ebm"] >= T_EBM).astype(int)
    glass_pred = s["p2"].astype(int)
    base = analyze_disagreements(s["y"], lr_pred, ebm_pred, glass_pred, covered)
    flipped = np.where(covered, glass_pred, 1 - glass_pred)          # uncovered rows only
    assert analyze_disagreements(s["y"], lr_pred, ebm_pred, flipped, covered) == base
    assert base["lr_glass_disagree_rate"] == pytest.approx((lr_pred != glass_pred)[covered].mean())
    assert base["glass_coverage_rate"] == pytest.approx(covered.mean())


def test_abstention_metrics_use_retained_rows_only(streams):
    y = streams["y"]
    pred, prob, _ = arbiter(streams, min_weighted_confidence=0.15)
    out = evaluate_with_abstention(y, pred, prob)
    kept = pred != -1
    assert out["coverage"] == pytest.approx(kept.mean())
    assert out["metrics"] == compute_metrics(y[kept], pred[kept], prob[kept])
    none = evaluate_with_abstention(y, np.full(len(y), -1), prob)
    assert none["coverage"] == 0.0 and none["metrics"] is None
