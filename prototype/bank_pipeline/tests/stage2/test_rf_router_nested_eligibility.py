"""
RF Router Pass 2 eligibility: the historical global ``t1`` and the opt-in
fold-nested cut (PR 37 Commit 3, F17; ``oof.eligibility``).

Contract
1. Default (``eligibility="global_t1"``) is the historical rule, unchanged:
   remainder = OOF ``p1 >= t1`` with one ``t1`` from every training label, and
   the fit report gains no keys.
2. ``"nested_t1"``: each inner fold's rows are cut by a ``t1_j`` solved only
   from the other folds' OOF scores (forests that never saw fold j). Changing
   fold j's labels cannot move fold j's eligibility. Under the global rule it
   can, which the control check shows.
3. The deployed ``t1`` is the same under both rules.
4. No fallback: an infeasible fold cut raises; an inconsistent fold plan or
   unscored row is refused; ``resolve_at`` is refused under the nested rule.
5. The outer cross-fit records the rule and each fold's realised remainder, and
   fold k's outputs ignore fold k's own labels under the nested rule too.
6. Band accounting (PR 37 F31): under the nested rule the Pass 1 band is the
   global rule's band, and the Pass 2 band is measured on the fold-nested
   remainder. ``predict_train_oof`` departs from that remainder only on the two
   boundary strips: strip A (``t1_j <= p1_oof < t1``) comes out as Pass 1,
   strip B (``t1 <= p1_oof < t1_j``) abstains (F31 C2, C3; W3, W4).

Label perturbations use unstratified inner folds (``stratify_oof=False``): a
stratified splitter reassigns folds when labels move, which would perturb the
fold plan rather than test the cut.

Scope: the synthetic features here are fixed binaries, not target-encoded, so
these tests establish ROUTER-level isolation only. In the real pipeline the
features are engineered on the whole outer-training partition with its labels
(PR 37 F17 limitation L1); PR 38 gate G13 audits that end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.rf_router import (
    OperatingConstraints, RFRouter, RFRouterConfig, ThresholdSolver, crossfit_rf_router, make_folds,
)
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.oof import OOFScorer
from blackbox_pipeline.models.rf_router.oof.eligibility import (
    ELIGIBILITY_GLOBAL_T1, ELIGIBILITY_NESTED_T1, NESTED_SOURCE, nested_pass1_eligibility,
)
from tests.support.recorders import CallLog
from tests.support.rf_router_world import recording_forests, serial_forests
from tests.support.synthetic_rf_router import CONSTRAINTS_OK, SMALL_PARAMS, make_data, router_config

N_FOLDS, E_FOLDS, SEED = 3, 3, 7

#: Looser floors for the outer cross-fit: each fold router sees 2/3 of the rows.
LOOSE = OperatingConstraints(min_precision_not_subscribe=0.80, max_subscriber_leakage_rate=0.30,
                             max_subscriber_leakage_absolute=10_000, min_precision_subscribe=0.50,
                             min_band_support_pass1=30, min_band_support_pass2=20)


def _fit(X, y, **overrides):
    # Serial forests: parallel predict_proba sums trees in a nondeterministic
    # order, so bit-level comparisons need n_jobs=1 (as the other suites do).
    with serial_forests():
        return RFRouter(router_config(**overrides)).fit(X, y, "fp")


@pytest.fixture(scope="module")
def data():
    return make_data(n=1500, seed=0)


@pytest.fixture(scope="module")
def routers(data):
    X, y = data
    kw = dict(stratify_oof=False, eligibility_folds=E_FOLDS)
    return {
        "default": _fit(X, y),
        "global": _fit(X, y, stratify_oof=False),
        "nested": _fit(X, y, eligibility=ELIGIBILITY_NESTED_T1, **kw),
    }


# ----------------------------------------------------------------------
# 1. the historical default
# ----------------------------------------------------------------------
def test_the_default_is_the_historical_global_cut_with_an_unchanged_report(routers):
    r = routers["default"]
    assert r.config.eligibility == ELIGIBILITY_GLOBAL_T1
    np.testing.assert_array_equal(r.remainder_mask_, r.oof_pass1_proba_ >= r.thresholds_.t1)
    assert set(r.fit_report_.remainder) == {"n_rows", "fraction", "n_positives", "base_rate", "source"}
    assert r.fit_report_.remainder["source"] == "OOF pass1 probabilities >= t1"


def test_an_explicit_global_rule_reproduces_the_default_exactly(data):
    X, y = data
    a, b = _fit(X, y), _fit(X, y, eligibility=ELIGIBILITY_GLOBAL_T1)
    np.testing.assert_array_equal(a.remainder_mask_, b.remainder_mask_)
    assert a.thresholds_.to_dict() == b.thresholds_.to_dict() and a.bands_.to_dict() == b.bands_.to_dict()
    for x, z in zip(a.predict(X), b.predict(X)):
        np.testing.assert_array_equal(x, z)


def test_old_configs_without_the_field_load_as_the_historical_rule():
    d = router_config().to_dict()
    del d["eligibility"], d["eligibility_folds"]
    assert RFRouterConfig.from_dict(d).eligibility == ELIGIBILITY_GLOBAL_T1


# ----------------------------------------------------------------------
# 2. the nested cut
# ----------------------------------------------------------------------
def test_the_nested_cut_matches_an_independent_recomputation(data, routers):
    X, y = data
    r = routers["nested"]
    fold = r.oof_pass1_fold_id_
    solver = ThresholdSolver(CONSTRAINTS_OK, verbose=False)
    scorer = OOFScorer(ForestTrainer(SMALL_PARAMS, SEED), E_FOLDS, SEED, stratify=False)
    want, want_t1 = np.zeros(len(y), bool), []
    for j in np.unique(fold):
        sub = scorer.score(X, y, subset_mask=fold != j)
        t1_j = solver.solve_pass1(sub.scored_proba(), sub.scored_labels(y), keep_sweep=False).threshold
        want[fold == j] = r.oof_pass1_proba_[fold == j] >= t1_j
        want_t1.append(t1_j)
    np.testing.assert_array_equal(r.remainder_mask_, want)
    rep = r.fit_report_.remainder
    assert rep["eligibility_fold_t1"] == pytest.approx(want_t1)
    assert rep["source"] == NESTED_SOURCE and rep["eligibility"] == ELIGIBILITY_NESTED_T1
    assert rep["n_rows"] == int(want.sum()) and rep["n_positives"] == int(np.asarray(y)[want].sum())
    assert sum(rep["eligibility_fold_remainder"]) == rep["n_rows"]


def test_no_eligibility_forest_is_fitted_on_the_fold_it_cuts(data):
    X, y = data
    log = CallLog()
    with recording_forests(log):
        r = _fit(X, y, eligibility=ELIGIBILITY_NESTED_T1, stratify_oof=False, eligibility_folds=E_FOLDS)
    fold = r.oof_pass1_fold_id_
    fits = [set(c.keys) for c in log if c.method == "fit"]
    elig = fits[N_FOLDS:N_FOLDS + N_FOLDS * E_FOLDS]                   # after the Pass 1 OOF fits
    assert len(elig) == N_FOLDS * E_FOLDS
    for j in range(N_FOLDS):
        held = set(X.index[fold == j])
        for rows in elig[j * E_FOLDS:(j + 1) * E_FOLDS]:
            assert not rows & held, f"an eligibility forest for fold {j} saw fold {j}"
            assert rows < set(X.index[fold != j])


def test_the_deployed_t1_is_the_same_under_both_rules(routers):
    g, n = routers["global"], routers["nested"]
    assert g.thresholds_.t1 == n.thresholds_.t1
    np.testing.assert_array_equal(g.oof_pass1_proba_, n.oof_pass1_proba_)


def _flip_negatives_below_the_cut(r, y, fold_j, k=10):
    """Make fold j's negatives just below t1 positive: the band's NPV drops, t1 moves."""
    fold, p, t1 = r.oof_pass1_fold_id_, r.oof_pass1_proba_, r.thresholds_.t1
    cand = np.flatnonzero((fold == fold_j) & (np.asarray(y) == 0) & (p < t1))
    pick = cand[np.argsort(-p[cand])[:k]]
    y2 = y.copy()
    y2.iloc[pick] = 1
    return y2


def test_a_folds_own_labels_cannot_move_its_eligibility_under_the_nested_rule(data, routers):
    X, y = data
    r = routers["nested"]
    j0 = r.oof_pass1_fold_id_ == 0
    y2 = _flip_negatives_below_the_cut(r, y, 0)
    moved = _fit(X, y2, eligibility=ELIGIBILITY_NESTED_T1, stratify_oof=False, eligibility_folds=E_FOLDS)
    np.testing.assert_array_equal(moved.oof_pass1_fold_id_, r.oof_pass1_fold_id_)     # same plan
    np.testing.assert_array_equal(moved.remainder_mask_[j0], r.remainder_mask_[j0])
    assert moved.fit_report_.remainder["eligibility_fold_t1"][0] == r.fit_report_.remainder["eligibility_fold_t1"][0]


def test_control_the_global_rule_lets_those_labels_move_the_same_rows(data, routers):
    X, y = data
    r = routers["global"]
    j0 = r.oof_pass1_fold_id_ == 0
    moved = _fit(X, _flip_negatives_below_the_cut(r, y, 0), stratify_oof=False)
    assert moved.thresholds_.t1 != r.thresholds_.t1
    assert not np.array_equal(moved.remainder_mask_[j0], r.remainder_mask_[j0])


def test_the_nested_rule_is_deterministic(data, routers):
    X, y = data
    again = _fit(X, y, eligibility=ELIGIBILITY_NESTED_T1, stratify_oof=False, eligibility_folds=E_FOLDS)
    np.testing.assert_array_equal(again.remainder_mask_, routers["nested"].remainder_mask_)
    assert again.fit_report_.remainder == routers["nested"].fit_report_.remainder


# ----------------------------------------------------------------------
# 4. refusals, no fallback
# ----------------------------------------------------------------------
def test_an_infeasible_fold_cut_raises_instead_of_falling_back(data, routers):
    X, y = data
    r = routers["nested"]
    impossible = ThresholdSolver(OperatingConstraints(
        min_precision_not_subscribe=1.0, max_subscriber_leakage_rate=0.0,
        max_subscriber_leakage_absolute=0, min_band_support_pass1=len(y)), verbose=False)
    with pytest.raises(ValueError, match="nested eligibility, inner fold 0"):
        nested_pass1_eligibility(X, y, pass1_proba=r.oof_pass1_proba_, fold_id=r.oof_pass1_fold_id_,
                                 trainer=ForestTrainer(SMALL_PARAMS, SEED), solver=impossible,
                                 n_folds=E_FOLDS, random_state=SEED, stratify=False)


@pytest.mark.parametrize("defect", ["unscored row", "one fold", "unassigned row", "wrong length"])
def test_inputs_that_break_the_fold_contract_are_refused(data, routers, defect):
    X, y = data
    r = routers["nested"]
    p1, fold = r.oof_pass1_proba_.copy(), r.oof_pass1_fold_id_.copy()
    if defect == "unscored row":
        p1[3] = np.nan
    elif defect == "one fold":
        fold[:] = 0
    elif defect == "unassigned row":
        fold[5] = -1
    else:
        p1 = p1[:-1]
    with pytest.raises(ValueError):
        nested_pass1_eligibility(X, y, pass1_proba=p1, fold_id=fold,
                                 trainer=ForestTrainer(SMALL_PARAMS, SEED),
                                 solver=ThresholdSolver(CONSTRAINTS_OK, verbose=False),
                                 n_folds=E_FOLDS, random_state=SEED)


def test_resolve_at_is_refused_under_the_nested_rule(routers):
    with pytest.raises(ValueError, match="need a refit"):
        routers["nested"].resolve_at(CONSTRAINTS_OK)


@pytest.mark.parametrize("kw, message", [
    (dict(eligibility="nested"), "eligibility must be one of"),
    (dict(eligibility=ELIGIBILITY_NESTED_T1, mode="single_rf"), "two_pass cascade only"),
    (dict(eligibility_folds=1), "eligibility_folds must be >= 2"),
])
def test_invalid_eligibility_settings_are_refused(kw, message):
    with pytest.raises(ValueError, match=message):
        router_config(**kw)


# ----------------------------------------------------------------------
# 5. the outer cross-fit
# ----------------------------------------------------------------------
def _identity(X_tr, X_va, y_tr):
    return X_tr, X_va


def _crossfit(X, y, folds, **overrides):
    kw = {"constraints": LOOSE, **overrides}
    with serial_forests():
        return crossfit_rf_router(X, y, folds, engineer=_identity,
                                  make_config=lambda: router_config(**kw), split_fp="fp",
                                  fold_fp="ff", random_state=SEED, verbose=False)


@pytest.fixture(scope="module")
def outer(data):
    X, y = data
    folds = make_folds(y, 3, SEED)
    return folds, _crossfit(X, y, folds, eligibility=ELIGIBILITY_NESTED_T1, eligibility_folds=E_FOLDS)


def test_the_crossfit_records_the_rule_and_each_folds_realised_remainder(data, outer):
    X, y = data
    folds, cf = outer
    assert cf.protocol["pass2_eligibility"] == ELIGIBILITY_NESTED_T1
    for k, (tr, _) in enumerate(folds):
        row = cf.folds.loc[k]
        assert row["remainder_rows"] == round(row["remainder_frac"] * len(tr))
        assert 0 < row["remainder_positives"] < row["remainder_rows"]
    default = _crossfit(X, y, folds)
    assert default.protocol["pass2_eligibility"] == ELIGIBILITY_GLOBAL_T1


def test_fold_routers_with_different_rules_are_refused(data):
    X, y = data
    configs = iter([router_config(constraints=LOOSE),
                    router_config(constraints=LOOSE, eligibility=ELIGIBILITY_NESTED_T1),
                    router_config(constraints=LOOSE)])
    with pytest.raises(AssertionError, match="different Pass 2 eligibility rules"):
        crossfit_rf_router(X, y, make_folds(y, 3, SEED), engineer=_identity,
                           make_config=lambda: next(configs), split_fp="fp", fold_fp="ff",
                           random_state=SEED, verbose=False)


def test_outer_fold_k_ignores_its_own_labels_under_the_nested_rule(data, outer):
    """Corrected V5: moving fold 0's labels may change the other folds (fold 0
    is in their training partitions) but never fold 0's own outputs."""
    X, y = data
    folds, cf = outer
    va0 = folds[0][1]
    y2 = y.copy()
    y2.iloc[va0[:25]] = 1 - y2.iloc[va0[:25]]
    moved = _crossfit(X, y2, folds, eligibility=ELIGIBILITY_NESTED_T1, eligibility_folds=E_FOLDS)
    for key in ("pred", "confidence", "decisions", "proba"):
        np.testing.assert_array_equal(moved.oof[key][va0], cf.oof[key][va0], err_msg=key)
    rest = np.setdiff1d(np.arange(len(y)), va0)
    assert not np.array_equal(moved.oof["proba"][rest], cf.oof["proba"][rest])  # the check can fail


# ----------------------------------------------------------------------
# 6. band accounting and the training-OOF diagnostic (PR 37 F31)
# ----------------------------------------------------------------------
def _strips(r):
    p1, t1, rem = r.oof_pass1_proba_, r.thresholds_.t1, r.remainder_mask_
    return rem & (p1 < t1), ~rem & (p1 >= t1)


def test_the_bands_are_measured_on_the_populations_the_f31_record_names(data, routers):
    _, y = data
    y = np.asarray(y)
    g, n = routers["global"], routers["nested"]
    assert n.bands_.pass1_confidence == g.bands_.pass1_confidence           # Pass 1 band unchanged
    assert n.bands_.pass1_support == g.bands_.pass1_support
    for r in (g, n):
        flagged = r.remainder_mask_ & (r.oof_pass2_proba_ > r.thresholds_.t2)
        assert np.isnan(r.oof_pass2_proba_[~r.remainder_mask_]).all()
        assert r.bands_.pass2_support == int(flagged.sum())
        assert r.bands_.pass2_confidence == pytest.approx((y[flagged] == 1).mean(), abs=1e-12)
    a, b = _strips(n)
    assert a.any() and b.any(), "the fixture must exercise both boundary strips"
    assert not any(_strips(g)[0]) and not any(_strips(g)[1])                # global: a partition


def test_predict_train_oof_departs_from_the_remainder_only_on_the_boundary_strips(routers):
    r = routers["nested"]
    p1, t1, rem = r.oof_pass1_proba_, r.thresholds_.t1, r.remainder_mask_
    p2, t2 = r.oof_pass2_proba_, r.thresholds_.t2
    a, b = _strips(r)
    decisions = r.predict_train_oof()[2]
    assert (decisions[a] == "pass1").all()                                   # in the remainder, routed to Pass 1
    assert (decisions[b] == "uncertain").all()                               # eligible at t1, no p2: abstains
    inside = rem & ~a
    want = np.where(p2[inside] > t2, "pass2", "uncertain")
    np.testing.assert_array_equal(decisions[inside], want)
    assert (decisions[~rem & ~b] == "pass1").all()
