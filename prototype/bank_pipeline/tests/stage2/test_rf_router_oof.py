"""
Black-box Stage 2 (RF Router): out-of-fold integrity, the Pass 2 remainder
and the outer cross-fit, on the notebook's Stage 2 path (synthetic bank frame,
80/20 split, 29 engineered bins, 5 folds, seed 42; tests.support.rf_router_world).

Contract (``RFRouter.fit``):
1. Pass 1 is scored out of fold on the shared 5-fold plan: every training row
   once, by a forest that never saw it.
2. The remainder is ``OOF p1 >= t1`` — decided by forests that never saw the
   row, not by the in-sample refit (F17: the RF arm's declared Pass 2
   population; the GLASS notebook profile's ``"full"`` is the cross-arm
   question left for Step 3).
3. Pass 2 is scored out of fold WITHIN the remainder and refitted on exactly
   the remainder; no Pass 2 forest ever sees a row Pass 1 routed.
4. The operating point realised on training OOF is the one the solver chose:
   the largest feasible Pass 1 band and the largest feasible Pass 2 band.
5. At predict time Pass 2 scores only what the refitted Pass 1 left.
6. No forest is ever fitted on a test row.

Contract (``crossfit_rf_router``, the Stage 4 train side): every outer fold's
router — encoder, forests, remainder and thresholds — is fitted on the other
folds only, so a held-out row's own label cannot move its outputs
(conditional on the fixed hyperparameters, F24); each fold router keeps the
same remainder discipline; fold plans that miss or repeat a row are refused.

The cheap ``predict_train_oof`` path is diagnostic: its Pass 1 scores are out
of fold, but ``t1`` and the remainder use every label (documented residual
dependence in ``router/two_pass.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import blackbox_pipeline.models.rf_router.crossfit as crossfit_mod
import blackbox_pipeline.models.rf_router.oof.scorer as scorer_mod
from blackbox_pipeline.models.rf_router import RFRouter, crossfit_rf_router, make_folds
from blackbox_pipeline.models.rf_router.evaluation import evaluate_router
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.router.base import apply_cuts
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.recorders import CallLog
from tests.support.rf_router_world import (
    N_STAGE_FOLDS, RANDOM_STATE, TEST_SIZE, bank_world, engineer, fit_ids, fold_segments,
    recording_forests, router_config, serial_forests,
)
from tests.support.stage2_synthetic import (
    ALT_PARAMS, N_FOLDS, ROUTER_SEED, SMALL_PARAMS, make_data,
)
from tests.support.stage2_synthetic import router_config as small_router_config


@pytest.fixture(scope="module")
def world():
    return bank_world()


def sub_log(log: CallLog, ids) -> CallLog:
    keep = set(ids)
    return CallLog(c for c in log if c.fit_id in keep)


def _roles(router, log):
    """(Pass 1 OOF fit ids, Pass 2 OOF fit ids, Pass 1 refit id, Pass 2 refit id)."""
    k1 = router.fit_report_.oof_summary["pass1"]["n_folds"]
    k2 = router.fit_report_.oof_summary["pass2"]["n_folds"]
    ids = fit_ids(log)
    assert len(ids) == k1 + k2 + 2, "fit sequence: Pass 1 OOF, Pass 2 OOF, two refits"
    return ids[:k1], ids[k1:k1 + k2], ids[k1 + k2], ids[k1 + k2 + 1]


def _rows(world, mask):
    return set(world.stage2["X_train"].index[np.asarray(mask)])


# ----------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------
def test_the_router_runs_the_stage2_protocol_on_the_global_split(world):
    gs, s2, r = world.global_split, world.stage2, world.router
    n_tr, n_te = len(gs["X_train"]), len(gs["X_test"])
    assert n_te / (n_tr + n_te) == pytest.approx(TEST_SIZE)                  # 80/20, untouched
    assert s2["X_train"].index.equals(gs["X_train"].index)
    assert s2["X_test"].index.equals(gs["X_test"].index)
    assert r.config.n_oof_folds == N_STAGE_FOLDS == r.fit_report_.oof_summary["pass1"]["n_folds"]
    assert r.fit_report_.oof_summary["pass1"]["random_state"] == RANDOM_STATE
    np.testing.assert_array_equal(r.oof_pass1_fold_id_, world.fold_id)       # notebook Cell 22
    assert r.split_fingerprint_ == s2["split_fingerprint"] == r.fit_report_.split_fingerprint


# ----------------------------------------------------------------------
# Pass 1, the remainder, Pass 2
# ----------------------------------------------------------------------
def test_pass1_scores_every_training_row_once_from_a_forest_that_never_saw_it(world):
    p1_ids, _, _, _ = _roles(world.router, world.router_log)
    train = world.stage2["X_train"].index
    assert_oof_provenance(sub_log(world.router_log, p1_ids), train, methods=("predict_proba",))
    for k, fid in enumerate(p1_ids):                           # fold ids are the folds used
        scored = {r for c in world.router_log.calls(fid, ["predict_proba"]) for r in c.keys}
        assert scored == _rows(world, world.router.oof_pass1_fold_id_ == k)


def test_the_remainder_is_decided_out_of_fold_not_in_sample_f17(world):
    r = world.router
    t1 = r.thresholds_.t1
    np.testing.assert_array_equal(r.remainder_mask_, r.oof_pass1_proba_ >= t1)
    in_sample = ForestTrainer.positive_proba(r.pass1_model_, world.stage2["X_train"]) >= t1
    assert not np.array_equal(in_sample, r.remainder_mask_)                  # the check can fail
    y = world.stage2["y_train"].to_numpy()
    rem = r.fit_report_.remainder
    assert (rem["n_rows"], rem["n_positives"]) == (r.remainder_mask_.sum(), y[r.remainder_mask_].sum())
    assert rem["source"] == "OOF pass1 probabilities >= t1"


def test_pass2_is_fitted_and_scored_only_within_the_remainder_f17(world):
    r, log = world.router, world.router_log
    _, p2_ids, refit1, refit2 = _roles(r, log)
    remainder, routed = _rows(world, r.remainder_mask_), _rows(world, ~r.remainder_mask_)
    p2_log = sub_log(log, p2_ids)
    assert_oof_provenance(p2_log, remainder, methods=("predict_proba",))      # each remainder row once
    assert_never_fit_on(p2_log, routed)
    assert not {k for c in p2_log for k in c.keys} & routed                   # nor scored a routed row
    assert np.isnan(r.oof_pass2_proba_[~r.remainder_mask_]).all()
    assert not np.isnan(r.oof_pass2_proba_[r.remainder_mask_]).any()
    fits = log.fits()
    assert fits[refit1] == set(world.stage2["X_train"].index) and fits[refit2] == remainder
    assert (r.pass1_model_._rec_fit_id, r.pass2_model_._rec_fit_id) == (refit1, refit2)


def test_no_stage2_forest_is_fitted_on_a_test_row(world):
    test_rows = world.stage2["X_test"].index
    for log in (world.router_log, world.crossfit_log):
        assert_never_fit_on(log, test_rows)
        assert not {k for c in log for k in c.keys} & set(test_rows)


def test_at_predict_time_pass2_scores_only_what_the_refitted_pass1_left(world):
    r, X_te = world.router, world.stage2["X_test"]
    log = CallLog()
    with recording_forests(log):
        _, _, decisions = r.predict(X_te)
    p1, p2 = r.last_scores_["p1"], r.last_scores_["p2"]
    runtime_remainder = set(X_te.index[p1 >= r.thresholds_.t1])
    by_model = {c.fit_id: set(c.keys) for c in log if c.method == "predict_proba"}
    assert by_model == {r.pass1_model_._rec_fit_id: set(X_te.index),
                        r.pass2_model_._rec_fit_id: runtime_remainder}
    assert np.isnan(p2[p1 < r.thresholds_.t1]).all()
    assert set(X_te.index[decisions != "pass1"]) == runtime_remainder
    np.testing.assert_array_equal(decisions, world.test_out[2])              # = the saved one-pass output


# ----------------------------------------------------------------------
# the operating point
# ----------------------------------------------------------------------
def test_the_realised_oof_operating_point_is_the_solved_one(world):
    r, y = world.router, world.stage2["y_train"]
    c, (pass1, pass2) = r.config.constraints, (r.thresholds_.pass1, r.thresholds_.pass2)
    m = evaluate_router(*r.predict_train_oof(), y, label="oof")
    assert (m["pass1_n"], m["pass1_subscribers_lost"]) == (pass1.band_size, pass1.positives_in_band)
    assert m["pass1_npv"] == pytest.approx(pass1.band_precision)
    assert m["pass1_npv"] == pytest.approx(r.bands_.pass1_confidence)
    assert m["pass1_leakage_rate"] == pytest.approx(pass1.leakage_rate)
    assert (m["pass2_n"], m["pass2_positives_found"]) == (pass2.band_size, pass2.positives_in_band)
    assert m["pass2_precision"] == pytest.approx(pass2.band_precision)
    assert m["pass2_precision"] == pytest.approx(r.bands_.pass2_confidence)
    assert m["pass2_recall"] == pytest.approx(pass2.recall)                   # global denominator
    assert m["pass1_npv"] >= c.min_precision_not_subscribe
    assert m["pass1_leakage_rate"] <= c.max_subscriber_leakage_rate
    assert m["pass2_precision"] >= c.min_precision_subscribe
    assert m["pass1_npv"] > 1 - m["split_base_rate"] and m["pass2_precision"] > m["split_base_rate"]
    for res in (pass1, pass2):                                                # the largest feasible band
        sweep = res.sweep
        assert not sweep.loc[sweep["band_size"] > res.band_size, "feasible"].any()
    assert {"pass1", "pass2", "uncertain"} == set(r.predict_train_oof()[2])


def test_the_cheap_train_path_is_out_of_fold_not_in_sample(world):
    r, X = world.router, world.stage2["X_train"]
    preds, conf, decisions = r.predict_train_oof()
    want_p, want_d = apply_cuts(r.oof_pass1_proba_, r.oof_pass2_proba_, r.thresholds_.t1, r.thresholds_.t2)
    np.testing.assert_array_equal(preds, want_p)
    np.testing.assert_array_equal(decisions, want_d)
    np.testing.assert_array_equal(conf, r.bands_.assign(decisions))
    assert not np.array_equal(decisions, r.predict(X)[2])                    # in-sample differs


def test_each_pass_trains_with_its_own_hyperparameters():
    """``pass2_rf_params`` specialises the Pass 2 forests (OOF and refit) only."""
    X, y = make_data()
    log, params = CallLog(), {}
    with serial_forests(), recording_forests(log, params):
        r = RFRouter(small_router_config(pass2_rf_params=ALT_PARAMS)).fit(X, y, "fp")
    p1_ids, p2_ids, refit1, refit2 = _roles(r, log)
    assert all(params[f] == SMALL_PARAMS for f in [*p1_ids, refit1])
    assert all(params[f] == ALT_PARAMS for f in [*p2_ids, refit2])
    clf1, clf2 = r.pass1_model_.named_steps["clf"], r.pass2_model_.named_steps["clf"]
    assert (clf2.max_depth, clf2.min_samples_leaf, clf2.max_features) == (2, 40, 0.5)
    assert (clf1.max_depth, clf1.min_samples_leaf, clf1.max_features) == (None, 10, "sqrt")


# ----------------------------------------------------------------------
# held-out labels
# ----------------------------------------------------------------------
def test_pass1_oof_ignores_a_folds_own_labels_on_pinned_folds(monkeypatch):
    """Stratified folds move with the labels, so the outer plan is pinned (as
    in the Stage 1 suite); within it, fold 0's Pass 1 scores cannot move."""
    X, y = make_data()
    plan = make_folds(y, N_FOLDS, ROUTER_SEED)
    real = scorer_mod.make_folds
    monkeypatch.setattr(scorer_mod, "make_folds",
                        lambda y_sub, **kw: plan if len(y_sub) == len(y) else real(y_sub, **kw))
    fold0 = np.zeros(len(y), bool)
    fold0[plan[0][1]] = True

    def fit(labels):
        with serial_forests():
            return RFRouter(small_router_config()).fit(X, labels, "fp")
    base = fit(y)
    moved = fit(mut.permute_labels(y, seed=3, rows=y.index[fold0][:80]))
    np.testing.assert_array_equal(moved.oof_pass1_proba_[fold0], base.oof_pass1_proba_[fold0])
    assert not np.array_equal(moved.oof_pass1_proba_[~fold0], base.oof_pass1_proba_[~fold0])


def test_crossfit_outputs_ignore_a_held_out_folds_own_labels(world):
    """The Stage 4 train side: fold 0 is routed by a router fitted on folds
    1-4, so its outputs cannot move when fold 0's labels do (hyperparameters
    fixed, F24)."""
    y, folds = world.stage2["y_train"], world.folds
    fold0 = np.zeros(len(y), bool)
    fold0[folds[0][1]] = True
    moved_y = mut.permute_labels(y, seed=1, rows=y.index[fold0][:60])
    with serial_forests():
        moved = crossfit_rf_router(world.global_split["X_train"], moved_y, folds, engineer=engineer,
                                   make_config=router_config, split_fp="fp", fold_fp="ff",
                                   random_state=RANDOM_STATE, verbose=False)
    for key in ("pred", "confidence", "decisions", "proba"):
        base, new = world.crossfit.oof[key], moved.oof[key]
        np.testing.assert_array_equal(new[fold0], base[fold0], err_msg=key)
        assert not np.array_equal(new[~fold0], base[~fold0]), key            # the check can fail


# ----------------------------------------------------------------------
# outer cross-fit, fold by fold
# ----------------------------------------------------------------------
def test_each_outer_fold_router_keeps_the_remainder_discipline_f17(world):
    index = world.stage2["X_train"].index
    segments = fold_segments(world.crossfit_log)
    assert len(segments) == len(world.folds) == N_STAGE_FOLDS
    decisions = world.crossfit.oof["decisions"]
    for k, (seg, (tr, va)) in enumerate(zip(segments, world.folds)):
        train_k, held_k = set(index[tr]), set(index[va])
        ids = fit_ids(seg)
        k1 = N_STAGE_FOLDS
        p1_ids, p2_ids, refit1, refit2 = ids[:k1], ids[k1:-2], ids[-2], ids[-1]
        assert_never_fit_on(seg, held_k)
        assert_oof_provenance(sub_log(seg, p1_ids), train_k, methods=("predict_proba",))
        fits = seg.fits()
        remainder_k = set().union(*(fits[f] for f in p2_ids))           # Pass 2's OOF population
        assert remainder_k < train_k and fits[refit2] == remainder_k and fits[refit1] == train_k
        assert_oof_provenance(sub_log(seg, p2_ids), remainder_k, methods=("predict_proba",))
        scored = {c.fit_id: set(c.keys) for c in seg if c.method == "predict_proba"
                  and c.fit_id in (refit1, refit2)}
        assert scored[refit1] == held_k, f"fold {k}"
        assert scored.get(refit2, set()) == set(index[va][decisions[va] != "pass1"]), f"fold {k}"


def test_crossfit_records_its_protocol(world):
    proto, folds = world.crossfit.protocol, world.crossfit.folds
    assert proto["train_outputs_oof"] is True and proto["n_folds"] == N_STAGE_FOLDS
    assert proto["fold_fingerprint"] == world.fold_fp and proto["random_state"] == RANDOM_STATE
    assert proto["pass2_population"] == "OOF Pass 1 remainder (RF router design)"
    assert proto["split_fingerprint"] == world.stage2["split_fingerprint"]
    assert list(folds["fold"]) == list(range(N_STAGE_FOLDS))
    y = world.stage2["y_train"].to_numpy()
    for k, (_, va) in enumerate(world.folds):
        p = world.crossfit.oof["pred"][va]
        assert folds.loc[k, "pass1"] == pytest.approx((p == 0).mean())
        assert folds.loc[k, "leak"] == pytest.approx(((p == 0) & (y[va] == 1)).sum() / (y[va] == 1).sum())


class _StubRouter:
    """Stands in for RFRouter inside crossfit_rf_router: routes everything to Pass 1."""

    def __init__(self, config):
        self.thresholds_ = SimpleNamespace(t1=0.5, t2=0.5)
        self.fit_report_ = SimpleNamespace(remainder={"fraction": 0.0})

    def fit(self, X, y, split_fp):
        return self

    def predict_with_proba(self, X):
        n = len(X)
        return (np.zeros(n, int), np.ones(n), np.array(["pass1"] * n, dtype=object),
                np.tile([1.0, 0.0], (n, 1)))


@pytest.mark.parametrize("defect, message", [
    ("missing_row", "do not partition"), ("repeated_row", "do not partition"),
    ("train_val_overlap", "overlap"),
])
def test_fold_plans_that_miss_or_repeat_rows_are_refused(monkeypatch, defect, message):
    monkeypatch.setattr(crossfit_mod, "RFRouter", _StubRouter)
    X, y = make_data(n=300)
    folds = [(tr.copy(), va.copy()) for tr, va in make_folds(y, N_FOLDS, ROUTER_SEED)]
    tr, va = folds[-1]
    if defect == "missing_row":
        folds[-1] = (tr, va[1:])
    elif defect == "repeated_row":
        folds[-1] = (tr[tr != folds[0][1][0]], np.append(va, folds[0][1][0]))
    else:
        folds[0] = (np.append(folds[0][0], folds[0][1][0]), folds[0][1])
    with pytest.raises(AssertionError, match=message):
        crossfit_rf_router(X, y, folds, engineer=lambda a, b, c: (a, b),
                           make_config=small_router_config, split_fp="s", fold_fp="f",
                           random_state=ROUTER_SEED, verbose=False)
