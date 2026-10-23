"""
GLASS Stage 2 cross-fitting (glass_pipeline.glass_router.pipeline.crossfit).

Contract: per fold, feature engineering, the RF and the GLASS Router are
fitted on that fold's training rows only and score only its held-out rows;
every training row gets exactly one OOF output, placed back in training-row
order; the fold plan is the shared one; a fold's own labels never reach its
OOF outputs; and malformed inputs are refused before anything is fitted.

The router is a recording stub (see conftest); feature engineering and the RF
are real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glass_pipeline.glass_router.core.rule import ABSTAIN
from glass_pipeline.glass_router.pipeline.crossfit import fold_fingerprint, make_folds
from shared.stage2 import RF_FEATURES_BINARY
from shared.stage3.config import Stage3Config
from shared.stage3.data import FoldPlan
from shared.stage_io import fold_assignment
from tests.stage2.conftest import TINY_RF, run_crossfit, stub_proba, stub_route
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.synthetic import make_bank_frame, split_frame


def _split():
    return split_frame(make_bank_frame(n=400))


@pytest.fixture(scope="module")
def split():
    return _split()


@pytest.fixture(scope="module")
def run(split):
    """Default fold plan (folds=None), as the GLASS notebook runs it."""
    return run_crossfit(split)


def _plan(y, k=3, seed=1):
    return make_folds(y, n_splits=k, random_state=seed)


# ----------------------------------------------------------------------
# per-fold provenance
# ----------------------------------------------------------------------
def test_feature_engineering_is_refit_on_each_folds_training_rows(run, split):
    train = split["X_train"].index
    assert_oof_provenance(run.fe_log, train, methods=("transform",))
    assert_never_fit_on(run.fe_log, split["X_test"].index)
    assert all(f["columns"] == list(split["X_train"].columns) for f in run.fe_inputs)  # raw input


def test_rf_and_router_are_fitted_per_fold_and_score_only_held_out_rows(run, split):
    train = split["X_train"].index
    assert_oof_provenance(run.rf_log, train)
    assert_oof_provenance(run.router_log, train, methods=("predict",))
    assert_oof_provenance(run.router_log, train, methods=("predict_proba",))
    for log in (run.rf_log, run.router_log):
        assert_never_fit_on(log, split["X_test"].index)


def test_each_router_is_given_the_rf_fitted_on_its_own_fold(run):
    assert len(run.routers) == len(run.pipes) == 5
    rf_fits, router_fits = run.rf_log.fits(), run.router_log.fits()
    for router, pipe in zip(run.routers, run.pipes):
        assert router.rf_model is pipe.named_steps["clf"]
        assert rf_fits[pipe.fit_id] == router_fits[router.fit_id]


def test_rf_sample_weights_follow_the_tuned_minority_weight(run, split):
    for pipe in run.pipes:
        y_fold = split["y_train"].loc[list(run.rf_log.fits()[pipe.fit_id])]
        weights = sorted(set(np.asarray(pipe.sample_weight)))
        assert weights == [1.0, TINY_RF["minority_weight"]] and len(pipe.sample_weight) == len(y_fold)


def test_glass_rf_model_none_gives_the_router_no_rf(split):
    run = run_crossfit(split, folds=_plan(split["y_train"], k=2), glass_rf_model="none")
    assert [r.rf_model for r in run.routers] == [None, None]
    assert run.oof.protocol["glass_rf_model"] == "none"


# ----------------------------------------------------------------------
# assembly of the OOF block
# ----------------------------------------------------------------------
def test_oof_block_holds_each_fold_models_output_in_training_row_order(run, split):
    oof, index = run.oof, split["X_train"].index
    assert oof.index.equals(index)
    pred, conf, dec = stub_route(index)
    np.testing.assert_array_equal(oof.glass_pred, pred)
    np.testing.assert_array_equal(oof.glass_confidence, conf)
    np.testing.assert_array_equal(oof.glass_decisions, dec)
    for k, (pipe, router, fe) in enumerate(zip(run.pipes, run.routers, run.fe_inputs)):
        va = np.flatnonzero(oof.fold_id == k)
        assert list(fe["val"].index) == list(index[va])                # positional order kept
        np.testing.assert_array_equal(oof.glass_proba[va],
                                      stub_proba(index[va], router.training_base_rate))
        np.testing.assert_array_equal(oof.rf_proba[va], pipe.pipe.predict_proba(fe["val"])[:, 1])


def test_frame_and_output_views_are_aligned_and_consistent(run):
    oof = run.oof
    frame = oof.to_frame()
    assert frame.index.equals(oof.index)
    np.testing.assert_array_equal(frame["glass_p_subscribe"], oof.glass_proba[:, 1])
    np.testing.assert_array_equal(frame["rf_p_subscribe"], oof.rf_proba)
    out = oof.glass_train_out()
    assert ((out["covered"] ^ out["abstained"]).all())                 # exclusive and exhaustive
    np.testing.assert_array_equal(out["abstained"], oof.glass_pred == ABSTAIN)
    np.testing.assert_array_equal(out["abstained"], oof.glass_decisions == "uncertain")
    assert np.isfinite(oof.rf_proba).all() and ((oof.rf_proba >= 0) & (oof.rf_proba <= 1)).all()
    np.testing.assert_allclose(oof.glass_proba.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# fold plan and protocol
# ----------------------------------------------------------------------
def test_default_fold_plan_is_the_shared_stage3_tuning_plan(run, split):
    y = split["y_train"]
    shared = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    np.testing.assert_array_equal(run.oof.fold_id, shared)
    cfg = Stage3Config()
    stage3 = FoldPlan.build(y, n_splits=cfg.n_tune_folds, random_state=cfg.random_state)
    np.testing.assert_array_equal(run.oof.fold_id, stage3.fold_id)
    assert run.oof.protocol["fold_fingerprint"] == fold_fingerprint(shared)
    assert run.oof.protocol["fold_source"].startswith("default StratifiedKFold(n_splits=5")


def test_a_provided_fold_plan_is_used_as_given(split):
    y = split["y_train"]
    folds = _plan(y, k=3, seed=9)
    run = run_crossfit(split, folds=folds)
    expected = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(folds):
        expected[va] = k
    np.testing.assert_array_equal(run.oof.fold_id, expected)
    assert run.oof.protocol["fold_source"] == "provided" and run.oof.protocol["n_folds"] == 3


def test_protocol_records_how_the_block_was_made(run):
    p = run.oof.protocol
    assert p["train_outputs_oof"] is True and p["n_folds"] == 5 and p["random_state"] == 42
    assert p["glass_pass2_population"] == "full"                      # the active legacy profile
    assert p["glass_abstain_label"] == "uncertain" and p["glass_rf_model"] == "fold_rf"
    assert "full-train refit" in p["test_outputs"]
    assert [s["fold"] for s in run.oof.fold_summaries] == list(range(5))
    assert sum(s["n_val"] for s in run.oof.fold_summaries) == len(run.oof.index)


def test_router_inputs_follow_the_engineered_binary_schema(run, split):
    for router, fe in zip(run.routers, run.fe_inputs):
        for X in (router.fit_X, *router.seen):
            assert list(X.columns) == list(RF_FEATURES_BINARY)
            assert set(np.unique(X.to_numpy())) <= {0, 1}
        pd.testing.assert_frame_equal(router.seen[0], fe["val"])


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def test_fold_labels_never_reach_their_own_oof_outputs(split):
    y = split["y_train"]
    folds = _plan(y)
    fold0 = y.index[folds[0][1]]
    runs = {
        "base": run_crossfit(split, folds=folds),
        "own": run_crossfit(split, folds=folds, y_train=mut.permute_labels(y, seed=3, rows=fold0)),
        "others": run_crossfit(split, folds=folds, y_train=mut.permute_labels(
            y, seed=3, rows=y.index.difference(fold0))),
    }
    va0 = folds[0][1]
    base, own, others = (runs[k] for k in ("base", "own", "others"))
    np.testing.assert_array_equal(own.oof.rf_proba[va0], base.oof.rf_proba[va0])
    pd.testing.assert_frame_equal(own.routers[0].seen[0], base.routers[0].seen[0])
    pd.testing.assert_frame_equal(own.routers[0].fit_X, base.routers[0].fit_X)
    assert not np.array_equal(others.oof.rf_proba[va0], base.oof.rf_proba[va0])  # can fail


# ----------------------------------------------------------------------
# refused inputs (all raise before anything is fitted)
# ----------------------------------------------------------------------
def _bad_folds(y, kind):
    folds = [(np.asarray(tr), np.asarray(va)) for tr, va in _plan(y)]
    (tr0, va0), (tr1, va1) = folds[0], folds[1]
    if kind == "overlap":
        folds[0] = (np.append(tr0, va0[0]), va0)
    elif kind == "out_of_range":
        folds[0] = (tr0, np.append(va0, len(y)))
    elif kind == "held_out_twice":
        folds[1] = (tr1[tr1 != va0[0]], np.append(va1, va0[0]))
    else:
        folds = folds[1:]
    return folds


@pytest.mark.parametrize("kind, message", [
    ("overlap", "overlap"),
    ("out_of_range", "out of range"),
    ("held_out_twice", "more than one fold"),
    ("never_held_out", "never held out"),
])
def test_malformed_fold_plans_are_refused(split, kind, message):
    with pytest.raises(ValueError, match=message):
        run_crossfit(split, folds=_bad_folds(split["y_train"], kind))


@pytest.mark.parametrize("case, message", [
    ("shuffled_labels", "same index"),
    ("foreign_label", "same index"),
    ("bad_rf_mode", "glass_rf_model"),
])
def test_misaligned_or_invalid_arguments_are_refused(split, case, message):
    y, kwargs = split["y_train"], {}
    if case == "shuffled_labels":
        kwargs["y_train"] = mut.shuffle_rows(y)
    elif case == "foreign_label":
        kwargs["y_train"] = mut.replace_index_label(y)
    else:
        kwargs["glass_rf_model"] = "full_train_rf"
    with pytest.raises(ValueError, match=message):
        run_crossfit(split, **kwargs)
