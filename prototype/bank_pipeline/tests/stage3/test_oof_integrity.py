"""
Shared Stage 3 protocol: out-of-fold integrity, refit population and decisions,
checked from what each arm's models actually saw (recorded calls), for EBM,
XGBoost and the Meta-XGB learner alike. Includes the inner out-of-fold fits
that calibrate and threshold each outer fold (audit F22).

Per-fold feature fitting inside ``Stage3Block`` is covered in
tests/preprocessing/test_feature_contracts.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.stage_io import fold_assignment
from tests.stage3.conftest import N_EVAL, N_TUNE
from tests.support.checks import assert_never_fit_on, assert_oof_provenance
from tests.support.recorders import CallLog


def _plans(run):
    """Fold fits in run order: k-fold report (N_EVAL), OOF (N_TUNE), then the
    inner fits (N_TUNE per outer fold, F22)."""
    train = set(run.block.index_train)
    fold_fits = sorted(fid for fid, rows in run.log.fits().items() if rows < train)
    assert len(fold_fits) == N_EVAL + N_TUNE + N_TUNE * N_TUNE
    inner = fold_fits[N_EVAL + N_TUNE:]
    return {"report": fold_fits[:N_EVAL], "oof": fold_fits[N_EVAL:N_EVAL + N_TUNE],
            "inner": [inner[k * N_TUNE:(k + 1) * N_TUNE] for k in range(N_TUNE)]}


def test_every_training_row_is_scored_once_by_a_model_that_never_saw_it(arm_run):
    train_idx = arm_run.block.index_train
    plans = _plans(arm_run)
    for name in ("report", "oof"):
        sub_log = CallLog(c for c in arm_run.log if c.fit_id in set(plans[name]))
        assert_oof_provenance(sub_log, train_idx)


def test_inner_fits_stay_inside_their_outer_folds_training_rows(arm_run):
    """F22: fold k's inner models are fitted and scored on fold k's training
    rows only, each of which they score exactly once."""
    block, art = arm_run.block, arm_run.artifact
    index = block.index_train
    inner = np.asarray(art.oof["inner_proba_by_fold"])
    for k, ((tr, va), fits) in enumerate(zip(block.tune_plan, _plans(arm_run)["inner"])):
        sub_log = CallLog(c for c in arm_run.log if c.fit_id in set(fits))
        assert_oof_provenance(sub_log, index[tr])
        assert_never_fit_on(sub_log, index[va])
        assert not any(set(c.keys) & set(index[va]) for c in sub_log)    # nor scored
        assert np.isnan(inner[k, va]).all() and np.isfinite(inner[k, tr]).all()
    ids = np.asarray(art.oof["inner_fold_id_by_fold"])
    np.testing.assert_array_equal(ids, block.inner_fold_ids())
    assert ((ids == -1) == (np.arange(N_TUNE)[:, None] == block.tune_plan.fold_id[None, :])).all()


def test_no_model_is_ever_fitted_on_test_rows(arm_run):
    assert_never_fit_on(arm_run.log, arm_run.block.index_test)


def test_refit_uses_exactly_the_declared_training_population(arm_run):
    fits = arm_run.log.fits().values()
    assert sum(rows == set(arm_run.block.index_train) for rows in fits) == 1
    assert arm_run.artifact.population == arm_run.config.population == "full_train"


def test_oof_block_is_aligned_to_the_shared_fold_plan(arm_run):
    art, block = arm_run.artifact, arm_run.block
    assert list(art.index_train) == list(block.index_train)
    assert list(art.index_test) == list(block.index_test)
    np.testing.assert_array_equal(art.oof["fold_id"], block.tune_plan.fold_id)
    shared = fold_assignment(block.y_train, cv_folds=N_TUNE, random_state=arm_run.config.random_state)
    np.testing.assert_array_equal(art.oof["fold_id"], shared.to_numpy())
    np.testing.assert_array_equal(art.fold_plan["fold_id"], block.tune_plan.fold_id)


def test_all_probabilities_are_finite_and_in_unit_interval(arm_run):
    art = arm_run.artifact
    arrays = {
        "oof": art.oof["proba"], "oof_calibrated": art.oof["proba_calibrated"],
        "refit_test": art.refit["test_proba"], "refit_test_calibrated": art.refit["test_proba_calibrated"],
        "fold_mean_test": art.fold_mean["test_proba"],
    }
    for name, p in arrays.items():
        p = np.asarray(p, dtype=float)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), name
    assert len(art.oof["proba"]) == len(art.index_train)
    assert len(art.refit["test_proba"]) == len(art.index_test)


def test_stored_decisions_reproduce_from_probabilities_and_thresholds(arm_run):
    art = arm_run.artifact
    t_train = np.asarray(art.oof["threshold_per_row"], dtype=float)
    np.testing.assert_array_equal(art.oof["decision"],
                                  (np.asarray(art.oof["proba"]) >= t_train).astype(int))
    t = art.threshold["oof"]["threshold"]
    np.testing.assert_array_equal(art.refit["test_decision"],
                                  (np.asarray(art.refit["test_proba"]) >= t).astype(int))
    np.testing.assert_array_equal(art.fold_mean["test_decision"],
                                  (np.asarray(art.fold_mean["test_proba"]) >= t).astype(int))


def test_training_decisions_use_fold_nested_thresholds(arm_run):
    """With nested_oof_threshold=True each fold's rows get that fold's threshold."""
    art = arm_run.artifact
    assert art.threshold["oof_nested"]["drives_train_decisions"] is True
    per_row = pd.Series(art.oof["threshold_per_row"])
    by_fold = per_row.groupby(np.asarray(art.oof["fold_id"])).unique()
    assert all(len(v) == 1 for v in by_fold)
    assert [float(v[0]) for v in by_fold] == art.threshold["oof_nested"]["by_fold"]
