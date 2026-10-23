"""
Black-box Stage 4: the Meta-XGB estimator and configuration
(``blackbox_pipeline.models.meta_xgb``), on the Stage 4 feature block built
from black-box-format Stage 1-3 artifacts (tests.support.meta_xgb_world).

Contract: ``MetaXGBEstimator`` implements the shared ``Stage3Estimator``
interface; ``MetaXGBConfig`` is the shared Stage 3 protocol with exactly the
documented Stage 4 changes (stage, 50 trials, 10 pruner start-up trials, no
imputation, the six-column feature contract, the meta search space) and
refuses a different feature contract, operating space, abstain encoding,
imputation of the router's abstain-row NaN (F38) or identity (F37); Optuna
draws the declared space in a fixed order; ``normalize_params`` accepts
exactly the seven tuned keys as finite numbers in XGBoost's domain, so a
hyperparameter can no longer be recorded without being applied (F1); the
fitted model uses exactly those values plus the fixed conditions
(``colsample_bytree`` from the space, native missing values); the router's
abstain-row NaN is fitted and scored natively; an identical configuration
reproduces bit for bit, the thread count does not change a fit without row
subsampling, sample weights reach the fit; probabilities are P(class 1),
finite and in [0, 1]; scoring frames are matched by column name; a fitted
model survives serialisation; a tuned run records and applies the same
parameters, and the runner refuses bad supplied parameters before fitting.

What the shared runner does with the estimator (OOF, inner folds,
calibration, thresholds, label isolation, store) is covered for this
estimator in tests/stage3; the Stage 4 orchestration in
test_meta_xgb_stage.py.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import FloatDistribution, IntDistribution

import blackbox_pipeline.models.meta_xgb.estimator as estimator_mod
from blackbox_pipeline.models.meta_xgb import (
    STAGE4_FEATURES, MetaXGBConfig, MetaXGBEstimator, MetaXGBRunner, MetaXGBSearchSpace,
)
from shared.stage3.config import Stage3Config
from shared.stage3.data import PassthroughPipeline, Stage3Block
from shared.stage3.estimator import Stage3Estimator
from tests.support.meta_xgb_world import META_PARAMS, N_META_EVAL, blackbox_world, stage4_frames
from tests.support.recorders import CallLog, RecordingStage3Estimator

#: The seven tuned hyperparameters, in ``suggest`` order (spelled out: the
#: contract, not read back from the module under test).
TUNED = ("n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
         "reg_lambda", "reg_alpha")
SAMPLED = {**META_PARAMS, "subsample": 0.7}          # the row sampler draws


@pytest.fixture(scope="module")
def frames():
    """(X_train, y_train, X_test, y_test): the Meta-XGB block, abstain rows NaN."""
    w, arts = blackbox_world()
    inputs, X_tr, X_te = stage4_frames(w, arts)
    return (X_tr, pd.Series(inputs.y_train, index=inputs.index_train),
            X_te, pd.Series(inputs.y_test, index=inputs.index_test))


@pytest.fixture(scope="module")
def fitted(frames):
    X, y, _, _ = frames
    return MetaXGBEstimator(n_jobs=1).fit(META_PARAMS, X, y)


def _score(estimator, frames, params, sample_weight=None):
    X, y, X_te, _ = frames
    return estimator.positive_proba(estimator.fit(params, X, y, sample_weight=sample_weight), X_te)


# ----------------------------------------------------------------------
# interface and configuration
# ----------------------------------------------------------------------
def test_estimator_implements_the_shared_stage3_interface():
    est = MetaXGBEstimator()
    assert isinstance(est, Stage3Estimator)
    assert (est.family, est.as_array, est.random_state, est.n_jobs) == ("meta_xgb", False, 42, 1)


def test_the_tuned_keys_are_published_in_suggest_order():
    assert estimator_mod.META_XGB_TUNED_PARAM_NAMES == TUNED


def test_config_is_the_shared_protocol_with_exactly_the_stage4_changes():
    meta, shared = MetaXGBConfig().protocol(), Stage3Config().protocol()
    changed = {k for k in meta if meta[k] != shared[k]}
    assert changed == {"stage", "n_trials", "pruner_startup_trials", "impute_missing",
                       "expected_features", "search_space"}
    assert (meta["stage"], meta["n_trials"], meta["pruner_startup_trials"], meta["impute_missing"]) == (
        "stage4", 50, 10, False)
    assert meta["expected_features"] == list(STAGE4_FEATURES)
    assert meta["search_space"] == MetaXGBSearchSpace().to_dict()
    cfg = MetaXGBConfig()
    assert (cfg.model_family, cfg.arm, cfg.column_prefix, cfg.artifact_stem) == (
        "meta_xgb", "blackbox", "meta_xgb_", "meta_xgb_stage4")
    assert (cfg.n_tune_folds, cfg.n_eval_folds, cfg.random_state, cfg.n_jobs) == (5, 10, 42, 1)
    assert (cfg.operating_space, cfg.router_proba_on_abstain) == ("raw", "missing")


@pytest.mark.parametrize("override, message", [
    ({"expected_features": list(STAGE4_FEATURES)[::-1]}, "STAGE4_FEATURES"),
    ({"expected_features": list(STAGE4_FEATURES) + ["raw_age"]}, "STAGE4_FEATURES"),
    ({"expected_features": None}, "STAGE4_FEATURES"),
    ({"operating_space": "logit"}, "operating_space"),
    ({"router_proba_on_abstain": "zero"}, "router_proba_on_abstain"),
    ({"impute_missing": True}, "impute_missing must be False"),
    ({"model_family": "ebm", "arm": "glass"}, "identity"),
    ({"arm": "glass"}, "identity"),
    ({"stage": "stage3"}, "identity"),
], ids=["features_reordered", "extra_feature", "no_contract", "operating_space", "abstain_encoding",
        "imputation_f38", "glass_identity_f37", "arm_f37", "stage_f37"])
def test_config_refuses_anything_off_the_stage4_contract(override, message):
    """Before PR 36 the last four were accepted: imputation replaced the
    router's abstain-row NaN with the training median (F38), and the learner
    artifact took the overridden labels (F37)."""
    with pytest.raises(ValueError, match=message):
        MetaXGBConfig(**override)


def test_config_round_trips_through_its_json_form():
    cfg = MetaXGBConfig(n_trials=7, operating_space="calibrated",
                        search_space=MetaXGBSearchSpace(max_depth=(1, 2)))
    restored = MetaXGBConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.to_dict() == json.loads(json.dumps(cfg.to_dict()))
    assert isinstance(restored.search_space, MetaXGBSearchSpace)
    assert MetaXGBEstimator(restored.search_space).describe_space() == MetaXGBEstimator(
        cfg.search_space).describe_space()


def test_suggestions_follow_the_declared_space_in_order():
    est = MetaXGBEstimator(MetaXGBSearchSpace(max_depth=(1, 2)))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(lambda t: float(est.normalize_params(est.suggest(t))["learning_rate"]), n_trials=20)
    assert list(study.trials[0].params) == list(TUNED)
    assert study.trials[0].distributions == {
        "n_estimators": IntDistribution(50, 600, step=50),
        "learning_rate": FloatDistribution(0.01, 0.20, log=True),
        "max_depth": IntDistribution(1, 2),
        "min_child_weight": FloatDistribution(1.0, 50.0, log=True),
        "subsample": FloatDistribution(0.6, 1.0),
        "reg_lambda": FloatDistribution(0.1, 20.0, log=True),
        "reg_alpha": FloatDistribution(1e-3, 5.0, log=True),
    }
    for t in study.trials:                                    # every draw is a valid parameter set
        assert est.normalize_params(t.params) == t.params
    assert list(est.describe_space()) == list(TUNED) + ["colsample_bytree"]


# ----------------------------------------------------------------------
# hyperparameters (audit F1)
# ----------------------------------------------------------------------
def test_normalize_params_returns_exactly_the_tuned_keys_in_canonical_types():
    raw = {**{k: np.float64(v) for k, v in META_PARAMS.items()}, "n_estimators": np.int64(20),
           "max_depth": 2.0}
    out = MetaXGBEstimator().normalize_params(dict(reversed(list(raw.items()))))
    assert list(out) == list(TUNED) and out == META_PARAMS
    assert all(type(out[k]) is (int if k in ("n_estimators", "max_depth") else float) for k in out)


def _without(key):
    return {k: v for k, v in META_PARAMS.items() if k != key}


@pytest.mark.parametrize("params", [
    _without("max_depth"),                                    # was: XGBoost's default depth 6
    {**META_PARAMS, "max_dept": 4},                           # was: recorded, never applied
    {**_without("max_depth"), "max_dept": 4},
    {**META_PARAMS, "eta": 0.3}, {**META_PARAMS, "seed": 7}, {**META_PARAMS, "alpha": 1.0},
    {**META_PARAMS, "colsample_bytree": 0.5}, {**META_PARAMS, "missing": 0.0},
    {**META_PARAMS, "random_state": 7}, {**META_PARAMS, "objective": "binary:hinge"},
    {**META_PARAMS, "n_jobs": 4}, {**META_PARAMS, "tree_method": "exact"},
], ids=["missing", "unknown", "misspelled", "alias_eta", "alias_seed", "alias_alpha",
        "fixed_colsample", "fixed_missing", "fixed_random_state", "fixed_objective", "fixed_n_jobs",
        "fixed_tree_method"])
def test_missing_or_unknown_keys_are_refused_f1(params):
    with pytest.raises(KeyError):
        MetaXGBEstimator().normalize_params(params)


@pytest.mark.parametrize("key, value", [
    ("n_estimators", 0), ("n_estimators", -5), ("n_estimators", 20.7), ("n_estimators", "20"),
    ("max_depth", True), ("max_depth", -1), ("max_depth", 2.5),
    ("learning_rate", 0.0), ("learning_rate", float("nan")), ("learning_rate", math.inf),
    ("learning_rate", None), ("subsample", 0.0), ("subsample", 1.5),
    ("min_child_weight", -1.0), ("reg_lambda", -0.1), ("reg_alpha", -1.0),
    ("reg_alpha", np.bool_(True)),
], ids=lambda v: repr(v))
def test_values_outside_xgboost_domain_are_refused_f1(key, value):
    """XGBoost accepted 0 trees, a learning rate of 0 or NaN, subsample 0,
    depth True (as 1) and 20.7 trees (as 20) without a word."""
    with pytest.raises(ValueError, match=key):
        MetaXGBEstimator().normalize_params({**META_PARAMS, key: value})


@pytest.mark.parametrize("key, value", [
    ("max_depth", 0), ("min_child_weight", 0.0), ("reg_lambda", 0.0), ("reg_alpha", 0.0),
    ("subsample", 1.0), ("n_estimators", 1),
])
def test_values_at_xgboost_domain_edges_are_accepted(key, value):
    assert MetaXGBEstimator().normalize_params({**META_PARAMS, key: value})[key] == value


def test_fit_applies_exactly_the_normalized_params_and_the_fixed_conditions(fitted):
    got = fitted.get_params()
    assert {k: got[k] for k in TUNED} == META_PARAMS
    assert got["colsample_bytree"] == MetaXGBSearchSpace().colsample_bytree == 1.0
    assert math.isnan(got["missing"])
    assert (got["objective"], got["eval_metric"], got["tree_method"], got["random_state"],
            got["n_jobs"], got["verbosity"]) == ("binary:logistic", "logloss", "hist", 42, 1, 0)
    assert list(fitted.feature_names_in_) == list(STAGE4_FEATURES)


def test_fit_refuses_bad_params_before_training(frames):
    X, y, _, _ = frames
    for bad, error in (({**META_PARAMS, "max_dept": 4}, KeyError),
                       ({**META_PARAMS, "n_estimators": 0}, ValueError)):
        with pytest.raises(error):
            MetaXGBEstimator(n_jobs=1).fit(bad, X, y)


def test_the_runner_refuses_bad_supplied_params_before_anything_is_fitted(frames, tmp_path):
    X, y, X_te, y_te = frames
    cfg = MetaXGBConfig(n_eval_folds=N_META_EVAL, verbose=False, show_progress_bar=False,
                        artifact_dir=str(tmp_path))
    block = Stage3Block.build(X, y, X_te, y_te, PassthroughPipeline, n_eval_folds=N_META_EVAL,
                              impute_missing=False, expected_features=cfg.expected_features,
                              verbose=False)
    log = CallLog()
    runner = MetaXGBRunner(cfg, RecordingStage3Estimator(MetaXGBEstimator(n_jobs=1), log))
    with pytest.raises(KeyError, match="max_dept"):
        runner.fit(block, params={**META_PARAMS, "max_dept": 4}, params_source="typo")
    assert not log.fits() and runner.artifact_ is None


# ----------------------------------------------------------------------
# fitting and scoring
# ----------------------------------------------------------------------
def test_the_routers_abstain_rows_are_fitted_and_scored_natively(frames, fitted):
    """The router probability is NaN exactly where it abstains; nothing is
    imputed, and every row still gets a finite probability."""
    X, _, X_te, _ = frames
    assert X["stage2_proba"].isna().any() and X_te["stage2_proba"].isna().any()
    assert X.drop(columns="stage2_proba").notna().all().all()
    p = MetaXGBEstimator().positive_proba(fitted, X_te)
    assert np.isfinite(p).all()
    abstain = X_te["stage2_abstain"].to_numpy() == 1
    filled = X_te.assign(stage2_proba=X_te["stage2_proba"].fillna(0.5))     # a real 0.5 is not missing
    assert not np.allclose(MetaXGBEstimator().positive_proba(fitted, filled)[abstain], p[abstain])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_an_identical_configuration_reproduces_bit_for_bit(frames, n_jobs):
    base = _score(MetaXGBEstimator(n_jobs=n_jobs), frames, SAMPLED)
    np.testing.assert_array_equal(_score(MetaXGBEstimator(n_jobs=n_jobs), frames, SAMPLED), base)
    assert not np.allclose(_score(MetaXGBEstimator(random_state=7, n_jobs=n_jobs), frames, SAMPLED), base)


def test_the_thread_count_does_not_change_a_fit_without_row_subsampling(frames):
    """``colsample_bytree`` is fixed at 1.0, so with ``subsample=1.0`` no
    sampler draws. Row subsampling is left out: XGBoost's row sampler is
    thread-count independent only where the C++ library takes one engine
    value per Bernoulli trial (see test_blackbox_xgb_estimator.py), so Stage 4
    pins ``n_jobs`` like the seed (a protocol field, default 1)."""
    assert META_PARAMS["subsample"] == 1.0
    base = _score(MetaXGBEstimator(n_jobs=1), frames, META_PARAMS)
    np.testing.assert_array_equal(_score(MetaXGBEstimator(n_jobs=2), frames, META_PARAMS), base)
    assert MetaXGBConfig().protocol()["n_jobs"] == 1


def test_sample_weights_reach_the_fit(frames):
    y = frames[1]
    weights = np.where(y == 1, 4.0, 1.0)
    plain = _score(MetaXGBEstimator(n_jobs=1), frames, META_PARAMS)
    weighted = _score(MetaXGBEstimator(n_jobs=1), frames, META_PARAMS, weights)
    np.testing.assert_array_equal(_score(MetaXGBEstimator(n_jobs=1), frames, META_PARAMS, weights), weighted)
    assert not np.allclose(weighted, plain) and weighted.mean() > plain.mean()


def test_positive_proba_is_p_of_class_one_finite_and_bounded(fitted, frames):
    X_te = frames[2]
    p = MetaXGBEstimator().positive_proba(fitted, X_te)
    assert fitted.classes_.tolist() == [0, 1]
    np.testing.assert_array_equal(p, fitted.predict_proba(X_te)[:, 1])
    assert p.shape == (len(X_te),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_scoring_frames_are_matched_by_column_name(fitted, frames):
    X_te = frames[2]
    cols = list(X_te.columns)
    for frame in (X_te[cols[::-1]], X_te[cols[:-1]], X_te.assign(raw_age=0.0)):
        with pytest.raises(ValueError, match="feature_names mismatch"):
            MetaXGBEstimator().positive_proba(fitted, frame)


def test_a_fitted_model_survives_serialisation(fitted, frames):
    X_te = frames[2]
    buf = io.BytesIO()
    joblib.dump(fitted, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    est = MetaXGBEstimator()
    np.testing.assert_array_equal(est.positive_proba(restored, X_te), est.positive_proba(fitted, X_te))
    a, b = restored.get_params(), fitted.get_params()
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] or (isinstance(b[k], float) and math.isnan(b[k]) and math.isnan(a[k]))
               for k in b)


def test_estimator_records_everything_needed_to_rebuild_it():
    space = MetaXGBSearchSpace(max_depth=(1, 2))
    assert MetaXGBEstimator(space, random_state=7, n_jobs=3).to_dict() == {
        "family": "meta_xgb", "search_space": space.to_dict(), "random_state": 7, "n_jobs": 3,
        "fixed": {"objective": "binary:logistic", "tree_method": "hist",
                  "missing": "nan (native)", "eval_metric": "logloss"}}


# ----------------------------------------------------------------------
# tuning time vs final fit
# ----------------------------------------------------------------------
def test_a_tuned_run_records_and_applies_the_same_parameters(frames, tmp_path):
    X, y, X_te, y_te = frames
    space = MetaXGBSearchSpace(n_estimators=(20, 60, 20), max_depth=(1, 2))
    cfg = MetaXGBConfig(n_trials=3, n_eval_folds=N_META_EVAL, search_space=space, verbose=False,
                        show_progress_bar=False, artifact_dir=str(tmp_path))
    block = Stage3Block.build(X, y, X_te, y_te, PassthroughPipeline, n_eval_folds=N_META_EVAL,
                              impute_missing=False, expected_features=cfg.expected_features,
                              verbose=False)
    art = MetaXGBRunner(cfg, MetaXGBEstimator(space, cfg.random_state, 1)).fit(block)
    best = art.best_params
    assert list(best) == list(TUNED) and art.tuning["n_complete"] == 3
    assert best == MetaXGBEstimator(space).normalize_params(art.tuning["top_trials"][0]["params"])
    applied = art.refit["model"].get_params()
    assert {k: applied[k] for k in TUNED} == best
    assert art.config["search_space"] == dataclasses.asdict(space)
