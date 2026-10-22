"""
Black-box Stage 3: the XGBoost estimator and configuration on the shared
Stage 3 protocol (``blackbox_pipeline.models.xgb.shared_protocol`` — the
implementation ``blackbox_cascade`` imports; the package's standalone modules
are legacy, removed in Commit 4, and not tested here).

Contract: ``XGBEstimator`` implements the shared ``Stage3Estimator`` interface
on DataFrames; ``XGBStage3Config`` carries the black-box XGBoost identity and
otherwise IS the GLASS EBM protocol (5 tuning folds, 10 reporting folds, 150
trials, the same objective, calibration and threshold rule); Optuna draws the
declared space in a fixed order; ``normalize_params`` accepts exactly the tuned
keys as finite numbers in XGBoost's domain, so a hyperparameter can no longer
be recorded without being applied (F1); the fitted model uses exactly those
values plus the fixed conditions; an identical configuration reproduces bit
for bit (both samplers drawing), the thread count does not change a fit without
row subsampling and is pinned by the protocol, and sample weights reach the
fit; scoring frames are matched by column name; a tuned run records and applies
the same parameters.

Everything the runner does with the estimator (OOF, inner folds, calibration,
thresholds, label isolation, store and reuse) is covered for this arm in the
shared Stage 3 modules; the notebook path and the artifact in
test_blackbox_xgb_stage.py.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution

import blackbox_pipeline.models.xgb as xgb_pkg
import blackbox_pipeline.models.xgb.shared_protocol as protocol_mod
from blackbox_pipeline.models.xgb.shared_protocol import (
    XGBEstimator, XGBSearchSpace, XGBStage3Config, train_xgb_stage,
)
from glass_pipeline.ebm.config import EBMStage3Config
from shared.stage3.config import PROTOCOL_FIELDS
from shared.stage3.data import Stage3Block
from shared.stage3.estimator import Stage3Estimator
from shared.stage3.runner import Stage3Runner
from shared.stage_io import split_fingerprint
from tests.stage3.conftest import ARMS, N_EVAL, N_TUNE, Passthrough, stage3_split
from tests.support.notebooks import BANK_ROOT
from tests.support.recorders import CallLog, RecordingStage3Estimator

PARAMS = dict(ARMS["xgboost"].params)                 # small, valid, exactly the tuned keys
#: The seven tuned hyperparameters, in search-space order (spelled out: the
#: contract, not read back from the module under test).
TUNED = ("n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
         "colsample_bytree", "reg_lambda")
NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"


@pytest.fixture(scope="module")
def frames(ebm_features):
    """The engineered EBM block features XGBoost consumes, with labels."""
    split, _, X_tr, X_te = ebm_features
    return X_tr, split["y_train"], X_te


@pytest.fixture(scope="module")
def fitted(frames):
    X, y, _ = frames
    return XGBEstimator(n_jobs=1).fit(PARAMS, X, y)


def _passthrough_block(cfg):
    split = stage3_split()
    fp = split_fingerprint(split["X_train"].index, split["X_test"].index,
                           split["y_train"].astype(int), split["y_test"].astype(int))
    return Stage3Block.from_global_split(split, Passthrough, cfg, split_fingerprint=fp)


def _small_config(tmp_path, **overrides) -> XGBStage3Config:
    return XGBStage3Config(**{
        "n_tune_folds": N_TUNE, "n_eval_folds": N_EVAL, "n_jobs": 1, "verbose": False,
        "show_progress_bar": False, "artifact_dir": str(tmp_path),
        "expected_features": list(stage3_split()["X_train"].columns), **overrides})


# ----------------------------------------------------------------------
# interface, identity, protocol
# ----------------------------------------------------------------------
def test_the_notebook_imports_the_shared_protocol_implementation():
    for name in ("train_xgb_stage", "XGBStage3Config", "XGBSearchSpace", "XGBEstimator"):
        assert getattr(xgb_pkg, name) is getattr(protocol_mod, name), name


def test_the_tuned_keys_are_published_in_search_space_order():
    assert protocol_mod.XGB_TUNED_PARAM_NAMES == TUNED
    assert tuple(f.name for f in dataclasses.fields(XGBSearchSpace)) == TUNED


def test_estimator_implements_the_shared_stage3_interface():
    est = XGBEstimator()
    assert isinstance(est, Stage3Estimator)
    assert (est.family, est.as_array, est.random_state, est.n_jobs) == ("xgboost", False, 42, 1)


def test_config_sets_the_blackbox_identity_and_is_otherwise_the_glass_protocol():
    cfg, ebm = XGBStage3Config(), EBMStage3Config()
    assert (cfg.model_family, cfg.arm, cfg.counterpart, cfg.column_prefix) == (
        "xgboost", "blackbox", "glass_pipeline.ebm", "xgb_stage3_")
    assert (cfg.artifact_stem, cfg.study_name) == ("xgb_stage3", "xgb_stage3_recall_biased")
    assert ebm.counterpart == "blackbox_pipeline.models.xgb"
    mine, theirs = cfg.protocol(), ebm.protocol()
    assert mine.pop("search_space") == XGBSearchSpace().to_dict()
    theirs.pop("search_space")
    assert mine == theirs                                     # every mirrored field
    assert (cfg.n_tune_folds, cfg.n_eval_folds, cfg.n_trials, cfg.random_state) == (5, 10, 150, 42)


def test_config_round_trips_through_its_json_form():
    cfg = XGBStage3Config(n_trials=7, search_space=XGBSearchSpace(max_depth=(2, 4)),
                          expected_features=["a", "b"])
    restored = XGBStage3Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.to_dict() == json.loads(json.dumps(cfg.to_dict()))
    assert isinstance(restored.search_space, XGBSearchSpace)
    assert XGBEstimator(restored.search_space).describe_space() == XGBEstimator(
        cfg.search_space).describe_space()


def test_suggestions_follow_the_declared_space_in_order():
    space = XGBSearchSpace(max_depth=(2, 3))
    est = XGBEstimator(space)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(lambda t: float(est.normalize_params(est.suggest(t))["learning_rate"]), n_trials=20)
    assert list(study.trials[0].params) == list(TUNED)
    assert study.trials[0].distributions == {
        "n_estimators": IntDistribution(200, 2000, step=100),
        "learning_rate": FloatDistribution(0.005, 0.05, log=True),
        "max_depth": IntDistribution(2, 3),
        "min_child_weight": FloatDistribution(1.0, 20.0, log=True),
        "subsample": FloatDistribution(0.6, 1.0),
        "colsample_bytree": FloatDistribution(0.6, 1.0),
        "reg_lambda": FloatDistribution(0.001, 10.0, log=True),
    }
    for t in study.trials:                                    # every draw is a valid parameter set
        assert est.normalize_params(t.params) == t.params
    assert list(est.describe_space()) == list(TUNED)


# ----------------------------------------------------------------------
# hyperparameters (audit F1)
# ----------------------------------------------------------------------
def test_normalize_params_returns_exactly_the_tuned_keys_in_canonical_types():
    raw = {**{k: np.float64(v) for k, v in PARAMS.items()}, "n_estimators": np.int64(20),
           "max_depth": 2.0}
    out = XGBEstimator().normalize_params(dict(reversed(list(raw.items()))))
    assert list(out) == list(TUNED) and out == PARAMS
    assert all(type(out[k]) is (int if k in ("n_estimators", "max_depth") else float) for k in out)


def _without(key):
    return {k: v for k, v in PARAMS.items() if k != key}


@pytest.mark.parametrize("params", [
    _without("max_depth"),                                    # missing: XGBoost's default would apply
    {**PARAMS, "max_dept": 4},                                # unknown: ignored by XGBoost at verbosity=0
    {**_without("max_depth"), "max_dept": 4},                 # misspelled tuned key
    {**PARAMS, "eta": 0.3}, {**PARAMS, "seed": 7}, {**PARAMS, "nthread": 2},   # aliases
    {**PARAMS, "random_state": 7}, {**PARAMS, "objective": "binary:hinge"},    # fixed conditions
    {**PARAMS, "n_jobs": 4}, {**PARAMS, "tree_method": "exact"},
], ids=["missing", "unknown", "misspelled", "alias_eta", "alias_seed", "alias_nthread",
        "fixed_random_state", "fixed_objective", "fixed_n_jobs", "fixed_tree_method"])
def test_missing_or_unknown_keys_are_refused_f1(params):
    with pytest.raises(KeyError):
        XGBEstimator().normalize_params(params)


@pytest.mark.parametrize("key, value", [
    ("n_estimators", 0), ("n_estimators", -5), ("n_estimators", 20.7), ("n_estimators", "20"),
    ("max_depth", True), ("max_depth", -1), ("max_depth", 2.5), ("max_depth", "4"),
    ("learning_rate", 0.0), ("learning_rate", float("nan")), ("learning_rate", math.inf),
    ("learning_rate", "0.1"), ("learning_rate", None),
    ("subsample", 0.0), ("subsample", 1.5), ("colsample_bytree", 0.0), ("colsample_bytree", 1.01),
    ("min_child_weight", -1.0), ("reg_lambda", -0.1), ("reg_lambda", np.bool_(True)),
])
def test_values_outside_xgboost_domain_are_refused_f1(key, value):
    """Several of these XGBoost would accept silently (0 trees, eta 0 or NaN,
    subsample 0, depth True -> 1, 20.7 trees -> 20)."""
    with pytest.raises(ValueError, match=key):
        XGBEstimator().normalize_params({**PARAMS, key: value})


@pytest.mark.parametrize("key, value", [
    ("max_depth", 0), ("min_child_weight", 0.0), ("reg_lambda", 0.0), ("subsample", 1.0),
    ("colsample_bytree", 1.0), ("n_estimators", 1),
])
def test_values_at_xgboost_domain_edges_are_accepted(key, value):
    assert XGBEstimator().normalize_params({**PARAMS, key: value})[key] == value


def test_fit_applies_exactly_the_normalized_params_and_the_fixed_conditions(fitted):
    got = fitted.get_params()
    assert {k: got[k] for k in TUNED} == PARAMS
    assert (got["objective"], got["eval_metric"], got["tree_method"], got["random_state"],
            got["n_jobs"], got["verbosity"]) == ("binary:logistic", "logloss", "hist", 42, 1, 0)


def test_fit_refuses_bad_params_before_training(frames):
    X, y, _ = frames
    for bad, error in (({**PARAMS, "max_dept": 4}, KeyError), ({**PARAMS, "n_estimators": 0}, ValueError)):
        with pytest.raises(error):
            XGBEstimator(n_jobs=1).fit(bad, X, y)


def test_the_runner_refuses_bad_supplied_params_before_anything_is_fitted(tmp_path):
    cfg = _small_config(tmp_path)
    log = CallLog()
    runner = Stage3Runner(cfg, RecordingStage3Estimator(XGBEstimator(n_jobs=1), log))
    with pytest.raises(KeyError, match="max_dept"):
        runner.fit(_passthrough_block(cfg), params={**PARAMS, "max_dept": 4}, params_source="typo")
    assert not log.fits()


def test_the_notebooks_explicit_params_satisfy_the_contract():
    """``XGB_PARAMS_EXPLICIT`` (the ``"explicit"`` source) is the one hand-typed set."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        src = "".join(cell["source"]) if cell["cell_type"] == "code" else ""
        if "XGB_PARAMS_EXPLICIT = {" in src:
            node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                        and getattr(n.targets[0], "id", None) == "XGB_PARAMS_EXPLICIT")
            explicit = ast.literal_eval(node.value)
            break
    assert XGBEstimator().normalize_params(explicit) == explicit


# ----------------------------------------------------------------------
# fitting and scoring
# ----------------------------------------------------------------------
#: Row and column subsampling on, so both of XGBoost's samplers draw.
SAMPLED = {**PARAMS, "subsample": 0.7, "colsample_bytree": 0.7}


def _score(estimator, frames, params, sample_weight=None):
    X, y, X_te = frames
    return estimator.positive_proba(estimator.fit(params, X, y, sample_weight=sample_weight), X_te)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_an_identical_configuration_reproduces_bit_for_bit(frames, n_jobs):
    """Same params, seed, data and thread count: the same model, with the row
    and column samplers both drawing. The seed is applied (another seed
    samples other rows and columns)."""
    base = _score(XGBEstimator(n_jobs=n_jobs), frames, SAMPLED)
    np.testing.assert_array_equal(_score(XGBEstimator(n_jobs=n_jobs), frames, SAMPLED), base)
    assert not np.allclose(_score(XGBEstimator(random_state=7, n_jobs=n_jobs), frames, SAMPLED), base)


def test_the_thread_count_does_not_change_a_fit_without_row_subsampling(frames):
    """No sampler draws per row here: column sampling (``colsample_bytree``)
    is one serial shuffle per tree, and histogram sums, split search and
    prediction do not depend on the thread count, so the fit is bit-identical.

    Row subsampling is left out on purpose. XGBoost's CPU row sampler
    (``ParallelSampling`` / ``UniformSample`` in src/tree/hist/sampler.cc)
    jumps each thread's engine ahead by exactly one value per preceding row,
    so it is thread-count independent only if ``std::bernoulli_distribution``
    takes one engine value per row. The MSVC standard library behind the
    Windows wheels (P0952R2 ``generate_canonical``) rejects and redraws about
    1 in 1,024 values of XGBoost's engine, whose range 2^63 - 1 is not a power
    of two; from then on a single-threaded fit and a two-threaded fit sample
    different rows. Linux wheels built with libstdc++ before GCC 16 draw once
    per row. The thread count is therefore pinned like the seed (next test)."""
    params = {**PARAMS, "colsample_bytree": 0.7}
    assert params["subsample"] == 1.0
    base = _score(XGBEstimator(n_jobs=1), frames, params)
    np.testing.assert_array_equal(_score(XGBEstimator(n_jobs=2), frames, params), base)


def test_the_thread_count_is_pinned_by_the_protocol(fitted):
    """With row subsampling the thread count can select other rows (above), so
    Stage 3 fixes it in the protocol both arms must match, records it with
    every artifact, and the fitted model runs with it."""
    cfg = XGBStage3Config()
    assert "n_jobs" in PROTOCOL_FIELDS and cfg.n_jobs == 1 and cfg.protocol()["n_jobs"] == 1
    assert cfg.to_dict()["n_jobs"] == 1
    assert fitted.get_params()["n_jobs"] == XGBEstimator(n_jobs=1).n_jobs == 1


def test_sample_weights_reach_the_fit(frames):
    """No sampler draws, so only the weights differ between the two fits.
    Up-weighting the positive class (as balanced Stage 3 weights do) raises
    P(y=1); the weighted fit is itself reproducible."""
    y = frames[1]
    weights = np.where(y == 1, 4.0, 1.0)
    plain = _score(XGBEstimator(n_jobs=1), frames, PARAMS)
    weighted = _score(XGBEstimator(n_jobs=1), frames, PARAMS, weights)
    np.testing.assert_array_equal(_score(XGBEstimator(n_jobs=1), frames, PARAMS, weights), weighted)
    assert not np.allclose(weighted, plain)
    assert weighted.mean() > plain.mean()


def test_positive_proba_is_p_of_class_one_and_bounded(fitted, frames):
    _, _, X_te = frames
    p = XGBEstimator().positive_proba(fitted, X_te)
    assert fitted.classes_.tolist() == [0, 1]
    np.testing.assert_array_equal(p, fitted.predict_proba(X_te)[:, 1])
    assert p.shape == (len(X_te),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_scoring_frames_are_matched_by_column_name(fitted, frames):
    X, _, X_te = frames
    assert list(fitted.feature_names_in_) == list(X.columns)
    cols = list(X_te.columns)
    for frame in (X_te[cols[::-1]], X_te[cols[:-1]], X_te.assign(extra=0.0)):
        with pytest.raises(ValueError, match="feature_names mismatch"):
            XGBEstimator().positive_proba(fitted, frame)


def test_estimator_records_everything_needed_to_rebuild_it():
    space = XGBSearchSpace(max_depth=(2, 4))
    d = XGBEstimator(space, random_state=7, n_jobs=3).to_dict()
    assert d == {"family": "xgboost", "search_space": space.to_dict(), "random_state": 7, "n_jobs": 3,
                 "fixed": {"objective": "binary:logistic", "tree_method": "hist", "eval_metric": "logloss"}}


# ----------------------------------------------------------------------
# tuning time vs final fit
# ----------------------------------------------------------------------
def test_a_tuned_run_records_and_applies_the_same_parameters(tmp_path):
    space = XGBSearchSpace(n_estimators=(20, 60, 20), max_depth=(2, 3))
    cfg = _small_config(tmp_path, n_trials=3, search_space=space)
    art, _ = train_xgb_stage(_passthrough_block(cfg), config=cfg, save=False)
    best = art.best_params
    assert list(best) == list(TUNED) and art.tuning["n_complete"] == 3
    assert best == XGBEstimator(space).normalize_params(art.tuning["top_trials"][0]["params"])
    assert space.n_estimators[0] <= best["n_estimators"] <= space.n_estimators[1]
    applied = art.refit["model"].get_params()
    assert {k: applied[k] for k in TUNED} == best
    assert art.config["search_space"] == dataclasses.asdict(space)
