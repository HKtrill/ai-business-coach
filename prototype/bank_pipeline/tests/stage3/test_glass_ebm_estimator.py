"""
``EBMFactory`` and ``EBMStage3Config`` (glass_pipeline.ebm): the GLASS arm's
implementation of the shared Stage 3 estimator interface.

Contract: the factory satisfies ``Stage3Estimator``; parameters are exactly
the four tuned keys (missing or unknown keys raise) and suggestions follow the
declared space in the documented order; ``build`` passes parameters,
interaction pairs, seed and n_jobs through; the fitted EBM carries one main
effect per block feature plus exactly the configured pairs, bound by column
name; fits are deterministic for a seed and honour sample weights; labels may
be any binary dtype; ``positive_proba`` is P(y = 1) with shape (n,), matches
the shared helper and is defined for single-class fits. The config inherits
every protocol field from ``Stage3Config`` and round-trips through a dict.

The four-key rule is also the F14 regression for the research trainer
(tests/feature_research/test_stage_trainers.py).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from glass_pipeline.ebm.config import EBMStage3Config
from glass_pipeline.ebm.estimator import EBM_TUNED_PARAM_NAMES, EBMFactory, EBMSearchSpace
from glass_pipeline.ebm.interactions import define_ebm_interactions
from shared.stage3.config import PROTOCOL_FIELDS, Stage3Config
from shared.stage3.estimator import Stage3Estimator, positive_proba
from shared.stage3.feature_engineering import EBM_FEATURES
from tests.stage3.conftest import EBM_TEST_PARAMS, cheap_ebm


@pytest.fixture(scope="module")
def fitted(ebm_features):
    """The production pairs on the engineered block, fitted once."""
    split, _, X_tr, X_te = ebm_features
    factory = cheap_ebm(define_ebm_interactions(X_tr))
    return factory, factory.fit(EBM_TEST_PARAMS, X_tr, split["y_train"]), X_tr, X_te


class RecordingTrial:
    """Stands in for an Optuna trial: records each suggestion, returns the low end."""

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, log=False):
        self.calls.append((name, low, high, "log" if log else "linear"))
        return low

    def suggest_int(self, name, low, high, step=1):
        self.calls.append((name, low, high, step))
        return low


# ----------------------------------------------------------------------
# shared interface and configuration
# ----------------------------------------------------------------------
def test_factory_implements_the_shared_stage3_interface():
    factory = EBMFactory(interactions=[])
    assert isinstance(factory, Stage3Estimator)
    assert factory.family == "ebm" and factory.as_array is False      # fitted on named columns
    assert (factory.random_state, factory.n_jobs) == (42, 1)


def test_config_sets_the_glass_identity_and_inherits_the_protocol():
    cfg = EBMStage3Config()
    assert (cfg.model_family, cfg.arm, cfg.column_prefix, cfg.artifact_stem) == (
        "ebm", "glass", "ebm_stage3_", "ebm_stage3")
    assert cfg.counterpart == "blackbox_pipeline.models.xgb"
    assert cfg.search_space == EBMSearchSpace()
    base = Stage3Config()
    assert {k: getattr(cfg, k) for k in PROTOCOL_FIELDS} == {k: getattr(base, k) for k in PROTOCOL_FIELDS}


def test_config_round_trips_through_its_dict():
    """An artifact stores ``config.to_dict()``; ``from_dict`` rebuilds the
    search space as an ``EBMSearchSpace``, not a plain dict."""
    cfg = EBMStage3Config(expected_features=list(EBM_FEATURES), n_trials=7,
                          search_space=EBMSearchSpace(max_rounds_high=900))
    again = EBMStage3Config.from_dict(cfg.to_dict())
    assert again == cfg and isinstance(again.search_space, EBMSearchSpace)


# ----------------------------------------------------------------------
# parameters and search space
# ----------------------------------------------------------------------
def test_normalize_params_accepts_exactly_the_tuned_keys():
    shuffled = dict(reversed(list(EBM_TEST_PARAMS.items())))
    out = EBMFactory.normalize_params({**shuffled, "max_rounds": 40.0, "max_bins": "32"})
    assert list(out) == list(EBM_TUNED_PARAM_NAMES)
    assert out == EBM_TEST_PARAMS
    assert [type(v) for v in out.values()] == [float, int, int, int]
    with pytest.raises(KeyError, match="missing canonical keys: \\['max_bins'\\]"):
        EBMFactory.normalize_params({k: v for k, v in EBM_TEST_PARAMS.items() if k != "max_bins"})
    with pytest.raises(KeyError, match="Unknown EBM hyperparameters \\['max_depth'\\]"):
        EBMFactory.normalize_params({**EBM_TEST_PARAMS, "max_depth": 3})


def test_suggestions_follow_the_declared_space_in_order():
    """Order matters: under seed 42 TPE draws in suggestion order, so a
    reordering would change every tuned EBM."""
    space = EBMSearchSpace(max_rounds_low=100, max_rounds_high=900)
    trial = RecordingTrial()
    params = EBMFactory([], search_space=space).suggest(trial)
    assert [c[0] for c in trial.calls] == list(EBM_TUNED_PARAM_NAMES) == list(params)
    s = space
    assert trial.calls == [
        ("learning_rate", s.learning_rate_low, s.learning_rate_high, "log"),
        ("max_rounds", 100, 900, s.max_rounds_step),
        ("max_bins", s.max_bins_low, s.max_bins_high, s.max_bins_step),
        ("max_interaction_bins", s.max_interaction_bins_low, s.max_interaction_bins_high,
         s.max_interaction_bins_step),
    ]
    assert EBMFactory.normalize_params(params) == params
    assert set(EBMFactory([], search_space=space).describe_space()) == set(EBM_TUNED_PARAM_NAMES)


def test_build_passes_parameters_pairs_seed_and_jobs():
    pairs = [("campaign", "emp_var_rate_sigmoid_slope")]
    factory = EBMFactory(pairs, random_state=7, n_jobs=1)
    got = factory.build(EBM_TEST_PARAMS).get_params()
    assert {k: got[k] for k in EBM_TUNED_PARAM_NAMES} == EBM_TEST_PARAMS
    assert (got["interactions"], got["random_state"], got["n_jobs"]) == (pairs, 7, 1)
    record = factory.to_dict()
    assert record["interactions"] == [list(p) for p in pairs]
    assert record["tuned_param_names"] == list(EBM_TUNED_PARAM_NAMES)
    assert record["search_space"] == EBMSearchSpace().to_dict()


# ----------------------------------------------------------------------
# fitted model
# ----------------------------------------------------------------------
def test_fitted_ebm_has_one_term_per_feature_plus_exactly_the_pairs(fitted):
    factory, model, X_tr, _ = fitted
    pair_terms = {f"{a} & {b}" for a, b in factory.interactions}
    assert len(pair_terms) == 3
    names = list(model.term_names_)
    assert names[:len(EBM_FEATURES)] == list(EBM_FEATURES)
    assert set(names[len(EBM_FEATURES):]) == pair_terms               # no auto-detected pairs


def _pairs(model) -> set:
    names = list(model.feature_names_in_)
    return {frozenset(names[i] for i in term) for term in model.term_features_ if len(term) == 2}


def test_pairs_and_scoring_are_bound_by_column_name(fitted, ebm_features):
    """Pairs are declared by name (``as_array = False``): reordering the
    training columns cannot move a pair onto other features, and a fitted
    model scores a reordered frame identically. (Boosting itself cycles in
    column order, which is why ``Stage3Block`` pins the order to
    ``EBM_FEATURES``; see test_glass_ebm_stage.py.)"""
    factory, model, X_tr, X_te = fitted
    expected = {frozenset(p) for p in factory.interactions}
    again = factory.fit(EBM_TEST_PARAMS, X_tr.iloc[:, ::-1], ebm_features[0]["y_train"])
    assert _pairs(again) == _pairs(model) == expected
    np.testing.assert_array_equal(factory.positive_proba(model, X_te.iloc[:, ::-1]),
                                  factory.positive_proba(model, X_te))


def test_an_unknown_pair_member_is_refused_at_fit(ebm_features):
    _, _, X_tr, _ = ebm_features
    with pytest.raises(ValueError, match="not_a_feature"):
        cheap_ebm([("campaign", "not_a_feature")]).fit(EBM_TEST_PARAMS, X_tr, ebm_features[0]["y_train"])


def test_fits_are_deterministic_for_a_seed_and_use_the_weights(fitted, ebm_features):
    factory, model, X_tr, X_te = fitted
    y = ebm_features[0]["y_train"]
    p = factory.positive_proba(model, X_te)
    np.testing.assert_array_equal(factory.positive_proba(factory.fit(EBM_TEST_PARAMS, X_tr, y), X_te), p)

    reseeded = cheap_ebm(factory.interactions, random_state=7)
    assert not np.allclose(reseeded.positive_proba(reseeded.fit(EBM_TEST_PARAMS, X_tr, y), X_te), p)
    weights = np.where(y.to_numpy() == 1, 4.0, 1.0)          # positives up-weighted, as in the runner
    weighted = factory.fit(EBM_TEST_PARAMS, X_tr, y, sample_weight=weights)
    assert not np.allclose(factory.positive_proba(weighted, X_te), p)


@pytest.mark.parametrize("as_type", ["int8", "int64", "bool", "ndarray"])
def test_any_binary_label_dtype_gives_the_same_model(fitted, ebm_features, as_type):
    factory, model, X_tr, X_te = fitted
    y = ebm_features[0]["y_train"]
    labels = y.to_numpy() if as_type == "ndarray" else y.astype(as_type)
    other = factory.fit(EBM_TEST_PARAMS, X_tr, labels)
    assert list(other.classes_) == [0, 1]
    np.testing.assert_array_equal(factory.positive_proba(other, X_te), factory.positive_proba(model, X_te))


def test_positive_proba_is_the_shared_p_of_class_one(fitted):
    factory, model, _, X_te = fitted
    p = factory.positive_proba(model, X_te)
    assert p.shape == (len(X_te),) and p.dtype == float
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(p, model.predict_proba(X_te)[:, list(model.classes_).index(1)])
    np.testing.assert_array_equal(p, positive_proba(model, X_te, as_array=factory.as_array))


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_fit_scores_that_class(ebm_features, label):
    """The shared interface promises P(y = 1) even when a fit sees one class:
    0 everywhere for all-negative labels, 1 for all-positive."""
    _, _, X_tr, X_te = ebm_features
    factory = cheap_ebm()
    model = factory.fit(EBM_TEST_PARAMS, X_tr, np.full(len(X_tr), label))
    np.testing.assert_array_equal(factory.positive_proba(model, X_te), np.full(len(X_te), float(label)))


def test_factory_records_everything_needed_to_rebuild_it(fitted):
    factory = fitted[0]
    record = factory.to_dict()
    rebuilt = EBMFactory([tuple(p) for p in record["interactions"]],
                         EBMSearchSpace(**record["search_space"]),
                         record["random_state"], record["n_jobs"], record["extra_params"])
    assert dataclasses.asdict(rebuilt.search_space) == dataclasses.asdict(factory.search_space)
    assert rebuilt.build(EBM_TEST_PARAMS).get_params() == factory.build(EBM_TEST_PARAMS).get_params()
