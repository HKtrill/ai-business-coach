"""
Black-box Stage 1 (MLP) hyperparameters: PR 36 audit F1.

Contract: ``params_to_kwargs`` — the one translation from Optuna's flat
parameters to ``Stage1MLPClassifier`` kwargs, used by every tuning trial and
by the final fit — accepts exactly the seven tuned keys (``KeyError``
otherwise) with valid values (``ValueError`` otherwise: non-finite, boolean,
non-integral where an integer is meant, outside the estimator's domain, or an
activation ``MLPClassifier`` does not offer; ``n_layers`` must be 1 or 2, the
architectures it can build), and only ``MLP_FIXED_PARAM_NAMES`` as fixed
kwargs, checked the same way and before any tuning trial is fitted;
``Stage1MLPConfig`` refuses non-integer integer settings. Every Optuna draw
is valid and maps to exactly the kwargs it mapped to before, so valid
configurations behave identically; a fitted stage records exactly the
parameters its final model runs with.

Before PR 36: unknown keys were dropped (recorded, never applied), ``width``
/ ``n_layers`` / ``batch_size`` were truncated (``width=32.9`` built 32 units,
``n_layers=True`` one layer), any ``n_layers`` other than 1 (0 and 3 included)
built two layers, and ``max_epochs=True`` trained one epoch.
"""

from __future__ import annotations

import math

import numpy as np
import optuna
import pytest

import blackbox_pipeline.models.mlp.tuning as tuning_mod
from blackbox_pipeline.models.mlp import CalibratedStage1MLP, Stage1MLPConfig
from blackbox_pipeline.models.mlp.estimator import Stage1MLPClassifier
from blackbox_pipeline.models.mlp.tuning import params_to_kwargs, sample_params, tune_stage1_mlp_auc

pytestmark = pytest.mark.filterwarnings(        # the MLP's tiny synthetic folds (see Stage 1 suite)
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")

TUNED = ("n_layers", "width", "activation", "alpha", "learning_rate_init", "batch_size", "pos_ratio")
PARAMS = {"n_layers": 2, "width": 16, "activation": "relu", "alpha": 1e-4,
          "learning_rate_init": 1e-3, "batch_size": 128, "pos_ratio": 0.5}
FIXED = Stage1MLPConfig().estimator_fixed_kwargs


def _legacy_params_to_kwargs(params, **fixed):
    """``params_to_kwargs`` before PR 36, for the identity check on valid input."""
    width, n_layers = int(params["width"]), int(params["n_layers"])
    hidden = (width,) if n_layers == 1 else (width, max(width // 2, 2))
    return dict(hidden_layer_sizes=hidden, activation=params["activation"],
                alpha=float(params["alpha"]), learning_rate_init=float(params["learning_rate_init"]),
                batch_size=int(params["batch_size"]), pos_ratio=float(params["pos_ratio"]), **fixed)


def normalize_mlp_params(params):
    """Resolved at call time, so this module also runs against the pre-PR-36 code."""
    return tuning_mod.normalize_mlp_params(params)


def test_the_tuned_and_fixed_names_are_published():
    assert tuning_mod.MLP_TUNED_PARAM_NAMES == TUNED
    assert tuning_mod.MLP_FIXED_PARAM_NAMES == tuple(FIXED) == ("max_epochs", "patience", "val_fraction",
                                                       "random_state")


def test_valid_params_come_back_exactly_in_canonical_types():
    raw = {"pos_ratio": np.float64(0.5), "batch_size": 128.0, "learning_rate_init": np.float32(1e-3),
           "alpha": 1e-4, "activation": "relu", "width": np.int64(16), "n_layers": 2}
    out = normalize_mlp_params(raw)
    assert list(out) == list(TUNED)
    assert out == {**PARAMS, "learning_rate_init": float(np.float32(1e-3))}
    assert [type(out[k]) for k in TUNED] == [int, int, str, float, float, int, float]


def _without(key):
    return {k: v for k, v in PARAMS.items() if k != key}


@pytest.mark.parametrize("params", [
    _without("width"), {**PARAMS, "widht": 8}, {**_without("width"), "widht": 8},
    {**PARAMS, "hidden_layer_sizes": (8,)}, {**PARAMS, "max_epochs": 5}, {**PARAMS, "solver": "sgd"},
], ids=["missing", "unknown", "misspelled", "estimator_kwarg", "fixed_kwarg_as_tuned", "sklearn_kwarg"])
def test_missing_or_unknown_tuned_keys_are_refused_f1(params):
    with pytest.raises(KeyError):
        params_to_kwargs(params, **FIXED)


@pytest.mark.parametrize("key, value", [
    ("n_layers", 0), ("n_layers", 3), ("n_layers", 1.5), ("n_layers", True),
    ("width", 32.9), ("width", 0), ("width", True), ("width", "16"),
    ("batch_size", 128.5), ("batch_size", 0), ("batch_size", False),
    ("alpha", -1e-3), ("alpha", math.nan), ("alpha", math.inf), ("alpha", None),
    ("learning_rate_init", 0.0), ("learning_rate_init", -1e-3), ("learning_rate_init", math.nan),
    ("pos_ratio", -0.1), ("pos_ratio", math.nan), ("pos_ratio", np.bool_(True)),
    ("activation", "softmax"), ("activation", 1), ("activation", None),
], ids=lambda v: repr(v))
def test_invalid_tuned_values_are_refused_f1(key, value):
    with pytest.raises(ValueError, match=key):
        params_to_kwargs({**PARAMS, key: value}, **FIXED)


@pytest.mark.parametrize("key, value", [
    ("n_layers", 1), ("width", 1), ("alpha", 0.0), ("pos_ratio", 0.0), ("batch_size", 1),
    ("activation", "identity"), ("activation", "logistic"),
])
def test_values_at_the_domain_edges_are_accepted(key, value):
    assert normalize_mlp_params({**PARAMS, key: value})[key] == value


@pytest.mark.parametrize("fixed, error", [
    ({"max_epoch": 5}, KeyError), ({"random_seed": 1}, KeyError), ({"alpha": 1e-3}, KeyError),
    ({"max_epochs": 0}, ValueError), ({"max_epochs": 2.5}, ValueError),
    ({"max_epochs": True}, ValueError), ({"patience": 0}, ValueError),
    ({"val_fraction": 0.0}, ValueError), ({"val_fraction": 1.0}, ValueError),
    ({"val_fraction": math.nan}, ValueError), ({"random_state": -1}, ValueError),
    ({"random_state": True}, ValueError),
])
def test_invalid_fixed_kwargs_are_refused_f1(fixed, error):
    with pytest.raises(error):
        params_to_kwargs(PARAMS, **fixed)


def test_a_partial_set_of_fixed_kwargs_is_a_valid_override():
    kwargs = params_to_kwargs(PARAMS, max_epochs=1)
    assert kwargs["max_epochs"] == 1 and "patience" not in kwargs
    assert Stage1MLPClassifier(**kwargs).get_params()["patience"] == Stage1MLPClassifier().patience


def test_every_optuna_draw_is_valid_and_maps_as_before():
    """The search space only produces valid parameters, and for every one the
    kwargs are exactly what the pre-PR-36 translation produced."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    study.optimize(lambda t: normalize_mlp_params(sample_params(t))["alpha"], n_trials=60)
    for t in study.trials:
        assert normalize_mlp_params(t.params) == {k: t.params[k] for k in TUNED}
        assert params_to_kwargs(t.params, **FIXED) == _legacy_params_to_kwargs(t.params, **FIXED)
    assert {t.params["n_layers"] for t in study.trials} == {1, 2}


def test_tuning_refuses_bad_fixed_kwargs_before_any_trial_is_fitted(monkeypatch):
    fits, studies = [], []
    monkeypatch.setattr(tuning_mod.Stage1MLPClassifier, "fit",
                        lambda self, X, y: fits.append(1) or self)
    real_study = tuning_mod.optuna.create_study
    monkeypatch.setattr(tuning_mod.optuna, "create_study",
                        lambda *a, **k: studies.append(1) or real_study(*a, **k))
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), np.r_[np.zeros(30), np.ones(10)].astype(int)
    from sklearn.model_selection import StratifiedKFold
    for fixed, error in (({**FIXED, "max_epoch": 5}, KeyError), ({**FIXED, "max_epochs": True}, ValueError)):
        with pytest.raises(error):
            tune_stage1_mlp_auc(X, y, StratifiedKFold(3, shuffle=True, random_state=0), 2, fixed, 0,
                                verbose=False)
    assert not fits and not studies                       # refused before a study is opened


@pytest.mark.parametrize("override, message", [
    ({"max_epochs": True}, "max_epochs must be an integer"),
    ({"max_epochs": 10.5}, "max_epochs must be an integer"),
    ({"patience": 2.0}, "patience must be an integer"),
    ({"n_trials": True}, "n_trials must be an integer"),
    ({"cv_folds": 5.0}, "cv_folds must be an integer"),
    ({"random_state": "42"}, "random_state must be an integer"),
    ({"val_fraction": True}, "val_fraction must be a number"),
    ({"val_fraction": "0.1"}, "val_fraction must be a number"),
])
def test_the_stage_refuses_non_integer_settings_before_fitting_f1(override, message):
    with pytest.raises(ValueError, match=message):
        Stage1MLPConfig(**override)
    with pytest.raises(ValueError, match=message):
        CalibratedStage1MLP(**override)


def test_the_documented_and_test_configurations_stay_valid():
    from tests.stage1.conftest import MLP_TEST_CONFIG
    for cfg in (Stage1MLPConfig(), Stage1MLPConfig(**MLP_TEST_CONFIG)):
        assert params_to_kwargs(PARAMS, **cfg.estimator_fixed_kwargs)["max_epochs"] == cfg.max_epochs


def test_a_fitted_stage_records_exactly_the_parameters_its_final_model_runs_with(mlp_run):
    stage = mlp_run.stage
    assert normalize_mlp_params(stage.best_params) == stage.best_params
    kwargs = params_to_kwargs(stage.best_params, **stage.config.estimator_fixed_kwargs)
    assert {k: stage.model.get_params()[k] for k in kwargs} == kwargs
    assert mlp_run.output.best_params == stage.best_params
