"""
Black-box Stage 2 (RF Router) hyperparameters: PR 36 audit F1.

Contract: ``validate_rf_params`` accepts exactly the six normalised keys, each
in the domain ``forest.build_pipe`` applies it in, and is applied wherever RF
parameters enter: ``RFRouterConfig`` (``rf_params`` and ``pass2_rf_params``,
every problem reported at once, before any forest is fitted),
``ForestTrainer`` (every forest the router family builds), ``normalise_rf_params``
(the explicit, artifact and tuned paths of ``resolve_rf_params``; documented
raw-Optuna defaults kept) and the fixed values of the specialization search
(``suggest_rf_params``). ``pass2_rf_params`` is refused by the single-forest
ablation, which would never apply it; the config's integer settings are
integers. Every Optuna draw is valid and normalises as before, valid inputs
come back unchanged, and a fitted router's forests and saved artifact carry
exactly the configured values for each pass.

Before PR 36: an unknown or misspelled key was kept in the config, saved in
the artifact and never applied; ``normalise_rf_params`` dropped extra keys of
an explicit dict; fractional ``n_estimators`` / ``min_samples_leaf`` were
truncated; out-of-domain values failed only when a forest was fitted;
``max_features_type='fraction'`` without a fraction became ``None`` (all
features).

Not changed: routing, thresholds, remainder, constraints, the search space.
"""

from __future__ import annotations

import ast
import json
import math

import numpy as np
import optuna
import pytest

import blackbox_pipeline.models.rf_router.config as config_mod
import blackbox_pipeline.models.rf_router.forest as forest_mod
from blackbox_pipeline.models.rf_router import (
    DEFAULT_RF_PARAMS, RFRouter, RFRouterConfig, load_rf_router, save_rf_router,
)
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.tuning import (
    REQUIRED_PARAM_KEYS, normalise_rf_params, resolve_rf_params, suggest_rf_params,
)
from tests.support.notebooks import BANK_ROOT
from tests.support.rf_router_world import quiet
from tests.support.stage2_synthetic import ALT_PARAMS, SMALL_PARAMS, make_data, router_config

NOTEBOOK = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"
KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "minority_weight",
        "class_weight")


def validate_rf_params(params, label="rf_params"):
    """Resolved at call time, so this module also runs against the pre-PR-36 code."""
    return config_mod.validate_rf_params(params, label)


def check_rf_param(key, value, label="rf_params"):
    return config_mod.check_rf_param(key, value, label)


def _legacy_normalise(raw):
    """``normalise_rf_params`` before PR 36, for the identity check on valid input."""
    if all(k in raw for k in KEYS):
        return {k: raw[k] for k in KEYS}
    mf_type = raw.get("max_features_type", "sqrt")
    return {"n_estimators": raw["n_estimators"],
            "max_depth": raw.get("max_depth") if raw.get("use_max_depth") else None,
            "min_samples_leaf": raw["min_samples_leaf"],
            "max_features": raw.get("max_features_fraction") if mf_type == "fraction" else mf_type,
            "minority_weight": raw.get("minority_weight", 1.0), "class_weight": "balanced"}


# ----------------------------------------------------------------------
# the parameter contract
# ----------------------------------------------------------------------
def test_valid_params_come_back_unchanged_except_integral_values_as_int():
    assert REQUIRED_PARAM_KEYS == KEYS
    for params in (DEFAULT_RF_PARAMS, SMALL_PARAMS, ALT_PARAMS):
        out = validate_rf_params(dict(reversed(list(params.items()))))
        assert list(out) == list(KEYS) and out == params
    out = validate_rf_params({**SMALL_PARAMS, "n_estimators": np.int64(25), "min_samples_leaf": 10.0,
                              "max_depth": 4.0, "max_features": np.float64(0.5)})
    assert [type(out[k]) for k in ("n_estimators", "min_samples_leaf", "max_depth")] == [int, int, int]
    assert out["max_features"] == 0.5


def _without(key):
    return {k: v for k, v in SMALL_PARAMS.items() if k != key}


@pytest.mark.parametrize("label", ["rf_params", "pass2_rf_params"])
@pytest.mark.parametrize("params, message", [
    (_without("max_depth"), "missing keys: \\['max_depth'\\]"),
    ({**SMALL_PARAMS, "max_dept": 6}, "unknown keys: \\['max_dept'\\]"),
    ({**_without("max_depth"), "max_dept": 6}, "missing keys"),
    ({**SMALL_PARAMS, "criterion": "entropy"}, "unknown keys: \\['criterion'\\]"),
    ({**SMALL_PARAMS, "n_jobs": 4}, "unknown keys: \\['n_jobs'\\]"),
    ({**SMALL_PARAMS, "random_state": 1}, "unknown keys: \\['random_state'\\]"),
], ids=["missing", "unknown", "misspelled", "sklearn_kwarg", "fixed_n_jobs", "fixed_seed"])
def test_missing_or_unknown_keys_are_refused_for_either_pass_f1(label, params, message):
    with pytest.raises(ValueError, match=f"{label} {message}"):
        router_config(**{label: params})


@pytest.mark.parametrize("label", ["rf_params", "pass2_rf_params"])
@pytest.mark.parametrize("key, value", [
    ("n_estimators", 0), ("n_estimators", 20.7), ("n_estimators", True), ("n_estimators", "25"),
    ("n_estimators", math.nan), ("max_depth", 0), ("max_depth", 2.5), ("max_depth", True),
    ("min_samples_leaf", 0), ("min_samples_leaf", 2.5), ("min_samples_leaf", 0.05),
    ("max_features", "auto"), ("max_features", 0.0), ("max_features", 1.5), ("max_features", 2),
    ("max_features", True), ("max_features", math.nan),
    ("minority_weight", 0.0), ("minority_weight", -1.0), ("minority_weight", math.nan),
    ("minority_weight", math.inf), ("minority_weight", True), ("minority_weight", "2"),
    ("class_weight", "balanced_subsampled"), ("class_weight", {0: 1, 1: 3}),
], ids=lambda v: repr(v))
def test_invalid_values_are_refused_for_either_pass_before_any_fit_f1(label, key, value):
    with pytest.raises(ValueError, match=f"{label}.{key}"):
        router_config(**{label: {**SMALL_PARAMS, key: value}})


@pytest.mark.parametrize("key, value", [
    ("max_depth", None), ("max_depth", 1), ("max_features", None), ("max_features", "log2"),
    ("max_features", 1.0), ("max_features", 0.3), ("minority_weight", 0.5), ("minority_weight", 3),
    ("class_weight", None), ("class_weight", "balanced_subsample"), ("min_samples_leaf", 1),
    ("n_estimators", 1),
])
def test_values_at_the_domain_edges_are_accepted(key, value):
    assert router_config(pass2_rf_params={**SMALL_PARAMS, key: value}).pass2_rf_params[key] == value


def test_every_problem_in_both_passes_is_reported_at_once():
    with pytest.raises(ValueError) as err:
        router_config(rf_params={**SMALL_PARAMS, "max_dept": 6, "n_estimators": 2.5},
                      pass2_rf_params={**SMALL_PARAMS, "minority_weight": 0}, n_oof_folds=2.5)
    msg = str(err.value)
    for part in ("rf_params unknown keys: ['max_dept']", "rf_params.n_estimators",
                 "pass2_rf_params.minority_weight", "n_oof_folds must be an integer"):
        assert part in msg, part


def test_the_ablation_refuses_pass2_params_it_would_never_apply_f1():
    with pytest.raises(ValueError, match="pass2_rf_params applies to the two_pass cascade only"):
        router_config(mode="single_rf", pass2_rf_params=ALT_PARAMS)
    assert router_config(mode="single_rf").pass2_rf_params is None


@pytest.mark.parametrize("override, message", [
    ({"n_oof_folds": 5.5}, "n_oof_folds must be an integer"),
    ({"n_oof_folds": True}, "n_oof_folds must be an integer"),
    ({"n_inner_folds": 2.5}, "n_inner_folds must be an integer"),
    ({"remainder_min_size": True}, "remainder_min_size must be an integer"),
    ({"remainder_min_size": 0}, "remainder_min_size must be >= 1"),
    ({"random_state": -1}, "random_state must be >= 0"),
    ({"random_state": "7"}, "random_state must be an integer"),
])
def test_integer_settings_must_be_integers_f1(override, message):
    with pytest.raises(ValueError, match=message):
        router_config(**override)


def test_the_forest_builder_refuses_bad_params_before_any_fit_f1():
    for bad in ({**SMALL_PARAMS, "max_dept": 6}, _without("class_weight"),
                {**SMALL_PARAMS, "min_samples_leaf": 2.5}):
        with pytest.raises(ValueError, match="ForestTrainer params"):
            ForestTrainer(bad)
    assert ForestTrainer({**SMALL_PARAMS, "n_estimators": 25.0}).params == SMALL_PARAMS


# ----------------------------------------------------------------------
# where parameters enter: explicit, artifact, tuned, specialization
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw, message", [
    ({**SMALL_PARAMS, "max_dept": 6}, "unknown keys"),                       # was: dropped
    ({**SMALL_PARAMS, "min_samples_leaf": 7.5}, "min_samples_leaf"),
    ({"n_estimators": 300, "min_samples_leaf": 7, "max_dept": 6}, "unknown keys \\['max_dept'\\]"),
    ({"n_estimators": 300, "min_samples_leaf": 7, "max_features_type": "fraction"},
     "needs max_features_fraction"),                                          # was: None
    ({"n_estimators": 300, "min_samples_leaf": 7, "use_max_depth": True}, "needs max_depth"),
    ({"n_estimators": 300, "min_samples_leaf": 7, "max_features_type": "auto"}, "max_features"),
])
def test_normalise_refuses_what_it_used_to_drop_or_coerce_f1(raw, message):
    with pytest.raises(ValueError, match=message):
        normalise_rf_params(raw)


def test_normalise_keeps_the_documented_raw_optuna_defaults():
    assert normalise_rf_params({"n_estimators": 300, "min_samples_leaf": 7}) == {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 7, "max_features": "sqrt",
        "minority_weight": 1.0, "class_weight": "balanced"}
    raw = {"n_estimators": 300, "min_samples_leaf": 7, "use_max_depth": False, "max_depth": 6}
    assert normalise_rf_params(raw)["max_depth"] is None                    # Optuna's own semantics


def test_every_optuna_draw_is_valid_and_normalises_as_before():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=4))
    study.optimize(lambda t: float(suggest_rf_params(t)["minority_weight"]), n_trials=60)
    for t in study.trials:
        drawn = suggest_rf_params(optuna.trial.FixedTrial(t.params))
        assert validate_rf_params(drawn) == drawn
        assert normalise_rf_params(t.params) == _legacy_normalise(t.params) == drawn
    assert {t.params["max_features_type"] for t in study.trials} == {"sqrt", "log2", "fraction"}


def test_the_specialization_search_checks_its_fixed_values_f1():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for fixed in ({"n_estimators": 500.5}, {"n_estimators": 0}, {"max_depth": True},
                  {"minority_weight": -1.0}):
        with pytest.raises(ValueError, match="fixed"):
            suggest_rf_params(study.ask(), fixed=fixed)
    p = suggest_rf_params(study.ask(), fixed={"n_estimators": 500.0, "max_depth": None})
    assert p["n_estimators"] == 500 and type(p["n_estimators"]) is int


def test_resolve_refuses_an_explicit_or_artifact_dict_with_an_unknown_key_f1(tmp_path):
    import joblib
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_rf_params("explicit", split_fp="FP", explicit={**SMALL_PARAMS, "max_dept": 6},
                          verbose=False)
    joblib.dump({"split_fingerprint": "FP", "params": {**SMALL_PARAMS, "max_dept": 6}},
                tmp_path / "rf_result_1.joblib")
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_rf_params("artifact", split_fp="FP", artifact_dirs=[tmp_path], verbose=False)


def test_the_notebooks_explicit_and_default_params_are_valid():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    src = next("".join(c["source"]) for c in cells
               if c["cell_type"] == "code" and "RF_PARAMS_EXPLICIT = {" in "".join(c["source"]))
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "RF_PARAMS_EXPLICIT")
    explicit = ast.literal_eval(node.value)
    assert normalise_rf_params(explicit) == explicit == validate_rf_params(explicit)
    assert validate_rf_params(DEFAULT_RF_PARAMS) == DEFAULT_RF_PARAMS


def test_check_rf_param_names_the_failing_field():
    with pytest.raises(ValueError, match="here.n_estimators must be an integer"):
        check_rf_param("n_estimators", 2.5, "here")


# ----------------------------------------------------------------------
# recorded == applied, for each pass
# ----------------------------------------------------------------------
def test_each_pass_fits_and_saves_exactly_its_configured_params(tmp_path, monkeypatch):
    weights = []
    real = forest_mod.make_sample_weights
    monkeypatch.setattr(forest_mod, "make_sample_weights",
                        lambda y, w: weights.append(w) or real(y, w))
    X, y = make_data()
    cfg = router_config(pass2_rf_params={**ALT_PARAMS, "max_depth": 4})
    router = RFRouter(cfg).fit(X, y, "FP")
    for model, params in ((router.pass1_model_, cfg.rf_params),
                          (router.pass2_model_, cfg.effective_pass2_params)):
        got = model.named_steps["clf"].get_params()
        assert {k: got[k] for k in KEYS if k != "minority_weight"} == \
            {k: params[k] for k in KEYS if k != "minority_weight"}
    assert set(weights) == {SMALL_PARAMS["minority_weight"], ALT_PARAMS["minority_weight"]}
    path = quiet(save_rf_router, router, tmp_path, name="rf_router_two_pass")
    saved = load_rf_router(path, expected_split_fingerprint="FP")["config"]
    assert (saved["rf_params"], saved["pass2_rf_params"]) == (cfg.rf_params, cfg.pass2_rf_params)
    assert RFRouterConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).to_dict() == \
        json.loads(json.dumps(cfg.to_dict()))
