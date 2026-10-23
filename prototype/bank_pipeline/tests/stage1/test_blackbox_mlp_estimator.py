"""
Black-box Stage 1 MLP: the estimator, its configuration and its search space
(``blackbox_pipeline.models.mlp`` — ``estimator``, ``config``, ``tuning``).

Contract: ``Stage1MLPClassifier`` is a clone-safe scikit-learn classifier on
binary 0/1 labels (anything else is refused, not truncated: audit F34) that
returns finite ``[P(y=0), P(y=1)]`` rows; positives are oversampled only in the
fit's training part, never into its early-stopping rows; early stopping keeps
the best epoch's weights and stops exactly ``patience`` epochs after it; a seed
fixes the fit. ``Stage1MLPConfig`` refuses invalid settings, all at once, and
survives its JSON metadata. The Optuna search space is the documented one and
maps onto the estimator exactly.

Stage-level contracts (out-of-fold integrity, thresholds, artifacts, Stage 4)
are in test_blackbox_mlp_stage.py; fold isolation and calibration, shared with
GLASS LR, in test_stage1_fold_isolation.py and test_stage1_calibration.py.

Inputs here are small dense arrays with continuous features, so every row is
unique and can be identified by its bytes.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    CategoricalDistribution, FloatDistribution, IntDistribution,
)
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

import blackbox_pipeline.models.mlp.estimator as est_mod
from blackbox_pipeline.models.mlp import (
    CalibratedStage1MLP, Stage1MLPClassifier, Stage1MLPConfig, oversample_positives,
    params_to_kwargs, sample_params, tune_stage1_mlp_auc,
)

SEED = 1
#: Settings that early-stop well before ``max_epochs`` on ``_data(SEED)``.
FAST = dict(hidden_layer_sizes=(32,), learning_rate_init=0.05, alpha=1e-6, batch_size=32,
            max_epochs=60, patience=4, random_state=SEED)
TINY = dict(hidden_layer_sizes=(4,), max_epochs=3, patience=2, batch_size=64, random_state=SEED)


def _data(seed: int = SEED, n: int = 300):
    """Three continuous features, ~30% positives, a learnable signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = ((1.5 * X[:, 0] - X[:, 1] + rng.logistic(size=n)) > 1.2).astype(int)
    return X, y


def _rows(A) -> set:
    return {np.ascontiguousarray(r).tobytes() for r in np.asarray(A, dtype=float)}


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    return Stage1MLPClassifier(**FAST).fit(X, y)


# ----------------------------------------------------------------------
# scikit-learn estimator contract
# ----------------------------------------------------------------------
def test_constructor_stores_parameters_unchanged_and_clones_unfitted(fitted):
    params = Stage1MLPClassifier(**FAST).get_params()
    assert {k: params[k] for k in FAST} == FAST
    twin = clone(fitted)
    assert twin.get_params() == fitted.get_params()
    assert not hasattr(twin, "net_")                    # cross_val_predict / calibrate_oof refit it
    assert twin.set_params(patience=7).patience == 7


def test_fit_sets_the_documented_attributes(fitted, data):
    X, _ = data
    np.testing.assert_array_equal(fitted.classes_, [0, 1])
    assert fitted.n_features_in_ == X.shape[1]
    assert list(fitted.history_.columns) == ["epoch", "train_loss", "val_roc_auc"]
    assert fitted.history_["epoch"].tolist() == list(range(1, len(fitted.history_) + 1))
    assert isinstance(fitted.net_, MLPClassifier)


@pytest.mark.parametrize("hidden, expected", [((4,), 3 * 4 + 4 + 4 + 1),
                                              ((4, 2), 3 * 4 + 4 + 4 * 2 + 2 + 2 + 1)])
def test_trainable_parameter_count(data, hidden, expected):
    X, y = data
    model = Stage1MLPClassifier(**{**TINY, "hidden_layer_sizes": hidden}).fit(X, y)
    assert model.n_trainable_params_ == expected


def test_probabilities_are_finite_rows_summing_to_one_and_predict_cuts_at_half(fitted, data):
    X, _ = data
    p = fitted.predict_proba(X)
    assert p.shape == (len(X), 2) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_array_equal(fitted.predict(X), (p[:, 1] >= 0.5).astype(int))


def test_the_mlp_offers_probabilities_only(fitted):
    """The shared calibration scores models by ``decision_function`` when one
    exists (LR) and by ``predict_proba[:, 1]`` otherwise: the MLP takes the
    second path."""
    from shared.stage1.oof_calibration import base_scores
    assert not hasattr(fitted, "decision_function")
    X, _ = _data(seed=9, n=20)
    np.testing.assert_array_equal(base_scores(fitted, X), fitted.predict_proba(X)[:, 1])


def test_unfitted_and_wrong_width_calls_are_refused(fitted, data):
    X, _ = data
    with pytest.raises(NotFittedError):
        Stage1MLPClassifier().predict_proba(X)
    with pytest.raises(NotFittedError):
        _ = Stage1MLPClassifier().n_trainable_params_
    with pytest.raises(ValueError, match="Expected 3 features, got 2"):
        fitted.predict_proba(X[:, :2])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_features_are_refused_not_scored(fitted, data, value):
    X, y = data
    bad = X.copy()
    bad[0, 0] = value
    with pytest.raises(ValueError):
        fitted.predict_proba(bad)
    with pytest.raises(ValueError):
        Stage1MLPClassifier(**TINY).fit(bad, y)


# ----------------------------------------------------------------------
# labels (audit F34)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("encode", [
    lambda y: y.astype("int8"),
    lambda y: y.astype(bool),
    lambda y: y.astype(float),
], ids=["int8", "bool", "float_0_1"])
def test_binary_label_encodings_fit_the_same_model(data, encode):
    X, y = data
    reference = Stage1MLPClassifier(**TINY).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(Stage1MLPClassifier(**TINY).fit(X, encode(y)).predict_proba(X),
                                  reference)


@pytest.mark.parametrize("case", ["fractional", "nan", "one_two", "minus_one_one"])
def test_non_binary_labels_are_refused_not_truncated_f34(data, case):
    """Before F34 ``astype(int)`` turned 0.6 into 0 and trained on it."""
    X, y = data
    if case == "fractional":
        bad = y.astype(float)
        bad[:5] = 0.6
    elif case == "nan":
        bad = y.astype(float)
        bad[0] = np.nan
    elif case == "one_two":
        bad = y + 1
    else:
        bad = 2 * y - 1
    with pytest.raises(ValueError, match="binary 0/1"):
        Stage1MLPClassifier(**TINY).fit(X, bad)


def test_the_tuner_refuses_non_binary_labels_f34(data):
    X, y = data
    bad = y.astype(float)
    bad[:5] = 0.6
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    fixed = dict(max_epochs=2, patience=1, val_fraction=0.15, random_state=0)
    with pytest.raises(ValueError, match="binary 0/1"):
        tune_stage1_mlp_auc(X, bad, cv, 1, fixed, 0, verbose=False)


def test_the_tuner_refuses_non_binary_fold_labels_f34(data):
    import pandas as pd
    from shared.stage1.crossfit import build_fold_designs
    X, y = data
    Xf, ys = pd.DataFrame(X, columns=list("abc")), pd.Series(y)
    designs = build_fold_designs(Xf, ys, cv_folds=3, random_state=0)
    designs[0].y_val = designs[0].y_val.astype(float)
    designs[0].y_val[0] = 0.6
    fixed = dict(max_epochs=2, patience=1, val_fraction=0.15, random_state=0)
    with pytest.raises(ValueError, match="binary 0/1"):
        tune_stage1_mlp_auc(X, y, None, 1, fixed, 0, verbose=False, folds=designs)


@pytest.mark.parametrize("case, message", [("one_class", "both classes"),
                                           ("length", "length mismatch")])
def test_degenerate_training_data_is_refused(data, case, message):
    X, y = data
    if case == "one_class":
        y = np.zeros_like(y)
    else:
        y = y[:-1]
    with pytest.raises(ValueError, match=message):
        Stage1MLPClassifier(**TINY).fit(X, y)


# ----------------------------------------------------------------------
# hyperparameters
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    dict(activation="bogus"), dict(alpha=-1.0), dict(learning_rate_init=0.0),
    dict(hidden_layer_sizes=(0,)), dict(batch_size=0), dict(val_fraction=0.0),
    dict(val_fraction=1.0),
], ids=lambda d: next(iter(d)) + "=" + repr(next(iter(d.values()))))
def test_invalid_hyperparameters_fail_when_fitting(data, bad):
    """Validation is delegated to scikit-learn and fires at ``fit`` (the
    constructor stores parameters untouched, as ``clone`` requires)."""
    X, y = data
    model = Stage1MLPClassifier(**{**TINY, **bad})
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_zero_epochs_cannot_produce_a_model(data):
    X, y = data
    with pytest.raises(RuntimeError, match="no best epoch"):
        Stage1MLPClassifier(max_epochs=0).fit(X, y)


def test_a_seed_fixes_the_fit_and_another_seed_changes_it(fitted, data):
    X, y = data
    again = Stage1MLPClassifier(**FAST).fit(X, y)
    np.testing.assert_array_equal(again.predict_proba(X), fitted.predict_proba(X))
    assert again.history_.equals(fitted.history_)
    other = Stage1MLPClassifier(**{**FAST, "random_state": SEED + 1}).fit(X, y)
    assert not np.allclose(other.predict_proba(X), fitted.predict_proba(X))


# ----------------------------------------------------------------------
# oversampling
# ----------------------------------------------------------------------
def test_oversampling_appends_copies_of_positives_only(data):
    X, y = data
    X_in, y_in = X.copy(), y.copy()
    X_out, y_out = oversample_positives(X, y, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(X, X_in)                               # inputs untouched
    np.testing.assert_array_equal(y, y_in)
    np.testing.assert_array_equal(X_out[:len(X)], X)                     # originals first, in order
    extra_X, extra_y = X_out[len(X):], y_out[len(X):]
    assert (extra_y == 1).all() and _rows(extra_X) <= _rows(X[y == 1])
    assert y_out.sum() == (y_out == 0).sum() == (y == 0).sum()            # ratio 1.0 reached
    again = oversample_positives(X, y, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(again[0], X_out)                       # rng fixes the draw


@pytest.mark.parametrize("ratio, y_kind", [(0.1, "signal"), (-1.0, "signal"), (1.0, "no_positives")])
def test_oversampling_is_a_no_op_when_there_is_nothing_to_add(data, ratio, y_kind):
    X, y = data
    if y_kind == "no_positives":
        y = np.zeros_like(y)
    X_out, y_out = oversample_positives(X, y, ratio, np.random.default_rng(0))
    assert X_out is X and y_out is y


def test_oversampled_rows_never_reach_the_early_stopping_rows(data, monkeypatch):
    """The inner split happens first; duplicates are drawn from its training
    part only, so no copy of a row can sit on both sides."""
    X, y = data
    seen = {}
    real = est_mod.oversample_positives

    def spy(X_fit, y_fit, ratio, rng):
        seen["fit"], seen["y_fit"] = X_fit.copy(), y_fit.copy()
        seen["out"], seen["y_out"] = real(X_fit, y_fit, ratio, rng)
        return seen["out"], seen["y_out"]

    monkeypatch.setattr(est_mod, "oversample_positives", spy)
    Stage1MLPClassifier(**{**TINY, "pos_ratio": 1.0, "val_fraction": 0.2}).fit(X, y)
    fit_rows, out_rows = _rows(seen["fit"]), _rows(seen["out"])
    val_rows = _rows(X) - fit_rows
    assert len(val_rows) == round(0.2 * len(X)) and len(fit_rows) + len(val_rows) == len(X)
    assert out_rows <= fit_rows and not out_rows & val_rows
    assert len(seen["out"]) > len(seen["fit"])                            # something was added
    assert seen["y_out"].sum() == (seen["y_fit"] == 0).sum()


# ----------------------------------------------------------------------
# early stopping
# ----------------------------------------------------------------------
def test_early_stopping_keeps_the_best_epoch_and_stops_patience_epochs_later(data, monkeypatch):
    X, y = data
    snapshots = []

    class SnapshotMLP(MLPClassifier):
        def partial_fit(self, X, y, classes=None):
            out = super().partial_fit(X, y, classes=classes)
            snapshots.append([w.copy() for w in self.coefs_])
            return out

    monkeypatch.setattr(est_mod, "MLPClassifier", SnapshotMLP)
    model = Stage1MLPClassifier(**FAST).fit(X, y)
    h, best = model.history_, model.best_epoch_

    assert len(snapshots) == len(h) == min(FAST["max_epochs"], best + FAST["patience"])
    assert best < len(h)                                  # it did stop early, so the check can fail
    for kept, at_best in zip(model.net_.coefs_, snapshots[best - 1]):
        np.testing.assert_array_equal(kept, at_best)
    assert any(not np.array_equal(a, b) for a, b in zip(snapshots[best - 1], snapshots[-1]))

    assert model.best_val_roc_auc_ == h["val_roc_auc"].iloc[best - 1]
    assert model.best_val_roc_auc_ >= h["val_roc_auc"].max() - 1e-4      # improvement margin
    _, X_val, _, y_val = train_test_split(X, y, test_size=FAST.get("val_fraction", 0.15),
                                          stratify=y, random_state=FAST["random_state"])
    assert roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) == pytest.approx(
        model.best_val_roc_auc_)                          # the restored net scores like the best epoch


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def test_defaults_are_the_notebook_protocol():
    cfg = Stage1MLPConfig()
    assert (cfg.calibration_method, cfg.cv_folds, cfg.n_trials, cfg.random_state) == ("auto", 10, 100, 42)
    assert (cfg.threshold_beta, cfg.threshold_grid, cfg.strict_oof) == (2.0, (0.05, 0.50, 0.01), True)
    stage = CalibratedStage1MLP()
    assert stage.config == cfg and stage.cv_folds == 10 and stage.calibration_method == "auto"
    fixed = cfg.estimator_fixed_kwargs
    assert fixed == {"max_epochs": 200, "patience": 15, "val_fraction": 0.15, "random_state": 42}
    defaults = Stage1MLPClassifier().get_params()
    assert {k: defaults[k] for k in fixed} == fixed                     # stage and estimator agree


@pytest.mark.parametrize("field, value", [
    ("calibration_method", "temperature"), ("cv_folds", 1), ("n_trials", 0),
    ("val_fraction", 0.0), ("val_fraction", 0.5), ("max_epochs", 0), ("patience", 0),
    ("threshold_beta", 0.0), ("threshold_grid", (0.5, 0.05, 0.01)),
    ("threshold_grid", (0.05, 1.2, 0.01)), ("threshold_grid", (0.05, 0.5, 0.0)),
])
def test_invalid_config_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        Stage1MLPConfig(**{field: value})


def test_every_config_error_is_reported_at_once():
    with pytest.raises(ValueError) as err:
        Stage1MLPConfig(cv_folds=1, n_trials=0, patience=0)
    assert all(f in str(err.value) for f in ("cv_folds", "n_trials", "patience"))


def test_unknown_stage_settings_are_refused():
    with pytest.raises(TypeError):
        CalibratedStage1MLP(cv_folds=10, hidden_units=8)


def test_config_survives_its_json_metadata():
    cfg = Stage1MLPConfig(cv_folds=5, threshold_grid=(0.1, 0.4, 0.05), strict_oof=True, verbose=False)
    restored = Stage1MLPConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored == cfg and isinstance(restored.threshold_grid, tuple)
    assert {f.name for f in dataclasses.fields(cfg)} == set(cfg.to_dict())


# ----------------------------------------------------------------------
# search space
# ----------------------------------------------------------------------
SEARCH_SPACE = {
    "n_layers": IntDistribution(1, 2),
    "width": CategoricalDistribution([4, 8, 16, 32]),
    "activation": CategoricalDistribution(["relu", "tanh"]),
    "alpha": FloatDistribution(1e-6, 1e-1, log=True),
    "learning_rate_init": FloatDistribution(3e-4, 1e-2, log=True),
    "batch_size": CategoricalDistribution([128, 256, 512]),
    "pos_ratio": FloatDistribution(0.15, 1.0),
}


def test_the_search_space_is_the_documented_one():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda t: params_to_kwargs(sample_params(t))["alpha"], n_trials=25)
    assert study.trials[0].distributions == SEARCH_SPACE
    for t in study.trials:
        kwargs = params_to_kwargs(t.params, max_epochs=1)
        assert Stage1MLPClassifier(**kwargs).get_params()["pos_ratio"] == t.params["pos_ratio"]


@pytest.mark.parametrize("n_layers, width, hidden", [(1, 4, (4,)), (2, 4, (4, 2)), (2, 32, (32, 16))])
def test_params_map_onto_the_estimator(n_layers, width, hidden):
    params = {"n_layers": n_layers, "width": width, "activation": "tanh", "alpha": 1e-3,
              "learning_rate_init": 1e-3, "batch_size": 256, "pos_ratio": 0.5}
    fixed = Stage1MLPConfig().estimator_fixed_kwargs
    kwargs = params_to_kwargs(params, **fixed)
    assert kwargs == {"hidden_layer_sizes": hidden, "activation": "tanh", "alpha": 1e-3,
                      "learning_rate_init": 1e-3, "batch_size": 256, "pos_ratio": 0.5, **fixed}
    assert Stage1MLPClassifier(**kwargs).get_params() == {**Stage1MLPClassifier().get_params(), **kwargs}
    with pytest.raises(KeyError):
        params_to_kwargs({k: v for k, v in params.items() if k != "width"})
