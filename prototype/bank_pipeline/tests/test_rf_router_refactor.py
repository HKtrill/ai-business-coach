"""
PR 35 commit A — notebook-consolidation regression tests.

The Stage 2 cells moved into modules must behave exactly as the cells did.
These tests pin that: the search-space suggest order (so a seeded baseline tune
reproduces), the artifact lookup, the one-pass predict, the router's fold plan,
and the outer cross-fit's partition and OOF guarantees.
"""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pytest

from blackbox_pipeline.models.rf_router import (
    RFRouter,
    SingleRFRouter,
    crossfit_rf_router,
    find_artifact_for_split,
    make_folds,
)
from blackbox_pipeline.models.rf_router.evaluation.router_metrics import evaluate_router
from blackbox_pipeline.models.rf_router.reporting import (
    binding_after,
    pass_agreement,
    plot_threshold_sweeps,
    train_oof_summary,
)
from blackbox_pipeline.models.rf_router.tuning import (
    REQUIRED_PARAM_KEYS,
    normalise_rf_params,
    resolve_rf_params,
    suggest_rf_params,
)

try:
    from stage2_synthetic import (
        CONSTRAINTS_DEGENERATE, CONSTRAINTS_OK, N_FOLDS, ROUTER_SEED, SMALL_PARAMS,
        make_data, router_config,
    )
except ImportError:  # tests/ is a package in some layouts
    from .stage2_synthetic import (  # type: ignore[no-redef]
        CONSTRAINTS_DEGENERATE, CONSTRAINTS_OK, N_FOLDS, ROUTER_SEED, SMALL_PARAMS,
        make_data, router_config,
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def router(data):
    X, y = data
    return RFRouter(router_config()).fit(X, y, "fp-test")


# ----------------------------------------------------------------------
# search space / params
# ----------------------------------------------------------------------
def _cell22_suggest(trial):
    """Verbatim copy of the former Cell 22 suggest block — the reference."""
    mf_type = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "fraction"])
    use_depth = trial.suggest_categorical("use_max_depth", [True, False])
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=50),
        max_depth=(trial.suggest_int("max_depth", 2, 12) if use_depth else None),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        max_features=(trial.suggest_float("max_features_fraction", 0.3, 1.0)
                      if mf_type == "fraction" else mf_type),
    )
    mw = trial.suggest_float("minority_weight", 1.0, 3.0)
    return params, mw


def _score(params, mw):
    mf = params["max_features"]
    mf_val = mf if isinstance(mf, float) else {"sqrt": 0.1, "log2": 0.2}[mf]
    return (params["n_estimators"] / 1000 + params["min_samples_leaf"] / 50
            + (params["max_depth"] or 0) / 12 + mf_val + mw)


def test_suggest_order_reproduces_cell22_seeded_study():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ref = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
    ref.optimize(lambda t: _score(*_cell22_suggest(t)), n_trials=25)

    def new_obj(t):
        p = suggest_rf_params(t)
        return _score({k: p[k] for k in ("n_estimators", "max_depth",
                                         "min_samples_leaf", "max_features")},
                      p["minority_weight"])

    new = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
    new.optimize(new_obj, n_trials=25)

    assert [t.params for t in ref.trials] == [t.params for t in new.trials]
    assert normalise_rf_params(ref.best_params) == normalise_rf_params(new.best_params)


def test_suggest_fixed_keys_are_not_searched():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    p = suggest_rf_params(trial, fixed={"n_estimators": 500, "max_depth": None})
    assert p["n_estimators"] == 500 and p["max_depth"] is None
    assert "n_estimators" not in trial.params and "use_max_depth" not in trial.params
    assert tuple(p) == REQUIRED_PARAM_KEYS
    with pytest.raises(ValueError):
        suggest_rf_params(study.ask(), fixed={"bogus": 1})


def test_normalise_raw_and_passthrough():
    raw = {"max_features_type": "fraction", "max_features_fraction": 0.5,
           "use_max_depth": True, "max_depth": 6, "n_estimators": 300,
           "min_samples_leaf": 7, "minority_weight": 1.5}
    p = normalise_rf_params(raw)
    assert p == {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 7,
                 "max_features": 0.5, "minority_weight": 1.5, "class_weight": "balanced"}
    assert normalise_rf_params(p) == p
    assert normalise_rf_params({**raw, "use_max_depth": False})["max_depth"] is None


# ----------------------------------------------------------------------
# artifact lookup / param resolution
# ----------------------------------------------------------------------
def test_find_artifact_for_split_newest_matching(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(), d2.mkdir()
    now = time.time()

    def put(path, obj, age):
        joblib.dump(obj, path)
        os.utime(path, (now - age, now - age))

    put(d1 / "rf_result_old.joblib", {"split_fingerprint": "FP", "v": "old"}, 300)
    put(d2 / "rf_result_new.joblib", {"split_fingerprint": "FP", "v": "new"}, 100)
    put(d2 / "rf_result_other.joblib", {"split_fingerprint": "OTHER", "v": "x"}, 10)
    (d1 / "rf_result_broken.joblib").write_bytes(b"not a pickle")
    os.utime(d1 / "rf_result_broken.joblib", (now, now))

    obj, path = find_artifact_for_split([d1, d2, tmp_path / "missing"],
                                        "rf_result_*.joblib", "FP", verbose=False)
    assert obj["v"] == "new" and path.name == "rf_result_new.joblib"
    assert find_artifact_for_split([d1], "rf_result_*.joblib", "NOPE",
                                   verbose=False) == (None, None)


def test_resolve_rf_params_artifact_and_explicit(tmp_path):
    joblib.dump({"split_fingerprint": "FP", "params": dict(SMALL_PARAMS)},
                tmp_path / "rf_result_1.joblib")
    p, origin = resolve_rf_params("artifact", split_fp="FP",
                                  artifact_dirs=[tmp_path], verbose=False)
    assert p == SMALL_PARAMS and origin.endswith("rf_result_1.joblib")

    p2, origin2 = resolve_rf_params("explicit", split_fp="FP",
                                    explicit=SMALL_PARAMS, verbose=False)
    assert p2 == SMALL_PARAMS and origin2 == "RF_PARAMS_EXPLICIT"

    with pytest.raises(FileNotFoundError):
        resolve_rf_params("artifact", split_fp="OTHER", artifact_dirs=[tmp_path],
                          verbose=False)


def test_tune_rf_train_only_returns_best_trial_params(monkeypatch, data):
    pytest.importorskip("optuna")
    from blackbox_pipeline.models.rf_router.tuning import legacy

    # The CV loop is covered by the harness cross-check; here only the study
    # plumbing is under test, so stub the expensive objective.
    monkeypatch.setattr(legacy, "cv_auc_legacy",
                        lambda X, y, params, random_state: params["minority_weight"])
    X, y = data
    best = legacy.tune_rf_train_only(X, y, n_trials=6, random_state=0,
                                     show_progress_bar=False)
    assert tuple(best) == REQUIRED_PARAM_KEYS
    assert 1.0 <= best["minority_weight"] <= 3.0


# ----------------------------------------------------------------------
# router additions
# ----------------------------------------------------------------------
def test_predict_with_proba_matches_separate_calls(router, data):
    X, _ = data
    p, c, d, proba = router.predict_with_proba(X)
    p2, c2, d2 = router.predict(X)
    np.testing.assert_array_equal(p, p2)
    np.testing.assert_array_equal(c, c2)
    np.testing.assert_array_equal(d, d2)
    np.testing.assert_allclose(proba, router.predict_proba(X))


def test_router_fold_ids_are_the_shared_fold_plan(router, data):
    X, y = data
    stage_fold_id = np.full(len(y), -1)
    for k, (_, va) in enumerate(make_folds(y, N_FOLDS, ROUTER_SEED)):
        stage_fold_id[va] = k
    np.testing.assert_array_equal(router.oof_pass1_fold_id_, stage_fold_id)

    single = SingleRFRouter(router_config(
        constraints=CONSTRAINTS_OK.__class__(**{**CONSTRAINTS_OK.to_dict(),
                                                "require_band_ordering": False}),
        mode="single_rf",
    )).fit(X, y, "fp-test")
    np.testing.assert_array_equal(single.oof_fold_id_, stage_fold_id)


# ----------------------------------------------------------------------
# outer cross-fit
# ----------------------------------------------------------------------
def test_crossfit_partitions_rows_and_never_trains_on_held_out(data):
    X, y = data
    folds = make_folds(y, N_FOLDS, ROUTER_SEED)
    seen = []

    def engineer(X_tr, X_va, y_tr):
        seen.append((set(X_tr.index), set(X_va.index), set(y_tr.index)))
        return X_tr, X_va  # synthetic features are already binary bins

    res = crossfit_rf_router(
        X, y, folds, engineer=engineer, make_config=lambda: router_config(),
        split_fp="fp-test", fold_fp="fold-fp", random_state=ROUTER_SEED,
        verbose=False,
    )
    for (tr_idx, va_idx, ytr_idx), (_, va) in zip(seen, folds):
        assert not (tr_idx & va_idx) and tr_idx == ytr_idx
        assert va_idx == set(X.index[va])

    m = evaluate_router(res.oof["pred"], res.oof["confidence"],
                        res.oof["decisions"], y, label="cf")  # contract check
    assert m["n_samples"] == len(y)
    np.testing.assert_allclose(res.oof["proba"].sum(axis=1), 1.0)
    assert list(res.folds["fold"]) == list(range(N_FOLDS))
    assert res.protocol["fold_fingerprint"] == "fold-fp"
    assert res.protocol["train_outputs_oof"] is True


def test_crossfit_rejects_misaligned_inputs(data):
    X, y = data
    with pytest.raises(AssertionError):
        crossfit_rf_router(X, y.sort_index(), make_folds(y, N_FOLDS, 0),
                           engineer=lambda a, b, c: (a, b),
                           make_config=lambda: router_config(),
                           split_fp="s", fold_fp="f", random_state=0, verbose=False)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def test_reporting_helpers(router, data):
    import matplotlib
    matplotlib.use("Agg")

    table = train_oof_summary({"two_pass": router})
    assert table.loc["pass2 band frac", "two_pass"] == router.thresholds_.pass2.band_fraction
    agree = pass_agreement(router)
    assert 0.0 <= agree["agreement"] <= 1.0
    fig = plot_threshold_sweeps(router, CONSTRAINTS_OK)
    assert len(fig.axes) == 2

    X, y = data
    degenerate = RFRouter(router_config(constraints=CONSTRAINTS_DEGENERATE)).fit(X, y, "fp")
    assert degenerate.thresholds_.pass2.band_fraction == 1.0
    assert binding_after(degenerate.thresholds_.pass2.sweep,
                         degenerate.thresholds_.pass2.band_size).startswith("none")
