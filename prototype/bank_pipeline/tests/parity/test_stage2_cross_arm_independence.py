"""
Stage 2 cross-arm independence: fitted state never crosses arms (PR 37 Commit 3).

Traced (Step 3A): ``glass_rf_model="fold_rf"`` does NOT reuse the black-box
arm's forest. ``crossfit_stage2`` fits its own fold RF from ``rf_params``
(``pipeline/crossfit.py``), and the black-box arm takes only the hyperparameter
dict from the GLASS RF artifact (``tuning/legacy.load_rf_params_from_artifact``).
Sharing training rows, an algorithm or hyperparameters is allowed; sharing a
fitted estimator is not. These tests pin the boundaries where a fitted object
could leak across:

1. the only artifact channel from the GLASS arm into the black-box arm returns
   plain hyperparameters, even when the artifact also holds a fitted model;
2. neither arm's cross-fit accepts a fitted estimator from outside: GLASS's
   switch is a string, and the RF Router's params are exactly six
   hyperparameters;
3. each GLASS fold gets a fresh RF fitted on its own training rows only (the
   opt-in nested rankers are covered in test_glass_inner_feature_ranker.py).

The constraint channel (``match_glass_oof``) is covered by
tests/stage2/test_stage2_operating_contract_support.py.
"""

from __future__ import annotations

import numbers

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import glass_pipeline.glass_router.pipeline.crossfit as crossfit_mod
from blackbox_pipeline.models.rf_router import RFRouterConfig
from blackbox_pipeline.models.rf_router.tuning.legacy import load_rf_params_from_artifact
from glass_pipeline.glass_router.pipeline import crossfit_stage2
from tests.stage2.conftest import TINY_RF, legacy_config
from tests.support.stage2_synthetic import CONSTRAINTS_OK, SMALL_PARAMS, make_data
from tests.support.synthetic import make_bank_frame, split_frame


def test_the_glass_rf_artifact_hands_the_black_box_arm_hyperparameters_only(tmp_path):
    X, y = make_data(n=300, seed=1)
    fitted = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    joblib.dump({"split_fingerprint": "fp", "params": dict(SMALL_PARAMS), "model": fitted,
                 "pipe": fitted}, tmp_path / "rf_result_x.joblib")
    params, origin = load_rf_params_from_artifact([tmp_path], "fp")
    assert params == SMALL_PARAMS and origin.endswith("rf_result_x.joblib")
    assert all(v is None or isinstance(v, (str, numbers.Number)) for v in params.values())


@pytest.mark.parametrize("smuggled", ["as glass_rf_model", "as rf_params"])
def test_glass_crossfit_accepts_no_fitted_estimator_from_outside(smuggled):
    split = split_frame(make_bank_frame(n=200))
    X, y = make_data(n=200, seed=2)
    forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    kw = dict(glass_rf_model=forest) if smuggled == "as glass_rf_model" else {}
    rf_params = forest if smuggled == "as rf_params" else TINY_RF
    with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
        crossfit_stage2(split["X_train"], split["y_train"], legacy_config(), rf_params, **kw)


@pytest.mark.parametrize("params", [
    lambda f: {**SMALL_PARAMS, "model": f},          # an estimator riding along in the dict
    lambda f: f,                                     # an estimator instead of the dict
])
def test_the_rf_router_accepts_hyperparameters_not_a_fitted_forest(params):
    X, y = make_data(n=200, seed=2)
    forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        RFRouterConfig(mode="two_pass", constraints=CONSTRAINTS_OK, rf_params=params(forest))


def test_each_glass_fold_rf_is_fresh_and_fitted_on_its_own_training_rows():
    split = split_frame(make_bank_frame(n=400))
    seen = []

    class Capture:
        def __init__(self, config, rf_model=None, **kwargs):
            seen.append(rf_model)
            self.pass1_rules = self.pass2_rules = []

        def fit(self, X, y, X_val=None, y_val=None):
            self.rows, self.training_base_rate = frozenset(X.index), float(y.mean())
            seen[-1] = (seen[-1], self.rows)
            return self

        def predict(self, X):
            n = len(X)
            return np.full(n, -1), np.zeros(n), np.array(["uncertain"] * n, dtype=object)

        def predict_proba(self, X):
            return np.tile([1 - self.training_base_rate, self.training_base_rate], (len(X), 1))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(crossfit_mod, "GlassRouterPipeline", Capture)
        out = crossfit_stage2(split["X_train"], split["y_train"], legacy_config(), TINY_RF,
                              random_state=42)
    models = [m for m, _ in seen]
    assert len(models) == 5 and len({id(m) for m in models}) == 5
    for (model, rows), k in zip(seen, range(5)):
        assert isinstance(model, RandomForestClassifier) and model.n_features_in_ > 0
        held = frozenset(split["X_train"].index[out.fold_id == k])
        assert rows == frozenset(split["X_train"].index) - held
