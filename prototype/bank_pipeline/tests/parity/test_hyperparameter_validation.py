"""
Cross-pipeline parity: hyperparameter validation (PR 36 audit F1).

Equivalent safeguards, not identical search spaces. For every arm whose
hyperparameters can be supplied from outside its own search (explicit,
reused or restored values):

* the arm's normalizer accepts exactly its tuned keys, so an unknown,
  misspelled or missing key is refused and can never be recorded as applied;
* a valid set comes back unchanged.

The black-box arms (MLP, RF Router, XGBoost, Meta-XGB) also refuse a
non-integral integer and a non-finite number instead of coercing them.

Known difference, not asserted here: the GLASS EBM coerces values with
``int()`` / ``float()`` (Commit 2 pins ``"max_bins": "32"`` -> 32 in
test_glass_ebm_estimator.py), so a fractional ``max_rounds`` is truncated;
the truncated value is what it records and applies. The GLASS LR has no
outside entry point: its parameters come only from its own Optuna study
(``C``, ``penalty``, ``l1_ratio``) or the fixed default, so it has nothing
to validate.

Per-arm detail: tests/stage1/test_blackbox_mlp_params.py,
tests/stage2/test_rf_router_params.py, tests/stage3/test_*_estimator.py,
tests/stage4/test_meta_xgb_estimator.py.
"""

from __future__ import annotations

import inspect
import math
from typing import Callable, NamedTuple

import pytest

import blackbox_pipeline.models.mlp.tuning as mlp_tuning
import blackbox_pipeline.models.rf_router.config as rf_config
from blackbox_pipeline.models.meta_xgb import MetaXGBEstimator
from blackbox_pipeline.models.xgb import XGBEstimator
from glass_pipeline.ebm.estimator import EBMFactory
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from tests.stage3.conftest import ARMS, EBM_TEST_PARAMS
from tests.support.stage2_synthetic import SMALL_PARAMS


class Arm(NamedTuple):
    normalize: Callable[[dict], dict]
    valid: dict
    integer_key: str           # set to a fraction -> refused (black-box arms)
    float_key: str             # set to NaN -> refused (black-box arms)


MLP_PARAMS = {"n_layers": 2, "width": 16, "activation": "relu", "alpha": 1e-4,
              "learning_rate_init": 1e-3, "batch_size": 128, "pos_ratio": 0.5}

ARM_NORMALIZERS = {
    "glass_ebm": Arm(EBMFactory.normalize_params, EBM_TEST_PARAMS, "max_rounds", "learning_rate"),
    "blackbox_mlp": Arm(lambda p: mlp_tuning.normalize_mlp_params(p), MLP_PARAMS, "width", "alpha"),
    "rf_router": Arm(lambda p: rf_config.validate_rf_params(p), SMALL_PARAMS, "n_estimators",
                     "minority_weight"),
    "xgboost": Arm(XGBEstimator().normalize_params, dict(ARMS["xgboost"].params),
                   "n_estimators", "learning_rate"),
    "meta_xgb": Arm(MetaXGBEstimator().normalize_params, dict(ARMS["meta_xgb"].params),
                    "n_estimators", "learning_rate"),
}
BLACKBOX = ("blackbox_mlp", "rf_router", "xgboost", "meta_xgb")


@pytest.fixture(params=sorted(ARM_NORMALIZERS))
def arm(request):
    return ARM_NORMALIZERS[request.param]


@pytest.fixture(params=BLACKBOX)
def blackbox_arm(request):
    return ARM_NORMALIZERS[request.param]


def test_every_arm_returns_a_valid_set_unchanged(arm):
    assert arm.normalize(dict(arm.valid)) == arm.valid


@pytest.mark.parametrize("extra", ["max_dept", "eta", "random_state", "n_jobs"])
def test_no_arm_accepts_a_key_outside_its_tuned_set(arm, extra):
    with pytest.raises((KeyError, ValueError)):
        arm.normalize({**arm.valid, extra: 1})


def test_no_arm_accepts_a_missing_tuned_key(arm):
    for key in arm.valid:
        with pytest.raises((KeyError, ValueError)):
            arm.normalize({k: v for k, v in arm.valid.items() if k != key})


def test_no_blackbox_arm_coerces_a_fraction_or_a_non_finite_value(blackbox_arm):
    a = blackbox_arm
    for key, value in ((a.integer_key, a.valid[a.integer_key] + 0.5), (a.float_key, math.nan),
                       (a.integer_key, True)):
        with pytest.raises(ValueError, match=key):
            a.normalize({**a.valid, key: value})


def test_the_glass_lr_takes_no_hyperparameters_from_outside_its_search():
    params = set(inspect.signature(CalibratedLRStage.__init__).parameters)
    assert not params & {"params", "best_params", "lr_params", "C", "penalty", "l1_ratio"}
