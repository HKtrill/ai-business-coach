"""
LogisticRegression penalty spelling (glass_pipeline.lr.penalty).

Contract: ``penalty_kwargs`` selects exactly the model the legacy
``penalty=`` spelling selected - identical coefficients, intercepts and
iteration counts - on the installed scikit-learn, without its 1.8+
deprecation warnings; on scikit-learn < 1.8 it keeps the legacy spelling,
which is the only correct one there.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import glass_pipeline.lr.penalty as penalty_mod
from glass_pipeline.lr.lr_stage import LRStage
from glass_pipeline.lr.penalty import SKLEARN_PENALTY_DEPRECATED, penalty_kwargs
from tests.stage1.conftest import LABEL_FREE_FEATURES, fit_glass_stage1
from tests.support.synthetic import make_bank_frame

LEGACY = {
    "l2": {"penalty": "l2"},
    "l1": {"penalty": "l1"},
    "elasticnet": {"penalty": "elasticnet", "l1_ratio": 0.37},
}


@pytest.fixture(scope="module")
def data():
    df = make_bank_frame(n=600)
    return StandardScaler().fit_transform(df[LABEL_FREE_FEATURES]), df["y"].astype(int)


def _fit(kwargs, X, y, solver="saga"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(C=0.3, solver=solver, max_iter=5000, random_state=42,
                                   class_weight="balanced", **kwargs).fit(X, y)
    return model, [w for w in caught if "_logistic" in w.filename]


@pytest.mark.parametrize("penalty, solver", [("l2", "saga"), ("l1", "saga"),
                                             ("elasticnet", "saga"), ("l2", "liblinear")])
def test_new_spelling_fits_the_same_model_without_warnings(data, penalty, solver):
    X, y = data
    legacy, _ = _fit(LEGACY[penalty], X, y, solver)
    new, caught = _fit(penalty_kwargs(penalty, LEGACY[penalty].get("l1_ratio")), X, y, solver)
    np.testing.assert_array_equal(new.coef_, legacy.coef_)
    np.testing.assert_array_equal(new.intercept_, legacy.intercept_)
    np.testing.assert_array_equal(new.n_iter_, legacy.n_iter_)
    assert caught == []


def test_legacy_spelling_is_kept_before_scikit_learn_1_8(monkeypatch):
    monkeypatch.setattr(penalty_mod, "SKLEARN_PENALTY_DEPRECATED", False)
    assert penalty_mod.penalty_kwargs("l1") == {"penalty": "l1"}
    assert penalty_mod.penalty_kwargs("elasticnet", 0.4) == {"penalty": "elasticnet", "l1_ratio": 0.4}
    monkeypatch.setattr(penalty_mod, "SKLEARN_PENALTY_DEPRECATED", True)
    assert penalty_mod.penalty_kwargs("l1") == {"l1_ratio": 1.0}
    assert penalty_mod.penalty_kwargs("l2") == {"l1_ratio": 0.0}
    assert penalty_mod.penalty_kwargs("elasticnet", 0.4) == {"l1_ratio": 0.4}


def test_unmapped_penalties_pass_through_and_elasticnet_needs_a_ratio():
    assert penalty_kwargs(None) == {"penalty": None}            # its 1.8+ form also changes C
    with pytest.raises(ValueError, match="l1_ratio"):
        penalty_kwargs("elasticnet")


def test_version_flag_matches_the_installed_scikit_learn():
    import sklearn
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    assert SKLEARN_PENALTY_DEPRECATED == ((major, minor) >= (1, 8))


def test_stages_fit_without_penalty_warnings():
    """The tuned GLASS stage (l1 / l2 / elasticnet trials) and the baseline stage."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = fit_glass_stage1(tune_hyperparameters=True, n_trials=4, tune_cv_folds=3, cv_folds=3)
        df = make_bank_frame(n=300)
        with contextlib.redirect_stdout(io.StringIO()):
            LRStage().fit(df[LABEL_FREE_FEATURES], df["y"])
    assert run.stage.tuning_trials["penalty"].nunique() >= 2      # several spellings exercised
    assert [w for w in caught if "_logistic" in w.filename] == []
