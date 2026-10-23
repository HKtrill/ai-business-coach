"""
GLASS Router end-to-end fit (rule search + evaluation + ILP), opt-in.

One real fit of the notebook's "legacy" profile on synthetic engineered
features takes ~30 s (rule search and overlap diagnostics, not the ILP), so it
is marked ``slow`` and excluded by default. Run with ``pytest -m slow``.

ILP solves are capped at 2 s here, so the selected set may be time-limited
and is not pinned; the tests check invariants that must hold for whatever set
is returned, recomputed from the training rows.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from glass_pipeline.glass_router.core.rule import ABSTAIN, NOT_SUBSCRIBE, SUBSCRIBE
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import ILPRuleSelector
from shared.stage2 import RF_FEATURES_BINARY
from shared.stage2.feature_engineering import engineer_features
from tests.stage2.conftest import legacy_config, new_router
from tests.support.synthetic import make_bank_frame, split_frame

pytestmark = pytest.mark.slow

N = 1200
CFG = dict(min_support_pass1=N // 40, min_support_pass2=N // 120)   # supports scaled to N


@pytest.fixture(scope="module")
def fit():
    split = split_frame(make_bank_frame(n=N))
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(ILPRuleSelector, "TIME_LIMIT_PER_SOLVE", 2)
        X_tr, X_te = engineer_features(split["X_train"], split["X_test"], split["y_train"])
        router = new_router(legacy_config(**CFG)).fit(X_tr, split["y_train"])
    return router, X_tr, X_te, split["y_train"].to_numpy()


def _mask(rule, X):
    return np.logical_and.reduce([X[f].to_numpy() == v for f, v in rule.segment])


def test_rule_sets_respect_the_profile(fit):
    router, *_ = fit
    cfg = router.config
    assert cfg.min_pass1_rules <= len(router.pass1_rules) <= cfg.max_pass1_rules
    assert cfg.min_pass2_rules <= len(router.pass2_rules) <= cfg.max_pass2_rules
    for rules, cls, name in ((router.pass1_rules, NOT_SUBSCRIBE, "pass1"),
                             (router.pass2_rules, SUBSCRIBE, "pass2")):
        for r in rules:
            assert r.predicted_class == cls and r.pass_assignment == name
            assert 1 <= r.complexity == len(r.segment) <= cfg.max_complexity
            assert {f for f, _ in r.segment} <= set(RF_FEATURES_BINARY)


def test_stored_rule_metrics_match_the_training_rows(fit):
    router, X, _, y = fit
    for r in router.pass1_rules + router.pass2_rules:
        m = _mask(r, X)
        target = 0 if r.pass_assignment == "pass1" else 1
        assert r.precision == pytest.approx((y[m] == target).mean())
        assert r.coverage == pytest.approx(m.mean())


def test_per_rule_gates_hold_on_the_training_rows(fit):
    router, X, _, y = fit
    cfg, positives = router.config, y.sum()
    for r in router.pass1_rules:
        m = _mask(r, X)
        assert (y[m] == 0).mean() >= cfg.min_precision_not_subscribe
        assert (y[m] == 1).sum() <= cfg.max_subscriber_leakage_rate * positives
    for r in router.pass2_rules:
        m = _mask(r, X)
        assert (y[m] == 1).mean() >= cfg.min_precision_subscribe
        assert cfg.min_recall_subscribe <= (y[m] == 1).sum() / positives <= cfg.max_recall_subscribe


def test_pass1_union_leakage_cap_holds_on_the_training_rows(fit):
    router, X, _, y = fit
    union = np.logical_or.reduce([_mask(r, X) for r in router.pass1_rules])
    assert (union & (y == 1)).sum() <= router.config.max_union_leakage_rate_pass1 * y.sum()


@pytest.mark.parametrize("name", ["pass1", "pass2"])
def test_every_selected_rule_meets_the_set_level_novelty_floor(fit, name):
    router, X, _, _ = fit
    rules = router.pass1_rules if name == "pass1" else router.pass2_rules
    floor = getattr(router.config, f"min_novelty_ratio_{name}")
    masks = [_mask(r, X) for r in rules]
    for i, m in enumerate(masks):
        others = np.logical_or.reduce([o for j, o in enumerate(masks) if j != i])
        assert (m & ~others).sum() >= max(1, int(np.ceil(floor * m.sum() - 1e-9)))


def test_fitted_router_routes_train_and_test_into_three_states(fit):
    router, X_tr, X_te, _ = fit
    for X in (X_tr, X_te):
        pred, conf, dec = router.predict(X)
        assert set(pred) <= {NOT_SUBSCRIBE, SUBSCRIBE, ABSTAIN}
        np.testing.assert_array_equal(pred == ABSTAIN, dec == "uncertain")
        np.testing.assert_array_equal(pred == NOT_SUBSCRIBE, dec == "pass1")
        assert ((conf == 0) == (dec == "uncertain")).all()
