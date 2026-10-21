"""
Stage 3 operating points (shared.stage3.thresholding) and test-label isolation.

Contract: the F-beta threshold is the grid argmax of the same ``p >= t`` rule
that produces decisions; fold k's nested threshold is selected on the inner
out-of-fold scores of fold k's training rows, so it never uses fold k's labels
or scores from models trained on them (audit F22); and nothing the runner
learns depends on the held-out test labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import fbeta_score

from shared.stage3.data import FoldPlan
from shared.stage3.thresholding import F2ThresholdSelector, decision_block
from tests.stage3.conftest import ARMS, run_arm, stage3_split
from tests.support import mutations as mut


def _noisy(y, rng):
    return np.clip(0.2 + 0.35 * np.asarray(y) + rng.normal(0, 0.18, len(y)), 0.0, 1.0)


@pytest.fixture
def scored(bank_split):
    """OOF scores, a fold plan, and inner scores covering each fold's training rows."""
    y = bank_split["y_train"].astype(int)
    rng = np.random.default_rng(1)
    p = _noisy(y, rng)
    plan = FoldPlan.build(y, n_splits=3, random_state=0)
    inner = np.full((plan.n_splits, len(y)), np.nan)
    for k, (tr, _) in enumerate(plan):
        inner[k, tr] = _noisy(y.to_numpy()[tr], rng)
    return p, y, plan, inner


# ----------------------------------------------------------------------
# selector
# ----------------------------------------------------------------------
def test_selected_threshold_is_the_grid_argmax_of_the_decision_rule(scored):
    p, y, _, _ = scored
    sel = F2ThresholdSelector()
    choice = sel.select(y, p)
    scores = [fbeta_score(y, (p >= t).astype(int), beta=2.0, zero_division=0) for t in sel.grid]
    assert choice.score == pytest.approx(max(scores))
    assert choice.threshold == pytest.approx(sel.grid[int(np.argmax(scores))])


def test_ties_resolve_to_the_lowest_threshold():
    y = np.array([0] * 20 + [1] * 5)
    p = np.array([0.05] * 20 + [0.95] * 5)          # every grid point is perfect
    choice = F2ThresholdSelector().select(y, p)
    assert choice.score == pytest.approx(1.0)
    assert choice.threshold == pytest.approx(0.10)


@pytest.mark.parametrize("y, p, message", [
    ([0, 1, 0], [0.2, 0.8], "length mismatch"),
    ([], [], "empty population"),
    ([0, 1], [0.2, np.nan], "NaN"),
])
def test_selector_refuses_invalid_inputs(y, p, message):
    with pytest.raises(ValueError, match=message):
        F2ThresholdSelector().select(np.array(y), np.array(p, dtype=float))


@pytest.mark.parametrize("kwargs", [dict(low=0.5, high=0.4), dict(low=0.0), dict(steps=1)])
def test_selector_refuses_an_invalid_grid(kwargs):
    with pytest.raises(ValueError):
        F2ThresholdSelector(**kwargs)


def test_nested_thresholds_are_selected_on_inner_scores(scored):
    p, y, plan, inner = scored
    sel = F2ThresholdSelector()
    nested = sel.select_nested(y, p, plan, inner)
    y_arr = y.to_numpy()
    for k, (tr, va) in enumerate(plan):
        assert nested["by_fold"][k] == sel.select(y_arr[tr], inner[k, tr]).threshold
        assert (nested["per_row"][va] == nested["by_fold"][k]).all()
    assert "inner" in nested["selected_on"]


def test_nested_thresholds_never_use_their_own_fold_labels(scored):
    p, y, plan, inner = scored
    fold0 = plan.fold_id == 0
    sel = F2ThresholdSelector()
    base = sel.select_nested(y, p, plan, inner)
    own = sel.select_nested(mut.permute_labels(y, seed=7, rows=y.index[fold0]), p, plan, inner)
    others = sel.select_nested(mut.permute_labels(y, seed=7, rows=y.index[~fold0]), p, plan, inner)
    assert own["by_fold"][0] == base["by_fold"][0]
    np.testing.assert_array_equal(own["per_row"][fold0], base["per_row"][fold0])
    assert others["by_fold"][0] != base["by_fold"][0]          # the check can fail


@pytest.mark.parametrize("case, message", [
    ("foreign_plan", "must not be reused"),
    ("outer_oof_as_inner", "must leave its held-out rows unscored"),
    ("no_inner_scores", "must have shape"),
])
def test_nested_selection_refuses_invalid_inputs(scored, case, message):
    p, y, plan, inner = scored
    sel = F2ThresholdSelector()
    with pytest.raises(ValueError, match=message):
        if case == "foreign_plan":
            sel.select_nested(y.iloc[:-1], p[:-1], plan, inner[:, :-1])
        elif case == "outer_oof_as_inner":                    # the pre-F22 procedure
            sel.select_nested(y, p, plan, np.tile(p, (plan.n_splits, 1)))
        else:
            sel.select_nested(y, p, plan, p)


def test_decision_boundary_and_confidence():
    out = decision_block(np.array([0.2, 0.3, 0.31, 0.9]), 0.3)
    np.testing.assert_array_equal(out["decision"], [0, 1, 1, 1])          # p == t is flagged
    assert out["margin"][1] == 0.0 and out["confidence"][1] == 0.0
    assert ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all()
    assert list(out["state"]) == ["not_flagged", "flagged", "flagged", "flagged"]
    per_row = decision_block(np.array([0.5, 0.5]), np.array([0.4, 0.6]))
    np.testing.assert_array_equal(per_row["decision"], [1, 0])


# ----------------------------------------------------------------------
# held-out test labels cannot influence anything learned (every arm)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ARMS))
def test_test_labels_do_not_influence_anything_learned(arm_runs, tmp_path, name):
    base = arm_runs[name].artifact
    split = stage3_split()
    moved = run_arm(ARMS[name], tmp_path, split=stage3_split(
        y_test=mut.permute_labels(split["y_test"], seed=11))).artifact

    assert moved.y_test != base.y_test                                  # the labels did change
    assert moved.best_params == base.best_params
    for key in ("proba", "proba_calibrated", "decision", "threshold_per_row", "fold_id"):
        np.testing.assert_array_equal(moved.oof[key], base.oof[key], err_msg=key)
    for key in ("oof", "oof_calibrated_space"):
        assert moved.threshold[key]["threshold"] == base.threshold[key]["threshold"]
    assert moved.threshold["oof_nested"]["by_fold"] == base.threshold["oof_nested"]["by_fold"]
    for key in ("applied", "ece_before", "ece_after"):
        assert moved.calibration[key] == base.calibration[key]
    for key in ("test_proba", "test_proba_calibrated", "test_decision"):
        np.testing.assert_array_equal(moved.refit[key], base.refit[key], err_msg=key)
    np.testing.assert_array_equal(moved.fold_mean["test_proba"], base.fold_mean["test_proba"])
