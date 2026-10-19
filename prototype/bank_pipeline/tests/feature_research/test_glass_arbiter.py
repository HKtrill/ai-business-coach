"""
Feature-research GLASS Arbiter (model_training.glass_arbiter).

The arbiter combines the three stage OOF scores. Its cross-fit must freeze
each fold's configuration before that fold is touched, so every row's
decision comes from thresholds and weights chosen without its label.
Includes the arbiter checks from notebook Cell 18.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit, StratifiedKFold

from feature_research.model_training.glass_arbiter import cross_fit_arbiter, fit_arbiter
from feature_research.model_training.glass_arbiter.arbiter import MODELS, check_scores
from tests.support import mutations as mut

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


@pytest.fixture
def scores(bank_split):
    """Stage-like OOF scores: noisy, informative, finite, aligned to y_train."""
    y = bank_split["y_train"]
    rng = np.random.default_rng(0)
    cols = {m: np.clip(0.25 + 0.3 * y.to_numpy() + rng.normal(0, s, len(y)), 0.001, 0.999)
            for m, s in zip(MODELS, (0.20, 0.25, 0.15))}
    return pd.DataFrame(cols, index=y.index), y


def _cross_fit(P, y, cv=CV):
    return cross_fit_arbiter(P, y, cv, rule="recall", target_recall=0.70, verbose=False)


def test_outputs_are_aligned_and_well_formed(scores):
    P, y = scores
    res = _cross_fit(P, y)
    for series in (res.proba, res.pred, res.pred_full):
        assert series.index.equals(y.index)
    assert np.isfinite(res.proba.to_numpy()).all()
    assert set(np.unique(res.pred)) <= {-1, 0, 1}
    assert set(np.unique(res.pred_full)) <= {0, 1}
    assert list(res.model_preds.columns) == list(MODELS)


def test_reported_config_is_the_reproducible_all_rows_fit(scores):
    """Notebook Cell 18: refitting on P_oof reproduces arbiter_config exactly."""
    P, y = scores
    res = _cross_fit(P, y)
    refit = fit_arbiter(P, y, rule="recall", target_recall=0.70)
    assert (refit.thresholds, refit.weights, refit.min_conf) == (
        res.config.thresholds, res.config.weights, res.config.min_conf)


def test_each_row_uses_its_own_fold_configuration(scores):
    P, y = scores
    res = _cross_fit(P, y)
    for cfg, (_, va) in zip(res.fold_configs, CV.split(P, y)):
        expected = np.array([cfg.thresholds[m] for m in MODELS])
        np.testing.assert_allclose(res.row_thresholds.iloc[va].to_numpy(),
                                   np.broadcast_to(expected, (len(va), len(MODELS))))


def test_a_folds_labels_never_shape_its_own_decisions(scores):
    """With folds fixed, permuting fold 0's labels leaves fold 0's scores and
    decisions unchanged, while the folds that trained on them move."""
    P, y = scores
    test_fold = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(CV.split(P, y)):
        test_fold[va] = k
    fixed = PredefinedSplit(test_fold)
    fold0 = y.index[test_fold == 0]

    base = _cross_fit(P, y, fixed)
    moved = _cross_fit(P, mut.permute_labels(y, seed=2, rows=fold0), fixed)
    pd.testing.assert_series_equal(moved.proba.loc[fold0], base.proba.loc[fold0])
    pd.testing.assert_series_equal(moved.pred.loc[fold0], base.pred.loc[fold0])
    assert moved.fold_configs[1].thresholds != base.fold_configs[1].thresholds or \
        moved.fold_configs[1].weights != base.fold_configs[1].weights


@pytest.mark.parametrize("corrupt, message", [
    (lambda P: mut.shuffle_rows(P), "not aligned"),
    (lambda P: mut.swap_columns(P), "score columns must be"),
    (lambda P: P.assign(lr=mut.corrupt_probabilities(P["lr"], "nan")), "finite probabilities"),
    (lambda P: P.assign(rf=mut.corrupt_probabilities(P["rf"], "above_one")), "finite probabilities"),
], ids=["misaligned", "reordered_columns", "nan", "above_one"])
def test_invalid_scores_are_rejected(scores, corrupt, message):
    P, y = scores
    with pytest.raises(ValueError, match=message):
        check_scores(corrupt(P), y)
    with pytest.raises(ValueError, match=message):
        _cross_fit(corrupt(P), y)
