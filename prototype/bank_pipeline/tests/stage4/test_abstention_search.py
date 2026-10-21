"""
Stage 4 abstention cut search (PR 36 F30): ``shared.stage4.tune_min_confidence``,
the rule both Stage 4 arms use.

Contract: abstain where confidence < cut; the candidates are every distinct
training confidence value (exact search); the objective is F2 on the retained
rows; a cut is eligible when retained coverage >= ``min_coverage``; the highest
F2 wins and ties go to the smaller cut; None only when no cut is eligible.

The result equals a brute-force evaluation of every distinct cut with
sklearn's F2. It finds optima between the values of the pre-F30 fixed grid
(0.03-0.53, step 0.02), keeps boundary optima (no abstention; the coverage
floor), never violates the coverage floor, moves tied confidences as one
unit, abstains on nothing when the confidence is constant, does not depend on
row order, refuses non-finite confidence, and reads only the arrays it is
given. Meta-EBM (``tune_arbiter_threshold``) and Meta-XGB both call it with
the exact search on their own training-side confidence; that the Meta-EBM's
cut uses training rows only is checked in test_meta_ebm_stage*.py.

Before F30 the fixed grid left the Meta-EBM one eligible value once F25 put
the router's confidence on the LR / EBM scale (median weighted confidence
~0.04): the cut sat at 0.03 (~68% coverage) whatever the objective said.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

import blackbox_pipeline.models.meta_xgb.runner as meta_xgb_runner
from blackbox_pipeline.models.meta_xgb.artifact import MetaXGBArtifact
from blackbox_pipeline.models.meta_xgb.config import MetaXGBConfig
from blackbox_pipeline.models.meta_xgb.estimator import MetaXGBEstimator
from blackbox_pipeline.models.meta_xgb.runner import MetaXGBRunner
from shared.stage3.data import PassthroughPipeline, Stage3Block
from shared.stage4 import (
    ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, ABSTENTION_SEARCH, tune_min_confidence,
)
from tests.stage3.conftest import ARMS, N_EVAL, N_TUNE
from tests.stage3.test_fold_label_isolation import _stage4_split


def scores(seed, n=3000, decimals=None):
    """Labels, decisions at a recall-style cut, and |p - t| confidence."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    p = np.clip(0.25 * y + rng.beta(1.2, 6.0, n), 0, 1)
    t = 0.12
    c = np.abs(p - t)
    if decimals is not None:
        c = np.round(c, decimals)                          # many tied confidences
    return y, (p >= t).astype(int), c


def brute_force(y, d, c, cuts, min_coverage):
    """The declared rule, cut by cut, with sklearn's F2 (ascending cuts, strict >)."""
    best = None
    for u in np.unique(cuts):
        keep = c >= u
        if keep.mean() < min_coverage:
            continue
        f2 = fbeta_score(y[keep], d[keep], beta=2, zero_division=0)
        if best is None or f2 > best[1]:
            best = (float(u), float(f2), float(keep.mean()))
    return best


# ----------------------------------------------------------------------
# the declared contract: exhaustive, exact, tie rule
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("decimals", [None, 3, 2])
@pytest.mark.parametrize("min_coverage", [0.5, 0.3, 0.8])
def test_exact_search_equals_brute_force_over_every_distinct_cut(seed, decimals, min_coverage):
    y, d, c = scores(seed, n=600, decimals=decimals)
    got = tune_min_confidence(y, d, c, min_coverage=min_coverage)
    want = brute_force(y, d, c, c, min_coverage)
    assert got["search"] == ABSTENTION_SEARCH
    assert got["min_confidence"] == want[0]
    assert got["train_f2"] == pytest.approx(want[1], abs=1e-12)
    assert got["train_coverage"] == want[2]
    assert got["n_candidates"] == len(np.unique(c))


def test_the_sweep_lists_every_distinct_cut_with_its_retained_set():
    y, d, c = scores(3, decimals=3)
    got = tune_min_confidence(y, d, c)
    sw = got["sweep"]
    np.testing.assert_array_equal(sw["min_confidence"], np.unique(c))
    for row in sw.iloc[:: max(1, len(sw) // 25)].itertuples():
        keep = c >= row.min_confidence
        assert row.coverage == keep.mean()
        assert row.f2 == pytest.approx(fbeta_score(y[keep], d[keep], beta=2, zero_division=0))
        assert row.eligible == (keep.mean() >= ABSTENTION_MIN_COVERAGE)
    assert got["n_eligible"] == int(sw["eligible"].sum())
    assert got["train_f2"] == sw.loc[sw["eligible"], "f2"].max()


def test_an_interior_optimum_between_the_old_grid_values_is_found():
    """Lowest band: false positives (abstaining helps); middle band: true
    positives (abstaining hurts). The optimum cut is the middle band's lowest
    confidence, 0.041, between the old grid's 0.03 and 0.05."""
    a, b, rest = np.linspace(0.001, 0.0399, 300), np.linspace(0.041, 0.06, 200), np.linspace(0.07, 0.5, 500)
    c = np.r_[a, b, rest]
    d = np.r_[np.ones(300), np.ones(200), np.repeat([1, 0, 1, 0], [100, 300, 50, 50])].astype(int)
    y = np.r_[np.zeros(300), np.ones(200), np.repeat([1, 0, 0, 1], [100, 300, 50, 50])].astype(int)
    got = tune_min_confidence(y, d, c)
    assert got["min_confidence"] == 0.041 and got["train_coverage"] == 0.7
    assert got["train_f2"] == pytest.approx(5 * 300 / (5 * 300 + 4 * 50 + 50))
    old = tune_min_confidence(y, d, c, grid=ABSTENTION_GRID)
    assert old["min_confidence"] == pytest.approx(0.03) and old["train_f2"] < got["train_f2"]


def test_no_abstention_is_kept_when_it_is_optimal():
    """The least confident rows are all true positives: abstaining on any of
    them lowers F2, so the cut retains every row (a boundary optimum)."""
    rng = np.random.default_rng(4)
    c = np.r_[np.linspace(0.0, 0.02, 300), rng.uniform(0.05, 0.5, 700)]
    d = np.r_[np.ones(300), rng.integers(0, 2, 700)].astype(int)
    y = np.r_[np.ones(300), rng.integers(0, 2, 700)].astype(int)
    got = tune_min_confidence(y, d, c)
    assert got["min_confidence"] == c.min() and got["train_coverage"] == 1.0
    assert tune_min_confidence(y, d, c, grid=ABSTENTION_GRID)["train_coverage"] < 1.0   # old grid abstains


@pytest.mark.parametrize("min_coverage", [0.5, 0.6, 0.75])
def test_the_coverage_floor_binds_when_abstaining_keeps_helping(min_coverage):
    """F2 rises with every abstained row (the least confident are all false
    positives), so the optimum is the most selective eligible cut."""
    rng = np.random.default_rng(5)
    n = 2000
    c = np.sort(rng.uniform(0, 0.3, n))
    d = np.ones(n, int)
    y = (np.arange(n) >= n // 2).astype(int)                # low half FP, high half TP
    got = tune_min_confidence(y, d, c, min_coverage=min_coverage)
    covs = np.array([(c >= u).mean() for u in np.unique(c)])
    assert got["train_coverage"] == covs[covs >= min_coverage].min()


def test_the_coverage_floor_is_never_violated_and_infeasible_floors_return_none():
    y, d, c = scores(6, decimals=3)
    for m in np.linspace(0.05, 1.0, 20):
        got = tune_min_confidence(y, d, c, min_coverage=m)
        assert got["train_coverage"] >= m
        assert (c >= got["min_confidence"]).mean() == got["train_coverage"]
    assert tune_min_confidence(y, d, c, min_coverage=1.0)["min_confidence"] == c.min()
    assert tune_min_confidence(y, d, c, min_coverage=1.01) is None


# ----------------------------------------------------------------------
# ties, degenerate inputs, determinism
# ----------------------------------------------------------------------
def test_tied_confidences_move_as_one_unit():
    """600 false positives share one confidence: the cut can retain all of
    them or none, never part, so at a 50% floor nothing is abstained even
    though dropping some of them would raise F2."""
    c = np.r_[np.full(600, 0.02), np.linspace(0.05, 0.4, 400)]
    y = np.r_[np.zeros(600), np.ones(400)].astype(int)
    d = np.ones(1000, int)
    got = tune_min_confidence(y, d, c)
    assert got["min_confidence"] == 0.02 and got["train_coverage"] == 1.0
    sw = got["sweep"]
    assert list(sw["coverage"].iloc[:2]) == [1.0, 0.4] and sw["eligible"].sum() == 1
    got = tune_min_confidence(y, d, c, min_coverage=0.4)
    assert got["min_confidence"] == 0.05 and got["train_f2"] == 1.0   # the whole group goes


@pytest.mark.parametrize("value", [0.0, 0.037, 0.4])
def test_constant_confidence_abstains_on_nothing(value):
    """No ranking, nothing to abstain on. Before F30 the grid found no cut
    below 0.03 and the arms fell back to 0.07 / 0.03, which abstains on
    every row when the confidence is below it."""
    y, d, _ = scores(7)
    got = tune_min_confidence(y, d, np.full(len(y), value))
    assert got["min_confidence"] == value and got["train_coverage"] == 1.0
    assert got["n_candidates"] == 1


def test_an_eligible_cut_is_chosen_even_when_every_f2_is_zero():
    """Before F30 the sweep started from a score of 0 and kept only strictly
    better cuts, so no positives returned None (the arm fallback)."""
    rng = np.random.default_rng(8)
    c = rng.uniform(0, 0.2, 500)
    got = tune_min_confidence(np.zeros(500, int), rng.integers(0, 2, 500), c)
    assert got["train_f2"] == 0.0 and got["min_confidence"] == c.min() and got["train_coverage"] == 1.0


def test_selection_does_not_depend_on_row_order_and_is_reproducible():
    y, d, c = scores(9, decimals=3)
    a = tune_min_confidence(y, d, c)
    order = np.random.default_rng(0).permutation(len(y))
    b = tune_min_confidence(y[order], d[order], c[order])
    again = tune_min_confidence(y, d, c)
    for k in ("min_confidence", "train_f2", "train_coverage", "n_candidates", "n_eligible"):
        assert a[k] == b[k] == again[k], k
    pd.testing.assert_frame_equal(a["sweep"], b["sweep"])
    pd.testing.assert_frame_equal(a["sweep"], again["sweep"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_or_misaligned_inputs_are_refused(bad):
    y, d, c = scores(10, n=200)
    c[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        tune_min_confidence(y, d, c)
    with pytest.raises(ValueError, match="lengths differ"):
        tune_min_confidence(y[:-1], d, np.abs(np.nan_to_num(c, posinf=0)))
    assert tune_min_confidence(np.array([]), np.array([]), np.array([])) is None


def test_an_explicit_grid_is_searched_with_the_same_rule():
    y, d, c = scores(11)
    got = tune_min_confidence(y, d, c, grid=ABSTENTION_GRID)
    want = brute_force(y, d, c, ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE)
    assert got["search"] == "explicit grid" and got["min_confidence"] == want[0]
    assert got["train_f2"] == pytest.approx(want[1], abs=1e-12)


# ----------------------------------------------------------------------
# both Stage 4 arms use it
# ----------------------------------------------------------------------
def test_meta_xgb_selects_its_cut_by_the_shared_exact_search_on_meta_oof(monkeypatch, tmp_path):
    X_tr, y_tr, X_te, y_te = _stage4_split()
    cfg = MetaXGBConfig(n_tune_folds=N_TUNE, n_eval_folds=N_EVAL, verbose=False,
                        show_progress_bar=False, artifact_dir=str(tmp_path))
    block = Stage3Block.build(X_tr, y_tr, X_te, y_te, PassthroughPipeline,
                              n_tune_folds=N_TUNE, n_eval_folds=N_EVAL,
                              impute_missing=cfg.impute_missing,
                              expected_features=cfg.expected_features, verbose=False)
    runner = MetaXGBRunner(cfg, MetaXGBEstimator(cfg.search_space, cfg.random_state, 1))
    seen = []
    real = meta_xgb_runner.tune_min_confidence

    def spy(y, pred, conf, grid=None, min_coverage=ABSTENTION_MIN_COVERAGE):
        out = real(y, pred, conf, grid, min_coverage)
        seen.append((np.asarray(y).copy(), np.asarray(conf).copy(), grid, min_coverage, out))
        return out

    monkeypatch.setattr(meta_xgb_runner, "tune_min_confidence", spy)
    runner.fit(block, params=dict(ARMS["meta_xgb"].params))
    (y_seen, conf, grid, min_cov, out), = seen
    assert grid is None and min_cov == ABSTENTION_MIN_COVERAGE
    np.testing.assert_array_equal(y_seen, np.asarray(y_tr))             # training rows only
    ab = runner.abstention_
    assert ab["search"] == ABSTENTION_SEARCH and "grid" not in ab
    assert ab["min_confidence"] == out["min_confidence"] and ab["min_confidence"] in conf
    assert ab["train_coverage"] >= ABSTENTION_MIN_COVERAGE
    assert runner.frozen_["min_confidence"] == ab["min_confidence"]
    assert runner.train_eval_["with_abstain"]["coverage"] == pytest.approx(ab["train_coverage"])
    assert len(ab["sweep"]) == len(np.unique(conf))                    # full curve in the artifact,
    art = MetaXGBArtifact(                                             # not in the JSON sidecar
        created_at="", split_fingerprint="", split_fingerprints={}, index_train=[], index_test=[],
        feature_names=[], feature_contract={}, stage4_inputs={"streams": {}}, fold_provenance={},
        learner=runner.artifact_, operating=runner.operating_, abstention=dict(ab),
        train_oof_eval={}, test={"no_abstain": {}, "with_abstain": {}}, events=[], validation={})
    summary = art.summary()["abstention"]
    assert "sweep" not in summary and summary["search"] == ABSTENTION_SEARCH
    json.dumps(summary)
