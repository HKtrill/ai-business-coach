"""
Fold-label isolation of every training-side Stage 3 output (audit F22), for
each arm on the shared runner: GLASS EBM, black-box XGBoost and the Meta-XGB
learner.

Contract: with the outer folds held fixed, shuffling the labels inside one
outer fold changes nothing the runner stores for that fold's rows: raw and
calibrated OOF probabilities, the inner scores and calibration gate fitted
for it, its nested threshold and its decision / margin / confidence / state
columns, which are what Stage 4 consumes. Shuffling the other folds' labels
does change them, so the check can fail.

Before F22 the calibrated column and the nested threshold of fold k were
fitted on the other folds' OOF scores (from models trained on fold k) and the
calibration gate on every row's labels. With the real GLASS feature DAG the
same contract is checked in test_glass_ebm_stage.py.

The calibrated-space nested thresholds (Meta-XGB ``operating_space=
"calibrated"``) obey the same rule: fold k's is selected on its inner scores
mapped by its own calibrator.

Hyperparameters are supplied, as when the notebooks reuse them. Selecting
them by Optuna on these same tuning folds is a separate, still-open channel
(F24 in the PR 36 test report; Stage 1 tunes the same way).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES, MetaXGBConfig
from blackbox_pipeline.models.meta_xgb.estimator import MetaXGBEstimator
from blackbox_pipeline.models.meta_xgb.runner import MetaXGBRunner
from shared.stage3.calibration import Stage3Calibrator
from shared.stage3.data import PassthroughPipeline, Stage3Block
from shared.stage3.thresholding import F2ThresholdSelector
from tests.stage3.conftest import ARMS, N_EVAL, N_TUNE, run_arm, stage3_split
from tests.support import mutations as mut

TRAIN_SIDE = ("proba", "proba_calibrated", "decision", "margin", "confidence",
              "threshold_per_row", "state")


@pytest.fixture(scope="module")
def moved(arm_runs, tmp_path_factory):
    """Per arm: runs whose labels moved inside tuning fold 0 ("own") or outside it."""
    out = {}
    for name, base in arm_runs.items():
        y = base.block.y_train
        fold0 = y.index[base.block.tune_plan.holdout(0)]
        out[name] = {}
        for which, rows in (("own", fold0), ("others", y.index.difference(fold0))):
            split = {**stage3_split(), "y_train": mut.permute_labels(y, seed=5, rows=rows)}
            out[name][which] = run_arm(ARMS[name], tmp_path_factory.mktemp(f"{name}_{which}"),
                                       split=split, fold_labels=y).artifact
    return out


@pytest.mark.parametrize("name", sorted(ARMS))
def test_a_folds_own_labels_never_reach_its_training_side_outputs(arm_runs, moved, name):
    base = arm_runs[name].artifact
    own, others = moved[name]["own"], moved[name]["others"]
    pos = arm_runs[name].block.tune_plan.holdout(0)
    np.testing.assert_array_equal(own.oof["fold_id"], base.oof["fold_id"])      # folds held fixed
    assert base.calibration["applied"] and base.calibration["fold_applied"][0]   # not trivially equal

    for key in TRAIN_SIDE:
        np.testing.assert_array_equal(np.asarray(own.oof[key])[pos], np.asarray(base.oof[key])[pos],
                                      err_msg=key)
    np.testing.assert_array_equal(own.oof["inner_proba_by_fold"][0], base.oof["inner_proba_by_fold"][0])
    assert own.calibration["fold_applied"][0] == base.calibration["fold_applied"][0]
    assert own.calibration["fold_ece_inner"][0] == base.calibration["fold_ece_inner"][0]
    assert own.threshold["oof_nested"]["by_fold"][0] == base.threshold["oof_nested"]["by_fold"][0]
    s4_own, s4_base = own.stage4_frame(), base.stage4_frame()
    np.testing.assert_array_equal(s4_own.iloc[pos].to_numpy(), s4_base.iloc[pos].to_numpy())

    # the check can fail: other folds' labels reach fold 0, fold 0's reach the rest
    for key in ("proba", "proba_calibrated"):
        assert not np.allclose(np.asarray(others.oof[key])[pos], np.asarray(base.oof[key])[pos]), key
    assert others.calibration["fold_ece_inner"][0] != base.calibration["fold_ece_inner"][0]
    rest = np.setdiff1d(np.arange(len(base.oof["proba"])), pos)
    assert not np.allclose(np.asarray(own.oof["proba_calibrated"])[rest],
                           np.asarray(base.oof["proba_calibrated"])[rest])


@pytest.mark.parametrize("name", sorted(ARMS))
def test_test_side_decisions_still_use_every_training_row(arm_runs, name):
    """The final model's gate, map and operating thresholds are fitted on all
    OOF rows (as in Stage 1); only the training-side outputs are nested."""
    art = arm_runs[name].artifact
    assert art.calibration["n_rows"] == len(art.oof["proba"])
    assert art.threshold["oof"]["n_rows"] == len(art.oof["proba"])
    assert len(art.calibration["fold_applied"]) == len(art.threshold["oof_nested"]["by_fold"]) == N_TUNE
    assert "inner" in art.threshold["oof_nested"]["selected_on"]


# ----------------------------------------------------------------------
# calibrated-space nested thresholds (Meta-XGB operating_space="calibrated")
# ----------------------------------------------------------------------
def test_calibrated_space_nested_thresholds_ignore_a_folds_own_labels(arm_runs):
    """Fold k's calibrated-space threshold is selected on its inner scores
    mapped by its own calibrator (``inner_calibrated_``), never on the other
    folds' calibrated OOF."""
    run = arm_runs["xgboost"]
    art, block = run.artifact, run.block
    p, inner, plan = np.asarray(art.oof["proba"]), art.oof["inner_proba_by_fold"], block.tune_plan
    y = block.y_train
    fold0 = plan.fold_id == 0

    def fold0_inputs_and_threshold(labels):
        cal = Stage3Calibrator(force=True).fit(p, labels, plan, inner)
        t = F2ThresholdSelector().select_nested(labels, cal.oof_calibrated_, plan,
                                                cal.inner_calibrated_)["by_fold"][0]
        return cal.inner_calibrated_[0], t

    base_scores, base_t = fold0_inputs_and_threshold(y)
    own_scores, own_t = fold0_inputs_and_threshold(mut.permute_labels(y, seed=2, rows=y.index[fold0]))
    np.testing.assert_array_equal(own_scores, base_scores)
    assert own_t == base_t
    others_scores, _ = fold0_inputs_and_threshold(mut.permute_labels(y, seed=2, rows=y.index[~fold0]))
    assert not np.allclose(others_scores[~fold0], base_scores[~fold0])          # the check can fail


def _stage4_split():
    """Synthetic Stage 4 feature frames (``STAGE4_FEATURES``) on the Stage 3 split."""
    split = stage3_split()
    rng = np.random.default_rng(0)

    def frame(y: pd.Series) -> pd.DataFrame:
        signal, n = y.to_numpy(dtype=float), len(y)
        route = rng.integers(0, 3, n)
        return pd.DataFrame({
            "stage1_proba": np.clip(0.3 * signal + rng.uniform(0, 0.7, n), 0, 1),
            "stage2_proba": np.clip(0.2 * signal + rng.uniform(0, 0.8, n), 0, 1),
            "stage2_pass1": (route == 0).astype(int),
            "stage2_pass2": (route == 1).astype(int),
            "stage2_abstain": (route == 2).astype(int),
            "stage3_proba": np.clip(0.35 * signal + rng.uniform(0, 0.65, n), 0, 1),
        }, index=y.index)[list(STAGE4_FEATURES)]

    return frame(split["y_train"]), split["y_train"], frame(split["y_test"]), split["y_test"]


def test_meta_xgb_calibrated_operating_space_uses_inner_calibrated_thresholds(tmp_path, monkeypatch):
    X_tr, y_tr, X_te, y_te = _stage4_split()
    cfg = MetaXGBConfig(operating_space="calibrated", n_tune_folds=N_TUNE, n_eval_folds=N_EVAL,
                        verbose=False, show_progress_bar=False, artifact_dir=str(tmp_path))
    block = Stage3Block.build(X_tr, y_tr, X_te, y_te, PassthroughPipeline,
                              n_tune_folds=N_TUNE, n_eval_folds=N_EVAL,
                              impute_missing=cfg.impute_missing,
                              expected_features=cfg.expected_features, verbose=False)
    runner = MetaXGBRunner(cfg, MetaXGBEstimator(cfg.search_space, cfg.random_state, 1))
    calls = []
    real = runner.selector.select_nested

    def spy(y, proba, folds, inner_proba):
        calls.append(inner_proba)
        return real(y, proba, folds, inner_proba)

    monkeypatch.setattr(runner.selector, "select_nested", spy)
    runner.fit(block, params=dict(ARMS["meta_xgb"].params))
    assert runner.operating_["space"] == "calibrated"
    assert len(calls) == 2                                    # raw-space (artifact) + calibrated-space
    np.testing.assert_array_equal(calls[1], runner.calibrator_.inner_calibrated_)
