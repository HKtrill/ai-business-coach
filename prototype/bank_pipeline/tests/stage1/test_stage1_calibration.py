"""
Stage 1 calibration for both arms (shared.stage1.oof_calibration): audit F16
(family selection) and F19 (preprocessing inside calibration folds).

Contract: sigmoid vs isotonic is chosen by Brier on nested out-of-fold
probabilities, each from a base model, encoder, scaler and calibrator fitted
without the row's label; every inner calibration fold refits the encoder and
scaler on its own training rows, inside its outer fold's training rows; the
final scorer is the full-train model plus one calibrator fitted on
out-of-fold scores; both arms run the same code; outputs are deterministic,
aligned and saved in the existing artifact contracts.

Independent reference: scikit-learn's ``CalibratedClassifierCV(Pipeline(
[encoder, scaler, model]), ensemble=False)``, which refits the whole pipeline
inside every calibration fold, cross-validated over the same outer folds.

MLP runs use tiny synthetic folds, smaller than the tuned batch sizes, so
scikit-learn clips the batch and warns (impossible at production scale); the
filter matches that one message only.
"""

from __future__ import annotations

import contextlib
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shared.stage1.crossfit as crossfit_mod
import shared.stage1.oof_calibration as ocal_mod
from blackbox_pipeline.models.mlp import CalibratedStage1MLP, save_stage1_mlp
from glass_pipeline.lr.artifacts import load_artifact, save_artifact
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from shared.stage1.calibration_guard import CalibrationLeakageError
from shared.stage1.crossfit import build_fold_designs
from shared.stage1.feature_engineering import (
    LR_FEATURES, LRFeatureEngineer, engineer_features,
)
from shared.stage1.oof_calibration import (
    CALIBRATION_FAMILIES, PlattCalibrator, Stage1CalibratedModel, base_scores, calibrate_oof,
    make_calibrator,
)
from shared.stage_io import StageOutput, fold_assignment
from tests.stage1.conftest import (
    LABEL_FREE_FEATURES, _pinned_folds, MLP_TEST_CONFIG, fit_glass_stage1, fit_mlp_stage1, fold_levels,
    stage1_split, sub_log, unrecorded_scale,
)
from tests.support import mutations as mut
from tests.support.checks import assert_never_fit_on, assert_oof_provenance

pytestmark = pytest.mark.filterwarnings(
    "ignore:Got `batch_size` less than 1 or larger than sample size:UserWarning")

SEED = 42
LEGACY_METRIC_KEYS = {"brier_uncalibrated", "brier_calibrated", "ece_uncalibrated", "ece_calibrated"}


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture
def lr5(lr_run_5):
    return lr_run_5


@pytest.fixture(scope="module")
def direct(lr_run_5):
    """``calibrate_oof`` called directly on the 5-fold GLASS run's data."""
    X, y, raw = lr_run_5.X_train, lr_run_5.y_train, lr_run_5.split["X_train"]
    designs = _quiet(build_fold_designs, X, y, cv_folds=5, random_state=SEED, X_raw=raw)
    return calibrate_oof(lr_run_5.stage.model, designs, X, y, cv_folds=5, random_state=SEED,
                         X_raw=raw, n_jobs=1)


@pytest.fixture(params=["glass_lr", "blackbox_mlp"])
def arm(request):
    return request.getfixturevalue("lr5" if request.param == "glass_lr" else "mlp_run")


def _folds(run) -> int:
    return run.stage.cv_folds if isinstance(run.stage, CalibratedLRStage) else run.stage.config.cv_folds


# ----------------------------------------------------------------------
# independent reference: CalibratedClassifierCV over the whole pipeline
# ----------------------------------------------------------------------
class _Encoder(BaseEstimator, TransformerMixin):
    """LRFeatureEngineer as an sklearn transformer (refit by the reference in every fold)."""

    def fit(self, X, y):
        self.engineer_ = _quiet(LRFeatureEngineer().fit, X, pd.Series(np.asarray(y), index=X.index))
        return self

    def transform(self, X):
        return _quiet(self.engineer_.transform, X)[LR_FEATURES]


def _reference(run, method):
    k = _folds(run)
    pipe = Pipeline([("encode", _Encoder()), ("scale", StandardScaler()),
                     ("model", clone(run.stage.model))])
    return CalibratedClassifierCV(pipe, method=method, ensemble=False,
                                  cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED))


@pytest.mark.parametrize("method", CALIBRATION_FAMILIES)
def test_nested_oof_equals_the_sklearn_pipeline_reference(arm, method):
    k = _folds(arm)
    expected = cross_val_predict(_reference(arm, method), arm.split["X_train"], arm.y_train,
                                 cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED),
                                 method="predict_proba")[:, 1]
    np.testing.assert_allclose(arm.stage.calibration_oof_[method].to_numpy(), expected,
                               rtol=0, atol=1e-8)


def test_final_scorer_equals_the_sklearn_pipeline_reference(arm):
    reference = _reference(arm, arm.stage.calibration_method).fit(arm.split["X_train"], arm.y_train)
    expected = reference.predict_proba(arm.split["X_test"])[:, 1]
    np.testing.assert_allclose(arm.output.test_proba.to_numpy(), expected, rtol=0, atol=1e-8)


# ----------------------------------------------------------------------
# F16: family selection on nested out-of-fold probabilities
# ----------------------------------------------------------------------
def test_final_model_family_is_selected_by_nested_oof_brier_f16(arm):
    stage, y = arm.stage, arm.y_train
    brier = {f: brier_score_loss(y, p) for f, p in stage.calibration_oof_.items()}
    assert set(brier) == set(CALIBRATION_FAMILIES)
    expected = min(CALIBRATION_FAMILIES, key=lambda f: (brier[f], CALIBRATION_FAMILIES.index(f)))
    assert stage.calibration_method == stage.calibrated_model.method == expected
    m = stage.calibration_metrics
    for f in CALIBRATION_FAMILIES:
        assert m[f"brier_oof_{f}"] == pytest.approx(brier[f])
    assert (m["selection"], m["oof_selection"], m["metrics_population"]) == (
        "nested_oof_brier", "per_fold_inner_brier", "out_of_fold")


def test_oof_output_uses_each_folds_own_family_choice_f16(arm):
    """Fold k's rows come from the family chosen on fold k's training rows."""
    stage, y = arm.stage, arm.y_train
    fold, families = stage.train_fold_id.to_numpy(), stage.calibration_fold_family_
    assert sorted(families) == list(range(_folds(arm)))
    expected = np.full(len(y), np.nan)
    for k, f in families.items():
        expected[fold == k] = stage.calibration_oof_[f].to_numpy()[fold == k]
    np.testing.assert_array_equal(stage.proba_train_oof.to_numpy(), expected)
    m = stage.calibration_metrics
    assert m["oof_family_by_fold"] == ",".join(families[k] for k in sorted(families))
    assert m["brier_calibrated"] == pytest.approx(brier_score_loss(y, stage.proba_train_oof))


def test_a_folds_family_choice_matches_an_independent_reference_f16(lr5, direct):
    """Fold 0's choice rebuilt with scikit-learn: inner out-of-fold scores of the
    whole pipeline over fold 0's training rows, calibrators cross-fitted over
    the same inner folds, Brier on those rows only."""
    k, y, raw = 5, lr5.y_train, lr5.split["X_train"]
    rows = lr5.stage.train_fold_id.to_numpy() != 0
    X_tr, y_tr = raw[rows], y[rows].to_numpy()
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    pipe = Pipeline([("encode", _Encoder()), ("scale", StandardScaler()),
                     ("model", clone(lr5.stage.model))])
    s = cross_val_predict(pipe, X_tr, y_tr, cv=inner_cv, method="decision_function")
    inner = np.empty(len(y_tr), dtype=int)
    for j, (_, va) in enumerate(inner_cv.split(np.zeros(len(y_tr)), y_tr)):
        inner[va] = j
    brier = {}
    for f in CALIBRATION_FAMILIES:
        p = np.empty(len(y_tr))
        for j in range(k):
            fit = inner != j
            p[~fit] = np.clip(make_calibrator(f).fit(s[fit], y_tr[fit]).predict(s[~fit]), 0, 1)
        brier[f] = brier_score_loss(y_tr, p)
        assert direct.fold_inner_brier[0][f] == pytest.approx(brier[f], abs=1e-10), f
    expected = min(CALIBRATION_FAMILIES, key=lambda f: (brier[f], CALIBRATION_FAMILIES.index(f)))
    assert direct.fold_family[0] == lr5.stage.calibration_fold_family_[0] == expected


def test_final_oof_output_ignores_a_folds_labels_even_when_the_global_choice_flips_f16(monkeypatch):
    """The case that a single choice over all rows gets wrong: shuffling fold
    0's labels flips the global family (synthetic data chosen so it does, by a
    clear margin), yet fold 0's output must not move."""
    split = stage1_split()
    X, y = split["X_train"][LABEL_FREE_FEATURES], split["y_train"]
    fold_ids = fold_assignment(y, cv_folds=5, random_state=SEED).to_numpy()
    monkeypatch.setattr(crossfit_mod, "StratifiedKFold", _pinned_folds(fold_ids)[0])
    fold0 = fold_ids == 0

    def run(labels):
        designs = build_fold_designs(X, labels, cv_folds=5, random_state=SEED)
        return calibrate_oof(LogisticRegression(), designs, X, labels, cv_folds=5,
                             random_state=SEED, n_jobs=1)

    base = run(y)
    moved = run(mut.permute_labels(y, seed=0, rows=y.index[fold0]))
    assert (base.method, moved.method) == ("isotonic", "sigmoid")       # the global choice flipped

    def chosen_once(r):                                                 # the first F16 patch's rule
        return r.calibrated_oof[r.method].to_numpy()[fold0]
    assert not np.allclose(chosen_once(moved), chosen_once(base))       # ...would have leaked
    np.testing.assert_allclose(moved.proba_oof.to_numpy()[fold0], base.proba_oof.to_numpy()[fold0],
                               rtol=0, atol=1e-12)
    assert moved.fold_family[0] == base.fold_family[0]


def test_selection_scores_are_cross_fitted_not_in_sample_f16(lr5, direct):
    """In-sample scoring of the calibrators (the F16 bug) flatters isotonic; the
    selection must use the nested numbers instead."""
    result, y = direct, lr5.y_train
    s = result.base_scores_oof
    in_sample = {f: brier_score_loss(y, make_calibrator(f).fit(s, y).predict(s))
                 for f in CALIBRATION_FAMILIES}
    assert in_sample["isotonic"] < result.brier_oof["isotonic"]           # optimism exists
    for f in CALIBRATION_FAMILIES:
        assert result.brier_oof[f] != pytest.approx(in_sample[f])
        assert result.metrics[f"brier_oof_{f}"] == result.brier_oof[f]


def test_a_requested_family_is_fitted_without_selection(serial_stage1):
    run = fit_glass_stage1(cv_folds=3, calibration_method="isotonic")
    assert run.stage.calibration_method == "isotonic"
    assert set(run.stage.calibration_oof_) == {"isotonic"}
    assert run.stage.calibration_metrics["selection"] == "requested"
    assert run.stage.calibration_metrics["oof_selection"] == "requested"
    assert set(run.stage.calibration_fold_family_.values()) == {"isotonic"}


def test_ties_go_to_sigmoid(monkeypatch, lr5):
    X, y = lr5.split["X_train"][LABEL_FREE_FEATURES], lr5.y_train
    designs = build_fold_designs(X, y, cv_folds=3, random_state=SEED)
    monkeypatch.setattr(ocal_mod, "brier_score_loss", lambda *a, **k: 0.25)
    result = calibrate_oof(LogisticRegression(), designs, X, y, cv_folds=3, random_state=SEED,
                           n_jobs=1)
    assert result.method == "sigmoid"
    assert set(result.fold_family.values()) == {"sigmoid"}


# ----------------------------------------------------------------------
# F19: every inner calibration fold refits its preprocessing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("log_name, methods", [
    ("encoder_log", ("transform",)),
    ("scaler_log", ("transform",)),
    ("oof_log", ("decision_function", "predict_proba")),
])
def test_inner_folds_refit_everything_within_their_outer_training_rows_f19(arm, log_name, methods):
    log, k = getattr(arm, log_name), _folds(arm)
    train = frozenset(arm.X_train.index)
    outer, inner = fold_levels(log, train)
    fits = log.fits()
    assert sorted(inner) == sorted(outer) and all(len(ids) == k for ids in inner.values())
    for outer_fid, inner_ids in inner.items():
        outer_train = fits[outer_fid]
        outer_val = train - outer_train
        inner_log = sub_log(log, inner_ids)
        assert_oof_provenance(inner_log, outer_train, methods=methods)
        assert_never_fit_on(inner_log, outer_val)
        assert not {r for c in inner_log for r in c.keys} & outer_val      # never even scored


def test_inner_encoder_scaler_and_model_share_rows_f19(arm):
    train = arm.X_train.index

    def inner_rows(log):
        fits = log.fits()
        _, inner = fold_levels(log, train)
        return {fits[fid] for ids in inner.values() for fid in ids}
    assert inner_rows(arm.encoder_log) == inner_rows(arm.scaler_log) == inner_rows(arm.oof_log)


# ----------------------------------------------------------------------
# held-out fold labels cannot reach either family's calibrated scores
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm_name, folds", [("glass_lr", 5), ("blackbox_mlp", 3)])
def test_a_folds_own_labels_never_reach_its_calibrated_scores_f3_f19(arm_name, folds):
    """Target encodings (F3), scalers (F4), inner calibration folds (F19), both
    candidate calibrators and the family choice (F16) all sit inside fold 0's
    training rows: neither candidate nor the final output moves."""
    split = stage1_split()
    y = split["y_train"]
    fold_ids = fold_assignment(y, cv_folds=folds, random_state=SEED).to_numpy()
    fold0 = y.index[fold_ids == 0]

    def calibrated(y_new):
        data = {**split, "y_train": y_new}
        fit = fit_glass_stage1 if arm_name == "glass_lr" else fit_mlp_stage1
        run = fit(data, fold_ids=fold_ids, cv_folds=folds)                 # auto: both families
        out = {f: p.loc[fold0].to_numpy() for f, p in run.stage.calibration_oof_.items()}
        out["final"] = run.stage.proba_train_oof.loc[fold0].to_numpy()     # what Stage 4 reads
        return out

    base = calibrated(y)
    own = calibrated(mut.permute_labels(y, seed=5, rows=fold0))
    others = calibrated(mut.permute_labels(y, seed=5, rows=y.index.difference(fold0)))
    for f in (*CALIBRATION_FAMILIES, "final"):
        np.testing.assert_allclose(own[f], base[f], rtol=0, atol=1e-10, err_msg=f)
        assert not np.allclose(others[f], base[f]), f                     # the check can fail


def test_the_shared_routine_reproduces_the_stage_when_called_directly(lr5, direct):
    """The stage adds nothing to the shared procedure: same folds, same
    per-fold preprocessing, same calibration, same output."""
    stage, result = lr5.stage, direct
    assert result.fold_family == stage.calibration_fold_family_
    pd.testing.assert_series_equal(result.proba_oof.rename("stage1_proba_oof"),
                                   stage.proba_train_oof)
    assert (result.method, result.provenance) == (stage.calibration_method, stage.oof_provenance_)
    assert result.metrics == stage.calibration_metrics


# ----------------------------------------------------------------------
# the final scorer
# ----------------------------------------------------------------------
def test_final_scorer_is_the_full_train_model_plus_one_calibrator(arm):
    stage, X_te = arm.stage, arm.X_test
    scorer = stage.calibrated_model
    assert isinstance(scorer, Stage1CalibratedModel) and scorer.base is stage.model
    assert scorer.method == stage.calibration_method
    X = unrecorded_scale(stage, X_te)
    expected = np.clip(scorer.calibrator.predict(base_scores(stage.model, X)), 0, 1)
    np.testing.assert_allclose(arm.output.test_proba.to_numpy(), expected)
    np.testing.assert_allclose(scorer.predict_proba(X).sum(axis=1), 1.0)


def test_the_base_model_is_the_unchanged_full_train_fit(lr5):
    stage = lr5.stage
    X = unrecorded_scale(stage, lr5.X_train)
    fresh = clone(stage.model).fit(X, lr5.y_train)
    np.testing.assert_array_equal(fresh.coef_, stage.model.coef_)
    np.testing.assert_array_equal(fresh.intercept_, stage.model.intercept_)


# ----------------------------------------------------------------------
# determinism, alignment, parity
# ----------------------------------------------------------------------
def test_refits_are_deterministic_and_aligned(arm):
    again = (fit_glass_stage1(cv_folds=5) if isinstance(arm.stage, CalibratedLRStage)
             else fit_mlp_stage1())
    pd.testing.assert_series_equal(again.stage.proba_train_oof, arm.stage.proba_train_oof)
    pd.testing.assert_series_equal(again.output.test_proba, arm.output.test_proba)
    assert again.stage.calibration_metrics == arm.stage.calibration_metrics
    for f, p in arm.stage.calibration_oof_.items():
        assert p.index.equals(arm.X_train.index), f


def test_both_arms_report_calibration_the_same_way(lr5, mlp_run):
    lr, mlp = lr5.stage, mlp_run.stage
    assert set(lr.calibration_metrics) == set(mlp.calibration_metrics)
    assert LEGACY_METRIC_KEYS <= set(lr.calibration_metrics)
    assert type(lr.calibrated_model) is type(mlp.calibrated_model) is Stage1CalibratedModel
    for stage, k in ((lr, 5), (mlp, 3)):
        assert stage.oof_provenance_ == (
            f"refittable (nested cross-fit: {k} outer x {k} inner folds; "
            "encoder, scaler, model and calibrator fitted per fold)")


# ----------------------------------------------------------------------
# calibrators and guards
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 80.0])          # 80: the large-score rescaling path
def test_platt_calibrator_matches_scikit_learn(scale):
    sk = pytest.importorskip("sklearn.calibration")
    if not hasattr(sk, "_SigmoidCalibration"):
        pytest.skip("scikit-learn's private sigmoid calibrator is not available")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = scale * (y - 0.5 + rng.normal(0, 0.8, 400))
    ours = PlattCalibrator().fit(s, y)
    theirs = sk._SigmoidCalibration().fit(s, y)
    np.testing.assert_allclose([ours.a_, ours.b_], [theirs.a_, theirs.b_], rtol=1e-6, atol=1e-9)
    grid = np.linspace(s.min(), s.max(), 25)
    np.testing.assert_allclose(ours.predict(grid), theirs.predict(grid), rtol=1e-6, atol=1e-9)


def test_calibrator_families_and_invalid_requests():
    iso = make_calibrator("isotonic")
    assert isinstance(iso, IsotonicRegression) and iso.out_of_bounds == "clip"
    assert isinstance(make_calibrator("sigmoid"), PlattCalibrator)
    with pytest.raises(ValueError, match="unknown calibration method"):
        make_calibrator("beta")
    X = stage1_split()["X_train"][LABEL_FREE_FEATURES]
    y = stage1_split()["y_train"]
    with pytest.raises(ValueError, match="method must be"):
        calibrate_oof(LogisticRegression(), build_fold_designs(X, y, cv_folds=3, random_state=SEED),
                      X, y, cv_folds=3, random_state=SEED, method="temperature", n_jobs=1)


def test_a_prefit_base_model_is_refused():
    FrozenEstimator = pytest.importorskip("sklearn.frozen").FrozenEstimator
    split = stage1_split()
    X, y = split["X_train"][LABEL_FREE_FEATURES], split["y_train"]
    designs = build_fold_designs(X, y, cv_folds=3, random_state=SEED)
    frozen = FrozenEstimator(LogisticRegression().fit([[0.0], [1.0]], [0, 1]))
    with pytest.raises(CalibrationLeakageError):
        calibrate_oof(frozen, designs, X, y, cv_folds=3, random_state=SEED, n_jobs=1)


# ----------------------------------------------------------------------
# artifact and StageOutput contracts (Stages 2-4 read these)
# ----------------------------------------------------------------------
@pytest.fixture
def plain_split():
    """Plain (unrecorded) fits: recorder subclasses are test-local and do not pickle."""
    split = stage1_split()
    X_tr, X_te, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    return split, X_tr, X_te


def test_lr_artifact_round_trip(plain_split, serial_stage1, tmp_path):
    split, X_tr, X_te = plain_split
    stage = _quiet(CalibratedLRStage(tune_hyperparameters=False, cv_folds=3).fit,
                   X_tr, split["y_train"], X_train_raw=split["X_train"])
    _quiet(stage.evaluate, X_te, split["y_test"], plot=False)
    path = _quiet(save_artifact, stage, str(tmp_path))["path"]
    art = _quiet(load_artifact, path)
    p = art["calibrated_model"].predict_proba(art["scaler"].transform(X_te))[:, 1]
    np.testing.assert_allclose(p, stage.predict_proba(X_te))
    assert art["calibration_method"] == stage.calibration_method
    assert LEGACY_METRIC_KEYS <= set(art["calibration_metrics"])
    assert not art["oof_provenance"].upper().startswith("PREFIT")       # Stage 4's check
    out = StageOutput.from_dict(art["stage_output"])
    pd.testing.assert_series_equal(out.train_proba_oof, stage.proba_train_oof)
    assert out.config["oof_preprocessing"]["applies_to"].endswith("calibration_inner_folds")


def test_mlp_artifact_round_trip(plain_split, serial_stage1, tmp_path):
    split, X_tr, X_te = plain_split
    stage = CalibratedStage1MLP(**MLP_TEST_CONFIG).fit(X_tr, split["y_train"],
                                                       X_train_raw=split["X_train"])
    output = stage.to_stage_output(X_te, split["y_test"])
    path = _quiet(save_stage1_mlp, stage, tmp_path, stage_output=output)
    art = joblib.load(path)
    p = art["calibrated_model"].predict_proba(art["scaler"].transform(X_te))[:, 1]
    np.testing.assert_allclose(p, output.test_proba.to_numpy())
    meta = json.loads((tmp_path / "stage1_mlp_metadata.json").read_text())
    assert LEGACY_METRIC_KEYS <= set(meta["calibration_metrics"])
    assert StageOutput.from_dict(art["stage_output"]).calibration_method == stage.calibration_method
