"""
PR 39, Commit 2, Step 2B (decision D2 (a)): the Stage 1 feature rename.

Three Stage 1 features were renamed; values, definitions, order and the
per-fold refits are unchanged:

    cellular_crisis       -> cellular_downturn
    euribor3m_local_rate  -> subscribe_rate_by_euribor_bin
    dow_month_encoded     -> subscribe_rate_by_dow_month

Covered here:

* the rename map, the order and the two versioned feature contracts
  (2.0.0 current, 1.0.0 historical);
* value identity: the engineered features and the per-fold designs equal
  anchors recorded from the pre-rename code (``fb233f9`` + Step 2A) on the
  same synthetic split, so a value change fails here although the names moved;
* no historical name left in the Stage 1 runtime modules;
* historical compatibility: artifacts saved before the rename (version 2.0,
  contract 1.0.0) load as saved, pass the Stage 2 boundary check and feed the
  Stage 4 adapter exactly like current (3.0) artifacts. Nothing is translated.

Stages 2 and 3 compute their own features and keep their names.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import blackbox_pipeline.models.mlp.stage as mlp_stage_mod
import glass_pipeline.lr.lr_stage_calibrated as lr_mod
import shared.stage1.oof_calibration as ocal_mod
from blackbox_pipeline.models.mlp import (STAGE1_FEATURES, CalibratedStage1MLP, load_stage1_mlp,
                                          save_stage1_mlp, select_features)
from glass_pipeline.lr.artifacts import ARTIFACT_VERSION, load_artifact, save_artifact
from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
from shared.stage1.crossfit import build_fold_designs, inner_fold_designs
from shared.stage1.feature_engineering import (
    HISTORICAL_LR_FEATURES, HISTORICAL_STAGE1_FEATURE_CONTRACT,
    HISTORICAL_TARGET_ENCODED_FEATURES, LR_FEATURES, STAGE1_FEATURE_CONTRACT,
    STAGE1_FEATURE_RENAMES, TARGET_ENCODED_FEATURES, LRFeatureEngineer,
    artifact_feature_contract, current_feature_contract_record, engineer_features,
    feature_contract_record, stage1_feature_contract,
)
from shared.stage1.oof_calibration import STAGE1_CALIBRATION_FAMILY
from shared.stage2.diagnostics import boundary as d
from shared.stage4.adapters import stage_output_stream
from tests.stage1.conftest import MLP_TEST_CONFIG, _serial_calibrate, stage1_split
from tests.support import diagnostics_world as dw

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_NAME = re.compile(r"cellular_crisis|euribor3m_local_rate|dow_month_encoded")


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _printed(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# ======================================================================
# names, order, contracts
# ======================================================================
def test_rename_map_takes_the_historical_names_to_the_current_ones_in_order():
    assert LR_FEATURES == ["cellular_downturn", "subscribe_rate_by_euribor_bin",
                           "subscribe_rate_by_dow_month"]
    assert HISTORICAL_LR_FEATURES == ("cellular_crisis", "euribor3m_local_rate",
                                      "dow_month_encoded")
    assert [STAGE1_FEATURE_RENAMES[f] for f in HISTORICAL_LR_FEATURES] == LR_FEATURES
    assert tuple(STAGE1_FEATURE_RENAMES[f] for f in HISTORICAL_TARGET_ENCODED_FEATURES) == \
        TARGET_ENCODED_FEATURES == ("subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month")
    with pytest.raises(TypeError):                                     # read-only map
        STAGE1_FEATURE_RENAMES["cellular_crisis"] = "x"                # type: ignore[index]


def test_both_arms_use_the_current_contract():
    assert STAGE1_FEATURES == tuple(LR_FEATURES)
    split = stage1_split()
    X_tr, _, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    assert list(X_tr.columns) == LR_FEATURES
    with pytest.raises(ValueError, match="Stage 1 expects exactly"):    # old names are not inputs
        select_features(X_tr.rename(columns={v: k for k, v in STAGE1_FEATURE_RENAMES.items()}))


def test_feature_contracts_are_versioned_and_pinned():
    for contract, version, names, fingerprint in (
            (STAGE1_FEATURE_CONTRACT, "2.0.0", LR_FEATURES, "d0d8b07f2ca5e81d"),
            (HISTORICAL_STAGE1_FEATURE_CONTRACT, "1.0.0", HISTORICAL_LR_FEATURES,
             "dae2c0b25f683f3a")):
        assert (contract["id"], contract["version"]) == ("stage1-features", version)
        assert contract["features"] == tuple(names)
        assert contract["fingerprint"] == fingerprint == \
            hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()[:16]
        record = feature_contract_record(contract)
        assert json.loads(json.dumps(record)) == record               # JSON- and pickle-safe
        assert record["features"] == list(names)


def test_contract_lookup_is_exact_and_order_sensitive():
    assert stage1_feature_contract(LR_FEATURES) is STAGE1_FEATURE_CONTRACT
    assert stage1_feature_contract(list(HISTORICAL_LR_FEATURES)) is \
        HISTORICAL_STAGE1_FEATURE_CONTRACT
    mixed = [LR_FEATURES[0], HISTORICAL_LR_FEATURES[1], LR_FEATURES[2]]
    for other in (list(reversed(LR_FEATURES)), mixed, LR_FEATURES[:2],
                  [*LR_FEATURES, "extra"], [], None):
        assert stage1_feature_contract(other) is None


def test_new_artifacts_can_only_carry_the_current_contract():
    assert current_feature_contract_record(LR_FEATURES) == \
        feature_contract_record(STAGE1_FEATURE_CONTRACT)
    with pytest.raises(ValueError, match="historical contract 1.0.0"):
        current_feature_contract_record(list(HISTORICAL_LR_FEATURES))
    for other in (["a"], list(reversed(LR_FEATURES)), None):
        with pytest.raises(ValueError, match="no registered contract"):
            current_feature_contract_record(other)


# ======================================================================
# value identity: anchors recorded from the pre-rename code
# ======================================================================
#: Recorded by running the code at fb233f9 + the two Step 2A patches (names
#: cellular_crisis / euribor3m_local_rate / dow_month_encoded) on
#: ``stage1_split()``: [sum, weighted sum] of each matrix, the weights
#: ((7 i + 13 j) mod 17) - 8 making each anchor sensitive to row and column
#: order. Outer folds: build_fold_designs(cv_folds=5, random_state=42);
#: inner: inner_fold_designs of outer fold 0, same settings. Each fold row is
#: [X_train sum, X_train weighted, X_val sum, X_val weighted] (scaled).
PRE_RENAME_FULL_TRAIN = [[184.85427893207589, 12.534524523295975],     # train
                         [55.397567607316546, 7.050708265042524]]      # test
PRE_RENAME_FIRST_COLUMN_POSITIVES = [65, 16]
PRE_RENAME_OUTER = [
    [6.465938895416912e-13, -73.17076113650738, 17.25198759867644, 74.31539448110536],
    [2.4868995751603507e-14, 5.1316554099933995, -0.6464719230900027, -80.43228284205045],
    [-2.4158453015843406e-13, 76.73674700835741, -6.462821134367025, -0.9701532447440151],
    [-9.947598300641403e-14, 45.43313711677947, -6.320874487419333, 50.31831877246715],
    [-1.7763568394002505e-14, -143.87913755827321, -5.668258432141386, -33.3932132045704],
]
PRE_RENAME_INNER = [
    [-5.844214001626824e-13, 89.10853446767042, -5.614643036693328, 37.33128674108369],
    [-3.907985046680551e-13, -146.20412121401085, -14.246843503551146, 16.29960325050811],
    [-1.829647544582258e-13, -41.47520288014806, 42.118020349366894, -9.873448429809777],
    [4.600764214046649e-13, 194.03062723519923, -12.816926185000025, 68.38790329231698],
    [2.984279490192421e-13, 87.16806649720752, -30.940078699135395, 28.858333123589873],
]


def _anchor(M) -> list:
    M = np.asarray(M, dtype=float)
    i, j = np.indices(M.shape)
    return [float(M.sum()), float((M * (((i * 7 + j * 13) % 17) - 8.0)).sum())]


def _close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10, atol=1e-8)


@pytest.fixture(scope="module")
def engineered():
    split = stage1_split()
    X_raw, y = split["X_train"], split["y_train"]
    eng = LRFeatureEngineer()
    tr = _quiet(eng.fit_transform, X_raw, y)
    te = _quiet(eng.transform, split["X_test"])
    return split, tr, te


def test_engineered_values_equal_the_pre_rename_anchors(engineered):
    split, tr, te = engineered
    assert list(tr.columns) == list(te.columns) == LR_FEATURES
    assert tr.index.equals(split["X_train"].index) and te.index.equals(split["X_test"].index)
    _close([_anchor(tr), _anchor(te)], PRE_RENAME_FULL_TRAIN)
    assert [int(tr["cellular_downturn"].sum()), int(te["cellular_downturn"].sum())] == \
        PRE_RENAME_FIRST_COLUMN_POSITIVES


def test_per_fold_designs_equal_the_pre_rename_anchors(engineered):
    """The per-fold refits (audit F3/F19) build the same designs as before."""
    split, tr, _ = engineered
    X_raw, y = split["X_train"], split["y_train"]
    outer = _quiet(build_fold_designs, tr, y, cv_folds=5, random_state=42, X_raw=X_raw)
    inner = _quiet(inner_fold_designs, outer[0], tr, y, cv_folds=5, random_state=42, X_raw=X_raw)
    _close([_anchor(f.X_train) + _anchor(f.X_val) for f in outer], PRE_RENAME_OUTER)
    _close([_anchor(f.X_train) + _anchor(f.X_val) for f in inner], PRE_RENAME_INNER)


def test_the_target_encoded_columns_are_still_refit_per_fold(engineered):
    split, tr, _ = engineered
    with pytest.raises(ValueError, match="subscribe_rate_by_euribor_bin"):
        build_fold_designs(tr, split["y_train"], cv_folds=5, random_state=42)     # no X_raw


def test_no_historical_name_is_left_in_the_stage1_runtime():
    """Old names remain only in the compatibility constants of
    ``shared.stage1.feature_engineering``, never in the Stage 1 code paths."""
    scope = [*ROOT.glob("glass_pipeline/lr/*.py"), *ROOT.glob("blackbox_pipeline/models/mlp/*.py"),
             *ROOT.glob("shared/stage1/*.py"), *ROOT.glob("stage1_calibration/*.py")]
    scope = [p for p in scope if p.name != "feature_engineering.py"]
    assert len(scope) > 10
    hits = {str(p.relative_to(ROOT)): HISTORICAL_NAME.findall(p.read_text(encoding="utf-8"))
            for p in scope}
    assert {k: v for k, v in hits.items() if v} == {}


# ======================================================================
# artifacts: current (3.0) and historical (2.0) side by side
# ======================================================================
@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """One LR and one MLP artifact written by the current code, plus each as
    the pre-rename code wrote it (version 2.0, historical names, no record)."""
    split = stage1_split()
    X_tr, X_te, _ = _quiet(engineer_features, split["X_train"], split["X_test"], split["y_train"])
    out = tmp_path_factory.mktemp("stage1_contract")
    with pytest.MonkeyPatch.context() as mp:
        serial = _serial_calibrate(ocal_mod.calibrate_oof)
        mp.setattr(lr_mod, "calibrate_oof", serial)
        mp.setattr(mlp_stage_mod, "calibrate_oof", serial)
        lr = _quiet(CalibratedLRStage(tune_hyperparameters=False, cv_folds=3).fit,
                    X_tr, split["y_train"], X_train_raw=split["X_train"])
        _quiet(lr.evaluate, X_te, split["y_test"], plot=False)
        mlp = _quiet(CalibratedStage1MLP(**MLP_TEST_CONFIG).fit, X_tr, split["y_train"],
                     X_train_raw=split["X_train"])
    lr_path = Path(_quiet(save_artifact, lr, str(out / "lr"))["path"])
    mlp_path = _quiet(save_stage1_mlp, mlp, out / "mlp",
                      stage_output=mlp.to_stage_output(X_te, split["y_test"]))
    paths = {"lr": lr_path, "mlp": mlp_path}
    for arm, path in list(paths.items()):
        old = _as_saved_before_the_rename(joblib.load(path))
        paths[f"{arm}_historical"] = out / f"{arm}_historical.joblib"
        joblib.dump(old, paths[f"{arm}_historical"])
    return split, paths, out


def _as_saved_before_the_rename(art: dict) -> dict:
    old = {k: v for k, v in art.items() if k != "feature_contract"}
    old["artifact_version"] = "2.0"
    old["feature_names"] = list(HISTORICAL_LR_FEATURES)
    old["stage_output"] = {**art["stage_output"], "feature_names": list(HISTORICAL_LR_FEATURES)}
    return old


LOADERS = {"lr": load_artifact, "mlp": load_stage1_mlp}


@pytest.mark.parametrize("arm", ["lr", "mlp"])
def test_new_artifacts_record_the_current_contract(artifacts, arm):
    _, paths, out = artifacts
    art, printed = _printed(LOADERS[arm], paths[arm])
    assert ARTIFACT_VERSION == art["artifact_version"] == "3.0" and "⚠️" not in printed
    assert art["feature_names"] == art["stage_output"]["feature_names"] == LR_FEATURES
    assert art["feature_contract"] == feature_contract_record(STAGE1_FEATURE_CONTRACT)
    assert artifact_feature_contract(art) is STAGE1_FEATURE_CONTRACT
    assert art["calibration_method"] == art["calibration_requested"] == STAGE1_CALIBRATION_FAMILY
    if arm == "mlp":
        meta = json.loads((out / "mlp" / "stage1_mlp_metadata.json").read_text())
        assert meta["feature_contract"] == art["feature_contract"]


@pytest.mark.parametrize("arm", ["lr", "mlp"])
def test_historical_artifacts_load_as_saved_without_a_warning(artifacts, arm):
    _, paths, _ = artifacts
    art, printed = _printed(LOADERS[arm], paths[f"{arm}_historical"])
    assert art["artifact_version"] == "2.0" and printed == ""
    assert "feature_contract" not in art                                   # nothing added
    assert art["feature_names"] == art["stage_output"]["feature_names"] == \
        list(HISTORICAL_LR_FEATURES)                                       # nothing renamed
    assert artifact_feature_contract(art) is HISTORICAL_STAGE1_FEATURE_CONTRACT
    assert artifact_feature_contract(art["stage_output"]) is HISTORICAL_STAGE1_FEATURE_CONTRACT


@pytest.mark.parametrize("arm", ["lr", "mlp"])
def test_a_contract_record_that_disagrees_with_the_names_is_refused(artifacts, arm, tmp_path):
    _, paths, _ = artifacts
    art = joblib.load(paths[arm])
    forged = {**art, "feature_contract": feature_contract_record(HISTORICAL_STAGE1_FEATURE_CONTRACT)}
    joblib.dump(forged, tmp_path / "forged.joblib")
    with pytest.raises(ValueError, match="does not match"):
        _quiet(LOADERS[arm], tmp_path / "forged.joblib")
    split_names = {**art, "stage_output": {**art["stage_output"],
                                           "feature_names": list(HISTORICAL_LR_FEATURES)}}
    with pytest.raises(ValueError, match="disagree"):
        artifact_feature_contract(split_names)
    with pytest.raises(ValueError, match="no registered"):
        artifact_feature_contract({**art, "feature_names": ["f1", "f2"]})


@pytest.mark.parametrize("arm", ["lr", "mlp"])
def test_unsupported_versions_still_warn(artifacts, arm, tmp_path):
    _, paths, _ = artifacts
    joblib.dump({**joblib.load(paths[f"{arm}_historical"]), "artifact_version": "1.0"},
                tmp_path / "v1.joblib")
    _, printed = _printed(LOADERS[arm], tmp_path / "v1.joblib")
    assert "⚠️" in printed and "1.0" in printed


@pytest.mark.parametrize("arm, model", [("lr", "calibrated_lr"), ("mlp", "calibrated_mlp")])
def test_stage4_reads_historical_and_current_stage1_artifacts_alike(artifacts, arm, model):
    split, paths, _ = artifacts
    n_train = len(split["y_train"])
    current, checks_new = stage_output_stream(paths[arm], arm, model, n_train)
    historical, checks_old = stage_output_stream(paths[f"{arm}_historical"], arm, model, n_train)
    for checks in (checks_new, checks_old):
        failed = [c[2] for c in checks if not c[3]()]
        assert failed == []
    np.testing.assert_array_equal(current.train_proba, historical.train_proba)
    np.testing.assert_array_equal(current.test_proba, historical.test_proba)
    np.testing.assert_array_equal(current.train_fold_id, historical.train_fold_id)
    assert current.train_index.equals(historical.train_index)
    assert current.standalone_threshold == historical.standalone_threshold
    assert current.notes["calibration_method"] == historical.notes["calibration_method"] == \
        STAGE1_CALIBRATION_FAMILY


# ======================================================================
# Stage 2 boundary: the frozen check label, either contract
# ======================================================================
@pytest.mark.parametrize("names", [LR_FEATURES, list(HISTORICAL_LR_FEATURES)],
                         ids=["current", "historical"])
def test_stage2_boundary_reads_either_stage1_contract(names):
    w = dw.world()
    w["lr"]["stage_output"]["feature_names"] = list(names)
    view = d.extract_lr_view(dw.trapped(w["lr"]), dw.reference(w))
    assert view is not None


@pytest.mark.parametrize("names", [
    list(reversed(LR_FEATURES)),
    [LR_FEATURES[0], HISTORICAL_LR_FEATURES[1], LR_FEATURES[2]],
    list(HISTORICAL_LR_FEATURES)[:2],
], ids=["reordered", "mixed", "subset"])
def test_stage2_boundary_refuses_any_other_list_under_the_same_label(names):
    w = dw.world()
    w["lr"]["stage_output"]["feature_names"] = names
    with pytest.raises(d.BoundaryCheckError, match=re.escape("feature_names = shared.stage1 LR_FEATURES")):
        d.extract_lr_view(dw.trapped(w["lr"]), dw.reference(w))
