"""
PR 37: fixed-name model saves keep every version (shared.artifact_versions).

``save_rf_router`` and ``save_stage1_mlp`` used to overwrite ``<stem>.joblib``
on every run. Each save now writes an immutable, read-only version
``<stem>_<id>.joblib`` (with its own sidecars) and returns its path;
``<stem>.joblib`` is a mutable latest alias, a byte copy of the newest version
whose metadata says so. A fixed-name file written before the change is kept as
``<stem>_preserved_<sha12>`` before the alias is replaced. Model payloads are
unchanged.

Stub models only; nothing is fitted and no real artifact is read.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from blackbox_pipeline.models.mlp import load_stage1_mlp, save_stage1_mlp
from blackbox_pipeline.models.rf_router import (find_artifact_for_split, load_rf_router,
                                                save_rf_router)
from shared import artifact_versions as av

STEM = "rf_router_two_pass"
RF_TABLES = ("_pass1_sweep.csv", "_pass2_sweep.csv")
MLP_TABLES = ("_threshold_sweep.csv", "_tuning_trials.csv", "_training_history.csv")


def _quiet(fn, *args, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _dicty(**fields):
    return SimpleNamespace(to_dict=lambda: dict(fields))


def _router(t1=0.2, fingerprint="FP"):
    sweep = pd.DataFrame({"threshold": [0.1, t1], "recall": [1.0, 0.9]})
    thresholds = SimpleNamespace(
        to_dict=lambda: {"t1": t1, "t2": 0.6},
        pass1=SimpleNamespace(sweep=sweep), pass2=SimpleNamespace(sweep=sweep * 2))
    return SimpleNamespace(
        is_fitted=True,
        config=SimpleNamespace(mode="two_pass", to_dict=lambda: {"mode": "two_pass"},
                               constraints=_dicty(min_recall=0.9)),
        feature_names_=["a", "b"], training_base_rate=0.11, thresholds_=thresholds,
        bands_=_dicty(high=0.8), fit_report_={"t1": t1}, split_fingerprint_=fingerprint,
        train_label_fingerprint_="LBL", pass1_model_={"forest": 1, "t1": t1},
        pass2_model_={"forest": 2}, remainder_mask_=[True, False])


def _mlp(seed=0):
    frame = pd.DataFrame({"threshold": [0.1, 0.2], "f2": [0.3, 0.4 + seed]})
    return SimpleNamespace(
        fitted=True, config=_dicty(hidden=8), feature_names_=["a"], scaler={"s": seed},
        model=SimpleNamespace(history_=frame), calibrated_model={"c": seed},
        best_params={"lr": 0.1}, best_cv_roc_auc=0.7, calibration_method="isotonic",
        calibration_metrics={}, optimal_threshold=0.3, cv_f2=0.5,
        oof_provenance_="nested cross-fit", THRESHOLD_SOURCE="oof",
        proba_train_oof=pd.Series([0.1, 0.2]), train_fold_id=pd.Series([0, 1]),
        y_train_=pd.Series([0, 1]), threshold_sweep=frame, tuning_trials=frame)


def _read_only(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _versions(folder: Path, stem: str):
    return sorted(p for p in folder.glob(f"{stem}_2*.joblib"))


# ======================================================================
# RF Router: every save is a new immutable version; the fixed name is an alias
# ======================================================================
def test_each_rf_router_save_keeps_its_own_immutable_version(tmp_path):
    first = _quiet(save_rf_router, _router(t1=0.2), tmp_path, name=STEM)
    first_bytes = _bytes(first)
    second = _quiet(save_rf_router, _router(t1=0.3), tmp_path, name=STEM)

    assert first != second and _versions(tmp_path, STEM) == sorted([first, second])
    assert _bytes(first) == first_bytes                                   # untouched
    assert load_rf_router(first)["thresholds"]["t1"] == 0.2
    assert load_rf_router(second)["thresholds"]["t1"] == 0.3
    for version in (first, second):
        vstem = version.name[:-len(".joblib")]
        assert _read_only(version)
        for suffix in ("_metadata.json", *RF_TABLES):
            assert _read_only(tmp_path / f"{vstem}{suffix}")
        meta = json.loads((tmp_path / f"{vstem}_metadata.json").read_text())
        assert meta["artifact_file"] == {
            "name": version.name, "version_id": vstem[len(STEM) + 1:],
            "sha256": av.file_sha256(version), "immutable": True}
        assert "latest_alias" not in meta


def test_the_fixed_name_is_a_mutable_alias_of_the_newest_version(tmp_path):
    _quiet(save_rf_router, _router(t1=0.2), tmp_path, name=STEM)
    newest = _quiet(save_rf_router, _router(t1=0.3), tmp_path, name=STEM)
    alias = tmp_path / f"{STEM}.joblib"
    assert _bytes(alias) == _bytes(newest) and not _read_only(alias)
    meta = json.loads((tmp_path / f"{STEM}_metadata.json").read_text())
    assert meta["latest_alias"]["mutable"] is True
    assert meta["latest_alias"]["points_to"] == newest.name
    assert meta["latest_alias"]["sha256"] == av.file_sha256(newest)
    assert av.alias_target(alias) == newest
    vstem = newest.name[:-len(".joblib")]
    for suffix in RF_TABLES:                         # the tracked sweep CSVs stay current
        assert _bytes(tmp_path / f"{STEM}{suffix}") == _bytes(tmp_path / f"{vstem}{suffix}")
    # readers of the fixed name (notebook Cells 35/37, meta_xgb runner) still load it
    assert load_rf_router(alias, expected_split_fingerprint="FP")["thresholds"]["t1"] == 0.3


def test_the_model_payload_is_unchanged(tmp_path):
    path = _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    assert set(joblib.load(path)) == {
        "artifact_version", "created_utc", "mode", "config", "constraints", "feature_names",
        "training_base_rate", "thresholds", "bands", "fit_report", "split_fingerprint",
        "train_label_fingerprint", "test_metrics", "environment", "pass1_model",
        "pass2_model", "remainder_mask"}


def test_an_unversioned_fixed_name_router_is_kept_before_the_alias_is_replaced(tmp_path):
    # A router saved before PR 37: fixed name, no immutable twin.
    legacy = {".joblib": b"october router", "_metadata.json": b"{}",
              "_pass1_sweep.csv": b"threshold\n0.1\n", "_pass2_sweep.csv": b"threshold\n0.2\n"}
    for suffix, data in legacy.items():
        (tmp_path / f"{STEM}{suffix}").write_bytes(data)
    old = 1_700_000_000
    os.utime(tmp_path / f"{STEM}.joblib", (old, old))
    keep = f"{STEM}_preserved_{av.file_sha256(tmp_path / f'{STEM}.joblib')[:12]}"

    _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    for suffix, data in legacy.items():
        kept = tmp_path / f"{keep}{suffix}"
        assert _bytes(kept) == data and _read_only(kept)
    assert (tmp_path / f"{keep}.joblib").stat().st_mtime == old      # still the older file
    assert _bytes(tmp_path / f"{STEM}.joblib") != legacy[".joblib"]

    # The alias is now versioned: the next save keeps nothing more.
    _quiet(save_rf_router, _router(t1=0.3), tmp_path, name=STEM)
    assert len(list(tmp_path.glob(f"{STEM}_preserved_*.joblib"))) == 1
    assert len(_versions(tmp_path, STEM)) == 2


def test_a_preserved_name_holding_other_content_is_never_overwritten(tmp_path):
    (tmp_path / f"{STEM}.joblib").write_bytes(b"october router")
    keep = tmp_path / f"{STEM}_preserved_{av.file_sha256(tmp_path / f'{STEM}.joblib')[:12]}.joblib"
    keep.write_bytes(b"something else")
    with pytest.raises(av.ArtifactExistsError):
        _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    assert _bytes(keep) == b"something else"
    assert _bytes(tmp_path / f"{STEM}.joblib") == b"october router"


def test_discovery_returns_the_immutable_version_not_the_alias(tmp_path):
    path = _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    _, found = _quiet(find_artifact_for_split, [tmp_path], "rf_router_*.joblib", "FP")
    assert found == path


# ======================================================================
# immutability itself
# ======================================================================
@pytest.mark.parametrize("hard_links", [True, False])
def test_an_existing_version_file_is_never_replaced(tmp_path, monkeypatch, hard_links):
    if not hard_links:
        def no_link(*_):
            raise OSError("hard links unsupported")
        monkeypatch.setattr(av.os, "link", no_link)
    final = tmp_path / "x_20260101_000000_000000.joblib"
    av._write_immutable(final, lambda p: p.write_bytes(b"first"))
    assert _bytes(final) == b"first" and _read_only(final)
    with pytest.raises(av.ArtifactExistsError):
        av._write_immutable(final, lambda p: p.write_bytes(b"second"))
    assert _bytes(final) == b"first"
    assert not list(tmp_path.glob(".tmp-*"))                  # no temporary files left


def test_a_failed_save_leaves_the_alias_and_earlier_versions_alone(tmp_path):
    first = _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    alias_before = _bytes(tmp_path / f"{STEM}.joblib")

    def broken(_path):
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        av.save_versioned(tmp_path, STEM, write_model=broken, metadata={},
                          dump_metadata=json.dumps)
    assert _versions(tmp_path, STEM) == [first]
    assert _bytes(tmp_path / f"{STEM}.joblib") == alias_before
    assert not list(tmp_path.glob(".tmp-*"))


def test_alias_target_refuses_an_alias_that_no_longer_matches(tmp_path):
    _quiet(save_rf_router, _router(), tmp_path, name=STEM)
    alias = tmp_path / f"{STEM}.joblib"
    assert av.alias_target(alias) is not None
    alias.write_bytes(b"edited by hand")
    assert av.alias_target(alias) is None
    meta_path = tmp_path / f"{STEM}_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["latest_alias"]["points_to"] = "../elsewhere.joblib"
    meta_path.write_text(json.dumps(meta))
    assert av.alias_target(alias) is None
    assert av.alias_target(tmp_path / "missing.joblib") is None


# ======================================================================
# Stage 1 MLP: same rule
# ======================================================================
def test_each_mlp_save_keeps_its_own_version_and_returns_it(tmp_path):
    first = _quiet(save_stage1_mlp, _mlp(seed=0), tmp_path)
    second = _quiet(save_stage1_mlp, _mlp(seed=1), tmp_path)
    assert first != second and _versions(tmp_path, "stage1_mlp") == sorted([first, second])
    assert load_stage1_mlp(first)["scaler"] == {"s": 0}
    assert load_stage1_mlp(second)["scaler"] == {"s": 1}
    assert _read_only(first) and _read_only(second)
    alias = tmp_path / "stage1_mlp.joblib"
    assert av.alias_target(alias) == second and not _read_only(alias)
    vstem = second.name[:-len(".joblib")]
    for suffix in ("_metadata.json", *MLP_TABLES):
        assert (tmp_path / f"{vstem}{suffix}").is_file()
        assert (tmp_path / f"stage1_mlp{suffix}").is_file()            # tracked alias names
    for suffix in MLP_TABLES:
        assert _bytes(tmp_path / f"stage1_mlp{suffix}") == _bytes(tmp_path / f"{vstem}{suffix}")


def test_an_unversioned_fixed_name_mlp_is_kept(tmp_path):
    (tmp_path / "stage1_mlp.joblib").write_bytes(b"november mlp")
    (tmp_path / "stage1_mlp_tuning_trials.csv").write_bytes(b"trial\n1\n")
    keep = f"stage1_mlp_preserved_{av.file_sha256(tmp_path / 'stage1_mlp.joblib')[:12]}"
    _quiet(save_stage1_mlp, _mlp(), tmp_path)
    assert _bytes(tmp_path / f"{keep}.joblib") == b"november mlp"
    assert _bytes(tmp_path / f"{keep}_tuning_trials.csv") == b"trial\n1\n"
