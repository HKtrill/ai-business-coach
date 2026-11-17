"""
PR 39 Step 4A: artifacts read by pinned path and SHA-256 (shared.pinned_artifacts).

* The Stage 2 pins are exactly the accepted PR 38 run's manifest entries, and
  the manifest is the version-controlled file with its pinned SHA-256.
* A missing, resized or changed file is refused; a file is unpickled from the
  same bytes that were checked.
* ``load_pinned_stage2`` refuses pins that are not the manifest's, and checks
  both routers against the Stage 4 router contract on the train and test side.
* A run's pins file: refused when the variable is unset, the file or its
  sidecar is missing, it was edited, the role is unknown or unpinned, or the
  pinned artifact changed; written once, every file verified first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest

import shared.pinned_artifacts as P
from shared.pinned_artifacts import PinError, PinnedFile
from tests.support.notebooks import BANK_ROOT
from tests.support.stage4_world import ROUTER_FP, build_world, make_router

RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file(root: Path, rel: str, data: bytes) -> PinnedFile:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return PinnedFile("stage1.lr", rel, _sha(data), len(data))


# ----------------------------------------------------------------------
# the accepted run's pins, against its committed manifest
# ----------------------------------------------------------------------
def test_the_stage2_pins_are_the_accepted_runs_manifest_entries():
    data = (BANK_ROOT / P.STAGE2_MANIFEST).read_bytes()
    assert _sha(data) == P.STAGE2_MANIFEST_SHA256
    man = json.loads(data)
    assert (man["run_id"], man["role"]) == (P.STAGE2_RUN_ID, "primary")
    assert P.stage2_manifest_problems(BANK_ROOT) == []
    roles = {p.role: p for p in P.STAGE2_PINS}
    assert set(roles) == {"stage2.glass_router", "stage2.rf_router"}
    prefix = f"models/pr38/{P.STAGE2_RUN_ID}/primary/"
    glass = roles["stage2.glass_router"].path[len(prefix):]
    rf = roles["stage2.rf_router"].path[len(prefix):]
    assert glass == f"glass_router/glass_router_pr38_primary_{P.STAGE2_RUN_ID}.joblib"
    # the immutable version, never the run's mutable alias (same bytes, other name)
    assert rf.startswith(f"rf_router/rf_router_pr38_primary_{P.STAGE2_RUN_ID}_2026")
    assert f"rf_router/rf_router_pr38_primary_{P.STAGE2_RUN_ID}.joblib" in man["files"]
    assert rf != f"rf_router/rf_router_pr38_primary_{P.STAGE2_RUN_ID}.joblib"
    # what the PR 38 acceptance itself names (acceptance.load_candidate's patterns)
    from stage2_crossfit import acceptance as A
    stem = f"pr38_primary_{P.STAGE2_RUN_ID}"
    assert A._one(man["files"], rf"glass_router/glass_router_{stem}\.joblib") == glass
    assert A._one(man["files"], rf"rf_router/rf_router_{stem}_\d{{8}}_\d{{6}}_\d{{6}}\.joblib") == rf


def test_a_changed_manifest_or_a_pin_it_does_not_list_is_refused(tmp_path):
    man = tmp_path / P.STAGE2_MANIFEST
    man.parent.mkdir(parents=True)
    man.write_bytes((BANK_ROOT / P.STAGE2_MANIFEST).read_bytes() + b" ")
    assert "is not the pinned" in P.stage2_manifest_problems(tmp_path)[0]
    man.write_bytes((BANK_ROOT / P.STAGE2_MANIFEST).read_bytes())
    wrong = PinnedFile("stage2.glass_router", P.STAGE2_PINS[0].path, "0" * 64, P.STAGE2_PINS[0].bytes)
    problems = P.stage2_manifest_problems(tmp_path, pins=(wrong, P.STAGE2_PINS[1]))
    assert len(problems) == 1 and "is not listed with this size and SHA-256" in problems[0]
    assert P.stage2_manifest_problems(tmp_path / "nowhere")[0].endswith("does not exist")


# ----------------------------------------------------------------------
# one file
# ----------------------------------------------------------------------
def test_a_pinned_file_is_verified_by_size_and_sha256(tmp_path):
    pin = _file(tmp_path, "models/lr/lr_calibrated_20261115_000000.joblib", b"abc")
    assert P.verify_pin(pin, tmp_path) == tmp_path / pin.path
    with pytest.raises(PinError, match="does not exist"):
        P.verify_pin(PinnedFile("stage1.lr", "models/lr/missing.joblib", pin.sha256, 3), tmp_path)
    (tmp_path / pin.path).write_bytes(b"abcd")
    with pytest.raises(PinError, match="4 bytes, pinned 3"):
        P.verify_pin(pin, tmp_path)
    (tmp_path / pin.path).write_bytes(b"abd")                    # same size, other content
    with pytest.raises(PinError, match="changed since it was pinned"):
        P.verify_pin(pin, tmp_path)
    for bad in ("/abs/x.joblib", "../x.joblib", "models\\lr\\x.joblib"):
        with pytest.raises(PinError, match="not a plain path"):
            P.verify_pin(PinnedFile("stage1.lr", bad, pin.sha256, 3), tmp_path)


# ----------------------------------------------------------------------
# Stage 2, on synthetic routers in the accepted run's layout
# ----------------------------------------------------------------------
def _synthetic_run(tmp_path, glass_population="oof_remainder"):
    w = build_world(n=1500, n_test=350)
    files = {"glass_router/glass_router_pr38_primary_c3_x.joblib":
             make_router(w, pass2_population=glass_population),
             "rf_router/rf_router_pr38_primary_c3_x_20260101_000000_000000.joblib":
             make_router(w, layout="extra", band_blend=True, pass2_population=RF_REMAINDER)}
    root = "models/pr38/c3_x/primary"
    listed, pins = {}, []
    for (rel, art), role in zip(files.items(), ("stage2.glass_router", "stage2.rf_router")):
        p = tmp_path / root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(art, p)
        data = p.read_bytes()
        listed[rel] = {"bytes": len(data), "sha256": _sha(data)}
        pins.append(PinnedFile(role, f"{root}/{rel}", _sha(data), len(data)))
    man = tmp_path / "research_logs/pr38/c3_x/primary_MANIFEST.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"files": listed}), encoding="utf-8")
    kw = dict(root=tmp_path, pins=tuple(pins), run_id="c3_x",
              manifest="research_logs/pr38/c3_x/primary_MANIFEST.json",
              manifest_sha256=_sha(man.read_bytes()), candidate_root=root,
              require_acceptance=False)
    return w, kw


def test_load_pinned_stage2_verifies_then_checks_both_routers_against_the_stage4_contract(
        tmp_path, monkeypatch):
    w, kw = _synthetic_run(tmp_path)
    import shared.stage4.contract as contract
    real = contract.build_stage4_inputs
    monkeypatch.setattr(contract, "build_stage4_inputs",
                        lambda *a, **k: real(*a, **{**k, "extra_fingerprints": ROUTER_FP}))
    import shared.stage4 as stage4
    monkeypatch.setattr(stage4, "build_stage4_inputs", contract.build_stage4_inputs)
    s2 = P.load_pinned_stage2(w["GS"], **kw)
    assert s2.glass_path == tmp_path / kw["pins"][0].path
    assert s2.glass_bundle["train_outputs_oof"] is True and "extra" in s2.rf_artifact
    assert s2.stage4_contract["glass"]["pass2_population"] == {"cross_fit": "oof_remainder",
                                                               "full_train": "oof_remainder"}
    assert s2.stage4_contract["rf_router"]["pass2_population"] == {"cross_fit": "oof_remainder",
                                                                   "full_train": "oof_remainder"}
    assert "NOT re-verified" in s2.acceptance and "Stage 4 contract [glass]" in s2.describe()
    # no split given: the files are still verified and loaded, nothing else runs
    assert P.load_pinned_stage2(None, **kw).stage4_contract == {}


def test_load_pinned_stage2_refuses_a_changed_router_or_pins_outside_the_manifest(tmp_path):
    w, kw = _synthetic_run(tmp_path)
    rf = tmp_path / kw["pins"][1].path
    rf.write_bytes(rf.read_bytes()[:-1] + b"\x00")
    with pytest.raises(PinError, match="stage2.rf_router: .* not the pinned"):
        P.load_pinned_stage2(None, **kw)
    with pytest.raises(PinError, match="not the accepted run's"):
        P.load_pinned_stage2(None, **{**kw, "manifest_sha256": "0" * 64})
    with pytest.raises(PinError, match="missing \\['stage2.rf_router'\\]"):
        P.load_pinned_stage2(None, **{**kw, "pins": kw["pins"][:1]})


def test_the_notebooks_load_the_real_pins_with_the_acceptance_re_verified():
    """The notebooks pass only GLOBAL_SPLIT: the real pins, manifest and acceptance."""
    import inspect
    sig = inspect.signature(P.load_pinned_stage2)
    assert sig.parameters["pins"].default is P.STAGE2_PINS
    assert sig.parameters["require_acceptance"].default is True
    assert sig.parameters["root"].default == BANK_ROOT


# ----------------------------------------------------------------------
# a run's pins file
# ----------------------------------------------------------------------
def test_a_pins_file_is_written_once_and_read_back_verified(tmp_path, monkeypatch):
    lr = _file(tmp_path, "models/lr/lr_calibrated_20261115_000000.joblib", b"lr")
    ebm = PinnedFile("stage3.ebm", "models/ebm/ebm_stage3_x.joblib", *(lambda d: (_sha(d), len(d)))(b"e"))
    (tmp_path / ebm.path).parent.mkdir(parents=True)
    (tmp_path / ebm.path).write_bytes(b"e")
    pins = tmp_path / "research_logs/pr39/c4_20261115_000000/upstream_pins.json"
    P.write_pins(pins, "c4_20261115_000000", {"stage1.lr": lr, "stage3.ebm": ebm}, root=tmp_path)
    with pytest.raises(FileExistsError):
        P.write_pins(pins, "c4_20261115_000000", {"stage1.lr": lr}, root=tmp_path)
    assert P.pinned_path("stage1.lr", pins_file=pins, root=tmp_path) == tmp_path / lr.path
    monkeypatch.setenv(P.PINS_ENV, str(pins))
    assert P.pinned_path("stage3.ebm", root=tmp_path) == tmp_path / ebm.path
    rec = P.read_pins()
    assert rec["format"] == P.PINS_FORMAT and rec["files"]["stage1.lr"]["producer"] == P.ROLES["stage1.lr"]
    with pytest.raises(PinError, match="pinned no stage1.mlp"):
        P.pinned_path("stage1.mlp", root=tmp_path)
    with pytest.raises(PinError, match="unknown role"):
        P.pinned_path("stage1.newest", root=tmp_path)
    (tmp_path / lr.path).write_bytes(b"LR")                       # changed after pinning
    with pytest.raises(PinError, match="changed since it was pinned"):
        P.pinned_path("stage1.lr", root=tmp_path)


def test_a_missing_or_edited_pins_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(P.PINS_ENV, raising=False)
    with pytest.raises(PinError, match="is not set"):
        P.read_pins()
    monkeypatch.setenv(P.PINS_ENV, "")
    with pytest.raises(PinError, match="no artifact is selected as the newest file"):
        P.read_pins()
    with pytest.raises(PinError, match="does not exist"):
        P.read_pins(tmp_path / "nowhere.json")
    lr = _file(tmp_path, "models/lr/lr_calibrated_20261115_000000.joblib", b"lr")
    pins = tmp_path / "pins.json"
    P.write_pins(pins, "c4_20261115_000000", {"stage1.lr": lr}, root=tmp_path)
    pins.write_text(pins.read_text().replace("stage1.lr", "stage1.lr "), encoding="utf-8")
    with pytest.raises(PinError, match="does not match its pins.json.sha256 sidecar"):
        P.read_pins(pins)
    (tmp_path / "pins.json.sha256").unlink()
    with pytest.raises(PinError, match="has no pins.json.sha256 sidecar"):
        P.read_pins(pins)


def test_write_pins_refuses_unknown_roles_bad_run_ids_and_unverifiable_files(tmp_path):
    lr = _file(tmp_path, "models/lr/lr_calibrated_20261115_000000.joblib", b"lr")
    with pytest.raises(PinError, match="not c<commit>_"):
        P.write_pins(tmp_path / "a.json", "run1", {"stage1.lr": lr}, root=tmp_path)
    with pytest.raises(PinError, match="unknown roles"):
        P.write_pins(tmp_path / "a.json", "c4_20261115_000000", {"stage9.x": lr}, root=tmp_path)
    with pytest.raises(PinError, match="holds a pin for"):
        P.write_pins(tmp_path / "a.json", "c4_20261115_000000", {"stage3.ebm": lr}, root=tmp_path)
    stale = PinnedFile("stage1.lr", lr.path, "0" * 64, lr.bytes)
    with pytest.raises(PinError, match="changed since it was pinned"):
        P.write_pins(tmp_path / "a.json", "c4_20261115_000000", {"stage1.lr": stale}, root=tmp_path)
    assert not (tmp_path / "a.json").exists()
