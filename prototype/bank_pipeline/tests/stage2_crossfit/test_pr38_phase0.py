"""
PR 38 Step 2A: Phase 0 (``stage2_crossfit.phase0``) and the command line (``python -m stage2_crossfit``).

1. The snapshot check verifies every archived file against SHA256SUMS and the
   SUMS file against the baseline's record; live files that differ are
   reported, not failed (since e3e27f6 the archive is the reference).
2. ``$BANK_ARTIFACT_MANIFEST`` must name exactly the four arbiters, each
   attested, the baseline's two included.
3. The inventory cites the baseline and checks each source against the archive.
4. The output root is created once, never inside a newest-file folder, and
   only after every check passed.
5. The CLI refuses a real-data fit without a passed Phase 0 record and never
   replaces a record.

A fake archive and bank root in tmp_path; the real baseline package for --verify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import stage2_crossfit.__main__ as cli
import stage2_crossfit.phase0 as P


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """bank/ (live), archive/ (SHA256SUMS + bank_pipeline/...), a matching baseline manifest."""
    bank, archive = tmp_path / "bank", tmp_path / "archive"
    files = {
        "models/meta_ebm/meta_ebm_1.joblib": b"glass arbiter",
        "blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_1.joblib": b"xgb arbiter",
        "blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/meta_ebm_2.joblib": b"x",
        "blackbox_pipeline/models/stage4_crossover/meta_xgb_on_glass/meta_xgb_stage4_2.joblib": b"y",
        "models/rf_router/rf_router_two_pass.joblib": b"october router",
        "models/glass_router/glass_router_1.joblib": b"october glass",
    }
    lines = []
    for rel, data in files.items():
        for root in (bank / rel, archive / "bank_pipeline" / rel):
            root.parent.mkdir(parents=True, exist_ok=True)
            root.write_bytes(data)
        lines.append(f"{_sha(data)}  bank_pipeline/{rel}")
    (bank / "models/rf_router/rf_router_two_pass.joblib").write_bytes(b"regenerated alias")
    sums = archive / "SHA256SUMS.txt"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "snapshot": {"entries": len(files), "sha256": _sha(sums.read_bytes())},
        "sources": [
            {"role": "stage4_arbiter.meta_ebm", "path": "models/meta_ebm/meta_ebm_1.joblib",
             "sha256": _sha(files["models/meta_ebm/meta_ebm_1.joblib"]), "snapshot_attested": True},
            {"role": "stage4_arbiter.meta_ebm_on_blackbox",
             "path": "blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/meta_ebm_2.joblib",
             "sha256": _sha(b"x"), "snapshot_attested": True},
            {"role": "stage2.rf_router", "path": "models/rf_router/rf_router_two_pass.joblib",
             "sha256": _sha(b"october router"), "snapshot_attested": True},
        ],
        "identities": {"router_split_fingerprint": "fc3e73a5ef91aea4"}, "routers": {"glass": {}},
    }
    baseline = bank / "baseline"
    baseline.mkdir()
    (baseline / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(P, "BANK_ROOT", bank)
    monkeypatch.setattr(P, "BASELINE_DIR", baseline)
    arbiters = {
        "meta_ebm": "models/meta_ebm/meta_ebm_1.joblib",
        "meta_xgb": "blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_1.joblib",
        "meta_ebm_on_blackbox": manifest["sources"][1]["path"],
        "meta_xgb_on_glass":
            "blackbox_pipeline/models/stage4_crossover/meta_xgb_on_glass/meta_xgb_stage4_2.joblib",
    }
    art = tmp_path / "artifact_manifest.json"
    art.write_text(json.dumps(arbiters), encoding="utf-8")
    return {"bank": bank, "archive": archive, "sums": sums, "art": art, "arbiters": arbiters}


# ======================================================================
# 1-3. snapshot, manifest, inventory
# ======================================================================
def test_an_intact_archive_passes_and_a_regenerated_live_alias_is_only_reported(world):
    out = P.verify_snapshot(world["sums"])
    assert out["passed"]
    assert out["live_counterparts"]["different"] == [
        "bank_pipeline/models/rf_router/rf_router_two_pass.joblib"]


def test_a_changed_archived_file_or_another_sums_file_fails(world):
    (world["archive"] / "bank_pipeline/models/meta_ebm/meta_ebm_1.joblib").write_bytes(b"tampered")
    out = P.verify_snapshot(world["sums"])
    assert not out["passed"] and out["checks"]["archived_files"]["problems"]
    world["sums"].write_text(world["sums"].read_text() + "\n", encoding="utf-8")
    assert not P.verify_snapshot(world["sums"])["checks"]["sums_file_sha256"]["passed"]


def test_the_artifact_manifest_must_pin_the_four_attested_arbiters(world):
    assert P.check_artifact_manifest(world["art"], world["sums"])["passed"]
    three = dict(world["arbiters"])
    three.pop("meta_xgb")
    world["art"].write_text(json.dumps(three))
    assert not P.check_artifact_manifest(world["art"], world["sums"])["passed"]
    other = dict(world["arbiters"], meta_ebm="models/glass_router/glass_router_1.joblib")
    world["art"].write_text(json.dumps(other))
    out = P.check_artifact_manifest(world["art"], world["sums"])
    assert not out["passed"] and any("baseline records" in p for p in out["problems"])
    (world["bank"] / world["arbiters"]["meta_xgb"]).write_bytes(b"newer arbiter")
    world["art"].write_text(json.dumps(world["arbiters"]))
    out = P.check_artifact_manifest(world["art"], world["sums"])
    assert not out["passed"] and any("differs" in p for p in out["problems"])


def test_the_inventory_checks_each_baseline_source_against_the_archive(world):
    out = P.inventory(world["sums"])
    assert out["passed"]
    rf = next(r for r in out["sources"] if r["role"] == "stage2.rf_router")
    assert rf["archive_matches"] and rf["live_present"] and not rf["live_matches"]
    (world["archive"] / "bank_pipeline/models/rf_router/rf_router_two_pass.joblib").unlink()
    assert not P.inventory(world["sums"])["passed"]


# ======================================================================
# 4. the output root
# ======================================================================
def test_the_output_root_is_created_once_and_never_in_a_newest_file_folder(world):
    root = world["bank"] / "models" / "pr38"
    out = P.prepare_output_root("r1", root=root)
    assert (Path(out["path"]) / "PR38_OUTPUT_ROOT.json").is_file()
    with pytest.raises(FileExistsError):
        P.prepare_output_root("r1", root=root)
    with pytest.raises(ValueError, match="newest-file"):
        P.prepare_output_root("r2", root=world["bank"] / "models" / "rf_router")
    with pytest.raises(ValueError, match="plain name"):
        P.prepare_output_root("../escape", root=root)


def test_no_output_root_unless_every_check_passed(world):
    root = world["bank"] / "models" / "pr38"
    rec = P.run_phase0(world["sums"], world["art"], "r9", root=root, baseline_checks=False)
    assert rec["passed"] is False and rec["output_root"] is None and not (root / "r9").exists()


def test_the_baseline_package_verifies_with_the_unchanged_cli():
    out = P._run(["research_logs/baselines/stage2_baseline.py", "--verify"],
                 {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert out["passed"], out["output_tail"]


# ======================================================================
# 5. the command line
# ======================================================================
def test_real_data_commands_refuse_without_a_passed_phase0(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path)
    for cmd in ("g13", "preflight"):
        with pytest.raises(SystemExit, match="Phase 0"):
            cli.main([cmd, "--run-id", "r1"])
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "phase0.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(SystemExit, match="did not pass"):
        cli.main(["g13", "--run-id", "r1"])


def test_the_configs_record_is_written_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path)
    assert cli.main(["configs", "--run-id", "r1"]) == 0
    rec = json.loads((tmp_path / "r1" / "configs.json").read_text(encoding="utf-8"))
    assert rec["primary"]["role"] == "primary" and rec["s1"]["glass_config"][
        "pass2_population"] == "full"
    with pytest.raises(FileExistsError):
        cli.main(["configs", "--run-id", "r1"])
