"""
PR 39 Step 5A: the four steps chained end to end (``stage4_regeneration.runner``)
on a temporary bank_pipeline/. The notebook executions are replaced by fakes
that write arbiter files. The data-dependent checks and the real-data audit are
replaced by stubs.

This pins the bookkeeping the run depends on. Each step requires the earlier
ones. The GLASS execution reads the accepted manifest and writes the pins file
the black-box execution reads, and that file holds the new GLASS Arbiter. The
black-box execution writes the arbiter manifest. A stray write, a code change
or an arbiter that read an unpinned upstream file stops the run. The run
record names the four arbiters with their size and SHA-256.
"""

from __future__ import annotations

import json

import pytest

from shared import pinned_artifacts as P
from stage4_regeneration import runner as R
from stage4_regeneration import stage4_copy as S
from upstream_regeneration import checks as C

RUN = "c5_20261116_120000"
VID = {"glass_arbiter": "20261116_121000_000001", "meta_xgb": "20261116_122000_000001",
       "glass_arbiter_on_blackbox": "20261116_123000_000001",
       "meta_xgb_on_glass": "20261116_124000_000001"}


def _rel(key):
    stem = "glass_arbiter" if key.startswith("glass") else "meta_xgb_stage4"
    folder = {"glass_arbiter": "models/glass_arbiter",
              "meta_xgb": "blackbox_pipeline/models/meta_xgb",
              "glass_arbiter_on_blackbox": f"{S._XO}/glass_arbiter_on_blackbox",
              "meta_xgb_on_glass": f"{S._XO}/meta_xgb_on_glass"}[key]
    return f"{folder}/{stem}_{VID[key]}.joblib"


@pytest.fixture
def bank(tmp_path, monkeypatch):
    root = tmp_path / "bank_pipeline"
    (root / "shared").mkdir(parents=True)
    (root / "shared" / "code.py").write_text("x = 1\n")
    pins = {}
    for role in R.UPSTREAM_ROLES:
        rel = f"models/{role.replace('.', '/')}_v.joblib"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(role.encode())
        pins[role] = P.PinnedFile(role, rel, P.sha256_bytes(role.encode()), len(role))
    monkeypatch.setattr(R.Rec, "BANK_ROOT", root)
    monkeypatch.setattr(R.Rec, "RECORD_ROOT", root / "research_logs" / "pr39")
    monkeypatch.setattr(R, "BANK_ROOT", root)
    P.write_pins(R.manifest_path(), R.UPSTREAM_RUN, pins, root=root)
    monkeypatch.setattr(R.Acc, "check", lambda run_id: {
        "run_id": run_id, "accepted": True, "manifest_sha256": "m" * 64,
        "acceptance_sha256": "a" * 64, "roles": {r: p.record() for r, p in pins.items()}})
    monkeypatch.setattr(R.Rec, "git_status", lambda root=None: [" M shared/code.py"])
    monkeypatch.setattr(R.Rec, "environment", lambda: {"python": "test"})

    def preservation(gate, sums):
        gate.check("7.5 stub", lambda: True)
        return {"snapshot": {"live_counterparts": {"same": 30, "different": [], "missing": []}}}

    monkeypatch.setattr(C, "preservation", preservation)
    monkeypatch.setattr(C, "global_split", lambda: {})
    monkeypatch.setattr(C, "frozen_population", lambda: {})
    monkeypatch.setattr(C, "identities", lambda split, pop: {"feature_contracts": {}})
    monkeypatch.setattr(S, "stage4_copy", lambda nb, root=None: (
        {"cells": []}, {"copy_sha256": "c" * 64, "stage4_cells": [{"cell": "CELL 19"}],
                        "not_run": ["CELL 20"]}))
    monkeypatch.setattr(S, "roles_read", lambda nb, root=None: (
        ["stage1.lr", "stage2.glass_router", "stage3.ebm"] if nb == "glass"
        else sorted(set(R.UPSTREAM_ROLES) | {S.GLASS_ARBITER_ROLE})))
    seen = {"stray": None}

    def fake_execute(nb, notebook, out_dir, run_id, *, env=None, kernel_name=None, root=None):
        seen[notebook] = dict(env or {})
        keys = list(S.OUTPUTS[notebook])
        if notebook == "blackbox":       # Cell 35 reads the native GLASS Arbiter by pin
            seen["glass_arbiter_read"] = P.pinned_path(S.GLASS_ARBITER_ROLE,
                                                       pins_file=env[P.PINS_ENV], root=root)
        for key in keys:
            p = root / _rel(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(key.encode())
            if key.startswith("meta_xgb"):
                p.with_suffix(".json").write_text("{}")
        if seen["stray"] == notebook:
            (root / "models/stage1/lr_extra.joblib").write_bytes(b"retrained")
        return {"completed": True, "error": None, "wall_seconds": 1.0,
                "kernel": {"name": "python3"}, "input_copy": {"sha256": "i" * 64},
                "executed_copy": {"sha256": "e" * 64}}

    monkeypatch.setattr(R.V, "execute", fake_execute)

    def arbiter(path):
        key = next(k for k in VID if VID[k] in str(path))
        arm = S.ROOT_UPSTREAM[key]
        stages = {s: {"artifact_path": str(root / pins[r].path)}
                  for s, r in R.UPSTREAM_ROLE_OF[arm].items()}
        if seen.get("unpinned") == key:
            stages["stage3"]["artifact_path"] = str(root / "models/stage1/lr_v.joblib")
        return {"schema": "test", "stage4_inputs": {"upstream": {
            "arm": arm, "stages": stages, "combination": f"{arm} upstream -> {key}",
            "crossover": key in ("glass_arbiter_on_blackbox", "meta_xgb_on_glass")}}}

    monkeypatch.setattr(R, "_load", arbiter)
    monkeypatch.setattr(R, "_audit", lambda mfile, sums: {
        "passed": True, "summary": {"tests": 14, "failures": 0, "errors": 0, "skipped": 0},
        "exit_code": 0, "output_tail": [], "command": "pytest stub",
        "junit": b'<testsuite tests="14" failures="0" errors="0" skipped="0"/>\n'})
    return root, pins, seen, tmp_path / "outside"


def test_the_four_steps_chain_and_the_run_record_names_the_four_arbiters(bank):
    root, pins, seen, out = bank
    assert R.preflight(RUN, "sums.txt")["passed"]
    glass = R.execute(RUN, "glass", out)
    assert seen["glass"] == {P.PINS_ENV: str(R.manifest_path())}        # the accepted manifest
    assert set(glass["produced"]) == {"glass_arbiter"}
    run_pins = P.read_pins(R.run_dir(RUN) / R.PINS_FILE)
    assert sorted(run_pins["files"]) == sorted(set(R.UPSTREAM_ROLES) | {S.GLASS_ARBITER_ROLE})
    assert all(run_pins["files"][r]["sha256"] == pins[r].sha256 for r in R.UPSTREAM_ROLES)
    black = R.execute(RUN, "blackbox", out)
    assert seen["blackbox"] == {P.PINS_ENV: str(R.run_dir(RUN) / R.PINS_FILE)}
    assert seen["glass_arbiter_read"] == root / _rel("glass_arbiter")
    manifest = json.loads((R.run_dir(RUN) / f"{R.ARBITER_MANIFEST}.json").read_text())
    assert manifest == {k: _rel(k) for k in S.ROOT_KEYS} == black["arbiter_manifest"]["entries"]
    rec = R.finalize(RUN, "sums.txt")
    assert rec["passed"] and "Step 5B" in rec["status"] and "accepted" not in rec
    assert sorted(rec["arbiters"]) == sorted(S.ROOT_KEYS)
    for key, a in rec["arbiters"].items():
        assert a["path"] == _rel(key) and a["sha256"] == P.sha256_bytes(key.encode())
        assert sorted(a["upstream_resolution"]) == ["stage1", "stage2", "stage3"]
    assert (R.run_dir(RUN) / R.AUDIT_FILE).is_file()
    assert rec["audit"]["sha256"] == P.sha256_bytes((R.run_dir(RUN) / R.AUDIT_FILE).read_bytes())
    assert rec["inventory_net"]["passed"] and len(rec["inventory_net"]["sidecars"]) == 2
    for name in ("preflight", "execution_glass", "execution_blackbox", "run_record"):
        assert R.require_passed(RUN, name)["run_id"] == RUN


def test_each_step_requires_the_earlier_ones_and_is_written_once(bank):
    _, _, _, out = bank
    with pytest.raises(R.Rec.RecordError):
        R.execute(RUN, "glass", out)                          # no preflight
    R.preflight(RUN, "sums.txt")
    with pytest.raises(R.Rec.RecordError):
        R.execute(RUN, "blackbox", out)                       # GLASS first
    with pytest.raises(R.Rec.RecordError):
        R.finalize(RUN, "sums.txt")
    R.execute(RUN, "glass", out)
    with pytest.raises(R.Rec.RecordError, match="already executed"):
        R.execute(RUN, "glass", out)
    with pytest.raises(R.Rec.RecordError, match="already has a preflight"):
        R.preflight(RUN, "sums.txt")


def test_a_stray_write_stops_the_execution_and_writes_no_pins(bank):
    root, _, seen, out = bank
    seen["stray"] = "glass"
    R.preflight(RUN, "sums.txt")
    with pytest.raises(R.StepFailed, match="execution_glass did not pass"):
        R.execute(RUN, "glass", out)
    assert not (R.run_dir(RUN) / R.PINS_FILE).exists()
    assert not (R.run_dir(RUN) / "execution_glass.json").exists()
    assert list((R.run_dir(RUN) / "failed").glob("execution_glass_*.json"))


def test_a_code_change_between_steps_stops_the_run(bank):
    root, _, _, out = bank
    R.preflight(RUN, "sums.txt")
    (root / "shared" / "code.py").write_text("x = 2\n")
    with pytest.raises(R.Rec.RecordError, match="code changed"):
        R.execute(RUN, "glass", out)


def test_an_arbiter_that_read_an_unpinned_upstream_file_fails_a51(bank):
    root, _, seen, out = bank
    (root / "models/stage1/lr_v.joblib").parent.mkdir(parents=True, exist_ok=True)
    seen["unpinned"] = "meta_xgb_on_glass"
    R.preflight(RUN, "sums.txt")
    R.execute(RUN, "glass", out)
    R.execute(RUN, "blackbox", out)
    with pytest.raises(R.StepFailed, match="meta_xgb_on_glass: each Stage 1-3 file"):
        R.finalize(RUN, "sums.txt")
    assert not (R.run_dir(RUN) / "run_record.json").exists()
    assert not (R.run_dir(RUN) / R.AUDIT_FILE).exists()
