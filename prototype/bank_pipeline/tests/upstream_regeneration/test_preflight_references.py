"""
PR 39 Step 4A: which historical references the run checks (execution contract
section 7.5), so a reference is never confused for another.

* The baseline step is PR 38's own: ``stage2_baseline.py --verify`` (the frozen
  package) and ``stage2_baseline_archive.py --check`` (a rebuild from the
  preserved archive). Never ``stage2_baseline.py --check``, which rebuilds from
  the live ``models/`` files: since 2fc030e / e3e27f6 those hold regenerated
  latest aliases (``models/rf_router/rf_router_two_pass.joblib`` among them),
  so it refuses by design, and the snapshot records such files as live
  counterparts that differ, for information only.
* No Step 4A module reads ``$BANK_ARTIFACT_MANIFEST``: it names the four Stage 4
  arbiters for the real-data audit and PR 38's phase 0, and Step 4A neither
  audits nor writes an arbiter.
"""

from __future__ import annotations

from pathlib import Path

from stage2_crossfit import acceptance, phase0
from upstream_regeneration import checks as C

BANK = Path(__file__).resolve().parents[2]


def test_the_baseline_step_runs_verify_and_the_archive_check_never_the_live_check(monkeypatch):
    commands = []

    def run(args, env):
        commands.append(list(args))
        return {"command": " ".join(args), "exit_code": 0, "passed": True, "output_tail": ["ok"]}

    monkeypatch.setattr(phase0, "_run", run)
    monkeypatch.setattr(phase0, "verify_snapshot", lambda sums: {
        "passed": True, "checks": {}, "live_counterparts": {
            "same": 36, "missing": [],
            "different": ["models/rf_router/rf_router_two_pass.joblib"]}})
    monkeypatch.setattr(C, "report_check", lambda: {"passed": True, "output_tail": []})

    class Candidate:
        manifest_sha256 = "m" * 64

    monkeypatch.setattr(acceptance, "require_accepted", lambda run_id: Candidate())
    gate = C.Gate("A4.1")
    out = C.preservation(gate, "SHA256SUMS.txt")
    assert gate.passed, gate.failed()
    assert commands == [["research_logs/baselines/stage2_baseline.py", "--verify"],
                        ["research_logs/baselines/stage2_baseline_archive.py", "--check", "--sums",
                         "SHA256SUMS.txt"]]
    # a regenerated latest alias is a recorded live difference, not a failure
    assert C.live_counterparts(out["snapshot"])["different"] == [
        "models/rf_router/rf_router_two_pass.joblib"]


def test_no_step4a_module_reads_the_arbiter_manifest_variable():
    sources = [*sorted((BANK / "upstream_regeneration").glob("*.py")),
               BANK / "shared" / "pinned_artifacts.py", BANK / "shared" / "notebook_switches.py"]
    readers = [p.name for p in sources if "BANK_ARTIFACT_MANIFEST" in p.read_text(encoding="utf-8")]
    assert len(sources) >= 9 and readers == []


def _pr38_live_counterparts():
    """The live counterparts PR 38's accepted run recorded (its committed phase0.json)."""
    import json
    rec = json.loads((BANK / "research_logs/pr38/c3_20261112_153333/phase0.json").read_text(
        encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            if "live_counterparts" in o:
                return o["live_counterparts"]
            return next((r for r in map(find, o.values()) if r), None)
        if isinstance(o, list):
            return next((r for r in map(find, o) if r), None)
        return None
    return C.live_counterparts({"live_counterparts": find(rec)})


def test_only_step4a_mutable_aliases_may_change_a_live_counterpart():
    before = _pr38_live_counterparts()
    assert "bank_pipeline/models/rf_router/rf_router_two_pass.joblib" in before["different"]
    assert "bank_pipeline/models/mlp/stage1_mlp.joblib" in before["different"]
    assert C.live_counterpart_problems(before, before) == []
    # a correct run: the D7 CSVs (if attested and still the snapshot's) are rewritten
    d7 = "bank_pipeline/models/mlp/stage1_mlp_threshold_sweep.csv"
    run = {**before, "same": before["same"] - 1, "different": sorted(before["different"] + [d7])}
    assert C.live_counterpart_problems(before, run) == []
    # a historical immutable version or a legacy alias that changes status is named
    lr = "bank_pipeline/models/lr/lr_calibrated_20261023_003319.joblib"
    bad = {**before, "same": before["same"] - 1, "different": sorted(before["different"] + [lr])}
    assert C.live_counterpart_problems(before, bad) == [
        "models/lr/lr_calibrated_20261023_003319.joblib: now different"]
    gone = {**before, "missing": ["bank_pipeline/models/rf_router/rf_router_two_pass.joblib"],
            "different": [x for x in before["different"] if "rf_router_two_pass.joblib" not in x]}
    assert C.live_counterpart_problems(before, gone) == [
        "models/rf_router/rf_router_two_pass.joblib: no longer different",
        "models/rf_router/rf_router_two_pass.joblib: now missing"]
    assert C.live_counterpart_problems(before, {"same": None, "different": [], "missing": []})[-1] \
        == "no live counterparts recorded after the run"
