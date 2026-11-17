"""
PR 39 Step 4A: the five steps chained end to end (upstream_regeneration.runner)
on a temporary bank_pipeline/, with the notebook executions replaced by fakes
that write files and the data-dependent checks replaced by stubs.

What this pins is the bookkeeping a multi-hour run depends on: each step
requires the earlier ones; the GLASS execution writes the pins file the
black-box execution reads; the inventory is chained from step to step; a stray
write, a code change or a repeated step stops the run; the run record names
every artifact with its size and SHA-256 and is not an acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from shared import pinned_artifacts as P
from upstream_regeneration import checks as C
from upstream_regeneration import records as Rec
from upstream_regeneration import runner as Run
from upstream_regeneration import vehicle as V

RUN = "c4_20261115_120000"


def _passing(gate_name):
    def stub(gate, *a, **k):
        gate.check(f"{gate_name} stub", lambda: True)
        return {"stub": gate_name}
    return stub


@pytest.fixture
def bank(tmp_path, monkeypatch):
    root = tmp_path / "bank_pipeline"
    for rel in ("models/pr38/c3_20261112_153333/primary/glass_router",
                "models/pr38/c3_20261112_153333/primary/rf_router", "models/lr", "models/ebm",
                "models/mlp", "blackbox_pipeline/models/xgb", "shared"):
        (root / rel).mkdir(parents=True)
    (root / "shared" / "code.py").write_text("x = 1\n")
    (root / "models/lr/lr_calibrated_20261023_101010.joblib").write_bytes(b"old lr")
    (root / "models/mlp/stage1_mlp_threshold_sweep.csv").write_text("old\n")
    stage2 = []
    for pin, payload in zip(P.STAGE2_PINS, (b"glass bundle", b"rf router")):
        (root / pin.path).write_bytes(payload)
        stage2.append(P.PinnedFile(pin.role, pin.path, P.sha256_bytes(payload), len(payload)))
    monkeypatch.setattr(P, "STAGE2_PINS", tuple(stage2))
    monkeypatch.setattr(Rec, "BANK_ROOT", root)
    monkeypatch.setattr(Rec, "RECORD_ROOT", root / "research_logs" / "pr39")
    monkeypatch.setattr(Rec, "git_status", lambda root=None: [" M shared/code.py"])
    monkeypatch.setattr(Rec, "environment", lambda: {"python": "test"})

    def preservation(gate, sums):
        gate.check("7.5 stub", lambda: True)
        return {"report_check": {"passed": True}, "snapshot": {"live_counterparts": {
            "same": 30, "different": ["a"], "missing": []}},
            "baseline": {"verify": {"output_tail": ["code identity note"]}}}

    monkeypatch.setattr(C, "preservation", preservation)
    monkeypatch.setattr(C, "global_split", lambda: {"X_train": None, "y_train": None, "X_test": None})
    monkeypatch.setattr(C, "frozen_population", lambda: {"population": "frozen"})
    monkeypatch.setattr(C, "stage2_reuse", _passing("A4.2"))
    monkeypatch.setattr(C, "stage1_checks", _passing("A4.3"))
    monkeypatch.setattr(C, "stage3_checks", _passing("A4.4"))
    monkeypatch.setattr(C, "stage4_input_checks", _passing("A4.5"))
    monkeypatch.setattr(C, "fresh_reload", _passing("7.4"))
    monkeypatch.setattr(C, "identities", lambda split, pop: {"identities": "stub"})
    import stage2_crossfit.runner as R
    monkeypatch.setattr(R, "check_population", lambda ts, protocol: {"passed": True, "checks": {}})
    monkeypatch.setattr(R, "make_train_side", lambda *a: None)
    monkeypatch.setattr(V, "truncated_copy", lambda nb, root=None: (
        {"cells": []}, {"truncated_copy_sha256": "t" * 64, "removed_code_cells": ["CELL 19"]}))
    seen = {}

    def fake_execute(nb, notebook, out_dir, run_id, *, env=None, kernel_name=None, root=None):
        seen[notebook] = dict(env or {})
        if notebook == "glass":
            joblib.dump({"lr": 1}, root / "models/lr/lr_calibrated_20261115_121000.joblib")
            joblib.dump({"model_name": "logistic_regression_baseline"},
                        root / "models/lr/lr_baseline_20261115_120900.joblib")     # GLASS Cell 6
            joblib.dump({"ebm": 1}, root / "models/ebm/ebm_stage3_20261115_122000_000001.joblib")
            (root / "models/ebm/ebm_stage3_20261115_122000_000001.json").write_text("{}")
            (root / "models/ebm/ebm_stage3_latest.joblib").write_bytes(b"alias")
        else:
            # the black-box notebook reads the GLASS LR through the pins file
            seen["lr_read"] = P.pinned_path("stage1.lr", pins_file=env[P.PINS_ENV], root=root)
            vid = "20261115_130000_000001"
            joblib.dump({"mlp": 1}, root / f"models/mlp/stage1_mlp_{vid}.joblib")
            (root / "models/mlp/stage1_mlp_threshold_sweep.csv").write_text("new\n")
            joblib.dump({"xgb": 1}, root / f"blackbox_pipeline/models/xgb/xgb_stage3_{vid}.joblib")
        return {"completed": True, "error": None, "wall_seconds": 1.0,
                "kernel": {"name": "python3"}, "input_copy": {"sha256": "i" * 64},
                "executed_copy": {"sha256": "e" * 64}}

    monkeypatch.setattr(V, "execute", fake_execute)
    return root, seen, tmp_path / "outside"


def test_the_five_steps_chain_and_the_run_record_names_every_artifact(bank, monkeypatch):
    root, seen, out = bank
    import joblib as jl
    monkeypatch.setattr(jl, "load", lambda p, *a, **k: {"loaded": str(p)})
    assert Run.preflight(RUN, "sums.txt")["passed"]
    assert Run.stage2_reuse(RUN)["decision"]["outcome"] == "reuse"
    glass = Run.execute(RUN, "glass", out)
    assert set(glass["produced"]) == {"stage1.lr", "stage3.ebm"}
    assert seen["glass"] == {P.PINS_ENV: ""}                         # GLASS reads no pins
    pins = json.loads((Rec.run_dir(RUN) / Run.PINS_FILE).read_text())
    assert sorted(pins["files"]) == ["stage1.lr", "stage2.glass_router", "stage2.rf_router",
                                     "stage3.ebm"]
    black = Run.execute(RUN, "blackbox", out)
    assert seen["lr_read"] == root / "models/lr/lr_calibrated_20261115_121000.joblib"
    assert set(black["produced"]) == {"stage1.mlp", "stage3.xgb"}
    rec = Run.finalize(RUN, "sums.txt")
    assert rec["passed"] and rec["accepted"] is False and "Step 4B" in rec["status"]
    assert sorted(rec["artifacts"]) == ["stage1.lr", "stage1.mlp", "stage2.glass_router",
                                        "stage2.rf_router", "stage3.ebm", "stage3.xgb"]
    for role, a in rec["artifacts"].items():
        assert a["sha256"] == P.sha256_file(root / a["path"]) and a["consumers"]
    assert rec["tracked_aliases_d7"]["models/mlp/stage1_mlp_threshold_sweep.csv"]["after"] != \
        rec["tracked_aliases_d7"]["models/mlp/stage1_mlp_threshold_sweep.csv"]["before"]
    assert {g for g in rec["gates"]} == {"A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6"}
    names = sorted(p.name for p in Rec.run_dir(RUN).iterdir())
    assert names == ["execution_blackbox.json", "execution_blackbox.json.sha256",
                     "execution_glass.json", "execution_glass.json.sha256", "preflight.json",
                     "preflight.json.sha256", "run_record.json", "run_record.json.sha256",
                     "stage2_reuse.json", "stage2_reuse.json.sha256", "upstream_pins.json",
                     "upstream_pins.json.sha256"]
    with pytest.raises(Rec.RecordError, match="already executed"):
        Run.execute(RUN, "glass", out)


def test_steps_out_of_order_are_refused(bank):
    _, _, out = bank
    with pytest.raises(Rec.RecordError, match="no preflight record"):
        Run.stage2_reuse(RUN)
    Run.preflight(RUN, "sums.txt")
    with pytest.raises(Rec.RecordError, match="no stage2_reuse record"):
        Run.execute(RUN, "glass", out)
    Run.stage2_reuse(RUN)
    with pytest.raises(Rec.RecordError, match="no execution_glass record"):
        Run.execute(RUN, "blackbox", out)
    with pytest.raises(Rec.RecordError, match="already has a preflight record"):
        Run.preflight(RUN, "sums.txt")


def test_a_stray_write_between_steps_or_a_code_change_stops_the_run(bank):
    root, _, out = bank
    Run.preflight(RUN, "sums.txt")
    Run.stage2_reuse(RUN)
    (root / "models/lr/lr_calibrated_20261115_000001.joblib").write_bytes(b"out of band")
    with pytest.raises(Run.StepFailed, match="nothing changed between steps"):
        Run.execute(RUN, "glass", out)
    (root / "shared" / "code.py").write_text("x = 2\n")
    with pytest.raises(Rec.RecordError, match="the code changed since preflight"):
        Run.execute(RUN, "glass", out)


def test_an_unregistered_write_by_a_notebook_fails_the_execution_and_writes_no_pins(bank, monkeypatch):
    root, _, out = bank
    Run.preflight(RUN, "sums.txt")
    Run.stage2_reuse(RUN)
    real = V.execute

    def legacy_refit(nb, notebook, out_dir, run_id, **k):
        got = real(nb, notebook, out_dir, run_id, **k)
        (root / "models/rf_router").mkdir(exist_ok=True)
        (root / "models/rf_router/rf_router_two_pass.joblib").write_bytes(b"legacy refit")
        return got

    monkeypatch.setattr(V, "execute", legacy_refit)
    with pytest.raises(Run.StepFailed, match="registered outputs"):
        Run.execute(RUN, "glass", out)
    assert not (Rec.run_dir(RUN) / Run.PINS_FILE).exists()
    assert not Rec.record_path(RUN, "execution_glass").exists()
    assert list((Rec.run_dir(RUN) / "failed").iterdir())


def test_a_regenerate_decision_stops_before_any_notebook_runs(bank, monkeypatch):
    _, seen, out = bank
    Run.preflight(RUN, "sums.txt")

    def failing(gate, split):
        gate.check("trigger 2: binning identity now = the Step 3A record's", lambda: False)
        return {}

    monkeypatch.setattr(C, "stage2_reuse", failing)
    with pytest.raises(Run.StepFailed, match=r"(?s)trigger 2.*decision: regenerate"):
        Run.stage2_reuse(RUN)
    with pytest.raises(Rec.RecordError, match="no stage2_reuse record"):
        Run.execute(RUN, "glass", out)
    assert seen == {}


def test_the_cli_reports_a_stop_and_exits_non_zero(bank, capsys):
    assert Run.main(["stage2-reuse", "--run-id", RUN]) == 1
    assert "STOPPED: no preflight record" in capsys.readouterr().err
    assert Run.main(["preflight", "--run-id", RUN, "--sums", "sums.txt"]) == 0
    assert f"OK preflight for run {RUN}" in capsys.readouterr().out
