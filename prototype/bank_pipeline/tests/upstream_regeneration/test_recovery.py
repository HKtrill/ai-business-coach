"""
PR 39 Step 4A: recovery of a completed GLASS execution (upstream_regeneration.recovery).

Each test starts from a real failed record: the real runner executes a fake
GLASS notebook (real artifact files and real input / executed notebook copies)
with the GLASS Cell 6 baseline LR missing from the allowlist, exactly the
refusal of run c4_20261115_204125. The recovery patch is then simulated (one
changed and one new file under ``upstream_regeneration/``) and:

* recovery passes, runs no notebook, leaves the failed record untouched,
  writes ``execution_glass_recovery.json`` once with both code identities and
  the run's pins file, and the black-box execution and ``finalize`` proceed on
  the recovered code without GLASS being run again;
* recovery is refused when the notebook copies, an artifact or the model
  folders changed, when a gate record was edited, when the code changed beyond
  the recovery's own files, when the attempt failed for another reason, when a
  legacy cell did not skip or an added file is not in the outputs, or when the
  accepted Stage 2 files changed;
* evidence stays write-once: no second recovery, no GLASS re-execution under
  the run id, no recovery of a run whose GLASS execution passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from shared import pinned_artifacts as P
from tests.support.stage4_world import build_world, make_stage3
from upstream_regeneration import checks as C
from upstream_regeneration import inventory as I
from upstream_regeneration import records as Rec
from upstream_regeneration import recovery as Recov
from upstream_regeneration import runner as Run
from upstream_regeneration import vehicle as V

RUN = "c4_20261115_204125"
REAL_COMPANIONS = copy.deepcopy(I.COMPANIONS)
LR = "models/lr/lr_calibrated_20261115_204501.joblib"
BASE = "models/lr/lr_baseline_20261115_204420.joblib"
EBM = "models/ebm/ebm_stage3_20261116_001529_813890.joblib"
EBM_JSON = "models/ebm/ebm_stage3_20261116_001529_813890.json"
BLOCK = "models/stage3/stage3_block_d6cddadc3439_c950583f1d15.joblib"
ALIAS = "models/ebm/ebm_stage3_latest.joblib"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _notebook():
    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}
    return {"cells": [cell("# CELL 6: BASELINE LR\ntrain_baseline_lr_stage(save=True)\n"),
                      cell("%%legacy_stage2\n# CELL 9: RF LIFT ANALYSIS\nanalyze_rf_lift()\n"),
                      cell("# CELL 8: STAGE 1 LR\ntrain_calibrated_lr_stage(save=True)\n"),
                      cell("# CELL 17: STAGE 3 BLOCK\nload_or_build_block()\n"),
                      cell("# CELL 18: STAGE 3 EBM\ntrain_ebm_stage()\n")],
            "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}


OUTPUTS = [f"💾 Saved baseline LR: ./models/lr\\{Path(BASE).name}",
           "skipped CELL 9: RF LIFT ANALYSIS (RUN_LEGACY_STAGE2 is False)",
           f"📦 Saved artifact → ./models/lr\\{Path(LR).name}",
           f"block built: {Path(BLOCK).name}",
           f"saved Stage 3 artifact → models/ebm\\{Path(EBM).name}\n"
           f"summary sidecar  → models/ebm\\{Path(EBM_JSON).name}"]


@pytest.fixture(scope="module")
def ebm_artifact():
    return make_stage3(build_world(n=600, n_test=150))


@pytest.fixture
def attempt(tmp_path, monkeypatch, ebm_artifact):
    """A run whose GLASS execution completed and was refused for the unregistered baseline."""
    root = tmp_path / "bank_pipeline"
    for rel in ("models/pr38/c3_20261112_153333/primary/glass_router",
                "models/pr38/c3_20261112_153333/primary/rf_router", "models/lr", "models/ebm",
                "models/stage3", "models/mlp", "blackbox_pipeline/models/xgb", "shared",
                "upstream_regeneration"):
        (root / rel).mkdir(parents=True)
    (root / "shared" / "code.py").write_text("x = 1\n")
    (root / "upstream_regeneration" / "inventory.py").write_text("REGISTERED = ()\n")
    (root / "models/lr/lr_calibrated_20261023_003319.joblib").write_bytes(b"PR 36 LR")
    (root / ALIAS).write_bytes(b"the previous EBM latest alias")
    stage2 = []
    for pin, payload in zip(P.STAGE2_PINS, (b"glass bundle", b"rf router")):
        (root / pin.path).write_bytes(payload)
        stage2.append(P.PinnedFile(pin.role, pin.path, _sha(payload), len(payload)))
    monkeypatch.setattr(P, "STAGE2_PINS", tuple(stage2))
    monkeypatch.setattr(Rec, "BANK_ROOT", root)
    monkeypatch.setattr(Rec, "RECORD_ROOT", root / "research_logs" / "pr39")
    monkeypatch.setattr(Rec, "git_status", lambda root=None: [" M glass_pipeline/glass_cascade.ipynb"])
    monkeypatch.setattr(Rec, "environment", lambda: {"python": "test"})

    def preservation(gate, sums):
        gate.check("7.5 stub", lambda: True)
        return {"report_check": {"passed": True}, "snapshot": {"live_counterparts": {
            "same": 24, "different": [], "missing": []}}, "baseline": {}}

    monkeypatch.setattr(C, "preservation", preservation)
    monkeypatch.setattr(C, "global_split", lambda: {"X_train": None, "y_train": None, "X_test": None})
    monkeypatch.setattr(C, "stage2_reuse", lambda gate, split: gate.check("A4.2 stub", lambda: True))
    import stage2_crossfit.acceptance as A
    import stage2_crossfit.runner as R
    monkeypatch.setattr(R, "check_population", lambda ts, protocol: {"passed": True, "checks": {}})
    monkeypatch.setattr(R, "make_train_side", lambda *a: None)
    monkeypatch.setattr(A, "require_accepted", lambda run_id: SimpleNamespace(manifest_sha256="m"))
    nb = _notebook()
    text = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    copy_rec = {"notebook": "glass_pipeline/glass_cascade.ipynb", "truncated_copy_sha256": _sha(text),
                "removed_code_cells": ["CELL 19", "CELL 20"], "legacy_switch": "RUN_LEGACY_STAGE2 = False"}
    monkeypatch.setattr(V, "truncated_copy", lambda name, root=None: (copy.deepcopy(nb), dict(copy_rec)))
    ctx = SimpleNamespace(root=root, out=tmp_path / "pr39_runs", executions=[], cell_error=False)

    def fake_execute(nb, notebook, out_dir, run_id, *, env=None, kernel_name=None, root=None):
        ctx.executions.append(notebook)
        out_dir.mkdir(parents=True, exist_ok=True)
        if notebook == "glass":
            joblib.dump({"artifact_version": "3.0", "stage_output": {"stage": "stage1"}}, root / LR)
            joblib.dump({"model_name": "logistic_regression_baseline", "hyperparameters": {"C": 1.0},
                         "model": None}, root / BASE)
            joblib.dump(ebm_artifact, root / EBM)
            joblib.dump(ebm_artifact, root / ALIAS)
            (root / EBM_JSON).write_text("{}")
            joblib.dump({"block": 1}, root / BLOCK)
        else:
            ctx.lr_read = P.pinned_path("stage1.lr", pins_file=env[P.PINS_ENV], root=root)
            vid = "20261116_030000_000001"
            joblib.dump({"mlp": 1}, root / f"models/mlp/stage1_mlp_{vid}.joblib")
            joblib.dump({"xgb": 1}, root / f"blackbox_pipeline/models/xgb/xgb_stage3_{vid}.joblib")
        src = out_dir / f"{run_id}_{notebook}_input.ipynb"
        exe = out_dir / f"{run_id}_{notebook}_executed.ipynb"
        src.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        done = copy.deepcopy(nb)
        for i, (c, line) in enumerate(zip(done["cells"], OUTPUTS), start=1):
            c["execution_count"] = i
            c["outputs"] = [{"output_type": "stream", "name": "stdout", "text": [line + "\n"]}]
        exe.write_text(json.dumps(done, indent=1), encoding="utf-8")
        return {"completed": not ctx.cell_error,
                "error": "ValueError: stage 1 failed" if ctx.cell_error else None,
                "wall_seconds": 12675.7, "kernel": {"name": "python3"},
                "input_copy": {"path": str(src), "sha256": _sha(src.read_bytes())},
                "executed_copy": {"path": str(exe), "sha256": _sha(exe.read_bytes())}}

    monkeypatch.setattr(V, "execute", fake_execute)

    def run_attempt(companions=None):
        Run.preflight(RUN, "SHA256SUMS.txt")
        Run.stage2_reuse(RUN)
        monkeypatch.setattr(I, "COMPANIONS", companions or {"glass": {}, "blackbox": {}})
        with pytest.raises(Run.StepFailed, match="execution_glass did not pass"):
            Run.execute(RUN, "glass", ctx.out)
        monkeypatch.setattr(I, "COMPANIONS", copy.deepcopy(REAL_COMPANIONS))
        (ctx.failed_path,) = (Rec.run_dir(RUN) / "failed").iterdir()
        ctx.failed_bytes = ctx.failed_path.read_bytes()
        # the recovery patch: one registry edit, one new module
        inv = root / "upstream_regeneration" / "inventory.py"
        ctx.changes = {"upstream_regeneration/inventory.py": Rec.file_sha256_lf(inv),
                       "upstream_regeneration/recovery.py": None}
        inv.write_text("REGISTERED = ('stage1.lr_baseline',)\n")
        (root / "upstream_regeneration" / "recovery.py").write_text("RECOVERY = True\n")
        return ctx

    ctx.run_attempt = run_attempt
    return ctx


def _recover(ctx, **kw):
    return Recov.recover_glass(RUN, changes=ctx.changes, **kw)


def _refused(ctx, failed_check, **kw):
    with pytest.raises(Run.StepFailed) as err:
        _recover(ctx, **kw)
    assert failed_check in str(err.value), str(err.value)
    assert not Rec.record_path(RUN, Recov.STEP).exists()
    assert not (Rec.run_dir(RUN) / Run.PINS_FILE).exists()
    assert ctx.failed_path.read_bytes() == ctx.failed_bytes            # original evidence untouched


# ----------------------------------------------------------------------
# the real failure, and its recovery
# ----------------------------------------------------------------------
def test_the_attempt_fails_exactly_as_run_c4_20261115_204125_did(attempt):
    ctx = attempt.run_attempt()
    failed = json.loads(ctx.failed_bytes)
    assert failed["inventory_delta"]["problems"] == [f"unregistered addition {BASE}"]
    assert failed["gates"]["execution"]["failed"] == [Recov.INVENTORY_FAILURE]
    assert failed["inventory_delta"]["produced"] == {"stage1.lr": LR, "stage3.ebm": EBM}
    assert failed["inventory_delta"]["diff"] == {"added": 5, "modified": 1, "deleted": 0}
    assert not Rec.record_path(RUN, "execution_glass").exists()


def test_recovery_validates_the_completed_execution_and_the_run_resumes_without_glass(attempt,
                                                                                      monkeypatch):
    ctx = attempt.run_attempt()
    rec = _recover(ctx)
    assert rec["passed"] and ctx.executions == ["glass"]            # GLASS ran once, never again
    assert ctx.failed_path.read_bytes() == ctx.failed_bytes          # the failed record untouched
    stored = Rec.require_passed(RUN, Recov.STEP)                     # written once, with a sidecar
    assert stored["original"]["failed_record"]["sha256"] == _sha(ctx.failed_bytes)
    assert stored["original"]["failed_record"]["inventory_problems"] == [f"unregistered addition {BASE}"]
    assert stored["code"]["run"]["code_sha256"] == stored["code"]["restored_identity"]["code_sha256"]
    assert stored["code"]["recovered_under"]["code_sha256"] != stored["code"]["run"]["code_sha256"]
    assert {c["path"] for c in stored["code"]["changes"]} == set(ctx.changes)
    assert stored["companions"]["stage1.lr_baseline"]["path"] == BASE
    assert sorted(stored["produced"]) == ["stage1.lr", "stage3.ebm"]
    assert stored["reload"]["stage3.ebm"]["type"] == "shared.stage3.artifact.Stage3Artifact"
    assert not Rec.record_path(RUN, "execution_glass").exists()     # never presented as executed
    pins = P.read_pins(Rec.run_dir(RUN) / Run.PINS_FILE)
    assert pins["files"]["stage1.lr"]["path"] == LR and pins["files"]["stage3.ebm"]["path"] == EBM
    # the black-box notebook, then finalize, on the recovered code
    Run.execute(RUN, "blackbox", ctx.out)
    assert ctx.executions == ["glass", "blackbox"] and ctx.lr_read == ctx.root / LR
    for name in ("stage1_checks", "stage3_checks", "stage4_input_checks", "fresh_reload"):
        monkeypatch.setattr(C, name, lambda gate, *a, **k: gate.check("stub", lambda: True))
    monkeypatch.setattr(C, "frozen_population", lambda: {})
    monkeypatch.setattr(C, "identities", lambda split, pop: {})
    monkeypatch.setattr(joblib, "load", lambda p, *a, **k: {"loaded": str(p)})
    record = Run.finalize(RUN, "SHA256SUMS.txt")
    assert record["passed"] and record["accepted"] is False
    assert record["executions"]["glass"]["record"] == f"{Recov.STEP}.json"
    assert record["glass_recovery"]["failed_record"]["sha256"] == _sha(ctx.failed_bytes)
    assert record["code"]["code_sha256"] == stored["code"]["recovered_under"]["code_sha256"]
    assert sorted(record["artifacts"]) == ["stage1.lr", "stage1.mlp", "stage2.glass_router",
                                           "stage2.rf_router", "stage3.ebm", "stage3.xgb"]


def test_the_cli_recovers_with_the_real_changes_table_only(attempt, capsys):
    """The command uses ``recovery.CHANGES`` (the patch's files), never the test's table: on
    this synthetic tree it cannot reproduce the run's code identity, so it refuses."""
    attempt.run_attempt()
    assert Run.main(["recover-glass", "--run-id", RUN]) == 1
    assert "restoring exactly those files reproduces the run's preflight code identity" \
        in capsys.readouterr().err


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------
def _rewrite_executed(ctx, edit):
    exe = next(ctx.out.glob("*_glass_executed.ipynb"))
    nb = json.loads(exe.read_text(encoding="utf-8"))
    edit(nb)
    exe.write_text(json.dumps(nb, indent=1), encoding="utf-8")


@pytest.mark.parametrize("tamper, failed_check", [
    (lambda c: next(c.out.glob("*_glass_executed.ipynb")).write_text("{}"),
     "the executed copy has its recorded SHA-256"),
    (lambda c: next(c.out.glob("*_glass_input.ipynb")).write_bytes(b"{}\n"),
     "the input copy has its recorded SHA-256"),
    (lambda c: joblib.dump({"artifact_version": "3.0", "retrained": True}, c.root / LR),
     "the model folders are byte for byte as the attempt left them"),
    (lambda c: (c.root / "models/lr/lr_calibrated_20261116_090000.joblib").write_bytes(b"x"),
     "the model folders are byte for byte as the attempt left them"),
    (lambda c: (c.root / "shared" / "code.py").write_text("x = 2\n"),
     "restoring exactly those files reproduces the run's preflight code identity"),
    (lambda c: c.changes.pop("upstream_regeneration/inventory.py"),
     "restoring exactly those files reproduces the run's preflight code identity"),
    (lambda c: c.changes.update({"shared/code.py": Rec.file_sha256_lf(c.root / "shared/code.py")}),
     "confined to upstream_regeneration code and its tests"),
    (lambda c: (c.root / P.STAGE2_PINS[0].path).write_bytes(b"glass bundle!"),
     "stage2.glass_router: the accepted file matches its pin"),
])
def test_recovery_is_refused_when_any_evidence_disagrees(attempt, tamper, failed_check):
    ctx = attempt.run_attempt()
    tamper(ctx)
    _refused(ctx, failed_check)


@pytest.mark.parametrize("edit, failed_check", [
    (lambda nb: nb["cells"][1]["outputs"].clear(), "every legacy Stage 2 cell printed its skip"),
    (lambda nb: nb["cells"][0]["outputs"].clear(), "is named in the executed notebook's outputs"),
    (lambda nb: nb["cells"][4].update(execution_count=None), "every code cell executed"),
    (lambda nb: nb["cells"][2]["outputs"].append({"output_type": "error", "ename": "KeyError"}),
     "every code cell executed, with no error output"),
])
def test_recovery_reads_the_executed_notebook_itself(attempt, edit, failed_check):
    """Edited copies whose hash is re-recorded would still be refused on their content."""
    ctx = attempt.run_attempt()
    _rewrite_executed(ctx, edit)
    exe = next(ctx.out.glob("*_glass_executed.ipynb"))
    failed = json.loads(ctx.failed_bytes)
    failed["execution"]["executed_copy"]["sha256"] = _sha(exe.read_bytes())
    ctx.failed_path.write_bytes(json.dumps(failed).encode("utf-8"))
    ctx.failed_bytes = ctx.failed_path.read_bytes()
    _refused(ctx, failed_check)


def test_an_attempt_that_failed_for_another_reason_is_not_recoverable(attempt):
    attempt.cell_error = True
    ctx = attempt.run_attempt()
    _refused(ctx, "its notebook execution completed with no cell error")


def test_an_attempt_with_another_unregistered_file_is_not_recoverable(attempt, monkeypatch):
    real = V.execute

    def legacy_write(*a, **k):
        out = real(*a, **k)
        (attempt.root / "models/glass_router").mkdir(exist_ok=True)
        (attempt.root / "models/glass_router/glass_router_20261115.joblib").write_bytes(b"legacy")
        return out

    monkeypatch.setattr(V, "execute", legacy_write)
    ctx = attempt.run_attempt()
    _refused(ctx, "each of its inventory problems is an addition this recovery registers")


def test_an_edited_gate_record_stops_recovery_before_any_check(attempt):
    ctx = attempt.run_attempt()
    pre = Rec.record_path(RUN, "preflight")
    pre.write_bytes(pre.read_bytes().replace(b'"passed": true', b'"passed": true '))
    with pytest.raises(Rec.RecordError, match="does not match its preflight.json.sha256 sidecar"):
        _recover(ctx)


# ----------------------------------------------------------------------
# write-once evidence and the code identity after recovery
# ----------------------------------------------------------------------
def test_recovery_is_written_once_and_glass_is_never_re_executed(attempt):
    ctx = attempt.run_attempt()
    _recover(ctx)
    with pytest.raises(Rec.RecordError, match="already recovered"):
        _recover(ctx)
    with pytest.raises(Rec.RecordError, match="already executed the glass notebook"):
        Run.execute(RUN, "glass", ctx.out)
    assert ctx.executions == ["glass"]


def test_a_passed_glass_execution_has_nothing_to_recover(attempt):
    ctx = attempt
    Run.preflight(RUN, "SHA256SUMS.txt")
    Run.stage2_reuse(RUN)
    Run.execute(RUN, "glass", ctx.out)                                 # registered: it passes
    with pytest.raises(Rec.RecordError, match="nothing to recover"):
        Recov.recover_glass(RUN, changes={"upstream_regeneration/recovery.py": None})


def test_after_recovery_the_run_holds_to_the_recovered_code(attempt):
    ctx = attempt.run_attempt()
    _recover(ctx)
    (ctx.root / "upstream_regeneration" / "recovery.py").unlink()     # back to the run's code
    (ctx.root / "upstream_regeneration" / "inventory.py").write_text("REGISTERED = ()\n")
    with pytest.raises(Rec.RecordError, match="the code changed since the recorded recovery"):
        Run.execute(RUN, "blackbox", ctx.out)
    assert ctx.executions == ["glass"]


def test_more_than_one_failed_attempt_must_be_named(attempt):
    ctx = attempt.run_attempt()
    twin = ctx.failed_path.with_name("execution_glass_20990101T000000000000Z.json")
    twin.write_bytes(ctx.failed_bytes)
    with pytest.raises(Rec.RecordError, match="2 failed GLASS execution records"):
        _recover(ctx)
    with pytest.raises(Rec.RecordError, match="is not a failed GLASS execution record"):
        _recover(ctx, failed_record=ctx.root / "elsewhere.json")
    assert _recover(ctx, failed_record=ctx.failed_path)["passed"]


def test_moved_notebook_copies_are_found_by_name_and_verified_by_hash(attempt, tmp_path):
    ctx = attempt.run_attempt()
    moved = tmp_path / "moved"
    ctx.out.rename(moved)
    _refused(ctx, "the input copy has its recorded SHA-256")         # a refused attempt ...
    assert len(list((Rec.run_dir(RUN) / "failed").glob("execution_glass_recovery_*.json"))) == 1
    assert _recover(ctx, copies_dir=moved)["passed"]                  # ... does not block a retry


# ----------------------------------------------------------------------
# the real recovery table
# ----------------------------------------------------------------------
def test_the_recovery_changes_table_names_only_this_recoverys_files():
    from tests.support.notebooks import BANK_ROOT
    assert Recov.CHANGES and Recov.CHANGES["upstream_regeneration/recovery.py"] is None
    for rel, before in Recov.CHANGES.items():
        assert rel.startswith(Recov.ALLOWED_PREFIXES), rel
        assert (BANK_ROOT / rel).is_file(), rel
        if before is not None:                                     # changed by the patch
            assert Rec.file_sha256_lf(BANK_ROOT / rel) != before, rel
    assert Recov.RESOLVES == (I.COMPANIONS["glass"]["stage1.lr_baseline"],)
