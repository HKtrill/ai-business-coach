"""
PR 39 Step 4A: the registered code transition after a failed ``finalize``
(upstream_regeneration.transition).

Each test drives the real runner on a temporary bank_pipeline/ (the fixture of
test_recovery.py): preflight, stage2_reuse, a GLASS execution that is refused
and recovered (run c4_20261115_204125's path) or passes, the black-box
execution, then a ``finalize`` that fails the one A4.4 check the run's code
got wrong. The transition patch is then simulated (one file changed again, one
new file, under ``upstream_regeneration/``) and:

* the transition passes, runs and fits nothing, writes
  ``finalize_transition.json`` once with both code identities and the owner's
  statement, names the failed attempt by SHA-256 without parsing it, and
  ``finalize`` then rechecks A4.3-A4.6 and writes the run record once, with
  every artifact hash unchanged and both failed attempts preserved;
* it is refused when the code changed beyond its files or outside
  ``upstream_regeneration``, when anything in the model folders changed, when
  the corrected A4.4 does not pass, without a statement, without a failed
  attempt, after a passed ``finalize`` or a second time;
* the code-identity guard is intact: without a transition the changed code
  is refused, after it the old code is, and an edited failed attempt stops
  ``finalize``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from shared import pinned_artifacts as P
from tests.upstream_regeneration.test_recovery import (RUN, attempt,  # noqa: F401 (fixtures)
                                                       ebm_artifact)
from upstream_regeneration import checks as C
from upstream_regeneration import records as Rec
from upstream_regeneration import recovery as Recov
from upstream_regeneration import runner as Run
from upstream_regeneration import transition as T

STATEMENT = "Owner: authorize commit4a-t1 for run c4_20261115_204125 (PR 39 Commit 4A review)."
OBSERVED = ("Stage 3 arms did not run the same protocol:\n"
            "  split_fingerprint: 'd6cddadc3439' vs 'fc3e73a5ef91aea4'")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _passing(gate, *a, **k):
    gate.check("stub", lambda: True)
    return {"stub": True}


def _as_the_run_code_did(gate, *a, **k):
    def raw():
        raise AssertionError(OBSERVED)
    gate.check(T.RESOLVES["failed_check"], raw)
    return {}


def _corrected(gate, *a, **k):
    assert "split" in k                                  # finalize and the transition pass the split
    gate.check(C.STAGE3_SPLIT_IDENTITY, lambda: True)
    gate.check(C.STAGE3_PROTOCOL, lambda: True)
    return {"split_identity": {"stage3.ebm": {"scheme": "stage_io"},
                               "stage3.xgb": {"scheme": "router"}}}


@pytest.fixture
def failed_finalize(attempt, monkeypatch):
    """Through the recovered GLASS execution and the black-box execution to a failed finalize."""
    def run(recover=True):
        ctx = attempt
        if recover:
            attempt.run_attempt()
            Recov.recover_glass(RUN, changes=ctx.changes)
        else:
            Run.preflight(RUN, "SHA256SUMS.txt")
            Run.stage2_reuse(RUN)
            Run.execute(RUN, "glass", ctx.out)
        Run.execute(RUN, "blackbox", ctx.out)
        for name in ("stage1_checks", "stage4_input_checks", "fresh_reload"):
            monkeypatch.setattr(C, name, _passing)
        monkeypatch.setattr(C, "frozen_population", lambda: {})
        monkeypatch.setattr(C, "identities", lambda split, pop: {})
        monkeypatch.setattr(C, "stage3_checks", _as_the_run_code_did)
        with pytest.raises(Run.StepFailed, match="run_record did not pass"):
            Run.finalize(RUN, "SHA256SUMS.txt")
        failed = Rec.run_dir(RUN) / "failed"
        (ctx.finalize_failed,) = failed.glob("run_record_*.json")
        ctx.finalize_failed_bytes = ctx.finalize_failed.read_bytes()
        ctx.all_failed = {p.name: p.read_bytes() for p in failed.iterdir()}
        ctx.artifacts = {p.relative_to(ctx.root).as_posix(): _sha(p.read_bytes())
                         for p in (ctx.root / "models").rglob("*.joblib")}
        # the transition patch: the recovery's module edited again, one new module
        inv = ctx.root / "upstream_regeneration" / "inventory.py"
        ctx.t_changes = {"upstream_regeneration/inventory.py": Rec.file_sha256_lf(inv),
                         "upstream_regeneration/transition.py": None}
        inv.write_text(inv.read_text() + "SCHEMES = ('stage_io', 'router')\n")
        (ctx.root / "upstream_regeneration" / "transition.py").write_text("TRANSITION = True\n")
        monkeypatch.setattr(C, "stage3_checks", _corrected)
        return ctx
    return run


def _transition(ctx, **kw):
    return T.transition_finalize(RUN, authorization=kw.pop("authorization", STATEMENT),
                                 changes=kw.pop("changes", ctx.t_changes), **kw)


def _refused(ctx, failed_check, **kw):
    with pytest.raises(Run.StepFailed) as err:
        _transition(ctx, **kw)
    assert failed_check in str(err.value), str(err.value)
    assert not Rec.record_path(RUN, T.STEP).exists()
    assert ctx.finalize_failed.read_bytes() == ctx.finalize_failed_bytes


# ----------------------------------------------------------------------
# the failure, the transition and the finalization
# ----------------------------------------------------------------------
def test_the_first_finalize_fails_only_the_raw_protocol_check(failed_finalize):
    ctx = failed_finalize()
    rec = json.loads(ctx.finalize_failed_bytes)
    assert {g: r["failed"] for g, r in rec["gates"].items() if isinstance(r, dict)
            and r.get("failed")} == {"A4.4": [T.RESOLVES["failed_check"]]}
    assert not Rec.record_path(RUN, "run_record").exists()


def test_the_transition_is_recorded_and_finalize_rechecks_everything(failed_finalize):
    ctx = failed_finalize()
    recovered = Rec.require_passed(RUN, Recov.STEP)["code"]["recovered_under"]
    tr = _transition(ctx)
    assert tr["passed"]
    stored = Rec.require_passed(RUN, T.STEP)                                   # once, with a sidecar
    assert stored["code"]["from"] == recovered                                 # the run's code
    assert stored["code"]["restored_identity"]["code_sha256"] == recovered["code_sha256"]
    assert stored["code"]["to"] == Rec.code_identity(ctx.root)
    assert stored["code"]["to"]["code_sha256"] != recovered["code_sha256"]
    assert {c["path"] for c in stored["code"]["changes"]} == set(ctx.t_changes)
    assert stored["authorization"]["statement"] == STATEMENT
    assert stored["original"]["failed_finalize"] == {
        "path": ctx.finalize_failed.relative_to(ctx.root).as_posix(),
        "bytes": len(ctx.finalize_failed_bytes), "sha256": _sha(ctx.finalize_failed_bytes),
        "read": False}
    assert stored["resolves"]["failed_check"] == T.RESOLVES["failed_check"]
    assert stored["evidence"]["stage3"]["split_identity"]["stage3.xgb"]["scheme"] == "router"
    assert set(stored["records"]) == {"preflight", "stage2_reuse", Recov.STEP, "execution_blackbox"}
    assert sorted(stored["artifacts"]) == ["stage1.lr", "stage1.mlp", "stage2.glass_router",
                                           "stage2.rf_router", "stage3.ebm", "stage3.xgb"]

    record = Run.finalize(RUN, "SHA256SUMS.txt")
    assert record["passed"] and record["accepted"] is False and "Step 4B" in record["status"]
    assert ctx.executions == ["glass", "blackbox"]                     # nothing re-executed
    assert record["code"]["code_sha256"] == stored["code"]["to"]["code_sha256"]
    ct = record["code_transition"]
    assert ct["sha256"] == (Rec.record_path(RUN, T.STEP).with_name(f"{T.STEP}.json.sha256")
                            .read_text().strip())
    assert ct["code"]["from"] == recovered and ct["failed_finalize"]["read"] is False
    assert record["glass_recovery"]["code"]["recovered_under"] == recovered
    assert [f["by"] for f in record["failed_attempts_preserved"]] == [Recov.STEP, T.STEP]
    a46 = {c["check"]: c["passed"] for c in record["gates"]["A4.6"]["checks"]}
    assert a46["the failed attempts the run's recovery and transition name are preserved "
               "(path, SHA-256; hashed, not read)"]
    assert {g for g in record["gates"]} == {"A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6"}
    assert [c["check"] for c in record["gates"]["A4.4"]["checks"]] == [
        C.STAGE3_SPLIT_IDENTITY, C.STAGE3_PROTOCOL]
    for role, a in record["artifacts"].items():                       # the files the run produced
        assert a["sha256"] == stored["artifacts"][role]["sha256"]
    assert {p.relative_to(ctx.root).as_posix(): _sha(p.read_bytes())
            for p in (ctx.root / "models").rglob("*.joblib")} == ctx.artifacts
    failed = Rec.run_dir(RUN) / "failed"
    assert {p.name: p.read_bytes() for p in failed.iterdir()} == ctx.all_failed   # untouched
    with pytest.raises(Rec.RecordError, match="written once"):
        Run.finalize(RUN, "SHA256SUMS.txt")


def test_a_run_without_recovery_transitions_from_its_preflight_code(failed_finalize):
    ctx = failed_finalize(recover=False)
    pre = Rec.require_passed(RUN, "preflight")
    assert _transition(ctx)["code"]["from"] == pre["code"]
    record = Run.finalize(RUN, "SHA256SUMS.txt")
    assert record["passed"] and record["glass_recovery"] is None
    assert [f["by"] for f in record["failed_attempts_preserved"]] == [T.STEP]


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tamper, failed_check", [
    (lambda c: (c.root / "shared" / "code.py").write_text("x = 2\n"),
     "restoring exactly those files reproduces the run's code identity"),
    (lambda c: c.t_changes.pop("upstream_regeneration/inventory.py"),
     "restoring exactly those files reproduces the run's code identity"),
    (lambda c: c.t_changes.update({"shared/code.py": Rec.file_sha256_lf(c.root / "shared/code.py")}),
     "confined to upstream_regeneration code and its tests"),
    (lambda c: (c.root / "models/lr/lr_calibrated_20261116_090000.joblib").write_bytes(b"x"),
     "the model folders are byte for byte as the black-box execution left them"),
    (lambda c: (c.root / P.STAGE2_PINS[1].path).write_bytes(b"rf router!"),
     "stage2.rf_router"),
])
def test_the_transition_is_refused_when_any_evidence_disagrees(failed_finalize, tamper,
                                                               failed_check):
    ctx = failed_finalize()
    tamper(ctx)
    _refused(ctx, failed_check)


def test_a_correction_that_does_not_resolve_a44_is_refused(failed_finalize, monkeypatch):
    ctx = failed_finalize()
    monkeypatch.setattr(C, "stage3_checks", _as_the_run_code_did)
    _refused(ctx, "the corrected A4.4 passes on the run's pinned EBM and XGBoost")


@pytest.mark.parametrize("statement", ["", "   ", "x" * 501])
def test_the_owners_statement_is_required(failed_finalize, statement):
    ctx = failed_finalize()
    _refused(ctx, "the owner's authorization of this transition is stated", authorization=statement)


def test_no_transition_without_a_failed_finalize(attempt):
    ctx = attempt
    Run.preflight(RUN, "SHA256SUMS.txt")
    Run.stage2_reuse(RUN)
    Run.execute(RUN, "glass", ctx.out)
    Run.execute(RUN, "blackbox", ctx.out)
    with pytest.raises(Rec.RecordError, match="0 failed finalize attempts"):
        T.transition_finalize(RUN, authorization=STATEMENT,
                              changes={"upstream_regeneration/transition.py": None})


def test_the_transition_is_written_once_and_not_after_a_passed_finalize(failed_finalize):
    ctx = failed_finalize()
    _transition(ctx)
    with pytest.raises(Rec.RecordError, match="already recorded"):
        _transition(ctx)
    Run.finalize(RUN, "SHA256SUMS.txt")
    with pytest.raises(Rec.RecordError, match="nothing to transition"):
        _transition(ctx)


def test_more_than_one_failed_finalize_must_be_named(failed_finalize):
    ctx = failed_finalize()
    twin = ctx.finalize_failed.with_name("run_record_20990101T000000000000Z.json")
    twin.write_bytes(ctx.finalize_failed_bytes)
    with pytest.raises(Rec.RecordError, match="2 failed finalize attempts"):
        _transition(ctx)
    with pytest.raises(Rec.RecordError, match="is not a failed finalize attempt"):
        _transition(ctx, failed_record=ctx.root / "elsewhere.json")
    rec = _transition(ctx, failed_record=twin)
    assert rec["original"]["failed_finalize"]["path"].endswith(twin.name)


@pytest.mark.parametrize("extra, passes", [
    ([], True),
    (["?? prototype/bank_pipeline/shared/stage3/new_module.py"], False),
])
def test_git_status_admits_exactly_the_recovery_and_transition_files(failed_finalize, monkeypatch,
                                                                     extra, passes):
    ctx = failed_finalize()
    _transition(ctx)
    pre = Rec.require_passed(RUN, "preflight")["git_status"]
    new = ["?? prototype/bank_pipeline/upstream_regeneration/recovery.py",
           "?? prototype/bank_pipeline/upstream_regeneration/transition.py"]
    monkeypatch.setattr(Rec, "git_status", lambda root=None: sorted(pre + new + extra))
    if passes:
        rec = Run.finalize(RUN, "SHA256SUMS.txt")
        assert rec["git_status_delta"]["new"] == new and not rec["git_status_delta"]["unexpected"]
    else:
        with pytest.raises(Run.StepFailed, match="7.5.6 git status"):
            Run.finalize(RUN, "SHA256SUMS.txt")


# ----------------------------------------------------------------------
# the code-identity guard
# ----------------------------------------------------------------------
def test_without_the_transition_the_changed_code_is_refused(failed_finalize):
    failed_finalize()
    with pytest.raises(Rec.RecordError, match="the code changed since the recorded recovery"):
        Run.finalize(RUN, "SHA256SUMS.txt")


def test_after_the_transition_the_run_holds_to_the_transitioned_code(failed_finalize):
    ctx = failed_finalize()
    _transition(ctx)
    (ctx.root / "upstream_regeneration" / "transition.py").unlink()     # back to the run's code
    with pytest.raises(Rec.RecordError, match="the code changed since the recorded code transition"):
        Run.finalize(RUN, "SHA256SUMS.txt")
    assert not Rec.record_path(RUN, "run_record").exists()


@pytest.mark.parametrize("which", ["finalize", "glass"])
def test_an_edited_or_missing_failed_attempt_stops_finalize(failed_finalize, which):
    ctx = failed_finalize()
    _transition(ctx)
    target = ctx.finalize_failed if which == "finalize" else next(
        (Rec.run_dir(RUN) / "failed").glob("execution_glass_2*.json"))
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Run.StepFailed, match="are preserved"):
        Run.finalize(RUN, "SHA256SUMS.txt")
    assert not Rec.record_path(RUN, "run_record").exists()


def test_an_edited_transition_record_stops_finalize(failed_finalize):
    ctx = failed_finalize()
    _transition(ctx)
    p = Rec.record_path(RUN, T.STEP)
    p.write_bytes(p.read_bytes().replace(b'"passed": true', b'"passed": true '))
    with pytest.raises(Rec.RecordError, match="does not match its finalize_transition.json.sha256"):
        Run.finalize(RUN, "SHA256SUMS.txt")


def test_the_cli_uses_the_real_changes_table_and_needs_a_statement(failed_finalize, capsys):
    """``CHANGES`` is the patch's files, never the test's table: on this synthetic tree it
    cannot reproduce the run's code identity, so the command refuses."""
    failed_finalize()
    with pytest.raises(SystemExit):
        Run.main(["finalize-transition", "--run-id", RUN])
    assert Run.main(["finalize-transition", "--run-id", RUN, "--authorization", STATEMENT]) == 1
    assert "restoring exactly those files reproduces the run's code identity" \
        in capsys.readouterr().err


# ----------------------------------------------------------------------
# the real transition table
# ----------------------------------------------------------------------
def test_the_transition_changes_table_names_only_this_transitions_files():
    from tests.support.notebooks import BANK_ROOT
    assert T.CHANGES and T.CHANGES["upstream_regeneration/transition.py"] is None
    assert {"upstream_regeneration/checks.py", "upstream_regeneration/runner.py"} <= set(T.CHANGES)
    for rel, before in T.CHANGES.items():
        assert rel.startswith(T.ALLOWED_PREFIXES), rel
        assert (BANK_ROOT / rel).is_file(), rel
        if before is not None:                                     # changed by the patch
            assert Rec.file_sha256_lf(BANK_ROOT / rel) != before, rel
    assert T.RESOLVES["replaced_by"] == [C.STAGE3_SPLIT_IDENTITY, C.STAGE3_PROTOCOL]
