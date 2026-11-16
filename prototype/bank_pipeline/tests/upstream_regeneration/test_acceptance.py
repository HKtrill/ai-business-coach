"""
PR 39 Commit 4, Step 4B: gate A4.7 (upstream_regeneration.acceptance).

Each test drives the real runner on a temporary bank_pipeline/ through run
c4_20261115_204125's path: preflight, stage2_reuse, a refused and recovered
GLASS execution, the black-box execution (which rewrites one of the three D7
CSVs), a failed ``finalize``, the registered transition and a passed
``finalize``. The Step 4B files are then added and:

* acceptance passes. It writes ``artifact_manifest.json`` (a valid pins file
  for all six roles) and ``upstream_acceptance.json`` once each, rewrites no
  record and keeps ``run_record.json`` at ``accepted: false``. It derives the
  run's, the recovered and the preflight code identities from the tree, stages
  only the D7 CSV whose content differs, and ``check`` re-verifies all of it;
* it is refused for a missing, edited or consistently rewritten record, a
  failed or altered recorded gate, altered evidence, a changed artifact,
  missing provenance (a failed attempt, a notebook copy), a changed notebook,
  code beyond its own new files, a tampered transition, bad statements, and a
  second acceptance.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from shared import pinned_artifacts as P
from tests.upstream_regeneration.test_finalize_transition import (  # noqa: F401 (fixtures)
    _transition, failed_finalize)
from tests.upstream_regeneration.test_recovery import (RUN, attempt,  # noqa: F401 (fixtures)
                                                       ebm_artifact)
from upstream_regeneration import acceptance as A
from upstream_regeneration import records as Rec
from upstream_regeneration import runner as Run
from upstream_regeneration import transition as T
from upstream_regeneration import vehicle as V

STATEMENTS = {
    "decision": "Owner: accept the Stage 1-3 artifacts of run c4_20261115_204125 (A4.7).",
    "disposition_7_1": "Owner: recover-glass parsed the failed GLASS record; accepted as a "
                       "recorded run-specific deviation.",
    "disposition_7_2": "Owner: the registered transition commit4a-t1 is approved as recorded.",
}
SWEEP = "models/mlp/stage1_mlp_threshold_sweep.csv"
STATUS = " M glass_pipeline/glass_cascade.ipynb"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def finalized(attempt, failed_finalize, monkeypatch, tmp_path):
    """A finalized run with the Step 4B files added; nothing accepted yet."""
    ctx = attempt
    for rel in A.D7:
        (ctx.root / rel).write_text(f"{rel}: committed content\n")
    real = V.execute

    def black_box_rewrites_one_csv(nb, notebook, out_dir, run_id, **kw):
        if notebook == "blackbox":
            (ctx.root / SWEEP).write_text("threshold,recall\n0.31,0.90\n")
        return real(nb, notebook, out_dir, run_id, **kw)

    monkeypatch.setattr(V, "execute", black_box_rewrites_one_csv)
    failed_finalize()
    _transition(ctx)
    Run.finalize(RUN, "SHA256SUMS.txt")
    for rel in A.ACCEPTANCE_FILES:                               # the Step 4B patch: new files only
        (ctx.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (ctx.root / rel).write_text("# Step 4B\n")
    ctx.git = [STATUS, f" M prototype/bank_pipeline/{SWEEP}",
               *(f"?? prototype/bank_pipeline/{rel}" for rel in A.ACCEPTANCE_FILES)]
    monkeypatch.setattr(Rec, "git_status", lambda root=None: list(ctx.git))
    ctx.statements = tmp_path / "statements.json"
    ctx.statements.write_text(json.dumps(STATEMENTS), encoding="utf-8")
    ctx.records = {p.name: p.read_bytes() for p in Rec.run_dir(RUN).rglob("*") if p.is_file()}
    return ctx


def _accept(ctx, **kw):
    return A.accept(RUN, "SHA256SUMS.txt", kw.pop("statements", ctx.statements), **kw)


def _refused(ctx, failed_check, **kw):
    with pytest.raises(Run.StepFailed) as err:
        _accept(ctx, **kw)
    assert failed_check in str(err.value), str(err.value)
    for name in (A.STEP, A.MANIFEST):
        assert not Rec.record_path(RUN, name).exists()
    assert list((Rec.run_dir(RUN) / "failed").glob(f"{A.STEP}_*.json"))


def _rewrite(name, edit):
    """Edit a record and rewrite its sidecar to match: only the links can catch it."""
    p = Rec.record_path(RUN, name)
    rec = json.loads(p.read_bytes())
    edit(rec)
    data = Rec._text(rec)
    p.write_bytes(data)
    p.with_name(p.name + ".sha256").write_text(_sha(data) + "\n")


# ----------------------------------------------------------------------
# the acceptance
# ----------------------------------------------------------------------
def test_acceptance_verifies_the_run_and_writes_two_records_once(finalized):
    ctx = finalized
    rr = Rec.require_passed(RUN, "run_record")
    rec = _accept(ctx)
    assert rec["passed"] and rec["decision"]["accepted"] is True
    assert rec["decision"]["statement"] == STATEMENTS["decision"]
    assert {k: v["statement"] for k, v in rec["dispositions"].items()} == {
        "7.1": STATEMENTS["disposition_7_1"], "7.2": STATEMENTS["disposition_7_2"]}
    stored = Rec.require_passed(RUN, A.STEP)                               # once, with its sidecar
    assert stored["artifact_manifest"]["sha256"] == Run._sidecar(RUN, A.MANIFEST)
    # nothing rewritten; the run record stays the regeneration record
    for p in Rec.run_dir(RUN).rglob("*"):
        if p.is_file() and p.name in ctx.records:
            assert p.read_bytes() == ctx.records[p.name], p.name
    assert Rec.require_passed(RUN, "run_record")["accepted"] is False
    assert set(stored["records"]) == {"preflight", "stage2_reuse", "execution_glass_recovery",
                                      "execution_blackbox", T.STEP, "run_record"}
    # the code chain, derived from the tree
    pre = Rec.require_passed(RUN, "preflight")
    recov = Rec.require_passed(RUN, "execution_glass_recovery")
    derived = stored["code"]["derived"]
    assert derived["run"]["code_sha256"] == rr["code"]["code_sha256"]
    assert derived["before_transition"]["code_sha256"] == recov["code"]["recovered_under"]["code_sha256"]
    assert derived["preflight"]["code_sha256"] == pre["code"]["code_sha256"]
    assert stored["code"]["acceptance"]["code_sha256"] != rr["code"]["code_sha256"]
    # recomputed gates, the same checks as recorded
    for g in ("A4.3", "A4.4", "A4.5", "A4.6"):
        assert [c["check"] for c in stored["gates"][f"{g} recomputed"]["checks"]] == \
            [c["check"] for c in rr["gates"][g]["checks"]]
    # the manifest: a pins file for the six roles, plus every changed file
    mpath = Rec.record_path(RUN, A.MANIFEST)
    manifest = P.read_pins(mpath)
    assert manifest["kind"] == "pr39.artifact_manifest/1"
    for role in A.ROLES:
        assert P.pinned_path(role, pins_file=mpath, root=ctx.root) == ctx.root / rr["artifacts"][role]["path"]
        assert manifest["files"][role]["sha256"] == rr["artifacts"][role]["sha256"]
    kinds = {f["path"]: f["kind"] for f in manifest["inventory"]["files"]}
    assert kinds[SWEEP] == "mutable_alias" and "unclassified" not in kinds.values()
    assert "companion" in kinds.values()
    # D7: decided by content, not assumed
    d7 = stored["d7"]
    assert d7[SWEEP]["content_differs_from_head"] and d7[SWEEP]["disposition"].startswith("stage")
    for rel in set(A.D7) - {SWEEP}:
        assert d7[rel]["disposition"].startswith("nothing to stage"), rel
    out = A.check(RUN)
    assert out["accepted"] and list(out["roles"]) == list(A.ROLES)
    assert {r: f["sha256"] for r, f in out["roles"].items()} == {
        r: a["sha256"] for r, a in rr["artifacts"].items()}
    with pytest.raises(Rec.RecordError, match="already accepted"):
        _accept(ctx)


def test_a_csv_git_reports_equal_to_head_is_not_staged(finalized):
    ctx = finalized
    ctx.git = [ln for ln in ctx.git if SWEEP not in ln]
    rec = _accept(ctx)
    assert all(v["disposition"].startswith("nothing to stage") for v in rec["d7"].values())


def test_a_partial_acceptance_reuses_only_an_identical_manifest(finalized, monkeypatch):
    ctx = finalized
    real = Run._finish
    calls = []

    def crash_after_manifest(run_id, step, rec, passed, root=None):
        if step == A.STEP and not calls:
            calls.append(1)
            raise KeyboardInterrupt
        return real(run_id, step, rec, passed, root=root)

    monkeypatch.setattr(Run, "_finish", crash_after_manifest)
    with pytest.raises(KeyboardInterrupt):
        _accept(ctx)
    written = Rec.record_path(RUN, A.MANIFEST).read_bytes()
    assert _accept(ctx)["passed"]                                       # same bytes: reused
    assert Rec.record_path(RUN, A.MANIFEST).read_bytes() == written


# ----------------------------------------------------------------------
# refusals: the records
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["preflight", "stage2_reuse", "execution_glass_recovery",
                                  "execution_blackbox", T.STEP, "run_record"])
@pytest.mark.parametrize("how", ["missing", "edited"])
def test_a_missing_or_edited_record_is_refused(finalized, name, how):
    ctx = finalized
    p = Rec.record_path(RUN, name)
    if how == "missing":
        p.unlink()
    else:
        p.write_bytes(p.read_bytes().replace(b'"run_id"', b'"run_id" ', 1))
    with pytest.raises(Rec.RecordError):
        _accept(ctx)
    assert not Rec.record_path(RUN, A.STEP).exists()


@pytest.mark.parametrize("name, edit, failed_check", [
    ("stage2_reuse", lambda r: r.update(environment_note="rewritten"),
     "names the earlier ones by their sidecar SHA-256"),
    ("execution_blackbox", lambda r: r.update(environment_note="rewritten"),
     "names the earlier ones by their sidecar SHA-256"),
    ("run_record", lambda r: r["gates"]["A4.4"]["checks"][0].update(passed=False),
     "the run record's A4.3-A4.6: every recorded check passed"),
    ("run_record", lambda r: r["gates"]["A4.6"]["checks"].pop(),
     "A4.6 recomputed now: every check passes, the same checks as the run record"),
    ("run_record", lambda r: r["evidence"].update(stage1={"stub": False}),
     "evidence and the section 7.3 identities are the run record's, value for value"),
    ("run_record", lambda r: r.update(accepted=True),
     "the run record is the regeneration record, not an acceptance"),
    ("run_record", lambda r: r["artifacts"]["stage3.xgb"].update(sha256="0" * 64),
     "one path, size and SHA-256 in every record that names them"),
    ("stage2_reuse", lambda r: r["evidence"].update(acceptance={"x": 1}) if isinstance(
        r["evidence"], dict) else r.update(evidence={"acceptance": {"x": 1}}),
     "A4.2 recomputed evidence is the record's, value for value"),
])
def test_a_consistently_rewritten_record_is_caught(finalized, name, edit, failed_check):
    ctx = finalized
    _rewrite(name, edit)
    _refused(ctx, failed_check)


# ----------------------------------------------------------------------
# refusals: artifacts, provenance, notebooks, code
# ----------------------------------------------------------------------
def test_a_changed_artifact_is_refused(finalized):
    ctx = finalized
    xgb = Rec.require_passed(RUN, "run_record")["artifacts"]["stage3.xgb"]["path"]
    (ctx.root / xgb).write_bytes(b"retrained")
    _refused(ctx, "stage3.xgb: " + xgb + " unchanged since it was produced")


@pytest.mark.parametrize("target, failed_check", [
    ("failed_finalize", "the failed attempts the run's recovery and transition name are preserved"),
    ("failed_glass", "the failed attempts the run's recovery and transition name are preserved"),
    ("executed_copy", "blackbox: the executed copy has its recorded SHA-256"),
])
def test_missing_provenance_is_refused(finalized, target, failed_check):
    ctx = finalized
    failed = Rec.run_dir(RUN) / "failed"
    if target == "failed_finalize":
        next(failed.glob("run_record_*.json")).unlink()
    elif target == "failed_glass":
        next(failed.glob("execution_glass_2*.json")).unlink()
    else:
        next(ctx.out.glob(f"{RUN}_blackbox_executed.ipynb")).unlink()
    _refused(ctx, failed_check)


def test_moved_notebook_copies_are_found_with_copies_dir(finalized, tmp_path):
    ctx = finalized
    moved = tmp_path / "moved"
    ctx.out.rename(moved)
    _refused(ctx, "glass: the input copy has its recorded SHA-256")
    assert _accept(ctx, copies_dir=moved)["passed"]


def test_a_changed_notebook_is_refused(finalized, monkeypatch):
    ctx = finalized
    real = V.truncated_copy
    monkeypatch.setattr(V, "truncated_copy", lambda nb, root=None: (
        real(nb, root=root)[0], {**real(nb, root=root)[1], "truncated_copy_sha256": "f" * 64}))
    _refused(ctx, "the committed notebook's truncated copy, rebuilt now, is the executed one")


def test_code_beyond_the_acceptance_files_is_refused(finalized):
    ctx = finalized
    (ctx.root / "shared" / "code.py").write_text("x = 3\n")
    _refused(ctx, "omitting those files reproduces the run record's code identity")


def test_the_acceptance_files_must_be_present(finalized):
    ctx = finalized
    (ctx.root / A.ACCEPTANCE_FILES[0]).unlink()
    _refused(ctx, "this step's own files are present and new")


def test_a_tampered_transition_is_refused(finalized):
    ctx = finalized

    def widen(r):
        r["code"]["changes"].append({"path": "shared/code.py", "before_lf_sha256": "0" * 64,
                                     "after_lf_sha256": "1" * 64})

    _rewrite(T.STEP, widen)
    with pytest.raises(Run.StepFailed) as err:
        _accept(ctx)
    for failed_check in ("names the earlier ones by their sidecar SHA-256",
                         "under upstream_regeneration code or its tests"):
        assert failed_check in str(err.value), str(err.value)


def test_an_unregistered_change_in_git_status_is_refused(finalized):
    ctx = finalized
    ctx.git.append("?? prototype/bank_pipeline/shared/stage4/new_reader.py")
    _refused(ctx, "A4.6 recomputed now")


# ----------------------------------------------------------------------
# refusals: statements, duplicates, check
# ----------------------------------------------------------------------
@pytest.mark.parametrize("content", [
    None, "not json", json.dumps({"decision": "yes"}),
    json.dumps({**STATEMENTS, "disposition_7_1": "  "}),
    json.dumps({**STATEMENTS, "disposition_7_2": "x" * 2001}),
    json.dumps({**STATEMENTS, "extra": "y"}),
])
def test_bad_statements_are_refused_before_anything_runs(finalized, tmp_path, content):
    ctx = finalized
    path = tmp_path / "bad.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(Rec.RecordError, match="statement"):
        _accept(ctx, statements=path)
    assert not list((Rec.run_dir(RUN) / "failed").glob(f"{A.STEP}_*.json"))


def test_check_refuses_an_edited_manifest_or_acceptance(finalized):
    ctx = finalized
    _accept(ctx)
    m = Rec.record_path(RUN, A.MANIFEST)
    good = m.read_bytes()
    m.write_bytes(good.replace(b'"kind"', b'"kind" ', 1))
    with pytest.raises(P.PinError, match="sidecar"):
        A.check(RUN)
    m.write_bytes(good)
    _rewrite(A.STEP, lambda r: r["decision"].update(accepted=False))
    with pytest.raises(Rec.RecordError, match="not an acceptance"):
        A.check(RUN)


def test_the_cli_runs_the_real_tables(finalized, capsys):
    ctx = finalized
    assert A.main(["accept", "--run-id", RUN, "--sums", "SHA256SUMS.txt",
                   "--statements", str(ctx.statements)]) == 0
    out = capsys.readouterr().out
    assert "OK upstream_acceptance" in out and "A4.7:" in out and f"D7 {SWEEP}: stage" in out
    assert A.main(["check", "--run-id", RUN]) == 0
    assert "stage3.xgb" in capsys.readouterr().out
    assert A.main(["accept", "--run-id", RUN, "--sums", "SHA256SUMS.txt",
                   "--statements", str(ctx.statements)]) == 2              # already accepted


def test_the_acceptance_adds_files_and_changes_none():
    from tests.support.notebooks import BANK_ROOT
    from upstream_regeneration import recovery as Recov
    for rel in A.ACCEPTANCE_FILES:
        assert (BANK_ROOT / rel).is_file(), rel
        assert rel.startswith(T.ALLOWED_PREFIXES)
        assert rel not in T.CHANGES and rel not in Recov.CHANGES
    assert set(A.D7) == {rel for rel in A.D7 if Run.TRACKED_ALIASES.search(rel)}
