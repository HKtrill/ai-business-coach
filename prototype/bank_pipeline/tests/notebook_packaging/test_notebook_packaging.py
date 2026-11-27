"""
PR 42, Step 6B — executed-notebook packaging (``notebook_packaging``).

Synthetic notebooks in a temporary Git repository stand in for the executed copies, which
live outside the repository on the owner's machine. The last tests check the repository's
own manifest against the tracked notebooks and the accepted PR 39 and PR 42 records.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import stat
import subprocess
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import pytest

import notebook_packaging as NP
from notebook_packaging import packaging as P

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")

BANK = Path(__file__).resolve().parents[2]
REPO = BANK.parents[1]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def notebook(run_id="r1", *, cells=3, error=False, skip=None, empty_cell=False, outputs=True, indent=1,
             ids=True) -> bytes:
    """An executed notebook: Markdown, code cells with streams, an HTML table and a PNG plot."""
    cs = [{"cell_type": "markdown", "metadata": {}, "source": ["# Title\n", "text"]}]
    for i in range(cells):
        outs = []
        if outputs:
            outs = [{"output_type": "stream", "name": "stdout", "text": [f"run {run_id} cell {i}\n"]},
                    {"output_type": "display_data", "metadata": {},
                     "data": {"text/html": ["<table><tr><td>1</td></tr></table>"], "text/plain": ["t"]}},
                    {"output_type": "display_data", "metadata": {},
                     "data": {"image/png": "iVBORw0KGgo=", "text/plain": ["<Figure>"]}}]
        if error and i == cells - 1:
            outs.append({"output_type": "error", "ename": "ValueError", "evalue": "x", "traceback": ["x"]})
        cs.append({"cell_type": "code", "execution_count": None if skip == i else i + 1,
                   "metadata": {"tags": ["t"]}, "outputs": [] if skip == i else outs,
                   "source": [f"# CELL {i}\n", "x = 1"]})
    if empty_cell:
        cs.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []})
    for k, c in enumerate(cs):
        if ids:
            c["id"] = f"c{k}"
    nb = {"cells": cs, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                   "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5 if ids else 4}
    return (json.dumps(nb, indent=indent, ensure_ascii=False) + "\n").encode("utf-8")


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(repo), *args],
                          check=True, capture_output=True).stdout


def _record(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(content, indent=1) + "\n").encode("utf-8")
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_text(_sha(raw) + "\n", encoding="ascii")


@pytest.fixture
def world(tmp_path):
    """A repository with three tracked notebooks, accepted executions outside it, and a manifest."""
    repo, exe, out = tmp_path / "repo", tmp_path / "executed", tmp_path / "out"
    bank = repo / "prototype" / "bank_pipeline"
    (bank / "alpha").mkdir(parents=True)
    (bank / "beta").mkdir()
    (repo / "legacy").mkdir()
    exe.mkdir()
    tracked_alpha = notebook("tracked", indent=2)
    (bank / "alpha" / "alpha.ipynb").write_bytes(tracked_alpha)
    (bank / "beta" / "beta.ipynb").write_bytes(notebook("blob"))
    (repo / "legacy" / "old.ipynb").write_bytes(notebook(outputs=False))
    files = {"r1_alpha_executed.ipynb": notebook("r1"), "r2_alpha_executed.ipynb": notebook("r2", cells=4),
             "r9_alpha_executed.ipynb": notebook("r9", error=True)}
    for name, data in files.items():
        (exe / name).write_bytes(data)
    _record(bank / "logs" / "r1" / "rec.json", {"run": {"run_id": "r1"}, "copy": {"sha256": _sha(files["r1_alpha_executed.ipynb"])},
                                               "passed": True})
    _record(bank / "logs" / "r2" / "rec.json", {"run": {"run_id": "r2"}})
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    blob = _git(repo, "rev-parse", "HEAD:prototype/bank_pipeline/beta/beta.ipynb").decode().strip()
    beta_lf = _sha(notebook("blob"))
    _record(bank / "logs" / "pin" / "rec.json", {"pin": {"blob": blob}})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "records")
    A, Bn = "prototype/bank_pipeline/alpha/alpha.ipynb", "prototype/bank_pipeline/beta/beta.ipynb"
    man = {"format": P.MANIFEST_FORMAT,
           "notebooks": {A: {"current": "r1/alpha"}, Bn: {"current": "pin/beta"},
                         "legacy/old.ipynb": {"current": None, "reason": "no accepted execution"}},
           "executions": {
               "r1/alpha": {"notebook": A, "run_id": "r1", "source": {"file": "r1_alpha_executed.ipynb"},
                            "acceptance": {"record": "prototype/bank_pipeline/logs/r1/rec.json",
                                           "sha256": "copy/sha256", "require": {"passed": True}}},
               "r2/alpha": {"notebook": A, "run_id": "r2", "source": {"file": "r2_alpha_executed.ipynb"},
                            "acceptance": {"record": "prototype/bank_pipeline/logs/r2/rec.json",
                                           "require": {"run/run_id": "r2"}, "run_id_in_outputs": True}},
               "pin/beta": {"notebook": Bn, "run_id": "pin", "source": {"git_blob": blob, "sha256_lf": beta_lf},
                            "acceptance": {"record": "prototype/bank_pipeline/logs/pin/rec.json",
                                           "require": {"pin/blob": blob}}}},
           "snapshots": {"prx": {"directory": "prx_executed_notebooks", "executions": ["r1/alpha", "r2/alpha"]}}}
    man_path = tmp_path / "accepted.json"
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")   # the real layout
    return {"repo": repo, "bank": bank, "exe": exe, "out": out, "man": man, "man_path": man_path,
            "files": files, "A": A, "B": Bn, "tracked_alpha": tracked_alpha}


def _rewrite(w, mutate):
    mutate(w["man"])
    w["man_path"].write_text(json.dumps(w["man"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _build(w, **kw):
    return NP.build_project(w["man_path"], w["repo"], [w["exe"]], w["out"], **kw)


def _snapshot(w, **kw):
    return NP.build_snapshot(w["man_path"], "prx", w["repo"], [w["exe"]], w["out"], **kw)


def _state(*roots):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for r in roots for p in sorted(Path(r).rglob("*"))
            if p.is_file() and ".git" not in p.parts}


# ======================================================================
# building
# ======================================================================
def test_the_project_archive_is_flat_and_preserves_repository_paths_in_manifest(world):
    rep = _build(world)
    with zipfile.ZipFile(rep["archive"]) as zf:
        names = sorted(zf.namelist())
        assert names == sorted(["alpha.ipynb", "beta.ipynb", "MANIFEST.json", "README.md"])
        assert zf.read("alpha.ipynb") == world["files"]["r1_alpha_executed.ipynb"]         # byte for byte
        assert zf.read("beta.ipynb") == notebook("blob")                                  # the pinned blob
        man = json.loads(zf.read("MANIFEST.json"))
    assert man["kind"] == "project" and [e["run_id"] for e in man["notebooks"]] == ["r1", "pin"]
    assert [(e["archive_path"], e["notebook"]) for e in man["notebooks"]] == [
        ("alpha.ipynb", world["A"]), ("beta.ipynb", world["B"])]
    assert man["not_included"] == [{"notebook": "legacy/old.ipynb", "reason": "no accepted execution"}]
    assert "legacy/old.ipynb" in zipfile.ZipFile(rep["archive"]).read("README.md").decode()


def test_the_snapshot_folder_and_zip_keep_run_ids_in_their_file_names(world):
    rep = _snapshot(world)
    folder = Path(rep["directory"])
    assert sorted(p.name for p in folder.iterdir()) == ["MANIFEST.json", "README.md", "r1_alpha_executed.ipynb",
                                                       "r2_alpha_executed.ipynb"]
    with zipfile.ZipFile(rep["archive"]) as zf:
        assert sorted(zf.namelist()) == sorted(f"prx_executed_notebooks/{n}" for n in
                                               ["MANIFEST.json", "README.md", "r1_alpha_executed.ipynb",
                                                "r2_alpha_executed.ipynb"])
        assert zf.read("prx_executed_notebooks/r2_alpha_executed.ipynb") == world["files"]["r2_alpha_executed.ipynb"]
    assert Path(rep["archive"]).name == "prx_executed_notebooks.zip"


def test_saved_outputs_execution_counts_tables_and_plots_are_preserved(world):
    rep = _build(world)
    entry = next(e for e in rep["notebooks"] if e["run_id"] == "r1")
    s = entry["summary"]
    assert (s["code_cells"], s["executed_cells"], s["tables"], s["images"], s["streams"]) == (3, 3, 3, 3, 3)
    with zipfile.ZipFile(rep["archive"]) as zf:
        nb = json.loads(zf.read("alpha.ipynb"))
    assert [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"] == [1, 2, 3]
    assert nb["metadata"]["kernelspec"]["name"] == "python3" and nb["cells"][1]["metadata"] == {"tags": ["t"]}


# ======================================================================
# rejection
# ======================================================================
def test_missing_or_invalid_inputs_are_rejected(world):
    (world["exe"] / "r1_alpha_executed.ipynb").unlink()
    with pytest.raises(NP.PackagingError, match="not found"):
        _build(world)
    (world["exe"] / "r1_alpha_executed.ipynb").write_text("{not json", encoding="utf-8")
    with pytest.raises(NP.PackagingError, match="not a readable notebook"):
        _build(world)
    (world["exe"] / "r1_alpha_executed.ipynb").write_text(json.dumps({"nbformat": 3, "cells": []}), encoding="utf-8")
    with pytest.raises(NP.PackagingError, match="nbformat 4"):
        _build(world)
    assert not (world["out"] / "executed_notebooks.zip").exists()


@pytest.mark.parametrize("data, reason", [
    (notebook("r2", cells=4, error=True), "failed execution"),
    (notebook("r2", cells=4, skip=2), "partial execution"),
    (notebook("r2", cells=4, outputs=False), "no saved outputs"),
])
def test_failed_partial_and_unexecuted_copies_are_rejected(world, data, reason):
    (world["exe"] / "r2_alpha_executed.ipynb").write_bytes(data)
    with pytest.raises(NP.PackagingError, match=reason):
        _snapshot(world)
    assert not (world["out"] / "prx_executed_notebooks").exists()


def test_an_empty_code_cell_that_never_ran_is_not_a_partial_execution():
    s = NP.notebook_summary(notebook(empty_cell=True))
    assert s["code_cells"] == 4 and s["executed_cells"] == 3 and not P.completeness_problems(s)


def test_unapproved_copies_are_rejected(world):
    # a different file under an accepted name: not the recorded executed copy
    (world["exe"] / "r1_alpha_executed.ipynb").write_bytes(notebook("r1", cells=5))
    with pytest.raises(NP.PackagingError, match="not the accepted executed copy"):
        _build(world)
    (world["exe"] / "r1_alpha_executed.ipynb").write_bytes(world["files"]["r1_alpha_executed.ipynb"])
    # a run id that the outputs do not carry
    (world["exe"] / "r2_alpha_executed.ipynb").write_bytes(notebook("r7", cells=4))
    with pytest.raises(NP.PackagingError, match="does not appear in the notebook's saved outputs"):
        _snapshot(world)
    (world["exe"] / "r2_alpha_executed.ipynb").write_bytes(world["files"]["r2_alpha_executed.ipynb"])
    # a record that does not accept it, and a record changed after its sidecar was written
    rec = world["bank"] / "logs" / "r1" / "rec.json"
    _record(rec, {**json.loads(rec.read_text()), "passed": False})
    with pytest.raises(NP.PackagingError, match="passed is False"):
        _build(world)
    rec.write_bytes(rec.read_bytes().replace(b"false", b"true"))
    with pytest.raises(NP.PackagingError, match="sidecar"):
        _build(world)


def test_a_manifest_sha256_pin_must_match(world):
    _rewrite(world, lambda m: m["executions"]["r2/alpha"].update(sha256="0" * 64))
    with pytest.raises(NP.PackagingError, match="manifest's pin"):
        _snapshot(world)


def test_files_not_listed_in_the_manifest_are_never_packaged(world):
    rep = NP.inventory(NP.load_manifest(world["man_path"]), world["repo"], [world["exe"]])
    assert rep["unlisted_files"] == [str(world["exe"] / "r9_alpha_executed.ipynb")]
    names = zipfile.ZipFile(_build(world)["archive"]).namelist() + zipfile.ZipFile(_snapshot(world)["archive"]).namelist()
    assert not any("r9" in n for n in names)


def test_every_tracked_notebook_must_be_listed_with_an_execution_or_a_reason(world):
    _rewrite(world, lambda m: m["notebooks"].pop("legacy/old.ipynb"))
    with pytest.raises(NP.PackagingError, match="legacy/old.ipynb is not listed"):
        _build(world)
    _rewrite(world, lambda m: m["notebooks"].update({"legacy/old.ipynb": {"current": None}}))
    with pytest.raises(NP.PackagingError, match="no current execution and no reason"):
        _build(world)
    _rewrite(world, lambda m: m["notebooks"].update({"legacy/old.ipynb": {"current": None, "reason": "x"},
                                                     "legacy/gone.ipynb": {"current": None, "reason": "x"}}))
    with pytest.raises(NP.PackagingError, match="not a tracked notebook"):
        _build(world)


def test_duplicate_and_colliding_archive_paths_are_detected(world):
    _rewrite(world, lambda m: m["notebooks"].update({world["A"].upper(): {"current": None, "reason": "x"}}))
    with pytest.raises(NP.PackagingError, match="collide on Windows"):
        NP.load_manifest(world["man_path"])
    _rewrite(world, lambda m: (m["notebooks"].pop(world["A"].upper()),
                               m["executions"].update({"r1b/alpha": {**m["executions"]["r1/alpha"]}}),
                               m["snapshots"]["prx"]["executions"].append("r1b/alpha")))
    with pytest.raises(NP.PackagingError, match="collide"):
        _snapshot(world)


@pytest.mark.parametrize("name", ["alpha.ipynb", "ALPHA.ipynb"])
def test_flat_project_archive_refuses_duplicate_basenames(world, name):
    old = world["B"]
    new = str(PurePosixPath(old).with_name(name))
    _git(world["repo"], "mv", old, new)
    def change(m):
        m["notebooks"][new] = m["notebooks"].pop(old)
        m["executions"]["pin/beta"]["notebook"] = new
    _rewrite(world, change)
    with pytest.raises(NP.PackagingError, match="collide"):
        _build(world)
    assert not (world["out"] / "executed_notebooks.zip").exists()


def test_windows_unsafe_manifest_paths_are_rejected(world):
    _rewrite(world, lambda m: m["executions"]["r1/alpha"]["source"].update(file="sub\\r1.ipynb"))
    with pytest.raises(NP.PackagingError, match="bare .ipynb file name"):
        NP.load_manifest(world["man_path"])
    _rewrite(world, lambda m: (m["executions"]["r1/alpha"]["source"].update(file="r1_alpha_executed.ipynb"),
                               m["executions"]["r1/alpha"]["acceptance"].update(record="C:\\logs\\rec.json")))
    with pytest.raises(NP.PackagingError, match="forward slashes"):
        NP.load_manifest(world["man_path"])


# ======================================================================
# non-destructive, rebuildable, outside the repository
# ======================================================================
def test_packaging_and_verification_leave_every_original_unchanged(world):
    for p in world["exe"].iterdir():
        p.chmod(stat.S_IREAD)                       # read-only originals (a write fails on Windows)
    before = _state(world["exe"], world["repo"])
    project, snap = _build(world), _snapshot(world)
    assert NP.verify_archive(project["archive"], repo=world["repo"], man_path=world["man_path"],
                             executed_dirs=[world["exe"]])["passed"]
    assert NP.verify_archive(snap["archive"], repo=world["repo"], man_path=world["man_path"],
                             executed_dirs=[world["exe"]])["passed"]
    assert _state(world["exe"], world["repo"]) == before
    for p in world["exe"].iterdir():
        p.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_the_project_archive_is_rebuilt_only_on_request_and_reports_what_changed(world):
    _build(world)
    with pytest.raises(NP.PackagingError, match="pass --replace"):
        _build(world)
    _rewrite(world, lambda m: m["notebooks"][world["A"]].update(current="r2/alpha"))
    rep = _build(world, replace=True)
    assert rep["changes"] == {"added": [], "removed": [], "replaced": ["alpha.ipynb"]}
    with zipfile.ZipFile(rep["archive"]) as zf:
        assert zf.read("alpha.ipynb") == world["files"]["r2_alpha_executed.ipynb"]


def test_a_snapshot_never_overwrites_a_different_copy_or_mixes_in_stray_files(world):
    rep = _snapshot(world)
    folder = Path(rep["directory"])
    Path(rep["archive"]).unlink()
    _snapshot(world)                                # identical copies: rebuilt
    Path(rep["archive"]).unlink()
    (folder / "r1_alpha_executed.ipynb").write_bytes(b"{}")
    with pytest.raises(NP.PackagingError, match="differ from the accepted copies"):
        _snapshot(world)
    (folder / "r1_alpha_executed.ipynb").write_bytes(world["files"]["r1_alpha_executed.ipynb"])
    (folder / "notes.txt").write_text("x")
    with pytest.raises(NP.PackagingError, match="does not produce"):
        _snapshot(world)


def test_archives_are_written_outside_the_repository(world):
    with pytest.raises(NP.PackagingError, match="inside the repository"):
        NP.build_project(world["man_path"], world["repo"], [world["exe"]], world["repo"] / "zips")


def test_member_names_are_forward_slash_paths(world):
    names = zipfile.ZipFile(_build(world)["archive"]).namelist()
    assert all("\\" not in n and not n.startswith("/") for n in names)


# ======================================================================
# verification
# ======================================================================
def _tamper(src: Path, dst: Path, name: str, data: bytes = None, extra: bool = False):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, data if (n == name and data is not None) else zin.read(n))
        if extra:
            zout.writestr("extra.txt", b"x")


def test_verification_extracts_and_detects_changed_extra_or_unreadable_members(world, tmp_path):
    rep = _build(world)
    ok = NP.verify_archive(rep["archive"], repo=world["repo"], man_path=world["man_path"],
                           executed_dirs=[world["exe"]], extract_to=tmp_path / "x")
    assert ok["passed"] and (tmp_path / "x" / "alpha.ipynb").is_file()
    bad = tmp_path / "bad.zip"
    _tamper(Path(rep["archive"]), bad, "alpha.ipynb", notebook("r1", cells=2))
    res = NP.verify_archive(bad, repo=world["repo"], man_path=world["man_path"])
    assert not res["passed"] and any("SHA-256" in c["check"] and not c["passed"] for c in res["checks"])
    _tamper(Path(rep["archive"]), bad, "alpha.ipynb", extra=True)
    res = NP.verify_archive(bad, repo=world["repo"], man_path=world["man_path"])
    assert not res["passed"] and any("exactly" in c["check"] and not c["passed"] for c in res["checks"])
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    assert not NP.verify_archive(rep["archive"], repo=world["repo"], man_path=world["man_path"],
                                 extract_to=tmp_path / "busy")["passed"]


def test_verification_with_the_sources_detects_a_changed_original(world):
    rep = _snapshot(world)
    (world["exe"] / "r2_alpha_executed.ipynb").write_bytes(notebook("r2", cells=6))
    res = NP.verify_archive(rep["archive"], repo=world["repo"], man_path=world["man_path"],
                            executed_dirs=[world["exe"]])
    assert not res["passed"]
    assert [c["check"] for c in res["checks"] if not c["passed"]] == [
        "prx_executed_notebooks/r2_alpha_executed.ipynb: equal to its accepted source, which is unchanged"]


def test_a_git_blob_source_falls_back_to_an_identical_working_copy_only(world, monkeypatch):
    monkeypatch.setattr(P, "_git", lambda repo, *a: (None if a[:1] == ("cat-file",) else
                                                     subprocess.run(["git", "-C", str(repo), *a],
                                                                    capture_output=True).stdout))
    man = NP.load_manifest(world["man_path"])
    r = P.resolve("pin/beta", man, world["repo"], [world["exe"]])
    assert r.data == notebook("blob") and "working copy" in r.origin
    (world["repo"] / world["B"]).write_bytes(notebook("blob", outputs=False))
    with pytest.raises(NP.PackagingError, match="not readable and the working copy is not it"):
        P.resolve("pin/beta", man, world["repo"], [world["exe"]])


# --- verification refuses unsafe or colliding member paths before extracting --------------------
def _with_member(src: Path, dst: Path, name: str, data: bytes = b"x") -> None:
    """Copy an archive and add a member stored under exactly ``name`` (zipfile would rewrite "\\")."""
    info = zipfile.ZipInfo("placeholder")
    info.filename = info.orig_filename = name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # zipfile warns about a duplicate name
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
            for n in zin.namelist():
                zout.writestr(n, zin.read(n))
            zout.writestr(info, data)


@pytest.mark.parametrize("member", ["../evil.ipynb", "/abs.ipynb", "sub\\evil.ipynb", "C:/evil.ipynb",
                                    "sub/CON.ipynb", "sub/a:b.ipynb", "sub/trailing. ", "UPPER", "DUPLICATE"])
def test_verification_refuses_unsafe_or_colliding_member_paths_before_extracting(world, tmp_path, member,
                                                                                  monkeypatch):
    rep = _build(world)
    name = {"UPPER": "ALPHA.IPYNB", "DUPLICATE": "alpha.ipynb"}.get(member, member)
    bad, dest = tmp_path / "bad.zip", tmp_path / "x" / "y"
    _with_member(Path(rep["archive"]), bad, name)
    touched = []
    monkeypatch.setattr(P, "_extract", lambda *a: touched.append(a))
    monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: touched.append("crc"))
    res = NP.verify_archive(bad, repo=world["repo"], man_path=world["man_path"], extract_to=dest)
    assert not res["passed"] and touched == []
    last = res["checks"][-1]
    assert not last["passed"] and "before extraction" in last["check"]
    assert [c for c in res["checks"] if not c["passed"]] == [last]          # refused at once, nothing after
    assert not (tmp_path / "x").exists() and not (tmp_path / "evil.ipynb").exists()


def test_verification_refuses_an_unsafe_path_listed_in_the_archive_manifest(world, tmp_path):
    rep = _build(world)
    with zipfile.ZipFile(rep["archive"]) as zf:
        blob = json.loads(zf.read("MANIFEST.json"))
    blob["notebooks"][0]["archive_path"] = "../outside.ipynb"
    bad = tmp_path / "bad.zip"
    _tamper(Path(rep["archive"]), bad, "MANIFEST.json", json.dumps(blob).encode())
    res = NP.verify_archive(bad, repo=world["repo"], man_path=world["man_path"], extract_to=tmp_path / "x")
    assert not res["passed"] and "MANIFEST.json lists" in res["checks"][-1]["check"] and not (tmp_path / "x").exists()


def test_extraction_never_writes_outside_its_folder(world, tmp_path):
    rep = _build(world)
    with zipfile.ZipFile(rep["archive"]) as zf, pytest.raises(NP.PackagingError, match="outside"):
        P._extract(zf, ["../escaped.ipynb"], tmp_path / "x")
    assert not (tmp_path / "escaped.ipynb").exists()


# ======================================================================
# clearing tracked notebooks: separate and explicit
# ======================================================================
def _clear(w, notebooks, archive, keep, **kw):
    return NP.clear_outputs(w["repo"], notebooks, archive, preserve_dir=keep, man_path=w["man_path"], **kw)


def _kept(keep: Path, rel: str, raw: bytes) -> Path:
    p = PurePosixPath(rel)
    return keep / p.parent / f"{p.stem}_{_sha(raw)[:12]}_tracked_outputs.ipynb"


def test_building_and_verifying_never_clear_a_tracked_notebook(world):
    rep = _build(world)
    NP.verify_archive(rep["archive"], repo=world["repo"], man_path=world["man_path"])
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"]


def test_clearing_needs_a_verified_archive_that_accounts_for_the_notebook(world, tmp_path):
    rep, snap, keep = _build(world), _snapshot(world), tmp_path / "keep"
    with pytest.raises(NP.PackagingError, match="not accounted for"):
        _clear(world, [world["B"]], snap["archive"], keep, apply=True)    # the snapshot holds no beta copy
    with pytest.raises(NP.PackagingError, match="never cleared"):
        _clear(world, [str(world["exe"] / "r1_alpha_executed.ipynb")], rep["archive"], keep)
    bad = tmp_path / "bad.zip"
    _tamper(Path(rep["archive"]), bad, "alpha.ipynb", notebook("r1", cells=2))
    with pytest.raises(NP.PackagingError, match="does not verify"):
        _clear(world, [world["A"]], bad, keep, apply=True)
    dry = _clear(world, [world["A"]], rep["archive"], keep)
    row = dry["notebooks"][0]
    assert row["written"] is False and row["outputs"] == 9
    assert row["preserved_to"] == str(_kept(keep.resolve(), world["A"], world["tracked_alpha"]))
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"]
    assert (world["repo"] / world["B"]).read_bytes() == notebook("blob") and not keep.exists()


def test_the_current_outputs_are_preserved_before_clearing(world, tmp_path):
    rep, keep = _build(world), tmp_path / "keep"
    res = _clear(world, [world["A"], world["B"]], rep["archive"], keep, apply=True)
    assert [r["written"] for r in res["notebooks"]] == [True, True]
    index = json.loads((keep / "PRESERVED.json").read_text(encoding="utf-8"))
    assert index["format"] == P.PRESERVED_FORMAT
    for rel, raw in ((world["A"], world["tracked_alpha"]), (world["B"], notebook("blob"))):
        kept = _kept(keep, rel, raw)
        assert kept.read_bytes() == raw                                     # byte-for-byte, outputs included
        assert NP.notebook_summary(kept.read_bytes())["outputs"] == NP.notebook_summary(raw)["outputs"] > 0
        entry = next(e for e in index["entries"] if e["notebook"] == rel)
        assert entry["sha256"] == _sha(raw) and entry["file"] == kept.relative_to(keep).as_posix()
        assert entry["equal_to_HEAD"] is True and entry["outputs"] == NP.notebook_summary(raw)["outputs"]
        assert NP.notebook_summary((world["repo"] / rel).read_bytes())["outputs"] == 0
    before = _state(keep)
    again = _clear(world, [world["A"], world["B"]], rep["archive"], keep, apply=True)
    assert all(r["already_clear"] and not r["written"] for r in again["notebooks"]) and _state(keep) == before


@pytest.mark.parametrize("failure", ["the copy cannot be written", "the copy does not read back"])
def test_a_failed_preservation_clears_nothing(world, tmp_path, monkeypatch, failure):
    rep, keep = _build(world), tmp_path / "keep"
    real = P._write_atomic

    def write(path, data):
        name = Path(path).name
        if name.startswith("beta_") and name.endswith("_tracked_outputs.ipynb"):    # beta's preserved copy
            if failure == "the copy cannot be written":
                raise OSError("disk full")
            data = data[:-20]
        real(path, data)

    monkeypatch.setattr(P, "_write_atomic", write)
    with pytest.raises((OSError, NP.PackagingError)):
        _clear(world, [world["A"], world["B"]], rep["archive"], keep, apply=True)
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"]       # alpha was preserved,
    assert (world["repo"] / world["B"]).read_bytes() == notebook("blob")             # but nothing is cleared
    assert not (keep / "PRESERVED.json").exists()


def test_a_notebook_changed_after_preservation_is_not_cleared(world, tmp_path, monkeypatch):
    rep, keep, path = _build(world), tmp_path / "keep", world["repo"] / world["A"]
    real = P._record_preservation
    monkeypatch.setattr(P, "_record_preservation", lambda d, e: (real(d, e), path.write_bytes(path.read_bytes() + b" ")))
    with pytest.raises(NP.PackagingError, match="changed after it was preserved"):
        _clear(world, [world["B"], world["A"]], rep["archive"], keep, apply=True)
    assert (world["repo"] / world["B"]).read_bytes() == notebook("blob")
    assert path.read_bytes() == world["tracked_alpha"] + b" "


def test_the_preserve_folder_is_outside_the_repository_and_never_overwritten(world, tmp_path):
    rep = _build(world)
    with pytest.raises(NP.PackagingError, match="inside the repository"):
        _clear(world, [world["A"]], rep["archive"], world["repo"] / "keep", apply=True)
    (tmp_path / "afile").write_text("x")
    with pytest.raises(NP.PackagingError, match="not a folder"):
        _clear(world, [world["A"]], rep["archive"], tmp_path / "afile", apply=True)
    keep = tmp_path / "keep"
    kept = _kept(keep, world["A"], world["tracked_alpha"])
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"something else")
    with pytest.raises(NP.PackagingError, match="never overwritten"):
        _clear(world, [world["A"]], rep["archive"], keep, apply=True)
    assert kept.read_bytes() == b"something else"
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"]
    assert not (world["repo"] / "keep").exists()


def test_an_output_free_notebook_is_a_harmless_no_op(world, tmp_path):
    for rel, raw in ((world["A"], world["tracked_alpha"]), (world["B"], notebook("blob"))):
        nb = NP.content_without_outputs(json.loads(raw))
        (world["repo"] / rel).write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode())
    snap, keep = _snapshot(world), tmp_path / "keep"           # the snapshot does not hold beta: no matter
    before = _state(world["repo"])
    res = _clear(world, [world["A"], world["B"]], snap["archive"], keep, apply=True)
    assert all(r["already_clear"] and not r["written"] and r["preserved_to"] is None for r in res["notebooks"])
    assert _state(world["repo"]) == before and not keep.exists()


def test_legacy_notebooks_without_an_accepted_execution_are_kept_separate(world, tmp_path):
    rep, keep, old = _build(world), tmp_path / "keep", world["repo"] / "legacy" / "old.ipynb"
    for raw in (old.read_bytes(), notebook("legacy")):                       # output-free, or not
        old.write_bytes(raw)
        with pytest.raises(NP.PackagingError, match="kept separate"):
            _clear(world, ["legacy/old.ipynb", world["A"]], rep["archive"], keep, apply=True)
        assert old.read_bytes() == raw
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"] and not keep.exists()


def test_clearing_keeps_cells_ids_order_metadata_layout_and_line_endings(world, tmp_path):
    path, keep = world["repo"] / world["A"], tmp_path / "keep"
    crlf = world["tracked_alpha"].replace(b"\n", b"\r\n")
    path.write_bytes(crlf)                                                    # a CRLF checkout
    rep = _build(world)
    res = _clear(world, [world["A"]], rep["archive"], keep, apply=True)
    row = res["notebooks"][0]
    assert row["written"] and all(row["checks"].values()) and row["equal_to_HEAD_except_outputs"]
    assert _kept(keep, world["A"], crlf).read_bytes() == crlf                 # preserved as it was on disk
    raw = path.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    before = json.loads(world["tracked_alpha"])
    after = json.loads(raw.replace(b"\r\n", b"\n"))
    assert after == NP.content_without_outputs(before)
    assert [c["id"] for c in after["cells"]] == [c["id"] for c in before["cells"]]
    assert [c["source"] for c in after["cells"]] == [c["source"] for c in before["cells"]]
    assert after["metadata"] == before["metadata"] and after["nbformat_minor"] == before["nbformat_minor"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in after["cells"] if c["cell_type"] == "code")
    assert raw.replace(b"\r\n", b"\n").decode() == json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    for name, data in world["files"].items():                              # executed copies untouched
        assert (world["exe"] / name).read_bytes() == data


# ======================================================================
# command line
# ======================================================================
def test_the_command_line_builds_verifies_and_clears_only_when_asked(world, tmp_path, capsys):
    base = ["--repo", str(world["repo"]), "--manifest", str(world["man_path"])]
    keep = str(tmp_path / "keep")
    assert NP.main(base + ["inventory", "--executed-dir", str(world["exe"])]) == 0
    assert NP.main(base + ["build", "--executed-dir", str(world["exe"]), "--out-dir", str(world["out"])]) == 0
    assert NP.main(base + ["snapshot", "prx", "--executed-dir", str(world["exe"]), "--out-dir", str(world["out"])]) == 0
    archive = str(world["out"] / "executed_notebooks.zip")
    assert NP.main(base + ["verify", archive, "--executed-dir", str(world["exe"])]) == 0
    with pytest.raises(SystemExit):                                         # --preserve-dir is required
        NP.main(base + ["clear-outputs", "--verified-archive", archive, "--notebook", world["A"]])
    clear = base + ["clear-outputs", "--verified-archive", archive, "--preserve-dir", keep, "--notebook", world["A"]]
    assert NP.main(clear) == 0
    assert "dry run" in capsys.readouterr().out
    assert (world["repo"] / world["A"]).read_bytes() == world["tracked_alpha"]
    assert NP.main(clear + ["--apply"]) == 0
    out = capsys.readouterr().out
    assert "preserved in" in out and "applied" in out
    assert _kept(Path(keep), world["A"], world["tracked_alpha"]).read_bytes() == world["tracked_alpha"]
    assert NP.main(clear + ["--apply"]) == 0 and "no-op" in capsys.readouterr().out
    assert NP.main(base + ["build", "--executed-dir", str(world["exe"]), "--out-dir", str(world["out"])]) == 1


# ======================================================================
# registering a newly accepted run, and excluding an unpreserved copy (PR 42, Step 6C)
# ======================================================================
def _new_run(w, run="r3", *, in_outputs=True, record=True):
    """An executed copy and a record for a new run of r2's kind."""
    (w["exe"] / f"{run}_alpha_executed.ipynb").write_bytes(notebook(run if in_outputs else "someone else"))
    if record:
        _record(w["bank"] / "logs" / run / "rec.json", {"run": {"run_id": run}})


def _register(w, run="r3", **kw):
    kw.setdefault("description", "a later run")
    return NP.register_execution(w["man_path"], "r2/alpha", run, w["repo"], [w["exe"]], **kw)


def test_a_new_run_is_registered_from_an_accepted_one_only_when_it_is_accepted(world):
    _new_run(world)
    before = world["man_path"].read_bytes()
    dry = _register(world, current=True, snapshots=["prx"])
    assert dry["execution"] == "r3/alpha" and not dry["applied"] and world["man_path"].read_bytes() == before
    rep = _register(world, current=True, snapshots=["prx"], apply=True)
    man = NP.load_manifest(world["man_path"])
    e = man["executions"]["r3/alpha"]
    assert e["source"] == {"file": "r3_alpha_executed.ipynb"} and e["run_id"] == "r3" and e["sha256"] is None
    assert e["acceptance"] == {"record": "prototype/bank_pipeline/logs/r3/rec.json", "require": {"run/run_id": "r3"},
                               "run_id_in_outputs": True}
    assert man["notebooks"][world["A"]]["current"] == "r3/alpha" and rep["sha256"] == _sha(notebook("r3"))
    assert man["snapshots"]["prx"]["executions"] == ["r1/alpha", "r2/alpha", "r3/alpha"]
    assert world["man_path"].read_text(encoding="utf-8") == json.dumps(man, indent=1, ensure_ascii=False) + "\n"
    snap = _snapshot(world)
    assert [e["run_id"] for e in snap["notebooks"]] == ["r1", "r2", "r3"]
    assert NP.verify_archive(snap["archive"], repo=world["repo"], man_path=world["man_path"],
                             executed_dirs=[world["exe"]])["passed"]


@pytest.mark.parametrize("case, match", [
    ("no copy", "not found"),
    ("the copy names another run", "does not appear in the notebook's saved outputs"),
    ("no record", "acceptance record .* is missing"),
    ("a changed record", "does not match its .sha256 sidecar"),
    ("the record is another run's", "run/run_id is 'r9'"),
    ("a failed execution", "error output"),
])
def test_a_run_that_is_not_accepted_is_never_registered(world, case, match):
    if case != "no copy":
        _new_run(world, in_outputs=case != "the copy names another run", record=case != "no record")
    rec = world["bank"] / "logs" / "r3" / "rec.json"
    if case == "a changed record":
        rec.write_text('{"run": {"run_id": "r3"}, "x": 1}', encoding="utf-8")
    if case == "the record is another run's":
        _record(rec, {"run": {"run_id": "r9"}})
    if case == "a failed execution":
        (world["exe"] / "r3_alpha_executed.ipynb").write_bytes(notebook("r3", error=True))
    before = world["man_path"].read_bytes()
    with pytest.raises(NP.PackagingError, match=match):
        _register(world, current=True, snapshots=["prx"], apply=True)
    assert world["man_path"].read_bytes() == before


def test_registering_refuses_a_listed_run_a_pinned_template_or_a_missing_description(world):
    _new_run(world, "r2")
    with pytest.raises(NP.PackagingError, match="not a new"):
        _register(world, "r2", apply=True)
    _new_run(world)
    with pytest.raises(NP.PackagingError, match="description"):
        _register(world, description=" ", apply=True)
    with pytest.raises(NP.PackagingError, match="unpinned"):
        _rewrite(world, lambda m: m["executions"]["r2/alpha"].update(sha256=_sha(b"x")))
        _register(world, apply=True)
    with pytest.raises(NP.PackagingError, match="template"):
        NP.register_execution(world["man_path"], "nope/alpha", "r3", world["repo"], [world["exe"]],
                              description="x")


def test_registering_keeps_a_crlf_manifest_crlf(world):
    _new_run(world)
    # Path.write_text already wrote CRLF on Windows: normalise to LF first, or each line would end in CR CR LF.
    crlf = P.lf(world["man_path"].read_bytes()).replace(b"\n", b"\r\n")
    world["man_path"].write_bytes(crlf)
    assert b"\n" not in crlf.replace(b"\r\n", b"") and b"\r\r" not in crlf          # a clean CRLF manifest
    _register(world, apply=True)
    raw = world["man_path"].read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert "r3/alpha" in NP.load_manifest(world["man_path"])["executions"]


def test_an_unavailable_snapshot_member_is_excluded_with_its_reason_and_keeps_its_records(world):
    (world["exe"] / "r1_alpha_executed.ipynb").unlink()
    _rewrite(world, lambda m: m["notebooks"][world["A"]].update(current="r2/alpha"))
    before = world["man_path"].read_bytes()
    dry = NP.exclude_execution(world["man_path"], "prx", "r1/alpha", "its executed copy was not kept",
                               world["repo"], [world["exe"]])
    assert not dry["applied"] and any("not found" in p for p in dry["unavailable"])
    assert world["man_path"].read_bytes() == before
    NP.exclude_execution(world["man_path"], "prx", "r1/alpha", "its executed copy was not kept", world["repo"],
                         [world["exe"]], apply=True)
    man = NP.load_manifest(world["man_path"])
    assert man["snapshots"]["prx"]["executions"] == ["r2/alpha"]
    assert man["snapshots"]["prx"]["excluded"] == {"r1/alpha": "its executed copy was not kept"}
    assert "r1/alpha" in man["executions"]                                  # its provenance stays listed
    snap = _snapshot(world)
    assert [e["run_id"] for e in snap["notebooks"]] == ["r2"] and snap["not_included"][0]["run_id"] == "r1"
    with zipfile.ZipFile(snap["archive"]) as zf:
        blob = json.loads(zf.read("prx_executed_notebooks/MANIFEST.json"))
        readme = zf.read("prx_executed_notebooks/README.md").decode("utf-8")
    assert blob["not_included"] == [{"execution": "r1/alpha", "run_id": "r1", "notebook": world["A"],
                                     "acceptance_record": "prototype/bank_pipeline/logs/r1/rec.json",
                                     "reason": "its executed copy was not kept"}]
    assert "`r1` (`prototype/bank_pipeline/logs/r1/rec.json`): its executed copy was not kept" in readme
    assert NP.verify_archive(snap["archive"], repo=world["repo"], man_path=world["man_path"])["passed"]
    rows = {r["execution"]: r for r in NP.inventory(man, world["repo"], [world["exe"]])["executions"]}
    assert rows["r1/alpha"]["status"] == "excluded"


def test_an_available_or_current_member_is_never_excluded(world):
    with pytest.raises(NP.PackagingError, match="current execution"):
        NP.exclude_execution(world["man_path"], "prx", "r1/alpha", "x", world["repo"], [world["exe"]])
    with pytest.raises(NP.PackagingError, match="available and accepted"):
        NP.exclude_execution(world["man_path"], "prx", "r2/alpha", "x", world["repo"], [world["exe"]])
    with pytest.raises(NP.PackagingError, match="every folder"):
        NP.exclude_execution(world["man_path"], "prx", "r2/alpha", "x", world["repo"], [])
    with pytest.raises(NP.PackagingError, match="not a member"):
        NP.exclude_execution(world["man_path"], "prx", "pin/beta", "x", world["repo"], [world["exe"]])


@pytest.mark.parametrize("excluded, match", [
    ({"r2/alpha": "x"}, "outside the snapshot's executions"),
    ({"nope/alpha": "x"}, "outside the snapshot's executions"),
    ({"pin/beta": " "}, "needs the reason"),
    (["r2/alpha"], "map execution ids to reasons"),
])
def test_a_snapshot_exclusion_must_name_a_listed_execution_and_a_reason(world, excluded, match):
    _rewrite(world, lambda m: m["snapshots"]["prx"].update(excluded=excluded))
    with pytest.raises(NP.PackagingError, match=match):
        NP.load_manifest(world["man_path"])


def test_the_command_line_registers_and_excludes_only_with_apply(world, capsys):
    base = ["--repo", str(world["repo"]), "--manifest", str(world["man_path"])]
    _new_run(world)
    reg = base + ["register", "r3", "--like", "r2/alpha", "--description", "later", "--current",
                  "--snapshot", "prx", "--executed-dir", str(world["exe"])]
    before = world["man_path"].read_bytes()
    assert NP.main(reg) == 0 and "dry run" in capsys.readouterr().out and world["man_path"].read_bytes() == before
    assert NP.main(reg + ["--apply"]) == 0 and "manifest written" in capsys.readouterr().out
    assert NP.main(reg + ["--apply"]) == 1 and "REFUSED" in capsys.readouterr().out
    (world["exe"] / "r2_alpha_executed.ipynb").unlink()
    exc = base + ["exclude", "prx", "r2/alpha", "--reason", "not kept", "--executed-dir", str(world["exe"])]
    assert NP.main(exc) == 0 and "dry run" in capsys.readouterr().out
    assert NP.main(exc + ["--apply"]) == 0
    assert NP.load_manifest(world["man_path"])["snapshots"]["prx"]["excluded"] == {"r2/alpha": "not kept"}


# ======================================================================
# Git keeps generated copies and archives out
# ======================================================================
GENERATED = ["executed_notebooks.zip", "pr42_executed_notebooks.zip", "pr42_executed_notebooks/MANIFEST.json",
             "pr42_executed_notebooks/c5_20261124_122029_feature_research_executed.ipynb",
             "prototype/bank_pipeline/c5_20261124_122029_feature_research_executed.ipynb",
             "prototype/bank_pipeline/c5_20261124_122029_feature_research_input.ipynb",
             "keep/prototype/bank_pipeline/glass_pipeline/glass_cascade_0123456789ab_tracked_outputs.ipynb"]


@pytest.mark.parametrize("name", GENERATED)
def test_git_ignores_generated_archives_and_copied_notebooks(name):
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", name]).returncode == 0


def test_no_tracked_file_is_a_generated_archive_or_copy_and_every_tracked_notebook_stays_tracked():
    tracked = _git(REPO, "ls-files", "-z").decode("utf-8").split("\0")
    rules = ["*_executed.ipynb", "*_input.ipynb", "executed_notebooks.zip", "*_executed_notebooks.zip",
             "*_tracked_outputs.ipynb"]
    assert [t for t in tracked if any(fnmatch.fnmatch(t.split("/")[-1], r) for r in rules)
            or "_executed_notebooks/" in t] == []
    notebooks = [t for t in tracked if t.endswith(".ipynb")]
    ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", *notebooks],
                             capture_output=True, text=True)
    assert notebooks and ignored.returncode == 1 and ignored.stdout == ""   # none of them is ignored


# ======================================================================
# the repository's manifest
# ======================================================================
def test_the_repository_manifest_lists_every_tracked_notebook():
    man = NP.load_manifest()
    assert P.coverage_problems(man, P.tracked_notebooks(REPO)) == []
    assert {nb for nb, s in man["notebooks"].items() if s["current"] is None} == {
        "prototype/churn_pipeline/notebooks/BestRulesWins_GlassBox_Meta.ipynb",
        "prototype/churn_pipeline/notebooks/churn_distributions.ipynb",
        "prototype/bank_pipeline/feature_research/training_research.ipynb"}     # until PR 43 Step 6B


#: The names the early PR 42 validation commands wrote their executed copies under (Step 6C).
AS_WRITTEN = {"c2_20261122_124809": "feature_research_executed.ipynb",
              "c3_20261122_201830": "feature_research_step3a_executed.ipynb",
              "c3_20261122_232809": "c3_20261122_232809_feature_research_step3b_executed.ipynb"}


def test_the_pr42_snapshot_is_the_accepted_pr42_runs_and_each_record_is_intact():
    """
    The snapshot holds the 6A document's accepted runs, in order, apart from any excluded with a reason, then
    any later run of the notebook registered since (Step 6C's final run): that one is the notebook's current.
    """
    man = NP.load_manifest()
    doc = (BANK / "research_logs" / "pr42" / "feature_contract_recommendations.md").read_text(encoding="utf-8")
    table = doc.split("<!-- BEGIN ACCEPTED RUNS -->")[1].split("<!-- END ACCEPTED RUNS -->")[0]
    accepted = [ln.split("|")[2].strip(" `") for ln in table.strip().splitlines()[2:]]
    snap = man["snapshots"]["pr42"]
    assert snap["directory"] == "pr42_executed_notebooks"
    runs = [man["executions"][e]["run_id"] for e in snap["executions"]]
    excluded = {man["executions"][e]["run_id"]: why for e, why in snap.get("excluded", {}).items()}
    assert set(excluded) <= set(accepted[:2]) and all(why.strip() for why in excluded.values())
    kept = [r for r in accepted if r not in excluded]
    assert runs[:len(kept)] == kept
    later = runs[len(kept):]
    assert later == sorted(later) and all(r > accepted[-1] for r in later)
    nb = "prototype/bank_pipeline/feature_research/feature_research.ipynb"
    assert man["notebooks"][nb]["current"] == f"{(later or accepted)[-1]}/feature_research"
    for eid in list(snap["executions"]) + list(snap.get("excluded", {})):
        e = man["executions"][eid]
        assert e["notebook"] == nb and eid == f"{e['run_id']}/feature_research"
        rec = REPO / e["acceptance"]["record"]
        raw = rec.read_bytes()
        assert rec.with_name(rec.name + ".sha256").read_text().strip() == _sha(raw)
        assert json.loads(raw)["run"]["run_id"] == e["run_id"] and e["acceptance"]["run_id_in_outputs"]
        assert e["acceptance"]["require"]["run/run_id"] == e["run_id"]
        assert e["source"]["file"] == AS_WRITTEN.get(e["run_id"], f"{e['run_id']}_feature_research_executed.ipynb")
    for r in later:                                                      # a full run: its record enforces acceptance
        e = man["executions"][f"{r}/feature_research"]
        assert e["acceptance"]["record"].endswith(f"/{r}/interaction_selection.json")
        assert e["acceptance"]["require"].get("acceptance/passed") is True


def test_the_cascade_executions_are_the_phase1_freezes_recorded_copies():
    man = NP.load_manifest()
    freeze = json.loads((BANK / "research_logs/pr39/c6_20261116_194111/phase1_freeze.json").read_text())
    assert freeze["passed"] is True
    for arm in ("glass", "blackbox"):
        e = man["executions"][f"c6_20261116_194111/{arm}"]
        assert e["sha256"] == freeze["executions"][arm]["executed_copy"]["sha256"]
        assert Path(freeze["executions"][arm]["executed_copy"]["path"].replace("\\", "/")).name == e["source"]["file"]


def test_the_exploratory_notebook_is_accepted_from_its_pinned_git_blob():
    man = NP.load_manifest()
    try:
        r = P.resolve("pr36_42753db/exploratory", man, REPO, [])
    except NP.PackagingError as exc:
        if any("not readable" in p for p in exc.problems):
            pytest.skip("the pinned blob is not in this checkout and the working copy is not it")
        raise
    assert r.summary["executed_cells"] == 31 and r.summary["images"] == 8 and r.summary["tables"] == 10
