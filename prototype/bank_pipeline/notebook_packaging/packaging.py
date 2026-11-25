"""
notebook_packaging.packaging
============================
Executed-notebook archives built from explicitly accepted executions (PR 42, Step 6B).

The accepted executions are listed in ``accepted_executions.json`` beside this module. Each
entry names the tracked notebook it belongs to, its run id, where its executed copy comes
from and the tracked record that accepts it. Nothing is accepted because it is the newest
file or because it exists. The listed executions are packaged; no other file is.

Every executed copy is *read*, never moved, renamed, rewritten or cleared. Its bytes are
copied unchanged, so saved outputs, execution counts, tables, plots and metadata are those
of the accepted execution. Its SHA-256 is recorded in the archive's ``MANIFEST.json``.

Commands (run from ``prototype/bank_pipeline``)::

    python -m notebook_packaging inventory --executed-dir DIR [--executed-dir DIR ...]
    python -m notebook_packaging build     --executed-dir DIR ... --out-dir OUT [--replace]
    python -m notebook_packaging snapshot pr42 --executed-dir DIR ... --out-dir OUT [--replace]
    python -m notebook_packaging verify    ARCHIVE.zip [--executed-dir DIR ...] [--extract-to DIR]
    python -m notebook_packaging register  RUN_ID --like EXECUTION --description TEXT [--current]
                                           [--snapshot NAME] --executed-dir DIR ... [--apply]
    python -m notebook_packaging exclude   SNAPSHOT EXECUTION --reason TEXT --executed-dir DIR ... [--apply]
    python -m notebook_packaging clear-outputs --verified-archive ARCHIVE.zip --preserve-dir DIR
                                               --notebook PATH ... [--apply]

``build`` writes ``OUT/executed_notebooks.zip``: the current accepted executed copy of every
tracked notebook that has one, at the ZIP root. Source paths remain in MANIFEST.json. Every tracked notebook must
be listed in the manifest, with an accepted execution or with the stated reason it has none,
so no notebook is omitted silently. ``snapshot NAME`` writes ``OUT/<directory>/`` and
``OUT/<directory>.zip`` for one PR, with run ids in the file names. Archives are written
outside the repository. An existing archive is replaced only with ``--replace``.

``register`` adds a new run of the same kind as a listed execution (its names and required record
values with the run id replaced), only if that run is accepted now: its record intact and
confirming it, its executed copy found, complete and naming the run. ``exclude`` records why a
snapshot member's executed copy is not included, only while it cannot be found or accepted; its
execution entry and research records stay. Both are dry runs without ``--apply``.

``verify`` refuses an archive at once, before extracting or reading any member, if a member
path (or a path its ``MANIFEST.json`` lists) is unsafe on any platform, appears twice or
differs from another only by case. It then extracts each member below the extraction folder
only and checks it.

``clear-outputs`` is the separate, explicit operation that clears outputs and execution
counts from *tracked source* notebooks. It never runs as a side effect of building or
verifying. It refuses unless the named archive verifies and accounts for each notebook, and
it writes nothing without ``--apply``. Before it clears anything it copies each notebook's
current file, saved outputs included, into ``--preserve-dir`` outside the repository and
checks the copy. A notebook that is already output-free is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["MANIFEST_FORMAT", "ARCHIVE_FORMAT", "PROJECT_ARCHIVE", "PackagingError", "load_manifest",
           "inventory", "build_project", "build_snapshot", "verify_archive", "clear_outputs",
           "notebook_summary", "content_without_outputs", "main"]

MANIFEST_FORMAT = "notebook_packaging.accepted_executions/1"
ARCHIVE_FORMAT = "notebook_packaging.archive/1"
PROJECT_ARCHIVE = "executed_notebooks.zip"
ARCHIVE_MANIFEST = "MANIFEST.json"
ARCHIVE_README = "README.md"

BANK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BANK_ROOT.parents[1]
MANIFEST_PATH = Path(__file__).with_name("accepted_executions.json")

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLOB = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_TYPES = ("image/png", "image/jpeg", "image/svg+xml", "image/gif")


class PackagingError(RuntimeError):
    """A manifest, an input or an archive is not what packaging requires; the reasons are listed."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


# ======================================================================
# small helpers
# ======================================================================
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(obj: Any, path: str) -> Any:
    """The value at a slash-separated key path of a parsed JSON record."""
    for part in path.split("/"):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise KeyError(path)
    return obj


def _git(repo: Path, *args: str) -> Optional[bytes]:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _inside(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ======================================================================
# the manifest of accepted executions
# ======================================================================
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}


def _repo_path_problem(p: Any) -> Optional[str]:
    """Why ``p`` is not a safe relative path on every platform (None when it is)."""
    if not isinstance(p, str) or not p:
        return "not a non-empty string"
    if "\\" in p or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        return "must be a repository-relative path with forward slashes"
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "must not contain empty, '.' or '..' parts"
    if any(re.search(r'[\x00-\x1f<>:"|?*]', part) or part[-1] in ". " for part in parts):
        return "must not contain characters Windows forbids, or parts ending in '.' or ' '"
    if any(part.split(".")[0].casefold() in _WINDOWS_RESERVED for part in parts):
        return "must not use a name Windows reserves (CON, NUL, COM1, ...)"
    return None


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """The manifest, checked: every reference resolves and every name is safe on Windows."""
    try:
        man = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackagingError(f"manifest {path}: {exc}") from None
    problems: List[str] = []
    if man.get("format") != MANIFEST_FORMAT:
        problems.append(f"manifest format is {man.get('format')!r}, not {MANIFEST_FORMAT!r}")
    notebooks, executions = man.get("notebooks"), man.get("executions")
    snapshots = man.get("snapshots", {})
    if not isinstance(notebooks, dict) or not isinstance(executions, dict) or not isinstance(snapshots, dict):
        raise PackagingError(problems + ["manifest needs 'notebooks' and 'executions' objects"])
    folded: Dict[str, str] = {}
    for nb, spec in notebooks.items():
        why = _repo_path_problem(nb)
        if why or not nb.endswith(".ipynb"):
            problems.append(f"notebook {nb!r}: {why or 'not an .ipynb path'}")
        if nb.casefold() in folded:
            problems.append(f"notebooks {folded[nb.casefold()]!r} and {nb!r} collide on Windows")
        folded[nb.casefold()] = nb
        cur = spec.get("current") if isinstance(spec, dict) else None
        if not isinstance(spec, dict) or "current" not in spec:
            problems.append(f"notebook {nb!r}: needs 'current' (an execution id, or null with a reason)")
        elif cur is None and not str(spec.get("reason", "")).strip():
            problems.append(f"notebook {nb!r}: no current execution and no reason")
        elif cur is not None and (cur not in executions or executions[cur].get("notebook") != nb):
            problems.append(f"notebook {nb!r}: current execution {cur!r} is not an execution of it")
    for eid, e in executions.items():
        where = f"execution {eid!r}"
        if not isinstance(e, dict):
            problems.append(f"{where}: not an object")
            continue
        if e.get("notebook") not in notebooks:
            problems.append(f"{where}: notebook {e.get('notebook')!r} is not listed under 'notebooks'")
        if not isinstance(e.get("run_id"), str) or not _RUN_ID.match(e["run_id"]):
            problems.append(f"{where}: run_id {e.get('run_id')!r} is not a safe identifier")
        src = e.get("source", {})
        if isinstance(src, dict) and ("file" in src) == ("git_blob" in src):
            problems.append(f"{where}: source needs exactly one of 'file' and 'git_blob'")
        elif not isinstance(src, dict):
            problems.append(f"{where}: source is not an object")
        elif "file" in src:
            name = src["file"]
            if not isinstance(name, str) or not name.endswith(".ipynb") or re.search(r"[\\/:]", name):
                problems.append(f"{where}: source file {name!r} must be a bare .ipynb file name")
        else:
            if not _BLOB.match(str(src.get("git_blob", ""))) or not _SHA256.match(str(src.get("sha256_lf", ""))):
                problems.append(f"{where}: a git_blob source needs a 40-hex blob and a 64-hex sha256_lf")
        if e.get("sha256") is not None and not _SHA256.match(str(e["sha256"])):
            problems.append(f"{where}: sha256 pin {e['sha256']!r} is not 64 hex digits")
        acc = e.get("acceptance")
        why = _repo_path_problem(acc.get("record")) if isinstance(acc, dict) else "acceptance is not an object"
        if why:
            problems.append(f"{where}: acceptance record {why}")
    for name, snap in snapshots.items():
        if not _RUN_ID.match(name) or not isinstance(snap, dict):
            problems.append(f"snapshot {name!r}: not a safe name or not an object")
            continue
        directory = snap.get("directory", "")
        if not isinstance(directory, str) or not _RUN_ID.match(directory):
            problems.append(f"snapshot {name!r}: directory {directory!r} is not a safe folder name")
        ids = snap.get("executions", [])
        if not ids or not all(i in executions for i in ids) or len(set(ids)) != len(ids):
            problems.append(f"snapshot {name!r}: executions must be distinct listed execution ids")
        excluded = snap.get("excluded", {})
        if not isinstance(excluded, dict):
            problems.append(f"snapshot {name!r}: excluded must map execution ids to reasons")
            continue
        for x, why in excluded.items():
            if x not in executions or x in ids:
                problems.append(f"snapshot {name!r}: excluded {x!r} must be a listed execution outside the "
                                "snapshot's executions")
            if not isinstance(why, str) or not why.strip():
                problems.append(f"snapshot {name!r}: excluded {x!r} needs the reason it is not included")
    if problems:
        raise PackagingError(problems)
    return man


def tracked_notebooks(repo: Path) -> List[str]:
    out = _git(repo, "ls-files", "-z", "--", "*.ipynb")
    if out is None:
        raise PackagingError(f"git ls-files failed in {repo}: the tracked notebooks cannot be listed")
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def coverage_problems(man: Mapping[str, Any], tracked: Sequence[str]) -> List[str]:
    """Every tracked notebook is listed (with an execution or a reason), and nothing else is."""
    listed = set(man["notebooks"])
    return ([f"tracked notebook {p} is not listed in the manifest" for p in sorted(set(tracked) - listed)]
            + [f"manifest notebook {p} is not a tracked notebook" for p in sorted(listed - set(tracked))])


# ======================================================================
# one notebook's content
# ======================================================================
def _outputs_text(nb: Mapping[str, Any]) -> str:
    parts = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []) or []:
            parts.append(json.dumps(out, ensure_ascii=False))
    return "\n".join(parts)


def notebook_summary(data: bytes) -> Dict[str, Any]:
    """What a notebook file holds: its cells, executed cells, saved outputs, tables and plots."""
    try:
        nb = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackagingError(f"not a readable notebook: {exc}") from None
    if not isinstance(nb, dict) or nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
        raise PackagingError("not an nbformat 4 notebook")
    try:                                    # the reference validator, where installed
        import warnings
        import nbformat
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nbformat.validate(nbformat.reads(data.decode("utf-8"), as_version=nbformat.NO_CONVERT))
    except ImportError:
        pass
    except Exception as exc:                # noqa: BLE001 - any validation failure is a reason
        raise PackagingError(f"nbformat validation failed: {str(exc).splitlines()[0][:300]}") from None
    s = {"cells": len(nb["cells"]), "code_cells": 0, "nonempty_code_cells": 0, "executed_cells": 0,
         "unexecuted_nonempty": [], "outputs": 0, "errors": 0, "streams": 0, "html": 0, "tables": 0,
         "images": 0, "nbformat": f"{nb['nbformat']}.{nb.get('nbformat_minor', 0)}"}
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        s["code_cells"] += 1
        src = "".join(cell.get("source", "")) if isinstance(cell.get("source"), list) else cell.get("source", "")
        nonempty = bool(str(src).strip())
        s["nonempty_code_cells"] += nonempty
        if isinstance(cell.get("execution_count"), int):
            s["executed_cells"] += 1
        elif nonempty:
            s["unexecuted_nonempty"].append(i)
        for out in cell.get("outputs", []) or []:
            s["outputs"] += 1
            kind = out.get("output_type")
            if kind == "error":
                s["errors"] += 1
            elif kind == "stream":
                s["streams"] += 1
            bundle = out.get("data", {}) or {}
            if "text/html" in bundle:
                s["html"] += 1
                html = "".join(bundle["text/html"]) if isinstance(bundle["text/html"], list) else bundle["text/html"]
                s["tables"] += "<table" in str(html)
            s["images"] += sum(1 for t in _IMAGE_TYPES if t in bundle)
    return s


def completeness_problems(s: Mapping[str, Any]) -> List[str]:
    """Why a notebook is not a complete, successful execution (empty when it is)."""
    out = []
    if s["errors"]:
        out.append(f"failed execution: {s['errors']} error output(s)")
    if s["unexecuted_nonempty"]:
        out.append(f"partial execution: {len(s['unexecuted_nonempty'])} of {s['nonempty_code_cells']} "
                   f"non-empty code cells never ran (first at cell index {s['unexecuted_nonempty'][0]})")
    if s["nonempty_code_cells"] == 0 or s["outputs"] == 0:
        out.append("no saved outputs: not an executed notebook")
    return out


# ======================================================================
# resolving and accepting one execution
# ======================================================================
@dataclass
class Resolved:
    execution_id: str
    entry: Dict[str, Any]
    data: bytes
    sha256: str
    origin: str
    source_path: Optional[Path] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    acceptance: Dict[str, Any] = field(default_factory=dict)


def _locate(name: str, executed_dirs: Sequence[Path]) -> Path:
    hits = [Path(d) / name for d in executed_dirs if (Path(d) / name).is_file()]
    if not hits:
        raise PackagingError(f"executed copy {name} not found in "
                             + (", ".join(str(d) for d in executed_dirs) or "any --executed-dir"))
    if len({sha256(h.read_bytes()) for h in hits}) > 1:
        raise PackagingError(f"executed copy {name} found with different contents in "
                             + ", ".join(str(h.parent) for h in hits))
    return hits[0]


def _read_source(eid: str, e: Mapping[str, Any], repo: Path,
                 executed_dirs: Sequence[Path]) -> Tuple[bytes, str, Optional[Path]]:
    src = e["source"]
    if "file" in src:
        path = _locate(src["file"], executed_dirs)
        with open(path, "rb") as fh:                  # read-only: the original is never opened to write
            return fh.read(), str(path), path
    blob, want = src["git_blob"], src["sha256_lf"]
    data = _git(repo, "cat-file", "blob", blob)
    origin = f"git blob {blob}"
    if data is None:                                  # no Git, or no blob: the working copy if it is the blob
        wc = repo / e["notebook"]
        if wc.is_file() and sha256(lf(wc.read_bytes())) == want:
            data, origin = wc.read_bytes(), f"working copy of {e['notebook']} (equal to blob {blob})"
        else:
            raise PackagingError(f"{eid}: git blob {blob} is not readable and the working copy is not it")
    data = lf(data)
    if sha256(data) != want:
        raise PackagingError(f"{eid}: {origin} has LF SHA-256 {sha256(data)[:12]}, not the pinned {want[:12]}")
    return data, origin, None


def _accept(eid: str, e: Mapping[str, Any], data: bytes, repo: Path) -> Dict[str, Any]:
    acc = e["acceptance"]
    record = repo / acc["record"]
    if not record.is_file():
        raise PackagingError(f"{eid}: acceptance record {acc['record']} is missing")
    raw = record.read_bytes()
    side = record.with_name(record.name + ".sha256")
    if not side.is_file() or side.read_text(encoding="ascii").strip() != sha256(raw):
        raise PackagingError(f"{eid}: acceptance record {acc['record']} does not match its .sha256 sidecar")
    rec = json.loads(raw.decode("utf-8"))
    problems = []
    if acc.get("sha256"):
        try:
            recorded = _key(rec, acc["sha256"])
        except (KeyError, IndexError, ValueError):
            recorded = None
        if recorded != sha256(data):
            problems.append(f"{eid}: SHA-256 {sha256(data)[:12]} is not the accepted executed copy's "
                            f"({str(recorded)[:12]}, {acc['record']} {acc['sha256']})")
    for path, want in (acc.get("require") or {}).items():
        try:
            got = _key(rec, path)
        except (KeyError, IndexError, ValueError):
            got = "<missing>"
        if got != want:
            problems.append(f"{eid}: {acc['record']} {path} is {got!r}, not {want!r}")
    if acc.get("run_id_in_outputs") and e["run_id"] not in _outputs_text(json.loads(data.decode("utf-8"))):
        problems.append(f"{eid}: run id {e['run_id']} does not appear in the notebook's saved outputs")
    if problems:
        raise PackagingError(problems)
    return {"record": acc["record"], "record_sha256": sha256(raw),
            "checked": sorted(k for k in ("sha256", "require", "run_id_in_outputs") if acc.get(k))}


def resolve(eid: str, man: Mapping[str, Any], repo: Path, executed_dirs: Sequence[Path]) -> Resolved:
    """One accepted execution, read, validated as a complete notebook and checked against its record."""
    e = man["executions"][eid]
    data, origin, path = _read_source(eid, e, repo, executed_dirs)
    if e.get("sha256") and sha256(data) != e["sha256"]:
        raise PackagingError(f"{eid}: SHA-256 {sha256(data)[:12]} is not the manifest's pin {e['sha256'][:12]}")
    try:
        summary = notebook_summary(data)
    except PackagingError as exc:
        raise PackagingError([f"{eid}: {p}" for p in exc.problems]) from None
    bad = completeness_problems(summary)
    if bad:
        raise PackagingError([f"{eid}: {p}" for p in bad])
    acceptance = _accept(eid, e, data, repo)
    return Resolved(eid, dict(e), data, sha256(data), origin, path, summary, acceptance)


def _resolve_all(ids: Iterable[str], man, repo, executed_dirs) -> List[Resolved]:
    out, problems = [], []
    for eid in ids:
        try:
            out.append(resolve(eid, man, repo, executed_dirs))
        except PackagingError as exc:
            problems += exc.problems
    if problems:
        raise PackagingError(problems)
    return out


# ======================================================================
# inventory
# ======================================================================
def needed_executions(man: Mapping[str, Any]) -> set:
    """The executions some archive holds: every notebook's current one and every snapshot member."""
    return {spec["current"] for spec in man["notebooks"].values() if spec.get("current")} | {
        x for snap in man.get("snapshots", {}).values() for x in snap.get("executions", [])}


def inventory(man: Mapping[str, Any], repo: Path, executed_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Every listed execution's status, every tracked notebook's disposition, and unlisted files seen."""
    rows = []
    excluded = {x: f"excluded from snapshot {name!r}: {why}" for name, snap in man.get("snapshots", {}).items()
                for x, why in snap.get("excluded", {}).items()}
    for eid, e in man["executions"].items():
        try:
            r = resolve(eid, man, repo, executed_dirs)
            rows.append({"execution": eid, "status": "accepted", "sha256": r.sha256, "origin": r.origin,
                         "summary": {k: r.summary[k] for k in ("code_cells", "executed_cells", "outputs",
                                                               "html", "tables", "images")}})
        except PackagingError as exc:
            needed = eid not in excluded or eid in needed_executions(man)
            rows.append({"execution": eid, "status": "REJECTED" if needed else "excluded",
                         "problems": exc.problems if needed else [excluded[eid]] + exc.problems})
    listed = {e["source"]["file"] for e in man["executions"].values() if "file" in e["source"]}
    unlisted = sorted(str(p) for d in executed_dirs if Path(d).is_dir()
                      for p in Path(d).glob("*.ipynb") if p.name not in listed)
    try:
        coverage = coverage_problems(man, tracked_notebooks(repo))
    except PackagingError as exc:
        coverage = exc.problems
    return {"executions": rows, "unlisted_files": unlisted, "coverage_problems": coverage,
            "notebooks": {nb: spec.get("current") or f"none: {spec.get('reason')}"
                          for nb, spec in man["notebooks"].items()}}


# ======================================================================
# building
# ======================================================================
def _archive_entry(r: Resolved, archive_path: str) -> Dict[str, Any]:
    e = r.entry
    return {"archive_path": archive_path, "notebook": e["notebook"], "execution": r.execution_id,
            "run_id": e["run_id"], "sha256": r.sha256, "bytes": len(r.data),
            "source": dict(e["source"]), "description": e.get("description", ""),
            "acceptance": r.acceptance, "summary": r.summary}


def _readme(kind: str, name: str, entries: Sequence[Mapping[str, Any]],
            excluded: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"# {name}", "",
             "Executed Jupyter notebooks from Project ChurnBot "
             "(https://github.com/HKtrill/ai-business-coach), with their saved tables, plots and printed "
             "results. Open them in Jupyter, JupyterLab, VS Code or any notebook viewer; nothing needs to be "
             "installed or rerun to read them.", "",
             "Each notebook is a byte-for-byte copy of an accepted execution, identified by its run id. "
             "`MANIFEST.json` lists each file's SHA-256, the tracked notebook it was executed from and the "
             "research record that accepts it.", "",
             "The cascade notebooks report results on a held-out test split that was examined during earlier "
             "development, so those results are descriptive. The Feature Research Notebook reads training "
             "rows only. All of it is random-split research evidence, not a forward-in-time performance claim.",
             ""]
    lines.append("| File | Notebook | Run |")
    lines.append("|---|---|---|")
    for e in entries:
        lines.append(f"| `{e['archive_path']}` | `{e['notebook']}` | `{e['run_id']}` |")
    if excluded and kind == "snapshot":
        lines += ["", "Accepted runs whose executed copy is not included (their research records are in the "
                  "repository, under the acceptance record named):", ""]
        lines += [f"- `{x['run_id']}` (`{x['acceptance_record']}`): {x['reason']}" for x in excluded]
    elif excluded:
        lines += ["", "Tracked notebooks without an accepted execution (not included):", ""]
        lines += [f"- `{x['notebook']}`: {x['reason']}" for x in excluded]
    return "\n".join(lines) + "\n"


def _manifest_blob(kind: str, name: str, entries, excluded, man_path: Path, repo: Path) -> Dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    return {"format": ARCHIVE_FORMAT, "kind": kind, "name": name, "built_utc": _now(),
            "repository": {"head": head.decode().strip() if head else None,
                           "accepted_executions_sha256_lf": sha256(lf(Path(man_path).read_bytes())),
                           "builder_sha256_lf": sha256(lf(Path(__file__).read_bytes()))},
            "notebooks": list(entries), "not_included": list(excluded)}


def _check_folds(paths: Sequence[str]) -> None:
    seen: Dict[str, str] = {}
    problems = []
    for p in paths:
        k = p.casefold()
        if k in seen:
            problems.append(f"archive paths {seen[k]!r} and {p!r} collide")
        seen[k] = p
    if problems:
        raise PackagingError(problems)


def _zip_bytes(members: Sequence[Tuple[str, bytes]], when: datetime) -> bytes:
    import io
    buf = io.BytesIO()
    stamp = when.timetuple()[:6]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return buf.getvalue()


def _require_outside(out: Path, repo: Path) -> Path:
    out = Path(out).resolve()
    if _inside(out, repo):
        raise PackagingError(f"{out} is inside the repository {repo}: archives, their extracted copies and "
                             "preserved outputs are written outside it, so that none of them is tracked")
    return out


def _unchanged(resolved: Sequence[Resolved]) -> None:
    problems = [f"{r.execution_id}: {r.source_path} changed while packaging"
                for r in resolved if r.source_path is not None and sha256(r.source_path.read_bytes()) != r.sha256]
    if problems:
        raise PackagingError(problems)


def build_project(man_path: Path, repo: Path, executed_dirs: Sequence[Path], out_dir: Path, *,
                  replace: bool = False) -> Dict[str, Any]:
    """``out_dir/executed_notebooks.zip``: the current accepted copy of each tracked notebook."""
    man = load_manifest(man_path)
    problems = coverage_problems(man, tracked_notebooks(repo))
    if problems:
        raise PackagingError(problems)
    out_dir = _require_outside(out_dir, repo)
    target = out_dir / PROJECT_ARCHIVE
    if target.exists() and not replace:
        raise PackagingError(f"{target} exists: pass --replace to rebuild it")
    current = [(nb, spec["current"]) for nb, spec in man["notebooks"].items() if spec.get("current")]
    excluded = [{"notebook": nb, "reason": spec["reason"]}
                for nb, spec in man["notebooks"].items() if not spec.get("current")]
    names = [PurePosixPath(nb).name for nb, _ in current]
    _check_folds(names + [ARCHIVE_MANIFEST, ARCHIVE_README])
    resolved = _resolve_all([eid for _, eid in current], man, repo, executed_dirs)
    entries = [_archive_entry(r, name) for r, name in zip(resolved, names)]
    previous = _read_archive_manifest(target) if target.exists() else None
    blob = _manifest_blob("project", "executed_notebooks", entries, excluded, man_path, repo)
    members = [(e["archive_path"], r.data) for e, r in zip(entries, resolved)]
    members += [(ARCHIVE_MANIFEST, (json.dumps(blob, indent=1, ensure_ascii=False) + "\n").encode("utf-8")),
                (ARCHIVE_README, _readme("project", "Executed notebooks", entries, excluded).encode("utf-8"))]
    _write_atomic(target, _zip_bytes(members, datetime.now()))
    _unchanged(resolved)
    changes = None
    if previous is not None:
        old = {e["archive_path"]: e["sha256"] for e in previous.get("notebooks", [])}
        new = {e["archive_path"]: e["sha256"] for e in entries}
        changes = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
                   "replaced": sorted(p for p in new if p in old and old[p] != new[p])}
    return {"archive": str(target), "notebooks": entries, "not_included": excluded, "changes": changes}


def build_snapshot(man_path: Path, name: str, repo: Path, executed_dirs: Sequence[Path], out_dir: Path,
                   *, replace: bool = False) -> Dict[str, Any]:
    """``out_dir/<directory>/`` with copies named ``<run_id>_<stem>_executed.ipynb``, and its zip."""
    man = load_manifest(man_path)
    if name not in man.get("snapshots", {}):
        raise PackagingError(f"no snapshot {name!r} in the manifest")
    snap = man["snapshots"][name]
    out_dir = _require_outside(out_dir, repo)
    folder, target = out_dir / snap["directory"], out_dir / f"{snap['directory']}.zip"
    if target.exists() and not replace:
        raise PackagingError(f"{target} exists: pass --replace to rebuild it")
    resolved = _resolve_all(snap["executions"], man, repo, executed_dirs)
    names = [f"{r.entry['run_id']}_{PurePosixPath(r.entry['notebook']).stem}_executed.ipynb" for r in resolved]
    _check_folds(names + [ARCHIVE_MANIFEST, ARCHIVE_README])
    entries = [_archive_entry(r, f"{snap['directory']}/{n}") for r, n in zip(resolved, names)]
    excluded = [{"execution": x, "run_id": man["executions"][x]["run_id"], "notebook": man["executions"][x]["notebook"],
                 "acceptance_record": man["executions"][x]["acceptance"]["record"], "reason": why}
                for x, why in snap.get("excluded", {}).items()]
    blob = _manifest_blob("snapshot", snap["directory"], entries, excluded, man_path, repo)
    files = [(n, r.data) for n, r in zip(names, resolved)]
    files += [(ARCHIVE_MANIFEST, (json.dumps(blob, indent=1, ensure_ascii=False) + "\n").encode("utf-8")),
              (ARCHIVE_README, _readme("snapshot", snap["directory"], entries, excluded).encode("utf-8"))]
    if folder.exists():
        expected = {n for n, _ in files}
        stray = sorted(p.name for p in folder.iterdir() if p.name not in expected)
        if stray:
            raise PackagingError(f"{folder} holds files this snapshot does not produce: {', '.join(stray)}")
        if not replace:
            clash = [n for n, d in files[:-2] if (folder / n).exists() and (folder / n).read_bytes() != d]
            if clash:
                raise PackagingError(f"{folder}: {', '.join(clash)} differ from the accepted copies; "
                                     "pass --replace to overwrite them")
    for n, d in files:
        _write_atomic(folder / n, d)
    copied = [(f"{snap['directory']}/{n}", (folder / n).read_bytes()) for n, _ in files]
    if any(sha256(d) != sha256(orig) for (_, d), (_, orig) in zip(copied, files)):
        raise PackagingError(f"{folder}: a copy does not match what was written")
    _write_atomic(target, _zip_bytes(copied, datetime.now()))
    _unchanged(resolved)
    return {"archive": str(target), "directory": str(folder), "notebooks": entries, "not_included": excluded}


# ======================================================================
# recording a newly accepted run, or a snapshot member that was not preserved
# ======================================================================
def _manifest_bytes(man_path: Path, man: Mapping[str, Any]) -> bytes:
    """``man`` in the manifest file's own layout (one-space indent, LF or CRLF), checked by loading it."""
    raw = Path(man_path).read_bytes()
    if (json.dumps(json.loads(lf(raw).decode("utf-8")), indent=1, ensure_ascii=False) + "\n").encode("utf-8") \
            != lf(raw):
        raise PackagingError(f"{man_path}: its JSON layout is not the one this tool writes; edit it by hand")
    new = (json.dumps(man, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r\n" in raw:
        new = new.replace(b"\n", b"\r\n")
    with tempfile.TemporaryDirectory(prefix="nbpack_manifest_") as d:
        probe = Path(d) / "accepted_executions.json"
        probe.write_bytes(new)
        load_manifest(probe)
    return new


def register_execution(man_path: Path, like: str, run_id: str, repo: Path, executed_dirs: Sequence[Path], *,
                       description: str, current: bool = False, snapshots: Sequence[str] = (),
                       apply: bool = False) -> Dict[str, Any]:
    """
    Add a new run of the same kind as the accepted execution ``like``: its id, executed-copy name,
    acceptance record and required record values are ``like``'s with the run id replaced. The run
    is added only if it is accepted now, in the folders given (``resolve``: its record intact and
    confirming it, its copy found, complete and naming the run). Optionally it becomes its notebook's
    current execution and joins snapshots. Nothing is written without ``apply``.
    """
    man = load_manifest(man_path)
    if like not in man["executions"]:
        raise PackagingError(f"no execution {like!r} to take as the template")
    tmpl, problems = man["executions"][like], []
    old = tmpl["run_id"]
    if not _RUN_ID.match(run_id) or run_id == old:
        problems.append(f"run id {run_id!r} is not a new, safe identifier")
    if any(e["run_id"] == run_id and e["notebook"] == tmpl["notebook"] for e in man["executions"].values()):
        problems.append(f"{run_id} is already listed for {tmpl['notebook']}")
    if "file" not in tmpl["source"] or tmpl.get("sha256"):
        problems.append(f"{like}: only an unpinned executed-copy file can serve as a template")
    if not description.strip():
        problems.append("a description of the run is required")
    eid = like.replace(old, run_id)
    for what, value in (("execution id", like), ("executed-copy name", tmpl["source"].get("file", "")),
                        ("acceptance record", tmpl["acceptance"]["record"])):
        if old not in value:
            problems.append(f"{like}: its {what} {value!r} does not name its run, so it cannot be renamed")
    unknown = [s for s in snapshots if s not in man.get("snapshots", {})]
    if unknown:
        problems.append(f"no snapshot {unknown} in the manifest")
    if problems:
        raise PackagingError(problems)
    entry = json.loads(json.dumps(tmpl))
    entry.update(run_id=run_id, description=description.strip(), sha256=None)
    entry["source"]["file"] = tmpl["source"]["file"].replace(old, run_id)
    acc = entry["acceptance"]
    acc["record"] = acc["record"].replace(old, run_id)
    acc["require"] = {k: (run_id if v == old else v) for k, v in acc.get("require", {}).items()}
    if run_id not in acc["require"].values() and not acc.get("run_id_in_outputs"):
        raise PackagingError(f"{like}: nothing in its acceptance ties a record or a copy to its run")
    new = json.loads(json.dumps(man))
    new["executions"][eid] = entry
    if current:
        new["notebooks"][entry["notebook"]]["current"] = eid
    for s in snapshots:
        if eid not in new["snapshots"][s]["executions"]:
            new["snapshots"][s]["executions"].append(eid)
    data = _manifest_bytes(man_path, new)
    try:
        r = resolve(eid, new, repo, executed_dirs)
    except PackagingError as exc:
        raise PackagingError([f"{eid} is not accepted, so it is not registered:"] + exc.problems) from None
    if apply:
        _write_atomic(Path(man_path), data)
    return {"execution": eid, "entry": entry, "sha256": r.sha256, "origin": r.origin, "summary": r.summary,
            "current": current, "snapshots": list(snapshots), "applied": apply}


def exclude_execution(man_path: Path, snapshot: str, execution: str, reason: str, repo: Path,
                      executed_dirs: Sequence[Path], *, apply: bool = False) -> Dict[str, Any]:
    """
    Record that a snapshot member's executed copy is not available: it leaves the snapshot's
    ``executions`` for its ``excluded`` map, with ``reason``. Its execution entry, acceptance record
    and the research records stay as they are. Refused while the copy is found and accepted in the
    folders given, or while it is a notebook's current execution. Nothing is written without ``apply``.
    """
    man = load_manifest(man_path)
    snap = man.get("snapshots", {}).get(snapshot)
    if snap is None or execution not in snap["executions"]:
        raise PackagingError(f"{execution!r} is not a member of snapshot {snapshot!r}")
    if not reason.strip() or not executed_dirs:
        raise PackagingError("give the reason and every folder that was searched (--executed-dir)")
    if any(spec.get("current") == execution for spec in man["notebooks"].values()):
        raise PackagingError(f"{execution} is a notebook's current execution: the project archive needs it")
    try:
        r = resolve(execution, man, repo, executed_dirs)
    except PackagingError as exc:
        why = exc.problems
    else:
        raise PackagingError(f"{execution} is available and accepted ({r.origin}): it stays in the snapshot")
    new = json.loads(json.dumps(man))
    s = new["snapshots"][snapshot]
    s["executions"].remove(execution)
    s.setdefault("excluded", {})[execution] = reason.strip()
    data = _manifest_bytes(man_path, new)
    if apply:
        _write_atomic(Path(man_path), data)
    return {"execution": execution, "snapshot": snapshot, "reason": reason.strip(), "unavailable": why,
            "applied": apply}


# ======================================================================
# verification
# ======================================================================
def _read_archive_manifest(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.split("/")[-1] == ARCHIVE_MANIFEST]
            return json.loads(zf.read(names[0]).decode("utf-8")) if len(names) == 1 else None
    except (OSError, zipfile.BadZipFile, ValueError, KeyError):
        return None


def path_problems(names: Sequence[str]) -> List[str]:
    """Unsafe, duplicate or (case-insensitively) colliding member paths; empty when all are safe."""
    out, seen = [], {}
    for n in names:
        why = _repo_path_problem(n)
        if why:
            out.append(f"{n!r}: {why}")
        key = n.casefold()
        if key in seen:
            out.append(f"{n!r} collides with {seen[key]!r}" if seen[key] != n else f"{n!r} appears twice")
        seen.setdefault(key, n)
    return out


def _extract(zf: zipfile.ZipFile, names: Sequence[str], dest: Path) -> None:
    """Write each (already checked) member below ``dest`` only, refusing anything that resolves outside."""
    root = Path(dest).resolve()
    for n in names:
        target = (root / n).resolve()
        if not _inside(target, root) or target == root:
            raise PackagingError(f"member {n!r} would be written outside {root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as src, open(target, "xb") as out:
            shutil.copyfileobj(src, out)


def verify_archive(path: Path, *, repo: Path = REPO_ROOT, man_path: Path = MANIFEST_PATH,
                   executed_dirs: Sequence[Path] = (), extract_to: Optional[Path] = None) -> Dict[str, Any]:
    """Extract an archive and check every notebook against its manifest (and its source, if given)."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = "") -> bool:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    path = Path(path)
    if not check("the archive exists", path.is_file(), str(path)):
        return {"passed": False, "checks": checks}
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        check("the archive is a ZIP file", False, str(exc))
        return {"passed": False, "checks": checks}
    with zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        # Refuse before anything is read into, or written to, the file system. The names as stored
        # are checked too: on Windows, zipfile turns a stored backslash into "/".
        unsafe = path_problems([i.orig_filename for i in infos]) or path_problems(names)
        if not check("every member path is relative and safe on every platform (checked before extraction)",
                     not unsafe, unsafe):
            return {"passed": False, "checks": checks}
        check("every member's CRC is intact", zf.testzip() is None)
        man_names = [n for n in names if n.split("/")[-1] == ARCHIVE_MANIFEST]
        if not check("one MANIFEST.json", len(man_names) == 1, man_names):
            return {"passed": False, "checks": checks}
        prefix = man_names[0][: -len(ARCHIVE_MANIFEST)]
        try:
            blob = json.loads(zf.read(man_names[0]).decode("utf-8"))
            entries = list(blob.get("notebooks", []))
            listed = [e["archive_path"] for e in entries]
        except (ValueError, AttributeError, KeyError, TypeError) as exc:
            check("MANIFEST.json is readable", False, str(exc))
            return {"passed": False, "checks": checks}
        if not check("every path MANIFEST.json lists is relative and safe (checked before extraction)",
                     not path_problems(listed), path_problems(listed)):
            return {"passed": False, "checks": checks}
        check("the manifest format", blob.get("format") == ARCHIVE_FORMAT, blob.get("format"))
        expected = sorted(listed + [prefix + ARCHIVE_MANIFEST, prefix + ARCHIVE_README])
        check("the members are exactly the manifest's notebooks, MANIFEST.json and README.md",
              sorted(names) == expected, {"extra": sorted(set(names) - set(expected)),
                                          "missing": sorted(set(expected) - set(names))})
        tmp = None
        if extract_to is None:
            tmp = tempfile.TemporaryDirectory(prefix="nbpack_verify_")
            dest = Path(tmp.name)
        else:
            dest = Path(extract_to).resolve()
            if _inside(dest, repo):
                check("the extraction folder is outside the repository", False, str(dest))
                return {"passed": False, "checks": checks}
            if dest.exists() and any(dest.iterdir()):
                check("the extraction folder is new or empty", False, str(dest))
                return {"passed": False, "checks": checks}
        try:
            _extract(zf, names, dest)
            for e in entries:
                p = dest / e["archive_path"]
                data = p.read_bytes() if p.is_file() else b""
                tag = e["archive_path"]
                check(f"{tag}: extracted and its SHA-256 is the manifest's", sha256(data) == e["sha256"],
                      sha256(data)[:12])
                try:
                    s = notebook_summary(data)
                except PackagingError as exc:
                    check(f"{tag}: a valid, readable notebook", False, exc.problems)
                    continue
                check(f"{tag}: a valid, readable notebook", True, s["nbformat"])
                check(f"{tag}: a complete execution (every non-empty code cell ran, no error output)",
                      not completeness_problems(s), completeness_problems(s))
                keep = ("code_cells", "executed_cells", "outputs", "streams", "html", "tables", "images")
                check(f"{tag}: outputs, execution counts, tables and plots as when it was packaged",
                      all(s[k] == e["summary"].get(k) for k in keep), {k: s[k] for k in keep})
        finally:
            if tmp is not None:
                tmp.cleanup()
    to_source = [e for e in entries if executed_dirs or "git_blob" in e.get("source", {})]
    if to_source:
        man = load_manifest(man_path)
        for e in to_source:
            try:
                data, origin, _ = _read_source(e["execution"], man["executions"][e["execution"]], repo,
                                               executed_dirs)
                check(f"{e['archive_path']}: equal to its accepted source, which is unchanged",
                      sha256(data) == e["sha256"], origin)
            except (PackagingError, KeyError) as exc:
                check(f"{e['archive_path']}: equal to its accepted source, which is unchanged", False,
                      getattr(exc, "problems", str(exc)))
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "manifest": blob}


# ======================================================================
# clearing tracked notebooks (separate and explicit)
# ======================================================================
def content_without_outputs(nb: Mapping[str, Any]) -> Dict[str, Any]:
    """Everything a notebook holds except saved outputs and execution counts."""
    out = json.loads(json.dumps(nb))
    for cell in out.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return out


def _format_of(text: str, nb: Any) -> Optional[Dict[str, Any]]:
    for indent in (1, 2, 4):
        for ascii_ in (False, True):
            for tail in ("\n", ""):
                if json.dumps(nb, indent=indent, ensure_ascii=ascii_) + tail == text:
                    return {"indent": indent, "ensure_ascii": ascii_, "tail": tail}
    return None


PRESERVED_FORMAT = "notebook_packaging.preserved_outputs/1"
PRESERVED_INDEX = "PRESERVED.json"


def _cell_checks(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, bool]:
    """What clearing must leave unchanged, compared separately so a failure names what moved."""
    b, a = before.get("cells", []), after.get("cells", [])
    pairs = list(zip(b, a))
    return {
        "cell_count_and_order": len(b) == len(a) and all(x.get("cell_type") == y.get("cell_type") for x, y in pairs),
        "cell_ids": [c.get("id") for c in b] == [c.get("id") for c in a],
        "code_and_markdown": all(x.get("source") == y.get("source") for x, y in pairs),
        "cell_metadata": all(x.get("metadata") == y.get("metadata") for x, y in pairs),
        "notebook_metadata": (before.get("metadata"), before.get("nbformat"), before.get("nbformat_minor"))
        == (after.get("metadata"), after.get("nbformat"), after.get("nbformat_minor")),
        "everything_but_outputs": content_without_outputs(before) == after,
        "outputs_cleared": all(c.get("outputs") == [] and c.get("execution_count") is None
                               for c in a if c.get("cell_type") == "code"),
    }


def _preserved_name(rel: str, raw: bytes) -> str:
    """Where a tracked notebook's current outputs are kept: its repository path, content-addressed."""
    p = PurePosixPath(rel)
    return str(p.parent / f"{p.stem}_{sha256(raw)[:12]}_tracked_outputs.ipynb")


def _record_preservation(preserve_dir: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    index = preserve_dir / PRESERVED_INDEX
    current = {"format": PRESERVED_FORMAT, "entries": []}
    if index.is_file():
        try:
            current = json.loads(index.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise PackagingError(f"{index} is not readable: {exc}") from None
        if not isinstance(current, dict) or current.get("format") != PRESERVED_FORMAT:
            raise PackagingError(f"{index} is not a {PRESERVED_FORMAT} index")
    known = {e["file"] for e in current["entries"]}
    current["entries"] += [dict(e) for e in entries if e["file"] not in known]
    _write_atomic(index, (json.dumps(current, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


def clear_outputs(repo: Path, notebooks: Sequence[str], verified_archive: Path, *, preserve_dir: Path,
                  apply: bool = False, man_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """
    Clear outputs and execution counts of tracked source notebooks: a separate, explicit step that
    runs only after the named archive verifies and accounts for each notebook.

    Before anything is cleared, each notebook's current file, with its saved outputs, is copied
    byte for byte into ``preserve_dir`` (outside the repository), read back and checked, and
    recorded in ``PRESERVED.json``. Only when every copy checks does any notebook change. Cells,
    their order and ids, code, Markdown, cell and notebook metadata, the JSON layout and the line
    endings are kept; if anything else would change, the notebook is restored. A notebook that is
    already output-free is a no-op. Notebooks with no accepted execution (the legacy
    ``churn_pipeline`` ones) are outside this operation. Dry run unless ``apply``.
    """
    report = verify_archive(verified_archive, repo=repo, man_path=man_path)
    if not report["passed"]:
        raise PackagingError(["the archive does not verify: clear nothing"]
                             + [c["check"] for c in report["checks"] if not c["passed"]])
    preserve_dir = _require_outside(preserve_dir, repo)
    if preserve_dir.exists() and not preserve_dir.is_dir():
        raise PackagingError(f"{preserve_dir} is not a folder: the current outputs cannot be preserved there")
    man = load_manifest(man_path)
    separate = {nb for nb, spec in man["notebooks"].items() if not spec.get("current")}
    archived = {e["notebook"] for e in report["manifest"].get("notebooks", [])}
    tracked = set(tracked_notebooks(repo))
    head = _git(Path(repo), "rev-parse", "HEAD")
    plan, problems, seen = [], [], set()
    for nb in notebooks:
        given = Path(nb)
        if not given.is_absolute():
            given = Path(repo) / nb if (Path(repo) / nb).is_file() else Path.cwd() / nb
        try:
            rel = PurePosixPath(given.resolve().relative_to(Path(repo).resolve()).as_posix()).as_posix()
        except ValueError:
            problems.append(f"{nb}: outside the repository {repo}; executed copies are never cleared")
            continue
        if rel not in tracked:
            problems.append(f"{nb}: not a tracked notebook of {repo}; executed copies are never cleared")
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if rel in separate:
            problems.append(f"{rel}: has no accepted execution and is kept separate; clear-outputs leaves it alone")
            continue
        p = Path(repo) / rel
        raw = p.read_bytes()
        text = lf(raw).decode("utf-8")
        nb_json = json.loads(text)
        code = [c for c in nb_json.get("cells", []) if c.get("cell_type") == "code"]
        n_out = sum(len(c.get("outputs", []) or []) for c in code)
        n_cnt = sum(c.get("execution_count") is not None for c in code)
        item = {"notebook": rel, "outputs": n_out, "execution_counts": n_cnt, "path": p, "old": raw,
                "already_clear": n_out == 0 and n_cnt == 0}
        if item["already_clear"]:
            plan.append(item)                        # nothing to preserve or write: a no-op
            continue
        if rel not in archived:
            problems.append(f"{rel}: the archive holds no accepted executed copy of it; it is not accounted for")
            continue
        fmt = _format_of(text, nb_json)
        if fmt is None:
            problems.append(f"{rel}: its JSON layout cannot be reproduced exactly; clear it in Jupyter instead")
            continue
        new = (json.dumps(content_without_outputs(nb_json), indent=fmt["indent"],
                          ensure_ascii=fmt["ensure_ascii"]) + fmt["tail"]).encode("utf-8")
        if b"\r\n" in raw:
            new = new.replace(b"\n", b"\r\n")
        kept = preserve_dir / _preserved_name(rel, raw)
        if kept.exists() and kept.read_bytes() != raw:
            problems.append(f"{kept} exists with other content: a preserved copy is never overwritten")
            continue
        at_head = _git(Path(repo), "show", f"HEAD:{rel}")
        item.update(new=new, kept=kept, entry={
            "notebook": rel, "file": kept.relative_to(preserve_dir).as_posix(), "sha256": sha256(raw),
            "bytes": len(raw), "outputs": n_out, "execution_counts": n_cnt, "preserved_utc": _now(),
            "repository_head": head.decode().strip() if head else None,
            "equal_to_HEAD": None if at_head is None else lf(at_head) == lf(raw)})
        plan.append(item)
    if problems:
        raise PackagingError(problems)

    results = [{"notebook": i["notebook"], "outputs": i["outputs"], "execution_counts": i["execution_counts"],
                "already_clear": i["already_clear"], "preserved_to": None if i["already_clear"] else str(i["kept"]),
                "written": False} for i in plan]
    todo = [i for i in plan if not i["already_clear"]]
    if not apply or not todo:
        return {"applied": apply, "preserve_dir": str(preserve_dir), "notebooks": results}

    # 1. Preserve every notebook's current outputs and check each copy, before anything is cleared.
    for i in todo:
        if not i["kept"].exists():
            _write_atomic(i["kept"], i["old"])
        copy = i["kept"].read_bytes()
        if sha256(copy) != sha256(i["old"]) or notebook_summary(copy)["outputs"] != i["outputs"]:
            raise PackagingError(f"{i['kept']}: the preserved copy does not match {i['notebook']}; nothing cleared")
    _record_preservation(preserve_dir, [i["entry"] for i in todo])
    # 2. Clear, only if each notebook is still the file that was preserved.
    for i in todo:
        if sha256(i["path"].read_bytes()) != sha256(i["old"]):
            raise PackagingError(f"{i['notebook']} changed after it was preserved; it was not cleared")
    for i, row in zip(plan, results):
        if i["already_clear"]:
            continue
        _write_atomic(i["path"], i["new"])
        after = json.loads(lf(i["path"].read_bytes()).decode("utf-8"))
        checks = _cell_checks(json.loads(lf(i["old"]).decode("utf-8")), after)
        if not all(checks.values()):
            _write_atomic(i["path"], i["old"])                    # restore, then stop
            raise PackagingError(f"{i['notebook']}: clearing changed {[k for k, v in checks.items() if not v]}; "
                                 "the notebook was restored")
        head_text = _git(Path(repo), "show", f"HEAD:{i['notebook']}")
        row.update(written=True, checks=checks, equal_to_HEAD_except_outputs=None if head_text is None else
                   content_without_outputs(json.loads(lf(head_text).decode("utf-8"))) == after)
    return {"applied": True, "preserve_dir": str(preserve_dir), "notebooks": results}


# ======================================================================
# command line
# ======================================================================
def _print_problems(exc: PackagingError) -> int:
    print("REFUSED:")
    for p in exc.problems:
        print(f"  - {p}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m notebook_packaging",
                                 description="Executed-notebook archives from accepted executions.")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="accepted-executions manifest")
    sub = ap.add_subparsers(dest="command", required=True)

    def dirs(p):
        p.add_argument("--executed-dir", type=Path, action="append", default=[],
                       help="folder holding executed copies (repeatable)")

    p = sub.add_parser("inventory", help="report every listed execution and tracked notebook")
    dirs(p)
    p = sub.add_parser("build", help="build executed_notebooks.zip")
    dirs(p)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("snapshot", help="build one PR snapshot folder and its zip")
    p.add_argument("name")
    dirs(p)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("verify", help="extract an archive and verify it")
    p.add_argument("archive", type=Path)
    dirs(p)
    p.add_argument("--extract-to", type=Path)
    p = sub.add_parser("register", help="add a newly accepted run of the same kind as a listed one")
    p.add_argument("run_id")
    p.add_argument("--like", required=True, help="the accepted execution id to take as the template")
    p.add_argument("--description", required=True)
    p.add_argument("--current", action="store_true", help="make it its notebook's current execution")
    p.add_argument("--snapshot", action="append", default=[], help="add it to this snapshot (repeatable)")
    dirs(p)
    p.add_argument("--apply", action="store_true", help="write the manifest (default: dry run)")
    p = sub.add_parser("exclude", help="record that a snapshot member's executed copy is not available")
    p.add_argument("snapshot")
    p.add_argument("execution")
    p.add_argument("--reason", required=True)
    dirs(p)
    p.add_argument("--apply", action="store_true", help="write the manifest (default: dry run)")
    p = sub.add_parser("clear-outputs", help="clear tracked notebooks' outputs (separate, explicit)")
    p.add_argument("--verified-archive", type=Path, required=True)
    p.add_argument("--preserve-dir", type=Path, required=True,
                   help="folder outside the repository that receives each notebook's current outputs first")
    p.add_argument("--notebook", action="append", required=True, help="tracked notebook path (repeatable)")
    p.add_argument("--apply", action="store_true", help="write (default: dry run)")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    try:
        if a.command == "inventory":
            rep = inventory(load_manifest(a.manifest), repo, a.executed_dir)
            for r in rep["executions"]:
                if r["status"] == "accepted":
                    s = r["summary"]
                    print(f"  accepted  {r['execution']:<40} {r['sha256'][:12]}  code {s['executed_cells']}/"
                          f"{s['code_cells']}  outputs {s['outputs']}  tables {s['tables']}  plots {s['images']}"
                          f"  <- {r['origin']}")
                else:
                    print(f"  {r['status']:<9} {r['execution']}")
                    for prob in r["problems"]:
                        print(f"            - {prob}")
            for nb, cur in rep["notebooks"].items():
                print(f"  notebook  {nb}: {cur}")
            for f in rep["unlisted_files"]:
                print(f"  unlisted  {f} (not in the manifest: never packaged)")
            for prob in rep["coverage_problems"]:
                print(f"  COVERAGE  {prob}")
            ok = all(r["status"] != "REJECTED" for r in rep["executions"]) and not rep["coverage_problems"]
            print("inventory: every listed execution is accepted" if ok else "inventory: PROBLEMS above")
            return 0 if ok else 1
        if a.command == "build":
            rep = build_project(a.manifest, repo, a.executed_dir, a.out_dir, replace=a.replace)
            for e in rep["notebooks"]:
                print(f"  {e['archive_path']}  <- {e['execution']}  {e['sha256'][:12]}")
            for x in rep["not_included"]:
                print(f"  not included: {x['notebook']} ({x['reason']})")
            if rep["changes"] is not None:
                print(f"  changes against the previous archive: {rep['changes']}")
            print(f"built {rep['archive']}")
            return 0
        if a.command == "snapshot":
            rep = build_snapshot(a.manifest, a.name, repo, a.executed_dir, a.out_dir, replace=a.replace)
            for e in rep["notebooks"]:
                print(f"  {e['archive_path']}  {e['sha256'][:12]}")
            for x in rep["not_included"]:
                print(f"  not included: {x['run_id']} ({x['reason']})")
            print(f"built {rep['directory']} and {rep['archive']}")
            return 0
        if a.command == "verify":
            rep = verify_archive(a.archive, repo=repo, man_path=a.manifest, executed_dirs=a.executed_dir,
                                 extract_to=a.extract_to)
            for c in rep["checks"]:
                print(f"  {'ok  ' if c['passed'] else 'FAIL'}  {c['check']}"
                      + ("" if c["passed"] or not c["detail"] else f"  ({c['detail']})"))
            m = rep.get("manifest") or {}
            for e in m.get("notebooks", []):
                s = e["summary"]
                print(f"  inspect   {e['archive_path']}: {s['tables']} tables, {s['images']} plots, "
                      f"{s['outputs']} outputs")
            print("verify: PASSED" if rep["passed"] else "verify: FAILED")
            return 0 if rep["passed"] else 1
        if a.command == "register":
            rep = register_execution(a.manifest, a.like, a.run_id, repo, a.executed_dir, description=a.description,
                                     current=a.current, snapshots=a.snapshot, apply=a.apply)
            s = rep["summary"]
            print(f"  accepted  {rep['execution']}  {rep['sha256'][:12]}  code {s['executed_cells']}/{s['code_cells']}"
                  f"  outputs {s['outputs']}  tables {s['tables']}  plots {s['images']}  <- {rep['origin']}")
            print(f"  record    {rep['entry']['acceptance']['record']}  requires {rep['entry']['acceptance']['require']}")
            print(f"  current   {'yes' if rep['current'] else 'no'};  snapshots {rep['snapshots'] or 'none'}")
            print("register: manifest written" if rep["applied"] else "register: dry run (pass --apply to write)")
            return 0
        if a.command == "exclude":
            rep = exclude_execution(a.manifest, a.snapshot, a.execution, a.reason, repo, a.executed_dir,
                                    apply=a.apply)
            print(f"  {rep['execution']} is not available:")
            for prob in rep["unavailable"]:
                print(f"    - {prob}")
            print(f"  excluded from {rep['snapshot']}: {rep['reason']}")
            print("exclude: manifest written" if rep["applied"] else "exclude: dry run (pass --apply to write)")
            return 0
        if a.command == "clear-outputs":
            rep = clear_outputs(repo, a.notebook, a.verified_archive, preserve_dir=a.preserve_dir,
                                apply=a.apply, man_path=a.manifest)
            for r in rep["notebooks"]:
                if r["already_clear"]:
                    print(f"  no-op          {r['notebook']}: already output-free")
                    continue
                state = "cleared" if r["written"] else "would clear"
                print(f"  {state:<14} {r['notebook']}: {r['outputs']} outputs, {r['execution_counts']} execution "
                      f"counts; current outputs {'preserved' if r['written'] else 'to be preserved'} in "
                      f"{r['preserved_to']}")
                if r["written"]:
                    print("                 checks passed: " + ", ".join(k for k, v in r["checks"].items() if v)
                          + (f"; equal to HEAD except outputs: {r['equal_to_HEAD_except_outputs']}"
                             if r["equal_to_HEAD_except_outputs"] is not None else ""))
            print("clear-outputs: applied" if rep["applied"] else "clear-outputs: dry run (pass --apply to write)")
            return 0
    except PackagingError as exc:
        return _print_problems(exc)
    return 2


if __name__ == "__main__":
    sys.exit(main())
