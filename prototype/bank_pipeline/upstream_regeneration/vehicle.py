"""
upstream_regeneration.vehicle
=============================
The Commit 4A regeneration vehicle (execution contract section 5.4, D4): one
fresh-kernel execution of a notebook with its Stage 4 cells excluded, the
executed copy written outside the repository.

Exclusion is by a truncated copy (the handoff section 11 B precedent), not a
notebook switch, so that Commit 5A runs the same Stage 4 cells unchanged:

* the copy keeps every cell before the notebook's first Stage 4 section and
  nothing after it; the removed code cells must be exactly the Stage 4 cells
  (GLASS Cells 19-20, black-box Cells 35-38) and no kept cell may call a
  Stage 4 entry point (``STAGE4_CALLS``);
* each kept cell's source is byte-identical to the committed notebook's and
  its SHA-256 is recorded; outputs are cleared in the copy only;
* the kept code must run the pinned Stage 2 source with the legacy switch
  off (``RUN_LEGACY_STAGE2 = False``).

``execute`` runs the copy with nbclient (the engine of ``jupyter nbconvert
--execute``) in a new kernel whose working directory is the notebook's own
folder, so the notebooks find ``bank_pipeline/`` as they do interactively. The
committed notebook files are never written.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = ["NOTEBOOKS", "STAGE4_CALLS", "VehicleError", "repository_root", "require_outside",
           "truncated_copy", "execute"]

BANK_ROOT = Path(__file__).resolve().parents[1]
#: notebook -> (path relative to bank_pipeline/, its Stage 4 code-cell headers, in order)
NOTEBOOKS = {
    "glass": ("glass_pipeline/glass_cascade.ipynb", ("19", "20")),
    "blackbox": ("blackbox_pipeline/blackbox_cascade.ipynb", ("35", "36", "37", "38")),
}
#: Stage 4 entry points: fitting, comparing or tracing an arbiter.
STAGE4_CALLS = ("train_glass_arbiter", "train_meta_stage", "train_meta_xgb_stage",
                "compare_with_glass_arbiter", "compare_with_meta_ebm", "run_cascade_trace",
                "tune_min_confidence")
_HEADER = re.compile(r"^#\s*CELL\s+([0-9]+[A-Z]?)\s*:", re.M)


class VehicleError(RuntimeError):
    """The notebook cannot be executed under the section 5.4 vehicle; the reason is stated."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_root(start: Path = BANK_ROOT) -> Path:
    """The Git work tree holding bank_pipeline/ (bank_pipeline/ itself if none is found)."""
    for p in (Path(start).resolve(), *Path(start).resolve().parents):
        if (p / ".git").exists():
            return p
    return Path(start).resolve()


def require_outside(path, start: Path = BANK_ROOT) -> Path:
    """``path`` resolved, refused if it lies inside the repository (section 7.1)."""
    p = Path(path).expanduser().resolve()
    repo = repository_root(start)
    if p == repo or repo in p.parents:
        raise VehicleError(f"{p} is inside the repository {repo}: executed copies are written "
                           "outside it (section 7.1)")
    return p


def _source(cell) -> str:
    return "".join(cell.get("source") or [])


def _header(cell) -> Optional[str]:
    m = _HEADER.search(_source(cell))
    return m.group(1) if m else None


def _call_names(src: str) -> set:
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
    return {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)}


def _assigned(src: str, name: str) -> list:
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
    return [n.value for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(isinstance(t, ast.Name) and t.id == name
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]


def truncated_copy(notebook: str, *, root: Path = BANK_ROOT) -> tuple:
    """``(notebook dict, record)``: the notebook without its Stage 4 cells (module docstring)."""
    if notebook not in NOTEBOOKS:
        raise VehicleError(f"unknown notebook {notebook!r}: expected one of {sorted(NOTEBOOKS)}")
    rel, stage4 = NOTEBOOKS[notebook]
    path = Path(root) / rel
    raw = path.read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    cells = nb["cells"]
    first = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and _header(c) == stage4[0]]
    if len(first) != 1:
        raise VehicleError(f"{rel}: {len(first)} code cells headed CELL {stage4[0]}, expected one")
    cut = first[0]
    while cut > 0 and cells[cut - 1]["cell_type"] == "markdown":   # its section's markdown
        cut -= 1
    removed = [_header(c) for c in cells[cut:] if c["cell_type"] == "code"]
    if tuple(removed) != stage4:
        raise VehicleError(f"{rel}: the cells after the cut are {removed}, not the Stage 4 cells "
                           f"{list(stage4)}")
    kept = cells[:cut]
    code = [(i, c) for i, c in enumerate(kept) if c["cell_type"] == "code"]
    calls = {i: sorted(_call_names(_source(c)) & set(STAGE4_CALLS)) for i, c in code}
    if any(calls.values()):
        raise VehicleError(f"{rel}: kept cells call Stage 4 entry points "
                           f"{ {i: v for i, v in calls.items() if v} }")
    kept_src = "\n".join(_source(c) for _, c in code)
    switch = [ast.unparse(v) for _, c in code for v in _assigned(_source(c), "RUN_LEGACY_STAGE2")]
    if switch != ["False"]:
        raise VehicleError(f"{rel}: the legacy Stage 2 switch must be set exactly once, to False "
                           f"(found {switch})")
    if "load_pinned_stage2(GLOBAL_SPLIT)" not in kept_src:
        raise VehicleError(f"{rel}: the kept cells do not load the pinned Stage 2 source")
    out = copy.deepcopy(nb)
    out["cells"] = copy.deepcopy(kept)
    for c in out["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    for (i, c), (_, o) in zip(code, [(i, c) for i, c in enumerate(out["cells"])
                                     if c["cell_type"] == "code"]):
        if _source(o) != _source(c):                                 # pragma: no cover
            raise VehicleError(f"{rel}: cell {i} source changed in the copy")
    text = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    record = {
        "notebook": rel, "notebook_sha256": _sha(raw),
        "cells": {"total": len(cells), "kept": len(kept), "removed": len(cells) - cut},
        "removed_code_cells": [f"CELL {h}" for h in removed],
        "kept_code_cells": [{"index": i, "cell": _header(c),
                             "source_sha256": _sha(_source(c).encode("utf-8")),
                             "gated": _source(c).startswith("%%legacy_stage2\n")}
                            for i, c in code],
        "stage4_calls_in_kept_cells": [],
        "legacy_switch": "RUN_LEGACY_STAGE2 = False",
        "truncated_copy_sha256": _sha(text),
    }
    return out, record


def _kernel_python(client) -> Optional[str]:
    try:
        argv = client.km.kernel_spec.argv
    except Exception:                                   # noqa: BLE001
        return None
    return argv[0] if argv else None


def _loop_running() -> bool:
    """Whether this thread is running an event loop (``get_running_loop`` raises only if not)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _selector_loop() -> asyncio.AbstractEventLoop:
    """``asyncio.Runner``'s loop factory: a new selector loop, made this thread's current loop."""
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)        # a custom factory's job, per the Runner documentation
    return loop


def _run_notebook(client) -> None:
    """
    Run the notebook: ``client.execute()``, or on Windows its coroutine on a selector loop.

    PR 41 Commit 4, a reviewed execution-path exception. nbclient's ``execute`` runs on
    Python's default loop, which on Windows is the Proactor loop: it lacks the
    ``add_reader`` methods zmq needs, so pyzmq adds a tornado selector thread and warns.
    On Windows (Python 3.11+, no loop running) the notebook runs instead on its own
    ``SelectorEventLoop`` via ``asyncio.Runner``. Like the loops ``Runner`` creates itself,
    it is this thread's current loop during the run, then closed and unset (as after
    ``asyncio.run``); no process-wide setting changes. The kernel is started with
    ``subprocess.Popen``, so the selector loop's lack of asyncio subprocess support does
    not apply. Elsewhere, and inside an already-running loop, ``client.execute()`` runs as
    before.
    """
    if sys.platform == "win32" and sys.version_info >= (3, 11) and not _loop_running():
        try:
            with asyncio.Runner(loop_factory=_selector_loop) as runner:
                runner.run(client.async_execute())
        finally:
            asyncio.set_event_loop(None)
    else:
        client.execute()


def execute(nb: Mapping[str, Any], notebook: str, out_dir, run_id: str, *,
            env: Optional[Mapping[str, str]] = None, kernel_name: Optional[str] = None,
            root: Path = BANK_ROOT) -> Dict[str, Any]:
    """Execute ``nb`` (a ``truncated_copy``) in a fresh kernel; write both copies outside the repo."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise VehicleError(f"the vehicle needs nbclient and nbformat (pip install nbconvert): "
                           f"{exc}") from None
    out_dir = require_outside(out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{notebook}"
    src_path, exe_path = out_dir / f"{stem}_input.ipynb", out_dir / f"{stem}_executed.ipynb"
    for p in (src_path, exe_path):
        if p.exists():
            raise VehicleError(f"{p} exists: each execution writes new copies (new run id)")
    text = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    src_path.write_bytes(text)
    node = nbformat.reads(text.decode("utf-8"), as_version=4)
    rel, _ = NOTEBOOKS[notebook]
    cwd = (Path(root) / rel).parent
    name = kernel_name or (nb.get("metadata", {}).get("kernelspec") or {}).get("name") or "python3"
    client = NotebookClient(node, timeout=None, kernel_name=name, allow_errors=False,
                            resources={"metadata": {"path": str(cwd)}})
    client.km = client.create_kernel_manager()
    kernel_python = _kernel_python(client)
    # jupyter_client starts a spec whose argv[0] is a bare "python" with this interpreter
    bare = {"python", f"python{sys.version_info[0]}", "python{}.{}".format(*sys.version_info[:2])}
    if kernel_python in bare:
        kernel_python = sys.executable
    if kernel_python is None or Path(kernel_python).resolve() != Path(sys.executable).resolve():
        raise VehicleError(f"kernel {name!r} runs {kernel_python}, not this interpreter "
                           f"{sys.executable}: register this environment's kernel "
                           "(python -m ipykernel install --user --name <name>) and pass --kernel")
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    started, t0, error = _now(), time.perf_counter(), None
    try:
        _run_notebook(client)
    except Exception as exc:                            # noqa: BLE001 - recorded, then raised
        error = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1] if str(exc).strip() else ''}"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        nbformat.write(node, str(exe_path))
    return {"kernel": {"name": name, "python": kernel_python, "tool_python": sys.executable},
            "working_directory": str(cwd), "environment_variables": dict(env or {}),
            "input_copy": {"path": str(src_path), "sha256": _sha(src_path.read_bytes())},
            "executed_copy": {"path": str(exe_path), "sha256": _sha(exe_path.read_bytes())},
            "started_utc": started, "wall_seconds": round(time.perf_counter() - t0, 1),
            "error": error, "completed": error is None}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
