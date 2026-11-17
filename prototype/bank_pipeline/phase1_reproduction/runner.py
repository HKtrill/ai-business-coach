"""
phase1_reproduction.runner
==========================
The Step 6B steps and their records (package docstring). Each step requires
the previous steps of the same run to have passed. It writes its record once
when every check passed; otherwise it writes ``failed/<step>_<utc>.json``
(Git-ignored, never read by code) and stops.

Code identity. ``production``: every code file the notebooks and checks run
(``records.code_identity``, the two notebooks and the baseline CLIs, with this
step's tests left out) and the ``stage4_regeneration`` package. ``analysis``:
this package and its tests. The executions require both to be preflight's.
``compare`` and ``freeze`` require the production code to be preflight's. A
correction to the analysis code after the executions is allowed and recorded
(both digests), so a comparison bug never forces the notebooks to run again;
``freeze`` requires the analysis code ``compare`` recorded.

Reused unchanged: ``upstream_regeneration`` (records, ``checks``: ``Gate``,
section 7.5, the Stage 1/3 contract checks, fresh reload, identities;
``inventory``; ``vehicle``: the fresh-kernel execution; ``acceptance.check``)
and ``stage4_regeneration`` (write-once records, ``upstream_resolution``, the
registered outputs, ``crossover.check`` and ``crossover.evaluate``), and
``tests.support.artifact_audit`` (arbiter loading, identity, the 2x2).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from stage4_regeneration import crossover as X
from stage4_regeneration import runner as R
from stage4_regeneration import stage4_copy as S
from upstream_regeneration import acceptance as Acc
from upstream_regeneration import checks as C
from upstream_regeneration import inventory as I
from upstream_regeneration import records as Rec
from upstream_regeneration import vehicle as V

from . import compare as Cmp

__all__ = ["RUN_ID", "UPSTREAM_RUN", "STAGE4_RUN", "PINS_FILE", "REPORTS", "TRACKED_WRITES",
           "STEP6_TESTS", "STEP6_CORRECTIONS", "SUPERSEDED_RUNS", "corrections", "superseded", "CONTRACT", "StepFailed", "new_run_id", "check_run_id", "run_dir",
           "code_identity", "step5a_identity", "full_copy", "classify", "since_step5a",
           "executed_in_order",
           "preflight", "execute", "compare", "freeze", "main"]

#: Commit 6 run ids (section 7.2): c6_<YYYYMMDD_HHMMSS>.
RUN_ID = re.compile(r"^c6_[0-9]{8}_[0-9]{6}$")
#: The accepted baseline: Commit 4 (Stages 1-3) and Commit 5 (Stage 4, the 2x2).
UPSTREAM_RUN = R.UPSTREAM_RUN
STAGE4_RUN = "c5_20261116_153459"
PINS_FILE = "upstream_pins.json"
#: A6.1: the saved suite reports, written into the run folder after preflight.
REPORTS = {"realdata": "step6_realdata_pytest.xml", "fast": "step6_fast_pytest.xml",
           "slow": "step6_slow_pytest.xml", "notebook": "step6_notebook_pytest.xml"}
#: Git-tracked files each notebook rewrites: GLASS Cell 20's trace figure and black-box
#: Cell 14A's three MLP CSVs (D7). Recorded before and after, copied out, never restored.
TRACKED_WRITES = {"glass": ("research_logs/figures/venn_cascade_trace.png",),
                  "blackbox": tuple(Acc.D7)}
#: This step's tests (under tests/, so inside records.code_identity): left out of the
#: production identity, part of the analysis identity.
STEP6_TESTS = ("tests/phase1_reproduction/__init__.py",
               "tests/phase1_reproduction/test_step6b.py")
CONTRACT = {"record": Rec.CONTRACT["record"], "version": Rec.CONTRACT["version"],
            "sections": ["5.4", "7.1", "7.2", "7.3", "7.4", "7.5", "12"],
            "gates": ["A6.1", "A6.2", "A6.3", "A6.4", "A6.5"], "decisions": ["D4", "D7", "D8"],
            "roadmap": "documentation/research_roadmap.md, Commit 6, Steps 6A and 6B"}
#: Code files Commit 6 corrected after Step 5A ran, each with its LF SHA-256 at Step 5A
#: (830a244) and the reason. ``step5a_identity`` restores those digests, so the check
#: that the notebooks and production code are Step 5A's still holds for every other file.
STEP6_CORRECTIONS = {
    "tests/realdata/test_current_artifacts.py": (
        "a10900f3ed344c17a9f3060e9f3f21dd88c588f3db8f03f405e53182b15fb32a",
        "the RF Pass 1 candidate test read rf_params_origin only as a GLASS-arm RF result "
        "path; a router fitted by the PR 38 cross-fit runner records the frozen RF profile "
        "identity instead (PR 37 M8, stage2_crossfit.runner). The test now verifies either "
        "origin; the path pathway is unchanged"),
}
#: Earlier Step 6B attempts, kept as they are (never read, hashed only) and named by the
#: freeze record.
SUPERSEDED_RUNS = {
    "c6_20261116_190848": (
        "preflight passed; the real-data suite stopped at 109 passed, 1 failed "
        "(tests/realdata/test_current_artifacts.py::"
        "test_the_current_rf_pass1_candidate_is_extracted_and_checked), corrected as "
        "STEP6_CORRECTIONS records"),
}
STATEMENT_KEYS = ("decision", "tracked_outputs")
EXCEPTION_KEY = "a6_2_exception"
MAX_STATEMENT = 2000


class StepFailed(RuntimeError):
    """A step's checks failed; its failure record is named."""


# ----------------------------------------------------------------------
# run ids and write-once records (sections 7.1, 7.2)
# ----------------------------------------------------------------------
def new_run_id() -> str:
    return "c6_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise Rec.RecordError(f"run id {run_id!r} is not c6_<YYYYMMDD_HHMMSS> (section 7.2)")
    return run_id


def run_dir(run_id: str) -> Path:
    return Rec.RECORD_ROOT / check_run_id(run_id)


def _record(run_id: str, name: str) -> Path:
    return run_dir(run_id) / f"{name}.json"


def read_record(run_id: str, name: str) -> Dict[str, Any]:
    rec = json.loads(R._verified_bytes(_record(run_id, name)).decode("utf-8"))
    if rec.get("run_id") != run_id:
        raise Rec.RecordError(f"{name}.json is not run {run_id}'s")
    return rec


def require_passed(run_id: str, name: str) -> Dict[str, Any]:
    rec = read_record(run_id, name)
    if rec.get("passed") is not True:
        raise Rec.RecordError(f"run {run_id}: {name} did not pass; the run stops there")
    return rec


def _sidecar(path: Path) -> str:
    return path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip()


def _rel(path) -> str:
    p = Path(path)
    return p.relative_to(Rec.BANK_ROOT).as_posix() if p.is_relative_to(Rec.BANK_ROOT) else str(p)


def _header(run_id: str, step: str) -> Dict[str, Any]:
    return {"format": f"pr39.commit6b.{step}/1", "run_id": run_id, "step": step,
            "contract": CONTRACT, "baseline": {"commit4": UPSTREAM_RUN, "commit5": STAGE4_RUN},
            "started_utc": Rec.now()}


def _finish(run_id: str, step: str, rec: Dict[str, Any], passed: bool) -> Dict[str, Any]:
    rec.update(finished_utc=Rec.now(), passed=bool(passed))
    if passed:
        rec["path"] = _rel(R._write_once(_record(run_id, step), R._text(rec)))
        return rec
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = run_dir(run_id) / "failed" / f"{step}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(R._text(rec))
    failed = [f"{g}: {c}" for g, r in (rec.get("gates") or {}).items() if isinstance(r, dict)
              for c in (r.get("failed") or [])]
    raise StepFailed(f"{step} did not pass; recorded in {path}" +
                     "".join(f"\n  - {f}" for f in failed[:20]))


# ----------------------------------------------------------------------
# code identity (module docstring)
# ----------------------------------------------------------------------
def _package(folder: str, omit=()) -> tuple:
    root, h, n = Rec.BANK_ROOT, hashlib.sha256(), 0
    for p in sorted((root / folder).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if "__pycache__" in p.parts or rel in omit:
            continue
        h.update(rel.encode("utf-8") + b"\0")
        h.update(bytes.fromhex(Rec.file_sha256_lf(p)))
        n += 1
    return h.hexdigest(), n


def code_identity() -> Dict[str, Any]:
    root = Rec.BANK_ROOT
    base = Rec.code_identity(root, overrides={f: None for f in STEP6_TESTS})
    s4, n4 = _package("stage4_regeneration")
    own, n6 = _package("phase1_reproduction")
    h = hashlib.sha256(bytes.fromhex(own))
    for f in STEP6_TESTS:
        if (root / f).is_file():
            h.update(f.encode("utf-8") + b"\0" + bytes.fromhex(Rec.file_sha256_lf(root / f)))
    return {"head": base["head"],
            "production": {"code_sha256": base["code_sha256"], "files": base["files"],
                           "stage4_regeneration_sha256": s4, "stage4_regeneration_files": n4},
            "analysis": {"sha256": h.hexdigest(), "phase1_reproduction_files": n6,
                         "tests": list(STEP6_TESTS)}}


def step5a_identity() -> Dict[str, Any]:
    """
    Step 5A's code identity re-derived: Step 5B's and this step's files left out,
    and each registered correction (``STEP6_CORRECTIONS``) at its Step 5A digest.
    """
    omit = [f for f in (*X.STEP5B_FILES, *STEP6_TESTS) if f.startswith("tests/")]
    overrides: Dict[str, Optional[str]] = {f: None for f in omit}
    overrides.update({f: sha for f, (sha, _) in STEP6_CORRECTIONS.items()
                      if (Rec.BANK_ROOT / f).is_file()})
    base = Rec.code_identity(Rec.BANK_ROOT, overrides=overrides)
    s4, n4 = _package("stage4_regeneration", omit=X.STEP5B_FILES)
    return {"code_sha256": base["code_sha256"], "files": base["files"],
            "stage4_regeneration_sha256": s4, "stage4_regeneration_files": n4}


def _same_code(pre: Mapping[str, Any], *, analysis: bool) -> Dict[str, Any]:
    now, want = code_identity(), pre["code"]
    if now["head"] != want["head"] or now["production"] != want["production"]:
        raise Rec.RecordError(
            f"run {pre['run_id']}: the production code or HEAD changed since preflight "
            f"({want['production']['code_sha256'][:16]} -> {now['production']['code_sha256'][:16]},"
            f" HEAD {str(want['head'])[:12]} -> {str(now['head'])[:12]}); the notebooks ran on "
            "preflight's code: start a new run id")
    if analysis and now["analysis"] != want["analysis"]:
        raise Rec.RecordError(f"run {pre['run_id']}: this package changed since preflight; the "
                              "executions use preflight's vehicle: start a new run id")
    return now


# ----------------------------------------------------------------------
# section 5.4: the full fresh-kernel copy (D4, Commit 6B)
# ----------------------------------------------------------------------
def full_copy(notebook: str, *, root: Optional[Path] = None) -> tuple:
    """
    ``(notebook dict, record)``: the committed notebook, every cell kept in
    order with its source byte-identical and hashed, outputs cleared. It must
    hold each Stage 4 cell once, set the legacy Stage 2 switch exactly once, to
    False, and load the pinned Stage 2 source. The committed file is not written.
    """
    root = Rec.BANK_ROOT if root is None else Path(root)
    if notebook not in V.NOTEBOOKS:
        raise V.VehicleError(f"unknown notebook {notebook!r}: expected one of {sorted(V.NOTEBOOKS)}")
    rel, stage4 = V.NOTEBOOKS[notebook]
    raw = (root / rel).read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    code = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    headers = [V._header(c) for _, c in code]
    for h in stage4:
        if headers.count(h) != 1:
            raise V.VehicleError(f"{rel}: {headers.count(h)} code cells headed CELL {h}, expected one")
    switch = [ast_unparse(v) for _, c in code for v in V._assigned(V._source(c), "RUN_LEGACY_STAGE2")]
    if switch != ["False"]:
        raise V.VehicleError(f"{rel}: the legacy Stage 2 switch must be set exactly once, to False "
                             f"(found {switch})")
    if "load_pinned_stage2(GLOBAL_SPLIT)" not in "\n".join(V._source(c) for _, c in code):
        raise V.VehicleError(f"{rel}: the notebook does not load the pinned Stage 2 source")
    out = copy.deepcopy(nb)
    for c in out["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    if [V._source(c) for c in out["cells"]] != [V._source(c) for c in nb["cells"]]:
        raise V.VehicleError(f"{rel}: a cell source changed in the copy")    # pragma: no cover
    text = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()           # noqa: E731
    cells = [{"index": i, "cell": f"CELL {V._header(c)}" if V._header(c) else None,
              "source_sha256": sha(V._source(c)),
              "gated": V._source(c).startswith("%%legacy_stage2\n")} for i, c in code]
    record = {"notebook": rel, "notebook_sha256": hashlib.sha256(raw).hexdigest(),
              "cells": {"total": len(nb["cells"]), "code": len(code),
                        "gated_legacy": sum(c["gated"] for c in cells),
                        "active": sum(not c["gated"] for c in cells)},
              "code_cells": cells, "stage4_cells": [f"CELL {h}" for h in stage4],
              "pins_read": sorted({r for _, c in code for r in S._pinned_roles(V._source(c))}),
              "legacy_switch": "RUN_LEGACY_STAGE2 = False", "copy_sha256": hashlib.sha256(text).hexdigest()}
    return out, record


def ast_unparse(node) -> str:
    import ast
    return ast.unparse(node)


def executed_in_order(path, n_code: int) -> Dict[str, Any]:
    """The executed copy: every code cell ran once, in order (counts 1..N), no error output."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    counts = [c.get("execution_count") for c in code]
    errors = [i for i, c in enumerate(code)
              if any(o.get("output_type") == "error" for o in c.get("outputs") or [])]
    ok = len(code) == n_code and counts == list(range(1, len(code) + 1)) and not errors
    return {"code_cells": len(code), "expected_code_cells": n_code,
            "execution_counts_1_to_n": counts == list(range(1, len(code) + 1)),
            "cells_with_error_output": errors, "passed": ok}


# ----------------------------------------------------------------------
# the model folders: Stages 1-3 by the Commit 4 rules, Stage 4 by Commit 5's
# ----------------------------------------------------------------------
RUN = "run"
_S4 = tuple(p for nb in S.OUTPUTS for p in (*S.OUTPUTS[nb].values(), *S.SIDECARS[nb]))


def _part(inv: Mapping[str, Any], stage4: bool) -> Dict[str, Any]:
    return {k: v for k, v in inv.items() if any(re.fullmatch(p, k) for p in _S4) == stage4}


def classify(before: Mapping[str, Mapping], after: Mapping[str, Mapping],
             notebook: str) -> Dict[str, Any]:
    """
    What one full execution (or, for ``RUN``, both) changed in the model
    folders: Stage 1-3 files by ``inventory.classify`` (one immutable version
    per role, companions, sidecars, mutable aliases, protected folders), Stage 4
    files by ``stage4_copy.classify`` (one version per arbiter, Meta-XGB
    sidecars). Anything else, and any modified or deleted file other than a
    documented alias, is a problem.
    """
    up = I.classify(_part(before, False), _part(after, False), I.RUN if notebook == RUN else notebook)
    s4 = S.classify(_part(before, True), _part(after, True), S.RUN if notebook == RUN else notebook)
    problems = [f"stages 1-3: {p}" for p in up["problems"]] + [f"stage 4: {p}" for p in s4["problems"]]
    return {"notebook": notebook, "stages_1_3": up, "stage4": s4,
            "produced": {**up["produced"], **s4["produced"]}, "problems": problems,
            "passed": not problems}


def since_step5a(want: Mapping[str, Mapping], now: Mapping[str, Mapping]) -> Dict[str, Any]:
    """
    The model folders against Step 5A's final inventory: nothing deleted, only
    documented aliases modified, and every addition a registered Stage 1-4
    version, companion, sidecar or alias, so an earlier failed attempt's files
    are kept, not deleted, and never mistaken for the baseline.
    """
    d = I.diff(want, now)
    aliases = {a for nb in I.MUTABLE_ALIASES for a in I.MUTABLE_ALIASES[nb]}
    patterns = [*(p for nb in I.OUTPUTS for p in I.OUTPUTS[nb].values()),
                *(p for nb in I.COMPANIONS for p in I.COMPANIONS[nb].values()),
                *(p for nb in I.SIDECARS for p in I.SIDECARS[nb]), *_S4]
    problems = ([f"unregistered addition {r}" for r in d["added"]
                 if r not in aliases and not any(re.fullmatch(p, r) for p in patterns)]
                + [f"existing file modified: {r}" for r in d["modified"] if r not in aliases]
                + [f"existing file deleted: {r}" for r in d["deleted"]])
    return {**d, "problems": problems}


def _tracked(rels) -> Dict[str, Optional[str]]:
    root = Rec.BANK_ROOT
    return {rel: (hashlib.sha256((root / rel).read_bytes()).hexdigest()
                  if (root / rel).is_file() else None) for rel in rels}


def _status_paths(status) -> List[str]:
    return [line[3:].strip().strip('"') for line in (status or [])]


def _git_unexpected(before, after, run_id: str, allowed_files=()) -> Dict[str, Any]:
    new = sorted(set(after or []) - set(before or []))
    gone = sorted(set(before or []) - set(after or []))

    def ok(line: str) -> bool:
        p = line[3:].strip().strip('"')
        return (f"research_logs/pr39/{run_id}/" in p or p.endswith(f"research_logs/pr39/{run_id}")
                or any(p == f or p.endswith("/" + f) for f in allowed_files))
    return {"new": new, "gone": gone,
            "unexpected": [ln for ln in new if not ok(ln)] + [ln for ln in gone if not ok(ln)]}


def _produced(delta: Mapping[str, Any], inv: Mapping[str, Mapping]) -> Dict[str, Dict[str, Any]]:
    return {k: {"path": rel, "bytes": inv[rel]["bytes"], "sha256": inv[rel]["sha256"]}
            for k, rel in delta["produced"].items()}


# ----------------------------------------------------------------------
# preflight
# ----------------------------------------------------------------------
def corrections() -> List[Dict[str, Any]]:
    """Each registered correction: its Step 5A digest, its digest now, and why."""
    root = Rec.BANK_ROOT
    return [{"path": f, "step5a_sha256": sha, "reason": why,
             "now_sha256": Rec.file_sha256_lf(root / f) if (root / f).is_file() else None}
            for f, (sha, why) in sorted(STEP6_CORRECTIONS.items())]


def superseded() -> Dict[str, Any]:
    """Each earlier attempt's files, hashed (never read), with why it was superseded."""
    out = {}
    for run, why in sorted(SUPERSEDED_RUNS.items()):
        d = Rec.RECORD_ROOT / run
        files = ({p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(d.rglob("*")) if p.is_file()} if d.is_dir() else None)
        out[run] = {"reason": why, "files": files}
    return out


def _references() -> Dict[str, Any]:
    """The accepted records' sidecars: the baseline every step must leave unchanged."""
    c4, c5 = Rec.RECORD_ROOT / UPSTREAM_RUN, Rec.RECORD_ROOT / STAGE4_RUN
    return {"commit4": {n: _sidecar(c4 / f"{n}.json") for n in (Acc.STEP, Acc.MANIFEST, "run_record")},
            "commit5": {n: _sidecar(c5 / f"{n}.json") for n in (*X.STEP5A_RECORDS, X.STEP,
                                                                 R.ARBITER_MANIFEST)}}


def preflight(run_id: str, sums_path) -> Dict[str, Any]:
    """The accepted baseline re-verified and section 7.5, before anything is written."""
    check_run_id(run_id)
    if _record(run_id, "preflight").exists():
        raise Rec.RecordError(f"run {run_id} already has a preflight record: new run id")
    root = Rec.BANK_ROOT
    rec = _header(run_id, "preflight")
    gate = C.Gate("preflight")
    held: Dict[str, Any] = {}

    def c4():
        held["c4"] = Acc.check(UPSTREAM_RUN)
        return held["c4"]["accepted"] is True

    def c5():
        held["c5"] = X.check(STAGE4_RUN)
        return held["c5"]["reproduced"] is True

    gate.check(f"the Commit 4 acceptance of {UPSTREAM_RUN} and its artifact manifest re-verify "
               "(acceptance.check: sidecars, links, the six Stage 1-3 files on disk by SHA-256)",
               c4, lambda: held["c4"]["manifest_sha256"])
    gate.check(f"the Commit 5 baseline of {STAGE4_RUN} reproduces (crossover.check: the Step 5A "
               "records and the four arbiters by SHA-256, every Step 5B result recomputed)", c5,
               lambda: held["c5"]["record_sha256"])
    code, s5a = code_identity(), step5a_identity()
    keys = ("code_sha256", "stage4_regeneration_sha256")

    def same_as_5a():
        held["c5_code"] = R.require_passed(STAGE4_RUN, "run_record")["code"]
        return all(s5a[k] == held["c5_code"][k] for k in keys)

    gate.check("the notebooks and production code are the ones Step 5A executed (code identity "
               "with Step 5B's and this step's files left out, and each registered correction "
               "at its Step 5A digest, = the Commit 5 run record's)",
               same_as_5a, lambda: {k: (held["c5_code"].get(k), s5a[k]) for k in keys})
    fixes = corrections()
    gate.check("each registered correction since Step 5A is present and changed (recorded with "
               "both digests and its reason)",
               lambda: all(c["now_sha256"] and c["now_sha256"] != c["step5a_sha256"] for c in fixes),
               lambda: [f"{c['path']}: {str(c['step5a_sha256'])[:12]} -> {str(c['now_sha256'])[:12]}"
                        for c in fixes])
    earlier = superseded()
    gate.check("each superseded Step 6B attempt is kept as it is (its files hashed, not read)",
               lambda: all(e["files"] for e in earlier.values()),
               lambda: {r: (len(e["files"]) if e["files"] else "missing") for r, e in earlier.items()})
    gate.check("the accepted Commit 4 and Commit 5 records' SHA-256 sidecars recorded (the two "
               "checks above verify each record against its sidecar; compared again after the run)",
               lambda: held.setdefault("refs", _references()) is not None)
    evidence = C.preservation(gate, sums_path)
    inv = I.snapshot(root)

    def models_since_5a():
        held["models"] = since_step5a(
            R.require_passed(STAGE4_RUN, "execution_blackbox")["inventory_after"], inv)
        return not held["models"]["problems"]

    gate.check("7.5.5 models/ since the Step 5A black-box execution: every existing file (each "
               "accepted version included) unchanged, nothing deleted; any addition a registered "
               "Stage 1-4 version, sidecar or alias (an earlier Step 6B attempt's, preserved)",
               models_since_5a, lambda: held["models"]["problems"][:8]
               or {k: len(held["models"][k]) for k in ("added", "modified")})
    status = Rec.git_status(root)
    tracked = _tracked(t for nb in TRACKED_WRITES for t in TRACKED_WRITES[nb])
    dirty = [p for p in _status_paths(status) for t in tracked if p == t or p.endswith("/" + t)]
    gate.check("7.5.6 git status recorded; each Git-tracked file a notebook writes is unmodified "
               "and present before the run", lambda: status is not None and not dirty
               and all(tracked.values()), lambda: dirty or {k: (v or "missing")[:16]
                                                            for k, v in tracked.items()})
    gate.check("no saved report of this run exists yet: each suite runs after preflight, on this "
               "code", lambda: not any((run_dir(run_id) / f).exists() for f in REPORTS.values()))
    copies: Dict[str, Any] = {}
    for nb in V.NOTEBOOKS:
        gate.check(f"the {nb} notebook's full copy builds: every cell in order, sources "
                   "byte-identical and hashed, the legacy Stage 2 switch off, Stage 2 by pin",
                   lambda nb=nb: copies.setdefault(nb, full_copy(nb, root=root)[1]) is not None,
                   lambda nb=nb: f"{copies[nb]['cells']['code']} code cells "
                                 f"({copies[nb]['cells']['gated_legacy']} legacy-gated)")
    env = Rec.environment()
    envs = {}
    for run, step in ((UPSTREAM_RUN, "run_record"), (STAGE4_RUN, "run_record")):
        try:
            then = (R.read_record(run, step) if run.startswith("c5")
                    else Rec.read_record(run, step))["environment"]
            envs[run] = Cmp.environment_differences(env, then)
        except Exception as exc:                            # noqa: BLE001 - recorded
            envs[run] = {"error": f"{type(exc).__name__}: {exc}"}
    rec.update(gates={"preflight": gate.record()},
               references={"commit4": held.get("c4"), "commit5": held.get("c5"),
                           "sidecars": held.get("refs"), "pr38_stage2_run": C.PR38_RUN_ID},
               evidence=evidence, inventory=inv, git_status=status, tracked_files=tracked,
               live_counterparts=C.live_counterparts(evidence.get("snapshot") or {}),
               full_copies=copies, reports=dict(REPORTS),
               layout={"records": f"research_logs/pr39/{run_id}/",
                       "pins_file": f"research_logs/pr39/{run_id}/{PINS_FILE}",
                       "executed_notebooks": "outside the repository (--out-dir); SHA-256 recorded",
                       "tracked_writes": {nb: list(v) for nb, v in TRACKED_WRITES.items()},
                       "failed_attempts": f"research_logs/pr39/{run_id}/failed/ (Git-ignored)"},
               code=code, step5a_code=s5a, corrections=fixes, superseded_runs=earlier,
               environment=env, environment_vs_baseline=envs,
               sums={"path": str(sums_path)})
    return _finish(run_id, "preflight", rec, gate.passed)


# ----------------------------------------------------------------------
# section 5.4: one full execution
# ----------------------------------------------------------------------
def execute(run_id: str, notebook: str, out_dir, *, kernel: Optional[str] = None) -> Dict[str, Any]:
    """One fresh-kernel, top-to-bottom execution of a committed notebook."""
    from shared.pinned_artifacts import PINS_ENV, STAGE2_PINS, PinnedFile, read_pins, write_pins
    pre = require_passed(run_id, "preflight")
    code = _same_code(pre, analysis=True)
    step = f"execution_{notebook}"
    if _record(run_id, step).exists():
        raise Rec.RecordError(f"run {run_id} already executed the {notebook} notebook")
    root, pins_path = Rec.BANK_ROOT, run_dir(run_id) / PINS_FILE
    if notebook == "glass":
        expected_before, env, glass = pre["inventory"], {PINS_ENV: ""}, None
        if pins_path.exists():
            raise Rec.RecordError(f"{pins_path} exists before the GLASS execution")
    elif notebook == "blackbox":
        glass = require_passed(run_id, "execution_glass")
        expected_before, env = glass["inventory_after"], {PINS_ENV: str(pins_path)}
    else:
        raise V.VehicleError(f"unknown notebook {notebook!r}: expected one of {sorted(V.NOTEBOOKS)}")
    rec = _header(run_id, step)
    gate = C.Gate(f"execution [{notebook}]")
    out_dir = V.require_outside(out_dir, root)
    nb, copy_rec = full_copy(notebook, root=root)
    rec["full_copy"] = copy_rec
    gate.check("the copy is the one preflight built (copy SHA-256)",
               lambda: copy_rec["copy_sha256"] == pre["full_copies"][notebook]["copy_sha256"])
    if glass is not None:
        def pins_ok():
            files = read_pins(pins_path)["files"]                       # sidecar-checked
            return (_sidecar(pins_path) == glass["pins_file"]["sha256"]
                    and set(copy_rec["pins_read"]) <= set(files))
        gate.check("the pins file is the one the GLASS execution wrote (sidecar, SHA-256) and "
                   "pins every role the notebook reads by pin", pins_ok, copy_rec["pins_read"])
    before = I.snapshot(root)
    gate.check("the model folders are as the previous step left them",
               lambda: I.diff(expected_before, before) == {"added": [], "modified": [], "deleted": []},
               lambda: {k: v[:5] for k, v in I.diff(expected_before, before).items()})
    status_before = Rec.git_status(root)
    tracked_before = _tracked(TRACKED_WRITES[notebook])
    gate.check("each Git-tracked file this notebook writes is as preflight recorded it",
               lambda: tracked_before == {t: pre["tracked_files"][t] for t in tracked_before})
    if not gate.passed:
        rec.update(gates={"execution": gate.record()}, code=code)
        return _finish(run_id, step, rec, False)
    run = V.execute(nb, notebook, out_dir, run_id, env=env, kernel_name=kernel, root=root)
    rec["execution"] = run
    gate.check("fresh-kernel execution completed with no cell error", lambda: run["completed"],
               run.get("error") or f"{run['wall_seconds']} s")
    order: Dict[str, Any] = {}
    gate.check("every code cell ran once, top to bottom (execution counts 1..N, no error output)",
               lambda: order.update(executed_in_order(run["executed_copy"]["path"],
                                                      copy_rec["cells"]["code"])) or order["passed"],
               lambda: order)
    rec["order"] = order
    after = I.snapshot(root)
    delta = classify(before, after, notebook)
    rec.update(inventory_after=after, inventory_delta=delta)
    gate.check(f"only the {notebook} notebook's registered outputs, sidecars and mutable aliases "
               "changed in the model folders; no existing version modified or deleted",
               lambda: delta["passed"], lambda: delta["problems"][:8] or delta["produced"])
    writes: Dict[str, Any] = {}

    def copied():
        for rel, sha in _tracked(TRACKED_WRITES[notebook]).items():
            w = writes[rel] = {"before": tracked_before[rel], "produced": sha,
                               "changed": sha != tracked_before[rel]}
            if w["changed"] and sha is not None:
                dest = out_dir / f"{run_id}_{notebook}_{PurePosixPath(rel).name}"
                with open(dest, "xb") as fh:                    # never over an earlier copy
                    fh.write((root / rel).read_bytes())
                w["copy"] = {"path": str(dest),
                             "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        return all(w["produced"] is not None and (not w["changed"] or w["copy"]["sha256"] ==
                                                  w["produced"]) for w in writes.values())

    gate.check("each tracked file it rewrote is copied outside the repository (SHA-256 "
               "recorded), none deleted; the working-tree file is left as the notebook wrote it",
               copied, lambda: {k: "rewritten" if w["changed"] else "unchanged"
                                for k, w in writes.items()})
    rec["tracked_writes"] = writes
    status = Rec.git_status(root)
    git = _git_unexpected(status_before, status, run_id, TRACKED_WRITES[notebook])
    rec["git_status_delta"] = git
    gate.check("git status: only this run's records and this notebook's registered tracked files "
               "changed", lambda: status is not None and not git["unexpected"],
               lambda: git["unexpected"][:8])
    if delta["passed"]:
        rec["produced"] = _produced(delta, after)
    if gate.passed and notebook == "glass":
        def pins():
            p = rec["produced"]
            files = {r: PinnedFile(r, p[r]["path"], p[r]["sha256"], int(p[r]["bytes"]))
                     for r in ("stage1.lr", "stage3.ebm")}
            g = p["glass_arbiter"]
            files[S.GLASS_ARBITER_ROLE] = PinnedFile(S.GLASS_ARBITER_ROLE, g["path"], g["sha256"],
                                                     int(g["bytes"]))
            files.update({s.role: s for s in STAGE2_PINS})
            write_pins(pins_path, run_id, files, root=root)
            rec["pins_file"] = {"path": _rel(pins_path), "sha256": _sidecar(pins_path),
                                "roles": sorted(files)}
            return True

        gate.check("the run's pins file written: this execution's GLASS LR, EBM and GLASS Arbiter "
                   "and the two Stage 2 pins (read by black-box Cells 14A, 33 and 35)", pins)
    rec.update(gates={"execution": gate.record()}, code=code)
    return _finish(run_id, step, rec, gate.passed)


# ----------------------------------------------------------------------
# section 7.4 and A6.3-A6.4
# ----------------------------------------------------------------------
def _pin(role, e):
    from shared.pinned_artifacts import PinnedFile
    return PinnedFile(role, str(e["path"]), str(e["sha256"]), int(e["bytes"]))


def accepted_pins() -> Dict[str, Any]:
    """The accepted baseline: Commit 4's six Stage 1-3 files and Commit 5's four arbiters."""
    from shared.pinned_artifacts import read_pins
    files = read_pins(R.manifest_path())["files"]
    out = {r: _pin(r, files[r]) for r in Acc.ROLES}
    arb = R.require_passed(STAGE4_RUN, "run_record")["arbiters"]
    out.update({k: _pin(k, arb[k]) for k in S.ROOT_KEYS})
    return out


def reproduced_pins(glass: Mapping[str, Any], black: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _pin(k, v) for k, v in {**glass["produced"], **black["produced"]}.items()}


STAGE1, STAGE3 = ("stage1.lr", "stage1.mlp"), ("stage3.ebm", "stage3.xgb")


def _stage2_file(art, root: Path) -> Optional[Path]:
    fields = art if isinstance(art, dict) else vars(art)
    up = (fields.get("stage4_inputs") or {}).get("upstream") or {}
    p = ((up.get("stages") or {}).get("stage2") or {}).get("artifact_path")
    if not p:
        return None
    p = Path(p) if Path(p).is_absolute() else root / p
    return p.resolve() if p.is_file() else None


def _same_file(a, b) -> bool:
    try:
        return bool(a and b) and os.path.samefile(a, b)
    except OSError:
        return False


def evaluate(accepted: Mapping[str, Any], reproduced: Mapping[str, Any], g2, g4) -> Dict[str, Any]:
    """The section 7.4 evaluation and the A6.4 contract checks of the reproduced artifacts."""
    import joblib
    from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
    from shared.pinned_artifacts import STAGE2_PINS
    from tests.support import artifact_audit as audit
    root = Rec.BANK_ROOT
    split, population = C.global_split(), C.frozen_population()
    load = lambda pin: joblib.load(pin.resolve(root))                          # noqa: E731
    roles: Dict[str, Any] = {}
    contract: Dict[str, Any] = {}
    rep = {r: load(reproduced[r]) for r in STAGE1}
    contract["stage1"] = C.stage1_checks(g4, rep, split, population)
    for r in STAGE1:
        roles[r] = Cmp.stage1(load(accepted[r]), rep[r])
    rep = {r: load(reproduced[r]) for r in STAGE3}
    contract["stage3"] = C.stage3_checks(g4, rep, population, split=split)
    for r in STAGE3:
        roles[r] = Cmp.stage3(load(accepted[r]), rep[r])
    del rep
    contract["reload"] = C.fresh_reload(g4, {k: p.resolve(root) for k, p in reproduced.items()})
    ids = C.identities(split, population)
    feats = [str(f) for f in STAGE4_FEATURES]
    ids["feature_contracts"]["stage4_meta_xgb"] = {
        "ordered": feats, "sha256": hashlib.sha256(json.dumps(feats).encode("utf-8")).hexdigest()}
    c5_ids = R.require_passed(STAGE4_RUN, "run_record")["identities"]
    g4.check("A6.4 the section 7.3 identities (population, split and label fingerprints, fold "
             "plans, feature contracts per stage, binning, configuration) are the Commit 5 run "
             "record's, value for value", lambda: Rec.canonical(ids) == Rec.canonical(c5_ids),
             lambda: X._first_difference(Rec.canonical(c5_ids), Rec.canonical(ids), "identities"))
    manifest = {"files": {r: reproduced[r].record() for r in (*STAGE1, *STAGE3)}}
    manifest["files"].update({s.role: s.record() for s in STAGE2_PINS})
    arts, paths = {}, {}
    for key in S.ROOT_KEYS:
        paths[key] = reproduced[key].resolve(root)

        def loaded(key=key):
            arts[key] = audit.load_arbiter(key, paths[key], split)
            return True

        if not g4.check(f"A6.4 reproduced {key} loads on this split (schema, split identity, "
                        "test index)", loaded):
            continue
        g4.check(f"A6.4 reproduced {key} is the {S.ROOT_UPSTREAM[key]}-upstream cell it records "
                 "(artifact_audit.check_arbiter_identity)",
                 lambda key=key: audit.check_arbiter_identity(key, arts[key],
                                                              generation=audit.GENERATION) is None)
        g4.check(f"A6.4 reproduced {key}: every Stage 1-3 file it consumed is this run's "
                 "(Stages 1 and 3: this reproduction's versions; Stage 2: the accepted pins)",
                 lambda key=key: bool(R.upstream_resolution(arts[key], S.ROOT_UPSTREAM[key],
                                                            manifest, root=root)))
    out: Dict[str, Any] = {"contract": contract, "identities": ids}
    if sorted(arts) != sorted(S.ROOT_KEYS):
        out["roles"] = roles
        return out
    g4.check("A6.4 the reproduced 2x2 holds each upstream fixed within a column and keeps the "
             "arms apart within a row (artifact_audit.check_crossover)",
             lambda: audit.check_crossover(arts, {k: str(p) for k, p in paths.items()},
                                           split) is not None)
    results = Rec.canonical(X.evaluate(split, arts))
    for key, cell in results["cells"].items():
        g4.check(f"A6.4 reproduced {key}: accounting identities hold, abstention never changes a "
                 "retained decision, its saved test metrics are the recount's",
                 lambda cell=cell: all(cell["identities"].values())
                 and cell["retained_decisions_unchanged"] and not cell["stored_metric_problems"],
                 lambda cell=cell: cell["stored_metric_problems"][:4] or "all hold")
    saved = (getattr(arts["meta_xgb"], "comparison", None) or {}).get("summary")
    redone = {k: results["comparisons"]["cascade"][k] for k in ("table", "paired", "bootstrap")}
    g4.check("A6.4 the reproduced cascade comparison is the one black-box Cell 36 saved in the "
             "reproduced Meta-XGB, value for value",
             lambda: saved is not None and Rec.canonical(saved) == Rec.canonical(redone))
    pins2 = {s.role: s.resolve(root).resolve() for s in STAGE2_PINS}
    want2 = {"glass": pins2["stage2.glass_router"], "blackbox": pins2["stage2.rf_router"]}
    y_train = np.asarray(split["y_train"]).astype(int)
    stage2, room = {}, {}
    for key in S.ROOT_KEYS:
        acc = audit.load_arbiter(key, accepted[key].resolve(root), split)
        arm = S.ROOT_UPSTREAM[key]
        fa, fr = _stage2_file(acc, root), _stage2_file(arts[key], root)
        stage2[key] = {"accepted": str(fa), "reproduced": str(fr), "pinned": str(want2[arm]),
                       "identical": _same_file(fa, want2[arm]) and _same_file(fr, want2[arm])}
        roles[key] = (Cmp.glass_arbiter(acc, arts[key], y_train) if key.startswith("glass")
                      else Cmp.meta_xgb(acc, arts[key]))
        room[key] = Cmp.confidence_room(key, acc)
        del acc
    base = R.require_passed(STAGE4_RUN, X.STEP)
    acc_results = base["results"]
    first = X._first_difference(acc_results, results)
    acc_st, rep_st = Cmp.statements(acc_results["comparisons"]), Cmp.statements(results["comparisons"])
    changes = Cmp.statement_changes(acc_st, rep_st)
    accounting = {}
    for key in S.ROOT_KEYS:
        a, b = acc_results["cells"][key], results["cells"][key]
        diff = {blk: {"accepted": a[blk]["confusion"], "reproduced": b[blk]["confusion"]}
                for blk in ("no_abstention", "retained_only") if a[blk]["confusion"] != b[blk]["confusion"]}
        accounting[key] = diff or "identical counts (no abstention and retained rows)"
    env = Rec.environment()
    envs = {}
    for run in (UPSTREAM_RUN, STAGE4_RUN):
        rr = Rec.read_record(run, "run_record") if run == UPSTREAM_RUN else \
            R.require_passed(run, "run_record")
        envs[run] = Cmp.environment_differences(env, rr["environment"])
    identical2 = all(v["identical"] for v in stage2.values())
    classification = Cmp.classify(roles, environment=envs, stage2_inputs_identical=identical2,
                                  statement_changes_=changes)
    g2.check("A6.2 the section 7.4 evaluation covers every reproduced artifact (Stages 1, 3 and 4; "
             "Stage 2 by its pinned inputs and routes), the environment, and every registered "
             "comparison statement",
             lambda: sorted(roles) == sorted((*STAGE1, *STAGE3, *S.ROOT_KEYS)) and bool(acc_st),
             lambda: f"{classification['vectors_compared']} vectors, "
                     f"{classification['thresholds_compared']} thresholds, "
                     f"{classification['decision_sets_compared']} decision sets, "
                     f"{len(acc_st)} statements")
    out.update(
        roles=roles, stage2_inputs=stage2, environment={"now": env, "differences": envs},
        registered_statements={"accepted": acc_st, "reproduced": rep_st, "changes": changes},
        crossover_results={"accepted_record": f"research_logs/pr39/{STAGE4_RUN}/{X.STEP}.json",
                           "accepted_sha256": _sidecar(Rec.RECORD_ROOT / STAGE4_RUN / f"{X.STEP}.json"),
                           "identical": first is None, "first_difference": first,
                           "reproduced": None if first is None else results},
        accounting_changes=accounting, confidence_room=room, classification=classification)
    return out


def compare(run_id: str, sums_path) -> Dict[str, Any]:
    """Section 7.4 (the evaluation, whatever the class), A6.3 and A6.4; ``reproduction.json``."""
    from shared.pinned_artifacts import read_pins, verify_pin
    pre = require_passed(run_id, "preflight")
    glass = require_passed(run_id, "execution_glass")
    black = require_passed(run_id, "execution_blackbox")
    code = _same_code(pre, analysis=False)
    step = "reproduction"
    if _record(run_id, step).exists():
        raise Rec.RecordError(f"run {run_id} already has a {step} record")
    root = Rec.BANK_ROOT
    rec = _header(run_id, step)
    gates = {g: C.Gate(g) for g in ("A6.2", "A6.3", "A6.4")}
    g2, g3, g4 = gates["A6.2"], gates["A6.3"], gates["A6.4"]
    accepted, reproduced = accepted_pins(), reproduced_pins(glass, black)
    for label, pins, gate in (("accepted", accepted, g3), ("reproduced", reproduced, g4)):
        for k, pin in sorted(pins.items()):
            gate.check(f"{gate.gate} {label} {k}: {pin.path} unchanged (SHA-256, size)",
                       lambda pin=pin: verify_pin(pin, root) is not None, pin.sha256[:16])
    g4.check("A6.4 both notebooks ran top to bottom in fresh kernels, every code cell once and in "
             "order", lambda: all(r["execution"]["completed"] and r["order"]["passed"]
                                  for r in (glass, black)))
    g4.check("A6.4 the black-box notebook read this run's GLASS LR, EBM and GLASS Arbiter by pin",
             lambda: {r: e["sha256"] for r, e in read_pins(run_dir(run_id) / PINS_FILE)["files"].items()
                      if r in ("stage1.lr", "stage3.ebm", S.GLASS_ARBITER_ROLE)}
             == {"stage1.lr": reproduced["stage1.lr"].sha256,
                 "stage3.ebm": reproduced["stage3.ebm"].sha256,
                 S.GLASS_ARBITER_ROLE: reproduced["glass_arbiter"].sha256})
    result: Dict[str, Any] = {}
    if g3.passed and g4.passed:
        result = evaluate(accepted, reproduced, g2, g4)
    evidence = {"preservation_after": C.preservation(g3, sums_path)}
    after = I.snapshot(root)
    g3.check("nothing written to the model folders since the black-box execution",
             lambda: I.diff(black["inventory_after"], after) == {"added": [], "modified": [],
                                                                  "deleted": []})
    net = classify(pre["inventory"], after, RUN)
    g3.check("7.5.5 models/ before and after: only the reproduction's registered versions, "
             "sidecars and mutable aliases; every existing version unchanged",
             lambda: net["passed"], lambda: net["problems"][:8])
    lc = C.live_counterparts(evidence["preservation_after"].get("snapshot") or {})
    g3.check("7.5.1 the snapshot's live counterparts are as before the run",
             lambda: lc == pre["live_counterparts"],
             lambda: {"before": pre["live_counterparts"], "after": lc})
    g3.check("the accepted Commit 4 and Commit 5 records are the ones preflight verified",
             lambda: _references() == pre["references"]["sidecars"])
    status = Rec.git_status(root)
    allowed = [t for nb in TRACKED_WRITES for t in TRACKED_WRITES[nb]]
    git = _git_unexpected(pre["git_status"], status, run_id, allowed)
    g3.check("7.5.6 git status since preflight: only this run's records and the registered "
             "tracked notebook outputs", lambda: status is not None and not git["unexpected"],
             lambda: git["unexpected"][:8])
    analysis_changed = code["analysis"] != pre["code"]["analysis"]
    rec.update(gates={g: x.record() for g, x in gates.items()},
               records={n: _sidecar(_record(run_id, n)) for n in
                        ("preflight", "execution_glass", "execution_blackbox")},
               accepted={k: p.record() for k, p in sorted(accepted.items())},
               reproduced={k: p.record() for k, p in sorted(reproduced.items())},
               pins_file={"path": _rel(run_dir(run_id) / PINS_FILE),
                          "sha256": _sidecar(run_dir(run_id) / PINS_FILE)},
               reproduction=result, evidence=evidence, inventory_net=net, git_status_delta=git,
               code=code, analysis_code={"changed_since_preflight": analysis_changed,
                                         "preflight": pre["code"]["analysis"],
                                         "now": code["analysis"]})
    return _finish(run_id, step, rec, all(g.passed for g in gates.values()))


# ----------------------------------------------------------------------
# A6.1, A6.2 and A6.5: the Phase 1 freeze record
# ----------------------------------------------------------------------
def read_statements(path, cls: str) -> Dict[str, str]:
    """The owner's statements, recorded verbatim; the A6.2 exception iff Not reproduced."""
    p = Path(path)
    if not p.is_file():
        raise Rec.RecordError(f"statements file {p} does not exist")
    raw = p.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"))           # a BOM is tolerated, not recorded
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rec.RecordError(f"statements file {p} is not UTF-8 JSON: {exc}") from None
    want = set(STATEMENT_KEYS) | ({EXCEPTION_KEY} if cls == "Not reproduced" else set())
    if not isinstance(data, dict) or set(data) != want:
        raise Rec.RecordError(f"{path}: expected exactly {sorted(want)} (class {cls}), got "
                              f"{sorted(data) if isinstance(data, dict) else type(data).__name__}")
    for k, v in data.items():
        if not isinstance(v, str) or not v.strip() or len(v) > MAX_STATEMENT:
            raise Rec.RecordError(f"{path}: {k} must be non-empty text of at most "
                                  f"{MAX_STATEMENT} characters")
    return {k: v.strip() for k, v in data.items()}


def _report(run_id: str, suite: str) -> Dict[str, Any]:
    path = run_dir(run_id) / REPORTS[suite]
    data = path.read_bytes()
    return {"file": REPORTS[suite], "sha256": hashlib.sha256(data).hexdigest(),
            "summary": R.junit_summary(data)}


LIMITATIONS = (
    "The test split is previously exposed (PR 38 protocol section 15; contract L1, section 12). "
    "Every Stage 4 and crossover result is descriptive; PR 39 claims only what contract section "
    "12 allows.",
    "Abstention, both arbiters: the registered cut maximises F2 on retained training rows subject "
    "to coverage >= 0.50 (shared.stage4.tune_min_confidence). F2 has no TN term, so retaining a "
    "correct negative never raises it, and abstention can fall mostly on correct negatives: "
    "retained precision and recall rise while retained accuracy falls, and retained recall "
    "overstates effective recall. The Commit 5 accounting reports this per cell; the protocol is "
    "unchanged.",
    "Correction (supersedes the third limitation of the Commit 5 crossover_baseline.json, which "
    "is not rewritten). That item attributes the concentration of abstention on negatives in both "
    "arbiters to a threshold-relative confidence bound. The bound is structural for the GLASS "
    "Arbiter only: a negative vote's confidence is at most its voter's threshold (the Stage 1 and "
    "Stage 3 recall cuts, the router's base rate for Pass 1), so a negative decision's weighted "
    "confidence cannot exceed the largest of them while a positive decision's can approach one "
    "minus the smallest (values: reproduction.json, confidence_room). Meta-XGB's confidence is "
    "|p - t| in its raw class-weighted space at one operating threshold t, so its sides have room "
    "t and 1 - t (recorded); its abstention pattern needs its own interpretation from its score "
    "distribution under the F2 criterion above, and this record makes no claim about its cause.",
    "The coverage floor binds on training rows only; test coverage is reported against it per "
    "cell and can fall below it.",
    "The GLASS Arbiter's score is a weighted average of upstream probabilities: it ranks (ROC-AUC, "
    "PR-AUC) but is not a calibrated probability, and it is not the arbiter's decision function.",
    "The existing metric code's zero-division conventions (0, and ROC-AUC 0.5 on one class) are "
    "reported as undefined by the accounting; no stored value of the accepted arbiters relies on "
    "them (A5.4).",
    "The bootstrap resamples test rows with the models fixed; it does not reflect training "
    "variability.",
    "Stage 2's role in the final cascade is descriptive (F10); the PR 38 Tier 2 results stay tied "
    "to the historical upstream. Stage 4 stacking keeps an indirect upstream dependence (L7, F39).",
    "The pytest 'notebook' marker selects no tests. The notebook evidence is this run's two "
    "fresh-kernel, top-to-bottom executions.",
    "The suites emit recurring scikit-learn MLP fixture warnings; their cleanup is PR 40.",
)
PHASE2_HANDOFF = (
    "Notebook restructuring and presentation: condensed outputs and stage summaries, final "
    "slicing and organisation, the cascade trace figure (contract section 9, Phase 2).",
    "Baseline presentation: the frozen Phase 1 baseline is Commit 4's Stage 1-3 artifacts and "
    "Commit 5's four arbiters, read by manifest, with the Commit 5 crossover baseline record.",
    "Research comparisons not registered in PR 39: the 2x2 rows (upstream effect), an interaction "
    "estimate, coverage grids and calibrated-space confidence tests (PR 43).",
    "Stage 4 training and abstention optimisation: an abstention policy and whole-system metrics, "
    "test coverage against the training floor, and an analysis of Meta-XGB's abstention pattern.",
    "Paper preparation: out-of-sample claims need independent validation (Tier 3); PR 39 results "
    "are descriptive on a previously exposed split.",
)


def freeze(run_id: str, statements_path) -> Dict[str, Any]:
    """A6.1-A6.5: the Phase 1 freeze record ``phase1_freeze.json``."""
    pre = require_passed(run_id, "preflight")
    glass = require_passed(run_id, "execution_glass")
    black = require_passed(run_id, "execution_blackbox")
    rep = require_passed(run_id, "reproduction")
    code = _same_code(pre, analysis=False)
    step = "phase1_freeze"
    if _record(run_id, step).exists():
        raise Rec.RecordError(f"run {run_id} already has a {step} record")
    if code["analysis"] != rep["code"]["analysis"]:
        raise Rec.RecordError(f"run {run_id}: this package changed since the reproduction record; "
                              "freeze records what compare evaluated")
    cls = rep["reproduction"]["classification"]["class"]
    statements = read_statements(statements_path, cls)
    root = Rec.BANK_ROOT
    rec = _header(run_id, step)
    gates = {g: C.Gate(g) for g in ("A6.1", "A6.2", "A6.3", "A6.4", "A6.5")}
    reports = {}
    for suite in REPORTS:
        def saved(suite=suite):
            reports[suite] = _report(run_id, suite)
            return True

        if not gates["A6.1"].check(f"A6.1 {suite}: saved report {REPORTS[suite]} (absent at "
                                   "preflight, so run on this code)", saved):
            continue
        s = reports[suite]["summary"]
        if suite == "notebook":
            gates["A6.1"].check("A6.1 notebook: the marker's run is recorded (no failures or "
                                "errors)", lambda s=s: s["failures"] == s["errors"] == 0, s)
        else:
            gates["A6.1"].check(f"A6.1 {suite}: tests ran, no failures or errors"
                                + (", none skipped" if suite == "realdata" else ""),
                                lambda s=s, suite=suite: s["tests"] > 0 and s["failures"] == 0
                                and s["errors"] == 0 and (suite != "realdata" or s["skipped"] == 0),
                                s)
    exception = statements.get(EXCEPTION_KEY)
    gates["A6.2"].check(
        "A6.2 " + ("reproduced: " + cls if exception is None else
                   "environmental exception: Not reproduced, with the root-cause investigation and "
                   "the owner's approval recorded (section 7.4)"),
        lambda: cls in Cmp.CLASSES[:2] or bool(exception), cls)
    g3 = gates["A6.3"]
    g3.check("A6.3 the reproduction record's section 7.5 checks passed",
             lambda: rep["gates"]["A6.3"]["passed"] is True)
    now = I.snapshot(root)
    g3.check("A6.3 the model folders are as the black-box execution left them",
             lambda: I.diff(black["inventory_after"], now) == {"added": [], "modified": [],
                                                                "deleted": []})
    g3.check("A6.3 the accepted Commit 4 and Commit 5 records are the ones preflight verified",
             lambda: _references() == pre["references"]["sidecars"])
    g3.check("A6.3 the superseded Step 6B attempts are as preflight hashed them",
             lambda: superseded() == pre["superseded_runs"])
    writes = {**glass["tracked_writes"], **black["tracked_writes"]}
    current = _tracked(writes)
    def state(w, now):
        if w["changed"] and now == w["produced"]:
            return "as produced (committed with Commit 6, D7)"
        if now == w["before"]:
            return "restored to HEAD" if w["changed"] else "unchanged by the run"
        return "other"

    disposition = {rel: {"before": w["before"], "produced": w["produced"], "now": current[rel],
                         "copy": w.get("copy"), "state": state(w, current[rel])}
                   for rel, w in writes.items()}
    g3.check("A6.3 each tracked notebook output is as produced or as before the run (nothing "
             "else)", lambda: all(d["state"] != "other" for d in disposition.values()),
             lambda: {k: d["state"] for k, d in disposition.items()})
    status = Rec.git_status(root)
    git = _git_unexpected(pre["git_status"], status, run_id, list(writes))
    g3.check("A6.3 git status since preflight: only this run's records and the tracked notebook "
             "outputs", lambda: status is not None and not git["unexpected"],
             lambda: git["unexpected"][:8])
    g4 = gates["A6.4"]
    g4.check("A6.4 both notebooks ran top to bottom from fresh kernels, every code cell once and in "
             "order, loading upstream artifacts by pin", lambda: all(
                 r["execution"]["completed"] and r["order"]["passed"] for r in (glass, black)))
    g4.check("A6.4 the reproduced artifacts pass the Stage 1-4 contract checks (reproduction "
             "record)", lambda: rep["gates"]["A6.4"]["passed"] is True)
    gates["A6.5"].check("A6.5 the freeze record holds the baseline, the regression reports, both "
                        "executions, the classification, the limitations and the Phase 2 handoff",
                        lambda: bool(reports) and bool(rep["accepted"]))
    limitations = list(LIMITATIONS)
    if exception is not None:
        limitations.append(f"A6.2 environmental exception (owner disposition): {exception}")
    limitations.append(f"Tracked notebook outputs (owner disposition): {statements['tracked_outputs']}")
    rr = rep["reproduction"]
    rec.update(
        gates={g: x.record() for g, x in gates.items()},
        decision={"statements": statements, "statements_file": {
                      "name": Path(statements_path).name,
                      "sha256": hashlib.sha256(Path(statements_path).read_bytes()).hexdigest()},
                  "reproduction_class": cls,
                  "a6_2_route": "reproduced" if exception is None else "environmental exception"},
        baseline={"commit4": {"run_id": UPSTREAM_RUN, "records": pre["references"]["sidecars"]["commit4"]},
                  "commit5": {"run_id": STAGE4_RUN, "records": pre["references"]["sidecars"]["commit5"]},
                  "pr38_stage2_run": C.PR38_RUN_ID, "artifacts": rep["accepted"],
                  "arbiter_manifest": f"research_logs/pr39/{STAGE4_RUN}/{R.ARBITER_MANIFEST}.json"},
        reproduction={"record": "reproduction.json", "sha256": _sidecar(_record(run_id, "reproduction")),
                      "class": cls, "conditions": {k: v["holds"] for k, v in
                                                   rr["classification"]["conditions"].items()},
                      "max_abs_drift": rr["classification"]["max_abs_drift"],
                      "artifacts": rep["reproduced"], "pins_file": rep["pins_file"]},
        regression=reports, corrections=pre["corrections"],
        superseded_runs=pre["superseded_runs"],
        executions={nb: {"record": f"execution_{nb}.json", "sha256": _sidecar(_record(run_id, f"execution_{nb}")),
                         "wall_seconds": r["execution"]["wall_seconds"],
                         "kernel": r["execution"]["kernel"],
                         "input_copy": r["execution"]["input_copy"],
                         "executed_copy": r["execution"]["executed_copy"],
                         "code_cells": r["full_copy"]["cells"], "order": r["order"]}
                    for nb, r in (("glass", glass), ("blackbox", black))},
        tracked_outputs=disposition,
        preservation={"before": pre["gates"]["preflight"]["evidence"],
                      "after": rep["gates"]["A6.3"]["evidence"]},
        confidence_room=rr.get("confidence_room"), limitations=limitations,
        limitation_correction={"supersedes": f"research_logs/pr39/{STAGE4_RUN}/{X.STEP}.json, "
                                             "limitations[2]",
                               "by": "limitations[2] of this record"},
        phase2_handoff=list(PHASE2_HANDOFF), git_status_delta=git, code=code,
        environment=Rec.environment())
    return _finish(run_id, step, rec, all(g.passed for g in gates.values()))


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    from . import __doc__ as doc
    ap = argparse.ArgumentParser(prog="python -m phase1_reproduction", description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="the accepted baseline and section 7.5, before any write")
    p.add_argument("--run-id", required=True, help="c6_<YYYYMMDD_HHMMSS>")
    p.add_argument("--sums", required=True, help="the baseline snapshot's SHA256SUMS.txt")
    p = sub.add_parser("execute", help="section 5.4: one notebook, top to bottom, fresh kernel")
    p.add_argument("--run-id", required=True)
    p.add_argument("--notebook", required=True, choices=sorted(V.NOTEBOOKS))
    p.add_argument("--out-dir", required=True, help="outside the repository")
    p.add_argument("--kernel", default=None, help="Jupyter kernel name (default: the notebook's)")
    p = sub.add_parser("compare", help="section 7.4, A6.3-A6.4: reproduction.json")
    p.add_argument("--run-id", required=True)
    p.add_argument("--sums", required=True)
    p = sub.add_parser("freeze", help="A6.1-A6.5: phase1_freeze.json")
    p.add_argument("--run-id", required=True)
    p.add_argument("--statements", required=True,
                   help=f"UTF-8 JSON with {list(STATEMENT_KEYS)} (and {EXCEPTION_KEY!r} only "
                        "when the class is Not reproduced), recorded verbatim")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "preflight":
            rec = preflight(args.run_id, args.sums)
        elif args.cmd == "execute":
            rec = execute(args.run_id, args.notebook, args.out_dir, kernel=args.kernel)
        elif args.cmd == "compare":
            rec = compare(args.run_id, args.sums)
        else:
            rec = freeze(args.run_id, args.statements)
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except (Rec.RecordError, V.VehicleError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print(f"OK {rec['step']} for run {rec['run_id']}: wrote {rec.get('path')}")
    for name, g in (rec.get("gates") or {}).items():
        print(f"  {name}: {g['evidence']}")
    if rec["step"].startswith("execution_"):
        print(f"  wall {rec['execution']['wall_seconds']} s; produced "
              + ", ".join(f"{k} {v['sha256'][:12]}" for k, v in rec["produced"].items()))
        for rel, w in rec["tracked_writes"].items():
            print(f"  tracked {rel}: {'rewritten, copy ' + w['copy']['path'] if w['changed'] else 'unchanged'}")
    if rec["step"] == "reproduction":
        c = rec["reproduction"]["classification"]
        print(f"  class: {c['class']} (max |delta| {c['max_abs_drift']})")
        for k, v in c["conditions"].items():
            print(f"    {'holds ' if v['holds'] else 'FAILS '} {k}"
                  + ("" if v["holds"] else f": {v['failing'][:3]}"))
        print(f"  registered statements changed: {len(rec['reproduction']['registered_statements']['changes'])}")
        print(f"  crossover results identical: {rec['reproduction']['crossover_results']['identical']}")
    if rec["step"] == "phase1_freeze":
        for suite, r in rec["regression"].items():
            print(f"  {suite}: {r['summary']}")
        for rel, d in rec["tracked_outputs"].items():
            print(f"  {rel}: {d['state']}")
    return 0
