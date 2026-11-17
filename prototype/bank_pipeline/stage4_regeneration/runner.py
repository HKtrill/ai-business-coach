"""
stage4_regeneration.runner
==========================
The Step 5A steps and their records (package docstring). Each step requires
the previous steps of the same run to have passed and the code to be the code
the run started on (``code_identity``). It writes its record once when every
check passed. Otherwise it writes ``failed/<step>_<utc>.json`` (Git-ignored,
never read by code) and stops; a new attempt gets a new run id.

The Commit 4 helpers are reused unchanged: ``upstream_regeneration.records``
(canonical JSON, the code and environment identities, ``git status``),
``checks`` (``Gate``, the section 7.5 checks, the split and the section 7.3
identities), ``inventory`` (the model-folder snapshot), ``vehicle`` (the
fresh-kernel execution) and ``acceptance.check`` (the Commit 4 acceptance and
its manifest, re-verified).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional

from upstream_regeneration import acceptance as Acc
from upstream_regeneration import checks as C
from upstream_regeneration import inventory as I
from upstream_regeneration import records as Rec
from upstream_regeneration import vehicle as V

from . import stage4_copy as S

__all__ = ["RUN_ID", "UPSTREAM_RUN", "UPSTREAM_ROLES", "UPSTREAM_ROLE_OF", "PINS_FILE",
           "ARBITER_MANIFEST", "AUDIT_FILE", "AUDIT_TESTS", "CONTRACT", "StepFailed",
           "new_run_id", "check_run_id", "run_dir", "write_record", "write_failed",
           "read_record", "require_passed", "code_identity", "manifest_path", "run_pins",
           "upstream_resolution", "junit_summary", "preflight", "execute", "finalize", "main"]

BANK_ROOT = Rec.BANK_ROOT
#: Commit 5 run ids (section 7.2): c5_<YYYYMMDD_HHMMSS>.
RUN_ID = re.compile(r"^c5_[0-9]{8}_[0-9]{6}$")
#: The accepted Commit 4 run (A4.7); Stage 4 reads its artifact manifest (A5.1).
UPSTREAM_RUN = "c4_20261115_204125"
UPSTREAM_ROLES = tuple(Acc.ROLES)
#: Upstream arm -> stage -> the role that fills it (shared.stage4.upstream.UPSTREAM).
UPSTREAM_ROLE_OF = {
    "glass": {"stage1": "stage1.lr", "stage2": "stage2.glass_router", "stage3": "stage3.ebm"},
    "blackbox": {"stage1": "stage1.mlp", "stage2": "stage2.rf_router", "stage3": "stage3.xgb"},
}
PINS_FILE = "upstream_pins.json"
ARBITER_MANIFEST = "arbiter_manifest"
AUDIT_FILE = "stage4_audit_pytest.xml"
#: The real-data audit of the four arbiters (PR 36 Commit 4, P5): gate A5.3.
AUDIT_TESTS = "tests/realdata/test_saved_artifacts.py"
CONTRACT = {"record": Rec.CONTRACT["record"], "version": Rec.CONTRACT["version"],
            "sections": ["5.1", "5.4", "6.3", "6.4", "6.5", "7.1", "7.2", "7.3", "7.4", "7.5"],
            "gates": ["A5.1", "A5.2", "A5.3"], "decisions": ["D4", "D5"]}


class StepFailed(RuntimeError):
    """A step's checks failed; its failure record is named."""


# ----------------------------------------------------------------------
# run ids and write-once records (sections 7.1, 7.2)
# ----------------------------------------------------------------------
def new_run_id() -> str:
    return "c5_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise Rec.RecordError(f"run id {run_id!r} is not c5_<YYYYMMDD_HHMMSS> (section 7.2)")
    return run_id


def run_dir(run_id: str) -> Path:
    return Rec.RECORD_ROOT / check_run_id(run_id)


def _text(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(Rec.canonical(record), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def _write_once(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise Rec.RecordError(f"{path} exists: records are written once (a new attempt gets a "
                              "new run id)") from None
    with open(path.with_name(path.name + ".sha256"), "x", encoding="ascii") as fh:
        fh.write(hashlib.sha256(data).hexdigest() + "\n")
    return path


def write_record(run_id: str, name: str, record: Mapping[str, Any]) -> Path:
    """``<run>/<name>.json`` and its ``.sha256`` sidecar, once each."""
    return _write_once(run_dir(run_id) / f"{name}.json", _text(record))


def write_failed(run_id: str, name: str, record: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = run_dir(run_id) / "failed" / f"{name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(_text(record))
    return path


def _verified_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise Rec.RecordError(f"no record at {path}: run that step first")
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.is_file() or side.read_text(encoding="ascii").strip() != \
            hashlib.sha256(data).hexdigest():
        raise Rec.RecordError(f"{path} does not match its {side.name} sidecar")
    return data


def read_record(run_id: str, name: str) -> Dict[str, Any]:
    rec = json.loads(_verified_bytes(run_dir(run_id) / f"{name}.json").decode("utf-8"))
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


def _rel(path: Path) -> str:
    p = Path(path)
    return p.relative_to(BANK_ROOT).as_posix() if p.is_relative_to(BANK_ROOT) else str(p)


def code_identity() -> Dict[str, Any]:
    """``records.code_identity`` (section 7.3) and a digest of this package's own files."""
    base = Rec.code_identity(BANK_ROOT)
    h, n = hashlib.sha256(), 0
    for p in sorted((BANK_ROOT / "stage4_regeneration").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(BANK_ROOT).as_posix().encode("utf-8") + b"\0")
        h.update(bytes.fromhex(Rec.file_sha256_lf(p)))
        n += 1
    return {**base, "stage4_regeneration_sha256": h.hexdigest(), "stage4_regeneration_files": n}


def _same_code(pre: Mapping[str, Any]) -> Dict[str, Any]:
    now, want = code_identity(), pre["code"]
    keys = ("head", "code_sha256", "stage4_regeneration_sha256")
    if any(now.get(k) != want.get(k) for k in keys):
        raise Rec.RecordError(f"run {pre['run_id']}: the code changed since preflight "
                              f"({want['code_sha256'][:16]} -> {now['code_sha256'][:16]}); a run "
                              "uses one code identity: start a new run id")
    return now


def _header(run_id: str, step: str) -> Dict[str, Any]:
    return {"format": f"pr39.commit5a.{step}/1", "run_id": run_id, "step": step,
            "contract": CONTRACT, "upstream_run": UPSTREAM_RUN, "started_utc": Rec.now()}


def _finish(run_id: str, step: str, rec: Dict[str, Any], passed: bool) -> Dict[str, Any]:
    rec.update(finished_utc=Rec.now(), passed=bool(passed))
    if passed:
        rec["path"] = _rel(write_record(run_id, step, rec))
        return rec
    path = write_failed(run_id, step, rec)
    failed = [f"{g}: {c}" for g, r in (rec.get("gates") or {}).items() if isinstance(r, dict)
              for c in (r.get("failed") or [])]
    raise StepFailed(f"{step} did not pass; recorded in {path}" +
                     "".join(f"\n  - {f}" for f in failed[:20]))


# ----------------------------------------------------------------------
# A5.1: the accepted upstream
# ----------------------------------------------------------------------
def manifest_path() -> Path:
    """The accepted Commit 4 artifact manifest (a ``pr39.upstream_pins/1`` file)."""
    return Rec.record_path(UPSTREAM_RUN, Acc.MANIFEST)


def _accepted_upstream(gate: C.Gate) -> Dict[str, Any]:
    from shared.pinned_artifacts import read_pins
    out: Dict[str, Any] = {}

    def accepted():
        out.update(Acc.check(UPSTREAM_RUN))
        return out["accepted"] is True

    gate.check(f"A5.1 the Commit 4 acceptance of {UPSTREAM_RUN} and its artifact manifest "
               "re-verify (acceptance.check: sidecars, links, every role on disk by SHA-256)",
               accepted, lambda: out["manifest_sha256"])
    gate.check("A5.1 the manifest pins exactly the six Stage 1-3 roles (no Stage 4 role)",
               lambda: sorted(read_pins(manifest_path())["files"]) == sorted(UPSTREAM_ROLES),
               lambda: sorted(read_pins(manifest_path())["files"]))
    return out


def run_pins(manifest: Mapping[str, Any], glass_arbiter: Mapping[str, Any]) -> Dict[str, Any]:
    """The run's pins: the manifest's six roles verbatim and the GLASS Arbiter just produced."""
    from shared.pinned_artifacts import PinnedFile
    files = {role: PinnedFile(role, str(e["path"]), str(e["sha256"]), int(e["bytes"]))
             for role, e in (manifest.get("files") or {}).items() if role in UPSTREAM_ROLES}
    if sorted(files) != sorted(UPSTREAM_ROLES):
        raise Rec.RecordError(f"the manifest pins {sorted(files)}, not {sorted(UPSTREAM_ROLES)}")
    files[S.GLASS_ARBITER_ROLE] = PinnedFile(S.GLASS_ARBITER_ROLE, str(glass_arbiter["path"]),
                                             str(glass_arbiter["sha256"]),
                                             int(glass_arbiter["bytes"]))
    return files


def upstream_resolution(art, arm: str, manifest: Mapping[str, Any], *,
                        root: Optional[Path] = None) -> Dict[str, str]:
    """
    ``{stage: role}`` when every Stage 1-3 file the arbiter recorded consuming
    (``stage4_inputs["upstream"]``) is the manifest's pinned file for that
    role: the same file on disk, with the pinned size and SHA-256 (A5.1).
    """
    from shared.pinned_artifacts import PinnedFile, verify_pin
    root = BANK_ROOT if root is None else Path(root)
    fields = art if isinstance(art, dict) else vars(art)
    rec = (fields.get("stage4_inputs") or {}).get("upstream") or {}
    if rec.get("arm") != arm:
        raise AssertionError(f"the arbiter records upstream {rec.get('arm')!r}, not {arm!r}")
    out = {}
    for stage, role in UPSTREAM_ROLE_OF[arm].items():
        recorded = ((rec.get("stages") or {}).get(stage) or {}).get("artifact_path")
        if not recorded:
            raise AssertionError(f"{stage}: no artifact_path recorded")
        e = manifest["files"][role]
        pinned = verify_pin(PinnedFile(role, e["path"], e["sha256"], int(e["bytes"])), root)
        used = Path(recorded) if Path(recorded).is_absolute() else Path(root) / recorded
        if not used.is_file() or not os.path.samefile(used, pinned):
            raise AssertionError(f"{stage}: read {recorded}, not the pinned {role} {e['path']}")
        out[stage] = f"{role} {e['sha256'][:16]}"
    return out


# ----------------------------------------------------------------------
# A5.3: the real-data audit
# ----------------------------------------------------------------------
def junit_summary(data: bytes) -> Dict[str, int]:
    """Totals of a pytest ``--junitxml`` report."""
    root = ET.fromstring(data)
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    keys = ("tests", "failures", "errors", "skipped")
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in keys}


def _audit(manifest_file: Path, sums_path) -> Dict[str, Any]:
    """``pytest -m realdata AUDIT_TESTS`` with ``$BANK_ARTIFACT_MANIFEST`` = this run's manifest."""
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "audit.xml"
        args = ["-m", "pytest", "-m", "realdata", AUDIT_TESTS, "-q", "-p", "no:cacheprovider",
                f"--junitxml={xml}"]
        env = {**os.environ, "BANK_ARTIFACT_MANIFEST": str(manifest_file),
               "PR37_BASELINE_SHA256SUMS": str(sums_path), "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, *args], cwd=BANK_ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        data = xml.read_bytes() if xml.is_file() else b""
    summary = junit_summary(data) if data else None
    passed = (proc.returncode == 0 and summary is not None and summary["tests"] > 0
              and summary["failures"] == summary["errors"] == summary["skipped"] == 0)
    return {"command": "python " + " ".join(args[:-1]) + " --junitxml=<run>/" + AUDIT_FILE,
            "env": {"BANK_ARTIFACT_MANIFEST": _rel(manifest_file)},
            "exit_code": proc.returncode, "summary": summary, "passed": passed,
            "output_tail": (proc.stdout + proc.stderr).strip().splitlines()[-15:],
            "junit": data}


# ----------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------
def _layout(run_id: str) -> Dict[str, Any]:
    return {"records": f"research_logs/pr39/{run_id}/",
            "upstream_manifest": _rel(manifest_path()),
            "pins_file": f"research_logs/pr39/{run_id}/{PINS_FILE}",
            "arbiter_manifest": f"research_logs/pr39/{run_id}/{ARBITER_MANIFEST}.json",
            "audit": f"research_logs/pr39/{run_id}/{AUDIT_FILE}",
            "failed_attempts": f"research_logs/pr39/{run_id}/failed/ (Git-ignored, never read)",
            "executed_notebooks": "outside the repository (--out-dir); SHA-256 recorded",
            "outputs": {nb: dict(S.OUTPUTS[nb]) for nb in S.OUTPUTS},
            "cells": {nb: {"split": list(s["split"]), "stage4": list(s["stage4"]),
                           "not_run": list(s["reserved"])} for nb, s in S.NOTEBOOKS.items()}}


def preflight(run_id: str, sums_path) -> Dict[str, Any]:
    """The accepted upstream (A5.1) and the section 7.5 checks, before anything is written."""
    check_run_id(run_id)
    if (run_dir(run_id) / "preflight.json").exists():
        raise Rec.RecordError(f"run {run_id} already has a preflight record: new run id")
    rec = _header(run_id, "preflight")
    gate = C.Gate("preflight")
    rec["upstream"] = _accepted_upstream(gate)
    rec["evidence"] = C.preservation(gate, sums_path)
    inv = I.snapshot(BANK_ROOT)
    gate.check("7.5.5 SHA-256 inventory of models/ and blackbox_pipeline/models/ taken before "
               "any write", lambda: bool(inv), f"{len(inv)} files")
    status = Rec.git_status(BANK_ROOT)
    gate.check("7.5.6 git status recorded (the applied Step 5A patch only)",
               lambda: status is not None, f"{len(status or [])} entries")
    rec.update(gates={"preflight": gate.record()}, inventory=inv, git_status=status,
               live_counterparts=C.live_counterparts(rec["evidence"].get("snapshot") or {}),
               layout=_layout(run_id), code=code_identity(), environment=Rec.environment(),
               sums={"path": str(sums_path)})
    return _finish(run_id, "preflight", rec, gate.passed)


def execute(run_id: str, notebook: str, out_dir, *, kernel: Optional[str] = None) -> Dict[str, Any]:
    """One fresh-kernel execution of a notebook's Stage 4 copy (``stage4_copy``)."""
    from shared.pinned_artifacts import PINS_ENV, read_pins, write_pins
    pre = require_passed(run_id, "preflight")
    code = _same_code(pre)
    step = f"execution_{notebook}"
    if (run_dir(run_id) / f"{step}.json").exists():
        raise Rec.RecordError(f"run {run_id} already executed the {notebook} Stage 4 cells")
    pins_path = run_dir(run_id) / PINS_FILE
    if notebook == "glass":
        expected_before, pins_read = pre["inventory"], manifest_path()
    elif notebook == "blackbox":
        glass = require_passed(run_id, "execution_glass")
        expected_before, pins_read = glass["inventory_after"], pins_path
    else:
        raise V.VehicleError(f"unknown notebook {notebook!r}: expected one of {sorted(S.NOTEBOOKS)}")
    rec = _header(run_id, step)
    gate = C.Gate(f"execution [{notebook}]")
    out_dir = V.require_outside(out_dir, BANK_ROOT)
    nb, copy_rec = S.stage4_copy(notebook, root=BANK_ROOT)
    rec["stage4_copy"] = copy_rec
    roles = S.roles_read(notebook, root=BANK_ROOT)
    rec["pins_read"] = {"path": _rel(pins_read), "roles": roles}
    gate.check(f"A5.1 the pins file the copy reads ({_rel(pins_read)}) matches its sidecar and "
               "pins every role the copy reads", lambda: set(roles) <= set(read_pins(pins_read)["files"]),
               lambda: sorted(set(roles) - set(read_pins(pins_read)["files"])) or "all pinned")
    before = I.snapshot(BANK_ROOT)
    gate.check("the model folders are as the previous step left them",
               lambda: I.diff(expected_before, before) == {"added": [], "modified": [], "deleted": []},
               lambda: {k: v[:5] for k, v in I.diff(expected_before, before).items()})
    if not gate.passed:
        rec.update(gates={"execution": gate.record()}, code=code)
        return _finish(run_id, step, rec, False)
    run = V.execute(nb, notebook, out_dir, run_id, env={PINS_ENV: str(pins_read)},
                    kernel_name=kernel, root=BANK_ROOT)
    rec["execution"] = run
    gate.check("fresh-kernel execution completed with no cell error", lambda: run["completed"],
               run.get("error") or f"{run['wall_seconds']} s")
    after = I.snapshot(BANK_ROOT)
    delta = S.classify(before, after, notebook)
    rec.update(inventory_after=after, inventory_delta=delta)
    gate.check(f"only the {notebook} Stage 4 outputs were added; nothing modified or deleted",
               lambda: delta["passed"], lambda: delta["problems"][:8] or delta["produced"])
    if delta["passed"]:
        rec["produced"] = {k: {"path": rel, "bytes": after[rel]["bytes"],
                               "sha256": after[rel]["sha256"]}
                           for k, rel in delta["produced"].items()}
    if gate.passed and notebook == "glass":
        def pins():
            files = run_pins(read_pins(manifest_path()), rec["produced"]["glass_arbiter"])
            write_pins(pins_path, run_id, files, root=BANK_ROOT)
            rec["pins_file"] = {"path": _rel(pins_path), "sha256": _sidecar(pins_path),
                                "roles": sorted(files)}
            return True

        gate.check("the run's pins file written: the manifest's six Stage 1-3 roles verbatim and "
                   f"{S.GLASS_ARBITER_ROLE} (read by black-box Cell 35)", pins)
    if gate.passed and notebook == "blackbox":
        def manifest():
            glass_out = require_passed(run_id, "execution_glass")["produced"]
            entries = {"glass_arbiter": glass_out["glass_arbiter"]["path"],
                       **{k: v["path"] for k, v in rec["produced"].items()}}
            path = _write_once(run_dir(run_id) / f"{ARBITER_MANIFEST}.json",
                               (json.dumps(entries, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            rec["arbiter_manifest"] = {"path": _rel(path), "sha256": _sidecar(path),
                                       "entries": entries}
            return sorted(entries) == sorted(S.ROOT_KEYS)

        gate.check("the PR 39 arbiter manifest written: exactly the four artifact_audit.ROOTS "
                   "keys, each the version this run produced (section 7.1)", manifest)
    rec.update(gates={"execution": gate.record()}, code=code)
    return _finish(run_id, step, rec, gate.passed)


def _load(path: Path):
    import joblib
    return joblib.load(path)


def finalize(run_id: str, sums_path) -> Dict[str, Any]:
    """A5.1-A5.3, the section 7.5 checks after the run, and the run record."""
    from shared.pinned_artifacts import PinnedFile, read_pins, verify_pin
    from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
    pre = require_passed(run_id, "preflight")
    glass = require_passed(run_id, "execution_glass")
    black = require_passed(run_id, "execution_blackbox")
    code = _same_code(pre)
    if (run_dir(run_id) / "run_record.json").exists():
        raise Rec.RecordError(f"run {run_id} already has a run record")
    rec = _header(run_id, "run_record")
    rec["status"] = ("Stage 4 arbiters regenerated on the accepted Commit 4 upstream (Step 5A). "
                     "Not the Commit 5 baseline: Step 5B records the registered comparisons "
                     "(A5.4) and the baseline (A5.5)")
    gates = {g: C.Gate(g) for g in ("A5.1", "A5.2", "A5.3", "7.5")}
    g1, g2, g3, g5 = (gates[g] for g in ("A5.1", "A5.2", "A5.3", "7.5"))
    produced = {**glass["produced"], **black["produced"]}
    pins = {k: PinnedFile(k, v["path"], v["sha256"], int(v["bytes"])) for k, v in produced.items()}
    g2.check("A5.2 four arbiters built, one per 2x2 cell (artifact_audit.ROOTS keys)",
             lambda: sorted(pins) == sorted(S.ROOT_KEYS), sorted(pins))
    paths = {}
    for key, pin in sorted(pins.items()):
        if g2.check(f"A5.2 {key}: {pin.path} unchanged since it was produced (SHA-256, size)",
                    lambda pin=pin: verify_pin(pin, BANK_ROOT) is not None, pin.sha256[:16]):
            paths[key] = pin.resolve(BANK_ROOT)

    # A5.1: every Stage 4 input is the manifest's pin
    upstream = _accepted_upstream(g1)
    manifest = read_pins(manifest_path())
    pins_path = run_dir(run_id) / PINS_FILE

    def same_pins():
        files = read_pins(pins_path)["files"]
        want = {r: {k: v.record()[k] for k in ("path", "sha256", "bytes")}
                for r, v in run_pins(manifest, produced["glass_arbiter"]).items()}
        return {r: {k: e[k] for k in ("path", "sha256", "bytes")} for r, e in files.items()} == want

    g1.check("A5.1 the run's pins file holds the manifest's six roles verbatim and the GLASS "
             "Arbiter this run produced, nothing else", same_pins)
    identities, resolution = {}, {}
    for key, path in sorted(paths.items()):
        def resolved(key=key, path=path):
            art = _load(path)
            fields = art if isinstance(art, dict) else vars(art)
            up = (fields.get("stage4_inputs") or {}).get("upstream") or {}
            identities[key] = {"schema": fields.get("schema"), "combination": up.get("combination"),
                               "crossover": up.get("crossover")}
            resolution[key] = upstream_resolution(art, S.ROOT_UPSTREAM[key], manifest)
            return True

        g1.check(f"A5.1 {key}: each Stage 1-3 file it recorded consuming is the manifest's pin "
                 "(same file, size and SHA-256); no newest-file selection", resolved,
                 lambda key=key: resolution.get(key))

    # A5.2-A5.3: the real-data audit under this run's arbiter manifest
    mfile = run_dir(run_id) / f"{ARBITER_MANIFEST}.json"
    audit: Dict[str, Any] = {}

    def manifest_ok():
        entries = json.loads(_verified_bytes(mfile).decode("utf-8"))
        return entries == {k: produced[k]["path"] for k in S.ROOT_KEYS}

    g3.check("A5.3 the arbiter manifest matches its sidecar and names exactly the four versions "
             "this run produced", manifest_ok)

    def audited():
        audit.update(_audit(mfile, sums_path))
        return audit["passed"]

    g3.check(f"A5.2-A5.3 the real-data audit ({AUDIT_TESTS}) passes under the PR 39 arbiter "
             "manifest: each 2x2 cell's identity and evidence, its upstream reloaded through "
             "its own loader, fresh-interpreter reload, serving, and both columns compared "
             "with the upstream held fixed (require_same_upstream=True)", audited,
             lambda: audit.get("summary") or audit.get("output_tail"))

    # section 7.5 after the run
    evidence: Dict[str, Any] = {"preservation_after": C.preservation(g5, sums_path)}
    after = I.snapshot(BANK_ROOT)
    g5.check("nothing written to the model folders since the black-box execution",
             lambda: I.diff(black["inventory_after"], after) == {"added": [], "modified": [],
                                                                  "deleted": []})
    net = S.classify(pre["inventory"], after, S.RUN)
    g5.check("7.5.5 models/ inventory before and after: only the four arbiters and the two "
             "Meta-XGB summary sidecars added", lambda: net["passed"], lambda: net["problems"][:8])
    lc_after = C.live_counterparts(evidence["preservation_after"].get("snapshot") or {})
    g5.check("7.5.1 the snapshot's live counterparts are as before the run",
             lambda: lc_after == pre["live_counterparts"],
             lambda: {"before": pre["live_counterparts"], "after": lc_after})
    status = Rec.git_status(BANK_ROOT)
    new = sorted(set(status or []) - set(pre["git_status"] or []))
    gone = sorted(set(pre["git_status"] or []) - set(status or []))
    unexpected = [ln for ln in new if f"research_logs/pr39/{run_id}/" not in ln] + gone
    g5.check("7.5.6 git status: only this run's records are new", lambda: status is not None
             and not unexpected, lambda: unexpected[:8])
    g5.check("the two Stage 4 executions are recorded (section 5.4)",
             lambda: all(r["execution"]["completed"] for r in (glass, black)))

    split = C.global_split()
    ids = C.identities(split, C.frozen_population())
    feats = [str(f) for f in STAGE4_FEATURES]
    ids["feature_contracts"]["stage4_meta_xgb"] = {
        "ordered": feats, "sha256": hashlib.sha256(json.dumps(feats).encode("utf-8")).hexdigest()}
    passed = all(g.passed for g in gates.values())
    if passed:
        _write_once(run_dir(run_id) / AUDIT_FILE, audit["junit"])
    rec.update(
        gates={"preflight": {"record": "preflight.json",
                             "sha256": _sidecar(run_dir(run_id) / "preflight.json"),
                             "passed": True},
               **{g: gate.record() for g, gate in gates.items()}},
        upstream={"run_id": UPSTREAM_RUN, "artifact_manifest": {
            "path": _rel(manifest_path()), "sha256": upstream.get("manifest_sha256")},
            "acceptance_sha256": upstream.get("acceptance_sha256"), "roles": upstream.get("roles")},
        pins_file={"path": _rel(pins_path), "sha256": _sidecar(pins_path)},
        arbiters={k: {**pins[k].record(), "producer": S.PRODUCERS[k],
                      "upstream": S.ROOT_UPSTREAM[k], **identities.get(k, {}),
                      "upstream_resolution": resolution.get(k)} for k in sorted(pins)},
        arbiter_manifest={"path": _rel(mfile), "sha256": _sidecar(mfile),
                          "env": "set BANK_ARTIFACT_MANIFEST to this file for every real-data "
                                 "check of the PR 39 arbiters"},
        audit={k: v for k, v in audit.items() if k != "junit"} | (
            {"record": AUDIT_FILE, "sha256": hashlib.sha256(audit["junit"]).hexdigest()}
            if passed else {}),
        executions={nb: {"record": f"execution_{nb}.json",
                         "sha256": _sidecar(run_dir(run_id) / f"execution_{nb}.json"),
                         "kernel": r["execution"]["kernel"],
                         "input_copy": r["execution"]["input_copy"],
                         "executed_copy": r["execution"]["executed_copy"],
                         "copy_sha256": r["stage4_copy"]["copy_sha256"],
                         "stage4_cells": [c["cell"] for c in r["stage4_copy"]["stage4_cells"]],
                         "not_run": r["stage4_copy"]["not_run"],
                         "wall_seconds": r["execution"]["wall_seconds"]}
                    for nb, r in (("glass", glass), ("blackbox", black))},
        identities=ids, inventory_net=net,
        git_status_delta={"new": new, "gone": gone, "unexpected": unexpected},
        evidence=evidence, code=code, environment=Rec.environment())
    return _finish(run_id, "run_record", rec, passed)


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    from . import __doc__ as doc
    ap = argparse.ArgumentParser(prog="python -m stage4_regeneration", description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="A5.1 and section 7.5 before any write")
    p.add_argument("--run-id", default=None, help="default: c5_<local YYYYMMDD_HHMMSS>")
    p.add_argument("--sums", required=True, help="the baseline snapshot's SHA256SUMS.txt")
    p = sub.add_parser("execute", help="section 5.4: one notebook's Stage 4 cells, Stages 1-3 "
                                       "by pin")
    p.add_argument("--run-id", required=True)
    p.add_argument("--notebook", required=True, choices=sorted(S.NOTEBOOKS))
    p.add_argument("--out-dir", required=True, help="outside the repository")
    p.add_argument("--kernel", default=None, help="Jupyter kernel name (default: the notebook's)")
    p = sub.add_parser("finalize", help="A5.1-A5.3, section 7.5 after, and the run record")
    p.add_argument("--run-id", required=True)
    p.add_argument("--sums", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "preflight":
            rec = preflight(args.run_id or new_run_id(), args.sums)
        elif args.cmd == "execute":
            rec = execute(args.run_id, args.notebook, args.out_dir, kernel=args.kernel)
        else:
            rec = finalize(args.run_id, args.sums)
    except (StepFailed, Rec.RecordError, V.VehicleError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 1
    print(f"OK {rec['step']} for run {rec['run_id']}: wrote {rec.get('path')}")
    for name, g in (rec.get("gates") or {}).items():
        if isinstance(g, dict) and "evidence" in g:
            print(f"  {name}: {g['evidence']}")
    return 0
