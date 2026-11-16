"""
upstream_regeneration.recovery
==============================
Recovery of a completed GLASS execution that was refused only because one of
its legitimate outputs was not yet registered (PR 39 Commit 4A, run
c4_20261115_204125: the GLASS Cell 6 baseline LR,
``models/lr/lr_baseline_<YYYYMMDD_HHMMSS>.joblib``, now
``inventory.COMPANIONS``). Nothing is re-executed and no training cell runs.

    python -m upstream_regeneration recover-glass --run-id <id> [--failed-record <path>] [--copies-dir <dir>]

Every claim the failed attempt made is re-established from primary evidence,
not taken from the failed record (which has no sidecar):

* the run's ``preflight`` (A4.1) and ``stage2_reuse`` (A4.2, ``reuse``) records
  passed and match their sidecars; there is no ``execution_glass.json`` and no
  earlier recovery;
* the failed record is this run's GLASS attempt, made on the run's code; its
  notebook execution completed with no cell error; the model folders were
  unchanged before it; its only failed check is the inventory classification
  and each of its inventory problems is an addition this recovery registers
  (``RESOLVES``);
* the code differs from the run's code only by this recovery's own files:
  restoring each file in ``CHANGES`` to its recorded pre-recovery SHA-256
  reproduces the preflight code identity exactly, and every listed file is
  under ``upstream_regeneration/`` or ``tests/upstream_regeneration/``, so no
  file a notebook imports changed;
* the committed notebook's truncated copy is the one the attempt executed;
  the input copy's SHA-256 is the recorded one and the truncated copy's; the
  executed copy's SHA-256 is the recorded one, its code cells are the input's,
  every one executed with no error output, every legacy cell printed its skip,
  and every file the execution added is named in its outputs;
* the model folders are byte for byte as the attempt left them, and with the
  registered companion the change since preflight is accepted, with the same
  produced roles; the accepted Stage 2 files match their pins and the PR 38
  run passes ``require_accepted``;
* the LR, EBM and baseline reload in a fresh interpreter as their types.

Only then is ``execution_glass_recovery.json`` written once (with a sidecar),
naming the failed record by path and SHA-256 and recording both code
identities, and the run's pins file is written. Later steps read it in place of
``execution_glass.json`` and hold the code to the recorded recovered identity.
The full Stage 1-3 checks remain ``finalize``'s (A4.3-A4.5), unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import checks as C
from . import inventory as I
from . import records as Rec
from . import vehicle as V

__all__ = ["RECOVERY_ID", "STEP", "RESOLVES", "CHANGES", "ALLOWED_PREFIXES",
           "INVENTORY_FAILURE", "failed_glass_record", "recover_glass", "run_code",
           "glass_execution"]

RECOVERY_ID = "commit4a-r1: GLASS execution, unregistered Cell 6 baseline LR"
STEP = "execution_glass_recovery"
#: The inventory problems this recovery may resolve: additions of exactly these patterns.
RESOLVES = (I.COMPANIONS["glass"]["stage1.lr_baseline"],)
#: The check a recoverable attempt failed, and only it.
INVENTORY_FAILURE = "only the glass notebook's registered outputs and mutable aliases changed"
UNCHANGED_FOLDERS = ("the model folders are as the previous step left them (nothing changed "
                     "between steps)")
#: The recovery patch's code: relative path -> LF SHA-256 before the patch (None: a new file).
#: Restoring these, and nothing else, must reproduce the run's preflight code identity.
CHANGES: Dict[str, Optional[str]] = {
    "tests/upstream_regeneration/test_inventory_and_records.py":
        "d677e488ce074632b7a3b05fe6cc7d57015144ad81569911800d6b768fb86b86",
    "tests/upstream_regeneration/test_recovery.py": None,
    "tests/upstream_regeneration/test_runner_orchestration.py":
        "3a335ea19cdacb1e79fc22c8ed273e709970dfb021a349b18f21bf99ad90a8a6",
    "upstream_regeneration/inventory.py":
        "4ffab6851d9b481bece5c4e528fc67d7a952e0519ab19a12efe1d0875db025e1",
    "upstream_regeneration/records.py":
        "818022b5a394deb622d1658242683cc1228cbe68fc08d546a8d16929ba8d09b8",
    "upstream_regeneration/recovery.py": None,
    "upstream_regeneration/runner.py":
        "b96e1c634136de16e9fcc7b601c9d9804648c2ee495eac3ae66639d248293d54",
}
ALLOWED_PREFIXES = ("upstream_regeneration/", "tests/upstream_regeneration/")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def failed_glass_record(run_id: str, path=None) -> Path:
    """The run's one failed GLASS execution record (or ``path``, which must be one of them)."""
    folder = Rec.run_dir(run_id) / "failed"
    found = sorted(f for f in folder.glob("execution_glass_*.json")            # not a recovery's
                   if re.fullmatch(r"execution_glass_[0-9]{8}T[0-9]{12}Z\.json", f.name)) \
        if folder.is_dir() else []
    if path is not None:
        p = Path(path).resolve()
        if p not in [f.resolve() for f in found]:
            raise Rec.RecordError(f"{path} is not a failed GLASS execution record of run {run_id} "
                                  f"(in {folder})")
        return p
    if len(found) != 1:
        raise Rec.RecordError(f"run {run_id} has {len(found)} failed GLASS execution records "
                              f"{[f.name for f in found]}: name one with --failed-record")
    return found[0]


def run_code(run_id: str, pre: Mapping[str, Any]) -> Dict[str, Any]:
    """The code identity a run's later steps must have: preflight's, or the recovered one."""
    if not Rec.record_path(run_id, STEP).exists():
        return pre["code"]
    rec = Rec.require_passed(run_id, STEP)
    if rec["code"]["run"] != pre["code"]:
        raise Rec.RecordError(f"run {run_id}: the recovery record's run code is not preflight's")
    return rec["code"]["recovered_under"]


def glass_execution(run_id: str):
    """``(record name, record)`` of the run's GLASS execution: executed or recovered, never both."""
    executed = Rec.record_path(run_id, "execution_glass").exists()
    recovered = Rec.record_path(run_id, STEP).exists()
    if executed and recovered:
        raise Rec.RecordError(f"run {run_id} has both execution_glass.json and {STEP}.json")
    name = STEP if recovered else "execution_glass"
    return name, Rec.require_passed(run_id, name)


def _name(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1]


def _copy(recorded: Mapping[str, Any], copies_dir) -> Path:
    return (Path(copies_dir) / _name(recorded["path"])) if copies_dir else Path(recorded["path"])


def _code_cells(nb: Mapping[str, Any]):
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _text(cell: Mapping[str, Any]) -> str:
    out = []
    for o in cell.get("outputs") or []:
        t = o.get("text")
        out.append("".join(t) if isinstance(t, list) else str(t or ""))
        for v in (o.get("data") or {}).values():
            out.append("".join(v) if isinstance(v, list) else str(v))
    return "\n".join(out)


def recover_glass(run_id: str, *, failed_record=None, copies_dir=None,
                  changes: Mapping[str, Optional[str]] = None) -> Dict[str, Any]:
    from shared.pinned_artifacts import STAGE2_PINS, verify_pin, write_pins
    from . import runner as Run
    changes = dict(CHANGES if changes is None else changes)
    root = Rec.BANK_ROOT
    pre = Rec.require_passed(run_id, "preflight")
    reuse = Rec.require_passed(run_id, "stage2_reuse")
    for name, why in (("execution_glass", "the GLASS execution passed: nothing to recover"),
                      (STEP, "the GLASS execution is already recovered (records are written once)")):
        if Rec.record_path(run_id, name).exists():
            raise Rec.RecordError(f"run {run_id}: {why}")
    failed_path = failed_glass_record(run_id, failed_record)
    raw = failed_path.read_bytes()
    failed = json.loads(raw.decode("utf-8"))
    rec = Run._header(run_id, STEP)
    rec["recovery_id"] = RECOVERY_ID
    gate = C.Gate("A4.R")

    # -- the original gates and the failed attempt -------------------------------------
    gate.check("A4.1 preflight and A4.2 stage2_reuse passed (sidecars verified); decision reuse",
               lambda: reuse["decision"]["outcome"] == "reuse")
    gate.check("the failed record is this run's GLASS execution attempt",
               lambda: (failed.get("run_id"), failed.get("step"), failed.get("passed"))
               == (run_id, "execution_glass", False))
    gate.check("the attempt ran on the run's code (preflight's code identity)",
               lambda: failed.get("code") == pre["code"])
    ex = failed.get("execution") or {}
    gate.check("its notebook execution completed with no cell error",
               lambda: ex.get("completed") is True and ex.get("error") is None,
               lambda: {"completed": ex.get("completed"), "error": ex.get("error"),
                        "wall_seconds": ex.get("wall_seconds")})
    checks = {c["check"]: c["passed"] for c in (failed.get("gates") or {}).get("execution", {})
              .get("checks", [])}
    gate.check("the model folders were unchanged before it ran", lambda: checks[UNCHANGED_FOLDERS])
    gate.check("its only failed check is the inventory classification",
               lambda: [k for k, v in checks.items() if not v] == [INVENTORY_FAILURE],
               lambda: [k for k, v in checks.items() if not v])
    problems = (failed.get("inventory_delta") or {}).get("problems") or []

    def resolvable():
        rels = [p[len("unregistered addition "):] for p in problems]
        return bool(problems) and all(p.startswith("unregistered addition ") for p in problems) \
            and all(any(re.fullmatch(pat, r) for pat in RESOLVES) for r in rels)

    gate.check("each of its inventory problems is an addition this recovery registers",
               resolvable, problems)

    # -- the code: the run's, plus only this recovery's files ---------------------------
    now = Rec.code_identity(root)
    gate.check("the recovery's changes are confined to upstream_regeneration code and its tests",
               lambda: bool(changes) and all(r.startswith(ALLOWED_PREFIXES) for r in changes))
    gate.check("every recorded change is present (new files exist; changed files differ)",
               lambda: all((root / r).is_file() and (base is None or Rec.file_sha256_lf(root / r)
                                                      != base) for r, base in changes.items()))
    undone = {}

    def reproduces():
        undone.update(Rec.code_identity(root, overrides=changes))
        return (undone["code_sha256"], undone["head"]) == (pre["code"]["code_sha256"],
                                                           pre["code"]["head"])

    gate.check("restoring exactly those files reproduces the run's preflight code identity",
               reproduces, lambda: {"preflight": pre["code"]["code_sha256"],
                                    "restored": undone.get("code_sha256")})

    # -- the notebook, its copies and its outputs ----------------------------------------
    copy_rec = {}

    def same_truncation():
        copy_rec.update(V.truncated_copy("glass", root=root)[1])
        return copy_rec == failed.get("truncated_copy")

    gate.check("the committed GLASS notebook's truncated copy is the one the attempt executed",
               same_truncation)
    paths = {k: _copy(ex.get(k) or {"path": ""}, copies_dir) for k in ("input_copy", "executed_copy")}
    data = {k: p.read_bytes() if p.is_file() else None for k, p in paths.items()}
    gate.check("the input copy has its recorded SHA-256, the truncated copy's",
               lambda: data["input_copy"] is not None and _sha(data["input_copy"])
               == ex["input_copy"]["sha256"] == failed["truncated_copy"]["truncated_copy_sha256"],
               str(paths["input_copy"]))
    gate.check("the executed copy has its recorded SHA-256",
               lambda: data["executed_copy"] is not None
               and _sha(data["executed_copy"]) == ex["executed_copy"]["sha256"],
               str(paths["executed_copy"]))
    executed = json.loads(data["executed_copy"]) if data["executed_copy"] else {}
    source = json.loads(data["input_copy"]) if data["input_copy"] else {}
    cells = _code_cells(executed)
    gate.check("the executed copy's code cells are the input copy's, in order",
               lambda: cells and ["".join(c["source"]) for c in cells]
               == ["".join(c["source"]) for c in _code_cells(source)])
    gate.check("every code cell executed, with no error output",
               lambda: cells and all(isinstance(c.get("execution_count"), int) for c in cells)
               and not [o for c in cells for o in c.get("outputs") or []
                        if o.get("output_type") == "error"])
    gate.check("every legacy Stage 2 cell printed its skip (RUN_LEGACY_STAGE2 was False)",
               lambda: all("skipped CELL" in _text(c) for c in cells
                           if "".join(c["source"]).startswith("%%legacy_stage2\n")))
    added = I.diff(pre["inventory"], failed.get("inventory_after") or {})["added"]
    outputs = "\n".join(_text(c) for c in cells)
    gate.check("every file the execution added is named in the executed notebook's outputs",
               lambda: added and all(_name(a) in outputs for a in added),
               lambda: [a for a in added if _name(a) not in outputs] or added)

    # -- the model folders and the historical artifacts ---------------------------------
    inv = I.snapshot(root)
    gate.check("the model folders are byte for byte as the attempt left them",
               lambda: I.diff(failed.get("inventory_after") or {}, inv)
               == {"added": [], "modified": [], "deleted": []},
               lambda: {k: v[:5] for k, v in I.diff(failed.get("inventory_after") or {}, inv).items()})
    delta = I.classify(pre["inventory"], inv, "glass")
    gate.check("since preflight, only the GLASS notebook's registered outputs, companion and "
               "mutable aliases changed", lambda: delta["passed"], lambda: delta["problems"][:8])
    gate.check("the produced roles are the attempt's",
               lambda: delta["produced"] == (failed.get("inventory_delta") or {}).get("produced"))
    for pin in STAGE2_PINS:
        gate.check(f"{pin.role}: the accepted file matches its pin (SHA-256, size)",
                   lambda pin=pin: verify_pin(pin, root) is not None, pin.sha256[:16])

    def accepted():
        from stage2_crossfit import acceptance
        acceptance.require_accepted(C.PR38_RUN_ID)
        return True

    gate.check(f"the PR 38 run {C.PR38_RUN_ID} passes require_accepted", accepted)
    reload = {}
    if delta["passed"]:
        files = {**{r: root / p for r, p in delta["produced"].items()},
                 **{c: root / p for c, p in delta["companions"].items()}}
        reload = C.fresh_reload(gate, files)
        gate.check("the baseline LR reloads as GLASS Cell 6's reference model, not a hand-off",
                   lambda: {"model_name", "hyperparameters"} <= set(
                       reload["stage1.lr_baseline"]["keys"] or [])
                   and "stage_output" not in reload["stage1.lr_baseline"]["keys"])

    produced = {}
    if gate.passed:
        produced = {role: Run._pin(role, rel, inv) for role, rel in delta["produced"].items()}

        def pins():
            pins_path = Rec.run_dir(run_id) / Run.PINS_FILE
            write_pins(pins_path, run_id, {**produced, **{p.role: p for p in STAGE2_PINS}}, root=root)
            rec["pins_file"] = {"path": pins_path.relative_to(root).as_posix(),
                                "roles": sorted({*produced, *(p.role for p in STAGE2_PINS)})}
            return True

        gate.check("the run's pins file written (stage1.lr, stage3.ebm and the two Stage 2 pins)",
                   pins)
    rec.update(
        gates={"A4.R": gate.record()},
        original={"failed_record": {"path": failed_path.relative_to(root).as_posix()
                                    if failed_path.is_relative_to(root) else str(failed_path),
                                    "sha256": _sha(raw), "started_utc": failed.get("started_utc"),
                                    "finished_utc": failed.get("finished_utc"),
                                    "failed_checks": [k for k, v in checks.items() if not v],
                                    "inventory_problems": problems},
                  "preflight_sha256": Run._sidecar(run_id, "preflight"),
                  "stage2_reuse_sha256": Run._sidecar(run_id, "stage2_reuse")},
        execution=ex, truncated_copy=failed.get("truncated_copy"),
        inventory_after=inv, inventory_delta=delta,
        produced={role: p.record() for role, p in produced.items()},
        companions={c: {"path": p, **inv[p]} for c, p in delta["companions"].items()}
        if delta["passed"] else {},
        registered=list(RESOLVES), reload=reload,
        code={"run": pre["code"], "recovered_under": now,
              "restored_identity": undone or None,
              "changes": [{"path": r, "before_lf_sha256": b,
                           "after_lf_sha256": Rec.file_sha256_lf(root / r)
                           if (root / r).is_file() else None} for r, b in sorted(changes.items())]})
    return Run._finish(run_id, STEP, rec, gate.passed)
