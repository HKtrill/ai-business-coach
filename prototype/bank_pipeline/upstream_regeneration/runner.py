"""
upstream_regeneration.runner
============================
The Step 4A steps and their records (package docstring). Each step requires
the previous steps of the same run to have passed and the code to be the code
the run started on (``records.code_identity``); it writes its record once when
every check passed, and otherwise writes ``failed/<step>_<utc>.json`` and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import checks as C
from . import inventory as I
from . import records as Rec
from . import vehicle as V

__all__ = ["StepFailed", "preflight", "stage2_reuse", "execute", "finalize", "main",
           "PINS_FILE", "CONSUMERS"]

PINS_FILE = "upstream_pins.json"
#: Who reads each role (sections 3.1, 5.1): recorded in the run record.
CONSUMERS = {
    "stage1.lr": ["black-box Cell 14A (GLASS_S1_PATH)", "GLASS Cell 19", "black-box Cell 37"],
    "stage1.mlp": ["black-box Cells 36-37 (MLP_S1_PATH)"],
    "stage2.glass_router": ["GLASS Cell 8B / Cell 19 (glass_path)",
                            "black-box Cell 18 / Cell 37 (GLASS_BUNDLE_PATH)"],
    "stage2.rf_router": ["black-box Cell 18 / Cells 35-37 (RF_ROUTER_PATH)"],
    "stage3.ebm": ["black-box Cell 33 (_ebm_path)", "GLASS Cell 19", "black-box Cell 37"],
    "stage3.xgb": ["black-box Cells 36-37 (XGB_STAGE3_PATH)"],
}
#: Git-tracked files a Commit 4A run may change (D7: committed in the producing commit).
TRACKED_ALIASES = re.compile(
    r"models/mlp/stage1_mlp_(threshold_sweep|training_history|tuning_trials)\.csv$")


class StepFailed(RuntimeError):
    """A step's checks failed; its failure record is named."""


def _header(run_id: str, step: str) -> Dict[str, Any]:
    return {"format": f"pr39.commit4a.{step}/1", "run_id": run_id, "step": step,
            "contract": Rec.CONTRACT, "started_utc": Rec.now()}


def _finish(run_id: str, step: str, rec: Dict[str, Any], passed: bool,
            root: Optional[Path] = None) -> Dict[str, Any]:
    rec.update(finished_utc=Rec.now(), passed=bool(passed))
    if passed:
        path = Rec.write_record(run_id, step, rec, root=root)
        rec["path"] = (path.relative_to(Rec.BANK_ROOT).as_posix()
                       if path.is_relative_to(Rec.BANK_ROOT) else str(path))
        return rec
    path = Rec.write_failed(run_id, step, rec, root=root)
    failed = [f"{g}: {c}" for g, r in (rec.get("gates") or {}).items() if isinstance(r, dict)
              for c in (r.get("failed") or [])]
    if (rec.get("decision") or {}).get("outcome") == "regenerate":
        failed.append(f"decision: regenerate. {rec['decision']['next']}")
    raise StepFailed(f"{step} did not pass; recorded in {path}" +
                     "".join(f"\n  - {f}" for f in failed[:20]))


def _same_code(run_id: str, pre: Mapping[str, Any]) -> Dict[str, Any]:
    """
    The code is the run's: preflight's identity or, after a recorded GLASS
    recovery (``recovery``) or code transition (``transition``), the identity
    the latest of them verified and recorded.
    """
    from .recovery import STEP as RECOVERED
    from .transition import STEP as TRANSITIONED, run_code
    want = run_code(run_id, pre)
    now = Rec.code_identity(Rec.BANK_ROOT)
    if now["code_sha256"] != want["code_sha256"] or now["head"] != want["head"]:
        since = ("the recorded code transition" if Rec.record_path(run_id, TRANSITIONED).exists()
                 else "the recorded recovery" if Rec.record_path(run_id, RECOVERED).exists()
                 else "preflight")
        raise Rec.RecordError(f"run {run_id}: the code changed since {since} "
                              f"({want['code_sha256'][:16]} -> {now['code_sha256'][:16]}); "
                              "a run uses one code identity: start a new run id")
    return now


def _layout(run_id: str) -> Dict[str, Any]:
    return {"records": f"research_logs/pr39/{run_id}/",
            "pins_file": f"research_logs/pr39/{run_id}/{PINS_FILE}",
            "failed_attempts": f"research_logs/pr39/{run_id}/failed/ (Git-ignored, never read)",
            "executed_notebooks": "outside the repository (--out-dir); SHA-256 recorded",
            "stage2": "reused read-only: models/pr38/c3_20261112_153333/primary/ (pinned)",
            "models_pr39": "not used: Stage 2 is reused (section 5.2)",
            "notebook_outputs": {nb: dict(I.OUTPUTS[nb]) for nb in I.OUTPUTS},
            "mutable_aliases": {nb: list(I.MUTABLE_ALIASES[nb]) for nb in I.MUTABLE_ALIASES}}


# ----------------------------------------------------------------------
# A4.1
# ----------------------------------------------------------------------
def preflight(run_id: str, sums_path) -> Dict[str, Any]:
    Rec.check_run_id(run_id)
    if Rec.record_path(run_id, "preflight").exists():
        raise Rec.RecordError(f"run {run_id} already has a preflight record: new run id")
    rec = _header(run_id, "preflight")
    gate = C.Gate("A4.1")
    rec["evidence"] = C.preservation(gate, sums_path)
    inv = I.snapshot(Rec.BANK_ROOT)
    gate.check("7.5.5 SHA-256 inventory of models/ and blackbox_pipeline/models/ taken before "
               "any write", lambda: bool(inv), f"{len(inv)} files")
    status = Rec.git_status(Rec.BANK_ROOT)
    gate.check("7.5.6 git status recorded (the applied Step 4A patch only)",
               lambda: status is not None, f"{len(status or [])} entries")
    rec.update(gates={"A4.1": gate.record()}, inventory=inv, git_status=status,
               live_counterparts=C.live_counterparts(rec["evidence"].get("snapshot") or {}),
               layout=_layout(run_id), code=Rec.code_identity(Rec.BANK_ROOT),
               environment=Rec.environment(),
               sums={"path": str(sums_path)})
    return _finish(run_id, "preflight", rec, gate.passed)


# ----------------------------------------------------------------------
# A4.2
# ----------------------------------------------------------------------
def stage2_reuse(run_id: str) -> Dict[str, Any]:
    pre = Rec.require_passed(run_id, "preflight")
    code = _same_code(run_id, pre)
    rec = _header(run_id, "stage2_reuse")
    split = C.global_split()
    gate = C.Gate("A4.2")
    from stage2_crossfit.runner import check_population, make_train_side
    import shared.stage2.evaluation_protocol as ep
    pop = check_population(make_train_side(split["X_train"], split["y_train"], split["X_test"]),
                           ep.load_frozen_protocol())
    gate.check("the train side is the frozen protocol's population (rows, positives, folds, "
               "split, labels)", lambda: pop["passed"], lambda: {k: v["got"] for k, v in
                                                                 pop["checks"].items()
                                                                 if not isinstance(v["got"], list)})
    rec["evidence"] = C.stage2_reuse(gate, split)
    decision = C.decide(pre, gate.record())
    rec.update(gates={"A4.2": gate.record()}, decision=decision, code=code,
               preflight_sha256=_sidecar(run_id, "preflight"))
    return _finish(run_id, "stage2_reuse", rec, decision["outcome"] == "reuse")


def _sidecar(run_id: str, name: str) -> str:
    p = Rec.record_path(run_id, name)
    return p.with_name(p.name + ".sha256").read_text(encoding="ascii").strip()


# ----------------------------------------------------------------------
# section 5.4
# ----------------------------------------------------------------------
def _pin(role: str, rel: str, inv: Mapping[str, Mapping]):
    from shared.pinned_artifacts import PinnedFile
    meta = inv[rel]
    return PinnedFile(role, rel, meta["sha256"], int(meta["bytes"]))


def execute(run_id: str, notebook: str, out_dir, *, kernel: Optional[str] = None) -> Dict[str, Any]:
    from shared.pinned_artifacts import PINS_ENV, STAGE2_PINS, write_pins
    pre = Rec.require_passed(run_id, "preflight")
    Rec.require_passed(run_id, "stage2_reuse")
    code = _same_code(run_id, pre)
    step = f"execution_{notebook}"
    from .recovery import STEP as RECOVERED, glass_execution
    if Rec.record_path(run_id, step).exists() or (
            notebook == "glass" and Rec.record_path(run_id, RECOVERED).exists()):
        raise Rec.RecordError(f"run {run_id} already executed the {notebook} notebook")
    pins_path = Rec.run_dir(run_id) / PINS_FILE
    if notebook == "glass":
        expected_before, env = pre["inventory"], {PINS_ENV: ""}
    elif notebook == "blackbox":
        _, glass = glass_execution(run_id)              # executed, or recovered
        expected_before, env = glass["inventory_after"], {PINS_ENV: str(pins_path)}
        if not pins_path.is_file():
            raise Rec.RecordError(f"no pins file {pins_path}: the GLASS execution writes it")
    else:
        raise V.VehicleError(f"unknown notebook {notebook!r}")
    rec = _header(run_id, step)
    gate = C.Gate(f"execution [{notebook}]")
    out_dir = V.require_outside(out_dir, Rec.BANK_ROOT)
    nb, copy_rec = V.truncated_copy(notebook, root=Rec.BANK_ROOT)
    rec["truncated_copy"] = copy_rec
    before = I.snapshot(Rec.BANK_ROOT)
    gate.check("the model folders are as the previous step left them (nothing changed between "
               "steps)", lambda: I.diff(expected_before, before) == {"added": [], "modified": [],
                                                                       "deleted": []},
               lambda: {k: v[:5] for k, v in I.diff(expected_before, before).items()})
    if not gate.passed:
        rec.update(gates={"execution": gate.record()}, code=code)
        return _finish(run_id, step, rec, False)
    run = V.execute(nb, notebook, out_dir, run_id, env=env, kernel_name=kernel, root=Rec.BANK_ROOT)
    rec["execution"] = run
    gate.check("fresh-kernel execution completed with no cell error", lambda: run["completed"],
               run.get("error") or f"{run['wall_seconds']} s")
    after = I.snapshot(Rec.BANK_ROOT)
    delta = I.classify(before, after, notebook)
    rec.update(inventory_after=after, inventory_delta=delta)
    gate.check(f"only the {notebook} notebook's registered outputs and mutable aliases changed",
               lambda: delta["passed"], lambda: delta["problems"][:8] or delta["produced"])
    produced = {}
    if delta["passed"]:
        produced = {role: _pin(role, rel, after) for role, rel in delta["produced"].items()}
        rec["produced"] = {role: p.record() for role, p in produced.items()}
    if gate.passed and notebook == "glass":
        def pins():
            files = {**produced, **{p.role: p for p in STAGE2_PINS}}
            write_pins(pins_path, run_id, files, root=Rec.BANK_ROOT)
            rec["pins_file"] = {"path": pins_path.relative_to(Rec.BANK_ROOT).as_posix(),
                                "roles": sorted(files)}
            return True
        gate.check("the run's pins file written (stage1.lr, stage3.ebm and the two Stage 2 pins)",
                   pins)
    rec.update(gates={"execution": gate.record()}, code=code)
    return _finish(run_id, step, rec, gate.passed)


# ----------------------------------------------------------------------
# A4.3 - A4.6
# ----------------------------------------------------------------------
def _git_delta(pre_status, post_status, run_id: str, code_changes=()):
    """New ``git status`` lines: this run's records, the D7 CSVs and, after a recorded
    recovery, exactly the code files that recovery records (``code_changes``)."""
    new = sorted(set(post_status or []) - set(pre_status or []))
    gone = sorted(set(pre_status or []) - set(post_status or []))

    def allowed(line: str) -> bool:
        path = line[3:].strip().strip('"')
        return (f"research_logs/pr39/{run_id}/" in path or bool(TRACKED_ALIASES.search(path))
                or any(path == c or path.endswith("/" + c) for c in code_changes))
    return {"new": new, "gone": gone, "unexpected": [ln for ln in new if not allowed(ln)] + gone}


def finalize(run_id: str, sums_path) -> Dict[str, Any]:
    import joblib
    from shared.pinned_artifacts import STAGE2_PINS, verify_pin
    pre = Rec.require_passed(run_id, "preflight")
    reuse = Rec.require_passed(run_id, "stage2_reuse")
    from .recovery import glass_execution
    glass_step, glass = glass_execution(run_id)           # executed, or recovered
    black = Rec.require_passed(run_id, "execution_blackbox")
    code = _same_code(run_id, pre)
    recovered = glass_step != "execution_glass"
    from . import transition as T
    transition = T.passed_record(run_id)
    code_changes = T.code_changes(run_id)                # recovery and transition files, if any
    named_failed = T.named_failed_attempts(run_id)
    rec = _header(run_id, "run_record")
    rec["status"] = ("Stage 1-3 regenerated (Stage 2 reused); NOT accepted: Step 4B is the "
                     "acceptance gate (A4.7)")
    rec["accepted"] = False
    from shared.pinned_artifacts import PinnedFile
    pins = {p.role: p for p in STAGE2_PINS}
    for role, r in {**glass["produced"], **black["produced"]}.items():
        pins[role] = PinnedFile(role, r["path"], r["sha256"], int(r["bytes"]))
    gates = {g: C.Gate(g) for g in ("A4.3", "A4.4", "A4.5", "A4.6")}
    g6 = gates["A4.6"]
    paths = {}
    for role, pin in sorted(pins.items()):
        if g6.check(f"{role}: {pin.path} unchanged since it was produced (SHA-256, size)",
                    lambda pin=pin: verify_pin(pin, Rec.BANK_ROOT) is not None, pin.sha256[:16]):
            paths[role] = pin.resolve(Rec.BANK_ROOT)
    split = C.global_split()
    population = C.frozen_population()
    evidence: Dict[str, Any] = {}
    if {"stage1.lr", "stage1.mlp"} <= set(paths):
        evidence["stage1"] = C.stage1_checks(
            gates["A4.3"], {r: joblib.load(paths[r]) for r in ("stage1.lr", "stage1.mlp")},
            split, population)
    if {"stage3.ebm", "stage3.xgb"} <= set(paths):
        evidence["stage3"] = C.stage3_checks(
            gates["A4.4"], {r: joblib.load(paths[r]) for r in ("stage3.ebm", "stage3.xgb")},
            population, split=split)
    if len(paths) == len(pins):
        evidence["stage4_inputs"] = C.stage4_input_checks(gates["A4.5"], split, paths)
        evidence["reload"] = C.fresh_reload(gates["A4.3"], paths)
    after = I.snapshot(Rec.BANK_ROOT)
    gates["A4.5"].check("no Stage 4 artifact written: the model folders are as the black-box "
                        "execution left them",
                        lambda: I.diff(black["inventory_after"], after) == {
                            "added": [], "modified": [], "deleted": []})
    # A4.6: section 7.5 after the run
    evidence["preservation_after"] = C.preservation(g6, sums_path)
    net = I.classify(pre["inventory"], after, I.RUN)
    g6.check("7.5.5 models/ inventory before and after: only the run's registered additions and "
             "documented mutable aliases", lambda: net["passed"], lambda: net["problems"][:8])
    status = Rec.git_status(Rec.BANK_ROOT)
    git = _git_delta(pre["git_status"], status, run_id, code_changes)
    g6.check("7.5.6 git status: only this run's records and the D7 MLP CSVs are new",
             lambda: status is not None and not git["unexpected"], lambda: git["unexpected"][:8])
    lc_pre = pre["live_counterparts"]
    lc_post = C.live_counterparts(evidence["preservation_after"].get("snapshot") or {})
    lc_problems = C.live_counterpart_problems(lc_pre, lc_post)
    g6.check("7.5.1 the snapshot's live counterparts: only Step 4A's registered mutable aliases "
             "changed; every historical artifact and legacy alias is as before the run",
             lambda: not lc_problems,
             lambda: lc_problems or {"before": lc_pre, "after": lc_post})
    g6.check("the two notebook executions are recorded (section 5.4)",
             lambda: all(r["execution"]["completed"] for r in (glass, black)))
    if named_failed:
        def preserved():
            for f in named_failed:
                p = Path(f["path"]) if Path(f["path"]).is_absolute() else Rec.BANK_ROOT / f["path"]
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
                    raise AssertionError(f"{f['path']} (named by {f['by']}) is missing or changed")
            return True

        g6.check("the failed attempts the run's recovery and transition name are preserved "
                 "(path, SHA-256; hashed, not read)", preserved,
                 [f"{f['by']}: {f['sha256'][:16]}" for f in named_failed])
    rec.update(
        gates={"A4.1": {"record": "preflight.json", "sha256": _sidecar(run_id, "preflight"),
                        "passed": True},
               "A4.2": {"record": "stage2_reuse.json", "sha256": _sidecar(run_id, "stage2_reuse"),
                        "passed": True, "decision": reuse["decision"]["outcome"]},
               **{g: gate.record() for g, gate in gates.items()}},
        stage2_disposition=reuse["decision"],
        artifacts={role: {**pin.record(), "producer": _producer(role), "consumers": CONSUMERS[role]}
                   for role, pin in sorted(pins.items())},
        identities=C.identities(split, population),
        executions={nb: {"record": f"{step}.json", "sha256": _sidecar(run_id, step),
                         "kernel": r["execution"]["kernel"],
                         "input_copy": r["execution"]["input_copy"],
                         "executed_copy": r["execution"]["executed_copy"],
                         "truncated_copy_sha256": r["truncated_copy"]["truncated_copy_sha256"],
                         "removed_code_cells": r["truncated_copy"]["removed_code_cells"],
                         "wall_seconds": r["execution"]["wall_seconds"]}
                    for nb, step, r in (("glass", glass_step, glass),
                                        ("blackbox", "execution_blackbox", black))},
        glass_recovery=({"record": f"{glass_step}.json", "sha256": _sidecar(run_id, glass_step),
                         "recovery_id": glass["recovery_id"],
                         "failed_record": glass["original"]["failed_record"],
                         "code": {"run": glass["code"]["run"],
                                  "recovered_under": glass["code"]["recovered_under"],
                                  "changed_files": [c["path"] for c in glass["code"]["changes"]]}}
                        if recovered else None),
        code_transition=({"record": f"{T.STEP}.json", "sha256": _sidecar(run_id, T.STEP),
                          "transition_id": transition["transition_id"],
                          "authorization": transition["authorization"],
                          "resolves": transition["resolves"],
                          "failed_finalize": transition["original"]["failed_finalize"],
                          "code": {"from": transition["code"]["from"],
                                   "to": transition["code"]["to"],
                                   "changed_files": [c["path"] for c in
                                                     transition["code"]["changes"]]}}
                         if transition else None),
        failed_attempts_preserved=named_failed,
        tracked_aliases_d7={rel: {"before": pre["inventory"].get(rel), "after": after.get(rel)}
                            for rel in sorted(set(pre["inventory"]) | set(after))
                            if TRACKED_ALIASES.search(rel)},
        inventory_net=net, git_status_delta=git,
        code_identity_notes={
            "before": ((pre["evidence"].get("baseline") or {}).get("verify") or {}).get("output_tail"),
            "after": ((evidence["preservation_after"].get("baseline") or {}).get("verify") or {})
            .get("output_tail"),
            "expected": "stage2_baseline_archive.py --check reports the deliberate pin changes "
                        "(tests/stage2/test_stage2_operating_contract_v2.DELIBERATE_PIN_CHANGES): "
                        "shared/stage2/diagnostics/boundary.py (2B) and "
                        "tests/support/artifact_audit.py (3B; 4A G10)"},
        evidence=evidence, code=code, environment=Rec.environment())
    return _finish(run_id, "run_record", rec, all(g.passed for g in gates.values()))


def _producer(role: str) -> str:
    from shared.pinned_artifacts import ROLES
    return ROLES[role]


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    from . import __doc__ as doc
    ap = argparse.ArgumentParser(prog="python -m upstream_regeneration", description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="A4.1: section 7.5 before any write")
    p.add_argument("--run-id", default=None, help="default: c4_<local YYYYMMDD_HHMMSS>")
    p.add_argument("--sums", required=True, help="the baseline snapshot's SHA256SUMS.txt")
    p = sub.add_parser("stage2-reuse", help="A4.2: the section 5.2 decision")
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("execute", help="section 5.4: one notebook, Stage 4 excluded")
    p.add_argument("--run-id", required=True)
    p.add_argument("--notebook", required=True, choices=sorted(V.NOTEBOOKS))
    p.add_argument("--out-dir", required=True, help="outside the repository")
    p.add_argument("--kernel", default=None, help="Jupyter kernel name (default: the notebook's)")
    p = sub.add_parser("recover-glass", help="validate a completed GLASS execution refused only "
                       "for an output since registered (recovery); runs no notebook")
    p.add_argument("--run-id", required=True)
    p.add_argument("--failed-record", default=None,
                   help="the failed/execution_glass_<utc>.json (default: the run's only one)")
    p.add_argument("--copies-dir", default=None,
                   help="where the input and executed copies are, if moved since")
    p = sub.add_parser("finalize-transition", help="authorize the registered code transition "
                       "after a failed finalize (transition); runs no notebook, fits nothing")
    p.add_argument("--run-id", required=True)
    p.add_argument("--authorization", required=True,
                   help="the owner's authorization statement, recorded verbatim (at most 500 "
                        "characters)")
    p.add_argument("--failed-record", default=None,
                   help="the failed/run_record_<utc>.json (default: the run's only one)")
    p = sub.add_parser("finalize", help="A4.3-A4.6 and the run record")
    p.add_argument("--run-id", required=True)
    p.add_argument("--sums", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "preflight":
            rec = preflight(args.run_id or Rec.new_run_id(), args.sums)
        elif args.cmd == "stage2-reuse":
            rec = stage2_reuse(args.run_id)
        elif args.cmd == "execute":
            rec = execute(args.run_id, args.notebook, args.out_dir, kernel=args.kernel)
        elif args.cmd == "finalize-transition":
            from .transition import transition_finalize
            rec = transition_finalize(args.run_id, authorization=args.authorization,
                                      failed_record=args.failed_record)
        elif args.cmd == "recover-glass":
            from .recovery import recover_glass
            rec = recover_glass(args.run_id, failed_record=args.failed_record,
                                copies_dir=args.copies_dir)
        else:
            rec = finalize(args.run_id, args.sums)
    except (StepFailed, Rec.RecordError, V.VehicleError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 1
    print(f"OK {rec['step']} for run {rec['run_id']}: wrote {rec.get('path')}")
    for name, g in (rec.get("gates") or {}).items():
        if isinstance(g, dict) and "evidence" in g:
            print(f"  {name}: {g['evidence']}")
    if rec["step"] == "stage2_reuse":
        print(f"  decision: {rec['decision']['outcome']}")
    return 0
