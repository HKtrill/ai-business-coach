"""
upstream_regeneration.acceptance
================================
PR 39 Commit 4, Step 4B: gate A4.7, the explicit Stage 1-3 acceptance decision
and the PR 39 upstream artifact manifest (execution contract 1.0.0: section
5.1 item 4, section 7.1, A4.7).

    python -m upstream_regeneration.acceptance accept --run-id <id> --sums <SHA256SUMS.txt> --statements <statements.json> [--copies-dir <dir>]
    python -m upstream_regeneration.acceptance check  --run-id <id>

``accept`` fits, executes and rewrites nothing. When every check passes it
writes, once each, with ``.sha256`` sidecars:

* ``artifact_manifest.json`` (section 7.1): every file the run consumed or
  produced, with role, path, size and SHA-256. It is also a valid
  ``pr39.upstream_pins/1`` file, so Commit 5 resolves each Stage 1-3 input by
  SHA-256 with ``shared.pinned_artifacts.pinned_path(role, pins_file=...)``
  (A5.1), with no newest-file selection;
* ``upstream_acceptance.json``: the A4.7 decision. It records the owner's
  decision and the two run-specific dispositions (sections 7.1 and 7.2)
  verbatim, everything below, and the manifest's SHA-256.

``run_record.json`` is not touched. It stays the regeneration record, with
``accepted: false``. The acceptance record is the decision.

What ``accept`` verifies. Recorded ``passed`` fields are never taken on trust:
each claim is rebuilt from primary evidence where it can be.

1. **Records.**
   * Every Step 4A record is present and matches its sidecar.
   * Each record is this run's own, and the records are in order.
   * Every later record names the earlier ones by their sidecar SHA-256.
2. **Code (section 7.2).**
   * Omitting this module's own new files (``ACCEPTANCE_FILES``) reproduces
     the run record's code identity.
   * Restoring the transition's recorded file hashes then reproduces the
     recovered identity. Restoring the recovery's as well reproduces the
     preflight identity.
   * Every step's code is the one that chain gives it.
   * Every recorded change lies under ``upstream_regeneration/`` or its tests.
   * The registered recovery and transition ids hold.
3. **Notebook executions (section 5.4).**
   * Each committed notebook's truncated copy, rebuilt now, is the executed one.
   * The input and executed copies have their recorded SHA-256.
4. **Recomputation.**
   * A4.2 (Stage 2 reuse), then A4.3-A4.6 exactly as ``finalize`` runs them.
     That covers the Stage 1 and Stage 3 checks, the four Stage 4 input builds
     (no arbiter fitted) and the section 7.5 checks.
   * Each gate must pass with the same checks as its record. Its evidence must
     equal the record's, value for value.
5. **Artifacts.** The six Stage 1-3 files have one path, size and SHA-256 in
   every record that names them, and on disk.
6. **D7.** Each tracked MLP CSV is as the black-box execution left it. Whether
   it is staged follows Git's own content comparison with ``HEAD``: one that
   differs is staged; one that does not is not.
7. **The owner's statements.** The decision and the 7.1 and 7.2 dispositions,
   each non-empty and at most 2,000 characters.

This module adds files and changes none (``ACCEPTANCE_FILES``), so the run
keeps its code identity. A failed acceptance writes
``failed/upstream_acceptance_<utc>.json`` and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import checks as C
from . import inventory as I
from . import records as Rec
from . import recovery as Recov
from . import runner as Run
from . import transition as T
from . import vehicle as V

__all__ = ["STEP", "MANIFEST", "ACCEPTANCE_FILES", "STATEMENT_KEYS", "ROLES", "D7",
           "read_statements", "accept", "check", "main"]

STEP = "upstream_acceptance"
MANIFEST = "artifact_manifest"
MANIFEST_KIND = "pr39.artifact_manifest/1"
#: This step's code: new files only. Omitting them must reproduce the run's code identity.
ACCEPTANCE_FILES = ("tests/upstream_regeneration/test_acceptance.py",
                    "upstream_regeneration/acceptance.py")
STATEMENT_KEYS = ("decision", "disposition_7_1", "disposition_7_2")
MAX_STATEMENT = 2000
ROLES = ("stage1.lr", "stage1.mlp", "stage2.glass_router", "stage2.rf_router", "stage3.ebm",
         "stage3.xgb")
#: The Git-tracked mutable aliases a Commit 4A run may change (F12, D7).
D7 = ("models/mlp/stage1_mlp_threshold_sweep.csv", "models/mlp/stage1_mlp_training_history.csv",
      "models/mlp/stage1_mlp_tuning_trials.csv")
#: Recomputed evidence compared value for value (hashes, names, counts, contracts).
STAGE2_EVIDENCE = ("acceptance", "manifest", "names", "binning", "stage4_router_contract",
                   "r4_notes")
RUN_EVIDENCE = ("stage1", "stage3", "stage4_inputs", "reload")
_CONTRACT = {"record": Rec.CONTRACT["record"], "version": Rec.CONTRACT["version"],
             "sections": ["5.1", "7.1", "7.2", "7.3"], "gates": ["A4.7"],
             "decisions": ["D3", "D7"]}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code(c: Mapping[str, Any]) -> tuple:
    return (c.get("code_sha256"), c.get("head")) if isinstance(c, Mapping) else (None, None)


def _pick(ev, keys):
    return {k: ev.get(k) for k in keys} if isinstance(ev, Mapping) else ev


def _diff_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _line_path(line: str) -> str:
    return line[3:].strip().strip('"')


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def read_statements(path) -> Dict[str, Any]:
    """The owner's statements file: exactly ``STATEMENT_KEYS``, each a non-empty string."""
    p = Path(path)
    if not p.is_file():
        raise Rec.RecordError(f"statements file {p} does not exist")
    raw = p.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rec.RecordError(f"statements file {p} is not UTF-8 JSON: {exc}") from None
    if not isinstance(data, dict) or sorted(data) != sorted(STATEMENT_KEYS):
        raise Rec.RecordError(f"statements file {p} must hold exactly {list(STATEMENT_KEYS)}")
    for k in STATEMENT_KEYS:
        v = data[k]
        if not isinstance(v, str) or not v.strip() or len(v.strip()) > MAX_STATEMENT:
            raise Rec.RecordError(f"statement {k!r} must be non-empty text of at most "
                                  f"{MAX_STATEMENT} characters")
    return {"file": p.name, "sha256": _sha(raw), **{k: data[k].strip() for k in STATEMENT_KEYS}}


def _records(run_id: str):
    """``(names in order, {name: record})``; every record sidecar-verified and passed."""
    glass_step, _ = Recov.glass_execution(run_id)
    names = ["preflight", "stage2_reuse", glass_step, "execution_blackbox"]
    if Rec.record_path(run_id, T.STEP).exists():
        names.append(T.STEP)
    names.append("run_record")
    recs = {n: Rec.require_passed(run_id, n) for n in names}
    named = {T.STEP: recs["run_record"].get("code_transition"),
             Recov.STEP: recs["run_record"].get("glass_recovery")}
    for name, entry in named.items():
        if entry and name not in recs:
            raise Rec.RecordError(f"run {run_id}: the run record names {name}.json, which is "
                                  "missing")
    return names, recs


def _before(changes) -> Dict[str, Optional[str]]:
    return {c["path"]: c["before_lf_sha256"] for c in changes}


# ----------------------------------------------------------------------
# recomputation, exactly as stage2_reuse and finalize run it
# ----------------------------------------------------------------------
def _recheck_stage2(split: dict):
    from stage2_crossfit.runner import check_population, make_train_side
    import shared.stage2.evaluation_protocol as ep
    gate = C.Gate("A4.2")
    pop = check_population(make_train_side(split["X_train"], split["y_train"], split["X_test"]),
                           ep.load_frozen_protocol())
    gate.check("the train side is the frozen protocol's population (rows, positives, folds, "
               "split, labels)", lambda: pop["passed"], lambda: {k: v["got"] for k, v in
                                                                 pop["checks"].items()
                                                                 if not isinstance(v["got"], list)})
    return gate, C.stage2_reuse(gate, split)


def _recheck_run(run_id: str, sums_path, pre, glass, black, split, population,
                 extra_code: Sequence[str]) -> Dict[str, Any]:
    """``runner.finalize``'s A4.3-A4.6, check for check, writing nothing."""
    import joblib
    from shared.pinned_artifacts import PinnedFile, STAGE2_PINS, verify_pin
    pins = {p.role: p for p in STAGE2_PINS}
    for role, r in {**glass["produced"], **black["produced"]}.items():
        pins[role] = PinnedFile(role, r["path"], r["sha256"], int(r["bytes"]))
    code_changes = sorted(set(T.code_changes(run_id)) | set(extra_code))
    named_failed = T.named_failed_attempts(run_id)
    gates = {g: C.Gate(g) for g in ("A4.3", "A4.4", "A4.5", "A4.6")}
    g6 = gates["A4.6"]
    paths = {}
    for role, pin in sorted(pins.items()):
        if g6.check(f"{role}: {pin.path} unchanged since it was produced (SHA-256, size)",
                    lambda pin=pin: verify_pin(pin, Rec.BANK_ROOT) is not None, pin.sha256[:16]):
            paths[role] = pin.resolve(Rec.BANK_ROOT)
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
    evidence["preservation_after"] = C.preservation(g6, sums_path)
    net = I.classify(pre["inventory"], after, I.RUN)
    g6.check("7.5.5 models/ inventory before and after: only the run's registered additions and "
             "documented mutable aliases", lambda: net["passed"], lambda: net["problems"][:8])
    status = Rec.git_status(Rec.BANK_ROOT)
    git = Run._git_delta(pre["git_status"], status, run_id, code_changes)
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
                if not p.is_file() or _sha(p.read_bytes()) != f["sha256"]:
                    raise AssertionError(f"{f['path']} (named by {f['by']}) is missing or changed")
            return True

        g6.check("the failed attempts the run's recovery and transition name are preserved "
                 "(path, SHA-256; hashed, not read)", preserved,
                 [f"{f['by']}: {f['sha256'][:16]}" for f in named_failed])
    return {"gates": gates, "evidence": evidence, "after": after, "net": net, "status": status,
            "git": git, "pins": pins, "named_failed": named_failed}


# ----------------------------------------------------------------------
# A4.7
# ----------------------------------------------------------------------
def accept(run_id: str, sums_path, statements_path, *, copies_dir=None,
           acceptance_files: Sequence[str] = ACCEPTANCE_FILES) -> Dict[str, Any]:
    from shared.pinned_artifacts import (PINS_FORMAT, ROLES as PRODUCERS, STAGE2_MANIFEST,
                                         STAGE2_MANIFEST_SHA256, STAGE2_PINS, STAGE2_RUN_ID,
                                         read_pins)
    root = Rec.BANK_ROOT
    if Rec.record_path(run_id, STEP).exists():
        raise Rec.RecordError(f"run {run_id} is already accepted ({STEP}.json is written once)")
    statements = read_statements(statements_path)
    names, recs = _records(run_id)
    pre, reuse, black, rr = (recs["preflight"], recs["stage2_reuse"],
                             recs["execution_blackbox"], recs["run_record"])
    glass_step = names[2]
    glass = recs[glass_step]
    recovered = glass_step == Recov.STEP
    tr = recs.get(T.STEP)
    side = {n: Run._sidecar(run_id, n) for n in names}
    rec = Run._header(run_id, STEP)
    rec["contract"] = _CONTRACT
    gate = C.Gate("A4.7")

    # -- 1. the records ----------------------------------------------------------------
    gate.check(f"every Step 4A record is present and matches its sidecar: {', '.join(names)}",
               lambda: len(recs) == len(names), [s[:16] for s in side.values()])
    gate.check("each record is this run's, under its own step name",
               lambda: all((r.get("run_id"), r.get("step")) == (run_id, n) for n, r in recs.items()))
    gate.check("the records are in order: each started after the previous one finished",
               lambda: all(recs[a]["finished_utc"] <= recs[b]["started_utc"]
                           for a, b in zip(names, names[1:])),
               lambda: [(n, recs[n]["started_utc"], recs[n]["finished_utc"]) for n in names])
    links = [("stage2_reuse.preflight_sha256", reuse.get("preflight_sha256"), "preflight"),
             ("run_record.gates.A4.1", rr["gates"]["A4.1"].get("sha256"), "preflight"),
             ("run_record.gates.A4.2", rr["gates"]["A4.2"].get("sha256"), "stage2_reuse"),
             ("run_record.executions.glass", rr["executions"]["glass"]["sha256"], glass_step),
             ("run_record.executions.blackbox", rr["executions"]["blackbox"]["sha256"],
              "execution_blackbox")]
    if recovered:
        links += [(f"{glass_step}.original.preflight_sha256",
                   glass["original"].get("preflight_sha256"), "preflight"),
                  (f"{glass_step}.original.stage2_reuse_sha256",
                   glass["original"].get("stage2_reuse_sha256"), "stage2_reuse"),
                  ("run_record.glass_recovery", (rr.get("glass_recovery") or {}).get("sha256"),
                   glass_step)]
    if tr is not None:
        links += [(f"{T.STEP}.records.{n}", tr["records"].get(n), n)
                  for n in ("preflight", "stage2_reuse", glass_step, "execution_blackbox")]
        links.append(("run_record.code_transition", (rr.get("code_transition") or {}).get("sha256"),
                      T.STEP))
    broken = [f"{where} != {target}" for where, value, target in links if value != side[target]]
    gate.check("every later record names the earlier ones by their sidecar SHA-256 "
               f"({len(links)} links)", lambda: not broken, lambda: broken or len(links))
    gate.check("the run record is the regeneration record, not an acceptance (accepted: false, "
               "Step 4B named)", lambda: rr.get("accepted") is False and "Step 4B" in rr["status"])
    gate.check("the run record's A4.1 and A4.2 are the preflight and stage2_reuse records; "
               "the Stage 2 decision is reuse",
               lambda: rr["gates"]["A4.1"]["passed"] is True and rr["gates"]["A4.2"]["passed"] is True
               and rr["gates"]["A4.2"].get("decision") == reuse["decision"]["outcome"] == "reuse"
               and rr["stage2_disposition"] == reuse["decision"])
    recorded_gates = {g: rr["gates"][g] for g in ("A4.3", "A4.4", "A4.5", "A4.6")}
    gate.check("the run record's A4.3-A4.6: every recorded check passed",
               lambda: all(r["passed"] and r["checks"] and all(c["passed"] for c in r["checks"])
                           for r in recorded_gates.values()),
               lambda: {g: r["evidence"] for g, r in recorded_gates.items()})

    # -- 2. the code (section 7.2) -----------------------------------------------------
    own = {r: None for r in acceptance_files}
    gate.check(f"this step's own files are present and new ({', '.join(acceptance_files)})",
               lambda: all((root / r).is_file() for r in acceptance_files)
               and not set(acceptance_files) & set(T.code_changes(run_id)))
    tr_changes = tr["code"]["changes"] if tr else []
    rv_changes = glass["code"]["changes"] if recovered else []
    ids: Dict[str, Any] = {"acceptance": Rec.code_identity(root)}

    def derive(name, overrides):
        ids[name] = Rec.code_identity(root, overrides=overrides)
        return _code(ids[name])

    run_code = rr["code"]
    gate.check("omitting those files reproduces the run record's code identity (the acceptance "
               "runs the run's validation code, unchanged)",
               lambda: derive("run", own) == _code(run_code),
               lambda: {"run_record": run_code.get("code_sha256"),
                        "derived": (ids.get("run") or {}).get("code_sha256")})
    recovered_code = glass["code"]["recovered_under"] if recovered else pre["code"]
    if tr is not None:
        gate.check("restoring the transition's recorded files as well gives the code it "
                   "transitioned from; the run record is the transitioned code",
                   lambda: derive("before_transition", {**own, **_before(tr_changes)})
                   == _code(tr["code"]["from"]) == _code(recovered_code)
                   and _code(tr["code"]["to"]) == _code(run_code),
                   lambda: (ids.get("before_transition") or {}).get("code_sha256"))
    if recovered:
        gate.check("restoring the recovery's recorded files as well gives the preflight code; "
                   "the recovery ran from it",
                   lambda: derive("preflight", {**own, **_before(tr_changes), **_before(rv_changes)})
                   == _code(pre["code"]) == _code(glass["code"]["run"]),
                   lambda: (ids.get("preflight") or {}).get("code_sha256"))
    gate.check("every step ran on the code the chain gives it (stage2_reuse: preflight's; the "
               "black-box execution: the recovered code)",
               lambda: _code(reuse["code"]) == _code(pre["code"])
               and _code(black["code"]) == _code(recovered_code))
    all_changes = list(rv_changes) + list(tr_changes)
    in_tr = {c["path"]: c for c in tr_changes}

    def as_recorded():
        for c in tr_changes:
            p = root / c["path"]
            if (Rec.file_sha256_lf(p) if p.is_file() else None) != c["after_lf_sha256"]:
                raise AssertionError(f"{c['path']} is not as the transition left it")
        for c in rv_changes:
            later = in_tr.get(c["path"])
            p = root / c["path"]
            now = later["before_lf_sha256"] if later else (Rec.file_sha256_lf(p) if p.is_file()
                                                           else None)
            if now != c["after_lf_sha256"]:
                raise AssertionError(f"{c['path']} is not as the recovery left it")
        return True

    gate.check("every recorded code change is under upstream_regeneration code or its tests, and "
               "each changed file is as its recovery or transition left it",
               lambda: all(c["path"].startswith(T.ALLOWED_PREFIXES) for c in all_changes)
               and as_recorded(), [c["path"] for c in all_changes])
    gate.check("the recovery and the transition are the registered ones (ids, resolved check, "
               "owner's statement)",
               lambda: (not recovered or glass["recovery_id"] == Recov.RECOVERY_ID)
               and (tr is None or (tr["transition_id"] == T.TRANSITION_ID
                                   and tr["resolves"] == Rec.canonical(T.RESOLVES)
                                   and bool(tr["authorization"]["statement"].strip()))))

    # -- 3. the notebook executions (section 5.4) --------------------------------------
    copies: Dict[str, Any] = {}
    for nb, step in (("glass", glass_step), ("blackbox", "execution_blackbox")):
        r, e = recs[step], rr["executions"][nb]
        gate.check(f"{nb}: the run record's execution entry is its record's (copies, truncated "
                   "copy)", lambda r=r, e=e: e["input_copy"] == r["execution"]["input_copy"]
                   and e["executed_copy"] == r["execution"]["executed_copy"]
                   and e["truncated_copy_sha256"] == r["truncated_copy"]["truncated_copy_sha256"])
        gate.check(f"{nb}: the committed notebook's truncated copy, rebuilt now, is the executed "
                   "one (the notebook is unchanged since)",
                   lambda nb=nb, r=r: Rec.canonical(V.truncated_copy(nb, root=root)[1])
                   == r["truncated_copy"])
        for kind in ("input_copy", "executed_copy"):
            p = Recov._copy(e[kind], copies_dir)
            got = _sha(p.read_bytes()) if p.is_file() else None
            copies[f"{nb}.{kind}"] = {"path": str(p), "sha256": got}
            want = [e[kind]["sha256"]] + ([e["truncated_copy_sha256"]] if kind == "input_copy" else [])
            gate.check(f"{nb}: the {kind.replace('_', ' ')} has its recorded SHA-256"
                       + (" (the truncated copy's)" if kind == "input_copy" else ""),
                       lambda got=got, want=want: got is not None and all(got == w for w in want),
                       str(p))

    # -- 4. recomputation: A4.2, then A4.3-A4.6 -----------------------------------------
    split = C.global_split()
    population = C.frozen_population()
    g2, ev2 = _recheck_stage2(split)
    decision = C.decide(pre, g2.record())

    def same(recomputed: C.Gate, recorded: Mapping[str, Any]) -> bool:
        mine = [(c["check"], c["passed"]) for c in recomputed.record()["checks"]]
        theirs = [(c["check"], c["passed"]) for c in recorded["checks"]]
        return recomputed.passed and mine == theirs

    gate.check("A4.2 recomputed now: every check passes, the same checks as stage2_reuse.json",
               lambda: same(g2, reuse["gates"]["A4.2"]),
               lambda: g2.failed() or g2.record()["evidence"])
    gate.check("A4.2 recomputed now: decision reuse, as recorded",
               lambda: Rec.canonical(decision) == reuse["decision"], decision.get("outcome"))
    d2 = _diff_keys(Rec.canonical(_pick(ev2, STAGE2_EVIDENCE)),
                    _pick(reuse["evidence"], STAGE2_EVIDENCE)) \
        if isinstance(ev2, Mapping) else ([] if Rec.canonical(ev2) == reuse["evidence"] else ["*"])
    gate.check("A4.2 recomputed evidence is the record's, value for value (PR 38 acceptance, "
               "Stage 2 manifest and pins, names and binning, router contracts, R4 notes)",
               lambda: not d2, lambda: d2 or list(STAGE2_EVIDENCE))
    now = _recheck_run(run_id, sums_path, pre, glass, black, split, population, acceptance_files)
    for g, recomputed in now["gates"].items():
        gate.check(f"{g} recomputed now: every check passes, the same checks as the run record",
                   lambda recomputed=recomputed, g=g: same(recomputed, recorded_gates[g]),
                   lambda recomputed=recomputed: recomputed.failed() or recomputed.record()["evidence"])
    ev = Rec.canonical({k: now["evidence"].get(k) for k in RUN_EVIDENCE})
    ident = Rec.canonical(C.identities(split, population))
    d_run = _diff_keys(ev, {k: rr["evidence"].get(k) for k in RUN_EVIDENCE}) + (
        [] if ident == rr["identities"] else ["identities"])
    gate.check("the recomputed Stage 1, Stage 3, Stage 4 input and reload evidence and the "
               "section 7.3 identities are the run record's, value for value",
               lambda: not d_run, lambda: d_run or list(RUN_EVIDENCE) + ["identities"])

    # -- 5. the artifacts ------------------------------------------------------------------
    pins = now["pins"]
    stage2 = {p.role: p.record() for p in STAGE2_PINS}
    want = {r: pins[r].record() for r in ROLES}
    pins_file = Rec.run_dir(run_id) / Run.PINS_FILE
    pinned: Dict[str, Any] = {}

    def agree():
        pinned.update(read_pins(pins_file).get("files") or {})

        def strip(d):
            return {r: {k: a[k] for k in ("path", "sha256", "bytes")} for r, a in d.items()}

        views = {"run record": (strip(rr["artifacts"]), set(ROLES)),
                 "execution records": ({**glass["produced"], **black["produced"], **stage2},
                                       set(ROLES)),
                 "pins file": (strip(pinned), {"stage1.lr", "stage3.ebm", *stage2})}
        if tr is not None:
            views["transition record"] = (strip(tr["artifacts"]), set(ROLES))
        bad = [name for name, (v, roles) in views.items()
               if set(v) != roles or any(v[r] != want[r] for r in roles)]
        if bad:
            raise AssertionError(f"disagree with the pins: {bad}")
        return True

    gate.check("the six Stage 1-3 artifacts have one path, size and SHA-256 in every record that "
               "names them (execution records, pins file, transition, run record) and on disk "
               "(A4.6 above)", agree, lambda: {r: want[r]["sha256"][:16] for r in ROLES})

    # -- 6. D7 and the inventory ------------------------------------------------------------
    after, status = now["after"], now["status"] or []
    d7: Dict[str, Any] = {}
    for rel in D7:
        line = next((ln for ln in status if _line_path(ln) == rel
                     or _line_path(ln).endswith("/" + rel)), None)
        before, ran = pre["inventory"].get(rel), black["inventory_after"].get(rel)
        d7[rel] = {"before_run": before, "after_run": ran, "now": after.get(rel),
                   "git_status": line[:2] if line else None,
                   "changed_by_run": before != ran,
                   "content_differs_from_head": line is not None,
                   "disposition": ("stage: its content differs from HEAD (D7: committed in the "
                                   "producing commit)") if line else
                                  "nothing to stage: Git reports its content equal to HEAD"}
    gate.check("D7: each tracked MLP CSV is as the black-box execution left it",
               lambda: all(v["now"] == v["after_run"] for v in d7.values()),
               lambda: {r: v["git_status"] for r, v in d7.items()})
    gate.check("D7: a CSV that differs from HEAD was changed by this run and was not already "
               "modified before it",
               lambda: all(not v["content_differs_from_head"]
                           or (v["changed_by_run"] and not any(
                               _line_path(ln) == r or _line_path(ln).endswith("/" + r)
                               for ln in pre["git_status"] or []))
                           for r, v in d7.items()))
    net = now["net"]
    d = I.diff(pre["inventory"], after)
    kinds = {**{p: ("role", role) for role, p in net["produced"].items()},
             **{p: ("companion", c) for c, p in net["companions"].items()},
             **{p: ("sidecar", None) for p in net["sidecars"]},
             **{p: ("mutable_alias", None) for p in net["mutable_aliases"]}}
    files = [{"path": p, "change": change, **after[p],
              "kind": kinds.get(p, ("unclassified", None))[0], "role": kinds.get(p, (None, None))[1]}
             for change in ("added", "modified") for p in d[change]]
    gate.check("every file the run added or changed in the model folders is classified (role, "
               "companion, sidecar or mutable alias); none deleted",
               lambda: not d["deleted"] and all(f["kind"] != "unclassified" for f in files),
               f"{len(files)} files")

    # -- 7. the owner's statements ----------------------------------------------------------
    gate.check("the owner's decision and the section 7.1 and 7.2 dispositions are stated",
               lambda: all(statements[k] for k in STATEMENT_KEYS),
               {k: statements[k][:80] for k in STATEMENT_KEYS})

    manifest = {
        "format": PINS_FORMAT, "kind": MANIFEST_KIND, "run_id": run_id, "contract": _CONTRACT,
        "files": {r: {**want[r], "producer": PRODUCERS[r], "consumers": Run.CONSUMERS[r]}
                  for r in ROLES},
        "stage2_source": {"run_id": STAGE2_RUN_ID, "manifest": STAGE2_MANIFEST,
                          "manifest_sha256": STAGE2_MANIFEST_SHA256,
                          "decision": reuse["decision"]["outcome"]},
        "inventory": {"base": "preflight.json inventory", "files": files},
        "run_record_sha256": side["run_record"],
        "accepted_by": f"{STEP}.json, written after this file; it names this file's SHA-256",
        "notes": ["Stage 4 reads each role by SHA-256: shared.pinned_artifacts.pinned_path(role, "
                  "pins_file=<this file>) (A5.1)",
                  "A mutable alias is never evidence (section 7.2); records name immutable "
                  "versions"],
    }
    rec.update(
        gates={"A4.7": gate.record(), "A4.2 recomputed": g2.record(),
               **{f"{g} recomputed": x.record() for g, x in now["gates"].items()}},
        decision={"accepted": bool(gate.passed), "scope": "Stage 1-3 artifacts of the run (A4.7)",
                  "statement": statements["decision"], "statements_file": {
                      "name": statements["file"], "sha256": statements["sha256"]}},
        dispositions={
            "7.1": {"statement": statements["disposition_7_1"],
                    "subject": "recover-glass parsed the failed GLASS execution record",
                    "failed_record": glass["original"]["failed_record"] if recovered else None,
                    "recovery_record_sha256": side.get(glass_step) if recovered else None,
                    "recovery_checks": (glass["gates"]["A4.R"]["evidence"] if recovered else None),
                    "transition_failed_attempt": (tr["original"]["failed_finalize"] if tr else None)},
            "7.2": {"statement": statements["disposition_7_2"],
                    "subject": "a registered code transition within the run",
                    "transition_record_sha256": side.get(T.STEP),
                    "authorization": tr["authorization"] if tr else None,
                    "codes": {k: (v or {}).get("code_sha256") for k, v in ids.items()}}},
        records={n: side[n] for n in names},
        artifacts={r: manifest["files"][r] for r in ROLES},
        code={"acceptance": ids["acceptance"], "acceptance_files": list(acceptance_files),
              "derived": {k: v for k, v in ids.items() if k != "acceptance"},
              "run_record": run_code},
        notebook_copies=copies, d7=d7, git_status_delta=now["git"],
        evidence_compared={"A4.2": list(STAGE2_EVIDENCE), "run_record": [*RUN_EVIDENCE,
                                                                          "identities"]},
        sums={"path": str(sums_path)}, environment=Rec.environment())
    if not gate.passed:
        return Run._finish(run_id, STEP, rec, False)
    mpath = Rec.record_path(run_id, MANIFEST)
    if mpath.exists():                         # a partial earlier acceptance: same bytes only
        if mpath.read_bytes() != Rec._text(manifest):
            raise Rec.RecordError(f"{mpath} exists with other content: records are written once")
    else:
        Rec.write_record(run_id, MANIFEST, manifest)
    rec["artifact_manifest"] = {"path": mpath.relative_to(root).as_posix()
                                if mpath.is_relative_to(root) else str(mpath),
                                "sha256": Run._sidecar(run_id, MANIFEST)}
    return Run._finish(run_id, STEP, rec, True)


# ----------------------------------------------------------------------
# re-verification of a written acceptance (Commit 4 review, Commit 5)
# ----------------------------------------------------------------------
def check(run_id: str) -> Dict[str, Any]:
    """Re-verify the acceptance and the manifest: sidecars, links, every role on disk."""
    from shared.pinned_artifacts import pinned_path, read_pins
    acc = Rec.require_passed(run_id, STEP)
    mpath = Rec.record_path(run_id, MANIFEST)
    m = read_pins(mpath)                                        # sidecar-checked
    problems = []
    if m.get("kind") != MANIFEST_KIND or m.get("run_id") != run_id:
        problems.append("the manifest is not this run's artifact manifest")
    if acc["artifact_manifest"]["sha256"] != Run._sidecar(run_id, MANIFEST):
        problems.append("the acceptance does not name this manifest")
    if acc["decision"]["accepted"] is not True:
        problems.append("the acceptance is not an acceptance")
    for name, sha in acc["records"].items():
        if Rec.require_passed(run_id, name) is None or Run._sidecar(run_id, name) != sha:
            problems.append(f"{name}.json is not the record the acceptance names")
    if m.get("run_record_sha256") != acc["records"].get("run_record"):
        problems.append("the manifest and the acceptance name different run records")
    if Rec.require_passed(run_id, "run_record").get("accepted") is not False:
        problems.append("run_record.json changed: it must keep accepted false")
    roles = {}
    for role in ROLES:
        pinned_path(role, pins_file=mpath, root=Rec.BANK_ROOT)            # size and SHA-256 on disk
        roles[role] = {k: m["files"][role][k] for k in ("path", "bytes", "sha256")}
    if problems:
        raise Rec.RecordError(f"run {run_id}: " + "; ".join(problems))
    return {"run_id": run_id, "accepted": True, "manifest_sha256": acc["artifact_manifest"]["sha256"],
            "acceptance_sha256": Run._sidecar(run_id, STEP), "roles": roles}


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m upstream_regeneration.acceptance",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("accept", help="A4.7: verify the run and record the acceptance decision")
    p.add_argument("--run-id", required=True)
    p.add_argument("--sums", required=True, help="the baseline SHA256SUMS.txt (section 7.5)")
    p.add_argument("--statements", required=True,
                   help=f"UTF-8 JSON with exactly {list(STATEMENT_KEYS)}, recorded verbatim")
    p.add_argument("--copies-dir", default=None,
                   help="where the four notebook copies are, if moved (default: recorded paths)")
    p = sub.add_parser("check", help="re-verify a written acceptance and its manifest")
    p.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "accept":
            rec = accept(args.run_id, args.sums, args.statements, copies_dir=args.copies_dir)
            print(f"OK {STEP} for run {args.run_id}: accepted")
            for g, r in rec["gates"].items():
                print(f"  {g}: {r['evidence']}")
            print(f"  {MANIFEST}.json sha256 {rec['artifact_manifest']['sha256']}")
            for rel, v in rec["d7"].items():
                print(f"  D7 {rel}: {v['disposition']}")
        else:
            out = check(args.run_id)
            print(f"OK {STEP} for run {args.run_id}: accepted")
            print(f"  {STEP}.json sha256 {out['acceptance_sha256']}")
            print(f"  {MANIFEST}.json sha256 {out['manifest_sha256']}")
            for role, f in out["roles"].items():
                print(f"  {role:20} {f['sha256']}  {f['bytes']:>10}  {f['path']}")
        return 0
    except Run.StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                        # noqa: BLE001 - a refusal, printed
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
