"""
upstream_regeneration.transition
================================
A registered code transition for ``finalize`` (PR 39 Commit 4A, run
c4_20261115_204125): the run's first ``finalize`` failed one A4.4 check,
because ``assert_matched_protocol`` compared the two Stage 3 arms'
``split_fingerprint`` strings, and the two notebooks stamp that field under
different registered schemes (GLASS Cell 17: ``stage_io``, 12 hex, rows and
labels; black-box Cell 30: ``router``, 16 hex, rows only). The correction
(``checks.stage3_split_identity``) is a change to ``upstream_regeneration``
itself, so it changes the run's code identity, which every step holds fixed.

    python -m upstream_regeneration finalize-transition --run-id <id> --authorization "<statement>" [--failed-record <path>]

This step authorizes exactly that change, once, and only after re-establishing
from primary evidence that nothing the run produced changed:

* preflight (A4.1), stage2_reuse (A4.2, ``reuse``), the GLASS execution
  (executed, or recovered) and the black-box execution passed and match their
  sidecars; there is no ``run_record.json`` and no earlier transition;
* a failed ``finalize`` attempt of this run exists. It is named by path, size
  and SHA-256 and is not read (execution contract 7.1: failed attempts are
  never read by code); ``finalize`` later checks that it is unchanged;
* the code differs from the run's code (preflight's, or the one a recorded
  recovery verified) only by this transition's files: restoring each file in
  ``CHANGES`` to its recorded SHA-256 reproduces that identity exactly, and
  every listed file is under ``upstream_regeneration/`` or
  ``tests/upstream_regeneration/``, which no notebook imports;
* the model folders are byte for byte as the black-box execution left them,
  and every produced and accepted Stage 2 file matches its pin;
* the corrected A4.4 passes on the run's pinned EBM and XGBoost.

Only then is ``finalize_transition.json`` written once (with a sidecar),
recording the owner's authorization statement, both code identities, every
changed file before and after, and the failed attempt's SHA-256. ``finalize``
then holds the code to the transitioned identity and rechecks A4.3-A4.6 in
full. Nothing is executed, fitted or rewritten here.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import checks as C
from . import inventory as I
from . import records as Rec

__all__ = ["TRANSITION_ID", "STEP", "RESOLVES", "CHANGES", "ALLOWED_PREFIXES",
           "failed_finalize_record", "run_code", "passed_record", "code_changes",
           "named_failed_attempts", "transition_finalize"]

TRANSITION_ID = ("commit4a-t1: A4.4 Stage 3 protocol comparison across the two registered "
                 "split-fingerprint schemes")
STEP = "finalize_transition"
#: What the transition corrects: the failed check (the run's code) and its replacements.
RESOLVES = {
    "gate": "A4.4",
    "failed_check": "stage 3: one shared protocol, family-specific fields only "
                    "(assert_matched_protocol)",
    "replaced_by": [C.STAGE3_SPLIT_IDENTITY, C.STAGE3_PROTOCOL],
    "cause": "assert_matched_protocol compares split_fingerprint as a string; GLASS Cell 17 "
             "stamps the stage_io scheme and black-box Cell 30 the router scheme "
             "(shared.stage4.FINGERPRINT_SCHEMES). Both are checked against GLOBAL_SPLIT "
             "under their own scheme with check_split_identity, the artifacts' recorded rows "
             "and labels are compared with GLOBAL_SPLIT's, and only then is split_fingerprint "
             "exempt from the string comparison. shared.stage3.protocol is unchanged.",
}
#: The transition patch's code: relative path -> LF SHA-256 before the patch (None: a new file).
#: Restoring these, and nothing else, must reproduce the run's code identity.
CHANGES: Dict[str, Optional[str]] = {
    "tests/upstream_regeneration/test_finalize_transition.py": None,
    "tests/upstream_regeneration/test_stage3_split_identity.py": None,
    "tests/upstream_regeneration/test_upstream_checks.py":
        "8de54fbc9092379d3a8b20b4ca128d8aae1cd54f39f22e602bfc7eaee6cce883",
    "upstream_regeneration/__init__.py":
        "52756a446f087e11e30f22442e4cd2b3d5c153c52e7b1bf30a9f4abd9b0d7995",
    "upstream_regeneration/checks.py":
        "c35fba69dc37f26a61b029f64eb425a8b2107ed4e91c3af73af300074a57e3fa",
    "upstream_regeneration/runner.py":
        "20b175d1da6cd78072bd58417bc2a0acfa8efb6439d0859462f96edc7717f7c6",
    "upstream_regeneration/transition.py": None,
}
ALLOWED_PREFIXES = ("upstream_regeneration/", "tests/upstream_regeneration/")
_FAILED = re.compile(r"run_record_[0-9]{8}T[0-9]{12}Z\.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def failed_finalize_record(run_id: str, path=None) -> Path:
    """The run's one failed ``finalize`` attempt (or ``path``, which must be one of them)."""
    folder = Rec.run_dir(run_id) / "failed"
    found = sorted(f for f in folder.iterdir() if _FAILED.fullmatch(f.name)) \
        if folder.is_dir() else []
    if path is not None:
        p = Path(path).resolve()
        if p not in [f.resolve() for f in found]:
            raise Rec.RecordError(f"{path} is not a failed finalize attempt of run {run_id} "
                                  f"(in {folder})")
        return p
    if len(found) != 1:
        raise Rec.RecordError(f"run {run_id} has {len(found)} failed finalize attempts "
                              f"{[f.name for f in found]}: name one with --failed-record")
    return found[0]


def passed_record(run_id: str) -> Optional[Dict[str, Any]]:
    """The run's transition record (sidecar verified), or None if it has none."""
    return Rec.require_passed(run_id, STEP) if Rec.record_path(run_id, STEP).exists() else None


def run_code(run_id: str, pre: Mapping[str, Any]) -> Dict[str, Any]:
    """The code identity a run's later steps must have: preflight's, recovered or transitioned."""
    from .recovery import run_code as before_transition
    code = before_transition(run_id, pre)
    rec = passed_record(run_id)
    if rec is None:
        return code
    if rec["code"]["from"] != code:
        raise Rec.RecordError(f"run {run_id}: the transition record's from-code is not the "
                              "run's code before it")
    return rec["code"]["to"]


def code_changes(run_id: str) -> list:
    """Every code file a recorded recovery or transition changed (for the git status check)."""
    from .recovery import STEP as RECOVERED
    out = []
    for name in (RECOVERED, STEP):
        if Rec.record_path(run_id, name).exists():
            out += [c["path"] for c in Rec.require_passed(run_id, name)["code"]["changes"]]
    return sorted(set(out))


def named_failed_attempts(run_id: str) -> list:
    """The failed attempts a recorded recovery or transition names: path, SHA-256, by which."""
    from .recovery import STEP as RECOVERED
    out = []
    if Rec.record_path(run_id, RECOVERED).exists():
        f = Rec.require_passed(run_id, RECOVERED)["original"]["failed_record"]
        out.append({"by": RECOVERED, "path": f["path"], "sha256": f["sha256"]})
    rec = passed_record(run_id)
    if rec is not None:
        f = rec["original"]["failed_finalize"]
        out.append({"by": STEP, "path": f["path"], "sha256": f["sha256"]})
    return out


def _sidecar(run_id: str, name: str) -> str:
    p = Rec.record_path(run_id, name)
    return p.with_name(p.name + ".sha256").read_text(encoding="ascii").strip()


def transition_finalize(run_id: str, *, authorization: str, failed_record=None,
                        changes: Mapping[str, Optional[str]] = None) -> Dict[str, Any]:
    import joblib
    from shared.pinned_artifacts import PinnedFile, STAGE2_PINS, verify_pin
    from . import runner as Run
    from .recovery import glass_execution, run_code as before_transition
    changes = dict(CHANGES if changes is None else changes)
    root = Rec.BANK_ROOT
    pre = Rec.require_passed(run_id, "preflight")
    reuse = Rec.require_passed(run_id, "stage2_reuse")
    glass_step, glass = glass_execution(run_id)
    black = Rec.require_passed(run_id, "execution_blackbox")
    for name, why in (("run_record", "finalize passed: there is nothing to transition"),
                      (STEP, "the code transition is already recorded (records are written "
                             "once)")):
        if Rec.record_path(run_id, name).exists():
            raise Rec.RecordError(f"run {run_id}: {why}")
    failed_path = failed_finalize_record(run_id, failed_record)
    raw = failed_path.read_bytes()                     # hashed only, never parsed (7.1)
    rec = Run._header(run_id, STEP)
    rec["transition_id"] = TRANSITION_ID
    gate = C.Gate("A4.T")

    # -- the run's records and the authorization ----------------------------------------
    gate.check("A4.1 preflight, A4.2 stage2_reuse (decision reuse), the GLASS and black-box "
               "executions passed (sidecars verified)",
               lambda: reuse["decision"]["outcome"] == "reuse"
               and all(r["execution"]["completed"] for r in (glass, black)))
    gate.check("a failed finalize attempt of this run is on record (named by path and SHA-256, "
               "not read: contract 7.1)", lambda: bool(raw), failed_path.name)
    statement = (authorization or "").strip()
    gate.check("the owner's authorization of this transition is stated",
               lambda: 0 < len(statement) <= 500, statement[:120])

    # -- the code: the run's, plus only this transition's files -------------------------
    want = before_transition(run_id, pre)
    now = Rec.code_identity(root)
    gate.check("the transition's changes are confined to upstream_regeneration code and its "
               "tests", lambda: bool(changes) and all(r.startswith(ALLOWED_PREFIXES)
                                                    for r in changes))
    gate.check("every recorded change is present (new files exist; changed files differ)",
               lambda: all((root / r).is_file() and (base is None or Rec.file_sha256_lf(root / r)
                                                      != base) for r, base in changes.items()))
    undone: Dict[str, Any] = {}

    def reproduces():
        undone.update(Rec.code_identity(root, overrides=changes))
        return (undone["code_sha256"], undone["head"]) == (want["code_sha256"], want["head"])

    gate.check("restoring exactly those files reproduces the run's code identity "
               f"({'recovered' if glass_step != 'execution_glass' else 'preflight'})",
               reproduces, lambda: {"run": want["code_sha256"],
                                    "restored": undone.get("code_sha256")})

    # -- nothing the run produced changed -------------------------------------------------
    inv = I.snapshot(root)
    gate.check("the model folders are byte for byte as the black-box execution left them",
               lambda: I.diff(black["inventory_after"], inv) == {"added": [], "modified": [],
                                                                 "deleted": []},
               lambda: {k: v[:5] for k, v in I.diff(black["inventory_after"], inv).items()})
    pins = {p.role: p for p in STAGE2_PINS}
    for role, r in {**glass["produced"], **black["produced"]}.items():
        pins[role] = PinnedFile(role, r["path"], r["sha256"], int(r["bytes"]))
    for role, pin in sorted(pins.items()):
        gate.check(f"{role}: {pin.path} matches its pin (SHA-256, size)",
                   lambda pin=pin: verify_pin(pin, root) is not None, pin.sha256[:16])

    # -- the correction resolves the failure, on the run's own Stage 3 files ------------
    a44 = C.Gate("A4.4")
    stage3: Dict[str, Any] = {}

    def corrected():
        arts = {r: joblib.load(pins[r].resolve(root)) for r in ("stage3.ebm", "stage3.xgb")}
        stage3.update(C.stage3_checks(a44, arts, C.frozen_population(), split=C.global_split()))
        return a44.passed

    gate.check("the corrected A4.4 passes on the run's pinned EBM and XGBoost (finalize "
               "rechecks it with A4.3, A4.5 and A4.6)",
               corrected if gate.passed else (lambda: False),
               lambda: (a44.failed() or a44.record()["evidence"]) if a44.checks
               else "not run: an earlier check failed")
    rec.update(
        gates={"A4.T": gate.record(), "A4.4 (corrected; recorded, not the run's A4.4)":
               a44.record()},
        authorization={"statement": statement,
                       "given_by": "the owner, by running finalize-transition"},
        resolves=RESOLVES,
        original={"failed_finalize": {"path": failed_path.relative_to(root).as_posix()
                                      if failed_path.is_relative_to(root) else str(failed_path),
                                      "bytes": len(raw), "sha256": _sha(raw),
                                      "read": False}},
        records={name: _sidecar(run_id, name) for name in
                 ("preflight", "stage2_reuse", glass_step, "execution_blackbox")},
        artifacts={role: pin.record() for role, pin in sorted(pins.items())},
        evidence={"stage3": stage3},
        code={"from": want, "to": now, "restored_identity": undone or None,
              "changes": [{"path": r, "before_lf_sha256": b,
                           "after_lf_sha256": Rec.file_sha256_lf(root / r)
                           if (root / r).is_file() else None} for r, b in sorted(changes.items())]})
    return Run._finish(run_id, STEP, rec, gate.passed)
