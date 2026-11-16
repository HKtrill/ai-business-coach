"""
shared.pinned_artifacts
=======================
Artifacts read by pinned path and SHA-256, never as "the newest file" (PR 39,
execution contract sections 5.3 and 7.1, finding F9).

Stage 2: the accepted PR 38 primary run
    ``STAGE2_RUN_ID``, its version-controlled ``primary_MANIFEST.json``
    (``STAGE2_MANIFEST``, pinned by ``STAGE2_MANIFEST_SHA256``) and the two
    files the notebooks consume (``STAGE2_PINS``: the GLASS Router bundle and
    the immutable RF Router version, each with its path, size and SHA-256) are
    code constants. ``load_pinned_stage2``:

    1. checks the manifest's SHA-256 and that both pins are its entries;
    2. reads each file once, checks its size and SHA-256 and unpickles those
       same bytes (a file cannot change between the check and the load);
    3. re-verifies the run's official structural acceptance
       (``stage2_crossfit.acceptance.require_accepted``) and that it names the
       pinned files;
    4. given the notebook's ``GLOBAL_SPLIT``, runs each router through the
       Stage 4 router contract (``router_stream`` and ``build_stage4_inputs``,
       train and test side).

    Nothing is fitted, and no Stage 2 file outside the pins is read.

A run's pins: files produced earlier in the same run
    ``$BANK_UPSTREAM_PINS`` names a pins file (``PINS_FORMAT``, with a
    ``.sha256`` sidecar) mapping a role (``ROLES``) to a path relative to
    bank_pipeline/, its size and SHA-256. ``pinned_path(role)`` returns the
    verified path. A missing variable, file, sidecar or role, a pins file that
    changed since it was written, or an artifact that changed since it was
    pinned is refused. ``upstream_regeneration`` writes pins files; the
    notebooks only read them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["BANK_ROOT", "PinError", "PinnedFile", "PinnedStage2", "STAGE2_RUN_ID",
           "STAGE2_MANIFEST", "STAGE2_MANIFEST_SHA256", "STAGE2_PINS", "ROLES", "PINS_ENV",
           "PINS_FORMAT", "sha256_bytes", "sha256_file", "read_verified", "verify_pin",
           "stage2_manifest_problems", "load_pinned_stage2", "write_pins", "read_pins",
           "pinned_path"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: The accepted PR 38 Stage 2 run (V1-V8 232/232, G7 passed) PR 39 reuses (D3).
STAGE2_RUN_ID = "c3_20261112_153333"
#: Its version-controlled candidate manifest and that file's SHA-256.
STAGE2_MANIFEST = f"research_logs/pr38/{STAGE2_RUN_ID}/primary_MANIFEST.json"
STAGE2_MANIFEST_SHA256 = "f70f12dc0380221563ced5f77e3caa5c156fa4d85dd4a3fca855260e77362a88"
_STAGE2_ROOT = f"models/pr38/{STAGE2_RUN_ID}/primary"

#: Pins file format and the variable that names it.
PINS_FORMAT = "pr39.upstream_pins/1"
PINS_ENV = "BANK_UPSTREAM_PINS"

#: Every role a pins file may hold: the stage, the arm and the producing cell.
ROLES = {
    "stage1.lr": "GLASS notebook Cell 8 (train_calibrated_lr_stage)",
    "stage1.mlp": "black-box notebook Cell 14A (save_stage1_mlp)",
    "stage2.glass_router": f"PR 38 primary run {STAGE2_RUN_ID} (stage2_crossfit)",
    "stage2.rf_router": f"PR 38 primary run {STAGE2_RUN_ID} (stage2_crossfit)",
    "stage3.ebm": "GLASS notebook Cell 18 (train_ebm_stage)",
    "stage3.xgb": "black-box notebook Cell 31 (train_xgb_stage)",
    "stage4.glass_arbiter": "GLASS notebook Cell 19 (train_glass_arbiter)",
}


class PinError(RuntimeError):
    """A pinned artifact is missing, changed or not pinned; the message says which."""


@dataclass(frozen=True)
class PinnedFile:
    role: str
    path: str                       # relative to bank_pipeline/, POSIX separators
    sha256: str
    bytes: int

    def resolve(self, root: Path = BANK_ROOT) -> Path:
        return Path(root) / PurePosixPath(self.path)

    def record(self) -> Dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


#: The two files Stage 4 consumes from the accepted run, as its manifest lists them.
STAGE2_PINS = (
    PinnedFile("stage2.glass_router",
               f"{_STAGE2_ROOT}/glass_router/glass_router_pr38_primary_{STAGE2_RUN_ID}.joblib",
               "889c8bd572047bebe83a9e60d4de564194d142ae9e91f2faf8750127d05bb753", 2481389),
    PinnedFile("stage2.rf_router",
               f"{_STAGE2_ROOT}/rf_router/rf_router_pr38_primary_{STAGE2_RUN_ID}"
               "_20261113_035911_620368.joblib",
               "aadde1ea93d0e300ca238f5b9f8844fa26646e131931e6c0e7d79714d043632e", 4158905),
)


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_relative(rel: str) -> PurePosixPath:
    p = PurePosixPath(str(rel))
    if p.is_absolute() or ".." in p.parts or "\\" in str(rel) or not p.parts:
        raise PinError(f"{rel!r} is not a plain path relative to bank_pipeline/")
    return p


def read_verified(pin: PinnedFile, root: Path = BANK_ROOT) -> bytes:
    """The pinned file's bytes, after checking its size and SHA-256."""
    _plain_relative(pin.path)
    path = pin.resolve(root)
    if not path.is_file():
        raise PinError(f"{pin.role}: {pin.path} does not exist (pinned {pin.sha256[:16]})")
    data = path.read_bytes()
    if len(data) != pin.bytes:
        raise PinError(f"{pin.role}: {pin.path} has {len(data):,} bytes, pinned {pin.bytes:,}")
    got = sha256_bytes(data)
    if got != pin.sha256:
        raise PinError(f"{pin.role}: {pin.path} SHA-256 {got[:16]} is not the pinned "
                       f"{pin.sha256[:16]} (changed since it was pinned)")
    return data


def verify_pin(pin: PinnedFile, root: Path = BANK_ROOT) -> Path:
    """The pinned file's path, after checking its size and SHA-256."""
    read_verified(pin, root)
    return pin.resolve(root)


def _load_bytes(data: bytes):
    import joblib
    return joblib.load(io.BytesIO(data))


# ----------------------------------------------------------------------
# Stage 2: the accepted PR 38 primary
# ----------------------------------------------------------------------
def stage2_manifest_problems(root: Path = BANK_ROOT, *, pins: Sequence[PinnedFile] = STAGE2_PINS,
                             manifest: str = STAGE2_MANIFEST,
                             manifest_sha256: str = STAGE2_MANIFEST_SHA256,
                             candidate_root: str = _STAGE2_ROOT) -> List[str]:
    """Why the pins are not the accepted run's manifest entries (empty: they are)."""
    path = Path(root) / PurePosixPath(manifest)
    if not path.is_file():
        return [f"{manifest} does not exist"]
    data = path.read_bytes()
    out = []
    if sha256_bytes(data) != manifest_sha256:
        out.append(f"{manifest} SHA-256 {sha256_bytes(data)[:16]} is not the pinned "
                   f"{manifest_sha256[:16]}")
    files = (json.loads(data.decode("utf-8")).get("files") or {})
    prefix = candidate_root.rstrip("/") + "/"
    for pin in pins:
        if not pin.path.startswith(prefix):
            out.append(f"{pin.role}: {pin.path} is not under {prefix}")
            continue
        entry = files.get(pin.path[len(prefix):])
        if entry != {"bytes": pin.bytes, "sha256": pin.sha256}:
            out.append(f"{pin.role}: {pin.path} is not listed with this size and SHA-256 in "
                       f"{manifest} (listed: {entry})")
    return out


@dataclass
class PinnedStage2:
    """The accepted Stage 2 routers, verified and loaded (``load_pinned_stage2``)."""

    run_id: str
    glass_path: Path
    rf_path: Path
    glass_bundle: Any
    rf_artifact: Any
    pins: Dict[str, Dict[str, Any]]
    acceptance: str
    stage4_contract: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"Stage 2 source: accepted PR 38 run {self.run_id} ({self.acceptance})"]
        for role, rec in self.pins.items():
            lines.append(f"  {role:<20} {rec['path']}  sha256 {rec['sha256'][:16]}…")
        for name, c in self.stage4_contract.items():
            lines.append(f"  Stage 4 contract [{name}]: {c['checks']} checks passed; Pass 2 "
                         f"population {c['pass2_population']}")
        return "\n".join(lines)


def _require_acceptance(run_id: str, pins: Sequence[PinnedFile], root: Path) -> str:
    from stage2_crossfit import acceptance
    candidate = acceptance.require_accepted(run_id)
    named = {Path(candidate.glass_path).resolve(), Path(candidate.rf_path).resolve()}
    want = {p.resolve(root).resolve() for p in pins}
    if named != want:
        raise PinError(f"run {run_id}'s acceptance names {sorted(map(str, named))}, "
                       f"not the pinned {sorted(map(str, want))}")
    if candidate.manifest_sha256 is None:                       # pragma: no cover
        raise PinError(f"run {run_id}: no candidate manifest")
    return "official structural acceptance re-verified (require_accepted)"


def _stage4_router_contract(GLOBAL_SPLIT, loaded: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from shared.stage4 import build_stage4_inputs, router_stream
    y = np.asarray(GLOBAL_SPLIT["y_train"]).astype(int)
    out = {}
    for name, (label, art) in loaded.items():
        stream, checks = router_stream(art, name, label, y)
        inputs = build_stage4_inputs(GLOBAL_SPLIT, [stream], checks, verbose=False)
        out[name] = {"checks": len(inputs.validation),
                     "pass2_population": stream.notes.get("pass2_population"),
                     "label_evidence": stream.notes.get("label_evidence")}
    return out


def load_pinned_stage2(GLOBAL_SPLIT: Optional[dict] = None, *, root: Path = BANK_ROOT,
                       pins: Sequence[PinnedFile] = STAGE2_PINS, run_id: str = STAGE2_RUN_ID,
                       manifest: str = STAGE2_MANIFEST,
                       manifest_sha256: str = STAGE2_MANIFEST_SHA256,
                       candidate_root: str = _STAGE2_ROOT,
                       require_acceptance: bool = True) -> PinnedStage2:
    """
    The accepted Stage 2 GLASS Router bundle and RF Router (module docstring).

    The keyword arguments other than ``GLOBAL_SPLIT`` exist for tests on
    synthetic files; the notebooks pass none of them.
    """
    root = Path(root)
    problems = stage2_manifest_problems(root, pins=pins, manifest=manifest,
                                        manifest_sha256=manifest_sha256,
                                        candidate_root=candidate_root)
    if problems:
        raise PinError("the Stage 2 pins are not the accepted run's:\n  - " + "\n  - ".join(problems))
    by_role = {p.role: p for p in pins}
    missing = {"stage2.glass_router", "stage2.rf_router"} - set(by_role)
    if missing:
        raise PinError(f"Stage 2 pins missing {sorted(missing)}")
    loaded = {role: _load_bytes(read_verified(by_role[role], root))
              for role in ("stage2.glass_router", "stage2.rf_router")}
    acceptance = (_require_acceptance(run_id, pins, root) if require_acceptance
                  else "acceptance NOT re-verified (require_acceptance=False)")
    contract = {}
    if GLOBAL_SPLIT is not None:
        contract = _stage4_router_contract(GLOBAL_SPLIT, {
            "glass": ("GLASS Router", loaded["stage2.glass_router"]),
            "rf_router": ("RF Router", loaded["stage2.rf_router"])})
    return PinnedStage2(run_id=run_id,
                        glass_path=by_role["stage2.glass_router"].resolve(root),
                        rf_path=by_role["stage2.rf_router"].resolve(root),
                        glass_bundle=loaded["stage2.glass_router"],
                        rf_artifact=loaded["stage2.rf_router"],
                        pins={p.role: p.record() for p in pins},
                        acceptance=acceptance, stage4_contract=contract)


# ----------------------------------------------------------------------
# a run's pins file
# ----------------------------------------------------------------------
_RUN_ID = re.compile(r"^c[0-9]+_[0-9]{8}_[0-9]{6}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_pins(path, run_id: str, files: Mapping[str, PinnedFile], *,
               root: Path = BANK_ROOT) -> Path:
    """Write a pins file once (never replacing one), each file verified first."""
    path = Path(path)
    if not _RUN_ID.match(run_id):
        raise PinError(f"run id {run_id!r} is not c<commit>_<YYYYMMDD_HHMMSS>")
    unknown = sorted(set(files) - set(ROLES))
    if unknown:
        raise PinError(f"unknown roles {unknown}; expected some of {sorted(ROLES)}")
    for role, pin in files.items():
        if pin.role != role:
            raise PinError(f"role {role!r} holds a pin for {pin.role!r}")
        verify_pin(pin, root)
    record = {"format": PINS_FORMAT, "run_id": run_id, "written_utc": _now(),
              "files": {role: {**files[role].record(), "producer": ROLES[role]}
                        for role in sorted(files)}}
    text = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:                               # never replaces a pins file
        fh.write(text)
    with open(path.with_name(path.name + ".sha256"), "x", encoding="ascii") as fh:
        fh.write(sha256_bytes(text) + "\n")
    return path


def read_pins(pins_file=None) -> Dict[str, Any]:
    """The pins file named by ``pins_file`` or ``$BANK_UPSTREAM_PINS``, sidecar-checked."""
    if pins_file is None:
        value = os.environ.get(PINS_ENV)
        if not value:
            raise PinError(f"${PINS_ENV} is not set: it must name the run's pins file "
                           "(python -m upstream_regeneration writes it); no artifact is "
                           "selected as the newest file")
        pins_file = value
    path = Path(pins_file)
    if not path.is_file():
        raise PinError(f"pins file {path} does not exist")
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.is_file():
        raise PinError(f"pins file {path} has no {side.name} sidecar")
    if side.read_text(encoding="ascii").strip() != sha256_bytes(data):
        raise PinError(f"pins file {path} does not match its {side.name} sidecar (edited)")
    record = json.loads(data.decode("utf-8"))
    if record.get("format") != PINS_FORMAT or not _RUN_ID.match(str(record.get("run_id"))):
        raise PinError(f"{path} is not a {PINS_FORMAT} pins file of a PR 39 run")
    return record


def pinned_path(role: str, *, pins_file=None, root: Path = BANK_ROOT) -> Path:
    """The verified path the run pinned for ``role`` (``ROLES``)."""
    if role not in ROLES:
        raise PinError(f"unknown role {role!r}; expected one of {sorted(ROLES)}")
    record = read_pins(pins_file)
    entry = (record.get("files") or {}).get(role)
    if not isinstance(entry, Mapping):
        raise PinError(f"run {record['run_id']} pinned no {role} ({ROLES[role]}); pinned: "
                       f"{sorted(record.get('files') or {})}")
    try:
        pin = PinnedFile(role, str(entry["path"]), str(entry["sha256"]), int(entry["bytes"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PinError(f"run {record['run_id']}: malformed pin for {role}: {exc}") from None
    return verify_pin(pin, root)
