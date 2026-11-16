"""
upstream_regeneration.records
=============================
Run ids, the record folder, write-once records with SHA-256 sidecars, and the
code and environment identities every PR 39 record carries (execution
contract sections 7.1-7.3).
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = ["BANK_ROOT", "RECORD_ROOT", "RUN_ID", "CONTRACT", "RecordError", "new_run_id",
           "check_run_id", "run_dir", "now", "canonical", "record_path", "write_record",
           "write_failed", "read_record", "require_passed", "code_files", "file_sha256_lf",
           "code_identity", "environment", "git_status"]

BANK_ROOT = Path(__file__).resolve().parents[1]
#: PR 39 records (version-controlled, byte-exact: research_logs/pr39/.gitattributes).
RECORD_ROOT = BANK_ROOT / "research_logs" / "pr39"
#: Commit 4 run ids (section 7.2): c4_<YYYYMMDD_HHMMSS>.
RUN_ID = re.compile(r"^c4_[0-9]{8}_[0-9]{6}$")
CONTRACT = {"record": "documentation/methodology/pr39_execution_contract.md", "version": "1.0.0",
            "sections": ["5.1", "5.2", "5.3", "5.4", "6.3", "6.4", "7.1", "7.2", "7.3", "7.4",
                         "7.5"],
            "gates": ["A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6"], "decisions": ["D3", "D4", "D7"]}
#: Packages whose Python source is "the code" of a run (plus the two notebooks and the
#: baseline CLIs): the code identity must not change between a run's steps.
CODE_ROOTS = ("shared", "data", "glass_pipeline", "blackbox_pipeline", "feature_research",
              "stage2_crossfit", "stage2_binning", "upstream_regeneration", "tests")
CODE_FILES = ("glass_pipeline/glass_cascade.ipynb", "blackbox_pipeline/blackbox_cascade.ipynb",
              "research_logs/baselines/stage2_baseline.py",
              "research_logs/baselines/stage2_baseline_archive.py")


class RecordError(RuntimeError):
    """A record cannot be written or read as required; the reason is stated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return "c4_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise RecordError(f"run id {run_id!r} is not c4_<YYYYMMDD_HHMMSS> (section 7.2)")
    return run_id


def run_dir(run_id: str, root: Optional[Path] = None) -> Path:
    return Path(RECORD_ROOT if root is None else root) / check_run_id(run_id)


def canonical(obj) -> Any:
    """JSON-safe, NaN-free, as the PR 38 and Step 3A records are written."""
    from stage2_crossfit.evidence import to_json_safe
    return json.loads(json.dumps(to_json_safe(obj), allow_nan=False))


def record_path(run_id: str, name: str, root: Optional[Path] = None) -> Path:
    return run_dir(run_id, root) / f"{name}.json"


def _text(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(canonical(record), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def write_record(run_id: str, name: str, record: Mapping[str, Any], *,
                 root: Optional[Path] = None) -> Path:
    """Write ``<run>/<name>.json`` and its ``.sha256`` sidecar once; never replace either."""
    path = record_path(run_id, name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _text(record)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise RecordError(f"{path} exists: records are written once (a new attempt gets a "
                          "new run id)") from None
    with open(path.with_name(path.name + ".sha256"), "x", encoding="ascii") as fh:
        fh.write(hashlib.sha256(data).hexdigest() + "\n")
    return path


def write_failed(run_id: str, name: str, record: Mapping[str, Any], *,
                 root: Optional[Path] = None) -> Path:
    """A failed or stopped step: ``<run>/failed/<name>_<utc>.json`` (Git-ignored, never read)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = run_dir(run_id, root) / "failed" / f"{name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(_text(record))
    return path


def read_record(run_id: str, name: str, *, root: Optional[Path] = None) -> Dict[str, Any]:
    """``<run>/<name>.json``, refused unless its sidecar matches (never edited)."""
    path = record_path(run_id, name, root)
    if not path.is_file():
        raise RecordError(f"no {name} record at {path}: run that step first")
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.is_file() or side.read_text(encoding="ascii").strip() != hashlib.sha256(data).hexdigest():
        raise RecordError(f"{path} does not match its {side.name} sidecar")
    rec = json.loads(data.decode("utf-8"))
    if rec.get("run_id") != run_id:
        raise RecordError(f"{path} is not run {run_id}'s")
    return rec


def require_passed(run_id: str, name: str, *, root: Optional[Path] = None) -> Dict[str, Any]:
    rec = read_record(run_id, name, root=root)
    if rec.get("passed") is not True:
        raise RecordError(f"run {run_id}: {name} did not pass; the run stops there")
    return rec


# ----------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------
def _git(*args, cwd: Path = BANK_ROOT) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_status(root: Optional[Path] = None) -> Optional[list]:
    """``git status --porcelain`` lines for bank_pipeline/ (untracked included), sorted."""
    root = BANK_ROOT if root is None else root
    out = _git("status", "--porcelain=v1", "--untracked-files=all", "--", ".", cwd=root)
    return None if out is None else sorted(line for line in out.splitlines() if line.strip())


def code_files(root: Optional[Path] = None) -> list:
    """Every code file of the run, in the order ``code_identity`` hashes them."""
    root = Path(BANK_ROOT if root is None else root)
    files = []
    for top in CODE_ROOTS:
        files += [p for p in (root / top).rglob("*.py") if "__pycache__" not in p.parts]
    files += [root / f for f in CODE_FILES if (root / f).is_file()]
    return sorted(set(files))


def file_sha256_lf(path) -> str:
    """A code file's SHA-256 with CRLF line endings normalised to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_identity(root: Optional[Path] = None, *,
                  overrides: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, Any]:
    """
    The commit, and one SHA-256 over every code file (LF-normalised), per section 7.3.

    ``overrides`` (relative path -> LF SHA-256, or None to leave the file out)
    recomputes the identity the tree had before a recorded set of changes,
    with every other file read as it is now (``recovery``).
    """
    root = Path(BANK_ROOT if root is None else root)
    overrides = dict(overrides or {})
    h, n = hashlib.sha256(), 0
    for p in code_files(root):
        rel = p.relative_to(root).as_posix()
        digest = overrides.pop(rel, file_sha256_lf(p)) if rel in overrides else file_sha256_lf(p)
        if digest is None:
            continue
        h.update(rel.encode("utf-8") + b"\0")
        h.update(bytes.fromhex(digest))
        n += 1
    for rel, digest in overrides.items():          # an override for a file that is not here
        if digest is not None:
            raise RecordError(f"{rel} is not a code file of this tree; it cannot be restored")
    head = _git("rev-parse", "HEAD", cwd=root)
    return {"head": None if head is None else head.strip(), "code_sha256": h.hexdigest(),
            "files": n}


def environment() -> Dict[str, Any]:
    """Python, the section 7.3 libraries and the operating system."""
    import sys
    from importlib import metadata
    versions = {}
    for dist in ("numpy", "pandas", "scikit-learn", "scipy", "interpret", "xgboost", "optuna",
                 "pulp", "joblib", "ipykernel", "nbclient", "nbformat"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "os": platform.system(), "packages": versions}
