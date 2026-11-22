"""
feature_research.research_records
=================================
PR 42 research records: one folder per run under ``research_logs/pr42/``,
written once (PR 42, Commit 2).

    run = start_run(commit=2)              # run id, seeds, code identity, environment
    ...                                    # the notebook's work
    path = write_run_record(run, "feature_inventory", sections)

Conventions (the same as the PR 39 records, ``research_logs/pr39/README.md``)
---------------------------------------------------------------------------
* A run id is ``c<commit>_<YYYYMMDD_HHMMSS>``; every attempt gets a new one.
* A record is ``<run_id>/<name>.json`` plus a ``.sha256`` sidecar. The run
  folder must not exist when the record is written and both files are opened
  in exclusive-create mode, so nothing is ever written over an existing
  record: a repeated run, or a run id reused by mistake, is refused.
* JSON is canonical (sorted keys, NaN-free) through
  ``upstream_regeneration.records.canonical``, so two runs on the same code
  and data produce the same bytes for the same content.

Identity recorded in every record
---------------------------------
``run``            run id, UTC start and finish, the notebook that wrote it
``seeds``          every seed the run used, by name
``code``           Git HEAD, whether tracked files had uncommitted changes, the
                   LF-normalised SHA-256 of each file in ``CODE_FILES`` and the
                   whole-tree digest of ``upstream_regeneration.records.code_identity``
``environment``    Python, platform and package versions
``content_sha256`` SHA-256 of the canonical content sections only (no run id,
                   times, code or environment): two runs that reproduce each
                   other have the same value.

Nothing here reads a split, a model artifact or a label.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = ["RECORD_ROOT", "RUN_ID", "CODE_FILES", "RecordError", "Run", "new_run_id",
           "check_run_id", "code_identity", "start_run", "content_digest",
           "write_run_record", "read_run_record"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: Where PR 42 records live, relative to bank_pipeline/.
RECORD_ROOT = BANK_ROOT / "research_logs" / "pr42"

#: ``c<commit>_<YYYYMMDD_HHMMSS>`` (PR 39 convention).
RUN_ID = re.compile(r"^c[0-9]+_[0-9]{8}_[0-9]{6}$")

#: Files whose LF-normalised SHA-256 every record carries: the notebook, the PR 42
#: research modules and the production feature code the inventory traces.
CODE_FILES = (
    "feature_research/feature_research.ipynb",
    "feature_research/research_records.py",
    "feature_research/training_side.py",
    "feature_research/feature_inventory.py",
    "feature_research/frozen_references.py",
    "feature_research/config.py",
    "feature_research/feature_engineering/__init__.py",
    "feature_research/feature_engineering/_checks.py",
    "feature_research/feature_engineering/crisis.py",
    "feature_research/feature_engineering/integrals.py",
    "feature_research/feature_engineering/derivatives.py",
    "feature_research/feature_engineering/temporal.py",
    "feature_research/feature_engineering/prior.py",
    "feature_research/feature_engineering/overlap.py",
    "feature_research/feature_engineering/pipeline.py",
    "shared/stage1/feature_engineering.py",
    "shared/stage1/crossfit.py",
    "shared/stage2/feature_engineering.py",
    "shared/stage2/binning.py",
    "shared/stage3/feature_engineering.py",
    "shared/stage3/data.py",
    "shared/stage3/block_cache.py",
    "shared/stage2/diagnostics/boundary.py",
    "glass_pipeline/ebm/interactions.py",
    "blackbox_pipeline/models/mlp/features.py",
    "blackbox_pipeline/models/meta_xgb/config.py",
    "stage2_crossfit/inner_features.py",
)

#: Top-level record keys that identify the run rather than its content.
VOLATILE_KEYS = ("run", "seeds", "code", "environment", "content_sha256", "format")

RECORD_FORMAT = "pr42.research_record/1"


class RecordError(RuntimeError):
    """A record cannot be written or read as required; the reason is stated."""


# ----------------------------------------------------------------------
# run identity
# ----------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(commit: int) -> str:
    """``c<commit>_<local YYYYMMDD_HHMMSS>``, as every PR 39 runner names its runs."""
    if not isinstance(commit, int) or commit < 1:
        raise RecordError(f"commit must be a positive int, got {commit!r}")
    return f"c{commit}_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise RecordError(f"run id {run_id!r} is not c<commit>_<YYYYMMDD_HHMMSS>")
    return run_id


def _git(*args: str, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def code_identity(root: Path = BANK_ROOT) -> Dict[str, Any]:
    """
    HEAD, the uncommitted-changes flag, ``CODE_FILES`` digests (LF-normalised;
    ``None`` for a file that does not exist) and the whole-tree digest.
    """
    from upstream_regeneration.records import code_identity as tree_identity
    from upstream_regeneration.records import file_sha256_lf

    root = Path(root)
    files = {rel: (file_sha256_lf(root / rel) if (root / rel).is_file() else None)
             for rel in CODE_FILES}
    head = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    tree = tree_identity(root)
    return {"head": None if head is None else head.strip(),
            "uncommitted_tracked_changes": None if status is None else bool(status.strip()),
            "files_sha256_lf": files,
            "tree": {"code_sha256": tree["code_sha256"], "files": tree["files"],
                     "definition": "upstream_regeneration.records.code_identity"}}


@dataclass
class Run:
    """One run: its id, seeds and identity, fixed when the run starts."""

    run_id: str
    commit: int
    notebook: str
    seeds: Dict[str, int]
    started_utc: str
    code: Dict[str, Any]
    environment: Dict[str, Any]
    root: Path = field(default=RECORD_ROOT)

    @property
    def folder(self) -> Path:
        return Path(self.root) / self.run_id

    def describe(self) -> str:
        head = (self.code.get("head") or "unknown")[:12]
        dirty = self.code.get("uncommitted_tracked_changes")
        return (f"run {self.run_id}  (HEAD {head}"
                f"{', uncommitted tracked changes' if dirty else ''})\n"
                f"seeds {self.seeds}\nrecords -> {self.folder}")


def start_run(*, commit: int, seeds: Mapping[str, int],
              notebook: str = "feature_research/feature_research.ipynb",
              root: Path = RECORD_ROOT, run_id: Optional[str] = None) -> Run:
    """
    A new run. Refused when its folder already exists (a run id is used once).
    Nothing is written until ``write_run_record``.
    """
    from upstream_regeneration.records import environment

    run_id = check_run_id(run_id or new_run_id(commit))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in seeds.values()):
        raise RecordError(f"seeds must be ints, got {dict(seeds)!r}")
    folder = Path(root) / run_id
    if folder.exists():
        raise RecordError(f"{folder} exists: a run id is used once (start a new run)")
    return Run(run_id=run_id, commit=commit, notebook=notebook, seeds=dict(seeds),
               started_utc=_now(), code=code_identity(), environment=environment(),
               root=Path(root))


# ----------------------------------------------------------------------
# writing and reading
# ----------------------------------------------------------------------
def _canonical(obj: Any) -> Any:
    from upstream_regeneration.records import canonical
    return canonical(obj)


def _text(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(_canonical(record), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def content_digest(sections: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical content sections (``VOLATILE_KEYS`` excluded)."""
    content = {k: v for k, v in _canonical(dict(sections)).items() if k not in VOLATILE_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def write_run_record(run: Run, name: str, sections: Mapping[str, Any]) -> Path:
    """
    Write ``<root>/<run_id>/<name>.json`` and its ``.sha256`` sidecar, once.

    The run folder is created here and must not exist; both files are opened
    exclusively. ``sections`` may not use the identity keys (``VOLATILE_KEYS``).
    """
    if not re.fullmatch(r"[a-z0-9_]+", str(name)):
        raise RecordError(f"record name {name!r} must be lower-case letters, digits and _")
    clash = sorted(set(sections) & set(VOLATILE_KEYS))
    if clash:
        raise RecordError(f"sections may not set the identity keys {clash}")
    record = {
        "format": RECORD_FORMAT,
        "run": {"run_id": run.run_id, "commit": run.commit, "notebook": run.notebook,
                "started_utc": run.started_utc, "finished_utc": _now()},
        "seeds": run.seeds,
        "code": run.code,
        "environment": run.environment,
        **dict(sections),
    }
    record["content_sha256"] = content_digest(sections)
    data = _text(record)
    folder = run.folder
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RecordError(f"{folder} exists: records are written once (a new attempt gets "
                          "a new run id)") from None
    path = folder / f"{name}.json"
    with open(path, "xb") as fh:
        fh.write(data)
    with open(path.with_name(path.name + ".sha256"), "x", encoding="ascii", newline="\n") as fh:
        fh.write(hashlib.sha256(data).hexdigest() + "\n")
    return path


def read_run_record(run_id: str, name: str, *, root: Path = RECORD_ROOT) -> Dict[str, Any]:
    """A record, after checking its sidecar and its content digest."""
    path = Path(root) / check_run_id(run_id) / f"{name}.json"
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.is_file() or side.read_text(encoding="ascii").strip() != \
            hashlib.sha256(data).hexdigest():
        raise RecordError(f"{path} does not match its .sha256 sidecar")
    record = json.loads(data.decode("utf-8"))
    if record.get("format") != RECORD_FORMAT:
        raise RecordError(f"{path} is not a {RECORD_FORMAT} record")
    sections = {k: v for k, v in record.items() if k not in VOLATILE_KEYS}
    if content_digest(sections) != record.get("content_sha256"):
        raise RecordError(f"{path}: content_sha256 does not match its content")
    return record
