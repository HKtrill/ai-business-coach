"""
stage2_crossfit.phase0
======================
Phase 0 of the PR 38 handoff (section 3), as the post-handoff PR 37 commits
left it: ``2fc030e`` regenerated both notebooks' artifacts in the live
``models/`` folders, and ``e3e27f6`` made fixed-name artifacts immutable
versions plus a mutable alias and moved every historical check to the
preserved archive. So the October 23 references are the archive's files, and a
live file that differs from the archive (for example the RF Router alias) is
recorded, not treated as damage.

Every check here is read-only, except ``prepare_output_root``, which creates
one new, empty directory for a PR 38 run and refuses an existing one.

1. ``verify_snapshot``     every SHA256SUMS entry: the archived file exists and
                           matches; the SUMS file is the one the baseline names.
2. ``run_baseline_checks`` ``stage2_baseline.py --verify`` and
                           ``stage2_baseline_archive.py --check`` (unchanged CLIs).
3. ``check_artifact_manifest``  ``$BANK_ARTIFACT_MANIFEST`` names exactly the four
                           Stage 4 arbiters, each attested by SHA256SUMS, the two
                           baseline arbiters among them.
4. ``inventory``           the baseline's source artifacts: recorded, archived and
                           live identity (the baseline MANIFEST.json holds the rest).
5. ``prepare_output_root`` ``models/pr38/<run_id>/``: Git-ignored, outside every
                           folder a notebook or audit reads the newest file from.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from .runner import sha256_file

__all__ = ["BANK_ROOT", "BASELINE_DIR", "OUTPUT_ROOT", "RECORD_ROOT", "ARBITER_KEYS",
           "NEWEST_FILE_FOLDERS", "read_sums", "verify_snapshot", "run_baseline_checks",
           "check_artifact_manifest", "inventory", "prepare_output_root", "run_phase0"]

BANK_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = BANK_ROOT / "research_logs" / "baselines" / "pr37_stage2_pre_correction_v1"
#: PR 38 model artifacts (Git-ignored: ``**/models/**``).
OUTPUT_ROOT = BANK_ROOT / "models" / "pr38"
#: PR 38 records (version-controlled).
RECORD_ROOT = BANK_ROOT / "research_logs" / "pr38"
_PREFIX = "bank_pipeline/"
#: ``tests.support.artifact_audit.ROOTS`` (the four arbiters a manifest pins).
ARBITER_KEYS = ("meta_ebm", "meta_xgb", "meta_ebm_on_blackbox", "meta_xgb_on_glass")
_BASELINE_ARBITERS = {"stage4_arbiter.meta_ebm": "meta_ebm",
                      "stage4_arbiter.meta_ebm_on_blackbox": "meta_ebm_on_blackbox"}
#: Folders whose newest file a notebook, the Stage 4 audit or a loader reads
#: (handoff section 11, table A and "Newest-file readers"). A PR 38 run never
#: writes into any of them.
NEWEST_FILE_FOLDERS = (
    "models/glass_router", "models/rf_router", "models/rf", "models/lr", "models/ebm",
    "models/meta_ebm", "models/stage3", "models/mlp",
    "blackbox_pipeline/models/meta_xgb", "blackbox_pipeline/models/xgb",
    "blackbox_pipeline/models/stage4_crossover",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_sums(path) -> Dict[str, str]:
    """``{relative path: sha256}`` from a ``sha256sum``-style file."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            sha, rel = line.split("  ", 1)
            out[rel.strip().lstrip("*")] = sha.strip().lower()
    return out


def _baseline_manifest() -> dict:
    return json.loads((BASELINE_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


def verify_snapshot(sums_path) -> Dict[str, Any]:
    """Every archived file against SHA256SUMS; the SUMS file against the baseline's record."""
    sums_path = Path(sums_path)
    snap = _baseline_manifest()["snapshot"]
    sums = read_sums(sums_path)
    archive = sums_path.parent
    bad, live = [], {"same": 0, "different": [], "missing": []}
    for rel, sha in sorted(sums.items()):
        p = archive / rel
        if not p.is_file():
            bad.append(f"missing from the archive: {rel}")
            continue
        if sha256_file(p) != sha:
            bad.append(f"archive hash differs: {rel}")
        if rel.startswith(_PREFIX):
            q = BANK_ROOT / rel[len(_PREFIX):]
            if not q.is_file():
                live["missing"].append(rel)
            elif sha256_file(q) == sha:
                live["same"] += 1
            else:
                live["different"].append(rel)
    sums_sha = sha256_file(sums_path)
    checks = {
        "entries": {"got": len(sums), "baseline": snap["entries"],
                    "passed": len(sums) == snap["entries"]},
        "sums_file_sha256": {"got": sums_sha, "baseline": snap["sha256"],
                             "passed": sums_sha == snap["sha256"]},
        "archived_files": {"problems": bad, "passed": not bad},
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "sums_path": str(sums_path),
            "live_counterparts": {**live, "note": (
                "informational: since 2fc030e/e3e27f6 the live models/ folders hold regenerated "
                "files and mutable aliases; the archive is the historical reference")}}


def _run(args, env) -> Dict[str, Any]:
    proc = subprocess.run([sys.executable, *args], cwd=BANK_ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    return {"command": " ".join(["python", *args]), "exit_code": proc.returncode,
            "passed": proc.returncode == 0, "output_tail": tail}


def run_baseline_checks(sums_path) -> Dict[str, Any]:
    """The unchanged baseline CLIs: ``--verify`` (live package) and ``--check`` (archive)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PR37_BASELINE_SHA256SUMS": str(sums_path)}
    base = "research_logs/baselines"
    verify = _run([f"{base}/stage2_baseline.py", "--verify"], env)
    check = _run([f"{base}/stage2_baseline_archive.py", "--check", "--sums", str(sums_path)], env)
    return {"passed": verify["passed"] and check["passed"], "verify": verify, "archive_check": check}


def _relative(path) -> Optional[str]:
    p = Path(path)
    p = p if p.is_absolute() else BANK_ROOT / p
    try:
        return p.resolve().relative_to(BANK_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def check_artifact_manifest(manifest_path, sums_path) -> Dict[str, Any]:
    """``$BANK_ARTIFACT_MANIFEST``: the four arbiters, each attested; the baseline's two included."""
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    sums = read_sums(sums_path)
    problems = []
    if not isinstance(entries, dict) or sorted(entries) != sorted(ARBITER_KEYS):
        return {"passed": False, "problems": [f"needs exactly {list(ARBITER_KEYS)}"],
                "manifest": str(manifest_path)}
    resolved = {}
    for key, value in entries.items():
        rel = _relative(value)
        p = BANK_ROOT / rel if rel is not None else Path(value)
        if rel is None:
            problems.append(f"{key}: {value} is outside bank_pipeline/")
            continue
        if not p.is_file():
            problems.append(f"{key}: {p} does not exist")
            continue
        want = sums.get(_PREFIX + rel)
        got = sha256_file(p)
        if want is None:
            problems.append(f"{key}: {rel} is not attested by SHA256SUMS")
        elif got != want:
            problems.append(f"{key}: {rel} differs from its SHA256SUMS entry")
        resolved[key] = {"path": rel, "sha256": got, "attested": got == want}
    for src in _baseline_manifest()["sources"]:
        key = _BASELINE_ARBITERS.get(src["role"])
        if key and resolved.get(key, {}).get("path") != src["path"]:
            problems.append(f"{key}: the baseline records {src['path']}")
    return {"passed": not problems, "problems": problems, "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path), "arbiters": resolved,
            "env": "set BANK_ARTIFACT_MANIFEST to this file for every real-data check"}


def inventory(sums_path) -> Dict[str, Any]:
    """The baseline's source artifacts: recorded, archived and live identity."""
    sums = read_sums(sums_path)
    archive = Path(sums_path).parent
    man = _baseline_manifest()
    rows, problems = [], []
    for src in man["sources"]:
        rel = src["path"]
        if not PurePosixPath(rel).parts or rel.startswith("/"):
            problems.append(f"{src['role']}: unexpected path {rel}")
            continue
        key = _PREFIX + rel
        a = archive / key
        live = BANK_ROOT / rel
        row = {"role": src["role"], "path": rel, "recorded_sha256": src.get("sha256"),
               "bytes": src.get("bytes"), "sums_sha256": sums.get(key),
               "archived": a.is_file(),
               "archive_matches": a.is_file() and sha256_file(a) == src.get("sha256"),
               "live_present": live.is_file(),
               "live_matches": live.is_file() and sha256_file(live) == src.get("sha256")}
        if src.get("snapshot_attested") and not (row["sums_sha256"] == src.get("sha256")
                                                 and row["archive_matches"]):
            problems.append(f"{src['role']}: the archive does not hold the recorded file")
        rows.append(row)
    return {"passed": not problems, "problems": problems, "sources": rows,
            "cites": {"baseline_manifest_sha256": sha256_file(BASELINE_DIR / "MANIFEST.json"),
                      "identities": man["identities"], "routers": sorted(man["routers"]),
                      "note": "configuration identities (GLASS config_full; RF rf_params, "
                              "constraint source, eligibility), split, label and fold "
                              "fingerprints are recorded in the cited MANIFEST.json"}}


def prepare_output_root(run_id: str, *, root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    """Create ``<root>/<run_id>/`` once; refuse an existing one or a newest-file folder."""
    if not run_id or any(c in run_id for c in r'\/:*?"<>| ') or run_id.startswith("."):
        raise ValueError(f"run_id {run_id!r} must be a plain name")
    out = Path(root) / run_id
    rel = _relative(out)
    for folder in NEWEST_FILE_FOLDERS:
        if rel is not None and (rel == folder or rel.startswith(folder + "/")):
            raise ValueError(f"{rel} is inside {folder}, which a newest-file reader scans")
    out.mkdir(parents=True, exist_ok=False)
    marker = {"format": "pr38.stage2.output_root/1", "run_id": run_id, "created_utc": _now(),
              "note": "PR 38 outputs only; never historical evidence; never read as newest"}
    (out / "PR38_OUTPUT_ROOT.json").write_text(json.dumps(marker, indent=2) + "\n",
                                               encoding="utf-8")
    return {"path": str(out), "relative": rel, "created": True}


def run_phase0(sums_path, manifest_path, run_id: str, *, root: Path = OUTPUT_ROOT,
               baseline_checks: bool = True) -> Dict[str, Any]:
    """Phase 0 in order; the output root is created only after every check passed."""
    rec: Dict[str, Any] = {"format": "pr38.stage2.phase0/1", "run_id": run_id, "at_utc": _now()}
    rec["snapshot"] = verify_snapshot(sums_path)
    rec["baseline"] = run_baseline_checks(sums_path) if baseline_checks else {"passed": False,
                                                                             "skipped": True}
    rec["artifact_manifest"] = check_artifact_manifest(manifest_path, sums_path)
    rec["inventory"] = inventory(sums_path)
    checks_ok = all(rec[k]["passed"] for k in ("snapshot", "baseline", "artifact_manifest",
                                               "inventory"))
    rec["output_root"] = prepare_output_root(run_id, root=root) if checks_ok else None
    rec["passed"] = checks_ok
    return rec
