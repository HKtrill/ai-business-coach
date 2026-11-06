"""
Where the real-data tests read model artifacts from (PR 37, Commit 5 follow-up).

Two sources, never mixed:

* ``HistoricalReal`` - the October 23 regression tests. Every file is read from
  the preserved archive, the read-only snapshot whose SHA256SUMS.txt
  ``$PR37_BASELINE_SHA256SUMS`` names (files at ``<snapshot>/bank_pipeline/<path>``),
  never from the live ``models/`` folders. Regenerating models in the working
  checkout therefore cannot change what these tests read. The two Stage 4
  roots are the ones the frozen baseline MANIFEST.json records; every Stage 1-3
  file is the one those arbiters recorded consuming, mapped to its archived
  copy; and every archived file must match its SHA256SUMS entry before its
  first load, so a corrupt or substituted archive file fails.
* ``CurrentReal`` - the current-artifact tests: the newest artifacts in the live
  folders (or the arbiters ``$BANK_ARTIFACT_MANIFEST`` names). They are never
  compared with the October hashes; they must pass the structural, provenance
  and contract checks instead.

Both subclass ``tests.support.router_artifacts.Real``. That module, the artifact
audit and the baseline CLI are among the 15 files the frozen pre-correction
baseline pins; they are not modified. Only file location is overridden here:
what is loaded, and every check run on it, is the October 23 implementation,
which ``stage2_baseline.py --check`` (through ``stage2_baseline_archive.py``)
still reports as ``identical``. Nothing here writes, fits or regenerates
anything.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Optional

import joblib

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from tests.support import artifact_audit as audit
from tests.support.notebooks import BANK_ROOT
from tests.support.router_artifacts import (
    RAW_DATA_FILE,
    SUMS_ENV,
    Real,
    collect_baseline_sources,
    read_sums,
)

__all__ = ["BASELINE_MANIFEST", "RAW_DATA_DESCRIPTION", "Archive", "HistoricalReal",
           "CurrentReal", "baseline_arbiters", "collect_archived_baseline_sources",
           "archived_rehash"]

#: The frozen pre-correction baseline record; its ``sources`` name the two arbiters.
BASELINE_MANIFEST = (Path(BANK_ROOT) / b.BASELINES_DIR / b.BASELINE_ID / b.MANIFEST)
#: ``collect_baseline_sources`` records the raw dataset with this description.
RAW_DATA_DESCRIPTION = "raw dataset (UCI bank-additional-full)"
_ARBITER_ROLES = {"meta_ebm": b.ROLE_ARBITER_GLASS, "meta_ebm_on_blackbox": b.ROLE_ARBITER_RF}
_PREFIX = "bank_pipeline/"


class Archive:
    """The preserved snapshot: ``<dir>/SHA256SUMS.txt`` and ``<dir>/bank_pipeline/...``."""

    def __init__(self, sums_path=None):
        value = sums_path or os.environ.get(SUMS_ENV)
        if not value:
            raise AssertionError(
                f"set {SUMS_ENV} to the preserved October 23 snapshot's SHA256SUMS.txt: the "
                "historical tests read only the archived artifacts, never the live models/ "
                "folders")
        self.sums_path = Path(value)
        if not self.sums_path.is_file():
            raise AssertionError(f"{SUMS_ENV}: SHA256SUMS file not found: {self.sums_path}")
        self.root = self.sums_path.parent / "bank_pipeline"
        if not self.root.is_dir():
            raise AssertionError(f"the archive has no bank_pipeline/ folder: {self.root}")
        self.sums = read_sums(self.sums_path)
        self._verified: dict = {}

    def relative(self, recorded) -> str:
        """
        ``recorded`` (as an artifact stored it) as a path relative to bank_pipeline/.

        Relative paths (``./models/lr\\x.joblib``) are taken as they are. An
        absolute path may point into the archive, into this checkout's
        bank_pipeline/, or into the ``prototype/bank_pipeline/`` of the checkout
        the notebook ran in (another folder on the same machine); the part after
        that folder is the archived file's path. The file must then still be
        listed in SHA256SUMS and match its hash (``path``), so the mapping only
        names a file and cannot admit one that differs.
        """
        text = str(recorded).strip().replace("\\", "/")
        if not text:
            raise AssertionError("an empty artifact path was recorded")
        if text.startswith("/") or PureWindowsPath(str(recorded)).is_absolute():
            p = Path(str(recorded))
            for base in (self.root, Path(BANK_ROOT)) if p.is_absolute() else ():
                try:
                    return p.resolve().relative_to(base.resolve()).as_posix()
                except ValueError:
                    continue
            parts = [x for x in text.split("/") if x not in (".", "")]
            marks = [i for i in range(1, len(parts))
                     if parts[i - 1] == "prototype" and parts[i] == "bank_pipeline"]
            rest = parts[marks[-1] + 1:] if marks else []
            if not rest or ".." in rest:
                raise AssertionError(f"{recorded} is outside bank_pipeline/ and the archive: "
                                     "the historical tests cannot map it to an archived file")
            return "/".join(rest)
        parts = [x for x in PurePosixPath(text).parts if x not in (".", "")]
        if not parts or ".." in parts:
            raise AssertionError(f"recorded path {recorded!r} cannot be mapped into the archive")
        return "/".join(parts)

    def path(self, recorded) -> Path:
        """The archived copy of ``recorded``, its SHA-256 checked against SHA256SUMS once."""
        rel = self.relative(recorded)
        key = _PREFIX + rel
        if key not in self._verified:
            if key not in self.sums:
                raise AssertionError(f"{key} is not in the archive's {self.sums_path.name} "
                                     f"({self.sums_path})")
            p = self.root / rel
            if not p.is_file():
                raise AssertionError(f"{key} is listed in {self.sums_path.name} but missing "
                                     f"from the archive: {p}")
            digest = d.file_sha256(p)
            if digest != self.sums[key]:
                raise AssertionError(f"archived {key} has SHA-256 {digest}, not the recorded "
                                     f"{self.sums[key]}: the archive is corrupt or was changed")
            self._verified[key] = p
        return self._verified[key]

    def key(self, path) -> str:
        """``bank_pipeline/<relative path>`` of a file inside the archive."""
        try:
            rel = Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            raise AssertionError(f"{path} is not an archived file (archive: {self.root}); the "
                                 "historical tests read nothing else") from None
        return _PREFIX + rel


def baseline_arbiters(manifest=BASELINE_MANIFEST) -> dict:
    """``{root key: recorded path}`` of the two arbiters a baseline MANIFEST.json names."""
    frozen = manifest if isinstance(manifest, Mapping) else json.loads(
        Path(manifest).read_text(encoding="utf-8"))
    paths = {s["role"]: s["path"] for s in frozen.get("sources") or ()}
    missing = [role for role in _ARBITER_ROLES.values() if role not in paths]
    if missing:
        raise AssertionError(f"the baseline MANIFEST.json records no source for {missing}")
    return {key: paths[role] for key, role in _ARBITER_ROLES.items()}


class HistoricalReal(Real):
    """``Real`` reading only the archived October 23 artifacts (module docstring)."""

    def __init__(self, *, sums_path=None, arbiters: Optional[Mapping[str, str]] = None):
        super().__init__()
        self.archive = Archive(sums_path)
        self.sums = self.archive.sums
        self._arbiters = dict(arbiters if arbiters is not None else baseline_arbiters())

    def roots(self):
        return self._once("roots", lambda: {k: self.archive.path(v)
                                            for k, v in self._arbiters.items()})

    def paths(self, key):
        def build():
            art = joblib.load(self.roots()[key])
            stages = audit.upstream_record(key, art).get("stages") or {}
            out, missing = {}, []
            for stage in audit.STAGES:
                recorded = (stages.get(stage) or {}).get("artifact_path")
                if recorded:
                    out[stage] = self.archive.path(recorded)
                else:
                    missing.append(stage)
            if missing:
                raise audit.AuditError(f"{key}: no artifact_path recorded for {missing}")
            return out
        return self._once(("paths", key), build)

    def sha(self, path: Path):
        return self.sums[self.archive.key(path)]

    def rf_result_path(self) -> Path:
        origin = self.view("rf").metadata.get("rf_params_origin")
        if not origin:
            raise AssertionError("the archived RF Router records no rf_params_origin")
        return self.archive.path(origin)


class CurrentReal(Real):
    """``Real`` over the live, newest artifacts; never checked against October hashes."""

    def __init__(self):
        super().__init__()
        self.sums = {}


def collect_archived_baseline_sources(real: HistoricalReal) -> b.BaselineSources:
    """
    ``collect_baseline_sources`` over the archive. Source paths are recorded
    relative to the archive's bank_pipeline/ (the same relative paths as in the
    live tree); the raw dataset, which the archive does not hold, is the
    repository's own Git-tracked file.
    """
    sources = collect_baseline_sources(real, bank_root=real.archive.root,
                                       sums_path=real.archive.sums_path)
    raw = Path(BANK_ROOT) / RAW_DATA_FILE
    data_files = ((b.DataFileRecord(RAW_DATA_FILE, b.lf_sha256(raw.read_bytes()),
                                    RAW_DATA_DESCRIPTION),) if raw.is_file() else ())
    return dataclasses.replace(sources, data_files=data_files)


def archived_rehash(archive: Archive):
    """``stage2_baseline`` rehash: problems if an archived source changed or vanished."""
    def rehash(sources) -> list:
        out = []
        for a in sources.artifacts:
            p = archive.root / a.path
            if not p.is_file():
                out.append(f"{a.role}: {a.path} is gone from the archive")
            elif d.file_sha256(p) != a.sha256:
                out.append(f"{a.role}: {a.path} changed in the archive during the run")
        return out
    return rehash
