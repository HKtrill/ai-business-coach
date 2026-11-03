"""
Shared loader for the PR 37 real-artifact router diagnostics tests.

Moved verbatim from tests/realdata/test_router_diagnostics_artifacts.py
(Commit 1, Step 1C) in Commit 2, Step 2A, so the Step 2A trace tests load the
October 23 artifacts exactly the same way. Behaviour is unchanged:

* ``Real`` loads each artifact once per instance and re-raises a failure in
  every test that needs it;
* the train reference reads X_train.index, X_test.index and y_train only
  (never a test label), with the stage_io split fingerprint ATTESTED from the
  baseline snapshot's PROVENANCE.json, not recomputed;
* ``PR37_BASELINE_SHA256SUMS`` (the snapshot's SHA256SUMS.txt) makes each
  file's SHA-256 an attested match against the preserved baseline.

Nothing here fits, writes or regenerates anything.

Commit 2, Step 2E adds ``PinnedReal`` (``Real`` with the two arbiter roots fixed
by a frozen baseline's MANIFEST.json instead of "newest file") and
``collect_baseline_sources``, which hashes every file before its first load and
assembles ``shared.stage2.diagnostics.baseline.BaselineSources`` from the same
validated views.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import joblib

from shared.stage2 import diagnostics as d
from tests.support import artifact_audit as audit
from tests.support.notebooks import BANK_ROOT

__all__ = ["ROUTER_SPLIT_FP", "TRAIN_LABEL_FP", "STAGE_IO_SPLIT_FP", "ATTESTATION", "SUMS_ENV",
           "Real", "PinnedReal", "read_sums", "collect_baseline_sources",
           "RAW_DATA_FILE"]

#: Recorded identities of the October 23 split (baseline PROVENANCE.json).
ROUTER_SPLIT_FP = "fc3e73a5ef91aea4"
TRAIN_LABEL_FP = "b984d2a50527"
STAGE_IO_SPLIT_FP = os.environ.get("PR37_STAGE_IO_SPLIT_FINGERPRINT", "d6cddadc3439")
ATTESTATION = ("PR 37 baseline snapshot PROVENANCE.json, identities.stage_io_split_fingerprint "
               "(recorded 2026-11-01; not recomputed)")
SUMS_ENV = "PR37_BASELINE_SHA256SUMS"


def read_sums(path) -> dict:
    """``{"bank_pipeline/<relative path>": sha256}`` from a snapshot SHA256SUMS file."""
    out = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.strip():
            sha, rel = line.split("  ", 1)
            out[rel.strip()] = sha.strip()
    return out


def _sums() -> dict:
    value = os.environ.get(SUMS_ENV)
    return read_sums(value) if value else {}


class Real:
    """Each artifact once per module; a failure is re-raised by every test that needs it."""

    def __init__(self):
        self._done: dict = {}
        self.sums = _sums()

    def _once(self, name, step):
        if name not in self._done:
            try:
                self._done[name] = (True, step())
            except Exception as exc:          # noqa: BLE001 - re-raised below
                self._done[name] = (False, exc)
        ok, value = self._done[name]
        if not ok:
            raise value
        return value

    def sha(self, path: Path):
        if not self.sums:
            return None
        rel = "bank_pipeline/" + Path(path).resolve().relative_to(Path(BANK_ROOT).resolve()).as_posix()
        if rel not in self.sums:
            raise AssertionError(f"{rel} is not in {os.environ[SUMS_ENV]}")
        return self.sums[rel]

    def ref(self):
        return self._once("ref", lambda: d.train_reference_from_global_split(
            audit.real_global_split(), attested_stage_io_split_fingerprint=STAGE_IO_SPLIT_FP,
            attestation_source=ATTESTATION))

    def roots(self):
        return self._once("roots", lambda: audit.select_roots(BANK_ROOT, audit.manifest_from_env()))

    def paths(self, key):
        return self._once(("paths", key), lambda: audit.upstream_paths(
            key, joblib.load(self.roots()[key])))

    def view(self, which):
        def build():
            if which == "glass":
                p = self.paths("meta_ebm")["stage2"]
                return d.extract_router_view(p, self.ref(), expected_arm="glass",
                                             expected_sha256=self.sha(p))
            if which == "rf":
                p = self.paths("meta_ebm_on_blackbox")["stage2"]
                return d.extract_router_view(p, self.ref(), expected_arm="blackbox",
                                             expected_sha256=self.sha(p))
            if which == "lr":
                p = self.paths("meta_ebm")["stage1"]
                return d.extract_lr_view(p, self.ref(), expected_sha256=self.sha(p))
            p = self.paths("meta_ebm")["stage3"]
            return d.extract_ebm_view(p, self.ref(), expected_sha256=self.sha(p))
        return self._once(("view", which), build)

    def stage4(self):
        p = self.roots()["meta_ebm"]
        return self._once("stage4", lambda: d.extract_stage4_recall_thresholds(
            p, self.ref(), lr_view=self.view("lr"), ebm_view=self.view("ebm"),
            expected_sha256=self.sha(p)))

    def rf_result_path(self) -> Path:
        origin = self.view("rf").metadata.get("rf_params_origin")
        candidates = [Path(str(origin))] if origin else []
        candidates += [Path(BANK_ROOT) / c for c in candidates if not c.is_absolute()]
        found = [c for c in candidates if c.is_file()]
        if not found:
            raise AssertionError(f"RF Router rf_params_origin {origin!r} does not name an "
                                 "existing GLASS-arm RF result")
        return found[-1]


# ----------------------------------------------------------------------
# Commit 2, Step 2E: the Pre-Correction Stage 2 Baseline sources
# ----------------------------------------------------------------------
#: The raw dataset file recorded (LF-normalised SHA-256) for reference.
RAW_DATA_FILE = "data/raw/bank-additional-full.csv"


class PinnedReal(Real):
    """``Real`` with the arbiter roots fixed (a frozen baseline's record), never "newest"."""

    def __init__(self, roots: Mapping[str, Path]):
        super().__init__()
        self._pinned = {k: Path(v) for k, v in roots.items()}

    def roots(self):
        return dict(self._pinned)


def collect_baseline_sources(real: Real, *, bank_root=BANK_ROOT, sums_path=None):
    """
    ``baseline.BaselineSources`` from ``real``'s validated views. Every file is
    hashed before its first load; paths are recorded relative to bank_pipeline/.
    With ``sums_path`` (and ``real.sums`` read from it) each source's snapshot
    attestation and the snapshot's own identity are recorded; a source whose
    hash differs from its snapshot entry is refused.
    """
    from shared.stage2.diagnostics import baseline as b

    root = Path(bank_root).resolve()
    files, digests = {}, {}

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root).as_posix()
        except ValueError:
            raise AssertionError(f"{p} is not under {root}: baseline sources must live in "
                                 "bank_pipeline/") from None

    def remember(role: str, p) -> Path:
        p = Path(p)
        files[role], digests[role] = p, d.file_sha256(p)
        return p

    roots = real.roots()
    remember(b.ROLE_ARBITER_GLASS, roots["meta_ebm"])
    remember(b.ROLE_ARBITER_RF, roots["meta_ebm_on_blackbox"])
    native = real.paths("meta_ebm")
    remember(b.ROLE_LR, native["stage1"])
    remember(b.ROLE_GLASS, native["stage2"])
    remember(b.ROLE_EBM, native["stage3"])
    remember(b.ROLE_RF, real.paths("meta_ebm_on_blackbox")["stage2"])
    ref = real.ref()
    glass, rf, lr, ebm = (real.view(w) for w in ("glass", "rf", "lr", "ebm"))
    rf_result = remember(b.ROLE_RF_RESULT, real.rf_result_path())
    candidate = d.extract_rf_pass1_candidate(rf_result, ref, expected_sha256=real.sha(rf_result))

    def attested(role: str) -> Optional[bool]:
        if not real.sums:
            return None
        key = "bank_pipeline/" + rel(files[role])
        if key not in real.sums:
            return None
        if real.sums[key] != digests[role]:
            raise AssertionError(f"{role}: {key} differs from the snapshot SHA256SUMS entry")
        return True

    artifacts = tuple(b.SourceArtifact(role, rel(files[role]), digests[role],
                                       files[role].stat().st_size, attested(role))
                      for role in b.SOURCE_ROLES)
    snapshot = None
    if sums_path is not None:
        sp = Path(sums_path)
        snapshot = b.SnapshotRecord(f"{sp.parent.name}/{sp.name}", d.file_sha256(sp),
                                    len(read_sums(sp)))
    raw = root / RAW_DATA_FILE
    data_files = ((b.DataFileRecord(RAW_DATA_FILE, b.lf_sha256(raw.read_bytes()),
                                    "raw dataset (UCI bank-additional-full)"),)
                  if raw.is_file() else ())
    return b.BaselineSources(ref=ref, glass=glass, rf=rf, lr=lr, ebm=ebm, rf_candidate=candidate,
                             artifacts=artifacts, snapshot=snapshot, data_files=data_files)
