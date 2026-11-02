"""
PR 37 Commit 1 (Step 1C): the Stage 2 diagnostics boundary on the SAVED
artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_router_diagnostics_artifacts.py

Only builds the train-side views and resolves the registered operating
points: every mandatory boundary check must pass on the October 23 artifacts.
No metric is computed, nothing is fitted, written or regenerated, and no test
label is read (the train reference reads X_train.index, X_test.index and
y_train only).

Files: the GLASS LR / GLASS Router / GLASS EBM recorded by the native
Meta-EBM, the RF Router recorded by the black-box-upstream Meta-EBM, and the
GLASS-arm RF result named by the RF Router's ``rf_params_origin``. Optional:
``PR37_BASELINE_SHA256SUMS`` (the baseline snapshot's SHA256SUMS.txt) makes
each file's SHA-256 an attested match against the preserved baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pytest

from shared.stage2 import diagnostics as d
from tests.support import artifact_audit as audit
from tests.support.notebooks import BANK_ROOT

pytestmark = pytest.mark.realdata

#: Recorded identities of the October 23 split (baseline PROVENANCE.json).
ROUTER_SPLIT_FP = "fc3e73a5ef91aea4"
TRAIN_LABEL_FP = "b984d2a50527"
STAGE_IO_SPLIT_FP = os.environ.get("PR37_STAGE_IO_SPLIT_FINGERPRINT", "d6cddadc3439")
ATTESTATION = ("PR 37 baseline snapshot PROVENANCE.json, identities.stage_io_split_fingerprint "
               "(recorded 2026-11-01; not recomputed)")
SUMS_ENV = "PR37_BASELINE_SHA256SUMS"


def _sums() -> dict:
    value = os.environ.get(SUMS_ENV)
    if not value:
        return {}
    out = {}
    for line in Path(value).read_text(encoding="ascii").splitlines():
        if line.strip():
            sha, rel = line.split("  ", 1)
            out[rel.strip()] = sha.strip()
    return out


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


@pytest.fixture(scope="module")
def real():
    return Real()


def _status(view) -> dict:
    frame = view.provenance_frame()
    return dict(zip(frame["name"], frame["status"]))


def test_train_reference_reproduces_the_recorded_identities(real):
    ref = real.ref()
    assert ref.router_split_fingerprint == ROUTER_SPLIT_FP
    assert ref.train_label_fingerprint == TRAIN_LABEL_FP
    assert ref.stage_io_split_fingerprint == STAGE_IO_SPLIT_FP


@pytest.mark.parametrize("which", ["glass", "rf"])
def test_router_view_passes_every_mandatory_check(real, which):
    v = real.view(which)
    status = _status(v)
    for item in ("router_transform", "confidence", "decision_tokens", "fold_plan",
                 "train_label_fingerprint.cross_fit", "train_label_fingerprint.full_train",
                 "pass2_population"):
        assert status[item] == d.VALIDATED, item
    assert v.pass2_population["normalised"] == ("full" if which == "glass" else "oof_remainder")
    if real.sums:
        assert status["file_sha256"] == d.ATTESTED


def test_rf_view_has_one_t1_t2_per_outer_fold(real):
    v = real.view("rf")
    assert v.router_mode == "two_pass" and sorted(v.fold_thresholds) == list(range(5))


def test_upstream_views_register_the_primary_operating_points(real):
    lr, ebm = real.view("lr"), real.view("ebm")
    assert "lr.in_stage_f2_cv" in lr.thresholds
    assert "ebm.nested_f2" in ebm.stored_decisions
    assert _status(ebm)["split_fingerprint"] in (d.VALIDATED, d.ATTESTED)


@pytest.mark.parametrize("name", list(d.OPERATING_POINTS))
def test_every_operating_point_resolves(real, name):
    view = real.view("lr" if name.startswith("lr.") else "ebm")
    flags = d.resolve_operating_point(view, name, stage4=real.stage4())
    assert flags.n == real.ref().n_train


def test_rf_pass1_candidate_is_extracted_and_checked(real):
    p = real.rf_result_path()
    cand = d.extract_rf_pass1_candidate(p, real.ref(), expected_sha256=real.sha(p))
    report = d.check_rf_pass1_score_equivalence(real.view("rf"), cand)
    # The verdict is a finding, reported by the read-only runner; here only that
    # the check reaches one on the real artifacts.
    assert report.n_rows == real.ref().n_train and report.conclusion
