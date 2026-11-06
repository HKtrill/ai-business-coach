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
GLASS-arm RF result named by the RF Router's ``rf_params_origin``. All files
are read from the preserved October 23 archive (``PR37_BASELINE_SHA256SUMS``;
tests.support.archived_artifacts), never from the live models/ folders, and
each must equal its recorded SHA-256. The same checks on the newest live
artifacts are in test_current_artifacts.py.
"""

from __future__ import annotations

import pytest

from shared.stage2 import diagnostics as d
from tests.support.archived_artifacts import HistoricalReal
from tests.support.router_artifacts import (
    ROUTER_SPLIT_FP,
    STAGE_IO_SPLIT_FP,
    TRAIN_LABEL_FP,
)

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def real():
    return HistoricalReal()


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
