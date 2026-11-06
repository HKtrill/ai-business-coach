"""
PR 37 (Commit 5 follow-up): the Stage 2 diagnostics boundary on the CURRENT
artifacts - the newest ones in the live models/ folders, or the four arbiters
``$BANK_ARTIFACT_MANIFEST`` names - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_current_artifacts.py

These run after every regeneration. They never compare a file with the
October 23 hashes or results (the historical tests read the archived October
artifacts for that; tests.support.archived_artifacts). Instead every current
artifact must pass the same structural and provenance checks: each router
view passes every mandatory boundary check, the RF Router has one threshold
pair per outer fold, every registered upstream operating point resolves, the
three-state trace builds with every check passing, and its population is the
one the frozen PR 38 protocol pins (V1). The PR 36 audit
(test_saved_artifacts.py) checks the same arbiters' recorded provenance,
reloads and serving.

Nothing is fitted, written or regenerated, and no test label is read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import shared.stage2.evaluation_protocol as ep
from shared.stage2 import diagnostics as d
from tests.support.archived_artifacts import CurrentReal
from tests.support.notebooks import BANK_ROOT

pytestmark = pytest.mark.realdata

MANDATORY = ("router_transform", "confidence", "decision_tokens", "fold_plan",
             "train_label_fingerprint.cross_fit", "train_label_fingerprint.full_train",
             "pass2_population")


@pytest.fixture(scope="module")
def current():
    return CurrentReal()


def _status(view) -> dict:
    frame = view.provenance_frame()
    return dict(zip(frame["name"], frame["status"]))


def test_current_artifacts_come_from_the_live_folders_and_are_never_attested(current):
    assert current.sums == {}
    for key in ("meta_ebm", "meta_ebm_on_blackbox"):
        Path(current.roots()[key]).resolve().relative_to(Path(BANK_ROOT).resolve())


@pytest.mark.parametrize("which", ["glass", "rf"])
def test_current_router_view_passes_every_mandatory_check(current, which):
    v = current.view(which)
    status = _status(v)
    for item in MANDATORY:
        assert status[item] == d.VALIDATED, item
    assert v.pass2_population["normalised"] in ("full", "oof_remainder")
    assert status["file_sha256"] != d.ATTESTED


def test_current_rf_router_has_one_t1_t2_per_outer_fold(current):
    v = current.view("rf")
    assert v.router_mode == "two_pass" and sorted(v.fold_thresholds) == list(range(5))


@pytest.mark.parametrize("name", list(d.OPERATING_POINTS))
def test_every_operating_point_resolves_on_the_current_upstream(current, name):
    view = current.view("lr" if name.startswith("lr.") else "ebm")
    flags = d.resolve_operating_point(view, name, stage4=current.stage4())
    assert flags.n == current.ref().n_train


def test_the_current_trace_passes_every_check_and_has_the_protocol_population(current):
    ref = current.ref()
    trace = d.build_router_trace(current.view("glass"), current.view("rf"), ref)
    checks = trace.check_frame()
    assert checks["passed"].all(), checks.loc[~checks["passed"], ["stage", "check", "detail"]]
    assert trace.index.equals(ref.index_train)
    report = ep.check_population_identity(
        trace, ep.PopulationIdentity.from_protocol(ep.load_frozen_protocol()))
    assert report.passed and report.complete, report.failures()


def test_the_current_rf_pass1_candidate_is_extracted_and_checked(current):
    p = current.rf_result_path()
    cand = d.extract_rf_pass1_candidate(p, current.ref(), expected_sha256=None)
    report = d.check_rf_pass1_score_equivalence(current.view("rf"), cand)
    assert report.n_rows == current.ref().n_train and report.conclusion
