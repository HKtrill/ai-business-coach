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
three-state trace builds with every check passing, its population is the
one the frozen PR 38 protocol pins (V1), and the RF Router's parameter origin is
the one its generation records (a GLASS-arm RF result, or the frozen RF
profile, PR 37 decision M8). The PR 36 audit
(test_saved_artifacts.py) checks the same arbiters' recorded provenance,
reloads and serving.

Nothing is fitted, written or regenerated, and no test label is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import pytest

import shared.stage2.evaluation_protocol as ep
from shared import pinned_artifacts as pa
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


def test_the_current_rf_router_parameter_origin_is_verified(current):
    """
    The RF Router records where its forest parameters come from, and each
    generation records it differently:

    * a notebook-trained router (PR 36-37) names the GLASS-arm RF result it read
      them from at run time. That file's OOF scores must load as the Pass 1
      candidate and the equivalence check must reach a verdict;
    * a router fitted by the PR 38 cross-fit runner records the frozen RF profile
      that owns them, ``{"profile": <identity>}`` (PR 37 decision M8: no run reads
      them from a GLASS-side artifact; ``stage2_crossfit.runner``). Its identity
      must be the frozen profile's (status, pinned fingerprint, confirmed
      approval), the router's saved configuration exactly the primary one built
      from the frozen contract and that profile (``check_rf_primary_config``),
      the router file the accepted PR 38 run's (pinned size and SHA-256, the
      official structural acceptance), and that run's version-controlled
      manifest must record the same profile identity.

    Any other origin fails.
    """
    origin = current.view("rf").metadata.get("rf_params_origin")
    if not isinstance(origin, Mapping):
        p = current.rf_result_path()           # a path, or no origin: refused there
        cand = d.extract_rf_pass1_candidate(p, current.ref(), expected_sha256=None)
        report = d.check_rf_pass1_score_equivalence(current.view("rf"), cand)
        assert report.n_rows == current.ref().n_train and report.conclusion
        return
    from blackbox_pipeline.models.rf_router.config import RFRouterConfig
    from blackbox_pipeline.models.rf_router.profile import (check_rf_primary_config,
                                                            load_rf_primary_profile)
    from stage2_crossfit import acceptance

    assert set(origin) == {"profile"}, f"rf_params_origin {origin!r} is neither a path nor a profile"
    profile = load_rf_primary_profile()
    assert dict(origin["profile"]) == profile.identity(), origin
    router = Path(current.paths("meta_ebm_on_blackbox")["stage2"])
    check_rf_primary_config(RFRouterConfig.from_dict(joblib.load(router)["config"]))
    pin = next(p for p in pa.STAGE2_PINS if p.role == "stage2.rf_router")
    assert router.samefile(pa.verify_pin(pin))
    assert router.samefile(acceptance.require_accepted(pa.STAGE2_RUN_ID).rf_path)
    assert pa.stage2_manifest_problems() == []
    manifest = json.loads((Path(BANK_ROOT) / pa.STAGE2_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["identities"]["rf_profile"] == profile.identity()
