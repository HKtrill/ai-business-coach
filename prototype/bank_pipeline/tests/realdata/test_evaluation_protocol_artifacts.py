"""
PR 37 Commit 4 (Step 4C): the PR 38 evaluation protocol's pinned identities
against the SAVED October 23 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_evaluation_protocol_artifacts.py

The protocol pins the evaluation population (rows, positives, per-fold counts,
fold plan, fingerprints) and the frozen upstream operating points (artifact
SHA-256, LR threshold record, EBM fold-nested thresholds, flag counts). These
tests check those pins against the historical trace and the GLASS arm's LR /
EBM views that the pre-correction baseline was built from. Nothing is fitted,
written or regenerated, no corrected result exists, and no test label is read
(loader: tests.support.router_artifacts). All files are read from the
preserved October 23 archive (``PR37_BASELINE_SHA256SUMS``;
tests.support.archived_artifacts), never from the live models/ folders, and
each must equal its recorded SHA-256.

The row-aligned flag fingerprints are pinned at Step 4D (D-C10). Until then
they are reported as pending; print them, read-only, with

    python -m tests.realdata.test_evaluation_protocol_artifacts
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import shared.stage2.evaluation_protocol as ep
from shared.stage2 import diagnostics as d
from tests.realdata.test_router_trace_artifacts import HISTORICAL_STATE_COUNTS
from tests.support.archived_artifacts import HistoricalReal

pytestmark = pytest.mark.realdata

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN


def _load():
    real = HistoricalReal()
    ref = real.ref()
    glass, rf, lr, ebm = (real.view(w) for w in ("glass", "rf", "lr", "ebm"))
    protocol = ep.load_evaluation_protocol(ep.working_protocol_path())
    return SimpleNamespace(trace=d.build_router_trace(glass, rf, ref), glass=glass, lr=lr,
                           ebm=ebm, protocol=protocol)


@pytest.fixture(scope="module")
def run():
    return _load()


def _upstream(run):
    return ep.check_upstream_identity(d.primary_reference_flags(run.lr, run.ebm), run.protocol,
                                      lr_view=run.lr, ebm_view=run.ebm, population=run.trace)


def test_the_historical_trace_has_the_protocol_population_identity(run):
    report = ep.check_population_identity(run.trace,
                                          ep.PopulationIdentity.from_protocol(run.protocol))
    assert report.passed and report.complete, report.failures()


def test_the_frozen_upstream_operating_points_match_the_protocol(run):
    report = _upstream(run)
    assert report.passed, report.failures()
    pending = report.pending()
    assert all("row-aligned flag fingerprint" in p for p in pending)   # D-C10, until Step 4D
    assert set(report.values) == {"lr.in_stage_f2_cv", "ebm.nested_f2"}
    verified = ep.verify_reference(ep.UpstreamSources.from_views(run.lr, run.ebm), run.protocol,
                                   population=run.trace, complete=False)
    assert verified.combination == "or" and not verified.complete    # official only after 4D
    assert int(verified.flags.flags.sum()) == run.protocol["upstream"]["expected_flags"]["or"]
    with pytest.raises(ep.ProtocolError, match="incomplete"):
        ep.verify_reference(ep.UpstreamSources.from_views(run.lr, run.ebm), run.protocol,
                            population=run.trace)


def test_paired_historical_routings_reproduce_the_baseline_state_counts(run):
    paired = ep.PairedRouting.from_trace(run.trace)
    for arm, want in HISTORICAL_STATE_COUNTS.items():
        c = paired.counts(arm)
        assert (c.pass1_rows, c.pass2_rows, c.abstain_rows) == (want[P1], want[P2], want[AB])
    assert ep.PairedRouting.pre_post(run.glass, run.glass).changed_rows() == 0


if __name__ == "__main__":   # read-only: prints the values to pin at Step 4D (D-C10)
    report = _upstream(_load())
    for name, ok, detail in report.checks:
        print(f"{'PASS' if ok else 'PENDING' if ok is None else 'FAIL':8} {name}: {detail}")
    print("\nflag_fingerprints to pin:", dict(report.values))
