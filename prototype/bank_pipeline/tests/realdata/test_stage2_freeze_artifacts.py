"""
PR 37 Commit 4 (Step 4D): the frozen Stage 2 specification re-verified against
the SAVED October 23 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_stage2_freeze_artifacts.py

This is the re-verification the frozen protocol names (``freeze.reverification``,
D-C10). It must pass before Commit 4 and before any PR 38 run; a failure voids
the freeze. A copied fingerprint is never taken as proof: every check below is
recomputed from the artifacts.

1. The frozen protocol's population identity holds on the historical trace.
2. The five-layer upstream identity is complete: artifact SHA-256 at
   extraction and now, rows and labels, threshold provenance, per-row
   decisions, counts and the pinned row-aligned flag fingerprints. The
   complementarity reference verifies completely.
3. Both frozen arm profiles hold the values the historical artifacts record:
   the RF hyperparameters (RF Router artifact and GLASS-arm RF result), the
   GLASS legacy per-rule settings (GLASS Router artifact) and the GLASS ranker
   hyperparameters.

Every record is read through its strict loader (status, pinned fingerprint,
confirmed approval), as PR 38 reads it.

Nothing is fitted, written or regenerated, no corrected result exists, and no
test label is read (loader: tests.support.router_artifacts). All files are
read from the preserved October 23 archive (``PR37_BASELINE_SHA256SUMS``;
tests.support.archived_artifacts), never from the live models/ folders, and
each must equal its recorded SHA-256.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import profile as rf_profile
from glass_pipeline.glass_router import profile as glass_profile
from shared.stage2 import diagnostics as d
from tests.support.archived_artifacts import HistoricalReal

pytestmark = pytest.mark.realdata


def _load():
    real = HistoricalReal()
    ref = real.ref()
    glass, rf, lr, ebm = (real.view(w) for w in ("glass", "rf", "lr", "ebm"))
    return SimpleNamespace(real=real, ref=ref, trace=d.build_router_trace(glass, rf, ref),
                           glass=glass, rf=rf, lr=lr, ebm=ebm,
                           protocol=ep.load_frozen_protocol())


@pytest.fixture(scope="module")
def run():
    return _load()


def _upstream(run):
    return ep.check_upstream_identity(d.primary_reference_flags(run.lr, run.ebm), run.protocol,
                                      lr_view=run.lr, ebm_view=run.ebm, population=run.trace)


def test_the_frozen_protocol_identifies_the_real_population(run):
    report = ep.check_population_identity(run.trace,
                                          ep.PopulationIdentity.from_protocol(run.protocol))
    assert report.passed and report.complete, report.failures()


def test_the_frozen_upstream_identity_is_complete_on_the_real_artifacts(run):
    report = _upstream(run)
    report.require(complete=True)                      # every layer, fingerprints included
    assert dict(report.values) == dict(run.protocol["upstream"]["flag_fingerprints"])
    verified = ep.verify_reference(ep.UpstreamSources.from_views(run.lr, run.ebm), run.protocol,
                                   population=run.trace)
    assert verified.complete and verified.combination == "or"
    assert int(verified.flags.flags.sum()) == run.protocol["upstream"]["expected_flags"]["or"]


def test_the_frozen_arm_profiles_hold_the_historical_values(run):
    rf_prof = rf_profile.load_rf_primary_profile()
    rf_cfg = rf_prof.settings["router_config"]
    glass_prof = glass_profile.load_glass_primary_profile()
    want = dict(rf_cfg["rf_params"])
    assert dict(run.rf.metadata["rf_params"]) == want                 # RF Router artifact
    assert (run.rf.metadata["mode"], run.rf.metadata["n_oof_folds"]) == (
        rf_cfg["mode"], rf_cfg["n_oof_folds"])
    path = run.real.rf_result_path()
    sources = {s["role"]: s for s in rf_prof.provenance["historical_sources"]}
    assert d.file_sha256(path) == sources["glass_arm.rf_result"]["sha256"]
    candidate = d.extract_rf_pass1_candidate(path, run.ref, expected_sha256=run.real.sha(path))
    assert dict(candidate.params) == want                              # GLASS-arm RF result
    assert glass_profile.ranker_settings()["rf_params"] == want
    router = glass_prof.settings["router_config"]
    replaced = {"pass2_population", "max_union_leakage_rate_pass1"}   # F17 primary; contract
    history = dict(run.glass.metadata["config_full"])
    assert history and {k: v for k, v in history.items() if k not in replaced} == {
        k: router[k] for k in history if k not in replaced}


if __name__ == "__main__":   # read-only: prints every upstream identity check
    for name, ok, detail in _upstream(_load()).checks:
        print(f"{'PASS' if ok else 'PENDING' if ok is None else 'FAIL':8} {name}: {detail}")
