"""
PR 42 Commit 4, Step 4A on the real training rows - opt-in, like every
realdata test (it reads ``data/raw`` and ``data/db``):

    pytest -m realdata tests/realdata/test_pr42_binning_drift.py

The notebook's Step 4A path (Cells 31-38), without writing a record:

1. The PR 39 Step 3A evidence (E-B, T1-T4, the retain decision) reproduces the
   accepted record ``c3_20261114_231604`` exactly from the training rows.
2. The row-level sources rebuild E-B's tables; flipping every held-out label of
   a fold leaves that fold's NSD and ECI unchanged (with its control).
3. Pooled out-of-fold counts reconcile with the folds, and the focal figures
   are PR 39's (about 1.03x, 2.50x, 2.49x; one fold of nsd_elevated at 374 rows).
4. The in-sample deciles and bins reproduce the exploratory notebook's
   accepted printed Cells 15A-15B, read from the pinned Git blob, also with
   the working copy's outputs cleared; the historical transcription is
   verified where the commit is present.
5. Every Step 4A claim holds (the review document's statements).

Only training rows are bound; no artifact is opened.
"""

from __future__ import annotations

import pytest

from feature_research import binning_drift as BD
from feature_research import encoding_audit as EA
from feature_research import feature_inventory as FI
from feature_research.training_side import draw_training_side

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def step4a():
    side = draw_training_side(verbose=False)
    X, y = side.X_train, side.y_train
    frames = FI.build_training_frames(X, y)
    T = EA.build_time_index(X)
    record, _ = BD.read_pr39_record()
    s2 = BD.stage2_side(X, y)
    repro = BD.reproduce_pr39(s2, record)
    sources = BD.source_values(s2, frames["stage2_sources"], repro["eb"])
    states = BD.macro_states(X, T, y)
    svals = BD.state_values(states, sources, s2, BD.fold_fit_values(s2, states, repro["eb"]))
    encoder = BD.encoder_comparison(s2, sources, states, svals)
    deciles = BD.decile_tables(s2, sources)
    hist = BD.historical_thresholds(s2, sources)
    periods = BD.period_evidence(s2, sources, T)
    straddling = BD.straddling_states(states, svals)
    provenance = BD.label_provenance_by_bin(X, y, T, s2, sources)
    historical_check = BD.verify_historical()
    assignments = BD.focal_assignments(s2, sources, T, states, svals, deciles)
    found = BD.findings(eb=repro["eb"], historical_check=historical_check, implied=BD.implied_cut_rows(),
                        encoder=encoder, deciles=deciles, historical_cuts=hist, periods=periods,
                        straddling=straddling, provenance=provenance, assignments=assignments)
    return dict(side=s2, repro=repro, sources=sources, states=states, svals=svals, encoder=encoder,
                deciles=deciles, hist=hist, provenance=provenance, historical_check=historical_check,
                assignments=assignments, found=found)


def test_pr39_step3a_reproduces_exactly(step4a):
    checks = step4a["repro"]["checks"]
    assert checks["passed"].all(), checks.to_string()
    assert step4a["repro"]["decision"]["NSD"]["outcome"] == step4a["repro"]["decision"]["ECI"]["outcome"] == "retain"


def test_sources_rebuild_e_b_and_a_folds_labels_never_reach_its_encodings(step4a):
    assert step4a["sources"].checks["passed"].all(), step4a["sources"].checks.to_string()
    flips = BD.heldout_label_checks(step4a["side"], folds=[0])
    assert flips["passed"].all(), flips.to_string()


def test_pooled_counts_reconcile_with_the_folds_and_the_focal_figures_are_pr39s(step4a):
    eb = step4a["repro"]["eb"]["families"]
    for fam in BD.FAMILIES:
        for b, e in eb[fam]["oof"]["bins"].items():
            assert sum(c["support"] for c in e["folds"]) == e["pooled"]["support"]
            assert sum(c["positives"] for c in e["folds"]) == e["pooled"]["positives"]
    lifts = {b: eb[f]["oof"]["bins"][b]["pooled"]["lift"] for b, f in BD.FOCAL.items()}
    assert lifts == pytest.approx({"nsd_elevated": 2.486, "eci_warm": 1.031, "eci_hot": 2.501}, abs=5e-4)
    assert [c["support"] for c in eb["NSD"]["oof"]["bins"]["nsd_elevated"]["folds"]] == [374, 275, 255, 258, 259]


def test_macro_states_and_the_encoder_decomposition(step4a):
    assert len(step4a["states"].table) == 346
    assert step4a["svals"]["checks"]["passed"].all(), step4a["svals"]["checks"].to_string()
    d = step4a["encoder"]["decomposition"].set_index("bin")
    assert d.loc["nsd_elevated", "fold_fit_minus_in_sample"] == pytest.approx(-0.618, abs=1e-3)
    assert d.loc["nsd_elevated", "heldout_minus_fold_fit"] == pytest.approx(-0.022, abs=1e-3)
    assert abs(d.loc["eci_warm", "oof_minus_in_sample"]) < 0.01 and d.loc["eci_hot", "oof_minus_in_sample"] > 0
    pf = step4a["encoder"]["per_fold"]
    diffs = pf[pf["bin"] == "nsd_elevated"]["heldout_minus_fit"]
    assert diffs.min() < 0 < diffs.max()                      # the pooled contrast cancels per-fold differences


def test_the_exploratory_cells_and_the_historical_transcription_reproduce(step4a, tmp_path):
    out = BD.exploratory_outputs()
    assert out["status"] == "verified", out["detail"]
    checks = BD.exploratory_reproduction(step4a["deciles"], step4a["repro"]["eb"], out)
    assert checks["passed"].all() and not checks["check"].str.contains("not checked").any(), checks.to_string()
    if out["source"] == "git blob":                           # Commit 6 may clear the working copy's outputs
        import json
        nb = json.loads((BD.BANK_ROOT / BD.EXPLORATORY["notebook"]).read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        cleared = tmp_path / "exploratory_feature_research.ipynb"
        cleared.write_text(json.dumps(nb), encoding="utf-8")
        again = BD.exploratory_outputs(working_copy=cleared)
        assert (again["status"], again["source"]) == ("verified", "git blob")
        checks = BD.exploratory_reproduction(step4a["deciles"], step4a["repro"]["eb"], again)
        assert checks["passed"].all() and not checks["check"].str.contains("not checked").any()
    parity = BD.compute_lift_parity(step4a["side"], step4a["sources"], step4a["deciles"])
    assert parity["passed"].all(), parity.to_string()
    assert step4a["historical_check"]["status"] in ("verified", "unavailable")
    assert step4a["provenance"]["checks"]["passed"].all()
    overall = step4a["provenance"]["overall"]
    assert overall["NSD"]["oof_mean"] == pytest.approx(0.469, abs=1e-3)       # Commit 3's held-out share
    assert overall["ECI"]["oof_mean"] == pytest.approx(0.516, abs=1e-3)


def test_the_focal_identities_hold_on_rows_and_states(step4a):
    hot, nsd = step4a["assignments"]["eci_hot"], step4a["assignments"]["nsd_elevated"]
    assert hot["top_decile_is_tail"] and hot["cluster_is_whole_states"] and hot["oof_cluster_is_same_rows"]
    assert hot["cluster_in_bin_under_every_encoder"] and hot["cluster_rows"] == 1456
    assert hot["cluster_periods"] == ["2009-05"] and len(hot["cluster_states"]) == 4
    assert nsd["fold"] == 0 and len(nsd["states_in_bin_under_fold"]) == 7
    assert {s["periods"] for s in nsd["states_in_bin_under_fold"]} == {"2009-05"}
    assert nsd["in_sample_bin_states_bins_under_fold"] == ["nsd_hot"] and nsd["moved_rows_folds"] == [0]


def test_every_step4a_claim_holds(step4a):
    claims = BD.claims_table(step4a["found"])
    assert claims["holds"].all(), claims[~claims["holds"]].to_string()
    assert sorted(f["bin"] for f in step4a["found"]) == ["eci_hot", "eci_warm", "nsd_elevated"]
