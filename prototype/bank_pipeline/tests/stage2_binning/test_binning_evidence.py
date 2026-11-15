"""
PR 39, Commit 3, Step 3A: the NSD / ECI binning evidence (stage2_binning).

Covered: the binning identity; the statistics; T1-T4 at their D6 boundaries
and the retain-or-revise outcome; fold isolation of the out-of-fold engineer;
the E-B2 summary from a stand-in accepted candidate; the write-once record and
``check``. Synthetic data only (the real-data outcome: the realdata module).
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import shared.stage2.binning as binning
from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY, binning_identity
from stage2_binning import evidence as E
from stage2_binning import runner as R
from stage2_crossfit.g13 import bin_source, inner_refit_features
from tests.support import stage2_crossfit_world as W

#: The Stage 2 binning identity the Step 3A record freezes (thresholds retained).
FROZEN_BINNING_SHA256 = "9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376"


# ======================================================================
# binning identity
# ======================================================================
def test_the_binning_identity_is_the_frozen_one():
    ident = binning_identity()
    assert ident == {"id": "stage2-binning", "sha256": FROZEN_BINNING_SHA256,
                     "fingerprint": FROZEN_BINNING_SHA256[:16],
                     "covers": ["BINNING_STRATEGY", "RF_FEATURES_BINARY"]}


def test_any_threshold_name_or_order_change_moves_the_identity(monkeypatch):
    moved = copy.deepcopy(BINNING_STRATEGY)
    moved["ECI"]["bins"][2] = ("eci_warm", 0.0758, 0.0949)
    monkeypatch.setattr(binning, "BINNING_STRATEGY", moved)
    assert binning_identity()["sha256"] != FROZEN_BINNING_SHA256
    monkeypatch.setattr(binning, "BINNING_STRATEGY", BINNING_STRATEGY)
    monkeypatch.setattr(binning, "RF_FEATURES_BINARY",
                        [RF_FEATURES_BINARY[1], RF_FEATURES_BINARY[0], *RF_FEATURES_BINARY[2:]])
    assert binning_identity()["sha256"] != FROZEN_BINNING_SHA256


def test_the_revalidated_families_and_their_intended_semantics():
    nsd, eci = E.family_bins("NSD"), E.family_bins("ECI")
    assert nsd == ["nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot"]
    assert eci == ["eci_cold", "eci_mid", "eci_warm", "eci_hot"]
    for names in (nsd, eci):                                 # contiguous, in RF_FEATURES_BINARY
        i = RF_FEATURES_BINARY.index(names[0])
        assert RF_FEATURES_BINARY[i:i + 4] == names
    assert set(E.DIRECTIONS) == set(nsd + eci)
    assert [E.DIRECTIONS[b] for b in nsd] == ["below", None, "above", "above"]
    assert [E.DIRECTIONS[b] for b in eci] == ["below", None, "above", "above"]
    for fam in E.FAMILIES:                                   # thresholds rise, no gap
        bins = BINNING_STRATEGY[fam]["bins"]
        assert bins[0][1] is None and bins[-1][2] is None
        assert all(a[2] == b[1] for a, b in zip(bins, bins[1:]))
        assert all(a[2] < b[2] for a, b in zip(bins[:-2], bins[1:-1]))


# ======================================================================
# statistics
# ======================================================================
def test_wilson_interval_closed_form():
    lo, hi = E.wilson(5, 10)
    assert lo == pytest.approx(0.236593, abs=1e-6) and hi == pytest.approx(0.763407, abs=1e-6)
    lo, hi = E.wilson(1, 10)
    assert lo == pytest.approx(0.017876, abs=1e-6) and hi == pytest.approx(0.404150, abs=1e-6)
    assert E.wilson(0, 10)[0] == 0.0 and E.wilson(0, 0) == [None, None]


def test_bin_table_counts_and_fold_relative_lifts():
    labels = np.array(["a", "a", "b", "b", "a", "b", "a", "b"], dtype=object)
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    t = E.bin_table(labels, y, fold, ["a", "b"], 2)
    assert t["base_rate"]["pooled"] == {"rows": 8, "positives": 3, "base_rate": 3 / 8}
    a = t["bins"]["a"]
    assert (a["pooled"]["support"], a["pooled"]["positives"]) == (4, 2)
    assert a["pooled"]["lift"] == pytest.approx((2 / 4) / (3 / 8))
    assert a["folds"][0]["lift"] == pytest.approx((1 / 2) / (1 / 4))      # fold 0's base
    assert a["folds"][1]["lift"] == pytest.approx((1 / 2) / (2 / 4))      # fold 1's base
    assert a["folds"][0]["expected_positives"] == pytest.approx(2 * 1 / 4)


def test_family_labels_refuse_a_gap_or_an_overlap():
    frame = pd.DataFrame(np.eye(4, dtype=int), columns=E.family_bins("NSD"))
    assert list(E.family_labels(frame, "NSD")) == E.family_bins("NSD")
    frame.iloc[0, 1] = 1
    with pytest.raises(E.EvidenceError, match="exactly one bin"):
        E.family_labels(frame, "NSD")


# ======================================================================
# triggers T1-T4 (D6) and the decision
# ======================================================================
def _oof(family, *, pooled=(0.5, 1.0, 2.0, 4.0), folds=None, expected=50.0, ci=None):
    names = E.family_bins(family)
    folds = folds or [list(pooled)] * 5
    ci = ci or {}
    bins = {}
    for i, b in enumerate(names):
        default = (pooled[i] * 0.9, pooled[i] * 1.1)
        exp = expected[b] if isinstance(expected, dict) and b in expected else (
            expected if not isinstance(expected, dict) else 50.0)
        bins[b] = {"pooled": {"lift": pooled[i], "lift_wilson95": list(ci.get(b, default))},
                   "folds": [{"lift": f[i], "expected_positives": exp} for f in folds]}
    return {"bins": bins}


def _fired(family, oof):
    t = E.triggers(family, oof)
    return [k for k in ("T1", "T2", "T3", "T4") if t[k]["fired"]]


def test_a_monotonic_well_supported_family_fires_nothing_and_is_retained():
    for fam in E.FAMILIES:
        assert _fired(fam, _oof(fam)) == []
        d = E.decide({fam: E.triggers(fam, _oof(fam))})[fam]
        assert d["outcome"] == "retain" and d["candidate"] is None
        assert d["thresholds"] == [list(b) for b in BINNING_STRATEGY[fam]["bins"]]


def test_t1_fires_below_ten_expected_positives_in_any_fold():
    assert _fired("NSD", _oof("NSD", expected={"nsd_elevated": 9.999})) == ["T1"]
    assert _fired("NSD", _oof("NSD", expected={"nsd_elevated": 10.0})) == []


def test_t2_fires_only_wholly_on_the_wrong_side_of_one_for_a_directed_bin():
    assert _fired("ECI", _oof("ECI", ci={"eci_cold": (1.01, 1.3)})) == ["T2"]
    assert _fired("ECI", _oof("ECI", ci={"eci_cold": (0.99, 1.3)})) == []
    assert _fired("ECI", _oof("ECI", ci={"eci_warm": (0.80, 0.99)})) == ["T2"]
    assert _fired("ECI", _oof("ECI", ci={"eci_warm": (0.98, 1.08)})) == []     # straddles 1
    assert _fired("ECI", _oof("ECI", ci={"eci_mid": (1.5, 2.0)})) == []        # no direction
    assert _fired("NSD", _oof("NSD", ci={"nsd_warm": (0.1, 0.2)})) == []


def test_t3_needs_a_pooled_inversion_and_four_of_five_folds():
    inv = [0.5, 2.0, 1.0, 4.0]                               # warm > elevated
    ok = [0.5, 1.0, 2.0, 4.0]
    assert _fired("NSD", _oof("NSD", pooled=inv, folds=[inv] * 4 + [ok])) == ["T3"]
    assert _fired("NSD", _oof("NSD", pooled=inv, folds=[inv] * 3 + [ok] * 2)) == []
    assert _fired("NSD", _oof("NSD", pooled=ok, folds=[inv] * 5)) == []
    t = E.triggers("NSD", _oof("NSD", pooled=inv, folds=[inv] * 4 + [ok]))["T3"]
    assert [p["fired"] for p in t["pairs"]] == [False, True, False]
    assert t["pairs"][1]["inverted_folds"] == [0, 1, 2, 3]


def test_t4_fires_below_a_hot_lift_of_one_and_a_half():
    assert _fired("ECI", _oof("ECI", pooled=(0.5, 0.8, 1.2, 1.49))) == ["T4"]
    assert _fired("ECI", _oof("ECI", pooled=(0.5, 0.8, 1.2, 1.5))) == []


def test_a_trigger_requires_a_candidate_and_never_adopts_one():
    d = E.decide({"NSD": E.triggers("NSD", _oof("NSD", expected={"nsd_hot": 3.0}))})["NSD"]
    assert d["flagged"] and d["fired"] == ["T1"] and d["outcome"] == "candidate_required"
    assert d["candidate"] is None and d["preflight"].startswith("required")


# ======================================================================
# E-B on a synthetic train side: fold isolation and tables
# ======================================================================
@pytest.fixture(scope="module")
def ts():
    return W.synthetic_train_side()


def test_out_of_fold_sources_are_the_g13_refit_and_exclude_their_fold(ts):
    oof, prov = R.oof_sources(ts)
    pd.testing.assert_frame_equal(
        oof, inner_refit_features(ts.X_train, ts.y_train, np.asarray(ts.fold_id)))
    n = len(ts.y_train)
    for p in prov:
        k = p["fold"]
        assert p["heldout_rows_in_fit"] == 0
        assert p["heldout_rows"] == int((ts.fold_id == k).sum())
        assert p["fit_rows"] == n - p["heldout_rows"]


def test_a_folds_labels_never_reach_its_own_features(ts):
    base, _ = R.oof_sources(ts)
    y = ts.y_train.copy()
    held = np.asarray(ts.fold_id) == 0
    y[held] = 1 - y[held]                                    # every fold-0 label flipped
    moved, _ = R.oof_sources(SimpleNamespace(X_train=ts.X_train, y_train=y,
                                             fold_id=ts.fold_id, folds=ts.folds))
    cols = ["neighborhood_subscription_density", "economic_curvature_intensity"]
    pd.testing.assert_frame_equal(base.loc[held, cols], moved.loc[held, cols])
    assert not base.loc[~held, cols].equals(moved.loc[~held, cols])     # the others saw them


def test_eb_tables_partition_the_rows_in_every_fold(ts):
    eb = R.eb_evidence(ts)
    n, n_folds = len(ts.y_train), len(ts.folds)
    for fam in E.FAMILIES:
        for kind in ("oof", "in_sample"):
            bins = eb["families"][fam][kind]["bins"]
            assert sum(b["pooled"]["support"] for b in bins.values()) == n
        bins = eb["families"][fam]["oof"]["bins"]
        for k in range(n_folds):
            assert sum(b["folds"][k]["support"] for b in bins.values()) == \
                int((ts.fold_id == k).sum())
        assert all("folds" not in b for b in eb["families"][fam]["in_sample"]["bins"].values())
        oof_lab = E.family_labels(bin_source(R.oof_sources(ts)[0]), fam)
        assert eb["families"][fam]["oof"]["bins"][E.family_bins(fam)[0]]["pooled"]["support"] \
            == int((oof_lab == E.family_bins(fam)[0]).sum())
    joint = eb["nsd_eci_joint_oof"]
    assert sum(map(sum, joint["rows"])) == n


# ======================================================================
# E-B2 from a stand-in accepted candidate
# ======================================================================
def _rule(rid, *conds, pass_name=None):
    r = {"rule_id": rid, "segment": [list(c) for c in conds], "predicted_class": 0,
         "precision": 0.95}
    if pass_name:
        r.update({"pass": pass_name, "segment": [tuple(c) for c in conds]})
    return r


def fake_candidate(*, bundle_pass1=None):
    p1 = [[_rule(10 + k, ("nsd_cold", 1), ("dow_cold", 0)), _rule(20 + k, ("eci_warm", 0))]
          for k in range(5)]
    p1[3] = [_rule(33, ("jed_cold", 1))]                      # fold 3: no NSD/ECI bin
    full_p1 = [_rule(1, ("nsd_cold", 1), ("eci_mid", 1))]
    full_p2 = [_rule(2, ("nsd_hot", 1), ("cci_hot1", 1), pass_name="Pass 2")]
    folds = [{"fold": k, "pass1_rules": p1[k], "rule_ids": {"pass2": [100 + k]}} for k in range(5)]
    bundle_rules = [_rule(r["rule_id"], *r["segment"], pass_name="Pass 1")
                    for r in (bundle_pass1 or full_p1)] + full_p2
    imp = np.linspace(1, 29, 29) / 435.0
    rf = {"feature_names": list(RF_FEATURES_BINARY),
          "pass1_model": SimpleNamespace(named_steps={"clf": SimpleNamespace(
              feature_importances_=imp)}),
          "pass2_model": SimpleNamespace(named_steps={"clf": SimpleNamespace(
              feature_importances_=imp[::-1])})}
    files = {"glass.joblib": {"bundle": {"rules": bundle_rules}}, "rf.joblib": rf}
    return SimpleNamespace(
        run_id="c3_test", role="primary", manifest_sha256="0" * 64,
        glass_path="glass.joblib", rf_path="rf.joblib",
        evidence={"full_train": {"glass": {"pass1_rules": full_p1,
                                           "rule_ids": {"pass2": [2]}}}},
        fold_evidence=lambda arm, k: folds[k],
        load=lambda path: files[path]["bundle"] if path == "glass.joblib" else files[path],
        relative=lambda path: path, sha256=lambda path: "f" * 64)


def test_eb2_counts_rule_literals_conjunctions_and_rf_ranks():
    eb2 = R.eb2_evidence(fake_candidate())
    nsd_cold, eci_warm = eb2["bins"]["nsd_cold"], eb2["bins"]["eci_warm"]
    assert nsd_cold["glass_pass1_folds_with_bin"] == [0, 1, 2, 4]
    assert nsd_cold["glass_pass1_folds"] == {"as_1": 4, "as_0": 0}
    assert nsd_cold["glass_pass1_full_train"] == {"as_1": 1, "as_0": 0}
    assert nsd_cold["conjoined_with"] == ["dow_cold", "eci_mid"]
    assert eci_warm["glass_pass1_folds"] == {"as_1": 0, "as_0": 4}          # negated literal
    assert eb2["bins"]["nsd_hot"]["glass_pass2_full_train"] == {"as_1": 1, "as_0": 0}
    assert eb2["bins"]["nsd_cold"]["rf_mdi_rank_of_29"] == {"pass1": 29, "pass2": 1}
    assert eb2["pass2_rule_ids_by_fold"] == [[100], [101], [102], [103], [104]]
    assert eb2["glass_rules"]["pass1_by_fold"][0][0] == {
        "rule_id": 10, "conditions": [["dow_cold", 0], ["nsd_cold", 1]], "precision": 0.95}
    assert eb2["glass_rules"]["pass2_full_train"][0]["conditions"] == [["cci_hot1", 1],
                                                                        ["nsd_hot", 1]]
    assert eb2["not_recorded"] and eb2["source"]["run_id"] == "c3_test"


def test_eb2_refuses_a_bundle_that_is_not_the_full_train_evidence():
    with pytest.raises(E.EvidenceError, match="Pass 1 rules"):
        R.eb2_evidence(fake_candidate(bundle_pass1=[_rule(9, ("nsd_hot", 1))]))


def test_eb2_reads_importances_by_feature_name_and_refuses_other_names():
    cand = fake_candidate()
    rf = cand.load("rf.joblib")
    base = R.eb2_evidence(cand)["bins"]
    for p in ("pass1", "pass2"):                             # same forest, columns reversed
        clf = rf[f"{p}_model"].named_steps["clf"]
        clf.feature_importances_ = clf.feature_importances_[::-1]
    rf["feature_names"] = list(reversed(RF_FEATURES_BINARY))
    assert R.eb2_evidence(cand)["bins"] == base
    rf["feature_names"] = [*RF_FEATURES_BINARY[:-1], "other"]
    with pytest.raises(E.EvidenceError, match="RF_FEATURES_BINARY"):
        R.eb2_evidence(cand)


# ======================================================================
# the record: write once, check recomputes
# ======================================================================
def test_run_writes_once_and_check_recomputes_every_section(ts, tmp_path):
    record = R.run(run_id="c3_test", train_side=ts, candidate=fake_candidate(),
                   record_root=tmp_path)
    path = tmp_path / "c3_test" / R.RECORD_FILE
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and json.loads(text) == record
    assert record["registered"] is False and record["status"].startswith("NOT official")
    assert record["binning_identity"]["sha256"] == FROZEN_BINNING_SHA256
    assert record["eb2"]["source"]["run_id"] == "c3_test"
    assert R.check("c3_test", train_side=ts, candidate=fake_candidate(), record_root=tmp_path)
    with pytest.raises(R.RunnerError, match="written once"):
        R.run(run_id="c3_test", train_side=ts, candidate=fake_candidate(), record_root=tmp_path)

    tampered = json.loads(text)
    tampered["eb"]["families"]["NSD"]["oof"]["bins"]["nsd_hot"]["pooled"]["positives"] += 1
    (tmp_path / "c3_bad").mkdir()
    (tmp_path / "c3_bad" / R.RECORD_FILE).write_text(
        json.dumps({**tampered, "run_id": "c3_bad"}), encoding="utf-8")
    with pytest.raises(R.RunnerError, match="sections differ \\['eb'\\]"):
        R.check("c3_bad", train_side=ts, candidate=fake_candidate(), record_root=tmp_path)


def test_a_non_registered_run_needs_both_inputs(ts, tmp_path):
    with pytest.raises(R.RunnerError, match="both"):
        R.run(run_id="c3_x", train_side=ts, record_root=tmp_path)


# ======================================================================
# E-B2 on a real (synthetic-data) PR 38 candidate: the saved layouts
# ======================================================================
@pytest.fixture(scope="module")
def synthetic_candidate(tmp_path_factory):
    """A PR 38 primary written by the PR 38 runner on synthetic rows, loaded as a candidate."""
    import contextlib
    import io

    import stage2_crossfit.acceptance as A
    import stage2_crossfit.runner as SR
    from tests.support.rf_router_world import serial_forests
    base = tmp_path_factory.mktemp("eb2")
    run, role = "r3a_eb2", "test"
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())
    try:
        with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
            SR.run_primary(W.synthetic_train_side(), W.make_settings(), base / "models" / run / role,
                           run, base / "records" / run,
                           attempt={"authorization": {"name": "authorization.json",
                                                      "sha256": "0" * 64}})
    finally:
        mp.undo()
    cand = A.load_candidate(run, role=role, output_root=base / "models",
                            record_dir=base / "records" / run)
    assert cand.intact
    return cand


def test_eb2_reads_the_pr38_saved_layouts(synthetic_candidate):
    cand = synthetic_candidate
    eb2 = R.eb2_evidence(cand)
    assert set(eb2["bins"]) == {b for fam in E.FAMILIES for b in E.family_bins(fam)}
    assert len(eb2["glass_rules"]["pass1_by_fold"]) == 5
    assert [r["conditions"] for r in eb2["glass_rules"]["pass1_by_fold"][0]] == \
        E.rule_literals(cand.fold_evidence("glass", 0)["pass1_rules"])
    assert all("coverage" in r for r in eb2["glass_rules"]["pass2_full_train"])
    rf = cand.load(cand.rf_path)
    imp = rf["pass1_model"].named_steps["clf"].feature_importances_
    b = "nsd_cold"
    assert eb2["bins"][b]["rf_mdi"]["pass1"] == round(float(imp[rf["feature_names"].index(b)]), 12)
    ranks = [eb2["bins"][x]["rf_mdi_rank_of_29"]["pass1"] for x in eb2["bins"]]
    assert len(set(ranks)) == len(ranks) and all(1 <= r <= 29 for r in ranks)
    assert eb2["source"]["glass_bundle"]["sha256"] == cand.sha256(cand.glass_path)
    json.dumps(eb2, allow_nan=False)
