"""
PR 38 Step 2B: the G13 correction (``fold_label_excluded``), opt-in.

1. The versioned records load against pinned fingerprints and bind the frozen
   identities; the F31 clarification quotes the frozen wording exactly; an
   edited record is refused.
2. The provider: inner fold j's bins for EVERY row (training rows included)
   ignore fold j's labels, while the historical training-row bins read them;
   it refuses bins or labels that are not its own; it records fingerprints.
3. The routers consume it: GLASS fits (ranker, generation, selection) and
   routes inner fold j on bins[j]; the RF Router fits and scores fold j's OOF
   forest, and every forest t1_j is solved from, on bins[j]; neither stores
   the provider; a provider that returns the fit's own bins reproduces the
   historical fit exactly; GLASS ``full`` refuses one.
4. The runner and evidence: the pathway is a setting, not a config change;
   the evidence recounts each inner union on its fold's bins and checks them
   against the fit; the cross-fit, full-train fit and saving run on it.
5. The corrected G13 audit: the fit's inner bins are the audit's own
   recomputation and its inner fits reproduce from them; the correction's
   effect is recorded; a fit whose recorded bins differ is flagged.
6. Opt-in slow (real GLASS / CBC): the same on a real fold fit.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json

import numpy as np
import pandas as pd
import pytest

import glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector as ilp_mod
import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.evidence as E
import stage2_crossfit.g13 as G
import stage2_crossfit.inner_features as IF
import stage2_crossfit.runner as R
from blackbox_pipeline.models.rf_router import RFRouter
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.oof.scorer import OOFScorer
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.contract import glass_config_from_contract
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from glass_pipeline.glass_router.rf.feature_ranker import make_rf_feature_ranker_factory
from glass_pipeline.glass_router.rule_generator.rule_logger import RuleLogger
from shared.stage2.feature_engineering import engineer_features
from tests.support import pr38_world as W
from tests.support.rf_router_world import serial_forests

FOLD = IF.PATHWAY_FOLD


def _inner(y, k=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    out = np.full(len(y), -1)
    for j, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), np.asarray(y))):
        out[va] = j
    return out


@pytest.fixture(scope="module")
def side():
    X, y, X_va = W.synthetic_train_side().fold_rows(0)
    with contextlib.redirect_stdout(io.StringIO()):
        X_eng, _ = engineer_features(X, X_va, y)
    return X, y, X_eng, _inner(y)


@pytest.fixture(scope="module")
def fold_fits():
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())
    try:
        settings = dataclasses.replace(W.make_settings(), feature_pathway=FOLD)
        with serial_forests():
            res = R.fit_fold(W.synthetic_train_side(), settings, 0, keep_fits=True)
    finally:
        mp.undo()
    return res


# ======================================================================
# 1. the versioned records
# ======================================================================
def test_the_versioned_records_bind_the_frozen_identities(tmp_path, monkeypatch):
    spec = ep.load_frozen_specification()
    rec = IF.pathway_record()
    assert E.record_identity(rec) == ["stage2-inner-fold-feature-isolation", "1.0.0",
                                      E.FEATURE_PATHWAY_FINGERPRINT]
    assert rec["binds"]["contract"] == [spec.contract.contract_id,
                                        spec.contract.contract_version, spec.contract.fingerprint()]
    assert rec["binds"]["protocol"] == [spec.protocol.protocol_id, spec.protocol.version,
                                        spec.protocol.fingerprint()]
    assert rec["binds"]["glass_profile"] == list(
        glass_profile.load_glass_primary_profile().identity().values())
    clar = E.f31_r2_clarification()
    assert clar["status"] == "proposed" and clar["clarifies"]["protocol"] == rec["binds"]["protocol"]
    assert clar["frozen_wording"]["f31.measure"] == spec.protocol["f31"]["measure"]
    assert "inner-OOF" in clar["clarified_wording"]["f31.measure"]
    (tmp_path / E.FEATURE_PATHWAY_FILE).write_text(json.dumps({**rec, "status": "adopted"}))
    monkeypatch.setattr(E, "RECORDS_DIR", tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        IF.pathway_record()


def test_an_unknown_pathway_is_refused():
    with pytest.raises(ValueError, match="unknown feature pathway"):
        IF.pathway_identity("leave_one_out")
    assert IF.provider_factory(IF.PATHWAY_OUTER) is None
    assert IF.pathway_identity(IF.PATHWAY_OUTER) == {"pathway": IF.PATHWAY_OUTER, "record": None}


# ======================================================================
# 2. the provider
# ======================================================================
def test_inner_fold_bins_ignore_that_folds_labels_on_every_row(side):
    X, y, X_eng, plan = side
    rows = plan == 2
    y2 = y.copy()
    y2[rows] = 1 - y2[rows]                                   # every fold-2 label moves
    a = IF.FoldLabelExcludedFeatures(X, y)
    b = IF.FoldLabelExcludedFeatures(X, y2)
    fa = a(X_eng, y, plan)[2]
    fb = b(b.base_bins, y2, plan)[2]
    pd.testing.assert_frame_equal(fa.astype("int64"), fb)    # training rows included
    assert not a.base_bins[~rows].equals(b.base_bins[~rows])  # historical training bins do move
    expect = G.bin_source(G.fold_excluded_frames(a.engineer, X, np.asarray(y), plan)[2])
    pd.testing.assert_frame_equal(fa.astype("int64"), expect)


def test_the_provider_refuses_bins_or_labels_that_are_not_its_own(side):
    X, y, X_eng, plan = side
    prov = IF.FoldLabelExcludedFeatures(X, y)
    bad = X_eng.copy()
    bad.iloc[0] = 1 - bad.iloc[0]
    with pytest.raises(AssertionError, match="bins"):
        prov(bad, y, plan)
    with pytest.raises(AssertionError, match="labels"):
        prov(X_eng, 1 - y, plan)
    with pytest.raises(AssertionError, match="no frames"):
        prov.provenance(plan[::-1])


def test_the_provenance_records_fingerprints_not_rows(side):
    X, y, X_eng, plan = side
    prov = IF.FoldLabelExcludedFeatures(X, y)
    frames = prov(X_eng, y, plan)
    assert prov(X_eng, y, plan) is frames                     # built once per plan
    rec = prov.provenance(plan)
    assert rec["pathway"] == FOLD and rec["record"] == IF.pathway_identity(FOLD)["record"]
    assert rec["plan_fingerprint"] == IF.plan_fingerprint(plan)
    assert [f["bins_fingerprint"] for f in rec["inner_folds"]] == [
        IF.frame_fingerprint(frames[j]) for j in range(5)]
    assert [f["labels_excluded"] for f in rec["inner_folds"]] == [
        int((plan == j).sum()) for j in range(5)]
    assert len(json.dumps(E.to_json_safe(rec))) < 4000


# ======================================================================
# 3. the routers consume it
# ======================================================================
def _not_stored(obj):
    return not any(isinstance(v, IF.FoldLabelExcludedFeatures) for v in vars(obj).values())


def test_glass_fits_and_routes_each_inner_fold_on_its_bins(monkeypatch):
    cls = W.wired(monkeypatch, W.wired_glass_class())
    calls = []
    real = GlassRouterPipeline._inner_pass1_fit

    def spy(self, k, X_tr, y_tr, *a):
        calls.append((k, X_tr, y_tr))
        return real(self, k, X_tr, y_tr, *a)

    monkeypatch.setattr(cls, "_inner_pass1_fit", spy)
    settings = dataclasses.replace(W.make_settings(), feature_pathway=FOLD)
    with serial_forests():
        fit = R.fit_fold(W.synthetic_train_side(), settings, 0, arms=(E.ARM_GLASS,),
                         keep_fits=True)["fits"][E.ARM_GLASS]
    glass, prov = fit.glass, fit.inner_features
    inner = glass.inner_fold_id_
    frames = prov(fit.X_tr_eng, fit.y_tr, inner)
    assert [k for k, *_ in calls] == list(range(5))
    for k, X_tr, y_tr in calls:
        pd.testing.assert_frame_equal(X_tr, frames[k][inner != k])   # ranker, generation, ILP
        pd.testing.assert_series_equal(y_tr, fit.y_tr[inner != k])
    routed = ~np.asarray(glass.fit_report_["remainder_mask"], dtype=bool)
    for k, rules in enumerate(glass.inner_pass1_rules_):
        np.testing.assert_array_equal(routed[inner == k], GlassRouterPipeline._rules_mask(
            rules, frames[k][inner == k]))                           # eligibility
    assert glass.inner_feature_pathway_ == prov.provenance(inner) and _not_stored(glass)


def test_the_rf_router_fits_scores_and_solves_fold_j_on_its_bins(side, monkeypatch):
    X, y, X_eng, _ = side
    cfg = W.rf_test_config()
    prov = IF.FoldLabelExcludedFeatures(X, y)
    seen = {}
    real = OOFScorer.score

    def spy(self, X_, y_, subset_mask=None, label="OOF", fold_features=None):
        seen[label] = (X_, fold_features)
        return real(self, X_, y_, subset_mask=subset_mask, label=label,
                    fold_features=fold_features)

    monkeypatch.setattr(OOFScorer, "score", spy)
    with serial_forests():
        r = RFRouter(cfg).fit(X_eng, y, inner_features=prov)
    plan = r.oof_pass1_fold_id_
    frames = prov(X_eng, y, plan)
    assert seen["pass1 OOF"][1] is not None
    for j in range(5):
        assert seen[f"eligibility[{j}]"][0] is frames[j]          # every forest t1_j is solved from
    trainer = ForestTrainer(cfg.rf_params, cfg.random_state)
    for j in range(5):
        rows = plan == j
        p = ForestTrainer.positive_proba(trainer.fit(frames[j][~rows], pd.Series(
            np.asarray(y)[~rows])), frames[j][rows])
        assert np.max(np.abs(p - r.oof_pass1_proba_[rows])) <= 1e-12  # fold j's OOF forest
    assert r.inner_feature_pathway_ == prov.provenance(plan) and _not_stored(r)


class _SameBins:
    """Every inner fold gets the fit's own bins: the historical pathway, through the hook."""

    def __call__(self, X, y, fold_id):
        return {int(j): X for j in np.unique(fold_id) if j >= 0}

    def provenance(self, fold_id):
        return {"pathway": "same_bins"}


def test_the_fits_own_bins_through_the_hook_reproduce_the_historical_fit(side, monkeypatch):
    X, y, X_eng, _ = side
    cfg = W.rf_test_config()
    with serial_forests():
        hist = RFRouter(cfg).fit(X_eng, y)
        hook = RFRouter(cfg).fit(X_eng, y, inner_features=_SameBins())
    np.testing.assert_array_equal(hist.oof_pass1_proba_, hook.oof_pass1_proba_)
    np.testing.assert_array_equal(hist.remainder_mask_, hook.remainder_mask_)
    assert hist.thresholds_.to_dict() == hook.thresholds_.to_dict()
    assert hist.fit_report_.remainder == hook.fit_report_.remainder
    assert hist.inner_feature_pathway_ is None
    cls = W.wired(monkeypatch, W.wired_glass_class())
    config = W.make_settings().glass_config
    fits = []
    for kw in ({}, {"inner_features": _SameBins()}):
        g = cls(config, rf_model=None, rule_logger=RuleLogger(enabled=False),
                feature_ranker_factory=make_rf_feature_ranker_factory(
                    W.TEST_RANKER["rf_params"], 42))
        with contextlib.redirect_stdout(io.StringIO()), serial_forests():
            g.fit(X_eng, y, **kw)
        fits.append(g)
    np.testing.assert_array_equal(fits[0].fit_report_["remainder_mask"],
                                  fits[1].fit_report_["remainder_mask"])
    assert fits[0].inner_feature_pathway_ is None


def test_glass_full_and_the_rf_nested_check_refuse_inner_features(side):
    X, y, X_eng, _ = side
    g = GlassRouterPipeline(glass_profile.glass_s1_full_config(), rf_model=None,
                            rule_logger=RuleLogger(enabled=False))
    with pytest.raises(ValueError, match="oof_remainder"):
        g.fit(X_eng, y, inner_features=IF.FoldLabelExcludedFeatures(X, y))
    with pytest.raises(ValueError, match="validate_nested"):
        RFRouter(W.rf_test_config(validate_nested=True)).fit(
            X_eng, y, inner_features=IF.FoldLabelExcludedFeatures(X, y))


# ======================================================================
# 4. the runner and evidence
# ======================================================================
def test_the_pathway_is_a_setting_not_a_config_change():
    hist, fold = R.primary_settings(), R.primary_settings(FOLD)
    assert hist.glass_config.to_dict() == fold.glass_config.to_dict()
    assert hist.rf_config_factory().to_dict() == fold.rf_config_factory().to_dict()
    assert hist.checks["identities"] == fold.checks["identities"]
    assert fold.checks["feature_pathway"] == IF.pathway_identity(FOLD)
    assert hist.inner_features is None and fold.inner_features is IF.FoldLabelExcludedFeatures
    a, b = R.settings_record(hist), R.settings_record(fold)
    assert {k for k in a if a[k] != b[k]} == {"checks", "feature_pathway"}
    assert R.s1_settings(FOLD).glass_inner_features is None      # GLASS full: no inner folds


def test_the_corrected_evidence_recounts_each_inner_union_on_its_bins(fold_fits):
    for arm, ev in fold_fits["arms"].items():
        assert ev["inner_feature_pathway"]["pathway"] == FOLD, arm
        assert ev["isolation"]["inner_features_match_the_fit"] is True, arm
        assert ev["fit_status"]["accepted"] and not ev["fit_status"]["problems"], arm
    fit = fold_fits["fits"][E.ARM_GLASS]
    with pytest.raises(ValueError, match="provider"):
        E.glass_fold_evidence(fit.glass, fit.X_tr_eng, fit.y_tr, fold=0)
    router, rf_bins, y_rf = fold_fits["fits"][E.ARM_RF]
    with pytest.raises(ValueError, match="provider"):
        E.rf_fold_evidence(router, rf_bins, y_rf, fold=0)
    prov = copy.copy(fit.inner_features)
    inner = fit.glass.inner_fold_id_
    other = {j: f.copy() for j, f in prov(fit.X_tr_eng, fit.y_tr, inner).items()}
    other[0].iloc[:, 0] = 1 - other[0].iloc[:, 0]
    prov._frames = {np.asarray(inner).astype(int).tobytes(): other}
    ev = E.glass_fold_evidence(fit.glass, fit.X_tr_eng, fit.y_tr, fold=0, inner_features=prov)
    assert ev["isolation"]["inner_features_match_the_fit"] is False
    g3 = R.preflight_checks({"arms": {E.ARM_GLASS: {**ev, "fit_status": fold_fits["arms"][
        E.ARM_GLASS]["fit_status"]}}}, 50, protocol=ep.load_frozen_protocol())
    assert g3["arms"][E.ARM_GLASS]["checks"]["isolation_clean"] is False and not g3["feasible"]


def test_the_crossfit_full_train_and_saving_run_on_the_corrected_pathway(monkeypatch, tmp_path):
    W.wired(monkeypatch, W.wired_glass_class())
    ts = W.synthetic_train_side()
    settings = dataclasses.replace(W.make_settings(), feature_pathway=FOLD)
    with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
        res = R.run_crossfit(ts, settings)
        R.fit_full_train(ts, settings, res)
        info = R.save_run(res, ts, settings, tmp_path / "run", "t1")
    for arm, evs in res.evidence.items():
        assert [ev["inner_feature_pathway"]["pathway"] for ev in evs] == [FOLD] * 5, arm
        assert all(ev["fit_status"]["accepted"] for ev in evs), arm
    for arm, ev in res.full_train_evidence.items():
        assert ev["inner_feature_pathway"]["pathway"] == FOLD and ev["fit_status"]["accepted"]
    man = json.loads((info["root"] / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["feature_pathway"] == IF.pathway_identity(FOLD)
    assert _not_stored(res.glass_full["glass"]) and _not_stored(res.rf_full["router"])


# ======================================================================
# 5. the corrected G13 audit
# ======================================================================
def _audit(fits, **override):
    X_tr, y_tr, _ = W.synthetic_train_side().fold_rows(0)
    g = fits[E.ARM_GLASS]
    router, rf_bins, _ = fits[E.ARM_RF]
    kw = dict(glass=g.glass, glass_bins=g.X_tr_eng, rf_router=router, rf_bins=rf_bins)
    kw.update(override)
    with serial_forests():
        return G.audit_fold(X_tr, y_tr, fold=0, **kw)


def test_a_corrected_fit_reproduces_from_the_audits_own_label_excluded_bins(fold_fits):
    rec = _audit(fold_fits["fits"])
    assert rec["format"] == G.G13_FORMAT
    for arm in ("glass", "rf"):
        assert rec["pathways"][arm]["pathway"] == FOLD
        assert rec["isolation"][arm] == {"pathway_match": True, "reproduced_all": True,
                                         "complete": True}
        assert "not_applicable" in rec["eligibility"][arm]
        r = rec["eligibility_retrained"][arm]
        assert (r["stored_features"], r["alt_features"]) == (FOLD, IF.PATHWAY_OUTER)
        assert isinstance(r["rows_changed"], int)                 # the correction's effect
    rf = rec["eligibility_retrained"]["rf"]
    assert rf["score_reproduction_max_difference"] <= 1e-9
    assert rf["t1_reproduction_max_difference"] <= 1e-9
    assert rec["materiality_inputs"]["rf_feature_pathway"] == FOLD and rec["uses_test_rows"] is False


def test_a_corrected_fit_whose_recorded_bins_differ_is_flagged(fold_fits):
    router, rf_bins, y = fold_fits["fits"][E.ARM_RF]
    tampered = copy.deepcopy(router)
    tampered.inner_feature_pathway_["inner_folds"][0]["bins_fingerprint"] = "0" * 16
    rec = _audit(fold_fits["fits"], glass=None, glass_bins=None, rf_router=tampered)
    assert rec["isolation"]["rf"] == {"pathway_match": False, "reproduced_all": False,
                                      "complete": False}
    assert "glass" not in rec["eligibility_retrained"]


# ======================================================================
# 6. opt-in slow: a real GLASS fold on the corrected pathway
# ======================================================================
@pytest.mark.slow
def test_a_real_corrected_fold_is_recounted_and_reproduces_from_its_bins():
    gs = W.split_frame(W.make_bank_frame(n=3000, seed=7, positive_rate=0.12), test_size=0.2)
    ts = R.make_train_side(gs["X_train"], gs["y_train"].astype(int), gs["X_test"])
    base = dataclasses.replace(
        glass_profile.glass_base_config(glass_profile.load_glass_primary_profile()),
        min_support_pass1=40, min_support_pass2=15)
    glass = dataclasses.replace(glass_config_from_contract(base, W.LOOSE), remainder_min_size=100)
    settings = dataclasses.replace(W.make_settings(glass_config=glass), feature_pathway=FOLD)
    with pytest.MonkeyPatch.context() as mp, serial_forests():
        mp.setattr(ilp_mod.ILPRuleSelector, "TIME_LIMIT_PER_SOLVE", 30)
        res = R.fit_fold(ts, settings, 0, keep_fits=True)
        if any("stop" in ev for ev in res["arms"].values()):
            pytest.skip("a synthetic fold stopped; recorded, as G8 requires")
        ev = res["arms"][E.ARM_GLASS]
        reports = ev["selection_reports"]["inner_pass1"]
        for j, rep in enumerate(reports):                    # recounted on fold j's bins
            c = ev["counts"][f"inner_pass1[{j}]"]
            assert (c["routed_rows"], c["routed_rows"] - c["routed_positives"]) == (
                rep["union_rows"], rep["union_good"])
        X_tr, y_tr, _ = ts.fold_rows(0)
        g = res["fits"][E.ARM_GLASS]
        router, rf_bins, _ = res["fits"][E.ARM_RF]
        rec = G.audit_fold(X_tr, y_tr, fold=0, glass=g.glass, glass_bins=g.X_tr_eng,
                           rf_router=router, rf_bins=rf_bins)
    assert rec["isolation"]["rf"]["pathway_match"] and rec["isolation"]["glass"]["pathway_match"]
    for f in rec["eligibility_retrained"]["glass"]["folds"]:
        if f["base_reselection"]["lexicographic_guarantee"] == "proven":
            assert f["reproduced"], f                  # a proven optimum re-runs to itself
