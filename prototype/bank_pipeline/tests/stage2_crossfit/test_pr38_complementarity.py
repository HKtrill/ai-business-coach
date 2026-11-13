"""
PR 38 Step 3B: the comparative analysis of an accepted corrected primary
(``stage2_crossfit.complementarity``).

1. The gate: without an official structural acceptance that re-verifies in
   full nothing is read (no train side, no artifact, no baseline source); a
   missing, failed or edited acceptance or a changed candidate is refused. An
   accepted synthetic run passes the gate, and its real ``run_primary``
   evidence gives the fold ledger and the row / fold alignment; an upstream
   artifact that is not the protocol's frozen one is refused.
2. The analysis (``analyse``, on the Step 2A-2D synthetic world): three-state
   accounting; D-C1 statements; the registered bootstrap; section 11 keeps fit
   status and observation apart; pre/post pairs by row identity and takes the
   pre values from the frozen tables; R3; misaligned rows, references or
   evidence are refused; an incomplete ledger is refused.
3. The package: deterministic, written once (identical rerun writes nothing, a
   differing one is refused), declared sensitivities reported as a gap, and a
   changed acceptance after the gate stops the write. The command line.

Every input is synthetic: no primary training, no notebook, no real artifact.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.acceptance as A
import stage2_crossfit.complementarity as C
import stage2_crossfit.runner as R
from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from shared.stage_io import label_fingerprint
from tests.stage2.test_stage2_operating_contract_v2 import _v2
from tests.stage2_crossfit.test_pr38_acceptance import _accept, _build, _failed
from tests.support import diagnostics_world as dw
from tests.support import stage2_crossfit_world as W

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
ARMS = ("glass", "blackbox")


# ======================================================================
# a synthetic world for the analysis: the Step 2A-2D views, fit-time evidence
# aligned with their rows, a pre-correction routing and its frozen tables
# ======================================================================
def _counts(n, pos, routed, routed_pos):
    return {"population_rows": n, "population_positives": pos, "routed_rows": routed,
            "routed_positives": routed_pos}


def _evidence(trace):
    """Evidence in the runner's format, feasible under ``W.LOOSE``, for every fold of both arms."""
    idx, y, fold = trace.index, np.asarray(trace.y).astype(int), np.asarray(trace.fold_id)
    rep = {"source": "ilp", "lexicographic_guarantee": "proven", "selected_rule_ids": [7]}
    out = {a: [] for a in ARMS}
    for k in range(5):
        fit = fold != k
        n, pos = int(fit.sum()), int(y[fit].sum())
        rem, rem_y = idx[fit][: n // 2], y[fit][: n // 2]
        c1, c2 = _counts(n, pos, 300, 10), _counts(len(rem), int(rem_y.sum()), 100, 40)
        base = {"fold": k, "n_train": n, "train_positives": pos,
                "train_label_fingerprint": label_fingerprint(idx[fit], y[fit]),
                "remainder": {"rows": len(rem), "positives": int(rem_y.sum()),
                              "prevalence": float(rem_y.mean()), "index": list(rem)},
                "wall_seconds": 1.5}
        out["glass"].append({
            **base, "arm": "glass",
            "selection_reports": {"pass1": rep, "pass2": dict(rep, selected_rule_ids=[8]),
                                  "inner_pass1": [dict(rep) for _ in range(5)]},
            "counts": {"pass1": c1, "pass2": c2, **{f"inner_pass1[{j}]": c1 for j in range(5)}},
            "m6_inner_oof_pass1": {**_counts(n, pos, 400, 20), "npv": 0.95,
                                   "leakage_share": 20 / pos}})
        out["blackbox"].append({
            **base, "arm": "blackbox",
            "counts": {"pass1": c1, "pass2": c2,
                       **{f"eligibility_t1[{j}]": c1 for j in range(5)}},
            "thresholds": {"t1": 0.2, "t2": 0.5},
            "r1_bands": {"pass1_confidence": 290 / 300, "pass2_confidence": 0.4},
            "r2_strips": {"strip_a_in_band": {"rows": 6 if k == 2 else 3, "positives": 1},
                          "band_rows": 100}})
    return {"folds": out}


def _historical(view, seed):
    """A pre-correction routing of the same rows: 10% of the decisions changed consistently."""
    rng = np.random.default_rng(seed)
    tok = np.array(view.source_decision, dtype=object, copy=True)
    flip = rng.random(len(tok)) < 0.1
    tok[flip] = np.array(["pass1", "pass2", "uncertain"], dtype=object)[
        rng.integers(0, 3, int(flip.sum()))]
    st = np.array([d.SOURCE_DECISIONS[t][0] for t in tok], dtype=object)
    pred = np.array([d.SOURCE_DECISIONS[t][1] for t in tok]).astype(
        np.asarray(view.source_pred).dtype)
    return dataclasses.replace(view, state=st, source_decision=tok, source_pred=pred)


@pytest.fixture(scope="module")
def synth():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    trace = d.build_router_trace(glass, rf, ref)
    settings = W.make_settings()
    evidence = _evidence(trace)
    ledger = C.fold_ledger(evidence, settings)
    hist = {"glass": _historical(glass, 1), "blackbox": _historical(rf, 2)}
    ht = C.diagnostics_tables(hist["glass"], hist["blackbox"], ref, lr, ebm)
    frozen = {k: C._read_text_csv(b._csv_bytes(ht[k])) for k in ("states", "ratios")}
    inputs = C.Inputs(
        "synthetic", "test", settings.protocol, settings.contract, ref, trace, lr, ebm,
        ep.ReferenceFlags.from_upstream(d.primary_reference_flags(lr, ebm)["union"]), hist,
        frozen, ledger, evidence, C.check_alignment(trace, evidence, ledger.attempted),
        {"run": {"run_id": "synthetic"}})
    return SimpleNamespace(inputs=inputs, settings=settings, analysis=C.analyse(inputs))


# ======================================================================
# 1. the gate
# ======================================================================
@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    """An official acceptance of a synthetic primary (test_pr38_acceptance.accepted_world)."""
    frozen_ids = R.primary_settings().checks["identities"]
    settings = dataclasses.replace(W.make_settings(role=R.ROLE_PRIMARY),
                                   checks={"identities": frozen_ids})
    w = _build(tmp_path_factory.mktemp("step3b"), settings, "r3b", R.ROLE_PRIMARY)
    mp = pytest.MonkeyPatch()
    _population_doubles(mp)
    try:
        rec = _accept(w)
    finally:
        mp.undo()
    assert rec["accepted"], {g: _failed(rec, g) for g in rec["gates"]}
    A.write_acceptance(w["records"], rec)
    return w


def _population_doubles(mp):
    """The two population checks synthetic rows cannot pass (as accepted_world doubles them)."""
    mp.setattr(R, "check_population", lambda ts, protocol: {"passed": True, "checks": {}})
    mp.setattr(ep, "check_population_identity", lambda trace, identity: ep.IdentityReport(
        (("synthetic rows (test double)", True, "stub"),)))


@pytest.fixture
def gate_env(monkeypatch):
    """This run's authorization re-verifies; reading data or artifacts is recorded."""
    def verify(path, settings=None, *, run_id=None):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("run_id") != run_id:
            raise R.RunnerError("authorization belongs to another run")
        return rec
    monkeypatch.setattr(R, "require_authorization", verify)
    reads = []
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("load_train_side"))
    return reads


def _copy(w, tmp_path):
    base = tmp_path / "c"
    shutil.copytree(w["base"] / "models", base / "models")
    shutil.copytree(w["base"] / "records", base / "records")
    return base


def _load(w, base, reads, **kw):
    def train_side():
        reads.append("train_side")
        return w["ts"]

    def historical(sums):
        reads.append("historical")
        raise _Stop()
    kw = {"train_side": train_side, "historical": historical, **kw}
    return C.load_inputs(w["run"], "sums.txt", output_root=base / "models",
                         record_dir=base / "records" / w["run"], settings=w["settings"], **kw)


class _Stop(Exception):
    pass


def _refused(w, base, reads, monkeypatch, match):
    monkeypatch.setattr(A, "candidate_views", lambda *a: pytest.fail("an artifact was read"))
    monkeypatch.setattr(A.joblib, "load", lambda *a, **k: pytest.fail("an artifact was loaded"))
    with pytest.raises(R.RunnerError, match=match):
        _load(w, base, reads)
    assert reads == []


def test_without_an_acceptance_record_nothing_is_read(accepted, tmp_path, gate_env, monkeypatch):
    base = _copy(accepted, tmp_path)
    (base / "records" / accepted["run"] / A.ACCEPTANCE_FILE).unlink()
    _refused(accepted, base, gate_env, monkeypatch, "no structural acceptance record")


def test_a_failed_acceptance_is_never_read(accepted, tmp_path, gate_env, monkeypatch):
    base = _copy(accepted, tmp_path)
    records = base / "records" / accepted["run"]
    (records / A.ACCEPTANCE_FILE).unlink()
    cand = A.load_candidate(accepted["run"], output_root=base / "models", record_dir=records)
    with monkeypatch.context() as mp:
        _population_doubles(mp)
        rec = _accept(accepted, cand, upstream=lambda trace, ref: ep.IdentityReport(
            (("upstream", False, "differs"),)))
    assert not rec["accepted"]
    assert A.write_acceptance(records, rec).name.startswith("structural_acceptance_failed_")
    _refused(accepted, base, gate_env, monkeypatch, "no structural acceptance record")


def test_an_edited_acceptance_record_is_refused(accepted, tmp_path, gate_env, monkeypatch):
    base = _copy(accepted, tmp_path)
    path = base / "records" / accepted["run"] / A.ACCEPTANCE_FILE
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["gates"]["V3"]["checks"] = rec["gates"]["V3"]["checks"][:1]
    path.write_text(json.dumps(rec), encoding="utf-8")
    _refused(accepted, base, gate_env, monkeypatch, "does not establish an official")


def test_a_changed_candidate_is_refused(accepted, tmp_path, gate_env, monkeypatch):
    base = _copy(accepted, tmp_path)
    (base / "models" / accepted["run"] / R.ROLE_PRIMARY / "extra.txt").write_text("x")
    _refused(accepted, base, gate_env, monkeypatch, "fails its integrity checks")


def test_the_public_entry_point_cannot_bypass_structural_acceptance(accepted, synth, tmp_path,
                                                                    gate_env, monkeypatch):
    """``run`` has no ready-made inputs override: with a candidate, its data and complete
    synthetic inputs at hand, an unaccepted run is refused and nothing is analysed or written."""
    base = _copy(accepted, tmp_path)
    records = base / "records" / accepted["run"]
    (records / A.ACCEPTANCE_FILE).unlink()
    monkeypatch.setattr(C, "analyse", lambda inp: pytest.fail("analysed without acceptance"))
    kw = dict(output_root=base / "models", record_dir=records, settings=accepted["settings"],
              train_side=lambda: accepted["ts"],
              historical=lambda sums: pytest.fail("baseline sources read"))
    with pytest.raises(R.RunnerError, match="no structural acceptance record"):
        C.run(accepted["run"], "sums.txt", **kw)
    with pytest.raises(TypeError, match="inputs"):
        C.run(accepted["run"], "sums.txt", inputs=synth.inputs, **kw)
    assert not (records / C.PACKAGE_DIR).exists()


def test_an_accepted_run_passes_the_gate_then_loads_in_order(accepted, gate_env, monkeypatch):
    """Gate, train side, views and trace, ledger and alignment from the real evidence, then the
    baseline sources (stopped there: synthetic rows have no frozen upstream)."""
    _population_doubles(monkeypatch)
    seen = {}
    real = C.check_alignment

    def alignment(trace, evidence, attempted):
        seen["alignment"] = real(trace, evidence, attempted)
        return seen["alignment"]
    monkeypatch.setattr(C, "check_alignment", alignment)
    with pytest.raises(_Stop):
        _load(accepted, accepted["base"], gate_env)
    assert gate_env == ["train_side", "historical"]
    frame = seen["alignment"]
    assert frame["passed"].all() and len(frame) == 1 + 2 * (1 + 2 * 5)


def test_an_upstream_artifact_that_is_not_the_protocols_is_refused(accepted, gate_env,
                                                                   monkeypatch):
    _population_doubles(monkeypatch)
    arts = accepted["settings"].protocol["upstream"]["artifacts"]
    sources = SimpleNamespace(artifacts=tuple(
        b.SourceArtifact(a["role"], a["path"], ("0" * 64 if k == "lr" else a["sha256"]), 1)
        for k, a in arts.items()))
    with pytest.raises(C.ComplementarityError, match="not the protocol's frozen upstream"):
        _load(accepted, accepted["base"], gate_env,
              historical=lambda sums: (sources, SimpleNamespace(status="identical")))


@pytest.mark.parametrize("changed, status, error", [
    ([], "identical", None), ([], "reproduced_with_code_drift", None),
    (["stage2.rf_router: changed"], "identical", "changed during the rebuild"),
    ([], "differs", "does not reproduce"),
])
def test_historical_sources_follow_the_archive_check(monkeypatch, changed, status, error):
    """The archive check's sequence: frozen arbiters, rebuild, re-hash, compare (stubbed I/O)."""
    from tests.support import archived_artifacts as aa
    seen = {}
    sources = SimpleNamespace(artifacts=())

    def real(*, sums_path, arbiters):
        seen["arbiters"] = arbiters
        return SimpleNamespace(archive="archive")
    monkeypatch.setattr(aa, "HistoricalReal", real)
    monkeypatch.setattr(aa, "collect_archived_baseline_sources", lambda r: sources)
    monkeypatch.setattr(b, "build_baseline", lambda src, code, *, baseline_id: (src, baseline_id))
    monkeypatch.setattr(aa, "archived_rehash", lambda archive: lambda src: changed)
    report = b.ComparisonReport(status, b.VerificationReport("frozen", True, ()))
    monkeypatch.setattr(b, "compare_baseline", lambda pkg, directory: report)
    if error:
        with pytest.raises(C.ComplementarityError, match=error):
            C.historical_sources("sums.txt")
    else:
        assert C.historical_sources("sums.txt") == (sources, report)
    assert set(seen["arbiters"]) == set(aa.baseline_arbiters())


def test_the_frozen_baseline_tables_are_read_only_from_the_pinned_package(synth, tmp_path):
    protocol = synth.settings.protocol
    frozen = C._frozen_tables(A.phase0.BASELINE_DIR, protocol)
    assert list(frozen["ratios"].columns) == list(b._LONG_COLUMNS)
    assert {"arm", "fold", "state", "n"} <= set(frozen["states"].columns)
    shutil.copytree(A.phase0.BASELINE_DIR, tmp_path / "pkg")
    (tmp_path / "pkg" / "tables" / "states.csv").write_bytes(b"edited\n")
    with pytest.raises(C.ComplementarityError, match="fails verification"):
        C._frozen_tables(tmp_path / "pkg", protocol)


# ======================================================================
# 2. the analysis
# ======================================================================
def test_three_state_accounting_is_exclusive_exhaustive_and_reconciled(synth):
    t = synth.analysis.tables["accounting"]
    assert set(t["routing"]) == {"glass", "blackbox", "glass_pre_correction",
                                 "blackbox_pre_correction"}
    assert t[["exclusive", "exhaustive", "reconciles_with_routing_counts"]].all(axis=None)
    trace = synth.inputs.trace
    for arm in ARMS:
        rows = t[(t["routing"] == arm) & (t["fold"] != "all")]
        pooled = t[(t["routing"] == arm) & (t["fold"] == "all")].iloc[0]
        st = np.asarray(trace.views[arm].state)
        assert pooled["pass1_rows"] == int((st == P1).sum()) == rows["pass1_rows"].sum()
        assert pooled["abstain_positives"] == int(np.asarray(trace.y)[st == AB].sum())


@pytest.mark.parametrize("row, statement", [
    ({"interval_status": "defined", "difference_low": 0.01, "difference_high": 0.2},
     "GLASS higher than RF"),
    ({"interval_status": "defined", "difference_low": -0.2, "difference_high": -0.01},
     "GLASS lower than RF"),
    ({"interval_status": "defined", "difference_low": 0.0, "difference_high": 0.2},
     "not resolved at this sample"),
    ({"interval_status": "defined", "difference_low": -0.1, "difference_high": 0.1},
     "not resolved at this sample"),
    ({"interval_status": "undefined_empty_denominator", "difference_low": None,
      "difference_high": None}, "not resolved at this sample (interval undefined)"),
    ({"interval_status": "defined", "difference_low": float("nan"), "difference_high": 0.1},
     "not resolved at this sample (interval undefined)"),
])
def test_the_d_c1_statement_rule(row, statement):
    assert C._statement(row) == statement


def test_primary_outcomes_are_the_registered_run_in_protocol_order(synth):
    an, protocol = synth.analysis, synth.settings.protocol
    t = an.tables["primary_outcomes"]
    ids = [o["id"] for o in protocol["hierarchy"]["primary_outcomes"]]
    assert list(t["outcome"][:len(ids)]) == ids
    assert list(t["outcome"][len(ids):]) == ["descriptive_total"] * 2
    run = an.bootstrap["cross_arm"]
    assert run.verify() and run.provenance["procedure_id"] == ep.BOOTSTRAP_PROCEDURE_ID
    assert (run.provenance["replicates"], run.provenance["seed"]) == (2000, 42)
    assert run.provenance["reference_fingerprint"] == synth.inputs.reference.fingerprint
    c = ep.PairedRouting.from_trace(synth.inputs.trace).counts("glass")
    row = t.set_index("metric").loc["pass1_rate"]
    assert row["glass"] == pytest.approx(c.pass1_rows / c.rows)
    assert all(s == C._statement(r) for r, s in zip(t.to_dict("records"), t["statement"])
               if r["outcome"] in ids)
    folds = an.tables["primary_outcomes_by_fold"]
    assert sorted(set(folds["fold"])) == [0, 1, 2, 3, 4]


def test_the_analysis_is_reproducible(synth):
    again = C.analyse(synth.inputs)
    assert again.bootstrap["cross_arm"].digest == synth.analysis.bootstrap["cross_arm"].digest
    assert C.build_package(synth.inputs, again) == C.build_package(synth.inputs, synth.analysis)


def test_fit_time_feasibility_and_observed_outer_oof_compliance_stay_apart(synth):
    strict = _v2(min_pass1_npv=0.999, max_pass1_leakage_rate=0.001, min_pass2_precision=0.99,
                 min_pass1_support=20, min_pass2_support=10)
    an = C.analyse(dataclasses.replace(synth.inputs, contract=strict))
    sf = an.tables["shortfalls"]
    assert set(sf["fit_status"]) == {"valid_optimal"}
    folds = sf[sf["fold"] != "all"]
    assert (folds["fit_time_status"] == "satisfied").all()
    violated = sf[sf["observed_status"] == "violated"]
    assert len(violated) and (violated["classification"] == "valid_fit_outer_oof_shortfall").all()
    assert violated["material"].notna().all()
    assert set(sf["fold"]) == {0, 1, 2, 3, 4, "all"}
    assert (sf[sf["fold"] == "all"]["pool"] == "all attempted folds").all()
    layer = an.layers["3_observed_outer_oof_compliance"]
    assert layer["glass/pooled"]["observed_status"].get("violated", 0) > 0
    assert an.layers["2_fit_time_feasibility_and_acceptance"]["fit_time_obligations_satisfied"]


def test_an_incomplete_fold_ledger_is_refused(synth):
    led = synth.inputs.ledger
    partial = ep.FoldLedger(led.attempted, {"glass": led.status["glass"],
                                            "blackbox": {**led.status["blackbox"],
                                                         3: "fit_time_invalid"}},
                            {"glass": led.fits["glass"],
                             "blackbox": {**led.fits["blackbox"], 3: None}})
    with pytest.raises(ep.ProtocolError, match="blackbox fold 3: fit_time_invalid"):
        C.analyse(dataclasses.replace(synth.inputs, ledger=partial))


def test_pre_post_pairs_rows_and_takes_pre_values_from_the_frozen_tables(synth):
    t, inp = synth.analysis.tables, synth.inputs
    tr = t["pre_post_transitions"]
    for arm in ARMS:
        pooled = tr[(tr["arm"] == arm) & (tr["fold"] == "all")]
        pre = np.asarray(inp.historical[arm].state)
        post = np.asarray(inp.trace.views[arm].state)
        assert pooled["rows"].sum() == inp.trace.n
        assert pooled["changed_rows"].iloc[0] == int((pre != post).sum()) > 0
    unc = t["pre_post_uncertainty"]
    assert set(unc["difference_definition"]) == {"pre - post"}
    assert set(unc["metric"]) == set(C.PRE_POST_METRICS)
    edited = {k: v.copy() for k, v in inp.baseline_tables.items()}
    edited["states"].loc[0, "n"] = "999999"
    states = C.analyse(dataclasses.replace(inp, baseline_tables=edited)).tables["pre_post_states"]
    assert "999999" in set(states["n_pre"])


def test_misaligned_rows_references_or_evidence_are_refused(synth):
    inp = synth.inputs
    glass = inp.historical["glass"]
    perm = np.random.default_rng(0).permutation(glass.n)
    shuffled = dataclasses.replace(glass, index=glass.index[perm])
    with pytest.raises(ep.ProtocolError, match="row identities differ"):
        C.analyse(dataclasses.replace(inp, historical={**inp.historical, "glass": shuffled}))
    ref = inp.reference
    moved = ep.ReferenceFlags.build(ref.name, ref.index[perm], ref.y[perm], ref.flags[perm],
                                    ref.selection)
    with pytest.raises(ep.ProtocolError, match="never aligned by position"):
        C.analyse(dataclasses.replace(inp, reference=moved))
    ev = json.loads(json.dumps(inp.evidence, default=str))
    ev["folds"]["blackbox"][2]["n_train"] += 1
    with pytest.raises(C.ComplementarityError, match="blackbox fold 2"):
        C.check_alignment(inp.trace, ev, inp.ledger.attempted)
    ev = json.loads(json.dumps(inp.evidence, default=str))
    fold = np.asarray(inp.trace.fold_id)
    ev["folds"]["glass"][1]["remainder"]["index"][0] = inp.trace.index[fold == 1][0]
    with pytest.raises(C.ComplementarityError, match="remainder rows inside"):
        C.check_alignment(inp.trace, ev, inp.ledger.attempted)


def test_r3_compares_each_rf_band_with_its_realised_outer_fold(synth):
    t = synth.analysis.tables["f31_r3_optimism"]
    trace = synth.inputs.trace
    st, y, fold = (np.asarray(trace.views["blackbox"].state), np.asarray(trace.y),
                   np.asarray(trace.fold_id))
    row = t[(t["fold"] == 1) & (t["metric"] == "pass1_npv")].iloc[0]
    m = (fold == 1) & (st == P1)
    assert (row["band_numerator"], row["band_denominator"]) == (290, 300)
    assert (row["realised_numerator"], row["realised_denominator"]) == (int((y[m] == 0).sum()),
                                                                      int(m.sum()))
    assert row["optimism_gap"] == pytest.approx(290 / 300 - (y[m] == 0).mean())
    pooled = t[(t["fold"] == "all") & (t["metric"] == "pass2_precision")].iloc[0]
    assert pooled["band_denominator"] == 500
    assert pooled["realised_denominator"] == int((st == P2).sum())
    strips = synth.analysis.tables["f31_r2_strips"]
    assert list(strips[strips["review_required"]]["fold"]) == [2]


def test_routing_policy_claims_keep_their_evidence_status(synth):
    audit = synth.analysis.tables["diagnostics/policy_audit"].set_index(["arm", "claim"])
    assert audit.loc[("glass", "decision.pass1_precedence"), "status"] == "code_level_only"
    assert audit.loc[("blackbox", "decision.pass1_precedence"), "status"] == "not_verifiable"


# ======================================================================
# 3. the package and the command line
# ======================================================================
def test_the_package_lists_every_section_and_reports_the_pending_sensitivity(synth):
    files = C.build_package(synth.inputs, synth.analysis)
    man = json.loads(files[C.MANIFEST])
    assert files[C.SIDECAR].decode().strip() == __import__("hashlib").sha256(
        files[C.MANIFEST]).hexdigest()
    protocol = synth.settings.protocol
    required = protocol["acceptance"]["comparative_sections"]
    assert set(man["comparative_sections"]) | set(man["comparative_section_gaps"]) == set(required)
    assert list(man["comparative_section_gaps"]) == ["declared_sensitivities"]
    for paths in man["comparative_sections"].values():
        assert all(p in files for p in paths)
    assert set(man["outputs"]) == {p for p in files if p.startswith("tables/")}
    assert man["layers"]["4_empirical_performance_and_complementarity"]["overall_verdict"] \
        .startswith("none")
    assert man["bootstrap"]["cross_arm"]["verified"] is True
    text = b"".join(files.values()).decode("utf-8")
    assert "/tmp/" not in text and "/home/" not in text


def test_written_once_identical_rerun_writes_nothing_and_a_difference_is_refused(synth,
                                                                                tmp_path):
    files = C.build_package(synth.inputs, synth.analysis)
    target = tmp_path / C.PACKAGE_DIR
    assert C.write_package(target, files) == "written"
    assert C.compare_package(target, files) == ("identical", [])
    assert C.write_package(target, files) == "identical"
    changed = dict(files)
    changed["tables/accounting.csv"] = files["tables/accounting.csv"] + b"x\n"
    with pytest.raises(C.ComplementarityConflict, match="content differs: tables/accounting"):
        C.write_package(target, changed)
    assert (target / "tables" / "accounting.csv").read_bytes() == files["tables/accounting.csv"]
    drift = json.loads(files[C.MANIFEST])
    drift["code"]["files"]["stage2_crossfit/complementarity.py"] = "0" * 64
    drifted = dict(files, **{C.MANIFEST: b._json_bytes(drift)})
    assert C.compare_package(target, drifted)[0] == "reproduced_with_code_drift"
    assert not list(tmp_path.glob(".*partial"))


def test_a_changed_acceptance_after_the_gate_stops_the_write(synth, tmp_path, monkeypatch):
    record = tmp_path / A.ACCEPTANCE_FILE
    record.write_text("{}")
    inp = dataclasses.replace(synth.inputs, record_dir=tmp_path, gate_hashes={
        "acceptance": (str(record), R.sha256_file(record))})
    monkeypatch.setattr(C, "load_inputs", lambda run_id, sums, **k: inp)
    record.write_text('{"edited": true}')
    with pytest.raises(C.ComplementarityError, match="acceptance changed after"):
        C.run("synthetic", "sums.txt")
    assert not (tmp_path / C.PACKAGE_DIR).exists()


def test_the_command_refuses_before_any_data_and_writes_then_checks(synth, tmp_path,
                                                                     monkeypatch):
    import stage2_crossfit.__main__ as cli
    import stage2_crossfit.phase0 as P
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("data loaded"))
    (tmp_path / "logs" / "r9").mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = cli.main(["complementarity", "--run-id", "r9", "--sums", "s"])
    assert code == 1 and "REFUSED" in out.getvalue() and "no structural acceptance" in \
        out.getvalue()

    inp = dataclasses.replace(synth.inputs, record_dir=tmp_path / "logs" / "r9")
    monkeypatch.setattr(C, "load_inputs", lambda run_id, sums, **k: inp)
    for argv, status in ((["--check"], "DIFFERS"), ([], "WRITTEN"), (["--check"], "IDENTICAL"),
                         ([], "IDENTICAL")):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main(["complementarity", "--run-id", "r9", "--sums", "s", *argv])
        assert f"complementarity package {status}" in out.getvalue()
        assert code == (1 if status == "DIFFERS" else 0)
    assert "O1" in out.getvalue() and "declared_sensitivities" in out.getvalue()
    manifest = json.loads((tmp_path / "logs" / "r9" / C.PACKAGE_DIR / C.MANIFEST).read_text(
        encoding="utf-8"))
    assert manifest["format"] == C.FORMAT and pd.notna(manifest["outputs"])
