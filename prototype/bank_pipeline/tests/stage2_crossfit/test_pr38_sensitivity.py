"""
PR 38 Step 3C: the G12 sensitivity S1 and its mandatory G6
(``stage2_crossfit.sensitivity``; F17 decision record, sections 6 and 8).

1. Evidence and fit status: every GLASS fit records its deployed Pass 1 rules;
   an S1 fold's completeness is Pass 1 and Pass 2 only, each selection
   classified by the frozen per-selection rules; the runner dispatches by role.
2. The S1 run (a scripted synthetic primary, accepted): refused before any
   data without the primary's acceptance, and before any fit unless S1 is the
   primary with ``pass2_population`` alone changed; once per run; the primary's
   artifacts and records are untouched.
3. S1 acceptance: G6 passes and S1 is accepted; a Pass 1 difference fails G6,
   is recorded explicitly, and nothing reports it; an edited acceptance or a
   changed candidate is refused; the report loads through both gates in order.
4. The S1 report (``analyse_s1`` on the Step 2A-2D synthetic world): single
   factor, routing transitions, within-S1 and baseline differences kept
   apart, a deterministic write-once package, the combined comparative report.

Every input is synthetic: no real-data fit, notebook or artifact.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
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
import stage2_crossfit.evidence as E
import stage2_crossfit.runner as R
import stage2_crossfit.sensitivity as S
from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from tests.stage2_crossfit.test_pr38_acceptance import _accept, _build, _failed
from tests.stage2_crossfit.test_pr38_complementarity import (
    _evidence, _historical, _population_doubles,
)
from tests.support import diagnostics_world as dw
from tests.support import stage2_crossfit_world as W
from tests.support.rf_router_world import serial_forests

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN


# ======================================================================
# worlds
# ======================================================================
def _authorization_double(mp):
    def verify(path, settings=None, *, run_id=None):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("run_id") != run_id:
            raise R.RunnerError("authorization belongs to another run")
        return rec
    mp.setattr(R, "require_authorization", verify)


def _s1_settings(w, **changes):
    s = w["settings"]
    glass = dataclasses.replace(s.glass_config, **{"pass2_population": "full", **changes})
    return dataclasses.replace(s, role=R.ROLE_S1, glass_config=glass, rf_config_factory=None)


def _paths(w, base):
    return dict(output_root=base / "models", record_dir=base / "records" / w["run"],
                primary_settings=w["settings"])


def _snapshot(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _diverging_glass_class():
    """The scripted GLASS, but under ``full`` its Pass 1 rule's precision differs (a defect)."""
    base = W.wired_glass_class()

    class Sel(W._Sel):
        def _pass(self, scope, X, y):
            rule = super()._pass(scope, X, y)
            if scope == "pass1" and self.owner.pass2_population == "full":
                rule = dataclasses.replace(rule, precision=0.91)
            return rule

    class Diverging(base):
        def _new_selector(self):
            return Sel(self)
    return Diverging


def _run_s1(w, base, glass_class=None, **kw):
    mp = pytest.MonkeyPatch()
    W.wired(mp, glass_class or W.wired_glass_class())
    _population_doubles(mp)
    _authorization_double(mp)
    try:
        with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
            S.run_s1(w["run"], settings=_s1_settings(w), train_side=lambda: w["ts"],
                     **_paths(w, base), **kw)
            return S.s1_acceptance(w["run"], s1_settings=_s1_settings(w),
                                   train_side=lambda: w["ts"], **_paths(w, base))
    finally:
        mp.undo()


def _copy(w, base):
    shutil.copytree(w["base"] / "models", base / "models")
    shutil.copytree(w["base"] / "records", base / "records")
    return base


@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    """An official acceptance of a synthetic primary (as test_pr38_acceptance.accepted_world)."""
    frozen_ids = R.primary_settings().checks["identities"]
    settings = dataclasses.replace(W.make_settings(role=R.ROLE_PRIMARY),
                                   checks={"identities": frozen_ids})
    w = _build(tmp_path_factory.mktemp("step3c"), settings, "r3c", R.ROLE_PRIMARY)
    mp = pytest.MonkeyPatch()
    _population_doubles(mp)
    try:
        rec = _accept(w)
    finally:
        mp.undo()
    assert rec["accepted"], {g: _failed(rec, g) for g in rec["gates"]}
    A.write_acceptance(w["records"], rec)
    return w


@pytest.fixture(scope="module")
def s1_world(accepted, tmp_path_factory):
    base = _copy(accepted, tmp_path_factory.mktemp("s1"))
    before = {"models": _snapshot(base / "models" / accepted["run"] / R.ROLE_PRIMARY),
              "records": _snapshot(base / "records" / accepted["run"])}
    rec, path = _run_s1(accepted, base)
    return SimpleNamespace(w=accepted, base=base, rec=rec, path=path, before=before)


@pytest.fixture
def doubles(monkeypatch):
    _population_doubles(monkeypatch)
    _authorization_double(monkeypatch)
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("load_train_side"))


# ======================================================================
# 1. evidence and fit status
# ======================================================================
def test_every_glass_fit_records_its_deployed_pass1_rules(accepted):
    want = [{"rule_id": 1, "segment": [list(W.PASS1_SEGMENT[0])], "predicted_class": 0,
             "precision": 0.9}]
    evidence = accepted["cand"].evidence
    for ev in [*evidence["folds"]["glass"], evidence["full_train"]["glass"]]:
        assert ev["pass1_rules"] == want, ev["fold"]
    assert accepted["g7"]["arms"]["glass"]["evidence"]["bit_identical"]   # G7 compares them too


def _s1_ev(**changes):
    rep = {"source": "ilp", "lexicographic_guarantee": "proven", "selected_rule_ids": [7]}
    c1 = {"population_rows": 1000, "population_positives": 100, "routed_rows": 300,
          "routed_positives": 5}
    c2 = {"population_rows": 1000, "population_positives": 100, "routed_rows": 200,
          "routed_positives": 80}
    ev = {"arm": "glass", "fold": 0, "pass2_population": "full",
          "selection_reports": {"pass1": rep, "pass2": dict(rep, selected_rule_ids=[8]),
                                "inner_pass1": []},
          "counts": {"pass1": c1, "pass2": c2}}
    for key, value in changes.items():
        if key == "reports":
            ev["selection_reports"] = {**ev["selection_reports"], **value}
        else:
            ev[key] = value
    return ev


@pytest.mark.parametrize("ev, status, problem", [
    (_s1_ev(), "valid_optimal", None),
    (_s1_ev(reports={"inner_pass1": [{"source": "ilp"}]}), "fit_time_invalid",
     "unexpected report inner_pass1[0]"),
    (_s1_ev(reports={"pass2": None}), "fit_time_invalid", "missing pass2"),
    (_s1_ev(pass2_population="oof_remainder"), "fit_time_invalid", "not an S1 GLASS fit"),
    (_s1_ev(reports={"pass2": {"source": "greedy_fallback", "selected_rule_ids": [8],
                               "lexicographic_guarantee": "not applicable"}}),
     ep.FIT_FALLBACK_UNREVIEWED, None),
])
def test_an_s1_fold_is_pass1_and_pass2_under_the_frozen_selection_rules(ev, status, problem):
    st = E.s1_fold_status(ev, W.LOOSE)
    assert st["status"] == status
    assert st["accepted"] is (status == "valid_optimal")
    if problem:
        assert any(problem in p for p in st["problems"]), st["problems"]
    primary = E.fold_status(_s1_ev(), W.LOOSE, ep.load_frozen_protocol())
    assert primary["status"] == "fit_time_invalid" and "missing inner_pass1[0]" in \
        primary["problems"]
    s1, other = W.make_settings(role=R.ROLE_S1), W.make_settings(role=R.ROLE_PRIMARY)
    assert R.fit_status(ev, s1) == st
    assert R.fit_status(_s1_ev(), other) == primary


# ======================================================================
# 2. the S1 run
# ======================================================================
def test_s1_is_refused_before_anything_without_the_primarys_acceptance(accepted, tmp_path,
                                                                        doubles):
    base = _copy(accepted, tmp_path)
    (base / "records" / accepted["run"] / A.ACCEPTANCE_FILE).unlink()
    with pytest.raises(R.RunnerError, match="no structural acceptance record"):
        S.run_s1(accepted["run"], settings=_s1_settings(accepted),
                 train_side=lambda: pytest.fail("data read"), **_paths(accepted, base))
    assert not (base / "models" / accepted["run"] / R.ROLE_S1).exists()
    assert not any((base / "records" / accepted["run"] / n).exists() for n in S.S1_RECORDS)


@pytest.mark.parametrize("settings, match", [
    (lambda w: _s1_settings(w, pass2_population="oof_remainder"), "pass2_population"),
    (lambda w: _s1_settings(w, remainder_min_size=60), "remainder_min_size"),
    (lambda w: dataclasses.replace(_s1_settings(w), role=R.ROLE_PRIMARY), "is not S1"),
    (lambda w: dataclasses.replace(_s1_settings(w), ranker={"glass_rf_model": "outer_rf"}),
     "ranker differs"),
])
def test_s1_must_be_the_primary_with_pass2_population_alone_changed(accepted, tmp_path, doubles,
                                                                    settings, match):
    base = _copy(accepted, tmp_path)
    with pytest.raises(S.SensitivityError, match=match):
        S.run_s1(accepted["run"], settings=settings(accepted),
                 train_side=lambda: pytest.fail("data read"), **_paths(accepted, base))
    assert not any((base / "records" / accepted["run"] / n).exists() for n in S.S1_RECORDS)


def test_s1_runs_once_beside_an_untouched_primary(s1_world, doubles):
    w, base = s1_world.w, s1_world.base
    s1_root = base / "models" / w["run"] / R.ROLE_S1
    files = _snapshot(s1_root)
    assert any(f.startswith("glass_router/glass_router_pr38_s1_full_") for f in files)
    assert not any(f.startswith("rf_router/") for f in files)
    assert _snapshot(base / "models" / w["run"] / R.ROLE_PRIMARY) == s1_world.before["models"]
    records = _snapshot(base / "records" / w["run"])
    assert {k: v for k, v in records.items() if k in s1_world.before["records"]} \
        == s1_world.before["records"]
    assert set(records) - set(s1_world.before["records"]) == {
        S.S1_ATTEMPT_FILE, S.S1_MANIFEST_FILE, S.S1_ACCEPTANCE_FILE}
    ev = json.loads((s1_root / "evidence.json").read_text(encoding="utf-8"))
    assert {e["fit_status"]["status"] for e in ev["folds"]["glass"]} == {"valid_optimal"}
    assert all(e["pass2_population"] == "full" and not e["selection_reports"]["inner_pass1"]
               for e in ev["folds"]["glass"])
    with pytest.raises(S.SensitivityError, match="already has an S1 attempt"):
        S.run_s1(w["run"], settings=_s1_settings(w), train_side=lambda: pytest.fail("read"),
                 **_paths(w, base))


# ======================================================================
# 3. S1 acceptance and G6
# ======================================================================
def test_g6_passes_and_s1_is_accepted(s1_world, doubles):
    rec, path = s1_world.rec, s1_world.path
    assert path.name == S.S1_ACCEPTANCE_FILE and rec["accepted"] and rec["g6_passed"]
    assert rec["record_digest"] == A._digest(json.loads(path.read_text(encoding="utf-8")))
    assert {n: g["passed"] for n, g in rec["gates"].items()} == dict.fromkeys(S.GROUPS, True)
    g6 = [c["check"] for c in rec["gates"]["G6"]["checks"]]
    assert len(g6) == 5 * 6 + 4
    assert any("full_train: deployed Pass 1 rules identical" in c for c in g6)
    assert any("fold 3: outer-OOF Pass 1 confidences identical" in c for c in g6)
    population = [c["check"] for c in rec["gates"]["S1.population"]["checks"]]
    for stage, groups in A.TRACE_COVERAGE.items():
        for t in groups:
            assert any(c.startswith(f"trace [{stage}] {t} ") for c in population), (stage, t)
        assert f"trace coverage [{stage}] {'/'.join(groups)} all evaluated" in population
    w = s1_world.w
    with pytest.raises(S.SensitivityError, match="never replaced"):
        S.s1_acceptance(w["run"], train_side=lambda: pytest.fail("read"),
                        **_paths(w, s1_world.base))


def test_a_pass1_difference_fails_g6_is_recorded_and_is_never_reported(accepted, tmp_path,
                                                                       doubles):
    base = _copy(accepted, tmp_path)
    rec, path = _run_s1(accepted, base, glass_class=_diverging_glass_class())
    assert not rec["accepted"] and not rec["g6_passed"]
    assert path.name.startswith("s1_acceptance_failed_") and "G6" in rec["statement"]
    failed = [c["check"] for c in rec["gates"]["G6"]["checks"] if not c["passed"]]
    assert "fold 0: deployed Pass 1 rules identical (segments, class, precision, order)" in failed
    assert "fold 0: outer-OOF Pass 1 confidences identical row by row" in failed
    assert "fold 0: fit-time Pass 1 counts identical" not in failed
    records = base / "records" / accepted["run"]
    assert not (records / S.S1_ACCEPTANCE_FILE).exists()
    with pytest.raises(S.SensitivityError, match="no S1 acceptance record"):
        S.load_s1_inputs(accepted["run"], "sums.txt", train_side=lambda: pytest.fail("read"),
                         **_paths(accepted, base))


class _Stop(Exception):
    pass


def test_a_failing_s1_trace_check_rejects_s1(s1_world, tmp_path, doubles, monkeypatch):
    """One S1 Pass 2 row relabelled Abstain without its decision token: T5 fails, S1 is refused."""
    w = s1_world.w
    base = _copy({"base": s1_world.base}, tmp_path)
    (base / "records" / w["run"] / S.S1_ACCEPTANCE_FILE).unlink()
    s1_glass = str(base / "models" / w["run"] / R.ROLE_S1 / "glass_router")
    real = d.extract_router_view

    def extract(path, ref, **kw):
        view = real(path, ref, **kw)
        if not str(path).startswith(s1_glass):
            return view
        st = np.array(view.state, dtype=object, copy=True)
        st[int(np.flatnonzero(st == P2)[0])] = AB
        return dataclasses.replace(view, state=st)
    monkeypatch.setattr(d, "extract_router_view", extract)
    rec, path = S.s1_acceptance(w["run"], s1_settings=_s1_settings(w),
                                train_side=lambda: w["ts"], **_paths(w, base))
    assert not rec["accepted"] and path.name.startswith("s1_acceptance_failed_")
    population = rec["gates"]["S1.population"]
    assert not population["passed"]
    failed = [c["check"] for c in population["checks"] if not c["passed"]]
    assert "the S1 trace (S1 GLASS, primary RF) is built" in failed
    assert "trace [glass] T5 state = SOURCE_DECISIONS[token] on every row" in failed
    assert not (base / "records" / w["run"] / S.S1_ACCEPTANCE_FILE).exists()


def test_the_report_loads_through_both_gates_in_order(s1_world, doubles):
    w, order = s1_world.w, []

    def train_side():
        order.append("train_side")
        return w["ts"]

    def upstream(ref, protocol, sums):
        order.append("upstream")
        raise _Stop()
    with pytest.raises(_Stop):
        S.load_s1_inputs(w["run"], "sums.txt", s1_settings=_s1_settings(w),
                         train_side=train_side, upstream=upstream, **_paths(w, s1_world.base))
    assert order == ["train_side", "upstream"]


def test_an_edited_s1_acceptance_or_a_changed_s1_candidate_is_refused(s1_world, tmp_path,
                                                                     doubles):
    w = s1_world.w
    base = _copy({"base": s1_world.base}, tmp_path / "a")
    path = base / "records" / w["run"] / S.S1_ACCEPTANCE_FILE
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["gates"]["G6"]["checks"] = rec["gates"]["G6"]["checks"][:1]
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(S.SensitivityError, match="does not establish an accepted S1"):
        S.load_s1_inputs(w["run"], "sums.txt", train_side=lambda: pytest.fail("read"),
                         **_paths(w, base))
    base = _copy({"base": s1_world.base}, tmp_path / "b")
    (base / "models" / w["run"] / R.ROLE_S1 / "extra.txt").write_text("x")
    with pytest.raises(S.SensitivityError, match="integrity"):
        S.load_s1_inputs(w["run"], "sums.txt", train_side=lambda: pytest.fail("read"),
                         **_paths(w, base))


def test_the_commands_refuse_without_their_gates(accepted, tmp_path, monkeypatch):
    import stage2_crossfit.__main__ as cli
    import stage2_crossfit.phase0 as P
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    monkeypatch.setattr(P, "OUTPUT_ROOT", tmp_path / "models")
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("data loaded"))
    (tmp_path / "logs" / "r9").mkdir(parents=True)
    for argv in (["s1"], ["s1-accept"], ["s1-report", "--sums", "s"]):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main([argv[0], "--run-id", "r9", *argv[1:]])
        assert code == 1 and "REFUSED" in out.getvalue(), argv
        assert "no structural acceptance record" in out.getvalue(), argv
    assert not (tmp_path / "models").exists()


# ======================================================================
# 4. the S1 report
# ======================================================================
def _full_evidence(primary_ev, trace):
    """The same folds under ``full``: Pass 2 fitted on every fold row, no inner selection."""
    out = json.loads(json.dumps(primary_ev, default=str))
    for ev in out["folds"]["glass"]:
        n, pos = ev["n_train"], ev["train_positives"]
        ev["pass2_population"] = "full"
        ev["selection_reports"]["inner_pass1"] = []
        ev["counts"] = {"pass1": ev["counts"]["pass1"],
                        "pass2": {"population_rows": n, "population_positives": pos,
                                  "routed_rows": 150, "routed_positives": 50}}
        ev.pop("remainder", None)
        ev.pop("m6_inner_oof_pass1", None)
    return out


def _with_pass1_rules(evidence):
    for ev in evidence["folds"]["glass"]:
        ev["pass1_rules"] = [{"rule_id": 1, "segment": [["behav_cold", 1]],
                              "predicted_class": 0, "precision": 0.9}]
    return evidence


def _s1_view(view, seed):
    """S1's GLASS outer OOF: Pass 1 rows untouched (G6), Pass 2 / Abstain rows reshuffled."""
    changed = _historical(view, seed)
    keep = np.asarray(view.state) == P1
    tok = np.where(keep, view.source_decision, changed.source_decision)
    tok = np.where(~keep & (tok == "pass1"), "uncertain", tok).astype(object)
    st = np.array([d.SOURCE_DECISIONS[t][0] for t in tok], dtype=object)
    pred = np.array([d.SOURCE_DECISIONS[t][1] for t in tok]).astype(
        np.asarray(view.source_pred).dtype)
    return dataclasses.replace(view, state=st, source_decision=tok, source_pred=pred)


@pytest.fixture(scope="module")
def synth():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    primary = d.build_router_trace(glass, rf, ref)
    s1 = d.build_router_trace(_s1_view(glass, 3), rf, ref)
    settings = W.make_settings()
    rec = {k: v for k, v in R.settings_record(settings).items()}
    rec_p = A._canon(dict(rec, role=R.ROLE_PRIMARY))
    rec_s = A._canon(dict(rec, role=R.ROLE_S1,
                          glass_config=dict(rec_p["glass_config"], pass2_population="full")))
    rec_s.pop("rf_config", None)
    ev_p = _with_pass1_rules(_evidence(primary))
    ev_s = _with_pass1_rules(_full_evidence(ev_p, primary))
    frozen = {"ratios": C._read_text_csv(b._csv_bytes(C.diagnostics_tables(
        _historical(glass, 1), _historical(rf, 2), ref, lr, ebm)["ratios"]))}
    acceptance = {"accepted": True, "g6_passed": True, "statement": "synthetic",
                  "gates": {"G6": {"checks": [{"check": "synthetic", "passed": True,
                                               "detail": ""}]}}}
    inputs = S.S1Inputs(
        "synthetic", settings.protocol, ref, primary, s1, lr, ebm,
        ep.ReferenceFlags.from_upstream(d.primary_reference_flags(lr, ebm)["union"]),
        {"oof_remainder": ev_p, "full": ev_s}, {"oof_remainder": rec_p, "full": rec_s},
        acceptance, frozen, {"run": {"run_id": "synthetic"}})
    return SimpleNamespace(inputs=inputs, tables=S.analyse_s1(inputs))


def test_the_report_is_single_factor_and_keeps_pass1_fixed(synth):
    t = synth.tables
    cfg = t["configuration"]
    assert list(cfg.loc[cfg["differs"], "setting"]) == ["glass_config.pass2_population"]
    tr = t["routing_transitions"]
    pooled = tr[tr["fold"] == "all"]
    p1 = pooled[(pooled["state_oof_remainder"] == P1) | (pooled["state_full"] == P1)]
    assert (p1[p1["changed"]]["rows"] == 0).all()
    assert pooled["rows"].sum() == synth.inputs.primary_trace.n
    assert pooled["changed_rows_in_scope"].iloc[0] > 0
    g6 = t["g6_pass1_invariance"]
    assert (g6["pass1_rules_digest_oof_remainder"] == g6["pass1_rules_digest_full"]).all()
    assert (g6["outer_pass1_rows_oof_remainder"] == g6["outer_pass1_rows_full"]).all()
    pop = t["pass2_fitting_population"]
    full = pop[pop["config"] == "full"]
    assert (full["pass2_population_rows"] == full["n_train"]).all()
    assert (pop[pop["config"] == "oof_remainder"]["pass2_population_share_of_fold"] < 1).all()
    feas = t["fold_feasibility"]
    assert set(feas[feas["config"] == "full"]["inner_pass1_selections"]) == {0}


def test_within_s1_and_baseline_differences_are_kept_apart(synth):
    t, inp = synth.tables, synth.inputs
    r = t["ratios_comparison"]
    row = r[(r["section"] == "2A.routing_rates") & (r["arm"] == "glass")
            & (r["fold"].astype(str) == "all") & (r["metric"] == "pass2_rate")].iloc[0]
    st_p = np.asarray(inp.primary_trace.views["glass"].state)
    st_s = np.asarray(inp.s1_trace.views["glass"].state)
    assert row["value_full_minus_oof_remainder"] == pytest.approx(
        (st_s == P2).mean() - (st_p == P2).mean(), abs=1e-11)
    rf = r[(r["section"] == "2A.routing_rates") & (r["arm"] == "blackbox")]
    assert (rf["value_full_minus_oof_remainder"].fillna(0) == 0).all()
    base = t["baseline_comparison"]
    assert set(base["arm"]) == {"glass"} and base["attribution"].str.contains("confounded").all()
    assert {"value_full_minus_oof_remainder", "value_oof_remainder_minus_baseline",
            "value_full_minus_baseline"} <= set(base.columns)
    assert "diagnostics/ratios" in t and "diagnostics/states" in t


def test_an_s1_without_g6_is_never_analysed(synth):
    bad = dict(synth.inputs.acceptance, g6_passed=False, accepted=False)
    with pytest.raises(S.SensitivityError, match="G6"):
        S.analyse_s1(dataclasses.replace(synth.inputs, acceptance=bad))


def test_the_package_is_deterministic_write_once_and_completes_the_report(synth, tmp_path):
    files = S.build_package(synth.inputs, synth.tables)
    assert S.build_package(synth.inputs, S.analyse_s1(synth.inputs)) == files
    man = json.loads(files[C.MANIFEST])
    assert list(man["comparative_sections"]) == ["declared_sensitivities"]
    assert any("never chooses, replaces or alters the primary" in x for x in man["disclosures"])
    protocol = synth.inputs.protocol
    records = tmp_path / "records"
    three_b = {s: ["tables/x.csv"] for s in protocol["acceptance"]["comparative_sections"]
               if s != "declared_sensitivities"}
    man_b = b._json_bytes({"comparative_sections": three_b})
    C.write_package(records / C.PACKAGE_DIR, {C.MANIFEST: man_b, C.SIDECAR: (
        hashlib.sha256(man_b).hexdigest() + "\n").encode()})
    assert S.comparative_report_status(records, protocol)["gaps"] == ["declared_sensitivities"]
    assert C.write_package(records / S.PACKAGE_DIR, files) == "written"
    assert C.write_package(records / S.PACKAGE_DIR, files) == "identical"
    status = S.comparative_report_status(records, protocol)
    assert status == {"packages": [C.PACKAGE_DIR, S.PACKAGE_DIR], "gaps": []}
    text = b"".join(files.values()).decode("utf-8")
    assert "/tmp/" not in text and "/home/" not in text
    pd.testing.assert_frame_equal(C._read_text_csv(files["tables/configuration.csv"]),
                                  C._read_text_csv(b._csv_bytes(synth.tables["configuration"])))
