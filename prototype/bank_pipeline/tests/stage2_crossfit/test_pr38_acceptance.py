"""
PR 38 Step 3A: the five-fold primary once per run, the G7 reproduction and
structural acceptance V1-V8 (``stage2_crossfit.acceptance``).

1. ``run_primary``: a durable attempt record before the first fit; a completed
   primary writes the candidate and its record copy; a stop or an interruption
   writes ``primary_stopped.json`` and saves nothing; no attempt is rerun.
2. The candidate: manifest, SHA-256, record copy, attempt; any change is an
   integrity failure, and acceptance then fails closed without loading.
3. G7: a refit of the pre-declared fold reproduces the candidate exactly; a
   record of another fold, run, role or pathway, or with one arm short, fails.
4. Owner dispositions: only an explicit satisfying decision counts.
5. V1-V8: every T1-T7 trace check is evaluated; on a synthetic run every gate
   but V1 (not the pinned population) and V2 (not the frozen settings) passes;
   each named defect fails its gate. No outer-OOF rate is recorded.
6. ``require_accepted``: an official acceptance re-verifies in full; a minimal,
   edited, incomplete or stale record, or a changed input, is refused.

Synthetic data, as in ``test_pr38_runner``: the GLASS rule search and ILP are
scripted; the RF Router is real with small forests. ``accepted_world`` stands
in for the frozen population with test doubles for the two population checks
(the only V1 checks synthetic rows cannot pass) and runs under the frozen
identities, so a complete official acceptance can be written and read back.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json
import shutil

import pytest

import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.acceptance as A
import stage2_crossfit.evidence as E
import stage2_crossfit.inner_features as IF
import stage2_crossfit.runner as R
from shared.stage2 import diagnostics as d
from tests.support import stage2_crossfit_world as W
from tests.support.rf_router_world import serial_forests

RUN, ROLE = "r3a", "test"
MAN_SHA = "a" * 64                       # the pinned artifact manifest, as Phase 0 recorded it
OK = {"reviewer": "owner", "rationale": "synthetic test", "date": "2026-11-11"}
F31_OK = dict(OK, decision="disposition_a_retained")


def _g10_ok():
    return dict(OK, decision="clarification_approved",
                record=E.record_identity(A.g10_clarification()))


def _authorization(pathway=E.PATHWAY_OUTER):
    return {"g13_decision": {"adopted_pathway": pathway},
            "f31_r2_clarification": {"record": E.record_identity(E.f31_r2_clarification())}}


def _g11(settings):
    return {"snapshot": {"passed": True}, "baseline": {"passed": True},
            "artifact_manifest": {"passed": True, "manifest_sha256": MAN_SHA},
            "baseline_manifest_sha256": settings.protocol["population"]["baseline_manifest_sha256"]}


def _approvals(cand, protocol):
    return {i: (_g10_ok() if i.startswith("G10") else F31_OK)
            for i in A.required_dispositions(cand, E.PATHWAY_OUTER, protocol)}


def _build(base, settings, run, role, *, accept_population=False):
    """A primary, its G7 record and an acceptance's inputs, under ``base``."""
    root, records = base / "models" / run / role, base / "records" / run
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())
    ts = W.synthetic_train_side()
    try:
        with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
            R.run_primary(ts, settings, root, run, records,
                          attempt={"authorization": {"name": "authorization.json",
                                                     "sha256": "0" * 64}})
            cand = A.load_candidate(run, role=role, output_root=base / "models",
                                    record_dir=records)
            g7 = A.reproduce_fold(ts, settings, cand)
    finally:
        mp.undo()
    R.write_record(records / A.G7_FILE, g7)
    R.write_record(records / "phase0.json", {"passed": True,
                                             "artifact_manifest": {"manifest_sha256": MAN_SHA}})
    (records / "authorization.json").write_text(json.dumps({"run_id": run}), encoding="utf-8")
    disp = {"format": A.DISPOSITIONS_FORMAT, "run_id": run,
            "dispositions": _approvals(cand, settings.protocol)}
    (records / A.DISPOSITIONS_FILE).write_text(json.dumps(disp), encoding="utf-8")
    (base / "sums.txt").write_text("synthetic")
    (base / "manifest.json").write_text("{}")
    inputs = A.acceptance_inputs(records, records / "authorization.json", base / "sums.txt",
                                 base / "manifest.json")
    # test doubles for the two external files: the hashes the real ones would have
    inputs["sums_sha256"] = json.loads((A.phase0.BASELINE_DIR / "MANIFEST.json").read_text(
        encoding="utf-8"))["snapshot"]["sha256"]
    inputs["artifact_manifest_sha256"] = MAN_SHA
    return dict(base=base, ts=ts, settings=settings, root=root, records=records, cand=cand,
                g7=g7, inputs=inputs, run=run, role=role)


def _accept(w, cand=None, **kw):
    cand = cand or w["cand"]
    args = dict(authorization=_authorization(), inputs=w["inputs"], g7=w["g7"], other_g7=(),
                dispositions=_approvals(cand, w["settings"].protocol),
                upstream=lambda trace, ref: ep.IdentityReport((("synthetic", True, "stub"),)),
                g11=_g11(w["settings"]))
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return A.acceptance(cand, w["ts"], w["settings"], **args)


def _failed(rec, gate):
    return [c["check"] for c in rec["gates"][gate]["checks"] if not c["passed"]]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("step3a"), W.make_settings(), RUN, ROLE)


@pytest.fixture(scope="module")
def accepted_world(tmp_path_factory):
    """Role ``primary`` under the frozen identities; the population checks doubled (module doc)."""
    frozen_ids = R.primary_settings().checks["identities"]
    settings = dataclasses.replace(W.make_settings(role=R.ROLE_PRIMARY),
                                   checks={"identities": frozen_ids})
    w = _build(tmp_path_factory.mktemp("accepted"), settings, "r3a_ok", R.ROLE_PRIMARY)
    mp = pytest.MonkeyPatch()
    mp.setattr(R, "check_population", lambda ts, protocol: {"passed": True, "checks": {}})
    mp.setattr(ep, "check_population_identity",
               lambda trace, identity: ep.IdentityReport((("synthetic rows (test double)", True,
                                                           "stub"),)))
    try:
        rec = _accept(w)
    finally:
        mp.undo()
    assert rec["accepted"], {g: _failed(rec, g) for g in rec["gates"]}
    w["path"] = A.write_acceptance(w["records"], rec)
    return w


def _copy(w, tmp_path):
    base = tmp_path / "c"
    shutil.copytree(w["base"] / "models", base / "models")
    shutil.copytree(w["base"] / "records", base / "records")
    return base


def _load(w, base):
    return A.load_candidate(w["run"], role=w["role"], output_root=base / "models",
                            record_dir=base / "records" / w["run"])


# ======================================================================
# 1. the primary, once per run, durable before the first fit
# ======================================================================
def test_a_completed_primary_writes_its_attempt_candidate_and_record_copy(world):
    assert (world["records"] / "primary_MANIFEST.json").read_bytes() == (
        world["root"] / "MANIFEST.json").read_bytes()
    attempt = json.loads((world["records"] / R.PRIMARY_ATTEMPT_FILE).read_text(encoding="utf-8"))
    assert (attempt["format"], attempt["run_id"], attempt["role"]) == (
        R.PRIMARY_ATTEMPT_FORMAT, RUN, ROLE)
    assert attempt["authorization"]["name"] == "authorization.json"
    assert not (world["records"] / "primary_stopped.json").exists()


def test_a_primary_is_never_rerun_under_the_same_run(world, monkeypatch):
    monkeypatch.setattr(R, "run_crossfit", lambda *a, **k: pytest.fail("refit"))
    with pytest.raises(R.RunnerError, match="already has a primary attempt"):
        R.run_primary(world["ts"], world["settings"], world["root"], RUN, world["records"])


def test_the_attempt_is_durable_before_the_first_fit_and_an_interruption_is_recorded(tmp_path,
                                                                                     monkeypatch):
    records, root = tmp_path / "records" / "i", tmp_path / "models" / "i" / ROLE

    def interrupted(*a, **k):
        assert (records / R.PRIMARY_ATTEMPT_FILE).is_file()      # on disk before any fit
        raise KeyboardInterrupt
    monkeypatch.setattr(R, "run_crossfit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        R.run_primary(W.synthetic_train_side(), W.make_settings(), root, "i", records)
    stop = json.loads((records / "primary_stopped.json").read_text(encoding="utf-8"))
    assert stop["exception"] == "KeyboardInterrupt" and stop["accepted"] is False
    with pytest.raises(R.RunnerError, match="already has a primary attempt"):
        R.run_primary(W.synthetic_train_side(), W.make_settings(), root, "i", records)


def test_an_attempt_without_an_outcome_still_blocks_the_run(tmp_path, monkeypatch):
    records = tmp_path / "records" / "p"
    records.mkdir(parents=True)
    (records / R.PRIMARY_ATTEMPT_FILE).write_text("{}")        # e.g. a power loss mid-run
    monkeypatch.setattr(R, "run_crossfit", lambda *a, **k: pytest.fail("refit"))
    with pytest.raises(R.RunnerError, match="interrupted"):
        R.run_primary(W.synthetic_train_side(), W.make_settings(), tmp_path / "m", "p", records)


@pytest.mark.parametrize("script, exception, status", [
    ("unproven", "FoldNotAccepted", "valid_unproven_undocumented"),   # fitted, not accepted
    ("infeasible", "RunnerError", "proven_infeasible"),               # the fit raised (G8)
])
def test_a_stopped_primary_records_its_evidence_and_saves_nothing(tmp_path, monkeypatch, script,
                                                                  exception, status):
    W.wired(monkeypatch, W.wired_glass_class(script={1: script}))
    root, records = tmp_path / "models" / "s" / ROLE, tmp_path / "records" / "s"
    with serial_forests(), pytest.raises(R.RunnerError, match="stopped during the cross-fit"):
        R.run_primary(W.synthetic_train_side(), W.make_settings(), root, "s", records)
    assert not root.exists()
    rec = json.loads((records / "primary_stopped.json").read_text(encoding="utf-8"))
    assert rec["accepted"] is False and rec["exception"] == exception
    assert rec["fit_status_by_fold"] == {"glass": {"0": status}}
    fold0 = rec["folds"]["glass"][0]
    assert fold0["selection_reports"] and "index" not in (fold0.get("remainder") or {})


# ======================================================================
# 2. the candidate, and acceptance failing closed
# ======================================================================
def test_the_candidate_is_intact_and_names_its_artifacts(world):
    cand = world["cand"]
    assert cand.intact, [c for c in cand.integrity if not c["passed"]]
    assert cand.glass_path.name == f"glass_router_pr38_{ROLE}_{RUN}.joblib"
    assert cand.rf_path.name.startswith(f"rf_router_pr38_{ROLE}_{RUN}_")


CHANGES = {
    "every listed file is present": lambda b: (b / "models" / RUN / ROLE / "evidence.json")
    .write_text("{}"),
    "no unlisted file": lambda b: (b / "models" / RUN / ROLE / "extra.txt").write_text("x"),
    "version-controlled copy": lambda b: (b / "records" / RUN / "primary_MANIFEST.json")
    .write_text("{}"),
    "durable primary attempt": lambda b: (b / "records" / RUN / R.PRIMARY_ATTEMPT_FILE).unlink(),
    "not in a stop": lambda b: (b / "records" / RUN / "primary_stopped.json").write_text("{}"),
}


@pytest.mark.parametrize("failing", list(CHANGES))
def test_any_change_is_an_integrity_failure_and_acceptance_fails_closed(world, tmp_path,
                                                                        monkeypatch, failing):
    base = _copy(world, tmp_path)
    CHANGES[failing](base)
    cand = _load(world, base)
    assert not cand.intact
    assert any(failing in c["check"] for c in cand.integrity if not c["passed"])
    monkeypatch.setattr(A.joblib, "load", lambda *a, **k: pytest.fail("an artifact was loaded"))
    with pytest.raises(R.RunnerError, match="nothing is loaded"):
        cand.load(cand.glass_path)
    rec = _accept(dict(world, ts=None), cand)                   # no data either
    sa = rec["structural_acceptance"]
    assert not rec["accepted"] and sa["failed"] == ["V7"]
    assert sa["not_run"] == [g for g in ("V1", "V2", "V3", "V4", "V5", "V6", "V8")]


# ======================================================================
# 3. G7, bound to the pre-declared fold
# ======================================================================
def test_a_refit_of_the_pre_declared_fold_reproduces_the_candidate_exactly(world):
    g7 = world["g7"]
    assert g7["fold"] == A.G7_FOLD == R.PREFLIGHT_FOLD and g7["pre_declared_fold"]
    assert g7["passed"] and g7["bit_identical"]
    assert A.g7_problems(g7, world["cand"], world["settings"],
                         held_out_rows=len(world["ts"].folds[A.G7_FOLD][1])) == []


def _g7(world, **changes):
    g7 = copy.deepcopy(world["g7"])
    for path, value in changes.items():
        *keys, last = path.split(".")
        node = g7
        for key in keys:
            node = node[key]
        if value is KeyError:
            del node[last]
        else:
            node[last] = value
    return g7


@pytest.mark.parametrize("changes, match", [
    ({"fold": 1, "pre_declared_fold": False}, "fold"),
    ({"role": "primary"}, "role"),
    ({"run_id": "another"}, "run_id"),
    ({"feature_pathway": {"pathway": E.PATHWAY_FOLD, "record": None}}, "feature_pathway"),
    ({"candidate_manifest_sha256": "0" * 64}, "candidate_manifest_sha256"),
    ({"arms.blackbox.passed": False, "arms.blackbox.held_out.passed": False}, "blackbox"),
    ({"arms.glass.evidence.n_differences": 1}, "glass"),
    ({"arms.blackbox": KeyError}, "arms"),
    ({"passed": False}, "not passed in both arms"),
])
def test_a_g7_record_that_is_not_this_candidates_pre_declared_pass_is_refused(world, changes,
                                                                              match):
    problems = A.g7_problems(_g7(world, **changes), world["cand"], world["settings"])
    assert any(match in p for p in problems), problems


def test_a_reproduction_that_differs_is_named_and_fails(world, monkeypatch):
    ev = copy.deepcopy(world["cand"].evidence)
    ev["folds"][E.ARM_RF][0]["thresholds"]["t1"] += 1e-9
    cand = dataclasses.replace(world["cand"], evidence=ev)
    W.wired(monkeypatch, W.wired_glass_class())
    with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
        g7 = A.reproduce_fold(world["ts"], world["settings"], cand)
    assert not g7["passed"] and g7["arms"][E.ARM_GLASS]["passed"]
    assert g7["arms"][E.ARM_RF]["evidence"]["differences"][0].startswith("/thresholds/t1")


def test_v6_reads_only_the_pre_declared_fold_and_refuses_another_on_the_record(world):
    assert "V6" not in _accept(world)["structural_acceptance"]["failed"]
    assert "V6" in _accept(world, g7=None)["structural_acceptance"]["failed"]
    rec = _accept(world, other_g7=["g7_reproduction_fold3.json"])
    assert _failed(rec, "V6") == ["G7: no reproduction of another fold is on the record"]


def test_the_json_comparison_keeps_floats_within_tolerance_and_everything_else_exact():
    assert A.compare_json({"t": 0.5}, {"t": 0.5 + 1e-13})["identical_within_tolerance"]
    assert not A.compare_json({"t": 0.5}, {"t": 0.5 + 1e-13})["bit_identical"]
    assert not A.compare_json({"t": 0.5}, {"t": 0.5 + 1e-9})["identical_within_tolerance"]
    assert not A.compare_json({"ids": [1, 2]}, {"ids": [2, 1]})["identical_within_tolerance"]
    assert not A.compare_json({"a": 1}, {"a": 1, "b": 2})["identical_within_tolerance"]
    assert A.compare_json({"wall_seconds": 1.0}, {"wall_seconds": 9.0})["bit_identical"]


# ======================================================================
# 4. owner dispositions: explicit approval only
# ======================================================================
@pytest.mark.parametrize("ident, entry, match", [
    ("F31:condition2", dict(OK, decision="reopened"), "cannot pass"),
    ("F31:condition2", OK, "missing"),                                   # documentation only
    ("F31:condition2", dict(OK, decision="approved"), "not one of"),
    ("F31:condition2", dict(F31_OK, rationale="<decision>"), "placeholder"),
    ("G10:artifact_audit_pins", dict(OK, decision="deferred"), "not one of"),
    ("G10:artifact_audit_pins", dict(OK, decision="rejected"), "cannot pass"),
    ("G10:artifact_audit_pins", dict(OK, decision="clarification_approved", record=["x"]),
     "names record"),
    ("G10:artifact_audit_pins", dict(OK, decision="pins_updated"), "still expects GLASS 'full'"),
    ("G10:artifact_audit_pins", None, "not recorded"),
])
def test_only_an_explicit_satisfying_decision_counts(ident, entry, match):
    problems = A.disposition_problems(ident, entry)
    assert problems and match in problems[0], problems


def test_pins_updated_names_both_g10_pins_while_neither_is_updated():
    problems = A.disposition_problems("G10:artifact_audit_pins", dict(OK, decision="pins_updated"))
    assert len(problems) == 1
    for pin in ("tests/support/artifact_audit.py", A.G10_PIN_TEST[1], A.GLASS_NOTEBOOK):
        assert pin in problems[0], problems


def _pinned_bank(tmp_path, *, pin_test_full: bool, notebook_profile: str):
    """A bank root holding the second pin's test file and the GLASS notebook, as edited."""
    from tests.support.notebooks import BANK_ROOT
    bank = tmp_path / "bank"
    (bank / "tests" / "parity").mkdir(parents=True)
    (bank / "glass_pipeline").mkdir()
    literal = 'pass2_population="full"' if pin_test_full else 'pass2_population="oof_remainder"'
    (bank / A.G10_PIN_TEST[0]).write_text(
        f"def {A.G10_PIN_TEST[1]}():\n    assert {literal!r} in 'x'\n", encoding="utf-8")
    nb = (BANK_ROOT / A.GLASS_NOTEBOOK).read_text(encoding="utf-8")
    assert 'CONFIG_PROFILE = \\"legacy\\"' in nb
    (bank / A.GLASS_NOTEBOOK).write_text(
        nb.replace('CONFIG_PROFILE = \\"legacy\\"', f'CONFIG_PROFILE = \\"{notebook_profile}\\"'),
        encoding="utf-8")
    return bank


@pytest.mark.parametrize("audit, pin_test_full, profile, missing", [
    ("oof_remainder", True, "legacy", (A.G10_PIN_TEST[1], A.GLASS_NOTEBOOK)),   # one pin only
    ("oof_remainder", False, "legacy", (A.GLASS_NOTEBOOK,)),     # pin edited, notebook not
    ("oof_remainder", True, "recommended", (A.G10_PIN_TEST[1],)),  # notebook, old pin kept
    ("full", False, "recommended", ("tests/support/artifact_audit.py",)),
    ("oof_remainder", False, "recommended", ()),                  # both pins established
])
def test_pins_updated_needs_both_g10_pins_established(tmp_path, monkeypatch, audit,
                                                      pin_test_full, profile, missing):
    from tests.support import artifact_audit
    monkeypatch.setitem(artifact_audit.PASS2_POPULATION, "glass", audit)
    monkeypatch.setattr(A.phase0, "BANK_ROOT",
                        _pinned_bank(tmp_path, pin_test_full=pin_test_full,
                                     notebook_profile=profile))
    problems = A.g10_pin_problems()
    assert len(problems) == len(missing) and all(
        any(m in p for p in problems) for m in missing), problems
    satisfied = A.disposition_problems("G10:artifact_audit_pins",
                                       dict(OK, decision="pins_updated")) == []
    assert satisfied is (not missing)


@pytest.mark.parametrize("source, count", [
    ("def test_the_glass_notebook_runs_the_corrected_profile():\n    pass\n", 0),   # renamed
    ("", 0),                                                                     # missing
    (f"def {A.G10_PIN_TEST[1]}():\n    pass\n\n\ndef {A.G10_PIN_TEST[1]}():\n    pass\n", 2),
])
def test_a_missing_renamed_or_duplicated_pin_test_does_not_establish_g10(tmp_path, monkeypatch,
                                                                         source, count):
    from tests.support import artifact_audit
    monkeypatch.setitem(artifact_audit.PASS2_POPULATION, "glass", "oof_remainder")
    bank = _pinned_bank(tmp_path, pin_test_full=False, notebook_profile="recommended")
    (bank / A.G10_PIN_TEST[0]).write_text(source, encoding="utf-8")
    monkeypatch.setattr(A.phase0, "BANK_ROOT", bank)
    problems = A.g10_pin_problems()
    assert len(problems) == 1, problems                   # the other two pins hold
    assert A.G10_PIN_TEST[1] in problems[0] and f"{count} definitions" in problems[0]
    assert "expected exactly one" in problems[0]
    assert A.disposition_problems("G10:artifact_audit_pins",
                                  dict(OK, decision="pins_updated")) != []


def test_an_approved_clarification_and_a_retained_disposition_satisfy():
    assert A.disposition_problems("G10:artifact_audit_pins", _g10_ok()) == []
    assert A.disposition_problems("F31:R2:fold0:triggered", F31_OK) == []
    assert A.g10_clarification()["status"] == "proposed"         # inactive until approved


def test_missing_or_refused_dispositions_fail_their_gates(world):
    rec = _accept(world, dispositions={})
    assert any("G10:artifact_audit_pins" in c for c in _failed(rec, "V7"))
    flagged = [r for r in rec["f31_strip_review"] if r["review_required"]]
    assert flagged                                   # the synthetic strips cross 1/20
    assert len([c for c in _failed(rec, "V8") if "F31:R2:" in c]) == len(flagged)
    refused = {i: dict(e, decision="reopened") if i.startswith("F31") else e
               for i, e in _approvals(world["cand"], world["settings"].protocol).items()}
    assert "V8" in _accept(world, dispositions=refused)["structural_acceptance"]["failed"]


def test_the_adopted_correction_requires_the_f31_condition_2_disposition(world):
    req = A.required_dispositions(world["cand"], E.PATHWAY_FOLD, world["settings"].protocol)
    assert "F31:condition2" in req and "G10:artifact_audit_pins" in req
    assert "F31:condition2" not in A.required_dispositions(world["cand"], E.PATHWAY_OUTER,
                                                           world["settings"].protocol)


V4_G13 = "G13: every fit ran on the adopted feature pathway"
V8_R8 = "R8: bands come from fits on the adopted G13 pathway (fold and full-train routers)"


def _corrected(ev, ts):
    """Every fit records the provider's provenance: the pathway identity plus fingerprints."""
    for rec in [*ev["folds"][E.ARM_GLASS], *ev["folds"][E.ARM_RF], *ev["full_train"].values()]:
        rec["inner_feature_pathway"] = {**IF.pathway_identity(E.PATHWAY_FOLD),
                                        "plan_fingerprint": "0" * 16, "fit_rows": rec["n_train"],
                                        "inner_folds": [{"fold": 0, "bins_fingerprint": "1" * 16}]}


def _other_record(ev, ts):
    _corrected(ev, ts)
    ev["folds"][E.ARM_RF][3]["inner_feature_pathway"]["record"] = ["another", "1.0.0", "0" * 16]


def _glass_full_unrecorded(ev, ts):
    _corrected(ev, ts)
    del ev["full_train"][E.ARM_GLASS]["inner_feature_pathway"]


def _historical(ev, ts):
    pass                                        # the world's fits ran on outer_engineer


@pytest.mark.parametrize("change, v4, r8", [(_corrected, True, True), (_other_record, False, False),
                                            (_glass_full_unrecorded, False, True),
                                            (_historical, False, False)])
def test_v4_and_r8_compare_the_recorded_pathway_identity_not_its_provenance(world, change, v4, r8):
    ev = copy.deepcopy(world["cand"].evidence)
    change(ev, world["ts"])
    rec = _accept(world, dataclasses.replace(world["cand"], evidence=ev),
                  authorization=_authorization(E.PATHWAY_FOLD))
    assert (V4_G13 not in _failed(rec, "V4"), V8_R8 not in _failed(rec, "V8")) == (v4, r8)


def test_dispositions_belong_to_the_run_and_tolerate_a_bom(tmp_path):
    assert A.load_dispositions(tmp_path, RUN) == ({}, None)
    body = {"format": A.DISPOSITIONS_FORMAT, "run_id": RUN,
            "dispositions": {"G10:artifact_audit_pins": _g10_ok()}}
    (tmp_path / A.DISPOSITIONS_FILE).write_text(json.dumps(body), encoding="utf-8-sig")
    got, sha = A.load_dispositions(tmp_path, RUN)
    assert got == body["dispositions"] and len(sha) == 64
    with pytest.raises(R.RunnerError, match="expected format"):
        A.load_dispositions(tmp_path, "another_run")


# ======================================================================
# 5. V1-V8
# ======================================================================
def test_every_gate_but_the_frozen_identities_passes_on_a_synthetic_run(world):
    rec = _accept(world)
    sa = rec["structural_acceptance"]
    assert sa["official"] and not rec["accepted"]
    assert sa["failed"] == ["V1", "V2"] and set(sa["passed"]) == {f"V{i}" for i in range(3, 9)}
    assert all("population" in c for c in _failed(rec, "V1"))           # not the pinned rows
    assert _failed(rec, "V2") == ["the run's settings are the frozen primary (role, frozen "
                                  "identities)"]


def test_v1_evaluates_every_trace_check_and_each_t_group(world):
    names = [c["check"] for c in _accept(world)["gates"]["V1"]["checks"]]
    trace = [n for n in names if n.startswith("trace [")]
    for stage, groups in A.TRACE_COVERAGE.items():
        for t in groups:
            assert any(n.startswith(f"trace [{stage}] {t} ") for n in trace), (stage, t)
        assert any(n.startswith(f"trace coverage [{stage}] ") for n in names)
    assert any(n.startswith("boundary [glass] ") for n in names)
    assert any(n.startswith("boundary [blackbox] ") for n in names)


@pytest.mark.parametrize("edit, match", [
    (lambda recs: [r for r in recs if not str(r["check"]).startswith("T7")], "trace coverage"),
    (lambda recs: [dict(r, passed=False) if str(r["check"]).startswith("T4") else r
                   for r in recs], "trace [glass] T4"),
])
def test_a_trace_missing_a_t_group_or_failing_a_check_fails_v1(world, monkeypatch, edit, match):
    real = d.build_router_trace

    def edited(*a, **k):
        t = real(*a, **k)
        return dataclasses.replace(t, checks=tuple(edit([dict(r) for r in t.checks])))
    monkeypatch.setattr(d, "build_router_trace", edited)
    assert any(c.startswith(match) for c in _failed(_accept(world), "V1"))


def test_a_trace_that_raises_records_each_of_its_checks(world, monkeypatch):
    real = d.build_router_trace

    def failing(glass, rf, ref):
        return real(glass, rf, dataclasses.replace(ref, y_train=1 - ref.y_train))   # T3 fails
    monkeypatch.setattr(d, "build_router_trace", failing)
    failed = _failed(_accept(world), "V1")
    assert "both views extract and the router trace is built" in failed
    assert any(c.startswith("trace [glass] T3") for c in failed), failed


def test_the_acceptance_record_holds_no_outer_oof_rate(world):
    text = json.dumps(_accept(world))
    for token in ("train_pos_rate", "train_coverage", "pass1_rate", "pass2_rate", "abstain_rate",
                  "pass1_npv", "pass2_precision", "leakage_share\": 0", "auc"):
        assert token not in text, token


def _held_out_row_in_remainder(ev, ts):
    rem = ev["folds"][E.ARM_RF][2]["remainder"]["index"]
    rem[0] = E.to_json_safe(ts.X_train.index[ts.folds[2][1][0]])


def _no_strips(ev, ts):
    del ev["folds"][E.ARM_RF][3]["r2_strips"]


def _relabelled(ev, ts):
    fold1 = ev["folds"][E.ARM_GLASS][1]
    fold1["selection_reports"]["pass2"] = None
    fold1["fit_status"].update(status="valid_optimal", accepted=True)


@pytest.mark.parametrize("change, gate", [(_held_out_row_in_remainder, "V4"), (_no_strips, "V5"),
                                          (_relabelled, "V3")])
def test_isolation_persistence_and_relabelling_fail_their_gates(world, change, gate):
    ev = copy.deepcopy(world["cand"].evidence)
    change(ev, world["ts"])
    rec = _accept(world, dataclasses.replace(world["cand"], evidence=ev))
    assert gate in rec["structural_acceptance"]["failed"]


def _boom(*a, **k):
    raise RuntimeError("consumer failed")


@pytest.mark.parametrize("module, attr, consumers", [
    ("shared.stage4", "router_stream",
     ("router_stream", "meta-EBM router evidence", "meta-XGB feature block")),
    ("glass_pipeline.meta_ebm.loader", "_router_evidence_checks", ("meta-EBM router evidence",)),
    ("blackbox_pipeline.models.meta_xgb.inputs", "build_feature_frames",
     ("meta-XGB feature block",)),
])
def test_a_raising_r4_consumer_is_a_recorded_v7_failure(world, monkeypatch, module, attr,
                                                        consumers):
    import importlib
    monkeypatch.setattr(importlib.import_module(module), attr, _boom)
    rec = _accept(world)                                         # completes; nothing raised
    sa = rec["structural_acceptance"]
    assert "V7" in sa["failed"] and {"V3", "V4", "V5", "V6", "V8"} <= set(sa["passed"])
    failed = _failed(rec, "V7")
    for arm in ("glass", "blackbox"):
        for consumer in consumers:
            assert f"R4 {consumer} [{arm}] ran" in failed, failed
    details = [c["detail"] for c in rec["gates"]["V7"]["checks"] if c["check"].endswith("ran")]
    assert any("RuntimeError: consumer failed" in d for d in details)


def test_an_r4_failure_outside_the_consumers_is_a_recorded_v7_failure(world, monkeypatch):
    monkeypatch.setattr(A, "_r4_checks", lambda *a: _boom())
    rec = _accept(world)
    assert "R4 Stage 4 consumers ran" in _failed(rec, "V7")
    assert "V7" in rec["structural_acceptance"]["failed"]


def test_g11_upstream_identity_and_the_phase0_manifest_are_required(world):
    rec = _accept(world, g11=None, upstream=None)
    assert any(c.startswith("G11") for c in _failed(rec, "V7"))
    assert any("check_upstream_identity" in c for c in _failed(rec, "V1"))
    other = dict(_g11(world["settings"]), artifact_manifest={"passed": True,
                                                             "manifest_sha256": "b" * 64})
    assert any("Phase 0 checked" in c for c in _failed(_accept(world, g11=other), "V7"))


# ======================================================================
# 6. Step 3B reads only a complete, re-verified official acceptance
# ======================================================================
def _require(w, base=None, **kw):
    base = base or w["base"]
    return A.require_accepted(w["run"], output_root=base / "models",
                              record_dir=base / "records" / w["run"], settings=w["settings"], **kw)


@pytest.fixture
def stub_authorization(monkeypatch):
    def verify(path, settings=None, *, run_id=None):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("run_id") != run_id:
            raise R.RunnerError("authorization belongs to another run")
        return rec
    monkeypatch.setattr(R, "require_authorization", verify)


def test_a_complete_official_acceptance_re_verifies(accepted_world, stub_authorization):
    cand = _require(accepted_world)
    assert cand.manifest_sha256 == accepted_world["cand"].manifest_sha256


def _rewrite(base, w, edit, *, redigest=True):
    path = base / "records" / w["run"] / A.ACCEPTANCE_FILE
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec = edit(rec) or rec
    if redigest:
        rec["record_digest"] = A._digest(rec)
    path.write_text(json.dumps(rec), encoding="utf-8")


def _drop_check(rec, gate, prefix):
    rec["gates"][gate]["checks"] = [c for c in rec["gates"][gate]["checks"]
                                    if not c["check"].startswith(prefix)]


def _fail_one(rec):
    rec["gates"]["V4"]["checks"][0]["passed"] = False


REFUSALS = {
    "the minimal accepted record of the earlier tests": (lambda rec: {
        "format": A.ACCEPTANCE_FORMAT, "run_id": rec["run_id"], "accepted": True,
        "candidate_manifest_sha256": rec["candidate_manifest_sha256"],
        "structural_acceptance": {"accepted": True, "official": True}}, True, "gates"),
    "an edit after writing": (lambda rec: rec["gates"]["V1"]["checks"][0].update(passed=False),
                              False, "record_digest"),
    "a failed check inside a passed gate": (_fail_one, True, "V4: not passed"),
    "a missing gate": (lambda rec: rec["gates"].pop("V8"), True, "gates"),
    "missing mandatory evidence": (lambda rec: _drop_check(rec, "V1", "check_upstream_identity"),
                                   True, "V1: incomplete evidence"),
    "another role": (lambda rec: rec.update(role="test"), True, "role"),
    "another protocol": (lambda rec: rec.update(protocol=["x", "1", "0"]), True, "protocol"),
    "a verdict without V8": (lambda rec: rec["structural_acceptance"].update(
        passed=rec["structural_acceptance"]["passed"][:-1]), True, "recorded verdict"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_an_incomplete_or_fabricated_acceptance_record_is_refused(accepted_world, tmp_path,
                                                                  stub_authorization, name):
    edit, redigest, match = REFUSALS[name]
    base = _copy(accepted_world, tmp_path)
    _rewrite(base, accepted_world, edit, redigest=redigest)
    with pytest.raises(R.RunnerError, match="does not establish") as exc:
        _require(accepted_world, base)
    assert match in str(exc.value), str(exc.value)


INPUT_CHANGES = {
    "dispositions": lambda r: (r / A.DISPOSITIONS_FILE).write_text(json.dumps({
        "format": A.DISPOSITIONS_FORMAT, "run_id": "r3a_ok", "dispositions": {}})),
    "g7": lambda r: (r / A.G7_FILE).write_text("{}"),
    "authorization": lambda r: (r / "authorization.json").write_text(json.dumps({"run_id": "x"})),
    "phase0": lambda r: (r / "phase0.json").write_text("{}"),
    "G7: reproductions of other folds": lambda r: (r / "g7_reproduction_fold2.json")
    .write_text("{}"),
}


@pytest.mark.parametrize("name", list(INPUT_CHANGES))
def test_a_changed_or_added_input_after_acceptance_is_refused(accepted_world, tmp_path,
                                                              stub_authorization, name):
    base = _copy(accepted_world, tmp_path)
    INPUT_CHANGES[name](base / "records" / accepted_world["run"])
    with pytest.raises(R.RunnerError) as exc:
        _require(accepted_world, base)
    assert name in str(exc.value), str(exc.value)


def test_a_changed_candidate_is_refused_before_anything_is_loaded(accepted_world, tmp_path,
                                                                  monkeypatch):
    base = _copy(accepted_world, tmp_path)
    (base / "models" / accepted_world["run"] / R.ROLE_PRIMARY / "extra.txt").write_text("x")
    monkeypatch.setattr(A.joblib, "load", lambda *a, **k: pytest.fail("an artifact was loaded"))
    with pytest.raises(R.RunnerError, match="fails its integrity checks"):
        _require(accepted_world, base)


def test_require_accepted_refuses_a_run_without_an_acceptance_and_never_reads_a_failed_one(
        world, tmp_path):
    base = _copy(world, tmp_path)
    records = base / "records" / RUN
    with pytest.raises(R.RunnerError, match="no structural acceptance record"):
        A.require_accepted(RUN, role=ROLE, output_root=base / "models", record_dir=records,
                           settings=world["settings"])
    failed = A.write_acceptance(records, _accept(world, _load(world, base)))
    assert failed.name.startswith("structural_acceptance_failed_")
    with pytest.raises(R.RunnerError, match="no structural acceptance record"):
        A.require_accepted(RUN, role=ROLE, output_root=base / "models", record_dir=records,
                           settings=world["settings"])


# ======================================================================
# the command line
# ======================================================================
def test_reproduce_and_accept_need_this_runs_authorization_before_any_data(tmp_path,
                                                                           monkeypatch):
    import stage2_crossfit.__main__ as cli
    import stage2_crossfit.phase0 as P
    from tests.support import authorization_world as S
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    good = S.authorization_files(tmp_path / "logs" / "r1")
    (tmp_path / "logs" / "r2").mkdir()
    (tmp_path / "logs" / "r2" / "phase0.json").write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(R, "load_train_side", lambda **k: pytest.fail("data loaded"))
    for argv in (["reproduce"], ["accept", "--sums", "s", "--artifact-manifest", "m"]):
        with pytest.raises(R.RunnerError, match="not in run 'r2'"):
            cli.main([*argv, "--run-id", "r2", "--authorization", str(good),
                      "--feature-pathway", E.PATHWAY_FOLD])


def test_the_accept_command_writes_an_acceptance_that_require_accepted_re_verifies(
        accepted_world, tmp_path, monkeypatch, stub_authorization):
    """The command's wiring, end to end; doubles only where accepted_world has them."""
    import stage2_crossfit.__main__ as cli
    import stage2_crossfit.phase0 as P
    w = accepted_world
    base = _copy(w, tmp_path)
    records = base / "records" / w["run"]
    (records / A.ACCEPTANCE_FILE).unlink()
    monkeypatch.setattr(P, "RECORD_ROOT", base / "records")
    monkeypatch.setattr(P, "OUTPUT_ROOT", base / "models")
    monkeypatch.setattr(R, "primary_settings", lambda pathway=E.PATHWAY_OUTER: w["settings"])
    monkeypatch.setattr(R, "require_authorization", lambda path, settings=None, *, run_id=None:
                        _authorization())
    monkeypatch.setattr(R, "load_train_side", lambda **k: w["ts"])
    monkeypatch.setattr(R, "check_population", lambda ts, protocol: {"passed": True, "checks": {}})
    monkeypatch.setattr(ep, "check_population_identity", lambda trace, identity: ep.IdentityReport(
        (("synthetic rows (test double)", True, "stub"),)))
    monkeypatch.setattr(A, "upstream_identity", lambda trace, ref, protocol, sums:
                        ep.IdentityReport((("synthetic", True, "stub"),)))
    monkeypatch.setattr(A, "g11_checks", lambda sums, manifest: _g11(w["settings"]))
    real_inputs = A.acceptance_inputs
    monkeypatch.setattr(A, "acceptance_inputs", lambda *a: dict(
        real_inputs(*a), sums_sha256=w["inputs"]["sums_sha256"],
        artifact_manifest_sha256=MAN_SHA))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = cli.main(["accept", "--run-id", w["run"], "--authorization",
                         str(records / "authorization.json"), "--sums", str(w["base"] / "sums.txt"),
                         "--artifact-manifest", str(w["base"] / "manifest.json")])
    assert code == 0 and "ACCEPTED" in out.getvalue(), out.getvalue()
    assert _require(w, base).manifest_sha256 == w["cand"].manifest_sha256


def test_reproduce_takes_no_fold_argument():
    import stage2_crossfit.__main__ as cli
    with pytest.raises(SystemExit):
        cli.main(["reproduce", "--run-id", "r", "--authorization", "a", "--fold", "1"])
