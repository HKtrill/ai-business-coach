"""
PR 38 Step 2B: the evidence-backed authorization (``require_authorization``)
and the final review command (``python -m stage2_crossfit authorize``).

1. A complete authorization adopting the correction passes, for a run on the
   adopted pathway only.
2. Every gate is checked against its evidence, not the record's existence:
   G13 (both arms' audits complete and reproduced, the corrected audit's bins
   matching, the decision coherent, no test row), G2 / G3 / G5 (the preflight
   on the adopted pathway, wired as the primary, every field and union
   recorded), G1 / G9 (the configs record equal to the settings built now),
   G8 (a passing test report), F31 (the clarification approved).
3. A GLASS non-reproduction needs an explicit recorded disposition.
4. ``authorize`` writes the review on success and refuses (exit 1) otherwise.
5. Run binding: the authorization, and every current-run record, belong to
   one run (its directory, its ``run_id``, its passed Phase 0) and to the
   pre-declared fold 0 with one training population and fold identity; the
   original G13 audit keeps its own run, whose provenance is checked; neither
   ``authorize`` nor ``primary`` accepts another run's authorization.
"""

from __future__ import annotations

import json
import shutil

import pytest

import stage2_crossfit.__main__ as cli
import stage2_crossfit.inner_features as IF
import stage2_crossfit.phase0 as P
import stage2_crossfit.runner as R
from tests.support import authorization_world as S

FOLD, OUTER = IF.PATHWAY_FOLD, IF.PATHWAY_OUTER


def _refused(tmp_path, match, **kw):
    with pytest.raises(R.RunnerError, match=match):
        R.require_authorization(S.authorization_files(tmp_path, **kw))


# ======================================================================
# 1. a complete authorization
# ======================================================================
def test_a_complete_authorization_passes_for_the_adopted_pathway_only(tmp_path):
    path = S.authorization_files(tmp_path)
    assert R.require_authorization(path)["g13_decision"]["adopted_pathway"] == FOLD
    assert R.require_authorization(path, R.primary_settings(FOLD))
    with pytest.raises(R.RunnerError, match="not the adopted 'fold_label_excluded'"):
        R.require_authorization(path, R.primary_settings())


# ======================================================================
# 2. every gate against its evidence
# ======================================================================
def _edit(edit):
    bodies = S.records()
    edit(bodies)
    return bodies


def _drop(mapping, key):
    mapping.pop(key)


EVIDENCE_FAILURES = {
    "G13 original: no rf audit evidence":
        lambda b: _drop(b["g13"]["g13"]["eligibility_retrained"], "rf"),
    "G13 original: the glass retrained eligibility comparison is incomplete":
        lambda b: b["g13"]["g13"]["eligibility_retrained"]["glass"].update(complete=False),
    "G13 original: the rf t1_j reproduction":
        lambda b: b["g13"]["g13"]["eligibility_retrained"]["rf"].update(
            t1_reproduction_max_difference=1e-3),
    "uses_test_rows": lambda b: b["g13"]["g13"].update(uses_test_rows=True),
    "G13 corrected: rf was audited on 'outer_engineer'":
        lambda b: b.update(g13_corrected=b["g13"]),
    "G13 corrected: the glass fit's inner bins are not the audit's own":
        lambda b: b["g13_corrected"]["g13"]["isolation"]["glass"].update(pathway_match=False),
    "G13 corrected: the rf retrained eligibility comparison is incomplete":
        lambda b: b["g13_corrected"]["g13"]["eligibility_retrained"]["rf"].update(complete=False),
    "G3: the preflight ran on 'outer_engineer'":
        lambda b: b["g3"].update(feature_pathway=IF.pathway_identity(OUTER)),
    "G2: GLASS pass2_population oof_remainder":
        lambda b: b["g3"]["arms"]["glass"].update(pass2_population="full"),
    "G2: RF nested_t1":
        lambda b: b["g3"]["arms"]["blackbox"]["eligibility"].update(rule="global_t1"),
    "G2: glass inner bins match the fit":
        lambda b: b["g3"]["arms"]["glass"]["isolation"].update(
            inner_features_match_the_fit=False),
    r"G5: blackbox evidence lacks \['r2_strips'\]":
        lambda b: _drop(b["g3"]["arms"]["blackbox"], "r2_strips"),
    "G5: glass union counts":
        lambda b: _drop(b["g3"]["arms"]["glass"]["counts"], "inner_pass1[4]"),
    "G1: the configs record is not the primary settings":
        lambda b: b["configs"]["primary"]["checks"].update(check_rf_primary_config="skipped"),
}


@pytest.mark.parametrize("match", list(EVIDENCE_FAILURES))
def test_each_gate_is_refused_on_its_evidence(tmp_path, match):
    _refused(tmp_path, match, bodies=_edit(EVIDENCE_FAILURES[match]))


def test_the_g13_decision_must_be_coherent(tmp_path):
    base = dict(S.REVIEW, decision="material")
    cases = {
        "materiality must be": dict(base, materiality="unsure", adopted_pathway=FOLD,
                                    adopted_record=IF.pathway_identity(FOLD)["record"]),
        "adopted_pathway must be one of": dict(base, materiality="material",
                                               adopted_pathway="leave_one_out"),
        "adopted_record is not": dict(base, materiality="material", adopted_pathway=FOLD,
                                      adopted_record=["x", "0", "0"]),
        "needs the correction adopted": dict(base, materiality="material",
                                             adopted_pathway=OUTER, adopted_record=None),
    }
    for match, decision in cases.items():
        _refused(tmp_path / match.split()[0], match, g13_decision=decision)


def test_an_adopted_correction_needs_its_corrected_audit(tmp_path):
    _refused(tmp_path, "no valid corrected audit", g13_corrected_record=None)


def test_g8_needs_a_passing_report_from_every_required_module(tmp_path):
    report = S.junit(tmp_path / "r1.xml", failing=["tests.stage2_crossfit.test_pr38_runner"])
    _refused(tmp_path / "a", "G8: 1 failing tests", report=report)
    report = S.junit(tmp_path / "r2.xml", modules=R.AUTHORIZATION_TEST_MODULES[:-1])
    _refused(tmp_path / "b", "G8: no passing test from tests.stage2_crossfit.test_step2b_author",
             report=report)
    (tmp_path / "r3.xml").write_text("not xml")
    _refused(tmp_path / "c", "G8: the test report is not JUnit XML", report=tmp_path / "r3.xml")


def test_the_f31_r2_clarification_must_be_approved(tmp_path):
    ident = R.E.record_identity(R.E.f31_r2_clarification())
    _refused(tmp_path / "a", "F31: the R2 clarification",
             f31_r2_clarification={"record": ident, "approved_by": "", "approved_on": "x"})
    _refused(tmp_path / "b", "F31: the R2 clarification",
             f31_r2_clarification={"record": [ident[0], "0.9.0", ident[2]],
                                   "approved_by": "owner", "approved_on": "2026-11-12"})


def test_every_problem_is_reported_together(tmp_path):
    def two(b):
        b["g3"]["arms"]["glass"].update(pass2_population="full")
        b["g13"]["g13"].update(uses_test_rows=True)
    with pytest.raises(R.RunnerError) as exc:
        R.require_authorization(S.authorization_files(tmp_path, bodies=_edit(two)))
    assert "G2:" in str(exc.value) and "uses_test_rows" in str(exc.value)


# ======================================================================
# 3. dispositions
# ======================================================================
def test_a_glass_non_reproduction_needs_a_recorded_disposition(tmp_path):
    def unreproduced(b):
        b["g13_corrected"]["g13"]["eligibility_retrained"]["glass"].update(reproduced_all=False)
    _refused(tmp_path / "a", "G13:corrected:glass:reproduction", bodies=_edit(unreproduced))
    path = S.authorization_files(tmp_path / "b", bodies=_edit(unreproduced), dispositions={
        "G13:corrected:glass:reproduction": dict(S.REVIEW, rationale="CBC time limit; unproven")})
    assert R.require_authorization(path)


# ======================================================================
# 4. the authorize command
# ======================================================================
def test_authorize_writes_the_review_or_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    good = S.authorization_files(tmp_path / "logs" / "r1")
    assert cli.main(["authorize", "--run-id", "r1", "--authorization", str(good),
                     "--feature-pathway", FOLD]) == 0
    review = json.loads((tmp_path / "logs" / "r1" / "authorization_review.json").read_text(
        encoding="utf-8"))
    assert review["feature_pathway"] == IF.pathway_identity(FOLD)
    assert set(review["referenced"]) == {"g3_record", "g13_record", "g13_corrected_record",
                                         "configs_record", "test_report"}
    (tmp_path / "logs" / "r1" / "authorization_review.json").unlink()
    assert cli.main(["authorize", "--run-id", "r1", "--authorization", str(good)]) == 1
    assert "AUTHORIZATION REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "logs" / "r1" / "authorization_review.json").exists()


# ======================================================================
# 5. run binding
# ======================================================================
def test_neither_authorize_nor_primary_accepts_another_runs_authorization(tmp_path,
                                                                          monkeypatch, capsys):
    monkeypatch.setattr(P, "RECORD_ROOT", tmp_path / "logs")
    good = S.authorization_files(tmp_path / "logs" / "r1")
    with pytest.raises(R.RunnerError, match="belongs to run 'r1', not 'r2'"):
        R.require_authorization(good, run_id="r2")
    assert cli.main(["authorize", "--run-id", "r2", "--authorization", str(good),
                     "--feature-pathway", FOLD]) == 1
    assert "not in run 'r2'" in capsys.readouterr().out
    (tmp_path / "logs" / "r2").mkdir()
    (tmp_path / "logs" / "r2" / "phase0.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(R.RunnerError, match="not in run 'r2'"):     # before any data is loaded
        cli.main(["primary", "--run-id", "r2", "--authorization", str(good),
                  "--feature-pathway", FOLD])
    _refused(tmp_path / "r3", "is not its run directory", run_id="r9")


RUN_FAILURES = {
    "g3_record belongs to run 'other'": lambda b: b["g3"].update(run_id="other"),
    "configs_record belongs to run 'other'": lambda b: b["configs"].update(run_id="other"),
    "g13_corrected_record is fold 1": lambda b: b["g13_corrected"].update(fold=1),
    "g13_record is fold 1": lambda b: b["g13"].update(fold=1),
    "g13_record belongs to run 'other'": lambda b: b["g13"].update(run_id="other"),
    "training-population identities": lambda b: b["g13_corrected"]["population"]["checks"][
        "train_label_fingerprint"].update(got="000000000000"),
    "glass fold-0 training rows differ": lambda b: b["g3"]["arms"]["glass"].update(
        train_label_fingerprint="000000000000"),
}


@pytest.mark.parametrize("match", list(RUN_FAILURES))
def test_every_record_belongs_to_the_run_fold_0_and_one_population(tmp_path, match):
    _refused(tmp_path / "run", match, bodies=_edit(RUN_FAILURES[match]))


def test_records_outside_the_run_directory_are_refused(tmp_path):
    good = S.authorization_files(tmp_path / "a")
    rec = json.loads(good.read_text(encoding="utf-8"))
    elsewhere = shutil.copy(rec["g3_record"], tmp_path / "g3_elsewhere.json")
    _refused(tmp_path / "b", "g3_record is not in run 'b'", g3_record=str(elsewhere),
             g3_record_sha256=R.sha256_file(elsewhere))
    _refused(tmp_path / "c", "g13_record is not in the directory of its run",
             g13_record_run_id="s2a_other")


def test_each_run_needs_its_passed_phase0(tmp_path):
    _refused(tmp_path / "a", "run 'a' has no passed Phase 0 record", phase0=False)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "phase0.json").write_text(json.dumps({"passed": True}))
    _refused(tmp_path / "b", "original G13 run 'b__s2a' has no passed Phase 0", phase0=False)
