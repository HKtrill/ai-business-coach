"""
PR 39 Step 4A: the Stage 2 reuse step's own integration with PR 38's V7 checks
(upstream_regeneration.checks.stage2_reuse; execution contract section 5.2).

The real ``stage2_reuse`` runs on synthetic routers saved in the formats the
accepted run wrote. Only what needs the owner's artifacts or database is
replaced: the accepted run's ``require_accepted``, its file hashes, its pinned
files (read through the real ``load_pinned_stage2`` from a synthetic run in the
same layout) and the Step 3A recompute. PR 38's own R4 checker
(``acceptance._r4_checks``) runs unchanged.

* R4 receives the artifacts under its own arm keys (``glass``, ``blackbox``)
  and runs every consumer for both arms; V7's R5 check reads its notes.
* The names check reads the GLASS rules as ``ModelSaver`` saves them: a list
  of rule dicts whose ``segment`` holds ``(feature, value)`` conditions.
* A malformed artifact or a checker that raises is a recorded failure, never
  an abort of the step.
"""

from __future__ import annotations

import pytest

import shared.pinned_artifacts as P
from shared.stage2.binning import RF_FEATURES_BINARY
from stage2_crossfit import acceptance as A
from tests.support.stage4_world import ROUTER_FP
from tests.upstream_regeneration.test_pinned_artifacts import _synthetic_run
from upstream_regeneration import checks as C


def _saved_rules():
    """Rules as glass_router.model_saver.ModelSaver._serialize_rules writes them."""
    return [{"rule_id": 1, "pass": "Pass 1", "segment": [(RF_FEATURES_BINARY[0], 1)]},
            {"rule_id": 2, "pass": "Pass 2",
             "segment": [(RF_FEATURES_BINARY[3], 1), (RF_FEATURES_BINARY[7], 0)]}]


@pytest.fixture
def reuse(tmp_path, monkeypatch):
    """``stage2_reuse``'s inputs: a synthetic pinned run and the real-data parts stubbed."""
    w, kw = _synthetic_run(tmp_path)
    real_load = P.load_pinned_stage2
    import shared.stage4 as stage4
    real_build = stage4.build_stage4_inputs
    monkeypatch.setattr(stage4, "build_stage4_inputs",
                        lambda *a, **k: real_build(*a, **{**k, "extra_fingerprints": ROUTER_FP}))
    edits = {}

    def load_synthetic(split):
        s2 = real_load(split, **kw)
        s2.glass_bundle.update(features=list(RF_FEATURES_BINARY), rules=_saved_rules())
        rf = s2.rf_artifact
        rf["feature_names"] = list(RF_FEATURES_BINARY)
        rf["extra"]["oof_protocol"]["pass2_eligibility"] = "nested_t1"     # as the accepted run
        rf["config"] = {**(rf.get("config") or {}), "eligibility": "nested_t1"}
        for edit in edits.values():
            edit(s2)
        return s2

    monkeypatch.setattr(P, "load_pinned_stage2", load_synthetic)

    class Candidate:
        manifest_sha256, glass_path, rf_path = "m" * 64, "g.joblib", "r.joblib"

        def relative(self, p):
            return p

    monkeypatch.setattr(A, "require_accepted", lambda run_id: Candidate())
    monkeypatch.setattr(C, "_file_manifest_checks", lambda gate: {"stub": "file hashes"})
    monkeypatch.setattr(C, "_binning_recheck", lambda gate: {"stub": "Step 3A recompute"})
    return w, edits


def test_stage2_reuse_runs_pr38s_r4_checks_for_both_arms_under_their_own_keys(reuse):
    w, _ = reuse
    gate = C.Gate("A4.2")
    out = C.stage2_reuse(gate, w["GS"])
    assert gate.passed, gate.failed()
    assert set(out["r4_notes"]) == {"glass", "blackbox"}
    names = [c["check"] for c in gate.checks]
    for arm in ("glass", "blackbox"):                         # every R4 consumer, both arms
        for consumer in ("router_stream", "meta-EBM router evidence", "meta-XGB feature block"):
            assert any(n.startswith(f"R4 {consumer} [{arm}] ") for n in names), (consumer, arm)
    assert not [n for n in names if n.endswith(" ran")]       # no consumer failed to run
    assert "V7 R5 at Stage 4: pass2_eligibility nested_t1 on both sides of the RF Router" in names
    assert out["names"]["glass_rule_features"] == sorted(
        {RF_FEATURES_BINARY[0], RF_FEATURES_BINARY[3], RF_FEATURES_BINARY[7]})


def test_the_r4_call_matches_the_input_contract_pr38s_v7_uses():
    """``acceptance._r4_checks`` reads ``arts[arm]`` for each of its arms."""
    from stage2_crossfit import evidence as E
    assert A._ARMS == (E.ARM_GLASS, E.ARM_RF) == ("glass", "blackbox")
    calls = []
    gate = C.Gate("A4.2")
    real = A._r4_checks
    try:
        A._r4_checks = lambda g, arts, ts: calls.append(list(arts)) or {}
        C.r4_checks(gate, {"bundle": 1}, {"router": 2}, ts=None)
    finally:
        A._r4_checks = real
    assert calls == [list(A._ARMS)]
    assert gate.failed() == ["V7 R5 at Stage 4: pass2_eligibility nested_t1 on both sides of "
                             "the RF Router"]                    # no notes: recorded, not raised


@pytest.mark.parametrize("edit, failed", [
    (lambda s2: s2.glass_bundle.update(rules={"pass1_rules": [], "pass2_rules": []}),
     "every feature the accepted GLASS rules use is a current bin"),
    (lambda s2: s2.glass_bundle["rules"][1].update(segment=[("dow_month_encoded_hot", 1)]),
     "every feature the accepted GLASS rules use is a current bin"),
    (lambda s2: s2.rf_artifact.update(feature_names=list(RF_FEATURES_BINARY)[::-1]),
     "the RF Router's feature_names = RF_FEATURES_BINARY"),
    (lambda s2: s2.rf_artifact["config"].update(eligibility="global_t1"),
     "V7 R5 at Stage 4: pass2_eligibility nested_t1"),
])
def test_a_departure_in_the_accepted_routers_is_a_recorded_failure(reuse, edit, failed):
    w, edits = reuse
    edits["case"] = edit
    gate = C.Gate("A4.2")
    C.stage2_reuse(gate, w["GS"])                             # never raises
    assert any(failed in f for f in gate.failed()), gate.failed()


def test_a_raising_r4_checker_is_recorded_as_v7_records_it(reuse, monkeypatch):
    w, _ = reuse

    def broken(*a, **k):
        raise ImportError("a Stage 4 consumer moved")

    monkeypatch.setattr(A, "_r4_checks", broken)
    gate = C.Gate("A4.2")
    C.stage2_reuse(gate, w["GS"])
    assert "R4 Stage 4 consumers ran" in gate.failed()
    assert C.decide({"evidence": {"report_check": {"passed": True}}}, gate.record())["outcome"] \
        == "regenerate"
