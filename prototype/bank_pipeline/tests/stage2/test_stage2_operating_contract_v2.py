"""
Schema-2 Stage 2 operating contract (PR 37 Commit 4, Step 4A).

Contract
1. The Step 4A working profile loads from its JSON record with the approved
   working values, the lexicographic-coverage objective and the metric set, and
   is marked working, not frozen. Its content identity is pinned.
2. Every schema-2 field is required; out-of-range, vacuous and inconsistent
   settings are refused, as are malformed records.
3. Provenance is structured and complete. No value may be derived from an arm's
   realised performance, and no rationale may cite the historical GLASS-derived
   pathway, though the record may describe it as history. Exposure that
   informed the decisions must be disclosed as such. Test-derived evidence is
   refused. Numerical equality with a historical value is not by itself a
   violation. Recorded schema-2 artifact provenance must match the contract.
4. The loader reads one file and nothing else.
5. The RF adapter takes every value, the support floors included, from a
   schema-2 contract; the Commit 3 schema-1 path is unchanged.
6. Router defaults and the baseline-pinned code are unchanged.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import shared.stage2.contract as contract_mod
from blackbox_pipeline.models.rf_router import OperatingConstraints
from blackbox_pipeline.models.rf_router.config import ELIGIBILITY_GLOBAL_T1, RFRouterConfig
from blackbox_pipeline.models.rf_router.contract import (
    NO_COUNT_CAP, check_rf_constraints_follow_contract, rf_constraints_from_contract,
    rf_constraints_from_contract_v2,
)
from glass_pipeline.glass_router.core.config import INNER_RANKER_OUTER_RF, GlassRouterConfig
from shared.stage2.contract import (
    CONSTRAINT_SOURCE_SHARED_CONTRACT, METRIC_DEFINITIONS_ID, OBJECTIVE_LEXICOGRAPHIC_COVERAGE,
    SCHEMA_VERSION_2, STATUS_FROZEN, STATUS_WORKING, ContractViolation, SharedOperatingContract,
    SharedOperatingContractV2, check_contract_provenance, constraint_provenance,
    load_operating_contract, working_profile_path,
)

BANK_ROOT = Path(__file__).resolve().parents[2]
BASELINE = BANK_ROOT / "research_logs" / "baselines" / "pr37_stage2_pre_correction_v1"

#: Content identity of the Step 4A working profile as reviewed. Any edit to the
#: record changes it; update it here deliberately, in the same change.
WORKING_PROFILE_FINGERPRINT = "a6ca7462d4773005"


def _record() -> dict:
    return json.loads(working_profile_path().read_text(encoding="utf-8"))


def _v2(**overrides) -> SharedOperatingContractV2:
    return SharedOperatingContractV2.from_dict({**_record(), **overrides})


def _with_provenance(mutate) -> dict:
    rec = _record()
    mutate(rec["provenance_record"])
    return rec


def _frozen_record() -> dict:
    rec = _record()
    rec["status"] = STATUS_FROZEN
    rec["contract_version"] = "1.0.0"
    rec["provenance_record"]["approval"].update(state=STATUS_FROZEN, approved_by="reviewer",
                                                approved_on="2026-11-05")
    return rec


# ----------------------------------------------------------------------
# 1. the working profile
# ----------------------------------------------------------------------
def test_the_working_profile_loads_with_the_approved_working_values():
    c = load_operating_contract(working_profile_path())
    assert isinstance(c, SharedOperatingContractV2) and isinstance(c, SharedOperatingContract)
    assert (c.min_pass1_npv, c.max_pass1_leakage_rate, c.max_pass1_leakage_count,
            c.min_pass2_precision, c.min_pass1_support, c.min_pass2_support) == (
        0.95, 0.15, None, 0.30, 125, 75)
    assert c.schema_version == SCHEMA_VERSION_2
    assert c.objective == OBJECTIVE_LEXICOGRAPHIC_COVERAGE
    assert "max_pass1_routed_rows,then_max_pass2_routed_rows" in c.objective
    assert c.metric_definitions == METRIC_DEFINITIONS_ID
    assert c.recall_denominator == "global"


def test_the_working_profile_is_marked_working_and_not_frozen():
    c = load_operating_contract(working_profile_path())
    assert c.status == STATUS_WORKING and c.contract_version.endswith("-working")
    approval = c.provenance_record["approval"]
    assert approval["state"] == STATUS_WORKING
    assert approval["approved_by"] is None and approval["approved_on"] is None
    with pytest.raises(ContractViolation, match="'frozen' is required"):
        load_operating_contract(working_profile_path(), require_status=STATUS_FROZEN)
    assert "working" in working_profile_path().name


def test_the_working_profile_identity_is_pinned_and_deterministic(tmp_path):
    c = load_operating_contract(working_profile_path(),
                                expected_fingerprint=WORKING_PROFILE_FINGERPRINT)
    assert c.fingerprint() == WORKING_PROFILE_FINGERPRINT
    assert c.content_sha256()[:16] == c.fingerprint() and len(c.content_sha256()) == 64
    again = SharedOperatingContractV2.from_dict(c.to_dict())
    assert again == c and again.fingerprint() == c.fingerprint()
    # Key order and layout of the file do not move the identity.
    shuffled = tmp_path / "reordered.json"
    shuffled.write_text(json.dumps(dict(reversed(list(c.to_dict().items()))), indent=None),
                        encoding="utf-8")
    assert load_operating_contract(shuffled).fingerprint() == c.fingerprint()
    with pytest.raises(ContractViolation, match="does not match the expected"):
        load_operating_contract(working_profile_path(), expected_fingerprint="0" * 16)


def test_the_fingerprint_covers_every_field():
    base = _v2().fingerprint()
    changes = {
        "min_pass1_npv": 0.951, "max_pass1_leakage_rate": 0.149, "max_pass1_leakage_count": 556,
        "min_pass2_precision": 0.301, "min_pass1_support": 126, "min_pass2_support": 76,
        "contract_version": "0.1.1-working", "provenance": "another summary",
    }
    for name, value in changes.items():
        assert _v2(**{name: value}).fingerprint() != base, name
    rec = _with_provenance(lambda p: p["limitations"].append("one more limitation"))
    assert SharedOperatingContractV2.from_dict(rec).fingerprint() != base


def test_the_contract_is_immutable_and_to_dict_is_a_copy():
    c = _v2()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.min_pass1_npv = 0.5
    with pytest.raises(TypeError):
        c.provenance_record["derived_from_arm_realised_performance"] = True
    d = c.to_dict()
    d["provenance_record"]["limitations"].clear()
    assert c.to_dict()["provenance_record"]["limitations"]
    assert hash(c) == hash(_v2())


# ----------------------------------------------------------------------
# 2. required fields, ranges and consistency
# ----------------------------------------------------------------------
@pytest.mark.parametrize("missing", [f.name for f in dataclasses.fields(SharedOperatingContractV2)])
def test_every_schema_2_field_is_required(missing):
    rec = _record()
    del rec[missing]
    with pytest.raises(ContractViolation, match="missing"):
        SharedOperatingContractV2.from_dict(rec)


def test_unknown_fields_are_refused():
    with pytest.raises(ContractViolation, match="unknown"):
        SharedOperatingContractV2.from_dict({**_record(), "min_pass2_recall": 0.5})


@pytest.mark.parametrize("field, value, match", [
    ("min_pass1_npv", 0.0, "above 0"),
    ("min_pass1_npv", 1.2, r"\[0, 1\]"),
    ("min_pass2_precision", 0.0, "above 0"),
    ("min_pass2_precision", float("nan"), r"\[0, 1\]"),
    ("max_pass1_leakage_rate", 1.0, "below 1"),
    ("max_pass1_leakage_rate", -0.1, r"\[0, 1\]"),
    ("max_pass1_leakage_rate", True, r"\[0, 1\]"),
    ("max_pass1_leakage_count", -1, "max_pass1_leakage_count"),
    ("max_pass1_leakage_count", 2.5, "max_pass1_leakage_count"),
    ("min_pass1_support", 0, "integer >= 1"),
    ("min_pass2_support", True, "integer >= 1"),
    ("min_pass2_support", 75.0, "integer >= 1"),
    ("schema_version", "stage2-operating-contract/1", "schema_version"),
    ("objective", "joint_total_coverage", "objective"),
    ("metric_definitions", "other/1", "metric_definitions"),
    ("status", "draft", "status"),
    ("contract_version", "v1", "MAJOR.MINOR.PATCH"),
    ("contract_version", "0.1.0", "pre-release label"),
    ("recall_denominator", "remainder", "recall_denominator"),
    ("contract_id", " ", "contract_id"),
])
def test_out_of_range_vacuous_and_inconsistent_settings_are_refused(field, value, match):
    with pytest.raises(ContractViolation, match=match):
        _v2(**{field: value})


def test_a_frozen_record_needs_an_unlabelled_version_and_an_approval():
    frozen = SharedOperatingContractV2.from_dict(_frozen_record())
    assert frozen.status == STATUS_FROZEN
    rec = _frozen_record()
    rec["contract_version"] = "1.0.0-rc1"
    with pytest.raises(ContractViolation, match="must not carry a pre-release label"):
        SharedOperatingContractV2.from_dict(rec)
    for key, value, match in [("approved_by", None, "approved_by"),
                              ("approved_on", "5 Nov 2026", "YYYY-MM-DD"),
                              ("state", STATUS_WORKING, "does not match")]:
        rec = _frozen_record()
        rec["provenance_record"]["approval"][key] = value
        with pytest.raises(ContractViolation, match=match):
            SharedOperatingContractV2.from_dict(rec)
    rec = _record()
    rec["provenance_record"]["approval"]["approved_by"] = "someone"
    with pytest.raises(ContractViolation, match="working contract is not approved"):
        SharedOperatingContractV2.from_dict(rec)


def test_malformed_records_are_refused(tmp_path):
    text = working_profile_path().read_text(encoding="utf-8")
    dup = tmp_path / "dup.json"
    dup.write_text(text.replace('"min_pass2_support": 75,', '"min_pass2_support": 75,\n'
                                '  "min_pass2_support": 50,', 1), encoding="utf-8")
    with pytest.raises(ContractViolation, match="repeats the key"):
        load_operating_contract(dup)
    nan = tmp_path / "nan.json"
    nan.write_text(text.replace('"min_pass1_npv": 0.95', '"min_pass1_npv": NaN', 1),
                   encoding="utf-8")
    with pytest.raises(ContractViolation, match="non-finite"):
        load_operating_contract(nan)
    broken = tmp_path / "broken.json"
    broken.write_text(text[:-10], encoding="utf-8")
    with pytest.raises(ContractViolation, match="not valid JSON"):
        load_operating_contract(broken)
    with pytest.raises(ValueError, match="require_status"):
        load_operating_contract(working_profile_path(), require_status="approved")


# ----------------------------------------------------------------------
# 3. provenance
# ----------------------------------------------------------------------
def test_the_working_profile_discloses_its_exposure_and_limits():
    p = _v2().provenance_record
    assert p["exposure"]["historical_oof_examined"] is True
    assert p["exposure"]["test_split_examined"] is True
    assert p["derived_from_arm_realised_performance"] is False
    assert p["historical_exposure_informed_decisions"] is True
    assert any("match_glass_oof" in d for d in p["exposure"]["details"])     # described as history
    assert all(e["split"] != "test" for e in p["evidence"])
    assert any(e["role"] == "feasibility_screen" for e in p["evidence"])
    text = " ".join(p["limitations"]) + " " + " ".join(p["feasibility_screening"]["limitations"])
    assert "GLASS" in text and "preflight" in text and "not independently established" in text
    assert set(p["rationale"]) == set(contract_mod.CONTRACT_SETTINGS)
    assert p["rationale"]["min_pass1_npv"]["basis"] == "protective_guard"
    assert p["rationale"]["max_pass1_leakage_count"]["basis"] == "structural_argument"
    assert {p["rationale"][s]["basis"] for s in ("min_pass1_support", "min_pass2_support")} == {
        "historical_configuration_anchor"}


@pytest.mark.parametrize("mutate, match", [
    (lambda p: p.pop("exposure"), "missing"),
    (lambda p: p.update(extra_note="x"), "unknown keys"),
    (lambda p: p["rationale"].pop("min_pass2_support"), "missing"),
    (lambda p: p["rationale"]["min_pass1_npv"].update(basis="arm_realised_performance"),
     "not an allowed basis"),
    (lambda p: p["rationale"]["min_pass2_precision"].update(text=" "), "text must be"),
    (lambda p: p.update(evidence=[]), "evidence must be a non-empty list"),
    (lambda p: p["evidence"][0].update(kind="realised_metrics"), "kind must be one of"),
    (lambda p: p["evidence"][1].update(sha256="ABC"), "sha256"),
    (lambda p: p["evidence"][1].update(id=p["evidence"][0]["id"]), "not unique"),
    (lambda p: p["exposure"].update(test_split_examined="yes"), "true or false"),
    (lambda p: p.pop("historical_exposure_informed_decisions"), "missing"),
    (lambda p: p.update(historical_exposure_informed_decisions="yes"), "true or false"),
    (lambda p: p["exposure"].update(details=[]), "details"),
    (lambda p: p["feasibility_screening"].update(limitations=[]), "limitations"),
    (lambda p: p.update(limitations=[]), "limitations"),
    (lambda p: p.update(relation_to_pr38=""), "relation_to_pr38"),
])
def test_incomplete_provenance_is_refused(mutate, match):
    with pytest.raises(ContractViolation, match=match):
        SharedOperatingContractV2.from_dict(_with_provenance(mutate))


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_performance_derived_provenance_is_refused(value):
    rec = _with_provenance(lambda p: p.update(derived_from_arm_realised_performance=value))
    with pytest.raises(ContractViolation, match="realised performance"):
        SharedOperatingContractV2.from_dict(rec)


def test_exposure_that_informed_the_decisions_must_be_disclosed_as_such():
    rec = _with_provenance(lambda p: p.update(historical_exposure_informed_decisions=False))
    with pytest.raises(ContractViolation, match="may not claim that the exposure"):
        SharedOperatingContractV2.from_dict(rec)

    def unexposed(p):
        p["exposure"].update(historical_oof_examined=False)
        p["historical_exposure_informed_decisions"] = False
    SharedOperatingContractV2.from_dict(_with_provenance(unexposed))


@pytest.mark.parametrize("where", ["min_pass1_npv", "max_pass1_leakage_rate", "min_pass2_support"])
def test_no_rationale_may_cite_the_historical_glass_derived_pathway(where):
    text = "set to the match_glass_oof value"
    rec = _with_provenance(lambda p: p["rationale"][where].update(text=text))
    with pytest.raises(ContractViolation, match="no setting may be justified"):
        SharedOperatingContractV2.from_dict(rec)


def test_the_summary_may_not_name_the_historical_pathway():
    """The Commit 3 rule on the provenance summary, inherited unchanged."""
    with pytest.raises(ContractViolation, match="match_glass_oof"):
        _v2(provenance="values copied from match_glass_oof metrics")


@pytest.mark.parametrize("mutate", [
    lambda p: p["evidence"][0].update(
        description="Historical RF constraints came from match_glass_oof (D13)."),
    lambda p: p["exposure"]["details"].append("The historical source was match_glass_oof."),
    lambda p: p["limitations"].append("Historical artifacts were fitted under match_glass_oof."),
    lambda p: p["feasibility_screening"]["limitations"].append("t1 came from match_glass_oof."),
])
def test_the_historical_pathway_may_be_described_as_history(mutate):
    SharedOperatingContractV2.from_dict(_with_provenance(mutate))


@pytest.mark.parametrize("mutate", [
    lambda p: p["evidence"][1].update(split="test"),
    lambda p: p["evidence"][1]["fields_used"].append("test_metrics"),
    lambda p: p["evidence"][1]["fields_used"].append("extra.test_proba"),
    lambda p: p["evidence"][2]["fields_used"].append("holdout_labels"),
])
def test_test_derived_evidence_is_refused(mutate):
    with pytest.raises(ContractViolation, match="test-derived"):
        SharedOperatingContractV2.from_dict(_with_provenance(mutate))


def test_a_field_merely_containing_the_letters_test_is_not_refused():
    rec = _with_provenance(lambda p: p["evidence"][1]["fields_used"].append("fit_report.latest"))
    SharedOperatingContractV2.from_dict(rec)


def test_numerical_equality_with_a_historical_value_is_not_a_provenance_violation():
    """The historical GLASS-derived RF constraints, entered with their own rationale."""
    c = _v2(min_pass1_npv=0.9553485943075986, max_pass1_leakage_rate=0.13820043103448276,
            min_pass2_precision=0.2759801606046292)
    assert c.min_pass2_precision == 0.2759801606046292


# ----------------------------------------------------------------------
# 4. the loader reads one file and nothing else
# ----------------------------------------------------------------------
def test_the_loader_takes_a_path_and_reads_nothing_else(tmp_path):
    params = inspect.signature(load_operating_contract).parameters
    assert list(params) == ["path", "expected_fingerprint", "require_status"]
    copy_path = tmp_path / "elsewhere.json"           # no repository, no artifacts nearby
    copy_path.write_text(working_profile_path().read_text(encoding="utf-8"), encoding="utf-8")
    assert load_operating_contract(copy_path).fingerprint() == WORKING_PROFILE_FINGERPRINT


def test_the_contract_module_cannot_read_artifacts_or_import_an_arm():
    tree = ast.parse(Path(contract_mod.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    banned = ("glass_pipeline", "blackbox_pipeline", "joblib", "pickle", "pandas", "numpy",
              "shared.stage2.diagnostics", "shared.stage4")
    assert not [m for m in names if m.startswith(banned)]


def test_values_live_only_in_versioned_records_and_every_record_loads():
    """No contract instance in code (Commit 3 invariant, kept); every record is valid."""
    assert not [v for v in vars(contract_mod).values() if isinstance(v, SharedOperatingContract)]
    records = sorted(contract_mod.CONTRACTS_DIR.glob("*.json"))
    assert working_profile_path() in records
    for path in records:
        assert isinstance(load_operating_contract(path), SharedOperatingContractV2), path.name


# ----------------------------------------------------------------------
# 5. pathways and the RF adapter
# ----------------------------------------------------------------------
def test_schema_1_is_unchanged_and_schema_2_extends_its_provenance_block():
    assert [f.name for f in dataclasses.fields(SharedOperatingContract)] == [
        "contract_id", "min_pass1_npv", "max_pass1_leakage_rate", "max_pass1_leakage_count",
        "min_pass2_precision", "provenance", "recall_denominator"]
    v1 = SharedOperatingContract(contract_id="synthetic", min_pass1_npv=0.9,
                                 max_pass1_leakage_rate=0.2, max_pass1_leakage_count=None,
                                 min_pass2_precision=0.4, provenance="unit-test fixture")
    assert set(constraint_provenance(v1)) == {"constraint_source", "contract_id",
                                              "contract_fingerprint"}
    c = _v2()
    block = constraint_provenance(c)
    assert block == {"constraint_source": CONSTRAINT_SOURCE_SHARED_CONTRACT,
                     "contract_id": "pr37-stage2-primary", "contract_fingerprint": c.fingerprint(),
                     "schema_version": SCHEMA_VERSION_2, "contract_version": "0.1.0-working",
                     "contract_status": STATUS_WORKING}
    check_contract_provenance(block, c)
    with pytest.raises(ContractViolation, match="contract_fingerprint"):
        check_contract_provenance(block, _v2(min_pass2_precision=0.31))


@pytest.mark.parametrize("key", ["contract_id", "schema_version", "contract_version",
                                 "contract_status"])
def test_recorded_schema_2_metadata_must_be_present_and_match(key):
    c = _v2()
    block = constraint_provenance(c)
    missing = {k: v for k, v in block.items() if k != key}
    with pytest.raises(ContractViolation, match=f"{key} missing"):
        check_contract_provenance(missing, c)
    wrong = {**block, key: "pr37-stage2-other" if key == "contract_id" else "other"}
    with pytest.raises(ContractViolation, match=f"{key}='"):
        check_contract_provenance(wrong, c)


def test_schema_1_artifact_provenance_is_checked_as_in_commit_3():
    v1 = SharedOperatingContract(contract_id="synthetic", min_pass1_npv=0.9,
                                 max_pass1_leakage_rate=0.2, max_pass1_leakage_count=None,
                                 min_pass2_precision=0.4, provenance="unit-test fixture")
    block = constraint_provenance(v1)
    check_contract_provenance(block, v1)
    check_contract_provenance({k: block[k] for k in ("constraint_source",
                                                     "contract_fingerprint")}, v1)
    check_contract_provenance({**block, "contract_id": "anything", "contract_status": "x"}, v1)


def test_the_rf_adapter_takes_every_value_from_a_schema_2_contract():
    c = _v2()
    rf = rf_constraints_from_contract_v2(c)
    assert rf == OperatingConstraints(
        min_precision_not_subscribe=0.95, max_subscriber_leakage_rate=0.15,
        max_subscriber_leakage_absolute=NO_COUNT_CAP, min_pass1_coverage=None,
        min_precision_subscribe=0.30, min_recall_subscribe=None, recall_denominator="global",
        min_band_support_pass1=125, min_band_support_pass2=75)
    assert rf.require_band_ordering is True                 # unchanged default
    check_rf_constraints_follow_contract(rf, c)
    check_rf_constraints_follow_contract(rf.to_dict(), c, extra=constraint_provenance(c))
    assert rf_constraints_from_contract(c, min_band_support_pass1=125,
                                        min_band_support_pass2=75) == rf
    assert list(inspect.signature(rf_constraints_from_contract_v2).parameters) == ["contract"]


def test_a_schema_2_contract_refuses_any_other_support_floors():
    c = _v2()
    with pytest.raises(ContractViolation, match="fixes the routed-set support floors"):
        rf_constraints_from_contract(c, min_band_support_pass1=100, min_band_support_pass2=75)
    drifted = rf_constraints_from_contract_v2(c).to_dict()
    drifted["min_band_support_pass2"] = 50
    with pytest.raises(ContractViolation, match="min_band_support_pass2"):
        check_rf_constraints_follow_contract(drifted, c)
    v1 = SharedOperatingContract(contract_id="synthetic", min_pass1_npv=0.9,
                                 max_pass1_leakage_rate=0.2, max_pass1_leakage_count=None,
                                 min_pass2_precision=0.4, provenance="unit-test fixture")
    with pytest.raises(ContractViolation, match="needs a schema-2 contract"):
        rf_constraints_from_contract_v2(v1)


def test_historical_rf_constraints_never_follow_the_working_contract():
    """The constraints recorded in the frozen baseline (D13) are refused under it."""
    manifest = json.loads((BASELINE / "MANIFEST.json").read_text(encoding="utf-8"))
    historical = manifest["rf_constraint_provenance"]["recorded_constraints"]
    with pytest.raises(ContractViolation, match="do not follow the shared contract"):
        check_rf_constraints_follow_contract(historical, _v2())


def test_the_rf_adapter_reads_nothing_but_the_contract():
    from blackbox_pipeline.models.rf_router import contract as rf_contract
    tree = ast.parse(Path(rf_contract.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [("." * n.level) + (n.module or "") for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom)]
    assert sorted(set(names)) == ["." + "constraints", "__future__", "shared.stage2.contract",
                                  "typing"]


# ----------------------------------------------------------------------
# 6. defaults and pinned code unchanged
# ----------------------------------------------------------------------
def test_router_defaults_are_unchanged():
    oc = {f.name: f.default for f in dataclasses.fields(OperatingConstraints)
          if f.default is not dataclasses.MISSING}
    assert oc == {"min_pass1_coverage": None, "min_precision_subscribe": 0.0,
                  "min_recall_subscribe": None, "recall_denominator": "global",
                  "min_band_support_pass1": 100, "min_band_support_pass2": 50,
                  "require_band_ordering": True}
    rf = {f.name: f.default for f in dataclasses.fields(RFRouterConfig)
          if f.default is not dataclasses.MISSING and not f.name.startswith("_")}
    assert rf["eligibility"] == ELIGIBILITY_GLOBAL_T1 and rf["n_oof_folds"] == 10
    assert rf["remainder_min_size"] == 500 and rf["random_state"] == 42
    g = {f.name: f.default for f in dataclasses.fields(GlassRouterConfig)}
    assert g["pass2_population"] == "full" and g["inner_feature_ranker"] == INNER_RANKER_OUTER_RF
    assert g["max_union_leakage_rate_pass1"] is None


def test_the_baseline_pinned_code_is_unchanged():
    """
    The 15 code files the pre-correction baseline pins (LF-normalised SHA-256).
    PR 38 gate G10 changes tests/support/artifact_audit.py deliberately; record
    that here in the same change, never by deleting this test.
    """
    manifest = json.loads((BASELINE / "MANIFEST.json").read_text(encoding="utf-8"))
    pinned = manifest["code"]["files"]
    assert len(pinned) == 15
    changed = []
    for rel, want in pinned.items():
        data = (BANK_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != want:
            changed.append(rel)
    assert changed == []
