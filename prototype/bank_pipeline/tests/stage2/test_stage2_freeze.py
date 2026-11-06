"""
PR 37 Commit 4, Step 4D: the frozen Stage 2 specification.

1. The frozen operating contract (``pr37-stage2-primary`` 1.0.0) has a pinned
   identity and approval, and changes no value of the Step 4A working profile.
2. The frozen PR 38 protocol (``pr38-stage2-evaluation`` 1.0.0) has a pinned
   identity, resolves D-C1 to D-C11, pins the D-C10 fingerprints, binds only
   frozen records, and differs from the working protocol only in its freeze
   fields.
3. Both arms' configurations are frozen, versioned profiles owned by each arm:
   historical values preserved, union obligations and constraints only from the
   contract, F17 primary settings; the RF pathway reads no GLASS artifact.
4. D-C11: a GLASS fold is accepted only with selection reports that correspond
   exactly to the required selections.

Synthetic inputs and version-controlled records only; nothing is fitted or read
from an artifact. ``tests/realdata/test_stage2_freeze_artifacts.py`` re-verifies
the pinned values on the real artifacts.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

import shared.stage2.contract as contract_mod
import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import profile as rf_profile
from glass_pipeline.glass_router import profile as glass_profile
from glass_pipeline.glass_router.contract import glass_fit_outcome
from shared.stage2.arm_profile import ArmProfile, load_arm_profile
from shared.stage2.contract import (
    ContractViolation,
    load_frozen_contract,
    load_operating_contract,
    working_profile_path,
)
from shared.stage2.obligations import PassCounts, check_pass1_fit, check_pass2_fit

BANK_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = BANK_ROOT / "research_logs/baselines/pr37_stage2_pre_correction_v1/MANIFEST.json"

#: Pinned identities of the Step 4D frozen records.
FROZEN_CONTRACT_FINGERPRINT = "0817825b9eecbd11"
FROZEN_PROTOCOL_FINGERPRINT = "245a1a68fb68cf4e"
GLASS_PROFILE_FINGERPRINT = "198731d5cdf263f4"
RF_PROFILE_FINGERPRINT = "4e3f9c07e6ecc8ad"
#: The working records they freeze.
WORKING_CONTRACT_FINGERPRINT = "a6ca7462d4773005"
WORKING_PROTOCOL_FINGERPRINT = "161c6c589696b859"
#: D-C10: computed by the read-only audit of the pinned artifacts.
LR_FLAG_FINGERPRINT = "64d511516d69e73e"
EBM_FLAG_FINGERPRINT = "907a7d2d1cad6908"

HISTORICAL_RF_PARAMS = {"n_estimators": 650, "max_depth": 6, "min_samples_leaf": 12,
                        "max_features": 0.39989815740885837,
                        "minority_weight": 1.3001084911026792, "class_weight": "balanced"}


#: The formal Step 4D approval recorded in all four frozen records.
APPROVER, APPROVED_ON = "Phillip Harris", "2026-11-05"


# Every frozen record is read through its strict loader (status, pinned fingerprint,
# confirmed approval): there is no review or override switch.
@pytest.fixture(scope="module")
def contract():
    return load_frozen_contract()


@pytest.fixture(scope="module")
def protocol():
    return ep.load_frozen_protocol()


@pytest.fixture(scope="module")
def glass_prof():
    return glass_profile.load_glass_primary_profile()


@pytest.fixture(scope="module")
def rf_prof():
    return rf_profile.load_rf_primary_profile()


@pytest.fixture(scope="module")
def working_protocol():
    return ep.load_evaluation_protocol(ep.working_protocol_path())


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _leaves(obj, prefix=""):
    """{dotted path: leaf value} of a JSON record (lists as one leaf)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: obj}


# ============================================================================
# 1. the frozen operating contract
# ============================================================================
def test_the_frozen_contract_loads_with_its_pinned_identity(contract):
    assert contract.fingerprint() == FROZEN_CONTRACT_FINGERPRINT
    assert contract_mod.FROZEN_PROFILE_FINGERPRINT == FROZEN_CONTRACT_FINGERPRINT
    assert (contract.contract_id, contract.contract_version, contract.status) == (
        "pr37-stage2-primary", "1.0.0", "frozen")
    approval = contract.provenance_record["approval"]
    assert (approval["state"], approval["approved_by"], approval["approved_on"]) == (
        "frozen", APPROVER, APPROVED_ON)
    assert contract_mod.approval_confirmed(approval)


def test_the_freeze_changes_no_value_of_the_working_profile(contract):
    working = load_operating_contract(working_profile_path())
    assert working.fingerprint() == WORKING_CONTRACT_FINGERPRINT       # kept, unchanged
    same = ("contract_id", "schema_version", "objective", "metric_definitions",
            "min_pass1_npv", "max_pass1_leakage_rate", "max_pass1_leakage_count",
            "min_pass2_precision", "min_pass1_support", "min_pass2_support",
            "recall_denominator")
    assert {k: getattr(contract, k) for k in same} == {k: getattr(working, k) for k in same}
    assert (contract.min_pass1_npv, contract.max_pass1_leakage_rate,
            contract.max_pass1_leakage_count, contract.min_pass2_precision,
            contract.min_pass1_support, contract.min_pass2_support) == (0.95, 0.15, None, 0.3,
                                                                       125, 75)
    assert contract.objective == contract_mod.OBJECTIVE_LEXICOGRAPHIC_COVERAGE
    differ = {f.name for f in dataclasses.fields(contract)
              if getattr(contract, f.name) != getattr(working, f.name)}
    assert differ == {"contract_version", "status", "provenance", "provenance_record"}
    bases = {k: v["basis"] for k, v in contract.provenance_record["rationale"].items()}
    assert bases == {k: v["basis"] for k, v in working.provenance_record["rationale"].items()}
    assert contract.provenance_record["derived_from_arm_realised_performance"] is False
    assert contract.provenance_record["historical_exposure_informed_decisions"] is True


def test_only_the_pinned_frozen_record_loads_as_the_frozen_contract(tmp_path):
    with pytest.raises(ContractViolation, match="status is 'working'"):
        load_frozen_contract(working_profile_path())
    rec = _json(contract_mod.frozen_profile_path())
    rec["min_pass2_support"] = 74
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ContractViolation, match="does not match the expected"):
        load_frozen_contract(edited)
    rec = _json(contract_mod.frozen_profile_path())
    rec["provenance_record"]["approval"]["approved_by"] = None
    with pytest.raises(ContractViolation, match="approved_by"):
        contract_mod.SharedOperatingContractV2.from_dict(rec)
    rec = _json(contract_mod.frozen_profile_path())
    rec["contract_version"] = "1.0.0-rc1"
    with pytest.raises(ContractViolation, match="pre-release label"):
        contract_mod.SharedOperatingContractV2.from_dict(rec)


# ============================================================================
# 2. the frozen PR 38 protocol
# ============================================================================
def test_the_frozen_protocol_loads_with_its_pinned_identity(protocol, working_protocol):
    assert protocol.fingerprint() == FROZEN_PROTOCOL_FINGERPRINT
    assert ep.FROZEN_PROTOCOL_FINGERPRINT == FROZEN_PROTOCOL_FINGERPRINT
    assert (protocol.protocol_id, protocol.version, protocol.status) == (
        "pr38-stage2-evaluation", "1.0.0", "frozen")
    assert protocol["pending_approvals"] == ()
    assert ep.require_frozen(protocol) is protocol and ep.is_pinned_frozen(protocol)
    assert (protocol["freeze"]["approved_by"], protocol["freeze"]["approved_on"]) == (
        APPROVER, APPROVED_ON)
    assert working_protocol.fingerprint() == WORKING_PROTOCOL_FINGERPRINT   # kept, unchanged
    with pytest.raises(ep.ProtocolError, match="only a frozen protocol"):
        ep.require_frozen(working_protocol)
    with pytest.raises(ep.ProtocolError, match=f"{WORKING_PROTOCOL_FINGERPRINT} != expected"):
        ep.load_frozen_protocol(ep.working_protocol_path())


def test_the_freeze_changes_only_the_freeze_fields():
    """Every methodological setting is the working protocol's, verbatim."""
    working = _leaves(_json(ep.working_protocol_path()))
    frozen = _leaves(_json(ep.frozen_protocol_path()))
    changed = {k for k in set(working) | set(frozen) if working.get(k, "<absent>")
               != frozen.get(k, "<absent>")}
    allowed = {"protocol_version", "status", "purpose", "hierarchy.statement_rule.status",
               "upstream.flag_fingerprints.lr.in_stage_f2_cv",
               "upstream.flag_fingerprints.ebm.nested_f2", "upstream.flag_fingerprints_status",
               "shortfall.materiality.status", "complementarity.status",
               "fit_status.glass_selection_reports", "pending_approvals"}
    allowed |= {k for k in changed if k.startswith(("contract.", "arm_profiles.", "freeze."))}
    assert changed <= allowed, sorted(changed - allowed)
    assert frozen["fit_status.glass_selection_reports"] == "required"
    assert "fit_status.glass_selection_reports" not in working


def test_d_c10_pins_the_audited_flag_fingerprints(protocol):
    up = protocol["upstream"]
    assert dict(up["flag_fingerprints"]) == {"lr.in_stage_f2_cv": LR_FLAG_FINGERPRINT,
                                             "ebm.nested_f2": EBM_FLAG_FINGERPRINT}
    assert dict(up["expected_flags"]) == {"lr.in_stage_f2_cv": 7727, "ebm.nested_f2": 6113,
                                          "or": 8001, "and": 5839}
    d10 = next(d for d in protocol["freeze"]["decisions"] if d["id"] == "D-C10")
    assert LR_FLAG_FINGERPRINT in d10["decision"] and EBM_FLAG_FINGERPRINT in d10["decision"]
    rv = protocol["freeze"]["reverification"]
    assert rv["test"] == "tests/realdata/test_stage2_freeze_artifacts.py"
    assert (BANK_ROOT / rv["test"]).is_file() and "voids the freeze" in rv["requirement"]


def test_the_freeze_resolves_every_decision(protocol):
    fz = protocol["freeze"]
    assert tuple(d["id"] for d in fz["decisions"]) == ep.FREEZE_DECISIONS == tuple(
        f"D-C{i}" for i in range(1, 12))
    kinds = {d["id"]: (d["resolution"], d["kind"]) for d in fz["decisions"]}
    assert kinds["D-C1"] == ("approved", "reporting_rule")
    assert kinds["D-C2"] == ("approved", "descriptive_heuristic")       # not a compliance test
    assert kinds["D-C3"] == ("approved", "registered_procedure")
    assert kinds["D-C4"] == ("approved", "reporting_convention")
    assert {kinds[i] for i in ("D-C5", "D-C6", "D-C7")} == {("approved", "registered_criterion")}
    assert kinds["D-C8"] == ("not_required", "optional_diagnostic")
    assert {kinds[i] for i in ("D-C9", "D-C10")} == {("approved", "binding")}
    assert kinds["D-C11"] == ("approved", "registered_procedure")
    text = {d["id"]: d["decision"] + " " + d["rationale"] for d in fz["decisions"]}
    assert "not a compliance test" in text["D-C2"]
    assert "conditional on the fitted" in text["D-C3"]
    assert "judgements" in text["D-C5"] and "pooled in C1 and per fold in C5" in text["D-C5"]
    assert "not required" in text["D-C8"] and "prerequisite" in text["D-C8"]
    assert dict(fz["supersedes"]) == {"protocol_version": "0.1.0-working",
                                      "fingerprint": WORKING_PROTOCOL_FINGERPRINT}


def test_the_frozen_protocol_binds_only_frozen_records(protocol, contract):
    ep.check_contract_binding(protocol, contract)
    assert dict(protocol["contract"]) == {
        "contract_id": "pr37-stage2-primary", "contract_version": "1.0.0",
        "contract_fingerprint": FROZEN_CONTRACT_FINGERPRINT,
        "binding": protocol["contract"]["binding"]}
    with pytest.raises(ep.ProtocolError, match="is not the contract the protocol binds"):
        ep.check_contract_binding(protocol, load_operating_contract(working_profile_path()))
    working = load_operating_contract(working_profile_path())
    rec = _json(ep.frozen_protocol_path())
    rec["contract"].update(contract_version=working.contract_version,
                           contract_fingerprint=working.fingerprint())
    forged = ep.EvaluationProtocol.from_dict(rec)
    with pytest.raises(ep.ProtocolError, match="binds only a frozen contract"):
        ep.check_contract_binding(forged, working)


# ============================================================================
# 2b. the formal approval, and the approval gate
# ============================================================================
def test_the_four_frozen_records_carry_the_same_formal_approval(contract, protocol, glass_prof,
                                                               rf_prof):
    blocks = {"contract": contract.provenance_record["approval"],
              "protocol": protocol["freeze"],
              "glass": glass_prof.approval, "rf": rf_prof.approval}
    for name, block in blocks.items():
        assert (block["approved_by"], block["approved_on"]) == (APPROVER, APPROVED_ON), name
        assert contract_mod.approval_confirmed(block), name
    for name in ("contract", "glass", "rf"):
        assert blocks[name]["state"] == "frozen" and "approved by Phillip Harris" in blocks[
            name]["note"], name
    assert "legacy GLASS Router profile" in glass_prof.approval["note"]
    assert ep.freeze_approval_confirmed(protocol)
    assert glass_prof.approval_confirmed and rf_prof.approval_confirmed


def _unconfirmed_copy(path, tmp_path, block):
    """A copy of a frozen record whose approver is the UNCONFIRMED marker."""
    rec = _json(path)
    block(rec)["approved_by"] = f"{contract_mod.APPROVAL_UNCONFIRMED}: owner approval pending"
    out = tmp_path / Path(path).name
    out.write_text(json.dumps(rec), encoding="utf-8")
    return out, rec


def test_an_unconfirmed_approval_is_refused_by_every_strict_loader(tmp_path, monkeypatch):
    """Only the approval differs: the copy's fingerprint is pinned for the test, so the
    approval gate is what refuses it."""
    refused = "is a review candidate"
    path, rec = _unconfirmed_copy(contract_mod.frozen_profile_path(), tmp_path,
                                  lambda r: r["provenance_record"]["approval"])
    fp = contract_mod.SharedOperatingContractV2.from_dict(rec).fingerprint()
    monkeypatch.setattr(contract_mod, "FROZEN_PROFILE_FINGERPRINT", fp)
    with pytest.raises(ContractViolation, match=refused):
        load_frozen_contract(path)
    path, rec = _unconfirmed_copy(ep.frozen_protocol_path(), tmp_path, lambda r: r["freeze"])
    monkeypatch.setattr(ep, "FROZEN_PROTOCOL_FINGERPRINT",
                        ep.EvaluationProtocol.from_dict(rec).fingerprint())
    with pytest.raises(ep.ProtocolError, match=refused):
        ep.load_frozen_protocol(path)
    with pytest.raises(ep.ProtocolError, match=refused):
        ep.require_frozen(ep.EvaluationProtocol.from_dict(rec))
    assert not ep.is_pinned_frozen(ep.EvaluationProtocol.from_dict(rec))
    for module, loader in ((glass_profile, glass_profile.load_glass_primary_profile),
                           (rf_profile, rf_profile.load_rf_primary_profile)):
        path, rec = _unconfirmed_copy(module.primary_profile_path(), tmp_path,
                                      lambda r: r["approval"])
        monkeypatch.setattr(module, "PRIMARY_PROFILE_FINGERPRINT",
                            ArmProfile.from_dict(rec).fingerprint())
        with pytest.raises(ContractViolation, match=refused):
            loader(path)
        with pytest.raises(ContractViolation, match=refused):
            load_arm_profile(path, arm=rec["arm"], require_confirmed_approval=True)
        assert load_arm_profile(path, arm=rec["arm"]).approval_confirmed is False


def test_the_strict_loaders_have_no_override():
    import inspect
    for fn in (load_frozen_contract, ep.load_frozen_protocol,
               glass_profile.load_glass_primary_profile, rf_profile.load_rf_primary_profile):
        assert list(inspect.signature(fn).parameters) == ["path"], fn.__name__


@pytest.mark.parametrize("approved_by, confirmed", [
    ("UNCONFIRMED: owner approval pending", False), ("unconfirmed", False), ("  ", False),
    (None, False), ("A. Owner", True),
])
def test_only_a_named_approver_confirms_an_approval(approved_by, confirmed, rf_prof):
    assert contract_mod.approval_confirmed({"approved_by": approved_by}) is confirmed
    assert contract_mod.approval_confirmed(None) is False
    if approved_by and approved_by.strip():
        rec = rf_prof.to_dict()
        rec["approval"]["approved_by"] = approved_by
        assert ArmProfile.from_dict(rec).approval_confirmed is confirmed


@pytest.mark.parametrize("edit, match", [
    (lambda r: r.pop("freeze"), "needs 'freeze'"),
    (lambda r: r.pop("arm_profiles"), "needs 'arm_profiles'"),
    (lambda r: r.update(protocol_version="1.0.0-rc1"), "pre-release label"),
    (lambda r: r["fit_status"].update(glass_selection_reports="optional"), "D-C11"),
    (lambda r: r["fit_status"].update(glass_selection_reports="sometimes"), "glass_selection"),
    (lambda r: r["freeze"]["decisions"].pop(), "must resolve"),
    (lambda r: r["freeze"]["decisions"].reverse(), "must resolve"),
    (lambda r: r["freeze"]["decisions"][1].update(kind="compliance_test"), "kind must be"),
    (lambda r: r["freeze"]["decisions"][7].update(resolution="maybe"), "resolution must be"),
    (lambda r: r["freeze"]["decisions"][0].update(rationale=" "), "must be non-empty"),
    (lambda r: r["freeze"].update(approved_on="5 Nov 2026"), "YYYY-MM-DD"),
    (lambda r: r["freeze"].pop("reverification"), "freeze must hold"),
    (lambda r: r["arm_profiles"]["rf"].update(fingerprint="xyz"), "16-hex"),
    (lambda r: r["arm_profiles"].pop("glass"), "exactly the 'glass' and 'rf'"),
    (lambda r: r["upstream"]["flag_fingerprints"].update({"ebm.nested_f2": None}), "must pin"),
    (lambda r: r.update(pending_approvals=["D-C1"]), "pending approvals"),
])
def test_frozen_records_are_validated(edit, match):
    rec = _json(ep.frozen_protocol_path())
    edit(rec)
    errors = ep.validate_protocol_record(rec)
    assert any(match in e for e in errors), errors


def test_a_working_record_cannot_claim_a_freeze():
    rec = _json(ep.working_protocol_path())
    rec["freeze"] = _json(ep.frozen_protocol_path())["freeze"]
    assert any("may not carry a freeze" in e for e in ep.validate_protocol_record(rec))
    rec = _json(ep.working_protocol_path())
    rec["protocol_version"] = "0.2.0"
    assert any("must carry a pre-release label" in e for e in ep.validate_protocol_record(rec))
    assert ep.validate_protocol_record(_json(ep.working_protocol_path())) == []


def test_every_frozen_record_is_version_controlled():
    """No frozen record sits where .gitignore drops model artifacts (``**/models/**/*.json``)."""
    records = [contract_mod.frozen_profile_path(), ep.frozen_protocol_path(),
               glass_profile.primary_profile_path(), rf_profile.primary_profile_path()]
    root = BANK_ROOT.parents[1]
    ignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "**/models/**/*.json" in [line.strip() for line in ignore]
    for path in records:
        rel = path.resolve().relative_to(root)
        assert path.is_file() and "models" not in rel.parts[:-1], rel


def test_every_protocol_record_is_valid():
    for path in sorted(ep.PROTOCOLS_DIR.glob("*.json")):
        assert ep.validate_protocol_record(_json(path)) == [], path.name


# ============================================================================
# 3. the frozen arm profiles
# ============================================================================



def test_both_profiles_have_pinned_identities_bound_by_the_protocol(protocol, glass_prof,
                                                                    rf_prof):
    assert glass_prof.fingerprint() == GLASS_PROFILE_FINGERPRINT
    assert rf_prof.fingerprint() == RF_PROFILE_FINGERPRINT
    assert glass_profile.PRIMARY_PROFILE_FINGERPRINT == GLASS_PROFILE_FINGERPRINT
    assert rf_profile.PRIMARY_PROFILE_FINGERPRINT == RF_PROFILE_FINGERPRINT
    for prof in (glass_prof, rf_prof):
        assert (prof.status, prof.profile_version) == ("frozen", "1.0.0")
        ep.check_arm_profile_binding(protocol, prof)
    rec = glass_prof.to_dict()
    rec["settings"]["router_config"]["max_pass1_rules"] = 9
    with pytest.raises(ep.ProtocolError, match="is not the profile the protocol binds"):
        ep.check_arm_profile_binding(protocol, ArmProfile.from_dict(rec))
    with pytest.raises(ep.ProtocolError, match="expected an ArmProfile"):
        ep.check_arm_profile_binding(protocol, dict(glass_prof.identity()))


def test_the_rf_profile_preserves_the_historical_hyperparameters(rf_prof):
    cfg = rf_prof.settings["router_config"]
    manifest = _json(MANIFEST)["routers"]["blackbox"]["metadata"]
    assert dict(cfg["rf_params"]) == HISTORICAL_RF_PARAMS == manifest["rf_params"]
    assert cfg["pass2_rf_params"] is None
    assert (cfg["n_oof_folds"], cfg["random_state"], cfg["mode"]) == (
        manifest["n_oof_folds"], manifest["random_state"], manifest["mode"])
    sources = {s["role"]: s["sha256"] for s in rf_prof.provenance["historical_sources"]}
    baseline = {s["role"]: s["sha256"] for s in _json(MANIFEST)["sources"]}
    assert sources == {r: baseline[r] for r in ("glass_arm.rf_result", "stage2.rf_router")}


def _notebook_legacy_profile() -> dict:
    """The GLASS notebook's Cell 13 ``_SHARED`` + ``_PROFILES['legacy']`` (read, not run)."""
    nb = _json(BANK_ROOT / "glass_pipeline/glass_cascade.ipynb")
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "_PROFILES = {" in "".join(c["source"]))
    block = src[src.index("_SHARED = dict("):src.index("config = GlassRouterConfig(")]
    namespace: dict = {}
    exec(compile(block, "glass_cascade.ipynb:cell13", "exec"), {"__builtins__": {"dict": dict}},
         namespace)
    return {**namespace["_SHARED"], **namespace["_PROFILES"]["legacy"]}


def test_the_glass_profile_preserves_the_legacy_profile(glass_prof):
    router = dict(glass_prof.settings["router_config"])
    legacy = _notebook_legacy_profile()
    replaced = {"pass2_population", "max_union_leakage_rate_pass1"}   # F17 primary; contract
    f17 = dict(glass_profile.GLASS_F17_PRIMARY)
    assert {k: v for k, v in legacy.items() if k not in replaced} == {
        k: router[k] for k in legacy if k not in replaced}
    assert {k: router[k] for k in f17} == f17
    assert (legacy["pass2_population"], legacy["max_union_leakage_rate_pass1"]) == ("full", 0.14)
    assert "max_union_leakage_rate_pass1" not in router
    unset_by_legacy = set(glass_profile.PROFILE_FIELDS) - set(legacy) - set(f17)
    assert {k: router[k] for k in unset_by_legacy} == {
        "lambda_rf_uncertainty": None, "lambda_rf_misalignment": None}
    history = _json(MANIFEST)["routers"]["glass"]["metadata"]["config_full"]
    assert {k: v for k, v in history.items() if k not in replaced} == {
        k: router[k] for k in history if k not in replaced}
    ranker = glass_profile.ranker_settings()
    assert ranker == glass_profile.ranker_settings_from_profile(glass_prof) == {
        "glass_rf_model": "fold_rf", "rf_params": HISTORICAL_RF_PARAMS}


def test_both_arms_build_their_primary_configs_from_the_frozen_specification(contract,
                                                                             glass_prof,
                                                                             rf_prof):
    g = glass_profile.glass_primary_config()
    r = rf_profile.rf_primary_config()
    assert g == glass_profile.glass_config_from_profile(contract, glass_prof)
    assert r.to_dict() == rf_profile.rf_config_from_profile(contract, rf_prof).to_dict()
    glass_numbers = (g.min_union_npv_pass1, g.max_union_leakage_rate_pass1,
                     g.min_union_precision_pass2, g.min_union_support_pass1,
                     g.min_union_support_pass2)
    c = r.constraints
    rf_numbers = (c.min_precision_not_subscribe, c.max_subscriber_leakage_rate,
                  c.min_precision_subscribe, c.min_band_support_pass1, c.min_band_support_pass2)
    assert glass_numbers == rf_numbers == (0.95, 0.15, 0.3, 125, 75)
    assert g.pass2_union_objective == "rows"
    assert g.contract_provenance["contract_fingerprint"] == FROZEN_CONTRACT_FINGERPRINT
    assert g.contract_provenance["contract_status"] == "frozen"
    base = glass_profile.glass_base_config(glass_prof)
    assert {k: getattr(g, k) for k in glass_profile.PROFILE_FIELDS} == {
        k: getattr(base, k) for k in glass_profile.PROFILE_FIELDS}
    assert r.rf_params == HISTORICAL_RF_PARAMS and r.pass2_rf_params is None
    assert (r.eligibility, r.verbose, r.validate_nested) == ("nested_t1", False, False)
    loud = rf_profile.rf_primary_config(verbose=True, validate_nested=True)
    differ = {k for k, v in loud.to_dict().items() if r.to_dict()[k] != v}
    assert differ == set(rf_profile.RUNTIME_OPTIONS)         # run-time options only
    with pytest.raises(ContractViolation, match="schema-2"):
        glass_profile.glass_config_from_profile(object(), glass_prof)


def _profile_dict(prof, **router_changes):
    rec = prof.to_dict()
    rec["settings"]["router_config"].update(router_changes)
    return rec


def test_the_pr38_entry_points_take_no_injected_records(contract, protocol, glass_prof,
                                                        rf_prof):
    """PR 38 builds its configs only from the strictly loaded frozen records; a config
    built from any other record is refused as the primary."""
    import inspect
    assert list(inspect.signature(glass_profile.glass_primary_config).parameters) == []
    assert list(inspect.signature(glass_profile.ranker_settings).parameters) == []
    assert list(inspect.signature(rf_profile.rf_primary_config).parameters) == list(
        rf_profile.RUNTIME_OPTIONS)
    assert list(inspect.signature(ep.load_frozen_specification).parameters) == []
    spec = ep.load_frozen_specification()
    assert spec.protocol == protocol and spec.contract == contract
    for prof in (glass_prof, rf_prof):
        ep.check_arm_profile_binding(spec.protocol, prof)
    glass_profile.check_glass_primary_config(glass_profile.glass_primary_config())
    rf_profile.check_rf_primary_config(rf_profile.rf_primary_config(verbose=True,
                                                                    validate_nested=True))
    altered = glass_prof.to_dict()
    altered["settings"]["router_config"]["max_pass1_rules"] = 9
    injected = glass_profile.glass_config_from_profile(contract, ArmProfile.from_dict(altered))
    with pytest.raises(ContractViolation, match=r"\['max_pass1_rules'\] differ"):
        glass_profile.check_glass_primary_config(injected)
    with pytest.raises(ContractViolation, match="contract_provenance"):
        glass_profile.check_glass_primary_config(glass_profile.glass_config_from_profile(
            load_operating_contract(working_profile_path()), glass_prof))   # working contract
    altered = rf_prof.to_dict()
    altered["settings"]["router_config"]["rf_params"]["n_estimators"] = 651
    with pytest.raises(ContractViolation, match=r"\['rf_params'\] differ"):
        rf_profile.check_rf_primary_config(rf_profile.rf_config_from_profile(
            contract, ArmProfile.from_dict(altered)))
    with pytest.raises(ContractViolation, match="expected a GlassRouterConfig"):
        glass_profile.check_glass_primary_config(rf_profile.rf_primary_config())


def test_only_the_pinned_frozen_protocol_decides_officially(protocol):
    gates = [ep.GateResult(g, "pass", "evidence") for g in protocol["acceptance"]["structural"]]
    assert ep.structural_acceptance(gates, protocol).official
    rec = _json(ep.frozen_protocol_path())
    rec["purpose"] += " (rebuilt by hand)"
    rebuilt = ep.EvaluationProtocol.from_dict(rec)
    assert ep.require_frozen(rebuilt) is rebuilt and not ep.is_pinned_frozen(rebuilt)
    assert not ep.structural_acceptance(gates, rebuilt).official


def test_profiles_refuse_contract_fields_drift_and_foreign_records(contract, glass_prof,
                                                                   rf_prof, tmp_path):
    with pytest.raises(ContractViolation, match="union obligations come only from the contract"):
        glass_profile.glass_base_config(ArmProfile.from_dict(
            _profile_dict(glass_prof, max_union_leakage_rate_pass1=0.14)))
    with pytest.raises(ContractViolation, match="F17 primary"):
        glass_profile.glass_base_config(ArmProfile.from_dict(
            _profile_dict(glass_prof, pass2_population="full")))
    rec = glass_prof.to_dict()
    rec["settings"]["ranker"]["glass_rf_model"] = "none"
    with pytest.raises(ContractViolation, match="fold_rf"):
        glass_profile.ranker_settings_from_profile(ArmProfile.from_dict(rec))
    with pytest.raises(ContractViolation, match="constraints come only from the contract"):
        rf_profile.rf_config_from_profile(contract, ArmProfile.from_dict(
            _profile_dict(rf_prof, constraints={"min_precision_not_subscribe": 0.9})))
    with pytest.raises(ContractViolation, match="F17 primary"):
        rf_profile.rf_config_from_profile(contract, ArmProfile.from_dict(
            _profile_dict(rf_prof, eligibility="global_t1")))
    bad = _profile_dict(rf_prof)
    bad["settings"]["router_config"]["rf_params"]["max_dept"] = 6
    with pytest.raises(ValueError, match="unknown keys"):
        rf_profile.rf_config_from_profile(contract, ArmProfile.from_dict(bad))
    with pytest.raises(ContractViolation, match="RF arm"):
        rf_profile.rf_config_from_profile(contract, glass_prof)
    with pytest.raises(ContractViolation, match="not the 'rf' arm's"):
        load_arm_profile(glass_profile.primary_profile_path(), arm="rf")
    edited = tmp_path / "rf.json"
    edited.write_text(json.dumps(_profile_dict(rf_prof, n_inner_folds=4)), encoding="utf-8")
    with pytest.raises(ContractViolation, match="does not match the expected"):
        rf_profile.load_rf_primary_profile(edited)


@pytest.mark.parametrize("edit, match", [
    (lambda r: r["approval"].update(approved_by=None), "approved_by"),
    (lambda r: r["approval"].update(state="working"), "does not match status"),
    (lambda r: r.update(profile_version="1.0.0-rc1"), "pre-release label"),
    (lambda r: r.update(arm="xgb"), "arm must be one of"),
    (lambda r: r["provenance"]["historical_sources"][0].update(fields=["test_metrics"]),
     "test-derived field"),
    (lambda r: r["provenance"]["historical_sources"][0].update(sha256="abc"), "64 lowercase"),
    (lambda r: r["provenance"].pop("tuning"), "missing ['tuning']"),
    (lambda r: r.update(settings={}), "settings must be a non-empty object"),
])
def test_arm_profile_records_are_validated(rf_prof, edit, match):
    rec = rf_prof.to_dict()
    edit(rec)
    with pytest.raises(ContractViolation, match=re.escape(match)):
        ArmProfile.from_dict(rec)


def _imports(path) -> list:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [("." * n.level) + (n.module or "") for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom)]
    return names


def test_the_rf_primary_pathway_reads_no_glass_artifact():
    names = _imports(rf_profile.__file__)
    assert sorted(set(names)) == [".config", ".contract", "__future__", "pathlib",
                                  "shared.stage2.arm_profile", "shared.stage2.contract"]
    assert not [m for m in names if "tuning" in m or "artifacts" in m or "glass" in m]
    from blackbox_pipeline.models.rf_router.tuning import legacy
    assert callable(legacy.load_rf_params_from_artifact)    # the historical pathway is kept


def test_the_shared_profile_module_and_the_glass_profile_import_no_other_arm():
    shared = _imports(Path(contract_mod.__file__).with_name("arm_profile.py"))
    assert not [m for m in shared if m.startswith(("glass_pipeline", "blackbox_pipeline",
                                                   "joblib", "pickle", "pandas", "numpy"))]
    glass = _imports(glass_profile.__file__)
    assert not [m for m in glass if m.startswith(("blackbox_pipeline", "joblib", "pickle"))]


# ============================================================================
# 4. D-C11: GLASS selection reports are required, exactly
# ============================================================================
def _report(source="ilp", *, proven=True, ids=("r0",)):
    rep = {"source": source, "selected_rule_ids": list(ids)}
    if source == "ilp":
        rep["lexicographic_guarantee"] = "proven" if proven else "unproven"
    return rep


def _reports(n_inner=5):
    return {"pass1": _report(ids=("p1",)), "pass2": _report(ids=("p2",)),
            "inner_pass1": [_report(ids=(f"i{j}",)) for j in range(n_inner)]}


def _obligation_report(scope, contract):
    if scope == "pass1":
        return check_pass1_fit(PassCounts(population_rows=1000, population_positives=100,
                                          routed_rows=300, routed_positives=5), contract)
    return check_pass2_fit(PassCounts(population_rows=1000, population_positives=150,
                                      routed_rows=200, routed_positives=80), contract)


def _records(reports, contract, protocol):
    """The selection records glass_fit_outcome makes from ``reports``."""
    entries = {"pass1": reports["pass1"], "pass2": reports["pass2"]}
    entries.update({f"inner_pass1[{j}]": r for j, r in enumerate(reports["inner_pass1"])})
    out = []
    for key in ep.required_selection_keys("glass", protocol):
        scope = "pass2" if key == "pass2" else "pass1"
        feasible = entries[key].get("source") in ("ilp", "greedy_fallback")
        report = _obligation_report(scope, contract) if feasible else None
        out.append(ep.SelectionRecord(key, glass_fit_outcome(scope, entries[key], report)))
    return out


def _classify(records, protocol, reports):
    return ep.classify_fold_fit(records, arm="glass", protocol=protocol, selection_reports=reports)


def test_glass_acceptance_needs_its_selection_reports(contract, protocol, working_protocol):
    reports = _reports()
    records = _records(reports, contract, protocol)
    ok = _classify(records, protocol, reports)
    assert ok.status == "valid_optimal" and ok.completeness.complete
    assert ep.FoldLedger.build({"glass": {0: ok}}, protocol).accepted_folds("glass") == (0,)
    assert ep.glass_reports_required(protocol) and not ep.glass_reports_required(working_protocol)
    none = _classify(records, protocol, None)
    assert none.status == "fit_time_invalid"
    assert any("D-C11" in u for u in none.completeness.unsupported)
    assert _classify(records, working_protocol, None).status == "valid_optimal"   # optional there
    with pytest.raises(ep.ProtocolError, match="does not reproduce"):
        ep.FoldLedger.build({"glass": {0: dataclasses.replace(ok, selection_reports=None)}},
                            protocol)                            # reports stripped afterwards


def test_reports_must_correspond_exactly_to_the_required_selections(contract, protocol):
    reports = _reports()
    records = _records(reports, contract, protocol)
    extra = {**reports, "inner_pass1": reports["inner_pass1"] + [_report(ids=("i5",))]}
    fit = _classify(records, protocol, extra)
    assert fit.status == "fit_time_invalid"
    assert "report inner_pass1[5]" in fit.completeness.unexpected
    short = {**reports, "inner_pass1": reports["inner_pass1"][:4]}
    fit = _classify(records, protocol, short)
    assert fit.status == "fit_time_invalid" and "inner_pass1[4]" in fit.completeness.missing
    inner = list(reports["inner_pass1"])
    inner[2] = inner[1]                                          # one report for two selections
    dup = {**reports, "inner_pass1": inner}
    fit = _classify(_records(dup, contract, protocol), protocol, dup)
    assert fit.status == "fit_time_invalid"
    assert any("inner_pass1[2] (the same report as inner_pass1[1])" in d
               for d in fit.completeness.duplicated)
    with pytest.raises(ep.ProtocolError, match="list of reports"):
        _classify(records, protocol, {**reports, "inner_pass1": reports["inner_pass1"][0]})


def test_null_report_slots_count_toward_the_exact_collection(contract, protocol,
                                                           working_protocol):
    """``None`` entries are slots: a trailing one is unexpected, a required one missing."""
    reports = _reports()
    records = _records(reports, contract, protocol)
    for prot in (protocol, working_protocol):
        trailing = {**reports, "inner_pass1": reports["inner_pass1"] + [None]}
        fit = _classify(records, prot, trailing)
        assert fit.status == "fit_time_invalid"
        assert fit.completeness.unexpected == ("report inner_pass1[5] (null)",)
        assert not fit.completeness.missing
        many = {**reports, "inner_pass1": reports["inner_pass1"] + [None, None, _report()]}
        assert set(_classify(records, prot, many).completeness.unexpected) == {
            "report inner_pass1[5] (null)", "report inner_pass1[6] (null)",
            "report inner_pass1[7]"}
        inner = list(reports["inner_pass1"])
        inner[2] = None                                         # a required slot left empty
        fit = _classify(records, prot, {**reports, "inner_pass1": inner})
        assert fit.status == "fit_time_invalid"
        assert fit.completeness.missing == ("inner_pass1[2]",)
        assert any("inner_pass1[2]: no selection report" in u
                   for u in fit.completeness.unsupported)
        fit = _classify(records, prot, {**reports, "pass2": None})
        assert fit.status == "fit_time_invalid" and fit.completeness.missing == ("pass2",)
        fit = _classify(records, prot, {**reports, "inner_pass1": [None] * 5})
        assert fit.completeness.missing == tuple(f"inner_pass1[{j}]" for j in range(5))
        fit = _classify(records, prot, {**reports, "inner_pass1": None})
        assert fit.completeness.missing == tuple(f"inner_pass1[{j}]" for j in range(5))
    ok = _classify(records, protocol, reports)
    padded = dataclasses.replace(ok, selection_reports={
        **reports, "inner_pass1": reports["inner_pass1"] + [None]})
    with pytest.raises(ep.ProtocolError, match="does not reproduce"):
        ep.FoldLedger.build({"glass": {0: padded}}, protocol)    # padded after classification
    with pytest.raises(ep.ProtocolError, match="selection_reports must be"):
        _classify(records, protocol, {"pass1": reports["pass1"]})


@pytest.mark.parametrize("key, report, why", [
    ("pass2", _report("greedy_fallback", ids=("p2",)), "is not what its selection report shows"),
    ("pass1", _report(proven=False, ids=("p1",)), "is not what its selection report shows"),
    ("inner_pass1[3]", _report(ids=("other",)), "selected rule ids differ"),
    ("inner_pass1[0]", {"source": "manual", "selected_rule_ids": ["i0"]}, "not a recognised"),
    ("pass1", ["p1"], "not a mapping"),
])
def test_a_record_that_is_not_its_reports_classification_is_unsupported(contract, protocol,
                                                                        key, report, why):
    reports = _reports()
    records = _records(reports, contract, protocol)
    changed = copy.deepcopy(reports)
    if key.startswith("inner_pass1"):
        changed["inner_pass1"][int(key[-2])] = report
    else:
        changed[key] = report
    fit = _classify(records, protocol, changed)
    assert fit.status == "fit_time_invalid"
    assert any(why in u for u in fit.completeness.unsupported), fit.completeness.unsupported


def test_rf_fit_evidence_is_unchanged(contract, protocol):
    from shared.stage2.obligations import OUTCOME_FEASIBLE_OPTIMAL, classify_fit_outcome
    records = [ep.SelectionRecord(k, classify_fit_outcome(
        OUTCOME_FEASIBLE_OPTIMAL, scope="pass2" if k == "pass2" else "pass1", reason="unit",
        report=_obligation_report("pass2" if k == "pass2" else "pass1", contract)))
        for k in ep.required_selection_keys("blackbox", protocol)]
    fit = ep.classify_fold_fit(records, arm="blackbox", protocol=protocol)
    assert fit.status == "valid_optimal"
    with pytest.raises(ep.ProtocolError, match="GLASS pipeline"):
        ep.classify_fold_fit(records, arm="blackbox", protocol=protocol,
                             selection_reports=_reports())


def test_glass_fit_outcomes_from_real_report_shapes_reproduce(contract, protocol):
    """glass_fit_outcome's mapping and the protocol's report check agree on every source."""
    for report, status in ((_report(), "valid_optimal"),
                           (_report(proven=False), "valid_unproven_undocumented"),
                           (_report("greedy_fallback"), "fallback_unreviewed"),
                           ({"source": "infeasible", "stage": "quality gates"},
                            "proven_infeasible"),
                           ({"source": "empty (min_rules = 0)"},
                            "insufficient_population_or_support")):
        reports = _reports()
        reports["pass1"] = report
        fit = _classify(_records(reports, contract, protocol), protocol, reports)
        assert fit.completeness.complete, (report, fit.completeness.problems())
        assert fit.by_key["pass1"] == status
