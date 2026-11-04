"""
Independent Stage 2 operating-contract support (PR 37 Commit 3; values in Commit 4).

Contract
1. ``SharedOperatingContract`` has no defaults for any requirement, validates
   ranges, and refuses provenance that names ``match_glass_oof``. Commit 3
   ships no contract instance and no numbers.
2. Under the independent pathway only ``"shared_contract"`` is accepted, and
   ``"match_glass_oof"`` is refused by name. The historical pathway still
   reads every recorded source.
3. The RF adapter builds the router's constraints from the contract alone (no
   artifact or metrics argument). The check refuses constraints that came from
   anywhere else, including GLASS-derived values relabelled as the contract.
4. Fingerprints are deterministic and change with every value.
5. The shared module imports neither arm.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

import shared.stage2.contract as contract_mod
from blackbox_pipeline.models.rf_router import OperatingConstraints
from blackbox_pipeline.models.rf_router.contract import (
    NO_COUNT_CAP, check_rf_constraints_follow_contract, rf_constraints_from_contract,
)
from shared.stage2.contract import (
    CONSTRAINT_SOURCE_MATCH_GLASS_OOF, CONSTRAINT_SOURCE_SHARED_CONTRACT, PATHWAY_HISTORICAL,
    PATHWAY_INDEPENDENT, ContractViolation, SharedOperatingContract, check_contract_provenance,
    constraint_provenance, validate_constraint_source,
)

# Synthetic test values only: NOT a proposal for Commit 4.
VALUES = dict(contract_id="synthetic-test", min_pass1_npv=0.9, max_pass1_leakage_rate=0.2,
              max_pass1_leakage_count=None, min_pass2_precision=0.4,
              provenance="unit-test fixture")


def _contract(**overrides):
    return SharedOperatingContract(**{**VALUES, **overrides})


# ----------------------------------------------------------------------
# 1. the contract container
# ----------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["min_pass1_npv", "max_pass1_leakage_rate",
                                     "max_pass1_leakage_count", "min_pass2_precision",
                                     "contract_id", "provenance"])
def test_every_requirement_must_be_given_explicitly(missing):
    kw = dict(VALUES)
    del kw[missing]
    with pytest.raises(TypeError):
        SharedOperatingContract(**kw)


@pytest.mark.parametrize("field, value", [
    ("min_pass1_npv", 1.2), ("max_pass1_leakage_rate", -0.1), ("min_pass2_precision", float("nan")),
    ("min_pass2_precision", True), ("max_pass1_leakage_count", -1), ("max_pass1_leakage_count", 2.5),
    ("contract_id", " "), ("recall_denominator", "remainder"),
    ("provenance", "copied from match_glass_oof metrics"),
])
def test_invalid_or_performance_derived_values_are_refused(field, value):
    with pytest.raises(ContractViolation):
        _contract(**{field: value})


def test_commit_3_ships_no_contract_values():
    instances = [v for v in vars(contract_mod).values() if isinstance(v, SharedOperatingContract)]
    assert instances == []
    defaults = {f.name for f in dataclasses.fields(SharedOperatingContract)
                if f.default is not dataclasses.MISSING}
    assert defaults == {"recall_denominator"}                      # the only fixed choice


# ----------------------------------------------------------------------
# 2. pathways
# ----------------------------------------------------------------------
def test_the_independent_pathway_refuses_cross_arm_performance_derived_constraints():
    with pytest.raises(ContractViolation, match="realised OOF performance"):
        validate_constraint_source(CONSTRAINT_SOURCE_MATCH_GLASS_OOF, pathway=PATHWAY_INDEPENDENT)
    with pytest.raises(ContractViolation, match="not the shared contract"):
        validate_constraint_source("hand_set", pathway=PATHWAY_INDEPENDENT)
    assert (validate_constraint_source(CONSTRAINT_SOURCE_SHARED_CONTRACT, pathway=PATHWAY_INDEPENDENT)
            == CONSTRAINT_SOURCE_SHARED_CONTRACT)


def test_the_historical_pathway_still_reads_match_glass_oof():
    assert (validate_constraint_source(CONSTRAINT_SOURCE_MATCH_GLASS_OOF, pathway=PATHWAY_HISTORICAL)
            == "match_glass_oof")
    with pytest.raises(ValueError, match="pathway must be one of"):
        validate_constraint_source("shared_contract", pathway="primary")
    with pytest.raises(ContractViolation):
        validate_constraint_source(None, pathway=PATHWAY_HISTORICAL)


def test_recorded_provenance_must_name_this_contract():
    c = _contract()
    extra = constraint_provenance(c)
    check_contract_provenance(extra, c)
    with pytest.raises(ContractViolation, match="contract_fingerprint"):
        check_contract_provenance(extra, _contract(min_pass2_precision=0.41))
    with pytest.raises(ContractViolation, match="realised OOF performance"):
        check_contract_provenance({**extra, "constraint_source": "match_glass_oof"}, c)


# ----------------------------------------------------------------------
# 3. the RF adapter
# ----------------------------------------------------------------------
def test_the_rf_constraints_come_from_the_contract_alone():
    c = _contract(max_pass1_leakage_count=300)
    rf = rf_constraints_from_contract(c, min_band_support_pass1=125, min_band_support_pass2=75)
    assert rf == OperatingConstraints(
        min_precision_not_subscribe=0.9, max_subscriber_leakage_rate=0.2,
        max_subscriber_leakage_absolute=300, min_pass1_coverage=None, min_precision_subscribe=0.4,
        min_recall_subscribe=None, recall_denominator="global",
        min_band_support_pass1=125, min_band_support_pass2=75)
    check_rf_constraints_follow_contract(rf, c)
    check_rf_constraints_follow_contract(rf.to_dict(), c, extra=constraint_provenance(c))
    params = inspect.signature(rf_constraints_from_contract).parameters
    assert list(params) == ["contract", "min_band_support_pass1", "min_band_support_pass2"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())


def test_no_count_cap_is_explicit_and_never_binds():
    rf = rf_constraints_from_contract(_contract(), min_band_support_pass1=1, min_band_support_pass2=1)
    assert rf.max_subscriber_leakage_absolute == NO_COUNT_CAP > 10 ** 15


def test_glass_derived_constraints_relabelled_as_the_contract_are_refused():
    """The historical Cell 19 recipe: values read off GLASS's realised OOF metrics."""
    c = _contract()
    glass_oof = {"pass1_npv": 0.95535, "pass1_leakage_rate": 0.13820,
                 "pass1_subscribers_lost": 513, "pass2_precision": 0.27598}
    historical = OperatingConstraints(
        min_precision_not_subscribe=glass_oof["pass1_npv"],
        max_subscriber_leakage_rate=glass_oof["pass1_leakage_rate"],
        max_subscriber_leakage_absolute=glass_oof["pass1_subscribers_lost"],
        min_precision_subscribe=glass_oof["pass2_precision"], recall_denominator="global",
        min_band_support_pass1=125, min_band_support_pass2=75)
    with pytest.raises(ContractViolation, match="do not follow the shared contract"):
        check_rf_constraints_follow_contract(historical, c, extra=constraint_provenance(c))


@pytest.mark.parametrize("field, value", [("min_pass1_coverage", 0.3), ("min_recall_subscribe", 0.1),
                                          ("recall_denominator", "remainder")])
def test_extra_rf_floors_beyond_the_contract_are_refused(field, value):
    c = _contract()
    d = rf_constraints_from_contract(c, min_band_support_pass1=1, min_band_support_pass2=1).to_dict()
    d[field] = value
    with pytest.raises(ContractViolation):
        check_rf_constraints_follow_contract(d, c)


# ----------------------------------------------------------------------
# 4-5. identity and independence
# ----------------------------------------------------------------------
def test_the_fingerprint_is_deterministic_and_covers_every_value():
    base = _contract().fingerprint()
    assert base == _contract().fingerprint() and len(base) == 16
    for field, value in [("min_pass1_npv", 0.91), ("max_pass1_leakage_rate", 0.21),
                         ("max_pass1_leakage_count", 10), ("min_pass2_precision", 0.41),
                         ("contract_id", "other"), ("provenance", "other")]:
        assert _contract(**{field: value}).fingerprint() != base, field


def test_the_shared_module_imports_neither_arm():
    tree = ast.parse(Path(contract_mod.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.startswith(("glass_pipeline", "blackbox_pipeline"))]
