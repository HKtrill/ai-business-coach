"""
blackbox_pipeline.models.rf_router.contract
===========================================
RF Router adapter for the shared Stage 2 operating contract (PR 37 Commit 3).

Support only, and not wired into the notebook. ``rf_constraints_from_contract``
builds the router's ``OperatingConstraints`` from a ``SharedOperatingContract``
and nothing else: no GLASS artifact, no realised metrics, no cross-arm lookup.
The historical ``match_glass_oof`` construction (blackbox notebook Cell 19) is
untouched and remains the source of the historical artifacts.

Mapping (values only from the contract):

    min_precision_not_subscribe      <- min_pass1_npv
    max_subscriber_leakage_rate      <- max_pass1_leakage_rate
    max_subscriber_leakage_absolute  <- max_pass1_leakage_count (None -> NO_COUNT_CAP)
    min_precision_subscribe          <- min_pass2_precision
    min_pass1_coverage               =  None  (the router maximises coverage)
    min_recall_subscribe             =  None
    recall_denominator               =  "global"

The band-support floors are the router's own estimation safeguards, not shared
requirements, so the caller passes them explicitly.

Schema 2 (PR 37 Commit 4, Step 4A)
----------------------------------
A ``SharedOperatingContractV2`` also fixes the routed-set support floors, so
they become shared requirements too:

    min_band_support_pass1  <- min_pass1_support
    min_band_support_pass2  <- min_pass2_support

``rf_constraints_from_contract_v2`` takes every value, the floors included,
from the contract. ``rf_constraints_from_contract`` keeps its Commit 3
signature; given a schema-2 contract it refuses floors that differ from the
contract's. The solver's semantics already match the shared obligations: a band
is feasible when its size >= the support floor, its NPV (precision) >= the
floor and its leakage <= the ceiling, and the solver keeps the largest feasible
band in each pass, which is the lexicographic-coverage objective.
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

from shared.stage2.contract import (
    ContractViolation,
    SharedOperatingContract,
    SharedOperatingContractV2,
    check_contract_provenance,
)

from .constraints import OperatingConstraints

__all__ = ["NO_COUNT_CAP", "RF_F17_PRIMARY", "rf_constraints_from_contract",
           "rf_constraints_from_contract_v2", "check_rf_constraints_follow_contract",
           "rf_config_from_contract", "check_rf_config_follows_contract"]

#: The approved F17 primary settings for the RF Router (PR 37 F17 record, gate G1).
RF_F17_PRIMARY = (
    ("mode", "two_pass"), ("eligibility", "nested_t1"), ("eligibility_folds", 5),
    ("n_oof_folds", 5), ("random_state", 42), ("remainder_min_size", 500),
)

#: ``max_subscriber_leakage_absolute`` when the contract sets no count cap.
#: The solver's check (``positives > cap``) can never bind at this value.
NO_COUNT_CAP = 2 ** 62

_FIELD_MAP = (
    ("min_precision_not_subscribe", "min_pass1_npv"),
    ("max_subscriber_leakage_rate", "max_pass1_leakage_rate"),
    ("min_precision_subscribe", "min_pass2_precision"),
)


def rf_constraints_from_contract(
    contract: SharedOperatingContract,
    *,
    min_band_support_pass1: int,
    min_band_support_pass2: int,
) -> OperatingConstraints:
    """The RF Router's constraints under the shared contract, from it alone."""
    if not isinstance(contract, SharedOperatingContract):
        raise ContractViolation(f"expected a SharedOperatingContract, got {type(contract).__name__}")
    if isinstance(contract, SharedOperatingContractV2):
        given = (min_band_support_pass1, min_band_support_pass2)
        fixed = (contract.min_pass1_support, contract.min_pass2_support)
        if given != fixed:
            raise ContractViolation(
                f"a schema-2 contract fixes the routed-set support floors {fixed}; the band-support "
                f"floors given {given} differ (use rf_constraints_from_contract_v2)")
    count = contract.max_pass1_leakage_count
    return OperatingConstraints(
        min_precision_not_subscribe=float(contract.min_pass1_npv),
        max_subscriber_leakage_rate=float(contract.max_pass1_leakage_rate),
        max_subscriber_leakage_absolute=NO_COUNT_CAP if count is None else int(count),
        min_pass1_coverage=None,
        min_precision_subscribe=float(contract.min_pass2_precision),
        min_recall_subscribe=None,
        recall_denominator="global",
        min_band_support_pass1=int(min_band_support_pass1),
        min_band_support_pass2=int(min_band_support_pass2),
    )


def rf_constraints_from_contract_v2(contract: SharedOperatingContractV2) -> OperatingConstraints:
    """The RF Router's constraints under a schema-2 contract: every value from it, floors included."""
    if not isinstance(contract, SharedOperatingContractV2):
        raise ContractViolation("rf_constraints_from_contract_v2 needs a schema-2 contract "
                                f"(SharedOperatingContractV2), got {type(contract).__name__}")
    return rf_constraints_from_contract(contract,
                                        min_band_support_pass1=contract.min_pass1_support,
                                        min_band_support_pass2=contract.min_pass2_support)


def check_rf_constraints_follow_contract(
    constraints: Union[OperatingConstraints, Mapping],
    contract: SharedOperatingContract,
    *,
    extra: Optional[Mapping] = None,
) -> None:
    """
    Refuse RF constraints (an object or a saved ``constraints`` dict) that do
    not follow ``contract``. With ``extra`` (the artifact's ``extra`` block),
    the recorded source and contract fingerprint are checked too. Under a
    schema-2 contract the band-support floors must equal its support floors.
    """
    c = constraints.to_dict() if isinstance(constraints, OperatingConstraints) else dict(constraints)
    problems = []
    for rf_name, contract_name in _FIELD_MAP:
        if float(c.get(rf_name, float("nan"))) != float(getattr(contract, contract_name)):
            problems.append(f"{rf_name}={c.get(rf_name)!r}, contract "
                            f"{contract_name}={getattr(contract, contract_name)!r}")
    want_count = (NO_COUNT_CAP if contract.max_pass1_leakage_count is None
                  else int(contract.max_pass1_leakage_count))
    if c.get("max_subscriber_leakage_absolute") != want_count:
        problems.append(f"max_subscriber_leakage_absolute={c.get('max_subscriber_leakage_absolute')!r}, "
                        f"contract count cap -> {want_count!r}")
    for name, want in (("min_pass1_coverage", None), ("min_recall_subscribe", None),
                       ("recall_denominator", "global")):
        if c.get(name) != want:
            problems.append(f"{name}={c.get(name)!r}, contract requires {want!r}")
    if isinstance(contract, SharedOperatingContractV2):
        for rf_name, want in (("min_band_support_pass1", contract.min_pass1_support),
                              ("min_band_support_pass2", contract.min_pass2_support)):
            if c.get(rf_name) != want:
                problems.append(f"{rf_name}={c.get(rf_name)!r}, schema-2 contract requires {want!r}")
    if problems:
        raise ContractViolation("RF constraints do not follow the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    if extra is not None:
        check_contract_provenance(extra, contract)


# ----------------------------------------------------------------------
# Router configuration (PR 37 Commit 4, Step 4B)
# ----------------------------------------------------------------------
def _f17_problems(config) -> list:
    for attr in ("eligibility", "to_dict", "constraints"):
        if not hasattr(config, attr):
            raise ContractViolation(f"expected an RFRouterConfig, got {type(config).__name__}")
    return [f"{name}={getattr(config, name)!r}, the F17 primary requires {want!r}"
            for name, want in RF_F17_PRIMARY if getattr(config, name) != want]


def rf_config_from_contract(base, contract: SharedOperatingContractV2):
    """
    ``base`` (an ``RFRouterConfig`` holding the F17 primary settings) with its
    constraints replaced by the schema-2 contract's. Everything else in
    ``base`` (forest hyperparameters, verbosity, ...) is kept: those are the
    arm's own settings, owned independently of GLASS.
    """
    problems = _f17_problems(base)
    if problems:
        raise ContractViolation("cannot build an RF Router config under the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    constraints = rf_constraints_from_contract_v2(contract)
    return type(base).from_dict({**base.to_dict(), "constraints": constraints.to_dict()})


def check_rf_config_follows_contract(config, contract: SharedOperatingContractV2, *,
                                     extra: Optional[Mapping] = None) -> None:
    """Refuse an RF Router config whose F17 settings or constraints do not follow ``contract``."""
    problems = _f17_problems(config)
    if problems:
        raise ContractViolation("RF Router config does not follow the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    if not isinstance(contract, SharedOperatingContractV2):
        raise ContractViolation("check_rf_config_follows_contract needs a schema-2 contract")
    check_rf_constraints_follow_contract(config.constraints, contract, extra=extra)
