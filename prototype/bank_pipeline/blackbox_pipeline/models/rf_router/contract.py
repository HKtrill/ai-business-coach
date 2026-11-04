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
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

from shared.stage2.contract import (
    ContractViolation,
    SharedOperatingContract,
    check_contract_provenance,
)

from .constraints import OperatingConstraints

__all__ = ["NO_COUNT_CAP", "rf_constraints_from_contract", "check_rf_constraints_follow_contract"]

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


def check_rf_constraints_follow_contract(
    constraints: Union[OperatingConstraints, Mapping],
    contract: SharedOperatingContract,
    *,
    extra: Optional[Mapping] = None,
) -> None:
    """
    Refuse RF constraints (an object or a saved ``constraints`` dict) that do
    not follow ``contract``. With ``extra`` (the artifact's ``extra`` block),
    the recorded source and contract fingerprint are checked too.
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
    if problems:
        raise ContractViolation("RF constraints do not follow the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    if extra is not None:
        check_contract_provenance(extra, contract)
