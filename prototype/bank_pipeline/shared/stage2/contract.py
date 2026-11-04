"""
shared.stage2.contract
======================
Arm-neutral support for an externally frozen Stage 2 operating contract.

PR 37 Commit 3 (F17 / cross-arm independence). Support only: this module
defines NO contract values. Commit 4 freezes the numbers; PR 38 activates them.

Why
---
Historically the black-box notebook derived the RF Router's constraints from
GLASS's realised pooled OOF metrics (``CONSTRAINT_SOURCE = "match_glass_oof"``,
blackbox notebook Cell 19). One arm's operating point was therefore a function
of the other arm's fitted performance and, through those pooled metrics, of
every outer fold's labels. The future primary comparison gives BOTH arms the
same externally frozen requirements and lets each optimise its own feasible
coverage under them.

What is here
------------
``SharedOperatingContract``
    The frozen requirements both arms receive. Every value is required, so
    there are no defaults to inherit silently. ``fingerprint()`` identifies it.
``validate_constraint_source(source, pathway=...)``
    ``"independent"`` (future primary) accepts only ``"shared_contract"`` and
    refuses ``"match_glass_oof"`` by name. ``"historical"`` accepts any recorded
    source, so historical artifacts stay readable and reproducible.
``constraint_provenance(contract)``
    The provenance block an artifact records under the independent pathway.
``check_contract_provenance(extra, contract)``
    Refuses an artifact whose recorded source or contract fingerprint does not
    match the contract it claims to follow.

Arm adapters live with each arm. The RF Router's is
``blackbox_pipeline.models.rf_router.contract``. GLASS has none yet: how
set-level requirements map onto its per-rule gates, its Pass 1 union cap and
its in-sample selection is a Commit 4 decision, and is deliberately not
guessed here.

This module imports neither arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "CONSTRAINT_SOURCE_MATCH_GLASS_OOF",
    "CONSTRAINT_SOURCE_SHARED_CONTRACT",
    "PATHWAY_HISTORICAL",
    "PATHWAY_INDEPENDENT",
    "ContractViolation",
    "SharedOperatingContract",
    "validate_constraint_source",
    "constraint_provenance",
    "check_contract_provenance",
]

#: Historical only: RF constraints derived from GLASS's realised pooled OOF metrics.
CONSTRAINT_SOURCE_MATCH_GLASS_OOF = "match_glass_oof"
#: Future primary: both arms receive one externally frozen contract.
CONSTRAINT_SOURCE_SHARED_CONTRACT = "shared_contract"

PATHWAY_HISTORICAL = "historical"
PATHWAY_INDEPENDENT = "independent"
_PATHWAYS = (PATHWAY_HISTORICAL, PATHWAY_INDEPENDENT)


class ContractViolation(ValueError):
    """A configuration or artifact breaks the independent operating contract."""


def _rate(name: str, value: Any, errors: list) -> None:
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
        errors.append(f"{name} must be a finite number in [0, 1], got {value!r}")


@dataclass(frozen=True)
class SharedOperatingContract:
    """
    Externally frozen Stage 2 requirements, identical for both arms.

    contract_id
        Human-readable identifier (for example the Commit 4 record's name).
    min_pass1_npv
        Floor on P(y = 0 | routed by Pass 1), measured on the routed set.
    max_pass1_leakage_rate
        Cap on Pass 1 positives / all positives (global denominator).
    max_pass1_leakage_count
        Optional count cap. It must be passed explicitly: ``None`` means no
        count cap (the rate alone binds). A count depends on population size,
        so whether to keep one is a Commit 4 decision.
    min_pass2_precision
        Floor on P(y = 1 | flagged by Pass 2), measured on the flagged set.
    provenance
        How the values were set. Must not name an arm's realised performance.
    recall_denominator
        Always ``"global"``: leakage and recall use all positives.
    """

    contract_id: str
    min_pass1_npv: float
    max_pass1_leakage_rate: float
    max_pass1_leakage_count: Optional[int]
    min_pass2_precision: float
    provenance: str
    recall_denominator: str = "global"

    def __post_init__(self) -> None:
        errors: list = []
        for name in ("contract_id", "provenance"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{name} must be a non-empty string, got {v!r}")
        for name in ("min_pass1_npv", "max_pass1_leakage_rate", "min_pass2_precision"):
            _rate(name, getattr(self, name), errors)
        c = self.max_pass1_leakage_count
        if c is not None and (isinstance(c, bool) or not isinstance(c, numbers.Integral) or c < 0):
            errors.append(f"max_pass1_leakage_count must be None or an integer >= 0, got {c!r}")
        if self.recall_denominator != "global":
            errors.append(f"recall_denominator must be 'global', got {self.recall_denominator!r}")
        if isinstance(self.provenance, str) and CONSTRAINT_SOURCE_MATCH_GLASS_OOF in self.provenance.lower():
            errors.append("provenance names 'match_glass_oof': a shared contract may not be derived "
                          "from an arm's realised OOF performance")
        if errors:
            raise ContractViolation("SharedOperatingContract validation errors:\n"
                                    + "\n".join(f"  - {e}" for e in errors))

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """SHA-256 (16 hex) of the canonical JSON of every field."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_constraint_source(source: Any, *, pathway: str) -> str:
    """
    Check a recorded constraint source against the pathway it is used under.

    ``"independent"``: only ``"shared_contract"``; ``"match_glass_oof"`` is
    refused by name. ``"historical"``: any non-empty source (provenance only).
    """
    if pathway not in _PATHWAYS:
        raise ValueError(f"pathway must be one of {list(_PATHWAYS)}, got {pathway!r}")
    if not isinstance(source, str) or not source.strip():
        raise ContractViolation(f"constraint source must be a non-empty string, got {source!r}")
    if pathway == PATHWAY_HISTORICAL:
        return source
    if source == CONSTRAINT_SOURCE_MATCH_GLASS_OOF:
        raise ContractViolation(
            "constraint_source='match_glass_oof' derives the RF Router's constraints from "
            "GLASS's realised OOF performance; it is kept for historical provenance only and "
            "is refused under the independent pathway"
        )
    if source != CONSTRAINT_SOURCE_SHARED_CONTRACT:
        raise ContractViolation(
            f"constraint_source={source!r} is not the shared contract; the independent "
            f"pathway accepts only {CONSTRAINT_SOURCE_SHARED_CONTRACT!r}"
        )
    return source


def constraint_provenance(contract: SharedOperatingContract) -> dict:
    """The provenance block an artifact records under the independent pathway."""
    if not isinstance(contract, SharedOperatingContract):
        raise ContractViolation(f"expected a SharedOperatingContract, got {type(contract).__name__}")
    return {
        "constraint_source": CONSTRAINT_SOURCE_SHARED_CONTRACT,
        "contract_id": contract.contract_id,
        "contract_fingerprint": contract.fingerprint(),
    }


def check_contract_provenance(extra: Mapping[str, Any], contract: SharedOperatingContract) -> None:
    """Refuse recorded provenance that does not follow ``contract``."""
    validate_constraint_source(extra.get("constraint_source"), pathway=PATHWAY_INDEPENDENT)
    want = contract.fingerprint()
    got = extra.get("contract_fingerprint")
    if got != want:
        raise ContractViolation(f"contract_fingerprint {got!r} does not match the contract ({want})")
