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

Schema 2 (PR 37 Commit 4, Step 4A)
----------------------------------
``SharedOperatingContractV2``
    Extends ``SharedOperatingContract`` (a subclass, so every Commit 3 check
    and adapter accepts it) with what the shared contract must also fix:
    schema and contract versions, a working/frozen status, the approved
    lexicographic-coverage objective, the metric-definition identifier, the
    routed-set support floors and a structured provenance record. Every field
    is required. The Commit 3 class is unchanged: schema 1 stays readable.
``load_operating_contract(path)``
    Reads one versioned JSON record and nothing else: no artifact, no
    realised metric, no other arm. ``working_profile_path()`` is the Step 4A
    working profile, which is NOT frozen. Step 4D froze the final record:
    ``frozen_profile_path()`` (``pr37-stage2-primary`` 1.0.0, same values),
    loaded by ``load_frozen_contract`` against its pinned fingerprint.

Values live only in the JSON records under ``shared/stage2/contracts/``; this
module still defines none. The arm-neutral obligation checks are in
``shared.stage2.obligations``.

This module imports neither arm.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import numbers
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
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
    # schema 2 (PR 37 Commit 4, Step 4A)
    "SCHEMA_VERSION_2",
    "OBJECTIVE_LEXICOGRAPHIC_COVERAGE",
    "METRIC_DEFINITIONS_ID",
    "STATUS_WORKING",
    "STATUS_FROZEN",
    "CONTRACT_STATUSES",
    "CONTRACT_SETTINGS",
    "RATIONALE_BASES",
    "EVIDENCE_KINDS",
    "EVIDENCE_SPLITS",
    "EVIDENCE_ROLES",
    "TEST_DERIVED_FIELD_TOKENS",
    "CONTRACTS_DIR",
    "SharedOperatingContractV2",
    "working_profile_path",
    "load_operating_contract",
    # Step 4D freeze
    "FROZEN_PROFILE_FINGERPRINT",
    "frozen_profile_path",
    "load_frozen_contract",
    "APPROVAL_UNCONFIRMED",
    "approval_confirmed",
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
    block = {
        "constraint_source": CONSTRAINT_SOURCE_SHARED_CONTRACT,
        "contract_id": contract.contract_id,
        "contract_fingerprint": contract.fingerprint(),
    }
    if isinstance(contract, SharedOperatingContractV2):
        # Schema 2 also names its version and status; schema 1 is unchanged.
        block.update(schema_version=contract.schema_version,
                     contract_version=contract.contract_version,
                     contract_status=contract.status)
    return block


def check_contract_provenance(extra: Mapping[str, Any], contract: SharedOperatingContract) -> None:
    """
    Refuse recorded provenance that does not follow ``contract``.

    Schema 1: the source and the fingerprint (Commit 3 behaviour, unchanged).
    Schema 2: also ``contract_id``, ``schema_version``, ``contract_version`` and
    ``contract_status``, each present and equal to the contract's.
    """
    validate_constraint_source(extra.get("constraint_source"), pathway=PATHWAY_INDEPENDENT)
    want = contract.fingerprint()
    got = extra.get("contract_fingerprint")
    if got != want:
        raise ContractViolation(f"contract_fingerprint {got!r} does not match the contract ({want})")
    if isinstance(contract, SharedOperatingContractV2):
        expected = {
            "contract_id": contract.contract_id,
            "schema_version": contract.schema_version,
            "contract_version": contract.contract_version,
            "contract_status": contract.status,
        }
        problems = [f"{key} missing" if key not in extra
                    else f"{key}={extra[key]!r}, the contract has {value!r}"
                    for key, value in expected.items()
                    if key not in extra or extra[key] != value]
        if problems:
            raise ContractViolation("recorded schema-2 provenance does not match the contract: "
                                    + "; ".join(problems))


# ======================================================================
# Schema 2 (PR 37 Commit 4, Step 4A)
# ======================================================================
SCHEMA_VERSION_2 = "stage2-operating-contract/2"

#: The approved shared objective (Commit 4, decision M1). Each arm maximises
#: its own feasible Pass 1 routed rows; then, on the remainder its own Pass 1
#: leaves, its own feasible Pass 2 routed rows. Lexicographic: neither arm
#: maximises total coverage jointly, and Pass 1 is never traded for Pass 2.
OBJECTIVE_LEXICOGRAPHIC_COVERAGE = (
    "lexicographic_coverage:max_pass1_routed_rows,then_max_pass2_routed_rows")

#: Names the shared metric set: ``shared.stage2.obligations.CONTRACT_METRICS``,
#: whose numerators and denominators are the diagnostics registry's.
METRIC_DEFINITIONS_ID = "pr37.stage2.contract_metrics/1"

STATUS_WORKING = "working"
STATUS_FROZEN = "frozen"
CONTRACT_STATUSES = (STATUS_WORKING, STATUS_FROZEN)

#: Every numerical setting a schema-2 contract must justify.
CONTRACT_SETTINGS = (
    "min_pass1_npv", "max_pass1_leakage_rate", "max_pass1_leakage_count",
    "min_pass2_precision", "min_pass1_support", "min_pass2_support",
)

#: Allowed bases for a setting's rationale. An arm's realised performance is
#: deliberately not one of them: no setting may be justified by it.
RATIONALE_BASES = (
    "substantive_requirement", "protective_guard", "structural_argument",
    "historical_configuration_anchor",
)

EVIDENCE_KINDS = (
    "baseline_package", "artifact_sidecar", "threshold_sweep", "methodology_record",
    "code_reference",
)
#: Where a piece of evidence comes from. There is no test-split value: evidence
#: derived from the held-out split is refused.
EVIDENCE_SPLITS = ("train_oof", "train_in_sample", "synthetic", "not_applicable")
EVIDENCE_ROLES = ("context", "feasibility_screen", "provenance")
#: Field-name segments that mark a test-derived input (``test_metrics``,
#: ``holdout_proba``, ...). Matched on whole ``[a-z0-9]`` segments.
TEST_DERIVED_FIELD_TOKENS = ("test", "holdout", "oracle")

#: The versioned contract records.
CONTRACTS_DIR = Path(__file__).resolve().parent / "contracts"
_WORKING_PROFILE = "stage2_primary_working.json"
#: The frozen PR 38 primary contract (PR 37 Commit 4, Step 4D) and its pinned
#: identity. The values live only in the record; this pins which record.
_FROZEN_PROFILE = "stage2_primary_v1.json"
#: ``approved_by`` prefix of a frozen record whose owner approval is not yet
#: confirmed (a review candidate). Such a record validates, but no frozen loader
#: returns it and it cannot govern PR 38.
APPROVAL_UNCONFIRMED = "UNCONFIRMED"
FROZEN_PROFILE_FINGERPRINT = "0817825b9eecbd11"

_PROVENANCE_KEYS = (
    "rationale", "evidence", "exposure", "derived_from_arm_realised_performance",
    "historical_exposure_informed_decisions", "feasibility_screening", "limitations",
    "relation_to_pr38", "approval",
)
_RATIONALE_KEYS = ("basis", "text")
_EVIDENCE_KEYS = ("id", "kind", "split", "role", "location", "sha256", "fields_used",
                  "description")
_EXPOSURE_KEYS = ("historical_oof_examined", "test_split_examined", "details")
_SCREENING_KEYS = ("scope", "result", "limitations")
_APPROVAL_KEYS = ("state", "approved_by", "approved_on", "note")
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SEGMENT_RE = re.compile(r"[a-z0-9]+")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _texts(value: Any, *, allow_empty: bool = False) -> bool:
    return (isinstance(value, (list, tuple)) and (allow_empty or len(value) > 0)
            and all(_text(v) for v in value))


def _exact_keys(where: str, value: Any, keys: tuple, errors: list) -> bool:
    """``value`` is a mapping with exactly ``keys``; report what is missing or extra."""
    if not isinstance(value, Mapping):
        errors.append(f"{where} must be an object, got {type(value).__name__}")
        return False
    missing = [k for k in keys if k not in value]
    unknown = [k for k in value if k not in keys]
    if missing:
        errors.append(f"{where} is missing {missing}")
    if unknown:
        errors.append(f"{where} has unknown keys {unknown}")
    return not missing


def _strings(obj: Any):
    """Every string in a nested record, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield k
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _freeze(obj: Any, where: str) -> Any:
    """A read-only copy of a JSON-shaped value (objects -> mappingproxy, arrays -> tuples)."""
    if isinstance(obj, Mapping):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise ContractViolation(f"{where}: object keys must be strings, got {bad!r}")
        return MappingProxyType({k: _freeze(v, f"{where}.{k}") for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v, f"{where}[{i}]") for i, v in enumerate(obj))
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float) and math.isfinite(obj):
        return obj
    raise ContractViolation(f"{where}: {obj!r} is not a JSON scalar, array or object")


def _thaw(obj: Any) -> Any:
    """Plain JSON-ready copy of a value produced by ``_freeze``."""
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


def _validate_provenance_record(rec: Any, status: str, errors: list) -> None:
    """Structured provenance: complete, sourced, and never performance-derived."""
    if not _exact_keys("provenance_record", rec, _PROVENANCE_KEYS, errors):
        return

    rationale = rec["rationale"]
    if _exact_keys("provenance_record.rationale", rationale, CONTRACT_SETTINGS, errors):
        for name in CONTRACT_SETTINGS:
            entry, where = rationale[name], f"provenance_record.rationale.{name}"
            if not _exact_keys(where, entry, _RATIONALE_KEYS, errors):
                continue
            if entry["basis"] not in RATIONALE_BASES:
                errors.append(f"{where}.basis must be one of {list(RATIONALE_BASES)}, got "
                              f"{entry['basis']!r} (an arm's realised performance is not an "
                              "allowed basis)")
            if not _text(entry["text"]):
                errors.append(f"{where}.text must be a non-empty string")

    evidence = rec["evidence"]
    if not isinstance(evidence, (list, tuple)) or not evidence:
        errors.append("provenance_record.evidence must be a non-empty list")
    else:
        seen = set()
        for i, item in enumerate(evidence):
            where = f"provenance_record.evidence[{i}]"
            if not _exact_keys(where, item, _EVIDENCE_KEYS, errors):
                continue
            for key in ("id", "location", "description"):
                if not _text(item[key]):
                    errors.append(f"{where}.{key} must be a non-empty string")
            if item["id"] in seen:
                errors.append(f"{where}.id {item['id']!r} is not unique")
            seen.add(item["id"])
            if item["kind"] not in EVIDENCE_KINDS:
                errors.append(f"{where}.kind must be one of {list(EVIDENCE_KINDS)}, got "
                              f"{item['kind']!r}")
            if item["split"] not in EVIDENCE_SPLITS:
                errors.append(f"{where}.split must be one of {list(EVIDENCE_SPLITS)}, got "
                              f"{item['split']!r}: test-derived evidence is refused")
            if item["role"] not in EVIDENCE_ROLES:
                errors.append(f"{where}.role must be one of {list(EVIDENCE_ROLES)}, got "
                              f"{item['role']!r}")
            sha = item["sha256"]
            if sha is not None and not (isinstance(sha, str) and _SHA256_RE.fullmatch(sha)):
                errors.append(f"{where}.sha256 must be null or 64 lowercase hex characters")
            used = item["fields_used"]
            if not _texts(used, allow_empty=True):
                errors.append(f"{where}.fields_used must be a list of non-empty strings")
            else:
                for name in used:
                    hit = set(_SEGMENT_RE.findall(name.lower())) & set(TEST_DERIVED_FIELD_TOKENS)
                    if hit:
                        errors.append(f"{where}.fields_used names a test-derived field {name!r} "
                                      f"({sorted(hit)}): test-derived inputs are refused")

    exposure = rec["exposure"]
    if _exact_keys("provenance_record.exposure", exposure, _EXPOSURE_KEYS, errors):
        for key in ("historical_oof_examined", "test_split_examined"):
            if not isinstance(exposure[key], bool):
                errors.append(f"provenance_record.exposure.{key} must be true or false")
        if not _texts(exposure["details"]):
            errors.append("provenance_record.exposure.details must be a non-empty list of strings")

    # Two separate questions (module notes, "Performance dependence"):
    #   derived_from_arm_realised_performance   was any value computed from, copied from or
    #                                           mechanically set to match an arm's realised
    #                                           performance? Must be false.
    #   historical_exposure_informed_decisions  did examined historical results inform the
    #                                           methodological choices? Disclosed; it must be
    #                                           true whenever historical OOF was examined.
    derived = rec["derived_from_arm_realised_performance"]
    if derived is not False:
        errors.append("provenance_record.derived_from_arm_realised_performance must be false: no "
                      "contract value may be computed from, copied from or set to match an arm's "
                      f"realised performance, got {derived!r}")
    informed = rec["historical_exposure_informed_decisions"]
    if not isinstance(informed, bool):
        errors.append("provenance_record.historical_exposure_informed_decisions must be true or "
                      f"false, got {informed!r}")
    elif (not informed and isinstance(exposure, Mapping)
          and exposure.get("historical_oof_examined") is True):
        errors.append("provenance_record.historical_exposure_informed_decisions must be true when "
                      "historical OOF results were examined: the record may not claim that the "
                      "exposure left its methodological decisions uninformed")

    screening = rec["feasibility_screening"]
    if _exact_keys("provenance_record.feasibility_screening", screening, _SCREENING_KEYS, errors):
        for key in ("scope", "result"):
            if not _text(screening[key]):
                errors.append(f"provenance_record.feasibility_screening.{key} must be a "
                              "non-empty string")
        if not _texts(screening["limitations"]):
            errors.append("provenance_record.feasibility_screening.limitations must be a "
                          "non-empty list of strings")

    if not _texts(rec["limitations"]):
        errors.append("provenance_record.limitations must be a non-empty list of strings")
    if not _text(rec["relation_to_pr38"]):
        errors.append("provenance_record.relation_to_pr38 must be a non-empty string")

    approval = rec["approval"]
    if _exact_keys("provenance_record.approval", approval, _APPROVAL_KEYS, errors):
        if approval["state"] != status:
            errors.append(f"provenance_record.approval.state {approval['state']!r} does not match "
                          f"the contract status {status!r}")
        if not _text(approval["note"]):
            errors.append("provenance_record.approval.note must be a non-empty string")
        if status == STATUS_WORKING:
            if approval["approved_by"] is not None or approval["approved_on"] is not None:
                errors.append("a working contract is not approved: approved_by and approved_on "
                              "must be null")
        elif status == STATUS_FROZEN:
            if not _text(approval["approved_by"]):
                errors.append("a frozen contract needs provenance_record.approval.approved_by")
            try:
                _dt.date.fromisoformat(approval["approved_on"])
            except (TypeError, ValueError):
                errors.append("a frozen contract needs provenance_record.approval.approved_on "
                              f"as YYYY-MM-DD, got {approval['approved_on']!r}")

    if isinstance(rationale, Mapping) and any(
            CONSTRAINT_SOURCE_MATCH_GLASS_OOF in s.lower() for s in _strings(rationale)):
        errors.append("provenance_record.rationale names 'match_glass_oof': no setting may be "
                      "justified by the historical GLASS-derived pathway. It may be described as "
                      "historical evidence elsewhere in the record (evidence, exposure, "
                      "limitations)")


@dataclass(frozen=True, kw_only=True)
class SharedOperatingContractV2(SharedOperatingContract):
    """
    Schema-2 Stage 2 operating contract (PR 37 Commit 4, Step 4A).

    Inherits the Commit 3 requirements and their validation unchanged:
    ``min_pass1_npv`` (floor), ``max_pass1_leakage_rate`` (ceiling, global
    denominator), ``max_pass1_leakage_count`` (``None`` = no count cap),
    ``min_pass2_precision`` (floor), ``provenance`` (here: the one-paragraph
    summary) and ``recall_denominator`` (``"global"``). Adds, all required:

    schema_version
        ``SCHEMA_VERSION_2``.
    contract_version
        ``MAJOR.MINOR.PATCH``. A working contract carries a pre-release label
        (``0.1.0-working``); a frozen one carries none.
    status
        ``"working"`` or ``"frozen"``. Only a frozen record may govern PR 38.
    objective
        ``OBJECTIVE_LEXICOGRAPHIC_COVERAGE``.
    metric_definitions
        ``METRIC_DEFINITIONS_ID``.
    min_pass1_support, min_pass2_support
        Floors on the rows a pass routes, at fit time (integers >= 1).
    provenance_record
        Structured provenance: per-setting rationale with an allowed basis,
        evidence with split and role, exposure disclosures, the two
        performance-dependence flags, the feasibility screening, limitations,
        the relation to PR 38 and the approval state. Stored read-only.

    Performance dependence. ``derived_from_arm_realised_performance`` must be
    false: no value was computed from, copied from or mechanically set to match
    an arm's realised performance (the historical ``match_glass_oof`` pathway
    did exactly that). ``historical_exposure_informed_decisions`` discloses the
    separate fact that examined historical results informed the methodological
    choices; it must be true whenever historical OOF results were examined. The
    historical pathway may be described as history in the evidence, exposure
    and limitations, but no setting's rationale may cite it.

    Schema 2 also refuses vacuous limits: floors must be above 0 and the
    leakage ceiling below 1.
    """

    schema_version: str
    contract_version: str
    status: str
    objective: str
    metric_definitions: str
    min_pass1_support: int
    min_pass2_support: int
    provenance_record: Mapping[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        errors: list = []
        if self.schema_version != SCHEMA_VERSION_2:
            errors.append(f"schema_version must be {SCHEMA_VERSION_2!r}, got {self.schema_version!r}")
        if not (isinstance(self.contract_version, str)
                and _VERSION_RE.fullmatch(self.contract_version)):
            errors.append(f"contract_version must be MAJOR.MINOR.PATCH[-label], got "
                          f"{self.contract_version!r}")
        if self.status not in CONTRACT_STATUSES:
            errors.append(f"status must be one of {list(CONTRACT_STATUSES)}, got {self.status!r}")
        elif isinstance(self.contract_version, str):
            labelled = "-" in self.contract_version
            if self.status == STATUS_WORKING and not labelled:
                errors.append("a working contract's version must carry a pre-release label "
                              f"(for example 0.1.0-working), got {self.contract_version!r}")
            if self.status == STATUS_FROZEN and labelled:
                errors.append("a frozen contract's version must not carry a pre-release label, "
                              f"got {self.contract_version!r}")
        if self.objective != OBJECTIVE_LEXICOGRAPHIC_COVERAGE:
            errors.append(f"objective must be {OBJECTIVE_LEXICOGRAPHIC_COVERAGE!r}, got "
                          f"{self.objective!r}")
        if self.metric_definitions != METRIC_DEFINITIONS_ID:
            errors.append(f"metric_definitions must be {METRIC_DEFINITIONS_ID!r}, got "
                          f"{self.metric_definitions!r}")
        for name in ("min_pass1_npv", "min_pass2_precision"):
            if float(getattr(self, name)) <= 0.0:
                errors.append(f"{name} must be above 0 (a zero floor imposes nothing)")
        if float(self.max_pass1_leakage_rate) >= 1.0:
            errors.append("max_pass1_leakage_rate must be below 1 (a ceiling of 1 imposes nothing)")
        for name in ("min_pass1_support", "min_pass2_support"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
                errors.append(f"{name} must be an integer >= 1 (an empty routed set is never "
                              f"feasible), got {v!r}")
        _validate_provenance_record(self.provenance_record, self.status, errors)
        if errors:
            raise ContractViolation("SharedOperatingContractV2 validation errors:\n"
                                    + "\n".join(f"  - {e}" for e in errors))
        object.__setattr__(self, "provenance_record",
                           _freeze(self.provenance_record, "provenance_record"))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Plain JSON-ready copy of every field (the record's on-disk form)."""
        return {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}

    def content_sha256(self) -> str:
        """Full SHA-256 of the canonical JSON; ``fingerprint()`` is its first 16 characters."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping) -> "SharedOperatingContractV2":
        """Build from a record. Every field must be present; unknown keys are refused."""
        if not isinstance(data, Mapping):
            raise ContractViolation(f"a contract record must be an object, got {type(data).__name__}")
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in data]
        unknown = [k for k in data if k not in names]
        if missing or unknown:
            raise ContractViolation(f"contract record fields: missing {missing}, unknown {unknown}")
        return cls(**{n: data[n] for n in names})


def working_profile_path() -> Path:
    """The Step 4A working profile. Not frozen: it may not govern PR 38."""
    return CONTRACTS_DIR / _WORKING_PROFILE


def frozen_profile_path() -> Path:
    """The Step 4D frozen record (``pr37-stage2-primary`` 1.0.0): the one PR 38 runs under."""
    return CONTRACTS_DIR / _FROZEN_PROFILE


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ContractViolation(f"contract record repeats the key {k!r}")
        out[k] = v
    return out


def _no_constant(token):
    raise ContractViolation(f"contract record contains the non-finite number {token}")


def load_operating_contract(
    path,
    *,
    expected_fingerprint: Optional[str] = None,
    require_status: Optional[str] = None,
) -> SharedOperatingContractV2:
    """
    Load one schema-2 contract record from a JSON file.

    Reads that file and nothing else: the values cannot depend on any artifact,
    realised metric or other arm at load time. ``expected_fingerprint`` pins the
    content identity; ``require_status="frozen"`` is what PR 38 passes.
    """
    if require_status is not None and require_status not in CONTRACT_STATUSES:
        raise ValueError(f"require_status must be one of {list(CONTRACT_STATUSES)} or None, got "
                         f"{require_status!r}")
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{path}: not valid JSON ({exc})") from exc
    contract = SharedOperatingContractV2.from_dict(data)
    if require_status is not None and contract.status != require_status:
        raise ContractViolation(f"{path}: contract status is {contract.status!r}, "
                                f"{require_status!r} is required")
    if expected_fingerprint is not None and contract.fingerprint() != expected_fingerprint:
        raise ContractViolation(f"{path}: contract fingerprint {contract.fingerprint()} does not "
                                f"match the expected {expected_fingerprint}")
    return contract


def approval_confirmed(approval: Any) -> bool:
    """
    Whether an approval block names a confirmed approver: a non-empty
    ``approved_by`` that does not start with ``APPROVAL_UNCONFIRMED``. A name
    found elsewhere (an author field, say) is never taken as an approval.
    """
    by = approval.get("approved_by") if isinstance(approval, Mapping) else None
    return (isinstance(by, str) and bool(by.strip())
            and not by.strip().upper().startswith(APPROVAL_UNCONFIRMED))


def load_frozen_contract(path=None) -> SharedOperatingContractV2:
    """
    The frozen PR 38 primary contract: ``frozen_profile_path()`` (or ``path``),
    refused unless its status is ``"frozen"``, its fingerprint is
    ``FROZEN_PROFILE_FINGERPRINT`` and its owner approval is confirmed. The only
    way PR 38 obtains its contract; there is no review or override switch.
    """
    contract = load_operating_contract(path or frozen_profile_path(),
                                       expected_fingerprint=FROZEN_PROFILE_FINGERPRINT,
                                       require_status=STATUS_FROZEN)
    if not approval_confirmed(contract.provenance_record["approval"]):
        raise ContractViolation("the frozen contract is a review candidate: its owner approval "
                                "is unconfirmed, so it cannot govern PR 38")
    return contract
