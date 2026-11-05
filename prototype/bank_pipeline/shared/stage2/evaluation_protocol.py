"""
shared.stage2.evaluation_protocol
=================================
The pre-registered PR 38 Stage 2 evaluation protocol (PR 37 Commit 4, Step 4C;
frozen in Step 4D).

The protocol is a versioned record (``protocols/pr38_evaluation_working.json``)
plus the executable invariants it depends on. Nothing here fits, tunes or
reads an artifact, and nothing here has seen a corrected result. The metrics
themselves are the diagnostics registry's (``shared.stage2.diagnostics``,
pinned by the pre-correction baseline and only imported) and the Step 4A
obligation checks (``shared.stage2.obligations``); this module adds only what
the protocol needs on top of them.

    load_evaluation_protocol    validated record, deterministic fingerprint;
                                a working protocol may not govern PR 38
                                (``require_frozen``)
    load_frozen_protocol        the Step 4D frozen record against its pinned
                                fingerprint; ``check_arm_profile_binding``
                                binds the frozen arm profiles
    check_contract_binding      the protocol names exactly one contract
    check_population_identity   a trace against the pinned rows, labels,
                                folds and fingerprints (V1)
    check_upstream_identity     the frozen LR / EBM operating points: artifact
                                SHA-256, per-row recomputation, EBM fold
                                thresholds, counts and row-aligned flag
                                fingerprints (``flag_fingerprint``)
    PairedRouting               two routings of the same rows, aligned by row
                                identity (cross-arm, pre vs post, ablation vs
                                primary); never by position
    ReferenceFlags              row-indexed reference flags bound to a pairing
    verify_reference            the complementarity reference, only from
                                upstream sources whose identity passed
    metric_table / fold_dispersion
                                per-fold and pooled metrics; pooled from summed
                                counts, never averaged ratios
    stratified_paired_bootstrap the protocol's uncertainty procedure;
    run_registered_bootstrap    the same, with its settings taken from the
                                protocol and a versioned, digest-checked
                                provenance (``BootstrapRun``)
    wilson_interval             descriptive single-proportion interval
    check_selection_completeness / classify_fold_fit / FoldLedger / pooled_observation
                                every required selection of a fold present
                                exactly once, fit status per fold, the
                                attempted / accepted ledger, and pooling that
                                never drops a fold silently
    classify_observation        outer-OOF shortfall classification; an observed
                                violation is never relabelled compliant
    f31_strip_review            the R2 trigger (strip A >= 1/20 of a band)
    structural_acceptance / comparative_report_gaps
                                artifact acceptance, kept apart from performance
    evaluate_adoption           complementarity-ablation adoption gates, on a
                                verified reference; official only under a
                                frozen protocol with complete upstream identity
    evidence_tier / check_inputs
                                evidence tiers; a name-based input guard
                                (defense in depth only, not provenance)

Arm-neutral: nothing here imports either router.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from shared.stage2.contract import (
    TEST_DERIVED_FIELD_TOKENS,
    SharedOperatingContractV2,
    approval_confirmed,
)
from shared.stage2.diagnostics.contribution import (
    CONTRIBUTION_LABELS,
    MATCH_SCORES,
    POP_COMMON,
    TIE_BREAK_RULE,
)
from shared.stage2.diagnostics.metrics import METRIC_DEFINITIONS, Ratio
from shared.stage2.diagnostics.operating_points import (
    PRIMARY_OPERATING_POINTS,
    UpstreamFlags,
    resolve_operating_point,
)
from shared.stage2.diagnostics.schema import (
    ABSTAIN,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    ROUTER_STATES,
    SELECTION_FOLD_NESTED,
    SELECTION_GLOBAL_TRAIN_LABELS,
    RouterOOFView,
    UpstreamOOFView,
)
from shared.stage2.obligations import (
    OUTCOME_FEASIBLE_GREEDY,
    OUTCOME_FEASIBLE_OPTIMAL,
    OUTCOME_FEASIBLE_UNPROVEN,
    OUTCOME_INFEASIBLE,
    OUTCOME_INSUFFICIENT,
    OUTCOME_NO_SOLUTION,
    SATISFIED,
    UNDEFINED_STATUS,
    VIOLATED,
    FitOutcome,
    ObligationReport,
    ObligationResult,
    RoutingCounts,
    observe_folds,
    observe_routing,
    pool_counts,
    routing_ratios,
)

__all__ = [
    "SCHEMA_VERSION", "STATUS_WORKING", "STATUS_FROZEN", "PROTOCOLS_DIR", "PROTOCOL_METRICS",
    "FIT_STATUSES", "ACCEPTED_FIT_STATUSES", "ProtocolError",
    "EvaluationProtocol", "validate_protocol_record", "load_evaluation_protocol",
    "working_protocol_path", "require_frozen", "check_contract_binding",
    "FROZEN_PROTOCOL_FINGERPRINT", "frozen_protocol_path", "load_frozen_protocol",
    "check_arm_profile_binding", "FREEZE_DECISIONS", "glass_reports_required",
    "freeze_approval_confirmed", "is_pinned_frozen", "FrozenSpecification",
    "load_frozen_specification",
    "BOOTSTRAP_PROCEDURE_ID", "FLAG_FINGERPRINT_METHOD",
    "PopulationIdentity", "IdentityReport", "check_population_identity", "flag_fingerprint",
    "check_upstream_identity",
    "PairedRouting", "ReferenceFlags", "UpstreamSources", "VerifiedReference",
    "verify_reference", "class_conditional_counts", "protocol_ratios",
    "metric_table", "fold_dispersion", "BOOTSTRAP_METRICS", "bootstrap_settings",
    "stratified_paired_bootstrap", "BootstrapRun", "run_registered_bootstrap",
    "wilson_interval", "SelectionRecord", "CompletenessReport", "required_selection_keys",
    "check_selection_completeness", "FoldFit", "classify_fold_fit", "FoldLedger",
    "pooled_observation",
    "materiality_tolerance", "ShortfallObservation", "classify_observation", "f31_strip_review",
    "GateResult", "StructuralAcceptance", "structural_acceptance", "comparative_report_gaps",
    "ADOPTION_METRICS", "AdoptionEvidence", "AdoptionDecision", "evaluate_adoption",
    "evidence_tier",
    "check_inputs",
]

SCHEMA_VERSION = "stage2-evaluation-protocol/1"
STATUS_WORKING, STATUS_FROZEN = "working", "frozen"
PROTOCOLS_DIR = Path(__file__).resolve().parent / "protocols"
_WORKING = "pr38_evaluation_working.json"
#: The frozen protocol (PR 37 Commit 4, Step 4D) and its pinned identity.
_FROZEN = "pr38_evaluation_v1.json"
FROZEN_PROTOCOL_FINGERPRINT = "245a1a68fb68cf4e"


#: Versioned identity of ``stratified_paired_bootstrap``. Any change to the
#: algorithm (strata, resampling, interval, quantile method, undefined handling)
#: needs a new identity and a new protocol version.
BOOTSTRAP_PROCEDURE_ID = "pr38.stage2.stratified_paired_bootstrap/1"
#: Versioned identity of ``flag_fingerprint``.
FLAG_FINGERPRINT_METHOD = "pr38.stage2.row_flag_fingerprint/1"


class ProtocolError(ValueError):
    """The protocol record, or an input to one of its procedures, breaks the protocol."""


#: Protocol metrics outside the diagnostics registry (which is pinned and only imported).
PROTOCOL_METRICS: Mapping[str, Tuple[str, str]] = MappingProxyType({
    "pass2_negative_routing_rate": ("negatives flagged by Pass 2", "all negatives"),
    "abstain_negative_share": ("negatives abstained on", "all negatives"),
})

# Fit status per arm and fold, most severe first (``classify_fold_fit``).
FIT_TIME_INVALID = "fit_time_invalid"
FIT_INFEASIBLE = "proven_infeasible"
FIT_NO_SOLUTION = "no_feasible_solution"
FIT_INSUFFICIENT = "insufficient_population_or_support"
FIT_FALLBACK_UNREVIEWED = "fallback_unreviewed"
FIT_UNPROVEN_UNDOCUMENTED = "valid_unproven_undocumented"
FIT_FALLBACK_ACCEPTED = "fallback_accepted_after_review"
FIT_UNPROVEN_DOCUMENTED = "valid_unproven_documented"
FIT_OPTIMAL = "valid_optimal"
FIT_STATUSES = (FIT_TIME_INVALID, FIT_INFEASIBLE, FIT_NO_SOLUTION, FIT_INSUFFICIENT,
                FIT_FALLBACK_UNREVIEWED, FIT_UNPROVEN_UNDOCUMENTED, FIT_FALLBACK_ACCEPTED,
                FIT_UNPROVEN_DOCUMENTED, FIT_OPTIMAL)
ACCEPTED_FIT_STATUSES = (FIT_OPTIMAL, FIT_UNPROVEN_DOCUMENTED, FIT_FALLBACK_ACCEPTED)
_STOP_STATUS = MappingProxyType({OUTCOME_INFEASIBLE: FIT_INFEASIBLE,
                                 OUTCOME_NO_SOLUTION: FIT_NO_SOLUTION,
                                 OUTCOME_INSUFFICIENT: FIT_INSUFFICIENT})

_SEGMENT_RE = re.compile(r"[a-z0-9]+")


# ======================================================================
# The record
# ======================================================================
def _freeze(obj):
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


def _fraction(value, where: str, errors: List[str]) -> Optional[Fraction]:
    try:
        if isinstance(value, bool):
            raise TypeError
        return Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        errors.append(f"{where}: {value!r} is not an exact number")
        return None


def _int(value, where: str, errors: List[str], *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        errors.append(f"{where}: expected an integer >= {minimum}, got {value!r}")
        return None
    return int(value)


_TOP_KEYS = ("protocol_id", "schema_version", "protocol_version", "status", "purpose",
             "contract", "population", "f17_primary", "hierarchy", "upstream", "matched_budget",
             "uncertainty", "shortfall", "fit_status", "folds", "f31", "evidence_tiers",
             "complementarity", "acceptance", "inputs", "pending_approvals")
#: Step 4D: a frozen record must also carry these; a working record may not
#: carry ``freeze``.
_FREEZE_KEYS = ("arm_profiles", "freeze")
#: The decisions a freeze resolves (protocol document, section 17).
FREEZE_DECISIONS = tuple(f"D-C{i}" for i in range(1, 12))
DECISION_RESOLUTIONS = ("approved", "approved_with_revision", "not_required")
DECISION_KINDS = ("registered_criterion", "registered_procedure", "descriptive_heuristic",
                  "reporting_rule", "reporting_convention", "optional_diagnostic", "binding")
_FREEZE_RECORD_KEYS = ("approved_by", "approved_on", "supersedes", "reverification",
                       "decisions")
_DECISION_KEYS = ("id", "resolution", "kind", "decision", "rationale")
GLASS_REPORTS_REQUIRED = "required"
GLASS_REPORTS_OPTIONAL = "optional"
_HEX = re.compile(r"[0-9a-f]+")
_KNOWN_METRICS = set(METRIC_DEFINITIONS) | set(PROTOCOL_METRICS) | {"unique_tp", "unique_fp"}


def _known_metric(name) -> bool:
    return name in _KNOWN_METRICS or name in CONTRIBUTION_LABELS


_SECTIONS = ("contract", "population", "f17_primary", "hierarchy", "upstream", "matched_budget",
             "uncertainty", "shortfall", "fit_status", "folds", "f31", "evidence_tiers",
             "complementarity", "acceptance", "inputs")


def _freeze_errors(rec: Mapping) -> List[str]:
    """The Step 4D freeze record and arm-profile bindings (required when frozen)."""
    errors: List[str] = []
    frozen = rec["status"] == STATUS_FROZEN
    if not frozen:
        if "freeze" in rec:
            errors.append("a working protocol may not carry a freeze record")
    else:
        errors += [f"a frozen protocol needs {k!r}" for k in _FREEZE_KEYS if k not in rec]
    profiles = rec.get("arm_profiles")
    if profiles is not None:
        if not isinstance(profiles, Mapping) or set(profiles) != {"glass", "rf"}:
            errors.append("arm_profiles must bind exactly the 'glass' and 'rf' profiles")
        else:
            for arm, ident in profiles.items():
                fp = ident.get("fingerprint") if isinstance(ident, Mapping) else None
                if not (isinstance(ident, Mapping)
                        and set(ident) == {"profile_id", "profile_version", "fingerprint"}
                        and isinstance(fp, str) and len(fp) == 16 and _HEX.fullmatch(fp)):
                    errors.append(f"arm_profiles.{arm} must hold profile_id, profile_version and "
                                  "a 16-hex fingerprint")
    fz = rec.get("freeze")
    if fz is None:
        return errors
    if not isinstance(fz, Mapping) or set(fz) != set(_FREEZE_RECORD_KEYS):
        return errors + [f"freeze must hold exactly {list(_FREEZE_RECORD_KEYS)}"]
    if not (isinstance(fz["approved_by"], str) and fz["approved_by"].strip()):
        errors.append("freeze.approved_by must be a non-empty string")
    if not (isinstance(fz["approved_on"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}",
                                                               fz["approved_on"])):
        errors.append("freeze.approved_on must be YYYY-MM-DD")
    sup = fz["supersedes"]
    if not (isinstance(sup, Mapping) and set(sup) == {"protocol_version", "fingerprint"}):
        errors.append("freeze.supersedes must name the working protocol's version and fingerprint")
    rv = fz["reverification"]
    if not (isinstance(rv, Mapping) and set(rv) == {"test", "requirement"}
            and all(isinstance(rv[k], str) and rv[k] for k in rv)):
        errors.append("freeze.reverification must name the real-data test and its requirement")
    decisions = fz["decisions"]
    ids = [d.get("id") if isinstance(d, Mapping) else None for d in decisions or ()]
    if tuple(ids) != FREEZE_DECISIONS:
        errors.append(f"freeze.decisions must resolve {list(FREEZE_DECISIONS)}, in order")
    for d in decisions or ():
        if not (isinstance(d, Mapping) and set(d) == set(_DECISION_KEYS)):
            errors.append(f"freeze decision {d!r} must hold exactly {list(_DECISION_KEYS)}")
            continue
        if d["resolution"] not in DECISION_RESOLUTIONS:
            errors.append(f"{d['id']}: resolution must be one of {list(DECISION_RESOLUTIONS)}")
        if d["kind"] not in DECISION_KINDS:
            errors.append(f"{d['id']}: kind must be one of {list(DECISION_KINDS)}")
        if not all(isinstance(d[k], str) and d[k].strip() for k in ("decision", "rationale")):
            errors.append(f"{d['id']}: decision and rationale must be non-empty")
    return errors


def validate_protocol_record(rec: Any) -> List[str]:
    """Every problem with a protocol record (an empty list means valid)."""
    try:
        return _validate(rec)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return [f"malformed protocol record: {type(exc).__name__}: {exc}"]


def _validate(rec: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(rec, Mapping):
        return ["the protocol record must be a JSON object"]
    missing = [k for k in _TOP_KEYS if k not in rec]
    extra = sorted(set(rec) - set(_TOP_KEYS) - set(_FREEZE_KEYS))
    if missing:
        errors.append(f"missing keys {missing}")
    if extra:
        errors.append(f"unknown keys {extra}")
    if missing:
        return errors
    bad_sections = [k for k in _SECTIONS if not isinstance(rec[k], Mapping)]
    if bad_sections or not isinstance(rec["pending_approvals"], (list, tuple)):
        return errors + [f"sections must be JSON objects: {bad_sections}"
                         if bad_sections else "pending_approvals must be a list"]
    if rec["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if rec["status"] not in (STATUS_WORKING, STATUS_FROZEN):
        errors.append(f"status must be {STATUS_WORKING!r} or {STATUS_FROZEN!r}")
    version = rec["protocol_version"]
    if not (isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?",
                                                      version)):
        errors.append("protocol_version must be MAJOR.MINOR.PATCH[-label]")
    elif rec["status"] == STATUS_FROZEN and "-" in version:
        errors.append("a frozen protocol's version must not carry a pre-release label")
    elif rec["status"] == STATUS_WORKING and "-" not in version:
        errors.append("a working protocol's version must carry a pre-release label")
    errors += _freeze_errors(rec)

    c = rec["contract"]
    for k in ("contract_id", "contract_version", "contract_fingerprint"):
        if not isinstance(c.get(k), str) or not c.get(k):
            errors.append(f"contract.{k} must be a non-empty string")
    if isinstance(c.get("contract_fingerprint"), str) and not (
            len(c["contract_fingerprint"]) == 16 and _HEX.fullmatch(c["contract_fingerprint"])):
        errors.append("contract.contract_fingerprint must be 16 lowercase hex")

    p = rec["population"]
    rows = _int(p.get("rows"), "population.rows", errors, minimum=1)
    pos = _int(p.get("positives"), "population.positives", errors, minimum=1)
    k = _int(p.get("outer_folds"), "population.outer_folds", errors, minimum=2)
    _int(p.get("fold_seed"), "population.fold_seed", errors)
    fr, fp = p.get("fold_rows"), p.get("fold_positives")
    if not (isinstance(fr, (list, tuple)) and isinstance(fp, (list, tuple))) or (
            k is not None and (len(fr) != k or len(fp) != k)):
        errors.append("population.fold_rows / fold_positives must list one value per outer fold")
    elif rows is not None and pos is not None and (sum(fr) != rows or sum(fp) != pos):
        errors.append("population.fold_rows / fold_positives must sum to rows / positives")
    for key, width in (("router_split_fingerprint", 16), ("train_label_fingerprint", 12),
                       ("fold_fingerprint", 16), ("stage_io_split_fingerprint", 12)):
        v = p.get(key)
        if not (isinstance(v, str) and len(v) == width and _HEX.fullmatch(v)):
            errors.append(f"population.{key} must be {width} lowercase hex")

    h = rec["hierarchy"]
    gates = [g.get("id") for g in h.get("validity_gates", ())]
    if not gates or len(set(gates)) != len(gates):
        errors.append("hierarchy.validity_gates must list unique gate ids")
    outcomes = h.get("primary_outcomes", ())
    if not outcomes:
        errors.append("hierarchy.primary_outcomes must not be empty")
    for o in outcomes:
        if not _known_metric(o.get("metric")):
            errors.append(f"primary outcome {o.get('id')}: unknown metric {o.get('metric')!r}")
    for m in tuple(h.get("descriptive_totals", ())) + tuple(h.get("secondary_diagnostics", ())):
        if not _known_metric(m):
            errors.append(f"unknown diagnostic metric {m!r}")

    u = rec["upstream"]
    if tuple(u.get("operating_points", ())) != tuple(PRIMARY_OPERATING_POINTS):
        errors.append(f"upstream.operating_points must be the primary operating points "
                      f"{list(PRIMARY_OPERATING_POINTS)}")
    arts = u.get("artifacts")
    if not isinstance(arts, Mapping) or set(arts) != {"lr", "ebm"}:
        errors.append("upstream.artifacts must record the 'lr' and 'ebm' artifacts")
    else:
        for model, a in arts.items():
            sha = a.get("sha256") if isinstance(a, Mapping) else None
            if not (isinstance(sha, str) and len(sha) == 64 and _HEX.fullmatch(sha)):
                errors.append(f"upstream.artifacts.{model}.sha256 must be 64 lowercase hex")
            if not (isinstance(a, Mapping) and isinstance(a.get("path"), str) and a["path"]):
                errors.append(f"upstream.artifacts.{model}.path must be recorded")
    if u.get("flag_fingerprint_method") != FLAG_FINGERPRINT_METHOD:
        errors.append(f"upstream.flag_fingerprint_method must be {FLAG_FINGERPRINT_METHOD!r}")
    fps = u.get("flag_fingerprints")
    if not isinstance(fps, Mapping) or set(fps) != set(PRIMARY_OPERATING_POINTS):
        errors.append("upstream.flag_fingerprints must name the primary operating points")
    else:
        for name, v in fps.items():
            if v is not None and not (isinstance(v, str) and len(v) == 16 and _HEX.fullmatch(v)):
                errors.append(f"upstream.flag_fingerprints.{name} must be 16 lowercase hex or null")
            if v is None and rec["status"] == STATUS_FROZEN:
                errors.append(f"a frozen protocol must pin upstream.flag_fingerprints.{name}")
    if not isinstance(u.get("ebm_fold_thresholds"), (list, tuple)) or (
            k is not None and len(u["ebm_fold_thresholds"]) != k):
        errors.append("upstream.ebm_fold_thresholds must list one threshold per outer fold")

    mb = rec["matched_budget"]
    if mb.get("population") != POP_COMMON:
        errors.append(f"matched_budget.population must be {POP_COMMON!r} (Step 2C)")
    if dict(mb.get("scores", {})) != dict(MATCH_SCORES):
        errors.append("matched_budget.scores must equal contribution.MATCH_SCORES")
    if mb.get("tie_break") != TIE_BREAK_RULE:
        errors.append("matched_budget.tie_break must equal contribution.TIE_BREAK_RULE")
    if mb.get("ranking") != "highest" or mb.get("flag_event") != PASS_2_SUBSCRIBE:
        errors.append("matched_budget must rank highest scores for the PASS_2_SUBSCRIBE event")

    un = rec["uncertainty"]
    if un.get("method") != "stratified_paired_bootstrap":
        errors.append("uncertainty.method must be 'stratified_paired_bootstrap'")
    if un.get("procedure_id") != BOOTSTRAP_PROCEDURE_ID:
        errors.append(f"uncertainty.procedure_id must be {BOOTSTRAP_PROCEDURE_ID!r} (the "
                      "implemented procedure's versioned identity)")
    if tuple(un.get("strata", ())) != ("outer_fold", "label"):
        errors.append("uncertainty.strata must be ['outer_fold', 'label']")
    _int(un.get("replicates"), "uncertainty.replicates", errors, minimum=1000)
    _int(un.get("seed"), "uncertainty.seed", errors)
    lvl = un.get("level")
    if not (isinstance(lvl, float) and 0.5 < lvl < 1):
        errors.append("uncertainty.level must be a float in (0.5, 1)")
    if un.get("interval") != "percentile" or un.get("quantile_method") != "linear":
        errors.append("uncertainty must use percentile intervals with the 'linear' quantile")

    sf = rec["shortfall"]
    mat = sf.get("materiality", {})
    if mat.get("rule") != "binomial_se_at_limit":
        errors.append("shortfall.materiality.rule must be 'binomial_se_at_limit'")
    mult = _fraction(mat.get("multiplier"), "shortfall.materiality.multiplier", errors)
    if mult is not None and mult <= 0:
        errors.append("shortfall.materiality.multiplier must be positive")
    _int(sf.get("low_support_threshold"), "shortfall.low_support_threshold", errors, minimum=1)

    fs = rec["fit_status"]
    acc, nacc = tuple(fs.get("accepted", ())), tuple(fs.get("not_accepted", ()))
    if set(acc) != set(ACCEPTED_FIT_STATUSES) or set(acc) | set(nacc) != set(FIT_STATUSES) \
            or set(acc) & set(nacc):
        errors.append("fit_status must partition FIT_STATUSES with accepted = "
                      f"{list(ACCEPTED_FIT_STATUSES)}")
    reports = fs.get("glass_selection_reports", GLASS_REPORTS_OPTIONAL)
    if reports not in (GLASS_REPORTS_REQUIRED, GLASS_REPORTS_OPTIONAL):
        errors.append("fit_status.glass_selection_reports must be 'required' or 'optional'")
    elif rec["status"] == STATUS_FROZEN and reports != GLASS_REPORTS_REQUIRED:
        errors.append("a frozen protocol must require the GLASS selection reports (D-C11)")

    attempted = rec["folds"].get("attempted")
    if k is not None and tuple(attempted or ()) != tuple(range(k)):
        errors.append(f"folds.attempted must be 0..{k - 1 if k else 0}")

    trig = _fraction(rec["f31"].get("trigger"), "f31.trigger", errors)
    if trig is not None and trig != Fraction(1, 20):
        errors.append("f31.trigger must be 1/20 (F31 R2, approved in Step 3C)")

    et = rec["evidence_tiers"]
    srcs = et.get("sources", {})
    if srcs.get("bank_marketing_test_split") != "previously_exposed":
        errors.append("the Bank Marketing test split must be tagged 'previously_exposed'")
    bad = sorted(s for s, t in srcs.items() if t not in ("tier1", "tier2", "tier3",
                                                         "previously_exposed"))
    if bad:
        errors.append(f"evidence_tiers.sources with unknown tiers: {bad}")
    if et.get("tier3_available") is not (any(t == "tier3" for t in srcs.values())):
        errors.append("evidence_tiers.tier3_available must agree with the source registry")

    cm = rec["complementarity"]
    if cm.get("reference") not in _COMBINATION_KEYS or cm.get("reference") not in tuple(
            rec["upstream"].get("combinations", ())):
        errors.append("complementarity.reference must be one of upstream.combinations")
    cr = cm.get("criteria", {})
    want = ("C1_unique_tp_gain", "C2_pass1_leakage", "C3_obligations", "C4_coverage",
            "C5_fold_reproducibility", "C6_uncertainty", "C7_solver", "C8_class_conditional",
            "C9_complexity")
    if tuple(cr) != want:
        errors.append(f"complementarity.criteria must be {list(want)} in order")
    else:
        _fraction(cr["C1_unique_tp_gain"].get("min_relative_gain"), "C1", errors)
        _fraction(cr["C2_pass1_leakage"].get("otherwise_upper_bound_at_most"), "C2", errors)
        _fraction(cr["C4_coverage"].get("total_coverage_lower_bound_at_least"), "C4", errors)
        _int(cr["C5_fold_reproducibility"].get("folds_with_nonnegative_unique_tp_gain_at_least"),
             "C5", errors, minimum=1)
        if not isinstance(cr["C5_fold_reproducibility"].get(
                "fold_counts_only_without_pass2_budget_inflation"), bool):
            errors.append("C5.fold_counts_only_without_pass2_budget_inflation must be a boolean")
        _fraction(cr["C7_solver"].get("max_runtime_ratio"), "C7", errors)
        for key in ("pass2_negative_routing_rate_upper_bound_at_most",
                    "pass1_negative_recall_lower_bound_at_least"):
            _fraction(cr["C8_class_conditional"].get(key), f"C8.{key}", errors)

    ac = rec["acceptance"]
    if tuple(ac.get("structural", ())) != tuple(gates):
        errors.append("acceptance.structural must list every validity gate, in order")
    tokens = tuple(rec["inputs"].get("prohibited_tokens", ()))
    if not set(TEST_DERIVED_FIELD_TOKENS) <= set(tokens):
        errors.append(f"inputs.prohibited_tokens must include {list(TEST_DERIVED_FIELD_TOKENS)}")
    if rec["status"] == STATUS_FROZEN and rec["pending_approvals"]:
        errors.append("a frozen protocol may not list pending approvals")
    return errors


@dataclass(frozen=True)
class EvaluationProtocol:
    """A validated protocol record (read-only)."""

    record: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return self.record["protocol_id"]

    @property
    def version(self) -> str:
        return self.record["protocol_version"]

    @property
    def status(self) -> str:
        return self.record["status"]

    def __getitem__(self, key: str) -> Any:
        return self.record[key]

    def to_dict(self) -> dict:
        return _thaw(self.record)

    def fingerprint(self) -> str:
        """SHA-256 (16 hex) of the canonical JSON of the whole record."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, rec: Mapping) -> "EvaluationProtocol":
        errors = validate_protocol_record(rec)
        if errors:
            raise ProtocolError("invalid evaluation protocol:\n  - " + "\n  - ".join(errors))
        return cls(_freeze(json.loads(json.dumps(rec))))


def working_protocol_path() -> Path:
    """The Step 4C working protocol. Not frozen: it may not govern PR 38."""
    return PROTOCOLS_DIR / _WORKING


def frozen_protocol_path() -> Path:
    """The Step 4D frozen protocol (``pr38-stage2-evaluation`` 1.0.0): the one PR 38 runs under."""
    return PROTOCOLS_DIR / _FROZEN


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ProtocolError(f"duplicate key {k!r} in the protocol record")
        out[k] = v
    return out


def load_evaluation_protocol(path, *, expected_fingerprint: Optional[str] = None,
                             require_status: Optional[str] = None) -> EvaluationProtocol:
    """Load and validate a protocol record; optionally pin its fingerprint and status."""
    text = Path(path).read_text(encoding="utf-8")
    protocol = EvaluationProtocol.from_dict(json.loads(text, object_pairs_hook=_no_duplicate_keys))
    if expected_fingerprint is not None and protocol.fingerprint() != expected_fingerprint:
        raise ProtocolError(f"protocol fingerprint {protocol.fingerprint()} != expected "
                            f"{expected_fingerprint}")
    if require_status is not None and protocol.status != require_status:
        raise ProtocolError(f"protocol status {protocol.status!r} != required {require_status!r}")
    return protocol


def load_frozen_protocol(path=None) -> EvaluationProtocol:
    """
    The frozen PR 38 protocol, refused unless its status is ``"frozen"``, its
    fingerprint is ``FROZEN_PROTOCOL_FINGERPRINT`` and its freeze approval is
    confirmed (``require_frozen``). Its pinned upstream values are verified
    again on the real artifacts by ``tests/realdata/test_stage2_freeze_artifacts.py``
    (D-C10), which must pass before the freeze is relied on. The only way PR 38
    obtains its protocol; there is no review or override switch.
    """
    protocol = load_evaluation_protocol(path or frozen_protocol_path(),
                                        expected_fingerprint=FROZEN_PROTOCOL_FINGERPRINT,
                                        require_status=STATUS_FROZEN)
    return require_frozen(protocol)


def is_pinned_frozen(protocol: EvaluationProtocol) -> bool:
    """
    Whether ``protocol`` is the pinned frozen record itself (status, confirmed
    approval and ``FROZEN_PROTOCOL_FINGERPRINT``). Any other frozen record,
    however it was built, may be used to exercise the protocol but never
    yields an official decision.
    """
    return (isinstance(protocol, EvaluationProtocol) and protocol.status == STATUS_FROZEN
            and freeze_approval_confirmed(protocol)
            and protocol.fingerprint() == FROZEN_PROTOCOL_FINGERPRINT)


@dataclass(frozen=True)
class FrozenSpecification:
    """The pinned frozen protocol and the frozen contract it binds, both strictly loaded."""

    protocol: EvaluationProtocol
    contract: SharedOperatingContractV2


def load_frozen_specification() -> FrozenSpecification:
    """
    PR 38's arm-neutral entry point: ``load_frozen_protocol()`` and
    ``load_frozen_contract()`` (status, pinned fingerprint, confirmed
    approval), bound by ``check_contract_binding``. Takes no record from the
    caller. The arms' configs come from ``glass_router.profile.glass_primary_config()``
    and ``rf_router.profile.rf_primary_config()``, checked against this
    protocol with ``check_arm_profile_binding``.
    """
    from shared.stage2.contract import load_frozen_contract
    protocol = load_frozen_protocol()
    contract = load_frozen_contract()
    check_contract_binding(protocol, contract)
    return FrozenSpecification(protocol, contract)


def freeze_approval_confirmed(protocol: EvaluationProtocol) -> bool:
    """Whether a frozen protocol's ``freeze`` block names a confirmed approver."""
    fz = protocol.record.get("freeze")
    return isinstance(fz, Mapping) and approval_confirmed(fz)


def require_frozen(protocol: EvaluationProtocol) -> EvaluationProtocol:
    """PR 38 evaluates only under a frozen protocol."""
    if not isinstance(protocol, EvaluationProtocol):
        raise ProtocolError(f"expected an EvaluationProtocol, got {type(protocol).__name__}")
    if protocol.status != STATUS_FROZEN:
        raise ProtocolError(f"protocol {protocol.protocol_id} {protocol.version} is "
                            f"{protocol.status!r}: only a frozen protocol may govern PR 38 "
                            "(Step 4D freezes it)")
    if not freeze_approval_confirmed(protocol):
        raise ProtocolError(f"protocol {protocol.protocol_id} {protocol.version} is a review "
                            "candidate: its freeze approval is unconfirmed, so it may not govern "
                            "PR 38")
    return protocol


def _gate(protocol, allow_working: bool) -> None:
    if not allow_working:
        require_frozen(protocol)


def check_contract_binding(protocol: EvaluationProtocol,
                           contract: SharedOperatingContractV2) -> None:
    """The contract must be exactly the one the protocol names."""
    if not isinstance(contract, SharedOperatingContractV2):
        raise ProtocolError("the protocol binds a schema-2 contract")
    c = protocol["contract"]
    got = (contract.contract_id, contract.contract_version, contract.fingerprint())
    want = (c["contract_id"], c["contract_version"], c["contract_fingerprint"])
    if got != want:
        raise ProtocolError(f"contract {got} is not the contract the protocol binds {want}")
    if protocol.status == STATUS_FROZEN and contract.status != STATUS_FROZEN:
        raise ProtocolError("a frozen protocol binds only a frozen contract (D-C9)")


def check_arm_profile_binding(protocol: EvaluationProtocol, profile) -> None:
    """
    An arm's configuration profile (``shared.stage2.arm_profile.ArmProfile``)
    must be exactly the one the protocol records for that arm (id, version,
    fingerprint); under a frozen protocol it must itself be frozen.
    """
    from shared.stage2.arm_profile import ArmProfile
    if not isinstance(profile, ArmProfile):
        raise ProtocolError("expected an ArmProfile (shared.stage2.arm_profile)")
    bound = protocol.record.get("arm_profiles")
    if not isinstance(bound, Mapping) or profile.arm not in bound:
        raise ProtocolError(f"the protocol binds no {profile.arm!r} arm profile")
    want = dict(bound[profile.arm])
    if profile.identity() != want:
        raise ProtocolError(f"{profile.arm} profile {profile.identity()} is not the profile the "
                            f"protocol binds {want}")
    if protocol.status == STATUS_FROZEN and profile.status != STATUS_FROZEN:
        raise ProtocolError("a frozen protocol binds only frozen arm profiles")


def check_inputs(names: Iterable[str], protocol: Optional[EvaluationProtocol] = None) -> None:
    """
    Refuse any input name with a test-derived segment (``test``, ``holdout``, ``oracle``).

    Defense in depth only: a name says nothing about where data came from, and a
    renamed test-derived value passes. Provenance is established by the checks
    the PR 38 runner must run (protocol document, section 15): the allowlisted
    boundary extractors, artifact SHA-256 against the pinned records, the
    population and upstream identity checks and the evidence-source registry.
    """
    tokens = set(protocol["inputs"]["prohibited_tokens"]) if protocol else set(
        TEST_DERIVED_FIELD_TOKENS)
    bad = sorted(n for n in names if set(_SEGMENT_RE.findall(str(n).lower())) & tokens)
    if bad:
        raise ProtocolError(f"test-derived inputs are prohibited in PR 38 evaluation, selection "
                            f"and adoption decisions: {bad}")


# ======================================================================
# Population identity (V1)
# ======================================================================
@dataclass(frozen=True)
class PopulationIdentity:
    rows: int
    positives: int
    outer_folds: int
    fold_seed: int
    fold_rows: Tuple[int, ...]
    fold_positives: Tuple[int, ...]
    router_split_fingerprint: str
    train_label_fingerprint: str
    fold_fingerprint: str

    @classmethod
    def from_protocol(cls, protocol: EvaluationProtocol) -> "PopulationIdentity":
        p = protocol["population"]
        return cls(p["rows"], p["positives"], p["outer_folds"], p["fold_seed"],
                   tuple(p["fold_rows"]), tuple(p["fold_positives"]),
                   p["router_split_fingerprint"], p["train_label_fingerprint"],
                   p["fold_fingerprint"])


@dataclass(frozen=True)
class IdentityReport:
    """
    Identity checks. ``passed`` is True, False, or None for a check that could
    not run because its pinned value is not recorded yet (``pending``); only a
    working protocol can leave a value unpinned. ``values`` holds computed
    identities (for example flag fingerprints to pin).
    """

    checks: Tuple[Tuple[str, Optional[bool], str], ...]
    values: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def passed(self) -> bool:
        return all(ok is not False for _, ok, _ in self.checks)

    @property
    def complete(self) -> bool:
        return all(ok is not None for _, ok, _ in self.checks)

    def failures(self) -> List[str]:
        return [f"{name}: {detail}" for name, ok, detail in self.checks if ok is False]

    def pending(self) -> List[str]:
        return [f"{name}: {detail}" for name, ok, detail in self.checks if ok is None]

    def require(self, *, complete: bool = False) -> "IdentityReport":
        if not self.passed:
            raise ProtocolError("identity check failed:\n  - " + "\n  - ".join(self.failures()))
        if complete and not self.complete:
            raise ProtocolError("identity check incomplete (unpinned values):\n  - "
                                + "\n  - ".join(self.pending()))
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks, columns=["check", "passed", "detail"])


def _declared_fold_fp(view) -> Optional[str]:
    proto = view.metadata.get("oof_protocol", {}) if isinstance(view.metadata, Mapping) else {}
    return proto.get("fold_fingerprint") if isinstance(proto, Mapping) else None


def check_population_identity(trace, identity: PopulationIdentity) -> IdentityReport:
    """
    A validated ``RouterTrace`` (checks T1-T7 already passed) against the pinned
    population: rows, positives, per-fold counts, fold plan and fingerprints.
    """
    from shared.stage2.diagnostics.trace import RouterTrace
    from shared.stage_io import fold_assignment, label_fingerprint

    if not isinstance(trace, RouterTrace):
        raise ProtocolError("expected a RouterTrace from build_router_trace")
    y = np.asarray(trace.y).astype(int)
    fold = np.asarray(trace.fold_id).astype(int)
    checks: List[Tuple[str, bool, str]] = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), str(detail)))

    add("rows", trace.n == identity.rows, f"{trace.n} vs {identity.rows}")
    add("positives", int(y.sum()) == identity.positives, f"{int(y.sum())} vs {identity.positives}")
    folds = sorted(int(k) for k in np.unique(fold))
    add("outer folds", folds == list(range(identity.outer_folds)), f"{folds}")
    by_rows = tuple(int((fold == k).sum()) for k in range(identity.outer_folds))
    by_pos = tuple(int(y[fold == k].sum()) for k in range(identity.outer_folds))
    add("rows per fold", by_rows == identity.fold_rows, f"{by_rows} vs {identity.fold_rows}")
    add("positives per fold", by_pos == identity.fold_positives,
        f"{by_pos} vs {identity.fold_positives}")
    plan = fold_assignment(y, cv_folds=identity.outer_folds,
                           random_state=identity.fold_seed).to_numpy()
    add("fold plan is the seeded stratified plan", np.array_equal(plan, fold),
        f"{int((plan != fold).sum())} row(s) differ from StratifiedKFold("
        f"{identity.outer_folds}, seed {identity.fold_seed})")
    for arm, view in trace.views.items():
        fp = _declared_fold_fp(view)
        add(f"{arm} fold fingerprint", fp == identity.fold_fingerprint,
            f"{fp} vs {identity.fold_fingerprint}")
    add("router split fingerprint", trace.router_split_fingerprint ==
        identity.router_split_fingerprint,
        f"{trace.router_split_fingerprint} vs {identity.router_split_fingerprint}")
    recomputed = label_fingerprint(trace.index, y)
    add("train label fingerprint", trace.train_label_fingerprint ==
        identity.train_label_fingerprint == recomputed,
        f"recorded {trace.train_label_fingerprint}, recomputed {recomputed}, pinned "
        f"{identity.train_label_fingerprint}")
    return IdentityReport(tuple(checks))


def flag_fingerprint(index, flags) -> str:
    """
    Row-aligned identity of one flag set (``FLAG_FINGERPRINT_METHOD``): SHA-256
    (16 hex) over every row's index label (string form, as the
    ``shared.stage_io`` fingerprints) and its 0/1 flag, in row order. Moving a
    flag from one row to another changes it even when every count is kept.
    """
    idx = pd.Index(index)
    f = np.asarray(flags)
    if f.shape != (len(idx),) or not np.isin(f, (0, 1, True, False)).all():
        raise ProtocolError("flag_fingerprint needs one 0/1 flag per row")
    h = hashlib.sha256(FLAG_FINGERPRINT_METHOD.encode("utf-8"))
    for label, v in zip(idx, f.astype(bool)):
        h.update(f"{label}\t{int(v)}\n".encode("utf-8"))
    return h.hexdigest()[:16]


#: ``suppression.primary_reference_flags`` key -> the protocol's ``upstream.expected_flags`` key.
_UPSTREAM_KEYS = MappingProxyType({"lr": "lr.in_stage_f2_cv", "ebm": "ebm.nested_f2",
                                   "union": "or", "intersection": "and"})
_SHA_PREFIX = re.compile(r"^([0-9a-f]{64})")


def _extraction_digest(view) -> Optional[str]:
    """The SHA-256 the boundary extractor recorded when it opened the artifact."""
    for item in view.provenance:
        if item.name == "file_sha256":
            m = _SHA_PREFIX.match(str(item.detail or ""))
            return m.group(1) if m else None
    return None


def check_upstream_identity(refs: Mapping[str, Any], protocol: EvaluationProtocol, *,
                            lr_view: UpstreamOOFView, ebm_view: UpstreamOOFView,
                            population) -> IdentityReport:
    """
    The frozen LR and EBM operating points, row by row.

    ``refs``: ``suppression.primary_reference_flags(lr_view, ebm_view)``.
    ``population``: the evaluation rows (a ``PairedRouting`` or ``RouterTrace``).

    1. Artifact identity: each view came from a file (``source.path``) whose
       SHA-256, both when the extractor opened it and now, is the one the
       baseline manifest records (``upstream.artifacts``).
    2. Rows: each view covers the evaluation rows, in order, with their labels;
       the EBM's fold ids are the evaluation folds.
    3. Threshold provenance: the LR threshold record (value, field,
       global selection); the EBM's fold-nested thresholds (one per fold,
       equal to the pinned values) and every row's threshold equal to its
       fold's.
    4. Decisions: each supplied flag set equals the registered operating point
       recomputed from its view, row by row; the union and intersection equal
       the recomputed OR / AND.
    5. Counts, and the row-aligned flag fingerprints (pending until pinned, D-C10).
    """
    from shared.stage2.diagnostics.boundary import file_sha256

    up = protocol["upstream"]
    checks: List[Tuple[str, Optional[bool], str]] = []
    values: Dict[str, str] = {}

    def add(name, ok, detail):
        checks.append((name, None if ok is None else bool(ok), str(detail)))

    idx, y, fold = pd.Index(population.index), np.asarray(population.y), np.asarray(
        population.fold_id)
    recomputed: Dict[str, np.ndarray] = {}
    for model, view in (("lr", lr_view), ("ebm", ebm_view)):
        if not isinstance(view, UpstreamOOFView) or view.model != model:
            add(f"{model} view", False, f"expected the {model} UpstreamOOFView")
            continue
        pinned = up["artifacts"][model]["sha256"]
        path = view.source.get("path") if isinstance(view.source, Mapping) else None
        add(f"{model} artifact path recorded", path is not None,
            path if path else "the view was not extracted from a file")
        at_extraction = _extraction_digest(view)
        add(f"{model} artifact SHA-256 at extraction", at_extraction == pinned,
            f"{at_extraction} vs {pinned}")
        now = file_sha256(path) if path is not None and Path(path).is_file() else None
        add(f"{model} artifact SHA-256 now", now == pinned, f"{now} vs {pinned}")
        aligned = view.index.equals(idx) and np.array_equal(np.asarray(view.y), y)
        add(f"{model} rows and labels are the evaluation rows", aligned,
            "same rows, order and labels" if aligned else "rows, order or labels differ")
        name = _UPSTREAM_KEYS[model]
        try:
            recomputed[model] = np.asarray(resolve_operating_point(view, name).flags, bool)
        except Exception as exc:  # noqa: BLE001 - reported as a failed check
            add(f"{name} resolves from its view", False, f"{type(exc).__name__}: {exc}")

    if isinstance(lr_view, UpstreamOOFView) and lr_view.model == "lr":
        rec = lr_view.thresholds.get("lr.in_stage_f2_cv")
        ok = (rec is not None and float(rec.value) == float(up["lr_threshold"])
              and rec.field == up["lr_threshold_field"]
              and rec.selection == SELECTION_GLOBAL_TRAIN_LABELS)
        add("lr.in_stage_f2_cv threshold record", ok,
            "missing" if rec is None else f"{rec.value} / {rec.field} vs {up['lr_threshold']} / "
            f"{up['lr_threshold_field']}")
    if isinstance(ebm_view, UpstreamOOFView) and ebm_view.model == "ebm":
        add("ebm fold ids are the evaluation folds", np.array_equal(
            np.asarray(ebm_view.fold_id), fold), "EBM shares the routers' outer folds")
        sd = ebm_view.stored_decisions.get("ebm.nested_f2")
        want = tuple(float(v) for v in up["ebm_fold_thresholds"])
        if sd is None:
            add("ebm.nested_f2 fold thresholds", False, "no stored nested decisions")
        else:
            add("ebm.nested_f2 fold thresholds", tuple(float(v) for v in sd.by_fold) == want
                and sd.field == up["ebm_decision_field"] and sd.selection == SELECTION_FOLD_NESTED,
                f"{tuple(sd.by_fold)} / {sd.field} vs {want} / {up['ebm_decision_field']}")
            per_row = np.asarray(sd.per_row_threshold, float)
            ebm_fold = np.asarray(ebm_view.fold_id).astype(int)
            expect = (np.asarray(sd.by_fold, float)[ebm_fold]
                      if len(sd.by_fold) and ebm_fold.max() < len(sd.by_fold) else None)
            add("ebm.nested_f2 per-row thresholds are their fold's",
                expect is not None and np.array_equal(per_row, expect),
                "every row's threshold equals its fold's nested threshold")

    if "lr" in recomputed and "ebm" in recomputed:
        recomputed["union"] = recomputed["lr"] | recomputed["ebm"]
        recomputed["intersection"] = recomputed["lr"] & recomputed["ebm"]
    for key, pinned in _UPSTREAM_KEYS.items():
        flags = refs.get(key)
        if not isinstance(flags, UpstreamFlags):
            add(f"{pinned} flags supplied", False, "expected UpstreamFlags")
            continue
        got = np.asarray(flags.flags, bool)
        if key in recomputed:
            same = flags.index.equals(idx) and np.array_equal(got, recomputed[key])
            differ = (int((got != recomputed[key]).sum()) if got.shape == recomputed[key].shape
                      else "n/a")
            add(f"{pinned} flags equal the registered operating point, row by row", same,
                "identical" if same else f"{differ} row(s) differ or rows are misaligned")
        add(f"{pinned} flag count", int(got.sum()) == int(up["expected_flags"][pinned]),
            f"{int(got.sum())} vs {up['expected_flags'][pinned]}")
    for model in ("lr", "ebm"):
        name = _UPSTREAM_KEYS[model]
        if model not in recomputed:
            continue
        fp = flag_fingerprint(idx, recomputed[model])
        values[name] = fp
        pinned_fp = up["flag_fingerprints"][name]
        if pinned_fp is None:
            add(f"{name} row-aligned flag fingerprint", None,
                f"{fp}; not pinned in this working protocol (D-C10)")
        else:
            add(f"{name} row-aligned flag fingerprint", fp == pinned_fp, f"{fp} vs {pinned_fp}")
    return IdentityReport(tuple(checks), MappingProxyType(values))


# ======================================================================
# Paired routings
# ======================================================================
@dataclass(frozen=True)
class PairedRouting:
    """
    Routings of the same rows, aligned by row identity. ``kind`` is
    ``"cross_arm"`` (GLASS vs RF), ``"pre_post"`` (one arm before and after
    correction) or ``"ablation"`` (ablation vs primary; differences are
    ablation - primary). Built only by the constructors, which refuse different
    rows, labels or fold ids instead of aligning by position.
    """

    kind: str
    index: pd.Index
    y: np.ndarray
    fold_id: np.ndarray
    states: Mapping[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.states)

    @staticmethod
    def build(kind: str, index, y, fold_id, states: Mapping[str, Any]) -> "PairedRouting":
        if kind not in _PAIR_KINDS:
            raise ProtocolError(f"kind must be one of {_PAIR_KINDS}, got {kind!r}")
        idx = pd.Index(index)
        if not idx.is_unique:
            raise ProtocolError("row identities must be unique")
        lab = np.asarray(y)
        fold = np.asarray(fold_id)
        if lab.shape != (len(idx),) or fold.shape != (len(idx),):
            raise ProtocolError("y and fold_id need one value per row")
        if not np.isin(lab, (0, 1)).all():
            raise ProtocolError("y must be binary 0/1")
        if not np.issubdtype(fold.dtype, np.integer):
            raise ProtocolError("fold_id must be integers")
        if len(states) != 2:
            raise ProtocolError("a paired routing has exactly two routings")
        fixed = _PAIR_NAMES.get(kind)
        if fixed is not None and tuple(states) != fixed:
            raise ProtocolError(f"a {kind} pairing names its routings {fixed}, in that order")
        out = {}
        for name, st in states.items():
            a = np.asarray(st, dtype=object)
            if a.shape != (len(idx),):
                raise ProtocolError(f"{name}: one state per row required")
            bad = sorted({str(s) for s in a if s not in ROUTER_STATES})
            if bad:
                raise ProtocolError(f"{name}: unknown states {bad}; expected {ROUTER_STATES}")
            a = a.copy()
            a.setflags(write=False)
            out[str(name)] = a
        y_ro, f_ro = lab.astype(int).copy(), fold.astype(int).copy()
        y_ro.setflags(write=False)
        f_ro.setflags(write=False)
        return PairedRouting(kind, idx, y_ro, f_ro, MappingProxyType(out))

    @classmethod
    def from_trace(cls, trace) -> "PairedRouting":
        """GLASS vs RF on a validated trace (T1-T7 guarantee identical rows, labels, folds)."""
        from shared.stage2.diagnostics.trace import TRACE_ARMS, RouterTrace
        if not isinstance(trace, RouterTrace):
            raise ProtocolError("expected a RouterTrace from build_router_trace")
        return cls.build("cross_arm", trace.index, trace.y, trace.fold_id,
                         {arm: trace.views[arm].state for arm in TRACE_ARMS})

    @classmethod
    def _from_views(cls, kind: str, names: Tuple[str, str], a, b) -> "PairedRouting":
        for name, v in zip(names, (a, b)):
            if not isinstance(v, RouterOOFView):
                raise ProtocolError(f"{name}: expected a RouterOOFView from extract_router_view")
        problems = []
        if a.arm != b.arm:
            problems.append(f"different arms ({a.arm} vs {b.arm})")
        if not a.index.equals(b.index):
            problems.append("row identities differ (same rows in the same order required)")
        elif not np.array_equal(a.y, b.y):
            problems.append("labels differ")
        elif not np.array_equal(a.fold_id, b.fold_id):
            problems.append("outer fold ids differ")
        if problems:
            raise ProtocolError(f"{' / '.join(names)} views are not paired: " + "; ".join(problems))
        return cls.build(kind, a.index, a.y, a.fold_id, {names[0]: a.state, names[1]: b.state})

    @classmethod
    def pre_post(cls, pre: RouterOOFView, post: RouterOOFView) -> "PairedRouting":
        """One arm before (pre-correction baseline views) and after correction."""
        return cls._from_views("pre_post", ("pre", "post"), pre, post)

    @classmethod
    def ablation(cls, primary: RouterOOFView, ablation: RouterOOFView) -> "PairedRouting":
        """The complementarity ablation against the primary corrected arm (ablation first)."""
        return cls._from_views("ablation", ("ablation", "primary"), ablation, primary)

    def digest(self) -> str:
        """SHA-256 of the pairing: kind, names, row labels, labels, folds and both routings."""
        h = hashlib.sha256(f"{self.kind}|{'|'.join(self.names)}|{self.n}\n".encode("utf-8"))
        for label in self.index:
            h.update(f"{label}\n".encode("utf-8"))
        h.update(np.ascontiguousarray(self.y, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(self.fold_id, dtype=np.int64).tobytes())
        for name in self.names:
            h.update("|".join(map(str, self.states[name])).encode("utf-8"))
        return h.hexdigest()

    def _rows(self, fold) -> np.ndarray:
        return np.ones(self.n, bool) if fold is None else (self.fold_id == int(fold))

    def counts(self, name: str, fold=None) -> RoutingCounts:
        st = np.asarray(self.states[name])
        m = self._rows(fold)
        tokens = np.where(st[m] == PASS_1_NOT_SUBSCRIBE, "pass1",
                          np.where(st[m] == PASS_2_SUBSCRIBE, "pass2", "uncertain"))
        return RoutingCounts.from_decisions(tokens, self.y[m])

    @property
    def folds(self) -> Tuple[int, ...]:
        return tuple(int(k) for k in np.unique(self.fold_id))

    def fold_counts(self, name: str) -> Dict[int, RoutingCounts]:
        return {k: self.counts(name, k) for k in self.folds}

    def transitions(self, fold=None) -> pd.DataFrame:
        """3 x 3 cross-tabulation of the two routings, with positives and negatives."""
        a_name, b_name = self.names
        m = self._rows(fold)
        a, b, y = np.asarray(self.states[a_name])[m], np.asarray(self.states[b_name])[m], self.y[m]
        rows = []
        for sa in ROUTER_STATES:
            for sb in ROUTER_STATES:
                cell = (a == sa) & (b == sb)
                rows.append({f"{a_name}_state": sa, f"{b_name}_state": sb,
                             "rows": int(cell.sum()), "positives": int(y[cell].sum()),
                             "negatives": int((y[cell] == 0).sum()), "changed": sa != sb})
        return pd.DataFrame(rows)

    def changed_rows(self, fold=None) -> int:
        a_name, b_name = self.names
        m = self._rows(fold)
        return int((np.asarray(self.states[a_name])[m] != np.asarray(self.states[b_name])[m]).sum())


_PAIR_KINDS = ("cross_arm", "pre_post", "ablation")
_PAIR_NAMES = MappingProxyType({"pre_post": ("pre", "post"), "ablation": ("ablation", "primary")})


@dataclass(frozen=True)
class ReferenceFlags:
    """
    Frozen reference flags for the contribution metrics, indexed by row. A
    pairing accepts them only if the rows, their order and their labels are
    the pairing's own (``aligned_to``); a bare array is never accepted.
    """

    name: str
    index: pd.Index
    y: np.ndarray
    flags: np.ndarray
    selection: str

    @classmethod
    def from_upstream(cls, flags: UpstreamFlags) -> "ReferenceFlags":
        """From ``suppression.primary_reference_flags`` (or ``resolve_operating_point``)."""
        if not isinstance(flags, UpstreamFlags):
            raise ProtocolError("expected UpstreamFlags from the frozen upstream operating points")
        return cls.build(flags.name, flags.index, flags.y, flags.flags, flags.selection)

    @classmethod
    def build(cls, name: str, index, y, flags, selection: str) -> "ReferenceFlags":
        idx = pd.Index(index)
        f, lab = np.asarray(flags), np.asarray(y)
        if not idx.is_unique or f.shape != (len(idx),) or lab.shape != (len(idx),):
            raise ProtocolError("reference flags need unique rows and one flag and label per row")
        if not np.isin(f, (0, 1, True, False)).all() or not np.isin(lab, (0, 1)).all():
            raise ProtocolError("reference flags and labels must be binary")
        f_ro, y_ro = f.astype(bool).copy(), lab.astype(int).copy()
        f_ro.setflags(write=False)
        y_ro.setflags(write=False)
        return cls(str(name), idx, y_ro, f_ro, str(selection))

    @property
    def fingerprint(self) -> str:
        return flag_fingerprint(self.index, self.flags)

    def aligned_to(self, paired: "PairedRouting") -> np.ndarray:
        if not self.index.equals(paired.index):
            raise ProtocolError(f"reference {self.name!r}: rows or their order differ from the "
                                "paired routing; reference flags are never aligned by position")
        if not np.array_equal(self.y, paired.y):
            raise ProtocolError(f"reference {self.name!r}: labels differ from the paired routing")
        return self.flags


#: ``upstream.combinations`` entry -> its ``suppression.primary_reference_flags`` key.
_COMBINATION_KEYS = MappingProxyType({"or": "union", "and": "intersection"})


@dataclass(frozen=True)
class UpstreamSources:
    """
    The frozen upstream reference flags and the views they were resolved from:
    what ``check_upstream_identity`` verifies. ``refs`` is
    ``suppression.primary_reference_flags(lr_view, ebm_view)``; the views come
    from the boundary extractors, opened with ``expected_sha256``.
    """

    refs: Mapping[str, Any]
    lr_view: Any
    ebm_view: Any

    @classmethod
    def from_views(cls, lr_view, ebm_view) -> "UpstreamSources":
        from shared.stage2.diagnostics.suppression import primary_reference_flags
        return cls(primary_reference_flags(lr_view, ebm_view), lr_view, ebm_view)


@dataclass(frozen=True)
class VerifiedReference:
    """Reference flags from upstream sources that passed ``check_upstream_identity``."""

    flags: ReferenceFlags
    combination: str
    identity: IdentityReport

    @property
    def complete(self) -> bool:
        return self.identity.complete


def verify_reference(upstream: UpstreamSources, protocol: EvaluationProtocol, *, population,
                     complete: bool = True) -> VerifiedReference:
    """
    The complementarity reference (``complementarity.reference``: the frozen
    LR-or-EBM union of O4a), taken from ``upstream`` only after
    ``check_upstream_identity`` has passed on ``population``'s rows (artifact
    SHA-256, rows and labels, threshold provenance, per-row decisions, counts,
    fingerprints). ``complete=True``, as for an official decision, also
    requires every pinned value, the row-aligned flag fingerprints included;
    ``complete=False`` allows values still pending (D-C10) and reports them.
    """
    if not isinstance(upstream, UpstreamSources) or not isinstance(upstream.refs, Mapping):
        raise ProtocolError("verified upstream reference provenance needs UpstreamSources (the "
                            "frozen upstream flags and the views they were resolved from), not "
                            f"{type(upstream).__name__}")
    report = check_upstream_identity(upstream.refs, protocol, lr_view=upstream.lr_view,
                                     ebm_view=upstream.ebm_view, population=population)
    try:
        report.require(complete=complete)
    except ProtocolError as exc:
        raise ProtocolError(f"the upstream reference provenance is not verified: {exc}") from exc
    combination = protocol["complementarity"]["reference"]
    flags = ReferenceFlags.from_upstream(upstream.refs[_COMBINATION_KEYS[combination]])
    return VerifiedReference(flags, combination, report)


def protocol_ratios(counts: RoutingCounts) -> Mapping[str, Ratio]:
    """The contract metrics plus the protocol's class-conditional ones, as registry Ratios."""
    c = counts
    neg = c.rows - c.positives
    out = dict(routing_ratios(c))
    out.update({
        "pass1_negative_recall": Ratio("pass1_negative_recall",
                                       c.pass1_rows - c.pass1_positives, neg),
        "pass2_recall": Ratio("pass2_recall", c.pass2_positives, c.positives),
        "pass2_recall_eligible": Ratio("pass2_recall_eligible", c.pass2_positives,
                                       c.positives - c.pass1_positives),
        "pass2_negative_routing_rate": Ratio("pass2_negative_routing_rate",
                                             c.pass2_rows - c.pass2_positives, neg),
        "abstain_positive_rate": Ratio("abstain_positive_rate", c.abstain_positives,
                                       c.abstain_rows),
        "abstain_negative_share": Ratio("abstain_negative_share",
                                        c.abstain_rows - c.abstain_positives, neg),
    })
    return MappingProxyType(out)


def class_conditional_counts(paired: PairedRouting, name: str, fold=None) -> pd.DataFrame:
    """Per state: positives and negatives, and their shares of all positives / negatives."""
    c = paired.counts(name, fold)
    pos = {PASS_1_NOT_SUBSCRIBE: c.pass1_positives, PASS_2_SUBSCRIBE: c.pass2_positives,
           ABSTAIN: c.abstain_positives}
    rows = {PASS_1_NOT_SUBSCRIBE: c.pass1_rows, PASS_2_SUBSCRIBE: c.pass2_rows,
            ABSTAIN: c.abstain_rows}
    neg_total = c.rows - c.positives
    out = []
    for s in ROUTER_STATES:
        p, n = pos[s], rows[s] - pos[s]
        ps, ns = Ratio("positive_share", p, c.positives), Ratio("negative_share", n, neg_total)
        out.append({"state": s, "rows": rows[s], "positives": p, "negatives": n,
                    "share_of_positives": ps.as_float(), "share_of_positives_status": ps.status,
                    "share_of_negatives": ns.as_float(), "share_of_negatives_status": ns.status})
    frame = pd.DataFrame(out)
    if int(frame["positives"].sum()) != c.positives or int(frame["rows"].sum()) != c.rows:
        raise ProtocolError("internal: class-conditional counts do not reconcile")
    return frame


_TABLE_METRICS = ("pass1_rate", "pass2_rate", "coverage", "abstain_rate", "pass1_npv",
                  "pass1_leakage_share", "pass2_precision", "pass2_remainder_rate",
                  "pass1_negative_recall", "pass2_recall", "pass2_recall_eligible",
                  "pass2_negative_routing_rate", "abstain_positive_rate",
                  "abstain_negative_share")


def metric_table(paired: PairedRouting, name: str,
                 metrics: Sequence[str] = _TABLE_METRICS) -> pd.DataFrame:
    """One row per (fold, metric); fold "all" is pooled from summed counts."""
    per_fold = paired.fold_counts(name)
    groups = [(k, c) for k, c in per_fold.items()] + [("all", pool_counts(per_fold.values()))]
    rows = []
    for fold, c in groups:
        ratios = protocol_ratios(c)
        for m in metrics:
            r = ratios[m]
            rows.append({"routing": name, "fold": fold, "metric": m,
                         "numerator": int(r.numerator), "denominator": int(r.denominator),
                         "value": r.value, "status": r.status})
    return pd.DataFrame(rows)


def fold_dispersion(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric: defined / undefined folds, minimum, maximum, range and the pooled value."""
    out = []
    for (routing, metric), g in table.groupby(["routing", "metric"], sort=False):
        folds = g[g["fold"].map(lambda f: f != "all")]
        pooled = g[g["fold"].map(lambda f: f == "all")]
        vals = [v for v in folds["value"] if v is not None and not pd.isna(v)]
        out.append({"routing": routing, "metric": metric, "folds_defined": len(vals),
                    "folds_undefined": int(len(folds) - len(vals)),
                    "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                    "range": (max(vals) - min(vals)) if vals else None,
                    "pooled": pooled["value"].iloc[0] if len(pooled) else None})
    return pd.DataFrame(out)


# ======================================================================
# Uncertainty
# ======================================================================
def _masks(state: np.ndarray, y: np.ndarray, reference: Optional[np.ndarray]):
    p1, p2, ab = state == PASS_1_NOT_SUBSCRIBE, state == PASS_2_SUBSCRIBE, state == ABSTAIN
    pos, neg = y == 1, y == 0
    out = {"rows": np.ones(len(y), bool), "pos": pos, "neg": neg, "p1": p1, "p2": p2, "ab": ab,
           "p1_neg": p1 & neg, "p1_pos": p1 & pos, "p2_pos": p2 & pos, "p2_neg": p2 & neg,
           "routed": p1 | p2, "elig_pos": pos & ~p1}
    if reference is not None:
        f = np.asarray(reference, bool)
        out.update({"unique_tp": p2 & pos & ~f, "unique_fp": p2 & neg & ~f,
                    "ref_fp": f & neg, "ref_tp": f & pos,
                    "p1_ref_fp": p1 & f & neg, "p1_ref_tp": p1 & f & pos})
    return out


#: Bootstrap metric -> (numerator mask, denominator mask or None for a count).
BOOTSTRAP_METRICS: Mapping[str, Tuple[str, Optional[str]]] = MappingProxyType({
    "pass1_rate": ("p1", "rows"),
    "pass2_rate": ("p2", "rows"),
    "coverage": ("routed", "rows"),
    "abstain_rate": ("ab", "rows"),
    "pass1_npv": ("p1_neg", "p1"),
    "pass1_leakage_share": ("p1_pos", "pos"),
    "pass2_precision": ("p2_pos", "p2"),
    "pass1_negative_recall": ("p1_neg", "neg"),
    "pass2_recall": ("p2_pos", "pos"),
    "pass2_recall_eligible": ("p2_pos", "elig_pos"),
    "pass2_negative_routing_rate": ("p2_neg", "neg"),
    "pass2_rows": ("p2", None),
    "unique_tp": ("unique_tp", None),
    "unique_fp": ("unique_fp", None),
    "pass1_fp_suppression": ("p1_ref_fp", "ref_fp"),
    "pass1_tp_cost": ("p1_ref_tp", "ref_tp"),
})
_REFERENCE_METRICS = ("unique_tp", "unique_fp", "pass1_fp_suppression", "pass1_tp_cost")


def bootstrap_settings(protocol: EvaluationProtocol) -> Dict[str, Any]:
    u = protocol["uncertainty"]
    return {"replicates": int(u["replicates"]), "seed": int(u["seed"]), "level": float(u["level"])}


def stratified_paired_bootstrap(paired: PairedRouting, metrics: Sequence[str], *,
                                replicates: int, seed: int, level: float,
                                reference: Optional[ReferenceFlags] = None) -> pd.DataFrame:
    """
    The protocol's uncertainty procedure. Each replicate resamples rows with
    replacement inside every (outer fold, label) stratum, keeping stratum sizes,
    and scores BOTH routings on that same resample. Fitted routers and the
    ``reference`` flags (one bool per row, for the contribution metrics) are
    held fixed (``ReferenceFlags``, bound to the pairing's rows). Point values
    are the exact counts; intervals are percentile
    intervals (numpy ``linear`` quantiles). A replicate with an empty
    denominator is counted, and any such replicate leaves that interval
    undefined rather than computed over a biased subset.
    """
    if not isinstance(paired, PairedRouting):
        raise ProtocolError("expected a PairedRouting")
    unknown = [m for m in metrics if m not in BOOTSTRAP_METRICS]
    if unknown:
        raise ProtocolError(f"unknown bootstrap metrics {unknown}; known {list(BOOTSTRAP_METRICS)}")
    if any(m in _REFERENCE_METRICS for m in metrics) and reference is None:
        raise ProtocolError("contribution metrics need the frozen reference flags")
    if reference is not None and not isinstance(reference, ReferenceFlags):
        raise ProtocolError("reference must be ReferenceFlags (row-indexed), not a bare array")
    ref_flags = None if reference is None else reference.aligned_to(paired)
    if int(replicates) < 1 or not (0 < float(level) < 1):
        raise ProtocolError("replicates must be >= 1 and level in (0, 1)")
    a_name, b_name = paired.names
    mask_sets = {name: _masks(np.asarray(paired.states[name]), paired.y, ref_flags)
                 for name in (a_name, b_name)}
    keys = sorted({k for m in metrics for k in BOOTSTRAP_METRICS[m] if k is not None})
    cols = [(name, k) for name in (a_name, b_name) for k in keys]
    M = np.vstack([mask_sets[name][k].astype(np.float64) for name, k in cols])
    pos_of = {c: i for i, c in enumerate(cols)}

    strata = sorted({(int(f), int(v)) for f, v in zip(paired.fold_id, paired.y)})
    members = [np.flatnonzero((paired.fold_id == f) & (paired.y == v)) for f, v in strata]
    rng = np.random.default_rng(int(seed))
    sums = np.empty((int(replicates), len(cols)))
    w = np.zeros(paired.n)
    for r in range(int(replicates)):
        for rows in members:
            w[rows] = np.bincount(rng.integers(0, len(rows), len(rows)), minlength=len(rows))
        sums[r] = M @ w

    def values(source, name, metric):
        num, den = BOOTSTRAP_METRICS[metric]
        n = source[..., pos_of[(name, num)]]
        if den is None:
            return n, np.ones_like(n, dtype=bool)
        d = source[..., pos_of[(name, den)]]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan), d > 0

    exact = M.sum(axis=1)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = []
    for m in metrics:
        a_pt, a_ok = values(exact, a_name, m)
        b_pt, b_ok = values(exact, b_name, m)
        a_bs, a_def = values(sums, a_name, m)
        b_bs, b_def = values(sums, b_name, m)
        both = a_def & b_def
        undefined = int((~both).sum())
        rec = {"metric": m, "kind": "count" if BOOTSTRAP_METRICS[m][1] is None else "ratio",
               f"{a_name}": float(a_pt) if a_ok else None,
               f"{b_name}": float(b_pt) if b_ok else None,
               "difference": float(a_pt - b_pt) if (a_ok and b_ok) else None,
               "undefined_replicates": undefined, "replicates": int(replicates),
               "level": float(level)}
        for label, arr, ok in ((a_name, a_bs, a_def), (b_name, b_bs, b_def),
                               ("difference", a_bs - b_bs, both)):
            if bool(ok.all()):
                rec[f"{label}_low"] = float(np.quantile(arr, lo_q, method="linear"))
                rec[f"{label}_high"] = float(np.quantile(arr, hi_q, method="linear"))
            else:
                rec[f"{label}_low"] = rec[f"{label}_high"] = None
        rec["interval_status"] = "defined" if undefined == 0 else "undefined_empty_denominator"
        out.append(rec)
    return pd.DataFrame(out)


@dataclass(frozen=True)
class BootstrapRun:
    """
    One registered run of the protocol's bootstrap: its rows (``table``), the
    exact per-fold point values and a provenance naming the procedure identity,
    the protocol, the settings and the paired data digest. ``digest`` covers
    all of it; ``verify`` detects any later edit.
    """

    rows: Tuple[Mapping[str, Any], ...]
    fold_points: Mapping[str, Mapping[int, Mapping[str, Optional[float]]]]
    provenance: Mapping[str, Any]
    digest: str

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([dict(r) for r in self.rows])

    def row(self, metric: str) -> Mapping[str, Any]:
        for r in self.rows:
            if r["metric"] == metric:
                return r
        raise KeyError(metric)

    @staticmethod
    def compute_digest(rows, fold_points, provenance) -> str:
        blob = json.dumps({"rows": _thaw(rows), "fold_points": _thaw(fold_points),
                           "provenance": _thaw(provenance)},
                          sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        return self.compute_digest(self.rows, self.fold_points, self.provenance) == self.digest


def _exact_value(paired, name, metric, ref_flags, keep) -> Optional[float]:
    m = _masks(np.asarray(paired.states[name])[keep], paired.y[keep],
               None if ref_flags is None else ref_flags[keep])
    num, den = BOOTSTRAP_METRICS[metric]
    n = int(m[num].sum())
    if den is None:
        return float(n)
    d = int(m[den].sum())
    return None if d == 0 else n / d


def _registered_settings(protocol: EvaluationProtocol) -> Dict[str, Any]:
    u = protocol["uncertainty"]
    return {"procedure_id": u["procedure_id"], "replicates": int(u["replicates"]),
            "seed": int(u["seed"]), "level": float(u["level"]), "strata": list(u["strata"]),
            "interval": u["interval"], "quantile_method": u["quantile_method"]}


def run_registered_bootstrap(paired: PairedRouting, metrics: Sequence[str],
                             protocol: EvaluationProtocol, *,
                             reference: Optional[ReferenceFlags] = None) -> BootstrapRun:
    """``stratified_paired_bootstrap`` with the protocol's registered settings, recorded."""
    settings = _registered_settings(protocol)
    if settings["procedure_id"] != BOOTSTRAP_PROCEDURE_ID:
        raise ProtocolError(f"protocol registers {settings['procedure_id']!r}; this code "
                            f"implements {BOOTSTRAP_PROCEDURE_ID!r}")
    table = stratified_paired_bootstrap(paired, metrics, replicates=settings["replicates"],
                                        seed=settings["seed"], level=settings["level"],
                                        reference=reference)
    ref_flags = None if reference is None else reference.aligned_to(paired)
    fold_points = {m: {int(k): {name: _exact_value(paired, name, m, ref_flags,
                                                   paired.fold_id == k)
                                for name in paired.names}
                       for k in paired.folds}
                   for m in metrics}
    provenance = {**settings, "protocol_id": protocol.protocol_id,
                  "protocol_version": protocol.version,
                  "protocol_fingerprint": protocol.fingerprint(), "kind": paired.kind,
                  "names": list(paired.names), "rows": paired.n, "metrics": list(metrics),
                  "reference": None if reference is None else reference.name,
                  "reference_fingerprint": None if reference is None else reference.fingerprint,
                  "data_digest": paired.digest()}
    rows = tuple(_freeze({k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                          for k, v in r.items()}) for r in table.to_dict("records"))
    fp = _freeze(fold_points)
    prov = _freeze(provenance)
    return BootstrapRun(rows, fp, prov, BootstrapRun.compute_digest(rows, fp, prov))


def wilson_interval(numerator: int, denominator: int,
                    level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Wilson score interval for one proportion; ``None`` when the denominator is 0."""
    num, den = int(numerator), int(denominator)
    if den < 0 or num < 0 or num > den:
        raise ProtocolError(f"invalid proportion {num}/{den}")
    if den == 0:
        return None
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = num / den
    centre = (p + z * z / (2 * den)) / (1 + z * z / den)
    half = z * math.sqrt(p * (1 - p) / den + z * z / (4 * den * den)) / (1 + z * z / den)
    return max(0.0, centre - half), min(1.0, centre + half)


# ======================================================================
# Fit status, fold ledger, pooling
# ======================================================================
#: Trace arm -> its ``f17_primary`` section.
_ARM_F17 = MappingProxyType({"glass": "glass", "blackbox": "rf"})
#: Trace arm -> (inner selection kind, the F17 setting that fixes how many there are).
#: GLASS: one inner Pass 1 selection per ``pass1_oof_folds`` fold
#: (``GlassRouterPipeline.selection_reports()["inner_pass1"]``). RF: one nested
#: ``t1_j`` solve per Pass 1 OOF fold (``rf_router.oof.eligibility``: one
#: ``solve_pass1`` per distinct ``fold_id`` of the ``n_oof_folds`` OOF run).
_INNER = MappingProxyType({"glass": ("inner_pass1", "pass1_oof_folds"),
                           "blackbox": ("eligibility_t1", "n_oof_folds")})
_KEY_RE = re.compile(r"^(pass1|pass2|inner_pass1|eligibility_t1)(?:\[(\d+)\])?$")


@dataclass(frozen=True)
class SelectionRecord:
    """One fit-time selection of an outer fold: its key and its ``FitOutcome``."""

    key: str
    outcome: Any


def required_selection_keys(arm: str, protocol: EvaluationProtocol) -> Tuple[str, ...]:
    """Every selection one outer fold of ``arm`` must record under the F17 primary settings."""
    if arm not in _ARM_F17:
        raise ProtocolError(f"arm must be one of {sorted(_ARM_F17)}, got {arm!r}")
    kind, setting = _INNER[arm]
    n = int(protocol["f17_primary"][_ARM_F17[arm]][setting])
    return ("pass1", "pass2") + tuple(f"{kind}[{j}]" for j in range(n))


def _key_scope(key: str) -> str:
    return "pass2" if key == "pass2" else "pass1"


def _report_slots(selection_reports: Mapping[str, Any]) -> Dict[str, Any]:
    """
    ``{key: report or None}`` for every slot a GLASS ``selection_reports()``
    holds: ``pass1``, ``pass2`` and one ``inner_pass1[j]`` per list entry,
    ``None`` entries included. A ``None`` slot is a selection that left no
    report; it is never dropped, so the inner collection's length is checked
    exactly against the required selections (an extra trailing ``None`` is an
    unexpected slot, a ``None`` in a required slot is a missing report).
    """
    if not isinstance(selection_reports, Mapping) or set(selection_reports) != {
            "pass1", "pass2", "inner_pass1"}:
        raise ProtocolError("selection_reports must be GlassRouterPipeline.selection_reports(): "
                            "{'pass1', 'pass2', 'inner_pass1'}")
    inner = selection_reports["inner_pass1"]
    if inner is not None and not isinstance(inner, (list, tuple)):
        raise ProtocolError("selection_reports['inner_pass1'] must be a list of reports")
    out = {k: selection_reports[k] for k in ("pass1", "pass2")}
    out.update({f"inner_pass1[{j}]": r for j, r in enumerate(inner or ())})
    return out


#: GLASS ``selection_report_`` source -> the fit outcomes it can classify to
#: (``glass_pipeline.glass_router.contract.glass_fit_outcome``; mirrored here so
#: this module stays arm-neutral). An ILP report classifies to optimal only
#: with ``lexicographic_guarantee == "proven"``.
_REPORT_OUTCOMES = MappingProxyType({
    "ilp": (OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN),
    "greedy_fallback": (OUTCOME_FEASIBLE_GREEDY,),
    "infeasible": (OUTCOME_INFEASIBLE, OUTCOME_NO_SOLUTION),
    "empty": (OUTCOME_INSUFFICIENT,),
})


def _report_problem(key: str, outcome: Any, report: Any) -> Optional[str]:
    """Why a selection record is not the classification of its GLASS report (or None)."""
    if not isinstance(report, Mapping):
        return f"{key}: its selection report is not a mapping"
    source = report.get("source")
    kind = "empty" if isinstance(source, str) and source.startswith("empty") else source
    if kind not in _REPORT_OUTCOMES:
        return f"{key}: selection report source {source!r} is not a recognised fit outcome"
    if not isinstance(outcome, FitOutcome):
        return None                                     # reported as unsupported already
    allowed = _REPORT_OUTCOMES[kind]
    if kind == "ilp":
        allowed = ((OUTCOME_FEASIBLE_OPTIMAL,) if report.get("lexicographic_guarantee") == "proven"
                   else (OUTCOME_FEASIBLE_UNPROVEN,))
    if outcome.outcome not in allowed:
        return (f"{key}: the record's outcome {outcome.outcome!r} is not what its selection "
                f"report shows (source {source!r})")
    solver = outcome.solver if isinstance(outcome.solver, Mapping) else {}
    if tuple(report.get("selected_rule_ids") or ()) != tuple(solver.get("selected_rule_ids")
                                                             or ()):
        return f"{key}: the selected rule ids differ between the record and its selection report"
    return None


def glass_reports_required(protocol: EvaluationProtocol) -> bool:
    """Whether the protocol makes GLASS selection reports part of a fold's fit evidence (D-C11)."""
    return protocol["fit_status"].get("glass_selection_reports",
                                      GLASS_REPORTS_OPTIONAL) == GLASS_REPORTS_REQUIRED


@dataclass(frozen=True)
class CompletenessReport:
    arm: str
    required: Tuple[str, ...]
    present: Tuple[str, ...]
    missing: Tuple[str, ...]
    duplicated: Tuple[str, ...]
    unexpected: Tuple[str, ...]
    unsupported: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (self.missing or self.duplicated or self.unexpected or self.unsupported)

    def problems(self) -> List[str]:
        return ([f"missing {k}" for k in self.missing]
                + [f"duplicated {k}" for k in self.duplicated]
                + [f"unexpected {k}" for k in self.unexpected] + list(self.unsupported))


def check_selection_completeness(records: Sequence[SelectionRecord], *, arm: str,
                                 protocol: EvaluationProtocol,
                                 selection_reports: Optional[Mapping[str, Any]] = None
                                 ) -> CompletenessReport:
    """
    Are this outer fold's selection records exactly the required ones?

    Missing: a required key without a record. Duplicated: a key recorded more
    than once. Unexpected: a key outside the required set (an unknown kind, an
    inner index out of range, the other arm's inner kind). Unsupported: a
    record that is not a ``SelectionRecord`` holding a ``FitOutcome`` whose
    scope matches its key.

    GLASS ``selection_reports`` must correspond exactly to the required
    selections, slot by slot (``None`` entries count as slots): a required
    selection without a report, or with a ``None`` one, is missing; a slot for
    a selection that is not required (``inner_pass1[5]``, a trailing ``None``
    included) is unexpected;
    one report object standing for two selections is duplicated; a record
    that is not its report's classification (outcome, ILP guarantee, selected
    rule ids) is unsupported. When the protocol requires the reports
    (``fit_status.glass_selection_reports``, D-C11), a GLASS fold without
    them is unsupported, so it can never be accepted.
    """
    required = required_selection_keys(arm, protocol)
    seen: Dict[str, int] = {}
    unsupported: List[str] = []
    for i, r in enumerate(records):
        if not isinstance(r, SelectionRecord):
            unsupported.append(f"record {i}: not a SelectionRecord ({type(r).__name__})")
            continue
        if not isinstance(r.outcome, FitOutcome):
            unsupported.append(f"{r.key}: outcome is not a FitOutcome")
        elif _KEY_RE.match(str(r.key)) and r.outcome.scope != _key_scope(r.key):
            unsupported.append(f"{r.key}: outcome scope {r.outcome.scope!r} != "
                               f"{_key_scope(r.key)!r}")
        seen[str(r.key)] = seen.get(str(r.key), 0) + 1
    missing_reports: List[str] = []
    report_unexpected: List[str] = []
    report_duplicated: List[str] = []
    if selection_reports is not None:
        if arm != "glass":
            raise ProtocolError("selection_reports() is the GLASS pipeline's record")
        slots = _report_slots(selection_reports)
        entries = {k: r for k, r in slots.items() if r is not None}
        unsupported += [f"{k}: no selection report behind the record" for k in seen
                        if k not in entries and k in required]
        missing_reports = [k for k in required if k not in entries]
        report_unexpected = [f"report {k}" + ("" if r is not None else " (null)")
                             for k, r in slots.items() if k not in required]
        first: Dict[int, str] = {}
        for k, report in entries.items():
            if id(report) in first:
                report_duplicated.append(f"report {k} (the same report as {first[id(report)]})")
            else:
                first[id(report)] = k
        for r in records:
            if isinstance(r, SelectionRecord) and str(r.key) in entries and r.key in required:
                problem = _report_problem(str(r.key), r.outcome, entries[str(r.key)])
                if problem:
                    unsupported.append(problem)
    elif arm == "glass" and glass_reports_required(protocol):
        unsupported.append("no GLASS selection reports: this protocol requires them as a "
                           "fold's fit evidence (fit_status.glass_selection_reports, D-C11)")
    present = tuple(k for k in required if k in seen)
    missing = tuple(k for k in required if k not in seen or k in missing_reports)
    duplicated = tuple(sorted(k for k, c in seen.items() if c > 1)) + tuple(report_duplicated)
    unexpected = tuple(sorted(k for k in seen if k not in required)) + tuple(report_unexpected)
    return CompletenessReport(arm, required, present, missing, duplicated, unexpected,
                              tuple(unsupported))


def _outcome_status(o: FitOutcome, review: Optional[Mapping[str, str]]) -> str:
    if o.outcome in _STOP_STATUS:
        return _STOP_STATUS[o.outcome]
    if not (isinstance(o.report, ObligationReport) and o.report.stage == "fit"
            and o.report.all_satisfied):
        return FIT_TIME_INVALID
    if o.outcome == OUTCOME_FEASIBLE_GREEDY:
        rec = review or {}
        reviewed = all(isinstance(rec.get(k), str) and rec.get(k).strip()
                       for k in ("reviewer", "rationale"))
        return FIT_FALLBACK_ACCEPTED if reviewed else FIT_FALLBACK_UNREVIEWED
    if o.outcome == OUTCOME_FEASIBLE_UNPROVEN:
        return FIT_UNPROVEN_DOCUMENTED if o.documentation_complete else FIT_UNPROVEN_UNDOCUMENTED
    if o.outcome == OUTCOME_FEASIBLE_OPTIMAL:
        return FIT_OPTIMAL
    return FIT_TIME_INVALID  # pragma: no cover - FIT_OUTCOMES is closed


@dataclass(frozen=True)
class FoldFit:
    """
    One arm's outer fold: its status, why, the status of each selection, and
    the evidence it was classified from (the selection records, the
    greedy-fallback reviews and, for GLASS, the selection reports).

    Build it with ``classify_fold_fit``. An accepted status is refused at
    construction unless the completeness report is complete and the records
    are exactly the required selections, each with its status;
    ``FoldLedger.build`` then re-classifies the evidence and refuses a
    ``FoldFit`` whose status does not reproduce.
    """

    arm: str
    status: str
    completeness: CompletenessReport
    by_key: Mapping[str, str]
    records: Tuple[Any, ...] = ()
    reviews: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    selection_reports: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        if self.status not in FIT_STATUSES:
            raise ProtocolError(f"{self.arm}: unknown fit status {self.status!r}")
        if self.status in ACCEPTED_FIT_STATUSES:
            problems = _accepted_fit_problems(self)
            if problems:
                raise ProtocolError(f"{self.arm}: the accepted fit status {self.status!r} needs "
                                    "complete selection records (classify_fold_fit): "
                                    + "; ".join(problems))


def _accepted_fit_problems(fit: "FoldFit") -> List[str]:
    """Why an accepted ``FoldFit`` is not backed by its complete selection records."""
    comp = fit.completeness
    if not isinstance(comp, CompletenessReport):
        return ["no completeness report"]
    problems = [] if comp.arm == fit.arm else [f"completeness report is for arm {comp.arm!r}"]
    if not comp.complete:
        problems += comp.problems()
    if not all(isinstance(r, SelectionRecord) for r in fit.records):
        return problems + ["a record is not a SelectionRecord"]
    keys = sorted(str(r.key) for r in fit.records)
    if keys != sorted(comp.required):
        problems.append("the records are not exactly the required selections "
                        f"({len(keys)} recorded, {len(comp.required)} required)")
    by_key = dict(fit.by_key) if isinstance(fit.by_key, Mapping) else {}
    if sorted(by_key) != sorted(comp.required) or not all(
            v in FIT_STATUSES for v in by_key.values()):
        problems.append("a required selection has no fit status")
    elif min(by_key.values(), key=FIT_STATUSES.index) != fit.status:
        problems.append("the status is not the most severe selection status")
    return problems


def classify_fold_fit(records: Sequence[SelectionRecord], *, arm: str,
                      protocol: EvaluationProtocol,
                      reviews: Optional[Mapping[str, Mapping[str, str]]] = None,
                      selection_reports: Optional[Mapping[str, Any]] = None) -> FoldFit:
    """
    The status of one arm's outer fold from every selection it must record.

    The records must be complete (``check_selection_completeness``): an
    incomplete, duplicated, unexpected or unsupported set is
    ``fit_time_invalid``, so an accepted status can never come from a subset
    of the required fits. Otherwise the most severe selection status wins:
    stop outcomes; a feasible claim without a satisfied fit-time report
    (``fit_time_invalid``); a greedy fallback, accepted only with a review in
    ``reviews`` (``{key: {"reviewer": ..., "rationale": ...}}``); an unproven
    incumbent, accepted only when documented (status, gap, reproduction).
    """
    comp = check_selection_completeness(records, arm=arm, protocol=protocol,
                                        selection_reports=selection_reports)
    evidence = dict(records=tuple(records), reviews=MappingProxyType(dict(reviews or {})),
                    selection_reports=selection_reports)
    if not comp.complete:
        return FoldFit(arm, FIT_TIME_INVALID, comp, MappingProxyType({}), **evidence)
    by_key = {r.key: _outcome_status(r.outcome, (reviews or {}).get(r.key)) for r in records}
    status = min(by_key.values(), key=FIT_STATUSES.index)
    return FoldFit(arm, status, comp, MappingProxyType(by_key), **evidence)


@dataclass(frozen=True)
class FoldLedger:
    """
    Attempted, completed and accepted outer folds per arm.

    ``fits`` holds each fold's ``FoldFit`` (``None`` for a fold recorded by a
    status string). A fold is accepted only through a ``FoldFit`` with that
    accepted status: a status string can record a fold as not accepted, never
    as accepted, however the ledger is constructed.
    """

    attempted: Tuple[int, ...]
    status: Mapping[str, Mapping[int, str]]
    fits: Mapping[str, Mapping[int, Optional[FoldFit]]] = field(
        default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        for arm, by_fold in self.status.items():
            for k, st in by_fold.items():
                if st not in ACCEPTED_FIT_STATUSES:
                    continue
                fit = self.fits.get(arm, {}).get(k) if isinstance(self.fits, Mapping) else None
                if not (isinstance(fit, FoldFit) and fit.arm == arm and fit.status == st):
                    raise ProtocolError(f"{arm} fold {k}: the accepted status {st!r} has no "
                                        "FoldFit with complete selection records behind it")

    @classmethod
    def build(cls, status: Mapping[str, Mapping[int, Any]],
              protocol: EvaluationProtocol) -> "FoldLedger":
        """
        ``status``: ``{arm: {fold: FoldFit}}``. Each ``FoldFit`` is re-classified
        from its own evidence under ``protocol`` (``classify_fold_fit``) and must
        reproduce exactly. A fit-status string is accepted only for a status
        that is not accepted (for example a fold whose fit stopped before it
        produced records); an accepted status needs the fold's ``FoldFit``.
        """
        attempted = tuple(int(k) for k in protocol["folds"]["attempted"])
        out, fits = {}, {}
        for arm, by_fold in status.items():
            extra = sorted(set(int(k) for k in by_fold) - set(attempted))
            if extra:
                raise ProtocolError(f"{arm}: folds {extra} are not attempted folds")
            clean, evidence = {}, {}
            for k, st in by_fold.items():
                fit = None
                if isinstance(st, FoldFit):
                    if st.arm != arm:
                        raise ProtocolError(f"{arm} fold {k}: FoldFit is for arm {st.arm!r}")
                    _require_reproduced(st, arm, int(k), protocol)
                    fit, st = st, st.status
                elif st not in FIT_STATUSES:
                    raise ProtocolError(f"{arm} fold {k}: unknown fit status {st!r}")
                elif st in ACCEPTED_FIT_STATUSES:
                    raise ProtocolError(
                        f"{arm} fold {k}: {st!r} is an accepted fit status; acceptance needs the "
                        "fold's FoldFit from classify_fold_fit, with complete selection records, "
                        "not a status string")
                clean[int(k)], evidence[int(k)] = st, fit
            out[str(arm)] = MappingProxyType(clean)
            fits[str(arm)] = MappingProxyType(evidence)
        return cls(attempted, MappingProxyType(out), MappingProxyType(fits))

    def accepted_folds(self, arm: str) -> Tuple[int, ...]:
        s = self.status.get(arm, {})
        return tuple(k for k in self.attempted if s.get(k) in ACCEPTED_FIT_STATUSES)

    def summary(self) -> pd.DataFrame:
        rows = []
        for arm, s in self.status.items():
            reported = [k for k in self.attempted if k in s]
            completed = [k for k in reported if s[k] not in
                         (FIT_TIME_INVALID, FIT_INFEASIBLE, FIT_NO_SOLUTION, FIT_INSUFFICIENT)]
            fits = self.fits.get(arm, {}) if isinstance(self.fits, Mapping) else {}
            rows.append({"arm": arm, "attempted": len(self.attempted),
                         "reported": len(reported), "fit_completed": len(completed),
                         "accepted": len(self.accepted_folds(arm)),
                         "missing": [k for k in self.attempted if k not in s],
                         "status_only": [k for k in reported if fits.get(k) is None],
                         "not_accepted": {k: s[k] for k in reported
                                          if s[k] not in ACCEPTED_FIT_STATUSES}})
        return pd.DataFrame(rows)

    def complete(self, arms: Sequence[str]) -> bool:
        return all(len(self.accepted_folds(a)) == len(self.attempted) for a in arms)

    def require_complete(self, arms: Sequence[str]) -> "FoldLedger":
        problems = []
        for a in arms:
            s = self.status.get(a, {})
            for k in self.attempted:
                if k not in s:
                    problems.append(f"{a} fold {k}: no fit record")
                elif s[k] not in ACCEPTED_FIT_STATUSES:
                    problems.append(f"{a} fold {k}: {s[k]}")
        if problems:
            raise ProtocolError("the primary comparison needs every attempted fold accepted in "
                                "both arms (stop and investigate, M4):\n  - "
                                + "\n  - ".join(problems))
        return self


def _require_reproduced(fit: FoldFit, arm: str, fold: int, protocol: EvaluationProtocol) -> None:
    """A ``FoldFit`` must re-classify to itself from its evidence under ``protocol``."""
    try:
        replay = classify_fold_fit(fit.records, arm=arm, protocol=protocol, reviews=fit.reviews,
                                   selection_reports=fit.selection_reports)
    except (ProtocolError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"{arm} fold {fold}: the FoldFit's evidence cannot be "
                            f"re-classified: {exc}") from exc
    same = (replay.status == fit.status and dict(replay.by_key) == dict(fit.by_key)
            and replay.completeness == fit.completeness)
    if not same:
        raise ProtocolError(f"{arm} fold {fold}: the FoldFit does not reproduce from its "
                            f"selection records under this protocol ({fit.status!r} recorded, "
                            f"{replay.status!r} re-classified)")


def pooled_observation(paired: PairedRouting, name: str, ledger: FoldLedger,
                       contract: SharedOperatingContractV2, *,
                       allow_partial: bool = False) -> Dict[str, Any]:
    """
    Per-fold and pooled contract observations for one routing. Every attempted
    fold is always reported. The pool covers all attempted folds, or, only
    with ``allow_partial``, the accepted folds under an explicit label.
    """
    accepted = ledger.accepted_folds(name)
    if set(paired.folds) != set(ledger.attempted):
        raise ProtocolError(f"routing folds {paired.folds} != attempted folds {ledger.attempted}")
    per_fold, pooled_all = observe_folds(paired.fold_counts(name), contract)
    complete = len(accepted) == len(ledger.attempted)
    if complete:
        pooled, label, used = pooled_all, "all attempted folds", tuple(ledger.attempted)
    elif allow_partial:
        if not accepted:
            raise ProtocolError(f"{name}: no accepted fold to pool")
        pooled = observe_routing(pool_counts(paired.counts(name, k) for k in accepted), contract)
        label = f"accepted folds only ({len(accepted)} of {len(ledger.attempted)})"
        used = accepted
    else:
        raise ProtocolError(f"{name}: {len(accepted)} of {len(ledger.attempted)} folds accepted; "
                            "a pooled value over fewer folds needs allow_partial and is never "
                            "the primary comparison")
    return {"routing": name, "per_fold": per_fold, "pooled": pooled, "pool_label": label,
            "pooled_folds": used, "primary_eligible": complete,
            "fold_status": {k: ledger.status.get(name, {}).get(k) for k in ledger.attempted}}


# ======================================================================
# Shortfalls
# ======================================================================
def materiality_tolerance(limit, n: int, multiplier) -> Optional[float]:
    """``multiplier * sqrt(L (1 - L) / n)``; ``None`` for an empty denominator."""
    if n is None or int(n) <= 0:
        return None
    L = float(Fraction(str(limit)))
    return float(Fraction(str(multiplier))) * math.sqrt(L * (1 - L) / int(n))


@dataclass(frozen=True)
class ShortfallObservation:
    obligation: str
    metric: str
    numerator: int
    denominator: Optional[int]
    limit: str
    observed_status: str
    shortfall: Optional[Fraction]
    fit_status: str
    classification: str
    low_support: bool
    tolerance: Optional[float]
    material: Optional[bool]
    materiality_rule_status: str

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["shortfall"] = None if self.shortfall is None else str(self.shortfall)
        return d


def classify_observation(result: ObligationResult, *, fit_status: str,
                         protocol: EvaluationProtocol) -> ShortfallObservation:
    """
    Classify one outer-OOF obligation observation without hiding it.

    ``observed_status`` is always the exact Step 4A status. ``classification``
    names what the observation is evidence of: a fold whose fit was not
    accepted is reported under its fit status, not as an outer-OOF result.
    ``material`` applies the proposed 2-SE rule to violations only, and
    ``low_support`` marks denominators below the reporting threshold. Neither
    turns a violation into compliance.
    """
    if not isinstance(result, ObligationResult):
        raise ProtocolError("expected an ObligationResult from observe_routing")
    if fit_status not in FIT_STATUSES:
        raise ProtocolError(f"unknown fit status {fit_status!r}")
    sf = protocol["shortfall"]
    den = result.denominator
    low = den is not None and den < int(sf["low_support_threshold"])
    tol = material = None
    if result.status == VIOLATED and den is not None:
        tol = materiality_tolerance(result.limit, den, sf["materiality"]["multiplier"])
        material = tol is not None and float(result.shortfall) > tol
    if fit_status not in ACCEPTED_FIT_STATUSES:
        cls = fit_status
    elif result.status == UNDEFINED_STATUS:
        cls = "undefined_evaluation"
    elif result.status == VIOLATED:
        cls = "valid_fit_outer_oof_shortfall"
    elif result.status == SATISFIED:
        cls = "valid_fit_observed_satisfied"
    else:  # pragma: no cover
        raise ProtocolError(f"unknown observed status {result.status!r}")
    return ShortfallObservation(result.obligation, result.metric, result.numerator, den,
                                result.limit, result.status, result.shortfall, fit_status, cls,
                                bool(low), tol, material, sf["materiality"]["status"])


# ======================================================================
# F31 R2 trigger
# ======================================================================
def f31_strip_review(fold_records: Mapping[int, Tuple[int, int]],
                     protocol: EvaluationProtocol) -> pd.DataFrame:
    """
    Per RF outer fold router: strip A rows inside its flagged Pass 2 band over
    that band's rows, against the 1/20 trigger. ``fold_records``:
    ``{fold: (strip_a_rows_in_band, band_rows)}``.
    """
    trigger = Fraction(str(protocol["f31"]["trigger"]))
    rows = []
    for k in sorted(fold_records):
        strip, band = (int(v) for v in fold_records[k])
        if strip < 0 or band < 0 or strip > band:
            raise ProtocolError(f"fold {k}: invalid strip / band counts {strip} / {band}")
        if band == 0:
            rows.append({"fold": k, "strip_a_in_band": strip, "band_rows": band, "share": None,
                         "status": "undefined_empty_band", "review_required": True})
            continue
        share = Fraction(strip, band)
        hit = share >= trigger
        rows.append({"fold": k, "strip_a_in_band": strip, "band_rows": band,
                     "share": float(share), "status": "triggered" if hit else "below_trigger",
                     "review_required": bool(hit)})
    return pd.DataFrame(rows)


# ======================================================================
# Artifact acceptance (structural, kept apart from performance)
# ======================================================================
GATE_PASS, GATE_FAIL, GATE_NOT_RUN = "pass", "fail", "not_run"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    evidence: str

    def __post_init__(self) -> None:
        if self.status not in (GATE_PASS, GATE_FAIL, GATE_NOT_RUN):
            raise ProtocolError(f"gate {self.gate}: status must be pass / fail / not_run")
        if self.status == GATE_PASS and not str(self.evidence).strip():
            raise ProtocolError(f"gate {self.gate}: a pass needs recorded evidence")


@dataclass(frozen=True)
class StructuralAcceptance:
    accepted: bool
    passed: Tuple[str, ...]
    failed: Tuple[str, ...]
    not_run: Tuple[str, ...]
    missing: Tuple[str, ...]
    note: str = ("Structural validity only: it is not evidence of better predictive "
                 "performance, and a weak result does not make an artifact invalid.")
    #: True only under the pinned frozen protocol (``is_pinned_frozen``).
    official: bool = False


def structural_acceptance(results: Iterable[GateResult], protocol: EvaluationProtocol, *,
                          allow_working: bool = False) -> StructuralAcceptance:
    """Every structural gate must pass with evidence. Takes no performance input."""
    _gate(protocol, allow_working)
    required = tuple(protocol["acceptance"]["structural"])
    got: Dict[str, GateResult] = {}
    for r in results:
        if not isinstance(r, GateResult):
            raise ProtocolError(f"expected GateResult, got {type(r).__name__}")
        if r.gate not in required:
            raise ProtocolError(f"{r.gate} is not a structural gate {list(required)}")
        if r.gate in got:
            raise ProtocolError(f"gate {r.gate} reported twice")
        got[r.gate] = r
    passed = tuple(g for g in required if g in got and got[g].status == GATE_PASS)
    failed = tuple(g for g in required if g in got and got[g].status == GATE_FAIL)
    not_run = tuple(g for g in required if g in got and got[g].status == GATE_NOT_RUN)
    missing = tuple(g for g in required if g not in got)
    return StructuralAcceptance(len(passed) == len(required), passed, failed, not_run, missing,
                                official=is_pinned_frozen(protocol))


def comparative_report_gaps(sections: Iterable[str], protocol: EvaluationProtocol) -> List[str]:
    """Required comparative-report sections that are missing (reporting completeness only)."""
    have = set(sections)
    return [s for s in protocol["acceptance"]["comparative_sections"] if s not in have]


# ======================================================================
# Evidence tiers
# ======================================================================
def evidence_tier(source_id: str, protocol: EvaluationProtocol) -> str:
    """The registered tier of an evidence source; unknown sources are refused."""
    sources = protocol["evidence_tiers"]["sources"]
    if source_id not in sources:
        raise ProtocolError(f"unregistered evidence source {source_id!r}; register it in the "
                            "protocol before it is used")
    return sources[source_id]


# ======================================================================
# Complementarity-ablation adoption
# ======================================================================
#: The registered bootstrap metrics adoption is decided on (ablation - primary).
ADOPTION_METRICS = ("unique_tp", "pass2_rows", "coverage", "pass1_leakage_share",
                    "pass2_negative_routing_rate", "pass1_negative_recall")


@dataclass(frozen=True)
class AdoptionEvidence:
    """
    What the PR 38 ablation report supplies beside the paired data.

    ``uncertainty`` is the ``BootstrapRun`` the report presents. Every interval
    and count the criteria use is recomputed by ``evaluate_adoption`` from the
    paired ablation routing with the registered procedure; the reported run
    must be reproduced exactly (C6). The remaining fields are fit-time and
    review facts that the outer-OOF data cannot show.
    """

    sources: Tuple[str, ...]
    uncertainty: Optional[BootstrapRun] = None
    ablation_changes_pass1: Optional[bool] = None
    pass1_rules_identical_every_fold: Optional[bool] = None
    fit_compliance_every_fold: Optional[bool] = None
    new_material_violation_folds: Optional[int] = None
    worse_fit_status_folds: Optional[int] = None
    runtime_ratio: Optional[float] = None
    complexity_documented: Optional[bool] = None
    rf_symmetry: Optional[str] = None


@dataclass(frozen=True)
class AdoptionDecision:
    """
    ``official`` is True only under the pinned frozen protocol
    (``is_pinned_frozen``: loaded by ``load_frozen_protocol``) with a complete
    upstream identity (every pinned value checked). A frozen record built any
    other way exercises the gates but never decides officially. ``provenance`` names the protocol,
    the verified reference and the upstream identity status.
    """

    decision: str
    criteria: Mapping[str, str]
    reasons: Mapping[str, str]
    evidence_label: str
    note: str
    official: bool = False
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


#: Provenance entries every registered run records (``run_registered_bootstrap``).
_RUN_PROVENANCE_KEYS = ("procedure_id", "replicates", "seed", "level", "strata", "interval",
                        "quantile_method", "protocol_fingerprint", "kind", "names", "metrics",
                        "reference_fingerprint", "data_digest")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _run_problems(run: BootstrapRun) -> List[str]:
    """Why a reported ``BootstrapRun`` is malformed or incomplete (empty when well formed)."""
    problems = []
    if not isinstance(run.digest, str) or not _SHA256_HEX.match(run.digest):
        problems.append("its digest is not a SHA-256 hex digest")
    prov = run.provenance
    if not isinstance(prov, Mapping):
        problems.append("its provenance is not a mapping")
    else:
        missing = [k for k in _RUN_PROVENANCE_KEYS if k not in prov]
        if missing:
            problems.append(f"its provenance lacks {missing}")
    rows = run.rows
    names: List[str] = []
    if not isinstance(rows, (tuple, list)) or not all(
            isinstance(r, Mapping) and isinstance(r.get("metric"), str) for r in rows):
        problems.append("its rows are not one mapping per metric, each naming its metric")
    else:
        names = [r["metric"] for r in rows]
        dup = sorted({m for m in names if names.count(m) > 1})
        if dup:
            problems.append(f"it has duplicated rows for {dup}")
        claimed = prov.get("metrics") if isinstance(prov, Mapping) else None
        if isinstance(claimed, (tuple, list)):
            without = [m for m in claimed if m not in names]
            if without:
                problems.append(f"its provenance lists metrics without rows: {without}")
    fp = run.fold_points
    if not isinstance(fp, Mapping):
        problems.append("its per-fold points are not a mapping")
    else:
        bad = [m for m in names if not (isinstance(fp.get(m), Mapping) and all(
            isinstance(v, Mapping) for v in fp.get(m).values()))]
        if bad:
            problems.append(f"its per-fold points are missing or malformed for {bad}")
    return problems


def _uncertainty_verdict(reported: Any, replay: BootstrapRun) -> Tuple[str, str]:
    """C6: the reported run must be the registered procedure's output, reproduced exactly."""
    if reported is None:
        return "not_evaluable", "no registered bootstrap run was reported"
    if not isinstance(reported, BootstrapRun):
        return "fail", (f"the reported uncertainty is a {type(reported).__name__}, "
                        "not a BootstrapRun")
    problems = _run_problems(reported)
    if problems:
        return "fail", "the reported run is malformed or incomplete: " + "; ".join(problems)
    try:
        return _compare_runs(reported, replay)
    except Exception as exc:  # noqa: BLE001 - a run that cannot be checked fails C6, explicitly
        return "fail", (f"the reported run could not be checked against the registered run "
                        f"({type(exc).__name__}: {exc})")


def _compare_runs(reported: BootstrapRun, replay: BootstrapRun) -> Tuple[str, str]:
    if not reported.verify():
        return "fail", "the reported run was edited after it was produced (digest mismatch)"
    rp, pp = reported.provenance, replay.provenance
    if rp.get("procedure_id") != BOOTSTRAP_PROCEDURE_ID:
        return "fail", f"unsupported procedure {rp.get('procedure_id')!r}"
    for key in ("replicates", "seed", "level", "strata", "interval", "quantile_method"):
        if _thaw(rp.get(key)) != _thaw(pp.get(key)):
            return "fail", (f"{key} {_thaw(rp.get(key))!r} is not the registered "
                            f"{_thaw(pp.get(key))!r}")
    if rp.get("protocol_fingerprint") != pp.get("protocol_fingerprint"):
        return "fail", "the run was produced under a different protocol"
    if rp.get("kind") != "ablation" or tuple(rp.get("names", ())) != ("ablation", "primary"):
        return "fail", "the run is not an ablation-vs-primary pairing"
    if rp.get("data_digest") != pp.get("data_digest") or rp.get(
            "reference_fingerprint") != pp.get("reference_fingerprint"):
        return "fail", "the run was computed on different paired data or reference flags"
    missing = [m for m in ADOPTION_METRICS if m not in rp.get("metrics", ())]
    if missing:
        return "fail", f"the run lacks the adoption metrics {missing}"
    for m in ADOPTION_METRICS:
        if dict(reported.row(m)) != dict(replay.row(m)) or _thaw(
                reported.fold_points.get(m)) != _thaw(replay.fold_points.get(m)):
            return "fail", f"{m}: the reported values do not reproduce"
        if replay.row(m)["interval_status"] != "defined":
            return "fail", f"{m}: interval undefined in the registered run"
    return "pass", "reproduced exactly with the registered procedure"


def _adoption_values(replay: BootstrapRun, paired: PairedRouting) -> Dict[str, Any]:
    r = {m: replay.row(m) for m in ADOPTION_METRICS}
    folds = replay.fold_points["unique_tp"]
    budget = replay.fold_points["pass2_rows"]
    gained = tuple(sorted(int(k) for k, v in folds.items() if v["ablation"] - v["primary"] >= 0))
    inflated = tuple(sorted(int(k) for k, v in budget.items() if v["ablation"] > v["primary"]))
    return {
        "nonnegative_gain_folds": gained,
        "pass2_budget_inflated_folds": inflated,
        "folds_with_nonnegative_unique_tp_gain_without_budget_inflation": sum(
            1 for k in gained if k not in inflated),
        "unique_tp_difference": r["unique_tp"]["difference"],
        "unique_tp_difference_low": r["unique_tp"]["difference_low"],
        "primary_unique_tp": r["unique_tp"]["primary"],
        "ablation_pass2_rows": r["pass2_rows"]["ablation"],
        "primary_pass2_rows": r["pass2_rows"]["primary"],
        "coverage_difference_low": r["coverage"]["difference_low"],
        "leakage_difference_high": r["pass1_leakage_share"]["difference_high"],
        "pass2_negative_routing_difference_high":
            r["pass2_negative_routing_rate"]["difference_high"],
        "pass1_negative_recall_difference_low": r["pass1_negative_recall"]["difference_low"],
        "folds_with_nonnegative_unique_tp_gain": sum(
            1 for v in folds.values() if v["ablation"] - v["primary"] >= 0),
        "pass1_routing_identical": bool(np.array_equal(
            np.asarray(paired.states["ablation"]) == PASS_1_NOT_SUBSCRIBE,
            np.asarray(paired.states["primary"]) == PASS_1_NOT_SUBSCRIBE)),
    }


def _adoption_criteria(v: Mapping[str, Any], e: AdoptionEvidence, cr: Mapping[str, Any],
                       c6: str) -> Dict[str, str]:
    F = lambda x: Fraction(str(x))  # noqa: E731

    def verdict(*values, test) -> str:
        if any(x is None for x in values):
            return "not_evaluable"
        return "pass" if test() else "fail"

    c1 = cr["C1_unique_tp_gain"]
    out = {"C1_unique_tp_gain": verdict(
        v["unique_tp_difference"], v["unique_tp_difference_low"], v["primary_unique_tp"],
        v["ablation_pass2_rows"], v["primary_pass2_rows"],
        test=lambda: (F(v["unique_tp_difference_low"]) > F(c1["lower_bound_above"])
                      and v["primary_unique_tp"] > 0
                      and F(v["unique_tp_difference"]) >= F(c1["min_relative_gain"])
                      * F(v["primary_unique_tp"])
                      and v["ablation_pass2_rows"] <= v["primary_pass2_rows"]))}
    if e.ablation_changes_pass1 is False:
        out["C2_pass1_leakage"] = verdict(
            e.pass1_rules_identical_every_fold, v["pass1_routing_identical"],
            test=lambda: e.pass1_rules_identical_every_fold is True
            and v["pass1_routing_identical"] is True)
    else:
        out["C2_pass1_leakage"] = verdict(
            e.ablation_changes_pass1, v["leakage_difference_high"],
            test=lambda: F(v["leakage_difference_high"])
            <= F(cr["C2_pass1_leakage"]["otherwise_upper_bound_at_most"]))
    out["C3_obligations"] = verdict(e.fit_compliance_every_fold, e.new_material_violation_folds,
                                    test=lambda: e.fit_compliance_every_fold is True
                                    and e.new_material_violation_folds == 0)
    c4 = cr["C4_coverage"]["total_coverage_lower_bound_at_least"]
    out["C4_coverage"] = verdict(v["coverage_difference_low"],
                                 test=lambda: F(v["coverage_difference_low"]) >= F(c4))
    c5 = cr["C5_fold_reproducibility"]
    counted = (v["folds_with_nonnegative_unique_tp_gain_without_budget_inflation"]
               if c5["fold_counts_only_without_pass2_budget_inflation"]
               else v["folds_with_nonnegative_unique_tp_gain"])
    out["C5_fold_reproducibility"] = verdict(
        counted, test=lambda: counted >= c5["folds_with_nonnegative_unique_tp_gain_at_least"])
    out["C6_uncertainty"] = c6
    out["C7_solver"] = verdict(e.worse_fit_status_folds, e.runtime_ratio,
                               test=lambda: e.worse_fit_status_folds == 0
                               and F(e.runtime_ratio) <= F(cr["C7_solver"]["max_runtime_ratio"]))
    c8 = cr["C8_class_conditional"]
    out["C8_class_conditional"] = verdict(
        v["pass2_negative_routing_difference_high"], v["pass1_negative_recall_difference_low"],
        test=lambda: F(v["pass2_negative_routing_difference_high"])
        <= F(c8["pass2_negative_routing_rate_upper_bound_at_most"])
        and F(v["pass1_negative_recall_difference_low"])
        >= F(c8["pass1_negative_recall_lower_bound_at_least"]))
    out["C9_complexity"] = verdict(e.complexity_documented, e.rf_symmetry,
                                   test=lambda: e.complexity_documented is True and e.rf_symmetry
                                   in ("equivalent_objective", "asymmetry_documented"))
    return out


def evaluate_adoption(evidence: AdoptionEvidence, protocol: EvaluationProtocol, *,
                      paired: PairedRouting, upstream: UpstreamSources,
                      allow_working: bool = False) -> AdoptionDecision:
    """
    The pre-registered adoption gates. Every criterion must pass; one that
    fails or cannot be evaluated keeps the primary selector.

    The reference flags are never taken from the caller: they are the
    complementarity reference of ``upstream`` after ``verify_reference`` has
    passed on the paired rows (complete under a frozen protocol; a working
    protocol, with ``allow_working``, may leave pinned values pending). Only
    the pinned frozen protocol (``is_pinned_frozen``) yields an official
    decision. The intervals, counts, per-fold
    gains and per-fold Pass 2 budgets are recomputed here from ``paired``
    (``PairedRouting.ablation``) with the registered bootstrap
    (``run_registered_bootstrap``). C6 passes only if the reported
    ``evidence.uncertainty`` reproduces that run exactly; a malformed or
    incomplete run fails C6 with its reason. Evidence must come from
    registered Tier 1 / Tier 2 sources: test-derived results are refused.
    """
    _gate(protocol, allow_working)
    if not isinstance(evidence, AdoptionEvidence) or not evidence.sources:
        raise ProtocolError("adoption needs AdoptionEvidence naming its evidence sources")
    check_inputs(evidence.sources, protocol)
    for src in evidence.sources:
        tier = evidence_tier(src, protocol)
        if tier not in ("tier1", "tier2"):
            raise ProtocolError(f"evidence source {src!r} is {tier}: it may not inform adoption")
    if not isinstance(paired, PairedRouting) or paired.kind != "ablation":
        raise ProtocolError("adoption needs the paired ablation routing (PairedRouting.ablation)")
    frozen = protocol.status == STATUS_FROZEN
    verified = verify_reference(upstream, protocol, population=paired, complete=frozen)
    pinned = is_pinned_frozen(protocol)
    reference = verified.flags
    replay = run_registered_bootstrap(paired, ADOPTION_METRICS, protocol, reference=reference)
    c6, why = _uncertainty_verdict(evidence.uncertainty, replay)
    v = _adoption_values(replay, paired)
    out = _adoption_criteria(v, evidence, protocol["complementarity"]["criteria"], c6)
    decision = "adopt_candidate" if all(x == "pass" for x in out.values()) else "retain_primary"
    reasons = {
        "C1_unique_tp_gain": (f"pooled Pass 2 rows: ablation {v['ablation_pass2_rows']:g} vs "
                              f"primary {v['primary_pass2_rows']:g}"),
        "C5_fold_reproducibility": (
            f"non-negative unique-TP gain in folds {list(v['nonnegative_gain_folds'])}; Pass 2 "
            f"budget inflated in folds {list(v['pass2_budget_inflated_folds'])}"),
        "C6_uncertainty": why}
    pending = verified.identity.pending()
    provenance = {"protocol_id": protocol.protocol_id, "protocol_version": protocol.version,
                  "protocol_fingerprint": protocol.fingerprint(),
                  "protocol_status": protocol.status, "reference": verified.combination,
                  "reference_fingerprint": reference.fingerprint,
                  "upstream_identity": ("verified, complete" if not pending
                                        else "verified; pending: " + "; ".join(pending))}
    note = protocol["complementarity"]["adoption_effect"]
    if not frozen:
        note = ("not an official decision: the protocol is working"
                + (" and the upstream identity is incomplete (D-C10)" if pending else "")
                + ". " + note)
    elif not pinned:
        note = ("not an official decision: the protocol is not the pinned frozen record "
                "(load_frozen_protocol). " + note)
    return AdoptionDecision(decision, MappingProxyType(out), MappingProxyType(reasons),
                            protocol["complementarity"]["evidence_label"], note,
                            official=pinned and verified.complete,
                            provenance=MappingProxyType(provenance))
