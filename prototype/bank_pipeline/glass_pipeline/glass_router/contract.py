"""
glass_pipeline.glass_router.contract
====================================
GLASS Router adapter for the shared Stage 2 operating contract (PR 37 Commit 4,
Step 4B). Support only: nothing here is wired into a notebook.

Mapping (values only from a schema-2 ``SharedOperatingContractV2``):

    max_union_leakage_rate_pass1  <- max_pass1_leakage_rate   (existing union cap)
    min_union_npv_pass1           <- min_pass1_npv
    min_union_support_pass1       <- min_pass1_support
    min_union_precision_pass2     <- min_pass2_precision
    min_union_support_pass2       <- min_pass2_support
    pass2_union_objective         =  "rows"   (lexicographic coverage)
    contract_provenance           =  constraint_provenance(contract)

Every obligation is enforced on the SELECTED UNION by the ILP
(``ILPRuleSelector``); the per-rule quality gates stay as arm-internal
candidate filters and never stand in for union compliance. A contract with a
leakage count cap is refused: GLASS has no union count constraint, and the
working profile sets none.

``glass_config_from_contract`` also requires the approved F17 primary settings
on the base profile and refuses a base that already sets a union obligation to
another value. ``check_glass_config_follows_contract`` refuses a config that
does not follow a contract, including one that claims compliance through
per-rule gates alone. ``glass_union_counts`` and ``glass_fit_outcome`` turn a
selection into the arm-neutral obligation check and fit outcome
(``shared.stage2.obligations``); ``glass_record_reproduction`` records an
explicit deterministic rerun. The pipeline's ``selection_reports()`` supplies
the per-pass reports, inner-fold Pass 1 selections included.

GLASS's fit-time estimates are in-sample on the rows each pass was selected on;
the RF Router's are inner out-of-fold. The obligations are the same; the
estimators are not (PR 37 F17 L2, F31 K2).

This module imports no other arm and reads no artifact.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from shared.stage2.contract import (
    ContractViolation,
    SharedOperatingContractV2,
    check_contract_provenance,
    constraint_provenance,
)
from shared.stage2.obligations import (
    OUTCOME_FEASIBLE_GREEDY,
    OUTCOME_FEASIBLE_OPTIMAL,
    OUTCOME_FEASIBLE_UNPROVEN,
    OUTCOME_INFEASIBLE,
    OUTCOME_INSUFFICIENT,
    OUTCOME_NO_SOLUTION,
    FitOutcome,
    ObligationReport,
    PassCounts,
    check_pass1_fit,
    check_pass2_fit,
    classify_fit_outcome,
)

from .core.config import (
    INNER_RANKER_FOLD_NESTED_RF,
    PASS2_OBJECTIVE_POSITIVES,
    PASS2_OBJECTIVE_ROWS,
    GlassRouterConfig,
)
from .ilp_rule_selector.ilp_rule_selector import INFEASIBLE as ILP_INFEASIBLE
from .ilp_rule_selector.utils import get_covered_array

__all__ = [
    "GLASS_F17_PRIMARY",
    "UNION_OBLIGATION_FIELDS",
    "glass_union_obligations_from_contract",
    "glass_config_from_contract",
    "check_glass_config_follows_contract",
    "glass_union_counts",
    "glass_pass_report",
    "glass_fit_outcome",
    "glass_record_reproduction",
]

#: The approved F17 primary settings (PR 37 F17 decision record, gate G1).
GLASS_F17_PRIMARY = MappingProxyType({
    "pass2_population": "oof_remainder",
    "pass1_oof_folds": 5,
    "random_state": 42,
    "remainder_min_size": 500,
    "inner_feature_ranker": INNER_RANKER_FOLD_NESTED_RF,
})

#: The GlassRouterConfig fields the shared contract fixes.
UNION_OBLIGATION_FIELDS = (
    "max_union_leakage_rate_pass1", "min_union_npv_pass1", "min_union_support_pass1",
    "min_union_precision_pass2", "min_union_support_pass2", "pass2_union_objective",
)
_UNSET = MappingProxyType({f: None for f in UNION_OBLIGATION_FIELDS[:-1]}
                          | {"pass2_union_objective": PASS2_OBJECTIVE_POSITIVES})


def _require_v2(contract: Any) -> SharedOperatingContractV2:
    if not isinstance(contract, SharedOperatingContractV2):
        raise ContractViolation("the GLASS adapter needs a schema-2 contract "
                                f"(SharedOperatingContractV2), got {type(contract).__name__}")
    return contract


def glass_union_obligations_from_contract(contract: SharedOperatingContractV2) -> dict:
    """The GlassRouterConfig union-obligation fields under ``contract``, from it alone."""
    c = _require_v2(contract)
    if c.max_pass1_leakage_count is not None:
        raise ContractViolation(
            "the GLASS adapter has no union leakage-count constraint; a contract with "
            f"max_pass1_leakage_count={c.max_pass1_leakage_count!r} is not supported")
    return {
        "max_union_leakage_rate_pass1": float(c.max_pass1_leakage_rate),
        "min_union_npv_pass1": float(c.min_pass1_npv),
        "min_union_support_pass1": int(c.min_pass1_support),
        "min_union_precision_pass2": float(c.min_pass2_precision),
        "min_union_support_pass2": int(c.min_pass2_support),
        "pass2_union_objective": PASS2_OBJECTIVE_ROWS,
    }


def _f17_problems(config: GlassRouterConfig) -> list:
    return [f"{name}={getattr(config, name)!r}, the F17 primary requires {want!r}"
            for name, want in GLASS_F17_PRIMARY.items() if getattr(config, name) != want]


def glass_config_from_contract(base: GlassRouterConfig,
                               contract: SharedOperatingContractV2) -> GlassRouterConfig:
    """
    ``base`` with the contract's union obligations and provenance.

    ``base`` carries the arm-internal settings (per-rule gates, rule counts,
    novelty, complexity, ...) and must already hold the F17 primary settings.
    A union field it sets to another value is refused, not overwritten.
    """
    if not isinstance(base, GlassRouterConfig):
        raise ContractViolation(f"expected a GlassRouterConfig, got {type(base).__name__}")
    obligations = glass_union_obligations_from_contract(contract)
    problems = _f17_problems(base)
    for name, want in obligations.items():
        have = getattr(base, name)
        if have != _UNSET[name] and have != want:
            problems.append(f"{name}={have!r} in the base profile conflicts with the contract's "
                            f"{want!r}")
    if base.contract_provenance is not None:
        problems.append("the base profile already records a contract_provenance")
    if problems:
        raise ContractViolation("cannot build a GLASS config under the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    return dataclasses.replace(base, **obligations,
                               contract_provenance=constraint_provenance(contract))


def check_glass_config_follows_contract(config: GlassRouterConfig,
                                        contract: SharedOperatingContractV2) -> None:
    """Refuse a GLASS config that does not enforce ``contract`` on its selected unions."""
    if not isinstance(config, GlassRouterConfig):
        raise ContractViolation(f"expected a GlassRouterConfig, got {type(config).__name__}")
    obligations = glass_union_obligations_from_contract(contract)
    problems = _f17_problems(config)
    problems += [f"{name}={getattr(config, name)!r}, the contract requires {want!r}"
                 for name, want in obligations.items() if getattr(config, name) != want]
    if config.contract_provenance is None:
        problems.append("contract_provenance is missing: the config claims no shared contract")
    if problems:
        raise ContractViolation("GLASS config does not follow the shared contract:\n"
                                + "\n".join(f"  - {p}" for p in problems))
    check_contract_provenance(config.contract_provenance, contract)


def glass_union_counts(rules: Iterable[Any], y: Any) -> PassCounts:
    """Counts of the union of ``rules`` on the rows they were selected on (labels ``y``)."""
    y_arr = np.asarray(y).astype(int)
    covered = np.zeros(len(y_arr), dtype=bool)
    for rule in rules:
        covered[get_covered_array(rule, len(y_arr))] = True
    return PassCounts(population_rows=len(y_arr), population_positives=int(y_arr.sum()),
                      routed_rows=int(covered.sum()), routed_positives=int(y_arr[covered].sum()))


def glass_pass_report(scope: str, rules: Iterable[Any], y: Any,
                      contract: SharedOperatingContractV2) -> ObligationReport:
    """The arm-neutral fit-time obligation check of one GLASS pass's selected union."""
    counts = glass_union_counts(rules, y)
    if scope == "pass1":
        return check_pass1_fit(counts, contract)
    if scope == "pass2":
        return check_pass2_fit(counts, contract)
    raise ContractViolation(f"scope must be 'pass1' or 'pass2', got {scope!r}")


def _phase_solver(info: Mapping[str, Any]) -> dict:
    log = info.get("solver_log") or {}
    return {"status": info.get("status"), "sol_status": info.get("sol_status"),
            "state": info.get("state"), "objective": log.get("objective"),
            "bound": log.get("bound"), "gap_abs": log.get("gap_abs"),
            "gap_rel": log.get("gap_rel"), "result": log.get("result")}


def glass_fit_outcome(scope: str, selection_report: Mapping[str, Any],
                      report: Optional[ObligationReport]) -> FitOutcome:
    """
    Classify one ILP pass from its ``selection_report_`` entry.

    ``solver`` records both lexicographic phases. The MIP gap is Phase 1's
    (the coverage objective), read from CBC's log by the selector under
    set-level obligations: PuLP's CBC interface returns no bound, so when
    the log is unavailable ``gap`` stays ``None`` (never estimated) and an
    unproven incumbent cannot complete its documentation (Commit 4, M3).
    ``reproduced`` is set only by ``glass_record_reproduction`` after an
    explicit rerun, never assumed.
    """
    source = selection_report.get("source")
    phase1 = selection_report.get("phase1") or {}
    phase2 = selection_report.get("phase2") or {}
    p1 = _phase_solver(phase1)
    solver = {"status": phase1.get("status"), "sol_status": phase1.get("sol_status"),
              "gap": p1["gap_rel"], "gap_abs": p1["gap_abs"],
              "gap_available": p1["gap_rel"] is not None, "gap_source": (
                  "cbc_log" if p1["gap_rel"] is not None else None),
              "phase1": p1, "phase2": _phase_solver(phase2) if phase2 else None,
              "reproduced": None,
              "lexicographic_guarantee": selection_report.get("lexicographic_guarantee"),
              "selected_rule_ids": tuple(selection_report.get("selected_rule_ids") or ())}
    if source == "ilp":
        proven = selection_report.get("lexicographic_guarantee") == "proven"
        outcome = OUTCOME_FEASIBLE_OPTIMAL if proven else OUTCOME_FEASIBLE_UNPROVEN
        reason = "CBC proved both lexicographic phases optimal" if proven else (
            "time-limited incumbent: feasible after validation, optimality not proven")
    elif source == "greedy_fallback":
        outcome, reason = OUTCOME_FEASIBLE_GREEDY, "CBC returned no usable solution"
    elif source == "infeasible":
        if selection_report.get("stage") is not None or phase1.get("state") == ILP_INFEASIBLE:
            outcome, reason = OUTCOME_INFEASIBLE, f"proven infeasible ({selection_report.get('stage') or 'ILP'})"
        else:
            outcome, reason = OUTCOME_NO_SOLUTION, "no validated feasible selection found"
    elif isinstance(source, str) and source.startswith("empty"):
        outcome, reason = OUTCOME_INSUFFICIENT, "no rule selected (min_rules = 0)"
    else:
        raise ContractViolation(f"unrecognised selection report source {source!r}")
    return classify_fit_outcome(outcome, scope=scope, reason=reason, report=report, solver=solver)


_REPRODUCED_KEYS = ("source", "selected_rule_ids", "union_rows", "union_good", "union_coverage")


def glass_record_reproduction(outcome: FitOutcome,
                              first_report: Mapping[str, Any],
                              rerun_report: Mapping[str, Any]) -> FitOutcome:
    """
    ``outcome`` with ``solver["reproduced"]`` set from an explicit rerun.

    The rerun must be the same fit repeated (same data, config and seed). It
    reproduces the selection only if the source, the selected rule ids in
    order and the union counts are identical. A time-limited CBC search
    depends on wall-clock progress, so a mismatch is a real possibility and
    is recorded as ``False``, not retried.
    """
    if not isinstance(outcome, FitOutcome):
        raise ContractViolation(f"expected a FitOutcome, got {type(outcome).__name__}")
    missing = [k for k in ("source", "selected_rule_ids") if first_report.get(k) is None]
    if missing:
        raise ContractViolation(f"the first selection report lacks {missing}: a reproduction "
                                "check needs a set-level selection report")
    same = all(first_report.get(k) == rerun_report.get(k) for k in _REPRODUCED_KEYS)
    solver = dict(outcome.solver)
    solver["reproduced"] = bool(same)
    return dataclasses.replace(outcome, solver=MappingProxyType(solver))
