"""
shared.stage2.obligations
=========================
Arm-neutral checks of the shared Stage 2 obligations (PR 37 Commit 4, Step 4A).

Two uses, kept apart.

Fit time (binding)
    ``check_pass1_fit`` / ``check_pass2_fit`` decide whether one pass's routed
    set meets a schema-2 contract on the population that pass was selected on:
    a fit's Pass 1 population, or the remainder its own Pass 1 left. Every
    comparison is exact integer arithmetic on counts. An empty routed set, a set
    below its support floor, or an undefined ratio is infeasible, never
    "satisfied". A population without both classes is rejected.
    ``require_fit`` raises ``ObligationInfeasible`` unless every obligation holds.

Evaluation (observed, not certified)
    ``observe_routing`` reports what a routed population achieved (one outer
    fold, or the folds pooled by summing their counts with ``pool_counts``):
    the contract metrics with numerators and denominators, and each outcome
    obligation as satisfied, violated or undefined, with its exact signed
    shortfall. An observed violation is a violation however small it is.
    Undefined stays undefined: never 0, never compliant. Support floors are
    fit-time obligations; on evaluation rows the routed counts are reported,
    not judged. Classifying shortfalls, and what follows from them, is
    Step 4C; nothing here does either.

Fit outcomes
    ``classify_fit_outcome`` records how a fit ended (feasible optimum,
    feasible unproven incumbent, greedy fallback, proven infeasibility, no
    feasible solution found, insufficient population or support) and refuses
    a claim that contradicts the independent obligation check.

Limits are read as the decimals a contract writes (0.15 is exactly 3/20, not
the nearest binary float). Ratios use the diagnostics registry's ``Ratio`` and
metric definitions, imported and never modified: the registry is pinned by the
pre-correction baseline. Nothing here imports either router.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np

from shared.stage2.contract import (
    METRIC_DEFINITIONS_ID,
    ContractViolation,
    SharedOperatingContractV2,
)
from shared.stage2.diagnostics.metrics import METRIC_DEFINITIONS, Ratio

__all__ = [
    "CONTRACT_METRICS",
    "SATISFIED", "VIOLATED", "UNDEFINED_STATUS",
    "POPULATION_OK", "POPULATION_EMPTY", "POPULATION_NO_POSITIVES", "POPULATION_NO_NEGATIVES",
    "OUTCOME_FEASIBLE_OPTIMAL", "OUTCOME_FEASIBLE_UNPROVEN", "OUTCOME_FEASIBLE_GREEDY",
    "OUTCOME_INFEASIBLE", "OUTCOME_NO_SOLUTION", "OUTCOME_INSUFFICIENT", "FIT_OUTCOMES",
    "DISPOSITION_PRIMARY", "DISPOSITION_PRIMARY_IF_DOCUMENTED", "DISPOSITION_REQUIRES_REVIEW",
    "DISPOSITION_STOP",
    "ObligationInfeasible",
    "exact_limit", "leakage_cap_count", "contract_metric_definitions",
    "PassCounts", "RoutingCounts", "ObligationResult", "ObligationReport", "FitOutcome",
    "check_pass1_fit", "check_pass2_fit", "require_fit",
    "pool_counts", "routing_ratios", "observe_routing", "observe_folds",
    "classify_fit_outcome",
]

#: The shared metric set named by ``METRIC_DEFINITIONS_ID``. Names are the
#: diagnostics registry's; so are their numerators and denominators.
CONTRACT_METRICS: Tuple[str, ...] = (
    "pass1_npv",             # TN_P1 / N_P1
    "pass1_leakage_share",   # FN_P1 / P_total            (ceiling)
    "pass2_precision",       # TP_P2 / N_P2
    "pass1_rate",            # N_P1 / N_evaluated          (Pass 1 coverage)
    "pass2_rate",            # N_P2 / N_evaluated          (Pass 2 coverage)
    "coverage",              # (N_P1 + N_P2) / N_evaluated (total coverage)
    "abstain_rate",          # N_abstain / N_evaluated
    "pass2_remainder_rate",  # N_P2 / (N_evaluated - N_P1) (reported, not constrained)
)
_METRIC_SETS = MappingProxyType({METRIC_DEFINITIONS_ID: CONTRACT_METRICS})
_unknown = [m for m in CONTRACT_METRICS if m not in METRIC_DEFINITIONS]
if _unknown:  # pragma: no cover - the registry is pinned
    raise ImportError(f"contract metrics missing from the diagnostics registry: {_unknown}")

SATISFIED, VIOLATED, UNDEFINED_STATUS = "satisfied", "violated", "undefined"

POPULATION_OK = "ok"
POPULATION_EMPTY = "rejected_empty"
POPULATION_NO_POSITIVES = "rejected_no_positives"
POPULATION_NO_NEGATIVES = "rejected_no_negatives"

OUTCOME_FEASIBLE_OPTIMAL = "feasible_optimal"
OUTCOME_FEASIBLE_UNPROVEN = "feasible_unproven_incumbent"
OUTCOME_FEASIBLE_GREEDY = "feasible_greedy_fallback"
OUTCOME_INFEASIBLE = "proven_infeasible"
OUTCOME_NO_SOLUTION = "no_feasible_solution_found"
OUTCOME_INSUFFICIENT = "insufficient_population_or_support"
FIT_OUTCOMES = (OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN, OUTCOME_FEASIBLE_GREEDY,
                OUTCOME_INFEASIBLE, OUTCOME_NO_SOLUTION, OUTCOME_INSUFFICIENT)
_FEASIBLE_OUTCOMES = (OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN, OUTCOME_FEASIBLE_GREEDY)

DISPOSITION_PRIMARY = "primary"
DISPOSITION_PRIMARY_IF_DOCUMENTED = "primary_if_documented"
DISPOSITION_REQUIRES_REVIEW = "requires_review"
DISPOSITION_STOP = "stop"
_DISPOSITION = MappingProxyType({
    OUTCOME_FEASIBLE_OPTIMAL: DISPOSITION_PRIMARY,
    # Commit 4, M3: acceptable only with every obligation verified independently
    # and the solver status, gap and a deterministic reproduction documented.
    OUTCOME_FEASIBLE_UNPROVEN: DISPOSITION_PRIMARY_IF_DOCUMENTED,
    # M3: never equivalent to an optimised result; needs an explicit review of
    # its objective gap and comparability.
    OUTCOME_FEASIBLE_GREEDY: DISPOSITION_REQUIRES_REVIEW,
    OUTCOME_INFEASIBLE: DISPOSITION_STOP,
    OUTCOME_NO_SOLUTION: DISPOSITION_STOP,
    OUTCOME_INSUFFICIENT: DISPOSITION_STOP,
})
_UNPROVEN_DOCUMENTATION = ("status", "gap", "reproduced")


class ObligationInfeasible(ContractViolation):
    """A routed set does not meet the shared contract at fit time."""

    def __init__(self, report: "ObligationReport"):
        self.report = report
        failed = [f"{r.obligation}={r.status}" for r in report.results if r.status != SATISFIED]
        if report.population_status != POPULATION_OK:
            failed.insert(0, f"population={report.population_status}")
        super().__init__(f"{report.scope} does not meet the shared contract at fit time: "
                         + ", ".join(failed))


# ----------------------------------------------------------------------
# Limits and definitions
# ----------------------------------------------------------------------
def _require_v2(contract: Any) -> SharedOperatingContractV2:
    if not isinstance(contract, SharedOperatingContractV2):
        raise ContractViolation("the shared obligations need a schema-2 contract "
                                f"(SharedOperatingContractV2), got {type(contract).__name__}")
    return contract


def exact_limit(value: Any) -> Fraction:
    """The decimal a contract writes, as an exact fraction: 0.15 -> 3/20."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ContractViolation(f"a limit must be a real number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ContractViolation(f"a limit must be finite, got {value!r}")
    return Fraction(repr(v))


def leakage_cap_count(contract: SharedOperatingContractV2, positives: int) -> int:
    """Most positives Pass 1 may route in a population with ``positives`` positives."""
    c = _require_v2(contract)
    p = _count("positives", positives)
    rate = exact_limit(c.max_pass1_leakage_rate)
    cap = (rate.numerator * p) // rate.denominator
    if c.max_pass1_leakage_count is not None:
        cap = min(cap, int(c.max_pass1_leakage_count))
    return cap


def contract_metric_definitions(contract: SharedOperatingContractV2) -> Mapping[str, Tuple[str, str]]:
    """Name -> (numerator, denominator) for the contract's metric set (registry wording)."""
    names = _METRIC_SETS[_require_v2(contract).metric_definitions]
    return MappingProxyType({n: METRIC_DEFINITIONS[n] for n in names})


def _count(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ContractViolation(f"{name} must be a count (an integer >= 0), got {value!r}")
    return int(value)


# ----------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PassCounts:
    """One pass's routed set inside the population that pass was selected on."""

    population_rows: int
    population_positives: int
    routed_rows: int
    routed_positives: int

    def __post_init__(self) -> None:
        for name in ("population_rows", "population_positives", "routed_rows", "routed_positives"):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        problems = []
        if self.population_positives > self.population_rows:
            problems.append("population_positives > population_rows")
        if self.routed_rows > self.population_rows:
            problems.append("routed_rows > population_rows")
        if self.routed_positives > min(self.routed_rows, self.population_positives):
            problems.append("routed_positives exceeds routed_rows or population_positives")
        if self.routed_negatives > self.population_negatives:
            problems.append("routed negatives exceed population negatives")
        if problems:
            raise ContractViolation("inconsistent pass counts: " + "; ".join(problems))

    @property
    def population_negatives(self) -> int:
        return self.population_rows - self.population_positives

    @property
    def routed_negatives(self) -> int:
        return self.routed_rows - self.routed_positives


#: Decision tokens: the routers' (``pass1`` / ``pass2`` / ``uncertain``, both
#: arms) and the diagnostics schema's router states.
_TOKENS = MappingProxyType({
    "pass1": "pass1", "pass2": "pass2", "uncertain": "abstain",
    "PASS_1_NOT_SUBSCRIBE": "pass1", "PASS_2_SUBSCRIBE": "pass2", "ABSTAIN": "abstain",
})


@dataclass(frozen=True)
class RoutingCounts:
    """Three-state routing outcome of one evaluated population, as counts."""

    rows: int
    positives: int
    pass1_rows: int
    pass1_positives: int
    pass2_rows: int
    pass2_positives: int

    def __post_init__(self) -> None:
        for name in ("rows", "positives", "pass1_rows", "pass1_positives", "pass2_rows",
                     "pass2_positives"):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        problems = []
        if self.positives > self.rows:
            problems.append("positives > rows")
        if self.pass1_rows + self.pass2_rows > self.rows:
            problems.append("pass1_rows + pass2_rows > rows (the states must be exclusive)")
        if self.pass1_positives > self.pass1_rows or self.pass2_positives > self.pass2_rows:
            problems.append("a pass holds more positives than rows")
        if self.pass1_positives + self.pass2_positives > self.positives:
            problems.append("routed positives exceed all positives")
        if ((self.pass1_rows - self.pass1_positives) + (self.pass2_rows - self.pass2_positives)
                > self.rows - self.positives):
            problems.append("routed negatives exceed all negatives")
        if problems:
            raise ContractViolation("inconsistent routing counts: " + "; ".join(problems))

    @property
    def abstain_rows(self) -> int:
        return self.rows - self.pass1_rows - self.pass2_rows

    @property
    def abstain_positives(self) -> int:
        return self.positives - self.pass1_positives - self.pass2_positives

    def __add__(self, other: "RoutingCounts") -> "RoutingCounts":
        if not isinstance(other, RoutingCounts):
            return NotImplemented
        return RoutingCounts(*(getattr(self, f) + getattr(other, f) for f in
                               ("rows", "positives", "pass1_rows", "pass1_positives",
                                "pass2_rows", "pass2_positives")))

    @classmethod
    def from_decisions(cls, decisions: Any, y: Any) -> "RoutingCounts":
        """Counts from per-row decision tokens and binary labels for the same rows."""
        d = np.asarray(decisions, dtype=object)
        lab = np.asarray(y)
        if d.ndim != 1 or lab.ndim != 1 or len(d) != len(lab):
            raise ContractViolation(f"decisions and y must be 1-d and the same length, got "
                                    f"{d.shape} and {lab.shape}")
        if len(d) == 0:
            raise ContractViolation("an evaluated population must not be empty")
        unknown = sorted({str(t) for t in d if t not in _TOKENS})
        if unknown:
            raise ContractViolation(f"unknown decision tokens {unknown}; expected one of "
                                    f"{sorted(_TOKENS)}")
        if not np.isin(lab, (0, 1)).all():
            raise ContractViolation("y must be binary 0/1")
        state = np.array([_TOKENS[t] for t in d], dtype=object)
        pos = lab.astype(int) == 1
        p1, p2 = state == "pass1", state == "pass2"
        return cls(rows=len(lab), positives=int(pos.sum()),
                   pass1_rows=int(p1.sum()), pass1_positives=int((p1 & pos).sum()),
                   pass2_rows=int(p2.sum()), pass2_positives=int((p2 & pos).sum()))


def pool_counts(counts: Iterable[RoutingCounts]) -> RoutingCounts:
    """Pooled counts: numerators and denominators summed across folds, never ratios averaged."""
    items = list(counts)
    if not items:
        raise ContractViolation("nothing to pool")
    if not all(isinstance(c, RoutingCounts) for c in items):
        raise ContractViolation("pool_counts takes RoutingCounts only")
    total = items[0]
    for c in items[1:]:
        total = total + c
    return total


def routing_ratios(counts: RoutingCounts) -> Mapping[str, Ratio]:
    """Every contract metric as a registry ``Ratio`` (value None when undefined)."""
    if not isinstance(counts, RoutingCounts):
        raise ContractViolation(f"expected RoutingCounts, got {type(counts).__name__}")
    c = counts
    return MappingProxyType({
        "pass1_npv": Ratio("pass1_npv", c.pass1_rows - c.pass1_positives, c.pass1_rows),
        "pass1_leakage_share": Ratio("pass1_leakage_share", c.pass1_positives, c.positives),
        "pass2_precision": Ratio("pass2_precision", c.pass2_positives, c.pass2_rows),
        "pass1_rate": Ratio("pass1_rate", c.pass1_rows, c.rows),
        "pass2_rate": Ratio("pass2_rate", c.pass2_rows, c.rows),
        "coverage": Ratio("coverage", c.pass1_rows + c.pass2_rows, c.rows),
        "abstain_rate": Ratio("abstain_rate", c.abstain_rows, c.rows),
        "pass2_remainder_rate": Ratio("pass2_remainder_rate", c.pass2_rows,
                                      c.rows - c.pass1_rows),
    })


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ObligationResult:
    """
    One obligation on one population.

    ``shortfall`` is exact and signed: how far the observed value misses the
    limit (floor - value, or value - ceiling). Positive = violated by that
    much; zero or negative = met with that slack. ``None`` when undefined.
    """

    obligation: str
    metric: str
    numerator: int
    denominator: Optional[int]
    limit: str
    comparison: str
    status: str
    shortfall: Optional[Fraction]

    @property
    def value(self) -> Optional[float]:
        if self.denominator is None:
            return float(self.numerator)
        return None if self.denominator == 0 else self.numerator / self.denominator

    def to_dict(self) -> dict:
        return {
            "obligation": self.obligation, "metric": self.metric,
            "numerator": self.numerator, "denominator": self.denominator,
            "value": self.value, "limit": self.limit, "comparison": self.comparison,
            "status": self.status,
            "shortfall": None if self.shortfall is None else str(self.shortfall),
            "shortfall_float": None if self.shortfall is None else float(self.shortfall),
        }


def _ratio_floor(obligation, metric, num, den, limit) -> ObligationResult:
    lim = exact_limit(limit)
    if den == 0:
        return ObligationResult(obligation, metric, num, den, str(lim), ">=", UNDEFINED_STATUS, None)
    gap = lim - Fraction(num, den)
    return ObligationResult(obligation, metric, num, den, str(lim), ">=",
                            SATISFIED if gap <= 0 else VIOLATED, gap)


def _ratio_ceiling(obligation, metric, num, den, limit) -> ObligationResult:
    lim = exact_limit(limit)
    if den == 0:
        return ObligationResult(obligation, metric, num, den, str(lim), "<=", UNDEFINED_STATUS, None)
    gap = Fraction(num, den) - lim
    return ObligationResult(obligation, metric, num, den, str(lim), "<=",
                            SATISFIED if gap <= 0 else VIOLATED, gap)


def _count_floor(obligation, metric, count, floor) -> ObligationResult:
    gap = Fraction(int(floor) - count)
    return ObligationResult(obligation, metric, count, None, str(int(floor)), ">=",
                            SATISFIED if gap <= 0 else VIOLATED, gap)


def _count_ceiling(obligation, metric, count, cap) -> ObligationResult:
    gap = Fraction(count - int(cap))
    return ObligationResult(obligation, metric, count, None, str(int(cap)), "<=",
                            SATISFIED if gap <= 0 else VIOLATED, gap)


def _population_status(rows: int, positives: int) -> str:
    if rows == 0:
        return POPULATION_EMPTY
    if positives == 0:
        return POPULATION_NO_POSITIVES
    if positives == rows:
        return POPULATION_NO_NEGATIVES
    return POPULATION_OK


@dataclass(frozen=True)
class ObligationReport:
    """Every obligation checked on one population, with the contract it was checked against."""

    stage: str                 # "fit" | "evaluation"
    scope: str                 # "pass1" | "pass2" | "routing"
    contract_id: str
    contract_fingerprint: str
    population_rows: int
    population_positives: int
    population_status: str
    results: Tuple[ObligationResult, ...]
    counts: Optional[RoutingCounts] = None
    ratios: Tuple[Ratio, ...] = field(default=())

    @property
    def all_satisfied(self) -> bool:
        return (self.population_status == POPULATION_OK
                and all(r.status == SATISFIED for r in self.results))

    @property
    def violated(self) -> Tuple[ObligationResult, ...]:
        return tuple(r for r in self.results if r.status == VIOLATED)

    @property
    def undefined(self) -> Tuple[ObligationResult, ...]:
        return tuple(r for r in self.results if r.status == UNDEFINED_STATUS)

    def result(self, obligation: str) -> ObligationResult:
        for r in self.results:
            if r.obligation == obligation:
                return r
        raise KeyError(obligation)

    def ratio(self, name: str) -> Ratio:
        for r in self.ratios:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "scope": self.scope, "contract_id": self.contract_id,
            "contract_fingerprint": self.contract_fingerprint,
            "population_rows": self.population_rows,
            "population_positives": self.population_positives,
            "population_status": self.population_status,
            "all_satisfied": self.all_satisfied,
            "results": [r.to_dict() for r in self.results],
            "counts": None if self.counts is None else {
                f: getattr(self.counts, f) for f in ("rows", "positives", "pass1_rows",
                                                     "pass1_positives", "pass2_rows",
                                                     "pass2_positives")},
            "ratios": {r.name: {"numerator": int(r.numerator), "denominator": int(r.denominator),
                                "value": r.value, "status": r.status} for r in self.ratios},
        }


# ----------------------------------------------------------------------
# Fit time
# ----------------------------------------------------------------------
def check_pass1_fit(counts: PassCounts, contract: SharedOperatingContractV2) -> ObligationReport:
    """Pass 1 at fit time: support, NPV floor, leakage ceiling (and count cap, if any)."""
    c = _require_v2(contract)
    if not isinstance(counts, PassCounts):
        raise ContractViolation(f"expected PassCounts, got {type(counts).__name__}")
    results = [
        _count_floor("pass1_support_floor", "pass1_rows", counts.routed_rows, c.min_pass1_support),
        _ratio_floor("pass1_npv_floor", "pass1_npv", counts.routed_negatives, counts.routed_rows,
                     c.min_pass1_npv),
        _ratio_ceiling("pass1_leakage_ceiling", "pass1_leakage_share", counts.routed_positives,
                       counts.population_positives, c.max_pass1_leakage_rate),
    ]
    if c.max_pass1_leakage_count is not None:
        results.append(_count_ceiling("pass1_leakage_count_cap", "pass1_positives",
                                      counts.routed_positives, c.max_pass1_leakage_count))
    return ObligationReport("fit", "pass1", c.contract_id, c.fingerprint(),
                            counts.population_rows, counts.population_positives,
                            _population_status(counts.population_rows,
                                               counts.population_positives),
                            tuple(results))


def check_pass2_fit(counts: PassCounts, contract: SharedOperatingContractV2) -> ObligationReport:
    """Pass 2 at fit time, on the remainder its own Pass 1 left: support and precision floor."""
    c = _require_v2(contract)
    if not isinstance(counts, PassCounts):
        raise ContractViolation(f"expected PassCounts, got {type(counts).__name__}")
    results = (
        _count_floor("pass2_support_floor", "pass2_rows", counts.routed_rows, c.min_pass2_support),
        _ratio_floor("pass2_precision_floor", "pass2_precision", counts.routed_positives,
                     counts.routed_rows, c.min_pass2_precision),
    )
    return ObligationReport("fit", "pass2", c.contract_id, c.fingerprint(),
                            counts.population_rows, counts.population_positives,
                            _population_status(counts.population_rows,
                                               counts.population_positives),
                            results)


def require_fit(report: ObligationReport) -> ObligationReport:
    """Return ``report`` if every fit-time obligation holds; otherwise raise. No relaxation."""
    if not isinstance(report, ObligationReport) or report.stage != "fit":
        raise ContractViolation("require_fit takes a fit-time ObligationReport")
    if not report.all_satisfied:
        raise ObligationInfeasible(report)
    return report


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def observe_routing(counts: RoutingCounts, contract: SharedOperatingContractV2) -> ObligationReport:
    """
    What a routed population achieved against the contract's outcome obligations.

    Observed, not certified: an undefined ratio is reported as undefined, and a
    violation is reported however small. Support floors are not judged here.
    """
    c = _require_v2(contract)
    ratios = routing_ratios(counts)
    results = [
        _ratio_floor("pass1_npv_floor", "pass1_npv", counts.pass1_rows - counts.pass1_positives,
                     counts.pass1_rows, c.min_pass1_npv),
        _ratio_ceiling("pass1_leakage_ceiling", "pass1_leakage_share", counts.pass1_positives,
                       counts.positives, c.max_pass1_leakage_rate),
    ]
    if c.max_pass1_leakage_count is not None:
        results.append(_count_ceiling("pass1_leakage_count_cap", "pass1_positives",
                                      counts.pass1_positives, c.max_pass1_leakage_count))
    results.append(_ratio_floor("pass2_precision_floor", "pass2_precision",
                                counts.pass2_positives, counts.pass2_rows, c.min_pass2_precision))
    return ObligationReport("evaluation", "routing", c.contract_id, c.fingerprint(),
                            counts.rows, counts.positives,
                            _population_status(counts.rows, counts.positives),
                            tuple(results), counts=counts,
                            ratios=tuple(ratios[m] for m in CONTRACT_METRICS))


def observe_folds(fold_counts: Mapping[Any, RoutingCounts], contract: SharedOperatingContractV2):
    """Per-fold observations and the pooled one, from summed counts. Returns (folds, pooled)."""
    if not isinstance(fold_counts, Mapping) or not fold_counts:
        raise ContractViolation("observe_folds takes a non-empty mapping fold -> RoutingCounts")
    per_fold = MappingProxyType({k: observe_routing(v, contract) for k, v in fold_counts.items()})
    return per_fold, observe_routing(pool_counts(fold_counts.values()), contract)


# ----------------------------------------------------------------------
# Fit outcomes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FitOutcome:
    """How one fit-time selection ended, and what the PR 38 protocol may do with it."""

    scope: str
    outcome: str
    disposition: str
    reason: str
    report: Optional[ObligationReport]
    solver: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), hash=False)

    @property
    def documentation_complete(self) -> bool:
        """For an unproven incumbent: solver status, gap and a reproduction are all recorded."""
        if self.outcome != OUTCOME_FEASIBLE_UNPROVEN:
            return True
        s = self.solver
        return (s.get("status") is not None and s.get("gap") is not None
                and s.get("reproduced") is True)

    @property
    def primary_eligible(self) -> bool:
        return (self.disposition == DISPOSITION_PRIMARY
                or (self.disposition == DISPOSITION_PRIMARY_IF_DOCUMENTED
                    and self.documentation_complete))


def classify_fit_outcome(
    outcome: str,
    *,
    scope: str,
    reason: str,
    report: Optional[ObligationReport] = None,
    solver: Optional[Mapping[str, Any]] = None,
) -> FitOutcome:
    """
    Record how a fit ended and check the claim against the obligation report.

    A feasible outcome needs a fit-time report in which every obligation holds
    (independent verification). A stop outcome may not come with a report in
    which every obligation holds. Insufficient population or support must be
    what the report shows, when one is given.
    """
    if outcome not in FIT_OUTCOMES:
        raise ContractViolation(f"outcome must be one of {list(FIT_OUTCOMES)}, got {outcome!r}")
    if scope not in ("pass1", "pass2"):
        raise ContractViolation(f"scope must be 'pass1' or 'pass2', got {scope!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ContractViolation("reason must be a non-empty string")
    if report is not None:
        if not isinstance(report, ObligationReport) or report.stage != "fit":
            raise ContractViolation("report must be a fit-time ObligationReport")
        if report.scope != scope:
            raise ContractViolation(f"report scope {report.scope!r} does not match {scope!r}")
    if outcome in _FEASIBLE_OUTCOMES:
        if report is None:
            raise ContractViolation(f"{outcome} needs the fit-time obligation report that "
                                    "verifies it")
        if not report.all_satisfied:
            raise ContractViolation(f"{outcome} is contradicted by the independent obligation "
                                    f"check: {ObligationInfeasible(report)}")
    elif report is not None and report.all_satisfied:
        raise ContractViolation(f"{outcome} is contradicted by a report in which every "
                                "obligation holds")
    if outcome == OUTCOME_INSUFFICIENT and report is not None:
        support_failed = any(r.obligation.endswith("_support_floor") and r.status != SATISFIED
                             for r in report.results)
        if report.population_status == POPULATION_OK and not support_failed:
            raise ContractViolation(f"{outcome} but the report shows an adequate population and "
                                    "support")
    s = MappingProxyType(dict(solver or {}))
    return FitOutcome(scope, outcome, _DISPOSITION[outcome], reason, report, s)
