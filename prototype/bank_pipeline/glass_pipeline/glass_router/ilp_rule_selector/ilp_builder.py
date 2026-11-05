# ============================================================
# GLASS ROUTER: ILP BUILDER MODULE
# ============================================================
# Construct the Integer Linear Programming optimization problem
# Works with EvaluatedRule objects
# ============================================================

import os
import re
import tempfile
from typing import List, Dict, Optional, Tuple

import numpy as np
from pulp import (
    LpProblem, LpVariable, LpMaximize, LpAffineExpression, lpSum, PULP_CBC_CMD, LpStatus,
)

from glass_pipeline.glass_router.core.rule import EvaluatedRule
from .utils import compute_rule_quality, membership_atoms


_CBC_NUMBER = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
_CBC_PARTIAL = re.compile(r"Partial search - best objective " + _CBC_NUMBER
                          + r" \(best possible " + _CBC_NUMBER + r"\)")
_CBC_COMPLETE = re.compile(r"Search completed - best objective " + _CBC_NUMBER)
_CBC_RESULT = re.compile(r"^Result - (.+?)\s*$", re.M)
_CBC_OBJECTIVE = re.compile(r"^Objective value:\s+" + _CBC_NUMBER, re.M)
_CBC_NO_INCUMBENT = 1e49   # CBC prints 1e+50 as the objective when it has no solution
_CBC_PROVEN = "Optimal solution found"


def parse_cbc_log(text: str, maximize: bool) -> dict:
    """
    Final incumbent objective and best bound from a CBC log (PR 37 Commit 4, Step 4B).

    PuLP's CBC interface returns neither. CBC 2.10 prints, in order of use:

    - "Partial search - best objective A (best possible B)" when a search
      stops early (bound B);
    - "Search completed - best objective A" when it proves optimality;
    - otherwise "Result - Optimal solution found" and "Objective value: A"
      (e.g. a model solved in preprocessing).

    In the first two lines CBC prints its internal minimisation, so a
    maximisation's values are negated back; "Objective value" is already in
    the model's sense. A proven optimum's bound is its objective (bound_basis
    "proven_optimal"); a search line's bound is CBC's best possible
    ("search"). A value the log does not state stays None: nothing is
    estimated. gap_abs = |bound - objective|; gap_rel = gap_abs / |objective|.
    CBC prints the "Partial search" values with %g (six significant digits),
    so their gap carries that precision; the other lines print more digits.
    """
    text = text or ""
    sign = -1.0 if maximize else 1.0
    out = {"source": "cbc_log", "result": None, "objective": None, "bound": None,
           "bound_basis": None, "gap_abs": None, "gap_rel": None}
    result = _CBC_RESULT.findall(text)
    out["result"] = result[-1] if result else None
    proven = out["result"] == _CBC_PROVEN
    partial = _CBC_PARTIAL.findall(text)
    complete = _CBC_COMPLETE.findall(text)
    objective = None
    if partial:
        internal_obj, internal_bound = (float(v) for v in partial[-1])
        out["bound"], out["bound_basis"] = sign * internal_bound, "search"
        objective = None if abs(internal_obj) >= _CBC_NO_INCUMBENT else sign * internal_obj
    elif complete and abs(float(complete[-1])) < _CBC_NO_INCUMBENT:
        objective = sign * float(complete[-1])
    elif proven and _CBC_OBJECTIVE.findall(text):
        objective = float(_CBC_OBJECTIVE.findall(text)[-1])
    if objective is None:
        return out
    out["objective"] = objective
    if out["bound"] is None and (proven or complete):
        out["bound"], out["bound_basis"] = objective, "proven_optimal"
    if out["bound"] is not None:
        out["gap_abs"] = abs(out["bound"] - objective)
        out["gap_rel"] = None if objective == 0 else out["gap_abs"] / abs(objective)
    return out


class ILPBuilder:
    """Build and solve Integer Linear Programming problems for rule selection."""

    def __init__(
        self,
        lambda_rf_uncertainty: float,
        lambda_rf_misalignment: float,
    ):
        self.lambda_rf_uncertainty = lambda_rf_uncertainty
        self.lambda_rf_misalignment = lambda_rf_misalignment
        self.last_sol_status = None
        # PR 37 Commit 4, Step 4B: when True, solve() also captures CBC's log
        # (PuLP logPath: the CBC command line is unchanged) and records its
        # final objective and best bound in last_solver_log. Off by default.
        self.record_solver_log = False
        self.last_solver_log: Optional[dict] = None

    def adjusted_quality(self, rule: EvaluatedRule, scoring_mode: str) -> float:
        """Standalone per-rule quality minus RF penalties (secondary criterion only)."""
        return (
            compute_rule_quality(rule, scoring_mode)
            - self.lambda_rf_uncertainty * (1.0 - rule.rf_confidence)
            - self.lambda_rf_misalignment * (1.0 - rule.rf_alignment)
        )

    def build_objective(
        self,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        scoring_mode: str
    ) -> List:
        """
        Per-rule quality terms  sum_j q_j * x_j.

        This is NOT a coverage objective: overlapping rows are credited once
        per rule. Not used by the current selection path:
        ILPRuleSelector._optimize_pass builds its Phase 2 quality tie-break
        inline from adjusted_quality. Kept for compatibility.
        """
        return [
            decision_vars[rule.rule_id] * self.adjusted_quality(rule, scoring_mode)
            for rule in rules
        ]

    def add_union_coverage(
        self,
        prob: LpProblem,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        arrays: Dict[int, np.ndarray],
        row_weight: np.ndarray,
        atoms=None,
    ) -> Tuple[LpAffineExpression, Dict[int, int], int, List[Tuple[LpVariable, List[int]]]]:
        """
        Add union-coverage variables and return the union-coverage expression.

        z_a = 1 iff atom a (rows with one membership pattern) is covered by at
        least one selected rule:
            z_a <= sum_{j covers a} x_j,   0 <= z_a <= 1
        Maximising sum_a w_a z_a counts each covered row ONCE, however many
        selected rules match it (w_a = summed row weight of the atom).
        Atoms matched by a single rule need no z variable (z = x_j at the
        optimum), so they contribute w_a * x_j directly.

        Args:
            atoms: (members, atom_of_row) from utils.membership_atoms; computed
                   here when not supplied.

        Returns:
            (coverage_expr, raw_weight_by_rule_id, n_weighted_atoms, z_links)
            z_links: [(z_var, [rule_ids covering that atom]), ...]
        """
        n_rows = len(row_weight)
        ids = [r.rule_id for r in rules]
        raw_weight = {rid: int(row_weight[arrays[rid]].sum()) for rid in ids}
        members, atom_of_row = atoms if atoms is not None else membership_atoms(ids, arrays, n_rows)
        if not members:
            return lpSum([]), raw_weight, 0, []

        covered = atom_of_row >= 0
        weight = np.bincount(
            atom_of_row[covered], weights=row_weight[covered], minlength=len(members)
        ).round().astype(np.int64)

        terms = []
        z_links = []
        n_weighted = 0
        for a, member_ids in enumerate(members):
            w = int(weight[a])
            if w <= 0:
                continue
            n_weighted += 1
            if len(member_ids) == 1:
                terms.append(w * decision_vars[member_ids[0]])
                continue
            z = LpVariable(f"z_{len(z_links)}", lowBound=0, upBound=1)
            prob += z <= lpSum(decision_vars[rid] for rid in member_ids), f"link_z_{len(z_links)}"
            terms.append(w * z)
            z_links.append((z, list(member_ids)))

        return lpSum(terms), raw_weight, n_weighted, z_links

    def add_union_leakage_cap(
        self,
        prob: LpProblem,
        decision_vars: Dict[int, LpVariable],
        atoms,
        pos_mask: np.ndarray,
        cap: int,
    ) -> int:
        """
        Cap the positives (subscribers) routed by the selected UNION:
            sum_a p_a * u_a <= cap,   u_a >= x_j for every rule j covering atom a
        p_a = positives in atom a. The lower-bound links force u_a = 1 whenever
        any selected rule covers the atom, so every covered positive is counted
        exactly once (the coverage variables z_a are only upper-bounded and
        could not be used here). Single-rule atoms use p_a * x_j directly.

        Returns:
            number of atoms carrying positives
        """
        members, atom_of_row = atoms
        covered = atom_of_row >= 0
        pos_counts = np.bincount(
            atom_of_row[covered], weights=pos_mask[covered].astype(float), minlength=len(members)
        ).round().astype(np.int64)
        terms = []
        n_pos_atoms = 0
        for a, member_ids in enumerate(members):
            p_a = int(pos_counts[a])
            if p_a == 0:
                continue
            n_pos_atoms += 1
            if len(member_ids) == 1:
                terms.append(p_a * decision_vars[member_ids[0]])
                continue
            u = LpVariable(f"leak_{a}", lowBound=0, upBound=1)
            for k, rid in enumerate(member_ids):
                prob += u >= decision_vars[rid], f"leak_link_{a}_{k}"
            terms.append(p_a * u)
        prob += lpSum(terms) <= int(cap), "union_leakage_cap"
        return n_pos_atoms

    def add_union_set_obligations(
        self,
        prob: LpProblem,
        decision_vars: Dict[int, LpVariable],
        atoms,
        good_mask: np.ndarray,
        floor,
        min_rows: int,
    ) -> int:
        """
        Set-level obligations on the selected UNION (PR 37 Commit 4, Step 4B).

        Exact membership indicator for every covered atom a, including atoms
        with no positives (they are false positives under a precision floor):

            y_a >= x_j  for every candidate j covering a   (covered -> counted)
            y_a <= sum_{j covers a} x_j                   (uncovered -> not counted)

        so y_a = [atom a in the selected union] for binary x, whatever the sign
        of its coefficient. Atoms covered by one candidate use y_a = x_j.

            ratio floor n/d:  sum_a (d * g_a - n * r_a) * y_a >= 0
            support floor S:  sum_a r_a * y_a >= S

        r_a = rows of atom a, g_a = its "good" rows (negatives for the Pass 1
        NPV floor, positives for the Pass 2 precision floor). All coefficients
        are integers. Returns the number of indicator variables added.
        """
        members, atom_of_row = atoms
        covered = atom_of_row >= 0
        rows = np.bincount(atom_of_row[covered], minlength=len(members)).astype(np.int64)
        good = np.bincount(atom_of_row[covered], weights=good_mask[covered].astype(float),
                           minlength=len(members)).round().astype(np.int64)
        indicator = {}
        n_vars = 0
        for a, member_ids in enumerate(members):
            if len(member_ids) == 1:
                indicator[a] = decision_vars[member_ids[0]]
                continue
            y = LpVariable(f"in_{a}", lowBound=0, upBound=1)
            for k, rid in enumerate(member_ids):
                prob += y >= decision_vars[rid], f"in_lo_{a}_{k}"
            prob += y <= lpSum(decision_vars[rid] for rid in member_ids), f"in_hi_{a}"
            indicator[a] = y
            n_vars += 1
        if floor is not None:
            num, den = int(floor.numerator), int(floor.denominator)
            terms = [int(den * good[a] - num * rows[a]) * indicator[a]
                     for a in range(len(members)) if den * good[a] != num * rows[a]]
            prob += lpSum(terms) >= 0, "union_ratio_floor"
        prob += lpSum(int(rows[a]) * indicator[a] for a in range(len(members))) >= int(min_rows), \
            "union_support_floor"
        return n_vars

    def add_cardinality_constraints(
        self,
        prob: LpProblem,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable],
        min_rules: int,
        max_rules: int
    ):
        """
        Add constraints on the number of rules selected.

        Args:
            prob: PuLP problem object
            rules: List of EvaluatedRule objects
            decision_vars: Dict mapping rule_id to LpVariable
            min_rules: Minimum number of rules to select
            max_rules: Maximum number of rules to select
        """
        total_vars = lpSum(decision_vars[r.rule_id] for r in rules)
        prob += total_vars >= min_rules, "min_rules"
        prob += total_vars <= max_rules, "max_rules"

    def solve(
        self,
        prob: LpProblem,
        time_limit: int = 300,
        warm_start: bool = False,
        gap_rel: float = None,
        gap_abs: float = None,
    ) -> str:
        """
        Solve the ILP problem.

        Args:
            prob: PuLP problem object
            time_limit: Time limit in seconds
            warm_start: pass current variable values to CBC as a MIP start
            gap_rel: stop once the solution is proven within this relative gap
            gap_abs: stop once the solution is proven within this absolute gap

        Returns:
            Status string ("Optimal", "Infeasible", etc.). PuLP can report
            "Optimal" for a run stopped by the time limit (even with an unusable
            incumbent); self.last_sol_status records PuLP's solution status
            (1 = proven optimal, 2 = feasible incumbent only, 0 = none,
            -1 = infeasible). Callers classify with both values and validate
            any returned solution. With record_solver_log, last_solver_log
            holds parse_cbc_log's reading of CBC's log for this solve (all
            None when the log has no such line or PuLP lacks logPath).
        """
        kwargs = dict(msg=0, timeLimit=max(1, int(time_limit)), warmStart=warm_start,
                      gapRel=gap_rel, gapAbs=gap_abs)
        self.last_solver_log = None
        if not self.record_solver_log:
            prob.solve(PULP_CBC_CMD(**kwargs))
            self.last_sol_status = getattr(prob, "sol_status", None)
            return LpStatus[prob.status]
        fd, log_path = tempfile.mkstemp(prefix="glass_cbc_", suffix=".log")
        os.close(fd)
        try:
            try:
                solver = PULP_CBC_CMD(**kwargs, logPath=log_path)
            except TypeError:      # a PuLP without logPath: record the absence
                solver = None
            if solver is None:
                prob.solve(PULP_CBC_CMD(**kwargs))
                self.last_solver_log = {"source": None, "result": None, "objective": None,
                                        "bound": None, "bound_basis": None, "gap_abs": None,
                                        "gap_rel": None}
            else:
                prob.solve(solver)
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    self.last_solver_log = parse_cbc_log(fh.read(), prob.sense == LpMaximize)
        finally:
            try:
                os.remove(log_path)
            except OSError:
                pass
        self.last_sol_status = getattr(prob, "sol_status", None)
        return LpStatus[prob.status]

    def extract_selected_rules(
        self,
        rules: List[EvaluatedRule],
        decision_vars: Dict[int, LpVariable]
    ) -> List[EvaluatedRule]:
        """
        Extract rules that were selected in the solution.

        Args:
            rules: List of EvaluatedRule objects
            decision_vars: Dict mapping rule_id to LpVariable

        Returns:
            List of selected EvaluatedRule objects
        """
        return [
            r for r in rules
            if decision_vars[r.rule_id].varValue is not None
            and decision_vars[r.rule_id].varValue > 0.5
        ]

    def create_problem(self, pass_name: str) -> LpProblem:
        problem_name = f"GLASS_ROUTER_{pass_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        return LpProblem(problem_name, LpMaximize)

    def create_decision_variables(self, rules: List[EvaluatedRule]) -> Dict[int, LpVariable]:
        return {
            r.rule_id: LpVariable(f"x_{r.rule_id}", cat="Binary")
            for r in rules
        }
