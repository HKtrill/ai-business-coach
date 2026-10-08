# ============================================================
# GLASS ROUTER: ILP RULE SELECTOR MODULE (REFACTORED)
# ============================================================
# Main orchestrator for ILP-based rule selection using modular components
# 
# Input:  List[EvaluatedRule] from RuleEvaluator
# Output: Dict with List[SelectedRule] for each pass
# ============================================================

from typing import List, Dict, Tuple, Optional

import numpy as np
from pulp import lpSum
from pulp.constants import (
    LpSolutionOptimal, LpSolutionIntegerFeasible, LpSolutionInfeasible,
)

from .quality_gate_filter import QualityGateFilter
from .novelty_analyzer import NoveltyAnalyzer
from .diversity_analyzer import DiversityAnalyzer
from .ilp_builder import ILPBuilder
from .greedy_selector import GreedySelector
from .ilp_deduplicator import ILPDeduplicator
from .utils import (
    get_covered_array, union_row_weights, membership_atoms, validate_scoring_mode,
)
from glass_pipeline.glass_router.rule_generator.feature_validator import FeatureValidator
from glass_pipeline.glass_router.core.rule import EvaluatedRule, SelectedRule


# Solver outcome classes (PuLP status AND solution status)
PROVEN = "proven_optimal"          # status Optimal and sol_status 1
INCUMBENT = "feasible_unproven"    # sol_status 2: e.g. time limit hit with an incumbent
INFEASIBLE = "infeasible"          # proven: no solution satisfies the constraints
NO_SOLUTION = "no_solution"        # stopped without any usable solution
REJECTED = "rejected_incumbent"    # unproven incumbent that failed validation


class SelectorInfeasibleError(ValueError):
    """No selection satisfies min_rules plus the active hard constraints.

    proven=True  -> the ILP proved infeasibility.
    proven=False -> the solver returned nothing usable and the greedy fallback
                    could not build a feasible selection either.
    """

    def __init__(self, message: str, proven: bool):
        super().__init__(message)
        self.proven = proven


class ILPRuleSelector:
    """
    Selects optimal rule subsets via Integer Linear Programming.
    
    Input:  List[EvaluatedRule] from RuleEvaluator
    Output: Dict with List[SelectedRule] for pass1_rules and pass2_rules
    
    Key Features:
    - Required tuning parameters must be passed explicitly (missing ones raise);
      optional parameters have defaults (lambda_rf_* None -> 0.15 / 0.08,
      max_feature_usage = 40, the greedy fallback settings, rule_prefixes)
    - Precision / leakage: per-rule candidate gates (QualityGateFilter), exactly
      as before. The gates are not selected-set constraints; the only set-level
      label constraint is the optional union leakage cap below.
    - Lexicographic ILP, both phases solved with gapRel = 0 (exact when CBC
      proves optimality within the time limit; otherwise reported UNPROVEN):
        Phase 1  maximize UNION coverage (each row counted once;
                 Pass 1 = rows routed, Pass 2 = positives captured)
        Phase 2  union coverage held >= the Phase 1 result (the optimum when
                 Phase 1 is proven); minimize double-counted overlap, then
                 maximize rule quality
    - Exact set-level novelty floor: every selected rule keeps >= min_novelty_ratio
      of its rows (and >= 1 row) uniquely - compact formulation, one solve
    - Greedy fallback enforces the same hard constraints and is validated
    - No feasible selection -> SelectorInfeasibleError (never fewer rules silently)
    - Optional Pass 1 union leakage cap (max_union_leakage_rate_pass1): subscribers
      routed by the selected UNION, each counted once, <= rate * all subscribers.
      Off (None) by default; the per-rule gates are unchanged either way.
    - Solve outcomes are recorded in self.selection_report_[pass_name]
    - Modular architecture with reusable components
    """

    # CBC time limit per solve (the original policy)
    TIME_LIMIT_PER_SOLVE = 300
    
    def __init__(
        self,
        # ============================================================
        # PASS 1 CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_pass1_rules: Optional[int] = None,
        max_pass1_rules: Optional[int] = None,
        min_precision_pass1: Optional[float] = None,
        max_precision_pass1: Optional[float] = None,
        max_subscriber_leakage_rate_pass1: Optional[float] = None,
        max_subscriber_leakage_absolute_pass1: Optional[int] = None,
        max_base_reuse_pass1: Optional[int] = None,
        
        # ============================================================
        # PASS 2 CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_pass2_rules: Optional[int] = None,
        max_pass2_rules: Optional[int] = None,
        min_precision_pass2: Optional[float] = None,
        max_precision_pass2: Optional[float] = None,
        min_recall_pass2: Optional[float] = None,
        max_recall_pass2: Optional[float] = None,
        max_base_reuse_pass2: Optional[int] = None,
        # Optional selected-set cap: subscribers routed by the Pass 1 UNION
        # <= rate * all subscribers. None = off (per-rule gates only, as before).
        max_union_leakage_rate_pass1: Optional[float] = None,
        
        # ============================================================
        # NOVELTY CONSTRAINTS - REQUIRED (tuning params)
        # ============================================================
        min_novelty_ratio_pass1: Optional[float] = None,
        min_novelty_ratio_pass2: Optional[float] = None,
        enable_novelty_constraints: Optional[bool] = None,
        
        # ============================================================
        # SHARED TUNING - REQUIRED
        # ============================================================
        diversity_weight: Optional[float] = None,
        
        # ============================================================
        # STRUCTURAL/BEHAVIORAL - OPTIONAL (rarely tuned)
        # ============================================================
        max_feature_usage: int = 40,
        lambda_rf_uncertainty: Optional[float] = None,
        lambda_rf_misalignment: Optional[float] = None,
        
        # Greedy fallback controls
        min_novelty_greedy: float = 0.15,
        greedy_novelty_weight: float = 0.5,
        greedy_hard_novelty_cutoff: bool = True,
        min_absolute_new_samples: int = 30,
        
        # Feature validation
        rule_prefixes: Tuple[str, ...] = (
            'nsd',
            'jed',
            'cci',
            'eci',
            'dow',
            'behav',
            'campaign',
            'cpi',
        ),
    ):
        # ============================================================
        # VALIDATE REQUIRED PARAMETERS
        # ============================================================
        required_params = {
            'min_pass1_rules': min_pass1_rules,
            'max_pass1_rules': max_pass1_rules,
            'min_precision_pass1': min_precision_pass1,
            'max_precision_pass1': max_precision_pass1,
            'max_subscriber_leakage_rate_pass1': max_subscriber_leakage_rate_pass1,
            'max_subscriber_leakage_absolute_pass1': max_subscriber_leakage_absolute_pass1,
            'min_pass2_rules': min_pass2_rules,
            'max_pass2_rules': max_pass2_rules,
            'min_precision_pass2': min_precision_pass2,
            'max_precision_pass2': max_precision_pass2,
            'min_recall_pass2': min_recall_pass2,
            'max_recall_pass2': max_recall_pass2,
            'min_novelty_ratio_pass1': min_novelty_ratio_pass1,
            'min_novelty_ratio_pass2': min_novelty_ratio_pass2,
            'enable_novelty_constraints': enable_novelty_constraints,
            'diversity_weight': diversity_weight,
        }
        
        missing = [k for k, v in required_params.items() if v is None]
        if missing:
            raise ValueError(
                f"ILPRuleSelector missing required parameters: {missing}\n"
                f"All tuning parameters must be explicitly passed from GlassRouterConfig."
            )
        self._validate_cardinality("Pass 1", min_pass1_rules, max_pass1_rules)
        self._validate_cardinality("Pass 2", min_pass2_rules, max_pass2_rules)
        if max_union_leakage_rate_pass1 is not None and not (0.0 <= max_union_leakage_rate_pass1 <= 1.0):
            raise ValueError(
                f"max_union_leakage_rate_pass1 must be in [0, 1] or None, "
                f"got {max_union_leakage_rate_pass1!r}"
            )
        
        # ============================================================
        # STORE PASS-LEVEL PARAMETERS
        # ============================================================
        self.min_pass1_rules = min_pass1_rules
        self.max_pass1_rules = max_pass1_rules
        self.max_base_reuse_pass1 = max_base_reuse_pass1
        
        self.min_pass2_rules = min_pass2_rules
        self.max_pass2_rules = max_pass2_rules
        self.max_base_reuse_pass2 = max_base_reuse_pass2
        
        self.min_novelty_ratio_pass1 = min_novelty_ratio_pass1
        self.min_novelty_ratio_pass2 = min_novelty_ratio_pass2
        self.max_union_leakage_rate_pass1 = max_union_leakage_rate_pass1
        
        # ============================================================
        # INITIALIZE MODULAR COMPONENTS
        # ============================================================
        self.validator = FeatureValidator(rule_prefixes=rule_prefixes)
        
        self.quality_filter = QualityGateFilter(
            min_precision_pass1=min_precision_pass1,
            max_precision_pass1=max_precision_pass1,
            max_subscriber_leakage_rate_pass1=max_subscriber_leakage_rate_pass1,
            max_subscriber_leakage_absolute_pass1=max_subscriber_leakage_absolute_pass1,
            min_precision_pass2=min_precision_pass2,
            max_precision_pass2=max_precision_pass2,
            min_recall_pass2=min_recall_pass2,
            max_recall_pass2=max_recall_pass2,
        )
        
        self.novelty_analyzer = NoveltyAnalyzer(
            enable_novelty_constraints=enable_novelty_constraints
        )
        
        self.diversity_analyzer = DiversityAnalyzer(
            validator=self.validator,
            max_feature_usage=max_feature_usage
        )
        
        self.ilp_builder = ILPBuilder(
            lambda_rf_uncertainty=(
                0.15 if lambda_rf_uncertainty is None else lambda_rf_uncertainty
            ),
            lambda_rf_misalignment=(
                0.08 if lambda_rf_misalignment is None else lambda_rf_misalignment
            ),
        )
        
        self.greedy_selector = GreedySelector(
            diversity_analyzer=self.diversity_analyzer,
            min_novelty_greedy=min_novelty_greedy,
            greedy_novelty_weight=greedy_novelty_weight,
            greedy_hard_novelty_cutoff=greedy_hard_novelty_cutoff,
            min_absolute_new_samples=min_absolute_new_samples,
            diversity_weight=diversity_weight,
        )
        
        self.deduplicator = ILPDeduplicator()
        self.selection_report_: Dict[str, dict] = {}
        
        print("✅ ILPRuleSelector initialized (all tuning params validated)")
    
    # ============================================================
    # STRUCTURAL FILTERING
    # ============================================================
    
    def _filter_invalid_rules(
        self,
        candidates: List[EvaluatedRule]
    ) -> Tuple[List[EvaluatedRule], List[EvaluatedRule]]:
        valid, rejected = [], []
        for rule in candidates:
            if self.validator.has_duplicate_base_features(rule.segment):
                rejected.append(rule)
            else:
                valid.append(rule)
        return valid, rejected
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    def select_rules(
        self,
        evaluated_rules: List[EvaluatedRule],
        y_val=None,
        X_val=None,
        segment_builder=None,
        passes: Tuple[str, ...] = ("pass1", "pass2"),
    ) -> Dict[str, List[SelectedRule]]:
        """
        Public entry point for ILP rule selection.

        Args:
            evaluated_rules: List of EvaluatedRule objects from RuleEvaluator
            y_val: Validation labels (required for Pass 1 leakage calculation)
            X_val: Validation features (for computing covered indices)
            segment_builder: Segment builder (for computing covered indices)
            passes: which passes to select. A pass that is not requested returns
                    [] without running; every requested pass must satisfy its
                    configured min_rules or SelectorInfeasibleError is raised.
                    Callers that fit a single pass (e.g. Pass 1 only) pass
                    ("pass1",) instead of relying on an empty candidate list.

        Returns:
            Dict with {"pass1_rules": List[SelectedRule], "pass2_rules": List[SelectedRule]}
        """
        passes = tuple(passes)
        unknown = [p for p in passes if p not in ("pass1", "pass2")]
        if unknown or not passes:
            raise ValueError(f"passes must be a non-empty subset of ('pass1', 'pass2'); got {passes}")

        self.selection_report_ = {}

        if not evaluated_rules:
            print("⚠️  No evaluated rules provided to ILP selector.")
            valid_rules = []
        else:
            if y_val is None:
                raise ValueError(
                    "ILPRuleSelector.select_rules requires y_val: union coverage (Pass 1 rows, "
                    "Pass 2 positives) and Pass 1 leakage are defined over the evaluation rows."
                )

            print("\n" + "=" * 80)
            print("🧮 ILP RULE SELECTION")
            print("=" * 80)

            if X_val is not None:
                print("  Computing covered indices for novelty analysis...")
                self._precompute_covered_indices(evaluated_rules, X_val, segment_builder)

            self._print_configuration()

            valid_rules, rejected = self._filter_invalid_rules(evaluated_rules)
            if rejected:
                print(f"  🚫 Rejected {len(rejected)} structurally invalid rules")
            if not valid_rules:
                print("⚠️  No valid rules remain after structural filtering.")

        pass1_candidates = [r for r in valid_rules if r.predicted_class == 0]
        pass2_candidates = [r for r in valid_rules if r.predicted_class == 1]

        print(f"\nCandidate split (passes requested: {', '.join(passes)}):")
        print(f"  Pass 1 candidates: {len(pass1_candidates)}")
        print(f"  Pass 2 candidates: {len(pass2_candidates)}")

        # An empty candidate list still goes through _optimize_pass, which raises
        # when the configured min_rules > 0 (A1: never return fewer rules silently).
        pass1_evaluated = self._optimize_pass(
            candidates=pass1_candidates,
            min_rules=self.min_pass1_rules,
            max_rules=self.max_pass1_rules,
            pass_name="Pass 1 (NOT_SUBSCRIBE)",
            scoring_mode="precision_first",
            y_val=y_val,
            max_base_reuse=self.max_base_reuse_pass1,
            min_novelty_ratio=self.min_novelty_ratio_pass1,
            max_union_leakage_rate=self.max_union_leakage_rate_pass1,
        ) if "pass1" in passes else []

        pass2_evaluated = self._optimize_pass(
            candidates=pass2_candidates,
            min_rules=self.min_pass2_rules,
            max_rules=self.max_pass2_rules,
            pass_name="Pass 2 (SUBSCRIBE)",
            scoring_mode="recall_first",
            y_val=y_val,  # needed for union recall; gate choice still keys on pass_name
            max_base_reuse=self.max_base_reuse_pass2,
            min_novelty_ratio=self.min_novelty_ratio_pass2,
        ) if "pass2" in passes else []

        if "pass1" in passes:
            self.novelty_analyzer.analyze_selection_novelty(pass1_evaluated, "Pass 1", y=y_val)
        if "pass2" in passes:
            self.novelty_analyzer.analyze_selection_novelty(pass2_evaluated, "Pass 2", y=y_val)
        
        pass1_selected = [
            SelectedRule.from_evaluated(r, pass_assignment="pass1")
            for r in pass1_evaluated
        ]
        pass2_selected = [
            SelectedRule.from_evaluated(r, pass_assignment="pass2")
            for r in pass2_evaluated
        ]
        
        print("\n✅ ILP SELECTION COMPLETE")
        print(f"  Pass 1 selected: {len(pass1_selected)} rules"
              f"{'' if 'pass1' in passes else ' (not requested)'}")
        print(f"  Pass 2 selected: {len(pass2_selected)} rules"
              f"{'' if 'pass2' in passes else ' (not requested)'}")
        
        return {
            "pass1_rules": pass1_selected,
            "pass2_rules": pass2_selected,
        }
    
    def _precompute_covered_indices(self, rules, X_val, segment_builder):
        """Precompute and cache covered indices for novelty calculations."""
        for rule in rules:
            rule._cached_covered_idx = rule.compute_covered_indices(X_val, segment_builder)
    
    # ============================================================
    # PASS OPTIMIZATION
    # ============================================================
    
    def _optimize_pass(
        self,
        candidates: List[EvaluatedRule],
        min_rules: int,
        max_rules: int,
        pass_name: str,
        scoring_mode: str,
        y_val=None,
        max_base_reuse: Optional[int] = None,
        min_novelty_ratio: float = 0.20,
        max_union_leakage_rate: Optional[float] = None,
    ) -> List[EvaluatedRule]:
        """Optimize a single pass using ILP or greedy fallback."""
        validate_scoring_mode(scoring_mode)
        self._validate_cardinality(pass_name, min_rules, max_rules)

        def require_min_rules(available: int, stage: str) -> None:
            """A1: fewer candidates than the configured minimum -> no feasible set."""
            if available < min_rules:
                self.selection_report_[pass_name] = {
                    "source": "infeasible", "min_rules": min_rules,
                    "candidates": available, "stage": stage,
                }
                raise SelectorInfeasibleError(
                    f"{pass_name}: min_rules = {min_rules} but only {available} candidate "
                    f"rule(s) remain after {stage} - no selection can satisfy the configured "
                    f"minimum.\n  Lower min_rules explicitly in the notebook config.",
                    proven=True,
                )

        if not candidates:
            print(f"⚠️  No candidates for {pass_name}")
            require_min_rules(0, "candidate generation")
            self.selection_report_[pass_name] = {"source": "empty (min_rules = 0)",
                                                 "min_rules": min_rules, "n_selected": 0}
            return []
        
        print(f"\n🧪 ENTERING ILP OPTIMIZATION: {pass_name}")
        print("-" * 80)
        print(f"Total incoming candidates: {len(candidates)}")
        
        if "Pass 1" in pass_name and y_val is not None:
            valid, rejected = self.quality_filter.apply_quality_gates_pass1(candidates, y_val)
        else:
            valid, rejected = self.quality_filter.apply_quality_gates_pass2(candidates)
        
        print(f"  ✅ Passed gates: {len(valid)}/{len(candidates)}")
        print(f"  ❌ Rejected: {len(rejected)}/{len(candidates)}")
        
        if len(valid) == 0:
            print(f"  ⚠️  No valid candidates after quality gates!")
        require_min_rules(len(valid), "quality gates")
        if len(valid) == 0:
            self.selection_report_[pass_name] = {"source": "empty (min_rules = 0)",
                                                 "min_rules": min_rules, "n_selected": 0}
            return []
        
        valid = self.deduplicator.deduplicate_by_segment(valid, scoring_mode)
        print(f"  After deduplication: {len(valid)} unique rules")
        require_min_rules(len(valid), "quality gates + segment deduplication")
        
        if y_val is None:
            raise ValueError(f"{pass_name}: y_val is required for union-coverage selection")
        n_rows = len(y_val)
        row_weight = union_row_weights(y_val, scoring_mode)
        arrays = {r.rule_id: get_covered_array(r, n_rows) for r in valid}
        novelty_on = self.novelty_analyzer.enable_novelty_constraints
        # Set-level floor (ratio, min unique rows); min 1 row => never keep a zero-marginal rule
        floor = (min_novelty_ratio, 1) if novelty_on else None
        base_limit = (max_base_reuse if max_base_reuse is not None
                      else self.diversity_analyzer.max_feature_usage)
        weight_name = "rows" if scoring_mode == "precision_first" else "positives"

        # Optional union leakage cap: positives (subscribers) routed by the selected
        # set, each counted once, <= rate * all positives in y_val - the same
        # denominator the per-rule leakage gate uses.
        pos_mask = np.asarray(y_val) == 1
        total_pos = int(pos_mask.sum())
        leakage = None
        if max_union_leakage_rate is not None:
            leak_cap = int(np.floor(max_union_leakage_rate * total_pos + 1e-9))
            leakage = (pos_mask, leak_cap)
            print(f"  Union leakage cap: <= {leak_cap} of {total_pos} positives "
                  f"({max_union_leakage_rate:.1%}) in the selected union")

        # Merge rules interchangeable in EVERY constraint and objective term: same
        # covered rows AND same base-feature signature. Exact only while the floor is
        # active (it forbids selecting two such twins together), so skipped otherwise.
        n_gated = len(valid)
        if floor is not None:
            valid = self._merge_equivalent_rules(valid, arrays, scoring_mode)
            print(f"  Constraint-equivalent merge: {n_gated} -> {len(valid)} rules "
                  f"(same covered rows + same base-feature signature)")
            require_min_rules(
                len(valid),
                "the constraint-equivalent merge (identical covered rows + base-feature "
                "signature; such twins cannot be selected together under the novelty floor)",
            )

        # min_rules is the configured hard minimum (never lowered); max_rules may be
        # bounded by the candidates actually available.
        actual_max = min(max_rules, len(valid))

        # candidates_after_gates counts candidates after the quality gates AND
        # segment deduplication (key name kept for compatibility).
        report = {
            "source": None, "scoring_mode": scoring_mode, "union_weight": weight_name,
            "min_rules": min_rules, "max_rules": actual_max, "novelty_floor": floor,
            "base_reuse_limit": base_limit, "candidates_after_gates": n_gated,
            "candidates_after_merge": len(valid),
            "union_leakage_cap": None if leakage is None else leakage[1],
        }
        self.selection_report_[pass_name] = report

        def union_weight(rules):
            mask = np.zeros(n_rows, dtype=bool)
            for r in rules:
                mask[arrays[r.rule_id]] = True
            return int(row_weight[mask].sum())

        def union_positives(rules):
            mask = np.zeros(n_rows, dtype=bool)
            for r in rules:
                mask[arrays[r.rule_id]] = True
            return int(pos_mask[mask].sum())

        def violations(rules):
            return self._hard_constraint_violations(
                rules, min_rules, actual_max, max_base_reuse, floor, n_rows, leakage
            )

        def infeasible(proven, problems=None):
            return SelectorInfeasibleError(
                self._infeasible_message(pass_name, proven, min_rules, actual_max,
                                         floor, base_limit, n_gated, len(valid), problems,
                                         None if leakage is None else
                                         (max_union_leakage_rate, leakage[1], total_pos)),
                proven=proven,
            )

        atoms = membership_atoms([r.rule_id for r in valid], arrays, n_rows)
        prob = self.ilp_builder.create_problem(pass_name)
        decision_vars = self.ilp_builder.create_decision_variables(valid)

        # Union coverage: each row counted once, however many selected rules match it
        coverage_expr, raw_weight, n_weighted_atoms, z_links = self.ilp_builder.add_union_coverage(
            prob, valid, decision_vars, arrays, row_weight, atoms=atoms
        )

        self.ilp_builder.add_cardinality_constraints(
            prob, valid, decision_vars, min_rules, actual_max
        )
        self.diversity_analyzer.add_diversity_constraints(
            prob, valid, decision_vars, max_base_reuse
        )
        n_constraints = self.novelty_analyzer.add_novelty_constraints(
            prob, valid, decision_vars, min_novelty_ratio, min_new_rows=1, n_rows=n_rows
        )
        if floor is not None:
            self.novelty_analyzer.add_floor_constraints(
                prob, valid, decision_vars, arrays, atoms, floor[0], floor[1], actual_max
            )
        if leakage is not None:
            self.ilp_builder.add_union_leakage_cap(prob, decision_vars, atoms, pos_mask, leakage[1])

        # Same-contract greedy: the fallback if CBC returns nothing usable. It is NOT
        # passed to CBC as a MIP start: in testing, CBC's MIP start produced a wrong
        # "proven optimal" Phase 2 answer, which would void the lexicographic guarantee.
        incumbent = self.greedy_selector.greedy_select(
            valid, actual_max, scoring_mode, row_weight,
            floor=floor, base_reuse_limit=base_limit, min_rules=min_rules, verbose=False,
            leakage=leakage,
        )
        incumbent_problems = violations(incumbent)

        print(f"\nSolving ILP for {pass_name}...")
        print(f"  Variables: {len(decision_vars)} rules, {len(atoms[0])} coverage atoms "
              f"({n_weighted_atoms} with positive weight)")
        print(f"  Cardinality: [{min_rules}, {actual_max}]")
        print(f"  Objective: Phase 1 max union {weight_name} (exact); "
              f"Phase 2 min overlap, then quality (exact)")
        print(f"  Novelty pairwise tightening: {n_constraints}")
        print(f"  Greedy incumbent: {len(incumbent)} rules, union {union_weight(incumbent)} {weight_name}"
              f"{'' if not incumbent_problems else ' (not feasible: ' + '; '.join(incumbent_problems) + ')'}")

        # ---- Phase 1: maximize union coverage ---------------------------
        prob.setObjective(coverage_expr)
        state1, selected, info1 = self._solve_phase(
            prob, valid, decision_vars, coverage_expr, union_weight, violations, "Phase 1",
        )
        report["phase1"] = info1
        print(f"  Phase 1: {state1} (status={info1['status']}, sol_status={info1['sol_status']})")

        if state1 == INFEASIBLE:
            report["source"] = "infeasible"
            raise infeasible(proven=True)

        if state1 not in (PROVEN, INCUMBENT):
            fallback = self.greedy_selector.greedy_select(
                valid, actual_max, scoring_mode, row_weight,
                floor=floor, base_reuse_limit=base_limit, min_rules=min_rules, verbose=True,
                leakage=leakage,
            )
            problems = violations(fallback)
            if problems:
                report["source"] = "infeasible"
                raise infeasible(proven=False, problems=problems)
            report.update(source="greedy_fallback", lexicographic_guarantee="not applicable",
                          union_coverage=union_weight(fallback), n_selected=len(fallback),
                          union_positives=union_positives(fallback))
            print(f"  ✅ Selected {len(fallback)} rules via validated greedy fallback "
                  f"(NOT an ILP optimum)")
            return fallback

        if state1 == INCUMBENT:
            print(f"  ⚠️  Phase 1 result is a time-limited incumbent - NOT proven optimal")
            if not incumbent_problems and union_weight(incumbent) > union_weight(selected):
                print(f"  ⚠️  Greedy incumbent covers more than the solver incumbent - using it")
                selected = incumbent
        z_star = union_weight(selected)
        print(f"  Phase 1 union coverage: {z_star} {weight_name} with {len(selected)} rules"
              f"{'' if state1 == PROVEN else ' (unproven)'}")

        # ---- Phase 2: lock coverage, remove redundancy, then quality -----
        # overlap = sum_j raw_j x_j - union  (integer, in the same units as union)
        # quality tie-break is scaled so its total range stays < 1 overlap unit.
        prob += coverage_expr >= z_star - 0.5, "lock_union_coverage"
        q = {r.rule_id: self.ilp_builder.adjusted_quality(r, scoring_mode) for r in valid}
        q_max = max((abs(v) for v in q.values()), default=0.0)
        eps = 0.5 / (2.0 * actual_max * q_max) if q_max > 0 else 0.0
        prob.setObjective(
            coverage_expr
            - lpSum(raw_weight[rid] * decision_vars[rid] for rid in raw_weight)
            + lpSum(eps * q[rid] * decision_vars[rid] for rid in q)
        )
        state2, selected2, info2 = self._solve_phase(
            prob, valid, decision_vars, coverage_expr, union_weight, violations, "Phase 2",
            extra_check=lambda sel: ([] if union_weight(sel) >= z_star else
                                     [f"union {union_weight(sel)} below locked {z_star}"]),
        )
        report["phase2"] = info2
        print(f"  Phase 2: {state2} (status={info2['status']}, sol_status={info2['sol_status']})")
        if state2 in (PROVEN, INCUMBENT):
            selected = selected2
            if state2 == INCUMBENT:
                print(f"  ⚠️  Phase 2 result is a time-limited incumbent - NOT proven optimal")
        else:
            print(f"  ⚠️  Phase 2 returned no usable solution - keeping the Phase 1 selection")

        guarantee = "proven" if (state1 == PROVEN and state2 == PROVEN) else "UNPROVEN"
        report.update(source="ilp", lexicographic_guarantee=guarantee,
                      union_coverage=union_weight(selected), n_selected=len(selected),
                      union_positives=union_positives(selected))
        print(f"  Lexicographic guarantee: {guarantee}")
        if leakage is not None:
            print(f"  Union leakage: {union_positives(selected)} / {total_pos} positives "
                  f"(cap {leakage[1]})")
        print(f"  ✅ Selected {len(selected)} rules")

        return selected

    # ============================================================
    # SOLVE / VALIDATION HELPERS
    # ============================================================

    @staticmethod
    def _validate_cardinality(label: str, min_rules, max_rules) -> None:
        """Configured cardinality must be 0 <= min_rules <= max_rules, max_rules >= 1."""
        ok = (isinstance(min_rules, (int, np.integer)) and isinstance(max_rules, (int, np.integer))
              and not isinstance(min_rules, bool) and not isinstance(max_rules, bool)
              and 0 <= min_rules <= max_rules and max_rules >= 1)
        if not ok:
            raise ValueError(
                f"{label}: invalid rule-count configuration min_rules={min_rules!r}, "
                f"max_rules={max_rules!r}; require integers with 0 <= min_rules <= max_rules "
                f"and max_rules >= 1."
            )

    @staticmethod
    def _classify_solution(status: str, sol_status) -> str:
        """Combine PuLP status and solution status into one outcome class."""
        if status == "Optimal" and sol_status == LpSolutionOptimal:
            return PROVEN
        if sol_status == LpSolutionIntegerFeasible:
            return INCUMBENT
        if status == "Infeasible" or sol_status == LpSolutionInfeasible:
            return INFEASIBLE
        if status == "Optimal" and sol_status is None:
            return INCUMBENT  # PuLP without sol_status: cannot confirm proof -> unproven
        return NO_SOLUTION

    def _solve_phase(self, prob, valid, decision_vars, coverage_expr, union_weight, violations,
                     phase, extra_check=None):
        """
        Solve exactly (gapRel = 0), classify the outcome and validate the solution.

        Returns (state, selected_rules, info). A proven optimum that fails the
        independent validation raises RuntimeError (formulation bug); an
        unproven incumbent that fails it is rejected.
        """
        status = self.ilp_builder.solve(
            prob, time_limit=self.TIME_LIMIT_PER_SOLVE, warm_start=False, gap_rel=0.0
        )
        sol_status = self.ilp_builder.last_sol_status
        state = self._classify_solution(status, sol_status)
        info = {"status": status, "sol_status": sol_status, "state": state, "problems": []}
        if state in (INFEASIBLE, NO_SOLUTION):
            return state, [], info

        values = [v.varValue for v in decision_vars.values()]
        problems = []
        if any(v is None for v in values):
            problems.append("solver returned no value for some rule variables")
        elif any(min(abs(v), abs(1.0 - v)) > 1e-4 for v in values):
            problems.append("non-integral rule variables")
        selected = self.ilp_builder.extract_selected_rules(valid, decision_vars)
        problems += violations(selected)
        expr_value = coverage_expr.value()
        if expr_value is not None and union_weight(selected) + 0.5 < expr_value:
            problems.append(f"objective {expr_value:.0f} exceeds the true union {union_weight(selected)}")
        if extra_check is not None:
            problems += extra_check(selected)
        info["problems"] = problems

        if problems:
            if state == PROVEN:
                raise RuntimeError(
                    f"{phase}: CBC reported a proven optimum that fails independent "
                    f"validation: {problems}"
                )
            print(f"  ⚠️  {phase}: rejecting unproven solver incumbent: {'; '.join(problems)}")
            info["state"] = REJECTED
            return REJECTED, [], info
        return state, selected, info

    def _hard_constraint_violations(self, rules, min_rules, max_rules, max_base_reuse,
                                    floor=None, n_rows=None, leakage=None) -> List[str]:
        """Every hard selector constraint, checked directly on a rule list."""
        problems = []
        ids = [r.rule_id for r in rules]
        if len(set(ids)) != len(ids):
            problems.append("duplicate rules")
        if not (min_rules <= len(rules) <= max_rules):
            problems.append(f"{len(rules)} rules outside cardinality [{min_rules}, {max_rules}]")
        limit = max_base_reuse if max_base_reuse is not None else self.diversity_analyzer.max_feature_usage
        usage = self.diversity_analyzer.get_feature_usage_summary(rules)
        over = {b: c for b, c in usage.items() if c > limit}
        if over:
            problems.append(f"base-feature reuse above {limit}: {over}")
        if floor is not None and rules:
            if n_rows is None:
                raise ValueError("n_rows is required to check the novelty floor")
            bad = self.novelty_analyzer.find_floor_violations(rules, n_rows, *floor)
            if bad:
                problems.append(f"novelty floor violated by rule(s) {[rid for rid, _ in bad]}")
        if leakage is not None and rules:
            pos_mask, cap = leakage
            covered = np.zeros(len(pos_mask), dtype=bool)
            for r in rules:
                covered[get_covered_array(r, len(pos_mask))] = True
            leaked = int(pos_mask[covered].sum())
            if leaked > cap:
                problems.append(f"union leakage {leaked} positives above cap {cap}")
        return problems

    def _meets_hard_constraints(self, rules, min_rules, max_rules, max_base_reuse,
                                floor=None, n_rows=None, leakage=None) -> bool:
        """Cardinality + base-feature reuse + (when active) the novelty floor and
        the union leakage cap."""
        return not self._hard_constraint_violations(
            rules, min_rules, max_rules, max_base_reuse, floor, n_rows, leakage
        )

    def _merge_equivalent_rules(self, rules, arrays, scoring_mode):
        """
        Keep one rule per group of CONSTRAINT-EQUIVALENT rules.

        Key = (covered rows, sorted base-feature occurrences). Rules sharing the
        key have identical columns in the coverage atoms, the novelty floor, the
        pairwise tightening, the cardinality and the base-feature reuse
        constraints; only the Phase 2 quality tie-break differs, so the best
        representative dominates the others. Twins can never be selected
        together while the floor is active (each would have 0 unique rows).
        Ties keep the earliest candidate; candidate order is preserved.
        """
        validator = self.diversity_analyzer.validator
        groups: Dict[tuple, list] = {}
        for pos, r in enumerate(rules):
            key = (arrays[r.rule_id].tobytes(),
                   tuple(sorted(validator.extract_base_feature(f) for f, _ in r.segment)))
            groups.setdefault(key, []).append((pos, r))
        kept = [
            max(members, key=lambda t: (self.ilp_builder.adjusted_quality(t[1], scoring_mode), -t[0]))
            for members in groups.values()
        ]
        kept.sort(key=lambda t: t[0])
        return [r for _, r in kept]

    @staticmethod
    def _infeasible_message(pass_name, proven, min_rules, actual_max, floor,
                            base_limit, n_gated, n_merged, problems=None, leakage=None) -> str:
        how = ("the ILP PROVED that no feasible selection exists" if proven else
               "CBC returned no usable solution (stopped without an incumbent, or its "
               "incumbent failed validation) and the greedy fallback could not build "
               "a feasible selection")
        floor_txt = (f"every selected rule keeps >= {floor[0]:.0%} of its rows and >= {floor[1]} "
                     f"row(s) that no other selected rule covers" if floor else "disabled")
        lines = [
            f"{pass_name}: no rule set satisfies the hard selector constraints - {how}.",
            f"  min_rules        = {min_rules} (configured, hard); max_rules = {actual_max}",
            f"  novelty floor    = {floor_txt}",
            f"  base-feature use <= {base_limit} selected rules per base feature",
            f"  union leakage    = " + ("off" if leakage is None else
                                        f"<= {leakage[1]} of {leakage[2]} positives ({leakage[0]:.1%})"),
            f"  candidates       = {n_gated} after quality gates + segment dedup, "
            f"{n_merged} after constraint-equivalent merge",
        ]
        if problems:
            lines.append(f"  fallback result  : {'; '.join(problems)}")
        lines.append("  Lower min_rules, the novelty floor or (if set) raise the union leakage "
                     "cap explicitly in the notebook config.")
        return "\n".join(lines)

    # ============================================================
    # CONFIGURATION PRINTING
    # ============================================================
    
    def _print_configuration(self):
        """Print ILP selector configuration."""
        print(f"\n📊 Configuration:")
        print(f"   Novelty constraints enabled: {self.novelty_analyzer.enable_novelty_constraints}")
        if self.novelty_analyzer.enable_novelty_constraints:
            print(f"   Pass 1 min novelty (ILP, set-level unique share): {self.min_novelty_ratio_pass1:.0%}")
            print(f"   Pass 2 min novelty (ILP, set-level unique share): {self.min_novelty_ratio_pass2:.0%}")
        print(f"   Greedy fallback: pass-specific novelty floor + base-feature reuse (ILP parity)")
        print(f"   (superseded, unused: min_novelty_greedy={self.greedy_selector.min_novelty_greedy}, "
              f"min_absolute_new_samples={self.greedy_selector.min_absolute_new_samples}, "
              f"greedy_hard_novelty_cutoff={self.greedy_selector.greedy_hard_novelty_cutoff})")
        print(f"   ILP: exact (gapRel=0), {self.TIME_LIMIT_PER_SOLVE}s per solve")
        print(f"   RF uncertainty penalty: {self.ilp_builder.lambda_rf_uncertainty}")
        print(f"   RF misalignment penalty: {self.ilp_builder.lambda_rf_misalignment}")
        print(f"   Pass 1 coverage: [{self.quality_filter.min_coverage_pass1}, {self.quality_filter.max_coverage_pass1}]")
        print(f"   Pass 2 coverage: [{self.quality_filter.min_coverage_pass2}, {self.quality_filter.max_coverage_pass2}]")
    
    def __repr__(self):
        return (
            f"ILPRuleSelector("
            f"pass1=[{self.min_pass1_rules}-{self.max_pass1_rules}], "
            f"pass2=[{self.min_pass2_rules}-{self.max_pass2_rules}], "
            f"novelty={self.novelty_analyzer.enable_novelty_constraints})"
        )