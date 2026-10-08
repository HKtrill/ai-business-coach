# ============================================================
# GLASS ROUTER: GREEDY SELECTOR MODULE
# ============================================================
# Greedy fallback selection when ILP fails
# Works with EvaluatedRule objects
# ============================================================

from collections import Counter
from typing import List, Optional, Tuple
import numpy as np

from glass_pipeline.glass_router.core.rule import EvaluatedRule
from .utils import (
    get_covered_array,
    compute_rule_quality,
    meets_novelty_floor,
    validate_scoring_mode,
)


class GreedySelector:
    """
    Greedy rule selection under the SAME contract as the ILP.

    Primary criterion: marginal UNION gain (weighted by the pass's row
    weights: rows for Pass 1, positives for Pass 2), i.e. the standard greedy
    for max-coverage. Standalone quality x diversity x novelty only breaks ties.

    Hard constraints (identical to the ILP, supplied by the caller):
      * novelty floor - the ACTIVE pass-specific floor (min_ratio, min_new) or
        None when novelty constraints are disabled. Set-level and
        order-independent: a candidate is admissible only if, after adding it,
        EVERY selected rule (earlier picks included) keeps the floor's share of
        rows that no other selected rule covers;
      * base-feature reuse - at most `base_reuse_limit` selected rules per base
        feature, counted exactly like DiversityAnalyzer.add_diversity_constraints;
      * optional union leakage cap - positives covered by the selected union
        (each counted once) <= cap, when the ILP has one;
      * at most max_rules rules. min_rules cannot be guaranteed by a greedy;
        the caller validates the result and raises if it falls short.

    Compatibility: min_novelty_greedy, min_absolute_new_samples and
    greedy_hard_novelty_cutoff are still accepted but no longer used - the
    fallback must enforce the pass floor established by the ILP path.

    Fallback when ILP solver fails or times out.
    Works with EvaluatedRule objects.
    """

    def __init__(
        self,
        diversity_analyzer,
        min_novelty_greedy: float,
        greedy_novelty_weight: float,
        greedy_hard_novelty_cutoff: bool,
        min_absolute_new_samples: int,
        diversity_weight: float,
    ):
        self.diversity_analyzer = diversity_analyzer
        # Superseded by the pass-specific floor (kept for signature compatibility)
        self.min_novelty_greedy = min_novelty_greedy
        self.greedy_hard_novelty_cutoff = greedy_hard_novelty_cutoff
        self.min_absolute_new_samples = min_absolute_new_samples
        # Tie-break weights (still used)
        self.greedy_novelty_weight = greedy_novelty_weight
        self.diversity_weight = diversity_weight

    def _base_usage(self, rule: EvaluatedRule) -> Counter:
        """Base-feature occurrences of a rule, as counted by the ILP reuse constraint."""
        validator = self.diversity_analyzer.validator
        return Counter(validator.extract_base_feature(f) for f, _ in rule.segment)

    def greedy_select(
        self,
        rules: List[EvaluatedRule],
        max_rules: int,
        scoring_mode: str,
        row_weight: np.ndarray,
        *,
        floor: Optional[Tuple[float, int]],
        base_reuse_limit: int,
        min_rules: int = 0,
        verbose: bool = True,
        leakage: Optional[Tuple[np.ndarray, int]] = None,
    ) -> List[EvaluatedRule]:
        """
        Perform greedy selection by marginal union gain.

        Args:
            rules: List of EvaluatedRule objects
            max_rules: Maximum number of rules to select
            scoring_mode: "precision_first" or "recall_first"
            row_weight: per-row union weights (utils.union_row_weights)
            floor: the ACTIVE pass floor (min_ratio, min_new), or None when the
                   ILP has novelty constraints disabled
            base_reuse_limit: max selected rules per base feature (ILP limit)
            leakage: (positive-row mask, cap) when the ILP has a union leakage cap
            min_rules: only used for the log message; the caller validates
            verbose: print progress

        Returns:
            List of selected EvaluatedRule objects
        """
        validate_scoring_mode(scoring_mode)
        log = print if verbose else (lambda *a, **k: None)

        def floor_ok(unique: int, size: int) -> bool:
            return floor is None or meets_novelty_floor(unique, size, floor[0], floor[1])

        log(f"\n  ⚠️  ILP did not return a usable solution - greedy fallback "
            f"(union gain, same hard constraints as the ILP)")
        if floor is None:
            log(f"      Novelty floor: disabled (matches ILP)")
        else:
            log(f"      Novelty floor (pass-specific): >= {floor[0]:.0%} unique and >= {floor[1]} row(s)")
        log(f"      Base-feature reuse limit: {base_reuse_limit}")
        if leakage is not None:
            log(f"      Union leakage cap: <= {leakage[1]} positives")
        log(f"      Novelty weight (tie-break): {self.greedy_novelty_weight}")

        n_rows = len(row_weight)
        arrays = {r.rule_id: get_covered_array(r, n_rows) for r in rules}
        usage_of = {r.rule_id: self._base_usage(r) for r in rules}

        cnt = np.zeros(n_rows, dtype=np.int32)       # selected rules covering each row
        owner = np.full(n_rows, -1, dtype=np.int64)  # position in `selected` of the sole coverer (cnt == 1)
        unique: List[int] = []                       # unique rows of each selected rule
        base_used: Counter = Counter()
        leaked = 0                                   # positives in the selected union
        selected: List[EvaluatedRule] = []
        remaining = rules.copy()

        while len(selected) < max_rules and remaining:
            best_rule, best_key, best_info = None, None, None

            for rule in remaining:
                idx = arrays[rule.rule_id]
                if idx.size == 0:
                    continue
                if any(base_used[b] + n > base_reuse_limit for b, n in usage_of[rule.rule_id].items()):
                    continue
                c = cnt[idx]
                fresh = idx[c == 0]
                gain = int(row_weight[fresh].sum())
                if gain <= 0:
                    continue  # adds nothing to the union objective
                if leakage is not None and leaked + int(leakage[0][fresh].sum()) > leakage[1]:
                    continue  # would push the union over the leakage cap
                new_samples = int(fresh.size)
                novelty = new_samples / idx.size

                if floor is not None:
                    if not floor_ok(new_samples, idx.size):
                        continue
                    taken = owner[idx[c == 1]]  # earlier picks that would lose unique rows
                    if taken.size:
                        losses = np.bincount(taken, minlength=len(selected))
                        if any(
                            not floor_ok(unique[k] - int(losses[k]),
                                         arrays[selected[k].rule_id].size)
                            for k in np.flatnonzero(losses)
                        ):
                            continue

                quality = compute_rule_quality(rule, scoring_mode)
                diversity = self.diversity_analyzer.compute_diversity_score(rule, selected)
                novelty_factor = 1.0 + self.greedy_novelty_weight * novelty
                secondary = quality * (1 + self.diversity_weight * diversity) * novelty_factor

                key = (gain, secondary)
                if best_key is None or key > best_key:
                    best_rule, best_key, best_info = rule, key, (novelty, new_samples, gain)

            if best_rule is None:
                log(f"      No admissible rule adds union coverage - stopping at {len(selected)} rules")
                break

            idx = arrays[best_rule.rule_id]
            c = cnt[idx]
            if leakage is not None:
                leaked += int(leakage[0][idx[c == 0]].sum())
            shared = idx[c == 1]
            if shared.size:
                losses = np.bincount(owner[shared], minlength=len(selected))
                for k in np.flatnonzero(losses):
                    unique[k] -= int(losses[k])
                owner[shared] = -1
            owner[idx[c == 0]] = len(selected)
            cnt[idx] += 1
            unique.append(int((c == 0).sum()))
            base_used.update(usage_of[best_rule.rule_id])

            novelty, new_samples, gain = best_info
            log(f"      Selected rule {best_rule.rule_id}: "
                f"prec={best_rule.precision:.3f}, "
                f"recall={best_rule.recall:.3f}, "
                f"cov={best_rule.coverage:.3f}, "
                f"novelty={novelty:.1%} (+{new_samples} new, union gain={gain})")

            selected.append(best_rule)
            remaining.remove(best_rule)

        if len(selected) < min_rules:
            log(f"      ⚠️  Greedy found {len(selected)} rules (< min {min_rules}) under the "
                f"hard constraints")
        log(f"\n  Greedy selected {len(selected)} rules")
        log(f"     Total unique samples covered: {int((cnt > 0).sum())}")

        return selected
