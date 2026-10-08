# ============================================================
# GLASS ROUTER: NOVELTY ANALYZER MODULE
# ============================================================
# Set-level novelty floor (exact compact formulation + pairwise tightening)
# and post-selection union diagnostics.
# Works with EvaluatedRule objects
# ============================================================

from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import csc_matrix
from pulp import LpVariable, lpSum

from glass_pipeline.glass_router.core.rule import EvaluatedRule
from .utils import (
    get_covered_indices,
    get_covered_array,
    coverage_count,
    unique_counts,
    meets_novelty_floor,
    novelty_floor_rows,
)


class NoveltyAnalyzer:
    """
    Novelty floor and overlap metrics for rule selection.

    Novelty floor (set-level, order-independent):
        every selected rule j must keep at least ``min_novelty_ratio * |R_j|``
        rows (and at least ``min_new_rows``) that NO other selected rule covers.

    If the floor holds, every rule shows novelty >= min_novelty_ratio in the
    ordered report for ANY ordering, because a rule's unique rows are new
    whatever precedes it. The ILP enforces it EXACTLY in one solve with
      * add_floor_constraints - compact formulation over coverage atoms, and
      * add_novelty_constraints - pairwise implications (LP tightening only).
    find_floor_violations independently re-checks any returned selection.

    Uses _cached_covered_idx (precomputed by ILPRuleSelector).
    """

    def __init__(self, enable_novelty_constraints: bool = True):
        self.enable_novelty_constraints = enable_novelty_constraints

    def compute_pairwise_overlap(
        self,
        rule_i: EvaluatedRule,
        rule_j: EvaluatedRule
    ) -> Tuple[float, float, float]:
        """
        Compute overlap ratios between two rules.

        Returns:
            (overlap_ratio_i, overlap_ratio_j, jaccard) tuple
        """
        covered_i = get_covered_indices(rule_i)
        covered_j = get_covered_indices(rule_j)

        if not covered_i or not covered_j:
            return 0.0, 0.0, 0.0

        intersection = len(covered_i & covered_j)
        union = len(covered_i | covered_j)

        return (
            intersection / len(covered_i),
            intersection / len(covered_j),
            intersection / union if union > 0 else 0.0,
        )

    def add_novelty_constraints(
        self,
        prob,
        rules: List[EvaluatedRule],
        decision_vars: dict,
        min_novelty_ratio: float,
        min_new_rows: int = 1,
        n_rows: Optional[int] = None,
    ) -> int:
        """
        Add pairwise pre-cuts implied by the set-level novelty floor.

        If |R_i minus R_j| alone already fails the floor for rule i, then i can
        never be selected together with j, whatever else is selected:
            x_i + x_j <= 1
        This fires when EITHER rule is (almost) contained in the other. (The
        previous condition required BOTH overlaps > 1 - ratio, i.e. only
        near-identical, near-equal-size pairs, so a subset rule was never
        constrained.) These cuts are implied by add_floor_constraints and are
        kept only because they tighten the LP relaxation.

        Returns:
            Number of constraints added
        """
        if not self.enable_novelty_constraints:
            print("  ⚠️  Novelty constraints DISABLED")
            return 0

        n = len(rules)
        if n < 2:
            return 0
        if n_rows is None:
            n_rows = 1 + max((max(get_covered_indices(r), default=-1) for r in rules), default=-1)

        arrays = [get_covered_array(r, n_rows) for r in rules]
        sizes = np.array([a.size for a in arrays], dtype=np.int64)
        rows = np.concatenate(arrays)
        cols = np.concatenate([np.full(a.size, k, dtype=np.int64) for k, a in enumerate(arrays)])
        m = csc_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n_rows, n))
        inter = (m.T @ m).toarray()                         # |R_i ∩ R_j|
        rest = sizes[:, None] - inter                       # |R_i minus R_j|
        need = np.maximum(min_novelty_ratio * sizes, min_new_rows)[:, None]
        fails = rest < need - 1e-9                          # rule i fails the floor if j is present
        np.fill_diagonal(fails, False)
        pairs = np.argwhere(np.triu(fails | fails.T, k=1))

        print(f"\n  📊 Novelty floor (set-level): every selected rule keeps "
              f">= {min_novelty_ratio:.0%} of its rows (and >= {min_new_rows}) uniquely")
        print(f"     Pairwise pre-cuts: forbid (i, j) if EITHER rule leaves the other below the floor")

        for i, j in pairs:
            ri, rj = rules[i].rule_id, rules[j].rule_id
            prob += decision_vars[ri] + decision_vars[rj] <= 1

        print(f"     Pairs evaluated: {n * (n - 1) // 2}")
        print(f"     Constraints added: {len(pairs)}")
        if len(pairs):
            print(f"\n     Sample pre-cut pairs (first 10):")
            for i, j in pairs[:10]:
                print(f"       Rule {rules[i].rule_id} ↔ Rule {rules[j].rule_id}: "
                      f"overlap_i={inter[i, j] / max(sizes[i], 1):.1%}, "
                      f"overlap_j={inter[i, j] / max(sizes[j], 1):.1%}")
        return len(pairs)

    def find_floor_violations(
        self,
        selected: List[EvaluatedRule],
        n_rows: int,
        min_ratio: float,
        min_new: int,
    ) -> List[Tuple[int, List[int]]]:
        """
        Selected rules that fail the set-level floor, each with a small set of
        co-selected "coverers" T that alone already push it below the floor.
        Independent post-solve check of the floor (used by the selector's
        hard-constraint validation of every ILP / fallback selection).

        Returns:
            [(rule_id, [coverer_rule_ids]), ...]
        """
        arrays = [get_covered_array(r, n_rows) for r in selected]
        cnt = coverage_count(arrays, n_rows)
        out = []
        for j, arr in enumerate(arrays):
            size = arr.size
            if meets_novelty_floor(int((cnt[arr] == 1).sum()), size, min_ratio, min_new):
                continue
            in_j = np.zeros(n_rows, dtype=bool)
            in_j[arr] = True
            overlaps = [(k, int(in_j[arrays[k]].sum())) for k in range(len(arrays)) if k != j]
            coverers = [k for k, o in sorted(overlaps, key=lambda t: t[1]) if o > 0]
            for k in list(coverers):  # drop coverers not needed for the violation
                trial = [t for t in coverers if t != k]
                covered = np.zeros(n_rows, dtype=bool)
                for t in trial:
                    covered[arrays[t]] = True
                if not meets_novelty_floor(int((~covered[arr]).sum()), size, min_ratio, min_new):
                    coverers = trial
            out.append((selected[j].rule_id, [selected[k].rule_id for k in coverers]))
        return out

    def add_floor_constraints(
        self,
        prob,
        rules: List[EvaluatedRule],
        decision_vars: dict,
        arrays: dict,
        atoms,
        min_ratio: float,
        min_new: int,
        max_rules: int,
    ) -> Tuple[int, int]:
        """
        Exact set-level novelty floor in a single model.

        Atoms a (utils.membership_atoms over ALL covered rows) with n_a rows and
        covering set S_a. c_a = sum_{k in S_a} x_k selected rules cover atom a.
        Binary m_a = 1 when atom a is covered by >= 2 selected rules:
            c_a - 1 <= (min(|S_a|, max_rules) - 1) * m_a
        (at most max_rules rules are selected, so c_a - 1 never exceeds that).
        Unique rows of rule j = |R_j| - sum_{a covers j, |S_a| >= 2} n_a * m_a,
        and the floor is
            |R_j| - sum_{a} n_a * m_a  >=  f_j * x_j,
            f_j = max(min_new, ceil(min_ratio * |R_j|)).
        The linking constraint only bounds m_a from below, and a larger m_a only
        tightens the floor constraints, so a feasible x always admits
        m_a = [c_a >= 2]; the feasible x are exactly the floor-feasible sets.
        For x_j = 0 the constraint is always slack (sum_a n_a m_a <= |R_j|).

        Returns:
            (n_binary_atom_vars, n_floor_constraints)
        """
        members, atom_of_row = atoms
        covered = atom_of_row >= 0
        counts = np.bincount(atom_of_row[covered], minlength=len(members))

        multi = {}
        rule_atoms = defaultdict(list)
        for a, member_ids in enumerate(members):
            big_m = min(len(member_ids), max_rules) - 1
            if big_m <= 0:
                continue
            m_a = LpVariable(f"m_{a}", cat="Binary")
            prob += (lpSum(decision_vars[rid] for rid in member_ids) - 1 <= big_m * m_a,
                     f"multi_cover_{a}")
            multi[a] = m_a
            for rid in member_ids:
                rule_atoms[rid].append(a)

        for k, rule in enumerate(rules):
            rid = rule.rule_id
            size = int(arrays[rid].size)
            need = novelty_floor_rows(size, min_ratio, min_new)
            shared = lpSum(int(counts[a]) * multi[a] for a in rule_atoms[rid])
            prob += size - shared >= need * decision_vars[rid], f"novelty_floor_{k}"

        print(f"     Exact floor: {len(multi)} multi-cover atom binaries, {len(rules)} floor constraints")
        return len(multi), len(rules)

    def analyze_selection_novelty(
        self,
        selected_rules: List[EvaluatedRule],
        pass_name: str,
        y=None,
    ) -> Optional[dict]:
        """
        Post-selection diagnostics.

        novelty  = ordered marginal share (depends on list order)
        unique   = rows no other selected rule covers (order-independent;
                   0 => the rule is redundant in this set)
        Union stats count each row once (not summed per rule).
        """
        if len(selected_rules) < 2:
            return None

        if y is not None:
            n_rows = len(y)
        else:
            n_rows = 1 + max(max(get_covered_indices(r), default=-1) for r in selected_rules)
        arrays = [get_covered_array(r, n_rows) for r in selected_rules]
        uniq = unique_counts(arrays, n_rows)

        print(f"\n📈 {pass_name} Selection Novelty Analysis:")

        all_covered = set()
        novelty_ratios = []

        for i, rule in enumerate(selected_rules):
            rule_covered = get_covered_indices(rule)
            novelty = 1.0 if i == 0 else (
                len(rule_covered - all_covered) / len(rule_covered) if rule_covered else 0.0
            )
            novelty_ratios.append(novelty)
            all_covered |= rule_covered
            size = len(rule_covered)

            print(f"   Rule {i+1} (id={rule.rule_id}): "
                  f"covers {size}, "
                  f"novelty={novelty:.1%}, "
                  f"cumulative={len(all_covered)}, "
                  f"unique={uniq[i]} ({uniq[i] / size if size else 0.0:.1%})")

        avg_novelty = sum(novelty_ratios[1:]) / len(novelty_ratios[1:]) if len(novelty_ratios) > 1 else 1.0
        print(f"   Average novelty (excluding first rule): {avg_novelty:.1%}")

        raw = int(sum(a.size for a in arrays))
        union = len(all_covered)
        redundant = [selected_rules[i].rule_id for i, u in enumerate(uniq) if u == 0]
        min_unique_share = min(u / a.size for u, a in zip(uniq, arrays) if a.size) if raw else 0.0
        print(f"   Union coverage: {union} rows | sum of raw coverage: {raw} | "
              f"double-counted: {raw - union} ({(raw - union) / raw if raw else 0.0:.1%} of raw)")
        print(f"   Min unique share: {min_unique_share:.1%} | "
              f"rules with zero unique contribution: {len(redundant)} {redundant if redundant else ''}")

        stats = {
            "union_rows": union,
            "raw_rows": raw,
            "double_counted_rows": raw - union,
            "unique_rows": dict(zip([r.rule_id for r in selected_rules], uniq)),
            "novelty": dict(zip([r.rule_id for r in selected_rules], novelty_ratios)),
            "redundant_rule_ids": redundant,
        }

        if y is not None:
            y_arr = np.asarray(y)
            mask = np.zeros(n_rows, dtype=bool)
            for a in arrays:
                mask[a] = True
            union_pos = int((y_arr[mask] == 1).sum())
            summed_pos = int(sum((y_arr[a] == 1).sum() for a in arrays))
            total_pos = int((y_arr == 1).sum())
            print(f"   Union labels: positives(y=1)={union_pos} "
                  f"({union_pos / total_pos if total_pos else 0.0:.1%} of all positives), "
                  f"negatives={union - union_pos}, positive share={union_pos / union if union else 0.0:.1%} "
                  f"| per-rule positives summed={summed_pos}")
            stats.update({
                "union_positives": union_pos,
                "union_negatives": union - union_pos,
                "summed_rule_positives": summed_pos,
                "total_positives": total_pos,
            })
        return stats
