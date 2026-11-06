from fractions import Fraction
from typing import NamedTuple, Optional

import numpy as np
from scipy.sparse import csr_matrix

from glass_pipeline.glass_router.core.rule import EvaluatedRule


def get_covered_indices(rule):
    idx = getattr(rule, '_cached_covered_idx', None)
    if idx is None:
        raise ValueError(f"covered indices not precomputed for rule {rule.rule_id}")
    return idx


SCORING_MODES = ("precision_first", "recall_first")


def validate_scoring_mode(scoring_mode: str) -> str:
    """Reject anything other than the two supported scoring modes."""
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring_mode {scoring_mode!r}; expected one of {SCORING_MODES}")
    return scoring_mode


def compute_rule_quality(rule: EvaluatedRule, scoring_mode: str) -> float:
    """
    Compute base quality score for a rule.

    precision_first: (precision^3) * coverage
    recall_first:    (recall^3) * precision * coverage

    NOTE: this is a STANDALONE per-rule score. Summing it over a rule set
    counts overlapping rows once per rule, so it must never be used as the
    set-level coverage objective (see ILPBuilder.add_union_coverage).
    """
    validate_scoring_mode(scoring_mode)
    if scoring_mode == "precision_first":
        return (rule.precision ** 3) * rule.coverage
    return (rule.recall ** 3) * rule.precision * rule.coverage


def get_base_features(rule: EvaluatedRule, validator) -> set:
    """Set of base features used by a rule (deduplicated)."""
    return {validator.extract_base_feature(f) for f, _ in rule.segment}


# ============================================================
# SET-LEVEL (UNION) COVERAGE HELPERS
# ============================================================
# Terminology used throughout the selector:
#   raw coverage      |R_j|                   rows matched by one rule
#   marginal coverage |R_j minus U_prefix|    rows a rule adds after an ordered prefix
#                                             (order-dependent; the printed "novelty")
#   unique coverage   |R_j minus U_{S-j}|     rows ONLY rule j covers within set S
#                                             (order-independent; 0 => redundant rule)
#   union coverage    |U_S|                   rows routed by the selected set S
# ============================================================

def get_covered_array(rule, n_rows: int) -> np.ndarray:
    """Sorted int64 array of the rule's covered POSITIONAL row indices."""
    idx = get_covered_indices(rule)
    arr = np.fromiter(idx, dtype=np.int64, count=len(idx))
    arr.sort()
    if arr.size and (arr[0] < 0 or arr[-1] >= n_rows):
        raise ValueError(
            f"Rule {rule.rule_id}: covered indices must be positional row indices "
            f"in [0, {n_rows}); got range [{arr[0]}, {arr[-1]}]."
        )
    return arr


def union_row_weights(y, scoring_mode: str) -> np.ndarray:
    """
    Per-row weight of the union-coverage objective.

    precision_first (Pass 1): every routed row counts   -> union coverage
        (the selector contract: maximise the rows the selected set routes).
    recall_first    (Pass 2): only positive rows count  -> union recall.

    Why positives are the faithful union analogue for Pass 2. For a rule with
    n_j matched rows of which TP_j are positive (P positives, N rows in all):
        recall^3 * precision * coverage
          = (TP_j / P)^3 * (TP_j / n_j) * (n_j / N)
          = TP_j^4 / (P^3 * N)
    precision and coverage cancel, so the standalone score is a strictly
    increasing function of TP_j alone. In pass2_population="oof_remainder" the
    pipeline rescales recall and coverage by constants and precision is measured
    on the remainder, which only multiplies the same TP_j^4 by a constant. The
    set-level version of "true positives captured" counts each positive once:
    sum over positive rows of [row covered by >= 1 selected rule].

    (Pass 1 does not collapse this way: precision^3 * coverage =
    TN_j^3 / (n_j^2 * N) still depends on precision. Its union objective follows
    the stated contract - maximise union coverage - with precision enforced by
    the per-rule quality gates.)
    """
    validate_scoring_mode(scoring_mode)
    y_arr = np.asarray(y)
    if scoring_mode == "precision_first":
        return np.ones(len(y_arr), dtype=np.int64)
    return (y_arr == 1).astype(np.int64)


def membership_atoms(rule_ids, arrays, n_rows: int):
    """
    Group covered rows by the exact set of rules that cover them.

    Rows in the same atom are indistinguishable to every set-level quantity
    (union coverage, unique coverage, overlap), so the ILP needs one variable
    per atom instead of one per row. Exact, not an approximation.

    Returns:
        members:     list of tuples of rule_ids, one per atom
        atom_of_row: int array (n_rows,), atom index of each row, -1 if uncovered
    """
    atom_of_row = np.full(n_rows, -1, dtype=np.int64)
    parts = [arrays[rid] for rid in rule_ids]
    if not parts or sum(p.size for p in parts) == 0:
        return [], atom_of_row
    rows = np.concatenate(parts)
    cols = np.concatenate([np.full(p.size, k, dtype=np.int64) for k, p in enumerate(parts)])
    m = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_rows, len(parts)))
    m.sort_indices()
    indptr, indices = m.indptr, m.indices
    key_to_atom, members = {}, []
    for i in np.flatnonzero(np.diff(indptr)):
        key = indices[indptr[i]:indptr[i + 1]].tobytes()
        a = key_to_atom.get(key)
        if a is None:
            a = len(members)
            key_to_atom[key] = a
            members.append(tuple(rule_ids[int(k)] for k in np.frombuffer(key, dtype=indices.dtype)))
        atom_of_row[i] = a
    return members, atom_of_row


def coverage_count(arrays, n_rows: int) -> np.ndarray:
    """Number of rules (from `arrays`) covering each row."""
    cnt = np.zeros(n_rows, dtype=np.int32)
    for arr in arrays:
        cnt[arr] += 1  # entries within one rule are unique, so fancy += is exact
    return cnt


def unique_counts(arrays, n_rows: int) -> list:
    """Unique (leave-one-out) coverage of each rule within the given set."""
    cnt = coverage_count(arrays, n_rows)
    return [int((cnt[arr] == 1).sum()) for arr in arrays]


def novelty_floor_rows(size: int, min_ratio: float, min_new: int) -> int:
    """Minimum number of UNIQUE rows a rule of `size` rows must keep."""
    return max(int(min_new), int(np.ceil(min_ratio * size - 1e-9)))


def meets_novelty_floor(unique: int, size: int, min_ratio: float, min_new: int) -> bool:
    """Set-level novelty floor: a rule must keep >= min_ratio of its rows (and
    >= min_new rows) that no other selected rule covers."""
    return unique >= novelty_floor_rows(size, min_ratio, min_new)


# ============================================================
# SET-LEVEL (UNION) OBLIGATIONS (PR 37 Commit 4, Step 4B)
# ============================================================

def exact_fraction(value) -> Fraction:
    """The decimal a limit is written as, exactly: 0.95 -> 19/20, 0.30 -> 3/10.

    Same reading as ``shared.stage2.obligations.exact_limit`` (a test pins the
    equivalence), so the ILP and the arm-neutral checker compare the same numbers.
    """
    return Fraction(repr(float(value)))


class UnionObligations(NamedTuple):
    """
    Set-level obligations on one pass's selected union, on its selection rows.

    good       rows that count toward the ratio's numerator: negatives for the
               Pass 1 NPV floor, positives for the Pass 2 precision floor
    floor      exact ratio floor (good rows / union rows), or None
    support    minimum union rows (>= 1: an empty union is never feasible)
    ratio_name "NPV" or "precision", for messages and reports
    """

    good: np.ndarray
    floor: Optional[Fraction]
    support: int
    ratio_name: str


def union_obligation_violations(covered: np.ndarray, spec: UnionObligations) -> list:
    """Exact integer check of ``spec`` on a covered-row mask (the selected union)."""
    rows = int(covered.sum())
    good = int(spec.good[covered].sum())
    problems = []
    if rows < spec.support:
        problems.append(f"union rows {rows} below the support floor {spec.support}")
    if spec.floor is not None:
        if rows == 0:
            problems.append(f"empty union: its {spec.ratio_name} is undefined")
        elif good * spec.floor.denominator < spec.floor.numerator * rows:
            problems.append(f"union {spec.ratio_name} {good}/{rows} below the floor {spec.floor}")
    return problems
