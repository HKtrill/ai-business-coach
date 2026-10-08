# GLASS Router — ILP Rule Selector: Design and Behaviour

Status: describes the selector as implemented in `glass_router/ilp_rule_selector/`
(union-aware revision). Scope is limited to rule **selection**; see
[Scope boundaries](#18-scope-boundaries--future-documentation).

---

## Contents

1. [Overview](#1-overview)
2. [Why the previous selector was incorrect](#2-why-the-previous-selector-was-incorrect)
3. [Notation](#3-notation)
4. [Coverage-atom compression](#4-coverage-atom-compression)
5. [Pass 1 objective](#5-pass-1-objective)
6. [Pass 2 objective](#6-pass-2-objective)
7. [Selected-set novelty](#7-selected-set-novelty)
8. [Constraint-equivalent rule merging](#8-constraint-equivalent-rule-merging)
9. [Cardinality semantics](#9-cardinality-semantics)
10. [Pass 1 union leakage cap](#10-pass-1-union-leakage-cap)
11. [Lexicographic optimization](#11-lexicographic-optimization)
12. [Solver outcome semantics](#12-solver-outcome-semantics)
13. [Greedy fallback](#13-greedy-fallback)
14. [Pass-specific invocation](#14-pass-specific-invocation)
15. [Diagnostics and reporting](#15-diagnostics-and-reporting)
16. [Regression-test contract](#16-regression-test-contract)
17. [Configuration surface](#17-configuration-surface)
18. [Scope boundaries / future documentation](#18-scope-boundaries--future-documentation)
19. [Empirical validation](#19-empirical-validation)

---

## 1. Overview

The GLASS Router routes rows through two passes of explicit conjunctive rules:

- **Pass 1** (`NOT_SUBSCRIBE`, `predicted_class == 0`) routes rows that are
  confidently non-subscribers.
- **Pass 2** (`SUBSCRIBE`, `predicted_class == 1`) flags likely subscribers
  among the rows Pass 1 did not route.
- Rows matched by neither pass abstain.

The selector receives an **already generated and evaluated** pool of candidate
rules (`EvaluatedRule` objects) and chooses a small, complementary set of rules
for each pass. It does not generate, extend or re-score candidates.

The flow is:

```
candidate rules
  -> covered-row indices from X_val            (rule.compute_covered_indices; when X_val is given)
  -> structural filter                         (FeatureValidator; duplicate base features)
  -> split by predicted_class into Pass 1 / Pass 2 candidates
  for each requested pass:
    -> pass-specific quality gates             (QualityGateFilter; per rule)
    -> segment deduplication                   (ILPDeduplicator; identical segments)
    -> constraint-equivalent merge             (ILPRuleSelector._merge_equivalent_rules;
                                                only while the novelty floor is active)
    -> set-level constrained optimization      (two-phase ILP, CBC)
    -> independent validation of the result    (ILPRuleSelector._hard_constraint_violations)
    -> validated symbolic rule set             (List[SelectedRule])
```

Entry point: `ILPRuleSelector.select_rules(evaluated_rules, y_val, X_val,
segment_builder, passes=("pass1", "pass2"))`. Each requested pass is solved by
`ILPRuleSelector._optimize_pass(...)`.

All set-level quantities are computed over the **evaluation population** handed
to `select_rules` (`y_val`, and the row positions of each rule's
`_cached_covered_idx`, which must be positional indices into `y_val`; when
`X_val` is supplied, `select_rules` recomputes `_cached_covered_idx` from it).

---

## 2. Why the previous selector was incorrect

Rule selection is a **set-selection** problem: the value of a rule depends on
which other rules are selected with it, because rules overlap. The previous
selector treated it largely as a **rule-ranking** problem.

| Defect | Mechanism in the previous code |
|---|---|
| Overlap rewarded repeatedly | The ILP objective was `sum_j q_j * x_j` with a standalone per-rule score `q_j` (`precision^3 * coverage` in Pass 1, minus the RF penalties). A row matched by `k` selected rules contributed `k` times. The model had no variable representing the union of selected rules. |
| Novelty not enforced at set level | Novelty was a pairwise constraint that forbade a pair only if **both** rules overlapped the other by more than `1 - min_novelty_ratio`. A subset rule was never constrained unless its size exceeded `1 - min_novelty_ratio` of the superset's size (80 % at the configured 0.20), and a rule covered jointly by several selected rules was invisible to it. |
| Zero-marginal selections | Because of the two points above, a rule could be selected while adding no rows to the union (observed: a 3,744-row rule with 0 % marginal novelty). |
| Scale dominance (prior-profile symptom, not the root defect) | Observed with the prior (legacy) profile: its default penalties ($\lambda_u = 0.15$, $\lambda_m = 0.08$) and, per the notebook configuration comment, `rf_confidence = 0.5` and `rf_alignment = 0.0` for every rule (the evaluator is outside this document) cost each selected rule a constant 0.155. This exceeded the standalone score of most Pass 1 candidates, so most objective weights were negative and the solver selected exactly `min_rules` rules (observed: 6 = `min_pass1_rules`, with `max_pass1_rules = 8`), favouring individually high-scoring rules. This follows from those parameter values; the root defects are the per-rule objective and pairwise-only novelty in the first two rows, which do not depend on the penalty values. |
| Diagnostics measured something else | The printed "novelty" was an ordered, post-hoc incremental share; it differed from the constraint and depended on list order. |
| Silent cardinality weakening | `min_rules` was clamped to the number of available candidates, and several early-exit paths returned an empty selection, so fewer rules than configured could be returned without an error. |
| Greedy fallback mismatch | The fallback ranked by standalone quality multiplied by diversity and novelty factors, not by marginal union gain, and checked novelty only for the incoming rule (not for the first pick, and not for rules already selected). |

---

## 3. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $j \in J$ | candidate rule after gates, deduplication and merge | `valid` in `_optimize_pass` |
| $R_j$ | set of evaluation rows matched by rule $j$ | `arrays[rule_id]` |
| $x_j \in \lbrace 0,1 \rbrace$ | rule $j$ selected | `decision_vars[rule_id]` (`x_<id>`) |
| $a \in A$ | coverage atom: rows with an identical set of covering candidates | `membership_atoms` |
| $S_a \subseteq J$ | candidates covering atom $a$ | `members[a]` |
| $n_a$ | rows in atom $a$ | `bincount(atom_of_row)` |
| $w_a$ | objective weight of atom $a$ (rows or positives, §5–6) | `union_row_weights` summed per atom |
| $p_a$ | positive (subscriber) rows in atom $a$; equals $w_a$ in Pass 2 | leakage cap (`add_union_leakage_cap`) |
| $z_a \in [0,1]$ | atom covered by the selection (coverage objective); exists only if $\lvert S_a \rvert \ge 2$ and $w_a > 0$ | `z_<k>`, numbered in creation order |
| $m_a \in \lbrace 0,1 \rbrace$ | atom covered by $\ge 2$ selected rules (novelty); exists only if $\min(\lvert S_a \rvert, K) \ge 2$ | `m_<a>` |
| $u_a \in [0,1]$ | atom covered by the selection (leakage cap); exists only if $\lvert S_a \rvert \ge 2$ and $p_a > 0$ | `leak_<a>` |
| $P$ | positive rows in the evaluation population | `(y_val == 1).sum()` |
| $K$ | effective maximum rule count: the smaller of `max_rules` and $\lvert J \rvert$ | `actual_max` |
| $\theta$ | pass novelty floor ratio | `min_novelty_ratio_pass{1,2}` |
| $f_j$ | minimum unique rows for rule $j$: $\max(1, \lceil \theta \lvert R_j \rvert \rceil)$ | `novelty_floor_rows` |

---

## 4. Coverage-atom compression

`utils.membership_atoms(rule_ids, arrays, n_rows)` groups every covered row by
the exact set of candidates that match it. Two rows in the same atom are
matched by exactly the same candidates, so for **any** selection they are
either both covered or both uncovered, and both covered by the same number of
selected rules. Every set-level quantity the selector uses — union coverage,
union positives, per-rule unique rows, overlap — is therefore a function of
atom membership and atom counts only.

Consequences:

- **Exact union accounting.** Each atom enters each set-level term at most
  once, with its row, weight or positive count; a row can never be counted
  twice.
- **Smaller model.** At most one auxiliary variable per atom for each
  set-level quantity, instead of one per row. In one full-training run on the
  bank data (legacy profile, before the union leakage cap existed) the Pass 1
  model had 37 atoms for 53 rules; Pass 2 had 167 atoms for 85 rules, 129 of
  them carrying positives.
- **Atoms matched by a single candidate need no auxiliary variable**: at any
  solution the atom is covered iff that rule is selected, so it contributes
  $w_a x_j$ directly.

The number of atoms depends on data structure. Independent, high-cardinality
features produce many distinct membership patterns and a correspondingly
larger model.

---

## 5. Pass 1 objective

Pass 1 (`scoring_mode="precision_first"`) maximizes the number of evaluation
rows routed by the **union** of selected rules. Row weight is 1
(`utils.union_row_weights`), so $w_a = n_a$:

$$
\max \sum_{a \in A,\ |S_a| \ge 2} n_a z_a + \sum_{a \in A,\ S_a = \lbrace j \rbrace} n_a x_j
$$

$$
z_a \le \sum_{j \in S_a} x_j, \qquad 0 \le z_a \le 1
$$

(`ILPBuilder.add_union_coverage`). At an optimum $z_a = \min(1, \sum_{j \in S_a} x_j)$,
i.e. $z_a$ is the covered indicator, so the objective equals the union size.

Precision is **not** part of the Pass 1 objective. Candidates are screened by
the per-rule precision gates (`min_precision_not_subscribe` /
`max_precision_not_subscribe`) and leakage gates; the optional union leakage
cap (§10) bounds the number of subscribers routed by the selected set. There
is no set-level precision constraint. The previous standalone score
$\text{precision}^3 \cdot \text{coverage} = TN_j^3 / (|R_j|^2 N)$ does not
reduce to a function of rows alone; the union objective follows the selector
contract (maximize feasible union coverage) rather than an algebraic
equivalence.

---

## 6. Pass 2 objective

Pass 2 (`scoring_mode="recall_first"`) maximizes the number of **positive**
rows captured by the union. Row weight is $[y_i = 1]$, so $w_a = p_a$ and atoms
without positives do not enter the objective.

Motivation from the previous standalone score. For a rule with $|R_j|$ rows of
which $TP_j$ are positive, in a population of $N$ rows with $P$ positives:

$$
\text{recall}^3 \cdot \text{precision} \cdot \text{coverage}
= \left(\frac{TP_j}{P}\right)^3 \cdot \frac{TP_j}{|R_j|} \cdot \frac{|R_j|}{N}
= \frac{TP_j^4}{P^3 N}
$$

Precision and coverage cancel, so the standalone score is a strictly
increasing function of $TP_j$ alone. The set-level analogue of "true positives
captured" is the number of distinct positive rows covered by at least one
selected rule, which is what the Pass 2 objective counts.

Population and denominator:

- `pass2_population="full"`: $P$ and the objective are over the full
  evaluation population passed to `select_rules`.
- `pass2_population="oof_remainder"`: the pipeline calls the selector with the
  out-of-fold Pass 1 remainder (`X_rem`, `y_rem`). The objective counts
  positives **in the remainder**. The pipeline rescales each rule's `recall`
  and `coverage` attributes to global denominators before selection. This
  can change only per-rule quantities — which candidates pass the quality
  gates, the equivalence-merge representative and the tie-break score (whose
  standalone part becomes a constant multiple of $TP_j^4$ with $TP_j$
  counted in the remainder) — not the union objective.

Pass 2 candidates are screened by per-rule gates on precision
(`min_precision_subscribe` / `max_precision_subscribe`) and recall
(`min_recall_subscribe` / `max_recall_subscribe`). There is no set-level
Pass 2 precision constraint.

---

## 7. Selected-set novelty

Three different quantities are easily confused:

| Quantity | Definition | Order-dependent | Used for |
|---|---|---|---|
| Pairwise overlap | $\lvert R_i \cap R_j \rvert / \lvert R_i \rvert$ | no | pre-cuts only |
| Marginal (ordered) novelty | $\lvert R_k \setminus \bigcup_{l<k} R_l \rvert / \lvert R_k \rvert$ in list order | **yes** | diagnostics (`novelty=`) |
| Selected-set unique contribution | $u_j(S) = \lvert R_j \setminus \bigcup_{k \in S \setminus \lbrace j \rbrace} R_k \rvert$ | no | **the constraint** (`unique=`) |

**Contract.** When `enable_novelty_constraints=True`, every selected rule must
keep at least $f_j = \max(1, \lceil \theta |R_j| \rceil)$ rows that no other
selected rule covers:

$$
x_j = 1 \quad \Rightarrow \quad u_j(S) \ge f_j \qquad \forall j
$$

Because a rule's unique rows are new whatever precedes it, this implies that
the ordered novelty report shows at least $\theta$ for every rule in **any**
ordering. The converse does not hold: the ordered report can show a positive
share for a rule that is fully covered by the union of the others
(`test_ordered_novelty_can_hide_redundancy_unique_share_cannot`). The $\ge 1$
row term means a selected rule can never have zero unique contribution while
novelty constraints are enabled.

**Exact compact formulation** (`NoveltyAnalyzer.add_floor_constraints`). Atoms
are built over all covered rows (unweighted). For each atom with
$\min(|S_a|, K) \ge 2$:

$$
\sum_{j \in S_a} x_j - 1 \le \big(\min(|S_a|, K) - 1\big) m_a, \qquad m_a \in \lbrace 0,1 \rbrace
$$

so $m_a = 1$ whenever the atom is covered by two or more selected rules. For
each rule:

$$
|R_j| - \sum_{a:\ j \in S_a,\ m_a \text{ defined}} n_a m_a \ge f_j x_j
$$

For $x_j = 1$ the left side is $u_j(S)$ when $m_a$ equals the "multiply
covered" indicator. The linking constraint only forces $m_a$ up, and raising
$m_a$ never helps satisfy a floor constraint, so any feasible $x$ admits
$m_a = [\text{atom covered} \ge 2 \text{ times}]$, and the feasible $x$ are
exactly the floor-satisfying selections. For $x_j = 0$ the constraint is slack
because $\sum_a n_a m_a \le |R_j|$.

**Pairwise pre-cuts** (`NoveltyAnalyzer.add_novelty_constraints`). For each
pair, if either rule alone already leaves the other below its floor
($|R_i \setminus R_j| < f_i$ or $|R_j \setminus R_i| < f_j$), the model adds
$x_i + x_j \le 1$. These constraints are implied by the compact formulation and
are kept only because they tighten the LP relaxation; they are not what makes
the floor exact. Pair intersections are computed with a sparse matrix product.

**Independent check.** `NoveltyAnalyzer.find_floor_violations` recomputes
unique contributions directly from the covered-row arrays of a returned
selection. It is used by the hard-constraint validation of every ILP and
fallback result (§12).

When `enable_novelty_constraints=False` there is no floor and no pre-cuts;
redundancy is still discouraged by Phase 2 (§11).

---

## 8. Constraint-equivalent rule merging

`ILPRuleSelector._merge_equivalent_rules` keeps one representative per group of
rules that are interchangeable in **every** constraint and objective term. The
equivalence key is:

```python
(covered_rows_bytes, tuple(sorted(base_feature(f) for f, _ in rule.segment)))
```

- **Identical covered rows** make the rules identical in coverage atoms, the
  union objective, the novelty floor, pairwise pre-cuts, overlap and the
  union leakage cap.
- **Identical base-feature occurrence signature** makes them identical in the
  base-feature reuse constraint (`DiversityAnalyzer.add_diversity_constraints`
  counts one use per base-feature occurrence).
- Cardinality treats all rules alike.

Within a group only the ILP's Phase 2 tie-break term differs, so the
representative with the highest `ILPBuilder.adjusted_quality` is kept (ties:
earliest candidate). Candidate order is preserved.

**Why coverage alone is not a safe key.** Two rules can match the same rows but
use different base features. If a strong selected rule already consumes the
reuse budget for base feature `euribor`, a `euribor` twin may be infeasible
while a `campaign` twin is feasible. Merging on coverage alone could discard
the only feasible member of the group
(`test_merge_does_not_collapse_twins_with_different_bases`).

**When the merge runs.** Only when the novelty floor is active. Under the floor
two identical twins can never be selected together (each would have zero unique
rows), so every feasible selection that uses a non-representative member has a
feasible counterpart using the representative, with the same union and overlap
and a tie-break score at least as high; the optimum is unchanged. Without the
floor, twins may legitimately coexist and the merge is skipped
(`test_no_merge_when_novelty_disabled`).

Segment deduplication (`ILPDeduplicator.deduplicate_by_segment`) runs before
the merge and removes candidates with identical segments.

---

## 9. Cardinality semantics

`min_rules` is a hard contract; it is never lowered by the selector.

- `_validate_cardinality` requires integers with
  `0 <= min_rules <= max_rules` and `max_rules >= 1`. Violations raise
  `ValueError` at `ILPRuleSelector` construction (both passes) and in
  `_optimize_pass`. `GlassRouterConfig` also rejects `min > max`.
- `_optimize_pass` raises `SelectorInfeasibleError(proven=True)` when fewer
  than `min_rules` candidates remain at any of these points:
  1. the pass received no candidates at all (reported as stage
     "candidate generation", also when the list is empty because of
     structural filtering or an empty input); a non-empty list that is
     shorter than `min_rules` is reported at the next stage,
  2. after quality gates,
  3. after segment deduplication,
  4. after the constraint-equivalent merge (when it runs).
  The message names the pass, the stage, `min_rules` and the available count.
- The upper bound may shrink to the available candidates:
  `actual_max = min(max_rules, number of candidates after deduplication and
  merge)`.
- An empty selection is returned only when `min_rules == 0`.

A selection that silently contains fewer rules than configured cannot be
distinguished from a legitimate result in downstream metrics; failing loudly
keeps each run's configuration and its output consistent, and makes
infeasible configurations visible so they are changed explicitly.

---

## 10. Pass 1 union leakage cap

Two distinct leakage controls exist.

**Per-rule leakage gates** (`QualityGateFilter.apply_quality_gates_pass1`)
screen each Pass 1 candidate individually: a rule is rejected if the
subscribers it matches exceed `max_subscriber_leakage_rate * P` or
`max_subscriber_leakage_absolute`. They say nothing about the selected set: $k$
rules at the per-rule limit can together leak up to $k$ times that amount.

**Selected-union leakage cap** (`max_union_leakage_rate_pass1`, optional)
limits the subscribers routed by the union of all selected Pass 1 rules, each
subscriber counted once:

$$
\sum_{a:\ p_a > 0,\ |S_a| \ge 2} p_a u_a + \sum_{a:\ p_a > 0,\ S_a = \lbrace j \rbrace} p_a x_j
\le C, \qquad C = \lfloor L \cdot P + 10^{-9} \rfloor
$$

$$
u_a \ge x_j \quad \forall j \in S_a, \qquad 0 \le u_a \le 1
$$

(`ILPBuilder.add_union_leakage_cap`), with $L$ the configured rate and $P$ the
positives in the Pass 1 evaluation population — the same denominator the
per-rule gate uses.

- The **lower-bound** links force $u_a = 1$ whenever any selected rule covers
  the atom, so every covered subscriber is counted, and counted once
  regardless of how many selected rules match it. The coverage variables $z_a$
  are only upper-bounded and cannot be reused for this purpose.
- The cap is a hard constraint in both ILP phases, in the greedy fallback and
  in result validation. `None` (the default) disables it; with `None` the
  selections are the same as without the feature
  (`test_union_leakage_cap_off_changes_nothing`).
- The cap applies to Pass 1 only. In `pass2_population="oof_remainder"` it
  applies to the full-population Pass 1 fit and to the inner out-of-fold
  Pass 1 fits, because they use the same selector configuration.
- An infeasible combination of cap, `min_rules` and novelty floor raises
  `SelectorInfeasibleError`; the message includes the cap.

**Scope of the guarantee.** The cap holds on the evaluation population passed
to `select_rules` for Pass 1 (by default the training rows; the training fold
in cross-fitting). It does not bound leakage on held-out rows; out-of-fold or
test leakage can exceed $L$ and must be measured empirically.

---

## 11. Lexicographic optimization

Each pass is solved in two phases on the same model
(`ILPRuleSelector._optimize_pass`).

**Phase 1** maximizes the union objective of §5/§6 subject to all hard
constraints (cardinality, base-feature reuse, novelty floor and pre-cuts,
union leakage cap).

**Phase 2** requires the union to be at least the Phase 1 value $Z^\ast$
(measured directly on the returned selection) and optimizes secondary
criteria. Below, $\sum_a w_a z_a$ denotes the full coverage expression of §5,
including the single-rule atom terms $w_a x_j$:

$$
\sum_a w_a z_a \ge Z^\ast - 0.5
$$

Union weights are integers, so this is equivalent to a union of at least
$Z^\ast$. When Phase 1 is proven optimal, no feasible selection exceeds
$Z^\ast$, so the union is held exactly at the optimum.

$$
\max \quad \underbrace{\sum_a w_a z_a - \sum_j \text{raw}_j x_j} _{-\text{overlap}} + \varepsilon \sum_j q_j x_j
$$

- $\text{raw}_j = \sum _{i \in R_j} w_i$, so the first term is minus the
  double-counted weight. Among union-optimal sets, the least overlapping is
  preferred; adding a rule that contributes no union weight lowers the first
  term by $\text{raw}_j$, which outweighs any tie-break gain when
  $\text{raw}_j \ge 1$.
- Overlap is measured in the pass's weights: rows in Pass 1, **positives** in
  Pass 2. In Pass 2, overlap among negative rows is not penalized by Phase 2;
  it is limited only by the novelty floor (measured in rows) when novelty
  constraints are enabled, and not at all otherwise.
- $q_j$ is `ILPBuilder.adjusted_quality`: the standalone score
  (`utils.compute_rule_quality`) minus the RF penalties
  $\lambda_u (1 - c_j) + \lambda_m (1 - a_j)$, where $c_j$ and $a_j$ are the rule's
  `rf_confidence` and `rf_alignment`.
- $\varepsilon = 0.5 / (2 K \max_j |q_j|)$ (0 if all $q_j = 0$). Between any
  two selections of at most $K$ rules the quality term differs by at most
  0.5, less than one unit of overlap (union and overlap weights are
  integers), so quality only breaks ties.

**Exactness.** Both phases are solved with CBC (`PULP_CBC_CMD`) using
`gapRel=0.0` (passed to CBC as `-ratio 0.0`) and a time limit of
`TIME_LIMIT_PER_SOLVE = 300` seconds per solve (`-sec 300`). The selector
does not set `timeMode`; PuLP's default `timeMode="elapsed"` (PuLP 3.3.2, the
version tested) makes the limit wall-clock (`-timeMode elapsed`). No MIP
start is passed to CBC: in testing, CBC's MIP start produced a Phase 2 result
labelled optimal that was not optimal, which would void the lexicographic
guarantee.

**`Lexicographic guarantee: proven`** is reported when both phases returned a
proven optimum (§12). It means: over the candidates remaining after gates,
deduplication and merge, the selected set has the maximum feasible union
objective, and among sets achieving it, the minimum overlap (in pass
weights), with quality as the final tie-break — all subject to the hard
constraints and within solver numerical tolerances.

If Phase 2 returns no usable solution (including an infeasible or rejected
result), the Phase 1 selection is kept and the guarantee is reported as
`UNPROVEN`; the union objective of that selection is still proven optimal if
Phase 1 was.

If Phase 1 returns only a validated feasible incumbent, the Phase 1 result is
that incumbent, or the same-contract greedy selection (§13) if that selection
is feasible and has a larger union. Phase 2 then runs with $Z^\ast$ set to the
union of the Phase 1 result, and the guarantee is reported as `UNPROVEN`: the
result is feasible but not known to be optimal.

---

## 12. Solver outcome semantics

`ILPRuleSelector._classify_solution(status, sol_status)` combines PuLP's
problem status and solution status; its conditions are tested in the order
of the first four rows below, and the first match applies. PuLP can report
`"Optimal"` for a CBC run stopped by the time limit, so the solution status is
required to distinguish a proof from an incumbent.

| Outcome | Condition | Handling |
|---|---|---|
| `proven_optimal` | `status == "Optimal"` and `sol_status == 1` | accepted after validation; a proven result that fails validation raises `RuntimeError` (indicates a formulation or solver defect) |
| `feasible_unproven` | `sol_status == 2` (e.g. time limit with an incumbent); or `"Optimal"` without `sol_status` | accepted only if validation passes; labelled not proven; guarantee `UNPROVEN` |
| `infeasible` | `status == "Infeasible"` or `sol_status == -1` | Phase 1: `SelectorInfeasibleError(proven=True)`; Phase 2: keep the Phase 1 selection, guarantee `UNPROVEN` |
| `no_solution` | anything else (e.g. stopped without incumbent) | Phase 1: greedy fallback (§13); Phase 2: keep the Phase 1 selection, guarantee `UNPROVEN` |
| `rejected_incumbent` | unproven incumbent that failed validation | as `no_solution` |

**Independent validation** (`_solve_phase`) of every returned solution:

- all rule variables have values and are integral (tolerance `1e-4`);
- `_hard_constraint_violations`: no duplicate rules, cardinality within
  `[min_rules, actual_max]`, base-feature reuse within limit, novelty floor
  recomputed from covered rows, union leakage recomputed from covered rows
  (when the cap is active);
- the value of the union-coverage expression (not the phase objective) does
  not exceed the union recomputed from the selection by more than 0.5;
- Phase 2 only: the recomputed union is at least $Z^\ast$.

Outcomes are recorded in `selection_report_[pass_name]["phase1" | "phase2"]`
(`status`, `sol_status`, `state`, `problems`).

---

## 13. Greedy fallback

`GreedySelector.greedy_select` is a robustness fallback. It is not the primary
optimizer and carries no optimality guarantee. `_optimize_pass` runs it
silently once per pass, before Phase 1, and uses its result in two cases:

- as the returned selection when Phase 1 yields no usable solution
  (`no_solution` or `rejected_incumbent`); it is then re-run with logging
  enabled;
- in place of an unproven Phase 1 incumbent when it is feasible and has a
  larger union (§11).

Otherwise it only appears in the log as the "Greedy incumbent". It is never
passed to CBC as a MIP start.

It enforces the same hard constraints as the ILP, passed explicitly by the
caller (keyword-only arguments):

- **Objective direction:** at each step, the admissible candidate with the
  largest marginal union gain in the pass's row weights (rows for Pass 1,
  positives for Pass 2). Ties are broken by
  $q^{\text{std}}_j \cdot (1 + \omega_d d_j) \cdot (1 + \omega_n \nu_j)$, where
  $q^{\text{std}}_j$ is the standalone score (`utils.compute_rule_quality`),
  $\omega_d$ is `diversity_weight`, $\omega_n$ is `greedy_novelty_weight`,
  $d_j$ is base-feature diversity against the current selection and
  $\nu_j$ the candidate's marginal novelty. Candidates with zero gain are never
  added.
- **Novelty floor:** the active pass floor `(min_novelty_ratio, 1)`, or none
  when the ILP has novelty disabled. A candidate is admissible only if, after
  adding it, **every** selected rule — including earlier picks — still meets
  its floor.
- **Base-feature reuse:** the same per-base-feature limit as the ILP.
- **Union leakage cap:** when active, a candidate is admissible only if the
  union's covered positives stay within the cap.
- **Cardinality:** stops at `actual_max`. It cannot guarantee `min_rules`.

The caller validates the greedy result with `_hard_constraint_violations`. Any
violation — most commonly fewer than `min_rules` rules — raises
`SelectorInfeasibleError(proven=False)`. A greedy failure does not prove that
no feasible selection exists; the error reports it as unproven. A valid greedy
result is returned with `source = "greedy_fallback"` and
`lexicographic_guarantee = "not applicable"`; Phase 2 is not run for it.

`min_novelty_greedy`, `min_absolute_new_samples` and
`greedy_hard_novelty_cutoff` remain in the constructor for compatibility and
do not affect selection (they are only echoed in the configuration
printout): the fallback must enforce the floor established on the ILP path.

---

## 14. Pass-specific invocation

```python
select_rules(evaluated_rules, y_val=None, X_val=None, segment_builder=None,
             passes=("pass1", "pass2"))
```

`y_val` defaults to `None` in the signature but is required whenever
`evaluated_rules` is non-empty (`ValueError` otherwise).

- Default: both passes are selected, and each requested pass must satisfy its
  configured `min_rules` (§9). `GlassRouterPipeline._fit_full`
  (`pass2_population="full"`) uses the default.
- A pass not listed in `passes` is not run, returns `[]`, gets no
  `selection_report_` entry and no post-selection analysis.
- `passes` must be a non-empty subset of `("pass1", "pass2")`.

Training routines that fit a single pass request it explicitly:

- `GlassRouterPipeline._fit_pass1_only` (in `oof_remainder` mode: the
  full-population Pass 1 fit and the inner out-of-fold Pass 1 fits):
  `passes=("pass1",)`.
- `GlassRouterPipeline._fit_oof_remainder` (Pass 2 on the remainder):
  `passes=("pass2",)`.

Skipping a pass that was not requested is different from accepting an
infeasible requested pass. Without `passes`, a single-pass call would present
zero candidates to the other pass, which under §9 correctly raises when that
pass's `min_rules > 0`.

---

## 15. Diagnostics and reporting

**Console log per pass** (`_optimize_pass`, `QualityGateFilter`,
`DiversityAnalyzer`, `NoveltyAnalyzer`):

- candidates in, passed/rejected by quality gates with rejection breakdown;
- candidates after segment deduplication and after the equivalence merge;
- union leakage cap as a count of positives (when active);
- base-feature reuse limit applied;
- pairwise tightening constraints, multi-cover atom binaries, floor
  constraints (when novelty constraints are enabled);
- number of coverage atoms and atoms with positive weight;
- greedy incumbent (size, union, feasibility);
- per phase: outcome, PuLP status, solution status; Phase 1 union value;
- lexicographic guarantee; union leakage vs cap (when active).

**Post-selection analysis** (`NoveltyAnalyzer.analyze_selection_novelty`, per
pass, with at least two rules):

- per rule: rows covered, ordered novelty, cumulative union, unique rows and
  unique share;
- union coverage, sum of raw coverage, double-counted rows;
- minimum unique share, rules with zero unique contribution;
- union positives and negatives, positive share, per-rule positives summed
  (for comparison with the union count).

**Structured report** `ILPRuleSelector.selection_report_[pass_name]`, reset at
the start of every `select_rules` call:

| Key | Meaning |
|---|---|
| `source` | `"ilp"`, `"greedy_fallback"`, `"infeasible"`, or `"empty (min_rules = 0)"` |
| `scoring_mode`, `union_weight` | pass mode; `"rows"` or `"positives"` |
| `min_rules`, `max_rules` | configured minimum; effective maximum |
| `novelty_floor` | `(ratio, 1)` or `None` |
| `base_reuse_limit` | per-base-feature limit applied |
| `candidates_after_gates` | candidates after quality gates **and** segment deduplication |
| `candidates_after_merge` | candidates after the equivalence merge (equal to the previous count when the merge is skipped) |
| `union_leakage_cap` | cap in positives, or `None` |
| `phase1`, `phase2` | `status`, `sol_status`, `state`, `problems`; `phase2` is absent when Phase 1 is infeasible or the fallback is used |
| `lexicographic_guarantee` | `"proven"`, `"UNPROVEN"`, `"not applicable"` |
| `union_coverage` | union of the final selection in the pass's weights (rows in Pass 1, positives in Pass 2) |
| `union_positives`, `n_selected` | positives covered by the final selection; number of rules |

For insufficient-candidate failures the entry holds `source`, `min_rules`,
`candidates` and `stage`. For an empty selection with `min_rules = 0` it holds
`source`, `min_rules` and `n_selected`.

---

## 16. Regression-test contract

Suite: `bank_pipeline/tests/test_ilp_rule_selector_union.py`. 49 test
functions, 66 cases with parametrization. Tests drive
`ILPRuleSelector._optimize_pass`, `select_rules`, `GreedySelector`,
`ILPBuilder`, `NoveltyAnalyzer`, `QualityGateFilter`, `GlassRouterConfig` and
the utilities directly, with duck-typed rules of fixed coverage and a stub
base-feature validator; `SelectedRule` is replaced by a stub in the
`select_rules` tests. Solver-failure paths are exercised by scripting
`ILPBuilder.solve`.

| Area | Tests |
|---|---|
| Overlap / union accounting | `test_abc_union_beats_redundant_raw_coverage`, `test_union_coverage_expression_counts_each_row_once`, `test_atoms_are_exact_membership_patterns`, `test_selection_report_union_matches_true_union` |
| Union-contained and zero-marginal rules | `test_rule_inside_union_of_others_never_selected` (novelty on/off × 3 candidate orders), `test_no_zero_marginal_rule_with_default_rf_penalties` |
| Set-level novelty | `test_novelty_floor_is_set_level_not_pairwise`, `test_ordered_novelty_can_hide_redundancy_unique_share_cannot`, `test_floor_boundaries` |
| Exactness against exhaustive search | `test_compact_floor_matches_exhaustive_search` (60 random instances; union and overlap must equal the enumerated optimum; infeasible instances must raise proven), `test_union_leakage_cap_matches_exhaustive_search` (40 instances with the cap) |
| Cardinality | `test_min_rules_above_candidates_after_quality_gates_raises`, `test_min_rules_above_candidates_after_equivalent_merge_raises`, `test_zero_candidates_with_positive_min_rules_raises`, `test_zero_valid_rules_after_gates_with_positive_min_rules_raises`, `test_invalid_cardinality_configuration_is_a_config_error`, `test_feasible_selection_with_exactly_min_rules_succeeds` |
| Infeasibility | `test_proven_infeasible_raises_with_constraint_details`, `test_union_leakage_cap_infeasible_raises` |
| Greedy behaviour and parity | `test_greedy_prefers_union_gain_over_standalone_quality`, `test_greedy_rechecks_earlier_picks_against_floor`, `test_greedy_and_ilp_satisfy_identical_hard_constraints`, `test_base_feature_reuse_enforced_by_ilp_and_fallback`, `test_fallback_uses_pass_floor_not_greedy_defaults`, `test_fallback_respects_union_leakage_cap` |
| Fallback rejection | `test_fallback_that_cannot_meet_min_rules_raises_unproven`, `test_greedy_fallback_below_min_rules_is_rejected` |
| Solver outcomes | `test_solution_classification` (8 cases), `test_invalid_unproven_incumbent_is_rejected`, `test_proven_optimum_failing_validation_raises`, `test_time_limited_incumbent_accepted_but_marked_unproven` |
| Pass 2 objective | `test_pass2_union_recall_counts_positives_once`, `test_recall_first_score_is_a_function_of_true_positives_only` |
| Quality gates (per rule) | `test_per_rule_gates_are_hard_and_unchanged`, `test_leakage_cap_is_per_rule_not_union` (cap disabled) |
| Union diagnostics | `test_union_precision_and_leakage_diagnostics_use_union` |
| Equivalence merge | `test_merge_keeps_best_of_truly_equivalent_twins`, `test_merge_does_not_collapse_twins_with_different_bases`, `test_no_merge_when_novelty_disabled` |
| Single-pass invocation | `test_select_rules_single_pass_does_not_run_the_other_pass`, `test_select_rules_empty_input_obeys_min_rules`, `test_select_rules_rejects_unknown_passes` |
| Union leakage cap | `test_union_leakage_cap_off_changes_nothing`, `test_union_leakage_cap_binds_on_the_union`, `test_union_leakage_counts_shared_positives_once`, `test_union_leakage_rate_validation`, `test_select_rules_applies_cap_to_pass1` |
| Misc | `test_unknown_scoring_mode_rejected`, `test_selected_set_is_independent_of_candidate_order` |

Coverage gaps: acceptance of a selection whose union leakage **exactly equals**
the cap has no dedicated case; it is exercised by the exhaustive-search test,
where most feasible instances end exactly at the cap. Real `EvaluatedRule` and
`SelectedRule` objects are not used, and the real `FeatureValidator` is
constructed but replaced by the stub before use.

### Regression test record

| Field               | Value |
|---------------------|-------|
| Date                | 2026-09-30 |
| Commit / branch     | `f845448` / `pipeline-refactor-router-alignment` |
| Python / PuLP / CBC | `3.14.3` / `3.3.2` / `2.10.3` |
| pytest              | `9.1.1` |
| Platform            | `Windows-11-10.0.26200-SP0` |
| Command             | `python -m pytest tests/test_ilp_rule_selector_union.py -q -W ignore::DeprecationWarning` (run from `bank_pipeline/`) |
| Result              | 66 passed, 0 failed (15.10 s) |
| Notes               | Run from the regression cell at the very top of the notebook to avoid dependencies on the Stage 2 fit |
---

## 17. Configuration surface

### Selector-relevant `GlassRouterConfig` fields

Config fields reach the selector through `GlassRouterPipeline._new_selector()`,
which renames several of them.

| Config field | Selector parameter | Used by the selector as | Required |
|---|---|---|---|
| `min_pass1_rules`, `max_pass1_rules` | same | Pass 1 cardinality (hard) | yes |
| `min_pass2_rules`, `max_pass2_rules` | same | Pass 2 cardinality (hard) | yes |
| `min_precision_not_subscribe`, `max_precision_not_subscribe` | `min_precision_pass1`, `max_precision_pass1` | Pass 1 per-rule precision gate | yes |
| `max_subscriber_leakage_rate`, `max_subscriber_leakage_absolute` | `max_subscriber_leakage_rate_pass1`, `max_subscriber_leakage_absolute_pass1` | Pass 1 per-rule leakage gate | yes |
| `max_union_leakage_rate_pass1` | same | Pass 1 selected-union leakage cap; `None` = off | no |
| `min_precision_subscribe`, `max_precision_subscribe` | `min_precision_pass2`, `max_precision_pass2` | Pass 2 per-rule precision gate | yes |
| `min_recall_subscribe`, `max_recall_subscribe` | `min_recall_pass2`, `max_recall_pass2` | Pass 2 per-rule recall gate | yes |
| `min_novelty_ratio_pass1`, `min_novelty_ratio_pass2` | same | set-level novelty floor $\theta$ per pass | yes |
| `enable_novelty_constraints` | same | floor, pre-cuts and equivalence merge on/off | yes |
| `diversity_weight` | same | greedy tie-break weight (the pipeline also passes it to the rule generator, outside this document) | yes |
| `lambda_rf_uncertainty`, `lambda_rf_misalignment` | same | RF penalties in `adjusted_quality`: Phase 2 tie-break and equivalence-merge representative; `None` -> 0.15 / 0.08 | no |
| `pass2_population` | — | not read by the selector; the pipeline uses it to choose the population passed to `select_rules` for Pass 2 (§6, §14) | no (`"full"`) |

### Selector-only parameters (not in `GlassRouterConfig`)

| Parameter | Default | Effect |
|---|---|---|
| `max_base_reuse_pass1`, `max_base_reuse_pass2` | `None` -> `max_feature_usage` | per-base-feature reuse limit |
| `max_feature_usage` | 40 | reuse limit when the above are `None` |
| `greedy_novelty_weight` | 0.5 | greedy tie-break |
| `min_novelty_greedy`, `min_absolute_new_samples`, `greedy_hard_novelty_cutoff` | 0.15 / 30 / `True` | accepted; no effect on selection |
| `rule_prefixes` | `('nsd', 'jed', 'cci', 'eci', 'dow', 'behav', 'campaign', 'cpi')` | passed to `FeatureValidator`, which provides base-feature extraction (reuse limit, merge key, greedy diversity) and the structural duplicate-base-feature filter |
| `TIME_LIMIT_PER_SOLVE` (class attribute) | 300 s | CBC time limit per solve; wall-clock under PuLP's default `timeMode` (§11) |

The pipeline does not pass `max_base_reuse_pass1`/`_pass2`, so the limit is
`max_feature_usage` (40). Because the structural filter removes rules that
repeat a base feature, each selected rule uses a base feature at most once,
and the limit cannot bind unless `max_rules` exceeds 40.

### Implementation constants

- Novelty floor minimum unique rows: 1 (ILP and fallback).
- CBC relative gap: 0.
- Integrality tolerance in validation: `1e-4`; a rule counts as selected when
  its variable value exceeds 0.5.

Experiment values (e.g. `min_pass1_rules = 2`) belong to notebook profiles and
are not architectural requirements.

---

## 18. Scope boundaries / future documentation

The following are outside this document and will be documented separately:

- symbolic rule-lattice construction;
- RF-guided beam search;
- depth-aware (depth 1 / 2 / 3) pruning;
- canonical conjunction handling;
- candidate-generation internals and complexity;
- the cross-fitting architecture;
- Stage 4 interaction with the router outputs.

Inference-time routing (`GlassRouterPipeline.predict`) is unchanged by the
selector revision and is not described here.

---

## 19. Empirical validation

End-to-end run of 2026-09-30: legacy profile, `pass2_population="full"`,
`max_union_leakage_rate_pass1 = 0.14`, 32,950 training rows (3,712
positives), 8,238 test rows, 5-fold cross-fit.

The configured union leakage cap (§10) is a hard constraint only on the
selector's fit population (the rows passed to `select_rules` for Pass 1). The
leakage figures below are held-out observations (test rows for the full-train
model, out-of-fold rows for train OOF) and may exceed the cap.

| Metric | Full-train model (test) | Train OOF |
|---|---|---|
| Profile | legacy | legacy |
| Pass 1 / Pass 2 rules | 4 / 3 | per fold: 3–4 / 2–4 |
| Pass 1 routed | 37.0% | 34.9% |
| Pass 2 flagged | 23.5% | 25.7% |
| Abstained | 39.4% | 39.4% |
| Pass 1 union leakage, held-out (fit-population cap 14.0%) | 14.9% | 13.8% (folds 9.4–15.8%) |
| Pass 2 recall (all subscribers) | 61.9% | 63.0% |
| Covered precision / recall / F1 (non-abstained rows) | 29.6% / 80.6% / 43.3% | 27.6% / 82.0% / 41.3% |
| Lexicographic guarantee (both passes) | proven | not recorded per fold |

On the fit population, the full-train Pass 1 selection routes 11,824 rows with
union leakage exactly at the cap (519 of 3,712 positives); Pass 2 captures
2,379 positives (64.1%). The selected sets double-count 5.4% (Pass 1) and
27.2% (Pass 2) of their raw row coverage, and no selected rule has zero unique
contribution.