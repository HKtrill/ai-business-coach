# Stage 2 Constrained Symbolic Rule Router Functional Overview

Stage 2 is the symbolic routing layer of the GLASS Cascade. It operates over a curated binary predicate space derived from the Stage 2 source features. In this implementation, 8 source features are transformed through a locked binning strategy into 29 canonical binary segment features. These binary features define the predicate atoms searched by the rule system.

## Terminology

```text
Binary feature = one generated bin indicator
Predicate atom = one feature-value condition
Node = one candidate conjunction in the symbolic lattice
Rule = one selected conjunction used for routing
```

Example:

```text
Binary feature:
    campaign_fresh

Predicate atom:
    campaign_fresh = 1

Depth 1 node:
    campaign_fresh = 1

Depth 2 node:
    campaign_fresh = 1 AND eci_hot = 1

Depth 3 node:
    campaign_fresh = 1 AND eci_hot = 1 AND nsd_hot = 1
```

## Symbolic Rule Lattice

The Stage 2 symbolic router searches a constrained symbolic rule lattice over these predicate atoms. Each lattice node represents a unique conjunction of one or more atoms. Equivalent permutations are collapsed into one canonical rule node, so `A AND B AND C`, `A AND C AND B`, and `B AND A AND C` all represent the same rule.

## RF-Guided Beam Search

The rule generator does not exhaustively search the full lattice. It uses RF-guided feature ordering and beam search to explore candidate rules. Random Forest feature importance helps order the binary predicate space so higher-signal features are considered earlier during expansion.

Random Forest is used as a training-time guide for symbolic rule discovery. It does not make inference-time decisions in the deployed Stage 2 router.

## Feasible Region

The rule configuration defines the feasible region of the symbolic lattice. This feasible region is shaped by structural constraints, support thresholds, precision gates, recall gates, coverage limits, leakage limits, novelty controls, diversity controls, beam width, and max rule depth.

## Depth-Staged Candidate Generation

Depth-staged constraints are used during candidate generation:

```text
Depth 1:
    Structural validity only

Depth 2:
    Light pruning and leakage guardrails

Depth 3:
    Full quality constraints
```

This allows simple predicates to survive long enough to form useful higher-depth conjunctions.

## Validation and Rule Selection

After candidate generation, rules are evaluated on validation data. This produces validation-grounded estimates of precision, recall, coverage, support, overlap, and RF-related diagnostics.

The final rule set is selected with integer-linear-programming-based optimization. ILP converts the larger feasible candidate pool into a compact rule portfolio using pass-specific quality gates, cardinality limits, novelty constraints, and diversity constraints.

## Two-Pass Routing

Stage 2 uses a two-pass routing design:

```text
Pass 1:
    Precision-first NOT_SUBSCRIBE routing

Pass 2:
    Recall-oriented SUBSCRIBE recovery
```

During prediction, Pass 1 routes high-confidence `NOT_SUBSCRIBE` regions first. Pass 2 then acts only on samples that remain unresolved. Samples not covered by either pass remain abstained rather than forced into a symbolic decision.

This makes Stage 2 a sequential symbolic routing layer rather than a flat rule classifier. Its role is not simply to maximize standalone F1 or F2. Its role is to add compact, interpretable, nonredundant routing behavior to the full GLASS Cascade.

## Configurable Routing Behavior

Stage 2 configuration can be adjusted depending on the desired routing behavior:

```text
Aggressive positive recovery:
    Wider feasible region
    Lower precision floor
    More tolerant leakage settings
    More Pass 2 rules

Conservative routing:
    Smaller feasible region
    Higher precision floor
    Stricter leakage limits
    Fewer selected rules

Interpretability-first routing:
    Lower max depth
    Fewer selected rules
    Stronger novelty and diversity controls
```

## Next PR Scope

For the next cascade-finalization PR, the main Stage 2 goal is to finalize the symbolic router configuration and improve notebook diagnostics around how the rule system works.

## Planned Stage 2 Additions

```text
- Finalize rule generation configuration
- Validate depth-2 and depth-3 behavior
- Confirm duplicate-path canonicalization
- Confirm structural feasibility constraints
- Tune precision / recall / leakage tradeoffs
- Track selected rule count and depth distribution
- Measure unique positive contributions
- Measure cascade agreement and all-wrong-together behavior
- Add concise Stage 2 summary outputs
- Add visual diagnostics for the rule generation process
```

## Planned Stage 2 Visuals

```text
- Symbolic rule lattice diagram
- Binary predicate atom source diagram
- Feasible-region / rule-selection funnel
- Rule-quality scatter plot
- Rule overlap or activation heatmap
```

## Core Visual Language

```text
Source features
→ locked binary predicate space
→ symbolic rule lattice
→ feasibility constraints
→ RF-guided feature ordering and beam search
→ validation-scored candidates
→ ILP-selected rule portfolio
→ sequential routing with abstention
```

## Summary

Stage 2 transforms audited source features into canonical binary predicate atoms, searches a constrained symbolic rule lattice with RF-guided feature ordering and beam search, validates candidate rules, and uses ILP selection to choose a compact two-pass routing rule set. The resulting symbolic layer can be tuned toward aggressive positive recovery, conservative routing, or interpretability-first symbolic coverage.
