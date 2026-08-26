# Future Research Plan: Efficient Local Conversational Intelligence

## ChurnBot / GLASS Research Roadmap

> **Status:** Proposed Future Research  
> **Current Primary Research:** GLASS vs. Black-Box Predictive Modeling  
> **Future Focus:** Efficient local conversational inference and Transformer re-engineering

---

## 1. Overview

The current ChurnBot research focuses on whether interpretable glass-box systems can compete with or outperform conventional black-box machine-learning systems while preserving auditability, traceability, and meaningful abstention.

A future phase of the project will investigate the **conversational component of ChurnBot**.

Rather than assuming that a conventional large language model is required to provide a capable natural-language interface, this research will ask a more fundamental question:

> **Can a capable domain-specific conversational system be constructed with dramatically less computation by combining classical computer-science algorithms, symbolic methods, specialized models, and redesigned Transformer inference?**

The objective is not to assume that this hypothesis is correct. The research will remain deliberately neutral and experimentally determine which approaches are feasible.

---

# 2. Core Research Question

Modern Transformer attention evaluates a large number of query-key interactions.

For a sequence containing $n$ elements, dense self-attention considers approximately:

$$
n^2
$$

pairwise relationships during prefill, with computational cost commonly expressed as $\Theta(n^2d)$ when the query-key dimension $d$ is included.

Prefill and autoregressive decode represent different computational settings. During prefill, many queries and keys are processed simultaneously as an $n \times n$ interaction structure. During decode, each new query attends over a growing KV cache, and memory movement may become as important as arithmetic cost.

This roadmap considers both settings, although a future formal thesis should select one or analyze them separately.

Unless otherwise stated, the primary interest is in **training-free inference-time methods applied to existing pretrained models**, rather than requiring the base model to be retrained around a new attention mechanism.

The central research question is:

> **On realistic Transformer activations, can large portions of the interaction or search space be eliminated or approximated before exact evaluation, producing substantially lower computational cost while preserving acceptable model quality?**

The research does not require a universally sub-quadratic worst case. Some inputs may still require examination of most possible interactions:

$$
T_{\mathrm{worst}}(n)=O(n^2)
$$

Indeed, conditional fine-grained complexity results suggest that truly sub-quadratic attention is unlikely in important worst-case regimes under assumptions such as SETH.

The research instead asks whether realistic workloads possess exploitable structure that permits substantially better **instance-dependent behavior and empirical average-case scaling**, ideally approaching:

$$
O(n)
$$

or:

$$
O(n\log n)
$$

under favorable but measurable conditions.

The research does **not** begin with the assumption that this is possible.

The hypothesis may be confirmed, partially confirmed, or rejected.

---

# 3. Central Idea: Do Not Compute What Can Be Safely Rejected

Many optimization strategies perform computation and subsequently determine that some computed values are unimportant.

This research will investigate a more aggressive alternative:

> **Determine that a region of computation is unnecessary before calculating the individual interactions contained within it.**

Conceptually, instead of evaluating an entire attention matrix:

```text
FULL DENSE ATTENTION

########################
########################
########################
########################
########################
########################
```

the algorithm could identify entire regions that do not require exploration:

```text
HIERARCHICALLY PRUNED ATTENTION

######......######......
######......######......
######......######......
............######......
............######......
######......######......
```

where:

```text
# = evaluated interaction region
. = region rejected before exact evaluation
```

The central optimization target is therefore:

$$
C_{\mathrm{effective}}
=
C_{\mathrm{pruning}}
+
C_{\mathrm{retained}}
$$

where:

- $C_{\mathrm{pruning}}$ is the cost of searching, bounding, and pruning candidate regions.
- $C_{\mathrm{retained}}$ is the cost of computing the attention interactions that survive pruning.

The resulting computational savings are:

$$
C_{\mathrm{savings}}
=
C_{\mathrm{dense}}
-
C_{\mathrm{effective}}
$$

Therefore, a pruning strategy is beneficial only when:

$$
C_{\mathrm{pruning}}
+
C_{\mathrm{retained}}
<
C_{\mathrm{dense}}
$$

In other words, the cost of deciding what not to compute must remain **substantially smaller than the computation that is eliminated**.

---

# 4. Candidate Algorithm Families

The following approaches represent candidate tools, baselines, and hybrid components.

Several have substantial prior work and should not be interpreted as novel individually. Their purpose in this roadmap is to identify algorithmic mechanisms that may contribute to a more narrowly defined future thesis.

The ordering remains provisional and should change as literature review, mathematical analysis, and benchmarking produce evidence.

---

## Priority 1 — Hierarchical Divide-and-Conquer + Branch-and-Bound

### Hypothesis

Represent attention computation hierarchically and recursively divide the candidate interaction space into regions.

For a query $q$ and key $k_j$, define the scaled attention score:

$$
s_j = \frac{q \cdot k_j}{\sqrt{d}}
$$

Let $E$ represent the set of keys already evaluated exactly.

Define the running maximum score:

$$
m = \max_{j \in E} s_j
$$

and the partial shifted partition function:

$$
Z_E = \sum_{j \in E} e^{s_j-m}
$$

When $m$ changes, the accumulated quantities must be rescaled consistently with numerically stable online-softmax computation.

For a candidate region $R$:

* $|R|$ is the number of keys contained in the region.
* $\hat{s}(R)$ is an admissible upper bound on the score of every key contained in $R$.

Instead of bounding only the strongest individual interaction, define an upper bound on the region's possible contribution to the softmax denominator:

$$
B_{\mathrm{mass}}(R)
=
|R|e^{\hat{s}(R)-m}
$$

If $B_{\mathrm{drop}}$ represents the accumulated upper bound associated with regions already eliminated, a candidate region may be eliminated only when the total omitted-mass budget remains within a predefined tolerance:

$$
B_{\mathrm{drop}}
+
B_{\mathrm{mass}}(R)
\leq
\varepsilon Z_E
$$

where $\varepsilon$ controls the maximum tolerated omitted denominator mass relative to the contribution already evaluated.

This condition is intentionally more conservative than simply testing whether the maximum score inside a region falls below a threshold. A large region containing many individually weak interactions may still carry substantial aggregate softmax mass.

This bound addresses the softmax denominator only. A rigorous guarantee on the final attention output must also account for the associated value vectors $V$, normalization effects, and the chosen error norm. **Deriving a sufficiently tight and inexpensive output-level bound remains part of the proposed research rather than an assumed result.**

Candidate region bounds may include established techniques such as coordinate-wise min/max bounds, centroid-radius bounds, norm bounds, or other geometric summaries. These techniques have prior art and are considered possible components rather than novelty claims.

### Region Decision

Each region may result in one of three decisions:

1. **Eliminate** — its maximum possible aggregate contribution fits within the remaining error budget.
2. **Approximate** — its contribution cannot be eliminated safely but can potentially be represented using a cheaper coarse approximation with bounded error.
3. **Expand** — subdivide the region and inspect its children, eventually performing exact evaluation when necessary.

Conceptually:

```text
                     Full Search Region
                           |
              ---------------------------
              |            |            |
           Region A     Region B     Region C
              |                         |
          ---------                 ---------
          |       |                 |       |
         A1      A2                C1      C2
```

If `Region B` can be safely eliminated or approximated, its individual interactions do not require exact evaluation.

If `Region A` cannot yet be bounded tightly enough, it can be subdivided and searched at finer resolution.

### Why This Is a High-Priority Direction

A useful hierarchical bound could avoid entire regions of exact attention computation rather than pruning individual interactions one at a time.

The research challenge is finding bounds and traversal strategies that are simultaneously:

* inexpensive,
* sufficiently tight,
* size-aware,
* numerically stable,
* compatible with attention normalization,
* robust under realistic numerical precision,
* and cheaper than the computation they eliminate.

The central question is not merely whether hierarchical pruning is possible — substantial prior work already establishes that it is — but whether a **training-free, size-aware bounded-search formulation can provide useful guarantees and favorable instance-dependent scaling on real Transformer activations**.

---

# 5. Priority 2 — Graph Representation of Attention

Another major direction is to stop treating attention primarily as a matrix operation.

Instead:

> **Represent tokens or internal elements as vertices and candidate interactions as edges.**

Dense attention can be viewed as a weighted, highly connected graph in which tokens or internal representations are vertices and pairwise attention interactions are candidate edges.

The research question becomes:

> Which edges actually need to exist?

This formulation opens the problem to classical graph algorithms.

Candidate techniques include:

* graph sparsification,
* graph partitioning,
* community detection,
* connected-component reasoning,
* spectral graph methods,
* minimum-cut-inspired separation,
* neighborhood search,
* bounded-degree graphs,
* hierarchical graphs,
* shortest-path or constrained-path formulations,
* dynamic graphs,
* and graph coarsening.

Rather than evaluating every possible vertex-to-vertex interaction, the algorithm would attempt to identify and construct only the subset of edges likely to carry useful information, avoiding unnecessary pairwise computation before it occurs..

---

# 6. Priority 3 — Best-First Search / A* / Beam Search

The attention search space could potentially be explored as an ordered search problem.

Regions or candidate interactions would be placed into a priority structure according to estimated importance.

The algorithm could then expand the highest-value candidates first.

Possible methods include:

* best-first search,
* A* search,
* beam search,
* priority queues,
* bounded priority search,
* iterative deepening,
* and anytime search.

A possible termination condition is:

> Stop exploring when the upper bound of every remaining candidate falls below the minimum value required to affect the retained attention result.

This could complement branch-and-bound particularly well.

---

# 7. Priority 4 — Graph Partitioning + Hierarchical Clustering

Tokens could first be grouped into computational neighborhoods.

Interactions could first be approximated or bounded at the cluster level before expanding to individual elements.

Only clusters displaying potentially important relationships would be expanded.

```text
Tokens
  |
  +-- Cluster A
  |     +-- A1
  |     +-- A2
  |
  +-- Cluster B
  |     +-- B1
  |     +-- B2
  |
  +-- Cluster C
```

If a cheap cluster-level bound indicates that interactions between Cluster A and Cluster C cannot materially affect the result, the entire $A \times C$ interaction region can be rejected without examining every individual pair.

Candidate methods include:

* recursive graph partitioning,
* hierarchical clustering,
* k-d-tree-style decomposition,
* ball-tree-style decomposition,
* spatial partitioning analogies,
* centroid bounds,
* and recursive bisection.

---

# 8. Priority 5 — Approximate Nearest-Neighbor / Maximum Inner-Product Search

Attention can also be viewed as a search for highly relevant query-key interactions.

Instead of comparing every query with every key, investigate algorithms that retrieve promising candidates directly.

Candidate approaches include:

* locality-sensitive hashing,
* approximate nearest-neighbor search,
* maximum inner-product search,
* tree-based retrieval,
* hashing,
* bucketing,
* quantization,
* and coarse-to-fine retrieval.

The key research question is whether candidate retrieval can be performed cheaply enough that:

$$
C_{\mathrm{retrieval}} + C_{\mathrm{sparse}} \ll C_{\mathrm{dense}}
$$

where:

- $C_{\mathrm{retrieval}}$ is the computational cost of identifying promising query-key candidates.
- $C_{\mathrm{sparse}}$ is the cost of evaluating the retained attention interactions.
- $C_{\mathrm{dense}}$ is the cost of the corresponding dense-attention baseline.

while maintaining acceptable model quality.

---

# 9. Priority 6 — Sparse + Low-Rank Hybrid Decomposition

Attention may contain both:

* broad global structure, and
* small numbers of unusually important individual interactions.

A hybrid strategy could represent the broad structure approximately while explicitly calculating only important sparse relationships.

Conceptually:

$$
A \approx L + S
$$

where:

- $A$ represents the original attention interaction structure.
- $L$ represents a low-rank approximation capturing broad global structure.
- $S$ represents a sparse component containing high-importance interactions that require more precise evaluation.

The research question is whether inexpensive global approximation can provide enough information to identify where exact sparse computation is actually necessary.

This approach could be combined with hierarchical pruning so that expensive exact computation is reserved for regions where the low-rank approximation is insufficient.

---

# 10. Priority 7 — Dynamic Programming and Memoization

Repeated or overlapping subcomputations should not be recomputed when previously derived results can be safely reused.

Potential research directions include:

* caching partial dot products,
* caching bounds,
* memoizing repeated subproblems,
* maintaining intermediate representations,
* reusing previous search results,
* incremental attention updates,
* and exploiting overlap between neighboring inference steps.

The research question is whether attention computation contains sufficiently overlapping or reusable subproblems for dynamic programming and memoization to reduce total work beyond existing inference caching mechanisms.

---

# 11. Priority 8 — Multiresolution / Coarse-to-Fine Analysis

Instead of immediately examining the attention space at full resolution:

1. evaluate a coarse or compressed representation,
2. identify promising regions,
3. increase resolution only where necessary,
4. discard low-value regions.

Conceptually:

```text
Coarse Matrix
     |
 identify important regions
     |
Medium Resolution
     |
 identify important regions
     |
Fine Resolution
```

Potential mathematical tools include:

* multiresolution analysis,
* hierarchical matrices,
* wavelets,
* recursive aggregation,
* pyramid structures,
* and coarse-to-fine optimization.

The research question is whether coarse representations can provide sufficiently reliable bounds or importance estimates to eliminate low-value regions before expensive fine-resolution attention is computed.

This approach may combine naturally with divide-and-conquer and branch-and-bound.

---

# 12. Priority 9 — Randomized and Probabilistic Algorithms

Not every decision necessarily requires deterministic exhaustive evaluation.

Possible approaches include:

* randomized sampling,
* importance sampling,
* reservoir sampling,
* Monte Carlo estimation,
* randomized projections,
* sketching algorithms,
* probabilistic bounds,
* and randomized matrix approximation.

A probabilistic pruning rule might allow a region to be rejected when the probability that it contains an important interaction falls below a controlled threshold.

The research question is whether probabilistic estimates can reject enough low-value regions to produce meaningful computational savings while keeping the probability of pruning important interactions within a controlled error bound.

This introduces an explicit accuracy/computation tradeoff that can be measured experimentally.

---

# 13. Priority 10 — Predictive Routing / Learned Indexers

A lightweight learned mechanism can predict which portions of the interaction space are most likely to require exact evaluation.

Possible selectors include:

* logistic regression,
* small decision trees,
* compact neural networks,
* learned hash functions,
* specialized classifiers,
* or dedicated learned attention indexers.

Learned selection is already a strong existing paradigm and should therefore be treated as an important baseline rather than merely a speculative alternative.

However, any selector still introduces additional computational and memory cost.

Therefore:

$$
C_{\mathrm{router}} \ll C_{\mathrm{avoided}}
$$

should remain an explicit systems objective, where:

* $C_{\mathrm{router}}$ is the total cost of selection, including model execution, metadata access, and indexing where applicable.
* $C_{\mathrm{avoided}}$ is the attention work and memory traffic eliminated by the resulting selection decisions.

Existing learned indexers demonstrate that lightweight selection can be highly effective, but selector and indexer cost can itself become a scaling concern as context length increases.

The research question is therefore not whether learned routing is inherently preferable or inferior to geometric bounds.

Instead:

> **Under what conditions can training-free geometric or algorithmic selection compete with learned indexers in accuracy, overhead, scaling behavior, and hardware efficiency?**

Learned selection should consequently be treated as both a competing baseline and a possible hybrid component.

---

# 14. Priority 11 — Online Algorithms and Predictive Caching

Transformer inference may also be considered an online computation problem in which the information required at the next inference step is not known with certainty in advance.

Recent interaction history could be used to predict which information is most likely to be required next.

This connects the research to classical:

- paging,
- cache replacement,
- LRU,
- LFU,
- working-set models,
- predictive caching,
- locality analysis,
- and amortized analysis.

Context could conceptually be divided into "pages."

Frequently or recently relevant context could remain hot, while lower-probability information remains cold until required.

This direction could be particularly useful for the conversational ChurnBot system, where repeated discussion of the same customer, prediction, feature, or explanation may create strong temporal locality.

The research question is whether predictable locality in conversational inference can reduce repeated context retrieval or computation enough to produce measurable amortized savings.

---

# 15. Priority 12 — Constraint Satisfaction and Formal Methods

Because ChurnBot operates within a narrow domain, some natural-language processing may not require unrestricted generative reasoning.

Parts of the system could instead use:

* finite-state machines,
* formal grammars,
* regular languages,
* parsing,
* constraint satisfaction,
* symbolic execution,
* rule engines,
* compiler-style intermediate representations,
* type systems,
* and formal verification.

For example:

```text
User Language
      ↓
Parser
      ↓
Structured Intent Representation
      ↓
Validated Query
      ↓
GLASS Prediction / Trace
      ↓
Constrained Natural-Language Explanation
```

This architecture could reduce the risk of generating predictions, explanations, or claims that are not supported by the authoritative GLASS record.

The research question is whether a constrained symbolic layer can handle a substantial fraction of domain-specific conversational requests reliably enough to reduce dependence on unrestricted generative inference

---

# 16. Compiler-Inspired Conversational Architecture

One possible ChurnBot architecture is to treat natural-language interaction similarly to compilation.

```text
Natural Language
       ↓
Lexical / Semantic Analysis
       ↓
Intent Parser
       ↓
Intermediate Representation (IR)
       ↓
Query Planner
       ↓
GLASS / Database / Knowledge System
       ↓
Verified Result
       ↓
Natural-Language Renderer
```

Instead of requiring one enormous model to perform every operation, specialized stages could solve smaller deterministic problems.

This architecture could combine:

* symbolic parsing,
* small classifiers,
* retrieval,
* caching,
* deterministic business logic,
* constrained generation,
* and a small language model only when necessary.

The research question is whether decomposing conversational inference into specialized compiler-like stages can reduce computational requirements while improving traceability and control compared with a monolithic language model.

---

# 17. Escalation Architecture

The system does not necessarily need to choose between:

```text
Rules
```

and:

```text
Large Language Model
```

A cascade could instead be used:

```text
Formal Parser
      ↓
Can solve?
  YES → Answer

  NO
      ↓
Small Specialized Model
      ↓
Can solve?
  YES → Answer

  NO
      ↓
Advanced Local Model
      ↓
Can solve?
  YES → Answer

  NO
      ↓
Abstain / Escalate
```
Escalation between stages would be governed by explicit confidence, coverage, validation, or parsing criteria rather than by an unrestricted model choosing when to escalate.

Most routine queries could therefore be resolved using inexpensive deterministic or specialized mechanisms, while more computationally expensive inference would occur only when simpler stages cannot produce a sufficiently reliable result.

The research question is whether this staged architecture can resolve a large proportion of domain-specific interactions at substantially lower average computational cost than routing every request directly through a general-purpose language model.

---

# 18. Relationship to GLASS

The conversational layer may retrieve, translate, summarize, and explain information contained within the authoritative prediction and auditable decision trace.

It must **not silently replace, modify, reinterpret, or invent the underlying prediction**.

When the available GLASS record does not contain sufficient information to support an explanation, the conversational system should explicitly acknowledge that limitation or abstain rather than generate an unsupported conclusion.

The architecture should maintain:

```text
GLASS Cascade
     ↓
Authoritative Prediction
     ↓
Auditable Decision Trace
     ↓
Conversational Layer
     ↓
Human-Readable Explanation
```

This preserves the central ChurnBot principles of:

* interpretability,
* auditability,
* reproducibility,
* traceability,
* uncertainty awareness,
* and controlled abstention.

---

# 19. Experimental Program

Every proposed optimization should be evaluated against dense attention under controlled experimental conditions.

## Baselines

For sequence length $n$, dense self-attention requires pairwise query-key interactions whose computational growth is quadratic in $n$ when the attention dimension is treated as fixed:

$$
T_{\mathrm{dense}}(n) = \Theta(n^2)
$$

More explicitly, if the query-key dimension $d$ is included:

$$
T_{\mathrm{dense}}(n,d) = \Theta(n^2 d)
$$

The experimental baselines should include:

- a standard dense-attention implementation,
- an optimized dense-attention implementation where applicable,
- and relevant existing efficient-attention methods identified during the literature review.

All comparisons should use equivalent models, hardware, numerical precision, sequence lengths, and evaluation workloads whenever possible.

## Measurements

Experiments should evaluate four major categories.

### Computational Performance

- wall-clock latency,
- CPU execution time,
- GPU execution time where applicable,
- FLOPs or equivalent operation counts,
- throughput,
- memory consumption,
- memory bandwidth,
- cache behavior,
- and energy consumption where measurable.

### Pruning and Search Behavior

- percentage of attention interactions avoided,
- percentage of candidate regions rejected before exact evaluation,
- number of regions explored,
- search depth,
- pruning or routing overhead,
- index-construction overhead where applicable,
- and retained attention computation.

### Model Quality

- task accuracy,
- perplexity where applicable,
- approximation error,
- output similarity,
- and task-specific quality metrics.

### Scaling Behavior

Measurements should be repeated across increasing sequence lengths to estimate:

- empirical growth rate,
- best-case behavior,
- average-case behavior,
- worst-case behavior,
- and the conditions under which pruning succeeds or collapses toward dense computation.

The primary research objective is not merely to reduce the number of evaluated interactions, but to determine whether total end-to-end computation grows substantially more slowly than dense attention while preserving acceptable model quality.

---

# 20. Most Important Metric: Effective Work Avoided

A high pruning percentage alone is not sufficient.

For example, an algorithm that eliminates 80% of attention interactions but spends nearly the same amount of computation determining which interactions to eliminate has achieved little practical improvement.

Define the effective computational cost of the proposed method as:

$$
C_{\mathrm{effective}}
=
C_{\mathrm{pruning}}
+
C_{\mathrm{retained}}
$$

where:

- $C_{\mathrm{pruning}}$ includes all work required to search, bound, route, index, partition, or otherwise determine which computations can be avoided.
- $C_{\mathrm{retained}}$ is the cost of performing the attention computation that remains after pruning.
- $C_{\mathrm{dense}}$ is the cost of the corresponding dense-attention baseline.

The absolute computational savings are therefore:

$$
C_{\mathrm{savings}}
=
C_{\mathrm{dense}}
-
C_{\mathrm{effective}}
$$

The normalized fraction of work avoided is:

$$
S_{\mathrm{relative}}
=
1 -
\frac{C_{\mathrm{effective}}}{C_{\mathrm{dense}}}
$$

and computational speedup is:

$$
\mathrm{Speedup}
=
\frac{C_{\mathrm{dense}}}
{C_{\mathrm{effective}}}
$$

A successful pruning strategy therefore requires:

$$
C_{\mathrm{effective}} < C_{\mathrm{dense}}
$$

with the stronger practical objective:

$$
C_{\mathrm{effective}} \ll C_{\mathrm{dense}}
$$

while maintaining model quality within a predefined acceptable tolerance.

The central experimental question is therefore not simply **"How much attention was pruned?"** but rather:

> **How much total computational work was actually avoided after accounting for every cost required to decide what not to compute?**

---

# 21. Instance-Dependent Complexity and Empirical Average-Case Scaling

The research does not require eliminating the quadratic worst case.

Some inputs may genuinely require examination of most possible query-key relationships. Therefore:

$$
T_{\mathrm{worst}}(n)=O(n^2)
$$

may remain possible.

The theoretical target is better expressed using **instance-dependent or output-sensitive complexity**, where computational cost depends on measurable structural properties of the particular attention instance.

For a query $q$, define:

* $k_{\varepsilon}(q)$ as the number of retained keys required to capture a specified fraction of the relevant attention mass under tolerance $\varepsilon$.
* $N_{\mathrm{vis}}(q)$ as the number of hierarchy or search nodes visited, including nodes that are ultimately rejected.
* $C_{\mathrm{maint}}$ as the cost of maintaining any search structure or index as the KV cache changes.

For decode, a candidate cost model is:

$$
T(q)
=
O\left(
d\left[
k_{\varepsilon}(q)
+
N_{\mathrm{vis}}(q)
\right]
+
C_{\mathrm{maint}}
\right)
$$

where the exact constants and lower-order terms depend on the bound, index structure, memory layout, and hardware implementation.

Across $n$ decode steps:

$$
T_{\mathrm{decode}}(n)
=
O\left(
d
\sum_{t=1}^{n}
\left[
k_{\varepsilon}(q_t)
+
N_{\mathrm{vis}}(q_t)
\right]
+
\sum_{t=1}^{n}
C_{\mathrm{maint}}(t)
\right)
$$

If the average retained support and number of visited regions remain approximately bounded as context grows, the resulting behavior may approach linear scaling in $n$ when $d$ is treated as fixed.

If these quantities grow approximately logarithmically, behavior closer to:

$$
O(n\log n)
$$

may still represent a substantial improvement over dense quadratic attention.

For prefill, an analogous analysis would require hierarchical treatment of both query and key sets and should be formulated separately if prefill becomes the selected thesis setting.

Alongside this theoretical analysis, experiments should measure **empirical average-case scaling** across explicitly defined workloads.

The research will therefore distinguish between:

* conditional worst-case complexity,
* instance-dependent or output-sensitive theoretical complexity,
* empirical average-case behavior across representative workloads,
* and actual end-to-end systems performance.

The central hypothesis is that real Transformer activations may possess structural properties — such as limited effective support, favorable score separation, or sufficiently low intrinsic geometric complexity — that permit hierarchical search to visit only a small fraction of the possible interaction space.

Whether those properties actually hold is an experimental question.

---

# 22. Failure Conditions

The hypothesis should be considered unsupported or unsuccessful under the tested conditions if any of the following consistently occur:

* pruning, routing, indexing, or search overhead eliminates the computational savings,
* important interactions cannot be bounded or estimated cheaply without effectively performing the exact computation being avoided,
* model quality degrades beyond a predefined acceptable tolerance,
* the average number of retained interactions per query grows proportionally with sequence length, causing total computation to approach quadratic growth,
* pruning effectiveness deteriorates substantially as sequence length increases,
* apparent improvements occur only on artificial or unusually favorable inputs,
* results fail to generalize across representative workloads,
* specialized hardware behavior eliminates theoretical computational gains,
* memory movement or indexing costs dominate despite reductions in arithmetic operations,
* preprocessing costs outweigh or substantially reduce inference-time savings,
* reductions in FLOPs fail to produce meaningful end-to-end latency or energy improvements,
* or the method merely relocates quadratic computation into another stage of the system.

These outcomes would still constitute scientifically useful findings because they would identify the conditions and structural limitations under which near-linear average-case Transformer inference is or is not feasible.

---

# 23. Scientific Position

This research deliberately takes **no predetermined position** on whether dense quadratic attention is necessary.

The research question is not:

> "How do we prove Transformers should be linear?"

It is:

> **"Under what conditions, if any, can the computational requirements of Transformer inference be reduced toward near-linear average-case behavior without unacceptable degradation in model capability?"**

The experimental and theoretical evidence will determine the answer.

Possible conclusions include:

1. near-linear average-case inference is feasible under broad realistic workloads;
2. near-linear behavior is feasible only under specific structural or workload assumptions;
3. meaningful sub-quadratic improvement is feasible, but near-linear behavior is not;
4. improvements exist only for particular model architectures, sequence regimes, or tasks;
5. computational savings are offset by pruning, routing, indexing, or search overhead;
6. important interactions cannot be identified reliably without performing enough computation to eliminate the expected advantage;
7. theoretical arithmetic reductions do not translate into practical systems-level improvements;
8. or dense quadratic attention remains competitive or necessary under the conditions tested.

All of these outcomes are scientifically meaningful.

The purpose of the research is therefore not to defend a preferred computational model, but to experimentally and mathematically determine **where the boundary lies between computation that is genuinely necessary and computation that can be avoided before it occurs**.


# 24. Novelty and Prior-Work Requirement

No claim of novelty should be made until a comprehensive literature review and prior-art analysis have been completed.

Substantial prior work already exists in areas including:

* sparse and block-sparse attention,
* hierarchical attention and coarse-to-fine selection,
* region-level score bounds,
* approximate nearest-neighbor and MIPS retrieval,
* randomized and sampling-based attention,
* low-rank and sparse decompositions,
* multiresolution attention,
* graph-based sparsification,
* learned indexers and sparse selection,
* KV-cache paging and retrieval,
* hardware-aware attention,
* and efficient Transformer architectures more broadly.

Representative related systems and methods identified for detailed comparison include work such as Quest, Squeezed Attention, Fast Multipole or multiresolution attention methods, RetrievalAttention, MagicPIG, vAttention, learned sparse indexers, sparse-plus-low-rank methods, and hierarchical KV-selection systems.

These methods must be documented and cited properly in the final literature review.

The following ideas are therefore **not claimed as novel by themselves**:

* hierarchical decomposition,
* pruning entire regions before exact evaluation,
* min/max or centroid-based region summaries,
* ANN/MIPS retrieval,
* graph sparsification,
* low-rank plus sparse decomposition,
* randomized sampling,
* learned routing,
* or KV-cache paging.

The candidate research gap is narrower.

A potentially distinct direction is:

> **training-free hierarchical bounded search using a deterministic, size-aware aggregate contribution criterion, combined with adaptive eliminate/approximate/expand traversal and an instance-dependent complexity analysis parameterized by measurable properties of real Transformer activations.**

Potentially relevant structural parameters include:

* effective support $k_{\varepsilon}$,
* number of hierarchy nodes visited,
* intrinsic or doubling dimension,
* score-gap behavior,
* bound tightness,
* and the prevalence of attention patterns for which deterministic pruning fails.

Whether this combination is genuinely novel remains an open literature-review question.

The research contribution, if one exists, may ultimately consist of:

* a new bound,
* a new traversal strategy,
* a new combination of established mechanisms,
* a stronger complexity analysis,
* a systems implementation,
* a characterization of real Transformer activation geometry,
* or a rigorous negative result identifying why the proposed approach fails.

A prior-art matrix should therefore be maintained throughout the project, comparing relevant methods by dimensions such as:

* training-free vs. trained,
* prefill vs. decode,
* selection granularity,
* bound or estimator type,
* approximation guarantee,
* hardware target,
* index-maintenance requirements,
* and computational complexity.

The purpose of this matrix is to identify genuine research gaps and appropriate baselines, not to assume novelty in advance.

---

# 25. Initial Research Order

The research sequence should prioritize inexpensive measurements capable of invalidating the hypothesis before substantial implementation work begins.

```text
0. Profile Representative Target Workloads
   - context lengths
   - prefill vs. decode costs
   - attention share of latency and memory traffic
   - include ChurnBot when a representative conversational workload exists

        ↓

1. Complete Literature Review / Prior-Art Matrix
   - identify nearest existing methods
   - establish appropriate baselines
   - verify whether a technically meaningful research gap remains

        ↓

2. Formalize the Research Setting
   - prefill, decode, or separate treatment of both
   - training-free scope
   - error metric
   - admissibility requirements
   - exact definition of the pruning / approximation budget

        ↓

3. Structural Measurement Study on Real Transformer Activations
   - effective support k_epsilon
   - score-gap distributions
   - bound tightness
   - intrinsic / doubling dimension where meaningful
   - behavior across layers and heads
   - effect of attention-sink or near-uniform heads

        ↓

4. Establish Strong Dense Baselines
   - optimized dense attention
   - CPU baseline where relevant
   - GPU / FlashAttention-class baseline where relevant

        ↓

5. Establish Nearest Efficient-Attention Baselines
   - hierarchical pruning
   - sampling-based methods
   - ANN / retrieval methods
   - learned indexers
   - other methods identified by the prior-art matrix

        ↓

6. Implement the Hierarchical Bounded-Search Prototype
   - hierarchical partitioning
   - admissible region bounds
   - best-first or bounded traversal
   - eliminate / approximate / expand decisions
   - explicit error-budget accounting

        ↓

7. Measure Search and Index Overhead
   - nodes visited
   - metadata accessed
   - KV bytes loaded
   - index construction and maintenance
   - pruning / selection latency

        ↓

8. Perform Scaling and Failure-Mode Analysis
   - increasing context lengths
   - per-layer and per-head behavior
   - conditions under which search approaches dense computation

        ↓

9. Investigate Hybrids Only If Justified
   - randomized fallback
   - sparse + low-rank
   - ANN / MIPS
   - learned selection
   - caching or cross-layer reuse

        ↓

10. End-to-End Systems Evaluation
    - latency
    - memory traffic
    - throughput
    - energy where measurable
    - model quality
    - empirical scaling
```

Methods should earn additional algorithmic complexity only when simpler mechanisms fail to provide sufficient computational savings or model quality.

This ordering deliberately places **literature review and structural measurement before algorithm development**.

If the measurements show that real Transformer activations do not possess the structural properties required for efficient hierarchical search, that result should redirect or terminate the branch-and-bound direction before substantial engineering effort is spent.

Such a negative finding would itself be scientifically meaningful.

---

# 26. Long-Term Objective

The ultimate objective is broader than accelerating a single benchmark.

The research asks whether principles from classical computer science can be used to redesign modern neural inference so that capable AI systems require substantially fewer computational resources.

If successful, this research could contribute toward:

* capable local AI,
* CPU-friendly inference,
* reduced dependence on high-end GPUs,
* lower memory requirements,
* lower inference energy consumption,
* longer usable context windows,
* reduced deployment costs,
* and more accessible AI systems.

For ChurnBot specifically, the long-term objective is a conversational system capable of operating locally on ordinary personal hardware while preserving the GLASS project's core principles of transparency, verification, auditability, evidence grounding, controlled abstention, and computational efficiency.

The broader research objective is to determine whether computational efficiency can be achieved not merely through faster hardware, but through algorithms that avoid unnecessary work before it occurs.

---

# 27. Current Status

This document describes **proposed future research only**.

The current ChurnBot research priority remains:

> **Evaluating interpretable GLASS models and cascades against comparable black-box machine-learning systems under controlled experimental conditions.**

The Transformer-inference and conversational-computing research described in this document represents a possible future thesis and research direction.

No claim is currently made that the proposed methods will achieve near-linear average-case inference, that the proposed combination is novel, or that the approach is technically feasible.

Those questions are the subject of the proposed research.

The purpose of this document is to establish the research questions, candidate algorithmic directions, experimental criteria, and scientific constraints that will guide future investigation.
