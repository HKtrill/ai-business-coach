# Potential Sources: Efficient Attention via Classical CS Algorithms

**For:** ChurnBot / GLASS Future Research Roadmap — Track A (attention inference) and Track B (conversational layer)
**Compiled:** August 2026
**Status:** Working source base. Verify every entry marked ⚠ before citing.

> **Disclaimer:**  This is a working list of candidate sources, not a verified bibliography. Entries were assembled with AI assistance, preliminary literature searches, and prior research notes. Author lists, publication venues, dates, DOIs, arXiv identifiers, and other bibliographic details have not all been independently verified against the original sources. Entries marked ⚠ require particular verification. All sources will be checked against the original publication before being cited in formal research. This list will continue to evolve as the literature review progresses.

---

## Conventions

Each entry uses the following fields:

- **Citation** — title, authors, year, venue, identifier.
- **Summary** — what the work does and its main contribution.
- **Relevance** — why it matters to this roadmap.
- **Supports** — which roadmap topics it backs (using the topic labels below).
- **Class** — Foundational background · Closest prior art · Competing baseline · Theoretical constraint · Supporting technique · Systems/hardware evidence.
- **Cite** — *Roadmap now* · *Thesis later* · *Background only*.
- **Forces revision** — claims in the current document this source contradicts or qualifies. "None" if it doesn't.
- **Overlap** — which entry is primary when several overlap.

Topic labels: `dense-complexity`, `branch-and-bound`, `hierarchical-pruning`, `graph-attention`, `ann-mips`, `sparse-lowrank`, `multiresolution`, `sampling`, `learned-routing`, `kv-paging`, `lower-bounds`, `instance-dependent`, `cascade-grounding`, `baseline`.

Roadmap section references (§) use the *current* numbering of the document; after the planned merge, §4/§6/§7/§11 become "Direction A."

**⚠ verify** marks any field I could not confirm from memory or from the search results gathered during the review. Do not cite those fields without checking the arXiv page or DOI.

---

## Group A — Core Attention Complexity and Lower Bounds

### [A1] Attention Is All You Need
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). *NeurIPS 2017.* arXiv:1706.03762.

- **Summary:** Introduces the Transformer and scaled dot-product attention, softmax(QKᵀ/√d)V, whose cost is Θ(n²d) per head for sequence length n. This is the operation every later entry tries to accelerate.
- **Relevance:** Defines the baseline object; §2 and §19 should cite it for the Θ(n²d) statement.
- **Supports:** `dense-complexity`, §2, §19.
- **Class:** Foundational background. **Cite:** Roadmap now.
- **Forces revision:** None.

### [A2] On the Computational Complexity of Self-Attention
Duman Keles, F., Wijewardena, P. M., Hegde, C. (2023). *ALT 2023 (PMLR 201).* arXiv:2209.04881.

- **Summary:** Proves, under the Strong Exponential Time Hypothesis, that *exact* self-attention cannot be computed in truly subquadratic time, via reductions from Orthogonal Vectors. Gives matching results for several attention variants.
- **Relevance:** Establishes that the O(n²) worst case the proposal "allows" is in fact required for exact computation. The proposal's hypothesis must therefore be about approximation and about the data regime, not about exact algorithms.
- **Supports:** `lower-bounds`, §2, §21, §23.
- **Class:** Theoretical constraint. **Cite:** Roadmap now.
- **Forces revision:** §2 and §21 should state that quadratic worst-case is a theorem (conditional on SETH), not a design concession.
- **Overlap:** Companion to [A3]; cite both — [A2] for exact, [A3] for approximate.

### [A3] Fast Attention Requires Bounded Entries
Alman, J., Song, Z. (2023). *NeurIPS 2023.* arXiv:2302.13214.

- **Summary:** Shows a sharp transition for *approximate* attention when d = Θ(log n): if |Q|,|K| entries are bounded by B = o(√log n), attention is computable in n^{1+o(1)} time (via the polynomial method); if B = Ω(√log n), no truly subquadratic algorithm exists under SETH. Equivalently, subquadratic attention is possible iff softmax is applied at high temperature.
- **Relevance:** The single most important theoretical constraint on the roadmap. Any near-linear instance-dependent claim is implicitly a claim that real Q/K live in a "bounded-entry-like" regime; the proposal should say so explicitly and measure it.
- **Supports:** `lower-bounds`, `instance-dependent`, §2, §21, §22, §23.
- **Class:** Theoretical constraint. **Cite:** Roadmap now.
- **Forces revision:** §21's "average-case" hypothesis must be reframed as a data-regime claim; §22 should add "score magnitudes on real inputs exceed the tractable regime" as a failure condition.
- **Overlap:** Primary. [A4] extends it; [A5] generalizes the hardness to non-Transformer alternatives.

### [A4] Subquadratic Algorithms and Hardness for Attention with Any Temperature
Gupta, S., Huang, B., Saha, B., Xu, Y., Ye, C. (2025). *ICLR 2026.* arXiv:2505.14840. **⚠ verify author list.**

- **Summary:** Extends [A3] to arbitrary temperatures: characterizes when attention is computable in truly subquadratic time as a function of head dimension d (constant vs. polynomial in n), using a geometric "relevant region" argument for constant d and proving the standard algorithm optimal for d = poly(n).
- **Relevance:** Provides the most current statement of the tractability frontier; the constant-d algorithms are themselves geometric-search algorithms, which supports the roadmap's premise that computational geometry applies.
- **Supports:** `lower-bounds`, §21.
- **Class:** Theoretical constraint. **Cite:** Thesis later (roadmap may cite alongside [A3]).
- **Forces revision:** None beyond [A3].
- **Overlap:** Secondary to [A3].

### [A5] Fundamental Limitations on Subquadratic Alternatives to Transformers
Alman, J., Yu, H. (2024). *ICLR 2025.* arXiv:2410.04271. **⚠ verify venue.**

- **Summary:** Proves that *any* architecture running in subquadratic time (not just Transformer variants) cannot solve certain document-similarity tasks that standard attention solves, under SETH.
- **Relevance:** Cuts off the escape route of "replace attention entirely"; supports the roadmap's choice to keep softmax attention and reduce its cost on typical instances rather than change the model class.
- **Supports:** `lower-bounds`, §23, §26.
- **Class:** Theoretical constraint. **Cite:** Thesis later.
- **Forces revision:** None.

### [A6] The Fine-Grained Complexity of Gradient Computation for Training Large Language Models
Alman, J., Song, Z. (2024). arXiv:2402.04497. **⚠ verify venue (NeurIPS 2024?).**

- **Summary:** Extends the bounded-entry dichotomy from inference to the backward pass.
- **Relevance:** Only relevant if the thesis ever touches training. Include for completeness of the Alman–Song line.
- **Supports:** `lower-bounds`.
- **Class:** Theoretical constraint. **Cite:** Background only.
- **Forces revision:** None.

### [A7] Support Basis: Fast Attention Beyond Bounded Entries
Aliakbarpour, M., Braverman, V., Yin, J., Zhang, H. (2025). arXiv:2510.01643. **⚠ verify venue.**

- **Summary:** Empirically shows Q/K entries are sub-Gaussian and uses this to split large and small entries — exact computation on the sparse large-entry component, polynomial approximation on the dense small-entry component — obtaining subquadratic attention without the bounded-entry assumption.
- **Relevance:** This is a theoretically-motivated sparse+low-rank split driven by the *distribution* of real activations, which is exactly the instance-dependent stance the roadmap should adopt. Also a modern statement of the A ≈ L + S idea in §9.
- **Supports:** `lower-bounds`, `sparse-lowrank`, `instance-dependent`, §9, §21.
- **Class:** Closest prior art (for the instance-dependent framing). **Cite:** Thesis later.
- **Forces revision:** §9 should cite this as evidence that the sparse/dense split can be justified from measured activation statistics rather than assumed.

### [A8] Efficient Transformers: A Survey
Tay, Y., Dehghani, M., Bahri, D., Metzler, D. (2022). *ACM Computing Surveys 55(6).* arXiv:2009.06732.

- **Summary:** Taxonomy of efficient-attention methods circa 2020–2022: fixed patterns, learnable patterns, low-rank, kernel, recurrence, memory, downsampling.
- **Relevance:** The organizing vocabulary for §24.1's prior-art matrix. Dated (pre-FlashAttention, pre-KV-cache-sparsity era) but still the standard taxonomy citation.
- **Supports:** §24.
- **Class:** Foundational background. **Cite:** Roadmap now.
- **Forces revision:** None.

### [A9] Long Range Arena: A Benchmark for Efficient Transformers
Tay, Y., Dehghani, M., Abnar, S., Shen, Y., Bahri, D., Pham, P., Rao, J., Yang, L., Ruder, S., Metzler, D. (2021). *ICLR 2021.* arXiv:2011.04006.

- **Summary:** Benchmark suite for long-sequence models; showed that many efficient-attention variants trade accuracy for speed and that no single method dominated.
- **Relevance:** Historical caution for §22–23: a method that wins on synthetic long-range tasks may lose on real ones. Also the benchmark used by [D1]–[D3].
- **Supports:** §19, §22.
- **Class:** Competing baseline (benchmark). **Cite:** Thesis later.
- **Forces revision:** None.

### [A10] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. (2022). *NeurIPS 2022.* arXiv:2205.14135.

- **Summary:** Computes exact attention with tiling and online softmax so that the n×n matrix is never materialized in HBM; reduces memory traffic from O(n²) to O(n²d²/M) and gives large wall-clock speedups without changing FLOPs.
- **Summary (cont.):** Demonstrates that the binding constraint is memory movement, not arithmetic.
- **Relevance:** This is the dense baseline. Every speedup in the roadmap must be measured against a FlashAttention-class kernel, not naive dense. Its online-softmax running max is also precisely the "incumbent" the branch-and-bound formulation needs, so Direction A can be threaded into this kernel structure.
- **Supports:** `baseline`, `dense-complexity`, §3, §19, §20.
- **Class:** Competing baseline / Systems evidence. **Cite:** Roadmap now.
- **Forces revision:** §19 "standard dense-attention implementation" must be replaced by FlashAttention-class dense; §20 must account cost in bytes moved, not only FLOPs.
- **Overlap:** Primary. [A11]–[A12] are successors; cite the version you actually benchmark against.

### [A11] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
Dao, T. (2024). *ICLR 2024.* arXiv:2307.08691.

- **Summary:** Reworks parallelism and warp partitioning for ~2× over FlashAttention.
- **Relevance:** Likely the practical GPU baseline; for CPU work, cite [A10] and use the best available CPU kernel.
- **Supports:** `baseline`, §19. **Class:** Competing baseline. **Cite:** Thesis later. **Forces revision:** None. **Overlap:** Secondary to [A10].

### [A12] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision
Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. (2024). *NeurIPS 2024.* arXiv:2407.08608. **⚠ verify author list.**

- **Summary:** Hopper-specific asynchrony and FP8 support.
- **Relevance:** Only if benchmarking on H100-class hardware; otherwise background.
- **Supports:** `baseline`. **Class:** Competing baseline. **Cite:** Background only. **Overlap:** Secondary to [A10].

### [A13] Roofline: An Insightful Visual Performance Model for Multicore Architectures
Williams, S., Waterman, A., Patterson, D. (2009). *Communications of the ACM 52(4), 65–76.*

- **Summary:** The roofline model bounds attainable performance by min(peak compute, bandwidth × arithmetic intensity).
- **Relevance:** The correct framework for §6/§19/§20 hidden-cost analysis: decode attention has arithmetic intensity proportional to the number of queries per KV head and is bandwidth-bound; prefill is compute-bound. The roadmap's cost model should be expressed in roofline terms.
- **Supports:** §3, §19, §20, §22.
- **Class:** Foundational background / Systems evidence. **Cite:** Roadmap now.
- **Forces revision:** §20's cost model must add a bandwidth term.

---

## Group B — Hierarchical / Branch-and-Bound / Region-Pruning Methods

### [B1] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., Han, S. (2024). *ICML 2024.* arXiv:2406.10774. Code: github.com/mit-han-lab/Quest.

- **Summary:** Partitions the KV cache into pages of 16 tokens; for each page stores element-wise min and max key vectors; for a query q, computes an upper bound on the maximum dot product in the page as Σᵢ max(qᵢ·kminᵢ, qᵢ·kmaxᵢ); loads only the top-K pages by that bound and runs exact attention on them. Reports up to 2.23× self-attention speedup and 7.03× latency reduction at 32K context with negligible accuracy loss. Gives an explicit cost identity: fraction of KV loaded = 1/pagesize + K/pagecount.
- **Relevance:** The closest single prior-art match for §4. The min/max box bound *is* an admissible branch-and-bound bound; Quest is a one-level, top-K-ranked instance of Direction A. The cost identity is the template for §20.
- **Supports:** `branch-and-bound`, `hierarchical-pruning`, `kv-paging`, §4, §20, §24.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §4's claim that region bounds are a research direction must be rewritten to acknowledge Quest's bound and locate the gap in (a) size-aware aggregate bounds, (b) hierarchy, (c) certified elimination rather than top-K ranking. §24's "candidate contribution" sentence is directly contradicted.
- **Overlap:** Primary for box bounds. [B2] adds the hierarchy; [B5] adds the exact-support variant.

### [B2] Squeezed Attention: Accelerating Long Context Length LLM Inference
Hooper, C., Kim, S., Mohammadzadeh, H., Maheswaran, M., Zhao, S., Paik, J., Mahoney, M. W., Keutzer, K., Gholami, A. (2025). *ACL 2025 (Long Papers).* arXiv:2411.09688.

- **Summary:** For a *fixed* context (e.g., a long system prompt or document), clusters keys offline with k-means and represents each cluster by a centroid; at inference compares the query against centroids to select semantically relevant keys and computes exact attention only on them. A hierarchical variant clusters centroids into coarser centroids, reducing lookup from linear to logarithmic in the fixed-context length. Reports 3.1× KV-budget reduction with no measurable loss on LongBench.
- **Relevance:** This is §4 + §7 in assembled form: hierarchical decomposition by content clustering, centroid-based region scoring, prune coarse regions before examining fine ones. Its fixed-context assumption matches the ChurnBot shape (a large GLASS trace as fixed context, many short queries).
- **Supports:** `hierarchical-pruning`, `branch-and-bound`, §4, §7, §24; also relevant to Track B workload characterization.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §7's hierarchical clustering is prior work; §24 must cite. Note that its centroid comparison is a *heuristic score*, not an admissible bound — a genuine distinction the thesis can exploit (centroid + radius gives an admissible ball bound).
- **Overlap:** Primary for hierarchical content clustering. [B3] extends it with multipole approximation.

### [B3] Multipole Attention for Efficient Long Context Reasoning
Hooper, C., et al. (2025). arXiv:2506.13059. **⚠ verify full author list and venue.**

- **Summary:** Extends [B2] to reasoning workloads where the KV cache grows during generation: uses hierarchical k-means to produce progressively coarser centroids and, rather than dropping non-selected clusters, *approximates* their contribution via the centroid (a multipole-style far-field approximation), so no key is entirely ignored. Addresses the centroid-lookup bottleneck as context grows.
- **Relevance:** The closest existing implementation of "reject-or-approximate" over a hierarchy on a pretrained model's KV cache — i.e., the corrected §4 decision rule. Also handles incremental growth, which [B2] does not.
- **Supports:** `hierarchical-pruning`, `multiresolution`, §4, §10, §11, §24.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §11's "coarse-to-fine with far-field approximation" is prior work in the training-free KV-cache setting; §10's incremental-maintenance question is partially addressed here.
- **Overlap:** Secondary to [B2] for the base method; primary for approximate-not-drop.

### [B4] HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention
(2026). arXiv:2603.28458. **⚠ verify author list and venue.**

- **Summary:** Training-free, drop-in replacement for the token-level indexer in DSA-style models ([F1]): a block-level coarse filter scores pooled block representations to discard irrelevant regions, then the original indexer runs only within retained blocks. Reports up to 3.75× indexer-kernel speedup while preserving token-level selection. Explicitly argues the indexer had become "the new quadratic bottleneck."
- **Relevance:** Confirms two roadmap premises — that the selection mechanism's own cost is the real constraint (§3, §13, §20) and that hierarchical filtering is how production systems address it — while also showing the idea is already deployed. Pooled block keys are a (non-admissible) analogue of the box bound.
- **Supports:** `hierarchical-pruning`, `learned-routing`, §3, §4, §13, §20, §24.
- **Class:** Closest prior art / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §13's assumption that learned routers are the expensive option; §20's cost-of-deciding argument now has a production case study.
- **Overlap:** Primary for hierarchical indexing over learned indexers; complements [B1]–[B2] which are training-free throughout.

### [B5] EntmaxKV: Support-Aware Decoding for Entmax Attention
(2026). arXiv:2605.21649. **⚠ verify author list and venue.**

- **Summary:** Uses α-entmax attention (finite support) with a paged KV cache; each page stores coordinate-wise min/max keys and the Quest box bound is used to identify pages that cannot enter the support, enabling exact (not approximate) skipping during decoding.
- **Relevance:** Directly implements the "exact elimination under sparse-support attention" variant recommended for the revised §4. Establishes that this lever is known, so the thesis must position relative to it (e.g., certified budget, incumbent-driven best-first order, dual-tree prefill).
- **Supports:** `branch-and-bound`, `hierarchical-pruning`, §4, §24.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** Any claim that exact region elimination is unexplored.
- **Overlap:** Depends on [B1] for the bound and on [D10]–[D11] for entmax.

### [B6] MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., Yang, Y., Qiu, L. (2024). *NeurIPS 2024.* arXiv:2407.02490.

- **Summary:** Prefill-side dynamic sparsity: classifies each head offline into one of three pattern families (A-shape, vertical-slash, block-sparse), then at runtime estimates the pattern with a cheap partial computation and runs a block-sparse kernel. Reports up to 10× prefill speedup at 1M tokens.
- **Relevance:** The main *prefill* prior art for region elimination; shows that positional-block structure (not content clustering) is what current prefill methods exploit, and that head-level heterogeneity is central. If the thesis chooses prefill, this is a required baseline.
- **Supports:** `hierarchical-pruning`, `baseline`, §2 (prefill/decode distinction), §19, §22.
- **Class:** Competing baseline. **Cite:** Thesis later (roadmap: mention in §2).
- **Forces revision:** §2 must distinguish prefill and decode; §22 should add head heterogeneity.

### [B7] Sparse Transformer — Generating Long Sequences with Sparse Transformers
Child, R., Gray, S., Radford, A., Sutskever, I. (2019). arXiv:1904.10509.

- **Summary:** Fixed strided and local block-sparse attention patterns giving O(n√n) cost, trained from scratch.
- **Relevance:** Earliest block-sparse attention; establishes that block granularity is the unit at which sparsity becomes hardware-efficient. Historical citation for §4's block decomposition.
- **Supports:** `hierarchical-pruning`, §4, §24.
- **Class:** Foundational background. **Cite:** Thesis later.
- **Forces revision:** None.
- **Overlap:** With [B8]; both are fixed-pattern baselines. Cite [B7] for block-sparsity, [B8] for local+global.

### [B8] Longformer: The Long-Document Transformer
Beltagy, I., Peters, M. E., Cohan, A. (2020). arXiv:2004.05150.

- **Summary:** Sliding-window local attention plus a few global tokens, linear in n.
- **Relevance:** Data-independent baseline; the "sink + local window" pattern reappears inside [E4]'s deterministic component.
- **Supports:** `hierarchical-pruning`, §24.
- **Class:** Foundational background. **Cite:** Thesis later. **Overlap:** Secondary to [B7].

---

## Group C — ANN, MIPS, Graph, and Geometric Search

### [C1] RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval
Liu, D., Chen, M., Lu, B., Jiang, H., Han, Z., Zhang, Q., Chen, Q., Zhang, C., Ding, B., Zhang, K., Chen, C., Yang, F., Yang, Y., Qiu, L. (2024). arXiv:2409.10516. **⚠ verify author list and venue (ICLR 2025?).**

- **Summary:** Treats decode attention as approximate nearest-neighbor search over the KV cache: builds a graph-based ANN index (a RoarGraph/HNSW-style structure) on keys, kept on CPU, and retrieves ~1–3% of keys per query. Identifies the query/key distribution mismatch (queries are out-of-distribution relative to keys) as the reason off-the-shelf ANN indices underperform, and adapts the index accordingly.
- **Relevance:** The main ANN prior art for §8. The OOD observation is a structural fact about attention geometry that any tree/graph-based method in Direction A must contend with, and it is a measurable parameter for the structural study.
- **Supports:** `ann-mips`, `instance-dependent`, §8, §21, §22.
- **Class:** Closest prior art / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §8 should state that ANN attention exists and note the Q/K distribution mismatch; §22 should add index-construction cost as a failure mode (graph indices are expensive to build and update).
- **Overlap:** Primary for ANN; [C2] is the learned-hash competitor; [C3] the storage-engine successor.

### [C2] HashAttention: Semantic Sparsity for Faster Inference
Desai, A., Yang, S., Cuadron, A., Klimovic, A., Zaharia, M., Gonzalez, J. E., Stoica, I. (2024). arXiv:2412.14468. **⚠ verify author list and venue (ICML 2025?).**

- **Summary:** Learns hash functions that map queries and keys into a compact bit space so that top-k retrieval reduces to Hamming-distance operations; contrasts itself with [C1]'s graph search, which is CPU-bound due to irregular access.
- **Relevance:** Represents the "learned hashing" branch of §8 and §13. Its critique of graph-based ANN (not GPU-friendly) is directly relevant to §6's hardware-utilization risks — and inverts for a CPU-first thesis.
- **Supports:** `ann-mips`, `learned-routing`, §8, §13.
- **Class:** Competing baseline. **Cite:** Thesis later.
- **Forces revision:** None.
- **Overlap:** Secondary to [C1].

### [C3] RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference
(2025). arXiv:2505.02922. **⚠ verify author list and venue.**

- **Summary:** Treats the KV cache as a vector database with attention-aware clustering (a "wave index") and tiered CPU/GPU storage, exploiting attention sparsity for retrieval and overlapping data movement.
- **Relevance:** Systems-level successor to [C1]; relevant to §14 (paging) and to the ChurnBot local-deployment framing where the KV cache lives in host memory.
- **Supports:** `ann-mips`, `kv-paging`, §8, §14.
- **Class:** Systems/hardware evidence. **Cite:** Background only.
- **Overlap:** Secondary to [C1].

### [C4] Big Bird: Transformers for Longer Sequences
Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., Ahmed, A. (2020). *NeurIPS 2020.* arXiv:2007.14062.

- **Summary:** Explicitly frames attention as a graph and its sparsification as a graph-sparsification problem; combines random (Erdős–Rényi), local-window, and global edges to get linear cost while proving the sparse pattern is a universal approximator and Turing complete.
- **Relevance:** The primary citation for §5 — it is the paper that already did the "graph representation of attention" framing. It also demonstrates the limitation: the resulting sparsity is data-independent.
- **Supports:** `graph-attention`, §5, §24.
- **Class:** Foundational background / Closest prior art (for §5). **Cite:** Roadmap now.
- **Forces revision:** §5 must acknowledge BigBird as the origin of the graph framing and state that this framing yields data-independent patterns.
- **Overlap:** Primary for the framing; [C5] gives the sharper spectral theory.

### [C5] Exphormer: Sparse Transformers for Graphs
Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., Sinop, A. K. (2023). *ICML 2023.* arXiv:2303.06147.

- **Summary:** Replaces BigBird's Erdős–Rényi edges with d-regular expander graphs, which keep a linear edge count while guaranteeing spectral expansion (BigBird's p = Θ(1/n) choice loses expansion). Proves O(log n) layers suffice for global mixing.
- **Relevance:** If §5 is kept as more than vocabulary, this is the correct modern reference for what spectral graph theory actually buys; also makes clear that expander-based sparsity is *structural*, not query-dependent, and therefore orthogonal to Direction A.
- **Supports:** `graph-attention`, §5.
- **Class:** Supporting technique. **Cite:** Thesis later.
- **Forces revision:** §5's list of graph techniques should distinguish structural sparsification (this) from query-dependent selection (Direction A).
- **Overlap:** Secondary to [C4]; follow-up "Even Sparser Graph Transformers" (Shirzad et al., 2024, arXiv:2411.16278) is background only.

### [C6] Reformer: The Efficient Transformer
Kitaev, N., Kaiser, Ł., Levskaya, A. (2020). *ICLR 2020.* arXiv:2001.04451.

- **Summary:** Uses angular LSH to bucket queries/keys and attends only within buckets (O(n log n)); plus reversible layers for memory.
- **Relevance:** The original LSH-attention paper; the ancestor of [E2]'s use of LSH as a sampler and of §8's hashing bullet. Trained-from-scratch, so not a training-free baseline.
- **Supports:** `ann-mips`, §8, §24.
- **Class:** Foundational background. **Cite:** Thesis later.
- **Overlap:** Cite with [C7]; both are 2020 content-based-bucketing architectures.

### [C7] Efficient Content-Based Sparse Attention with Routing Transformers
Roy, A., Saffar, M., Vaswani, A., Grangier, D. (2021). *TACL 9.* arXiv:2003.05997.

- **Summary:** Online k-means clustering of queries and keys; attention restricted within clusters; O(n^{1.5}).
- **Relevance:** Ancestor of [B2]'s clustering; earliest "route by centroid" attention.
- **Supports:** `hierarchical-pruning`, `ann-mips`, §7, §8.
- **Class:** Foundational background. **Cite:** Thesis later. **Overlap:** With [C6].

### [C8] SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems
Chen, B., Medini, T., Farwell, J., Gobriel, S., Tai, C., Shrivastava, A. (2020). *MLSys 2020.* arXiv:1903.03129.

- **Summary:** Uses LSH to select a sparse subset of neurons per forward/backward pass, achieving CPU training throughput competitive with a V100 GPU on wide networks.
- **Relevance:** The strongest existing precedent for the roadmap's thesis statement in §26 — that algorithmic sparsity can substitute for hardware acceleration on CPUs. Worth citing precisely because it shows both the promise and the narrow conditions (very wide layers, extreme sparsity) under which it held.
- **Supports:** §26, `ann-mips`.
- **Class:** Systems/hardware evidence. **Cite:** Roadmap now.
- **Forces revision:** §26's local-CPU claims should be conditioned the way SLIDE's were.

### [C9] Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search
Shrivastava, A., Li, P. (2014). *NeurIPS 2014.*

- **Summary:** Shows MIPS is not directly LSH-able but becomes so after an asymmetric transformation of queries and keys; first provably sublinear MIPS.
- **Relevance:** Attention's q·k is a MIPS problem, not a nearest-neighbor problem; this is the paper that establishes the distinction §8 glosses over. Any LSH-based bound in Direction A inherits the asymmetry issue.
- **Supports:** `ann-mips`, §8.
- **Class:** Supporting technique. **Cite:** Thesis later.
- **Forces revision:** §8 should say "MIPS" consistently and note the norm-dependence.

### [C10] Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs
Malkov, Y. A., Yashunin, D. A. (2020). *IEEE TPAMI 42(4), 824–836.* arXiv:1603.09320.

- **Summary:** HNSW: multi-layer proximity graph searched greedily/best-first from coarse to fine layers; the dominant practical ANN index.
- **Relevance:** The concrete instantiation of "best-first search over a hierarchy" (§6) in the ANN world, and the index family [C1] builds on. Its insertion cost is the reason §10's incremental-maintenance question matters.
- **Supports:** `ann-mips`, `hierarchical-pruning`, §6, §8, §10.
- **Class:** Supporting technique. **Cite:** Thesis later.

### [C11] Product Quantization for Nearest Neighbor Search
Jégou, H., Douze, M., Schmid, C. (2011). *IEEE TPAMI 33(1), 117–128.*

- **Summary:** Compresses vectors into subspace codebook indices; asymmetric distance computation gives fast approximate distances from lookup tables.
- **Relevance:** PQ codes yield cheap *bounds* on inner products (via per-subspace min/max over a codebook), a candidate bound family for the revised §4 that is finer than Quest's box and cheaper than exact.
- **Supports:** `ann-mips`, `branch-and-bound`, §4, §8.
- **Class:** Supporting technique. **Cite:** Thesis later.

### [C12] Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality
Indyk, P., Motwani, R. (1998). *STOC 1998.*

- **Summary:** Introduces locality-sensitive hashing and (c, r)-approximate NN with sublinear query time.
- **Relevance:** Foundational citation for every LSH mention (§8, §12). See also Andoni & Indyk, *CACM* 51(1), 2008, for the survey form.
- **Supports:** `ann-mips`, `sampling`. **Class:** Foundational background. **Cite:** Thesis later.

### [C13] A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces
Weber, R., Schek, H.-J., Blott, S. (1998). *VLDB 1998.*

- **Summary:** Shows that tree-based partitioning indices degrade to linear scan beyond roughly 10–20 dimensions for uniformly distributed data.
- **Relevance:** The classical statement of the risk in §22 that tree search collapses; the roadmap's hope rests on *intrinsic* dimension being far below the nominal head dimension. Pair with [C14].
- **Supports:** `instance-dependent`, §21, §22.
- **Class:** Theoretical constraint. **Cite:** Roadmap now.
- **Forces revision:** §22 should add "intrinsic dimension of keys too high for tree search" as a failure condition.

### [C14] When Is "Nearest Neighbor" Meaningful?
Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U. (1999). *ICDT 1999.*

- **Summary:** Proves that under broad conditions the ratio of farthest to nearest distance tends to 1 as dimension grows, making NN search ill-posed.
- **Relevance:** Same role as [C13]; the two together justify making intrinsic-dimension measurement Stage 3 of the revised research order.
- **Supports:** `instance-dependent`, §21, §22. **Class:** Theoretical constraint. **Cite:** Thesis later. **Overlap:** Secondary to [C13].

---

## Group D — Sparse, Low-Rank, and Multiresolution Attention

### [D1] H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences
Zhu, Z., Soricut, R. (2021). *ACL-IJCNLP 2021.* arXiv:2107.11906.

- **Summary:** Imposes a hierarchical-matrix (H-matrix) structure on the attention matrix: full resolution near the diagonal, progressively coarser averaged blocks away from it; linear time and memory. +6 points average on LRA over other subquadratic methods.
- **Relevance:** The first explicit import of H-matrices into attention — a direct precedent for §11 and for the "hierarchical matrices" bullet. It is a trained architecture with a *fixed* (positional) hierarchy, which is the key distinction from Direction A's query-dependent elimination.
- **Supports:** `multiresolution`, `hierarchical-pruning`, §11, §24.
- **Class:** Closest prior art (trained-architecture branch). **Cite:** Roadmap now.
- **Forces revision:** §11 must cite as prior work; the roadmap should say whether it targets pretrained models (training-free) or new architectures — these papers are the latter.
- **Overlap:** [D1], [D2], [D3] form one lineage; [D2] is the most adaptive, [D3] the most general. Cite all three in §24; [D2] is the primary for coarse-to-fine refinement.

### [D2] Multi Resolution Analysis (MRA) for Approximate Self-Attention
Zeng, Z., Pal, S., Kline, J., Fung, G., Singh, V. (2022). *ICML 2022 (PMLR 162).* arXiv:2207.10284.

- **Summary:** Revisits wavelet-style multiresolution analysis: approximates the attention matrix at coarse resolution, then adaptively refines only where large scores appear, independent of distance to the diagonal. Reports strong LRA and language-modeling results with hardware-aware implementation choices.
- **Summary (cont.):** The adaptive refinement logic is acknowledged to complicate implementation and training.
- **Relevance:** This *is* §11's coarse-to-fine procedure. The adaptive, score-driven refinement is the closest published analogue of Direction A's "expand" decision.
- **Supports:** `multiresolution`, `branch-and-bound`, §4, §11, §24.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §11 in full.
- **Overlap:** Primary for adaptive coarse-to-fine.

### [D3] Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences
Kang, Y., Tran, G., De Sterck, H. (2023/2025). arXiv:2310.11960. **⚠ verify author list; venue uncertain (ICLR 2024 submission; later revision 2025).**

- **Summary:** Adapts the Fast Multipole Method to attention: queries/keys/values are grouped into O(log n) resolution levels with *learned* basis functions for group summaries; nearby tokens interact at full resolution, distant tokens through coarser summaries. O(n log n) or O(n) with global receptive field; outperforms [D1] and [D2] on the reported benchmarks.
- **Relevance:** The most direct existing bridge between the FMM (Group I) and attention, and the paper whose "divide-and-conquer" title matches §4's. Again a trained architecture.
- **Supports:** `multiresolution`, `branch-and-bound`, §4, §11, §24; also Group I [I5].
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §4's "divide-and-conquer" label must acknowledge this.
- **Overlap:** With FMMformer (Nguyen et al., *NeurIPS 2021*, arXiv:2108.02347 — earlier FMM-inspired attention, background only).

### [D4] Scatterbrain: Unifying Sparse and Low-Rank Attention Approximation
Chen, B., Dao, T., Winsor, E., Song, Z., Rudra, A., Ré, C. (2021). *NeurIPS 2021.* arXiv:2110.15343.

- **Summary:** Approximates attention as A ≈ L + S with a random-feature low-rank term (Performer-style) and an LSH sparse term (Reformer-style), with error analysis showing the two are complementary. Training-free application to pretrained models is demonstrated.
- **Relevance:** This is §9, exactly. Its error decomposition is also the right starting point for a §4 budget that combines approximated regions (low-rank) with exactly-evaluated regions (sparse).
- **Supports:** `sparse-lowrank`, §9, §24.
- **Class:** Closest prior art. **Cite:** Roadmap now.
- **Forces revision:** §9 must cite; the A ≈ L + S equation is theirs in this context.
- **Overlap:** Primary; [A7] is the 2025 theoretical successor; [D5]–[D7] supply the low-rank components.

### [D5] Rethinking Attention with Performers
Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, Ł., Belanger, D., Colwell, L., Weller, A. (2021). *ICLR 2021.* arXiv:2009.14794.

- **Summary:** FAVOR+: unbiased positive random features for the softmax kernel, giving linear-time attention with provable approximation.
- **Relevance:** The low-rank/kernel component of [D4]; also the canonical "randomized projection" citation for §12.
- **Supports:** `sparse-lowrank`, `sampling`, §9, §12. **Class:** Foundational background. **Cite:** Thesis later.

### [D6] Linformer: Self-Attention with Linear Complexity
Wang, S., Li, B. Z., Khabsa, M., Fang, H., Ma, H. (2020). arXiv:2006.04768.

- **Summary:** Projects keys/values to a fixed low dimension, arguing the attention matrix is approximately low-rank.
- **Relevance:** The low-rank hypothesis in its simplest form; useful as a contrast (global low-rank fails on the sparse spikes that [D4] handles).
- **Supports:** `sparse-lowrank`, §9. **Class:** Foundational background. **Cite:** Background only. **Overlap:** Secondary to [D4].

### [D7] Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention
Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V. (2021). *AAAI 2021.* arXiv:2102.03902.

- **Summary:** Nyström approximation of the softmax matrix using landmark tokens; O(n).
- **Relevance:** Landmark/centroid approximation of the *matrix* rather than of the keys — a low-rank analogue of [B2]'s centroids, relevant if the "approximate" branch of Direction A uses Nyström-style far-field summaries.
- **Supports:** `sparse-lowrank`, `multiresolution`, §9, §11. **Class:** Supporting technique. **Cite:** Thesis later.

### [D8] Loki: Low-Rank Keys for Efficient Sparse Attention
Singhania, P., Singh, S., De, S., Bhatele, A. (2024). *NeurIPS 2024.* arXiv:2406.02542. **⚠ verify author list.**

- **Summary:** Observes that keys lie in a low-dimensional subspace (measured via PCA on real models) and computes approximate scores in that subspace to select top-k keys, then exact attention on the selection.
- **Relevance:** Direct empirical evidence for the low-intrinsic-dimension assumption that Direction A's tree bounds require; provides a measurement methodology the structural study (Stage 3) can reuse.
- **Supports:** `sparse-lowrank`, `instance-dependent`, §9, §21, §22.
- **Class:** Supporting technique / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §21 should cite as evidence that keys have low-dimensional structure — while noting that low *linear* rank is not the same as low *doubling* dimension.

### [D9] From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
Martins, A. F. T., Astudillo, R. F. (2016). *ICML 2016.* arXiv:1602.02068.

- **Summary:** Sparsemax: Euclidean projection onto the simplex, producing exactly-zero attention weights with a query-dependent threshold.
- **Relevance:** Foundational for the exact-elimination variant of §4; the threshold's monotonicity in the candidate set is what makes branch-and-bound sound.
- **Supports:** `branch-and-bound`, §4. **Class:** Foundational background. **Cite:** Thesis later. **Overlap:** [D9] → [D10] → [D11] lineage; cite [D11] as primary for Transformers.

### [D10] Sparse Sequence-to-Sequence Models
Peters, B., Niculae, V., Martins, A. F. T. (2019). *ACL 2019.* arXiv:1905.05702.

- **Summary:** Introduces α-entmax, interpolating between softmax (α=1) and sparsemax (α=2), with efficient bisection computation.
- **Relevance:** The entmax family used by [B5].
- **Supports:** `branch-and-bound`, §4. **Class:** Supporting technique. **Cite:** Thesis later.

### [D11] Adaptively Sparse Transformers
Correia, G. M., Niculae, V., Martins, A. F. T. (2019). *EMNLP-IJCNLP 2019.* arXiv:1909.00015.

- **Summary:** Replaces softmax with α-entmax in Transformer attention with learned per-head α; heads become exactly sparse and interpretable.
- **Relevance:** Establishes that Transformers can be trained with finite-support attention, which is the precondition for exact region elimination. The primary citation for the entmax variant in §4.
- **Supports:** `branch-and-bound`, §4, §24.
- **Class:** Supporting technique. **Cite:** Roadmap now.
- **Forces revision:** §4 should add the exact-elimination variant and cite this plus [B5].

---

## Group E — Randomized / Sampling / Approximation Methods

### [E1] KDEformer: Accelerating Transformers via Kernel Density Estimation
Zandieh, A., Han, I., Daliri, M., Karbasi, A. (2023). *ICML 2023 (PMLR 202).* arXiv:2302.02451.

- **Summary:** Observes that the softmax denominator is a kernel density estimation problem; uses a fast KDE solver (via LSH-based estimators) to compute sampling probabilities and then a subsampled matrix product for the numerator. First attention approximation with *spectral-norm* (not entrywise) guarantees, in subquadratic time under bounded entries.
- **Relevance:** Makes the attention-as-kernel-summation connection formal and links §12 directly to the dual-tree/KDE literature in Group I. Its spectral-norm error criterion is a candidate for the thesis's quality metric.
- **Supports:** `sampling`, `instance-dependent`, §12, §21, §24.
- **Class:** Closest prior art (theory branch). **Cite:** Roadmap now.
- **Forces revision:** §12 should cite; the roadmap's error metric discussion (§19) should mention spectral vs. entrywise guarantees.
- **Overlap:** [E1] → [E2] lineage; [E2] is the more practical and general result.

### [E2] HyperAttention: Long-Context Attention in Near-Linear Time
Han, I., Jayaram, R., Karbasi, A., Mirrokni, V., Woodruff, D. P., Zandieh, A. (2024). *ICLR 2024.* arXiv:2310.05869.

- **Summary:** Two-phase algorithm: find large attention entries with sortLSH, then estimate the rest by uniform sampling; provably near-linear under fine-grained parameters (a bound on the number/magnitude of large entries and a column-norm ratio) that can be small even when entries are unbounded. Inspired by the hard instance in [A3].
- **Relevance:** The closest existing *instance-dependent* complexity analysis of attention: cost is parameterized by measurable properties of the specific matrix. This is the template for the revised §21, and the "large entries + sampled remainder" structure is the same two-branch design as Direction A (eliminate/exact + sample).
- **Supports:** `sampling`, `instance-dependent`, `lower-bounds`, §12, §21, §24.
- **Class:** Closest prior art (for instance-dependent analysis). **Cite:** Roadmap now.
- **Forces revision:** §21 must acknowledge that parameterized near-linear analyses of attention exist; the thesis's parameters must be distinguished from HyperAttention's.
- **Overlap:** Primary over [E1].

### [E3] MagicPIG: LSH Sampling for Efficient LLM Generation
Chen, Z., Sadhukhan, R., Ye, Z., Zhou, Y., Zhang, J., Nolte, N., Tian, Y., Douze, M., Bottou, L., Jia, Z., Chen, B. (2025). *ICLR 2025.* arXiv:2410.16179. **⚠ verify author list.**

- **Summary:** Shows empirically that top-k attention approximation degrades on some tasks because attention is not always sparse (long-tailed, sink-dominated distributions); proposes instead unbiased *importance sampling* of keys using LSH collision probabilities as a sampler, with hash tables and attention computed on CPU. Reports 1.9–3.9× decode throughput with high accuracy.
- **Relevance:** The paper that documents the failure of max-score/top-k selection on flat heads — i.e., the reason the original §4 bound was unsound. Also a CPU-resident design, matching the local-inference goal. Its explicit cost decomposition (search cost vs. retained attention cost) is the same as §3's C_pruning + C_retained.
- **Supports:** `sampling`, `ann-mips`, §3, §4, §12, §22, §24.
- **Class:** Closest prior art / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §4 (max-bound insufficiency), §12 (sampling is the remedy, already done), §22 (add flat-head failure).
- **Overlap:** Primary for sampling in practice; [E4] supersedes with guarantees.

### [E4] vAttention: Verified Sparse Attention
Desai, A., Agrawal, K. K., Yang, S., Cuadron, A., Schroeder, L. G., Zaharia, M., Gonzalez, J. E., Stoica, I. (2025). arXiv:2510.05688. **⚠ verify venue (OpenReview submission observed).**

- **Summary:** Combines deterministic selection (sink tokens, local window, predicted top-k from any method) with uniform random sampling of the remainder, and uses concentration inequalities to give user-specified (ε, δ) guarantees on approximation error — focusing on the softmax denominator, whose bias compounds across layers. Adapts the sample count per query/head; matches dense quality at up to ~10–20× sparsity on RULER/LongBench/AIME.
- **Relevance:** Occupies the "certified error" slot with a *statistical* guarantee. Direction A must therefore offer a *deterministic* certificate to be distinct, and must justify why deterministic elimination is preferable (or complementary) to sampling. Its denominator-first design is the right template for the revised §4 budget.
- **Supports:** `sampling`, `branch-and-bound`, §4, §12, §19, §22, §24.
- **Class:** Closest prior art / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §24 cannot claim error certification as novel; §19's quality metric should mirror vAttention's per-instance error control so results are comparable.
- **Overlap:** Primary for guarantees; builds on [E3].

### [E5] Sparse-Attention Sampling — SampleAttention (Zhu et al., 2024)
arXiv:2406.15486. **⚠ verify title, author list, and venue.**

- **Summary:** Samples structured patterns (column stripes, block-local) to approximate attention during prefill.
- **Relevance:** Sampling on the prefill side; minor.
- **Supports:** `sampling`. **Class:** Competing baseline. **Cite:** Background only. **Overlap:** Secondary to [E3]/[E4].

### [E6] Hashing-Based-Estimators for Kernel Density in High Dimensions
Charikar, M., Siminelakis, P. (2017). *FOCS 2017.*

- **Summary:** LSH-based unbiased estimators for KDE with query time sublinear in n and polynomial in 1/τ (the density lower bound).
- **Relevance:** The theoretical machinery underlying [E1] and the LSH-as-sampler idea in [E3]; belongs in the thesis's related-work chain from KDE to attention.
- **Supports:** `sampling`, Group I. **Class:** Foundational background. **Cite:** Thesis later.

### [E7] On the Fine-Grained Complexity of Empirical Risk Minimization: Kernel Methods and Neural Networks
Backurs, A., Indyk, P., Schmidt, L. (2017). *NeurIPS 2017.*

- **Summary:** SETH-based hardness for kernel PCA, kernel SVM, and related kernel computations; establishes that many kernel-summation problems are quadratic-hard.
- **Relevance:** Since attention is a kernel summation, this and [A3] together explain why any near-linear result must be conditional on the data.
- **Supports:** `lower-bounds`. **Class:** Theoretical constraint. **Cite:** Thesis later.

### [E8] Probability Inequalities for Sums of Bounded Random Variables
Hoeffding, W. (1963). *Journal of the American Statistical Association 58(301), 13–30.*

- **Summary:** Hoeffding's inequality.
- **Relevance:** The concentration bound behind any (ε, δ) sampling certificate ([E4]) and behind a probabilistic elimination rule in §12.
- **Supports:** `sampling`, §12. **Class:** Foundational background. **Cite:** Thesis later.

---

## Group F — Learned Routing and Sparse Indexers

### [F1] DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention
DeepSeek-AI (2025). Technical report, September 2025. **⚠ verify identifier — the report was released via GitHub/Hugging Face; an arXiv version may exist under a different title (DeepSeek-V3.2).**

- **Summary:** Introduces DeepSeek Sparse Attention (DSA): a "lightning indexer" with its own low-rank query/key projections and ReLU scoring computes a relevance score for every prior token, trained by KL-divergence distillation from the full-attention branch; the model then attends to a fixed top-k (2048) tokens under Multi-head Latent Attention. Two-stage continued pre-training from a dense checkpoint. Reported near-lossless versus dense at large context-length reductions in cost.
- **Relevance:** The strongest existing learned-selection baseline and now a production architecture (also GLM-5/5.1, DeepSeek-V4). §13's framing of learned routing as risky is obsolete; the open question is whether training-free geometric methods can approach this in the regime where retraining is not an option.
- **Supports:** `learned-routing`, `baseline`, §13, §22, §24.
- **Class:** Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §13 (status paragraph); §24 (learned indexers satisfy the router-cost constraint).
- **Overlap:** Primary for token-level learned selection. [F2] is the peer-reviewed synthesis of NSA+DSA; cite [F1] for DSA specifics and [F2] for the design principles.

### [F2] Native Sparse Attention: Co-Designing Algorithms and Hardware for Practical Long-Context Efficiency
Yuan, J., Zhang, M., et al. (2026). *National Science Review 13(9), nwag212.* DOI: 10.1093/nsr/nwag212. **⚠ verify full author list.**

- **Summary:** Peer-reviewed perspective covering NSA and DSA as complementary approaches (native sparse training vs. supervised indexer distillation), with design principles emphasizing balanced arithmetic intensity across phases over theoretical complexity reduction.
- **Relevance:** The most citable statement that hardware-aware design, not asymptotic complexity, determines practical wins — directly relevant to §6 hidden costs and §22 failure conditions.
- **Supports:** `learned-routing`, §13, §19, §22.
- **Class:** Competing baseline / Systems evidence. **Cite:** Roadmap now.
- **Forces revision:** §22 should include "theoretical complexity reduction does not translate to balanced arithmetic intensity."

### [F3] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y. X., Wang, L., Xiao, Z., Wang, Y., Ruan, C., Zhang, M., Liang, W., Zeng, W. (2025). *ACL 2025.* arXiv:2502.11089. **⚠ verify author list.**

- **Summary:** Trains sparsity from scratch with three branches — compressed (block-summarized) keys, selected fine blocks chosen via the compression branch's scores, and a sliding window — fused by learned gates; hardware-aligned kernels.
- **Relevance:** The block-level learned-routing baseline; its compression branch is a learned analogue of the coarse level in a hierarchy.
- **Supports:** `learned-routing`, `hierarchical-pruning`, §13. **Class:** Competing baseline. **Cite:** Thesis later. **Overlap:** Secondary to [F1]/[F2].

### [F4] MoBA: Mixture of Block Attention for Long-Context LLMs
Lu, E., et al. (2025). arXiv:2502.13189. **⚠ verify author list.**

- **Summary:** Mixture-of-experts-style gating over KV blocks: each query attends to the top blocks by affinity with mean-pooled block keys.
- **Relevance:** Mean-pooled block keys are the learned counterpart of a centroid score; one of the three main learned block-selection methods.
- **Supports:** `learned-routing`, §13. **Class:** Competing baseline. **Cite:** Thesis later. **Overlap:** Secondary to [F3].

### [F5] SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs
Gao, Y., et al. (2024). arXiv:2410.13276. **⚠ verify author list and venue.**

- **Summary:** Learns a lightweight block-level attention gate by self-distillation from the pretrained model's own attention maps; no change to the base model weights.
- **Relevance:** A middle ground between training-free and native-sparse: cheap to train, applicable post hoc. This is the most realistic learned baseline for a thesis with limited compute.
- **Supports:** `learned-routing`, §13. **Class:** Competing baseline. **Cite:** Thesis later.

### [F6] SpotAttention: Plug-In Block-Sparse Routing for Pretrained Long-Context Transformers
(2026). arXiv:2606.22874. **⚠ verify author list and venue.**

- **Summary:** Trains only a DSA-style selector against a frozen pretrained backbone's dense attention.
- **Relevance:** Shows the learned-indexer approach reaching the "post-hoc on pretrained models" regime, narrowing the space where training-free geometric methods are the only option.
- **Supports:** `learned-routing`, §13. **Class:** Competing baseline. **Cite:** Thesis later.

### [F7] LongCat Sparse Attention: Taming the Lightning via Streaming-Aware Hierarchical Cross-Layer Indexing
(2026). arXiv:2608.01662. **⚠ verify author list and venue.**

- **Summary:** Hierarchical indexer with cross-layer reuse of selections for streaming (growing-cache) decode, built on DSA.
- **Relevance:** Evidence for §10's cross-layer memoization and for incremental index maintenance; both ideas the roadmap lists as open are being engineered in production.
- **Supports:** `learned-routing`, `hierarchical-pruning`, §10, §13. **Class:** Competing baseline. **Cite:** Thesis later.

### [F8] TidalDecode: Fast and Accurate LLM Decoding with Position-Persistent Sparse Attention
Yang, L., et al. (2024). arXiv:2410.05076. **⚠ verify author list and venue.**

- **Summary:** Performs token selection at a few layers and reuses the selected positions in subsequent layers.
- **Relevance:** Concrete precedent for §10's cross-layer reuse; cheap to implement as a component of Direction A.
- **Supports:** `learned-routing`, `kv-paging`, §10. **Class:** Supporting technique. **Cite:** Thesis later. **Overlap:** With [F7]; cite [F8] as the earlier idea.

### [F9] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Fu, Y., Han, S. (2025). *ICLR 2025.* arXiv:2410.10819. **⚠ verify author list.**

- **Summary:** Classifies heads into "retrieval heads" (need full KV) and "streaming heads" (need only sinks + recent window) using a lightweight optimization, and allocates KV budget accordingly.
- **Relevance:** The clearest evidence for head heterogeneity: any Direction A method needs a per-head policy, and the retrieval/streaming split is a natural first cut. Directly supports the proposed "conclusion 9" in §23.
- **Supports:** `learned-routing`, `instance-dependent`, §13, §22, §23.
- **Class:** Supporting technique / Competing baseline. **Cite:** Roadmap now.
- **Forces revision:** §22 should list per-head heterogeneity; §19 should measure per head.

---

## Group G — KV Cache, Paging, and Memory Systems

### [G1] Efficient Memory Management for Large Language Model Serving with PagedAttention
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., Stoica, I. (2023). *SOSP 2023.* arXiv:2309.06180.

- **Summary:** vLLM: stores the KV cache in fixed-size non-contiguous blocks ("pages") managed like virtual memory, eliminating fragmentation and enabling sharing; the attention kernel gathers from pages.
- **Relevance:** The page abstraction that [B1], [B5], and §14 assume. §14's "context divided into pages" is this; the roadmap should cite it rather than reinvent the term.
- **Supports:** `kv-paging`, §14, §24. **Class:** Foundational background / Systems evidence. **Cite:** Roadmap now.
- **Forces revision:** §14 must acknowledge that paging exists as infrastructure; the open question is *predictive* policy, not paging itself.

### [G2] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., Chen, B. (2023). *NeurIPS 2023.* arXiv:2306.14048.

- **Summary:** Evicts KV entries with low accumulated attention scores, keeping "heavy hitters" plus recent tokens; formulates eviction as a dynamic submodular problem.
- **Relevance:** The LFU-style eviction policy §14 lists; also the paper whose weakness (evicting tokens needed later) motivated query-aware selection in [B1]. Useful as the "cache replacement" bridge to Group I's competitive analysis.
- **Supports:** `kv-paging`, §14. **Class:** Competing baseline. **Cite:** Thesis later.
- **Overlap:** With [G3] (StreamingLLM) and SnapKV (Li et al., *NeurIPS 2024*, arXiv:2404.14469 — background only). [G2] is primary for score-based eviction.

### [G3] Efficient Streaming Language Models with Attention Sinks
Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). *ICLR 2024.* arXiv:2309.17453.

- **Summary:** Identifies "attention sinks" — initial tokens that absorb large attention mass regardless of content — and shows that keeping sinks plus a sliding window enables stable streaming.
- **Relevance:** Attention sinks distort the geometry of keys (see [E3]'s discussion) and must be handled explicitly by any bound-based method; the structural study should measure intrinsic dimension with and without sinks.
- **Supports:** `kv-paging`, `instance-dependent`, §14, §21, §22. **Class:** Supporting technique. **Cite:** Roadmap now.
- **Forces revision:** §22 should note sinks as a geometric complication.

### [G4] InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
Lee, W., Lee, J., Seo, J., Sim, J. (2024). *OSDI 2024.* arXiv:2406.19707. **⚠ verify author list.**

- **Summary:** Keeps the full KV cache in host memory and *prefetches* the entries predicted to be important for the next layer, using a speculative computation of the next layer's query with partial weights.
- **Relevance:** This is §14's "predictive caching" in implemented form, including the prediction mechanism. The roadmap's contribution in §14 reduces to workload-specific policy.
- **Supports:** `kv-paging`, §14, §24. **Class:** Closest prior art (for §14). **Cite:** Roadmap now.
- **Forces revision:** §14 must cite; "predictive caching" is not open.

### [G5] HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management
(2026). arXiv:2608.07009. **⚠ verify author list and venue.**

- **Summary:** Indexer-agnostic hierarchical KV cache: full state in host memory, fixed-size GPU cache, LRU replacement that exploits selection locality, and layer-wise prefetching; evaluated on DSA, NSA, and Quest workloads.
- **Relevance:** Shows that the classical paging machinery (LRU, prefetch) has been applied directly to sparse-attention selection; §14's list of cache policies is now a systems literature.
- **Supports:** `kv-paging`, §14. **Class:** Systems/hardware evidence. **Cite:** Thesis later.

### [G6] SGLang / RadixAttention — Efficiently Programming Large Language Models Using SGLang
Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., Sheng, Y. (2024). *NeurIPS 2024.* arXiv:2312.07104. **⚠ verify author list.**

- **Summary:** Prefix caching of KV via a radix tree so that shared prompt prefixes across requests are computed once.
- **Relevance:** The concrete form of §10's memoization at the request level; for ChurnBot, a fixed GLASS-trace prefix is exactly what this exploits.
- **Supports:** `kv-paging`, §10, §14; Track B. **Class:** Supporting technique. **Cite:** Thesis later.

### [G7] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., Gholami, A. (2024). *NeurIPS 2024.* arXiv:2401.18079.

- **Summary:** Per-channel key quantization and pre-RoPE quantization to 3–4 bits with minimal perplexity loss; includes an explicit analysis of when the KV cache versus weights dominates memory.
- **Relevance:** Two uses: (a) the source for the claim that at short context, weights dominate and attention is not the bottleneck (§1, §26 caveats); (b) quantization is orthogonal to and composable with elimination, and bounds must remain admissible under quantized keys (revised §4 numerical note).
- **Supports:** `kv-paging`, §1, §4, §19, §26. **Class:** Systems/hardware evidence. **Cite:** Roadmap now.
- **Forces revision:** §1/§26 must condition the "attention is the bottleneck" claim on context length.

### [G8] Fast Inference from Transformers via Speculative Decoding
Leviathan, Y., Kalman, M., Matias, Y. (2023). *ICML 2023.* arXiv:2211.17192.

- **Summary:** A small draft model proposes several tokens; the target model verifies them in one parallel step; exact output distribution preserved.
- **Relevance:** For small local models at short context, this is the dominant practical lever for decode speed — the roadmap's §26 should acknowledge it so that a committee does not ask why the thesis ignores the biggest known win for its stated goal.
- **Supports:** §26. **Class:** Competing baseline (for the local-inference goal). **Cite:** Roadmap now.
- **Forces revision:** §26's causal chain from attention algorithms to local-AI feasibility.
- **Overlap:** Chen et al. 2023 (arXiv:2302.01318) is the concurrent DeepMind paper; cite [G8] as primary.

### [G9] AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration
Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., Han, S. (2024). *MLSys 2024.* arXiv:2306.00978.

- **Summary:** 4-bit weight quantization preserving salient channels; the standard for on-device deployment.
- **Relevance:** Same role as [G8]: the weight-side lever for local inference. Cite alongside GPTQ (Frantar et al., *ICLR 2023*, arXiv:2210.17323 — background only).
- **Supports:** §26. **Class:** Competing baseline. **Cite:** Thesis later.

---

## Group H — Formal Methods / Cascaded Conversational Systems (Track B)

These support the separate ChurnBot conversational document (current §15–§18). None are attention papers.

### [H1] FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
Chen, L., Zaharia, M., Zou, J. (2023). arXiv:2305.05176. **⚠ verify venue (TMLR 2024?).**

- **Summary:** LLM cascade: route each query through progressively more expensive models, stopping when a learned scorer judges the answer reliable; plus prompt adaptation and caching. Reports large cost reductions at matched accuracy.
- **Relevance:** The direct precedent for §17's escalation architecture. Its scorer-based stopping rule is what §17 calls "explicit confidence criteria."
- **Supports:** `cascade-grounding`, §17. **Class:** Closest prior art (Track B). **Cite:** Roadmap now (Track B doc).
- **Forces revision:** §17 cannot present cascading as new; the Track B contribution must be the grounding contract and domain-specific coverage measurement.
- **Overlap:** Primary; [H2], [H3] are refinements.

### [H2] AutoMix: Automatically Mixing Language Models
Aggarwal, P., Madaan, A., et al. (2024). *NeurIPS 2024.* arXiv:2310.12963. **⚠ verify author list.**

- **Summary:** Uses self-verification by the small model plus a POMDP-based router to decide when to escalate.
- **Relevance:** Escalation governed by verification — closer to §17's "validation or parsing criteria."
- **Supports:** `cascade-grounding`, §17. **Class:** Competing baseline. **Cite:** Thesis later. **Overlap:** Secondary to [H1].

### [H3] RouteLLM: Learning to Route LLMs with Preference Data
Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., Stoica, I. (2025). *ICLR 2025.* arXiv:2406.18665. **⚠ verify author list.**

- **Summary:** Learns a router between a strong and a weak model from human preference data.
- **Relevance:** Learned routing at the query level; the analogue of §13 for Track B.
- **Supports:** `cascade-grounding`, §17. **Class:** Competing baseline. **Cite:** Thesis later.

### [H4] Rapid Object Detection Using a Boosted Cascade of Simple Features
Viola, P., Jones, M. (2001). *CVPR 2001.*

- **Summary:** The classic cascade classifier: cheap stages reject most negatives early; expensive stages run only on survivors.
- **Relevance:** The original formalization of "most inputs resolved cheaply, expensive computation only on the residual" — the principle underlying both §17 (cascade) and §3 (region elimination). Worth citing in both tracks as the classical root.
- **Supports:** `cascade-grounding`, §3, §17. **Class:** Foundational background. **Cite:** Thesis later.

### [H5] On Optimum Recognition Error and Reject Tradeoff
Chow, C. K. (1970). *IEEE Transactions on Information Theory 16(1), 41–46.*

- **Summary:** Derives the optimal reject rule for classification with a reject option; the error–reject tradeoff curve.
- **Relevance:** The classical basis for "controlled abstention" in §18 and the GLASS cascade generally. Pair with Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks," *NeurIPS 2017* (modern form).
- **Supports:** `cascade-grounding`, §17, §18. **Class:** Foundational background. **Cite:** Roadmap now (Track B doc).

### [H6] Learning Executable Semantic Parsers for Natural Language Understanding
Liang, P. (2016). *Communications of the ACM 59(9), 68–76.*

- **Summary:** Survey of semantic parsing: mapping utterances to executable logical forms, with learning from denotations.
- **Relevance:** §15–§16's "parser → structured intent → validated query" pipeline is semantic parsing. This is the canonical entry point; Zelle & Mooney (*AAAI 1996*) is the historical origin.
- **Supports:** `cascade-grounding`, §15, §16. **Class:** Foundational background. **Cite:** Roadmap now (Track B doc).
- **Forces revision:** §15–16 should name semantic parsing explicitly.

### [H7] PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models
Scholak, T., Schucher, N., Bahdanau, D. (2021). *EMNLP 2021.* arXiv:2109.05093.

- **Summary:** Constrains an LLM's decoding at each step to prefixes that a parser accepts, for text-to-SQL.
- **Relevance:** Shows how to combine a generative model with a formal grammar so output is guaranteed well-formed — the "constrained natural-language explanation" step in §15.
- **Supports:** `cascade-grounding`, §15, §16. **Class:** Supporting technique. **Cite:** Thesis later.
- **Overlap:** With [H8], [H9]; [H7] for parser-constrained decoding, [H8]/[H9] for general grammar-constrained generation.

### [H8] Efficient Guided Generation for Large Language Models
Willard, B. T., Louf, R. (2023). arXiv:2307.09702.

- **Summary:** Compiles regular expressions / grammars into finite-state machines indexed against the vocabulary so constrained decoding has negligible overhead (the Outlines library).
- **Relevance:** Finite-state machines and formal grammars from §15, implemented for LLM output; supports the "formal parser" stage.
- **Supports:** `cascade-grounding`, §15. **Class:** Supporting technique. **Cite:** Thesis later.

### [H9] XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models
Dong, Y., Ruan, C. F., Cai, Y., Lai, R., Xu, Z., Zhao, Y., Chen, T. (2025). *MLSys 2025.* arXiv:2411.15100. **⚠ verify author list.**

- **Summary:** Context-free-grammar-constrained decoding with precomputed token masks and a persistent stack, near-zero overhead.
- **Relevance:** Current state of grammar-constrained generation; cite as the practical tool if the Track B implementation uses CFG constraints.
- **Supports:** `cascade-grounding`, §15. **Class:** Supporting technique. **Cite:** Thesis later.

### [H10] POMDP-Based Statistical Spoken Dialog Systems: A Review
Young, S., Gašić, M., Thomson, B., Williams, J. D. (2013). *Proceedings of the IEEE 101(5), 1160–1179.*

- **Summary:** Review of task-oriented dialogue as belief tracking plus policy over a structured state.
- **Relevance:** §16's compiler-style pipeline is a task-oriented dialogue architecture; this review is the standard citation for the pre-LLM field that already solved narrow-domain conversational NLU with structured intermediate representations.
- **Supports:** `cascade-grounding`, §16. **Class:** Foundational background. **Cite:** Thesis later.

### [H11] Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead
Rudin, C. (2019). *Nature Machine Intelligence 1, 206–215.*

- **Summary:** Argues that post-hoc explanation of black boxes is unreliable for high-stakes decisions and that inherently interpretable models should be used.
- **Relevance:** The intellectual basis for GLASS and for §18's rule that the conversational layer renders but never reinterprets the authoritative trace. Ties Track B to the primary ChurnBot research.
- **Supports:** `cascade-grounding`, §18. **Class:** Foundational background. **Cite:** Roadmap now (Track B doc).

---

## Group I — Classical CS Foundations

Older sources the thesis will need for the algorithmic and analytical framework. Entries are shorter; the "Relevance" field explains how each maps onto the roadmap.

### I.1 Branch-and-bound, best-first search, anytime algorithms

**[I1] An Automatic Method of Solving Discrete Programming Problems.** Land, A. H., Doig, A. G. (1960). *Econometrica 28(3), 497–520.*
The origin of branch-and-bound. *Relevance:* the citation for the technique's name in §4. *Class:* Foundational. *Cite:* Thesis later.

**[I2] Branch-and-Bound Methods: A Survey.** Lawler, E. L., Wood, D. E. (1966). *Operations Research 14(4), 699–719.*
Formalizes the branching rule, bounding function, and incumbent; the vocabulary (admissible bound, incumbent, dominance) the revised §4 should use. *Class:* Foundational. *Cite:* Thesis later. *Overlap:* cite [I2] over [I1] for the general framework.

**[I3] A Formal Basis for the Heuristic Determination of Minimum Cost Paths.** Hart, P. E., Nilsson, N. J., Raphael, B. (1968). *IEEE Transactions on Systems Science and Cybernetics 4(2), 100–107.*
A*, with the definition of an admissible heuristic and the optimality proof. *Relevance:* §6's A* proposal; the admissibility concept is exactly what the score bounds in §4 must satisfy. *Class:* Foundational. *Cite:* Thesis later. See also Pearl, J. (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving.* Addison-Wesley — the standard treatment of best-first search and bounding.

**[I4] An Analysis of Time-Dependent Planning.** Dean, T., Boddy, M. (1988). *AAAI 1988.*
Introduces anytime algorithms: solution quality improves monotonically with time and can be interrupted. *Relevance:* §6's "anytime search" and the termination rule "stop when remaining upper bounds cannot change the result." See also Zilberstein, S. (1996). "Using Anytime Algorithms in Intelligent Systems." *AI Magazine 17(3).* *Class:* Foundational. *Cite:* Thesis later.

### I.2 Dual-tree algorithms and fast kernel summation

**[I5] A Fast Algorithm for Particle Simulations.** Greengard, L., Rokhlin, V. (1987). *Journal of Computational Physics 73(2), 325–348.*
The Fast Multipole Method: hierarchical decomposition of space, multipole expansions for well-separated (admissible) box pairs, O(n) for n-body sums with controlled error. *Relevance:* the classical origin of "reject-or-approximate"; far-field interactions are never dropped, only approximated to tolerance. Direct ancestor of [D3] and the model for the revised §4 decision rule. See also Barnes, J., Hut, P. (1986). "A Hierarchical O(N log N) Force-Calculation Algorithm." *Nature 324, 446–449* — the simpler tree-code with a single opening-angle criterion. *Class:* Foundational. *Cite:* Roadmap now.

**[I6] 'N-Body' Problems in Statistical Learning.** Gray, A. G., Moore, A. W. (2000). *NIPS 2000.*
Introduces dual-tree algorithms: build a tree on the query set and a tree on the reference set, traverse node pairs, and prune or approximate a pair when bounds show its contribution is negligible or can be summarized. Applied to kernel density estimation, nearest neighbors, and related "all-pairs" statistics. *Relevance:* this is the classical formulation of Direction A for the *prefill* case (both queries and keys are sets), decades before Transformers. *Class:* Foundational / Closest prior art (classical). *Cite:* Roadmap now.

**[I7] Nonparametric Density Estimation: Toward Computational Tractability.** Gray, A. G., Moore, A. W. (2003). *SIAM International Conference on Data Mining 2003.*
Dual-tree KDE with per-node error bounds and a global error budget; the exact "size-aware budget" logic recommended for §4 (a node pair is pruned only if |Q|·|R|·(K_max − K_min) fits within the remaining error allowance). *Class:* Foundational. *Cite:* Roadmap now. *Overlap:* [I6] introduces the framework, [I7] the error-budget machinery; cite both.

**[I8] Dual-Tree Fast Gauss Transforms.** Lee, D., Gray, A. G., Moore, A. W. (2006). *NIPS 2005 (Advances in NIPS 18).* arXiv:1102.2878. **⚠ verify proceedings year.**
Combines dual-tree traversal with Hermite/Taylor expansions of the Gaussian kernel, choosing per node pair among direct evaluation, far-field expansion, local accumulation, or translation, under a user-specified relative error bound. *Relevance:* the closest classical analogue to a certified attention algorithm, since softmax attention is a Gaussian-kernel summation (exp(q·k) ∝ exp(−‖q−k‖²/2) up to norms). *Class:* Foundational / Supporting technique. *Cite:* Roadmap now.

**[I9] Linear-Time Algorithms for Pairwise Statistical Problems.** Ram, P., Lee, D., March, W. B., Gray, A. G. (2009). *NIPS 2009.*
Proves that dual-tree algorithms for all-nearest-neighbors and related problems run in O(n) time under a bounded expansion constant (intrinsic-dimension) assumption on the data, with cover trees. *Relevance:* the exact template for the revised §21 — linear expected cost parameterized by a measurable data property, with quadratic worst case. This is the paper to build the instance-dependent analysis on. *Class:* Foundational / Theoretical constraint. *Cite:* Roadmap now.

**[I10] Tree-Independent Dual-Tree Algorithms.** Curtin, R. R., March, W. B., Ram, P., Anderson, D. V., Gray, A. G., Isbell, C. L. (2013). *ICML 2013.* arXiv:1304.4327.
Abstracts dual-tree algorithms into a tree type + traversal + problem-specific "BaseCase" and "Score" functions, so any space tree (kd, ball, cover) can be plugged in. *Relevance:* a clean software architecture for a Direction A prototype that must compare tree types. *Class:* Supporting technique. *Cite:* Thesis later.

### I.3 Metric trees, doubling dimension, and the curse of dimensionality

**[I11] Cover Trees for Nearest Neighbor.** Beygelzimer, A., Kakade, S., Langford, J. (2006). *ICML 2006.*
A tree with O(n) space and query time O(c^{12} log n) where c is the expansion constant; the structure used in [I9]. *Relevance:* the candidate index for Direction A when the doubling dimension of keys is small; its explicit dependence on c makes the collapse condition (§22) quantitative. *Class:* Foundational. *Cite:* Roadmap now.

**[I12] Finding Nearest Neighbors in Growth-Restricted Metrics.** Karger, D. R., Ruhl, M. (2002). *STOC 2002.*
Introduces the expansion (growth) constant and shows NN search is efficient when it is bounded. *Relevance:* origin of the expansion-constant assumption. *Class:* Foundational. *Cite:* Thesis later. See also Krauthgamer, R., Lee, J. R. (2004). "Navigating Nets: Simple Algorithms for Proximity Search." *SODA 2004* — doubling dimension formulation.

**[I13] An Algorithm for Finding Best Matches in Logarithmic Expected Time.** Friedman, J. H., Bentley, J. L., Finkel, R. A. (1977). *ACM Transactions on Mathematical Software 3(3), 209–226.*
kd-tree nearest-neighbor search with expected logarithmic time under distributional assumptions — the earliest "instance-dependent" analysis of tree search. *Class:* Foundational. *Cite:* Thesis later.

**[I14] Five Balltree Construction Algorithms.** Omohundro, S. M. (1989). *ICSI Technical Report TR-89-063.* **⚠ verify report number.**
Ball trees; the centroid + radius bound (q·c + ‖q‖r) proposed for the revised §4 is the ball-tree bound. *Class:* Foundational. *Cite:* Thesis later.

**[I15] An Optimal Algorithm for Approximate Nearest Neighbor Searching in Fixed Dimensions.** Arya, S., Mount, D. M., Netanyahu, N. S., Silverman, R., Wu, A. Y. (1998). *Journal of the ACM 45(6), 891–923.*
BBD-trees with priority (best-first) search and (1+ε)-approximate guarantees. *Relevance:* the canonical best-first tree search with an approximation parameter — the shape of §6 + §4 combined. *Class:* Foundational. *Cite:* Thesis later.

**[I16] A Decomposition of Multidimensional Point Sets with Applications to k-Nearest-Neighbors and n-Body Potential Fields.** Callahan, P. B., Kosaraju, S. R. (1995). *Journal of the ACM 42(1), 67–90.*
The well-separated pair decomposition: O(n) pairs of point sets that cover all interactions, each pair "well separated" so interactions can be approximated. *Relevance:* the formal object behind FMM admissibility and behind dual-tree pruning; provides the combinatorial bound on how many node pairs a prefill dual-tree traversal must visit. *Class:* Foundational. *Cite:* Thesis later.

(See also [C13] and [C14] in Group C for the high-dimensional collapse results.)

### I.4 Hierarchical matrices

**[I17] A Sparse Matrix Arithmetic Based on H-Matrices. Part I: Introduction to H-Matrices.** Hackbusch, W. (1999). *Computing 62(2), 89–108.*
Defines hierarchical matrices: block-cluster trees with low-rank approximation on admissible off-diagonal blocks; O(n log n) storage and arithmetic. *Relevance:* the numerical-analysis structure [D1] imported; the admissibility condition is the same as FMM's. *Class:* Foundational. *Cite:* Roadmap now.

**[I18] Hierarchical Matrices: Algorithms and Analysis.** Hackbusch, W. (2015). *Springer Series in Computational Mathematics 49.*
Book-length treatment including H²-matrices. *Cite:* Thesis later. See also Börm, S., Grasedyck, L., Hackbusch, W. (2003). "Introduction to Hierarchical Matrices with Applications." *Engineering Analysis with Boundary Elements 27(5), 405–422* — the accessible survey.

### I.5 Output-sensitive and instance-dependent analysis

**[I19] The Ultimate Planar Convex Hull Algorithm?** Kirkpatrick, D. G., Seidel, R. (1986). *SIAM Journal on Computing 15(1), 287–299.*
O(n log h) convex hull where h is the output size — the paradigm of output-sensitive analysis. *Relevance:* the model for expressing Direction A's cost as O(n·(k_ε + polylog)) with k_ε the effective attention support; the terminology "output-sensitive" recommended for §21 comes from here. See also Chan, T. M. (1996). "Optimal Output-Sensitive Convex Hull Algorithms in Two and Three Dimensions." *Discrete & Computational Geometry 16(4), 361–368.* *Class:* Foundational. *Cite:* Roadmap now.

**[I20] Amortized Computational Complexity.** Tarjan, R. E. (1985). *SIAM Journal on Algebraic and Discrete Methods 6(2), 306–318.*
Potential-function amortized analysis. *Relevance:* needed for §10/§14 claims about incremental index maintenance costs across decode steps. *Class:* Foundational. *Cite:* Thesis later.

### I.6 Online algorithms and caching

**[I21] Amortized Efficiency of List Update and Paging Rules.** Sleator, D. D., Tarjan, R. E. (1985). *Communications of the ACM 28(2), 202–208.*
Introduces competitive analysis; proves LRU is k-competitive for paging. *Relevance:* the analytical standard §14 must meet if it claims anything about predictive caching policies. See also Belady, L. A. (1966). "A Study of Replacement Algorithms for a Virtual-Storage Computer." *IBM Systems Journal 5(2), 78–101* (the optimal offline policy); Borodin, A., El-Yaniv, R. (1998). *Online Computation and Competitive Analysis.* Cambridge University Press. *Class:* Foundational. *Cite:* Thesis later.

**[I22] The Log-Structured Merge-Tree (LSM-Tree).** O'Neil, P., Cheng, E., Gawlick, D., O'Neil, E. (1996). *Acta Informatica 33(4), 351–385.*
Tiered index with an in-memory write buffer and periodic merges into sorted on-disk levels. *Relevance:* the classical design for maintaining a searchable index under append-only growth — the KV cache's access pattern — and the suggested answer to §10's incremental-maintenance question (fresh tokens in a flat tier, older tokens in clustered tiers). *Class:* Supporting technique. *Cite:* Thesis later.

### I.7 Fine-grained complexity foundations

**[I23] On the Complexity of k-SAT.** Impagliazzo, R., Paturi, R. (2001). *Journal of Computer and System Sciences 62(2), 367–375.*
States the (Strong) Exponential Time Hypothesis. *Relevance:* the assumption underlying [A2]–[A5]. *Class:* Foundational. *Cite:* Thesis later.

**[I24] On Some Fine-Grained Questions in Algorithms and Complexity.** Vassilevska Williams, V. (2018). *Proceedings of the International Congress of Mathematicians 2018.*
Survey of SETH, Orthogonal Vectors, and related conjectures and reductions. *Relevance:* the accessible reference for the reductions used in [A2]–[A3]. *Class:* Foundational. *Cite:* Thesis later.

### I.8 Graph sparsification (only if §5 is retained as a method)

**[I25] Spectral Sparsification of Graphs.** Spielman, D. A., Teng, S.-H. (2011). *SIAM Journal on Computing 40(4), 981–1025.*
Every graph has a sparse spectral approximation with O(n log n / ε²) edges. See also Spielman, D. A., Srivastava, N. (2011). "Graph Sparsification by Effective Resistances." *SIAM Journal on Computing 40(6), 1913–1926.* *Relevance:* the theory behind [C5]; note it produces data-independent sparsifiers of a *given* graph — using it for attention requires the dense attention graph first, which defeats the purpose unless combined with sampling. *Class:* Foundational. *Cite:* Thesis later (only if §5 survives).

### I.9 Sketching, sampling, and kernel regression

**[I26] Extensions of Lipschitz Mappings into a Hilbert Space.** Johnson, W. B., Lindenstrauss, J. (1984). *Contemporary Mathematics 26, 189–206.*
The JL lemma; foundation for random projections and sketching in §12. *Class:* Foundational. *Cite:* Thesis later.

**[I27] Random Sampling with a Reservoir.** Vitter, J. S. (1985). *ACM Transactions on Mathematical Software 11(1), 37–57.*
Reservoir sampling, listed in §12. *Class:* Foundational. *Cite:* Background only.

**[I28] Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel.** Tsai, Y.-H. H., Bai, S., Yamada, M., Morency, L.-P., Salakhutdinov, R. (2019). *EMNLP-IJCNLP 2019.* arXiv:1908.11775.
Formalizes attention as kernel smoothing (Nadaraya–Watson regression) with the exponential kernel. *Relevance:* the citation that licenses treating attention as a kernel summation and hence applying [I5]–[I9]. Cite with Nadaraya, E. A. (1964), *Theory of Probability and Its Applications 9(1)*, and Watson, G. S. (1964), *Sankhyā A 26(4)*, for the regression estimator itself. *Class:* Foundational. *Cite:* Roadmap now.

**[I29] Introduction to Algorithms.** Cormen, T. H., Leiserson, C. E., Rivest, R. L., Stein, C. (2022). *4th ed., MIT Press.*
Reference for divide-and-conquer, dynamic programming, amortized analysis, and priority queues as used throughout §4–§10. *Class:* Foundational. *Cite:* Background only.

---

## Top 15 Must-Read Sources (ranked by importance to the likely thesis direction)

1. **[B1] Quest (Tang et al., ICML 2024)** — the admissible box bound on KV pages is the single closest match to the proposal's core mechanism, and its cost identity is the template for honest overhead accounting.
2. **[E4] vAttention (Desai et al., 2025)** — occupies the "certified sparse attention" slot with statistical guarantees, so it defines what a deterministic certificate must improve on.
3. **[E3] MagicPIG (Chen et al., ICLR 2025)** — documents the failure of max-score/top-k selection on flat attention heads, which is the reason the original §4 bound was unsound.
4. **[B2] Squeezed Attention (Hooper et al., ACL 2025)** — hierarchical content clustering with coarse-to-fine pruning on a pretrained model, assembled exactly as §4 + §7 propose.
5. **[B3] Multipole Attention (Hooper et al., 2025)** — the training-free "approximate rather than drop" hierarchy on a growing KV cache, i.e., the corrected §4 decision rule already implemented.
6. **[A3] Alman & Song, Fast Attention Requires Bounded Entries (NeurIPS 2023)** — the theoretical constraint that turns the roadmap's hypothesis into a measurable claim about the data regime of real activations.
7. **[A2] Keles et al., On the Computational Complexity of Self-Attention (ALT 2023)** — establishes that quadratic worst case is required for exact attention, not a concession.
8. **[I9] Ram, Lee, March & Gray, Linear-Time Algorithms for Pairwise Statistical Problems (NIPS 2009)** — the exact analytical template for "linear expected cost under a measurable intrinsic-dimension parameter, quadratic worst case."
9. **[I6]/[I7] Gray & Moore, dual-tree algorithms (NIPS 2000; SDM 2003)** — the classical formulation of two-sided hierarchical pruning with a global error budget, which is Direction A for prefill.
10. **[I8] Lee, Gray & Moore, Dual-Tree Fast Gauss Transforms (NIPS 2005)** — the closest classical analogue to certified attention, because softmax attention is a Gaussian-kernel summation.
11. **[I11] Beygelzimer, Kakade & Langford, Cover Trees (ICML 2006)** — the index whose explicit dependence on the expansion constant makes the tree-collapse failure condition quantitative.
12. **[I5] Greengard & Rokhlin, Fast Multipole Method (JCP 1987)** — the origin of admissibility-based hierarchical approximation and of the reject-or-approximate principle.
13. **[E2] HyperAttention (Han et al., ICLR 2024)** — the closest existing instance-dependent complexity analysis of attention; the thesis's parameters must be distinguished from its.
14. **[F1]/[F2] DeepSeek Sparse Attention (2025) and the NSR synthesis (2026)** — the strongest learned-selection baseline and now a production standard; any training-free geometric method must be positioned against it.
15. **[A10] FlashAttention (Dao et al., NeurIPS 2022)** — the dense baseline every speedup must be measured against, and the kernel structure (online softmax, running max) into which a branch-and-bound incumbent naturally threads.

Near misses worth reading next: [B4] HISA, [B5] EntmaxKV with [D11] Correia et al., [D2] MRA attention, [D4] Scatterbrain, [E1] KDEformer, [C1] RetrievalAttention, [F9] DuoAttention, [I17] Hackbusch, [I19] Kirkpatrick–Seidel, [C13] Weber et al.

---

## Revision map: which sources force which changes

| Roadmap section | Sources that force revision or qualification |
|---|---|
| §1 / §26 (attention as the local-inference bottleneck) | [G7] KVQuant, [G8] speculative decoding, [G9] AWQ, [A13] roofline, [C8] SLIDE |
| §2 / §21 (average-case → instance-dependent; worst case required) | [A2], [A3], [A4], [E2], [I9], [I19], [C13], [D8] |
| §3 / §20 (cost in bytes; reject-or-approximate) | [A10], [A13], [B1] cost identity, [B4], [F2], [I5] |
| §4 (bound insufficiency; exact variant) | [E3], [E4], [B1], [B5], [D11], [I7], [I8] |
| §5 (graph framing already done; data-independent) | [C4], [C5], [I25] |
| §7 / §11 (hierarchical clustering and coarse-to-fine exist) | [B2], [B3], [D1], [D2], [D3] |
| §8 (ANN attention exists; MIPS asymmetry; Q/K mismatch) | [C1], [C2], [C9] |
| §9 (sparse + low-rank exists) | [D4], [A7] |
| §10 (memoization = KV/prefix cache; cross-layer reuse) | [G6], [F7], [F8], [I22] |
| §12 (sampling is the flat-head remedy; certification occupied) | [E1], [E2], [E3], [E4] |
| §13 (learned routing is now the production standard) | [F1], [F2], [F3], [F4], [F5], [F6] |
| §14 (paging and predictive prefetch exist) | [G1], [G4], [G5], [I21] |
| §15–§18 (cascades, semantic parsing, abstention exist) | [H1], [H5], [H6], [H7], [H10], [H11] |
| §19 (baselines; per-head measurement) | [A10], [A11], [B6], [F9] |
| §22 (add: intrinsic dimension, flat heads, sinks, indexer cost) | [C13], [C14], [E3], [G3], [B4], [F2] |
| §24 (candidate contribution contradicted) | [B1], [B2], [B3], [B4], [B5], [D2], [D3], [E4] |

---

## Verification checklist before first citation

1. Every entry marked ⚠ — confirm author list and venue on arXiv / DOI landing page.
2. All 2026 entries ([B4], [B5], [F2], [F6], [F7], [G5]) — check for updated versions and formal venues; these were observed only as preprints.
3. [F1] — locate the canonical DeepSeek-V3.2 / DSA citation (technical report vs. arXiv).
4. [D3] Fast Multipole Attention — confirm final venue; multiple revisions exist.
5. [C1] RetrievalAttention, [C2] HashAttention, [E4] vAttention — confirm whether accepted and where.
6. Classical entries [I8], [I14] — confirm proceedings year and report number respectively.