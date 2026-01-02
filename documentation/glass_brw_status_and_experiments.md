# 🧭 Project Status & TODO Roadmap

## ✅ Current Status (Checkpointed)

### Core Architecture Stabilized
- **Glass-box cascade pipeline**: Logistic Regression → Sequential GLASS-BRW → EBM → Meta-EBM
- **Sequential GLASS-BRW (Stage 2)** optimizations:
  - Early pruning relaxed for better exploration
  - Full semantic constraints enforced at depth 3
  - Runtime reduced: **~12–14 hours → ~40 minutes** per split
- **Canonical train/test split** established
  - Shared across all stages for strict meta-alignment
- **Interpretable rule generation**
  - Depth-3 rejections align with precision, leakage, support, and recall constraints
- **Checkpoint commit saved** ✓
  - Safe baseline preserved for further experimentation via `glass-brw-coverage` branch

---

## 🔬 Active Research & Experiments (In Progress)

### Pass 1 Semantic Redesign (Experimental)
**Shift from NOT_SUBSCRIBE prediction → FN-risk routing**

Pass 1 objectives:
- ✓ Deterministically identify **high false-negative risk** regions explicitly instead of implicitly
- ✓ Pass forward risky samples (potential subscribers) to pass 2
- ✓ Abstain on safe regions (confident non-subscribers)
- **Goal**: Isolate the majority of subscriber cases before Pass 2

### Pass 2 Role Refinement
- Maintain **high-precision SUBSCRIBE detection**
- Operate on a **subscriber-enriched survivor pool**
- Allow downstream EBM + Meta-EBM to resolve remaining uncertainty
- **Goal**: Capture most to all subscribers in the target-rich subsample. Let the other stages capture any missed target samples.

### Strategic Summary
**Pass 1** will be a **target router** and **Pass 2** will be a **subscriber detector**. Any abstained samples will be covered by the other stages. This, in theory, should maximize accuracy at the stage 4 router level.

### Depth-Aware Pruning Strategies
| Depth | Strategy | Purpose | Note |
|-------|----------|---------|------|
| **Depth 2** | Permissive expansion | Rule Exploration | (Later we will prune, CAREFULLY) |
| **Depth 3** | Strict semantic enforcement | Rule Decision | — |

Investigating depth-specific overlap and support thresholds

---

## ⚠️ Known Limitations (Under Investigation)

### 🔍 Feature-Bin Dominance
- Certain strong features (specifically, `duration` bins) dominate rule selection
- This leads to under-representation of other lifecycle segments
- We will address this through various innovative experimental approaches that maintain XAI philosophy

### 🎲 Rule Selection Diversity
- Current selection may collapse around a single dominant bin, as can be seen in the rule list
- Structural diversity across bins is not yet enforced, so we will be integrating this with the current Jiccard index scoring logic

### ⚙️ ILP Solver Status
- **Temporarily falls back to greedy selection**
  - Solver thresholds need adjustment to match current candidate volumes to achieve feasibility
  - However, we might be able to improve rule quality to where it passes the current thresholds. Further investigation is required before we can make this conlusion
  - Greedy diversity-based fallback maintains stability and simplicity by design (Simply, greedily choose the top rules)
  - Current focus: semantic correctness and rule quality
  - Will shift focus back to solver threshold optimization once rules are fine-tuned (Priority #1)

---

## 🛠️ Planned Enhancements (Next Iterations)

### Near-Term Improvements
- [ ] **Bin-aware diversity constraints**
  - Enforce coverage across multiple bins of key features (`duration`, `engagement_high/low`, `month_bin`)
- [ ] **Reintroduce ILP-based rule selection**
  - After rule pools are sufficiently pruned and stabilized
- [ ] **Depth-2 pruning optimization**
  - Structural pruning before full depth-3 expansion to reduce runtime
  - Requires careful attention to detail for full rule metrics for both classes

### Medium-Term Goals
- [ ] **Research paper preparation**
  - Formalization of FN-risk routing and abstention-aware cascade architecture

  - Target venues: FAccT 2026, AIES 2026, or NeurIPS XAI Workshop 2026
  - **Timeline**: Manuscript draft by August 2026, submission Fall 2026
  - Leverage this work for graduate program applications

### Visualization & Tooling
- [ ] **Rule coverage maps**
- [ ] **Bin-level population tiling diagnostics**

---

## 📄 Publication Status

**Manuscript in Preparation**: "Glass-Box Cascade Architectures for Interpretable Binary Classification"

- **Current Status**: Architectural validation complete, cross-dataset validation in progress
- **Target Venues**: FAccT 2026, AIES 2026, or NeurIPS XAI Workshop 2026
- **Timeline**: 
  - Manuscript draft: August 2026
  - Submission: Fall 2026
  - Preprint (arXiv): Available upon submission
- **Thesis**: Glass-box ensembles can outperform black-box models while maintaining complete interpretability—the accuracy-interpretability trade-off is an architectural choice, not a fundamental constraint

---

## 📌 Interpretation Note

> **Important**: Reported metrics should be considered **preliminary**. 
> 
> Ongoing work focuses on:
> - Semantic correctness
> - Coverage geometry
> - Optimization behavior
> 
> Not just point estimates of accuracy.