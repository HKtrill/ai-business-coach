<img src="assets/churnbot_icon.png" align="right" width="96">

# Project ChurnBot — Interpretable Customer Decision Intelligence

> A research-driven glass-box cascade for transparent customer decision modeling.

**Current Research Application:**  
Predicting term-deposit subscription likelihood using interpretable cascade architectures on the UCI Bank Marketing dataset.

ChurnBot explores whether carefully designed interpretable models can deliver competitive predictive performance while preserving full stage-by-stage explainability.

**Tech Stack:**
<img src="https://cdn.simpleicons.org/sqlite/003B57" alt="SQLite" width="24"/> SQLite, 
<img src="https://cdn.simpleicons.org/jupyter/F37626" alt="Jupyter" width="24"/> Jupyter, 
<img src="https://cdn.simpleicons.org/python/3776AB" alt="Python" width="24"/> Python, 
<img src="https://cdn.simpleicons.org/pytorch/EE4C2C" alt="PyTorch" width="24"/> PyTorch, 
<img src="https://cdn.simpleicons.org/cplusplus/00599C" alt="C++" width="24"/> C++, 
<img src="https://cdn.simpleicons.org/typescript/3178C6" alt="TypeScript" width="24"/> TypeScript, 
<img src="https://cdn.simpleicons.org/docker/2496ED" alt="Docker" width="24"/> Docker, 
<img src="https://cdn.simpleicons.org/react/61DAFB" alt="React" width="24"/> React, 
<img src="https://cdn.simpleicons.org/nodedotjs/5FA04E" alt="Node.js" width="24"/> Node.js

**Author:** 👤 Phillip Harris

---

## ⚙️ Installation & Environment Setup

ChurnBot runs fully locally with no external services, API keys, or cloud dependencies.

For installation steps, OS-specific virtual environment commands, and hardware recommendations, see:

**[Setup & Hardware Guide](documentation/setup_and_hardware.md)**

---

## 📖 Synopsis

Project ChurnBot is a research-driven customer decision intelligence system built around interpretable cascade architectures.

Rather than treating customer behavior as a single black-box prediction task, the system decomposes decision-making into explicit stages that capture:

- linear effects
- interaction-driven rules
- non-linear response curves
- abstention-aware arbitration

The cascade serves as the core reasoning engine, producing transparent and fully traceable predictions.

An optional natural-language interface enables conversational interaction with model outputs and explanations while remaining independent from the core modeling pipeline.

The result is a transparent, high-performance cascade where every prediction can be traced to human-readable logic — enabling trustworthy deployment without sacrificing predictive performance.

---

> ⚠️ **Research Status & Dataset Audit Notice**
>
> ChurnBot is under active research and architectural refinement. The current focus has shifted from early Telco churn prototyping to rigorous validation on the UCI Bank Marketing term-deposit dataset.
>
> During leakage and temporal-validity auditing, `duration`, `poutcome`, and `pdays` were identified as problematic for deployment-realistic pre-contact prediction. As a result, ChurnBot evaluates the harder task of estimating subscription likelihood before contact occurs, rather than relying on post-call or prior-campaign artifacts.
>
> These findings were surfaced through the interpretable cascade and symbolic rule diagnostics.
>
> See the full research status and dataset audit notes here:  
> **[Dataset Audit & Research Status](documentation/dataset_audit_and_research_status.md)**

---

## 🚨 Problem: The Interpretability–Performance Trade-off Myth

The ML industry often treats interpretability and predictive performance as mutually exclusive objectives.

Project ChurnBot challenges that assumption directly.

### Common Industry Pattern

- Deploy black-box models in high-stakes decision systems
- Apply post-hoc explanation tools (SHAP, LIME, etc.)
- Approximate decision logic after training
- Sacrifice transparency for benchmark performance

### ChurnBot Approach

Instead of approximating model behavior after deployment, ChurnBot builds interpretability directly into the architecture itself.

Every prediction is grounded in:

- explicit rules
- interpretable coefficients
- additive shape functions
- abstention-aware routing logic

This enables faithful, exact explanations rather than post-hoc approximations.

---

## 🎯 Architecture: GLASS Cascade

**GLASS** stands for **Glass-box Layered Abstention-aware Scoring System**.

GLASS Cascade is a four-stage interpretable decision architecture designed to preserve traceable inference while remaining competitive with black-box baselines.

```text
Stage 1: Calibrated Logistic Regression
  ↓ Captures global linear trends through interpretable coefficients
  ↓ Provides a balanced, calibrated baseline prediction signal
  ↓ Serves as the transparent linear reference model for downstream arbitration

Stage 2: Constrained Symbolic Rule Router
  ↓ Operates over binary predicate atoms derived from audited source features
  ↓ Searches a constrained symbolic rule lattice with RF-guided feature ordering and beam search
  ↓ Applies depth-staged feasibility constraints, validation scoring, and ILP rule selection
  ↓ Routes samples through sequential pass-specific rules with abstention

  Pass 1: NOT_SUBSCRIBE Rule Pass
    ↓ Identifies high-confidence non-subscriber regions
    ↓ Samples not captured by Pass 1 are passed forward to Pass 2

  Pass 2: SUBSCRIBE Rule Pass
    ↓ Identifies subscriber regions among remaining samples
    ↓ Samples not captured by Pass 2 remain abstained

Stage 3: Explainable Boosting Machine
  ↓ Models nonlinear effects through interpretable additive shape functions
  ↓ Provides a complementary nonlinear prediction signal
  ↓ Captures response curves and limited interaction effects

Stage 4: Meta-EBM Arbiter
  ↓ Aligns stage thresholds for comparable arbitration
  ↓ Computes trust weights from calibration and validation behavior
  ↓ Combines LR, symbolic-router, and EBM outputs through weighted confidence
  ↓ Selects the final prediction when confidence is sufficient
  ↓ Abstains when confidence is too low

Customer-Level Predictions with Stage-by-Stage Explainability
```

### Key Innovation: Glass-Box Inference with Stage-Level Traceability

* **Logistic Regression:** Direct coefficient inspection and calibrated linear scoring
* **Constrained Symbolic Rule Router:** Explicit IF–THEN routing rules, pass-level decisions, and abstention behavior
* **EBM:** Additive shape functions exposing nonlinear feature effects
* **Meta-EBM Arbiter:** Transparent confidence-weighted arbitration between stage outputs

The goal is not simply to stack interpretable models. The cascade assigns each stage a distinct decision role and evaluates the system through complementarity, disagreement, abstention behavior, and shared-failure reduction.

The cascade is designed as a glass-box inference architecture. Some training-time components, such as Random Forest feature importance, are used to guide symbolic rule discovery, but inference-time decisions remain traceable through calibrated scores, explicit symbolic rules, EBM effects, and Meta-EBM arbitration.

---

## 🧠 Core Thesis: Glass Boxes Can Compete with Black Boxes

### Research Hypothesis

Carefully designed interpretable cascade architectures can match or exceed black-box performance while preserving full transparency — particularly in structured decision domains such as subscription modeling and customer retention.

### Supporting Observations

- Competitive performance relative to black-box baselines
- Stable validation behavior due to deterministic routing and abstention
- Full prediction traceability through interpretable intermediate stages
- Operational value through transparency, auditability, and actionable intervention logic

This work argues that the perceived interpretability–performance trade-off is largely an architectural choice rather than a fundamental limitation.

---

## 🗣️ Planned Optional NLP Interface

Project ChurnBot is designed to support an optional natural-language interface that streamlines interaction with model outputs and explanations.

Users can:
1. Submit natural-language queries
2. Route requests through interpretable pipeline stages
3. Receive transparent predictions with explicit reasoning

This allows analysts and decision-makers to interact with complex ML systems through conversational workflows while preserving full interpretability.

---

## 🎯 Planned Interfaces

⚡ **Terminal Version (Lightweight)**  
Designed for analysts and technical users requiring fast, efficient inspection of rules, coefficients, and predictions.

📈 **Dashboard Version (Heavyweight)**  
Designed for executive and presentation-oriented workflows with visualizations of:

- rule networks
- shape functions
- routing behavior
- model arbitration

Both versions maintain full local execution and complete interpretability.

---

## 🔒 Privacy & Security: Local-First Philosophy

ChurnBot runs entirely locally with zero cloud dependencies.

### Advantages
- No external data transfers
- No API fees or cloud subscriptions
- Full data sovereignty and compliance control
- No network latency or cloud downtime
- Fully auditable predictions and decision traces

This contrasts sharply with opaque cloud-hosted black-box systems where both the model logic and data handling are externalized.

---

## 💼 Real-World Impact

### Business ROI
- Reduce acquisition costs through more precise targeting
- Improve decision-making with transparent intervention logic
- Reduce unnecessary marketing spend
- Eliminate recurring cloud API costs
- Maintain full organizational data control

### Security & Compliance ROI
- Complete local data privacy
- Full auditability for high-stakes decisions
- Enterprise-friendly deployment model
- Improved regulatory transparency for explainable AI requirements

---

## 🎯 Current Research Focus

- ✅ GLASS Cascade architecture for interpretable staged prediction
- ✅ Dataset audit for leakage-prone and deployment-inconsistent features
- ✅ Calibrated Logistic Regression baseline stage
- ✅ Constrained symbolic rule router with explicit abstention
- ✅ EBM integration for nonlinear response modeling
- ✅ Meta-EBM arbitration layer for confidence-weighted abstention
- 🔄 Validate Stages 1–4 after final configuration updates
- 🔄 Clean notebook outputs and make stage summaries more readable
- 🔄 Add fair baseline comparisons under the audited feature setting
- 🔄 Formal research paper preparation

---

## ⚠️ Limitations

- Dataset variability introduces generalization challenges
- Rule consolidation may require domain-specific threshold tuning
- Interpretable cascade conversion introduces computational overhead
- Shape function interpretation still requires statistical literacy

---

## 📚 Dataset Sources & Citations

### **1) Bank Marketing – Term Deposit Subscription (Current Benchmark)**

This project uses the **Bank Marketing** dataset for primary empirical evaluation.
The dataset is publicly available for research use via the UCI Machine Learning Repository.

**Dataset Source:**  
Moro, S., Rita, P., & Cortez, P. (2014).  
*Bank Marketing Dataset.*  
UCI Machine Learning Repository.  
DOI: https://doi.org/10.24432/C5K306

**Required Citation (Academic):**  
Moro, S., Laureano, R., & Cortez, P. (2011).  
*Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.*  
In P. Novais et al. (Eds.), *Proceedings of the European Simulation and Modelling Conference – ESM’2011*,  
pp. 117–121, Guimarães, Portugal. EUROSIS.

Available at:  
- PDF: http://hdl.handle.net/1822/14838  
- BibTeX: http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

**Public Access:**  
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/bank+marketing

---

### **2) IBM Telco Customer Churn Dataset (Exploratory / Feasibility)**

The IBM Telco Customer Churn dataset was used during early experimentation to validate
the feasibility of the glass-box cascade architecture. Results derived from this dataset
should be interpreted as **architectural validation**, not final performance claims.

Originally published by IBM as part of the **IBM Analytics Accelerator Catalog**.

**Original Source (IBM):**  
https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-customer-churn-dataset/

**Public Mirrors:**  
- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- OpenML: https://www.openml.org/d/42178

---

## 📂 Project Structure

> ⚠️ Project structure documentation will be updated as the modular refactor and documentation cleanup continue.
