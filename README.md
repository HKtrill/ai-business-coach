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
- interaction-driven symbolic rules
- nonlinear response curves
- selective routing and abstention
- confidence-weighted final arbitration

The primary architecture, **GLASS**, serves as the interpretable reasoning engine. Each stage has a distinct role, and predictions can be traced through calibrated scores, symbolic routing decisions, additive feature effects, and final arbitration. A structurally matched black-box cascade is developed alongside GLASS to measure the predictive cost, if any, of the interpretability constraints imposed at each stage. An optional local natural-language layer is planned for conversational interaction with predictions and explanations while remaining subordinate to — and independent from — the core decision pipeline.

The project therefore focuses not only on predictive performance, but on **traceability, abstention, model complementarity, and controlled glass-box vs. black-box comparison**.

---

> ⚠️ **Research Status & Dataset Audit Notice**
>
> ChurnBot is under active research and architectural refinement. The current research phase focuses on rigorous validation using the **UCI Bank Marketing term-deposit dataset** before the architecture is transferred back to customer-churn modeling.
>
> During leakage and temporal-validity auditing, `duration`, `poutcome`, and `pdays` were identified as problematic for deployment-realistic pre-contact prediction. The current experiment therefore evaluates the harder task of estimating subscription likelihood before contact occurs rather than relying on post-call or prior-campaign artifacts.
>
> These issues were surfaced through dataset auditing, interpretable modeling, and symbolic-rule diagnostics.
>
> A second audit pass tightened the **stage-to-stage contracts**: every stage now hands downstream stages out-of-fold (OOF) training predictions, operating thresholds are selected on training data only, and every artifact carries split, index, and fold provenance that downstream stages verify before use.
>
> See the full research status and dataset audit notes here:  
> **[Dataset Audit & Research Status](documentation/dataset_audit_and_research_status.md)**

---

## 🚨 Research Problem: The Interpretability–Performance Trade-off

Machine-learning systems often improve predictive flexibility by introducing model structures that are difficult to inspect directly.

A common deployment pattern is therefore to:

- train a high-capacity black-box model
- apply post-hoc explanation tools such as SHAP or LIME
- approximate the model's reasoning after training
- accept reduced transparency in exchange for predictive flexibility

Project ChurnBot investigates a different question:

> **How much predictive performance is actually lost when interpretability is imposed as an architectural constraint rather than added after training?**

### ChurnBot Approach

Instead of relying primarily on post-hoc approximation, GLASS builds interpretability into the inference architecture itself.

Predictions are grounded in:

- interpretable coefficients
- explicit symbolic rules
- additive shape functions
- explicit routing states
- confidence-weighted arbitration
- abstention when confidence is insufficient

A matched black-box cascade removes these constraints stage by stage while preserving the same data, splits, feature contracts, and evaluation protocol.

This allows the project to measure the interpretability–performance trade-off directly rather than assume it.

---

## 🎯 Architecture: GLASS Cascade

**GLASS** stands for **Glass-box Layered Abstention-aware Scoring System**.

GLASS is a four-stage interpretable decision architecture designed to preserve traceable inference while remaining competitive with less constrained alternatives.

```text
Stage 1: Calibrated Logistic Regression

  ↓ Captures global linear trends through interpretable coefficients

  ↓ Produces calibrated, out-of-fold training probabilities

  ↓ Provides the transparent linear signal used downstream


Stage 2: GLASS Router

  ↓ Operates over binary predicate atoms derived from audited source features

  ↓ Searches a constrained symbolic rule space using RF-guided ordering,
    beam pruning, feasibility checks, and rule selection

  ↓ Applies sequential pass-specific symbolic rules

  ↓ Train-side routes come from an outer cross-fit, so every training
    row is routed by rules that never saw its label

  Pass 1: NOT_SUBSCRIBE

    ↓ Routes high-confidence non-subscriber regions

    ↓ Unresolved samples continue to Pass 2

  Pass 2: SUBSCRIBE

    ↓ Routes subscriber regions among the remainder

    ↓ Unresolved samples remain abstained


Stage 3: Explainable Boosting Machine

  ↓ Models nonlinear effects through additive shape functions

  ↓ Captures nonlinear response curves and limited interactions

  ↓ Out-of-fold probabilities, feature engineering refit per fold,
    OOF-gated fold-nested calibration

  ↓ Provides a complementary interpretable probability signal


Stage 4: GLASS Arbiter

  ↓ Receives aligned, verified out-of-fold outputs from Stages 1–3

  ↓ Computes calibration-aware trust weights

  ↓ Combines model signals through weighted confidence

  ↓ Resolves disagreement when confidence is sufficient

  ↓ Abstains when aggregate confidence is too low


Customer-Level Prediction with Stage-by-Stage Traceability
```

### Key Innovation: Glass-Box Inference with Stage-Level Traceability

* **Logistic Regression:** Direct coefficient inspection and calibrated linear scoring
* **Constrained Symbolic Rule Router:** Explicit IF–THEN routing rules, pass-level decisions, and abstention behavior
* **EBM:** Additive shape functions exposing nonlinear feature effects
* **GLASS Arbiter:** Transparent, hand-designed confidence-weighted arbitration between stage outputs — every weight, threshold, and abstention cut is an inspectable scalar

The goal is not simply to stack interpretable models. The cascade assigns each stage a distinct decision role and evaluates the system through complementarity, disagreement, abstention behavior, and shared-failure reduction.

The cascade is designed as a glass-box inference architecture. Some training-time components, such as Random Forest feature importance, are used to guide symbolic rule discovery, but inference-time decisions remain traceable through calibrated scores, explicit symbolic rules, EBM effects, and GLASS Arbiter decisions.

---

## ⚖️ Black-Box Counterpart Cascade
The black-box cascade mirrors GLASS stage for stage. Each stage keeps its GLASS counterpart's inputs, split, folds, objective, and evaluation protocol, and removes one interpretability constraint:

| Stage | GLASS | Black-box counterpart | Constraint removed |
| --- | --- | --- | --- |
| 1 | Calibrated Logistic Regression | Calibrated MLP | linearity |
| 2 | GLASS Router (symbolic rules) | Two-pass RF Router | explicit IF–THEN rules |
| 3 | Explainable Boosting Machine | XGBoost | additive structure |
| 4 | GLASS Arbiter (weighted confidence) | Meta-XGB (stacked XGBoost) | explicitly specified arbitration structure |

Stage 4 isolates the arbitration question: **Meta-XGB** receives exactly the same information channels as the GLASS Arbiter — the three stage probabilities and the router's Pass 1 / Pass 2 / abstain state — and nothing else (no raw features, labels, upstream thresholds, or test-derived values).

The GLASS Arbiter combines these signals through an explicitly specified weighted-confidence structure with data-derived weights and operating thresholds, whereas Meta-XGB learns the combination function from training data.

### Shared evaluation protocol

- **One split:** both arms use the same `GLOBAL_SPLIT` (80/20, `random_state=42`); every artifact records a split fingerprint that downstream stages recompute and validate.
- **One fold plan:** a shared stratified 5-fold partition drives Stage 2–4 cross-fitting in both arms, while Stage 1 uses a shared 10-fold partition.
- **Out-of-fold hand-offs:** each stage passes OOF training predictions downstream; in-sample predictions are retained for diagnostics only and are rejected by the Stage 4 loader.
- **Training-only configuration:** thresholds, calibration, trust weights, and abstention cuts are chosen using training data only. The held-out test split is scored after configuration is frozen. Test-fitted thresholds may be recorded for reference but are never used operationally.
- **Shared Stage 4 contract:** `shared/stage4/` reads and validates both arms' Stage 1–3 artifacts using the same alignment and provenance checks, including split identity, row and index alignment, labels, fold provenance, OOF honesty, and router semantics.
- **Paired statistics:** arm comparisons use the same held-out rows, paired bootstrap confidence intervals, and McNemar tests. Metric differences are treated as statistically supported only when the corresponding confidence interval excludes zero.

---

## 🧠 Core Thesis: Glass Boxes Can Compete with Black Boxes

### Research Hypothesis

Carefully designed interpretable cascade architectures can match or exceed black-box performance while preserving full transparency — particularly in structured decision domains such as subscription modeling and customer retention.

### How the Hypothesis Is Tested

- Stage-by-stage comparison against a structurally matched black-box counterpart under one shared protocol
- Threshold-free (ROC-AUC, PR-AUC) and operating-point (recall, precision, F2) metrics on identical held-out rows
- Abstention behavior compared at each model's own training-selected configuration
- Uncertainty reported through paired bootstrap intervals rather than point differences
- Full prediction traceability retained on the GLASS side through interpretable intermediate stages

This work argues that the perceived interpretability–performance trade-off is largely an architectural choice rather than a fundamental limitation — a claim the matched comparison is designed to confirm or refute.

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

## 🗺️ Research Roadmap

The project is currently completing cascade validation and preparing the
comparison pipeline.

See the detailed implementation and research roadmap:

[Research Roadmap](prototype/bank_pipeline/documentation/research_roadmap.md)

---

## ⚠️ Limitations

- Dataset variability introduces generalization challenges
- Rule consolidation may require domain-specific threshold tuning
- The multi-stage cascade introduces additional computational overhead
- Shape-function interpretation still requires statistical and domain expertise
- Results currently come from a single train/test split and seed; bootstrap intervals capture test-sampling uncertainty but not training variability
- The two Stage 4 arbiters select operating points differently (recall-targeted base thresholds vs. an F2-selected threshold), so threshold-dependent metrics partly reflect operating-point choice
- OOF hand-offs prevent direct training leakage, but any upstream model-selection or tuning decisions not fully nested within the outer folds may introduce mild training-side optimism

---

## 📚 Dataset Sources & Citations

### **1) Bank Marketing – Term Deposit Subscription**

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
