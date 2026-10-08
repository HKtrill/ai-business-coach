<img src="assets/churnbot_icon.png" align="right" width="96">

# Project ChurnBot — Interpretable Customer Decision Intelligence

> A research-driven glass-box cascade for transparent customer decision modeling.

**Current Research Application:**  
Predicting term-deposit subscription likelihood using interpretable cascade architectures on the UCI Bank Marketing dataset.

ChurnBot investigates whether carefully designed interpretable models can deliver competitive predictive performance while keeping every stage of a prediction explainable.

**Tech Stack:**
<img src="https://cdn.simpleicons.org/python/3776AB" alt="Python" width="24"/> Python,
<img src="https://cdn.simpleicons.org/jupyter/F37626" alt="Jupyter" width="24"/> Jupyter,
<img src="https://cdn.simpleicons.org/sqlite/003B57" alt="SQLite" width="24"/> SQLite,
<img src="https://cdn.simpleicons.org/typescript/3178C6" alt="TypeScript" width="24"/> TypeScript,
<img src="https://cdn.simpleicons.org/react/61DAFB" alt="React" width="24"/> React,
<img src="https://cdn.simpleicons.org/nodedotjs/5FA04E" alt="Node.js" width="24"/> Node.js,
<img src="https://cdn.simpleicons.org/docker/2496ED" alt="Docker" width="24"/> Docker

**Author:** 👤 Phillip Harris

---

## ⚙️ Installation & Environment Setup

ChurnBot runs fully locally with no external services, API keys, or cloud dependencies.

For installation steps, OS-specific virtual environment commands, and hardware recommendations, see the **[Setup & Hardware Guide](documentation/setup_and_hardware.md)**.

---

## 📖 Synopsis

Project ChurnBot is a research-driven customer decision intelligence system built around interpretable cascade architectures. Rather than treating customer behavior as a single black-box prediction task, it decomposes decision-making into explicit stages that capture:

- linear effects
- interaction-driven symbolic rules
- nonlinear response curves
- selective routing and abstention
- confidence-weighted final arbitration

The primary architecture, **GLASS**, is the interpretable reasoning engine. Each stage has a distinct role, and a prediction can be traced through calibrated scores, symbolic routing decisions, additive feature effects, and final arbitration.

A structurally matched black-box cascade is developed alongside GLASS to measure the predictive cost, if any, of the interpretability constraint imposed at each stage. An optional local natural-language layer is planned for conversational interaction with predictions and explanations; it would sit outside, and remain independent of, the core decision pipeline.

The project therefore focuses on **traceability, abstention, model complementarity, and a controlled glass-box vs. black-box comparison**, not on predictive performance alone.

---

> ⚠️ **Research Status & Dataset Audit Notice**
>
> ChurnBot is under active research and architectural refinement. The current phase validates the architecture on the **UCI Bank Marketing term-deposit dataset** before it is transferred back to customer-churn modeling.
>
> Leakage and temporal-validity auditing identified `duration`, `poutcome`, and `pdays` as unsuitable for deployment-realistic pre-contact prediction. The experiment therefore evaluates the harder task of estimating subscription likelihood before contact, without post-call or prior-campaign artifacts. These issues were surfaced through dataset auditing, interpretable modeling, and symbolic-rule diagnostics.
>
> Later audit passes tightened the **stage-to-stage contracts** and the **feature-research pipeline**:
>
> - every stage hands downstream stages out-of-fold (OOF) training predictions;
> - operating thresholds are selected on training data only;
> - every artifact carries split, index, and fold provenance that downstream stages verify before use;
> - learned feature transformations are fit on training rows only, and target-derived features are refit inside each CV fold.
>
> Full research status and dataset audit notes: **[Dataset Audit & Research Status](documentation/dataset_audit_and_research_status.md)**

---

## 🚨 Research Problem: The Interpretability–Performance Trade-off

Machine-learning systems often gain predictive flexibility by adopting model structures that are hard to inspect directly. A common deployment pattern is therefore to:

- train a high-capacity black-box model;
- apply post-hoc explanation tools such as SHAP or LIME;
- approximate the model's reasoning after training;
- accept reduced transparency in exchange for flexibility.

Project ChurnBot investigates a different question:

> **How much predictive performance is actually lost when interpretability is imposed as an architectural constraint rather than added after training?**

### ChurnBot Approach

Instead of relying on post-hoc approximation, GLASS builds interpretability into the inference architecture itself. Predictions are grounded in interpretable coefficients, explicit symbolic rules, additive shape functions, explicit routing states, and confidence-weighted arbitration that abstains when confidence is insufficient.

A matched black-box cascade removes these constraints stage by stage while keeping the same data, splits, feature contracts, and evaluation protocol. This lets the project measure the interpretability–performance trade-off directly rather than assume it.

---

## 🎯 Architecture: GLASS Cascade

**GLASS** stands for **Glass-box Layered Abstention-aware Scoring System**: a four-stage interpretable decision architecture designed to keep inference traceable while remaining competitive with less constrained alternatives.

```text
Stage 1: Calibrated Logistic Regression

  ↓ Captures global linear trends through interpretable coefficients
  ↓ Produces calibrated, out-of-fold training probabilities


Stage 2: GLASS Router

  ↓ Operates over binary predicate atoms derived from audited source features
  ↓ Searches a constrained symbolic rule space (RF-guided ordering,
    beam pruning, feasibility checks)
  ↓ Selects each pass's rule set with a set-aware ILP selector
    (union coverage, selected-set novelty, hard cardinality)
  ↓ Train-side routes come from an outer cross-fit, so every training
    row is routed by rules that never saw its label

  Pass 1: NOT_SUBSCRIBE
    ↓ Routes high-confidence non-subscriber regions
    ↓ Unresolved samples continue to Pass 2

  Pass 2: SUBSCRIBE
    ↓ Routes subscriber regions among the remainder
    ↓ Unresolved samples abstain


Stage 3: Explainable Boosting Machine

  ↓ Models nonlinear response curves through additive shape functions,
    with a limited number of pairwise interactions
  ↓ Out-of-fold probabilities; feature engineering refit per fold;
    OOF-gated, fold-nested calibration


Stage 4: GLASS Arbiter

  ↓ Receives aligned, verified out-of-fold outputs from Stages 1–3
    (the GLASS Router votes only on the rows it routes)
  ↓ Computes trust weights from calibration (Brier) and accuracy
  ↓ Combines stage signals through weighted confidence
  ↓ Abstains when the winning side's weighted confidence is too low


Customer-Level Prediction with Stage-by-Stage Traceability
```

Selector design: **[GLASS Router ILP Rule Selector](prototype/bank_pipeline/glass_pipeline/glass_router/ilp_rule_selector/documentation/selector_design.md)**

### Stage-Level Traceability

* **Logistic Regression:** direct coefficient inspection and calibrated linear scoring
* **GLASS Router:** explicit IF–THEN routing rules, pass-level decisions, and abstention behavior
* **EBM:** additive shape functions exposing nonlinear feature effects
* **GLASS Arbiter:** hand-designed confidence-weighted arbitration; every weight, threshold, and abstention cut is an inspectable scalar

The goal is not simply to stack interpretable models. Each stage has a distinct decision role, and the cascade is evaluated through complementarity, disagreement, abstention behavior, and shared-failure reduction.

Some training-time components, such as Random Forest feature importance, guide symbolic rule discovery. Inference-time decisions remain traceable through calibrated scores, explicit rules, EBM effects, and GLASS Arbiter decisions.

---

## ⚖️ Black-Box Counterpart Cascade

The black-box cascade mirrors GLASS stage for stage. Each stage keeps its GLASS counterpart's inputs, split, folds, objective, and evaluation protocol, and removes one interpretability constraint:

| Stage | GLASS | Black-box counterpart | Constraint removed |
| --- | --- | --- | --- |
| 1 | Calibrated Logistic Regression | Calibrated MLP | linearity |
| 2 | GLASS Router (symbolic rules) | Two-pass RF Router | explicit IF–THEN rules |
| 3 | Explainable Boosting Machine | XGBoost | additive structure |
| 4 | GLASS Arbiter (weighted confidence) | Meta-XGB (stacked XGBoost) | explicitly specified arbitration structure |

Stage 4 isolates the arbitration question. **Meta-XGB** receives exactly the same information channels as the GLASS Arbiter (the three stage probabilities and the router's Pass 1 / Pass 2 / abstain state) and nothing else: no raw features, labels, upstream thresholds, or test-derived values. The GLASS Arbiter combines these signals through an explicitly specified weighted-confidence structure with data-derived weights and thresholds; Meta-XGB learns the combination function from training data.

### Shared evaluation protocol

- **One split:** both arms use the same `GLOBAL_SPLIT` (80/20, `random_state=42`). Every artifact records a split fingerprint that downstream stages recompute and validate.
- **One fold plan:** a shared stratified 5-fold partition drives Stage 2–4 cross-fitting in both arms; Stage 1 uses a shared 10-fold partition.
- **Out-of-fold hand-offs:** each stage passes OOF training predictions downstream. In-sample predictions are kept for diagnostics only and are rejected by the Stage 4 loader.
- **Training-only configuration:** thresholds, calibration, trust weights, and abstention cuts are chosen on training data only, and the held-out test split is scored after configuration is frozen. Test-fitted thresholds may be recorded for reference but are never used operationally.
- **Shared Stage 4 contract:** `shared/stage4/` reads and validates both arms' Stage 1–3 artifacts with the same checks: split identity, row and index alignment, labels, fold provenance, OOF honesty, and router semantics.
- **Paired statistics (planned, Phase 3):** arm comparisons will use the same held-out rows, paired bootstrap confidence intervals, and McNemar tests. A metric difference will be treated as statistically supported only when its confidence interval excludes zero.

---

## 🧠 Core Thesis: Can Glass Boxes Compete with Black Boxes?

### Research Hypothesis

Carefully designed interpretable cascade architectures can approach or match black-box performance while keeping every stage transparent, particularly in structured decision domains such as subscription modeling and customer retention.

### How the Hypothesis Is Tested

- Stage-by-stage comparison against a structurally matched black-box counterpart under one shared protocol
- Threshold-free (ROC-AUC, PR-AUC) and operating-point (recall, precision, F2) metrics on identical held-out rows
- Abstention behavior compared at each arm's own training-selected configuration
- Uncertainty reported through paired bootstrap intervals rather than point differences
- Full prediction traceability retained on the GLASS side through interpretable intermediate stages

The working view is that much of the perceived interpretability–performance trade-off comes from architectural choices rather than a fundamental limitation. The matched comparison is designed to confirm or refute that view.

---

## 🗣️ Planned Optional NLP Interface

ChurnBot is designed to support an optional natural-language interface for interacting with model outputs and explanations. It would let users:

1. submit natural-language queries;
2. route requests through the interpretable pipeline stages;
3. receive predictions together with the stage-level reasoning behind them.

Explanations would come from the pipeline's own interpretable outputs; the language layer would present them, not replace them.

---

## 🎯 Planned Interfaces

⚡ **Terminal Version (Lightweight)**  
For analysts and technical users who need fast inspection of rules, coefficients, and predictions.

📈 **Dashboard Version (Heavyweight)**  
For presentation-oriented workflows, with visualizations of:

- rule networks
- shape functions
- routing behavior
- model arbitration

Both versions are intended to run fully locally.

---

## 🔒 Privacy & Security: Local-First Philosophy

ChurnBot runs entirely locally with no cloud dependencies.

### Advantages
- No external data transfers
- No API fees or cloud subscriptions
- Full data sovereignty and compliance control
- No network latency or cloud downtime
- Auditable predictions and decision traces

Cloud-hosted black-box systems, by contrast, typically externalize both the model logic and the data handling.

---

## 💼 Intended Real-World Impact

### Business
- More precise targeting, with the potential to reduce acquisition and marketing spend
- Decision-making supported by transparent intervention logic
- No recurring cloud API costs
- Full organizational data control

### Security & Compliance
- Local data privacy
- Auditability for high-stakes decisions
- A deployment model suited to enterprise environments
- Inspectable reasoning to support explainable-AI requirements

---

## 🗺️ Research Roadmap

The project is completing cascade validation (Phase 1); the comparisons pipeline follows in Phase 3.

See the detailed implementation and research roadmap: **[Research Roadmap](prototype/bank_pipeline/documentation/research_roadmap.md)**

---

## ⚠️ Limitations

- Dataset variability introduces generalization challenges
- Rule consolidation may require domain-specific threshold tuning
- The multi-stage cascade adds computational overhead
- Shape-function interpretation still requires statistical and domain expertise
- Results currently come from a single train/test split and seed; bootstrap intervals capture test-sampling uncertainty but not training variability
- The two Stage 4 arbiters select operating points differently (recall-targeted base thresholds vs. an F2-selected threshold), so threshold-dependent metrics partly reflect operating-point choice
- OOF hand-offs prevent direct training leakage, but upstream model-selection or tuning decisions not fully nested within the outer folds may introduce mild training-side optimism

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

> ⚠️ Partial overview; full structure documentation will follow the modular refactor and documentation cleanup.

```text
ai-business-coach/
├── assets/
├── documentation/                 setup guide, dataset audit & research status
└── prototype/
    └── bank_pipeline/
        ├── documentation/         research roadmap
        ├── feature_research/      feature-research notebook, stage trainers, research GLASS Arbiter
        ├── glass_pipeline/        GLASS cascade (GLASS Router, Stage 4 GLASS Arbiter)
        ├── blackbox_pipeline/     black-box cascade (RF Router, Meta-XGB)
        ├── shared/                stage I/O, Stage 3 and Stage 4 contracts, validation
        └── tests/                 regression tests (incl. the ILP rule selector)
```