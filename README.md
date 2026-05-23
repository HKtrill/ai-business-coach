<img src="assets/churnbot_icon.png" align="right" width="96">

# Project ChurnBot — Interpretable Customer Decision Intelligence
> *Proving that you can have your cake (performance) and eat it too (interpretability).*

**Current Research Application:** Predicting term deposit subscriptions using interpretable cascade architectures

*Predict, prevent, and proactively respond to customer behavior with a research-backed AI assistant*

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

ChurnBot is designed to run fully locally with no external services.

### Prerequisites
* **Python 3.10+** (recommended)
* **Git**
* **Virtual environment tool** (Optional: `venv`, `conda`, etc.)

### 🖥️ Hardware Requirements

**Lightweight / Minimal Setup (Tested):**
- 8GB RAM *(usable with lightweight Windows 10/11 setup)*  
- Intel i5 or equivalent CPU  
- SSD strongly recommended  
- Minimal background applications/services  

> Models have been successfully trained and developed on an optimized Windows 10 installation with 8GB RAM using only VS Code with the training pipeline active. During EBM training, memory usage typically ranges around ~5.5–7GB RAM depending on workload and multitasking.  
>
> However, browser usage and additional background applications can quickly exhaust available memory on 8GB systems, especially during Optuna hyperparameter optimization and extended EBM training sessions.

---

**Recommended (Comfortable Development & Full Pipeline Usage):**
- 16GB RAM  
- 4+ Core CPU (Ryzen 5 / Intel i5 equivalent)  
- SSD or NVMe SSD  

> Cross-device testing found that this configuration provides substantially smoother multitasking, improved notebook responsiveness, and more stable performance during longer research sessions and hyperparameter sweeps.

---

**Recommended for Heavy Research & Hyperparameter Optimization:**
- 32GB+ RAM  
- 8+ Core CPU (Ryzen 7 / Intel i7 equivalent or better)  
- NVMe SSD  

> Recommended for:
> - large Optuna sweeps
> - simultaneous notebooks/browser workloads
> - extensive EBM tuning
> - future GlassCUDA experimentation
> - parallel experimentation and diagnostics

---

### Quick Start (Cross-Platform)

Open a terminal and navigate to your projects/workspace directory.  
If you do not already have one, create a directory for local development projects.

Example:
```text
C:\Users\User\Projects
```

Install Python 3.14+ from:  
https://www.python.org/downloads/

Then run the following commands:

```bash
git clone https://github.com/HKtrill/ai-business-coach.git
cd ai-business-coach
python -m venv .venv
```

> **Important:** Make sure you are inside the `ai-business-coach` root directory when creating the virtual environment.  
> This helps avoid interpreter conflicts and ensures modular package imports resolve correctly.

> `venv` is Python’s built-in virtual environment system.  
> Conda environments may also be used as an alternative.

### macOS / Linux
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 📝 Notes
- All computation runs locally (CPU/GPU as available)
- No API keys or external services required
- Supported on Windows, macOS, and Linux
- VS Code is recommended for development and notebook execution

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

The result is a transparent, high-performance ensemble where every prediction can be traced to human-readable logic — enabling trustworthy deployment without sacrificing predictive performance.

![Dataset Overview](assets/dataset_overview.png)

*Dataset visualization from early Telco churn experiments. Updated bank marketing visualizations are currently being integrated.*

---

> ⚠️ **Research Status & Dataset Transition Notice**
>
> This project is under active research and architectural refinement.  
> While the core glass-box cascade methodology is now established, the current focus has shifted from architectural prototyping toward rigorous application, auditing, and validation on real-world data.
>
> **Latest development work is tracked on the `main` branch following merge [#21](https://github.com/HKtrill/ai-business-coach/pull/21).**
>
> ### Notable Updates
>
> - The project previously transitioned from the synthetic IBM Telco churn dataset
>   (used for early architectural validation) to a real-world bank marketing
>   subscription dataset to improve external validity.
>
> - During temporal validity and leakage auditing of the bank marketing dataset,
>   multiple structural issues were identified that materially affect deployment-oriented modeling:
>
>   - **`duration`** encodes post-outcome information and constitutes direct label leakage.
>
>   - **`poutcome`** contains outcome-derived information from prior campaigns. While technically
>     available at prediction time, it induces historically-dependent predictions rather than
>     actionable behavioral patterns.
>
>   - **`pdays`** strongly correlates with `poutcome` and acts as a numeric surrogate for the same signal,
>     effectively encoding prior campaign state through recency information.
>
>   - Approximately **82% of samples collapse into a single `unknown` regime**
>     (prospects with no prior campaign history), severely limiting meaningful segmentation once
>     history-dependent features are removed.
>
>   - Removing these variables produces a substantial performance drop, suggesting that much of the
>     dataset’s predictive power originates from prior campaign artifacts rather than customer behavior.
>
> - As a result, the project is actively evaluating the bank marketing dataset under stricter
>   temporal and interpretability constraints while also exploring alternative real-world
>   subscriber/churn datasets for comparative validation.
>
> - **Performance metrics are intentionally not directly comparable to many published benchmarks on this dataset.**
>
>   This system predicts subscription likelihood **before contact occurs** using only pre-contact features,
>   whereas many benchmark pipelines include post-outcome or history-dependent variables such as
>   `duration`, `poutcome`, and `pdays`.
>
>   This reflects a deliberate choice to solve the harder, deployment-realistic problem of determining
>   *whether to initiate contact* rather than performing post-hoc analysis of *what happened during the call*.
>
> - **Critically, these issues were first surfaced by the system itself via the interpretable rule lattice.**
>
>   Following removal of `duration`, rule generation became dominated by `poutcome_success`
>   rules exhibiting high precision but extremely low coverage — a signature of shortcut learning
>   rather than meaningful behavioral structure.
>
>   This anomaly triggered deeper temporal inspection of `poutcome` and `pdays`,
>   ultimately leading to confirmation of historical dependency and regime collapse.

### Recent Refactor Status (January 2026)

A major refactor phase has been completed and merged into `main`,
consolidating over one month of end-to-end research and engineering work.

#### Key Outcomes
- Full modularization of the four-stage glass-box cascade (Stages 1–4)
- Centralized data splitting, preprocessing, training, and evaluation
- Dataset upgrade to `bank-additional-full.csv` with macroeconomic context
- Significant runtime reductions (Stage 2 rule generation now completes in under 2 minutes locally)
- Improved non-leaky baseline performance using temporal and macro-context features
- One-command reproducible environment via `requirements.txt`

The project is now positioned for:
- deeper feature interaction analysis
- rigorous statistical validation
- expanded cross-dataset evaluation
- formal research documentation

---

### Dataset Strategy

**Research validation:**  
Bank Marketing dataset (UCI ML Repository) with full temporal auditing and leakage analysis documented above.

**Deployment demonstration:**  
Synthetic or permissively-licensed datasets (e.g., IBM Telco Customer Churn)
to avoid licensing complications while demonstrating cross-domain generalization.

This project is designed as a **methodological framework** that organizations can evaluate and adapt
for proprietary subscription, churn, or binary decision-modeling use cases.

---

### Current Research Objective

Develop and validate an abstention-aware, interpretable cascade architecture for
real-world customer decision modeling using rigorously audited, temporally-valid data.

---

### Broader Implication

This investigation provides empirical evidence that interpretability and dataset auditing
are not optional conveniences, but foundational requirements in applied machine learning.

The interpretable rule lattice exposed latent historical dependencies and regime collapse
that would likely remain undetected under purely black-box modeling despite strong benchmark performance.

More broadly, this highlights a critical failure mode observed in both academic research and deployed systems:
models may optimize against historical artifacts or proxy signals rather than underlying behavioral mechanisms,
leading to misleading confidence and brittle generalization.

Interpretable modeling and rigorous temporal auditing are therefore essential for building reliable,
decision-critical systems that must generalize beyond their training distribution.

---

📊 **Leakage & Regime Collapse Diagnostics**

<p align="center">
  <img src="assets/pdays_vs_previous.png" width="32%">
  <img src="assets/boxplot.png" width="32%">
  <img src="assets/job_by_poutcome_conversion.png" width="32%">
</p>

**Figure — Structural leakage and regime collapse in the bank marketing dataset**

**Left:** `pdays` is tightly coupled with `previous`, indicating that recency largely encodes prior contact history rather than independent behavioral signal.

**Center:** The dominant `poutcome=unknown` regime collapses near zero `pdays`, while non-unknown regimes exhibit strong separation — demonstrating that `pdays` acts as a numeric surrogate for campaign outcome state.

**Right:** Conversion rates diverge sharply only when conditioning on known `poutcome`, while the dominant unknown regime exhibits weak and compressed signal across job segments — confirming historical dependency and loss of meaningful segmentation once history-dependent variables are removed.

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

## 🎯 Architecture: Four-Stage Interpretable Cascade

```text
Stage 1: Logistic Regression (Linear Signals)
  ↓ Captures global linear trends via interpretable coefficients

Stage 2: Sequential GLASS-BRW
        (Gated Logistic Abstention Structured System — Best Rules Win)
  ↓ Routing-first, depth-aware rule lattice
  ↓ Explicitly isolates high false-negative risk regions
  ↓ Pass 1 isolates high-risk regions while abstaining on confident non-subscriber regions
  ↓ Pass 2 predicts SUBSCRIBE only when sufficiently confident; otherwise abstains

Stage 3: Explainable Boosting Machine (EBM)
  ↓ Models non-linear effects via additive interpretable shape functions
  ↓ Resolves uncertainty in routed or abstained samples

Stage 4: Meta-EBM (Abstention-Aware Decision Arbiter)
  ↓ Evaluates outputs from LR, GLASS-BRW, and EBM
  ↓ Selects the most reliable interpretable prediction based on confidence and agreement
  ↓ Optionally abstains when no stage is sufficiently certain
  ↓ Explicitly communicates uncertainty rather than forcing predictions

Customer-Level Predictions with End-to-End Explainability
```

### Key Innovation: Every Stage Remains Interpretable

- **Logistic Regression:** Direct coefficient inspection
- **Sequential GLASS-BRW:** Explicit IF–THEN routing and abstention rules
- **EBM:** Additive shape functions exposing non-linear relationships
- **Meta-EBM:** Transparent arbitration between stage outputs

---

## 🧠 Core Thesis: Glass Boxes Can Compete with Black Boxes

### Research Hypothesis

Carefully designed interpretable ensemble architectures can match or exceed black-box performance while preserving full transparency — particularly in structured decision domains such as subscription modeling and customer retention.

### Supporting Observations

- Competitive performance relative to black-box baselines
- Stable validation behavior due to deterministic routing and abstention
- Full prediction traceability through interpretable intermediate stages
- Operational value through transparency, auditability, and actionable intervention logic

This work argues that the perceived interpretability–performance trade-off is largely an architectural choice rather than a fundamental limitation.

---

## 🗣️ Optional NLP Interface

Project ChurnBot includes an optional natural-language interface that streamlines interaction with model outputs and explanations.

Users can:
1. Submit natural-language queries
2. Route requests through interpretable pipeline stages
3. Receive transparent predictions with explicit reasoning

This allows analysts and decision-makers to interact with complex ML systems through conversational workflows while preserving full interpretability.

---

## 🎯 Choose Your Experience

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

- ✅ Full interpretable cascade architecture achieved
- ✅ Rule extraction and routing from Random Forest models
- ✅ EBM integration for non-linear response modeling
- ✅ Meta-EBM arbitration layer
- 🔄 Cross-dataset validation
- 🔄 Visualization tooling
- 🔄 Formal research paper preparation

---

## ⚠️ Limitations

- Dataset variability introduces generalization challenges
- Rule consolidation may require domain-specific threshold tuning
- Interpretable ensemble conversion introduces computational overhead
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
> ⚠️ The project structure below reflects an earlier development phase and will be updated as the current modular refactor stabilizes near completion.
```
prototype/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── test_splits/
├── churn_pipeline/   # TODO: extract churn model interface into interfaces/
│   ├── __init__.py
│   ├── data_loader.py            ✅
│   ├── preprocessor.py           ✅
│   ├── feature_engineer.py       # Optimizing
│   ├── leakage_monitor.py        ✅
│   ├── cascade_model.py          ✅
│   ├── cascade_model_cpp_wrapper.py ✅
│   └── experiment_runner.py      ✅
├── chatbot_pipeline/
│   ├── __init__.py
│   ├── user_input_handler.py          # TODO: implement input parsing and validation
│   ├── query_processor.py             # TODO: implement query formatting for each model
│   ├── churn_prediction_interface.py  # TODO: connect to Churn model pipeline interface
│   ├── security_model_interface.py    # TODO: connect to Security pipeline interface
│   ├── it_model_interface.py          # TODO: connect to IT pipeline interface
│   └── response_generator.py          # TODO: implement response formatting and templates
├── security_pipeline/
│   ├── __init__.py
│   ├── threat_data_loader.py          # TODO: implement security data loading
│   ├── threat_preprocessor.py         # TODO: implement cleaning and preprocessing
│   ├── feature_engineer.py            # TODO: implement security-specific feature extraction
│   ├── anomaly_detector.py            # TODO: implement anomaly detection model
│   ├── security_model_cpp_wrapper.py  # TODO: implement C++ security model wrapper
│   └── experiment_runner.py           # TODO: implement experimentation framework
├── it_pipeline/
│   ├── __init__.py
│   ├── it_data_loader.py              # TODO: implement IT data loading
│   ├── it_preprocessor.py             # TODO: implement IT data cleaning and preprocessing
│   ├── feature_engineer.py            # TODO: implement IT-specific feature engineering
│   ├── predictive_model.py            # TODO: implement predictive model for IT metrics/outages
│   ├── it_model_cpp_wrapper.py        # TODO: implement C++ IT model wrapper
│   └── experiment_runner.py           # TODO: implement experimentation framework
├── interfaces/
│   ├── __init__.py
│   ├── churn_model_interface.py       # TODO: place extract churn model interface here
│   ├── security_model_interface.py    # TODO: define standard methods like train(), predict(), evaluate()
│   ├── it_model_interface.py          # TODO: define standard methods like train(), predict(), evaluate()
│   └── cpp_model_interface.py         # TODO: define standard C++ model interface
├── utils/
│   ├── utils.py                       # TODO: add additional shared utility functions
│   └── cpp_utils.py                   # TODO: add C++ integration utilities
├── notebooks/
│   ├── churn_pipeline_lab.ipynb       # TODO: Clean up
│   ├── chatbot_pipeline_lab.ipynb     # TODO: set up lab for multi-model chatbot experimentation
│   ├── security_pipeline_lab.ipynb    # TODO: set up lab for security experimentation
│   ├── it_pipeline_lab.ipynb          # TODO: set up lab for IT experimentation
│   └── cpp_benchmarking_lab.ipynb     # TODO: create C++ vs Python benchmarking notebook
├── cpp_models/                        # NEW: C++ optimized models directory
│   ├── shared_cpp/                    # NEW: Common C++ optimizations
│   │   ├── include/
│   │   │   ├── optimization_utils.h    # TODO: implement branch & bound, early termination
│   │   │   ├── data_structures.h       # TODO: implement cache-friendly containers
│   │   │   ├── memory_manager.h        # TODO: implement custom allocators
│   │   │   └── common_types.h          # TODO: define common data types
│   │   └── src/
│   │       ├── optimization_utils.cpp  # TODO: implement CS theory optimizations
│   │       ├── data_structures.cpp     # TODO: implement optimized data layouts
│   │       └── memory_manager.cpp      # TODO: implement memory management
│   ├── churn_pipeline_cpp/            # NEW: Churn C++ models
│   │   ├── include/
│   │   │   ├── churn_cascade.h         ✅
│   │   │   ├── random_forest.h         # Building
│   │   │   ├── neural_network.h        # Building
│   │   │   ├── recurrent_network.h     ✅
│   │   │   └── telecom_features.h      # Building
│   │   ├── src/
│   │   │   ├── churn_cascade.cpp       # Building/Opitmizing
│   │   │   ├── random_forest.cpp       # Building/Opitmizing
│   │   │   ├── neural_network.cpp      # Building/Opitmizing
│   │   │   ├── recurrent_network.cpp   # Building/Opitmizing
│   │   │   └── telecom_features.cpp    # Building
│   │   ├── bindings/
│   │   │   ├── python_bindings.cpp     ✅
│   │   │   └── __init__.py             ✅
│   │   ├── tests/
│   │   │   ├── test_rf.cpp             # TODO: implement unit tests for RF
│   │   │   ├── test_ann.cpp            # TODO: implement unit tests for ANN
│   │   │   └── test_cascade.cpp        # TODO: implement integration tests
│   │   └── CMakeLists.txt              # TODO: set up build configuration
│   ├── security_pipeline_cpp/         # NEW: Security C++ models
│   │   ├── include/
│   │   │   ├── security_cascade.h      # TODO: implement security model interface
│   │   │   ├── anomaly_detector.h      # TODO: implement anomaly detection algorithms
│   │   │   ├── bot_detector.h          # TODO: implement bot detection models
│   │   │   └── threat_classifier.h     # TODO: implement threat classification
│   │   ├── src/
│   │   │   ├── security_cascade.cpp    # TODO: implement security pipeline orchestrator
│   │   │   ├── anomaly_detector.cpp    # TODO: implement real-time anomaly detection
│   │   │   ├── bot_detector.cpp        # TODO: implement bot detection algorithms
│   │   │   └── threat_classifier.cpp   # TODO: implement threat classification
│   │   ├── bindings/
│   │   │   ├── python_bindings.cpp     # TODO: implement pybind11 security interface
│   │   │   └── __init__.py             # TODO: set up security Python module
│   │   ├── tests/
│   │   │   ├── test_anomaly.cpp        # TODO: implement anomaly detection tests
│   │   │   └── test_bot_detection.cpp  # TODO: implement bot detection tests
│   │   └── CMakeLists.txt              # TODO: set up security build configuration
│   ├── it_pipeline_cpp/               # NEW: IT C++ models
│   │   ├── include/
│   │   │   ├── it_cascade.h            # TODO: implement IT model interface
│   │   │   ├── outage_predictor.h      # TODO: implement outage prediction
│   │   │   ├── performance_monitor.h   # TODO: implement performance monitoring
│   │   │   └── servicenow_interface.h  # TODO: implement ServiceNow integration
│   │   ├── src/
│   │   │   ├── it_cascade.cpp          # TODO: implement IT pipeline orchestrator
│   │   │   ├── outage_predictor.cpp    # TODO: implement predictive maintenance
│   │   │   ├── performance_monitor.cpp # TODO: implement system performance analysis
│   │   │   └── servicenow_interface.cpp # TODO: implement ServiceNow API integration
│   │   ├── bindings/
│   │   │   ├── python_bindings.cpp     # TODO: implement pybind11 IT interface
│   │   │   └── __init__.py             # TODO: set up IT Python module
│   │   ├── tests/
│   │   │   ├── test_outage_prediction.cpp # TODO: implement outage prediction tests
│   │   │   └── test_performance.cpp    # TODO: implement performance monitoring tests
│   │   └── CMakeLists.txt              # TODO: set up IT build configuration
│   ├── benchmarks/                    # NEW: Performance benchmarking
│   │   ├── churn_benchmark.cpp         # TODO: implement churn model benchmarking
│   │   ├── security_benchmark.cpp      # TODO: implement security model benchmarking
│   │   ├── it_benchmark.cpp            # TODO: implement IT model benchmarking
│   │   ├── memory_profiling.cpp        # TODO: implement memory usage profiling
│   │   └── compare_all_pipelines.cpp   # TODO: implement comprehensive benchmarking
│   ├── scripts/                       # NEW: Build and deployment scripts
│   │   ├── build_all.sh               # TODO: create master build script
│   │   ├── install_dependencies.sh    # TODO: create dependency installation script
│   │   ├── run_benchmarks.sh          # TODO: create benchmark execution script
│   │   └── generate_bindings.sh       # TODO: create Python binding generation script
│   └── CMakeLists.txt                 ✅
├── BasePipeline.py                    # TODO: implement base class for pipelines
└── README.md
```
