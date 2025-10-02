# 🤖 Project ChurnBot (Research Branch)

## Branch Purpose
This branch is **not production-ready**.  
It exists for:
- Daily experiment notes ("standups")
- Scratch notebooks (messy, trial-and-error work)
- Draft versions of clean, paper-style notebooks
- Logs of dataset experiments and generalization performance

The `main` branch remains the **clean, reproducible, thesis-ready pipeline**.  
This branch is for **research in-progress**.

---

## 🛠 Tech Stack
🗄️ SQLite • 📊 Jupyter • 🐍 Python • 🔥 PyTorch • 💻 C++  
🔧 MLOps • 💻 TypeScript • 🐳 Docker • ⚛️ React • 🌐 Node.js  

Author: 👤 Phillip Harris  

---

## 📖 Synopsis
ChurnBot transforms telecom customer retention from guesswork into precision science.  
Unlike general-purpose models, ChurnBot focuses on **telecom-specific behaviors** to provide accurate, actionable insights.

This branch extends that vision with **experiments in cascade architectures, cross-dataset generalization, and feature diagnostics**.  
It is where raw research happens before findings are formalized.

---

## 🚨 Problem Statement
General-purpose models often miss **telecom-specific churn signals**:

- 📞 Call pattern anomalies  
- 💸 Billing disputes & payment behaviors  
- 📉 Service degradation indicators  
- 🔄 Subscription anomalies & plan changes  

The result is **high false positives/negatives** → wasted marketing spend & lost customers.  

**Current assumption:**  
Our prediction equation appears **imbalanced**, favoring churn predictions.  
This imbalance may be caused by temporal feature representations that overweight negative correlations.  
To address this, we will:  
- Engineer a **more balanced temporal feature set** (ensuring positive/negative signals are properly represented).  
- Experiment with **purely temporal** vs **partial temporal** features to study how different cascade stages (RF, ANN, RNN) behave under varying temporal loads.  

---

## 🧠 Core Thesis
**Hypothesis:**  
Domain-specific cascade architectures achieve superior **performance + interpretability trade-offs** compared to general-purpose models for specialized prediction tasks like telecom churn.

Key arguments:
- 🎯 **Architectural Interpretability** — stages map to telecom business logic  
- ⚡ **Computational Efficiency** — smaller, faster models rival big LLMs  
- 🔍 **Domain Structure Exploitation** — decomposed into interpretable sub-tasks  
- 💡 **Actionable Insights** — feature importance + decision paths for business use  
- 📊 **Measurable Explanations** — interpretability metrics for direct comparison  

This branch focuses on **validating and refining this thesis**.

---

## 🎯 Cascade Models Under Study
### Three-Stage Pipeline (baseline)
Random Forest → Artificial Neural Network → Recurrent Neural Network

- **RF** — fast baseline classification & feature ranking  
- **ANN** — nonlinear interactions and feature interactions  
- **RNN** — temporal sequence patterns in usage/behavior  

### Alternative Cascades (experiments in this branch)
- ANN → ANN → RNN  
- Logistic Regression → ANN → RNN  
- RF → ANN → RNN (with feature balancing)
- LR → ANN → RNN (with feature balancing)
- Purely Temporal Feature Sets → stress-test RNN performance  
- Partial Temporal Feature Sets → measure trade-offs in ANN/RF stages  

---

## 📊 Current Research Focus
- ✅ **Feature Diagnostics** — correlation, AUC, IV, PSI per dataset  
- ✅ **Cross-Dataset Generalization** — WA vs Iranian datasets  
- ✅ **False Positive Reduction** — threshold tuning + class balancing  
- ✅ **Semantic Buckets** — grouping features into `business`, `technical`, `spending`, `temporal`  
- ✅ **Temporal Feature Balance** — rebalance equation to avoid over-prediction of churn  
- 🔄 **Daily Logs** — track findings and failed experiments  

---

## ⚡ C++ Optimizations
Custom C++ implementations for RF, ANN, and RNN:  
- SIMD matrix ops, cache-friendly data structures  
- Branch & bound optimizations  
- Specialized memory allocators for telecom data  
- Python integration via **pybind11**  

Goal: **5–20x faster inference** vs Python ML libs.  

---

## 🔒 Privacy & Security
- Local-first execution (no cloud dependencies)  
- No API costs, no external data exposure  
- Full compliance & data sovereignty  
- Enterprise-grade inference speed  

---

## 📈 Business ROI
- 📉 Reduce churn losses via precise targeting  
- 📈 Actionable insights for executives  
- 🛡️ Regulatory compliance maintained  
- 💰 Eliminate recurring cloud API fees  

---

## ⬇️ Clone or Download
```bash
git clone -b research https://github.com/<your-repo>/churnbot.git

---

## ⬇️ Clone or Download
```bash
git clone https://github.com/HKtrill/Project-ChurnBot.git
cd Project-ChurnBot
npm install # or yarn
```

## 📂 Project Structure
```
prototype/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── test_splits/
├── churn_pipeline/   # TODO: extract churn model interface into interfaces/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py       # Optimizing
│   ├── leakage_monitor.py
│   ├── cascade_model.py          # Optimizing
│   ├── cascade_model_cpp_wrapper.py    # NEW: TODO: implement C++ model wrapper
│   └── experiment_runner.py
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
│   │   │   ├── churn_cascade.h         # TODO: implement main cascade interface
│   │   │   ├── random_forest.h         # TODO: implement custom RF with optimizations
│   │   │   ├── neural_network.h        # TODO: implement custom ANN with sparse matrices
│   │   │   ├── recurrent_network.h     # TODO: implement custom RNN with early termination
│   │   │   └── telecom_features.h      # TODO: define telecom-specific data structures
│   │   ├── src/
│   │   │   ├── churn_cascade.cpp       # TODO: implement cascade orchestrator
│   │   │   ├── random_forest.cpp       # TODO: implement RF with branch & bound
│   │   │   ├── neural_network.cpp      # TODO: implement ANN with SIMD optimizations
│   │   │   ├── recurrent_network.cpp   # TODO: implement RNN with memory optimization
│   │   │   └── telecom_features.cpp    # TODO: implement feature processing
│   │   ├── bindings/
│   │   │   ├── python_bindings.cpp     # TODO: implement pybind11 interface
│   │   │   └── __init__.py             # TODO: set up Python module
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
│   └── CMakeLists.txt                 # TODO: set up master build configuration
├── BasePipeline.py                    # TODO: implement base class for pipelines
├── requirements.txt                   # TODO: add pybind11, cmake, and other C++ dependencies
└── README.md
```

## 📋 Requirements
### System Requirements
- Python 3.8+
- Node.js 16+
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

### Python Dependencies
```bash
torch>=1.9.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
jupyter>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pybind11>=2.8.0
cmake>=3.12.0
```

### Frontend Dependencies
```bash
react>=18.0.0
typescript>=4.4.0
@types/react>=18.0.0
@types/node>=16.0.0
```

## ⚙️ Installation & Setup
### Backend Setup:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup:
```bash
cd ../frontend
npm install
npm start
```

### Terminal Version:
```bash
python BasePipeline.py --mode terminal
```

### Dashboard Version:
```bash
python BasePipeline.py --mode dashboard
# Then navigate to http://localhost:3000
```

## 🧪 Testing
🧪 **Benchmark Testing**

## 🏗️ Architecture
ChurnBot demonstrates production-ready MLOps with careful handling of sensitive data:

**Core Components:**

- Data Pipeline: Secure local processing with leakage monitoring
- Model Pipeline: Three-stage cascade for optimal precision/recall
- Interface Pipeline: Dual-mode accessibility (terminal + dashboard)
- Experiment Pipeline: Reproducible benchmarking and validation

**Design Principles:**

🛡️ Privacy-first architecture
🎯 Domain-specific optimization
⚡ Performance-optimized inference
🔄 Reproducible experiments

## ❓ Why ChurnBot Matters
ChurnBot isn't just another AI tool — it's a research-backed, production-ready solution solving real-world telecom challenges:

📊 Evidence-based: Clear, reproducible benchmarks over marketing hype
🎓 Research-grade: Publication-ready methodology and results
🏭 Production-ready: Modular, scalable architecture for enterprise deployment
🔐 Security-first: Local execution addresses real enterprise concerns

This positions ChurnBot as a standout project in a market flooded with generic AI applications.

## 📞 Support
For questions or issues, please open a GitHub issue or contact the maintainer.

ChurnBot: Where telecom domain expertise meets cutting-edge ML — turning customer churn from reactive guesswork into proactive intelligence.

---

**ChurnBot:** Where telecom domain expertise meets cutting-edge ML — turning customer churn from reactive guesswork into proactive intelligence.
