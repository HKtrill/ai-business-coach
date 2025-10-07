# 🤖 Project ChurnBot: Turning Churn Into Intelligence

**Tech Stack**: 🗄️ SQLite, 📊 Jupyter, 🐍 Python, 🔥 PyTorch, 💻 C++, 🔧 MLOps, 💻 TypeScript, 🐳 Docker, ⚛️ React, 🌐 Node.js

**Author**: 👤 Phillip Harris

## 📖 Synopsis
ChurnBot transforms telecommunications customer retention from guesswork into precision science. It is an intelligent AI assistant built specifically for telecom churn patterns. Unlike general-purpose models, ChurnBot focuses on telecom-specific behaviors to provide accurate, actionable insights where it matters most.

## 🚨 Problem Statement: Traditional AI Approaches Miss Telecom-Specific Signals
General-purpose models often treat telecom churn like a standard classification task, potentially missing critical domain-specific signals:

- Call patterns and usage anomalies
- Billing disputes and payment behaviors
- Service degradation indicators
- Subscription anomalies and plan changes

**Result**: High false positives/negatives → wasted marketing spend & preventable customer churn.

The result is **high false positives/negatives** → wasted marketing spend & lost customers.  

**Current assumption:**  
Our prediction equation appears **imbalanced**, favoring churn predictions.  
This imbalance may be caused by temporal feature representations that overweight negative correlations.  
To address this, we will:  
- Engineer a **more balanced temporal feature set** (ensuring positive/negative signals are properly represented).  
- Experiment with **purely temporal** vs **partial temporal** features to study how different cascade stages (RF, ANN, RNN) behave under varying temporal loads.  

ChurnBot addresses these gaps with specialized telecom intelligence that general-purpose models may not fully capture.

# 📝 Research Abstract/Proposal

*Disclaimer: This is a preliminary draft subject to change as the project evolves with further testing and refinement.*

- **Project Goal:**  
  - Explore an innovative approach to churn prediction using a cascaded machine learning pipeline (Logistic Regression → Random Forest → Recurrent Neural Networks)  
  - Leverage **feature engineering** to capture customer behavior patterns, including spending trends, tenure segmentation, and charge distributions  
  - Key features: `tenure`, `MonthlyCharges`, `TotalCharges`, `spending_bin`, `monthly_bin`, `TenureBucket`, `stability_score`, `extreme_spender`, `extreme_monthly`, interaction terms (`monthly_tenure_`), `Contract`, `Dependents`, `SeniorCitizen`, and `OnlineSecurity`

## Baseline Performance (Original Features)

| Model              | Precision | Recall | F1    |
|-------------------|-----------|--------|-------|
| LogisticRegression | 0.672     | 0.532  | 0.594 |
| RandomForest       | 0.675     | 0.500  | 0.575 |
| GradientBoosting   | 0.651     | 0.519  | 0.577 |

- Initial models on original dataset showed **moderate performance**  
- F1 scores ranged from **0.575 to 0.594**, highlighting limitations in capturing complex churn patterns


## Enhanced Cascade Performance on Engineered Features

### Alternative Cascades (experiments in this branch)
- ANN → ANN → RNN  
- Logistic Regression → ANN → RNN  
- RF → ANN → RNN (with feature balancing)
- LR → ANN → RNN (with feature balancing)
- Purely Temporal Feature Sets → stress-test RNN performance  
- Partial Temporal Feature Sets → measure trade-offs in ANN/RF stages  

| Stages     | PR-AUC | Precision | Recall | F1    |
|------------|--------|-----------|--------|-------|
| LR-RF-RNN  | 0.712  | 71.42%    | 75.97% | 72.31% |

- **Improvement over baseline:**  
  - Recall increased ~20% (from ~52% to 75.97%)  
  - Minimal precision tradeoff  
  - Pipeline ensures **stability and generalization** by cleaning variables and models between runs  
- **Feature Engineering Details:**  
  - **Spending rate** (`spending_rate`) calculated as `TotalCharges / tenure` (or `MonthlyCharges` for zero tenure), binned into `spending_bin` (LowSpender, MidSpender, HighSpender), with `extreme_spender` flagging top 5%  
  - **Monthly charges bucketing** (`monthly_bin`) discretizes `MonthlyCharges` into LowMonthly, MidMonthly, HighMonthly, with `extreme_monthly` flagging top 5%  
  - **Tenure bucketing** (`TenureBucket`) segments `tenure` into months-based buckets  
  - **Stability score** (`stability_score`) aggregates `Contract`, `Dependents`, and `Partner` indicators  
  - **Interaction terms** (`monthly_tenure_`) combine `monthly_bin` and `TenureBucket` for nuanced pattern detection  
  - **Log transformations** applied to `TotalCharges`, `MonthlyCharges`, and `spending_rate` to reduce skewness  
  - Additional features: `Contract`, `Dependents`, `SeniorCitizen`, and `OnlineSecurity` these will be used to engineer additional features
- Cascade design:  
  - **LR** captures linear relationships  
  - **RF** captures clusters  
  - **RNN** captures temporal patterns  
- Outperforms standalone models on the original features, especially in precision-recall tradeoff

## 📊 Current Research Focus
- ✅ **Feature Diagnostics** — correlation, AUC, IV, PSI per dataset  
- ✅ **Cross-Dataset Generalization** — WA vs Iranian datasets  
- ✅ **False Positive Reduction** — threshold tuning + class balancing  
- ✅ **Semantic Buckets** — grouping features into `business`, `technical`, `spending`, `temporal`  
- ✅ **Temporal Feature Balance** — rebalance equation to avoid over-prediction of churn  
- 🔄 **Daily Logs** — track findings and failed experiments  

## Limitations

- Dataset variability (feature distributions, missing values) impose challenges to generalization  
- Most datasets share common features, allowing **partial transferability**  
- Data preprocessing, especially across sets has the potential for bias  
- Achieving additional 10% recall (~85-86%) may require:  
  - Innovative noise reduction techniques
  - Robust cross-validation  
  - Automated cleaning  
  - Expanding dataset diversity  

## Next Steps

- Enhance cascade with **deeper RNN layers**, optimized hyperparameters, and temporal features  
- Test on **all three datasets** using 10-fold cross-validation  
- Refine proposal with **statistical rigor** for academic submission  
- Explore **cost-based threshold tuning** to optimize retention expenses, targeting recall of ~85-86%

## 🧠 Core Thesis: Domain-Specific Cascade Architectures May Achieve Superior Performance-Interpretability Trade-offs

**Research Hypothesis**: Domain-specific cascade architectures may achieve superior performance–interpretability trade-offs compared to general-purpose models for specialized prediction tasks that can be decomposed into interpretable stages, as demonstrated through telecom churn prediction.

**Key Arguments**:

- 🎯 **Architectural Interpretability**: Each cascade stage serves a distinct, interpretable purpose mapping to real telecom business logic - RF for feature ranking, ANN for complex interactions, RNN for temporal patterns
- ⚡ **Computational Efficiency Trade-offs**: Specialized models achieve comparable accuracy with dramatically lower resource requirements and faster inference times
- 🔍 **Domain Structure Exploitation**: Cascade design decomposes telecom churn into manageable, interpretable components that avoid the opacity of massive parameter spaces
- 💡 **Actionable Insights**: Model predictions include clear feature importance and decision paths enabling targeted business interventions rather than black-box outputs
- 📊 **Measurable Explanations**: Quantifiable interpretability metrics enable direct comparison with general-purpose approaches on explanation quality

This thesis challenges the current industry assumption that "bigger is always better" by demonstrating measurable advantages in performance, interpretability, resource efficiency, and business actionability for domain-specific applications. The approach works best for problems where business processes can be decomposed into interpretable stages.
## 🎯 Domain-Specific Intelligence

### Three-Stage Cascade Model
**Logistic Regression → Random Forest → Recurrent Neural Network**

This specialized pipeline is optimized for precision + recall in telecom churn, detecting patterns that general-purpose models may not generalize effectively, with a target ~20% performance increase (e.g., F1 from 0.636 to ~0.8). The cascade leverages:

1. **Logistic Regression (LR)**: Establishes a linear baseline, capturing trends like tenure and TotalCharges.
2. **Random Forest (RF)**: Enhances classification with cluster detection and feature importance ranking, using metrics like FeatureClusterLabel.
3. **Recurrent Neural Network (RNN)**: Models temporal sequences and non-linear shapes, refining predictions with geometric features like CosineSimilarity and ChurnEdgeScore.

### Pipeline Architecture
```
data_loader → preprocessor → feature_engineer → leakage_monitor → cascade_model → experiment_runner
```

## ⚡ C++ Performance Optimization
ChurnBot leverages custom C++ implementations for maximum inference speed and memory efficiency:

- **Hand-optimized models**: RF, ANN, and RNN written from scratch in C++
- **CS Theory Optimizations**: Branch & bound algorithms, SIMD matrix operations, cache-friendly data structures
- **Custom Memory Management**: Specialized allocators for telecom data patterns
- **Python Integration**: Seamless pybind11 bindings maintain Python development experience
- **Boundary Elimination**: Direct C++ pipeline execution eliminates Python interface overhead

**Expected Performance Gains**: 5-20x faster inference compared to traditional Python ML libraries.

## 🎯 Choose Your Experience
⚡ **Terminal Version (Light)**: For telecom analysts and technical teams — fast, efficient insights through command-line interaction.

📈 **Dashboard Version (Heavy)**: For telecom executives — rich visualizations and executive-ready presentations.

Both versions are specialized for telecom churn, analyzing call patterns, data usage shifts, billing disputes, and service degradation that general-purpose models may not capture. All computations run locally, keeping sensitive subscriber data on your network.

## 🔒 Privacy & Security: Local-First Philosophy
ChurnBot runs entirely on your machine with zero cloud dependencies:

✅ No external data transfers — sensitive subscriber data never leaves your network
✅ No monthly fees or API costs
✅ Full data sovereignty — maintain compliance and avoid regulatory penalties
✅ Immediate analysis — no network latency or downtime
✅ C++ Performance — enterprise-grade speed with local execution

Compare this to general-purpose models that may rely on cloud APIs with inherent data exposure risks.

## 📊 Benchmark Superiority
💼 **Real-World Impact**
**Business ROI**:
- 📉 Reduce churn-related losses through precise targeting
- 📈 Improve executive decision-making with actionable insights
- 🛡️ Maintain full data sovereignty → avoid compliance penalties
- 💰 Eliminate cloud API costs and subscription fees

**Security ROI**:
- 🔒 Complete data privacy — no external data exposure
- 📋 Regulatory compliance maintained
- 🏢 Enterprise-grade security through local execution

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
│   └── CMakeLists.txt                 # TODO: set up master build configuration
├── BasePipeline.py                    # TODO: implement base class for pipelines
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
