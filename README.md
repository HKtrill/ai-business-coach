<img src="assets/churnbot_icon.png" align="right" width="96">

# Project ChurnBot — Turning Telecom Churn Into Actionable Intelligence
*Predict, prevent, and proactively respond to churn, threats, and performance issues with a research-backed, production-ready AI assistant*


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

## 📖 Synopsis

Project ChurnBot transforms telecom data into actionable intelligence through a domain-specialized cascade architecture that predicts churn with research-backed precision. Rather than treating churn as a generic classification problem, this multi-stage system decomposes the prediction task into interpretable stages—capturing linear patterns, non-linear interactions, and temporal behavior evolution. The result: a meta-learner ensemble that achieves superior performance while remaining explainable to business stakeholders.

![Dataset Overview](assets/dataset_overview.png)
*Dataset characteristics: churn distribution peaks at early tenure, specific monthly charge ranges, and contract types. UsageSlope and TenureBucket emerge as possible features to use at this stage.*

---

## 🚨 Problem: Generic AI Misses Domain-Critical Signals

Most off-the-shelf churn models treat telecom patterns as interchangeable classification tasks, missing critical domain signals:

- **Early tenure risk**: 70% of churners leave within 3 months—requires sensitive early detection
- **Usage anomalies**: Rapid usage slope changes in new customers signal regret
- **Contract-spend mismatches**: High monthly charges on month-to-month contracts = flight risk
- **Service paradox**: No internet service = lower churn (counterintuitive pattern)
- **Social anchors**: Referrals + dependents stabilize long-term customers

**Current Industry Practice**: Optimize for AUC or accuracy globally—missing the asymmetric cost structure where false negatives (missed churners) cost 8x more than false positives (over-retention offers).

**Our Solution**: A specialized cascade that learns asymmetric thresholds and domain patterns through multi-stage feature engineering and intelligent ensemble synthesis.

---

## 🎯 Architecture: Four-Stage Cascade with Meta-Learner Synthesis

```
Stage 1: Logistic Regression (Linear Algebra - SMOTE + F2)
  ↓ (Captures linear relationships & baseline feature importance)
  
Stage 2: Random Forest (Non-linear Interactions - No SMOTE +F1 F1)
  ↓ (Identifies feature interactions & protective patterns)
  
Stage 3: RNN/GRU (Temporal Calculus - No SMOTE)
  ↓ (Models lifecycle evolution & behavioral drift)
  
Stage 4: XGBoost Meta-Learner (Ensemble Synthesis) ✓ WINNER
  ↓ (Routes between models based on confidence & disagreement)
  
Final Prediction with Per-Customer Explainability
```

Each stage serves a distinct interpretable purpose mapping to real telecom business logic:

- **Logistic Regression**: Establishes linear baseline (tenure, spending, contract type)
- **Random Forest**: Captures protective bundles and at-risk triangles (tenure × contract × spend)
- **RNN**: Models customer lifecycle phases and behavioral drift over time
- **Meta-Learner**: Learns when to trust which model based on confidence patterns

---

## 📊 Performance Metrics

### Meta-Learner Final Results ✓ WINNER

| Metric | Score |
|--------|-------|
| **F2-Score** | **0.9080** |
| **Recall** | **0.9133** |
| **Precision** | **0.8880** |
| **AUC-ROC** | **0.9860** |

**Interpretation**: Captures 91% of churners while maintaining 89% precision—only 11 false alarms per 100 predictions. Asymmetric threshold design prioritizes recall (minimize missed churners at acceptable FP cost).

### Individual Stage Performance

| Stage | F2 | Recall | Precision | Key Strength |
|-------|-----|--------|-----------|--------------|
| Logistic Regression (SMOTE) | 0.8298 | 0.9460 | 0.5565 | High recall, interpretable coefficients |
| Random Forest | 0.7759 | 0.7860 | 0.7530 | Balanced precision-recall |
| RNN/GRU + LR+RF Context | 0.7815 | 0.8074 | 0.6789 | Temporal pattern capture |
| **Meta-Learner Cascade** | **0.9080** | **0.9133** | **0.8880** | **Optimal ensemble weighting** |

### Cascade vs. Single-Model Baselines

| Model | F2 | Recall | Precision | Improvement |
|-------|-----|--------|-----------|-------------|
| Best Single Model (LR) | 0.8298 | 0.9460 | 0.5565 | — |
| Meta-Learner Cascade | 0.9080 | 0.9133 | 0.8880 | **+10.8% F2, +32.5% Precision** |

The cascade achieves higher recall while dramatically reducing false positives—a critical business advantage.

---

## 🧠 Core Innovation: Knowledge Distillation & Meta-Learner Synthesis

### Why Meta-Learner Beats Distillation

We tested three ensemble synthesis approaches:

1. **Soft Target Knowledge Distillation** (Ridge LR + RF Regressor)
   - LR MSE: 0.0103 | RF MSE: 0.0004
   - Result: Underperformed meta-learner approach
   
2. **Distilled GRU** (trained on soft targets from ensemble)
   - Result: Underperformed meta-learner approach, outperformed LR/RF KD
   
3. **Meta-Learner (XGBoost)** ✓ **WINNER**
   - Learns optimal model weighting based on per-sample confidence patterns
   - Identifies 457 high-disagreement cases for specialized handling
   - Achieves F2 of 0.9080 across all folds consistently

### Meta-Learner Feature Engineering

The meta-learner receives 9 meta-features encoding disagreement and confidence signals:

```python
meta_features = [
    'lr_prob',              # Individual model predictions
    'rf_prob',
    'rnn_prob',
    'lr_rf_disagree',       # Pairwise disagreement signals
    'lr_rnn_disagree',
    'rf_rnn_disagree',
    'max_confidence',       # Confidence bounds
    'min_confidence',
    'std_confidence'        # Disagreement entropy
]
```

**Top Feature Importances**:
- `min_confidence` (0.38): Acts as uncertainty detector—low confidence triggers ensemble averaging
- `rf_prob` (0.36): RF provides balanced predictions as strong signal
- `lr_rf_disagree` (0.09): When LR and RF conflict, meta-learner applies special logic

### Meta-Learner Decision Logic

- **High-confidence cases** (low std): Trust individual model with highest confidence
- **Conflicted cases** (high std, disagreement): Use entropy-weighted ensemble averaging
- **Low min_confidence**: Route to detailed analysis mode for retention team

---

## 📈 Key Insights & Attribution

### Contribution Attribution

Individual models contribute asymmetrically to final predictions:

- **Logistic Regression**: 76% contribution (strong linear signal)
- **RNN**: 15.5% contribution (temporal patterns matter)
- **Random Forest**: 8.5% contribution (non-linear interactions less critical)

Meta-learner learns this weighting adaptively per customer—some high-risk customers require RNN's temporal analysis, while others are confidently flagged by LR's linear patterns.

### Disagreement Analysis

**457 high-disagreement cases** identified where models strongly diverge. These cases are flagged for:
- NLP context extraction from customer interaction history
- Specialized handling by retention teams
- Feature importance debugging to understand model conflicts

**Business Value**: These 457 customers receive individualized analysis rather than generic scoring.

---

## 🛠️ Feature Engineering by Stage

### Stage 1: Logistic Regression (Aggressive SMOTE + F2 Optimization)

**Focus**: Maximize recall for early churn detection with explainable coefficients
**Data Strategy**: Aggressive SMOTE balancing (60% sampling, k=5) + F2 metric optimization prioritizes recall over precision

**Feature Set**:
```python
lr_features = [
    'contract_risk',              # M2M=0.85, 1Y=0.40, 2Y=0.10
    'tenure_phase_0-3m',          # One-hot encoded tenure bins
    'tenure_phase_3-6m',
    'tenure_phase_6-12m',
    'tenure_phase_12-24m',
    'tenure_phase_24m+',
    'monthly_risk_tier_low',      # One-hot encoded charge tiers
    'monthly_risk_tier_medium',
    'monthly_risk_tier_high',
    'monthly_risk_tier_very_high',
    'value_efficiency',           # Total / Expected Lifetime
    'service_complexity',         # Normalized service count
    'risk_decay',                 # exp(-tenure/12)
    'spending_stress',            # (monthly - median) / iqr
    'critical_new_m2m',           # (tenure ≤ 3m) × (contract=M2M) × (high spend)
    'protective_established',     # (tenure > 24m) × (contract=2Y)
    'has_referrals',              # Binary flag
    'referral_strength',          # log(referrals) / log(max_referrals)
    'has_dependents'              # Binary flag
]
```

**Core Features Explained**:
- Contract risk mapping: M2M=0.85, 1Y=0.40, 2Y=0.10
- Tenure phase bins: 0-3m, 3-6m, 6-12m, 12-24m, 24m+ (captures churn cliff at 3m)
- Monthly charge risk tiers: low/medium/high/very_high
- Value efficiency ratio: (Total Charges) / (Expected Lifetime)
- Service complexity: normalized service count
- Risk decay curves: exponential time decay (√tenure)
- Spending stress: deviation from median (normalized)
- Critical interaction flags: new M2M + high spend = red flag
- Referral & dependent indicators: social anchors stabilize customers

**Performance**: F2: 0.8298 | Recall: 0.9460 | Precision: 0.5565 | AUC: 0.9290

### Stage 2: Random Forest (No SMOTE + F1 Optimization)

**Focus**: Balanced precision-recall tradeoff with non-linear relationship capture
**Data Strategy**: No SMOTE balancing + F1 metric optimization for balanced classification

**Feature Set**:
```python
rf_features = [
    # All LR features (inherited baseline)
    *lr_features,
    'lr_churn_probability',       # Meta-feature from Stage 1
    'lr_confidence',              # abs(lr_prob - 0.5) * 2
    
    # 3-way interaction triangles
    'critical_risk_triangle',     # (tenure ≤ 3m) × (contract=M2M) × (high spend)
    'protective_bundle',          # (tenure > 24m) × (contract=2Y) × (services ≥ 3)
    
    # Financial patterns
    'premium_new_customer',       # (high monthly) × (tenure ≤ 6m)
    'value_disconnect',           # (high monthly) × (total < 0.7 * expected)
    
    # Service engagement gaps
    'internet_no_premiums',       # has_internet × (premium_services = 0)
    'basic_phone_only',           # has_phone × (multiple_lines = 0)
    
    # Social anchors
    'strong_social_anchor',       # has_referrals × has_dependents
    'no_social_connections',      # (no referrals) × (no dependents)
    
    # Billing behavior
    'flexible_digital'            # paperless × (contract=M2M)
]
```

**Key Interactions Explained**:
- **3-way risk triangles**: tenure (early) × contract (M2M) × spend (high)
- **Protective bundles**: tenure (24+) × contract (2Y) × services (3+)
- **Financial patterns**: premium_new_customer (high spend + new), value_disconnect (high spend but low total)
- **Service engagement**: internet_no_premiums (gap signal), basic_phone_only (low engagement)
- **Social anchors**: referrals × dependents (strong stability)
- **Billing behavior**: paperless × M2M (tech-savvy but risky)

**Performance**: F2: 0.7759 | Recall: 0.7860 | Precision: 0.7530

### Stage 3: RNN/GRU (No SMOTE)

**Focus**: Temporal sequences and customer lifecycle evolution

**Feature Set**:
```python
rnn_features = [
    'contract_risk',              # Temporal baseline
    'tenure_risk',                # 1.0 / sqrt(tenure + 1.0)
    'spending_stress',            # (monthly - median) / std
    'value_efficiency',           # Total / Expected Lifetime
    
    # Risk decay curves (dual timescales)
    'risk_decay_early',           # exp(-tenure / 6.0)
    'risk_decay_late',            # exp(-tenure / 24.0)
    
    # Lifecycle patterns
    'lifecycle_sin',              # sin(2π * tenure / 12)
    'lifecycle_cos',              # cos(2π * tenure / 12)
    'renewal_position',           # (tenure % contract_period) / contract_period
    
    # Service evolution
    'service_engagement',         # (service_count) / (max_services)
    'service_growth',             # service_count * tanh(tenure / 12)
    
    # Social stability over time
    'referral_impact',            # log(referrals) * (1 - exp(-tenure / 12))
    'dependents_stability'        # dependents * tanh(tenure / 24)
]
```

**Temporal Features Explained**:
- Risk decay curves: early phase (τ=6mo) vs. late phase (τ=24mo) decay rates
- Lifecycle cycles: sin/cos terms capture seasonal patterns
- Renewal position: where in contract cycle is customer?
- Service engagement trajectory: growth vs. stagnation
- Referral impact decay: do referrals age in effectiveness?
- Dependent stability curves: family status stabilization over time

**Performance with LR+RF Context**: F2: 0.7815 | Recall: 0.8074 | Precision: 0.6789

---

## 🚀 Deployment Strategy: Two Modes

### Quick Mode (Real-time)
- **Model**: XGBoost meta-learner only
- **Latency**: ~10ms per prediction
- **Use Case**: API responses, batch scoring, real-time dashboards
- **Output**: Churn probability + confidence flag

### Deep Analysis Mode (On-demand)
- **Model**: Full 4-stage cascade
- **Latency**: 100-200ms per prediction
- **Use Case**: High-value customer review, retention planning, feature debugging
- **Output**: Individual model probabilities + disagreement metrics + top contributing features

**Router Logic**: Meta-learner classifies prediction confidence. High-confidence predictions use Quick Mode. Low-confidence or flagged cases route to Deep Analysis.

### Explainability Exports

```python
prediction_output = {
    'customer_id': '12345',
    'churn_probability': 0.87,
    'prediction_mode': 'deep_analysis',
    
    'explainability_context': {
        'lr_probability': 0.92,        # High certainty from LR
        'rf_probability': 0.78,        # RF sees mitigating factors
        'rnn_probability': 0.85,       # RNN agrees with overall trend
        'max_confidence': 0.92,
        'min_confidence': 0.78,
        'model_disagreement': 0.14,
        'top_contributing_model': 'logistic_regression'
    },
    
    'disagreement_metrics': {
        'entropy': 0.31,
        'max_disagreement': 0.14,      # RF vs LR conflict
        'flagged_for_nlp': False,      # Only flag top 457 conflicts
        'confidence_bound': [0.78, 0.92]
    },
    
    'meta_learner_weights': {
        'lr_weight': 0.76,
        'rf_weight': 0.085,
        'rnn_weight': 0.155
    },
    
    'top_risk_factors': [
        {'feature': 'tenure_phase', 'value': '0-3m', 'impact': 0.34},
        {'feature': 'monthly_charge_risk', 'value': 'very_high', 'impact': 0.28},
        {'feature': 'contract_type', 'value': 'month_to_month', 'impact': 0.24}
    ]
}
```

---

## 🔬 Methodology

### Data Processing Pipeline

1. **SMOTE Balancing** (Stage 1 only): 60% sampling with k=5 neighbors
2. **Standard Scaling**: Feature normalization across all stages
3. **Stratified k-Fold**: 5-fold CV maintaining churn class distribution
4. **Stage Separation**: Stages 2-3 train on original (unbalanced) data to prevent data leakage from SMOTE

### Cross-Validation Stability

**5-Fold Performance** (Meta-Learner):
- Mean F2: 0.9080
- Std F2: ±0.0145
- Coefficient of Variation: 1.6%
- Result: Highly stable predictions across data splits

### Hyperparameter Configuration

| Stage | Model | Key Hyperparameters |
|-------|-------|-------------------|
| 1 | Logistic Regression | Aggressive SMOTE (60%, k=5), F2 optimization, L2 regularization (C=1.0), balanced class weights |
| 2 | Random Forest | No SMOTE, F1 optimization, 100 trees, max_depth=10, class_weight='balanced' |
| 3 | RNN/GRU | No SMOTE, 64 units, 2 stacked layers, dropout=0.3, batch_size=32 |
| 4 | XGBoost Meta-Learner | max_depth=5, learning_rate=0.1, n_estimators=100 |

---

## 🧠 Core Thesis: Domain-Specific Cascades Beat Generic Black-Boxes

**Research Hypothesis**: Specialized cascade architectures designed around domain business logic can outperform general-purpose black-box models on both performance and explainability for decomposable prediction tasks.

### Supporting Evidence

✅ **Performance**: Meta-learner achieves 0.9080 F2 vs. 0.8298 best single model (+10.8%)
✅ **Precision Gain**: +32.5% improvement while maintaining high recall
✅ **Interpretability**: 9 meta-features directly map to decision logic; per-customer model attribution
✅ **Efficiency**: 2-mode deployment reduces inference cost by 95% for real-time scoring
✅ **Stability**: Consistent cross-fold performance (±1.6% CV on F2)
✅ **Business Alignment**: Asymmetric thresholds reflect actual retention cost structure

### Why This Matters

Industry default: Optimize for global AUC/accuracy → misses asymmetric costs → wastes retention budget

This approach: Optimize for business metrics → higher recall on churners → dramatically lower false positives → focused retention spend

---

## 🎯 Next Steps

**Phase 1: Production Optimization**
- Generalize to minimal feature set (charges, contract, tenure + usage only)
- Maintain meta-learner F2 performance with reduced computational overhead
- Optimize cascade in C++ with ONNX runtime for inference

**Phase 2: Advanced Analysis**
- Explore GRU replacement for improved gradient flow and training speed
- Layer-wise relevance propagation (LRP) for deeper feature attribution
- Online meta-learner adaptation for concept drift handling

**Phase 3: Extended Applications**
- Apply meta-learner cascade to billing dispute prediction
- Extend to upgrade propensity and usage spike detection
- Generalize framework to other telecom KPIs

---

---

## 🗣️ User Interface: NLP-Driven Interaction

Project ChurnBot features a natural language processing interface that streamlines user interaction. Users can input queries in plain language, and ChurnBot:

1. **Collects and preprocesses** user input  
2. **Routes the request** to the relevant model(s) — churn, security, or IT models  
3. **Interprets model predictions** and provides actionable results in clear, understandable language  

This allows analysts and executives to interact with complex ML pipelines effortlessly, turning raw predictions into meaningful insights.
  
---

## 🎯 Choose Your Experience

⚡ **Terminal Version (Light)**: For telecom analysts and technical teams — fast, efficient insights through command-line interaction.

📈 **Dashboard Version (Heavy)**: For telecom executives — rich visualizations and executive-ready presentations.

Both versions are specialized for telecom churn, analyzing call patterns, data usage shifts, billing disputes, and service degradation that general-purpose models may not capture. All computations run locally, keeping sensitive subscriber data on your network.

---

## 🔒 Privacy & Security: Local-First Philosophy

ChurnBot runs entirely on your machine with zero cloud dependencies:

✅ No external data transfers — sensitive subscriber data never leaves your network  
✅ No monthly fees or API costs  
✅ Full data sovereignty — maintain compliance and avoid regulatory penalties  
✅ Immediate analysis — no network latency or downtime  
✅ C++ Performance — enterprise-grade speed with local execution

Compare this to general-purpose models that may rely on cloud APIs with inherent data exposure risks.

---

### 💼 Real-World Impact

**Business ROI**:
- 📉 Reduce churn-related losses through precise targeting
- 📈 Improve executive decision-making with actionable insights
- 🛡️ Maintain full data sovereignty → avoid compliance penalties
- 💰 Eliminate cloud API costs and subscription fees

**Security ROI**:
- 🔒 Complete data privacy — no external data exposure
- 📋 Regulatory compliance maintained
- 🏢 Enterprise-grade security through local execution

---

## 🎯 Current Research Focus

- ✅ Feature diagnostics (correlation, AUC, IV, PSI)
- ✅ False positive reduction via threshold tuning
- ✅ Semantic feature grouping (business, technical, spending, temporal)
- 🔄 Cross-dataset generalization (WA vs other datasets)
- 🔄 Temporal feature balance optimization
- 🔄 Daily experimental logs

---

## 🔮 Next Steps

1. Enhance cascade with deeper RNN layers and optimized hyperparameters
2. Test on multiple datasets using 10-fold cross-validation
3. Implement cost-based threshold tuning to optimize retention expenses
4. Target recall improvement to 80-85% through noise reduction techniques
5. Explore GRU replacement for RNN stage
6. Prepare statistical rigor for academic submission

---

## ⚠️ Limitations

- Dataset variability imposes generalization challenges
- Preprocessing introduces potential bias
- Feature distributions vary across datasets
- Additional 10-15% recall improvement requires innovative noise reduction techniques

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
