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
> ⚠️ **Research Status & Dataset Transition Notice**
>
> This project is under active research and architectural refinement.
> While the core methodology and glass-box cascade design are stable,
> results and documentation are currently being revalidated.
>
> Notable updates:
> - The system is transitioning from the widely used synthetic Telco churn dataset
>   to a **real-world bank marketing subscription dataset** to ensure external validity.
> - As a result, performance metrics, feature attributions, and examples in this README
>   will be updated once the next cascade iteration is finalized and validated.
> - Earlier experimental components (e.g., RNN-based stages) are no longer part of the
>   canonical pipeline but may remain in the repository for historical reference.
>
> **Current research objective:**  
> Predicting **subscription to term deposits** using a fully interpretable,
> abstention-aware glass-box ensemble architecture.

---
## 📖 Synopsis
Project ChurnBot is a research-driven, glass-box decision intelligence system for modeling customer retention and subscription behavior using fully interpretable cascade architectures. Rather than treating churn or subscription as a single black-box prediction task, the system decomposes reasoning into explicit, interpretable stages that capture linear effects, interaction-driven rules, and non-linear response curves.

The cascade serves as the core reasoning engine, producing abstention-aware, fully explainable predictions. An NLP-based interface provides a natural language layer for interacting with model outputs and explanations. Ongoing research explores a custom, partially interpretable transformer-based interface, though this component remains experimental and is not required for the core system.

The result is a transparent, high-performance ensemble where every decision can be traced back to human-readable logic—enabling trustworthy deployment without sacrificing predictive power.

![Dataset Overview](assets/dataset_overview.png)
*Dataset characteristics: churn distribution peaks at early tenure, specific monthly charge ranges, and contract types.*

---

## 🚨 Problem: The Interpretability-Performance Trade-off Myth

The ML industry perpetuates a harmful misconception: **"You must sacrifice accuracy for interpretability."**

This leads to:
- Black-box models deployed in high-stakes churn prediction where explanations are critical
- Business teams unable to understand or trust model decisions
- Missed opportunities for actionable retention strategies
- Regulatory compliance risks in customer intervention policies

**Current Industry Practice**: Deploy XGBoost/Neural Networks and hope stakeholders trust the predictions. When explanations are needed, use post-hoc methods (SHAP, LIME) that approximate—not reveal—the actual decision logic.

**Our Solution**: A fully interpretable glass box cascade that **outperforms** traditional black-box approaches while providing complete transparency. Every prediction can be explained through explicit rules, linear coefficients, and additive shape functions.

---

## 🎯 Architecture: 100% Glass Box Four-Stage Cascade
```
Stage 1: Logistic Regression (Linear Coefficients – SMOTE + F2)
  ↓ Captures linear relationships with interpretable coefficients
  
Stage 2: Rule-Based Random Forest (Explicit Decision Rules – No SMOTE + F1)
  ↓ 100 interpretable IF-THEN rules for interaction patterns
  
Stage 3: Explainable Boosting Machine (Additive Shape Functions – No SMOTE)
  ↓ Non-linear patterns via interpretable shape functions
  
Stage 4: Meta-EBM (Glass Box Ensemble Synthesis)
  ↓ Learns optimal model weighting through interpretable importance
  
Customer-Level Predictions with Complete Explainability
```

### Key Innovation: Every Stage is Interpretable

- **Logistic Regression**: Direct coefficient inspection (e.g., "M2M contract increases log-odds by +2.3")
- **Rule-Based RF**: Explicit rules (e.g., "IF tenure≤3 AND contract=M2M AND charge>$75 THEN 85% churn probability")
- **EBM**: Additive shape functions (e.g., "Risk drops exponentially after 12 months tenure")
- **Meta-EBM**: Feature importance reveals model weighting strategy (e.g., "RF predictions weighted 5x higher in disagreement cases")

---

## 📊 Performance Metrics

### Meta-EBM Final Results (5-Fold CV)

| Metric | Score | vs. Previous Architecture |
|--------|-------|---------------------------|
| **F2-Score** | **0.9212** | **+1.3%** |
| **Recall** | **0.9276** | **+1.4%** |
| **Precision** | **0.8964** | **+0.8%** |
| **AUC-ROC** | **0.9872** | **+0.1%** |

**Interpretation**: Captures 92.76% of churners while maintaining 89.64% precision—achieving both higher recall AND higher precision than the previous black-box architecture.

### Individual Stage Performance

| Stage | F2 | Recall | Precision | AUC | Key Strength |
|-------|-----|--------|-----------|-----|--------------|
| Logistic Regression | 0.7725 | 0.7828 | 0.7337 | 0.9271 | Interpretable linear baseline |
| Rule-Based RF | 0.8686 | 0.8794 | 0.8283 | 0.9680 | Explicit decision logic |
| EBM | 0.7420 | 0.7292 | 0.7977 | 0.9281 | Non-linear shape functions |
| **Meta-EBM Cascade** | **0.9212** | **0.9276** | **0.8964** | **0.9872** | **Optimal glass box weighting** |

### Architecture Comparison

| Model | F2 | Recall | Precision | Interpretability |
|-------|-----|--------|-----------|------------------|
| Previous (RF+RNN+XGBoost) | 0.9080 | 0.9133 | 0.8880 | Partial (LR only) |
| **Glass Box Cascade** | **0.9212** | **0.9276** | **0.8964** | **100% transparent** |
| Improvement | **+1.3%** | **+1.4%** | **+0.8%** | **Complete explainability** |

**Key Result**: Glass box architecture OUTPERFORMS black box while providing complete transparency—directly contradicting the accuracy-interpretability trade-off myth.

This demonstrates that the cascade reliably reduces false positives without sacrificing churn coverage. Fewer false positives translate directly to more focused retention spending and higher ROI.

---

## 🔬 Key Innovations

### 1. Rule-Based Random Forest Conversion

**Challenge**: Random Forests with hundreds of trees are black boxes—no one can understand 459 trees voting.

**Solution**: Extract and consolidate decision paths into explicit rules.

**Process**:
1. Extract 59,478 decision paths from 459-tree RF
2. Consolidate similar rules (similarity threshold=0.85)
3. Rank by coverage × precision × F1
4. Keep top 100 high-performance rules

**Result**: 
- Correlation with original RF: 0.9999 (identical predictions)
- Every prediction explainable via IF-THEN rules
- Example rule: `IF tenure≤3 AND contract=M2M AND monthly_charge>75 THEN churn_prob=0.85`

**EBM Advantages**:
- Additive shape functions show exact non-linear relationships
- Pairwise interaction detection (e.g., tenure × contract)
- No hidden layers—complete transparency
- Visualizable risk curves (e.g., "risk drops exponentially after 12 months")

**Configuration**: max_bins=256, interactions=10, learning_rate=0.01

### 3. Meta-EBM Ensemble Layer

**Previous Approach**: XGBoost meta-learner (black box)

**New Approach**: Meta-EBM learns to weight base models interpretably

**Meta-Features** (9 total):
```python
meta_features = [
    'lr_prob',              # Base model predictions
    'rf_prob',
    'ebm_prob',
    'lr_rf_disagree',       # Disagreement signals
    'lr_ebm_disagree',
    'rf_ebm_disagree',
    'max_confidence',       # Confidence bounds
    'min_confidence',
    'std_confidence'        # Ensemble uncertainty
]
```

**Meta-EBM Feature Importance** (reveals weighting strategy):
- `rf_prob`: 5.02 (RF predictions most trusted)
- `lr_rf_disagree`: 1.84 (model conflicts trigger special handling)
- `rf_ebm_disagree`: 0.56
- `lr_prob`: 0.53

**Meta-Learner Decision Logic**:
- **High-confidence cases** (low std): Trust individual model with highest confidence
- **Conflicted cases** (high std, disagreement): Use entropy-weighted ensemble averaging
- **Low min_confidence**: Route to detailed analysis mode for retention team

---

## 📈 Key Insights & Attribution

### Contribution Attribution

Meta-EBM learned to weight base models adaptively per customer:
- **Rule-Based RF**: 5.02 importance – most trusted for balanced predictions
- **LR-RF Disagreement**: 1.84 importance – triggers special ensemble logic
- **RF-EBM Disagreement**: 0.56 importance – secondary conflict signal
- **Logistic Regression**: 0.53 importance – baseline linear patterns

Some high-risk customers are confidently flagged by LR's linear patterns, while others require RF's rule-based analysis or EBM's shape functions.

### Disagreement Analysis

High-disagreement cases where models strongly diverge are flagged for:
- Specialized handling by retention teams
- Feature-importance analysis to understand model conflicts
- Manual review of edge cases

**Business Value**: These customers receive individualized analysis rather than generic scoring.

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

**Performance**: F2=0.7725, Recall=0.7828, Precision=0.7337, AUC=0.9271

### Stage 2: Rule-Based Random Forest (No SMOTE + F1 Optimization)

**Focus**: Balanced precision-recall tradeoff with explicit decision rules

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

**Sample Extracted Rules**:
```
Rule #1: CHURN (confidence: 87%)
  IF tenure_months <= 3.0 AND 
     contract_risk >= 0.85 AND 
     monthly_charge > 75.0
  THEN churn_probability = 0.85
  Coverage: 12.3% | Precision: 89.2%

Rule #2: STAY (confidence: 92%)
  IF tenure_months > 24.0 AND 
     contract_risk <= 0.10 AND 
     service_count >= 3
  THEN churn_probability = 0.08
  Coverage: 8.7% | Precision: 91.5%
```

**Performance**: F2=0.8686, Recall=0.8794, Precision=0.8283, AUC=0.9680

### Stage 3: Explainable Boosting Machine (No SMOTE)

**Focus**: Non-linear patterns via additive shape functions

**Data Strategy**: No SMOTE balancing, learns directly from natural class distribution

**Feature Set**:
```python
ebm_features = [
    'contract_risk',              # Baseline risk
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
- Referral impact decay: effectiveness over time
- Dependent stability curves: family status stabilization over time

**Performance**: F2=0.7420, Recall=0.7292, Precision=0.7977, AUC=0.9281

**Explainability**: 
- Shape functions show exact risk curves (e.g., "Risk drops 60% after 12 months")
- Pairwise interactions reveal compound effects
- Additive decomposition: contribution of each feature visible

---

### Explainability Exports
```python
prediction_output = {
    'customer_id': '12345',
    'churn_probability': 0.87,
    'prediction_mode': 'glass_box_analysis',
    
    'explainability_context': {
        'lr_probability': 0.92,        # High certainty from LR
        'rf_probability': 0.78,        # RF sees mitigating factors via rules
        'ebm_probability': 0.85,       # EBM agrees with overall trend
        'max_confidence': 0.92,
        'min_confidence': 0.78,
        'model_disagreement': 0.14,
        'top_contributing_model': 'rule_based_rf'
    },
    
    'disagreement_metrics': {
        'entropy': 0.31,
        'max_disagreement': 0.14,      # RF vs LR conflict
        'confidence_bound': [0.78, 0.92]
    },
    
    'meta_ebm_weights': {
        'rf_importance': 5.02,         # Most trusted
        'lr_rf_disagree': 1.84,        # Conflict signal
        'lr_importance': 0.53
    },
    
    'matching_rules': [
        {
            'rule_id': 23,
            'conditions': 'tenure<=3 AND contract=M2M AND charge>75',
            'churn_prob': 0.85,
            'confidence': 0.87
        }
    ],
    
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
4. **Stage Separation**: Stages 2-4 train on original (unbalanced) data to prevent data leakage from SMOTE

### Cross-Validation Stability

**5-Fold Performance** (Meta-EBM):
- Mean F2: 0.9212
- Std F2: ±0.0000
- Coefficient of Variation: 0.0%
- Result: Perfect stability across data splits (random_state=42)

### Hyperparameter Configuration

| Stage | Model | Key Hyperparameters |
|-------|-------|---------------------|
| 1 | Logistic Regression | Aggressive SMOTE (60%, k=5), F2 optimization, L2 (C=1.0), balanced class weights |
| 2 | Rule-Based RF | F1 optimization, top_k_rules=100, similarity=0.85, min_samples=10 |
| 3 | EBM | max_bins=256, interactions=10, learning_rate=0.01 |
| 4 | Meta-EBM | max_bins=128, interactions=5, learning_rate=0.01 |

---

## 🧠 Core Thesis: Glass Boxes Can Outperform Black Boxes

**Research Hypothesis**: Carefully designed glass box ensembles can match or exceed black-box performance while providing complete transparency—particularly in domains with strong structural patterns like churn prediction.

### Evidence Highlights

- **Performance**: Glass box architecture achieved +1.3% F2, +1.4% Recall, +0.8% Precision vs. previous black-box approach
- **Stability**: Zero variance across 5-fold CV (perfect reproducibility)
- **Interpretability**: 100% of predictions explainable through linear coefficients, explicit rules, and shape functions
- **Business Value**: Stakeholders can understand and trust predictions, enabling targeted retention strategies

---

## 🗣️ User Interface: NLP-Driven Interaction

Project ChurnBot features a natural language processing interface that streamlines user interaction. Users can input queries in plain language, and ChurnBot:

1. **Collects and preprocesses** user input  
2. **Routes the request** to the relevant model(s) with full glass box transparency
3. **Interprets model predictions** and provides actionable results with explicit reasoning

This allows analysts and executives to interact with complex ML pipelines effortlessly, turning raw predictions into meaningful insights with complete explainability.

---

## 🎯 Choose Your Experience

⚡ **Terminal Version (Light)**: For telecom analysts and technical teams — fast, efficient insights through command-line interaction with full rule/coefficient visibility.

📈 **Dashboard Version (Heavy)**: For telecom executives — rich visualizations of shape functions, rule networks, and model weights for executive-ready presentations.

Both versions maintain 100% interpretability, analyzing call patterns, data usage shifts, billing disputes, and service degradation with complete transparency. All computations run locally, keeping sensitive subscriber data on your network.

---

## 🔒 Privacy & Security: Local-First Philosophy

ChurnBot runs entirely on your machine with zero cloud dependencies:

✅ No external data transfers — sensitive subscriber data never leaves your network  
✅ No monthly fees or API costs  
✅ Full data sovereignty — maintain compliance and avoid regulatory penalties  
✅ Immediate analysis — no network latency or downtime  
✅ Complete interpretability — every prediction fully explainable for audit trails

Compare this to black-box cloud APIs with inherent data exposure risks and unexplainable predictions.

---

### 💼 Real-World Impact

**Business ROI**:
- 📉 Reduce churn-related losses through precise targeting (+1.4% recall improvement)
- 📈 Improve executive decision-making with actionable insights (explicit rules + shape functions)
- 🛡️ Maintain full data sovereignty → avoid compliance penalties
- 💰 Eliminate cloud API costs and subscription fees
- 🎯 Reduce false positives by 8.4% → more focused retention spending

**Security & Compliance ROI**:
- 🔒 Complete data privacy — no external data exposure
- 📋 Regulatory compliance through complete audit trail
- 🏢 Enterprise-grade security through local execution
- 📊 Explainable AI for high-stakes decisions (GDPR, fair lending compliance)

---

## 🎯 Current Research Focus

- ✅ Full glass box architecture achieved
- ✅ Rule extraction from Random Forest (59k paths → 100 rules)
- ✅ EBM integration for non-linear patterns
- ✅ Meta-EBM for interpretable ensemble weighting
- 🔄 Cross-dataset validation (telecom, SaaS, retail)
- 🔄 Interactive visualization tools
- 🔄 Research paper preparation

---

## ⚠️ Limitations

- Dataset variability imposes generalization challenges
- Rule consolidation requires domain expertise for threshold tuning
- Glass box conversion adds one-time computational overhead
- Shape function interpretability requires statistical literacy

---

*ChurnBot: Proving that you can have your cake (performance) and eat it too (interpretability).*

---

## 📚 Dataset Sources & Citations

### **1) Bank Marketing – Term Deposit Subscription (Current Benchmark)**

This project uses the **Bank Marketing** dataset for primary empirical evaluation.
The dataset is publicly available for research use via the UCI Machine Learning Repository.
According to the dataset documentation, **both citations below are required**.

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
