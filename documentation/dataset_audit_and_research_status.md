# Dataset Audit & Research Status

This document summarizes ChurnBot’s transition from early Telco churn prototyping to deployment-oriented validation on the UCI Bank Marketing term-deposit dataset.

---

## Notable Updates

- The project transitioned from the synthetic IBM Telco churn dataset,
  used for early architectural validation, to a real-world bank marketing
  subscription dataset to improve external validity.

- During temporal validity and leakage auditing of the bank marketing dataset,
  multiple structural issues were identified that materially affect deployment-oriented modeling:

  - **`duration`** encodes post-outcome information and constitutes direct label leakage.

  - **`poutcome`** contains outcome-derived information from prior campaigns. While technically
    available at prediction time, it induces historically dependent predictions rather than
    actionable behavioral patterns.

  - **`pdays`** strongly correlates with `poutcome` and acts as a numeric surrogate for the same signal,
    effectively encoding prior campaign state through recency information.

  - Approximately **82% of samples collapse into a single `unknown` regime**,
    representing prospects with no prior campaign history, severely limiting meaningful segmentation once
    history-dependent features are removed.

  - Removing these variables produces a substantial performance drop, suggesting that much of the
    dataset’s predictive power originates from prior campaign artifacts rather than customer behavior.

- As a result, the project is actively evaluating the bank marketing dataset under stricter
  temporal and interpretability constraints while also exploring alternative real-world
  subscriber/churn datasets for comparative validation.

- **Performance metrics are intentionally not directly comparable to many published benchmarks on this dataset.**

  This system predicts subscription likelihood **before contact occurs** using only pre-contact features,
  whereas many benchmark pipelines include post-outcome or history-dependent variables such as
  `duration`, `poutcome`, and `pdays`.

  This reflects a deliberate choice to solve the harder, deployment-realistic problem of determining
  *whether to initiate contact* rather than performing post-hoc analysis of *what happened during the call*.

- **Critically, these issues were first surfaced by the system itself via the interpretable rule lattice.**

  Following removal of `duration`, rule generation became dominated by `poutcome_success`
  rules exhibiting high precision but extremely low coverage — a signature of shortcut learning
  rather than meaningful behavioral structure.

  This anomaly triggered deeper temporal inspection of `poutcome` and `pdays`,
  ultimately leading to confirmation of historical dependency and regime collapse.

---

## Dataset Strategy

**Research validation:**  
Bank Marketing dataset (UCI ML Repository) with full temporal auditing and leakage analysis documented above.

**Deployment demonstration:**  
Synthetic or permissively licensed datasets, such as IBM Telco Customer Churn, will be used.

This project is designed as a **methodological framework** that organizations can evaluate and adapt
for proprietary subscription, churn, or binary decision-modeling use cases.

---

## Current Research Objective

Develop and validate an abstention-aware, interpretable cascade architecture for
real-world customer decision modeling using rigorously audited, temporally valid data.

---

## Broader Implication

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

## Leakage & Regime Collapse Diagnostics

<p align="center">
  <img src="../assets/pdays_vs_previous.png" width="32%">
  <img src="../assets/boxplot.png" width="32%">
  <img src="../assets/job_by_poutcome_conversion.png" width="32%">
</p>

**Figure — Structural leakage and regime collapse in the bank marketing dataset**

**Left:** `pdays` is tightly coupled with `previous`, indicating that recency largely encodes prior contact history rather than independent behavioral signal.

**Center:** The dominant `poutcome=unknown` regime collapses near zero `pdays`, while non-unknown regimes exhibit strong separation — demonstrating that `pdays` acts as a numeric surrogate for campaign outcome state.

**Right:** Conversion rates diverge sharply only when conditioning on known `poutcome`, while the dominant unknown regime exhibits weak and compressed signal across job segments — confirming historical dependency and loss of meaningful segmentation once history-dependent variables are removed.