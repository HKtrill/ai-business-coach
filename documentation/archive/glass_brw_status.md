# Project Status & TODO Roadmap

This document summarizes the current research and engineering roadmap for Project ChurnBot.

ChurnBot has moved beyond early architectural prototyping. The current cascade now has a validated working baseline with finalized or near-finalized stage-specific feature sets, deployable notebook integration, corrected arbitration logic, and empirical evidence that the GLASS-BRW cascade topology reduces redundant ensemble behavior.

The current phase focuses on validation, ablation studies, fair baseline comparisons, and documentation needed to justify each architectural choice.

---

## ✅ Current Status

### Cascade Baseline Established

- Four-stage interpretable cascade architecture is implemented:
  - Stage 1: Logistic Regression
  - Stage 2: Sequential GLASS-BRW
  - Stage 3: Explainable Boosting Machine
  - Stage 4: Meta-EBM / Weighted confidence arbiter

- Stage-specific feature engineering has been ported from exploratory research into the deployable cascade notebook.
- LR feature engineering, tuning, and calibration are integrated.
- GLASS-BRW feature/binning strategy is integrated as the current symbolic-routing baseline.
- EBM feature engineering and tuning strategy are integrated as a strong additive nonlinear baseline.
- Meta-EBM / arbiter logic has been corrected and simplified.
- Abstention logic now uses confidence-weighted arbitration rather than outdated confidence-band logic.

### Architecture Justification Baseline

The current cascade addresses the earlier high-agreement ensemble failure mode observed in the LR + RF + EBM design.

| Metric | Previous LR + RF + EBM | Current LR + EBM + GLASS-BRW |
|---|---:|---:|
| All-stage agreement | 88.8% | 75.3% |
| All-wrong-together | 17.7% | 4.1% |
| Redundant positive capture | 69.0% | 13.5% |

Replacing RF with GLASS-BRW changed the cascade topology from a highly redundant ensemble into a more complementary interpretable decision system.

The current architecture is therefore considered a defensible working baseline. Future work focuses less on inventing the cascade and more on validating why each stage, feature set, binning choice, and arbitration rule exists.

### Dataset Audit

- Current validation uses the UCI Bank Marketing term-deposit dataset
- `duration` is treated as direct post-outcome leakage
- `poutcome` and `pdays` are treated as historically dependent prior-campaign signals
- Current modeling focus is pre-contact subscription prediction rather than post-call outcome analysis
- Full dataset audit details are documented in [`dataset_audit_and_research_status.md`](dataset_audit_and_research_status.md)

### Local-First Execution

- ChurnBot runs locally with no external services, API keys, or cloud dependencies
- Setup and hardware details are documented in [`setup_and_hardware.md`](setup_and_hardware.md)

---

## 🔬 Active Research Priorities

### 1. Fair Baseline and Ablation Comparisons

The next validation phase focuses on fair comparisons that justify the current architecture.

Planned comparisons include:

#### RF / GLASS-BRW Justification

- Raw RF baseline
- Binned baseline RF
- Feature-engineered continuous RF
- Feature-engineered binned RF
- GLASS-BRW symbolic rule variant

Purpose:

- Show why feature engineering is necessary
- Quantify the cost of binning alone
- Show how feature engineering mitigates binning-related expressivity loss
- Justify binary conjunction features for GLASS-BRW rule generation
- Explain why GLASS-BRW is used instead of a standard RF stage

### 2. EBM Validation

Planned comparisons:

- Baseline EBM on raw/default features
- Feature-engineered EBM
- Reduced feature-engineered EBM

Purpose:

- Justify EBM feature engineering
- Compare default features against engineered nonlinear structure
- Validate whether the current EBM strategy should remain locked
- Confirm whether specific engineered features should remain in the final EBM configuration

### 3. Meta-EBM / Arbiter Validation

Planned checks:

- Audit threshold recomputation
- Verify abstention math
- Validate coverage handling
- Inspect prediction and abstention scatter plots
- Test additional tuning strategies where useful
- Confirm metric reproducibility
- Evaluate whether abstention improves the covered-region decision quality

---

## Planned Engineering Work

- Update project structure documentation as the modular refactor continues
- Improve documentation around stage interfaces and pipeline execution
- Add clearer examples for running core experiments locally
- Continue organizing legacy documents into `documentation/archive/`
- Prepare cleaner public-facing documentation for research review

---

## 📄 Publication Direction

The project is moving toward a formal research write-up centered on the thesis:

> Glass-box cascade architectures can reduce redundant ensemble behavior while preserving interpretable, stage-by-stage decision logic.

The next phase focuses on answering the architectural defense questions:

- Why these features?
- Why binning?
- Why GLASS-BRW?
- Why EBM?
- Why abstention?
- Why weighted confidence arbitration?
- Why a cascade instead of a standalone model?

Answering these questions through ablations, diagnostics, and comparison tables will make the final paper easier to write and defend.

---

## Interpretation Note

Reported metrics should be interpreted in the context of the project’s stricter deployment-oriented setup.

Many benchmark results on the Bank Marketing dataset include post-outcome or historically dependent fields such as `duration`, `poutcome`, and `pdays`.

ChurnBot intentionally evaluates a harder pre-contact prediction problem using interpretable, auditable modeling stages.