## Roadmap

### Phase 2 — Research Organization, Training Research, and Optimized Cascades

#### PR 42 — Feature Research Notebook and Feature-Contract Investigation

**Objective:** Develop the Feature Research Notebook and determine, on training-side evidence, whether the accepted feature representations should be retained or separately revised before training optimization.

**Execution principles:**

- Use training rows only: the notebook draws the cascades' split and never binds a test row, test prediction, or test metric.
- Fit every target-dependent statistic, selection, and transformation within its training fold, and register evaluation criteria before candidate results are examined.
- Preserve the frozen PR 39 baseline, the accepted Phase 1 artifacts, and the production feature code, including `feature_research/feature_engineering/`, which Stage 3 imports for both arms. A revised feature contract is a new, separately versioned contract.
- Reuse existing research modules and accepted evidence code by import (`shared.stage1.crossfit`, `stage2_binning`, `stage2_crossfit`) without modifying them, and add infrastructure only for a demonstrated need.
- Measure predictive effects with the accepted model configurations. Tuning belongs to PR 44, contract adoption and artifact regeneration to PR 45, and temporal robustness execution to PR 49.
- Use targeted tests after code changes; reserve full regression validation for Commit 6.

**Commit order:** Commit 2 precedes the research commits. Commit 3 has the highest priority because its findings can invalidate later conclusions, so it precedes Commits 4 and 5, which are independent of each other. Commit 6 follows all of them.

---

#### Commit 1 — Roadmap Planning and Deferred Engineering Updates

**Objective:** Organize PR 42 into reviewable commits and update the Deferred Engineering entries for the completed PR 41 work. Documentation only.

- [x] Review the PR 42 scope and its dependencies in PRs 43–50 against the current repository.
- [x] Organize PR 42 into commits, retaining its full research scope and safeguards.
- [x] Update the Deferred Engineering entries: the Windows execution warning and PuLP 3.3.2 deprecations resolved in PR 41, a separate PuLP 4.x migration task, and the Joblib/Loky shutdown traceback.
- [x] Make no source-code, notebook, or artifact changes.

---

##### Commit 2 — Feature Research Notebook Foundation and Feature Inventory

**Objective:** Create the train-only Feature Research Notebook and inventory the accepted features and contracts that the research commits evaluate.

- [x] Create the Feature Research Notebook as a new file in `feature_research/`, built from the cells and modules mapped in `documentation/notebook_map.md` (section 2.1). Leave the existing notebooks unchanged.
- [x] Extend notebook tests G5 (one labelled top-to-bottom path) and G7 (the cascades' split protocol; no test row bound) to the new notebook, and read accepted router and upstream-model artifacts only through the PR 38 boundary extractors.
- [x] Write research records under `research_logs/pr42/` with run identifiers, seeds, and code identity, never over an existing record.
- [x] Inventory existing engineered features, feature contracts, and stage-specific feature populations.
- [x] For each feature, record its stage and arm, source variables, whether it is target-derived, where it is fitted (full training split or per-fold refits), and how it is applied to the test split.
- [x] Record the frozen references for later comparison: the Stage 1 feature contracts, the Stage 2 binning identity, the Stage 3 feature DAG and block-cache identity, and the three EBM interaction pairs.
- [x] Record the duplicated feature-engineering code in `shared/stage2/feature_engineering.py` and `feature_research/feature_engineering/` (notebook map section 5) for the Commit 6 decision.

---

##### Commit 3 — Stage 1 Temporal Validity and Target-Derived Feature Audit (high priority)

**Objective:** Determine whether the accepted target-derived features carry direct label leakage, temporal look-ahead, or both, evaluate past-only alternatives, and design the temporal robustness protocol before other feature research builds on these features.

Context from repository inspection: the dataset rows are in date order (May 2008 – November 2010), and the subscription rate rises from about 3–6% to about 45–57% over that period. Two of the three Stage 1 features, `subscribe_rate_by_euribor_bin` and `subscribe_rate_by_dow_month`, are target rates keyed on period-identifying variables, and most euribor quantile bins fall within one to three consecutive months. Under the random split, these encodings are fitted partly on rows contacted later in the same period.

The Stage 1 cross-validations (tuning, OOF, and inner calibration folds) already refit both encodings on each fold's training rows (`shared.stage1.crossfit`, audits F3 and F19), and the full-train fit scores the test split. The audit verifies these refits and addresses a separate question: temporal look-ahead under the random split.

**Step 3A — Encoding Construction and Leakage Audit**

- [x] Establish the time index used throughout PR 42 from the dataset's date order, and record its basis and limits.
- [x] Document exactly how both encodings are constructed, fitted, and applied (training split, Stage 1 fold designs, test mapping).
- [x] Determine whether the protocol contains direct label leakage, temporal look-ahead, or both.
- [x] Verify that every transformation is fitted within its proper training fold.
- [x] Extend the audit to the other target-derived features used by Stages 2–3, such as the neighborhood subscription density.

**Step 3B — Past-Only Encodings and Temporal Robustness Protocol**

- [x] Evaluate past-only (expanding-window) encodings: feature availability, deployment realism, and predictive effect.
- [x] Determine whether candidate revisions require separately versioned feature contracts; do not alter the accepted PR 39 baseline.
- [x] Design the temporal robustness protocol: chronological split, past-only encodings, and a policy for prevalence shift, which affects thresholds, abstention cuts, and the Stage 2 NPV and precision obligations. Execution follows in PR 49.
- [x] Record the interim temporal-validity finding and its implications for the frozen PR 39 evidence used in PR 43, the candidate contracts in PRs 44–45, and the claim boundaries in PR 49.

---

##### Commit 4 — Stage 2 NSD/ECI Binning Drift, Sensitivity, and Candidates

**Objective:** Explain the NSD/ECI lift drift deferred from PR 39, make the bin derivation procedural where feasible, and identify bin candidates for router testing in PR 44, without modifying the frozen binning.

**Step 4A — Drift Reproduction and Diagnosis**

- [x] Start from the PR 39 Step 3A evidence (`stage2_binning`, record `research_logs/pr39/c3_20261114_231604/`, outcome retain for both families): reuse its per-bin statistics, Wilson intervals, and triggers T1–T4 by import, without modifying the record or its code.
- [x] Revisit the substantial lift drift observed during PR 39 Step 3A, particularly:
  - `eci_warm`: original 1.49–1.67× lift versus approximately 1.03× OOF.
  - `eci_hot`: original 3.53× versus approximately 2.50× OOF.
  - `nsd_elevated`: approximately 2.49× pooled OOF lift, with significant fold-level variation.
- [x] Reproduce the original NSD/ECI feature-bin research using the corrected, fold-isolated feature engineering implementation.
- [x] Distinguish threshold-placement problems from encoder behavior, population shifts, and in-sample label leakage.
- [x] Use the Commit 3 findings on the target-derived Stage 2 sources, including period sensitivity, when separating these causes.
- [x] Reevaluate decile-lift distributions, candidate thresholds, bin counts, support, confidence intervals, and fold stability.

**Step 4B — Procedural Binning and Candidate Evaluation**

- [x] Where feasible, make the NSD/ECI bin derivation procedural and fold-fitted, so repeated-split and nested evaluations can refit it rather than reuse constants chosen on all training rows (a limitation recorded in PR 39 Step 3A).
- [x] Register the evaluation criteria before candidate results are examined.
- [x] Investigate whether transitional or weak bins retain meaningful predictive utility through conjunctions with other Stage 2 features.
- [x] Evaluate alternative threshold configurations and bin structures under leakage-safe training protocols, without modifying the frozen Phase 1 baseline.
- [x] Compare candidate configurations against the frozen PR 39 binning baseline using predefined evaluation criteria.
- [x] Identify the candidates that deserve downstream router testing in PR 44 (coverage, constraint feasibility, complementarity, and computational cost).
- [x] Determine whether the existing bins should remain unchanged or whether evidence warrants a separately versioned methodology revision.
- [x] If revision is justified, document the required feature-contract changes, G13 leakage audit, downstream artifact regeneration, and follow-up work rather than silently modifying frozen artifacts.

---

##### Commit 5 — Feature Analysis, Interactions, and EBM Interaction Selection

**Objective:** Use training-side evidence to identify a simpler Stage 3 feature set with interpretable EBM shapes that remain stable across folds and contact periods. Quantify the effects of simplification on predictive performance and calibration, and, where feasible, use fold-fitted interaction selection to assess what reselection recovers under the existing three-pair budget.

#### Step 5A — Feature Analysis and Stage 3 Simplification

- [ ] Review feature distributions, correlations, redundancy, predictive contributions, and stability across contact periods using the Commit 3 time index.
- [ ] Investigate interpretable feature interactions and candidate transformations. Verify that every candidate feature is available at the intended prediction time.
- [ ] Keep feature selection, encoding, and transformation fitting fold-isolated. Use reproducible training-side analyses and preserve shared feature comparability between cascade arms.
- [ ] Starting from the frozen PR 39 Stage 3 inventory, identify the remaining correlated pair and test removing each member separately. For `decay_x_density`, compare the accepted feature with `joint_economic_decay` alone and with a past-only density variant if its inputs are available at prediction time.
- [ ] Compare candidates on identical evaluation rows with the same folds and model settings. Keep unaffected interaction pairs fixed. When removing a feature, record which pairs disappear and leave those slots empty. Report paired out-of-fold performance, calibration, and feature and interaction counts.
- [ ] Assess EBM shape interpretability. Document what each changed feature measures, how its contribution varies over well-supported values, whether that pattern holds across folds and contact periods, and which interactions affect its interpretation. Favor clearer, more stable shapes when the interpretability gain justifies a modest loss in standalone Stage 3 performance. Report the change in predictive performance and calibration.
- [ ] Plot each EBM main-effect shape for the frozen Stage 3 model and the simplification candidates on matched axes. Label the feature values and their contribution to the model score, show how many training rows support each region, and overlay fold-level curves. For retained interactions, plot the pairwise effect with joint support.
- [ ] On identical out-of-fold rows, compare the baseline and candidate EBMs with Stage 1, reporting Stage 2 abstentions separately. Count how often each EBM is right when Stage 1 is wrong and vice versa. Carry promising candidates forward to verify whether that disagreement improves final cascade decisions at comparable coverage under the same operating obligations.

#### Step 5B — Procedural EBM Interaction Selection

- [ ] Where feasible, make EBM interaction selection procedural and fold-fitted so repeated-split and nested evaluations can rerun selection within each training fold.
- [ ] On the frozen PR 39 feature set, check whether selection on all training rows recovers the three accepted pairs. Record how often each pair is selected across folds.
- [ ] For candidates carried forward from 5A, rerun selection under the same three-pair budget. Compare with the corresponding 5A result on the same evaluation rows. Report which pairs recur and how reselection changes predictive performance and calibration. Compare different interaction budgets in PR 44 (Step 44C).

---

#### Commit 6 — Feature-Contract Recommendations, Validation, and PR Closure

**Objective:** Consolidate the PR 42 findings into feature-contract recommendations, validate the frozen baseline, and give readers direct access to the executed research notebooks from the root README and PR description.

##### Step 6A — Feature-Contract Recommendations and Research Handoff

- [ ] Record the candidate configurations, supporting evidence, rejected alternatives, and remaining uncertainties.
- [ ] Retain the frozen feature contracts unless the evidence supports a separately versioned change. For each recommended revision, document its leakage audit and required downstream artifact regeneration before adoption through PRs 44–45.
- [ ] Hand off the Stage 1 encoding candidates to Step 44A, Stage 2 bin and filter candidates to Step 44B, the EBM interaction procedure and budget comparison to Step 44C, and the temporal protocol and procedural feature refits to PR 49.
- [ ] Decide where the Stage 3 feature DAG belongs and whether to retire the copies in `shared/stage2/feature_engineering.py` identified in Section 5 of the notebook map. Defer any move that would change accepted feature values or recorded identities to the adopting contract version.

##### Step 6B — README, Repository Layout, and Executed-Notebook Packaging

- [ ] Rewrite the opening of the root `README.md` as a 15–30 second project pitch beneath the title and logo. Explain what ChurnBot predicts, how the GLASS cascade works, and why the research compares it with a matched black-box cascade.
- [ ] Place a **Download executed notebooks (.zip)** link immediately below the pitch, before the tech stack and installation instructions. Update the repository layout and notebook map so readers can locate the tracked source notebooks and their executed copies.
- [ ] Add a small packaging script that reuses existing executed notebooks and takes explicitly accepted runs as input. Preserve notebook source paths in the project-wide archive and run IDs in PR-specific filenames. Exclude partial and failed runs.
- [ ] Build `executed_notebooks.zip` with the latest accepted executed copy of each project notebook. Use a stable README download link and refresh the bundle whenever new notebook outputs are accepted.
- [ ] Create `pr42_executed_notebooks/` inside the local projects ZIP folder. Collect the accepted PR 42 executed notebooks there and package them as `pr42_executed_notebooks.zip`. Preserve this snapshot for the PR 42 description.
- [ ] Extract both archives and verify that the notebooks open with their saved tables and plots. Once the executed copies are accounted for, clear outputs and execution counts from tracked notebooks and confirm their code and Markdown cells are intact.

##### Step 6C — Final Validation and Reproducibility

- [ ] Run the targeted research, notebook, and packaging tests, followed by the full regression suite and applicable real-data tests.
- [ ] Run the applicable accepted-record `check` commands and confirm that frozen artifacts, feature contracts, and recorded identities match the accepted baseline.
- [ ] Execute the final Feature Research Notebook in a clean Windows kernel through the established PR 41 `_run_notebook` path. Confirm that its records reproduce and save the executed copy under `notebook_outputs/pr42/` with its run ID.
- [ ] Include the accepted final run in both bundles, verify their final contents, and confirm that the tracked notebooks remain output-free.

##### Step 6D — Publication and PR Closeout

- [ ] Publish both ZIPs as GitHub release assets. Link the current project-wide bundle from the root README and the fixed `pr42_executed_notebooks.zip` asset from the PR 42 description. Verify that each link downloads the intended archive.
- [ ] Document the closeout convention for future PRs that change notebook outputs: refresh the project-wide bundle after validation and publish a separate snapshot for that PR.
- [ ] Update the PR 42 research log, finish the roadmap heading cleanup, and decide whether to remove or archive completed Deferred Engineering items.
- [ ] Inspect the complete diff for unintended changes, generated files, and modifications to production feature code. Finalize the PR description with the findings, validation results, and PR-specific notebook download.

**Deliverable:** A reproducible Feature Research Notebook, a documented temporal-validity assessment and robustness protocol, a quantified explanation of NSD/ECI binning drift, and evidence-based feature-contract recommendations. The root README gives readers a short project overview and the current executed notebooks. The PR 42 description links directly to the accepted runs behind this PR.

---

#### PR 43 — Training Research Notebook and Stage 4 Policy Research

**Objective:** Establish the train-only training-research workflow and complete inexpensive Stage 4 methodology research, including the GLASS Linear Arbiter as a priority experimental candidate, on the frozen PR 39 training-side OOF evidence, before upstream optimization or regeneration.

Every Stage 4 candidate in this PR can be screened on the frozen training-side OOF table (Stage 1–3 predictions, router decisions, labels, and fold identities) without regenerating Stages 1–3. Small convex problems are solved with full-batch optimization; no mini-batch, online-learning, or reinforcement-learning infrastructure is introduced.

##### Step 43A — Training Experiment Framework

- [ ] Build the Training Research Notebook from existing research cells and modules; inventory tuning, calibration, cross-fitting, and training utilities before adding infrastructure.
- [ ] Enforce train-only selection with the PR 38 boundary extractors: the notebook loads no test split, test prediction, or test metric.
- [ ] Define stage-specific objectives, acceptance metrics, experiment budgets, and stopping criteria.
- [ ] Screen candidates inexpensively before full nested procedures, without weakening leakage safeguards.
- [ ] Separate raw model scoring, calibration, and decision-threshold selection.
- [ ] Record each arm's search spaces, trial budgets, and selection protocols.
- [ ] Measure hyperparameter-selection optimism. Stage 1 and Stage 3 currently tune on the same folds that produce their OOF predictions (calibration and thresholds are nested; hyperparameters are not). Define when nested selection is required for adoption.
- [ ] Build a nested Stage 4 validation harness on the frozen training-side OOF table: outer folds for evaluation, inner folds for selection, each fold fitted from scratch (no warm starts across folds), with the accepted GLASS Arbiter's procedure refitted per outer fold as the baseline.
- [ ] Track parameters, seeds, fold identities, elapsed time, and results, and support reproducible comparison against the accepted PR 39 configurations.
- [ ] Estimate the computational budget for the PR 45 regeneration and the PR 49 repeated-split experiments.

##### Step 43B — Stage 4 Objective and Abstention Policy

- [ ] Define the purpose and cost of abstention (for example, deferral to review) and a whole-system scoring policy for abstained rows; none is currently registered.
- [ ] Register coverage/risk criteria: selective risk at registered coverage levels, risk–coverage curves and AURC, or an explicit cost rule.
- [ ] Evaluate alternatives to retained F2 first. F2 has no true-negative term, so abstaining on a predicted negative never lowers retained F2; in all four accepted crossover cells the selected cut sits at the 50% training coverage floor.
- [ ] Compare Stage 4 scoring functions at matched coverage (AURC and selective risk on the registered coverage grid), separately from operating-point selection; never select weights or thresholds by retained F2.
- [ ] Apply one registered abstention/coverage policy to all Stage 4 candidates.
- [ ] Investigate excessive abstention and class-specific coverage (per-class retention, retained base rate) on training-side evidence, and reject candidates that improve retained-row metrics by abstaining on more correct predictions.

##### Step 43C — Stage 4 Voter Diagnostics

Measured overall and by routing state (Pass 1, Pass 2, router-deferred) on the frozen training-side OOF table, with voter decisions at the Stage 4 cuts and at matched flag budgets. High agreement is not treated as redundancy, and nearly uniform weights are not treated as inherently wrong.

- [ ] Vote-pattern table: counts and observed positive rates for each routing state × LR vote × EBM vote.
- [ ] LR–EBM agreement, error correlation, and double-fault rate (both voters wrong).
- [ ] Unique-correct predictions and correct lone dissents, by voter, class, and routing state.
- [ ] Router contributions: Pass 1 leaked positives flagged by LR or EBM, and the Pass 2 veto opportunity (false and true positives where both LR and EBM vote negative).
- [ ] Router-deferred rows: LR and EBM discrimination and calibration within that population.
- [ ] Segment calibration: intercept and slope of each voter's logit within each routing state.
- [ ] Abstention composition at the accepted cut, by class and routing state.

##### Step 43D — GLASS Arbiter Improvements

Context: the accepted weights (LR 0.343, EBM 0.355, router 0.302) are near-uniform by construction of the formula, and per-row influence is dominated by the confidence margins, whose negative side cannot exceed a voter's cut. The margins and the abstention objective are investigated before the weights.

- [ ] Analyze per-voter contributions and decision structure, including the structural confidence bound and its effect on Pass 2 and router-deferred rows.
- [ ] Evaluate log-odds confidence-margin normalization with the existing formula weights, isolating the effect of normalization.
- [ ] Evaluate sensitivity to the recall target (0.70), the weight blend (α = 0.5), and the abstention threshold.
- [ ] Evaluate nested joint selection of these parameters, where justified, against the current sequential fit.
- [ ] Compare in-sample rule evidence with cross-fitted router rule precision.
- [ ] Preserve the accepted configuration as the reproducible reference.

##### Step 43E — GLASS Linear Arbiter (priority experimental candidate)

- [ ] Define the candidate: regularized logistic regression with nonnegative coefficients and an intercept on log-odds voter-confidence signals: `logit(p) − logit(t)` for LR and EBM, and the signed `logit(r) − logit(b)` for the router where it routes (zero elsewhere). Its output is a probability estimate whose calibration must be assessed; any calibration correction uses training-only evidence.
- [ ] Use only the information channels Meta-XGBoost receives; give it no privileged information.
- [ ] Fit with full-batch convex optimization and a log-loss objective under the Step 43A nested harness, applying the registered abstention policy.
- [ ] Report coefficients with bootstrap intervals and fold-to-fold stability.
- [ ] Evaluate routing-state-specific coefficients, shrunk toward the global coefficients, only if Step 43C shows meaningful reliability differences between routing states.
- [ ] Compare it with the GLASS Arbiter at matched coverage; retain it only if its improvement exceeds fold-to-fold variability with no worse class-specific retention.
- [ ] Register it as a separate experimental arbiter, not a replacement; the GLASS Arbiter and Meta-XGBoost remain reference configurations.

##### Step 43F — Meta-XGBoost

- [ ] Compare its tuning objective (F2 at the 0.5 cut with balanced weights) with threshold-free alternatives.
- [ ] Compare confidence behavior in raw and calibrated probability space (`operating_space="calibrated"` is implemented).
- [ ] Evaluate its abstention policy under the registered criteria, without restricting its nonlinear learning.

##### Step 43G — Stage 4 Input Shift and Coverage Stability

- [ ] Meta-XGBoost: compare training-side coverage under its per-fold (fold-nested) thresholds and under the single global operating threshold used at test time.
- [ ] Investigate OOF-versus-refit input shift with a training-only simulated holdout: hold out a stratified part of the training split, fit the upstream stages on the remainder with the accepted hyperparameters, and compare OOF, refit, and fold-averaged inputs, confidence, and coverage. Start with the black-box upstream and pre-register the decision rule.
- [ ] Localize any shift by stage, including Meta-XGBoost's own meta-level OOF/refit asymmetry and the GLASS router evidence (fold-model rule rates and base rates on the training side, full-training values on the test side).
- [ ] Evaluate the retained Stage 4 candidates in the simulated holdout before PR 45; learned coefficients can amplify OOF-versus-refit shift.
- [ ] Measure abstention-cut and coverage stability across training-row resamples.

**Registration**

- [ ] Keep the registered PR 39 2×2 crossover unchanged.
- [ ] Register the GLASS Linear Arbiter comparisons (against the GLASS Arbiter and Meta-XGBoost, applied to both upstreams) separately, before their results are examined.

**Acceptance**

- [ ] Predefine the criteria for selecting the final GLASS Stage 4 approach: performance, calibration, coverage, interpretability, and computational efficiency, favoring the simpler interpretable model when performance is meaningfully comparable.
- [ ] Record the registered Stage 4 objectives, policies, candidates, and the provisional GLASS Stage 4 choice with their training-side evidence.
- [ ] Identify the upstream properties Stage 4 depends on (for example, probability quality on router-deferred rows) to direct PR 44.
- [ ] Regenerate no artifacts.

**Deliverable:** The Training Research Notebook, a registered Stage 4 abstention policy, training-side voter diagnostics, an evaluated GLASS Linear Arbiter candidate, a diagnosis of Stage 4 input shift, and predefined Stage 4 selection criteria for PR 45.

---

#### PR 44 — Stage 1–3 Optimization and Router Research

**Objective:** Systematically evaluate training objectives and optimization strategies for the upstream stages of both cascades (Phase 2 Optimization Principle), prioritizing ranking, probability quality, and whole-system contribution, without intermediate artifact regeneration.

Stages 1–3 are trained independently: no upstream stage reads another's outputs during training, and only Stage 4 consumes them. Their optimization can proceed in any order; one regeneration follows in PR 45.

##### Step 44A — Stage 1

**GLASS — Calibrated Logistic Regression**

- [ ] Evaluate regularization and penalty alternatives beyond the current search (100 trials, ROC-AUC, balanced class weights).
- [ ] Compare unweighted training with balanced weights followed by sigmoid calibration.
- [ ] Reevaluate calibration only when supported by training-side evidence.
- [ ] Preserve coefficient interpretability and the accepted feature contract.

**Black-box — Calibrated MLP (lower priority)**

Stage 1 has three inputs, so architecture and optimization changes have limited expected headroom; each experiment must justify its computational cost.

- [ ] Evaluate architecture width and depth, regularization, and optimization settings.
- [ ] Investigate batch size, learning rate, early stopping, and validation strategies.
- [ ] Compare positive oversampling with class-weighted or alternative objectives, including their effects on calibration.
- [ ] Examine training stability, initialization variance, and convergence.

**Shared evaluation**

- [ ] Compare ROC-AUC, PR-AUC, log-loss, Brier score, calibration, and computational efficiency.
- [ ] Treat Stage 1 decision thresholds as standalone reporting only: Stage 4 sets its own cuts, so threshold strategies are not Stage 1 optimization targets.
- [ ] Evaluate the incremental value of each change against the PR 39 baseline.

##### Step 44B — Stage 2 Routers

**GLASS Router**

- [ ] Preserve the two-pass routing structure and union-aware ILP selection.
- [ ] Investigate candidate-rule generation and selection efficiency.
- [ ] Evaluate redundancy, complementarity, and marginal coverage.
- [ ] Consider alternative selector objectives only when supported by measured deficiencies.
- [ ] Test the feature-bin candidates identified in PR 42.
- [ ] Preserve symbolic rule interpretability and auditable routing decisions.

**Random Forest Router**

- [ ] Start from the PR 35 objective-specialization results (the baseline configuration was retained for both passes); revisit only under a changed contract or new evidence.
- [ ] Evaluate hyperparameters, class weighting, and feature sampling where justified.
- [ ] Optimize thresholds under the same shared operating obligations.
- [ ] Compare routing coverage, constraint feasibility, and complementary predictions.

**Shared obligations**

- [ ] Preserve the Pass 1 NPV/leakage and Pass 2 precision/support requirements.
- [ ] Preserve the lexicographic coverage objective unless a separately justified methodology revision is approved.
- [ ] Maintain fold-level cross-fitting and nested eligibility safeguards.
- [ ] Preserve G13 leakage auditing and applicable F31 diagnostics.
- [ ] Evaluate routing efficiency, stability, abstentions, and unique-correct contributions.
- [ ] Register a new protocol version before any Stage 2 comparison against revised Stage 1/3 models (F10: the PR 38 protocol pins the historical LR/EBM flag fingerprints).

##### Step 44C — Stage 3

- [ ] Evaluate each Stage 3 model on the router-deferred rows it serves (`Stage3Artifact.evaluate_served`, training-side) before tuning.
- [ ] Compare the current tuning objective (F2 at the 0.5 cut with balanced weights) with threshold-free objectives (log-loss, PR-AUC) and with unweighted training. Before isotonic calibration, both accepted Stage 3 models have training OOF ECE of about 0.26–0.27.

**GLASS — Explainable Boosting Machine**

- [ ] Evaluate learning rate, boosting rounds, bagging, and regularization.
- [ ] Compare interaction budgets: the three fixed pairs, none, and capped automatic detection.
- [ ] Assess binning and smoothing parameters.
- [ ] Evaluate monotonic constraints only where supported by defensible hypotheses.
- [ ] Quantify the interpretability cost of additional terms, preserving transparent feature contributions and explanation fidelity.

**Black-box — XGBoost**

- [ ] Investigate tree depth, learning rate, and early stopping in place of a tuned estimator count.
- [ ] Evaluate regularization, subsampling, and column sampling.
- [ ] Compare objectives and class-imbalance strategies.
- [ ] Examine probability calibration and threshold selection separately.
- [ ] Evaluate training efficiency and the stability of selected configurations.

##### Step 44D — Optional Advanced Strategies

- [ ] Knowledge distillation and cascade compression: optional, separately identified experiments using leakage-safe (cross-fitted) teacher predictions. Report student fidelity, genuine-label performance, calibration, and interpretability separately, and report cross-family distillation as its own arm.
- [ ] Evaluate alternative proper scoring objectives where justified.
- [ ] Do not prioritize focal loss, reinforcement learning, or contextual bandits without new evidence: calibration matters to Stage 4, and the dataset is a static, fully labelled log without action-dependent outcomes.
- [ ] Promote a strategy only when inexpensive screening justifies further work.

**Acceptance**

- [ ] Select accepted Stage 1–3 configurations using predefined criteria.
- [ ] Preserve the PR 39 configurations for comparison.
- [ ] Version candidate training contracts.
- [ ] Defer artifact regeneration to PR 45.

**Deliverable:** Evaluated, versioned Stage 1–3 candidates for both cascades, ready for a single upstream regeneration.

---

#### PR 45 — Final Upstream Regeneration, Stage 4 Training, and Comparison Registration

**Objective:** Regenerate the accepted upstream candidates once, train the Stage 4 arbiters under the policies registered in PR 43, and register the comparison protocol before optimized results are presented.

**Upstream integration**

- [ ] Lock the accepted upstream configurations and feature contracts.
- [ ] Regenerate the affected upstream artifacts once, in dependency order, with correct fold isolation.
- [ ] Reuse unchanged artifacts only where provenance and compatibility checks establish that reuse is valid.
- [ ] Verify stacking dependencies, row alignment, fingerprints, and feature provenance.

**Stage 4 training**

- [ ] Fit the GLASS Arbiter, Meta-XGBoost, and the GLASS Linear Arbiter (if retained in PR 43) under the PR 43 objectives, using training-side evidence only for candidate selection, thresholds, and coverage cutoffs.
- [ ] Select the final GLASS Stage 4 approach by the criteria predefined in PR 43, favoring the simpler interpretable model when performance is meaningfully comparable.
- [ ] Evaluate retained-only and whole-system performance together, with class-specific abstention, retained base rates, error rates, and calibration where applicable.
- [ ] Reject degenerate solutions that improve retained-row metrics by abstaining on excessive numbers of otherwise correct predictions.
- [ ] Recompute the registered 2×2 crossover (four cells) on the regenerated upstream, and run any separately registered GLASS Linear Arbiter comparison apart from it.

**Comparison registration**

- [ ] Before PR 46 runs the optimized notebooks, register the comparison questions, primary metrics, coverage grids, uncertainty methods, and analysis plan.

**Validation and freeze**

- [ ] Perform targeted and integration regression validation.
- [ ] Preserve the PR 39 reference artifacts.
- [ ] Freeze the optimized candidate artifacts under new run identifiers with distinct provenance.

**Deliverable:** Optimized, reproducible GLASS and black-box cascade candidates, trained Stage 4 arbiters, and a registered comparison protocol.

---

#### PR 46 — Cascade Notebook Presentation and Comparison-Ready Freeze

**Objective:** Finalize the GLASS and black-box cascade notebooks as clear, reproducible presentations of the optimized architectures.

- [ ] Consolidate verbose notebook outputs into concise stage summaries.
- [ ] Build or reuse shared plotting and diagnostic utilities.
- [ ] Present architecture diagrams, feature and voter contributions, and performance results.
- [ ] Show routing as Pass 1 / Pass 2 / abstain rather than a binary outcome, including in the cascade trace.
- [ ] Include coverage, calibration, and stage-level diagnostics.
- [ ] Distinguish the frozen PR 39 baseline from the optimized Phase 2 candidates.
- [ ] Relocate research and comparison cells to the Training Research and Comparative Analysis notebooks, and remove duplicate training logic and unnecessary computation.
- [ ] Reproduce Phase 1 from the Phase 1 tag: its reproduction tooling depends on the PR 39 notebook cell headers.
- [ ] Standardize comparison-ready artifact manifests and loaders.
- [ ] Run both notebooks in clean kernels and confirm deterministic reproduction and preserved contracts.
- [ ] Freeze and tag the comparison-ready optimized cascade implementations.

**Deliverable:** Two finalized cascade notebooks and a versioned, reproducible comparison baseline.

### Phase 3 — Comparative Research and Robustness

**Objective:** Compare the finalized cascades and standalone models under the registered protocol, with robust statistical analysis and explicit claim boundaries.

#### PR 47 — Comparison Framework and Cascade-to-Cascade Evaluation

**Comparison infrastructure**

- [ ] Establish the Comparative Analysis Notebook.
- [ ] Load frozen artifacts without retraining.
- [ ] Enforce feature, split, model, and provenance compatibility, and reject stale or mismatched artifacts.
- [ ] Add targeted tests for comparison loaders and metric calculations.
- [ ] Implement the comparison questions, metrics, and result schemas registered in PR 45.

**Cascade comparison**

- [ ] GLASS versus black-box end-to-end performance.
- [ ] Stage-level performance and complementarity.
- [ ] Registered 2×2 Stage 4 arbiter crossover (four cells).
- [ ] Separately registered GLASS Linear Arbiter comparisons, reported apart from the accepted 2×2 crossover.
- [ ] Matched positive-flag-budget comparisons.
- [ ] Matched abstention-coverage comparisons.
- [ ] Coverage-prescription comparison:
  - registered coverage grid (30–80%)
  - training OOF cutoffs applied to test
  - retained and whole-system metrics per coverage
  - risk–coverage curves / AURC
- [ ] Stage 4 abstention drift, per-class exclusion, and retained base rate.
- [ ] Test-side compliance with training-side obligations (the Stage 4 coverage floor, Stage 2 Pass 1 leakage and Pass 2 precision).
- [ ] Prediction agreement, disagreement, and unique-correct analysis.
- [ ] Between-arm OOF score correlation per stage.
- [ ] Paired bootstrap confidence intervals.
- [ ] Calibration-quality comparisons on appropriate probability scales (the GLASS Arbiter's weighted score is not a calibrated probability).
- [ ] Documented differences in model capacity, interpretability, and optimization budgets.
- [ ] Interpretation of performance, complexity, and transparency tradeoffs.

**Deliverable:** Reproducible primary cascade-comparison results and visual evidence.

#### PR 48 — Standalone Model Baselines

**Glass-box candidates**

- [ ] Logistic Regression.
- [ ] Classical GAM.
- [ ] EBM.
- [ ] Shallow Decision Tree.
- [ ] Sparse Rule List or RuleFit-style model.

**Black-box candidates**

- [ ] XGBoost.
- [ ] Random Forest.
- [ ] MLP.
- [ ] RBF SVM.
- [ ] CatBoost or LightGBM.

**Methodology**

- [ ] Define the shared, audited standalone feature population explicitly, and fit all preprocessing within folds.
- [ ] Apply common split, fold, tuning-budget, calibration, and evaluation obligations as appropriate, and record each model's budget.
- [ ] Generate leakage-safe OOF and held-out predictions.
- [ ] Compare at matched coverage: each standalone model abstains on its own confidence under the registered policy, so the cascades' abstention does not confound the comparison.
- [ ] Compare the strongest standalone alternatives against the respective cascades.
- [ ] Evaluate whether cascade complexity delivers measurable additional predictive or decision-making value.
- [ ] Optionally include a compressed-cascade baseline from PR 44.
- [ ] Record the additional dependencies (GAM, rule-list or RuleFit, CatBoost or LightGBM), and use a kernel approximation if an RBF SVM is impractical at about 33,000 training rows.

**Deliverable:** Controlled standalone baselines and evidence of the cascades' incremental value.

#### PR 49 — Robustness, Ablations, and Scientific Validity

**Robustness**

- [ ] Five-fold versus ten-fold sensitivity on both arms.
- [ ] Repeated seeds and splits for model-family claims, refitting procedural feature decisions within each split.
- [ ] Whether GLASS/black-box differences persist across protocol changes.
- [ ] Stage-removal ablations.
- [ ] Decision-threshold sensitivity.
- [ ] Abstention and coverage sensitivity.
- [ ] Tuning-budget sensitivity.
- [ ] Router-contract and feature-binning sensitivity, including NSD/ECI drift using PR 42 evidence.
- [ ] Temporal robustness using the PR 42 protocol (train on earlier months, test on later), reported as robustness evidence rather than independent validation.

**Stacking methodology sensitivity (F39; includes the PR 36 deferral)**

- [ ] Assess indirect label dependence in the upstream OOF stacking features of the learned arbiters (Meta-XGBoost and, if retained, the GLASS Linear Arbiter).
- [ ] Evaluate the feasibility of fully nested upstream fitting within each meta-fold.
- [ ] Compare standard OOF stacking against fully nested stacking where computationally feasible.
- [ ] Measure effects on calibration, thresholds, abstention, and predictive performance.
- [ ] Document the methodological trade-offs and implications for the GLASS versus black-box comparison.

**Independent evaluation and claim boundaries**

- [ ] Record that no Tier 3 data exists for this dataset: every row has been used for training or in the previously exposed test split, and the other UCI Bank Marketing files come from the same campaigns.
- [ ] Identify any genuinely independent dataset or prospective evaluation, and freeze model selection and evaluation procedures before accessing new labels.
- [ ] Report independent findings separately from development and robustness results.
- [ ] Where independent evaluation is unavailable, explicitly limit generalization claims.

**Interpretability boundaries**

- [ ] Define which components the interpretability claims cover, including calibration maps, training-time guidance such as RF-guided rule search, and the GLASS Arbiter's weighted score.

**Deliverable:** Robustness evidence, quantified methodological sensitivity, and a transparent statement of scientific validity and claim boundaries.

### Phase 4 — Paper Readiness and Final Research Freeze

#### PR 50 — Paper Exports, Reproducibility, and Final Freeze

- [ ] Finalize the research questions and accepted comparison claims.
- [ ] Export publication-quality figures and tables.
- [ ] Consolidate model, cascade, calibration, and abstention results.
- [ ] Present interpretability/performance tradeoffs.
- [ ] Document statistical uncertainty and limitations.
- [ ] Distinguish development findings, robustness evidence, and independently validated findings.
- [ ] Verify experiment provenance, dependency versions, and artifact identities.
- [ ] Run final end-to-end reproducibility and regression validation.
- [ ] Freeze the research implementation and experiment artifacts.
- [ ] Tag the research release.
- [ ] Prepare paper-ready methods, results, and supplemental evidence.

**Deliverable:** Reproducible, publication-ready research artifacts and a finalized experimental record.

---

### Deferred Engineering and Optional Research

- [x] **Windows notebook-execution warning (deferred from PR 40; resolved in PR 41).** On Windows, nbclient ran on Python's default Proactor event loop, which lacks the reader methods zmq needs, so pyzmq added a tornado selector thread and emitted one `RuntimeWarning` per process. PR 41 Commit 4 added a Windows-only path to `upstream_regeneration/vehicle.py` (`_run_notebook`): on Windows with Python 3.11 or later and no running event loop, `client.async_execute()` runs under `asyncio.Runner` on its own `SelectorEventLoop`, which is closed and unset afterward. Other platforms, older Pythons, and calls from a running loop still use `client.execute()`. No global event-loop policy changes, the deprecated `WindowsSelectorEventLoopPolicy` is not used, and the warning is not suppressed. The change is a reviewed exception to the provenance-sensitive execution code. PR 41 Commit 6 validated it with the vehicle tests (`tests/upstream_regeneration/test_execution_path.py`) and a representative Windows kernel execution, confirming the warning's elimination, subprocess compatibility, exception propagation, and cleanup.
- [x] **PuLP 3.3.2 deprecations (resolved in PR 41).** The GLASS Router's ILP code constructed `LpVariable` objects directly and solved with `PULP_CBC_CMD`, both deprecated in PuLP 3.3. PR 41 Commit 6 replaced them with `prob.add_variable(...)` (`ilp_builder.py`, `novelty_analyzer.py`) and with `COIN_CMD` on the CBC executable bundled with PuLP 3.x (`PULP_CBC_CMD.pulp_cbc_path`; CBC 2.10.3 with the recorded PuLP 3.3.2), which is what `PULP_CBC_CMD` ran. The CBC version, solver options, ILP objectives, constraints, and solver-status semantics are unchanged, and a regression test pins the solver path and options (`tests/stage2/test_ilp_rule_selector_union.py`). Both PuLP warning filters were removed from `pytest.ini`; `requirements.txt` keeps `pulp>=2.8,<4`.
- [ ] **PuLP 4.x migration (future compatibility).** PuLP 4.0 removes the bundled CBC binary and `PULP_CBC_CMD`, from which `ilp_builder.py` reads its pinned CBC path (`_BUNDLED_CBC`). CBC then comes from the `pulp[cbc]` extra or a `cbc` executable on `PATH`, possibly a different build from CBC 2.10.3, and PuLP 4.0 changes other modelling and solve APIs (see its 3.x-to-4.0 migration guide). Migrating requires a dependency change, a new solver pin, and solver-parity validation against the accepted Stage 2 selections (objectives, termination statuses, time-limit and feasible-but-not-optimal outcomes, and the CBC log parsing in `parse_cbc_log`), as a separately versioned Stage 2 change. Until then, keep `pulp>=2.8,<4`.
- [ ] **Joblib/Loky shutdown traceback (low priority).** On Windows (Python 3.14.3, Joblib 1.6.0 with vendored Loky 3.6.0, ipykernel 6.31.0), the Joblib/Loky resource tracker can print a `KeyError` traceback for a `joblib_memmapping_folder_` entry during kernel shutdown, after notebook execution has completed. Local reproduction supports a shutdown-order conflict: ipykernel terminates the tracker, and Joblib's exit-time cleanup restarts it with an empty registry. No effect on notebook results, accepted artifacts, or execution records has been observed, and no repository change is required (`research_logs/pr41/pr41_loky_resource_tracker.md`). [Joblib PR #1829](https://github.com/joblib/joblib/pull/1829) concerns temporary-folder registration and cleanup and may address it; before upgrading Joblib, repeat the isolated reproduction with the candidate release.
- [ ] **Lightweight GLASS development utility.** A small command-driven tool (`run script`, `run notebook`, `run cell`, `apply test`, `scan artifacts`) that reuses existing Python, pytest, notebook, and artifact utilities, protects accepted artifacts, never retrains unexpectedly, and leaves Git operations to Git. Evaluate when its workflow benefit justifies implementation.
- [ ] Investigate advanced training strategies that survive preliminary scientific and computational screening.
- [ ] Avoid new framework dependencies or infrastructure without demonstrated need.