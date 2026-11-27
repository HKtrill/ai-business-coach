# Roadmap

## Phase 2 — Research Organization, Training Research, and Optimized Cascades

## PR 44 — Stage 1–3 Optimization and Router Research

**Objective:** Select upstream feature and training procedures for both cascades. Keep stage fitting independent. PR 45 owns production regeneration and Stage 4 adoption.

**References:** [PR 42 candidate contracts R1–R6](../research_logs/pr42/feature_contract_recommendations.md), [PR 43 decisions and cost evidence](../research_logs/pr43/README.md), [PR 38 evaluation protocol](methodology/pr38_evaluation_protocol.md). Historical Step 44A/B/C references correspond to Stage 1 (Commit 3), routers (Commits 5–6) and Stage 3 (Commit 4).

**Execution principles:** Optimize on training rows only. Preserve PR 39 pins, manifests and the four-cell crossover. Register decisions before inspecting new results. Keep random-split development findings, the Stage 1 chronological candidate check and PR 49 cascade robustness distinct. Use targeted tests and research acceptance checks after each step. Reuse compatible records and predictions by digest. Reserve comprehensive regression and publication for closure. Justify substantial additions against existing implementations.

### Commit 1 — Roadmap Planning and Deferred Engineering Updates

**Objective:** Set the research sequence and reconcile prior decisions.

- [x] Review the frozen references, notebook responsibilities, training paths and deferred work. Correct the served-row evaluation path and downstream Stage 4 shortlist.
- [x] Schedule independent Stage 1 and Stage 3 work before expensive router comparisons. Reserve new experimental registrations and implementation for subsequent commits.

### Commit 2 — Training Boundary, Baselines and Optimization Registration

**Objective:** Establish which questions frozen evidence answers and what a candidate must demonstrate.

#### Step 2A — Verify training inputs and reproduce references

- [ ] Extend the Training Research Notebook after its PR 43 sections. Reuse `training_side`, `training_inputs`, `frozen_references` and existing metric primitives. Verify the PR 39 six-role upstream manifest against the PR 43 frozen-input receipt, ordered 32,950 rows and 3,712 positives, ten Stage 1 folds, five shared Stage 2/3 folds, route states, producing-model base rates and probability spaces.
- [ ] Establish both arms' all-row and served-row references before optimization. Join training OOF predictions to each arm's frozen deferral mask by row identity. Reproduce PR 43's GLASS reference on 14,628 deferrals (Stage 1/3 average precision 0.168/0.214). The artifact's `evaluate_served` method indexes test rows. Exercise the labelled training path and reject misaligned or test rows.
- [ ] Identify which summaries can use committed records and which comparisons need verified local numerical archives. Give each new execution a write-once ID under `research_logs/pr44/` using existing research-record utilities. Bind seeds, environment, code, folds, feature and parent digests. Record missing evidence at the affected comparison.

#### Step 2B — Register decisions and cost before screening

- [ ] For each contrast, lock comparator, primary metric and direction, practical margin, safeguards, paired uncertainty method, multiplicity treatment and simplicity tie-break. Use probability/ranking metrics for Stage 1/3 and the registered feasibility/coverage hierarchy for routers. Define improvement and equivalence using paired intervals. Record a failed safeguard as rejection and inadequate support, precision or budget as inconclusive. A comparable simpler procedure may advance under the registered tie-break.
- [ ] Specify finite spaces, trial and seed caps, inner/outer folds, expected fit counts (including calibration, stopping, repeated ILPs and feature engineering), elapsed-time ceilings and conditional expansion rules. Start with frozen summaries, then matched one-factor screens. Reuse only fitting-local blocks with matching identities. Record cost per arm and stop an expansion when it cannot change a decision within budget.
- [ ] For Stage 1/3 shortlists, repeat all data-dependent screening and selection inside outer training partitions, or freeze the candidate list prospectively from historical evidence. Fit preprocessing, stopping, interaction selection and calibration inside the applicable fitting boundary. Lock each fit before outer scoring. Report paired outer effects, selected parameters and signed inner-selected versus outer gaps on the same metric, including sampling and training-size limitations. Previously explored training rows limit inference.
- [ ] Register descriptive whole-system diagnostics under PR 43 policy v1. Measure signed voter margins, unique correct votes, effective recall, retention by class, coverage and review-cost loss on matched locked actions. Use original formula/raw Meta controls and GLASS-only normalized margins when score semantics permit. Verify registration replay, partition isolation and mismatched-input failures.

### Commit 3 — Stage 1 Macro Encodings and Numerical Stability

**Objective:** Test whether the three-input LR and MLP retain useful ranking and transferable probabilities under deployable fitting rules.

#### Step 3A — Isolate encoding and early-stopping effects

- [ ] Reuse `encoding_audit` and `temporal_validity` for the accepted encoder versus `past_daily_lag0` (`stage1-features` 3.0.0-candidate) at fixed settings on both arms. Specify contact-day input, earlier-label availability, cold start and euribor fallback. Report paired random-fold and chronological results, pooled and within the five PR 42 windows with prevalence and support. Outcome timing is uncertain. PR 49 owns lag sensitivity and chronological cascades.
- [ ] Compare MLP encoder-isolated early stopping with the accepted split at matched features and seed. Draw the stopping subset before fitting the encoder/scaler or resampling. Verify earlier encodings are invariant to unavailable labels and stopping labels cannot influence preprocessing.

#### Step 3B — Optimize shortlisted procedures

- [ ] Reuse the LR tuning/penalty paths, MLP tuner and shared Stage 1 OOF calibration. Compare compatible LR penalty/solver and regularization settings, balanced and unweighted training. Trace seven SAG/SAGA warnings to fits and folds. Check scaling, iterations against cap, tolerance, coefficient/probability stability and elapsed time. Characterize unresolved configurations and exclude unstable fits.
- [ ] Give the MLP a small matched-seed comparison of oversampling, weighting and unweighted training. Expand width/depth, learning rate, batch size or regularization only on a registered signal. Measure seed variance and stopping epochs. Compare raw and calibrated log-loss, Brier, ECE, PR-/ROC-AUC and runtime. Keep standalone cuts descriptive.
- [ ] Apply Step 2B nesting and the chronological gate. Record separate R1 feature and R2 procedure decisions, fold isolation and convergence checks. Save aligned OOF inputs and signed margins for Commit 6.

### Commit 4 — Stage 3 Shared Features, EBM Interactions and Training Objectives

**Objective:** Separate changes to the shared feature block from changes to either learner. Compare models on the same frozen deferrals within each arm.

#### Step 4A — Test features and interactions at fixed training settings

- [ ] Compare an equivalently refitted 13-feature baseline with `jed_alone` for EBM and XGBoost (R5). Keep the historical prediction streams as separate references. Reuse `stage3_simplification`, the Stage 3 block cache and shared runner. Verify feature order, G2/G3 inventory and held-out invariance.
- [ ] Start the candidate EBM with the two surviving Step 5A pairs. Compare the three fixed baseline pairs on the original block, zero, one or two eligible pairs, a fixed campaign pair and capped fold-fitted selection using `interaction_selection` (R6). Record candidate eligibility per block. Apply the Step 5A net-corrections screen on fixed deferrals. The fixed candidate scored 486 versus 389 after three-pair reselection against the historical 456.86 cutoff. Recompute the registered screen against the paired refit when its reference changes.
- [ ] Report terms, supported shapes, fold stability, additive explanation fidelity and net corrections alongside discrimination and calibration. Require one shared feature list for both arms. Keep `drop_first` secondary, `drop_second` closed and `past_density` with PR 49.

#### Step 4B — Evaluate objectives and tune survivors

- [ ] Isolate the historical balanced F2-at-0.5 search, training weights and selection metric. Compare unweighted fitting, log-loss and average-precision selection under matched budgets. Evaluate raw and cross-fitted calibrated probabilities before choosing operating cuts.
- [ ] Bound EBM rate, rounds, bagging, regularization and smoothing/binning proposals. Compare XGBoost early stopping on fitting-local validation rows with tuned tree count, then depth, shrinkage, row/column sampling and regularization where justified. Reuse `shared/stage3/{runner,tuning,calibration}.py` and family fitters. Nest surviving procedures, verify stability, fit costs and served-row results, and freeze candidate OOF identities for F10.

### Commit 5 — Procedural Router Contracts and Independent Rule Evidence

**Objective:** Spend router fits on changes that can affect feasibility, coverage or reliability.

#### Step 5A — Screen bin and rule-space revisions

- [ ] Reuse `binning_candidates`, `stage2_binning`, `stage2_crossfit` and the two-pass selectors. Fit NSD `gap_placed` with both registered and fold-local reference inputs (R3). Record reference scope, cut, support, migration and fold-0 lift. Treat registered constants that saw outer rows as conditional evidence. Apply the shared ordered 29-bin contract to both routers.
- [ ] Separately evaluate `eci_warm` as a required filter on GLASS Pass 2 partner rules (R4). Measure candidate count, redundancy, overlap and marginal coverage on the actual remainder. Test a combination only if individual contrasts justify its cost. Extend G13 to the cut fit within each cross-fit fold and verify nested eligibility.
- [ ] Retain PR 35 RF settings as control. Reopen pass-specific tuning, weights, feature sampling or thresholds only for a changed contract or measured deficiency under identical obligations. Preserve F31's interpretation of RF band scores.

#### Step 5B — Validate surviving routers

- [ ] Preserve the lexicographic objective and PR 38 obligations. Report per-pass fit and outer OOF NPV/leakage, precision, support, feasibility, coverage and abstentions. Time candidate generation and solving separately. Record solver termination, gap and ordered rules. Apply PR 38 M3 repetition for eligible feasible incumbents, retaining failed attempts and authenticated recovery evidence.
- [ ] Report matches versus first-issued decisions and positives, producing-model base rates, precision, overlap, order and marginal correct decisions by fold. Recheck the two flagged Pass 2 rules and five unsupported full-fit rule transfers from PR 43 where their conditions recur. Require rule-specific independent support for transfer. Record unavailable ablations explicitly.
- [ ] Verify G13, fit statuses, RF diagnostics and ordered rule reproducibility. Reserve comparisons involving new Stage 1/3 flags for the F10 registration in Commit 6.

### Commit 6 — Cross-Stage Comparisons, Reproducibility and PR 45 Handoff

**Objective:** Determine whether independently trained revisions complement the other voters and whether their confidence scales transfer. Consolidate versioned candidate decisions, complete comprehensive validation, and prepare PR 45 for a single production regeneration.

#### Step 6A — F10 Comparisons and Shortlist Transfer Checks

- [ ] Before examining any new Stage 2 comparative result against Stage 1/3 flags outside the PR 38 pins (including PR 39), register a distinct F10 protocol with exact row, fold, flag, cut and candidate identities. Bind the Commit 2 design to the actual streams. Check stale fingerprints are rejected. Earlier served-row analyses condition on fixed masks and do not rank routers against revised voters.
- [ ] Swap one shortlisted stream at a time on identical training rows with companion streams fixed. Report route transitions, unique correct votes, signed margins, paired policy loss and retention. Report Stage 3 results on common frozen deferrals first, then on each router's actual deferrals with denominators and class mix. Identify comparisons conditional on historical producer fits.
- [ ] Evaluate the combined shortlist using compatible prediction streams where available. Distinguish individual stage improvements from their combined whole-system contribution without requiring production regeneration.
- [ ] Reuse PR 43's holdout evidence for historical full-refit versus fold-average shifts. For registered material changes in confidence, routing or coverage, refit only the necessary shortlisted producer on the fitting partition and score the same held-out rows at locked cuts. Verify compatibility of reused companions. Report gross decision/retention flips alongside net changes. The exposed simulated split supports development diagnostics. Require a costed unresolved decision before repeating a whole upstream.
- [ ] Keep advanced work conditional. A proper-score or distillation screen needs a concrete failure to address and spare registered budget. Use fitting-local cross-fitted teacher predictions and score genuine labels, calibration, fidelity, size and interpretability separately. Identify cross-family distillation as its own arm. Defer compression, focal loss, RL and contextual bandits absent a supported question and suitable data.

#### Step 6B — Final Decisions, Validation, Executed Notebooks and PR Closure

- [ ] Link accepted records under `research_logs/pr44/`. Summarize accepted, rejected, comparable and inconclusive procedures, including unchanged references. Record feature/training versions, fitting scopes, seeds, costs, limitations, and PR 45 rebuild/reuse dependencies and time estimates. Carry open numerical and rule-support questions into the adoption decision.
- [ ] Update the notebook map and research log. Use paired effects/cost tables, calibration or margin plots where scale changed, supported EBM shapes and router coverage/support tables. Reuse plotting and research utilities. Review implementation footprint and targeted checks.
- [ ] Reconcile Deferred Engineering with the completed PR 44 work. Resolve remaining issues where feasible within scope, verify fixes, and document outstanding blockers and their disposition before Phase 2 closure. Preserve completed deferred items as a resolution record.
- [ ] Run default, slow and comprehensive real-data suites, the complete frozen-baseline checks and accepted-record verification. Reproduce selected results within registered tolerances. Execute Training Research in a clean Windows kernel using the established execution path, with explicit verified readback for expensive historical results.
- [ ] Package the accepted executed notebook, refresh the flat project archive and prepare a run-identified PR 44 snapshot. Check extraction, manifest hashes, embedded results and output-free tracked notebooks. Publish the notebook assets at closeout and verify download links.

**Deliverable:** Versioned upstream candidates and a costed, evidence-linked PR 45 recommendation for each stage and arm.

---

## PR 45 — Final Upstream Regeneration, Stage 4 Training, and Comparison Registration

**Objective:** Regenerate the accepted upstream candidates once, train the Stage 4 arbiters under the policies registered in PR 43, and register the comparison protocol before optimized results are presented.

### Upstream Integration

- [ ] Lock the accepted upstream configurations and feature contracts.
- [ ] Adopt only PR 44-accepted candidates, each as a separately versioned contract that meets its leakage, G13 and regeneration requirements (PR 42 Step 6A, section 5).
- [ ] With the first adopted Stage 2 or Stage 3 revision, move the Stage 2/3 feature DAG to one production-owned implementation, keeping compatibility aliases for the PR 39 Stage 3 artifacts (PR 42 Step 6A, section 6).
- [ ] Regenerate the affected upstream artifacts once, in dependency order, with correct fold isolation.
- [ ] Reuse unchanged artifacts only where provenance and compatibility checks establish that reuse is valid.
- [ ] Verify stacking dependencies, row alignment, fingerprints, and feature provenance.

### Stage 4 Training

- [ ] Refit the original GLASS formula and raw Meta-XGBoost on both upstreams as common-policy controls. On GLASS inputs also fit normalized log-odds margins with the same voter cuts and weights. Distinguish the searched Meta reference from fixed-parameter deployment checks.
- [ ] Before examining regenerated results, register the inferential role of normalized versus original and normalized versus raw Meta on GLASS. Apply policy v1 loss, uncertainty, class-retention, support, calibration and partial-AURC safeguards at locked cuts. Check attainable coverage, full-refit transfer and independent first-issued rule support. Prefer simplicity under demonstrated comparability.
- [ ] Evaluate retained-only and whole-system performance together, with class-specific abstention, retained base rates, error rates, and calibration where applicable.
- [ ] Reject degenerate solutions that improve retained-row metrics by abstaining on excessive numbers of otherwise correct predictions.
- [ ] Recompute the registered 2×2 crossover (four cells) on the regenerated upstream, and report the separately registered normalized-margin comparisons outside those four cells.

### Comparison Registration

- [ ] Before PR 46 runs the optimized notebooks, register the comparison questions, primary metrics, coverage grids, uncertainty methods, and analysis plan.

### Validation and Freeze

- [ ] Perform targeted and integration regression validation.
- [ ] Preserve the PR 39 reference artifacts.
- [ ] Freeze the optimized candidate artifacts under new run identifiers with distinct provenance.

**Deliverable:** Optimized, reproducible GLASS and black-box cascade candidates, trained Stage 4 arbiters, and a registered comparison protocol.

---

## PR 46 — Cascade Notebook Presentation and Comparison-Ready Freeze

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

---

## Phase 3 — Comparative Research and Robustness

**Objective:** Compare the finalized cascades and standalone models under the registered protocol, with robust statistical analysis and explicit claim boundaries.

## PR 47 — Comparison Framework and Cascade-to-Cascade Evaluation

### Comparison Infrastructure

- [ ] Establish the Comparative Analysis Notebook.
- [ ] Load frozen artifacts without retraining.
- [ ] Enforce feature, split, model, and provenance compatibility, and reject stale or mismatched artifacts.
- [ ] Add targeted tests for comparison loaders and metric calculations.
- [ ] Implement the comparison questions, metrics, and result schemas registered in PR 45.

### Cascade Comparison

- [ ] GLASS versus black-box end-to-end performance.
- [ ] Stage-level performance and complementarity.
- [ ] Registered 2×2 Stage 4 arbiter crossover (four cells).
- [ ] Report the prospectively registered GLASS normalized-margin contrasts separately from the four-cell crossover, carrying forward their PR 45 disposition.
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

---

## PR 48 — Standalone Model Baselines

### Glass-Box Candidates

- [ ] Logistic Regression.
- [ ] Classical GAM.
- [ ] EBM.
- [ ] Shallow Decision Tree.
- [ ] Sparse Rule List or RuleFit-style model.

### Black-Box Candidates

- [ ] XGBoost.
- [ ] Random Forest.
- [ ] MLP.
- [ ] RBF SVM.
- [ ] CatBoost or LightGBM.

### Methodology

- [ ] Define the shared, audited standalone feature population explicitly, and fit all preprocessing within folds.
- [ ] Apply common split, fold, tuning-budget, calibration, and evaluation obligations as appropriate, and record each model's budget.
- [ ] Generate leakage-safe OOF and held-out predictions.
- [ ] Compare at matched coverage: each standalone model abstains on its own confidence under the registered policy, so the cascades' abstention does not confound the comparison.
- [ ] Compare the strongest standalone alternatives against the respective cascades.
- [ ] Evaluate whether cascade complexity delivers measurable additional predictive or decision-making value.
- [ ] Optionally include a compressed-cascade baseline from PR 44.
- [ ] Record the additional dependencies (GAM, rule-list or RuleFit, CatBoost or LightGBM), and use a kernel approximation if an RBF SVM is impractical at about 33,000 training rows.

**Deliverable:** Controlled standalone baselines and evidence of the cascades' incremental value.

---

## PR 49 — Robustness, Ablations, and Scientific Validity

### Robustness

- [ ] Five-fold versus ten-fold sensitivity on both arms.
- [ ] Repeated seeds and splits for model-family claims, refitting procedural feature decisions within each split.
- [ ] Whether GLASS/black-box differences persist across protocol changes.
- [ ] Stage-removal ablations.
- [ ] Decision-threshold sensitivity.
- [ ] Abstention and coverage sensitivity.
- [ ] Tuning-budget sensitivity.
- [ ] Router-contract and feature-binning sensitivity, including NSD/ECI drift using PR 42 evidence.
- [ ] Temporal robustness using the PR 42 protocol (train on earlier months, test on later), reported as robustness evidence rather than independent validation.
- [ ] Build the past-only Stage 2–3 constructions and run them in the chronological cascade (PR 42 Step 6A, sections 5.2–5.3):
  - NSD, ECI and the day × month rate;
  - a past-only fallback for the past-only density.

  Refit bin cuts and interaction pairs inside each split, with their reference inputs stated.

### Stacking Methodology Sensitivity (F39; includes the PR 36 deferral)

- [ ] Assess indirect label dependence in the upstream OOF stacking features of the retained learned arbiter (Meta-XGBoost).
- [ ] Evaluate the feasibility of fully nested upstream fitting within each meta-fold.
- [ ] Compare standard OOF stacking against fully nested stacking where computationally feasible.
- [ ] Measure effects on calibration, thresholds, abstention, and predictive performance.
- [ ] Document the methodological trade-offs and implications for the GLASS versus black-box comparison.

### Independent Evaluation and Claim Boundaries

- [ ] Record that no Tier 3 data exists for this dataset: every row has been used for training or in the previously exposed test split, and the other UCI Bank Marketing files come from the same campaigns.
- [ ] Identify any genuinely independent dataset or prospective evaluation, and freeze model selection and evaluation procedures before accessing new labels.
- [ ] Report independent findings separately from development and robustness results.
- [ ] Where independent evaluation is unavailable, explicitly limit generalization claims.

### Interpretability Boundaries

- [ ] Define which components the interpretability claims cover, including calibration maps, training-time guidance such as RF-guided rule search, and the GLASS Arbiter's weighted score.

**Deliverable:** Robustness evidence, quantified methodological sensitivity, and a transparent statement of scientific validity and claim boundaries.

---

## Phase 4 — Paper Readiness and Final Research Freeze

## PR 50 — Paper Exports, Reproducibility, and Final Freeze

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

## Deferred Engineering and Optional Research

- [x] **Windows notebook-execution warning (deferred from PR 40; resolved in PR 41).** On Windows, nbclient ran on Python's default Proactor event loop, which lacks the reader methods zmq needs, so pyzmq added a tornado selector thread and emitted one `RuntimeWarning` per process. PR 41 Commit 4 added a Windows-only path to `upstream_regeneration/vehicle.py` (`_run_notebook`): on Windows with Python 3.11 or later and no running event loop, `client.async_execute()` runs under `asyncio.Runner` on its own `SelectorEventLoop`, which is closed and unset afterward. Other platforms, older Pythons, and calls from a running loop still use `client.execute()`. No global event-loop policy changes, the deprecated `WindowsSelectorEventLoopPolicy` is not used, and the warning is not suppressed. The change is a reviewed exception to the provenance-sensitive execution code. PR 41 Commit 6 validated it with the vehicle tests (`tests/upstream_regeneration/test_execution_path.py`) and a representative Windows kernel execution, confirming the warning's elimination, subprocess compatibility, exception propagation, and cleanup.
- [x] **PuLP 3.3.2 deprecations (resolved in PR 41).** The GLASS Router's ILP code constructed `LpVariable` objects directly and solved with `PULP_CBC_CMD`, both deprecated in PuLP 3.3. PR 41 Commit 6 replaced them with `prob.add_variable(...)` (`ilp_builder.py`, `novelty_analyzer.py`) and with `COIN_CMD` on the CBC executable bundled with PuLP 3.x (`PULP_CBC_CMD.pulp_cbc_path`; CBC 2.10.3 with the recorded PuLP 3.3.2), which is what `PULP_CBC_CMD` ran. The CBC version, solver options, ILP objectives, constraints, and solver-status semantics are unchanged, and a regression test pins the solver path and options (`tests/stage2/test_ilp_rule_selector_union.py`). Both PuLP warning filters were removed from `pytest.ini`; `requirements.txt` keeps `pulp>=2.8,<4`.
- [ ] **PuLP 4.x migration (future compatibility).** PuLP 4.0 removes the bundled CBC binary and `PULP_CBC_CMD`, from which `ilp_builder.py` reads its pinned CBC path (`_BUNDLED_CBC`). CBC then comes from the `pulp[cbc]` extra or a `cbc` executable on `PATH`, possibly a different build from CBC 2.10.3, and PuLP 4.0 changes other modelling and solve APIs (see its 3.x-to-4.0 migration guide). Migrating requires a dependency change, a new solver pin, and solver-parity validation against the accepted Stage 2 selections (objectives, termination statuses, time-limit and feasible-but-not-optimal outcomes, and the CBC log parsing in `parse_cbc_log`), as a separately versioned Stage 2 change. Until then, keep `pulp>=2.8,<4`.
- [ ] **Joblib/Loky shutdown traceback (low priority).** On Windows (Python 3.14.3, Joblib 1.6.0 with vendored Loky 3.6.0, ipykernel 6.31.0), the Joblib/Loky resource tracker can print a `KeyError` traceback for a `joblib_memmapping_folder_` entry during kernel shutdown, after notebook execution has completed. Local reproduction supports a shutdown-order conflict: ipykernel terminates the tracker, and Joblib's exit-time cleanup restarts it with an empty registry. No effect on notebook results, accepted artifacts, or execution records has been observed, and no repository change is required (`research_logs/pr41/pr41_loky_resource_tracker.md`). [Joblib PR #1829](https://github.com/joblib/joblib/pull/1829) concerns temporary-folder registration and cleanup and may address it; before upgrading Joblib, repeat the isolated reproduction with the candidate release.
- [ ] **Lightweight GLASS development utility.** A small command-driven tool (`run script`, `run notebook`, `run cell`, `apply test`, `scan artifacts`) that reuses existing Python, pytest, notebook, and artifact utilities, protects accepted artifacts, never retrains unexpectedly, and leaves Git operations to Git. Evaluate when its workflow benefit justifies implementation.
- [ ] PR 44 Commit 6 may screen an advanced training strategy only under its evidence and cost gate.
- [ ] Avoid new framework dependencies or infrastructure without demonstrated need.
- [ ] **Scikit-learn SAG/SAGA convergence warnings.** Seven convergence warnings occurred during PR 42 temporal-validity real-data validation. PR 44 Step 3B owns reproduction, numerical diagnosis and the shortlist decision before PR 45 regeneration. Do not suppress the warnings.
- [ ] **Optuna notebook widget rendering (low priority).** Investigate interactive progress-bar rendering warnings in saved executed notebooks. At PR 44 packaging, check saved output. Use a local presentation-only fix if a warning obscures evidence and verify identical trial results.
- [ ] **Cross-platform compatibility.** Evaluate Windows, Linux, and macOS portability, including LF/CRLF, paths and notebook execution. Fix reproducible blockers in touched PR 44 paths with focused checks. Defer platform-wide CI and dependency changes to separate work.
- [ ] **Repository consolidation.** After PR 46 and before announcing Phase 2 completion, measure and reduce duplicated code, tests and documentation while preserving the historical scientific records and reproduction paths.
