# PR 38 research report: `c3_20261112_153333`

Consolidated from this run's accepted v1 packages by `python -m stage2_crossfit report --run-id c3_20261112_153333`. Every value is copied from the packages as written; no model, artifact or label is read. Numbers below are rounded for display; the CSV and JSON files hold the exact values.

## Status

- Structural acceptance V1-V8: accepted `true`, official `true` (`structural_acceptance.json`, SHA-256 `91612167dd258278`)
- S1 sensitivity: accepted `true`, G6 Pass 1 invariance `true`: S1 accepted as a descriptive sensitivity: G6 passed

## Primary outcomes (GLASS - RF, pooled; 95% paired bootstrap; D-C1)

| outcome | metric | GLASS | RF | GLASS - RF | 95% interval | statement |
|---|---|---:|---:|---:|---|---|
| O1 | pass1_rate | 0.3538 | 0.4201 | -0.06625 | [-0.07202, -0.06055] | GLASS lower than RF |
| O2 | pass2_rate | 0.2022 | 0.251 | -0.04874 | [-0.05232, -0.04519] | GLASS lower than RF |
| O3a | pass1_npv | 0.9534 | 0.9594 | -0.005972 | [-0.009689, -0.002293] | GLASS lower than RF |
| O3b | pass1_leakage_share | 0.1463 | 0.1514 | -0.005119 | [-0.01778, 0.00728] | not resolved at this sample |
| O3c | pass2_precision | 0.3075 | 0.298 | 0.009539 | [0.003066, 0.01602] | GLASS higher than RF |
| O4a | unique_tp | 86 | 60 | 26 | [5.975, 47] | GLASS higher than RF |
| O4b | unique_fp | 1140 | 939 | 201 | [124, 273] | GLASS higher than RF |
| O4c | pass1_fp_suppression | 0.1084 | 0.002896 | 0.1055 | [0.09712, 0.114] | GLASS higher than RF |
| O4d | pass1_tp_cost | 0.04239 | 0.002826 | 0.03956 | [0.03128, 0.04747] | GLASS higher than RF |
| descriptive_total | coverage | 0.5561 | 0.671 | -0.115 | [-0.1214, -0.1085] | descriptive total (no statement rule) |
| descriptive_total | abstain_rate | 0.4439 | 0.329 | 0.115 | [0.1085, 0.1214] | descriptive total (no statement rule) |

## Tables

Each row is one row of a v1 package table: `package` and `table` name it. The identifying columns come first, then each source table's own columns side by side; a cell under another table's column is empty, not a value. Filter on `table` to see one source table whole. Renamed columns: `metric_table.routing` is `arm`; `fold_dispersion.routing` is `arm`; `pre_post_metrics.routing` is `correction`; `class_conditional.routing` is `arm`.
Rows from `complementarity_v1` are the corrected primary (GLASS Pass 2 fitted on the OOF remainder); diagnostics rows from `s1_sensitivity_v1` are the S1 configuration (GLASS Pass 2 fitted on every fold row; RF unchanged).

| file | contents | rows | columns | v1 tables |
|---|---|---:|---:|---|
| [01_registered_outcomes.csv](01_registered_outcomes.csv) | Registered outcomes with 95% paired bootstrap intervals, pooled and per fold: GLASS vs RF (D-C1 statements) and pre vs post correction per arm | 198 | 29 | primary_outcomes, primary_outcomes_by_fold, pre_post_uncertainty, pre_post_by_fold |
| [02_router_metrics.csv](02_router_metrics.csv) | Router metric tables (numerator, denominator, value, status) per fold and pooled, corrected and pre/post correction, and their fold dispersion | 532 | 16 | metric_table, pre_post_metrics, fold_dispersion |
| [03_routing_states.csv](03_routing_states.csv) | Three-state routing: counts, shares and class composition, and state transitions (pre to post correction; S1 oof_remainder to full) | 270 | 26 | diagnostics/states, diagnostics/states (S1), class_conditional, pre_post_transitions, routing_transitions (S1) |
| [04_unique_contribution.csv](04_unique_contribution.csv) | Pass 2 discovery against the frozen LR/EBM references, cross-router overlap, the common eligible population and chance baselines | 288 | 50 | diagnostics/pass2_breakdown, diagnostics/pass2_breakdown (S1), diagnostics/pass2_cross_router, diagnostics/pass2_cross_router (S1), diagnostics/common_population, diagnostics/common_population (S1), diagnostics/chance_baselines, diagnostics/chance_baselines (S1) |
| [05_matched_budget.csv](05_matched_budget.csv) | The frozen LR/EBM reference flags, label-free rate-matched selections and each router's Pass 1 set against the matched LR set | 60 | 34 | diagnostics/reference_flags, diagnostics/matched_selection, diagnostics/matched_selection (S1), pass1_matched_lr |
| [06_agreement_diagnostics.csv](06_agreement_diagnostics.csv) | Router-reference contingency, cross-router crosstab, LR/EBM disagreement and observed-versus-expected | 972 | 46 | diagnostics/contingency, diagnostics/contingency (S1), diagnostics/crosstab, diagnostics/crosstab (S1), diagnostics/model_disagreement, diagnostics/model_disagreement (S1), diagnostics/observed_expected, diagnostics/observed_expected (S1) |
| [07_score_strata.csv](07_score_strata.csv) | Routing by score stratum, and stratum contrasts | 4008 | 26 | diagnostics/score_conditional, diagnostics/score_conditional (S1), diagnostics/stratum_contrasts, diagnostics/stratum_contrasts (S1) |
| [08_diagnostic_ratios.csv](08_diagnostic_ratios.csv) | Every Step 2A-2D count and ratio (numerator, denominator, value, status) | 12303 | 15 | diagnostics/ratios, diagnostics/ratios (S1) |
| [09_contract_compliance.csv](09_contract_compliance.csv) | Fit-time obligations and observed outer-OOF shortfalls (kept apart), F31 R2 strips, R3 optimism and M6 inner-OOF Pass 1 | 258 | 53 | shortfalls, fit_time_obligations, f31_r2_strips, f31_r3_optimism, m6_inner_oof_pass1 |
| [10_solver_and_validation.csv](10_solver_and_validation.csv) | Fold ledger, solver outcomes, three-state accounting, trace and alignment checks, routing-policy audit and signal availability | 236 | 60 | fold_ledger, fold_ledger_summary, solver_outcomes, accounting, trace_checks, alignment, diagnostics/policy_audit, diagnostics/policy_audit (S1), diagnostics/signal_availability, diagnostics/signal_availability (S1) |
| [11_s1_sensitivity.csv](11_s1_sensitivity.csv) | S1 (GLASS pass2_population full vs oof_remainder): configuration, Pass 2 fitting populations, fold feasibility and G6 Pass 1 invariance | 72 | 35 | configuration (S1), pass2_fitting_population (S1), fold_feasibility (S1), g6_pass1_invariance (S1) |

| file | contents |
|---|---|
| [run_summary.json](run_summary.json) | run identity, acceptance, provenance records |
| [complementarity_summary.json](complementarity_summary.json) | primary outcomes, pre/post intervals, layers, limitations |
| [s1_summary.json](s1_summary.json) | S1 acceptance, single factor, pooled GLASS changes |
| [source_manifest_complementarity_v1.json](source_manifest_complementarity_v1.json) | byte-exact copy of the historical `complementarity_v1/MANIFEST.json` |
| [source_manifest_s1_sensitivity_v1.json](source_manifest_s1_sensitivity_v1.json) | byte-exact copy of the historical `s1_sensitivity_v1/MANIFEST.json` |
| [MANIFEST.json](MANIFEST.json) | historical v1 table checksums and dispositions; this report's file checksums |

## v1 package tables not repeated here

Rebuilt byte for byte whenever the report is made or checked (`stage2_crossfit.reporting`).

| v1 table | disposition | from |
|---|---|---|
| complementarity_v1/pre_post_states | derived | outer join of the frozen pr37_stage2_pre_correction_v1 states with complementarity_v1/diagnostics/states (03_routing_states.csv); pre - post |
| complementarity_v1/pre_post_ratios | derived | outer join of the frozen pr37_stage2_pre_correction_v1 ratios with complementarity_v1/diagnostics/ratios (08_diagnostic_ratios.csv); pre - post |
| s1_sensitivity_v1/states_comparison | derived | outer join of diagnostics/states of both packages (03_routing_states.csv); full - oof_remainder |
| s1_sensitivity_v1/ratios_comparison | derived | outer join of diagnostics/ratios of both packages (08_diagnostic_ratios.csv); full - oof_remainder |
| s1_sensitivity_v1/baseline_comparison | derived | outer join of the GLASS rows of the frozen pr37_stage2_pre_correction_v1 ratios and of both packages' diagnostics/ratios (08_diagnostic_ratios.csv) |
| s1_sensitivity_v1/diagnostics/reference_flags | duplicate | complementarity_v1/diagnostics/reference_flags (05_matched_budget.csv) |
| s1_sensitivity_v1/s1_acceptance_checks | restated | s1_acceptance.json (every gate's checks) |

## Limitations (as disclosed by the packages)

- Evidence tier 2 (pr38_corrected_outer_oof); the pre-correction baseline is tier 1. The Bank Marketing held-out test split was exposed in PRs 33-36: no test label, test prediction or test-derived threshold is read here.
- Pre/post differences are a COMBINED correction (GLASS selector, shared contract, F17 populations, nested rankers); no single component is attributed (protocol section 8). Only S1 (Step 3C) isolates the GLASS population policy.
- Known asymmetries: pre GLASS fitted Pass 2 on 'full'; pre RF used match_glass_oof constraints and global_t1; the common eligible population changes when either Pass 1 changes, so matched-budget populations differ before and after; the historical GLASS confidence is in-sample rule precision.
- Pre/post differences are pre - post (the registered procedure's order); cross-arm differences are GLASS - RF.
- Intervals: stratified paired bootstrap (fold x label), conditional on the fitted routers and the frozen upstream flags; evaluation-row sampling only. No multiplicity adjustment is pre-registered. No interval establishes contract compliance.
- The five outer folds are not independent replications: per-fold values, minimum, maximum and range only.
- Matched-budget results are descriptive, not a deployable threshold; each arm is matched at its own budget, so cross-arm budget equality is not achieved by this family.
- RF Router confidence is a heuristic blend (F31 Disposition A), not a calibrated probability. R2 strips use the fold router's inner-OOF band (the provisional basis of the F31 R2 clarification); R3 optimism gaps are reported, never used to tune.
- Routing-policy claims keep their evidence status: GLASS Pass 1 precedence and Pass 2 eligibility are code_level_only; the corrected RF decision-time claims are not_verifiable because no outer Pass 1 score or GLASS-arm RF result is persisted.
- Statements follow D-C1 only: higher / lower when the 95% paired interval excludes 0, otherwise not resolved at this sample. No weighted score and no overall winner.
- Outer-OOF observations never revise the contract, re-estimate bands or relabel a violation (protocol section 11); fit status and observation are separate columns.
- S1 is descriptive (F17 section 6): it quantifies what the Pass 2 population policy does to GLASS. The primary policy (oof_remainder) was fixed before S1 ran; S1 never chooses, replaces or alters the primary and is reported whatever its direction.
- Single factor: only pass2_population differs (configuration.csv). Both traces use the accepted primary's RF Router, so RF rows are unchanged; no RF counterpart is run.
- Within-S1 differences are full - oof_remainder, paired by fold. No significance claim is made and no interval is computed: no test is pre-specified for S1.
- Differences against pr37_stage2_pre_correction_v1 are confounded by the selector correction and the contract change; only the within-S1 difference is attributed to the population policy.
- G6 passed (s1_acceptance.json): Pass 1 is identical in both runs, so differences arise in Pass 2 and abstention only.
- Precision-estimation asymmetry (F17 L2): each Pass 2 rule's precision is in-sample on the population it was selected on (all fold rows under full, the remainder under oof_remainder).
- Evidence tier 2 (training-side outer OOF); no test label is read. Matched-budget (2C) results are descriptive, not deployable thresholds.

## Verification

`MANIFEST.json` keeps two kinds of checksum apart: the historical SHA-256 of every v1 package table (`historical_v1_sources`) and the SHA-256 of every file here (`outputs`). From `prototype/bank_pipeline`:

    python -m stage2_crossfit report --run-id c3_20261112_153333 --check

reads no v1 package: it checks every file here against the manifest, the run's accepted records against the ones listed, reproduces every v1 table from these tables (and the frozen baseline) against its historical SHA-256, and rebuilds the report byte for byte. The v1 packages' own `--check` needs their tables and runs only while they exist.
