"""
feature_research.feature_inventory
==================================
PR 42, Commit 2: the inventory of the accepted features, feature contracts and
stage feature populations of both cascades, traced to their definitions and
callers on this branch.

Declared here (``NODES``, ``POPULATIONS``, ``DUPLICATE_PAIRS``) and verified,
never assumed:

``check_registries``        every declared population equals the production
                            registry it names, and the bins follow
                            ``BINNING_STRATEGY``.
``check_call_sites``        the notebook cells that build or read each
                            population call what the inventory says (AST of the
                            two cascade notebooks; nothing executed).
``check_fit_protocols``     the fold and fit settings the inventory states are
                            the production defaults.
``check_dag_columns``       every declared node is produced by each
                            implementation that claims it.
``probe_target_dependence`` the declared target dependence of every node and
                            bin is what the code does: each implementation is
                            fitted twice on the same training rows, with the
                            true and with permuted labels, and the outputs of a
                            fixed row sample are compared.
``duplicate_callers``       the current importers of the two Stage 2/3
                            feature-engineering implementations (AST).
``duplicate_parity``        how far the two implementations agree on training
                            rows (in-sample and on held-out folds).

Test-time application is stated from the production code; no test row is read
here or by any function in this module. Every function that takes data takes
training rows only.

``render_markdown()`` produces the generated tables of
``research_logs/pr42/feature_inventory.md`` (``python -m
feature_research.feature_inventory`` prints them).
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "NONE", "FITTED", "INHERITED", "STAGE_OUTPUT", "RAW_VARIABLES", "LEAKY_DROPPED",
    "Node", "NODES", "Population", "POPULATIONS", "STAGE1_FEATURES", "STAGE2_SOURCES",
    "STAGE2_BINS", "STAGE3_FEATURES", "EBM_INTERACTIONS", "STAGE4_META_XGB",
    "STAGE4_GLASS_ARBITER", "bin_nodes", "feature_table", "population_table", "node_table",
    "check_registries", "check_call_sites", "check_fit_protocols", "check_dag_columns",
    "build_training_frames", "training_build_summary", "require", "require_probe",
    "check_nsd_reference", "nsd_fit_sizes", "compare_values",
    "callers_summary",
    "stage2_dag_frame", "stage3_dag_frame", "probe_target_dependence", "DUPLICATE_PAIRS",
    "duplicate_callers", "duplicate_parity", "render_markdown", "PROBE_SEED", "PROBE_ROWS",
]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: Target dependence of a feature.
NONE = "none"                    # a deterministic function of the row's own raw values
FITTED = "fitted"                # a statistic of the fit's training labels
INHERITED = "inherited"          # a deterministic function of a FITTED input
STAGE_OUTPUT = "stage_output"    # an upstream stage's prediction (Stage 4 inputs)

#: Label permutation probe settings (recorded in the run's seeds).
PROBE_SEED = 42
PROBE_ROWS = 2000

#: The preprocessed columns every stage starts from (BankPreprocessor(drop_leaky=True)).
RAW_VARIABLES: Tuple[str, ...] = (
    "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "day_of_week", "campaign", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
)
#: Dropped by BankPreprocessor(drop_leaky=True) before any stage (and again,
#: defensively, by ``shared.stage3.feature_engineering.LEAKY_FEATURES``).
LEAKY_DROPPED: Tuple[str, ...] = ("duration", "pdays", "poutcome")

# Implementation keys
S1 = "shared.stage1.feature_engineering"         # LRFeatureEngineer (Stage 1, both arms)
S2 = "shared.stage2.feature_engineering"         # RFFeatureEngineer (Stage 2, both arms)
S3 = "feature_research.feature_engineering"      # via shared.stage3 EBMFeaturePipeline (Stage 3)
IMPLEMENTATIONS = {
    S1: "shared.stage1.feature_engineering.LRFeatureEngineer",
    S2: "shared.stage2.feature_engineering.RFFeatureEngineer (self-contained copies)",
    S3: ("shared.stage3.feature_engineering.EBMFeaturePipeline -> "
         "feature_research.feature_engineering.{crisis,integrals,derivatives,temporal,prior,overlap}"),
}


# ======================================================================
# Nodes: every engineered column, intermediates included
# ======================================================================
@dataclass(frozen=True)
class Node:
    name: str
    module: str                   # DAG step (crisis, integrals, ..., stage1, binning)
    definition: str
    inputs: Tuple[str, ...]       # direct inputs: raw variables or other nodes
    target: str                   # NONE / FITTED / INHERITED / STAGE_OUTPUT
    implementations: Tuple[str, ...]
    apply: str = "computed directly from the row"
    constants: str = ""
    fit_rows: str = ""            # how the fit's own rows are encoded (FITTED only)

    def record(self) -> Dict[str, Any]:
        return {"name": self.name, "module": self.module, "definition": self.definition,
                "inputs": list(self.inputs), "target": self.target,
                "implementations": list(self.implementations), "apply": self.apply,
                "constants": self.constants, "fit_rows": self.fit_rows}


_DAG = (S2, S3)
_CRISIS = ("crisis thresholds and weights (euribor3m < 1.5: 3, emp.var.rate < -1.0: 2, "
           "nr.employed < 5100: 1; score >= 3) from the Cell 10A importance analysis")
#: How each label-dependent statistic encodes the fit's OWN rows (the final fit's training rows,
#: or a fold's training rows); rows outside the fit follow the node's ``apply`` rule.
_BIN_RATE_ROWS = ("every fit row is counted in the rate of its own bin, so its own label enters "
                  "its value")
_CELL_ROWS = ("every fit row is counted in its own day x month cell, so its own label enters "
              "its value (directly with weight 1 / (n_cell + 100), and through the fit's global "
              "rate)")
_CURVATURE_ROWS = ("every fit row is counted in its bin's rate; that rate enters the curvature of "
                   "its bin and of the neighbouring bins (two np.gradient passes), so its own "
                   "label enters its value")
_NSD_ROWS = ("only the reference rows' labels contribute. Every production fit has more than "
             "10,000 rows (check_nsd_reference), so the reference set is 5,000 fit rows sampled "
             "with seed 42: a sampled row's own label enters its density (kernel weight 1 at "
             "distance 0); every other fit row is transformed against the reference set, as a "
             "held-out or test row is. Stage 2's label-excluded inner folds also drop the "
             "reference rows that fall in inner fold j")

_NODE_LIST = (
    # ---- Stage 1 (shared.stage1.feature_engineering) -------------------------------------
    Node("cellular_downturn", "stage1",
         "(contact == 0) and crisis score >= 3 (historically cellular_crisis)",
         ("contact", "euribor3m", "emp.var.rate", "nr.employed"), NONE, (S1,), constants=_CRISIS),
    Node("subscribe_rate_by_euribor_bin", "stage1",
         "training subscription rate of the row's euribor3m quantile bin (20 qcut bins, "
         "duplicates dropped; equal-width fallback); historically euribor3m_local_rate",
         ("euribor3m",), FITTED, (S1,),
         apply=("pd.cut with the fit's IntervalIndex categories; outside them -> median of "
                "the fitted bin rates"), fit_rows=_BIN_RATE_ROWS),
    Node("subscribe_rate_by_dow_month", "stage1",
         "(n*p + 100*p_global) / (n + 100) of the row's day_of_week x month cell "
         "(historically dow_month_encoded)",
         ("day_of_week", "month"), FITTED, (S1,),
         apply="lookup of the fit's cell (string key); unseen cell -> the fit's global rate",
         fit_rows=_CELL_ROWS),
    # ---- crisis ------------------------------------------------------------------------
    Node("economic_crisis_score", "crisis",
         "3*[euribor3m < 1.5] + 2*[emp.var.rate < -1.0] + 1*[nr.employed < 5100]",
         ("euribor3m", "emp.var.rate", "nr.employed"), NONE, _DAG, constants=_CRISIS),
    Node("cellular_crisis", "crisis", "(contact == 0) and economic_crisis_score >= 3",
         ("contact", "economic_crisis_score"), NONE, _DAG, constants=_CRISIS),
    Node("high_conversion_month", "crisis", "month in {3, 9, 10, 12}", ("month",), NONE, _DAG,
         constants="months chosen in the Cell 10A/10F conversion analysis"),
    Node("cellular_contact", "crisis", "contact == 0", ("contact",), NONE, _DAG),
    Node("default_clean", "crisis", "default == 0", ("default",), NONE, _DAG),
    # ---- integrals ---------------------------------------------------------------------
    Node("economic_stress_integral", "integrals",
         "(3*max(0, (1.5 - euribor3m)/1.5) + 2*max(0, -emp.var.rate/3) "
         "+ 1*max(0, (5200 - nr.employed)/100)) / 6",
         ("euribor3m", "emp.var.rate", "nr.employed"), NONE, _DAG,
         constants="cliffs 1.5 / 5200 and weights 3:2:1 from Cell 10B"),
    Node("neighborhood_subscription_density", "integrals",
         "Gaussian-kernel (bandwidth 1.0) weighted mean of reference rows' labels in "
         "standardized (euribor3m, nr.employed, emp.var.rate) space",
         ("euribor3m", "nr.employed", "emp.var.rate"), FITTED, _DAG,
         apply=("the fit's scaler and reference rows (all fit rows, or 5,000 sampled with "
                "seed 42 when the fit has more than 10,000 rows)"),
         fit_rows=_NSD_ROWS),
    # ---- derivatives -------------------------------------------------------------------
    Node("euribor3m_sigmoid", "derivatives", "expit(-(euribor3m - 1.5) / 0.5)",
         ("euribor3m",), NONE, _DAG),
    Node("euribor3m_sigmoid_slope", "derivatives",
         "euribor3m_sigmoid * (1 - euribor3m_sigmoid) / 0.25", ("euribor3m_sigmoid",), NONE, _DAG),
    Node("emp_var_rate_sigmoid", "derivatives", "expit(-(emp.var.rate + 1.0) / 0.5)",
         ("emp.var.rate",), NONE, _DAG),
    Node("emp_var_rate_sigmoid_slope", "derivatives",
         "emp_var_rate_sigmoid * (1 - emp_var_rate_sigmoid) / 0.25",
         ("emp_var_rate_sigmoid",), NONE, _DAG),
    Node("euribor_decay", "derivatives", "exp(-0.50 * max(0, euribor3m - 0.60))",
         ("euribor3m",), NONE, _DAG),
    Node("nr_employed_decay", "derivatives", "exp(-0.02 * max(0, nr.employed - 4964))",
         ("nr.employed",), NONE, _DAG),
    Node("emp_var_decay", "derivatives", "exp(-0.80 * max(0, emp.var.rate + 2.0))",
         ("emp.var.rate",), NONE, _DAG),
    Node("euribor3m_local_rate", "derivatives",
         "training subscription rate of the row's euribor3m quantile bin (20 bins)",
         ("euribor3m",), FITTED, _DAG,
         apply=("the fit's bins (Stage 2 copy: IntervalIndex categories; research module: "
                "exact numeric edges); outside them -> median of the bin rates"),
         fit_rows=_BIN_RATE_ROWS),
    Node("euribor3m_abs_curvature", "derivatives",
         "|second difference| of the training subscription rates over 20 euribor3m quantile bins",
         ("euribor3m",), FITTED, _DAG, apply="the fit's bin of the row; outside the edges -> 0.0",
         fit_rows=_CURVATURE_ROWS),
    Node("nr_employed_abs_curvature", "derivatives",
         "|second difference| of the training subscription rates over 20 nr.employed quantile bins",
         ("nr.employed",), FITTED, _DAG, apply="the fit's bin of the row; outside the edges -> 0.0",
         fit_rows=_CURVATURE_ROWS),
    Node("emp_var_rate_abs_curvature", "derivatives",
         "|second difference| of the training subscription rates over 20 emp.var.rate quantile bins",
         ("emp.var.rate",), FITTED, _DAG, apply="the fit's bin of the row; outside the edges -> 0.0",
         fit_rows=_CURVATURE_ROWS),
    Node("economic_curvature_intensity", "derivatives", "mean of the three *_abs_curvature columns",
         ("euribor3m_abs_curvature", "nr_employed_abs_curvature", "emp_var_rate_abs_curvature"),
         INHERITED, _DAG, apply="the mean of the row's three curvatures from the fit's bins",
         fit_rows="as its three *_abs_curvature inputs"),
    Node("joint_economic_decay", "derivatives", "euribor_decay * emp_var_decay * nr_employed_decay",
         ("euribor_decay", "emp_var_decay", "nr_employed_decay"), NONE, _DAG),
    Node("decay_x_density", "derivatives",
         "joint_economic_decay * neighborhood_subscription_density",
         ("joint_economic_decay", "neighborhood_subscription_density"), INHERITED, _DAG,
         apply="the row's joint_economic_decay times its density from the fit's reference rows",
         fit_rows="as neighborhood_subscription_density, its only label-dependent input"),
    # ---- temporal ----------------------------------------------------------------------
    Node("dow_month_encoded", "temporal",
         "(n*p + 100*p_global) / (n + 100) of the row's day_of_week x month cell",
         ("day_of_week", "month"), FITTED, _DAG,
         apply="lookup of the fit's cell; unseen cell -> the fit's global rate",
         fit_rows=_CELL_ROWS),
    # ---- prior -------------------------------------------------------------------------
    Node("has_prior_contact", "prior", "previous >= 1", ("previous",), NONE, _DAG),
    Node("prior_x_stress", "prior", "has_prior_contact * economic_stress_integral",
         ("has_prior_contact", "economic_stress_integral"), NONE, _DAG),
    # ---- overlap -----------------------------------------------------------------------
    Node("low_stress_zone", "overlap", "economic_stress_integral < 0.2",
         ("economic_stress_integral",), NONE, _DAG, constants="0.2 from Cell 10F"),
    Node("cpi_high_cellular", "overlap", "(cons.price.idx >= 93.5) and (contact == 0)",
         ("cons.price.idx", "contact"), NONE, _DAG, constants="93.5 from the Cell 10F MI analysis"),
    Node("behavioral_favorability", "overlap",
         "(3*cellular_contact + 3*high_conversion_month + 2*has_prior_contact + "
         "1*default_clean + 1*[age >= 50]) / 10",
         ("cellular_contact", "high_conversion_month", "has_prior_contact", "default_clean", "age"),
         NONE, _DAG, constants="weights 3:3:2:1:1 and age 50 from Cell 10F"),
    Node("overlap_default_clean", "overlap", "low_stress_zone * default_clean",
         ("low_stress_zone", "default_clean"), NONE, _DAG),
    Node("overlap_behavioral_score", "overlap", "low_stress_zone * behavioral_favorability",
         ("low_stress_zone", "behavioral_favorability"), NONE, _DAG),
)
NODES: Dict[str, Node] = {n.name: n for n in _NODE_LIST}


def bin_nodes() -> Dict[str, Node]:
    """The 29 Stage 2 bins as nodes, generated from ``BINNING_STRATEGY`` (source of truth)."""
    from shared.stage2.binning import BINNING_STRATEGY
    out: Dict[str, Node] = {}
    for family, group in BINNING_STRATEGY.items():
        source = group["source"]
        src = NODES.get(source)
        target = NONE if src is None else (INHERITED if src.target in (FITTED, INHERITED) else NONE)
        if "passthrough" in group:
            name = group["passthrough"]
            out[name] = Node(name, "binning", f"{source} (passthrough, int8)", (source,), target,
                             (S2,), apply="the source value, cast to int8")
            continue
        for name, lo, hi in group["bins"]:
            rule = (f"{source} <= {hi}" if lo is None else f"{source} > {lo}" if hi is None
                    else f"{lo} < {source} <= {hi}")
            out[name] = Node(name, "binning", f"{family}: {rule}", (source,), target, (S2,),
                             apply=("locked thresholds on the source value as the fit "
                                    "transforms the row"),
                             fit_rows=f"as {source}" if target != NONE else "",
                             constants="BINNING_STRATEGY thresholds, fixed constants chosen on all "
                                       "training labels (Cell 13C lift analysis); never refit")
    return out


def all_nodes() -> Dict[str, Node]:
    return {**NODES, **bin_nodes()}


def source_variables(name: str, nodes: Optional[Mapping[str, Node]] = None) -> Tuple[str, ...]:
    """The raw variables a node depends on, transitively, in ``RAW_VARIABLES`` order."""
    nodes = all_nodes() if nodes is None else nodes
    seen, todo = set(), [name]
    while todo:
        n = todo.pop()
        if n in RAW_VARIABLES:
            seen.add(n)
        elif n in nodes:
            todo.extend(nodes[n].inputs)
        else:
            raise KeyError(f"{n!r} is neither a raw variable nor a declared node")
    return tuple(v for v in RAW_VARIABLES if v in seen)


def fitted_ancestors(name: str, nodes: Optional[Mapping[str, Node]] = None) -> Tuple[str, ...]:
    """The FITTED nodes a node depends on (itself included), transitively."""
    nodes = all_nodes() if nodes is None else nodes
    out, todo = [], [name]
    while todo:
        n = todo.pop()
        if n in nodes:
            if nodes[n].target == FITTED and n not in out:
                out.append(n)
            todo.extend(nodes[n].inputs)
    return tuple(sorted(out))


# ======================================================================
# Populations: what each stage of each arm consumes
# ======================================================================
STAGE1_FEATURES: Tuple[str, ...] = (
    "cellular_downturn", "subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month")
STAGE2_SOURCES: Tuple[str, ...] = (
    "campaign", "neighborhood_subscription_density", "cons.conf.idx",
    "economic_curvature_intensity", "joint_economic_decay", "dow_month_encoded",
    "behavioral_favorability", "cpi_high_cellular")
STAGE2_BINS: Tuple[str, ...] = (
    "nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot",
    "jed_cold", "jed_transition", "jed_warm", "jed_hot",
    "cci_low", "cci_dead_mid", "cci_hot1", "cci_valley", "cci_hot2",
    "eci_cold", "eci_mid", "eci_warm", "eci_hot",
    "dow_cold", "dow_low", "dow_mid", "dow_hot",
    "behav_cold", "behav_baseline", "behav_warm", "behav_hot",
    "campaign_fresh", "campaign_moderate", "campaign_heavy",
    "cpi_cellular")
STAGE3_FEATURES: Tuple[str, ...] = (
    "decay_x_density", "euribor3m_sigmoid_slope", "economic_curvature_intensity",
    "cons.conf.idx", "age", "dow_month_encoded", "default", "prior_x_stress", "campaign",
    "overlap_default_clean", "cpi_high_cellular", "behavioral_favorability",
    "emp_var_rate_sigmoid_slope")
EBM_INTERACTIONS: Tuple[Tuple[str, str], ...] = (
    ("campaign", "emp_var_rate_sigmoid_slope"),
    ("euribor3m_sigmoid_slope", "economic_curvature_intensity"),
    ("prior_x_stress", "decay_x_density"))
STAGE4_META_XGB: Tuple[str, ...] = (
    "stage1_proba", "stage2_proba", "stage2_pass1", "stage2_pass2", "stage2_abstain",
    "stage3_proba")
#: The GLASS Arbiter's voters (glass_pipeline.glass_arbiter.loader.load_stage_outputs).
STAGE4_GLASS_ARBITER: Tuple[str, ...] = ("lr", "glass", "ebm")

_STAGE4_CHANNELS = {
    "stage1_proba": "Stage 1 P(y=1): MLP StageOutput (calibrated)",
    "stage2_proba": "RF Router P(y=1); NaN where the router abstains",
    "stage2_pass1": "RF Router routed Pass 1 (NOT_SUBSCRIBE)",
    "stage2_pass2": "RF Router routed Pass 2 (SUBSCRIBE)",
    "stage2_abstain": "RF Router abstained",
    "stage3_proba": "XGBoost Stage3Artifact P(y=1), matched calibration space",
    "lr": "GLASS Stage 1 calibrated LR probability (StageOutput)",
    "glass": "GLASS Router route (Pass 1 / Pass 2 / uncertain) and probability",
    "ebm": "GLASS Stage 3 EBM probability (Stage3Artifact)",
}

_FIT_STAGE1 = {
    "final_fit": ("engineer_features(X_train, X_test, y_train) fits LRFeatureEngineer once on all "
                  "training rows; the final base model (LR / MLP with the tuned hyperparameters) "
                  "and its StandardScaler are fitted on that full-train feature frame. The final "
                  "calibrator (sigmoid in both arms) is fitted on the out-of-fold base scores of "
                  "all training rows, not on scores of the full-train frame, and is composed with "
                  "the final base model (shared.stage1.oof_calibration, step 4)"),
    "cross_validation": ("10-fold Optuna tuning, the 10 outer folds that produce the out-of-fold "
                         "base scores, and the nested calibration (10 outer x 10 inner folds) each "
                         "refit LRFeatureEngineer and the StandardScaler on the fold's training "
                         "rows from the raw rows (shared.stage1.crossfit build_fold_designs / "
                         "inner_fold_designs; audits F3, F4, F19). The out-of-fold calibrated "
                         "probabilities (threshold selection, the Stage 4 train column) come from "
                         "a calibrator per outer fold fitted on that fold's inner out-of-fold "
                         "scores; seed 42"),
    "fold_refit": "yes (per fold, from X_train_raw)",
    "applied": ("the final base model scores the full-train engineer's transform of X_test (with "
                "the full-train scaler); the final calibrator maps those scores"),
    "differs": ("the final base model is fitted on full-train encodings, in which each training "
                "row's own label enters both target rates (G3, 'Fit rows'); the final calibrator "
                "is fitted on out-of-fold base scores; every cross-validated output uses per-fold "
                "encoders and scalers"),
}
_FIT_STAGE2 = {
    "final_fit": ("PR 38 primary run c3_20261112_153333 (stage2_crossfit.runner.fit_full_train): "
                  "engineer_features(X_train, X_test, y_train) fits RFFeatureEngineer on all "
                  "training rows; the 8 sources are binned with BINNING_STRATEGY; the routers' "
                  "own inner folds read FoldLabelExcludedFeatures bins"),
    "cross_validation": ("5-fold outer cross-fit (shared plan, seed 42): engineer_features(X_tr, "
                         "X_va, y_tr) refits RFFeatureEngineer on each outer fold's training rows "
                         "and transforms the held-out fold; inside each outer fit, inner fold j's "
                         "bins come from FoldLabelExcludedFeatures (pathway fold_label_excluded: "
                         "label statistics recounted without fold j; scaler, KDE reference sample "
                         "and quantile edges kept from the outer fit)"),
    "fold_refit": "yes (outer folds refit; inner folds label-excluded); bin thresholds never refit",
    "applied": ("the full-train engineer transforms X_test and the same thresholds bin it "
                  "(_apply_binning_thresholds); the full-train routers score it"),
    "differs": ("cross-fit outputs (Stage 4 train side) come from outer-fold fits; test-side "
                "outputs from the full-train fit. The notebooks fit no Stage 2: G 8B and B 18 "
                "read both routers by pinned path and SHA-256; B 17 builds full-train bins for "
                "the split checks only"),
}
_FIT_STAGE3 = {
    "final_fit": ("Stage3Block.build fits EBMFeaturePipeline on all training rows (drop leaky -> "
                  "crisis -> integrals -> derivatives -> temporal -> prior -> overlap -> select), "
                  "then FeatureCleaner (inf -> NaN -> training median); the refit model (EBM / "
                  "XGBoost with the tuned hyperparameters) is fitted on that full-train frame. "
                  "The final calibrator (isotonic, applied when the out-of-fold ECE exceeds 0.05) "
                  "and the operating thresholds are fitted on the out-of-fold scores of all "
                  "training rows (shared.stage3.runner)"),
    "cross_validation": ("feature_fit_scope 'per_fold': a fresh pipeline and cleaner on each "
                         "fold's training rows for the 5 tuning folds (tuning, out-of-fold "
                         "scores), the 10 reporting folds and the 5 x 5 inner folds (F22). The "
                         "training-side calibrated scores, nested thresholds and decisions Stage 4 "
                         "reads come from each outer fold's gate, map and threshold fitted on its "
                         "inner out-of-fold scores; seed 42"),
    "fold_refit": "yes (per fold)",
    "applied": ("the refit model scores the full-train pipeline's and cleaner's transform of "
                "X_test; the final calibrator and operating thresholds apply. Each tuning fold's "
                "pipeline also transforms X_test (FoldFrames.X_test)"),
    "differs": ("the refit model is fitted on full-train features (G3, 'Fit rows', says how each "
                "label-dependent column encodes its own fit rows); its calibrator and thresholds "
                "are fitted on out-of-fold scores; the training-side outputs are fold-nested. "
                "GLASS G 17 reuses a cached block only when shared.stage3.block_cache."
                "block_identity matches; black-box B 30 rebuilds it; both recorded block "
                "fingerprint 9cc10fa1... in PR 39 Commit 4"),
}
_FIT_STAGE4 = {
    "final_fit": ("the arbiter is fitted on the training-side OOF outputs of Stages 1-3 "
                  "(shared.stage4 contract checks split, index, labels, folds, router semantics)"),
    "cross_validation": ("inputs are upstream OOF predictions; Meta-XGB is cross-fitted by the "
                         "shared Stage 3 runner (5 folds); the GLASS Arbiter's weights, recall "
                         "cuts and abstention cut are selected on the training OOF table"),
    "fold_refit": "n/a (no engineered features)",
    "applied": "the upstream full-train / refit outputs on X_test",
    "differs": ("train side = OOF predictions, test side = full-train / refit predictions "
                "(the OOF-versus-refit input shift is PR 43 Step 43G)"),
}


@dataclass(frozen=True)
class Population:
    key: str
    stage: int
    arm: str
    model: str
    consumer: str
    registry: str
    features: Tuple[str, ...]
    contract: str
    cells: Tuple[Tuple[str, str], ...]       # (notebook cell, call)
    fit: Mapping[str, str]
    engineer: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        return {"key": self.key, "stage": self.stage, "arm": self.arm, "model": self.model,
                "consumer": self.consumer, "registry": self.registry,
                "features": list(self.features), "n_features": len(self.features),
                "contract": self.contract, "engineer": self.engineer,
                "cells": [list(c) for c in self.cells], "fit": dict(self.fit),
                "extra": {k: (list(map(list, v)) if k == "interactions" else v)
                          for k, v in self.extra.items()}}


POPULATIONS: Tuple[Population, ...] = (
    Population("stage1.glass", 1, "glass", "Calibrated Logistic Regression",
               "glass_pipeline.lr.lr_stage_calibrated.CalibratedLRStage",
               "shared.stage1.feature_engineering.LR_FEATURES", STAGE1_FEATURES,
               "stage1-features 2.0.0 (STAGE1_FEATURE_CONTRACT)",
               (("G 5", "engineer_features"), ("G 8", "train_calibrated_lr_stage")),
               _FIT_STAGE1, engineer=IMPLEMENTATIONS[S1]),
    Population("stage1.blackbox", 1, "blackbox", "Calibrated MLP",
               "blackbox_pipeline.models.mlp.stage.CalibratedStage1MLP",
               "blackbox_pipeline.models.mlp.features.STAGE1_FEATURES", STAGE1_FEATURES,
               "stage1-features 2.0.0 (assert_matches_glass)",
               (("B 7", "engineer_features"), ("B 9", "fit")),
               _FIT_STAGE1, engineer=IMPLEMENTATIONS[S1]),
    Population("stage2.glass", 2, "glass", "GLASS Router (two-pass rules, union-aware ILP)",
               "glass_pipeline.glass_router.pipeline.GlassRouterPipeline "
               "(BankSegmentBuilder.SEGMENT_FEATURES)",
               "shared.stage2.binning.RF_FEATURES_BINARY", STAGE2_BINS,
               "stage2-binning identity (binning_identity)",
               (("G 8B", "load_pinned_stage2"),), _FIT_STAGE2, engineer=IMPLEMENTATIONS[S2],
               extra={"sources": list(STAGE2_SOURCES)}),
    Population("stage2.blackbox", 2, "blackbox", "Two-pass Random Forest Router",
               "blackbox_pipeline.models.rf_router.RFRouter",
               "shared.stage2.binning.RF_FEATURES_BINARY", STAGE2_BINS,
               "stage2-binning identity (binning_identity)",
               (("B 16", "engineer_features"), ("B 17", "engineer_rf_features"),
                ("B 18", "load_pinned_stage2")), _FIT_STAGE2, engineer=IMPLEMENTATIONS[S2],
               extra={"sources": list(STAGE2_SOURCES)}),
    Population("stage3.glass", 3, "glass", "Explainable Boosting Machine",
               "glass_pipeline.ebm.ebm_stage.train_ebm_stage",
               "shared.stage3.feature_engineering.EBM_FEATURES", STAGE3_FEATURES,
               "EBM_FEATURES, 13 features; interactions define_ebm_interactions",
               (("G 17", "load_or_build_block"), ("G 18", "train_ebm_stage")),
               _FIT_STAGE3, engineer=IMPLEMENTATIONS[S3],
               extra={"interactions": EBM_INTERACTIONS}),
    Population("stage3.blackbox", 3, "blackbox", "XGBoost",
               "blackbox_pipeline.models.xgb.train_xgb_stage",
               "shared.stage3.feature_engineering.EBM_FEATURES", STAGE3_FEATURES,
               "EBM_FEATURES, 13 features (the same block as the EBM)",
               (("B 30", "from_global_split"), ("B 31", "train_xgb_stage")),
               _FIT_STAGE3, engineer=IMPLEMENTATIONS[S3]),
    Population("stage4.glass", 4, "glass", "GLASS Arbiter (weighted confidence)",
               "glass_pipeline.glass_arbiter.meta_stage.train_glass_arbiter",
               "glass_pipeline.glass_arbiter.loader.load_stage_outputs", STAGE4_GLASS_ARBITER,
               "Stage 4 contract (shared.stage4); no engineered or raw features",
               (("G 19", "train_glass_arbiter"),), _FIT_STAGE4),
    Population("stage4.blackbox", 4, "blackbox", "Meta-XGBoost",
               "blackbox_pipeline.models.meta_xgb.train_meta_xgb_stage",
               "blackbox_pipeline.models.meta_xgb.config.STAGE4_FEATURES", STAGE4_META_XGB,
               "STAGE4_FEATURES (6 channels; FEATURE_CONTRACT)",
               (("B 35", "MetaXGBConfig"), ("B 36", "train_meta_xgb_stage")), _FIT_STAGE4),
)
POPULATION_BY_KEY: Dict[str, Population] = {p.key: p for p in POPULATIONS}


def _kind(name: str, nodes: Mapping[str, Node]) -> str:
    if name in _STAGE4_CHANNELS:
        return "stage_output"
    if name in RAW_VARIABLES:
        return "raw"
    return "bin" if nodes[name].module == "binning" else "engineered"


def feature_table() -> pd.DataFrame:
    """
    One row per (stage, arm, consumed feature): the Commit 2 per-feature inventory.
    The final-fit and cross-validation scopes are the population's (``POPULATIONS``,
    ``population`` column); ``out_of_fit_rule`` is the feature's own rule for rows
    outside the fit (held-out fold rows and, through the final fit, the test split).
    """
    nodes = all_nodes()
    rows = []
    for p in POPULATIONS:
        for f in p.features:
            kind = _kind(f, nodes)
            fit_rows = ""
            if kind == "stage_output":
                target, sources, path, definition, apply, constants = (
                    STAGE_OUTPUT, (), (), _STAGE4_CHANNELS[f], p.fit["applied"], "")
            elif kind == "raw":
                target, sources, path, definition = NONE, (f,), (), "preprocessed column"
                apply, constants = "the row's value (no transformation)", ""
            else:
                n = nodes[f]
                target, sources, path = n.target, source_variables(f, nodes), fitted_ancestors(f, nodes)
                definition, apply, constants, fit_rows = n.definition, n.apply, n.constants, n.fit_rows
            rows.append({
                "stage": p.stage, "arm": p.arm, "population": p.key, "model": p.model,
                "feature": f, "kind": kind, "definition": definition,
                "source_variables": list(sources), "target_dependence": target,
                "target_path": list(path), "fold_refit": p.fit["fold_refit"],
                "fit_rows": fit_rows, "out_of_fit_rule": apply, "constants": constants,
            })
    return pd.DataFrame(rows)


def population_table() -> pd.DataFrame:
    return pd.DataFrame([p.record() for p in POPULATIONS])


def node_table() -> pd.DataFrame:
    nodes = all_nodes()
    consumers: Dict[str, List[str]] = {}
    for p in POPULATIONS:
        for f in p.features:
            consumers.setdefault(f, []).append(p.key)
        for f in p.extra.get("sources", ()):
            consumers.setdefault(f, []).append(f"{p.key} (source)")
    inputs_of: Dict[str, List[str]] = {}
    for n in nodes.values():
        for i in n.inputs:
            inputs_of.setdefault(i, []).append(n.name)
    rows = []
    for n in nodes.values():
        r = n.record()
        r["source_variables"] = list(source_variables(n.name, nodes))
        r["consumed_by"] = consumers.get(n.name, [])
        r["input_to"] = inputs_of.get(n.name, [])
        rows.append(r)
    return pd.DataFrame(rows)


# ======================================================================
# Verification against the code (no data)
# ======================================================================
def _report():
    from shared.validation import ValidationReport
    return ValidationReport()


def check_registries():
    """Every declared population equals the production registry it names."""
    import pandas as _pd
    from blackbox_pipeline.models.meta_xgb.config import STAGE4_FEATURES
    from blackbox_pipeline.models.mlp.features import STAGE1_FEATURES as MLP_FEATURES
    from feature_research import feature_engineering as research
    from glass_pipeline.ebm.interactions import define_ebm_interactions
    from glass_pipeline.glass_router.core.segment_builder import BankSegmentBuilder
    from shared.stage1.feature_engineering import (
        HISTORICAL_LR_FEATURES, LR_FEATURES, STAGE1_FEATURE_CONTRACT, TARGET_ENCODED_FEATURES)
    from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
    from shared.stage2.feature_engineering import RF_SOURCE_FEATURES
    from shared.stage3.feature_engineering import EBM_FEATURES, LEAKY_FEATURES

    rep = _report()
    c = lambda stage, name, fn, detail="": rep.check("registries", stage, name, fn, detail)  # noqa: E731
    c("Stage 1", "GLASS LR population = shared.stage1 LR_FEATURES (ordered)",
      lambda: tuple(LR_FEATURES) == STAGE1_FEATURES, list(LR_FEATURES))
    c("Stage 1", "= STAGE1_FEATURE_CONTRACT features (2.0.0)",
      lambda: tuple(STAGE1_FEATURE_CONTRACT["features"]) == STAGE1_FEATURES)
    c("Stage 1", "MLP population = blackbox STAGE1_FEATURES (ordered)",
      lambda: tuple(MLP_FEATURES) == STAGE1_FEATURES, list(MLP_FEATURES))
    c("Stage 1", "declared FITTED Stage 1 features = TARGET_ENCODED_FEATURES",
      lambda: tuple(f for f in STAGE1_FEATURES if NODES[f].target == FITTED)
      == tuple(TARGET_ENCODED_FEATURES), list(TARGET_ENCODED_FEATURES))
    c("Stage 2", "sources = shared.stage2 RF_SOURCE_FEATURES (ordered)",
      lambda: tuple(RF_SOURCE_FEATURES) == STAGE2_SOURCES, list(RF_SOURCE_FEATURES))
    c("Stage 2", "bins = shared.stage2 RF_FEATURES_BINARY (ordered)",
      lambda: tuple(RF_FEATURES_BINARY) == STAGE2_BINS, f"{len(RF_FEATURES_BINARY)} bins")
    c("Stage 2", "GLASS Router segment features = RF_FEATURES_BINARY",
      lambda: tuple(BankSegmentBuilder.SEGMENT_FEATURES) == STAGE2_BINS)
    c("Stage 2", "bins in BINNING_STRATEGY order, from the declared sources",
      lambda: tuple(bin_nodes()) == STAGE2_BINS and
      {g["source"] for g in BINNING_STRATEGY.values()} == set(STAGE2_SOURCES))
    c("Stage 3", "population = shared.stage3 EBM_FEATURES (ordered)",
      lambda: tuple(EBM_FEATURES) == STAGE3_FEATURES, f"{len(EBM_FEATURES)} features")
    c("Stage 3", "EBM interactions = define_ebm_interactions(EBM_FEATURES)",
      lambda: tuple(define_ebm_interactions(_pd.DataFrame(columns=list(EBM_FEATURES))))
      == EBM_INTERACTIONS, [list(p) for p in EBM_INTERACTIONS])
    c("Stage 3", "both members of each interaction are Stage 3 features",
      lambda: all(a in STAGE3_FEATURES and b in STAGE3_FEATURES for a, b in EBM_INTERACTIONS))
    c("Stage 3", "LEAKY_FEATURES = the columns BankPreprocessor drops",
      lambda: tuple(LEAKY_FEATURES) == ("poutcome", "pdays", "duration")
      and set(LEAKY_FEATURES) == set(LEAKY_DROPPED))
    c("Stage 4", "Meta-XGB population = STAGE4_FEATURES (ordered)",
      lambda: tuple(STAGE4_FEATURES) == STAGE4_META_XGB, list(STAGE4_FEATURES))
    c("research", "research LR registry = historical Stage 1 names (1.0.0), not consumed",
      lambda: tuple(research.LR_FEATURES) == tuple(HISTORICAL_LR_FEATURES),
      list(research.LR_FEATURES))
    c("research", "research RF registry = RF_SOURCE_FEATURES",
      lambda: tuple(research.RF_FEATURES) == STAGE2_SOURCES)
    c("research", "research EBM registry = EBM_FEATURES + overlap_behavioral_score, not consumed",
      lambda: [f for f in research.EBM_FEATURES if f != "overlap_behavioral_score"]
      == list(STAGE3_FEATURES) and "overlap_behavioral_score" in research.EBM_FEATURES,
      f"{len(research.EBM_FEATURES)} features")
    c("nodes", "every consumed engineered feature is a declared node",
      lambda: all(f in all_nodes() or f in RAW_VARIABLES or f in _STAGE4_CHANNELS
                  for p in POPULATIONS for f in (*p.features, *p.extra.get("sources", ()))))
    return rep.frame()


def _cells(path: Path) -> Dict[str, ast.Module]:
    header = re.compile(r"^#\s*CELL\s+([0-9]+[A-Z]?)\s*:", re.M)
    out = {}
    for c in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        m = header.search(src)
        if m:
            out[m.group(1)] = ast.parse("\n".join(
                ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))))
    return out


def _referenced(tree: ast.AST) -> set:
    """Every name and attribute a cell spells (a callee, or a function passed to a wrapper)."""
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})


NOTEBOOKS = {"G": BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb",
             "B": BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"}


def check_call_sites():
    """The cells each population names call what the inventory says (source only)."""
    rep = _report()
    cells = {k: _cells(p) for k, p in NOTEBOOKS.items()}
    for p in POPULATIONS:
        for cell, call in p.cells:
            nb, cid = cell.split()
            rep.check("call sites", p.key, f"{cell} uses {call}",
                      lambda nb=nb, cid=cid, call=call: call in _referenced(cells[nb][cid]))
    # the per-fold refits are switched on where the stages are fitted
    def kw(nb, cid, call, name):
        return [ast.unparse(k.value) for n in ast.walk(cells[nb][cid]) if isinstance(n, ast.Call)
                and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == call
                for k in n.keywords if k.arg == name]
    rep.check("call sites", "stage1.glass", "G 8 passes X_train_raw (per-fold encoder refits)",
              lambda: kw("G", "8", "train_calibrated_lr_stage", "X_train_raw") == ["X_train"])
    rep.check("call sites", "stage1.blackbox", "B 9 passes X_train_raw (per-fold encoder refits)",
              lambda: kw("B", "9", "fit", "X_train_raw") == ["GLOBAL_SPLIT['X_train']"])
    rep.check("call sites", "stage1.glass", "G 8 calibrates with the sigmoid family",
              lambda: kw("G", "8", "train_calibrated_lr_stage", "calibration_method")
              == ["'sigmoid'"])
    rep.check("call sites", "stage1.blackbox", "B 9 calibrates with the sigmoid family",
              lambda: kw("B", "9", "CalibratedStage1MLP", "calibration_method") == ["'sigmoid'"])
    rep.check("call sites", "stage3.glass", "G 17 builds the block with EBMFeaturePipeline",
              lambda: "EBMFeaturePipeline" in _referenced(cells["G"]["17"]))
    rep.check("call sites", "stage3.blackbox", "B 30 builds the block with EBMFeaturePipeline",
              lambda: "EBMFeaturePipeline" in _referenced(cells["B"]["30"]))
    rep.check("call sites", "stage2", "Stage 2 is read by pin, not fitted (RUN_LEGACY_STAGE2 False)",
              lambda: all(_legacy_off(cells[nb]) for nb in ("G", "B")))
    return rep.frame()


def _legacy_off(cells: Mapping[str, ast.Module]) -> bool:
    values = [n.value for t in cells.values() for n in ast.walk(t) if isinstance(n, ast.Assign)
              and any(isinstance(x, ast.Name) and x.id == "RUN_LEGACY_STAGE2" for x in n.targets)]
    return bool(values) and all(isinstance(v, ast.Constant) and v.value is False for v in values)


def check_fit_protocols():
    """The fit and fold settings the inventory states are the production defaults."""
    import inspect
    import pandas as _pd
    from blackbox_pipeline.models.mlp.config import Stage1MLPConfig
    from blackbox_pipeline.models.xgb import XGBStage3Config
    from glass_pipeline.ebm import EBMStage3Config
    from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
    from shared.pinned_artifacts import STAGE2_RUN_ID
    from shared.stage1.crossfit import preprocessing_record
    from shared.stage1.oof_calibration import STAGE1_CALIBRATION_FAMILY
    from shared.stage3.feature_engineering import EBM_FEATURES
    from stage2_crossfit import inner_features as IF

    rep = _report()
    c = lambda stage, name, fn, detail="": rep.check("fit protocols", stage, name, fn, detail)  # noqa: E731
    lr = inspect.signature(CalibratedLRStage.__init__).parameters
    c("Stage 1", "per-fold encoder and scaler refits given the raw rows",
      lambda: preprocessing_record(_pd.DataFrame())["feature_fit_scope"] == "per_fold"
      and preprocessing_record(_pd.DataFrame())["scaler_fit_scope"] == "per_fold")
    c("Stage 1", "GLASS LR: 10 CV folds, 10 tuning folds, seed 42",
      lambda: (lr["cv_folds"].default, lr["tune_cv_folds"].default, lr["random_state"].default)
      == (10, 10, 42))
    c("Stage 1", "MLP: 10 CV folds", lambda: Stage1MLPConfig().cv_folds == 10)
    c("Stage 1", "shared Stage 1 calibration family is sigmoid (both arms)",
      lambda: STAGE1_CALIBRATION_FAMILY == "sigmoid"
      and Stage1MLPConfig().calibration_method == "sigmoid")
    c("Stage 2", "accepted Stage 2 = PR 38 primary c3_20261112_153333",
      lambda: STAGE2_RUN_ID == "c3_20261112_153333")
    pathway = IF.pathway_identity(IF.PATHWAY_FOLD)
    c("Stage 2", "fold_label_excluded pathway record",
      lambda: list(pathway["record"]) ==
      ["stage2-inner-fold-feature-isolation", "1.0.0", "0d4953ef34d1d3dc"], str(pathway))
    for cfg in (EBMStage3Config, XGBStage3Config):
        k = cfg(expected_features=list(EBM_FEATURES))
        got = (k.feature_fit_scope, k.n_tune_folds, k.n_eval_folds, k.random_state,
               k.clean_inf, k.impute_missing)
        c("Stage 3", f"{cfg.__name__}: per_fold, 5 tuning / 10 reporting folds, seed 42, "
          "inf cleaning and median imputation",
          lambda got=got: got == ("per_fold", 5, 10, 42, True, True), str(got))
        cal = (k.calibration_method, k.ece_threshold, k.nested_oof_threshold)
        c("Stage 3", f"{cfg.__name__}: isotonic calibration gated at OOF ECE > 0.05, nested "
          "out-of-fold thresholds", lambda cal=cal: cal == ("isotonic", 0.05, True), str(cal))
    return rep.frame()


# ======================================================================
# Verification on training rows
# ======================================================================
def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def stage2_dag_frame(engineer, X: pd.DataFrame) -> pd.DataFrame:
    """
    The full Stage 2 DAG frame of a fitted ``RFFeatureEngineer``: the steps of
    its ``transform`` in its own order, before the selection of the 8 sources
    (``check_dag_columns`` proves the selection equals ``transform``).
    """
    import shared.stage2.feature_engineering as s2
    X = s2._add_crisis_features(X)
    X = engineer._integral_eng.transform(X)
    X = s2._add_prior_features(X)
    X = engineer._derivative_eng.transform(X)
    X = engineer._temporal_eng.transform(X)
    return s2._add_overlap_features(X)


def _stage3_columns() -> List[str]:
    """The raw variables and every DAG node the Stage 3 pipeline produces."""
    return [*RAW_VARIABLES, *(n.name for n in _NODE_LIST if S3 in n.implementations)]


def stage3_dag_frame(X: pd.DataFrame, y: pd.Series, rows: Optional[pd.DataFrame] = None):
    """``EBMFeaturePipeline`` fitted on (X, y), selecting every DAG node; transforms ``rows``."""
    from shared.stage3.feature_engineering import EBMFeaturePipeline
    pipe = EBMFeaturePipeline(features=_stage3_columns())
    fitted = pipe.fit_transform(X, y)
    return fitted if rows is None or rows is X else pipe.transform(rows)


def build_training_frames(X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
    """
    Each production feature engineer fitted on all training rows and applied to
    them, as the final fits do (G 5 / B 7, B 17, G 17 / B 30), training side
    only. ``stage2_bins`` come from ``create_binary_features`` (the training path);
    ``stage2_bins_test_path`` from ``_apply_binning_thresholds`` on the same rows
    (the path the test split takes), so the two can be compared.
    """
    from shared.stage1.feature_engineering import LRFeatureEngineer
    from shared.stage2.binning import RF_FEATURES_BINARY, create_binary_features
    from shared.stage2.feature_engineering import RFFeatureEngineer, _apply_binning_thresholds
    from shared.stage3.feature_engineering import EBMFeaturePipeline

    e2 = RFFeatureEngineer().fit(X, y)
    sources = e2.transform(X)
    binned, _ = _quiet(create_binary_features, sources.assign(__y__=y.to_numpy()), "__y__")
    p3 = EBMFeaturePipeline(features=_stage3_columns())
    full3 = p3.fit_transform(X, y)
    reference = pd.DataFrame([
        {"implementation": S2, "fit": "full training split", "fit_rows": int(len(X)),
         "reference_rows": int(len(e2._integral_eng._X_ref))},
        {"implementation": S3, "fit": "full training split", "fit_rows": int(len(X)),
         "reference_rows": int(len(p3.engineers_["integral"]._X_ref))},
    ])
    return {
        "stage1": _quiet(lambda: LRFeatureEngineer().fit(X, y).transform(X)),
        "stage2_dag": stage2_dag_frame(e2, X),
        "stage2_sources": sources,
        "stage2_bins": binned[list(RF_FEATURES_BINARY)],
        "stage2_bins_test_path": _apply_binning_thresholds(sources, list(RF_FEATURES_BINARY)),
        "stage3_dag": full3,
        "stage3": EBMFeaturePipeline().fit_transform(X, y),
        "nsd_reference": reference,
        "nsd_fit_sizes": nsd_fit_sizes(y),
    }


def nsd_fit_sizes(y: pd.Series) -> pd.DataFrame:
    """
    Rows of every production fit of neighborhood_subscription_density, from the
    fold plans on the training labels: the final fits, Stage 2's 5 outer folds
    and Stage 3's 5 tuning, 10 reporting and 5 x 5 inner folds. (Stage 2's inner
    folds refit nothing: they drop fold j's rows from the outer fit's reference
    sample.)
    """
    from glass_pipeline.glass_router.pipeline import make_folds
    from shared.stage_io import fold_assignment

    y = pd.Series(np.asarray(y).astype(int))
    n = len(y)
    tune = fold_assignment(y, cv_folds=5, random_state=42).to_numpy()
    report = fold_assignment(y, cv_folds=10, random_state=42).to_numpy()
    inner = []
    for k in range(5):
        y_tr = y[tune != k].reset_index(drop=True)
        f = fold_assignment(y_tr, cv_folds=5, random_state=42).to_numpy()
        inner += [int((f != j).sum()) for j in range(5)]
    scopes = (("final fits (Stage 2 and Stage 3)", [n]),
              ("Stage 2 outer folds (5)", [len(tr) for tr, _ in make_folds(y, 5, 42)]),
              ("Stage 3 tuning folds (5)", [int((tune != k).sum()) for k in range(5)]),
              ("Stage 3 reporting folds (10)", [int((report != k).sum()) for k in range(10)]),
              ("Stage 3 inner folds (5 x 5)", inner))
    return pd.DataFrame([{"scope": s, "fits": len(v), "min_fit_rows": int(min(v)),
                          "max_fit_rows": int(max(v))} for s, v in scopes])


def check_dag_columns(frames: Mapping[str, pd.DataFrame]):
    """Every declared node is produced by each implementation that claims it."""
    from shared.stage1.feature_engineering import LR_FEATURES
    from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
    from shared.stage3.feature_engineering import EBM_FEATURES

    rep = _report()
    produced = {S1: set(frames["stage1"].columns), S2: set(frames["stage2_dag"].columns),
                S3: set(frames["stage3_dag"].columns)}
    for n in _NODE_LIST:
        for impl in n.implementations:
            rep.check("DAG columns", impl, f"produces {n.name}", lambda n=n, impl=impl:
                      n.name in produced[impl])
    rep.check("DAG columns", S1, "transform returns exactly LR_FEATURES",
              lambda: list(frames["stage1"].columns) == list(LR_FEATURES))
    rep.check("DAG columns", S2, "stage2_dag_frame selects to RFFeatureEngineer.transform",
              lambda: frames["stage2_dag"][list(STAGE2_SOURCES)].equals(frames["stage2_sources"]))
    rep.check("DAG columns", S2, "training-path bins = test-path bins on the same rows",
              lambda: frames["stage2_bins"].astype("int8").equals(
                  frames["stage2_bins_test_path"].astype("int8")))
    for fam, g in BINNING_STRATEGY.items():
        if "bins" in g:
            cols = [b[0] for b in g["bins"]]
            rep.check("DAG columns", S2, f"{fam}: every row in exactly one bin",
                      lambda cols=cols: bool((frames["stage2_bins"][cols].sum(axis=1) == 1).all()))
    rep.check("DAG columns", S2, "29 binary columns", lambda: list(frames["stage2_bins"].columns)
              == list(RF_FEATURES_BINARY) and frames["stage2_bins"].isin([0, 1]).all().all())
    rep.check("DAG columns", S3, "EBMFeaturePipeline selects EBM_FEATURES from the DAG",
              lambda: list(frames["stage3"].columns) == list(EBM_FEATURES)
              and frames["stage3"].equals(frames["stage3_dag"][list(EBM_FEATURES)]))
    for key in ("stage1", "stage2_sources", "stage3"):
        rep.check("DAG columns", key, "all values finite on the training rows",
                  lambda key=key: bool(np.isfinite(frames[key].to_numpy(dtype=float)).all()))
    return rep.frame()


def check_nsd_reference(frames: Mapping[str, pd.DataFrame]):
    """
    The fit-row rule ``_NSD_ROWS`` states, on the real training split: every
    production NSD fit is large enough to sample its reference rows, and the
    full-train fits of both implementations keep exactly the sampled count.
    """
    import shared.stage2.feature_engineering as s2
    from feature_research.feature_engineering import integrals as s3

    rep = _report()
    sizes, ref = frames["nsd_fit_sizes"], frames["nsd_reference"]
    smallest = int(sizes["min_fit_rows"].min())
    rep.check("NSD reference", "both", "the two implementations share the sampling constants",
              lambda: (s2._KDE_LARGE_DS_THRESH, s2._KDE_N_REF)
              == (s3._KDE_LARGE_DS_THRESH, s3._KDE_N_REF))
    rep.check("NSD reference", "both", "every production NSD fit samples its reference rows "
              f"(more than {s2._KDE_LARGE_DS_THRESH:,} fit rows)",
              lambda: smallest > s2._KDE_LARGE_DS_THRESH, f"smallest fit {smallest:,} rows")
    rep.check("NSD reference", "both", f"the full-train fits keep {s2._KDE_N_REF:,} reference "
              "rows in both implementations",
              lambda: bool((ref["reference_rows"] == s2._KDE_N_REF).all()),
              str(ref["reference_rows"].tolist()))
    return rep.frame()


def training_build_summary(frames: Mapping[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
    """Per consumed column of each training frame: distribution and positive rate when active."""
    rows = []
    for key in ("stage1", "stage2_sources", "stage2_bins", "stage3"):
        df = frames[key]
        for col in df.columns:
            v = df[col].to_numpy(dtype=float)
            r = {"frame": key, "column": col, "rows": int(len(v)), "mean": float(v.mean()),
                 "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                 "n_unique": int(pd.Series(v).nunique())}
            if key == "stage2_bins":
                on = v == 1
                r["active_rows"] = int(on.sum())
                r["active_positive_rate"] = float(y.to_numpy()[on].mean()) if on.any() else None
            rows.append(r)
    return pd.DataFrame(rows)


def require(checks: pd.DataFrame, what: str) -> str:
    """'<what>: n/n passed', or AssertionError naming every failed check."""
    bad = checks[~checks["passed"].astype(bool)]
    if len(bad):
        raise AssertionError(f"{what}: {len(bad)} of {len(checks)} checks failed:\n" + "\n".join(
            f"  - [{r.stage}] {r.check}: {r.error or r.detail}" for r in bad.itertuples()))
    return f"{what}: {len(checks)}/{len(checks)} checks passed"


def require_probe(probe: pd.DataFrame) -> str:
    """The target-dependence probe agrees with every declaration, or AssertionError."""
    bad = probe[~probe["agrees"]]
    if len(bad):
        raise AssertionError("declared target dependence disagrees with the code:\n" + "\n".join(
            f"  - {r.implementation} {r.column}: declared {r.declared}, observed {r.observed}"
            for r in bad.itertuples()))
    return (f"target dependence: {len(probe)} columns across {probe['implementation'].nunique()} "
            f"implementations agree with the declarations ({int(probe['rows'].iloc[0])} rows)")


def compare_values(a, b) -> Dict[str, Any]:
    """
    Row-by-row comparison of two equal-length numeric vectors.

    Missing (NaN) on both sides matches; missing on one side only is a mismatch
    (``one_sided_missing``). Two non-missing values match when they are equal
    (so +inf matches +inf). ``max_abs_diff`` is the largest |a - b| over the rows
    where both values are finite (``None`` when no row has two finite values).

    >>> compare_values([np.nan, 10], [2, 13])["rows_differ"], compare_values(
    ...     [np.nan, 10], [2, 13])["max_abs_diff"]
    (2, 3.0)
    """
    x, z = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != z.shape:
        raise ValueError(f"cannot compare shapes {x.shape} and {z.shape}")
    nx, nz = np.isnan(x), np.isnan(z)
    one_sided = nx ^ nz
    both = ~nx & ~nz
    differ = one_sided | (both & (x != z))
    finite = np.isfinite(x) & np.isfinite(z)
    d = np.abs(x[finite] - z[finite])
    return {"rows": int(x.size), "rows_differ": int(differ.sum()),
            "one_sided_missing": int(one_sided.sum()), "both_missing": int((nx & nz).sum()),
            "max_abs_diff": float(d.max()) if d.size else None}


def _changed(a: pd.Series, b: pd.Series) -> int:
    return compare_values(a, b)["rows_differ"]


def probe_target_dependence(X: pd.DataFrame, y: pd.Series, *, seed: int = PROBE_SEED,
                            n_rows: int = PROBE_ROWS) -> pd.DataFrame:
    """
    Fit each implementation on the training rows twice, with the true labels and
    with a seeded permutation of them, and transform the same seeded sample of
    those rows. A column whose values change depends on the labels.

    The declared dependence agrees when: NONE -> no row changes; FITTED or
    INHERITED -> some row changes. A bin is a fixed-threshold function of its
    source (``check_dag_columns``), so a bin of a label-dependent source agrees
    when its family or its source changes (coarse bins can keep every sampled
    row in place), and every bin of a label-free source must not change.
    """
    from shared.stage1.feature_engineering import LRFeatureEngineer
    from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
    from shared.stage2.feature_engineering import RFFeatureEngineer, _apply_binning_thresholds

    rng = np.random.default_rng(seed)
    y_perm = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)
    rows = np.sort(rng.choice(len(X), size=min(int(n_rows), len(X)), replace=False))
    sample = X.iloc[rows]
    nodes = all_nodes()

    def stage1(labels):
        return _quiet(lambda: LRFeatureEngineer().fit(X, labels).transform(sample))

    def stage2(labels):
        frame = stage2_dag_frame(RFFeatureEngineer().fit(X, labels), sample)
        bins = _apply_binning_thresholds(frame[list(STAGE2_SOURCES)], list(RF_FEATURES_BINARY))
        return pd.concat([frame, bins], axis=1)

    def stage3(labels):
        return stage3_dag_frame(X, labels, rows=sample)

    out = []
    for impl, build in ((S1, stage1), (S2, stage2), (S3, stage3)):
        a, b = build(y), build(y_perm)
        for name, n in nodes.items():
            if impl not in n.implementations or name not in a.columns:
                continue
            out.append({"implementation": impl, "column": name, "module": n.module,
                        "declared": n.target, "rows_changed": _changed(a[name], b[name]),
                        "rows": int(len(sample))})
    df = pd.DataFrame(out)
    family = {name: fam for fam, g in BINNING_STRATEGY.items()
              for name in ([g["passthrough"]] if "passthrough" in g else [b[0] for b in g["bins"]])}
    source = {fam: g["source"] for fam, g in BINNING_STRATEGY.items()}
    df["family"] = df["column"].map(family)
    fam_changed = df[df["family"].notna()].groupby("family")["rows_changed"].sum()
    changed = df.set_index(["implementation", "column"])["rows_changed"]

    def agrees(r):
        dependent = r["declared"] in (FITTED, INHERITED)
        if isinstance(r["family"], str):
            src = changed.get((r["implementation"], source[r["family"]]), 0)
            return (fam_changed[r["family"]] > 0 or src > 0) if dependent else r["rows_changed"] == 0
        return (r["rows_changed"] > 0) if dependent else (r["rows_changed"] == 0)

    df["observed"] = np.where(df["rows_changed"] > 0, "label-dependent", "label-free")
    df["agrees"] = df.apply(agrees, axis=1)
    return df


# ======================================================================
# Duplicated implementations (notebook map section 5)
# ======================================================================
DUPLICATE_PAIRS: Tuple[Dict[str, str], ...] = (
    {"stage2_copy": "_add_crisis_features", "research": "crisis.add_crisis_features",
     "difference": "same constants and outputs"},
    {"stage2_copy": "_IntegralFeatureEngineer", "research": "integrals.IntegralFeatureEngineer",
     "difference": ("same scaler, KDE and reference sampling; research checks X/y index "
                    "alignment (check_xy_aligned); seed literal 42 vs config.RANDOM_SEED (42)")},
    {"stage2_copy": "_add_prior_features", "research": "prior.add_prior_features",
     "difference": "same outputs; research also requires the 'previous' column explicitly"},
    {"stage2_copy": "_DerivativeFeatureEngineer",
     "research": "derivatives.DerivativeFeatureEngineer",
     "difference": ("bins: the copy re-cuts with pandas IntervalIndex categories (edges "
                    "rounded to 3 decimals), research with the exact numeric edges; the copy's "
                    "_fill_nans imputes any remaining NaN/inf with the median of the frame being "
                    "transformed, research raises (assert_finite); research checks X/y alignment")},
    {"stage2_copy": "_TemporalFeatureEngineer", "research": "temporal.TemporalFeatureEngineer",
     "difference": ("(sum + k*g)/(count + k) on integer keys vs (count*mean + k*g)/(count + k) "
                    "on string keys: equal up to floating-point rounding; research requires "
                    "integer-encoded day_of_week and month")},
    {"stage2_copy": "_add_overlap_features", "research": "overlap.add_overlap_features",
     "difference": "same constants and outputs"},
    {"stage2_copy": ("_economic_stress_integral, _apply_kde, _add_sigmoid_features, "
                     "_add_decay_features, _add_composites"),
     "research": "the same-named helpers in integrals.py and derivatives.py",
     "difference": "same formulas and constants"},
)
#: Related copies outside notebook map section 5, recorded for the same decision.
RELATED_COPIES: Tuple[Dict[str, str], ...] = (
    {"copy": "shared.stage1.feature_engineering.LRFeatureEngineer",
     "of": ("crisis.cellular_crisis (cellular_downturn), derivatives euribor3m_local_rate "
            "(subscribe_rate_by_euribor_bin, IntervalIndex variant), temporal.dow_month_encoded "
            "(subscribe_rate_by_dow_month, string-key variant)"),
     "note": "Stage 1 production code; its own constants are mirrored, not imported"},
    {"copy": "feature_research.model_training.rf.binning",
     "of": "shared.stage2.binning (BINNING_STRATEGY)",
     "note": "research port; checked equal by duplicate_parity"},
)

#: The version-controlled notebooks the caller scan reads (local executed copies are not callers).
NOTEBOOK_FILES = ("glass_pipeline/glass_cascade.ipynb", "blackbox_pipeline/blackbox_cascade.ipynb",
                  "feature_research/exploratory_feature_research.ipynb",
                  "feature_research/feature_research.ipynb")
TARGET_MODULES = {"stage2_copy": "shared.stage2.feature_engineering",
                  "research": "feature_research.feature_engineering"}
_SCAN_ROOTS = ("shared", "data", "glass_pipeline", "blackbox_pipeline", "feature_research",
               "stage1_calibration", "stage2_binning", "stage2_crossfit", "stage4_regeneration",
               "upstream_regeneration", "phase1_reproduction", "research_logs", "tests")
_PRODUCTION = ("shared", "data", "glass_pipeline", "blackbox_pipeline")


def _module_name(path: Path, root: Path) -> Tuple[str, str]:
    """(module name, package used to resolve its relative imports)."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        return ".".join(parts), ".".join(parts)
    return ".".join(parts), ".".join(parts[:-1])


def _imports(tree: ast.AST, package: str) -> List[Tuple[str, Optional[str]]]:
    """(module, imported name or None) for every import in ``tree``."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[:len(base) - node.level + 1] if node.level > 1 else base
                mod = ".".join([*base, node.module] if node.module else base)
            else:
                mod = node.module or ""
            out += [(mod, a.name) for a in node.names]
    return out


def _targets(mod: str, name: Optional[str], target: str) -> bool:
    full = f"{mod}.{name}" if name else mod
    return (mod == target or mod.startswith(target + ".") or full == target
            or full.startswith(target + "."))


def duplicate_callers(root: Path = BANK_ROOT) -> Dict[str, Any]:
    """
    Current importers of each implementation: Python files (production,
    evidence/protected, tests) and notebooks (by cell), direct imports and
    imports of names a package ``__init__`` re-exports from it.
    """
    root = Path(root)
    files = sorted((p for top in _SCAN_ROOTS for p in (root / top).rglob("*.py")
                    if "__pycache__" not in p.parts),
                   key=lambda p: p.relative_to(root).as_posix())
    parsed = {}
    for p in files:
        try:
            parsed[p] = ast.parse(p.read_bytes())
        except SyntaxError:
            continue
    out: Dict[str, Any] = {}
    for key, target in TARGET_MODULES.items():
        reexports = set()                                  # (package, name)
        for p, tree in parsed.items():
            if p.name != "__init__.py":
                continue
            mod, pkg = _module_name(p, root)
            if _targets(mod, None, target):
                continue
            for m, n in _imports(tree, pkg):
                if n and _targets(m, n, target):
                    reexports.add((mod, n))
        direct, via, internal = {}, {}, []
        for p, tree in parsed.items():
            mod, pkg = _module_name(p, root)
            rel = p.relative_to(root).as_posix()
            if _targets(mod, None, target):
                internal.append(rel)
                continue
            names = _imports(tree, pkg)
            d = sorted({n or m for m, n in names if _targets(m, n, target)})
            v = sorted({f"{m}.{n}" for m, n in names if (m, n) in reexports})
            if d:
                direct[rel] = d
            if v:
                via[rel] = v
        notebooks = {}
        for rel in NOTEBOOK_FILES:
            nb = root / rel
            if not nb.is_file():
                continue
            try:
                cells = _cells(nb)
            except (SyntaxError, ValueError, KeyError):
                continue
            hits = {}
            for cid, tree in cells.items():
                names = _imports(tree, "")
                d = sorted({n or m for m, n in names if _targets(m, n, target)})
                v = sorted({f"{m}.{n}" for m, n in names if (m, n) in reexports})
                if d or v:
                    hits[cid] = {"direct": d, "via_reexport": v}
            if hits:
                notebooks[rel] = hits

        def group(d):
            g: Dict[str, Dict[str, List[str]]] = {"production": {}, "research": {},
                                                  "evidence": {}, "tests": {}}
            for rel, names in d.items():
                top = rel.split("/")[0]
                g["tests" if top == "tests" else "research" if top == "feature_research"
                  else "production" if top in _PRODUCTION else "evidence"][rel] = names
            return g

        out[key] = {"module": target, "internal": sorted(internal),
                    "reexported_by": sorted(f"{m}.{n}" for m, n in reexports),
                    "direct": group(direct), "via_reexport": group(via), "notebooks": notebooks}
    return out


def duplicate_parity(X: pd.DataFrame, y: pd.Series, *, folds: int = 5, seed: int = 42):
    """
    The two Stage 2/3 implementations (and Stage 1's mirrored features) on the
    same training rows: fitted on all training rows and transforming them
    (in-sample), and fitted on each shared fold's training rows and transforming
    its held-out rows. Columns compared with ``compare_values`` (one-sided missing
    values count as differences; the maximum is over comparable finite pairs);
    bins compared by assignment.
    """
    from glass_pipeline.glass_router.pipeline import make_folds
    from shared.stage1.feature_engineering import LRFeatureEngineer
    from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
    from shared.stage2.feature_engineering import RFFeatureEngineer, _apply_binning_thresholds
    from feature_research.model_training.rf.binning import BINNING_STRATEGY as RESEARCH_BINNING

    shared_cols = [n.name for n in _NODE_LIST if set(_DAG) <= set(n.implementations)]
    mirrored = (("cellular_downturn", "cellular_crisis"),
                ("subscribe_rate_by_euribor_bin", "euribor3m_local_rate"),
                ("subscribe_rate_by_dow_month", "dow_month_encoded"))
    acc: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(scope, column, a, b):
        c = compare_values(a, b)
        r = acc.setdefault((scope, column), {"scope": scope, "column": column, "rows": 0,
                                             "rows_differ": 0, "one_sided_missing": 0,
                                             "both_missing": 0, "max_abs_diff": None})
        for k in ("rows", "rows_differ", "one_sided_missing", "both_missing"):
            r[k] += c[k]
        if c["max_abs_diff"] is not None:
            r["max_abs_diff"] = max(r["max_abs_diff"] or 0.0, c["max_abs_diff"])

    def compare(scope, X_fit, y_fit, X_out):
        a = stage2_dag_frame(RFFeatureEngineer().fit(X_fit, y_fit), X_out)
        b = stage3_dag_frame(X_fit, y_fit, rows=X_out)
        for c in shared_cols:
            add(f"{scope}: stage2 copy vs research", c, a[c], b[c])
        bins_a = _apply_binning_thresholds(a[list(STAGE2_SOURCES)], list(RF_FEATURES_BINARY))
        bins_b = _apply_binning_thresholds(b[list(STAGE2_SOURCES)], list(RF_FEATURES_BINARY))
        add(f"{scope}: stage2 copy vs research", "any of the 29 bins (rows)",
            (bins_a.to_numpy() != bins_b.to_numpy()).any(axis=1), np.zeros(len(bins_a)))
        s1 = _quiet(lambda: LRFeatureEngineer().fit(X_fit, y_fit).transform(X_out))
        for s1c, other in mirrored:
            add(f"{scope}: stage1 vs stage2 copy", f"{s1c} ~ {other}", s1[s1c], a[other])
            add(f"{scope}: stage1 vs research", f"{s1c} ~ {other}", s1[s1c], b[other])

    compare("in-sample (all training rows)", X, y, X)
    for tr, va in make_folds(y, folds, seed):
        compare(f"held-out ({folds} shared folds)", X.iloc[tr], y.iloc[tr], X.iloc[va])
    df = pd.DataFrame(list(acc.values()))
    binning = {"research port equals shared.stage2.binning BINNING_STRATEGY":
               RESEARCH_BINNING == BINNING_STRATEGY}
    return df, binning


def callers_summary(callers: Mapping[str, Any]) -> str:
    """One line per implementation and caller group: how many files import it."""
    lines = []
    for c in callers.values():
        counts = []
        for group in ("production", "research", "evidence", "tests"):
            n = len(set(c["direct"][group]) | set(c["via_reexport"][group]))
            counts.append(f"{group} {n}")
        nb = sum(len(v) for v in c["notebooks"].values())
        lines.append(f"{c['module']}: " + ", ".join(counts) + f", notebook cells {nb}")
    return "\n".join(lines)


# ======================================================================
# Markdown (the generated part of research_logs/pr42/feature_inventory.md)
# ======================================================================
BEGIN = "<!-- BEGIN GENERATED: python -m feature_research.feature_inventory -->"
END = "<!-- END GENERATED -->"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(v):
        s = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        return s.replace("|", "\\|").replace("*", "\\*").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _code(s: str) -> str:
    return f"`{s}`" if s else ""


def render_markdown(root: Path = BANK_ROOT) -> str:
    """The generated tables, between ``BEGIN`` and ``END`` (data-free, deterministic)."""
    from feature_research import frozen_references as FR

    parts = [BEGIN, ""]
    parts.append("### G1. Stage feature populations\n")
    parts.append(_md_table(
        ("Stage", "Arm", "Model", "Consumed features", "Registry", "Contract", "Notebook cells"),
        [(p.stage, p.arm, p.model, f"{len(p.features)}: " + ", ".join(map(_code, p.features)),
          _code(p.registry), p.contract, "; ".join(f"{c} `{f}`" for c, f in p.cells))
         for p in POPULATIONS]))
    parts.append("\nStage 2 sources (both routers, before binning): "
                 + ", ".join(map(_code, STAGE2_SOURCES)) + ".")
    parts.append("\nEBM interaction pairs (Stage 3, GLASS arm only): "
                 + "; ".join(f"`{a}` x `{b}`" for a, b in EBM_INTERACTIONS) + ".\n")
    parts.append("### G2. Fitting scope by stage\n")
    seen = []
    rows = []
    for p in POPULATIONS:
        if p.fit in seen:
            continue
        seen.append(p.fit)
        arms = ", ".join(q.key for q in POPULATIONS if q.fit is p.fit)
        rows.append((arms, p.fit["final_fit"], p.fit["cross_validation"], p.fit["fold_refit"],
                     p.fit["applied"], p.fit["differs"]))
    parts.append(_md_table(("Populations", "Final fit (full training split)",
                            "Cross-validation", "Fold refit", "Test-time application",
                            "Where CV and final fitting differ"), rows))
    parts.append("\n### G3. Per-feature inventory\n")
    parts.append("Arms: `G` GLASS, `B` black-box. Target dependence: `none` (a function of the "
                 "row), `fitted` (a statistic of the fit's training labels), `inherited` "
                 "(computed from a fitted input), `stage_output` (an upstream prediction). "
                 "Fitting scope and fold refits are the stage's (G2). 'Fit rows' states how a "
                 "label-dependent column encodes the rows of the fit that computed it (the final "
                 "fit's training rows, or a fold's training rows); every other row follows the "
                 "rule outside the fit.\n")
    table = feature_table()
    for stage in (1, 2, 3, 4):
        sub = table[table["stage"] == stage]
        parts.append(f"**Stage {stage}**\n")
        rows = []
        for f in dict.fromkeys(sub["feature"]):
            r = sub[sub["feature"] == f]
            arms = "/".join("G" if a == "glass" else "B" for a in r["arm"])
            first = r.iloc[0]
            rows.append((_code(f), arms, first["kind"], first["definition"],
                         ", ".join(map(_code, first["source_variables"])),
                         first["target_dependence"],
                         ", ".join(map(_code, first["target_path"])), first["fit_rows"],
                         first["out_of_fit_rule"], first["constants"]))
        parts.append(_md_table(("Feature", "Arms", "Kind", "Definition", "Source variables",
                                "Target", "Fitted ancestors", "Fit rows (the fit's own rows)",
                                "Rule outside the fit (held-out and test rows)",
                                "Fixed constants"), rows))
        parts.append("")
    parts.append("### G4. Intermediate DAG nodes (not consumed directly)\n")
    listed = set(table["feature"])
    nt = node_table()
    def used(r):
        out = list(r["consumed_by"])
        if r["input_to"]:
            out.append("input to " + ", ".join(f"`{x}`" for x in r["input_to"]))
        return "; ".join(out) or "unused by the cascades"
    rows = [(_code(r["name"]), r["module"], r["definition"], ", ".join(map(_code, r["inputs"])),
             r["target"], r["fit_rows"], used(r))
            for _, r in nt.iterrows() if r["name"] not in listed and r["module"] != "binning"]
    parts.append(_md_table(("Node", "Step", "Definition", "Inputs", "Target", "Fit rows",
                            "Used by"), rows))
    parts.append("\n### G5. Frozen references\n")
    parts.append(_md_table(("Reference", "Value on this branch", "Recorded value",
                            "Recorded in", "Status", "Note"),
                           [(r["name"], r["computed"], r["recorded"], r["recorded_in"],
                             r["status"], r["note"]) for r in FR.static_references(root)]))
    parts.append("\n### G6. Duplicated feature-engineering implementations\n")
    parts.append(_md_table(("`shared/stage2/feature_engineering.py`",
                            "`feature_research/feature_engineering/`", "Difference (from code)"),
                           [(_code(d["stage2_copy"]), _code(d["research"]), d["difference"])
                            for d in DUPLICATE_PAIRS]))
    parts.append("")
    parts.append(_md_table(("Related copy", "Of", "Note"),
                           [(_code(d["copy"]), d["of"], d["note"]) for d in RELATED_COPIES]))
    parts.append("\n### G7. Current callers\n")
    callers = duplicate_callers(root)
    for key, c in callers.items():
        parts.append(f"**`{c['module']}`** (re-exported as "
                     f"{', '.join(map(_code, c['reexported_by'])) or 'nothing'})\n")
        rows = []
        for how in ("direct", "via_reexport"):
            for group, files in c[how].items():
                for rel, names in sorted(files.items()):
                    rows.append((group, how.replace("_", " "), _code(rel),
                                 ", ".join(map(_code, names))))
        for rel, cells in c["notebooks"].items():
            for cid, hit in cells.items():
                for how in ("direct", "via_reexport"):
                    if hit[how]:
                        rows.append(("notebook", how.replace("_", " "), f"`{rel}` cell {cid}",
                                     ", ".join(map(_code, hit[how]))))
        parts.append(_md_table(("Group", "Import", "File", "Names"), rows))
        parts.append("")
    parts.append(END)
    return "\n".join(parts)


def generated_block(text: str) -> str:
    """The generated block of a document (``BEGIN`` through ``END``)."""
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0:
        raise ValueError("no generated block in the document")
    return text[start:end + len(END)]


if __name__ == "__main__":                                   # pragma: no cover
    print(render_markdown())
