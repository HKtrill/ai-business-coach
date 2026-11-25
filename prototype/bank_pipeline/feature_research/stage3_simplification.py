"""
feature_research.stage3_simplification
======================================
PR 42 Commit 5, Step 5A: can the frozen PR 39 Stage 3 EBM be simplified so its
contributions are easier to explain and more stable across folds and contact
periods, and what does each simplification gain or cost in predictive
performance, calibration and complementarity with Stage 1?

The comparison is registered first (``REGISTRATION``; its SHA-256 is pinned in
``REGISTERED_SHA256`` and checked before any candidate is fitted). The baseline
is the accepted 13-feature, three-pair EBM, refitted on the training side with
the accepted settings through the existing Stage 3 code:

* feature block: ``shared.stage3.data.Stage3Block.build`` with the production
  ``EBMFeaturePipeline`` refitted inside every outer and inner fold (the test
  split is never bound; a one-row placeholder from the training rows stands in
  for the test frame the builder requires and is never scored or reported);
* fits: ``glass_pipeline.ebm.estimator.EBMFactory`` with the accepted
  hyperparameters, out-of-fold and inner out-of-fold scores through
  ``shared.stage3.evaluation.OOFGenerator``;
* calibration and thresholds: ``shared.stage3.calibration.Stage3Calibrator``
  and ``shared.stage3.thresholding.F2ThresholdSelector`` with the
  ``EBMStage3Config`` protocol (fold-nested isotonic map and thresholds);
* metrics: ``shared.stage3.evaluation.Stage3Metrics`` (ROC-AUC, PR-AUC, Brier,
  ECE on 10 equal-width bins, F2 / recall / precision).

Every candidate uses the same folds, rows, seeds, settings and protocol. Pairs
are resolved by feature name; a pair whose feature is removed or replaced is
recorded as lost and its slot is left empty (Step 5B owns reselection).

Accepted artifacts are read only through the PR 38 boundary extractors, and
only where they exist locally (``models/`` is Git-ignored): the Stage 1 LR, the
GLASS router and the Stage 3 EBM views supply row-aligned out-of-fold vectors
for baseline parity and the complementarity tables. Training side only.

Like Steps 4A and 4B, this module imports no Stage 2 or research feature-
engineering module (it reaches the production engineers only through
``shared.stage3.feature_engineering``), so the Commit 2 caller scan and the
Step 3A digests are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BANK_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 5

#: The frozen Stage 3 inventory (``shared.stage3.feature_engineering.EBM_FEATURES``; PR 39 Commit 4
#: run record ``evidence.stage3.stage3.ebm.features``) and its three pairs
#: (``glass_pipeline.ebm.interactions.define_ebm_interactions``; glass_cascade Cell 18).
BASELINE_FEATURES = (
    "decay_x_density", "euribor3m_sigmoid_slope", "economic_curvature_intensity", "cons.conf.idx", "age",
    "dow_month_encoded", "default", "prior_x_stress", "campaign", "overlap_default_clean",
    "cpi_high_cellular", "behavioral_favorability", "emp_var_rate_sigmoid_slope")
BASELINE_PAIRS = (("campaign", "emp_var_rate_sigmoid_slope"),
                  ("euribor3m_sigmoid_slope", "economic_curvature_intensity"),
                  ("prior_x_stress", "decay_x_density"))
#: The remaining correlated pair: the frozen inventory's pruning note (shared.stage3.feature_engineering,
#: "cut 2": |r| = 0.724). First and second member in EBM_FEATURES order.
CORRELATED_PAIR = ("euribor3m_sigmoid_slope", "emp_var_rate_sigmoid_slope")
CORRELATION_THRESHOLD = 0.7          # feature_research.model_training.ebm.ranking.correlated_pairs default
JED = "joint_economic_decay"
NSD = "neighborhood_subscription_density"
PAST_FEATURE = "decay_x_density_past"
PAST_NSD = "nsd_past"
#: Extra block columns: JED (a candidate feature) and NSD (to check decay_x_density = JED x NSD).
BLOCK_EXTRA = (JED, NSD)

#: The accepted EBM hyperparameters: glass_cascade.ipynb Cell 18's printed "best params" (the only tracked
#: source; printed to six decimals), read from the notebook's Git blob below.
PARAMS = {"learning_rate": 0.04994, "max_rounds": 2100, "max_bins": 480, "max_interaction_bins": 80}
PARAMS_SOURCE = {"path": "glass_pipeline/glass_cascade.ipynb", "commit": "0062625fd8ab97e54b9f214affc2fbba79101b5c",
                 "blob": "a366cc403e58d559d76702d54e771ec25ee83af5", "cell": "CELL 18: STAGE 3 — EBM",
                 "printed": {"learning_rate": "0.049940", "max_rounds": "2100", "max_bins": "480",
                             "max_interaction_bins": "80"}}
#: The accepted training-side figures the baseline refit is compared with (tracked records).
ACCEPTED_RECORDS = {
    "c4_run": "research_logs/pr39/c4_20261115_204125/run_record.json",
    "c6_reproduction": "research_logs/pr39/c6_20261116_194111/reproduction.json",
}
FOLD_FINGERPRINT = "8aea082fcaf371b7"

#: Past-only density (Step 3B rules): outcome lag 0, cold start, minimum history.
COLD_START_DAYS = 3
MIN_HISTORY_ROWS = 500
MIN_HISTORY_POSITIVES = 10
KDE_BANDWIDTH = 1.0
MACRO = ("euribor3m", "nr.employed", "emp.var.rate")

#: Nomination rules (registered tolerances).
TOL = {"roc_auc": -0.005, "roc_auc_lower": -0.010, "pr_auc": -0.010, "nested_f2": -0.010,
       "calibrated_brier": 0.002, "ece_gate": 0.05, "net_abstain_share": -0.005}

#: Shapes.
GRID_POINTS = 60
GRID_QUANTILES = (0.005, 0.995)
DISCRETE_MAX_UNIQUE = 30
SUPPORT_BINS = 30
SPARSE_SHARE = 0.01
SINGLE_PERIOD_SHARE = 0.5

CANDIDATES = ("drop_first", "drop_second", "jed_alone", "past_density")


def _pair_name(p: Sequence[str]) -> str:
    return f"{p[0]} x {p[1]}"


# ======================================================================
# 1. registration (fixed before any candidate result was computed)
# ======================================================================
REGISTRATION = {
    "id": "pr42-step5a-registration",
    "version": "1.0.0",
    "revisions": [{"version": "1.0.0", "status": "fixed before any candidate result was computed"}],
    "question": (
        "Can the frozen PR 39 Stage 3 EBM be simplified so its contributions are easier to explain and more "
        "stable across folds and contact periods? What does each simplification gain or cost in out-of-fold "
        "predictive performance, calibration and its ability to correct Stage 1 mistakes, especially on rows "
        "where Stage 2 abstains?"),
    "known_before_registration": (
        "The baseline refit (no candidate) was run once in the sandbox to check parity against the PR 39 "
        "records: its global raw F2 threshold equals the record's (0.53); its out-of-fold ECE before and after "
        "calibration, its calibrated threshold and three of its five fold-nested thresholds differ slightly "
        "(Linux against the accepted Windows run, and a learning rate transcribed to six decimals). No "
        "candidate feature set was fitted, and no candidate out-of-fold statistic computed, before this "
        "registration was fixed."),
    "baseline": {
        "features": list(BASELINE_FEATURES),
        "features_source": ("shared.stage3.feature_engineering.EBM_FEATURES; equal to the PR 39 Commit 4 run "
                            "record evidence.stage3.stage3.ebm.features (checked)"),
        "pairs": [list(p) for p in BASELINE_PAIRS],
        "pairs_source": ("glass_pipeline.ebm.interactions.define_ebm_interactions on the baseline columns "
                         "(checked); glass_cascade Cell 18 prints the same three"),
        "params": dict(PARAMS),
        "params_source": PARAMS_SOURCE,
        "fixed_settings": ("EBMFactory(random_state=42, n_jobs=1), interpret's other defaults; balanced sample "
                           "weights; features refitted per fold"),
        "fold_plan": (f"shared.stage_io.fold_assignment(y, {N_FOLDS}, {SEED}) (Stage3Block's tune plan), "
                      f"fingerprint {FOLD_FINGERPRINT} = the PR 39 record's stage3.ebm.fold_fingerprint; inner "
                      "plans FoldPlan.build(y_fit, 5, 42) on each fit side"),
        "calibration": ("EBMStage3Config: isotonic, gated at out-of-fold ECE > 0.05 on 10 equal-width bins; on "
                        "the training side each fold's gate and map come from inner out-of-fold scores of its fit "
                        "rows (Stage3Calibrator, PR 36 F22)"),
        "thresholds": ("F2ThresholdSelector (beta 2, grid 0.10-0.90 in 81 steps): fold-nested raw thresholds (the "
                       "primary operating point ebm.nested_f2), the global raw F2 threshold and the global "
                       "calibrated threshold (both in-selection)"),
    },
    "parity": {
        "records": ("the refitted baseline's out-of-fold ECE before and after calibration, global raw and "
                    "calibrated thresholds and fold-nested thresholds against the PR 39 Commit 4 run record and "
                    "the Commit 6 reproduction record (training side)"),
        "row_aligned": ("where the accepted EBM artifact exists locally, read through "
                        "shared.stage2.diagnostics.boundary.extract_ebm_view at its pinned SHA-256: raw and "
                        "calibrated out-of-fold scores and stored nested decisions, row by row (maximum absolute "
                        "difference, correlation, ROC-AUC difference, decisions that differ)"),
        "use": ("descriptive, and explained where it differs. Every candidate is compared with the refitted "
                "baseline (same code, platform, settings), never with the accepted artifact's vectors"),
    },
    "correlated_pair": {
        "rule": (f"the frozen inventory's pruning note names {CORRELATED_PAIR[0]} and {CORRELATED_PAIR[1]} "
                 f"(|r| = 0.724). Verified on the training rows: on the full-train block and on every outer fit "
                 f"side it must be the only pair of the 13 features with |Pearson r| >= {CORRELATION_THRESHOLD}. "
                 "Otherwise the run stops (the registration would need a revision)"),
        "order": "first and second member in EBM_FEATURES order",
    },
    "candidates": {
        "drop_first": {"question": "correlated pair, first ablation", "remove": [CORRELATED_PAIR[0]],
                       "add": [], "expected_lost_pairs": [_pair_name(BASELINE_PAIRS[1])]},
        "drop_second": {"question": "correlated pair, second ablation", "remove": [CORRELATED_PAIR[1]],
                        "add": [], "expected_lost_pairs": [_pair_name(BASELINE_PAIRS[0])]},
        "jed_alone": {"question": "decay_x_density simplification", "remove": ["decay_x_density"], "add": [JED],
                      "expected_lost_pairs": [_pair_name(BASELINE_PAIRS[2])],
                      "note": (f"{JED} is not among the 13, so the candidate keeps 13 features; it does not "
                               "inherit decay_x_density's pair")},
        "past_density": {"question": "past-only density variant", "remove": ["decay_x_density"],
                         "add": [PAST_FEATURE], "expected_lost_pairs": [_pair_name(BASELINE_PAIRS[2])],
                         "note": (f"{PAST_FEATURE} = {JED} x the past-only density; a replacement feature, so it "
                                  "does not inherit decay_x_density's pair")},
        "pair_rule": ("each candidate's pairs are the baseline pairs whose two features it keeps, resolved by "
                      "name; the fitted EBM's term names must be exactly its features plus those pairs"),
        "identical": "two candidates with the same feature set and pairs are fitted once and both mapped to it",
        "attribution": ("each difference from the baseline is the whole simplification's: removing a feature "
                        "and its pair, or replacing a feature and losing its pair, measures their combined effect"),
        "exploratory": "any further simplification suggested by the results is labelled exploratory",
    },
    "past_only_density": {
        "accepted": ("neighborhood_subscription_density: a Gaussian kernel average (bandwidth 1.0) of the fit "
                     "rows' labels in standardized (euribor3m, nr.employed, emp.var.rate) space; the scaler is "
                     "fitted on the fit rows; above 10,000 fit rows the reference is a 5,000-row draw "
                     "(default_rng(42)); a fit row's own label is in its own value"),
        "past_only": (f"for a row contacted on day d (Commit 3 contact days), the same kernel average over the "
                      f"context's rows contacted on a day before d, with the scaler fitted on those rows and every "
                      f"one of them in the reference (the history is grown, not redrawn: Step 3B left this design "
                      f"choice open). Same-day labels are never used"),
        "outcome_availability": ("assumption (Step 3B's primary rule, lag 0): the outcome of a contact is known by "
                                 "the next contact day. Feasibility at prediction time depends on it; the outcome "
                                 "time is not recorded"),
        "contexts": ("outer fold k: its fit rows (values for its fit rows and its held-out rows); inner fold j of "
                     "fold k: the inner fit rows. A held-out row's value never uses another held-out row's label"),
        "cold_start": (f"a row has a value only from the first contact day at which its context's history holds "
                       f"at least {MIN_HISTORY_ROWS} rows and {MIN_HISTORY_POSITIVES} positives, and never on the "
                       f"first {COLD_START_DAYS} contact days; earlier rows are missing and the production "
                       "FeatureCleaner imputes them with the fit rows' median, as for every Stage 3 column. The "
                       "missing rows are counted per context"),
        "verification": ("on sampled days of one context, the production IntegralFeatureEngineer fitted on exactly "
                         "the available rows (n_ref = their count, so no subsample) gives the same values"),
        "other_inputs": "joint_economic_decay is a deterministic function of the current macro values",
    },
    "evaluation": {
        "rows": "the 32,950 training rows; the test split is never bound (G7)",
        "folds": "the baseline's outer and inner plans for every candidate",
        "isolation": ("features refitted inside each fold by the production pipeline; the past-only density is "
                      "fold-isolated and contact-ordered; calibration and thresholds fold-nested"),
        "same_settings": "the baseline's hyperparameters, seeds and protocol; no retuning",
    },
    "metrics": [
        "feature count; fitted interaction count; retained and lost pairs, by name",
        "out-of-fold ROC-AUC and PR-AUC (raw scores), Brier and ECE (10 equal-width bins) on raw scores and on "
        "fold-nested calibrated probabilities, each labelled raw or calibrated",
        "threshold metrics (F2, recall, precision, flagged share) at the fold-nested raw thresholds (primary), the "
        "global raw F2 threshold and the global calibrated threshold (both in-selection)",
        "paired differences from the baseline on the same rows: ROC-AUC with DeLong's paired standard error "
        "(feature_research.temporal_validity.delong_paired), PR-AUC, Brier, ECE, nested F2; per-fold values and "
        "differences (mean, sd, range)",
        "decisions at the nested thresholds that differ from the baseline's",
    ],
    "shapes": {
        "models": "each configuration's five outer-fold models and its all-rows fit (for shapes only; never scored)",
        "units": ("main effects in the EBM's additive term score (log-odds); the calibrated probability is a "
                  "separate quantity and is never plotted as a term"),
        "grid": (f"{GRID_POINTS} points between the {GRID_QUANTILES[0]:.1%} and {GRID_QUANTILES[1]:.1%} quantiles "
                 f"of the full-train block values, or every value for a feature with at most "
                 f"{DISCRETE_MAX_UNIQUE} distinct values; the same grid for a feature in every model"),
        "centering": ("each curve is evaluated on the grid with the model's own bins and shifted so its mean over "
                      "the 32,950 training rows is 0 (the EBM centres each term on its own fit rows; the shift is "
                      "recorded)"),
        "support": (f"training rows per {SUPPORT_BINS} equal-width bins of the plotted range; bins under "
                    f"{SPARSE_SHARE:.0%} of the rows are sparse and shaded"),
        "periods": (f"rows per contact period per bin; a bin whose largest period holds more than "
                    f"{SINGLE_PERIOD_SHARE:.0%} of its rows is a single-period region"),
        "interactions": ("each retained pair's term score on a grid of both features (all-rows fit), with joint "
                         "row support; one symmetric colour scale per pair across models"),
    },
    "complementarity": {
        "decisions": ("Stage 1: lr.in_stage_f2_cv (calibrated LR out-of-fold probability >= its stored threshold); "
                      "the accepted EBM: ebm.nested_f2 (stored decisions); refitted EBMs: raw out-of-fold score >= "
                      "their own fold-nested threshold (the same rule). Correct = decision equals the label"),
        "population": ("all training rows with out-of-fold outputs, aligned by row identity; Stage 2 abstentions "
                       "= the accepted GLASS router's out-of-fold 'uncertain' rows"),
        "counts": ("both correct, Stage 1 only correct, EBM only correct, both wrong, with denominators, for all "
                   "rows and for abstentions; each table must add up to its population"),
        "comparison": ("Stage 1 errors the EBM corrects (EBM only correct) and correct Stage 1 decisions it "
                       "overturns (Stage 1 only correct); net = corrected - overturned"),
        "availability": ("computed only where the accepted Stage 1 LR, GLASS router and EBM artifacts exist "
                         "locally, through the boundary extractors; otherwise reported as unavailable"),
        "limit": "a disagreement count does not establish a better cascade decision",
    },
    "nomination": {
        "step5b": (f"a candidate is recommended for Step 5B reselection if its paired ROC-AUC difference is >= "
                   f"{TOL['roc_auc']} with a 95% lower bound >= {TOL['roc_auc_lower']}, its PR-AUC difference "
                   f">= {TOL['pr_auc']}, its nested F2 difference >= {TOL['nested_f2']}, its calibrated ECE <= "
                   f"{TOL['ece_gate']} and its calibrated Brier <= the baseline's + {TOL['calibrated_brier']}, "
                   "and its shapes give a documented interpretability gain"),
        "cascade": (f"a candidate meeting those tolerances is nominated for later cascade evaluation if, on Stage 2 "
                    f"abstentions, its net count is at least the baseline's net count minus "
                    f"{-TOL['net_abstain_share']:.1%} of the abstention rows (computed only where the accepted "
                    "artifacts are available locally)"),
        "modest_loss": ("a simpler model may advance with a loss inside these tolerances when the interpretability "
                        "gain and calibration trade-off are documented"),
    },
    "out_of_scope": ["interaction reselection (Step 5B)", "interaction budgets (PR 44 Step 44C)",
                     "end-to-end cascade comparisons and adoption (PRs 44-45)", "chronological splits (PR 49)"],
}

#: SHA-256 of the 1.0.0 registration, fixed before any candidate result was computed.
REGISTERED_SHA256 = "08b3a74474a3cd35dbd4d68e39b13cf015c9b9a1a655080a2802ad3cafbb09bc"


def registration_digest(registration: Mapping[str, Any] = REGISTRATION) -> str:
    """SHA-256 of the canonical registration JSON (recorded beside the results)."""
    return hashlib.sha256(json.dumps(registration, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


# ======================================================================
# helpers
# ======================================================================
def _check(name: str, passed: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def require(checks: pd.DataFrame, what: str) -> str:
    """Raise if any check failed; otherwise a one-line summary."""
    col = "passed" if "passed" in checks else "holds"
    bad = checks[~checks[col].astype(bool)]
    if len(bad):
        raise AssertionError(f"{what}: {len(bad)} check(s) failed\n{bad.to_string()}")
    return f"{what}: {len(checks)} checks passed"


def _canonical(obj: Any) -> Any:
    from upstream_regeneration.records import canonical
    return canonical(obj)


def _read_json(rel: str, root: Path = BANK_ROOT) -> Dict[str, Any]:
    return json.loads((root / rel).read_text(encoding="utf-8"))


# ======================================================================
# 2. the baseline's sources
# ======================================================================
def _git_blob(blob: str, root: Path) -> Optional[bytes]:
    import subprocess
    try:
        out = subprocess.run(["git", "cat-file", "blob", blob], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def params_transcription(root: Path = BANK_ROOT) -> Dict[str, Any]:
    """
    The accepted EBM hyperparameters as glass_cascade Cell 18 printed them, read from the notebook's Git
    blob (or the working copy, LF-normalized, only if its blob id is the pinned one), and compared with
    ``PARAMS``. Status: ``verified``, ``unavailable`` (no source) or ``mismatch``.
    """
    data, source = _git_blob(PARAMS_SOURCE["blob"], root), "git blob"
    if data is None:
        wc = root / PARAMS_SOURCE["path"]
        raw = wc.read_bytes().replace(b"\r\n", b"\n") if wc.is_file() else None
        data, source = (raw, "working copy (blob id verified)") if raw is not None and \
            _blob_id(raw) == PARAMS_SOURCE["blob"] else (None, "unavailable")
    if data is None:
        return {"status": "unavailable", "source": source, "printed": None}
    nb = json.loads(data.decode("utf-8"))
    text = ""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and PARAMS_SOURCE["cell"] in "".join(c["source"]):
            text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("text"))
    printed = {}
    if "best params:" in text:
        block = text.split("best params:", 1)[1].split("-----", 1)[0]
        for line in block.splitlines():
            if "=" in line:
                k, v = (s.strip() for s in line.split("=", 1))
                printed[k] = v
    ok = printed == PARAMS_SOURCE["printed"] and all(
        abs(float(printed[k]) - float(PARAMS[k])) == 0.0 for k in PARAMS)
    return {"status": "verified" if ok else "mismatch", "source": source, "printed": printed}


def baseline_checks(root: Path = BANK_ROOT) -> pd.DataFrame:
    """The baseline's features, pairs, settings and parameters against their authoritative sources."""
    from glass_pipeline.ebm.config import EBMStage3Config
    from glass_pipeline.ebm.interactions import define_ebm_interactions
    from shared.stage3.feature_engineering import EBM_FEATURES
    rec = _read_json(ACCEPTED_RECORDS["c4_run"], root)["evidence"]["stage3"]["stage3.ebm"]
    cfg = EBMStage3Config(expected_features=list(EBM_FEATURES))
    pairs = define_ebm_interactions(pd.DataFrame(columns=list(BASELINE_FEATURES)))
    tr = params_transcription(root)
    out = [
        _check("the 13 features are shared.stage3 EBM_FEATURES and the PR 39 record's, in order",
               tuple(EBM_FEATURES) == BASELINE_FEATURES == tuple(rec["features"]), len(EBM_FEATURES)),
        _check("the three pairs are define_ebm_interactions' on the baseline columns",
               tuple(map(tuple, pairs)) == BASELINE_PAIRS, pairs),
        _check("the recorded fold fingerprint is the registered one", rec["fold_fingerprint"] == FOLD_FINGERPRINT,
               rec["fold_fingerprint"]),
        _check("the protocol is the accepted one (isotonic, gate 0.05, 10 bins, nested thresholds, F2 grid "
               "0.10-0.90 x 81, balanced weights, per-fold features)",
               (cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins, cfg.nested_oof_threshold, cfg.beta,
                cfg.threshold_low, cfg.threshold_high, cfg.threshold_steps, cfg.class_weight,
                cfg.feature_fit_scope, cfg.random_state, cfg.n_jobs, cfg.n_tune_folds) ==
               ("isotonic", 0.05, 10, True, 2.0, 0.10, 0.90, 81, "balanced", "per_fold", 42, 1, 5),
               cfg.calibration_method),
        _check(f"the hyperparameters are glass_cascade Cell 18's printed best params ({tr['source']})",
               tr["status"] == "verified", tr["printed"]),
        _check("the PR 39 record applied isotonic calibration and selected the global raw threshold 0.53",
               rec["calibration"]["applied"] and rec["calibration"]["method"] == "isotonic"
               and rec["threshold"] == 0.53, rec["calibration"]),
    ]
    return pd.DataFrame(out)


def accepted_figures(root: Path = BANK_ROOT) -> Dict[str, Any]:
    """The accepted EBM's training-side figures from the tracked PR 39 records."""
    c4 = _read_json(ACCEPTED_RECORDS["c4_run"], root)["evidence"]["stage3"]["stage3.ebm"]
    c6 = _read_json(ACCEPTED_RECORDS["c6_reproduction"], root)["reproduction"]["roles"]["stage3.ebm"]["thresholds"]
    return {"ece_before": c4["calibration"]["ece_before"], "ece_after": c4["calibration"]["ece_after"],
            "threshold_raw": c6["OOF (F2)"]["accepted"],
            "threshold_calibrated": c6["OOF, calibrated space"]["accepted"],
            "nested_by_fold": [c6[f"OOF fold-nested, fold {k}"]["accepted"] for k in range(N_FOLDS)]}


# ======================================================================
# 3. the training-side feature block
# ======================================================================
BLOCK_FEATURES = BASELINE_FEATURES + BLOCK_EXTRA


def training_block(X: pd.DataFrame, y: pd.Series, *, features: Sequence[str] = BLOCK_FEATURES, verbose=False):
    """
    ``Stage3Block.build`` on the training rows with the production ``EBMFeaturePipeline`` (selecting
    ``features``), refitted inside every outer and inner fold. The builder requires a test frame; a one-row
    placeholder taken from the training rows stands in for it and is never scored or reported.
    """
    from shared.stage3.data import Stage3Block
    from shared.stage3.feature_engineering import EBMFeaturePipeline
    feats = list(features)
    return Stage3Block.build(X, y, X.iloc[:1], y.iloc[:1], lambda: EBMFeaturePipeline(features=feats),
                             feature_fit_scope="per_fold", n_tune_folds=N_FOLDS, n_eval_folds=2,
                             random_state=SEED, stratify=True, class_weight="balanced", clean_inf=True,
                             impute_missing=True, expected_features=None, verbose=verbose)


def _fold_fp(fold_id: np.ndarray) -> str:
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    return fold_fingerprint(np.asarray(fold_id))


def block_checks(block, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """The block's rows, folds and frames; decay_x_density = JED x NSD in every frame."""
    out = [
        _check("the block holds exactly the training rows, in order", block.X_train.index.equals(X.index)
               and np.array_equal(block.y_train.to_numpy(), y.to_numpy()), len(block.X_train)),
        _check("the outer plan is the registered fold plan", _fold_fp(block.tune_plan.fold_id) == FOLD_FINGERPRINT,
               _fold_fp(block.tune_plan.fold_id)),
        _check("every outer fold has five inner folds over its fit rows",
               len(block.inner_folds) == N_FOLDS and all(len(i) == N_FOLDS for i in block.inner_folds),
               [len(i) for i in block.inner_folds]),
    ]
    worst = 0.0
    frames = [(block.X_train, "all rows")] + [(f.X_tr, "fit") for f in block.tune_folds] + \
        [(f.X_val, "held out") for f in block.tune_folds] + \
        [(g.X_tr, "inner") for i in block.inner_folds for g in i] + [(g.X_val, "inner") for i in block.inner_folds for g in i]
    for frame, _ in frames:
        worst = max(worst, float(np.abs(frame["decay_x_density"] - frame[JED] * frame[NSD]).max()))
    out.append(_check("decay_x_density = joint_economic_decay x neighborhood_subscription_density in every frame",
                      worst == 0.0, f"largest difference {worst:.1e}"))
    held = np.concatenate([f.val_idx for f in block.tune_folds])
    out.append(_check("the outer held-out rows partition the training rows",
                      np.array_equal(np.sort(held), np.arange(len(X))), len(held)))
    out.append(_check("no inner frame touches its outer fold's held-out rows",
                      all(not np.intersect1d(np.r_[g.train_idx, g.val_idx], block.tune_folds[k].val_idx).size
                          for k, i in enumerate(block.inner_folds) for g in i), "inner frames"))
    return pd.DataFrame(out)


# ======================================================================
# 4. the past-only density
# ======================================================================
def past_density(ctx_macro: np.ndarray, ctx_y: np.ndarray, ctx_day: np.ndarray, q_macro: np.ndarray,
                 q_day: np.ndarray, *, bandwidth: float = KDE_BANDWIDTH) -> Dict[str, np.ndarray]:
    """
    For each query row on day d: the Gaussian kernel average of the context's labels over the context rows
    contacted before d, in coordinates standardized on those rows (the production kernel and scaler). NaN
    on cold-start days and while the history holds fewer than ``MIN_HISTORY_ROWS`` rows or
    ``MIN_HISTORY_POSITIVES`` positives. Returns the values and each query's history rows and positives.
    """
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler
    ctx_macro, q_macro = np.asarray(ctx_macro, dtype=float), np.asarray(q_macro, dtype=float)
    ctx_y, ctx_day, q_day = np.asarray(ctx_y, dtype=float), np.asarray(ctx_day), np.asarray(q_day)
    order = np.argsort(ctx_day, kind="mergesort")
    sd, sm, sy = ctx_day[order], ctx_macro[order], ctx_y[order]
    val = np.full(len(q_day), np.nan)
    hist_rows = np.zeros(len(q_day), dtype=np.int64)
    hist_pos = np.zeros(len(q_day), dtype=np.int64)
    for d in np.unique(q_day):
        rows = np.flatnonzero(q_day == d)
        n = int(np.searchsorted(sd, d, side="left"))          # context rows contacted before day d
        npos = int(sy[:n].sum())
        hist_rows[rows], hist_pos[rows] = n, npos
        if d < COLD_START_DAYS or n < MIN_HISTORY_ROWS or npos < MIN_HISTORY_POSITIVES:
            continue
        sc = StandardScaler().fit(sm[:n])
        ref = sc.transform(sm[:n])
        states, inv = np.unique(q_macro[rows], axis=0, return_inverse=True)
        dist = cdist(sc.transform(states), ref, metric="euclidean")
        w = np.exp(-(dist ** 2) / (2 * bandwidth ** 2))
        w /= w.sum(axis=1, keepdims=True) + 1e-10
        val[rows] = (w @ sy[:n])[np.asarray(inv).ravel()]
    return {"value": val, "history_rows": hist_rows, "history_positives": hist_pos}


def _macro(X: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return X[list(MACRO)].to_numpy(dtype=float)[rows]


def _clean_pair(col_fit: np.ndarray, col_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """The production FeatureCleaner on one column: medians from the fit rows, applied to both."""
    from shared.stage3.data import FeatureCleaner
    c = FeatureCleaner(True, True).fit(pd.DataFrame({PAST_FEATURE: col_fit}))
    a = c.transform(pd.DataFrame({PAST_FEATURE: col_fit}))[PAST_FEATURE].to_numpy()
    b = c.transform(pd.DataFrame({PAST_FEATURE: col_val}))[PAST_FEATURE].to_numpy()
    return a, b, float(c.medians_[PAST_FEATURE])


def add_past_density(block, X: pd.DataFrame, y: pd.Series, T: pd.DataFrame) -> Dict[str, Any]:
    """
    The past-only density in every outer and inner context, and ``decay_x_density_past`` = JED x it, cleaned
    with the fit rows' median, added to copies of the block's frames. Also the all-rows context (shapes).
    Returns the augmented frames, per-context counts and the raw values (for checks and plots).
    """
    from copy import copy
    if not T.index.equals(X.index):
        raise ValueError("the time index is not aligned with the training rows")
    day, yv = T["day"].to_numpy(), y.astype(int).to_numpy()
    contexts, outer, inner = [], [], []

    def augment(f, label):
        ctx = f.train_idx
        a = past_density(_macro(X, ctx), yv[ctx], day[ctx], _macro(X, ctx), day[ctx])
        b = past_density(_macro(X, ctx), yv[ctx], day[ctx], _macro(X, f.val_idx), day[f.val_idx])
        pa = f.X_tr[JED].to_numpy() * a["value"]
        pb = f.X_val[JED].to_numpy() * b["value"]
        ca, cb, med = _clean_pair(pa, pb)
        g = copy(f)
        g.X_tr = f.X_tr.assign(**{PAST_FEATURE: ca, PAST_NSD: a["value"]})
        g.X_val = f.X_val.assign(**{PAST_FEATURE: cb, PAST_NSD: b["value"]})
        g.X_test = None
        ok = lambda h: h["history_rows"][~np.isnan(h["value"])]  # noqa: E731
        okp = lambda h: h["history_positives"][~np.isnan(h["value"])]  # noqa: E731
        contexts.append({"context": label, "fit_rows": int(len(ctx)),
                         "fit_missing": int(np.isnan(a["value"]).sum()),
                         "scored_missing": int(np.isnan(b["value"]).sum()), "scored_rows": int(len(f.val_idx)),
                         "first_valued_day": int(day[ctx][~np.isnan(a["value"])].min()),
                         "min_history_rows": int(min(ok(a).min(), ok(b).min())),
                         "min_history_positives": int(min(okp(a).min(), okp(b).min())),
                         "imputed_median": med})
        return g, a, b

    raw = {}
    for f in block.tune_folds:
        g, a, b = augment(f, f"fold {f.k}")
        outer.append(g)
        raw[f.k] = {"fit": a["value"], "held_out": b["value"]}
    for k, fs in enumerate(block.inner_folds):
        inner.append([augment(g, f"fold {k} inner {g.k}")[0] for g in fs])
    every = np.arange(len(X))
    allrows = past_density(_macro(X, every), yv, day, _macro(X, every), day)
    pa = block.X_train[JED].to_numpy() * allrows["value"]
    ca, _, med = _clean_pair(pa, pa)
    contexts.append({"context": "all rows", "fit_rows": len(X), "fit_missing": int(np.isnan(allrows["value"]).sum()),
                     "scored_missing": int(np.isnan(allrows["value"]).sum()), "scored_rows": len(X),
                     "first_valued_day": int(day[~np.isnan(allrows["value"])].min()),
                     "min_history_rows": int(allrows["history_rows"][~np.isnan(allrows["value"])].min()),
                     "min_history_positives": int(allrows["history_positives"][~np.isnan(allrows["value"])].min()),
                     "imputed_median": med})
    X_all = block.X_train.assign(**{PAST_FEATURE: ca, PAST_NSD: allrows["value"]})
    return {"outer": outer, "inner": inner, "X_all": X_all, "contexts": pd.DataFrame(contexts), "raw": raw,
            "all_rows": allrows}


def past_density_checks(block, X: pd.DataFrame, y: pd.Series, T: pd.DataFrame, past: Mapping[str, Any], *,
                        n_days: int = 12) -> pd.DataFrame:
    """
    Verification and isolation of the past-only density, in outer fold 0's context:
    * on ``n_days`` days spread over the timeline, the production IntegralFeatureEngineer fitted on exactly the
      available rows (n_ref = their count) reproduces the values;
    * flipping every held-out label of each outer fold leaves that fold's values unchanged (control: flipping
      a tenth of its fit labels moves them);
    * flipping the labels of every context row contacted on or after a day leaves that day's values unchanged
      (control: flipping earlier rows moves them).
    """
    from shared.stage3 import feature_engineering as S3FE
    day, yv = T["day"].to_numpy(), y.astype(int).to_numpy()
    out = []
    f0 = block.tune_folds[0]
    ctx = f0.train_idx
    vals = past["raw"][0]["held_out"]
    vdays = np.unique(day[f0.val_idx][~np.isnan(vals)])
    pick = vdays[np.linspace(0, len(vdays) - 1, n_days).round().astype(int)]
    worst = 0.0
    for d in pick:
        hist = ctx[day[ctx] < d]
        q = f0.val_idx[day[f0.val_idx] == d]
        eng = S3FE.IntegralFeatureEngineer(n_ref=len(hist)).fit(X.iloc[hist], y.iloc[hist])
        prod = eng.transform(X.iloc[q])[NSD].to_numpy()
        mine = vals[np.searchsorted(f0.val_idx, q)]
        worst = max(worst, float(np.abs(prod - mine).max()))
    out.append(_check(f"on {len(pick)} days the production IntegralFeatureEngineer fitted on exactly the "
                      "available rows gives the same past-only values (fold 0, held-out rows)", worst <= 1e-12,
                      f"largest difference {worst:.1e}"))
    for f in block.tune_folds:
        held = f.val_idx
        flipped = yv.copy()
        flipped[held] = 1 - flipped[held]
        c = f.train_idx
        a = past_density(_macro(X, c), flipped[c], day[c], _macro(X, held), day[held])["value"]
        same = np.array_equal(a, past["raw"][f.k]["held_out"], equal_nan=True)
        out.append(_check(f"fold {f.k}: flipping its {len(held):,} held-out labels leaves its past-only values "
                          "unchanged", same, "identical" if same else "moved"))
        ctrl = yv.copy()
        sel = c[::10]
        ctrl[sel] = 1 - ctrl[sel]
        b = past_density(_macro(X, c), ctrl[c], day[c], _macro(X, held), day[held])["value"]
        moved = int(np.sum(~np.isclose(b, past["raw"][f.k]["held_out"], equal_nan=True)))
        out.append(_check(f"fold {f.k}: control, flipping {len(sel):,} fit labels moves its past-only values",
                          moved > 0, f"{moved:,} values moved"))
    later_ok, earlier_moved = True, 0
    for d in pick[::3]:
        q = f0.val_idx[day[f0.val_idx] == d]
        lab = yv.copy()
        late = ctx[day[ctx] >= d]
        lab[late] = 1 - lab[late]
        a = past_density(_macro(X, ctx), lab[ctx], day[ctx], _macro(X, q), day[q])["value"]
        later_ok &= np.array_equal(a, vals[np.searchsorted(f0.val_idx, q)], equal_nan=True)
        lab2 = yv.copy()
        early = ctx[day[ctx] < d]
        lab2[early] = 1 - lab2[early]
        b = past_density(_macro(X, ctx), lab2[ctx], day[ctx], _macro(X, q), day[q])["value"]
        earlier_moved += int(np.sum(~np.isclose(b, vals[np.searchsorted(f0.val_idx, q)])))
    out.append(_check("flipping the labels of every row contacted on or after a day leaves that day's values "
                      "unchanged (fold 0)", later_ok, "identical" if later_ok else "moved"))
    out.append(_check("control: flipping the labels of earlier rows moves them", earlier_moved > 0,
                      f"{earlier_moved:,} values moved"))
    cx = past["contexts"]
    out.append(_check(f"every value uses at least {MIN_HISTORY_ROWS} history rows and {MIN_HISTORY_POSITIVES} "
                      "positives", bool((cx["min_history_rows"] >= MIN_HISTORY_ROWS).all()
                                        and (cx["min_history_positives"] >= MIN_HISTORY_POSITIVES).all()),
                      f"{int(cx['min_history_rows'].min())} rows, {int(cx['min_history_positives'].min())} positives"))
    out.append(_check(f"no value on the first {COLD_START_DAYS} contact days",
                      bool((cx["first_valued_day"] >= COLD_START_DAYS).all()), int(cx["first_valued_day"].min())))
    return pd.DataFrame(out)


FALLBACK_TRACE = (
    "The past-only candidate's final feature, decay_x_density_past, reaches the EBM in two forms. A row with a "
    "raw past-only density gets joint_economic_decay x that density, computed from labels of its context's rows "
    "contacted on earlier days only. A row without one (the first three contact days, or a history under 500 "
    "rows or 10 positives) gets the production FeatureCleaner's fallback: the median of decay_x_density_past "
    "over the whole fit side of its fold, whose values use labels of contacts up to the fold's last day. The "
    "fallback is therefore not past-only for the row it fills. The trace verifies both forms in the frames the "
    "EBM is fitted on and scores, counts the held-out rows that get the fallback, and flips fit-side labels "
    "contacted on or after the first held-out day with a raw value (a tenth of them, then all). Those contacts "
    "come after every fallback row's day, and no raw value up to that day can use them, so the raw values of "
    "that first valued day are a control that must not move; the question is whether the fallback does.")


def fallback_trace(block, X: pd.DataFrame, y: pd.Series, T: pd.DataFrame, past: Mapping[str, Any], *,
                   seed: int = SEED, share: float = 0.1) -> Dict[str, Any]:
    """See ``FALLBACK_TRACE``. Outer folds; the candidate as registered and fitted is unchanged."""
    day, yv = T["day"].to_numpy(), y.astype(int).to_numpy()
    ctx = past["contexts"].set_index("context")
    folds, sens = [], []
    for f, g in zip(block.tune_folds, past["outer"]):
        k = f.k
        raw_fit, raw_val = past["raw"][k]["fit"], past["raw"][k]["held_out"]
        jed_fit, jed_val = f.X_tr[JED].to_numpy(), f.X_val[JED].to_numpy()
        miss_fit, miss_val = np.isnan(raw_fit), np.isnan(raw_val)
        prod_fit = jed_fit * raw_fit
        fallback = float(np.nanmedian(prod_fit))
        col_fit, col_val = g.X_tr[PAST_FEATURE].to_numpy(), g.X_val[PAST_FEATURE].to_numpy()
        valued = np.sort(prod_fit[~miss_fit])
        n = len(valued)
        middle = valued[[(n - 1) // 2, n // 2]]
        fit_day, val_day = day[f.train_idx], day[f.val_idx]
        mid_days = fit_day[np.isin(prod_fit, middle)]
        fb_days = val_day[miss_val]
        folds.append({
            "fold": k, "held_out_rows": int(len(f.val_idx)), "held_out_raw": int((~miss_val).sum()),
            "held_out_fallback": int(miss_val.sum()),
            "held_out_fallback_days": None if not len(fb_days) else [int(fb_days.min()), int(fb_days.max())],
            "fit_rows": int(len(f.train_idx)), "fit_fallback": int(miss_fit.sum()), "fallback_value": fallback,
            "cleaner_median": float(ctx.loc[f"fold {k}", "imputed_median"]),
            "fallback_is_cleaner_median": bool(np.isclose(fallback, float(ctx.loc[f"fold {k}", "imputed_median"]),
                                                          rtol=1e-12, atol=0.0)),
            "ebm_column_is_fallback": bool(np.all(col_val[miss_val] == fallback) and np.all(col_fit[miss_fit] == fallback)),
            "ebm_column_is_raw_product": bool(np.array_equal(col_val[~miss_val], (jed_val * raw_val)[~miss_val])
                                              and np.array_equal(col_fit[~miss_fit], prod_fit[~miss_fit])),
            "median_row_days": [int(mid_days.min()), int(mid_days.max())],
            "latest_fit_day_in_median": int(fit_day[~miss_fit].max()),
        })
        if not len(fb_days):
            continue
        last = int(fb_days.max())
        first_valued = int(val_day[~miss_val].min()) if (~miss_val).any() else last + 1
        later = np.flatnonzero(fit_day >= first_valued)            # after every fallback row's day, and in no
        q = np.flatnonzero(val_day == first_valued)                # raw value up to the first valued day
        rng = np.random.default_rng(seed + k)
        for label, flip in ((f"{share:.0%} of later fit labels", rng.choice(later, int(round(share * len(later))),
                                                                          replace=False)),
                            ("every later fit label", later)):
            yy = yv[f.train_idx].copy()
            yy[flip] = 1 - yy[flip]
            a2 = past_density(_macro(X, f.train_idx), yy, fit_day, _macro(X, f.train_idx), fit_day)
            fb2 = float(np.nanmedian(jed_fit * a2["value"]))
            same = True
            if len(q):
                b1 = past_density(_macro(X, f.train_idx), yv[f.train_idx], fit_day, _macro(X, f.val_idx[q]), val_day[q])
                b2 = past_density(_macro(X, f.train_idx), yy, fit_day, _macro(X, f.val_idx[q]), val_day[q])
                same = bool(np.array_equal(b1["value"], b2["value"], equal_nan=True))
            sens.append({"fold": k, "perturbation": label, "last_fallback_day": last,
                         "flipped_from_day": first_valued, "labels_flipped": int(len(flip)),
                         "fallback_before": fallback, "fallback_after": fb2, "fallback_moved": bool(fb2 != fallback),
                         "relative_change": (fb2 - fallback) / fallback if fallback else None,
                         "control_day": first_valued, "control_rows": int(len(q)),
                         "control_raw_values_unchanged": same})
    per_fold = pd.DataFrame(folds)
    inner = ctx.loc[[c for c in ctx.index if " inner " in c], "scored_missing"]
    totals = {"held_out_rows": int(per_fold["held_out_rows"].sum()),
              "held_out_raw": int(per_fold["held_out_raw"].sum()),
              "held_out_fallback": int(per_fold["held_out_fallback"].sum()),
              "held_out_fallback_share": float(per_fold["held_out_fallback"].sum() / per_fold["held_out_rows"].sum()),
              "fit_fallback_by_fold": [int(v) for v in per_fold["fit_fallback"]],
              "inner_held_out_fallback": int(inner.sum())}
    return {"description": FALLBACK_TRACE, "per_fold": per_fold, "sensitivity": pd.DataFrame(sens), "totals": totals}


def fallback_checks(trace: Mapping[str, Any], past: Mapping[str, Any]) -> pd.DataFrame:
    """The trace's own correctness (not its finding): both forms verified in the EBM's frames, counts reconcile."""
    pf, ctx = trace["per_fold"], past["contexts"].set_index("context")
    outer_missing = int(sum(ctx.loc[f"fold {k}", "scored_missing"] for k in pf["fold"]))
    out = [_check("every fallback row's EBM column is its fold's fit-side median (held-out and fit rows)",
                  bool(pf["ebm_column_is_fallback"].all()), pf["held_out_fallback"].tolist()),
           _check("every other row's EBM column is joint_economic_decay x its raw past-only density",
                  bool(pf["ebm_column_is_raw_product"].all()), pf["held_out_raw"].tolist()),
           _check("the fallback is the production cleaner's fitted median in every fold",
                  bool(pf["fallback_is_cleaner_median"].all()), pf["fallback_value"].round(9).tolist()),
           _check("held-out fallback rows = the contexts' missing held-out rows; raw + fallback = every row",
                  trace["totals"]["held_out_fallback"] == outer_missing
                  and trace["totals"]["held_out_raw"] + trace["totals"]["held_out_fallback"]
                  == trace["totals"]["held_out_rows"], (trace["totals"]["held_out_fallback"], outer_missing))]
    sv = trace["sensitivity"]
    if len(sv):
        out.append(_check("control: flipping fit labels after the last fallback day leaves the first valued day's "
                          "raw values unchanged", bool(sv["control_raw_values_unchanged"].all()),
                          int(sv["control_rows"].sum())))
    return pd.DataFrame(out)


# ======================================================================
# 5. configurations and fits
# ======================================================================
def configurations() -> Dict[str, Dict[str, Any]]:
    """
    The baseline and each candidate: features, pairs resolved by name (define_ebm_interactions on its
    columns), retained and lost pairs, and the research questions mapped to it (identical configurations
    are fitted once).
    """
    from glass_pipeline.ebm.interactions import define_ebm_interactions
    specs = {"baseline": {"question": "frozen PR 39 Stage 3 EBM", "remove": [], "add": []}}
    specs.update({k: REGISTRATION["candidates"][k] for k in CANDIDATES})
    out: Dict[str, Dict[str, Any]] = {}
    seen: Dict[Tuple, str] = {}
    for name, sp in specs.items():
        feats = [f for f in BASELINE_FEATURES if f not in sp["remove"]]
        for f, new in zip(sp["remove"], sp["add"]):
            feats.insert(BASELINE_FEATURES.index(f), new)
        feats += [n for n in sp["add"][len(sp["remove"]):] if n not in feats]
        feats = list(dict.fromkeys(feats))              # an added feature already present is kept once
        pairs = [tuple(p) for p in define_ebm_interactions(pd.DataFrame(columns=feats))]
        key = (tuple(sorted(feats)), tuple(sorted(pairs)))
        if key in seen:
            out[seen[key]]["questions"].append(sp["question"])
            out[name] = {"same_as": seen[key], "questions": [sp["question"]]}
            continue
        seen[key] = name
        lost = [p for p in BASELINE_PAIRS if p not in pairs]
        out[name] = {"features": feats, "pairs": pairs, "retained_pairs": [p for p in BASELINE_PAIRS if p in pairs],
                     "lost_pairs": lost, "n_features": len(feats), "n_pairs": len(pairs),
                     "questions": [sp["question"]], "removed": list(sp["remove"]), "added": list(sp["add"])}
    return out


def configuration_checks(configs: Mapping[str, Any]) -> pd.DataFrame:
    """Each candidate's lost pairs are the registered ones; no candidate inherits a pair."""
    out = []
    for name in CANDIDATES:
        c = configs[name]
        if "same_as" in c:
            continue
        want = REGISTRATION["candidates"][name]["expected_lost_pairs"]
        got = [_pair_name(p) for p in c["lost_pairs"]]
        out.append(_check(f"{name}: loses exactly {want}", got == want, got))
        out.append(_check(f"{name}: every retained pair is an accepted pair, by name",
                          all(p in BASELINE_PAIRS for p in c["pairs"]), c["pairs"]))
    out.append(_check("the baseline keeps all three pairs and 13 features",
                      configs["baseline"]["n_pairs"] == 3 and configs["baseline"]["n_features"] == 13,
                      configs["baseline"]["n_features"]))
    return pd.DataFrame(out)


class _Recording:
    """An estimator wrapper that keeps the fitted models (OOFGenerator drops its local references)."""

    def __init__(self, inner):
        self.inner, self.models = inner, []

    def fit(self, params, X, y, sample_weight=None):
        m = self.inner.fit(params, X, y, sample_weight=sample_weight)
        self.models.append(m)
        return m

    def positive_proba(self, model, X):
        return self.inner.positive_proba(model, X)


def _subset(frames, feats):
    from copy import copy
    out = []
    for f in frames:
        g = copy(f)
        g.X_tr, g.X_val, g.X_test = f.X_tr[list(feats)], f.X_val[list(feats)], None
        out.append(g)
    return out


def expected_terms(config: Mapping[str, Any]) -> List[str]:
    return list(config["features"]) + [f"{a} & {b}" for a, b in config["pairs"]]


def _terms_match(got: Sequence[str], want: Sequence[str]) -> bool:
    """Main terms in feature order, then exactly the configuration's pairs (interpret orders pair terms itself)."""
    n = sum(" & " not in t for t in want)
    return list(got[:n]) == list(want[:n]) and sorted(got[n:]) == sorted(want[n:]) and len(got) == len(want)


def _terms_ok(got: Sequence[str], config: Mapping[str, Any]) -> bool:
    """A fixed-pair configuration: ``_terms_match``. A procedural one (``budget``): its features as main terms,
    in order, then exactly ``budget`` pair terms among them (which pairs is the selector's result)."""
    if config.get("budget") is None:
        return _terms_match(got, expected_terms(config))
    feats, n = list(config["features"]), len(config["features"])
    pairs = [t.split(" & ") for t in got[n:]]
    return (list(got[:n]) == feats and len(pairs) == int(config["budget"])
            and all(len(q) == 2 and set(q) <= set(feats) for q in pairs))


def fit_configuration(config: Mapping[str, Any], block, frames: Mapping[str, Any], *, verbose: bool = False,
                      factory=None) -> Dict[str, Any]:
    """
    One configuration through the Stage 3 protocol on the training side: outer out-of-fold and inner
    out-of-fold scores (OOFGenerator), the fold-nested isotonic calibration (Stage3Calibrator), the
    thresholds (F2ThresholdSelector: global raw, global calibrated, fold-nested), and an all-rows fit for the
    shapes. ``frames`` holds the outer and inner frames (with the past-only column) and the all-rows frame.
    ``factory`` replaces the fixed-pair ``EBMFactory`` (Step 5B passes one that selects its pairs inside every
    fit); every other part of the protocol is unchanged.
    """
    import time
    from glass_pipeline.ebm.config import EBMStage3Config
    from glass_pipeline.ebm.estimator import EBMFactory
    from shared.stage3.calibration import Stage3Calibrator
    from shared.stage3.evaluation import OOFGenerator
    from shared.stage3.thresholding import F2ThresholdSelector
    t0 = time.perf_counter()
    cfg = EBMStage3Config(expected_features=list(BASELINE_FEATURES))
    feats = list(config["features"])
    fac = factory if factory is not None else EBMFactory(interactions=list(config["pairs"]),
                                                         random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    rec = _Recording(fac)
    plan, y = block.tune_plan, block.y_train
    oof = OOFGenerator(rec, plan).generate(_subset(frames["outer"], feats), PARAMS, score_test=False)
    inner, sizes = OOFGenerator(fac, plan).generate_inner([_subset(i, feats) for i in frames["inner"]], PARAMS)
    cal = Stage3Calibrator(cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins, cfg.force_calibration,
                           cfg.random_state).fit(oof.proba, y, plan, inner)
    sel = F2ThresholdSelector(beta=cfg.beta, low=cfg.threshold_low, high=cfg.threshold_high,
                              steps=cfg.threshold_steps)
    choice = sel.select(y, oof.proba, selected_on="out-of-fold training predictions", keep_sweep=False)
    choice_cal = sel.select(y, cal.oof_calibrated_, selected_on="out-of-fold training predictions (calibrated)",
                            keep_sweep=False)
    nested = sel.select_nested(y, oof.proba, plan, inner)
    X_all = frames["X_all"][feats]
    full = fac.fit(PARAMS, X_all, y, sample_weight=block.weights.vector)
    want = expected_terms(config) if config.get("budget") is None else list(config["features"])
    terms = [list(m.term_names_) for m in rec.models] + [list(full.term_names_)]
    if verbose:
        print(f"  {config['questions'][0]}: {len(rec.models)} outer + {sum(map(len, sizes))} inner + 1 all-rows "
              f"fits, {time.perf_counter() - t0:.0f} s")
    return {"proba": oof.proba, "calibrated": cal.oof_calibrated_, "inner": inner, "inner_sizes": sizes,
            "nested": nested, "threshold_raw": float(choice.threshold),
            "threshold_calibrated": float(choice_cal.threshold), "calibration": cal.report_.to_dict(),
            "fold_applied": list(cal.fold_applied_), "applied": bool(cal.applied_),
            "models": rec.models, "full": full, "terms": terms, "terms_ok": all(_terms_ok(t, config) for t in terms),
            "expected_terms": want, "fit_rows": oof.fit_sizes}


def frames_for_fits(block, past: Mapping[str, Any]) -> Dict[str, Any]:
    """The outer and inner frames with the past-only column, and the all-rows frame."""
    return {"outer": past["outer"], "inner": past["inner"], "X_all": past["X_all"]}


def fit_all(configs: Mapping[str, Any], block, frames, *, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """Every distinct configuration, baseline first."""
    return {name: fit_configuration(c, block, frames, verbose=verbose)
            for name, c in configs.items() if "same_as" not in c}


def fit_checks(configs, fits) -> pd.DataFrame:
    """Fitted terms, fit sizes and calibration gates for every configuration."""
    out = []
    for name, r in fits.items():
        out.append(_check(f"{name}: every fitted EBM's terms are its features plus its pairs, by name",
                          r["terms_ok"], len(r["expected_terms"])))
        out.append(_check(f"{name}: five outer fits on 26,360 rows and 25 inner fits",
                          r["fit_rows"] == [26360] * N_FOLDS and sum(map(len, r["inner_sizes"])) == 25,
                          r["fit_rows"]))
        out.append(_check(f"{name}: out-of-fold scores for every training row, in [0, 1]",
                          bool(np.isfinite(r["proba"]).all() and (r["proba"] >= 0).all() and (r["proba"] <= 1).all()),
                          len(r["proba"])))
    return pd.DataFrame(out)


def isolation_checks(fit: Mapping[str, Any], block) -> pd.DataFrame:
    """
    For each outer fold k, with the fitted scores fixed: flipping every held-out label of k leaves k's
    calibrated values and its nested threshold unchanged (they come from inner scores of k's fit rows), and,
    as a control, moves another fold's. The models themselves see only their frames' fit rows (OOFGenerator
    checks the frames against the plan; Stage3Block builds each frame's features on its fit rows).
    """
    from glass_pipeline.ebm.config import EBMStage3Config
    from shared.stage3.calibration import Stage3Calibrator
    from shared.stage3.thresholding import F2ThresholdSelector
    cfg = EBMStage3Config(expected_features=list(BASELINE_FEATURES))
    sel = F2ThresholdSelector(beta=cfg.beta, low=cfg.threshold_low, high=cfg.threshold_high,
                              steps=cfg.threshold_steps)
    plan, y = block.tune_plan, block.y_train
    out = []
    for k in range(N_FOLDS):
        held = plan.fold_id == k
        yf = y.copy()
        yf.iloc[np.flatnonzero(held)] = 1 - yf.iloc[np.flatnonzero(held)]
        cal = Stage3Calibrator(cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins, cfg.force_calibration,
                               cfg.random_state).fit(fit["proba"], yf, plan, fit["inner"])
        nested = sel.select_nested(yf, fit["proba"], plan, fit["inner"])
        same = (np.array_equal(cal.oof_calibrated_[held], fit["calibrated"][held])
                and float(nested["by_fold"][k]) == float(fit["nested"]["by_fold"][k]))
        moved = int((cal.oof_calibrated_[~held] != fit["calibrated"][~held]).sum())
        out.append(_check(f"fold {k}: flipping its {int(held.sum()):,} held-out labels leaves its calibrated values "
                          "and nested threshold unchanged", same, "identical" if same else "moved"))
        out.append(_check(f"fold {k}: control, the same flip moves other folds' calibrated values", moved > 0,
                          f"{moved:,} values moved"))
    return pd.DataFrame(out)


# ======================================================================
# 6. metrics and paired differences
# ======================================================================
def _metrics(y, p, thr, space, decision=None) -> Dict[str, Any]:
    from shared.stage3.evaluation import Stage3Metrics
    return Stage3Metrics.compute(y, p, thr, probability_space=space, ece_bins=10, decision=decision).to_dict()


def operating_point_table(fits: Mapping[str, Any], y: pd.Series) -> pd.DataFrame:
    """Per configuration: the three established operating points (Stage3Metrics), labelled raw or calibrated."""
    yv = y.astype(int).to_numpy()
    rows = []
    for name, r in fits.items():
        nested_dec = (r["proba"] >= r["nested"]["per_row"]).astype(int)
        for point, p, thr, space, dec in (
                ("fold-nested raw thresholds (primary)", r["proba"], r["nested"]["mean"], "raw", nested_dec),
                ("global raw F2 threshold (in-selection)", r["proba"], r["threshold_raw"], "raw", None),
                ("global calibrated threshold (in-selection)", r["calibrated"], r["threshold_calibrated"],
                 "calibrated", None)):
            m = _metrics(yv, p, thr, space, dec)
            rows.append({"configuration": name, "operating_point": point, "probabilities": space, **m})
    return pd.DataFrame(rows)


def _fold_values(y: np.ndarray, p: np.ndarray, cal: np.ndarray, dec: np.ndarray, fold_id: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score
    from shared.metrics import calculate_ece
    rows = []
    for k in range(N_FOLDS):
        m = fold_id == k
        rows.append({"fold": k, "roc_auc": roc_auc_score(y[m], p[m]), "pr_auc": average_precision_score(y[m], p[m]),
                     "brier_raw": brier_score_loss(y[m], p[m]), "brier_calibrated": brier_score_loss(y[m], cal[m]),
                     "ece_raw": calculate_ece(y[m], p[m], 10), "ece_calibrated": calculate_ece(y[m], cal[m], 10),
                     "nested_f2": fbeta_score(y[m], dec[m], beta=2.0, zero_division=0.0)})
    return pd.DataFrame(rows)


def paired_differences(fits: Mapping[str, Any], y: pd.Series, fold_id: np.ndarray) -> Dict[str, pd.DataFrame]:
    """
    Each candidate against the baseline on the same rows: pooled values and differences (ROC-AUC with
    DeLong's paired standard error), per-fold values and differences, and decision disagreement.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score
    from shared.metrics import calculate_ece
    from feature_research import temporal_validity as TV
    yv = y.astype(int).to_numpy()
    base = fits["baseline"]
    bdec = (base["proba"] >= base["nested"]["per_row"]).astype(int)
    pooled, per_fold, decisions = [], [], []
    for name, r in fits.items():
        dec = (r["proba"] >= r["nested"]["per_row"]).astype(int)
        row = {"configuration": name, "roc_auc": None, "pr_auc": average_precision_score(yv, r["proba"]),
               "brier_raw": brier_score_loss(yv, r["proba"]), "brier_calibrated": brier_score_loss(yv, r["calibrated"]),
               "ece_raw": calculate_ece(yv, r["proba"], 10), "ece_calibrated": calculate_ece(yv, r["calibrated"], 10),
               "nested_f2": fbeta_score(yv, dec, beta=2.0, zero_division=0.0)}
        dl = TV.delong_paired(yv, base["proba"], r["proba"])
        row["roc_auc"] = dl["auc_b"]
        if name != "baseline":
            row.update({"d_roc_auc": dl["diff"], "d_roc_auc_se": dl["se"],
                        "d_roc_auc_low": dl["ci_low"], "d_roc_auc_high": dl["ci_high"],
                        "d_roc_auc_p": dl["p_value"]})
        pooled.append(row)
        fv = _fold_values(yv, r["proba"], r["calibrated"], dec, fold_id).assign(configuration=name)
        per_fold.append(fv)
        if name != "baseline":
            decisions.append({"configuration": name, "differ": int((dec != bdec).sum()),
                              "flag_now_not_baseline": int(((dec == 1) & (bdec == 0)).sum()),
                              "baseline_flag_not_now": int(((dec == 0) & (bdec == 1)).sum()),
                              "flagged": int(dec.sum()), "baseline_flagged": int(bdec.sum())})
    pooled = pd.DataFrame(pooled)
    b = pooled.set_index("configuration").loc["baseline"]
    for col in ("pr_auc", "brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated", "nested_f2"):
        pooled[f"d_{col}"] = pooled[col] - b[col]
    per_fold = pd.concat(per_fold, ignore_index=True)
    bf = per_fold[per_fold["configuration"] == "baseline"].set_index("fold")
    spread = []
    for name in fits:
        if name == "baseline":
            continue
        cf = per_fold[per_fold["configuration"] == name].set_index("fold")
        for col in ("roc_auc", "pr_auc", "brier_calibrated", "ece_calibrated", "nested_f2"):
            d = cf[col] - bf[col]
            spread.append({"configuration": name, "metric": col, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                           "min": float(d.min()), "max": float(d.max()), "folds_better": int(
                               (d > 0).sum() if col in ("roc_auc", "pr_auc", "nested_f2") else (d < 0).sum()),
                           "by_fold": [float(v) for v in d]})
    return {"pooled": pooled, "per_fold": per_fold, "fold_differences": pd.DataFrame(spread),
            "decisions": pd.DataFrame(decisions)}


def parity(base: Mapping[str, Any], accepted: Mapping[str, Any], views: Optional[Mapping[str, Any]] = None
           ) -> pd.DataFrame:
    """The refitted baseline against the PR 39 records and, where available, the accepted EBM row by row."""
    rows = [
        {"quantity": "out-of-fold ECE before calibration (raw)", "accepted": accepted["ece_before"],
         "refit": base["calibration"].get("ece_before")},
        {"quantity": "out-of-fold ECE after calibration", "accepted": accepted["ece_after"],
         "refit": base["calibration"].get("ece_after")},
        {"quantity": "global raw F2 threshold", "accepted": accepted["threshold_raw"], "refit": base["threshold_raw"]},
        {"quantity": "global calibrated threshold", "accepted": accepted["threshold_calibrated"],
         "refit": base["threshold_calibrated"]},
    ] + [{"quantity": f"fold-nested threshold, fold {k}", "accepted": a, "refit": float(b)}
         for k, (a, b) in enumerate(zip(accepted["nested_by_fold"], base["nested"]["by_fold"]))]
    if views and views.get("ebm") is not None:
        from sklearn.metrics import roc_auc_score
        v = views["ebm"]
        y = v["y"]
        for label, mine, theirs in (("raw out-of-fold score", base["proba"], v["raw"]),
                                    ("calibrated out-of-fold probability", base["calibrated"], v["calibrated"])):
            rows.append({"quantity": f"{label}: largest absolute difference (row-aligned)", "accepted": 0.0,
                         "refit": float(np.abs(mine - theirs).max())})
            rows.append({"quantity": f"{label}: Pearson correlation (row-aligned)", "accepted": 1.0,
                         "refit": float(np.corrcoef(mine, theirs)[0, 1])})
        rows.append({"quantity": "out-of-fold ROC-AUC (raw)", "accepted": float(roc_auc_score(y, v["raw"])),
                     "refit": float(roc_auc_score(y, base["proba"]))})
        dec = (base["proba"] >= base["nested"]["per_row"]).astype(int)
        rows.append({"quantity": "nested decisions that differ from the stored ones", "accepted": 0,
                     "refit": int((dec != v["decision"]).sum())})
    out = pd.DataFrame(rows)
    out["difference"] = [None if a is None or b is None else float(b) - float(a)
                         for a, b in zip(out["accepted"], out["refit"])]
    return out


# ======================================================================
# 7. review of the inventory (distributions, correlation, contributions, periods)
# ======================================================================
REVIEW_FEATURES = BASELINE_FEATURES + (JED, PAST_FEATURE)


def distributions(X_all: pd.DataFrame) -> pd.DataFrame:
    """Each feature's distribution on the full-train block (values the all-rows pipeline gives)."""
    rows = []
    for f in REVIEW_FEATURES:
        v = X_all[f].to_numpy(dtype=float)
        q = np.quantile(v, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        rows.append({"feature": f, "distinct": int(len(np.unique(v))), "mean": float(v.mean()),
                     "sd": float(v.std()), "min": q[0], "p01": q[1], "p25": q[2], "median": q[3], "p75": q[4],
                     "p99": q[5], "max": q[6], "share_zero": float((v == 0).mean()),
                     "share_mode": float(pd.Series(v).value_counts(normalize=True).iloc[0])})
    return pd.DataFrame(rows)


def correlations(block, X_all: pd.DataFrame) -> Dict[str, Any]:
    """|Pearson r| among the 13 features on the full-train block and on each outer fit side."""
    feats = list(BASELINE_FEATURES)
    full = X_all[feats].corr()
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = float(full.loc[a, b])
            folds = [float(f.X_tr[[a, b]].corr().iloc[0, 1]) for f in block.tune_folds]
            pairs.append({"a": a, "b": b, "r": r, "abs_r": abs(r), "fold_abs_r_min": min(abs(x) for x in folds),
                          "fold_abs_r_max": max(abs(x) for x in folds)})
    pairs = pd.DataFrame(pairs).sort_values("abs_r", ascending=False, kind="mergesort").reset_index(drop=True)
    extra = {f"{a} ~ {b}": float(X_all[[a, b]].corr().iloc[0, 1]) for a, b in
             (("decay_x_density", JED), ("decay_x_density", PAST_FEATURE), (JED, PAST_FEATURE),
              (NSD, PAST_NSD))}
    return {"matrix": full, "pairs": pairs, "extra": extra}


def correlation_checks(corr: Mapping[str, Any]) -> pd.DataFrame:
    p = corr["pairs"]
    over = p[(p["abs_r"] >= CORRELATION_THRESHOLD) | (p["fold_abs_r_max"] >= CORRELATION_THRESHOLD)]
    named = [tuple(sorted(x)) for x in zip(over["a"], over["b"])]
    ok = named == [tuple(sorted(CORRELATED_PAIR))] and bool((over["fold_abs_r_min"] >= CORRELATION_THRESHOLD).all())
    return pd.DataFrame([_check(f"the only pair of the 13 with |r| >= {CORRELATION_THRESHOLD} (full train and every "
                                f"outer fit side) is {CORRELATED_PAIR[0]} / {CORRELATED_PAIR[1]}", ok,
                                [(a, b, round(r, 4)) for a, b, r in zip(over["a"], over["b"], over["abs_r"])])])


def oof_contributions(fit: Mapping[str, Any], frames: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    """Each held-out row's term contributions from its own fold's model (out of fold), in log-odds."""
    feats = list(config["features"])
    parts = []
    for f, m in zip(frames["outer"], fit["models"]):
        c = m.eval_terms(f.X_val[feats])
        parts.append(pd.DataFrame(c, columns=list(m.term_names_), index=f.val_idx))
    return pd.concat(parts).sort_index()


def contribution_review(fits, frames, configs, T: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For the baseline: term importance (mean |score|) of every fold model and the all-rows fit; the out-of-fold
    mean |contribution| of each term by contact period, relative to its overall mean.
    """
    base, cb = fits["baseline"], configs["baseline"]
    imp = []
    for label, m in [(f"fold {k}", m) for k, m in enumerate(base["models"])] + [("all rows", base["full"])]:
        for t, v in zip(m.term_names_, m.term_importances()):
            imp.append({"model": label, "term": t, "importance": float(v)})
    imp = pd.DataFrame(imp)
    folds = imp[imp["model"] != "all rows"].groupby("term")["importance"]
    summary = pd.DataFrame({"mean": folds.mean(), "sd": folds.std(ddof=1)})
    summary["cv"] = summary["sd"] / summary["mean"]
    summary["all_rows"] = imp[imp["model"] == "all rows"].set_index("term")["importance"]
    summary = summary.sort_values("all_rows", ascending=False).reset_index()
    oof = oof_contributions(base, frames, cb)
    per = T["period"].to_numpy()
    absc = oof.abs()
    by_period = absc.groupby(per).mean() / absc.mean()
    counts = pd.Series(per).value_counts().sort_index()
    return {"importance": imp, "importance_summary": summary, "period_relative": by_period,
            "period_rows": counts}


# ======================================================================
# 8. shapes
# ======================================================================
FEATURE_UNITS = {
    "decay_x_density": "joint decay x neighbourhood subscription rate",
    "decay_x_density_past": "joint decay x past-only neighbourhood subscription rate",
    "joint_economic_decay": "product of three macro decays (0-1)",
    "euribor3m_sigmoid_slope": "normalised sigmoid slope of euribor3m (0-1)",
    "emp_var_rate_sigmoid_slope": "normalised sigmoid slope of emp.var.rate (0-1)",
    "economic_curvature_intensity": "mean |label-rate curvature| of three macro bins",
    "cons.conf.idx": "consumer confidence index",
    "age": "age (years)",
    "dow_month_encoded": "smoothed subscription rate of the day x month cell",
    "default": "credit default flag (encoded)",
    "prior_x_stress": "prior contact x economic stress integral",
    "campaign": "contacts in this campaign",
    "overlap_default_clean": "low-stress zone x no default (0/1)",
    "cpi_high_cellular": "cons.price.idx >= 93.5 and cellular contact (0/1)",
    "behavioral_favorability": "weighted behavioural composite",
}


def feature_grid(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    u = np.unique(v)
    if len(u) <= DISCRETE_MAX_UNIQUE:
        return u
    lo, hi = np.quantile(v, GRID_QUANTILES)
    return np.linspace(lo, hi, GRID_POINTS)


def _term_means(model, X_ref: pd.DataFrame) -> Dict[str, float]:
    c = model.eval_terms(X_ref[list(model.feature_names_in_)])
    return dict(zip(model.term_names_, c.mean(axis=0)))


def _eval_term(model, base_row: pd.DataFrame, updates: Mapping[str, np.ndarray], term: str) -> np.ndarray:
    n = len(next(iter(updates.values())))
    frame = base_row.loc[base_row.index.repeat(n)].reset_index(drop=True)
    for k, v in updates.items():
        frame[k] = v
    i = list(model.term_names_).index(term)
    return model.eval_terms(frame[list(model.feature_names_in_)])[:, i]


def shape_curves(fits, configs, frames) -> pd.DataFrame:
    """
    Every main-effect curve: each configuration's five outer-fold models and its all-rows fit, evaluated on
    the feature's common grid, shifted to mean 0 over the training rows (shift recorded). Log-odds units.
    """
    X_all = frames["X_all"]
    rows = []
    for name, r in fits.items():
        feats = list(configs[name]["features"])
        ref = X_all[feats]
        base_row = ref.iloc[[0]]
        for label, m in [(f"fold {k}", m) for k, m in enumerate(r["models"])] + [("all rows", r["full"])]:
            means = _term_means(m, ref)
            for f in feats:
                g = feature_grid(X_all[f].to_numpy())
                s = _eval_term(m, base_row, {f: g}, f)
                for x, v in zip(g, s):
                    rows.append({"configuration": name, "model": label, "feature": f, "x": float(x),
                                 "score": float(v - means[f]), "shift": float(means[f])})
    return pd.DataFrame(rows)


PAIR_QUANTILE_BINS = 20
PAIR_SPARSE_SHARE = 0.001
#: Values that differ by less than this, relative to max(1, |value|), are equal up to floating-point noise
#: (reductions such as the NSD kernel's row sums and matrix-vector product can differ in the last bits by a
#: row's position in the array and by the platform's BLAS and SIMD paths).
NOISE_REL = 1e-9


def _fmt(x: float, digits: int = 3) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.{digits}g}"


def pair_bins(values: np.ndarray, bins: int = PAIR_QUANTILE_BINS) -> Tuple[np.ndarray, List[str]]:
    """
    Support-defined cells for one axis of a pair plot. A feature with at most ``DISCRETE_MAX_UNIQUE`` distinct
    values (values equal up to floating-point noise, 1e-9 relative, count as one) gets one cell per value.
    Otherwise the cell edges are the distinct ``bins``-quantiles (observed values), each cell ``(e[j-1], e[j]]``;
    the minimum has a cell of its own only when it is a mass point (it is also the first quantile), else it joins
    the first cell. No cell is empty. Returns each row's cell and the cells' labels (the observed value, or the
    observed min-max of the cell).
    """
    v = np.asarray(values, dtype=float)
    u = np.unique(v)
    last = np.r_[np.diff(u) > NOISE_REL * np.maximum(1.0, np.abs(u[1:])), True]
    u = u[last]                                       # values equal up to floating-point noise are one value
    if len(u) <= DISCRETE_MAX_UNIQUE:
        edges = u
    else:
        q = np.quantile(v, np.linspace(0, 1, bins + 1), method="inverted_cdf")
        edges = np.unique(q if q[1] == q[0] else q[1:])   # the minimum gets its own cell only if it is a mass point
    idx = np.searchsorted(edges, v, side="left")
    ranges = [(v[idx == j].min(), v[idx == j].max()) for j in range(len(edges))]
    for digits in range(3, 9):                       # enough digits that no two cells share a label
        text = [(_fmt(lo, digits), _fmt(hi, digits)) for lo, hi in ranges]
        labels = [a if a == b else f"{a}-{b}" for a, b in text]
        if len(set(labels)) == len(labels):
            break
    return idx, labels


def fallback_term_effect(fit: Mapping[str, Any], config: Mapping[str, Any], frames: Mapping[str, Any],
                         trace: Mapping[str, Any]) -> pd.DataFrame:
    """
    How far the fallback's dependence on later labels reaches into the fitted candidate: each outer-fold model
    of the past-only candidate, held fixed, scored at its fold's fallback and at each perturbed fallback from
    ``fallback_trace`` (the decay_x_density_past term, log-odds). Every held-out fallback row of the fold gets
    that term score, so the change is what those rows' scores would move by if only the fallback moved.
    """
    feats = list(config["features"])
    pf = trace["per_fold"].set_index("fold")
    sv = trace["sensitivity"]
    rows = []
    for k, m in enumerate(fit["models"]):
        s = sv[sv["fold"] == k]
        if not len(s):
            continue
        before = float(pf.loc[k, "fallback_value"])
        vals = np.r_[before, s["fallback_after"].to_numpy(dtype=float)]
        t = _eval_term(m, frames["outer"][k].X_val[feats].iloc[[0]], {PAST_FEATURE: vals}, PAST_FEATURE)
        for (_, r), after in zip(s.iterrows(), t[1:]):
            rows.append({"fold": k, "perturbation": r["perturbation"],
                         "held_out_fallback_rows": int(pf.loc[k, "held_out_fallback"]),
                         "fallback_before": before, "fallback_after": float(r["fallback_after"]),
                         "term_score_before": float(t[0]), "term_score_after": float(after),
                         "term_score_change": float(after - t[0])})
    return pd.DataFrame(rows)


def pair_surfaces(fits, configs, frames, T: Optional[pd.DataFrame] = None, *, bins: int = PAIR_QUANTILE_BINS
                  ) -> Dict[str, Any]:
    """
    Each retained pair's term score on support-defined cells of its two features (``pair_bins``; the same
    cells in every model). Per model, a cell's value is the mean over the cell's training rows of the pair
    term's contribution, centred so its mean over all training rows is 0 (the registered centring). The
    all-rows fit gives the surface and the five outer-fold models the per-cell fold sd. Cells without rows
    stay empty, so nothing is extrapolated beyond the joint support. The same rows give the joint support and
    the largest contact period's share in each cell.
    """
    X_all = frames["X_all"]
    n = len(X_all)
    per = None if T is None else T["period"].to_numpy()
    cells: Dict[str, Any] = {}
    out = []
    for name in [c for c in ORDER if c in fits] + [c for c in fits if c not in ORDER]:
        r = fits[name]
        feats = list(configs[name]["features"])
        if not configs[name]["pairs"]:
            continue
        contrib = {}
        for label, m in [(f"fold {k}", m) for k, m in enumerate(r["models"])] + [("all rows", r["full"])]:
            c = m.eval_terms(X_all[feats])
            contrib[label] = dict(zip(m.term_names_, (c - c.mean(axis=0)).T))
        for a, b in configs[name]["pairs"]:
            term = f"{a} & {b}"
            if term not in cells:
                ia, la = pair_bins(X_all[a].to_numpy(), bins)
                ib, lb = pair_bins(X_all[b].to_numpy(), bins)
                cell = ia * len(lb) + ib
                H = np.bincount(cell, minlength=len(la) * len(lb)).reshape(len(la), len(lb))
                share = np.full(H.shape, np.nan)
                if per is not None:
                    top = pd.DataFrame({"c": cell, "p": per}).value_counts().groupby(level=0).max()
                    at = top.index.to_numpy()
                    share.flat[at] = top.to_numpy() / H.flat[at]
                cells[term] = (cell, H, la, lb, share)
            cell, H, la, lb, share = cells[term]

            def means(v, cell=cell, H=H):
                s = np.bincount(cell, weights=v, minlength=H.size).reshape(H.shape)
                return np.where(H > 0, s / np.maximum(H, 1), np.nan)

            surf = means(contrib["all rows"][term])
            fold_sd = np.stack([means(contrib[f"fold {k}"][term]) for k in range(N_FOLDS)]).std(axis=0, ddof=1)
            ok = H >= PAIR_SPARSE_SHARE * n
            out.append({"configuration": name, "pair": term, "a": a, "b": b, "labels_a": la, "labels_b": lb,
                        "surface": surf, "fold_sd": fold_sd, "support": H, "period_share": share,
                        "cells": int(H.size), "cells_with_rows": int((H > 0).sum()), "cells_supported": int(ok.sum()),
                        "rows_in_sparse_cells": int(H[(H > 0) & ~ok].sum()),
                        "max_abs": float(np.abs(surf[ok]).max()) if ok.any() else 0.0,
                        "range_supported": float(np.ptp(surf[ok])) if ok.any() else 0.0,
                        "fold_sd_rows": float((fold_sd[H > 0] * H[H > 0]).sum() / H.sum()),
                        "fold_sd_max_supported": float(fold_sd[ok].max()) if ok.any() else 0.0})
    return {"surfaces": out}


def pair_table(pairs: Mapping[str, Any]) -> pd.DataFrame:
    """One row per configuration and retained pair: cells, support, effect size and fold spread."""
    keys = ("configuration", "pair", "cells", "cells_with_rows", "cells_supported", "rows_in_sparse_cells",
            "max_abs", "range_supported", "fold_sd_rows", "fold_sd_max_supported")
    t = pd.DataFrame([{k: s[k] for k in keys} for s in pairs["surfaces"]])
    t["fold_sd_to_effect"] = t["fold_sd_max_supported"] / t["max_abs"].where(t["max_abs"] > 0)
    return t


def _edges(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    if len(g) == 1:
        return np.array([g[0] - 0.5, g[0] + 0.5])
    mid = (g[1:] + g[:-1]) / 2
    return np.r_[g[0] - (mid[0] - g[0]), mid, g[-1] + (g[-1] - mid[-1])]


def support(X_all: pd.DataFrame, T: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Training rows per bin of the plotted range (or per value), the largest period's share, sparse flags."""
    v = X_all[feature].to_numpy(dtype=float)
    per = T["period"].to_numpy()
    year = T["year"].to_numpy()
    g = feature_grid(v)
    if len(g) <= DISCRETE_MAX_UNIQUE and len(np.unique(v)) <= DISCRETE_MAX_UNIQUE:
        edges = _edges(g)
    else:
        edges = np.linspace(g.min(), g.max(), SUPPORT_BINS + 1)
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    inside = (v >= edges[0]) & (v <= edges[-1])
    rows = []
    for b in range(len(edges) - 1):
        m = (idx == b) & inside
        n = int(m.sum())
        pc = pd.Series(per[m]).value_counts()
        yc = pd.Series(year[m]).value_counts()
        rows.append({"feature": feature, "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "rows": n,
                     "share": n / len(v), "sparse": n < SPARSE_SHARE * len(v),
                     "largest_period": None if pc.empty else str(pc.index[0]),
                     "largest_period_share": None if pc.empty else float(pc.iloc[0] / n),
                     "single_period": bool(n and pc.iloc[0] / n > SINGLE_PERIOD_SHARE),
                     **{f"rows_{y}": int(yc.get(y, 0)) for y in (2008, 2009, 2010)}})
    out = pd.DataFrame(rows)
    out.attrs["outside_range"] = int((~inside).sum())
    return out


def support_tables(X_all: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([support(X_all, T, f) for f in REVIEW_FEATURES], ignore_index=True)


def shape_summary(curves: pd.DataFrame, supports: pd.DataFrame) -> pd.DataFrame:
    """
    Per configuration and feature, on grid points inside non-sparse support bins: the all-rows curve's range
    (max - min), the mean fold spread (sd across the five fold curves), the number of slope sign changes of
    the all-rows curve (a shape-complexity count), and the share of the curve's range that lies in sparse or
    single-period bins.
    """
    rows = []
    for (name, f), g in curves.groupby(["configuration", "feature"], sort=False):
        s = supports[supports["feature"] == f]
        grid = np.sort(g["x"].unique())
        bins = np.clip(np.searchsorted(s["lo"].to_numpy(), grid, side="right") - 1, 0, len(s) - 1)
        dense = ~s["sparse"].to_numpy()[bins]
        single = s["single_period"].to_numpy()[bins]
        allc = g[g["model"] == "all rows"].set_index("x")["score"].reindex(grid).to_numpy()
        fold = np.stack([g[g["model"] == f"fold {k}"].set_index("x")["score"].reindex(grid).to_numpy()
                         for k in range(N_FOLDS)])
        sd = fold.std(axis=0, ddof=1)
        d = allc[dense]
        slope = np.sign(np.diff(d)) if len(d) > 1 else np.array([])
        slope = slope[slope != 0]
        full_range = float(np.nanmax(allc) - np.nanmin(allc))
        rows.append({"configuration": name, "feature": f, "grid_points": int(len(grid)),
                     "dense_points": int(dense.sum()),
                     "range_dense": float(d.max() - d.min()) if len(d) else None,
                     "range_all": full_range, "fold_sd_dense": float(sd[dense].mean()) if dense.any() else None,
                     "sign_changes_dense": int((np.diff(slope) != 0).sum()) if len(slope) > 1 else 0,
                     "single_period_points": int((single & dense).sum())})
    return pd.DataFrame(rows)


def within_day_spread(X_all: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """
    For each review feature, its largest range within one contact day and whether that is floating-point noise
    (at most NOISE_REL x max(1, largest |value|)). A feature constant within every day up to that noise is
    period-level. Exact equality is not required: decay_x_density's NSD factor comes from a kernel row sum and a
    matrix-vector product whose last bits can differ between rows of the same day on some platforms (the
    accepted Windows run), although every row of a day has the same inputs and reference.
    """
    day = T["day"].to_numpy()
    rows = []
    for f in [f for f in REVIEW_FEATURES if f in X_all.columns]:
        v = X_all[f].astype(float)
        g = v.groupby(day)
        spread = float((g.max() - g.min()).max())
        scale = max(1.0, float(v.abs().max()))
        rows.append({"feature": f, "largest_within_day_range": spread, "noise_bound": NOISE_REL * scale,
                     "exactly_constant": spread == 0.0, "period_level": spread <= NOISE_REL * scale})
    return pd.DataFrame(rows)


def period_level_features(X_all: pd.DataFrame, T: pd.DataFrame) -> List[str]:
    """Review features constant within every contact day up to floating-point noise (``within_day_spread``)."""
    t = within_day_spread(X_all, T)
    return t.loc[t["period_level"], "feature"].tolist()


CENTRING_CHECK = (
    "Post-registration check, added after the shape figures were inspected; it changes no fit, metric, "
    "threshold or ranking. The fold curves of the correlated slopes look vertically offset. Under the "
    "registered centring every curve has mean 0 over all training rows. Here each fold curve is re-centred "
    "over the rows in well-supported bins only (support bins holding at least SPARSE_SHARE of the rows, inside "
    "the plotted range); each fold's shift is reported, with their largest absolute value (how far any fold "
    "curve moves) and their range across folds (how far the curves move relative to one another). Two "
    "centring-free quantities are taken per fold: the curve's amplitude (max - min) "
    "over those rows, and its regime step (mean contribution of the 2009-2010 rows minus that of the 2008 "
    "rows). The step is also taken for the sum of the two slopes, for the sum of every period-level main term "
    "(features constant within a contact day up to floating-point noise), for that sum without the decay slot "
    "(decay_x_density or its replacement) and for the whole additive score, to separate disagreement "
    "about the regime effect from disagreement about how it is split between correlated terms. With five folds "
    "the correlation of the two slopes' steps is descriptive only (no test).")


def centring_check(fits, configs, frames, supports: pd.DataFrame, T: pd.DataFrame,
                   features: Sequence[str] = CORRELATED_PAIR) -> Dict[str, Any]:
    """See ``CENTRING_CHECK``. Uses the five outer-fold models of each configuration; no labels."""
    X_all = frames["X_all"]
    year = T["year"].to_numpy()
    late, early = year > 2008, year == 2008
    spread_table = within_day_spread(X_all, T)
    level = spread_table.loc[spread_table["period_level"], "feature"].tolist()
    slot = {"decay_x_density", JED, PAST_FEATURE}

    def step(v):
        return float(v[late].mean() - v[early].mean())

    cen, steps, trade = [], [], []
    for name in [c for c in ORDER if c in fits]:
        feats = list(configs[name]["features"])
        C = []
        for m in fits[name]["models"]:
            c = m.eval_terms(X_all[feats])
            C.append(dict(zip(m.term_names_, (c - c.mean(axis=0)).T)))
        for f in [f for f in features if f in feats]:
            s = supports[supports["feature"] == f]
            edges = np.r_[s["lo"].to_numpy(), s["hi"].to_numpy()[-1]]
            v = X_all[f].to_numpy(dtype=float)
            b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(s) - 1)
            dense = ~s["sparse"].to_numpy()[b] & (v >= edges[0]) & (v <= edges[-1])
            M = np.stack([c[f] for c in C])
            shift = M[:, dense].mean(axis=1)
            amp = [float(np.ptp(M[k][dense])) for k in range(len(C))]
            st = [step(M[k]) for k in range(len(C))]
            cen.append({"configuration": name, "feature": f, "dense_row_share": float(dense.mean()),
                        "fold_sd_registered": float(M[:, dense].std(axis=0, ddof=1).mean()),
                        "fold_sd_recentred": float((M - shift[:, None])[:, dense].std(axis=0, ddof=1).mean()),
                        "recentring_shift_max_abs": float(np.abs(shift).max()),
                        "recentring_shift_range": float(np.ptp(shift)),
                        "recentring_shift_by_fold": [float(v) for v in shift],
                        "amplitude_by_fold": amp, "amplitude_sd": float(np.std(amp, ddof=1)),
                        "step_by_fold": st, "step_sd": float(np.std(st, ddof=1))})
        both = all(f in feats for f in features)
        for k, c in enumerate(C):
            row = {"configuration": name, "fold": k}
            for f in features:
                row[f] = step(c[f]) if f in c else None
            row["both slopes"] = step(c[features[0]] + c[features[1]]) if both else None
            row["period-level terms"] = step(np.sum([c[f] for f in feats if f in level], axis=0))
            row["period-level terms except the decay slot"] = step(
                np.sum([c[f] for f in feats if f in level and f not in slot], axis=0))
            row["additive score"] = step(np.sum(list(c.values()), axis=0))
            steps.append(row)
        if both:
            A = np.stack([c[features[0]] for c in C])
            B = np.stack([c[features[1]] for c in C])
            va, vb, vs = (float(x.var(axis=0, ddof=1).mean()) for x in (A, B, A + B))
            sa = [step(a) for a in A]
            sb = [step(x) for x in B]
            trade.append({"configuration": name, "fold_sd_first": va ** 0.5, "fold_sd_second": vb ** 0.5,
                          "fold_sd_sum": vs ** 0.5, "fold_deviation_corr": (vs - va - vb) / (2 * (va * vb) ** 0.5),
                          "step_corr": float(np.corrcoef(sa, sb)[0, 1])})
    steps = pd.DataFrame(steps)
    cols = [*features, "both slopes", "period-level terms", "period-level terms except the decay slot",
            "additive score"]
    spread = steps.groupby("configuration", sort=False)[cols].agg(lambda s: s.std(ddof=1) if s.notna().all() else None)
    spread = spread.reset_index().assign(statistic="sd across folds")
    rng = steps.groupby("configuration", sort=False)[cols].agg(lambda s: s.max() - s.min() if s.notna().all() else None)
    rng = rng.reset_index().assign(statistic="range across folds")
    return {"description": CENTRING_CHECK, "period_level_features": level, "within_day_spread": spread_table,
            "centring": pd.DataFrame(cen),
            "steps": steps, "step_spread": pd.concat([spread, rng], ignore_index=True),
            "trade_off": pd.DataFrame(trade)}


# ======================================================================
# 9. complementarity with Stage 1 (accepted artifacts, where available)
# ======================================================================
VIEW_ROLES = {"stage1.lr": "extract_lr_view", "stage2.glass_router": "extract_router_view",
              "stage3.ebm": "extract_ebm_view"}
#: Set to 1 for the acceptance run: every accepted view must then be present and pass, so a completed run has
#: computed and reconciled the complementarity tables (Cell 57). Unset or 0: missing artifacts are reported.
REQUIRE_VIEWS_ENV = "PR42_REQUIRE_ACCEPTED_VIEWS"


class AcceptedViewError(RuntimeError):
    """An accepted artifact is present but fails its pin or boundary check, or the views are required and an
    artifact is missing; the message names the role and the reason."""


def views_required(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether this run must have all three accepted views (``PR42_REQUIRE_ACCEPTED_VIEWS`` = 1)."""
    value = str((os.environ if environ is None else environ).get(REQUIRE_VIEWS_ENV, "")).strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{REQUIRE_VIEWS_ENV} must be 1 (require the accepted views) or 0 / unset, not {value!r}")


def accepted_views(ref, root: Path = BANK_ROOT, *, require: bool = False,
                   y: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    The accepted GLASS-arm Stage 1 LR, Stage 2 router and Stage 3 EBM out-of-fold outputs, each read through
    its PR 38 boundary extractor at the SHA-256 the PR 39 Commit 4 manifest pins. Called before any fit.

    * An artifact whose file does not exist is reported as not available locally (``models/`` is Git-ignored):
      the sandbox path, where complementarity is not computed.
    * An artifact that exists but fails its pin (``pinned_path``) or its boundary extraction, or whose labels
      are not ``y`` in row order, raises ``AcceptedViewError`` naming the role and the reason: a present
      artifact is never silently skipped.
    * ``require=True`` (the acceptance run, ``views_required()``) raises unless all three views passed.

    Only the training-side vectors this step uses are kept: Stage 1 flags at ``lr.in_stage_f2_cv``, the
    router's abstentions, and the EBM's raw and calibrated scores and its ``ebm.nested_f2`` decisions.
    """
    from feature_research.frozen_references import PINS_FILE
    from shared.pinned_artifacts import pinned_path, read_pins
    from shared.stage2.diagnostics import boundary
    from shared.stage2.diagnostics.operating_points import resolve_operating_point
    root = Path(root)
    pins_file = root / PINS_FILE
    try:
        pins = read_pins(pins_file)
    except Exception as exc:
        raise AcceptedViewError(f"pins file {PINS_FILE}: {type(exc).__name__}: {exc}") from exc
    status: Dict[str, str] = {}
    out: Dict[str, Any] = {}
    for role, fn in VIEW_ROLES.items():
        entry = pins["files"][role]
        if not (root / entry["path"]).is_file():
            status[role] = "not available locally"
            continue
        try:
            verified = pinned_path(role, pins_file=pins_file, root=root)
        except Exception as exc:
            raise AcceptedViewError(f"{role}: {entry['path']} is present but fails its pin check: "
                                    f"{type(exc).__name__}: {exc}") from exc
        try:
            view = getattr(boundary, fn)(str(verified), ref, expected_arm="glass", expected_sha256=entry["sha256"])
            vy = np.asarray(view.y).astype(int)
            if role == "stage1.lr":
                fl = resolve_operating_point(view, "lr.in_stage_f2_cv")
                kept = {"flags": np.asarray(fl.flags, dtype=bool), "y": vy, "threshold": fl.threshold,
                        "space": fl.score_space, "field": fl.field, "selection": fl.selection,
                        "disclosure": fl.disclosure}
            elif role == "stage2.glass_router":
                kept = {"abstain": np.asarray(view.abstain, dtype=bool), "y": vy}
            else:
                fl = resolve_operating_point(view, "ebm.nested_f2")
                kept = {"raw": np.asarray(view.scores["raw_model_score"].values, dtype=float),
                        "calibrated": np.asarray(view.scores["calibrated_probability"].values, dtype=float),
                        "decision": np.asarray(fl.flags).astype(int), "y": vy, "field": fl.field,
                        "selection": fl.selection}
        except Exception as exc:
            raise AcceptedViewError(f"{role}: {entry['path']} passes its pin but fails its boundary check: "
                                    f"{type(exc).__name__}: {exc}") from exc
        if y is not None and not np.array_equal(vy, np.asarray(y).astype(int)):
            raise AcceptedViewError(f"{role}: {entry['path']} passes its pin but fails its boundary check: its "
                                    f"{len(vy):,} labels are not the {len(y):,} training rows' labels in row order")
        del view
        status[role] = "passed"
        out[{"stage1.lr": "lr", "stage2.glass_router": "router", "stage3.ebm": "ebm"}[role]] = {
            **kept, "path": entry["path"], "sha256": entry["sha256"]}
    missing = sorted(r for r, v in status.items() if v != "passed")
    if require and missing:
        raise AcceptedViewError(f"the accepted views are required ({REQUIRE_VIEWS_ENV}=1) but not available "
                                f"locally: {', '.join(missing)}")
    out["status"] = status
    out["required"] = bool(require)
    out["available"] = not missing
    return out


def _four(y: np.ndarray, s1: np.ndarray, ebm: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    c1, c3 = (s1.astype(int) == y), (ebm.astype(int) == y)
    m = np.asarray(mask, dtype=bool)
    return {"rows": int(m.sum()), "both_correct": int((c1 & c3 & m).sum()),
            "stage1_only_correct": int((c1 & ~c3 & m).sum()), "ebm_only_correct": int((~c1 & c3 & m).sum()),
            "both_wrong": int((~c1 & ~c3 & m).sum()), "positives": int(y[m].sum())}


def complementarity(y: np.ndarray, s1_flags: np.ndarray, abstain: np.ndarray,
                    decisions: Mapping[str, np.ndarray], *, reference: str = "baseline") -> Dict[str, pd.DataFrame]:
    """
    The four paired counts of Stage 1 and each EBM, for all rows and for Stage 2 abstentions, reconciled to
    their populations; and each EBM against ``reference``: Stage 1 errors it corrects, correct Stage 1
    decisions it overturns, and the net.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    rows = []
    for name, dec in decisions.items():
        for pop, mask in (("all rows", np.ones(n, dtype=bool)), ("Stage 2 abstentions", np.asarray(abstain))):
            c = _four(y, np.asarray(s1_flags), np.asarray(dec), mask)
            c["reconciled"] = (c["both_correct"] + c["stage1_only_correct"] + c["ebm_only_correct"]
                               + c["both_wrong"]) == c["rows"]
            c["net"] = c["ebm_only_correct"] - c["stage1_only_correct"]
            rows.append({"ebm": name, "population": pop, **c})
    table = pd.DataFrame(rows)
    ref = table[table["ebm"] == reference].set_index("population")
    cmp_rows = []
    for _, r in table.iterrows():
        if r["ebm"] == reference:
            continue
        b = ref.loc[r["population"]]
        cmp_rows.append({"ebm": r["ebm"], "population": r["population"], "rows": r["rows"],
                         "corrects_stage1_errors": r["ebm_only_correct"],
                         "reference_corrects": b["ebm_only_correct"],
                         "overturns_correct_stage1": r["stage1_only_correct"],
                         "reference_overturns": b["stage1_only_correct"],
                         "net": r["net"], "reference_net": b["net"], "net_change": r["net"] - b["net"]})
    return {"table": table, "against_reference": pd.DataFrame(cmp_rows)}


def complementarity_checks(comp: Mapping[str, pd.DataFrame], n_rows: int) -> pd.DataFrame:
    t = comp["table"]
    out = [_check("every table adds up to its population", bool(t["reconciled"].all()), len(t)),
           _check("the all-rows tables cover every training row",
                  bool((t[t["population"] == "all rows"]["rows"] == n_rows).all()), n_rows)]
    ab = t[t["population"] == "Stage 2 abstentions"]["rows"].unique()
    out.append(_check("every EBM is compared on the same abstention rows", len(ab) == 1, list(ab)))
    out.append(_check("the abstention population is not empty", bool(len(ab) and ab[0] > 0), list(ab)))
    pops = t.groupby("ebm")["population"].apply(lambda p: sorted(set(p)))
    out.append(_check("every EBM has an all-rows and a Stage 2 abstention table",
                      bool(all(v == ["Stage 2 abstentions", "all rows"] for v in pops)), int(len(pops))))
    return pd.DataFrame(out)


def complementarity_requirement(views: Mapping[str, Any], comp: Optional[Mapping[str, pd.DataFrame]],
                                comp_chk: Optional[pd.DataFrame], expected: Sequence[str]) -> pd.DataFrame:
    """
    When the accepted views are required (the acceptance run), the run must have computed complementarity for
    every expected EBM, with an all-rows and a Stage 2 abstention table each, and every reconciliation check
    must have passed; otherwise one row records that the views were not required and what was available.
    """
    if not views.get("required"):
        return pd.DataFrame([_check(f"accepted views not required ({REQUIRE_VIEWS_ENV} unset): complementarity is "
                                    "computed only when all three are present", True, dict(views["status"]))])
    out = [_check("all three accepted views passed their pin and boundary checks", bool(views.get("available")),
                  dict(views["status"])),
           _check("complementarity was computed", comp is not None, None if comp is None else len(comp["table"])),
           _check("every complementarity check passed",
                  comp_chk is not None and len(comp_chk) > 0 and bool(comp_chk["passed"].all()),
                  None if comp_chk is None else int(len(comp_chk)))]
    t = None if comp is None else comp["table"]
    for e in expected:
        pops = [] if t is None else sorted(set(t.loc[t["ebm"] == e, "population"]))
        out.append(_check(f"{e}: an all-rows and a Stage 2 abstention table, reconciled",
                          pops == ["Stage 2 abstentions", "all rows"]
                          and bool(t.loc[t["ebm"] == e, "reconciled"].all()), pops))
    return pd.DataFrame(out)


# ======================================================================
# 10. nomination
# ======================================================================
#: The interpretability judgement for each candidate, written from the shapes, the pair cells and the
#: post-registration centring check (Cells 54-56) and the review document; the registered rule requires a
#: documented gain. Written after the results were seen (a judgement, not a registered threshold).
INTERPRETATION: Dict[str, Dict[str, Any]] = {
    "drop_first": {
        "gain": True,
        "rationale": (
            "Removes euribor3m_sigmoid_slope, the less stable of the two correlated regime slopes (r 0.725): its "
            "fold models disagree on the size of its 2009-10 vs 2008 step (sd 0.12 log-odds, 0.06 to 0.32) and 8 "
            "of its 12 well-supported grid points are single-period; in the baseline the two slopes split that "
            "step differently from fold to fold (their five per-fold steps correlate at -0.47: descriptive, "
            "consistent with the two slopes sharing one regime signal). Also drops "
            "euribor3m_sigmoid_slope x economic_curvature_intensity, at most about 0.05 log-odds in any "
            "well-supported cell (0.048 in the sandbox fit, 0.051 in the Windows fit) while its fold sd reaches "
            "about 0.2 there. 12 features, 2 pairs, one empty slot. Cost inside the "
            "tolerances: ROC-AUC -0.0004 (95% CI -0.0009 to 0.0002), PR-AUC -0.0018, nested F2 +0.0004, "
            "calibrated ECE 0.0018. It leaves decay_x_density's fold-0 excursion in place (period-level step "
            "sd 0.75 vs 0.63)."),
    },
    "drop_second": {
        "gain": False,
        "rationale": (
            "Removes emp_var_rate_sigmoid_slope, the more stable of the two slopes (step sd 0.077 vs 0.119), and "
            "with it campaign x emp_var_rate_sigmoid_slope, the TIER 1 pair and the only retained pair with a "
            "substantive well-supported effect (0.81 to 0.87 log-odds across its supported cells in the two "
            "platforms' fits, fold sd at most about 0.06). "
            "The kept euribor slope keeps its fold disagreement (step sd 0.11). ROC-AUC -0.0013 (95% CI -0.0023 "
            "to -0.0003, p 0.011): inside the tolerance, but the only interval that excludes zero."),
    },
    "jed_alone": {
        "gain": True,
        "rationale": (
            "Replaces the label-derived decay_x_density (joint decay x NSD) with its label-free macro factor "
            "joint_economic_decay (r 0.9935). The slot's fold sd over well-supported values falls from 0.513 to "
            "0.085 log-odds, and the period-level regime step agrees across folds (sd 0.064 vs 0.632; the "
            "baseline's fold 0 put 2.75 log-odds on it against 1.1 to 1.6, through decay_x_density alone). Loses "
            "prior_x_stress x decay_x_density, at most about 0.07 log-odds in any well-supported cell (0.068 "
            "sandbox, 0.072 Windows). 13 features, 2 pairs, one empty "
            "slot. Performance unchanged: ROC-AUC +0.0001 (95% CI -0.0004 to 0.0005), PR-AUC +0.0006, nested F2 "
            "+0.0010. Trade-offs: calibrated ECE 0.0027 vs 0.0019 (gate 0.05), and emp_var_rate_sigmoid_slope's "
            "top value (0.99, 519 rows) now scores -0.61 in the all-rows fit (fold models -0.13 to -0.44), part "
            "of the late-2010 decline decay_x_density carried."),
    },
    "past_density": {
        "gain": False,
        "rationale": (
            "More stable than the baseline (slot fold sd 0.172 vs 0.513; period-level step sd 0.071) and inside "
            "the tolerances (ROC-AUC +0.0005, 95% CI -0.0005 to 0.0014; PR-AUC +0.0017; nested F2 -0.0011 with "
            "the widest fold spread, sd 0.0087), but not easier to explain: its shape changes direction 5 times "
            "over well-supported values (joint_economic_decay 2), and the feature keeps a label-dependent kernel "
            "with a history state and a lag-0 outcome assumption. It is also not fully past-only: the 917 held-out "
            "rows contacted on days 0-2 take a fallback computed over the whole fit fold, and flipping a tenth of "
            "the later fit labels raises that fallback 3.1- to 3.6-fold and, with the fitted fold models held "
            "fixed, those rows' term score by 0.07 to 0.16 log-odds. joint_economic_decay reaches the same "
            "stability with no labels. Its case is temporal "
            "validity, which PR 49 tests once its fallback uses earlier data only; it is not carried into Step 5B "
            "reselection."),
    },
}

#: Short interpretation of each changed feature and each lost pair (Cells 55-56; review document).
FEATURE_NOTES: Dict[str, str] = {
    "euribor3m_sigmoid_slope": (
        "A two-regime step: -0.12 to -0.26 for the 2008 rows (slope below 0.04, 21,971 rows), rising from -0.20 "
        "at 0.53 to about +0.1 between 0.73 and 0.93 and +0.33 at the top value (2009-10). The fold models agree "
        "on direction, not on the step's size (0.06 to 0.32 log-odds), and disagree most at the two ends (-0.35 "
        "to +0.05 at the lowest value, +0.11 to +0.56 at the top). Re-centring over well-supported values moves "
        "a fold curve by at most 0.0085 (the five shifts span 0.0135) and leaves their spread unchanged, so the "
        "apparent offset is a shape difference, not a centring artefact. 8 of 12 well-supported points are "
        "single-period."),
    "emp_var_rate_sigmoid_slope": (
        "Values up to 0.058 score -0.1 to -0.2 and the rest +0.1 to +0.3. Fold 0 gives a larger step (0.34 vs "
        "0.17 in the other folds) in the baseline and again in drop_first (0.37 vs 0.20), so removing the euribor "
        "slope does not remove it. In jed_alone the top value (0.99) turns negative. Most stable in past_density "
        "(fold sd 0.016)."),
    "decay_x_density": (
        "Rises from about -0.1 at zero density (2008 rows, 21,991 in the lowest grid bin) to a plateau of +0.6 "
        "to +1.3 and falls below 0 above 0.13 (sparse, 2010). Fold 0 departs from the others everywhere: -2.2 at "
        "zero and +1.1 to +2.0 above it, against -0.7 to -0.2 and -0.1 to +1.1. Fold sd over well-supported "
        "values 0.513, the largest of any term; this is the fold-0 regime-step excursion."),
    "joint_economic_decay": (
        "A monotone rise in three levels: -0.13 to -0.04 for the 2008 rows (values up to 0.036), +0.47 to +0.66 "
        "for 0.07 to 0.09, and +0.75 to +1.09 for 0.32 to 0.39. The fold curves stay close (fold sd 0.085). "
        "Label-free: its only inputs are the three macro series."),
    "decay_x_density_past": (
        "Not monotone: -0.58 at zero past density, +0.76 just above it, down to +0.18 at 0.008 to 0.010, then "
        "rising to +1.25 at 0.116 and falling to +0.92; 5 direction changes over well-supported values and the "
        "folds disagree most at low values (0.01 to 0.82). The 917 held-out rows contacted on days 0-2 have no "
        "raw value and carry their fold's fit-side median (about 2.8e-06), a fallback that depends on later "
        "labels."),
    "euribor3m_sigmoid_slope x economic_curvature_intensity": (
        "Lost in drop_first. Within its 44 well-supported cells the effect is at most about 0.05 log-odds "
        "(0.048 in the sandbox fit, 0.051 in the Windows fit) while the fold sd reaches about 0.2: noise-level."),
    "prior_x_stress x decay_x_density": (
        "Lost in jed_alone and past_density. At most about 0.07 log-odds in a well-supported cell (0.068 in the "
        "sandbox fit, 0.072 in the Windows fit); in the sandbox fit the largest is -0.07 at prior_x_stress 0.42 "
        "to 0.67 with decay_x_density 0.0074 to 0.0075 (577 rows)."),
    "campaign x emp_var_rate_sigmoid_slope": (
        "Lost in drop_second; kept by the other candidates with the same pattern. Repeated contacts (campaign "
        "6 and above) at emp.var.rate slopes of 0.487 (October-November 2008) and 0.559 (March-May 2009 and "
        "2010) score from about -0.4 down to -0.67 (sandbox fit) or -0.62 (Windows fit) log-odds beyond the main "
        "effects; at higher slopes such cells are sparse. Fold sd at most about 0.06 in well-supported cells."),
}


def nominations(diff: Mapping[str, pd.DataFrame], comp: Optional[Mapping[str, pd.DataFrame]] = None
                ) -> pd.DataFrame:
    """The registered Step 5B and cascade nomination rules, per candidate."""
    pooled = diff["pooled"].set_index("configuration")
    b = pooled.loc["baseline"]
    rows = []
    for name in pooled.index:
        if name == "baseline":
            continue
        r = pooled.loc[name]
        crit = {"d_roc_auc_ok": bool(r["d_roc_auc"] >= TOL["roc_auc"]),
                "d_roc_auc_lower_ok": bool(r["d_roc_auc_low"] >= TOL["roc_auc_lower"]),
                "d_pr_auc_ok": bool(r["d_pr_auc"] >= TOL["pr_auc"]),
                "d_nested_f2_ok": bool(r["d_nested_f2"] >= TOL["nested_f2"]),
                "calibrated_ece_ok": bool(r["ece_calibrated"] <= TOL["ece_gate"]),
                "calibrated_brier_ok": bool(r["brier_calibrated"] <= b["brier_calibrated"] + TOL["calibrated_brier"])}
        interp = INTERPRETATION.get(name, {})
        crit["interpretability_gain"] = bool(interp.get("gain", False))
        perf = all(v for k, v in crit.items() if k != "interpretability_gain")
        row = {"configuration": name, **crit, "within_tolerances": perf,
               "step5b": "recommended" if perf and crit["interpretability_gain"] else "not recommended",
               "rationale": interp.get("rationale", "")}
        if comp is not None:
            a = comp["against_reference"]
            a = a[(a["ebm"] == name) & (a["population"] == "Stage 2 abstentions")]
            if len(a):
                a = a.iloc[0]
                need = a["reference_net"] + TOL["net_abstain_share"] * a["rows"]
                row["cascade"] = ("nominated" if perf and a["net"] >= need else "not nominated")
                row["abstention_net"], row["abstention_net_needed"] = int(a["net"]), float(need)
        else:
            row["cascade"] = "not evaluated (accepted artifacts not available locally)"
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# 11. figures
# ======================================================================
ORDER = ("baseline",) + CANDIDATES
# fixed categorical order (one hue per configuration, never cycled); this order passes the lightness, chroma,
# colour-vision-deficiency and normal-vision separation checks on a light surface. Three hues sit under 3:1
# contrast, so every panel names its configuration in text as well.
COLORS = {"baseline": "#2a78d6", "drop_first": "#eb6834", "drop_second": "#1baf7a", "jed_alone": "#eda100",
          "past_density": "#e87ba4"}
SHORT = {"baseline": "baseline", "drop_first": "drop\neuribor3m slope", "drop_second": "drop\nemp.var.rate slope",
         "jed_alone": "joint_economic\n_decay alone", "past_density": "past-only\ndensity"}
# diverging (blue <-> red, neutral grey midpoint) for signed term scores; one-hue sequential ramps for magnitudes
_DIVERGING = ("#184f95", "#6da7ec", "#f0efec", "#ef8a89", "#a32b2a")
_SEQ_ROWS = ("#f0efec", "#9a998f", "#1a1a19")
_SEQ_SD = ("#f3f1fd", "#9085e9", "#2a1f78")
_SEQ_PERIOD = ("#fdefe8", "#eb6834", "#7a2908")
LABELS = {"baseline": "baseline (frozen 13, 3 pairs)", "drop_first": "drop euribor3m slope",
          "drop_second": "drop emp.var.rate slope", "jed_alone": "decay_x_density -> joint_economic_decay",
          "past_density": "decay_x_density -> past-only density"}


def _support_panel(ax, sup: pd.DataFrame, discrete: bool, x_of_bin) -> None:
    left = sup["lo"].to_numpy()
    width = (sup["hi"] - sup["lo"]).to_numpy()
    bottom = np.zeros(len(sup))
    for yr, col in ((2008, "#90a4ae"), (2009, "#546e7a"), (2010, "#263238")):
        h = sup[f"rows_{yr}"].to_numpy()
        ax.bar(left, h, width=width, bottom=bottom, align="edge", color=col, label=str(yr), linewidth=0)
        bottom = bottom + h
    for _, r in sup[sup["single_period"]].iterrows():
        ax.bar(r["lo"], r["rows"], width=r["hi"] - r["lo"], align="edge", fill=False, hatch="///",
               edgecolor="#ff8f00", linewidth=0)
    ax.set_ylabel("training rows")
    ax.set_yscale("symlog", linthresh=10)


def feature_figure(feature: str, curves: pd.DataFrame, supports: pd.DataFrame,
                   configs_present: Optional[Sequence[str]] = None):
    """
    One feature across every configuration that has it: per column, the five fold curves (thin) and the
    all-rows curve (thick) on matched axes, sparse bins shaded; below, training rows per bin stacked by
    year, single-period bins hatched.
    """
    import matplotlib.pyplot as plt
    have = [c for c in ORDER if c in set(curves.loc[curves["feature"] == feature, "configuration"])]
    have = [c for c in have if configs_present is None or c in configs_present]
    sup = supports[supports["feature"] == feature]
    g = curves[curves["feature"] == feature]
    discrete = len(g["x"].unique()) <= DISCRETE_MAX_UNIQUE
    lo, hi = g["score"].min(), g["score"].max()
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    fig, axes = plt.subplots(2, len(have), figsize=(3.6 * len(have) + 0.8, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for j, name in enumerate(have):
        ax, axs = axes[0, j], axes[1, j]
        for _, r in sup[sup["sparse"]].iterrows():
            ax.axvspan(r["lo"], r["hi"], color="#eeeeee", zorder=0)
        for k in range(N_FOLDS):
            c = g[(g["configuration"] == name) & (g["model"] == f"fold {k}")].sort_values("x")
            ax.plot(c["x"], c["score"], color=COLORS[name], alpha=0.35, linewidth=0.9,
                    marker="o" if discrete else None, markersize=3, label="fold models" if k == 0 else None)
        c = g[(g["configuration"] == name) & (g["model"] == "all rows")].sort_values("x")
        ax.plot(c["x"], c["score"], color=COLORS[name], linewidth=2.2, marker="o" if discrete else None,
                markersize=4, label="all-rows fit")
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(LABELS[name], fontsize=9)
        if j == 0:
            ax.set_ylabel("additive term score (log-odds)")
            ax.legend(fontsize=7, loc="best")
        _support_panel(axs, sup, discrete, None)
        if j == 0:
            axs.legend(fontsize=7, loc="upper right", title="contact year", title_fontsize=7)
        axs.set_xlabel(FEATURE_UNITS.get(feature, feature), fontsize=8)
    fig.suptitle(f"{feature}: main-effect shape (centred on the training rows) and support; grey = sparse "
                 f"(<{SPARSE_SHARE:.0%} of rows), hatched = one contact period holds >{SINGLE_PERIOD_SHARE:.0%}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def decay_slot_figure(curves: pd.DataFrame, supports: pd.DataFrame):
    """The decay slot across configurations: decay_x_density, joint_economic_decay, decay_x_density_past."""
    import matplotlib.pyplot as plt
    slot = (("baseline", "decay_x_density"), ("jed_alone", JED), ("past_density", PAST_FEATURE))
    g = curves[curves["model"] == "all rows"]
    lo = min(g[(g["configuration"] == c) & (g["feature"] == f)]["score"].min() for c, f in slot)
    hi = max(g[(g["configuration"] == c) & (g["feature"] == f)]["score"].max() for c, f in slot)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), gridspec_kw={"height_ratios": [2.2, 1]})
    for j, (name, f) in enumerate(slot):
        ax = axes[0, j]
        sup = supports[supports["feature"] == f]
        for _, r in sup[sup["sparse"]].iterrows():
            ax.axvspan(r["lo"], r["hi"], color="#eeeeee", zorder=0)
        for k in range(N_FOLDS):
            c = curves[(curves["configuration"] == name) & (curves["feature"] == f)
                       & (curves["model"] == f"fold {k}")].sort_values("x")
            ax.plot(c["x"], c["score"], color=COLORS[name], alpha=0.35, linewidth=0.9)
        c = g[(g["configuration"] == name) & (g["feature"] == f)].sort_values("x")
        ax.plot(c["x"], c["score"], color=COLORS[name], linewidth=2.2)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(f"{LABELS[name]}\n{f}", fontsize=9)
        if j == 0:
            ax.set_ylabel("additive term score (log-odds)")
        _support_panel(axes[1, j], sup, False, None)
        axes[1, j].set_xlabel(FEATURE_UNITS.get(f, f), fontsize=8)
        axes[1, j].sharex(ax)
    fig.suptitle("The decay slot: the accepted feature and its two replacements (matched vertical axis)", fontsize=10)
    fig.tight_layout()
    return fig


FIGURE_DPI = 90


def show(fig, dpi: int = FIGURE_DPI) -> None:
    """
    Put a figure into the notebook's output as a PNG and close it. Importing glass_pipeline.glass_arbiter.tracer
    selects matplotlib's Agg backend, so a displayed Figure would otherwise be stored as its text repr only; the
    PNG is written explicitly so the executed copy holds the picture whatever the backend.
    """
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="png"))


def _cell_axes(ax, la, lb, xlabel=None, ylabel=None) -> None:
    def ticks(labels):
        step = 1 if len(labels) <= 12 else 2
        return list(range(0, len(labels), step)), labels[::step]
    xt, xl = ticks(la)
    yt, yl = ticks(lb)
    ax.set_xticks(xt, xl, rotation=90, fontsize=6)
    ax.set_yticks(yt, yl, fontsize=6)
    ax.set_xlim(-0.5, len(la) - 0.5)
    ax.set_ylim(-0.5, len(lb) - 0.5)
    ax.patch.set_facecolor("#ffffff")
    ax.patch.set_hatch("////")
    ax.patch.set_edgecolor("#d9d8d2")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def pair_figure(pair: str, surfaces: Sequence[Mapping[str, Any]], *, order: Sequence[str] = None,
                labels: Optional[Mapping[str, str]] = None):
    """
    One retained pair across the configurations that keep it, on the same support-defined cells (matched
    axes). Top row: joint support, then each model's all-rows term score on one symmetric log-odds scale.
    Bottom row: the largest contact period's share per cell, then each model's per-cell sd across its five
    outer-fold models on one shared scale. Hatched = no training rows; x = sparse cell.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
    order, labels = (order or ORDER), (labels or LABELS)
    have = [s for name in order for s in surfaces if s["pair"] == pair and s["configuration"] == name]
    s0 = have[0]
    H = s0["support"].astype(float)
    la, lb = s0["labels_a"], s0["labels_b"]
    n = H.sum()
    ok = H >= PAIR_SPARSE_SHARE * n
    sparse = (H > 0) & ~ok
    sx, sy = np.nonzero(sparse)
    vmax = max(s["max_abs"] for s in have) or 1.0
    sdmax = max((s["fold_sd_max_supported"] for s in have if s.get("fold_sd_max_supported") is not None),
                default=0.0) or 1.0
    div = LinearSegmentedColormap.from_list("score", _DIVERGING)
    ncol = len(have) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(max(2.75 * ncol + 1.2, 9.0), 6.6), squeeze=False, sharex=True,
                             sharey=True, constrained_layout=True)
    annotate = H.size <= 120

    def mesh(ax, Z, **kw):
        return ax.pcolormesh(np.arange(len(la) + 1) - 0.5, np.arange(len(lb) + 1) - 0.5,
                             np.ma.masked_invalid(Z.T), shading="flat", edgecolors="#fcfcfb", linewidth=0.4, **kw)

    def mark_sparse(ax):
        if len(sx):
            ax.scatter(sx, sy, marker="x", s=10, linewidths=0.6, color="#3d3d3a", zorder=3)

    ax = axes[0, 0]
    im_n = mesh(ax, np.where(H > 0, H, np.nan), cmap=LinearSegmentedColormap.from_list("rows", _SEQ_ROWS),
                norm=LogNorm(vmin=1, vmax=max(H.max(), 10)))
    if annotate:
        for i, j in zip(*np.nonzero(H)):
            v = H[i, j]
            ax.text(i, j, f"{v / 1000:.1f}k" if v >= 1000 else f"{int(v)}", ha="center", va="center", fontsize=5,
                    color="#ffffff" if v >= 300 else "#1a1a19")
    ax.set_title("joint support\n(training rows per cell)", fontsize=8)
    _cell_axes(ax, la, lb, ylabel=s0["b"])
    fig.colorbar(im_n, ax=ax, location="bottom", shrink=0.85, pad=0.02).ax.tick_params(labelsize=6)
    ax = axes[1, 0]
    im_p = mesh(ax, s0["period_share"], cmap=LinearSegmentedColormap.from_list("period", _SEQ_PERIOD), vmin=0, vmax=1)
    if annotate:
        for i, j in zip(*np.nonzero(np.nan_to_num(s0["period_share"]) > SINGLE_PERIOD_SHARE)):
            ax.text(i, j, "1p", ha="center", va="center", fontsize=5, color="#ffffff")
    ax.set_title(f"largest contact period's share\n(1p = over {SINGLE_PERIOD_SHARE:.0%})", fontsize=8)
    _cell_axes(ax, la, lb, xlabel=s0["a"], ylabel=s0["b"])
    fig.colorbar(im_p, ax=ax, location="bottom", shrink=0.85, pad=0.02).ax.tick_params(labelsize=6)
    im_s = im_d = None
    for j, s in enumerate(have, start=1):
        ax = axes[0, j]
        im_s = mesh(ax, s["surface"], cmap=div, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
        mark_sparse(ax)
        ax.set_title(f"{labels[s['configuration']]}\nterm score, all-rows fit", fontsize=8)
        _cell_axes(ax, la, lb)
        ax = axes[1, j]
        nf = s.get("fold_models")
        if nf is not None and nf < 2:
            # a fold sd needs at least two selecting folds: say so instead of drawing an empty (or zero) map
            ax.text((len(la) - 1) / 2, (len(lb) - 1) / 2, f"fold sd unavailable:\n{nf} of {N_FOLDS} folds\nselect "
                    "this pair", ha="center", va="center", fontsize=7.5, color="#3d3d3a")
            ax.set_xlim(-0.5, len(la) - 0.5)
            ax.set_ylim(-0.5, len(lb) - 0.5)
            ax.set_title(f"{labels[s['configuration']]}\nfold sd unavailable ({nf} of {N_FOLDS} folds)", fontsize=8)
        else:
            im_d = mesh(ax, s["fold_sd"], cmap=LinearSegmentedColormap.from_list("fold_sd", _SEQ_SD), vmin=0,
                        vmax=sdmax)
            mark_sparse(ax)
            ax.set_title(f"{labels[s['configuration']]}\nsd across the {s.get('fold_models', 5)} fold models"
                         + ("" if nf is None else f" ({nf} of {N_FOLDS} folds select it)"), fontsize=8)
        _cell_axes(ax, la, lb, xlabel=s0["a"])
    c = fig.colorbar(im_s, ax=axes[0, 1:], location="right", shrink=0.9, pad=0.01)
    c.set_label("pair term score (log-odds; mean 0 over training rows)", fontsize=7)
    c.ax.tick_params(labelsize=6)
    if im_d is not None:
        c = fig.colorbar(im_d, ax=axes[1, 1:], location="right", shrink=0.9, pad=0.01)
        c.set_label("fold sd of the cell score (log-odds)", fontsize=7)
        c.ax.tick_params(labelsize=6)
    fig.suptitle(f"{pair.replace(' & ', ' x ')}: support-defined cells, the same in every panel; cell value = mean "
                 f"term contribution of the cell's training rows\nhatched = no training rows; x = sparse cell "
                 f"(<{PAIR_SPARSE_SHARE:.1%} of rows); colour scales span the well-supported cells", fontsize=8.5)
    return fig


def regime_step_figure(check: Mapping[str, Any]):
    """Per configuration and outer fold: the 2009-10 minus 2008 step of each slope, of every period-level
    main term together, and of the whole additive score (log-odds). Each panel has its own vertical axis."""
    import matplotlib.pyplot as plt
    steps = check["steps"]
    panels = [*CORRELATED_PAIR, "period-level terms", "period-level terms except the decay slot", "additive score"]
    titles = {CORRELATED_PAIR[0]: "euribor3m_sigmoid_slope term", CORRELATED_PAIR[1]: "emp_var_rate_sigmoid_slope term",
              "period-level terms": "all period-level main terms",
              "period-level terms except the decay slot": "period-level terms\nwithout the decay slot",
              "additive score": "whole additive score"}
    names = [c for c in ORDER if c in set(steps["configuration"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.9), constrained_layout=True)
    for ax, col in zip(axes, panels):
        for i, name in enumerate(names):
            g = steps[steps["configuration"] == name]
            if g[col].isna().all():
                ax.text(i, 0.5, "not in\nmodel", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=6.5, color="#6b6a63")
                continue
            xs = i + np.linspace(-0.18, 0.18, len(g))
            ax.scatter(xs, g[col], s=34, color=COLORS[name], edgecolors="#fcfcfb", linewidths=1.0, zorder=3)
            for x, v, k in zip(xs, g[col], g["fold"]):
                ax.annotate(str(k), (x, v), xytext=(4, 0), textcoords="offset points", fontsize=5.5,
                            color="#3d3d3a", va="center")
        ax.set_xticks(range(len(names)), [SHORT[n] for n in names], fontsize=6, rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_title(titles[col], fontsize=8.5)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("2009-10 minus 2008 mean contribution\n(log-odds; numbers = outer fold)", fontsize=7.5)
    fig.suptitle("Post-registration centring check: how much regime step each outer-fold model gives each term "
                 "(centring-free)", fontsize=9)
    return fig


def period_figure(review: Mapping[str, Any]):
    """Out-of-fold mean |term contribution| by contact period relative to its overall mean (baseline)."""
    import matplotlib.pyplot as plt
    rel = review["period_relative"].T
    fig, ax = plt.subplots(figsize=(13, 0.35 * len(rel) + 1.6))
    im = ax.imshow(np.log2(rel.to_numpy()), aspect="auto", cmap="PuOr_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(rel)), rel.index, fontsize=7)
    ax.set_xticks(range(rel.shape[1]), [f"{p}\n{int(review['period_rows'].get(p, 0))}" for p in rel.columns],
                  fontsize=6)
    fig.colorbar(im, ax=ax, label="log2(period mean |contribution| / overall mean)")
    ax.set_title("Baseline: where each term's out-of-fold contribution is concentrated (contact period; rows below)",
                 fontsize=10)
    fig.tight_layout()
    return fig


# ======================================================================
# 12. registration checks, statements, limits and the record
# ======================================================================
def registration_checks(root: Path = BANK_ROOT) -> pd.DataFrame:
    """Before any candidate is fitted: the pinned registration and the baseline's sources."""
    d = registration_digest()
    return pd.concat([pd.DataFrame([_check("the registration's SHA-256 is the pinned 1.0.0 digest",
                                           d == REGISTERED_SHA256, d)]), baseline_checks(root)], ignore_index=True)


#: The review document's statements, checked in the run that writes the record. Exact for structure and
#: integer counts that do not depend on a fitted model; within the stated tolerance for model-dependent
#: values (the accepted Windows run may differ from the sandbox in the last digits). Filled after results.
STATED: Dict[str, Any] = {
    "configurations": {
        "baseline": (13, 3, []),
        "drop_first": (12, 2, ["euribor3m_sigmoid_slope x economic_curvature_intensity"]),
        "drop_second": (12, 2, ["campaign x emp_var_rate_sigmoid_slope"]),
        "jed_alone": (13, 2, ["prior_x_stress x decay_x_density"]),
        "past_density": (13, 2, ["prior_x_stress x decay_x_density"]),
    },
    "correlated_r": (0.725, 0.002),
    "past_contexts": {("all rows", "fit_missing"): 917, ("all rows", "first_valued_day"): 3},
    "pooled": {
        ("baseline", "roc_auc"): (0.7933, 0.0005), ("baseline", "pr_auc"): (0.4170, 0.001),
        ("baseline", "nested_f2"): (0.5508, 0.002), ("baseline", "ece_calibrated"): (0.0019, 0.001),
        ("drop_first", "d_roc_auc"): (-0.0004, 0.0005), ("drop_first", "d_pr_auc"): (-0.0018, 0.001),
        ("drop_first", "d_nested_f2"): (0.0004, 0.003), ("drop_first", "ece_calibrated"): (0.0018, 0.001),
        ("drop_second", "d_roc_auc"): (-0.0013, 0.0005), ("drop_second", "d_roc_auc_high"): (-0.0003, 0.0003),
        ("drop_second", "d_pr_auc"): (-0.0020, 0.001), ("drop_second", "d_nested_f2"): (0.0007, 0.003),
        ("drop_second", "ece_calibrated"): (0.0021, 0.001),
        ("jed_alone", "d_roc_auc"): (0.0001, 0.0005), ("jed_alone", "d_pr_auc"): (0.0006, 0.001),
        ("jed_alone", "d_nested_f2"): (0.0010, 0.003), ("jed_alone", "ece_calibrated"): (0.0027, 0.001),
        ("past_density", "d_roc_auc"): (0.0005, 0.0005), ("past_density", "d_pr_auc"): (0.0017, 0.001),
        ("past_density", "d_nested_f2"): (-0.0011, 0.003), ("past_density", "ece_calibrated"): (0.0025, 0.001),
    },
    "fold_differences": {("past_density", "nested_f2", "sd"): (0.0087, 0.003)},
    "shapes": {
        ("baseline", "decay_x_density", "fold_sd_dense"): (0.513, 0.05),
        ("jed_alone", "joint_economic_decay", "fold_sd_dense"): (0.085, 0.02),
        ("past_density", "decay_x_density_past", "fold_sd_dense"): (0.172, 0.03),
        ("past_density", "decay_x_density_past", "sign_changes_dense"): (5, 1),
        ("jed_alone", "joint_economic_decay", "sign_changes_dense"): (2, 1),
        ("baseline", "euribor3m_sigmoid_slope", "fold_sd_dense"): (0.119, 0.02),
        ("baseline", "euribor3m_sigmoid_slope", "single_period_points"): (8, 1),
        ("baseline", "emp_var_rate_sigmoid_slope", "fold_sd_dense"): (0.053, 0.01),
        ("past_density", "emp_var_rate_sigmoid_slope", "fold_sd_dense"): (0.016, 0.005),
    },
    "centring": {
        ("baseline", "euribor3m_sigmoid_slope", "recentring_shift_max_abs"): (0.0085, 0.006),
        ("baseline", "euribor3m_sigmoid_slope", "recentring_shift_range"): (0.0135, 0.01),
        ("baseline", "emp_var_rate_sigmoid_slope", "recentring_shift_max_abs"): (0.0002, 0.001),
        ("baseline", "euribor3m_sigmoid_slope", "fold_sd_recentred"): (0.097, 0.02),
        ("baseline", "euribor3m_sigmoid_slope", "step_sd"): (0.119, 0.03),
        ("baseline", "emp_var_rate_sigmoid_slope", "fold_sd_recentred"): (0.0465, 0.01),
        ("baseline", "emp_var_rate_sigmoid_slope", "step_sd"): (0.077, 0.02),
        ("drop_first", "emp_var_rate_sigmoid_slope", "step_sd"): (0.073, 0.02),
    },
    "step_spread": {
        ("baseline", "period-level terms"): (0.632, 0.1), ("drop_first", "period-level terms"): (0.755, 0.1),
        ("jed_alone", "period-level terms"): (0.064, 0.03), ("past_density", "period-level terms"): (0.071, 0.03),
        ("baseline", "period-level terms except the decay slot"): (0.034, 0.01),
        ("baseline", "additive score"): (0.604, 0.03), ("jed_alone", "additive score"): (0.063, 0.01),
    },
    "period_level": {"decay_x_density": True, "joint_economic_decay": True, "decay_x_density_past": True,
                     "euribor3m_sigmoid_slope": True, "emp_var_rate_sigmoid_slope": True, "cpi_high_cellular": False},
    "trade_off": {("baseline", "step_corr"): (-0.47, 0.15)},
    # the cells are fixed by the data: exact on every platform
    "pair_cells": {
        ("baseline", "campaign & emp_var_rate_sigmoid_slope"): (56, 56, 39, 236),
        ("baseline", "euribor3m_sigmoid_slope & economic_curvature_intensity"): (360, 50, 44, 72),
        ("baseline", "prior_x_stress & decay_x_density"): (80, 31, 30, 20),
    },
    # the effect sizes come from each platform's fits (Windows 0.620 / 0.813 / 0.057 for the TIER 1 pair, the
    # sandbox 0.671 / 0.871 / 0.049): the statements are the claims both support, not a third-decimal value
    "pair_claims": {
        ("baseline", "campaign & emp_var_rate_sigmoid_slope", "max_abs"): (">=", 0.5),
        ("baseline", "campaign & emp_var_rate_sigmoid_slope", "range_supported"): (">=", 0.7),
        ("baseline", "campaign & emp_var_rate_sigmoid_slope", "fold_sd_max_supported"): ("<=", 0.1),
        ("baseline", "euribor3m_sigmoid_slope & economic_curvature_intensity", "max_abs"): ("<", 0.1),
        ("baseline", "euribor3m_sigmoid_slope & economic_curvature_intensity", "fold_sd_to_effect"): (">=", 2.0),
        ("baseline", "prior_x_stress & decay_x_density", "max_abs"): ("<", 0.1),
    },
    # Windows only (the accepted views): counts and verdicts of the accepted run, exact
    "complementarity": {
        "abstention_rows": 14628, "abstention_positives": 1120,
        "abstention_net": {"accepted": 447, "baseline": 530, "drop_first": 441, "drop_second": 633,
                           "jed_alone": 486, "past_density": 689},
        "cascade": {"drop_first": "not nominated", "drop_second": "nominated", "jed_alone": "nominated",
                    "past_density": "nominated"},
    },
    "fallback": {"held_out_fallback": 917, "held_out_raw": 32033, "held_out_fallback_days": [0, 2],
                 "inner_held_out_fallback": 4374, "fit_fallback_by_fold": [733, 740, 726, 737, 732],
                 "moved_in_every_fold": True, "control_unchanged": True},
    "fallback_term_change": {("10% of later fit labels", "min"): (0.074, 0.04),
                             ("10% of later fit labels", "max"): (0.165, 0.05),
                             ("every later fit label", "min"): (0.089, 0.04),
                             ("every later fit label", "max"): (0.278, 0.06)},
    "step5b": {"drop_first": "recommended", "drop_second": "not recommended", "jed_alone": "recommended",
               "past_density": "not recommended"},
    "within_tolerances": {"drop_first": True, "drop_second": True, "jed_alone": True, "past_density": True},
}


def _close(got, want, tol) -> bool:
    return got is not None and not (isinstance(got, float) and np.isnan(got)) and abs(float(got) - want) <= tol


def stated_checks(*, configs, corr, past, diff, mtable, summary, noms, centring=None,
                  pairs=None, fallback=None, comp=None) -> pd.DataFrame:
    """Each statement of the review document, in this run."""
    out = []
    for name, (nf, npairs, lost) in STATED.get("configurations", {}).items():
        c = configs[name]
        got = (c["n_features"], c["n_pairs"], [_pair_name(p) for p in c["lost_pairs"]])
        out.append(_check(f"{name}: {nf} features, {npairs} pairs, lost {lost}", got == (nf, npairs, lost), got))
    if "correlated_r" in STATED:
        want, tol = STATED["correlated_r"]
        r = float(corr["pairs"].iloc[0]["r"])
        out.append(_check(f"the correlated pair's r = {want} (+/- {tol})", _close(r, want, tol), round(r, 4)))
    for key, want in STATED.get("past_contexts", {}).items():
        row = past["contexts"].set_index("context").loc[key[0]]
        out.append(_check(f"past-only {key[0]} {key[1]} = {want}", int(row[key[1]]) == want, int(row[key[1]])))
    pooled = diff["pooled"].set_index("configuration")
    for (name, col), (want, tol) in STATED.get("pooled", {}).items():
        got = pooled.loc[name, col]
        out.append(_check(f"{name} {col} = {want} (+/- {tol})", _close(got, want, tol), got))
    fd = diff["fold_differences"].set_index(["configuration", "metric"])
    for (name, metric, col), want in STATED.get("fold_differences", {}).items():
        got = fd.loc[(name, metric), col]
        ok = _close(got, want[0], want[1]) if isinstance(want, tuple) else int(got) == want
        out.append(_check(f"{name} {metric} per-fold {col} = {want}", ok, got))
    sm = summary.set_index(["configuration", "feature"])
    for (name, feat, col), (want, tol) in STATED.get("shapes", {}).items():
        got = sm.loc[(name, feat), col]
        out.append(_check(f"{name} {feat} {col} = {want} (+/- {tol})", _close(got, want, tol), got))
    if centring is not None:
        cc = centring["centring"].set_index(["configuration", "feature"])
        for (name, feat, col), (want, tol) in STATED.get("centring", {}).items():
            got = float(cc.loc[(name, feat), col])
            out.append(_check(f"{name} {feat} {col} = {want} (+/- {tol})", _close(got, want, tol), round(got, 4)))
        sp = centring["step_spread"]
        sp = sp[sp["statistic"] == "sd across folds"].set_index("configuration")
        for (name, col), (want, tol) in STATED.get("step_spread", {}).items():
            got = float(sp.loc[name, col])
            out.append(_check(f"{name} {col} regime-step sd across folds = {want} (+/- {tol})",
                              _close(got, want, tol), round(got, 4)))
        to = centring["trade_off"].set_index("configuration")
        for (name, col), (want, tol) in STATED.get("trade_off", {}).items():
            got = float(to.loc[name, col])
            out.append(_check(f"{name} {col} = {want} (+/- {tol})", _close(got, want, tol), round(got, 4)))
        wd = centring["within_day_spread"].set_index("feature")
        for f, want in STATED.get("period_level", {}).items():
            got = bool(wd.loc[f, "period_level"])
            out.append(_check(f"{f} is period-level (constant within a contact day up to floating-point noise) = "
                              f"{want}", got == want, float(wd.loc[f, "largest_within_day_range"])))
    if pairs is not None:
        pt = pair_table(pairs).set_index(["configuration", "pair"])
        cols = ("cells", "cells_with_rows", "cells_supported", "rows_in_sparse_cells")
        for (name, pair), want in STATED.get("pair_cells", {}).items():
            got = tuple(int(pt.loc[(name, pair), c]) for c in cols)
            out.append(_check(f"{name} {pair} cells / with rows / well-supported / rows in sparse cells = {want}",
                              got == tuple(want), got))
        ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b}
        for (name, pair, col), (op, bound) in STATED.get("pair_claims", {}).items():
            got = float(pt.loc[(name, pair), col])
            out.append(_check(f"{name} {pair} {col} {op} {bound}", bool(ops[op](got, bound)), round(got, 4)))
    if fallback is not None:
        fb, tot = STATED.get("fallback", {}), fallback["totals"]
        for key in ("held_out_fallback", "held_out_raw", "inner_held_out_fallback", "fit_fallback_by_fold"):
            if key in fb:
                out.append(_check(f"past-only fallback: {key} = {fb[key]}", tot[key] == fb[key], tot[key]))
        if "held_out_fallback_days" in fb:
            days = fallback["per_fold"]["held_out_fallback_days"].tolist()
            got = [min(d[0] for d in days), max(d[1] for d in days)]
            out.append(_check(f"past-only fallback rows are contacted on days {fb['held_out_fallback_days']}",
                              got == fb["held_out_fallback_days"], got))
        sv = fallback["sensitivity"]
        if "moved_in_every_fold" in fb:
            got = bool(sv["fallback_moved"].all()) and set(sv["fold"]) == set(range(N_FOLDS))
            out.append(_check(f"flipping later fit labels moves the fallback in every fold = {fb['moved_in_every_fold']}",
                              got == fb["moved_in_every_fold"], got))
        if "control_unchanged" in fb:
            got = bool(sv["control_raw_values_unchanged"].all())
            out.append(_check(f"the first valued day's raw values are unchanged = {fb['control_unchanged']}",
                              got == fb["control_unchanged"], got))
        te = fallback.get("term_effect")
        for (pert, stat), (want, tol) in STATED.get("fallback_term_change", {}).items():
            got = None if te is None else float(getattr(te.loc[te["perturbation"] == pert, "term_score_change"].abs(),
                                                        stat)())
            out.append(_check(f"fallback term-score change, {pert}, {stat} |change| = {want} (+/- {tol})",
                              _close(got, want, tol), None if got is None else round(got, 4)))
    cs = STATED.get("complementarity")
    if cs:
        if comp is None:
            out.append(_check("complementarity statements: not checkable in this run (the accepted views are not "
                              "available; with PR42_REQUIRE_ACCEPTED_VIEWS=1 they are required)", True, None))
        else:
            t = comp["table"]
            ab = t[t["population"] == "Stage 2 abstentions"].set_index("ebm")
            got = (int(ab["rows"].iloc[0]), int(ab["positives"].iloc[0]))
            want = (cs["abstention_rows"], cs["abstention_positives"])
            out.append(_check(f"Stage 2 abstentions: rows, positives = {want}", got == want, got))
            for e, w in cs["abstention_net"].items():
                g = int(ab.loc[e, "net"]) if e in ab.index else None
                out.append(_check(f"{e}: net Stage 1 corrections on Stage 2 abstentions = {w}", g == w, g))
            nmc = noms.set_index("configuration")["cascade"]
            for e, w in cs["cascade"].items():
                g = str(nmc.loc[e]) if e in nmc.index else None
                out.append(_check(f"{e}: cascade {w}", g == w, g))
    nm = noms.set_index("configuration")
    for name, want in STATED.get("step5b", {}).items():
        got = nm.loc[name, "step5b"]
        out.append(_check(f"{name}: Step 5B {want}", got == want, got))
    for name, want in STATED.get("within_tolerances", {}).items():
        got = bool(nm.loc[name, "within_tolerances"])
        out.append(_check(f"{name}: within the registered tolerances = {want}", got == want, got))
    if not out:
        out.append(_check("the review document's statements are registered in STATED", False, "empty"))
    return pd.DataFrame(out)


LIMITS = (
    "Training side only, random outer folds. A simplification's value under a chronological split is PR 49's "
    "question; the past-only density changes only that one feature's construction.",
    "The baseline is refitted from the tracked settings: the learning rate is transcribed to six decimals from "
    "glass_cascade Cell 18 (the only tracked source) and the sandbox runs Linux, not the accepted Windows "
    "environment. Parity is reported against the PR 39 records and, where the accepted artifact is available, "
    "row by row. Every candidate is compared with the refitted baseline, not with the artifact.",
    "Each difference measures the whole simplification. Removing a feature also removes its pair; replacing "
    "decay_x_density also removes prior_x_stress x decay_x_density. Neither part is isolated, and no pair is "
    "reselected (Step 5B).",
    "No candidate is retuned. The accepted hyperparameters were chosen for the 13-feature, three-pair model; a "
    "candidate's own optimum may differ.",
    "The past-only density assumes an outcome is known by the next contact day (Step 3B's lag-0 rule); the "
    "outcome time is not recorded. Its history is grown, not redrawn. Its raw values are past-only; its "
    "fallback is not: rows without a raw value (917 held-out rows, contact days 0-2; 726-740 fit rows per fold; "
    "4,374 inner held-out rows) get the production cleaner's median over the whole fit fold, which depends on "
    "labels of later contacts (fallback_trace). The candidate is evaluated as registered and fitted; a "
    "deployable version needs a fallback fixed from earlier data only.",
    "The global raw and calibrated thresholds are chosen on the same rows they are applied to (in-selection); "
    "the fold-nested thresholds are the strictly nested operating point.",
    "Shapes are fold models' and an all-rows fit's term scores on a common grid, each centred on the training "
    "rows. Sparse and single-period regions are marked, not removed; a period concentrated in a narrow range "
    "is evidence only there. For a period-level feature most values belong to one contact period, so a shape "
    "point is evidence about that period; the regime step compares periods.",
    "Two presentation choices were made after the registered figures were inspected and change no fit, metric, "
    "threshold or ranking: pair surfaces are drawn on support-defined cells (the registered grid of both "
    "features left almost every cell empty, so its colour scale was set by unsupported extrapolation), and the "
    "fold offsets of the correlated slopes are re-examined by the centring check. The registered centring "
    "stays the reported one. The main-effect grid keeps every distinct value as registered, so two "
    "emp_var_rate_sigmoid_slope values equal up to floating-point noise appear as separate grid points.",
    "Values that come from fitted models differ slightly between platforms (the sandbox's Linux fits and the "
    "accepted Windows run's): the pooled metrics in the last digits, the shape and pair summaries by up to about "
    "0.05 log-odds (the TIER 1 pair's largest cell 0.671 vs 0.620, about its own fold spread). Data-determined "
    "quantities (rows, folds, cells, support, the fallback rows) are identical. The document's pair sizes are "
    "therefore stated for both platforms and checked as claims, not as third-decimal values.",
    "The interpretability judgements (INTERPRETATION) are written after the results were seen; the registered "
    "rule asks for a documented gain, not a threshold, and the evidence for each is in the record.",
    "Complementarity uses the accepted Stage 1 and Stage 2 decisions and is computed only where their artifacts "
    "exist locally. A disagreement count does not establish a better cascade decision; PRs 44-45 compare "
    "cascades at comparable coverage under the shared obligations.",
    "Every target-dependent encoder (NSD, ECI, day x month) uses labels of later contacts under the accepted "
    "protocol (Commit 3); only the past-only candidate removes that for one feature.",
)


def _curves_compact(curves: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for (name, f), g in curves.groupby(["configuration", "feature"], sort=False):
        xs = np.sort(g["x"].unique())
        models = {}
        shifts = {}
        for m, h in g.groupby("model", sort=False):
            h = h.set_index("x").reindex(xs)
            models[m] = [float(v) for v in h["score"]]
            shifts[m] = float(h["shift"].iloc[0])
        out.append({"configuration": name, "feature": f, "x": [float(v) for v in xs], "scores": models,
                    "shift": shifts})
    return out


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


POST_REGISTRATION = (
    "accepted views checked before any fit: a present artifact that fails its pin or boundary check (or the "
    "training rows' labels) raises with its role; PR42_REQUIRE_ACCEPTED_VIEWS=1 requires all three and the "
    "reconciled complementarity tables (complementarity_requirement)",
    "past-only fallback trace (fallback_trace, fallback_term_effect): the cleaner's fit-fold median fallback "
    "traced to the EBM frames, its held-out rows counted, and its dependence on later fit labels measured; the "
    "candidate is unchanged",
    "pair surfaces on support-defined cells (pair_bins): the mean centred term contribution of each cell's "
    "training rows, empty cells left empty; replaces the registered regular grid for display",
    "centring check (centring_check): re-centring over well-supported values and centring-free fold "
    "amplitudes and regime steps for the correlated slopes, the period-level terms and the additive score",
    "figure palette: a fixed categorical order checked for colour-vision-deficiency separation; a diverging "
    "scale with a neutral grey midpoint for signed scores",
)


def record_sections(*, reg_checks, block, X, block_chk, past, past_chk, dists, corr, corr_chk, configs, cfg_chk,
                    fits, fit_chk, parity_table, mtable, diff, review, curves, supports, summary, pairs,
                    views_status, comp, comp_chk, noms, stated, centring=None, fallback=None,
                    fallback_chk=None) -> Dict[str, Any]:
    """The ``stage3_simplification`` record's content sections (identity keys are added by the writer)."""
    def recs(df):
        return None if df is None else df.to_dict("records")

    fit_out = {}
    for name, r in fits.items():
        fit_out[name] = {"threshold_raw": r["threshold_raw"], "threshold_calibrated": r["threshold_calibrated"],
                         "nested_by_fold": [float(v) for v in r["nested"]["by_fold"]],
                         "nested_mean": float(r["nested"]["mean"]), "calibration": r["calibration"],
                         "fold_applied": r["fold_applied"], "terms": r["terms"], "expected_terms": r["expected_terms"],
                         "oof_raw_sha256": _sha(r["proba"]), "oof_calibrated_sha256": _sha(r["calibrated"]),
                         "fit_rows": r["fit_rows"], "inner_fit_rows": r["inner_sizes"]}
    return {
        "scope": ("PR 42 Commit 5 Step 5A: feature analysis and Stage 3 EBM simplification (training side; "
                  "standalone Stage 3 and its relationship to Stage 1 and Stage 2 decisions)"),
        "registration": {"content": REGISTRATION, "sha256": registration_digest(), "checks": recs(reg_checks)},
        "sources": {"params_transcription": params_transcription(), "accepted_figures": accepted_figures()},
        "rows_and_folds": {"rows": int(len(X)), "row_index": [int(i) for i in X.index],
                           "outer_fold_id": [int(v) for v in block.tune_plan.fold_id],
                           "fold_fingerprint": _fold_fp(block.tune_plan.fold_id),
                           "inner_fold_id": block.inner_fold_ids().tolist(),
                           "block_settings": block.settings, "block_checks": recs(block_chk)},
        "past_only_density": {"definition": REGISTRATION["past_only_density"], "contexts": recs(past["contexts"]),
                              "checks": recs(past_chk),
                              "fallback_trace": None if fallback is None else {
                                  "description": fallback["description"], "per_fold": recs(fallback["per_fold"]),
                                  "sensitivity": recs(fallback["sensitivity"]), "totals": fallback["totals"],
                                  "term_effect": recs(fallback.get("term_effect")), "checks": recs(fallback_chk)}},
        "review": {"distributions": recs(dists), "correlation_pairs": recs(corr["pairs"]),
                   "correlations_extra": corr["extra"], "correlation_checks": recs(corr_chk),
                   "importance": recs(review["importance_summary"]),
                   "period_relative_contribution": {str(k): v for k, v in
                                                    review["period_relative"].to_dict("index").items()},
                   "period_rows": {str(k): int(v) for k, v in review["period_rows"].items()}},
        "configurations": {"definitions": {k: {kk: vv for kk, vv in v.items()} for k, v in configs.items()},
                           "checks": recs(cfg_chk)},
        "fits": {"settings": {"params": PARAMS, "seed": SEED}, "by_configuration": fit_out, "checks": recs(fit_chk)},
        "parity": recs(parity_table),
        "metrics": {"operating_points": recs(mtable), "pooled": recs(diff["pooled"]),
                    "per_fold": recs(diff["per_fold"]), "fold_differences": recs(diff["fold_differences"]),
                    "decisions": recs(diff["decisions"])},
        "post_registration": list(POST_REGISTRATION),
        "shapes": {"conventions": REGISTRATION["shapes"],
                   "curves": _curves_compact(curves), "support": recs(supports), "summary": recs(summary),
                   "pair_table": recs(pair_table(pairs)),
                   "pairs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
                             for s in pairs["surfaces"]],
                   "centring_check": None if centring is None else {
                       "description": centring["description"],
                       "period_level_features": centring["period_level_features"],
                       "within_day_spread": recs(centring["within_day_spread"]),
                       "centring": recs(centring["centring"]), "steps": recs(centring["steps"]),
                       "step_spread": recs(centring["step_spread"]), "trade_off": recs(centring["trade_off"])}},
        "complementarity": {"views": views_status, "table": None if comp is None else recs(comp["table"]),
                            "against_baseline": None if comp is None else recs(comp["against_reference"]),
                            "checks": recs(comp_chk)},
        "nominations": recs(noms),
        "interpretation": INTERPRETATION,
        "feature_notes": FEATURE_NOTES,
        "stated": recs(stated),
        "limits": list(LIMITS),
    }
