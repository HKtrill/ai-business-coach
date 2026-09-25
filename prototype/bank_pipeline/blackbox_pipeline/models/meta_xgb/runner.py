"""
blackbox_pipeline.models.meta_xgb.runner
========================================
Stage 4 of the black-box cascade: Meta-XGB, the learned counterpart of the
GLASS Meta-EBM weighted-confidence arbiter.

    meta_xgb_artifact, meta_xgb_path = train_meta_xgb_stage(
        GLOBAL_SPLIT,
        mlp_path=MLP_S1_PATH,
        rf_path=f"{STAGE2_OUT}/rf_router_two_pass.joblib",
        xgb_path=XGB_STAGE3_PATH,
        meta_ebm_path=META_EBM_PATH,         # optional: held-out comparison
    )

Order of operations (each step reads only what is listed)
---------------------------------------------------------
1. inputs     black-box Stage 1–3 artifacts → shared.stage4 checks
2. features   STAGE4_FEATURES from upstream OOF (train) / refit (test)
3. learner    shared Stage3Runner protocol, via ``MetaXGBRunner``:
                tune          Optuna, meta train folds only (5-fold, seed 42)
                CV report     10-fold, train only
                refit         all train rows
                meta OOF      one model per fold, scores only its held-out fold
                calibrate     OOF-ECE gate; fold-nested on train, all-OOF map → test
                threshold     F2 grid on META OOF; fold-nested per-row for train
                abstention    min |p − t| by F2 on META OOF, coverage ≥ 50%
              ── configuration frozen ──
                test          refit model scores the test block once
4. evaluate   no-abstention + with-abstention on test
5. compare    (optional) against the saved Meta-EBM on the same test rows

The learner is the same shared runner both Stage 3 arms use; only step 3's
abstention insert and the test step's position differ (``MetaXGBRunner.fit``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from shared.stage3 import PassthroughPipeline, Stage3Block, Stage3Runner
except ImportError:                     # pragma: no cover - legacy monolith
    from shared.stage_runner import PassthroughPipeline, Stage3Block, Stage3Runner

from shared.stage4 import ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, tune_min_confidence

from .artifact import MetaXGBArtifact, save_meta_xgb
from .config import EXCLUDED_CHANNELS, FEATURE_CONTRACT, STAGE4_FEATURES, MetaXGBConfig
from .estimator import MetaXGBEstimator
from .evaluation import abstention_metrics, classification_metrics
from .inputs import STREAMS, build_feature_frames, load_blackbox_stage4_inputs

SEC_META = "4g. Meta-level honesty"


# ======================================================================
# Runner
# ======================================================================
class MetaXGBRunner(Stage3Runner):
    """
    ``Stage3Runner`` with the Stage 4 abstention choice inserted before any
    test data is touched. Everything else is the shared implementation.
    """

    def fit(self, block, params: Optional[dict] = None, params_source: Optional[str] = None,
            params_provenance: Optional[dict] = None, extras: Optional[dict] = None):
        cfg = self.config
        self.events_: list = []
        ev = self.events_.append

        self._validate(block)
        best_params, cv_score, tuning_dict = self._tune_or_reuse(
            block, params, params_source, params_provenance)
        ev("tuned (train folds)")
        cv_report = self._report_cv(block, best_params)
        ev("cv report (train folds)")
        self._refit(block, best_params)
        ev("refit (all train rows)")
        oof = self._out_of_fold(block, best_params)
        ev("meta OOF generated")
        self._calibrate(block, oof)
        ev("calibration decided on meta OOF")
        choice, choice_cal, nested = self._select_thresholds(block, oof)
        ev("operating threshold selected on meta OOF")

        # ---- Stage 4 train-side choice: META OOF ONLY -------------------
        y_tr = np.asarray(block.y_train).astype(int)
        if cfg.operating_space == "raw":
            p_oof, t_glob, t_key = oof.proba, choice.threshold, "oof"
            t_rows = nested["per_row"] if cfg.nested_oof_threshold \
                else np.full(len(y_tr), t_glob)
        else:
            p_oof = self.calibrator_.oof_calibrated_
            t_glob, t_key = choice_cal.threshold, "oof_calibrated_space"
            t_rows = (self.selector.select_nested(y_tr, p_oof, block.tune_plan)["per_row"]
                      if cfg.nested_oof_threshold else np.full(len(y_tr), t_glob))
        pred_oof = (p_oof >= t_rows).astype(int)
        conf_oof = np.abs(p_oof - t_rows)
        best = tune_min_confidence(y_tr, pred_oof, conf_oof, ABSTENTION_GRID,
                                   ABSTENTION_MIN_COVERAGE)
        if best is None:
            best = {"min_confidence": float(ABSTENTION_GRID[0]), "train_f2": None,
                    "train_coverage": None, "sweep": [],
                    "selected_by": "fallback: smallest grid value (no value met coverage floor)"}
        else:
            best["selected_by"] = (f"F2 on meta OOF retained rows, coverage ≥ "
                                   f"{ABSTENTION_MIN_COVERAGE:.0%} (Meta-EBM rule)")
        self.operating_ = {
            "space": cfg.operating_space, "threshold": float(t_glob),
            "threshold_key": t_key,
            "threshold_source": f"learner.threshold[{t_key!r}] — F2 grid on meta OOF "
                                "train predictions",
            "train_side_thresholds": "fold-nested per row" if cfg.nested_oof_threshold
                                     else "global",
        }
        self.abstention_ = {
            **{k: v for k, v in best.items() if k != "sweep"},
            "sweep": best.get("sweep", []),
            "confidence": "|p − t|, p = meta probability in the operating space, "
                          "t = operating threshold (fold-nested on train rows)",
            "grid": [float(g) for g in ABSTENTION_GRID],
            "min_coverage": ABSTENTION_MIN_COVERAGE,
            "tuned_on": "meta OOF train predictions",
            "tuned_on_n": int(len(y_tr)),
        }
        self.train_eval_ = {
            "note": "meta OOF; per-row fold-nested thresholds; abstention cut chosen "
                    "on these same rows (in-selection, as Meta-EBM's)",
            "no_abstain": classification_metrics(y_tr, pred_oof, p_oof),
            "with_abstain": abstention_metrics(y_tr, pred_oof, p_oof,
                                               conf_oof >= self.abstention_["min_confidence"]),
        }
        self.frozen_ = {"operating": dict(self.operating_),
                        "min_confidence": self.abstention_["min_confidence"],
                        "best_params": dict(best_params)}
        ev("configuration frozen")

        # ---- first contact with the test split --------------------------
        scored = self._evaluate(block, oof, choice, choice_cal, nested)
        ev("test scored (refit model, once)")
        artifact = self._build_artifact(block, best_params, cv_score, tuning_dict,
                                        cv_report, oof, choice, choice_cal, nested,
                                        scored, extras)
        self.artifact_ = artifact
        return artifact


# ======================================================================
# Orchestration
# ======================================================================
def _run_checks(checks: list, verbose: bool, section: Optional[str] = None) -> list:
    from shared.validation import ValidationReport
    rep = ValidationReport()
    for sec, stage, name, fn, detail in checks:
        rep.check(sec, stage, name, fn, detail)
    if verbose:
        rep.show()
    rep.raise_if_failed()
    return list(rep.rows)


def _meta_checks(learner, runner, inputs, n_train: int) -> list:
    oof, thr, cal = learner.oof, learner.threshold, learner.calibration
    fold = np.asarray(oof["fold_id"])
    k = len(np.unique(fold))
    same_k = [n for n, s in inputs.streams.items()
              if s.train_fold_id is not None and len(np.unique(s.train_fold_id)) == k]
    op, ev = runner.operating_, runner.events_
    c = []
    c.append((SEC_META, "stage4", "meta OOF: every train row scored once",
              lambda: len(oof["proba"]) == n_train and np.isfinite(oof["proba"]).all(), ""))
    c.append((SEC_META, "stage4", "meta OOF: each fold model excluded its own rows",
              lambda: all(fs + int((fold == i).sum()) == n_train
                          for i, fs in enumerate(oof["summary"]["fit_sizes"])),
              str(oof["summary"]["fit_sizes"])))
    c.append((SEC_META, "stage4", "meta OOF ≠ refit in-sample train proba",
              lambda: not np.array_equal(np.asarray(oof["proba"]),
                                         np.asarray(learner.refit["train_proba"])), ""))
    for n in same_k:
        c.append((SEC_META, "stage4", f"meta folds = {n}'s {k}-fold partition",
                  lambda n=n: np.array_equal(fold, np.asarray(inputs[n].train_fold_id, int)),
                  "shared cascade splitter"))
    c.append((SEC_META, "stage4", "operating threshold selected on meta OOF",
              lambda: op["threshold_key"] in ("oof", "oof_calibrated_space")
              and "out-of-fold" in thr[op["threshold_key"]]["selected_on"]
              and thr[op["threshold_key"]]["n_rows"] == n_train,
              f"{op['threshold']:.4f} ← threshold[{op['threshold_key']!r}]"))
    c.append((SEC_META, "stage4", "operating threshold is not the test oracle",
              lambda: op["threshold_key"] != "test_oracle", ""))
    c.append((SEC_META, "stage4", "calibration gate measured on meta OOF (train)",
              lambda: cal.get("gate_measured_on") == "out-of-fold training predictions"
              and cal.get("n_rows") == n_train,
              f"{'applied' if cal.get('applied') else 'skipped'} — {cal.get('reason')}"))
    c.append((SEC_META, "stage4", "abstention tuned on meta OOF (train) only",
              lambda: runner.abstention_["tuned_on_n"] == n_train,
              f"min |p−t| = {runner.abstention_['min_confidence']:.2f}"))
    c.append((SEC_META, "stage4", "configuration frozen before the test split is scored",
              lambda: ev.index("configuration frozen") < ev.index("test scored (refit model, once)"),
              " → ".join(ev)))
    c.append((SEC_META, "stage4", "no upstream Stage 3 test-oracle threshold consumed",
              lambda: inputs[STREAMS["stage3"]].notes.get("test_oracle_threshold_consumed") is False,
              ""))
    return c


def train_meta_xgb_stage(
    GLOBAL_SPLIT: dict,
    mlp_path,
    rf_path,
    xgb_path,
    *,
    config: Optional[MetaXGBConfig] = None,
    params: Optional[dict] = None,
    params_source: Optional[str] = None,
    meta_ebm_path=None,
    artifact_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    extra_fingerprints: Optional[dict] = None,
) -> tuple[MetaXGBArtifact, Optional[str]]:
    """
    Train, cross-fit, calibrate, threshold, evaluate and save Meta-XGB.

    ``params``: skip Optuna and use these hyperparameters (recorded as such).
    ``meta_ebm_path``: saved GLASS Meta-EBM artifact (schema meta_ebm/2) for the
    held-out comparison; optional.
    """
    cfg = config or MetaXGBConfig(verbose=verbose, show_progress_bar=verbose)
    if artifact_dir is not None:
        cfg.artifact_dir = artifact_dir
    say = print if cfg.verbose else (lambda *a, **k: None)

    say("\n" + "=" * 80)
    say("🎯 META-XGB — STAGE 4 LEARNED ARBITER (black-box counterpart of Meta-EBM)")
    say("=" * 80)

    # ---- 1. inputs ------------------------------------------------------
    say("\n📂 1. Loading and verifying black-box Stage 1–3 outputs...")
    inputs = load_blackbox_stage4_inputs(GLOBAL_SPLIT, mlp_path, rf_path, xgb_path,
                                         verbose=cfg.verbose,
                                         extra_fingerprints=extra_fingerprints)
    for s in inputs.streams.values():
        say(f"   {s.model:<30} train ← {s.train_source}   test ← {s.test_source}")
        say(f"   {'':<30} [{s.probability_space}]")

    # ---- 2. features ----------------------------------------------------
    say("\n🧱 2. Building the Stage 4 feature block...")
    X_tr, X_te, fchecks = build_feature_frames(inputs)
    feature_validation = _run_checks(fchecks, cfg.verbose)
    say(f"   train {X_tr.shape} | test {X_te.shape} | features: {list(X_tr.columns)}")

    # ---- 3. learner ------------------------------------------------------
    say("\n🔧 3. Meta-XGB learner (shared Stage 3 protocol, Stage 4 insert)...")
    block = Stage3Block.build(
        X_tr, pd.Series(inputs.y_train, index=inputs.index_train),
        X_te, pd.Series(inputs.y_test, index=inputs.index_test),
        PassthroughPipeline,
        feature_fit_scope=cfg.feature_fit_scope, n_tune_folds=cfg.n_tune_folds,
        n_eval_folds=cfg.n_eval_folds, random_state=cfg.random_state,
        stratify=cfg.stratify, class_weight=cfg.class_weight, clean_inf=cfg.clean_inf,
        impute_missing=cfg.impute_missing, expected_features=cfg.expected_features,
        split_fingerprint=inputs.split_fingerprint, verbose=cfg.verbose,
    )
    runner = MetaXGBRunner(cfg, MetaXGBEstimator(cfg.search_space, cfg.random_state, cfg.n_jobs))
    learner = runner.fit(block, params=params, params_source=params_source)
    meta_validation = _run_checks(_meta_checks(learner, runner, inputs, inputs.n_train),
                                  cfg.verbose)

    # ---- 4. held-out evaluation (configuration already frozen) ----------
    say("\n📊 4. Held-out test evaluation (once)...")
    op, ab = runner.operating_, runner.abstention_
    p_te = np.asarray(learner.refit["test_proba"] if op["space"] == "raw"
                      else learner.refit["test_proba_calibrated"], dtype=float)
    pred_te = (p_te >= op["threshold"]).astype(int)
    retained = np.abs(p_te - op["threshold"]) >= ab["min_confidence"]
    y_te = inputs.y_test
    test = {
        "proba": p_te,
        "proba_raw": np.asarray(learner.refit["test_proba"], dtype=float),
        "proba_calibrated": np.asarray(learner.refit["test_proba_calibrated"], dtype=float),
        "pred": pred_te,
        "retained": retained,
        "pred_with_abstain": np.where(retained, pred_te, -1),
        "no_abstain": classification_metrics(y_te, pred_te, p_te),
        "with_abstain": abstention_metrics(y_te, pred_te, p_te, retained),
    }
    _print_eval(say, test)

    artifact = MetaXGBArtifact(
        created_at=datetime.now().isoformat(timespec="seconds"),
        split_fingerprint=inputs.split_fingerprint,
        split_fingerprints=dict(inputs.split_fingerprints),
        index_train=list(inputs.index_train),
        index_test=list(inputs.index_test),
        feature_names=list(STAGE4_FEATURES),
        feature_contract={"features": dict(FEATURE_CONTRACT),
                          "excluded_channels": list(EXCLUDED_CHANNELS),
                          "router_proba_on_abstain": cfg.router_proba_on_abstain},
        stage4_inputs=inputs.provenance(),
        fold_provenance={
            "meta": {k: v for k, v in learner.fold_plan.items() if k != "fold_id"},
            "upstream": inputs.fold_provenance,
        },
        learner=learner,
        operating=dict(op),
        abstention=dict(ab),
        train_oof_eval=runner.train_eval_,
        test=test,
        events=list(runner.events_),
        validation={"inputs": inputs.validation, "features": feature_validation,
                    "meta": meta_validation},
    )

    if meta_ebm_path is not None:
        from .comparison import compare_with_meta_ebm
        say("\n⚖️  5. Meta-EBM vs Meta-XGB (same held-out rows)...")
        artifact.comparison = compare_with_meta_ebm(
            GLOBAL_SPLIT, meta_ebm_path, artifact, verbose=cfg.verbose,
            extra_fingerprints=extra_fingerprints)

    path = save_meta_xgb(artifact, cfg.artifact_dir, cfg.artifact_stem) if save else None
    if path:
        say(f"\n💾 saved → {path}")
    say("\n🎉 META-XGB COMPLETE\n" + "=" * 80)
    return artifact, path


def _print_eval(say, test: dict) -> None:
    rows = {"no abstention": test["no_abstain"]}
    if test["with_abstain"]["metrics"]:
        rows[f"with abstention (coverage {test['with_abstain']['coverage']:.1%})"] = \
            test["with_abstain"]["metrics"]
    cols = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    df = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in rows.items()}).T
    say(df.round(4).to_string())
