"""
shared.stage3.runner
====================
``Stage3Runner``: orchestration only. ``fit`` reads as the experiment outline;
each step delegates to the subsystem that owns it:

    validate           Stage3Block.check_config                (data)
    tune / reuse       Stage3Tuner                             (tuning)
    k-fold report      CVEvaluator                             (evaluation)
    refit              estimator.fit on all of train           (estimator)
    OOF                OOFGenerator                            (evaluation)
    calibrate          Stage3Calibrator                        (calibration)
    threshold          F2ThresholdSelector                     (thresholding)
    evaluate           decision_block + stage3_metric_suite    (thresholding, evaluation)
    build artifact     Stage3Artifact / Stage3ArtifactStore    (artifact)

Step ORDER is the pre-split order (refit happens before OOF): estimator fits
may consume global RNG state, so reordering could change results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np

from .artifact import Stage3Artifact, Stage3ArtifactStore
from .calibration import Stage3Calibrator, calibration_diagnostics
from .data import Stage3Block
from .estimator import Stage3ServingModel
from .evaluation import CVEvaluator, CVReport, OOFGenerator, OOFPredictions, stage3_metric_suite
from .thresholding import F2ThresholdSelector, decision_block
from .tuning import Stage3Tuner


class Stage3Runner:
    def __init__(self, config, estimator):
        self.config = config
        self.estimator = estimator
        self.selector = F2ThresholdSelector(
            beta=config.beta, low=config.threshold_low,
            high=config.threshold_high, steps=config.threshold_steps,
        )
        self.artifact_: Optional[Stage3Artifact] = None
        self.model_ = None
        self.calibrator_: Optional[Stage3Calibrator] = None

    # ==================================================================
    def fit(
        self,
        block: Stage3Block,
        params: Optional[dict[str, Any]] = None,
        params_source: Optional[str] = None,
        params_provenance: Optional[dict] = None,
        extras: Optional[dict] = None,
    ) -> Stage3Artifact:
        # ---- 1. check -------------------------------------------------
        self._validate(block)
        # ---- 2. tuning ------------------------------------------------
        best_params, cv_score, tuning_dict = self._tune_or_reuse(
            block, params, params_source, params_provenance)
        # ---- 3. k-fold report -----------------------------------------
        cv_report = self._report_cv(block, best_params)
        # ---- 4. refit -------------------------------------------------
        self._refit(block, best_params)
        # ---- 5. OOF ---------------------------------------------------
        oof = self._out_of_fold(block, best_params)
        # ---- 6. calibration -------------------------------------------
        self._calibrate(block, oof)
        # ---- 7. operating point ---------------------------------------
        choice, choice_cal, nested = self._select_thresholds(block, oof)
        # ---- 8. test split --------------------------------------------
        scored = self._evaluate(block, oof, choice, choice_cal, nested)
        # ---- 9. assemble ----------------------------------------------
        artifact = self._build_artifact(
            block, best_params, cv_score, tuning_dict, cv_report, oof,
            choice, choice_cal, nested, scored, extras,
        )
        self.artifact_ = artifact
        if self.config.verbose:
            artifact.describe()
        return artifact

    # ==================================================================
    def save(self, verbose: bool = True) -> str:
        if self.artifact_ is None:
            raise ValueError("Call fit() first")
        cfg = self.config
        return Stage3ArtifactStore(
            cfg.artifact_dir, cfg.artifact_stem, cfg.timestamped,
            cfg.write_latest_pointer,
        ).save(self.artifact_, verbose=verbose)

    # ==================================================================
    # steps
    # ==================================================================
    def _say(self, *a, **k) -> None:
        if self.config.verbose:
            print(*a, **k)

    # ------------------------------------------------------------------
    def _validate(self, block: Stage3Block) -> None:
        cfg, say = self.config, self._say
        block.check_config(cfg)
        if cfg.expected_features is None:
            say("  ⚠️  config.expected_features is None — the shared feature "
                "contract is NOT being enforced. Pass EBM_FEATURES.")
        say("\n" + "=" * 78)
        say(f"  STAGE 3 — {cfg.model_family.upper()}  (shared protocol; "
            f"counterpart {cfg.counterpart})")
        say("=" * 78)
        say(f"  train {len(block.X_train):,} | test {len(block.X_test):,} | "
            f"{len(block.features)} features | fitted {block.feature_fit_scope}")
        say(f"  {block.weights.describe()}")
        say(f"  block fingerprint {block.block_fingerprint[:16]}…")

    # ------------------------------------------------------------------
    def _tune_or_reuse(
        self,
        block: Stage3Block,
        params: Optional[dict[str, Any]],
        params_source: Optional[str],
        params_provenance: Optional[dict],
    ) -> tuple[dict[str, Any], float, dict]:
        cfg, est = self.config, self.estimator
        fold_plan_meta = {k: v for k, v in block.tune_plan.to_dict().items()
                          if k != "fold_id"}
        if params is None:
            tuning = Stage3Tuner(cfg, est, block.tune_folds, fold_plan_meta,
                                 block.feature_fit_scope).tune()
            best_params, cv_score = tuning.best_params, tuning.best_value
            tuning_dict = tuning.to_dict()
        else:
            best_params = est.normalize_params(params)
            cv_score = float("nan")
            tuning_dict = {
                "tuned": False,
                "best_params": best_params,
                "source": "supplied — Optuna not run"
                          + (f" ({params_source})" if params_source else ""),
                "provenance": params_provenance,
            }
            self._say(f"\n  using supplied hyperparameters (no search): {best_params}")
        return best_params, cv_score, tuning_dict

    # ------------------------------------------------------------------
    def _report_cv(self, block: Stage3Block, best_params: dict) -> CVReport:
        cfg = self.config
        return CVEvaluator(
            self.estimator, cfg.tuning_decision_threshold, cfg.random_state,
            block.feature_fit_scope, cfg.verbose,
        ).run(block.report_folds, best_params)

    # ------------------------------------------------------------------
    def _refit(self, block: Stage3Block, best_params: dict) -> None:
        self._say("\n  refitting on the full training split")
        self.model_ = self.estimator.fit(best_params, block.X_train, block.y_train,
                                         sample_weight=block.weights.vector)

    # ------------------------------------------------------------------
    def _out_of_fold(self, block: Stage3Block, best_params: dict) -> OOFPredictions:
        return OOFGenerator(self.estimator, block.tune_plan, self.config.verbose).generate(
            block.tune_folds, best_params, score_test=True)

    # ------------------------------------------------------------------
    def _calibrate(self, block: Stage3Block, oof: OOFPredictions) -> None:
        cfg = self.config
        self.calibrator_ = Stage3Calibrator(
            cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins,
            cfg.force_calibration, cfg.random_state,
        ).fit(oof.proba, block.y_train, block.tune_plan)
        self._say("\n  " + self.calibrator_.report_.describe().replace("\n", "\n  "))

    # ------------------------------------------------------------------
    def _select_thresholds(self, block: Stage3Block, oof: OOFPredictions):
        y_tr = block.y_train
        choice = self.selector.select(
            y_tr, oof.proba, selected_on="out-of-fold training predictions")
        choice_cal = self.selector.select(
            y_tr, self.calibrator_.oof_calibrated_,
            selected_on="out-of-fold training predictions (calibrated space)",
            keep_sweep=False)
        nested = self.selector.select_nested(y_tr, oof.proba, block.tune_plan)
        self._say(f"\n  operating point: {choice.describe()}")
        self._say(f"  fold-nested thresholds: {nested['by_fold']} (sd {nested['std']:.4f})")
        return choice, choice_cal, nested

    # ------------------------------------------------------------------
    def _evaluate(self, block: Stage3Block, oof: OOFPredictions,
                  choice, choice_cal, nested: dict) -> dict:
        """Score train (in-sample) / test / fold-mean; decisions; metrics."""
        cfg, est, cal = self.config, self.estimator, self.calibrator_
        y_tr, y_te = block.y_train, block.y_test

        train_proba_ins = est.positive_proba(self.model_, block.X_train)
        test_proba = est.positive_proba(self.model_, block.X_test)
        test_proba_cal = cal.transform(test_proba)
        train_proba_cal_ins = cal.transform(train_proba_ins)
        fm_proba = oof.test_proba_fold_mean
        fm_proba_cal = cal.transform(fm_proba)

        t = choice.threshold
        t_train = nested["per_row"] if cfg.nested_oof_threshold else t
        oof_dec = decision_block(oof.proba, t_train)
        test_dec = decision_block(test_proba, t)
        fm_dec = decision_block(fm_proba, t)
        oracle = self.selector.oracle_on_test(y_te, test_proba)

        metrics = stage3_metric_suite(
            y_train=y_tr, y_test=y_te,
            oof_proba=oof.proba, oof_proba_calibrated=cal.oof_calibrated_,
            threshold=t, threshold_calibrated=choice_cal.threshold,
            nested=nested,
            test_proba=test_proba, test_proba_calibrated=test_proba_cal,
            fold_mean_proba=fm_proba, fold_mean_proba_calibrated=fm_proba_cal,
            reference_threshold=cfg.reference_threshold,
            oracle_threshold=oracle.threshold,
            ece_bins=cfg.ece_bins,
        )
        calib = calibration_diagnostics(
            cal.report_, y_te, test_proba, test_proba_cal, cfg.ece_bins)

        serving = Stage3ServingModel(
            block.pipeline, block.cleaner, self.model_,
            cal.final_calibrator_, est.as_array, block.features,
        )
        return {
            "train_proba_ins": train_proba_ins,
            "train_proba_cal_ins": train_proba_cal_ins,
            "test_proba": test_proba,
            "test_proba_cal": test_proba_cal,
            "fm_proba": fm_proba,
            "fm_proba_cal": fm_proba_cal,
            "t": t,
            "t_train": t_train,
            "oof_dec": oof_dec,
            "test_dec": test_dec,
            "fm_dec": fm_dec,
            "oracle": oracle,
            "metrics": metrics,
            "calib": calib,
            "serving": serving,
        }

    # ------------------------------------------------------------------
    def _build_artifact(self, block: Stage3Block, best_params: dict,
                        cv_score: float, tuning_dict: dict, cv_report: CVReport,
                        oof: OOFPredictions, choice, choice_cal, nested: dict,
                        s: dict, extras: Optional[dict]) -> Stage3Artifact:
        cfg, est = self.config, self.estimator
        y_tr, y_te = block.y_train, block.y_test
        t, t_train = s["t"], s["t_train"]
        oof_dec, test_dec, fm_dec = s["oof_dec"], s["test_dec"], s["fm_dec"]
        return Stage3Artifact(
            stage=cfg.stage,
            model_family=cfg.model_family,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config=cfg.to_dict(),
            mirror_report=cfg.mirror_report(),
            column_prefix=cfg.column_prefix,
            population=cfg.population,
            features=list(block.features),
            index_train=list(block.index_train),
            index_test=list(block.index_test),
            y_train=[int(v) for v in y_tr],
            y_test=[int(v) for v in y_te],
            split_fingerprint=block.split_fingerprint,
            block_fingerprint=block.block_fingerprint,
            feature_fit_scope=block.feature_fit_scope,
            block_settings=dict(block.settings),
            cleaning_report=block.cleaner.report_,
            fold_plan=block.tune_plan.to_dict(),
            report_fold_plan={k: v for k, v in block.report_plan.to_dict().items()
                              if k != "fold_id"},
            class_weights=block.weights.to_dict(),
            tuning=tuning_dict,
            cv_report=cv_report.to_dict(),
            best_params=best_params,
            cv_score=cv_score,
            oof={
                "description": "Out-of-fold over the TRAINING split; per-fold "
                               "features and models. Stage 4 trains on this.",
                "proba": oof.proba,
                "proba_calibrated": self.calibrator_.oof_calibrated_,
                "fold_id": oof.fold_id,
                "threshold": t,
                "threshold_per_row": np.broadcast_to(
                    np.asarray(t_train, dtype=float), oof.proba.shape).copy(),
                "summary": oof.summary(),
                **oof_dec,
            },
            refit={
                "description": "Full-train refit. test_* is held-out; train_* "
                               "is IN-SAMPLE — never fit anything on it.",
                "model": self.model_,
                "calibrator": self.calibrator_.final_calibrator_,
                "cleaner": block.cleaner,
                "feature_pipeline": block.pipeline,
                "serving_model": s["serving"],
                "as_array": bool(est.as_array),
                "estimator": est.to_dict(),
                "train_proba": s["train_proba_ins"],
                "train_proba_calibrated": s["train_proba_cal_ins"],
                "test_proba": s["test_proba"],
                "test_proba_calibrated": s["test_proba_cal"],
                "test_decision": test_dec["decision"],
                "test_margin": test_dec["margin"],
                "test_confidence": test_dec["confidence"],
                "test_state": test_dec["state"],
            },
            fold_mean={
                "description": "Test split scored by each OOF fold model "
                               "through its own fold pipeline, averaged.",
                "test_proba": s["fm_proba"],
                "test_proba_calibrated": s["fm_proba_cal"],
                "test_proba_by_fold": oof.test_proba_by_fold,
                "test_decision": fm_dec["decision"],
                "test_margin": fm_dec["margin"],
                "test_confidence": fm_dec["confidence"],
                "test_state": fm_dec["state"],
            },
            threshold={
                "oof": choice.to_dict(),
                "oof_calibrated_space": choice_cal.to_dict(),
                "oof_nested": {
                    **{k: v for k, v in nested.items() if k != "per_row"},
                    "drives_train_decisions": bool(cfg.nested_oof_threshold),
                },
                "test_oracle": {
                    **s["oracle"].to_dict(),
                    "warning": "Reference only — the pre-PR-33 GLASS rule of "
                               "fitting the threshold on y_test. Nothing "
                               "downstream consumes this value.",
                },
                "sweep": choice.sweep,
            },
            calibration=s["calib"],
            metrics=s["metrics"],
            extras=dict(extras or {}),
        )
