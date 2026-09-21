"""
blackbox_pipeline.models.xgb.stage
===================================
Stage 3 orchestrator.

    artifact, path = train_xgb_stage(GLOBAL_SPLIT, engineered=STAGE3_DATA)

Order of operations, and why it is this order
---------------------------------------------
1.  contract + clean   the two arms must see identical columns and values
2.  weights + folds    computed once, reused by every later phase
3.  tune               Optuna, train-only, F2 at the 0.5 cut
4.  cv report          the EBM's 10-fold summary block
5.  refit              one model on the full training split
6.  OOF                one booster per fold; this is Stage 4's input
7.  calibrate          gate on OOF ECE, fit nested-OOF isotonic
8.  threshold          F2 grid on OOF predictions
9.  evaluate           the test split is touched HERE and nowhere earlier
10. persist

Steps 2–8 read the training split only. Before step 9, ``X_test`` is only
column-aligned and imputed with TRAIN medians, and ``y_test`` only
validated; step 9 is the first time either reaches a model or decision rule.

Supplying ``params`` skips step 3 (``cv_score`` is then NaN).

Step 8 also picks one threshold per fold without that fold's labels. With
``nested_oof_threshold=True`` these drive the train-side decision columns, so
Stage 4's inputs are fold-nested throughout. ``oof_at_operating_threshold``
is in-selection (mildly optimistic); ``oof_at_nested_thresholds`` is its
honest counterpart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from .artifacts import Stage3Artifact, Stage3ArtifactStore, _decision_block
from .calibration import Stage3Calibrator
from .config import XGBStage3Config
from .estimator import XGBFactory
from .evaluation.cv import CVEvaluator
from .evaluation.metrics import Stage3Metrics
from .features import FeatureCleaner, Stage3FeatureContract
from .folds import FoldPlan
from .oof import OOFGenerator
from .thresholds import F2ThresholdSelector
from .weights import BalancedWeights

__all__ = ["XGBStage3Pipeline", "train_xgb_stage"]


class XGBStage3Pipeline:
    """Stage 3 XGBoost — the counterpart of ``glass_pipeline.ebm``."""

    def __init__(self, config: Optional[XGBStage3Config] = None):
        self.config = config or XGBStage3Config()
        self.factory = XGBFactory(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.contract = Stage3FeatureContract(self.config.expected_features)
        self.selector = F2ThresholdSelector(
            beta=self.config.beta,
            low=self.config.threshold_low,
            high=self.config.threshold_high,
            steps=self.config.threshold_steps,
        )

        self.is_fitted = False
        self.artifact_: Optional[Stage3Artifact] = None
        self.folds_: Optional[FoldPlan] = None
        self.weights_: Optional[BalancedWeights] = None
        self.cleaner_: Optional[FeatureCleaner] = None
        self.calibrator_: Optional[Stage3Calibrator] = None
        self.model_: Optional[Any] = None

    # ==================================================================
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_test: pd.DataFrame,
        y_test,
        params: Optional[dict[str, Any]] = None,
        split_fingerprint: Optional[str] = None,
        params_source: Optional[str] = None,
    ) -> Stage3Artifact:
        cfg = self.config
        say = print if cfg.verbose else (lambda *a, **k: None)

        # ---- 1. contract + hygiene ------------------------------------
        X_tr, X_te, feature_order = self.contract.align(X_train, X_test)
        y_tr = self._as_series(y_train, X_tr, "y_train")
        y_te = self._as_series(y_test, X_te, "y_test")
        if cfg.expected_features is None:
            say("  ⚠️  config.expected_features is None — the EBM feature "
                "contract is NOT being enforced. Pass EBM_FEATURES.")

        self.cleaner_ = FeatureCleaner(cfg.clean_inf, cfg.impute_missing)
        X_tr = self.cleaner_.fit_transform(X_tr)
        X_te = self.cleaner_.transform(X_te)

        say("\n" + "=" * 78)
        say(f"  STAGE 3 — XGBOOST  (counterpart of {cfg.counterpart})")
        say("=" * 78)
        say(f"  train {len(X_tr):,} rows | test {len(X_te):,} rows | "
            f"{len(feature_order)} features")

        # ---- 2. weights + folds ---------------------------------------
        self.weights_ = BalancedWeights.balanced(y_tr, cfg.class_weight)
        self.folds_ = FoldPlan.build(
            y_tr, cfg.n_tune_folds, cfg.random_state, cfg.stratify,
            purpose="tuning+oof+calibration+threshold",
        )
        say(f"  {self.weights_.describe()}")
        kind = "stratified" if self.folds_.stratified else "unstratified"
        say(f"  fold plan: {self.folds_.n_splits}-fold {kind}, "
            f"seed {self.folds_.random_state} (shared by tuning, OOF, "
            f"calibration and threshold selection)")

        # ---- 3. tuning -------------------------------------------------
        if params is None:
            from .tuning import XGBTuner
            tuning = XGBTuner(cfg, self.factory, self.weights_,
                              self.folds_).tune(X_tr, y_tr)
            best_params, cv_score = tuning.best_params, tuning.best_value
            tuning_dict = tuning.to_dict()
        else:
            best_params = XGBFactory.normalize_params(params)
            cv_score = float("nan")
            tuning_dict = {
                "best_params": best_params,
                "source": "supplied — Optuna not run"
                          + (f" ({params_source})" if params_source else ""),
            }
            say(f"\n  using supplied hyperparameters (no search): {best_params}")

        # ---- 4. k-fold report ------------------------------------------
        cv_report = CVEvaluator(
            self.factory, self.weights_, cfg.n_eval_folds, cfg.random_state,
            cfg.stratify, cfg.tuning_decision_threshold, cfg.verbose,
        ).run(X_tr, y_tr, best_params)

        # ---- 5. refit on the full training split -----------------------
        say("\n  refitting on the full training split")
        self.model_ = self.factory.fit(
            best_params, X_tr, y_tr, sample_weight=self.weights_.vector
        )

        # ---- 6. OOF — Stage 4's input ----------------------------------
        oof = OOFGenerator(
            self.factory, self.weights_, self.folds_, cfg.verbose
        ).generate(X_tr, y_tr, best_params)

        # ---- 7. calibration, gated and fitted on OOF only --------------
        self.calibrator_ = Stage3Calibrator(
            cfg.calibration_method, cfg.ece_threshold, cfg.ece_bins,
            cfg.force_calibration, cfg.random_state,
        ).fit(oof.proba, y_tr, self.folds_)
        say("\n  " + self.calibrator_.report_.describe().replace("\n", "\n  "))

        # ---- 8. operating point, from OOF ------------------------------
        choice = self.selector.select(
            y_tr, oof.proba,
            selected_on="out-of-fold training predictions",
        )
        choice_cal = self.selector.select(
            y_tr, self.calibrator_.oof_calibrated_,
            selected_on="out-of-fold training predictions (calibrated space)",
            keep_sweep=False,
        )
        say(f"\n  operating point: {choice.describe()}")

        # Train-side decisions: per-fold thresholds chosen without that
        # fold's labels (Stage 4 trains on these columns).
        nested = self.selector.select_nested(y_tr, oof.proba, self.folds_)
        say(f"  fold-nested thresholds: {nested['by_fold']} "
            f"(sd {nested['std']:.4f})")

        # ---- 9. the test split, touched for the first time -------------
        train_proba_insample = XGBFactory.positive_proba(self.model_, X_tr)
        test_proba = XGBFactory.positive_proba(self.model_, X_te)
        test_proba_cal = self.calibrator_.transform(test_proba)
        train_proba_cal_insample = self.calibrator_.transform(
            train_proba_insample
        )

        t = choice.threshold
        t_train = nested["per_row"] if cfg.nested_oof_threshold else t
        oof_dec = _decision_block(oof.proba, t_train)
        test_dec = _decision_block(test_proba, t)

        oracle = self.selector.oracle_on_test(y_te, test_proba)

        metrics = {
            # In-selection: t was chosen on these same OOF rows.
            "oof_at_operating_threshold": Stage3Metrics.compute(
                y_tr, oof.proba, t, split="train (OOF)",
                ece_bins=cfg.ece_bins).to_dict(),
            # Honest counterpart: each fold's decisions at a threshold chosen
            # without its labels ("threshold" = mean of per-fold values).
            "oof_at_nested_thresholds": Stage3Metrics.compute(
                y_tr, oof.proba, nested["mean"],
                split="train (OOF, fold-nested thresholds)",
                ece_bins=cfg.ece_bins,
                decision=(oof.proba >= nested["per_row"]).astype(int),
            ).to_dict(),
            "test_at_operating_threshold": Stage3Metrics.compute(
                y_te, test_proba, t, split="test",
                ece_bins=cfg.ece_bins).to_dict(),
            "test_at_half": Stage3Metrics.compute(
                y_te, test_proba, cfg.reference_threshold, split="test",
                ece_bins=cfg.ece_bins).to_dict(),
            "test_calibrated_at_operating_threshold": Stage3Metrics.compute(
                y_te, test_proba_cal, choice_cal.threshold, split="test",
                probability_space="calibrated",
                ece_bins=cfg.ece_bins).to_dict(),
            "test_at_oracle_threshold_REFERENCE_ONLY": Stage3Metrics.compute(
                y_te, test_proba, oracle.threshold, split="test",
                ece_bins=cfg.ece_bins).to_dict(),
        }

        # ---- 10. assemble ----------------------------------------------
        artifact = Stage3Artifact(
            stage=cfg.stage,
            model_family=cfg.model_family,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config=cfg.to_dict(),
            mirror_report=cfg.mirror_report(),
            features=list(feature_order),
            index_train=list(X_tr.index),
            index_test=list(X_te.index),
            split_fingerprint=split_fingerprint,
            cleaning_report=self.cleaner_.report_,
            fold_plan=self.folds_.to_dict(),
            class_weights=self.weights_.to_dict(),
            tuning=tuning_dict,
            cv_report=cv_report.to_dict(),
            best_params=best_params,
            cv_score=cv_score,
            oof={
                "description": (
                    "Out-of-fold over the TRAINING split. Every row scored by a "
                    "booster that did not train on it. Stage 4 trains on this."
                ),
                "proba": oof.proba,
                "proba_calibrated": self.calibrator_.oof_calibrated_,
                "fold_id": oof.fold_id,
                "threshold": t,
                # threshold actually behind decision/margin/confidence below
                "threshold_per_row": np.broadcast_to(
                    np.asarray(t_train, dtype=float), oof.proba.shape
                ).copy(),
                "summary": oof.summary(),
                **oof_dec,
            },
            refit={
                "description": (
                    "Single model refitted on the full training split. "
                    "test_* is the held-out evaluation; train_* is IN-SAMPLE "
                    "and exists only for parity with the GLASS artifact — "
                    "never fit anything on it."
                ),
                "model": self.model_,
                "calibrator": self.calibrator_.final_calibrator_,
                "cleaner": self.cleaner_,
                "train_proba": train_proba_insample,
                "train_proba_calibrated": train_proba_cal_insample,
                "test_proba": test_proba,
                "test_proba_calibrated": test_proba_cal,
                "test_decision": test_dec["decision"],
                "test_margin": test_dec["margin"],
                "test_confidence": test_dec["confidence"],
                "test_state": test_dec["state"],
            },
            threshold={
                "oof": choice.to_dict(),
                "oof_calibrated_space": choice_cal.to_dict(),
                "oof_nested": {
                    **{k: v for k, v in nested.items() if k != "per_row"},
                    "drives_train_decisions": bool(cfg.nested_oof_threshold),
                },
                "test_oracle": {
                    **oracle.to_dict(),
                    "warning": (
                        "Reference only. Reproduces the GLASS rule of fitting "
                        "the threshold on y_test. Nothing downstream consumes "
                        "this value."
                    ),
                },
                "sweep": choice.sweep,
            },
            calibration=self.calibrator_.report_.to_dict(),
            metrics=metrics,
        )

        self.artifact_ = artifact
        self.is_fitted = True
        if cfg.verbose:
            artifact.describe()
        return artifact

    # ==================================================================
    def save(self, verbose: bool = True) -> str:
        if not self.is_fitted:
            raise ValueError("Call fit() first")
        store = Stage3ArtifactStore(
            self.config.artifact_dir, self.config.artifact_stem,
            self.config.timestamped, self.config.write_latest_pointer,
        )
        return store.save(self.artifact_, verbose=verbose)

    # ==================================================================
    @staticmethod
    def _as_series(y, X: pd.DataFrame, name: str) -> pd.Series:
        """Positional alignment guard — the Stage 2 rule, applied here too."""
        if isinstance(y, pd.Series):
            if len(y) != len(X):
                raise ValueError(
                    f"{name} length {len(y)} != {len(X)} feature rows"
                )
            if not X.index.equals(y.index):
                raise ValueError(
                    f"{name}.index differs from its feature frame. Reindex "
                    f"before calling ({name} = {name}.loc[X.index])."
                )
            out = y
        else:
            arr = np.asarray(y)
            if len(arr) != len(X):
                raise ValueError(
                    f"{name} length {len(arr)} != {len(X)} feature rows"
                )
            out = pd.Series(arr, index=X.index)

        values = np.asarray(out)
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must be binary 0/1")
        return out.astype(int)


# ======================================================================
def train_xgb_stage(
    GLOBAL_SPLIT: dict,
    engineered: Optional[dict] = None,
    config: Optional[XGBStage3Config] = None,
    params: Optional[dict[str, Any]] = None,
    split_fingerprint: Optional[str] = None,
    save: bool = True,
    params_source: Optional[str] = None,
) -> tuple[Stage3Artifact, Optional[str]]:
    """
    Notebook entry point, shaped like ``train_ebm_stage(GLOBAL_SPLIT)``.

    Parameters
    ----------
    GLOBAL_SPLIT
        ``X_train``, ``X_test``, ``y_train``, ``y_test``.
    engineered
        ``{"X_stage3_train": ..., "X_stage3_test": ...}`` — the frames produced
        by the GLASS Stage 3 DAG (``drop_leaky_features`` →
        ``engineer_ebm_features`` → ``select_ebm_features``). Stage 3 does not
        engineer features itself; both arms must consume one run of that DAG.
        When omitted, ``GLOBAL_SPLIT``'s raw frames are used, which is only
        correct if they have already been engineered.
    config
        ``XGBStage3Config``; defaults to ``XGBStage3Config()``.
    params
        Skip the Optuna search and use these hyperparameters. Must contain
        exactly ``estimator.TUNED_PARAM_NAMES``. Mirrors the notebook's
        ``*_PARAM_SOURCE = "explicit"`` path.
    split_fingerprint
        Recorded in the artifact for provenance; not verified here.
    params_source
        Free-text provenance for supplied ``params`` (e.g. the artifact they
        came from); recorded in ``artifact.tuning["source"]``.
    save
        Persist via ``Stage3ArtifactStore`` under ``config.artifact_dir``.

    Returns
    -------
    ``(Stage3Artifact, path or None)``.

    Raises ``KeyError`` on missing split / engineered keys and ``ValueError``
    when engineered frames differ from ``GLOBAL_SPLIT`` in row count or index.
    """
    required = ("X_train", "X_test", "y_train", "y_test")
    missing = [k for k in required if k not in GLOBAL_SPLIT]
    if missing:
        raise KeyError(f"GLOBAL_SPLIT missing keys: {missing}")

    if engineered is not None:
        for key in ("X_stage3_train", "X_stage3_test"):
            if key not in engineered:
                raise KeyError(f"engineered dict missing key '{key}'")
        X_tr, X_te = engineered["X_stage3_train"], engineered["X_stage3_test"]
        if len(X_tr) != len(GLOBAL_SPLIT["X_train"]) or \
           len(X_te) != len(GLOBAL_SPLIT["X_test"]):
            raise ValueError(
                "Engineered Stage 3 frames do not match the split row counts."
            )
        if not X_tr.index.equals(GLOBAL_SPLIT["X_train"].index) or \
           not X_te.index.equals(GLOBAL_SPLIT["X_test"].index):
            raise ValueError(
                "Engineered Stage 3 frames carry a different index from "
                "GLOBAL_SPLIT. Feature rows and label rows must line up."
            )
    else:
        X_tr, X_te = GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["X_test"]

    pipeline = XGBStage3Pipeline(config)
    artifact = pipeline.fit(
        X_tr, GLOBAL_SPLIT["y_train"], X_te, GLOBAL_SPLIT["y_test"],
        params=params, split_fingerprint=split_fingerprint,
        params_source=params_source,
    )
    path = pipeline.save() if save else None
    return artifact, path