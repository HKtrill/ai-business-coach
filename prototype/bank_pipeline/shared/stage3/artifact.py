"""
shared.stage3.artifact
======================
The canonical Stage 3 payload (``Stage3Artifact``) with its Stage 4 frame /
metadata interface, served-population evaluation, ``StageOutput`` export and
legacy GLASS schema; plus persistence (``Stage3ArtifactStore``) and
fingerprint-matched lookup (``find_stage3_artifact``).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .estimator import CalibratedModel
from .evaluation import Stage3Metrics


STAGE4_FEATURE_SUFFIXES = ("proba", "proba_calibrated", "decision",
                           "confidence", "margin")


TEST_SOURCES = ("refit", "fold_mean")


@dataclass
class Stage3Artifact:
    """Canonical Stage 3 payload (EBM and XGBoost alike)."""

    # ---- identity ---------------------------------------------------------
    stage: str
    model_family: str
    created_at: str
    config: dict
    mirror_report: dict
    column_prefix: str = "stage3_"
    population: str = "full_train"

    # ---- input contract ---------------------------------------------------
    features: list[str] = field(default_factory=list)
    index_train: list = field(default_factory=list)
    index_test: list = field(default_factory=list)
    y_train: list = field(default_factory=list)
    y_test: list = field(default_factory=list)
    split_fingerprint: Optional[str] = None
    block_fingerprint: Optional[str] = None
    feature_fit_scope: str = "per_fold"
    block_settings: dict = field(default_factory=dict)
    cleaning_report: dict = field(default_factory=dict)

    # ---- partition --------------------------------------------------------
    fold_plan: dict = field(default_factory=dict)
    report_fold_plan: dict = field(default_factory=dict)
    class_weights: dict = field(default_factory=dict)

    # ---- search -----------------------------------------------------------
    tuning: dict = field(default_factory=dict)
    cv_report: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    cv_score: float = float("nan")

    # ---- predictions ------------------------------------------------------
    oof: dict = field(default_factory=dict)
    refit: dict = field(default_factory=dict)
    fold_mean: dict = field(default_factory=dict)

    # ---- operating point + calibration ------------------------------------
    threshold: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    # ---- metrics + family extras -----------------------------------------
    metrics: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    # ==================================================================
    # Stage 4 interface
    # ==================================================================
    def _cols(self, block: dict, key_map: dict) -> dict:
        p = self.column_prefix
        return {f"{p}{suffix}": np.asarray(block[key])
                for suffix, key in key_map.items()}

    def stage4_frame(self) -> pd.DataFrame:
        """Training-side Stage 4 FEATURES. OOF, leakage-safe."""
        keys = {s: s for s in STAGE4_FEATURE_SUFFIXES}
        return pd.DataFrame(self._cols(self.oof, keys),
                            index=pd.Index(self.index_train))

    def stage4_metadata(self) -> pd.DataFrame:
        """Training-side bookkeeping. NOT features."""
        p = self.column_prefix
        return pd.DataFrame(
            {
                f"{p}fold_id": np.asarray(self.oof["fold_id"]),
                f"{p}threshold": np.asarray(self.oof["threshold_per_row"]),
                f"{p}state": np.asarray(self.oof["state"], dtype=object),
            },
            index=pd.Index(self.index_train),
        )

    def _test_block(self, source: str) -> dict:
        if source not in TEST_SOURCES:
            raise ValueError(f"source must be one of {TEST_SOURCES}")
        return self.refit if source == "refit" else self.fold_mean

    def stage4_test_frame(self, source: str = "refit") -> pd.DataFrame:
        """
        Test-side counterpart of ``stage4_frame()``; used to EVALUATE Stage 4.

        ``source="refit"``     the full-train model (default)
        ``source="fold_mean"`` mean of the 5 OOF fold models
        """
        b = self._test_block(source)
        keys = {
            "proba": "test_proba",
            "proba_calibrated": "test_proba_calibrated",
            "decision": "test_decision",
            "confidence": "test_confidence",
            "margin": "test_margin",
        }
        return pd.DataFrame(self._cols(b, keys), index=pd.Index(self.index_test))

    def stage4_test_metadata(self, source: str = "refit") -> pd.DataFrame:
        b = self._test_block(source)
        p = self.column_prefix
        n = len(b["test_proba"])
        return pd.DataFrame(
            {
                f"{p}fold_id": np.full(n, -1),
                f"{p}threshold": np.full(n, float(self.threshold["oof"]["threshold"])),
                f"{p}state": np.asarray(b["test_state"], dtype=object),
                f"{p}source": np.full(n, source, dtype=object),
            },
            index=pd.Index(self.index_test),
        )

    # ==================================================================
    # Cascade population
    # ==================================================================
    def evaluate_served(
        self,
        y_test,
        served_mask,
        source: str = "refit",
        space: str = "raw",
        label: str = "served",
    ) -> dict:
        """
        Stage 3 metrics on the test rows Stage 3 actually serves.

        ``served_mask``: boolean, aligned to ``index_test`` (a Series is
        reindexed by label and must cover every test row).
        ``space``: ``"raw"`` uses the OOF threshold, ``"calibrated"`` the OOF
        calibrated-space threshold. ``coverage`` = served / all test rows.
        """

        idx = pd.Index(self.index_test)
        if isinstance(served_mask, pd.Series):
            if not served_mask.index.isin(idx).all() or len(served_mask) != len(idx):
                raise ValueError("served_mask index must match index_test exactly")
            mask = served_mask.reindex(idx).to_numpy(dtype=bool)
        else:
            mask = np.asarray(served_mask, dtype=bool)
            if len(mask) != len(idx):
                raise ValueError(f"served_mask has {len(mask)} rows, test has {len(idx)}")
        y = pd.Series(np.asarray(y_test), index=idx) if not isinstance(y_test, pd.Series) \
            else y_test.reindex(idx)
        b = self._test_block(source)
        if space == "raw":
            p, t = np.asarray(b["test_proba"]), self.threshold["oof"]["threshold"]
        elif space == "calibrated":
            p = np.asarray(b["test_proba_calibrated"])
            t = self.threshold["oof_calibrated_space"]["threshold"]
        else:
            raise ValueError("space must be 'raw' or 'calibrated'")
        if not mask.any():
            raise ValueError("served_mask selects no rows")
        m = Stage3Metrics.compute(
            y.to_numpy()[mask], p[mask], t,
            split=f"test ({label}, {source})", probability_space=space,
            n_population=len(idx),
        )
        return m.to_dict()

    # ==================================================================
    # Cross-stage contract
    # ==================================================================
    def to_stage_output(self):
        """
        The ``shared.stage_io.StageOutput`` Stages 1–2 already hand to Stage 4.

        Carries the CALIBRATED probabilities (the contract's definition):
        train side = fold-nested calibrated OOF, test side = refit model
        through the all-OOF calibrator, with the calibrated-space OOF
        threshold. ``threshold_source='in_stage_f2_cv'``: that scalar was
        chosen on every training row's OOF labels, so ``StageOutput`` will
        refuse train-side decisions from it — use ``stage4_frame()``'s
        fold-nested decision columns for those.
        """
        from shared.stage_io import StageOutput

        itr, ite = pd.Index(self.index_train), pd.Index(self.index_test)
        c = self.calibration
        cfg = self.config
        cal_thr = self.threshold["oof_calibrated_space"]
        return StageOutput(
            stage=self.stage,
            arm=cfg.get("arm", self.model_family),
            model=self.model_family,
            feature_names=list(self.features),
            train_proba_oof=pd.Series(np.asarray(self.oof["proba_calibrated"], float),
                                      index=itr, name=f"{self.stage}_proba_oof"),
            test_proba=pd.Series(np.asarray(self.refit["test_proba_calibrated"], float),
                                 index=ite, name=f"{self.stage}_proba"),
            y_train=pd.Series(np.asarray(self.y_train, int), index=itr, name="y"),
            y_test=pd.Series(np.asarray(self.y_test, int), index=ite, name="y"),
            train_fold_id=pd.Series(np.asarray(self.oof["fold_id"], int),
                                    index=itr, name="fold_id"),
            threshold=float(cal_thr["threshold"]),
            threshold_source="in_stage_f2_cv",
            calibration_method=cfg.get("calibration_method") if c.get("applied") else "none",
            calibration_requested=cfg.get("calibration_method", "isotonic"),
            oof_provenance=(
                f"refittable (per-fold refit: features {self.feature_fit_scope}, "
                f"model, calibration map; {self.fold_plan.get('n_splits')}-fold "
                f"stratified, seed {self.fold_plan.get('random_state')})"
            ),
            best_params=dict(self.best_params),
            best_cv_score=None if not np.isfinite(self.cv_score) else float(self.cv_score),
            cv_f2=float(cal_thr["score"]),
            threshold_sweep=None,
            calibration_metrics=dict(c),
            metrics_test=dict(self.metrics.get("test_calibrated_at_operating_threshold", {})),
            config={
                **cfg,
                "threshold_grid": [cfg.get("threshold_low"), cfg.get("threshold_high"),
                                   cfg.get("threshold_steps")],
                "block_fingerprint": self.block_fingerprint,
                "stage3_split_fingerprint": self.split_fingerprint,
            },
        )

    # ==================================================================
    # Serving + legacy schema
    # ==================================================================
    @property
    def serving_model(self):
        """Raw rows → probabilities (fitted pipeline + cleaner + model + calibrator)."""
        return self.refit.get("serving_model")

    def to_glass_schema(self) -> dict:
        """
        Pre-PR-33 GLASS key names, for comparison / Venn-tracing code.

        ``train_predictions`` is the IN-SAMPLE refit column, because that is
        what the old key meant. Anything that FITS on Stage 3 output must use
        ``stage4_frame()`` instead. ``calibrated_model`` is a model callable on
        engineered features. ``ece`` is the OOF ECE before calibration.
        """
        cal_model = CalibratedModel(
            self.refit.get("model"), self.refit.get("calibrator"),
            self.refit.get("as_array", False), self.features,
        )
        return {
            "model": self.refit.get("model"),
            "calibrated_model": cal_model,
            "train_predictions": self.refit.get("train_proba"),
            "test_predictions": self.refit.get("test_proba"),
            "train_predictions_calibrated": self.refit.get("train_proba_calibrated"),
            "test_predictions_calibrated": self.refit.get("test_proba_calibrated"),
            "optimal_threshold": self.threshold.get("oof", {}).get("threshold"),
            "optimal_threshold_space": "raw",
            "metrics": self.metrics.get("test_at_operating_threshold"),
            "metrics_at_half": self.metrics.get("test_at_half"),
            "ece": self.calibration.get("ece_before"),
            "ece_semantics": "OOF ECE before calibration",
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "features": self.features,
            "interactions": self.extras.get("interactions"),
            "split_fingerprint": self.split_fingerprint,
            "index_test": list(self.index_test),
        }

    # ==================================================================
    def summary(self) -> dict:
        """Small, JSON-serialisable sidecar. No arrays, no models."""
        return {
            "stage": self.stage,
            "model_family": self.model_family,
            "created_at": self.created_at,
            "population": self.population,
            "n_features": len(self.features),
            "features": self.features,
            "split_fingerprint": self.split_fingerprint,
            "block_fingerprint": self.block_fingerprint,
            "feature_fit_scope": self.feature_fit_scope,
            "protocol": {k: v for k, v in self.config.items()
                         if k not in ("verbose", "show_progress_bar")},
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "tuning": {k: v for k, v in self.tuning.items() if k != "top_trials"},
            "cv_report": self.cv_report.get("aggregate", {}),
            "threshold": {k: v for k, v in self.threshold.items() if k != "sweep"},
            "calibration": self.calibration,
            "class_weights": self.class_weights,
            "fold_plan": {k: v for k, v in self.fold_plan.items() if k != "fold_id"},
            "metrics": self.metrics,
            "mirror_report": self.mirror_report,
            "cleaning_report": self.cleaning_report,
            "extras": self.extras,
        }

    def describe(self) -> None:
        print("=" * 78)
        print(f"  STAGE 3 — {self.model_family.upper()}  ({self.created_at})")
        print("=" * 78)
        print(f"  features        : {len(self.features)} "
              f"(fitted {self.feature_fit_scope})")
        print(f"  best CV F2      : {self.cv_score:.6f}")
        print(f"  operating thresh: {self.threshold['oof']['threshold']:.4f} "
              f"({self.threshold['oof']['selected_on']})")
        if "test_oracle" in self.threshold:
            print(f"  test-oracle th. : {self.threshold['test_oracle']['threshold']:.4f} "
                  f"(reference only)")
        c = self.calibration
        print(f"  calibration     : {'applied' if c.get('applied') else 'skipped'} "
              f"— {c.get('reason')}")
        print(f"  ECE  OOF {c.get('ece_before', float('nan')):.4f} → "
              f"{c.get('ece_after', float('nan')):.4f}  |  test "
              f"{c.get('test_ece_raw', float('nan')):.4f} → "
              f"{c.get('test_ece_calibrated', float('nan')):.4f}")
        for name in ("oof_at_nested_thresholds", "test_at_operating_threshold",
                     "test_fold_mean_at_operating_threshold", "test_at_half"):
            m = self.metrics.get(name)
            if m:
                print(f"  {name:<38} F2={m['f2']:.4f}  rec={m['recall']:.4f}  "
                      f"prec={m['precision']:.4f}  AUC={m['roc_auc']:.4f}")
        print("=" * 78)


class Stage3ArtifactStore:
    """
    ``save`` writes ``<stem>_<timestamp>.joblib`` + a JSON sidecar, and (if
    timestamped) a full ``<stem>_latest.joblib`` copy.
    """

    def __init__(self, base_path: str = "models/stage3", stem: str = "stage3",
                 timestamped: bool = True, write_latest_pointer: bool = True):
        self.base_path = base_path
        self.stem = stem
        self.timestamped = bool(timestamped)
        self.write_latest_pointer = bool(write_latest_pointer)

    def save(self, artifact: Stage3Artifact, verbose: bool = True) -> str:
        os.makedirs(self.base_path, exist_ok=True)
        if self.timestamped:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"{self.stem}_{ts}.joblib"
        else:
            name = f"{self.stem}.joblib"
        path = os.path.join(self.base_path, name)
        joblib.dump(artifact, path)
        side = path[:-len(".joblib")] + ".json"
        with open(side, "w") as fh:
            json.dump(artifact.summary(), fh, indent=2, default=str)
        if self.write_latest_pointer and self.timestamped:
            joblib.dump(artifact, os.path.join(self.base_path,
                                               f"{self.stem}_latest.joblib"))
        if verbose:
            print(f"\n  saved Stage 3 artifact → {path}")
            print(f"        summary sidecar  → {side}")
        return path

    @staticmethod
    def load(path: str) -> Stage3Artifact:
        return joblib.load(path)

    def load_latest(self) -> Stage3Artifact:
        latest = os.path.join(self.base_path, f"{self.stem}_latest.joblib")
        if os.path.exists(latest):
            return self.load(latest)
        candidates = sorted(
            f for f in os.listdir(self.base_path)
            if f.startswith(self.stem) and f.endswith(".joblib")
        )
        if not candidates:
            raise FileNotFoundError(f"no {self.stem}*.joblib under {self.base_path}")
        return self.load(os.path.join(self.base_path, candidates[-1]))


def find_stage3_artifact(
    dirs: Iterable,
    stem: str,
    split_fingerprint: str,
    block_fingerprint: Optional[str] = None,
    features: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[Stage3Artifact, str]:
    """
    The newest ``<stem>_*.joblib`` whose sidecar matches the current split
    (and, if given, block fingerprint and feature list). Raises when nothing
    matches — never falls back to modification time or row counts
    (audit item 11).
    """
    seen: list[str] = []
    matches: list[tuple[str, str]] = []
    for d in dirs:
        for side in glob.glob(os.path.join(str(d), f"{stem}_*.json")):
            try:
                with open(side) as fh:
                    s = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            why = []
            if s.get("split_fingerprint") != split_fingerprint:
                why.append("split")
            if block_fingerprint is not None and s.get("block_fingerprint") != block_fingerprint:
                why.append("block")
            if features is not None and list(s.get("features", [])) != list(features):
                why.append("features")
            seen.append(f"{os.path.basename(side)}: "
                        f"{'match' if not why else 'mismatch ' + '/'.join(why)}")
            jl = side[:-len(".json")] + ".joblib"
            if not why and os.path.exists(jl):
                matches.append((s.get("created_at", ""), jl))
    if not matches:
        raise FileNotFoundError(
            f"No {stem} artifact matches split {split_fingerprint!r}"
            + (f" / block {block_fingerprint[:16]!r}…" if block_fingerprint else "")
            + ". Re-run that arm's Stage 3 on this split.\n  candidates:\n    "
            + ("\n    ".join(seen) if seen else "(none with a JSON sidecar — "
               "pre-PR-33 artifacts are not loadable here)")
        )
    matches.sort()
    path = matches[-1][1]
    art = joblib.load(path)
    if art.split_fingerprint != split_fingerprint:
        raise ValueError(f"{path}: payload fingerprint differs from its sidecar")
    if verbose:
        print(f"  {stem} ← {path}  (matched by split"
              + (" + block" if block_fingerprint else "") + " fingerprint)")
    return art, path
