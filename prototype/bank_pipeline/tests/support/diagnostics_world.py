"""
Synthetic, mutually consistent Stage 1-4 artifacts in the cascade's REAL
formats, for the PR 37 Stage 2 diagnostics tests.

Built on ``tests.support.stage4_world`` (the split, the Stage 1 StageOutput and
the Stage 3 Stage3Artifact) and extended where the diagnostics are stricter
than Stage 4:

* the 16-hex router split fingerprint is the repo's real
  ``rf_router.split_fingerprint`` (stage4_world uses a stand-in);
* routers carry the route transform with FOLD base rates on every row,
  including abstained rows (stage4_world's GLASS blend leaves those NaN);
* LR / EBM carry the repo's LR_FEATURES / EBM_FEATURES; the EBM stores
  fold-nested raw-space decisions and a stage_io split fingerprint, as the
  GLASS notebook writes it;
* the RF Router's outer Pass 1 routes are cut from a known score per fold
  (``p1 < t1[fold]``), which the GLASS-arm RF result stores as ``oof_proba``;
* every artifact also carries test-side fields, so ``trapped`` can prove they
  are never read.
"""

from __future__ import annotations

import copy
import functools

import numpy as np

from shared.stage2.binning import RF_FEATURES_BINARY
from shared.stage2.diagnostics import boundary
from shared.stage1.feature_engineering import LR_FEATURES
from shared.stage3.feature_engineering import EBM_FEATURES
from shared.stage_io import fold_assignment, label_fingerprint
from tests.support.stage4_world import (
    PASS1_PRECISIONS,
    PASS2_PRECISIONS,
    RF_REMAINDER,
    SEED,
    build_world,
    make_stage1,
    make_stage3,
)

N_FOLDS = 5
RF_PARAMS = {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20,
             "max_features": "sqrt", "minority_weight": 1.0, "class_weight": "balanced"}
ATTESTATION = "synthetic PROVENANCE.json (test fixture)"
EBM_NESTED = (0.50, 0.52, 0.54, 0.56, 0.58)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fold_fp(fold_id) -> str:
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    return fold_fingerprint(np.asarray(fold_id))


def fold_base_rates(y, fold) -> np.ndarray:
    y, fold = np.asarray(y, dtype=float), np.asarray(fold)
    b = np.empty(len(y))
    for k in np.unique(fold):
        b[fold == k] = y[fold != k].mean()
    return b


def transform(dec, conf, b) -> np.ndarray:
    return np.where(dec == "pass1", b * (1.0 - conf),
                    np.where(dec == "pass2", b + (1.0 - b) * conf, b))


def preds(dec) -> np.ndarray:
    return np.where(dec == "pass1", 0, np.where(dec == "pass2", 1, -1))


# ======================================================================
# trapping: prove test-side fields are never read
# ======================================================================
TRAP_TOKENS = ("test", "oracle", "holdout", "refit", "fold_mean", "metrics", "glass_router_data",
               "no_abstain", "with_abstain")


class ForbiddenRead(AssertionError):
    pass


def _trapped_key(key) -> bool:
    return any(t in str(key).lower() for t in TRAP_TOKENS)


class TrapMapping(dict):
    """A dict whose test-side keys raise when read; bulk reads raise too."""

    def __getitem__(self, key):
        if _trapped_key(key):
            raise ForbiddenRead(f"read test-side key {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if _trapped_key(key):
            raise ForbiddenRead(f"read test-side key {key!r}")
        return super().get(key, default)

    def items(self):
        raise ForbiddenRead("bulk read of an artifact mapping (items)")

    def values(self):
        raise ForbiddenRead("bulk read of an artifact mapping (values)")


def trapped(obj):
    """Recursively wrap dicts that hold test-side keys (call after any mutation)."""
    if isinstance(obj, dict) and not isinstance(obj, TrapMapping):
        inner = {k: trapped(v) for k, v in dict.items(obj)}
        return TrapMapping(inner) if any(_trapped_key(k) for k in inner) else inner
    return obj


def trapped_stage3(art):
    """A Stage3Artifact whose test-side attributes and keys raise when read."""
    from shared.stage3.artifact import Stage3Artifact

    class TrappedStage3Artifact(Stage3Artifact):
        def __getattribute__(self, name):
            if name in ("refit", "fold_mean", "metrics", "y_test", "index_test"):
                raise ForbiddenRead(f"read Stage3Artifact.{name}")
            return super().__getattribute__(name)

    kw = {f: getattr(art, f) for f in art.__dataclass_fields__}
    kw["threshold"] = trapped(kw["threshold"])
    kw["calibration"] = trapped(kw["calibration"])
    return TrappedStage3Artifact(**kw)


# ======================================================================
# the world
# ======================================================================
@functools.lru_cache(maxsize=None)
def _base(seed: int = 0) -> dict:
    w = build_world(seed=seed)
    GS = w["GS"]
    idx_tr, idx_te = GS["X_train"].index, GS["X_test"].index
    y = GS["y_train"].to_numpy().astype(int)
    fold = fold_assignment(y, cv_folds=N_FOLDS, random_state=SEED).to_numpy()
    w.update(idx_tr=idx_tr, idx_te=idx_te, y=y, fold=fold, fold_fp=fold_fp(fold),
             router_fp=boundary.router_split_fingerprint(idx_tr, idx_te),
             label_fp=label_fingerprint(idx_tr, y), b=fold_base_rates(y, fold))
    rng = np.random.default_rng(seed + 11)
    w["glass"] = _glass_bundle(w, rng)
    w["rf"], w["rf_result"] = _rf_pair(w, rng)
    w["lr"] = _lr(w)
    w["ebm"] = _ebm(w)
    w["meta"] = _meta(w)
    return w


def world(seed: int = 0) -> dict:
    """A fresh deep copy: tests may mutate any artifact."""
    base = _base(seed)
    return {k: (copy.deepcopy(v) if k in ("glass", "rf", "rf_result", "lr", "ebm", "meta") else v)
            for k, v in base.items()}


def reference(w, *, attested: bool = True, **kw):
    return boundary.build_train_reference(
        index_train=w["idx_tr"], y_train=w["GS"]["y_train"], index_test=w["idx_te"],
        attested_stage_io_split_fingerprint=w["fp_io"] if attested else None,
        attestation_source=ATTESTATION if attested else None, **kw)


def _glass_bundle(w, rng) -> dict:
    z, y = w["z_tr"], w["y"]
    zz = z + rng.normal(0, .2, len(z))
    dec = np.where(zz < -0.3, "pass1", np.where(zz > 1.3, "pass2", "uncertain")).astype(object)
    conf = np.where(dec == "pass1", rng.choice(PASS1_PRECISIONS, len(dec)),
                    np.where(dec == "pass2", rng.choice(PASS2_PRECISIONS, len(dec)), 0.0))
    p = transform(dec, conf, w["b"])
    zt = w["z_te"]
    dte = np.where(zt < -0.3, "pass1", np.where(zt > 1.3, "pass2", "uncertain")).astype(object)
    full_b = float(y.mean())
    return {
        "features": list(RF_FEATURES_BINARY),
        "train_idx": w["idx_tr"], "test_idx": w["idx_te"], "index_test": w["idx_te"],
        "split_fingerprint": w["router_fp"],
        "train_pred": preds(dec), "train_confidence": conf,
        "train_covered": dec != "uncertain", "train_abstained": dec == "uncertain",
        "train_decisions": dec, "train_proba": np.column_stack([1 - p, p]),
        "train_outputs_oof": True, "train_fold_id": w["fold"].copy(),
        "oof_protocol": {"train_outputs_oof": True, "n_folds": N_FOLDS,
                         "fold_fingerprint": w["fold_fp"],
                         "train_label_fingerprint": w["label_fp"], "fold_source": "provided",
                         "random_state": SEED, "glass_rf_model": "fold_rf",
                         "glass_pass2_population": "full", "glass_abstain_label": "uncertain",
                         "test_outputs": "from full-train refit models (not produced here)"},
        "train_insample_proba_diagnostic": np.column_stack([1 - p, p])[::-1].copy(),
        "test_pred": preds(dte), "test_confidence": np.zeros(len(dte)), "test_decisions": dte,
        "test_proba": np.full((len(dte), 2), [1 - full_b, full_b]),
        "rules": [{"pass": "Pass 1", "precision": 0.95}, {"pass": "Pass 1", "precision": 0.93},
                  {"pass": "Pass 2", "precision": 0.45}],
        "config": {"mode": "two_pass"},
        "config_full": {"mode": "two_pass", "pass2_population": "full", "min_support_pass1": 50,
                        "min_support_pass2": 30, "min_precision_not_subscribe": 0.93},
        "training_base_rate": full_b, "train_label_fingerprint": w["label_fp"],
        "fit_report": {"pass2_population": "full"},
        "timestamp": "20261023_025611",
    }


def _cut_between(values, q) -> float:
    """A cut strictly between two adjacent distinct values near quantile ``q``."""
    v = np.unique(values)
    j = int(np.clip(round(q * (len(v) - 1)), 0, len(v) - 2))
    return float((v[j] + v[j + 1]) / 2.0)


def _rf_pair(w, rng):
    z, y, fold, b = w["z_tr"], w["y"], w["fold"], w["b"]
    n = len(y)
    p1 = _sig(1.6 * z - 1.0 + rng.normal(0, .4, n))
    p2 = _sig(1.8 * z - 1.5 + rng.normal(0, .4, n))
    t1 = {k: _cut_between(p1[fold == k], 0.45) for k in range(N_FOLDS)}
    pass1 = p1 < np.array([t1[k] for k in fold])
    t2 = {k: _cut_between(p2[(fold == k) & ~pass1], 0.6) for k in range(N_FOLDS)}
    pass2 = ~pass1 & (p2 > np.array([t2[k] for k in fold]))
    dec = np.where(pass1, "pass1", np.where(pass2, "pass2", "uncertain")).astype(object)
    conf = np.zeros(n)
    for k in range(N_FOLDS):
        fit, rows = fold != k, fold == k
        npv = float((y[fit & pass1] == 0).mean())
        prec = float((y[fit & pass2] == 1).mean())
        conf[rows] = np.where(dec[rows] == "pass1", npv, np.where(dec[rows] == "pass2", prec, 0.0))
    p = transform(dec, conf, b)
    n_te = len(w["idx_te"])
    rf_protocol = {"train_outputs_oof": True, "n_folds": N_FOLDS, "fold_fingerprint": w["fold_fp"],
                   "train_label_fingerprint": w["label_fp"], "random_state": SEED,
                   "feature_engineering": "engineer_features refit per fold; locked bins",
                   "router": "full RFRouter refit per fold (inner OOF + per-fold t1/t2)",
                   "pass2_population": RF_REMAINDER,
                   "hyperparameters": "resolved once on full X_train (not refitted per fold)",
                   "test_outputs": "from the full-train router",
                   "split_fingerprint": w["router_fp"]}
    extra = {
        "index_train": w["idx_tr"], "index_test": w["idx_te"],
        "constraint_source": "match_glass_oof",
        "rf_params_origin": "models/rf/rf_result_20261023_025611.joblib",
        "glass_artifact": "models/glass_router/glass_router_20261023_025611.joblib",
        "train_outputs_oof": True, "train_pred": preds(dec), "train_confidence": conf,
        "train_decisions": dec, "train_proba": np.column_stack([1 - p, p]),
        "train_fold_id": fold.copy(), "oof_protocol": rf_protocol,
        "oof_fold_summary": [{"fold": k, "t1": t1[k], "t2": t2[k], "remainder_frac": 0.55,
                              "pass1": float((pass1 & (fold == k)).sum() / (fold == k).sum()),
                              "pass2": 0.1, "abstain": 0.35, "leak": 0.2, "seconds": 1.0}
                             for k in range(N_FOLDS)],
        "test_pred": np.full(n_te, -1), "test_confidence": np.zeros(n_te),
        "test_decisions": np.array(["uncertain"] * n_te, dtype=object),
        "test_proba": np.full((n_te, 2), [1 - y.mean(), y.mean()]),
    }
    rf = {
        "artifact_version": "1.0", "created_utc": "2026-10-23T09:00:00+00:00", "mode": "two_pass",
        "config": {"mode": "two_pass", "rf_params": dict(RF_PARAMS), "pass2_rf_params": None,
                   "n_oof_folds": N_FOLDS, "random_state": SEED},
        "constraints": {"min_precision_not_subscribe": 0.95, "max_subscriber_leakage_rate": 0.25},
        "feature_names": list(RF_FEATURES_BINARY), "training_base_rate": float(y.mean()),
        "thresholds": {"t1": float(np.median(list(t1.values()))),
                       "t2": float(np.median(list(t2.values())))},
        "bands": {"kind": "flat", "pass1_confidence": 0.95, "pass2_confidence": 0.45},
        "fit_report": {"mode": "two_pass (GLASS counterpart)"},
        "split_fingerprint": w["router_fp"], "train_label_fingerprint": w["label_fp"],
        "test_metrics": {"pass2_precision": 0.5}, "environment": {"python": "3.14.3"},
        "extra": extra,
    }
    rf_result = {
        "pipe": None, "model": None, "params": dict(RF_PARAMS),
        "metrics_cv": {"auc_mean": 0.79}, "metrics_test": {"auc": 0.8},
        "feature_importance": None,
        "glass_router_data": {"y_eng_test": w["GS"]["y_test"].copy()},
        "training_date": "20261023_025611", "split_fingerprint": w["router_fp"],
        "index_test": w["idx_te"], "train_outputs_oof": True,
        "oof_proba": p1, "oof_index": w["idx_tr"], "oof_fold_id": fold.copy(),
        "oof_protocol": {"train_outputs_oof": True, "n_folds": N_FOLDS,
                         "fold_fingerprint": w["fold_fp"], "train_label_fingerprint": w["label_fp"],
                         "fold_source": "provided", "random_state": SEED,
                         "feature_engineering": "RFFeatureEngineer refit per fold; locked bins",
                         "rf": "tuned params reused, refit per fold with train_rf_stage sample "
                               "weights (no re-tuning)",
                         "glass_rf_model": "fold_rf", "glass_pass2_population": "full",
                         "test_outputs": "from full-train refit models (not produced here)"},
    }
    return rf, rf_result


def _lr(w) -> dict:
    art = make_stage1(w)
    so = art["stage_output"]
    so["feature_names"] = list(LR_FEATURES)
    so["threshold"] = 0.10
    return art


def _ebm(w):
    art = make_stage3(w)
    art.features = list(EBM_FEATURES)
    art.split_fingerprint = w["fp_io"]               # GLASS notebook Cell 17: stage_io scheme
    fold = np.asarray(art.oof["fold_id"])
    per_row = np.array([EBM_NESTED[k] for k in fold], dtype=float)
    art.oof["threshold_per_row"] = per_row
    art.oof["decision"] = (np.asarray(art.oof["proba"]) >= per_row).astype(int)
    art.threshold["oof"]["threshold"] = 0.53
    art.threshold["oof_calibrated_space"]["threshold"] = 0.11
    art.threshold["oof_nested"] = {"by_fold": list(EBM_NESTED), "mean": float(np.mean(EBM_NESTED)),
                                   "drives_train_decisions": True,
                                   "selected_on": "inner out-of-fold scores of each fold's "
                                                  "training rows"}
    art.metrics = {"test": {"auc": 0.8}}
    return art


def _meta(w) -> dict:
    from glass_pipeline.meta_ebm.meta_stage import find_recall_threshold
    y = w["y"]
    lr_p = np.asarray(w["lr"]["stage_output"]["train_proba_oof"], dtype=float)
    ebm_p = np.asarray(w["ebm"].oof["proba_calibrated"], dtype=float)
    return {
        "schema": "meta_ebm/test-fixture", "split_fingerprint": w["fp_io"],
        "thresholds": {"lr": find_recall_threshold(y, lr_p, 0.70),
                       "ebm": find_recall_threshold(y, ebm_p, 0.70),
                       "glass": float(y.mean()), "target_recall": 0.70,
                       "sources": {"lr": "find_recall_threshold(y_train, LR OOF, 0.7)",
                                   "ebm": "find_recall_threshold(y_train, EBM OOF, 0.7)"}},
        "no_abstain": {"test_auc": 0.8}, "with_abstain": {"test_f2": 0.5},
        "artifact_thresholds": {"used_by_stage4": False},
    }


def views(w, ref=None, *, trap: bool = True):
    """(ref, glass, rf, lr, ebm) views of world ``w``."""
    from shared.stage2.diagnostics import extract_ebm_view, extract_lr_view, extract_router_view
    ref = ref or reference(w)
    t = trapped if trap else (lambda x: x)
    glass = extract_router_view(t(w["glass"]), ref, expected_arm="glass")
    rf = extract_router_view(t(w["rf"]), ref, expected_arm="blackbox")
    lr = extract_lr_view(t(w["lr"]), ref)
    ebm = extract_ebm_view(trapped_stage3(w["ebm"]) if trap else w["ebm"], ref)
    return ref, glass, rf, lr, ebm

