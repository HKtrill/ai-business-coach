"""
Synthetic Stage 1–3 artifacts in the cascade's REAL formats, for Stage 4 tests.

* Stage 1: ``StageOutput.to_dict()`` (as ``StageOutput.save`` writes it) or a
  dict holding it under ``"stage_output"`` (as the LR / MLP savers do); 10 folds.
* Stage 2: GLASS Router bundle (top-level fields) or RF Router artifact (fields
  under ``"extra"``), with ``train_outputs_oof=True``, "pass1"/"pass2"/"uncertain"
  decisions, ``train_pred``, ``oof_protocol`` and 5 shared folds.
* Stage 3: ``Stage3Artifact`` with the real ``Stage3Calibrator``; 5 shared folds.

Stage 1 carries the ``stage_io`` fingerprint; Stages 2–3 carry the 16-hex
``router`` fingerprint, exactly as in the notebooks. The router scheme is
supplied to the loaders through ``extra_fingerprints=ROUTER_FP``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from shared.stage_io import StageOutput, fold_assignment, split_fingerprint
from shared.stage3.artifact import Stage3Artifact
from shared.stage3.calibration import Stage3Calibrator
from shared.stage3.data import FoldPlan

SEED = 42


def router_split_fp(GS) -> str:
    """Stand-in for the router packages' 16-hex split_fingerprint(X_train, X_test)."""
    h = hashlib.sha256()
    for part in (GS["X_train"].index, GS["X_test"].index):
        h.update(",".join(map(str, part)).encode())
    return h.hexdigest()[:16]


ROUTER_FP = {"router": router_split_fp}


def fold_fp(fold_id) -> str:
    """The repo's glass_router fold_fingerprint when importable, else a stand-in."""
    try:
        from glass_pipeline.glass_router.pipeline import fold_fingerprint
        return fold_fingerprint(np.asarray(fold_id))
    except ImportError:
        return "fixture"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_world(n=3000, n_test=700, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < _sig(2.2 * z - 2.6)).astype(int)
    labels = rng.permutation(np.arange(10_000, 10_000 + n))
    X = pd.DataFrame({"f1": z + rng.normal(0, .5, n), "f2": rng.normal(size=n)}, index=labels)
    tr, te = np.arange(n - n_test), np.arange(n - n_test, n)
    GS = {"X_train": X.iloc[tr], "X_test": X.iloc[te],
          "y_train": pd.Series(y[tr], index=X.index[tr]),
          "y_test": pd.Series(y[te], index=X.index[te])}
    return {"GS": GS, "z_tr": z[tr], "z_te": z[te], "rng": rng,
            "fp_io": split_fingerprint(GS["X_train"].index, GS["X_test"].index, y[tr], y[te]),
            "fp_router": router_split_fp(GS)}


def make_stage1(w, arm="glass", model="calibrated_lr", noise=0.6, wrap=True):
    GS, rng, z_tr, z_te = w["GS"], w["rng"], w["z_tr"], w["z_te"]
    ytr, itr, ite = GS["y_train"], GS["X_train"].index, GS["X_test"].index
    so = StageOutput(
        stage="stage1", arm=arm, model=model, feature_names=["f1", "f2"],
        train_proba_oof=pd.Series(_sig(2 * z_tr - 2.4 + rng.normal(0, noise, len(z_tr))), index=itr),
        test_proba=pd.Series(_sig(2 * z_te - 2.4 + rng.normal(0, noise * .8, len(z_te))), index=ite),
        y_train=ytr.astype(int), y_test=GS["y_test"].astype(int),
        train_fold_id=fold_assignment(ytr, cv_folds=10, random_state=SEED),
        threshold=0.21, threshold_source="in_stage_f2_cv",
        calibration_method="isotonic", calibration_requested="auto",
        oof_provenance="refittable (test fixture)", config={"random_state": SEED},
    )
    d = so.to_dict()
    if not wrap:
        return d                                     # StageOutput.save() file
    return {"stage_output": d, "train_predictions": _sig(2.5 * z_tr - 2.4)}


def _routes(z, rng, abstain_band=True):
    zz = z + rng.normal(0, .2, len(z))
    hi = 1.3 if abstain_band else -0.3
    return np.where(zz < -0.3, "pass1", np.where(zz > hi, "pass2", "uncertain")).astype(object)


def _route_proba(dec, rng, noise, nan_on_abstain):
    p = np.where(dec == "pass1", 0.05, np.where(dec == "pass2", 0.6, 0.12))
    p = np.clip(p + rng.normal(0, noise, len(dec)), 0, 1)
    if nan_on_abstain:
        p = np.where(dec == "uncertain", np.nan, p)
    return p


def make_router(w, layout="top-level", abstain_band=True, two_col=False):
    """GLASS bundle (top-level) or RF Router artifact (layout='extra')."""
    GS, rng = w["GS"], w["rng"]
    ytr = GS["y_train"]
    dtr, dte = _routes(w["z_tr"], rng, abstain_band), _routes(w["z_te"], rng, abstain_band)
    ptr = _route_proba(dtr, rng, .05, nan_on_abstain=(layout == "top-level"))
    pte = _route_proba(dte, rng, .02, nan_on_abstain=(layout == "top-level"))
    if two_col:
        ptr, pte = np.column_stack([1 - ptr, ptr]), np.column_stack([1 - pte, pte])
    pred = lambda d: np.where(d == "pass1", 0, np.where(d == "pass2", 1, -1))
    fold = fold_assignment(ytr, cv_folds=5, random_state=SEED).to_numpy()
    f = {
        "train_outputs_oof": True,
        "train_proba": ptr, "train_decisions": dtr, "train_pred": pred(dtr),
        "train_confidence": np.abs(np.nan_to_num(np.asarray(ptr).reshape(len(dtr), -1)[:, -1], nan=.5) - .5) * 2,
        "train_fold_id": fold,
        "oof_protocol": {"train_outputs_oof": True, "n_folds": 5, "random_state": SEED,
                         "fold_fingerprint": fold_fp(fold)},
        "index_train": list(GS["X_train"].index), "index_test": list(GS["X_test"].index),
        "test_proba": pte, "test_decisions": dte, "test_pred": pred(dte),
        "test_confidence": np.zeros(len(dte)),
    }
    if layout == "top-level":                         # GLASS Router bundle
        f["train_idx"] = f.pop("index_train")
        f["split_fingerprint"] = w["fp_router"]
        f["train_insample_proba_diagnostic"] = _route_proba(dtr, rng, .01, True)
        f["training_base_rate"] = float(ytr.mean())
        return f
    return {"split_fingerprint": w["fp_router"],       # RF Router artifact
            "thresholds": {"t1": 0.79, "t2": 0.27}, "extra": f}


def make_stage3(w, family="ebm", applied=True, shift=0.0):
    GS, rng, z_tr, z_te = w["GS"], w["rng"], w["z_tr"], w["z_te"]
    ytr, yte = GS["y_train"].to_numpy(), GS["y_test"].to_numpy()
    plan = FoldPlan.build(ytr, 5, SEED, True)
    raw_oof = _sig(2.1 * z_tr - 1.0 + shift + rng.normal(0, .5, len(z_tr)))
    raw_te = _sig(2.1 * z_te - 1.0 + shift + rng.normal(0, .4, len(z_te)))
    cal = Stage3Calibrator("isotonic", ece_threshold=(0.0 if applied else 1.0)).fit(raw_oof, ytr, plan)
    ins = _sig(2.4 * z_tr - 1.0)
    return Stage3Artifact(
        stage="stage3", model_family=family, created_at="2026-09-23T00:00:00",
        config={"arm": "glass" if family == "ebm" else "blackbox",
                "calibration_method": "isotonic"}, mirror_report={},
        features=["f1", "f2"],
        index_train=list(GS["X_train"].index), index_test=list(GS["X_test"].index),
        y_train=[int(v) for v in ytr], y_test=[int(v) for v in yte],
        split_fingerprint=w["fp_router"], block_fingerprint="b" * 64,
        feature_fit_scope="per_fold", fold_plan=plan.to_dict(),
        oof={"proba": raw_oof, "proba_calibrated": cal.oof_calibrated_,
             "fold_id": plan.fold_id.copy(), "threshold": 0.55,
             "threshold_per_row": np.full(len(ytr), 0.55)},
        refit={"train_proba": ins, "train_proba_calibrated": cal.transform(ins),
               "test_proba": raw_te, "test_proba_calibrated": cal.transform(raw_te)},
        threshold={"oof": {"threshold": 0.55, "selected_on": "out-of-fold training predictions"},
                   "oof_calibrated_space": {"threshold": 0.19, "selected_on": "OOF (calibrated space)"},
                   "test_oracle": {"threshold": 0.777, "warning": "reference only"}},
        calibration={**cal.report_.to_dict()},
    )
