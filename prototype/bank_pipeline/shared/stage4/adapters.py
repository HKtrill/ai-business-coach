"""
shared.stage4.adapters
======================
Artifact-format adapters → ``StageStream``. Each format is read by ONE adapter
used by both arms, because both arms write the same formats:

``stage_output_stream``   Stage 1 — a ``shared.stage_io.StageOutput`` (GLASS LR
                          and MLP), as a ``to_dict()`` file, a dict holding it
                          under ``"stage_output"``, or an instance.
``router_stream``         Stage 2 — GLASS Router bundle (top-level fields) or RF
                          Router artifact (fields under ``"extra"``). Same
                          vocabulary: ``train_outputs_oof`` flag, ``train_proba``,
                          ``train_decisions``, ``train_pred``, ``train_fold_id``,
                          ``oof_protocol``, ``test_*``, ``index_test``.
``stage3_stream``         Stage 3 — ``Stage3Artifact`` (EBM and XGBoost).

Every adapter returns ``(stream, checks)``; ``checks`` feed the shared
``ValidationReport`` in ``build_stage4_inputs``.

Canonical fields (nothing else is consumed for training)
--------------------------------------------------------
Stage 1  train ``train_proba_oof`` · test ``test_proba`` (the contract defines
         both as calibrated) · folds ``train_fold_id``.
Stage 2  ``train_outputs_oof`` must be ``True``: it certifies that the top-level
         ``train_*`` arrays are the outer cross-fit (OOF) outputs. train
         ``train_proba`` (positive column) + ``train_decisions`` · test
         ``test_proba`` + ``test_decisions`` · folds ``train_fold_id``.
         ``train_insample_proba_diagnostic`` (GLASS only) is read solely to prove
         it is not what is consumed.
Stage 3  ``calibration["applied"]`` picks ONE matched pair:
         applied → ``oof["proba_calibrated"]`` + ``refit["test_proba_calibrated"]``
         skipped → ``oof["proba"]`` + ``refit["test_proba"]``.
         ``refit["train_proba*"]`` (in-sample) is read only for the aliasing
         guard; ``threshold["test_oracle"]`` is never read.
"""

from __future__ import annotations

import importlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .contract import SEC_OOF, SEC_SEM, Stage4InputError, StageStream, not_identical

PASS1, PASS2, ABSTAIN = "pass1", "pass2", "abstain"
_ABSTAIN_TOKENS = {"abstain", "uncertain", "defer", "deferred", "uncovered",
                   "none", "nan", "", "pass0", "remainder", "residual", "stage3"}


# ======================================================================
# helpers
# ======================================================================
def load(obj_or_path):
    """Path → joblib.load; an already-loaded object passes through."""
    if isinstance(obj_or_path, (str, bytes)) or hasattr(obj_or_path, "__fspath__"):
        return joblib.load(obj_or_path), str(obj_or_path)
    return obj_or_path, None


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _require(d, keys, where: str) -> None:
    missing = [k for k in keys if _get(d, k) is None]
    if missing:
        avail = sorted(map(str, d.keys())) if isinstance(d, dict) else type(d).__name__
        raise Stage4InputError(f"{where}: missing required field(s) {missing}. "
                               f"Available: {avail}")


def vec(x, where: str) -> np.ndarray:
    """1-D float vector; a 2-column predict_proba array → positive column."""
    a = np.asarray(x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else x,
                   dtype=float)
    if a.ndim == 2 and a.shape[1] == 2:
        a = a[:, 1]
    if a.ndim != 1:
        raise Stage4InputError(f"{where}: expected a 1-D probability vector, got {a.shape}")
    return a


def _index_of(x):
    return x.index if isinstance(x, (pd.Series, pd.DataFrame)) else None


def normalize_router_decisions(values, where: str) -> np.ndarray:
    """Routes → "pass1" / "pass2" / "abstain"; anything else is refused."""
    raw = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
    if raw.dtype.kind in "biuf":
        raise Stage4InputError(
            f"{where}: router decisions are numeric ({raw.dtype}). The route "
            "encoding must be explicit ('pass1' / 'pass2' / abstain).")
    out = np.empty(len(raw), dtype=object)
    unknown = set()
    for i, v in enumerate(raw):
        tok = "none" if v is None else str(v).strip().lower()
        tok = tok.replace("_", "").replace("-", "").replace(" ", "")
        if tok == PASS1:
            out[i] = PASS1
        elif tok == PASS2:
            out[i] = PASS2
        elif tok in _ABSTAIN_TOKENS:
            out[i] = ABSTAIN
        else:
            unknown.add(str(v))
    if unknown:
        raise Stage4InputError(
            f"{where}: unrecognised router decision values {sorted(unknown)[:10]}")
    return out


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


# ======================================================================
# Stage 1 — StageOutput
# ======================================================================
_S1_FIELDS = ("train_proba_oof", "test_proba", "train_fold_id", "split_fingerprint",
              "threshold", "threshold_source", "y_train", "y_test")
_S1_INSAMPLE_KEYS = ("train_predictions", "train_predictions_calibrated",
                     "train_proba", "train_proba_calibrated", "proba_train",
                     "proba_train_insample", "train_proba_insample")


def _as_stage_output(so, where: str):
    if isinstance(so, dict) and "__stage_output__" in so:
        from shared.stage_io import StageOutput
        return StageOutput.from_dict(so)          # re-runs every contract check
    if type(so).__name__ == "StageOutput":
        return so
    if isinstance(so, dict):
        _require(so, _S1_FIELDS, where)
        return so
    raise Stage4InputError(f"{where}: unsupported stage_output type {type(so).__name__}")


def stage_output_stream(obj_or_path, name: str, model: str, n_train: int):
    """Stage 1 (GLASS LR / MLP)."""
    art, path = load(obj_or_path)
    where = path or name
    if isinstance(art, dict) and "__stage_output__" in art:
        container, so_raw, src = {}, art, ""                 # StageOutput.save() file
    elif isinstance(art, dict) and "stage_output" in art:
        container, so_raw, src = art, art["stage_output"], "stage_output."
    elif type(art).__name__ == "StageOutput":
        container, so_raw, src = {}, art, ""
    else:
        keys = sorted(map(str, art.keys())) if isinstance(art, dict) else type(art).__name__
        raise Stage4InputError(
            f"Stage 1 artifact {where} holds no StageOutput contract (found {keys}). "
            "Pre-audit artifacts are refused: their train-side arrays were not "
            "guaranteed out-of-fold. Re-run Stage 1.")
    so = _as_stage_output(so_raw, f"{where}::stage_output")

    tr, te = _get(so, "train_proba_oof"), _get(so, "test_proba")
    cal = _get(so, "calibration_method")
    stream = StageStream(
        name=name, stage="stage1", model=str(_get(so, "model") or model),
        train_proba=vec(tr, f"{name} train_proba_oof"),
        test_proba=vec(te, f"{name} test_proba"),
        probability_space=(f"calibrated ({cal})" if cal and cal != "none"
                           else "raw" if cal == "none" else "as stored in StageOutput"),
        train_source=f"{src}train_proba_oof", test_source=f"{src}test_proba",
        split_fingerprint=_get(so, "split_fingerprint"),
        artifact_path=path,
        train_fold_id=np.asarray(_get(so, "train_fold_id")),
        fold_source=f"{src}train_fold_id",
        fold_seed=(_get(so, "config") or {}).get("random_state"),
        standalone_threshold=float(_get(so, "threshold")),
        standalone_threshold_source=f"{src}threshold ({_get(so, 'threshold_source')})",
        train_index=_index_of(tr), test_index=_index_of(te),
        index_source="StageOutput Series index" if _index_of(tr) is not None else None,
        y_train=np.asarray(_get(so, "y_train")), y_test=np.asarray(_get(so, "y_test")),
        notes={"oof_provenance": _get(so, "oof_provenance"),
               "calibration_method": cal,
               "calibration_requested": _get(so, "calibration_requested"),
               "stage_output_validated": type(so).__name__ == "StageOutput"},
    )

    checks = [(SEC_OOF, stream.stage, f"{name}: train stream is StageOutput.train_proba_oof",
               lambda: stream.train_source.endswith("train_proba_oof"), "")]
    for k in _S1_INSAMPLE_KEYS:
        v = container.get(k) if isinstance(container, dict) else None
        if v is not None and len(np.ravel(v)) in (n_train, 2 * n_train):
            checks.append((SEC_OOF, stream.stage, f"{name}: OOF ≠ in-sample key {k!r}",
                           lambda v=v, k=k: not_identical(stream.train_proba, vec(v, k)),
                           "present in artifact; never consumed"))
    prov = str(_get(so, "oof_provenance") or "")
    checks.append((SEC_OOF, stream.stage, f"{name}: OOF provenance is refittable",
                   lambda: not prov.upper().startswith("PREFIT"), prov[:80]))
    return stream, checks


# ======================================================================
# Stage 2 — routers (GLASS Router, RF Router)
# ======================================================================
def _router_fields(art, where: str):
    """(fields, split_fingerprint, layout) for either router artifact layout."""
    if not isinstance(art, dict):
        raise Stage4InputError(f"{where}: router artifact is {type(art).__name__}, "
                               "expected a dict")
    if isinstance(art.get("extra"), dict) and "train_outputs_oof" in art["extra"]:
        f = art["extra"]
        return f, art.get("split_fingerprint", f.get("split_fingerprint")), "extra"
    if "train_outputs_oof" in art:
        return art, art.get("split_fingerprint"), "top-level"
    raise Stage4InputError(
        f"{where}: no 'train_outputs_oof' in the artifact (top level or 'extra'). "
        f"Keys: {sorted(map(str, art.keys()))}")


def _fold_fp_fn():
    for mod in ("glass_pipeline.glass_router.pipeline",):
        try:
            return getattr(importlib.import_module(mod), "fold_fingerprint")
        except (ImportError, AttributeError):
            continue
    return None


def router_stream(obj_or_path, name: str, model: str, y_train: np.ndarray,
                  fold_fingerprint_fn=None):
    """Stage 2 (GLASS Router / RF Router)."""
    art, path = load(obj_or_path)
    where = path or name
    f, fp, layout = _router_fields(art, where)
    src = "" if layout == "top-level" else "extra."

    flag = f["train_outputs_oof"]
    if flag is not True:
        raise Stage4InputError(
            f"{where}: train_outputs_oof = {flag!r}. Stage 4 needs the router's "
            "outer cross-fit train outputs, certified by train_outputs_oof=True.")
    _require(f, ("train_proba", "train_decisions", "test_proba", "test_decisions"), where)

    train_dec = normalize_router_decisions(f["train_decisions"], f"{where} train_decisions")
    test_dec = normalize_router_decisions(f["test_decisions"], f"{where} test_decisions")
    train_proba = vec(f["train_proba"], f"{where} train_proba")
    test_proba = vec(f["test_proba"], f"{where} test_proba")

    # Probabilities only matter where the router votes; NaN on abstain rows is
    # allowed. Each Stage 4 consumer decides what an abstain row carries.
    cov_tr, cov_te = train_dec != ABSTAIN, test_dec != ABSTAIN
    same_tr, same_te = len(cov_tr) == len(train_proba), len(cov_te) == len(test_proba)

    fold = f.get("train_fold_id")
    stream = StageStream(
        name=name, stage="stage2", model=model,
        train_proba=train_proba, test_proba=test_proba,
        probability_space="router-native P(subscribe) (positive column)",
        train_source=f"{src}train_proba", test_source=f"{src}test_proba",
        split_fingerprint=fp, artifact_path=path,
        train_fold_id=None if fold is None else np.asarray(fold),
        fold_source=None if fold is None else f"{src}train_fold_id",
        fold_seed=(f.get("oof_protocol") or {}).get("random_state"),
        train_decisions=train_dec, test_decisions=test_dec,
        decision_source=f"{src}train_decisions / {src}test_decisions",
        train_defined=cov_tr if same_tr else None,
        test_defined=cov_te if same_te else None,
        train_index=f.get("index_train", f.get("train_idx")),
        test_index=f.get("index_test", f.get("test_idx")),
        index_source=f"{src}index_train|train_idx / {src}index_test|test_idx",
        y_train=None if f.get("y_train") is None else np.asarray(f["y_train"]),
        y_test=None if f.get("y_test") is None else np.asarray(f["y_test"]),
        notes={"artifact_layout": layout,
               "train_outputs_oof_flag": True,
               "oof_protocol": {k: v for k, v in (f.get("oof_protocol") or {}).items()
                                if not isinstance(v, (list, np.ndarray))}},
    )

    checks = []
    # --- OOF honesty ---------------------------------------------------------
    diag = f.get("train_insample_proba_diagnostic")
    checks.append((SEC_OOF, "stage2", f"{name}: consumed train proba ≠ in-sample diagnostic",
                   lambda: not_identical(train_proba, None if diag is None
                                         else vec(diag, "diag")),
                   "diagnostic absent from artifact" if diag is None
                   else "diagnostic present; never consumed"))
    proto = f.get("oof_protocol") or {}
    fn = fold_fingerprint_fn or _fold_fp_fn()
    if proto.get("fold_fingerprint") is not None and fold is not None and fn is not None:
        checks.append((SEC_OOF, "stage2", f"{name}: oof_protocol fold fingerprint = train_fold_id",
                       lambda: fn(np.asarray(fold)) == proto["fold_fingerprint"],
                       str(proto["fold_fingerprint"])))
    if proto.get("n_folds") is not None and fold is not None:
        checks.append((SEC_OOF, "stage2", f"{name}: oof_protocol n_folds = fold ids",
                       lambda: int(proto["n_folds"]) == len(np.unique(fold)), ""))
    if f.get("train_pred") is not None:
        def _pred_consistent(pred=f["train_pred"]):
            p = np.asarray(pred)
            want = np.where(train_dec == PASS1, 0, np.where(train_dec == PASS2, 1, -1))
            return np.array_equal(p.astype(int), want)
        checks.append((SEC_OOF, "stage2", f"{name}: train_pred agrees with train_decisions",
                       _pred_consistent, "pass1↔0, pass2↔1, abstain↔-1"))

    # --- semantics, on TRAIN OOF labels only ---------------------------------
    y = np.asarray(y_train).astype(int)
    base = float(y.mean())

    def _rate(route):
        m = train_dec == route
        if len(m) != len(y) or not m.any():
            raise Stage4InputError(f"no train rows routed to {route}")
        return float(y[m].mean())

    checks.append((SEC_SEM, "stage2", f"{name}: Pass 1 is the NOT_SUBSCRIBE route "
                   "(train pos-rate < base rate)", lambda: _rate(PASS1) < base,
                   f"base rate {base:.4f}"))
    checks.append((SEC_SEM, "stage2", f"{name}: Pass 2 is the SUBSCRIBE route "
                   "(train pos-rate > base rate)", lambda: _rate(PASS2) > base,
                   f"base rate {base:.4f}"))
    checks.append((SEC_SEM, "stage2", f"{name}: proba oriented as P(subscribe) "
                   "(covered-row OOF AUC > 0.5)",
                   lambda: _auc(y[cov_tr], train_proba[cov_tr]) > 0.5, ""))
    try:
        stream.notes.update({
            "train_coverage": float(cov_tr.mean()),
            "test_coverage": float(cov_te.mean()),
            "train_abstain_rows": int((~cov_tr).sum()),
            "test_abstain_rows": int((~cov_te).sum()),
            "train_pos_rate_pass1": _rate(PASS1),
            "train_pos_rate_pass2": _rate(PASS2),
            "train_pass1_with_p_ge_0.5": int(((train_dec == PASS1) & (train_proba >= 0.5)).sum()),
            "train_pass2_with_p_lt_0.5": int(((train_dec == PASS2) & (train_proba < 0.5)).sum()),
        })
    except Exception:  # noqa: BLE001 — the checks above report it
        pass
    return stream, checks


# ======================================================================
# Stage 3 — Stage3Artifact
# ======================================================================
def stage3_stream(obj_or_path, name: str, model: str):
    """Stage 3 (GLASS EBM / XGBoost)."""
    art, path = load(obj_or_path)
    where = path or name
    if isinstance(art, dict):
        raise Stage4InputError(
            f"{where} is a pre-PR-33 Stage 3 dict. Its train_predictions are "
            "in-sample and its optimal_threshold was fitted on y_test; Stage 4 "
            "requires the shared Stage3Artifact. Re-run Stage 3.")
    if type(art).__name__ != "Stage3Artifact":
        raise Stage4InputError(f"{where}: expected Stage3Artifact, got {type(art).__name__}")
    for attr in ("oof", "refit", "threshold", "calibration", "fold_plan",
                 "index_train", "index_test", "split_fingerprint"):
        if getattr(art, attr, None) is None:
            raise Stage4InputError(f"{where}: Stage3Artifact.{attr} missing")
    if art.feature_fit_scope != "per_fold":
        raise Stage4InputError(
            f"{where}: feature_fit_scope={art.feature_fit_scope!r}. Only 'per_fold' "
            "gives feature-level OOF honesty; 'global' is a comparison-only mode.")
    if "applied" not in art.calibration:
        raise Stage4InputError(f"{where}: calibration['applied'] missing — cannot "
                               "choose the matched probability representation")

    applied = bool(art.calibration["applied"])
    if applied:
        tr_key, te_key, thr_key = "proba_calibrated", "test_proba_calibrated", "oof_calibrated_space"
        space = (f"calibrated ({art.calibration.get('method')}; train: fold-nested "
                 "calibrator, test: calibrator fit on all OOF rows)")
    else:
        tr_key, te_key, thr_key = "proba", "test_proba", "oof"
        space = "raw (calibration gate not breached; calibrator is identity)"
    _require(art.oof, (tr_key, "fold_id"), f"{where}::oof")
    _require(art.refit, (te_key,), f"{where}::refit")
    _require(art.threshold, (thr_key,), f"{where}::threshold")

    stream = StageStream(
        name=name, stage="stage3", model=f"{model} ({art.model_family})",
        train_proba=vec(art.oof[tr_key], f"oof[{tr_key}]"),
        test_proba=vec(art.refit[te_key], f"refit[{te_key}]"),
        probability_space=space,
        train_source=f"Stage3Artifact.oof[{tr_key!r}]",
        test_source=f"Stage3Artifact.refit[{te_key!r}]",
        split_fingerprint=art.split_fingerprint, artifact_path=path,
        train_fold_id=np.asarray(art.oof["fold_id"]),
        fold_source="Stage3Artifact.oof['fold_id']",
        fold_seed=art.fold_plan.get("random_state"),
        standalone_threshold=float(art.threshold[thr_key]["threshold"]),
        standalone_threshold_source=f"Stage3Artifact.threshold[{thr_key!r}] "
                                    f"(selected on {art.threshold[thr_key].get('selected_on')})",
        train_index=list(art.index_train), test_index=list(art.index_test),
        index_source="Stage3Artifact.index_train / index_test",
        y_train=np.asarray(art.y_train) if len(art.y_train) else None,
        y_test=np.asarray(art.y_test) if len(art.y_test) else None,
        notes={"calibration_applied": applied,
               "calibration_method": art.calibration.get("method"),
               "calibration_reason": art.calibration.get("reason"),
               "feature_fit_scope": art.feature_fit_scope,
               "block_fingerprint": getattr(art, "block_fingerprint", None),
               "test_oracle_threshold_consumed": False},
    )

    checks = [
        (SEC_OOF, "stage3", f"{name}: oof fold_id = fold_plan fold_id",
         lambda: np.array_equal(np.asarray(art.oof["fold_id"]),
                                np.asarray(art.fold_plan.get("fold_id"))), ""),
        (SEC_OOF, "stage3", f"{name}: train stream ≠ refit in-sample train_proba",
         lambda: not_identical(stream.train_proba, art.refit.get("train_proba")), ""),
        (SEC_OOF, "stage3", f"{name}: train stream ≠ refit in-sample train_proba_calibrated",
         lambda: not_identical(stream.train_proba, art.refit.get("train_proba_calibrated")), ""),
        (SEC_OOF, "stage3", f"{name}: standalone threshold is not the test oracle",
         lambda: thr_key != "test_oracle", thr_key),
    ]
    if not applied:
        checks.append((SEC_OOF, "stage3", f"{name}: calibration skipped ⇒ calibrated slots = raw",
                       lambda: np.array_equal(np.asarray(art.oof["proba_calibrated"], float),
                                              np.asarray(art.oof["proba"], float))
                       and np.array_equal(np.asarray(art.refit["test_proba_calibrated"], float),
                                          np.asarray(art.refit["test_proba"], float)), ""))
    return stream, checks
