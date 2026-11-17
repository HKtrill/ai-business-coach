"""
phase1_reproduction.compare
===========================
The section 7.4 reproduction evaluation (D8): each artifact the Step 6B
notebook executions produced against its accepted counterpart, and the
reproduction class. Pure functions on loaded artifacts. Nothing is fitted,
tuned, rescored or written.

Per pair of artifacts (``stage1``, ``stage3``, ``glass_arbiter``,
``meta_xgb``):

* ``configuration``: the selected configuration (hyperparameters, calibration
  family, feature names, block and split identities, operating rules) equal
  value for value, or the fields that differ;
* ``fold_ids``: the fold assignment, row for row;
* ``vectors``: numerical drift of every OOF and test vector (maximum and
  99.9th-percentile |delta|, Spearman rank correlation);
* ``thresholds``: every operating threshold (Stage 1 F2 thresholds, Stage 3
  OOF thresholds, Stage 4 recall thresholds and abstention cuts): accepted
  and reproduced values, |delta t|, the same value (within ``TOL``), within
  the maximum |delta| of the vector it is chosen on, and the change in the
  number of rows it flags;
* ``decisions``: Stage 1 and 3 decisions at their operating thresholds, each
  arbiter's decisions and abstentions, and the Stage 2 three-state routes:
  the rows that differ, with their distance to the threshold. A decision
  defined by a threshold on a score is within drift when that distance is at
  most the score's maximum |delta| plus the threshold's own movement; a
  route has no score, so any differing route is outside the condition.

The GLASS Arbiter's weighted confidences and decision margins are not
stored. They are recomputed from the artifact's own saved inputs and frozen
configuration with the production router-evidence helpers, and used only if
they reproduce the saved test decisions and abstentions exactly and the
abstention cut is one of the recomputed training confidences
(``glass_confidences``).

``statements`` reduces the registered comparisons (section 6.5) to what the
comparison reports as a finding: a difference is reported only where its
95% bootstrap CI excludes 0, and the exact McNemar tests at 0.05.

``classify`` applies the section 7.4 table: Deterministic, Numerically
equivalent or Not reproduced, with every condition and the ones that failed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["TOL", "CAP", "ENV_FIELDS", "ENV_PACKAGES", "vector", "threshold", "decisions",
           "configuration", "stage1", "stage3", "glass_confidences", "glass_arbiter",
           "meta_xgb", "environment_differences", "statements", "statement_changes",
           "confidence_room", "classify", "CLASSES"]

#: D8: the deterministic target, a maximum |delta| of 1e-9 per probability vector.
TOL = 1e-9
#: Differing rows listed per decision comparison (all are counted).
CAP = 25
#: Section 7.3: the environment fields a deterministic reproduction must share.
ENV_FIELDS = ("python", "platform")
ENV_PACKAGES = ("numpy", "pandas", "scikit-learn", "interpret", "xgboost", "optuna", "pulp")
CLASSES = ("Deterministic", "Numerically equivalent", "Not reproduced")


def _canon(x) -> Any:
    from upstream_regeneration.records import canonical
    return canonical(x)


def _digest(x) -> str:
    return hashlib.sha256(json.dumps(_canon(x), sort_keys=True).encode("utf-8")).hexdigest()


def _arr(x) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=float).ravel()


def _get(obj, path: str):
    """``obj`` field by dotted path through mappings and attributes; ``KeyError`` if absent."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            raise KeyError(path)
    return cur


# ======================================================================
# building blocks
# ======================================================================
def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def vector(a, b) -> Dict[str, Any]:
    """Numerical drift of one vector: max and 99.9th-percentile |delta|, rank correlation."""
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        return {"comparable": False, "n": [int(a.size), int(b.size)],
                "why": "different lengths", "max_abs": None, "identical": False}
    na, nb = np.isnan(a), np.isnan(b)
    both = ~(na | nb)
    d = np.abs(a[both] - b[both])
    return {"comparable": True, "n": int(a.size), "identical": bool(np.array_equal(a, b, equal_nan=True)),
            "nan_mismatch": int((na != nb).sum()),
            "max_abs": float(d.max()) if d.size else 0.0,
            "p999_abs": float(np.quantile(d, 0.999)) if d.size else 0.0,
            "spearman": _spearman(a[both], b[both])}


def _absent(why: str) -> Dict[str, Any]:
    return {"comparable": False, "why": why, "max_abs": None, "identical": False}


def _drift(v: Optional[Mapping[str, Any]]) -> float:
    """A vector's maximum |delta|; infinite when it cannot be compared."""
    if not v or not v.get("comparable") or v.get("nan_mismatch"):
        return float("inf")
    return float(v["max_abs"])


def _allowance(v: Optional[Mapping[str, Any]]) -> float:
    """The drift a threshold or decision may be explained by: none when it is unknown."""
    d = _drift(v)
    return d if np.isfinite(d) else 0.0


def threshold(ta, tb, chosen_on: Optional[Mapping[str, Any]] = None, *, chosen_on_name=None,
              flags_a=None, flags_b=None) -> Dict[str, Any]:
    """One operating threshold, accepted against reproduced (module docstring)."""
    ta, tb = float(ta), float(tb)
    dt = abs(tb - ta)
    drift = _allowance(chosen_on) if chosen_on is not None else 0.0
    out = {"accepted": ta, "reproduced": tb, "abs_delta": dt, "same_value": dt <= TOL,
           "chosen_on": chosen_on_name, "vector_max_abs": None if chosen_on is None
           else (chosen_on or {}).get("max_abs"),
           "within_vector_drift": dt <= max(drift, TOL)}
    if chosen_on is not None and not np.isfinite(_drift(chosen_on)):
        out["note"] = "the vector it is chosen on is not comparable: only an unchanged value passes"
    if flags_a is not None:
        out["flag_count"] = {"accepted": int(flags_a), "reproduced": int(flags_b),
                             "change": int(flags_b) - int(flags_a)}
    return out


def decisions(da, db, *, score_a=None, score_b=None, ta=0.0, tb=0.0,
              score_drift: Optional[Mapping[str, Any]] = None, index=None) -> Dict[str, Any]:
    """
    Rows whose decision differs; with a score, each one's distance to its threshold
    and whether it lies within the score's drift plus the threshold's movement.
    """
    da, db = np.asarray(da), np.asarray(db)
    if da.shape != db.shape:
        return {"comparable": False, "n": [int(da.size), int(db.size)], "differ": None,
                "all_within_drift": False}
    rows = np.flatnonzero(da != db)
    out = {"comparable": True, "n": int(da.size), "differ": int(rows.size)}
    if not rows.size:
        out["all_within_drift"] = True
        return out
    label = (lambda i: _canon(index[i])) if index is not None else int
    listed = []
    if score_a is not None:
        sa, sb = _arr(score_a), _arr(score_b)
        ta_r = np.broadcast_to(np.asarray(ta, dtype=float), sa.shape)
        tb_r = np.broadcast_to(np.asarray(tb, dtype=float), sb.shape)
        allow = _allowance(score_drift) + float(np.max(np.abs(tb_r - ta_r)))
        if not np.isfinite(_drift(score_drift)):
            out["note"] = "the score's drift is unknown: only the threshold's movement is allowed"
        dist_a, dist_b = np.abs(sa - ta_r)[rows], np.abs(sb - tb_r)[rows]
        within = dist_a <= allow
        out.update(allowed_distance=allow, all_within_drift=bool(within.all()),
                   outside=int((~within).sum()), max_distance=float(dist_a.max()))
        for k, i in enumerate(rows[:CAP]):
            listed.append({"row": label(i), "accepted": _canon(da[i]), "reproduced": _canon(db[i]),
                           "distance_accepted": float(dist_a[k]),
                           "distance_reproduced": float(dist_b[k])})
    else:
        out.update(all_within_drift=False, outside=int(rows.size),
                   note="no score: a differing route or category is outside the condition")
        listed = [{"row": label(i), "accepted": _canon(da[i]), "reproduced": _canon(db[i])}
                  for i in rows[:CAP]]
    out["rows"] = listed
    return out


def configuration(pairs: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    """Each named field equal value for value (canonical JSON), or how it differs."""
    differ = {}
    for name, (a, b) in pairs.items():
        ca, cb = _canon(a), _canon(b)
        if ca != cb:
            text = lambda c: json.dumps(c, sort_keys=True)[:300]         # noqa: E731
            differ[name] = {"accepted": text(ca), "reproduced": text(cb),
                            "accepted_sha256": _digest(a), "reproduced_sha256": _digest(b)}
    return {"identical": not differ, "compared": sorted(pairs), "differences": differ}


def _folds(a, b) -> Dict[str, Any]:
    a, b = np.asarray(a), np.asarray(b)
    same = a.shape == b.shape and bool(np.array_equal(a, b))
    return {"identical": same, "n": int(a.size),
            "differ": None if a.shape != b.shape else int((a != b).sum())}


def _vectors(a, b, paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in paths:
        try:
            va = _get(a, p)
        except KeyError:
            va = None
        try:
            vb = _get(b, p)
        except KeyError:
            vb = None
        if va is None and vb is None:
            continue                                     # not stored by this artifact family
        out[p] = (vector(va, vb) if va is not None and vb is not None
                  else _absent("stored on one side only"))
    return out


# ======================================================================
# Stage 1: StageOutput payloads (GLASS LR, MLP)
# ======================================================================
def stage1(pa: Mapping[str, Any], pb: Mapping[str, Any]) -> Dict[str, Any]:
    from upstream_regeneration.checks import _stage_output
    a, b = _stage_output(pa), _stage_output(pb)
    vec = {"train_proba_oof": vector(a.train_proba_oof, b.train_proba_oof),
           "test_proba": vector(a.test_proba, b.test_proba)}
    cfg = configuration({
        "artifact_version": (pa.get("artifact_version"), pb.get("artifact_version")),
        "feature_contract": (pa.get("feature_contract"), pb.get("feature_contract")),
        "stage / arm / model": ((a.stage, a.arm, a.model), (b.stage, b.arm, b.model)),
        "feature_names": (list(a.feature_names), list(b.feature_names)),
        "best_params": (a.best_params, b.best_params),
        "calibration_requested": (a.calibration_requested, b.calibration_requested),
        "calibration_method": (a.calibration_method, b.calibration_method),
        "threshold_source": (a.threshold_source, b.threshold_source),
        "train_index": (list(a.train_proba_oof.index), list(b.train_proba_oof.index)),
        "test_index": (list(a.test_proba.index), list(b.test_proba.index))})
    oa, ob = _arr(a.train_proba_oof), _arr(b.train_proba_oof)
    xa, xb = _arr(a.test_proba), _arr(b.test_proba)
    ta, tb = float(a.threshold), float(b.threshold)
    return {
        "configuration": cfg,
        "fold_ids": _folds(a.train_fold_id.to_numpy(), b.train_fold_id.to_numpy()),
        "vectors": vec,
        "thresholds": {"operating (F2, on training OOF)": threshold(
            ta, tb, vec["train_proba_oof"], chosen_on_name="train_proba_oof",
            flags_a=(xa >= ta).sum(), flags_b=(xb >= tb).sum())},
        "decisions": {
            "train OOF at the operating threshold": decisions(
                oa >= ta, ob >= tb, score_a=oa, score_b=ob, ta=ta, tb=tb,
                score_drift=vec["train_proba_oof"]),
            "test at the operating threshold": decisions(
                xa >= ta, xb >= tb, score_a=xa, score_b=xb, ta=ta, tb=tb,
                score_drift=vec["test_proba"], index=list(a.test_proba.index))},
    }


# ======================================================================
# Stage 3: Stage3Artifact (GLASS EBM, XGBoost)
# ======================================================================
STAGE3_VECTORS = ("oof.proba", "oof.proba_calibrated", "oof.margin", "refit.test_proba",
                  "refit.test_proba_calibrated", "refit.test_margin", "fold_mean.test_proba",
                  "fold_mean.test_margin")
STAGE3_DECISIONS = (("train OOF", "oof.decision", "oof.margin"),
                    ("test (full-train refit)", "refit.test_decision", "refit.test_margin"),
                    ("test (fold mean)", "fold_mean.test_decision", "fold_mean.test_margin"))


def stage3(a, b) -> Dict[str, Any]:
    cal = lambda x: {k: (x.calibration or {}).get(k)                        # noqa: E731
                     for k in ("applied", "method", "fold_applied")}
    cfg = configuration({
        "model_family / arm": ((a.model_family, (a.config or {}).get("arm")),
                               (b.model_family, (b.config or {}).get("arm"))),
        "features": (list(a.features), list(b.features)),
        "best_params": (a.best_params, b.best_params),
        "calibration family (applied, method, per fold)": (cal(a), cal(b)),
        "block_fingerprint": (a.block_fingerprint, b.block_fingerprint),
        "split_fingerprint": (a.split_fingerprint, b.split_fingerprint),
        "feature_fit_scope": (a.feature_fit_scope, b.feature_fit_scope),
        "class_weights": (a.class_weights, b.class_weights),
        "index_train / index_test": ((list(a.index_train), list(a.index_test)),
                                     (list(b.index_train), list(b.index_test)))})
    vec = _vectors(a, b, STAGE3_VECTORS)
    th = {}
    for name, path, on in (("OOF (F2)", "threshold.oof.threshold", "oof.proba"),
                           ("OOF, calibrated space", "threshold.oof_calibrated_space.threshold",
                            "oof.proba_calibrated")):
        try:
            ta, tb = _get(a, path), _get(b, path)
        except KeyError:
            continue
        flags = {}
        try:
            flags = {"flags_a": (_arr(_get(a, "refit.test_proba")) >= float(ta)).sum(),
                     "flags_b": (_arr(_get(b, "refit.test_proba")) >= float(tb)).sum()} \
                if on == "oof.proba" else {}
        except KeyError:
            pass
        th[name] = threshold(ta, tb, vec.get(on), chosen_on_name=on, **flags)
    try:
        fa, fb = _get(a, "threshold.oof_nested.by_fold"), _get(b, "threshold.oof_nested.by_fold")
        fa, fb = (list(x.values()) if isinstance(x, Mapping) else list(x) for x in (fa, fb))
        if len(fa) == len(fb):
            for k, (x, z) in enumerate(zip(fa, fb)):
                th[f"OOF fold-nested, fold {k}"] = threshold(x, z, vec.get("oof.proba"),
                                                             chosen_on_name="oof.proba")
        else:
            th["OOF fold-nested"] = {"same_value": False, "within_vector_drift": False,
                                     "why": f"{len(fa)} vs {len(fb)} folds"}
    except KeyError:
        pass
    dec = {}
    for name, dpath, mpath in STAGE3_DECISIONS:
        try:
            da, db = _get(a, dpath), _get(b, dpath)
        except KeyError:
            continue
        dec[name] = decisions(da, db, score_a=_get(a, mpath), score_b=_get(b, mpath),
                              score_drift=vec.get(mpath),
                              index=list(a.index_test) if "test" in name else None)
    return {"configuration": cfg, "fold_ids": _folds(a.oof["fold_id"], b.oof["fold_id"]),
            "vectors": vec, "thresholds": th, "decisions": dec}


# ======================================================================
# Stage 4: the GLASS Arbiter (glass_arbiter/1 dicts)
# ======================================================================
def glass_confidences(art: Mapping[str, Any], side: str, y_train=None) -> Dict[str, np.ndarray]:
    """
    ``{"margin": conf1 - conf0, "confidence": max(conf1, conf0)}`` per row on
    ``side`` ("test" or "train"), recomputed from the artifact's saved inputs
    and frozen thresholds and weights exactly as ``meta_arbiter`` computes them
    (the router's evidence through ``router_evidence``, the same products and
    summation order). The training side needs ``y_train`` for each out-of-fold
    row's fold base rate.
    """
    from glass_pipeline.glass_arbiter.router_evidence import (directional_confidence,
                                                              fold_base_rates, rule_positive_rate)
    from shared.stage4 import PASS1, PASS2
    th, w = art["thresholds"], art["weights"]
    if side == "test":
        p, dec, base = art["test_probs"], art["glass_decisions_test"], float(th["glass"])
    else:
        p, dec = art["train_probs"], art["glass_decisions_train"]
        base = fold_base_rates(np.asarray(y_train), np.asarray(art["train_fold_ids"]["glass"]))
    dec = np.asarray(dec).astype(str)
    lr, ebm, g = _arr(p["lr"]), _arr(p["ebm"]), _arr(p["glass"])
    p1, p2 = dec == PASS1, dec == PASS2
    cg = directional_confidence(rule_positive_rate(g, p1, p2, base), p1, p2, base)
    t_lr, t_ebm = float(th["lr"]), float(th["ebm"])
    vl, ve = lr >= t_lr, ebm >= t_ebm
    cl, ce = np.abs(lr - t_lr), np.abs(ebm - t_ebm)
    routed = p1 | p2
    total = (w["lr"] + w["ebm"]) + np.where(routed, w["glass"], 0.0)
    nl, ne, ng = w["lr"] / total, w["ebm"] / total, np.where(routed, w["glass"] / total, 0.0)
    c1 = (cl * nl * vl + ce * ne * ve) + cg * ng * p2
    c0 = (cl * nl * ~vl + ce * ne * ~ve) + cg * ng * p1
    return {"margin": c1 - c0, "confidence": np.maximum(c1, c0)}


def _glass_recompute(art, y_train) -> Dict[str, Any]:
    """Recomputed confidences, used only if they reproduce what the artifact saved."""
    try:
        te = glass_confidences(art, "test")
        tr = glass_confidences(art, "train", y_train)
    except Exception as exc:                            # noqa: BLE001 - recorded
        return {"valid": False, "why": f"{type(exc).__name__}: {exc}"[:300]}
    cut = float(art["thresholds"]["min_weighted_confidence"])
    pred = np.asarray(art["test_preds"]["no_abstain"]).astype(int)
    keep = np.asarray(art["test_preds"]["with_abstain"]).astype(int) != -1
    checks = {
        "test decisions = (conf1 > conf0)": bool(np.array_equal((te["margin"] > 0).astype(int), pred)),
        "test retained = (max(conf1, conf0) >= cut)": bool(np.array_equal(te["confidence"] >= cut, keep)),
        "the cut is one of the training confidences": bool(np.isclose(tr["confidence"], cut,
                                                                     rtol=0, atol=1e-12).any()),
    }
    return {"valid": all(checks.values()), "checks": checks, "test": te, "train": tr}


def glass_arbiter(a: Mapping[str, Any], b: Mapping[str, Any], y_train) -> Dict[str, Any]:
    from shared.stage4 import PASS1, PASS2
    up = lambda x: ((x.get("stage4_inputs") or {}).get("upstream") or {})  # noqa: E731
    ab = lambda x: {k: (x.get("abstention") or {}).get(k)                   # noqa: E731
                    for k in ("selected_by", "min_coverage", "search")}
    tha, thb = a["thresholds"], b["thresholds"]
    cfg = configuration({
        "schema": (a.get("schema"), b.get("schema")),
        "upstream combination": (up(a).get("combination"), up(b).get("combination")),
        "target_recall": (tha.get("target_recall"), thb.get("target_recall")),
        "router_confidence rule": (tha.get("router_confidence"), thb.get("router_confidence")),
        "abstention rule (selected_by, min_coverage, search)": (ab(a), ab(b)),
        "index_train / index_test": ((list(a["index_train"]), list(a["index_test"])),
                                     (list(b["index_train"]), list(b["index_test"])))})
    streams = sorted(set(a.get("train_fold_ids") or {}) | set(b.get("train_fold_ids") or {}))
    folds = {s: _folds((a.get("train_fold_ids") or {}).get(s, []),
                       (b.get("train_fold_ids") or {}).get(s, [])) for s in streams}
    vec = _vectors(a, b, [f"test_probs.{k}" for k in ("lr", "ebm", "glass", "meta")]
                   + [f"train_probs.{k}" for k in ("lr", "ebm", "glass")])
    vec["weights (lr, ebm, glass)"] = vector([a["weights"][k] for k in ("lr", "ebm", "glass")],
                                             [b["weights"][k] for k in ("lr", "ebm", "glass")])
    ra, rb = _glass_recompute(a, y_train), _glass_recompute(b, y_train)
    recompute = {"accepted": {k: v for k, v in ra.items() if k in ("valid", "checks", "why")},
                 "reproduced": {k: v for k, v in rb.items() if k in ("valid", "checks", "why")}}
    valid = ra["valid"] and rb["valid"]
    if valid:
        for side in ("test", "train"):
            vec[f"{side}.weighted_confidence"] = vector(ra[side]["confidence"],
                                                       rb[side]["confidence"])
        vec["test.margin (conf1 - conf0)"] = vector(ra["test"]["margin"], rb["test"]["margin"])
    tla, tlb = _arr(a["test_probs"]["lr"]), _arr(b["test_probs"]["lr"])
    tea, teb = _arr(a["test_probs"]["ebm"]), _arr(b["test_probs"]["ebm"])
    th = {
        "Stage 1 recall cut (stage 1 slot)": threshold(
            tha["lr"], thb["lr"], vec.get("train_probs.lr"), chosen_on_name="train_probs.lr",
            flags_a=(tla >= tha["lr"]).sum(), flags_b=(tlb >= thb["lr"]).sum()),
        "Stage 3 recall cut (stage 3 slot)": threshold(
            tha["ebm"], thb["ebm"], vec.get("train_probs.ebm"), chosen_on_name="train_probs.ebm",
            flags_a=(tea >= tha["ebm"]).sum(), flags_b=(teb >= thb["ebm"]).sum()),
        "router base rate (training labels)": threshold(tha["glass"], thb["glass"]),
        "abstention cut (min_weighted_confidence)": threshold(
            tha["min_weighted_confidence"], thb["min_weighted_confidence"],
            vec.get("train.weighted_confidence") if valid else _absent("not recomputed"),
            chosen_on_name="train.weighted_confidence")}
    fa, fb = (tha.get("glass_train_base_rates") or {}), (thb.get("glass_train_base_rates") or {})
    for k in sorted(set(fa) | set(fb), key=str):
        th[f"router fold base rate, fold {k}"] = (threshold(fa[k], fb[k]) if k in fa and k in fb
                                                  else {"same_value": False,
                                                        "within_vector_drift": False,
                                                        "why": "fold on one side only"})
    pa, pb = (np.asarray(x["test_preds"]["no_abstain"]).astype(int) for x in (a, b))
    ka, kb = (np.asarray(x["test_preds"]["with_abstain"]).astype(int) != -1 for x in (a, b))
    idx = list(a["index_test"])
    dec = {
        "test: stage 1 slot vote": decisions(tla >= tha["lr"], tlb >= thb["lr"], score_a=tla,
                                             score_b=tlb, ta=tha["lr"], tb=thb["lr"],
                                             score_drift=vec.get("test_probs.lr"), index=idx),
        "test: stage 3 slot vote": decisions(tea >= tha["ebm"], teb >= thb["ebm"], score_a=tea,
                                             score_b=teb, ta=tha["ebm"], tb=thb["ebm"],
                                             score_drift=vec.get("test_probs.ebm"), index=idx),
        "Stage 2 routes, training OOF": decisions(np.asarray(a["glass_decisions_train"]).astype(str),
                                                  np.asarray(b["glass_decisions_train"]).astype(str)),
        "Stage 2 routes, test": decisions(np.asarray(a["glass_decisions_test"]).astype(str),
                                          np.asarray(b["glass_decisions_test"]).astype(str),
                                          index=idx),
    }
    if valid:
        dec["test: arbiter decision without abstention"] = decisions(
            pa, pb, score_a=ra["test"]["margin"], score_b=rb["test"]["margin"],
            score_drift=vec["test.margin (conf1 - conf0)"], index=idx)
        dec["test: retained (abstention)"] = decisions(
            ka, kb, score_a=ra["test"]["confidence"], score_b=rb["test"]["confidence"],
            ta=tha["min_weighted_confidence"], tb=thb["min_weighted_confidence"],
            score_drift=vec["test.weighted_confidence"], index=idx)
    else:
        dec["test: arbiter decision without abstention"] = decisions(pa, pb, index=idx)
        dec["test: retained (abstention)"] = decisions(ka, kb, index=idx)
    routes = {s: {st: int((np.asarray(x[f"glass_decisions_{s}"]).astype(str) == st).sum())
                  for st in (PASS1, PASS2)} for s, x in (("train", a), ("test", a))}
    return {"configuration": cfg, "fold_ids": folds, "vectors": vec, "thresholds": th,
            "decisions": dec, "recomputed_confidences": recompute, "accepted_routes": routes}


# ======================================================================
# Stage 4: Meta-XGB (MetaXGBArtifact)
# ======================================================================
def _meta_oof(art) -> np.ndarray:
    L = art.learner
    return _arr(L.oof["proba"] if art.operating.get("space") == "raw"
                else L.oof["proba_calibrated"])


def meta_xgb(a, b) -> Dict[str, Any]:
    cal = lambda x: {k: x.calibration.get(k) for k in ("applied", "method")}   # noqa: E731
    up = lambda x: ((x.stage4_inputs or {}).get("upstream") or {})            # noqa: E731
    ab = lambda x: {k: (x.abstention or {}).get(k)                            # noqa: E731
                    for k in ("selected_by", "confidence", "search", "min_coverage")}
    op = lambda x: {k: x.operating.get(k) for k in ("space", "threshold_key",  # noqa: E731
                                                     "train_side_thresholds")}
    cfg = configuration({
        "schema / model_family / arm": ((a.schema, a.model_family, a.arm),
                                        (b.schema, b.model_family, b.arm)),
        "upstream combination": (up(a).get("combination"), up(b).get("combination")),
        "feature_names": (list(a.feature_names), list(b.feature_names)),
        "best_params": (a.best_params, b.best_params),
        "calibration family": (cal(a), cal(b)),
        "operating rule (space, threshold key, train-side thresholds)": (op(a), op(b)),
        "abstention rule": (ab(a), ab(b)),
        "index_train / index_test": ((list(a.index_train), list(a.index_test)),
                                     (list(b.index_train), list(b.index_test)))})
    vec = _vectors(a, b, ("learner.oof.proba", "learner.oof.proba_calibrated",
                          "learner.oof.threshold_per_row", "test.proba", "test.proba_raw",
                          "test.proba_calibrated"))
    ta, tb = float(a.operating["threshold"]), float(b.operating["threshold"])
    ca, cb = float(a.abstention["min_confidence"]), float(b.abstention["min_confidence"])
    on = "learner.oof.proba" if a.operating.get("space") == "raw" else "learner.oof.proba_calibrated"
    xa, xb = _arr(a.test["proba"]), _arr(b.test["proba"])
    # the cut is chosen on |p - t| of the meta OOF, t fold-nested per row
    try:
        conf = vector(np.abs(_meta_oof(a) - _arr(a.learner.oof["threshold_per_row"])),
                      np.abs(_meta_oof(b) - _arr(b.learner.oof["threshold_per_row"])))
    except KeyError:
        conf = _absent("no per-row training thresholds stored")
    vec["train |p - t| (abstention search)"] = conf
    vec["test |p - t|"] = vector(np.abs(xa - ta), np.abs(xb - tb))
    th = {"operating threshold (F2, meta OOF)": threshold(
              ta, tb, vec.get(on), chosen_on_name=on, flags_a=(xa >= ta).sum(),
              flags_b=(xb >= tb).sum()),
          "abstention cut (min |p - t|)": threshold(
              ca, cb, conf, chosen_on_name="train |p - t| (abstention search)")}
    idx = list(a.index_test)
    dec = {"test: decision without abstention": decisions(
               np.asarray(a.test["pred"]).astype(int), np.asarray(b.test["pred"]).astype(int),
               score_a=xa, score_b=xb, ta=ta, tb=tb, score_drift=vec["test.proba"], index=idx),
           "test: retained (abstention)": decisions(
               np.asarray(a.test["retained"], bool), np.asarray(b.test["retained"], bool),
               score_a=np.abs(xa - ta), score_b=np.abs(xb - tb), ta=ca, tb=cb,
               score_drift=vec["test |p - t|"], index=idx)}
    return {"configuration": cfg, "fold_ids": _folds(a.learner.oof["fold_id"],
                                                     b.learner.oof["fold_id"]),
            "vectors": vec, "thresholds": th, "decisions": dec}


# ======================================================================
# environment, registered statements, confidence room
# ======================================================================
def environment_differences(now: Mapping[str, Any], then: Mapping[str, Any]) -> Dict[str, Any]:
    """The section 7.3 fields that differ (``class_fields``) and every other recorded difference."""
    cls, other = {}, {}
    for f in ENV_FIELDS:
        if now.get(f) != then.get(f):
            cls[f] = {"accepted": then.get(f), "reproduced": now.get(f)}
    pn, pt = now.get("packages") or {}, then.get("packages") or {}
    for pkg in sorted(set(pn) | set(pt)):
        if pn.get(pkg) != pt.get(pkg):
            (cls if pkg in ENV_PACKAGES else other)[pkg] = {"accepted": pt.get(pkg),
                                                           "reproduced": pn.get(pkg)}
    for f in ("os", "executable"):
        if now.get(f) != then.get(f):
            other[f] = {"accepted": then.get(f), "reproduced": now.get(f)}
    return {"class_fields": cls, "other": other}


def statements(comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    """
    What each registered comparison reports as a finding: a metric difference
    only where its 95% CI excludes 0 (with its direction), and the exact
    McNemar tests at 0.05 (with the arbiter more often correct alone).
    """
    out = {}
    for name in sorted(comparisons):
        c = comparisons[name]
        for row in c.get("bootstrap") or []:
            delta = next(v for k, v in row.items() if str(k).startswith("Δ"))
            out[f"{name}: {row['metric']}"] = (
                ("Meta-XGB higher" if delta > 0 else "GLASS Arbiter higher")
                if row.get("CI excludes 0") else "not distinguished (95% CI includes 0)")
        for k in ("no_abstain", "with_abstain_both_retained"):
            t = (c.get("paired") or {}).get(k) or {}
            if "mcnemar_exact_p" not in t:
                continue
            more = ("Meta-XGB" if t["meta_xgb_only_correct"] > t["glass_arbiter_only_correct"]
                    else "GLASS Arbiter")
            out[f"{name}: McNemar {k}"] = (f"{more} more often correct alone (exact p < 0.05)"
                                           if t["mcnemar_exact_p"] < 0.05
                                           else "not distinguished (exact p >= 0.05)")
    return out


def statement_changes(accepted: Mapping[str, str], reproduced: Mapping[str, str]) -> List[dict]:
    return [{"statement": k, "accepted": accepted.get(k), "reproduced": reproduced.get(k)}
            for k in sorted(set(accepted) | set(reproduced)) if accepted.get(k) != reproduced.get(k)]


def confidence_room(key: str, art) -> Dict[str, Any]:
    """
    How far each side of an arbiter's decision can carry confidence, from its
    own frozen configuration, and how often each side's decisions are retained
    on the test rows. For the GLASS Arbiter a negative vote's confidence is at
    most its voter's threshold (LR and EBM slots: the recall cut; router Pass 1:
    the base rate), so a negative decision's weighted confidence is at most the
    largest of them. Meta-XGB's confidence is |p - t| at one threshold t.
    """
    if key.startswith("glass_arbiter"):
        th = art["thresholds"]
        pred = np.asarray(art["test_preds"]["no_abstain"]).astype(int)
        keep = np.asarray(art["test_preds"]["with_abstain"]).astype(int) != -1
        neg = max(float(th["lr"]), float(th["ebm"]), float(th["glass"]))
        pos = max(1 - float(th["lr"]), 1 - float(th["ebm"]), 1 - float(th["glass"]))
        cut = float(th["min_weighted_confidence"])
        rule = "weighted |p - t| of the voters; router: rule rate against the base rate"
        out = {"thresholds": {"stage1_slot": th["lr"], "stage3_slot": th["ebm"],
                              "router_base_rate": th["glass"]}}
    else:
        t, cut = float(art.operating["threshold"]), float(art.abstention["min_confidence"])
        pred = np.asarray(art.test["pred"]).astype(int)
        keep = np.asarray(art.test["retained"], bool)
        neg, pos = t, 1 - t
        rule = f"|p - t| in the {art.operating.get('space')} operating space"
        out = {"thresholds": {"operating": t}}
    share = lambda m: None if not m.sum() else float((keep & m).sum() / m.sum())  # noqa: E731
    out.update(confidence=rule, negative_side_bound=neg, positive_side_bound=pos,
               abstention_cut=cut, cut_above_negative_bound=bool(cut > neg),
               test_retained_share={"negative decisions": share(pred == 0),
                                    "positive decisions": share(pred == 1)})
    return out


# ======================================================================
# the class
# ======================================================================
def _items(roles: Mapping[str, Mapping[str, Any]], part: str):
    for role, r in roles.items():
        block = r.get(part) or {}
        for name, v in block.items():
            yield f"{role}: {name}", v


def classify(roles: Mapping[str, Mapping[str, Any]], *, environment: Mapping[str, Any],
             stage2_inputs_identical: bool, statement_changes_: Sequence[Mapping[str, Any]]
             ) -> Dict[str, Any]:
    """
    The section 7.4 class from the per-artifact comparisons. ``environment``:
    ``{reference run: environment_differences(...)}``.
    """
    folds = []
    for role, r in roles.items():
        f = r.get("fold_ids") or {}
        folds += ([(f"{role}", f)] if "identical" in f else
                  [(f"{role}: {s}", v) for s, v in f.items()])
    vecs = dict(_items(roles, "vectors"))
    ths = dict(_items(roles, "thresholds"))
    decs = dict(_items(roles, "decisions"))
    drift = [_drift(v) for v in vecs.values()]
    failing = lambda pairs, ok: sorted(k for k, v in pairs.items() if not ok(v))  # noqa: E731
    cond = {
        "same recorded environment (section 7.3 fields)":
            [k for k, d in environment.items() if (d or {}).get("class_fields")],
        "identical selected configurations": sorted(r for r, x in roles.items()
                                                    if not x["configuration"]["identical"]),
        "identical fold ids": [k for k, f in folds if not f.get("identical")],
        "identical Stage 2 inputs (SHA-256)": [] if stage2_inputs_identical else ["stage2"],
        "every vector comparable": failing(vecs, lambda v: v.get("comparable")
                                           and not v.get("nan_mismatch")),
        "probability vectors within max |delta| 1e-9": failing(vecs, lambda v: _drift(v) <= TOL),
        "identical thresholds": failing(ths, lambda t: t.get("same_value")),
        "every threshold the same or within its vector's drift": failing(
            ths, lambda t: t.get("same_value") or t.get("within_vector_drift")),
        "identical decisions, abstentions and routes": failing(
            decs, lambda d: d.get("comparable") and d.get("differ") == 0),
        "every differing decision within drift of its threshold": failing(
            decs, lambda d: d.get("comparable") and d.get("all_within_drift")),
        "no registered comparison statement changes": [s["statement"] for s in statement_changes_],
    }
    deterministic = ("same recorded environment (section 7.3 fields)",
                     "identical selected configurations", "identical fold ids",
                     "identical Stage 2 inputs (SHA-256)", "every vector comparable",
                     "probability vectors within max |delta| 1e-9", "identical thresholds",
                     "identical decisions, abstentions and routes",
                     "no registered comparison statement changes")
    equivalent = ("identical selected configurations", "identical fold ids",
                  "identical Stage 2 inputs (SHA-256)", "every vector comparable",
                  "every threshold the same or within its vector's drift",
                  "every differing decision within drift of its threshold",
                  "no registered comparison statement changes")
    holds = {k: not v for k, v in cond.items()}
    cls = ("Deterministic" if all(holds[k] for k in deterministic) else
           "Numerically equivalent" if all(holds[k] for k in equivalent) else "Not reproduced")
    finite = [d for d in drift if np.isfinite(d)]
    return {"class": cls, "conditions": {k: {"holds": holds[k], "failing": v[:40],
                                             "n_failing": len(v)} for k, v in cond.items()},
            "deterministic_requires": list(deterministic),
            "numerically_equivalent_requires": list(equivalent),
            "max_abs_drift": max(finite) if finite else None,
            "vectors_compared": len(vecs), "thresholds_compared": len(ths),
            "decision_sets_compared": len(decs)}
