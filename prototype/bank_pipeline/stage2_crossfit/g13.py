"""
stage2_crossfit.g13
===================
Gate G13 (PR 37 F17 decision record, section 8; limitation L1): does a
training row's own label reach the engineered features, and their bins, that
decide its inner-OOF eligibility, in either arm?

The Stage 2 feature engineer (``shared.stage2.feature_engineering``) is fitted
once per outer fold on all of P_k with its labels. Three of its encoders read
the labels, and each feeds bins the routers use:

    neighborhood_subscription_density  KDE over reference rows' labels   nsd_*
    economic_curvature_intensity       curvature of label-rate bins      eci_*
    dow_month_encoded                  Laplace day x month label rate    dow_*

Method (the F17 option "refitted inside its inner fold"). For one outer fold,
the inner folds are the arms' own (GLASS ``inner_fold_id_``; the RF Router's
Pass 1 OOF folds, which coincide). For each inner fold j the rows of j get
features from encoders that never read fold j's labels -- their own label
included -- and are binned with the locked thresholds. Two variants:

``label_excluded`` (the audit)
    Only the label-derived statistics are recomputed without fold j: the
    KDE drops the reference points that are fold-j rows, and the bin and
    day x month rates are recounted without fold-j rows. The label-free
    structure (scaler, KDE reference sample, quantile bin edges, smoothing)
    stays the outer fit's, so every change is caused by labels.
``inner_refit`` (context)
    A fresh engineer fitted on the rows outside j. This also moves the
    label-free structure (a new KDE reference sample, new quantile edges), so
    its changes are an upper bound mixing label and sampling effects.

Each arm's inner-OOF eligibility is then recomputed through two channels:

``eligibility`` (direct: fold j's own features, models fixed)
    Fold j's rows get the ``label_excluded`` bins; every fitted model is held
    fixed. GLASS routes fold j with its inner fold's Pass 1 rules; the RF
    Router rescores fold j with that inner fold's Pass 1 forest (refitted from
    the same seed and rows, checked to reproduce the stored OOF scores) and
    cuts at the row's ``t1_j`` (``nested_t1``) or ``t1``. All three encoders
    together and each alone.
``eligibility_retrained`` (indirect: the inner models' training features)
    The inner models for fold j were trained on rows outside j whose features
    the outer engineer computed with fold j's labels. Here every row gets the
    features of ``label_excluded_engineer`` without fold j, and fold j's inner
    models are refitted on them: GLASS re-runs inner fold j's Pass 1 fit
    (nested ranker, generation, ILP) on its training rows; the RF Router refits
    fold j's Pass 1 forest and re-solves ``t1_j`` (``nested_t1``) from forests
    fitted on the rows outside j. Fold j is routed and cut with those, so no
    label of fold j reaches its eligibility through any engineered feature.
    The same re-runs on the base bins must reproduce the stored fit (RF:
    scores and ``t1_j``, else the audit stops; GLASS: the selected rules,
    recorded per fold, since a time-limited CBC solve need not reproduce).
    All three encoders together.

A fit on the corrected pathway (``inner_features``, ``fold_label_excluded``)
is audited the other way round: its recorded inner bins must equal the
audit's own label-excluded recomputation (``isolation.pathway_match``), its
inner fits re-run on them must reproduce its rules, scores and ``t1_j``
(``reproduced_all``), and the fits re-run on the base bins give the
correction's effect (``eligibility_retrained``, ``stored_features``
``fold_label_excluded``). The direct channel does not apply to it.

Every row whose bins or eligibility change is recorded by index label. Nothing
is held out, scored or evaluated, and no test row or label is read. Whether
the dependency is material is the owner's decision, documented before any
primary result; no correction is adopted here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from blackbox_pipeline.models.rf_router.config import ELIGIBILITY_NESTED_T1
from blackbox_pipeline.models.rf_router.forest import ForestTrainer
from blackbox_pipeline.models.rf_router.oof.scorer import OOFScorer
from glass_pipeline.glass_router.core.config import INNER_RANKER_FOLD_NESTED_RF
from glass_pipeline.glass_router.pipeline import GlassRouterPipeline
from glass_pipeline.glass_router.pipeline.glass_router_pipeline import PASS1_REPORT_KEY
from shared.stage2.binning import RF_FEATURES_BINARY
from shared.stage2.feature_engineering import (
    _KDE_KEY_FEATURES, _KDE_LARGE_DS_THRESH, RF_SOURCE_FEATURES, RFFeatureEngineer,
    _apply_binning_thresholds,
)
from shared.stage_io import label_fingerprint

from . import evidence as E
from .evidence import PATHWAY_FOLD, PATHWAY_OUTER

__all__ = ["G13_FORMAT", "LABEL_FITTED_FEATURES", "kde_reference_positions",
           "label_excluded_engineer", "fold_excluded_features", "fold_excluded_frames",
           "inner_refit_features", "bin_source", "compare_features", "glass_eligibility_shift",
           "rf_eligibility_shift", "glass_retrained_shift", "rf_retrained_shift",
           "glass_correction_shift", "rf_correction_shift", "audit_fold"]

G13_FORMAT = "pr38.stage2.g13_audit/3"
#: The label-fitted Stage 2 source features and the bin families they feed.
LABEL_FITTED_FEATURES = {
    "neighborhood_subscription_density": ("KDE over reference labels", "nsd_"),
    "economic_curvature_intensity": ("curvature of label-rate bins", "eci_"),
    "dow_month_encoded": ("Laplace day x month label rate", "dow_"),
}
_FAMILY = {"nsd_": "nsd", "jed_": "jed", "cci_": "cci", "eci_": "eci", "dow_": "dow",
           "behav_": "behav", "campaign_": "campaign", "cpi_": "cpi"}


# ----------------------------------------------------------------------
# Label-excluded encoders
# ----------------------------------------------------------------------
def kde_reference_positions(engineer: RFFeatureEngineer, X: pd.DataFrame) -> np.ndarray:
    """The positions in ``X`` (the fit rows) of the KDE reference points, verified."""
    ie = engineer._integral_eng
    n = len(X)
    if n > _KDE_LARGE_DS_THRESH:                      # the engineer's own sampling rule
        pos = np.random.default_rng(ie.random_state).choice(n, size=min(ie.n_ref, n),
                                                           replace=False)
    else:
        pos = np.arange(n)
    scaled = ie._scaler.transform(X[list(_KDE_KEY_FEATURES)])[pos]
    if not np.array_equal(scaled, ie._X_ref):
        raise AssertionError("the KDE reference points are not the rows the audit identifies; "
                             "the engineer was not fitted on X")
    return pos


def _fit_bins(X: pd.DataFrame, feat: str, n_bins: int, categories) -> pd.Series:
    """
    Each row's bin exactly as ``_DerivativeFeatureEngineer.fit`` assigned it (the
    quantile cut of the fit rows, label-free), checked against the stored categories.
    """
    values = pd.Series(X[feat].values)
    try:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(values, bins=n_bins, duplicates="drop")
    if not bins.cat.categories.equals(categories):
        raise AssertionError(f"{feat}: the fit's bins are not the stored ones; the engineer was "
                             "not fitted on X")
    return bins


def _bin_rates(bins: pd.Series, y: np.ndarray, keep: np.ndarray) -> pd.Series:
    """The fit's per-bin label rate over the kept rows only (``groupby(observed=True)``)."""
    df = pd.DataFrame({"y": y})
    return df[keep].groupby(bins[keep], observed=True)["y"].mean()


def label_excluded_engineer(engineer: RFFeatureEngineer, X: pd.DataFrame, y,
                            keep: np.ndarray) -> RFFeatureEngineer:
    """
    A copy of ``engineer`` (fitted on ``X``, ``y``) whose label-derived
    statistics use only the rows ``keep``; its label-free structure is unchanged.
    With every row kept it reproduces ``engineer`` exactly.
    """
    y = np.asarray(y).astype(int)
    keep = np.asarray(keep, dtype=bool)
    eng = copy.deepcopy(engineer)
    # KDE: drop the reference points that are excluded rows (same sample otherwise).
    ref = kde_reference_positions(engineer, X)
    keep_ref = keep[ref]
    eng._integral_eng._X_ref = engineer._integral_eng._X_ref[keep_ref]
    eng._integral_eng._y_ref = engineer._integral_eng._y_ref[keep_ref]
    # Bin rates and curvature on the fit's own bins, kept rows only.
    de = eng._derivative_eng
    Xk, yk = X[keep], y[keep]
    for feat, (cats, _) in list(engineer._derivative_eng._local_rate_store.items()):
        de._local_rate_store[feat] = (cats, _bin_rates(_fit_bins(X, feat, de.n_bins, cats), y,
                                                       keep))
    for feat, (cats, _) in list(engineer._derivative_eng._curvature_store.items()):
        rates = _bin_rates(_fit_bins(X, feat, de.n_bins, cats), y, keep)
        values = rates.values
        curvature = (np.abs(np.gradient(np.gradient(values))) if len(values) >= 3
                     else np.zeros_like(values))
        de._curvature_store[feat] = (cats, dict(zip(rates.index, curvature)))
    # Laplace day x month rates, kept rows only (same smoothing).
    te = eng._temporal_eng
    df = pd.DataFrame({"dow": Xk["day_of_week"].to_numpy(), "month": Xk["month"].to_numpy(),
                       "y": yk})
    te._global_rate = float(df["y"].mean())
    k = te.smoothing_factor
    stats = df.groupby(["dow", "month"])["y"].agg(["sum", "count"])
    te._rates = {(int(d), int(m)): (row["sum"] + k * te._global_rate) / (row["count"] + k)
                 for (d, m), row in stats.iterrows()}
    return eng


def fold_excluded_features(engineer: RFFeatureEngineer, X: pd.DataFrame, y,
                           groups: np.ndarray) -> pd.DataFrame:
    """Source features of each group's rows from ``label_excluded_engineer`` without that group."""
    groups = np.asarray(groups)
    parts = []
    for g in np.unique(groups):
        rows = groups == g
        parts.append(label_excluded_engineer(engineer, X, y, ~rows).transform(X[rows]))
    return pd.concat(parts).loc[X.index]


def fold_excluded_frames(engineer: RFFeatureEngineer, X: pd.DataFrame, y,
                         groups: np.ndarray) -> Dict[int, pd.DataFrame]:
    """
    ``{j: source features of EVERY row from label_excluded_engineer without
    group j}``: what fold j's inner models are trained and scored on once fold
    j's labels are excluded. Rows of j equal ``fold_excluded_features``.
    """
    groups = np.asarray(groups)
    return {int(g): label_excluded_engineer(engineer, X, y, groups != g).transform(X)
            for g in np.unique(groups)}


def inner_refit_features(X: pd.DataFrame, y, groups: np.ndarray) -> pd.DataFrame:
    """Source features of each group's rows from a fresh engineer fitted without that group."""
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    groups = np.asarray(groups)
    parts = []
    for g in np.unique(groups):
        rows = groups == g
        parts.append(RFFeatureEngineer().fit(X[~rows], y[~rows]).transform(X[rows]))
    return pd.concat(parts).loc[X.index]


def bin_source(source: pd.DataFrame) -> pd.DataFrame:
    """The locked 29 bins (label-free thresholds; ``engineer_features``' test-side path)."""
    return _apply_binning_thresholds(source[RF_SOURCE_FEATURES], RF_FEATURES_BINARY)[
        RF_FEATURES_BINARY].astype("int64")


def _family(col: str) -> str:
    for prefix, name in _FAMILY.items():
        if col.startswith(prefix):
            return name
    return col


def compare_features(base_src, alt_src, base_bins, alt_bins, y, groups) -> Dict[str, Any]:
    """Rows whose source features or bins differ, per feature, bin family and inner fold."""
    y = np.asarray(y).astype(int)
    idx = base_src.index
    out: Dict[str, Any] = {"source_features": {}, "bin_families": {}}
    for feat in RF_SOURCE_FEATURES:
        diff = np.abs(alt_src[feat].to_numpy(float) - base_src[feat].to_numpy(float))
        changed = diff > 0
        out["source_features"][feat] = {
            "label_fitted": feat in LABEL_FITTED_FEATURES,
            "rows_changed": int(changed.sum()),
            "max_abs_change": float(diff.max()) if len(diff) else 0.0,
            "median_abs_change_where_changed": (float(np.median(diff[changed]))
                                                if changed.any() else 0.0)}
    any_bin = np.zeros(len(idx), dtype=bool)
    fams = {}
    for col in RF_FEATURES_BINARY:
        fams.setdefault(_family(col), []).append(col)
    for fam, cols in fams.items():
        moved = (base_bins[cols].to_numpy() != alt_bins[cols].to_numpy()).any(axis=1)
        any_bin |= moved
        out["bin_families"][fam] = {"rows_changed": int(moved.sum()),
                                    "positives_changed": int(y[moved].sum())}
    out["rows_any_bin_changed"] = int(any_bin.sum())
    out["positives_any_bin_changed"] = int(y[any_bin].sum())
    out["share_rows_any_bin_changed"] = float(any_bin.mean()) if len(idx) else 0.0
    out["per_inner_fold"] = [{"inner_fold": int(g), "rows": int((groups == g).sum()),
                              "rows_any_bin_changed": int((any_bin & (groups == g)).sum())}
                             for g in np.unique(groups)]
    out["rows_any_bin_changed_index"] = E.to_json_safe(idx[any_bin])
    return out


# ----------------------------------------------------------------------
# Eligibility under the label-excluded bins, every fitted model held fixed
# ----------------------------------------------------------------------
def _eligibility_summary(base: np.ndarray, alt: np.ndarray, y: np.ndarray, index,
                         groups) -> Dict[str, Any]:
    gained, lost = alt & ~base, base & ~alt
    changed = gained | lost
    return {"eligible_rows": int(base.sum()), "eligible_rows_label_excluded": int(alt.sum()),
            "rows_changed": int(changed.sum()), "positives_changed": int(y[changed].sum()),
            "share_of_fit_rows": float(changed.mean()) if len(y) else 0.0,
            "share_of_remainder": (float(changed.sum() / base.sum()) if base.sum() else None),
            "became_eligible": int(gained.sum()), "became_ineligible": int(lost.sum()),
            "per_inner_fold": [{"inner_fold": int(g),
                                "rows_changed": int((changed & (groups == g)).sum())}
                               for g in np.unique(groups)],
            "rows_changed_index": E.to_json_safe(pd.Index(index)[changed])}


def _variants(alt) -> Dict[str, pd.DataFrame]:
    return dict(alt) if isinstance(alt, Mapping) else {"all": alt}


def _unwrap(alt, out: Dict[str, Any]):
    return out if isinstance(alt, Mapping) else out["all"]


def glass_eligibility_shift(glass, base_bins: pd.DataFrame, alt, y) -> Dict[str, Any]:
    """
    GLASS ``oof_remainder``: each inner fold's rows routed by its own Pass 1
    rules, with the base bins (must reproduce the stored remainder) and with
    ``alt`` (one frame, or ``{name: frame}`` for several variants).
    """
    y = np.asarray(y).astype(int)
    inner = np.asarray(glass.inner_fold_id_)
    variants = _variants(alt)
    base_routed = np.zeros(len(y), dtype=bool)
    alt_routed = {name: np.zeros(len(y), dtype=bool) for name in variants}
    for j, rules in enumerate(glass.inner_pass1_rules_):
        rows = inner == j
        base_routed[rows] = GlassRouterPipeline._rules_mask(rules, base_bins[rows])
        for name, frame in variants.items():
            alt_routed[name][rows] = GlassRouterPipeline._rules_mask(rules, frame[rows])
    stored = ~np.asarray(glass.fit_report_["remainder_mask"], dtype=bool)
    if not np.array_equal(base_routed, stored):
        raise AssertionError("GLASS: the inner Pass 1 rules do not reproduce the stored remainder")
    return _unwrap(alt, {name: _eligibility_summary(~base_routed, ~routed, y, base_bins.index,
                                                    inner)
                         for name, routed in alt_routed.items()})


def rf_eligibility_shift(router, base_bins: pd.DataFrame, alt, y, *,
                         reproduce_tol: float = 1e-9) -> Dict[str, Any]:
    """
    RF Router: each inner fold's Pass 1 forest (same seed, same rows, fitted
    once) rescores its fold with the base bins (must reproduce the stored OOF
    scores) and with each ``alt`` variant; the cut is the row's ``t1_j`` under
    ``nested_t1`` (else ``t1``) and must reproduce the stored remainder.
    """
    cfg = router.config
    y = np.asarray(y).astype(int)
    inner = np.asarray(router.oof_pass1_fold_id_)
    p_base = np.asarray(router.oof_pass1_proba_, dtype=float)
    variants = _variants(alt)
    p_alt = {name: np.full(len(y), np.nan) for name in variants}
    trainer = ForestTrainer(cfg.rf_params, cfg.random_state)
    worst = 0.0
    for j in np.unique(inner):
        rows = inner == j
        pipe = trainer.fit(base_bins[~rows], pd.Series(y[~rows]))
        rescored = ForestTrainer.positive_proba(pipe, base_bins[rows])
        worst = max(worst, float(np.max(np.abs(rescored - p_base[rows]))))
        for name, frame in variants.items():
            p_alt[name][rows] = ForestTrainer.positive_proba(pipe, frame[rows])
    if worst > reproduce_tol:
        raise AssertionError(f"RF: the refitted inner forests do not reproduce the stored OOF "
                             f"Pass 1 scores (max difference {worst:.3g})")
    if getattr(cfg, "eligibility", None) == ELIGIBILITY_NESTED_T1:
        rep = router.fit_report_.remainder
        cut = np.full(len(y), np.nan)
        for j, t in zip(rep["eligibility_fold_ids"], rep["eligibility_fold_t1"]):
            cut[inner == j] = t
    else:
        cut = np.full(len(y), float(router.thresholds_.t1))
    base = p_base >= cut
    if not np.array_equal(base, np.asarray(router.remainder_mask_, dtype=bool)):
        raise AssertionError("RF: the cut does not reproduce the stored remainder")
    out = {}
    for name, p in p_alt.items():
        out[name] = _eligibility_summary(base, p >= cut, y, base_bins.index, inner)
        out[name]["max_score_reproduction_difference"] = worst
        out[name]["max_score_change"] = float(np.max(np.abs(p - p_base)))
    return _unwrap(alt, out)


# ----------------------------------------------------------------------
# Eligibility with fold j's inner models refitted without fold j's labels
# ----------------------------------------------------------------------
def _segments(rules):
    return sorted(tuple(sorted((str(f), str(v)) for f, v in r.segment)) for r in rules)


def _glass_reselect(glass, j: int, bins: pd.DataFrame, y_s: pd.Series, rows: np.ndarray):
    """Inner fold j's Pass 1 fit (``_inner_pass1_fit``) on ``bins``' rows outside j."""
    selectors: list = []
    nested = glass.inner_feature_ranker == INNER_RANKER_FOLD_NESTED_RF
    try:
        rules, error = glass._inner_pass1_fit(j, bins[~rows], y_s[~rows], nested, [], [],
                                              selectors), None
    except ValueError as exc:     # infeasible / no solution (SelectorInfeasibleError): recorded
        rules, error = None, f"{type(exc).__name__}: {exc}"
    report = ((getattr(selectors[-1], "selection_report_", None) or {}).get(PASS1_REPORT_KEY)
              if selectors else None) or {}
    return rules, {"source": report.get("source"),
                   "lexicographic_guarantee": report.get("lexicographic_guarantee"),
                   "rules": None if rules is None else len(rules), "error": error}


def _glass_inner_shift(glass, stored: Mapping[int, pd.DataFrame],
                       alt: Mapping[int, pd.DataFrame], index, y, *,
                       reproduce: bool) -> Dict[str, Any]:
    """
    Inner fold j's stored rules must route fold j's rows of ``stored[j]`` (the
    bins the fit used) exactly as stored. ``reproduce`` re-runs fold j's Pass 1
    fit on ``stored[j]``; the fit is then re-run on ``alt[j]`` and fold j routed
    with those rules and bins.
    """
    y = np.asarray(y).astype(int)
    y_s = pd.Series(y, index=index)
    inner = np.asarray(glass.inner_fold_id_)
    base_routed = ~np.asarray(glass.fit_report_["remainder_mask"], dtype=bool)
    alt_routed = base_routed.copy()
    folds, complete = [], True
    for j, rules_s in enumerate(glass.inner_pass1_rules_):
        rows = inner == j
        if not np.array_equal(GlassRouterPipeline._rules_mask(rules_s, stored[j][rows]),
                              base_routed[rows]):
            raise AssertionError(f"GLASS inner fold {j}: the stored rules do not reproduce the "
                                 "stored remainder")
        rec: Dict[str, Any] = {"inner_fold": j, "stored_rules": len(rules_s)}
        if reproduce:
            rules_b, rec["base_reselection"] = _glass_reselect(glass, j, stored[j], y_s, rows)
            rec["reproduced"] = rules_b is not None and _segments(rules_b) == _segments(rules_s)
        rules_a, rec["retrained_selection"] = _glass_reselect(glass, j, alt[j], y_s, rows)
        if rules_a is None:
            complete = False
            rec["compared"] = False
        else:
            alt_routed[rows] = GlassRouterPipeline._rules_mask(rules_a, alt[j][rows])
            seg_a, seg_s = set(_segments(rules_a)), set(_segments(rules_s))
            rec.update(compared=True, rules_identical=seg_a == seg_s,
                       rules_added=len(seg_a - seg_s), rules_removed=len(seg_s - seg_a),
                       rows_changed=int((alt_routed[rows] != base_routed[rows]).sum()))
        folds.append(rec)
    out = _eligibility_summary(~base_routed, ~alt_routed, y, index, inner)
    out.update(folds=folds, complete=complete,
               reproduced_all=(all(f["reproduced"] for f in folds) if reproduce else None))
    return out


def glass_retrained_shift(glass, base_bins: pd.DataFrame, frame_bins: Mapping[int, pd.DataFrame],
                          y, *, reproduce: bool = True) -> Dict[str, Any]:
    """
    GLASS ``oof_remainder``: inner fold j's Pass 1 fit re-run on ``frame_bins[j]``
    (every row's bins without fold j's labels), then fold j routed with those
    rules and bins. ``reproduce`` re-runs the fit on the base bins too; a fold
    whose stored rules do not reproduce is flagged (``reproduced: False``).
    """
    base = {j: base_bins for j in range(len(glass.inner_pass1_rules_))}
    out = _glass_inner_shift(glass, base, frame_bins, base_bins.index, y, reproduce=reproduce)
    out.update(stored_features=PATHWAY_OUTER, alt_features=PATHWAY_FOLD)
    return out


def glass_correction_shift(glass, base_bins: pd.DataFrame,
                           frame_bins: Mapping[int, pd.DataFrame], y) -> Dict[str, Any]:
    """
    A GLASS fit on the corrected pathway: its stored remainder must come from
    ``frame_bins`` (the audit's own recomputation), every inner Pass 1 fit
    re-run on ``frame_bins[j]`` must reproduce its rules (``reproduced``), and
    the fit re-run on the base bins gives the correction's effect.
    """
    base = {j: base_bins for j in range(len(glass.inner_pass1_rules_))}
    out = _glass_inner_shift(glass, frame_bins, base, base_bins.index, y, reproduce=True)
    out.update(stored_features=PATHWAY_FOLD, alt_features=PATHWAY_OUTER)
    return out


def _t1_without(router, scorer: OOFScorer, bins: pd.DataFrame, y_s: pd.Series,
                rows: np.ndarray, j: int) -> float:
    """``t1_j`` exactly as ``nested_pass1_eligibility`` solves it, on ``bins``."""
    sub = scorer.score(bins, y_s, subset_mask=~rows, label=f"g13 eligibility[{j}]")
    res = router.solver.solve_pass1(sub.scored_proba(), sub.scored_labels(y_s), keep_sweep=False)
    res.raise_if_infeasible(context=f"G13 nested eligibility, inner fold {j}")
    return float(res.threshold)


def _rf_inner_shift(router, stored: Mapping[int, pd.DataFrame], alt: Mapping[int, pd.DataFrame],
                    index, y, *, reproduce_tol: float) -> Dict[str, Any]:
    """
    Fold j's Pass 1 forest refitted on ``stored[j]`` must reproduce the stored
    OOF scores, and (``nested_t1``) ``t1_j`` re-solved on ``stored[j]`` the
    stored ``t1_j``; otherwise the audit stops. Then both are refitted on
    ``alt[j]`` and fold j cut with them (``global_t1``: the stored ``t1`` held).
    """
    cfg = router.config
    y = np.asarray(y).astype(int)
    y_s = pd.Series(y, index=index)
    inner = np.asarray(router.oof_pass1_fold_id_)
    p_base = np.asarray(router.oof_pass1_proba_, dtype=float)
    base = np.asarray(router.remainder_mask_, dtype=bool)
    nested = getattr(cfg, "eligibility", None) == ELIGIBILITY_NESTED_T1
    trainer = ForestTrainer(cfg.rf_params, cfg.random_state)
    scorer = OOFScorer(trainer, cfg.eligibility_folds, cfg.random_state, cfg.stratify_oof,
                       verbose=False)
    stored_t1 = (dict(zip(router.fit_report_.remainder["eligibility_fold_ids"],
                          router.fit_report_.remainder["eligibility_fold_t1"]))
                 if nested else None)
    alt_rem = base.copy()
    folds, complete, worst_t1, worst_p = [], True, 0.0, 0.0
    for j in (int(g) for g in np.unique(inner)):
        rows = inner == j
        S, A = stored[j], alt[j]
        p_s = ForestTrainer.positive_proba(trainer.fit(S[~rows], pd.Series(y[~rows])), S[rows])
        worst_p = max(worst_p, float(np.max(np.abs(p_s - p_base[rows]))))
        p_alt = ForestTrainer.positive_proba(trainer.fit(A[~rows], pd.Series(y[~rows])), A[rows])
        rec: Dict[str, Any] = {"inner_fold": j,
                               "max_score_change": float(np.max(np.abs(p_alt - p_base[rows])))}
        if nested:
            rec["t1_j"] = float(stored_t1[j])
            worst_t1 = max(worst_t1, abs(_t1_without(router, scorer, S, y_s, rows, j)
                                         - rec["t1_j"]))
            try:
                t1_a = _t1_without(router, scorer, A, y_s, rows, j)
            except ValueError as exc:            # infeasible t1_j: recorded, no fallback
                rec.update(compared=False, error=f"{type(exc).__name__}: {exc}")
                complete = False
                folds.append(rec)
                continue
            rec["t1_j_retrained"] = t1_a
        else:
            t1_a = float(router.thresholds_.t1)
            rec["t1_held"] = t1_a
        alt_rem[rows] = p_alt >= t1_a
        rec.update(compared=True, rows_changed=int((alt_rem[rows] != base[rows]).sum()))
        folds.append(rec)
    if worst_p > reproduce_tol:
        raise AssertionError(f"RF: refitting the inner forests on the bins the fit used does not "
                             f"reproduce the stored OOF scores (max difference {worst_p:.3g})")
    if worst_t1 > reproduce_tol:
        raise AssertionError(f"RF: re-solving t1_j on the bins the fit used does not reproduce "
                             f"the stored t1_j (max difference {worst_t1:.3g})")
    out = _eligibility_summary(base, alt_rem, y, index, inner)
    out.update(folds=folds, complete=complete, cut=("t1_j re-solved" if nested else "t1 held"),
               t1_reproduction_max_difference=worst_t1 if nested else None,
               score_reproduction_max_difference=worst_p, reproduced_all=True)
    return out


def rf_retrained_shift(router, base_bins: pd.DataFrame, frame_bins: Mapping[int, pd.DataFrame],
                       y, *, reproduce_tol: float = 1e-9) -> Dict[str, Any]:
    """
    RF Router: fold j's Pass 1 forest refitted on ``frame_bins[j]`` rows outside
    j and fold j rescored on ``frame_bins[j]``; under ``nested_t1`` its ``t1_j``
    is re-solved from forests fitted on ``frame_bins[j]`` outside j (the same
    refits and solve on the base bins must reproduce the stored scores and
    ``t1_j``). Under ``global_t1`` the stored ``t1`` is held (recorded).
    """
    base = {int(j): base_bins for j in np.unique(router.oof_pass1_fold_id_)}
    out = _rf_inner_shift(router, base, frame_bins, base_bins.index, y,
                          reproduce_tol=reproduce_tol)
    out.update(stored_features=PATHWAY_OUTER, alt_features=PATHWAY_FOLD)
    return out


def rf_correction_shift(router, base_bins: pd.DataFrame, frame_bins: Mapping[int, pd.DataFrame],
                        y, *, reproduce_tol: float = 1e-9) -> Dict[str, Any]:
    """
    An RF fit on the corrected pathway: its stored OOF scores and ``t1_j`` must
    reproduce from ``frame_bins`` (the audit's own recomputation), or the audit
    stops; refitting on the base bins gives the correction's effect.
    """
    base = {int(j): base_bins for j in np.unique(router.oof_pass1_fold_id_)}
    out = _rf_inner_shift(router, frame_bins, base, base_bins.index, y,
                          reproduce_tol=reproduce_tol)
    out.update(stored_features=PATHWAY_FOLD, alt_features=PATHWAY_OUTER)
    return out


# ----------------------------------------------------------------------
# One outer fold, both arms
# ----------------------------------------------------------------------
def audit_fold(X_tr: pd.DataFrame, y_tr: pd.Series, *, fold: int, glass=None,
               glass_bins: Optional[pd.DataFrame] = None, rf_router=None,
               rf_bins: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    The G13 audit of one outer fold. ``glass`` / ``rf_router`` are that fold's
    fitted routers (``runner.fit_fold(..., keep_fits=True)``), with the bins
    each was fitted on. Either arm may be omitted. Both channels are recorded
    (module docstring); ``eligibility_retrained`` re-runs each arm's inner fits.
    """
    y = np.asarray(y_tr).astype(int)
    engineer = RFFeatureEngineer().fit(X_tr, y_tr)
    base_src = engineer.transform(X_tr)
    base_bins = bin_source(base_src)
    arms_inner = {}
    for name, bins, folds in (
            ("glass", glass_bins, None if glass is None else glass.inner_fold_id_),
            ("rf", rf_bins, None if rf_router is None else rf_router.oof_pass1_fold_id_)):
        if bins is None:
            continue
        if not bins.astype("int64").equals(base_bins):
            raise AssertionError(f"{name}: the audit's bins are not the bins the arm was fitted on")
        arms_inner[name] = np.asarray(folds)
    if not arms_inner:
        raise ValueError("audit_fold needs at least one fitted arm")
    if glass is not None and glass.fit_report_.get("pass2_population") != "oof_remainder":
        raise ValueError("G13: the GLASS fit must be the oof_remainder primary")
    ref = kde_reference_positions(engineer, X_tr)

    # Each arm's labels are excluded on that arm's own inner plan (one pass per distinct plan).
    plans: Dict[bytes, Dict[str, Any]] = {}

    def plan_audit(inner: np.ndarray) -> Dict[str, Any]:
        key = inner.tobytes()
        if key not in plans:
            groups = np.unique(inner)
            frames = fold_excluded_frames(engineer, X_tr, y, inner)
            alt_src = pd.concat([frames[int(g)][inner == g] for g in groups]).loc[X_tr.index]
            alt_bins = bin_source(alt_src)
            refit_src = inner_refit_features(X_tr, y, inner)
            variants = {"all": alt_bins}        # all three encoders together, and each alone
            for feat in LABEL_FITTED_FEATURES:
                one = base_src.copy()
                one[feat] = alt_src.loc[base_src.index, feat]
                variants[f"{feat}_only"] = bin_source(one)
            plans[key] = {"variants": variants,
                          "frame_bins": {g: bin_source(f) for g, f in frames.items()},
                          "summary": {
                "inner_fold_rows": [int((inner == g).sum()) for g in groups],
                "kde_reference_rows_per_inner_fold": [int((inner[ref] == g).sum())
                                                      for g in groups],
                "label_excluded": compare_features(base_src, alt_src, base_bins, alt_bins, y,
                                                   inner),
                "inner_refit_context": compare_features(base_src, refit_src, base_bins,
                                                        bin_source(refit_src), y, inner),
            }}
        return plans[key]

    per_arm = {arm: plan_audit(inner) for arm, inner in arms_inner.items()}
    first = next(iter(per_arm.values()))
    same_inner = len(plans) == 1
    record: Dict[str, Any] = {
        "format": G13_FORMAT, "fold": fold, "method": "label_excluded (inner fold)",
        "fit_rows": len(y), "fit_positives": int(y.sum()),
        "fit_label_fingerprint": label_fingerprint(X_tr.index, y),
        "label_fitted_features": {k: {"encoder": v[0], "bins": v[1] + "*"}
                                  for k, v in LABEL_FITTED_FEATURES.items()},
        "kde_reference_rows": int(len(ref)),
        "inner_folds_shared_by_both_arms": bool(same_inner),
        **first["summary"],
        "eligibility": {},
        "eligibility_retrained": {},
        "uses_test_rows": False,
    }
    if not same_inner:
        record["per_arm_inner_plan"] = {arm: a["summary"] for arm, a in per_arm.items()}
    from .inner_features import frame_fingerprint, pathway_identity, plan_fingerprint
    record["pathways"], record["isolation"] = {}, {}
    fitted = {"glass": glass, "rf": rf_router}
    for arm in ("glass", "rf"):
        if arm not in arms_inner:
            continue
        obj, inner, frames = fitted[arm], arms_inner[arm], per_arm[arm]["frame_bins"]
        pw = E.to_json_safe(getattr(obj, "inner_feature_pathway_", None)
                            or {"pathway": PATHWAY_OUTER, "record": None})
        record["pathways"][arm] = pw
        if pw["pathway"] == PATHWAY_OUTER:
            direct = (glass_eligibility_shift if arm == "glass" else rf_eligibility_shift)(
                obj, base_bins, per_arm[arm]["variants"], y)
            retrained = (glass_retrained_shift if arm == "glass" else rf_retrained_shift)(
                obj, base_bins, frames, y)
        else:
            # The corrected pathway: the fit's inner bins must be the audit's own
            # label-excluded recomputation, and its inner fits must reproduce from them.
            match = bool(pw.get("record") == pathway_identity(PATHWAY_FOLD)["record"]
                         and pw.get("plan_fingerprint") == plan_fingerprint(inner)
                         and [f["bins_fingerprint"] for f in pw.get("inner_folds", [])]
                         == [frame_fingerprint(frames[j]) for j in sorted(frames)])
            direct = {"not_applicable": "corrected pathway: each inner fold is already routed "
                                        "with its label-excluded bins (eligibility_retrained)"}
            retrained = ((glass_correction_shift if arm == "glass" else rf_correction_shift)(
                obj, base_bins, frames, y) if match else
                {"complete": False, "reproduced_all": False, "rows_changed": None,
                 "share_of_remainder": None, "not_run": "pathway mismatch"})
            record["isolation"][arm] = {"pathway_match": match,
                                        "reproduced_all": retrained["reproduced_all"],
                                        "complete": retrained["complete"]}
        record["eligibility"][arm] = direct
        record["eligibility_retrained"][arm] = retrained
    record["materiality_inputs"] = {
        **{f"{arm}_rows_any_bin_changed": a["summary"]["label_excluded"]["rows_any_bin_changed"]
           for arm, a in per_arm.items()},
        **{f"{arm}_eligibility_rows_changed": v["all"]["rows_changed"]
           for arm, v in record["eligibility"].items() if "all" in v},
        **{f"{arm}_eligibility_share_of_remainder": v["all"]["share_of_remainder"]
           for arm, v in record["eligibility"].items() if "all" in v},
        **{f"{arm}_eligibility_rows_changed_by_encoder": {
            feat: v[f"{feat}_only"]["rows_changed"] for feat in LABEL_FITTED_FEATURES}
           for arm, v in record["eligibility"].items() if "all" in v},
        **{f"{arm}_retrained_eligibility_rows_changed": v["rows_changed"]
           for arm, v in record["eligibility_retrained"].items()},
        **{f"{arm}_retrained_eligibility_share_of_remainder": v["share_of_remainder"]
           for arm, v in record["eligibility_retrained"].items()},
        **{f"{arm}_retrained_complete": v["complete"]
           for arm, v in record["eligibility_retrained"].items()},
        **{f"{arm}_feature_pathway": pw["pathway"] for arm, pw in record["pathways"].items()},
        "decision": "owner: material / not material, documented before any primary result",
    }
    return record
