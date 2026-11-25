"""
feature_research.encoding_audit
===============================
PR 42 Commit 3, Step 3A: how the accepted target-derived encodings are built,
and, for a row being encoded, which rows and labels reach its value and which
of those labels could have been known when it was contacted. Training side
only.

Three mechanisms are kept apart throughout:

(a) ``self``       a fit row's own label in its own (training) representation;
(b) ``held_out``   a held-out row's label in its evaluation representation;
(c) ``look_ahead`` labels of later contacts in the encoding of an earlier row.

Sections
--------
1. Time index (``verify_source_order``, ``build_time_index``)
   Each training row's index label is its line in the source CSV, verified
   against the CSV itself. Periods (year, month) and contact days are derived
   from that order; ``RESOLUTION`` states what the index can and cannot order.
2. Construction (``stage1_construction``, ``CONSTRUCTION``)
   How each encoding is fitted, with what fallbacks, and how it is applied.
3. Fitting paths (``fit_paths``)
   Every path that fits an encoding: the final fits (which also map the test
   split), Stage 1 tuning / OOF / inner calibration folds, Stage 2 outer folds
   and label-excluded inner folds, Stage 3 tuning, reporting and inner folds.
4. Attribution (``audit_paths``)
   Every encoded value written as a combination of fit-row labels (``Block``),
   reconstructed and compared with the production output, and its label mass
   split by when the contributing contacts happened relative to the encoded row.
5. Isolation checks (``isolation_checks``)
   Label flips through the production fold builders, which fail if a held-out
   label reaches its own encoding, plus the code-level label arguments.
6. Provenance (``provenance_table``)
   Constants and frozen thresholds, separately from the labels a row's
   transformation uses.

Nothing here reads a test row, a test label or a model artifact. The source
check reads the training rows' lines of the CSV; every other line is skipped.
"""

from __future__ import annotations

import ast
import calendar
import contextlib
import csv
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["RELATIONS", "LATER", "RESOLUTION", "SOURCE_ORDER", "CONSTRUCTION", "MECHANISMS",
           "verify_source_order", "build_time_index", "time_index_checks", "period_table",
           "stage1_construction", "FitPath", "fit_paths", "PATH_USE", "path_table",
           "plan_checks", "router_plan_reference", "Block", "attribute",
           "audit_paths", "path_summary", "mapping_by_period", "isolation_checks",
           "provenance_table", "findings_table", "family_findings", "compact_fits",
           "QUESTION", "DESCENDANTS", "accepted_fit_scopes", "mlp_early_stopping",
           "MLP_EARLY_STOPPING_ORDER", "MLP_ISOLATED_SPLIT", "MLP_OUTER_ISOLATION", "TOLERANCE", "COMPARISON_POLICY",
           "expected_comparisons", "comparison_coverage", "held_out_own_labels", "require"]

BANK_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# ----------------------------------------------------------------------
# 1. time index
# ----------------------------------------------------------------------
#: Where each contributing contact sits relative to the encoded row.
RELATIONS = ("earlier_day", "same_day", "later_day_same_month", "later_month")
#: The relations that are demonstrably later contacts.
LATER = ("later_day_same_month", "later_month")

SOURCE_ORDER = {
    "dataset": "UCI Bank Marketing, data/raw/bank-additional-full.csv (41,188 contacts)",
    "documented_order": ("rows ordered by date, May 2008 to November 2010 (Moro, Cortez and "
                         "Rita 2014; UCI dataset description); the file has no year and no "
                         "day of month"),
    "first_period": "2008-05",
    "last_period": "2010-11",
}

RESOLUTION = {
    "unit": ("contact day: a maximal run of training rows, in source order, sharing year, "
             "month, day_of_week and euribor3m (a daily rate)"),
    "year": "inferred: 2008 at the first row, plus one wherever the month decreases in source order",
    "between_periods": "calendar order of (year, month); relation later_month",
    "between_days_of_a_period": ("source order, i.e. the documented date order (weekdays "
                                 "advance within each week); relation later_day_same_month"),
    "within_a_day": ("unresolved: no time of day is recorded, so contacts of the same day are "
                     "treated as simultaneous; relation same_day"),
    "outcome_timing": ("unknown: the label records whether the client subscribed, not when "
                       "that became known. An earlier-day label is at most plausibly known at "
                       "a later contact; a same-day label is unresolved; a later-day label "
                       "was not yet known"),
    "training_rows_only": ("the index is built from training rows: a day whose contacts all "
                           "fell in the test split is absent, which can at most merge two days "
                           "with the same key into one (classified same_day, never later)"),
}

_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_WEEKDAYS = {d: i for i, d in enumerate("mon tue wed thu fri".split())}
_INT_COLUMNS = ("age", "campaign", "previous")
_FLOAT_COLUMNS = ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
_MONTHLY = ("cons.price.idx", "cons.conf.idx", "emp.var.rate", "nr.employed")


def _check(name: str, passed: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def require(checks: pd.DataFrame, what: str) -> str:
    """Raise if any check failed; otherwise a one-line summary."""
    bad = checks[~checks["passed"].astype(bool)]
    if len(bad):
        raise AssertionError(f"{what}: {len(bad)} check(s) failed\n{bad.to_string()}")
    return f"{what}: {len(checks)} checks passed"


def verify_source_order(X: pd.DataFrame, y: pd.Series, *, csv_path=None, db_path=None) -> pd.DataFrame:
    """
    Verify that each training row's index label is its 0-based data line in
    the source CSV, through the database load, preprocessing and the split.

    The CSV is read as ``build_database`` reads it (``csv.reader``, ``;``,
    blank lines skipped) and only the lines at training positions are kept;
    the database contributes its row count and rowid range only.
    """
    from data.bank_database import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, RAW_TABLE

    csv_path, db_path = Path(csv_path or DEFAULT_CSV_PATH), Path(db_path or DEFAULT_DB_PATH)
    labels = np.asarray(X.index)
    keep = {int(v) for v in labels}
    lines: Dict[int, List[str]] = {}
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            if n in keep:
                lines[n] = row
            n += 1
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        count, low, high = conn.execute(
            f"SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM {RAW_TABLE}").fetchone()
    out = [_check("the CSV and raw_bank hold the same number of rows; raw_bank rowids run "
                  "1..n (insertion order = CSV order; transform.sql reads ORDER BY rowid)",
                  n == count and low == 1 and high == count,
                  f"CSV {n:,} data rows; raw_bank {count:,} rows, rowid {low}..{high}")]
    ok_labels = (np.issubdtype(labels.dtype, np.integer) and len(set(labels)) == len(labels)
                 and labels.min() >= 0 and labels.max() < n)
    out.append(_check("training index labels are distinct source positions 0..n-1", ok_labels,
                      f"{len(labels):,} labels in [{labels.min()}, {labels.max()}]"))
    if not ok_labels or len(lines) != len(labels):
        return pd.DataFrame(out)
    src = pd.DataFrame.from_dict(lines, orient="index", columns=header).loc[labels]
    same = {c: np.array_equal(src[c].astype(int).to_numpy(), X[c].to_numpy()) for c in _INT_COLUMNS}
    same.update({c: np.array_equal(src[c].astype(float).to_numpy(), X[c].to_numpy(dtype=float))
                 for c in _FLOAT_COLUMNS})
    same["month"] = np.array_equal(src["month"].map(_MONTHS).to_numpy(), X["month"].to_numpy())
    same["day_of_week"] = np.array_equal(src["day_of_week"].map(_WEEKDAYS).to_numpy(),
                                         X["day_of_week"].to_numpy())
    out.append(_check("each training row equals its source line (the columns the time index and "
                      "the encodings read, and the other pass-through numerics)", all(same.values()),
                      ", ".join(f"{c}={'ok' if v else 'DIFFERS'}" for c, v in same.items())))
    out.append(_check("each training label equals its source line's y",
                      np.array_equal((src["y"] == "yes").astype(int).to_numpy(),
                                     np.asarray(y).astype(int)), f"{int(np.sum(y)):,} positives"))
    coded = [c for c in X.columns if c in src.columns and c not in same]
    pairs = {c: pd.DataFrame({"s": src[c].to_numpy(), "p": X[c].to_numpy()}).drop_duplicates()
             for c in coded}
    one_to_one = {c: d["s"].is_unique and d["p"].is_unique for c, d in pairs.items()}
    out.append(_check("every other column's source value and code correspond one to one across "
                      "the training rows (a misaligned row would break the correspondence)",
                      all(one_to_one.values()),
                      ", ".join(f"{c}={len(pairs[c])} codes" for c in coded)))
    twins = int(X.assign(_y=np.asarray(y)).duplicated(keep=False).sum())
    out.append(_check("rows identical in every column and label are interchangeable: a swap "
                      "among them changes no time key and no label", True,
                      f"{twins:,} training rows have an identical twin"))
    return pd.DataFrame(out)


def build_time_index(X: pd.DataFrame) -> pd.DataFrame:
    """
    The PR 42 time index of the training rows, in ``X``'s row order:
    ``position`` (source line), ``year``, ``month``, ``period`` ('YYYY-MM'),
    ``period_index``, ``day`` (contact day, numbered in source order) and
    ``next_period_day`` (the first day of the following period).
    """
    pos = np.asarray(X.index).astype(np.int64)
    order = np.argsort(pos, kind="stable")
    s = X.iloc[order]
    month = s["month"].to_numpy().astype(int)
    year = int(SOURCE_ORDER["first_period"][:4]) + np.r_[0, np.cumsum(np.diff(month) < 0)]
    dow = s["day_of_week"].to_numpy().astype(int)
    key = np.c_[year, month, dow, s["euribor3m"].to_numpy(dtype=float)]
    day = np.r_[0, np.cumsum((key[1:] != key[:-1]).any(axis=1))].astype(np.int64)
    code = year * 12 + month
    _, period_index = np.unique(code, return_inverse=True)
    first_day = pd.Series(day).groupby(period_index).min().to_numpy()
    next_first = np.r_[first_day[1:], day.max() + 1]
    T = pd.DataFrame({
        "position": pos[order], "year": year, "month": month,
        "period": [f"{a}-{b:02d}" for a, b in zip(year, month)],
        "period_index": period_index.astype(np.int64), "day": day,
        "next_period_day": next_first[period_index].astype(np.int64), "day_of_week": dow,
    }, index=X.index[order])
    return T.loc[X.index]


def time_index_checks(X: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Plausibility checks of the time index against the calendar and the indicators."""
    s = T.sort_values("position")
    Xs = X.loc[s.index]
    runs = int((s["period"] != s["period"].shift()).sum())
    periods = s["period"].unique()
    out = [_check("each period is one contiguous run in source order", runs == len(periods),
                  f"{runs} runs, {len(periods)} periods"),
           _check("the periods span the documented range",
                  periods[0] == SOURCE_ORDER["first_period"] and periods[-1] == SOURCE_ORDER["last_period"],
                  f"{periods[0]} .. {periods[-1]}")]
    per = Xs.groupby(s["period"].to_numpy())
    nunique = per[list(_MONTHLY)].nunique()
    out.append(_check("each period has one value of each monthly or quarterly indicator",
                      bool((nunique == 1).all().all()), f"{', '.join(_MONTHLY)}"))
    cpi = per["cons.price.idx"].first()
    out.append(_check("the consumer price index differs between every two periods (so periods "
                      "with the same month are different years)", cpi.nunique() == len(cpi),
                      f"{cpi.nunique()} distinct values over {len(cpi)} periods"))
    days = s.drop_duplicates("day")
    over, wraps = [], []
    for (yy, mm), sub in days.groupby(["year", "month"]):
        weekdays = sum(1 for d in range(1, calendar.monthrange(int(yy), int(mm))[1] + 1)
                       if calendar.weekday(int(yy), int(mm), d) < 5)
        if len(sub) > weekdays:
            over.append(f"{yy}-{mm:02d}: {len(sub)} > {weekdays}")
        w = int((np.diff(sub["day_of_week"].to_numpy()) <= 0).sum())
        if w > 4:
            wraps.append(f"{yy}-{mm:02d}: {w}")
    out.append(_check("no period has more contact days than its calendar month has weekdays",
                      not over, over or f"{len(days)} contact days"))
    out.append(_check("within each period the weekday advances except at most 4 week changes "
                      "(a month spans at most 5 weekday weeks)", not wraps, wraps or "all periods"))
    return pd.DataFrame(out)


def period_table(T: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rows, positives, subscription rate and contact days per period (training rows)."""
    df = T.assign(y=np.asarray(y).astype(int))
    g = df.groupby(["period_index", "period"])
    out = pd.DataFrame({"rows": g.size(), "positives": g["y"].sum(), "days": g["day"].nunique(),
                        "first_position": g["position"].min(), "last_position": g["position"].max()})
    out["rate"] = out["positives"] / out["rows"]
    return out.reset_index()


def recurring_day_keys(T: pd.DataFrame, X: pd.DataFrame) -> int:
    """Day keys (year, month, weekday, euribor3m) that recur in separate runs (later weeks)."""
    s = T.sort_values("position")
    keys = pd.DataFrame({"y": s["year"], "m": s["month"], "d": s["day_of_week"],
                         "e": X.loc[s.index, "euribor3m"], "day": s["day"]}).drop_duplicates("day")
    return int((keys.groupby(["y", "m", "d", "e"]).size() > 1).sum())


# ----------------------------------------------------------------------
# 2. construction (code trace)
# ----------------------------------------------------------------------
CONSTRUCTION = {
    "subscribe_rate_by_euribor_bin": {
        "implementation": "shared.stage1.feature_engineering.LRFeatureEngineer",
        "fit": ("pd.qcut(euribor3m of the fit rows, q=20, duplicates='drop') (equal-width pd.cut "
                "if qcut fails); rate = mean label of the fit rows in each bin "
                "(groupby observed=True); the IntervalIndex categories are stored"),
        "apply": ("pd.cut(euribor3m, the stored categories, include_lowest=True) -> that bin's "
                  "rate. The categories carry rounded edges, so a row near an edge can map to a "
                  "neighbouring bin of the one its own fit assigned it"),
        "fallback": ("outside the stored categories (or an empty bin): the median of the bin "
                     "rates, i.e. the mean label of the middle bin, or the average of the two "
                     "middle bins' rates"),
        "smoothing": "none",
        "fit_rows_in_their_own_representation": "yes: a fit row's label is in its bin's rate (1/n_bin)",
    },
    "subscribe_rate_by_dow_month": {
        "implementation": "shared.stage1.feature_engineering.LRFeatureEngineer",
        "fit": ("global = mean label of the fit rows; per cell day_of_week x month (string key, "
                "years pooled): (n_cell * mean_cell + 100 * global) / (n_cell + 100)"),
        "apply": "cell key -> smoothed rate",
        "fallback": "a cell absent from the fit rows: the global rate",
        "smoothing": "k = 100 toward the global rate of the fit rows (every fit row's label)",
        "fit_rows_in_their_own_representation": ("yes: 1/(n_cell + 100) through the cell plus "
                                                  "100/((n_cell + 100) N) through the global rate"),
    },
    "neighborhood_subscription_density": {
        "implementation": ("shared.stage2.feature_engineering._IntegralFeatureEngineer (Stage 2); "
                           "feature_research.feature_engineering.integrals.IntegralFeatureEngineer "
                           "(Stage 3)"),
        "fit": ("StandardScaler on (euribor3m, nr.employed, emp.var.rate) of ALL fit rows; "
                "reference = 5,000 fit rows drawn with default_rng(42) when the fit has more than "
                "10,000 rows (every production fit does), else every fit row"),
        "apply": ("Gaussian kernel (bandwidth 1.0, scaled space) over the reference rows: "
                  "sum_j K_j y_j / (sum_j K_j + 1e-10)"),
        "fallback": "none (every row gets a kernel average)",
        "smoothing": "the kernel; only reference rows' labels contribute",
        "fit_rows_in_their_own_representation": ("only a reference row (5,000 of the fit rows), "
                                                 "with weight K(0) / sum_j K_j"),
    },
    "economic_curvature_intensity": {
        "implementation": ("shared.stage2.feature_engineering._DerivativeFeatureEngineer; "
                           "feature_research.feature_engineering.derivatives.DerivativeFeatureEngineer"),
        "fit": ("for each of euribor3m, nr.employed, emp.var.rate: qcut(20, duplicates='drop') "
                "of the fit rows, rate per non-empty bin, |np.gradient(np.gradient(rates))| over "
                "the non-empty bins (zeros if fewer than 3)"),
        "apply": ("row's bin -> its |curvature| (Stage 2: rounded IntervalIndex categories; "
                  "Stage 3: exact numeric edges); ECI = mean of the three"),
        "fallback": "outside the bins or an empty bin: 0.0",
        "smoothing": "none; each curvature combines the rates of up to five neighbouring bins",
        "fit_rows_in_their_own_representation": "yes, through its bin's rate and its neighbours'",
    },
    "dow_month_encoded": {
        "implementation": ("shared.stage2.feature_engineering._TemporalFeatureEngineer (integer "
                           "keys); feature_research.feature_engineering.temporal.TemporalFeatureEngineer "
                           "(string keys)"),
        "fit": "as subscribe_rate_by_dow_month: k = 100 toward the fit rows' global rate",
        "apply": "cell -> smoothed rate", "fallback": "unseen cell: the global rate",
        "smoothing": "k = 100", "fit_rows_in_their_own_representation": "yes, as Stage 1",
    },
}

#: Descendants that inherit an audited mechanism (label-free functions of it).
DESCENDANTS = {
    "neighborhood_subscription_density": ["nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot",
                                          "decay_x_density"],
    "economic_curvature_intensity": ["eci_cold", "eci_mid", "eci_warm", "eci_hot"],
    "dow_month_encoded": ["dow_cold", "dow_low", "dow_mid", "dow_hot"],
}

QUESTION = ("For a row being encoded, exactly which rows and labels can influence its feature "
            "value, and which of those labels could have been known at the time of contact?")

MECHANISMS = {
    "self": "(a) a fit row's own label in its own training representation",
    "held_out": "(b) a held-out row's label in its evaluation representation",
    "look_ahead": "(c) labels of later contacts in the encoding of an earlier row",
}


def _quiet(fn: Callable, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _qcut_codes(values, n_bins: int, categories=None, edges=None) -> np.ndarray:
    """
    Each fit row's bin exactly as the production fit assigned it (the label-free
    quantile cut of the fit rows), checked against the stored categories / edges.
    """
    v = pd.Series(np.asarray(values, dtype=float))
    try:
        bins, found = pd.qcut(v, q=n_bins, duplicates="drop", retbins=True)
    except ValueError:
        bins, found = pd.cut(v, bins=n_bins, duplicates="drop", retbins=True)
    if categories is not None and not bins.cat.categories.equals(categories):
        raise AssertionError("the fit rows' quantile bins are not the stored categories")
    if edges is not None and not np.array_equal(found, np.asarray(edges)):
        raise AssertionError("the fit rows' quantile edges are not the stored edges")
    return bins.cat.codes.to_numpy().astype(np.int64)


def stage1_construction(X: pd.DataFrame, y: pd.Series, T: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    The full-train Stage 1 encodings on the training rows: each euribor bin's
    rows, rate and the periods it spans, and each day x month cell's rows and
    the years it pools.
    """
    from shared.stage1.feature_engineering import LRFeatureEngineer

    eng = _quiet(LRFeatureEngineer().fit, X, y)
    cats = eng._euribor_intervals
    code = _qcut_codes(X["euribor3m"], eng.n_bins, categories=cats)
    yv = np.asarray(y).astype(int)
    rows = []
    for c, interval in enumerate(cats):
        m = code == c
        per = T["period"].to_numpy()[m]
        vc = pd.Series(per).value_counts()
        rows.append({"bin": c, "interval": str(interval), "rows": int(m.sum()),
                     "positives": int(yv[m].sum()), "rate": float(yv[m].mean()),
                     "periods": int(vc.size), "first_period": str(min(per)),
                     "last_period": str(max(per)), "modal_period": str(vc.index[0]),
                     "modal_period_share": float(vc.iloc[0] / m.sum())})
    bins = pd.DataFrame(rows)
    key = X["day_of_week"].astype(int).to_numpy() * 100 + X["month"].astype(int).to_numpy()
    cells = []
    for k in np.unique(key):
        m = key == k
        n = int(m.sum())
        cells.append({"day_of_week": int(k // 100), "month": int(k % 100), "rows": n,
                      "positives": int(yv[m].sum()), "rate": float(yv[m].mean()),
                      "years": ",".join(map(str, sorted(set(T["year"].to_numpy()[m])))),
                      "n_years": len(set(T["year"].to_numpy()[m])),
                      "smoothed": float((yv[m].sum() + eng.smoothing_factor * eng._global_mean)
                                        / (n + eng.smoothing_factor)),
                      "cell_weight": n / (n + eng.smoothing_factor)})
    return {"bins": bins, "cells": pd.DataFrame(cells),
            "settings": pd.DataFrame([{"n_bins_requested": eng.n_bins, "n_bins_fitted": len(cats),
                                       "smoothing_factor": eng.smoothing_factor,
                                       "global_rate": eng._global_mean}])}


# ----------------------------------------------------------------------
# 3. fitting paths
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FitPath:
    """
    One fit of an encoding: ``fit`` are the training positions whose labels it
    may use (after any exclusion), ``structure`` the positions whose covariates
    fix its label-free structure (scaler, reference sample, quantile edges)
    when they differ from ``fit``, and ``query`` the rows it encodes for that
    path (held-out rows, or the fit rows themselves when ``in_sample``).
    """

    stage: str
    path: str
    fold: Optional[int]
    inner: Optional[int]
    fit: np.ndarray
    query: np.ndarray
    in_sample: bool = False
    structure: Optional[np.ndarray] = None

    @property
    def key(self) -> str:
        parts = [self.stage, self.path]
        parts += [f"k{self.fold}"] if self.fold is not None else []
        parts += [f"j{self.inner}"] if self.inner is not None else []
        return ".".join(parts)


def _stage1_plan(y, rows=None) -> List[Tuple[np.ndarray, np.ndarray]]:
    from shared.stage_io import fold_assignment
    yy = np.asarray(y) if rows is None else np.asarray(y)[rows]
    fid = fold_assignment(yy, cv_folds=10, random_state=SEED).to_numpy()
    base = np.arange(len(yy)) if rows is None else np.asarray(rows)
    return [(base[fid != k], base[fid == k]) for k in range(10)]


def _router_inner_plan(y_fit) -> np.ndarray:
    """Both routers' inner fold of each outer-training row (``make_folds``, 5, seed 42)."""
    from blackbox_pipeline.models.rf_router.oof.folds import make_folds as rf_make_folds
    plan = np.full(len(y_fit), -1, dtype=int)
    for j, (_, va) in enumerate(rf_make_folds(pd.Series(np.asarray(y_fit)), 5, SEED)):
        plan[va] = j
    return plan


def fit_paths(y: pd.Series) -> Dict[str, List[FitPath]]:
    """
    Every fitting path of the audited encodings, by stage, as training positions.

    Stage 1 (both arms, ``shared.stage1.crossfit``): the final fit (in-sample;
    it also maps the test split), 10 outer folds (``fold_assignment(y, 10, 42)``,
    the OOF probabilities and, with tune_cv_folds = cv_folds = 10, the tuning
    folds of both arms) and 10 x 10 inner calibration folds. Stage 2 (both
    routers): the final fit, 5 outer folds (``make_folds(y, 5, 42)``) and 5 x 5
    label-excluded inner folds over each outer fold's training rows, plus the
    final fit's own 5 label-excluded inner folds over all training rows. Stage 3
    (both arms, ``Stage3Block``): the final fit, 5 tuning, 10 reporting and
    5 x 5 inner folds (``FoldPlan``).
    """
    from glass_pipeline.glass_router.pipeline import make_folds
    from shared.stage3.data import FoldPlan

    n = len(y)
    allrows = np.arange(n)
    yv = np.asarray(y).astype(int)
    s1 = [FitPath("stage1", "final", None, None, allrows, allrows, in_sample=True)]
    for k, (tr, va) in enumerate(_stage1_plan(yv)):
        s1.append(FitPath("stage1", "outer", k, None, tr, va))
        for j, (itr, iva) in enumerate(_stage1_plan(yv, tr)):
            s1.append(FitPath("stage1", "inner", k, j, itr, iva))
    s2 = [FitPath("stage2", "final", None, None, allrows, allrows, in_sample=True)]
    final_plan = _router_inner_plan(yv)
    for j in range(5):
        s2.append(FitPath("stage2", "final_inner_label_excluded", None, j, allrows[final_plan != j],
                          allrows[final_plan == j], structure=allrows))
    for k, (tr, va) in enumerate(make_folds(pd.Series(yv), 5, SEED)):
        s2.append(FitPath("stage2", "outer", k, None, tr, va))
        plan = _router_inner_plan(yv[tr])
        for j in range(5):
            s2.append(FitPath("stage2", "inner_label_excluded", k, j, tr[plan != j], tr[plan == j],
                              structure=tr))
    s3 = [FitPath("stage3", "final", None, None, allrows, allrows, in_sample=True)]
    for name, splits in (("tuning", 5), ("reporting", 10)):
        for k, (tr, va) in enumerate(FoldPlan.build(yv, splits, SEED).splits()):
            s3.append(FitPath("stage3", name, k, None, np.asarray(tr), np.asarray(va)))
            if name == "tuning":
                for j, (itr, iva) in enumerate(FoldPlan.build(yv[tr], 5, SEED).splits()):
                    s3.append(FitPath("stage3", "inner", k, j, np.asarray(tr)[itr],
                                      np.asarray(tr)[iva]))
    return {"stage1": s1, "stage2": s2, "stage3": s3}


#: What each fitting path is used for, by whom, and the production call that fits it.
PATH_USE = (
    ("stage1", "final", "GLASS LR and black-box MLP",
     "final base model and its scaler; maps the test split (the final calibrator is fitted on "
     "the OOF base scores, not on these features)",
     "shared.stage1.feature_engineering.engineer_features(X_train, X_test, y_train)"),
    ("stage1", "outer", "both arms, shared code",
     "the 10 Optuna tuning folds (tune_cv_folds = cv_folds = 10, the same designs) and the 10 "
     "outer folds of the OOF base scores and nested calibration (OOF threshold, Stage 4 train column)",
     "shared.stage1.crossfit.build_fold_designs (audits F3/F4)"),
    ("stage1", "inner", "both arms, shared code",
     "10 x 10 inner calibration folds: per-outer-fold calibrator family and fit",
     "shared.stage1.crossfit.inner_fold_designs via oof_calibration.calibrate_oof (F19)"),
    ("stage2", "final", "GLASS Router and RF Router",
     "full-train routers; maps the test split",
     "stage2_crossfit.runner.fit_full_train -> shared.stage2.engineer_features"),
    ("stage2", "final_inner_label_excluded", "both routers",
     "the full-train routers' own Pass 1 OOF / eligibility folds (5, over all training rows)",
     "stage2_crossfit.inner_features.FoldLabelExcludedFeatures (fold_label_excluded)"),
    ("stage2", "outer", "both routers",
     "5-fold outer cross-fit: the Stage 4 train-side Stage 2 outputs",
     "crossfit_stage2 / crossfit_rf_router -> engineer_features(X_tr, X_va, y_tr)"),
    ("stage2", "inner_label_excluded", "both routers",
     "each outer fit's Pass 1 OOF / eligibility folds (5 x 5)",
     "FoldLabelExcludedFeatures -> stage2_crossfit.g13.label_excluded_engineer"),
    ("stage3", "final", "GLASS EBM and black-box XGBoost",
     "refit model; maps the test split (calibrator and thresholds are fitted on OOF scores)",
     "shared.stage3.data.Stage3Block.build: EBMFeaturePipeline on all training rows"),
    ("stage3", "tuning", "both arms",
     "5 tuning folds: tuning, OOF scores, calibration gate, thresholds, Stage 4 train side",
     "Stage3Block fold_frames(tune_plan), feature_fit_scope 'per_fold'"),
    ("stage3", "reporting", "both arms", "10 reporting folds", "Stage3Block fold_frames(report_plan)"),
    ("stage3", "inner", "both arms", "5 x 5 inner folds: per-tuning-fold calibration and thresholds (F22)",
     "Stage3Block inner_folds"),
)


def path_table(paths: Mapping[str, List[FitPath]]) -> pd.DataFrame:
    """One row per fitting path: use, production call, fits and their row counts."""
    rows = []
    for stage, path, arms, use, call in PATH_USE:
        group = [p for p in paths[stage] if p.path == path]
        label = [len(p.fit) for p in group]
        struct = [len(p.structure) if p.structure is not None else len(p.fit) for p in group]
        held = [len(p.query) for p in group if not p.in_sample]
        rows.append({"stage": stage, "path": path, "arms": arms, "used_for": use,
                     "production_call": call, "fits": len(group),
                     "label_rows": f"{min(label):,}-{max(label):,}" if min(label) != max(label)
                     else f"{label[0]:,}",
                     "structure_rows": f"{min(struct):,}-{max(struct):,}" if min(struct) != max(struct)
                     else f"{struct[0]:,}",
                     "held_out_rows": (f"{min(held):,}-{max(held):,}" if held and min(held) != max(held)
                                       else f"{held[0]:,}" if held else "none (in-sample)")})
    return pd.DataFrame(rows)


def plan_checks(y: pd.Series, paths: Mapping[str, List[FitPath]]) -> pd.DataFrame:
    """Coincidences between plans that let paths share evidence, and every path's partition."""
    from glass_pipeline.glass_router.contract import GLASS_F17_PRIMARY
    from blackbox_pipeline.models.rf_router.contract import RF_F17_PRIMARY

    def held(stage, path):
        return [p.query for p in paths[stage] if p.path == path]

    s2o, s3t = held("stage2", "outer"), held("stage3", "tuning")
    rf = dict(RF_F17_PRIMARY)
    out = [
        _check("Stage 3 tuning folds = Stage 2 outer folds (same rows held out)",
               len(s2o) == len(s3t) and all(np.array_equal(np.sort(a), np.sort(b))
                                            for a, b in zip(s2o, s3t)), "5 folds"),
        _check("the final Stage 2 fit's inner folds hold out the outer folds' rows (both are "
               "5 stratified folds of the training split, seed 42)",
               all(np.array_equal(np.sort(a.query), np.sort(b))
                   for a, b in zip([p for p in paths["stage2"] if p.path == "final_inner_label_excluded"],
                                   s2o)), "5 folds"),
        _check("both routers' inner folds: 5 stratified folds, seed 42 (GLASS pass1_oof_folds, "
               "RF n_oof_folds)",
               GLASS_F17_PRIMARY["pass1_oof_folds"] == rf["n_oof_folds"] == 5
               and GLASS_F17_PRIMARY["random_state"] == rf["random_state"] == SEED,
               f"GLASS {GLASS_F17_PRIMARY['pass1_oof_folds']}/{GLASS_F17_PRIMARY['random_state']}, "
               f"RF {rf['n_oof_folds']}/{rf['random_state']}"),
    ]
    n = len(y)
    for stage, ps in paths.items():
        outer = {p.fold: p for p in ps if p.path in ("outer", "tuning")}
        outer[None] = [p for p in ps if p.path == "final"][0]
        for name in sorted({p.path for p in ps} - {"final"}):
            group = [p for p in ps if p.path == name]
            disjoint = all(not np.intersect1d(p.fit, p.query).size for p in group)
            if "inner" in name:
                folds = {p.fold for p in group}
                cover = all(
                    np.array_equal(np.sort(np.concatenate([p.query for p in group if p.fold == k])),
                                   np.sort(outer[k].fit))
                    and (k is None or all(not np.intersect1d(np.r_[p.fit, p.query],
                                                             outer[k].query).size
                                          for p in group if p.fold == k))
                    for k in folds)
                what = "partition each outer fold's training rows and never touch its held-out rows"
            else:
                cover = (np.array_equal(np.sort(np.concatenate([p.query for p in group])), np.arange(n))
                         and all(len(p.fit) + len(p.query) == n for p in group))
                what = "partition the training split"
            out.append(_check(f"{stage} {name}: fit and held-out rows are disjoint; the held-out "
                              f"rows {what}", disjoint and cover, f"{len(group)} fits"))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 4. attribution
# ----------------------------------------------------------------------
@dataclass
class Block:
    """
    One additive part of an encoding, as label mass on groups of fit rows.

    ``member[f]``     the group of fit row f (-1: its label is not used here)
    ``qcls[h]``       the class of query row h (rows of a class share weights)
    ``mass[c, g]``    weight of each member of group g in class c's value
    ``coef[c, g]``    signed coefficient on each member's label (default ``mass``)
    ``post``          'identity', or 'abs' (curvature) applied to this block's sum
    ``scale``         the block's factor in the encoding (1/3 for each curvature)
    ``kind``          'direct' (bin, cell, kernel, curvature) or 'prior' (the
                      smoothing toward the global rate, every fit row)
    """

    name: str
    kind: str
    member: np.ndarray
    qcls: np.ndarray
    mass: np.ndarray
    coef: Optional[np.ndarray] = None
    post: str = "identity"
    scale: float = 1.0


def attribute(blocks: Sequence[Block], y_fit: np.ndarray, fit_day: np.ndarray,
              query_day: np.ndarray, next_day: np.ndarray,
              query_fit_pos: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    The encoding of every query row, reconstructed from the blocks, and its
    label mass and contributing fit rows by ``RELATIONS`` and block kind.

    ``next_day[d]`` is the first day of the period after day d's.
    ``query_fit_pos[h]`` is query row h's position among the fit's source rows,
    or -1 if it is not one of them. Wherever the query row is a member of a
    block, its own label is reported as ``self`` rather than ``same_day``; a row
    outside the source, or inside it but excluded from the label statistics
    (member -1), has self weight 0. ``None`` turns the accounting off: the final
    mapping's proxy convention, where a training row stands for a new contact
    on its day and its own label counts as a same-day contact.
    """
    y_fit = np.asarray(y_fit, dtype=float)
    nq = len(query_day)
    value = np.zeros(nq)
    mass = {k: {r: np.zeros(nq) for r in RELATIONS} for k in ("direct", "prior")}
    rows = {k: {r: np.zeros(nq, dtype=np.int64) for r in RELATIONS} for k in ("direct", "prior")}
    self_mass = np.zeros(nq)
    self_rows = np.zeros(nq, dtype=np.int64)
    for b in blocks:
        n_groups = b.mass.shape[1]
        used = b.member >= 0
        ysum = np.bincount(b.member[used], weights=y_fit[used], minlength=n_groups)
        coef = b.mass if b.coef is None else b.coef
        lin = (coef @ ysum)[b.qcls]
        value += b.scale * (np.abs(lin) if b.post == "abs" else lin)
        pairs, inv = np.unique(np.c_[b.qcls, query_day], axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        pc, pdays = pairs[:, 0], pairs[:, 1]
        pnext = next_day[pdays]
        pm = {r: np.zeros(len(pairs)) for r in RELATIONS}
        pr = {r: np.zeros(len(pairs), dtype=np.int64) for r in RELATIONS}
        order = np.argsort(b.member[used], kind="stable")
        gm, gd = b.member[used][order], fit_day[used][order]
        bounds = np.searchsorted(gm, np.arange(n_groups + 1))
        for g in range(n_groups):
            w = b.mass[pc, g]
            if not w.any():
                continue
            days = np.sort(gd[bounds[g]:bounds[g + 1]])
            e = np.searchsorted(days, pdays, "left")
            s = np.searchsorted(days, pdays, "right")
            nx = np.searchsorted(days, pnext, "left")
            counts = (e, s - e, nx - s, len(days) - nx)
            for r, c in zip(RELATIONS, counts):
                pm[r] += w * c
                pr[r] += (w > 0) * c
        for r in RELATIONS:
            mass[b.kind][r] += b.scale * pm[r][inv]
            rows[b.kind][r] += pr[r][inv]
        if query_fit_pos is not None:
            own = np.where(query_fit_pos >= 0, b.member[np.clip(query_fit_pos, 0, None)], -1)
            has = own >= 0
            w_self = np.zeros(nq)
            w_self[has] = b.mass[b.qcls[has], own[has]]
            self_mass += b.scale * w_self
            mass[b.kind]["same_day"] -= b.scale * w_self
            rows[b.kind]["same_day"] -= (w_self > 0)
            self_rows += (w_self > 0)
    return {"value": value, "mass": mass, "rows": rows, "self": self_mass, "self_rows": self_rows}


# -- blocks per encoding ------------------------------------------------
def _bin_rate_block(categories, code_fit: np.ndarray, y_fit: np.ndarray, query_values) -> Block:
    """
    Stage 1 euribor bin rate: the mean label of the query's bin (by the stored,
    rounded categories); outside them or in an empty bin, the median of the bin
    rates, i.e. the middle bin's rate or the mean of the two middle bins' rates.
    """
    L = len(categories)
    n = np.bincount(code_fit, minlength=L).astype(float)
    observed = n > 0
    q = pd.cut(pd.Series(np.asarray(query_values, dtype=float)), bins=categories,
               include_lowest=True).cat.codes.to_numpy()
    qcls = np.where((q >= 0) & observed[np.clip(q, 0, L - 1)], q, L)
    mass = np.zeros((L + 1, L))
    obs = np.flatnonzero(observed)
    mass[obs, obs] = 1.0 / n[obs]
    rates = np.bincount(code_fit, weights=np.asarray(y_fit, dtype=float), minlength=L)[obs] / n[obs]
    srt = obs[np.argsort(rates, kind="stable")]
    m = len(srt)
    mid = srt[[m // 2]] if m % 2 else srt[m // 2 - 1:m // 2 + 1]
    mass[L, mid] = 1.0 / (len(mid) * n[mid])
    return Block("bin", "direct", code_fit, qcls, mass)


def _cell_blocks(fit_dow, fit_month, query_dow, query_month, k: float,
                 member_of: Optional[np.ndarray] = None, n_rows: Optional[int] = None) -> List[Block]:
    """
    Laplace day x month rate: the cell (direct) and the global rate (prior);
    an unseen cell takes the global rate. ``member_of`` maps the given fit rows
    to positions among ``n_rows`` (rows outside them are not members).
    """
    fk = np.asarray(fit_dow).astype(int) * 100 + np.asarray(fit_month).astype(int)
    qk = np.asarray(query_dow).astype(int) * 100 + np.asarray(query_month).astype(int)
    cells, member = np.unique(fk, return_inverse=True)
    member = np.asarray(member).ravel().astype(np.int64)
    C, N = len(cells), len(fk)
    n = np.bincount(member, minlength=C).astype(float)
    pos = np.searchsorted(cells, qk)
    seen = (pos < C) & (cells[np.clip(pos, 0, C - 1)] == qk)
    qcls = np.where(seen, pos, C)
    cell = np.zeros((C + 1, C))
    cell[np.arange(C), np.arange(C)] = 1.0 / (n + k)
    prior = np.zeros((C + 1, 1))
    prior[:C, 0] = k / ((n + k) * N)
    prior[C, 0] = 1.0 / N
    members = [member, np.zeros(N, dtype=np.int64)]
    if member_of is not None:
        full = []
        for mm in members:
            f = np.full(n_rows, -1, dtype=np.int64)
            f[member_of] = mm
            full.append(f)
        members = full
    return [Block("cell", "direct", members[0], qcls, cell),
            Block("prior", "prior", members[1], qcls, prior)]


def _kde_block(scaler, bandwidth: float, fit_coords: pd.DataFrame, ref_pos: np.ndarray,
               query_coords: pd.DataFrame) -> Block:
    """NSD: kernel-weighted mean of the reference rows' labels; rows grouped by coordinates."""
    from scipy.spatial.distance import cdist

    cols = list(fit_coords.columns)
    member = np.full(len(fit_coords), -1, dtype=np.int64)
    U, ginv = np.unique(fit_coords.to_numpy(dtype=float)[ref_pos], axis=0, return_inverse=True)
    member[ref_pos] = np.asarray(ginv).ravel()
    n = np.bincount(member[member >= 0], minlength=len(U)).astype(float)
    Q, qinv = np.unique(query_coords.to_numpy(dtype=float), axis=0, return_inverse=True)
    d = cdist(scaler.transform(pd.DataFrame(Q, columns=cols)),
              scaler.transform(pd.DataFrame(U, columns=cols)), metric="euclidean")
    K = np.exp(-(d ** 2) / (2 * bandwidth ** 2))
    mass = K / ((K @ n) + 1e-10)[:, None]
    return Block("kernel", "direct", member, np.asarray(qinv).ravel(), mass)


def _gradient2(L: int) -> np.ndarray:
    """The matrix of ``np.gradient(np.gradient(r))`` for a length-L vector."""
    I = np.eye(L)
    D = np.column_stack([np.gradient(I[:, i]) for i in range(L)])
    return D @ D


def _curvature_block(name: str, code_fit: np.ndarray, L: int, query_code: np.ndarray,
                     scale: float) -> Block:
    """
    |second difference| of the non-empty bins' rates (over the bins that hold
    member rows, in bin order; zeros when fewer than 3). A query outside the
    bins or in a bin without members gets 0 (no labels). ``code_fit`` is -1 for
    a row that is not a member (excluded label).
    """
    used = code_fit >= 0
    n = np.bincount(code_fit[used], minlength=L).astype(float)
    obs = np.flatnonzero(n > 0)
    G = len(obs)
    remap = np.full(L, -1, dtype=np.int64)
    remap[obs] = np.arange(G)
    member = np.where(used, remap[np.clip(code_fit, 0, L - 1)], -1)
    qc = np.where(query_code >= 0, remap[np.clip(query_code, 0, L - 1)], -1)
    qcls = np.where(qc >= 0, qc, G)
    coef = np.zeros((G + 1, G))
    if G >= 3:
        coef[:G, :] = _gradient2(G) / n[obs][None, :]
    return Block(name, "direct", member, qcls, np.abs(coef), coef=coef, post="abs", scale=scale)


# -- per-implementation adapters ---------------------------------------
_CURV_FEATS = ("euribor3m", "nr.employed", "emp.var.rate")
_KDE_COLS = ["euribor3m", "nr.employed", "emp.var.rate"]
STAGE1_FEATURES = ("subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month")
STAGE23_FEATURES = ("neighborhood_subscription_density", "economic_curvature_intensity",
                    "dow_month_encoded")


def _reference_positions(ie, X_fit: pd.DataFrame, threshold: int) -> np.ndarray:
    """The positions in the fit rows of the KDE reference points, verified against the fit."""
    n = len(X_fit)
    pos = (np.random.default_rng(ie.random_state).choice(n, size=min(ie.n_ref, n), replace=False)
           if n > threshold else np.arange(n))
    if not np.array_equal(ie._scaler.transform(X_fit[_KDE_COLS])[pos], ie._X_ref):
        raise AssertionError("the KDE reference points are not the fit rows the audit identifies")
    return pos


@dataclass
class _Fitted:
    """
    One production fit: blocks per (feature, role), production values per
    (feature, role) or None where not recomputed, and the label-used rows.
    Members index ``rows`` (positions in the training frame).
    """

    rows: np.ndarray
    blocks: Dict[Tuple[str, str], List[Block]]
    values: Dict[Tuple[str, str], Optional[np.ndarray]]
    notes: Dict[str, Any]


def _stage1_fit(X: pd.DataFrame, y: np.ndarray, path: FitPath) -> _Fitted:
    """``LRFeatureEngineer`` on the path's fit rows; blocks for the held-out and fit rows."""
    from shared.stage1.feature_engineering import LRFeatureEngineer

    Xf = X.iloc[path.fit]
    yf = pd.Series(y[path.fit], index=Xf.index)
    eng = _quiet(LRFeatureEngineer().fit, Xf, yf)
    code = _qcut_codes(Xf["euribor3m"], eng.n_bins, categories=eng._euribor_intervals)
    roles = {"fit_rows": path.fit} if path.in_sample else {"held_out": path.query, "fit_rows": path.fit}
    blocks, values, notes = {}, {}, {}
    for role, rows in roles.items():
        Xq = X.iloc[rows]
        prod = _quiet(eng.transform, Xq)
        b = _bin_rate_block(eng._euribor_intervals, code, yf.to_numpy(), Xq["euribor3m"])
        blocks[(STAGE1_FEATURES[0], role)] = [b]
        blocks[(STAGE1_FEATURES[1], role)] = _cell_blocks(
            Xf["day_of_week"], Xf["month"], Xq["day_of_week"], Xq["month"], eng.smoothing_factor)
        for f in STAGE1_FEATURES:
            values[(f, role)] = prod[f].to_numpy(float)
        notes[(STAGE1_FEATURES[0], role)] = {"fallback_rows": int((b.qcls == len(eng._euribor_intervals)).sum())}
        cb = blocks[(STAGE1_FEATURES[1], role)][0]
        notes[(STAGE1_FEATURES[1], role)] = {"fallback_rows": int((cb.qcls == cb.mass.shape[1]).sum())}
    return _Fitted(path.fit, blocks, values, notes)


def _stage2_blocks(engineer, Xs: pd.DataFrame, keep: np.ndarray, Xq: pd.DataFrame) -> Dict[str, List[Block]]:
    """
    Blocks of ``RFFeatureEngineer`` fitted on ``Xs`` with its label statistics over
    the rows ``keep`` (every row: the fit itself; otherwise ``label_excluded_engineer``:
    the same scaler, reference sample and quantile bins, excluded labels dropped).
    """
    from stage2_crossfit.g13 import kde_reference_positions

    ie, de, te = engineer._integral_eng, engineer._derivative_eng, engineer._temporal_eng
    ref = kde_reference_positions(engineer, Xs)
    out = {"neighborhood_subscription_density": [
        _kde_block(ie._scaler, ie.bandwidth, Xs[_KDE_COLS], ref[keep[ref]], Xq[_KDE_COLS])]}
    curv = []
    for feat in _CURV_FEATS:
        cats = de._curvature_store[feat][0]
        code = np.where(keep, _qcut_codes(Xs[feat], de.n_bins, categories=cats), -1)
        qcode = pd.cut(Xq[feat].astype(float), bins=cats, include_lowest=True).cat.codes.to_numpy()
        curv.append(_curvature_block(f"curvature[{feat}]", code, len(cats), qcode, 1 / 3))
    out["economic_curvature_intensity"] = curv
    kept = np.flatnonzero(keep)
    out["dow_month_encoded"] = _cell_blocks(
        Xs["day_of_week"].to_numpy()[kept], Xs["month"].to_numpy()[kept], Xq["day_of_week"],
        Xq["month"], te.smoothing_factor, member_of=kept, n_rows=len(Xs))
    return out


def _stage2_fit(X: pd.DataFrame, y: np.ndarray, path: FitPath, engineers: Dict[int, Any]) -> _Fitted:
    """
    ``RFFeatureEngineer`` on the fit rows (final, outer folds), or the outer
    fold's engineer with inner fold j's labels excluded (``label_excluded_engineer``,
    the ``fold_label_excluded`` pathway). Production values are recomputed for
    the held-out rows and for the final fit's rows; a fold's own fit rows are
    attributed with the same blocks (verified on its held-out rows).
    """
    from stage2_crossfit.g13 import label_excluded_engineer

    rows = path.structure if path.structure is not None else path.fit
    Xs = X.iloc[rows]
    if path.structure is None:
        eng = engineers[path.fold]
        keep = np.ones(len(rows), dtype=bool)
        encoder = eng
    else:
        eng = engineers[path.fold]
        keep = np.isin(rows, path.fit)
        encoder = label_excluded_engineer(eng, Xs, y[rows], keep)
    roles = {"fit_rows": path.fit} if path.in_sample else {"held_out": path.query, "fit_rows": path.fit}
    blocks, values, notes = {}, {}, {}
    for role, qrows in roles.items():
        Xq = X.iloc[qrows]
        b = _stage2_blocks(eng, Xs, keep, Xq)
        verify = role == "held_out" or path.in_sample
        prod = encoder.transform(Xq) if verify else None
        for f in STAGE23_FEATURES:
            blocks[(f, role)] = b[f]
            values[(f, role)] = prod[f].to_numpy(float) if verify else None
        cb = b["dow_month_encoded"][0]
        notes[("dow_month_encoded", role)] = {"fallback_rows": int((cb.qcls == cb.mass.shape[1]).sum())}
        notes[("neighborhood_subscription_density", role)] = {
            "nsd_scaler_rows": int(len(rows)), "nsd_sample_pool_rows": int(len(rows)),
            "nsd_label_rows": int((b["neighborhood_subscription_density"][0].member >= 0).sum())}
    return _Fitted(rows, blocks, values, notes)


def _stage3_fit(X: pd.DataFrame, y: np.ndarray, path: FitPath) -> _Fitted:
    """``EBMFeaturePipeline`` (Stage 3's per-fold pipeline) on the fit rows."""
    from feature_research import feature_inventory as FI
    from feature_research.feature_engineering.integrals import _KDE_LARGE_DS_THRESH
    from shared.stage3.feature_engineering import EBMFeaturePipeline

    Xf = X.iloc[path.fit]
    yf = pd.Series(y[path.fit], index=Xf.index)
    pipe = EBMFeaturePipeline(features=FI._stage3_columns())
    in_fit = pipe.fit_transform(Xf, yf)
    ie, de, te = (pipe.engineers_[k] for k in ("integral", "derivative", "temporal"))
    ref = _reference_positions(ie, Xf, _KDE_LARGE_DS_THRESH)
    codes = {feat: _qcut_codes(Xf[feat], de.n_bins, edges=de._curvature_store[feat][0])
             for feat in _CURV_FEATS}
    roles = {"fit_rows": path.fit} if path.in_sample else {"held_out": path.query, "fit_rows": path.fit}
    blocks, values, notes = {}, {}, {}
    for role, qrows in roles.items():
        Xq = X.iloc[qrows]
        prod = in_fit if role == "fit_rows" else pipe.transform(Xq)
        blocks[("neighborhood_subscription_density", role)] = [
            _kde_block(ie._scaler, ie.bandwidth, Xf[_KDE_COLS], ref, Xq[_KDE_COLS])]
        curv = []
        for feat in _CURV_FEATS:
            edges = de._curvature_store[feat][0]
            qcode = pd.cut(Xq[feat].astype(float), bins=edges, labels=False,
                           include_lowest=True).to_numpy()
            qcode = np.where(np.isnan(qcode), -1, qcode).astype(np.int64)
            curv.append(_curvature_block(f"curvature[{feat}]", codes[feat], len(edges) - 1, qcode, 1 / 3))
        blocks[("economic_curvature_intensity", role)] = curv
        blocks[("dow_month_encoded", role)] = _cell_blocks(
            Xf["day_of_week"], Xf["month"], Xq["day_of_week"], Xq["month"], te.smoothing_factor)
        for f in STAGE23_FEATURES:
            values[(f, role)] = prod[f].to_numpy(float)
        cb = blocks[("dow_month_encoded", role)][0]
        notes[("dow_month_encoded", role)] = {"fallback_rows": int((cb.qcls == cb.mass.shape[1]).sum())}
        product = _compare(prod["joint_economic_decay"].to_numpy(float)
                           * prod["neighborhood_subscription_density"].to_numpy(float),
                           prod["decay_x_density"].to_numpy(float))
        notes[("neighborhood_subscription_density", role)] = {
            "decay_x_density_status": product["status"],
            "decay_x_density_max_abs_diff": product["max_abs_diff"],
            "decay_x_density_nonfinite": product["nonfinite_reconstructed"] + product["nonfinite_production"],
            "nsd_scaler_rows": int(len(path.fit)), "nsd_sample_pool_rows": int(len(path.fit)),
            "nsd_label_rows": int(len(ref))}
    return _Fitted(path.fit, blocks, values, notes)


# -- running the paths --------------------------------------------------
def _next_day(T: pd.DataFrame) -> np.ndarray:
    day = T["day"].to_numpy()
    nxt = np.zeros(day.max() + 2, dtype=np.int64)
    nxt[day] = T["next_period_day"].to_numpy()
    nxt[-1] = day.max() + 1
    return nxt


def _shares(res: Dict[str, Any]) -> Dict[str, np.ndarray]:
    m = res["mass"]
    total = sum(m[k][x] for k in m for x in RELATIONS) + res["self"]
    total = np.where(total > 0, total, 1.0)
    out = {f"{k}_{x}": m[k][x] / total for k in m for x in RELATIONS}
    out["self"] = res["self"] / total
    out["later"] = sum(out[f"{k}_{x}"] for k in m for x in LATER)
    return out


#: Largest |reconstructed - production| a comparison may show.
TOLERANCE = 1e-12


def _compare(reconstructed: np.ndarray, production: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    One reconstruction-versus-production comparison, before any aggregation.

    ``skipped`` when no production value was computed (``production is None``);
    otherwise ``passed`` only if both arrays have the same length, every value of
    both is finite and the largest difference is within ``TOLERANCE``. A
    nonfinite value fails the comparison: it is never dropped.
    """
    rec = np.asarray(reconstructed, dtype=float)
    if production is None:
        return {"status": "skipped", "max_abs_diff": None, "nonfinite_reconstructed": 0,
                "nonfinite_production": 0, "length_mismatch": False}
    prod = np.asarray(production, dtype=float)
    bad_rec, bad_prod = int((~np.isfinite(rec)).sum()), int((~np.isfinite(prod)).sum())
    mismatch = rec.shape != prod.shape
    diff = None
    if not mismatch:
        both = np.isfinite(rec) & np.isfinite(prod)
        diff = float(np.max(np.abs(rec[both] - prod[both]))) if both.any() else None
    ok = (not mismatch and bad_rec == 0 and bad_prod == 0 and len(rec) > 0
          and diff is not None and diff <= TOLERANCE)
    return {"status": "passed" if ok else "failed", "max_abs_diff": diff,
            "nonfinite_reconstructed": bad_rec, "nonfinite_production": bad_prod,
            "length_mismatch": bool(mismatch)}


def _fit_row(feature: str, path: FitPath, role: str, res: Dict[str, Any],
             production: Optional[np.ndarray], notes: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-fit totals for one feature and role: sums and counts, so fits add up."""
    sh = _shares(res)
    cmp = _compare(res["value"], production)
    r = res["rows"]
    any_later = np.zeros(len(res["value"]), dtype=bool)
    for k in r:
        for x in LATER:
            any_later |= r[k][x] > 0
    out = {"feature": feature, "stage": path.stage, "path": path.path, "role": role,
           "fold": path.fold, "inner": path.inner, "fit_rows": int(len(path.fit)),
           "rows": int(len(res["value"])),
           "comparison": cmp["status"], "max_abs_diff": cmp["max_abs_diff"],
           "nonfinite_reconstructed": cmp["nonfinite_reconstructed"],
           "nonfinite_production": cmp["nonfinite_production"],
           "self_rows": int((res["self"] > 0).sum()), "self_share_sum": float(sh["self"].sum()),
           "self_share_max": float(sh["self"].max()) if len(sh["self"]) else 0.0,
           "any_later_rows": int(any_later.sum()),
           "majority_later_rows": int((sh["later"] > 0.5).sum()),
           "later_share_sum": float(sh["later"].sum()),
           "fallback_rows": notes.get("fallback_rows")}
    for k in r:
        for x in RELATIONS:
            out[f"{k}_{x}_share_sum"] = float(sh[f"{k}_{x}"].sum())
            out[f"{k}_{x}_rows"] = int((r[k][x] > 0).sum())
            out[f"{k}_{x}_contributors"] = int(r[k][x].sum())
    for key in ("decay_x_density_status", "decay_x_density_max_abs_diff", "decay_x_density_nonfinite",
                "nsd_scaler_rows", "nsd_sample_pool_rows", "nsd_label_rows"):
        if key in notes:
            out[key] = notes[key]
    return out, sh["later"]


def audit_paths(X: pd.DataFrame, y: pd.Series, T: pd.DataFrame,
                paths: Mapping[str, List[FitPath]], *, verbose: bool = True) -> Dict[str, Any]:
    """
    Attribution of every audited feature on every fitting path, by role:

    ``held_out``      a fold's held-out rows (mechanisms b and c). Each is located
                      among the fit's source rows, so a held-out label that entered
                      its own encoding would be measured as ``self`` (b); a row
                      outside the source, or in it but excluded from the label
                      statistics (Stage 2 label-excluded folds), measures zero;
    ``fit_rows``      the fit's own rows, i.e. the training representation of the
                      models fitted on that path (a, and its look-ahead);
    ``final_mapping`` the final fits only: each training row's day and covariates
                      standing for a test row contacted then. By convention the row's
                      own label counts as a same-day contact (a test row's label is
                      never fitted), so self accounting is off for this role only.

    Returns per-fit rows, every row's later share pooled per (feature, stage,
    path, role), and the per-row relation shares of the final mappings.
    """
    from shared.stage2.feature_engineering import RFFeatureEngineer

    yv = np.asarray(y).astype(int)
    day = T["day"].to_numpy()
    nxt = _next_day(T)
    engineers = {p.fold: RFFeatureEngineer().fit(X.iloc[p.fit], pd.Series(yv[p.fit], index=X.index[p.fit]))
                 for p in paths.get("stage2", []) if p.path in ("outer", "final")}
    fits, later, mapping = [], {}, {}
    for stage, ps in paths.items():
        for p in ps:
            fitted = (_stage1_fit(X, yv, p) if stage == "stage1" else
                      _stage2_fit(X, yv, p, engineers) if stage == "stage2" else _stage3_fit(X, yv, p))
            members = fitted.rows
            where = np.full(len(yv), -1, dtype=np.int64)
            where[members] = np.arange(len(members))
            for (feat, role), bl in fitted.blocks.items():
                q = p.fit if role == "fit_rows" else p.query
                runs = [(role, where[q])]                     # every row located in the source
                if p.path == "final":
                    runs.append(("final_mapping", None))      # the proxy's same-day convention
                notes = fitted.notes.get((feat, role), {})
                for name, own in runs:
                    res = attribute(bl, yv[members], day[members], day[q], nxt, own)
                    row, sh = _fit_row(feat, p, name, res, fitted.values[(feat, role)],
                                       notes if name == role else
                                       {k: v for k, v in notes.items()
                                        if not k.startswith("decay_x_density")})
                    fits.append(row)
                    later.setdefault((feat, stage, p.path, name), []).append(sh)
                    if name == "final_mapping":
                        full = _shares(res)
                        mapping[(feat, stage)] = pd.DataFrame({
                            "period": T["period"].to_numpy()[q],
                            **{x: full[f"direct_{x}"] + full[f"prior_{x}"] for x in RELATIONS}})
        if verbose:
            print(f"  {stage}: {len(ps)} fits attributed")
    return {"fits": fits, "later": {k: np.concatenate(v) for k, v in later.items()},
            "final_mapping": mapping}


#: The per-fit columns a record keeps (the summary keeps every total).
COMPACT_COLUMNS = ("feature", "stage", "path", "role", "fold", "inner", "fit_rows", "rows",
                   "comparison", "max_abs_diff", "self_rows", "any_later_rows", "direct_later_month_rows",
                   "direct_later_day_same_month_rows", "direct_same_day_rows",
                   "majority_later_rows", "later_share_sum")


def compact_fits(audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-fit rows of the held-out and final-mapping roles, ``COMPACT_COLUMNS`` only."""
    return [{k: r[k] for k in COMPACT_COLUMNS} for r in audit["fits"]
            if r["role"] in ("held_out", "final_mapping")]


def path_summary(audit: Mapping[str, Any]) -> pd.DataFrame:
    """Totals per (feature, stage, path, role): counts with denominators and mean shares."""
    df = pd.DataFrame(audit["fits"])
    rows = []
    for (feat, stage, path, role), g in df.groupby(["feature", "stage", "path", "role"], sort=False):
        q = int(g["rows"].sum())
        later = audit["later"][(feat, stage, path, role)]
        status = g["comparison"].value_counts()
        diffs = g["max_abs_diff"].dropna()
        row = {"feature": feat, "stage": stage, "path": path, "role": role, "fits": len(g),
               "fit_rows_min": int(g["fit_rows"].min()), "fit_rows_max": int(g["fit_rows"].max()),
               "rows": q, "comparisons_passed": int(status.get("passed", 0)),
               "comparisons_failed": int(status.get("failed", 0)),
               "comparisons_skipped": int(status.get("skipped", 0)),
               "nonfinite_values": int(g["nonfinite_reconstructed"].sum() + g["nonfinite_production"].sum()),
               "max_abs_diff": float(diffs.max()) if len(diffs) else None,
               "self_rows": int(g["self_rows"].sum()),
               "self_share_mean": float(g["self_share_sum"].sum() / q),
               "self_share_max": float(g["self_share_max"].max()),
               "any_later_rows": int(g["any_later_rows"].sum()),
               "later_month_rows": int(g["direct_later_month_rows"].sum()),
               "later_day_rows": int(g["direct_later_day_same_month_rows"].sum()),
               "same_day_rows": int(g["direct_same_day_rows"].sum()),
               "prior_later_rows": int(g["prior_later_month_rows"].sum()),
               "majority_later_rows": int(g["majority_later_rows"].sum()),
               "later_share_mean": float(later.mean()), "later_share_median": float(np.median(later)),
               "later_share_p90": float(np.quantile(later, 0.9)), "later_share_max": float(later.max()),
               "fallback_rows": (int(g["fallback_rows"].dropna().sum())
                                 if g["fallback_rows"].notna().any() else None),
               "mean_later_month_contributors": float(g["direct_later_month_contributors"].sum() / q),
               "mean_contributors": float(sum(g[f"direct_{x}_contributors"].sum()
                                              for x in RELATIONS) / q)}
        for kind in ("direct", "prior"):
            for x in RELATIONS:
                row[f"{kind}_{x}_share"] = float(g[f"{kind}_{x}_share_sum"].sum() / q)
        if "decay_x_density_status" in g and g["decay_x_density_status"].notna().any():
            st = g["decay_x_density_status"].dropna().value_counts()
            v = g["decay_x_density_max_abs_diff"].dropna()
            row["decay_x_density_passed"] = int(st.get("passed", 0))
            row["decay_x_density_failed"] = int(st.get("failed", 0))
            row["decay_x_density_max_abs_diff"] = float(v.max()) if len(v) else None
        for key in ("nsd_scaler_rows", "nsd_sample_pool_rows", "nsd_label_rows"):
            if key in g and g[key].notna().any():
                v = g[key].dropna().astype(int)
                row[f"{key}_min"], row[f"{key}_max"] = int(v.min()), int(v.max())
        rows.append(row)
    return pd.DataFrame(rows)


#: The production comparisons each path is expected to make, by role. ``skipped``
#: is intentional: a Stage 2 fold fit's own rows are not re-encoded by production
#: (the same blocks are compared on that fit's held-out rows).
COMPARISON_POLICY = {
    "stage1": {"final": {"fit_rows": "compared", "final_mapping": "compared"},
               "*": {"held_out": "compared", "fit_rows": "compared"}},
    "stage2": {"final": {"fit_rows": "compared", "final_mapping": "compared"},
               "*": {"held_out": "compared", "fit_rows": "skipped"}},
    "stage3": {"final": {"fit_rows": "compared", "final_mapping": "compared"},
               "*": {"held_out": "compared", "fit_rows": "compared"}},
}
#: Stage 3's ``decay_x_density = joint_economic_decay x NSD`` check, per fitted role.
PRODUCT_ROLES = {"final": ("fit_rows",), "*": ("held_out", "fit_rows")}
_STAGE_FEATURES = {"stage1": STAGE1_FEATURES, "stage2": STAGE23_FEATURES, "stage3": STAGE23_FEATURES}


def expected_comparisons(paths: Mapping[str, List[FitPath]]) -> pd.DataFrame:
    """The comparisons ``COMPARISON_POLICY`` expects from these paths, by group."""
    rows = []
    for stage, ps in paths.items():
        by = {}
        for p in ps:
            by[p.path] = by.get(p.path, 0) + 1
        for path, fits in by.items():
            policy = COMPARISON_POLICY[stage].get(path, COMPARISON_POLICY[stage]["*"])
            for role, want in policy.items():
                for feat in _STAGE_FEATURES[stage]:
                    rows.append({"stage": stage, "path": path, "role": role, "feature": feat,
                                 "fits": fits, "expected": want})
    return pd.DataFrame(rows)


def comparison_coverage(audit: Mapping[str, Any], paths: Mapping[str, List[FitPath]]) -> pd.DataFrame:
    """
    Expected versus completed reconstruction comparisons, as checks: per (stage,
    path, role, feature) every expected comparison was made and passed, every
    intentional skip was a skip, nothing failed (a nonfinite value is a
    failure) and nothing unexpected ran; the same for Stage 3's
    ``decay_x_density`` product checks; and the totals.
    """
    fits = pd.DataFrame(audit["fits"])
    exp = expected_comparisons(paths)
    out = []
    keys = ["stage", "path", "role", "feature"]
    done = fits.groupby(keys)["comparison"].value_counts().unstack(fill_value=0)
    for c in ("passed", "failed", "skipped"):
        if c not in done:
            done[c] = 0
    seen = set()
    totals = {"expected_compared": 0, "expected_skipped": 0, "passed": 0, "failed": 0, "skipped": 0}
    for _, e in exp.iterrows():
        key = tuple(e[k] for k in keys)
        seen.add(key)
        got = done.loc[key] if key in done.index else pd.Series({"passed": 0, "failed": 0, "skipped": 0})
        want_c = e["fits"] if e["expected"] == "compared" else 0
        want_s = e["fits"] if e["expected"] == "skipped" else 0
        ok = int(got["failed"]) == 0 and int(got["passed"]) == want_c and int(got["skipped"]) == want_s
        totals["expected_compared"] += want_c
        totals["expected_skipped"] += want_s
        for c in ("passed", "failed", "skipped"):
            totals[c] += int(got[c])
        out.append(_check(f"{e['stage']} {e['path']} {e['role']} {e['feature']}: expected "
                          f"{want_c} compared / {want_s} skipped", ok,
                          f"passed {int(got['passed'])}, failed {int(got['failed'])}, "
                          f"skipped {int(got['skipped'])}"))
    extra = sorted(set(done.index) - seen)
    out.append(_check("no comparison group ran that the policy does not expect", not extra,
                      extra or "none"))
    product_expected = sum(len(PRODUCT_ROLES.get(p.path, PRODUCT_ROLES["*"]))
                           for p in paths.get("stage3", []))
    nsd = fits[(fits["stage"] == "stage3") & (fits["feature"] == "neighborhood_subscription_density")]
    st = (nsd["decay_x_density_status"].dropna().value_counts()
          if "decay_x_density_status" in nsd else pd.Series(dtype=int))
    out.append(_check(f"stage3 decay_x_density = joint_economic_decay x NSD: expected "
                      f"{product_expected} checks", int(st.get("passed", 0)) == product_expected
                      and int(st.get("failed", 0)) == 0,
                      f"passed {int(st.get('passed', 0))}, failed {int(st.get('failed', 0))}"))
    out.append(_check(f"all comparisons: expected {totals['expected_compared']} compared and "
                      f"{totals['expected_skipped']} skipped",
                      totals["passed"] == totals["expected_compared"] and totals["failed"] == 0
                      and totals["skipped"] == totals["expected_skipped"],
                      f"passed {totals['passed']}, failed {totals['failed']}, skipped {totals['skipped']} "
                      f"of {len(fits)} attributions"))
    return pd.DataFrame(out)


def held_out_own_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Mechanism (b), measured: held-out encodings that carry the row's own label, per path."""
    held = summary[summary["role"] == "held_out"]
    return pd.DataFrame([_check(f"{r['stage']} {r['path']} {r['feature']}: held-out encodings "
                                "carrying their own label", int(r["self_rows"]) == 0,
                                f"{int(r['self_rows']):,}/{int(r['rows']):,}")
                         for _, r in held.iterrows()])


def mapping_by_period(audit: Mapping[str, Any]) -> pd.DataFrame:
    """
    The final fits' mappings (the ones that score the test split), at each
    training row's contact day: mean label share by relation, per feature,
    stage and period. Training rows stand in for contemporaneous test rows.
    """
    out = []
    for (feat, stage), frame in audit["final_mapping"].items():
        g = frame.groupby("period")
        t = g[list(RELATIONS)].mean()
        t.insert(0, "rows", g.size())
        t.insert(0, "stage", stage)
        t.insert(0, "feature", feat)
        out.append(t.reset_index())
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# 5. isolation checks
# ----------------------------------------------------------------------
def _designs_equal(a, b) -> bool:
    return (np.array_equal(a.train, b.train) and np.array_equal(a.val, b.val)
            and np.array_equal(a.X_train, b.X_train) and np.array_equal(a.X_val, b.X_val))


def _flipped(y: pd.Series, rows) -> pd.Series:
    out = y.copy()
    if rows is not None:
        out.iloc[rows] = 1 - out.iloc[rows]
    return out


class _Relabelled:
    """
    ``LRFeatureEngineer`` that is fitted with the labels of ``labels`` for the
    rows it is given, whatever labels its caller passes. Through the builders'
    ``make_engineer`` hook it changes the labels every encoder fit sees while the
    fold plans (stratified on the true labels) stay fixed.
    """

    def __init__(self, labels: pd.Series):
        from shared.stage1.feature_engineering import LRFeatureEngineer
        self.labels, self.engineer = labels, LRFeatureEngineer()

    def fit(self, X, y):
        self.engineer.fit(X, self.labels.loc[X.index])
        return self

    def transform(self, X):
        return self.engineer.transform(X)


def _outer_designs(X: pd.DataFrame, y: pd.Series, seen: Optional[pd.Series] = None):
    """
    ``build_fold_designs`` as both Stage 1 arms call it (raw rows, 10 folds, seed 42);
    ``seen``: the labels the encoders are fitted with (default ``y``).
    """
    from shared.stage1.crossfit import build_fold_designs
    from shared.stage1.feature_engineering import LR_FEATURES

    seen = y if seen is None else seen
    make = lambda: _Relabelled(seen)  # noqa: E731
    feats = _quiet(_quiet(make().fit, X, y).transform, X)[LR_FEATURES]
    return feats, build_fold_designs(feats, y, cv_folds=10, random_state=SEED, X_raw=X,
                                     make_engineer=make)


def _inner_designs(design, feats, X, y, seen: Optional[pd.Series] = None):
    """``inner_fold_designs`` as ``calibrate_oof`` calls it for one outer fold."""
    from shared.stage1.crossfit import inner_fold_designs

    seen = y if seen is None else seen
    return inner_fold_designs(design, feats, y, cv_folds=10, random_state=SEED, X_raw=X,
                              make_engineer=lambda: _Relabelled(seen))


def isolation_checks(X: pd.DataFrame, y: pd.Series, paths: Mapping[str, List[FitPath]],
                     *, self_sample: int = 20) -> pd.DataFrame:
    """
    Label flips through the production fold builders, on the training rows.

    Stage 1 (``build_fold_designs`` / ``inner_fold_designs``, the calls both arms
    make for tuning, OOF and calibration): the fold plans are stratified on the
    labels, so the labels the encoders see are flipped through the builders'
    ``make_engineer`` hook while the plans stay those of the true labels.
    Flipping fold k's held-out labels must leave design k's training and
    held-out matrices unchanged; flipping its training labels must change its
    held-out matrix (so the check can fail). The same for one inner fold of
    every outer fold; and flipping an outer fold's held-out labels (in the label
    vector itself) must leave all of its inner designs unchanged. Mechanism (a):
    flipping one training row's label moves its own full-train encodings by
    exactly the self weight the attribution assigns it.
    Stage 2 (``label_excluded_engineer``, the ``fold_label_excluded`` encoder):
    flipping inner fold j's labels leaves its label-excluded features unchanged.
    The Stage 2 outer and Stage 3 fold fits are passed only the fold's training
    labels (``label_argument_checks``).
    """
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    out = []
    feats, base = _outer_designs(X, y)
    plan = [p for p in paths["stage1"] if p.path == "outer"]
    same_plan = all(np.array_equal(d.val, p.query) and np.array_equal(d.train, p.fit)
                    for d, p in zip(base, plan))
    out.append(_check("stage1: the production designs' folds are the audited plan "
                      "(fold_assignment(y, 10, 42))", same_plan and len(base) == len(plan),
                      f"{len(base)} folds"))
    held_ok, ctrl_ok = [], []
    for k, d in enumerate(base):
        held_ok.append(_designs_equal(_outer_designs(X, y, _flipped(y, d.val))[1][k], d))
        ctrl_ok.append(not np.array_equal(
            _outer_designs(X, y, _flipped(y, d.train))[1][k].X_val, d.X_val))
    out.append(_check("stage1 outer folds (tuning and OOF): with fold k's held-out labels "
                      "flipped for every encoder fit, design k's training and held-out matrices "
                      "are unchanged (mechanism b absent)", all(held_ok),
                      f"{sum(held_ok)}/{len(held_ok)} folds"))
    out.append(_check("stage1 outer folds: with fold k's training labels flipped, its held-out "
                      "matrix changes (control: the check can fail)", all(ctrl_ok),
                      f"{sum(ctrl_ok)}/{len(ctrl_ok)} folds"))
    inner_held, outer_into_inner, inner_ctrl = [], [], []
    for k, d in enumerate(base):
        inner = _inner_designs(d, feats, X, y)
        inner_plan = [p for p in paths["stage1"] if p.path == "inner" and p.fold == k]
        same_plan = len(inner) == len(inner_plan) and all(
            np.array_equal(d.train[i.val], p.query) for i, p in zip(inner, inner_plan))
        j = k % len(inner)
        inner_held.append(same_plan and _designs_equal(
            _inner_designs(d, feats, X, y, _flipped(y, d.train[inner[j].val]))[j], inner[j]))
        outer_into_inner.append(all(_designs_equal(a, b) for a, b in zip(
            _inner_designs(d, feats, X, _flipped(y, d.val)), inner)))
        inner_ctrl.append(not np.array_equal(
            _inner_designs(d, feats, X, y, _flipped(y, d.train[inner[j].train]))[j].X_val,
            inner[j].X_val))
    out.append(_check("stage1 inner calibration folds: with inner fold j's held-out labels "
                      "flipped, inner design j is unchanged (inner fold k mod 10 of each outer "
                      "fold k; the plans are the audited ones)", all(inner_held),
                      f"{sum(inner_held)}/{len(inner_held)}"))
    out.append(_check("stage1 inner calibration folds: flipping the outer fold's held-out labels "
                      "leaves all ten inner designs unchanged (F19)", all(outer_into_inner),
                      f"{sum(outer_into_inner)}/{len(outer_into_inner)} outer folds"))
    out.append(_check("stage1 inner calibration folds: with inner fold j's training labels "
                      "flipped, its held-out matrix changes (control)", all(inner_ctrl),
                      f"{sum(inner_ctrl)}/{len(inner_ctrl)}"))
    out.append(_self_flip_check(X, y, self_sample))
    out += _stage2_inner_flips(X, y, paths)
    out += label_argument_checks()
    return pd.DataFrame(out)


def _self_flip_check(X: pd.DataFrame, y: pd.Series, sample: int) -> Dict[str, Any]:
    """One training row's label flipped: its own full-train encodings move by its self weight."""
    from shared.stage1.feature_engineering import LRFeatureEngineer

    yv = y.to_numpy()
    n = len(yv)
    full = _stage1_fit(X, yv, FitPath("stage1", "final", None, None, np.arange(n), np.arange(n),
                                      in_sample=True))
    rows = np.random.default_rng(SEED).choice(n, size=sample, replace=False)
    worst, moved = 0.0, 0
    for r in rows:
        eng = _quiet(LRFeatureEngineer().fit, X, _flipped(y, [r]))
        new = _quiet(eng.transform, X.iloc[[r]])
        for feat in STAGE1_FEATURES:
            bl = full.blocks[(feat, "fit_rows")]
            w = sum(b.scale * b.mass[b.qcls[r], b.member[r]] for b in bl if b.member[r] >= 0)
            predicted = full.values[(feat, "fit_rows")][r] + w * (1 - 2 * yv[r])
            got = float(new[feat].iloc[0])
            worst = max(worst, abs(got - predicted))
            moved += got != full.values[(feat, "fit_rows")][r]
    return _check("stage1 final fit: flipping one training row's label moves its own encodings "
                  "by exactly the self weight attributed to it (mechanism a)",
                  worst < 1e-12 and moved > 0,
                  f"{sample} rows x 2 features: {moved} values moved, max deviation {worst:.1e}")


def _stage2_inner_flips(X: pd.DataFrame, y: pd.Series, paths) -> List[Dict[str, Any]]:
    from shared.stage2.feature_engineering import RFFeatureEngineer
    from stage2_crossfit.g13 import label_excluded_engineer

    yv = y.to_numpy()
    same, ctrl = [], []
    for p in [p for p in paths["stage2"] if p.path == "outer"]:
        tr = p.fit
        Xs, ys = X.iloc[tr], yv[tr]
        plan = _router_inner_plan(ys)

        def features(labels, j):
            eng = RFFeatureEngineer().fit(Xs, pd.Series(labels, index=Xs.index))
            return label_excluded_engineer(eng, Xs, labels, plan != j).transform(
                Xs[plan == j]).to_numpy(float)

        for j in range(5):
            yf = ys.copy()
            yf[plan == j] = 1 - yf[plan == j]
            same.append(bool(np.array_equal(features(ys, j), features(yf, j))))
        yc = ys.copy()
        yc[plan != 0] = 1 - yc[plan != 0]
        ctrl.append(not np.array_equal(features(ys, 0), features(yc, 0)))
    return [_check("stage2 label-excluded inner folds: flipping inner fold j's labels leaves "
                   "its label-excluded source features unchanged (all 5 x 5)", all(same),
                   f"{sum(same)}/{len(same)}"),
            _check("stage2 label-excluded inner folds: flipping the kept rows' labels changes "
                   "inner fold 0's features (control)", all(ctrl),
                   f"{sum(ctrl)}/{len(ctrl)} outer folds")]


#: The production calls that fit an encoding inside a fold: (file, enclosing
#: function, callee, label argument position, its expected source, and the
#: binding that must define it in that function, if any).
LABEL_ARGUMENTS = (
    ("shared/stage1/crossfit.py", "_designs", "_quiet_engineer", 2, "y.iloc[tr]", None),
    ("glass_pipeline/glass_router/pipeline/crossfit.py", "crossfit_stage2", "fit_glass_fold", 1,
     "y_tr", "y_train.iloc[tr]"),
    ("glass_pipeline/glass_router/pipeline/crossfit.py", "fit_glass_fold", "engineer_features", 2,
     "y_tr", None),
    ("blackbox_pipeline/models/rf_router/crossfit.py", "crossfit_rf_router", "engineer", 2,
     "y.iloc[tr]", None),
    ("shared/stage3/data.py", "fold_frames", "p.fit_transform", 1, "y_tr.iloc[tr]", None),
    # the black-box MLP: its tuning folds are the per-fold designs, and every fold
    # model (whose early-stopping split is drawn inside ``fit``) sees one design's rows
    ("blackbox_pipeline/models/mlp/stage.py", "fit", "tune_stage1_mlp_auc", "folds", "designs", None),
    ("blackbox_pipeline/models/mlp/tuning.py", "objective", "Stage1MLPClassifier(**kwargs).fit", 1,
     "y_tr", None),
    ("shared/stage1/oof_calibration.py", "_fit_on", "clone(base).fit", 1, "design.y_train", None),
)


def label_argument_checks(root: Path = BANK_ROOT) -> List[Dict[str, Any]]:
    """AST: each in-fold encoder fit is passed the fold's training labels only."""
    out = []
    for rel, where, func, arg, expected, binding in LABEL_ARGUMENTS:
        tree = ast.parse((Path(root) / rel).read_bytes())
        scopes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == where]
        calls = [n for sc in scopes for n in ast.walk(sc)
                 if isinstance(n, ast.Call) and ast.unparse(n.func) == func]
        if isinstance(arg, str):
            found = sorted({ast.unparse(k.value) for n in calls for k in n.keywords if k.arg == arg})
        else:
            found = sorted({ast.unparse(n.args[arg]) for n in calls if len(n.args) > arg})
        ok = len(scopes) == 1 and found == [expected]
        detail = f"in {where}: label argument {found}"
        if binding is not None:
            bound = [ast.unparse(n) for sc in scopes for n in ast.walk(sc)
                     if isinstance(n, ast.Assign) and expected in ast.unparse(n.targets[0])
                     and binding in ast.unparse(n.value)]
            ok = ok and bool(bound)
            detail += f"; bound by {bound}"
        what = "the per-fold designs" if isinstance(arg, str) else "the fold's training labels"
        out.append(_check(f"{rel}: {func}(...) is passed {what}", ok, detail))
    return out


def router_plan_reference(X: pd.DataFrame, y: pd.Series, paths) -> pd.DataFrame:
    """
    Stage 2 outer fold 0 and its inner plan against the inner-feature pathway the
    accepted Stage 2 run (PR 38 ``c3_20261112_153333``, pinned by PR 39) recorded
    for fold 0 (``g3_preflight_fold0.json``).
    """
    import hashlib
    import json

    from shared.stage2.feature_engineering import RFFeatureEngineer
    from shared.stage_io import label_fingerprint
    from stage2_crossfit.g13 import kde_reference_positions
    from stage2_crossfit.inner_features import plan_fingerprint

    rel = "research_logs/pr38/c3_20261112_153333/g3_preflight_fold0.json"
    data = (BANK_ROOT / rel).read_bytes()
    found, stack = [], [json.loads(data.decode("utf-8"))]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("inner_feature_pathway"), dict):
                found.append(node["inner_feature_pathway"])
            stack += list(node.values())
        elif isinstance(node, list):
            stack += node
    recs = {json.dumps(r, sort_keys=True) for r in found}
    if len(recs) != 1:
        raise AssertionError(f"{rel}: expected one inner_feature_pathway, found {len(recs)} variants")
    rec = found[0]
    p = [q for q in paths["stage2"] if q.path == "outer" and q.fold == 0][0]
    yv = np.asarray(y).astype(int)
    tr = p.fit
    plan = _router_inner_plan(yv[tr])
    eng = RFFeatureEngineer().fit(X.iloc[tr], pd.Series(yv[tr], index=X.index[tr]))
    ref = kde_reference_positions(eng, X.iloc[tr])
    computed = {"fit_rows": int(len(tr)), "fit_positives": int(yv[tr].sum()),
                "fit_label_fingerprint": label_fingerprint(X.index[tr], yv[tr]),
                "plan_fingerprint": plan_fingerprint(plan), "kde_reference_rows": int(len(ref)),
                "labels_excluded": [int((plan == j).sum()) for j in range(5)],
                "kde_reference_rows_kept": [int((plan[ref] != j).sum()) for j in range(5)]}
    recorded = {"fit_rows": rec["fit_rows"], "fit_positives": rec["fit_positives"],
                "fit_label_fingerprint": rec["fit_label_fingerprint"],
                "plan_fingerprint": rec["plan_fingerprint"],
                "kde_reference_rows": rec["kde_reference_rows"],
                "labels_excluded": [f["labels_excluded"] for f in rec["inner_folds"]],
                "kde_reference_rows_kept": [f["kde_reference_rows_kept"] for f in rec["inner_folds"]]}
    src = f"{rel} (sha256 {hashlib.sha256(data).hexdigest()})"
    return pd.DataFrame([{"item": k, "computed": str(computed[k]), "recorded": str(recorded[k]),
                          "passed": computed[k] == recorded[k], "source": src} for k in computed])


#: The accepted Stage 1 fits (PR 39 Commit 2 official E-C evidence, both arms).
STAGE1_FIT_RECORD = "research_logs/pr39/c2_20261114_184433/stage1_calibration_evidence.json"
_FOLD_CHECK_WORDS = ("fold", "scope", "nested")


def accepted_fit_scopes(accepted: Sequence[Mapping[str, Any]],
                        commit2: Mapping[str, Any]) -> pd.DataFrame:
    """
    Whether the accepted fits used the fold-refitted transformations: the
    accepted artifacts' boundary-extractor fold checks (this run's Cell 11 and
    the verified Commit 2 record, where the artifacts were available), and the
    accepted Stage 1 fits of both arms as PR 39 Commit 2 recorded them (the MLP
    has no extractor). Read only; no artifact is opened here.
    """
    import hashlib
    import json

    rows = []
    now = {r["role"]: r for r in accepted}
    for r in commit2["accepted_artifacts"]:
        names = [c for c in r.get("check_names") or [] if any(w in c.lower() for w in _FOLD_CHECK_WORDS)]
        rows.append({"item": f"accepted {r['role']}: boundary-extractor fold checks",
                     "evidence": "; ".join(names) or "no extractor (not read)",
                     "this_run": now.get(r["role"], {}).get("status"),
                     "commit2_record": r["status"],
                     "passed": r["status"] in ("passed", "not read")})
    data = (BANK_ROOT / STAGE1_FIT_RECORD).read_bytes()
    rec = json.loads(data.decode("utf-8"))
    src = f"{STAGE1_FIT_RECORD} (sha256 {hashlib.sha256(data).hexdigest()[:16]})"
    lr, mlp = rec["fits"]["glass_lr"], rec["fits"]["blackbox_mlp"]
    per_fold = "encoder, scaler, model and calibrator fitted per fold"
    pre = lr["config"].get("oof_preprocessing", {})
    rows += [
        {"item": "accepted GLASS LR fit: per-fold encoders in tuning, OOF and calibration folds",
         "evidence": f"oof_preprocessing {pre}; tune_cv_folds {lr['config']['tune_cv_folds']} = "
                     f"cv_folds {lr['config']['cv_folds']}; oof_provenance {lr['oof_provenance']!r}",
         "this_run": "recorded", "commit2_record": src,
         "passed": (pre.get("feature_fit_scope") == "per_fold"
                    and "calibration_inner_folds" in pre.get("applies_to", "")
                    and lr["config"]["tune_cv_folds"] == lr["config"]["cv_folds"] == 10
                    and per_fold in lr["oof_provenance"])},
        {"item": "accepted black-box MLP fit: per-fold encoders (tuning reuses the OOF designs)",
         "evidence": f"cv_folds {mlp['config']['cv_folds']}; oof_provenance {mlp['oof_provenance']!r}",
         "this_run": "recorded", "commit2_record": src,
         "passed": mlp["config"]["cv_folds"] == 10 and per_fold in mlp["oof_provenance"]},
    ]
    for r in commit2["checks"]["fit_protocols"]:
        rows.append({"item": f"{r['stage']} fit protocol (code): {r['check']}",
                     "evidence": str(r.get("detail") or ""), "this_run": "code",
                     "commit2_record": "checks.fit_protocols", "passed": bool(r["passed"])})
    return pd.DataFrame(rows)


MLP_EARLY_STOPPING_ORDER = (
    "Inside every black-box MLP fit - each Optuna trial on each of the 10 outer designs, each "
    "calibration refit on an outer or inner design (oof_calibration._fit_on) and the final model - "
    "the Stage 1 encoders and scaler are fitted first, on all of that fit's rows "
    "(build_fold_designs / inner_fold_designs / the full-train engineer). Stage1MLPClassifier.fit then "
    "draws its early-stopping subset from those already-encoded rows: train_test_split(test_size="
    "val_fraction, stratify=y, random_state), with oversampling after the split. The subset's labels "
    "were therefore among the encoders' fit labels: they enter the encodings of the optimization rows "
    "(the rows the network is trained on) and of the validation rows themselves - each validation "
    "row's own label and the other validation rows' labels.")

MLP_ISOLATED_SPLIT = (
    "A properly isolated internal validation split excludes the entire early-stopping subset from "
    "encoder fitting: the encoders (and scaler) are fitted on the optimization rows only, so no "
    "subset label enters any encoding, of optimization or of validation rows. The accepted fits do "
    "not do this. The effects on the stopping epoch, the predictions and the AUC are not quantified "
    "here.")

MLP_OUTER_ISOLATION = (
    "Separately, the outer held-out rows are isolated: they are excluded from the encoders, the "
    "scaler, the network fit and its early-stopping subset, so the tuning-fold scores, the OOF scores "
    "and the Stage 4 column are scored on encoder-held-out rows (isolation checks: outer-fold flips, "
    "call sites, accepted fits).")


def mlp_early_stopping(X: pd.DataFrame, y: pd.Series, paths: Mapping[str, List[FitPath]],
                       *, verify: bool = True) -> Dict[str, Any]:
    """
    The black-box MLP's early-stopping subset against the Stage 1 encoders, on
    every Stage 1 fit (final, 10 outer = tuning and calibration-outer designs,
    100 inner calibration designs): how much of the optimization rows' and the
    validation rows' encodings comes from the subset's labels
    (``MLP_EARLY_STOPPING_ORDER``; an isolated split would make it zero,
    ``MLP_ISOLATED_SPLIT``).

    The split is reproduced exactly as ``Stage1MLPClassifier.fit`` draws it
    (accepted ``Stage1MLPConfig`` val_fraction and seed); with ``verify``, a
    one-epoch fit on the final and outer fold 0 rows confirms it: the
    estimator's recorded validation ROC-AUC equals the AUC on the reproduced
    validation rows. The attribution reuses ``attribute`` with the two sides of
    the split in place of contact days.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from blackbox_pipeline.models.mlp.config import Stage1MLPConfig
    from blackbox_pipeline.models.mlp.estimator import Stage1MLPClassifier
    from blackbox_pipeline.models.mlp.features import STAGE1_FEATURES as MLP_FEATURES
    from shared.stage1.feature_engineering import LRFeatureEngineer

    cfg = Stage1MLPConfig()
    frac, seed = cfg.val_fraction, cfg.random_state
    yv = np.asarray(y).astype(int)
    nxt = np.array([2, 2, 2], dtype=np.int64)       # side 0 = optimization rows, 1 = validation
    rows, checks = [], []
    for p in paths["stage1"]:
        n = len(p.fit)
        _, val = train_test_split(np.arange(n), test_size=frac, stratify=yv[p.fit], random_state=seed)
        side = np.zeros(n, dtype=np.int64)
        side[val] = 1
        fitted = _stage1_fit(X, yv, FitPath(p.stage, p.path, p.fold, p.inner, p.fit, p.fit, in_sample=True))
        if verify and p.path in ("final", "outer") and p.fold in (None, 0):
            eng = _quiet(LRFeatureEngineer().fit, X.iloc[p.fit], pd.Series(yv[p.fit], index=X.index[p.fit]))
            Z = StandardScaler().fit_transform(_quiet(eng.transform, X.iloc[p.fit])[list(MLP_FEATURES)])
            est = Stage1MLPClassifier(max_epochs=1, patience=1, val_fraction=frac,
                                      random_state=seed).fit(Z, yv[p.fit])
            mine = float(roc_auc_score(yv[p.fit][val], est.predict_proba(Z[val])[:, 1]))
            got = float(est.history_["val_roc_auc"].iloc[0])
            checks.append(_check(f"stage1 {p.key}: the reproduced early-stopping split is the "
                                 "estimator's (its recorded validation ROC-AUC on these rows)",
                                 abs(mine - got) <= 1e-12, f"{got:.12f} vs {mine:.12f}; "
                                 f"{len(val):,} validation of {n:,} rows"))
        for feat in STAGE1_FEATURES:
            res = attribute(fitted.blocks[(feat, "fit_rows")], yv[p.fit], side, side, nxt, np.arange(n))
            sh = _shares(res)
            v, o = side == 1, side == 0
            own = sh["self"]
            other = sh["direct_same_day"] + sh["prior_same_day"]            # other validation rows
            opt = sh["direct_earlier_day"] + sh["prior_earlier_day"]         # optimization rows
            to_opt = sh["direct_later_day_same_month"] + sh["prior_later_day_same_month"]
            carried = (res["rows"]["direct"]["later_day_same_month"]
                       + res["rows"]["prior"]["later_day_same_month"])
            rows.append({
                "feature": feat, "path": p.path, "fold": p.fold, "inner": p.inner, "fit_rows": n,
                "validation_rows": int(v.sum()),
                "validation_rows_with_own_label": int((res["self"][v] > 0).sum()),
                "validation_from_subset_share_sum": float((own + other)[v].sum()),
                "validation_own_share_sum": float(own[v].sum()),
                "validation_from_other_validation_share_sum": float(other[v].sum()),
                "validation_from_optimization_share_sum": float(opt[v].sum()),
                "optimization_rows": int(o.sum()),
                "optimization_rows_with_subset_labels": int((carried[o] > 0).sum()),
                "optimization_from_subset_share_sum": float(to_opt[o].sum()),
            })
    per_fit = pd.DataFrame(rows)
    g = per_fit.groupby(["feature", "path"], sort=False)
    summary = pd.DataFrame({
        "fits": g.size(), "validation_rows": g["validation_rows"].sum(),
        "validation_rows_with_own_label": g["validation_rows_with_own_label"].sum(),
        "optimization_rows": g["optimization_rows"].sum(),
        "optimization_rows_with_subset_labels": g["optimization_rows_with_subset_labels"].sum()})
    for col in ("validation_from_subset", "validation_own", "validation_from_other_validation",
                "validation_from_optimization"):
        summary[f"{col}_share_mean"] = g[f"{col}_share_sum"].sum() / summary["validation_rows"]
    summary["optimization_from_subset_share_mean"] = (g["optimization_from_subset_share_sum"].sum()
                                                      / summary["optimization_rows"])
    summary = summary.reset_index()
    return {"order": MLP_EARLY_STOPPING_ORDER, "isolated_split": MLP_ISOLATED_SPLIT,
            "outer_isolation": MLP_OUTER_ISOLATION, "val_fraction": frac, "random_state": seed,
            "summary": summary, "checks": pd.DataFrame(checks), "finding": _mlp_finding(summary, frac)}


def _mlp_finding(summary: pd.DataFrame, frac: float) -> str:
    """The early-stopping finding in words, from the measured shares."""
    def rng(col, digits=1):
        lo, hi = round(100 * summary[col].min(), digits), round(100 * summary[col].max(), digits)
        return f"{lo:.{digits}f}%" if lo == hi else f"{lo:.{digits}f}-{hi:.{digits}f}%"

    v, o = int(summary["validation_rows"].sum()), int(summary["optimization_rows"].sum())
    return (f"The early-stopping subset ({100 * frac:.0f}% of each MLP fit's rows) is not isolated "
            f"from the Stage 1 encoders. Its labels make up {rng('optimization_from_subset_share_mean')} "
            f"of the optimization rows' label mass and {rng('validation_from_subset_share_mean')} of the "
            f"validation rows' (own label {rng('validation_own_share_mean', 3)}, other validation rows "
            f"{rng('validation_from_other_validation_share_mean')}); "
            f"{int(summary['optimization_rows_with_subset_labels'].sum()):,}/{o:,} optimization-row "
            f"encodings carry subset labels and {int(summary['validation_rows_with_own_label'].sum()):,}/"
            f"{v:,} validation-row encodings carry their own label (row-feature encodings over "
            f"{int(summary.groupby('feature')['fits'].sum().iloc[0])} fits). "
            + MLP_ISOLATED_SPLIT + " " + MLP_OUTER_ISOLATION)


# ----------------------------------------------------------------------
# 6. provenance
# ----------------------------------------------------------------------
def provenance_table() -> pd.DataFrame:
    """
    Where each constant and frozen threshold of the audited encodings came from,
    separately from the labels a row's transformation uses (sections 3-4).
    """
    from feature_research import frozen_references as FR
    from shared.stage2.binning import BINNING_STRATEGY

    rec, meta = FR.read_record("c3_binning")
    fams = rec["decision"]["families"]
    revalidated = set(rec["scope"]["revalidated"])
    limitation = rec["limitations"][0]
    rows = []
    for fam in ("NSD", "ECI", "DOW"):
        spec = BINNING_STRATEGY[fam]
        status = (f"revalidated in PR 39 Step 3A, outcome {fams[fam]['outcome']}"
                  if fam in revalidated else "retained in PR 39 Step 3A without revalidation")
        rows.append({
            "item": f"Stage 2 {fam} thresholds on {spec['source']}",
            "value": "; ".join(f"{b[0]} ({b[1]}, {b[2]}]" for b in spec["bins"]),
            "used_by": "both routers' 29 locked bins: every Stage 2 fit, fold and the test mapping",
            "provenance": f"Cell 13C lift analysis (shared.stage2.binning); {status}. "
                          f"PR 39 limitation: {limitation}",
            "labels_used_when_a_row_is_transformed": ("none of its own: a fixed threshold on "
                                                      "the source value, whose label inputs are "
                                                      "the source's (attribution)"),
            "record": f"{meta['path']} ({meta['sidecar']}, sha256 {meta['sha256'][:16]})"})
    for item, used, prov in (
        ("Stage 1 n_bins = 20, smoothing k = 100",
         "subscribe_rate_by_euribor_bin, subscribe_rate_by_dow_month",
         "shared.stage1.feature_engineering constants ('mirrored from feature_research "
         "modules'); k 'tuned in Cell 10D' per feature_research.feature_engineering.temporal. "
         "The rows behind those choices are not recorded"),
        ("KDE bandwidth 1.0; 5,000 reference rows above 10,000 fit rows; seed 42",
         "neighborhood_subscription_density", "module constants; no tuning record"),
        ("curvature n_bins = 20", "economic_curvature_intensity", "module constant"),
        ("smoothing k = 100", "dow_month_encoded", "module constant ('tuned in Cell 10D')"),
    ):
        rows.append({"item": item, "value": "", "used_by": used, "provenance": prov,
                     "labels_used_when_a_row_is_transformed": "none (a setting)", "record": "code"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 7. findings
# ----------------------------------------------------------------------
_FAMILY = {"subscribe_rate_by_euribor_bin": "Stage 1 euribor bin rate (both arms)",
           "subscribe_rate_by_dow_month": "Stage 1 day x month rate (both arms)",
           "neighborhood_subscription_density": "NSD -> nsd_* bins (Stage 2), decay_x_density (Stage 3)",
           "economic_curvature_intensity": "ECI -> eci_* bins (Stage 2), ECI (Stage 3)",
           "dow_month_encoded": "day x month rate -> dow_* bins (Stage 2), Stage 3 feature"}


def findings_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One line per feature, stage and fitting path, with the three mechanisms kept
    apart and every count over its denominator:

    ``a_self``      fit-row encodings that carry the row's own label / fit-row encodings;
    ``b_held_out``  held-out encodings that carry the row's own label / held-out encodings
                    (a final fit has none: it maps the test split, whose labels it never fits);
    ``c_*``         on the held-out rows (fold paths) or the final mapping at the training
                    rows' contact days (final fits): rows with a label from a later month, from
                    any later day (direct or through a smoothing prior), the mean later share,
                    the rows whose label mass is mostly later, and the unresolved same-day share.
    """
    rows = []
    for (feat, stage, path), g in summary.groupby(["feature", "stage", "path"], sort=False):
        by = {r["role"]: r for _, r in g.iterrows()}
        fit, held = by["fit_rows"], by.get("held_out")
        c = held if held is not None else by["final_mapping"]
        q = int(c["rows"])
        rows.append({
            "family": _FAMILY.get(feat, feat), "feature": feat, "stage": stage, "path": path,
            "fits": int(fit["fits"]),
            "a_self": f"{int(fit['self_rows']):,}/{int(fit['rows']):,}",
            "a_self_share_mean": float(fit["self_share_mean"]),
            "b_held_out": (f"{int(held['self_rows']):,}/{int(held['rows']):,}" if held is not None
                           else "no held-out rows; no test label is fitted"),
            "c_rows": "held_out" if held is not None else "final_mapping",
            "c_later_month": f"{int(c['later_month_rows']):,}/{q:,}",
            "c_any_later_day": f"{int(c['any_later_rows']):,}/{q:,}",
            "c_later_share_mean": float(c["later_share_mean"]),
            "c_later_majority": f"{int(c['majority_later_rows']):,}/{q:,}",
            "same_day_share_mean": float(c["direct_same_day_share"] + c["prior_same_day_share"]),
            "comparisons": (f"passed {int(g['comparisons_passed'].sum())}, failed "
                            f"{int(g['comparisons_failed'].sum())}, skipped "
                            f"{int(g['comparisons_skipped'].sum())} of {int(g['fits'].sum())}"),
            "max_abs_diff": (float(g["max_abs_diff"].dropna().max())
                             if g["max_abs_diff"].notna().any() else None),
        })
    return pd.DataFrame(rows)


#: The held-out path that produces each stage's out-of-fold outputs.
PRIMARY_HELD_OUT = {"stage1": "outer", "stage2": "outer", "stage3": "tuning"}


def family_findings(summary: pd.DataFrame) -> pd.DataFrame:
    """
    The finding per feature family and stage: mechanisms (a), (b) and (c) on
    every fitting path, in counts with denominators.
    """
    def one(feat, stage, path, role):
        sel = summary[(summary["feature"] == feat) & (summary["stage"] == stage)
                      & (summary["path"] == path) & (summary["role"] == role)]
        return None if sel.empty else sel.iloc[0]

    def later(r):
        return (f"{r['later_month_rows']:,}/{r['rows']:,} rows draw labels from a later month and "
                f"{r['any_later_rows']:,}/{r['rows']:,} from a later day; mean later share "
                f"{r['later_share_mean']:.3f} (median {r['later_share_median']:.3f}, p90 "
                f"{r['later_share_p90']:.3f}); later-majority {r['majority_later_rows']:,}/{r['rows']:,}")

    rows = []
    for (feat, stage), g in summary.groupby(["feature", "stage"], sort=False):
        final = one(feat, stage, "final", "fit_rows")
        mapping = one(feat, stage, "final", "final_mapping")
        held = g[g["role"] == "held_out"]
        fits_rows = g[(g["role"] == "fit_rows") & (g["path"] != "final")]
        primary = one(feat, stage, PRIMARY_HELD_OUT[stage], "held_out")
        a = (f"final fit: {final['self_rows']:,}/{final['rows']:,} training rows carry their own "
             f"label (mean share {final['self_share_mean']:.1e}, max {final['self_share_max']:.1e}); "
             f"fold fits: {int(fits_rows['self_rows'].sum()):,}/{int(fits_rows['rows'].sum()):,} "
             "fit-row encodings")
        b = (f"measured {int(held['self_rows'].sum()):,}/{int(held['rows'].sum()):,} held-out "
             f"encodings carrying their own label over {int(held['fits'].sum())} fold fits "
             f"({', '.join(held['path'])}); comparisons passed "
             f"{int(held['comparisons_passed'].sum())}/{int(held['fits'].sum())}, max |diff| "
             f"{held['max_abs_diff'].dropna().max():.1e}")
        rows.append({
            "feature": feat, "family": _FAMILY.get(feat, feat), "stage": stage,
            "descendants": ", ".join(DESCENDANTS.get(feat, [])) or "none (consumed directly)",
            "a_self": a, "b_held_out": b,
            "c_out_of_fold": f"{PRIMARY_HELD_OUT[stage]} folds: {later(primary)}",
            "c_inner_folds": "; ".join(f"{r['path']}: mean later share {r['later_share_mean']:.3f}, "
                                       f"later-month rows {r['later_month_rows']:,}/{r['rows']:,}"
                                       for _, r in held.iterrows()
                                       if r["path"] != PRIMARY_HELD_OUT[stage]) or "none",
            "c_final_mapping": f"at the training rows' contact days: {later(mapping)}",
            "unresolved_same_day_share": round(float(primary["direct_same_day_share"]
                                                     + primary["prior_same_day_share"]), 4),
        })
    return pd.DataFrame(rows)
