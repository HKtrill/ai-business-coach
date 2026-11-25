"""
feature_research.temporal_validity
==================================
PR 42 Commit 3, Step 3B: past-only Stage 1 encodings, their predictive effect,
the feature-contract decision, the PR 49 temporal robustness protocol and the
interim temporal-validity finding. Training side only, on the Step 3A time
index (``encoding_audit.build_time_index``).

Sections
--------
1. Registration (``REGISTRATION``): what is known when each prediction is
   made, the candidate rules, cold start, unseen groups, update cadence, the
   chronological windows, the models, the comparisons and the contract
   criteria. Fixed in code before any result is computed; its digest is
   recorded beside the results.
2. Past-only encoders (``StateBank``): the production ``LRFeatureEngineer``
   fit, applied to the rows available before each contact day. State ``t`` of
   a context (a set of training positions whose labels may be used) is fitted
   on the context's rows contacted before day ``t``: the quantile edges, the
   bin rates, the median fallback, the cell rates and the global prior.
   ``verify_against_production`` fits the production encoder on exactly those
   rows and requires identical values.
3. Availability (``availability_table``, ``availability_checks``): rows
   encoded, cold start, euribor fallbacks, unseen day x month cells and the
   history behind each state.
4. Invariance (``invariance_checks``): flipping every label that is
   unavailable when a row is encoded (same day, later days, inside the lag,
   the same period under the monthly rule) never changes its encoding; flipping
   the available ones does (control). ``evaluate`` adds the same check on a
   deployed model's scores.
5. Predictive effect (``evaluate``, ``comparisons``, ``factorial``): accepted
   and past-only encodings with the accepted LR and MLP settings, under the
   accepted random 10-fold plan and under chronological windows, compared by
   paired DeLong on identical rows. The 2 x 2 factorial on the windows' rows
   contrasts the encoding protocols and the splits; each contrast is a
   difference of whole protocols, not of one change. The chronological lag arms
   are run both as a lag-0 model fed slower encodings and lag-consistent (model,
   scaler and encodings under the delay). The MLP is fitted with its accepted
   early stopping and with an isolated split.
6. Decisions (``contract_decisions``, ``candidate_contract``), the Stage 2-3
   extension (``stage23_extension``), the PR 49 protocol (``PR49_PROTOCOL``)
   and the interim finding (``interim_finding``).

Nothing here reads a test row, a test label or a model artifact. The frozen
PR 39 contracts are read, never changed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["REGISTRATION", "registration_digest", "RULES", "PRIMARY", "chronological_windows",
           "known_at_cutoff",
           "segments", "window_table", "availability_tables", "verification", "invariance_contexts",
           "StateBank", "rule_states", "verify_against_production",
           "availability_table", "availability_by_period", "availability_checks",
           "invariance_checks", "accepted_settings", "evaluate", "delong", "delong_paired",
           "COMPARISONS", "comparisons", "factorial", "protocol_table", "reproduction_checks",
           "early_stopping_table", "candidate_contract", "contract_decisions",
           "stage23_extension", "accepted_routing_counts", "stage2_prevalence_projection",
           "PR49_PROTOCOL", "interim_finding", "STEP3A_RECORD",
           "STEP3A_NAMES", "STEP3A_ABSENT", "ENVIRONMENT_KEYS", "step3a_reproduction", "step3a_checks",
           "require"]

BANK_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# ----------------------------------------------------------------------
# 1. registration
# ----------------------------------------------------------------------
#: Contact days, from the start of the timeline, before any past-only encoding exists.
COLD_START_DAYS = 3
#: Every state a past-only rule uses must hold at least this much history (checked, not tuned).
MIN_STATE_ROWS = 500
MIN_STATE_POSITIVES = 10
#: Chronological windows: periods kept for training only, then windows of at least this many rows.
BURN_IN_PERIODS = 4
MIN_WINDOW_ROWS = 1500
#: The two target encodings (the third Stage 1 feature, cellular_downturn, is label-free).
TARGET_FEATURES = ("subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month")
FEATURES = ("cellular_downturn",) + TARGET_FEATURES

#: Past-only rules. ``lag`` counts contact days; ``monthly`` refits at each period start.
RULES = {
    "past_daily_lag0": {"cadence": "daily", "lag": 0, "role": "primary candidate"},
    "past_daily_lag5": {"cadence": "daily", "lag": 5, "role": "outcome-lag sensitivity"},
    "past_daily_lag20": {"cadence": "daily", "lag": 20, "role": "outcome-lag sensitivity"},
    "past_monthly": {"cadence": "monthly", "lag": None, "role": "outcome-lag sensitivity"},
}
PRIMARY = "past_daily_lag0"

REGISTRATION = {
    "id": "pr42-step3b-registration",
    "version": "1.1.0",
    "revisions": [
        {"version": "1.0.0", "sha256": "88c004c31d31d05bc3b329bab5a5e65264fd99c43f2f4e573bb6bfee9baa4c31",
         "status": "fixed before any result was computed"},
        {"version": "1.1.0", "after": "review of the 1.0.0 results",
         "changes": [
             "adds the lag-consistent chronological arm (model, scaler and training rows under the "
             "rule's outcome delay) and its invariance regression; the 1.0.0 lag arms are kept and "
             "described as what they measure (a lag-0 model fed slower encodings at prediction)",
             "names every comparison and factorial contrast as a difference between protocols: the "
             "past-only protocol changes several things at once and no contrast isolates one",
             "the Step 3A record check: integrity always, exact reproduction only in the matching "
             "environment, enforced before the record is written",
         ],
         "unchanged": "every 1.0.0 comparison's protocols, rows and computation"},
    ],
    "question": ("Do past-only Stage 1 target encodings, which use only labels known when a row is "
                 "scored, change predictive performance, and how much of any difference comes "
                 "from the encoding rather than from the evaluation split?"),
    "time_unit": "contact day of the Step 3A time index (485 days, 26 periods, training rows)",
    "known_at_contact": [
        "the row's own covariates (euribor3m, emp.var.rate, nr.employed, contact, day_of_week, "
        "month), its contact day and its period",
        "the covariates and labels of the context's rows that the rule makes available",
        "never: its own label, the labels of same-day contacts, the labels of later contacts",
    ],
    "rules": {
        "accepted": ("the production LRFeatureEngineer fitted once on all of a fit's rows (a random "
                     "fold, a chronological training window, the full split): the edges, bin "
                     "rates, median fallback, cell rates and global prior use the labels of "
                     "earlier, same-day and later contacts (Step 3A mechanisms a and c)"),
        "past_daily_lagL": ("refitted before each contact day d on the context's rows contacted on "
                            "days < d - L: the qcut edges (from those rows' euribor3m), the bin "
                            "rates, the median fallback, the day x month cell rates and the global "
                            "prior all come from those rows and nothing from day d - L or later; "
                            "L = 0 (primary), 5 and 20"),
        "past_monthly": ("refitted at the first contact day of each period on the context's rows of "
                         "earlier periods; every row of the period uses that fit"),
        "roles": {k: v["role"] for k, v in RULES.items()},
    },
    "same_day": ("never used: contacts on one day are simultaneous for this purpose (the data "
                 "record no time of day, so the order within a day is not known)"),
    "outcome_timing": ("unknown. Each row is a client's last contact of the campaign and its label "
                       "the campaign outcome; when that outcome became known is not recorded. Lag 0 "
                       "assumes it is known by the next contact day; lags 5 and 20 contact days "
                       "(about a week and a month) and the monthly rule are the sensitivities"),
    "cold_start": (f"rows contacted on the first {COLD_START_DAYS} contact days (plus the lag), or in "
                   "the first period under the monthly rule, have no past-only encoding. They are "
                   "left out of that rule's model fits and of every comparison that includes the "
                   f"rule; every state used holds at least {MIN_STATE_ROWS} rows and "
                   f"{MIN_STATE_POSITIVES} positives (checked)"),
    "unseen": {"euribor": ("a value outside the available rows' bins, or in an empty bin: the "
                           "median of the available bin rates (the production fallback)"),
               "cell": "a day x month cell absent from the available rows: the available global rate",
               "pooled_years": ("cells pool years (as in production): a cell's rate can be a year "
                                "old when the month recurs")},
    "chronological_windows": (f"the first {BURN_IN_PERIODS} periods (2008-05..2008-08) train only; "
                              f"later periods form consecutive windows of at least "
                              f"{MIN_WINDOW_ROWS:,} training rows, a remainder joining the last "
                              "window. Each window is predicted by models fitted on every earlier "
                              "training row and frozen at the window's first day"),
    "chronological_encodings": {
        "accepted_window": ("the production encoder fitted on the training rows (all their labels), "
                            "applied in-sample to them and to the predicted window"),
        "past_frozen": ("training rows encoded by the primary rule; predicted rows by the state at "
                        "the cutoff, which is fitted on every training row (so the predicted "
                        "encodings equal accepted_window's, checked)"),
        "past_updating": ("training rows as past_frozen; predicted rows by the rule's state at their "
                          "own contact day, so earlier predicted-window labels enter as outcomes "
                          "arrive; models stay frozen. Run for every rule with the same lag-0 model, "
                          "trained on every pre-cutoff outcome: for lag 5, lag 20 and monthly it "
                          "measures a lag-0 model whose incoming encodings update more slowly, not "
                          "operation under that delay"),
        "lag_consistent": ("operation under the rule's outcome delay throughout: the scaler and the "
                           "model are fitted on the training rows whose outcome is known at the "
                           "cutoff (contacted before cutoff - L; before the cutoff for the monthly "
                           "rule) and that the rule encodes, with the rule's encodings; predicted "
                           "rows are encoded by the rule's state at their own contact day. At lag 0 "
                           "it equals past_updating (checked)"),
    },
    "models": {
        "lr": ("accepted GLASS LR: LogisticRegression(saga, max_iter 5000, class_weight balanced, "
               "seed 42) with the PR 39 Commit 2 best parameters, on a StandardScaler of its fit "
               "rows; P(y = 1), as the accepted tuning and OOF scores"),
        "mlp": ("accepted black-box MLP: Stage1MLPClassifier with the PR 39 Commit 2 best parameters "
                "and fixed kwargs (200 epochs, patience 15, val_fraction 0.15, seed 42), on a "
                "StandardScaler of its fit rows; P(y = 1). Primary rule only"),
    },
    "mlp_early_stopping": {
        "accepted": ("the estimator draws its early-stopping subset from rows whose encodings (and "
                     "scaler) were fitted with that subset's labels (Step 3A)"),
        "isolated": ("the estimator's own split of the rows it will receive is drawn first "
                     "(train_test_split, stratified, val_fraction, seed 42); the encoders and the "
                     "scaler are fitted without the subset; every isolated fit checks that the "
                     "estimator's recorded best validation ROC-AUC equals the ROC-AUC on the "
                     "reproduced subset"),
    },
    "metrics": ["ROC-AUC", "average precision", "prevalence"],
    "paired_comparison": ("DeLong (midranks) on identical rows, 95% normal interval; pooled over the "
                          "rows and per segment (burn-in, W1..Wn); the factorial also reports the "
                          "unweighted mean over windows with independent-window standard errors"),
    "factorial": ("on the windows' rows: random/accepted (A), random/past-only (B), chrono/accepted "
                  "(C), chrono/past-only updating (D). Protocol differences: past-only vs accepted "
                  "B - A (random) and D - C (chrono); chronological vs random C - A (accepted) and "
                  "D - B (past-only); interaction (D - C) - (B - A). The past-only protocol removes "
                  "same-day and later labels, removes each training row's own label from its "
                  "encoding, fits edges, rates and fallbacks on fewer rows and leaves the cold-start "
                  "rows out of its fits; a contrast is the difference of whole protocols and does not "
                  "isolate any one of these"),
    "contract_criteria": {
        "realizable": "no same-day, later or lagged-out label reaches any row's encoding (invariance)",
        "differs": "its values differ from the accepted encoding's on at least 1% of the rows",
        "available": "the encoding exists for at least 95% of the training rows",
        "rule": ("a realizable rule whose values differ is a new feature definition and needs its "
                 "own versioned contract with distinct feature names before any artifact carries it. "
                 "The primary rule, if available, goes to PR 44 as the candidate; the sensitivity "
                 "rules go to PR 49. Predictive results are recorded beside the decision; adoption "
                 "is PR 44's (Stage 1) and PR 45's (Stages 2-3)"),
    },
}


def registration_digest(registration: Mapping[str, Any] = REGISTRATION) -> str:
    """SHA-256 of the canonical registration JSON (recorded beside the results)."""
    return hashlib.sha256(json.dumps(registration, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _check(name: str, passed: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def require(checks: pd.DataFrame, what: str) -> str:
    """Raise if any check failed; otherwise a one-line summary."""
    bad = checks[~checks["passed"].astype(bool)]
    if len(bad):
        raise AssertionError(f"{what}: {len(bad)} check(s) failed\n{bad.to_string()}")
    return f"{what}: {len(checks)} checks passed"


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def chronological_windows(T: pd.DataFrame) -> List[Dict[str, Any]]:
    """The registered windows (name, periods, period indices, first day, rows), from the time index."""
    per = (T.groupby(["period_index", "period"]).agg(rows=("day", "size"), first_day=("day", "min"))
           .reset_index().sort_values("period_index"))
    groups, cur = [], []
    for _, r in per.iloc[BURN_IN_PERIODS:].iterrows():
        cur.append(r)
        if sum(int(x["rows"]) for x in cur) >= MIN_WINDOW_ROWS:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1] = groups[-1] + cur
        else:
            groups.append(cur)
    return [{"window": f"W{i + 1}", "periods": [str(x["period"]) for x in g],
             "period_indices": [int(x["period_index"]) for x in g],
             "first_day": int(g[0]["first_day"]), "rows": int(sum(int(x["rows"]) for x in g))}
            for i, g in enumerate(groups)]


def segments(T: pd.DataFrame, windows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Per training row: 'burn-in' or the window it belongs to."""
    pidx = T["period_index"].to_numpy()
    out = np.full(len(T), "burn-in", dtype=object)
    for w in windows:
        out[np.isin(pidx, w["period_indices"])] = w["window"]
    return out


def window_table(T: pd.DataFrame, y) -> pd.DataFrame:
    """The burn-in and each window: periods, contact days, rows, positives and prevalence."""
    yv = np.asarray(y).astype(int)
    windows = chronological_windows(T)
    seg = segments(T, windows)
    day = T["day"].to_numpy()
    out = []
    for name in ["burn-in"] + [w["window"] for w in windows]:
        m = seg == name
        per = sorted(set(T["period"].to_numpy()[m]))
        out.append({"segment": name, "periods": f"{per[0]}..{per[-1]}", "period_count": len(per),
                    "first_day": int(day[m].min()), "last_day": int(day[m].max()),
                    "rows": int(m.sum()), "positives": int(yv[m].sum()), "prevalence": float(yv[m].mean()),
                    "training_rows_before": int((day < day[m].min()).sum())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 2. past-only encoders
# ----------------------------------------------------------------------
@dataclass
class _State:
    """One fitted state: the production encoder's fitted quantities, as arrays."""

    rows: int
    positives: int
    categories: Any            # pandas IntervalIndex (the qcut categories)
    rate: np.ndarray           # rate per category, NaN for an empty one
    median: float              # median of the observed rates (the fallback)
    cells: np.ndarray          # sorted cell keys day_of_week * 100 + month
    smoothed: np.ndarray       # smoothed rate per cell
    global_rate: float


def _cells(X: pd.DataFrame, rows=None) -> np.ndarray:
    dow = X["day_of_week"].to_numpy().astype(int)
    month = X["month"].to_numpy().astype(int)
    key = dow * 100 + month
    return key if rows is None else key[rows]


def _fit_state(eur: np.ndarray, yv: np.ndarray, cell: np.ndarray, n_bins: int, k: float) -> _State:
    """
    ``LRFeatureEngineer.fit`` on these rows, as arrays: the same pandas qcut (the
    cut fallback if it fails) for the categories, per-category means over the
    observed categories, the median fallback, the global mean and the smoothed
    cells, in the same floating-point operations (verified, not assumed).
    """
    s = pd.Series(eur)
    try:
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(s, bins=n_bins, duplicates="drop")
    cats = bins.cat.categories
    codes = bins.cat.codes.to_numpy()
    if len(cats) == 0 or (codes < 0).any():
        raise ValueError("a state's euribor3m values must fall in its own bins (one contact day "
                         "carries one value, so a state needs more than one day)")
    cnt = np.bincount(codes, minlength=len(cats)).astype(float)
    pos = np.bincount(codes, weights=yv, minlength=len(cats))
    rate = np.full(len(cats), np.nan)
    obs = cnt > 0
    rate[obs] = pos[obs] / cnt[obs]
    g = float(yv.mean())
    keys, inv = np.unique(cell, return_inverse=True)
    inv = np.asarray(inv).ravel()
    c_cnt = np.bincount(inv, minlength=len(keys)).astype(float)
    c_pos = np.bincount(inv, weights=yv, minlength=len(keys))
    smoothed = (c_cnt * (c_pos / c_cnt) + k * g) / (c_cnt + k)
    return _State(len(yv), int(yv.sum()), cats, rate, float(np.median(rate[obs])), keys, smoothed, g)


def _apply_state(st: _State, eur: np.ndarray, cell: np.ndarray):
    """Both encodings of some rows under one state, and their fallback flags."""
    code = pd.cut(pd.Series(eur), bins=st.categories, include_lowest=True).cat.codes.to_numpy()
    hit = np.zeros(len(eur), dtype=bool)
    ok = code >= 0
    hit[ok] = ~np.isnan(st.rate[code[ok]])
    e = np.full(len(eur), st.median)
    e[hit] = st.rate[code[hit]]
    pos = np.searchsorted(st.cells, cell)
    seen = (pos < len(st.cells)) & (st.cells[np.clip(pos, 0, len(st.cells) - 1)] == cell)
    d = np.full(len(cell), st.global_rate)
    d[seen] = st.smoothed[pos[seen]]
    return e, d, ~hit, ~seen


class StateBank:
    """
    Past-only encoder states of one context: state ``t`` is fitted on the
    context's rows contacted before contact day ``t``. Built on demand, cached
    by the number of rows they hold.
    """

    def __init__(self, X: pd.DataFrame, y, day: np.ndarray, context, *, n_bins: int = 20,
                 k: float = 100.0):
        self.n_bins, self.k = n_bins, float(k)
        ctx = np.asarray(context, dtype=np.int64)
        self.context = ctx[np.argsort(day[ctx], kind="stable")]
        self.days = day[self.context]
        self.eur = X["euribor3m"].to_numpy(dtype=float)[self.context]
        self.yv = np.asarray(y, dtype=float)[self.context]
        self.cell = _cells(X, self.context)
        self._cache: Dict[int, Optional[_State]] = {}

    def size_before(self, t: int) -> int:
        return int(np.searchsorted(self.days, t, "left"))

    def state(self, t: int) -> Optional[_State]:
        """The state fitted on the context's rows with day < t (None when there are none)."""
        n = self.size_before(t)
        if n not in self._cache:
            self._cache[n] = (None if n == 0 else
                              _fit_state(self.eur[:n], self.yv[:n], self.cell[:n], self.n_bins, self.k))
        return self._cache[n]


def _period_first_day(T: pd.DataFrame) -> np.ndarray:
    return T.groupby("period_index")["day"].transform("min").to_numpy()


def _state_day(rule: str, day: np.ndarray, first_day: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: the state its rule uses, and whether the rule encodes the row at all."""
    spec = RULES[rule]
    if spec["cadence"] == "daily":
        t = day - spec["lag"]
        return np.maximum(t, 0), t >= COLD_START_DAYS
    return first_day, first_day > 0


def rule_states(bank: StateBank, X: pd.DataFrame, T: pd.DataFrame, rule: str,
                query) -> Dict[str, np.ndarray]:
    """
    The two past-only encodings of the ``query`` rows (training positions) under
    ``rule`` with ``bank``'s states, with availability, fallback flags and the
    history behind each. Rows the rule cannot encode are NaN.
    """
    query = np.asarray(query, dtype=np.int64)
    t, avail = _state_day(rule, T["day"].to_numpy()[query], _period_first_day(T)[query])
    eur = X["euribor3m"].to_numpy(dtype=float)[query]
    cell = _cells(X, query)
    n = len(query)
    out = {TARGET_FEATURES[0]: np.full(n, np.nan), TARGET_FEATURES[1]: np.full(n, np.nan),
           "available": avail.copy(), "euribor_fallback": np.zeros(n, dtype=bool),
           "cell_unseen": np.zeros(n, dtype=bool), "history_rows": np.zeros(n, dtype=np.int64),
           "history_positives": np.zeros(n, dtype=np.int64), "history_bins": np.zeros(n, dtype=np.int64),
           "state_day": t}
    for tt in np.unique(t[avail]):
        m = avail & (t == tt)
        st = bank.state(int(tt))
        if st is None:
            out["available"][m] = False
            continue
        e, d, fb, un = _apply_state(st, eur[m], cell[m])
        out[TARGET_FEATURES[0]][m], out[TARGET_FEATURES[1]][m] = e, d
        out["euribor_fallback"][m], out["cell_unseen"][m] = fb, un
        out["history_rows"][m], out["history_positives"][m] = st.rows, st.positives
        out["history_bins"][m] = len(st.categories)
    return out


def _merge(store: Dict[str, Dict[str, np.ndarray]], key: str, rows, enc: Mapping[str, np.ndarray]):
    """Append one fold's encodings of ``rows`` to ``store[key]`` (kept in row order at the end)."""
    cur = store.setdefault(key, {"rows": np.empty(0, dtype=np.int64)})
    cur["rows"] = np.r_[cur["rows"], np.asarray(rows, dtype=np.int64)]
    for f, v in enc.items():
        if f == "state_day":
            continue
        cur[f] = np.r_[cur[f], v] if f in cur else np.asarray(v)


def _sorted(store):
    for v in store.values():
        order = np.argsort(v["rows"], kind="stable")
        for f in list(v):
            v[f] = v[f][order]
    return store


def verify_against_production(X: pd.DataFrame, y, T: pd.DataFrame, bank: StateBank, *,
                              rule: str = PRIMARY, days: Optional[Sequence[int]] = None,
                              label: str = "") -> Dict[str, Any]:
    """
    For every state ``rule`` uses for the rows contacted on ``days`` (all days
    by default): fit the production ``LRFeatureEngineer`` on exactly that
    state's rows and transform the rows that use it. Both encodings must equal
    the state's, exactly, and be finite. Covers the lagged and monthly uses
    (a state applied to rows of later days, where the fallbacks occur).
    """
    from shared.stage1.feature_engineering import LRFeatureEngineer

    yv = pd.Series(np.asarray(y).astype(int), index=X.index)
    day = T["day"].to_numpy()
    q_all = np.flatnonzero(np.isin(day, np.unique(day) if days is None else np.asarray(days)))
    t, avail = _state_day(rule, day[q_all], _period_first_day(T)[q_all])
    eur = X["euribor3m"].to_numpy(dtype=float)
    cells = _cells(X)
    compared, rows_compared, fallbacks, worst, failures = 0, 0, 0, 0.0, []
    for tt in np.unique(t[avail]):
        n = bank.size_before(int(tt))
        if n == 0:
            continue
        q = q_all[avail & (t == tt)]
        fit = bank.context[:n]
        eng = _quiet(LRFeatureEngineer().fit, X.iloc[fit], yv.iloc[fit])
        prod = _quiet(eng.transform, X.iloc[q])
        e, dd, fb, un = _apply_state(bank.state(int(tt)), eur[q], cells[q])
        a = prod[TARGET_FEATURES[0]].to_numpy(float)
        b = prod[TARGET_FEATURES[1]].to_numpy(float)
        finite = all(np.isfinite(v).all() for v in (a, b, e, dd))
        diff = max(float(np.max(np.abs(a - e))), float(np.max(np.abs(b - dd)))) if finite else float("inf")
        worst = max(worst, diff)
        compared += 1
        rows_compared += len(q)
        fallbacks += int(fb.sum() + un.sum())
        if diff != 0.0:
            failures.append(int(tt))
    return {"context": label, "rule": rule, "context_rows": int(len(bank.context)),
            "states_compared": compared, "rows_compared": rows_compared,
            "fallback_values_compared": fallbacks, "max_abs_diff": worst,
            "failed_states": failures[:10], "passed": compared > 0 and not failures}


# ----------------------------------------------------------------------
# 3. availability
# ----------------------------------------------------------------------
def availability_table(encodings: Mapping[str, Mapping[str, np.ndarray]],
                       accepted: Optional[Mapping[str, np.ndarray]] = None, *,
                       scope: str = "") -> pd.DataFrame:
    """Rows encoded, cold start, fallbacks, unseen cells and history, per rule (over the rows given)."""
    rows = []
    for rule, enc in encodings.items():
        a = enc["available"].astype(bool)
        n, k = len(a), int(a.sum())
        row = {"scope": scope, "rule": rule, "rows": n, "available": k,
               "available_share": k / n if n else float("nan"), "cold_start_rows": n - k,
               "euribor_fallback_rows": int(enc["euribor_fallback"][a].sum()),
               "cell_unseen_rows": int(enc["cell_unseen"][a].sum()),
               "min_history_rows": int(enc["history_rows"][a].min()) if k else 0,
               "min_history_positives": int(enc["history_positives"][a].min()) if k else 0,
               "median_history_rows": float(np.median(enc["history_rows"][a])) if k else 0.0,
               "min_bins": int(enc["history_bins"][a].min()) if k else 0,
               "median_bins": float(np.median(enc["history_bins"][a])) if k else 0.0}
        if accepted is not None and k:
            for f in TARGET_FEATURES:
                diff = np.abs(enc[f][a] - accepted[f][a])
                row[f"{f}_mean_abs_diff"] = float(diff.mean())
                row[f"{f}_rows_differ"] = int((diff > 1e-12).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def availability_by_period(enc: Mapping[str, np.ndarray], T: pd.DataFrame, rows) -> pd.DataFrame:
    """One rule's availability and fallbacks by period (``rows`` = the encoded positions)."""
    a = enc["available"].astype(bool)
    df = pd.DataFrame({"period": T["period"].to_numpy()[rows], "available": a,
                       "euribor_fallback": enc["euribor_fallback"] & a,
                       "cell_unseen": enc["cell_unseen"] & a})
    return (df.groupby("period").agg(rows=("available", "size"), available=("available", "sum"),
                                     euribor_fallback=("euribor_fallback", "sum"),
                                     cell_unseen=("cell_unseen", "sum")).reset_index())


def availability_checks(tables: pd.DataFrame) -> pd.DataFrame:
    """Every state used holds the registered minimum history; cold start is the registered rule."""
    out = []
    for _, r in tables.iterrows():
        out.append(_check(f"{r['scope']} {r['rule']}: every state used holds >= {MIN_STATE_ROWS} rows "
                          f"and >= {MIN_STATE_POSITIVES} positives",
                          r["min_history_rows"] >= MIN_STATE_ROWS
                          and r["min_history_positives"] >= MIN_STATE_POSITIVES,
                          f"min {r['min_history_rows']:,} rows, {r['min_history_positives']} positives"))
    return pd.DataFrame(out)


#: The availability scopes: (scope, encodings key per rule, accepted key) in ``evaluate``'s store.
_SCOPES = (
    ("training rows, full history", "all|{rule}", "all|accepted"),
    ("random held-out rows", "random|{rule}", "random|accepted"),
    ("chronological predicted rows (updating)", "chrono|past_updating|{rule}", "chrono|past_frozen"),
)


def availability_tables(ev: Mapping[str, Any]) -> pd.DataFrame:
    """
    Availability per rule and scope: every training row with the full history
    (deployment-like), the random-split held-out rows with their fold's history
    (the contract criteria), and the chronological predicted rows; the
    accepted encodings of the same rows for the value differences (for the
    predicted rows, the frozen cutoff state, which equals the accepted window
    fit). The frozen state's own row closes the table.
    """
    E = ev["encodings"]
    tables = []
    for scope, key, acc in _SCOPES:
        encs = {rule: E[key.format(rule=rule)] for rule in RULES if key.format(rule=rule) in E}
        if encs:
            tables.append(availability_table(encs, E.get(acc), scope=scope))
    if "chrono|past_frozen" in E:
        tables.append(availability_table({"past_frozen (cutoff state)": E["chrono|past_frozen"]},
                                         scope="chronological predicted rows (frozen)"))
    return pd.concat(tables, ignore_index=True)


def verification(X: pd.DataFrame, y, T: pd.DataFrame, ev: Mapping[str, Any], *,
                 sample_days: int = 60) -> pd.DataFrame:
    """
    ``verify_against_production`` for every rule on the full-history context
    (every day), the primary rule on two random-fold contexts (every day) and,
    on a sample of days, the lagged rules on a fold context and the isolated
    early-stopping contexts the MLP used.
    """
    day = np.unique(T["day"].to_numpy())
    sample = day[np.linspace(COLD_START_DAYS, len(day) - 1, sample_days).astype(int)]
    B = ev["banks"]
    runs = [("all", rule, None) for rule in RULES]
    runs += [(f"fold{k}", PRIMARY, None) for k in (0, 5) if f"fold{k}" in B]
    runs += [("fold3", rule, sample) for rule in RULES if rule != PRIMARY and "fold3" in B]
    runs += [(name, PRIMARY, sample) for name in B if name.endswith("_es_isolated")][:4]
    rows = [verify_against_production(X, y, T, B[name], rule=rule, days=days, label=name)
            for name, rule, days in runs]
    out = pd.DataFrame(rows)
    out["days"] = ["all" if d is None else f"{len(d)} sampled" for _, _, d in runs]
    out["failed_states"] = out["failed_states"].map(lambda v: ", ".join(map(str, v)) or "none")
    return out


def invariance_contexts(ev: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """The contexts the invariance checks flip labels in: full history, a random fold, an isolated one."""
    B = ev["banks"]
    names = ["all", "fold0"] + [n for n in B if n.endswith("_es_isolated")][:1]
    return {n: B[n].context for n in names if n in B}


# ----------------------------------------------------------------------
# 4. invariance
# ----------------------------------------------------------------------
def _unavailable(rule: str, d: int, first_day: int, day: np.ndarray) -> np.ndarray:
    """Rows whose labels are unavailable to an encoding made on day d under ``rule``."""
    spec = RULES[rule]
    if spec["cadence"] == "daily":
        return day >= d - spec["lag"]
    return day >= first_day


def invariance_checks(X: pd.DataFrame, y, T: pd.DataFrame, contexts: Mapping[str, Any],
                      *, probes: int = 40) -> pd.DataFrame:
    """
    For each context and rule, at ``probes`` contact days spread over the
    timeline: flipping every label unavailable at that day leaves the day's
    encodings unchanged; flipping only the same day's labels leaves them
    unchanged; flipping the available labels changes them (control: the check
    can fail). For contrast, the accepted rule (the production encoder fitted
    on the whole context): flipping the same day's and later labels changes
    the day's encodings (the look-ahead the past-only rules remove).
    """
    from shared.stage1.feature_engineering import LRFeatureEngineer

    yv = np.asarray(y).astype(int)
    day = T["day"].to_numpy()
    first = _period_first_day(T)
    days = np.unique(day)
    picks = days[np.linspace(COLD_START_DAYS + 25, len(days) - 1, probes).astype(int)]
    out = []
    for cname, ctx in contexts.items():
        for rule in RULES:
            same_un, same_day, moved, done = 0, 0, 0, 0
            for d in picks:
                q = np.flatnonzero(day == d)
                encode = lambda labels: rule_states(StateBank(X, labels, day, ctx), X, T, rule, q)  # noqa: E731
                base = encode(yv)
                a = base["available"]
                if not a.any():
                    continue
                done += 1
                un = _unavailable(rule, int(d), int(first[q[0]]), day)
                flips = []
                for mask in (un, day == d, ~un):
                    y2 = yv.copy()
                    y2[mask] = 1 - y2[mask]
                    f2 = encode(y2)
                    flips.append(all(np.array_equal(base[f][a], f2[f][a]) for f in TARGET_FEATURES))
                same_un += flips[0]
                same_day += flips[1]
                moved += not flips[2]
            out.append(_check(f"{cname} {rule}: flipping every unavailable label (same day, later, "
                              "inside the lag or the period) leaves the encodings unchanged",
                              done > 0 and same_un == done, f"{same_un}/{done} probe days"))
            out.append(_check(f"{cname} {rule}: flipping the same day's labels alone leaves them "
                              "unchanged", done > 0 and same_day == done, f"{same_day}/{done} probe days"))
            out.append(_check(f"{cname} {rule}: flipping the available labels changes them (control)",
                              done > 0 and moved == done, f"{moved}/{done} probe days"))
        ctx = np.asarray(ctx, dtype=np.int64)
        looked, done = 0, 0
        for d in picks:
            q = np.flatnonzero(day == d)
            y2 = yv.copy()
            y2[day >= d] = 1 - y2[day >= d]
            encs = []
            for labels in (yv, y2):
                eng = _quiet(LRFeatureEngineer().fit, X.iloc[ctx], pd.Series(labels[ctx], index=X.index[ctx]))
                encs.append(_quiet(eng.transform, X.iloc[q]))
            done += 1
            looked += any(not np.array_equal(encs[0][f].to_numpy(float), encs[1][f].to_numpy(float))
                          for f in TARGET_FEATURES)
        out.append(_check(f"{cname} accepted rule (contrast): flipping the same day's and later labels "
                          "changes the day's encodings", done > 0 and looked == done,
                          f"{looked}/{done} probe days"))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 5. predictive effect
# ----------------------------------------------------------------------
STAGE1_FIT_RECORD = "research_logs/pr39/c2_20261114_184433/stage1_calibration_evidence.json"


def accepted_settings() -> Dict[str, Any]:
    """The accepted Stage 1 model settings, read from the PR 39 Commit 2 record (fingerprinted)."""
    from blackbox_pipeline.models.mlp.config import Stage1MLPConfig
    from blackbox_pipeline.models.mlp.features import STAGE1_FEATURES as MLP_FEATURES
    from blackbox_pipeline.models.mlp.tuning import params_to_kwargs
    from glass_pipeline.lr.penalty import penalty_kwargs
    from shared.stage1.feature_engineering import LR_FEATURES

    if tuple(LR_FEATURES) != FEATURES or tuple(MLP_FEATURES) != FEATURES:
        raise AssertionError(f"Stage 1 feature order drifted: {LR_FEATURES} / {MLP_FEATURES}")
    data = (BANK_ROOT / STAGE1_FIT_RECORD).read_bytes()
    rec = json.loads(data.decode("utf-8"))
    lr, mlp = rec["fits"]["glass_lr"], rec["fits"]["blackbox_mlp"]
    p = lr["best_params"]
    fixed = Stage1MLPConfig().estimator_fixed_kwargs
    return {"source": STAGE1_FIT_RECORD, "sha256": hashlib.sha256(data).hexdigest(),
            "lr_params": dict(p), "mlp_params": dict(mlp["best_params"]), "mlp_fixed": dict(fixed),
            "lr_kwargs": dict(C=p["C"], solver="saga", max_iter=5000, random_state=SEED,
                              class_weight="balanced", **penalty_kwargs(p["penalty"], p.get("l1_ratio"))),
            "mlp_kwargs": params_to_kwargs(mlp["best_params"], **fixed),
            "recorded_cv_roc_auc": {"lr": lr["best_cv_roc_auc"], "mlp": mlp["best_cv_roc_auc"]}}


def _matrix(cellular: np.ndarray, enc: Mapping[str, np.ndarray], rows) -> np.ndarray:
    return np.c_[cellular[rows], enc[TARGET_FEATURES[0]], enc[TARGET_FEATURES[1]]]


def _scaled(M_fit: np.ndarray, *others: np.ndarray, fit_on: Optional[np.ndarray] = None):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(M_fit if fit_on is None else fit_on)
    return [sc.transform(M_fit)] + [sc.transform(o) for o in others]


def _lr(settings, Z_fit, y_fit, Z_eval) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**settings["lr_kwargs"]).fit(Z_fit, y_fit).predict_proba(Z_eval)[:, 1]


def _es_split(y_fit: np.ndarray, settings) -> Tuple[np.ndarray, np.ndarray]:
    """The estimator's own early-stopping split of the rows it receives: (optimization, validation)."""
    from sklearn.model_selection import train_test_split
    k = settings["mlp_kwargs"]
    return train_test_split(np.arange(len(y_fit)), test_size=k["val_fraction"], stratify=y_fit,
                            random_state=k["random_state"])


def _mlp(settings, Z_fit, y_fit, Z_eval, *, validation: Optional[np.ndarray] = None):
    """Fit the accepted MLP; with ``validation``, confirm its early-stopping subset is those rows."""
    from sklearn.metrics import roc_auc_score
    from blackbox_pipeline.models.mlp.estimator import Stage1MLPClassifier

    est = Stage1MLPClassifier(**settings["mlp_kwargs"]).fit(Z_fit, y_fit)
    info = {"best_epoch": int(est.best_epoch_), "best_val_roc_auc": float(est.best_val_roc_auc_),
            "split_reproduced": None}
    if validation is not None:
        auc = float(roc_auc_score(y_fit[validation], est.predict_proba(Z_fit[validation])[:, 1]))
        info["split_reproduced"] = abs(auc - est.best_val_roc_auc_) <= 1e-12
    return est.predict_proba(Z_eval)[:, 1], info


def _score_summary(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s))}


def _accepted_encode(X, yv, fit_rows, *queries):
    """The production encoder (accepted rule) fitted on ``fit_rows``, applied to each query set."""
    from shared.stage1.feature_engineering import LRFeatureEngineer

    ys = pd.Series(np.asarray(yv).astype(int), index=X.index)
    eng = _quiet(LRFeatureEngineer().fit, X.iloc[fit_rows], ys.iloc[fit_rows])
    out = []
    for q in queries:
        p = _quiet(eng.transform, X.iloc[q])
        out.append({f: p[f].to_numpy(float) for f in TARGET_FEATURES})
    return out


def _frozen(bank: StateBank, X, t: int, rows) -> Dict[str, np.ndarray]:
    st = bank.state(t)
    e, d, fb, un = _apply_state(st, X["euribor3m"].to_numpy(dtype=float)[rows], _cells(X, rows))
    n = len(rows)
    return {TARGET_FEATURES[0]: e, TARGET_FEATURES[1]: d, "available": np.ones(n, dtype=bool),
            "euribor_fallback": fb, "cell_unseen": un, "history_rows": np.full(n, st.rows),
            "history_positives": np.full(n, st.positives), "history_bins": np.full(n, len(st.categories))}


class _Scores:
    def __init__(self):
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self.by_fold: Dict[str, List[float]] = {}

    def put(self, name, rows, s, y=None):
        cur = self.data.setdefault(name, {"rows": np.empty(0, dtype=np.int64), "scores": np.empty(0)})
        cur["rows"], cur["scores"] = np.r_[cur["rows"], rows], np.r_[cur["scores"], s]
        if y is not None:
            from sklearn.metrics import roc_auc_score
            self.by_fold.setdefault(name, []).append(float(roc_auc_score(y, s)))


def evaluate(X: pd.DataFrame, y, T: pd.DataFrame, *, settings=None, mlp: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
    """
    The registered predictive comparisons. Returns ``scores`` (protocol ->
    positions and scores), per-fold ROC-AUCs of the random-split protocols, the
    MLP fit notes, the encodings behind the availability tables, the state banks
    (for verification), the windows and the chronological checks.
    """
    from shared.stage1.crossfit import build_fold_designs
    from shared.stage1.feature_engineering import LRFeatureEngineer

    settings = settings or accepted_settings()
    yv = np.asarray(y).astype(int)
    ys = pd.Series(yv, index=X.index)
    day = T["day"].to_numpy()
    n = len(yv)
    full = _quiet(LRFeatureEngineer().fit_transform, X, ys)
    cellular = full["cellular_downturn"].to_numpy(float)
    designs = build_fold_designs(full[list(FEATURES)], ys, cv_folds=10, random_state=SEED, X_raw=X)
    S = _Scores()
    enc: Dict[str, Dict[str, np.ndarray]] = {}
    notes: List[Dict[str, Any]] = []
    banks: Dict[str, StateBank] = {}

    def note(split, fold, encoding, es, info):
        notes.append({"split": split, "fold": fold, "encoding": encoding, "early_stopping": es, **info})

    # --- the accepted random 10-fold plan -------------------------------
    for d in designs:
        k = d.fold
        tr_rows, va_rows = d.train, d.val
        S.put("lr|random|accepted", va_rows, _lr(settings, d.X_train, d.y_train, d.X_val), d.y_val)
        (acc_va,) = _accepted_encode(X, yv, tr_rows, va_rows)
        _merge(enc, "random|accepted", va_rows, acc_va)
        bank = banks[f"fold{k}"] = StateBank(X, yv, day, tr_rows)
        rule_enc = {}
        for rule in RULES:
            tr = rule_states(bank, X, T, rule, tr_rows)
            va = rule_states(bank, X, T, rule, va_rows)
            rule_enc[rule] = (tr, va)
            _merge(enc, f"random|{rule}", va_rows, va)
            a_tr, a_va = tr["available"], va["available"]
            Z_tr, Z_va = _scaled(_matrix(cellular, tr, tr_rows)[a_tr], _matrix(cellular, va, va_rows)[a_va])
            S.put(f"lr|random|{rule}", va_rows[a_va], _lr(settings, Z_tr, yv[tr_rows][a_tr], Z_va),
                  yv[va_rows][a_va])
        if mlp:
            s, info = _mlp(settings, d.X_train, d.y_train, d.X_val)
            S.put("mlp|random|accepted|es_accepted", va_rows, s, d.y_val)
            note("random", k, "accepted", "accepted", info)
            opt, val = _es_split(yv[tr_rows], settings)
            i_tr, i_va = _accepted_encode(X, yv, tr_rows[opt], tr_rows, va_rows)
            M = _matrix(cellular, i_tr, tr_rows)
            Z_tr, Z_va = _scaled(M, _matrix(cellular, i_va, va_rows), fit_on=M[opt])
            s, info = _mlp(settings, Z_tr, yv[tr_rows], Z_va, validation=val)
            S.put("mlp|random|accepted|es_isolated", va_rows, s, d.y_val)
            note("random", k, "accepted", "isolated", info)
            tr, va = rule_enc[PRIMARY]
            a_tr, a_va = tr["available"], va["available"]
            fit_rows, ev_rows = tr_rows[a_tr], va_rows[a_va]
            Z_tr, Z_va = _scaled(_matrix(cellular, tr, tr_rows)[a_tr], _matrix(cellular, va, va_rows)[a_va])
            s, info = _mlp(settings, Z_tr, yv[fit_rows], Z_va)
            S.put(f"mlp|random|{PRIMARY}|es_accepted", ev_rows, s, yv[ev_rows])
            note("random", k, PRIMARY, "accepted", info)
            opt, val = _es_split(yv[fit_rows], settings)
            iso = banks[f"fold{k}_es_isolated"] = StateBank(X, yv, day, np.setdiff1d(tr_rows, fit_rows[val]))
            i_tr = rule_states(iso, X, T, PRIMARY, fit_rows)
            i_va = rule_states(iso, X, T, PRIMARY, ev_rows)
            if not (i_tr["available"].all() and i_va["available"].all()):
                raise AssertionError("isolated past-only encodings missing where the rule encodes")
            M = _matrix(cellular, i_tr, fit_rows)
            Z_tr, Z_va = _scaled(M, _matrix(cellular, i_va, ev_rows), fit_on=M[opt])
            s, info = _mlp(settings, Z_tr, yv[fit_rows], Z_va, validation=val)
            S.put(f"mlp|random|{PRIMARY}|es_isolated", ev_rows, s, yv[ev_rows])
            note("random", k, PRIMARY, "isolated", info)
        if verbose:
            print(f"  random fold {k}: done")

    # --- chronological windows ------------------------------------------
    windows = chronological_windows(T)
    pidx = T["period_index"].to_numpy()
    G = banks["all"] = StateBank(X, yv, day, np.arange(n))
    _merge(enc, "all|accepted", np.arange(n), {f: full[f].to_numpy(float) for f in TARGET_FEATURES})
    for rule in RULES:
        _merge(enc, f"all|{rule}", np.arange(n), rule_states(G, X, T, rule, np.arange(n)))
    checks, lag_rows = [], []
    for w in windows:
        tr_rows = np.flatnonzero(pidx < w["period_indices"][0])
        ev_rows = np.flatnonzero(np.isin(pidx, w["period_indices"]))
        cut = w["first_day"]
        checks.append(_check(f"{w['window']}: its training rows are exactly the rows contacted before "
                             f"its first day ({cut})", np.array_equal(tr_rows, np.flatnonzero(day < cut)),
                             f"{len(tr_rows):,} rows"))
        acc_tr, acc_ev = _accepted_encode(X, yv, tr_rows, tr_rows, ev_rows)
        Z_tr, Z_ev = _scaled(_matrix(cellular, acc_tr, tr_rows), _matrix(cellular, acc_ev, ev_rows))
        S.put("lr|chrono|accepted_window", ev_rows, _lr(settings, Z_tr, yv[tr_rows], Z_ev))
        p_tr = rule_states(G, X, T, PRIMARY, tr_rows)
        a_tr = p_tr["available"]
        frozen = _frozen(G, X, cut, ev_rows)
        _merge(enc, "chrono|past_frozen", ev_rows, frozen)
        checks.append(_check(f"{w['window']}: past_frozen's predicted-window encodings equal the "
                             "accepted window fit's", all(np.array_equal(frozen[f], acc_ev[f])
                                                          for f in TARGET_FEATURES), f"{len(ev_rows):,} rows"))
        M_tr = _matrix(cellular, p_tr, tr_rows)[a_tr]
        Z_tr, Z_ev = _scaled(M_tr, _matrix(cellular, frozen, ev_rows))
        S.put("lr|chrono|past_frozen", ev_rows, _lr(settings, Z_tr, yv[tr_rows][a_tr], Z_ev))
        for rule in RULES:
            p_ev = rule_states(G, X, T, rule, ev_rows)
            if not p_ev["available"].all():
                raise AssertionError(f"{w['window']}: {rule} does not encode every predicted row")
            _merge(enc, f"chrono|past_updating|{rule}", ev_rows, p_ev)
            Z_tr, Z_ev = _scaled(M_tr, _matrix(cellular, p_ev, ev_rows))
            S.put(f"lr|chrono|past_updating|{rule}", ev_rows, _lr(settings, Z_tr, yv[tr_rows][a_tr], Z_ev))
            fit = _lag_consistent_fit(G, X, yv, T, rule, tr_rows, ev_rows, cellular, settings, cut)
            S.put(f"lr|chrono|lag_consistent|{rule}", ev_rows, fit["scores"])
            lag_rows.append({"window": w["window"], "rule": rule, "cutoff_day": cut,
                             "training_rows_before_cutoff": int(len(tr_rows)),
                             "outcomes_known_at_cutoff": int(fit["known"]),
                             "fit_rows": int(len(fit["fit_rows"])),
                             "fit_positives": int(yv[fit["fit_rows"]].sum())})
        same = np.array_equal(S.data[f"lr|chrono|lag_consistent|{PRIMARY}"]["scores"][-len(ev_rows):],
                              S.data[f"lr|chrono|past_updating|{PRIMARY}"]["scores"][-len(ev_rows):])
        checks.append(_check(f"{w['window']}: at lag 0 the lag-consistent model is the past-only updating "
                             "model (same rows, encodings, outcomes and scores)", same, f"{len(ev_rows):,} rows"))
        if mlp:
            Z_tr, Z_ev = _scaled(_matrix(cellular, acc_tr, tr_rows), _matrix(cellular, acc_ev, ev_rows))
            s, info = _mlp(settings, Z_tr, yv[tr_rows], Z_ev)
            S.put("mlp|chrono|accepted_window|es_accepted", ev_rows, s)
            note("chrono", w["window"], "accepted_window", "accepted", info)
            opt, val = _es_split(yv[tr_rows], settings)
            i_tr, i_ev = _accepted_encode(X, yv, tr_rows[opt], tr_rows, ev_rows)
            M = _matrix(cellular, i_tr, tr_rows)
            Z_tr, Z_ev = _scaled(M, _matrix(cellular, i_ev, ev_rows), fit_on=M[opt])
            s, info = _mlp(settings, Z_tr, yv[tr_rows], Z_ev, validation=val)
            S.put("mlp|chrono|accepted_window|es_isolated", ev_rows, s)
            note("chrono", w["window"], "accepted_window", "isolated", info)
            p_ev = rule_states(G, X, T, PRIMARY, ev_rows)
            Z_tr, Z_ev = _scaled(M_tr, _matrix(cellular, p_ev, ev_rows))
            fit_rows = tr_rows[a_tr]
            s, info = _mlp(settings, Z_tr, yv[fit_rows], Z_ev)
            S.put(f"mlp|chrono|past_updating|{PRIMARY}|es_accepted", ev_rows, s)
            note("chrono", w["window"], PRIMARY, "accepted", info)
            opt, val = _es_split(yv[fit_rows], settings)
            iso = banks[f"{w['window']}_es_isolated"] = StateBank(X, yv, day,
                                                                  np.setdiff1d(np.arange(n), fit_rows[val]))
            i_tr = rule_states(iso, X, T, PRIMARY, fit_rows)
            i_ev = rule_states(iso, X, T, PRIMARY, ev_rows)
            M = _matrix(cellular, i_tr, fit_rows)
            Z_tr, Z_ev = _scaled(M, _matrix(cellular, i_ev, ev_rows), fit_on=M[opt])
            s, info = _mlp(settings, Z_tr, yv[fit_rows], Z_ev, validation=val)
            S.put(f"mlp|chrono|past_updating|{PRIMARY}|es_isolated", ev_rows, s)
            note("chrono", w["window"], PRIMARY, "isolated", info)
        if verbose:
            print(f"  {w['window']} ({w['periods'][0]}..{w['periods'][-1]}): done")
    checks.append(_prediction_invariance(X, yv, T, windows, cellular, settings))
    checks += _lag_consistent_invariance(X, yv, T, windows, cellular, settings)
    for v in S.data.values():
        order = np.argsort(v["rows"], kind="stable")
        v["rows"], v["scores"] = v["rows"][order], v["scores"][order]
    return {"scores": S.data, "fold_roc_auc": S.by_fold, "notes": pd.DataFrame(notes),
            "encodings": _sorted(enc), "banks": banks, "windows": windows,
            "segments": segments(T, windows), "chrono_checks": pd.DataFrame(checks),
            "lag_consistent_fits": pd.DataFrame(lag_rows), "settings": settings, "y": yv}


def known_at_cutoff(rule: str, cutoff: int, day: np.ndarray) -> np.ndarray:
    """
    Rows whose outcome is known at a cutoff (a window's first contact day, which
    is a period start) under ``rule``: contacted before ``cutoff - lag`` for a
    daily rule; before the cutoff's period, i.e. before the cutoff, for the
    monthly rule (each period's outcomes are known by the next period's start).
    """
    spec = RULES[rule]
    return day < (cutoff - spec["lag"] if spec["cadence"] == "daily" else cutoff)


def _lag_consistent_fit(bank: StateBank, X, yv, T, rule, tr_rows, ev_rows, cellular, settings,
                        cutoff: int) -> Dict[str, Any]:
    """
    The model operated under ``rule``'s outcome delay throughout: it is fitted
    (scaler and LR) on the training rows whose outcome is known at the cutoff
    and that the rule encodes, with the rule's encodings (each fitted on outcomes
    known before that row); predicted rows are encoded by the rule's state at
    their own contact day. ``bank`` holds the labels the encodings may use.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    day = T["day"].to_numpy()
    known = known_at_cutoff(rule, cutoff, day[tr_rows])
    enc_tr = rule_states(bank, X, T, rule, tr_rows)
    use = known & enc_tr["available"]
    fit_rows = tr_rows[use]
    if len(np.unique(yv[fit_rows])) < 2:
        raise AssertionError(f"{rule} at cutoff {cutoff}: {len(fit_rows)} training rows with a known "
                             "outcome and an encoding, too few to fit")
    enc_ev = rule_states(bank, X, T, rule, ev_rows)
    if not enc_ev["available"].all():
        raise AssertionError(f"{rule} does not encode every predicted row")
    M_fit = _matrix(cellular, enc_tr, tr_rows)[use]
    sc = StandardScaler().fit(M_fit)
    model = LogisticRegression(**settings["lr_kwargs"]).fit(sc.transform(M_fit), yv[fit_rows])
    scores = model.predict_proba(sc.transform(_matrix(cellular, enc_ev, ev_rows)))[:, 1]
    return {"known": int(known.sum()), "fit_rows": fit_rows, "scores": scores,
            "state_day": enc_ev["state_day"],
            "params": np.r_[sc.mean_, sc.scale_, model.coef_.ravel(), model.intercept_]}


def _lag0_model_params(X, labels, T, tr_rows, cellular, settings) -> np.ndarray:
    """The model the prediction-time-only lag arms use: lag-0 encodings, every training outcome."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    bank = StateBank(X, labels, T["day"].to_numpy(), np.arange(len(labels)))
    enc = rule_states(bank, X, T, PRIMARY, tr_rows)
    a = enc["available"]
    M = _matrix(cellular, enc, tr_rows)[a]
    sc = StandardScaler().fit(M)
    model = LogisticRegression(**settings["lr_kwargs"]).fit(sc.transform(M), labels[tr_rows][a])
    return np.r_[sc.mean_, sc.scale_, model.coef_.ravel(), model.intercept_]


def _lag_consistent_invariance(X, yv, T, windows, cellular, settings) -> List[Dict[str, Any]]:
    """
    For the lagged rules, in the first and the last window: flipping the
    outcomes that are still unavailable at the cutoff (pre-cutoff rows
    contacted within the lag) leaves the lag-consistent model (scaler,
    coefficients, intercept) unchanged, and the deployed scores of every
    predicted row whose encoding was fixed before those outcomes arrive;
    flipping outcomes known at the cutoff changes the model (control). For
    contrast, the same flip changes the lag-0 model the prediction-time-only
    arms use, which is trained on every pre-cutoff outcome. Under the monthly
    rule no pre-cutoff outcome is unavailable at a period-start cutoff, so the
    flip is of the window's first-period outcomes (unavailable throughout that
    period).
    """
    day = T["day"].to_numpy()
    pidx = T["period_index"].to_numpy()
    n = len(yv)
    out = []
    for w in (windows[0], windows[-1]) if len(windows) > 1 else windows[:1]:
        tr = np.flatnonzero(pidx < w["period_indices"][0])
        ev = np.flatnonzero(np.isin(pidx, w["period_indices"]))
        cut = w["first_day"]
        base_params = None
        for rule, spec in RULES.items():
            if rule == PRIMARY:
                continue
            if spec["cadence"] == "daily":
                unknown = (day >= cut - spec["lag"]) & (day < cut)
            else:
                unknown = np.isin(np.arange(n), ev) & (pidx == w["period_indices"][0])
            known = (day < cut) & ~unknown & known_at_cutoff(rule, cut, day)

            def fit(labels):
                return _lag_consistent_fit(StateBank(X, labels, day, np.arange(n)), X, labels, T, rule,
                                           tr, ev, cellular, settings, cut)

            base = fit(yv)
            y_un, y_kn = yv.copy(), yv.copy()
            y_un[unknown] = 1 - y_un[unknown]
            y_kn[known] = 1 - y_kn[known]
            flipped, control = fit(y_un), fit(y_kn)
            fixed = base["state_day"] <= day[unknown].min()
            model_same = np.array_equal(base["params"], flipped["params"])
            scores_same = np.array_equal(base["scores"][fixed], flipped["scores"][fixed])
            control_moved = not np.array_equal(base["params"], control["params"])
            out.append(_check(
                f"{w['window']} {rule}: flipping the {int(unknown.sum()):,} outcomes still unavailable "
                "(pre-cutoff within the lag; the window's first period for the monthly rule) leaves the "
                "lag-consistent model and the deployed scores fixed before they arrive unchanged; "
                "flipping known outcomes changes the model (control)",
                model_same and scores_same and control_moved and fixed.any(),
                f"model unchanged {model_same}; {int(fixed.sum()):,} fixed rows' scores unchanged "
                f"{scores_same}; control changed the model {control_moved}"))
            if spec["cadence"] == "daily":
                if base_params is None:
                    base_params = _lag0_model_params(X, yv, T, tr, cellular, settings)
                moved = not np.array_equal(base_params, _lag0_model_params(X, y_un, T, tr, cellular, settings))
                out.append(_check(
                    f"{w['window']} {rule} (contrast): the same flip changes the lag-0 model of the "
                    "prediction-time-only arm, which trains on outcomes unavailable at the cutoff under "
                    "this lag", moved, f"lag-0 model changed {moved}"))
    return out


def _prediction_invariance(X, yv, T, windows, cellular, settings) -> Dict[str, Any]:
    """
    The deployed past-only LR of the first window: flipping the labels of a
    mid-window day's and later predicted-window rows leaves that day's
    encodings and scores unchanged; flipping the earlier predicted-window labels
    changes its encodings (control; whether the scores then move as well depends
    on the fitted coefficients and is reported).
    """
    day = T["day"].to_numpy()
    pidx = T["period_index"].to_numpy()
    w = windows[0]
    tr = np.flatnonzero(pidx < w["period_indices"][0])
    ev = np.flatnonzero(np.isin(pidx, w["period_indices"]))
    ev_days = np.unique(day[ev])
    d0 = int(ev_days[len(ev_days) // 2])
    q = ev[day[ev] == d0]

    def run(labels):
        G = StateBank(X, labels, day, np.arange(len(labels)))
        p_tr = rule_states(G, X, T, PRIMARY, tr)
        a = p_tr["available"]
        p_q = rule_states(G, X, T, PRIMARY, q)
        Z_tr, Z_q = _scaled(_matrix(cellular, p_tr, tr)[a], _matrix(cellular, p_q, q))
        return _matrix(cellular, p_q, q), _lr(settings, Z_tr, labels[tr][a], Z_q)

    base_enc, base = run(yv)
    in_ev = np.isin(np.arange(len(yv)), ev)
    late, early = yv.copy(), yv.copy()
    late[in_ev & (day >= d0)] = 1 - late[in_ev & (day >= d0)]
    early[in_ev & (day < d0)] = 1 - early[in_ev & (day < d0)]
    late_enc, late_s = run(late)
    early_enc, early_s = run(early)
    same = np.array_equal(base_enc, late_enc) and np.array_equal(base, late_s)
    enc_moved, score_moved = not np.array_equal(base_enc, early_enc), not np.allclose(base, early_s)
    return _check(f"{w['window']}: the deployed past-only LR's encodings and scores on contact day {d0} are "
                  "unchanged when that day's and later predicted-window labels flip; earlier flips change "
                  "the encodings (control)", same and enc_moved,
                  f"{len(q)} rows; unchanged {same}; control: encodings moved {enc_moved}, "
                  f"scores moved {score_moved}")


def delong(y: np.ndarray, scores: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """ROC-AUCs of several scores on the same rows and their DeLong covariance matrix."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v10, v01 = [], [], []
    for s in scores:
        s = np.asarray(s, float)
        tx, ty, tz = rankdata(s[pos]), rankdata(s[neg]), rankdata(np.r_[s[pos], s[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    cov = np.atleast_2d(np.cov(np.vstack(v10))) / m + np.atleast_2d(np.cov(np.vstack(v01))) / n
    return np.asarray(aucs), cov


def _contrast(aucs, cov, c) -> Tuple[float, float]:
    c = np.asarray(c, float)
    return float(c @ aucs), float(np.sqrt(max(float(c @ cov @ c), 0.0)))


def _interval(diff: float, se: float) -> Dict[str, float]:
    from scipy.stats import norm
    z = diff / se if se > 0 else (float("inf") if diff else 0.0)
    return {"diff": diff, "se": se, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            "p_value": float(2 * norm.sf(abs(z)))}


def delong_paired(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired DeLong comparison of two scores' ROC-AUCs on the same rows: b - a."""
    aucs, cov = delong(y, [a, b])
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), **_interval(*_contrast(aucs, cov, [-1, 1])),
            "rows": int(len(y)), "positives": int(np.sum(y))}


def _aligned(ev: Mapping[str, Any], names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """The rows every named protocol scored, and each protocol's scores on them."""
    rows = None
    for nm in names:
        r = ev["scores"][nm]["rows"]
        rows = r if rows is None else np.intersect1d(rows, r)
    out = []
    for nm in names:
        r, s = ev["scores"][nm]["rows"], ev["scores"][nm]["scores"]
        out.append(s[np.searchsorted(r, rows)])
    return rows, out


#: The registered comparisons: (question, arm, protocol a, protocol b); each reports b - a. A
#: comparison is a difference between two protocols on the rows both scored. "Past-only" and
#: "accepted" name encoding protocols: the past-only protocol differs from the accepted one in
#: several ways at once (no same-day or later labels, no own label in the training encodings,
#: edges, rates and fallbacks fitted on fewer rows, cold-start rows left out of its fits), and
#: no comparison here separates them.
COMPARISONS = (
    ("past-only vs accepted protocol, random split", "lr", "lr|random|accepted", "lr|random|past_daily_lag0"),
    ("past-only (lag 5) vs accepted protocol, random split", "lr", "lr|random|accepted",
     "lr|random|past_daily_lag5"),
    ("past-only (lag 20) vs accepted protocol, random split", "lr", "lr|random|accepted",
     "lr|random|past_daily_lag20"),
    ("past-only (monthly) vs accepted protocol, random split", "lr", "lr|random|accepted",
     "lr|random|past_monthly"),
    ("past-only frozen vs accepted protocol, chronological", "lr", "lr|chrono|accepted_window",
     "lr|chrono|past_frozen"),
    ("past-only updating vs accepted protocol, chronological", "lr", "lr|chrono|accepted_window",
     "lr|chrono|past_updating|past_daily_lag0"),
    ("updating vs frozen past-only, chronological", "lr", "lr|chrono|past_frozen",
     "lr|chrono|past_updating|past_daily_lag0"),
    ("lag-0 model fed lag-5 encodings at prediction, chronological", "lr",
     "lr|chrono|past_updating|past_daily_lag0", "lr|chrono|past_updating|past_daily_lag5"),
    ("lag-0 model fed lag-20 encodings at prediction, chronological", "lr",
     "lr|chrono|past_updating|past_daily_lag0", "lr|chrono|past_updating|past_daily_lag20"),
    ("lag-0 model fed monthly encodings at prediction, chronological", "lr",
     "lr|chrono|past_updating|past_daily_lag0", "lr|chrono|past_updating|past_monthly"),
    ("outcome lag 5 throughout (lag-consistent) vs lag 0, chronological", "lr",
     "lr|chrono|lag_consistent|past_daily_lag0", "lr|chrono|lag_consistent|past_daily_lag5"),
    ("outcome lag 20 throughout (lag-consistent) vs lag 0, chronological", "lr",
     "lr|chrono|lag_consistent|past_daily_lag0", "lr|chrono|lag_consistent|past_daily_lag20"),
    ("monthly outcomes throughout (lag-consistent) vs lag 0, chronological", "lr",
     "lr|chrono|lag_consistent|past_daily_lag0", "lr|chrono|lag_consistent|past_monthly"),
    ("lag 20: lag-consistent vs prediction-time-only, chronological", "lr",
     "lr|chrono|past_updating|past_daily_lag20", "lr|chrono|lag_consistent|past_daily_lag20"),
    ("chronological vs random split, accepted protocol", "lr", "lr|random|accepted", "lr|chrono|accepted_window"),
    ("chronological vs random split, past-only protocol", "lr", "lr|random|past_daily_lag0",
     "lr|chrono|past_updating|past_daily_lag0"),
    ("isolated vs accepted early stopping, accepted protocol, random split", "mlp",
     "mlp|random|accepted|es_accepted", "mlp|random|accepted|es_isolated"),
    ("isolated vs accepted early stopping, past-only protocol, random split", "mlp",
     "mlp|random|past_daily_lag0|es_accepted", "mlp|random|past_daily_lag0|es_isolated"),
    ("isolated vs accepted early stopping, accepted protocol, chronological", "mlp",
     "mlp|chrono|accepted_window|es_accepted", "mlp|chrono|accepted_window|es_isolated"),
    ("isolated vs accepted early stopping, past-only protocol, chronological", "mlp",
     "mlp|chrono|past_updating|past_daily_lag0|es_accepted",
     "mlp|chrono|past_updating|past_daily_lag0|es_isolated"),
    ("past-only vs accepted protocol, random split (isolated early stopping)", "mlp",
     "mlp|random|accepted|es_isolated", "mlp|random|past_daily_lag0|es_isolated"),
    ("past-only updating vs accepted protocol, chronological (isolated early stopping)", "mlp",
     "mlp|chrono|accepted_window|es_isolated", "mlp|chrono|past_updating|past_daily_lag0|es_isolated"),
    ("chronological vs random split, accepted protocol (isolated early stopping)", "mlp",
     "mlp|random|accepted|es_isolated", "mlp|chrono|accepted_window|es_isolated"),
    ("chronological vs random split, past-only protocol (isolated early stopping)", "mlp",
     "mlp|random|past_daily_lag0|es_isolated", "mlp|chrono|past_updating|past_daily_lag0|es_isolated"),
)


def comparisons(ev: Mapping[str, Any]) -> pd.DataFrame:
    """
    Every registered comparison whose protocols ``ev`` holds: paired DeLong on
    the rows both scored, pooled and per segment with both classes present,
    and, where every window is scored, the unweighted mean over windows
    (independent-window standard error). Pooled chronological scores mix five
    models' score levels across windows of very different prevalence; the mean
    over windows measures ranking within them.
    """
    seg = ev["segments"]
    windows = [w["window"] for w in ev["windows"]]
    out = []
    for question, arm, a, b in COMPARISONS:
        if a not in ev["scores"] or b not in ev["scores"]:
            continue
        rows, (sa, sb) = _aligned(ev, [a, b])
        y = ev["y"][rows]
        scopes = [("pooled", np.ones(len(rows), dtype=bool))]
        scopes += [(s, seg[rows] == s) for s in pd.unique(seg[rows])]
        per_window = []
        for scope, m in scopes:
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                continue
            res = delong_paired(y[m], sa[m], sb[m])
            out.append({"question": question, "arm": arm, "scope": scope, "a": a, "b": b, **res,
                        "prevalence": float(y[m].mean()),
                        "ap_a": _score_summary(y[m], sa[m])["average_precision"],
                        "ap_b": _score_summary(y[m], sb[m])["average_precision"]})
            if scope in windows:
                per_window.append(res)
        if len(per_window) == len(windows):
            k = len(per_window)
            mean = float(np.mean([r["diff"] for r in per_window]))
            se = float(np.sqrt(sum(r["se"] ** 2 for r in per_window))) / k
            out.append({"question": question, "arm": arm, "scope": "mean over windows", "a": a, "b": b,
                        "auc_a": float(np.mean([r["auc_a"] for r in per_window])),
                        "auc_b": float(np.mean([r["auc_b"] for r in per_window])),
                        **_interval(mean, se), "rows": int(sum(r["rows"] for r in per_window)),
                        "positives": int(sum(r["positives"] for r in per_window))})
    return pd.DataFrame(out)


FACTORIAL = {
    "lr": ("lr|random|accepted", "lr|random|past_daily_lag0", "lr|chrono|accepted_window",
           "lr|chrono|past_updating|past_daily_lag0"),
    "mlp": ("mlp|random|accepted|es_isolated", "mlp|random|past_daily_lag0|es_isolated",
            "mlp|chrono|accepted_window|es_isolated", "mlp|chrono|past_updating|past_daily_lag0|es_isolated"),
}
_EFFECTS = {"past-only vs accepted protocol, random split (B - A)": [-1, 1, 0, 0],
            "past-only vs accepted protocol, chronological (D - C)": [0, 0, -1, 1],
            "chronological vs random split, accepted protocol (C - A)": [-1, 0, 1, 0],
            "chronological vs random split, past-only protocol (D - B)": [0, -1, 0, 1],
            "interaction ((D - C) - (B - A))": [1, -1, -1, 1]}


def factorial(ev: Mapping[str, Any], arm: str) -> Dict[str, pd.DataFrame]:
    """
    The 2 x 2 (split x encoding protocol) on the rows all four protocols scored
    (the windows' rows): cell ROC-AUCs and the contrasts, pooled (DeLong
    covariance of the four) and as the unweighted mean over windows (windows
    independent). Each contrast is a difference between protocols; it does not
    isolate any one of the changes a protocol makes (``COMPARISONS``).
    """
    names = FACTORIAL[arm]
    if not all(nm in ev["scores"] for nm in names):
        return {"cells": pd.DataFrame(), "effects": pd.DataFrame()}
    rows, sc = _aligned(ev, names)
    y = ev["y"][rows]
    seg = ev["segments"][rows]
    wins = [w["window"] for w in ev["windows"] if (seg == w["window"]).any()]
    aucs, cov = delong(y, sc)
    per = {w: delong(y[seg == w], [s[seg == w] for s in sc]) for w in wins}
    cells = []
    for i, (label, nm) in enumerate(zip("ABCD", names)):
        cells.append({"cell": label, "split": "random" if i < 2 else "chronological",
                      "encoding": "accepted" if i % 2 == 0 else "past-only", "protocol": nm,
                      "pooled_roc_auc": float(aucs[i]),
                      "mean_window_roc_auc": float(np.mean([per[w][0][i] for w in wins])),
                      **{f"{w}_roc_auc": float(per[w][0][i]) for w in wins},
                      "pooled_average_precision": _score_summary(y, sc[i])["average_precision"]})
    effects = []
    for label, c in _EFFECTS.items():
        pooled = _interval(*_contrast(aucs, cov, c))
        diffs = [_contrast(*per[w], c) for w in wins]
        mean = float(np.mean([d for d, _ in diffs]))
        se = float(np.sqrt(sum(s ** 2 for _, s in diffs))) / len(wins)
        effects.append({"effect": label, "pooled_diff": pooled["diff"], "pooled_ci_low": pooled["ci_low"],
                        "pooled_ci_high": pooled["ci_high"], "pooled_p": pooled["p_value"],
                        "mean_window_diff": mean, "mean_window_ci_low": mean - 1.96 * se,
                        "mean_window_ci_high": mean + 1.96 * se,
                        **{f"{w}_diff": d for w, (d, _) in zip(wins, diffs)}})
    return {"cells": pd.DataFrame(cells), "effects": pd.DataFrame(effects),
            "rows": int(len(rows)), "positives": int(y.sum()), "windows": wins}


def protocol_table(ev: Mapping[str, Any]) -> pd.DataFrame:
    """Each protocol on its own rows, pooled and per segment: rows, prevalence, ROC-AUC, AP."""
    seg = ev["segments"]
    out = []
    for name, v in ev["scores"].items():
        r, s = v["rows"], v["scores"]
        y = ev["y"][r]
        out.append({"protocol": name, "scope": "pooled", **_score_summary(y, s)})
        for sname in pd.unique(seg[r]):
            m = seg[r] == sname
            if len(np.unique(y[m])) == 2:
                out.append({"protocol": name, "scope": sname, **_score_summary(y[m], s[m])})
    return pd.DataFrame(out)


#: The LR (saga, seeded) reproduces bit for bit. The MLP's floating-point path depends on the BLAS
#: build, so a platform other than the one that wrote the PR 39 record can differ in the last digits.
REPRODUCTION_TOLERANCE = {"lr": 1e-9, "mlp": 1e-5}


def reproduction_checks(ev: Mapping[str, Any]) -> pd.DataFrame:
    """
    The accepted protocols reproduce PR 39: the mean fold ROC-AUC of the accepted
    LR and MLP on the accepted designs equals the recorded best CV ROC-AUC; every
    isolated MLP fit's early-stopping subset is the reproduced one.
    """
    rec = ev["settings"]["recorded_cv_roc_auc"]
    out = []
    for arm, name in (("lr", "lr|random|accepted"), ("mlp", "mlp|random|accepted|es_accepted")):
        if name not in ev["fold_roc_auc"]:
            continue
        got = float(np.mean(ev["fold_roc_auc"][name]))
        tol = REPRODUCTION_TOLERANCE[arm]
        out.append(_check(f"accepted {arm} on the accepted designs reproduces the PR 39 Commit 2 best CV "
                          f"ROC-AUC (mean over the 10 folds; tolerance {tol:.0e})", abs(got - rec[arm]) <= tol,
                          f"{got:.12f} vs recorded {rec[arm]:.12f} (difference {got - rec[arm]:+.1e})"))
    notes = ev["notes"]
    if len(notes):
        iso = notes[notes["early_stopping"] == "isolated"]
        out.append(_check("every isolated MLP fit early-stopped on the reproduced subset (its recorded best "
                          "validation ROC-AUC equals the AUC on those rows)",
                          len(iso) > 0 and bool(iso["split_reproduced"].all()),
                          f"{int(iso['split_reproduced'].sum())}/{len(iso)} fits"))
    return pd.DataFrame(out)


def early_stopping_table(ev: Mapping[str, Any]) -> pd.DataFrame:
    """Stopping epochs and best validation ROC-AUC per MLP protocol (accepted vs isolated)."""
    notes = ev["notes"]
    if not len(notes):
        return pd.DataFrame()
    g = notes.groupby(["split", "encoding", "early_stopping"], sort=False)
    return g.agg(fits=("best_epoch", "size"), best_epoch_median=("best_epoch", "median"),
                 best_epoch_min=("best_epoch", "min"), best_epoch_max=("best_epoch", "max"),
                 best_val_roc_auc_mean=("best_val_roc_auc", "mean")).reset_index()


# ----------------------------------------------------------------------
# 6. decisions, extension, protocol, finding
# ----------------------------------------------------------------------
def candidate_contract(rule: str = PRIMARY) -> Dict[str, Any]:
    """
    The proposed (not registered) Stage 1 feature contract of a past-only rule:
    distinct feature names, because the accepted contract identifies features by
    name and could not tell the two definitions apart; a pre-release version;
    and a fingerprint of the definition as well as of the names.
    """
    from shared.stage1.feature_engineering import STAGE1_FEATURE_CONTRACT as C

    names = ["cellular_downturn", "subscribe_rate_by_euribor_bin_past", "subscribe_rate_by_dow_month_past"]
    definition = {"rule": rule, "cadence": RULES[rule]["cadence"], "lag_contact_days": RULES[rule]["lag"],
                  "same_day_labels": "excluded", "cold_start_contact_days": COLD_START_DAYS,
                  "min_state_rows": MIN_STATE_ROWS, "min_state_positives": MIN_STATE_POSITIVES,
                  "fitted_at_each_update": ["qcut edges", "bin rates", "median fallback",
                                            "day x month cell rates", "global prior"],
                  "n_bins": 20, "smoothing_factor": 100,
                  "requires_input": "the PR 42 contact-day time index (encoding_audit.build_time_index)"}
    return {"id": C["id"], "version": "3.0.0-candidate", "features": names,
            "fingerprint": hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest()[:16],
            "definition": definition,
            "definition_fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True)
                                                     .encode("utf-8")).hexdigest()[:16],
            "accepted_contract": {"id": C["id"], "version": C["version"], "fingerprint": C["fingerprint"],
                                  "status": "frozen (PR 39): unchanged and still the one artifacts carry"},
            "status": "proposed in PR 42 Step 3B; not registered; adoption is PR 44's"}


def contract_decisions(availability: pd.DataFrame, invariance: pd.DataFrame,
                       compared: pd.DataFrame) -> pd.DataFrame:
    """The registered contract criteria applied to each candidate (``REGISTRATION``)."""
    av = availability.set_index("rule")
    rows = []
    for rule, spec in RULES.items():
        inv = invariance[invariance["check"].str.contains(f" {rule}: ", regex=False)]
        realizable = bool(len(inv)) and bool(inv["passed"].all())
        share = float(av.loc[rule, "available_share"])
        differ = max(int(av.loc[rule, f"{f}_rows_differ"]) for f in TARGET_FEATURES)
        differs = differ >= 0.01 * int(av.loc[rule, "available"])
        contract = realizable and differs
        if not contract:
            dest = "none (no new definition)"
        elif spec["role"] == "primary candidate" and share >= 0.95:
            dest = "PR 44 Stage 1 candidate (versioned contract)"
        elif spec["role"] == "primary candidate":
            dest = "PR 49 only (too little availability for a contract)"
        else:
            dest = "PR 49 outcome-lag sensitivity (versioned contract only if adopted)"
        c = compared[(compared["b"] == f"lr|random|{rule}") & (compared["scope"] == "pooled")]
        rows.append({"candidate": rule, "role": spec["role"], "realizable": realizable,
                     "available_share": share, "rows_differing": differ, "differs": differs,
                     "needs_versioned_contract": contract, "destination": dest,
                     "lr_random_roc_auc_diff": float(c["diff"].iloc[0]) if len(c) else None})
    rows.append({"candidate": "MLP isolated early stopping", "role": "training procedure",
                 "realizable": True, "available_share": 1.0, "rows_differing": 0, "differs": False,
                 "needs_versioned_contract": False,
                 "destination": "PR 44 training-procedure candidate (feature contract unchanged)",
                 "lr_random_roc_auc_diff": None})
    rows.append({"candidate": "accepted (PR 39)", "role": "frozen baseline", "realizable": False,
                 "available_share": 1.0, "rows_differing": 0, "differs": False,
                 "needs_versioned_contract": False,
                 "destination": "unchanged; the baseline PR 43 reads", "lr_random_roc_auc_diff": 0.0})
    return pd.DataFrame(rows)


#: How each Stage 2-3 family's past-only version would be built, and what this step tested.
_STAGE23 = {
    "dow_month_encoded": {
        "past_only_rule": ("the primary rule unchanged: the Stage 2 and Stage 3 implementations use the "
                           "same smoothed day x month formula as Stage 1 (k = 100, global prior, unseen "
                           "cell -> global rate; Step 3A construction and reconstruction)"),
        "direct_evidence": ("the rule's realizability, availability, fallback frequency and invariance "
                            "(Stage 1 day x month, here); not its predictive effect in Stages 2-3"),
        "inferred": ("its dow_* bins inherit the change; the Stage 2 obligations and the Stage 3 "
                     "models' use of it were not refitted"),
    },
    "neighborhood_subscription_density": {
        "past_only_rule": ("a kernel average over reference rows contacted before day d (and the "
                           "scaler fitted on rows before d); the reference sample must be redrawn or "
                           "grown as rows arrive, a design choice this step does not make"),
        "direct_evidence": "none for a past-only version (Step 3A attribution only)",
        "inferred": ("realizable in principle; its Step 3A look-ahead share (about 0.47) is larger "
                     "than Stage 1's (about 0.37-0.41); its predictive change is not estimated"),
    },
    "economic_curvature_intensity": {
        "past_only_rule": ("bin rates per macro variable from rows before day d, then the curvature over "
                           "the non-empty bins; early states have few distinct euribor and employment "
                           "values, so the curvature is often zero (the production fallback)"),
        "direct_evidence": ("none for ECI itself; the euribor bin-rate rule (Stage 1, here) is its "
                            "first step for one of its three variables"),
        "inferred": ("realizable; the qcut-on-the-past behaviour (edges, empty bins, fallback) seen for "
                     "the euribor bin rate applies to each of its three variables"),
    },
}


def stage23_extension(summary: pd.DataFrame) -> pd.DataFrame:
    """The Stage 2-3 target-derived families: Step 3A's look-ahead and what Step 3B does and does not show."""
    from feature_research import encoding_audit as EA

    rows = []
    for feat, spec in _STAGE23.items():
        for stage in ("stage2", "stage3"):
            sel = summary[(summary["feature"] == feat) & (summary["stage"] == stage)
                          & (summary["path"] == EA.PRIMARY_HELD_OUT[stage]) & (summary["role"] == "held_out")]
            later = float(sel["later_share_mean"].iloc[0]) if len(sel) else None
            rows.append({"family": feat, "stage": stage,
                         "descendants": ", ".join(EA.DESCENDANTS.get(feat, [])) or "none",
                         "step3a_held_out_later_share_mean": later, **spec})
    return pd.DataFrame(rows)


PR49_PROTOCOL = {
    "id": "pr42-pr49-temporal-protocol",
    "version": "1.0.0-design",
    "status": ("designed in PR 42 Step 3B; executed in PR 49; reported as temporal robustness "
               "evidence, never as independent validation"),
    "data": ("the training split on the PR 42 time index. Whether PR 49 adds the previously exposed "
             "test rows is PR 49's decision, disclosed with that exposure"),
    "split": {"kind": "forward chaining over contact periods",
              "windows": ("the Step 3B windows (registration): 2008-05..2008-08 train only, then "
                          "consecutive windows of at least 1,500 rows; each window is predicted by a "
                          "cascade fitted on every earlier row and frozen at the window's first day"),
              "gap": "none in the primary run; a one-period gap is the outcome-lag sensitivity"},
    "fit_at_each_cutoff": {
        "stage1": ("encoders (past-only rule on the training rows; frozen and updating arms for the "
                   "predicted rows), scaler, model with the accepted hyperparameters (no retuning in "
                   "the primary run), sigmoid or isotonic calibrator by nested OOF within the training "
                   "rows, the F2 threshold from those OOF scores. Under an outcome lag, every one of "
                   "these is fitted on the training rows whose outcomes are known at the cutoff "
                   "(known_at_cutoff), as Step 3B's lag-consistent arm does"),
        "stage2": ("NSD, ECI and day x month sources (past-only where PR 45 adopted a candidate, "
                   "otherwise the accepted fit on the training rows), the bins (the frozen PR 39 cut "
                   "points, or Commit 4's procedural bins refitted on the training rows), the GLASS "
                   "rule ILP and the RF router under the frozen operating contract "
                   "(pr37-stage2-primary 1.0.0)"),
        "stage3": "EBM and XGBoost, their calibrators (ECE gate) and nested thresholds, on the training rows",
        "stage4": "arbiters, weights and the abstention cut (registered coverage floor) on the training rows' OOF table",
    },
    "updated_after_cutoff": [
        "only the label statistics of past-only encodings, under the registered rule and lag, as "
        "outcomes arrive (the updating arm)",
        "nothing else: models, calibrators, thresholds, bins, routes, rules and the abstention cut "
        "stay frozen within a window",
        "a separately labelled recalibration arm may refit calibrators and thresholds on labels known "
        "before each window; it is reported beside the frozen arm and never replaces it",
    ],
    "prevalence_shift": {
        "measure": "each window's subscription rate beside its training rows' (about 3-6% in 2008, 34-60% from 2009-06)",
        "calibration": ("calibration in the large (mean predicted vs observed), reliability curve, "
                        "Brier and ECE per window with the cutoff calibrator; a prior-shift-corrected "
                        "curve as a labelled diagnostic"),
        "thresholds": ("F2, recall, precision and the confusion counts at the frozen threshold per "
                       "window, beside the window's own F2-optimal threshold as a diagnostic only"),
        "abstention": ("realised coverage and accuracy on the covered rows against the registered "
                       "coverage floor per window; the cut is not moved to meet the floor"),
        "stage2_obligations": ("Pass 1 NPV (>= 0.95), the Pass 1 leakage share of all positives "
                               "(<= 0.15) and Pass 2 precision (>= 0.30) observed per window with "
                               "shared.stage2.obligations.observe_routing; support floors are fit-time "
                               "only. With rules frozen, a higher prevalence lowers Pass 1 NPV and "
                               "raises Pass 2 precision, so the NPV floor and the leakage share are "
                               "the obligations at risk in the high-prevalence windows"),
    },
    "prior_shift_attribution": ("for every window and obligation, the observed value beside its "
                                "pure-prior-shift projection from the training OOF class-conditional "
                                "routing rates (stage2_prevalence_projection): the projection says what "
                                "the prevalence alone predicts, the difference is the change in routing "
                                "rates (covariate or concept shift). Step 3B's projection for the "
                                "frozen routing: GLASS meets all three obligations only for prevalence "
                                "between about 10.9% and 12.0%; the RF router's training OOF leakage "
                                "share already exceeds the ceiling, and a prior shift does not move it"),
    "missed_obligation_report": [
        "obligation, window, arm (frozen, updating, recalibrated) and stage",
        "numerator / denominator, observed value, limit and exact signed shortfall (observe_routing), "
        "routed rows and positives, window prevalence",
        "status satisfied, violated or undefined: undefined is never compliant, any violation is "
        "reported however small, and nothing is refitted or re-thresholded to remove it",
        "whether the same obligation held on the training rows' OOF routing at that cutoff",
        "the claim it bounds: a miss limits the temporal claim for that window and arm; it is a "
        "robustness finding, not a pipeline failure, and it is never averaged away across windows",
    ],
    "comparisons": ["accepted vs past-only encoding protocols; frozen vs updating; both Stage 1 arms",
                    "random-split vs chronological results on the same rows (a difference of protocols)",
                    "outcome lag 0, 5 and 20 contact days and the monthly rule, lag-consistent "
                    "throughout: every component fitted at a cutoff uses only outcomes known then "
                    "under the lag; the lag-0 model fed slower encodings as a secondary arm",
                    "MLP accepted vs isolated early stopping"],
    "pre_registration": ("windows, rules, lags, metrics, arms and the missed-obligation report are "
                         "fixed by this design before PR 49 examines any chronological cascade result"),
}


#: The accepted Stage 2 routing on the training rows (PR 38 run c3_20261112_153333, reused by PR 39):
#: pooled out-of-fold counts per arm, from the run's consolidated research report.
STAGE2_ROUTING_SOURCE = "research_logs/pr38/c3_20261112_153333/report_v1/02_router_metrics.csv"


def accepted_routing_counts() -> Dict[str, Any]:
    """
    The accepted routers' pooled training-side OOF routing counts (GLASS and RF),
    read from the PR 38 report (no model, artifact or label is read) and checked
    for internal consistency.
    """
    import csv

    data = (BANK_ROOT / STAGE2_ROUTING_SOURCE).read_bytes()
    want = {"pass1_npv", "pass1_leakage_share", "pass2_precision", "pass1_rate"}
    got: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for r in csv.DictReader(io.StringIO(data.decode("utf-8"))):
        if (r["package"] == "complementarity_v1" and r["table"] == "metric_table" and r["fold"] == "all"
                and r["metric"] in want):
            got.setdefault(r["arm"], {})[r["metric"]] = (int(float(r["numerator"])), int(float(r["denominator"])))
    counts = {}
    for arm, m in got.items():
        rows, positives = m["pass1_rate"][1], m["pass1_leakage_share"][1]
        p1_rows, p1_pos = m["pass1_npv"][1], m["pass1_leakage_share"][0]
        if m["pass1_npv"][0] != p1_rows - p1_pos or m["pass1_rate"][0] != p1_rows:
            raise AssertionError(f"{arm}: the report's Pass 1 counts disagree")
        counts[arm] = {"rows": rows, "positives": positives, "pass1_rows": p1_rows, "pass1_positives": p1_pos,
                       "pass2_rows": m["pass2_precision"][1], "pass2_positives": m["pass2_precision"][0]}
    return {"source": STAGE2_ROUTING_SOURCE, "sha256": hashlib.sha256(data).hexdigest(), "counts": counts}


def stage2_prevalence_projection(prevalences: Mapping[str, float], *, routing=None,
                                 nominal_rows: int = 100_000) -> Dict[str, Any]:
    """
    What a pure prior shift does to the Stage 2 obligations of the frozen
    routing: each arm's class-conditional routing rates (training OOF) held
    fixed, the prevalence moved to each value. Pass 1 NPV falls and Pass 2
    precision rises with prevalence; the Pass 1 leakage share (a share of the
    positives) does not move. Projected counts on ``nominal_rows`` rows are
    observed with ``observe_routing`` against the frozen contract, so each row
    is reported exactly as PR 49 reports a window (status and signed shortfall).
    Also the prevalence band in which every obligation holds. A projection, not
    an observation: PR 49 observes the windows, where covariate shift changes
    the routing rates as well.
    """
    from shared.stage2.contract import load_frozen_contract
    from shared.stage2.obligations import RoutingCounts, observe_routing

    contract = load_frozen_contract()
    routing = routing or accepted_routing_counts()
    f_npv, c_leak, f_prec = contract.min_pass1_npv, contract.max_pass1_leakage_rate, contract.min_pass2_precision
    rows, bands = [], []
    for arm, c in routing["counts"].items():
        neg = c["rows"] - c["positives"]
        r1, r0 = c["pass1_positives"] / c["positives"], (c["pass1_rows"] - c["pass1_positives"]) / neg
        q1, q0 = c["pass2_positives"] / c["positives"], (c["pass2_rows"] - c["pass2_positives"]) / neg
        # NPV >= f  <=>  odds <= r0 (1 - f) / (f r1);  precision >= f  <=>  odds >= f q0 / ((1 - f) q1)
        hi_odds, lo_odds = r0 * (1 - f_npv) / (f_npv * r1), f_prec * q0 / ((1 - f_prec) * q1)
        bands.append({"arm": arm, "pass1_positive_rate": r1, "pass1_negative_rate": r0,
                      "pass2_positive_rate": q1, "pass2_negative_rate": q0,
                      "npv_holds_up_to_prevalence": hi_odds / (1 + hi_odds),
                      "precision_holds_from_prevalence": lo_odds / (1 + lo_odds),
                      "leakage_share_under_prior_shift": r1, "leakage_holds": r1 <= c_leak,
                      "band_nonempty": lo_odds <= hi_odds and r1 <= c_leak})
        for label, pi in [("training OOF (observed)", c["positives"] / c["rows"])] + list(prevalences.items()):
            if label.startswith("training"):
                rc = RoutingCounts(**c)
            else:
                pos = int(round(nominal_rows * pi))
                p1p, p2p = int(round(pos * r1)), int(round(pos * q1))
                p1n, p2n = int(round((nominal_rows - pos) * r0)), int(round((nominal_rows - pos) * q0))
                rc = RoutingCounts(nominal_rows, pos, p1p + p1n, p1p, p2p + p2n, p2p)
            rep = observe_routing(rc, contract).to_dict()
            row = {"arm": arm, "population": label, "prevalence": rc.positives / rc.rows,
                   "basis": "observed" if label.startswith("training") else "projected (pure prior shift)"}
            for r in rep["results"]:
                row[f"{r['metric']}"] = r["value"]
                row[f"{r['metric']}_status"] = r["status"]
                row[f"{r['metric']}_shortfall"] = r["shortfall_float"]
            rows.append(row)
    return {"contract": {"id": contract.contract_id, "fingerprint": contract.fingerprint(),
                         "min_pass1_npv": f_npv, "max_pass1_leakage_rate": c_leak,
                         "min_pass2_precision": f_prec},
            "routing": routing, "nominal_rows": nominal_rows,
            "bands": pd.DataFrame(bands), "projection": pd.DataFrame(rows)}


def interim_finding(compared: pd.DataFrame, fact: Mapping[str, Mapping[str, Any]],
                    decisions: pd.DataFrame, projection: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The interim temporal-validity finding and what PR 43, PRs 44-45 and PR 49 rely on."""
    def pooled(question, arm):
        r = compared[(compared["question"] == question) & (compared["arm"] == arm)
                     & (compared["scope"] == "pooled")]
        if not len(r):
            return None
        r = r.iloc[0]
        return {"a": r["a"], "b": r["b"], "auc_a": round(float(r["auc_a"]), 4),
                "auc_b": round(float(r["auc_b"]), 4), "diff": round(float(r["diff"]), 4),
                "ci95": [round(float(r["ci_low"]), 4), round(float(r["ci_high"]), 4)], "rows": int(r["rows"])}

    def effects(arm):
        f = fact.get(arm) or {}
        e = f.get("effects")
        if e is None or not len(e):
            return None
        return {r["effect"]: {"pooled": round(float(r["pooled_diff"]), 4),
                              "pooled_ci95": [round(float(r["pooled_ci_low"]), 4),
                                              round(float(r["pooled_ci_high"]), 4)],
                              "mean_window": round(float(r["mean_window_diff"]), 4),
                              "mean_window_ci95": [round(float(r["mean_window_ci_low"]), 4),
                                                   round(float(r["mean_window_ci_high"]), 4)]}
                for _, r in e.iterrows()}

    def resolution(arm):
        f = fact.get(arm) or {}
        e = f.get("effects")
        if e is None or not len(e):
            return None
        out = {}
        for _, r in e.iterrows():
            for scope, lo, hi in (("pooled", "pooled_ci_low", "pooled_ci_high"),
                                  ("mean_window", "mean_window_ci_low", "mean_window_ci_high")):
                out[f"{r['effect']} [{scope}]"] = (
                    "resolved" if r[lo] > 0 or r[hi] < 0 else
                    f"not resolved: the 95% interval [{r[lo]:.4f}, {r[hi]:.4f}] includes 0")
        return out

    return {
        "finding": ("The accepted Stage 1 target encodings are fold-isolated but look ahead in time "
                    "(Step 3A). Their past-only versions are realizable, verified equal to the "
                    "production encoder fitted on the available rows, and the primary rule encodes "
                    "all but the cold-start rows. The differences between the accepted and the "
                    "past-only protocols, and between the random and the chronological splits, are "
                    "measured on identical rows below. Each is a difference of whole protocols; the "
                    "Stage 2-3 families are extended by construction, not by experiment"),
        "interpretation": {
            "contrasts": ("The past-only protocol differs from the accepted one in several ways at "
                          "once: it removes same-day and later labels, removes each training row's own "
                          "label from its encoding, fits edges, rates and fallbacks on fewer rows and "
                          "leaves the cold-start rows out of its fits. A difference between the two is "
                          "consistent with a contribution from look-ahead, which Step 3A found in the "
                          "accepted encodings, but this experiment does not isolate it from the other "
                          "changes. An interval that includes 0 leaves that difference unresolved; it "
                          "does not show that the difference is absent"),
            "resolution": {"lr": resolution("lr"), "mlp_isolated_early_stopping": resolution("mlp")},
            "lags": ("The past_updating lag arms keep the lag-0 model, trained on every pre-cutoff "
                     "outcome, and slow only its prediction-time encodings: what a lag-0 model loses "
                     "when its incoming features update more slowly. The lag_consistent arms fit the "
                     "scaler and the model on outcomes known at the cutoff under the lag, with "
                     "lag-consistent encodings: the cost of operating under that delay throughout"),
        },
        "lr": {q: pooled(q, "lr") for q, arm, *_ in COMPARISONS if arm == "lr"},
        "mlp": {q: pooled(q, "mlp") for q, arm, *_ in COMPARISONS if arm == "mlp"},
        "factorial": {"lr": effects("lr"), "mlp_isolated_early_stopping": effects("mlp")},
        "pr43": ("PR 43 may use the frozen PR 39 training-side OOF evidence as random-split evidence: "
                 "fold-isolated (Step 3A mechanism b absent) but with every target-derived feature "
                 "looking ahead in time, and the MLP's early stopping not encoder-isolated. It may "
                 "not present that evidence, or Stage 4 conclusions built on it, as forward-in-time "
                 "performance"),
        "pr44_45": ("PRs 44-45 need the versioned candidate contract (distinct feature names, the "
                    "contact-day time index as a pipeline input, the registered cold start), the "
                    "isolated early-stopping procedure as an MLP training candidate, past-only "
                    "Stage 2-3 constructions (not built here), and adoption criteria that include "
                    "a chronological comparison, not the random split alone"),
        "pr49_open": ("the chronological cascade across Stages 1-4; prevalence-shift effects on "
                      "calibration, thresholds, abstention and the Stage 2 obligations; past-only "
                      "NSD and ECI; the outcome-lag sensitivity at cascade level; whether exposed "
                      "test rows join"),
        "contracts": decisions[["candidate", "needs_versioned_contract", "destination"]].to_dict("records"),
        "stage2_prior_shift": (None if projection is None else
                               projection["bands"][["arm", "precision_holds_from_prevalence",
                                                    "npv_holds_up_to_prevalence", "leakage_holds",
                                                    "band_nonempty"]].to_dict("records")),
    }


# ----------------------------------------------------------------------
# the accepted Step 3A record
# ----------------------------------------------------------------------
#: The Windows Step 3A run accepted locally: run id and its two records' content digests.
STEP3A_RECORD = {
    "run_id": "c3_20261122_201830",
    "feature_inventory": "ec6e2201e3ffcd85a3b35d57fbdb48268283bc67b8e2a1d974347a98a651a55a",
    "encoding_audit": "9751f4affcf612dc2f364567e3aaa97da4f436a0b5bff45af4731e95ddb721cb",
}


#: The accepted Step 3A run's records. Its directory is either absent (the comparison is unavailable)
#: or must hold every one of them, each with its ``.sha256`` sidecar.
STEP3A_NAMES = ("feature_inventory", "encoding_audit")
#: How an absent accepted run directory is reported.
STEP3A_ABSENT = "not available locally (no accepted run directory)"

#: What must match for this run to be expected to reproduce the accepted Step 3A records exactly:
#: the interpreter, platform and package versions the records were written with, and which
#: accepted artifacts were available to the inventory (both shape the content).
ENVIRONMENT_KEYS = ("os", "platform", "python", "executable", "packages")


def _artifact_availability(inventory: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return sorted((str(r.get("role")), str(r.get("status"))) for r in inventory.get("accepted_artifacts") or [])


def step3a_reproduction(this_run: Mapping[str, Mapping[str, Any]], environment: Mapping[str, Any], *,
                        accepted: Mapping[str, str] = STEP3A_RECORD, root: Optional[Path] = None) -> pd.DataFrame:
    """
    This run's Step 3A records (``this_run``: name -> the record read back)
    against the accepted Step 3A run (``accepted``: run id and content digests).

    If the accepted run's directory is absent, the comparison is reported as
    unavailable. If it exists, it must be complete: every record in
    ``STEP3A_NAMES`` must be there with its ``.sha256`` sidecar, and a missing
    file fails that record's integrity in any environment. Each record present
    is checked on its own: its sidecar and content digest (``read_run_record``),
    and its content digest is the accepted one. Exact
    reproduction is required only when this run matches the accepted run's
    environment (``ENVIRONMENT_KEYS``) and the same accepted artifacts were
    available to both inventories; elsewhere (another platform, interpreter,
    package set or artifact set) it is reported, with the reason, and not
    required. Nothing is written or replaced here.
    """
    from feature_research import research_records as RR

    root = Path(root or RR.RECORD_ROOT)
    folder = root / accepted["run_id"]
    directory = folder.exists()
    acc_records: Dict[str, Optional[Dict[str, Any]]] = {}
    integrity: Dict[str, Tuple[str, bool]] = {}
    for name in STEP3A_NAMES:
        if not directory:
            acc_records[name], integrity[name] = None, (STEP3A_ABSENT, True)
            continue
        record, sidecar = folder / f"{name}.json", folder / f"{name}.json.sha256"
        missing = [p.name for p in (record, sidecar) if not p.is_file()]
        if missing:
            acc_records[name], integrity[name] = None, (
                f"integrity failed: the accepted run directory {folder.name} is incomplete "
                f"(missing {', '.join(missing)})", False)
            continue
        try:
            rec = RR.read_run_record(accepted["run_id"], name, root=root)
        except (RR.RecordError, OSError, ValueError) as exc:
            acc_records[name], integrity[name] = None, (f"integrity failed: {exc}", False)
            continue
        acc_records[name] = rec
        ok = rec["content_sha256"] == accepted[name]
        integrity[name] = (("verified (sidecar, content digest, accepted digest)" if ok else
                            f"content digest {rec['content_sha256']} is not the accepted {accepted[name]}"), ok)
    reasons = []
    ref = next((r for r in acc_records.values() if r is not None), None)
    if ref is not None:
        env_a = ref.get("environment") or {}
        reasons += [f"{k} differs" for k in ENVIRONMENT_KEYS if env_a.get(k) != environment.get(k)]
        inv_a, inv_b = acc_records.get("feature_inventory"), this_run.get("feature_inventory")
        if inv_a is None or inv_b is None or _artifact_availability(inv_a) != _artifact_availability(inv_b):
            reasons.append("accepted artifact availability differs")
    out = []
    for name in STEP3A_NAMES:
        status, ok = integrity[name]
        present = directory
        required = present and ok and not reasons
        got = this_run[name]["content_sha256"]
        out.append({"record": f"{accepted['run_id']}/{name}", "accepted_content_sha256": accepted[name],
                    "accepted_record": status, "integrity_ok": ok,
                    "this_run_content_sha256": got, "reproduced": got == accepted[name],
                    "reproduction_required": required,
                    "reproduction_basis": ("matching environment and artifacts" if required else
                                           "accepted run directory absent" if not present else
                                           "integrity failed" if not ok else "; ".join(reasons))})
    return pd.DataFrame(out)


def step3a_checks(table: pd.DataFrame) -> pd.DataFrame:
    """
    The applicable checks: the accepted records' integrity wherever the
    accepted run directory exists (a missing record or sidecar fails); exact
    reproduction only where ``step3a_reproduction`` requires it.
    """
    out = []
    for _, r in table.iterrows():
        present = r["accepted_record"] != STEP3A_ABSENT
        out.append(_check(f"{r['record']}: integrity" + ("" if present else " (accepted run directory absent)"),
                          bool(r["integrity_ok"]), r["accepted_record"]))
        if r["reproduction_required"]:
            out.append(_check(f"{r['record']}: reproduced exactly by this run (matching environment)",
                              bool(r["reproduced"]), f"this run {r['this_run_content_sha256']}"))
        else:
            out.append(_check(f"{r['record']}: exact reproduction not required ({r['reproduction_basis']})",
                              True, f"reproduced {bool(r['reproduced'])}"))
    return pd.DataFrame(out)
