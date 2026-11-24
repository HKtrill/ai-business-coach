"""
feature_research.interaction_selection — PR 42 Commit 5, Step 5B: procedural EBM interaction selection
======================================================================================================
Turns the Stage 3 EBM's interaction choice into a procedure that is fitted inside every training fit, then asks
(1) whether it recovers PR 39's three accepted pairs on the frozen 13 features, on all training rows and inside
each outer fold, and (2) what it selects under the same three-pair budget for the two Step 5A candidates carried
forward (``jed_alone`` and ``drop_first``), compared with each candidate's own 5A fixed-pair fit on the same
held-out rows.

The selector is interpret's own (FAST, Lou et al. 2013): the accepted Stage 3 EBM with ``interactions=3`` in
place of an explicit pair list. Inside one fit, each of the EBM's 14 outer bags boosts the main effects, then
ranks every pair of the fit's features by FAST strength on that bag's training rows; the pairs with the smallest
mean rank across bags are boosted, ties broken by column order. Every outer, inner and all-rows fit therefore
selects from its own rows only. A recorder observes the per-bag rankings (it does not change the fit), so every
eligible pair's mean rank is reported and checked against the pairs the fitted model holds.

Everything else is Step 5A's: features, rows, folds, seeds, EBM settings, calibration and operating points
(``SS.fit_configuration`` with a selecting factory). Budgets other than three belong to PR 44, Step 44C.

Two kinds of check close the step (Cells 67-68), and the record is written before either is enforced, so a run's
evidence is kept whatever they find:
- *Acceptance* holds in every run: fit isolation, every selection's ranking agrees with the fitted pair terms,
  paired comparisons use identical rows, the accepted Step 5A record is reproduced where Step 3B's rule requires
  it, and the materiality and cascade verdicts reported are the registered rules applied to this run's numbers.
- *Same-sandbox reproduction* compares this run with the sandbox run that the review document quotes
  (``SANDBOX_REFERENCE``): every selection, margin, count, estimate and verdict, with the difference. It is
  required only in that sandbox's environment. Elsewhere (the Windows acceptance run) the differences are
  recorded, and a changed selection, a change of direction below the materiality threshold, a changed verdict
  or a review-document statement that no longer holds is flagged for review, not failed.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_research import binning_drift as BD          # Step 4A's pinned exploratory notebook
from feature_research import stage3_simplification as SS
from glass_pipeline.ebm.estimator import EBMFactory

BANK_ROOT = Path(__file__).resolve().parents[1]
SEED = SS.SEED
N_FOLDS = SS.N_FOLDS
BUDGET = 3
CANDIDATES = ("jed_alone", "drop_first")
ACCEPTED_PAIRS = tuple(SS.BASELINE_PAIRS)
RECURRING_MIN = 3                       # selected in at least 3 of the 5 outer fits
MATERIAL = {"roc_auc": 0.002, "pr_auc": 0.005, "nested_f2": 0.005, "brier_calibrated": 0.0005}

#: PR 39's research round: the printed setting and selected pairs of the exploratory notebook's Cells 16A/16B,
#: read from Step 4A's verified source (``BD.EXPLORATORY``: the content-addressed Git blob, never whatever the
#: working copy holds). Cells are found by their header lines.
RESEARCH_ROUND = {
    "notebook": BD.EXPLORATORY["notebook"],
    "cells": {"16A": "# CELL 16A: STAGE 3 EBM — TRAINING (OUT-OF-FOLD)",
              "16B": "# CELL 16B: STAGE 3 EBM — DIAGNOSTICS (IN-SAMPLE, NO REFIT)"},
    "setting": "interactions 4",
    "pairs": [("decay_x_density", "economic_curvature_intensity", 0.0579),
              ("campaign", "emp_var_rate_sigmoid_slope", 0.0529),
              ("economic_curvature_intensity", "dow_month_encoded", 0.0448),
              ("decay_x_density", "emp_var_rate_sigmoid_slope", 0.0095)],
    "main_effects": 14,
}

# ======================================================================
# 1. registration (fixed before any Step 5B selection was run on the training rows)
# ======================================================================
REGISTRATION: Dict[str, Any] = {
    "id": "pr42-step5b-registration",
    "version": "1.0.0",
    "question": [
        "On the frozen PR 39 Stage 3 inventory of 13 features, does a reproducible selection procedure recover "
        "the three accepted pairs when fitted on all training rows, and which pairs recur when selection is "
        "repeated independently inside each outer fold?",
        "For jed_alone and drop_first, what does reselection under the existing three-pair budget recover, and "
        "how does each reselected fit compare with its own Step 5A fixed-pair fit on identical held-out rows?",
    ],
    "known_before_registration": [
        "Step 5A's registered results for every configuration (fixed pairs: jed_alone keeps campaign x "
        "emp_var_rate_sigmoid_slope and euribor3m_sigmoid_slope x economic_curvature_intensity; drop_first "
        "keeps campaign x emp_var_rate_sigmoid_slope and prior_x_stress x decay_x_density), including the "
        "support-defined pair surfaces.",
        "The accepted Windows Step 5A run's complementarity: on Stage 2 abstentions the refitted baseline nets "
        "530, jed_alone 486 (meets the cascade screen), drop_first 441 (misses it; needed 456.86).",
        "PR 39's research round (exploratory notebook Cells 16A/16B): interpret's FAST with interactions 4 on "
        "14 main effects selected decay_x_density x economic_curvature_intensity, campaign x "
        "emp_var_rate_sigmoid_slope, economic_curvature_intensity x dow_month_encoded and decay_x_density x "
        "emp_var_rate_sigmoid_slope.",
        "No Step 5B selection had been run on the training rows.",
    ],
    "history": {
        "reproducible": "the selection mechanism: interpret's built-in FAST ranking (the research round set "
                        "interactions 4, tuned by Optuna between 0 and 5)",
        "not_reproduced": [
            "the research round itself: 14 main effects, an earlier pipeline and other hyperparameters "
            "(learning_rate 0.005384, max_rounds 1300, max_bins 416, max_interaction_bins 128)",
            "the TIER 1 'importance=0.205' in glass_pipeline/ebm/interactions.py: no tracked output gives it "
            "(the research round prints 0.0529 for that pair)",
            "the two TIER 2 pairs: chosen as 'exploratory, consistent with research findings', not by the "
            "ranking (neither is among the research round's four)",
        ],
        "judgement_calls_made_rules": [
            "the budget is the existing three pairs (other budgets: PR 44, Step 44C)",
            "the ranking is interpret's own, unmodified: no tiers, no manual exclusions, no hand-picked pairs",
            "every fit selects for itself: there is no consensus or frequency step across folds",
        ],
    },
    "selector": {
        "implementation": "the accepted Stage 3 EBM (glass_pipeline.ebm.estimator.EBMFactory settings, "
                          "interpret 0.7.8) with interactions=3 in place of an explicit pair list",
        "ranking": "in each of the 14 outer bags, after boosting the main effects, FAST ranks every eligible "
                   "pair by interaction strength on the bag's training rows (the fit's balanced weights apply); "
                   "a pair's score is its mean rank across the 14 bags",
        "selection": "the 3 pairs with the smallest mean rank",
        "ties": "equal mean ranks are broken by the pair's column indices in the fit's feature order "
                "(interpret's rule: smallest first)",
        "settings": "Step 5A's PARAMS (learning_rate, max_rounds 2100, max_bins 480, max_interaction_bins 80), "
                    "random_state 42, n_jobs 1, outer_bags 14",
        "observation": "a recorder captures each bag's ranking without changing the fit; the recorded top 3 "
                       "must equal the fitted model's pair terms",
    },
    "eligible_pairs": "every unordered pair of the configuration's features, with no exclusion: 78 for the "
                      "13 frozen features and for jed_alone (13), 66 for drop_first (12); campaign x "
                      "emp_var_rate_sigmoid_slope is eligible wherever both features are present",
    "where_selection_runs": {
        "recovery": "the frozen 13 features: once on all 32,950 training rows (descriptive) and once inside "
                    "each outer fold's 26,360 fit rows",
        "candidates": "every fit of jed_alone and drop_first: 5 outer fits, the 25 inner fits whose scores "
                      "feed calibration and the fold-nested thresholds, and the all-rows fit for the surfaces",
        "rows": "a fit selects on its own fit rows only; held-out rows never enter",
    },
    "comparisons": {
        "recovery": "how many of the three accepted pairs the all-rows selection contains; each eligible "
                    "pair's outer-fold selection count (0-5) and mean rank",
        "recurrence": "stable: 5 of 5 outer fits; recurring: at least 3; occasional: 1-2; never: 0. Inner "
                      "fits (25) are reported as supporting counts",
        "change_against_5a": "against the candidate's 5A pairs: a 5A pair persists if selected in at least 3 "
                             "outer fits and disappears otherwise; a new pair enters if selected in at least 3, "
                             "and is an occasional entrant if selected in 1-2",
        "performance": "each reselected candidate against its own 5A fixed-pair fit on the same out-of-fold "
                       "rows: pooled ROC-AUC (DeLong paired difference, 95% CI), PR-AUC, nested F2, Brier and "
                       "ECE (raw and calibrated, 10 bins), per-fold differences (mean, sd, range, folds better) "
                       "and changed decisions; the pair count differs (5A: 2, reselected: 3), so a difference "
                       "belongs to the whole procedure",
        "material": "described as a material change if the ROC-AUC CI excludes 0 or |dROC-AUC| >= 0.002, "
                    "|dPR-AUC| >= 0.005, |dnested F2| >= 0.005, or |dcalibrated Brier| >= 0.0005; descriptive "
                    "thresholds, not an adoption rule",
        "surfaces": "every pair that recurs (>= 3 outer fits) in any configuration and every 5A pair that "
                    "disappears, on Step 5A's support-defined cells, matched axes and one symmetric colour "
                    "scale per pair across the models compared",
        "cascade_screen": "Step 5A's rule, unchanged, for the 5A and reselected fits (Windows only): within "
                          "Step 5A's performance tolerances against the 5A refitted baseline, and Stage 2 "
                          "abstention net >= the baseline's net - 0.5% of the abstention rows; a changed verdict "
                          "is reported as such, not as evidence of a better cascade",
    },
    "out_of_scope": ["other interaction budgets (PR 44, Step 44C)", "cascade comparisons (PRs 44-45)",
                     "chronological evaluation (PR 49)", "any change to the frozen PR 39 artifacts"],
}

REGISTERED_SHA256 = "9d18391acf8ab38c3d60ca59a86b6a9afaaff60e8f0e5d7d67b4a7137cf321ba"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def registration_digest() -> str:
    return hashlib.sha256(_canonical(REGISTRATION).encode("utf-8")).hexdigest()


def _check(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def registration_checks(root: Path = BANK_ROOT, *, history: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """The registration's digest, and the history it relies on read back from the tracked sources (the
    exploratory notebook from Step 4A's pinned blob). A pinned source that is unavailable here (a shallow clone
    whose working copy is not the pinned notebook) is reported as not checked; a mismatch fails."""
    prov = history if history is not None else historical_provenance(root)
    src = prov["source"]
    out = [_check("the Step 5B registration matches its registered digest", registration_digest() == REGISTERED_SHA256,
                  registration_digest()[:16]),
           _check("the accepted pairs are define_ebm_interactions' three", list(ACCEPTED_PAIRS) == [
               ("campaign", "emp_var_rate_sigmoid_slope"), ("euribor3m_sigmoid_slope", "economic_curvature_intensity"),
               ("prior_x_stress", "decay_x_density")], len(ACCEPTED_PAIRS)),
           _check(f"history source: Step 4A's pinned exploratory notebook (blob {src['blob'][:12]}, LF SHA-256 "
                  f"{src['sha256_lf'][:12]}) is {src['status']}" + (f" from the {src['source']}" if src["source"] else ""),
                  src["status"] != "mismatch", src["detail"])]
    out += [_check(f"history: {r['statement']}" + ("" if r["checked"] else " (not checked: source unavailable)"),
                   r["verified"], r["detail"]) for r in prov["statements"]]
    return pd.DataFrame(out)


def _outputs_text(cell: Mapping[str, Any]) -> str:
    """A code cell's printed output: streams and the plain-text form of results and displays."""
    s = ""
    for o in cell.get("outputs", []):
        s += "".join(o.get("text", "")) if o.get("output_type") == "stream" else "".join(
            o.get("data", {}).get("text/plain", ""))
    return s


def _one_cell(nb: Mapping[str, Any], header: str) -> Mapping[str, Any]:
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and header in "".join(c["source"]).splitlines()]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} code cells have the header {header!r}")
    return hits[0]


def historical_provenance(root: Path = BANK_ROOT, *, working_copy: Optional[Path] = None) -> Dict[str, Any]:
    """
    What the tracked sources show about how PR 39 chose its pairs. The exploratory notebook is Step 4A's
    verified source (``BD._pinned_exploratory``: the pinned Git blob, or the working copy only where Git lacks
    the blob and the working copy is byte-for-byte the pinned notebook), so clearing the working copy's outputs
    changes nothing here. ``source`` records the identity read (commit, blob, LF SHA-256, where from, status);
    each statement records whether it was checked. The statement about 'importance=0.205' covers only the
    outputs inspected: every printed output of the pinned notebook (Cells 16A and 16B among them).
    """
    root = Path(root)
    statements = []
    txt = (root / "glass_pipeline/ebm/interactions.py").read_text(encoding="utf-8")
    statements.append({
        "statement": "glass_pipeline/ebm/interactions.py labels one TIER 1 and two TIER 2 (exploratory) pairs; its "
                     "TIER 1 comment gives 'importance=0.205'",
        "checked": True, "verified": "TIER 1 (importance=0.205, research confirmed)" in txt and txt.count("TIER 2") >= 2,
        "detail": "TIER 1 campaign x emp_var_rate_sigmoid_slope; TIER 2 'exploratory'"})
    data, src = BD._pinned_exploratory(root, working_copy)
    source = {k: src.get(k) for k in ("commit", "blob", "path", "sha256_lf", "status", "source", "detail")}
    named = ["the pinned notebook holds Cells 16A and 16B, each once, with printed outputs",
             "Cell 16A printed 'interactions 4' (interpret's FAST)",
             "Cell 16B printed the research round's four pairs with their importances",
             "'0.205' appears in none of the pinned notebook's printed outputs (scope: that notebook only)"]
    if data is None:
        statements += [{"statement": s, "checked": False, "verified": src["status"] != "mismatch",
                        "detail": src["detail"]} for s in named]
        return {"source": source, "statements": statements}
    nb = json.loads(data.decode("utf-8"))
    try:
        o16a, o16b = (_outputs_text(_one_cell(nb, RESEARCH_ROUND["cells"][k])) for k in ("16A", "16B"))
        found_cells = bool(o16a) and bool(o16b)
    except ValueError as exc:
        o16a = o16b = ""
        found_cells = False
        source["cells"] = str(exc)
    pairs_line = next((ln for ln in o16b.splitlines() if ln.startswith("Pairs")), "")
    want = [f"{a} & {b} ({v:.4f})" for a, b, v in RESEARCH_ROUND["pairs"]]
    printed = [(i, _outputs_text(c)) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    printed = [(i, t) for i, t in printed if t]
    hits = [i for i, t in printed if "0.205" in t]
    statements += [
        {"statement": named[0], "checked": True, "verified": found_cells,
         "detail": {k: RESEARCH_ROUND["cells"][k] for k in ("16A", "16B")}},
        {"statement": named[1], "checked": True, "verified": "interactions 4" in o16a, "detail": RESEARCH_ROUND["setting"]},
        {"statement": named[2], "checked": True, "verified": bool(pairs_line) and all(w in pairs_line for w in want),
         "detail": pairs_line[:200]},
        {"statement": named[3], "checked": True, "verified": not hits,
         "detail": (f"inspected: the printed outputs of all {len(printed)} code cells with output in blob "
                    f"{source['blob'][:12]}; Cell 16B prints 0.0529 for campaign x emp_var_rate_sigmoid_slope. Not "
                    "inspected: other notebooks, logs or commits" if not hits else f"found in notebook cells {hits}")},
    ]
    source["code_cells_with_output"] = len(printed)
    return {"source": source, "statements": statements}


#: The accepted Windows Step 5A run (the replacement patch with its follow-up): Cell 57 9/9, Cell 58 100/100.
ACCEPTED_5A = {"run_id": "c5_20261124_002645", "record": "stage3_simplification",
               "content_sha256": "d70330e1d8fa37de5cc7288f80b2d954d510b8579a0619ee8f32f6653e83f033"}


def accepted_5a_reproduction(this_record: Mapping[str, Any], environment: Mapping[str, Any],
                             views: Mapping[str, Any], *, root: Optional[Path] = None,
                             accepted: Mapping[str, str] = ACCEPTED_5A) -> pd.DataFrame:
    """
    This run's Step 5A record (Cell 59) against the accepted Windows Step 5A run, by Step 3B's rule
    (``TV.step3a_reproduction``): wherever the accepted run's directory exists, its record must pass its sidecar
    and content digest and be the accepted digest; exact reproduction is required only where this run matches
    the accepted run's environment (``TV.ENVIRONMENT_KEYS``) and the same accepted views were available to both.
    In the acceptance run (accepted views required) the accepted directory must be present. The content digest
    leaves out run, seeds, code and environment, so the Step 5B code does not move it.
    """
    from feature_research import research_records as RR
    from feature_research import temporal_validity as TV
    root = Path(root or RR.RECORD_ROOT)
    folder = root / accepted["run_id"]
    acc = None
    if folder.exists():
        try:
            acc = RR.read_run_record(accepted["run_id"], accepted["record"], root=root)
        except (RR.RecordError, OSError, ValueError) as exc:
            status, ok = f"integrity failed: {exc}", False
        else:
            ok = acc["content_sha256"] == accepted["content_sha256"]
            status = ("verified (sidecar, content digest, accepted digest)" if ok else
                      f"content digest {acc['content_sha256']} is not the accepted {accepted['content_sha256']}")
    elif views.get("required"):
        status, ok = "integrity failed: the accepted run directory is absent in the acceptance run", False
    else:
        status, ok = "accepted run directory absent", True
    reasons = []
    if acc is not None:
        env_a = acc.get("environment") or {}
        reasons += [f"{k} differs" for k in TV.ENVIRONMENT_KEYS if env_a.get(k) != environment.get(k)]
        if ((acc.get("complementarity") or {}).get("views") or {}).get("status") != views.get("status"):
            reasons.append("accepted view availability differs")
    required = acc is not None and ok and not reasons
    got = this_record["content_sha256"]
    return pd.DataFrame([{
        "record": f"{accepted['run_id']}/{accepted['record']}", "accepted_content_sha256": accepted["content_sha256"],
        "accepted_record": status, "integrity_ok": ok, "this_run_content_sha256": got,
        "reproduced": got == accepted["content_sha256"], "reproduction_required": required,
        "reproduction_basis": ("matching environment and views" if required else
                               "accepted run directory absent" if acc is None and ok else
                               "integrity failed" if not ok else "; ".join(reasons))}])


def accepted_5a_checks(table: pd.DataFrame) -> pd.DataFrame:
    """Integrity wherever the accepted directory exists (or must); exact reproduction only where required."""
    r = table.iloc[0]
    absent = r["accepted_record"] == "accepted run directory absent"
    out = [_check(f"accepted Step 5A record {r['record']}: integrity" + (" (accepted run directory absent)" if absent
                                                                        else ""), bool(r["integrity_ok"]),
                  r["accepted_record"])]
    if r["reproduction_required"]:
        out.append(_check(f"{r['record']}: reproduced exactly by this run's Step 5A (matching environment and views)",
                          bool(r["reproduced"]), f"this run {r['this_run_content_sha256']}"))
    else:
        out.append(_check(f"{r['record']}: exact reproduction not required ({r['reproduction_basis']})", True,
                          f"reproduced {bool(r['reproduced'])}"))
    return pd.DataFrame(out)


# ======================================================================
# 2. the selector
# ======================================================================
@contextmanager
def _ranking_recorder(store: list):
    """Observe interpret's per-bag FAST rankings during one fit (n_jobs=1 runs the bags in this process)."""
    import interpret.glassbox._ebm._ebm as E
    original = E.rank_interactions

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if not isinstance(out, Exception):
            store.append([(float(s), tuple(int(i) for i in idx)) for s, idx in out])
        return out

    E.rank_interactions = wrapped
    try:
        yield store
    finally:
        E.rank_interactions = original


def mean_ranks(bags: Sequence[Sequence[Tuple[float, Tuple[int, int]]]]) -> Dict[Tuple[int, int], float]:
    """Each pair's mean rank across bags, accumulated exactly as interpret does (running mean, bag order)."""
    ranks: Dict[Tuple[int, int], float] = {}
    for n, ranked in enumerate(bags):
        for rank, (_, idx) in enumerate(ranked):
            old = ranks.get(idx, 0)
            ranks[idx] = old + (rank - old) / (n + 1)
    return ranks


def pair_label(a: str, b: str) -> str:
    """'a x b', in the accepted orientation for an accepted pair, otherwise as given (feature order)."""
    if (b, a) in ACCEPTED_PAIRS:
        a, b = b, a
    return f"{a} x {b}"


def _index_sha(index) -> str:
    return hashlib.sha256(np.sort(np.asarray(index, dtype=np.int64)).tobytes()).hexdigest()


def _selection_record(model, X: pd.DataFrame, bags, budget: int) -> Dict[str, Any]:
    feats = list(X.columns)
    ranks = mean_ranks(bags)
    order = heapq.nsmallest(len(ranks), ranks, key=lambda i: (ranks[i], i))
    top = order[:budget]
    fitted = []
    for t in model.term_names_[len(feats):]:
        a, b = t.split(" & ")
        fitted.append(tuple(sorted((feats.index(a), feats.index(b)))))
    per_bag = {}
    for ranked in bags:
        for r, (s, idx) in enumerate(ranked):
            per_bag.setdefault(idx, []).append((r, s))
    ranking = []
    for pos, idx in enumerate(order, start=1):
        rs = np.array([r for r, _ in per_bag[idx]], dtype=float)
        ss = np.array([s for _, s in per_bag[idx]], dtype=float)
        ranking.append({"pair": pair_label(feats[idx[0]], feats[idx[1]]), "position": pos,
                        "mean_rank": float(ranks[idx]), "rank_sd": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
                        "mean_strength": float(ss.mean()), "selected": idx in top})
    return {"rows": int(len(X)), "index_sha256": _index_sha(X.index), "features": feats, "bags": len(bags),
            "eligible_per_bag": sorted({len(r) for r in bags}),
            "selected": [pair_label(feats[i], feats[j]) for i, j in top],
            "fitted": sorted(pair_label(feats[i], feats[j]) for i, j in fitted),
            "selection_matches_fit": sorted(top) == sorted(fitted),
            "margin": (ranking[budget]["mean_rank"] - ranking[budget - 1]["mean_rank"]) if len(ranking) > budget else None,
            "ranking": ranking}


class ProceduralEBMFactory(EBMFactory):
    """The Stage 3 EBM factory with ``interactions=budget``: each fit selects its own pairs (interpret's FAST)
    from its own rows. Records every fit's selection, in call order."""

    def __init__(self, budget: int = BUDGET, random_state: int = SEED, n_jobs: int = 1):
        super().__init__(interactions=[], random_state=random_state, n_jobs=n_jobs)
        self.budget = int(budget)
        self.selections: List[Dict[str, Any]] = []

    def build(self, params):
        model = super().build(params)
        model.set_params(interactions=self.budget)
        return model

    def fit(self, params, X: pd.DataFrame, y, sample_weight=None):
        bags: list = []
        with _ranking_recorder(bags):
            model = super().fit(params, X, y, sample_weight=sample_weight)
        self.selections.append(_selection_record(model, X, bags, self.budget))
        return model

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["interactions"] = self.budget
        d["selection"] = REGISTRATION["selector"]["selection"]
        return d


def select_pairs(X: pd.DataFrame, y, sample_weight=None, *, params: Optional[Mapping[str, Any]] = None,
                 budget: int = BUDGET) -> Dict[str, Any]:
    """One selection on exactly the rows given (the fitted model is kept under ``model``)."""
    fac = ProceduralEBMFactory(budget)
    model = fac.fit(dict(params or SS.PARAMS), X, y, sample_weight=sample_weight)
    return {**fac.selections[0], "model": model}


# ======================================================================
# 3. where each selection ran
# ======================================================================
def fit_frame_index(frames: Mapping[str, Any]) -> Dict[str, str]:
    """index fingerprint -> label for every fit frame: 'outer k', 'inner k.j', 'all rows'."""
    out = {_index_sha(frames["X_all"].index): "all rows"}
    for k, f in enumerate(frames["outer"]):
        out[_index_sha(f.X_tr.index)] = f"outer {k}"
    for k, inner in enumerate(frames["inner"]):
        for j, f in enumerate(inner):
            out[_index_sha(f.X_tr.index)] = f"inner {k}.{j}"
    return out


def label_selections(selections: Sequence[Mapping[str, Any]], frames: Mapping[str, Any]) -> List[Dict[str, Any]]:
    lookup = fit_frame_index(frames)
    return [{**s, "fit": lookup.get(s["index_sha256"], "unmatched")} for s in selections]


def selection_checks(name: str, selections: Sequence[Mapping[str, Any]], n_features: int, *,
                     expect: Mapping[str, int]) -> pd.DataFrame:
    """Every selection: on exactly one fit frame's rows, all eligible pairs ranked in all 14 bags, and the
    recorded top three are the fitted model's pairs."""
    n_pairs = n_features * (n_features - 1) // 2
    fits = [s["fit"] for s in selections]
    kinds = {"outer": sum(f.startswith("outer") for f in fits), "inner": sum(f.startswith("inner") for f in fits),
             "all rows": fits.count("all rows")}
    return pd.DataFrame([
        _check(f"{name}: every selection ran on exactly one fit frame's rows (outer, inner or all rows)",
               "unmatched" not in fits and len(set(fits)) == len(fits), kinds),
        _check(f"{name}: selections per kind = {dict(expect)}", all(kinds[k] == v for k, v in expect.items()), kinds),
        _check(f"{name}: all {n_pairs} eligible pairs ranked in each of 14 bags",
               all(s["bags"] == 14 and s["eligible_per_bag"] == [n_pairs] for s in selections), n_pairs),
        _check(f"{name}: the recorded top {BUDGET} by mean rank are the fitted model's pairs",
               all(s["selection_matches_fit"] for s in selections), len(selections)),
    ])


# ======================================================================
# 4. recovery on the frozen 13 features
# ======================================================================
def recovery(block, frames: Mapping[str, Any]) -> Dict[str, Any]:
    """Selection on the frozen 13 features: inside each outer fold's fit rows (OOFGenerator, as every
    configuration is fitted) and once on all training rows (descriptive)."""
    from shared.stage3.evaluation import OOFGenerator
    feats = list(SS.BASELINE_FEATURES)
    fac = ProceduralEBMFactory()
    oof = OOFGenerator(fac, block.tune_plan).generate(SS._subset(frames["outer"], feats), SS.PARAMS, score_test=False)
    fac.fit(SS.PARAMS, frames["X_all"][feats], block.y_train, sample_weight=block.weights.vector)
    sel = label_selections(fac.selections, frames)
    return {"features": feats, "selections": sel, "oof_sha256": hashlib.sha256(
        np.round(oof.proba, 12).tobytes()).hexdigest()}


# ======================================================================
# 5. the candidates, reselected
# ======================================================================
def configurations(configs_5a: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """jed_alone and drop_first with their 5A features and a three-pair selection budget."""
    out = {}
    for name in CANDIDATES:
        c = configs_5a[name]
        out[f"{name}_selected"] = {"features": list(c["features"]), "pairs": None, "budget": BUDGET,
                                   "five_a": name, "five_a_pairs": [tuple(p) for p in c["pairs"]],
                                   "questions": [f"{name}, reselected under the three-pair budget"]}
    return out


def fit_selected(config: Mapping[str, Any], block, frames: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 5A's protocol (SS.fit_configuration) with a factory that selects inside every fit."""
    fac = ProceduralEBMFactory()
    fit = SS.fit_configuration(config, block, frames, factory=fac)
    fit["selections"] = label_selections(fac.selections, frames)
    return fit


def recovery_summary(rec: Mapping[str, Any], table: pd.DataFrame) -> Dict[str, Any]:
    """The recovery question in one place: the all-rows selection, how many accepted pairs it holds, and each
    accepted pair's outer-fold count."""
    full = next(s for s in rec["selections"] if s["fit"] == "all rows")
    acc = [pair_label(a, b) for a, b in ACCEPTED_PAIRS]
    t = table.set_index("pair")
    return {"all_rows_selected": list(full["selected"]),
            "accepted_recovered_all_rows": [p for p in acc if p in full["selected"]],
            "accepted_outer_counts": {p: int(t.loc[p, "outer_selected"]) for p in acc},
            "accepted_all_rows_positions": {p: int(t.loc[p, "all_rows_position"]) for p in acc},
            "recurring": t.index[t["outer_selected"] >= RECURRING_MIN].tolist()}


# ======================================================================
# 6. recurrence and change
# ======================================================================
def _status(outer: int) -> str:
    return "stable" if outer == N_FOLDS else "recurring" if outer >= RECURRING_MIN else (
        "occasional" if outer > 0 else "never")


def recurrence(selections: Sequence[Mapping[str, Any]], reference: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """Per eligible pair: outer-fold and inner-fold selection counts, all-rows selection, mean rank and its
    spread over the outer fits, recurrence class, and its change against ``reference`` (accepted or 5A pairs)."""
    ref = {pair_label(a, b) for a, b in reference}
    outer = [s for s in selections if s["fit"].startswith("outer")]
    inner = [s for s in selections if s["fit"].startswith("inner")]
    full = [s for s in selections if s["fit"] == "all rows"]
    pairs = [r["pair"] for r in (full or outer)[0]["ranking"]]
    ranks = {p: [] for p in pairs}
    for s in outer:
        for r in s["ranking"]:
            ranks[r["pair"]].append(r["mean_rank"])
    rows = []
    for p in pairs:
        n_out = sum(p in s["selected"] for s in outer)
        in_ref = p in ref
        change = ("persists" if n_out >= RECURRING_MIN else "disappears") if in_ref else (
            "enters" if n_out >= RECURRING_MIN else "occasional entrant" if n_out else "")
        all_rank = next((r for r in full[0]["ranking"] if r["pair"] == p), None) if full else None
        rows.append({"pair": p, "reference": in_ref, "outer_selected": n_out,
                     "inner_selected": sum(p in s["selected"] for s in inner) if inner else None,
                     "inner_fits": len(inner) or None,
                     "all_rows_selected": bool(full and p in full[0]["selected"]),
                     "all_rows_position": None if all_rank is None else all_rank["position"],
                     "outer_mean_rank": float(np.mean(ranks[p])), "outer_rank_sd": float(np.std(ranks[p], ddof=1)),
                     "recurrence": _status(n_out), "change": change})
    t = pd.DataFrame(rows)
    return t.sort_values(["outer_selected", "outer_mean_rank"], ascending=[False, True]).reset_index(drop=True)


def change_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The reference pairs and every pair any fit selected, per configuration, with its class and change."""
    rows = []
    for name, t in tables.items():
        keep = t[t["reference"] | (t["outer_selected"] > 0) | t["all_rows_selected"]]
        for _, r in keep.iterrows():
            rows.append({"configuration": name, "pair": r["pair"], "reference": bool(r["reference"]),
                         "outer_selected": int(r["outer_selected"]), "inner_selected": r["inner_selected"],
                         "all_rows_selected": bool(r["all_rows_selected"]), "recurrence": r["recurrence"],
                         "change": r["change"]})
    return pd.DataFrame(rows)


def selected_by_fit(selections: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The pairs each fit selected, in rank order."""
    return pd.DataFrame([{"fit": s["fit"], "rows": s["rows"], **{f"pair_{i + 1}": p for i, p in enumerate(s["selected"])},
                          "margin": s.get("margin")}
                         for s in sorted(selections, key=lambda s: _fit_order(s["fit"]))])


def _fit_order(fit: str):
    kind, _, rest = fit.partition(" ")
    rank = {"outer": 0, "inner": 1, "all": 2}[kind]
    return (rank, [float(x) for x in rest.split(".")] if kind != "all" else [0])


# ======================================================================
# 7. against each candidate's own 5A fit
# ======================================================================
def paired_against(reference: Mapping[str, Any], fit: Mapping[str, Any], y: pd.Series,
                   fold_id: np.ndarray) -> Dict[str, pd.DataFrame]:
    """``fit`` against ``reference`` on the same out-of-fold rows: pooled values and differences (ROC-AUC with
    DeLong's paired CI), per-fold values and differences, and changed decisions."""
    from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score
    from shared.metrics import calculate_ece
    from feature_research import temporal_validity as TV
    yv = y.astype(int).to_numpy()

    def pooled(r):
        dec = (r["proba"] >= r["nested"]["per_row"]).astype(int)
        return {"pr_auc": average_precision_score(yv, r["proba"]), "nested_f2": fbeta_score(yv, dec, beta=2.0),
                "brier_raw": brier_score_loss(yv, r["proba"]), "brier_calibrated": brier_score_loss(yv, r["calibrated"]),
                "ece_raw": calculate_ece(yv, r["proba"], 10), "ece_calibrated": calculate_ece(yv, r["calibrated"], 10)}, dec
    a, adec = pooled(reference)
    b, bdec = pooled(fit)
    dl = TV.delong_paired(yv, reference["proba"], fit["proba"])
    row = {"roc_auc_5a": dl["auc_a"], "roc_auc": dl["auc_b"], "d_roc_auc": dl["diff"], "d_roc_auc_low": dl["ci_low"],
           "d_roc_auc_high": dl["ci_high"], "d_roc_auc_p": dl["p_value"]}
    for k in a:
        row.update({f"{k}_5a": a[k], k: b[k], f"d_{k}": b[k] - a[k]})
    fa = SS._fold_values(yv, reference["proba"], reference["calibrated"], adec, fold_id).set_index("fold")
    fb = SS._fold_values(yv, fit["proba"], fit["calibrated"], bdec, fold_id).set_index("fold")
    spread = []
    for col in ("roc_auc", "pr_auc", "brier_calibrated", "ece_calibrated", "nested_f2"):
        d = fb[col] - fa[col]
        better = (d > 0) if col in ("roc_auc", "pr_auc", "nested_f2") else (d < 0)
        spread.append({"metric": col, "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "min": float(d.min()),
                       "max": float(d.max()), "folds_better": int(better.sum()), "by_fold": [float(v) for v in d]})
    decisions = {"differ": int((adec != bdec).sum()), "flag_now_not_5a": int(((bdec == 1) & (adec == 0)).sum()),
                 "flag_5a_not_now": int(((bdec == 0) & (adec == 1)).sum()), "flagged": int(bdec.sum()),
                 "flagged_5a": int(adec.sum())}
    return {"pooled": row, "fold_differences": pd.DataFrame(spread), "decisions": decisions}


def materiality(pooled: Mapping[str, float]) -> Dict[str, Any]:
    """The registered descriptive thresholds."""
    roc = (pooled["d_roc_auc_low"] > 0 or pooled["d_roc_auc_high"] < 0) or abs(pooled["d_roc_auc"]) >= MATERIAL["roc_auc"]
    flags = {"roc_auc": bool(roc), "pr_auc": abs(pooled["d_pr_auc"]) >= MATERIAL["pr_auc"],
             "nested_f2": abs(pooled["d_nested_f2"]) >= MATERIAL["nested_f2"],
             "brier_calibrated": abs(pooled["d_brier_calibrated"]) >= MATERIAL["brier_calibrated"]}
    gate = pooled["ece_calibrated"] <= SS.TOL["ece_gate"] and pooled["ece_calibrated_5a"] <= SS.TOL["ece_gate"]
    return {**{f"material_{k}": bool(v) for k, v in flags.items()}, "calibrated_ece_gate_ok": bool(gate),
            "material_change": bool(any(flags.values()))}


def comparison_table(comparisons: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, c in comparisons.items():
        rows.append({"configuration": name, **c["pooled"], **c["decisions"], **materiality(c["pooled"])})
    return pd.DataFrame(rows)


# ======================================================================
# 8. surfaces for the recurring and changed pairs
# ======================================================================
def pairs_to_show(tables: Mapping[str, pd.DataFrame]) -> List[str]:
    """Every pair recurring (>= 3 outer fits) in any configuration, and every reference pair that disappears."""
    out = []
    for t in tables.values():
        for _, r in t.iterrows():
            if (r["outer_selected"] >= RECURRING_MIN or r["change"] == "disappears") and r["pair"] not in out:
                out.append(r["pair"])
    return out


def _term_name(model, pair: str) -> Optional[str]:
    a, b = pair.split(" x ")
    names = list(model.term_names_)
    return f"{a} & {b}" if f"{a} & {b}" in names else f"{b} & {a}" if f"{b} & {a}" in names else None


def surfaces(models: Mapping[str, Mapping[str, Any]], X_all: pd.DataFrame, T: pd.DataFrame,
             pairs: Sequence[str]) -> List[Dict[str, Any]]:
    """
    For each pair and each model set that holds it in its all-rows fit: the pair term's mean centred
    contribution on Step 5A's support-defined cells (SS.pair_bins), with the per-cell sd across the outer-fold
    models that selected it, the joint support and the largest contact period's share. ``models`` maps a
    configuration name to {"features", "full", "folds"} (fixed-pair or reselected fits alike).

    ``fold_models`` is the number of outer folds whose model holds the pair (the selecting folds). With fewer
    than two, a fold sd does not exist: ``fold_sd`` is all NaN, ``fold_sd_available`` is False, and
    ``fold_sd_max_supported`` and ``strong_cells_sign_agreement`` are None (unavailable, never 0).
    """
    n = len(X_all)
    per = T["period"].to_numpy()
    out = []
    for pair in pairs:
        a, b = pair.split(" x ")
        if a not in X_all or b not in X_all:
            continue
        ia, la = SS.pair_bins(X_all[a].to_numpy())
        ib, lb = SS.pair_bins(X_all[b].to_numpy())
        cell = ia * len(lb) + ib
        H = np.bincount(cell, minlength=len(la) * len(lb)).reshape(len(la), len(lb))
        share = np.full(H.shape, np.nan)
        top = pd.DataFrame({"c": cell, "p": per}).value_counts().groupby(level=0).max()
        share.flat[top.index.to_numpy()] = top.to_numpy() / H.flat[top.index.to_numpy()]
        ok = H >= SS.PAIR_SPARSE_SHARE * n

        def means(v):
            s = np.bincount(cell, weights=v, minlength=H.size).reshape(H.shape)
            return np.where(H > 0, s / np.maximum(H, 1), np.nan)

        for name, m in models.items():
            feats = list(m["features"])
            if a not in feats or b not in feats or _term_name(m["full"], pair) is None:
                continue

            def contrib(model):
                t = _term_name(model, pair)
                if t is None:
                    return None
                c = model.eval_terms(X_all[feats])[:, list(model.term_names_).index(t)]
                return c - c.mean()
            surf = means(contrib(m["full"]))
            folds = [contrib(f) for f in m["folds"]]
            fm = [means(c) for c in folds if c is not None]
            available = len(fm) >= 2
            fold_sd = np.stack(fm).std(axis=0, ddof=1) if available else np.full(H.shape, np.nan)
            agree = None
            strong = ok & (np.abs(surf) >= 0.05)
            if available and strong.any():
                agree = float(np.mean([np.all(np.sign(np.stack(fm)[:, i, j]) == np.sign(surf[i, j]))
                                       for i, j in zip(*np.nonzero(strong))]))
            out.append({"configuration": name, "pair": f"{a} & {b}", "label": pair, "a": a, "b": b,
                        "labels_a": la, "labels_b": lb, "surface": surf, "fold_sd": fold_sd, "support": H,
                        "period_share": share, "fold_models": len(fm), "fold_sd_available": available,
                        "cells": int(H.size), "cells_supported": int(ok.sum()),
                        "max_abs": float(np.abs(surf[ok]).max()) if ok.any() else 0.0,
                        "range_supported": float(np.ptp(surf[ok])) if ok.any() else 0.0,
                        "fold_sd_max_supported": float(np.nanmax(fold_sd[ok])) if ok.any() and available else None,
                        "strong_cells": int(strong.sum()), "strong_cells_sign_agreement": agree})
    return out


def model_sets(fits_5a: Mapping[str, Any], configs_5a: Mapping[str, Any], fits_5b: Mapping[str, Any],
               configs_5b: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The models whose surfaces are compared: the 5A baseline and each candidate's 5A and reselected fits."""
    out = {}
    for name in ORDER:
        if name in fits_5b:
            out[name] = {"features": configs_5b[name]["features"], "full": fits_5b[name]["full"],
                         "folds": fits_5b[name]["models"]}
        elif name in fits_5a:
            out[name] = {"features": configs_5a[name]["features"], "full": fits_5a[name]["full"],
                         "folds": fits_5a[name]["models"]}
    return out


def interaction_value(models: Mapping[str, Mapping[str, Any]], X_all: pd.DataFrame,
                      surfs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """What each shown pair adds over its features' main effects, in the all-rows fit: mean |contribution| of
    the pair term and of each main effect (log-odds), the pair's share of all terms' mean |contribution|, and
    the largest supported cell effect with its fold agreement. ``fold_models_with_pair`` is the number of
    selecting folds; with fewer than two the fold sd and sign agreement are unavailable (None)."""
    rows = []
    for s in surfs:
        m = models[s["configuration"]]
        feats = list(m["features"])
        c = m["full"].eval_terms(X_all[feats])
        c = c - c.mean(axis=0)
        names = list(m["full"].term_names_)
        imp = np.abs(c).mean(axis=0)
        t = _term_name(m["full"], s["label"])
        rows.append({"configuration": s["configuration"], "pair": s["label"],
                     "pair_importance": float(imp[names.index(t)]), "main_a_importance": float(imp[names.index(s["a"])]),
                     "main_b_importance": float(imp[names.index(s["b"])]), "pair_share_of_all_terms": float(
                         imp[names.index(t)] / imp.sum()), "max_abs_supported": s["max_abs"],
                     "range_supported": s["range_supported"], "fold_models_with_pair": s["fold_models"],
                     "fold_sd_available": bool(s.get("fold_sd_available", s["fold_models"] >= 2)),
                     "fold_sd_max_supported": s["fold_sd_max_supported"], "strong_cells": s["strong_cells"],
                     "strong_cells_sign_agreement": s["strong_cells_sign_agreement"]})
    return pd.DataFrame(rows)


def _fold_cell(value: Any, available: bool, *, none: str = "") -> Any:
    if not available:
        return "unavailable"
    if value is None or pd.isna(value):
        return none
    return round(float(value), 4)


def value_summary(value: pd.DataFrame) -> pd.DataFrame:
    """The value table for reading: selecting folds as 'n of 5'; fold sd and sign agreement 'unavailable' where
    fewer than two folds select the pair (sign agreement 'no strong cell' where no cell reaches 0.05)."""
    t = value.copy()
    ok = t["fold_sd_available"].astype(bool)
    t["selecting_folds"] = [f"{int(n)} of {N_FOLDS}" for n in t["fold_models_with_pair"]]
    t["fold_sd_max_supported"] = [_fold_cell(v, a) for v, a in zip(t["fold_sd_max_supported"], ok)]
    t["strong_cells_sign_agreement"] = [_fold_cell(v, a, none="no strong cell")
                                        for v, a in zip(t["strong_cells_sign_agreement"], ok)]
    num = ["pair_importance", "main_a_importance", "main_b_importance", "pair_share_of_all_terms", "max_abs_supported"]
    t[num] = t[num].astype(float).round(4)
    return t[["configuration", "pair", *num, "selecting_folds", "fold_sd_max_supported", "strong_cells",
              "strong_cells_sign_agreement"]]


LABELS = {"jed_alone": "jed_alone (5A, fixed pairs)", "jed_alone_selected": "jed_alone, reselected",
          "drop_first": "drop_first (5A, fixed pairs)", "drop_first_selected": "drop_first, reselected",
          "baseline": "baseline (5A, accepted pairs)"}
ORDER = ("baseline", "jed_alone", "jed_alone_selected", "drop_first", "drop_first_selected")


def recurrence_figure(tables: Mapping[str, pd.DataFrame], selections: Mapping[str, Sequence[Mapping[str, Any]]],
                      top: int = 8):
    """For each configuration: which pairs each outer fit and the all-rows fit selected (filled = selected),
    for the pairs selected anywhere, with the reference pairs marked."""
    import matplotlib.pyplot as plt
    names = list(tables)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        t = tables[name]
        shown = t[(t["outer_selected"] > 0) | t["all_rows_selected"] | t["reference"]].head(top)
        fits = [s for s in sorted(selections[name], key=lambda s: _fit_order(s["fit"]))
                if not s["fit"].startswith("inner")]
        M = np.array([[p in s["selected"] for s in fits] for p in shown["pair"]], dtype=float)
        ax.imshow(M, cmap=plt.matplotlib.colors.ListedColormap(["#f0efec", "#2a78d6"]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fits)), [s["fit"].replace("outer ", "fold ") for s in fits], fontsize=7, rotation=30,
                      ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(shown)), [("* " if r else "") + p.replace(" x ", " ×\n") for p, r in
                                          zip(shown["pair"], shown["reference"])], fontsize=6.5)
        for i, (_, r) in enumerate(shown.iterrows()):
            ax.text(len(fits) - 0.5 + 0.15, i, f"{r['outer_selected']}/5", va="center", fontsize=7, color="#3d3d3a")
        ax.set_xlim(-0.5, len(fits) - 0.5 + 0.9)
        ax.set_title(name.replace("_selected", ", reselected").replace("baseline", "frozen 13 features"), fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Pairs selected by each outer fit and the all-rows fit (blue = selected; * = accepted or 5A pair; "
                 "n/5 = outer fits selecting it)", fontsize=9)
    return fig


# ======================================================================
# 9. complementarity under Step 5A's cascade screen (Windows)
# ======================================================================
def cascade_screen(views: Mapping[str, Any], fits_5a: Mapping[str, Any], fits_5b: Mapping[str, Any],
                   y: pd.Series, fold_id: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Step 5A's complementarity, with the reselected fits added: the four paired counts and the net corrections
    against the 5A refitted baseline, for all rows and Stage 2 abstentions; and Step 5A's cascade screen
    (within the 5A tolerances against the baseline, and abstention net >= baseline net - 0.5% of the
    abstention rows) for each 5A candidate and its reselected fit. None without the accepted views.
    """
    if not views.get("available"):
        return None
    yv = y.astype(int).to_numpy()
    cands = [c for c in CANDIDATES if c in fits_5a and f"{c}_selected" in fits_5b]
    names = ["baseline", *cands, *[f"{c}_selected" for c in cands]]
    allf = {**{k: fits_5a[k] for k in ("baseline", *CANDIDATES)}, **fits_5b}
    dec = {"accepted": views["ebm"]["decision"],
           **{k: (allf[k]["proba"] >= allf[k]["nested"]["per_row"]).astype(int) for k in names}}
    comp = SS.complementarity(yv, views["lr"]["flags"], views["router"]["abstain"], dec)
    diff = SS.paired_differences({k: allf[k] for k in names}, y, fold_id)
    pooled = diff["pooled"].set_index("configuration")
    b = pooled.loc["baseline"]
    ag = comp["against_reference"]
    rows = []
    for name in names[1:]:
        r = pooled.loc[name]
        perf = bool(r["d_roc_auc"] >= SS.TOL["roc_auc"] and r["d_roc_auc_low"] >= SS.TOL["roc_auc_lower"]
                    and r["d_pr_auc"] >= SS.TOL["pr_auc"] and r["d_nested_f2"] >= SS.TOL["nested_f2"]
                    and r["ece_calibrated"] <= SS.TOL["ece_gate"]
                    and r["brier_calibrated"] <= b["brier_calibrated"] + SS.TOL["calibrated_brier"])
        a = ag[(ag["ebm"] == name) & (ag["population"] == "Stage 2 abstentions")].iloc[0]
        need = float(a["reference_net"] + SS.TOL["net_abstain_share"] * a["rows"])
        rows.append({"configuration": name, "within_5a_tolerances": perf, "abstention_net": int(a["net"]),
                     "abstention_net_needed": need, "screen": "meets" if perf and a["net"] >= need else "misses"})
    screen = pd.DataFrame(rows)
    s = screen.set_index("configuration")["screen"]
    change = pd.DataFrame([{"candidate": c, "five_a": s[c], "reselected": s[f"{c}_selected"],
                            "conclusion_changes": bool(s[c] != s[f"{c}_selected"])} for c in cands])
    return {"comp": comp, "checks": SS.complementarity_checks(comp, len(yv)), "screen": screen, "change": change,
            "pooled": diff["pooled"]}


# ======================================================================
# 10. statements, limits and the record
# ======================================================================
#: The review document's statements, from the sandbox run (``SANDBOX_REFERENCE``); Cell 67 evaluates every one
#: on the run that writes the record. They are required in the sandbox's environment; elsewhere one that no
#: longer holds is recorded and flagged for review (``sandbox_reproduction``). Outer and all-rows selections
#: have a third-to-fourth mean-rank margin of at least 0.64 (about 9 bag-rank swaps), except drop_first's outer
#: fold 3 (0.071, one swap): that fold's third pair is stated as a claim, not exactly. Inner selections include
#: near-ties, so their counts carry a tolerance.
STATED: Dict[str, Any] = {
    "recovery": {
        "all_rows_selected": ["campaign x emp_var_rate_sigmoid_slope", "euribor3m_sigmoid_slope x campaign", "decay_x_density x campaign"],
        "accepted_recovered_all_rows": ["campaign x emp_var_rate_sigmoid_slope"],
        "accepted_outer_counts": {"campaign x emp_var_rate_sigmoid_slope": 5, "euribor3m_sigmoid_slope x economic_curvature_intensity": 0,
                                  "prior_x_stress x decay_x_density": 0},
        "accepted_all_rows_positions": {"euribor3m_sigmoid_slope x economic_curvature_intensity": (30, 3),
                                        "prior_x_stress x decay_x_density": (38, 3)},
    },
    "outer_counts": {
        ("baseline", "campaign x emp_var_rate_sigmoid_slope"): 5, ("baseline", "decay_x_density x campaign"): 4,
        ("baseline", "euribor3m_sigmoid_slope x campaign"): 3,
        ("jed_alone_selected", "campaign x emp_var_rate_sigmoid_slope"): 5, ("jed_alone_selected", "joint_economic_decay x campaign"): 4,
        ("jed_alone_selected", "euribor3m_sigmoid_slope x campaign"): 3,
        ("jed_alone_selected", "euribor3m_sigmoid_slope x economic_curvature_intensity"): 0,
        ("drop_first_selected", "campaign x emp_var_rate_sigmoid_slope"): 5, ("drop_first_selected", "prior_x_stress x decay_x_density"): 0,
    },
    "outer_claims": {("drop_first_selected", "decay_x_density x campaign"): (">=", 4)},
    "recurring": {"baseline": ["campaign x emp_var_rate_sigmoid_slope", "decay_x_density x campaign", "euribor3m_sigmoid_slope x campaign"],
                  "jed_alone_selected": ["campaign x emp_var_rate_sigmoid_slope", "joint_economic_decay x campaign",
                                         "euribor3m_sigmoid_slope x campaign"],
                  "drop_first_selected": ["campaign x emp_var_rate_sigmoid_slope", "decay_x_density x campaign"]},
    "third_slot_max_outer": {"drop_first_selected": 2},
    "all_rows_selected": {"jed_alone_selected": ["campaign x emp_var_rate_sigmoid_slope", "euribor3m_sigmoid_slope x campaign",
                                                  "joint_economic_decay x campaign"],
                          "drop_first_selected": ["campaign x emp_var_rate_sigmoid_slope", "decay_x_density x campaign", "cons.conf.idx x campaign"]},
    "inner_counts": {("jed_alone_selected", "campaign x emp_var_rate_sigmoid_slope"): (17, 3), ("drop_first_selected", "campaign x emp_var_rate_sigmoid_slope"): (19, 3),
                     ("jed_alone_selected", "joint_economic_decay x campaign"): (20, 3),
                     ("drop_first_selected", "decay_x_density x campaign"): (22, 3)},
    "comparison": {
        ("jed_alone_selected", "d_roc_auc"): (-0.0006, 0.0005), ("jed_alone_selected", "d_pr_auc"): (-0.0012, 0.001),
        ("jed_alone_selected", "d_nested_f2"): (0.0001, 0.003),
        ("jed_alone_selected", "d_brier_calibrated"): (0.0001, 0.0003),
        ("jed_alone_selected", "ece_calibrated"): (0.0026, 0.001),
        ("drop_first_selected", "d_roc_auc"): (0.0005, 0.0005), ("drop_first_selected", "d_pr_auc"): (0.0013, 0.001),
        ("drop_first_selected", "d_nested_f2"): (-0.0031, 0.003),
        ("drop_first_selected", "d_brier_calibrated"): (-0.0001, 0.0003),
        ("drop_first_selected", "ece_calibrated"): (0.0028, 0.001),
    },
    "material_change": {"jed_alone_selected": False, "drop_first_selected": False},
    "folds_better": {("drop_first_selected", "pr_auc"): 5},
    "more_flags": {"jed_alone_selected": 200, "drop_first_selected": 200},
    "value_claims": {
        ("jed_alone", "campaign x emp_var_rate_sigmoid_slope", "max_abs_supported"): (">=", 0.5),
        ("drop_first", "campaign x emp_var_rate_sigmoid_slope", "max_abs_supported"): (">=", 0.5),
        ("jed_alone_selected", "campaign x emp_var_rate_sigmoid_slope", "max_abs_supported"): ("<=", 0.4),
        ("drop_first_selected", "campaign x emp_var_rate_sigmoid_slope", "max_abs_supported"): ("<=", 0.4),
        ("jed_alone_selected", "campaign x emp_var_rate_sigmoid_slope", "strong_cells_sign_agreement"): (">=", 1.0),
        ("drop_first_selected", "campaign x emp_var_rate_sigmoid_slope", "strong_cells_sign_agreement"): (">=", 1.0),
        ("drop_first_selected", "decay_x_density x campaign", "strong_cells_sign_agreement"): ("<", 1.0),
    },
    "shown": ["campaign x emp_var_rate_sigmoid_slope", "decay_x_density x campaign", "euribor3m_sigmoid_slope x campaign",
              "euribor3m_sigmoid_slope x economic_curvature_intensity", "prior_x_stress x decay_x_density",
              "joint_economic_decay x campaign"],
    # Windows only: the accepted 5A run's counts and verdicts must reproduce beside the reselected fits
    "screen_anchors": {"baseline_net": 530, "abstention_net": {"jed_alone": 486, "drop_first": 441},
                       "screen": {"jed_alone": "meets", "drop_first": "misses"}},
}

_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b}


def _value_stat(value: pd.DataFrame, name: str, pair: str, col: str) -> Tuple[Optional[float], Optional[str]]:
    """
    One statistic of the interaction-value table, or why it cannot be read: the surface is absent (the
    configuration's all-rows fit does not hold the pair, or the pair is not among those shown), the column is
    missing, or the value is unavailable (None: fewer than two selecting folds, or no well-supported cell of at
    least 0.05 log-odds) or not finite. Never raises on a legitimate difference from the sandbox run.
    """
    if not {"configuration", "pair"} <= set(value.columns):
        return None, "surface absent: the interaction-value table is empty"
    rows = value[(value["configuration"] == name) & (value["pair"] == pair)]
    if rows.empty:
        return None, (f"surface absent: no {pair} surface for {name} (its all-rows fit does not hold the pair, or the "
                      "pair is not among those shown)")
    if col not in rows.columns:
        return None, f"statistic missing: the interaction-value table has no {col}"
    r = rows.iloc[0]
    x = r[col]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        folds = r.get("fold_models_with_pair")
        if col in ("fold_sd_max_supported", "strong_cells_sign_agreement") and folds is not None and folds < 2:
            return None, f"statistic unavailable: {col} needs two selecting folds; {int(folds)} of {N_FOLDS} select it"
        if col == "strong_cells_sign_agreement" and r.get("strong_cells") == 0:
            return None, f"statistic unavailable: {col} has no well-supported cell of at least 0.05 log-odds"
        return None, f"statistic unavailable: {col} is missing for {name} {pair}"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None, f"statistic unavailable: {col} is {x!r}, not a number"
    if not math.isfinite(f):
        return None, f"statistic not finite: {col} is {f}"
    return f, None


def stated_checks(*, recovery: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], fits: Mapping[str, Any],
                  cmp: pd.DataFrame, vs: Mapping[str, Any], value: pd.DataFrame,
                  screen: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    """Every statement of research_logs/pr42/interaction_selection.md against this run."""
    st, out = STATED, []
    r = st.get("recovery", {})
    rs = recovery_summary(recovery, tables["baseline"]) if r else {}
    for key in ("all_rows_selected", "accepted_recovered_all_rows", "accepted_outer_counts"):
        if key in r:
            out.append(_check(f"recovery: {key} = {r[key]}", rs[key] == r[key], rs[key]))
    for p, (want, tol) in r.get("accepted_all_rows_positions", {}).items():
        got = rs["accepted_all_rows_positions"][p]
        out.append(_check(f"recovery: {p} ranks {want} (+/- {tol}) of 78 on all rows", abs(got - want) <= tol, got))
    for (name, p), want in st.get("outer_counts", {}).items():
        got = int(tables[name].set_index("pair").loc[p, "outer_selected"])
        out.append(_check(f"{name}: {p} selected by {want} of 5 outer fits", got == want, got))
    for (name, p), (op, bound) in st.get("outer_claims", {}).items():
        got = int(tables[name].set_index("pair").loc[p, "outer_selected"])
        out.append(_check(f"{name}: {p} selected by {op} {bound} of 5 outer fits", _OPS[op](got, bound), got))
    for name, want in st.get("recurring", {}).items():
        t = tables[name]
        got = sorted(t.loc[t["outer_selected"] >= RECURRING_MIN, "pair"])
        out.append(_check(f"{name}: the recurring pairs are {sorted(want)}", got == sorted(want), got))
    for name, want in st.get("third_slot_max_outer", {}).items():
        t = tables[name]
        rest = t.loc[~t["pair"].isin(st["recurring"][name]), "outer_selected"]
        out.append(_check(f"{name}: no other pair is selected by more than {want} outer fits",
                          int(rest.max()) <= want, int(rest.max())))
    for name, want in st.get("all_rows_selected", {}).items():
        got = next(s for s in fits[name]["selections"] if s["fit"] == "all rows")["selected"]
        out.append(_check(f"{name}: the all-rows fit selects {want}", list(got) == list(want), got))
    for (name, p), (want, tol) in st.get("inner_counts", {}).items():
        got = int(tables[name].set_index("pair").loc[p, "inner_selected"])
        out.append(_check(f"{name}: {p} selected by {want} (+/- {tol}) of 25 inner fits", abs(got - want) <= tol, got))
    c = cmp.set_index("configuration") if "configuration" in cmp.columns else cmp
    for (name, col), (want, tol) in st.get("comparison", {}).items():
        got = float(c.loc[name, col])
        out.append(_check(f"{name} against its 5A fit: {col} = {want} (+/- {tol})", abs(got - want) <= tol + 1e-12,
                          round(got, 5)))
    for name, want in st.get("material_change", {}).items():
        got = bool(c.loc[name, "material_change"])
        out.append(_check(f"{name}: material change against its 5A fit = {want}", got == want, got))
    for (name, metric), want in st.get("folds_better", {}).items():
        fd = vs[name]["fold_differences"].set_index("metric")
        got = int(fd.loc[metric, "folds_better"])
        out.append(_check(f"{name}: {metric} is better than its 5A fit's in {want} of 5 folds", got == want, got))
    for name, want in st.get("more_flags", {}).items():
        got = int(vs[name]["decisions"]["flagged"] - vs[name]["decisions"]["flagged_5a"])
        out.append(_check(f"{name}: flags at least {want} more rows than its 5A fit", got >= want, got))
    for (name, p, col), (op, bound) in st.get("value_claims", {}).items():
        # an absent surface or an unavailable statistic is a statement that no longer holds, with its reason: it is
        # recorded and flagged for review (and fails the same-sandbox reproduction), never an exception before the
        # record is written
        got, reason = _value_stat(value, name, p, col)
        out.append(_check(f"{name} {p}: {col} {op} {bound}", reason is None and _OPS[op](got, bound),
                          reason if reason is not None else round(got, 4)))
    if "shown" in st:
        got = pairs_to_show(tables)
        out.append(_check(f"the surfaces shown are {len(st['shown'])} pairs", got == st["shown"], got))
    a = st.get("screen_anchors")
    if a:
        if screen is None:
            out.append(_check("complementarity statements: not checkable in this run (the accepted views are not "
                              "available; with PR42_REQUIRE_ACCEPTED_VIEWS=1 they are required)", True, None))
        else:
            ag = screen["comp"]["against_reference"]
            ab = ag[ag["population"] == "Stage 2 abstentions"].set_index("ebm")
            base = int(ab["reference_net"].iloc[0])
            out.append(_check(f"the 5A baseline's abstention net = {a['baseline_net']}", base == a["baseline_net"], base))
            for name, want in a["abstention_net"].items():
                got = int(ab.loc[name, "net"])
                out.append(_check(f"{name} (5A): abstention net = {want}", got == want, got))
            sc = screen["screen"].set_index("configuration")["screen"]
            for name, want in a["screen"].items():
                out.append(_check(f"{name} (5A): cascade screen {want}", sc[name] == want, sc[name]))
            out.append(_check("the reselected fits' screen verdicts are computed",
                              all(f"{c}_selected" in sc.index for c in CANDIDATES), dict(sc)))
    return pd.DataFrame(out)


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def _recs(df: Optional[pd.DataFrame]):
    return None if df is None else json.loads(df.to_json(orient="records"))


# ======================================================================
# 11. acceptance: what every run must satisfy
# ======================================================================
#: Step 5A's registered cascade-screen tolerances, as the Step 5B registration adopts them unchanged.
SCREEN_TOL = {"roc_auc": -0.005, "roc_auc_lower": -0.010, "pr_auc": -0.010, "nested_f2": -0.010,
              "calibrated_brier": 0.002, "ece_gate": 0.05, "net_abstain_share": -0.005}
_MATERIAL_TEXT = {"roc_auc": "|dROC-AUC| >= 0.002", "pr_auc": "|dPR-AUC| >= 0.005",
                  "nested_f2": "|dnested F2| >= 0.005", "brier_calibrated": "|dcalibrated Brier| >= 0.0005"}


def _pairset(names: Sequence[str]) -> set:
    """Unordered feature pairs of term names ('a & b') or pair labels ('a x b')."""
    return {frozenset(n.replace(" x ", " & ").split(" & ")) for n in names if " & " in n.replace(" x ", " & ")}


def acceptance_checks(*, recovery: Mapping[str, Any], fits_5a: Mapping[str, Any], fits_5b: Mapping[str, Any],
                      configs_5b: Mapping[str, Any], fit_checks: pd.DataFrame, frames: Mapping[str, Any], y: pd.Series,
                      fold_id: np.ndarray, cmp: pd.DataFrame, screen: Optional[Mapping[str, Any]],
                      noms_5a: pd.DataFrame, stated_5a: pd.DataFrame, accepted_5a: pd.DataFrame,
                      views: Mapping[str, Any]) -> pd.DataFrame:
    """
    What every run must satisfy, whatever its platform:
    - fit isolation: Step 5A's fit and fold-isolation checks pass for every reselected fit, and every selection
      ran on its own fit frame's rows (each outer selection on exactly the rows outside its held-out fold);
    - ranking-to-fitted-term agreement: in every selection the recorded top three by mean rank are the fitted
      model's pair terms, every eligible pair was ranked in all 14 bags, and the models that scored the held-out
      rows (and the all-rows models) hold exactly their selection's pairs;
    - paired row identity: labels, fold plan and frames are the same rows in the same order, each outer frame's
      held-out rows are its fold in that plan, and each reselected fit and its 5A fit score the same rows from
      the same fit and inner frames;
    - accepted Step 5A reproduction (Cell 60's integrity and required reproduction; Cell 58's statements);
    - the registered thresholds are intact, and the materiality and cascade verdicts reported are those rules
      applied to this run's numbers (recomputed here, not read back from the code that produced them).
    """
    out = []
    out.append(_check("fit isolation: every reselected fit passes Step 5A's fitted-terms, fit-size and fold-isolation "
                      "checks", len(fit_checks) > 0 and bool(fit_checks["passed"].all()), int(len(fit_checks))))
    sels = {"recovery": recovery["selections"], **{n: r["selections"] for n, r in fits_5b.items()}}
    n_sel = sum(len(v) for v in sels.values())
    idx, fid = np.asarray(frames["X_all"].index), np.asarray(fold_id)
    outer_rows = {f"outer {k}": _index_sha(idx[fid != k]) for k in range(N_FOLDS)}
    stray = []
    for name, ss in sels.items():
        fits = [s["fit"] for s in ss]
        if "unmatched" in fits or len(set(fits)) != len(fits):
            stray.append(f"{name}: a selection is unmatched or repeated")
        stray += [f"{name} {s['fit']}" for s in ss if s["fit"] in outer_rows and s["index_sha256"] != outer_rows[s["fit"]]]
        stray += [f"{name} all rows" for s in ss if s["fit"] == "all rows" and s["index_sha256"] != _index_sha(idx)]
    out.append(_check(f"fit isolation: all {n_sel} selections ran on one fit frame's rows; each outer selection on "
                      "exactly the rows outside its held-out fold", not stray, stray or n_sel))
    nfeat = {"recovery": len(recovery["features"]), **{n: len(configs_5b[n]["features"]) for n in fits_5b}}
    mism = [f"{n} {s['fit']}" for n, ss in sels.items() for s in ss if not s["selection_matches_fit"]]
    out.append(_check(f"ranking-to-fitted-term agreement: in all {n_sel} selections the recorded top {BUDGET} by mean "
                      "rank are the fitted model's pair terms", not mism, mism or n_sel))
    short = [f"{n} {s['fit']}" for n, ss in sels.items() for s in ss
             if s["bags"] != 14 or s["eligible_per_bag"] != [nfeat[n] * (nfeat[n] - 1) // 2]]
    out.append(_check("ranking-to-fitted-term agreement: every eligible pair is ranked in each of the 14 bags of every "
                      "selection", not short, short or n_sel))
    scored = []
    for n, r in fits_5b.items():
        by_fit = {s["fit"]: s for s in r["selections"]}
        for k, terms in enumerate(r["terms"]):
            fit = f"outer {k}" if k < N_FOLDS else "all rows"
            if fit not in by_fit or _pairset(terms) != _pairset(by_fit[fit]["selected"]):
                scored.append(f"{n} {fit}")
    out.append(_check("ranking-to-fitted-term agreement: each outer model that scored its held-out rows, and each "
                      "all-rows model, holds exactly its selection's pairs", not scored,
                      scored or sum(len(r["terms"]) for r in fits_5b.values())))
    yv = np.asarray(y)
    same = len(yv) == len(fid) == len(idx) and np.array_equal(np.asarray(y.index), idx)
    out.append(_check(f"paired row identity: labels, fold plan and frames are the same {len(idx):,} rows in the same "
                      "order", same, len(idx)))
    plan = [k for k, f in enumerate(frames["outer"])
            if _index_sha(f.X_val.index) != _index_sha(idx[fid == k]) or _index_sha(f.X_tr.index) != outer_rows[f"outer {k}"]]
    out.append(_check("paired row identity: each outer frame's held-out rows are exactly its fold in the plan used "
                      "for pairing", not plan, plan or N_FOLDS))
    for n, c in configs_5b.items():
        a, b = fits_5a[c["five_a"]], fits_5b[n]
        lengths = {len(a["proba"]), len(b["proba"]), len(a["calibrated"]), len(b["calibrated"]),
                   len(a["nested"]["per_row"]), len(b["nested"]["per_row"])}
        ok = lengths == {len(yv)} and a["fit_rows"] == b["fit_rows"] and a["inner_sizes"] == b["inner_sizes"]
        out.append(_check(f"paired row identity: {n} and its 5A fit ({c['five_a']}) score the same rows from the same "
                          "fit and inner frames", ok, {"rows": sorted(lengths), "fit_rows": b["fit_rows"]}))
    out.append(_check("accepted Step 5A reproduction: the accepted record's integrity, and exact reproduction where "
                      "Step 3B's rule requires it (Cell 60)", bool(accepted_5a["passed"].all()),
                      accepted_5a["check"].tolist()))
    out.append(_check("accepted Step 5A reproduction: this run's Step 5A review-document statements all hold (Cell 58)",
                      len(stated_5a) > 0 and bool(stated_5a["passed"].all()), int(len(stated_5a))))
    text = REGISTRATION["comparisons"]
    intact = (registration_digest() == REGISTERED_SHA256
              and MATERIAL == {"roc_auc": 0.002, "pr_auc": 0.005, "nested_f2": 0.005, "brier_calibrated": 0.0005}
              and all(t in text["material"] for t in _MATERIAL_TEXT.values())
              and all(SS.TOL[k] == v for k, v in SCREEN_TOL.items())
              and "0.5% of the abstention rows" in text["cascade_screen"])
    out.append(_check("registered thresholds intact: the materiality thresholds and Step 5A's screen tolerances are "
                      "the registered ones", intact, {"material": dict(MATERIAL), "screen": dict(SCREEN_TOL)}))
    for _, r in cmp.iterrows():
        want = {"material_roc_auc": bool(r["d_roc_auc_low"] > 0 or r["d_roc_auc_high"] < 0 or abs(r["d_roc_auc"]) >= 0.002),
                "material_pr_auc": bool(abs(r["d_pr_auc"]) >= 0.005),
                "material_nested_f2": bool(abs(r["d_nested_f2"]) >= 0.005),
                "material_brier_calibrated": bool(abs(r["d_brier_calibrated"]) >= 0.0005)}
        want["material_change"] = any(want.values())
        got = {k: bool(r[k]) for k in want}
        arith = (abs(r["d_roc_auc"] - (r["roc_auc"] - r["roc_auc_5a"])) < 1e-9
                 and all(abs(r[f"d_{m}"] - (r[m] - r[f"{m}_5a"])) < 1e-12
                         for m in ("pr_auc", "nested_f2", "brier_calibrated", "ece_calibrated")))
        out.append(_check(f"truthful materiality: {r['configuration']} is reported "
                          f"{'material' if got['material_change'] else 'not material'}, the registered thresholds "
                          "applied to its pooled differences", got == want and arith, want))
    if screen is None:
        out.append(_check("truthful cascade verdicts: not computed in this run (the accepted views are not available); "
                          "the acceptance run requires them", not views.get("required"), views.get("status")))
        return pd.DataFrame(out)
    pooled = screen["pooled"].set_index("configuration")
    b = pooled.loc["baseline"]
    ag = screen["comp"]["against_reference"]
    wrong = []
    for _, s in screen["screen"].iterrows():
        r = pooled.loc[s["configuration"]]
        within = bool(r["d_roc_auc"] >= SCREEN_TOL["roc_auc"] and r["d_roc_auc_low"] >= SCREEN_TOL["roc_auc_lower"]
                      and r["d_pr_auc"] >= SCREEN_TOL["pr_auc"] and r["d_nested_f2"] >= SCREEN_TOL["nested_f2"]
                      and r["ece_calibrated"] <= SCREEN_TOL["ece_gate"]
                      and r["brier_calibrated"] <= b["brier_calibrated"] + SCREEN_TOL["calibrated_brier"])
        a = ag[(ag["ebm"] == s["configuration"]) & (ag["population"] == "Stage 2 abstentions")].iloc[0]
        need = float(a["reference_net"] + SCREEN_TOL["net_abstain_share"] * a["rows"])
        verdict = "meets" if within and a["net"] >= need else "misses"
        if (bool(s["within_5a_tolerances"]), s["screen"], int(s["abstention_net"])) != (within, verdict, int(a["net"])) \
                or abs(float(s["abstention_net_needed"]) - need) > 1e-9:
            wrong.append(s["configuration"])
    out.append(_check("truthful cascade verdicts: every screen verdict reported is Step 5A's registered rule applied to "
                      "this run's pooled differences and abstention nets", not wrong, wrong or len(screen["screen"])))
    sc = screen["screen"].set_index("configuration")["screen"]
    noms = noms_5a.set_index("configuration")
    pairs = {c: (sc[c], noms.loc[c, "cascade"]) for c in CANDIDATES}
    out.append(_check("truthful cascade verdicts: the 5A candidates' verdicts here are this run's Step 5A cascade "
                      "verdicts (meets = nominated)", all((v == "meets") == (n == "nominated") for v, n in pairs.values()),
                      pairs))
    ch = screen["change"]
    ok = all(bool(r["conclusion_changes"]) == (r["five_a"] != r["reselected"]) and r["five_a"] == sc[r["candidate"]]
             and r["reselected"] == sc[f"{r['candidate']}_selected"] for _, r in ch.iterrows())
    out.append(_check("truthful cascade verdicts: the change table gives each candidate's 5A and reselected verdict "
                      "and whether reselection changes it", ok and len(ch) == len(CANDIDATES), _recs(ch)))
    return pd.DataFrame(out)


# ======================================================================
# 12. same-sandbox reproduction: what is required only in the sandbox's environment
# ======================================================================
#: The sandbox run the review document quotes. Its values (every selection, margin, count, estimate and verdict
#: below) were taken from that run's record with ``observed``; the file is pinned by its LF SHA-256.
SANDBOX_REFERENCE = {"path": "feature_research/interaction_selection_sandbox.json",
                     "sha256_lf": "07342f583c49606320c75c57961c292d17221f1bece735b424e764e5e7f6e082"}
_DIRECTION = ("roc_auc", "pr_auc", "nested_f2", "brier_calibrated", "ece_calibrated")   # registered comparisons
_POOLED = ("roc_auc_5a", "roc_auc", "d_roc_auc", "d_roc_auc_low", "d_roc_auc_high", "d_roc_auc_p", "pr_auc_5a",
           "pr_auc", "d_pr_auc", "nested_f2_5a", "nested_f2", "d_nested_f2", "brier_raw_5a", "brier_raw", "d_brier_raw",
           "brier_calibrated_5a", "brier_calibrated", "d_brier_calibrated", "ece_raw_5a", "ece_raw", "d_ece_raw",
           "ece_calibrated_5a", "ece_calibrated", "d_ece_calibrated")
_DECISIONS = ("differ", "flag_now_not_5a", "flag_5a_not_now", "flagged", "flagged_5a")
_SURFACE = ("pair_importance", "pair_share_of_all_terms", "max_abs_supported", "fold_sd_max_supported",
            "strong_cells_sign_agreement")


def sandbox_reference(root: Path = BANK_ROOT) -> Dict[str, Any]:
    """The sandbox run's values, after checking the file's pinned LF SHA-256."""
    data = (Path(root) / SANDBOX_REFERENCE["path"]).read_bytes()
    got = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    if got != SANDBOX_REFERENCE["sha256_lf"]:
        raise RuntimeError(f"{SANDBOX_REFERENCE['path']}: LF SHA-256 {got} is not the pinned "
                           f"{SANDBOX_REFERENCE['sha256_lf']}")
    return json.loads(data.decode("utf-8"))


def observed(sections: Mapping[str, Any], *, step5a_content: str) -> Dict[str, Dict[str, Any]]:
    """
    The run's reproducible quantities, keyed 'configuration | fit or pair | quantity', each with its kind, read
    from the record sections as they are written (``canonical``), so the same sandbox gives equal values. The
    sandbox reference was made by this function from the sandbox run's record.
    """
    from upstream_regeneration.records import canonical
    s = canonical({"recovery": sections["recovery"], "candidates": sections["candidates"],
                   "recurrence": sections["recurrence"], "against_5a": sections["against_5a"],
                   "value": sections["surfaces"]["value"], "shown": sections["surfaces"]["shown"]})
    out: Dict[str, Dict[str, Any]] = {}

    def put(key, kind, value):
        out[key] = {"kind": kind, "value": value}
    put("step 5A record content", "identity", step5a_content)
    sels = {"baseline": s["recovery"]["selections"], **{n: c["selections"] for n, c in s["candidates"].items()}}
    for name, ss in sels.items():
        for x in sorted(ss, key=lambda x: _fit_order(x["fit"])):
            put(f"{name} | {x['fit']} | pairs", "inner selection" if x["fit"].startswith("inner") else "selection",
                list(x["selected"]))
            put(f"{name} | {x['fit']} | margin", "margin", x["margin"])
    for name, rows in s["recurrence"].items():
        for r in rows:
            put(f"{name} | {r['pair']} | outer fits selecting", "count", r["outer_selected"])
            if r.get("inner_selected") is not None:
                put(f"{name} | {r['pair']} | inner fits selecting", "inner count", r["inner_selected"])
            put(f"{name} | {r['pair']} | all-rows position", "position", r["all_rows_position"])
    for r in s["against_5a"]["table"]:
        n = r["configuration"]
        for col in _POOLED:
            direction = col.startswith("d_") and col[2:] in _DIRECTION
            put(f"{n} | vs 5A | {col}", "difference" if direction else "estimate", r[col])
        for col in _DECISIONS:
            put(f"{n} | vs 5A | {col}", "decisions", r[col])
        put(f"{n} | vs 5A | material_change", "verdict", r["material_change"])
    for n, rows in s["against_5a"]["fold_differences"].items():
        for r in rows:
            put(f"{n} | per fold | {r['metric']} mean", "difference" if r["metric"] in _DIRECTION else "estimate",
                r["mean"])
            put(f"{n} | per fold | {r['metric']} sd", "estimate", r["sd"])
            put(f"{n} | per fold | {r['metric']} folds better", "folds better", r["folds_better"])
    for r in s["value"]:
        k = f"{r['configuration']} | {r['pair']}"
        for col in _SURFACE:
            put(f"{k} | surface {col}", "estimate", r.get(col))
        put(f"{k} | surface selecting folds", "count", r["fold_models_with_pair"])
    put("surfaces shown", "list", s["shown"])
    return out


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flag(key: str, kind: str, a: Any, b: Any, equal: bool, material: Mapping[str, bool]) -> Tuple[str, bool]:
    """(flag, for review) for one item that differs from the sandbox."""
    if equal:
        return "", False
    if a is None or b is None:
        if kind in ("estimate", "margin"):
            return "available in only one run", kind == "estimate" and "fold_sd" in key
        return ("not in the sandbox run" if a is None else "not in this run"), True
    if kind == "selection":
        return "selection changed", True
    if kind == "inner selection":
        return "inner selection changed (a supporting count)", False
    if kind == "count":
        return "count changed", True
    if kind == "inner count":
        far = abs(b - a) > 3
        return ("inner count changed beyond the review document's +/- 3" if far else "inner count changed"), far
    if kind == "difference" and np.sign(a) != np.sign(b) and a != 0 and b != 0:
        below = not material.get(key.split(" | ")[0], False)
        return "direction changed" + (" (below the materiality threshold)" if below else ""), True
    if kind == "verdict":
        return "verdict changed", True
    if kind == "list":
        return "changed", True
    return "differs", False


def sandbox_reproduction(sections: Mapping[str, Any], statements: pd.DataFrame, *, step5a_content: str,
                         environment: Mapping[str, Any], views: Mapping[str, Any],
                         reference: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    This run against the sandbox run the review document quotes: every item of ``observed`` with the sandbox
    value, this run's value, the difference and a flag, and the review document's statements (``stated_checks``).
    Exact reproduction is required only where this run has the sandbox's environment (``TV.ENVIRONMENT_KEYS``)
    and view availability; elsewhere nothing here fails, and changed selections and counts, a change of
    direction in a registered comparison (below the materiality threshold or not), a changed verdict and a
    statement that no longer holds are flagged for review. The registered thresholds are not touched.
    """
    from upstream_regeneration.records import canonical
    from feature_research import temporal_validity as TV
    ref = reference if reference is not None else sandbox_reference()
    now = observed(sections, step5a_content=step5a_content)
    reasons = [f"{k} differs" for k in TV.ENVIRONMENT_KEYS if ref["environment"].get(k) != environment.get(k)]
    if bool(views.get("available")) != bool(ref["views_available"]):
        reasons.append("accepted view availability differs")
    required = not reasons
    material = {r["configuration"]: bool(r["material_change"]) for r in canonical(sections["against_5a"]["table"])}
    rows = []
    for key in [*ref["values"], *(k for k in now if k not in ref["values"])]:
        a, b = ref["values"].get(key), now.get(key)
        kind = (a or b)["kind"]
        va, vb = (None if a is None else a["value"]), (None if b is None else b["value"])
        equal = a is not None and b is not None and va == vb
        flag, review = _flag(key, kind, va, vb, equal, material)
        rows.append({"item": key, "kind": kind, "sandbox": va, "this_run": vb,
                     "difference": (vb - va) if _numeric(va) and _numeric(vb) else None,
                     "equal": equal, "flag": flag, "review": review})
    table = pd.DataFrame(rows)
    st = statements.reset_index(drop=True)
    differ, n_st_fail = int((~table["equal"]).sum()), int((~st["passed"]).sum())
    if required:
        checks = [_check(f"same-sandbox reproduction: all {len(table)} items equal the sandbox run "
                         f"{ref['run_id']}'s", differ == 0, differ),
                  _check(f"same-sandbox reproduction: all {len(st)} review-document statements hold", n_st_fail == 0,
                         n_st_fail)]
    else:
        checks = [_check(f"same-sandbox reproduction not required ({'; '.join(reasons)}): {differ} of {len(table)} "
                         f"items differ from the sandbox run {ref['run_id']}, {int(table['review'].sum())} flagged; "
                         f"{n_st_fail} of {len(st)} statements no longer hold", True, None)]
    return {"reference": {"run_id": ref["run_id"], "path": SANDBOX_REFERENCE["path"],
                          "sha256_lf": SANDBOX_REFERENCE["sha256_lf"], "environment": ref["environment"],
                          "views_available": ref["views_available"]},
            "required": required, "basis": ("the sandbox's environment and view availability" if required else
                                            "; ".join(reasons)),
            "table": table, "statements": st, "checks": pd.DataFrame(checks)}


def review_flags(repro: Mapping[str, Any], screen: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    """Everything flagged for review: items that changed against the sandbox, statements that no longer hold,
    and any cascade verdict that reselection changes (reported as such, not as a better cascade)."""
    t = repro["table"]
    rows = [{"source": "against the sandbox", "item": r["item"], "flag": r["flag"], "sandbox": r["sandbox"],
             "this_run": r["this_run"]} for _, r in t[t["review"]].iterrows()]
    st = repro["statements"]
    rows += [{"source": "review document", "item": r["check"], "flag": "statement no longer holds", "sandbox": None,
              "this_run": r["detail"]} for _, r in st[~st["passed"]].iterrows()]
    if screen is not None:
        rows += [{"source": "cascade screen", "item": r["candidate"], "flag": "reselection changes the cascade verdict "
                  "(not evidence of a better cascade)", "sandbox": r["five_a"], "this_run": r["reselected"]}
                 for _, r in screen["change"].iterrows() if r["conclusion_changes"]]
    return pd.DataFrame(rows, columns=["source", "item", "flag", "sandbox", "this_run"])


def reproduction_summary(repro: Mapping[str, Any]) -> pd.DataFrame:
    """Items per kind: how many, how many equal the sandbox's, how many differ and how many are flagged."""
    t = repro["table"]
    g = t.groupby("kind", sort=False)
    return pd.DataFrame({"items": g.size(), "equal": g["equal"].sum(), "differ": g["equal"].apply(lambda e: int((~e).sum())),
                         "flagged": g["review"].sum()}).reset_index()


# ======================================================================
# 13. the record
# ======================================================================
def record_sections(*, reg_checks, history, accepted_5a, accepted_5a_checks, recovery, rec_checks, configs, fits,
                    fit_checks, tables, vs, cmp, shown, surfaces, value, views, screen, screen_required
                    ) -> Dict[str, Any]:
    """The interaction_selection record, before acceptance and reproduction (``final_sections``)."""
    def sel(s):
        return {k: s.get(k) for k in ("fit", "rows", "index_sha256", "features", "bags", "eligible_per_bag", "selected",
                                      "fitted", "selection_matches_fit", "margin", "ranking")}

    def surf(s):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    return {
        "registration": {"content": REGISTRATION, "sha256": registration_digest(), "checks": _recs(reg_checks),
                         "history": history},
        "accepted_5a": {"accepted": dict(ACCEPTED_5A), "comparison": _recs(accepted_5a),
                        "checks": _recs(accepted_5a_checks)},
        "recovery": {"features": recovery["features"], "selections": [sel(s) for s in recovery["selections"]],
                     "checks": _recs(rec_checks), "summary": recovery_summary(recovery, tables["baseline"]),
                     "oof_sha256": recovery["oof_sha256"]},
        "candidates": {name: {"features": configs[name]["features"], "five_a": configs[name]["five_a"],
                               "five_a_pairs": [pair_label(a, b) for a, b in configs[name]["five_a_pairs"]],
                               "budget": configs[name]["budget"], "selections": [sel(s) for s in r["selections"]],
                               "threshold_raw": r["threshold_raw"], "threshold_calibrated": r["threshold_calibrated"],
                               "nested_by_fold": [float(v) for v in r["nested"]["by_fold"]],
                               "calibration": r["calibration"], "oof_raw_sha256": _sha(r["proba"]),
                               "oof_calibrated_sha256": _sha(r["calibrated"]), "terms": r["terms"]}
                       for name, r in fits.items()},
        "fit_checks": _recs(fit_checks),
        "recurrence": {name: _recs(t) for name, t in tables.items()},
        "change": _recs(change_summary(tables)),
        "against_5a": {"table": _recs(cmp), "fold_differences": {
            n: _recs(c["fold_differences"]) for n, c in vs.items()}, "decisions": {n: c["decisions"] for n, c in vs.items()}},
        "surfaces": {"shown": list(shown), "cells": [surf(s) for s in surfaces], "value": _recs(value)},
        "complementarity": {"views": {"status": views["status"], "required": views["required"]},
                            "table": None if screen is None else _recs(screen["comp"]["table"]),
                            "against_reference": None if screen is None else _recs(screen["comp"]["against_reference"]),
                            "pooled": None if screen is None else _recs(screen["pooled"]),
                            "screen": None if screen is None else _recs(screen["screen"]),
                            "change": None if screen is None else _recs(screen["change"]),
                            "checks": _recs(pd.concat([c for c in ((screen or {}).get("checks"), screen_required)
                                                       if c is not None], ignore_index=True))},
        "limits": list(LIMITS),
    }


def final_sections(sections: Mapping[str, Any], acceptance: pd.DataFrame, repro: Mapping[str, Any],
                   review: pd.DataFrame) -> Dict[str, Any]:
    """The record as written (Cell 68, before either gate is enforced): the sections, the acceptance checks
    with their verdict, the sandbox reproduction and everything flagged for review."""
    return {**sections,
            "acceptance": {"passed": bool(acceptance["passed"].all()), "checks": _recs(acceptance)},
            "reproduction": {**repro["reference"], "required": repro["required"], "basis": repro["basis"],
                             "summary": _recs(reproduction_summary(repro)), "checks": _recs(repro["checks"]),
                             "table": _recs(repro["table"]), "statements": _recs(repro["statements"])},
            "review": {"flagged": int(len(review)), "items": _recs(review)}}


LIMITS = (
    "The selector is interpret's FAST ranking as implemented in interpret 0.7.8; another version may rank "
    "differently. The recorder observes interpret's internal ranking function and is checked against the "
    "fitted pairs in every fit.",
    "Selection uses each fit's balanced weights and its own bags' main-effect residuals; the pair strengths are "
    "not a test of interaction and the ranks are not comparable across feature sets.",
    "The comparison with 5A changes the pair count as well as the pairs (2 to 3): it measures the procedure as "
    "a whole. Budgets other than three are PR 44 Step 44C's.",
    "The recovery exercise on all training rows is descriptive; the outer-fold selections are the evidence of "
    "recurrence, and inner selections are supporting counts.",
    "Fit-dependent values differ slightly between platforms (Step 5A), and a pair near the selection boundary can "
    "change. Acceptance is checked in every run; the sandbox figures the review document quotes are reproduced "
    "exactly only in that sandbox's environment, and elsewhere their differences are recorded and changed "
    "selections, directions and verdicts are flagged for review. The Windows run is the accepted one.",
    "Complementarity needs the accepted views (the Windows run); its counts are that run's, and a changed "
    "screen verdict is not evidence of a better cascade (PRs 44-45).",
    "That the correlated campaign pairs share one campaign-by-regime effect is an interpretation of the observed "
    "surfaces, not a tested decomposition of attribution.",
    "A pair selected by fewer than two outer folds has no fold sd; it is reported as unavailable.",
    "Training side, random outer folds; chronological evaluation is PR 49's.",
)
