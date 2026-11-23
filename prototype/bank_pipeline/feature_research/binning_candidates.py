"""
feature_research.binning_candidates
===================================
PR 42 Commit 4, Step 4B: fold-fitted NSD/ECI binning procedures compared with
the frozen PR 39 bins, and ``eci_warm`` tested as a filter on partner rules.

The comparison is registered first (``REGISTRATION``; its SHA-256 is pinned in
``REGISTERED_SHA256`` and checked before any candidate is computed). Step 4A
(``feature_research.binning_drift``) motivated the hypotheses; what it had
established is stated in the registration, apart from what this step fixes.

Each procedure is a fit-and-apply pair. ``fit_*`` sees one outer fold's fit
side only: its rows, their labels, their macro states and the fold engineer's
value of each state. It returns three cut points. Applying them bins any value
with ``(lo, hi]`` intervals, as the frozen bins do. A held-out row is scored
with its outer fold's engineer (``stage2_binning.runner.oof_sources``, PR 39's)
and its fold's cuts. Label-using choices (the ``tail_selected`` tail, the
partner rules) use inner cross-fitted values: each fit row is valued by an
engineer fitted on the other four fifths of the fit side, through the same
``oof_sources``. No held-out label enters anything fitted for its own fold.

Training side only. The frozen binning (``shared.stage2.binning``),
``stage2_binning`` and the accepted PR 39 and Step 4A records are read, never
changed. Like Step 4A, this module imports no feature-engineering module, so
the Commit 2 caller scan and the Step 3A digests are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_research import binning_drift as BD

__all__ = ["SEED", "N_FOLDS", "INNER_FOLDS", "FAMILIES", "FOCAL", "FILTER_BIN", "PROCEDURES",
           "CANDIDATES", "FROZEN_SUPPORTS", "REGISTRATION", "REGISTERED_SHA256", "registration_digest",
           "registration_checks", "registration_table", "frozen_shares", "window_halfwidths",
           "state_rows", "blocks", "boundaries", "fit_frozen", "fit_share_matched", "fit_gap_placed",
           "fit_tail_selected", "apply_fit", "heldout_sources", "inner_sources", "fit_fold", "fit_folds",
           "deployment_fits", "cut_table", "isolation_checks", "compare", "dependence", "partner_pool",
           "partner_stats", "select_partners", "partition_counts", "warm_filter", "decide", "STATED", "stated_checks",
           "LIMITS", "record_sections", "require", "reference_inputs", "REFERENCE_SCOPE", "ISOLATION_SCOPE",
           "FILTER_SCOPE", "SENSITIVITY", "fold_local_reference", "fit_fold_local", "sensitivity_all_rows",
           "sensitivity", "STATED_SENSITIVITY", "sensitivity_checks"]

BANK_ROOT = BD.BANK_ROOT
SEED = BD.SEED
N_FOLDS = BD.N_FOLDS
INNER_FOLDS = 5
FAMILIES = BD.FAMILIES
#: The bin each family's candidates are judged on (Step 4A's focal bins; eci_warm is judged as a filter).
FOCAL = {"NSD": "nsd_elevated", "ECI": "eci_hot"}
FILTER_BIN = "eci_warm"
PROCEDURES = ("frozen", "share_matched", "gap_placed", "tail_selected")
CANDIDATES = {"NSD": ("share_matched", "gap_placed"),
              "ECI": ("share_matched", "gap_placed", "tail_selected")}
#: Simplest first; the order breaks ties between advancing candidates.
SIMPLICITY = {"share_matched": 0, "gap_placed": 1, "tail_selected": 2}

#: The frozen bins' in-sample supports on the 32,950 training rows (accepted PR 39 record, E-B
#: ``families.<F>.in_sample``; ``registration_checks`` compares them with the record).
FROZEN_SUPPORTS = {"NSD": (22120, 6839, 1326, 2665), "ECI": (7392, 9078, 12491, 3989)}
TRAINING_ROWS = 32950
FROZEN_BINNING_SHA256 = "9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376"
FOLD_FINGERPRINT = "8aea082fcaf371b7"

WINDOW_FRACTION = 0.25
TAIL_SHARE_MIN_FACTOR = 0.5            # = SUPPORT_FRACTION: a smaller tail could not meet A2
TAIL_SHARE_MAX_FACTOR = 2.0

#: Partner rules and filter variants. The GLASS values are the frozen primary profile's
#: (``glass_router_primary_v1.json``) and the union values the frozen shared contract's
#: (``stage2_primary_v1.json``); ``registration_checks`` compares them with both files.
GLASS = {"max_complexity": 3, "min_support_pass2": 25, "min_precision_subscribe": 0.2,
         "min_recall_subscribe": 0.01, "max_recall_subscribe": 0.35, "max_jaccard_overlap": 0.5}
UNION = {"min_pass2_precision": 0.3, "min_pass2_support": 75}
FIT_TO_HELDOUT = 4                       # 26,360 fit rows : 6,590 held-out rows per outer fold
PARTNER_K = 5
PARTNER_MAX_LITERALS = 2
PARTNER_FIT_SUPPORT_MIN = UNION["min_pass2_support"] * FIT_TO_HELDOUT          # 300
VARIANT_FIT_SUPPORT_MIN = GLASS["min_support_pass2"] * FIT_TO_HELDOUT         # 100
FILTER_GAIN = 1.10
FOLD_MAJORITY = 4
SLOT_MAJORITY = 3
SUPPORT_FRACTION = 0.5
HOT_LIFT_GAIN = 1.10
PERIOD_MIN_ROWS = BD.PERIOD_MIN_ROWS
WILSON_Z = 1.959963984540054             # stage2_binning.evidence.CRITERIA["T2"]["z"]
#: Distances, gaps and bounds within this of each other are equal (floating-point ties are ties).
TIE_TOLERANCE = 1e-12


def frozen_shares(family: str) -> Tuple[float, float, float]:
    """The frozen bins' cumulative in-sample row shares below each cut (``FROZEN_SUPPORTS``)."""
    s = np.cumsum(FROZEN_SUPPORTS[family])[:3] / TRAINING_ROWS
    return tuple(round(float(v), 6) for v in s)


def window_halfwidths(family: str) -> Tuple[float, float, float]:
    """``gap_placed``'s window half-width per cut: WINDOW_FRACTION x the smaller adjacent bin share."""
    share = np.asarray(FROZEN_SUPPORTS[family], dtype=float) / TRAINING_ROWS
    return tuple(round(float(WINDOW_FRACTION * min(share[j], share[j + 1])), 6) for j in range(3))


def _tail_range() -> List[float]:
    hot = FROZEN_SUPPORTS["ECI"][3] / TRAINING_ROWS
    return [round(TAIL_SHARE_MIN_FACTOR * hot, 6), round(TAIL_SHARE_MAX_FACTOR * hot, 6)]


# ======================================================================
# 1. registration (fixed before any candidate result was computed)
# ======================================================================
REGISTRATION = {
    "id": "pr42-step4b-registration",
    "version": "1.0.0",
    "revisions": [{"version": "1.0.0", "status": "fixed before any candidate result was computed"}],
    "question": (
        "On identical held-out training rows, do fold-fitted derivations of the NSD and ECI bins improve "
        "on the frozen PR 39 bins enough, at usable support and with fold stability, to earn a PR 44 "
        "router comparison? And does the frozen eci_warm, used as a filter on partner rules, help "
        "identify subscribers or exclude false positives at usable support?"),
    "known_before_registration": {
        "source": (
            "Step 4A (feature_research.binning_drift; record binning_drift.json, review document "
            "research_logs/pr42/binning_drift.md), accepted on Windows as run c4_20261123_081749, and "
            "the PR 39 Step 3A record (pinned SHA-256). Cells 31-39 recompute Step 4A in the same run, "
            "and require every 4A claim, before this registration is printed."),
        "eci_warm": (
            "No drift in the bin: the quoted 1.49-1.67x were two historical deciles inside it; the "
            "historical whole bin was 0.92x. On the training rows the bin is 1.038x in sample and "
            "1.031x out of fold, with rows from periods between 0.26x and 5.5x."),
        "eci_hot": (
            "Pooled out-of-fold lift 2.501x (in sample 2.457x). In sample the bin is the 2,533 rows of "
            "periods at or above the base rate (3.48x) plus all rows of four May 2009 macro states "
            "(1,456 rows, 0.68x); all six training-side encoders place those four states above the "
            "frozen cut 0.0948."),
        "nsd_elevated": (
            "Pooled out-of-fold lift 2.486x, folds 0.62-3.36x. Fold 0's encoder moved whole macro states "
            "across both cuts (seven May 2009 states into the bin, the 50 in-sample states into nsd_hot); "
            "all 653 rows entering or leaving the bin between the in-sample and out-of-fold encoders are "
            "fold-0 rows."),
        "both": (
            "NSD and ECI are functions of the macro state (euribor3m, nr.employed, emp.var.rate): 346 "
            "states over 26 periods, so a cut moves whole states. NSD 0.192 / 0.23 and ECI 0.0291 / "
            "0.0758 lie inside the value range some state takes across the six encoders."),
        "motivated": {
            "share_matched": (
                "nsd_elevated: fold 0's encoder shifted whole states across fixed cuts. If the shift is "
                "a change of scale, cuts placed at the frozen bins' row shares of each fold engineer's "
                "own values restore the membership."),
            "gap_placed": (
                "both families: cuts that land among closely spaced states let a small change between "
                "encoders move states across them. A cut placed in the widest gap between fit-side "
                "states near the frozen share leaves the most room."),
            "tail_selected": (
                "eci_hot: every encoder places four low-rate May 2009 states above the frozen cut, so no "
                "label-free re-placement of the frozen structure can remove them. A label-using choice "
                "of the upper tail, made on the fit side, can try, at a cost in support."),
            "eci_warm_filter": (
                "eci_warm's whole-bin lift is about 1.03x; its value, if any, is as a condition on "
                "another rule. It keeps the frozen definition and is judged as a filter, not re-cut."),
        },
        "not_computed_before_registration": (
            "No candidate cut, partner rule, filter variant or held-out statistic of one had been "
            "computed when this registration was fixed. Its design used the Step 4A findings above, the "
            "PR 39 record's in-sample bin supports, and the frozen GLASS profile and shared contract."),
    },
    "comparator": {
        "name": "frozen",
        "definition": (
            "the frozen PR 39 bins (shared.stage2.binning.BINNING_STRATEGY, identity "
            f"{FROZEN_BINNING_SHA256}) applied to each held-out row's out-of-fold value from "
            "stage2_binning.runner.oof_sources: the PR 39 E-B out-of-fold tables"),
        "parity": ("the comparator computed here must equal E-B's out-of-fold table for both families, "
                   "and the frozen cuts applied through the candidates' apply path must equal the "
                   "production bins row by row"),
    },
    "rows_and_folds": {
        "rows": (f"the {TRAINING_ROWS:,} training rows; the test split is never bound (G7)"),
        "outer": (f"make_folds(y, {N_FOLDS}, {SEED}), fingerprint {FOLD_FINGERPRINT} (PR 39's): each "
                  "held-out row is scored once, by its own fold's engineer and fold-fitted cuts"),
        "inner": (f"make_folds(y_fit, {INNER_FOLDS}, {SEED}) on each outer fit side; inner values "
                  "come from stage2_binning.runner.oof_sources on that fit side"),
        "identical_rows": ("every procedure, the comparator included, is evaluated on the same held-out "
                           "rows with the same out-of-fold values; only the cuts differ"),
    },
    "values": {
        "scoring": ("the outer fold engineer's value of each macro state; held-out rows are scored "
                    "with it. Label-free placements (share_matched, gap_placed) place cuts among the "
                    "fit-side states on these values, each state weighted by its fit rows"),
        "selection": ("inner cross-fitted values: each fit row valued by an inner engineer that never "
                      "saw its label. A state's selection value is the mean over its fit rows. Every "
                      "label-using choice (tail_selected's tail; partner rules and their warm "
                      "variants, binned with the frozen cuts) uses these"),
        "deployment": ("for refit stability only: the same procedure fitted on all training rows "
                       "(scoring: the all-rows engineer's in-sample values; selection: the PR 39 "
                       "out-of-fold values). Never evaluated"),
    },
    "states_and_ties": (
        f"fit-side states whose values differ by at most {BD.STATE_TOLERANCE:g} form one block. Cuts lie "
        "only between blocks, at the midpoint of the two neighbouring values, so a cut never splits a "
        "macro state on the fit side and never equals a fit-side value. A held-out value equal to a "
        "cut goes to the lower bin ((lo, hi] intervals). Equal distances to a target share: the lower "
        "boundary. Equal gaps: the boundary nearest the target share, then the lower one"),
    "procedures": {
        "share_matched": {
            "families": ["NSD", "ECI"], "uses_labels": "no (beyond the fold engineer's own fit)",
            "rule": ("cut j at the block boundary whose fit-row share below is nearest q_j, the frozen "
                     "bins' cumulative in-sample row share"),
            "targets": {f: list(frozen_shares(f)) for f in FAMILIES},
            "engineer_fits_per_outer_fold": 0,
        },
        "gap_placed": {
            "families": ["NSD", "ECI"], "uses_labels": "no (beyond the fold engineer's own fit)",
            "rule": (f"cut j at the widest value gap among boundaries whose fit-row share below lies "
                     f"within q_j +/- h_j, h_j = {WINDOW_FRACTION} x the smaller frozen share of the two "
                     "bins the cut separates (each bin keeps at least half its share outside both "
                     "windows). An empty window falls back to share_matched's boundary"),
            "halfwidths": {f: list(window_halfwidths(f)) for f in FAMILIES},
            "engineer_fits_per_outer_fold": 0,
        },
        "tail_selected": {
            "families": ["ECI"], "uses_labels": "yes, fit side only, on selection values",
            "rule": ("cuts 1-2 as share_matched. On the selection values, among block boundaries whose "
                     f"upper tail holds a fit-row share in {_tail_range()} ({TAIL_SHARE_MIN_FACTOR} to "
                     f"{TAIL_SHARE_MAX_FACTOR} x the frozen eci_hot share; a smaller tail could not "
                     "meet A2), choose the tail with the "
                     "largest Wilson 95% lower bound of its fit-side subscriber rate (ties: more rows, then "
                     "the lower boundary). Cut 3 is the scoring-value boundary above cut 2 whose tail "
                     "share is nearest the chosen share (ties: the larger tail)"),
            "tail_share_range": _tail_range(), "wilson_z": WILSON_Z,
            "engineer_fits_per_outer_fold": INNER_FOLDS,
            "why_ECI_only": ("Step 4A ties NSD's instability to an encoder moving whole states, which the "
                             "label-free placements address; a label-using search there would add "
                             "selection variance without targeting that cause"),
        },
    },
    "partner_rules": {
        "pool": ("conjunctions of one or two `bin = 1` literals from the 25 non-ECI frozen bins, from "
                 "different families (GLASS's base-feature rule), so partner AND eci_warm (= 1 or = 0) has "
                 f"at most {GLASS['max_complexity']} conditions (GLASS max_complexity): 25 singles and 263 "
                 "pairs. `bin = 0` literals, which GLASS also uses, are left to PR 44"),
        "eligibility_fit_side": {
            "precision": f">= {GLASS['min_precision_subscribe']} (GLASS min_precision_subscribe)",
            "recall": (f"in [{GLASS['min_recall_subscribe']}, {GLASS['max_recall_subscribe']}] of the fit "
                       "side's subscribers (GLASS min/max_recall_subscribe)"),
            "support": (f">= {PARTNER_FIT_SUPPORT_MIN} = {UNION['min_pass2_support']} x {FIT_TO_HELDOUT} "
                        "(the union support obligation at the 4:1 fit-to-held-out ratio)"),
            "variants": (f"partner AND eci_warm and partner AND NOT eci_warm each >= "
                         f"{VARIANT_FIT_SUPPORT_MIN} = {GLASS['min_support_pass2']} x {FIT_TO_HELDOUT} "
                         "fit rows (GLASS min_support_pass2 at the same ratio)"),
        },
        "ranking": ("fit-side Wilson 95% lower bound of precision, descending; ties: more fit rows, fewer "
                    "literals, then name"),
        "selection": (f"greedy down the ranking, the first {PARTNER_K} partners whose fit-row Jaccard "
                      f"overlap with every partner already chosen is <= {GLASS['max_jaccard_overlap']} "
                      "(GLASS max_jaccard_overlap), so they are not near-copies of one rule. Fewer "
                      "eligible: those there are"),
        "slots": "slot r is the r-th partner chosen in a fold; partners may differ between folds",
    },
    "filter_variants": {
        "partner_alone": "the held-out rows of the fold satisfying the partner",
        "warm_required": "those rows with eci_warm = 1",
        "warm_excluded": "those rows with eci_warm = 0",
        "eci_warm": "the frozen bin on the held-out row's out-of-fold ECI (identical to E-B's labels)",
        "same_partner": "the three variants of a slot use the same partner within a fold",
        "reconciliation": ("required and excluded are disjoint, their union is the partner's rows, and "
                           "their rows, subscribers and non-subscribers add up to the partner's"),
        "removed": ("warm_required removes the partner's eci_warm = 0 rows, warm_excluded its "
                    "eci_warm = 1 rows; each count of subscribers and non-subscribers removed is reported"),
    },
    "population": (
        "Diagnostic: every held-out training row of the fold. The Pass 2 obligations (shared contract: "
        "precision >= 0.30 and support >= 75 for the selected union, on the Pass 2 population, the "
        "remainder after Pass 1, at fit time) cannot be applied at their defined scope: no per-row "
        "Pass 1 route is tracked and rebuilding one needs a GLASS Pass 1 fit (PR 44). Here they are a "
        "screen applied to one conjunction on the broader population, pooled over the five folds (the "
        "same row scale as the full training side GLASS fits on); the historical remainder's prevalence "
        "was about 0.158 against the 0.113 base rate. Passing the screen is not compliance"),
    "metrics": [
        "per procedure, family and fold: the learned cuts, their fit-row shares and gaps, fit-side rows "
        "and states per bin, feasibility",
        "per bin, pooled and per fold (stage2_binning.evidence.bin_table): support, subscribers, rate, "
        "lift and its Wilson 95% interval; PR 39's T1-T4 (evidence.triggers)",
        "focal bin: smallest and largest fold lift, smallest fold support, refit migration (held-out rows "
        "whose focal membership differs between their fold's fit and the all-rows fit)",
        "dependence: focal lift with each outer fold removed and with each contact period of at least "
        f"{PERIOD_MIN_ROWS} rows removed, against the frozen bin on the same rows; the focal support ratio",
        "complementarity: NSD x ECI joint rows and subscribers with the other family frozen",
        "filter: per fold and slot, pooled per slot and per partner: rows, subscribers, non-subscribers, "
        "precision with Wilson 95%, and the subscribers and non-subscribers each filter removes",
    ],
    "criteria": {
        "A0_feasible": "a valid fit in all five folds (strictly increasing cuts; every bin holds a fit-side state)",
        "A1_triggers": "no PR 39 trigger (T1-T4) fires for the candidate's family on the held-out rows",
        "A2_coverage": (f"in every fold the focal bin's held-out support is >= {SUPPORT_FRACTION} x the frozen "
                        "focal bin's support in that fold"),
        "A3_NSD": ("nsd_elevated: (a) its smallest fold lift exceeds the frozen bin's; (b) its pooled lift "
                   "is >= the frozen bin's pooled Wilson 95% lower lift bound; (c) its refit migration is "
                   "<= the frozen bin's"),
        "A3_ECI": (f"eci_hot: (a) pooled lift >= {HOT_LIFT_GAIN} x the frozen bin's; (b) its smallest fold "
                   "lift >= the frozen bin's"),
        "A4_robust": ("with any one outer fold removed, and with any one contact period of at least "
                      f"{PERIOD_MIN_ROWS} rows removed, the candidate's pooled focal lift stays >= the "
                      "frozen bin's (eci_hot) or >= the frozen bin's pooled Wilson 95% lower lift bound "
                      "(nsd_elevated), on the same remaining rows"),
        "ranking": ("advancing candidates of one family: NSD by smallest fold lift, ECI by pooled Wilson "
                    "95% lower lift bound; ties: the simpler procedure (share_matched, gap_placed, "
                    "tail_selected)"),
        "filter_slot": (f"a slot shows a filter effect in a direction (required or excluded) if that "
                        f"variant's pooled precision is >= {FILTER_GAIN} x the partner's, its precision "
                        f"exceeds the partner's in >= {FOLD_MAJORITY} folds (the slot needs >= "
                        f"{FOLD_MAJORITY} folds), and it passes the union screen pooled (support >= "
                        f"{UNION['min_pass2_support']}, precision >= {UNION['min_pass2_precision']})"),
        "filter_verdict": (f"eci_warm advances to PR 44 as a filter in a direction if >= {SLOT_MAJORITY} "
                           f"of the {PARTNER_K} slots show the effect in that direction; otherwise it is "
                           "closed out as a filter"),
    },
    "handling": {
        "ties": ("cuts and boundaries: states_and_ties; the tail choice: procedures.tail_selected; "
                 "partners: partner_rules.ranking"),
        "sparse_or_empty": (
            "a fit-side bin with no state makes the fit infeasible in that fold. A held-out bin with no "
            "row has no lift and fires T1 (PR 39's convention). A filter variant with no held-out row in "
            "a fold has no precision there and counts as not exceeding the partner. A fold with fewer "
            "eligible partners evaluates those it has"),
        "infeasible": ("an infeasible fold is recorded with its reason and nothing is substituted; the "
                       "candidate fails A0 and its held-out tables are not formed"),
        "lift_versus_support": (
            "tail_selected trades lift against support through the Wilson lower bound, within the "
            "registered share range. No candidate advances on lift bought with support below A2's "
            "floor, and a filter effect counts only if the variant passes the union screen"),
    },
    "decisions": {
        "per_candidate": "advance to PR 44 (A0-A4 hold and it ranks first in its family) or close out, with the reason",
        "per_family": ("the frozen bins are kept in PR 42 in every case. If a candidate advances, the "
                       "family's record states the versioned revision it would need (contract, G13 "
                       "audit, regeneration); PR 44 owns the router comparison and adoption"),
        "eci_warm_filter": "a separate verdict: advance in a direction, or close out as a filter",
    },
    "out_of_scope": [
        "router fits, Pass 1 routes and the Pass 2 population (PR 44)",
        "chronological validity of any candidate (PR 49)",
        "any change to the frozen bins, contracts, artifacts or accepted records",
    ],
}

#: SHA-256 of the 1.0.0 registration, fixed before any candidate result was computed.
REGISTERED_SHA256 = "066195f04c96c5873952c02d4fd80707e32ef51af8a7a6926153ecd49ddf35f5"


def registration_digest(registration: Mapping[str, Any] = REGISTRATION) -> str:
    """SHA-256 of the canonical registration JSON (recorded beside the results)."""
    return hashlib.sha256(json.dumps(registration, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _check(name: str, passed: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


require = BD.require
_canonical = BD._canonical


def _family_of(bin_name: str) -> str:
    return bin_name.split("_")[0]


def _partner_bins() -> Tuple[str, ...]:
    from shared.stage2.binning import RF_FEATURES_BINARY
    return tuple(b for b in RF_FEATURES_BINARY if _family_of(b) != "eci")


def registration_checks(pr39_record: Mapping[str, Any], root: Path = BANK_ROOT) -> pd.DataFrame:
    """
    Before any candidate is computed: the registration is the pinned 1.0.0, its frozen supports are
    the PR 39 record's, its GLASS and union values are the frozen profile's and contract's, the frozen
    binning is unchanged, and the partner pool is what GLASS can express.
    """
    from shared.stage2.binning import binning_identity
    from stage2_binning import evidence as E
    out = []
    digest = registration_digest()
    out.append(_check("the registration's SHA-256 is the pinned 1.0.0 digest", digest == REGISTERED_SHA256, digest))
    for fam in FAMILIES:
        ins = pr39_record["eb"]["families"][fam]["in_sample"]
        sup = tuple(int(ins["bins"][b]["pooled"]["support"]) for b in E.family_bins(fam))
        out.append(_check(f"{fam}: the registered frozen supports are the PR 39 record's in-sample supports",
                          sup == FROZEN_SUPPORTS[fam] and ins["base_rate"]["pooled"]["rows"] == TRAINING_ROWS,
                          sup))
    ident = binning_identity()["sha256"]
    out.append(_check("the frozen binning identity is unchanged", ident == FROZEN_BINNING_SHA256, ident[:16]))
    profile = json.loads((root / "glass_pipeline" / "glass_router" / "profiles" /
                          "glass_router_primary_v1.json").read_text(encoding="utf-8"))
    rc = profile["settings"]["router_config"]
    glass = {k: rc[k] for k in GLASS}
    out.append(_check("the GLASS values are the frozen primary profile's", glass == GLASS and
                      profile["status"] == "frozen", glass))
    contract = json.loads((root / "shared" / "stage2" / "contracts" / "stage2_primary_v1.json")
                          .read_text(encoding="utf-8"))
    union = {k: contract[k] for k in UNION}
    out.append(_check("the union values are the frozen shared contract's", union == UNION and
                      contract["status"] == "frozen", union))
    from glass_pipeline.glass_router.rule_generator.feature_validator import _BIN_FAMILY
    pool = partner_pool()
    fams = [[_BIN_FAMILY[b] for b in lits] for lits in pool]
    out.append(_check(f"the partner pool holds {len(pool)} conjunctions (25 singles, 263 pairs) of non-ECI bins "
                      "from distinct GLASS families",
                      len(pool) == 288 and all(len(set(f)) == len(f) and "eci" not in f for f in fams), len(pool)))
    return pd.DataFrame(out)


def registration_table() -> pd.DataFrame:
    """The registered procedures and criteria, one line each, for printing."""
    rows = [{"item": f"procedure {k}", "registered": v["rule"]} for k, v in REGISTRATION["procedures"].items()]
    rows += [{"item": f"criterion {k}", "registered": v} for k, v in REGISTRATION["criteria"].items()]
    rows += [{"item": f"motivated by 4A: {k}", "registered": v}
             for k, v in REGISTRATION["known_before_registration"]["motivated"].items()]
    return pd.DataFrame(rows)


# ======================================================================
# 2. fit and apply
# ======================================================================
def state_rows(codes: np.ndarray, values: np.ndarray, y: np.ndarray, *, exact: bool) -> pd.DataFrame:
    """
    One row per macro state present: its value, rows and positives. ``exact``: the values are one
    encoder's (constant within a state, checked) and the state's value is that value; otherwise the
    state's value is the mean over its rows (inner cross-fitted values differ between inner folds).
    """
    df = pd.DataFrame({"state": np.asarray(codes), "v": np.asarray(values, dtype=float),
                       "y": np.asarray(y).astype(int)})
    if df["v"].isna().any():
        raise ValueError("NaN values cannot be placed among states")
    g = df.groupby("state", sort=True)
    spread = (g["v"].max() - g["v"].min())
    if exact and float(spread.max()) > BD.STATE_TOLERANCE:
        raise ValueError(f"values are not constant within a state (spread {float(spread.max()):.2e})")
    return pd.DataFrame({"value": g["v"].first() if exact else g["v"].mean(), "spread": spread,
                         "rows": g.size(), "positives": g["y"].sum()})


def blocks(states: pd.DataFrame, tol: float = BD.STATE_TOLERANCE) -> pd.DataFrame:
    """States in value order; states whose values differ by at most ``tol`` form one block."""
    s = states.sort_values("value", kind="mergesort")
    v = s["value"].to_numpy(dtype=float)
    bid = np.cumsum(np.r_[True, np.diff(v) > tol]) - 1
    g = pd.DataFrame({"block": bid, "value": v, "rows": s["rows"].to_numpy(),
                      "positives": s["positives"].to_numpy()}).groupby("block", sort=True)
    return pd.DataFrame({"lo": g["value"].min(), "hi": g["value"].max(), "rows": g["rows"].sum(),
                         "positives": g["positives"].sum(), "states": g.size()}).reset_index(drop=True)


def boundaries(blk: pd.DataFrame) -> pd.DataFrame:
    """Each boundary between consecutive blocks: the fit-row share below it, the gap, the midpoint cut."""
    rows = blk["rows"].to_numpy(dtype=np.int64)
    pos = blk["positives"].to_numpy(dtype=np.int64)
    cum, cpos = np.cumsum(rows), np.cumsum(pos)
    lower, upper = blk["hi"].to_numpy()[:-1], blk["lo"].to_numpy()[1:]
    return pd.DataFrame({"rows_below": cum[:-1], "positives_below": cpos[:-1],
                         "share_below": cum[:-1] / cum[-1] if len(cum) else [],
                         "lower_value": lower, "upper_value": upper, "gap": upper - lower,
                         "cut": (lower + upper) / 2.0})


def _frozen_cuts(family: str) -> List[float]:
    from shared.stage2.binning import BINNING_STRATEGY
    return [float(hi) for _, _, hi in BINNING_STRATEGY[family]["bins"][:3]]


def _nearest(bnd: pd.DataFrame, target: float, allowed: Optional[np.ndarray] = None) -> Optional[int]:
    """The boundary whose share below is nearest ``target``; equal distances: the lower one."""
    d = np.abs(bnd["share_below"].to_numpy() - target)
    if allowed is not None:
        d = np.where(allowed, d, np.inf)
    if not len(d) or not np.isfinite(d).any():
        return None
    return int(np.flatnonzero(d <= d.min() + TIE_TOLERANCE)[0])


def _result(family: str, procedure: str, blk: pd.DataFrame, bnd: pd.DataFrame,
            idx: Optional[Sequence[Optional[int]]], per_cut: List[Dict[str, Any]],
            extra: Optional[Dict[str, Any]] = None, reason: str = "") -> Dict[str, Any]:
    ok = (idx is not None and len(idx) == 3 and all(i is not None for i in idx)
          and all(b > a for a, b in zip(idx, idx[1:])))
    out = {"family": family, "procedure": procedure, "feasible": bool(ok), "reason": "" if ok else
           (reason or "the chosen boundaries are not strictly increasing, so a bin would hold no "
                      "fit-side state"),
           "fit_rows": int(blk["rows"].sum()), "fit_states": int(blk["states"].sum()),
           "fit_blocks": int(len(blk)), "cuts": None, "per_cut": per_cut,
           "bin_rows": None, "bin_states": None, "bin_positives": None}
    if ok:
        i = list(idx)
        b = bnd.iloc[i]
        out["cuts"] = [float(c) for c in b["cut"]]
        cum = list(b["rows_below"].astype(int)) + [out["fit_rows"]]
        out["bin_rows"] = [int(cum[0])] + [int(cum[j] - cum[j - 1]) for j in (1, 2, 3)]
        cpos = list(b["positives_below"].astype(int)) + [int(blk["positives"].sum())]
        out["bin_positives"] = [int(cpos[0])] + [int(cpos[j] - cpos[j - 1]) for j in (1, 2, 3)]
        st = np.cumsum(blk["states"].to_numpy())
        cst = [int(st[j]) for j in i] + [int(st[-1])]
        out["bin_states"] = [cst[0]] + [cst[j] - cst[j - 1] for j in (1, 2, 3)]
    if extra:
        out.update(extra)
    return out


def _cut_entry(bnd: pd.DataFrame, i: Optional[int], target: float, chosen_by: str,
               window: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    e = {"target_share": target, "window": None if window is None else [float(w) for w in window],
         "chosen_by": chosen_by, "boundary": None if i is None else int(i)}
    if i is not None:
        r = bnd.iloc[i]
        e.update({"cut": float(r["cut"]), "share_below": float(r["share_below"]), "gap": float(r["gap"]),
                  "lower_value": float(r["lower_value"]), "upper_value": float(r["upper_value"])})
    return e


def fit_frozen(family: str, blk: pd.DataFrame) -> Dict[str, Any]:
    """The comparator: the frozen cuts, described on the fit side (their fit-row share below)."""
    cuts = _frozen_cuts(family)
    rows, pos, states = (blk[c].to_numpy() for c in ("rows", "positives", "states"))
    lo = blk["lo"].to_numpy()
    total = int(rows.sum())
    below = [int(rows[lo <= c].sum()) for c in cuts]
    pbelow = [int(pos[lo <= c].sum()) for c in cuts]
    sbelow = [int(states[lo <= c].sum()) for c in cuts]
    return {"family": family, "procedure": "frozen", "feasible": True, "reason": "",
            "fit_rows": total, "fit_states": int(states.sum()), "fit_blocks": int(len(blk)), "cuts": cuts,
            "per_cut": [{"cut": c, "share_below": b / total if total else None, "chosen_by": "frozen"}
                        for c, b in zip(cuts, below)],
            "bin_rows": [below[0], below[1] - below[0], below[2] - below[1], total - below[2]],
            "bin_positives": [pbelow[0], pbelow[1] - pbelow[0], pbelow[2] - pbelow[1],
                              int(pos.sum()) - pbelow[2]],
            "bin_states": [sbelow[0], sbelow[1] - sbelow[0], sbelow[2] - sbelow[1],
                           int(states.sum()) - sbelow[2]]}


def fit_share_matched(family: str, blk: pd.DataFrame) -> Dict[str, Any]:
    """Cut j at the boundary whose fit-row share below is nearest the frozen share q_j."""
    bnd = boundaries(blk)
    q = frozen_shares(family)
    idx = [_nearest(bnd, t) for t in q]
    return _result(family, "share_matched", blk, bnd, idx,
                   [_cut_entry(bnd, i, t, "nearest share") for i, t in zip(idx, q)])


def fit_gap_placed(family: str, blk: pd.DataFrame, *, shares: Optional[Sequence[float]] = None,
                   halfwidths: Optional[Sequence[float]] = None, procedure: str = "gap_placed") -> Dict[str, Any]:
    """
    Cut j at the widest gap among boundaries within q_j +/- h_j; an empty window: the nearest boundary.
    Registered: q_j = ``frozen_shares``, h_j = ``window_halfwidths`` (full-training constants). The
    post-registration sensitivity passes fold-local ones (``fold_local_reference``); the placement is the same.
    """
    bnd = boundaries(blk)
    share, gap = bnd["share_below"].to_numpy(), bnd["gap"].to_numpy()
    shares = frozen_shares(family) if shares is None else shares
    halfwidths = window_halfwidths(family) if halfwidths is None else halfwidths
    idx, per = [], []
    for t, h in zip(shares, halfwidths):
        inside = np.flatnonzero(np.abs(share - t) <= h)
        if not len(inside):
            i, how = _nearest(bnd, t), "nearest share (empty window)"
        else:
            widest = inside[gap[inside] >= gap[inside].max() - TIE_TOLERANCE]
            d = np.abs(share[widest] - t)
            i, how = int(widest[np.flatnonzero(d <= d.min() + TIE_TOLERANCE)[0]]), "widest gap in window"
        idx.append(i)
        per.append(_cut_entry(bnd, i, t, how, (t - h, t + h)))
    return _result(family, procedure, blk, bnd, idx, per)


def _wilson_lower(pos: np.ndarray, n: np.ndarray, z: float = WILSON_Z) -> np.ndarray:
    pos, n = np.asarray(pos, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = pos / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
        lo = np.maximum(0.0, centre - half)
    return np.where(n > 0, lo, np.nan)


def fit_tail_selected(family: str, blk: pd.DataFrame, sel: pd.DataFrame) -> Dict[str, Any]:
    """
    Cuts 1-2 as ``share_matched`` on the scoring blocks ``blk``. The upper tail is chosen on the
    selection blocks ``sel`` (inner cross-fitted values, fit labels): the largest Wilson lower bound
    of the tail's subscriber rate among tails whose fit-row share is in the registered range. Cut 3
    is placed on the scoring values at the boundary above cut 2 whose tail share is nearest it.
    """
    if family != "ECI":
        raise ValueError("tail_selected is registered for ECI only")
    bnd = boundaries(blk)
    q = frozen_shares(family)
    i1, i2 = _nearest(bnd, q[0]), _nearest(bnd, q[1])
    per = [_cut_entry(bnd, i1, q[0], "nearest share"), _cut_entry(bnd, i2, q[1], "nearest share")]
    sb = boundaries(sel)
    total, tpos = int(sel["rows"].sum()), int(sel["positives"].sum())
    tail_rows = total - sb["rows_below"].to_numpy()
    tail_pos = tpos - sb["positives_below"].to_numpy()
    tail_share = tail_rows / total if total else np.array([])
    lo, hi = _tail_range()
    cand = np.flatnonzero((tail_share >= lo) & (tail_share <= hi))
    selection = {"tail_share_range": [lo, hi], "candidates": int(len(cand)), "chosen": None}
    if not len(cand):
        per.append(_cut_entry(bnd, None, float("nan"), "no selection boundary in the tail range"))
        return _result(family, "tail_selected", blk, bnd, None, per, {"selection": selection},
                       "no selection-value boundary leaves a tail in the registered share range")
    wl = _wilson_lower(tail_pos[cand], tail_rows[cand])
    best = cand[wl >= np.nanmax(wl) - TIE_TOLERANCE]
    best = best[tail_rows[best] == tail_rows[best].max()]
    b = int(best.min())
    tau = float(tail_share[b])
    selection["chosen"] = {"boundary": b, "selection_cut": float(sb["cut"].iloc[b]), "tail_share": tau,
                           "tail_rows": int(tail_rows[b]), "tail_positives": int(tail_pos[b]),
                           "tail_rate": float(tail_pos[b] / tail_rows[b]),
                           "wilson_lower": float(_wilson_lower(tail_pos[b], tail_rows[b]))}
    sc_tail = 1.0 - bnd["share_below"].to_numpy()
    above = np.arange(len(bnd)) > (i2 if i2 is not None else len(bnd))
    if not above.any():
        per.append(_cut_entry(bnd, None, 1.0 - tau, "no scoring boundary above cut 2"))
        return _result(family, "tail_selected", blk, bnd, None, per, {"selection": selection},
                       "the chosen tail share leaves no scoring boundary above cut 2")
    d = np.where(above, np.abs(sc_tail - tau), np.inf)
    i3 = int(np.flatnonzero(d <= d.min() + TIE_TOLERANCE)[0])     # the lowest index: the larger tail
    per.append(_cut_entry(bnd, i3, 1.0 - tau, "tail share nearest the selected share"))
    selection["scoring_tail_share"] = float(sc_tail[i3])
    return _result(family, "tail_selected", blk, bnd, [i1, i2, i3], per, {"selection": selection})


FIT = {"share_matched": fit_share_matched, "gap_placed": fit_gap_placed}


def apply_fit(values, fit: Mapping[str, Any]) -> np.ndarray:
    """Bin values with a fit's cuts ((lo, hi] intervals); the frozen fit through the production rule."""
    from stage2_binning import evidence as E
    if fit["procedure"] == "frozen":
        return BD.assign_bins(values, fit["family"])
    if not fit["feasible"]:
        raise ValueError(f"{fit['family']} {fit['procedure']}: an infeasible fit has no cuts to apply")
    return BD.assign_intervals(values, fit["cuts"], E.family_bins(fit["family"]))


# ======================================================================
# 3. fold fits
# ======================================================================
def heldout_sources(side: BD.Stage2Side, sources: BD.Sources) -> Dict[str, Any]:
    """
    All eight out-of-fold sources of every training row, binned (the partner rules' held-out bins),
    from ``stage2_binning.runner.oof_sources``: the same fits as Step 4A's, checked against them.
    """
    from stage2_binning import runner as R
    from stage2_crossfit.g13 import bin_source
    src, prov = R.oof_sources(side)
    src = src.loc[side.X_train.index]
    bins = bin_source(src).astype("int8")
    checks = [_check("the out-of-fold fits are Step 4A's (rows, positives, label fingerprints)",
                     _canonical(prov) == sources.provenance, "fold_fits")]
    for fam in FAMILIES:
        same = np.array_equal(src[BD.SOURCES[fam]].to_numpy(dtype=float), sources.values["oof"][fam])
        checks.append(_check(f"{fam}: the out-of-fold values are Step 4A's", same, fam))
    warm = sources.labels["oof"]["ECI"] == FILTER_BIN
    checks.append(_check("the held-out eci_warm is E-B's frozen bin on the out-of-fold ECI",
                         np.array_equal(bins[FILTER_BIN].to_numpy() == 1, warm), int(warm.sum())))
    checks.append(_check("every held-out bin is 0/1", bool(bins.isin([0, 1]).all().all()), bins.shape))
    return {"bins": bins, "checks": pd.DataFrame(checks)}


def inner_sources(side: BD.Stage2Side, k: int, *,
                  oof_fn: Optional[Callable[[BD.Stage2Side], Tuple[pd.DataFrame, list]]] = None
                  ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """
    Inner cross-fitted sources of outer fold ``k``'s fit rows: ``oof_sources`` on the fit side with
    the inner plan ``make_folds(y_fit, 5, 42)``. Only fit rows and fit labels are passed.
    """
    from stage2_binning import runner as R
    tr, va = side.folds[k]
    X, y = side.X_train.iloc[tr], side.y_train.iloc[tr].astype(int)
    inner_side = BD.stage2_side(X, y)
    src, prov = (oof_fn or R.oof_sources)(inner_side)
    src = src.loc[X.index]
    held = side.X_train.index[va]
    prov = _canonical(prov)
    checks = [
        _check(f"fold {k}: the inner side holds exactly the fold's {len(tr):,} fit rows",
               len(src) == len(tr) and not src.index.isin(held).any(), len(src)),
        _check(f"fold {k}: every inner fit excludes its own rows and the inner folds cover the fit side",
               all(p["heldout_rows_in_fit"] == 0 for p in prov) and
               sum(p["heldout_rows"] for p in prov) == len(tr) and
               all(p["fit_rows"] + p["heldout_rows"] == len(tr) for p in prov), len(prov)),
    ]
    return src, prov, pd.DataFrame(checks)


def _fit_family(family: str, sc: pd.DataFrame, se: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    fits = {"frozen": fit_frozen(family, sc)}
    for proc in CANDIDATES[family]:
        fits[proc] = fit_tail_selected(family, sc, se) if proc == "tail_selected" else FIT[proc](family, sc)
    return fits


def fit_fold(side: BD.Stage2Side, states: BD.MacroStates, state_vals: Mapping[str, Any], k: int, *,
             oof_fn=None) -> Dict[str, Any]:
    """
    Everything fitted for outer fold ``k``, from its fit side only: the inner cross-fitted values, each
    procedure's cuts per family, and the partner rules. ``state_vals`` gives fold k's engineer's value
    per macro state (``binning_drift.state_values``; Step 4A checked it against fold k's held-out values).
    """
    from stage2_crossfit.g13 import bin_source
    tr, _ = side.folds[k]
    codes = np.asarray(states.code)[tr]
    yv = side.y_train.iloc[tr].astype(int).to_numpy()
    src, prov, checks = inner_sources(side, k, oof_fn=oof_fn)
    fits, inner_values = {}, {}
    for fam in FAMILIES:
        sv = state_vals["values"][fam][f"fold_{k}"].to_numpy(dtype=float)
        sc = blocks(state_rows(codes, sv[codes], yv, exact=True))
        inner_values[fam] = src[BD.SOURCES[fam]].to_numpy(dtype=float)
        se = blocks(state_rows(codes, inner_values[fam], yv, exact=False))
        fits[fam] = _fit_family(fam, sc, se)
        if fam == SENSITIVITY["family"]:
            sensitivity = {fam: fit_fold_local(fam, sc)}
    table, chosen = select_partners(bin_source(src), yv)
    return {"fold": k, "inner_provenance": prov, "inner_values": inner_values, "fits": fits,
            "partner_table": table, "partners": chosen, "checks": checks, "sensitivity": sensitivity}


def fit_folds(side: BD.Stage2Side, states: BD.MacroStates, state_vals: Mapping[str, Any], *,
              oof_fn=None) -> List[Dict[str, Any]]:
    """``fit_fold`` for every outer fold."""
    return [fit_fold(side, states, state_vals, k, oof_fn=oof_fn) for k in range(len(side.folds))]


def deployment_fits(side: BD.Stage2Side, states: BD.MacroStates, state_vals: Mapping[str, Any],
                    sources: BD.Sources) -> Dict[str, Dict[str, Any]]:
    """
    Each procedure fitted on all training rows, as deployment would fit it: scoring values from the
    all-rows engineer, selection values from the PR 39 out-of-fold values. For refit stability only.
    """
    codes = np.asarray(states.code)
    yv = side.y_train.astype(int).to_numpy()
    out = {}
    for fam in FAMILIES:
        sv = state_vals["values"][fam]["in_sample"].to_numpy(dtype=float)
        sc = blocks(state_rows(codes, sv[codes], yv, exact=True))
        se = blocks(state_rows(codes, sources.values["oof"][fam], yv, exact=False))
        out[fam] = _fit_family(fam, sc, se)
    return out


def cut_table(fold_fits: Sequence[Mapping[str, Any]], deployed: Mapping[str, Any]) -> pd.DataFrame:
    """The learned cuts per family, procedure and fit, with their fit-row shares, gaps and bin sizes."""
    rows = []
    fits = [(f"fold {ff['fold']}", ff["fits"]) for ff in fold_fits] + [("all rows", deployed)]
    for fam in FAMILIES:
        for proc in ("frozen",) + CANDIDATES[fam]:
            for label, fx in fits:
                f = fx[fam][proc]
                r = {"family": fam, "procedure": proc, "fit": label, "feasible": f["feasible"],
                     "reason": f["reason"], "fit_states": f["fit_states"]}
                for j, c in enumerate(f["per_cut"], start=1):
                    r[f"cut_{j}"] = c.get("cut")
                    r[f"share_{j}"] = c.get("share_below")
                    r[f"gap_{j}"] = c.get("gap")
                    r[f"how_{j}"] = c.get("chosen_by")
                r["bin_rows"] = f["bin_rows"]
                r["bin_states"] = f["bin_states"]
                sel = (f.get("selection") or {}).get("chosen")
                r["selected_tail_share"] = None if not sel else sel["tail_share"]
                r["selected_tail_rate"] = None if not sel else sel["tail_rate"]
                rows.append(r)
    return pd.DataFrame(rows)


def _signature(ff: Mapping[str, Any]) -> Dict[str, Any]:
    return _canonical({"fits": ff["fits"], "partners": ff["partners"], "sensitivity": ff.get("sensitivity"),
                       "inner": {f: np.round(v, 15).tolist() for f, v in ff["inner_values"].items()},
                       "provenance": ff["inner_provenance"]})


def isolation_checks(side: BD.Stage2Side, states: BD.MacroStates, state_vals: Mapping[str, Any],
                     fold_fits: Sequence[Mapping[str, Any]], *, oof_fn=None,
                     folds: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    For each outer fold k: flip every held-out label of k and refit fold k (inner values, cuts, tail
    choice, partners, the sensitivity fit). Nothing may change. As a control, flipping every tenth
    fit-side label of k must move its inner values, so the check can fail. (Fold k's engineer, which
    supplies the scoring values, is Step 4A's, whose held-out flips Cell 33 runs.)

    Conditional on the fixed reference inputs (``ISOLATION_SCOPE``): the frozen cuts, ``frozen_shares``,
    ``window_halfwidths`` and ``_tail_range`` are constants that each fold's held-out rows and labels
    helped form, and a flip leaves them as they are.
    """
    from dataclasses import replace
    out = []
    fold_id = np.asarray(side.fold_id)
    for k in (range(len(side.folds)) if folds is None else folds):
        y = side.y_train.copy()
        held = fold_id == k
        y[held] = 1 - y[held]
        again = fit_fold(replace(side, y_train=y), states, state_vals, k, oof_fn=oof_fn)
        same = _signature(again) == _signature(fold_fits[k])
        out.append(_check(f"fold {k}: given the fixed reference inputs, flipping its {int(held.sum()):,} held-out "
                          "labels leaves its inner values, cuts, tail choice and partner rules unchanged", same,
                          "identical" if same else "changed"))
        y2 = side.y_train.copy()
        fit_pos = np.flatnonzero(~held)[::10]
        y2.iloc[fit_pos] = 1 - y2.iloc[fit_pos]
        ctrl = fit_fold(replace(side, y_train=y2), states, state_vals, k, oof_fn=oof_fn)
        moved = sum(int((ctrl["inner_values"][f] != fold_fits[k]["inner_values"][f]).sum()) for f in FAMILIES)
        out.append(_check(f"fold {k}: control, flipping {len(fit_pos):,} fit-side labels moves its inner values",
                          moved > 0, f"{moved:,} inner values moved"))
    return pd.DataFrame(out)


# ======================================================================
# 4. the comparison on identical held-out rows
# ======================================================================
def _procs(family: str) -> Tuple[str, ...]:
    return ("frozen",) + CANDIDATES[family]


def compare(side: BD.Stage2Side, sources: BD.Sources, fold_fits: Sequence[Mapping[str, Any]],
            deployed: Mapping[str, Any], eb: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Every procedure's held-out bins (each row with its fold's engineer and cuts), their tables
    (``evidence.bin_table``) and T1-T4 (``evidence.triggers``), the comparator's parity with E-B,
    focal summaries, refit migration and NSD x ECI joint tables.
    """
    from stage2_binning import evidence as E
    y = side.y_train.astype(int).to_numpy()
    fold_id = np.asarray(side.fold_id)
    n = len(y)
    labels, deploy_labels, tables, trig, feasible = {}, {}, {}, {}, {}
    checks = []
    for fam in FAMILIES:
        names = E.family_bins(fam)
        oof, ins = sources.values["oof"][fam], sources.values["in_sample"][fam]
        for proc in _procs(fam):
            fits = [ff["fits"][fam][proc] for ff in fold_fits]
            feasible[(fam, proc)] = [bool(f["feasible"]) for f in fits]
            if not all(feasible[(fam, proc)]):
                continue
            lab = np.empty(n, dtype=object)
            for k, f in enumerate(fits):
                va = side.folds[k][1]
                lab[va] = apply_fit(oof[va], f)
            labels[(fam, proc)] = lab
            tables[(fam, proc)] = E.bin_table(lab, y, fold_id, names, len(side.folds))
            trig[(fam, proc)] = E.triggers(fam, tables[(fam, proc)])
            dep = deployed[fam][proc]
            if dep["feasible"]:
                deploy_labels[(fam, proc)] = apply_fit(ins, dep)
        frozen = labels[(fam, "frozen")]
        checks += [
            _check(f"{fam}: the comparator's held-out bins are E-B's, row by row",
                   np.array_equal(frozen, sources.labels["oof"][fam]), fam),
            _check(f"{fam}: the comparator's table is E-B's out-of-fold table",
                   _canonical(tables[(fam, "frozen")]) == eb["families"][fam]["oof"], fam),
            _check(f"{fam}: the frozen cuts through the candidates' apply path give the production bins",
                   np.array_equal(BD.assign_intervals(oof, _frozen_cuts(fam), names), frozen), fam),
            _check(f"{fam}: the all-rows comparator is E-B's in-sample binning",
                   np.array_equal(deploy_labels[(fam, "frozen")], sources.labels["in_sample"][fam]), fam),
        ]
    focal_rows, migration, tables_flat = [], [], []
    for fam in FAMILIES:
        focal = FOCAL[fam]
        base = tables[(fam, "frozen")]["bins"][focal]
        for proc in _procs(fam):
            row = {"family": fam, "bin": focal, "procedure": proc,
                   "feasible_folds": int(sum(feasible[(fam, proc)]))}
            if (fam, proc) not in labels:
                focal_rows.append(row)
                continue
            t = tables[(fam, proc)]["bins"][focal]
            folds = t["folds"]
            lifts = [f["lift"] for f in folds]
            ratio = [None if not b["support"] else f["support"] / b["support"]
                     for f, b in zip(folds, base["folds"])]
            lab, dep = labels[(fam, proc)], deploy_labels.get((fam, proc))
            mig = {}
            if dep is not None:
                into, out_ = (lab == focal) & (dep != focal), (lab != focal) & (dep == focal)
                mig = {"migration_rows": int((into | out_).sum()), "entering": int(into.sum()),
                       "leaving": int(out_.sum()),
                       "migration_by_fold": [int(((into | out_) & (fold_id == k)).sum())
                                             for k in range(len(side.folds))],
                       "any_bin_changed": int((lab != dep).sum())}
                migration.append({"family": fam, "procedure": proc, **mig})
            fired = [k for k in ("T1", "T2", "T3", "T4") if trig[(fam, proc)][k]["fired"]]
            row.update({"support": t["pooled"]["support"], "positives": t["pooled"]["positives"],
                        "rate": t["pooled"]["rate"], "lift": t["pooled"]["lift"],
                        "lift_lower": t["pooled"]["lift_wilson95"][0],
                        "lift_upper": t["pooled"]["lift_wilson95"][1],
                        "fold_lifts": lifts,
                        "min_fold_lift": None if any(v is None for v in lifts) else min(lifts),
                        "max_fold_lift": None if any(v is None for v in lifts) else max(lifts),
                        "fold_supports": [f["support"] for f in folds],
                        "min_support_ratio": None if any(v is None for v in ratio) else min(ratio),
                        "triggers_fired": fired,
                        "overlap_with_frozen": _jaccard(lab == focal, labels[(fam, "frozen")] == focal),
                        "within_frozen": bool(not ((lab == focal) & (labels[(fam, "frozen")] != focal)).any()),
                        "same_rows_as_frozen_in_sample": bool(np.array_equal(
                            lab == focal, sources.labels["in_sample"][fam] == focal)),
                        **{k: v for k, v in mig.items() if k != "migration_by_fold"}})
            focal_rows.append(row)
            for b, e in tables[(fam, proc)]["bins"].items():
                tables_flat.append({"family": fam, "procedure": proc, "bin": b, **e["pooled"],
                                    "fold_lifts": [f["lift"] for f in e["folds"]],
                                    "fold_supports": [f["support"] for f in e["folds"]]})
    joint = {}
    for proc in _procs("ECI"):
        if ("ECI", proc) in labels:
            joint[f"ECI {proc}"] = E.joint_table(labels[("NSD", "frozen")], labels[("ECI", proc)], y)
    for proc in CANDIDATES["NSD"]:
        if ("NSD", proc) in labels:
            joint[f"NSD {proc}"] = E.joint_table(labels[("NSD", proc)], labels[("ECI", "frozen")], y)
    return {"labels": labels, "deploy_labels": deploy_labels, "tables": tables, "triggers": trig,
            "feasible": feasible, "focal": pd.DataFrame(focal_rows), "migration": pd.DataFrame(migration),
            "bins": pd.DataFrame(tables_flat), "joint": joint, "checks": pd.DataFrame(checks)}


def _jaccard(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    union = int((a | b).sum())
    return None if not union else int((a & b).sum()) / union


def dependence(side: BD.Stage2Side, T: pd.DataFrame, comparison: Mapping[str, Any],
               procedures: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """
    The focal bin's pooled lift with each outer fold removed and with each contact period of at least
    ``PERIOD_MIN_ROWS`` rows removed, for every procedure on the same remaining rows; and where the
    focal bin's subscribers come from (largest period share).
    """
    from stage2_binning import evidence as E
    if not T.index.equals(side.X_train.index):
        raise ValueError("the time index is not aligned with the training rows")
    y = side.y_train.astype(int).to_numpy()
    fold_id = np.asarray(side.fold_id)
    period = T["period"].to_numpy()
    counts = pd.Series(period).value_counts()
    removals = [(f"fold {k}", fold_id != k, "fold") for k in range(len(side.folds))]
    removals += [(f"period {p}", period != p, "period")
                 for p in sorted(counts[counts >= PERIOD_MIN_ROWS].index)]
    rows, sources = [], []
    for fam in FAMILIES:
        focal = FOCAL[fam]
        for proc in (_procs(fam) if procedures is None else procedures.get(fam, ())):
            lab = comparison["labels"].get((fam, proc))
            if lab is None:
                continue
            m = lab == focal
            for name, keep, kind in removals:
                n, pos = int(keep.sum()), int(y[keep].sum())
                s, sp = int((m & keep).sum()), int(y[m & keep].sum())
                base = pos / n
                lo, hi = E.wilson(sp, s)
                rows.append({"family": fam, "procedure": proc, "removed": name, "kind": kind,
                             "rows": n, "support": s, "positives": sp,
                             "lift": None if not s else sp / s / base,
                             "lift_lower": None if lo is None else lo / base})
            fp = pd.Series(period[m & (y == 1)]).value_counts()
            sources.append({"family": fam, "procedure": proc, "subscribers": int(fp.sum()),
                            "largest_period": None if fp.empty else str(fp.index[0]),
                            "largest_period_share": None if fp.empty else float(fp.iloc[0] / fp.sum())})
    return {"removals": pd.DataFrame(rows), "subscriber_sources": pd.DataFrame(sources),
            "periods_removed": [r[0] for r in removals if r[2] == "period"]}


# ======================================================================
# 5. eci_warm as a filter on partner rules
# ======================================================================
def partner_pool() -> List[Tuple[str, ...]]:
    """The 25 non-ECI singles and the 263 cross-family pairs, in RF_FEATURES_BINARY order."""
    bins = _partner_bins()
    pairs = [(a, b) for i, a in enumerate(bins) for b in bins[i + 1:] if _family_of(a) != _family_of(b)]
    return [(b,) for b in bins] + pairs


def _name(literals: Sequence[str]) -> str:
    return " & ".join(literals)


def partner_stats(bins: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Every pool partner's rows, subscribers and its eci_warm split, on the rows of ``bins``."""
    pb = _partner_bins()
    B = bins[list(pb)].to_numpy(dtype=np.int64)
    w = bins[FILTER_BIN].to_numpy(dtype=np.int64)
    y = np.asarray(y).astype(np.int64)
    if not (np.isin(B, (0, 1)).all() and np.isin(w, (0, 1)).all()):
        raise ValueError("partner bins must be 0/1")
    S, P = B.T @ B, B.T @ (B * y[:, None])
    W, WP = B.T @ (B * w[:, None]), B.T @ (B * (w * y)[:, None])
    pos = {b: i for i, b in enumerate(pb)}
    total = int(y.sum())
    rows = []
    for lits in partner_pool():
        i, j = pos[lits[0]], pos[lits[-1]]
        n, s, nw, sw = int(S[i, j]), int(P[i, j]), int(W[i, j]), int(WP[i, j])
        rows.append({"partner": _name(lits), "literals": len(lits), "support": n, "positives": s,
                     "precision": s / n if n else None, "recall": s / total if total else None,
                     "warm_support": nw, "warm_positives": sw, "cold_support": n - nw,
                     "cold_positives": s - sw})
    df = pd.DataFrame(rows)
    df["wilson_lower"] = _wilson_lower(df["positives"].to_numpy(), df["support"].to_numpy())
    return df


def select_partners(bins: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    The registered fit-side selection: eligibility (GLASS precision and recall gates, the scaled support
    floors for the partner and both warm variants), Wilson-lower-bound ranking, then greedy choice of up
    to ``PARTNER_K`` partners with fit-row Jaccard overlap <= ``max_jaccard_overlap``.
    """
    st = partner_stats(bins, y)
    st["eligible"] = ((st["precision"].fillna(0) >= GLASS["min_precision_subscribe"])
                      & (st["recall"] >= GLASS["min_recall_subscribe"])
                      & (st["recall"] <= GLASS["max_recall_subscribe"])
                      & (st["support"] >= PARTNER_FIT_SUPPORT_MIN)
                      & (st["warm_support"] >= VARIANT_FIT_SUPPORT_MIN)
                      & (st["cold_support"] >= VARIANT_FIT_SUPPORT_MIN))
    order = st[st["eligible"]].sort_values(["wilson_lower", "support", "literals", "partner"],
                                           ascending=[False, False, True, True], kind="mergesort")
    st["rank"] = pd.Series(np.arange(1, len(order) + 1), index=order.index)
    st["slot"] = pd.Series(dtype="float")
    chosen, masks = [], []
    for rank, (idx, r) in enumerate(order.iterrows(), start=1):
        lits = r["partner"].split(" & ")
        m = np.all(bins[lits].to_numpy() == 1, axis=1)
        overlaps = [_jaccard(m, other) or 0.0 for other in masks]
        if all(o <= GLASS["max_jaccard_overlap"] for o in overlaps):
            masks.append(m)
            st.loc[idx, "slot"] = len(masks)
            chosen.append({"slot": len(masks), "partner": r["partner"], "literals": lits,
                           "rank": rank, "fit_support": int(r["support"]),
                           "fit_positives": int(r["positives"]), "fit_precision": float(r["precision"]),
                           "fit_wilson_lower": float(r["wilson_lower"]),
                           "fit_warm_support": int(r["warm_support"]),
                           "fit_warm_positives": int(r["warm_positives"]),
                           "max_overlap_with_earlier": max(overlaps) if overlaps else None})
            if len(chosen) == PARTNER_K:
                break
    return st, chosen


def _prec(s: int, n: int) -> Optional[float]:
    return s / n if n else None


def warm_filter(side: BD.Stage2Side, heldout: Mapping[str, Any],
                fold_fits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Each fold's partners (chosen on its fit side) on its held-out rows: partner alone, warm required,
    warm excluded; the partition and count reconciliation; what each filter removes; pooled per slot
    and per partner; the union screen and the registered per-slot effects.
    """
    bins = heldout["bins"]
    y = side.y_train.astype(int).to_numpy()
    per, checks = [], []
    for ff in fold_fits:
        k = ff["fold"]
        va = side.folds[k][1]
        hb, yv = bins.iloc[va], y[va]
        warm = hb[FILTER_BIN].to_numpy() == 1
        for p in ff["partners"]:
            m = np.all(hb[list(p["literals"])].to_numpy() == 1, axis=1)
            req, exc = m & warm, m & ~warm
            c, ok = partition_counts(m, req, exc, yv)
            checks.append(_check(f"fold {k} slot {p['slot']} ({p['partner']}): required and excluded partition "
                                 "the partner's rows and their counts add up", ok, c))
            row = {"fold": k, "slot": p["slot"], "partner": p["partner"]}
            for v, (n, s) in c.items():
                row.update({f"{v}_rows": n, f"{v}_subscribers": s, f"{v}_non_subscribers": n - s,
                            f"{v}_precision": _prec(s, n)})
            row.update({"required_removes_subscribers": c["excluded"][1],
                        "required_removes_non_subscribers": c["excluded"][0] - c["excluded"][1],
                        "excluded_removes_subscribers": c["required"][1],
                        "excluded_removes_non_subscribers": c["required"][0] - c["required"][1]})
            per.append(row)
    per = pd.DataFrame(per)
    by_slot = _pool(per, "slot") if len(per) else pd.DataFrame()
    by_partner = _pool(per, "partner") if len(per) else pd.DataFrame()
    return {"population": REGISTRATION["population"], "support_scope": FILTER_SCOPE, "per_fold": per,
            "by_slot": by_slot, "by_partner": by_partner,
            "checks": pd.DataFrame(checks, columns=["check", "passed", "detail"])}


FILTER_SCOPE = (
    "The rows and subscribers each variant keeps or removes are those of a standalone conjunction on held-out "
    "rows: requiring eci_warm drops the partner's other rows from that one rule. That is not the router-level "
    "coverage effect. In a router those rows can still be routed by other Pass 2 rules of the selected union, "
    "Pass 1 changes the population the rules see, and a precision gain can change which rules the union-aware "
    "selection admits, so router coverage can move by less, by more or in either direction. PR 44 measures it.")


def partition_counts(m: np.ndarray, req: np.ndarray, exc: np.ndarray, y: np.ndarray
                     ) -> Tuple[Dict[str, Tuple[int, int]], bool]:
    """Rows and subscribers of a partner and its two variants, and whether the variants partition it."""
    c = {v: (int(mask.sum()), int(y[mask].sum())) for v, mask in
         (("alone", m), ("required", req), ("excluded", exc))}
    ok = (not (req & exc).any() and np.array_equal(req | exc, m)
          and c["required"][0] + c["excluded"][0] == c["alone"][0]
          and c["required"][1] + c["excluded"][1] == c["alone"][1])
    return c, bool(ok)


def _pool(per: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for val, g in per.groupby(key, sort=True):
        r = {key: val, "folds": int(len(g)), "fold_list": sorted(int(f) for f in g["fold"])}
        if key == "slot":
            r["partners"] = sorted(set(g["partner"]))
        for v in ("alone", "required", "excluded"):
            n, s = int(g[f"{v}_rows"].sum()), int(g[f"{v}_subscribers"].sum())
            lo, hi = _wilson(s, n)
            r.update({f"{v}_rows": n, f"{v}_subscribers": s, f"{v}_non_subscribers": n - s,
                      f"{v}_precision": _prec(s, n), f"{v}_wilson95": [lo, hi]})
        for c in ("required_removes_subscribers", "required_removes_non_subscribers",
                  "excluded_removes_subscribers", "excluded_removes_non_subscribers"):
            r[c] = int(g[c].sum())
        for v in ("required", "excluded"):
            pa, pv = r["alone_precision"], r[f"{v}_precision"]
            r[f"{v}_ratio"] = None if not pa or pv is None else pv / pa
            better = [(a is not None and b is not None and b > a)
                      for a, b in zip(g["alone_precision"], g[f"{v}_precision"])]
            r[f"{v}_folds_above_partner"] = int(sum(better))
            r[f"{v}_screen"] = bool(r[f"{v}_rows"] >= UNION["min_pass2_support"] and pv is not None
                                    and pv >= UNION["min_pass2_precision"])
            r[f"{v}_rule_floors"] = bool(r[f"{v}_rows"] >= GLASS["min_support_pass2"] and pv is not None
                                         and pv >= GLASS["min_precision_subscribe"])
            if key == "slot":
                r[f"{v}_effect"] = bool(r["folds"] >= FOLD_MAJORITY and r[f"{v}_ratio"] is not None
                                        and r[f"{v}_ratio"] >= FILTER_GAIN
                                        and r[f"{v}_folds_above_partner"] >= FOLD_MAJORITY
                                        and r[f"{v}_screen"])
        r["alone_screen"] = bool(r["alone_rows"] >= UNION["min_pass2_support"] and r["alone_precision"] is not None
                                 and r["alone_precision"] >= UNION["min_pass2_precision"])
        rows.append(r)
    return pd.DataFrame(rows)


def _wilson(s: int, n: int) -> List[Optional[float]]:
    from stage2_binning import evidence as E
    return E.wilson(s, n)


# ======================================================================
# 6. the decision, by the registered criteria
# ======================================================================
def _ge(a, b) -> bool:
    return a is not None and b is not None and not (isinstance(a, float) and np.isnan(a)) and a >= b


def _gt(a, b) -> bool:
    return a is not None and b is not None and not (isinstance(a, float) and np.isnan(a)) and a > b


REVISION_NEEDS = {
    "contract": ("shared.stage2.binning would carry a fitted NSD/ECI cut procedure in place of constant "
                 "thresholds: a new binning version and identity, the 29 bin names and order unchanged, "
                 "and the feature contract's bin definitions stated as fitted values"),
    "g13": ("the cuts become fitted state: stage2_crossfit.g13's audit of label-fitted features must cover "
            "the cut fit and require it inside every Stage 2 cross-fit fold. Its reference inputs need a stated "
            "scope: as registered, gap_placed's targets and windows are full-training constants "
            "(frozen_shares(), window_halfwidths()) that each fold's held-out rows and labels helped form; the "
            "fold-local sensitivity derives them from each fit side and does not repair fold 0. Beyond those "
            "inputs, share_matched and gap_placed use no label but the engineer's own fit; tail_selected "
            "chooses with labels"),
    "regeneration": ("Stage 2 binary frames, GLASS rules, RF Router forests and every downstream Stage 3-4 "
                     "artifact would be regenerated, beside the frozen PR 38-39 ones"),
    "follow_up": ("PR 44 Step 44B fits the routers with the candidate under the shared obligations; PR 49 "
                  "checks it chronologically. PR 42 changes no frozen bin"),
}


def decide(comparison: Mapping[str, Any], dep: Mapping[str, Any], filt: Mapping[str, Any]) -> Dict[str, Any]:
    """A0-A4 per candidate, the family verdicts, and the separate eci_warm filter verdict."""
    focal = comparison["focal"].set_index(["family", "procedure"])
    rem = dep["removals"]
    crit = []
    for fam in FAMILIES:
        fz = focal.loc[(fam, "frozen")]
        for proc in CANDIDATES[fam]:
            c = focal.loc[(fam, proc)]
            r = {"family": fam, "procedure": proc, "bin": FOCAL[fam]}
            r["A0_feasible"] = bool(c["feasible_folds"] == N_FOLDS)
            if not r["A0_feasible"]:
                r.update({"passes": False, "rank_value": None,
                          "failed": ["A0"], "detail": f"feasible in {int(c['feasible_folds'])} of {N_FOLDS} folds"})
                crit.append(r)
                continue
            r["A1_no_trigger"] = not c["triggers_fired"]
            r["A2_coverage"] = _ge(c["min_support_ratio"], SUPPORT_FRACTION)
            if fam == "NSD":
                r["A3a_min_fold_lift_above_frozen"] = _gt(c["min_fold_lift"], fz["min_fold_lift"])
                r["A3b_lift_at_least_frozen_lower"] = _ge(c["lift"], fz["lift_lower"])
                r["A3c_migration_at_most_frozen"] = _ge(fz["migration_rows"], c["migration_rows"])
                a3 = ("A3a_min_fold_lift_above_frozen", "A3b_lift_at_least_frozen_lower",
                      "A3c_migration_at_most_frozen")
                rank = c["min_fold_lift"]
            else:
                r["A3a_lift_gain"] = _ge(c["lift"], HOT_LIFT_GAIN * fz["lift"])
                r["A3b_min_fold_lift_at_least_frozen"] = _ge(c["min_fold_lift"], fz["min_fold_lift"])
                a3 = ("A3a_lift_gain", "A3b_min_fold_lift_at_least_frozen")
                rank = c["lift_lower"]
            mine = rem[(rem["family"] == fam) & (rem["procedure"] == proc)].set_index("removed")
            ref = rem[(rem["family"] == fam) & (rem["procedure"] == "frozen")].set_index("removed")
            bound = "lift_lower" if fam == "NSD" else "lift"
            failing = [name for name in ref.index if not _ge(mine.loc[name, "lift"], ref.loc[name, bound])]
            r["A4_robust"] = not failing
            r["A4_failing_removals"] = failing
            names = ["A1_no_trigger", "A2_coverage", *a3, "A4_robust"]
            r["failed"] = [n.split("_")[0] for n in names if not r[n]]
            r["passes"] = not r["failed"]
            r["rank_value"] = rank
            r["detail"] = ""
            crit.append(r)
    crit = pd.DataFrame(crit)
    verdicts, families = [], {}
    for fam in FAMILIES:
        sub = crit[crit["family"] == fam]
        ok = sub[sub["passes"]].copy()
        ok["simplicity"] = ok["procedure"].map(SIMPLICITY)
        ok = ok.sort_values(["rank_value", "simplicity"], ascending=[False, True], kind="mergesort")
        winner = None if ok.empty else ok.iloc[0]["procedure"]
        for _, r in sub.iterrows():
            if r["procedure"] == winner:
                v, why = "advance to PR 44", "meets A0-A4" + (" and ranks first" if len(ok) > 1 else "")
            elif r["passes"]:
                v, why = "close out", f"meets A0-A4 but is outranked by {winner}"
            else:
                v, why = "close out", "fails " + ", ".join(r["failed"]) + (f" ({r['detail']})" if r["detail"] else "")
            verdicts.append({"family": fam, "procedure": r["procedure"], "verdict": v, "reason": why})
        families[fam] = {"frozen_bins": "kept unchanged in PR 42", "advance": winner,
                         "revision": None if winner is None else dict(REVISION_NEEDS)}
    bs = filt["by_slot"]
    n_req = int(bs["required_effect"].sum()) if len(bs) else 0
    n_exc = int(bs["excluded_effect"].sum()) if len(bs) else 0
    if n_req >= SLOT_MAJORITY:
        fv = {"verdict": "advance to PR 44 as a filter", "direction": "warm_required"}
    elif n_exc >= SLOT_MAJORITY:
        fv = {"verdict": "advance to PR 44 as a filter", "direction": "warm_excluded"}
    else:
        fv = {"verdict": "close out as a filter", "direction": None}
    fv.update({"slots": int(len(bs)), "slots_required_effect": n_req, "slots_excluded_effect": n_exc,
               "needed": SLOT_MAJORITY, "population": "diagnostic: all held-out training rows of each fold"})
    return {"criteria": crit, "verdicts": pd.DataFrame(verdicts), "families": families, "eci_warm_filter": fv,
            "revision": ({"warranted": False, "text": "no candidate advances: the frozen bins stay unchanged "
                          "and no versioned revision is proposed"}
                         if all(v["advance"] is None for v in families.values())
                         else {"warranted": "pending PR 44", "needs": dict(REVISION_NEEDS)})}


#: What the review document states, filled in from the accepted sandbox run; ``stated_checks``
#: requires the run that writes the record to agree, so the document cannot drift from it.
STATED: Dict[str, Any] = {
    "verdicts": {("NSD", "share_matched"): "close out", ("NSD", "gap_placed"): "advance to PR 44",
                 ("ECI", "share_matched"): "close out", ("ECI", "gap_placed"): "close out",
                 ("ECI", "tail_selected"): "close out"},
    "advance": {"NSD": "gap_placed", "ECI": None},
    "eci_warm_filter": ("advance to PR 44 as a filter", "warm_required"),
    "focal": {
        ("NSD", "frozen", "lift"): 2.486, ("NSD", "frozen", "min_fold_lift"): 0.617,
        ("NSD", "frozen", "migration_rows"): 653, ("NSD", "frozen", "support"): 1421,
        ("NSD", "share_matched", "lift"): 3.131, ("NSD", "share_matched", "min_fold_lift"): 2.808,
        ("NSD", "share_matched", "migration_rows"): 27, ("NSD", "share_matched", "support"): 1321,
        ("NSD", "gap_placed", "lift"): 3.126, ("NSD", "gap_placed", "lift_lower"): 2.902,
        ("NSD", "gap_placed", "lift_upper"): 3.358, ("NSD", "gap_placed", "min_fold_lift"): 2.821,
        ("NSD", "gap_placed", "max_fold_lift"): 3.356, ("NSD", "gap_placed", "migration_rows"): 0,
        ("NSD", "gap_placed", "support"): 1326, ("NSD", "gap_placed", "min_support_ratio"): 0.746,
        ("NSD", "gap_placed", "same_rows_as_frozen_in_sample"): True,
        ("ECI", "frozen", "lift"): 2.501, ("ECI", "frozen", "min_fold_lift"): 2.388,
        ("ECI", "frozen", "support"): 4074, ("ECI", "frozen", "migration_rows"): 91,
        ("ECI", "share_matched", "lift"): 2.457, ("ECI", "gap_placed", "lift"): 2.807,
        ("ECI", "gap_placed", "support"): 4614,
        ("ECI", "tail_selected", "lift"): 3.481, ("ECI", "tail_selected", "lift_lower"): 3.314,
        ("ECI", "tail_selected", "min_fold_lift"): 3.403, ("ECI", "tail_selected", "max_fold_lift"): 3.566,
        ("ECI", "tail_selected", "support"): 2532, ("ECI", "tail_selected", "min_support_ratio"): 0.564,
        ("ECI", "tail_selected", "migration_rows"): 735, ("ECI", "tail_selected", "within_frozen"): True,
    },
    "criteria": {
        ("NSD", "share_matched", "failed"): [], ("NSD", "gap_placed", "failed"): [],
        ("ECI", "share_matched", "failed"): ["A3a", "A4"], ("ECI", "gap_placed", "failed"): ["A1"],
        ("ECI", "tail_selected", "failed"): ["A4"],
        ("ECI", "tail_selected", "A4_failing_removals"): ["period 2009-05"],
    },
    "triggers": {("ECI", "gap_placed"): ("T2", ["eci_warm"])},
    "joint": {("NSD gap_placed", "rows", 1): [0, 206, 5171, 1456],
              ("NSD gap_placed", "rows", 2): [0, 0, 0, 1326]},
    "removals": {("NSD", "frozen", "period 2009-05"): 3.045, ("NSD", "gap_placed", "period 2009-05"): 3.018,
                 ("NSD", "frozen", "fold 0"): 3.153, ("NSD", "gap_placed", "fold 0"): 3.153,
                 ("ECI", "frozen", "period 2009-05"): 3.395, ("ECI", "tail_selected", "period 2009-05"): 3.361},
    "periods_removed": 23,
    "slots": {
        (1, "required_effect"): False, (2, "required_effect"): False, (3, "required_effect"): True,
        (4, "required_effect"): True, (5, "required_effect"): True,
        (1, "required_ratio"): 1.07, (2, "required_ratio"): 1.065, (3, "required_ratio"): 1.194,
        (4, "required_ratio"): 1.135, (5, "required_ratio"): 1.271,
        (3, "alone_rows"): 1643, (3, "required_rows"): 467, (3, "required_removes_subscribers"): 557,
        (3, "required_removes_non_subscribers"): 619,
    },
}


def stated_checks(decision: Mapping[str, Any], comparison: Mapping[str, Any],
                  filt: Mapping[str, Any], dep: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Each verdict and figure the review document states, checked in this run."""
    out = []
    v = decision["verdicts"].set_index(["family", "procedure"])["verdict"]
    for (fam, proc), want in STATED.get("verdicts", {}).items():
        got = v.get((fam, proc))
        out.append(_check(f"{fam} {proc}: {want}", got == want, got))
    for fam, want in STATED.get("advance", {}).items():
        got = decision["families"][fam]["advance"]
        out.append(_check(f"{fam}: advancing candidate {want}", got == want, got))
    if "eci_warm_filter" in STATED:
        got = decision["eci_warm_filter"]["verdict"], decision["eci_warm_filter"]["direction"]
        out.append(_check(f"eci_warm filter: {STATED['eci_warm_filter']}", got == STATED["eci_warm_filter"], got))
    focal = comparison["focal"].set_index(["family", "procedure"])
    crit = decision["criteria"].set_index(["family", "procedure"])
    slots = filt["by_slot"].set_index("slot") if len(filt["by_slot"]) else pd.DataFrame()
    for where, table in (("focal", focal), ("criteria", crit), ("slots", slots)):
        for key, want in STATED.get(where, {}).items():
            *row, field = key
            got = table.loc[tuple(row) if len(row) > 1 else row[0], field]
            same = (round(float(got), 3) == want) if isinstance(want, float) else (_canonical(got) == want)
            out.append(_check(f"{where} {' '.join(map(str, row))} {field} = {want}", same, got))
    for (fam, proc), (trigger, bins) in STATED.get("triggers", {}).items():
        got = comparison["triggers"][(fam, proc)][trigger]["bins"]
        out.append(_check(f"{fam} {proc}: {trigger} fires for {bins}", got == bins, got))
    for (name, part, i), want in STATED.get("joint", {}).items():
        got = comparison["joint"][name][part][i]
        out.append(_check(f"joint {name} {part}[{i}] = {want}", got == want, got))
    if dep is not None:
        rem = dep["removals"].set_index(["family", "procedure", "removed"])
        for key, want in STATED.get("removals", {}).items():
            got = rem.loc[key, "lift"]
            out.append(_check(f"{' '.join(key)}: lift {want}", round(float(got), 3) == want, got))
        n = len(dep["periods_removed"])
        out.append(_check(f"{STATED['periods_removed']} contact periods are removed one at a time",
                          n == STATED["periods_removed"], n))
    if not out:
        out.append(_check("the review document's statements are registered in STATED", False, "empty"))
    return pd.DataFrame(out)


# ======================================================================
# 7. reference inputs, and the post-registration sensitivity
# ======================================================================
def reference_inputs() -> pd.DataFrame:
    """
    The fixed reference inputs every registered procedure uses, and where they come from: all three are
    functions of ``FROZEN_SUPPORTS``, the frozen bins' in-sample supports on all 32,950 training rows
    (PR 39 E-B: the all-rows engineer, fitted with every training label, binned with the frozen cuts).
    Each fold's fit therefore depends on its own held-out rows and labels through these constants.
    """
    src = ("FROZEN_SUPPORTS: the frozen bins' in-sample row counts on all 32,950 training rows (PR 39 record "
           "E-B, families.<F>.in_sample): the all-rows engineer fitted with every training label, each fold's "
           "held-out rows and labels included, binned with the frozen cuts")
    rows = []
    for fam in FAMILIES:
        rows.append({"input": "frozen_shares()", "family": fam, "values": list(frozen_shares(fam)),
                     "derived_from": src, "used_by": ("share_matched (targets), gap_placed (targets), "
                                                     "tail_selected (cuts 1-2)" if fam == "ECI" else
                                                     "share_matched (targets), gap_placed (targets)")})
        rows.append({"input": "window_halfwidths()", "family": fam, "values": list(window_halfwidths(fam)),
                     "derived_from": f"{WINDOW_FRACTION} x the smaller of the two adjacent shares in {src}",
                     "used_by": "gap_placed (windows)"})
    rows.append({"input": "_tail_range()", "family": "ECI", "values": _tail_range(),
                 "derived_from": f"{TAIL_SHARE_MIN_FACTOR}-{TAIL_SHARE_MAX_FACTOR} x the eci_hot share in {src}",
                 "used_by": "tail_selected (candidate tails)"})
    return pd.DataFrame(rows)


REFERENCE_SCOPE = (
    "Every registered procedure reads fixed reference inputs: share_matched its targets (frozen_shares()), "
    "gap_placed its targets and windows (frozen_shares(), window_halfwidths()), tail_selected its first two "
    "targets and its tail range (frozen_shares(), _tail_range()). All are functions of FROZEN_SUPPORTS, the "
    "frozen bins' in-sample supports on all 32,950 training rows, computed by the all-rows engineer with every "
    "training label: each fold's held-out rows and labels contribute to them. The frozen cuts behind those "
    "supports were read from in-sample deciles on all 41,188 rows (PR 39 limitation; Step 4A). No registered "
    "procedure is therefore fully nested. Like the frozen comparator, they use these historical inputs.")

ISOLATION_SCOPE = (
    "The held-out label flips (Cell 42) are conditional on the fixed historical inputs: the frozen cut points, "
    "frozen_shares(), window_halfwidths() and _tail_range(). Flipping fold k's held-out labels leaves those "
    "constants as they are, so the flips show that no fit-side computation reads fold k's held-out labels; they "
    "cannot show, and do not claim, that the constants are independent of fold k's rows and labels. They are "
    "not. Fold k's engineer (the scoring values) is flipped in Cell 33.")

#: Post-registration sensitivity check, added at review after the registered 1.0.0 results. It is not part
#: of REGISTRATION (its digest is unchanged) and never enters ``decide`` or the registered ranking.
SENSITIVITY = {
    "id": "pr42-step4b-sensitivity-fold-local-reference",
    "status": ("post-registration: added at review after the registered 1.0.0 results were seen; outside the "
               "registration and the registered ranking"),
    "family": "NSD",
    "procedure": "gap_placed",
    "name": "gap_placed_fold_local",
    "question": ("Do gap_placed's fold-0 repair and zero refit migration for nsd_elevated persist when each "
                 "outer fold derives its reference shares and window widths from its own fit rows?"),
    "rule": ("within outer fold k, the frozen comparator cuts are applied to fold k's engineer values of its fit "
             "rows. The fit-row share below each cut is the fold-local target q_j(k); the fit-row share of each "
             f"frozen bin gives h_j(k) = {WINDOW_FRACTION} x the smaller of the two bins cut j separates. "
             "gap_placed's placement then runs unchanged with q(k) and h(k). The all-rows reference fit uses all "
             "training rows the same way"),
    "unchanged": ["the placement algorithm (fit_gap_placed)", "the outer folds and the scoring values",
                  "the frozen comparator", "the held-out evaluation rows and their out-of-fold values",
                  "the registered result, ranking and verdicts"],
    "comparator_provenance": ("the frozen cuts 0.0524 / 0.192 / 0.23 keep their documented historical "
                              "provenance: read from in-sample decile tables on all 41,188 rows with an encoder "
                              "fitted on them (Step 4A). Only the shares and widths become fold-local"),
}


def fold_local_reference(family: str, blk: pd.DataFrame) -> Dict[str, Any]:
    """The frozen cuts on a fit side's own values: its targets, bin shares and window half-widths."""
    fr = fit_frozen(family, blk)
    total = fr["fit_rows"]
    bins = [r / total for r in fr["bin_rows"]]
    shares = tuple(round(float(v), 6) for v in np.cumsum(bins)[:3])
    halfwidths = tuple(round(float(WINDOW_FRACTION * min(bins[j], bins[j + 1])), 6) for j in range(3))
    return {"shares": shares, "halfwidths": halfwidths, "bin_shares": [round(b, 6) for b in bins],
            "bin_rows": fr["bin_rows"], "bin_states": fr["bin_states"]}


def fit_fold_local(family: str, blk: pd.DataFrame) -> Dict[str, Any]:
    """The sensitivity fit: ``gap_placed``'s placement with fold-local targets and windows."""
    ref = fold_local_reference(family, blk)
    fit = fit_gap_placed(family, blk, shares=ref["shares"], halfwidths=ref["halfwidths"],
                         procedure=SENSITIVITY["name"])
    fit["reference"] = ref
    return fit


def sensitivity_all_rows(side: BD.Stage2Side, states: BD.MacroStates,
                         state_vals: Mapping[str, Any]) -> Dict[str, Any]:
    """The sensitivity rule fitted on all training rows (its refit-migration reference)."""
    fam = SENSITIVITY["family"]
    codes = np.asarray(states.code)
    sv = state_vals["values"][fam]["in_sample"].to_numpy(dtype=float)
    sc = blocks(state_rows(codes, sv[codes], side.y_train.astype(int).to_numpy(), exact=True))
    return fit_fold_local(fam, sc)


def sensitivity(side: BD.Stage2Side, sources: BD.Sources, T: pd.DataFrame,
                fold_fits: Sequence[Mapping[str, Any]], all_rows_fit: Mapping[str, Any],
                comparison: Mapping[str, Any], deployed: Mapping[str, Any]) -> Dict[str, Any]:
    """
    The fold-local sensitivity of NSD ``gap_placed`` beside the registered result and the frozen comparator,
    on the same held-out rows: fold-local targets and windows, learned cuts, focal support, pooled and
    worst-fold lift, refit migration, per-fold membership against the registered fit, the fold and period
    removals, and the registered criteria applied descriptively (never a verdict).
    """
    from stage2_binning import evidence as E
    fam, focal, name = SENSITIVITY["family"], FOCAL[SENSITIVITY["family"]], SENSITIVITY["name"]
    names = E.family_bins(fam)
    y = side.y_train.astype(int).to_numpy()
    fold_id = np.asarray(side.fold_id)
    oof, ins = sources.values["oof"][fam], sources.values["in_sample"][fam]
    fits = [ff["sensitivity"][fam] for ff in fold_fits]
    feasible = [bool(f["feasible"]) for f in fits]
    reg = comparison["labels"][(fam, "gap_placed")]
    frozen = comparison["labels"][(fam, "frozen")]
    per_fold = []
    for k, (f, ff) in enumerate(zip(fits, fold_fits)):
        rf = ff["fits"][fam]["gap_placed"]
        per_fold.append({"fold": k, "feasible": f["feasible"],
                         "fold_local_targets": list(f["reference"]["shares"]),
                         "registered_targets": list(frozen_shares(fam)),
                         "fold_local_halfwidths": list(f["reference"]["halfwidths"]),
                         "registered_halfwidths": list(window_halfwidths(fam)),
                         "fit_side_frozen_bin_rows": f["reference"]["bin_rows"],
                         "fit_side_frozen_bin_states": f["reference"]["bin_states"],
                         "cuts": f["cuts"], "registered_cuts": rf["cuts"],
                         "how": [c["chosen_by"] for c in f["per_cut"]],
                         "fit_bin_rows": f["bin_rows"], "fit_bin_states": f["bin_states"]})
    out: Dict[str, Any] = {"definition": SENSITIVITY, "feasible": feasible, "per_fold": per_fold}
    if not all(feasible):
        out.update({"summary": None, "persists": None})
        return out
    lab = np.empty(len(y), dtype=object)
    for k, f in enumerate(fits):
        va = side.folds[k][1]
        lab[va] = apply_fit(oof[va], f)
    table = E.bin_table(lab, y, fold_id, names, len(side.folds))
    trig = E.triggers(fam, table)
    dep_lab = apply_fit(ins, all_rows_fit)
    reg_dep = comparison["deploy_labels"][(fam, "gap_placed")]
    rows = []
    for proc, l, d, t in (("frozen", frozen, comparison["deploy_labels"][(fam, "frozen")],
                           comparison["tables"][(fam, "frozen")]),
                          ("gap_placed", reg, reg_dep, comparison["tables"][(fam, "gap_placed")]),
                          (name, lab, dep_lab, table)):
        e = t["bins"][focal]
        lifts = [x["lift"] for x in e["folds"]]
        base = comparison["tables"][(fam, "frozen")]["bins"][focal]["folds"]
        mig = (l == focal) != (d == focal)
        rows.append({"procedure": proc, "status": "post-registration sensitivity" if proc == name else
                     ("registered comparator" if proc == "frozen" else "registered candidate (advancing)"),
                     "support": e["pooled"]["support"], "positives": e["pooled"]["positives"],
                     "lift": e["pooled"]["lift"], "lift_lower": e["pooled"]["lift_wilson95"][0],
                     "lift_upper": e["pooled"]["lift_wilson95"][1], "fold_lifts": lifts,
                     "fold_supports": [x["support"] for x in e["folds"]],
                     "min_fold_lift": None if None in lifts else min(lifts),
                     "fold0_lift": lifts[0],
                     "min_support_ratio": min(x["support"] / b["support"] for x, b in zip(e["folds"], base)),
                     "migration_rows": int(mig.sum()),
                     "migration_by_fold": [int((mig & (fold_id == k)).sum()) for k in range(len(side.folds))],
                     "triggers_fired": [k for k in ("T1", "T2", "T3", "T4") if
                                        (trig if proc == name else comparison["triggers"][(fam, proc)])[k]["fired"]],
                     "rows_differing_from_registered": int(((l == focal) != (reg == focal)).sum()),
                     "differing_by_fold": [int((((l == focal) != (reg == focal)) & (fold_id == k)).sum())
                                           for k in range(len(side.folds))],
                     "same_rows_as_frozen_in_sample": bool(np.array_equal(
                         l == focal, sources.labels["in_sample"][fam] == focal))})
    summary = pd.DataFrame(rows)
    dep = dependence(side, T, {"labels": {(fam, "frozen"): frozen, (fam, "gap_placed"): reg, (fam, name): lab}},
                     procedures={fam: ("frozen", "gap_placed", name)})
    rem = dep["removals"].set_index(["procedure", "removed"])
    s, fz = summary.set_index("procedure").loc[name], summary.set_index("procedure").loc["frozen"]
    removals = sorted(set(dep["removals"]["removed"]))
    failing = [r for r in removals if not _ge(rem.loc[(name, r), "lift"], rem.loc[("frozen", r), "lift_lower"])]
    criteria = {"note": "the registered criteria applied descriptively; not a verdict and not in the ranking",
                "A0_feasible": bool(all(feasible)), "A1_no_trigger": not s["triggers_fired"],
                "A2_coverage": bool(_ge(s["min_support_ratio"], SUPPORT_FRACTION)),
                "A3a_min_fold_lift_above_frozen": bool(_gt(s["min_fold_lift"], fz["min_fold_lift"])),
                "A3b_lift_at_least_frozen_lower": bool(_ge(s["lift"], fz["lift_lower"])),
                "A3c_migration_at_most_frozen": bool(_ge(fz["migration_rows"], s["migration_rows"])),
                "A4_robust": not failing, "A4_failing_removals": failing}
    r_reg = summary.set_index("procedure").loc["gap_placed"]
    persists = {
        "fold0_repair": bool(s["differing_by_fold"][0] == 0),
        "fold0_repair_definition": ("fold 0's held-out nsd_elevated rows are the registered gap_placed's, row for "
                                    "row (the registered fit repairs fold 0)"),
        "zero_migration": bool(s["migration_rows"] == 0),
        "all_rows_fit_equals_registered": bool(np.allclose(all_rows_fit["cuts"], deployed[fam]["gap_placed"]["cuts"],
                                                           rtol=0, atol=1e-12)),
        "material_change": bool(s["differing_by_fold"][0] > 0 or s["migration_rows"] > 0
                                or not _ge(s["min_fold_lift"], fz["lift_lower"])),
        "material_change_definition": ("post-registration, for the handoff only: fold 0's membership differs from "
                                       "the registered fit's, or some held-out row changes membership on refit, or "
                                       "the worst fold's lift falls below the frozen bin's pooled Wilson lower bound"),
        "registered_fold0_lift": r_reg["fold0_lift"], "sensitivity_fold0_lift": s["fold0_lift"],
    }
    out.update({"summary": summary, "table": table, "triggers": trig, "labels": lab, "deploy_labels": dep_lab,
                "all_rows_fit": dict(all_rows_fit), "removals": dep["removals"], "criteria": criteria,
                "persists": persists})
    return out


#: What the review document states about the sensitivity, checked in the run (separately from STATED).
STATED_SENSITIVITY: Dict[str, Any] = {
    "summary": {
        ("gap_placed", "lift"): 3.126, ("gap_placed", "fold0_lift"): 3.024, ("gap_placed", "migration_rows"): 0,
        ("gap_placed", "support"): 1326,
        ("gap_placed_fold_local", "support"): 1323, ("gap_placed_fold_local", "positives"): 392,
        ("gap_placed_fold_local", "lift"): 2.630, ("gap_placed_fold_local", "lift_lower"): 2.417,
        ("gap_placed_fold_local", "lift_upper"): 2.854, ("gap_placed_fold_local", "min_fold_lift"): 0.644,
        ("gap_placed_fold_local", "fold0_lift"): 0.644, ("gap_placed_fold_local", "min_support_ratio"): 0.738,
        ("gap_placed_fold_local", "migration_rows"): 555,
        ("gap_placed_fold_local", "migration_by_fold"): [555, 0, 0, 0, 0],
        ("gap_placed_fold_local", "differing_by_fold"): [555, 0, 0, 0, 0],
        ("gap_placed_fold_local", "fold_supports"): [276, 275, 255, 258, 259],
        ("gap_placed_fold_local", "triggers_fired"): [],
        ("gap_placed_fold_local", "same_rows_as_frozen_in_sample"): False,
    },
    "persists": {"fold0_repair": False, "zero_migration": False, "all_rows_fit_equals_registered": True,
                 "material_change": True},
    "per_fold": {
        (0, "fold_local_targets"): [0.66931, 0.813809, 0.879439],
        (0, "fold_local_halfwidths"): [0.036125, 0.016407, 0.016407],
        (0, "cuts"): [0.1243, 0.1922, 0.2114], (0, "registered_cuts"): [0.1243, 0.2114, 0.2655],
        (0, "fit_side_frozen_bin_states"): [55, 51, 7, 231],
        (1, "fold_local_targets"): [0.582511, 0.880273, 0.920144],
        (3, "fold_local_targets"): [0.671624, 0.878225, 0.918741],
    },
    "criteria": {"A0_feasible": True, "A1_no_trigger": True, "A2_coverage": True,
                 "A3a_min_fold_lift_above_frozen": True, "A3b_lift_at_least_frozen_lower": True,
                 "A3c_migration_at_most_frozen": True, "A4_robust": True},
}


def sensitivity_checks(sens: Mapping[str, Any]) -> pd.DataFrame:
    """The review document's sensitivity statements, each checked in this run."""
    out = []
    if sens.get("summary") is None:
        out.append(_check("the sensitivity fit is feasible in every fold", False, sens["feasible"]))
        return pd.DataFrame(out)
    summ = sens["summary"].set_index("procedure")
    for (proc, field), want in STATED_SENSITIVITY.get("summary", {}).items():
        got = summ.loc[proc, field]
        same = (round(float(got), 3) == want) if isinstance(want, float) else (_canonical(got) == want)
        out.append(_check(f"{proc} {field} = {want}", same, got))
    for field, want in STATED_SENSITIVITY.get("persists", {}).items():
        got = sens["persists"][field]
        same = (round(float(got), 3) == want) if isinstance(want, float) else (got == want)
        out.append(_check(f"persists {field} = {want}", same, got))
    for (k, field), want in STATED_SENSITIVITY.get("per_fold", {}).items():
        got = sens["per_fold"][k][field]
        same = (np.allclose(got, want, rtol=0, atol=5e-4) if field.endswith(("targets", "halfwidths", "cuts"))
                else _canonical(got) == want)
        out.append(_check(f"fold {k} {field} = {want}", bool(same), got))
    for field, want in STATED_SENSITIVITY.get("criteria", {}).items():
        got = sens["criteria"][field]
        out.append(_check(f"criteria {field} = {want}", _canonical(got) == want, got))
    if not out:
        out.append(_check("the sensitivity statements are registered in STATED_SENSITIVITY", False, "empty"))
    return pd.DataFrame(out)


LIMITS = (
    "Random outer folds only. Under them period mix reproduces these bins' aggregate counts closely (Step "
    "4A); a candidate's value under a chronological split is PR 49's question.",
    "The filter is evaluated on a diagnostic population, every held-out training row, not on the Pass 2 "
    "population (the remainder after a GLASS Pass 1 fit, which no tracked artifact holds). One "
    "conjunction is screened against obligations defined for a selected union; GLASS checks them "
    "in sample at fit time, this step on held-out rows. Passing the screen is not compliance.",
    "Every registered procedure reads full-training reference inputs: share_matched its targets "
    "(frozen_shares()), gap_placed its targets and windows (frozen_shares(), window_halfwidths()), "
    "tail_selected its first two targets and its tail range (frozen_shares(), _tail_range()). All come from "
    "FROZEN_SUPPORTS, the frozen bins' in-sample supports on all 32,950 training rows under the all-rows "
    "engineer, so each fold's held-out rows and labels helped form them. The frozen cuts have the same "
    "historical provenance. No arm of the comparison is fully nested.",
    "The held-out label flips are conditional on those fixed inputs: they show that no fit-side computation "
    "reads a fold's held-out labels, not that the constants are independent of them.",
    "The fold-local sensitivity (post-registration) replaces only gap_placed's shares and windows with "
    "fit-side ones; the frozen cuts it applies keep their historical provenance. It is outside the "
    "registration and the ranking.",
    "Filter support figures are a standalone conjunction's; the router-level coverage effect is PR 44's.",
    "Inner engineers are fitted on four fifths of a fit side (about 21,088 rows), the scoring engineer on "
    "all of it (26,360). tail_selected maps the chosen tail share onto the scoring values, where the tail "
    "can hold different states from those chosen.",
    "The partner pool has `bin = 1` literals of up to two non-ECI bins. GLASS's rule space (`bin = 0` "
    "literals, beam search, the union-aware selection) is PR 44's.",
    "Refit migration compares each fold's fit with the all-rows fit, which uses every training label; it "
    "measures stability, not accuracy.",
    "Wilson intervals cover binomial sampling of the held-out rows with every fit fixed. Fold-to-fold "
    "variation is shown per fold; the fold and period removals overlap and are not independent.",
    "Five candidate fits and five partner slots are judged. The criteria were fixed first, but a chance "
    "pass remains possible; PR 44's router comparison is the confirmatory step.",
    "Every encoder uses labels of later contacts (Commit 3); this applies to all procedures alike. A "
    "past-only NSD or ECI is not built.",
)


def record_sections(*, reg_checks, heldout, fold_fits, deployed, cuts, isolation, comparison,
                    dep, filt, decision, stated, sens, sens_checks) -> Dict[str, Any]:
    """The ``binning_candidates`` record's content sections (identity keys are added by the writer)."""
    def recs(df):
        return df.to_dict("records")

    return {
        "scope": ("PR 42 Commit 4 Step 4B: fold-fitted NSD/ECI binning procedures against the frozen "
                  "PR 39 bins, and eci_warm as a filter on partner rules (training side)"),
        "registration": {"content": REGISTRATION, "sha256": registration_digest(),
                         "checks": recs(reg_checks)},
        "settings": {"seed": SEED, "n_folds": N_FOLDS, "inner_folds": INNER_FOLDS, "focal": dict(FOCAL),
                     "candidates": {f: list(v) for f, v in CANDIDATES.items()},
                     "frozen_shares": {f: list(frozen_shares(f)) for f in FAMILIES},
                     "window_halfwidths": {f: list(window_halfwidths(f)) for f in FAMILIES},
                     "tail_share_range": _tail_range(), "glass": dict(GLASS), "union": dict(UNION),
                     "partner_k": PARTNER_K, "partner_fit_support_min": PARTNER_FIT_SUPPORT_MIN,
                     "variant_fit_support_min": VARIANT_FIT_SUPPORT_MIN},
        "inputs": {"heldout_checks": recs(heldout["checks"]),
                   "inner": [{"fold": ff["fold"], "provenance": ff["inner_provenance"],
                              "checks": recs(ff["checks"])} for ff in fold_fits]},
        "fits": {"cuts": recs(cuts),
                 "folds": [{"fold": ff["fold"], "fits": ff["fits"]} for ff in fold_fits],
                 "all_rows": deployed},
        "reference_inputs": {"scope": REFERENCE_SCOPE, "inputs": recs(reference_inputs()),
                             "frozen_supports": {f: list(v) for f, v in FROZEN_SUPPORTS.items()}},
        "isolation": {"scope": ISOLATION_SCOPE, "checks": recs(isolation)},
        "comparison": {"checks": recs(comparison["checks"]), "focal": recs(comparison["focal"]),
                       "bins": recs(comparison["bins"]),
                       "tables": {f"{f} {p}": t for (f, p), t in comparison["tables"].items()},
                       "triggers": {f"{f} {p}": t for (f, p), t in comparison["triggers"].items()},
                       "migration": recs(comparison["migration"]), "joint": comparison["joint"]},
        "dependence": {"removals": recs(dep["removals"]),
                       "subscriber_sources": recs(dep["subscriber_sources"]),
                       "periods_removed": dep["periods_removed"]},
        "filter": {"population": filt["population"], "support_scope": filt["support_scope"],
                   "partners": [{"fold": ff["fold"], "chosen": ff["partners"],
                                 "eligible": int(ff["partner_table"]["eligible"].sum()),
                                 "eligible_table": recs(ff["partner_table"][ff["partner_table"]["eligible"]])}
                                for ff in fold_fits],
                   "per_fold": recs(filt["per_fold"]), "by_slot": recs(filt["by_slot"]),
                   "by_partner": recs(filt["by_partner"]), "checks": recs(filt["checks"])},
        "decision": {"criteria": recs(decision["criteria"]), "verdicts": recs(decision["verdicts"]),
                     "families": decision["families"], "eci_warm_filter": decision["eci_warm_filter"],
                     "revision": decision["revision"]},
        "stated": recs(stated),
        "sensitivity": None if sens is None else {
            "definition": sens["definition"], "feasible": sens["feasible"], "per_fold": sens["per_fold"],
            "all_rows_fit": sens.get("all_rows_fit"),
            "summary": None if sens.get("summary") is None else recs(sens["summary"]),
            "table": sens.get("table"), "triggers": sens.get("triggers"),
            "removals": None if sens.get("removals") is None else recs(sens["removals"]),
            "criteria": sens.get("criteria"), "persists": sens.get("persists"),
            "stated": recs(sens_checks)},
        "limits": list(LIMITS),
    }
