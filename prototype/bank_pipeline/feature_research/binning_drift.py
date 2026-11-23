"""
feature_research.binning_drift
==============================
PR 42 Commit 4, Step 4A: reproduction and diagnosis of the Stage 2 NSD/ECI
binning drift deferred from PR 39 Step 3A (``documentation/research_roadmap.md``,
Commit 4). Training side only, and diagnosis only: no threshold is proposed,
no binning procedure is defined and no configuration is selected (Step 4B).
The frozen binning (``shared.stage2.binning``), the PR 39 implementation
(``stage2_binning``) and its accepted record
(``research_logs/pr39/c3_20261114_231604/``) are read, never changed.

Question
--------
The lifts quoted when the NSD/ECI thresholds were locked (``eci_warm`` 1.49-1.67x,
``eci_hot`` 3.53x, ``nsd_elevated`` 1.04-1.43x) differ from the PR 39
out-of-fold lifts (about 1.03x, 2.50x and 2.49x, the last with one fold far
below the others). Why? Four candidate causes are examined, and separated only
where the evidence permits:

threshold placement   where the frozen cut points sit relative to the encoded
                      values and to the intervals the quoted figures describe;
encoder behaviour     how the fitted NSD/ECI maps change with the rows they are
                      fitted on (all training rows, or one fold's training rows);
population            which rows are evaluated: the full population or the
                      training rows, and the mix of contact periods in a bin;
label dependence      in-sample evaluation: the rows whose labels built the
                      encoding are the rows the lift is measured on.

Temporal look-ahead is a fifth. Commit 3 measured it as label provenance (which
labels enter the encodings), not as an effect on lift. The shares are reported
here by bin; their effect on any lift gap is left unresolved.

The comparisons are descriptive. Two contrasts between the in-sample, fold-fit
and out-of-fold lifts are named for what they compute, and neither isolates a
cause; in particular the held-out-minus-fit contrast is not the effect of a
row's own label. What an encoder does is shown directly by the macro states it
assigns to each bin. Identities of rows or states are checked on row masks or
on per-state, per-encoder assignments.

Sections
--------
1. Figures and provenance: ``HISTORICAL`` (transcribed, ``verify_historical``
   checks it against Git), ``exploratory_outputs`` (the tracked exploratory
   notebook's Cells 15A-15B), ``read_pr39_record`` (pinned SHA-256),
   ``implied_cut_rows``, ``figure_provenance``.
2. Reproduction: ``stage2_side``, ``reproduce_pr39`` (E-B, T1-T4 and the
   decision by import, compared with the accepted record), ``source_values``,
   ``heldout_label_checks`` (label flips that fail if a held-out label reaches
   its own encoding, each with a control that must change).
3. Encoder comparison: ``macro_states``, ``fold_fit_values``, ``state_values``,
   ``encoder_comparison`` (frozen cut points; in-sample, fold-fit and out-of-fold
   lifts and the two descriptive contrasts between them; rows that change bin).
4. Support and shape: ``support_table``, ``fold_stability``, ``decile_tables``,
   ``historical_thresholds``.
5. Contact periods: ``period_evidence`` (with non-cancelling within-period
   measures beside the period-standardized ratio), ``straddling_states``,
   ``label_provenance_by_bin``.
6. Findings: ``focal_assignments`` (the row- and state-level facts), ``findings``
   (each claim with the numbers it rests on and whether it holds in this run),
   ``claims_table``.

Every bin statistic is ``stage2_binning.evidence.bin_table`` (support, positives,
rate, lift, Wilson 95% interval of the rate divided by the base rate, expected
positives); T1-T4 are ``stage2_binning.evidence.triggers``.

No new importer of the Stage 2 feature code
-------------------------------------------
Nothing here fits an engineer. Out-of-fold values come from
``stage2_binning.runner.oof_sources`` (the PR 39 path), in-sample values from
the notebook's Cell 7 frame (``feature_inventory.build_training_frames``), and
label provenance from ``encoding_audit.audit_paths`` (Commit 3). The exploratory
notebook's printed outputs are read from their pinned Git blob (``EXPLORATORY``).
This module
imports neither ``shared.stage2.feature_engineering`` nor
``feature_research.feature_engineering``, so the Commit 2 inventory's caller
scan, and the Step 3A records Cell 30 reproduces from it, are unchanged
(``tests/feature_research/test_pr42_binning_drift.py`` keeps it so).

Nothing here reads a test row, a test label or a model artifact. The historical
figures (section 1) were computed in February 2026 on all 41,188 rows, before
the cascades' split existed. They are transcribed from the committed notebook
output, checked against Git where that commit is present, and never recomputed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["SEED", "N_FOLDS", "FAMILIES", "SOURCES", "FOCAL", "MACRO_COLUMNS", "PR39_RECORD",
           "HISTORICAL", "EXPLORATORY", "QUESTION", "CAUSES", "WILSON_ESTIMAND",
           "DECILE_DEFINITION", "STATE_TOLERANCE", "require", "read_pr39_record",
           "verify_historical", "historical_consistency", "implied_cut_rows",
           "exploratory_outputs", "figure_provenance", "Stage2Side", "stage2_side",
           "reproduce_pr39", "Sources", "source_values", "assign_bins", "assign_intervals",
           "heldout_label_checks", "MacroStates", "macro_states", "fold_fit_values", "state_values",
           "encoder_comparison", "support_table", "fold_stability", "decile_tables",
           "exploratory_reproduction", "compute_lift_parity", "historical_thresholds", "period_evidence",
           "straddling_states", "label_provenance_by_bin", "focal_assignments", "findings",
           "claims_table", "LIMITS", "PERIOD_MIN_ROWS", "record_sections"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: The shared Stage 2-3 fold plan (``make_folds(y, 5, 42)``, PR 39 ``8aea082fcaf371b7``).
SEED = 42
N_FOLDS = 5
FAMILIES = ("NSD", "ECI")
SOURCES = MappingProxyType({"NSD": "neighborhood_subscription_density",
                            "ECI": "economic_curvature_intensity"})
#: The three bins the roadmap names, with their families.
FOCAL = MappingProxyType({"nsd_elevated": "NSD", "eci_warm": "ECI", "eci_hot": "ECI"})
#: Both encodings read only these columns of a row (the KDE coordinates; the
#: three curvature variables): a row's NSD and ECI are functions of this triple.
MACRO_COLUMNS = ("euribor3m", "nr.employed", "emp.var.rate")
#: Largest spread of one encoder's value within one macro state still taken as one value.
STATE_TOLERANCE = 1e-12
N_DECILES = 10

QUESTION = ("Why do the NSD/ECI lifts quoted when the PR 39 thresholds were locked differ from the "
            "PR 39 out-of-fold lifts (eci_warm 1.49-1.67x vs about 1.03x, eci_hot 3.53x vs about "
            "2.50x, nsd_elevated about 2.49x pooled with large fold variation), and which of "
            "threshold placement, encoder behaviour, population and in-sample label dependence "
            "does the training-side evidence support?")

CAUSES = MappingProxyType({
    "threshold_placement": "where the frozen cut points sit relative to the encoded values and to the "
                           "intervals the quoted figures describe",
    "encoder_behaviour": "how the fitted NSD/ECI maps change with the rows they are fitted on",
    "population": "which rows are evaluated (full population or training rows) and the mix of "
                  "contact periods inside a bin",
    "label_dependence": "in-sample evaluation: the rows whose labels built the encoding are the rows "
                        "the lift is measured on",
    "look_ahead": "labels of later contacts in an earlier row's encoding (Commit 3: provenance, not "
                  "effect)",
})

WILSON_ESTIMAND = (
    "lift = the bin's subscription rate on the evaluated rows / the base rate of the same rows "
    "(pooled: all evaluated rows; per fold: that fold's rows). lift_wilson95 = the Wilson 95% "
    "interval of the bin's rate (stage2_binning.evidence.wilson) divided by that base rate. It "
    "covers binomial sampling of the evaluated rows only: the base rate, the fitted encoders and "
    "the cut points are held fixed, so it does not cover fold-to-fold encoder variation, and "
    "pooled out-of-fold rows, which come from five encoders, are treated as one sample. Tables "
    "over (fold, fit row) pairs count each training row four times; their pooled intervals are "
    "omitted.")

DECILE_DEFINITION = (
    "pandas.qcut(values, q=10, duplicates='drop') on the evaluated rows' values, as "
    "feature_research.model_training.rf.lift_analysis.compute_lift forms them (exact edges from "
    "retbins; intervals (lo, hi], the first including its lower edge). Rows with equal values "
    "always share a decile, so deciles of these clustered encodings hold unequal row counts. "
    "In-sample deciles: edges from the in-sample values. Out-of-fold deciles: edges from the "
    "pooled out-of-fold values, applied unchanged to each fold's held-out rows (each fold's "
    "lift against that fold's base rate).")

PR39_RECORD = MappingProxyType({
    "run_id": "c3_20261114_231604",
    "path": "research_logs/pr39/c3_20261114_231604/stage2_binning_evidence.json",
    "sha256": "799602bac7b29287dd22a5cbc391aa0d015b463cbb0df049cb717d7ed03ece6f",
})


def _check(name: str, passed: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def require(checks: pd.DataFrame, what: str) -> str:
    """Raise if any check failed; otherwise a one-line summary."""
    col = "passed" if "passed" in checks else "holds"
    bad = checks[~checks[col].astype(bool)]
    if len(bad):
        raise AssertionError(f"{what}: {len(bad)} check(s) failed\n{bad.to_string()}")
    return f"{what}: {len(checks)} checks passed"


def _canonical(obj: Any) -> Any:
    from upstream_regeneration.records import canonical
    return canonical(obj)


# ======================================================================
# 1. figures and their provenance
# ======================================================================
def _d(lo: str, hi: str, count: int, rate: str, lift: str) -> Dict[str, Any]:
    return {"lo": lo, "hi": hi, "count": count, "rate": rate, "lift": lift}


#: Transcribed from the committed output of the exploratory notebook at commit
#: 1ce1649 (2026-02-28), the run the frozen thresholds were locked from. Values
#: are the printed strings (interval edges at pandas' display precision, rates
#: to 4 and lifts to 2 decimals).
HISTORICAL = MappingProxyType({
    "source": {
        "commit": "1ce164940894bd6f745b4739e91305b5f2a3d213",
        "notebook": "prototype/glass_pipeline/exploratory_feature_research.ipynb",
        "decile_cell": "CELL 13C: RF LIFT ANALYSIS — BINNING STRATEGY",
        "bin_cell": "CELL 13D: FEATURE BINNING — LIFT-DRIVEN MULTI-BIN ENCODING",
        "date": "2026-02-28",
        "later_name": ("the cells are the exploratory notebook's Cells 15A-15B today; the frozen "
                       "thresholds' comments cite 'Cell 13C'"),
    },
    "population": {
        "rows": 41_188, "base_rate_printed": "0.1127",
        "rows_are": ("every row of bank-additional-full.csv, before the cascades' split existed "
                     "(the rows that later formed the test split are among them)"),
    },
    "encoder": ("one fit of the same NSD and ECI constructions on all 41,188 rows, applied to "
                "them (in sample): KDE bandwidth 1.0 over 5,000 reference rows drawn with "
                "np.random.seed(42) and np.random.choice (the production engineer draws with "
                "np.random.default_rng(42)); curvature of 20 quantile-bin rates per macro "
                "variable"),
    "estimands": {"deciles": "in-sample decile lift (Cell 13C)",
                  "bins": "in-sample lift of the locked bins, printed beside them (Cell 13D)"},
    "deciles": {
        "NSD": [_d("0.0481728", "0.048175", 4664, "0.0547", "0.49"),
                _d("0.048175", "0.0481754", 5100, "0.0529", "0.47"),
                _d("0.0481754", "0.0481758", 2915, "0.0504", "0.45"),
                _d("0.0481758", "0.0483684", 3994, "0.0546", "0.48"),
                _d("0.0483684", "0.0483724", 5274, "0.0309", "0.27"),
                _d("0.0483724", "0.0524146", 2785, "0.0488", "0.43"),
                _d("0.0524146", "0.190835", 4148, "0.1020", "0.91"),
                _d("0.190835", "0.192003", 4192, "0.1174", "1.04"),
                _d("0.192003", "0.230462", 4003, "0.1614", "1.43"),
                _d("0.230462", "0.350394", 4113, "0.4595", "4.08")],
        "ECI": [_d("0.00666", "0.0264", 6172, "0.0585", "0.52"),
                _d("0.0264", "0.0271", 2613, "0.0551", "0.49"),
                _d("0.0271", "0.0275", 4267, "0.0492", "0.44"),
                _d("0.0275", "0.0291", 3595, "0.0562", "0.50"),
                _d("0.0291", "0.0747", 4851, "0.1503", "1.33"),
                _d("0.0747", "0.0758", 3994, "0.0744", "0.66"),
                _d("0.0758", "0.0777", 5508, "0.0392", "0.35"),
                _d("0.0777", "0.0797", 2019, "0.1877", "1.67"),
                _d("0.0797", "0.0948", 4991, "0.1681", "1.49"),
                _d("0.0948", "0.153", 3178, "0.3974", "3.53")],
    },
    "bins": {
        "NSD": {"nsd_cold": (24122, "0.0485", "0.43"), "nsd_warm": (8430, "0.1078", "0.96"),
                "nsd_elevated": (4394, "0.1411", "1.25"), "nsd_hot": (4242, "0.4576", "4.06")},
        "ECI": {"eci_cold": (16647, "0.0551", "0.49"), "eci_mid": (6733, "0.1396", "1.24"),
                "eci_warm": (14630, "0.1039", "0.92"), "eci_hot": (3178, "0.3974", "3.53")},
    },
    #: Which historical deciles each quoted figure is (frozen code comments; roadmap).
    "quoted": {
        "eci_warm": {"figure": "1.49-1.67x", "deciles": [8, 9],
                     "comment": "warm 0.076-0.095 (1.49-1.67x)"},
        "eci_hot": {"figure": "3.53x", "deciles": [10], "comment": "hot >0.095 (3.53x)"},
        "nsd_elevated": {"figure": "1.04-1.43x", "deciles": [8, 9],
                         "comment": "warm 0.19-0.23 (1.04-1.43x)"},
    },
})

#: The exploratory notebook's accepted printed outputs: the same analysis on the training
#: rows. Read from the content-addressed Git blob committed in PR 36 (unchanged through
#: 1cd13b9), never from whatever the working copy holds: Commit 6 may clear its outputs.
EXPLORATORY = MappingProxyType({
    "notebook": "feature_research/exploratory_feature_research.ipynb",
    "repository_path": "prototype/bank_pipeline/feature_research/exploratory_feature_research.ipynb",
    "commit": "42753db5562a38de49355a6462cb9ee29c4e860a",
    "blob": "266860bac589f67ad1cc82bf007a5868cde4f652",
    "sha256_lf": "136c8fb7f44c072a6a97a6be53302e729a9660d8af7495cc065371eaabe0b6d5",
    "decile_cell": "CELL 15A: RF LIFT ANALYSIS",
    "bin_cell": "CELL 15B: FEATURE BINNING",
    "rows": ("the 32,950 training rows of the cascades' split; the research copy of the feature "
             "engineering (feature_research.feature_engineering, parity with the production copy "
             "verified in Commit 2) fitted on them and applied to them (in sample)"),
})


def read_pr39_record(root: Path = BANK_ROOT) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The accepted PR 39 Step 3A record, after checking its pinned SHA-256."""
    path = Path(root) / PR39_RECORD["path"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PR39_RECORD["sha256"]:
        raise RuntimeError(f"{PR39_RECORD['path']}: SHA-256 {digest} is not the pinned "
                           f"{PR39_RECORD['sha256']}")
    record = json.loads(data.decode("utf-8"))
    if record.get("run_id") != PR39_RECORD["run_id"]:
        raise RuntimeError(f"{PR39_RECORD['path']} is not run {PR39_RECORD['run_id']}")
    return record, {"run_id": PR39_RECORD["run_id"], "path": PR39_RECORD["path"],
                    "sha256": digest, "pinned": True,
                    "outcome": record["decision"]["outcome"]}


def _num(s: str) -> float:
    return float(str(s).replace(",", ""))


def historical_consistency(historical: Mapping[str, Any] = HISTORICAL) -> pd.DataFrame:
    """Internal arithmetic of the transcription (rows add up; deciles and bins agree)."""
    from stage2_binning import evidence as E
    out = []
    n = historical["population"]["rows"]
    for fam in FAMILIES:
        dec = historical["deciles"][fam]
        out.append(_check(f"{fam}: the ten deciles hold all {n:,} rows", sum(d["count"] for d in dec) == n,
                          sum(d["count"] for d in dec)))
        out.append(_check(f"{fam}: consecutive deciles share their edges",
                          all(a["hi"] == b["lo"] for a, b in zip(dec, dec[1:])), "edges chain"))
        bins = historical["bins"][fam]
        out.append(_check(f"{fam}: the four bins hold all {n:,} rows",
                          sum(v[0] for v in bins.values()) == n, sum(v[0] for v in bins.values())))
        out.append(_check(f"{fam}: the bins are the frozen family's, in order",
                          list(bins) == E.family_bins(fam), list(bins)))
    return pd.DataFrame(out)


def implied_cut_rows(historical: Mapping[str, Any] = HISTORICAL) -> pd.DataFrame:
    """
    For each frozen cut, the printed decile edge it was read from (the nearest)
    and the rows the printed counts put between them: decile rows at or below
    the edge minus bin rows at or below the cut. Positive: rows above the cut
    but inside that decile (the printed edges are rounded, the assignment was
    exact), so the bin and the decile differ by those rows at lock time.
    """
    from shared.stage2.binning import BINNING_STRATEGY
    rows = []
    for fam in FAMILIES:
        dec = historical["deciles"][fam]
        edges = [_num(d["hi"]) for d in dec]
        cum_dec = np.cumsum([d["count"] for d in dec])
        cum_bin = np.cumsum([v[0] for v in historical["bins"][fam].values()])
        for j, (name, _, hi) in enumerate(BINNING_STRATEGY[fam]["bins"][:-1]):
            i = int(np.argmin([abs(e - hi) for e in edges[:-1]]))
            rows.append({"family": fam, "cut": hi, "upper_of": name, "printed_edge": dec[i]["hi"],
                         "decile_upper_of": i + 1,
                         "rows_between": int(cum_dec[i] - cum_bin[j])})
    return pd.DataFrame(rows)


_DECILE_LINE = re.compile(r"^\s*\(([-0-9.e]+),\s*([-0-9.e]+)\]\s+([\d,]+)\s+([0-9.]+)\s+([0-9.]+)x")
_BIN_LINE = re.compile(r"^\s*((?:nsd|eci)_[a-z]+)\s+([\d,]+)\s+([0-9.]+)\s+([0-9.]+)x")


def _cell_text(nb: Mapping[str, Any], marker: str) -> Tuple[str, str]:
    """(source, output text) of the one code cell whose source contains ``marker``."""
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} cells contain {marker!r}")
    c = hits[0]
    text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []))
    return "".join(c["source"]), text


def _parse_historical(nb: Mapping[str, Any], historical: Mapping[str, Any]) -> Dict[str, Any]:
    _, dtext = _cell_text(nb, historical["source"]["decile_cell"])
    _, btext = _cell_text(nb, historical["source"]["bin_cell"])
    deciles: Dict[str, List[Dict[str, Any]]] = {}
    current = None
    for line in dtext.splitlines():
        if "🔹" in line:
            current = next((f for f in FAMILIES if SOURCES[f] in line), None)
            if current:
                deciles[current] = []
            continue
        m = _DECILE_LINE.match(line)
        if m and current:
            deciles[current].append(_d(m.group(1), m.group(2), int(m.group(3).replace(",", "")),
                                       m.group(4), m.group(5)))
    bins: Dict[str, Dict[str, Tuple[int, str, str]]] = {f: {} for f in FAMILIES}
    for line in btext.splitlines():
        m = _BIN_LINE.match(line)
        if m:
            fam = "NSD" if m.group(1).startswith("nsd") else "ECI"
            bins[fam][m.group(1)] = (int(m.group(2).replace(",", "")), m.group(3), m.group(4))
    return {"deciles": deciles, "bins": bins}


def verify_historical(root: Path = BANK_ROOT, historical: Mapping[str, Any] = HISTORICAL
                      ) -> Dict[str, Any]:
    """
    The transcription against the committed notebook output, read with
    ``git show <commit>:<notebook>``. ``status`` is ``verified``, ``mismatch``
    or ``unavailable`` (no Git, or the commit is not in this checkout, e.g. a
    shallow clone); the internal arithmetic is checked either way.
    """
    src = historical["source"]
    arithmetic = historical_consistency(historical)
    try:
        out = subprocess.run(["git", "show", f"{src['commit']}:{src['notebook']}"], cwd=root,
                             capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "detail": f"git not runnable: {exc}",
                "arithmetic": arithmetic.to_dict("records")}
    if out.returncode != 0:
        return {"status": "unavailable",
                "detail": f"commit {src['commit'][:7]} not readable here: "
                          f"{out.stderr.decode('utf-8', 'replace').strip()[:200]}",
                "arithmetic": arithmetic.to_dict("records")}
    nb = json.loads(out.stdout.decode("utf-8"))
    parsed = _parse_historical(nb, historical)
    want = {"deciles": {f: list(historical["deciles"][f]) for f in FAMILIES},
            "bins": {f: dict(historical["bins"][f]) for f in FAMILIES}}
    differing = [f"{part}.{f}" for part in ("deciles", "bins") for f in FAMILIES
                 if _canonical(parsed[part].get(f)) != _canonical(want[part][f])]
    blob = hashlib.sha256(out.stdout).hexdigest()
    return {"status": "mismatch" if differing else "verified",
            "detail": (f"differs: {differing}" if differing else
                       f"every transcribed decile and bin row equals {src['commit'][:7]}'s output"),
            "notebook_sha256": blob, "arithmetic": arithmetic.to_dict("records")}


_LIFT_LINE = re.compile(r"^\s{2}(\S+)\s+((?:[0-9.]+[▲▼]?\s*)+)$")
_SHARE_ITEM = re.compile(r"([a-z0-9_]+) ([0-9.]+)×[▲▼]? \((\d+)%\)")


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _pinned_exploratory(root: Path, working_copy: Optional[Path]) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """The pinned notebook's bytes and where they came from (module docstring of ``exploratory_outputs``)."""
    src = {"commit": EXPLORATORY["commit"], "blob": EXPLORATORY["blob"],
           "path": EXPLORATORY["repository_path"], "sha256_lf": EXPLORATORY["sha256_lf"]}
    git_detail = None
    try:
        out = subprocess.run(["git", "cat-file", "blob", EXPLORATORY["blob"]], cwd=root,
                             capture_output=True, timeout=60)
        if out.returncode == 0:
            if _lf_sha256(out.stdout) == EXPLORATORY["sha256_lf"]:
                return out.stdout, {**src, "status": "verified", "source": "git blob",
                                    "detail": f"blob {EXPLORATORY['blob'][:12]} read from Git; its SHA-256 "
                                              "is the pinned one"}
            return None, {**src, "status": "mismatch", "source": "git blob",
                          "detail": "the Git blob's SHA-256 is not the pinned one"}
        git_detail = out.stderr.decode("utf-8", "replace").strip()[:200]
    except (OSError, subprocess.SubprocessError) as exc:
        git_detail = f"git not runnable: {exc}"
    wc = Path(working_copy) if working_copy is not None else Path(root) / EXPLORATORY["notebook"]
    if wc.is_file():
        data = wc.read_bytes()
        if _lf_sha256(data) == EXPLORATORY["sha256_lf"]:
            return data, {**src, "status": "verified", "source": "working copy",
                          "detail": f"blob not in Git here ({git_detail}); the working copy's LF SHA-256 is "
                                    "the pinned one"}
        return None, {**src, "status": "unavailable", "source": None,
                      "detail": f"blob not in Git here ({git_detail}); the working copy differs from the "
                                "pinned notebook (for example, its outputs were cleared)"}
    return None, {**src, "status": "unavailable", "source": None,
                  "detail": f"blob not in Git here ({git_detail}); no working copy"}


def exploratory_outputs(root: Path = BANK_ROOT, *, working_copy: Optional[Path] = None) -> Dict[str, Any]:
    """
    The printed NSD/ECI decile lifts (Cell 15A) and bin lifts and row shares
    (Cell 15B) of the exploratory notebook as accepted (``EXPLORATORY``): the
    content-addressed blob ``git cat-file blob <blob>``, checked against its
    pinned SHA-256. Only when Git does not have the blob (a shallow clone) is
    the working copy read, and then only if its LF SHA-256 is the pinned one.
    ``status`` is ``verified`` (with ``source``), ``mismatch`` or
    ``unavailable``; the values are None unless verified, and the
    reproduction in Cell 36 is then reported as not checked.
    """
    data, src = _pinned_exploratory(Path(root), working_copy)
    if data is None:
        return {**src, "deciles": None, "bins": None}
    nb = json.loads(data.decode("utf-8"))
    _, dtext = _cell_text(nb, EXPLORATORY["decile_cell"])
    _, btext = _cell_text(nb, EXPLORATORY["bin_cell"])
    deciles = {}
    for line in dtext.splitlines():
        m = _LIFT_LINE.match(line)
        if m and m.group(1) in SOURCES.values():
            fam = next(f for f in FAMILIES if SOURCES[f] == m.group(1))
            deciles[fam] = re.findall(r"[0-9]+\.[0-9]{2}", m.group(2))
    bins = {}
    for line in btext.splitlines():
        s = line.strip()
        fam = s.split(" ")[0] if s else ""
        if fam in FAMILIES:
            bins[fam] = {f"{fam.lower()}_{n}": {"lift": l, "share_pct": int(p)}
                         for n, l, p in _SHARE_ITEM.findall(s)}
    return {**src, "deciles": deciles, "bins": bins}


def figure_provenance(pr39: Mapping[str, Any], historical: Mapping[str, Any] = HISTORICAL
                      ) -> pd.DataFrame:
    """What each quoted figure measures: rows, encoder fit, interval, denominator, reproducibility."""
    fams = pr39["eb"]["families"]
    hist_pop = (f"{historical['population']['rows']:,} rows (full population, before the split)")
    fit_all = "one fit on the same 41,188 rows (in sample; legacy reference draw)"
    rows = []
    for b, q in historical["quoted"].items():
        fam = FOCAL[b]
        dec = historical["deciles"][fam]
        parts = [f"({dec[i - 1]['lo']}, {dec[i - 1]['hi']}] {dec[i - 1]['lift']}x on "
                 f"{dec[i - 1]['count']:,} rows" for i in q["deciles"]]
        hb = historical["bins"][fam][b]
        oof = fams[fam]["oof"]["bins"][b]["pooled"]
        ins = fams[fam]["in_sample"]["bins"][b]["pooled"]
        rows += [
            {"bin": b, "figure": q["figure"], "source": f"commit {historical['source']['commit'][:7]}, "
             "Cell 13C (frozen code comment)", "estimand": "in-sample decile lift",
             "intervals": "; ".join(parts), "rows": hist_pop, "encoder_fit": fit_all,
             "denominator": f"base rate of the 41,188 rows ({historical['population']['base_rate_printed']})",
             "reproduced_here": "no: population includes the later test rows (G7); transcribed, "
                                "Git-verified where the commit is present"},
            {"bin": b, "figure": f"{hb[2]}x", "source": f"commit {historical['source']['commit'][:7]}, "
             "Cell 13D (printed beside the locked bins)", "estimand": "in-sample bin lift",
             "intervals": f"the frozen bin, {hb[0]:,} rows", "rows": hist_pop, "encoder_fit": fit_all,
             "denominator": f"base rate of the 41,188 rows ({historical['population']['base_rate_printed']})",
             "reproduced_here": "no (as above)"},
            {"bin": b, "figure": f"{ins['lift']:.3f}x", "source": f"PR 39 {pr39['run_id']} E-B in_sample; "
             "exploratory notebook Cell 15B", "estimand": "in-sample bin lift",
             "intervals": f"the frozen bin, {ins['support']:,} rows",
             "rows": "32,950 training rows", "encoder_fit": "one fit on the 32,950 training rows (in sample)",
             "denominator": "base rate of the training rows (3,712 / 32,950)",
             "reproduced_here": "yes (exactly, Cell 32)"},
            {"bin": b, "figure": f"{oof['lift']:.3f}x", "source": f"PR 39 {pr39['run_id']} E-B oof",
             "estimand": "pooled out-of-fold bin lift", "intervals": f"the frozen bin, {oof['support']:,} rows",
             "rows": "32,950 training rows, each scored by the engineer of its outer fold",
             "encoder_fit": "five fits, each on the 26,360 rows outside one outer fold",
             "denominator": "base rate of the training rows (3,712 / 32,950)",
             "reproduced_here": "yes (exactly, Cell 32)"},
        ]
    return pd.DataFrame(rows)


# ======================================================================
# 2. reproduction
# ======================================================================
@dataclass(frozen=True)
class Stage2Side:
    """The training rows and the shared outer fold plan, as ``stage2_binning.runner`` reads them."""

    X_train: pd.DataFrame
    y_train: pd.Series
    folds: tuple
    fold_id: np.ndarray
    fold_fingerprint: str


def stage2_side(X: pd.DataFrame, y: pd.Series) -> Stage2Side:
    """The PR 39 outer plan on the training rows (``make_folds(y, 5, 42)``); no test row."""
    from glass_pipeline.glass_router.pipeline import fold_fingerprint, make_folds
    if not X.index.equals(y.index):
        raise ValueError("X and y are not index-aligned")
    y = y.astype(int)
    folds = tuple((np.asarray(tr), np.asarray(va)) for tr, va in make_folds(y, N_FOLDS, SEED))
    fold_id = np.full(len(y), -1, dtype=int)
    for k, (_, va) in enumerate(folds):
        fold_id[va] = k
    return Stage2Side(X, y, folds, fold_id, fold_fingerprint(fold_id))


def reproduce_pr39(side: Stage2Side, record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    E-B (``stage2_binning.runner.eb_evidence``), T1-T4 (``evidence.triggers``)
    and the retain-or-revise decision (``evidence.decide``), recomputed by
    import and compared with the accepted record section by section.
    """
    from stage2_binning import evidence as E
    from stage2_binning import runner as R
    eb = _canonical(R.eb_evidence(side))
    trig = _canonical({fam: E.triggers(fam, eb["families"][fam]["oof"]) for fam in E.FAMILIES})
    decision = _canonical(E.decide(trig))
    pop = record["population"]
    pinned = pop["frozen_protocol_check"]["checks"]
    y = side.y_train.to_numpy()
    fold_rows = [int((side.fold_id == k).sum()) for k in range(len(side.folds))]
    fold_pos = [int(y[side.fold_id == k].sum()) for k in range(len(side.folds))]
    diff = [k for k in sorted(set(eb) | set(record["eb"])) if eb.get(k) != record["eb"].get(k)]
    fam_diff = [f"{f}.{k}" for f in E.FAMILIES for k in eb["families"][f]
                if eb["families"][f][k] != record["eb"]["families"][f].get(k)]
    checks = [
        _check("the training rows are the PR 39 population (rows, positives)",
               (len(y), int(y.sum())) == (pop["rows"], pop["positives"]), f"{len(y)} rows, {int(y.sum())} positives"),
        _check("the outer folds are the PR 39 plan (fold fingerprint)",
               side.fold_fingerprint == pop["fold_fingerprint"], side.fold_fingerprint),
        _check("fold rows and positives are the pinned ones",
               fold_rows == pinned["fold_rows"]["pinned"] and fold_pos == pinned["fold_positives"]["pinned"],
               f"{fold_rows} / {fold_pos}"),
        _check("no held-out row is in its own fold's engineer fit",
               all(p["heldout_rows_in_fit"] == 0 for p in eb["fold_fits"]),
               [p["heldout_rows_in_fit"] for p in eb["fold_fits"]]),
        _check("E-B (out-of-fold and in-sample tables, moved rows, NSD x ECI overlap, fold fits) "
               "reproduces the accepted record exactly", not diff,
               "identical" if not diff else f"differs: {diff} {fam_diff}"),
        _check("T1-T4 reproduce the accepted record exactly", trig == record["triggers"],
               {f: [k for k in ("T1", "T2", "T3", "T4") if trig[f][k]["fired"]] for f in trig}),
        _check("the decision (retain both families) reproduces the accepted record",
               decision == record["decision"]["families"],
               {f: d["outcome"] for f, d in decision.items()}),
    ]
    return {"eb": eb, "triggers": trig, "decision": decision, "checks": pd.DataFrame(checks)}


def assign_bins(values, family: str) -> np.ndarray:
    """Each value's frozen bin name in ``family`` ((lo, hi] intervals, ``BINNING_STRATEGY``)."""
    from shared.stage2.binning import BINNING_STRATEGY
    v = np.asarray(values, dtype=float)
    if np.isnan(v).any():
        raise ValueError(f"{family}: NaN values cannot be binned")
    out = np.empty(len(v), dtype=object)
    for name, lo, hi in BINNING_STRATEGY[family]["bins"]:
        m = (v <= hi) if lo is None else ((v > lo) if hi is None else ((v > lo) & (v <= hi)))
        out[m] = name
    return out


def assign_intervals(values, edges: Sequence[float], names: Sequence[str]) -> np.ndarray:
    """Each value's interval among (-inf, e1], (e1, e2], ..., (ek, inf); ``names`` has k + 1 entries."""
    edges = np.asarray(edges, dtype=float)
    if len(names) != len(edges) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("need increasing edges and one more name than edges")
    idx = np.searchsorted(edges, np.asarray(values, dtype=float), side="left")
    return np.asarray(names, dtype=object)[idx]


@dataclass(frozen=True)
class Sources:
    """Row-level NSD/ECI values and frozen-bin labels, in training-row order."""

    values: Mapping[str, Mapping[str, np.ndarray]]      # kind -> family -> values
    labels: Mapping[str, Mapping[str, np.ndarray]]      # kind -> family -> bin names
    provenance: List[Dict[str, Any]]                     # the out-of-fold fits
    checks: pd.DataFrame


def source_values(side: Stage2Side, in_sample_sources: pd.DataFrame,
                  eb: Mapping[str, Any]) -> Sources:
    """
    Out-of-fold values from ``stage2_binning.runner.oof_sources`` (PR 39's fresh
    engineer per outer fold), in-sample values from the caller (the notebook's
    Cell 7 ``TRAINING_FRAMES['stage2_sources']``: the production engineer fitted
    on all training rows), both binned through the production path
    (``stage2_crossfit.g13.bin_source``). Their tables must equal E-B's.
    """
    from stage2_binning import evidence as E
    from stage2_binning import runner as R
    from stage2_crossfit.g13 import bin_source
    oof_src, provenance = R.oof_sources(side)
    idx = side.X_train.index
    aligned = in_sample_sources.index.equals(idx) or (
        len(in_sample_sources) == len(idx) and in_sample_sources.index.isin(idx).all())
    ins = in_sample_sources.loc[idx]
    y = side.y_train.to_numpy()
    values, labels = {"oof": {}, "in_sample": {}}, {"oof": {}, "in_sample": {}}
    checks = [_check("the in-sample frame holds exactly the training rows", aligned, len(in_sample_sources)),
              _check("the out-of-fold fits are E-B's (rows, positives, label fingerprints, isolation)",
                     _canonical(provenance) == eb["fold_fits"], "fold_fits")]
    for kind, src in (("oof", oof_src), ("in_sample", ins)):
        binned = bin_source(src)
        for fam in FAMILIES:
            lab = E.family_labels(binned, fam)
            labels[kind][fam] = lab
            values[kind][fam] = src[SOURCES[fam]].to_numpy(dtype=float)
            table = E.bin_table(lab, y, side.fold_id, E.family_bins(fam), len(side.folds),
                                per_fold=kind == "oof")
            checks.append(_check(f"{fam} {kind}: the bin table from these values is E-B's",
                                 _canonical(table) == eb["families"][fam][kind], kind))
            checks.append(_check(f"{fam} {kind}: assign_bins agrees with the production binning path",
                                 bool(np.array_equal(assign_bins(values[kind][fam], fam), lab)), kind))
    return Sources(values=values, labels=labels, provenance=_canonical(provenance),
                   checks=pd.DataFrame(checks))


def heldout_label_checks(side: Stage2Side, *, folds: Optional[Sequence[int]] = None,
                         oof_fn: Optional[Callable[[Stage2Side], pd.DataFrame]] = None) -> pd.DataFrame:
    """
    For each outer fold k: flip every held-out label of k and recompute the
    out-of-fold sources. Fold k's NSD/ECI must not move (its labels never reach
    its own encodings); as a control, some other fold's values must move (the
    flipped labels are in their fits), so the check can fail. ``oof_fn`` is
    the out-of-fold computation (default ``stage2_binning.runner.oof_sources``).
    """
    from stage2_binning import runner as R
    fn = oof_fn or (lambda s: R.oof_sources(s)[0])
    cols = [SOURCES[f] for f in FAMILIES]
    base = fn(side)[cols]
    out = []
    for k in (range(len(side.folds)) if folds is None else folds):
        held = np.asarray(side.fold_id) == k
        y = side.y_train.copy()
        y[held] = 1 - y[held]
        moved = fn(replace(side, y_train=y))[cols]
        same = base.loc[held].equals(moved.loc[held])
        changed = int((base.loc[~held].to_numpy() != moved.loc[~held].to_numpy()).any(axis=1).sum())
        out.append(_check(f"fold {k}: flipping its {int(held.sum()):,} held-out labels leaves its held-out "
                          "NSD and ECI unchanged", same, "identical" if same else "values moved"))
        out.append(_check(f"fold {k}: control, the same flip moves other folds' held-out values",
                          changed > 0, f"{changed:,} rows of other folds moved"))
    return pd.DataFrame(out)


# ======================================================================
# 3. macro states and the encoder comparison
# ======================================================================
@dataclass(frozen=True)
class MacroStates:
    """Each training row's macro state (``MACRO_COLUMNS`` triple), numbered in source order."""

    code: np.ndarray
    table: pd.DataFrame


def macro_states(X: pd.DataFrame, T: pd.DataFrame, y: pd.Series) -> MacroStates:
    """States numbered by first contact (source position); rows, positives, periods per state."""
    if not T.index.equals(X.index):
        raise ValueError("the time index is not aligned with the training rows")
    key = X[list(MACRO_COLUMNS)].to_numpy(dtype=float)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    pos = T["position"].to_numpy()
    first = pd.Series(pos).groupby(inv).min()
    rank = np.empty(len(first), dtype=np.int64)
    rank[first.sort_values(kind="stable").index.to_numpy()] = np.arange(len(first))
    code = rank[inv]
    yv = np.asarray(y).astype(int)
    df = pd.DataFrame({"state": code, "y": yv, "period": T["period"].to_numpy(),
                       "period_index": T["period_index"].to_numpy(),
                       **{c: X[c].to_numpy(dtype=float) for c in MACRO_COLUMNS}})
    g = df.groupby("state")
    table = pd.DataFrame({
        "rows": g.size(), "positives": g["y"].sum(),
        "periods": g["period"].agg(lambda s: ",".join(sorted(set(s)))),
        "first_period_index": g["period_index"].min(),
        **{c: g[c].first() for c in MACRO_COLUMNS}})
    table["rate"] = table["positives"] / table["rows"]
    table["lift"] = table["rate"] / yv.mean()
    return MacroStates(code=code, table=table.reset_index())


def fold_fit_values(side: Stage2Side, states: MacroStates, eb: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Each outer fold's engineer evaluated on its own fit rows, through the PR 39
    function itself (``stage2_binning.runner.oof_sources``). For fold k the
    side has two groups: group 0 is fold k's fit rows in their original order,
    group 1 one representative row per macro state among them (new index
    labels). ``oof_sources`` fits group 1's engineer on group 0, i.e. on exactly
    fold k's fit rows (same rows, order and labels; its label fingerprint must
    be E-B's for fold k), and transforms the representatives. Group 0's
    engineer (fitted on the representatives) is discarded.

    Returns per fold the value per macro state (``{family: {state: value}}``)
    and the checks; ``state_values`` verifies these values against fold k's
    held-out values for every state both contain.
    """
    from stage2_binning import runner as R
    X, y, code = side.X_train, side.y_train, states.code
    out, checks = {}, []
    for k, (tr, _) in enumerate(side.folds):
        A, yA, cA = X.iloc[tr], y.iloc[tr], code[tr]
        first = pd.Series(np.arange(len(tr))).groupby(cA).first()
        reps = A.iloc[first.to_numpy()].copy()
        reps.index = pd.Index(-1 - np.arange(len(reps)), dtype=X.index.dtype)
        X2 = pd.concat([A, reps])
        y2 = pd.concat([yA, pd.Series(yA.iloc[first.to_numpy()].to_numpy(), index=reps.index)])
        g = np.r_[np.zeros(len(A), dtype=int), np.ones(len(reps), dtype=int)]
        n2 = np.arange(len(X2))
        side2 = Stage2Side(X2, y2.astype(int), ((n2[g != 0], n2[g == 0]), (n2[g != 1], n2[g == 1])),
                           g, "two-group side for fold-fit values")
        src2, prov2 = R.oof_sources(side2)
        fit = _canonical(prov2[1])
        want = eb["fold_fits"][k]
        same_fit = (fit["fit_rows"], fit["fit_positives"], fit["fit_label_fingerprint"]) == \
            (want["fit_rows"], want["fit_positives"], want["fit_label_fingerprint"])
        checks.append(_check(f"fold {k}: the evaluating engineer is fitted on exactly fold {k}'s fit rows "
                             "(rows, positives, label fingerprint)", same_fit,
                             f"{fit['fit_rows']:,} rows, {fit['fit_label_fingerprint']}"))
        vals = src2.loc[reps.index]
        out[k] = {fam: dict(zip(first.index.to_numpy().tolist(), vals[SOURCES[fam]].to_numpy(dtype=float)))
                  for fam in FAMILIES}
    return {"values": out, "checks": pd.DataFrame(checks)}


def state_values(states: MacroStates, sources: Sources, side: Stage2Side,
                 fold_fit: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Each training-side encoder's NSD/ECI value per macro state: ``in_sample``
    (the engineer fitted on all training rows) and ``fold_k`` (fold k's engineer:
    from its held-out rows, and from ``fold_fit_values`` for the states of its
    fit rows). Checks that each encoder's value is constant within a state and
    that the two sources of ``fold_k`` agree wherever both have the state.
    """
    code = states.code
    S = len(states.table)
    out, checks = {}, []
    for fam in FAMILIES:
        cols, spreads, agree = {}, {}, {}
        for enc in ["in_sample"] + [f"fold_{k}" for k in range(len(side.folds))]:
            if enc == "in_sample":
                rows = np.arange(len(code))
                v = sources.values["in_sample"][fam]
            else:
                rows = side.folds[int(enc.split("_")[1])][1]
                v = sources.values["oof"][fam][rows]
            s = pd.Series(v).groupby(code[rows])
            vmin, vmax = s.min(), s.max()
            col = np.full(S, np.nan)
            col[vmin.index.to_numpy()] = vmin.to_numpy()
            spreads[enc] = float((vmax - vmin).max())
            if enc != "in_sample":
                fit = fold_fit["values"][int(enc.split("_")[1])][fam]
                st = np.fromiter(fit.keys(), dtype=np.int64)
                fv = np.fromiter(fit.values(), dtype=float)
                both = ~np.isnan(col[st])
                agree[enc] = float(np.abs(col[st][both] - fv[both]).max()) if both.any() else 0.0
                col[st[~both]] = fv[~both]
            cols[enc] = col
        frame = pd.DataFrame(cols)
        frame.index.name = "state"
        out[fam] = frame
        worst = max(spreads.values())
        checks.append(_check(f"{fam}: every encoder's value is a function of the macro state "
                             f"(largest within-state spread {worst:.1e})", worst <= STATE_TOLERANCE, spreads))
        worst_a = max(agree.values())
        checks.append(_check(f"{fam}: each fold engineer's values on its fit rows' states equal its held-out "
                             f"values wherever both exist (largest difference {worst_a:.1e})",
                             worst_a <= STATE_TOLERANCE, agree))
        checks.append(_check(f"{fam}: every encoder has a value for every macro state",
                             not frame.isna().any().any(), int(frame.isna().sum().sum())))
    return {"values": out, "checks": pd.concat([fold_fit["checks"], pd.DataFrame(checks)],
                                               ignore_index=True)}


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """a - b, or None when either lift is undefined (an empty bin)."""
    return None if a is None or b is None else a - b


def _strip_pooled_interval(table: Mapping[str, Any]) -> Dict[str, Any]:
    t = json.loads(json.dumps(_canonical(table)))
    for b in t["bins"].values():
        b["pooled"].pop("lift_wilson95", None)
    return t


def encoder_comparison(side: Stage2Side, sources: Sources, states: MacroStates,
                       svals: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Three evaluations per bin with the frozen cut points, each lift against the
    base rate of the rows it evaluates:

    ``in_sample``  the engineer fitted on all training rows, on those rows;
    ``fold_fit``   each fold's engineer on its own fit rows (pairs (fold, row):
                   every training row appears in four fits);
    ``oof``        each fold's engineer on its held-out rows (PR 39).

    Two descriptive contrasts, named for what they compute, add up to
    ``oof_minus_in_sample``:

    ``fold_fit_minus_in_sample``  different encoders (five fold fits against the
                                  all-rows fit), each evaluated on the rows that
                                  built it;
    ``heldout_minus_fold_fit``    the same five encoders, evaluated on different
                                  rows (held-out against fit rows), pooled.

    Neither isolates a cause. ``heldout_minus_fold_fit`` is not the effect of a
    row's own label: within a fold the held-out and fit rows differ in
    composition, and fold-level differences of opposite sign cancel in the pool
    (``per_fold`` gives them). The direct evidence of what an encoder does is
    which macro states it assigns to each bin (``focal_assignments``,
    ``straddling_states``) and which rows change bin (``migration``).
    """
    from stage2_binning import evidence as E
    y = side.y_train.to_numpy()
    fid = np.asarray(side.fold_id)
    nf = len(side.folds)
    code = states.code
    out: Dict[str, Any] = {"families": {}, "definition": encoder_comparison.__doc__.strip()}
    decomposition, per_fold, migration = [], [], []
    for fam in FAMILIES:
        names = E.family_bins(fam)
        labs, ys, ks = [], [], []
        for k, (tr, _) in enumerate(side.folds):
            v = svals["values"][fam][f"fold_{k}"].to_numpy()[code[tr]]
            ok = ~np.isnan(v)
            labs.append(assign_bins(v[ok], fam))
            ys.append(y[tr][ok])
            ks.append(np.full(int(ok.sum()), k))
        fit = E.bin_table(np.concatenate(labs), np.concatenate(ys), np.concatenate(ks), names, nf)
        oof = E.bin_table(sources.labels["oof"][fam], y, fid, names, nf)
        ins = E.bin_table(sources.labels["in_sample"][fam], y, fid, names, nf, per_fold=False)
        for b in names:
            i, f, o = (t["bins"][b]["pooled"] for t in (ins, fit, oof))
            decomposition.append({
                "family": fam, "bin": b, "in_sample_support": i["support"], "in_sample_lift": i["lift"],
                "fold_fit_pairs": f["support"], "fold_fit_lift": f["lift"],
                "oof_support": o["support"], "oof_lift": o["lift"],
                "fold_fit_minus_in_sample": _sub(f["lift"], i["lift"]),
                "heldout_minus_fold_fit": _sub(o["lift"], f["lift"]),
                "oof_minus_in_sample": _sub(o["lift"], i["lift"])})
            for k in range(nf):
                fk, ok_ = fit["bins"][b]["folds"][k], oof["bins"][b]["folds"][k]
                per_fold.append({"family": fam, "bin": b, "fold": k,
                                 "fit_rows": fk["support"], "fit_lift": fk["lift"],
                                 "heldout_rows": ok_["support"], "heldout_lift": ok_["lift"],
                                 "heldout_minus_fit": _sub(ok_["lift"], fk["lift"])})
        li, lo = sources.labels["in_sample"][fam], sources.labels["oof"][fam]
        base = y.mean()
        for a in names:
            for b in names:
                m = (li == a) & (lo == b)
                if m.any():
                    migration.append({"family": fam, "in_sample_bin": a, "oof_bin": b,
                                      "rows": int(m.sum()), "positives": int(y[m].sum()),
                                      "lift": float(y[m].mean() / base),
                                      **{f"fold_{k}_rows": int((m & (fid == k)).sum()) for k in range(nf)}})
        out["families"][fam] = {"fold_fit": _strip_pooled_interval(fit)}
    out["decomposition"] = pd.DataFrame(decomposition)
    out["per_fold"] = pd.DataFrame(per_fold)
    out["migration"] = pd.DataFrame(migration)
    return out


# ======================================================================
# 4. support, uncertainty and shape
# ======================================================================
def _flat(table: Mapping[str, Any], fam: str, scope: str, per_fold: bool) -> List[Dict[str, Any]]:
    rows = []
    for b, e in table["bins"].items():
        cells = [("pooled", e["pooled"])] + ([(f"fold_{k}", c) for k, c in enumerate(e["folds"])]
                                             if per_fold else [])
        for where, c in cells:
            lo, hi = c.get("lift_wilson95", [None, None])
            rows.append({"family": fam, "bin": b, "scope": scope, "rows": where, "support": c["support"],
                         "positives": c["positives"], "rate": c["rate"], "lift": c["lift"],
                         "lift_wilson95_lo": lo, "lift_wilson95_hi": hi,
                         "expected_positives": c["expected_positives"]})
    return rows


def support_table(eb: Mapping[str, Any]) -> pd.DataFrame:
    """The frozen bins in sample (pooled) and out of fold (pooled and per fold), flattened."""
    rows = []
    for fam in FAMILIES:
        rows += _flat(eb["families"][fam]["in_sample"], fam, "in_sample", False)
        rows += _flat(eb["families"][fam]["oof"], fam, "oof", True)
    return pd.DataFrame(rows)


def fold_stability(eb: Mapping[str, Any], triggers: Mapping[str, Any]) -> pd.DataFrame:
    """Per bin: the out-of-fold fold lifts' range, folds above 1 and the T1 margin."""
    from stage2_binning import evidence as E
    rows = []
    for fam in FAMILIES:
        for b in E.family_bins(fam):
            e = eb["families"][fam]["oof"]["bins"][b]
            lifts = [c["lift"] for c in e["folds"]]
            sup = [c["support"] for c in e["folds"]]
            rows.append({"family": fam, "bin": b, "pooled_lift": e["pooled"]["lift"],
                         "fold_lift_min": min(lifts), "fold_lift_max": max(lifts),
                         "fold_lift_range": max(lifts) - min(lifts),
                         "folds_above_1": int(sum(l > 1 for l in lifts)),
                         "fold_support_min": min(sup), "fold_support_max": max(sup),
                         "t1_min_expected_positives":
                             triggers[fam]["T1"]["min_expected_positives_by_bin"][b]})
    return pd.DataFrame(rows)


def _deciles(values: np.ndarray, y: np.ndarray, fold_id: np.ndarray, fam: str, prefix: str,
             per_fold: bool, n_folds: int) -> Dict[str, Any]:
    from shared.stage2.binning import BINNING_STRATEGY
    from stage2_binning import evidence as E
    cats, edges = pd.qcut(pd.Series(values), q=N_DECILES, duplicates="drop", retbins=True)
    codes = cats.cat.codes.to_numpy()
    names = [f"{prefix}{i + 1:02d}" for i in range(len(edges) - 1)]
    lab = np.asarray(names, dtype=object)[codes]
    table = E.bin_table(lab, y, fold_id, names, n_folds, per_fold=per_fold)
    cuts = [hi for _, _, hi in BINNING_STRATEGY[fam]["bins"][:-1]]
    frozen = assign_bins(values, fam)
    intervals = []
    for i, iv in enumerate(cats.cat.categories):
        m = codes == i
        fb = pd.Series(frozen[m]).value_counts()
        intervals.append({"decile": names[i], "lo": float(edges[i]), "hi": float(edges[i + 1]),
                          "display": str(iv), "rows": int(m.sum()),
                          "frozen_bin_rows": {b: int(fb.get(b, 0)) for b in E.family_bins(fam)
                                              if fb.get(b, 0)},
                          "frozen_cuts_inside": [c for c in cuts if edges[i] < c < edges[i + 1]]})
    return {"intervals": intervals, "table": _canonical(table), "labels": lab}


def decile_tables(side: Stage2Side, sources: Sources) -> Dict[str, Any]:
    """
    In-sample deciles (in-sample edges) and out-of-fold deciles (pooled edges,
    per fold). ``labels`` holds each row's decile for the row-level checks of
    ``focal_assignments``; it is not part of the record.
    """
    y = side.y_train.to_numpy()
    fid = np.asarray(side.fold_id)
    out: Dict[str, Any] = {"definition": DECILE_DEFINITION, "families": {}, "labels": {}}
    flat = []
    for fam in FAMILIES:
        out["families"][fam] = {}
        out["labels"][fam] = {}
        for kind in ("in_sample", "oof"):
            d = _deciles(sources.values[kind][fam], y, fid, fam, f"{fam.lower()}_d", kind == "oof",
                         len(side.folds))
            out["families"][fam][kind] = {"intervals": d["intervals"], "table": d["table"]}
            out["labels"][fam][kind] = d["labels"]            # row labels: in memory, not recorded
            for iv in d["intervals"]:
                e = d["table"]["bins"][iv["decile"]]
                row = {"family": fam, "kind": kind, "decile": iv["decile"], "display": iv["display"],
                       "lo": iv["lo"], "hi": iv["hi"], "rows": e["pooled"]["support"],
                       "positives": e["pooled"]["positives"], "lift": e["pooled"]["lift"],
                       "lift_wilson95_lo": e["pooled"]["lift_wilson95"][0],
                       "lift_wilson95_hi": e["pooled"]["lift_wilson95"][1],
                       "frozen_bins": ", ".join(f"{b} {n:,}" for b, n in iv["frozen_bin_rows"].items()),
                       "frozen_cuts_inside": ", ".join(map(str, iv["frozen_cuts_inside"]))}
                if kind == "oof":
                    lifts = [c["lift"] for c in e["folds"]]
                    seen = [l for l in lifts if l is not None]
                    row.update({f"fold_{k}_lift": l for k, l in enumerate(lifts)})
                    row["folds_with_rows"] = len(seen)
                    row["fold_lift_min"] = min(seen) if seen else None
                    row["fold_lift_max"] = max(seen) if seen else None
                flat.append(row)
    out["table"] = pd.DataFrame(flat)
    return out


def exploratory_reproduction(deciles: Mapping[str, Any], eb: Mapping[str, Any],
                             exploratory: Mapping[str, Any]) -> pd.DataFrame:
    """
    The in-sample deciles and bins against the exploratory notebook's accepted
    printed values (``exploratory_outputs``: Cells 15A-15B, lifts to 2
    decimals, row shares to whole percent). If that source is not verified,
    each comparison is reported as not checked, with the reason; a mismatch of
    the pinned blob fails.
    """
    out = []
    if exploratory.get("status") != "verified":
        bad = exploratory.get("status") == "mismatch"
        for fam in FAMILIES:
            for cell in ("15A decile lifts", "15B bin lifts and shares"):
                out.append(_check(f"{fam}: exploratory Cell {cell} not checked (pinned source "
                                  f"{exploratory.get('status')})", not bad, exploratory.get("detail")))
        return pd.DataFrame(out)
    n = eb["families"]["NSD"]["in_sample"]["base_rate"]["pooled"]["rows"]
    for fam in FAMILIES:
        t = deciles["families"][fam]["in_sample"]
        mine = [f"{t['table']['bins'][iv['decile']]['pooled']['lift']:.2f}" for iv in t["intervals"]]
        theirs = exploratory["deciles"].get(fam)
        out.append(_check(f"{fam}: in-sample decile lifts equal the exploratory Cell 15A output",
                          mine == theirs, f"{' '.join(mine)} | printed {' '.join(theirs or [])}"))
        bins = eb["families"][fam]["in_sample"]["bins"]
        mine_b = {b: {"lift": f"{v['pooled']['lift']:.2f}",
                      "share_pct": int(f"{100 * v['pooled']['support'] / n:.0f}")} for b, v in bins.items()}
        out.append(_check(f"{fam}: in-sample bin lifts and shares equal the exploratory Cell 15B output",
                          mine_b == exploratory["bins"].get(fam), mine_b))
    return pd.DataFrame(out)


def compute_lift_parity(side: Stage2Side, sources: Sources, deciles: Mapping[str, Any]) -> pd.DataFrame:
    """The deciles formed here against ``compute_lift`` (counts and lifts, exactly)."""
    from feature_research.model_training.rf.lift_analysis import compute_lift
    y = side.y_train.to_numpy()
    out = []
    for fam in FAMILIES:
        for kind in ("in_sample", "oof"):
            frame = pd.DataFrame({SOURCES[fam]: sources.values[kind][fam], "y": y})
            ref = compute_lift(frame, [SOURCES[fam]], "y", N_DECILES)[SOURCES[fam]]
            t = deciles["families"][fam][kind]
            mine = [(t["table"]["bins"][iv["decile"]]["pooled"]["support"],
                     t["table"]["bins"][iv["decile"]]["pooled"]["lift"]) for iv in t["intervals"]]
            theirs = list(zip(ref["count"].astype(int).tolist(), ref["lift"].astype(float).tolist()))
            same = len(mine) == len(theirs) and all(a[0] == b[0] and math.isclose(a[1], b[1], rel_tol=1e-12)
                                                    for a, b in zip(mine, theirs))
            out.append(_check(f"{fam} {kind}: deciles equal compute_lift's (counts, lifts)", same, len(mine)))
    return pd.DataFrame(out)


def historical_thresholds(side: Stage2Side, sources: Sources,
                          historical: Mapping[str, Any] = HISTORICAL) -> Dict[str, Any]:
    """
    The historical candidate thresholds (the nine printed decile edges of
    Cell 13C per family) applied, unchanged, to the training-side encodings:
    in sample and out of fold (pooled and per fold), with the historical
    counts and lifts beside them. Descriptive: nothing is selected.
    """
    from stage2_binning import evidence as E
    y = side.y_train.to_numpy()
    fid = np.asarray(side.fold_id)
    nf = len(side.folds)
    out: Dict[str, Any] = {"definition": ("historical intervals (lo, hi] with the printed Cell 13C "
                                          "decile edges (pandas' display precision; the exact "
                                          "historical edges were not recorded); first and last "
                                          "intervals open-ended"), "families": {}}
    flat = []
    for fam in FAMILIES:
        dec = historical["deciles"][fam]
        edges = [_num(d["hi"]) for d in dec[:-1]]
        names = [f"{fam.lower()}_h{i + 1:02d}" for i in range(len(dec))]
        tables = {}
        for kind in ("in_sample", "oof"):
            lab = assign_intervals(sources.values[kind][fam], edges, names)
            tables[kind] = _canonical(E.bin_table(lab, y, fid, names, nf, per_fold=kind == "oof"))
        out["families"][fam] = {"edges": edges, "names": names, **tables}
        frozen_of = assign_bins(np.asarray([(_num(d["lo"]) + _num(d["hi"])) / 2 for d in dec]), fam)
        for i, d in enumerate(dec):
            ins, oo = tables["in_sample"]["bins"][names[i]], tables["oof"]["bins"][names[i]]
            lifts = [c["lift"] for c in oo["folds"]]
            ok = [l for l in lifts if l is not None]
            flat.append({"family": fam, "interval": names[i],
                         "historical": f"({d['lo']}, {d['hi']}]", "frozen_bin": frozen_of[i],
                         "historical_rows": d["count"], "historical_lift": _num(d["lift"]),
                         "in_sample_rows": ins["pooled"]["support"], "in_sample_lift": ins["pooled"]["lift"],
                         "oof_rows": oo["pooled"]["support"], "oof_lift": oo["pooled"]["lift"],
                         "oof_lift_wilson95_lo": oo["pooled"]["lift_wilson95"][0],
                         "oof_lift_wilson95_hi": oo["pooled"]["lift_wilson95"][1],
                         "folds_with_rows": int(sum(c["support"] > 0 for c in oo["folds"])),
                         "fold_lift_min": min(ok) if ok else None, "fold_lift_max": max(ok) if ok else None})
    out["table"] = pd.DataFrame(flat)
    return out


# ======================================================================
# 5. contact periods
# ======================================================================
def _period_rates(T: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"period": T["period"].to_numpy(), "period_index": T["period_index"].to_numpy(),
                       "y": y})
    g = df.groupby(["period_index", "period"])
    out = pd.DataFrame({"rows": g.size(), "positives": g["y"].sum()}).reset_index()
    out["rate"] = out["positives"] / out["rows"]
    return out


#: Periods with fewer rows in a bin are left out of the within-period ratio range (display only).
PERIOD_MIN_ROWS = 100


def period_evidence(side: Stage2Side, sources: Sources, T: pd.DataFrame) -> Dict[str, Any]:
    """
    Per bin and contact period: rows, positives, rate, lift and the
    within-period ratio (bin rate / period rate), in sample and out of fold
    (with each fold's rows). Per bin (``standardized``):

    ``observed_over_expected``       observed / expected subscribers, expected =
                                     sum over periods of the bin's rows x the
                                     period's rate (all training rows). Near 1, the
                                     period mix reproduces the bin's aggregate
                                     subscriber count; within-period differences
                                     of opposite sign can cancel in it;
    ``abs_deviation_over_expected``  sum over periods of |observed - expected| /
                                     expected: the within-period differences
                                     without cancellation;
    ``within_period_ratio_min/max``  over the bin's periods with at least
                                     ``PERIOD_MIN_ROWS`` rows in it;
    ``shared_periods``               the bin's periods that also have rows in
                                     another bin of its family (same kind); with
                                     none, the bin is a union of whole periods and
                                     its ratio is 1 by construction.
    """
    from stage2_binning import evidence as E
    if not T.index.equals(side.X_train.index):
        raise ValueError("the time index is not aligned with the training rows")
    y = side.y_train.to_numpy()
    fid = np.asarray(side.fold_id)
    P = T["period"].to_numpy()
    rates = _period_rates(T, y)
    prate = dict(zip(rates["period"], rates["rate"]))
    exp_row = np.array([prate[p] for p in P])
    base = y.mean()
    comp, std = [], []
    for fam in FAMILIES:
        for kind in ("in_sample", "oof"):
            lab = sources.labels[kind][fam]
            names = E.family_bins(fam)
            bins_of_period = pd.Series(lab).groupby(P).nunique()
            for b in names:
                m = lab == b
                scopes = [("pooled", m)] + ([(f"fold_{k}", m & (fid == k)) for k in range(len(side.folds))]
                                            if kind == "oof" else [])
                for scope, mm in scopes:
                    obs, exp = int(y[mm].sum()), float(exp_row[mm].sum())
                    per = pd.DataFrame({"p": P[mm], "y": y[mm], "e": exp_row[mm]}).groupby("p").agg(
                        rows=("y", "size"), obs=("y", "sum"), exp=("e", "sum"))
                    ratios = per[per["rows"] >= PERIOD_MIN_ROWS]
                    ratios = ratios["obs"] / ratios["exp"]
                    std.append({"family": fam, "kind": kind, "bin": b, "rows": scope,
                                "support": int(mm.sum()), "observed_positives": obs,
                                "expected_positives": exp,
                                "observed_over_expected": obs / exp if exp else None,
                                "abs_deviation_over_expected":
                                    float((per["obs"] - per["exp"]).abs().sum() / exp) if exp else None,
                                "within_period_ratio_min": float(ratios.min()) if len(ratios) else None,
                                "within_period_ratio_max": float(ratios.max()) if len(ratios) else None,
                                "periods": int(len(per)),
                                "shared_periods": int((bins_of_period.reindex(per.index) > 1).sum())})
                for per_name in rates["period"]:
                    mp = m & (P == per_name)
                    if not mp.any():
                        continue
                    r = float(y[mp].mean())
                    row = {"family": fam, "kind": kind, "bin": b, "period": per_name, "rows": int(mp.sum()),
                           "positives": int(y[mp].sum()), "rate": r, "lift": r / base,
                           "period_rate": prate[per_name],
                           "within_period_ratio": r / prate[per_name] if prate[per_name] else None}
                    if kind == "oof":
                        row.update({f"fold_{k}_rows": int((mp & (fid == k)).sum()) for k in range(len(side.folds))})
                    comp.append(row)
    return {"period_rates": rates, "composition": pd.DataFrame(comp), "standardized": pd.DataFrame(std),
            "min_rows_for_ratio_range": PERIOD_MIN_ROWS, "definition": period_evidence.__doc__.strip()}


def straddling_states(states: MacroStates, svals: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Macro states whose frozen bin differs between the six training-side
    encoders (in sample, folds 0-4, each where the state is observed): their
    periods, rows, rate, the range of their encoded values and the cuts that
    range crosses. Per bin: the rows (in sample) of straddling states.
    """
    from shared.stage2.binning import BINNING_STRATEGY
    t = states.table.set_index("state")
    rows, by_bin = [], []
    for fam in FAMILIES:
        V = svals["values"][fam]
        cuts = [hi for _, _, hi in BINNING_STRATEGY[fam]["bins"][:-1]]
        bins = {enc: pd.Series(V[enc]).dropna().pipe(lambda s: pd.Series(assign_bins(s.to_numpy(), fam),
                                                                          index=s.index)) for enc in V}
        B = pd.DataFrame(bins)
        straddle = B.nunique(axis=1) > 1
        for s in B.index[straddle]:
            v = V.loc[s].dropna()
            crossed = [c for c in cuts if (v <= c).any() and (v > c).any()]
            rows.append({"family": fam, "state": int(s), "periods": t.loc[s, "periods"],
                         "rows": int(t.loc[s, "rows"]), "positives": int(t.loc[s, "positives"]),
                         "lift": float(t.loc[s, "lift"]), "value_min": float(v.min()),
                         "value_max": float(v.max()), "cuts_crossed": ", ".join(map(str, crossed)),
                         "bins": ", ".join(f"{enc}:{B.loc[s, enc]}" for enc in B.columns
                                           if pd.notna(B.loc[s, enc]))})
        ins = B["in_sample"]
        for b in [n for n, _, _ in BINNING_STRATEGY[fam]["bins"]]:
            in_b = ins.index[ins == b]
            st = [s for s in in_b if straddle.loc[s]]
            by_bin.append({"family": fam, "bin": b, "in_sample_states": int(len(in_b)),
                           "in_sample_rows": int(t.loc[in_b, "rows"].sum()),
                           "straddling_states": int(len(st)),
                           "straddling_rows": int(t.loc[st, "rows"].sum()) if st else 0})
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["family", "rows"], ascending=[True, False]).reset_index(drop=True)
    return {"states": table, "by_bin": pd.DataFrame(by_bin), "definition": straddling_states.__doc__.strip()}


def label_provenance_by_bin(X: pd.DataFrame, y: pd.Series, T: pd.DataFrame, side: Stage2Side,
                            sources: Sources) -> Dict[str, Any]:
    """
    Commit 3's attribution (``encoding_audit.audit_paths``) on the Stage 2 final
    fit and the five outer folds: each row's share of label mass (influence, for
    ECI) from later contact days, by frozen bin: out of fold (each fold's
    held-out rows) and in sample (the final fit's own rows). Provenance, not
    an effect on lift.
    """
    from feature_research import encoding_audit as EA
    from stage2_binning import evidence as E
    paths = EA.fit_paths(y)
    s2 = [p for p in paths["stage2"] if p.path in ("final", "outer")]
    outer = [p for p in s2 if p.path == "outer"]
    same_plan = all(np.array_equal(np.sort(p.query), np.sort(side.folds[p.fold][1]))
                    and np.array_equal(np.sort(p.fit), np.sort(side.folds[p.fold][0])) for p in outer)
    audit = EA.audit_paths(X, y, T, {"stage2": s2}, verbose=False)
    n = len(y)
    q = np.concatenate([p.query for p in outer])
    rows, overall = [], {}
    for fam in FAMILIES:
        feat = SOURCES[fam]
        oof = np.full(n, np.nan)
        oof[q] = audit["later"][(feat, "stage2", "outer", "held_out")]
        ins = np.asarray(audit["later"][(feat, "stage2", "final", "fit_rows")])
        overall[fam] = {"oof_mean": float(np.nanmean(oof)), "in_sample_mean": float(ins.mean())}
        for b in E.family_bins(fam):
            mo, mi = sources.labels["oof"][fam] == b, sources.labels["in_sample"][fam] == b
            rows.append({"family": fam, "bin": b, "oof_rows": int(mo.sum()),
                         "oof_later_share_mean": float(oof[mo].mean()),
                         "oof_majority_later_rows": int((oof[mo] > 0.5).sum()),
                         "in_sample_rows": int(mi.sum()),
                         "in_sample_later_share_mean": float(ins[mi].mean()),
                         "difference": float(oof[mo].mean() - ins[mi].mean())})
    checks = pd.DataFrame([
        _check("Commit 3's Stage 2 outer paths are this plan's folds", same_plan, len(outer)),
        _check("every training row has an out-of-fold later share", not np.isnan(
            np.r_[[overall[f]["oof_mean"] for f in FAMILIES]]).any() and len(q) == n and
               len(np.unique(q)) == n, len(q))])
    return {"table": pd.DataFrame(rows), "overall": overall, "checks": checks,
            "definition": label_provenance_by_bin.__doc__.strip()}


def focal_assignments(side: Stage2Side, sources: Sources, T: pd.DataFrame, states: MacroStates,
                      svals: Mapping[str, Any], deciles: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Row- and state-level facts the focal findings state, each checked on row
    masks or on the per-state, per-encoder assignments (``state_values``), not
    on matching counts or lifts.

    ``eci_hot``       the in-sample bin split by whether a row's contact period is
                      below the base rate; whether the below-base rows are exactly
                      the rows of a set of macro states; each such state's bin
                      under each of the six encoders; whether the out-of-fold
                      bin's below-base rows are the same rows; whether the
                      in-sample top decile is exactly the bin's other rows.
    ``nsd_elevated``  for the fold with the lowest out-of-fold lift: the states its
                      encoder assigns to the bin, their periods and in-sample bins,
                      and the bins it assigns to the in-sample bin's states; the
                      other encoders' bins for those states; the folds of every row
                      that enters or leaves the bin between the in-sample and the
                      out-of-fold encoders.
    """
    y = side.y_train.to_numpy()
    fid = np.asarray(side.fold_id)
    base = y.mean()
    code = states.code
    tab = states.table.set_index("state")
    rates = _period_rates(T, y)
    prate = dict(zip(rates["period"], rates["rate"]))
    below = np.array([prate[p] < base for p in T["period"].to_numpy()])
    encoders = list(svals["values"]["NSD"].columns)
    enc_bins = {fam: {enc: assign_bins(svals["values"][fam][enc].to_numpy(), fam) for enc in encoders}
                for fam in FAMILIES}

    def lift(mask):
        return float(y[mask].mean() / base) if mask.any() else None

    def state_rows(ids):
        return [{"state": int(s), "periods": tab.loc[s, "periods"], "rows": int(tab.loc[s, "rows"]),
                 "lift": float(tab.loc[s, "lift"])} for s in ids]

    # ---- eci_hot ----
    ins, oof = sources.labels["in_sample"]["ECI"], sources.labels["oof"]["ECI"]
    cluster, tail = (ins == "eci_hot") & below, (ins == "eci_hot") & ~below
    cl_states = np.unique(code[cluster])
    by_enc = {enc: sorted(set(enc_bins["ECI"][enc][cl_states].tolist())) for enc in encoders}
    oof_cluster = (oof == "eci_hot") & below
    top_name = deciles["families"]["ECI"]["in_sample"]["intervals"][-1]["decile"]
    top = deciles["labels"]["ECI"]["in_sample"] == top_name
    eci_hot = {
        "cluster_rows": int(cluster.sum()), "cluster_lift": lift(cluster),
        "cluster_periods": sorted(set(T["period"].to_numpy()[cluster])),
        "cluster_states": state_rows(cl_states),
        "cluster_is_whole_states": bool(np.array_equal(np.isin(code, cl_states), cluster)),
        "cluster_bins_by_encoder": by_enc,
        "cluster_in_bin_under_every_encoder": all(v == ["eci_hot"] for v in by_enc.values()),
        "oof_cluster_is_same_rows": bool(np.array_equal(oof_cluster, cluster)),
        "oof_cluster_rows_by_fold": {k: int((oof_cluster & (fid == k)).sum()) for k in range(len(side.folds))},
        "tail_rows": int(tail.sum()), "tail_lift": lift(tail),
        "oof_tail_rows": int(((oof == "eci_hot") & ~below).sum()),
        "oof_tail_lift": lift((oof == "eci_hot") & ~below),
        "top_decile": deciles["families"]["ECI"]["in_sample"]["intervals"][-1]["display"],
        "top_decile_is_tail": bool(np.array_equal(top, tail)),
    }

    # ---- nsd_elevated ----
    ins, oof = sources.labels["in_sample"]["NSD"], sources.labels["oof"]["NSD"]
    fold_lifts = []
    for k in range(len(side.folds)):
        m = (oof == "nsd_elevated") & (fid == k)
        fold_lifts.append(float(y[m].mean() / y[fid == k].mean()) if m.any() else None)
    worst = int(np.nanargmin([np.nan if v is None else v for v in fold_lifts]))
    enc = f"fold_{worst}"
    moved_in = np.flatnonzero(enc_bins["NSD"][enc] == "nsd_elevated")
    in_states = np.flatnonzero(enc_bins["NSD"]["in_sample"] == "nsd_elevated")
    moved = (ins != oof) & ((ins == "nsd_elevated") | (oof == "nsd_elevated"))
    period_rates_of = {int(s): [prate[p] for p in tab.loc[s, "periods"].split(",")] for s in moved_in}
    nsd = {
        "fold": worst, "fold_lifts": fold_lifts,
        "states_in_bin_under_fold": state_rows(moved_in),
        "those_states_all_below_base_periods": all(max(r) < base for r in period_rates_of.values()),
        "those_states_in_sample_bins": sorted(set(enc_bins["NSD"]["in_sample"][moved_in].tolist())),
        "those_states_bins_under_other_folds": {e: sorted(set(enc_bins["NSD"][e][moved_in].tolist()))
                                                for e in encoders if e not in ("in_sample", enc)},
        "in_sample_bin_states": int(len(in_states)),
        "in_sample_bin_states_bins_under_fold": sorted(set(enc_bins["NSD"][enc][in_states].tolist())),
        "moved_rows": int(moved.sum()), "moved_rows_folds": sorted(set(fid[moved].tolist())),
    }
    return {"eci_hot": eci_hot, "nsd_elevated": nsd, "definition": focal_assignments.__doc__.strip()}


# ======================================================================
# 6. findings
# ======================================================================
def _row(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise KeyError(f"{m.sum()} rows for {kw}")
    return df[m].iloc[0]


def _claim(claim: str, holds: bool, evidence: Mapping[str, Any]) -> Dict[str, Any]:
    return {"claim": claim, "holds": bool(holds), "evidence": _canonical(dict(evidence))}


def findings(*, eb: Mapping[str, Any], historical_check: Mapping[str, Any], implied: pd.DataFrame,
             encoder: Mapping[str, Any], deciles: Mapping[str, Any], historical_cuts: Mapping[str, Any],
             periods: Mapping[str, Any], straddling: Mapping[str, Any], provenance: Mapping[str, Any],
             assignments: Mapping[str, Any], historical: Mapping[str, Any] = HISTORICAL
             ) -> List[Dict[str, Any]]:
    """
    One finding per focal bin: the observation, the best-supported explanation,
    the causes the comparisons leave entangled, and each claim with the numbers
    it rests on and whether it holds in this run. Each claim says only what its
    check establishes: identities of rows or states are checked on row masks or
    on per-state, per-encoder assignments; the two encoder contrasts are
    described, not attributed to a cause; later-day shares are provenance and
    period-standardized ratios aggregates. Cell 38 requires every claim to
    hold, so the review document's statements are this run's.
    """
    fams = eb["families"]
    dec = encoder["decomposition"]
    pf = encoder["per_fold"]
    comp = periods["composition"]
    std = periods["standardized"]
    hc = historical_cuts["table"]
    prov = provenance["table"]
    sd = straddling["states"]
    transcription = historical_check["status"]
    arithmetic_ok = all(r["passed"] for r in historical_check["arithmetic"])
    look_note = ("provenance: similar mean shares can come from different contributing labels and give "
                 "different encoded values, so their effect on the lift gap is unresolved")

    def hist_decile(fam, i):
        d = historical["deciles"][fam][i - 1]
        return {"interval": f"({d['lo']}, {d['hi']}]", "rows": d["count"], "lift": _num(d["lift"])}

    def fold_lifts(fam, b):
        return [c["lift"] for c in fams[fam]["oof"]["bins"][b]["folds"]]

    def look_ahead_claim(fam, b):
        r = _row(prov, family=fam, bin=b)
        ev = {"oof_later_share_mean": r["oof_later_share_mean"],
              "in_sample_later_share_mean": r["in_sample_later_share_mean"], "reading": look_note}
        return _claim("Measured, as provenance: the bin's rows have mean later-day label shares (influence, for "
                      "ECI) that differ by less than 0.05 between in sample and out of fold; this does not show "
                      "that look-ahead leaves the lift gap unchanged",
                      abs(ev["oof_later_share_mean"] - ev["in_sample_later_share_mean"]) < 0.05, ev), ev

    def contrasts(d):
        return {"in_sample": d["in_sample_lift"], "fold_fit": d["fold_fit_lift"], "oof": d["oof_lift"],
                "fold_fit_minus_in_sample": d["fold_fit_minus_in_sample"],
                "heldout_minus_fold_fit": d["heldout_minus_fold_fit"],
                "oof_minus_in_sample": d["oof_minus_in_sample"]}

    def per_fold_diffs(fam, b):
        return [r["heldout_minus_fit"] for _, r in pf[(pf["family"] == fam) & (pf["bin"] == b)].iterrows()]

    out = []

    # ---------------- eci_warm ----------------
    d = _row(dec, family="ECI", bin="eci_warm")
    hb = historical["bins"]["ECI"]["eci_warm"]
    imp = _row(implied, family="ECI", cut=0.0758)
    base = fams["ECI"]["oof"]["base_rate"]["pooled"]["base_rate"]
    cw = comp[(comp["family"] == "ECI") & (comp["kind"] == "oof") & (comp["bin"] == "eci_warm")]
    st = _row(std, family="ECI", kind="oof", bin="eci_warm", rows="pooled")
    h = {i: _row(hc, family="ECI", interval=f"eci_h{i:02d}") for i in (7, 8, 9)}
    fl = fold_lifts("ECI", "eci_warm")
    la_claim, la = look_ahead_claim("ECI", "eci_warm")
    claims = [
        _claim("The quoted 1.49-1.67x are the lifts of two historical deciles inside the bin, "
               "(0.0777, 0.0797] and (0.0797, 0.0948]; the bin also spans (0.0758, 0.0777] at 0.35x, "
               "its largest decile, and at lock time the bin's own printed lift was 0.92x",
               arithmetic_ok and hb[2] == "0.92" and historical["deciles"]["ECI"][6]["lift"] == "0.35",
               {"transcription": transcription, "decile_7": hist_decile("ECI", 7),
                "decile_8": hist_decile("ECI", 8), "decile_9": hist_decile("ECI", 9),
                "bin_at_lock": {"rows": hb[0], "lift": _num(hb[2])},
                "rows_between_cut_0.0758_and_printed_edge": int(imp["rows_between"])}),
        _claim("Fold isolation leaves the bin near 1x: the in-sample and out-of-fold lifts differ by less than "
               "0.05, and so does each of the two descriptive contrasts (fold fit minus in sample; held out "
               "minus fold fit)",
               abs(d["oof_minus_in_sample"]) < 0.05 and abs(d["fold_fit_minus_in_sample"]) < 0.05
               and abs(d["heldout_minus_fold_fit"]) < 0.05,
               {**contrasts(d), "oof_wilson95": fams["ECI"]["oof"]["bins"]["eci_warm"]["pooled"]["lift_wilson95"],
                "fold_lifts": fl, "per_fold_heldout_minus_fit": per_fold_diffs("ECI", "eci_warm")}),
        _claim("The bin's rows come from contact periods far below and far above the base rate (period lifts "
               "below 0.5x and above 3x), and the period mix reproduces its aggregate subscriber count "
               "(observed / expected within 0.05 of 1); that ratio can hide within-period differences, "
               "which the absolute deviation and the within-period ratios show",
               cw["lift"].min() < 0.5 and cw["lift"].max() > 3 and abs(st["observed_over_expected"] - 1) < 0.05,
               {"periods": int(len(cw)), "period_lift_min": float(cw["lift"].min()),
                "period_lift_max": float(cw["lift"].max()),
                "observed_over_expected": st["observed_over_expected"],
                "abs_deviation_over_expected": st["abs_deviation_over_expected"],
                "within_period_ratio_range": [st["within_period_ratio_min"], st["within_period_ratio_max"]],
                "shared_periods": int(st["shared_periods"]),
                "largest_periods": cw.nlargest(4, "rows")[["period", "rows", "lift"]].to_dict("records")}),
        _claim("On the training-side encodings the historical sub-intervals do not keep their lifts: "
               "out of fold, (0.0777, 0.0797] is below 1x and the three intervals range widely across folds",
               h[8]["oof_lift"] < 1 and max(h[i]["fold_lift_max"] - h[i]["fold_lift_min"] for i in (7, 8, 9)) > 0.5,
               {f"eci_h{i:02d}": {"historical_lift": h[i]["historical_lift"], "in_sample": h[i]["in_sample_lift"],
                                  "oof": h[i]["oof_lift"], "fold_min": h[i]["fold_lift_min"],
                                  "fold_max": h[i]["fold_lift_max"]} for i in (7, 8, 9)}),
        la_claim,
    ]
    out.append({
        "bin": "eci_warm",
        "observation": "historical 1.49-1.67x; PR 39 out of fold about 1.03x",
        "explanation": (
            "Threshold placement and estimand, not drift in the bin. The quoted figures are two historical deciles, "
            "(0.0777, 0.0797] at 1.67x and (0.0797, 0.0948] at 1.49x. The frozen bin starts at 0.0758 and also holds "
            "the (0.0758, 0.0777] decile at 0.35x, its largest, plus "
            f"{int(imp['rows_between']):,} rows of the decile below (printed edges are rounded). Its own lift when "
            f"the cut points were locked was {hb[2]}x, on the full population with an in-sample encoder. On the "
            f"training rows it is {d['in_sample_lift']:.3f}x in sample and {d['oof_lift']:.3f}x out of fold (folds "
            f"{min(fl):.2f}-{max(fl):.2f}); both descriptive contrasts are below 0.01 in magnitude. ECI is a "
            f"curvature of rate curves, not a rate: the bin collects macro states of very different periods ({len(cw)} "
            f"out of fold, lifts {cw['lift'].min():.2f}-{cw['lift'].max():.2f}x), and that period mix reproduces its "
            f"aggregate subscriber count (observed / expected {st['observed_over_expected']:.2f}; absolute "
            f"within-period deviations {st['abs_deviation_over_expected']:.1%} of expected)."),
        "entangled": ("historical 0.92x versus training 1.04x in sample: full population versus training rows, "
                      "fit size and reference draw, not separable without the historical fit (it used labels of "
                      "rows now in the test split); whether look-ahead contributes to either lift is unresolved "
                      "(no past-only ECI is built)"),
        "claims": claims})

    # ---------------- eci_hot ----------------
    d = _row(dec, family="ECI", bin="eci_hot")
    hb = historical["bins"]["ECI"]["eci_hot"]
    a = assignments["eci_hot"]
    fl = fold_lifts("ECI", "eci_hot")
    la_claim, la = look_ahead_claim("ECI", "eci_hot")
    claims = [
        _claim("The quoted 3.53x is the historical top decile, which is the bin itself: both are upper intervals "
               "of the same values, the cut 0.0948 is the decile's printed lower edge with no rows between, and "
               "both hold 3,178 rows",
               arithmetic_ok and hb[0] == historical["deciles"]["ECI"][9]["count"] and
               int(_row(implied, family="ECI", cut=0.0948)["rows_between"]) == 0,
               {"transcription": transcription, "decile_10": hist_decile("ECI", 10),
                "bin_at_lock": {"rows": hb[0], "lift": _num(hb[2])}}),
        _claim("The attenuation is already present in sample on the training rows; the out-of-fold lift is not "
               "lower than the in-sample lift by more than 0.05",
               d["in_sample_lift"] < 3 and d["oof_minus_in_sample"] > -0.05,
               {**contrasts(d), "oof_wilson95": fams["ECI"]["oof"]["bins"]["eci_hot"]["pooled"]["lift_wilson95"]}),
        _claim("In sample, the bin's rows in periods at or above the base rate are exactly the in-sample top "
               "decile (row masks equal), at above 3.2x; its rows in below-base periods are exactly all the rows "
               "of a set of macro states, at under 1x",
               a["top_decile_is_tail"] and a["cluster_is_whole_states"] and a["tail_lift"] > 3.2
               and a["cluster_lift"] < 1,
               {k: a[k] for k in ("tail_rows", "tail_lift", "top_decile", "top_decile_is_tail", "cluster_rows",
                                  "cluster_lift", "cluster_periods", "cluster_is_whole_states")}
               | {"cluster_states": len(a["cluster_states"])}),
        _claim("Every one of the six training-side encoders assigns each of those states to eci_hot, so the "
               "out-of-fold bin's rows in below-base periods are exactly the same rows, present in every fold; "
               "the out-of-fold fold lifts stay within 2.3-2.8x",
               a["cluster_in_bin_under_every_encoder"] and a["oof_cluster_is_same_rows"]
               and all(v > 0 for v in a["oof_cluster_rows_by_fold"].values()) and min(fl) > 2.3 and max(fl) < 2.8,
               {"cluster_bins_by_encoder": a["cluster_bins_by_encoder"],
                "oof_cluster_rows_by_fold": a["oof_cluster_rows_by_fold"], "fold_lifts": fl,
                "oof_tail": {"rows": a["oof_tail_rows"], "lift": a["oof_tail_lift"]}}),
        la_claim,
    ]
    out.append({
        "bin": "eci_hot",
        "observation": "historical 3.53x; PR 39 out of fold about 2.50x",
        "explanation": (
            f"The drop lies between the historical full-population fit and the training-side fits, not in fold "
            f"isolation. Historically the bin was the top decile ({hb[0]:,} rows, {hb[2]}x). On the training rows it "
            f"is already {d['in_sample_lift']:.3f}x in sample and {d['oof_lift']:.3f}x out of fold. In sample it "
            f"holds {a['cluster_rows']:,} rows of below-base periods ({', '.join(a['cluster_periods'])}), exactly the "
            f"rows of {len(a['cluster_states'])} macro states, at {a['cluster_lift']:.2f}x. Its other "
            f"{a['tail_rows']:,} rows are exactly the in-sample top decile, {a['top_decile']}, at "
            f"{a['tail_lift']:.2f}x ({a['oof_tail_lift']:.2f}x out of fold). Every training-side encoder assigns "
            f"those states to the bin, so the same rows dilute it in every fold, and the fold lifts are tight "
            f"({min(fl):.2f}-{max(fl):.2f}x). The upper tail retains its lift; the bin includes a low-rate cluster "
            f"just above 0.0948."),
        "entangled": ("whether the May 2009 rows fell below 0.0948 in the historical fit remains open: that fit used "
                      "labels of rows now in the test split, and its population, size and reference draw cannot be "
                      "rebuilt on the training side; whether look-ahead contributes to either lift is unresolved"),
        "claims": claims})

    # ---------------- nsd_elevated ----------------
    d = _row(dec, family="NSD", bin="nsd_elevated")
    hb = historical["bins"]["NSD"]["nsd_elevated"]
    base = fams["NSD"]["oof"]["base_rate"]["pooled"]["base_rate"]
    a = assignments["nsd_elevated"]
    worst = a["fold"]
    fl = fold_lifts("NSD", "nsd_elevated")
    e = fams["NSD"]["oof"]["bins"]["nsd_elevated"]
    pooled_wo = ((e["pooled"]["positives"] - e["folds"][worst]["positives"])
                 / (e["pooled"]["support"] - e["folds"][worst]["support"]) / base)
    nsd_st = sd[sd["family"] == "NSD"] if len(sd) else sd
    cross192 = nsd_st[nsd_st["cuts_crossed"].str.contains("0.192")] if len(nsd_st) else nsd_st
    wf = _row(pf, family="NSD", bin="nsd_elevated", fold=worst)
    others = pf[(pf["family"] == "NSD") & (pf["bin"] == "nsd_elevated") & (pf["fold"] != worst)]
    diffs = per_fold_diffs("NSD", "nsd_elevated")
    ce = comp[(comp["family"] == "NSD") & (comp["kind"] == "oof") & (comp["bin"] == "nsd_elevated")]
    wrows = ce[ce[f"fold_{worst}_rows"] > 0]
    other_periods = ce[ce[[f"fold_{k}_rows" for k in range(N_FOLDS) if k != worst]].sum(axis=1) > 0]
    la_claim, la = look_ahead_claim("NSD", "nsd_elevated")
    moved_states = a["states_in_bin_under_fold"]
    claims = [
        _claim("The fold with the lowest out-of-fold lift is the fold whose encoder moved macro states across "
               "both cuts: it assigns nsd_elevated only to states of below-base periods, which are nsd_warm in "
               "sample and under every other fold's encoder, and it assigns nsd_hot to every state of the "
               "in-sample bin",
               fl[worst] < 1 and len(moved_states) > 0 and a["those_states_all_below_base_periods"]
               and a["those_states_in_sample_bins"] == ["nsd_warm"]
               and all(v == ["nsd_warm"] for v in a["those_states_bins_under_other_folds"].values())
               and a["in_sample_bin_states_bins_under_fold"] == ["nsd_hot"],
               {"fold": worst, "fold_lifts": fl, "states_in_bin_under_fold": moved_states,
                "those_states_in_sample_bins": a["those_states_in_sample_bins"],
                "those_states_bins_under_other_folds": a["those_states_bins_under_other_folds"],
                "in_sample_bin_states": a["in_sample_bin_states"],
                "in_sample_bin_states_bins_under_fold": a["in_sample_bin_states_bins_under_fold"]}),
        _claim("Every row that enters or leaves nsd_elevated between the in-sample and the out-of-fold encoders "
               "is a row of that fold", a["moved_rows"] > 0 and a["moved_rows_folds"] == [worst],
               {"moved_rows": a["moved_rows"], "folds": a["moved_rows_folds"]}),
        _claim("Under that fold's own encoder the bin's fit rows are also below 1.5x, as its held-out rows are; the "
               "other folds' encoders give above 2.5x on both their fit and their held-out rows",
               (wf["fit_lift"] or 0) < 1.5 and fl[worst] < 1 and (others["fit_lift"] > 2.5).all()
               and (others["heldout_lift"] > 2.5).all(),
               {"fold_fit_lift": wf["fit_lift"], "fold_heldout_lift": wf["heldout_lift"],
                "other_folds_fit_lift": others["fit_lift"].tolist(),
                "other_folds_heldout_lift": others["heldout_lift"].tolist()}),
        _claim("Without that fold the pooled out-of-fold lift is above 3x",
               pooled_wo > 3, {"pooled": e["pooled"]["lift"], "without_fold": pooled_wo,
                               "fold_share_of_rows": e["folds"][worst]["support"] / e["pooled"]["support"]}),
        _claim("The pooled held-out-minus-fit difference is a pool of per-fold differences of both signs, so it "
               "does not measure the effect of a row's own label",
               min(diffs) < 0 < max(diffs),
               {**contrasts(d), "per_fold_heldout_minus_fit": diffs}),
        _claim("States straddle the 0.192 cut across the six training-side encoders",
               len(cross192) > 0,
               {"states": cross192.head(8)[["state", "periods", "rows", "lift", "value_min", "value_max", "bins"]]
                .to_dict("records") if len(cross192) else []}),
        la_claim,
    ]
    out.append({
        "bin": "nsd_elevated",
        "observation": f"PR 39 pooled out of fold about {e['pooled']['lift']:.2f}x, folds {min(fl):.2f}-{max(fl):.2f}x",
        "explanation": (
            f"Encoder behaviour at cuts that sit inside the ranges whole macro states take across fits. NSD and ECI "
            f"are functions of the macro state (euribor3m, nr.employed, emp.var.rate), so all rows of a state cross a cut "
            f"together. Fold {worst}'s encoder assigns nsd_elevated to "
            f"{len(moved_states)} states of {', '.join(wrows['period'])} (prevalence {wrows['period_rate'].min():.3f}), "
            f"which are nsd_warm in sample and under every other fold's encoder, and assigns nsd_hot to all {a['in_sample_bin_states']} states of the in-sample "
            f"bin ({', '.join(other_periods['period'])}). Every row that enters or leaves the bin between the in-sample "
            f"and out-of-fold encoders is a fold-{worst} row. Its {e['folds'][worst]['support']:,} held-out bin rows "
            f"have {fl[worst]:.2f}x, and that encoder's own fit rows in the bin {wf['fit_lift']:.2f}x; the other folds "
            f"have {min(l for k, l in enumerate(fl) if k != worst):.2f}-"
            f"{max(l for k, l in enumerate(fl) if k != worst):.2f}x, and without fold {worst} the pooled lift is "
            f"{pooled_wo:.2f}x. The fold encoders' pooled lift on their own fit rows ({d['fold_fit_lift']:.3f}x) is "
            f"close to the out-of-fold {d['oof_lift']:.3f}x, against {d['in_sample_lift']:.3f}x in sample; the "
            f"held-out-minus-fit contrast ({d['heldout_minus_fold_fit']:+.3f}) pools per-fold differences from "
            f"{min(diffs):+.2f} to {max(diffs):+.2f} and does not isolate a row's own label. Historically the figures "
            f"1.04-1.43x were two deciles (the 1.04x one lies in nsd_warm), and the bin's own lift at lock time was "
            f"{hb[2]}x on {hb[0] / historical['population']['rows']:.1%} of rows, against "
            f"{d['in_sample_support'] / fams['NSD']['in_sample']['base_rate']['pooled']['rows']:.1%} in sample on the "
            f"training rows: the same cut selected a different set of states."),
        "entangled": ("which part of an engineer's fit moves the states (its fit rows' labels, including a row's own, "
                      "or its 5,000-row reference draw) is not separated: the six encoders differ in both. The "
                      "historical bin's contents are not recoverable without the historical fit; whether look-ahead "
                      "contributes to either lift is unresolved"),
        "claims": claims})
    return out


def claims_table(found: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Every claim of every finding, flat, for ``require``."""
    return pd.DataFrame([{"bin": f["bin"], "claim": c["claim"], "holds": c["holds"]}
                         for f in found for c in f["claims"]])


#: What this step's evidence does not cover (recorded with every run).
LIMITS = (
    "The historical figures were computed on all 41,188 rows (the rows now in the test split among "
    "them) with an encoder fitted on those rows. They are transcribed and Git-verified, never "
    "recomputed (G7), so the historical-to-training difference combines population, fit size and "
    "reference draw (legacy np.random against np.random.default_rng), which the training side cannot "
    "separate.",
    "Every comparison here uses random folds. All encoders, in sample and out of fold, use labels of "
    "later contacts (Commit 3: about 47% of the NSD and 52% of the ECI label mass or influence). The mean "
    "shares are similar in sample and out of fold, but similar means can come from different "
    "contributing labels and give different encoded values. A past-only NSD or ECI is not built (Commit 3, "
    "Step 3B section 5), so the effect of look-ahead on these lifts, and on the gaps between them, is "
    "unresolved.",
    "The two encoder contrasts are descriptive. fold_fit_minus_in_sample changes the encoders and the "
    "weighting of rows at once; heldout_minus_fold_fit compares the same encoders on different rows, so "
    "fold composition and the cancellation of per-fold differences affect it. Neither isolates a row's "
    "own label.",
    "Period-standardized ratios near 1 show that period mix reproduces a bin's aggregate subscriber "
    "count; within-period differences can cancel in them (the absolute-deviation share and the "
    "within-period ratios are reported beside them).",
    "The six training-side encoders differ in their fit rows' labels and in their 5,000-row KDE "
    "reference draw at once; which of the two moves the NSD states across a cut is not separated.",
    "The frozen cut points are constants chosen with all labels (PR 39 limitation). No threshold is "
    "evaluated as a candidate for adoption here: the historical edges are described, not compared "
    "under registered criteria. Fold-fitted procedures, registered criteria and candidate "
    "comparisons are Step 4B.",
    "Wilson intervals cover binomial sampling of the evaluated rows only (WILSON_ESTIMAND); "
    "fold-to-fold encoder variation is shown by the per-fold tables, not by the intervals.",
    "Fold-fit tables pool (fold, fit row) pairs: each training row counts four times.",
)


def record_sections(*, pr39_provenance, historical_check, implied, exploratory, figures, reproduction,
                    sources, flips, states, state_vals, encoder, support, stability, deciles,
                    decile_checks, historical_cuts, periods, straddling, provenance, assignments,
                    found) -> Dict[str, Any]:
    """The ``binning_drift`` record's content sections (identity keys are added by the writer)."""
    def recs(df):
        return df.to_dict("records")

    return {
        "scope": ("PR 42 Commit 4 Step 4A: Stage 2 NSD/ECI binning drift, reproduction and diagnosis "
                  "(training side; no candidate, procedure or selection)"),
        "question": QUESTION,
        "causes": dict(CAUSES),
        "settings": {"seed": SEED, "n_folds": N_FOLDS, "focal": dict(FOCAL), "sources": dict(SOURCES),
                     "macro_columns": list(MACRO_COLUMNS), "state_tolerance": STATE_TOLERANCE,
                     "n_deciles": N_DECILES},
        "provenance": {"pr39_record": dict(pr39_provenance),
                       "historical": {"transcription": _canonical(dict(HISTORICAL)),
                                      "verification": historical_check},
                       "implied_cut_rows": recs(implied),
                       "exploratory": exploratory,
                       "figures": recs(figures)},
        "reproduction": {"checks": recs(reproduction["checks"]),
                         "eb_sha256": hashlib.sha256(json.dumps(reproduction["eb"], sort_keys=True,
                                                                separators=(",", ":")).encode()).hexdigest(),
                         "triggers": reproduction["triggers"], "decision": reproduction["decision"]},
        "sources": {"checks": recs(sources.checks), "fold_fits": sources.provenance},
        "isolation": {"heldout_label_flips": recs(flips)},
        "macro_states": {"states": recs(states.table), "checks": recs(state_vals["checks"]),
                         "values": {fam: {enc: state_vals["values"][fam][enc].tolist()
                                          for enc in state_vals["values"][fam]} for fam in FAMILIES}},
        "encoder_comparison": {"definition": encoder["definition"],
                               "decomposition": recs(encoder["decomposition"]),
                               "per_fold": recs(encoder["per_fold"]),
                               "migration": recs(encoder["migration"]),
                               "fold_fit_tables": encoder["families"]},
        "support": {"estimand": WILSON_ESTIMAND, "table": recs(support), "fold_stability": recs(stability)},
        "deciles": {"definition": deciles["definition"], "families": deciles["families"],
                    "checks": recs(decile_checks)},
        "historical_thresholds": {"definition": historical_cuts["definition"],
                                  "table": recs(historical_cuts["table"]),
                                  "families": historical_cuts["families"]},
        "periods": {"definition": periods["definition"], "min_rows_for_ratio_range": periods["min_rows_for_ratio_range"],
                    "period_rates": recs(periods["period_rates"]),
                    "composition": recs(periods["composition"]),
                    "standardized": recs(periods["standardized"])},
        "straddling": {"definition": straddling["definition"], "states": recs(straddling["states"]),
                       "by_bin": recs(straddling["by_bin"])},
        "label_provenance": {"definition": provenance["definition"], "table": recs(provenance["table"]),
                             "overall": provenance["overall"], "checks": recs(provenance["checks"])},
        "focal_assignments": assignments,
        "findings": list(found),
        "claims": recs(claims_table(found)),
        "limits": list(LIMITS),
    }
