"""
stage2_crossfit.complementarity
===============================
Commit 3, Step 3B: the comparative analysis of an ACCEPTED corrected primary
(protocol sections 3.2-3.3 and 5-12, ``acceptance.comparative_sections``).
Nothing here runs before ``acceptance.require_accepted`` has re-verified the
run's official structural acceptance in full, and nothing here fits, tunes,
selects, writes into a model folder or reads a test label.

    load_inputs     the gate first (``require_accepted``); then the train side
                    and the candidate's outer-OOF views and trace (population
                    identity re-checked), the fold ledger re-derived from the
                    candidate's evidence, the row / fold alignment of that
                    evidence with the trace, the pre-correction baseline's own
                    historical sources (read from the archive, rebuilt and
                    compared with the frozen package: protocol section 8) and
                    the frozen upstream reference (``verify_reference``,
                    complete)
    analyse         -> Analysis: every table, in memory, through the validated
                    protocol and Step 2A-2D APIs (the baseline's own producers,
                    so pre and post stay comparable); nothing is re-implemented
    build_package   -> {relative path: bytes}: deterministic CSV and JSON
    write_package   write once; an identical rerun writes nothing; a differing
                    package is refused (ComplementarityConflict)
    run             all of the above; ``check=True`` rebuilds and compares only

Four layers, never merged (MANIFEST.json ``layers``):

    1 implementation correctness   trace checks T1-T7, three-state accounting,
                                   row / fold alignment, routing-policy claims
                                   with their evidence status
    2 fit-time feasibility and     the acceptance record, the fold ledger,
      acceptance                   every selection's fit-time obligations
    3 observed outer-OOF           section 11: per fold and pooled, kept apart
      compliance                   from the fit status
    4 empirical performance and    sections 3.2-3.3, 6-9 and 12; statements
      complementarity              only by the D-C1 rule, no overall verdict

The package goes to ``research_logs/pr38/<run_id>/complementarity_v1/``:
``MANIFEST.json`` (identities, inputs, bootstrap provenance, layers, sections,
disclosures, outputs, code identity), its ``.sha256`` sidecar and
``tables/*.csv`` (``tables/diagnostics/`` holds the baseline's Step 2A-2D
tables recomputed for the corrected routers). No timestamps or local paths.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import shared.stage2.evaluation_protocol as ep
from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from shared.stage2.diagnostics.contribution import MATCH_SCORES
from shared.stage2.diagnostics.trace import TRACE_ARMS
from shared.stage_io import label_fingerprint

from . import acceptance as A
from . import evidence as E
from . import inner_features as IF
from . import phase0
from . import runner as R

__all__ = ["FORMAT", "PACKAGE_DIR", "EVIDENCE_SOURCE", "BASELINE_SOURCE", "PRE_POST_METRICS",
           "ComplementarityError", "ComplementarityConflict", "Inputs", "Analysis",
           "fold_ledger", "check_alignment", "historical_sources", "diagnostics_tables",
           "load_inputs", "analyse", "build_package", "compare_package", "write_package", "run"]

FORMAT = "pr38.stage2.complementarity/1"
PACKAGE_DIR = "complementarity_v1"
EVIDENCE_SOURCE = "pr38_corrected_outer_oof"
BASELINE_SOURCE = b.BASELINE_ID
MANIFEST, SIDECAR = "MANIFEST.json", "MANIFEST.json.sha256"
#: Section 8: the pre/post changes reported with registered intervals (difference = pre - post).
PRE_POST_METRICS = ("pass1_rate", "pass2_rate", "coverage", "abstain_rate", "pass1_npv",
                    "pass1_leakage_share", "pass2_precision", "pass2_recall",
                    "pass2_recall_eligible", "unique_tp", "pass1_fp_suppression")
#: Names of what this analysis reads (ep.check_inputs: defense in depth only).
INPUT_NAMES = ("corrected_outer_oof_routes", "training_labels", "outer_fold_ids",
               "frozen_upstream_oof_flags", "frozen_upstream_oof_scores",
               "pre_correction_outer_oof_routes", "fit_time_selection_records",
               "fit_time_fold_evidence", "pre_correction_baseline_tables")
#: Baseline tables that need the GLASS-arm RF result or the source records (not recomputed).
_NOT_RECOMPUTED = ("validation",)
_FROZEN_TABLES = ("states", "ratios")
CODE_FILES = (("stage2_crossfit/complementarity.py", "stage2_crossfit/acceptance.py",
               "stage2_crossfit/evidence.py", "stage2_crossfit/runner.py",
               "shared/stage2/evaluation_protocol.py", "shared/stage2/obligations.py",
               "tests/support/archived_artifacts.py") + tuple(b.CODE_FILES))


class ComplementarityError(R.RunnerError):
    """The analysis cannot run on these inputs; nothing was written."""


class ComplementarityConflict(ComplementarityError):
    """A package already exists and differs; nothing was written."""


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Inputs:
    """Everything ``analyse`` reads, each piece verified by ``load_inputs``."""

    run_id: str
    role: str
    protocol: Any
    contract: Any
    ref: Any
    trace: Any
    lr: Any
    ebm: Any
    reference: ep.ReferenceFlags
    historical: Mapping[str, Any]
    baseline_tables: Mapping[str, pd.DataFrame]
    ledger: ep.FoldLedger
    evidence: Mapping[str, Any]
    alignment: pd.DataFrame
    identities: Mapping[str, Any]
    record_dir: Optional[Path] = None
    gate_hashes: Mapping[str, Tuple[str, str]] = dataclasses.field(default_factory=dict)


def fold_ledger(evidence: Mapping[str, Any], settings: R.ArmSettings) -> ep.FoldLedger:
    """The ledger re-derived from the candidate's evidence, as acceptance re-derives it."""
    folds = evidence.get("folds") or {}
    try:
        return ep.FoldLedger.build(
            {arm: {int(ev["fold"]): (E.classify_fold(ev, settings.contract, settings.protocol)
                                     or ev["fit_status"]["status"])
                   for ev in folds.get(arm) or []} for arm in TRACE_ARMS},
            settings.protocol).require_complete(list(TRACE_ARMS))
    except (ep.ProtocolError, KeyError, TypeError) as exc:
        raise ComplementarityError(f"the candidate's evidence does not give a complete fold "
                                   f"ledger: {str(exc).splitlines()[0]}") from exc


def check_alignment(trace, evidence: Mapping[str, Any], attempted: Sequence[int]) -> pd.DataFrame:
    """
    Row and fold alignment of the fit-time evidence with the outer-OOF trace:
    the trace's folds are the attempted folds; each fold router of each arm was
    fitted on exactly the trace's rows outside its outer fold (count, positives
    and label fingerprint) and its remainder lies inside them. Refuses on any
    failure; returns every check.
    """
    idx, y, fold = trace.index, np.asarray(trace.y).astype(int), np.asarray(trace.fold_id)
    rows: List[Dict[str, Any]] = []

    def add(check, arm, k, ok, detail):
        rows.append({"check": check, "arm": arm, "fold": k, "passed": bool(ok),
                     "detail": str(detail)})

    folds = sorted(int(k) for k in np.unique(fold))
    add("trace folds are the attempted folds", "", "all", folds == sorted(attempted),
        f"{folds} vs {list(attempted)}")
    for arm in TRACE_ARMS:
        evs = {int(ev["fold"]): ev for ev in (evidence.get("folds") or {}).get(arm) or []}
        add("one evidence record per attempted fold", arm, "all",
            sorted(evs) == sorted(attempted), sorted(evs))
        for k in attempted:
            ev = evs.get(int(k)) or {}
            fit = fold != int(k)
            fp = label_fingerprint(idx[fit], y[fit])
            add("fitted on the trace's rows outside its outer fold", arm, int(k),
                int(ev.get("n_train", -1)) == int(fit.sum())
                and int(ev.get("train_positives", -1)) == int(y[fit].sum())
                and ev.get("train_label_fingerprint") == fp,
                f"n_train {ev.get('n_train')} vs {int(fit.sum())}; fingerprint "
                f"{ev.get('train_label_fingerprint')} vs {fp}")
            rem = pd.Index((ev.get("remainder") or {}).get("index") or [])
            add("remainder rows inside the fold's fitting rows", arm, int(k),
                len(rem) == int((ev.get("remainder") or {}).get("rows", -1))
                and bool(rem.isin(idx[fit]).all()),
                f"{len(rem)} remainder rows")
    frame = pd.DataFrame(rows, columns=["check", "arm", "fold", "passed", "detail"])
    if not frame["passed"].all():
        bad = frame[~frame["passed"]]
        raise ComplementarityError("the fit-time evidence is not aligned with the outer-OOF "
                                   "trace:\n  - " + "\n  - ".join(
                                       f"{r.arm} fold {r.fold}: {r.check} ({r.detail})"
                                       for r in bad.itertuples()))
    return frame


def historical_sources(sums_path):
    """
    The pre-correction baseline's own sources (protocol section 8), exactly as
    ``stage2_baseline_archive.py --check`` reads them: the arbiters the frozen
    MANIFEST.json records, every archived file checked against SHA256SUMS
    before its first load (``tests.support.archived_artifacts``), the package
    rebuilt in memory, the archive re-hashed, and the rebuild compared with
    the frozen package, which must reproduce. Returns the sources and the
    comparison.
    """
    from tests.support import archived_artifacts as aa
    frozen = b.read_manifest(phase0.BASELINE_DIR)
    real = aa.HistoricalReal(sums_path=sums_path, arbiters=aa.baseline_arbiters(frozen))
    sources = aa.collect_archived_baseline_sources(real)
    package = b.build_baseline(sources, b.code_identity(phase0.BANK_ROOT),
                               baseline_id=str(frozen.get("baseline_id")))
    changed = aa.archived_rehash(real.archive)(sources)
    if changed:
        raise ComplementarityError("archived sources changed during the rebuild:\n  - "
                                   + "\n  - ".join(changed))
    report = b.compare_baseline(package, phase0.BASELINE_DIR)
    if not report.reproduced:
        raise ComplementarityError("the pre-correction baseline does not reproduce from the "
                                   "archive:\n  - " + "\n  - ".join(report.lines()))
    return sources, report


def _frozen_tables(baseline_dir: Path, protocol) -> Dict[str, pd.DataFrame]:
    """The frozen package's tables the pre/post joins read, as stored text."""
    rep = b.verify_baseline(baseline_dir, expected_id=b.BASELINE_ID)
    if not rep.ok:
        raise ComplementarityError("the frozen baseline package fails verification: "
                                   + "; ".join(rep.problems))
    if rep.manifest_sha256 != protocol["population"]["baseline_manifest_sha256"]:
        raise ComplementarityError(f"baseline MANIFEST.json {rep.manifest_sha256} is not the "
                                   "protocol's pinned baseline manifest")
    return {name: _read_text_csv((Path(baseline_dir) / b.TABLES[name][0]).read_bytes())
            for name in _FROZEN_TABLES}


def _read_text_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)


def _artifact(sources, role: str) -> Dict[str, Any]:
    found = [a for a in sources.artifacts if a.role == role]
    if len(found) != 1:
        raise ComplementarityError(f"the historical sources name {len(found)} {role} files")
    return {"role": role, "path": found[0].path, "sha256": found[0].sha256}


def load_inputs(run_id: str, sums_path, *, role: str = R.ROLE_PRIMARY,
                settings: Optional[R.ArmSettings] = None, output_root=None, record_dir=None,
                baseline_dir=None, train_side: Optional[Callable[[], R.TrainSide]] = None,
                historical: Optional[Callable[[Any], Tuple[Any, Any]]] = None) -> Inputs:
    """
    Every input of ``analyse``, verified. ``require_accepted`` runs first:
    without an official, fully re-verified structural acceptance nothing else
    is read (no train side, no artifact, no baseline source).
    ``train_side`` / ``historical`` replace ``runner.load_train_side`` /
    ``historical_sources`` (tests).
    """
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    acc_path = record_dir / A.ACCEPTANCE_FILE
    if settings is None and acc_path.is_file():
        try:                                          # unreadable: require_accepted refuses
            pathway = (json.loads(acc_path.read_text(encoding="utf-8")).get("feature_pathway")
                       or {}).get("pathway")
        except (ValueError, AttributeError):
            pathway = None
        settings = R.primary_settings(pathway) if pathway in IF.PATHWAYS else None
    candidate = A.require_accepted(run_id, role=role, output_root=output_root,
                                   record_dir=record_dir, settings=settings)
    # --- accepted: from here on the run's artifacts may be read --------------------
    acc_sha = R.sha256_file(acc_path)
    acc = json.loads(acc_path.read_text(encoding="utf-8"))
    if settings is None:                              # the record appeared after the check
        settings = R.primary_settings(acc["feature_pathway"]["pathway"])
    protocol = settings.protocol
    ep.check_inputs(INPUT_NAMES, protocol)
    ts = (train_side or R.load_train_side)()
    pop = R.check_population(ts, protocol)
    if not pop["passed"]:
        raise ComplementarityError("the train side does not match the frozen protocol's "
                                   "population")
    ref = A.train_reference(ts, protocol)
    views = A.candidate_views(candidate, ref)
    trace = d.build_router_trace(views[E.ARM_GLASS], views[E.ARM_RF], ref)
    population = ep.check_population_identity(trace, ep.PopulationIdentity.from_protocol(
        protocol)).require(complete=True)
    ledger = fold_ledger(candidate.evidence, settings)
    alignment = check_alignment(trace, candidate.evidence, ledger.attempted)

    sources, comparison = (historical or historical_sources)(sums_path)
    upstream = {k: _artifact(sources, v["role"]) for k, v in
                protocol["upstream"]["artifacts"].items()}
    for k, art in upstream.items():
        want = protocol["upstream"]["artifacts"][k]
        if (art["path"], art["sha256"]) != (want["path"], want["sha256"]):
            raise ComplementarityError(f"the baseline's {art['role']} is not the protocol's "
                                       f"frozen upstream artifact ({art} vs {dict(want)})")
    verified = ep.verify_reference(ep.UpstreamSources.from_views(sources.lr, sources.ebm),
                                   protocol, population=trace, complete=True)
    hist = {E.ARM_GLASS: sources.glass, E.ARM_RF: sources.rf}
    frozen = _frozen_tables(Path(baseline_dir or phase0.BASELINE_DIR), protocol)

    contract = settings.contract
    identities = {
        "run": {"run_id": run_id, "role": role, "feature_pathway": acc.get("feature_pathway")},
        "structural_acceptance": {
            "file": A.ACCEPTANCE_FILE, "sha256": acc_sha, "record_digest": acc.get("record_digest"),
            "accepted": acc.get("accepted"),
            "official": (acc.get("structural_acceptance") or {}).get("official"),
            "re_verified_by": "stage2_crossfit.acceptance.require_accepted"},
        "candidate": {"manifest_sha256": candidate.manifest_sha256,
                      "artifacts": {arm: {"path": candidate.relative(p),
                                          "sha256": candidate.sha256(p)}
                                    for arm, p in ((E.ARM_GLASS, candidate.glass_path),
                                                   (E.ARM_RF, candidate.rf_path))}},
        "protocol": {"id": protocol.protocol_id, "version": protocol.version,
                     "fingerprint": protocol.fingerprint(),
                     "pinned_frozen": ep.is_pinned_frozen(protocol)},
        "contract": {"id": contract.contract_id, "fingerprint": contract.fingerprint()},
        "population": {"checks": len(population.checks), "passed": population.passed,
                       "complete": population.complete,
                       "rows": int(trace.n), "positives": int(trace.n_positives)},
        "upstream": {**upstream, "combination": verified.combination,
                     "reference": verified.flags.name,
                     "reference_fingerprint": verified.flags.fingerprint,
                     "identity_checks": len(verified.identity.checks),
                     "identity_complete": verified.complete},
        "baseline": {"id": b.BASELINE_ID,
                     "manifest_sha256": protocol["population"]["baseline_manifest_sha256"],
                     "rebuild": getattr(comparison, "status", None),
                     "code_changes": list(getattr(comparison, "code_changes", ()) or ()),
                     "historical_artifacts": {
                         E.ARM_GLASS: _artifact(sources, b.ROLE_GLASS),
                         E.ARM_RF: _artifact(sources, b.ROLE_RF)},
                     "snapshot": None if getattr(sources, "snapshot", None) is None else
                     dataclasses.asdict(sources.snapshot)},
        "evidence_tiers": {EVIDENCE_SOURCE: ep.evidence_tier(EVIDENCE_SOURCE, protocol),
                           BASELINE_SOURCE: ep.evidence_tier(BASELINE_SOURCE, protocol)},
    }
    hashes = {"acceptance": (str(acc_path), acc_sha),
              "candidate_manifest": (str(candidate.root / "MANIFEST.json"),
                                     candidate.manifest_sha256)}
    return Inputs(run_id, role, protocol, contract, ref, trace, sources.lr, sources.ebm,
                  verified.flags, hist, frozen, ledger, candidate.evidence, alignment,
                  identities, record_dir, hashes)


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Analysis:
    tables: Mapping[str, pd.DataFrame]
    bootstrap: Mapping[str, ep.BootstrapRun]
    sections: Mapping[str, Tuple[str, ...]]
    gaps: Tuple[str, ...]
    layers: Mapping[str, Any]


def diagnostics_tables(glass, rf, ref, lr, ebm) -> Dict[str, pd.DataFrame]:
    """
    The baseline's Step 2A-2D tables for these views, through the baseline's own
    producers (``baseline._compute``), so corrected and pre-correction tables
    are computed identically. No GLASS-arm RF result exists for a corrected run,
    so the RF decision-time claims are ``not_verifiable`` (``routing_policy_audit``).
    """
    src = b.BaselineSources(ref=ref, glass=glass, rf=rf, lr=lr, ebm=ebm, rf_candidate=None,
                            artifacts=())
    r = b._compute(src)
    parts = {"2A.routing_rates": r["rates"], "2B.pass1_population": r["p1pop"],
             "2B.pass1_suppression": r["supp"].table, "2C.pass2_population": r["p2pop"],
             "2C.pass2_frozen": r["frozen"].table, "2C.pass2_matched": r["matched"].table,
             "2D.router_reference_agreement": r["agreement"], "2D.crosstab_summary": r["xsum"],
             "2D.boundary_positions": r["bnd"]}
    out = {
        "states": r["states"],
        "ratios": pd.concat([b._long(s, parts[s]) for s, _ in b._RATIO_SECTIONS],
                            ignore_index=True),
        "policy_audit": r["policy"], "reference_flags": r["refsum"],
        "pass2_breakdown": r["breakdown"], "common_population": r["common"],
        "matched_selection": b._matched_frame(r["matched"], r["trace"]),
        "contingency": r["contingency"], "crosstab": r["crosstab"],
        "pass2_cross_router": r["p2x"], "signal_availability": r["signal"],
        "score_conditional": r["sc"], "stratum_contrasts": r["contrasts"],
        "observed_expected": r["goe"], "chance_baselines": r["chance"],
        "model_disagreement": r["mdis"]}
    missing = sorted(set(b.TABLES) - set(out) - set(_NOT_RECOMPUTED))
    if missing:
        raise ComplementarityError(f"internal: baseline tables not reproduced {missing}")
    return out


def _counts_row(st, y, m) -> Dict[str, int]:
    st, y = st[m], y[m]
    p1, p2, ab = (st == d.PASS_1_NOT_SUBSCRIBE), (st == d.PASS_2_SUBSCRIBE), (st == d.ABSTAIN)
    return {"rows": int(m.sum()), "positives": int(y.sum()),
            "pass1_rows": int(p1.sum()), "pass1_positives": int(y[p1].sum()),
            "pass2_rows": int(p2.sum()), "pass2_positives": int(y[p2].sum()),
            "abstain_rows": int(ab.sum()), "abstain_positives": int(y[ab].sum()),
            "overlapping_rows": int(((p1 & p2) | (p1 & ab) | (p2 & ab)).sum()),
            "unassigned_rows": int((~(p1 | p2 | ab)).sum())}


def _accounting(pairs: Mapping[str, ep.PairedRouting]) -> pd.DataFrame:
    """Three-state accounting of every routing, per fold and pooled, reconciled exactly."""
    rows = []
    for label, (paired, name) in pairs.items():
        st, y, fold = np.asarray(paired.states[name]), paired.y, paired.fold_id
        for k in (*paired.folds, "all"):
            m = np.ones(paired.n, bool) if k == "all" else fold == k
            c = _counts_row(st, y, m)
            rc = paired.counts(name, None if k == "all" else k)
            reconciled = (c["rows"], c["positives"], c["pass1_rows"], c["pass1_positives"],
                          c["pass2_rows"], c["pass2_positives"]) == (
                rc.rows, rc.positives, rc.pass1_rows, rc.pass1_positives, rc.pass2_rows,
                rc.pass2_positives)
            exhaustive = (c["pass1_rows"] + c["pass2_rows"] + c["abstain_rows"] == c["rows"]
                          and c["pass1_positives"] + c["pass2_positives"]
                          + c["abstain_positives"] == c["positives"])
            rows.append({"routing": label, "fold": k, **c,
                         "exclusive": c["overlapping_rows"] == 0,
                         "exhaustive": exhaustive and c["unassigned_rows"] == 0,
                         "reconciles_with_routing_counts": reconciled})
    frame = pd.DataFrame(rows)
    ok = frame[["exclusive", "exhaustive", "reconciles_with_routing_counts"]].all(axis=None)
    if not ok:
        raise ComplementarityError("three-state accounting failed (states not exclusive, not "
                                   "exhaustive or not reconciled)")
    return frame


def _outcome_ids(protocol) -> Dict[str, str]:
    return {o["metric"]: o["id"] for o in protocol["hierarchy"]["primary_outcomes"]}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _statement(row: Mapping[str, Any]) -> str:
    """D-C1: higher / lower only when the 95% paired interval excludes 0."""
    lo, hi = row.get("difference_low"), row.get("difference_high")
    if row.get("interval_status") != "defined" or _missing(lo) or _missing(hi):
        return "not resolved at this sample (interval undefined)"
    if lo > 0:
        return "GLASS higher than RF"
    if hi < 0:
        return "GLASS lower than RF"
    return "not resolved at this sample"


def _primary_tables(run: ep.BootstrapRun, protocol) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = _outcome_ids(protocol)
    totals = set(protocol["hierarchy"]["descriptive_totals"])
    table = run.table()
    table.insert(0, "outcome", [ids.get(m, "descriptive_total" if m in totals else "")
                                for m in table["metric"]])
    table["statement"] = [(_statement(r) if r["metric"] in ids
                           else "descriptive total (no statement rule)")
                          for r in table.to_dict("records")]
    table["difference_definition"] = "glass - blackbox"
    by_fold = []
    for m, folds in run.fold_points.items():
        for k, pts in sorted(folds.items()):
            g, x = pts.get("glass"), pts.get("blackbox")
            by_fold.append({"outcome": ids.get(m, "descriptive_total"), "metric": m,
                            "fold": int(k), "glass": g, "blackbox": x,
                            "difference": None if g is None or x is None else g - x})
    return table, pd.DataFrame(by_fold)


def _fit_results(fit: Optional[ep.FoldFit], key: str) -> Dict[str, Any]:
    if fit is None:
        return {}
    for rec in fit.records:
        rep = getattr(getattr(rec, "outcome", None), "report", None)
        if rec.key == key and rep is not None:
            return {r.obligation: r for r in rep.results}
    return {}


def _shortfalls(cross: ep.PairedRouting, ledger: ep.FoldLedger, contract,
                protocol) -> pd.DataFrame:
    """Section 11: each obligation per fold and pooled; fit-time values alongside, not merged."""
    rows = []
    for arm in TRACE_ARMS:
        obs = ep.pooled_observation(cross, arm, ledger, contract)
        fits = ledger.fits.get(arm, {})
        groups = [(k, rep, ledger.status[arm][k]) for k, rep in obs["per_fold"].items()]
        pooled_status = min((ledger.status[arm][k] for k in ledger.attempted),
                            key=ep.FIT_STATUSES.index)
        groups.append(("all", obs["pooled"], pooled_status))
        for k, rep, status in groups:
            for res in rep.results:
                o = ep.classify_observation(res, fit_status=status, protocol=protocol).to_dict()
                scope = "pass2" if res.obligation.startswith("pass2") else "pass1"
                fr = _fit_results(fits.get(k), scope).get(res.obligation) if k != "all" else None
                rows.append({"arm": arm, "fold": k,
                             "pool": obs["pool_label"] if k == "all" else "",
                             "obligation": o["obligation"], "metric": o["metric"],
                             "limit": o["limit"], "fit_status": o["fit_status"],
                             "fit_time_numerator": None if fr is None else fr.numerator,
                             "fit_time_denominator": None if fr is None else fr.denominator,
                             "fit_time_value": None if fr is None else fr.value,
                             "fit_time_status": None if fr is None else fr.status,
                             "observed_numerator": o["numerator"],
                             "observed_denominator": o["denominator"],
                             "observed_value": res.value, "observed_status": o["observed_status"],
                             "shortfall": o["shortfall"], "classification": o["classification"],
                             "low_support": o["low_support"], "tolerance": o["tolerance"],
                             "material": o["material"],
                             "materiality_rule_status": o["materiality_rule_status"]})
    return pd.DataFrame(rows)


_M6_COLUMNS = ("fold", "population_rows", "population_positives", "routed_rows",
               "routed_positives", "npv", "leakage_share")
_R2_COLUMNS = ("fold", "strip_a_in_band", "band_rows", "share", "status", "review_required")


def _fit_tables(ledger: ep.FoldLedger, evidence: Mapping[str, Any],
                trace) -> Dict[str, pd.DataFrame]:
    fold = np.asarray(trace.fold_id)
    y = np.asarray(trace.y).astype(int)
    evs = {arm: {int(ev["fold"]): ev for ev in (evidence.get("folds") or {}).get(arm) or []}
           for arm in TRACE_ARMS}
    led, obligations, solver, m6 = [], [], [], []
    for arm in TRACE_ARMS:
        for k in ledger.attempted:
            ev, fit = evs[arm].get(k) or {}, ledger.fits.get(arm, {}).get(k)
            rem = ev.get("remainder") or {}
            led.append({"arm": arm, "fold": k, "status": ledger.status[arm].get(k),
                        "accepted": k in ledger.accepted_folds(arm), "status_only": fit is None,
                        "n_train": ev.get("n_train"), "train_positives": ev.get("train_positives"),
                        "remainder_rows": rem.get("rows"),
                        "remainder_positives": rem.get("positives"),
                        "remainder_prevalence": rem.get("prevalence"),
                        "outer_rows": int((fold == k).sum()),
                        "outer_positives": int(y[fold == k].sum()),
                        "wall_seconds": ev.get("wall_seconds"),
                        "problems": list(fit.completeness.problems()) if fit else []})
            for rec in (fit.records if fit else ()):
                o = rec.outcome
                s = dict(o.solver or {})
                solver.append({"arm": arm, "fold": k, "selection": rec.key,
                               "outcome": o.outcome, "fit_status": fit.by_key.get(rec.key),
                               "disposition": o.disposition, "reason": o.reason,
                               "method": s.get("method"),
                               "solver_status": s.get("status"), "gap": s.get("gap"),
                               "gap_source": s.get("gap_source"),
                               "lexicographic_guarantee": s.get("lexicographic_guarantee"),
                               "reproduced": s.get("reproduced"),
                               "selected_rules": (len(s["selected_rule_ids"])
                                                  if s.get("selected_rule_ids") is not None
                                                  else None)})
                rep = o.report
                for r in (rep.results if rep is not None else ()):
                    obligations.append({"arm": arm, "fold": k, "selection": rec.key,
                                        "population_rows": rep.population_rows,
                                        "population_positives": rep.population_positives,
                                        "population_status": rep.population_status,
                                        **r.to_dict()})
            if arm == E.ARM_GLASS and ev.get("m6_inner_oof_pass1"):
                m6.append({"fold": k, **ev["m6_inner_oof_pass1"]})
    summary = ledger.summary()
    return {"fold_ledger": pd.DataFrame(led), "fold_ledger_summary": summary,
            "solver_outcomes": pd.DataFrame(solver),
            "fit_time_obligations": pd.DataFrame(obligations),
            "m6_inner_oof_pass1": pd.DataFrame(m6, columns=list(_M6_COLUMNS))}


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _r3(evidence: Mapping[str, Any], trace) -> pd.DataFrame:
    """F31 R3: each RF fold router's inner-OOF band against its realised outer-OOF fold."""
    st = np.asarray(trace.views[E.ARM_RF].state)
    y, fold = np.asarray(trace.y).astype(int), np.asarray(trace.fold_id)
    rows, pooled = [], {}
    spec = (("pass1_npv", "pass1", d.PASS_1_NOT_SUBSCRIBE, "pass1_confidence", False),
            ("pass2_precision", "pass2", d.PASS_2_SUBSCRIBE, "pass2_confidence", True))
    for ev in sorted((evidence.get("folds") or {}).get(E.ARM_RF) or [], key=lambda e: e["fold"]):
        k = int(ev["fold"])
        for metric, key, state, r1_key, positive in spec:
            c = ev["counts"][key]
            band_num = (c["routed_positives"] if positive
                        else c["routed_rows"] - c["routed_positives"])
            band_den = c["routed_rows"]
            m = (fold == k) & (st == state)
            real_num = int(y[m].sum()) if positive else int((y[m] == 0).sum())
            real_den = int(m.sum())
            p = pooled.setdefault(metric, [0, 0, 0, 0])
            p[0] += band_num; p[1] += band_den; p[2] += real_num; p[3] += real_den
            rows.append((k, metric, band_num, band_den, (ev.get("r1_bands") or {}).get(r1_key),
                         real_num, real_den))
    for metric, (bn, bd, rn, rd) in pooled.items():
        rows.append(("all", metric, bn, bd, None, rn, rd))
    out = []
    for k, metric, bn, bd, r1, rn, rd in rows:
        bv, rv = _ratio(bn, bd), _ratio(rn, rd)
        out.append({"fold": k, "metric": metric, "band_numerator": bn, "band_denominator": bd,
                    "band_value": bv, "r1_recorded_confidence": r1,
                    "r1_matches_band": None if r1 is None or bv is None else abs(r1 - bv) <= 1e-12,
                    "realised_numerator": rn, "realised_denominator": rd, "realised_value": rv,
                    "optimism_gap": None if bv is None or rv is None else bv - rv,
                    "band_basis": "the fold router's inner-OOF band on its fitting rows (R1)"
                    if k != "all" else "summed fold-router band counts (pool_counts convention)",
                    "use": "reported, never used to tune (protocol section 12)"})
    return pd.DataFrame(out)


def _pass1_matched(trace, lr) -> pd.DataFrame:
    """Section 7, Pass 1 (secondary): each arm's Pass 1 set vs the K lowest LR rows of all rows."""
    score = lr.scores[MATCH_SCORES["lr"]]
    rows = []
    for arm in TRACE_ARMS:
        res = d.matched_rate_comparison(trace.views[arm], d.PASS_1_NOT_SUBSCRIBE, score,
                                        population="all", by_fold=True)
        cut = res.meta["cut"]
        for r in res.table.to_dict("records"):
            rows.append({"arm": arm, "fold": "all", "set": r["set"], "n": r["n"],
                         "positives": r["positives"], "positive_rate": r["positive_rate"],
                         "positive_rate_status": r["positive_rate_status"],
                         "overlap_with_router": r["overlap_with_router"],
                         "jaccard_with_router": r["jaccard_with_router"]})
        for k, bf in sorted((cut.by_fold or {}).items()):
            rows.append({"arm": arm, "fold": int(k), "set": "cut_bounds",
                         "target_count": int(bf["target_count"]), "exact": bool(bf["exact"]),
                         "tie_block": int(bf["tie_block"]), "strict_count": int(bf["strict_count"]),
                         "inclusive_count": int(bf["inclusive_count"]),
                         "cut_value": bf.get("cut_value")})
    frame = pd.DataFrame(rows)
    frame["score"] = f"lr.{MATCH_SCORES['lr']}"
    frame["label"] = "descriptive matched-budget comparison, not a deployable threshold"
    return frame


def _pre_post_join(frozen: pd.DataFrame, post: pd.DataFrame, keys: Sequence[str],
                   values: Sequence[str], diff: Sequence[str]) -> pd.DataFrame:
    """Pre from the frozen package's stored text; post serialised and read back the same way."""
    post_txt = _read_text_csv(b._csv_bytes(post))
    for col in (*keys, *values):
        for side, f in (("frozen", frozen), ("corrected", post_txt)):
            if col not in f.columns:
                raise ComplementarityError(f"{side} table has no column {col!r}")
    m = frozen[list(keys) + list(values)].merge(post_txt[list(keys) + list(values)], on=list(keys),
                                                how="outer", suffixes=("_pre", "_post"),
                                                indicator=True, sort=False)
    m["presence"] = m.pop("_merge").map({"both": "both", "left_only": "pre_only",
                                         "right_only": "post_only"}).astype(str)
    for c in diff:
        pre = pd.to_numeric(m[f"{c}_pre"].replace("", np.nan), errors="coerce")
        post_v = pd.to_numeric(m[f"{c}_post"].replace("", np.nan), errors="coerce")
        m[f"{c}_pre_minus_post"] = pre - post_v
    return m


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Lists and mappings as canonical JSON text, so every cell serialises deterministically."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = [json.dumps(_plain(v), sort_keys=True) if isinstance(
                v, (list, tuple, dict, Mapping, set, frozenset)) else v for v in out[c]]
    return out


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)]
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


_SECTIONS = {
    "primary_outcomes": ("primary_outcomes", "primary_outcomes_by_fold"),
    "secondary_diagnostics": ("metric_table", "class_conditional", "pass1_matched_lr",
                              "diagnostics/states", "diagnostics/ratios",
                              "diagnostics/reference_flags", "diagnostics/common_population",
                              "diagnostics/contingency", "diagnostics/crosstab",
                              "diagnostics/pass2_cross_router", "diagnostics/score_conditional",
                              "diagnostics/stratum_contrasts", "diagnostics/observed_expected",
                              "diagnostics/model_disagreement", "m6_inner_oof_pass1"),
    "fold_variation": ("fold_dispersion", "primary_outcomes_by_fold"),
    "shortfalls": ("shortfalls",),
    "unique_contribution": ("primary_outcomes", "diagnostics/ratios",
                            "diagnostics/pass2_breakdown", "diagnostics/chance_baselines"),
    "matched_budget": ("diagnostics/matched_selection", "diagnostics/ratios", "pass1_matched_lr"),
    "pre_post": ("pre_post_transitions", "pre_post_metrics", "pre_post_uncertainty",
                 "pre_post_by_fold", "pre_post_states", "pre_post_ratios"),
    "uncertainty": ("primary_outcomes", "pre_post_uncertainty"),
    "fold_ledger": ("fold_ledger", "fold_ledger_summary"),
    "solver_outcomes": ("solver_outcomes", "fit_time_obligations"),
}
_LAYER_TABLES = {
    "1_implementation_correctness": ("trace_checks", "accounting", "alignment",
                                     "diagnostics/policy_audit", "diagnostics/signal_availability"),
    "2_fit_time_feasibility_and_acceptance": ("fold_ledger", "fold_ledger_summary",
                                              "fit_time_obligations", "solver_outcomes",
                                              "f31_r2_strips"),
    "3_observed_outer_oof_compliance": ("shortfalls",),
    "4_empirical_performance_and_complementarity": (
        "primary_outcomes", "primary_outcomes_by_fold", "metric_table", "fold_dispersion",
        "class_conditional", "pass1_matched_lr", "pre_post_transitions", "pre_post_metrics",
        "pre_post_uncertainty", "pre_post_by_fold", "pre_post_states", "pre_post_ratios",
        "f31_r3_optimism", "m6_inner_oof_pass1"),
}


def _counts_by(frame: pd.DataFrame, col: str) -> Dict[str, int]:
    return {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).sort_index().items()}


def analyse(inp: Inputs) -> Analysis:
    """Every table of the comparative report, in memory (protocol sections 3-12)."""
    protocol, contract, trace = inp.protocol, inp.contract, inp.trace
    ep.check_inputs(INPUT_NAMES, protocol)
    cross = ep.PairedRouting.from_trace(trace)
    inp.reference.aligned_to(cross)                       # rows, order and labels, or refuse
    pre_post = {arm: ep.PairedRouting.pre_post(inp.historical[arm], trace.views[arm])
                for arm in TRACE_ARMS}
    if set(cross.folds) != set(inp.ledger.attempted):
        raise ComplementarityError(f"routing folds {cross.folds} != attempted folds "
                                   f"{inp.ledger.attempted}")
    inp.ledger.require_complete(list(TRACE_ARMS))
    t: Dict[str, pd.DataFrame] = {}

    # 1. implementation correctness
    t["trace_checks"] = trace.check_frame()
    if not t["trace_checks"]["passed"].astype(bool).all():
        raise ComplementarityError("a trace check failed")
    t["accounting"] = _accounting({
        **{arm: (cross, arm) for arm in TRACE_ARMS},
        **{f"{arm}_pre_correction": (pre_post[arm], "pre") for arm in TRACE_ARMS}})
    t["alignment"] = inp.alignment
    diag = diagnostics_tables(trace.views[E.ARM_GLASS], trace.views[E.ARM_RF], inp.ref,
                              inp.lr, inp.ebm)
    t.update({f"diagnostics/{k}": v for k, v in diag.items()})

    # 2. fit time
    t.update(_fit_tables(inp.ledger, inp.evidence, trace))
    strips = pd.DataFrame(A.strip_review(inp.evidence, protocol), columns=list(_R2_COLUMNS))
    strips["basis"] = E.R2_BASIS
    t["f31_r2_strips"] = strips

    # 3. observed outer-OOF compliance
    t["shortfalls"] = _shortfalls(cross, inp.ledger, contract, protocol)

    # 4. performance and complementarity
    primary = [o["metric"] for o in protocol["hierarchy"]["primary_outcomes"]]
    totals = list(protocol["hierarchy"]["descriptive_totals"])
    runs = {"cross_arm": ep.run_registered_bootstrap(cross, primary + totals, protocol,
                                                     reference=inp.reference)}
    t["primary_outcomes"], t["primary_outcomes_by_fold"] = _primary_tables(runs["cross_arm"],
                                                                           protocol)
    t["metric_table"] = pd.concat([ep.metric_table(cross, arm) for arm in TRACE_ARMS],
                                  ignore_index=True)
    t["fold_dispersion"] = ep.fold_dispersion(t["metric_table"])
    t["class_conditional"] = pd.concat(
        [ep.class_conditional_counts(cross, arm, None if k == "all" else k).assign(
            routing=arm, fold=k) for arm in TRACE_ARMS for k in (*cross.folds, "all")],
        ignore_index=True)
    t["pass1_matched_lr"] = _pass1_matched(trace, inp.lr)

    trans, metrics, unc, by_fold = [], [], [], []
    for arm in TRACE_ARMS:
        pp = pre_post[arm]
        for k in (*pp.folds, "all"):
            trans.append(pp.transitions(None if k == "all" else k).assign(
                arm=arm, fold=k, changed_rows=pp.changed_rows(None if k == "all" else k)))
        metrics += [ep.metric_table(pp, "pre").assign(arm=arm),
                    ep.metric_table(pp, "post").assign(arm=arm)]
        run = runs[f"pre_post_{arm}"] = ep.run_registered_bootstrap(
            pp, list(PRE_POST_METRICS), protocol, reference=inp.reference)
        unc.append(run.table().assign(arm=arm, difference_definition="pre - post"))
        for m, folds in run.fold_points.items():
            for k, pts in sorted(folds.items()):
                pre, post = pts.get("pre"), pts.get("post")
                by_fold.append({"arm": arm, "metric": m, "fold": int(k), "pre": pre, "post": post,
                                "pre_minus_post": None if pre is None or post is None
                                else pre - post})
    t["pre_post_transitions"] = pd.concat(trans, ignore_index=True)
    t["pre_post_metrics"] = pd.concat(metrics, ignore_index=True)
    t["pre_post_uncertainty"] = pd.concat(unc, ignore_index=True)
    t["pre_post_by_fold"] = pd.DataFrame(by_fold)
    t["pre_post_states"] = _pre_post_join(
        inp.baseline_tables["states"], diag["states"], ("arm", "fold", "state"),
        ("rows", "n", "positives", "negatives", "share", "positive_rate"),
        ("n", "positives", "share", "positive_rate"))
    t["pre_post_ratios"] = _pre_post_join(
        inp.baseline_tables["ratios"], diag["ratios"], b._RATIO_KEY,
        ("label", "numerator", "denominator", "value", "status"), ("numerator", "value"))
    t["f31_r3_optimism"] = _r3(inp.evidence, trace)

    tables = {k: _flat(v) for k, v in t.items()}
    sections = {s: tuple(v) for s, v in _SECTIONS.items()}
    gaps = tuple(ep.comparative_report_gaps(sections, protocol))
    sf = t["shortfalls"]
    layers = {
        "1_implementation_correctness": {
            "trace_checks": {"total": len(t["trace_checks"]),
                             "passed": int(t["trace_checks"]["passed"].astype(bool).sum())},
            "accounting_all_exclusive_exhaustive_reconciled": True,
            "alignment_checks_passed": int(t["alignment"]["passed"].sum()),
            "policy_audit_status": {arm: _counts_by(g, "status") for arm, g in
                                    diag["policy_audit"].groupby("arm", sort=True)}},
        "2_fit_time_feasibility_and_acceptance": {
            "structural_acceptance": dict(inp.identities.get("structural_acceptance") or {}),
            "accepted_folds": {arm: list(inp.ledger.accepted_folds(arm)) for arm in TRACE_ARMS},
            "fit_time_obligations_satisfied": bool(
                (t["fit_time_obligations"]["status"] == "satisfied").all())
            if len(t["fit_time_obligations"]) else None},
        "3_observed_outer_oof_compliance": {
            f"{arm}/{scope}": {"observed_status": _counts_by(g, "observed_status"),
                               "material": int(g["material"].fillna(False).astype(bool).sum()),
                               "low_support": int(g["low_support"].astype(bool).sum())}
            for (arm, scope), g in sf.assign(scope=np.where(sf["fold"].astype(str) == "all",
                                                             "pooled", "per_fold"))
            .groupby(["arm", "scope"], sort=True)},
        "4_empirical_performance_and_complementarity": {
            "statement_rule": protocol["hierarchy"]["statement_rule"]["rule"],
            "overall_verdict": "none: no weighted score and no overall winner (D-C1)"},
    }
    for name, tabs in _LAYER_TABLES.items():
        layers[name] = {**layers[name], "tables": list(tabs)}
    return Analysis(tables, runs, sections, gaps, layers)


# ----------------------------------------------------------------------
# Package
# ----------------------------------------------------------------------
DISCLOSURES = (
    "Evidence tier 2 (pr38_corrected_outer_oof); the pre-correction baseline is tier 1. The "
    "Bank Marketing held-out test split was exposed in PRs 33-36: no test label, test "
    "prediction or test-derived threshold is read here.",
    "Pre/post differences are a COMBINED correction (GLASS selector, shared contract, F17 "
    "populations, nested rankers); no single component is attributed (protocol section 8). "
    "Only S1 (Step 3C) isolates the GLASS population policy.",
    "Known asymmetries: pre GLASS fitted Pass 2 on 'full'; pre RF used match_glass_oof "
    "constraints and global_t1; the common eligible population changes when either Pass 1 "
    "changes, so matched-budget populations differ before and after; the historical GLASS "
    "confidence is in-sample rule precision.",
    "Pre/post differences are pre - post (the registered procedure's order); cross-arm "
    "differences are GLASS - RF.",
    "Intervals: stratified paired bootstrap (fold x label), conditional on the fitted routers "
    "and the frozen upstream flags; evaluation-row sampling only. No multiplicity adjustment "
    "is pre-registered. No interval establishes contract compliance.",
    "The five outer folds are not independent replications: per-fold values, minimum, maximum "
    "and range only.",
    "Matched-budget results are descriptive, not a deployable threshold; each arm is matched "
    "at its own budget, so cross-arm budget equality is not achieved by this family.",
    "RF Router confidence is a heuristic blend (F31 Disposition A), not a calibrated "
    "probability. R2 strips use the fold router's inner-OOF band (the provisional basis of "
    "the F31 R2 clarification); R3 optimism gaps are reported, never used to tune.",
    "Routing-policy claims keep their evidence status: GLASS Pass 1 precedence and Pass 2 "
    "eligibility are code_level_only; the corrected RF decision-time claims are "
    "not_verifiable because no outer Pass 1 score or GLASS-arm RF result is persisted.",
    "Statements follow D-C1 only: higher / lower when the 95% paired interval excludes 0, "
    "otherwise not resolved at this sample. No weighted score and no overall winner.",
    "Outer-OOF observations never revise the contract, re-estimate bands or relabel a "
    "violation (protocol section 11); fit status and observation are separate columns.",
)


def _table_path(name: str) -> str:
    return f"tables/{name}.csv"


def build_package(inp: Inputs, analysis: Analysis, *, bank_root=None) -> Dict[str, bytes]:
    """Every file of the package, in memory: deterministic, no timestamps or local paths."""
    files: Dict[str, bytes] = {}
    outputs: Dict[str, Any] = {}
    for name in sorted(analysis.tables):
        df = analysis.tables[name]
        data = b._csv_bytes(df)
        path = _table_path(name)
        files[path] = data
        outputs[path] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                         "rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    code = b.code_identity(bank_root or phase0.BANK_ROOT, CODE_FILES)
    manifest = {
        "format": FORMAT, "package": PACKAGE_DIR,
        "scope": "PR 38 Commit 3 Step 3B: comparative analysis of the accepted corrected "
                 "primary's training-side outer-OOF routes (protocol sections 3-12)",
        "identities": inp.identities,
        "inputs": list(INPUT_NAMES),
        "bootstrap": {name: {"digest": run.digest, "verified": run.verify(),
                             "provenance": _plain(run.provenance)}
                      for name, run in analysis.bootstrap.items()},
        "layers": analysis.layers,
        "comparative_sections": {k: [_table_path(t) for t in v]
                                 for k, v in analysis.sections.items()},
        "comparative_section_gaps": {g: ("pending: Step 3C (S1, F17 sensitivity); not run"
                                         if g == "declared_sensitivities" else "missing")
                                     for g in analysis.gaps},
        "not_recomputed": {f"diagnostics/{t}": "needs the historical source records"
                           for t in _NOT_RECOMPUTED},
        "disclosures": list(DISCLOSURES),
        "outputs": outputs,
        "code": {"method": code.method, "files": dict(code.files)},
    }
    man = b._json_bytes(manifest)
    files[MANIFEST] = man
    files[SIDECAR] = (hashlib.sha256(man).hexdigest() + "\n").encode("ascii")
    return files


def _listing(directory: Path) -> List[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


def compare_package(directory, files: Mapping[str, bytes]) -> Tuple[str, List[str]]:
    """``identical``, ``reproduced_with_code_drift`` (only code identity differs) or ``differs``."""
    d_ = Path(directory)
    if not d_.is_dir():
        return "differs", [f"{d_}: no package"]
    present = set(_listing(d_))
    problems = [f"missing: {p}" for p in sorted(set(files) - present)]
    problems += [f"unexpected: {p}" for p in sorted(present - set(files))]
    differing = [p for p in sorted(set(files) & present) if (d_ / p).read_bytes() != files[p]]
    if not problems and not differing:
        return "identical", []
    if not problems and set(differing) <= {MANIFEST, SIDECAR}:
        old = json.loads((d_ / MANIFEST).read_bytes())
        new = json.loads(files[MANIFEST])
        old.pop("code", None)
        new.pop("code", None)
        if old == new:
            return "reproduced_with_code_drift", [f"code identity differs: {p}" for p in differing]
    return "differs", problems + [f"content differs: {p}" for p in differing]


def write_package(directory, files: Mapping[str, bytes]) -> str:
    """Write once, atomically. Identical package: ``identical``, nothing written; else refused."""
    target = Path(directory)
    if target.exists():
        status, problems = compare_package(target, files)
        if status == "identical":
            return "identical"
        raise ComplementarityConflict(f"{target} exists and differs ({status}); nothing was "
                                      "written:\n  - " + "\n  - ".join(problems))
    tmp = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        for rel, data in files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return "written"


def _require_unchanged(inp: Inputs) -> None:
    """The acceptance record and the candidate manifest are those the gate verified."""
    for what, (path, sha) in inp.gate_hashes.items():
        if not Path(path).is_file() or R.sha256_file(path) != sha:
            raise ComplementarityError(f"the {what} changed after the acceptance gate; nothing "
                                       "was written")


def run(run_id: str, sums_path, *, check: bool = False, out_dir=None,
        **load_kw) -> Dict[str, Any]:
    """
    Gate, load, analyse, package; write once (or, with ``check``, compare only).
    The public entry point takes no ready-made ``Inputs``: every package comes
    through ``load_inputs`` and therefore through ``require_accepted``.
    """
    inp = load_inputs(run_id, sums_path, **load_kw)
    analysis = analyse(inp)
    files = build_package(inp, analysis)
    _require_unchanged(inp)
    target = Path(out_dir) if out_dir else Path(inp.record_dir or phase0.RECORD_ROOT / run_id) \
        / PACKAGE_DIR
    if check:
        status, problems = compare_package(target, files)
    else:
        status, problems = write_package(target, files), []
    return {"status": status, "problems": problems, "directory": target,
            "statements": [{k: b._plain(v) for k, v in r.items()} for r in
                           analysis.tables["primary_outcomes"][
                               ["outcome", "metric", "glass", "blackbox", "difference",
                                "difference_low", "difference_high", "statement"]
                           ].to_dict("records")],
            "gaps": list(analysis.gaps), "analysis": analysis, "files": files}
