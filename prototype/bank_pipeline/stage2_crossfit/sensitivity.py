"""
stage2_crossfit.sensitivity
===========================
Commit 3, Step 3C: the G12 sensitivity S1 and its mandatory gate G6 (F17
decision record, sections 6 and 8). S1 fits GLASS once more with
``pass2_population="full"``, everything else the accepted primary's. It is
descriptive: it never chooses, replaces or alters the primary, and it is
reported apart from the corrected primary comparison (Step 3B).

    run_s1            G12's entry: the primary's structural acceptance
                      re-verifies (``require_accepted``: G1-G5, G7-G11 and G13
                      hold), and the S1 settings differ from the settings the
                      accepted primary recorded in ``pass2_population`` only.
                      Then the GLASS ``full`` cross-fit, full-train fit and
                      candidate through the runner, once per run
                      (``s1_attempt.json``; a stop writes ``s1_stopped.json``)
    load_s1_candidate the S1 candidate, every integrity check recorded
    s1_acceptance     G6 and the S1 checks, recomputed from both candidates'
                      artifacts and evidence; written once: accepted,
                      ``s1_acceptance.json``; otherwise a dated
                      ``s1_acceptance_failed_*.json`` that nothing reads
    load_s1_inputs    both gates (the primary's acceptance; the S1 acceptance
                      recomputed, equal to its record), then the views, the
                      frozen upstream reference and the frozen baseline tables
    analyse_s1        -> the S1 tables (the Step 2A-2D definitions, both configs)
    report_s1         ``research_logs/pr38/<run_id>/s1_sensitivity_v1/``, written
                      once (``check=True``: rebuild and compare only)

G6 (mandatory): in every outer fold and in the full-train fit, both GLASS
runs were fitted on the same rows, deployed identical Pass 1 rules (segments,
class, precision, order) and recounted identical fit-time Pass 1 counts; on
the outer OOF, their Pass 1 decisions and confidences are identical row by
row. Every comparison is exact. A difference is a defect: S1 is not accepted,
the failure is recorded, and no comparison is produced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import shared.stage2.evaluation_protocol as ep
from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b

from . import acceptance as A
from . import complementarity as C
from . import evidence as E
from . import inner_features as IF
from . import phase0
from . import runner as R

__all__ = ["FORMAT", "PACKAGE_DIR", "CONFIGS", "S1_ACCEPTANCE_FILE", "S1_RECORDS",
           "SensitivityError", "S1Inputs", "single_factor_problems", "run_s1",
           "load_s1_candidate", "s1_acceptance_record", "s1_acceptance", "load_s1_inputs",
           "analyse_s1", "report_s1", "comparative_report_status"]

S1_ATTEMPT_FORMAT = "pr38.stage2.s1_attempt/1"
S1_STOP_FORMAT = "pr38.stage2.s1_stopped/1"
S1_ACCEPTANCE_FORMAT = "pr38.stage2.s1_acceptance/1"
S1_ATTEMPT_FILE, S1_MANIFEST_FILE, S1_STOP_FILE = ("s1_attempt.json", "s1_MANIFEST.json",
                                                   "s1_stopped.json")
S1_RECORDS = (S1_ATTEMPT_FILE, S1_MANIFEST_FILE, S1_STOP_FILE)
S1_ACCEPTANCE_FILE = "s1_acceptance.json"
FORMAT = "pr38.stage2.s1_sensitivity/1"
PACKAGE_DIR = "s1_sensitivity_v1"
#: The pair: the accepted primary's GLASS, then S1's (F17 section 6).
CONFIGS = ("oof_remainder", "full")
GROUPS = ("S1.integrity", "S1.configuration", "S1.population", "S1.fit_time", "G6")
P1 = d.PASS_1_NOT_SUBSCRIBE


class SensitivityError(R.RunnerError):
    """S1 cannot run, be accepted or be reported on these inputs; nothing was written."""


# ----------------------------------------------------------------------
# The gate and the single factor
# ----------------------------------------------------------------------
def _primary_gate(run_id: str, settings, output_root, record_dir):
    """``require_accepted`` for the run's primary; nothing is read before it passes."""
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    acc_path = record_dir / A.ACCEPTANCE_FILE
    if settings is None and acc_path.is_file():
        try:                                      # unreadable: require_accepted refuses
            pathway = (A._json(acc_path).get("feature_pathway") or {}).get("pathway")
        except (ValueError, AttributeError):
            pathway = None
        settings = R.primary_settings(pathway) if pathway in IF.PATHWAYS else None
    primary = A.require_accepted(run_id, role=R.ROLE_PRIMARY, output_root=output_root,
                                 record_dir=record_dir, settings=settings)
    acc = A._json(acc_path)
    if settings is None:
        settings = R.primary_settings(acc["feature_pathway"]["pathway"])
    return primary, settings, record_dir, R.sha256_file(acc_path), acc


def _recorded_settings(candidate: A.Candidate) -> Dict[str, Any]:
    """The settings record inside the candidate's GLASS bundle (loaded at its manifest SHA-256)."""
    return A._canon((candidate.load(candidate.glass_path).get("pr38_evidence") or {})
                    .get("settings") or {})


def single_factor_problems(primary: Mapping[str, Any], s1: Mapping[str, Any]) -> List[str]:
    """
    Why the S1 settings record is not the primary's with ``pass2_population``
    changed to ``"full"`` alone (F17 section 6, "held constant").
    """
    out = []
    if (primary.get("role"), s1.get("role")) != (R.ROLE_PRIMARY, R.ROLE_S1):
        out.append(f"roles {primary.get('role')!r} / {s1.get('role')!r}, expected "
                   f"{R.ROLE_PRIMARY!r} / {R.ROLE_S1!r}")
    gp, gs = primary.get("glass_config") or {}, s1.get("glass_config") or {}
    differ = sorted(k for k in set(gp) | set(gs) if gp.get(k) != gs.get(k))
    if differ != ["pass2_population"]:
        out.append(f"GLASS config fields differing: {differ}, expected ['pass2_population']")
    if (gp.get("pass2_population"), gs.get("pass2_population")) != CONFIGS:
        out.append(f"pass2_population {gp.get('pass2_population')!r} -> "
                   f"{gs.get('pass2_population')!r}, expected {CONFIGS[0]!r} -> {CONFIGS[1]!r}")
    for key in ("ranker", "feature_pathway"):
        if primary.get(key) != s1.get(key):
            out.append(f"{key} differs")
    if (primary.get("checks") or {}).get("identities") != (s1.get("checks") or {}).get(
            "identities"):
        out.append("frozen identities (contract, protocol, profiles) differ")
    if "rf_config" in s1:
        out.append("S1 has an RF arm; F17 runs no RF counterpart")
    return out


# ----------------------------------------------------------------------
# The S1 run (G12)
# ----------------------------------------------------------------------
def run_s1(run_id: str, *, primary_settings: Optional[R.ArmSettings] = None,
           settings: Optional[R.ArmSettings] = None, output_root=None, record_dir=None,
           train_side: Optional[Callable[[], R.TrainSide]] = None,
           verbose: bool = False) -> Dict[str, Any]:
    """
    S1 for an accepted primary, once per run: refused before any data is read
    unless ``require_accepted`` passes, and before any fit unless the S1
    settings are the primary's recorded settings with ``pass2_population``
    alone changed. A stop writes ``s1_stopped.json`` and raises (never retried
    under the run). The candidate goes to ``<output_root>/<run_id>/s1_full/``;
    the primary's candidate and records are never written.
    """
    primary, psettings, record_dir, acc_sha, acc = _primary_gate(
        run_id, primary_settings, output_root, record_dir)
    settings = settings or R.s1_settings(acc["feature_pathway"]["pathway"])
    problems = ([] if settings.role == R.ROLE_S1 else [f"role {settings.role!r} is not S1"])
    problems += single_factor_problems(_recorded_settings(primary),
                                       A._canon(R.settings_record(settings)))
    if problems:
        raise SensitivityError("S1 refused: not the accepted primary with pass2_population "
                               "alone changed:\n  - " + "\n  - ".join(problems))
    run_root = Path(output_root or phase0.OUTPUT_ROOT) / run_id / R.ROLE_S1
    done = [str(p) for p in (run_root, *(record_dir / n for n in S1_RECORDS)) if p.exists()]
    if done:
        raise SensitivityError(f"run {run_id!r} already has an S1 attempt {done}: a stopped, "
                               "interrupted or completed S1 is investigated, never rerun")
    ts = (train_side or R.load_train_side)()
    if not R.check_population(ts, settings.protocol)["passed"]:
        raise SensitivityError("the train side does not match the frozen protocol's population")
    binding = {"primary_acceptance_sha256": acc_sha,
               "primary_acceptance_digest": acc.get("record_digest"),
               "primary_manifest_sha256": primary.manifest_sha256}
    R._write_durable(record_dir / S1_ATTEMPT_FILE, {
        "format": S1_ATTEMPT_FORMAT, "run_id": run_id, "role": R.ROLE_S1,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_pathway": IF.pathway_identity(settings.feature_pathway),
        "identities": settings.checks.get("identities"), **binding,
        "note": "written before the first fit; an attempt with neither s1_MANIFEST.json nor "
                "s1_stopped.json was interrupted and is investigated, never rerun"})
    result, step = R.RunResult(settings.role), "cross-fit"
    arms = (E.ARM_GLASS,)
    try:
        R.run_crossfit(ts, settings, arms=arms, verbose=verbose, result=result)
        step = "full-train fit"
        R.fit_full_train(ts, settings, result, arms=arms, verbose=verbose)
        step = "save"
        saved = R.save_run(result, ts, settings, run_root, run_id, extra_identity=binding)
    except BaseException as exc:                     # an interruption is recorded too
        import traceback
        path = R._write_durable(record_dir / S1_STOP_FILE, {
            "format": S1_STOP_FORMAT, "run_id": run_id, "role": R.ROLE_S1, "stopped_at": step,
            "exception": type(exc).__name__, "reason": str(exc),
            "traceback": traceback.format_exc().splitlines()[-12:], "accepted": False,
            "fit_status_by_fold": {str(ev["fold"]): (ev.get("fit_status") or {}).get("status")
                                   for ev in result.evidence.get(E.ARM_GLASS, [])},
            "folds": [R._slim(ev) for ev in result.evidence.get(E.ARM_GLASS, [])],
            "full_train": {a: R._slim(ev) for a, ev in result.full_train_evidence.items()},
            "seconds": result.seconds, **binding,
            "note": "not an S1 result: no sensitivity comparison was computed"})
        if not isinstance(exc, Exception):
            raise
        raise SensitivityError(f"S1 stopped during the {step}: {exc} (evidence: {path})") from exc
    R.write_record(record_dir / S1_MANIFEST_FILE, saved["manifest"])
    return {"saved": saved, "result": result}


def load_s1_candidate(run_id: str, *, output_root=None, record_dir=None) -> A.Candidate:
    """The S1 candidate ``run_s1`` wrote (GLASS only), every integrity check recorded."""
    root = Path(output_root or phase0.OUTPUT_ROOT) / run_id / R.ROLE_S1
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    man_path = root / "MANIFEST.json"
    if not man_path.is_file():
        raise SensitivityError(f"no S1 candidate MANIFEST.json at {man_path}: run `s1` first")
    g = A._Gate("S1.integrity")
    man_sha = R.sha256_file(man_path)
    manifest = A._json(man_path)
    g.check("MANIFEST.json.sha256 is the SHA-256 of MANIFEST.json",
            lambda: (root / "MANIFEST.json.sha256").read_text(encoding="utf-8").strip() == man_sha,
            man_sha)
    g.check("the version-controlled copy s1_MANIFEST.json is byte-identical",
            lambda: R.sha256_file(record_dir / S1_MANIFEST_FILE) == man_sha)
    g.check("manifest format, run and role",
            lambda: (manifest.get("format"), manifest.get("run_id"), manifest.get("role"))
            == (R.RUN_FORMAT, run_id, R.ROLE_S1))
    g.check("one durable S1 attempt for this run",
            lambda: (lambda a: (a.get("format"), a.get("run_id"), a.get("role"))
                     == (S1_ATTEMPT_FORMAT, run_id, R.ROLE_S1))(A._json(record_dir
                                                                     / S1_ATTEMPT_FILE)))
    g.check("the attempt ended in this candidate, not in a stop",
            lambda: not (record_dir / S1_STOP_FILE).exists())
    files = manifest.get("files") or {}
    bad = [rel for rel, meta in files.items()
           if not (root / rel).is_file() or (root / rel).stat().st_size != meta.get("bytes")
           or R.sha256_file(root / rel) != meta.get("sha256")]
    g.check("every listed file is present with its SHA-256 and size",
            lambda: bool(files) and not bad, f"{len(files)} files; differing {bad[:5]}")
    listed = set(files) | {"MANIFEST.json", "MANIFEST.json.sha256"}
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix() not in listed)
    g.check("no unlisted file in the candidate directory", lambda: not extra, f"{extra[:5]}")
    glass = A._one(files, rf"glass_router/glass_router_pr38_{R.ROLE_S1}_{run_id}\.joblib")
    g.check("one GLASS bundle named for the run, and no RF Router (F17: no RF counterpart)",
            lambda: glass is not None and not any(r.startswith("rf_router/") for r in files))
    evidence = {}
    if "evidence.json" in files and "evidence.json" not in bad:
        evidence = A._json(root / "evidence.json")
    g.check("evidence.json is the run's",
            lambda: (evidence.get("format"), evidence.get("run_id"), evidence.get("role"))
            == (R.RUN_FORMAT, run_id, R.ROLE_S1))
    return A.Candidate(run_id, R.ROLE_S1, root, record_dir, manifest, man_sha,
                       None if glass is None else root / glass, None, evidence, tuple(g.checks))


# ----------------------------------------------------------------------
# S1 acceptance: G6 and the S1 checks
# ----------------------------------------------------------------------
def _fits(evidence: Mapping[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """A candidate's GLASS fits by fold, the full-train fit under ``"full_train"``."""
    out: Dict[Any, Dict[str, Any]] = {int(ev["fold"]): ev for ev in
                                      (evidence.get("folds") or {}).get(E.ARM_GLASS) or []}
    full = (evidence.get("full_train") or {}).get(E.ARM_GLASS)
    if full is not None:
        out["full_train"] = full
    return out


def _rule_key(rules) -> List[Any]:
    """What G6 compares: segment, class and precision of each rule, in deployment order."""
    return [[r.get("segment"), r.get("predicted_class"), r.get("precision")] for r in rules]


def _s1_gates(primary: A.Candidate, s1: A.Candidate, ts: R.TrainSide,
              psettings: R.ArmSettings, s1_settings: R.ArmSettings,
              binding: Mapping[str, Any]) -> Dict[str, A._Gate]:
    gates = {name: A._Gate(name) for name in GROUPS}
    gates["S1.integrity"].checks.extend(dict(c) for c in s1.integrity)
    if not s1.intact:
        for name in GROUPS[1:]:
            gates[name].not_run = "the S1 candidate failed its integrity checks; nothing loaded"
        return gates
    protocol, contract = psettings.protocol, psettings.contract
    attempted = [int(k) for k in protocol["folds"]["attempted"]]

    # configuration: the single factor, from the settings both bundles recorded
    g = gates["S1.configuration"]
    rec_p, rec_s = _recorded_settings(primary), _recorded_settings(s1)
    g.check("S1 is the accepted primary with pass2_population alone changed "
            "(oof_remainder -> full)", lambda: not single_factor_problems(rec_p, rec_s),
            lambda: single_factor_problems(rec_p, rec_s) or "single factor")
    g.check("the S1 bundle records the S1 settings built for the primary's pathway",
            lambda: A._same(rec_s, R.settings_record(s1_settings)))
    g.check("S1 manifest identities and feature pathway are the primary's",
            lambda: A._same(s1.manifest.get("identities"), primary.manifest.get("identities"))
            and A._same(s1.manifest.get("feature_pathway"), primary.manifest.get(
                "feature_pathway")))
    g.check("S1 was run for this accepted primary (acceptance and manifest SHA-256)",
            lambda: A._same(s1.manifest.get("extra_identity"), binding))

    # population: rows, labels and folds; the S1 trace through T1-T7 and the frozen identity
    g = gates["S1.population"]
    ref = A.train_reference(ts, protocol)
    views: Dict[str, Any] = {}

    def load_views():
        views.update(A.candidate_views(primary, ref))
        views["s1"] = d.extract_router_view(s1.glass_path, ref, expected_arm="glass",
                                            expected_sha256=s1.sha256(s1.glass_path))
        return True
    g.check("both GLASS outer-OOF blocks extract (boundary extractor, manifest SHA-256)",
            load_views)
    g.check("S1 manifest population (router split, training labels, fold plan) is the primary's",
            lambda: A._same(s1.manifest.get("population"), primary.manifest.get("population")))
    pv, sv = views.get(E.ARM_GLASS), views.get("s1")
    g.check("outer-OOF rows, labels and folds are identical",
            lambda: pv.index.equals(sv.index) and np.array_equal(pv.y, sv.y)
            and np.array_equal(pv.fold_id, sv.fold_id))
    g.check("outer-OOF blocks declare oof_remainder (primary) and full (S1)",
            lambda: (pv.pass2_population["cross_fit"], sv.pass2_population["cross_fit"])
            == CONFIGS and (pv.pass2_population["full_train"],
                            sv.pass2_population["full_train"]) == CONFIGS)
    # the S1 trace (S1 GLASS, primary RF): every T1-T7 check recorded one by one, as V1 does
    trace = None
    try:
        trace = d.build_router_trace(sv, views[E.ARM_RF], ref)
    except Exception as exc:                        # noqa: BLE001 - recorded as failures
        g.check("the S1 trace (S1 GLASS, primary RF) is built", lambda: False,
                f"{type(exc).__name__}: " + " | ".join(str(exc).splitlines()[:12]))
        report = getattr(exc, "report", None)       # TraceCheckError: every check it ran
        if hasattr(report, "to_dict"):
            for r in report.to_dict("records"):
                g.check(f"trace [{r.get('stage')}] {r.get('check')}",
                        lambda r=r: r.get("passed") is True)
    if trace is not None:
        A._trace_checks(g, trace)
        held: Dict[str, Any] = {}
        g.check("check_population_identity of the S1 trace, complete",
                lambda: (lambda rep: rep.passed and rep.complete)(held.setdefault(
                    "rep", ep.check_population_identity(
                        trace, ep.PopulationIdentity.from_protocol(protocol)))),
                lambda: held["rep"].failures() + held["rep"].pending() or "every check passed")

    # fit time: every S1 fit re-classified and accepted, isolation clean
    g = gates["S1.fit_time"]
    fits_p, fits_s = _fits(primary.evidence), _fits(s1.evidence)
    g.check("S1 fits are exactly the attempted folds and the full-train fit",
            lambda: sorted(fits_s, key=str) == sorted([*attempted, "full_train"], key=str),
            lambda: sorted(fits_s, key=str))
    for k, ev in sorted(fits_s.items(), key=lambda kv: str(kv[0])):
        g.check(f"fold {k}: S1 fit status re-classified (E.s1_fold_status) and accepted",
                lambda ev=ev: (lambda st: st["accepted"] and A._same(st, ev.get("fit_status")))(
                    E.s1_fold_status(ev, contract)),
                lambda ev=ev: (ev.get("fit_status") or {}).get("status"))
        g.check(f"fold {k}: isolation clean", lambda ev=ev: R._isolation_clean(ev["isolation"]))

    # G6
    g = gates["G6"]
    fold = None if pv is None else np.asarray(pv.fold_id)
    for k in [*attempted, "full_train"]:
        ep_, es = fits_p.get(k) or {}, fits_s.get(k) or {}
        g.check(f"fold {k}: both runs fitted on the same rows (rows, positives, label fingerprint)",
                lambda ep_=ep_, es=es: bool(ep_) and bool(es) and all(
                    ep_.get(f) == es.get(f) for f in ("n_train", "train_positives",
                                                       "train_label_fingerprint")))
        g.check(f"fold {k}: deployed Pass 1 rules recorded in both runs",
                lambda ep_=ep_, es=es: isinstance(ep_.get("pass1_rules"), list)
                and isinstance(es.get("pass1_rules"), list))
        g.check(f"fold {k}: deployed Pass 1 rules identical (segments, class, precision, order)",
                lambda ep_=ep_, es=es: _rule_key(ep_["pass1_rules"]) == _rule_key(
                    es["pass1_rules"]),
                lambda ep_=ep_, es=es: f"{len(ep_.get('pass1_rules') or [])} vs "
                                       f"{len(es.get('pass1_rules') or [])} rules")
        g.check(f"fold {k}: fit-time Pass 1 counts identical",
                lambda ep_=ep_, es=es: ep_["counts"]["pass1"] == es["counts"]["pass1"])
        if k == "full_train":
            continue
        m = None if fold is None else fold == k
        g.check(f"fold {k}: outer-OOF Pass 1 decisions identical row by row",
                lambda m=m: np.array_equal(np.asarray(pv.state)[m] == P1,
                                           np.asarray(sv.state)[m] == P1),
                lambda m=m: f"{int((np.asarray(pv.state)[m] == P1).sum())} vs "
                            f"{int((np.asarray(sv.state)[m] == P1).sum())} rows")
        g.check(f"fold {k}: outer-OOF Pass 1 confidences identical row by row",
                lambda m=m: (lambda p1: np.array_equal(np.asarray(pv.confidence)[m][p1],
                                                       np.asarray(sv.confidence)[m][p1]))(
                    np.asarray(pv.state)[m] == P1))
    return gates


def s1_acceptance_record(primary: A.Candidate, s1: A.Candidate, ts: R.TrainSide,
                         psettings: R.ArmSettings, s1_settings: R.ArmSettings, *,
                         acceptance_sha256: str, acceptance: Mapping[str, Any]) -> Dict[str, Any]:
    """The S1 acceptance (deterministic: no timestamp), without its digest."""
    binding = {"primary_acceptance_sha256": acceptance_sha256,
               "primary_acceptance_digest": acceptance.get("record_digest"),
               "primary_manifest_sha256": primary.manifest_sha256}
    gates = _s1_gates(primary, s1, ts, psettings, s1_settings, binding)
    accepted = all(g.passed for g in gates.values())
    g6 = gates["G6"].passed
    return A._canon({
        "format": S1_ACCEPTANCE_FORMAT, "run_id": s1.run_id, "role": R.ROLE_S1,
        "accepted": accepted, "g6_passed": g6,
        "primary": {"acceptance_sha256": acceptance_sha256,
                    "acceptance_digest": acceptance.get("record_digest"),
                    "manifest_sha256": primary.manifest_sha256},
        "s1": {"manifest_sha256": s1.manifest_sha256,
               "glass": s1.relative(s1.glass_path)},
        "protocol": A._protocol_identity(psettings.protocol),
        "feature_pathway": IF.pathway_identity(psettings.feature_pathway),
        "gates": {name: g.record() for name, g in gates.items()},
        "statement": ("S1 accepted as a descriptive sensitivity: G6 passed" if accepted else
                      "S1 NOT accepted" + ("" if g6 else ": G6 Pass 1 invariance failed (a "
                                           "defect; no sensitivity comparison is produced)")),
    })


def s1_acceptance(run_id: str, *, primary_settings=None, s1_settings=None, output_root=None,
                  record_dir=None, train_side=None) -> Tuple[Dict[str, Any], Path]:
    """
    G6 and the S1 checks for the run, after the primary's gate; written once
    with its digest: accepted, ``s1_acceptance.json``; otherwise a dated
    ``s1_acceptance_failed_*.json`` (an explicit record, never read as accepted).
    """
    primary, psettings, record_dir, acc_sha, acc = _primary_gate(
        run_id, primary_settings, output_root, record_dir)
    if (record_dir / S1_ACCEPTANCE_FILE).exists():
        raise SensitivityError(f"run {run_id!r} already has {S1_ACCEPTANCE_FILE}; it is never "
                               "replaced (s1-report re-verifies it)")
    s1_settings = s1_settings or R.s1_settings(acc["feature_pathway"]["pathway"])
    s1 = load_s1_candidate(run_id, output_root=output_root, record_dir=record_dir)
    ts = (train_side or R.load_train_side)() if s1.intact else None
    if ts is not None and not R.check_population(ts, psettings.protocol)["passed"]:
        raise SensitivityError("the train side does not match the frozen protocol's population")
    rec = s1_acceptance_record(primary, s1, ts, psettings, s1_settings,
                               acceptance_sha256=acc_sha, acceptance=acc)
    rec["record_digest"] = A._digest(rec)
    if rec["accepted"]:
        return rec, R.write_record(record_dir / S1_ACCEPTANCE_FILE, rec)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return rec, R.write_record(record_dir / f"s1_acceptance_failed_{stamp}.json", rec)


# ----------------------------------------------------------------------
# Inputs of the S1 report
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class S1Inputs:
    run_id: str
    protocol: Any
    ref: Any
    primary_trace: Any                      # primary GLASS + primary RF
    s1_trace: Any                           # S1 GLASS + primary RF
    lr: Any
    ebm: Any
    reference: ep.ReferenceFlags
    evidence: Mapping[str, Any]             # CONFIGS -> candidate evidence
    settings: Mapping[str, Any]             # CONFIGS -> recorded settings
    acceptance: Mapping[str, Any]           # the S1 acceptance record
    baseline_tables: Mapping[str, pd.DataFrame]
    identities: Mapping[str, Any]
    record_dir: Optional[Path] = None
    gate_hashes: Mapping[str, Tuple[str, str]] = dataclasses.field(default_factory=dict)


def upstream_views(ref, protocol, sums_path):
    """The frozen LR / EBM views (protocol section 5) from the archive at the pinned SHA-256."""
    archive = Path(sums_path).parent / "bank_pipeline"
    arts = protocol["upstream"]["artifacts"]
    return (d.extract_lr_view(archive / arts["lr"]["path"], ref,
                              expected_sha256=arts["lr"]["sha256"]),
            d.extract_ebm_view(archive / arts["ebm"]["path"], ref,
                               expected_sha256=arts["ebm"]["sha256"]))


def load_s1_inputs(run_id: str, sums_path, *, primary_settings=None, s1_settings=None,
                   output_root=None, record_dir=None, baseline_dir=None,
                   train_side=None, upstream=None) -> S1Inputs:
    """
    Both gates before any data: the primary's ``require_accepted``, then an
    accepted ``s1_acceptance.json`` with a valid digest; after the train side
    is loaded, the S1 acceptance is recomputed and must equal its record. Then
    the views, the verified frozen upstream reference and the frozen baseline.
    ``train_side`` / ``upstream`` replace ``load_train_side`` / ``upstream_views`` (tests).
    """
    primary, psettings, record_dir, acc_sha, acc = _primary_gate(
        run_id, primary_settings, output_root, record_dir)
    s1_path = record_dir / S1_ACCEPTANCE_FILE
    if not s1_path.is_file():
        raise SensitivityError(f"no S1 acceptance record at {s1_path}: S1 is reported only "
                               "after G6 passes (`s1-accept`)")
    s1_rec = A._json(s1_path)
    if s1_rec.get("record_digest") != A._digest(s1_rec) or s1_rec.get("accepted") is not True:
        raise SensitivityError("s1_acceptance.json does not establish an accepted S1 (digest or "
                               "verdict)")
    s1_sha = R.sha256_file(s1_path)
    s1_settings = s1_settings or R.s1_settings(acc["feature_pathway"]["pathway"])
    s1 = load_s1_candidate(run_id, output_root=output_root, record_dir=record_dir)
    if not s1.intact:
        raise SensitivityError("the S1 candidate fails its integrity checks; nothing is loaded")
    protocol = psettings.protocol
    ts = (train_side or R.load_train_side)()
    if not R.check_population(ts, protocol)["passed"]:
        raise SensitivityError("the train side does not match the frozen protocol's population")
    now = s1_acceptance_record(primary, s1, ts, psettings, s1_settings,
                               acceptance_sha256=acc_sha, acceptance=acc)
    if now != {k: v for k, v in s1_rec.items() if k != "record_digest"}:
        raise SensitivityError("the S1 acceptance does not re-verify from the artifacts now")
    ref = A.train_reference(ts, protocol)
    views = A.candidate_views(primary, ref)
    s1_view = d.extract_router_view(s1.glass_path, ref, expected_arm="glass",
                                    expected_sha256=s1.sha256(s1.glass_path))
    primary_trace = d.build_router_trace(views[E.ARM_GLASS], views[E.ARM_RF], ref)
    s1_trace = d.build_router_trace(s1_view, views[E.ARM_RF], ref)
    lr, ebm = (upstream or upstream_views)(ref, protocol, sums_path)
    verified = ep.verify_reference(ep.UpstreamSources.from_views(lr, ebm), protocol,
                                   population=primary_trace, complete=True)
    frozen = C._frozen_tables(Path(baseline_dir or phase0.BASELINE_DIR), protocol)
    identities = {
        "run": {"run_id": run_id, "role": R.ROLE_S1,
                "feature_pathway": IF.pathway_identity(psettings.feature_pathway)},
        "primary": {"structural_acceptance_sha256": acc_sha,
                    "structural_acceptance_digest": acc.get("record_digest"),
                    "manifest_sha256": primary.manifest_sha256,
                    "glass": {"path": primary.relative(primary.glass_path),
                              "sha256": primary.sha256(primary.glass_path)},
                    "rf": {"path": primary.relative(primary.rf_path),
                           "sha256": primary.sha256(primary.rf_path)}},
        "s1": {"acceptance_sha256": s1_sha, "acceptance_digest": s1_rec["record_digest"],
               "manifest_sha256": s1.manifest_sha256,
               "glass": {"path": s1.relative(s1.glass_path),
                         "sha256": s1.sha256(s1.glass_path)}},
        "protocol": {"id": protocol.protocol_id, "version": protocol.version,
                     "fingerprint": protocol.fingerprint()},
        "contract": {"id": psettings.contract.contract_id,
                     "fingerprint": psettings.contract.fingerprint()},
        "upstream": {**{k: dict(v) for k, v in protocol["upstream"]["artifacts"].items()},
                     "reference": verified.flags.name,
                     "reference_fingerprint": verified.flags.fingerprint,
                     "identity_complete": verified.complete},
        "baseline": {"id": b.BASELINE_ID,
                     "manifest_sha256": protocol["population"]["baseline_manifest_sha256"]},
        "evidence_tier": {C.EVIDENCE_SOURCE: ep.evidence_tier(C.EVIDENCE_SOURCE, protocol)},
    }
    hashes = {"primary acceptance": (str(record_dir / A.ACCEPTANCE_FILE), acc_sha),
              "S1 acceptance": (str(s1_path), s1_sha),
              "S1 manifest": (str(s1.root / "MANIFEST.json"), s1.manifest_sha256)}
    return S1Inputs(run_id, protocol, ref, primary_trace, s1_trace, lr, ebm, verified.flags,
                    {CONFIGS[0]: primary.evidence, CONFIGS[1]: s1.evidence},
                    {CONFIGS[0]: _recorded_settings(primary), CONFIGS[1]: _recorded_settings(s1)},
                    s1_rec, frozen, identities, record_dir, hashes)


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _text(df: pd.DataFrame) -> pd.DataFrame:
    return C._read_text_csv(b._csv_bytes(df))


def _join(frames: Mapping[str, pd.DataFrame], keys: Sequence[str], values: Sequence[str],
          diffs: Sequence[Tuple[str, str, str]] = ()) -> pd.DataFrame:
    """
    Outer join of text tables (``_text``; the frozen ones as stored) on ``keys``;
    each value suffixed with its table's name; ``diffs``: ``(column, a, b)`` adds
    ``<column>_<a>_minus_<b>``.
    """
    out = None
    for name, f in frames.items():
        missing = [c for c in (*keys, *values) if c not in f.columns]
        if missing:
            raise SensitivityError(f"{name} table has no columns {missing}")
        part = f[list(keys) + list(values)].rename(columns={v: f"{v}_{name}" for v in values})
        part = part.assign(**{f"in_{name}": True})
        out = part if out is None else out.merge(part, on=list(keys), how="outer", sort=False)
    for name in frames:
        out[f"in_{name}"] = out[f"in_{name}"].eq(True)
    for col, a, b_ in diffs:
        num = {n: pd.to_numeric(out[f"{col}_{n}"].replace("", np.nan), errors="coerce")
               for n in (a, b_)}
        out[f"{col}_{a}_minus_{b_}"] = num[a] - num[b_]
    return out


def _pass1_digest(rules) -> Optional[str]:
    if rules is None:
        return None
    return hashlib.sha256(json.dumps(_rule_key(rules), sort_keys=True).encode()).hexdigest()[:16]


def _fit_rows(inp: S1Inputs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pop, feas, g6 = [], [], []
    fits = {cfg: _fits(inp.evidence[cfg]) for cfg in CONFIGS}
    trace = inp.primary_trace
    fold = np.asarray(trace.fold_id)
    for cfg in CONFIGS:
        st = np.asarray((trace if cfg == CONFIGS[0] else inp.s1_trace).views["glass"].state)
        for k, ev in sorted(fits[cfg].items(), key=lambda kv: str(kv[0])):
            c1, c2 = ev["counts"]["pass1"], ev["counts"]["pass2"]
            pop.append({"config": cfg, "fold": k, "n_train": ev["n_train"],
                        "train_positives": ev["train_positives"],
                        "pass2_population_rows": c2["population_rows"],
                        "pass2_population_positives": c2["population_positives"],
                        "pass2_population_prevalence": C._ratio(c2["population_positives"],
                                                                c2["population_rows"]),
                        "pass2_population_share_of_fold": C._ratio(c2["population_rows"],
                                                                   ev["n_train"]),
                        "fit_pass2_routed_rows": c2["routed_rows"],
                        "fit_pass2_routed_positives": c2["routed_positives"],
                        "fit_pass2_precision": C._ratio(c2["routed_positives"], c2["routed_rows"]),
                        "fit_pass1_routed_rows": c1["routed_rows"],
                        "fit_pass1_routed_positives": c1["routed_positives"]})
            reps = ev.get("selection_reports") or {}
            fs = ev.get("fit_status") or {}
            feas.append({"config": cfg, "fold": k, "status": fs.get("status"),
                         "accepted": fs.get("accepted"), "by_key": fs.get("by_key"),
                         "pass1_source": (reps.get("pass1") or {}).get("source"),
                         "pass1_lexicographic_guarantee": (reps.get("pass1") or {}).get(
                             "lexicographic_guarantee"),
                         "pass2_source": (reps.get("pass2") or {}).get("source"),
                         "pass2_lexicographic_guarantee": (reps.get("pass2") or {}).get(
                             "lexicographic_guarantee"),
                         "inner_pass1_selections": len(reps.get("inner_pass1") or []),
                         "wall_seconds": ev.get("wall_seconds")})
            if cfg == CONFIGS[1]:
                continue
            es = fits[CONFIGS[1]].get(k) or {}
            row = {"fold": k, "pass1_rules": len(ev.get("pass1_rules") or []),
                   "pass1_rules_digest_oof_remainder": _pass1_digest(ev.get("pass1_rules")),
                   "pass1_rules_digest_full": _pass1_digest(es.get("pass1_rules")),
                   "fit_pass1_routed_rows_oof_remainder": c1["routed_rows"],
                   "fit_pass1_routed_rows_full": (es.get("counts") or {}).get("pass1", {}).get(
                       "routed_rows")}
            if k != "full_train":
                m = fold == k
                s1_st = np.asarray(inp.s1_trace.views["glass"].state)
                row.update({"outer_pass1_rows_oof_remainder": int((st[m] == P1).sum()),
                            "outer_pass1_rows_full": int((s1_st[m] == P1).sum())})
            g6.append(row)
    return pd.DataFrame(pop), pd.DataFrame(feas), pd.DataFrame(g6)


def _transitions(inp: S1Inputs) -> pd.DataFrame:
    """Each row's GLASS state under oof_remainder (primary) and full (S1), per fold and pooled."""
    pre = np.asarray(inp.primary_trace.views["glass"].state)
    post = np.asarray(inp.s1_trace.views["glass"].state)
    y, fold = np.asarray(inp.primary_trace.y).astype(int), np.asarray(inp.primary_trace.fold_id)
    rows = []
    for k in (*sorted(int(f) for f in np.unique(fold)), "all"):
        m = np.ones(len(y), bool) if k == "all" else fold == k
        changed = int((pre[m] != post[m]).sum())
        for a in d.ROUTER_STATES:
            for b_ in d.ROUTER_STATES:
                cell = m & (pre == a) & (post == b_)
                rows.append({"fold": k, "state_oof_remainder": a, "state_full": b_,
                             "rows": int(cell.sum()), "positives": int(y[cell].sum()),
                             "negatives": int((y[cell] == 0).sum()), "changed": a != b_,
                             "changed_rows_in_scope": changed})
    return pd.DataFrame(rows)


def _configuration(inp: S1Inputs) -> pd.DataFrame:
    p, s = inp.settings[CONFIGS[0]], inp.settings[CONFIGS[1]]
    rows = [{"setting": f"glass_config.{k}", CONFIGS[0]: json.dumps(p["glass_config"].get(k),
                                                                    sort_keys=True),
             CONFIGS[1]: json.dumps(s["glass_config"].get(k), sort_keys=True),
             "differs": p["glass_config"].get(k) != s["glass_config"].get(k)}
            for k in sorted(set(p["glass_config"]) | set(s["glass_config"]))]
    for key in ("ranker", "feature_pathway"):
        rows.append({"setting": key, CONFIGS[0]: json.dumps(p.get(key), sort_keys=True),
                     CONFIGS[1]: json.dumps(s.get(key), sort_keys=True),
                     "differs": p.get(key) != s.get(key)})
    return pd.DataFrame(rows)


def analyse_s1(inp: S1Inputs) -> Dict[str, pd.DataFrame]:
    """Every S1 table (F17 section 6, required comparisons 1-7), in memory."""
    if inp.acceptance.get("accepted") is not True or inp.acceptance.get("g6_passed") is not True:
        raise SensitivityError("S1 is not accepted (G6): no sensitivity comparison is produced")
    pv, sv = inp.primary_trace.views["glass"], inp.s1_trace.views["glass"]
    if not (pv.index.equals(sv.index) and np.array_equal(pv.y, sv.y)
            and np.array_equal(pv.fold_id, sv.fold_id)
            and inp.primary_trace.views["blackbox"] is inp.s1_trace.views["blackbox"]):
        raise SensitivityError("the S1 pair is not the same rows, labels, folds and RF routing")
    inp.reference.aligned_to(ep.PairedRouting.from_trace(inp.s1_trace))
    t: Dict[str, pd.DataFrame] = {}
    t["configuration"] = _configuration(inp)
    t["s1_acceptance_checks"] = pd.DataFrame(
        [{"group": name, **c} for name, gate in inp.acceptance["gates"].items()
         for c in gate["checks"]], columns=["group", "check", "passed", "detail"])
    t["pass2_fitting_population"], t["fold_feasibility"], t["g6_pass1_invariance"] = \
        _fit_rows(inp)
    t["routing_transitions"] = _transitions(inp)
    diag = {CONFIGS[0]: C.diagnostics_tables(inp.primary_trace.views["glass"],
                                             inp.primary_trace.views["blackbox"], inp.ref,
                                             inp.lr, inp.ebm),
            CONFIGS[1]: C.diagnostics_tables(inp.s1_trace.views["glass"],
                                             inp.s1_trace.views["blackbox"], inp.ref,
                                             inp.lr, inp.ebm)}
    within = (CONFIGS[1], CONFIGS[1], CONFIGS[0])
    t["states_comparison"] = _join(
        {c: _text(diag[c]["states"]) for c in CONFIGS}, ("arm", "fold", "state"),
        ("n", "positives", "negatives", "share", "positive_rate"),
        [(col, within[1], within[2]) for col in ("n", "positives", "share", "positive_rate")])
    t["ratios_comparison"] = _join(
        {c: _text(diag[c]["ratios"]) for c in CONFIGS}, b._RATIO_KEY,
        ("label", "numerator", "denominator", "value", "status"),
        [(col, within[1], within[2]) for col in ("numerator", "denominator", "value")])
    glass_rows = {"baseline": inp.baseline_tables["ratios"],
                  **{c: _text(diag[c]["ratios"]) for c in CONFIGS}}
    base = _join({k: f[f["arm"] == "glass"] for k, f in glass_rows.items()}, b._RATIO_KEY,
                 ("numerator", "denominator", "value"),
                 [("value", CONFIGS[1], CONFIGS[0]), ("value", CONFIGS[0], "baseline"),
                  ("value", CONFIGS[1], "baseline")])
    base["attribution"] = ("full - oof_remainder: the population policy (within S1); "
                           "either - baseline: confounded (selector correction and contract)")
    t["baseline_comparison"] = base
    t.update({f"diagnostics/{k}": v for k, v in diag[CONFIGS[1]].items()})
    return {k: C._flat(v) for k, v in t.items()}


# ----------------------------------------------------------------------
# Package
# ----------------------------------------------------------------------
DISCLOSURES = (
    "S1 is descriptive (F17 section 6): it quantifies what the Pass 2 population policy does "
    "to GLASS. The primary policy (oof_remainder) was fixed before S1 ran; S1 never chooses, "
    "replaces or alters the primary and is reported whatever its direction.",
    "Single factor: only pass2_population differs (configuration.csv). Both traces use the "
    "accepted primary's RF Router, so RF rows are unchanged; no RF counterpart is run.",
    "Within-S1 differences are full - oof_remainder, paired by fold. No significance claim is "
    "made and no interval is computed: no test is pre-specified for S1.",
    "Differences against pr37_stage2_pre_correction_v1 are confounded by the selector "
    "correction and the contract change; only the within-S1 difference is attributed to the "
    "population policy.",
    "G6 passed (s1_acceptance.json): Pass 1 is identical in both runs, so differences arise "
    "in Pass 2 and abstention only.",
    "Precision-estimation asymmetry (F17 L2): each Pass 2 rule's precision is in-sample on "
    "the population it was selected on (all fold rows under full, the remainder under "
    "oof_remainder).",
    "Evidence tier 2 (training-side outer OOF); no test label is read. Matched-budget (2C) "
    "results are descriptive, not deployable thresholds.",
)


def build_package(inp: S1Inputs, tables: Mapping[str, pd.DataFrame], *,
                  bank_root=None) -> Dict[str, bytes]:
    """Every file of the S1 package, in memory: deterministic, no timestamps or local paths."""
    files: Dict[str, bytes] = {}
    outputs: Dict[str, Any] = {}
    for name in sorted(tables):
        data = b._csv_bytes(tables[name])
        files[f"tables/{name}.csv"] = data
        outputs[f"tables/{name}.csv"] = {"sha256": hashlib.sha256(data).hexdigest(),
                                         "bytes": len(data), "rows": int(len(tables[name])),
                                         "columns": [str(c) for c in tables[name].columns]}
    code = b.code_identity(bank_root or phase0.BANK_ROOT,
                           ("stage2_crossfit/sensitivity.py", *C.CODE_FILES))
    manifest = {
        "format": FORMAT, "package": PACKAGE_DIR,
        "scope": "PR 38 Commit 3 Step 3C: G12 sensitivity S1 (GLASS pass2_population full vs "
                 "oof_remainder), descriptive, reported apart from the primary comparison",
        "identities": inp.identities,
        "s1_acceptance": {"g6_passed": inp.acceptance.get("g6_passed"),
                          "statement": inp.acceptance.get("statement")},
        "comparative_sections": {"declared_sensitivities": sorted(outputs)},
        "disclosures": list(DISCLOSURES),
        "outputs": outputs,
        "code": {"method": code.method, "files": dict(code.files)},
    }
    man = b._json_bytes(manifest)
    files[C.MANIFEST] = man
    files[C.SIDECAR] = (hashlib.sha256(man).hexdigest() + "\n").encode("ascii")
    return files


def comparative_report_status(record_dir, protocol) -> Dict[str, Any]:
    """The comparative report across the Step 3B and 3C packages (sidecar-verified manifests)."""
    sections: Dict[str, Any] = {}
    found = []
    for name in (C.PACKAGE_DIR, PACKAGE_DIR):
        man = Path(record_dir) / name / C.MANIFEST
        side = Path(record_dir) / name / C.SIDECAR
        if man.is_file() and side.is_file() and side.read_text().strip() == R.sha256_file(man):
            sections.update(A._json(man).get("comparative_sections") or {})
            found.append(name)
    return {"packages": found, "gaps": ep.comparative_report_gaps(sections, protocol)}


def report_s1(run_id: str, sums_path, *, check: bool = False, out_dir=None,
              **load_kw) -> Dict[str, Any]:
    """Gates, load, analyse, package; write once (``check``: compare only). No inputs override."""
    inp = load_s1_inputs(run_id, sums_path, **load_kw)
    tables = analyse_s1(inp)
    files = build_package(inp, tables)
    for what, (path, sha) in inp.gate_hashes.items():
        if not Path(path).is_file() or R.sha256_file(path) != sha:
            raise SensitivityError(f"the {what} changed after the gates; nothing was written")
    target = Path(out_dir) if out_dir else Path(inp.record_dir) / PACKAGE_DIR
    if check:
        status, problems = C.compare_package(target, files)
    else:
        status, problems = C.write_package(target, files), []
    return {"status": status, "problems": problems, "directory": target, "tables": tables,
            "files": files, "report": comparative_report_status(target.parent, inp.protocol)}
