"""
stage2_crossfit.acceptance
==========================
Commit 3A after ``primary``: the G7 reproduction of the pre-declared outer
fold, and structural acceptance V1-V8 of the candidate artifacts (protocol
sections 3.1, 14 and 16, steps 6-9; handoff Phase 3). Nothing here takes a
performance input or records one: label-dependent checks (the trace T1-T7,
the Stage 4 R4 orientation checks) are kept as pass / fail only, and no
outer-OOF routing count, rate or metric is written or printed.

    load_candidate    the run's MANIFEST.json: its ``.sha256``, every file's
                      SHA-256 and size, no unlisted file, byte-identical to the
                      version-controlled copy, one durable primary attempt
                      that ended in this candidate. No artifact is unpickled
                      unless all of it holds (fail closed)
    reproduce_fold    G7 on ``G7_FOLD``: one outer fold of both arms refitted
                      as ``run_crossfit`` fits it; its fit-time evidence and
                      held-out decisions, confidences and scores against the
                      candidate's (no label is read for the comparison)
    acceptance        V1-V8 from recorded checks; ``ep.structural_acceptance``
                      decides, official only under the pinned frozen protocol
    require_accepted  the Step 3B entry point: a complete, digest-intact,
                      official acceptance whose verdict, identities and every
                      bound input re-verify, for the unchanged candidate

Owner dispositions sit in the run's record directory, in
``acceptance_dispositions.json``
(``{"format", "run_id", "dispositions": {id: {"decision", "reviewer",
"rationale", "date", ...}}}``). Only an explicit satisfying ``decision``
counts: a rejection, an unknown decision, a missing one (documentation only)
or a template placeholder fails its gate.

    G10:artifact_audit_pins   F17 G10's two GLASS ``full`` pins. Satisfied by
        "pins_updated"            (both pins are checked, not taken on trust:
                                  ``g10_pin_problems``)
        "clarification_approved"  with ``record``: the identity of the versioned
                                  clarification ``records/g10_pin_scope_
                                  clarification_v1.json`` (proposed; inactive
                                  until so approved)
    F31:condition2            when ``fold_label_excluded`` is adopted (F31
                              section 7, condition 2): "disposition_a_retained"
    F31:R2:fold<k>:<status>   each RF fold the R2 strip review flags:
                              "disposition_a_retained"
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np

import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import RFRouter
from blackbox_pipeline.models.rf_router.contract import check_rf_constraints_follow_contract
from shared.stage2 import diagnostics as d
from shared.stage2.feature_engineering import engineer_features
from shared.stage_io import label_fingerprint

from . import evidence as E
from . import inner_features as IF
from . import phase0
from . import runner as R

__all__ = ["ACCEPTANCE_FORMAT", "G7_FORMAT", "DISPOSITIONS_FORMAT", "DISPOSITIONS_FILE",
           "ACCEPTANCE_FILE", "G7_FOLD", "G7_FILE", "REPRODUCTION_TOL", "DECISIONS",
           "Candidate", "load_candidate", "train_reference", "candidate_views", "compare_json",
           "reproduce_fold", "g7_problems", "load_g7", "load_dispositions", "strip_review",
           "required_dispositions", "disposition_problems", "g10_clarification",
           "upstream_identity", "g11_checks", "acceptance_inputs", "acceptance",
           "write_acceptance", "acceptance_record_problems", "require_accepted"]

ACCEPTANCE_FORMAT = "pr38.stage2.structural_acceptance/1"
G7_FORMAT = "pr38.stage2.g7_reproduction/1"
DISPOSITIONS_FORMAT = "pr38.stage2.acceptance_dispositions/1"
DISPOSITIONS_FILE = "acceptance_dispositions.json"
ACCEPTANCE_FILE = "structural_acceptance.json"
#: G7's fold, declared before any primary result: the fold the protocol
#: pre-declares for the G3 preflight and the G13 audit (section 10).
G7_FOLD = R.PREFLIGHT_FOLD
G7_FILE = f"g7_reproduction_fold{G7_FOLD}.json"
#: G7 (F17 record): thresholds agree within 1e-12 unless every fit runs with n_jobs=1.
REPRODUCTION_TOL = 1e-12
#: Evidence fields an identical fit repeated may change (timing only).
VOLATILE_KEYS = frozenset({"wall_seconds"})
#: Owner decisions per disposition kind: the ones that satisfy, the ones that refuse.
DECISIONS = {"G10": {"satisfy": ("pins_updated", "clarification_approved"),
                     "refuse": ("rejected",)},
             "F31": {"satisfy": ("disposition_a_retained",), "refuse": ("reopened",)}}
G10_CLARIFICATION_FILE, G10_CLARIFICATION_FINGERPRINT = ("g10_pin_scope_clarification_v1.json",
                                                         "11ecf7c53546c6fe")
#: The fold evidence each arm persists (G5, R1, R2, R5): what the G3 preflight required.
REQUIRED_EVIDENCE = {
    E.ARM_GLASS: ("selection_reports", "counts", "remainder", "m6_inner_oof_pass1",
                  "inner_ranker_provenance", "isolation", "fit_status", "inner_feature_pathway"),
    E.ARM_RF: ("counts", "remainder", "r1_bands", "r2_strips", "eligibility", "isolation",
               "fit_status", "inner_feature_pathway"),
}
#: The router trace's checks by stage: every T-group must be present and pass.
TRACE_COVERAGE = {"reference": ("T1",),
                  "glass": tuple(f"T{i}" for i in range(1, 8)),
                  "blackbox": tuple(f"T{i}" for i in range(1, 8)),
                  "glass x blackbox": ("T2", "T3", "T6")}
_ARMS = (E.ARM_GLASS, E.ARM_RF)
_VIEW_ARM = {E.ARM_GLASS: "glass", E.ARM_RF: "blackbox"}
_STAGE_IO_ATTESTATION = ("frozen protocol pr38-stage2-evaluation 1.0.0, "
                         "population.stage_io_split_fingerprint (attested, never recomputed)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canon(obj) -> Any:
    """JSON-safe and round-tripped, so in-memory and reloaded records compare equal."""
    return json.loads(json.dumps(E.to_json_safe(obj), sort_keys=True))


def _json(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _protocol_identity(protocol) -> List[str]:
    return [protocol.protocol_id, protocol.version, protocol.fingerprint()]


def _digest(rec: Mapping[str, Any]) -> str:
    body = {k: v for k, v in _canon(rec).items() if k != "record_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Recorded checks
# ----------------------------------------------------------------------
class _Gate:
    """Named checks; a check that raises is recorded as failed with its error."""

    def __init__(self, gate: str):
        self.gate, self.checks, self.not_run = gate, [], None

    def check(self, name: str, fn: Callable[[], Any], detail: Any = "") -> bool:
        err = None
        try:
            ok = bool(fn())
        except Exception as exc:                    # noqa: BLE001 - recorded as a failure
            ok, err = False, f"{type(exc).__name__}: {exc}"
        if err is None and callable(detail):
            try:
                detail = detail()
            except Exception as exc:                # noqa: BLE001
                detail = f"(detail unavailable: {type(exc).__name__})"
        self.checks.append({"check": name, "passed": ok,
                            "detail": err if err is not None else str(detail)})
        return ok

    @property
    def passed(self) -> bool:
        return self.not_run is None and bool(self.checks) and all(c["passed"] for c in self.checks)

    @property
    def status(self) -> str:
        return ep.GATE_NOT_RUN if self.not_run else ep.GATE_PASS if self.passed else ep.GATE_FAIL

    def result(self) -> ep.GateResult:
        if self.not_run:
            return ep.GateResult(self.gate, ep.GATE_NOT_RUN, self.not_run)
        failed = [c["check"] for c in self.checks if not c["passed"]]
        evidence = f"{len(self.checks) - len(failed)}/{len(self.checks)} checks passed"
        if failed:
            evidence += f"; failed: {failed}"
        return ep.GateResult(self.gate, self.status, evidence)

    def record(self) -> Dict[str, Any]:
        return {"gate": self.gate, "status": self.status, "passed": self.passed,
                "not_run": self.not_run, "evidence": self.result().evidence,
                "checks": list(self.checks)}


# ----------------------------------------------------------------------
# The candidate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    run_id: str
    role: str
    root: Path
    record_dir: Path
    manifest: Dict[str, Any]
    manifest_sha256: str
    glass_path: Optional[Path]
    rf_path: Optional[Path]
    evidence: Dict[str, Any]
    integrity: Tuple[Dict[str, Any], ...]

    @property
    def intact(self) -> bool:
        return bool(self.integrity) and all(c["passed"] for c in self.integrity)

    def relative(self, path: Optional[Path]) -> Optional[str]:
        return None if path is None else Path(path).relative_to(self.root).as_posix()

    def sha256(self, path: Path) -> str:
        return self.manifest["files"][self.relative(path)]["sha256"]

    def load(self, path: Path):
        """Unpickle ``path`` only if the candidate is intact and the file is its manifest's."""
        if not self.intact:
            raise R.RunnerError("the candidate failed its integrity checks; nothing is loaded")
        if R.sha256_file(path) != self.sha256(path):
            raise R.RunnerError(f"{path} changed since the candidate was loaded")
        return joblib.load(path)

    def fold_evidence(self, arm: str, k: int) -> Dict[str, Any]:
        found = [ev for ev in (self.evidence.get("folds") or {}).get(arm) or []
                 if ev.get("fold") == k]
        if len(found) != 1:
            raise R.RunnerError(f"the candidate has {len(found)} {arm} fold-{k} records")
        return found[0]


def _one(files: Mapping[str, Any], pattern: str) -> Optional[str]:
    hits = [rel for rel in files if re.fullmatch(pattern, rel)]
    return hits[0] if len(hits) == 1 else None


def load_candidate(run_id: str, *, role: str = R.ROLE_PRIMARY, output_root=None,
                   record_dir=None) -> Candidate:
    """The candidate ``primary`` wrote, every integrity check recorded; nothing unpickled."""
    root = Path(output_root or phase0.OUTPUT_ROOT) / run_id / role
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    man_path = root / "MANIFEST.json"
    if not man_path.is_file():
        raise R.RunnerError(f"no candidate MANIFEST.json at {man_path}: run `primary` first")
    g = _Gate("integrity")
    man_sha = R.sha256_file(man_path)
    manifest = _json(man_path)
    g.check("MANIFEST.json.sha256 is the SHA-256 of MANIFEST.json",
            lambda: (root / "MANIFEST.json.sha256").read_text(encoding="utf-8").strip() == man_sha,
            man_sha)
    copy = record_dir / "primary_MANIFEST.json"
    g.check("the version-controlled copy primary_MANIFEST.json is byte-identical",
            lambda: R.sha256_file(copy) == man_sha, str(copy))
    g.check("manifest format, run and role",
            lambda: (manifest.get("format"), manifest.get("run_id"), manifest.get("role"))
            == (R.RUN_FORMAT, run_id, role),
            lambda: [manifest.get("format"), manifest.get("run_id"), manifest.get("role")])
    attempt = record_dir / R.PRIMARY_ATTEMPT_FILE
    g.check("one durable primary attempt for this run and role",
            lambda: (lambda a: (a.get("format"), a.get("run_id"), a.get("role"))
                     == (R.PRIMARY_ATTEMPT_FORMAT, run_id, role))(_json(attempt)), str(attempt))
    g.check("the attempt ended in this candidate, not in a stop",
            lambda: not (record_dir / "primary_stopped.json").exists())
    files = manifest.get("files") or {}
    bad = [rel for rel, meta in files.items()
           if not (root / rel).is_file() or (root / rel).stat().st_size != meta.get("bytes")
           or R.sha256_file(root / rel) != meta.get("sha256")]
    g.check("every listed file is present with its SHA-256 and size",
            lambda: bool(files) and not bad, f"{len(files)} files; differing {bad[:5]}")
    listed = set(files) | {"MANIFEST.json", "MANIFEST.json.sha256"}
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix() not in listed)
    g.check("no unlisted file in the candidate directory", lambda: not extra, f"unlisted {extra[:5]}")
    stem = f"pr38_{re.escape(role)}_{re.escape(run_id)}"
    glass = _one(files, rf"glass_router/glass_router_{stem}\.joblib")
    rf = _one(files, rf"rf_router/rf_router_{stem}_\d{{8}}_\d{{6}}_\d{{6}}\.joblib")
    g.check("one GLASS bundle and one immutable RF Router version, named for the run",
            lambda: glass is not None and rf is not None, f"{glass}, {rf}")
    evidence = {}
    if "evidence.json" in files and "evidence.json" not in bad:     # only a verified file
        evidence = _json(root / "evidence.json")
    g.check("evidence.json is the run's",
            lambda: (evidence.get("format"), evidence.get("run_id"), evidence.get("role"))
            == (R.RUN_FORMAT, run_id, role))
    return Candidate(run_id, role, root, record_dir, manifest, man_sha,
                     None if glass is None else root / glass, None if rf is None else root / rf,
                     evidence, tuple(g.checks))


def train_reference(ts: R.TrainSide, protocol=None) -> d.TrainReference:
    """The boundary's reference: training rows and labels, test row ids only."""
    kw = {}
    if protocol is not None:
        kw = {"attested_stage_io_split_fingerprint":
              protocol["population"]["stage_io_split_fingerprint"],
              "attestation_source": _STAGE_IO_ATTESTATION}
    return d.build_train_reference(index_train=ts.X_train.index, y_train=ts.y_train,
                                   index_test=ts.X_test.index, **kw)


def candidate_views(candidate: Candidate, ref) -> Dict[str, Any]:
    """Both outer-OOF views through the boundary extractor, each file at its manifest SHA-256."""
    if not candidate.intact:
        raise R.RunnerError("the candidate failed its integrity checks; nothing is loaded")
    return {E.ARM_GLASS: d.extract_router_view(candidate.glass_path, ref, expected_arm="glass",
                                               expected_sha256=candidate.sha256(candidate.glass_path)),
            E.ARM_RF: d.extract_router_view(candidate.rf_path, ref, expected_arm="blackbox",
                                            expected_sha256=candidate.sha256(candidate.rf_path))}


# ----------------------------------------------------------------------
# G7: reproduction of the pre-declared outer fold
# ----------------------------------------------------------------------
def compare_json(a, b, *, tol: float = REPRODUCTION_TOL,
                 ignore: frozenset = VOLATILE_KEYS) -> Dict[str, Any]:
    """Leaf-by-leaf comparison of two JSON records; floats within ``tol``, all else exact."""
    diffs: List[str] = []
    worst = [0.0]

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted((set(x) ^ set(y)) - ignore):
                diffs.append(f"{path}/{k}: present in one record only")
            for k in sorted((set(x) & set(y)) - ignore):
                walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path}: length {len(x)} vs {len(y)}")
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif (isinstance(x, float) or isinstance(y, float)) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
            gap = abs(float(x) - float(y))
            worst[0] = max(worst[0], gap)
            if not gap <= tol:
                diffs.append(f"{path}: {x!r} vs {y!r}")
        elif x != y:
            diffs.append(f"{path}: {x!r} vs {y!r}")

    walk(_canon(a), _canon(b), "")
    return {"identical_within_tolerance": not diffs, "bit_identical": not diffs and worst[0] == 0.0,
            "max_float_difference": worst[0], "tolerance": tol, "n_differences": len(diffs),
            "differences": diffs[:50]}


def _held_out(view, va, decisions, confidence, proba) -> Dict[str, Any]:
    """The refit's held-out outputs against the candidate's stored ones (no label read)."""
    stored = np.asarray(view.source_decision, dtype=str)[va]
    got = np.asarray(decisions, dtype=str)
    score = np.asarray(proba, dtype=float)
    score = score[:, 1] if score.ndim == 2 else score
    dc = np.abs(np.asarray(confidence, float) - np.asarray(view.confidence, float)[va])
    dp = np.abs(score - np.asarray(view.router_score.values, float)[va])
    worst = float(max(dc.max(initial=0.0), dp.max(initial=0.0)))
    same = got.shape == stored.shape and bool((got == stored).all())
    return {"rows": int(len(va)), "decisions_identical": same,
            "decisions_differing": int((got != stored).sum()) if got.shape == stored.shape else None,
            "max_confidence_difference": float(dc.max(initial=0.0)),
            "max_score_difference": float(dp.max(initial=0.0)),
            "passed": same and worst <= REPRODUCTION_TOL, "bit_identical": same and worst == 0.0}


def reproduce_fold(ts: R.TrainSide, settings: R.ArmSettings, candidate: Candidate,
                   k: int = G7_FOLD, *, verbose: bool = False) -> Dict[str, Any]:
    """
    G7: refit outer fold ``k`` of both arms exactly as ``run_crossfit`` does,
    rebuild its evidence and route its held-out rows; compare both with the
    candidate. Identical rule sets, remainders, selections and decisions are
    required; thresholds, confidences and scores agree within ``REPRODUCTION_TOL``
    and ``bit_identical`` says whether they agree exactly. Only ``G7_FOLD``
    counts for acceptance (``g7_problems``).
    """
    if not candidate.intact:
        raise R.RunnerError("the candidate failed its integrity checks; nothing to reproduce")
    _, va = ts.folds[k]
    held = ts.X_train.index[va]
    views = candidate_views(candidate, train_reference(ts))
    arms: Dict[str, Any] = {}

    t0 = time.perf_counter()
    fit = R._glass_fold(ts, settings, k, verbose)
    ev = E.glass_fold_evidence(fit.glass, fit.X_tr_eng, fit.y_tr, fold=k, heldout_index=held,
                               inner_features=fit.inner_features)
    ev = R.document_unproven(ev, lambda: R._glass_fold(ts, settings, k, verbose).glass)
    ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
    with R._quiet(verbose):
        _, conf, dec = fit.glass.predict(fit.X_va_eng)
        proba = fit.glass.predict_proba(fit.X_va_eng)
    arms[E.ARM_GLASS] = {"evidence": compare_json(ev, candidate.fold_evidence(E.ARM_GLASS, k)),
                         "held_out": _held_out(views[E.ARM_GLASS], va, dec, conf, proba),
                         "wall_seconds": round(time.perf_counter() - t0, 1)}

    t0 = time.perf_counter()
    X_tr, y_tr, X_va = ts.fold_rows(k)
    with R._quiet(verbose):
        X_tr_eng, X_va_eng = engineer_features(X_tr, X_va, y_tr)
        router = RFRouter(settings.rf_config_factory())
        provider = R._rf_fit(router, X_tr_eng, X_tr, y_tr, ts.router_split_fingerprint, settings)
        _, conf, dec, proba = router.predict_with_proba(X_va_eng)
    ev = E.rf_fold_evidence(router, X_tr_eng, y_tr, fold=k, heldout_index=held,
                            inner_features=provider)
    ev["fit_status"] = E.fold_status(ev, settings.contract, settings.protocol)
    arms[E.ARM_RF] = {"evidence": compare_json(ev, candidate.fold_evidence(E.ARM_RF, k)),
                      "held_out": _held_out(views[E.ARM_RF], va, dec, conf, proba),
                      "wall_seconds": round(time.perf_counter() - t0, 1)}
    for a in arms.values():
        a["passed"] = a["evidence"]["identical_within_tolerance"] and a["held_out"]["passed"]
        a["bit_identical"] = a["evidence"]["bit_identical"] and a["held_out"]["bit_identical"]
    return {"format": G7_FORMAT, "run_id": candidate.run_id, "role": candidate.role, "fold": k,
            "pre_declared_fold": k == G7_FOLD,
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "feature_pathway": IF.pathway_identity(settings.feature_pathway),
            "tolerance": REPRODUCTION_TOL, "ignored_keys": sorted(VOLATILE_KEYS),
            "arms": arms, "passed": all(a["passed"] for a in arms.values()),
            "bit_identical": all(a["bit_identical"] for a in arms.values()),
            "note": "identity only: decisions, confidences and scores of the held-out rows are "
                    "compared with the candidate's; no label is read and no metric computed"}


def g7_problems(g7: Optional[Mapping[str, Any]], candidate: Candidate, settings: R.ArmSettings,
                *, held_out_rows: Optional[int] = None) -> List[str]:
    """Why ``g7`` is not the passed G7 record of this candidate's pre-declared fold."""
    if not isinstance(g7, Mapping):
        return [f"no G7 record {G7_FILE}"]
    out = []
    want = {"format": G7_FORMAT, "run_id": candidate.run_id, "role": candidate.role,
            "fold": G7_FOLD, "pre_declared_fold": True,
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "tolerance": REPRODUCTION_TOL, "ignored_keys": sorted(VOLATILE_KEYS)}
    out += [f"{k}: {g7.get(k)!r}, expected {v!r}" for k, v in want.items() if g7.get(k) != v]
    if not _canon(g7.get("feature_pathway")) == _canon(
            IF.pathway_identity(settings.feature_pathway)):
        out.append(f"feature_pathway {g7.get('feature_pathway')!r} is not the run's")
    arms = g7.get("arms") or {}
    if set(arms) != set(_ARMS):
        return out + [f"arms {sorted(arms)}, expected {sorted(_ARMS)}"]
    for arm in _ARMS:
        a, ev, ho = arms[arm], arms[arm].get("evidence") or {}, arms[arm].get("held_out") or {}
        if not (ev.get("identical_within_tolerance") is True and ev.get("n_differences") == 0
                and float(ev.get("max_float_difference", np.inf)) <= REPRODUCTION_TOL):
            out.append(f"{arm}: the refit's fit-time evidence differs from the candidate's")
        if not (ho.get("decisions_identical") is True and ho.get("decisions_differing") == 0
                and float(ho.get("max_confidence_difference", np.inf)) <= REPRODUCTION_TOL
                and float(ho.get("max_score_difference", np.inf)) <= REPRODUCTION_TOL
                and ho.get("passed") is True):
            out.append(f"{arm}: the refit's held-out outputs differ from the candidate's")
        if held_out_rows is not None and ho.get("rows") != held_out_rows:
            out.append(f"{arm}: {ho.get('rows')} held-out rows, expected {held_out_rows}")
        if a.get("passed") is not True:
            out.append(f"{arm}: not passed")
    if g7.get("passed") is not True or not all(arms[a].get("passed") is True for a in _ARMS):
        out.append("the record is not passed in both arms")
    return out


def load_g7(record_dir) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The pre-declared fold's G7 record (or None) and any other G7 records present."""
    record_dir = Path(record_dir)
    path = record_dir / G7_FILE
    others = sorted(p.name for p in record_dir.glob("g7_reproduction_fold*.json")
                    if p.name != G7_FILE)
    return (_json(path) if path.is_file() else None), others


# ----------------------------------------------------------------------
# Owner dispositions
# ----------------------------------------------------------------------
def load_dispositions(record_dir, run_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """The owner's acceptance dispositions (``{}`` if none) and the file's SHA-256."""
    path = Path(record_dir) / DISPOSITIONS_FILE
    if not path.is_file():
        return {}, None
    rec = json.loads(path.read_text(encoding="utf-8-sig"))     # owner-written: BOM tolerated
    if rec.get("format") != DISPOSITIONS_FORMAT or rec.get("run_id") != run_id:
        raise R.RunnerError(f"{path}: expected format {DISPOSITIONS_FORMAT!r} for run {run_id!r}")
    return dict(rec.get("dispositions") or {}), R.sha256_file(path)


def strip_review(evidence: Mapping[str, Any], protocol) -> List[Dict[str, Any]]:
    """F31 R2 trigger review over the RF fold routers' recorded strips (fit-time, inner OOF)."""
    folds = {int(ev["fold"]): (ev["r2_strips"]["strip_a_in_band"]["rows"],
                               ev["r2_strips"]["band_rows"])
             for ev in (evidence.get("folds") or {}).get(E.ARM_RF) or [] if "r2_strips" in ev}
    return [] if not folds else ep.f31_strip_review(folds, protocol).to_dict("records")


def required_dispositions(candidate: Candidate, adopted_pathway: Optional[str],
                          protocol) -> List[str]:
    out = ["G10:artifact_audit_pins"]
    if adopted_pathway == IF.PATHWAY_FOLD:
        out.append("F31:condition2")
    out += [f"F31:R2:fold{r['fold']}:{r['status']}"
            for r in strip_review(candidate.evidence, protocol) if r["review_required"]]
    return out


def g10_clarification() -> Dict[str, Any]:
    """The proposed G10 pin-scope clarification, at its pinned fingerprint."""
    return E.load_versioned_record(G10_CLARIFICATION_FILE, G10_CLARIFICATION_FINGERPRINT)


def _placeholder(value) -> bool:
    s = str(value).strip()
    return not s or (s.startswith("<") and s.endswith(">"))


def disposition_problems(ident: str, entry: Any) -> List[str]:
    """Why ``entry`` is not the owner's explicit, satisfying disposition of ``ident``."""
    kind = ident.split(":", 1)[0]
    if kind not in DECISIONS:
        return [f"{ident}: unknown disposition"]
    if not isinstance(entry, Mapping):
        return [f"{ident}: not recorded in {DISPOSITIONS_FILE}"]
    missing = [k for k in ("decision", "reviewer", "rationale", "date")
               if not isinstance(entry.get(k), str) or _placeholder(entry[k])]
    if missing:
        return [f"{ident}: {missing} missing or a template placeholder (a rationale alone "
                "documents; it does not decide)"]
    decision, allowed = entry["decision"], DECISIONS[kind]
    if decision in allowed["refuse"]:
        return [f"{ident}: the owner's decision is {decision!r}; the gate cannot pass"]
    if decision not in allowed["satisfy"]:
        return [f"{ident}: decision {decision!r} is not one of {list(allowed['satisfy'])}"]
    if decision == "clarification_approved":
        want = E.record_identity(g10_clarification())
        if entry.get("record") != want:
            return [f"{ident}: 'clarification_approved' names record {entry.get('record')!r}, "
                    f"not {want}"]
    if decision == "pins_updated":
        pins = g10_pin_problems()
        if pins:
            return [f"{ident}: 'pins_updated', but " + "; ".join(pins)]
    return []


#: F17 G10's second pin, and the notebook pathway it pins (handoff section 5).
G10_PIN_TEST = ("tests/parity/test_router_operating_contract.py",
                "test_the_glass_notebook_runs_the_full_population_profile_for_both_fits")
GLASS_NOTEBOOK = "glass_pipeline/glass_cascade.ipynb"


def _active_glass_profile(src: str) -> Optional[str]:
    """The ``dict(...)`` of the GLASS notebook's active ``CONFIG_PROFILE``, or None."""
    m = re.search(r'CONFIG_PROFILE\s*=\s*"([^"]+)"', src)
    start = -1 if m is None else src.find(f'"{m.group(1)}": dict(')
    if start < 0:
        return None
    depth, i = 0, src.index("(", start)
    for j in range(i, len(src)):
        depth += {"(": 1, ")": -1}.get(src[j], 0)
        if depth == 0:
            return src[i:j + 1]
    return None


def g10_pin_problems() -> List[str]:
    """
    Why F17 G10's two GLASS ``full`` pins are not both established as updated:
    ``tests/support/artifact_audit.py`` must expect GLASS ``oof_remainder``; the
    pin test (``G10_PIN_TEST``) must be defined exactly once and no longer
    assert the ``full`` profile (missing, renamed or duplicated: refused), and
    the GLASS notebook's active profile, which it pins, must declare
    ``oof_remainder``. Anything unreadable is a problem (fail closed).
    """
    import ast
    out = []
    try:
        from tests.support import artifact_audit
        got = artifact_audit.PASS2_POPULATION.get("glass")
        if got != "oof_remainder":
            out.append(f"tests/support/artifact_audit.py still expects GLASS {got!r}")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"tests/support/artifact_audit.py: {type(exc).__name__}: {exc}")
    path, name = G10_PIN_TEST
    try:
        tree = ast.parse((phase0.BANK_ROOT / path).read_text(encoding="utf-8"))
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(fn) != 1:
            out.append(f"{path}::{name}: {len(fn)} definitions, expected exactly one (a missing, "
                       "renamed or duplicated pin test cannot establish the G10 pin)")
        elif any(isinstance(c, ast.Constant) and c.value == 'pass2_population="full"'
                 for c in ast.walk(fn[0])):
            out.append(f"{path}::{name} still asserts the GLASS 'full' profile")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"{path}: {type(exc).__name__}: {exc}")
    try:
        nb = json.loads((phase0.BANK_ROOT / GLASS_NOTEBOOK).read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        profile = _active_glass_profile(src)
        if profile is None or 'pass2_population="oof_remainder"' not in profile:
            out.append(f"{GLASS_NOTEBOOK}: the active CONFIG_PROFILE does not declare "
                       "pass2_population=\"oof_remainder\"")
    except Exception as exc:                        # noqa: BLE001
        out.append(f"{GLASS_NOTEBOOK}: {type(exc).__name__}: {exc}")
    return out


def _disposition_check(gate: _Gate, dispositions: Mapping[str, Any], ident: str, why: str):
    problems = disposition_problems(ident, dispositions.get(ident))
    gate.check(f"owner disposition {ident} ({why})", lambda: not problems,
               "; ".join(problems) or f"decision {dispositions[ident]['decision']!r}")


# ----------------------------------------------------------------------
# Upstream identity, G11 and the bound inputs
# ----------------------------------------------------------------------
def upstream_identity(trace, ref, protocol, sums_path) -> ep.IdentityReport:
    """The frozen LR / EBM operating points (protocol section 5), read from the archive."""
    archive = Path(sums_path).parent / "bank_pipeline"
    arts = protocol["upstream"]["artifacts"]
    lr = d.extract_lr_view(archive / arts["lr"]["path"], ref, expected_sha256=arts["lr"]["sha256"])
    ebm = d.extract_ebm_view(archive / arts["ebm"]["path"], ref,
                             expected_sha256=arts["ebm"]["sha256"])
    return ep.check_upstream_identity(d.primary_reference_flags(lr, ebm), protocol,
                                      lr_view=lr, ebm_view=ebm, population=trace)


def g11_checks(sums_path, artifact_manifest) -> Dict[str, Any]:
    """Phase 0 checks 0.1-0.3 repeated after the run (read-only): G11."""
    return {"snapshot": phase0.verify_snapshot(sums_path),
            "baseline": phase0.run_baseline_checks(sums_path),
            "artifact_manifest": phase0.check_artifact_manifest(artifact_manifest, sums_path),
            "baseline_manifest_sha256": R.sha256_file(phase0.BASELINE_DIR / "MANIFEST.json")}


#: Record-directory files an acceptance binds by SHA-256; all but dispositions are required.
_BOUND = {"authorization": None, "g7": G7_FILE, "dispositions": DISPOSITIONS_FILE,
          "primary_attempt": R.PRIMARY_ATTEMPT_FILE, "primary_manifest": "primary_MANIFEST.json",
          "phase0": "phase0.json"}


def acceptance_inputs(record_dir, authorization_path, sums_path, artifact_manifest
                      ) -> Dict[str, Any]:
    """Every input of an acceptance, bound by name (in the run's record directory) and SHA-256."""
    record_dir = Path(record_dir)
    out = {}
    for key, name in _BOUND.items():
        name = name or Path(authorization_path).name
        p = record_dir / name
        out[key] = {"name": name, "sha256": R.sha256_file(p)} if p.is_file() else None
    out["sums_sha256"] = R.sha256_file(sums_path)
    out["artifact_manifest_sha256"] = R.sha256_file(artifact_manifest)
    return out


def _bound_problems(inputs: Mapping[str, Any], record_dir: Path, *,
                    dispositions_required: bool) -> List[str]:
    out = []
    for key in _BOUND:
        ref = (inputs or {}).get(key)
        if ref is None:
            if key != "dispositions" or dispositions_required:
                out.append(f"input {key}: not bound")
            continue
        p = record_dir / str(ref.get("name"))
        if Path(str(ref.get("name"))).name != ref.get("name") or not p.is_file():
            out.append(f"input {key}: {ref.get('name')!r} is not a file in the record directory")
        elif R.sha256_file(p) != ref.get("sha256"):
            out.append(f"input {key}: {ref['name']} changed since the acceptance (SHA-256)")
    try:
        snap = _json(phase0.BASELINE_DIR / "MANIFEST.json")["snapshot"]["sha256"]
        if (inputs or {}).get("sums_sha256") != snap:
            out.append("input sums: not the snapshot SHA256SUMS the baseline records")
        man = (_json(record_dir / "phase0.json").get("artifact_manifest") or {}).get(
            "manifest_sha256")
        if not man or (inputs or {}).get("artifact_manifest_sha256") != man:
            out.append("input artifact_manifest: not the manifest pinned at this run's Phase 0")
    except (OSError, ValueError, KeyError) as exc:
        out.append(f"inputs: {type(exc).__name__}: {exc}")
    return out


def _fresh_reload(paths: Sequence[Path]) -> Dict[str, Any]:
    """Each file unpickled in a new interpreter: its type and top-level keys."""
    code = ("import json, sys, joblib\n"
            "out = {}\n"
            "for p in sys.argv[1:]:\n"
            "    a = joblib.load(p)\n"
            "    out[p] = [type(a).__name__, sorted(map(str, a)) if isinstance(a, dict) else None]\n"
            "print(json.dumps(out))\n")
    proc = subprocess.run([sys.executable, "-c", code, *map(str, paths)], cwd=phase0.BANK_ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class _RouterOnlyInputs:
    """Meta-XGB's feature-block inputs with only the router stream real (R4)."""

    def __init__(self, ts: R.TrainSide, router_stream):
        stub = type("Placeholder", (), {})()
        stub.train_proba, stub.test_proba = np.zeros(len(ts.X_train)), np.zeros(len(ts.X_test))
        stub.train_source = stub.test_source = "placeholder (R4 checks the router columns only)"
        self.streams = {"mlp": stub, "rf_router": router_stream, "xgb": stub}
        self.index_train, self.index_test = ts.X_train.index, ts.X_test.index

    def __getitem__(self, name):
        return self.streams[name]


def _r4_checks(gate: _Gate, arts: Mapping[str, Any], ts: R.TrainSide) -> Dict[str, Any]:
    """
    R4: the Stage 4 consumers' own checks that read only the router and the
    training labels, for both arms (either can fill the 2x2's router slot).
    The Stage 4 input builder also reads the test labels (``split_reference``),
    which no PR 38 step may, so it is not run here. A consumer that raises is a
    recorded V7 failure (``R4 <consumer> [<arm>] ran``), never an abort; the
    consumers that need its output are recorded as not run.
    """
    from blackbox_pipeline.models.meta_xgb.inputs import build_feature_frames
    from glass_pipeline.meta_ebm.loader import _router_evidence_checks
    from shared.stage4 import router_stream
    y = ts.y_train.to_numpy()
    notes = {}

    def run(consumer, arm, call):
        try:
            return call()
        except Exception as exc:                    # noqa: BLE001 - recorded, never raised
            gate.check(f"R4 {consumer} [{arm}] ran", lambda: False,
                       f"{type(exc).__name__}: {exc}")
            return None

    for arm in _ARMS:
        art = arts[arm]
        got = run("router_stream", arm, lambda: router_stream(art, _VIEW_ARM[arm],
                                                              _VIEW_ARM[arm], y))
        if got is None:
            for consumer in ("meta-EBM router evidence", "meta-XGB feature block"):
                gate.check(f"R4 {consumer} [{arm}] ran", lambda: False,
                           "not run: it reads the router_stream output, which failed")
            continue
        stream, checks = got
        notes[arm] = {k: stream.notes.get(k) for k in ("pass2_population", "pass2_eligibility")}
        for _, _, name, fn, _ in checks:            # pass / fail only: no rate is recorded
            gate.check(f"R4 router_stream [{arm}] {name}", fn)
        # F25 (native GLASS) and F29 (RF in the crossover): the arbiter's blend read-back
        evidence = run("meta-EBM router evidence", arm,
                       lambda: _router_evidence_checks(art, stream, y, _VIEW_ARM[arm]))
        for _, _, name, fn, _ in evidence or ():
            gate.check(f"R4 meta-EBM router evidence [{arm}] {name}", fn)
        frames = run("meta-XGB feature block", arm,
                     lambda: build_feature_frames(_RouterOnlyInputs(ts, stream)))
        for _, _, name, fn, _ in (frames or (None, None, ()))[2]:
            if "stage1" in name or "stage3" in name or "no other missing" in name:
                continue                            # placeholder columns: not checked here
            gate.check(f"R4 meta-XGB feature block [{arm}] {name}", fn)
    return notes


# ----------------------------------------------------------------------
# V1-V8
# ----------------------------------------------------------------------
#: Check names every gate of a complete acceptance holds (``startswith``), per arm / fold
#: where ``{arm}`` / ``{k}`` appear; ``require_accepted`` refuses a record without them.
MANDATORY_CHECKS = {
    "V1": ("the train side is the frozen protocol's population",
           "boundary [{arm}] ", "trace coverage [{stage}] ", "trace [glass] T7", "trace [blackbox] T7",
           "check_population_identity, complete", "check_upstream_identity, complete"),
    "V2": ("the run's settings are the frozen primary", "the Step 2B authorization re-verified",
           "the manifest's identities", "both artifacts record the settings",
           "the GLASS bundle's config_full", "the RF Router's config",
           "RF constraints follow the bound contract"),
    "V3": ("{arm}: every attempted fold has a record", "{arm}: each fold re-classifies",
           "FoldLedger: every attempted fold accepted", "{arm}: the full-train fit's status",
           "the artifacts embed exactly the evidence.json"),
    "V4": ("{arm} fold {k}: fitted on exactly", "{arm} fold {k}: isolation clean",
           "{arm} fold {k}: remainder rows inside", "G13: every fit ran on the adopted",
           "no test label or test metric is stored"),
    "V5": ("{arm}: every fold persists", "{arm}: union counts recorded", "R1: each fold router",
           "R1: the full-train router", "R2: strip records", "R5: nested_t1"),
    "V6": ("G7: the pre-declared fold's reproduction", "G7: no reproduction of another fold"),
    "V7": ("candidate integrity", "both artifacts load", "new versioned location",
           "both artifacts reload in a fresh interpreter", "G2: GLASS oof_remainder",
           "R4 router_stream [{arm}] ", "R4 meta-EBM router evidence [{arm}] ",
           "R4 meta-XGB feature block [{arm}] ", "R5 at Stage 4",
           "G11: snapshot", "G11: baseline", "G11: artifact_manifest",
           "G11: the baseline MANIFEST.json", "G11: the pinned artifact manifest",
           "owner disposition G10:artifact_audit_pins"),
    "V8": ("F31 A:", "R7:", "R8:", "the R2 clarification"),
}


def _mandatory(protocol) -> Dict[str, List[str]]:
    folds = [int(k) for k in protocol["folds"]["attempted"]]
    out = {}
    for gate, names in MANDATORY_CHECKS.items():
        expanded = []
        for n in names:
            arms = _ARMS if "{arm}" in n else (None,)
            ks = folds if "{k}" in n else (None,)
            stages = TRACE_COVERAGE if "{stage}" in n else (None,)
            expanded += [n.format(arm=a, k=k, stage=s) for a in arms for k in ks for s in stages]
        out[gate] = expanded
    return out


def _same(a, b) -> bool:
    return _canon(a) == _canon(b)


def _pathways_ok(evs, adopted: Optional[str]) -> bool:
    """
    Every fit's recorded pathway identity (``pathway``, ``record``) is the
    adopted one. Under the correction a fit records the provider's full
    provenance (that identity plus fingerprints); the identity fields are
    compared, as G3 compares them (``runner``).
    """
    want = None if adopted is None else IF.pathway_identity(adopted)
    return bool(evs) and want is not None and all(
        isinstance(ev.get("inner_feature_pathway"), Mapping)
        and _same({k: ev["inner_feature_pathway"].get(k) for k in want}, want) for ev in evs)


def _trace_checks(v: _Gate, trace) -> None:
    """Every T1-T7 check the trace ran, one by one, and the T-group coverage per stage."""
    recs = [dict(r) for r in trace.checks]
    for r in recs:
        v.check(f"trace [{r.get('stage')}] {r.get('check')}", lambda r=r: r.get("passed") is True)
    for stage, groups in TRACE_COVERAGE.items():
        have = {str(r.get("check")).split(" ", 1)[0] for r in recs if r.get("stage") == stage}
        v.check(f"trace coverage [{stage}] {'/'.join(groups)} all evaluated",
                lambda have=have, groups=groups: set(groups) <= have, lambda have=have: sorted(have))


def _base_record(candidate: Candidate, settings: R.ArmSettings, inputs) -> Dict[str, Any]:
    return {"format": ACCEPTANCE_FORMAT, "run_id": candidate.run_id, "role": candidate.role,
            "at_utc": _now(), "protocol": _protocol_identity(settings.protocol),
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "artifacts": {E.ARM_GLASS: candidate.relative(candidate.glass_path),
                          E.ARM_RF: candidate.relative(candidate.rf_path)},
            "feature_pathway": IF.pathway_identity(settings.feature_pathway),
            "inputs": dict(inputs or {})}


def _finish(rec: Dict[str, Any], gates: Mapping[str, _Gate], protocol) -> Dict[str, Any]:
    sa = ep.structural_acceptance([g.result() for g in gates.values()], protocol)
    rec.update(gates={g: gates[g].record() for g in gates},
               structural_acceptance={"accepted": sa.accepted, "official": sa.official,
                                      "passed": list(sa.passed), "failed": list(sa.failed),
                                      "not_run": list(sa.not_run), "missing": list(sa.missing),
                                      "note": sa.note},
               accepted=bool(sa.accepted and sa.official),
               no_performance="no outer-OOF routing count, rate or metric was computed for "
                              "this record beyond pass / fail of the registered checks")
    return rec


def acceptance(candidate: Candidate, ts: Optional[R.TrainSide], settings: R.ArmSettings, *,
               authorization: Mapping[str, Any], inputs: Optional[Mapping[str, Any]],
               g7: Optional[Mapping[str, Any]], other_g7: Sequence[str] = (),
               dispositions: Mapping[str, Any],
               upstream: Optional[Callable[[Any, Any], ep.IdentityReport]],
               g11: Optional[Mapping[str, Any]], reload: bool = True) -> Dict[str, Any]:
    """
    V1-V8 for the candidate, every check recorded, and ``ep.structural_acceptance``
    over the eight results. ``authorization`` is the record ``require_authorization``
    returned for this run; ``inputs`` is ``acceptance_inputs``; ``upstream(trace,
    ref)`` returns the upstream identity report; ``g11`` is ``g11_checks``. A
    candidate that fails integrity is refused before anything is loaded: V7
    fails with the integrity checks and every other gate is ``not_run``.
    """
    protocol, contract = settings.protocol, settings.contract
    gates = {g: _Gate(g) for g in protocol["acceptance"]["structural"]}
    rec = _base_record(candidate, settings, inputs)
    if not candidate.intact:                        # fail closed: no artifact, no data
        for c in candidate.integrity:
            gates["V7"].check(f"candidate integrity: {c['check']}", lambda c=c: c["passed"],
                              c["detail"])
        for g, gate in gates.items():
            if g != "V7":
                gate.not_run = "candidate integrity failed: nothing was loaded"
        return _finish(rec, gates, protocol)

    adopted = ((authorization or {}).get("g13_decision") or {}).get("adopted_pathway")
    ev_all = candidate.evidence
    folds = {arm: list((ev_all.get("folds") or {}).get(arm) or []) for arm in _ARMS}
    full = ev_all.get("full_train") or {}
    attempted = [int(k) for k in protocol["folds"]["attempted"]]
    arts, load_error = {}, None
    try:
        arts = {E.ARM_GLASS: candidate.load(candidate.glass_path),
                E.ARM_RF: candidate.load(candidate.rf_path)}
    except Exception as exc:                        # noqa: BLE001 - V7 records the failure
        load_error = f"{type(exc).__name__}: {exc}"
    g_art, r_art = arts.get(E.ARM_GLASS) or {}, arts.get(E.ARM_RF) or {}
    r_extra = r_art.get("extra") or {}

    # V1: population identity (T1-T7, the pinned population) and the upstream identity
    v = gates["V1"]
    pop = R.check_population(ts, protocol)
    v.check("the train side is the frozen protocol's population (rows, labels, fold plan)",
            lambda: pop["passed"], lambda: {k: c["got"] for k, c in pop["checks"].items()})
    ref = trace = None
    views: Dict[str, Any] = {}
    try:
        ref = train_reference(ts, protocol)
        views = candidate_views(candidate, ref)
        for arm, view in views.items():
            for c in view.source["checks"]:         # every mandatory boundary check, by name
                v.check(f"boundary [{arm}] {c['check']}", lambda c=c: c["passed"] is True)
        trace = d.build_router_trace(views[E.ARM_GLASS], views[E.ARM_RF], ref)
    except Exception as exc:                        # noqa: BLE001
        v.check("both views extract and the router trace is built", lambda: False,
                f"{type(exc).__name__}: " + " | ".join(str(exc).splitlines()[:12]))
        report = getattr(exc, "report", None)       # Boundary / TraceCheckError: every check
        if hasattr(report, "to_dict"):
            for r in report.to_dict("records"):
                v.check(f"{'trace' if isinstance(exc, d.TraceCheckError) else 'boundary'} "
                        f"[{r.get('stage')}] {r.get('check')}", lambda r=r: r.get("passed") is True)
    if trace is not None:
        _trace_checks(v, trace)
        rep = ep.check_population_identity(trace, ep.PopulationIdentity.from_protocol(protocol))
        v.check("check_population_identity, complete", lambda: rep.passed and rep.complete,
                lambda: rep.failures() + rep.pending() or "every check passed")
        up, up_error = None, "not run"
        if upstream is not None:
            try:
                up = upstream(trace, ref)
            except Exception as exc:                # noqa: BLE001
                up_error = f"{type(exc).__name__}: {exc}"
        v.check("check_upstream_identity, complete (D-C10 fingerprints included)",
                lambda: up is not None and up.passed and up.complete,
                lambda: up_error if up is None else (up.failures() + up.pending()
                                                     or "every check passed"))

    # V2: configuration and contract (G1, G9)
    v = gates["V2"]
    now = _canon(R.settings_record(settings))
    ids = settings.checks.get("identities") or {}
    v.check("the run's settings are the frozen primary (role, frozen identities)",
            lambda: settings.role == R.ROLE_PRIMARY and set(ids) >= {
                "contract", "protocol", "glass_profile", "rf_profile"}, ids)
    v.check("the Step 2B authorization re-verified for this run and pathway",
            lambda: bool(authorization) and adopted == settings.feature_pathway, adopted)
    v.check("the manifest's identities are the settings' identities",
            lambda: _same(candidate.manifest.get("identities"), ids))
    v.check("both artifacts record the settings built now",
            lambda: _same((g_art.get("pr38_evidence") or {}).get("settings"), now)
            and _same((r_extra.get("pr38_evidence") or {}).get("settings"), now))
    v.check("the GLASS bundle's config_full is the primary GLASS config",
            lambda: _same(g_art.get("config_full"), settings.glass_config.to_dict()))
    v.check("the RF Router's config is the primary RF config",
            lambda: _same(r_art.get("config"), settings.rf_config_factory().to_dict()))

    def rf_follows_contract():
        check_rf_constraints_follow_contract(r_art["constraints"], contract, extra=r_extra)
        return r_extra.get("constraint_source") == "shared_contract"
    v.check("RF constraints follow the bound contract (constraint_source shared_contract)",
            rf_follows_contract, lambda: r_extra.get("contract_fingerprint"))

    # V3: fit-time compliance and accepted fit outcomes (M3, G8)
    v = gates["V3"]
    fits: Dict[str, Dict[int, Any]] = {}
    table: Dict[str, Dict[int, str]] = {}
    for arm in _ARMS:
        fits[arm], table[arm] = {}, {}
        for ev in folds[arm]:
            fit = E.classify_fold(ev, contract, protocol)
            fits[arm][int(ev["fold"])] = fit if fit is not None else ev["fit_status"]["status"]
            table[arm][int(ev["fold"])] = fit.status if fit is not None else (
                ev["fit_status"]["status"])
        v.check(f"{arm}: every attempted fold has a record",
                lambda arm=arm: sorted(fits[arm]) == attempted, lambda arm=arm: sorted(fits[arm]))
        v.check(f"{arm}: each fold re-classifies from its evidence to its recorded status",
                lambda arm=arm: all(ev.get("fit_status", {}).get("status") == table[arm].get(
                    int(ev["fold"])) for ev in folds[arm]))
    ledger = None
    try:
        ledger = ep.FoldLedger.build(fits, protocol).require_complete(list(_ARMS))
        v.check("FoldLedger: every attempted fold accepted in both arms (D-C6)", lambda: True,
                lambda: table)
    except ep.ProtocolError as exc:
        v.check("FoldLedger: every attempted fold accepted in both arms (D-C6)", lambda: False,
                str(exc).replace("\n", " "))
    for arm in _ARMS:
        fev = full.get(arm) or {}
        v.check(f"{arm}: the full-train fit's status is accepted on re-classification",
                lambda fev=fev: "stop" not in fev and E.classify_fold(fev, contract, protocol)
                .status in ep.ACCEPTED_FIT_STATUSES,
                lambda fev=fev: (fev.get("fit_status") or {}).get("status"))
    v.check("the artifacts embed exactly the evidence.json fold records",
            lambda: _same((g_art.get("pr38_evidence") or {}).get("folds"), folds[E.ARM_GLASS])
            and _same((r_extra.get("pr38_evidence") or {}).get("folds"), folds[E.ARM_RF]))

    # V4: isolation and label leakage (G4, G13, no test labels)
    v = gates["V4"]
    inner_folds = {E.ARM_GLASS: int(settings.glass_config.pass1_oof_folds),
                   E.ARM_RF: int(settings.rf_config_factory().n_oof_folds)}
    for arm in _ARMS:
        for ev in folds[arm]:
            k = int(ev["fold"])
            tr, va = ts.folds[k]
            P = ts.X_train.index[tr]
            held = set(ts.X_train.index[va])
            rem = (ev.get("remainder") or {}).get("index") or []
            v.check(f"{arm} fold {k}: fitted on exactly its outer-training rows",
                    lambda ev=ev, tr=tr, P=P: int(ev["n_train"]) == len(tr)
                    and ev["train_label_fingerprint"] == label_fingerprint(P, ts.y_train.iloc[tr]))
            v.check(f"{arm} fold {k}: isolation clean (no held-out row in any fit)",
                    lambda ev=ev: R._isolation_clean(ev.get("isolation") or {}))
            v.check(f"{arm} fold {k}: remainder rows inside the fold's training rows",
                    lambda ev=ev, rem=rem, P=P, held=held: len(rem) == int(ev["remainder"]["rows"])
                    and set(rem) <= set(P) and not (set(rem) & held))
            iso = ev.get("isolation") or {}
            flag = ("inner_rankers_fitted_on_their_inner_train_rows" if arm == E.ARM_GLASS
                    else "eligibility_solves_exclude_their_fold")
            v.check(f"{arm} fold {k}: {flag.replace('_', ' ')} (every inner fold)",
                    lambda iso=iso, flag=flag, n=inner_folds[arm]:
                    len(iso.get(flag) or []) == n > 0 and all(iso[flag]),
                    lambda iso=iso, flag=flag: iso.get(flag))
            if adopted == IF.PATHWAY_FOLD:
                v.check(f"{arm} fold {k}: inner bins are the fold-label-excluded ones (G13)",
                        lambda iso=iso: iso.get("inner_features_match_the_fit") is True)
    v.check("G13: every fit ran on the adopted feature pathway",
            lambda: _pathways_ok(folds[E.ARM_GLASS] + folds[E.ARM_RF]
                                 + [full.get(a) or {} for a in _ARMS], adopted), adopted)
    v.check("no test label or test metric is stored",
            lambda: bool(arts) and "test_metrics" not in g_art and g_art.get("y_test") is None
            and r_art.get("test_metrics") is None and r_art.get("y_test") is None
            and r_extra.get("y_test") is None)

    # V5: persistence and provenance (G5, R1, R2, R5)
    v = gates["V5"]
    for arm in _ARMS:
        keys = set(ep.required_selection_keys(arm, protocol))
        v.check(f"{arm}: every fold persists {list(REQUIRED_EVIDENCE[arm])}",
                lambda arm=arm: bool(folds[arm]) and all(
                    set(REQUIRED_EVIDENCE[arm]) <= set(ev) for ev in folds[arm]))
        v.check(f"{arm}: union counts recorded for every required selection, records complete",
                lambda arm=arm, keys=keys: bool(folds[arm]) and all(
                    set(ev["counts"]) == keys and not ev["fit_status"]["problems"]
                    for ev in folds[arm]))
    rf_evs = folds[E.ARM_RF]
    summary = r_extra.get("oof_fold_summary") or []
    v.check("R1: each fold router's bands and supports, at or above their floors",
            lambda: len(rf_evs) == len(attempted)
            and all(ev["r1_bands"]["supports_at_or_above_floors"] for ev in rf_evs)
            and sorted(int(r["fold"]) for r in summary) == attempted
            and all({"pass1_confidence", "pass2_confidence", "pass1_support",
                     "pass2_support"} <= set(r) for r in summary))
    v.check("R1: the full-train router's bands",
            lambda: {"pass1_confidence", "pass2_confidence", "pass1_support",
                     "pass2_support"} <= set(r_art.get("bands") or {}))
    v.check("R2: strip records for every fold router, computed inside the cross-fit",
            lambda: all({"strip_a", "strip_b", "strip_a_in_band", "band_rows"}
                        <= set(ev["r2_strips"]) for ev in rf_evs) and len(rf_evs) == len(attempted))
    rem = (r_art.get("fit_report") or {}).get("remainder") or {}
    v.check("R5: nested_t1 in the config, the fit report (per-fold t1_j) and oof_protocol",
            lambda: (r_art["config"]["eligibility"], rem.get("eligibility"),
                     (r_extra.get("oof_protocol") or {}).get("pass2_eligibility"))
            == ("nested_t1",) * 3
            and len(rem.get("eligibility_fold_t1") or []) == inner_folds[E.ARM_RF]
            and bool(rf_evs) and all(ev["eligibility"]["rule"] == "nested_t1" and len(
                ev["eligibility"]["fold_t1"]) == inner_folds[E.ARM_RF] for ev in rf_evs))

    # V6: reproducibility (G7), on the pre-declared fold only
    v = gates["V6"]
    g7_bad = g7_problems(g7, candidate, settings, held_out_rows=len(ts.folds[G7_FOLD][1]))
    if not ((inputs or {}).get("g7") or {}).get("name") == G7_FILE:
        g7_bad.append(f"the acceptance inputs do not bind {G7_FILE}")
    v.check(f"G7: the pre-declared fold's reproduction (fold {G7_FOLD}) passed for this "
            "candidate, run, role and pathway in both arms", lambda: not g7_bad,
            lambda: "; ".join(g7_bad) or {"bit_identical": g7.get("bit_identical")})
    v.check("G7: no reproduction of another fold is on the record", lambda: not other_g7,
            list(other_g7))

    # V7: artifacts, Stage 4 interface, wiring, baseline (G2, G10, G11, R4)
    v = gates["V7"]
    v.check("candidate integrity (manifest, SHA-256, attempt, no unlisted file)",
            lambda: candidate.intact)
    v.check("both artifacts load", lambda: set(arts) == set(_ARMS), load_error or "loaded")
    rel = phase0._relative(candidate.root)
    v.check("new versioned location, outside every newest-file folder (G10)",
            lambda: not any(rel is not None and (rel == f or rel.startswith(f + "/"))
                            for f in phase0.NEWEST_FILE_FOLDERS), rel)

    def reloaded():
        if not reload:
            raise RuntimeError("not run")
        got = _fresh_reload([candidate.glass_path, candidate.rf_path])
        return all(got[str(p)] == ["dict", sorted(map(str, arts[a]))]
                   for a, p in ((E.ARM_GLASS, candidate.glass_path),
                                (E.ARM_RF, candidate.rf_path)))
    v.check("both artifacts reload in a fresh interpreter as written", reloaded)
    v.check("G2: GLASS oof_remainder with fold_nested_rf, declared by the cross-fit and the fit",
            lambda: ((g_art.get("oof_protocol") or {}).get("glass_pass2_population"),
                     (g_art.get("fit_report") or {}).get("pass2_population"),
                     (g_art.get("oof_protocol") or {}).get("glass_inner_feature_ranker"))
            == ("oof_remainder", "oof_remainder", "fold_nested_rf"))
    notes = {}
    if set(arts) == set(_ARMS):
        try:
            notes = _r4_checks(v, arts, ts)
        except Exception as exc:                    # noqa: BLE001 - e.g. a consumer import
            v.check("R4 Stage 4 consumers ran", lambda: False, f"{type(exc).__name__}: {exc}")
    v.check("R5 at Stage 4: pass2_eligibility nested_t1 on both sides of the RF Router",
            lambda: notes[E.ARM_RF]["pass2_eligibility"] == {"cross_fit": "nested_t1",
                                                             "full_train": "nested_t1"})
    g11 = g11 or {}
    for key in ("snapshot", "baseline", "artifact_manifest"):
        v.check(f"G11: {key} re-verified after the run",
                lambda key=key: (g11.get(key) or {}).get("passed") is True,
                lambda key=key: "not run" if key not in g11 else "")
    v.check("G11: the baseline MANIFEST.json is the one the protocol pins",
            lambda: g11.get("baseline_manifest_sha256")
            == protocol["population"]["baseline_manifest_sha256"])
    v.check("G11: the pinned artifact manifest is the one this run's Phase 0 checked",
            lambda: (g11.get("artifact_manifest") or {}).get("manifest_sha256") is not None
            and (g11["artifact_manifest"]["manifest_sha256"]
                 == (_json(candidate.record_dir / "phase0.json").get("artifact_manifest")
                     or {}).get("manifest_sha256")))
    _disposition_check(v, dispositions, "G10:artifact_audit_pins", "F17 G10: the GLASS "
                       "'full' pins, updated or scoped by an approved versioned clarification")

    # V8: F31 confidence semantics (F31 A, R7, R8)
    v = gates["V8"]
    rfv = views.get(E.ARM_RF)
    v.check("F31 A: the RF confidence construction is unchanged (flat bands; router "
            "transform, not calibrated)",
            lambda: (r_art.get("bands") or {}).get("kind") == "flat" and rfv is not None
            and rfv.router_score.space == "router_transform")
    v.check("R7: the Cell 21 predict_train_oof agreement is replaced by R2 in this pathway",
            lambda: len(rf_evs) == len(attempted) and all("r2_strips" in ev for ev in rf_evs)
            and not any("predict_train_oof" in str(k) for k in r_extra),
            "the PR 38 runner never calls predict_train_oof; the notebook is not run")
    v.check("R8: bands come from fits on the adopted G13 pathway (fold and full-train routers)",
            lambda: _pathways_ok(rf_evs + [full.get(E.ARM_RF) or {}], adopted), adopted)
    clar = ((authorization or {}).get("f31_r2_clarification") or {}).get("record")
    v.check("the R2 clarification the strips use is the approved record",
            lambda: clar == E.record_identity(E.f31_r2_clarification()), clar)
    strips = strip_review(ev_all, protocol)
    for r in strips:
        if r["review_required"]:
            _disposition_check(v, dispositions, f"F31:R2:fold{r['fold']}:{r['status']}",
                               "F31 section 7, condition 1: the R2 strip review")
    if adopted == IF.PATHWAY_FOLD:
        _disposition_check(v, dispositions, "F31:condition2",
                           "F31 section 7, condition 2: the adopted G13 correction")

    rec.update(fold_ledger=None if ledger is None else ledger.summary().to_dict("records"),
               fit_status_by_fold=table, f31_strip_review=strips,
               required_dispositions=required_dispositions(candidate, adopted, protocol))
    return _finish(rec, gates, protocol)


def write_acceptance(record_dir, rec: Mapping[str, Any]) -> Path:
    """
    With its ``record_digest``: accepted, ``structural_acceptance.json`` (once);
    not accepted, a dated attempt record (never read by ``require_accepted``).
    """
    rec = dict(_canon(rec))
    rec["record_digest"] = _digest(rec)
    if rec["accepted"]:
        return R.write_record(Path(record_dir) / ACCEPTANCE_FILE, rec)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return R.write_record(Path(record_dir) / f"structural_acceptance_failed_{stamp}.json", rec)


# ----------------------------------------------------------------------
# Step 3B's gate
# ----------------------------------------------------------------------
def acceptance_record_problems(rec: Mapping[str, Any], candidate: Candidate,
                               settings: R.ArmSettings) -> List[str]:
    """
    Why ``rec`` does not establish an official structural acceptance of
    ``candidate``. Checked: the digest; run, role, candidate and protocol
    identity; all eight gates complete (every mandatory check present, every
    check passed); the verdict recomputed from the gate records; every bound
    input unchanged and re-verified (the authorization through
    ``require_authorization``, G7 through ``g7_problems``, each required
    disposition through ``disposition_problems``); the fold ledger re-derived
    from the candidate's own evidence.
    """
    protocol, run_id, role = settings.protocol, candidate.run_id, candidate.role
    out: List[str] = []
    if rec.get("record_digest") != _digest(rec):
        out.append("record_digest does not match the record (edited after it was written)")
    want = {"format": ACCEPTANCE_FORMAT, "run_id": run_id, "role": role, "accepted": True,
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "protocol": _protocol_identity(protocol),
            "artifacts": {E.ARM_GLASS: candidate.relative(candidate.glass_path),
                          E.ARM_RF: candidate.relative(candidate.rf_path)},
            "feature_pathway": _canon(IF.pathway_identity(settings.feature_pathway))}
    out += [f"{k}: {rec.get(k)!r}, expected {v!r}" for k, v in want.items()
            if _canon(rec.get(k)) != _canon(v)]
    if not ep.is_pinned_frozen(protocol):
        out.append("the protocol is not the pinned frozen record")

    required = list(protocol["acceptance"]["structural"])
    gates = rec.get("gates") or {}
    if sorted(gates) != sorted(required):
        out.append(f"gates {sorted(gates)}, expected {required}")
    mandatory = _mandatory(protocol)
    results = []
    for g in required:
        gr = gates.get(g) or {}
        checks = gr.get("checks") or []
        names = [str(c.get("check")) for c in checks]
        failed = [n for n, c in zip(names, checks) if c.get("passed") is not True]
        if gr.get("status") != ep.GATE_PASS or gr.get("passed") is not True or failed or not checks:
            out.append(f"{g}: not passed with every check passed (failed {failed[:3]})")
        lacking = [m for m in mandatory.get(g, []) if not any(n.startswith(m) for n in names)]
        if lacking:
            out.append(f"{g}: incomplete evidence, no check {lacking[:3]}")
        try:
            results.append(ep.GateResult(g, ep.GATE_PASS if gr.get("passed") is True and not failed
                                         and checks else ep.GATE_FAIL,
                                         str(gr.get("evidence") or "")))
        except ep.ProtocolError as exc:
            out.append(f"{g}: {exc}")
    sa = ep.structural_acceptance(results, protocol) if len(results) == len(required) else None
    rsa = rec.get("structural_acceptance") or {}
    if sa is None or not (sa.accepted and sa.official):
        out.append("the verdict recomputed from the gate records is not an official acceptance")
    if not (rsa.get("accepted") is True and rsa.get("official") is True
            and rsa.get("passed") == required and rsa.get("failed") == []
            and rsa.get("not_run") == [] and rsa.get("missing") == []):
        out.append("the recorded verdict is not an official acceptance of all eight gates")

    adopted = settings.feature_pathway
    req_disp = required_dispositions(candidate, adopted, protocol)
    record_dir = candidate.record_dir
    inputs = rec.get("inputs") or {}
    out += _bound_problems(inputs, record_dir, dispositions_required=bool(req_disp))
    auth_ref = inputs.get("authorization") or {}
    try:
        R.require_authorization(record_dir / str(auth_ref.get("name")), settings, run_id=run_id)
    except (R.RunnerError, OSError, ValueError) as exc:
        out.append(f"the bound authorization does not re-verify: {str(exc).splitlines()[0]}")
    g7, others = load_g7(record_dir)
    out += [f"G7: {p}" for p in g7_problems(g7, candidate, settings)]
    if others:
        out.append(f"G7: reproductions of other folds on the record {others}")
    try:
        disp, _ = load_dispositions(record_dir, run_id)
    except R.RunnerError as exc:
        disp = {}
        out.append(str(exc))
    for ident in req_disp:
        out += disposition_problems(ident, disp.get(ident))
    folds = candidate.evidence.get("folds") or {}
    try:
        ep.FoldLedger.build({arm: {int(ev["fold"]): (E.classify_fold(ev, settings.contract,
                                                                     protocol)
                                                     or ev["fit_status"]["status"])
                                   for ev in folds.get(arm) or []} for arm in _ARMS},
                            protocol).require_complete(list(_ARMS))
    except (ep.ProtocolError, KeyError, TypeError) as exc:
        out.append(f"the candidate's own evidence does not give a complete fold ledger: "
                   f"{str(exc).splitlines()[0]}")
    return out


def require_accepted(run_id: str, *, role: str = R.ROLE_PRIMARY, output_root=None,
                     record_dir=None, settings: Optional[R.ArmSettings] = None) -> Candidate:
    """
    Step 3B's entry point: the unchanged candidate of an official structural
    acceptance that re-verifies in full (``acceptance_record_problems``).
    ``settings`` defaults to the primary settings built now for the recorded
    pathway. Fails closed: an integrity failure refuses before anything else.
    """
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    path = record_dir / ACCEPTANCE_FILE
    if not path.is_file():
        raise R.RunnerError(f"run {run_id!r} has no structural acceptance record ({path}); "
                            "no comparative analysis may read its artifacts")
    candidate = load_candidate(run_id, role=role, output_root=output_root, record_dir=record_dir)
    if not candidate.intact:
        raise R.RunnerError(f"the candidate of run {run_id!r} fails its integrity checks: "
                            + "; ".join(c["check"] for c in candidate.integrity if not c["passed"]))
    rec = _json(path)
    if settings is None:
        pathway = (rec.get("feature_pathway") or {}).get("pathway")
        if pathway not in IF.PATHWAYS:
            raise R.RunnerError(f"{path} records no known feature pathway")
        settings = R.primary_settings(pathway)
    problems = acceptance_record_problems(rec, candidate, settings)
    if problems:
        raise R.RunnerError(f"{path} does not establish an official structural acceptance:\n  - "
                            + "\n  - ".join(problems))
    return candidate
