"""
shared.stage2.diagnostics.baseline
==================================
The Pre-Correction Stage 2 Baseline (PR 37, Commit 2, Step 2E): a frozen,
versioned record of the validated Steps 2A-2D router diagnostics on the
historical training-side outer OOF outputs, kept unchanged for comparison with
PR 38's corrected-selector results.

    BaselineSources    the validated views plus the identities of the files they
                       came from (assembled by the caller; this module opens no
                       model artifact)
    build_baseline     -> BaselinePackage: every file's bytes, built in memory
                       through the validated Step 2A-2D APIs (nothing recomputed)
    write_baseline     first write only, atomically; an identical rerun writes
                       nothing; a differing package is refused (BaselineConflict)
    verify_baseline    integrity only, no artifacts: manifest, sidecar hash, each
                       listed file's SHA-256 / size / rows / header, no extra files
    compare_baseline   a freshly built package against the frozen directory
    code_identity      SHA-256 of the diagnostic code (LF-normalised text)

This is the only module of ``shared.stage2.diagnostics`` that writes files, and
only through ``write_baseline`` (research_logs/baselines/stage2_baseline.py
--generate). It is imported explicitly, not re-exported by the package.

Package ``<parent>/<baseline_id>/``
-----------------------------------
    MANIFEST.json          identity, designation, population, fingerprints,
                           sources, code identity, reference operating points,
                           constraint provenance, analysis settings, validation,
                           outputs (SHA-256 / bytes / rows / columns), disclosures
    MANIFEST.json.sha256   SHA-256 of MANIFEST.json (the manifest never hashes itself)
    REPORT.md              the human-readable report
    definitions.json       metric registry, table / column definitions, conventions
    tables/*.csv           the exported diagnostics (``TABLES``)

Determinism
-----------
No timestamps, absolute paths or environment details; rows in the producing
function's deterministic order; floats with 12 significant digits; LF line
endings; UTF-8. Machine-specific absolute paths in free text become
``<local path>``; any that remain refuse the build.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import overlap as ov
from .boundary import FORBIDDEN_FIELD_TOKENS, TRANSFORM_ATOL
from .contribution import (
    MATCH_SCORES,
    TIE_BREAK_RULE,
    common_eligible_population,
    pass2_frozen_contribution,
    pass2_matched_contribution,
    pass2_population,
    pass2_positive_breakdown,
)
from .metrics import METRIC_DEFINITIONS, Ratio, router_router_table, router_upstream_metrics
from .schema import (
    ABSTAIN,
    HISTORICAL_TEST_EXPOSURE,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    ROUTER_STATES,
    RFPass1ScoreCandidate,
    RouterDiagnosticError,
    RouterOOFView,
    TrainReference,
    UpstreamOOFView,
)
from .suppression import (
    pass1_population,
    pass1_suppression_report,
    primary_reference_flags,
    reference_flag_summary,
)
from .trace import (
    ARM_LABELS,
    GLASS_ARM,
    RF_ARM,
    TRACE_ARMS,
    build_router_trace,
    compare_state_distributions,
    routing_policy_audit,
    routing_rates,
    state_distribution,
)

__all__ = [
    "BASELINE_ID", "BASELINE_LABEL", "DESIGNATION", "FORMAT", "FORMAT_VERSION", "MANIFEST",
    "SIDECAR", "REPORT", "DEFINITIONS", "TABLES", "SOURCE_ROLES", "CODE_FILES",
    "BASELINES_DIR", "BaselineError", "BaselineConflict", "SourceArtifact", "SnapshotRecord",
    "DataFileRecord", "BaselineSources", "CodeIdentity", "BaselinePackage",
    "VerificationReport", "ComparisonReport", "WriteResult", "lf_sha256", "code_identity",
    "build_baseline", "write_baseline", "verify_baseline", "compare_baseline", "read_manifest",
]

# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------
BASELINE_ID = "pr37_stage2_pre_correction_v1"
BASELINE_LABEL = "Pre-Correction Stage 2 Baseline"
DESIGNATION = "pre_correction"
FORMAT = "pr37.stage2_router_baseline"
FORMAT_VERSION = 1
#: Where frozen packages live, relative to bank_pipeline/.
BASELINES_DIR = "research_logs/baselines"

MANIFEST = "MANIFEST.json"
SIDECAR = "MANIFEST.json.sha256"
REPORT = "REPORT.md"
DEFINITIONS = "definitions.json"
TABLES_DIR = "tables"
FLOAT_FORMAT = ".12g"
LOCAL_PATH = "<local path>"

#: The files a package is built from, by role (``SourceArtifact.role``).
ROLE_ARBITER_GLASS = "stage4_arbiter.meta_ebm"
ROLE_ARBITER_RF = "stage4_arbiter.meta_ebm_on_blackbox"
ROLE_LR = "stage1.lr"
ROLE_GLASS = "stage2.glass_router"
ROLE_EBM = "stage3.ebm"
ROLE_RF = "stage2.rf_router"
ROLE_RF_RESULT = "glass_arm.rf_result"
SOURCE_ROLES: Tuple[str, ...] = (ROLE_ARBITER_GLASS, ROLE_ARBITER_RF, ROLE_LR, ROLE_GLASS,
                                 ROLE_EBM, ROLE_RF, ROLE_RF_RESULT)
_ROLE_READER = MappingProxyType({
    ROLE_ARBITER_GLASS: "tests.support.artifact_audit.upstream_paths (recorded Stage 1-3 paths only)",
    ROLE_ARBITER_RF: "tests.support.artifact_audit.upstream_paths (recorded Stage 2 path only)",
    ROLE_LR: "shared.stage2.diagnostics.boundary.extract_lr_view",
    ROLE_GLASS: "shared.stage2.diagnostics.boundary.extract_router_view (expected_arm='glass')",
    ROLE_EBM: "shared.stage2.diagnostics.boundary.extract_ebm_view",
    ROLE_RF: "shared.stage2.diagnostics.boundary.extract_router_view (expected_arm='blackbox')",
    ROLE_RF_RESULT: "shared.stage2.diagnostics.boundary.extract_rf_pass1_candidate",
})

#: The code whose identity a package records (paths relative to bank_pipeline/).
CODE_FILES: Tuple[str, ...] = tuple(
    [f"shared/stage2/diagnostics/{m}.py" for m in (
        "__init__", "schema", "boundary", "operating_points", "metrics", "equivalence",
        "trace", "suppression", "contribution", "overlap", "baseline")]
    + ["shared/stage_io.py", "tests/support/router_artifacts.py",
       "tests/support/artifact_audit.py", f"{BASELINES_DIR}/stage2_baseline.py"])

_P1, _P2, _AB = PASS_1_NOT_SUBSCRIBE, PASS_2_SUBSCRIBE, ABSTAIN
_ABS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s,;'\"()\]\[}{|<>]*"
                       r"|(?<![\w.~-])/(?:home|Users|mnt|tmp|root|var|private|opt|srv|media)"
                       r"/[^\s,;'\"()\]\[}{|<>]*"
                       r"|\\\\[A-Za-z0-9._$-]+\\[^\s,;'\"()\]\[}{|<>]*")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class BaselineError(RouterDiagnosticError):
    """The baseline cannot be built, written or verified as requested."""


class BaselineConflict(BaselineError):
    """A frozen package already exists and differs; nothing was written."""

    def __init__(self, message: str, comparison: "ComparisonReport"):
        super().__init__(message)
        self.comparison = comparison


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceArtifact:
    """One file the package was built from. ``path``: POSIX, relative to bank_pipeline/."""

    role: str
    path: str
    sha256: str
    size: int
    snapshot_attested: Optional[bool] = None   # True: equals the snapshot SHA256SUMS entry


@dataclass(frozen=True)
class SnapshotRecord:
    """The preserved-baseline SHA256SUMS the sources were attested against (no local path)."""

    label: str
    sha256: str
    entries: int


@dataclass(frozen=True)
class DataFileRecord:
    """A data file identity recorded for reference (``sha256`` over LF-normalised bytes)."""

    path: str
    sha256: str
    role: str


@dataclass(frozen=True)
class BaselineSources:
    """The validated views (``boundary`` extractors) and the files they came from."""

    ref: TrainReference
    glass: RouterOOFView
    rf: RouterOOFView
    lr: UpstreamOOFView
    ebm: UpstreamOOFView
    rf_candidate: RFPass1ScoreCandidate
    artifacts: Tuple[SourceArtifact, ...]
    snapshot: Optional[SnapshotRecord] = None
    data_files: Tuple[DataFileRecord, ...] = ()


@dataclass(frozen=True)
class CodeIdentity:
    method: str
    files: Mapping[str, Optional[str]]


@dataclass(frozen=True)
class BaselinePackage:
    baseline_id: str
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.files[MANIFEST]).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    directory: str
    ok: bool
    problems: Tuple[str, ...]
    manifest_sha256: Optional[str] = None
    files_checked: int = 0


@dataclass(frozen=True)
class ComparisonReport:
    """``status``: identical, reproduced_with_code_drift or differs."""

    status: str
    integrity: VerificationReport
    missing: Tuple[str, ...] = ()
    unexpected: Tuple[str, ...] = ()
    differing: Tuple[str, ...] = ()
    manifest_sections: Tuple[str, ...] = ()
    code_changes: Tuple[str, ...] = ()

    @property
    def reproduced(self) -> bool:
        return self.status in ("identical", "reproduced_with_code_drift")

    def lines(self) -> List[str]:
        out = [f"status: {self.status}"]
        out += [f"integrity: {p}" for p in self.integrity.problems]
        out += [f"missing in the frozen package: {p}" for p in self.missing]
        out += [f"not produced by the rebuild: {p}" for p in self.unexpected]
        out += [f"content differs: {p}" for p in self.differing]
        out += [f"MANIFEST.json section differs: {s}" for s in self.manifest_sections]
        out += [f"code identity: {c}" for c in self.code_changes]
        return out


@dataclass(frozen=True)
class WriteResult:
    status: str               # "written" or "identical"
    directory: str
    manifest_sha256: str


# ----------------------------------------------------------------------
# Hashing and serialisation
# ----------------------------------------------------------------------
def lf_sha256(data: bytes) -> str:
    """SHA-256 of ``data`` with CRLF normalised to LF (Windows / Linux checkouts agree)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_identity(bank_root, files: Sequence[str] = CODE_FILES) -> CodeIdentity:
    """LF-normalised SHA-256 of each code file (``None`` when the file is absent)."""
    root = Path(bank_root)
    out = {}
    for rel in files:
        p = root / rel
        out[rel] = lf_sha256(p.read_bytes()) if p.is_file() else None
    return CodeIdentity("sha256 of the UTF-8 file with CRLF normalised to LF", MappingProxyType(out))


def _scrub(text: str) -> str:
    return _ABS_PATH.sub(LOCAL_PATH, text)


def _cell(v) -> str:
    if v is None or v is pd.NA:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return ""
        if math.isinf(f):
            raise BaselineError("an exported value is infinite")
        s = format(f, FLOAT_FORMAT)
        return "0" if s == "-0" else s
    return _scrub(str(v))


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow([str(c) for c in df.columns])
    for row in df.itertuples(index=False, name=None):
        w.writerow([_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")


def _plain(v):
    """JSON-safe, deterministic: mappings keep order, NaN -> None, free text scrubbed."""
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if v is None or v is pd.NA:
        return None
    return _scrub(str(v))


def _json_bytes(obj) -> bytes:
    return (json.dumps(_plain(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8")


def _round(x: float) -> float:
    return float(format(float(x), FLOAT_FORMAT))


def _is_all(f) -> bool:
    return isinstance(f, str) and f == "all"


def _all(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["fold"].map(_is_all)]


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
_LONG_COLUMNS = ["section", "arm", "reference", "population", "group", "score", "fold", "metric",
                 "label", "numerator", "denominator", "value", "status"]

#: name -> (file, step, producer, description, key columns)
TABLES: Mapping[str, Tuple[str, str, str, str, Tuple[str, ...]]] = MappingProxyType({
    "states": ("tables/states.csv", "2A", "trace.state_distribution",
               "Three-state counts and class composition per router and fold.",
               ("arm", "fold", "state")),
    "ratios": ("tables/ratios.csv", "2A-2D",
               "trace.routing_rates; suppression.pass1_population / pass1_suppression_report; "
               "contribution.pass2_population / pass2_frozen_contribution / "
               "pass2_matched_contribution; overlap.router_reference_agreement / "
               "crosstab_summary / boundary_diagnostics",
               "Every count and ratio of the long-format diagnostics, with explicit numerator, "
               "denominator, value and status.",
               ("section", "arm", "reference", "population", "group", "score", "fold", "metric")),
    "policy_audit": ("tables/policy_audit.csv", "2A", "trace.routing_policy_audit",
                     "Routing-policy claims per router: what the artifacts verify, what is "
                     "code-level only, declared or not verifiable.", ("arm", "claim")),
    "reference_flags": ("tables/reference_flags.csv", "2B", "suppression.reference_flag_summary",
                        "The frozen LR / EBM reference flags (and their union / intersection) "
                        "used by every router-to-reference comparison.", ("reference",)),
    "pass2_breakdown": ("tables/pass2_breakdown.csv", "2C", "contribution.pass2_positive_breakdown",
                        "Pass 2 TP / FP split by which frozen references flag them.",
                        ("arm", "fold")),
    "common_population": ("tables/common_population.csv", "2C",
                          "contribution.common_eligible_population",
                          "The common eligible population (rows neither router routed to "
                          "Pass 1) and what it excludes.", ("fold",)),
    "matched_selection": ("tables/matched_selection.csv", "2C",
                          "contribution.pass2_matched_contribution (meta['selections'])",
                          "The label-free rate-matched reference selections: K, cut, tie block "
                          "and tie-broken rows per arm, model and fold.",
                          ("arm", "reference", "fold")),
    "contingency": ("tables/contingency.csv", "2D", "overlap.router_reference_contingency",
                    "Router state x reference flag: rows, positives and negatives.",
                    ("arm", "reference", "fold", "state", "reference_flag")),
    "crosstab": ("tables/crosstab.csv", "2D", "overlap.router_crosstab",
                 "GLASS state x RF state cells with label composition and reference flags.",
                 ("fold", "glass_state", "rf_state")),
    "pass2_cross_router": ("tables/pass2_cross_router.csv", "2D",
                           "overlap.pass2_cross_router_overlap",
                           "Pass 2 positives and negatives both routers / one router flag, "
                           "overall and unique relative to each reference.", ("reference", "fold")),
    "signal_availability": ("tables/signal_availability.csv", "2D", "overlap.signal_availability",
                            "Which pre-routing signals the artifacts persist, and which overlap "
                            "analyses are therefore unavailable.", ("arm", "signal")),
    "score_conditional": ("tables/score_conditional.csv", "2D", "overlap.score_conditional_states",
                          "Router state x label-free score stratum: label composition, calibrated "
                          "expectations and reference flags.",
                          ("arm", "stratifier", "fold", "stratum", "state")),
    "stratum_contrasts": ("tables/stratum_contrasts.csv", "2D", "overlap.within_stratum_contrasts",
                          "Positive-rate differences between router states within a stratum "
                          "(defined only where both states have minimum support).",
                          ("arm", "stratifier", "fold", "stratum", "contrast")),
    "observed_expected": ("tables/observed_expected.csv", "2D", "overlap.group_observed_expected",
                          "Observed vs expected positives for the router states and the named "
                          "disagreement populations (descriptive; not tests).", ("group", "fold")),
    "chance_baselines": ("tables/chance_baselines.csv", "2D",
                         "overlap.unique_positive_chance_baselines",
                         "Arithmetic chance expectations for unique positives and Pass 1 "
                         "interceptions (not statistical tests).",
                         ("setting", "arm", "reference", "fold")),
    "model_disagreement": ("tables/model_disagreement.csv", "2D",
                           "overlap.model_disagreement_baseline",
                           "How much LR and EBM already disagree (frozen, and at each arm's "
                           "matched K), next to the routers' unique positives.",
                           ("setting", "arm", "fold")),
    "validation": ("tables/validation.csv", "2A / Commit 1",
                   "RouterTrace.checks; each view's boundary extractor checks",
                   "Every trace check (T1-T7) and boundary extractor check, with its result.",
                   ("source", "stage", "check")),
})

#: Free-text and identifier column definitions shared by several tables.
_COMMON_COLUMNS = {
    "arm": "router arm: 'glass' = GLASS Router, 'blackbox' = RF Router ('' = not arm-specific)",
    "fold": "'all' = pooled over the five router outer folds; 0-4 = router outer fold id",
    "state": "router state: PASS_1_NOT_SUBSCRIBE, PASS_2_SUBSCRIBE or ABSTAIN",
    "reference": ("reference flag set: lr = lr.in_stage_f2_cv, ebm = ebm.nested_f2, union = either "
                  "flags, intersection = both flag; lr_matched / ebm_matched / lr_or_ebm_matched = "
                  "label-free rate-matched selections; all_pass2 = no reference (all Pass 2 rows)"),
    "metric": "metric name; ratios are registered in definitions.json 'metrics'",
    "label": "the producing function's label for the metric in this section",
    "numerator": "numerator count (empty for quantile rows)",
    "denominator": "denominator count (empty for count and quantile rows)",
    "value": ("numerator / denominator for ratios (empty when undefined); the count for count rows; "
              "the quantile for quantile rows"),
    "status": ("defined = ratio with a non-zero denominator; undefined_empty_denominator = "
               "denominator 0, value left empty (never 0); count = a count; quantile = a "
               "distribution quantile"),
    "n": "rows in the cell / group",
    "positives": "actual positives (y = 1) among the rows",
    "negatives": "actual negatives (y = 0) among the rows",
    "positive_rate": "positives / n (empty when n = 0; see the status column)",
    "positive_rate_status": "defined or undefined_empty_denominator",
}

_TABLE_COLUMNS: Mapping[str, Mapping[str, str]] = {
    "states": {
        "rows": "rows in the fold (or all rows for 'all')",
        "share": "n / rows (registered pass1_rate / pass2_rate / abstain_rate)",
        "share_status": "status of share", "negative_rate": "negatives / n",
        "negative_rate_status": "status of negative_rate"},
    "ratios": {
        "section": "the step and producing function (see definitions.json tables.ratios.sections)",
        "population": "all_rows or common_eligible (Step 2C); '' elsewhere",
        "group": "named disagreement population (Step 2D boundary positions); '' elsewhere",
        "score": "lr = LR calibrated probability vs its stored threshold; ebm = EBM raw score vs "
                 "its fold's nested threshold (Step 2D boundary positions); '' elsewhere"},
    "policy_audit": {
        "router": "GLASS Router / RF Router", "claim": "the routing-policy claim",
        "description": "what the claim asserts",
        "status": ("verified_from_artifacts, evidence_failed, declared, code_level_only or "
                   "not_verifiable (trace.AUDIT_STATUSES)"),
        "basis": "artifact, code or none", "violations": "0 when verified; empty otherwise",
        "evidence": "the evidence or the reason it is unavailable"},
    "reference_flags": {
        "operating_point": "registered operating point (operating_points.OPERATING_POINTS)",
        "score_space": "score space the flag is defined in",
        "threshold": "stored scalar threshold (empty for fold-nested decisions: see MANIFEST.json "
                     "upstream_operating_points)",
        "selection": "how the threshold / decisions were selected (disclosed)",
        "fold_plan": "the reference's fold plan relative to the routers' outer folds",
        "flags": "rows flagged", "tp": "flagged positives", "fp": "flagged negatives",
        "flag_rate": "flags / all training rows", "precision": "tp / flags",
        "precision_status": "status of precision"},
    "pass2_breakdown": {
        "pass2_tp": "Pass 2 actual positives", "pass2_fp": "Pass 2 actual negatives",
        "tp_both": "Pass 2 TP flagged by LR and EBM", "fp_both": "Pass 2 FP flagged by LR and EBM",
        "tp_lr_only": "Pass 2 TP flagged by LR only", "fp_lr_only": "Pass 2 FP flagged by LR only",
        "tp_ebm_only": "Pass 2 TP flagged by EBM only",
        "fp_ebm_only": "Pass 2 FP flagged by EBM only",
        "tp_neither": "Pass 2 TP flagged by neither", "fp_neither": "Pass 2 FP flagged by neither"},
    "common_population": {
        "rows": "rows in the fold / pool", "positives": "actual positives in the fold / pool",
        "glass_remainder_rows": "rows GLASS did not route to Pass 1",
        "rf_remainder_rows": "rows RF did not route to Pass 1",
        "common_rows": "rows neither router routed to Pass 1 (common eligible population)",
        "common_positives": "actual positives in the common eligible population",
        "glass_pass2_in_common": "GLASS Pass 2 rows inside the common population",
        "rf_pass2_in_common": "RF Pass 2 rows inside the common population",
        "glass_pass2_outside_common": "GLASS Pass 2 rows RF routed to Pass 1 (excluded)",
        "rf_pass2_outside_common": "RF Pass 2 rows GLASS routed to Pass 1 (excluded)"},
    "matched_selection": {
        "model": "the reference model ranked (lr or ebm)",
        "score_space": "score space ranked (contribution.MATCH_SCORES)",
        "target_count": "K = the arm's Pass 2 rows in the fold's common eligible population",
        "selected_rows": "rows selected (always K): the K highest scores, ties broken by the "
                         "label-free tie-break rule (MANIFEST.json analysis_settings)",
        "cut_value": "the K-th highest score", "exact": "true when no tie straddles the cut",
        "tie_block": "rows sharing the K-th score (1 = no tie)",
        "strict_count": "rows strictly above the tied block (strict bound, <= K)",
        "inclusive_count": "rows including the whole tied block (inclusive bound, >= K)",
        "tie_broken_rows": "rows chosen from the tied block by the tie-break rule"},
    "contingency": {"reference_flag": "true = the reference flags the row"},
    "crosstab": {
        "glass_state": "GLASS Router state", "rf_state": "RF Router state",
        "share": "n / rows in the fold group (registered cross_cell_share)",
        **{f"{k}_flags": f"rows in the cell the {k} reference flags"
           for k in ("lr", "ebm", "union", "intersection")},
        **{f"{k}_flagged_positives": f"positives in the cell the {k} reference flags"
           for k in ("lr", "ebm", "union", "intersection")}},
    "pass2_cross_router": {
        "glass_unique_tp": "GLASS Pass 2 positives the reference does not flag",
        "rf_unique_tp": "RF Pass 2 positives the reference does not flag",
        "shared_tp": "of those, positives both routers flag",
        "glass_only_tp": "unique positives GLASS flags and RF does not",
        "rf_only_tp": "unique positives RF flags and GLASS does not",
        "glass_unique_fp": "GLASS Pass 2 negatives the reference does not flag",
        "rf_unique_fp": "RF Pass 2 negatives the reference does not flag",
        "shared_fp": "of those, negatives both routers flag",
        "glass_only_fp": "unique negatives GLASS flags and RF does not",
        "rf_only_fp": "unique negatives RF flags and GLASS does not",
        "glass_only_tp_in_rf_pass1": "GLASS-only unique positives RF routed to Pass 1",
        "glass_only_tp_in_rf_abstain": "GLASS-only unique positives RF abstained on",
        "rf_only_tp_in_glass_pass1": "RF-only unique positives GLASS routed to Pass 1",
        "rf_only_tp_in_glass_abstain": "RF-only unique positives GLASS abstained on",
        "tp_jaccard": "shared_tp / (glass_unique_tp + rf_unique_tp - shared_tp)",
        "tp_jaccard_status": "status of tp_jaccard",
        "pool_positives": "positives neither reference-flagged nor in either router's Pass 1",
        "expected_shared_tp_if_unrelated": ("glass_unique_tp x rf_unique_tp / pool_positives "
                                            "(arithmetic, not a test; empty when the pool is "
                                            "empty)")},
    "signal_availability": {
        "signal": "the signal or analysis", "status": "audit status (trace.AUDIT_STATUSES)",
        "basis": "artifact, code, none or invalid population",
        "rows": "rows the signal covers (empty when unavailable)", "detail": "evidence / reason"},
    "score_conditional": {
        "stratifier": "lr (LR calibrated probability deciles), ebm_raw (EBM raw score deciles) or "
                      "lr_x_ebm (3 x 3 joint grid); always within the router outer fold",
        "stratum": "bin index within the fold (merged edges: see the known limitations); pooled "
                   "rows sum bin k of every fold",
        "stratum_rows": "rows in the (fold, stratum) cell",
        "stratum_positives": "positives in the (fold, stratum) cell",
        "expected_lr_calibrated": "sum of LR calibrated OOF probabilities over the rows",
        "expected_ebm_calibrated": "sum of EBM calibrated OOF probabilities over the rows",
        "low_support": "true when n < the minimum support",
        **{f"{k}_flags": f"rows the {k} reference flags"
           for k in ("lr", "ebm", "union", "intersection")}},
    "stratum_contrasts": {
        "stratifier": "as in score_conditional", "stratum": "as in score_conditional",
        "contrast": "pass1 - abstain, pass2 - abstain or pass2 - pass1",
        "n_a": "rows in the first state", "n_b": "rows in the second state",
        "difference_pp": "100 x (positive rate of the first state - of the second)",
        "status": "defined, or low_support (difference left empty)"},
    "observed_expected": {
        "group": "named population (overlap.key_groups)", "description": "what the group is",
        "observed": "actual positives in the group",
        "expected_lr_calibrated": "sum of LR calibrated OOF probabilities over the group",
        "expected_ebm_calibrated": "sum of EBM calibrated OOF probabilities over the group",
        **{f"expected_stratified_{s}": (f"sum over the group's rows of their (fold, {s} stratum) "
                                       "cell's observed positive rate (internal standardisation "
                                       "on the same labels)") for s in ov.STRATIFIERS},
        **{f"o_minus_e_{e}": f"observed - expected_{e}"
           for e in ("lr_calibrated", "ebm_calibrated",
                     *(f"stratified_{s}" for s in ov.STRATIFIERS))}},
    "chance_baselines": {
        "setting": ("frozen_pass2_unique (unique flags vs rows not reference-flagged and not the "
                    "arm's own Pass 1), pass1_random_veto (Pass 1 interceptions vs the reference "
                    "flags), matched_pass2_unique (common eligible rows outside the K-matched "
                    "reference)"),
        "flags": "the router's rows in the setting", "pool_rows": "rows of the pool",
        "pool_positives": "positives of the pool",
        "pool_rate": "pool_positives / pool_rows (empty when the pool is empty)",
        "expected_positives": "flags x pool_rate (arithmetic, not a test)",
        "observed_positives": "the router's actual positives in the setting",
        "excess_positives": "observed_positives - expected_positives"},
    "model_disagreement": {
        "setting": "frozen (the stored operating points) or matched_same_k (LR / EBM re-cut at "
                   "the arm's K on the common eligible population)",
        "k": "K (matched_same_k only)", "lr_flags": "LR flags", "ebm_flags": "EBM flags",
        "lr_only_tp": "positives LR flags and EBM does not",
        "ebm_only_tp": "positives EBM flags and LR does not",
        "lr_only_fp": "negatives LR flags and EBM does not",
        "ebm_only_fp": "negatives EBM flags and LR does not",
        "ebm_minus_lr_tp": "EBM TP - LR TP at the same K",
        "ebm_only_tp_expected_from_pool": "EBM-only flags x the positive rate of the rows LR does "
                                          "not select (arithmetic)",
        "router_unique_tp_vs_lr": "the arm's Pass 2 TP the matched LR selection misses (Step 2C)",
        "router_unique_tp_vs_ebm": "the arm's Pass 2 TP the matched EBM selection misses (2C)"},
    "validation": {
        "source": "trace, or the source role whose boundary extractor ran the check",
        "section": "check section", "stage": "what was checked", "check": "the check",
        "passed": "true when the check passed", "detail": "the check's evidence"},
}

_RATIO_SECTIONS = (
    ("2A.routing_rates", "trace.routing_rates"),
    ("2B.pass1_population", "suppression.pass1_population"),
    ("2B.pass1_suppression", "suppression.pass1_suppression_report"),
    ("2C.pass2_population", "contribution.pass2_population"),
    ("2C.pass2_frozen", "contribution.pass2_frozen_contribution"),
    ("2C.pass2_matched", "contribution.pass2_matched_contribution"),
    ("2D.router_reference_agreement", "overlap.router_reference_agreement"),
    ("2D.crosstab_summary", "overlap.crosstab_summary"),
    ("2D.boundary_positions", "overlap.boundary_diagnostics"),
)


_RATIO_KEY = ("section", "arm", "reference", "population", "group", "score", "fold", "metric")


def _ratio_index(ratios: pd.DataFrame) -> Dict[tuple, Tuple[Any, Any, str]]:
    """``(section, arm, reference, population, group, score, fold, metric)`` -> (num, den, status)."""
    cols = [ratios[c].tolist() for c in (*_RATIO_KEY, "numerator", "denominator", "status")]
    out: Dict[tuple, Tuple[Any, Any, str]] = {}
    for row in zip(*cols):
        key = row[:8]
        if key in out:
            raise BaselineError(f"ratios: duplicate key {key}")
        out[key] = (row[8], row[9], row[10])
    return out


def _long(section: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "section", section)
    for c in _LONG_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[_LONG_COLUMNS]


def _matched_frame(matched, trace) -> pd.DataFrame:
    fold = np.asarray(trace.fold_id)
    rows = []
    for arm in TRACE_ARMS:
        for model in ("lr", "ebm"):
            sel = matched.meta["selections"][arm][model]
            for k in sorted(sel.by_fold):
                b = sel.by_fold[k]
                rows.append({"arm": arm, "reference": sel.reference, "model": model,
                             "score_space": sel.score_space, "fold": int(k),
                             "target_count": int(b["target_count"]),
                             "selected_rows": int((np.asarray(sel.flags) & (fold == k)).sum()),
                             "cut_value": b.get("cut_value"), "exact": bool(b["exact"]),
                             "tie_block": int(b["tie_block"]),
                             "strict_count": int(b["strict_count"]),
                             "inclusive_count": int(b["inclusive_count"]),
                             "tie_broken_rows": int(b["tie_broken_rows"])})
    return pd.DataFrame(rows)


def _validation_frame(trace, sources: BaselineSources) -> pd.DataFrame:
    rows = [{"source": "trace", **{c: r.get(c) for c in ("section", "stage", "check", "passed",
                                                        "detail")}} for r in trace.checks]
    for role, obj in _role_objects(sources).items():
        for r in (obj.source.get("checks") or ()):
            rows.append({"source": role, **{c: r.get(c) for c in ("section", "stage", "check",
                                                                  "passed", "detail")}})
    return pd.DataFrame(rows, columns=["source", "section", "stage", "check", "passed", "detail"])


def _role_objects(sources: BaselineSources) -> Dict[str, Any]:
    return {ROLE_LR: sources.lr, ROLE_GLASS: sources.glass, ROLE_EBM: sources.ebm,
            ROLE_RF: sources.rf, ROLE_RF_RESULT: sources.rf_candidate}


# ----------------------------------------------------------------------
# Inputs: checks
# ----------------------------------------------------------------------
def _check_sources(sources: BaselineSources) -> None:
    if not isinstance(sources, BaselineSources):
        raise BaselineError("expected BaselineSources")
    roles = [a.role for a in sources.artifacts]
    if sorted(roles) != sorted(SOURCE_ROLES) or len(set(roles)) != len(roles):
        raise BaselineError(f"sources must name each role exactly once: {SOURCE_ROLES}; got {roles}")
    for a in sources.artifacts:
        p = PurePosixPath(a.path)
        if (not a.path or p.is_absolute() or ".." in p.parts or "\\" in a.path
                or _ABS_PATH.search(a.path)):
            raise BaselineError(f"{a.role}: path {a.path!r} must be POSIX and relative to "
                                "bank_pipeline/")
        if not _HEX64.fullmatch(str(a.sha256)):
            raise BaselineError(f"{a.role}: sha256 {a.sha256!r} is not 64 lowercase hex digits")
    for role, obj in _role_objects(sources).items():
        fields = tuple(obj.source.get("fields_read") or ())
        bad = [f for f in fields if any(t in str(f).lower() for t in FORBIDDEN_FIELD_TOKENS)]
        if bad:
            raise BaselineError(f"{role}: test-side or forbidden fields were read: {bad}")
        if not fields:
            raise BaselineError(f"{role}: no fields_read record (not a boundary-extracted view)")
    if sources.snapshot is not None and not _HEX64.fullmatch(sources.snapshot.sha256):
        raise BaselineError("snapshot sha256 is not 64 lowercase hex digits")


# ----------------------------------------------------------------------
# Compute (validated APIs only)
# ----------------------------------------------------------------------
def _compute(sources: BaselineSources) -> Dict[str, Any]:
    trace = build_router_trace(sources.glass, sources.rf, sources.ref)
    lr, ebm = sources.lr, sources.ebm
    refs = primary_reference_flags(lr, ebm)
    supp = pass1_suppression_report(trace, lr, ebm)
    frozen = pass2_frozen_contribution(trace, lr, ebm)
    matched = pass2_matched_contribution(trace, lr, ebm)
    strata = ov.score_strata(trace, lr, ebm)
    groups = ov.key_groups(trace, refs)
    sc = ov.score_conditional_states(trace, strata, refs)
    r = {
        "trace": trace, "refs": refs, "supp": supp, "frozen": frozen, "matched": matched,
        "strata": strata, "groups": groups,
        "states": state_distribution(trace), "rates": routing_rates(trace),
        "policy": routing_policy_audit(trace, rf_candidate=sources.rf_candidate),
        "compare": compare_state_distributions(trace),
        "p1pop": pass1_population(trace), "p2pop": pass2_population(trace),
        "refsum": reference_flag_summary(trace, refs),
        "breakdown": pass2_positive_breakdown(trace, refs),
        "common": common_eligible_population(trace),
        "contingency": ov.router_reference_contingency(trace, refs),
        "agreement": ov.router_reference_agreement(trace, refs),
        "crosstab": ov.router_crosstab(trace, refs), "xsum": ov.crosstab_summary(trace),
        "p2x": ov.pass2_cross_router_overlap(trace, refs),
        "signal": ov.signal_availability(trace, rf_candidate=sources.rf_candidate),
        "sc": sc, "contrasts": ov.within_stratum_contrasts(sc),
        "std": ov.state_standardization(trace, strata),
        "goe": ov.group_observed_expected(trace, strata, groups),
        "bnd": ov.boundary_diagnostics(trace, strata, groups),
        "chance": ov.unique_positive_chance_baselines(trace, refs, matched),
        "mdis": ov.model_disagreement_baseline(trace, refs, matched),
        "notes": ov.overlap_notes(trace, strata),
    }
    return r


def _tables(r: Dict[str, Any], sources: BaselineSources) -> Dict[str, pd.DataFrame]:
    parts = {"2A.routing_rates": r["rates"], "2B.pass1_population": r["p1pop"],
             "2B.pass1_suppression": r["supp"].table, "2C.pass2_population": r["p2pop"],
             "2C.pass2_frozen": r["frozen"].table, "2C.pass2_matched": r["matched"].table,
             "2D.router_reference_agreement": r["agreement"], "2D.crosstab_summary": r["xsum"],
             "2D.boundary_positions": r["bnd"]}
    ratios = pd.concat([_long(s, parts[s]) for s, _ in _RATIO_SECTIONS], ignore_index=True)
    return {
        "states": r["states"], "ratios": ratios, "policy_audit": r["policy"],
        "reference_flags": r["refsum"], "pass2_breakdown": r["breakdown"],
        "common_population": r["common"],
        "matched_selection": _matched_frame(r["matched"], r["trace"]),
        "contingency": r["contingency"], "crosstab": r["crosstab"],
        "pass2_cross_router": r["p2x"], "signal_availability": r["signal"],
        "score_conditional": r["sc"], "stratum_contrasts": r["contrasts"],
        "observed_expected": r["goe"], "chance_baselines": r["chance"],
        "model_disagreement": r["mdis"], "validation": _validation_frame(r["trace"], sources),
    }


# ----------------------------------------------------------------------
# Consistency (internal; every check must pass or nothing is built)
# ----------------------------------------------------------------------
_STATE_GROUPS = MappingProxyType({(GLASS_ARM, _P1): "glass_pass1", (GLASS_ARM, _P2): "glass_pass2",
                                  (GLASS_ARM, _AB): "glass_abstain", (RF_ARM, _P1): "rf_pass1",
                                  (RF_ARM, _P2): "rf_pass2", (RF_ARM, _AB): "rf_abstain"})


def _consistency(r: Dict[str, Any], tables: Dict[str, pd.DataFrame],
                 sources: BaselineSources) -> List[Dict[str, Any]]:
    trace, refs = r["trace"], r["refs"]
    out: List[Dict[str, Any]] = []

    def check(name, ok, detail=""):
        out.append({"check": name, "passed": bool(ok), "detail": detail})

    st = r["states"]
    y_all, f_all = np.asarray(trace.y), np.asarray(trace.fold_id)
    ok = True
    for (arm, fold), sub in st.groupby([st["arm"], st["fold"].astype(str)], sort=False):
        k = None if fold == "all" else int(fold)
        ok &= len(sub) == len(ROUTER_STATES)
        ok &= int(sub["n"].sum()) == int(sub["rows"].iloc[0]) == (
            trace.n if k is None else int((f_all == k).sum()))
        ok &= int(sub["positives"].sum()) == int(y_all[slice(None) if k is None
                                                       else f_all == k].sum())
    check("three-state accounting: states partition every fold and the pool", ok)

    ratios = tables["ratios"]
    index = _ratio_index(ratios)
    sums: Dict[tuple, List[int]] = {}
    for key, (num, den, status) in index.items():
        if status in ("count", "defined", "undefined_empty_denominator") and not _is_all(key[6]):
            acc = sums.setdefault(key[:6] + key[7:], [0, 0, 0])
            acc[0] += int(num)
            acc[1] += 0 if status == "count" else int(den)
            acc[2] += 1
    bad = checked = 0
    for k, (num, den, n_folds) in sums.items():
        pooled = index.get(k[:6] + ("all",) + k[6:])
        if pooled is None or n_folds != trace.n_folds:
            continue
        checked += 1
        if int(pooled[0]) != num or (pooled[2] != "count" and int(pooled[1]) != den):
            bad += 1
    check("fold vs pooled: every pooled count, numerator and denominator is the sum of its folds",
          bad == 0 and checked > 0, f"{checked} pooled rows checked, {bad} non-additive")

    agree = r["agreement"]
    parity = True
    for key in refs:
        for arm in TRACE_ARMS:
            a = _all(agree[(agree["arm"] == arm) & (agree["reference"] == key)]).set_index("metric")
            theirs = router_upstream_metrics(trace.views[arm], refs[key])
            for m in ("pass1_upstream_flag_rate", "pass2_upstream_redundancy", "covered_agreement",
                      "abstain_upstream_flag_rate", "abstain_upstream_precision"):
                parity &= (int(a.loc[m, "numerator"]), int(a.loc[m, "denominator"])) == (
                    theirs[m].numerator, theirs[m].denominator)
    rr = router_router_table(trace.views[GLASS_ARM], trace.views[RF_ARM])
    parity &= int(_all(r["xsum"]).set_index("metric").loc["cross_state_agreement", "numerator"]) \
        == rr["state_agreement"].numerator
    check("parity with Commit 1: router_upstream_metrics and router_router_table", parity)

    n = {k: refs[k].n_flagged for k in refs}
    check("reference flags: union = LR + EBM - intersection",
          n["union"] == n["lr"] + n["ebm"] - n["intersection"])

    std, goe = r["std"], r["goe"]
    same = True
    cols = [c for c in std.columns if c.startswith(("expected_", "o_minus_e_"))] + ["n", "observed"]
    for (arm, state), group in _STATE_GROUPS.items():
        a = std[(std["arm"] == arm) & (std["state"] == state)].reset_index(drop=True)
        b = goe[goe["group"] == group].reset_index(drop=True)
        same &= len(a) == len(b) and all(
            np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=0, atol=1e-9,
                        equal_nan=True) for c in cols)
    check("state_standardization equals observed_expected for the six router-state groups", same)

    sums = all(abs(float(_all(std[std["arm"] == arm])[f"expected_stratified_{s}"].sum())
                   - trace.n_positives) < 1e-6 for arm in TRACE_ARMS for s in ov.STRATIFIERS)
    check("stratified expectations sum to the observed positives", sums)

    y_perm = np.random.default_rng(5).permutation(np.asarray(trace.y))
    y_perm.setflags(write=False)
    s2 = ov.score_strata(replace(trace, y=y_perm), replace(sources.lr, y=y_perm),
                         replace(sources.ebm, y=y_perm))
    check("score strata are label-free (unchanged under permuted labels)",
          all(np.array_equal(r["strata"].bins[s], s2.bins[s]) for s in ov.STRATIFIERS))

    ms = tables["matched_selection"]
    cp = r["common"]
    ok = bool((ms["selected_rows"] == ms["target_count"]).all()
              and (ms["strict_count"] <= ms["target_count"]).all()
              and (ms["inclusive_count"] >= ms["target_count"]).all())
    for arm, col in ((GLASS_ARM, "glass_pass2_in_common"), (RF_ARM, "rf_pass2_in_common")):
        for model in ("lr", "ebm"):
            k = ms[(ms["arm"] == arm) & (ms["model"] == model)].set_index("fold")["target_count"]
            for fold in range(trace.n_folds):
                ok &= int(k.loc[fold]) == int(cp[cp["fold"].map(
                    lambda f, x=fold: f == x)][col].iloc[0])
    check("matched selections: K = the arm's Pass 2 rows in the common population; selected = K "
          "within the strict / inclusive tie bounds", ok)

    sig = r["signal"]
    unavailable = sig[sig["signal"].str.startswith("analysis:") & (sig["arm"] == "both")]
    check("signal-level Pass 1 / Pass 2 overlap is reported unavailable, final-state overlap zero",
          bool((unavailable[unavailable["signal"].str.contains("candidate|signal overlap")]
                ["status"] == "not_verifiable").all())
          and int(sig[sig["signal"] == "final Pass 1 & Pass 2 rows"]["rows"].sum()) == 0)

    reg = sorted({k[7] for k, (_, _, status) in index.items()
                  if status in ("defined", "undefined_empty_denominator")
                  and k[7] not in METRIC_DEFINITIONS})
    check("every exported ratio is registered in metrics.METRIC_DEFINITIONS", not reg,
          f"unregistered: {reg}" if reg else "")
    if not all(c["passed"] for c in out):
        failed = [f"{c['check']} {c['detail']}".strip() for c in out if not c["passed"]]
        raise BaselineError("baseline consistency checks failed; nothing built:\n  "
                            + "\n  ".join(failed))
    return out


# ----------------------------------------------------------------------
# Manifest pieces
# ----------------------------------------------------------------------
_REALISED = (("min_precision_not_subscribe", "pass1_npv"),
             ("max_subscriber_leakage_rate", "pass1_leakage_share"),
             ("max_subscriber_leakage_absolute", "pass1_positives"),
             ("min_precision_subscribe", "pass2_precision"))


def _constraint_provenance(r: Dict[str, Any]) -> Dict[str, Any]:
    trace = r["trace"]
    rf = trace.views[RF_ARM]
    c = rf.metadata.get("constraints")
    c = dict(c) if isinstance(c, Mapping) else c
    p1 = _all(r["p1pop"][r["p1pop"]["arm"] == GLASS_ARM]).set_index("metric")
    p2 = _all(r["p2pop"][r["p2pop"]["arm"] == GLASS_ARM]).set_index("metric")
    checks = []
    for key, metric in _REALISED:
        rec = c.get(key) if isinstance(c, Mapping) else None
        src = p2 if metric == "pass2_precision" else p1
        num = int(src.loc[metric, "numerator"])
        den = None if metric == "pass1_positives" else int(src.loc[metric, "denominator"])
        realised = float(num) if den is None else (num / den if den else None)
        if rec is None:
            status = "not_recorded"
        elif realised is None:
            status = "glass_value_undefined"
        else:
            status = ("equal" if abs(float(rec) - realised) <= 1e-12 else "differs")
        checks.append({"constraint": key, "recorded": rec, "glass_metric": metric,
                       "glass_numerator": num, "glass_denominator": den,
                       "glass_value": None if realised is None else _round(realised),
                       "status": status})
    return {
        "constraint_source": rf.metadata.get("constraint_source"),
        "recorded_constraints": c,
        "derivation_check": checks,
        "derivation_check_method": ("each recorded constraint with a GLASS counterpart compared "
                                    "with the GLASS Router's pooled outer-OOF value in this "
                                    "baseline (|difference| <= 1e-12 = equal)"),
        "status": ("historical. match_glass_oof makes the RF Router's targets depend on GLASS's "
                   "realised OOF metrics; Commit 4 defines the pre-registered successor and PR 38 "
                   "applies it. Not replaced here."),
    }


def _upstream_points(r: Dict[str, Any], sources: BaselineSources) -> Dict[str, Any]:
    out = {}
    for key, flags in r["refs"].items():
        out[key] = {"operating_point": flags.name, "model": flags.model,
                    "score_space": flags.score_space, "threshold": flags.threshold,
                    "field": flags.field, "selection": flags.selection,
                    "disclosure": flags.disclosure, "components": list(flags.components),
                    "flags": flags.n_flagged}
    lr_thr = sources.lr.thresholds.get("lr.in_stage_f2_cv")
    if lr_thr is not None:
        out["lr"]["threshold_record"] = {"value": lr_thr.value, "field": lr_thr.field,
                                         "selection": lr_thr.selection, "detail": lr_thr.detail}
    nested = sources.ebm.stored_decisions.get("ebm.nested_f2")
    if nested is not None:
        out["ebm"]["fold_thresholds"] = {str(k): float(v) for k, v in enumerate(nested.by_fold)}
        out["ebm"]["fold_thresholds_note"] = ("raw-score thresholds by router outer fold, each "
                                              "selected on inner OOF scores of that fold's training "
                                              "rows")
    return out


def _router_records(trace) -> Dict[str, Any]:
    out = {}
    for arm in TRACE_ARMS:
        v = trace.views[arm]
        meta = dict(v.metadata)
        if "rf_params_origin" in meta:
            meta["rf_params_origin"] = ("recorded in the artifact as a local absolute path; the "
                                        f"file is the source with role {ROLE_RF_RESULT!r}")
        out[arm] = {
            "router": ARM_LABELS[arm], "model_family": v.model_family,
            "router_mode": v.router_mode, "pass2_population": dict(v.pass2_population),
            "confidence_kind": v.confidence_kind, "metadata": meta,
            "fold_thresholds": (None if v.fold_thresholds is None else
                                {str(k): dict(x) for k, x in v.fold_thresholds.items()}),
            "provenance": [{"name": p.name, "status": p.status, "detail": p.detail}
                           for p in v.provenance],
        }
    return out


def _settings(r: Dict[str, Any]) -> Dict[str, Any]:
    strata, trace = r["strata"], r["trace"]
    fold = np.asarray(trace.fold_id)
    tp = {"lr": {}, "ebm": {}}
    for k in range(trace.n_folds):
        m = fold == k
        tp["lr"][str(k)] = _round(float(np.asarray(strata.lr_threshold_percentile)[m][0]))
        tp["ebm"][str(k)] = _round(float(np.asarray(strata.ebm_threshold_percentile)[m][0]))
    return {
        "score_bins": ov.SCORE_BINS, "joint_bins": ov.JOINT_BINS, "min_support": ov.MIN_SUPPORT,
        "threshold_band_within_fold_percentile": ov.BOUNDARY_BAND,
        "stratifiers": list(ov.STRATIFIERS),
        "strata_requested_bins": dict(strata.requested_bins),
        "strata_occupied_bins_by_fold": {s: {str(k): int(v) for k, v in strata.n_bins[s].items()}
                                         for s in ov.STRATIFIERS},
        "threshold_within_fold_percentile_by_fold": tp,
        "strata_provenance": dict(strata.provenance),
        "strata_fixed": "before any Step 2D result was inspected; label-free; never tuned",
        "matched_rate": {
            "population": "common eligible (rows neither router routed to Pass 1)",
            "k": "the arm's Pass 2 rows in each router fold's common eligible population",
            "ranking": "highest score first", "scores": dict(MATCH_SCORES),
            "tie_break": TIE_BREAK_RULE,
            "tie_bounds": "strict / inclusive cuts (operating_points.matched_rate_cut); neither "
                          "preferred"},
        "router_transform_atol": TRANSFORM_ATOL,
        "float_format": f"{FLOAT_FORMAT} (12 significant digits)",
    }


def _disclosures(r: Dict[str, Any]) -> List[Dict[str, str]]:
    trace, strata = r["trace"], r["strata"]
    g, rf = trace.views[GLASS_ARM], trace.views[RF_ARM]
    cs, c = rf.metadata.get("constraint_source"), rf.metadata.get("constraints")
    merged = {s: [k for k, v in strata.n_bins[s].items() if v < strata.requested_bins[s]]
              for s in ov.STRATIFIERS}
    occ = ", ".join(f"fold {k}: {v}" for k, v in strata.n_bins["lr"].items())
    items = [
        ("definitions", "Metric formulas and denominators: every ratio has its numerator, "
         "denominator, value and status in the table that reports it; ratio names are defined in "
         "metrics.METRIC_DEFINITIONS (definitions.json 'metrics'); wide-table columns are defined "
         "in definitions.json 'tables'. Undefined ratios stay undefined (empty value, status "
         "undefined_empty_denominator), never 0."),
        ("pass1_precedence", "Pass 1 precedence and Pass 2 eligibility: Pass 1 claims rows first; "
         "Pass 2 only claims rows Pass 1 left; everything else abstains. Final Pass 1 and Pass 2 "
         "never overlap by construction. For the RF Router this is verified from the artifacts "
         "(Pass 1 score equivalence); for GLASS it is code-level only (no per-fold rule sets)."),
        ("leakage", "Positive leakage: Pass 1 actual positives (routed NOT_SUBSCRIBE) / all "
         "positives (pass1_leakage_share); within-band leakage is Pass 1 positives / Pass 1 rows "
         "(pass1_positive_rate); Pass 1 NPV is Pass 1 negatives / Pass 1 rows."),
        ("reference_flags", "Frozen reference flags: lr.in_stage_f2_cv (LR calibrated "
         "probability >= its stored threshold, selected on ALL training labels: in-selection, "
         "rates optimistic) and ebm.nested_f2 (stored fold-nested EBM raw-score decisions). Both "
         "come from the GLASS arm and are the common reference for both routers; the RF Router's "
         "own black-box cascade uses other upstream models."),
        ("overall_vs_matched", "Overall (frozen) comparisons keep each side's own flag rate on all "
         "rows; matched-rate comparisons re-cut LR / EBM at each arm's Pass 2 count K per fold on "
         "the common eligible population, label-free. K is the router's observed count (a "
         "descriptive control, not a pre-registration); the two arms have different K."),
        ("ebm_fold_local", "EBM raw scores are never pooled or ranked across folds (their "
         "fold-specific nested thresholds show the scales differ): EBM strata, ranks and "
         "threshold positions are within the router outer fold."),
        ("ties", "LR calibrated probabilities are isotonic, so large tied blocks occur. Strata keep "
         "tied scores in one bin (merged edges). Matched selections break ties at the K-th score "
         f"by a predetermined label-free rule ({TIE_BREAK_RULE}) and report strict / inclusive "
         "bounds."),
        ("expectations", "Two kinds of expected positives: calibrated model expectations (sums of "
         "LR or EBM calibrated OOF probabilities; assume the calibration is right) and internal "
         "standardisation (expected_stratified_*: the group's rows' (fold, stratum) positive rates "
         "on the same labels). Neither is a null distribution or a test."),
        ("low_support", f"Cells with fewer than {ov.MIN_SUPPORT} rows are flagged low_support and "
         "excluded from within-stratum contrasts (difference left empty)."),
        ("signal_overlap", "Final-state Pass 1 / Pass 2 overlap is available and zero by "
         "construction (a routing invariant, not evidence of independent signals). Pre-routing "
         "signal overlap is unavailable: GLASS candidate-rule matches and per-fold rule sets, and "
         "RF Pass 2 outer scores, were never persisted. The full-train GLASS rule set is not "
         "applied to training rows (in-sample)."),
        ("shared_fold_rf", "Shared component, code-level only: crossfit_stage2 defaults to "
         "glass_rf_model='fold_rf', passing each outer fold's RF to the GLASS fit (segment "
         "prioritisation by its feature_importances_); that fold forest's oof_proba is the RF "
         "Router's Pass 1 score. The artifacts' own record of glass_rf_model is not read."),
        ("pass2_populations", "Pass 2 fit-time populations differ historically: GLASS "
         f"{g.pass2_population.get('normalised')!r}, RF {rf.pass2_population.get('normalised')!r} "
         "(F17, Commit 3). Pass 2 differences are not attributable to the selector alone."),
        ("match_glass_oof", f"RF constraint_source={cs!r} with recorded constraints "
         f"{dict(c) if isinstance(c, Mapping) else c}: derived from GLASS's realised OOF metrics "
         "(MANIFEST.json rf_constraint_provenance), so the arms are dependent and not exactly "
         "rate-matched. Commit 4 defines the successor; PR 38 applies it."),
        ("glass_unoptimised", "The GLASS configuration is historical and not optimised; this is a "
         "baseline, not its performance potential."),
        ("test_exposure", HISTORICAL_TEST_EXPOSURE + " No test label, test prediction or "
         "test-derived threshold is read or exported here."),
        ("scope", "No claim of Stage 3 / Stage 4 or integrated-cascade benefit, no independent "
         "external validation, no ranking of the routers, no significance test. Fold values are "
         "descriptive, not independent replicates; no standard errors are given."),
        ("joint_grid_threshold_groups", "Known limitation (identified after Step 2D; definitions "
         f"unchanged): the {ov.JOINT_BINS} x {ov.JOINT_BINS} joint grid splits each score at "
         "within-fold tertiles, so its top cell contains rows on both sides of both reference "
         "thresholds. For populations defined by those thresholds (reference-flag groups and "
         "'unique vs' groups), expected_stratified_lr_x_ebm compares them with rows across the "
         "threshold and is not interpretable; use the calibrated expectations. Decile "
         "expectations share the bias to a smaller degree."),
        ("pooled_lr_strata", "Known limitation (identified after Step 2D; definitions unchanged): "
         "pooled ('all') stratum rows sum bin k of every fold, and where LR isotonic ties merge "
         "decile edges the bin indices shift, so pooled LR bin k covers different score ranges "
         f"in different folds (LR occupied bins by fold: {occ}; merged in folds "
         f"{merged['lr'] or 'none'}). Pooled LR strata and LR-conditioned pooled contrasts are "
         "approximate; per-fold cells and all expectations are unaffected; the EBM strata are "
         f"{'aligned' if not merged['ebm_raw'] else 'also affected in folds ' + str(merged['ebm_raw'])}."),
        ("code_identity", "The artifacts do not record the code that produced them. MANIFEST.json "
         "'code' identifies the diagnostic and export code at generation (LF-normalised SHA-256), "
         "not the router implementations; router identity is the artifacts' own record "
         "(MANIFEST.json 'routers')."),
    ]
    return [{"id": f"D{i:02d}", "topic": t, "text": x} for i, (t, x) in enumerate(items, 1)]


def _notes(r: Dict[str, Any]) -> Dict[str, List[str]]:
    return {"2A.compare_state_distributions": list(r["compare"].notes),
            "2B.pass1_suppression_report": list(r["supp"].notes),
            "2C.pass2_frozen_contribution": list(r["frozen"].notes),
            "2C.pass2_matched_contribution": list(r["matched"].notes),
            "2D.overlap_notes": list(r["notes"])}


def _definitions(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ratios = tables["ratios"]
    metrics: Dict[str, Any] = {}
    for m in sorted(set(ratios["metric"])):
        rows = ratios[ratios["metric"] == m]
        kinds = sorted(set(rows["status"]))
        labels = sorted(set(str(x) for x in rows["label"]))
        if m in METRIC_DEFINITIONS:
            num, den = METRIC_DEFINITIONS[m]
            metrics[m] = {"kind": "ratio", "numerator": num, "denominator": den,
                          "registry": "shared.stage2.diagnostics.metrics.METRIC_DEFINITIONS",
                          "labels": labels}
        else:
            metrics[m] = {"kind": "quantile" if kinds == ["quantile"] else "count",
                          "labels": labels}
    for m, (num, den) in METRIC_DEFINITIONS.items():
        metrics.setdefault(m, {"kind": "ratio", "numerator": num, "denominator": den,
                               "registry": "shared.stage2.diagnostics.metrics."
                                           "METRIC_DEFINITIONS", "labels": []})
    out_tables = {}
    for name, (path, step, producer, desc, keys) in TABLES.items():
        cols = list(tables[name].columns)
        spec = {**_COMMON_COLUMNS, **_TABLE_COLUMNS.get(name, {})}
        missing = [c for c in cols if c not in spec]
        if missing:
            raise BaselineError(f"{name}: no definition for columns {missing}")
        entry = {"file": path, "step": step, "producer": producer, "description": desc,
                 "keys": list(keys), "columns": {c: spec[c] for c in cols}}
        if name == "ratios":
            entry["sections"] = {s: p for s, p in _RATIO_SECTIONS}
        out_tables[name] = entry
    return {
        "baseline_id": BASELINE_ID,
        "conventions": {
            "fold": _COMMON_COLUMNS["fold"],
            "empty_cell": "not applicable or undefined; a ratio's status says which. Never a "
                          "substitute for 0.",
            "booleans": "true / false",
            "floats": f"{FLOAT_FORMAT}: 12 significant digits",
            "rows": "in the producing function's order (deterministic)",
            "denominators": ("every ratio's denominator is explicit. 'routed only' in a label "
                             "means the denominator excludes abstained rows; otherwise it is the "
                             "stated population (all rows of the fold / pool unless labelled)."),
            "producers": "module paths are relative to shared.stage2.diagnostics",
        },
        "metrics": metrics,
        "tables": out_tables,
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _i(x) -> str:
    return f"{int(x):,}"


def _pct(num, den, digits=2) -> str:
    return "undefined (empty denominator)" if not den else f"{100.0 * num / den:.{digits}f}%"


def _rat(num, den, digits=2) -> str:
    if not den:
        return f"{_i(num)}/0 (undefined)"
    return f"{_i(num)}/{_i(den)} = {_pct(num, den, digits)}"


def _f1(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):,.1f}"


def _signed(x) -> str:
    return "0" if int(x) == 0 else f"{int(x):+,}"


def _row(df, **keys) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in keys.items():
        m &= df[k].map(lambda x, v=v: x == v).to_numpy(bool)
    sub = df[m]
    if len(sub) != 1:
        raise BaselineError(f"report: expected one row for {keys}, found {len(sub)}")
    return sub.iloc[0]


def _md(header: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


_ARMS_TXT = MappingProxyType({GLASS_ARM: "GLASS Router", RF_ARM: "RF Router"})
_STATE_TXT = MappingProxyType({_P1: "Pass 1 (NOT_SUBSCRIBE)", _P2: "Pass 2 (SUBSCRIBE)",
                               _AB: "Abstain"})
_THRESHOLD_GROUPS = ("flag", "unique_vs")


def _report(r: Dict[str, Any], tables: Dict[str, pd.DataFrame], manifest: Dict[str, Any]) -> str:
    trace = r["trace"]
    index = _ratio_index(tables["ratios"])

    def ratio(section, metric, *, arm="", reference="", population="", group="", score="",
              fold="all"):
        key = (section, arm, reference, population, group, score, fold, metric)
        if key not in index:
            raise BaselineError(f"report: no ratios row {key}")
        num, den, status = index[key]
        return int(num), (None if status == "count" else int(den))

    L: List[str] = []
    add = L.append
    add(f"# {BASELINE_LABEL} — `{BASELINE_ID}`")
    add("")
    add(f"Format `{FORMAT}` v{FORMAT_VERSION} · designation **{DESIGNATION}** · status "
        "**frozen** · PR 37, Commit 2, Step 2E")
    add("")
    add("This package is the frozen historical reference for the GLASS Router and the RF Router "
        "**before any corrected-selector change**. It records what the historical artifacts show "
        f"on the outer out-of-fold outputs of the {_i(trace.n)} training rows. PR 38's "
        "corrected-selector results are compared with it; they never replace it. A later "
        "baseline is a new version with a new identifier and directory: this package is never "
        "edited or regenerated in place, and later interpretation is added as a separate "
        "research log that cites it.")
    add("")
    add("It is **not**: corrected-selector performance; a held-out or test-set evaluation; a "
        "measurement of Stage 3, Stage 4 or the integrated cascade; a ranking of the routers; or "
        "a significance test. Every number is a historical reference anchor, not a target to "
        "optimise.")
    add("")

    # 1 identity
    ids = manifest["identities"]
    pop = manifest["population"]
    add("## 1. Identity and provenance")
    add("")
    snap = manifest.get("snapshot")
    rows = [
        ["Training rows / positives / outer folds",
         f"{_i(pop['training_rows'])} / {_i(pop['positives'])} / {pop['outer_folds']}",
         "validated"],
        ["Rows per outer fold", ", ".join(_i(x) for x in pop["rows_per_fold"]), "validated"],
        ["Router split fingerprint", f"`{ids['router_split_fingerprint']['value']}`",
         ids["router_split_fingerprint"]["status"]],
        ["Training-label fingerprint", f"`{ids['train_label_fingerprint']['value']}`",
         ids["train_label_fingerprint"]["status"]],
        ["Outer fold plan", f"seed {ids['fold_plan']['seed']}, fingerprint "
         f"`{ids['fold_plan']['fingerprint']}`", ids["fold_plan"]["status"]],
        ["stage_io split fingerprint", f"`{ids['stage_io_split_fingerprint']['value']}`",
         f"{ids['stage_io_split_fingerprint']['status']} (never recomputed: its hash includes "
         "test labels)"],
        ["Snapshot SHA256SUMS", "not supplied" if not snap else
         f"{snap['label']} ({snap['entries']} entries), `{snap['sha256'][:16]}…`",
         "attested" if snap else "-"],
    ]
    add(_md(["Item", "Value", "Status"], rows, "lll"))
    add("")
    add("Source artifacts (SHA-256 prefix; full values in `MANIFEST.json`):")
    add("")
    add(_md(["Role", "File (relative to bank_pipeline/)", "SHA-256", "Snapshot"],
            [[s["role"], f"`{s['path']}`", f"`{s['sha256'][:16]}…`",
              {True: "attested", False: "MISMATCH", None: "not attested"}[s["snapshot_attested"]]]
             for s in manifest["sources"]], "llll"))
    add("")
    add(f"Diagnostic code identity: LF-normalised SHA-256 of {len(manifest['code']['files'])} "
        "files (`MANIFEST.json` → `code`). The artifacts do not record the code that produced "
        "them; router identity is the artifacts' own record (`MANIFEST.json` → `routers`).")
    add("")

    # 2 operating points
    add("## 2. Frozen reference operating points and the RF constraint source")
    add("")
    rs = tables["reference_flags"]
    up = manifest["upstream_operating_points"]
    rows = []
    for rec in rs.itertuples(index=False):
        if rec.reference == "ebm" and "fold_thresholds" in up.get("ebm", {}):
            thr = "by fold: " + ", ".join(f"{v:g}" for v in up["ebm"]["fold_thresholds"].values())
        elif rec.threshold is None or (isinstance(rec.threshold, float)
                                       and math.isnan(rec.threshold)):
            thr = "-"
        else:
            thr = f"{float(rec.threshold):g}"
        rows.append([f"`{rec.reference}`", f"`{rec.operating_point}`", thr, _i(rec.flags),
                     _i(rec.tp), _rat(int(rec.tp), int(rec.flags))])
    add(_md(["Reference", "Operating point", "Threshold", "Flags", "TP", "Precision"], rows,
            "lllrrr"))
    add("")
    add("LR's threshold was selected on all training labels (in-selection; its rates are "
        "optimistic). EBM's decisions are fold-nested: each fold's raw-score threshold was "
        "selected on inner OOF scores of that fold's training rows.")
    add("")
    cp = manifest["rf_constraint_provenance"]
    add(f"RF Router `constraint_source = {cp['constraint_source']!r}`. Recorded constraints "
        "against the GLASS Router's pooled outer-OOF values in this baseline:")
    add("")
    add(_md(["Constraint", "Recorded", "GLASS realised", "Status"],
            [[f"`{c['constraint']}`", f"{c['recorded']}" if c["recorded"] is not None else "-",
              (f"{_i(c['glass_numerator'])}" if c["glass_denominator"] is None else
               f"{_i(c['glass_numerator'])}/{_i(c['glass_denominator'])}"), c["status"]]
             for c in cp["derivation_check"]], "lrrl"))
    add("")

    # 3 principal results
    add("## 3. Principal results (outer OOF, pooled; per-fold values in `tables/`)")
    add("")
    add("### 3.1 Router states")
    add("")
    st = _all(tables["states"])
    rows = []
    for s in ROUTER_STATES:
        cells = [_STATE_TXT[s]]
        for arm in TRACE_ARMS:
            x = _row(st, arm=arm, state=s)
            cells += [f"{_i(x['n'])} ({_pct(x['n'], x['rows'])})",
                      f"{_i(x['positives'])} ({_pct(x['positives'], x['n'])})"]
        rows.append(cells)
    for arm_metric in ("coverage", "abstain_rate"):
        cells = ["Coverage (Pass 1 + Pass 2)" if arm_metric == "coverage" else "Abstention rate"]
        for arm in TRACE_ARMS:
            n_, d_ = ratio("2A.routing_rates", arm_metric, arm=arm)
            cells += [_rat(n_, d_), ""]
        rows.append(cells)
    add(_md(["State", "GLASS rows (share)", "GLASS positives (rate)", "RF rows (share)",
             "RF positives (rate)"], rows))
    add("")

    add("### 3.2 Pass 1 (NOT_SUBSCRIBE): NPV, leakage and suppression of the reference flags")
    add("")
    rows = []
    for label, sec, metric, ref in (
            ("Pass 1 NPV", "2B.pass1_population", "pass1_npv", ""),
            ("Positive leakage (Pass 1 positives / all positives)", "2B.pass1_population",
             "pass1_leakage_share", ""),
            ("Union-flagged rows routed to Pass 1", "2B.pass1_suppression", "intercepted_flags",
             "union"),
            ("... of which actual positives (suppressed TP)", "2B.pass1_suppression",
             "suppressed_tp", "union"),
            ("Union FP suppression (intercepted FP / union FP)", "2B.pass1_suppression",
             "pass1_fp_suppression", "union"),
            ("Union TP cost (suppressed TP / union TP)", "2B.pass1_suppression", "pass1_tp_cost",
             "union"),
            ("LR-flagged rows routed to Pass 1", "2B.pass1_suppression", "intercepted_flags", "lr"),
            ("... of which actual positives", "2B.pass1_suppression", "suppressed_tp", "lr"),
            ("EBM-flagged rows routed to Pass 1", "2B.pass1_suppression", "intercepted_flags",
             "ebm"),
            ("... of which actual positives", "2B.pass1_suppression", "suppressed_tp", "ebm")):
        cells = [label]
        for arm in TRACE_ARMS:
            n_, d_ = ratio(sec, metric, arm=arm, reference=ref)
            cells.append(_i(n_) if d_ is None else _rat(n_, d_))
        rows.append(cells)
    add(_md(["", "GLASS Router", "RF Router"], rows))
    add("")

    add("### 3.3 Pass 2 (SUBSCRIBE) against the frozen references (all rows)")
    add("")
    rows = []
    for label, metric, ref in (("Pass 2 rows", "pass2_n", None), ("Pass 2 TP", "pass2_tp", None),
                               ("Pass 2 FP", "pass2_fp", None),
                               ("Pass 2 precision", "pass2_precision", None),
                               ("Pass 2 recall (all positives)", "pass2_recall", None),
                               ("Unique vs union: TP", "unique_tp", "union"),
                               ("Unique vs union: FP", "unique_fp", "union"),
                               ("Marginal precision vs union", "marginal_precision", "union"),
                               ("Unique vs LR: TP", "unique_tp", "lr"),
                               ("Unique vs EBM: TP", "unique_tp", "ebm")):
        cells = [label]
        for arm in TRACE_ARMS:
            if ref is None:
                n_, d_ = ratio("2C.pass2_population", metric, arm=arm)
            else:
                n_, d_ = ratio("2C.pass2_frozen", metric, arm=arm, reference=ref,
                               population="all_rows")
            cells.append(_i(n_) if d_ is None else _rat(n_, d_))
        rows.append(cells)
    add(_md(["", "GLASS Router", "RF Router"], rows))
    add("")

    add("### 3.4 Common eligible population and matched flag rate")
    add("")
    c = _row(tables["common_population"], fold="all")
    add(f"Common eligible population: {_i(c['common_rows'])} rows ({_i(c['common_positives'])} "
        f"positives) that neither router routed to Pass 1. It excludes "
        f"{_i(c['glass_pass2_outside_common'])} GLASS Pass 2 rows that RF routed to Pass 1 and "
        f"{_i(c['rf_pass2_outside_common'])} RF Pass 2 rows that GLASS routed to Pass 1. Within "
        "each router fold, LR / EBM are re-cut to the arm's K highest-scoring rows (label-free; "
        f"ties: {TIE_BREAK_RULE}).")
    add("")
    rows = []
    for arm in TRACE_ARMS:
        for ref in ("lr_matched", "ebm_matched"):
            g = lambda m: ratio("2C.pass2_matched", m, arm=arm, reference=ref,  # noqa: E731
                                population="common_eligible")[0]
            rows.append([_ARMS_TXT[arm], f"`{ref}`", _i(g("pass2_n")), _i(g("pass2_tp")),
                         _i(g("reference_tp")), _signed(g("pass2_tp") - g("reference_tp")),
                         f"{_i(g('unique_tp'))} [{_i(g('unique_tp_strict'))} – "
                         f"{_i(g('unique_tp_inclusive'))}]",
                         _i(g("tie_block")), _i(g("tie_broken_rows"))])
    add(_md(["Router", "Matched reference", "K (= Pass 2 rows)", "Pass 2 TP", "Reference TP",
             "Pass 2 − reference TP", "Unique TP [tie bounds]", "Tie block", "Tie-broken rows"],
            rows, "llrrrrrrr"))
    add("")
    add("K differs between the arms, so each arm is compared with its own matched references; "
        "this is not an equal-count comparison of the two routers.")
    add("")

    add("### 3.5 Agreement with the references (union of LR and EBM flags)")
    add("")
    rows = []
    for metric in ("coverage", "pass1_upstream_flag_rate", "pass2_upstream_redundancy",
                   "pass2_upstream_disagreement", "covered_agreement",
                   "abstain_upstream_flag_rate", "upstream_flags_in_pass1_share",
                   "upstream_flags_in_pass2_share", "upstream_flags_in_abstain_share"):
        num, den = METRIC_DEFINITIONS[metric]
        cells = [f"`{metric}` ({num} / {den})"]
        for arm in TRACE_ARMS:
            cells.append(_rat(*ratio("2D.router_reference_agreement", metric, arm=arm,
                                     reference="union")))
        rows.append(cells)
    add(_md(["Metric (numerator / denominator)", "GLASS Router", "RF Router"], rows))
    add("")

    add("### 3.6 GLASS state × RF state (rows / actual positives)")
    add("")
    xt = _all(tables["crosstab"])
    rows = []
    for gs in ROUTER_STATES:
        cells = [f"GLASS {_STATE_TXT[gs]}"]
        for rs_ in ROUTER_STATES:
            x = _row(xt, glass_state=gs, rf_state=rs_)
            cells.append(f"{_i(x['n'])} / {_i(x['positives'])}")
        rows.append(cells)
    add(_md(["", *[f"RF {_STATE_TXT[s]}" for s in ROUTER_STATES]], rows))
    add("")
    rows = []
    for metric in ("cross_state_agreement", "cross_both_routed_rate", "cross_routed_agreement",
                   "cross_opposite_rate", "cross_one_abstains_rate", "cross_both_abstain_rate"):
        num, den = METRIC_DEFINITIONS[metric]
        rows.append([f"`{metric}`", f"{num} / {den}", _rat(*ratio("2D.crosstab_summary", metric))])
    add(_md(["Metric", "Numerator / denominator", "Value"], rows, "llr"))
    add("")

    add("### 3.7 Cross-router Pass 2 discoveries (unique relative to the LR / EBM union)")
    add("")
    p = _row(tables["pass2_cross_router"], reference="union", fold="all")
    add(_md(["", "Both routers", "GLASS only", "RF only"],
            [["Actual positives", _i(p["shared_tp"]), _i(p["glass_only_tp"]), _i(p["rf_only_tp"])],
             ["Actual negatives", _i(p["shared_fp"]), _i(p["glass_only_fp"]),
              _i(p["rf_only_fp"])]]))
    add("")
    add(f"GLASS-only positives that RF routed to Pass 1 / abstained on: "
        f"{_i(p['glass_only_tp_in_rf_pass1'])} / {_i(p['glass_only_tp_in_rf_abstain'])}; "
        f"RF-only positives that GLASS routed to Pass 1 / abstained on: "
        f"{_i(p['rf_only_tp_in_glass_pass1'])} / {_i(p['rf_only_tp_in_glass_abstain'])}.")
    add("")

    add("### 3.8 Observed vs expected positives (descriptive, not tests)")
    add("")
    goe = _all(tables["observed_expected"]).set_index("group")
    rows = []
    for gname in goe.index:
        x = goe.loc[gname]
        mark = " †" if any(t in gname for t in _THRESHOLD_GROUPS) else ""
        rows.append([f"`{gname}`{mark}", _i(x["n"]), _i(x["observed"]),
                     _f1(x["expected_lr_calibrated"]), _f1(x["expected_ebm_calibrated"]),
                     _f1(x["expected_stratified_ebm_raw"]),
                     _f1(x["expected_stratified_lr_x_ebm"])])
    add(_md(["Group", "Rows", "Observed", "LR-calibrated", "EBM-calibrated",
             "EBM-decile stratified", "Joint-grid stratified"], rows))
    add("")
    add("† Defined by the reference thresholds: the joint-grid expectation is not interpretable "
        "for these groups (section 5).")
    add("")

    # 4 Step 2D findings
    add("## 4. Step 2D findings (validated summary)")
    add("")
    cov = {arm: ratio("2A.routing_rates", "coverage", arm=arm) for arm in TRACE_ARMS}
    xs = {m: ratio("2D.crosstab_summary", m) for m in
          ("cross_state_agreement", "cross_routed_agreement", "cross_one_abstains_rate",
           "cross_opposite_rate")}
    g1r2 = _row(xt, glass_state=_P1, rf_state=_P2)
    g2r1 = _row(xt, glass_state=_P2, rf_state=_P1)
    inter = {arm: (ratio("2B.pass1_suppression", "intercepted_flags", arm=arm,
                         reference="union")[0],
                   ratio("2B.pass1_suppression", "suppressed_tp", arm=arm, reference="union")[0])
             for arm in TRACE_ARMS}
    add(f"- Coverage: GLASS {_rat(*cov[GLASS_ARM])}; RF {_rat(*cov[RF_ARM])}.")
    add(f"- Same three-state decision: {_rat(*xs['cross_state_agreement'])}. Agreement when both "
        f"route (routed only): {_rat(*xs['cross_routed_agreement'])}. Exactly one router "
        f"abstains: {_rat(*xs['cross_one_abstains_rate'])}. Opposite routing decisions: "
        f"{_rat(*xs['cross_opposite_rate'])}.")
    add(f"- GLASS Pass 1 / RF Pass 2: {_i(g1r2['n'])} rows, {_i(g1r2['positives'])} actual "
        f"positives. GLASS Pass 2 / RF Pass 1: {_i(g2r1['n'])} rows, {_i(g2r1['positives'])} "
        "actual positives.")
    add(f"- Pass 1 interceptions of the LR / EBM union flags: GLASS {_i(inter[GLASS_ARM][0])} rows "
        f"({_i(inter[GLASS_ARM][1])} actual positives); RF {_i(inter[RF_ARM][0])} rows "
        f"({_i(inter[RF_ARM][1])} actual positives).")
    add(f"- Pass 2 positives the union misses: GLASS {_i(p['glass_unique_tp'])}, RF "
        f"{_i(p['rf_unique_tp'])}; of these {_i(p['shared_tp'])} are found by both routers, "
        f"{_i(p['glass_only_tp'])} by GLASS only and {_i(p['rf_only_tp'])} by RF only, with "
        f"{_i(p['shared_fp'])} / {_i(p['glass_only_fp'])} / {_i(p['rf_only_fp'])} accompanying "
        "negatives.")
    oe = [(g, float(goe.loc[g, "o_minus_e_ebm_calibrated"]),
           float(goe.loc[g, "expected_ebm_calibrated"])) for g in _STATE_GROUPS.values()]
    worst = max(oe, key=lambda t: abs(t[1]))
    rel = max(abs(o) / e for _, o, e in oe if e)
    add(f"- Router states against EBM's calibrated expectation: the largest difference is "
        f"{worst[1]:+,.1f} positives (`{worst[0]}`); every state is within "
        f"{100 * rel:.1f}% of its expectation.")
    add("")
    add("Interpretation (as of this version): the routers show different functional behaviour, "
        "especially in abstention and deferral rather than direct contradiction. Useful "
        "incremental predictive information beyond the LR and EBM references, and any Stage 4 "
        "benefit, remain unproven. Calibrated EBM expectations explain several observed "
        "disagreement populations reasonably well. The larger deviations in the coarse joint "
        "LR × EBM strata do not establish predictive independence, because within-bin score "
        "variation remains possible (and, for threshold-defined groups, the grid is not "
        "applicable: section 5). Descriptive expectations, chance arithmetic and empirical "
        "stratification are not statistical significance tests. This baseline describes the "
        "historical routers; it is not, and does not predict, corrected-selector performance.")
    add("")

    # 5 limitations
    add("## 5. Known limitations of the exported Step 2D tables")
    add("")
    for d in manifest["disclosures"]:
        if d["topic"] in ("joint_grid_threshold_groups", "pooled_lr_strata"):
            add(f"- **{d['id']}** {d['text']}")
    tpct = manifest["analysis_settings"]["threshold_within_fold_percentile_by_fold"]
    add(f"- Reference thresholds by within-fold percentile: LR "
        f"{', '.join(f'{100 * v:.1f}' for v in tpct['lr'].values())}; EBM "
        f"{', '.join(f'{100 * v:.1f}' for v in tpct['ebm'].values())} (folds 0-4); the joint "
        f"grid's edges are the within-fold 1/{ov.JOINT_BINS} and 2/{ov.JOINT_BINS} quantiles.")
    low = int((tables["stratum_contrasts"]["status"] == "low_support").sum())
    add(f"- Low-support contrasts (either state under {ov.MIN_SUPPORT} rows): {_i(low)}, left "
        "empty in `tables/stratum_contrasts.csv`.")
    add("")

    # 6 definitions
    add("## 6. Definitions")
    add("")
    for line in (
            "**Router states.** Each training row has exactly one state per router: Pass 1 "
            "(NOT_SUBSCRIBE), Pass 2 (SUBSCRIBE) or Abstain. Pass 1 claims first; Pass 2 only "
            "claims rows Pass 1 left.",
            "**Coverage** = (Pass 1 + Pass 2 rows) / all rows. **Pass 1 NPV** = Pass 1 negatives / "
            "Pass 1 rows. **Positive leakage** = Pass 1 positives / all positives. **Pass 2 "
            "precision** = Pass 2 positives / Pass 2 rows.",
            "**Suppression.** An intercepted reference flag is a flagged row routed to Pass 1: an "
            "intercepted false positive is a potential benefit (pass1_fp_suppression = "
            "intercepted FP / reference FP); a suppressed true positive is an error "
            "(pass1_tp_cost = suppressed TP / reference TP). They are never netted.",
            "**Unique positives** are Pass 2 positives the stated reference does not flag. The "
            "hypothetical OR (Pass 2 or the reference) is not Stage 4's behaviour.",
            "**Denominators.** Rates over all rows of the fold / pool unless the label says "
            "otherwise; 'routed only' rates exclude abstained rows. Undefined ratios are left "
            "empty with status undefined_empty_denominator, never 0.",
            "**Folds.** 'all' rows pool the five router outer folds; per-fold values are "
            "descriptive, not independent replicates.",
            "Every metric, table and column: `definitions.json`."):
        add(f"- {line}")
    add("")

    # 7 disclosures
    add("## 7. Research disclosures")
    add("")
    for d in manifest["disclosures"]:
        if d["topic"] not in ("joint_grid_threshold_groups", "pooled_lr_strata"):
            add(f"- **{d['id']}** {d['text']}")
    add("")

    # 8 files
    add("## 8. Files, verification and versioning")
    add("")
    rows = [[f"`{p}`", _i(o["rows"]) if "rows" in o else "-", o["description"]]
            for p, o in manifest["outputs"].items() if p != REPORT]
    rows.append([f"`{REPORT}`", "-", "this report"])
    rows.append([f"`{MANIFEST}` / `{SIDECAR}`", "-",
                 "identity, provenance, settings, output hashes / the manifest's SHA-256"])
    add(_md(["File", "Rows", "Contents"], rows, "lrl"))
    add("")
    add("From `prototype/bank_pipeline/`:")
    add("")
    add("```")
    add(f"python {BASELINES_DIR}/stage2_baseline.py --verify   # integrity only; no artifacts; "
        "writes nothing")
    add(f"python {BASELINES_DIR}/stage2_baseline.py --check    # rebuild from the local "
        "historical artifacts; compare; writes nothing")
    add("```")
    add("")
    add("`--verify` detects any change to the package's files. `--check` needs the historical "
        "artifacts recorded in `MANIFEST.json` and the snapshot SHA256SUMS "
        "(`PR37_BASELINE_SHA256SUMS`) and also detects changed source artifacts, fingerprints or "
        "results. A new baseline is a new version (new identifier, new directory); this one is "
        "never overwritten.")
    add("")
    return "\n".join(L)


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build_baseline(sources: BaselineSources, code: CodeIdentity, *,
                   baseline_id: str = BASELINE_ID) -> BaselinePackage:
    """
    Every file of the package, in memory. Reads no artifact (the views are the
    validated extractors' output) and writes nothing. Refuses on any failed
    consistency check, forbidden field, unscrubbed absolute path or undefined column.
    """
    _check_sources(sources)
    if not isinstance(code, CodeIdentity):
        raise BaselineError("expected a CodeIdentity (code_identity(bank_root))")
    r = _compute(sources)
    tables = _tables(r, sources)
    consistency = _consistency(r, tables, sources)
    trace = r["trace"]
    files: Dict[str, bytes] = {}
    outputs: Dict[str, Any] = {}

    for name, (path, *_rest) in TABLES.items():
        data = _csv_bytes(tables[name])
        files[path] = data
        outputs[path] = {"sha256": _sha(data), "bytes": len(data), "rows": int(len(tables[name])),
                         "columns": [str(c) for c in tables[name].columns],
                         "description": TABLES[name][3]}
    defs = _json_bytes(_definitions(tables))
    files[DEFINITIONS] = defs

    fold = np.asarray(trace.fold_id)
    y = np.asarray(trace.y)
    g = trace.views[GLASS_ARM]
    proto = dict(g.metadata.get("oof_protocol") or {})
    prov = {p.name: p for p in sources.ebm.provenance}
    by_role = _role_objects(sources)
    manifest: Dict[str, Any] = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "baseline_id": baseline_id,
        "label": BASELINE_LABEL, "designation": DESIGNATION, "status": "frozen",
        "supersedes": None,
        "scope": ("GLASS Router and RF Router, outer cross-fit (training-side, out-of-fold) "
                  "outputs of the historical artifacts; Steps 2A-2D diagnostics against the "
                  "GLASS arm's frozen Stage 1 LR / Stage 3 EBM reference flags. Train OOF only; no "
                  "test label, test prediction or test-derived threshold."),
        "relation_to_pr38": ("PR 38's corrected-selector results are compared with this "
                             "baseline; they never replace it. A future baseline gets a new "
                             "baseline_id and directory."),
        "population": {
            "training_rows": int(trace.n), "positives": int(trace.n_positives),
            "negatives": int(trace.n - trace.n_positives), "outer_folds": int(trace.n_folds),
            "rows_per_fold": [int((fold == k).sum()) for k in range(trace.n_folds)],
            "positives_per_fold": [int(y[fold == k].sum()) for k in range(trace.n_folds)]},
        "identities": {
            "router_split_fingerprint": {"value": trace.router_split_fingerprint,
                                         "status": "validated",
                                         "detail": "16-hex; hashes row indices only (no labels)"},
            "train_label_fingerprint": {"value": trace.train_label_fingerprint,
                                        "status": "validated",
                                        "detail": "training rows and labels; both routers' "
                                                  "cross-fit and full-train records match"},
            "stage_io_split_fingerprint": {
                "value": trace.stage_io_split_fingerprint,
                "status": "attested" if trace.stage_io_split_fingerprint else "not_recorded",
                "attestation": trace.stage_io_attestation,
                "detail": "never recomputed: its hash includes the test labels"},
            "fold_plan": {"n_folds": int(trace.n_folds), "seed": proto.get("random_state"),
                          "fingerprint": proto.get("fold_fingerprint"), "status": "validated",
                          "detail": "shared Stage 2-3 outer plan; both routers, LR grouping, EBM"},
            "ebm_block_fingerprint": (
                {"value": None, "status": "not_recorded"} if "block_fingerprint" not in prov else
                {"value": prov["block_fingerprint"].detail or None,
                 "status": prov["block_fingerprint"].status,
                 "detail": "recorded by the Stage 3 EBM artifact; not re-verifiable here"}),
            "feature_matrix_hash": {"value": None, "status": "not_recorded",
                                    "detail": "no feature-matrix hash is stored in the artifacts"},
            "data_files": [{"path": d.path, "sha256": d.sha256, "role": d.role,
                            "method": "sha256 of the file with CRLF normalised to LF"}
                           for d in sources.data_files],
        },
        "sources": [
            {"role": a.role, "path": a.path, "sha256": a.sha256, "bytes": int(a.size),
             "snapshot_attested": a.snapshot_attested, "read_by": _ROLE_READER[a.role],
             "fields_read": (list(by_role[a.role].source.get("fields_read") or ())
                             if a.role in by_role else None),
             "extractor_checks": (None if a.role not in by_role else {
                 "total": len(by_role[a.role].source.get("checks") or ()),
                 "passed": sum(1 for c in (by_role[a.role].source.get("checks") or ())
                               if c.get("passed"))})}
            for a in sorted(sources.artifacts, key=lambda a: SOURCE_ROLES.index(a.role))],
        "snapshot": (None if sources.snapshot is None else
                     {"label": sources.snapshot.label, "sha256": sources.snapshot.sha256,
                      "entries": int(sources.snapshot.entries)}),
        "code": {"method": code.method, "files": dict(code.files),
                 "note": "identity of the diagnostic and export code at generation; not the "
                         "router implementations (see routers) and not a Git commit"},
        "routers": _router_records(trace),
        "upstream_operating_points": _upstream_points(r, sources),
        "rf_constraint_provenance": _constraint_provenance(r),
        "analysis_settings": _settings(r),
        "validation": {
            "trace_checks": {"total": len(trace.checks),
                             "passed": sum(1 for c in trace.checks if c.get("passed"))},
            "rf_pass1_score_equivalence": str(_row(r["signal"], arm=RF_ARM,
                                                   signal="Pass 1 outer score (all rows)")
                                              ["status"]),
            "consistency": consistency},
        "outputs": {},
        "disclosures": _disclosures(r),
        "diagnostic_notes": _notes(r),
        "historical_test_exposure": HISTORICAL_TEST_EXPOSURE,
        "regeneration": {
            "tool": f"{BASELINES_DIR}/stage2_baseline.py (run from prototype/bank_pipeline/)",
            "verify": "--verify: integrity only; no artifacts; never writes",
            "check": "--check: rebuild from the recorded artifacts and compare; never writes",
            "generate": "--generate: first write only; identical rerun is a no-op; divergence "
                        "is refused"},
    }
    manifest["outputs"] = {DEFINITIONS: {"sha256": _sha(defs), "bytes": len(defs),
                                         "description": "metric registry, table / column "
                                                        "definitions and conventions"}}
    manifest["outputs"].update(outputs)
    report = _report(r, tables, _plain(manifest)).encode("utf-8")
    files[REPORT] = report
    manifest["outputs"][REPORT] = {"sha256": _sha(report), "bytes": len(report),
                                   "description": "human-readable report"}
    man_bytes = _json_bytes(manifest)
    files[MANIFEST] = man_bytes
    files[SIDECAR] = f"{_sha(man_bytes)}  {MANIFEST}\n".encode("ascii")

    for path, data in files.items():
        text = data.decode("utf-8")
        hit = _ABS_PATH.search(text)
        if hit:
            raise BaselineError(f"{path}: machine-specific absolute path {hit.group(0)!r} in the "
                                "baseline content")
        if "\r" in text:
            raise BaselineError(f"{path}: carriage return in the baseline content")
    ordered = {k: files[k] for k in (MANIFEST, SIDECAR, REPORT, DEFINITIONS,
                                     *[TABLES[n][0] for n in TABLES])}
    return BaselinePackage(baseline_id, MappingProxyType(ordered),
                           MappingProxyType(json.loads(man_bytes)),
                           MappingProxyType(dict(tables)))


# ----------------------------------------------------------------------
# Verify / compare / write
# ----------------------------------------------------------------------
def read_manifest(directory) -> Dict[str, Any]:
    p = Path(directory) / MANIFEST
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise BaselineError(f"{p}: missing") from None
    except (UnicodeDecodeError, ValueError) as exc:
        raise BaselineError(f"{p}: not valid UTF-8 JSON ({exc})") from None


def _listing(directory: Path) -> List[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


def _csv_shape(data: bytes) -> Tuple[List[str], int]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    return (rows[0] if rows else []), max(len(rows) - 1, 0)


def verify_baseline(directory, *, expected_id: Optional[str] = None) -> VerificationReport:
    """Integrity of a frozen package without any artifact. Reads only; never writes."""
    d = Path(directory)
    problems: List[str] = []
    if not d.is_dir():
        return VerificationReport(str(d), False, (f"{d}: not a directory",))
    man_path, side_path = d / MANIFEST, d / SIDECAR
    if not man_path.is_file():
        return VerificationReport(str(d), False, (f"{MANIFEST}: missing",))
    man_bytes = man_path.read_bytes()
    man_sha = _sha(man_bytes)
    if not side_path.is_file():
        problems.append(f"{SIDECAR}: missing")
    else:
        side = side_path.read_bytes()
        if side != f"{man_sha}  {MANIFEST}\n".encode("ascii"):
            problems.append(f"{SIDECAR}: does not hold the SHA-256 of {MANIFEST} "
                            f"({man_sha}); the manifest or the sidecar was changed")
    try:
        man = json.loads(man_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return VerificationReport(str(d), False, tuple(problems) + (f"{MANIFEST}: invalid ({exc})",),
                                  man_sha)
    for key, want in (("format", FORMAT), ("format_version", FORMAT_VERSION),
                      ("designation", DESIGNATION), ("status", "frozen")):
        if man.get(key) != want:
            problems.append(f"{MANIFEST}: {key} is {man.get(key)!r}, expected {want!r}")
    bid = man.get("baseline_id")
    if bid != d.name:
        problems.append(f"{MANIFEST}: baseline_id {bid!r} != directory name {d.name!r}")
    if expected_id is not None and bid != expected_id:
        problems.append(f"{MANIFEST}: baseline_id {bid!r}, expected {expected_id!r}")
    outputs = man.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        problems.append(f"{MANIFEST}: no outputs listed")
        outputs = {}
    checked = 0
    for rel, rec in outputs.items():
        p = d / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
            continue
        data = p.read_bytes()
        checked += 1
        if len(data) != rec.get("bytes") or _sha(data) != rec.get("sha256"):
            problems.append(f"{rel}: content differs from MANIFEST.json (sha256 {_sha(data)[:16]}… "
                            f"vs {str(rec.get('sha256'))[:16]}…, {len(data)} vs "
                            f"{rec.get('bytes')} bytes)")
            continue
        if rel.endswith(".csv"):
            header, n = _csv_shape(data)
            if header != rec.get("columns") or n != rec.get("rows"):
                problems.append(f"{rel}: header / row count differ from MANIFEST.json")
    expected = {MANIFEST, SIDECAR, *outputs}
    for rel in _listing(d):
        if rel not in expected:
            problems.append(f"{rel}: not part of the package (unexpected file)")
    return VerificationReport(str(d), not problems, tuple(problems), man_sha, checked)


def _sections(a: Mapping, b: Mapping) -> List[str]:
    keys = list(dict.fromkeys([*a.keys(), *b.keys()]))
    return [k for k in keys if k not in ("code", "outputs") and a.get(k) != b.get(k)]


def _code_changes(frozen: Mapping, rebuilt: Mapping) -> List[str]:
    fa = (frozen.get("code") or {}).get("files") or {}
    fb = (rebuilt.get("code") or {}).get("files") or {}
    out = []
    for rel in dict.fromkeys([*fa, *fb]):
        if fa.get(rel) != fb.get(rel):
            out.append(f"{rel}: {str(fa.get(rel))[:12]} -> {str(fb.get(rel))[:12]}")
    return out


def compare_baseline(package: BaselinePackage, directory) -> ComparisonReport:
    """A rebuilt package against a frozen directory. Reads only; never writes."""
    d = Path(directory)
    integrity = verify_baseline(d, expected_id=package.baseline_id)
    if not d.is_dir() or not (d / MANIFEST).is_file():
        return ComparisonReport("differs", integrity, missing=tuple(package.files))
    present = set(_listing(d))
    missing = tuple(p for p in package.files if p not in present)
    unexpected = tuple(p for p in sorted(present) if p not in package.files)
    differing = tuple(p for p in package.files
                      if p in present and p not in (MANIFEST, SIDECAR)
                      and (d / p).read_bytes() != package.files[p])
    try:
        frozen = read_manifest(d)
    except BaselineError as exc:
        return ComparisonReport("differs", integrity, missing, unexpected, differing,
                                (str(exc),))
    rebuilt = json.loads(package.files[MANIFEST])
    sections = tuple(_sections(frozen, rebuilt))
    if frozen.get("outputs") != rebuilt.get("outputs") and "outputs" not in sections:
        sections += ("outputs",)
    code = tuple(_code_changes(frozen, rebuilt))
    exact = ((d / MANIFEST).read_bytes() == package.files[MANIFEST]
             and (d / SIDECAR).is_file() and (d / SIDECAR).read_bytes() == package.files[SIDECAR])
    if integrity.ok and not (missing or unexpected or differing or sections):
        status = "identical" if exact and not code else "reproduced_with_code_drift"
    else:
        status = "differs"
    return ComparisonReport(status, integrity, missing, unexpected, differing, sections, code)


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def write_baseline(package: BaselinePackage, parent) -> WriteResult:
    """
    Write the package to ``parent/<baseline_id>/`` once. An existing identical
    package: nothing is written. An existing package that differs in any way
    (including a partial one): ``BaselineConflict``, nothing is written. The
    files are written to a temporary sibling directory, verified, then moved
    into place with one rename.
    """
    parent = Path(parent)
    if not parent.is_dir():
        raise BaselineError(f"{parent}: the baselines directory does not exist")
    target = parent / package.baseline_id
    if target.exists():
        cmp = compare_baseline(package, target)
        if cmp.status == "identical":
            return WriteResult("identical", str(target), package.manifest_sha256)
        raise BaselineConflict(
            f"{target} exists and differs from the rebuilt package; nothing was written. A "
            "frozen baseline is never overwritten: create a new version (new baseline_id) or, "
            "if this directory is an unintended partial or local copy, remove it yourself.\n  "
            + "\n  ".join(cmp.lines()), cmp)
    tmp = parent / f".{package.baseline_id}.partial-{uuid.uuid4().hex[:12]}"
    try:
        tmp.mkdir()
        for rel, data in package.files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            _fsync_write(p, data)
        check = verify_baseline(tmp, expected_id=None)
        bad = [x for x in check.problems if "directory name" not in x]
        if bad:
            raise BaselineError("the written package does not verify:\n  " + "\n  ".join(bad))
        os.rename(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    final = verify_baseline(target, expected_id=package.baseline_id)
    if not final.ok:
        raise BaselineError(f"{target}: written but does not verify:\n  "
                            + "\n  ".join(final.problems))
    return WriteResult("written", str(target), package.manifest_sha256)
