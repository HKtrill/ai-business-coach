"""
PR 38 Commit 4: the consolidated research report of a run (``report_v1/``).

A downstream, read-only layer over a run's v1 packages (``complementarity_v1/``
and, when present, ``s1_sensitivity_v1/``). It never reads a model, an
artifact, a label or the train side, and runs no gate. ``report_v1/`` holds::

    README.md                         where to start; headline results, file index
    NN_<subject>.csv                  eleven research tables (one row per source row)
    run_summary.json                  identities, acceptance, provenance records
    complementarity_summary.json
    s1_summary.json
    source_manifest_<package>.json    byte-exact copy of each v1 package MANIFEST.json
    MANIFEST.json (+ .sha256)         historical v1 checksums, dispositions, outputs, code

Every v1 package table has exactly one disposition (``GROUPS``, ``EXCLUDED``):
kept as rows of a report table (``package`` and ``table`` name each row's
source; renames are declared), or *derived* (an outer join of kept tables and
the frozen pre-correction baseline), a *duplicate* of a kept table, or
*restated* from a record kept beside it (``s1_acceptance.json``).

Two modes, one rendering:

migration (``run``, ``build``)
    Reads the v1 packages (their run folder, or ``packages_dir``). Before
    anything is written, every package table is reproduced byte for byte from
    the report's own rows or by its rebuild; a table without a disposition, a
    package that fails its manifest or does not belong to the run's accepted
    record, or a reproduction that differs stops it. Writes ``report_v1/``
    once. Never deletes a source.
verification (``run(check=True)``, ``verify``)
    Reads no v1 package. Checks the report's own MANIFEST, every file's
    SHA-256 and table schema, the run's accepted records, the manifest copies;
    reproduces every v1 table from the report and compares it with its
    historical SHA-256; renders the report again and compares byte for byte.

After a migration the v1 package folders are optional: ``report --check`` does
not need them. The packages' own ``--check`` (``complementarity``,
``s1-report``) needs their tables and is runnable only while they exist.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from shared.stage2.diagnostics import baseline as b

from . import acceptance as A
from . import complementarity as C
from . import phase0
from . import runner as R
from . import sensitivity as S

__all__ = ["FORMAT", "REPORT_DIR", "GROUPS", "EXCLUDED", "ReportError", "build", "verify",
           "reconstruct", "run"]

FORMAT = "pr38.stage2.research_report/1"
REPORT_DIR = "report_v1"
CP, SP = C.PACKAGE_DIR, S.PACKAGE_DIR
PACKAGE_FORMATS = {CP: C.FORMAT, SP: S.FORMAT}
PROVENANCE = ("package", "table")
README, MANIFEST, SIDECAR = "README.md", C.MANIFEST, C.SIDECAR
SUMMARIES = ("run_summary.json", "complementarity_summary.json", "s1_summary.json")
CODE_FILES = ("stage2_crossfit/reporting.py", "stage2_crossfit/complementarity.py",
              "stage2_crossfit/sensitivity.py", "shared/stage2/diagnostics/baseline.py")
#: Records kept beside the packages (re-verified by path and SHA-256 by the
#: acceptance and S1 gates); the report lists them, it never rewrites them.
RECORD_PATTERNS = ("phase0.json", "configs.json", "g3_preflight_fold*.json", "g13_fold*.json",
                   "authorization*.json", "pytest_fast.xml", "primary_attempt.json",
                   "primary_MANIFEST.json", "g7_reproduction_fold*.json",
                   "acceptance_dispositions.json", "structural_acceptance.json",
                   "s1_attempt.json", "s1_MANIFEST.json", "s1_acceptance.json")
_NOT_RECORDS = re.compile(r"(_failed_|_stopped|_population_failure)")
_ATTRIBUTION = ("full - oof_remainder: the population policy (within S1); "
                "either - baseline: confounded (selector correction and contract)")
_HISTORICAL_NOTE = (
    "SHA-256 of each v1 package table as written by its generator (historical). The tables "
    "need not exist after the migration: report --check reproduces every one from this report "
    "(and the frozen baseline) and compares it with these checksums. The packages' own --check "
    "(complementarity, s1-report) needs their tables and runs only while they exist.")


def source_manifest(package: str) -> str:
    return f"source_manifest_{package}.json"


class ReportError(R.RunnerError):
    """The report cannot be built or verified; nothing was written."""


@dataclass(frozen=True)
class Group:
    file: str
    title: str
    sources: Tuple[Tuple[str, str], ...]        # (package, table), in row order
    keys: Tuple[str, ...] = ()                  # identifying columns, placed first
    renames: Mapping[str, Mapping[str, str]] = field(default_factory=dict)   # by table


def _both(*tables: str) -> Tuple[Tuple[str, str], ...]:
    return tuple((p, t) for t in tables for p in (CP, SP))


GROUPS: Tuple[Group, ...] = (
    Group("01_registered_outcomes.csv",
          "Registered outcomes with 95% paired bootstrap intervals, pooled and per fold: GLASS "
          "vs RF (D-C1 statements) and pre vs post correction per arm",
          ((CP, "primary_outcomes"), (CP, "primary_outcomes_by_fold"),
           (CP, "pre_post_uncertainty"), (CP, "pre_post_by_fold")),
          keys=("outcome", "arm", "metric", "kind", "fold")),
    Group("02_router_metrics.csv",
          "Router metric tables (numerator, denominator, value, status) per fold and pooled, "
          "corrected and pre/post correction, and their fold dispersion",
          ((CP, "metric_table"), (CP, "pre_post_metrics"), (CP, "fold_dispersion")),
          keys=("arm", "correction", "fold", "metric"),
          renames={"metric_table": {"routing": "arm"}, "fold_dispersion": {"routing": "arm"},
                   "pre_post_metrics": {"routing": "correction"}}),
    Group("03_routing_states.csv",
          "Three-state routing: counts, shares and class composition, and state transitions "
          "(pre to post correction; S1 oof_remainder to full)",
          _both("diagnostics/states") + ((CP, "class_conditional"),
                                         (CP, "pre_post_transitions"),
                                         (SP, "routing_transitions")),
          keys=("arm", "fold", "state", "pre_state", "post_state", "state_oof_remainder",
                "state_full"),
          renames={"class_conditional": {"routing": "arm"}}),
    Group("04_unique_contribution.csv",
          "Pass 2 discovery against the frozen LR/EBM references, cross-router overlap, the "
          "common eligible population and chance baselines",
          _both("diagnostics/pass2_breakdown", "diagnostics/pass2_cross_router",
                "diagnostics/common_population", "diagnostics/chance_baselines"),
          keys=("setting", "arm", "reference", "fold")),
    Group("05_matched_budget.csv",
          "The frozen LR/EBM reference flags, label-free rate-matched selections and each "
          "router's Pass 1 set against the matched LR set",
          ((CP, "diagnostics/reference_flags"),) + _both("diagnostics/matched_selection")
          + ((CP, "pass1_matched_lr"),),
          keys=("reference", "arm", "model", "score_space", "set", "fold")),
    Group("06_agreement_diagnostics.csv",
          "Router-reference contingency, cross-router crosstab, LR/EBM disagreement and "
          "observed-versus-expected",
          _both("diagnostics/contingency", "diagnostics/crosstab",
                "diagnostics/model_disagreement", "diagnostics/observed_expected"),
          keys=("setting", "arm", "reference", "group", "description", "fold", "k", "state",
                "reference_flag", "glass_state", "rf_state")),
    Group("07_score_strata.csv",
          "Routing by score stratum, and stratum contrasts",
          _both("diagnostics/score_conditional", "diagnostics/stratum_contrasts"),
          keys=("arm", "stratifier", "fold", "stratum", "state", "contrast")),
    Group("08_diagnostic_ratios.csv",
          "Every Step 2A-2D count and ratio (numerator, denominator, value, status)",
          _both("diagnostics/ratios"), keys=b._RATIO_KEY),
    Group("09_contract_compliance.csv",
          "Fit-time obligations and observed outer-OOF shortfalls (kept apart), F31 R2 strips, "
          "R3 optimism and M6 inner-OOF Pass 1",
          ((CP, "shortfalls"), (CP, "fit_time_obligations"), (CP, "f31_r2_strips"),
           (CP, "f31_r3_optimism"), (CP, "m6_inner_oof_pass1")),
          keys=("arm", "fold", "pool", "selection", "obligation", "metric")),
    Group("10_solver_and_validation.csv",
          "Fold ledger, solver outcomes, three-state accounting, trace and alignment checks, "
          "routing-policy audit and signal availability",
          ((CP, "fold_ledger"), (CP, "fold_ledger_summary"), (CP, "solver_outcomes"),
           (CP, "accounting"), (CP, "trace_checks"), (CP, "alignment"))
          + _both("diagnostics/policy_audit", "diagnostics/signal_availability"),
          keys=("arm", "router", "routing", "fold", "selection", "section", "stage", "check",
                "claim", "signal")),
    Group("11_s1_sensitivity.csv",
          "S1 (GLASS pass2_population full vs oof_remainder): configuration, Pass 2 fitting "
          "populations, fold feasibility and G6 Pass 1 invariance",
          ((SP, "configuration"), (SP, "pass2_fitting_population"), (SP, "fold_feasibility"),
           (SP, "g6_pass1_invariance")),
          keys=("config", "setting", "fold")),
)

#: (package, table) -> (disposition, what it is rebuilt from)
EXCLUDED: Mapping[Tuple[str, str], Tuple[str, str]] = {
    (CP, "pre_post_states"): (
        "derived", f"outer join of the frozen {b.BASELINE_ID} states with "
        f"{CP}/diagnostics/states (03_routing_states.csv); pre - post"),
    (CP, "pre_post_ratios"): (
        "derived", f"outer join of the frozen {b.BASELINE_ID} ratios with "
        f"{CP}/diagnostics/ratios (08_diagnostic_ratios.csv); pre - post"),
    (SP, "states_comparison"): (
        "derived", "outer join of diagnostics/states of both packages (03_routing_states.csv); "
        "full - oof_remainder"),
    (SP, "ratios_comparison"): (
        "derived", "outer join of diagnostics/ratios of both packages "
        "(08_diagnostic_ratios.csv); full - oof_remainder"),
    (SP, "baseline_comparison"): (
        "derived", f"outer join of the GLASS rows of the frozen {b.BASELINE_ID} ratios and of "
        "both packages' diagnostics/ratios (08_diagnostic_ratios.csv)"),
    (SP, "diagnostics/reference_flags"): (
        "duplicate", f"{CP}/diagnostics/reference_flags (05_matched_budget.csv)"),
    (SP, "s1_acceptance_checks"): (
        "restated", f"{S.S1_ACCEPTANCE_FILE} (every gate's checks)"),
}


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
@dataclass
class Package:
    name: str
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    raw: Dict[str, bytes]                       # table -> CSV bytes (as written in v1)
    columns: Mapping[str, Tuple[str, ...]]      # table -> v1 manifest column order

    @property
    def manifest_sha256(self) -> str:
        return _sha(self.manifest_bytes)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(data: bytes) -> pd.DataFrame:
    return C._read_text_csv(data)


def _table_name(rel: str) -> str:
    if not (rel.startswith("tables/") and rel.endswith(".csv")):
        raise ReportError(f"package output {rel!r} is not a tables/*.csv table")
    return rel[len("tables/"):-len(".csv")]


def _manifest(name: str, data: bytes) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, ...]]]:
    man = json.loads(data.decode("utf-8"))
    if man.get("format") != PACKAGE_FORMATS[name]:
        raise ReportError(f"{name}: format {man.get('format')!r}, not {PACKAGE_FORMATS[name]!r}")
    cols = {_table_name(rel): tuple(o.get("columns") or ())
            for rel, o in (man.get("outputs") or {}).items()}
    return man, cols


def _load_package(directory: Path, name: str) -> Package:
    """A v1 package exactly as its manifest pins it, every table reproducible as stored text."""
    man_path, side_path = directory / MANIFEST, directory / SIDECAR
    if not man_path.is_file() or not side_path.is_file():
        raise ReportError(f"{directory}: no {MANIFEST} with its {SIDECAR}")
    man_b = man_path.read_bytes()
    side = side_path.read_text(encoding="ascii").split()
    if not side or side[0].lower() != _sha(man_b):
        raise ReportError(f"{name}: {SIDECAR} is not the SHA-256 of {MANIFEST}")
    man, columns = _manifest(name, man_b)
    outputs = man.get("outputs") or {}
    present = set(C._listing(directory)) - {MANIFEST, SIDECAR}
    problems = [f"missing {p}" for p in sorted(set(outputs) - present)]
    problems += [f"not in the manifest: {p}" for p in sorted(present - set(outputs))]
    raw = {}
    for rel in sorted(set(outputs) & present):
        data = (directory / rel).read_bytes()
        if _sha(data) != outputs[rel].get("sha256"):
            problems.append(f"{rel}: SHA-256 differs from the manifest")
            continue
        frame = _text(data)
        if b._csv_bytes(frame) != data:
            problems.append(f"{rel}: not reproducible from its stored text")
        if columns[_table_name(rel)] != tuple(frame.columns):
            problems.append(f"{rel}: header differs from the manifest's columns")
        raw[_table_name(rel)] = data
    if problems:
        raise ReportError(f"{name} fails its manifest:\n  - " + "\n  - ".join(problems))
    return Package(name, man, man_b, raw, columns)


def _ident(man: Mapping[str, Any], *path: str) -> Any:
    v: Any = man.get("identities") or {}
    for p in path:
        v = v.get(p) if isinstance(v, Mapping) else None
    return v


def _check_run(packages: Mapping[str, Package], record_dir: Path, run_id: str) -> Dict[str, str]:
    """Each package names this run and the accepted records beside it (hashes only; no gate)."""
    shas = {}
    for f in (A.ACCEPTANCE_FILE, S.S1_ACCEPTANCE_FILE):
        p = record_dir / f
        shas[f] = R.sha256_file(p) if p.is_file() else None
    if shas[A.ACCEPTANCE_FILE] is None:
        raise ReportError(f"run {run_id!r} has no {A.ACCEPTANCE_FILE}; a report is made only "
                          "beside the accepted record its packages name")
    cm = packages[CP].manifest
    want = {(CP, "run_id"): (_ident(cm, "run", "run_id"), run_id),
            (CP, A.ACCEPTANCE_FILE): (_ident(cm, "structural_acceptance", "sha256"),
                                      shas[A.ACCEPTANCE_FILE])}
    if SP in packages:
        sm = packages[SP].manifest
        want.update({(SP, "run_id"): (_ident(sm, "run", "run_id"), run_id),
                     (SP, A.ACCEPTANCE_FILE): (_ident(sm, "primary",
                                                      "structural_acceptance_sha256"),
                                               shas[A.ACCEPTANCE_FILE]),
                     (SP, S.S1_ACCEPTANCE_FILE): (_ident(sm, "s1", "acceptance_sha256"),
                                                  shas[S.S1_ACCEPTANCE_FILE])})
    bad = [f"{pkg}: {what} is {got!r} in its manifest, {have!r} here"
           for (pkg, what), (got, have) in want.items() if got is None or got != have]
    if bad:
        raise ReportError("the packages are not this run's accepted ones:\n  - "
                          + "\n  - ".join(bad))
    return shas


def _baseline(directory: Path, manifest_sha256: Optional[str]) -> Dict[str, Any]:
    """The frozen pre-correction tables the derived joins read: the package the analysis pinned."""
    man_path = directory / MANIFEST
    if not man_path.is_file():
        raise ReportError(f"no frozen baseline package at {directory}")
    man_b = man_path.read_bytes()
    side = (directory / SIDECAR).read_text(encoding="ascii").split() \
        if (directory / SIDECAR).is_file() else []
    if not manifest_sha256 or _sha(man_b) != manifest_sha256 or not side \
            or side[0].lower() != _sha(man_b):
        raise ReportError(f"the baseline at {directory} is not the package {CP} pinned "
                          f"({manifest_sha256})")
    outputs = json.loads(man_b.decode("utf-8")).get("outputs") or {}
    out = {"manifest_sha256": _sha(man_b), "tables": {}, "sha256": {}}
    for name in ("states", "ratios"):
        rel = f"tables/{name}.csv"
        data = (directory / rel).read_bytes() if (directory / rel).is_file() else b""
        if _sha(data) != (outputs.get(rel) or {}).get("sha256"):
            raise ReportError(f"baseline {rel}: SHA-256 differs from its manifest")
        out["tables"][name], out["sha256"][name] = _text(data), _sha(data)
    return out


def _records(record_dir: Path) -> Dict[str, Dict[str, Any]]:
    found = sorted({p for pat in RECORD_PATTERNS for p in record_dir.glob(pat)
                    if p.is_file() and not _NOT_RECORDS.search(p.name)})
    return {p.name: {"sha256": R.sha256_file(p), "bytes": p.stat().st_size} for p in found}


def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError as exc:
        raise ReportError(f"{path.name}: not JSON ({exc})") from None


# ----------------------------------------------------------------------
# Consolidation and reconstruction
# ----------------------------------------------------------------------
def _coverage(packages: Mapping[str, Package]) -> Dict[Tuple[str, str], str]:
    """Every package table -> its report table or exclusion; nothing twice, nothing left over."""
    where: Dict[Tuple[str, str], str] = {}
    for g in GROUPS:
        for src in g.sources:
            if src in where:
                raise ReportError(f"internal: {src} is in two report tables")
            where[src] = g.file
    for src, (kind, _) in EXCLUDED.items():
        if src in where:
            raise ReportError(f"internal: {src} is both kept and excluded")
        where[src] = kind
    problems = []
    for name, pkg in packages.items():
        have, spec = set(pkg.columns), {t for p, t in where if p == name}
        problems += [f"{name}/{t}: no disposition in the report" for t in sorted(have - spec)]
        problems += [f"{name}/{t}: expected by the report, not in the package"
                     for t in sorted(spec - have)]
    if problems:
        raise ReportError("the report does not cover the packages exactly:\n  - "
                          + "\n  - ".join(problems))
    return {src: w for src, w in where.items() if src[0] in packages}


def _layout(g: Group, columns: Mapping[Tuple[str, str], Sequence[str]]
            ) -> Tuple[List[str], Dict[str, List[str]]]:
    """The report columns (provenance, keys, then each table's own columns, adjacent)."""
    renamed = {}
    for p, t in g.sources:
        if (p, t) not in columns:
            continue
        ren = dict(g.renames.get(t, {}))
        cols = list(columns[(p, t)])
        if any(o not in cols or n in cols for o, n in ren.items()):
            raise ReportError(f"{p}/{t}: rename {ren} does not fit its columns {cols}")
        renamed[f"{p}/{t}"] = [ren.get(c, c) for c in cols]
        if set(PROVENANCE) & set(renamed[f"{p}/{t}"]):
            raise ReportError(f"{p}/{t}: a column is named like {PROVENANCE}")
    order = list(PROVENANCE) + [k for k in g.keys if any(k in c for c in renamed.values())]
    for cols in renamed.values():
        order += [c for c in cols if c not in order]
    return order, renamed


def _render_table(g: Group, packages: Mapping[str, Package]) -> bytes:
    columns = {(p, t): packages[p].columns[t] for p, t in g.sources if p in packages}
    order, _ = _layout(g, columns)
    parts = []
    for p, t in g.sources:
        if p in packages:
            f = _text(packages[p].raw[t]).rename(columns=dict(g.renames.get(t, {})))
            parts.append(f.assign(package=p, table=t).reindex(columns=order))
    return b._csv_bytes(pd.concat(parts, ignore_index=True).fillna(""))


def reconstruct(report_files: Mapping[str, bytes],
                columns: Mapping[Tuple[str, str], Sequence[str]]) -> Dict[Tuple[str, str], bytes]:
    """
    Every kept package table, from the report's CSV bytes alone: the rows of
    (``package``, ``table``) in order, its columns in ``columns`` order with
    the declared renames undone. The bytes equal the table as written in v1.
    """
    out = {}
    for g in GROUPS:
        if g.file not in report_files:
            continue
        rep = _text(report_files[g.file])
        for p, t in g.sources:
            if (p, t) not in columns:
                continue
            back = {n: o for o, n in (g.renames.get(t) or {}).items()}
            part = rep[(rep["package"] == p) & (rep["table"] == t)]
            part = part.drop(columns=[o for o in back.values() if o in part.columns])
            frame = part.rename(columns=back)[list(columns[(p, t)])]
            out[(p, t)] = b._csv_bytes(frame.reset_index(drop=True))
    return out


def _rebuilt(tab, packages: Mapping[str, Package], baseline: Mapping[str, Any],
             record_dir: Path) -> Dict[Tuple[str, str], bytes]:
    """Each excluded table, rebuilt by the generators' own joins from tables the report keeps."""
    def csv(frame):
        return b._csv_bytes(C._flat(frame))
    frozen = baseline["tables"]
    out = {(CP, "pre_post_states"): csv(C._pre_post_join(
               frozen["states"], tab(CP, "diagnostics/states"), ("arm", "fold", "state"),
               ("rows", "n", "positives", "negatives", "share", "positive_rate"),
               ("n", "positives", "share", "positive_rate"))),
           (CP, "pre_post_ratios"): csv(C._pre_post_join(
               frozen["ratios"], tab(CP, "diagnostics/ratios"), b._RATIO_KEY,
               ("label", "numerator", "denominator", "value", "status"),
               ("numerator", "value")))}
    if SP in packages:
        c0, c1 = S.CONFIGS
        states = {c0: tab(CP, "diagnostics/states"), c1: tab(SP, "diagnostics/states")}
        ratios = {c0: tab(CP, "diagnostics/ratios"), c1: tab(SP, "diagnostics/ratios")}
        out[(SP, "states_comparison")] = csv(S._join(
            states, ("arm", "fold", "state"), ("n", "positives", "negatives", "share",
                                               "positive_rate"),
            [(col, c1, c0) for col in ("n", "positives", "share", "positive_rate")]))
        out[(SP, "ratios_comparison")] = csv(S._join(
            ratios, b._RATIO_KEY, ("label", "numerator", "denominator", "value", "status"),
            [(col, c1, c0) for col in ("numerator", "denominator", "value")]))
        glass = {"baseline": frozen["ratios"], **ratios}
        base = S._join({k: f[f["arm"] == "glass"] for k, f in glass.items()}, b._RATIO_KEY,
                       ("numerator", "denominator", "value"),
                       [("value", c1, c0), ("value", c0, "baseline"), ("value", c1, "baseline")])
        base["attribution"] = _ATTRIBUTION
        out[(SP, "baseline_comparison")] = csv(base)
        acc = _json(record_dir / S.S1_ACCEPTANCE_FILE)
        checks = pd.DataFrame([{"group": name, **c} for name, gate in (acc.get("gates") or {})
                               .items() for c in gate.get("checks") or []],
                              columns=["group", "check", "passed", "detail"])
        out[(SP, "s1_acceptance_checks")] = csv(checks)
        out[(SP, "diagnostics/reference_flags")] = b._csv_bytes(
            tab(CP, "diagnostics/reference_flags"))
    return out


# ----------------------------------------------------------------------
# Summaries and README
# ----------------------------------------------------------------------
_INT, _FLOAT = re.compile(r"^-?\d+$"), re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _typed(v: str) -> Any:
    """A stored CSV cell as a JSON value: the same number or flag, '' as null."""
    if v == "":
        return None
    if v in ("true", "false"):
        return v == "true"
    if _INT.match(v):
        return int(v)
    if _FLOAT.match(v):
        return float(v)
    return v


def _rows(data: bytes, where: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    f = _text(data)
    for k, v in (where or {}).items():
        f = f[f[k] == v]
    return [{k: _typed(v) for k, v in r.items()} for r in f.to_dict("records")]


def _layers_with_files(layers: Mapping[str, Any], where) -> Dict[str, Any]:
    out = {}
    for name, body in (layers or {}).items():
        body = dict(body or {})
        tabs = body.pop("tables", None)
        if tabs is not None:
            body["tables"] = {t: where.get((CP, t)) for t in tabs}
        out[name] = body
    return out


def _summaries(run_id: str, packages, where, record_dir: Path, records,
               shas) -> Dict[str, bytes]:
    cm = packages[CP].manifest
    acc = _json(record_dir / A.ACCEPTANCE_FILE)
    auth_name = next((n for n in records if n.startswith("authorization")
                      and n != "authorization_review.json"), None)
    auth = _json(record_dir / auth_name) if auth_name else {}
    g7_name = next((n for n in records if n.startswith("g7_reproduction_fold")), None)
    g7 = _json(record_dir / g7_name) if g7_name else {}
    disp = _json(record_dir / A.DISPOSITIONS_FILE)
    s1 = _json(record_dir / S.S1_ACCEPTANCE_FILE)
    run = {
        "format": f"{FORMAT}#run_summary", "run_id": run_id,
        "identities": cm.get("identities"),
        "acceptance": {
            "structural_acceptance": {"file": A.ACCEPTANCE_FILE,
                                      "sha256": shas[A.ACCEPTANCE_FILE],
                                      "accepted": acc.get("accepted"),
                                      "verdict": acc.get("structural_acceptance"),
                                      "record_digest": acc.get("record_digest"),
                                      "feature_pathway": acc.get("feature_pathway"),
                                      "fit_status_by_fold": acc.get("fit_status_by_fold")},
            "g7": None if not g7_name else {"file": g7_name, "passed": g7.get("passed"),
                                             "fold": g7.get("fold")},
            "dispositions": disp.get("dispositions"),
            "authorization": None if not auth_name else {
                "file": auth_name, "gates": auth.get("gates"),
                "approved_on": auth.get("approved_on"), "g13_decision": auth.get("g13_decision"),
                "g13_record_run_id": auth.get("g13_record_run_id")},
            "s1": None if not s1 else {"file": S.S1_ACCEPTANCE_FILE,
                                       "sha256": shas[S.S1_ACCEPTANCE_FILE],
                                       "accepted": s1.get("accepted"),
                                       "g6_passed": s1.get("g6_passed"),
                                       "statement": s1.get("statement")}},
        "records": records,
        "historical_v1_packages": {
            n: {"format": p.manifest.get("format"), "manifest_sha256": p.manifest_sha256,
                "manifest_copy": source_manifest(n), "scope": p.manifest.get("scope"),
                "tables": len(p.columns)} for n, p in packages.items()},
    }
    comp = {
        "format": f"{FORMAT}#complementarity_summary", "run_id": run_id,
        "scope": cm.get("scope"),
        "primary_outcomes": _rows(packages[CP].raw["primary_outcomes"]),
        "pre_post_uncertainty": _rows(packages[CP].raw["pre_post_uncertainty"]),
        "layers": _layers_with_files(cm.get("layers"), where),
        "comparative_sections": {s: {_table_name(p): where.get((CP, _table_name(p)))
                                     for p in paths}
                                 for s, paths in (cm.get("comparative_sections") or {}).items()},
        "comparative_section_gaps": cm.get("comparative_section_gaps"),
        "not_recomputed": cm.get("not_recomputed"),
        "bootstrap": {k: {"digest": (v or {}).get("digest"), "verified": (v or {}).get("verified")}
                      for k, v in (cm.get("bootstrap") or {}).items()},
        "limitations": cm.get("disclosures"),
        "source": {"package": CP, "manifest_sha256": packages[CP].manifest_sha256},
    }
    files = {"run_summary.json": b._json_bytes(run),
             "complementarity_summary.json": b._json_bytes(comp)}
    if SP in packages:
        sp = packages[SP]
        sm = sp.manifest
        files["s1_summary.json"] = b._json_bytes({
            "format": f"{FORMAT}#s1_summary", "run_id": run_id, "scope": sm.get("scope"),
            "acceptance": {"record": run["acceptance"]["s1"], "package": sm.get("s1_acceptance")},
            "configuration_differences": _rows(sp.raw["configuration"], {"differs": "true"}),
            "g6_pass1_invariance": _rows(sp.raw["g6_pass1_invariance"]),
            "glass_states_pooled": _rows(sp.raw["states_comparison"],
                                         {"arm": "glass", "fold": "all"}),
            "glass_routing_rates_pooled": _rows(sp.raw["ratios_comparison"],
                                                {"section": "2A.routing_rates", "arm": "glass",
                                                 "fold": "all"}),
            "routing_transitions_pooled": _rows(sp.raw["routing_transitions"], {"fold": "all"}),
            "limitations": sm.get("disclosures"),
            "source": {"package": SP, "manifest_sha256": sp.manifest_sha256},
        })
    return files


def _fmt(v: str) -> str:
    return v if _INT.match(v) or not _FLOAT.match(v) else format(float(v), ".4g")


def _md(v: Any) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _readme(run_id: str, packages, files, shas, record_dir) -> bytes:
    acc, s1 = _json(record_dir / A.ACCEPTANCE_FILE), _json(record_dir / S.S1_ACCEPTANCE_FILE)
    verdict = acc.get("structural_acceptance") or {}
    L = [f"# PR 38 research report: `{run_id}`", "",
         "Consolidated from this run's accepted v1 packages by "
         f"`python -m stage2_crossfit report --run-id {run_id}`. Every value is copied from the "
         "packages as written; no model, artifact or label is read. Numbers below are rounded "
         "for display; the CSV and JSON files hold the exact values.", "",
         "## Status", "",
         f"- Structural acceptance V1-V8: accepted `{json.dumps(acc.get('accepted'))}`, "
         f"official `{json.dumps(verdict.get('official'))}` (`{A.ACCEPTANCE_FILE}`, SHA-256 "
         f"`{shas[A.ACCEPTANCE_FILE][:16]}`)"]
    if SP in packages:
        L.append(f"- S1 sensitivity: accepted `{json.dumps(s1.get('accepted'))}`, G6 Pass 1 "
                 f"invariance `{json.dumps(s1.get('g6_passed'))}`: "
                 f"{_md(s1.get('statement', ''))}")
    L += ["", "## Primary outcomes (GLASS - RF, pooled; 95% paired bootstrap; D-C1)", "",
          "| outcome | metric | GLASS | RF | GLASS - RF | 95% interval | statement |",
          "|---|---|---:|---:|---:|---|---|"]
    for r in _text(packages[CP].raw["primary_outcomes"]).to_dict("records"):
        lo, hi = r.get("difference_low", ""), r.get("difference_high", "")
        ci = f"[{_fmt(lo)}, {_fmt(hi)}]" if lo != "" and hi != "" else "undefined"
        L.append(f"| {_md(r.get('outcome', ''))} | {_md(r.get('metric', ''))} | "
                 f"{_fmt(r.get('glass', ''))} | {_fmt(r.get('blackbox', ''))} | "
                 f"{_fmt(r.get('difference', ''))} | {ci} | {_md(r.get('statement', ''))} |")
    L += ["", "## Tables", "",
          "Each row is one row of a v1 package table: `package` and `table` name it. The "
          "identifying columns come first, then each source table's own columns side by side; "
          "a cell under another table's column is empty, not a value. Filter on `table` to see "
          "one source table whole. Renamed columns: "
          + "; ".join(f"`{t}.{o}` is `{n}`" for g in GROUPS for t, r in g.renames.items()
                      for o, n in r.items()) + ".",
          f"Rows from `{CP}` are the corrected primary (GLASS Pass 2 fitted on the OOF "
          f"remainder); diagnostics rows from `{SP}` are the S1 configuration (GLASS Pass 2 "
          "fitted on every fold row; RF unchanged).", "",
          "| file | contents | rows | columns | v1 tables |", "|---|---|---:|---:|---|"]
    for g in GROUPS:
        if g.file in files:
            f = _text(files[g.file])
            srcs = ", ".join(t + ("" if p == CP else " (S1)") for p, t in g.sources
                             if p in packages)
            L.append(f"| [{g.file}]({g.file}) | {_md(g.title)} | {len(f)} | "
                     f"{len(f.columns)} | {srcs} |")
    L += ["", "| file | contents |", "|---|---|"]
    for name, text in (("run_summary.json", "run identity, acceptance, provenance records"),
                       ("complementarity_summary.json",
                        "primary outcomes, pre/post intervals, layers, limitations"),
                       ("s1_summary.json", "S1 acceptance, single factor, pooled GLASS changes")):
        if name in files:
            L.append(f"| [{name}]({name}) | {text} |")
    for n in packages:
        L.append(f"| [{source_manifest(n)}]({source_manifest(n)}) | byte-exact copy of the "
                 f"historical `{n}/MANIFEST.json` |")
    L.append(f"| [{MANIFEST}]({MANIFEST}) | historical v1 table checksums and dispositions; "
             "this report's file checksums |")
    L += ["", "## v1 package tables not repeated here", "",
          "Rebuilt byte for byte whenever the report is made or checked "
          "(`stage2_crossfit.reporting`).", "", "| v1 table | disposition | from |",
          "|---|---|---|"]
    for (p, t), (kind, note) in EXCLUDED.items():
        if p in packages:
            L.append(f"| {p}/{t} | {kind} | {_md(note)} |")
    L += ["", "## Limitations (as disclosed by the packages)", ""]
    for p in packages.values():
        L += [f"- {_md(x)}" for x in (p.manifest.get("disclosures") or [])]
    L += ["", "## Verification", "",
          f"`{MANIFEST}` keeps two kinds of checksum apart: the historical SHA-256 of every v1 "
          "package table (`historical_v1_sources`) and the SHA-256 of every file here "
          "(`outputs`). From `prototype/bank_pipeline`:", "",
          f"    python -m stage2_crossfit report --run-id {run_id} --check", "",
          "reads no v1 package: it checks every file here against the manifest, the run's "
          "accepted records against the ones listed, reproduces every v1 table from these "
          "tables (and the frozen baseline) against its historical SHA-256, and rebuilds the "
          "report byte for byte. The v1 packages' own `--check` needs their tables and runs "
          "only while they exist.", ""]
    return "\n".join(L).encode("utf-8")


# ----------------------------------------------------------------------
# Rendering, migration, verification
# ----------------------------------------------------------------------
def _render(run_id: str, packages: Mapping[str, Package], record_dir: Path,
            baseline: Mapping[str, Any], bank_root=None) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    """Every file of ``report_v1/``, after every package table is reproduced from it."""
    shas = _check_run(packages, record_dir, run_id)
    where = _coverage(packages)
    files: Dict[str, bytes] = {g.file: _render_table(g, packages) for g in GROUPS
                               if any(p in packages for p, _ in g.sources)}
    columns = {(p, t): pkg.columns[t] for p, pkg in packages.items() for t in pkg.columns}
    kept = reconstruct(files, columns)
    problems = [f"{where[k]} does not reproduce {k[0]}/{k[1]}" for k, data in kept.items()
                if data != packages[k[0]].raw.get(k[1])]
    rebuilt = _rebuilt(lambda p, t: _text(kept[(p, t)]), packages, baseline, record_dir)
    problems += [f"{k[0]}/{k[1]} ({EXCLUDED[k][0]}) does not rebuild from {EXCLUDED[k][1]}"
                 for k, data in rebuilt.items() if data != packages[k[0]].raw.get(k[1])]
    problems += [f"{p}/{t}: no reproduction" for p, pkg in packages.items() for t in pkg.columns
                 if (p, t) not in kept and (p, t) not in rebuilt]
    if problems:
        raise ReportError("v1 package tables are not reproduced by the report:\n  - "
                          + "\n  - ".join(problems))

    records = _records(record_dir)
    files.update(_summaries(run_id, packages, where, record_dir, records, shas))
    files.update({source_manifest(n): p.manifest_bytes for n, p in packages.items()})
    files[README] = _readme(run_id, packages, files, shas, record_dir)
    code = b.code_identity(bank_root or phase0.BANK_ROOT, CODE_FILES)
    outputs = {}
    for name in sorted(files):
        entry: Dict[str, Any] = {"sha256": _sha(files[name]), "bytes": len(files[name])}
        g = next((g for g in GROUPS if g.file == name), None)
        if g is not None:
            f = _text(files[name])
            order, renamed = _layout(g, columns)
            entry.update(rows=len(f), columns=list(f.columns), title=g.title, sources=renamed)
        outputs[name] = entry
    stats = {"package_tables": len(columns), "consolidated": len(kept),
             **{k: sum(1 for x in rebuilt if EXCLUDED[x][0] == k)
                for k in ("derived", "duplicate", "restated")},
             "report_tables": sum(1 for g in GROUPS if g.file in files)}
    manifest = {
        "format": FORMAT, "report": REPORT_DIR, "run_id": run_id,
        "scope": "PR 38 Commit 4: the consolidated research report of an accepted run, migrated "
                 "read-only from its v1 packages; no model, artifact or label is read",
        "historical_v1_sources": {
            "note": _HISTORICAL_NOTE,
            **{n: {"format": p.manifest.get("format"), "manifest_sha256": p.manifest_sha256,
                   "manifest_copy": source_manifest(n),
                   "tables": {t: {"sha256": _sha(p.raw[t]),
                                  "disposition": where[(n, t)],
                                  **({"from": EXCLUDED[(n, t)][1]}
                                     if (n, t) in EXCLUDED else {})}
                              for t in sorted(p.columns)}}
               for n, p in packages.items()}},
        "baseline": {"id": b.BASELINE_ID, "manifest_sha256": baseline["manifest_sha256"],
                     "tables": baseline["sha256"]},
        "records": records,
        "verification": {**stats, "method": "every v1 package table reproduced byte for byte "
                         "from the report (kept tables) or by its rebuild (excluded tables)"},
        "outputs": outputs,
        "code": {"method": code.method, "files": dict(code.files)},
    }
    man = b._json_bytes(manifest)
    files[MANIFEST] = man
    files[SIDECAR] = (_sha(man) + "\n").encode("ascii")
    return files, stats


def _baseline_for(packages, baseline_dir) -> Dict[str, Any]:
    return _baseline(Path(baseline_dir or phase0.BASELINE_DIR),
                     _ident(packages[CP].manifest, "baseline", "manifest_sha256"))


def build(run_id: str, *, record_dir=None, packages_dir=None, baseline_dir=None,
          bank_root=None) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    """Migration: the report from the v1 packages, every table reproduced (nothing written)."""
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    src = Path(packages_dir) if packages_dir else record_dir
    packages = {n: _load_package(src / n, n) for n in (CP, SP) if (src / n).exists()}
    if CP not in packages:
        raise ReportError(f"no {CP} package at {src / CP}: a migration reads the v1 packages "
                          "(--packages-dir); a written report is verified without them by --check")
    return _render(run_id, packages, record_dir, _baseline_for(packages, baseline_dir), bank_root)


def _packages_from_report(files: Mapping[str, bytes], man: Mapping[str, Any], record_dir: Path,
                          baseline_dir) -> Tuple[Dict[str, Package], Dict[str, Any]]:
    """The v1 tables reproduced from the report alone, each against its historical SHA-256."""
    hist = {k: v for k, v in (man.get("historical_v1_sources") or {}).items() if k != "note"}
    packages = {}
    for name, src in hist.items():
        if name not in PACKAGE_FORMATS:
            raise ReportError(f"unknown historical source {name!r}")
        copy = files.get(str(src.get("manifest_copy")))
        if copy is None or _sha(copy) != src.get("manifest_sha256"):
            raise ReportError(f"{source_manifest(name)}: missing or not the historical "
                              f"{name}/{MANIFEST} ({src.get('manifest_sha256')})")
        m, cols = _manifest(name, copy)
        packages[name] = Package(name, m, copy, {}, cols)
    if CP not in packages:
        raise ReportError(f"the report names no historical {CP} package")
    baseline = _baseline_for(packages, baseline_dir)
    columns = {(p, t): c for p, pkg in packages.items() for t, c in pkg.columns.items()}
    kept = reconstruct(files, columns)
    problems = []
    rebuilt = _rebuilt(lambda p, t: _text(kept[(p, t)]), packages, baseline, record_dir)
    for p, pkg in packages.items():
        for rel, o in (pkg.manifest.get("outputs") or {}).items():
            t = _table_name(rel)
            data = kept.get((p, t), rebuilt.get((p, t)))
            if data is None:
                problems.append(f"{p}/{t}: not reproduced by the report")
            elif _sha(data) != o.get("sha256"):
                problems.append(f"{p}/{t}: reproduced from the report, differs from its "
                                "historical v1 SHA-256")
            else:
                pkg.raw[t] = data
    if problems:
        raise ReportError("the report does not reproduce the historical v1 tables:\n  - "
                          + "\n  - ".join(problems))
    return packages, baseline


def verify(run_id: str, *, record_dir=None, report_dir=None, baseline_dir=None,
           bank_root=None) -> Dict[str, Any]:
    """Verification without any v1 package: ``status`` as ``complementarity.compare_package``."""
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    report_dir = Path(report_dir or record_dir / REPORT_DIR)
    if not (report_dir / MANIFEST).is_file():
        raise ReportError(f"no report at {report_dir}: run `report` (the migration) first")
    files = {rel: (report_dir / rel).read_bytes() for rel in C._listing(report_dir)}
    man_b, side = files[MANIFEST], (files.get(SIDECAR) or b"").decode("ascii").split()
    if not side or side[0].lower() != _sha(man_b):
        raise ReportError(f"{report_dir / SIDECAR} is not the SHA-256 of {MANIFEST}")
    man = json.loads(man_b.decode("utf-8"))
    if man.get("format") != FORMAT or man.get("run_id") != run_id:
        raise ReportError(f"{report_dir / MANIFEST} is not a {FORMAT} report of run {run_id!r}")
    outputs = man.get("outputs") or {}
    problems = [f"{n}: not in the report's {MANIFEST}" for n in sorted(
        set(files) - set(outputs) - {MANIFEST, SIDECAR})]
    for name, o in outputs.items():
        data = files.get(name)
        if data is None:
            problems.append(f"{name}: missing")
        elif _sha(data) != o.get("sha256"):
            problems.append(f"{name}: SHA-256 differs from the report's {MANIFEST}")
        elif name.endswith(".csv"):
            f = _text(data)
            if list(f.columns) != o.get("columns") or len(f) != o.get("rows"):
                problems.append(f"{name}: schema or row count differs from the report's "
                                f"{MANIFEST}")
    now = _records(record_dir)
    problems += [f"record {n}: {'missing' if n not in now else 'changed'} since the report"
                 for n, r in (man.get("records") or {}).items() if now.get(n) != r]
    if problems:
        raise ReportError(f"the report at {report_dir} does not verify:\n  - "
                          + "\n  - ".join(problems))
    packages, baseline = _packages_from_report(files, man, record_dir, baseline_dir)
    rendered, stats = _render(run_id, packages, record_dir, baseline, bank_root)
    status, diffs = C.compare_package(report_dir, rendered)
    return {"status": status, "problems": diffs, "files": rendered, "stats": stats}


def run(run_id: str, *, check: bool = False, record_dir=None, packages_dir=None,
        baseline_dir=None, out_dir=None) -> Dict[str, Any]:
    """Migrate and write ``report_v1/`` once, or (``check``) verify it without the packages."""
    record_dir = Path(record_dir or phase0.RECORD_ROOT / run_id)
    target = Path(out_dir) if out_dir else record_dir / REPORT_DIR
    if check:
        out = verify(run_id, record_dir=record_dir, report_dir=target, baseline_dir=baseline_dir)
        return {**out, "directory": target, "mode": "verification"}
    files, stats = build(run_id, record_dir=record_dir, packages_dir=packages_dir,
                         baseline_dir=baseline_dir)
    return {"status": C.write_package(target, files), "problems": [], "directory": target,
            "stats": stats, "files": files, "mode": "migration"}
