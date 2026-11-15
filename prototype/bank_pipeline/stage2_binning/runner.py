"""
stage2_binning.runner
=====================
PR 39 Commit 3, Step 3A: the NSD and ECI binning revalidation of the frozen
execution contract (section 6.2; gates A3.1-A3.3). From
``prototype/bank_pipeline``:

    python -m stage2_binning run    [--run-id c3_YYYYMMDD_HHMMSS]
    python -m stage2_binning check  --run-id <run_id>

run
    Rebuilds GLOBAL_SPLIT as the notebooks do and keeps the training side only
    (``stage2_crossfit.runner.load_train_side``: the test labels are dropped
    unread). The population must match the frozen PR 38 protocol's pins.

    E-B. For each outer fold k of the shared plan (``8aea082fcaf371b7``) a
    fresh Stage 2 engineer is fitted on the training rows outside k and
    transforms the rows of k from their features alone; the source features
    are binned with the locked thresholds (the production binning path).
    Pooled and per fold: support, positives, rate, lift and a Wilson 95%
    interval for every ``nsd_*`` / ``eci_*`` bin, beside the historical
    in-sample table (one engineer fitted on all training rows) and the rows
    whose bin it moves. T1-T4 (decision D6) are evaluated on the out-of-fold
    table and the family is retained or flagged.

    E-B2. From the accepted PR 38 run (``acceptance.require_accepted``): the
    selected GLASS Pass 1 rules of every outer fold and of the full-train fit,
    the full-train Pass 2 rules, and the saved RF Router forests' impurity
    importances. Fit-time records only; no outer-OOF route, Stage 4 or test
    result is read, and nothing here is a decision input.

    Written once, never replaced:

        research_logs/pr39/<run_id>/stage2_binning_evidence.json

check
    Recomputes every section of the record (population, frozen identities,
    binning identity, E-B, triggers, decision, E-B2) and requires each to
    equal the record exactly.

No router is fitted. A flagged family stops here: its candidate must be
derived and recorded by the scripted section 6.2 procedure before any router
fit, and that procedure is not part of this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from . import evidence as E

__all__ = ["BANK_ROOT", "RECORD_ROOT", "RECORD_FILE", "RECORD_FORMAT", "PR38_RUN_ID",
           "new_run_id", "oof_sources", "eb_evidence", "eb2_evidence", "computed_sections",
           "run", "check", "main"]

BANK_ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = BANK_ROOT / "research_logs" / "pr39"
RECORD_FILE = "stage2_binning_evidence.json"
RECORD_FORMAT = "pr39.stage2.binning_evidence/1"
PR38_RUN_ID = "c3_20261112_153333"
CONTRACT = {"record": "documentation/methodology/pr39_execution_contract.md",
            "version": "1.0.0", "section": "6.2", "gates": ["A3.1", "A3.2", "A3.3"],
            "decision": "D6"}
CODE_FILES = (
    "stage2_binning/*.py", "shared/stage2/binning.py", "shared/stage2/feature_engineering.py",
    "stage2_crossfit/g13.py", "stage2_crossfit/inner_features.py", "stage2_crossfit/runner.py",
    "stage2_crossfit/acceptance.py", "stage2_crossfit/evidence.py",
    "data/preprocessing/bank_preprocessing.py", "data/global_splits/split_manager.py",
)
LIMITATIONS = [
    "Every threshold, historical or revised, is a fixed constant chosen with all training "
    "labels. Outer-fold cross-fitting does not remove that dependence (section 6.2).",
    "PR 38 recorded the GLASS Pass 2 rules of each outer fold by rule id only, and saved only "
    "the full-train RF Router forests: E-B2 has the per-fold Pass 1 rules, the full-train "
    "Pass 1 and Pass 2 rules and the full-train forests' importances.",
]
_SKIP_IN_CHECK = ("run_id", "started_utc", "finished_utc", "code", "environment", "status",
                  "registered", "format", "contract", "limitations")


class RunnerError(RuntimeError):
    """The run cannot proceed or a check failed; the reason is stated."""


def new_run_id() -> str:
    return "c3_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _canonical(obj) -> Any:
    from stage2_crossfit.evidence import to_json_safe
    return json.loads(json.dumps(to_json_safe(obj), allow_nan=False))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_identity() -> Dict[str, Any]:
    files = {}
    for pattern in CODE_FILES:
        for p in sorted(BANK_ROOT.glob(pattern)):
            files[p.relative_to(BANK_ROOT).as_posix()] = _sha256(
                p.read_bytes().replace(b"\r\n", b"\n"))

    def git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=BANK_ROOT, capture_output=True,
                                 text=True, timeout=30)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"head": git("rev-parse", "HEAD"),
            "uncommitted_tracked_changes": None if status is None else bool(status),
            "files_sha256_lf": files}


def environment() -> Dict[str, Any]:
    import scipy
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__}


def _record_path(run_id: str, record_root: Path) -> Path:
    return Path(record_root) / run_id / RECORD_FILE


# ----------------------------------------------------------------------
# E-B
# ----------------------------------------------------------------------
def oof_sources(ts):
    """
    Out-of-fold Stage 2 source features: for each outer fold k a fresh
    ``RFFeatureEngineer`` fitted on the rows outside k transforms the rows of
    k (features only). The same computation as ``g13.inner_refit_features``,
    with each fit's provenance kept.
    """
    from shared.stage2.feature_engineering import RFFeatureEngineer
    from shared.stage_io import label_fingerprint
    X, y, fold_id = ts.X_train, ts.y_train.astype(int), np.asarray(ts.fold_id)
    parts, provenance = [], []
    for k in range(len(ts.folds)):
        held = fold_id == k
        fit_X, fit_y = X[~held], y[~held]
        engineer = RFFeatureEngineer().fit(fit_X, fit_y)
        parts.append(engineer.transform(X[held]))
        provenance.append({
            "fold": k, "fit_rows": int((~held).sum()), "fit_positives": int(fit_y.sum()),
            "fit_label_fingerprint": label_fingerprint(fit_X.index, fit_y),
            "heldout_rows": int(held.sum()),
            "heldout_rows_in_fit": int(fit_X.index.isin(X.index[held]).sum()),
        })
    return pd.concat(parts).loc[X.index], provenance


def eb_evidence(ts) -> Dict[str, Any]:
    """E-B: the out-of-fold and in-sample bin tables, overlap and moves."""
    from shared.stage2.feature_engineering import RFFeatureEngineer
    from stage2_crossfit.g13 import bin_source
    from stage2_crossfit.inner_features import frame_fingerprint
    y, fold_id, n_folds = ts.y_train.astype(int).to_numpy(), np.asarray(ts.fold_id), len(ts.folds)
    oof_src, provenance = oof_sources(ts)
    if any(p["heldout_rows_in_fit"] for p in provenance):
        raise E.EvidenceError("a held-out row is in its own fold's engineer fit")
    in_src = RFFeatureEngineer().fit_transform(ts.X_train, ts.y_train.astype(int))
    oof_bins, in_bins = bin_source(oof_src), bin_source(in_src)
    families, labels = {}, {}
    for fam in E.FAMILIES:
        names = E.family_bins(fam)
        lab_oof, lab_in = E.family_labels(oof_bins, fam), E.family_labels(in_bins, fam)
        labels[fam] = lab_oof
        families[fam] = {
            "source": E.BINNING_STRATEGY[fam]["source"],
            "thresholds": [list(b) for b in E.BINNING_STRATEGY[fam]["bins"]],
            "historical_rationale": E.HISTORICAL_RATIONALE[fam],
            "oof": E.bin_table(lab_oof, y, fold_id, names, n_folds),
            "in_sample": E.bin_table(lab_in, y, fold_id, names, n_folds, per_fold=False),
            "moved_by_in_sample_engineer": E.moved_rows(lab_oof, lab_in, names),
            "oof_bins_fingerprint": frame_fingerprint(oof_bins[names]),
        }
    return {
        "method": ("outer fold k: RFFeatureEngineer fitted on the training rows outside k, "
                   "applied to the rows of k (features only), binned with the thresholds "
                   "under test through the production binning path; in-sample: one engineer "
                   "fitted on all training rows (the historical method)"),
        "fold_fits": provenance,
        "families": families,
        "nsd_eci_joint_oof": E.joint_table(labels["NSD"], labels["ECI"], y),
    }


# ----------------------------------------------------------------------
# E-B2
# ----------------------------------------------------------------------
GATE_ACCEPTED = "stage2_crossfit.acceptance.require_accepted passed"
GATE_SUPPLIED = "caller-supplied candidate: not verified as accepted"


def eb2_evidence(candidate, n_folds: int = 5, *, gate: str = GATE_SUPPLIED) -> Dict[str, Any]:
    """Router utility from the accepted candidate's fit-time records and saved models."""
    folds = [candidate.fold_evidence("glass", k) for k in range(n_folds)]
    full = (candidate.evidence.get("full_train") or {}).get("glass")
    if not full or "pass1_rules" not in full:
        raise E.EvidenceError("the candidate has no full-train GLASS evidence")
    bundle = candidate.load(candidate.glass_path)
    rules = bundle["rules"]
    pass1_rules = [r for r in rules if r["pass"] == "Pass 1"]
    pass2_rules = [r for r in rules if r["pass"] == "Pass 2"]
    pass1_bundle, pass2_bundle = E.rule_literals(pass1_rules), E.rule_literals(pass2_rules)
    pass1_full = E.rule_literals(full["pass1_rules"])
    if sorted(pass1_bundle) != sorted(pass1_full):
        raise E.EvidenceError("the GLASS bundle's Pass 1 rules are not the full-train evidence's")
    if [r["rule_id"] for r in rules if r["pass"] == "Pass 2"] != \
            list(full["rule_ids"]["pass2"]):
        raise E.EvidenceError("the GLASS bundle's Pass 2 rules are not the full-train evidence's")
    rf = candidate.load(candidate.rf_path)
    importance = {p: rf[f"{p}_model"].named_steps["clf"].feature_importances_
                  for p in ("pass1", "pass2")}
    summary = E.eb2_summary(
        pass1_by_fold=[E.rule_literals(ev["pass1_rules"]) for ev in folds],
        pass2_ids_by_fold=[ev["rule_ids"]["pass2"] for ev in folds],
        pass1_full_train=pass1_full, pass2_full_train=pass2_bundle,
        rf_importance=importance, rf_feature_names=rf["feature_names"])
    summary["glass_rules"] = {
        "pass1_by_fold": [E.rule_entries(ev["pass1_rules"]) for ev in folds],
        "pass1_full_train": E.rule_entries(pass1_rules),
        "pass2_full_train": E.rule_entries(pass2_rules),
        "precision_and_coverage": "fit-time, on each fit's own training rows",
    }
    return {
        "source": {
            "run_id": candidate.run_id, "role": candidate.role,
            "gate": gate,
            "manifest_sha256": candidate.manifest_sha256,
            "glass_bundle": {"path": candidate.relative(candidate.glass_path),
                             "sha256": candidate.sha256(candidate.glass_path)},
            "rf_router": {"path": candidate.relative(candidate.rf_path),
                          "sha256": candidate.sha256(candidate.rf_path)},
        },
        "read": ("fit-time records and saved models only: per-fold GLASS Pass 1 rules and "
                 "Pass 2 rule ids (evidence.json), full-train GLASS rules (bundle), RF Router "
                 "impurity importances (saved full-train forests); no outer-OOF route, Stage 4 "
                 "or test result"),
        "not_recorded": ["GLASS Pass 2 rule conditions per outer fold (rule ids only)",
                         "RF Router per-fold forests (only the full-train forests are saved)"],
        **summary,
    }


def frozen_identities() -> Dict[str, Any]:
    """A3.3: the frozen records now, against the accepted run's committed manifest."""
    from stage2_crossfit import phase0
    from stage2_crossfit.runner import primary_settings
    settings = primary_settings("fold_label_excluded")
    current = {"identities": settings.checks["identities"],
               "feature_pathway": settings.checks["feature_pathway"]}
    path = phase0.RECORD_ROOT / PR38_RUN_ID / "primary_MANIFEST.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    accepted = {"identities": man["identities"], "feature_pathway": man["feature_pathway"]}
    current, accepted = _canonical(current), _canonical(accepted)
    return {"current": current, "accepted_run": accepted,
            "accepted_run_manifest": path.relative_to(BANK_ROOT).as_posix(),
            "unchanged": current == accepted}


# ----------------------------------------------------------------------
# the record
# ----------------------------------------------------------------------
def computed_sections(ts, candidate, *, population: Mapping[str, Any],
                      frozen: Optional[Mapping[str, Any]] = None,
                      gate: str = GATE_SUPPLIED) -> Dict[str, Any]:
    from shared.stage2.binning import binning_identity
    eb = eb_evidence(ts)
    trig = {fam: E.triggers(fam, eb["families"][fam]["oof"]) for fam in E.FAMILIES}
    fam_decision = E.decide(trig)
    retained = all(d["outcome"] == "retain" for d in fam_decision.values())
    identity = binning_identity()
    return _canonical({
        "population": population,
        "frozen_records": frozen,
        "scope": {"revalidated": list(E.FAMILIES),
                  "retained_without_revalidation": ["JED", "CCI", "DOW", "BEHAV", "CAMPAIGN",
                                                    "CPI"]},
        "binning_identity": identity,
        "intended_semantics": E.DIRECTIONS,
        "criteria": E.CRITERIA,
        "eb": eb,
        "triggers": trig,
        "decision": {
            "families": fam_decision,
            "outcome": "retain" if retained else "candidate_required",
            "binning_identity": identity if retained else None,
            "inner_feature_record": ("unchanged: inner_feature_isolation_v1.json (no adoption)"
                                     if retained else "a new version is required on adoption"),
            "stage2_regeneration": (
                "not required by the binning: no revision adopted (section 5.2); reuse of "
                f"{PR38_RUN_ID} remains subject to the Commit 4A checks" if retained else
                "required if a candidate is adopted (section 5.2), after a G13 re-audit"),
            "router_utility": ("E-B2 is reported; it vetoes only through the fold-0 preflight, "
                               "which no retained family needs; it is not a decision input"),
        },
        "eb2": eb2_evidence(candidate, len(ts.folds), gate=gate),
    })


def _official_inputs():
    import shared.stage2.evaluation_protocol as ep
    from stage2_crossfit import acceptance as A
    from stage2_crossfit.runner import check_population, load_train_side
    ts = load_train_side()
    population = check_population(ts, ep.load_frozen_protocol())
    if not population["passed"]:
        raise E.EvidenceError("the training side is not the frozen population: "
                              + json.dumps(_canonical(population["checks"])))
    candidate = A.require_accepted(PR38_RUN_ID)
    return ts, candidate, _population(ts, population), frozen_identities()


def _population(ts, check: Mapping[str, Any]) -> Dict[str, Any]:
    return {"rows": int(len(ts.y_train)), "positives": int(ts.y_train.sum()),
            "router_split_fingerprint": ts.router_split_fingerprint,
            "train_label_fingerprint": ts.train_label_fingerprint,
            "fold_fingerprint": ts.fold_fingerprint, "n_folds": len(ts.folds),
            "frozen_protocol_check": check,
            "test_split": ("labels never read; features unused; the test index enters only the "
                           "row-only router split fingerprint")}


def run(*, run_id: Optional[str] = None, train_side=None, candidate=None,
        record_root: Path = RECORD_ROOT) -> Dict[str, Any]:
    """The Step 3A evidence run; returns the record it wrote."""
    run_id = run_id or new_run_id()
    path = _record_path(run_id, record_root)
    if path.exists():
        raise RunnerError(f"{path} already exists; evidence is written once (new run id)")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    registered = train_side is None and candidate is None
    if registered:
        ts, candidate, population, frozen = _official_inputs()
    else:
        if train_side is None or candidate is None:
            raise RunnerError("a non-registered run needs both a train side and a candidate")
        ts, frozen = train_side, None
        population = _population(ts, {"passed": None,
                                      "note": "train side supplied by the caller; not checked"})
    sections = computed_sections(ts, candidate, population=population, frozen=frozen,
                                 gate=GATE_ACCEPTED if registered else GATE_SUPPLIED)
    if registered and not sections["frozen_records"]["unchanged"]:
        raise E.EvidenceError("the frozen contract, protocol, profile or pathway identities "
                              "differ from the accepted run's (A3.3)")
    record = _canonical({
        "format": RECORD_FORMAT, "run_id": run_id, "contract": CONTRACT,
        "registered": registered,
        "status": ("official Step 3A evidence" if registered else
                   "NOT official: caller-supplied train side or candidate"),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **sections,
        "limitations": LIMITATIONS,
        "code": code_identity(), "environment": environment(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:                      # never replaces an existing file
        fh.write((json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return record


def check(run_id: str, *, train_side=None, candidate=None,
          record_root: Path = RECORD_ROOT) -> Dict[str, Any]:
    """Recompute every section of a written record; refuse any difference."""
    path = _record_path(run_id, record_root)
    if not path.is_file():
        raise RunnerError(f"no evidence record at {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format") != RECORD_FORMAT or record.get("run_id") != run_id:
        raise RunnerError(f"{path}: not a {RECORD_FORMAT} record for {run_id!r}")
    if record.get("registered"):
        ts, candidate, population, frozen = _official_inputs()
    else:
        if train_side is None or candidate is None:
            raise RunnerError("a non-registered record is checked with its train side and "
                              "candidate")
        ts, frozen = train_side, None
        population = record["population"]
    computed = computed_sections(ts, candidate, population=population, frozen=frozen,
                                 gate=GATE_ACCEPTED if record.get("registered") else GATE_SUPPLIED)
    differing = [k for k in computed if computed[k] != record.get(k)]
    extra = sorted(k for k in record if k not in computed and k not in _SKIP_IN_CHECK)
    if differing or extra:
        raise RunnerError(f"{path} does not verify: sections differ {differing}; "
                          f"unexpected {extra}")
    return record


def _summary(record: Mapping[str, Any]) -> str:
    lines = [f"run {record['run_id']} ({record['status']})",
             f"  binning identity {record['binning_identity']['sha256']}"]
    for fam, t in record["triggers"].items():
        d = record["decision"]["families"][fam]
        lines.append(f"  {fam}: " + ", ".join(f"{k} {'FIRED' if t[k]['fired'] else 'clear'}"
                                             for k in ("T1", "T2", "T3", "T4"))
                     + f" -> {d['outcome']}")
    lines.append(f"  decision: {record['decision']['outcome']}; "
                 f"{record['decision']['stage2_regeneration']}")
    if record.get("frozen_records") is not None:
        lines.append(f"  frozen records unchanged: {record['frozen_records']['unchanged']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stage2_binning", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the registered Step 3A evidence run (E-B, E-B2, T1-T4)")
    r.add_argument("--run-id", help="default: c3_<local YYYYMMDD_HHMMSS>")
    c = sub.add_parser("check", help="recompute a written record")
    c.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    try:
        record = run(run_id=args.run_id) if args.cmd == "run" else check(args.run_id)
    except RuntimeError as exc:                    # this tool's, and the PR 38 gates' refusals
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if args.cmd == "run":
        print(f"wrote {_record_path(record['run_id'], RECORD_ROOT).relative_to(BANK_ROOT).as_posix()}")
    else:
        print("OK: every section recomputes exactly")
    print(_summary(record))
    return 0 if record["decision"]["outcome"] == "retain" else 2
