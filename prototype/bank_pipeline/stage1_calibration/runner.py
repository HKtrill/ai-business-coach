"""
stage1_calibration.runner
=========================
PR 39 Commit 2, Step 2A: the evidence run E-C of the frozen execution contract
(section 6.1). From ``prototype/bank_pipeline``:

    python -m stage1_calibration run    [--run-id c2_YYYYMMDD_HHMMSS]
    python -m stage1_calibration check  --run-id <run_id>
    python -m stage1_calibration decide --run-id <run_id>

run
    Rebuilds GLOBAL_SPLIT as the notebooks do and keeps the training side only
    (``stage2_crossfit.runner.load_train_side``: the test labels are dropped
    unread; the test index enters only the row-only split fingerprint). The
    population must match the frozen PR 38 protocol's pins. The three Stage 1
    features are engineered on the training rows (the notebooks' full-train
    frame), then each arm's own stage is fitted once, as the notebooks fit it
    but with no test scoring and no save: GLASS ``CalibratedLRStage`` and
    black-box ``CalibratedStage1MLP``, ``calibration_method="auto"``, 10 outer x
    10 inner folds, 100 Optuna trials, seed 42. Each fit's nested calibration
    gives both families' out-of-fold vectors from the same base scores. Written
    once, never replaced:

        models/pr39/<run_id>/stage1_calibration_oof.joblib        (Git-ignored)
        research_logs/pr39/<run_id>/stage1_calibration_evidence.json

    Nothing is written under the Stage 1 artifact folders.
check
    Verifies the vectors (SHA-256, size, rows, label and fold identities) and
    recomputes the evidence and the C1-C5 decision from them. Both must equal
    the record exactly. A decision record beside it is verified too.
decide
    Records the owner's approval of an accepted run: runs ``check``, requires
    registered evidence whose selection is the code's Stage 1 policy
    (``shared.stage1.oof_calibration.STAGE1_CALIBRATION_POLICY``), and writes,
    once, ``research_logs/pr39/<run_id>/stage1_calibration_decision.json``.
    It pins the evidence record by SHA-256 and states the selection as the
    registered rule's, not as the better family.

The registered settings cannot be changed from the command line. A run with
other settings or a caller-supplied train side (tests) is recorded with
``registered: false`` and is never official evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from . import evidence as E

__all__ = ["BANK_ROOT", "RECORD_ROOT", "OUTPUT_ROOT", "RECORD_FILE", "VECTORS_FILE",
           "RECORD_FORMAT", "REGISTERED_SETTINGS", "new_run_id", "fit_arms",
           "vectors_frame", "evidence_from_frame", "run", "check", "decide", "main"]

BANK_ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = BANK_ROOT / "research_logs" / "pr39"
OUTPUT_ROOT = BANK_ROOT / "models" / "pr39"
RECORD_FILE = "stage1_calibration_evidence.json"
VECTORS_FILE = "stage1_calibration_oof.joblib"
RECORD_FORMAT = "pr39.stage1.calibration_evidence/1"
DECISION_FILE = "stage1_calibration_decision.json"
DECISION_FORMAT = "pr39.stage1.calibration_decision/1"
CONTRACT = {"record": "documentation/methodology/pr39_execution_contract.md",
            "version": "1.0.0", "section": "6.1", "gates": ["A2.1", "A2.2"]}

#: As the notebooks fit Stage 1 (GLASS Cell 8 via ``train_lr_stage``; black-box
#: Cell 9), with every other argument at its default.
REGISTERED_SETTINGS: Dict[str, Any] = {
    "random_state": 42,
    "engineer": {"n_bins": 20, "smoothing_factor": 100},
    "glass_lr": {"tune_hyperparameters": True, "calibration_method": "auto",
                 "optimize_threshold": True, "cv_folds": 10, "n_trials": 100,
                 "tune_cv_folds": 10},
    "blackbox_mlp": {"calibration_method": "auto", "cv_folds": 10, "n_trials": 100},
}

#: The code the evidence depends on, hashed with CRLF normalised to LF (as the
#: PR 37 baseline does), so a Windows checkout gives the same identity.
CODE_FILES = (
    "stage1_calibration/*.py", "shared/stage1/*.py", "shared/metrics.py",
    "shared/thresholds.py", "shared/stage_io.py", "glass_pipeline/lr/*.py",
    "blackbox_pipeline/models/mlp/*.py", "blackbox_pipeline/models/mlp/evaluation/*.py",
    "glass_pipeline/glass_arbiter/meta_stage.py", "stage2_crossfit/runner.py",
    "data/preprocessing/bank_preprocessing.py", "data/global_splits/split_manager.py",
)


class RunnerError(RuntimeError):
    """The evidence run cannot proceed or a check failed; the reason is stated."""


def new_run_id() -> str:
    return "c2_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------------------------------------------------
# Fits and vectors
# ----------------------------------------------------------------------
def fit_arms(ts, settings: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each arm's own Stage 1 ``fit`` on the training rows (no test, no save)."""
    from blackbox_pipeline.models.mlp import CalibratedStage1MLP
    from glass_pipeline.lr.lr_stage_calibrated import CalibratedLRStage
    from shared.stage1.feature_engineering import LRFeatureEngineer

    seed = settings["random_state"]
    X_raw, y = ts.X_train, ts.y_train
    X_eng = LRFeatureEngineer(**settings["engineer"]).fit_transform(X_raw, y)
    fits = {}
    for arm, make in (("glass_lr", CalibratedLRStage), ("blackbox_mlp", CalibratedStage1MLP)):
        start = time.perf_counter()
        stage = make(random_state=seed, **settings[arm]).fit(X_eng, y, X_train_raw=X_raw)
        fits[arm] = {"stage": stage, "wall_seconds": round(time.perf_counter() - start, 1)}
    return fits


def _stage_config(stage) -> Dict[str, Any]:
    cfg = getattr(stage, "config", None)
    return cfg.to_dict() if hasattr(cfg, "to_dict") else stage.config_dict()


def vectors_frame(ts, fits: Mapping[str, Mapping[str, Any]], settings: Mapping[str, Any]):
    """The per-row evidence and each fit's provenance; refuses anything incomplete (C4)."""
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    from shared.stage_io import fold_assignment

    y = ts.y_train.astype(int)
    columns: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    fold_ids = {}
    for arm in E.ARMS:
        stage = fits[arm]["stage"]
        res = getattr(stage, "calibration_result_", None)
        if res is None:
            raise E.EvidenceError(f"{arm}: the fit kept no calibration result")
        if set(res.calibrated_oof) != set(E.FAMILIES):
            raise E.EvidenceError(f"{arm}: calibrated families {sorted(res.calibrated_oof)}; "
                                  f"E-C needs both {E.FAMILIES} (method='auto')")
        cv = settings[arm]["cv_folds"]
        want = fold_assignment(y, cv_folds=cv, random_state=settings["random_state"],
                               index=y.index).to_numpy()
        got = np.asarray(stage.train_fold_id)
        if not np.array_equal(got, want):
            raise E.EvidenceError(f"{arm}: the outer folds are not fold_assignment's")
        if set(res.fold_family) != set(range(cv)):
            raise E.EvidenceError(f"{arm}: inner selections for folds {sorted(res.fold_family)}")
        for name, s in (("base_score", res.base_scores_oof),
                        *((f, res.calibrated_oof[f]) for f in E.FAMILIES)):
            if not s.index.equals(y.index):
                raise E.EvidenceError(f"{arm} {name}: not aligned to the training rows")
            columns[f"{arm}_{name}"] = s.to_numpy(dtype=float)
        fold_ids[arm] = got
        meta[arm] = {
            "stage": type(stage).__name__,
            "config": _stage_config(stage),
            "best_params": stage.best_params,
            "best_cv_roc_auc": stage.best_cv_roc_auc,
            "final_model_family_auto": res.method,
            "oof_family_by_fold": res.metrics.get("oof_family_by_fold"),
            "fold_inner": {int(k): {"family": res.fold_family[k],
                                    "inner_brier": res.fold_inner_brier.get(k)}
                           for k in sorted(res.fold_family)},
            "oof_brier_by_family": res.brier_oof,
            "oof_provenance": res.provenance,
            "wall_seconds": fits[arm]["wall_seconds"],
        }
    if not np.array_equal(fold_ids["glass_lr"], fold_ids["blackbox_mlp"]):
        raise E.EvidenceError("the two arms used different outer folds")
    frame = pd.DataFrame({"fold_id": fold_ids["glass_lr"], "y": y.to_numpy(), **columns},
                         index=y.index)
    plan = {"folds": int(settings["glass_lr"]["cv_folds"]),
            "random_state": int(settings["random_state"]),
            "fold_fingerprint": fold_fingerprint(frame["fold_id"].to_numpy())}
    return frame, meta, plan


def evidence_from_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    """E-C for both arms and the C1-C5 decision, from the per-row vectors alone."""
    y, fold_id = frame["y"].to_numpy(), frame["fold_id"].to_numpy()
    arms = {arm: E.arm_evidence(y, fold_id, frame[f"{arm}_base_score"].to_numpy(),
                                {f: frame[f"{arm}_{f}"].to_numpy() for f in E.FAMILIES})
            for arm in E.ARMS}
    return {"arms": arms, "decision": E.apply_rule(arms)}


# ----------------------------------------------------------------------
# Provenance and files
# ----------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def _canonical(obj) -> Any:
    return json.loads(json.dumps(_json_safe(obj)))


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
    import joblib
    import optuna
    import scipy
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
            "optuna": optuna.__version__}


def _paths(run_id: str, record_root: Path, output_root: Path):
    return Path(record_root) / run_id / RECORD_FILE, Path(output_root) / run_id / VECTORS_FILE


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:                  # never replaces an existing file
        fh.write(data)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(BANK_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _vector_identity(frame: pd.DataFrame) -> Dict[str, Any]:
    from shared.stage_io import label_fingerprint
    return {"rows": int(len(frame)), "positives": int(frame["y"].sum()),
            "columns": list(frame.columns),
            "train_label_fingerprint": label_fingerprint(frame.index, frame["y"]),
            # each vector's own SHA-256 (contract 6.1), over its little-endian values
            "column_sha256": {
                "index": _sha256(np.asarray(frame.index, dtype="<i8").tobytes()),
                **{c: _sha256(np.ascontiguousarray(
                    frame[c].to_numpy(dtype="<i8" if c in ("fold_id", "y") else "<f8"))
                    .tobytes()) for c in frame.columns}}}


# ----------------------------------------------------------------------
# run / check
# ----------------------------------------------------------------------
def run(*, run_id: Optional[str] = None, settings: Mapping[str, Any] = REGISTERED_SETTINGS,
        train_side=None, record_root: Path = RECORD_ROOT,
        output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    """The evidence run; returns the record it wrote."""
    run_id = run_id or new_run_id()
    record_path, vectors_path = _paths(run_id, record_root, output_root)
    for p in (record_path, vectors_path):
        if p.exists():
            raise RunnerError(f"{p} already exists; evidence is written once (new run id)")

    if train_side is None:
        import shared.stage2.evaluation_protocol as ep
        from stage2_crossfit.runner import check_population, load_train_side
        ts = load_train_side()
        population = check_population(ts, ep.load_frozen_protocol())
        if not population["passed"]:
            raise E.EvidenceError("the training side is not the frozen population: "
                                  + json.dumps(_json_safe(population["checks"])))
    else:
        ts = train_side
        population = {"passed": None, "note": "train side supplied by the caller; not checked"}
    registered = train_side is None and _canonical(settings) == _canonical(REGISTERED_SETTINGS)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fits = fit_arms(ts, settings)
    frame, meta, plan = vectors_frame(ts, fits, settings)
    computed = evidence_from_frame(frame)

    buf = io.BytesIO()
    import joblib
    joblib.dump(frame, buf, compress=3)
    data = buf.getvalue()
    record = _canonical({
        "format": RECORD_FORMAT, "run_id": run_id, "contract": CONTRACT,
        "registered": registered,
        "status": ("official E-C evidence" if registered else
                   "NOT official: non-registered settings or caller-supplied train side"),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "population": {
            "rows": int(len(ts.y_train)), "positives": int(ts.y_train.sum()),
            "router_split_fingerprint": ts.router_split_fingerprint,
            "train_label_fingerprint": ts.train_label_fingerprint,
            "frozen_protocol_check": population,
            "stage1_outer_folds": plan,
            "test_split": ("labels never read; features unused; the test index enters only "
                           "the row-only router split fingerprint"),
        },
        "fits": meta,
        "vectors": {"path": _relative(vectors_path), "sha256": _sha256(data),
                    "bytes": len(data), **_vector_identity(frame)},
        "evidence": computed["arms"],
        "decision": computed["decision"],
        "code": code_identity(), "environment": environment(),
    })
    _write_once(vectors_path, data)
    _write_once(record_path, (json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                .encode("utf-8"))
    return record


def check(run_id: str, *, record_root: Path = RECORD_ROOT,
          output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    """Recompute the evidence and decision from the vectors; refuse any difference."""
    import joblib
    record_path, vectors_path = _paths(run_id, record_root, output_root)
    if not record_path.is_file():
        raise RunnerError(f"no evidence record at {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("format") != RECORD_FORMAT or record.get("run_id") != run_id:
        raise RunnerError(f"{record_path}: not a {RECORD_FORMAT} record for {run_id!r}")
    if not vectors_path.is_file():
        raise RunnerError(f"the vectors are missing: {vectors_path}")
    data = vectors_path.read_bytes()
    want = record["vectors"]
    if _sha256(data) != want["sha256"] or len(data) != want["bytes"]:
        # Fail closed: a file that is not the recorded one is never unpickled.
        raise RunnerError(f"{record_path} does not verify:\n  - vectors: SHA-256 or size "
                          "differs from the record")
    problems = []
    frame = joblib.load(io.BytesIO(data))
    ident = _canonical(_vector_identity(frame))
    problems += [f"vectors: {k} differs" for k in ident if ident[k] != want.get(k)]
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    if fold_fingerprint(frame["fold_id"].to_numpy()) != \
            record["population"]["stage1_outer_folds"]["fold_fingerprint"]:
        problems.append("vectors: the outer fold assignment differs")
    computed = _canonical(evidence_from_frame(frame))
    if computed["arms"] != record["evidence"]:
        problems.append("evidence: recomputed values differ from the record")
    if computed["decision"] != record["decision"]:
        problems.append("decision: the recomputed C1-C5 outcome differs from the record")
    if problems:
        raise RunnerError(f"{record_path} does not verify:\n  - " + "\n  - ".join(problems))
    decision_path = record_path.with_name(DECISION_FILE)
    if decision_path.is_file():
        _verify_decision(decision_path, record_path, record)
    return record


# ----------------------------------------------------------------------
# decide
# ----------------------------------------------------------------------
def _decision_record(evidence: Mapping[str, Any], evidence_bytes: bytes,
                     evidence_path: Path, policy: Mapping[str, Any]) -> Dict[str, Any]:
    """The decision record, as a pure function of the verified evidence (no timestamp)."""
    d = evidence["decision"]
    family = d["selected_family"]
    c3 = d["C3"]
    passed = {arm: sorted(k for k, v in c3[arm].items() if v["passed"]) for arm in E.ARMS}
    failed = {arm: sorted(k for k, v in c3[arm].items() if not v["passed"]) for arm in E.ARMS}
    by_arm = "; ".join(f"{arm}: passed {', '.join(passed[arm]) or 'none'}, "
                       f"failed {', '.join(failed[arm]) or 'none'}" for arm in E.ARMS)
    interpretation = [
        (f"{family} is the family the registered conservative rule C1-C5 selects. "
         f"Basis: {d['basis']}. Isotonic replaces the sigmoid default only if it "
         f"satisfies every C3 condition in both arms. Per arm: {by_arm}."),
        (f"This is a policy selection, not a finding that {family} is the "
         "better-calibrated family in either arm."),
    ]
    favoured = [arm for arm in E.ARMS if not failed[arm]]
    if family == "sigmoid" and favoured:
        interpretation.append(
            f"In {', '.join(favoured)} the evidence favoured isotonic on every C3 "
            "condition. One family for both arms (C1) gives that up: the tradeoff the "
            "contract records in section 6.1.")
    interpretation.append(
        f"The neutral alternative (D1) would select "
        f"{d['neutral_alternative']['would_select']}. It is reported for information "
        "and does not change the selection (C5).")

    def arm_summary(arm):
        a = evidence["evidence"][arm]
        boot = a["delta_brier"]["bootstrap"]
        return {
            "brier": {f: a["pooled"][f]["brier"] for f in E.FAMILIES},
            "ece": {f: a["pooled"][f]["ece"] for f in E.FAMILIES},
            "delta_brier_pooled": a["delta_brier"]["pooled"],
            "delta_brier_interval_95": [boot["lower"], boot["upper"]],
            "folds_isotonic_lower": a["delta_brier"]["folds_isotonic_lower"],
            "stage4_achieved_recall": {f: a["stage4"][f]["achieved_recall"]
                                       for f in E.FAMILIES},
            "stage4_vote_differences": a["stage4"]["vote_differences"]["rows"],
        }

    return _canonical({
        "record_id": policy["id"],
        "schema_version": DECISION_FORMAT,
        "record_version": policy["version"],
        "status": "frozen",
        "approval": ("the owner's approval of the accepted evidence run, recorded by "
                     "'python -m stage1_calibration decide' (PR 39, Commit 2, Step 2A); "
                     "takes effect when Commit 2 is created after local validation"),
        "contract": {**CONTRACT, "rule": "C1-C5", "gates": ["A2.2", "A2.3"]},
        "policy": {
            "family": family,
            "arms": list(E.ARMS),
            "outer_folds": "the same family in every outer fold; no per-fold or per-arm selection",
            "auto": "diagnostics only (the evidence run)",
            "code": "shared.stage1.oof_calibration.STAGE1_CALIBRATION_FAMILY",
        },
        "evidence": {
            "run_id": evidence["run_id"],
            "record": _relative(evidence_path),
            "record_sha256": _sha256(evidence_bytes),
            "record_bytes": len(evidence_bytes),
            "record_format": evidence["format"],
            "registered": evidence["registered"],
            "vectors": {k: evidence["vectors"][k] for k in ("path", "sha256", "bytes")},
            "verified": "stage1_calibration check: recomputed exactly from the vectors",
            "code_head": evidence["code"]["head"],
            "uncommitted_tracked_changes": evidence["code"]["uncommitted_tracked_changes"],
            "code_state": ("the Step 2A evidence code, before the policy change; its files "
                           "are hashed in the evidence record"),
            "environment": evidence["environment"],
        },
        "outcome": {"selected_family": family, "basis": d["basis"], "C3": c3,
                    "statement": d["statement"]},
        "summary": {arm: arm_summary(arm) for arm in E.ARMS},
        "interpretation": interpretation,
        "neutral_alternative": d["neutral_alternative"],
        "limitations": [
            ("The Optuna hyperparameters are tuned once with all training labels; the "
             "out-of-fold probabilities are conditional on that tuning, equally for both "
             "families."),
            "The bootstrap resamples rows with the fitted models held fixed, as in PR 38.",
            ("The historical per-arm selections were visible before the rule was frozen "
             "(contract F7); the rule was frozen with that exposure disclosed."),
            "E-C used training rows only; the test split was not read.",
        ],
    })


def _verify_decision(decision_path: Path, evidence_path: Path,
                     evidence: Mapping[str, Any]) -> None:
    from shared.stage1.oof_calibration import STAGE1_CALIBRATION_POLICY

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("schema_version") != DECISION_FORMAT:
        raise RunnerError(f"{decision_path}: not a {DECISION_FORMAT} record")
    policy = {"id": decision["record_id"], "version": decision["record_version"]}
    want = _decision_record(evidence, evidence_path.read_bytes(), evidence_path, policy)
    got = {k: v for k, v in decision.items() if k != "created_utc"}
    if got != want:
        diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise RunnerError(f"{decision_path} does not match the verified evidence: {diff}")
    if evidence["run_id"] == STAGE1_CALIBRATION_POLICY["evidence_run"] and \
            decision["policy"]["family"] != STAGE1_CALIBRATION_POLICY["family"]:
        raise RunnerError(f"{decision_path}: records {decision['policy']['family']!r}, the code "
                          f"uses {STAGE1_CALIBRATION_POLICY['family']!r}")


def decide(run_id: str, *, record_root: Path = RECORD_ROOT, output_root: Path = OUTPUT_ROOT,
           policy: Optional[Mapping[str, Any]] = None,
           require_registered: bool = True) -> Dict[str, Any]:
    """Write the decision record for a verified, registered run; returns it."""
    from shared.stage1.oof_calibration import STAGE1_CALIBRATION_POLICY

    policy = dict(policy or STAGE1_CALIBRATION_POLICY)
    evidence = check(run_id, record_root=record_root, output_root=output_root)
    evidence_path, _ = _paths(run_id, record_root, output_root)
    decision_path = evidence_path.with_name(DECISION_FILE)
    if decision_path.exists():
        raise RunnerError(f"{decision_path} already exists; a decision is written once")
    if require_registered and evidence.get("registered") is not True:
        raise RunnerError("only registered E-C evidence can be decided on")
    if run_id != policy["evidence_run"]:
        raise RunnerError(f"the code's Stage 1 policy names evidence run "
                          f"{policy['evidence_run']!r}, not {run_id!r}")
    if evidence["decision"]["selected_family"] != policy["family"]:
        raise RunnerError(f"the evidence selects {evidence['decision']['selected_family']!r}; "
                          f"the code's Stage 1 policy is {policy['family']!r}")
    record = {**_decision_record(evidence, evidence_path.read_bytes(), evidence_path, policy),
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_once(decision_path, (json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                .encode("utf-8"))
    return record


def _summary(record: Mapping[str, Any]) -> str:
    d = record["decision"]
    lines = [f"run {record['run_id']} ({record['status']})"]
    for arm, checks in d["C3"].items():
        lines.append(f"  {arm}: " + ", ".join(
            f"{k} {'pass' if v['passed'] else 'fail'}" for k, v in checks.items()))
    lines.append(f"  selected family: {d['selected_family']} ({d['basis']})")
    lines.append(f"  neutral alternative (descriptive): {d['neutral_alternative']['would_select']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stage1_calibration", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the registered E-C evidence run (both arms)")
    r.add_argument("--run-id", help="default: c2_<local YYYYMMDD_HHMMSS>")
    c = sub.add_parser("check", help="verify a written record against its vectors")
    c.add_argument("--run-id", required=True)
    d = sub.add_parser("decide", help="record the owner's approval of an accepted run")
    d.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            record = run(run_id=args.run_id)
        elif args.cmd == "check":
            record = check(args.run_id)
        else:
            decided = decide(args.run_id)
            record = check(args.run_id)
    except (RunnerError, E.EvidenceError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if args.cmd == "check":
        print("OK: the evidence and the decision recompute exactly from the vectors")
        if _paths(args.run_id, RECORD_ROOT, OUTPUT_ROOT)[0].with_name(DECISION_FILE).is_file():
            print("OK: the decision record matches the verified evidence")
    if args.cmd == "decide":
        path = _paths(args.run_id, RECORD_ROOT, OUTPUT_ROOT)[0].with_name(DECISION_FILE)
        print(f"wrote {_relative(path)}: policy {decided['policy']['family']}, evidence "
              f"sha256 {decided['evidence']['record_sha256']}")
    print(_summary(record))
    return 0
