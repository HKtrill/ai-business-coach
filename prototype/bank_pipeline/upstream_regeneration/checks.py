"""
upstream_regeneration.checks
============================
The checks of gates A4.1-A4.6 (execution contract sections 5.2, 7.4, 7.5).
Every check is recorded by name, passed or failed with its detail; a check
that raises is a recorded failure (``Gate``). Nothing here fits a model,
computes a Stage 4 metric or reads a Stage 4 result.

Section 7.5 (``preservation``)
    The snapshot (``phase0.verify_snapshot``), the baseline CLIs
    (``phase0.run_baseline_checks``: ``stage2_baseline.py --verify``, whose
    code-identity notes are recorded, and the archive ``--check``), the PR 38
    report (``report --check``, Windows only: F11) and the PR 38 run's
    structural acceptance (``require_accepted``).
Section 5.2 (``stage2_reuse``)
    The four regeneration triggers and the four reuse checks, on the final
    code. ``decide`` turns them into ``reuse`` or ``regenerate``.
A4.3-A4.5 (``stage1_checks``, ``stage3_checks``, ``stage4_input_checks``)
    Each Stage 1 and Stage 3 artifact against the frozen family, names,
    versions, fold plans, split and provenance (Stage 3: the two arms' split
    fingerprints, stamped under different registered schemes, compared by their
    canonical identity, ``stage3_split_identity``); the four arbiter-upstream
    input builds (``load_stage_outputs`` and ``load_blackbox_stage4_inputs``:
    ``build_stage4_inputs`` on the train and test side, inputs only); every
    file reloaded in a fresh interpreter (``fresh_reload``).
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["BANK_ROOT", "PR38_RUN_ID", "BINNING_RUN_ID", "Gate", "global_split",
           "frozen_population", "preservation", "report_check", "live_counterparts",
           "live_counterpart_problems", "r4_checks",
           "glass_rule_features", "stage2_reuse", "decide",
           "stage1_checks", "stage3_split_identity", "stage3_checks", "stage4_input_checks",
           "fresh_reload", "identities", "STAGE3_SPLIT_IDENTITY", "STAGE3_PROTOCOL"]

BANK_ROOT = Path(__file__).resolve().parents[1]
PR38_RUN_ID = "c3_20261112_153333"
#: The Step 3A evidence run that froze the bins (decision: retain).
BINNING_RUN_ID = "c3_20261114_231604"
BINNING_RECORD = f"research_logs/pr39/{BINNING_RUN_ID}/stage2_binning_evidence.json"
STAGE1_FAMILY = "sigmoid"                       # D1, Commit 2A
STAGE1_FOLDS, SEED = 10, 42


class Gate:
    """Named checks; a check that raises is recorded as failed with its error."""

    def __init__(self, gate: str):
        self.gate, self.checks = gate, []

    def check(self, name: str, fn: Callable[[], Any], detail: Any = "") -> bool:
        err = None
        try:
            ok = bool(fn())
        except Exception as exc:                    # noqa: BLE001 - recorded as a failure
            ok, err = False, f"{type(exc).__name__}: {str(exc)[:400]}"
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
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    def failed(self) -> List[str]:
        return [c["check"] for c in self.checks if not c["passed"]]

    def record(self) -> Dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed, "failed": self.failed(),
                "evidence": f"{len(self.checks) - len(self.failed())}/{len(self.checks)} checks "
                            "passed", "checks": list(self.checks)}


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def global_split() -> dict:
    """GLOBAL_SPLIT exactly as both notebooks build it (Cells 2-4 / 2-6); reads only."""
    from data.bank_database import DEFAULT_DB_PATH, raw_bank_exists
    if not raw_bank_exists(DEFAULT_DB_PATH):
        raise RuntimeError(f"no raw_bank table in {DEFAULT_DB_PATH}: run a cascade notebook's "
                           "database cell (build_database) first")
    from data.global_splits.split_manager import GlobalSplitManager
    from data.preprocessing.bank_preprocessing import BankPreprocessor
    with _quiet():
        df = BankPreprocessor(drop_leaky=True, verbose=False).fit_transform()
        return GlobalSplitManager(test_size=0.20, random_state=SEED).create_split(df, target_col="y")


def frozen_population() -> Dict[str, Any]:
    """The frozen protocol's pinned population (section 1)."""
    import shared.stage2.evaluation_protocol as ep
    return dict(ep.load_frozen_protocol()["population"])


# ----------------------------------------------------------------------
# section 7.5
# ----------------------------------------------------------------------
def report_check(run_id: str = PR38_RUN_ID) -> Dict[str, Any]:
    """``python -m stage2_crossfit report --run-id <id> --check`` (F11: on Windows)."""
    args = ["-m", "stage2_crossfit", "report", "--run-id", run_id, "--check"]
    proc = subprocess.run([sys.executable, *args], cwd=BANK_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    windows = platform.system() == "Windows"
    return {"command": "python " + " ".join(args), "exit_code": proc.returncode,
            "platform": platform.system(), "passed": proc.returncode == 0 and windows,
            "output_tail": tail,
            "note": None if windows else ("F11: report --check verifies on Windows only (record "
                                          "key order); run this step on Windows")}


def preservation(gate: Gate, sums_path) -> Dict[str, Any]:
    """Section 7.5 checks 1-4 (5 and 6, the inventory and git status, are the runner's)."""
    from stage2_crossfit import acceptance, phase0
    out: Dict[str, Any] = {}

    def snapshot():
        out["snapshot"] = phase0.verify_snapshot(sums_path)
        return out["snapshot"]["passed"]

    def baseline():
        out["baseline"] = phase0.run_baseline_checks(sums_path)
        return out["baseline"]["passed"]

    def report():
        out["report_check"] = report_check()
        return out["report_check"]["passed"]

    def accepted():
        cand = acceptance.require_accepted(PR38_RUN_ID)
        out["pr38_acceptance"] = {"run_id": PR38_RUN_ID, "accepted": True,
                                  "candidate_manifest_sha256": cand.manifest_sha256}
        return True

    gate.check("7.5.1 phase0.verify_snapshot passes (37 entries, the baseline's SHA256SUMS)",
               snapshot, lambda: out["snapshot"]["checks"])
    gate.check("7.5.2 phase0.run_baseline_checks passes (stage2_baseline.py --verify, "
               "stage2_baseline_archive.py --check)", baseline,
               lambda: {k: out["baseline"][k]["output_tail"][-4:] for k in ("verify", "archive_check")})
    gate.check(f"7.5.3 python -m stage2_crossfit report --run-id {PR38_RUN_ID} --check passes "
               "(Windows, F11)", report, lambda: out["report_check"]["output_tail"][-3:])
    gate.check(f"7.5.4 acceptance.require_accepted({PR38_RUN_ID!r}) passes", accepted,
               lambda: out["pr38_acceptance"])
    return out


def live_counterparts(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    lc = snapshot.get("live_counterparts") or {}
    return {"same": lc.get("same"), "different": sorted(lc.get("different") or []),
            "missing": sorted(lc.get("missing") or [])}


def live_counterpart_problems(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """
    The snapshot's live counterparts across a run. Only Step 4A's registered
    mutable aliases (``inventory.MUTABLE_ALIASES``: the latest copies and the D7
    CSVs its producers rewrite) may change status. Every other counterpart,
    historical immutable versions and the legacy routers' aliases alike, keeps
    its status: still the snapshot's bytes, or still the same recorded difference.
    """
    from .inventory import MUTABLE_ALIASES
    from stage2_crossfit.phase0 import _PREFIX
    mutable = {a for nb in MUTABLE_ALIASES for a in MUTABLE_ALIASES[nb]}

    def rel(path: str) -> str:
        return path[len(_PREFIX):] if path.startswith(_PREFIX) else path

    out = []
    for status in ("different", "missing"):
        b = {rel(x) for x in before.get(status) or []}
        a = {rel(x) for x in after.get(status) or []}
        out += [f"{x}: now {status}" for x in sorted(a - b) if x not in mutable]
        out += [f"{x}: no longer {status}" for x in sorted(b - a) if x not in mutable]
    if after.get("same") is None:
        out.append("no live counterparts recorded after the run")
    return out


# ----------------------------------------------------------------------
# section 5.2: Stage 2 reuse or regeneration
# ----------------------------------------------------------------------
def _file_manifest_checks(gate: Gate) -> Dict[str, Any]:
    """Reuse check 4: every candidate file's SHA-256 equals primary_MANIFEST.json."""
    from shared.pinned_artifacts import (STAGE2_MANIFEST, STAGE2_PINS, sha256_file,
                                         stage2_manifest_problems)
    man_path = BANK_ROOT / STAGE2_MANIFEST
    man = json.loads(man_path.read_text(encoding="utf-8"))
    root = BANK_ROOT / "models" / "pr38" / PR38_RUN_ID / "primary"
    rows = {}
    for rel, meta in sorted((man.get("files") or {}).items()):
        p = root / rel
        rows[rel] = {"listed": meta, "present": p.is_file(),
                     "bytes": p.stat().st_size if p.is_file() else None,
                     "sha256": sha256_file(p) if p.is_file() else None}
    gate.check("reuse 4: every file of the accepted candidate has the SHA-256 and size "
               "primary_MANIFEST.json lists",
               lambda: rows and all(r["present"] and r["bytes"] == r["listed"]["bytes"]
                                    and r["sha256"] == r["listed"]["sha256"] for r in rows.values()),
               lambda: {k: r["sha256"] == r["listed"]["sha256"] for k, r in rows.items()})
    problems = stage2_manifest_problems(BANK_ROOT)
    gate.check("reuse 4: the two pinned files (shared.pinned_artifacts.STAGE2_PINS) are the "
               "manifest's GLASS bundle and immutable RF Router, and primary_MANIFEST.json has "
               "its pinned SHA-256", lambda: not problems, "; ".join(problems) or "equal")
    return {"files": rows, "pins": [p.record() | {"role": p.role} for p in STAGE2_PINS]}


def _names_checks(gate: Gate, glass, rf) -> Dict[str, Any]:
    """Trigger 2: the accepted artifacts' Stage 2 names are the current code's."""
    from shared.stage2.binning import RF_FEATURES_BINARY, binning_identity
    from shared.stage2.feature_engineering import RF_SOURCE_FEATURES
    rec = json.loads((BANK_ROOT / BINNING_RECORD).read_text(encoding="utf-8"))
    ident = binning_identity()
    gate.check("trigger 2: binning identity now = the Step 3A record's (BINNING_STRATEGY, "
               "RF_FEATURES_BINARY)", lambda: ident["sha256"] == rec["binning_identity"]["sha256"],
               ident["sha256"])
    gate.check("trigger 2: the GLASS Router bundle's features = RF_FEATURES_BINARY, in order",
               lambda: list(glass["features"]) == list(RF_FEATURES_BINARY), len(RF_FEATURES_BINARY))
    gate.check("trigger 2: the RF Router's feature_names = RF_FEATURES_BINARY, in order",
               lambda: list(rf["feature_names"]) == list(RF_FEATURES_BINARY))
    used: List[str] = []

    def rules_use_current_bins():
        used[:] = glass_rule_features(glass)
        return bool(used) and set(used) <= set(RF_FEATURES_BINARY)

    gate.check("trigger 2: every feature the accepted GLASS rules use is a current bin",
               rules_use_current_bins, lambda: used)
    return {"binning_identity": ident, "rf_features_binary": list(RF_FEATURES_BINARY),
            "rf_source_features": list(RF_SOURCE_FEATURES), "glass_rule_features": list(used)}


def glass_rule_features(glass) -> List[str]:
    """
    Every feature the bundle's rules test, as ``ModelSaver._serialize_rules``
    saves them: a list of rule dicts (``"pass"`` Pass 1 / Pass 2), each
    ``"segment"`` a list of ``(feature, value)`` conditions. Anything else raises.
    """
    rules = glass["rules"]
    if not isinstance(rules, list) or not rules or not all(isinstance(r, Mapping) for r in rules):
        raise ValueError(f"rules is a {type(rules).__name__}, not ModelSaver's list of rule dicts")
    return sorted({str(feature) for r in rules for feature, _ in r["segment"]})


def _binning_recheck(gate: Gate) -> Dict[str, Any]:
    """Triggers 1-3: the Step 3A record recomputes exactly on the final code."""
    from stage2_binning import runner as B
    out = {}

    def recompute():
        rec = B.check(BINNING_RUN_ID)
        out.update(decision=rec["decision"]["outcome"],
                   frozen_records_unchanged=(rec.get("frozen_records") or {}).get("unchanged"))
        return True

    gate.check(f"triggers 1-3: python -m stage2_binning check --run-id {BINNING_RUN_ID} "
               "recomputes every section exactly (E-B, E-B2, T1-T4, frozen records)", recompute)
    gate.check("trigger 1: no NSD/ECI revision adopted (Step 3A decision: retain)",
               lambda: out.get("decision") == "retain", lambda: out.get("decision"))
    gate.check("trigger 3: the frozen contract, protocol, profiles and G13 pathway are the "
               "accepted run's (no frozen Stage 2 record changed version)",
               lambda: out.get("frozen_records_unchanged") is True)
    return out


def r4_checks(gate: Gate, glass_bundle, rf_artifact, ts) -> Dict[str, Any]:
    """
    V7's Stage 4 interface checks on the final code, run by PR 38's own checker
    (``acceptance._r4_checks``) with its own input contract: the artifacts keyed
    by its arms (``evidence.ARM_GLASS``, ``evidence.ARM_RF``), as its V7 builds
    them. Then V7's R5 check on the notes it returns: the RF Router's Pass 2
    eligibility is ``nested_t1`` on both sides. As in V7, a checker that raises
    is a recorded failure, never an abort.
    """
    from stage2_crossfit import acceptance as A
    from stage2_crossfit import evidence as E
    notes: Dict[str, Any] = {}
    try:
        notes = A._r4_checks(gate, {E.ARM_GLASS: glass_bundle, E.ARM_RF: rf_artifact}, ts)
    except Exception as exc:                        # noqa: BLE001 - recorded, as V7 does
        gate.check("R4 Stage 4 consumers ran", lambda: False, f"{type(exc).__name__}: {exc}")
    gate.check("V7 R5 at Stage 4: pass2_eligibility nested_t1 on both sides of the RF Router",
               lambda: notes[E.ARM_RF]["pass2_eligibility"] == {"cross_fit": "nested_t1",
                                                                "full_train": "nested_t1"},
               lambda: notes.get(E.ARM_RF))
    return notes


def stage2_reuse(gate: Gate, split: dict) -> Dict[str, Any]:
    """Section 5.2 on the final code: the triggers, then the reuse checks."""
    from shared.pinned_artifacts import load_pinned_stage2
    from stage2_crossfit import acceptance as A
    from stage2_crossfit import runner as R
    out: Dict[str, Any] = {}

    def accepted():
        cand = A.require_accepted(PR38_RUN_ID)
        out["acceptance"] = {"candidate_manifest_sha256": cand.manifest_sha256,
                             "glass": cand.relative(cand.glass_path), "rf": cand.relative(cand.rf_path)}
        return True

    gate.check(f"reuse 1: acceptance.require_accepted({PR38_RUN_ID!r}) passes on the final code",
               accepted, lambda: out["acceptance"])
    out["manifest"] = _file_manifest_checks(gate)
    pinned = {}

    def load():
        pinned["s2"] = load_pinned_stage2(split)          # SHA-256, acceptance, Stage 4 contract
        out["stage4_router_contract"] = pinned["s2"].stage4_contract
        return True

    gate.check("trigger 4 / reuse 3: both routers load by pinned path and SHA-256 and pass "
               "the Stage 4 router contract (router_stream + build_stage4_inputs), train and "
               "test side", load, lambda: out.get("stage4_router_contract"))
    if "s2" in pinned:
        s2 = pinned["s2"]
        ts = R.make_train_side(split["X_train"], split["y_train"], split["X_test"])
        out["r4_notes"] = r4_checks(gate, s2.glass_bundle, s2.rf_artifact, ts)
        out["names"] = _names_checks(gate, s2.glass_bundle, s2.rf_artifact)
        for arm, rec in s2.stage4_contract.items():
            gate.check(f"trigger 4: {arm} Pass 2 population is oof_remainder in both fits (F17)",
                       lambda rec=rec: rec["pass2_population"] == {"cross_fit": "oof_remainder",
                                                                   "full_train": "oof_remainder"},
                       rec["pass2_population"])
    out["binning"] = _binning_recheck(gate)
    return out


def decide(preflight: Mapping[str, Any], reuse_gate: Mapping[str, Any]) -> Dict[str, Any]:
    """``reuse`` only if every trigger is clear and every reuse check passed (section 5.2)."""
    failed = list(reuse_gate.get("failed") or [])
    report = (preflight.get("evidence") or {}).get("report_check") or {}
    if report.get("passed") is not True:
        failed.append("reuse 2: report --check (recorded by preflight) did not pass")
    if not reuse_gate.get("checks"):
        failed.append("no reuse check ran")
    if failed:
        return {"outcome": "regenerate", "failed": failed,
                "next": ("STOP. Stage 2 is regenerated only under the registered pathway "
                         "(section 5.2): stage2_crossfit with primary_settings("
                         "'fold_label_excluded'), a new run id, the full PR 38 gate sequence, "
                         "records in research_logs/pr39/<run_id>/ and models in "
                         "models/pr39/<run_id>/ through the F3 root option, PR 38 defaults "
                         "unchanged. That option is not in this patch; report the failed "
                         "checks before anything else runs.")}
    return {"outcome": "reuse", "failed": [],
            "authorization": {"decision": "D3", "approved": "owner, Commit 1 review, 2026-11-14",
                              "record": "documentation/methodology/pr39_execution_contract.md "
                                        "1.0.0 sections 5.2 and 10",
                              "amends": "PR 38 freeze record section 6 ('new run id')"},
            "pins": "shared.pinned_artifacts.STAGE2_PINS (run c3_20261112_153333)",
            "next": "pinned: the notebooks load Stage 2 by path and SHA-256; no Stage 2 fit"}


# ----------------------------------------------------------------------
# A4.3: Stage 1
# ----------------------------------------------------------------------
def _stage_output(payload):
    from shared.stage_io import StageOutput
    so = payload.get("stage_output") if isinstance(payload, Mapping) else None
    if not isinstance(so, Mapping):
        raise ValueError("the artifact holds no stage_output")
    return StageOutput.from_dict(dict(so))          # every StageOutput contract check re-runs


def stage1_checks(gate: Gate, artifacts: Mapping[str, Any], split: dict,
                  population: Mapping[str, Any]) -> Dict[str, Any]:
    """A4.3 for ``{"stage1.lr": payload, "stage1.mlp": payload}``."""
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    from shared.stage1.feature_engineering import (LR_FEATURES, STAGE1_FEATURE_CONTRACT,
                                                   artifact_feature_contract,
                                                   feature_contract_record)
    from shared.stage_io import THRESHOLD_SOURCES, assert_comparable, fold_assignment
    want_contract = feature_contract_record(STAGE1_FEATURE_CONTRACT)
    plan = fold_assignment(split["y_train"], cv_folds=STAGE1_FOLDS, random_state=SEED)
    out, outputs = {}, {}
    for role, arm in (("stage1.lr", "glass"), ("stage1.mlp", "blackbox")):
        payload = artifacts[role]
        so = {}

        def load(payload=payload, so=so):
            so["so"] = _stage_output(payload)
            return True

        gate.check(f"{role}: holds a StageOutput that passes its contract", load)
        if "so" not in so:
            continue
        s = outputs[role] = so["so"]
        gate.check(f"{role}: artifact version 3.0", lambda: payload.get("artifact_version") == "3.0",
                   payload.get("artifact_version"))
        gate.check(f"{role}: Stage 1 feature contract 2.0.0 recorded and consistent",
                   lambda: payload.get("feature_contract") == want_contract
                   and artifact_feature_contract(payload)["version"] == "2.0.0")
        gate.check(f"{role}: ordered feature names = LR_FEATURES",
                   lambda: list(s.feature_names) == list(LR_FEATURES), list(s.feature_names))
        gate.check(f"{role}: stage1 / {arm}", lambda: (s.stage, s.arm) == ("stage1", arm),
                   (s.stage, s.arm))
        gate.check(f"{role}: calibration family {STAGE1_FAMILY} requested and selected (D1)",
                   lambda: (s.calibration_requested, s.calibration_method) == (STAGE1_FAMILY,) * 2,
                   (s.calibration_requested, s.calibration_method))
        gate.check(f"{role}: OOF provenance is the nested cross-fit (F16/F19)",
                   lambda: "nested cross-fit" in str(s.oof_provenance), str(s.oof_provenance)[:120])
        gate.check(f"{role}: Stage 1 fold plan = {STAGE1_FOLDS} stratified folds, seed {SEED}, "
                   "on GLOBAL_SPLIT's training labels",
                   lambda: s.train_fold_id.index.equals(plan.index) and np.array_equal(
                       s.train_fold_id.to_numpy().astype(int), plan.to_numpy().astype(int)),
                   fold_fingerprint(s.train_fold_id.to_numpy()))
        gate.check(f"{role}: split fingerprint = the frozen protocol's stage I/O split",
                   lambda: s.split_fingerprint == population["stage_io_split_fingerprint"],
                   s.split_fingerprint)
        gate.check(f"{role}: threshold source is a registered one",
                   lambda: s.threshold_source in THRESHOLD_SOURCES, s.threshold_source)
        out[role] = {"artifact_version": payload.get("artifact_version"),
                     "feature_contract": payload.get("feature_contract"),
                     "calibration_method": s.calibration_method,
                     "threshold": float(s.threshold), "threshold_source": s.threshold_source,
                     "threshold_is_cross_fitted": THRESHOLD_SOURCES.get(s.threshold_source),
                     "split_fingerprint": s.split_fingerprint,
                     "fold_fingerprint": fold_fingerprint(s.train_fold_id.to_numpy()),
                     "n_folds": int(s.train_fold_id.nunique()),
                     "oof_provenance": str(s.oof_provenance)}
    if len(outputs) == 2:
        gate.check("stage 1: both arms comparable row for row (assert_comparable: split, features, "
                   "rows, labels, one fold partition)",
                   lambda: assert_comparable(outputs["stage1.lr"], outputs["stage1.mlp"]) is None)
    return out


# ----------------------------------------------------------------------
# A4.4: Stage 3
# ----------------------------------------------------------------------
#: The two notebooks stamp Stage 3 under different registered split-fingerprint
#: schemes (shared.stage4.FINGERPRINT_SCHEMES): GLASS Cell 17 the 12-hex
#: ``stage_io`` one (rows and labels), black-box Cell 30 the 16-hex ``router``
#: one (rows only). Black-box Cell 33 accepts either for the EBM.
STAGE3_SPLIT_IDENTITY = ("stage 3: both arms were fitted on the frozen GLOBAL_SPLIT (each split "
                         "fingerprint is GLOBAL_SPLIT's under a registered scheme, "
                         "check_split_identity; its recorded train/test rows, order and labels "
                         "are GLOBAL_SPLIT's; its canonical stage_io fingerprint is the frozen one)")
STAGE3_PROTOCOL = ("stage 3: one shared protocol, family-specific fields only "
                   "(assert_matched_protocol; split_fingerprint by its canonical identity)")


def _recorded_split(a, role: str) -> dict:
    """An artifact's own recorded rows and labels, in GLOBAL_SPLIT's shape (index-only frames)."""
    import pandas as pd
    tr, te = pd.Index(list(a.index_train)), pd.Index(list(a.index_test))
    for what, y, rows in (("training", a.y_train, tr), ("test", a.y_test, te)):
        if len(y) != len(rows):
            raise AssertionError(f"{role}: its recorded {what} labels ({len(y)}) do not cover its "
                                 f"recorded {what} rows ({len(rows)})")
    return {"X_train": pd.DataFrame(index=tr), "X_test": pd.DataFrame(index=te),
            "y_train": pd.Series(np.asarray(a.y_train), index=tr),
            "y_test": pd.Series(np.asarray(a.y_test), index=te)}


def stage3_split_identity(artifacts: Mapping[str, Any], split: dict,
                          population: Mapping[str, Any], *,
                          extra_fingerprints: Optional[dict] = None) -> Dict[str, Any]:
    """
    Raise unless every Stage 3 artifact was fitted on the frozen ``GLOBAL_SPLIT``.

    Not a comparison of two hashes: ``GLOBAL_SPLIT``'s fingerprints under every
    registered scheme must be the frozen protocol's; each artifact's
    ``split_fingerprint`` must be ``GLOBAL_SPLIT``'s under one of them
    (``check_split_identity``, as Stage 4 checks it); its recorded training and
    test rows (in order) and labels must be ``GLOBAL_SPLIT``'s; and the
    label-bearing ``stage_io`` fingerprint recomputed from those records (the
    canonical identity, whatever scheme the artifact was stamped with) must be
    the frozen one. Returns ``{role: {"recorded", "scheme", "canonical_stage_io"}}``;
    the serialized fingerprints are reported, never rewritten.
    """
    from shared.stage4.contract import check_split_identity, split_reference
    ref = split_reference(split, extra_fingerprints)
    frozen = {"stage_io": population["stage_io_split_fingerprint"],
              "router": population["router_split_fingerprint"]}
    got = {k: ref.fingerprints.get(k) for k in frozen}
    if got != frozen:
        raise AssertionError(f"GLOBAL_SPLIT is not the frozen protocol's split: {got} != {frozen}")
    out = {}
    for role, a in artifacts.items():
        scheme = check_split_identity({}, ref, role, primary=a.split_fingerprint)
        own = split_reference(_recorded_split(a, role), extra_fingerprints)
        for what, mine, theirs in (("training rows", own.index_train, ref.index_train),
                                   ("test rows", own.index_test, ref.index_test)):
            if not mine.equals(theirs):
                raise AssertionError(f"{role}: its recorded {what} are not GLOBAL_SPLIT's "
                                     f"(rows or order differ)")
        for what, mine, theirs in (("training labels", own.y_train, ref.y_train),
                                   ("test labels", own.y_test, ref.y_test)):
            if not np.array_equal(mine, theirs):
                raise AssertionError(f"{role}: its recorded {what} are not GLOBAL_SPLIT's "
                                     f"({int(np.sum(mine != theirs))} differ)")
        if own.fingerprints["stage_io"] != frozen["stage_io"]:
            raise AssertionError(f"{role}: its records' stage_io fingerprint "
                                 f"{own.fingerprints['stage_io']} is not the frozen "
                                 f"{frozen['stage_io']}")
        out[role] = {"recorded": a.split_fingerprint, "scheme": scheme,
                     "canonical_stage_io": own.fingerprints["stage_io"]}
    return out


def stage3_checks(gate: Gate, artifacts: Mapping[str, Any],
                  population: Mapping[str, Any], *, split: dict,
                  extra_fingerprints: Optional[dict] = None) -> Dict[str, Any]:
    """
    A4.4 for ``{"stage3.ebm": Stage3Artifact, "stage3.xgb": Stage3Artifact}`` on
    ``split`` (GLOBAL_SPLIT). ``extra_fingerprints`` is for synthetic splits only (tests).
    """
    from glass_pipeline.glass_router.pipeline import fold_fingerprint
    from shared.stage3.feature_engineering import EBM_FEATURES
    from shared.stage3.protocol import FAMILY_SPECIFIC, assert_matched_protocol
    splits = {population["router_split_fingerprint"], population["stage_io_split_fingerprint"]}
    out = {}
    for role, family, arm in (("stage3.ebm", "ebm", "glass"), ("stage3.xgb", "xgboost", "blackbox")):
        a = artifacts[role]
        gate.check(f"{role}: a Stage3Artifact", lambda a=a: type(a).__name__ == "Stage3Artifact",
                   type(a).__name__)
        if type(a).__name__ != "Stage3Artifact":
            continue
        cal = a.calibration or {}
        folds = np.asarray((a.oof or {}).get("fold_id", []))
        gate.check(f"{role}: model_family / arm = {family} / {arm} (F21, F37)",
                   lambda a=a: (a.model_family, (a.config or {}).get("arm")) == (family, arm),
                   (a.model_family, (a.config or {}).get("arm")))
        gate.check(f"{role}: features = EBM_FEATURES, in order",
                   lambda a=a: list(a.features) == list(EBM_FEATURES))
        gate.check(f"{role}: per-fold feature fit", lambda a=a: a.feature_fit_scope == "per_fold")
        gate.check(f"{role}: split fingerprint is this split's (either scheme)",
                   lambda a=a: a.split_fingerprint in splits, a.split_fingerprint)
        gate.check(f"{role}: OOF fold ids = the shared 5-fold plan "
                   f"{population['fold_fingerprint']}",
                   lambda: fold_fingerprint(folds) == population["fold_fingerprint"],
                   lambda: fold_fingerprint(folds))
        gate.check(f"{role}: OOF outputs recorded (proba, proba_calibrated, fold_id)",
                   lambda a=a: all(k in (a.oof or {}) for k in ("proba", "proba_calibrated", "fold_id")))
        gate.check(f"{role}: calibration gate recorded per fold (F22)",
                   lambda: "applied" in cal and len(cal.get("fold_applied") or []) == len(np.unique(folds)),
                   lambda: {k: cal.get(k) for k in ("applied", "method", "ece_before", "fold_applied")})
        gate.check(f"{role}: operating threshold selected on out-of-fold training predictions",
                   lambda a=a: str(((a.threshold or {}).get("oof") or {}).get("selected_on", ""))
                   .startswith("out-of-fold training predictions"),
                   lambda a=a: ((a.threshold or {}).get("oof") or {}).get("selected_on"))
        out[role] = {"model_family": a.model_family, "arm": (a.config or {}).get("arm"),
                     "features": list(a.features), "split_fingerprint": a.split_fingerprint,
                     "block_fingerprint": a.block_fingerprint,
                     "fold_fingerprint": fold_fingerprint(folds),
                     "calibration": {k: cal.get(k) for k in ("applied", "method", "ece_before",
                                                             "ece_after", "reason")},
                     "threshold": ((a.threshold or {}).get("oof") or {}).get("threshold")}
    if len(out) == 2:
        ebm, xgb = artifacts["stage3.ebm"], artifacts["stage3.xgb"]
        gate.check("stage 3: EBM and XGBoost were fitted on one block (block fingerprint)",
                   lambda: ebm.block_fingerprint == xgb.block_fingerprint and bool(ebm.block_fingerprint),
                   ebm.block_fingerprint)
        identity: Dict[str, Any] = {}

        def same_split():
            identity.update(stage3_split_identity({"stage3.ebm": ebm, "stage3.xgb": xgb}, split,
                                                  population, extra_fingerprints=extra_fingerprints))
            return True

        gate.check(STAGE3_SPLIT_IDENTITY, same_split,
                   lambda: {r: f"{v['recorded']} ({v['scheme']})" for r, v in identity.items()})
        out["split_identity"] = identity

        def matched():
            # Only the raw split_fingerprint may differ, and only once same_split proved both
            # are the frozen split; every other protocol field stays blocking.
            allow = ("split_fingerprint",) if len(identity) == 2 and \
                ebm.split_fingerprint != xgb.split_fingerprint else ()
            out["protocol_allowed"] = sorted(set(allow) | FAMILY_SPECIFIC)
            with _quiet():
                out["protocol_differences"] = [list(map(str, d)) for d in
                                               assert_matched_protocol(ebm, xgb, allow=allow,
                                                                       verbose=False)]
            return True

        gate.check(STAGE3_PROTOCOL, matched)
    return out


# ----------------------------------------------------------------------
# A4.5: Stage 4 inputs, nothing fitted
# ----------------------------------------------------------------------
def stage4_input_checks(gate: Gate, split: dict, paths: Mapping[str, Path], *,
                        extra_fingerprints: Optional[dict] = None) -> Dict[str, Any]:
    """
    Both arbiters' input builders on both upstreams: four ``build_stage4_inputs``
    runs. ``extra_fingerprints`` is for synthetic splits only (tests).
    """
    from blackbox_pipeline.models.meta_xgb.inputs import (build_feature_frames,
                                                          load_blackbox_stage4_inputs)
    from glass_pipeline.glass_arbiter.loader import load_stage_outputs
    up = {"glass": (paths["stage1.lr"], paths["stage2.glass_router"], paths["stage3.ebm"]),
          "blackbox": (paths["stage1.mlp"], paths["stage2.rf_router"], paths["stage3.xgb"])}
    out = {}
    for arbiter, loader in (("glass_arbiter", load_stage_outputs),
                            ("meta_xgb", load_blackbox_stage4_inputs)):
        for arm, files in up.items():
            key = f"{arbiter} <- {arm} upstream"
            held = {}

            def build(loader=loader, files=files, arm=arm, held=held, arbiter=arbiter):
                with _quiet():
                    held["inputs"] = loader(split, *map(str, files), verbose=False, upstream=arm,
                                            extra_fingerprints=extra_fingerprints)
                    if arbiter == "meta_xgb":
                        held["frames"] = build_feature_frames(held["inputs"])
                return True

            gate.check(f"A4.5 {key}: build_stage4_inputs passes, train and test side (inputs "
                       "only, no arbiter fitted)", build)
            if "inputs" in held:
                rows = held["inputs"].validation
                gate.check(f"A4.5 {key}: every recorded validation check passed",
                           lambda rows=rows: rows and all(r.get("passed") for r in rows),
                           f"{len(rows)} checks")
                rec = held["inputs"].upstream or {}
                gate.check(f"A4.5 {key}: the recorded upstream is the {arm} arm",
                           lambda rec=rec, arm=arm: rec.get("arm") == arm, rec.get("combination"))
                if "frames" in held:
                    tr, te, fchecks = held["frames"]
                    for _, _, name, fn, _ in fchecks:
                        gate.check(f"A4.5 {key}: Meta-XGB feature block: {name}", fn)
                out[key] = {"checks": len(rows), "combination": rec.get("combination"),
                            "streams": sorted(held["inputs"].streams)}
    return out


# ----------------------------------------------------------------------
# reload and identities
# ----------------------------------------------------------------------
_RELOAD = r"""
import json, sys, joblib
out = {}
for p in sys.argv[1:]:
    a = joblib.load(p)
    out[p] = {"type": f"{type(a).__module__}.{type(a).__name__}",
              "keys": sorted(map(str, a))[:60] if isinstance(a, dict) else None}
print(json.dumps(out))
"""


def fresh_reload(gate: Gate, paths: Mapping[str, Path]) -> Dict[str, Any]:
    """Every file unpickled in a new interpreter (section 7.4): its type and top-level keys."""
    files = [str(p) for p in paths.values()]
    proc = subprocess.run([sys.executable, "-c", _RELOAD, *files], cwd=BANK_ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    got = {}
    if proc.returncode == 0:
        got = json.loads(proc.stdout.strip().splitlines()[-1])
    gate.check("7.4 every artifact reloads in a fresh interpreter", lambda: proc.returncode == 0
               and set(got) == set(files),
               (proc.stderr.strip().splitlines() or [""])[-1] if proc.returncode else len(got))
    expect = {"stage1.lr": "builtins.dict", "stage1.mlp": "builtins.dict",
              "stage3.ebm": "shared.stage3.artifact.Stage3Artifact",
              "stage3.xgb": "shared.stage3.artifact.Stage3Artifact"}
    for role, want in expect.items():
        if role in paths:
            gate.check(f"7.4 {role} reloads as {want}",
                       lambda role=role, want=want: got[str(paths[role])]["type"] == want,
                       lambda role=role: got.get(str(paths[role]), {}).get("type"))
    return {role: got.get(str(p)) for role, p in paths.items()}


def identities(split: dict, population: Mapping[str, Any]) -> Dict[str, Any]:
    """Section 7.3: population, feature contracts per stage, binning, configuration."""
    import hashlib
    from glass_pipeline.glass_router.pipeline import fold_fingerprint, make_folds
    from shared.stage1.feature_engineering import LR_FEATURES, STAGE1_FEATURE_CONTRACT
    from shared.stage2.binning import RF_FEATURES_BINARY, binning_identity
    from shared.stage3.feature_engineering import EBM_FEATURES
    from shared.stage_io import fold_assignment, label_fingerprint, split_fingerprint
    from blackbox_pipeline.models.rf_router import split_fingerprint as router_fp
    from stage2_crossfit.runner import primary_settings

    def names(xs):
        xs = [str(x) for x in xs]
        return {"ordered": xs, "sha256": hashlib.sha256(json.dumps(xs).encode("utf-8")).hexdigest()}

    y = split["y_train"]
    fold5 = np.full(len(y), -1)
    for k, (_, va) in enumerate(make_folds(y, 5, SEED)):
        fold5[va] = k
    s1 = fold_assignment(y, cv_folds=STAGE1_FOLDS, random_state=SEED).to_numpy()
    settings = primary_settings("fold_label_excluded")
    return {
        "population": {
            "train_rows": int(len(y)), "train_positives": int(y.sum()),
            "test_rows": int(len(split["y_test"])),
            "stage_io_split_fingerprint": split_fingerprint(
                split["X_train"].index, split["X_test"].index, y, split["y_test"]),
            "router_split_fingerprint": router_fp(split["X_train"], split["X_test"]),
            "train_label_fingerprint": label_fingerprint(split["X_train"].index, y),
            "shared_fold_fingerprint": fold_fingerprint(fold5),
            "stage1_fold_plan": {"n_folds": STAGE1_FOLDS, "seed": SEED,
                                 "fingerprint": fold_fingerprint(s1)},
            "frozen_protocol_population": {k: population.get(k) for k in (
                "stage_io_split_fingerprint", "router_split_fingerprint",
                "train_label_fingerprint", "fold_fingerprint")},
        },
        "feature_contracts": {
            "stage1": {"contract": dict(STAGE1_FEATURE_CONTRACT, features=list(
                STAGE1_FEATURE_CONTRACT["features"])), **names(LR_FEATURES)},
            "stage2": names(RF_FEATURES_BINARY), "stage3": names(EBM_FEATURES)},
        "stage2_binning": binning_identity(),
        "configuration": {"stage1_calibration_family": STAGE1_FAMILY,
                          "frozen_records": settings.checks["identities"],
                          "feature_pathway": settings.checks["feature_pathway"]},
    }
