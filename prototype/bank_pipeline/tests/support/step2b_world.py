"""
Step 2B synthetic evidence: the records a valid authorization names, built
once by the real code paths (``fit_fold``, ``preflight_checks``,
``audit_fold``, ``settings_record``) on the PR 38 synthetic train side, with
the scripted GLASS selector and the small RF. ``authorization_files`` writes
them, a passing JUnit report, a passed Phase 0 record and an authorization
into a run directory (its name is the run id), and the original G13 audit
into its own run directory beside it, as Step 2A left it (no ``run_id``).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
from pathlib import Path

import pytest

import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.evidence as E
import stage2_crossfit.g13 as G
import stage2_crossfit.inner_features as IF
import stage2_crossfit.runner as R
from tests.support import pr38_world as W
from tests.support.rf_router_world import serial_forests

REVIEW = {"reviewer": "owner", "rationale": "reviewed", "date": "2026-11-12"}
F31_REVIEW_ID = "F31:blackbox:fold0:triggered"


def _slim(ev):
    out = dict(ev)
    if isinstance(out.get("remainder"), dict):
        out["remainder"] = {k: v for k, v in out["remainder"].items() if k != "index"}
    return out


def _population(ts):
    """``check_population``'s record, pinned to the synthetic train side itself."""
    got = R.check_population(ts, ep.load_frozen_protocol())["checks"]
    return {"passed": True, "checks": {k: {"got": v["got"], "pinned": v["got"], "passed": True}
                                       for k, v in got.items()}}


def _fold_record(ts, settings, fmt, **extra):
    res = R.fit_fold(ts, settings, 0, keep_fits=True)
    fits = res.pop("fits")
    return res, fits, {"format": fmt, "fold": 0, "population": _population(ts),
                       "feature_pathway": IF.pathway_identity(settings.feature_pathway),
                       "arms": {a: _slim(ev) for a, ev in res["arms"].items()}, **extra}


def _g13_record(ts, settings):
    _, fits, rec = _fold_record(ts, settings, R.G13_RECORD_FORMAT)
    X_tr, y_tr, _ = ts.fold_rows(0)
    g, (router, rf_bins, _) = fits[E.ARM_GLASS], fits[E.ARM_RF]
    rec["g13"] = G.audit_fold(X_tr, y_tr, fold=0, glass=g.glass, glass_bins=g.X_tr_eng,
                              rf_router=router, rf_bins=rf_bins)
    return rec


@functools.lru_cache(maxsize=None)
def _records() -> str:
    mp = pytest.MonkeyPatch()
    W.wired(mp, W.wired_glass_class())
    try:
        with serial_forests():
            ts = W.synthetic_train_side()
            outer = W.make_settings()
            fold = dataclasses.replace(outer, feature_pathway=IF.PATHWAY_FOLD)
            original = _g13_record(ts, outer)
            corrected = _g13_record(ts, fold)
            res, _, g3 = _fold_record(ts, fold, R.G3_RECORD_FORMAT)
            g3["g3"] = R.preflight_checks(res, 50, protocol=fold.protocol)
    finally:
        mp.undo()
    configs = {"format": R.CONFIGS_RECORD_FORMAT,
               "primary": R.settings_record(R.primary_settings(IF.PATHWAY_FOLD))}
    return json.dumps(E.to_json_safe({"g3": g3, "g13": original, "g13_corrected": corrected,
                                      "configs": configs}))


def records() -> dict:
    """Fresh, editable copies of the valid record bodies (built once)."""
    return copy.deepcopy(json.loads(_records()))


def junit(path: Path, modules=R.AUTHORIZATION_TEST_MODULES, failing=()) -> Path:
    cases = [f'<testcase classname="{m}" name="test_ok" time="0.1"/>' for m in modules]
    cases += [f'<testcase classname="{m}" name="test_bad"><failure message="x"/></testcase>'
              for m in failing]
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
                    f'tests="{len(cases)}">{"".join(cases)}</testsuite></testsuites>',
                    encoding="utf-8")
    return path


def authorization_files(directory: Path, *, bodies=None, report=None, phase0=True,
                        **fields) -> Path:
    """
    A valid authorization (adopting the correction) and its records in the run
    directory ``directory`` (run id: its name); ``fields`` override.
    """
    directory.mkdir(parents=True, exist_ok=True)
    run_id, original_run = directory.name, directory.name + "__s2a"
    original_dir = directory.parent / original_run
    original_dir.mkdir(exist_ok=True)
    bodies = bodies or records()
    for key in ("g3", "g13_corrected", "configs"):
        bodies[key].setdefault("run_id", run_id)
    for d in (directory, original_dir):
        if phase0 and not (d / "phase0.json").exists():
            R.write_record(d / "phase0.json", {"passed": True})
    paths = {key: R.write_record(directory / f"{key}.json", body)
             for key, body in (("g3_record", bodies["g3"]),
                               ("g13_corrected_record", bodies["g13_corrected"]),
                               ("configs_record", bodies["configs"]))}
    paths["g13_record"] = R.write_record(original_dir / f"g13_fold0_{run_id}.json", bodies["g13"])
    paths["test_report"] = report or junit(directory / "pytest.xml")
    reviews = {r: REVIEW for r in bodies["g3"]["g3"]["reviews_required"]}
    rec = {"run_id": run_id, "g13_record_run_id": original_run,
           "gates": {g: "pass" for g in R.AUTHORIZATION_GATES}, "approved_by": "owner",
           "approved_on": "2026-11-12", "reviews": reviews,
           "g13_decision": {**REVIEW, "decision": "material: inner-OOF eligibility moves",
                            "materiality": "material", "adopted_pathway": IF.PATHWAY_FOLD,
                            "adopted_record": IF.pathway_identity(IF.PATHWAY_FOLD)["record"]},
           "f31_r2_clarification": {"record": E.record_identity(E.f31_r2_clarification()),
                                    "approved_by": "owner", "approved_on": "2026-11-12"}}
    for key, p in paths.items():
        rec[key], rec[f"{key}_sha256"] = str(p), R.sha256_file(p)
    rec.update(fields)
    path = directory / "authorization.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path
