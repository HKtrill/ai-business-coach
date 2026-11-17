"""
PR 39 Step 4A: historical preservation and the run's records
(upstream_regeneration.inventory, .records, .runner; execution contract 7.1, 7.2, 7.5).

* The inventory accepts only a notebook's registered immutable outputs (each
  role exactly once), their sidecars and the documented mutable aliases; any
  other addition, any modification of an existing file and any deletion is a
  problem, named with its protected folder where it has one.
* Records are written once with a SHA-256 sidecar, read back only if the
  sidecar matches, refused for another run; failed attempts go to ``failed/``;
  run ids are ``c4_<YYYYMMDD_HHMMSS>``.
* The run's ``git status`` may gain only its own records and the D7 CSVs; the
  code identity changes with any code edit.
"""

from __future__ import annotations

import json

import pytest

from upstream_regeneration import inventory as I
from upstream_regeneration import records as Rec
from upstream_regeneration import runner as Run

H = {"sha256": "a" * 64, "bytes": 1}


def _with(inv, **changes):
    out = {k: dict(v) for k, v in inv.items()}
    for rel, sha in changes.items():
        out[rel.replace("__", "/")] = {"sha256": sha * 64, "bytes": 1}
    return out


BEFORE = {"models/lr/lr_calibrated_20261023_101010.joblib": H,
          "models/mlp/stage1_mlp.joblib": H,
          "models/mlp/stage1_mlp_threshold_sweep.csv": H,
          "models/mlp/stage1_blackbox_output.joblib": H,
          "models/rf_router/rf_router_two_pass_pass1_sweep.csv": H,
          "models/pr38/c3_20261112_153333/primary/MANIFEST.json": H,
          "models/ebm/ebm_stage3_latest.joblib": H}
GLASS_RUN = {"models/lr/lr_calibrated_20261115_120000.joblib": {"sha256": "b" * 64, "bytes": 2},
             # GLASS Cell 6: the baseline LR, written on every run, never pinned
             "models/lr/lr_baseline_20261115_115900.joblib": {"sha256": "f" * 64, "bytes": 4},
             "models/ebm/ebm_stage3_20261115_130000_000001.joblib": {"sha256": "c" * 64, "bytes": 3},
             "models/ebm/ebm_stage3_20261115_130000_000001.json": {"sha256": "d" * 64, "bytes": 1},
             "models/stage3/stage3_block_d6cddadc3439_0123456789ab.joblib": {"sha256": "e" * 64,
                                                                             "bytes": 9}}


def test_a_glass_execution_adds_its_two_versions_and_replaces_only_its_alias():
    after = {**BEFORE, **GLASS_RUN, "models/ebm/ebm_stage3_latest.joblib": {"sha256": "c" * 64, "bytes": 3}}
    got = I.classify(BEFORE, after, "glass")
    assert got["passed"], got["problems"]
    assert got["produced"] == {"stage1.lr": "models/lr/lr_calibrated_20261115_120000.joblib",
                               "stage3.ebm": "models/ebm/ebm_stage3_20261115_130000_000001.joblib"}
    assert got["mutable_aliases"] == ["models/ebm/ebm_stage3_latest.joblib"]
    assert got["companions"] == {"stage1.lr_baseline": "models/lr/lr_baseline_20261115_115900.joblib"}
    assert len(got["sidecars"]) == 2 and got["diff"] == {"added": 5, "modified": 1, "deleted": 0}


def test_the_owners_refused_glass_run_is_accepted_with_its_baseline():
    """Run c4_20261115_204125: the six changes its failed record lists, now accepted."""
    added = ["models/lr/lr_calibrated_20261115_204501.joblib",
             "models/lr/lr_baseline_20261115_204420.joblib",
             "models/ebm/ebm_stage3_20261116_001529_813890.joblib",
             "models/ebm/ebm_stage3_20261116_001529_813890.json",
             "models/stage3/stage3_block_d6cddadc3439_c950583f1d15.joblib"]
    after = {**BEFORE, **{a: {"sha256": "1" * 64, "bytes": 1} for a in added},
             "models/ebm/ebm_stage3_latest.joblib": {"sha256": "2" * 64, "bytes": 1}}
    got = I.classify(BEFORE, after, "glass")
    assert got["passed"] and got["diff"] == {"added": 5, "modified": 1, "deleted": 0}
    assert got["produced"] == {"stage1.lr": added[0], "stage3.ebm": added[2]}
    assert got["companions"] == {"stage1.lr_baseline": added[1]}


@pytest.mark.parametrize("extra, problem", [
    ({}, "stage1.lr_baseline: 0 new files ([]), expected exactly one"),
    ({"models/lr/lr_baseline_20261115_115901.joblib": "f"}, "stage1.lr_baseline: 2 new files"),
    ({"models/lr/lr_baseline_latest.joblib": "f"}, "unregistered addition models/lr/lr_baseline_latest"),
    ({"models/lr/lr_baseline_20261115_115900.json": "f"}, "unregistered addition"),
    ({"models/rf/lr_baseline_20261115_115900.joblib": "f"}, "(a protected folder)"),
    ({"models/lr/lr_baseline_20261115_1159.joblib": "f"}, "unregistered addition"),
])
def test_the_baseline_is_registered_exactly_once_and_nothing_like_it(extra, problem):
    after = {**BEFORE, **GLASS_RUN}
    if not extra:
        after.pop("models/lr/lr_baseline_20261115_115900.joblib")
    for rel, sha in extra.items():
        after[rel] = {"sha256": sha * 64, "bytes": 1}
    got = I.classify(BEFORE, after, "glass")
    assert not got["passed"] and any(problem in p for p in got["problems"]), got["problems"]


def test_an_existing_baseline_may_not_be_modified_nor_written_by_the_blackbox_notebook():
    old = "models/lr/lr_baseline_20261023_101000.joblib"
    before = {**BEFORE, old: H}
    after = {**before, **GLASS_RUN, old: {"sha256": "9" * 64, "bytes": 1}}
    assert f"existing file modified: {old}" in I.classify(before, after, "glass")["problems"]
    vid = "20261115_140000_123456"
    black = {**BEFORE, f"models/mlp/stage1_mlp_{vid}.joblib": H,
             f"blackbox_pipeline/models/xgb/xgb_stage3_{vid}.joblib": H,
             "models/lr/lr_baseline_20261115_150000.joblib": H}
    assert "unregistered addition models/lr/lr_baseline_20261115_150000.joblib" in \
        I.classify(BEFORE, black, "blackbox")["problems"]


@pytest.mark.parametrize("change, problem", [
    ({"models/mlp/stage1_blackbox_output.joblib": "f"}, "existing file modified: models/mlp/stage1_blackbox_output.joblib"),
    ({"models/rf_router/rf_router_two_pass_pass1_sweep.csv": "f"}, "(a protected folder)"),
    ({"models/pr38/c3_20261112_153333/primary/MANIFEST.json": "f"}, "(a protected folder)"),
    ({"models/glass_router/glass_router_20261115.joblib": "f"}, "unregistered addition models/glass_router"),
    ({"blackbox_pipeline/models/stage4_crossover/glass_arbiter_on_blackbox/glass_arbiter_x.joblib": "f"},
     "(a protected folder)"),
    ({"models/glass_arbiter/glass_arbiter_20261115.joblib": "f"}, "unregistered addition"),
    ({"models/lr/lr_calibrated_20261115_120001.joblib": "f"}, "stage1.lr: 2 new versions"),
])
def test_any_other_change_to_the_model_folders_is_a_problem(change, problem):
    after = {**BEFORE, **GLASS_RUN}
    for rel, sha in change.items():
        after[rel] = {"sha256": sha * 64, "bytes": 1}
    got = I.classify(BEFORE, after, "glass")
    assert not got["passed"] and any(problem in p for p in got["problems"]), got["problems"]


def test_a_deletion_or_a_missing_role_is_a_problem():
    after = {k: v for k, v in {**BEFORE, **GLASS_RUN}.items()
             if k != "models/lr/lr_calibrated_20261023_101010.joblib"}
    got = I.classify(BEFORE, after, "glass")
    assert "existing file deleted: models/lr/lr_calibrated_20261023_101010.joblib" in got["problems"]
    no_ebm = {k: v for k, v in {**BEFORE, **GLASS_RUN}.items() if "ebm_stage3_2026" not in k}
    assert "stage3.ebm: 0 new versions ([]), expected exactly one" in I.classify(BEFORE, no_ebm, "glass")["problems"]


def test_a_blackbox_execution_may_replace_the_mlp_aliases_and_the_d7_csvs():
    vid = "20261115_140000_123456"
    added = {f"models/mlp/stage1_mlp_{vid}.joblib": H, f"models/mlp/stage1_mlp_{vid}_metadata.json": H,
             f"models/mlp/stage1_mlp_{vid}_threshold_sweep.csv": H,
             f"blackbox_pipeline/models/xgb/xgb_stage3_{vid}.joblib": H,
             f"blackbox_pipeline/models/xgb/xgb_stage3_{vid}.json": H,
             "blackbox_pipeline/models/xgb/xgb_stage3_latest.joblib": H}      # alias, new here
    after = {**BEFORE, **added, "models/mlp/stage1_mlp.joblib": {"sha256": "9" * 64, "bytes": 1},
             "models/mlp/stage1_mlp_threshold_sweep.csv": {"sha256": "9" * 64, "bytes": 1}}
    got = I.classify(BEFORE, after, "blackbox")
    assert got["passed"], got["problems"]
    assert set(got["produced"]) == {"stage1.mlp", "stage3.xgb"}
    # a GLASS output during the black-box execution is not the black-box's
    late = {**after, "models/lr/lr_calibrated_20261115_150000.joblib": H}
    assert "unregistered addition models/lr/lr_calibrated_20261115_150000.joblib" in \
        I.classify(BEFORE, late, "blackbox")["problems"]
    # the whole run: both notebooks' outputs, each role once
    run = {**after, **GLASS_RUN}
    assert I.classify(BEFORE, run, I.RUN)["passed"]
    with pytest.raises(ValueError, match="unknown notebook"):
        I.classify(BEFORE, run, "stage4")


def test_the_snapshot_skips_python_sources_and_caches(tmp_path):
    (tmp_path / "models" / "lr").mkdir(parents=True)
    (tmp_path / "models" / "lr" / "lr_calibrated_20261115_120000.joblib").write_bytes(b"x")
    pkg = tmp_path / "blackbox_pipeline" / "models" / "xgb"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "estimator.py").write_text("x = 1\n")
    (pkg / "__pycache__" / "estimator.cpython-313.pyc").write_bytes(b"\0")
    (pkg / "xgb_stage3_latest.joblib").write_bytes(b"y")
    snap = I.snapshot(tmp_path)
    assert sorted(snap) == ["blackbox_pipeline/models/xgb/xgb_stage3_latest.joblib",
                            "models/lr/lr_calibrated_20261115_120000.joblib"]
    assert snap["models/lr/lr_calibrated_20261115_120000.joblib"]["bytes"] == 1


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------
def test_records_are_written_once_with_a_sidecar_and_read_back_verified(tmp_path):
    run = "c4_20261115_120000"
    path = Rec.write_record(run, "preflight", {"run_id": run, "passed": True}, root=tmp_path)
    assert path == tmp_path / run / "preflight.json"
    with pytest.raises(Rec.RecordError, match="written once"):
        Rec.write_record(run, "preflight", {"run_id": run, "passed": True}, root=tmp_path)
    assert Rec.require_passed(run, "preflight", root=tmp_path)["passed"] is True
    path.write_text(path.read_text().replace("true", "false"))
    with pytest.raises(Rec.RecordError, match="does not match its preflight.json.sha256 sidecar"):
        Rec.read_record(run, "preflight", root=tmp_path)
    Rec.write_record(run, "stage2_reuse", {"run_id": "c4_20261115_120001", "passed": True},
                     root=tmp_path)
    with pytest.raises(Rec.RecordError, match="is not run"):
        Rec.read_record(run, "stage2_reuse", root=tmp_path)
    Rec.write_record(run, "execution_glass", {"run_id": run, "passed": False}, root=tmp_path)
    with pytest.raises(Rec.RecordError, match="did not pass"):
        Rec.require_passed(run, "execution_glass", root=tmp_path)
    with pytest.raises(Rec.RecordError, match="no finalize record"):
        Rec.read_record(run, "finalize", root=tmp_path)


def test_a_failed_step_is_recorded_under_failed_and_stops(tmp_path):
    run = "c4_20261115_120000"
    rec = {"run_id": run, "step": "preflight",
           "gates": {"A4.1": {"failed": ["7.5.3 report --check"], "evidence": "3/4"}}}
    with pytest.raises(Run.StepFailed, match=r"(?s)did not pass; recorded in .*failed.*7\.5\.3"):
        Run._finish(run, "preflight", rec, False, root=tmp_path)
    (failed,) = (tmp_path / run / "failed").iterdir()
    assert failed.name.startswith("preflight_") and json.loads(failed.read_text())["passed"] is False
    assert not (tmp_path / run / "preflight.json").exists()


@pytest.mark.parametrize("bad", ["c3_20261115_120000", "c4_2026111_120000", "../c4_20261115_120000",
                                 "c4_20261115_120000/x", None])
def test_run_ids_are_commit_4_run_ids(bad):
    with pytest.raises(Rec.RecordError, match="is not c4_"):
        Rec.check_run_id(bad)
    assert Rec.RUN_ID.match(Rec.new_run_id())


def test_records_never_go_into_the_pr38_folders():
    assert Rec.RECORD_ROOT == Rec.BANK_ROOT / "research_logs" / "pr39"
    assert Rec.run_dir("c4_20261115_120000").parent == Rec.RECORD_ROOT


def test_the_git_status_may_gain_only_the_runs_records_and_the_d7_csvs():
    run = "c4_20261115_120000"
    pre = [" M prototype/bank_pipeline/glass_pipeline/glass_cascade.ipynb"]
    post = pre + [f"?? prototype/bank_pipeline/research_logs/pr39/{run}/preflight.json",
                  " M prototype/bank_pipeline/models/mlp/stage1_mlp_tuning_trials.csv"]
    assert Run._git_delta(pre, post, run)["unexpected"] == []
    bad = post + [" M prototype/bank_pipeline/models/rf_router/rf_router_two_pass_pass1_sweep.csv",
                  "?? prototype/bank_pipeline/research_logs/pr39/c4_20261115_120001/x.json"]
    assert len(Run._git_delta(pre, bad, run)["unexpected"]) == 2
    assert Run._git_delta(pre, [], run)["unexpected"] == pre               # a change undone


def test_the_code_identity_changes_with_any_code_edit(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "a.py").write_bytes(b"x = 1\n")            # exact bytes on every OS
    (tmp_path / "glass_pipeline").mkdir()
    (tmp_path / "glass_pipeline" / "glass_cascade.ipynb").write_text("{}\n")
    one = Rec.code_identity(tmp_path)
    (tmp_path / "shared" / "a.py").write_bytes(b"x = 1\r\n")          # line endings only
    assert Rec.code_identity(tmp_path)["code_sha256"] == one["code_sha256"]
    (tmp_path / "glass_pipeline" / "glass_cascade.ipynb").write_text('{"cells": []}\n')
    assert Rec.code_identity(tmp_path)["code_sha256"] != one["code_sha256"]
    assert one["files"] == 2
