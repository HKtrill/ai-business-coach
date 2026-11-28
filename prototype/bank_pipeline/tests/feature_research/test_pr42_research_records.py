"""
PR 42 Commit 2: the training side of the cascades' split
(``feature_research.training_side``) and the write-once research records
(``feature_research.research_records``).

``draw_training_side`` runs on a synthetic population (the database and the
preprocessor are replaced; the split manager and the boundary reference are
the real ones), so the test can see the test rows the function must not return.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from data.global_splits.split_manager import GlobalSplitManager
from feature_research import research_records as RR
from feature_research import training_side as TS
from tests.parity.test_notebook_protocol_parity import SPLIT_PROTOCOL
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic_bank import TARGET, make_bank_frame


# ----------------------------------------------------------------------
# training side
# ----------------------------------------------------------------------
def _calls(fn) -> dict:
    tree = ast.parse(inspect.getsource(fn).lstrip())
    out: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            out.setdefault(name, []).append(n)
    return out


def test_draw_training_side_runs_the_cascades_split_protocol():
    calls = _calls(TS.draw_training_side)
    (db,) = calls["build_database"]
    assert not db.args and not db.keywords                       # SPLIT_PROTOCOL["database"]
    (pre,) = calls["BankPreprocessor"]
    name, kwargs, method = SPLIT_PROTOCOL["population"]
    assert name == "BankPreprocessor" and method == "fit_transform()"
    assert {k.arg: ast.literal_eval(k.value) for k in pre.keywords} == kwargs and not pre.args
    (fit,) = calls["fit_transform"]
    assert ast.unparse(fit.func.value) == ast.unparse(pre) and not fit.args
    (manager,) = calls["GlobalSplitManager"]
    resolved = {k.arg: getattr(TS, ast.unparse(k.value)) for k in manager.keywords}
    assert ("GlobalSplitManager", resolved) == SPLIT_PROTOCOL["split"]
    (split,) = calls["create_split"]
    assert ast.unparse(split.func.value) == ast.unparse(manager)
    assert [ast.unparse(a) for a in split.args] == ["population"]
    assert {k.arg: ast.unparse(k.value) for k in split.keywords} == {"target_col": "target_col"}
    (target,) = calls["sanity_check_data"]
    assert [ast.unparse(a) for a in target.args] == ["population"]


@pytest.fixture
def synthetic(monkeypatch):
    frame = make_bank_frame(n=400, seed=5)

    class Preprocessor:
        def __init__(self, drop_leaky=True):
            assert drop_leaky is True

        def fit_transform(self):
            return frame.copy()

    monkeypatch.setattr(TS, "build_database", lambda: None)
    monkeypatch.setattr(TS, "BankPreprocessor", Preprocessor)
    monkeypatch.setattr(TS, "_attested_stage_io_fingerprint", lambda root=None: "0123456789ab")
    split = GlobalSplitManager(test_size=0.20, random_state=42).create_split(frame, TARGET)
    return frame, split


def _reachable_pandas(obj, seen=None):
    """Every pandas object and index reachable from ``obj`` through fields and mappings."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return [obj.index]
    if isinstance(obj, pd.Index):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [x for f in dataclasses.fields(obj)
                for x in _reachable_pandas(getattr(obj, f.name), seen)]
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return [x for v in obj.values() for x in _reachable_pandas(v, seen)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _reachable_pandas(v, seen)]
    return []


@pytest.mark.parametrize("indices_only", [False, True])
def test_draw_training_side_returns_the_training_rows_and_no_test_side(synthetic, capsys, indices_only, monkeypatch):
    frame, split = synthetic
    if indices_only:
        monkeypatch.setattr(GlobalSplitManager, "create_split", lambda *a, **kw: pytest.fail("two-sided split"))
    side = TS.draw_training_side(indices_only=indices_only)
    out = capsys.readouterr().out
    pd.testing.assert_frame_equal(side.X_train, split["X_train"])
    pd.testing.assert_series_equal(side.y_train, split["y_train"])
    assert side.target_col == "y"
    test_index = split["X_test"].index
    indexes = _reachable_pandas(side)
    assert indexes and not any(len(ix.intersection(test_index)) for ix in indexes)
    assert not {f.name for f in dataclasses.fields(side.train_reference)} & {
        "index_test", "y_test", "X_test"}
    assert side.train_reference.stage_io_split_fingerprint == "0123456789ab"
    assert side.identities["train_rows"] == len(split["y_train"])
    # the protocol's own output (population and test-split class counts) is not shown
    assert "CREATING GLOBAL TRAIN / TEST SPLIT" not in out and "SANITY CHECKS" not in out
    assert "test  :" not in out and out.startswith("training rows")


def test_the_router_split_fingerprint_hashes_row_ids_only(synthetic):
    _, split = synthetic
    from shared.stage2.diagnostics.boundary import router_split_fingerprint
    side = TS.draw_training_side(verbose=False)
    assert side.train_reference.router_split_fingerprint == router_split_fingerprint(
        split["X_train"].index, split["X_test"].index)


def test_the_attested_stage_io_fingerprint_is_the_accepted_runs():
    assert TS._attested_stage_io_fingerprint() == "d6cddadc3439"


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------
def test_run_ids_follow_the_pr39_convention():
    rid = RR.new_run_id(2)
    assert RR.RUN_ID.match(rid) and rid.startswith("c2_")
    for bad in ("c2_2026", "x2_20261122_101010", "c2_20261122_1010101"):
        with pytest.raises(RR.RecordError):
            RR.check_run_id(bad)


def test_every_code_file_a_record_hashes_exists():
    missing = [f for f in RR.CODE_FILES if not (BANK_ROOT / f).is_file()]
    assert not missing, missing


def _run(tmp_path, run_id="c2_20261122_101010"):
    return RR.start_run(commit=2, seeds={"split": 42}, root=tmp_path, run_id=run_id)


def test_a_record_is_written_once_with_its_sidecar_and_reads_back(tmp_path):
    run = _run(tmp_path)
    assert not run.folder.exists()                       # nothing written before the record
    assert set(run.code) == {"head", "uncommitted_tracked_changes", "files_sha256_lf", "tree"}
    path = RR.write_run_record(run, "feature_inventory", {"a": [1, 2], "b": {"x": np.int64(3)}})
    data = path.read_bytes()
    assert (path.parent / "feature_inventory.json.sha256").read_text().strip() == \
        hashlib.sha256(data).hexdigest()
    record = RR.read_run_record(run.run_id, "feature_inventory", root=tmp_path)
    assert record["b"] == {"x": 3} and record["seeds"] == {"split": 42}
    assert record["run"]["run_id"] == run.run_id
    with pytest.raises(RR.RecordError, match="written once"):
        RR.write_run_record(run, "feature_inventory", {"a": 1})
    with pytest.raises(RR.RecordError, match="used once"):
        _run(tmp_path)                                   # the same run id again
    assert path.read_bytes() == data                     # nothing was replaced


def test_a_changed_record_is_refused(tmp_path):
    run = _run(tmp_path)
    path = RR.write_run_record(run, "feature_inventory", {"a": 1})
    record = json.loads(path.read_text())
    record["a"] = 2
    body = json.dumps(record).encode()
    path.write_bytes(body)
    with pytest.raises(RR.RecordError, match="sidecar"):
        RR.read_run_record(run.run_id, "feature_inventory", root=tmp_path)
    (path.parent / "feature_inventory.json.sha256").write_text(hashlib.sha256(body).hexdigest())
    with pytest.raises(RR.RecordError, match="content_sha256"):
        RR.read_run_record(run.run_id, "feature_inventory", root=tmp_path)


def test_content_digest_ignores_the_run_and_needs_identity_keys_untouched(tmp_path):
    a = RR.write_run_record(_run(tmp_path, "c2_20261122_101010"), "r", {"x": [1.0, 2]})
    b = RR.write_run_record(_run(tmp_path, "c2_20261122_101011"), "r", {"x": [1.0, 2]})
    assert json.loads(a.read_text())["content_sha256"] == json.loads(b.read_text())["content_sha256"]
    with pytest.raises(RR.RecordError, match="identity keys"):
        RR.write_run_record(_run(tmp_path, "c2_20261122_101012"), "r", {"code": 1})
    with pytest.raises(RR.RecordError, match="seeds must be ints"):
        RR.start_run(commit=2, seeds={"s": 1.5}, root=tmp_path, run_id="c2_20261122_101013")
    with pytest.raises(RR.RecordError, match="record name"):
        RR.write_run_record(_run(tmp_path, "c2_20261122_101014"), "Bad Name", {})


def test_a_run_writes_several_named_records_each_once(tmp_path):
    run = RR.start_run(commit=3, seeds={"split": 42}, root=tmp_path, run_id="c3_20261122_101010")
    first = RR.write_run_record(run, "feature_inventory", {"a": 1})
    second = RR.write_run_record(run, "encoding_audit", {"b": [1, 2]})
    assert first.parent == second.parent == run.folder and run.written == ["feature_inventory",
                                                                          "encoding_audit"]
    assert sorted(p.name for p in run.folder.iterdir()) == [
        "encoding_audit.json", "encoding_audit.json.sha256",
        "feature_inventory.json", "feature_inventory.json.sha256"]
    before = second.read_bytes()
    with pytest.raises(RR.RecordError, match="written once"):
        RR.write_run_record(run, "encoding_audit", {"b": 3})
    assert second.read_bytes() == before
    for name in ("feature_inventory", "encoding_audit"):
        assert RR.read_run_record(run.run_id, name, root=tmp_path)["run"]["run_id"] == run.run_id
    other = RR.start_run(commit=3, seeds={"split": 42}, root=tmp_path, run_id="c3_20261122_101011")
    other.folder.mkdir()                                     # a folder this run did not create
    with pytest.raises(RR.RecordError, match="written once"):
        RR.write_run_record(other, "feature_inventory", {"a": 1})


def test_compare_records_locates_differing_sections(tmp_path):
    a = {"run": {"run_id": "x"}, "code": {"head": "1"}, "scope": "s", "duplicates": {"callers": 1, "pairs": 2}}
    b = {"run": {"run_id": "y"}, "code": {"head": "2"}, "scope": "s", "duplicates": {"callers": 3, "pairs": 2},
         "extra": 1}
    rows = {(r["section"], r["part"]): r for r in RR.compare_records(a, b)}
    assert rows[("scope", "")]["same"] and not rows[("duplicates", "")]["same"]
    assert ("duplicates", "callers") in rows and ("duplicates", "pairs") not in rows
    assert rows[("extra", "")]["detail"] == "only in the reference"
    assert not {k for k, _ in rows} & set(RR.VOLATILE_KEYS)


def test_the_commit2_reference_record_reads_back_verified():
    record = RR.read_run_record(*RR.COMMIT2_INVENTORY)
    assert record["run"]["run_id"] == RR.COMMIT2_INVENTORY[0] and record["run"]["commit"] == 2

def test_git_keeps_evidence_bytes_in_current_and_future_runs():
    paths = [f"research_logs/pr{pr}/c2_20261125_111609/frozen_inputs.json{suffix}"
             for pr in (43, 44) for suffix in ("", ".sha256")]
    result = subprocess.run(["git", "check-attr", "-z", "text", "--", *paths],
                            cwd=BANK_ROOT, check=True, capture_output=True)
    assert result.stdout.split(b"\0")[2::3] == [b"unset"] * len(paths)
