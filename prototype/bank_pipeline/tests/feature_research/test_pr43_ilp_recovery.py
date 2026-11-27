"""Synthetic recovery checks. No research artifact or production-model fits."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from feature_research import stage4_holdout as h
from tests.stage2_crossfit.test_pr38_m3_reproduction import _g3, _time_limited_class, W


@pytest.mark.parametrize("fold,unproven,matches", [(0, True, True), (-1, True, False), (1, False, True)])
def test_router_persists_audit_and_reproduces_only_unproven(tmp_path, monkeypatch, fold, unproven, matches):
    ev, _, cfg = _g3(W.synthetic_train_side(), _time_limited_class() if unproven else W.wired_glass_class())
    ev.pop(h.prod.router.E.M3_KEY, None)
    ev.pop("fit_status")
    frame = pd.DataFrame({"x": [0, 1]})
    p = h.prod.Producers.__new__(h.prod.Producers)
    p.arm, p.x, p.y, p.settings = "glass", frame, frame.x, cfg
    p.audit_root, p.audit_binding = tmp_path, "corrected"
    reports = deepcopy(ev["selection_reports"])
    if not matches:
        reports["pass2"]["selected_rule_ids"] = [-99]
    m = SimpleNamespace(training_base_rate=.5, predict=lambda x: (None, np.zeros(len(x)), ["uncertain"] * len(x)),
                        selection_reports=lambda: reports, inner_feature_ranker=None)
    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 2:
            assert (tmp_path / f"router_{fold}" / "primary.json").is_file()
        return SimpleNamespace(glass=m, X_va_eng=frame, X_tr_eng=frame, inner_features=None)

    monkeypatch.setattr(h.prod.router, "fit_glass_fold", fit)
    monkeypatch.setattr(h.prod.router.E, "glass_fold_evidence", lambda *a, **kw: deepcopy(ev))
    monkeypatch.setattr(h.prod, "rules", lambda *args: ([], np.empty((0, len(frame)), bool)))
    invoke = lambda: p.router(fold, [0, 1], [], frame, None)
    if matches:
        _, info = invoke()
        assert info["m3_router_fits"] == int(unproven)
    else:
        with pytest.raises(ValueError, match="requires review"):
            invoke()
        with pytest.raises(FileExistsError):
            invoke()
    path = tmp_path / f"router_{fold}"
    assert json.loads((path / "decision.json").read_text())["status"]["accepted"] == matches
    assert (path / "m3_reports.json").exists() == unproven
    assert len(calls) == 1 + int(unproven)
    if unproven:
        assert all(a is b for a, b in zip(calls[0][0], calls[1][0]))
        assert calls[0][1] == calls[1][1]


def test_historical_stage1_reuse_requires_all_original_archives(tmp_path):
    for k in range(10):
        h.checkpoint(tmp_path, "original", "stage1", k, lambda: ({"x": np.zeros(2)}, {}))
    with pytest.raises(ValueError, match="Missing required"):
        h.stage1_receipts(tmp_path, "original")
    assert not list(tmp_path.glob("stage1_-1_attempt_*"))
    h.checkpoint(tmp_path, "original", "stage1", -1, lambda: ({"x": np.zeros(2)}, {}))
    assert len(h.stage1_receipts(tmp_path, "original")) == 11
    with pytest.raises(ValueError, match="Stale or damaged"):
        h.stage1_receipts(tmp_path, "different")
    (tmp_path / "stage1_2" / "scores.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stale or damaged"):
        h.stage1_receipts(tmp_path, "original")


def test_reconstruct_only_reviewed_changes(tmp_path, monkeypatch):
    from feature_research import holdout_recovery as r
    path = "feature_research/holdout_producers.py"
    dep = tmp_path / "shared" / "producer.py"
    dep.parent.mkdir()
    dep.write_text("old dependency")
    changed = tmp_path / path
    changed.parent.mkdir()
    changed.write_text("former implementation")
    old = h.n.records.code_identity(tmp_path, code_files=[path])
    black = dict(run={"run_id": r.BLACK}, content_sha256=r.BLACK_SHA, code=old)
    changed.write_text("reviewed correction")
    current = h.n.records.code_identity(tmp_path, code_files=[path])
    monkeypatch.setattr(r, "REPLACEMENTS", dict(current["files_sha256_lf"]))
    assert r.original_code(tmp_path, current, black) == (old["files_sha256_lf"], old["tree"])
    changed.write_text("unreviewed correction")
    with pytest.raises(ValueError, match="Unreviewed"):
        r.original_code(tmp_path, h.n.records.code_identity(tmp_path, code_files=[path]), black)
    dep.write_text("unexpected upstream change")
    with pytest.raises(ValueError, match="reconstruct"):
        r.original_code(tmp_path, current, black)


def test_checkpoint_rechecks_solver_audit_on_resume(tmp_path):
    audit = tmp_path / "solver_audits" / "router_0"
    audit.mkdir(parents=True)
    report = audit / "primary.json"
    report.write_text("{}")
    info = dict(audit_sha256={"primary": h.sha256_file(report)})
    h.checkpoint(tmp_path, "corrected", "router", 0, lambda: ({"x": np.zeros(1)}, info))
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Solver audit differs"):
        h.checkpoint(tmp_path, "corrected", "router", 0, None)
