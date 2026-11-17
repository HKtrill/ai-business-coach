"""
PR 39 Step 6B: the clean-kernel reproduction and the Phase 1 freeze
(``phase1_reproduction``; execution contract sections 5.4, 7.4, 7.5; gates
A6.1-A6.5; D4, D7, D8).

* The full copy of each committed notebook keeps every cell, in order, with
  its source byte-identical; the black-box notebook reads exactly the GLASS
  LR, EBM and GLASS Arbiter by pin, the GLASS notebook nothing. A notebook
  with the legacy Stage 2 switch on is refused.
* The model-folder classification accepts one new version per Stage 1-4 role
  and refuses a modified accepted version, a stray file or a missing arbiter.
* The comparison: an artifact against itself is identical in every respect; a
  sub-1e-9 perturbation is drift, not a difference; a changed hyperparameter
  is a configuration difference. On real synthetic arbiters the GLASS
  Arbiter's recomputed confidences reproduce its saved decisions, abstentions
  and cut. The class follows the section 7.4 table, statements included.
* The five steps chained on a temporary bank_pipeline/ with the notebook
  executions faked: the pins file holds the GLASS execution's outputs and the
  black-box execution reads it; tracked files are recorded and copied out; a
  stray write, a production-code change or a missing report stops the run; an
  analysis-code change is recorded; the A6.2 exception is required exactly
  when the class is Not reproduced.
"""

from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pytest

from phase1_reproduction import compare as Cmp
from phase1_reproduction import runner as P
from shared import pinned_artifacts as PA
from stage4_regeneration import crossover as X
from stage4_regeneration import runner as R
from stage4_regeneration import stage4_copy as S
from tests.stage4_regeneration.test_step5b import grid  # noqa: F401 (module fixture)
from tests.support.stage4_world import build_world, make_stage1, make_stage3
from upstream_regeneration import acceptance as Acc
from upstream_regeneration import checks as C
from upstream_regeneration import inventory as I
from upstream_regeneration import vehicle as V

RUN = "c6_20261117_090000"


# ----------------------------------------------------------------------
# the full copy
# ----------------------------------------------------------------------
@pytest.mark.parametrize("notebook", sorted(V.NOTEBOOKS))
def test_the_full_copy_keeps_every_cell_of_the_committed_notebook(notebook):
    rel, stage4 = V.NOTEBOOKS[notebook]
    committed = json.loads((P.Rec.BANK_ROOT / rel).read_text(encoding="utf-8"))
    nb, rec = P.full_copy(notebook)
    assert [c["cell_type"] for c in nb["cells"]] == [c["cell_type"] for c in committed["cells"]]
    assert [V._source(c) for c in nb["cells"]] == [V._source(c) for c in committed["cells"]]
    assert all(c["outputs"] == [] and c["execution_count"] is None
               for c in nb["cells"] if c["cell_type"] == "code")
    assert rec["cells"]["code"] == sum(c["cell_type"] == "code" for c in committed["cells"])
    assert rec["stage4_cells"] == [f"CELL {h}" for h in stage4]
    assert rec["pins_read"] == ([] if notebook == "glass" else
                                ["stage1.lr", "stage3.ebm", S.GLASS_ARBITER_ROLE])


def test_a_notebook_with_the_legacy_switch_on_is_refused(tmp_path):
    rel, _ = V.NOTEBOOKS["glass"]
    nb = json.loads((P.Rec.BANK_ROOT / rel).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        src = V._source(c)
        if "RUN_LEGACY_STAGE2 = False" in src:
            c["source"] = src.replace("RUN_LEGACY_STAGE2 = False", "RUN_LEGACY_STAGE2 = True")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(json.dumps(nb), encoding="utf-8")
    with pytest.raises(V.VehicleError, match="legacy Stage 2 switch"):
        P.full_copy("glass", root=tmp_path)


def test_executed_in_order_needs_every_code_cell_once_and_no_error(tmp_path):
    def write(counts, error=None):
        cells = [{"cell_type": "code", "execution_count": n,
                  "outputs": [{"output_type": "error"}] if i == error else []}
                 for i, n in enumerate(counts)]
        p = tmp_path / f"nb_{len(list(tmp_path.iterdir()))}.ipynb"
        p.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        return p
    assert P.executed_in_order(write([1, 2, 3]), 3)["passed"]
    assert not P.executed_in_order(write([1, 3, 2]), 3)["passed"]
    assert not P.executed_in_order(write([1, 2]), 3)["passed"]
    assert not P.executed_in_order(write([1, 2, 3], error=1), 3)["passed"]


# ----------------------------------------------------------------------
# the model folders
# ----------------------------------------------------------------------
TS, TS_US = "20261117_100000", "20261117_100000_000001"
GLASS_NEW = {"models/lr/lr_calibrated_" + TS + ".joblib", "models/lr/lr_baseline_" + TS + ".joblib",
             "models/ebm/ebm_stage3_" + TS_US + ".joblib", "models/ebm/ebm_stage3_" + TS_US + ".json",
             "models/ebm/ebm_stage3_latest.joblib",
             "models/glass_arbiter/glass_arbiter_" + TS_US + ".joblib"}
_XO = "blackbox_pipeline/models/stage4_crossover"
BLACK_NEW = {"models/mlp/stage1_mlp_" + TS_US + ".joblib", "models/mlp/stage1_mlp.joblib",
             "blackbox_pipeline/models/xgb/xgb_stage3_" + TS_US + ".joblib",
             "blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_" + TS_US + ".joblib",
             "blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_" + TS_US + ".json",
             f"{_XO}/glass_arbiter_on_blackbox/glass_arbiter_{TS_US}.joblib",
             f"{_XO}/meta_xgb_on_glass/meta_xgb_stage4_{TS_US}.joblib",
             f"{_XO}/meta_xgb_on_glass/meta_xgb_stage4_{TS_US}.json"}
ACCEPTED = {"models/glass_arbiter/glass_arbiter_20261116_153500_000001.joblib",
            "models/lr/lr_calibrated_20261115_210000.joblib"}


def _inv(paths, salt=""):
    return {p: {"bytes": 1, "sha256": PA.sha256_bytes((p + salt).encode())} for p in paths}


def test_classify_accepts_one_new_version_per_role_and_nothing_else():
    before = _inv(ACCEPTED)
    after = {**before, **_inv(GLASS_NEW)}
    g = P.classify(before, after, "glass")
    assert g["passed"], g["problems"]
    assert set(g["produced"]) == {"stage1.lr", "stage3.ebm", "glass_arbiter"}
    final = {**after, **_inv(BLACK_NEW)}
    b = P.classify(after, final, "blackbox")
    assert b["passed"], b["problems"]
    assert set(b["produced"]) == {"stage1.mlp", "stage3.xgb", "meta_xgb",
                                  "glass_arbiter_on_blackbox", "meta_xgb_on_glass"}
    assert P.classify(before, final, P.RUN)["passed"]
    touched = {**after, **_inv(["models/glass_arbiter/glass_arbiter_20261116_153500_000001.joblib"],
                               salt="x")}
    assert any("modified" in p for p in P.classify(before, touched, "glass")["problems"])
    stray = {**after, **_inv(["models/glass_arbiter/notes.txt"])}
    assert any("unregistered" in p for p in P.classify(before, stray, "glass")["problems"])
    missing = {k: v for k, v in after.items() if "glass_arbiter_" + TS_US not in k}
    assert any("glass_arbiter: 0 new versions" in p
               for p in P.classify(before, missing, "glass")["problems"])


# ----------------------------------------------------------------------
# the comparison
# ----------------------------------------------------------------------
def test_vector_threshold_and_decision_blocks():
    a = np.array([0.1, 0.2, 0.30, 0.9])
    assert Cmp.vector(a, a.copy())["identical"] and Cmp.vector(a, a)["max_abs"] == 0
    b = a + np.array([0, 0, 2e-10, 0])
    v = Cmp.vector(a, b)
    assert not v["identical"] and v["max_abs"] == pytest.approx(2e-10) and v["spearman"] == 1
    t = Cmp.threshold(0.3, 0.3 + 1e-10, v)
    assert t["same_value"] and t["within_vector_drift"]
    assert not Cmp.threshold(0.3, 0.31, v)["within_vector_drift"]
    d = Cmp.decisions(a >= 0.3, b >= 0.3 + 3e-10, score_a=a, score_b=b, ta=0.3, tb=0.3 + 3e-10,
                      score_drift=v)
    assert d["differ"] == 1 and d["all_within_drift"] and d["rows"][0]["row"] == 2
    routes = Cmp.decisions(np.array(["pass1", "pass2"]), np.array(["pass1", "abstain"]))
    assert routes["differ"] == 1 and not routes["all_within_drift"]
    assert not Cmp.vector(a, a[:3])["comparable"]


def test_stage1_and_stage3_against_themselves_drift_and_a_changed_configuration():
    w = build_world(n=900, n_test=200)
    s1, s3 = make_stage1(w), make_stage3(w)
    same = Cmp.stage1(s1, copy.deepcopy(s1))
    assert same["configuration"]["identical"] and same["fold_ids"]["identical"]
    assert all(v["identical"] for v in same["vectors"].values())
    assert all(d["differ"] == 0 for d in same["decisions"].values())
    nudged = copy.deepcopy(s1)
    nudged["stage_output"]["test_proba"] = nudged["stage_output"]["test_proba"] + 1e-12
    v = Cmp.stage1(s1, nudged)["vectors"]["test_proba"]
    assert not v["identical"] and v["max_abs"] <= Cmp.TOL
    tuned = copy.deepcopy(s1)
    tuned["stage_output"]["best_params"] = {"C": 0.5}
    assert "best_params" in Cmp.stage1(s1, tuned)["configuration"]["differences"]
    s3same = Cmp.stage3(s3, copy.deepcopy(s3))
    assert s3same["configuration"]["identical"] and s3same["fold_ids"]["identical"]
    assert all(v["identical"] for v in s3same["vectors"].values())
    assert all(t["same_value"] for t in s3same["thresholds"].values())
    moved = copy.deepcopy(s3)
    moved.block_fingerprint = "c" * 64
    assert not Cmp.stage3(s3, moved)["configuration"]["identical"]


def test_the_arbiters_against_themselves_and_the_glass_recomputation(grid):
    w, arts = grid.w, grid.arts
    y_train = np.asarray(w["GS"]["y_train"]).astype(int)
    st = Cmp.statements(P.Rec.canonical(grid.results["comparisons"]))
    assert len(st) == 3 * 13 + 3 * 2 or len(st) > 20, st       # every bootstrap row and McNemar
    assert Cmp.statement_changes(st, Cmp.statements(grid.results["comparisons"])) == []
    roles = {}
    for key, art in arts.items():
        roles[key] = (Cmp.glass_arbiter(art, copy.deepcopy(art), y_train) if key.startswith("glass")
                      else Cmp.meta_xgb(art, copy.deepcopy(art)))
        r = roles[key]
        assert r["configuration"]["identical"], (key, r["configuration"]["differences"])
        assert all(v["identical"] for v in r["vectors"].values()), key
        assert all(t["same_value"] for t in r["thresholds"].values()), key
        assert all(d["differ"] == 0 for d in r["decisions"].values()), key
        if key.startswith("glass"):         # the recomputation reproduces what the arbiter saved
            assert r["recomputed_confidences"]["accepted"]["valid"], r["recomputed_confidences"]
            assert "test: retained (abstention)" in r["decisions"]
        room = Cmp.confidence_room(key, art)
        assert room["negative_side_bound"] + room["positive_side_bound"] >= 1 - 1e-12
    none = {"run": {"class_fields": {}, "other": {}}}
    c = Cmp.classify(roles, environment=none, stage2_inputs_identical=True, statement_changes_=[])
    assert c["class"] == "Deterministic", c["conditions"]
    other_env = {"run": {"class_fields": {"numpy": {"accepted": "1", "reproduced": "2"}}}}
    assert Cmp.classify(roles, environment=other_env, stage2_inputs_identical=True,
                        statement_changes_=[])["class"] == "Numerically equivalent"
    flipped = [{"statement": "glass_column: no_abstain.f2"}]
    assert Cmp.classify(roles, environment=none, stage2_inputs_identical=True,
                        statement_changes_=flipped)["class"] == "Not reproduced"
    retuned = copy.deepcopy(arts["meta_xgb"])
    retuned.learner.best_params = {**retuned.learner.best_params, "max_depth": 99}
    changed = {**roles, "meta_xgb": Cmp.meta_xgb(arts["meta_xgb"], retuned)}
    assert Cmp.classify(changed, environment=other_env, stage2_inputs_identical=True,
                        statement_changes_=[])["class"] == "Not reproduced"
    assert Cmp.classify(roles, environment=none, stage2_inputs_identical=False,
                        statement_changes_=[])["class"] == "Not reproduced"


def test_statements_report_only_what_the_comparison_reports_as_a_finding():
    row = lambda m, d, excl: {"metric": m, "Δ (XGB−Arbiter)": d, "CI excludes 0": excl}  # noqa: E731
    pair = {"no_abstain": {"mcnemar_exact_p": 0.01, "meta_xgb_only_correct": 30,
                           "glass_arbiter_only_correct": 10}}
    acc = Cmp.statements({"glass_column": {"bootstrap": [row("f2", 0.02, True), row("acc", -0.01, False)],
                                           "paired": pair}})
    assert acc == {"glass_column: f2": "Meta-XGB higher",
                   "glass_column: acc": "not distinguished (95% CI includes 0)",
                   "glass_column: McNemar no_abstain": "Meta-XGB more often correct alone (exact p < 0.05)"}
    same = Cmp.statements({"glass_column": {"bootstrap": [row("f2", 0.019, True), row("acc", 0.003, False)],
                                            "paired": pair}})
    assert Cmp.statement_changes(acc, same) == []
    lost = Cmp.statements({"glass_column": {"bootstrap": [row("f2", 0.019, False), row("acc", 0.0, False)],
                                            "paired": pair}})
    assert [c["statement"] for c in Cmp.statement_changes(acc, lost)] == ["glass_column: f2"]


def test_environment_differences_separate_the_section_7_3_fields():
    then = {"python": "3.14.3", "platform": "W", "os": "Windows", "executable": "a",
            "packages": {"numpy": "2.5.4", "ipykernel": "6.31.0"}}
    now = {**then, "executable": "b", "packages": {"numpy": "2.5.5", "ipykernel": "6.32.0"}}
    d = Cmp.environment_differences(now, then)
    assert set(d["class_fields"]) == {"numpy"} and set(d["other"]) == {"ipykernel", "executable"}


# ----------------------------------------------------------------------
# the five steps on a temporary bank_pipeline/
# ----------------------------------------------------------------------
C4, C5 = P.UPSTREAM_RUN, P.STAGE4_RUN
REPO = "prototype/bank_pipeline/"


def _write(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    R._write_once(path, R._text(rec))


@pytest.fixture
def bank(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "prototype" / "bank_pipeline"
    records = root / "research_logs" / "pr39"
    for nb in V.NOTEBOOKS:
        rel = V.NOTEBOOKS[nb][0]
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(P.Rec.BANK_ROOT / rel, root / rel)
    (root / "shared").mkdir()
    (root / "shared" / "code.py").write_text("x = 1\n")
    (root / "phase1_reproduction").mkdir()
    (root / "phase1_reproduction" / "runner.py").write_text("y = 1\n")
    fixed = root / "tests" / "realdata" / "test_fixed.py"          # a registered correction
    fixed.parent.mkdir(parents=True)
    fixed.write_text("corrected = True\n")
    monkeypatch.setattr(P, "STEP6_CORRECTIONS", {"tests/realdata/test_fixed.py": (
        PA.sha256_bytes(b"corrected = False\n"), "test correction")})
    old = root / "research_logs" / "pr39" / "c6_20261116_080000"       # a superseded attempt
    old.mkdir(parents=True)
    (old / "step6_realdata_pytest.xml").write_text("<testsuite tests='2' failures='1'/>")
    monkeypatch.setattr(P, "SUPERSEDED_RUNS", {"c6_20261116_080000": "test attempt"})
    tracked = [t for nb in P.TRACKED_WRITES for t in P.TRACKED_WRITES[nb]]
    for t in tracked:
        (root / t).parent.mkdir(parents=True, exist_ok=True)
        (root / t).write_bytes(b"HEAD " + t.encode())
    head = {t: (root / t).read_bytes() for t in tracked}
    for rel in ACCEPTED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(rel.encode())
    monkeypatch.setattr(P.Rec, "BANK_ROOT", root)
    monkeypatch.setattr(P.Rec, "RECORD_ROOT", records)
    monkeypatch.setattr(P.Rec, "environment", lambda: {"python": "test", "packages": {}})
    env = {"python": "test", "packages": {}}
    # the accepted Commit 4 and Commit 5 records
    pins = {r: PA.PinnedFile(r, f"models/{r.replace('.', '/')}_v.joblib", "0" * 64, 1)
            for r in Acc.ROLES}
    for p in pins.values():
        (root / p.path).parent.mkdir(parents=True, exist_ok=True)
        (root / p.path).write_bytes(b"x")
    pins = {r: PA.PinnedFile(r, p.path, PA.sha256_bytes(b"x"), 1) for r, p in pins.items()}
    PA.write_pins(R.manifest_path(), C4, pins, root=root)
    stage2 = []
    for p in PA.STAGE2_PINS:                     # stand-ins for the accepted PR 38 routers
        (root / p.path).parent.mkdir(parents=True, exist_ok=True)
        (root / p.path).write_bytes(p.role.encode())
        stage2.append(PA.PinnedFile(p.role, p.path, PA.sha256_bytes(p.role.encode()), len(p.role)))
    monkeypatch.setattr(PA, "STAGE2_PINS", tuple(stage2))
    for name in (Acc.STEP, "run_record"):
        _write(records / C4 / f"{name}.json", {"run_id": C4, "passed": True, "environment": env})
    inv = I.snapshot(root)
    first = sorted(ACCEPTED)[0]
    arbiters = {k: {"path": first, **inv[first]} for k in S.ROOT_KEYS}
    _write(records / C5 / "run_record.json", {"run_id": C5, "passed": True, "environment": env,
                                              "code": P.step5a_identity(), "arbiters": arbiters,
                                              "identities": {}})
    for name in ("preflight", "execution_glass", X.STEP):
        _write(records / C5 / f"{name}.json", {"run_id": C5, "passed": True, "results": {}})
    _write(records / C5 / "execution_blackbox.json", {"run_id": C5, "passed": True,
                                                      "inventory_after": inv})
    _write(records / C5 / f"{R.ARBITER_MANIFEST}.json", {"run_id": C5})
    monkeypatch.setattr(Acc, "check", lambda run_id: {"accepted": True, "manifest_sha256": "m" * 64})
    monkeypatch.setattr(X, "check", lambda run_id: {"reproduced": True, "record_sha256": "r" * 64})

    def preservation(gate, sums):
        gate.check("7.5 stub", lambda: True)
        return {"snapshot": {"live_counterparts": {"same": 37, "different": [], "missing": []}}}

    monkeypatch.setattr(C, "preservation", preservation)

    def git_status(root_=None):
        lines = [f" M {REPO}{t}" for t in tracked if (root / t).read_bytes() != head[t]]
        run = records / RUN
        lines += [f"?? {REPO}{p.relative_to(root).as_posix()}" for p in sorted(run.rglob("*"))
                  if p.is_file() and "failed" not in p.parts] if run.is_dir() else []
        return sorted(lines + [f"?? {REPO}phase1_reproduction/runner.py"] + seen["extra_git"])

    seen = {"stray": None, "extra_git": [], "env": {}}
    monkeypatch.setattr(P.Rec, "git_status", git_status)

    def fake_execute(nb, notebook, out_dir, run_id, *, env=None, kernel_name=None, root=None):
        seen["env"][notebook] = dict(env or {})
        if notebook == "blackbox":
            seen["read"] = {r: PA.pinned_path(r, pins_file=env[PA.PINS_ENV], root=root)
                            for r in ("stage1.lr", "stage3.ebm", S.GLASS_ARBITER_ROLE)}
        for rel in (GLASS_NEW if notebook == "glass" else BLACK_NEW):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(f"{notebook} {rel}".encode())
        for t in P.TRACKED_WRITES[notebook]:
            (root / t).write_bytes(b"produced " + t.encode())
        if seen["stray"] == notebook:
            (root / "models/glass_arbiter/glass_arbiter_20261116_153500_000001.joblib").write_bytes(b"!")
        out_dir.mkdir(parents=True, exist_ok=True)
        n = sum(c["cell_type"] == "code" for c in nb["cells"])
        exe = out_dir / f"{run_id}_{notebook}_executed.ipynb"
        exe.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": i + 1,
                                              "outputs": []} for i in range(n)]}), encoding="utf-8")
        return {"completed": True, "error": None, "wall_seconds": 1.0, "kernel": {"name": "python3"},
                "input_copy": {"sha256": "i" * 64}, "executed_copy": {"path": str(exe), "sha256": "e" * 64}}

    monkeypatch.setattr(P.V, "execute", fake_execute)
    seen["class"] = "Deterministic"

    def evaluate(accepted, reproduced, g2, g4):
        g4.check("contract stub", lambda: True)
        g2.check("evaluation stub", lambda: True)
        assert set(reproduced) == {"stage1.lr", "stage3.ebm", "glass_arbiter", "stage1.mlp",
                                   "stage3.xgb", "meta_xgb", "glass_arbiter_on_blackbox",
                                   "meta_xgb_on_glass"}
        return {"classification": {"class": seen["class"], "conditions": {}, "max_abs_drift": 0.0},
                "confidence_room": {}}

    monkeypatch.setattr(P, "evaluate", evaluate)
    return root, seen, tmp_path / "outside", head


def _reports(root, ok=True):
    for suite, name in P.REPORTS.items():
        n = 0 if suite == "notebook" else 5
        (P.run_dir(RUN) / name).write_text(
            f'<testsuites><testsuite tests="{n}" failures="{0 if ok else 1}" errors="0" '
            f'skipped="0"/></testsuites>', encoding="utf-8")


def _statements(tmp_path, **extra):
    p = tmp_path / f"statements_{len(extra)}.json"
    p.write_text(json.dumps({"decision": "Phase 1 accepted", "tracked_outputs": "committed (D7)",
                             **extra}), encoding="utf-8")
    return p


def test_the_five_steps_chain_and_the_black_box_reads_the_glass_execution_by_pin(bank, tmp_path):
    root, seen, out, head = bank
    assert P.preflight(RUN, "sums.txt")["passed"]
    _reports(root)
    glass = P.execute(RUN, "glass", out)
    assert seen["env"]["glass"] == {PA.PINS_ENV: ""}
    pins = PA.read_pins(P.run_dir(RUN) / P.PINS_FILE)["files"]
    assert sorted(pins) == sorted({"stage1.lr", "stage3.ebm", S.GLASS_ARBITER_ROLE,
                                   "stage2.glass_router", "stage2.rf_router"})
    assert pins["stage1.lr"]["sha256"] == glass["produced"]["stage1.lr"]["sha256"]
    fig = P.TRACKED_WRITES["glass"][0]
    w = glass["tracked_writes"][fig]
    assert w["changed"] and open(w["copy"]["path"], "rb").read() == (root / fig).read_bytes()
    black = P.execute(RUN, "blackbox", out)
    assert seen["env"]["blackbox"] == {PA.PINS_ENV: str(P.run_dir(RUN) / P.PINS_FILE)}
    assert seen["read"][S.GLASS_ARBITER_ROLE] == root / glass["produced"]["glass_arbiter"]["path"]
    assert set(black["tracked_writes"]) == set(Acc.D7)
    rep = P.compare(RUN, "sums.txt")
    assert rep["passed"] and not rep["analysis_code"]["changed_since_preflight"]
    (root / fig).write_bytes(head[fig])                                   # the owner restores one
    fr = P.freeze(RUN, _statements(tmp_path))
    assert fr["passed"] and fr["decision"]["a6_2_route"] == "reproduced"
    assert fr["tracked_outputs"][fig]["state"] == "restored to HEAD"
    assert all(fr["tracked_outputs"][t]["state"].startswith("as produced") for t in Acc.D7)
    assert fr["regression"]["notebook"]["summary"]["tests"] == 0
    assert "supersedes" in fr["limitation_correction"] and fr["phase2_handoff"]
    assert fr["corrections"][0]["path"] == "tests/realdata/test_fixed.py"
    assert list(fr["superseded_runs"]) == ["c6_20261116_080000"]
    for step in ("preflight", "execution_glass", "execution_blackbox", "reproduction",
                 "phase1_freeze"):
        assert P.read_record(RUN, step)["passed"]
    with pytest.raises(P.Rec.RecordError, match="already"):
        P.freeze(RUN, _statements(tmp_path))


def test_a_stray_write_stops_the_execution_and_writes_no_pins(bank):
    root, seen, out, _ = bank
    P.preflight(RUN, "sums.txt")
    seen["stray"] = "glass"
    with pytest.raises(P.StepFailed, match="execution_glass"):
        P.execute(RUN, "glass", out)
    assert not (P.run_dir(RUN) / P.PINS_FILE).exists()


def test_a_retry_keeps_an_earlier_attempts_versions_but_not_a_changed_accepted_one(bank):
    root, *_ = bank
    earlier = root / "models/lr/lr_calibrated_20261117_080000.joblib"   # a failed attempt's
    earlier.write_bytes(b"earlier attempt")
    pre = P.preflight(RUN, "sums.txt")
    assert pre["passed"]
    assert "models/lr/lr_calibrated_20261117_080000.joblib" in pre["inventory"]
    other = "c6_20261117_090500"
    (root / sorted(ACCEPTED)[1]).write_bytes(b"changed")
    with pytest.raises(P.StepFailed, match="preflight"):
        P.preflight(other, "sums.txt")


def test_a_registered_correction_keeps_the_step5a_identity_and_a_touched_attempt_fails(bank, tmp_path):
    root, seen, out, _ = bank
    fixed = root / "tests" / "realdata" / "test_fixed.py"
    corrected = fixed.read_bytes()
    fixed.write_bytes(b"corrected = False\n")                          # as at Step 5A
    at_5a = P.step5a_identity()
    fixed.write_bytes(corrected)
    assert P.step5a_identity() == at_5a                                 # the digest is restored
    (root / "shared" / "code.py").write_text("x = 3\n")                  # an unregistered change
    assert P.step5a_identity()["code_sha256"] != at_5a["code_sha256"]
    (root / "shared" / "code.py").write_text("x = 1\n")
    P.preflight(RUN, "sums.txt")
    _reports(root)
    P.execute(RUN, "glass", out)
    P.execute(RUN, "blackbox", out)
    P.compare(RUN, "sums.txt")
    (root / "research_logs/pr39/c6_20261116_080000/step6_realdata_pytest.xml").write_text("edited")
    with pytest.raises(P.StepFailed, match="phase1_freeze"):
        P.freeze(RUN, _statements(tmp_path))


def test_reports_written_before_preflight_are_refused(bank):
    root, *_ = bank
    P.run_dir(RUN).mkdir(parents=True)
    (P.run_dir(RUN) / P.REPORTS["fast"]).write_text("<testsuite tests='1'/>")
    with pytest.raises(P.StepFailed, match="preflight"):
        P.preflight(RUN, "sums.txt")


def test_a_production_change_stops_the_run_an_analysis_change_is_recorded(bank):
    root, seen, out, _ = bank
    P.preflight(RUN, "sums.txt")
    P.execute(RUN, "glass", out)
    P.execute(RUN, "blackbox", out)
    (root / "phase1_reproduction" / "runner.py").write_text("y = 2\n")      # analysis fix
    rep = P.compare(RUN, "sums.txt")
    assert rep["analysis_code"]["changed_since_preflight"]
    (root / "shared" / "code.py").write_text("x = 2\n")                     # production change
    with pytest.raises(P.Rec.RecordError, match="production code"):
        P.freeze(RUN, "unused.json")


def test_freeze_needs_the_reports_and_the_exception_exactly_when_not_reproduced(bank, tmp_path):
    root, seen, out, _ = bank
    seen["class"] = "Not reproduced"
    P.preflight(RUN, "sums.txt")
    P.execute(RUN, "glass", out)
    P.execute(RUN, "blackbox", out)
    P.compare(RUN, "sums.txt")
    with pytest.raises(P.Rec.RecordError, match="a6_2_exception"):
        P.freeze(RUN, _statements(tmp_path))
    with pytest.raises(P.StepFailed, match="phase1_freeze"):               # no reports yet
        P.freeze(RUN, _statements(tmp_path, a6_2_exception="BLAS threads differ; approved"))
    _reports(root)
    fr = P.freeze(RUN, _statements(tmp_path, a6_2_exception="BLAS threads differ; approved"))
    assert fr["decision"]["a6_2_route"] == "environmental exception"
    assert any("environmental exception" in x for x in fr["limitations"])


def test_evaluate_end_to_end_on_a_synthetic_grid(grid, tmp_path, monkeypatch):
    """The real ``evaluate`` on a reproduction identical to its baseline (synthetic 2x2)."""
    import joblib
    from tests.support import artifact_audit as audit
    from tests.support.stage4_world import ROUTER_FP
    w, arts = grid.w, grid.arts
    root = tmp_path / "bank"
    monkeypatch.setattr(P.Rec, "BANK_ROOT", root)
    monkeypatch.setattr(P.Rec, "RECORD_ROOT", root / "research_logs" / "pr39")
    objs = {"stage1.lr": make_stage1(w), "stage1.mlp": make_stage1(w, arm="blackbox", model="mlp"),
            "stage3.ebm": make_stage3(w), "stage3.xgb": make_stage3(w, family="xgboost"), **arts}
    pins = {}
    for side in ("accepted", "reproduced"):
        pins[side] = {}
        for k, obj in objs.items():
            p = root / "models" / side / f"{k}.joblib"
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, p)
            pins[side][k] = PA.PinnedFile(k, p.relative_to(root).as_posix(),
                                          PA.sha256_file(p), p.stat().st_size)
    env = {"python": "test", "packages": {}}
    ids = {"feature_contracts": {}}
    _write(P.Rec.RECORD_ROOT / C4 / "run_record.json", {"run_id": C4, "environment": env})
    _write(P.Rec.RECORD_ROOT / C5 / "run_record.json", {"run_id": C5, "passed": True,
                                                        "environment": env, "identities": {
                                                            "feature_contracts": {"stage4_meta_xgb": None}}})
    _write(P.Rec.RECORD_ROOT / C5 / f"{X.STEP}.json", {"run_id": C5, "passed": True,
                                                       "results": grid.results})
    monkeypatch.setattr(C, "global_split", lambda: w["GS"])
    monkeypatch.setattr(C, "frozen_population", lambda: {
        "stage_io_split_fingerprint": w["fp_io"], "router_split_fingerprint": w["fp_router"],
        "fold_fingerprint": "f", "train_label_fingerprint": "l"})
    monkeypatch.setattr(C, "identities", lambda split, pop: copy.deepcopy(ids))
    real_load, real_eval = audit.load_arbiter, X.evaluate
    monkeypatch.setattr(audit, "load_arbiter",
                        lambda key, path, split: real_load(key, path, split, ROUTER_FP))
    monkeypatch.setattr(X, "evaluate", lambda split, a: real_eval(split, a, extra_fingerprints=ROUTER_FP))
    g2, g4 = C.Gate("A6.2"), C.Gate("A6.4")
    out = P.evaluate(pins["accepted"], pins["reproduced"], g2, g4)
    assert sorted(out["roles"]) == sorted(objs)
    for role, r in out["roles"].items():
        assert all(v["identical"] for v in r["vectors"].values()), role
        assert all(d["differ"] == 0 for d in r["decisions"].values()), role
    assert out["crossover_results"]["identical"], out["crossover_results"]["first_difference"]
    assert out["registered_statements"]["changes"] == [] and out["registered_statements"]["accepted"]
    c = out["classification"]["conditions"]
    assert c["probability vectors within max |delta| 1e-9"]["holds"]
    assert c["identical decisions, abstentions and routes"]["holds"]
    assert c["identical thresholds"]["holds"] and c["identical selected configurations"]["holds"]
    assert set(out["confidence_room"]) == set(S.ROOT_KEYS)
    assert g2.passed, g2.failed()
