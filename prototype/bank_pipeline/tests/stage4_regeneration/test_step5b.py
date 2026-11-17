"""
PR 39 Step 5B: the registered 2x2 crossover and the baseline record
(``stage4_regeneration.crossover``; execution contract sections 6.4, 6.5, 7.5;
gates A5.4, A5.5; roadmap Commit 5, Stage 4 abstention accounting).

* The accounting reconciles every count, keeps the no-abstention,
  retained-only and whole-system populations apart, and reports a
  zero-denominator or one-class metric as undefined (never 0 or 0.5).
  Effective recall is retained recall x positive retention. The accuracy
  decomposition is exact. A hand-built case reproduces the pattern under
  audit (accuracy down while precision and recall rise) from counts.
* A stored metric on another denominator, or a substituted value, is found.
* On a synthetic 2x2 grid of real arbiter artifacts: every identity holds, and
  each arbiter's stored metrics are the recount. The registered table's "with
  abstention" rows are the retained-only metrics. The columns hold the
  upstream fixed and the cascade does not. The cascade comparison reproduces
  the one Cell 36 saved. Stage 2's tables add up, and the results are
  deterministic.
* Provenance: Step 5A's code identity is re-derived with only this step's
  files omitted. ``record`` writes once, refuses a changed tree, and ``check``
  reproduces.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from glass_pipeline.glass_arbiter import meta_stage as ms
from glass_pipeline.glass_arbiter.evaluation import compute_metrics, evaluate_with_abstention
from glass_pipeline.glass_arbiter.meta_stage import train_glass_arbiter
from stage4_regeneration import crossover as X
from stage4_regeneration import runner as R
from stage4_regeneration import stage4_copy as S
from tests.stage4_regeneration.test_step5a_orchestration import RUN, bank  # noqa: F401 (fixture)
from tests.support.meta_xgb_world import run_meta
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3


# ----------------------------------------------------------------------
# the accounting, on counts chosen by hand
# ----------------------------------------------------------------------
def _hand_case():
    """
    10 positives, 90 negatives. Without abstention: TP 7, FN 3, FP 15, TN 75.
    The arbiter abstains on 40 TN, 5 FP, 1 TP and 2 FN: retained TP 6, FN 1,
    FP 10, TN 35 (52 rows).
    """
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    pred = np.r_[np.ones(7, int), np.zeros(3, int), np.ones(15, int), np.zeros(75, int)]
    keep = np.ones(100, bool)
    keep[[0]] = False                        # 1 TP
    keep[[7, 8]] = False                     # 2 FN
    keep[10:15] = False                      # 5 FP
    keep[25:65] = False                      # 40 TN
    score = np.where(pred == 1, 0.6, 0.2) + np.linspace(0, 0.01, 100) * (y * 2 - 1)
    return y, pred, score, keep


def test_the_accounting_reconciles_every_count_and_keeps_the_populations_apart():
    acc = X.accounting(*_hand_case())
    every, kept, whole = acc["no_abstention"], acc["retained_only"], acc["whole_system"]
    assert every["confusion"] == {"tp": 7, "fp": 15, "tn": 75, "fn": 3}
    assert kept["confusion"] == {"tp": 6, "fp": 10, "tn": 35, "fn": 1}
    assert (every["n"], kept["n"], kept["positives"], kept["negatives"]) == (100, 52, 7, 45)
    assert all(acc["identities"].values()), acc["identities"]
    r = lambda m: (m["num"], m["den"])                                  # noqa: E731
    assert r(every["recall"]) == (7, 10) and r(kept["recall"]) == (6, 7)
    assert r(whole["effective_recall"]) == (6, 10)                   # TP retained / all positives
    assert r(whole["coverage"]) == (52, 100)
    assert r(whole["positive_retention"]) == (7, 10) and r(whole["negative_retention"]) == (45, 90)
    assert r(whole["positive_abstention"]) == (3, 10) and r(whole["negative_abstention"]) == (45, 90)
    assert r(whole["issued_positive_precision"]) == r(kept["precision"]) == (6, 16)
    assert r(whole["correct_positives_share"]) == (6, 100)
    assert set(whole["undefined_without_an_abstention_policy"]) == {
        "accuracy", "f1", "f2", "roc_auc", "pr_auc"}
    assert whole["effective_recall"]["value"] == pytest.approx(
        kept["recall"]["value"] * whole["positive_retention"]["value"], abs=1e-15)
    f = X.findings(acc, train_coverage=0.51, min_coverage=0.50)
    assert f["retained_accuracy_below_all_rows"] and f["retained_precision_above_all_rows"]
    assert f["retained_recall_above_all_rows"] and f["retained_specificity_below_all_rows"]
    assert f["effective_recall_below_no_abstention_recall"]
    assert f["abstention_falls_more_on_negatives"] and not f["test_coverage_below_training_floor"]


def test_the_accuracy_change_splits_exactly_into_mix_recall_and_specificity_terms():
    acc = X.accounting(*_hand_case())
    d = acc["decomposition_all_rows_to_retained"]
    pa, ra, sa = 0.10, 0.7, 75 / 90
    pr, rr, sr = 7 / 52, 6 / 7, 35 / 45
    assert d["accuracy_change"] == pytest.approx(41 / 52 - 82 / 100, abs=1e-15)
    assert d["prevalence_mix_term"] == pytest.approx((pr - pa) * (ra - sa), abs=1e-15)
    assert d["recall_term"] == pytest.approx(pr * (rr - ra), abs=1e-15)
    assert d["specificity_term"] == pytest.approx((1 - pr) * (sr - sa), abs=1e-15)
    assert abs(d["residual"]) < 1e-12 and d["specificity_term"] < 0 < d["recall_term"]
    rng = np.random.default_rng(0)
    for _ in range(50):                                     # the identity holds on any split
        y, p, keep = (rng.random(400) < .2).astype(int), (rng.random(400) < .3).astype(int), \
            rng.random(400) < .6
        out = X.accounting(y, p, rng.random(400), keep)
        assert all(out["identities"].values())


def test_an_undefined_metric_is_none_with_its_reason_never_zero_or_one_half():
    y = np.r_[np.ones(5, int), np.zeros(15, int)]
    pred = np.r_[np.ones(5, int), np.zeros(15, int)]
    keep = np.r_[np.zeros(5, bool), np.ones(15, bool)]        # only true negatives retained
    acc = X.accounting(y, pred, np.linspace(0, 1, 20), keep)
    kept = acc["retained_only"]
    for m in ("precision", "recall", "f1", "f2"):
        assert kept[m]["value"] is None and kept[m]["undefined"] == "zero denominator", m
    for m in ("roc_auc", "pr_auc"):
        assert kept[m] == {"value": None, "undefined": "one class only in this population"}
    assert acc["whole_system"]["effective_recall"]["value"] == 0.0   # 0 of 5 found: defined
    assert acc["decomposition_all_rows_to_retained"]["value"] is None
    assert all(acc["identities"].values())
    # the GLASS Arbiter's own metric code substitutes 0.5 here; the recount refuses it
    stored = {"no_abstain": compute_metrics(y, pred, np.linspace(0, 1, 20)),
              "with_abstain": evaluate_with_abstention(y, np.where(keep, pred, -1),
                                                       np.linspace(0, 1, 20))}
    assert stored["with_abstain"]["metrics"]["roc_auc"] == 0.5
    problems = X.stored_metric_problems("glass_arbiter", stored, acc)
    assert any("with_abstain.roc_auc" in p and "undefined" in p for p in problems)


def test_a_stored_metric_on_another_denominator_is_found():
    y, pred, score, keep = _hand_case()
    acc = X.accounting(y, pred, score, keep)
    stored = {"no_abstain": compute_metrics(y, pred, score),
              "with_abstain": evaluate_with_abstention(y, np.where(keep, pred, -1), score)}
    assert X.stored_metric_problems("glass_arbiter", stored, acc) == []
    stored["with_abstain"]["metrics"]["recall"] = 6 / 10       # effective recall in its place
    assert X.stored_metric_problems("glass_arbiter", stored, acc) == [
        f"with_abstain.recall: stored {6 / 10!r}, recount {6 / 7!r}"]
    stored["with_abstain"]["coverage"] = 1.0
    assert len(X.stored_metric_problems("glass_arbiter", stored, acc)) == 2


# ----------------------------------------------------------------------
# a synthetic 2x2 of real arbiter artifacts
# ----------------------------------------------------------------------
RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    import joblib
    w = build_world(n=1500, n_test=350)
    root = tmp_path_factory.mktemp("step5b_grid")
    up = {"glass": [make_stage1(w), make_router(w, pass2_population="full"), make_stage3(w)],
          "blackbox": [make_stage1(w, arm="blackbox", model="mlp"),
                       make_router(w, layout="extra", band_blend=True,
                                   pass2_population=RF_REMAINDER),
                       make_stage3(w, family="xgboost")]}
    files = {}
    for arm, arts in up.items():
        files[arm] = []
        for i, a in enumerate(arts):
            files[arm].append(str(root / f"{arm}_{i}.joblib"))
            joblib.dump(a, files[arm][-1])
    arts = {}
    with pytest.MonkeyPatch.context() as mp:
        real = ms.load_stage_outputs
        mp.setattr(ms, "load_stage_outputs",
                   lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))
        for key, arm, kw in (("glass_arbiter", "glass", {}),
                             ("glass_arbiter_on_blackbox", "blackbox", {"upstream": "blackbox"})):
            arts[key], path = train_glass_arbiter(w["GS"], *files[arm], verbose=False,
                                                  artifact_dir=str(root / key), **kw)
            if key == "glass_arbiter":
                glass_path = path
    names = ("mlp", "rf", "xgb")
    arts["meta_xgb"], _ = run_meta(w, dict(zip(names, files["blackbox"])), root / "meta",
                                   glass_arbiter_path=glass_path)          # Cell 36's comparison
    arts["meta_xgb_on_glass"], _ = run_meta(w, dict(zip(names, files["glass"])), root / "meta_g",
                                            upstream="glass")
    results = X.evaluate(w["GS"], arts, extra_fingerprints=ROUTER_FP)      # the registered 2,000
    return SimpleNamespace(w=w, arts=arts, results=results)


def test_every_cell_reconciles_and_its_stored_metrics_are_the_recount(grid):
    cells = grid.results["cells"]
    assert sorted(cells) == sorted(S.ROOT_KEYS)
    for key, c in cells.items():
        assert all(c["identities"].values()), (key, c["identities"])
        assert c["retained_decisions_unchanged"], key
        assert c["stored_metric_problems"] == [], (key, c["stored_metric_problems"])
        assert c["upstream"] == S.ROOT_UPSTREAM[key]
        assert c["crossover"] is (key in ("glass_arbiter_on_blackbox", "meta_xgb_on_glass"))
        cal = c["calibration"]
        if key.startswith("meta_xgb"):
            assert cal["applicable"] and X._calibration_problems(cal) == []
        else:
            assert cal["applicable"] is False and "not a calibrated probability" in cal["reason"]


def test_the_registered_with_abstention_rows_are_the_retained_only_metrics(grid):
    for arm, (g, x) in X.COLUMNS.items():
        table = grid.results["comparisons"][f"{arm}_column"]["table"]
        rows = {(r["level_0"], r["level_1"]): r for r in table}
        for key, label in ((g, "GLASS Arbiter"), (x, "Meta-XGB")):
            kept = grid.results["cells"][key]["retained_only"]
            row = rows[(label, "with abstention")]
            for m in ("accuracy", "precision", "recall", "f2"):
                assert row[m] == pytest.approx(kept[m]["value"], abs=5e-7), (key, m)
            assert row["coverage"] == pytest.approx(
                grid.results["cells"][key]["whole_system"]["coverage"]["value"], abs=5e-7)


def test_the_columns_hold_the_upstream_fixed_and_the_cascade_reproduces_cell_36(grid):
    comps = grid.results["comparisons"]
    assert all(comps[f"{arm}_column"]["protocol"]["upstream"]["held_fixed"] for arm in X.COLUMNS)
    assert comps["cascade"]["protocol"]["upstream"]["held_fixed"] is False
    assert comps["glass_column"]["protocol"]["n_boot"] == X.N_BOOT == 2000
    saved = grid.arts["meta_xgb"].comparison["summary"]
    redone = {k: comps["cascade"][k] for k in ("table", "paired", "bootstrap")}
    assert X.Rec.canonical(saved) == X.Rec.canonical(redone)


def test_stage2_tables_add_up_and_each_arbiter_is_split_by_route(grid):
    n = grid.results["population"]["n"]
    for arm, s2 in grid.results["stage2"].items():
        routes = s2["routes"]
        assert sum(r["n"] for r in routes.values()) == n
        assert sum(r["positives"] for r in routes.values()) == grid.results["population"]["positives"]
        sup = s2["pass1_suppression"]
        assert sup["intercepted_negatives"] + sup["suppressed_positives"] == \
            sup["pass1_rows_flagged_by_stage1_or_stage3"]
        c2 = s2["pass2_contribution"]
        assert c2["unique_positives"] + c2["redundant_positives"] == c2["pass2_positives"] == \
            routes["pass2"]["positives"]
        assert sorted(s2["arbiters"]) == sorted(X.COLUMNS[arm])
        for key, a in s2["arbiters"].items():
            kept = grid.results["cells"][key]["retained_only"]["confusion"]
            for m in ("tp", "fp", "tn", "fn"):
                assert sum(r[m] for r in a["by_route"].values()) == kept[m], (key, m)


def test_the_results_are_deterministic(grid):
    again = X.evaluate(grid.w["GS"], grid.arts, n_boot=50, extra_fingerprints=ROUTER_FP)
    twice = X.evaluate(grid.w["GS"], grid.arts, n_boot=50, extra_fingerprints=ROUTER_FP)
    assert X._first_difference(X.Rec.canonical(again), X.Rec.canonical(twice)) is None
    json.dumps(X.Rec.canonical(grid.results), allow_nan=False)


# ----------------------------------------------------------------------
# provenance and the record
# ----------------------------------------------------------------------
def test_step5a_code_identity_omits_only_this_steps_files(tmp_path, monkeypatch):
    root = tmp_path / "bank_pipeline"
    for rel in ("shared/a.py", "stage4_regeneration/runner.py", "tests/stage4_regeneration/t.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    monkeypatch.setattr(R.Rec, "BANK_ROOT", root)
    monkeypatch.setattr(R, "BANK_ROOT", root)
    before = R.code_identity()
    for rel in X.STEP5B_FILES:
        (root / rel).write_text("# Step 5B\n")
    assert R.code_identity() != before
    keys = ("code_sha256", "stage4_regeneration_sha256")
    assert {k: X.step5a_code_identity()[k] for k in keys} == {k: before[k] for k in keys}
    (root / "shared/a.py").write_text("# changed\n")
    assert X.step5a_code_identity()["code_sha256"] != before["code_sha256"]


def _cascade():
    return {"table": [{"level_0": "GLASS Arbiter", "level_1": "no abstention", "accuracy": 0.5}],
            "paired": {"no_abstain": {"n": 4}}, "bootstrap": [{"metric": "no_abstain.f2"}]}


@pytest.fixture
def recordable(bank, monkeypatch):
    """The Step 5A chain on the temporary bank, then Step 5B's data-dependent parts stubbed."""
    from tests.support import artifact_audit as audit
    root, pins, seen, out = bank
    R.preflight(RUN, "sums.txt")
    R.execute(RUN, "glass", out)
    R.execute(RUN, "blackbox", out)
    R.finalize(RUN, "sums.txt")
    monkeypatch.setattr(audit, "load_arbiter", lambda key, path, split: (
        SimpleNamespace(comparison={"summary": _cascade()}) if key == "meta_xgb" else {"key": key}))
    monkeypatch.setattr(audit, "check_arbiter_identity", lambda key, art, generation=None: None)
    monkeypatch.setattr(audit, "check_crossover", lambda arts, paths, split: {})
    monkeypatch.setattr(R, "upstream_resolution", lambda art, arm, manifest: {"stage1": "pinned"})
    ok = {"N_total = N_retained + N_abstained": True}

    def evaluate(split, arts, **kw):
        return {"cells": {k: {"identities": ok, "retained_decisions_unchanged": True,
                              "stored_metric_problems": [],
                              "calibration": {"applicable": False}} for k in S.ROOT_KEYS},
                "comparisons": {**{f"{a}_column": {"protocol": {"upstream": {"held_fixed": True}}}
                                   for a in X.COLUMNS},
                                "cascade": {"protocol": {"upstream": {"held_fixed": False}},
                                            **_cascade()}},
                "stage2": {a: {} for a in X.COLUMNS}}

    monkeypatch.setattr(X, "evaluate", evaluate)
    return root


def test_record_writes_the_baseline_once_and_check_reproduces_it(recordable):
    rec = X.record(RUN, "sums.txt")
    assert rec["passed"] and rec["gates"]["A5.4"]["passed"] and rec["gates"]["A5.5"]["passed"]
    assert rec["references"]["commit4"]["run_id"] == R.UPSTREAM_RUN
    assert sorted(rec["references"]["step5a"]["records"]) == sorted(
        f"{n}.json" for n in X.STEP5A_RECORDS)
    assert sorted(rec["arbiters"]) == sorted(S.ROOT_KEYS)
    assert all(sorted(a["upstream_files"]) == ["stage1", "stage2", "stage3"]
               for a in rec["arbiters"].values())
    assert X.check(RUN)["reproduced"]
    with pytest.raises(X.Rec.RecordError, match="written once"):
        X.record(RUN, "sums.txt")


def test_record_refuses_a_tree_whose_code_changed_since_step_5a(recordable):
    (recordable / "shared" / "code.py").write_text("x = 3\n")
    with pytest.raises(R.StepFailed, match="code identity is re-derived"):
        X.record(RUN, "sums.txt")
    assert not (R.run_dir(RUN) / f"{X.STEP}.json").exists()


def test_record_refuses_a_cascade_comparison_that_does_not_reproduce(recordable, monkeypatch):
    from tests.support import artifact_audit as audit
    other = {**_cascade(), "paired": {"no_abstain": {"n": 5}}}
    monkeypatch.setattr(audit, "load_arbiter", lambda key, path, split: (
        SimpleNamespace(comparison={"summary": other}) if key == "meta_xgb" else {"key": key}))
    with pytest.raises(R.StepFailed, match="reproduces the one black-box Cell 36 saved"):
        X.record(RUN, "sums.txt")
