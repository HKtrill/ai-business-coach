"""
PR 42 Commit 5, Step 5B: procedural EBM interaction selection (feature_research.interaction_selection).

Covers, without the training rows: the registration and the history it cites, read from Step 4A's pinned
exploratory blob even when the working copy's outputs are cleared; the accepted Step 5A record rule; the selector
(interpret's FAST inside the Stage 3 EBM) with its recorder agreeing with the fitted pairs, every eligible pair
ranked, and ties broken by column order; selection on fit rows only (perturbing held-out labels leaves a fold's
selected pairs unchanged, while a planted training-side interaction changes them); the procedural factory through
Step 5A's protocol (fitted-terms check, every outer, inner and all-rows selection matched to its fit frame);
recurrence and change classes; the paired comparison against a 5A fit; the surfaces and interaction value,
including a pair selected by one fold (fold sd unavailable); acceptance (it fails on a misreported verdict or
mismatched rows) and same-sandbox reproduction (required only in the sandbox's environment; elsewhere changed
directions and verdicts are flagged, not failed); and the notebook layout of Cells 60-68.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research import interaction_selection as IS
from feature_research import stage3_simplification as SS
from tests.support import stage2_crossfit_world as W

BANK = Path(__file__).resolve().parents[2]
FAST = {"learning_rate": 0.05, "max_rounds": 60, "max_bins": 32, "max_interaction_bins": 16}
FORBIDDEN = ("shared.stage2.feature_engineering", "feature_research.feature_engineering")


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_no_feature_engineering_imports():
    for path in (BANK / "feature_research/interaction_selection.py", Path(__file__)):
        assert not any(m.startswith(f) for m in _imports(path) for f in FORBIDDEN), path


def test_the_registration_and_the_history_it_cites():
    chk = IS.registration_checks(BANK)
    assert chk["passed"].all(), chk.to_string()
    r = IS.REGISTRATION
    assert r["version"] == "1.0.0" and IS.registration_digest() == IS.REGISTERED_SHA256
    assert "campaign x emp_var_rate_sigmoid_slope is eligible" in r["eligible_pairs"]
    assert "interactions=3" in r["selector"]["implementation"] and "column indices" in r["selector"]["ties"]


# ======================================================================
# the history: Step 4A's pinned exploratory notebook, never the working copy
# ======================================================================
def _pinned_bytes() -> bytes:
    import subprocess
    out = subprocess.run(["git", "cat-file", "blob", IS.BD.EXPLORATORY["blob"]], cwd=BANK, capture_output=True)
    if out.returncode != 0:
        pytest.skip("the pinned exploratory blob is not in this checkout's Git history")
    return out.stdout


def _cleared(data: bytes) -> bytes:
    nb = json.loads(data.decode("utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def test_the_history_comes_from_the_pinned_blob_even_when_the_working_copy_is_cleared(tmp_path):
    data = _pinned_bytes()
    cleared = tmp_path / "exploratory_feature_research.ipynb"
    cleared.write_bytes(_cleared(data))
    intact = IS.historical_provenance(BANK)
    h = IS.historical_provenance(BANK, working_copy=cleared)
    src = h["source"]
    assert (src["status"], src["source"], src["blob"], src["sha256_lf"]) == (
        "verified", "git blob", IS.BD.EXPLORATORY["blob"], IS.BD.EXPLORATORY["sha256_lf"])
    assert src["commit"] == IS.BD.EXPLORATORY["commit"] and src["code_cells_with_output"] > 2
    assert h["statements"] == intact["statements"] and all(r["checked"] and r["verified"] for r in h["statements"])
    by = {r["statement"]: r for r in h["statements"]}
    pairs = next(r for k, r in by.items() if k.startswith("Cell 16B"))
    assert "campaign & emp_var_rate_sigmoid_slope (0.0529)" in pairs["detail"]
    scoped = next(r for k, r in by.items() if k.startswith("'0.205' appears in none"))
    assert "scope: that notebook only" in scoped["statement"] and "Not inspected" in scoped["detail"]
    # the blob really holds the cited outputs: the cleared copy has none
    nb = json.loads(data.decode("utf-8"))
    o16b = IS._outputs_text(IS._one_cell(nb, IS.RESEARCH_ROUND["cells"]["16B"]))
    assert "Pairs" in o16b and not IS._outputs_text(IS._one_cell(json.loads(_cleared(data)), IS.RESEARCH_ROUND["cells"]["16B"]))
    chk = IS.registration_checks(BANK, history=h)
    assert chk["passed"].all() and chk["check"].str.contains("266860bac589").any()


def test_without_git_a_cleared_working_copy_leaves_the_history_unchecked_and_a_pinned_copy_verifies(tmp_path):
    data = _pinned_bytes()
    root = tmp_path / "not_a_repository"
    (root / "glass_pipeline/ebm").mkdir(parents=True)
    (root / "glass_pipeline/ebm/interactions.py").write_bytes((BANK / "glass_pipeline/ebm/interactions.py").read_bytes())
    cleared, crlf = tmp_path / "cleared.ipynb", tmp_path / "crlf.ipynb"
    cleared.write_bytes(_cleared(data))
    crlf.write_bytes(data.replace(b"\n", b"\r\n"))
    h = IS.historical_provenance(root, working_copy=cleared)
    assert h["source"]["status"] == "unavailable" and h["source"]["source"] is None
    notebook = [r for r in h["statements"] if not r["statement"].startswith("glass_pipeline")]
    assert len(notebook) == 4 and not any(r["checked"] for r in notebook) and all(r["verified"] for r in notebook)
    chk = IS.registration_checks(root, history=h)
    assert chk["passed"].all() and chk["check"].str.contains("not checked").sum() == 4
    h2 = IS.historical_provenance(root, working_copy=crlf)          # the pinned notebook, CRLF: still verified
    assert (h2["source"]["status"], h2["source"]["source"]) == ("verified", "working copy")
    assert all(r["checked"] and r["verified"] for r in h2["statements"])


# ======================================================================
# the accepted Step 5A record (Step 3B's rule)
# ======================================================================
def test_the_accepted_5a_record_rule(tmp_path):
    from feature_research import research_records as RR
    run = RR.start_run(commit=5, seeds={"x": 1}, root=tmp_path, run_id="c5_20261124_002645")
    RR.write_run_record(run, "stage3_simplification", {"complementarity": {"views": {"status": {"v": "ok"}}}, "x": 1})
    rec = RR.read_run_record(run.run_id, "stage3_simplification", root=tmp_path)
    acc = {**IS.ACCEPTED_5A, "content_sha256": rec["content_sha256"]}
    views = {"status": {"v": "ok"}, "required": True}
    same = IS.accepted_5a_reproduction(rec, run.environment, views, root=tmp_path, accepted=acc)
    assert bool(same["reproduction_required"].iloc[0]) and IS.accepted_5a_checks(same)["passed"].all()
    other = IS.accepted_5a_reproduction({"content_sha256": "0" * 64}, run.environment, views, root=tmp_path,
                                        accepted=acc)
    assert not IS.accepted_5a_checks(other)["passed"].all()          # required and not reproduced: fails
    moved = IS.accepted_5a_reproduction({"content_sha256": "0" * 64}, {**run.environment, "python": "other"}, views,
                                        root=tmp_path, accepted=acc)
    assert not bool(moved["reproduction_required"].iloc[0]) and IS.accepted_5a_checks(moved)["passed"].all()
    assert "python differs" in moved["reproduction_basis"].iloc[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    absent = IS.accepted_5a_reproduction(rec, run.environment, {"status": {}, "required": False}, root=empty)
    assert IS.accepted_5a_checks(absent)["passed"].all()                # absent, not required: reported
    needed = IS.accepted_5a_reproduction(rec, run.environment, views, root=empty)
    assert not IS.accepted_5a_checks(needed)["passed"].all()            # absent in the acceptance run: fails
    tampered = IS.accepted_5a_reproduction(rec, run.environment, views, root=tmp_path)   # the real accepted digest
    assert not IS.accepted_5a_checks(tampered)["passed"].all()


# ======================================================================
# the selector
# ======================================================================
def _planted(n=3000, seed=0, pair=("f2", "f3"), strength=2.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)], index=np.arange(5_000, 5_000 + n))
    logit = 0.8 * X.f0 - 0.5 * X.f1 + strength * X[pair[0]] * X[pair[1]]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_the_recorder_sees_every_pair_in_every_bag_and_agrees_with_the_fit():
    X, y = _planted()
    s = IS.select_pairs(X, y, params=FAST)
    assert s["bags"] == 14 and s["eligible_per_bag"] == [15]
    assert s["selection_matches_fit"] and len(s["selected"]) == IS.BUDGET
    assert s["selected"][0] == "f2 x f3"                               # the planted pair ranks first
    rk = pd.DataFrame(s["ranking"])
    assert len(rk) == 15 and rk["position"].tolist() == list(range(1, 16))
    assert (rk["mean_rank"].diff().dropna() >= 0).all()                # ordered by mean rank
    assert sorted(s["fitted"]) == sorted(s["selected"])


def test_mean_ranks_follow_interprets_rule_and_ties_go_to_column_order():
    bags = [[(0.9, (0, 1)), (0.5, (1, 2)), (0.1, (0, 2))],
            [(0.8, (1, 2)), (0.7, (0, 1)), (0.2, (0, 2))]]
    r = IS.mean_ranks(bags)
    assert r == {(0, 1): 0.5, (1, 2): 0.5, (0, 2): 2.0}
    import heapq
    assert heapq.nsmallest(2, r, key=lambda i: (r[i], i)) == [(0, 1), (1, 2)]   # tie: column order


def test_held_out_labels_never_reach_a_fold_selection_but_training_signal_does():
    # one fold: the selection sees only the fit rows; perturbing the held-out labels changes nothing, and a
    # planted training-side interaction (the control) changes the selected pairs
    X, y = _planted(n=4000, seed=1, strength=0.0)
    fit, held = np.arange(4000) < 3000, np.arange(4000) >= 3000
    base = IS.select_pairs(X[fit], y[fit], params=FAST)
    yy = y.copy()
    yy[held] = 1 - yy[held]
    again = IS.select_pairs(X[fit], yy[fit], params=FAST)
    assert again["selected"] == base["selected"] and again["ranking"] == base["ranking"]
    planted = y.copy()
    rng = np.random.default_rng(7)
    logit = 3.0 * X["f4"] * X["f5"]
    planted[fit] = (rng.random(fit.sum()) < 1 / (1 + np.exp(-logit[fit]))).astype(int)
    moved = IS.select_pairs(X[fit], planted[fit], params=FAST)
    assert moved["selected"][0] == "f4 x f5" and moved["selected"] != base["selected"]


# ======================================================================
# through Step 5A's protocol on the synthetic training side
# ======================================================================
@pytest.fixture(scope="module")
def world():
    import feature_research.encoding_audit as EA
    mp = pytest.MonkeyPatch()
    mp.setattr(SS, "PARAMS", FAST)
    mp.setattr(SS, "MIN_HISTORY_ROWS", 50)
    mp.setattr(SS, "MIN_HISTORY_POSITIVES", 3)
    ts = W.synthetic_train_side()
    X, y = ts.X_train, ts.y_train.astype(int)
    T = EA.build_time_index(X)
    block = SS.training_block(X, y)
    past = SS.add_past_density(block, X, y, T)
    frames = SS.frames_for_fits(block, past)
    configs_5a = SS.configurations()
    fits_5a = {k: SS.fit_configuration(configs_5a[k], block, frames) for k in ("baseline", "jed_alone")}
    cfg = IS.configurations(configs_5a)
    sel = {"jed_alone_selected": IS.fit_selected(cfg["jed_alone_selected"], block, frames)}
    rec = IS.recovery(block, frames)
    yield dict(X=X, y=y, T=T, block=block, past=past, frames=frames, configs_5a=configs_5a, fits_5a=fits_5a,
               cfg=cfg, sel=sel, rec=rec)
    mp.undo()


def test_every_selection_is_matched_to_its_own_fit_frame(world):
    s = world["sel"]["jed_alone_selected"]["selections"]
    chk = IS.selection_checks("jed_alone_selected", s, 13, expect={"outer": 5, "inner": 25, "all rows": 1})
    assert chk["passed"].all(), chk.to_string()
    lookup = IS.fit_frame_index(world["frames"])
    for sel in s:                                                      # a fold's selection never holds its rows
        if sel["fit"].startswith("outer"):
            k = int(sel["fit"].split()[1])
            held = set(world["frames"]["outer"][k].X_val.index)
            fit_rows = set(world["frames"]["outer"][k].X_tr.index)
            assert sel["rows"] == len(fit_rows) and not (held & fit_rows)
    rc = IS.selection_checks("recovery", world["rec"]["selections"], 13, expect={"outer": 5, "inner": 0, "all rows": 1})
    assert rc["passed"].all(), rc.to_string()
    assert len(lookup) == 1 + 5 + 25


def test_the_procedural_fit_passes_5a_s_terms_check_with_three_pairs(world):
    r = world["sel"]["jed_alone_selected"]
    assert r["terms_ok"] and r["expected_terms"] == list(world["cfg"]["jed_alone_selected"]["features"])
    assert all(sum(" & " in t for t in terms) == IS.BUDGET for terms in r["terms"])
    chk = SS.fit_checks(world["cfg"], world["sel"])
    real_size = chk["check"].str.contains("26,360")                    # the real fold size; synthetic folds differ
    assert chk.loc[~real_size, "passed"].all(), chk.to_string()
    assert r["fit_rows"] == [len(f.X_tr) for f in world["frames"]["outer"]]
    assert SS.isolation_checks(r, world["block"])["passed"].all()
    bad = {"features": ["a", "b"], "budget": 3}
    assert not SS._terms_ok(["a", "b", "a & b"], bad)                  # fewer pairs than the budget fails


def _flipped(frame, rows):
    """A copy of a fold frame with the labels of ``rows`` (index labels) flipped wherever they appear."""
    from copy import copy
    g = copy(frame)
    for X, attr in ((frame.X_tr, "y_tr"), (frame.X_val, "y_val")):
        y = getattr(frame, attr)
        m = np.isin(np.asarray(X.index), rows)
        v = np.asarray(y).copy()
        v[m] = 1 - v[m]
        setattr(g, attr, pd.Series(v, index=y.index) if isinstance(y, pd.Series) else v)
    return g


def _fold_selections(outer, inner, feats):
    fac = IS.ProceduralEBMFactory()
    for f in (outer, *inner):
        fac.fit(SS.PARAMS, f.X_tr[feats], f.y_tr, sample_weight=f.w_tr)
    return [s["selected"] for s in fac.selections]


def test_flipping_a_folds_held_out_labels_everywhere_leaves_its_selections_unchanged(world):
    # Fold 0's held-out rows are fit rows of folds 1-4; their labels are flipped in every frame. Fold 0's outer
    # selection and its five inner selections (which feed its calibration and nested threshold) are unchanged.
    # The control: a planted interaction in fold 0's own fit labels changes its selection.
    frames = world["frames"]
    feats = list(world["cfg"]["jed_alone_selected"]["features"])
    held0 = np.asarray(frames["outer"][0].X_val.index)
    reaches = int(np.isin(np.asarray(frames["outer"][1].X_tr.index), held0).sum())
    assert reaches > 0                                                  # the perturbation is real elsewhere
    assert not np.isin(np.asarray(frames["outer"][0].X_tr.index), held0).any()
    base = _fold_selections(frames["outer"][0], frames["inner"][0], feats)
    flipped = _fold_selections(_flipped(frames["outer"][0], held0), [_flipped(i, held0) for i in frames["inner"][0]],
                               feats)
    assert flipped == base and len(base) == 6
    f0 = frames["outer"][0]
    X = f0.X_tr
    z = lambda c: (X[c] - X[c].mean()) / X[c].std()  # noqa: E731
    rng = np.random.default_rng(3)
    planted = (rng.random(len(X)) < 1 / (1 + np.exp(-4.0 * z("age") * z("campaign")))).astype(int)
    fac = IS.ProceduralEBMFactory()
    fac.fit(SS.PARAMS, X[feats], planted, sample_weight=f0.w_tr)
    assert "age x campaign" == fac.selections[0]["selected"][0] and fac.selections[0]["selected"] != base[0]


def test_recurrence_and_change_classes(world):
    sels = world["sel"]["jed_alone_selected"]["selections"]
    ref = world["cfg"]["jed_alone_selected"]["five_a_pairs"]
    t = IS.recurrence(sels, ref)
    assert len(t) == 78 and t["outer_selected"].between(0, 5).all() and t["inner_fits"].iloc[0] == 25
    assert set(t.loc[t["reference"], "change"]) <= {"persists", "disappears"}
    assert set(t.loc[~t["reference"], "change"]) <= {"enters", "occasional entrant", ""}
    for _, r in t.iterrows():
        assert r["recurrence"] == ("stable" if r["outer_selected"] == 5 else "recurring" if r["outer_selected"] >= 3
                                   else "occasional" if r["outer_selected"] else "never")
    assert t["outer_selected"].sum() == 5 * IS.BUDGET
    by_fit = IS.selected_by_fit(sels)
    assert by_fit["fit"].iloc[0] == "outer 0" and by_fit["fit"].iloc[-1] == "all rows" and len(by_fit) == 31


def test_the_paired_comparison_against_the_5a_fit(world):
    fold_id = world["block"].tune_plan.fold_id
    c = IS.paired_against(world["fits_5a"]["jed_alone"], world["sel"]["jed_alone_selected"], world["y"], fold_id)
    p = c["pooled"]
    assert abs(p["d_pr_auc"] - (p["pr_auc"] - p["pr_auc_5a"])) < 1e-12
    assert p["d_roc_auc_low"] <= p["d_roc_auc"] <= p["d_roc_auc_high"]
    assert set(c["fold_differences"]["metric"]) == {"roc_auc", "pr_auc", "brier_calibrated", "ece_calibrated",
                                                   "nested_f2"}
    same = IS.paired_against(world["fits_5a"]["jed_alone"], world["fits_5a"]["jed_alone"], world["y"], fold_id)
    assert same["decisions"]["differ"] == 0 and abs(same["pooled"]["d_roc_auc"]) < 1e-12
    m = IS.materiality(same["pooled"])
    assert not m["material_change"]
    t = IS.comparison_table({"jed_alone_selected": c})
    assert {"material_change", "differ", "d_roc_auc"} <= set(t.columns)


def test_surfaces_and_interaction_value_for_a_selected_pair(world):
    r = world["sel"]["jed_alone_selected"]
    pair = r["selections"][-1]["selected"][0]                          # the all-rows fit's first pair
    models = {"jed_alone_selected": {"features": world["cfg"]["jed_alone_selected"]["features"], "full": r["full"],
                                     "folds": r["models"]}}
    s = IS.surfaces(models, world["frames"]["X_all"], world["T"], [pair])
    assert len(s) == 1 and s[0]["support"].sum() == len(world["y"])
    assert np.isnan(s[0]["surface"][s[0]["support"] == 0]).all()
    v = IS.interaction_value(models, world["frames"]["X_all"], s)
    assert 0 < v["pair_share_of_all_terms"].iloc[0] < 1 and v["fold_models_with_pair"].iloc[0] >= 1


class _Stub:
    """A fitted EBM stand-in: main terms a, b, c and, if it selected it, the pair a & b (contribution a * b)."""

    def __init__(self, pair: bool):
        self.term_names_ = ["a", "b", "c"] + (["a & b"] if pair else [])

    def eval_terms(self, X):
        cols = [X["a"], X["b"], X["c"]] + ([X["a"] * X["b"]] if "a & b" in self.term_names_ else [])
        return np.column_stack(cols).astype(float)


def test_a_pair_selected_by_one_fold_has_no_fold_sd():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    T = pd.DataFrame({"period": rng.integers(0, 4, 2000)})
    feats = ["a", "b", "c"]
    one = {"one": {"features": feats, "full": _Stub(True), "folds": [_Stub(True)] + [_Stub(False)] * 4}}
    s = IS.surfaces(one, X, T, ["a x b"])[0]
    assert s["fold_models"] == 1 and not s["fold_sd_available"] and np.isnan(s["fold_sd"]).all()
    assert s["fold_sd_max_supported"] is None and s["strong_cells_sign_agreement"] is None and s["max_abs"] > 0
    v = IS.interaction_value(one, X, [s])
    assert v["fold_models_with_pair"].iloc[0] == 1 and not v["fold_sd_available"].iloc[0]
    assert pd.isna(v["fold_sd_max_supported"].iloc[0])                 # unavailable, never 0
    row = IS.value_summary(v).iloc[0]
    assert (row["selecting_folds"], row["fold_sd_max_supported"], row["strong_cells_sign_agreement"]) == (
        "1 of 5", "unavailable", "unavailable")
    two = {"two": {"features": feats, "full": _Stub(True), "folds": [_Stub(True)] * 2 + [_Stub(False)] * 3}}
    s2 = IS.surfaces(two, X, T, ["a x b"])[0]
    assert s2["fold_models"] == 2 and s2["fold_sd_available"] and s2["fold_sd_max_supported"] == pytest.approx(0, abs=1e-12)
    assert s2["strong_cells_sign_agreement"] == 1.0
    assert IS.value_summary(IS.interaction_value(two, X, [s2])).iloc[0]["selecting_folds"] == "2 of 5"
    fig = SS.pair_figure("a & b", [s, s2], order=("one", "two"), labels={"one": "one fold", "two": "two folds"})
    texts = [t.get_text() for ax in fig.axes for t in ax.texts] + [ax.get_title() for ax in fig.axes]
    assert any("fold sd unavailable" in t and "1 of 5" in t for t in texts)
    assert any("2 of 5 folds select it" in t for t in texts)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_a_value_statement_on_an_absent_surface_or_an_unavailable_statistic_fails_with_its_reason(monkeypatch):
    # Cell 67's statements must not raise when a run legitimately differs from the sandbox: an all-rows fit that
    # does not hold a stated pair (absent surface), a statistic that is None (one selecting fold, or no strong
    # cell), missing or not finite. Each becomes a statement that does not hold, with its reason.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    T = pd.DataFrame({"period": rng.integers(0, 4, 2000)})
    feats = ["a", "b", "c"]
    one = {"one": {"features": feats, "full": _Stub(True), "folds": [_Stub(True)] + [_Stub(False)] * 4}}
    two = {"two": {"features": feats, "full": _Stub(True), "folds": [_Stub(True)] * 2 + [_Stub(False)] * 3}}
    v_one = IS.interaction_value(one, X, IS.surfaces(one, X, T, ["a x b"]))
    v_two = IS.interaction_value(two, X, IS.surfaces(two, X, T, ["a x b"]))
    weak = v_two.assign(configuration="weak", strong_cells=0, strong_cells_sign_agreement=None)
    inf = v_two.assign(configuration="inf", max_abs_supported=float("inf"))
    value = pd.concat([v_one, v_two, weak, inf], ignore_index=True)
    assert value.loc[0, "strong_cells_sign_agreement"] is None          # object column: float(None) would raise
    claims = {("two", "a x b", "max_abs_supported"): (">=", 0.0),                     # holds
              ("one", "a x b", "strong_cells_sign_agreement"): (">=", 1.0),           # one selecting fold
              ("one", "a x b", "fold_sd_max_supported"): ("<=", 0.1),                 # one selecting fold
              ("weak", "a x b", "strong_cells_sign_agreement"): (">=", 1.0),          # no strong cell
              ("absent", "a x b", "max_abs_supported"): ("<=", 0.4),                  # configuration not shown
              ("two", "c x a", "max_abs_supported"): ("<=", 0.4),                     # pair not held
              ("inf", "a x b", "max_abs_supported"): ("<=", 0.4),                     # not finite
              ("two", "a x b", "no_such_statistic"): (">=", 0.0)}                     # column missing
    monkeypatch.setattr(IS, "STATED", {"value_claims": claims})
    st = IS.stated_checks(recovery={}, tables={}, fits={}, cmp=pd.DataFrame(), vs={}, value=value, screen=None)
    assert len(st) == len(claims) and st["passed"].tolist() == [True] + [False] * 7
    why = dict(zip(st["check"], st["detail"]))
    assert why["one a x b: strong_cells_sign_agreement >= 1.0"].startswith("statistic unavailable") and \
        "1 of 5 select it" in why["one a x b: strong_cells_sign_agreement >= 1.0"]
    assert "1 of 5 select it" in why["one a x b: fold_sd_max_supported <= 0.1"]
    assert "no well-supported cell" in why["weak a x b: strong_cells_sign_agreement >= 1.0"]
    assert why["absent a x b: max_abs_supported <= 0.4"].startswith("surface absent")
    assert why["two c x a: max_abs_supported <= 0.4"].startswith("surface absent")
    assert why["inf a x b: max_abs_supported <= 0.4"] == "statistic not finite: max_abs_supported is inf"
    assert why["two a x b: no_such_statistic >= 0.0"].startswith("statistic missing")
    empty = IS.stated_checks(recovery={}, tables={}, fits={}, cmp=pd.DataFrame(), vs={}, value=pd.DataFrame(),
                             screen=None)
    assert not empty["passed"].any() and empty["detail"].str.startswith("surface absent").all()


# ======================================================================
# acceptance and same-sandbox reproduction
# ======================================================================
def _closing(world):
    """Cell 67's inputs on the synthetic world: jed_alone reselected against its 5A fit."""
    fold_id = world["block"].tune_plan.fold_id
    cfg = {"jed_alone_selected": world["cfg"]["jed_alone_selected"]}
    sel = world["sel"]
    vs = {"jed_alone_selected": IS.paired_against(world["fits_5a"]["jed_alone"], sel["jed_alone_selected"], world["y"],
                                                  fold_id)}
    cmp = IS.comparison_table(vs)
    chk = SS.fit_checks(cfg, sel)
    fit_checks = pd.concat([chk[~chk["check"].str.contains("26,360")],
                            SS.isolation_checks(sel["jed_alone_selected"], world["block"])], ignore_index=True)
    ok = pd.DataFrame([{"check": "stand-in", "passed": True, "detail": None}])
    views = {"status": {"v": "not available locally"}, "required": False, "available": False}
    acc_in = dict(recovery=world["rec"], fits_5a=world["fits_5a"], fits_5b=sel, configs_5b=cfg, fit_checks=fit_checks,
                  frames=world["frames"], y=world["y"], fold_id=fold_id, cmp=cmp, screen=None, noms_5a=pd.DataFrame(),
                  stated_5a=ok, accepted_5a=ok, views=views)
    tables = {"baseline": IS.recurrence(world["rec"]["selections"], IS.ACCEPTED_PAIRS),
              "jed_alone_selected": IS.recurrence(sel["jed_alone_selected"]["selections"],
                                                  cfg["jed_alone_selected"]["five_a_pairs"])}
    r = sel["jed_alone_selected"]
    pair = r["selections"][-1]["selected"][0]
    models = {"jed_alone_selected": {"features": cfg["jed_alone_selected"]["features"], "full": r["full"],
                                     "folds": r["models"]}}
    surfs = IS.surfaces(models, world["frames"]["X_all"], world["T"], [pair])
    value = IS.interaction_value(models, world["frames"]["X_all"], surfs)
    E = pd.DataFrame(columns=["check", "passed", "detail"])
    sec = IS.record_sections(reg_checks=E, history={"source": {}, "statements": []}, accepted_5a=E, accepted_5a_checks=E,
                             recovery=world["rec"], rec_checks=E, configs=cfg, fits=sel, fit_checks=fit_checks,
                             tables=tables, vs=vs, cmp=cmp, shown=[pair], surfaces=surfs, value=value, views=views,
                             screen=None, screen_required=E)
    return acc_in, sec, views


def test_acceptance_holds_and_fails_on_a_misreported_verdict_or_mismatched_rows(world):
    import copy
    acc_in, _, _ = _closing(world)
    acc = IS.acceptance_checks(**acc_in)
    assert acc["passed"].all(), acc.to_string()
    assert acc["check"].str.startswith("ranking-to-fitted-term agreement").sum() == 3
    assert acc["check"].str.startswith("paired row identity").sum() == 3
    flipped = acc_in["cmp"].copy()
    flipped["material_change"] = ~flipped["material_change"].astype(bool)
    bad = IS.acceptance_checks(**{**acc_in, "cmp": flipped})
    assert not bad.loc[bad["check"].str.startswith("truthful materiality"), "passed"].any()
    shifted = IS.acceptance_checks(**{**acc_in, "fold_id": np.roll(acc_in["fold_id"], 1)})
    failed = set(shifted.loc[~shifted["passed"], "check"].str.split(":").str[0])
    assert {"paired row identity", "fit isolation"} <= failed
    sel = copy.copy(acc_in["fits_5b"]["jed_alone_selected"])
    sel["selections"] = [dict(s) for s in sel["selections"]]
    sel["selections"][0]["selection_matches_fit"] = False
    sel["terms"] = [list(t) for t in sel["terms"]]
    sel["terms"][1] = [t for t in sel["terms"][1] if " & " not in t]
    worse = IS.acceptance_checks(**{**acc_in, "fits_5b": {"jed_alone_selected": sel}})
    assert worse.loc[worse["check"].str.startswith("ranking-to-fitted-term agreement"), "passed"].tolist() == [
        False, True, False]


def test_reproduction_is_required_only_in_the_sandbox_and_elsewhere_flags_changes(world):
    import copy
    from upstream_regeneration.records import canonical
    acc_in, sec, views = _closing(world)
    now = IS.observed(sec, step5a_content="abc")
    env = {"python": "3", "platform": "p", "os": "o", "executable": "e", "packages": {}}
    ref = {"run_id": "sandbox", "environment": env, "views_available": False, "values": copy.deepcopy(now)}
    holds = pd.DataFrame([{"check": "a statement", "passed": True, "detail": 1}])
    same = IS.sandbox_reproduction(sec, holds, step5a_content="abc", environment=env, views=views, reference=ref)
    assert same["required"] and same["checks"]["passed"].all() and same["table"]["equal"].all()
    key = next(k for k, v in now.items() if v["kind"] == "difference" and k.startswith("jed_alone_selected | vs 5A")
               and v["value"])
    verdict, pairs = "jed_alone_selected | vs 5A | material_change", "jed_alone_selected | outer 0 | pairs"
    moved = copy.deepcopy(ref)
    moved["values"][key]["value"] = -now[key]["value"]
    strict = IS.sandbox_reproduction(sec, holds, step5a_content="abc", environment=env, views=views, reference=moved)
    assert strict["required"] and not strict["checks"]["passed"].all()          # same sandbox: a difference fails
    moved["values"][verdict]["value"] = not now[verdict]["value"]
    moved["values"][pairs]["value"] = ["x", "y", "z"]
    gone = pd.DataFrame([{"check": "a statement that no longer holds", "passed": False, "detail": 2}])
    r = IS.sandbox_reproduction(sec, gone, step5a_content="abc", environment={**env, "platform": "windows"},
                                views=views, reference=moved)
    assert not r["required"] and r["checks"]["passed"].all() and "platform differs" in r["basis"]
    t = r["table"].set_index("item")
    material = bool(acc_in["cmp"]["material_change"].iloc[0])
    assert t.loc[key, "flag"] == "direction changed" + ("" if material else " (below the materiality threshold)")
    assert t.loc[key, "sandbox"] == -now[key]["value"] and t.loc[key, "this_run"] == now[key]["value"]
    assert t.loc[key, "difference"] == pytest.approx(2 * now[key]["value"])
    assert (t.loc[verdict, "flag"], t.loc[pairs, "flag"]) == ("verdict changed", "selection changed")
    assert t.loc[[key, verdict, pairs], "review"].all() and int(t["review"].sum()) == 3
    rev = IS.review_flags(r, None)
    assert len(rev) == 4 and "statement no longer holds" in set(rev["flag"])
    acc = IS.acceptance_checks(**acc_in)
    final = IS.final_sections(sec, acc, r, rev)
    assert final["acceptance"]["passed"] and final["review"]["flagged"] == 4 and not final["reproduction"]["required"]
    json.dumps(canonical(final), allow_nan=False)
    s = IS.reproduction_summary(r).set_index("kind")
    assert s["items"].sum() == len(r["table"]) and s.loc["verdict", "flagged"] == 1


def test_a_failed_value_statement_is_recorded_and_flagged_and_still_fails_the_same_sandbox(world, monkeypatch):
    # The review-reporting path of Cells 67-68: statements that no longer hold (an absent surface, an unavailable
    # statistic) reach the record and the review flags with their reasons, elsewhere nothing fails, and in the
    # sandbox's environment the reproduction still fails on them. Acceptance is unaffected.
    import copy
    from upstream_regeneration.records import canonical
    acc_in, sec, views = _closing(world)
    value = pd.DataFrame(sec["surfaces"]["value"])
    shown = value.iloc[0]
    one = value.copy().astype(object)
    one.loc[0, ["fold_models_with_pair", "fold_sd_available", "strong_cells_sign_agreement", "fold_sd_max_supported"]] = \
        [1, False, None, None]
    claims = {(shown["configuration"], shown["pair"], "max_abs_supported"): (">=", 0.0),
              (shown["configuration"], shown["pair"], "strong_cells_sign_agreement"): (">=", 0.5),
              ("drop_first_selected", "decay_x_density x campaign", "max_abs_supported"): (">=", 0.0)}
    monkeypatch.setattr(IS, "STATED", {"value_claims": claims})
    st = IS.stated_checks(recovery={}, tables={}, fits={}, cmp=pd.DataFrame(), vs={}, value=one, screen=None)
    assert st["passed"].tolist() == [True, False, False]
    now = IS.observed(sec, step5a_content="abc")
    env = {"python": "3", "platform": "p", "os": "o", "executable": "e", "packages": {}}
    ref = {"run_id": "sandbox", "environment": env, "views_available": False, "values": copy.deepcopy(now)}
    elsewhere = IS.sandbox_reproduction(sec, st, step5a_content="abc", environment={**env, "os": "Windows"},
                                        views=views, reference=ref)
    assert not elsewhere["required"] and elsewhere["checks"]["passed"].all()
    rev = IS.review_flags(elsewhere, None)
    assert len(rev) == 2 and set(rev["flag"]) == {"statement no longer holds"}
    assert rev["this_run"].str.startswith("statistic unavailable").sum() == 1
    assert rev["this_run"].str.startswith("surface absent").sum() == 1
    acc = IS.acceptance_checks(**acc_in)
    assert acc["passed"].all()                                           # acceptance does not depend on statements
    final = canonical(IS.final_sections(sec, acc, elsewhere, rev))
    json.dumps(final, allow_nan=False)
    assert final["acceptance"]["passed"] and final["review"]["flagged"] == 2
    assert [x["passed"] for x in final["reproduction"]["statements"]] == [True, False, False]
    same = IS.sandbox_reproduction(sec, st, step5a_content="abc", environment=env, views=views, reference=ref)
    assert same["required"] and same["table"]["equal"].all()
    assert same["checks"]["passed"].tolist() == [True, False]             # strict in the sandbox: the statements fail


def test_cascade_screen_needs_the_views():
    assert IS.cascade_screen({"available": False}, {}, {}, pd.Series([0, 1]), np.array([0, 0])) is None


# ======================================================================
# the notebook
# ======================================================================
def test_the_notebook_runs_step5b_after_step5a_and_writes_its_record_once():
    nb = json.loads((BANK / "feature_research/feature_research.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")
    idx = {}
    for i, s in enumerate(code):
        for line in s.splitlines():
            if line.startswith("# CELL "):
                idx[line.split()[2].rstrip(":")] = i
    for n in range(60, 69):
        assert str(n) in idx
    assert idx["59"] < idx["60"] < idx["68"]
    assert "from feature_research import interaction_selection as IS" in code[0]
    assert "IS.registration_checks(history=HISTORY_5B)" in code[idx["60"]]
    assert "IS.accepted_5a_reproduction(SIMPLIFICATION_RECORD, RUN.environment, VIEWS_5A)" in code[idx["60"]]
    assert "IS.recovery(BLOCK_5A, FRAMES_5A)" in code[idx["61"]]
    assert "IS.fit_selected(" in code[idx["62"]]
    assert "IS.cascade_screen(VIEWS_5A" in code[idx["66"]]
    assert "IS.acceptance_checks(" in code[idx["67"]] and "IS.sandbox_reproduction(" in code[idx["67"]]
    c68 = code[idx["68"]]
    assert c68.index('RR.write_run_record(RUN, "interaction_selection"') < c68.index("SS.require(ACCEPT_5B")
    assert c68.index("SS.require(ACCEPT_5B") < c68.index('SS.require(REPRO_5B["checks"]')
    joined = "\n".join(code)
    assert joined.count('RR.write_run_record(RUN, "interaction_selection"') == 1
    assert '"interaction_selection": IS.SEED' in code[idx["3"]]
    for s in code[idx["60"]:idx["68"] + 1]:
        for bad in ("test", "oracle", "holdout", "refit", "fold_mean", "metrics", "glass_router_data"):
            assert bad not in s, bad
