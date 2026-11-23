"""
PR 42 Commit 4, Step 4A: ``feature_research.binning_drift`` on synthetic data
(the real training rows: ``tests/realdata/test_pr42_binning_drift.py``).

Covered: the module adds no importer of the Stage 2 feature code (so the
Commit 2 caller scan and the Step 3A records stay reproducible); the
historical transcription's arithmetic and Git check; the binning helpers
against the production binning path; the PR 39 reproduction (and that a
changed record fails it); held-out label flips that pass for the PR 39 path
and fail for an in-sample encoder; fold-fit values from exactly each fold's
fit rows; the decomposition adding up; pooled tables reconciling with their
folds; deciles as ``compute_lift`` forms them; period standardization; and
straddling states. The notebook's Step 4A cells are checked by source.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_research import binning_drift as BD
from shared.stage2.binning import BINNING_STRATEGY
from stage2_binning import evidence as E
from stage2_binning import runner as R
from tests.support import stage2_crossfit_world as W

BANK = Path(BD.__file__).resolve().parents[1]
FEATURE_MODULES = ("shared.stage2.feature_engineering", "feature_research.feature_engineering")


# ======================================================================
# no new importer of the feature code
# ======================================================================
def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_bytes())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


@pytest.mark.parametrize("path", [BANK / "feature_research" / "binning_drift.py", Path(__file__)])
def test_step4a_code_imports_no_feature_engineering_module(path):
    mods = _imported_modules(path)
    assert not [m for m in mods for t in FEATURE_MODULES if m == t or m.startswith(t + ".")]


def test_the_commit2_caller_scan_does_not_list_step4a():
    from feature_research import feature_inventory as FI
    callers = json.dumps(FI.duplicate_callers())
    assert "binning_drift" not in callers


# ======================================================================
# figures and provenance
# ======================================================================
def test_the_historical_transcription_adds_up():
    checks = BD.historical_consistency()
    assert checks["passed"].all(), checks.to_string()


def test_each_frozen_cut_against_the_printed_decile_edge_it_was_read_from():
    t = BD.implied_cut_rows().set_index(["family", "cut"])["rows_between"]
    assert t.to_dict() == {("NSD", 0.0524): 610, ("NSD", 0.192): 520, ("NSD", 0.23): 129,
                           ("ECI", 0.0291): 0, ("ECI", 0.0758): 2112, ("ECI", 0.0948): 0}


def test_the_quoted_figures_are_the_historical_deciles():
    h = BD.HISTORICAL
    lifts = {b: [h["deciles"][BD.FOCAL[b]][i - 1]["lift"] for i in q["deciles"]] for b, q in h["quoted"].items()}
    assert lifts == {"eci_warm": ["1.67", "1.49"], "eci_hot": ["3.53"], "nsd_elevated": ["1.04", "1.43"]}
    assert h["bins"]["ECI"]["eci_warm"][2] == "0.92" and h["bins"]["NSD"]["nsd_elevated"][2] == "1.25"


def test_verify_historical_reports_unavailable_outside_a_repository(tmp_path):
    out = BD.verify_historical(root=tmp_path)
    assert out["status"] == "unavailable" and all(r["passed"] for r in out["arithmetic"])


def test_verify_historical_detects_a_changed_transcription():
    if BD.verify_historical()["status"] == "unavailable":
        pytest.skip("commit 1ce1649 is not in this checkout (shallow clone)")
    changed = copy.deepcopy(dict(BD.HISTORICAL))
    changed["deciles"] = copy.deepcopy(dict(BD.HISTORICAL["deciles"]))
    changed["deciles"]["ECI"][6] = dict(changed["deciles"]["ECI"][6], lift="0.36")
    assert BD.verify_historical(historical=changed)["status"] == "mismatch"


def test_the_pr39_record_is_read_only_at_its_pinned_digest(tmp_path):
    record, prov = BD.read_pr39_record()
    assert prov["sha256"] == BD.PR39_RECORD["sha256"] and record["decision"]["outcome"] == "retain"
    bad = tmp_path / BD.PR39_RECORD["path"]
    bad.parent.mkdir(parents=True)
    bad.write_bytes((BANK / BD.PR39_RECORD["path"]).read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="not the pinned"):
        BD.read_pr39_record(root=tmp_path)


# ======================================================================
# binning helpers
# ======================================================================
def _source_frame(values: dict, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for fam, g in BINNING_STRATEGY.items():
        src = g["source"]
        cols[src] = values.get(fam, rng.integers(0, 2, n) if "passthrough" in g else rng.normal(0, 1, n))
    return pd.DataFrame(cols)


def test_assign_bins_is_the_production_binning_path_including_the_cut_values():
    from stage2_crossfit.g13 import bin_source
    rng = np.random.default_rng(1)
    vals = {}
    for fam in BD.FAMILIES:
        cuts = [hi for _, _, hi in BINNING_STRATEGY[fam]["bins"][:-1]]
        lo, hi = cuts[0] * 0.5, cuts[-1] * 1.5
        vals[fam] = np.r_[rng.uniform(lo, hi, 500), cuts, np.nextafter(np.array(cuts), np.inf)]
    n = len(vals["NSD"])
    frame = _source_frame(vals, n)
    binned = bin_source(frame)
    for fam in BD.FAMILIES:
        assert np.array_equal(BD.assign_bins(vals[fam], fam), E.family_labels(binned, fam))


def test_assign_intervals_uses_right_closed_intervals():
    names = ["a", "b", "c"]
    got = BD.assign_intervals([-1.0, 0.0, 0.5, 1.0, 1.0000001, 5.0], [0.0, 1.0], names)
    assert got.tolist() == ["a", "a", "b", "b", "c", "c"]
    with pytest.raises(ValueError):
        BD.assign_intervals([0.0], [1.0, 0.0], names)


# ======================================================================
# reproduction on a synthetic train side
# ======================================================================
@pytest.fixture(scope="module")
def world():
    from feature_research import feature_inventory as FI
    ts = W.synthetic_train_side()
    side = BD.stage2_side(ts.X_train, ts.y_train)
    eb = BD._canonical(R.eb_evidence(side))
    trig = BD._canonical({f: E.triggers(f, eb["families"][f]["oof"]) for f in E.FAMILIES})
    y = side.y_train.to_numpy()
    rows = [int((side.fold_id == k).sum()) for k in range(BD.N_FOLDS)]
    pos = [int(y[side.fold_id == k].sum()) for k in range(BD.N_FOLDS)]
    record = {"population": {"rows": len(y), "positives": int(y.sum()),
                             "fold_fingerprint": side.fold_fingerprint,
                             "frozen_protocol_check": {"checks": {"fold_rows": {"pinned": rows},
                                                                  "fold_positives": {"pinned": pos}}}},
              "eb": eb, "triggers": trig, "decision": {"families": BD._canonical(E.decide(trig))}}
    frames = FI.build_training_frames(side.X_train, side.y_train)
    repro = BD.reproduce_pr39(side, record)
    sources = BD.source_values(side, frames["stage2_sources"], repro["eb"])
    T = pd.DataFrame({"position": np.arange(len(y)), "period": np.where(np.arange(len(y)) < len(y) // 2,
                                                                        "2008-05", "2008-06"),
                      "period_index": (np.arange(len(y)) >= len(y) // 2).astype(int)},
                     index=side.X_train.index)
    states = BD.macro_states(side.X_train, T, side.y_train)
    fold_fit = BD.fold_fit_values(side, states, repro["eb"])
    svals = BD.state_values(states, sources, side, fold_fit)
    return {"ts": ts, "side": side, "record": record, "repro": repro, "frames": frames,
            "sources": sources, "T": T, "states": states, "fold_fit": fold_fit, "svals": svals}


def test_the_stage2_side_is_the_pr39_fold_plan(world):
    ts, side = world["ts"], world["side"]
    assert side.fold_fingerprint == ts.fold_fingerprint
    assert np.array_equal(side.fold_id, ts.fold_id)


def test_reproduce_pr39_passes_on_the_record_and_fails_when_it_changes(world):
    assert world["repro"]["checks"]["passed"].all(), world["repro"]["checks"].to_string()
    changed = copy.deepcopy(world["record"])
    changed["eb"]["families"]["ECI"]["oof"]["bins"]["eci_hot"]["pooled"]["positives"] += 1
    checks = BD.reproduce_pr39(world["side"], changed)["checks"].set_index("check")["passed"]
    assert not checks[[c for c in checks.index if c.startswith("E-B")]].any()
    moved = copy.deepcopy(world["record"])
    moved["population"]["fold_fingerprint"] = "0" * 16
    assert not BD.reproduce_pr39(world["side"], moved)["checks"]["passed"].all()


def test_source_values_rebuild_e_b_tables_from_row_values(world):
    assert world["sources"].checks["passed"].all(), world["sources"].checks.to_string()


def test_source_values_refuse_an_in_sample_frame_that_is_not_the_training_rows(world):
    frame = world["frames"]["stage2_sources"].copy()
    col = frame.columns.get_loc(BD.SOURCES["NSD"])
    frame.iloc[:5, col] = frame.iloc[:5, col] * 1.5
    s = BD.source_values(world["side"], frame, world["repro"]["eb"])
    assert not s.checks["passed"].all()


def test_held_out_label_flips_pass_for_the_pr39_path(world):
    checks = BD.heldout_label_checks(world["side"], folds=[0, 3])
    assert checks["passed"].all(), checks.to_string()
    assert len(checks) == 4


def test_held_out_label_flips_catch_an_in_sample_encoder(world):
    from feature_research import feature_inventory as FI
    leaky = lambda s: FI.build_training_frames(s.X_train, s.y_train)["stage2_sources"]  # noqa: E731
    checks = BD.heldout_label_checks(world["side"], folds=[0], oof_fn=leaky).set_index("check")["passed"]
    unchanged = [c for c in checks.index if "unchanged" in c]
    assert not checks[unchanged].any()


def test_fold_fit_values_come_from_exactly_each_folds_fit_rows(world):
    assert world["svals"]["checks"]["passed"].all(), world["svals"]["checks"].to_string()
    assert len(world["fold_fit"]["checks"]) == BD.N_FOLDS


def test_state_values_flag_an_encoding_that_is_not_a_function_of_the_state(world):
    states, s = world["states"], world["sources"]
    nsd = s.values["in_sample"]["NSD"]
    a, b = 0, int(np.flatnonzero(np.abs(nsd - nsd[0]) > 1e-6)[0])   # two rows with different values
    code = states.code.copy()
    code[b] = code[a]                                                 # declared one macro state
    merged = BD.MacroStates(code=code, table=states.table)
    checks = BD.state_values(merged, s, world["side"], world["fold_fit"])["checks"]
    assert not checks[checks["check"].str.startswith("NSD: every encoder's value is a function")]["passed"].all()
    assert world["svals"]["checks"]["passed"].all()


def test_the_encoder_comparison_parts_add_up_and_reconcile(world):
    enc = BD.encoder_comparison(world["side"], world["sources"], world["states"], world["svals"])
    d = enc["decomposition"]
    cols = ["fold_fit_minus_in_sample", "heldout_minus_fold_fit", "oof_minus_in_sample"]
    assert not {"encoder_change", "label_dependence_change", "total_change"} & set(d.columns)
    full = d.dropna(subset=cols)
    assert len(full) >= 6                                    # empty synthetic bins have no lift
    assert np.allclose(full[cols[0]].astype(float) + full[cols[1]].astype(float), full[cols[2]].astype(float))
    n = len(world["side"].y_train)
    for fam in BD.FAMILIES:
        assert d[d["family"] == fam]["fold_fit_pairs"].sum() == 4 * n
        assert d[d["family"] == fam]["oof_support"].sum() == n
        assert enc["migration"][enc["migration"]["family"] == fam]["rows"].sum() == n
    pf = enc["per_fold"]
    assert (pf.groupby(["family", "fold"])["heldout_rows"].sum() == n // BD.N_FOLDS).all()
    for t in enc["families"].values():
        assert all("lift_wilson95" not in b["pooled"] for b in t["fold_fit"]["bins"].values())


def _reconciles(table, n_folds):
    for b in table["bins"].values():
        if "folds" not in b:
            continue
        assert sum(c["support"] for c in b["folds"]) == b["pooled"]["support"]
        assert sum(c["positives"] for c in b["folds"]) == b["pooled"]["positives"]
    for k in range(n_folds):
        assert sum(b["folds"][k]["support"] for b in table["bins"].values()) == \
            table["base_rate"]["folds"][k]["rows"]
    base = table["base_rate"]["pooled"]["base_rate"]
    for b in table["bins"].values():
        p = b["pooled"]
        if p["support"]:
            assert p["lift"] == pytest.approx(p["positives"] / p["support"] / base, rel=1e-12)


def test_pooled_bin_counts_reconcile_with_the_folds(world):
    eb, side = world["repro"]["eb"], world["side"]
    for fam in BD.FAMILIES:
        _reconciles(eb["families"][fam]["oof"], BD.N_FOLDS)
    deciles = BD.decile_tables(side, world["sources"])
    hist = BD.historical_thresholds(side, world["sources"])
    for fam in BD.FAMILIES:
        _reconciles(deciles["families"][fam]["oof"]["table"], BD.N_FOLDS)
        _reconciles(hist["families"][fam]["oof"], BD.N_FOLDS)


def test_deciles_are_compute_lifts_and_never_split_tied_values(world):
    side, sources = world["side"], world["sources"]
    deciles = BD.decile_tables(side, sources)
    assert BD.compute_lift_parity(side, sources, deciles)["passed"].all()
    tied = np.repeat([0.03, 0.05, 0.08, 0.09, 0.12], [700, 300, 400, 150, 50])
    y = (np.arange(len(tied)) % 7 == 0).astype(int)
    out = BD._deciles(tied, y, np.arange(len(tied)) % 5, "ECI", "x", True, 5)
    lab = pd.Series(out["labels"])
    assert (lab.groupby(tied).nunique() == 1).all()                  # a value never spans two deciles
    assert len(out["intervals"]) < BD.N_DECILES                         # ties collapse the edges
    assert sum(iv["rows"] for iv in out["intervals"]) == len(tied)


def test_period_standardization_is_one_for_bins_of_whole_periods():
    n = 400
    X = pd.DataFrame({"a": np.zeros(n)})
    y = pd.Series((np.arange(n) % 3 == 0).astype(int))
    side = BD.Stage2Side(X, y, ((np.arange(200, 400), np.arange(200)), (np.arange(200), np.arange(200, 400))),
                         (np.arange(n) >= 200).astype(int), "synthetic")
    T = pd.DataFrame({"period": np.repeat(["p1", "p2", "p3", "p4"], 100),
                      "period_index": np.repeat([0, 1, 2, 3], 100)})
    names = np.repeat(np.array(["nsd_cold", "nsd_warm", "nsd_elevated", "nsd_hot"], dtype=object), 100)
    labels = {"in_sample": {"NSD": names, "ECI": np.repeat(np.array(["eci_cold", "eci_hot"], dtype=object), 200)}}
    labels["oof"] = labels["in_sample"]
    src = BD.Sources(values={}, labels=labels, provenance=[], checks=pd.DataFrame())
    std = BD.period_evidence(side, src, T)["standardized"]
    got = std[std["support"] > 0]
    assert np.allclose(got["observed_over_expected"].dropna(), 1.0)
    nsd = got[(got["family"] == "NSD") & (got["rows"] == "pooled")]
    assert (nsd["abs_deviation_over_expected"] == 0).all() and (nsd["shared_periods"] == 0).all()


def test_a_period_ratio_of_one_can_hide_within_period_differences():
    """
    Two periods at 40% each, each split in half between two bins: bin A holds p1's 60% half and p2's 20%
    half, bin B the other halves. Each bin's pooled ratio is exactly 1, its absolute deviation is not.
    """
    n = 400
    X = pd.DataFrame({"a": np.zeros(n)})
    period = np.repeat(["p1", "p2"], 200)
    y = np.zeros(n, dtype=int)
    y[0:60] = 1                   # p1, rows 0-99 (bin A): 60 of 100
    y[100:120] = 1                # p1, rows 100-199 (bin B): 20 of 100
    y[200:220] = 1                # p2, rows 200-299 (bin A): 20 of 100
    y[300:360] = 1                # p2, rows 300-399 (bin B): 60 of 100
    side = BD.Stage2Side(X, pd.Series(y), ((np.arange(200, 400), np.arange(200)), (np.arange(200), np.arange(200, 400))),
                         (np.arange(n) >= 200).astype(int), "synthetic")
    T = pd.DataFrame({"period": period, "period_index": (period == "p2").astype(int)})

    def standardized(eci):
        labels = {"NSD": np.full(n, "nsd_cold", dtype=object), "ECI": eci}
        src = BD.Sources(values={}, labels={"in_sample": labels, "oof": labels}, provenance=[], checks=pd.DataFrame())
        std = BD.period_evidence(side, src, T)["standardized"]
        return std[(std["family"] == "ECI") & (std["kind"] == "in_sample")].set_index("bin")

    split = standardized(np.repeat(np.array(["eci_cold", "eci_hot", "eci_cold", "eci_hot"], dtype=object), 100))
    for b in ("eci_cold", "eci_hot"):
        r = split.loc[b]
        assert r["observed_over_expected"] == pytest.approx(1.0)        # 80 observed, 80 expected
        assert r["abs_deviation_over_expected"] == pytest.approx(0.5)   # |60 - 40| + |20 - 40| over 80
        assert (r["within_period_ratio_min"], r["within_period_ratio_max"]) == (pytest.approx(0.5), pytest.approx(1.5))
        assert r["shared_periods"] == 2
    whole = standardized(np.full(n, "eci_cold", dtype=object)).loc["eci_cold"]   # both periods whole
    assert whole["observed_over_expected"] == pytest.approx(1.0)
    assert whole["abs_deviation_over_expected"] == 0 and whole["shared_periods"] == 0


def test_straddling_states_name_the_cut_crossed():
    states = BD.MacroStates(code=np.array([0, 0, 1, 1]), table=pd.DataFrame({
        "state": [0, 1], "rows": [2, 2], "positives": [1, 0], "periods": ["p1", "p2"],
        "lift": [2.0, 0.0]}))
    nsd = pd.DataFrame({"in_sample": [0.19, 0.30], "fold_0": [0.195, 0.31]})
    eci = pd.DataFrame({"in_sample": [0.05, 0.05], "fold_0": [0.05, 0.05]})
    out = BD.straddling_states(states, {"values": {"NSD": nsd, "ECI": eci}})
    s = out["states"]
    assert s["state"].tolist() == [0] and s.loc[0, "cuts_crossed"] == "0.192"
    bb = out["by_bin"].set_index("bin")
    assert bb.loc["nsd_warm", "straddling_rows"] == 2 and bb.loc["nsd_hot", "straddling_rows"] == 0


# ======================================================================
# the notebook's Step 4A cells
# ======================================================================
def test_the_notebook_runs_step4a_after_step3b_and_writes_its_record_once():
    nb = json.loads((BANK / "feature_research" / "feature_research.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    headers = [next((ln for ln in s.splitlines() if ln.startswith("# CELL ")), "") for s in code]
    idx = {h.split(":")[0].replace("# CELL ", ""): i for i, h in enumerate(headers)}
    assert all(str(c) in idx for c in range(31, 40))
    assert idx["30"] < idx["31"] < idx["39"] < len(code)            # Step 4B's cells follow (Cells 40-47)
    joined = "\n".join(code)
    assert joined.count('"binning_drift"') == 3                  # seed name, record written, record read
    assert code[idx["39"]].count('RR.write_run_record(RUN, "binning_drift"') == 1
    assert 'RR.read_run_record(RUN.run_id, "binning_drift")' in code[idx["39"]]
    assert "RR.start_run(commit=4" in code[idx["3"]]
    assert 'TRAINING_FRAMES["stage2_sources"]' in code[idx["32"]]
    assert "BD.require(CLAIMS_4A" in code[idx["38"]]
    assert "BD.focal_assignments(" in code[idx["38"]] and "assignments=ASSIGNMENTS" in code[idx["38"]]
    assert "assignments=ASSIGNMENTS" in code[idx["39"]]
    assert 'EXPLORATORY_OUT["status"] == "mismatch"' in code[idx["31"]]


# ======================================================================
# the exploratory notebook's accepted printed outputs (pinned blob)
# ======================================================================
PRINTED_15A = {"NSD": "0.28 0.46 0.51 0.47 0.45 0.46 1.04 1.05 1.38 4.06".split(),
               "ECI": "0.57 0.47 0.49 0.49 0.82 0.97 0.50 1.69 1.49 3.48".split()}


def _pinned_bytes():
    import subprocess
    out = subprocess.run(["git", "cat-file", "blob", BD.EXPLORATORY["blob"]], cwd=BANK, capture_output=True)
    if out.returncode != 0:
        pytest.skip("the pinned exploratory blob is not in this checkout's Git history")
    return out.stdout


def _cleared(data: bytes) -> bytes:
    nb = json.loads(data.decode("utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def test_exploratory_outputs_come_from_the_pinned_blob_even_when_the_working_copy_is_cleared(tmp_path):
    data = _pinned_bytes()
    cleared = tmp_path / "exploratory_feature_research.ipynb"
    cleared.write_bytes(_cleared(data))
    out = BD.exploratory_outputs(working_copy=cleared)
    assert (out["status"], out["source"]) == ("verified", "git blob")
    assert out["deciles"] == PRINTED_15A
    assert out["bins"]["ECI"]["eci_warm"] == {"lift": "1.04", "share_pct": 38}
    assert out["bins"]["NSD"]["nsd_elevated"] == {"lift": "3.13", "share_pct": 4}


def test_without_git_the_working_copy_is_used_only_if_it_is_the_pinned_notebook(tmp_path):
    data = _pinned_bytes()
    intact, crlf, cleared = (tmp_path / n for n in ("intact.ipynb", "crlf.ipynb", "cleared.ipynb"))
    intact.write_bytes(data)
    crlf.write_bytes(data.replace(b"\n", b"\r\n"))
    cleared.write_bytes(_cleared(data))
    no_git = tmp_path / "not_a_repository"
    no_git.mkdir()
    for wc in (intact, crlf):
        out = BD.exploratory_outputs(root=no_git, working_copy=wc)
        assert (out["status"], out["source"]) == ("verified", "working copy") and out["deciles"] == PRINTED_15A
    out = BD.exploratory_outputs(root=no_git, working_copy=cleared)
    assert out["status"] == "unavailable" and out["deciles"] is None and "differs" in out["detail"]
    out = BD.exploratory_outputs(root=no_git, working_copy=tmp_path / "absent.ipynb")
    assert out["status"] == "unavailable" and "no working copy" in out["detail"]


def test_an_unavailable_source_is_reported_as_not_checked_and_a_mismatch_fails():
    for status, passed in (("unavailable", True), ("mismatch", False)):
        checks = BD.exploratory_reproduction({}, {}, {"status": status, "detail": "x"})
        assert len(checks) == 4 and (checks["passed"] == passed).all()
        assert checks["check"].str.contains("not checked").all()


# ======================================================================
# identities are checked on row masks and state assignments, not counts
# ======================================================================
def _tiny_world(top_rows):
    """8 rows, 4 macro states, 2 periods (p1 below the base rate), 2 folds."""
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    X = pd.DataFrame({"a": np.zeros(8)})
    fid = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = tuple((np.flatnonzero(fid != k), np.flatnonzero(fid == k)) for k in range(2))
    side = BD.Stage2Side(X, y, folds, fid, "tiny")
    T = pd.DataFrame({"period": ["p1"] * 4 + ["p2"] * 4, "period_index": [0] * 4 + [1] * 4})
    code = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    table = pd.DataFrame({"state": [0, 1, 2, 3], "rows": [2, 2, 2, 2], "positives": [0, 1, 2, 1],
                          "periods": ["p1", "p1", "p2", "p2"], "lift": [0.0, 1.0, 2.0, 1.0]})
    states = BD.MacroStates(code=code, table=table)
    enc = ["in_sample", "fold_0", "fold_1"]
    eci = pd.DataFrame({e: [0.01, 0.10, 0.12, 0.05] for e in enc})
    nsd = pd.DataFrame({"in_sample": [0.20, 0.10, 0.10, 0.10], "fold_0": [0.25, 0.20, 0.10, 0.10],
                        "fold_1": [0.20, 0.10, 0.10, 0.10]})
    svals = {"values": {"ECI": eci, "NSD": nsd}}
    lab = {fam: BD.assign_bins(v["in_sample"].to_numpy(), fam)[code] for fam, v in svals["values"].items()}
    oof = {fam: np.where(fid == 0, BD.assign_bins(v["fold_0"].to_numpy(), fam)[code],
                         BD.assign_bins(v["fold_1"].to_numpy(), fam)[code]).astype(object)
           for fam, v in svals["values"].items()}
    sources = BD.Sources(values={}, labels={"in_sample": lab, "oof": oof}, provenance=[], checks=pd.DataFrame())
    top = np.where(np.isin(np.arange(8), top_rows), "top", "rest").astype(object)
    deciles = {"families": {"ECI": {"in_sample": {"intervals": [{"decile": "rest"}, {"decile": "top", "display": "(x, y]"}]}}},
               "labels": {"ECI": {"in_sample": top}}}
    return side, sources, T, states, svals, deciles


def test_the_top_decile_identity_compares_rows_not_counts():
    a = BD.focal_assignments(*_tiny_world(top_rows=[4, 5]))["eci_hot"]
    assert a["top_decile_is_tail"] and a["cluster_is_whole_states"] and a["cluster_in_bin_under_every_encoder"]
    assert a["oof_cluster_is_same_rows"] and (a["tail_rows"], a["cluster_rows"]) == (2, 2)
    swapped = BD.focal_assignments(*_tiny_world(top_rows=[4, 6]))["eci_hot"]   # same count, other rows
    assert not swapped["top_decile_is_tail"]


def test_the_nsd_assignments_name_the_moved_states_and_the_folds_of_moved_rows():
    a = BD.focal_assignments(*_tiny_world(top_rows=[4, 5]))["nsd_elevated"]
    assert a["fold"] == 0 and [s["state"] for s in a["states_in_bin_under_fold"]] == [1]
    assert a["those_states_in_sample_bins"] == ["nsd_warm"] and a["those_states_all_below_base_periods"]
    assert a["those_states_bins_under_other_folds"] == {"fold_1": ["nsd_warm"]}
    assert a["in_sample_bin_states_bins_under_fold"] == ["nsd_hot"]
    assert a["moved_rows_folds"] == [0] and a["moved_rows"] == 2
