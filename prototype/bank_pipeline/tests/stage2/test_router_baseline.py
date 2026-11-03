"""
PR 37 Commit 2 (Step 2E): the Pre-Correction Stage 2 Baseline package
(shared.stage2.diagnostics.baseline), its source collector
(tests.support.router_artifacts.collect_baseline_sources) and its CLI
(research_logs/baselines/stage2_baseline.py) on the synthetic Commit 1 world.

Export, preservation and integration boundaries only: the diagnostics themselves
are covered by the Step 2A-2D suites and are reused here through the exported
tables. Historical values: tests/realdata/test_router_baseline_artifacts.py.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from shared.stage2.diagnostics import overlap as ov
from shared.stage2.diagnostics.contribution import TIE_BREAK_RULE
from shared.stage2.diagnostics.schema import ScoreStream
from tests.support import diagnostics_world as dw
from tests.support import router_artifacts as ra
from tests.support.notebooks import BANK_ROOT

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
ARMS = ("glass", "blackbox")
CLI_PATH = Path(BANK_ROOT) / b.BASELINES_DIR / "stage2_baseline.py"
REQUIRED_DISCLOSURES = (
    "definitions", "pass1_precedence", "leakage", "reference_flags", "overall_vs_matched",
    "ebm_fold_local", "ties", "expectations", "low_support", "signal_overlap", "shared_fold_rf",
    "pass2_populations", "match_glass_oof", "glass_unoptimised", "test_exposure", "scope",
    "joint_grid_threshold_groups", "pooled_lr_strata", "code_identity")


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def _artifacts(**sha_overrides):
    return tuple(b.SourceArtifact(role, f"models/synthetic/{role.replace('.', '_')}.joblib",
                                  sha_overrides.get(role,
                                                    hashlib.sha256(role.encode()).hexdigest()),
                                  1000 + i, True)
                 for i, role in enumerate(b.SOURCE_ROLES))


def _sources_from(w, ref=None, *, glass=None, rf=None, lr=None, ebm=None) -> b.BaselineSources:
    ref0, g, r, l_, e = dw.views(w, ref)
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref0)
    return b.BaselineSources(
        ref=ref0, glass=glass or g, rf=rf or r, lr=lr or l_, ebm=ebm or e, rf_candidate=cand,
        artifacts=_artifacts(),
        snapshot=b.SnapshotRecord("synthetic_snapshot/SHA256SUMS.txt", "a" * 64, 37),
        data_files=(b.DataFileRecord(ra.RAW_DATA_FILE, "c" * 64, "raw dataset"),))


def _synthetic_package() -> b.BaselinePackage:
    """Also run in subprocesses (hash-seed determinism)."""
    return b.build_baseline(_sources_from(dw.world()), b.code_identity(BANK_ROOT))


@pytest.fixture(scope="module")
def base():
    w = dw.world()
    sources = _sources_from(w)
    code = b.code_identity(BANK_ROOT)
    return {"w": w, "sources": sources, "code": code, "pkg": b.build_baseline(sources, code)}


@pytest.fixture
def frozen(base, tmp_path):
    parent = tmp_path / "baselines"
    parent.mkdir()
    res = b.write_baseline(base["pkg"], parent)
    return Path(res.directory)


def _csv(pkg_or_dir, name: str) -> pd.DataFrame:
    rel = b.TABLES[name][0]
    data = (pkg_or_dir.files[rel] if isinstance(pkg_or_dir, b.BaselinePackage)
            else (Path(pkg_or_dir) / rel).read_bytes())
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)


def _tree(root: Path) -> dict:
    """Every file and directory below ``root`` with its bytes and mtime (zero-write checks)."""
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        out[p.relative_to(root).as_posix()] = (
            ("dir", st.st_mtime_ns) if p.is_dir() else (p.read_bytes(), st.st_mtime_ns))
    return out


def _reroute(view, rows, token):
    dec = np.array(view.source_decision, dtype=object, copy=True)
    dec[rows] = token
    return replace(view, source_decision=dec,
                   state=np.array([d.SOURCE_DECISIONS[x][0] for x in dec], dtype=object),
                   source_pred=np.array([d.SOURCE_DECISIONS[x][1] for x in dec]))


def _with_score(view, space, values):
    old = view.scores[space]
    a = np.array(values, dtype=float)
    a.setflags(write=False)
    scores = dict(view.scores)
    scores[space] = ScoreStream(old.name, a, old.space, old.field, old.provenance)
    return replace(view, scores=MappingProxyType(scores))


def _forge(directory: Path, rel: str, data: bytes) -> None:
    """Replace a file AND re-hash it consistently in the manifest and sidecar."""
    (directory / rel).write_bytes(data)
    man = json.loads((directory / b.MANIFEST).read_bytes())
    man["outputs"][rel]["sha256"] = hashlib.sha256(data).hexdigest()
    man["outputs"][rel]["bytes"] = len(data)
    mb = (json.dumps(man, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (directory / b.MANIFEST).write_bytes(mb)
    (directory / b.SIDECAR).write_bytes(f"{hashlib.sha256(mb).hexdigest()}  {b.MANIFEST}\n"
                                        .encode("ascii"))


def _edit_last_digit(data: bytes) -> bytes:
    """Change one digit in the last data row (same header, same row count)."""
    lines = data.split(b"\n")
    row = bytearray(lines[-2])
    i = max(i for i, c in enumerate(row) if chr(c).isdigit())
    row[i] = ord("7") if row[i] != ord("7") else ord("3")
    lines[-2] = bytes(row)
    return b"\n".join(lines)


def _cli():
    spec = importlib.util.spec_from_file_location("pr37_stage2_baseline_cli", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# A. Package identity and content
# ======================================================================
def test_package_files_identity_and_designation(base):
    pkg = base["pkg"]
    tables = [spec[0] for spec in b.TABLES.values()]
    assert list(pkg.files) == [b.MANIFEST, b.SIDECAR, b.REPORT, b.DEFINITIONS, *tables]
    m = pkg.manifest
    assert (m["format"], m["format_version"], m["baseline_id"], m["label"], m["designation"],
            m["status"]) == (b.FORMAT, b.FORMAT_VERSION, "pr37_stage2_pre_correction_v1",
                             "Pre-Correction Stage 2 Baseline", "pre_correction", "frozen")
    assert set(m["outputs"]) == set(pkg.files) - {b.MANIFEST, b.SIDECAR}
    for rel, rec in m["outputs"].items():
        assert rec["sha256"] == hashlib.sha256(pkg.files[rel]).hexdigest()
        assert rec["bytes"] == len(pkg.files[rel])
    assert pkg.files[b.SIDECAR] == f"{pkg.manifest_sha256}  {b.MANIFEST}\n".encode()
    assert b.MANIFEST not in m["outputs"] and b.SIDECAR not in m["outputs"]   # no self-hash


def test_manifest_records_population_identities_sources_and_settings(base):
    m, src = base["pkg"].manifest, base["sources"]
    trace = d.build_router_trace(src.glass, src.rf, src.ref)
    fold = np.asarray(trace.fold_id)
    assert m["population"] == {
        "training_rows": trace.n, "positives": trace.n_positives,
        "negatives": trace.n - trace.n_positives, "outer_folds": 5,
        "rows_per_fold": [int((fold == k).sum()) for k in range(5)],
        "positives_per_fold": [int(np.asarray(trace.y)[fold == k].sum()) for k in range(5)]}
    ids = m["identities"]
    assert ids["router_split_fingerprint"] == {
        "value": src.ref.router_split_fingerprint, "status": "validated",
        "detail": ids["router_split_fingerprint"]["detail"]}
    assert ids["train_label_fingerprint"]["value"] == src.ref.train_label_fingerprint
    assert ids["stage_io_split_fingerprint"]["status"] == "attested"
    assert ids["stage_io_split_fingerprint"]["value"] == src.ref.stage_io_split_fingerprint
    assert ids["fold_plan"]["fingerprint"] == dw.fold_fp(np.asarray(src.glass.fold_id))
    assert ids["feature_matrix_hash"]["status"] == "not_recorded"
    assert [s["role"] for s in m["sources"]] == list(b.SOURCE_ROLES)
    assert all(s["snapshot_attested"] is True and len(s["sha256"]) == 64 for s in m["sources"])
    assert m["snapshot"] == {"label": "synthetic_snapshot/SHA256SUMS.txt", "sha256": "a" * 64,
                             "entries": 37}
    up = m["upstream_operating_points"]
    assert up["lr"]["threshold"] == src.lr.thresholds["lr.in_stage_f2_cv"].value
    assert tuple(up["ebm"]["fold_thresholds"].values()) == dw.EBM_NESTED
    assert set(up) == {"lr", "ebm", "union", "intersection"}
    s = m["analysis_settings"]
    assert (s["score_bins"], s["joint_bins"], s["min_support"],
            s["threshold_band_within_fold_percentile"], s["stratifiers"]) == (
        ov.SCORE_BINS, ov.JOINT_BINS, ov.MIN_SUPPORT, ov.BOUNDARY_BAND, list(ov.STRATIFIERS))
    assert s["matched_rate"]["tie_break"] == TIE_BREAK_RULE
    assert set(m["code"]["files"]) == set(b.CODE_FILES)
    assert m["code"]["files"]["shared/stage2/diagnostics/baseline.py"] == b.lf_sha256(
        (Path(BANK_ROOT) / "shared/stage2/diagnostics/baseline.py").read_bytes())
    assert [x["topic"] for x in m["disclosures"]] == list(REQUIRED_DISCLOSURES)
    assert m["historical_test_exposure"] == d.HISTORICAL_TEST_EXPOSURE
    assert all(c["passed"] for c in m["validation"]["consistency"])
    assert m["validation"]["trace_checks"]["passed"] == m["validation"]["trace_checks"]["total"]


def test_constraint_provenance_compares_recorded_constraints_with_glass(base):
    cp = base["pkg"].manifest["rf_constraint_provenance"]
    rf = base["sources"].rf
    assert cp["constraint_source"] == rf.metadata["constraint_source"] == "match_glass_oof"
    assert cp["recorded_constraints"] == dict(rf.metadata["constraints"])
    by = {c["constraint"]: c for c in cp["derivation_check"]}
    g = base["sources"].glass
    p1 = np.asarray(g.pass1)
    y = np.asarray(g.y) == 1
    npv = by["min_precision_not_subscribe"]
    assert (npv["glass_numerator"], npv["glass_denominator"]) == (int((p1 & ~y).sum()),
                                                                  int(p1.sum()))
    want = "equal" if abs(npv["recorded"] - npv["glass_numerator"] / npv["glass_denominator"]) \
        <= 1e-12 else "differs"
    assert npv["status"] == want
    assert by["max_subscriber_leakage_absolute"]["glass_numerator"] == int((p1 & y).sum())


def test_report_is_labelled_and_states_the_required_boundaries(base):
    text = base["pkg"].files[b.REPORT].decode("utf-8")
    for needle in ("Pre-Correction Stage 2 Baseline", "pr37_stage2_pre_correction_v1",
                   "**pre_correction**", "before any corrected-selector change",
                   "not a target to optimise", "Useful incremental predictive information",
                   "any Stage 4 benefit, remain unproven", "within-bin score variation",
                   "are not statistical significance tests",
                   "it is not, and does not predict, corrected-selector performance",
                   "## 4. Step 2D findings (validated summary)",
                   "## 5. Known limitations of the exported Step 2D tables",
                   "Calibrated EBM expectations explain several observed disagreement populations"):
        assert needle in text, needle
    xs = _csv(base["pkg"], "ratios")
    row = xs[(xs["section"] == "2D.crosstab_summary") & (xs["fold"] == "all")
             & (xs["metric"] == "cross_routed_agreement")].iloc[0]
    assert f"{int(row['numerator']):,}/{int(row['denominator']):,}" in text


# ======================================================================
# B. Exported tables against the validated APIs
# ======================================================================
def test_three_state_accounting_reconciles(base):
    st = _csv(base["pkg"], "states")
    for (arm, fold), grp in st.groupby(["arm", "fold"], sort=False):
        assert list(grp["state"]) == list(d.ROUTER_STATES)
        assert grp["n"].astype(int).sum() == int(grp["rows"].iloc[0])
        assert (grp["positives"].astype(int) + grp["negatives"].astype(int)
                == grp["n"].astype(int)).all()
    pooled = st[st["fold"] == "all"]
    per_fold = st[st["fold"] != "all"]
    for arm in ARMS:
        for s in d.ROUTER_STATES:
            k = (per_fold["arm"] == arm) & (per_fold["state"] == s)
            p = pooled[(pooled["arm"] == arm) & (pooled["state"] == s)].iloc[0]
            assert per_fold[k]["n"].astype(int).sum() == int(p["n"])


@pytest.mark.parametrize("section,producer", [
    ("2A.routing_rates", lambda t, s, rf: d.routing_rates(t)),
    ("2B.pass1_population", lambda t, s, rf: d.pass1_population(t)),
    ("2B.pass1_suppression", lambda t, s, rf: d.pass1_suppression_report(t, s.lr, s.ebm).table),
    ("2C.pass2_population", lambda t, s, rf: d.pass2_population(t)),
    ("2C.pass2_frozen", lambda t, s, rf: d.pass2_frozen_contribution(t, s.lr, s.ebm).table),
    ("2C.pass2_matched", lambda t, s, rf: d.pass2_matched_contribution(t, s.lr, s.ebm).table),
    ("2D.router_reference_agreement", lambda t, s, rf: d.router_reference_agreement(t, rf)),
    ("2D.crosstab_summary", lambda t, s, rf: d.crosstab_summary(t)),
])
def test_ratio_rows_round_trip_the_validated_tables(base, section, producer):
    s = base["sources"]
    t = d.build_router_trace(s.glass, s.rf, s.ref)
    api = producer(t, s, d.primary_reference_flags(s.lr, s.ebm)).reset_index(drop=True)
    out = _csv(base["pkg"], "ratios")
    out = out[out["section"] == section].reset_index(drop=True)
    assert len(out) == len(api)
    assert list(out["metric"]) == list(api["metric"])
    assert list(out["status"]) == list(api["status"])
    assert list(out["fold"]) == [str(f) for f in api["fold"]]
    num = [("" if pd.isna(x) else str(int(x))) for x in api["numerator"]]
    den = [("" if pd.isna(x) else str(int(x))) for x in api["denominator"]]
    assert list(out["numerator"]) == num and list(out["denominator"]) == den
    got = out["value"].replace("", np.nan).astype(float).to_numpy()
    assert np.allclose(got, api["value"].astype(float).to_numpy(), rtol=1e-11, atol=0,
                       equal_nan=True)


def test_every_pooled_row_is_the_sum_of_its_folds(base):
    xs = _csv(base["pkg"], "ratios")
    xs = xs[xs["status"].isin(["count", "defined", "undefined_empty_denominator"])]
    keys = ["section", "arm", "reference", "population", "group", "score", "metric"]
    checked = 0
    for _, grp in xs.groupby(keys, sort=False):
        pooled, folds = grp[grp["fold"] == "all"], grp[grp["fold"] != "all"]
        if len(pooled) != 1 or len(folds) != 5:
            continue
        checked += 1
        assert int(pooled["numerator"].iloc[0]) == folds["numerator"].astype(int).sum()
        if pooled["status"].iloc[0] != "count":
            assert int(pooled["denominator"].iloc[0]) == folds["denominator"].astype(int).sum()
    assert checked > 500


def test_pass1_suppression_and_leakage_partition(base):
    xs = _csv(base["pkg"], "ratios")
    sup = xs[(xs["section"] == "2B.pass1_suppression") & (xs["fold"] == "all")]
    pop = xs[(xs["section"] == "2B.pass1_population") & (xs["fold"] == "all")]
    for arm in ARMS:
        p = pop[pop["arm"] == arm].set_index("metric")["numerator"].astype(int)
        assert p["pass1_rows"] == p["pass1_negatives"] + p["pass1_positives"]
        leak = pop[(pop["arm"] == arm) & (pop["metric"] == "pass1_leakage_share")].iloc[0]
        assert (int(leak["numerator"]), int(leak["denominator"])) == (
            p["pass1_positives"], p["positives"])
        for ref in ("lr", "ebm", "union", "intersection"):
            v = sup[(sup["arm"] == arm) & (sup["reference"] == ref)].set_index("metric")
            n = v["numerator"].replace("", "0").astype(int)
            assert n["intercepted_flags"] == n["intercepted_fp"] + n["suppressed_tp"]
            assert n["reference_flags"] == n["intercepted_flags"] + n["retained_flags"]


def test_pass2_unique_contributions_partition(base):
    xs = _csv(base["pkg"], "ratios")
    fz = xs[(xs["section"] == "2C.pass2_frozen") & (xs["fold"] == "all")]
    bd = _csv(base["pkg"], "pass2_breakdown")
    x = _csv(base["pkg"], "pass2_cross_router")
    xu = x[(x["reference"] == "union") & (x["fold"] == "all")].iloc[0].astype(str)
    for arm in ARMS:
        v = fz[(fz["arm"] == arm) & (fz["reference"] == "union")].set_index("metric")
        n = v["numerator"].astype(int)
        assert n["pass2_tp"] == n["unique_tp"] + n["overlap_tp"]
        assert n["pass2_fp"] == n["unique_fp"] + n["overlap_fp"]
        b_ = bd[(bd["arm"] == arm) & (bd["fold"] == "all")].iloc[0]
        assert int(b_["tp_neither"]) == n["unique_tp"]
        key = "glass" if arm == "glass" else "rf"
        assert int(xu[f"{key}_unique_tp"]) == n["unique_tp"]
    assert int(xu["glass_unique_tp"]) == int(xu["shared_tp"]) + int(xu["glass_only_tp"])
    assert int(xu["rf_unique_tp"]) == int(xu["shared_tp"]) + int(xu["rf_only_tp"])


def test_matched_selection_table(base):
    ms = _csv(base["pkg"], "matched_selection")
    cp = _csv(base["pkg"], "common_population")
    assert len(ms) == 2 * 2 * 5
    for c in ("target_count", "selected_rows", "strict_count", "inclusive_count",
              "tie_block", "tie_broken_rows"):
        ms[c] = ms[c].astype(int)
    assert (ms["selected_rows"] == ms["target_count"]).all()
    assert ((ms["strict_count"] <= ms["target_count"])
            & (ms["target_count"] <= ms["inclusive_count"])).all()
    assert (ms["tie_broken_rows"] == ms["target_count"] - ms["strict_count"]).all()
    for arm, col in (("glass", "glass_pass2_in_common"), ("blackbox", "rf_pass2_in_common")):
        k = ms[(ms["arm"] == arm) & (ms["model"] == "lr")]["target_count"].tolist()
        assert k == cp[cp["fold"] != "all"][col].astype(int).tolist()


def test_tied_scores_are_broken_deterministically(base):
    w = base["w"]
    s = base["sources"]
    tied = _with_score(s.lr, "calibrated_probability",
                       np.round(np.asarray(s.lr.scores["calibrated_probability"].values), 1))
    src = replace(s, lr=tied)
    p1, p2 = b.build_baseline(src, base["code"]), b.build_baseline(src, base["code"])
    assert dict(p1.files) == dict(p2.files)
    ms = _csv(p1, "matched_selection")
    lr = ms[ms["model"] == "lr"]
    assert (lr["tie_block"].astype(int) > 1).any() and (lr["tie_broken_rows"].astype(int) > 0).any()
    assert (lr["selected_rows"] == lr["target_count"]).all()
    assert w is base["w"]


def test_agreement_denominators_distinguish_routed_only(base):
    xs = _csv(base["pkg"], "ratios")
    sm = xs[(xs["section"] == "2D.crosstab_summary")]
    for fold, grp in sm.groupby("fold"):
        v = grp.set_index("metric")
        n = int(v.loc["cross_state_agreement", "denominator"])
        assert int(v.loc["cross_routed_agreement", "denominator"]) == int(
            v.loc["cross_both_routed_rate", "numerator"])
        assert "routed only" in v.loc["cross_routed_agreement", "label"]
        for m in ("cross_state_agreement", "cross_both_routed_rate", "cross_opposite_rate",
                  "cross_one_abstains_rate", "cross_both_abstain_rate"):
            assert int(v.loc[m, "denominator"]) == n
    defs = json.loads(base["pkg"].files[b.DEFINITIONS])
    assert "routed only" in defs["metrics"]["cross_routed_agreement"]["denominator"]


def test_unavailable_signals_are_exported_as_unavailable(base):
    sig = _csv(base["pkg"], "signal_availability").set_index(["arm", "signal"])
    for key in (("glass", "Pass 1 candidate-rule matches per row"),
                ("glass", "Pass 2 candidate-rule matches per row"),
                ("blackbox", "Pass 2 outer score"),
                ("both", "analysis: GLASS Pass 1 vs Pass 2 candidate / signal overlap"),
                ("both", "analysis: RF Pass 1 vs Pass 2 signal overlap")):
        assert sig.loc[key, "status"] == "not_verifiable", key
        assert sig.loc[key, "rows"] == ""
    assert sig.loc[("glass", "fold RF passed to the GLASS fit"), "status"] == "code_level_only"
    assert sig.loc[("both", "analysis: final-state Pass 1 / Pass 2 overlap"), "rows"] == "0"


def test_undefined_ratios_stay_undefined_never_zero(base):
    w, s = base["w"], base["sources"]
    g = s.glass
    rows = np.flatnonzero(np.asarray(g.pass2) & (np.asarray(g.fold_id) == 0))
    src = replace(s, glass=_reroute(g, rows, "uncertain"))
    pkg = b.build_baseline(src, base["code"])
    xs = _csv(pkg, "ratios")
    r = xs[(xs["section"] == "2C.pass2_population") & (xs["arm"] == "glass")
           & (xs["fold"] == "0") & (xs["metric"] == "pass2_precision")].iloc[0]
    assert (r["numerator"], r["denominator"], r["value"], r["status"]) == (
        "0", "0", "", "undefined_empty_denominator")
    st = _csv(pkg, "states")
    e = st[(st["arm"] == "glass") & (st["fold"] == "0") & (st["state"] == P2)].iloc[0]
    assert (e["n"], e["positive_rate"], e["positive_rate_status"]) == (
        "0", "", "undefined_empty_denominator")
    assert w is base["w"]


def test_every_table_column_and_ratio_is_defined(base):
    defs = json.loads(base["pkg"].files[b.DEFINITIONS])
    assert set(defs["tables"]) == set(b.TABLES)
    for name, spec in defs["tables"].items():
        header = list(_csv(base["pkg"], name).columns)
        assert list(spec["columns"]) == header, name
        assert all(spec["columns"][c].strip() for c in header)
        assert spec["file"] == b.TABLES[name][0]
    xs = _csv(base["pkg"], "ratios")
    ratio_metrics = set(xs[xs["status"].isin(["defined", "undefined_empty_denominator"])]
                        ["metric"])
    for m in ratio_metrics:
        assert m in d.METRIC_DEFINITIONS, m
        assert (defs["metrics"][m]["numerator"], defs["metrics"][m]["denominator"]) == \
            d.METRIC_DEFINITIONS[m]
    for m in set(xs["metric"]):
        assert defs["metrics"][m]["labels"], m
    assert set(defs["metrics"]) >= set(d.METRIC_DEFINITIONS)


# ======================================================================
# C. Determinism
# ======================================================================
def test_rebuild_is_byte_identical(base):
    again = b.build_baseline(base["sources"], base["code"])
    assert dict(again.files) == dict(base["pkg"].files)


def test_byte_identical_across_processes_and_hash_seeds(base):
    code = ("import hashlib\n"
            "from tests.stage2.test_router_baseline import _synthetic_package\n"
            "p = _synthetic_package()\n"
            "for k, v in p.files.items(): print(k, hashlib.sha256(v).hexdigest())\n")
    outs = []
    for seed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(BANK_ROOT)}
        res = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(BANK_ROOT), env=env,
                             capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append(res.stdout)
    here = "".join(f"{k} {hashlib.sha256(v).hexdigest()}\n" for k, v in base["pkg"].files.items())
    assert outs[0] == outs[1] == here


def test_no_timestamps_absolute_paths_or_carriage_returns(base):
    today = datetime.date.today().isoformat()
    for rel, data in base["pkg"].files.items():
        text = data.decode("utf-8")
        assert today not in text, rel
        assert str(Path(BANK_ROOT).resolve()) not in text, rel
        assert str(Path.home()) not in text or len(str(Path.home())) < 4, rel
        assert "\r" not in text and not b._ABS_PATH.search(text), rel


# ======================================================================
# D. Preservation: write, verify, compare
# ======================================================================
def test_first_write_is_atomic_and_verifies(base, tmp_path):
    parent = tmp_path / "baselines"
    parent.mkdir()
    res = b.write_baseline(base["pkg"], parent)
    target = parent / base["pkg"].baseline_id
    assert (res.status, Path(res.directory)) == ("written", target)
    assert [p.name for p in parent.iterdir()] == [base["pkg"].baseline_id]   # no partial dirs
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) \
        == sorted(base["pkg"].files)
    for rel, data in base["pkg"].files.items():
        assert (target / rel).read_bytes() == data
    rep = b.verify_baseline(target)
    assert rep.ok and rep.files_checked == len(base["pkg"].files) - 2


def test_identical_rerun_writes_nothing(base, frozen):
    before = _tree(frozen.parent)
    res = b.write_baseline(base["pkg"], frozen.parent)
    assert res.status == "identical"
    assert _tree(frozen.parent) == before


def test_divergent_overwrite_is_refused(base, frozen):
    src = replace(base["sources"], snapshot=b.SnapshotRecord("other/SHA256SUMS.txt", "b" * 64, 36))
    other = b.build_baseline(src, base["code"])
    before = _tree(frozen.parent)
    with pytest.raises(b.BaselineConflict) as exc:
        b.write_baseline(other, frozen.parent)
    assert exc.value.comparison.status == "differs"
    assert "snapshot" in exc.value.comparison.manifest_sections
    assert "nothing was written" in str(exc.value)
    assert _tree(frozen.parent) == before


def test_partial_existing_package_is_refused(base, tmp_path):
    parent = tmp_path / "baselines"
    (parent / base["pkg"].baseline_id).mkdir(parents=True)
    (parent / base["pkg"].baseline_id / b.MANIFEST).write_bytes(base["pkg"].files[b.MANIFEST])
    before = _tree(parent)
    with pytest.raises(b.BaselineConflict):
        b.write_baseline(base["pkg"], parent)
    assert _tree(parent) == before


def test_failed_write_leaves_no_package(base, tmp_path, monkeypatch):
    parent = tmp_path / "baselines"
    parent.mkdir()
    real, calls = b._fsync_write, []

    def flaky(path, data):
        calls.append(path)
        if len(calls) == 5:
            raise OSError("disk full (simulated)")
        real(path, data)

    monkeypatch.setattr(b, "_fsync_write", flaky)
    with pytest.raises(OSError, match="simulated"):
        b.write_baseline(base["pkg"], parent)
    assert list(parent.iterdir()) == []


def test_missing_parent_directory_is_refused(base, tmp_path):
    with pytest.raises(b.BaselineError, match="does not exist"):
        b.write_baseline(base["pkg"], tmp_path / "nowhere")
    assert not (tmp_path / "nowhere").exists()


def test_verify_detects_a_modified_table(frozen):
    rel = b.TABLES["crosstab"][0]
    data = bytearray((frozen / rel).read_bytes())
    data[-3] = ord("9") if data[-3] != ord("9") else ord("8")
    (frozen / rel).write_bytes(bytes(data))
    rep = b.verify_baseline(frozen)
    assert not rep.ok and any(p.startswith(rel) and "content differs" in p for p in rep.problems)


def test_verify_detects_missing_extra_and_manifest_edits(base, tmp_path):
    def fresh(name):
        parent = tmp_path / name
        parent.mkdir()
        return Path(b.write_baseline(base["pkg"], parent).directory)

    d1 = fresh("missing")
    (d1 / b.TABLES["states"][0]).unlink()
    assert any("missing" in p for p in b.verify_baseline(d1).problems)
    d2 = fresh("extra")
    (d2 / "tables" / "notes.txt").write_text("x")
    assert any("unexpected file" in p for p in b.verify_baseline(d2).problems)
    d3 = fresh("manifest")
    man = json.loads((d3 / b.MANIFEST).read_bytes())
    man["label"] = "Something else"
    (d3 / b.MANIFEST).write_bytes(json.dumps(man, indent=2).encode())
    assert any(b.SIDECAR in p for p in b.verify_baseline(d3).problems)
    d4 = fresh("sidecar")
    (d4 / b.SIDECAR).unlink()
    assert any(b.SIDECAR in p for p in b.verify_baseline(d4).problems)


@pytest.mark.parametrize("edit,needle", [
    (lambda m: m.update(designation="post_correction"), "designation"),
    (lambda m: m.update(baseline_id="pr37_stage2_pre_correction_v2"), "directory name"),
    (lambda m: m.update(format_version=2), "format_version"),
    (lambda m: m.update(outputs={}), "no outputs"),
])
def test_verify_rejects_a_mislabelled_manifest(base, tmp_path, edit, needle):
    parent = tmp_path / "p"
    parent.mkdir()
    target = Path(b.write_baseline(base["pkg"], parent).directory)
    man = json.loads((target / b.MANIFEST).read_bytes())
    edit(man)
    mb = (json.dumps(man, indent=2) + "\n").encode()
    (target / b.MANIFEST).write_bytes(mb)
    (target / b.SIDECAR).write_bytes(f"{hashlib.sha256(mb).hexdigest()}  {b.MANIFEST}\n".encode())
    rep = b.verify_baseline(target)
    assert not rep.ok and any(needle in p for p in rep.problems), rep.problems


def test_verify_rejects_invalid_json_and_missing_directories(tmp_path):
    assert not b.verify_baseline(tmp_path / "absent").ok
    target = tmp_path / b.BASELINE_ID
    target.mkdir()
    assert any("missing" in p for p in b.verify_baseline(target).problems)
    (target / b.MANIFEST).write_text("{not json")
    assert any("invalid" in p for p in b.verify_baseline(target).problems)
    with pytest.raises(b.BaselineError, match="not valid"):
        b.read_manifest(target)


def test_check_catches_a_consistently_rehashed_edit_that_verify_cannot(base, frozen):
    rel = b.TABLES["pass2_cross_router"][0]
    _forge(frozen, rel, _edit_last_digit((frozen / rel).read_bytes()))
    assert b.verify_baseline(frozen).ok                      # integrity alone cannot see it
    cmp = b.compare_baseline(base["pkg"], frozen)
    assert cmp.status == "differs" and rel in cmp.differing and not cmp.reproduced


def test_verify_and_compare_write_nothing(base, frozen):
    before = _tree(frozen.parent)
    assert b.verify_baseline(frozen).ok
    assert b.compare_baseline(base["pkg"], frozen).status == "identical"
    assert _tree(frozen.parent) == before


def test_compare_reports_changed_source_identities(base, frozen):
    src = replace(base["sources"], artifacts=_artifacts(**{b.ROLE_GLASS: "f" * 64}))
    cmp = b.compare_baseline(b.build_baseline(src, base["code"]), frozen)
    assert cmp.status == "differs" and "sources" in cmp.manifest_sections
    assert cmp.differing == (b.REPORT,)                      # the report shows the hashes;
    #                                                          every table is unchanged
    assert any("sources" in line for line in cmp.lines())


def test_compare_reports_changed_fingerprints(base, frozen):
    w = base["w"]
    ref = d.build_train_reference(index_train=w["idx_tr"], y_train=w["GS"]["y_train"],
                                  index_test=w["idx_te"],
                                  attested_stage_io_split_fingerprint="0" * 12,
                                  attestation_source="a different attestation")
    src = replace(base["sources"], ref=ref)
    cmp = b.compare_baseline(b.build_baseline(src, base["code"]), frozen)
    assert cmp.status == "differs" and "identities" in cmp.manifest_sections


def test_code_drift_is_reported_separately(base, frozen):
    files = dict(base["code"].files)
    files["shared/stage2/diagnostics/metrics.py"] = "0" * 64
    pkg = b.build_baseline(base["sources"], b.CodeIdentity(base["code"].method, files))
    cmp = b.compare_baseline(pkg, frozen)
    assert cmp.status == "reproduced_with_code_drift" and cmp.reproduced
    assert cmp.code_changes == ("shared/stage2/diagnostics/metrics.py: "
                                f"{base['code'].files['shared/stage2/diagnostics/metrics.py'][:12]}"
                                f" -> {'0' * 12}",)
    with pytest.raises(b.BaselineConflict):                  # generate never rewrites a package
        b.write_baseline(pkg, frozen.parent)


# ======================================================================
# E. Input guards
# ======================================================================
def test_test_side_fields_refuse_the_build(base):
    s = base["sources"]
    bad = replace(s.lr, source=MappingProxyType({**dict(s.lr.source),
                                                 "fields_read": ("train_proba_oof",
                                                                 "test_proba")}))
    with pytest.raises(b.BaselineError, match="test-side or forbidden"):
        b.build_baseline(replace(s, lr=bad), base["code"])
    for src in base["pkg"].manifest["sources"]:
        for f in src["fields_read"] or ():
            assert not any(t in f.lower() for t in d.FORBIDDEN_FIELD_TOKENS)


def test_views_without_a_boundary_record_are_refused(base):
    s = base["sources"]
    bare = replace(s.ebm, source=MappingProxyType({}))
    with pytest.raises(b.BaselineError, match="fields_read"):
        b.build_baseline(replace(s, ebm=bare), base["code"])


@pytest.mark.parametrize("artifacts,needle", [
    (lambda a: a[:-1], "exactly once"),
    (lambda a: a + a[:1], "exactly once"),
    (lambda a: (replace(a[0], path="C:\\Users\\me\\models\\x.joblib"),) + a[1:], "relative"),
    (lambda a: (replace(a[0], path="/home/me/models/x.joblib"),) + a[1:], "relative"),
    (lambda a: (replace(a[0], path="../outside.joblib"),) + a[1:], "relative"),
    (lambda a: (replace(a[0], sha256="ABC"),) + a[1:], "64 lowercase"),
])
def test_source_records_are_validated(base, artifacts, needle):
    src = replace(base["sources"], artifacts=artifacts(base["sources"].artifacts))
    with pytest.raises(b.BaselineError, match=needle):
        b.build_baseline(src, base["code"])


def test_absolute_paths_in_free_text_are_scrubbed(base):
    s = base["sources"]
    meta = dict(s.rf.metadata)
    meta["rf_params_origin"] = "c:\\Users\\someone\\projects\\rf_result.joblib"
    meta["note"] = "copied from /home/someone/models/rf.joblib on a laptop"
    pkg = b.build_baseline(replace(s, rf=replace(s.rf, metadata=MappingProxyType(meta))),
                           base["code"])
    text = pkg.files[b.MANIFEST].decode()
    assert "someone" not in text and b.LOCAL_PATH in text
    assert "recorded in the artifact as a local absolute path" in \
        pkg.manifest["routers"]["blackbox"]["metadata"]["rf_params_origin"]


def test_failed_consistency_check_builds_nothing(base, monkeypatch):
    original = ov.group_observed_expected

    def broken(trace, strata, groups):
        out = original(trace, strata, groups)
        out.loc[out["group"] == "glass_pass1", "observed"] += 1
        return out

    monkeypatch.setattr(b.ov, "group_observed_expected", broken)
    with pytest.raises(b.BaselineError, match="consistency checks failed"):
        b.build_baseline(base["sources"], base["code"])


# ======================================================================
# F. Source collection (validated loader) and the CLI
# ======================================================================
class _FakeReal:
    """The ``Real`` interface over synthetic artifacts written below a stand-in bank root."""

    def __init__(self, files, ref, root, sums):
        self.files, self._ref, self.root, self.sums = files, ref, root, sums

    def roots(self):
        return {k: self.files[k] for k in ("meta_ebm", "meta_ebm_on_blackbox")}

    def paths(self, key):
        return ({"stage1": self.files["lr"], "stage2": self.files["glass"],
                 "stage3": self.files["ebm"]} if key == "meta_ebm"
                else {"stage2": self.files["rf"]})

    def ref(self):
        return self._ref

    def sha(self, p):
        return self.sums["bank_pipeline/" + Path(p).resolve().relative_to(self.root).as_posix()]

    def view(self, which):
        p = self.files[which]
        if which in ("glass", "rf"):
            return d.extract_router_view(p, self._ref, expected_arm={"glass": "glass",
                                                                     "rf": "blackbox"}[which],
                                         expected_sha256=self.sha(p))
        if which == "lr":
            return d.extract_lr_view(p, self._ref, expected_sha256=self.sha(p))
        return d.extract_ebm_view(p, self._ref, expected_sha256=self.sha(p))

    def rf_result_path(self):
        return self.files["rf_result"]


def _disk_world(tmp_path, *, corrupt_arbiter=False):
    w = dw.world()
    root = (tmp_path / "bank").resolve()
    (root / "models" / "x").mkdir(parents=True)
    files = {}
    for name in ("glass", "rf", "lr", "ebm", "rf_result"):
        files[name] = root / "models" / "x" / f"{name}.joblib"
        joblib.dump(w[name], files[name])
    for name in ("meta_ebm", "meta_ebm_on_blackbox"):
        files[name] = root / "models" / "x" / f"{name}.joblib"
        joblib.dump({"arbiter": name}, files[name])
    snap = tmp_path / "snap_2026" / "SHA256SUMS.txt"
    snap.parent.mkdir()
    lines = []
    for p in files.values():
        sha = d.file_sha256(p)
        if corrupt_arbiter and p.name == "meta_ebm.joblib":
            sha = "0" * 64
        lines.append(f"{sha}  bank_pipeline/{p.relative_to(root).as_posix()}")
    snap.write_text("\n".join(lines) + "\n", encoding="ascii")
    return w, root, files, snap


def test_collect_baseline_sources_hashes_relativises_and_attests(tmp_path):
    w, root, files, snap = _disk_world(tmp_path)
    real = _FakeReal(files, dw.reference(w), root, ra.read_sums(snap))
    src = ra.collect_baseline_sources(real, bank_root=root, sums_path=snap)
    assert [a.role for a in src.artifacts] == list(b.SOURCE_ROLES)
    by = {a.role: a for a in src.artifacts}
    assert by[b.ROLE_GLASS].path == "models/x/glass.joblib"
    assert by[b.ROLE_ARBITER_RF].path == "models/x/meta_ebm_on_blackbox.joblib"
    for a in src.artifacts:
        assert a.sha256 == d.file_sha256(root / a.path) and a.snapshot_attested is True
        assert a.size == (root / a.path).stat().st_size
    assert src.snapshot == b.SnapshotRecord("snap_2026/SHA256SUMS.txt", d.file_sha256(snap), 7)
    assert src.data_files == ()                        # no raw dataset under the stand-in root
    pkg = b.build_baseline(src, b.code_identity(BANK_ROOT))
    assert pkg.manifest["sources"][2]["fields_read"]   # real boundary extraction records


def test_collect_refuses_a_source_that_differs_from_the_snapshot(tmp_path):
    w, root, files, snap = _disk_world(tmp_path, corrupt_arbiter=True)
    real = _FakeReal(files, dw.reference(w), root, ra.read_sums(snap))
    with pytest.raises(AssertionError, match="differs from the snapshot"):
        ra.collect_baseline_sources(real, bank_root=root, sums_path=snap)


def test_collect_refuses_sources_outside_the_bank_root(tmp_path):
    w, root, files, snap = _disk_world(tmp_path)
    real = _FakeReal(files, dw.reference(w), root, ra.read_sums(snap))
    with pytest.raises(AssertionError, match="not under"):
        ra.collect_baseline_sources(real, bank_root=tmp_path / "elsewhere", sums_path=snap)


def test_pinned_real_uses_the_recorded_roots(tmp_path, monkeypatch):
    monkeypatch.delenv(ra.SUMS_ENV, raising=False)
    roots = {"meta_ebm": tmp_path / "a.joblib", "meta_ebm_on_blackbox": tmp_path / "b.joblib"}
    pinned = ra.PinnedReal(roots)
    assert isinstance(pinned, ra.Real) and pinned.roots() == roots and pinned.sums == {}


@pytest.fixture
def cli(base, tmp_path):
    mod = _cli()
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("")
    parent = tmp_path / "baselines"
    parent.mkdir()
    calls = []

    def collector(mode, frozen, sums_path):
        calls.append((mode, None if frozen is None else frozen["baseline_id"], sums_path))
        return base["sources"]

    def run(*args, collector=collector, rehash=lambda s: []):
        return mod.main(list(args), collector=collector, rehash=rehash)

    return {"mod": mod, "run": run, "sums": str(sums), "parent": parent,
            "target": parent / b.BASELINE_ID, "calls": calls}


def test_cli_generate_requires_an_attested_snapshot(cli, monkeypatch, capsys):
    monkeypatch.delenv(cli["mod"].SUMS_ENV, raising=False)
    assert cli["run"]("--generate", "--out-parent", str(cli["parent"])) == 2
    assert "refused" in capsys.readouterr().out
    assert cli["run"]("--generate", "--out-parent", str(cli["parent"]),
                      "--sums", str(cli["parent"] / "absent.txt")) == 2
    assert list(cli["parent"].iterdir()) == [] and cli["calls"] == []


def test_cli_generate_then_identical_then_refused(base, cli, capsys):
    run, parent = cli["run"], cli["parent"]
    assert run("--generate", "--out-parent", str(parent), "--sums", cli["sums"]) == 0
    assert "written" in capsys.readouterr().out
    assert b.verify_baseline(cli["target"]).ok
    before = _tree(parent)
    assert run("--generate", "--out-parent", str(parent), "--sums", cli["sums"]) == 0
    assert "nothing was written" in capsys.readouterr().out
    assert _tree(parent) == before
    other = replace(base["sources"], artifacts=_artifacts(**{b.ROLE_EBM: "e" * 64}))
    assert run("--generate", "--out-parent", str(parent), "--sums", cli["sums"],
               collector=lambda *a: other) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert _tree(parent) == before
    assert [c[0] for c in cli["calls"]] == ["generate", "generate"]


def test_cli_verify(cli, capsys):
    run, target = cli["run"], cli["target"]
    assert run("--verify", "--baseline-dir", str(target)) == 1          # nothing frozen yet
    assert run("--generate", "--out-parent", str(cli["parent"]), "--sums", cli["sums"]) == 0
    before = _tree(cli["parent"])
    assert run("--verify", "--baseline-dir", str(target)) == 0
    assert _tree(cli["parent"]) == before
    (target / b.REPORT).write_bytes((target / b.REPORT).read_bytes() + b"\nedited\n")
    capsys.readouterr()
    assert run("--verify", "--baseline-dir", str(target)) == 1
    assert "REPORT.md: content differs" in capsys.readouterr().out


def test_cli_check_reproduces_writes_nothing_and_detects_drift(base, cli, capsys):
    run, target = cli["run"], cli["target"]
    assert run("--check", "--baseline-dir", str(target), "--sums", cli["sums"]) == 1  # absent
    assert run("--generate", "--out-parent", str(cli["parent"]), "--sums", cli["sums"]) == 0
    before = _tree(cli["parent"])
    capsys.readouterr()
    assert run("--check", "--baseline-dir", str(target), "--sums", cli["sums"]) == 0
    assert "reproduced" in capsys.readouterr().out
    assert cli["calls"][-1] == ("check", b.BASELINE_ID, cli["sums"])     # frozen manifest passed
    assert _tree(cli["parent"]) == before
    assert run("--check", "--baseline-dir", str(target), "--sums", cli["sums"],
               rehash=lambda s: ["stage3.ebm: changed during the run"]) == 2

    def boom(*a):
        raise FileNotFoundError("models/ebm.joblib")
    assert run("--check", "--baseline-dir", str(target), "--sums", cli["sums"],
               collector=boom) == 2
    rel = b.TABLES["observed_expected"][0]
    _forge(target, rel, _edit_last_digit((target / rel).read_bytes()))
    capsys.readouterr()
    assert run("--check", "--baseline-dir", str(target), "--sums", cli["sums"]) == 1
    assert f"content differs: {rel}" in capsys.readouterr().out


def test_cli_parent_must_exist(cli, tmp_path):
    assert cli["run"]("--generate", "--out-parent", str(tmp_path / "missing"),
                      "--sums", cli["sums"]) == 1
    assert not (tmp_path / "missing").exists()


def test_gitattributes_keep_package_bytes_exact():
    text = (Path(BANK_ROOT) / b.BASELINES_DIR / ".gitattributes").read_text()
    assert "*/** -text" in text.splitlines()


def test_committed_baseline_packages_verify():
    """Every frozen package in the repository still matches its own manifest."""
    found = sorted((Path(BANK_ROOT) / b.BASELINES_DIR).glob("*/" + b.MANIFEST))
    if not found:
        pytest.skip("no frozen baseline package generated yet (run stage2_baseline.py --generate)")
    for man in found:
        rep = b.verify_baseline(man.parent)
        assert rep.ok, rep.problems
        m = json.loads(man.read_bytes())
        assert m["designation"] == b.DESIGNATION or m["baseline_id"] != b.BASELINE_ID
