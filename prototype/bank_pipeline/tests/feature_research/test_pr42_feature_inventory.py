"""
PR 42 Commit 2: the feature inventory and frozen references
(``feature_research.feature_inventory``, ``feature_research.frozen_references``).

The code-only checks run on this branch's production code; the checks that need
data run on a synthetic bank frame (training rows only, as in the notebook).
The review document ``research_logs/pr42/feature_inventory.md`` must hold
the generated tables with the PR 42 source identity in the historical DAG row.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from feature_research import feature_inventory as FI
from feature_research import frozen_references as FR
from shared.stage2.binning import BINNING_STRATEGY, RF_FEATURES_BINARY
from tests.support.notebooks import BANK_ROOT
from tests.support.synthetic_bank import TARGET, make_bank_frame

DOC = BANK_ROOT / "research_logs" / "pr42" / "feature_inventory.md"


@pytest.fixture(scope="module")
def train():
    frame = make_bank_frame(n=1200, seed=3)
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(int)


# ----------------------------------------------------------------------
# declarations against the code
# ----------------------------------------------------------------------
@pytest.mark.parametrize("check", [FI.check_registries, FI.check_call_sites,
                                   FI.check_fit_protocols])
def test_the_code_only_checks_pass_on_this_branch(check):
    FI.require(check(), check.__name__)


def test_the_inventory_has_one_well_formed_row_per_consumed_feature():
    table = FI.feature_table()
    assert len(table) == sum(len(p.features) for p in FI.POPULATIONS) == 99
    for p in FI.POPULATIONS:
        assert list(table.loc[table["population"] == p.key, "feature"]) == list(p.features)
    assert set(table["target_dependence"]) <= {FI.NONE, FI.FITTED, FI.INHERITED, FI.STAGE_OUTPUT}
    for r in table.itertuples():
        assert set(r.source_variables) <= set(FI.RAW_VARIABLES)
        assert bool(r.target_path) == (r.target_dependence in (FI.FITTED, FI.INHERITED)), r.feature
        assert r.out_of_fit_rule, r.feature
    # the label-dependent features the later commits audit
    dependent = table[table["target_dependence"].isin([FI.FITTED, FI.INHERITED])]
    assert dependent.groupby("population")["feature"].apply(list).to_dict() == {
        "stage1.glass": ["subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month"],
        "stage1.blackbox": ["subscribe_rate_by_euribor_bin", "subscribe_rate_by_dow_month"],
        "stage2.glass": [b for b in RF_FEATURES_BINARY if b.split("_")[0] in ("nsd", "eci", "dow")],
        "stage2.blackbox": [b for b in RF_FEATURES_BINARY
                            if b.split("_")[0] in ("nsd", "eci", "dow")],
        "stage3.glass": ["decay_x_density", "economic_curvature_intensity", "dow_month_encoded"],
        "stage3.blackbox": ["decay_x_density", "economic_curvature_intensity",
                            "dow_month_encoded"],
    }


def test_the_bins_are_generated_from_binning_strategy():
    bins = FI.bin_nodes()
    assert list(bins) == list(RF_FEATURES_BINARY)
    sources = {g["source"] for g in BINNING_STRATEGY.values()}
    assert {b.inputs[0] for b in bins.values()} == sources == set(FI.STAGE2_SOURCES)
    assert all(FI.source_variables(b) for b in bins)


def test_every_population_declares_both_fit_scopes_and_the_out_of_fit_rule():
    for p in FI.POPULATIONS:
        assert set(p.fit) == {"final_fit", "cross_validation", "fold_refit", "applied", "differs"}
        assert all(p.fit.values()), p.key


def test_stage1_separates_the_final_base_model_from_the_final_calibrator():
    for key in ("stage1.glass", "stage1.blackbox"):
        fit = FI.POPULATION_BY_KEY[key].fit
        assert "final base model" in fit["final_fit"] and "full-train feature frame" in fit["final_fit"]
        assert "final calibrator" in fit["final_fit"] and "out-of-fold base scores" in fit["final_fit"]
        assert "final calibrator is fitted on out-of-fold base scores" in fit["differs"]
    for key in ("stage3.glass", "stage3.blackbox"):
        assert "out-of-fold scores of all training rows" in FI.POPULATION_BY_KEY[key].fit["final_fit"]


def test_each_label_dependent_feature_states_how_it_encodes_its_own_fit_rows():
    table = FI.feature_table()
    dependent = table["target_dependence"].isin([FI.FITTED, FI.INHERITED])
    assert table.loc[dependent, "fit_rows"].str.len().gt(0).all()
    assert table.loc[~dependent, "fit_rows"].eq("").all()
    nsd = FI.NODES["neighborhood_subscription_density"].fit_rows
    assert nsd.startswith("only the reference rows' labels contribute")
    assert "every other fit row is transformed against the reference set" in nsd
    assert FI.NODES["decay_x_density"].fit_rows.startswith("as neighborhood_subscription_density")
    assert {FI.bin_nodes()[b].fit_rows for b in ("nsd_cold", "nsd_hot")} == {
        "as neighborhood_subscription_density"}
    texts = " ".join(v for p in FI.POPULATIONS for v in p.fit.values())
    assert "in-sample for" not in texts                    # no blanket in-sample claim


def _nsd_frames(min_fit_rows, reference_rows):
    return {"nsd_fit_sizes": pd.DataFrame({"scope": ["a", "b"], "fits": [1, 5],
                                           "min_fit_rows": [32950, min_fit_rows],
                                           "max_fit_rows": [32950, 26360]}),
            "nsd_reference": pd.DataFrame({"implementation": [FI.S2, FI.S3], "fit": ["f", "f"],
                                           "fit_rows": [32950, 32950],
                                           "reference_rows": reference_rows})}


def test_the_nsd_reference_check_passes_only_when_every_fit_samples_5000_rows():
    FI.require(FI.check_nsd_reference(_nsd_frames(21088, [5000, 5000])), "NSD")
    with pytest.raises(AssertionError, match="samples its reference rows"):
        FI.require(FI.check_nsd_reference(_nsd_frames(9000, [5000, 5000])), "NSD")
    with pytest.raises(AssertionError, match="keep 5,000 reference rows"):
        FI.require(FI.check_nsd_reference(_nsd_frames(21088, [5000, 4000])), "NSD")


def test_nsd_fit_sizes_cover_every_production_fit():
    y = pd.Series(np.r_[np.ones(300, int), np.zeros(2700, int)])
    sizes = FI.nsd_fit_sizes(y).set_index("scope")
    assert sizes["fits"].to_dict() == {
        "final fits (Stage 2 and Stage 3)": 1, "Stage 2 outer folds (5)": 5,
        "Stage 3 tuning folds (5)": 5, "Stage 3 reporting folds (10)": 10,
        "Stage 3 inner folds (5 x 5)": 25}
    assert sizes.loc["final fits (Stage 2 and Stage 3)", "min_fit_rows"] == 3000
    assert sizes.loc["Stage 3 inner folds (5 x 5)", "max_fit_rows"] < 3000 * 0.65


# ----------------------------------------------------------------------
# training-row checks (synthetic)
# ----------------------------------------------------------------------
def test_every_declared_node_is_built_and_the_selections_hold(train):
    X, y = train
    frames = FI.build_training_frames(X, y)
    FI.require(FI.check_dag_columns(frames), "DAG columns")
    assert set(frames["nsd_reference"]["implementation"]) == {FI.S2, FI.S3}
    assert (frames["nsd_reference"]["reference_rows"] == len(X)).all()   # below 10,000: all rows
    summary = FI.training_build_summary(frames, y)
    assert set(summary["frame"]) == {"stage1", "stage2_sources", "stage2_bins", "stage3"}
    assert summary.loc[summary["frame"] == "stage2_bins", "active_rows"].notna().all()


def test_the_target_probe_agrees_with_every_declaration(train):
    X, y = train
    probe = FI.probe_target_dependence(X, y, n_rows=300)
    FI.require_probe(probe)
    assert set(probe["implementation"]) == set(FI.IMPLEMENTATIONS)
    stage3 = probe[probe["implementation"] == FI.S3].set_index("column")
    assert stage3.loc["dow_month_encoded", "rows_changed"] > 0
    assert stage3.loc["prior_x_stress", "rows_changed"] == 0


def test_the_target_probe_catches_a_wrong_declaration(train, monkeypatch):
    X, y = train
    node = FI.NODES["cellular_downturn"]
    monkeypatch.setitem(FI.NODES, "cellular_downturn", dataclasses.replace(node, target=FI.FITTED))
    probe = FI.probe_target_dependence(X, y, n_rows=200)
    with pytest.raises(AssertionError, match="cellular_downturn"):
        FI.require_probe(probe)


def test_compare_values_counts_one_sided_missing_and_takes_the_finite_maximum():
    got = FI.compare_values([np.nan, 10], [2, 13])
    assert got["rows_differ"] == 2 and got["max_abs_diff"] == 3.0
    assert got["one_sided_missing"] == 1 and got["both_missing"] == 0
    same = FI.compare_values([np.nan, 1.0, np.inf], [np.nan, 1.0, np.inf])
    assert same["rows_differ"] == 0 and same["both_missing"] == 1 and same["max_abs_diff"] == 0.0
    inf = FI.compare_values([np.inf, 4.0], [7.0, 4.5])
    assert inf["rows_differ"] == 2 and inf["max_abs_diff"] == 0.5     # inf excluded from the max
    assert FI.compare_values([np.nan], [np.nan])["max_abs_diff"] is None
    with pytest.raises(ValueError):
        FI.compare_values([1.0], [1.0, 2.0])


def test_duplicate_parity_runs_on_training_rows(train):
    X, y = train
    parity, binning = FI.duplicate_parity(X, y, folds=3)
    assert all(binning.values())
    scopes = set(parity["scope"])
    assert {s.split(":")[0] for s in scopes} == {"in-sample (all training rows)",
                                                 "held-out (3 shared folds)"}
    held = parity[parity["scope"].str.startswith("held-out")]
    assert (held["rows"] == len(X)).all()               # every training row held out once
    assert {"one_sided_missing", "both_missing", "max_abs_diff"} <= set(parity.columns)
    assert (parity["rows_differ"] >= parity["one_sided_missing"]).all()


def test_duplicate_callers_find_the_known_importers():
    callers = FI.duplicate_callers()
    s2, research = callers["stage2_copy"], callers["research"]
    assert s2["internal"] == ["shared/stage2/feature_engineering.py"]
    assert "shared.stage2.engineer_features" in s2["reexported_by"]
    assert "stage2_crossfit/runner.py" in s2["direct"]["evidence"]
    assert "glass_pipeline/glass_router/pipeline/crossfit.py" in s2["direct"]["production"]
    assert s2["notebooks"]["glass_pipeline/glass_cascade.ipynb"]["10"]["via_reexport"] == [
        "glass_pipeline.glass_router.rf.engineer_features"]
    assert research["direct"]["production"] == {"shared/stage3/feature_engineering.py": [
        "DerivativeFeatureEngineer", "IntegralFeatureEngineer", "TemporalFeatureEngineer",
        "add_crisis_features", "add_overlap_features", "add_prior_features"]}


# ----------------------------------------------------------------------
# the review document
# ----------------------------------------------------------------------
def test_the_review_document_holds_the_current_generated_tables(stage3_dag_transition):
    text = DOC.read_text(encoding="utf-8")
    historical = FI.generated_block(text)
    old, current = stage3_dag_transition
    field = "| Stage 3 feature DAG implementation (LF SHA-256 over its modules) | "
    assert historical.count(field + old + " ") == 1
    assert historical.replace(field + old + " ", field + current + " ") == FI.render_markdown()


def test_dag_pin_detects_a_further_source_edit(stage3_dag_transition, monkeypatch):
    current = stage3_dag_transition[1]
    assert FR.dag_implementation()["sha256_lf_combined"] == current
    original = FR._lf_sha256

    def edited_digest(path):
        if path == BANK_ROOT / "shared/stage3/data.py":
            return hashlib.sha256(path.read_bytes() + b"# unreviewed edit").hexdigest()
        return original(path)

    monkeypatch.setattr(FR, "_lf_sha256", edited_digest)
    assert FR.dag_implementation()["sha256_lf_combined"] != current


# ----------------------------------------------------------------------
# frozen references
# ----------------------------------------------------------------------
def test_the_static_references_match_the_accepted_records():
    refs = {r["name"]: r for r in FR.static_references()}
    assert not [n for n, r in refs.items() if r["status"] == "MISMATCH"]
    assert refs["Stage 2 binning identity (SHA-256)"]["computed"] == (
        "9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376")
    assert refs["Stage 3 block content fingerprint"]["recorded"] == (
        "9cc10fa1f5cf8d858a9168a2169372758102d06283592bed024fb06e1040318c")
    assert refs["EBM interaction pairs (3)"]["status"] == "code only"
    assert "Not verified: the pairs stored in the accepted EBM" in refs["EBM interaction pairs (3)"]["note"]
    assert refs["Stage 3 feature DAG implementation (LF SHA-256 over its modules)"]["status"] == \
        "code only"
    assert refs["Stage 3 block-cache file"]["status"] == "local verification"
    assert {r["status"] for r in refs.values()} <= {"verified", "code only", "recorded only",
                                                    "local verification"}


def test_the_dag_implementation_digest_ignores_line_endings():
    dag = FR.dag_implementation()
    assert "shared.stage3.feature_engineering" in dag["modules"]
    for module, rel in dag["files"].items():
        raw = (BANK_ROOT / rel).read_bytes()
        assert dag["sha256_lf"][module] == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------
# accepted artifacts: boundary extractors only
# ----------------------------------------------------------------------
REALDATA = BANK_ROOT / "tests" / "realdata" / "test_pr42_frozen_references.py"


@pytest.mark.parametrize("path", [Path(FR.__file__), Path(FI.__file__), REALDATA])
def test_pr42_code_never_unpickles_an_artifact(path):
    tree = ast.parse(path.read_bytes())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom)}
    assert not imported & {"joblib", "pickle", "dill", "cloudpickle", "torch"}
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "shared.stage4.adapters" not in froms
    if path == REALDATA:                      # the realdata test reads no artifact bytes itself
        assert "shared.pinned_artifacts" not in froms
    calls = {getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"load", "read_pickle", "_load_bytes"}


def test_frozen_references_open_artifacts_only_through_the_boundary_extractors():
    from shared.stage2.diagnostics import boundary
    tree = ast.parse(Path(FR.__file__).read_bytes())
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "shared.stage4.adapters" not in froms
    assert all(callable(getattr(boundary, fn)) for fn, _ in FR.EXTRACTED_ROLES.values())
    src = inspect.getsource(FR.accepted_artifact_views)
    assert "getattr(boundary, fn_name)" in src and ".load(" not in src


def _tmp_root(tmp_path, files: dict) -> Path:
    """A bank root with the c4 run record and a pins file naming ``files`` (role -> bytes)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rec = tmp_path / FR.RECORDS["c4_run"]
    rec.parent.mkdir(parents=True)
    for suffix in ("", ".sha256"):
        shutil.copy(BANK_ROOT / (FR.RECORDS["c4_run"] + suffix), str(rec) + suffix)
    manifest = json.loads((BANK_ROOT / FR.PINS_FILE).read_text(encoding="utf-8"))
    for role, data in files.items():
        entry = manifest["files"][role]
        path = tmp_path / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    pins = tmp_path / FR.PINS_FILE
    body = json.dumps(manifest).encode("utf-8")
    pins.write_bytes(body)
    Path(str(pins) + ".sha256").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return tmp_path


def _view(block=None):
    prov = [SimpleNamespace(name="file_sha256", status="attested", detail="x")]
    if block is not None:
        prov.append(SimpleNamespace(name="block_fingerprint", status="recorded_only", detail=block))
    return SimpleNamespace(source={"checks": ({"check": "a", "passed": True},),
                                   "fields_read": ("f",)}, provenance=tuple(prov))


def test_absent_artifacts_are_not_available_and_unextractable_roles_are_never_read(tmp_path):
    root = _tmp_root(tmp_path, {})
    rows = {r["role"]: r["status"] for r in FR.accepted_artifact_views(object(), root)}
    assert rows == {"stage1.lr": "not available locally",
                    "stage2.glass_router": "not available locally",
                    "stage2.rf_router": "not available locally",
                    "stage3.ebm": "not available locally",
                    "stage1.mlp": "not read", "stage3.xgb": "not read"}


def test_a_changed_artifact_fails_and_a_pinned_one_goes_through_its_extractor(tmp_path):
    root = _tmp_root(tmp_path, {"stage1.lr": b"lr", "stage3.ebm": b"ebm",
                                "stage2.glass_router": b"glass"})
    (root / json.loads((root / FR.PINS_FILE).read_text())["files"]["stage2.glass_router"]
     ["path"]).write_bytes(b"changed")
    recorded = json.loads((BANK_ROOT / FR.RECORDS["c4_run"]).read_text())[
        "evidence"]["stage3"]["stage3.ebm"]["block_fingerprint"]
    seen = []

    def stub(path, ref, *, expected_arm, expected_sha256):
        seen.append((Path(path).name, expected_arm, expected_sha256))
        return _view(recorded if "ebm" in Path(path).name else None)

    rows = {r["role"]: r for r in FR.accepted_artifact_views(
        "REF", root, extractors={"extract_lr_view": stub, "extract_ebm_view": stub,
                                 "extract_router_view": stub})}
    assert rows["stage1.lr"]["status"] == "passed" and rows["stage1.lr"]["checks"] == 1
    assert rows["stage3.ebm"]["status"] == "passed"
    assert rows["stage3.ebm"]["block_fingerprint"]["match"]
    assert rows["stage2.glass_router"]["status"] == "FAILED"         # bytes differ from the pin
    assert rows["stage2.rf_router"]["status"] == "not available locally"
    assert [s[1] for s in seen] == ["glass", "glass"]                 # the changed file never opened


def test_the_real_boundary_extractors_read_synthetic_artifacts_in_the_accepted_formats(tmp_path):
    """End to end, no stubs: synthetic LR, routers and EBM in the cascade's real formats
    (tests.support.diagnostics_world), pinned by SHA-256, read by the PR 38 extractors."""
    import joblib
    from tests.support import diagnostics_world as dw
    w = dw.world()
    recorded = json.loads((BANK_ROOT / FR.RECORDS["c4_run"]).read_text())[
        "evidence"]["stage3"]["stage3.ebm"]["block_fingerprint"]
    w["ebm"].block_fingerprint = recorded
    payload = {}
    for role, key in (("stage1.lr", "lr"), ("stage2.glass_router", "glass"),
                      ("stage2.rf_router", "rf"), ("stage3.ebm", "ebm")):
        buf = tmp_path / f"{key}.joblib"
        joblib.dump(w[key], buf)                          # writes a synthetic artifact only
        payload[role] = buf.read_bytes()
    root = _tmp_root(tmp_path / "bank", payload)
    rows = {r["role"]: r for r in FR.accepted_artifact_views(dw.reference(w), root)}
    assert {k: r["status"] for k, r in rows.items()} == {
        "stage1.lr": "passed", "stage2.glass_router": "passed", "stage2.rf_router": "passed",
        "stage3.ebm": "passed", "stage1.mlp": "not read", "stage3.xgb": "not read"}
    for role, r in rows.items():
        if r["status"] == "passed":
            assert r["checks"] == r["checks_passed"] > 0 and r["fields_read"], role
            assert not [f for f in r["fields_read"] if "test" in f.lower()], role
    assert "feature_names = shared.stage1 LR_FEATURES" in rows["stage1.lr"]["check_names"]
    assert {"features = shared.stage3 EBM_FEATURES", "feature_fit_scope is 'per_fold'",
            "oof fold_id = fold_plan fold_id",
            "per-fold calibration gates recorded (PR 36 F22)"} <= set(rows["stage3.ebm"]["check_names"])
    assert rows["stage3.ebm"]["block_fingerprint"]["match"]


def test_an_ebm_with_another_block_fingerprint_fails(tmp_path):
    root = _tmp_root(tmp_path, {"stage3.ebm": b"ebm"})
    rows = {r["role"]: r for r in FR.accepted_artifact_views(
        "REF", root, extractors={"extract_ebm_view": lambda *a, **k: _view("0" * 64)})}
    assert rows["stage3.ebm"]["status"] == "FAILED"
    assert rows["stage3.ebm"]["block_fingerprint"]["match"] is False
