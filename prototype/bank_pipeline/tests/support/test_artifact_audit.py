"""
The real-artifact audit itself (tests.support.artifact_audit), on synthetic
artifacts in the saved formats — the default suite, no saved artifacts needed.

Contract: on a complete 2x2 written by the real producers (train_meta_stage,
train_meta_xgb_stage) from saved upstream files, every check passes. A
missing arbiter, a newest arbiter that is invalid (never an older one in its
place), a manifest naming a missing file, a wrong identity or incomplete
evidence, an arbiter written before the upstream record, an upstream file
missing or replaced after the arbiter consumed it, a 2x2 column whose
arbiters read different upstreams, an arbiter whose two upstreams share
outputs, a cascade comparison made with another Meta-EBM, a pickle of the
XGBoost package removed in P4a, and a Stage 3 model that does not reproduce
its saved outputs each fail with the file and the field. The audit looks where the
notebooks save, and the realdata suite runs only when ``-m`` names it.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import blackbox_pipeline.models.meta_xgb.comparison as comparison
from blackbox_pipeline.models.meta_xgb import MetaXGBConfig
from glass_pipeline.meta_ebm import meta_stage as ms
from glass_pipeline.meta_ebm.artifacts import save_meta_ebm
from glass_pipeline.meta_ebm.meta_stage import train_meta_stage
from shared.stage3.artifact import Stage3ArtifactStore
from tests.stage3.conftest import ARMS, run_arm
from tests.support import artifact_audit as audit
from tests.support.artifact_audit import AuditError, ROOTS
from tests.support.meta_xgb_world import run_meta
from tests.support.notebooks import BANK_ROOT
from tests.support.stage4_world import ROUTER_FP, build_world, make_router, make_stage1, make_stage3

RF_REMAINDER = "OOF Pass 1 remainder (RF router design)"


def _upstream(w):
    return {"glass": [make_stage1(w), make_router(w, pass2_population="full"), make_stage3(w)],
            "blackbox": [make_stage1(w, arm="blackbox", model="mlp", wrap=False),
                         make_router(w, layout="extra", band_blend=True, pass2_population=RF_REMAINDER),
                         make_stage3(w, family="xgboost")]}


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    """A bank_pipeline/ tree with a synthetic 2x2 where the notebooks save theirs."""
    root = tmp_path_factory.mktemp("bank_pipeline")
    w = build_world(n=1500, n_test=350)
    files = {}
    for arm, arts in _upstream(w).items():
        (root / "upstream").mkdir(exist_ok=True)
        files[arm] = [str(root / "upstream" / f"{arm}_{s}.joblib") for s in audit.STAGES]
        for art, path in zip(arts, files[arm]):
            joblib.dump(art, path)
    folder = {k: str(root / r.folder) for k, r in ROOTS.items()}
    paths = {}
    with pytest.MonkeyPatch.context() as mp:          # train_meta_stage takes no extra_fingerprints
        real = ms.load_stage_outputs
        mp.setattr(ms, "load_stage_outputs", lambda *a, **k: real(*a, extra_fingerprints=ROUTER_FP, **k))
        for key, up in (("meta_ebm", None), ("meta_ebm_on_blackbox", "blackbox")):
            _, paths[key] = train_meta_stage(w["GS"], *files[ROOTS[key].upstream], verbose=False,
                                             artifact_dir=folder[key], upstream=up)
        blackbox = dict(zip(("mlp", "rf", "xgb"), files["blackbox"]))
        _, paths["meta_xgb_on_glass"] = run_meta(w, dict(zip(("mlp", "rf", "xgb"), files["glass"])),
                                                 folder["meta_xgb_on_glass"], save=True, upstream="glass")
        quick = functools.partial(comparison.compare_with_meta_ebm, n_boot=20)   # speed only
        mp.setattr(comparison, "compare_with_meta_ebm", quick)
        _, paths["meta_xgb"] = run_meta(w, blackbox, folder["meta_xgb"], save=True,
                                        meta_ebm_path=paths["meta_ebm"])
    return SimpleNamespace(root=root, GS=w["GS"], files=files,
                           paths={k: Path(p) for k, p in paths.items()})


def _arts(bank):
    return {k: audit.load_arbiter(k, p, bank.GS, ROUTER_FP) for k, p in bank.paths.items()}


def _copy_bank(bank, tmp_path) -> Path:
    root = tmp_path / "bank_pipeline"
    shutil.copytree(bank.root, root)
    return root


# ----------------------------------------------------------------------
# a complete 2x2 passes
# ----------------------------------------------------------------------
def test_the_audit_passes_on_a_complete_2x2_in_the_saved_formats(bank):
    roots = audit.select_roots(bank.root)
    assert roots == bank.paths                        # the newest in each producer folder
    arts, expected = _arts(bank), {}
    for key, art in arts.items():
        audit.check_arbiter_identity(key, art)
        inputs, up = audit.reload_upstream(key, art, bank.GS, bank.root, ROUTER_FP)
        assert {s: str(p) for s, p in up.items()} == dict(zip(audit.STAGES, bank.files[ROOTS[key].upstream]))
        if ROOTS[key].arbiter == "meta_xgb":
            audit.check_meta_xgb_serving(key, art, inputs)
        expected.update(audit.expected_reload(key, roots[key], art, up))
    assert len(expected) == 10                        # four arbiters, six upstream files
    audit.check_fresh_reload(expected, bank.root)
    columns = audit.check_crossover(arts, roots, bank.GS, ROUTER_FP, n_boot=20)
    assert {arm: c["protocol"]["upstream"]["held_fixed"] for arm, c in columns.items()} == \
        {"glass": True, "blackbox": True}
    assert audit.unloadable_xgb_artifacts(bank.root) == []


# ----------------------------------------------------------------------
# selection: explicit or newest, never a fallback
# ----------------------------------------------------------------------
def test_a_missing_arbiter_fails_naming_where_it_is_written(bank, tmp_path):
    root = _copy_bank(bank, tmp_path)
    shutil.rmtree(root / ROOTS["meta_xgb_on_glass"].folder)
    with pytest.raises(AuditError, match=r"meta_xgb_on_glass: no meta_xgb_stage4_\*\.joblib in .*"
                                         r"meta_xgb_on_glass \(written by black-box notebook Cell 37\)"):
        audit.select_roots(root)


def test_the_newest_arbiter_is_audited_with_no_fallback_to_an_older_one(bank, tmp_path):
    root = _copy_bank(bank, tmp_path)
    older = joblib.load(bank.paths["meta_ebm"])
    newer = Path(save_meta_ebm({**older, "schema": "meta_ebm/1"}, str(root / ROOTS["meta_ebm"].folder)))
    os.utime(newer, (newer.stat().st_mtime + 60,) * 2)
    roots = audit.select_roots(root)
    assert roots["meta_ebm"] == newer
    with pytest.raises(AuditError, match=rf"^meta_ebm .*{re.escape(newer.name)}: Stage4InputError: "
                                         r"Meta-EBM artifact is not schema 'meta_ebm/2'"):
        audit.load_arbiter("meta_ebm", roots["meta_ebm"], bank.GS, ROUTER_FP)


def test_a_manifest_names_the_arbiters_explicitly(bank, tmp_path):
    root = _copy_bank(bank, tmp_path)
    manifest = tmp_path / "manifest.json"
    entries = {k: str(p.relative_to(bank.root)) for k, p in bank.paths.items()}
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    newer = Path(save_meta_ebm(joblib.load(bank.paths["meta_ebm"]), str(root / ROOTS["meta_ebm"].folder)))
    os.utime(newer, (newer.stat().st_mtime + 60,) * 2)
    assert audit.select_roots(root, manifest) == {k: root / v for k, v in entries.items()}
    manifest.write_text(json.dumps({k: v for k, v in entries.items() if k != "meta_xgb"}))
    with pytest.raises(AuditError, match=r"missing \['meta_xgb'\]"):
        audit.select_roots(root, manifest)
    manifest.write_text(json.dumps({**entries, "meta_xgb": "models/nowhere.joblib"}))
    with pytest.raises(AuditError, match=r"do not exist: \['meta_xgb: .*nowhere\.joblib'\]"):
        audit.select_roots(root, manifest)


# ----------------------------------------------------------------------
# identity and evidence
# ----------------------------------------------------------------------
def _stages(art):
    return audit._fields(art)["stage4_inputs"]["upstream"]["stages"]


IDENTITY_CASES = {
    "upstream arm swapped": ("meta_ebm", lambda a: audit._fields(a)["stage4_inputs"]["upstream"].update(arm="blackbox"),
                             "upstream.arm = 'blackbox', expected 'glass'"),
    "crossover not declared": ("meta_ebm_on_blackbox",
                               lambda a: audit._fields(a)["stage4_inputs"]["upstream"].update(crossover=False),
                               "upstream.crossover = False, expected True"),
    "wrong Pass 2 population": ("meta_xgb", lambda a: _stages(a)["stage2"].update(
        pass2_population={"cross_fit": "full", "full_train": "full"}), "pass2_population (F17)"),
    "label evidence missing": ("meta_xgb_on_glass", lambda a: _stages(a)["stage2"].update(label_evidence=None),
                               "stage2.label_evidence = None (F12)"),
    "Stage 3 labelled unset": ("meta_ebm", lambda a: _stages(a)["stage3"].update(arm="unset"),
                               "stages.stage3.arm = 'unset', expected 'glass'"),
    "feature contract reordered": ("meta_xgb", lambda a: audit._fields(a)["feature_names"].reverse(),
                                   "feature_names"),
    "label-fingerprint checks not recorded": (
        "meta_ebm", lambda a: audit._fields(a)["stage4_inputs"].update(validation=[
            r for r in audit._fields(a)["stage4_inputs"]["validation"]
            if "training rows and labels" not in r["check"]]), "no record of"),
    "a failed check recorded": ("meta_xgb", lambda a: audit._fields(a)["stage4_inputs"]["validation"][0]
                                .update(passed=False), "failed checks"),
}


@pytest.mark.parametrize("case", IDENTITY_CASES)
def test_identity_and_evidence_errors_name_the_artifact_and_field(bank, case):
    key, edit, message = IDENTITY_CASES[case]
    art = copy.deepcopy(_arts(bank)[key])
    audit.check_arbiter_identity(key, art)            # valid before the edit
    edit(art)
    with pytest.raises(AuditError) as err:
        audit.check_arbiter_identity(key, art)
    assert str(err.value).startswith(f"{key} is not the ") and message in str(err.value), str(err.value)


def test_an_arbiter_without_an_upstream_record_is_rejected_as_legacy(bank):
    art = copy.deepcopy(_arts(bank)["meta_ebm"])
    del art["stage4_inputs"]["upstream"]
    with pytest.raises(AuditError, match="written before PR 36 F29"):
        audit.check_arbiter_identity("meta_ebm", art)


CROSSOVER_CASES = {
    "a column's arbiters read different upstream outputs": (
        "meta_xgb_on_glass", lambda a: _stages(a)["stage3"].update(content_fingerprint="0" * 16),
        "glass column"),
    "one arbiter read the same outputs on both upstreams": (
        "meta_ebm_on_blackbox", lambda a: None, "outputs are identical on both upstreams"),
    "the cascade comparison used another Meta-EBM": (
        "meta_xgb", lambda a: a.comparison["meta_ebm_config"].update(timestamp="2000-01-01 00:00:00"),
        "cascade comparison used the Meta-EBM of '2000-01-01 00:00:00'"),
}


@pytest.mark.parametrize("case", CROSSOVER_CASES)
def test_the_2x2_check_refuses_mixed_or_unlinked_arbiters(bank, case):
    key, edit, message = CROSSOVER_CASES[case]
    arts = {k: copy.deepcopy(a) for k, a in _arts(bank).items()}
    if case.startswith("one arbiter"):                 # its stage1 outputs, recorded as the native's
        _stages(arts[key])["stage1"]["content_fingerprint"] = _stages(arts["meta_ebm"])["stage1"]["content_fingerprint"]
    edit(arts[key])
    with pytest.raises(AuditError, match=re.escape(message)):
        audit.check_crossover(arts, bank.paths, bank.GS, ROUTER_FP, n_boot=20)


def test_upstream_files_missing_or_replaced_after_the_arbiter_ran_are_caught(bank, tmp_path):
    key = "meta_xgb_on_glass"
    art = copy.deepcopy(_arts(bank)[key])
    stage1 = _stages(art)["stage1"]
    stage1["artifact_path"] = "upstream/nowhere.joblib"            # relative to bank_pipeline/
    with pytest.raises(AuditError, match=r"stage1: upstream/nowhere\.joblib not found"):
        audit.reload_upstream(key, art, bank.GS, bank.root, ROUTER_FP)
    replaced = joblib.load(bank.files["glass"][0])
    replaced["stage_output"]["test_proba"] = replaced["stage_output"]["test_proba"] * 0.99
    joblib.dump(replaced, tmp_path / "lr.joblib")
    stage1["artifact_path"] = str(tmp_path / "lr.joblib")
    with pytest.raises(AuditError, match="stage1 outputs differ .* not the outputs it consumed"):
        audit.reload_upstream(key, art, bank.GS, bank.root, ROUTER_FP)


# ----------------------------------------------------------------------
# legacy pickles and saved models
# ----------------------------------------------------------------------
def test_a_pickle_of_the_removed_xgboost_package_is_reported_with_its_cause(bank, tmp_path, monkeypatch):
    folder = tmp_path / audit.XGB_STAGE3_FOLDER
    folder.mkdir(parents=True)
    legacy = types.ModuleType("blackbox_pipeline.models.xgb.artifacts")   # as P4a removed it
    legacy.Stage3Artifact = type("Stage3Artifact", (), {"__module__": legacy.__name__})
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    with open(folder / "xgb_stage3_20250101_000000.joblib", "wb") as fh:
        # the C pickler: a class by reference, as that package saved its artifacts (the
        # pure-Python pickler joblib.dump uses pickles it by value once dill is imported)
        pickle.dump(legacy.Stage3Artifact(), fh, protocol=4)
    monkeypatch.delitem(sys.modules, legacy.__name__)
    shutil.copy(bank.files["blackbox"][2], folder / "xgb_stage3_20260101_000000.joblib")
    bad = audit.unloadable_xgb_artifacts(tmp_path)
    assert [Path(p).name for p, _ in bad] == ["xgb_stage3_20250101_000000.joblib"]
    assert "No module named 'blackbox_pipeline.models.xgb.artifacts'" in bad[0][1]
    assert "removed in PR 36 Commit 4 (P4a)" in bad[0][1]
    with pytest.raises(AuditError, match="removed in PR 36 Commit 4"):
        audit.check_fresh_reload({bad[0][0]: {"type": "shared.stage3.artifact.Stage3Artifact"}}, tmp_path)


def test_a_stage3_model_must_serve_its_saved_test_probabilities(tmp_path):
    run = run_arm(ARMS["xgboost"], tmp_path)
    store = Stage3ArtifactStore(str(tmp_path), "xgb_stage3")
    path = store.save(run.artifact, verbose=False)
    features = list(run.split["X_train"].columns)
    audit.check_stage3_serving(path, run.split, "blackbox", expected_features=features)
    with pytest.raises(AuditError, match=r"identity .*expected \{'model_family': 'ebm', 'arm': 'glass'\} \(F21\)"):
        audit.check_stage3_serving(path, run.split, "glass")
    edited = copy.deepcopy(run.artifact)
    edited.refit["test_proba"] = np.asarray(edited.refit["test_proba"]) * 0.99
    other = Stage3ArtifactStore(str(tmp_path / "edited"), "xgb_stage3").save(edited, verbose=False)
    with pytest.raises(AuditError, match=r"does not reproduce refit\['test_proba'\]"):
        audit.check_stage3_serving(other, run.split, "blackbox")
    sidecar = Path(other[:-len(".joblib")] + ".json")
    side = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**side, "features": side["features"][::-1]}))
    with pytest.raises(AuditError, match="sidecar features differs from the payload"):
        audit.check_stage3_serving(other, run.split, "blackbox")


# ----------------------------------------------------------------------
# the audit looks where the notebooks save; the realdata suite is opt-in
# ----------------------------------------------------------------------
def _calls(notebook, name):
    out = []
    for cell in json.loads((BANK_ROOT / notebook).read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            src = "\n".join(l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith("%"))
            out += [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
                    and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == name]
    return out


def _folder(node) -> str:
    """A notebook path expression (BLACKBOX_ROOT / ... / _XO_DIR / ...) relative to bank_pipeline/."""
    names = {"BANK_ROOT": PurePosixPath("."), "BLACKBOX_ROOT": PurePosixPath("blackbox_pipeline")}
    names["_XO_DIR"] = names["BLACKBOX_ROOT"] / "models" / "stage4_crossover"   # asserted below
    return str(eval(compile(ast.Expression(node), "<notebook>", "eval"), {"str": str, **names}))


def test_the_audit_finds_the_arbiters_where_the_notebooks_save_them():
    bb = "blackbox_pipeline/blackbox_cascade.ipynb"
    (glass_meta,) = _calls("glass_pipeline/glass_cascade.ipynb", "train_meta_stage")
    assert "artifact_dir" not in {k.arg for k in glass_meta.keywords}
    default = inspect.signature(train_meta_stage).parameters["artifact_dir"].default
    assert PurePosixPath(default) == PurePosixPath(ROOTS["meta_ebm"].folder)
    assigned = {}
    for c in json.loads((BANK_ROOT / bb).read_text(encoding="utf-8"))["cells"]:
        if c["cell_type"] == "code":
            for n in ast.walk(ast.parse("".join(c["source"]))):
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                    assigned.setdefault(n.targets[0].id, []).append(n.value)
    assert [ast.unparse(v) for v in assigned["_XO_DIR"]] == ["BLACKBOX_ROOT / 'models' / 'stage4_crossover'"]
    (crossover_ebm,) = _calls(bb, "train_meta_stage")
    kw = {k.arg: k.value for k in crossover_ebm.keywords}
    assert ast.literal_eval(kw["upstream"]) == ROOTS["meta_ebm_on_blackbox"].upstream
    assert _folder(kw["artifact_dir"]) == ROOTS["meta_ebm_on_blackbox"].folder
    configs = {}
    for call in _calls(bb, "train_meta_xgb_stage"):
        kw = {k.arg: k.value for k in call.keywords}
        upstream = ast.literal_eval(kw["upstream"]) if "upstream" in kw else "blackbox"
        configs[upstream] = (kw["config"], "meta_ebm_path" in kw)
    assert ast.unparse(configs["blackbox"][0]) == "META_XGB_CONFIG" and configs["blackbox"][1]
    (meta_xgb_config,) = assigned["META_XGB_CONFIG"]
    assert _folder({k.arg: k.value for k in meta_xgb_config.keywords}["artifact_dir"]) == ROOTS["meta_xgb"].folder
    glass_cfg = {k.arg: k.value for k in configs["glass"][0].keywords}
    assert _folder(glass_cfg["artifact_dir"]) == ROOTS["meta_xgb_on_glass"].folder
    assert 'f"meta_ebm_{ts}.joblib"' in inspect.getsource(save_meta_ebm)
    assert ROOTS["meta_ebm"].pattern == ROOTS["meta_ebm_on_blackbox"].pattern == "meta_ebm_*.joblib"
    stem = MetaXGBConfig().artifact_stem
    assert ROOTS["meta_xgb"].pattern == ROOTS["meta_xgb_on_glass"].pattern == f"{stem}_*.joblib"


def _collected(*marker_args) -> int:
    run = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          *marker_args, "tests/realdata"], cwd=BANK_ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=300)
    assert run.returncode in (0, 5), run.stdout[-2000:] + run.stderr[-2000:]   # 5: none selected
    return sum("::" in line for line in run.stdout.splitlines())


def test_the_realdata_audit_is_discoverable_and_opt_in():
    assert _collected() == 0                          # pytest.ini's default selection
    assert _collected("-m", "not slow") == 0          # an -m that does not name realdata
    assert _collected("-m", "realdata") >= 10
