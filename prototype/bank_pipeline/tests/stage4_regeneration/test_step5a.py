"""
PR 39 Step 5A: the Stage 4 regeneration vehicle (``stage4_regeneration``;
execution contract sections 5.1 item 5, 5.4, 7.1-7.5; gates A5.1-A5.3).
Notebooks are read, never executed; records and pins are synthetic.

* The copy: each committed notebook's split cells and Stage 4 fitting cells,
  byte-identical, around one generated entry cell that binds Stages 1-3 by
  pin. Nothing else is in it: no Stage 1-3 cell, and no reserved Stage 4
  cell (GLASS Cell 20, black-box Cell 38). It reads no name before a cell
  binds it, and every role it reads is in the run's pins.
* A notebook that no longer has the registered shape is refused.
* The outputs are the audit's four roots. ``classify`` accepts exactly one
  new version per arbiter and the Meta-XGB sidecars, and nothing else.
* Records are c5 run ids, written once, sidecar-checked. The run's pins are
  the manifest's six roles verbatim plus the GLASS Arbiter. A recorded
  upstream path that is not the pinned file is refused. The arbiter manifest
  is one ``select_roots`` accepts.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
from pathlib import Path

import pytest

from shared.pinned_artifacts import ROLES, PinError, PinnedFile, read_pins, sha256_bytes, write_pins
from stage4_regeneration import runner as R
from stage4_regeneration import stage4_copy as S
from tests.support import artifact_audit as audit
from tests.support.notebooks import BANK_ROOT, read_before_bound
from upstream_regeneration import vehicle as V

NOTEBOOK_FILES = {nb: BANK_ROOT / V.NOTEBOOKS[nb][0] for nb in S.NOTEBOOKS}


def _committed_cells(nb):
    cells = json.loads(NOTEBOOK_FILES[nb].read_text(encoding="utf-8"))["cells"]
    return {V._header(c): "".join(c["source"]) for c in cells if c["cell_type"] == "code"}


def _code(nb_dict):
    return ["".join(c["source"]) for c in nb_dict["cells"] if c["cell_type"] == "code"]


# ----------------------------------------------------------------------
# the copy
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nb", sorted(S.NOTEBOOKS))
def test_the_copy_is_the_split_cells_the_entry_cell_and_the_stage4_cells_unchanged(nb):
    spec, committed = S.NOTEBOOKS[nb], _committed_cells(nb)
    copy, rec = S.stage4_copy(nb)
    code = _code(copy)
    assert len(code) == len(spec["split"]) + 1 + len(spec["stage4"])
    kept = spec["split"] + spec["stage4"]
    assert code[:len(spec["split"])] + code[len(spec["split"]) + 1:] == [committed[h] for h in kept]
    assert code[len(spec["split"])] == S.entry_source(nb)
    assert all(c["outputs"] == [] and c["execution_count"] is None
               for c in copy["cells"] if c["cell_type"] == "code")
    assert [c["cell"] for c in rec["split_cells"]] == [f"CELL {h}" for h in spec["split"]]
    assert [c["cell"] for c in rec["stage4_cells"]] == [f"CELL {h}" for h in spec["stage4"]]
    assert {c["cell"]: c["source_sha256"] for c in rec["split_cells"] + rec["stage4_cells"]} == {
        f"CELL {h}": sha256_bytes(committed[h].encode("utf-8")) for h in kept}
    assert rec["notebook_sha256"] == sha256_bytes(NOTEBOOK_FILES[nb].read_bytes())
    assert rec["not_run"] == [f"CELL {h}" for h in spec["reserved"]]


@pytest.mark.parametrize("nb", sorted(S.NOTEBOOKS))
def test_5a_runs_the_fitting_cells_and_leaves_the_rest_of_stage4_to_5b_and_6b(nb):
    spec = S.NOTEBOOKS[nb]
    assert sorted(spec["stage4"] + spec["reserved"]) == sorted(V.NOTEBOOKS[nb][1])
    committed = _committed_cells(nb)
    calls = {h: V._call_names(committed[h]) for h in spec["stage4"] + spec["reserved"]}
    assert set().union(*(calls[h] for h in spec["stage4"])) >= set(S.FITTED[nb])
    assert not any(calls[h] & S.FITS for h in spec["reserved"])
    # the reserved cells are the comparisons (5B) and the trace (6B), never a fit
    assert set().union(*(calls[h] for h in spec["reserved"])) & {
        "compare_with_glass_arbiter", "run_cascade_trace"}
    # no Stage 1-3 cell is in the copy: the split cells precede the first Stage 1 cell
    order = list(committed)
    assert order.index(spec["split"][-1]) < order.index(spec["first_stage1"])


@pytest.mark.parametrize("nb", sorted(S.NOTEBOOKS))
def test_the_copy_reads_no_name_before_a_cell_binds_it(nb, tmp_path):
    copy, _ = S.stage4_copy(nb)
    path = tmp_path / f"{nb}_stage4.ipynb"
    path.write_text(json.dumps(copy), encoding="utf-8")
    assert read_before_bound(path) == []


@pytest.mark.parametrize("nb", sorted(S.NOTEBOOKS))
def test_the_entry_cell_binds_each_upstream_name_by_pin_and_only_those_names(nb):
    spec = S.NOTEBOOKS[nb]
    tree = ast.parse(S.entry_source(nb))
    bound = {n.targets[0].id: ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    assert bound == {name: f"str(pinned_path('{role}'))" for name, role in spec["bind"]}
    assert {role for _, role in spec["bind"]} <= set(R.UPSTREAM_ROLES)
    # each bound name is one a 5A Stage 4 cell reads
    committed = _committed_cells(nb)
    read = {n.id for h in spec["stage4"] for n in ast.walk(ast.parse(committed[h]))
            if isinstance(n, ast.Name)}
    assert set(bound) <= read


def test_the_roles_each_copy_reads_are_the_pins_of_its_execution():
    assert S.roles_read("glass") == sorted({"stage1.lr", "stage2.glass_router", "stage3.ebm"})
    # black-box Cell 35 reads the native GLASS Arbiter from the run's pins
    assert S.roles_read("blackbox") == sorted(set(R.UPSTREAM_ROLES) | {S.GLASS_ARBITER_ROLE})
    assert set(S.roles_read("blackbox")) <= set(ROLES)


def _notebook_root(tmp_path, nb, edit):
    root = tmp_path / "bank_pipeline"
    src = NOTEBOOK_FILES[nb]
    dst = root / V.NOTEBOOKS[nb][0]
    dst.parent.mkdir(parents=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    edit(data["cells"])
    dst.write_text(json.dumps(data), encoding="utf-8")
    return root


def _index(cells, header):
    return next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
                and V._header(c) == header)


@pytest.mark.parametrize("case", ["a cell added before Stage 1", "a Stage 4 cell duplicated",
                                  "a split cell fits an arbiter", "a fit moved to a reserved cell"])
def test_a_notebook_without_the_registered_shape_is_refused(tmp_path, case):
    nb = "blackbox"

    def edit(cells):
        if case == "a cell added before Stage 1":
            i = _index(cells, "6")
            cells.insert(i + 1, {"cell_type": "code", "metadata": {}, "outputs": [],
                                 "execution_count": None, "source": ["# CELL 6B: extra\nx = 1\n"]})
        elif case == "a Stage 4 cell duplicated":
            cells.append(json.loads(json.dumps(cells[_index(cells, "36")])))
        elif case == "a split cell fits an arbiter":
            c = cells[_index(cells, "5")]
            c["source"] = ["".join(c["source"]) + "\ntrain_glass_arbiter(GLOBAL_SPLIT)\n"]
        else:
            c = cells[_index(cells, "38")]
            c["source"] = ["".join(c["source"]) + "\ntrain_meta_xgb_stage(GLOBAL_SPLIT, 1, 2, 3)\n"]

    root = _notebook_root(tmp_path, nb, edit)
    with pytest.raises(V.VehicleError):
        S.stage4_copy(nb, root=root)


# ----------------------------------------------------------------------
# outputs and the inventory
# ----------------------------------------------------------------------
def test_the_outputs_are_the_audit_roots_in_their_producers_folders():
    assert sorted(S.ROOT_KEYS) == sorted(audit.ROOTS)
    for nb, outs in S.OUTPUTS.items():
        for key, pattern in outs.items():
            r = audit.ROOTS[key]
            sample = f"{r.folder}/{r.pattern.replace('*', '20261116_120000_123456')}"
            assert re.fullmatch(pattern, sample), (key, sample)
            assert fnmatch.fnmatch(sample.rsplit("/", 1)[1], r.pattern)
            assert S.ROOT_UPSTREAM[key] == r.upstream
            assert (r.producer.split(" Cell ")[1] in S.PRODUCERS[key])


def _file(sha="0" * 64, size=1):
    return {"bytes": size, "sha256": sha}


def _ts(i):
    return f"20261116_12000{i}_00000{i}"


def _outputs(nb):
    files = {}
    for i, (key, pattern) in enumerate(S.OUTPUTS[nb].items()):
        folder = audit.ROOTS[key].folder
        stem = "glass_arbiter" if key.startswith("glass") else "meta_xgb_stage4"
        files[f"{folder}/{stem}_{_ts(i)}.joblib"] = _file(f"{i}" * 64)
        if key.startswith("meta_xgb"):
            files[f"{folder}/{stem}_{_ts(i)}.json"] = _file(f"{i}" * 63 + "f")
    return files


@pytest.mark.parametrize("nb", sorted(S.OUTPUTS))
def test_classify_accepts_exactly_the_registered_stage4_outputs(nb):
    before = {"models/lr/lr_calibrated_20261115_204501.joblib": _file("a" * 64)}
    delta = S.classify(before, {**before, **_outputs(nb)}, nb)
    assert delta["passed"], delta["problems"]
    assert sorted(delta["produced"]) == sorted(S.OUTPUTS[nb])
    both = S.classify(before, {**before, **_outputs("glass"), **_outputs("blackbox")}, S.RUN)
    assert both["passed"] and len(both["sidecars"]) == 2


@pytest.mark.parametrize("case", ["modified", "deleted", "unregistered", "missing", "two versions",
                                  "sidecar alone", "no sidecar"])
def test_classify_refuses_any_other_change(case):
    before = {"models/lr/lr_calibrated_20261115_204501.joblib": _file("a" * 64),
              "models/ebm/ebm_stage3_latest.joblib": _file("b" * 64)}
    after = {**before, **_outputs("blackbox")}
    meta = next(p for p in after if p.startswith("blackbox_pipeline/models/meta_xgb/")
                and p.endswith(".joblib"))
    if case == "modified":
        after["models/ebm/ebm_stage3_latest.joblib"] = _file("c" * 64)
    elif case == "deleted":
        del after["models/lr/lr_calibrated_20261115_204501.joblib"]
    elif case == "unregistered":
        after["models/glass_router/glass_router_20261116.joblib"] = _file()
    elif case == "missing":
        del after[meta]
        del after[meta[:-len(".joblib")] + ".json"]
    elif case == "two versions":
        after[meta.replace("_00000", "_99999")] = _file()
    elif case == "sidecar alone":
        after["blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_20261116_130000_000000.json"] = _file()
    else:
        del after[meta[:-len(".joblib")] + ".json"]
    delta = S.classify(before, after, "blackbox")
    assert not delta["passed"] and delta["problems"]


# ----------------------------------------------------------------------
# records, pins, upstream resolution, the arbiter manifest
# ----------------------------------------------------------------------
@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(R.Rec, "RECORD_ROOT", tmp_path / "pr39")
    return tmp_path / "pr39"


def test_records_are_c5_run_ids_written_once_and_sidecar_checked(records):
    run = "c5_20261116_120000"
    R.write_record(run, "preflight", {"run_id": run, "passed": True})
    assert R.require_passed(run, "preflight")["passed"] is True
    with pytest.raises(R.Rec.RecordError, match="written once"):
        R.write_record(run, "preflight", {"run_id": run, "passed": True})
    (records / run / "preflight.json").write_text('{"run_id": "%s", "passed": true}\n' % run)
    with pytest.raises(R.Rec.RecordError, match="sidecar"):
        R.read_record(run, "preflight")
    for bad in ("c4_20261116_120000", "c5_2026111_120000", "c5_20261116_120000/x"):
        with pytest.raises(R.Rec.RecordError):
            R.check_run_id(bad)


def _bank(tmp_path):
    """Six pinned Stage 1-3 files, a manifest of them, and a new GLASS Arbiter."""
    root = tmp_path / "bank_pipeline"
    pins = {}
    for i, role in enumerate(R.UPSTREAM_ROLES):
        rel = f"models/{role.replace('.', '/')}_{i}.joblib"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        data = f"{role} artifact".encode()
        (root / rel).write_bytes(data)
        pins[role] = PinnedFile(role, rel, sha256_bytes(data), len(data))
    manifest = write_pins(tmp_path / "c4_20261115_204125" / "artifact_manifest.json",
                          "c4_20261115_204125", pins, root=root)
    arb = "models/glass_arbiter/glass_arbiter_20261116_120000_000001.joblib"
    (root / arb).parent.mkdir(parents=True)
    (root / arb).write_bytes(b"arbiter")
    produced = {"path": arb, "bytes": 7, "sha256": sha256_bytes(b"arbiter")}
    return root, pins, manifest, produced


def test_the_run_pins_are_the_manifests_six_roles_verbatim_and_the_glass_arbiter(tmp_path):
    root, pins, manifest, produced = _bank(tmp_path)
    files = R.run_pins(read_pins(manifest), produced)
    assert sorted(files) == sorted(set(R.UPSTREAM_ROLES) | {S.GLASS_ARBITER_ROLE})
    assert {r: files[r] for r in R.UPSTREAM_ROLES} == pins
    out = write_pins(tmp_path / "c5_20261116_120000" / R.PINS_FILE, "c5_20261116_120000", files,
                     root=root)
    from shared.pinned_artifacts import pinned_path
    assert pinned_path(S.GLASS_ARBITER_ROLE, pins_file=out, root=root) == root / produced["path"]
    (root / produced["path"]).write_bytes(b"changed")
    with pytest.raises(PinError):
        pinned_path(S.GLASS_ARBITER_ROLE, pins_file=out, root=root)
    short = read_pins(manifest)
    del short["files"]["stage3.xgb"]
    with pytest.raises(R.Rec.RecordError):
        R.run_pins(short, produced)


def _arbiter(root, arm, paths):
    stages = {s: {"artifact_path": str(paths[s])} for s in ("stage1", "stage2", "stage3")}
    return {"stage4_inputs": {"upstream": {"arm": arm, "stages": stages}}}


@pytest.mark.parametrize("arm", sorted(R.UPSTREAM_ROLE_OF))
def test_an_arbiter_resolves_only_when_it_read_the_pinned_files(tmp_path, arm):
    root, pins, manifest, _ = _bank(tmp_path)
    m = read_pins(manifest)
    good = {s: root / pins[r].path for s, r in R.UPSTREAM_ROLE_OF[arm].items()}
    got = R.upstream_resolution(_arbiter(root, arm, good), arm, m, root=root)
    assert sorted(got) == ["stage1", "stage2", "stage3"]
    relative = {s: pins[r].path for s, r in R.UPSTREAM_ROLE_OF[arm].items()}
    assert R.upstream_resolution(_arbiter(root, arm, relative), arm, m, root=root) == got
    other = "blackbox" if arm == "glass" else "glass"
    with pytest.raises(AssertionError, match="records upstream"):
        R.upstream_resolution(_arbiter(root, arm, good), other, m, root=root)
    twin = root / "models" / "twin.joblib"                     # same bytes, another file
    twin.write_bytes((root / pins[R.UPSTREAM_ROLE_OF[arm]["stage3"]].path).read_bytes())
    with pytest.raises(AssertionError, match="not the pinned"):
        R.upstream_resolution(_arbiter(root, arm, {**good, "stage3": twin}), arm, m, root=root)
    (root / pins[R.UPSTREAM_ROLE_OF[arm]["stage1"]].path).write_bytes(b"retrained")
    with pytest.raises(PinError):
        R.upstream_resolution(_arbiter(root, arm, good), arm, m, root=root)


def test_the_arbiter_manifest_is_one_the_audit_selects_as_the_pr39_generation(tmp_path):
    root = tmp_path / "bank_pipeline"
    entries = {}
    for nb in S.OUTPUTS:
        for rel in _outputs(nb):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x")
            key = next((k for k, p in S.OUTPUTS[nb].items() if re.fullmatch(p, rel)), None)
            if key:
                entries[key] = rel
    path = tmp_path / "arbiter_manifest.json"
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sel = audit.select_roots(root, path)
    assert sel.generation == audit.GENERATION and sorted(sel) == sorted(audit.ROOTS)


def test_the_junit_summary_totals_every_suite():
    xml = (b'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="14" failures="0" '
           b'errors="0" skipped="0"/></testsuites>')
    assert R.junit_summary(xml) == {"tests": 14, "failures": 0, "errors": 0, "skipped": 0}
    one = b'<testsuite tests="3" failures="1" errors="0" skipped="1"/>'
    assert R.junit_summary(one) == {"tests": 3, "failures": 1, "errors": 0, "skipped": 1}


def test_step5a_reads_the_accepted_commit4_manifest_and_cannot_write_a_commit4_record():
    assert R.UPSTREAM_RUN == "c4_20261115_204125"
    assert R.manifest_path() == BANK_ROOT / "research_logs/pr39/c4_20261115_204125/artifact_manifest.json"
    assert sorted(R.UPSTREAM_ROLES) == sorted(r for r in ROLES if not r.startswith("stage4."))
    # Commit 5 run ids only: a Step 5A record can never land in the Commit 4 folder
    with pytest.raises(R.Rec.RecordError):
        R.run_dir(R.UPSTREAM_RUN)
