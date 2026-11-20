"""
PR 39 Step 4A: the notebooks' execution path (execution contract sections 5.3,
5.4; F2, F9) and the regeneration vehicle (upstream_regeneration.vehicle).
Notebooks are read, never executed.

* Legacy Stage 2 is off by default: one ``RUN_LEGACY_STAGE2 = False`` per
  notebook, set (with the magic registered) before the first gated cell; the
  gated cells are exactly the legacy Stage 2 cells; every call that fits,
  tunes or saves a Stage 2 router sits in a gated cell.
* Pinned reads: the Stage 2 source is ``load_pinned_stage2(GLOBAL_SPLIT)``;
  every cross-notebook read is ``pinned_path(<role>)``; no cell outside the
  legacy switch reads a folder's newest file or a latest alias.
* Stage 4 fitting prevention: the truncated copy drops exactly the Stage 4
  cells, keeps every other cell byte-identical, and refuses a notebook whose
  kept cells would fit, compare or trace an arbiter, or whose switch is on.
* PR 41 Commit 4: on Windows the vehicle runs nbclient on its own selector loop
  (``asyncio.Runner``); elsewhere, below Python 3.11 and inside a running loop
  it calls ``client.execute()`` as before. Checked with a stand-in client, no kernel.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from shared.pinned_artifacts import ROLES
from tests.support.notebooks import BANK_ROOT
from upstream_regeneration import vehicle as V

NOTEBOOKS = {"glass": BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb",
             "blackbox": BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"}
GATED = {"glass": ["9", "10", "11", "12", "12B", "13", "14", "15", "16"],
         "blackbox": ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "28B"]}
#: Calls that fit, tune, analyse or save a Stage 2 router (F2).
STAGE2_FITS = {"GlassRouterPipeline", "crossfit_stage2", "train_rf_stage", "train_rf_baseline",
               "analyze_rf_lift", "save_rf_artifact", "save_model", "RFRouter", "SingleRFRouter",
               "crossfit_rf_router", "save_rf_router", "resolve_rf_params",
               "run_specialization_tuning", "fit_arm", "headroom_probe", "noise_floor"}
#: Newest-file and folder-search readers (F9).
NEWEST = {"glob", "rglob", "find_artifact_for_split", "find_stage3_artifact",
          "find_glass_arbiter_artifact", "load_latest", "iterdir", "listdir", "scandir"}
HEADER = re.compile(r"^#\s*CELL\s+([0-9]+[A-Z]?)\s*:", re.M)


def _cells(nb):
    out = []
    for i, c in enumerate(json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            tree = ast.parse("\n".join(line for line in src.splitlines()
                                       if not line.lstrip().startswith(("%", "!"))))
            out.append({"index": i, "src": src, "tree": tree, "cell": HEADER.search(src).group(1),
                        "gated": src.startswith("%%legacy_stage2\n")})
    return out


def _calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _name(call):
    return getattr(call.func, "id", None) or getattr(call.func, "attr", None)


@pytest.mark.parametrize("nb", sorted(NOTEBOOKS))
def test_the_legacy_stage2_cells_are_off_by_default_and_gated(nb):
    cells = _cells(nb)
    assert [c["cell"] for c in cells if c["gated"]] == GATED[nb]
    assert not [c for c in cells if "%%legacy_stage2" in c["src"] and not c["gated"]]
    switch = [(c, n) for c in cells for n in ast.walk(c["tree"]) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "RUN_LEGACY_STAGE2" for t in n.targets)]
    assert len(switch) == 1 and ast.unparse(switch[0][1].value) == "False"
    where = switch[0][0]
    assert not where["gated"] and 'register_switch("legacy_stage2", "RUN_LEGACY_STAGE2")' in where["src"]
    first_gated = min(c["index"] for c in cells if c["gated"])
    assert where["index"] < first_gated                          # set before any gated cell
    fits = [(c["cell"], _name(n)) for c in cells if not c["gated"]
            for n in _calls(c["tree"]) if _name(n) in STAGE2_FITS]
    assert fits == [], f"Stage 2 is fitted or saved outside the legacy switch: {fits}"
    assert [c["cell"] for c in cells if c["gated"] and
            any(_name(n) in STAGE2_FITS for n in _calls(c["tree"]))], "the check sees the fits"


@pytest.mark.parametrize("nb", sorted(NOTEBOOKS))
def test_the_stage2_source_is_the_pinned_accepted_run(nb):
    cells = _cells(nb)
    loads = [(c, n) for c in cells for n in _calls(c["tree"]) if _name(n) == "load_pinned_stage2"]
    assert len(loads) == 1 and ast.unparse(loads[0][1]) == "load_pinned_stage2(GLOBAL_SPLIT)"
    assert not loads[0][0]["gated"]
    # it runs only while the switch is off
    tree = loads[0][0]["tree"]
    guarded = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and any(m is loads[0][1] for m in ast.walk(n))]
    assert guarded and any("RUN_LEGACY_STAGE2" in ast.unparse(g.test) for g in guarded)


@pytest.mark.parametrize("nb", sorted(NOTEBOOKS))
def test_no_cell_outside_the_switch_reads_a_newest_file_or_a_latest_alias(nb):
    for c in _cells(nb):
        if c["gated"]:
            continue
        found = [_name(n) for n in _calls(c["tree"]) if _name(n) in NEWEST]
        if nb == "blackbox" and c["cell"] == "29":
            # Cell 29's ``_newest`` serves only XGB_PARAM_SOURCE = "artifact" (not the default)
            assert found == ["glob"] and 'XGB_PARAM_SOURCE = "tune"' in c["src"]
            continue
        assert found == [], (c["cell"], found)
        consts = [n.value for n in ast.walk(c["tree"]) if isinstance(n, ast.Constant)
                  and isinstance(n.value, str)]
        aliases = [s for s in consts if s.endswith(("_latest.joblib", "rf_router_two_pass.joblib",
                                                     "stage1_blackbox_output.joblib"))]
        assert aliases == [], (c["cell"], aliases)


def test_every_cross_notebook_read_is_a_pinned_role():
    roles = {}
    for c in _cells("blackbox"):
        for n in _calls(c["tree"]):
            if _name(n) == "pinned_path":
                (arg,) = n.args
                roles[c["cell"]] = roles.get(c["cell"], []) + [ast.literal_eval(arg)]
    assert roles == {"14A": ["stage1.lr"], "33": ["stage3.ebm"], "35": ["stage4.glass_arbiter"]}
    assert all(r in ROLES for rs in roles.values() for r in rs)
    glass = _cells("glass")
    assert not [n for c in glass for n in _calls(c["tree"]) if _name(n) == "pinned_path"]
    # GLASS Cell 19 reads the GLASS Router Cell 8B pinned
    (source,) = [c for c in glass if c["cell"] == "8B"]
    assert "glass_path = str(STAGE2_PINNED.glass_path)" in source["src"]


def test_the_stage1_handoff_and_the_stage3_block_cache_never_replace_an_existing_file():
    (s14a,) = [c for c in _cells("blackbox") if c["cell"] == "14A"]
    assert "MLP_S1_PATH = MLP_ARTIFACT_PATH" in s14a["src"]
    assert not [n for n in _calls(s14a["tree"]) if _name(n) == "save"
                and ast.unparse(n.func.value) == "MLP_S1"]
    (s17,) = [c for c in _cells("glass") if c["cell"] == "17"]
    assert 'f"stage3_block_{_FP_NOW}_{_BLOCK_ID}.joblib"' in s17["src"]
    assert "block_identity(GLOBAL_SPLIT, EBMFeaturePipeline, EBM_STAGE3_CONFIG, _FP_NOW)" in s17["src"]


# ----------------------------------------------------------------------
# the vehicle: Stage 4 excluded by a truncated, hashed copy
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nb", sorted(NOTEBOOKS))
def test_the_truncated_copy_drops_exactly_the_stage4_cells_and_keeps_the_rest_byte_identical(nb):
    import hashlib
    copy_nb, rec = V.truncated_copy(nb)
    full = json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))["cells"]
    kept = copy_nb["cells"]
    assert [c["source"] for c in kept] == [c["source"] for c in full[:len(kept)]]
    assert rec["removed_code_cells"] == [f"CELL {h}" for h in V.NOTEBOOKS[nb][1]]
    removed = [c for c in full[len(kept):] if c["cell_type"] == "code"]
    calls = {_name(n) for c in removed for n in ast.walk(ast.parse("".join(c["source"])))
             if isinstance(n, ast.Call)}
    assert calls & set(V.STAGE4_CALLS)                     # what is dropped is Stage 4
    for c in kept:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            names = {_name(n) for n in ast.walk(ast.parse("\n".join(
                ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith("%"))))
                if isinstance(n, ast.Call)}
            assert not names & set(V.STAGE4_CALLS)
    hashes = [k["source_sha256"] for k in rec["kept_code_cells"]]
    assert hashes == [hashlib.sha256("".join(c["source"]).encode("utf-8")).hexdigest()
                      for c in kept if c["cell_type"] == "code"]
    assert sum(k["gated"] for k in rec["kept_code_cells"]) == len(GATED[nb])
    text = (json.dumps(copy_nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    assert rec["truncated_copy_sha256"] == hashlib.sha256(text).hexdigest()
    assert V.truncated_copy(nb)[1] == rec                   # deterministic


def _probe(tmp_path, nb, edit):
    root = tmp_path / "bank"
    rel = V.NOTEBOOKS[nb][0]
    (root / rel).parent.mkdir(parents=True)
    data = json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))
    edit(data["cells"])
    (root / rel).write_text(json.dumps(data), encoding="utf-8")
    return root


def _code(cells, header):
    (c,) = [c for c in cells if c["cell_type"] == "code" and f"# CELL {header}:" in "".join(c["source"])]
    return c


@pytest.mark.parametrize("edit, message", [
    (lambda cells: _code(cells, "18").update(
        source=["# CELL 18: X\n", "train_glass_arbiter(GLOBAL_SPLIT)\n"]),
     "kept cells call Stage 4 entry points"),
    (lambda cells: _code(cells, "8B").update(source=[
        "".join(_code(cells, "8B")["source"]).replace("RUN_LEGACY_STAGE2 = False",
                                                      "RUN_LEGACY_STAGE2 = True")]),
     "must be set exactly once, to False"),
    (lambda cells: cells.pop(cells.index(_code(cells, "20"))), "not the Stage 4 cells"),
    (lambda cells: cells.append(dict(_code(cells, "17"))), "not the Stage 4 cells"),
])
def test_the_vehicle_refuses_a_notebook_that_would_fit_stage4_or_refit_stage2(tmp_path, edit, message):
    root = _probe(tmp_path, "glass", edit)
    with pytest.raises(V.VehicleError, match=message):
        V.truncated_copy("glass", root=root)


def test_executed_copies_are_written_outside_the_repository(tmp_path):
    repo = tmp_path / "repo"
    bank = repo / "prototype" / "bank_pipeline"
    bank.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert V.repository_root(bank) == repo.resolve()
    for inside in (repo, repo / "out", bank / "executed"):
        with pytest.raises(V.VehicleError, match="inside the repository"):
            V.require_outside(inside, bank)
    assert V.require_outside(tmp_path / "executed", bank) == (tmp_path / "executed").resolve()


def test_execute_refuses_an_output_folder_inside_the_repository_before_any_kernel(tmp_path,
                                                                                  monkeypatch):
    pytest.importorskip("nbclient")
    nb, _ = V.truncated_copy("glass")
    with pytest.raises(V.VehicleError, match="inside the repository"):
        V.execute(nb, "glass", V.repository_root() / "executed", "c4_20261115_000000")


def _probe_notebook(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": src} for src in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                        "language": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def test_execute_runs_a_fresh_kernel_in_the_notebook_folder_with_the_run_environment(tmp_path,
                                                                                     monkeypatch):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    root = tmp_path / "bank_pipeline"
    (root / "glass_pipeline").mkdir(parents=True)
    monkeypatch.setenv("BANK_UPSTREAM_PINS", "outer value")
    nb = _probe_notebook(["import os\nprint(os.getcwd())\nprint(os.environ.get('BANK_UPSTREAM_PINS'))"])
    out = tmp_path / "executed"
    run = V.execute(nb, "glass", out, "c4_20261115_000000", root=root,
                    env={"BANK_UPSTREAM_PINS": "the run's pins file"})
    assert run["completed"] and run["error"] is None
    text = "".join("".join(o.get("text", "")) for o in json.loads(
        Path(run["executed_copy"]["path"]).read_text(encoding="utf-8"))["cells"][0]["outputs"])
    assert str((root / "glass_pipeline").resolve()) in text and "the run's pins file" in text
    import os
    assert os.environ["BANK_UPSTREAM_PINS"] == "outer value"            # restored
    assert Path(run["input_copy"]["path"]).parent == out.resolve()
    with pytest.raises(V.VehicleError, match="exists"):                  # never overwritten
        V.execute(nb, "glass", out, "c4_20261115_000000", root=root)


def test_execute_records_a_failing_cell_and_keeps_the_partial_copy(tmp_path):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    root = tmp_path / "bank_pipeline"
    (root / "glass_pipeline").mkdir(parents=True)
    nb = _probe_notebook(["x = 1", "raise ValueError('stage 1 failed')", "y = 2"])
    run = V.execute(nb, "glass", tmp_path / "executed", "c4_20261115_000001", root=root)
    assert run["completed"] is False and "stage 1 failed" in run["error"]
    cells = json.loads(Path(run["executed_copy"]["path"]).read_text(encoding="utf-8"))["cells"]
    assert cells[2]["execution_count"] is None                        # stopped at the error


# ----------------------------------------------------------------------
# PR 41 Commit 4: the Windows selector-loop pathway (a stand-in client, no kernel)
# ----------------------------------------------------------------------
class _Client:
    """Records which nbclient entry point the vehicle called, and on which loop."""

    def __init__(self, error=None):
        self.calls, self.loop, self.error = [], None, error

    def execute(self):
        self.calls.append("execute")

    async def async_execute(self):
        self.calls.append("async_execute")
        self.loop = asyncio.get_running_loop()
        if self.error is not None:
            raise self.error


def _current_loops(monkeypatch):
    """Spy on ``asyncio.set_event_loop`` (calling through): the current loops the vehicle sets."""
    seen, real = [], asyncio.set_event_loop

    def spy(loop):
        seen.append(loop)
        real(loop)

    monkeypatch.setattr(asyncio, "set_event_loop", spy)
    return seen


def _as_platform(monkeypatch, platform, version_info=None):
    """The vehicle's view of ``sys`` only: the rest of the process keeps the real one."""
    monkeypatch.setattr(V, "sys", types.SimpleNamespace(
        platform=platform, version_info=version_info or sys.version_info,
        executable=sys.executable))


@pytest.mark.skipif(not hasattr(asyncio, "Runner"), reason="asyncio.Runner needs Python 3.11+")
def test_on_windows_the_notebook_runs_on_its_own_selector_loop(monkeypatch):
    _as_platform(monkeypatch, "win32")
    seen = _current_loops(monkeypatch)
    client = _Client()
    V._run_notebook(client)
    assert client.calls == ["async_execute"]
    assert isinstance(client.loop, asyncio.SelectorEventLoop) and client.loop.is_closed()
    assert seen == [client.loop, None]           # current during the run, unset afterwards


@pytest.mark.skipif(not hasattr(asyncio, "Runner"), reason="asyncio.Runner needs Python 3.11+")
def test_on_windows_a_failing_run_raises_to_the_vehicle_and_its_loop_still_closes(monkeypatch):
    _as_platform(monkeypatch, "win32")
    seen = _current_loops(monkeypatch)
    client = _Client(error=ValueError("stage 1 failed"))
    with pytest.raises(ValueError, match="stage 1 failed"):
        V._run_notebook(client)
    assert client.calls == ["async_execute"] and client.loop.is_closed()
    assert seen == [client.loop, None]


@pytest.mark.parametrize("platform, version_info", [
    ("linux", None), ("darwin", None), ("win32", (3, 10, 15)),
])
def test_other_platforms_and_older_pythons_call_client_execute_as_before(monkeypatch, platform,
                                                                        version_info):
    _as_platform(monkeypatch, platform, version_info)
    seen = _current_loops(monkeypatch)
    client = _Client()
    V._run_notebook(client)
    assert client.calls == ["execute"] and client.loop is None and seen == []


def test_inside_a_running_loop_windows_falls_back_to_client_execute(monkeypatch):
    _as_platform(monkeypatch, "win32")
    client = _Client()

    async def caller():
        V._run_notebook(client)                  # a synchronous call from inside a running loop

    asyncio.run(caller())                        # no RuntimeError from a nested Runner
    assert client.calls == ["execute"] and client.loop is None


@pytest.mark.parametrize("nb", sorted(NOTEBOOKS))
def test_the_switched_off_path_reads_no_name_only_a_legacy_cell_binds(nb, tmp_path):
    """With the legacy cells skipped, a clean top-to-bottom run (Stage 4 included, as in
    Commit 5A) reads no name before an active cell binds it."""
    from tests.support.notebooks import read_before_bound
    data = json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))
    data["cells"] = [c for c in data["cells"] if not (
        c["cell_type"] == "code" and "".join(c["source"]).startswith("%%legacy_stage2\n"))]
    probe = tmp_path / "active.ipynb"
    probe.write_text(json.dumps(data), encoding="utf-8")
    assert read_before_bound(probe) == []
