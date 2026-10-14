"""
PR 35 commit A — the notebook itself.

1. Behaviour preservation, checked independently of the specialization work:
   the former Stage 2 cells (verbatim fixture) and the 7 consolidated cells are
   executed against identical synthetic upstream state, and every Stage 2 name
   later stages read — plus the saved router artifacts — must match.
2. ``notebook/apply_stage2_cells.py`` puts the replacements into a real .ipynb:
   right positions, markdown untouched, idempotent, and it refuses unsafe states.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

try:
    from stage2_notebook_harness import (
        NEW_CELLS, OLD_CELLS, REPO_ROOT, diff, load_cells, run_cells, stub_glass_modules,
        upstream_namespace,
    )
except ImportError:  # tests/ is a package in some layouts
    from .stage2_notebook_harness import (  # type: ignore[no-redef]
        NEW_CELLS, OLD_CELLS, REPO_ROOT, diff, load_cells, run_cells, stub_glass_modules,
        upstream_namespace,
    )

STAGE2_MARKER = re.compile(r"^# STAGE 2 · [A-Z] —", re.MULTILINE)

#: Every Stage 2 name a later cell (or Stage 3/4) reads.
CONTRACT_NAMES = (
    "STAGE2", "SPLIT_FP", "STAGE_FOLDS", "STAGE_FOLD_ID", "STAGE_FOLD_FP",
    "GLASS_BUNDLE", "GLASS_OOF_METRICS", "CONSTRAINT_SOURCE", "CONSTRAINTS",
    "RF_PARAMS", "RF_PARAMS_ORIGIN", "ABLATION_CONSTRAINTS", "VALIDATE_NESTED",
    "ablation", "router", "RF_OOF", "HEADLINE", "stage2_results",
    "stage2_oof_results", "STAGE2_OUT", "STAGE2_OUTPUT",
)
#: Intentional, documented differences in the saved cross-fit protocol.
PROTOCOL_ADDED = {"hyperparameters", "split_fingerprint"}


def _load_apply_tool():
    spec = importlib.util.spec_from_file_location(
        "apply_stage2_cells", REPO_ROOT / "notebook" / "apply_stage2_cells.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


apply_tool = _load_apply_tool()


def consolidated_cells():
    return [c for c in load_cells(NEW_CELLS) if STAGE2_MARKER.search(c[1])]


def pr35_keys():
    """(replacement markers, new-cell markers) in cells-file order."""
    cells = apply_tool.parse_new_cells(NEW_CELLS)
    return [c.key for c in cells if c.replaces], [c.key for c in cells if not c.replaces]


# ----------------------------------------------------------------------
# 1. behaviour preservation
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    return upstream_namespace(tmp_path_factory.mktemp("glass_artifacts"))


def _execute(cells, upstream, workdir: Path, monkeypatch):
    for name, mod in stub_glass_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)
    return run_cells(cells, dict(upstream))


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="the pre-PR 35 Cell 28 uses a PEP 701 f-string")
def test_consolidated_cells_reproduce_pre_pr35_stage2(upstream, tmp_path, monkeypatch):
    old_cells, new_cells = load_cells(OLD_CELLS), consolidated_cells()
    assert len(old_cells) == 16 and len(new_cells) == 7

    old = _execute(old_cells, upstream, tmp_path / "old", monkeypatch)
    new = _execute(new_cells, upstream, tmp_path / "new", monkeypatch)

    problems = []
    for name in CONTRACT_NAMES:
        problems += diff(old[name], new[name], name)

    # Router internals the outputs depend on.
    for attr in ("remainder_mask_", "oof_pass1_proba_", "oof_pass2_proba_"):
        problems += diff(getattr(old["router"], attr), getattr(new["router"], attr), f"router.{attr}")
    problems += diff(old["router"].fit_report_.nested_check,
                     new["router"].fit_report_.nested_check, "router.nested_check")

    # Cross-fit outputs (wall-clock column excluded) and protocol.
    problems += diff(old["RF_OOF_FOLDS"].drop(columns="seconds"),
                     new["RF_OOF_FOLDS"].drop(columns="seconds"), "RF_OOF_FOLDS")
    op, np_ = old["RF_OOF_PROTOCOL"], new["RF_OOF_PROTOCOL"]
    assert set(np_) - set(op) == PROTOCOL_ADDED
    assert np_["test_outputs"] == "from the full-train router"
    problems += diff({k: v for k, v in op.items() if k != "test_outputs"},
                     {k: np_[k] for k in op if k != "test_outputs"}, "RF_OOF_PROTOCOL")

    # Saved artifacts: payloads, sidecar metadata and threshold sweeps.
    skip = ("created_utc", "model", "pass1_model", "pass2_model", "extra.oof_protocol")

    def without_wall_clock(payload):
        extra = payload.get("extra", {})
        if "oof_fold_summary" in extra:
            extra["oof_fold_summary"] = [{k: v for k, v in r.items() if k != "seconds"}
                                         for r in extra["oof_fold_summary"]]
        return payload

    for stem in ("rf_router_single", "rf_router_two_pass"):
        a = without_wall_clock(joblib.load(tmp_path / "old" / "models" / "rf_router" / f"{stem}.joblib"))
        b = without_wall_clock(joblib.load(tmp_path / "new" / "models" / "rf_router" / f"{stem}.joblib"))
        problems += diff({k: v for k, v in a.items() if k not in skip},
                         {k: v for k, v in b.items() if k not in skip}, stem,
                         skip=tuple(f"{stem}.{s}" for s in skip))
        ja = json.loads((tmp_path / "old" / "models" / "rf_router" / f"{stem}_metadata.json").read_text())
        jb = json.loads((tmp_path / "new" / "models" / "rf_router" / f"{stem}_metadata.json").read_text())
        for j in (ja, jb):
            j.pop("created_utc")
            j.get("extra", {}).pop("oof_protocol", None)
            without_wall_clock(j)
        problems += diff(ja, jb, f"{stem}_metadata.json")
        for pas in ("pass1", "pass2"):
            problems += diff(
                pd.read_csv(tmp_path / "old" / "models" / "rf_router" / f"{stem}_{pas}_sweep.csv"),
                pd.read_csv(tmp_path / "new" / "models" / "rf_router" / f"{stem}_{pas}_sweep.csv"),
                f"{stem}_{pas}_sweep.csv")

    assert not problems, "Stage 2 outputs changed:\n  " + "\n  ".join(problems[:40])


# ----------------------------------------------------------------------
# 2. applying the replacements to a real .ipynb
# ----------------------------------------------------------------------
def _code(src, i):
    return {"cell_type": "code", "execution_count": i, "id": f"c{i:02d}",
            "metadata": {"tags": [f"t{i}"]},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}],
            "source": src.splitlines(keepends=True)}


def _md(src, i):
    return {"cell_type": "markdown", "id": f"m{i:02d}", "metadata": {},
            "source": src.splitlines(keepends=True)}


def make_notebook(path: Path, *, drop=(), duplicate=(), merge=()):
    """A notebook laid out like the real one, Stage 2 cells taken from the fixture."""
    old = {re.search(r"CELL (\d+[A-Z]?):", src).group(1): src for _, src in load_cells(OLD_CELLS)}
    cells = [_md("# Stage 1\n", 0), _code("# CELL 17: STAGE 1 WORK\nx = 1\n", 1),
             _md("## Stage 2 — RF router\n", 2)]
    i = 3
    merged_done = False
    for cid, src in old.items():
        if cid in drop:
            continue
        if merge and cid in merge:
            if merged_done:
                continue
            src = "\n".join(old[m] for m in merge)
            merged_done = True
        cells.append(_code(src, i)); i += 1
        if cid in duplicate:
            cells.append(_code(src, i)); i += 1
        if cid == "25":
            cells.append(_md("### Outer cross-fit\n", i)); i += 1
    cells += [_code("# CELL 32: STAGE 3 ENTRY\ny = 2\n", i), _md("end\n", i + 1)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1))
    return nb


def _layout(nb_path):
    nb = json.loads(Path(nb_path).read_text())
    out = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        m = re.search(r"^# (STAGE 2b? · [A-Z]) —", src, re.MULTILINE)
        h = re.search(r"CELL (\d+[A-Z]?):", src)
        out.append(m.group(1) if m else (f"CELL {h.group(1)}" if h else f"md:{src.strip()}"))
    return nb, out


def test_apply_replaces_former_cells_in_place(tmp_path):
    path = tmp_path / "nb.ipynb"
    make_notebook(path)
    before = path.read_text()

    dry = apply_tool.apply(path, write=False)
    assert dry.changed and path.read_text() == before, "dry run must not write"

    apply_tool.apply(path, write=True)
    nb, layout = _layout(path)
    replacements, new = pr35_keys()
    assert replacements == [f"STAGE 2 · {x}" for x in "ABCDEFG"]
    assert layout == [
        "md:# Stage 1", "CELL 17", "md:## Stage 2 — RF router",
        "STAGE 2 · A", "STAGE 2 · B", "STAGE 2 · C", "md:### Outer cross-fit",
        "STAGE 2 · D", "STAGE 2 · E", "STAGE 2 · F", "STAGE 2 · G", *new,
        "CELL 32", "md:end",
    ]
    by_marker = {l: c for l, c in zip(layout, nb["cells"])}
    a = by_marker["STAGE 2 · A"]
    assert a["id"] == "c03" and a["metadata"] == {"tags": ["t3"]}  # former Cell 18's identity
    assert a["outputs"] == [] and a["execution_count"] is None
    assert by_marker["CELL 32"]["outputs"], "unrelated cells keep their outputs"
    new_sources = {k: s for k, s in
                   ((re.search(r"^# (STAGE 2 · [A-Z]) —", s, re.M).group(1), s)
                    for _, s in consolidated_cells())}
    assert "".join(a["source"]).strip() == new_sources["STAGE 2 · A"].strip()
    assert (tmp_path / "nb.pre-pr35.ipynb").read_text() == before


def test_apply_is_idempotent(tmp_path):
    path = tmp_path / "nb.ipynb"
    make_notebook(path)
    apply_tool.apply(path, write=True)
    once = path.read_text()
    again = apply_tool.apply(path, write=True)
    assert not again.changed and path.read_text() == once
    assert all(a.endswith(")") and "unchanged" in a for a in again.actions)


@pytest.mark.parametrize("kwargs, message", [
    (dict(drop=("21",)), "not in the notebook"),
    (dict(duplicate=("24",)), "appears in notebook cells"),
    (dict(merge=("26", "29")), "different replacements"),
])
def test_apply_refuses_unsafe_notebooks(tmp_path, kwargs, message):
    path = tmp_path / "nb.ipynb"
    make_notebook(path, **kwargs)
    before = path.read_text()
    with pytest.raises(apply_tool.ApplyError, match=message):
        apply_tool.apply(path, write=True)
    assert path.read_text() == before


def test_apply_accepts_former_cells_merged_within_one_group(tmp_path):
    path = tmp_path / "nb.ipynb"
    make_notebook(path, merge=("20", "21"))  # both belong to Stage 2 · B
    apply_tool.apply(path, write=True)
    _, layout = _layout(path)
    replacements, new = pr35_keys()
    assert [l for l in layout if l.startswith("STAGE")] == replacements + new


def test_apply_inserts_new_cells_after_their_predecessor(tmp_path):
    cells_src = tmp_path / "cells.py"
    cells_src.write_text(NEW_CELLS.read_text(encoding="utf-8").replace("\r\n", "\n")
                         + "\n\n# %%\n# STAGE 2b · Z — EXTRA\n# (new, PR 35)\nprint('z')\n")
    path = tmp_path / "nb.ipynb"
    make_notebook(path)
    apply_tool.apply(path, write=True, cells_source=cells_src)
    _, layout = _layout(path)
    assert layout[layout.index("CELL 32") - 1] == "STAGE 2b · Z"
    stage = [l for l in layout if l.startswith("STAGE")]
    assert stage[-1] == "STAGE 2b · Z" and stage[:7] == [f"STAGE 2 · {x}" for x in "ABCDEFG"]


def test_cells_file_covers_exactly_the_former_stage2_cells():
    cells = apply_tool.parse_new_cells(NEW_CELLS)
    covered = sorted(i for c in cells for i in c.replaces)
    assert set(covered) == apply_tool.FORMER_STAGE2_CELLS and len(covered) == 16
    assert [c.key for c in cells if c.replaces] == [f"STAGE 2 · {x}" for x in "ABCDEFG"]
