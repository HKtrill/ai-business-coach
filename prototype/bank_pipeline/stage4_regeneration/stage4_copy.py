"""
stage4_regeneration.stage4_copy
===============================
The Commit 5A notebook copy (execution contract section 5.4): a fresh kernel
rebuilds only the split and runs the Stage 4 fitting cells, unchanged, on the
upstream artifacts the run pinned. Stages 1-3 are not retrained.

The copy of a committed notebook holds, in order:

1. ``split``: every code cell before the notebook's first Stage 1 cell (GLASS
   Cells 0-4, black-box Cells 1-6), byte-identical. They set the paths, open
   the database, preprocess and bind ``GLOBAL_SPLIT``, as the notebook does.
2. One generated entry cell (``entry_source``). It binds each name the Stage 4
   cells read from Stages 1-3 to ``str(pinned_path(<role>))``: the run's pins
   file (``$BANK_UPSTREAM_PINS``), checked by path, size and SHA-256 (A5.1).
3. ``stage4``: the fitting cells, byte-identical. GLASS Cell 19 fits the native
   GLASS Arbiter. Black-box Cells 35-37 fit Meta-XGB (with its saved cascade
   comparison) and both crossovers (section 5.1 item 5).

The rest of the Commit 4 vehicle's Stage 4 cells (``reserved``) are not run.
GLASS Cell 20 (the cascade trace) rewrites the Git-tracked
``research_logs/figures/venn_cascade_trace.png``. Black-box Cell 38 holds the
column comparisons, which are Step 5B's record (A5.4). Commit 6B runs both
notebooks top to bottom.

Each kept cell's source SHA-256 is recorded. Outputs are cleared in the copy
only, and the committed notebooks are never written. ``classify`` holds an
execution to its registered Stage 4 outputs (``OUTPUTS``, keyed as
``tests.support.artifact_audit.ROOTS``). It accepts one immutable version per
arbiter and the Meta-XGB summary sidecars. Anything modified, deleted or
otherwise added fails it.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from upstream_regeneration import inventory as I
from upstream_regeneration import vehicle as V

__all__ = ["NOTEBOOKS", "ENTRY_HEADER", "OUTPUTS", "SIDECARS", "ROOT_KEYS", "ROOT_UPSTREAM",
           "PRODUCERS", "RUN", "GLASS_ARBITER_ROLE", "entry_source", "roles_read", "stage4_copy",
           "classify"]

BANK_ROOT = Path(__file__).resolve().parents[1]

#: The pins-file role of the native GLASS Arbiter, read by black-box Cell 35.
GLASS_ARBITER_ROLE = "stage4.glass_arbiter"

#: Per notebook: the split cells, the first Stage 1 cell, the Stage 4 cells 5A runs,
#: the Stage 4 cells it does not run, and the entry cell's (name, role) bindings.
NOTEBOOKS = {
    "glass": {
        "split": ("0", "1", "2", "2B", "3", "4"),
        "first_stage1": "5",
        "stage4": ("19",),
        "reserved": ("20",),
        "bind": (("lr_calibrated_path", "stage1.lr"),
                 ("glass_path", "stage2.glass_router"),
                 ("ebm_path", "stage3.ebm")),
    },
    "blackbox": {
        "split": ("1", "2", "3", "4", "5", "6"),
        "first_stage1": "7",
        "stage4": ("35", "36", "37"),
        "reserved": ("38",),
        "bind": (("MLP_S1_PATH", "stage1.mlp"),
                 ("GLASS_S1_PATH", "stage1.lr"),
                 ("RF_ROUTER_PATH", "stage2.rf_router"),
                 ("GLASS_BUNDLE_PATH", "stage2.glass_router"),
                 ("XGB_STAGE3_PATH", "stage3.xgb"),
                 ("_ebm_path", "stage3.ebm")),
    },
}
ENTRY_HEADER = "# STAGE 4 ENTRY: Stages 1-3 by pin (stage4_regeneration, not a notebook cell)"
#: The calls that fit an arbiter, and the ones each notebook's 5A cells must make.
FITS = frozenset({"train_glass_arbiter", "train_meta_stage", "train_meta_xgb_stage"})
FITTED = {"glass": ("train_glass_arbiter",),
          "blackbox": ("train_glass_arbiter", "train_meta_xgb_stage")}

_TS_US = r"[0-9]{8}_[0-9]{6}_[0-9]{6}"
_XO = "blackbox_pipeline/models/stage4_crossover"
#: Per notebook: arbiter key (artifact_audit.ROOTS) -> the immutable version its cell writes.
OUTPUTS = {
    "glass": {
        "glass_arbiter": rf"models/glass_arbiter/glass_arbiter_{_TS_US}\.joblib",
    },
    "blackbox": {
        "meta_xgb": rf"blackbox_pipeline/models/meta_xgb/meta_xgb_stage4_{_TS_US}\.joblib",
        "glass_arbiter_on_blackbox":
            rf"{_XO}/glass_arbiter_on_blackbox/glass_arbiter_{_TS_US}\.joblib",
        "meta_xgb_on_glass": rf"{_XO}/meta_xgb_on_glass/meta_xgb_stage4_{_TS_US}\.joblib",
    },
}
#: save_meta_xgb's JSON summary beside each Meta-XGB version (never read by role).
SIDECARS = {nb: tuple(p[:-len(r"\.joblib")] + r"\.json" for k, p in outs.items()
                      if k.startswith("meta_xgb"))
            for nb, outs in OUTPUTS.items()}
ROOT_KEYS = tuple(k for outs in OUTPUTS.values() for k in outs)
#: Arbiter key -> the upstream arm whose Stage 1-3 artifacts it reads.
ROOT_UPSTREAM = {"glass_arbiter": "glass", "meta_xgb": "blackbox",
                 "glass_arbiter_on_blackbox": "blackbox", "meta_xgb_on_glass": "glass"}
PRODUCERS = {"glass_arbiter": "GLASS notebook Cell 19 (train_glass_arbiter)",
             "meta_xgb": "black-box notebook Cell 36 (train_meta_xgb_stage)",
             "glass_arbiter_on_blackbox": "black-box notebook Cell 37 (train_glass_arbiter, "
                                          "upstream='blackbox')",
             "meta_xgb_on_glass": "black-box notebook Cell 37 (train_meta_xgb_stage, "
                                  "upstream='glass')"}
#: Both executions together: what ``finalize`` holds the run's net change to.
RUN = "run"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _src(cell) -> str:
    return "".join(cell.get("source") or [])


def entry_source(notebook: str) -> str:
    """The entry cell: each upstream name the Stage 4 cells read, bound by pin."""
    bind = NOTEBOOKS[notebook]["bind"]
    width = max(len(name) for name, _ in bind)
    lines = ["# " + "=" * 60, ENTRY_HEADER, "# " + "=" * 60,
             "# PR 39 Commit 5A (execution contract section 5.4): Stages 1-3 are not retrained.",
             "# Each upstream artifact is the run's pin ($BANK_UPSTREAM_PINS), checked by path,",
             "# size and SHA-256 before it is used (A5.1).",
             "from shared.pinned_artifacts import pinned_path", ""]
    lines += [f'{name} = str(pinned_path("{role}"))' for name, role in bind]
    lines += [""] + [f'print(f"{name:<{width}} <- {{{name}}}")' for name, _ in bind]
    return "\n".join(lines) + "\n"


def _pinned_roles(src: str) -> List[str]:
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!")))
    return [ast.literal_eval(n.args[0]) for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "pinned_path"]


def _code_cells(nb) -> List[Tuple[int, dict]]:
    return [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def roles_read(notebook: str, *, root: Path = BANK_ROOT) -> List[str]:
    """Every pins-file role the copy reads: the entry cell's and the Stage 4 cells' own."""
    nb, _ = stage4_copy(notebook, root=root)
    return sorted({r for _, c in _code_cells(nb) for r in _pinned_roles(_src(c))})


def stage4_copy(notebook: str, *, root: Path = BANK_ROOT) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(notebook dict, record)``: split cells, the entry cell, the Stage 4 cells (docstring)."""
    if notebook not in NOTEBOOKS:
        raise V.VehicleError(f"unknown notebook {notebook!r}: expected one of {sorted(NOTEBOOKS)}")
    spec = NOTEBOOKS[notebook]
    rel, vehicle_stage4 = V.NOTEBOOKS[notebook]
    if set(spec["stage4"]) & set(spec["reserved"]) or \
            sorted(spec["stage4"] + spec["reserved"]) != sorted(vehicle_stage4):
        raise V.VehicleError(f"{rel}: the 5A and reserved cells {spec['stage4']} / "
                             f"{spec['reserved']} must split the Stage 4 cells {vehicle_stage4}")
    raw = (Path(root) / rel).read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    code = _code_cells(nb)
    headers = [V._header(c) for _, c in code]

    def only(h):
        if headers.count(h) != 1:
            raise V.VehicleError(f"{rel}: {headers.count(h)} code cells headed CELL {h}, "
                                 "expected one")
        return headers.index(h)

    split = code[:only(spec["first_stage1"])]
    got = tuple(V._header(c) for _, c in split)
    if got != spec["split"]:
        raise V.VehicleError(f"{rel}: the code cells before CELL {spec['first_stage1']} are "
                             f"{list(got)}, not the registered split cells {list(spec['split'])}")
    stage4 = [code[only(h)] for h in spec["stage4"]]
    calls = {f"CELL {V._header(c)}": sorted(V._call_names(_src(c)) & set(V.STAGE4_CALLS))
             for _, c in split}
    if any(calls.values()):
        raise V.VehicleError(f"{rel}: split cells call Stage 4 entry points "
                             f"{ {k: v for k, v in calls.items() if v} }")
    if not V._assigned(_src(split[-1][1]), "GLOBAL_SPLIT"):
        raise V.VehicleError(f"{rel}: CELL {spec['split'][-1]} does not bind GLOBAL_SPLIT")
    fitted = sorted(set().union(*(V._call_names(_src(c)) & FITS for _, c in stage4)))
    if fitted != sorted(FITTED[notebook]):
        raise V.VehicleError(f"{rel}: the 5A cells call {fitted}, not {sorted(FITTED[notebook])}")
    reserved_fits = {h: sorted(V._call_names(_src(code[only(h)][1])) & FITS)
                     for h in spec["reserved"]}
    if any(reserved_fits.values()):
        raise V.VehicleError(f"{rel}: a cell 5A does not run fits an arbiter {reserved_fits}")

    entry_src = entry_source(notebook)
    entry = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": entry_src.splitlines(keepends=True)}
    out = {k: copy.deepcopy(v) for k, v in nb.items() if k != "cells"}
    out["cells"] = ([copy.deepcopy(c) for _, c in split] + [entry]
                    + [copy.deepcopy(c) for _, c in stage4])
    for c in out["cells"]:
        c["outputs"], c["execution_count"] = [], None
    kept = split + stage4
    for (_, c), o in zip(kept, out["cells"][:len(split)] + out["cells"][len(split) + 1:]):
        if _src(o) != _src(c):                                       # pragma: no cover
            raise V.VehicleError(f"{rel}: a cell's source changed in the copy")
    text = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")

    def cells(items):
        return [{"index": i, "cell": f"CELL {V._header(c)}",
                 "source_sha256": _sha(_src(c).encode("utf-8"))} for i, c in items]

    record = {
        "notebook": rel, "notebook_sha256": _sha(raw),
        "split_cells": cells(split),
        "entry_cell": {"source_sha256": _sha(entry_src.encode("utf-8")),
                       "bindings": {name: role for name, role in spec["bind"]}},
        "stage4_cells": cells(stage4),
        "not_run": [f"CELL {h}" for h in spec["reserved"]],
        "stage4_calls_in_split_cells": [],
        "copy_sha256": _sha(text),
    }
    return out, record


def classify(before: Mapping[str, Mapping], after: Mapping[str, Mapping],
             notebook: str) -> Dict[str, Any]:
    """
    What one execution (or, for ``RUN``, both) changed in the model folders.
    ``problems`` empty means accepted: exactly one new version per arbiter of
    that notebook, each Meta-XGB version with its summary sidecar, and nothing
    else added, modified or deleted.
    """
    if notebook == RUN:
        outputs = {k: p for nb in OUTPUTS for k, p in OUTPUTS[nb].items()}
        sidecars = tuple(p for nb in SIDECARS for p in SIDECARS[nb])
    elif notebook in OUTPUTS:
        outputs, sidecars = OUTPUTS[notebook], SIDECARS[notebook]
    else:
        raise ValueError(f"unknown notebook {notebook!r}: expected one of {sorted(OUTPUTS)} "
                         f"or {RUN!r}")
    d = I.diff(before, after)
    produced: Dict[str, List[str]] = {k: [] for k in outputs}
    side, problems = [], []
    for rel in d["added"]:
        key = next((k for k, p in outputs.items() if re.fullmatch(p, rel)), None)
        if key is not None:
            produced[key].append(rel)
        elif any(re.fullmatch(p, rel) for p in sidecars):
            side.append(rel)
        else:
            problems.append(f"unregistered addition {rel}")
    problems += [f"existing file modified: {rel}" for rel in d["modified"]]
    problems += [f"existing file deleted: {rel}" for rel in d["deleted"]]
    paired = set()
    for key, found in produced.items():
        if len(found) != 1:
            problems.append(f"{key}: {len(found)} new versions ({found}), expected exactly one")
        elif key.startswith("meta_xgb"):
            want = found[0][:-len(".joblib")] + ".json"
            if want in side:
                paired.add(want)
            else:
                problems.append(f"{key}: no summary sidecar {want}")
    problems += [f"sidecar without its version: {rel}" for rel in sorted(set(side) - paired)]
    return {"notebook": notebook, "diff": {k: len(v) for k, v in d.items()},
            "produced": {k: (f[0] if len(f) == 1 else f) for k, f in produced.items()},
            "sidecars": sorted(side), "problems": problems, "passed": not problems}
