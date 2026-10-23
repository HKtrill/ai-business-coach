"""
Notebook contracts across stages (PR 36, Commit 4 P3; notebooks read, never executed).

G4  The black-box notebook looks for upstream artifacts only in the folders the
    current producers write — each folder is checked against the producer's
    own default or the producing call — never in the pre-reorganisation
    locations (``GLASS_ROOT``, ``PROTOTYPE_ROOT``, a project-wide search) and
    with no fallback to a file no current producer writes.
G5  Each active notebook is one labelled top-to-bottom path: every code cell
    carries one unique ``CELL`` header, no code cell is a copy of another, no
    cell installs packages, and no cell reads a name that only a later cell
    binds.
G6  The concluded PR 35 specialization experiment is off by default: its
    three flags are False and every expensive or test-split step sits under
    the flag that gates it.
G7  The feature-research notebook draws the cascades' split under the same
    protocol and never binds a test row.

Stage-specific notebook contracts stay with their stages (tests/stage1-4,
tests/parity); import resolution is tests/notebooks/test_notebook_imports.py.
"""

from __future__ import annotations

import ast
import inspect
import json
import re

import pytest

import feature_research
from feature_research.validation import sanity_check_data
from glass_pipeline.glass_router.model_saver import ModelSaver
from glass_pipeline.glass_router.rf import save_rf_artifact
from glass_pipeline.lr import lr_stage_calibrated
from glass_pipeline.lr.artifacts import save_artifact
from glass_pipeline.meta_ebm.meta_stage import train_meta_stage
from tests.support.notebooks import ACTIVE_NOTEBOOKS, BANK_ROOT, read_before_bound
from tests.support.synthetic import make_bank_frame

GLASS = BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb"
BLACKBOX = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"
RESEARCH = BANK_ROOT / "feature_research" / "exploratory_feature_research.ipynb"
HEADER = re.compile(r"^#\s*CELL\s+([0-9]+[A-Z]?)\s*:", re.M)


def _code(path) -> list:
    """(cell index, source, tree) per code cell; IPython magics and shell lines dropped."""
    out = []
    for i, c in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            out.append((i, src, ast.parse("\n".join(
                ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))))))
    return out


def _nodes(path) -> list:
    return [n for _, _, t in _code(path) for n in ast.walk(t)]


def _assigned(path) -> dict:
    """name -> every value assigned to it (single-name targets)."""
    out: dict = {}
    for n in _nodes(path):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, []).append(n.value)
    return out


def _calls(path, name) -> list:
    return [n for n in _nodes(path) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == name]


def _same(node, source: str) -> bool:
    return ast.unparse(node) == ast.unparse(ast.parse(source, mode="eval").body)


def _kw(call) -> dict:
    return {k.arg: k.value for k in call.keywords}


# ----------------------------------------------------------------------
# G4 — artifact discovery
# ----------------------------------------------------------------------
DISCOVERY = {                      # black-box name -> (folder expression, folder relative to bank_pipeline)
    "RF_ARTIFACT_DIRS": ('[BANK_ROOT / "models" / "rf"]', "models/rf"),
    "GLASS_ARTIFACT_DIRS": ('[BANK_ROOT / "models" / "glass_router"]', "models/glass_router"),
    "XGB_ARTIFACT_DIRS": ('[BLACKBOX_ROOT / "models" / "xgb"]', "blackbox_pipeline/models/xgb"),
    "META_EBM_DIRS": ('[BANK_ROOT / "models" / "meta_ebm"]', "models/meta_ebm"),
}


def test_the_blackbox_notebook_searches_only_the_folders_current_producers_write():
    assigned = _assigned(BLACKBOX)
    for name, (expr, _) in DISCOVERY.items():
        (value,) = assigned[name]
        assert _same(value, expr), f"{name} = {ast.unparse(value)}"
    # ...and those are where the producers write (both notebooks run from bank_pipeline/)
    default = lambda fn, arg: inspect.signature(fn).parameters[arg].default    # noqa: E731
    assert default(save_rf_artifact, "output_dir").strip("./") == DISCOVERY["RF_ARTIFACT_DIRS"][1]
    assert all(_same(_kw(c)["output_dir"], '"./models/rf"') for c in _calls(GLASS, "save_rf_artifact"))
    assert '"./models/glass_router/glass_router_' in inspect.getsource(ModelSaver.save_model)
    assert all(not c.args and "output_path" not in _kw(c)            # its default folder
               for c in _calls(GLASS, "save_model"))
    assert default(train_meta_stage, "artifact_dir").strip("./") == DISCOVERY["META_EBM_DIRS"][1]
    assert all("artifact_dir" not in _kw(c) for c in _calls(GLASS, "train_meta_stage"))
    (xgb_cfg,) = _calls(BLACKBOX, "XGBStage3Config")
    assert _same(_kw(xgb_cfg)["artifact_dir"], 'str(BLACKBOX_ROOT / "models" / "xgb")')
    # GLASS Stage 1, found by Cell 14A
    (glob,) = [c for c in _calls(BLACKBOX, "glob") if _same(c.args[0], '"lr_calibrated_*.joblib"')]
    assert _same(glob.func.value, 'Path("./models/lr")')
    assert default(lr_stage_calibrated.train_lr_stage, "output_dir") == "./models/lr"
    assert 'f"lr_calibrated_{timestamp}.joblib"' in inspect.getsource(save_artifact)


def test_the_blackbox_notebook_has_no_legacy_artifact_locations_or_fallbacks():
    names = {n.id for n in _nodes(BLACKBOX) if isinstance(n, ast.Name)}
    assert not names & {"GLASS_ROOT", "PROTOTYPE_ROOT", "_GLASS_S1"}
    assert not _calls(BLACKBOX, "rglob")
    fallbacks = [c for c in _calls(BLACKBOX, "get") if _same(c.func.value, "globals()") and len(c.args) > 1]
    assert not fallbacks, [ast.unparse(c) for c in fallbacks]       # e.g. a path default
    strings = {n.value for n in _nodes(BLACKBOX) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not [s for s in strings if "stage1_glass_output" in s or s.startswith("stage3_*")]


# ----------------------------------------------------------------------
# G5 — one labelled top-to-bottom path
# ----------------------------------------------------------------------
@pytest.mark.parametrize("notebook", ACTIVE_NOTEBOOKS)
def test_each_active_notebook_is_one_labelled_top_to_bottom_path(notebook):
    path = BANK_ROOT / notebook
    cells = _code(path)
    headers = [HEADER.findall(src) for _, src, _ in cells]
    assert [i for (i, _, _), h in zip(cells, headers) if len(h) != 1] == [], "cells without one header"
    labels = [h[0] for h in headers]
    assert sorted({x for x in labels if labels.count(x) > 1}) == [], "repeated CELL headers"
    sources = [src.strip() for _, src, _ in cells]
    assert len(set(sources)) == len(sources), "a code cell is a copy of another"
    assert not [ln for src in sources for ln in src.splitlines()
                if re.match(r"\s*[%!]\s*(pip|conda|mamba)\b", ln)], "a cell installs packages"
    assert read_before_bound(path) == []


def _probe_notebook(tmp_path, sources):
    nb = {"cells": [{"cell_type": "code", "source": s, "metadata": {}, "outputs": [],
                     "execution_count": None} for s in sources], "metadata": {},
          "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "probe.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_the_execution_order_check_finds_a_name_read_before_its_cell(tmp_path):
    sources = ["x = 1", "print(y)", "y = 2", "z = [x + w for _ in range(2)]", "w = 3",
               "def f():\n    return later", "later = 4", "%time q = 1\nq2 = 1"]
    # y read in cell 1, bound in 2; w read inside a comprehension in 3, bound in 4.
    # A function body runs when called, so `later` is not a finding.
    assert read_before_bound(_probe_notebook(tmp_path, sources)) == [(1, "y", 2), (3, "w", 4)]


def test_the_execution_order_check_is_the_same_on_every_python_version(tmp_path):
    """
    Findings depend on the cells, not on how this interpreter's symtable
    describes them. Python 3.14 reports a compiler-generated
    ``__conditional_annotations__`` for any module annotation and moves the
    annotation's names into an ``__annotate__`` scope; Python 3.11 reports a
    nested comprehension's outer loop variables as free names. Neither is a
    finding, while every name a cell spells still is — annotation reads (3.13
    and earlier evaluate them when the cell runs) and dunder names included.
    """
    sources = [
        "flag = True",
        "x: int = 1",                                    # 3.14: __conditional_annotations__
        "if flag:\n    y: Later = 2",                   # an annotation read before its binding
        "Later = int",
        "z = {k: {v: a[k] for v in b} for k, a in d}",   # 3.11: a and k free in the inner scope
        "b, d = [1], {}",
        "print(__custom__)",                             # a dunder the cell spells
        "__custom__ = 1",
    ]
    assert read_before_bound(_probe_notebook(tmp_path, sources)) == [
        (2, "Later", 3), (4, "b", 5), (4, "d", 5), (6, "__custom__", 7)]


# ----------------------------------------------------------------------
# G6 — PR 35 specialization off by default
# ----------------------------------------------------------------------
GATED = {
    "SPEC_RUN_PROBE": {"noise_floor", "headroom_probe", "summarize_headroom"},
    "SPEC_RUN_TUNING": {"run_specialization_tuning", "fit_arm"},
    "SPEC_EVALUATE_TEST": {"register_test_evaluation", "paired_bootstrap", "slippage_table",
                           "specialization_verdict", "SPEC_TEST_OUT"},
}


def _guards(tree) -> dict:
    """id(node) -> source of every `if` test whose body (not else) contains it."""
    out: dict = {}

    def visit(node, tests):
        out[id(node)] = tests
        if isinstance(node, ast.If):
            visit(node.test, tests)
            for child in node.body:
                visit(child, tests + [ast.unparse(node.test)])
            for child in node.orelse:
                visit(child, tests)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, tests)

    visit(tree, [])
    return out


def test_the_pr35_specialization_cells_are_off_by_default_and_gated():
    assigned = _assigned(BLACKBOX)
    for flag in GATED:
        (value,) = assigned[flag]
        assert isinstance(value, ast.Constant) and value.value is False, flag
    seen = {flag: 0 for flag in GATED}
    for _, _, tree in _code(BLACKBOX):
        guards = _guards(tree)
        for node in ast.walk(tree):
            name = (getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                    if isinstance(node, ast.Call) else
                    node.id if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) else None)
            for flag, gated in GATED.items():
                if name in gated:
                    seen[flag] += 1
                    assert any(flag in t for t in guards[id(node)]), f"{name} runs without {flag}"
    assert all(seen.values()), seen                                  # the check can see them


# ----------------------------------------------------------------------
# G7 — the feature-research notebook's split
# ----------------------------------------------------------------------
def test_the_feature_research_notebook_draws_the_cascade_split_and_binds_no_test_row():
    (manager,) = _calls(RESEARCH, "GlobalSplitManager")
    kw = _kw(manager)
    assert ast.literal_eval(kw["test_size"]) == 0.20 and _same(kw["random_state"], "RANDOM_SEED")
    assert feature_research.RANDOM_SEED == 42                      # the cascades' seed
    (split,) = _calls(RESEARCH, "create_split")
    assert _same(split.args[0], "df_processed") and _same(_kw(split)["target_col"], "TARGET_COL")
    (target,) = _assigned(RESEARCH)["TARGET_COL"]
    assert _same(target, "sanity_check_data(df_processed)")
    assert sanity_check_data(make_bank_frame(n=200)) == "y"
    (pre,) = _calls(RESEARCH, "BankPreprocessor")
    assert ast.literal_eval(_kw(pre)["drop_leaky"]) is True and _calls(RESEARCH, "build_database")
    names = {n.id for n in _nodes(RESEARCH) if isinstance(n, ast.Name)}
    keys = {n.value for n in _nodes(RESEARCH) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not names & {"X_test", "y_test"} and not keys & {"X_test", "y_test"}
