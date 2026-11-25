"""
Notebook contracts across stages (PR 36, Commit 4 P3; notebooks read, never executed).

G4  The black-box notebook looks for upstream artifacts only in the folders the
    current producers write — each folder is checked against the producer's
    own default or the producing call — never in the pre-reorganisation
    locations (``GLASS_ROOT``, ``PROTOTYPE_ROOT``, a project-wide search) and
    with no fallback to a file no current producer writes. PR 39 (F9): the
    cross-notebook reads (GLASS LR, GLASS Router bundle, GLASS EBM, GLASS
    Arbiter) are by pinned path and SHA-256, never a folder search; the only
    folder left is the legacy Cell 19's RF parameter source.
G5  Each active notebook is one labelled top-to-bottom path: every code cell
    carries one unique ``CELL`` header, no code cell is a copy of another, no
    cell installs packages, and no cell reads a name that only a later cell
    binds.
G6  The concluded PR 35 specialization experiment is off by default: its
    three flags are False and every expensive or test-split step sits under
    the flag that gates it.
G7  The feature-research notebooks draw the cascades' split under the same
    protocol and never bind a test row. The PR 42 Feature Research Notebook
    goes further: it draws the split only through
    ``feature_research.training_side.draw_training_side`` (which returns the
    training side and nothing of the test side), spells no test-side name or
    key (the PR 38 boundary's ``FORBIDDEN_FIELD_TOKENS``), and opens accepted
    artifacts only through ``frozen_references.accepted_artifact_views`` (the
    PR 38 boundary extractors).

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
from glass_pipeline.glass_arbiter.meta_stage import train_glass_arbiter
from tests.support.notebooks import ACTIVE_NOTEBOOKS, BANK_ROOT, read_before_bound
from tests.support.synthetic_bank import make_bank_frame

GLASS = BANK_ROOT / "glass_pipeline" / "glass_cascade.ipynb"
BLACKBOX = BANK_ROOT / "blackbox_pipeline" / "blackbox_cascade.ipynb"
RESEARCH = BANK_ROOT / "feature_research" / "exploratory_feature_research.ipynb"
FEATURE_RESEARCH = BANK_ROOT / "feature_research" / "feature_research.ipynb"     # PR 42
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
    "XGB_ARTIFACT_DIRS": ('[BLACKBOX_ROOT / "models" / "xgb"]', "blackbox_pipeline/models/xgb"),
}
#: PR 39 (F9): the cross-notebook reads, each a role of the run's pins file.
PINNED_READS = {"GLASS_S1_PATH": "stage1.lr", "_ebm_path": "stage3.ebm",
                "GLASS_ARBITER_PATH": "stage4.glass_arbiter"}


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
    assert all("artifact_dir" not in _kw(c) for c in _calls(GLASS, "train_glass_arbiter"))
    (xgb_cfg,) = _calls(BLACKBOX, "XGBStage3Config")
    assert _same(_kw(xgb_cfg)["artifact_dir"], 'str(BLACKBOX_ROOT / "models" / "xgb")')
    assert default(lr_stage_calibrated.train_lr_stage, "output_dir") == "./models/lr"
    assert 'f"lr_calibrated_{timestamp}.joblib"' in inspect.getsource(save_artifact)
    # PR 39 (F9): every cross-notebook read is a pinned role, none a folder search
    assigned = _assigned(BLACKBOX)
    for name, role in PINNED_READS.items():
        (value,) = assigned[name]
        assert _same(value, f'pinned_path("{role}")'), f"{name} = {ast.unparse(value)}"
    # the one folder search left: Cell 29's ``_newest``, read only by the non-default
    # XGB_PARAM_SOURCE "artifact" (hyperparameters, re-validated by check_param_reuse)
    assert [ast.unparse(c.func) for c in _calls(BLACKBOX, "glob")] == ["d.glob"]
    (xgb_source,) = assigned["XGB_PARAM_SOURCE"]
    assert ast.literal_eval(xgb_source) == "tune"
    for finder in ("find_artifact_for_split", "find_stage3_artifact", "find_glass_arbiter_artifact"):
        assert not _calls(BLACKBOX, finder), finder
    assert [ast.unparse(c) for c in _calls(BLACKBOX, "load_pinned_stage2")] == [
        "load_pinned_stage2(GLOBAL_SPLIT)"]


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


# G7 (PR 42) — the Feature Research Notebook binds the training side only
#: Steps only ``draw_training_side`` takes: the population, the split and the test row ids.
SPLIT_STEPS = ("build_database", "BankPreprocessor", "sanity_check_data", "GlobalSplitManager",
               "create_split", "train_test_split", "load_split", "save_split", "split_frame",
               "build_train_reference", "train_reference_from_global_split", "read_csv",
               "read_sql_query")
#: Ways to open an artifact other than the boundary extractors (through
#: ``frozen_references.accepted_artifact_views``).
ARTIFACT_READERS = ("load", "read_pickle", "load_pinned_stage2", "pinned_path", "read_pins",
                    "read_verified", "verify_pin", "load_rf_router", "load_stage_outputs",
                    "build_stage4_inputs", "load_blackbox_stage4_inputs", "load_or_build_block",
                    "from_global_split", "find_stage3_artifact", "find_artifact_for_split",
                    "load_latest", "extract_router_view", "extract_lr_view", "extract_ebm_view",
                    "open")
BANNED_IMPORTS = ("joblib", "pickle", "dill", "cloudpickle", "shared.pinned_artifacts",
                  "shared.stage4", "shared.stage2.diagnostics", "data.global_splits",
                  "data.preprocessing", "sklearn.model_selection")


def _spelled(path) -> set:
    """Every name, attribute, keyword, definition, import alias and string a code cell spells."""
    out = set()
    for n in _nodes(path):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.alias):
            out.update(filter(None, (n.name, n.asname)))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def test_the_pr42_feature_research_notebook_draws_only_the_training_side():
    from feature_research import training_side
    (draw,) = _calls(FEATURE_RESEARCH, "draw_training_side")
    assert not draw.args and not draw.keywords
    assert [ast.unparse(v) for v in _assigned(FEATURE_RESEARCH)["TRAIN"]] == ["draw_training_side()"]
    imports = [n for n in _nodes(FEATURE_RESEARCH) if isinstance(n, ast.ImportFrom)
               and any(a.name == "draw_training_side" for a in n.names)]
    assert [n.module for n in imports] == ["feature_research.training_side"]
    assert training_side.TEST_SIZE == 0.20 and training_side.RANDOM_SEED == 42
    assert not [s for s in SPLIT_STEPS if _calls(FEATURE_RESEARCH, s)]


def test_the_pr42_feature_research_notebook_spells_no_test_side_name():
    from shared.stage2.diagnostics.boundary import FORBIDDEN_FIELD_TOKENS
    tokens = tuple(FORBIDDEN_FIELD_TOKENS) + ("holdout",)
    spelled = _spelled(FEATURE_RESEARCH)
    hits = sorted(s for s in spelled if any(t in s.lower() for t in tokens))
    assert not hits, hits
    assert not spelled & {"GLOBAL_SPLIT", "df_processed", "X_test", "y_test"}


def test_the_pr42_feature_research_notebook_reads_artifacts_only_through_the_boundary():
    assert not [r for r in ARTIFACT_READERS if _calls(FEATURE_RESEARCH, r)]
    modules = {a.name for n in _nodes(FEATURE_RESEARCH) if isinstance(n, ast.Import)
               for a in n.names}
    modules |= {n.module for n in _nodes(FEATURE_RESEARCH) if isinstance(n, ast.ImportFrom)}
    assert not [m for m in modules if any(m == b or m.startswith(b + ".") for b in BANNED_IMPORTS)]
    (views,) = _calls(FEATURE_RESEARCH, "accepted_artifact_views")
    assert ast.unparse(views.func) == "FR.accepted_artifact_views"
    assert [ast.unparse(a) for a in views.args] == ["TRAIN.train_reference"]
