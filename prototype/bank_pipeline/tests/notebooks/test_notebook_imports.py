"""
Import contract: every project import in the active research notebooks still
resolves (cells parsed, never executed), every module of the project packages
imports in a clean interpreter, and no module keeps a fallback import of a
project module that no longer exists.

This is the safety net for Commit 4: deleting or renaming a module that a
notebook or another module still imports fails here instead of mid-run in a
clean kernel — on any import path, not only the ones other tests happen to take.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.notebooks import (
    ACTIVE_NOTEBOOKS, BANK_ROOT, PROJECT_PACKAGES, project_imports, unresolved_imports,
)


@pytest.mark.parametrize("notebook", ACTIVE_NOTEBOOKS)
def test_active_notebook_project_imports_resolve(notebook):
    path = BANK_ROOT / notebook
    assert path.exists(), f"active notebook missing: {notebook}"
    imports = project_imports(path)
    assert imports, f"{notebook}: no project imports found (extractor broken?)"
    problems = unresolved_imports(imports)
    assert not problems, f"{notebook}:\n  " + "\n  ".join(problems)


def production_modules() -> list:
    """Dotted name of every module in the project packages."""
    out = []
    for package in PROJECT_PACKAGES:
        for path in sorted((BANK_ROOT / package).rglob("*.py")):
            parts = list(path.relative_to(BANK_ROOT).with_suffix("").parts)
            if "tests" in parts:
                continue
            out.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return out


_IMPORT_EACH = """
import importlib, json, sys
failures = []
for name in sys.stdin.read().split():
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
print(json.dumps(failures))
"""


def test_every_production_module_imports_in_a_clean_interpreter():
    modules = production_modules()
    assert len(modules) > 100 and "blackbox_pipeline.models.xgb" in modules
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(BANK_ROOT), os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, "-c", _IMPORT_EACH], input="\n".join(modules),
                         capture_output=True, text=True, encoding="utf-8", cwd=BANK_ROOT,
                         env=env, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    failures = json.loads(run.stdout.strip().splitlines()[-1])
    assert not failures, "\n  ".join(failures)


def test_no_production_module_names_a_project_module_that_does_not_exist():
    """Every absolute project import a module states — the branches importing
    never executes included, such as an ``except ImportError`` fallback — names
    a module that exists."""
    missing = []
    for name in production_modules():
        spec = importlib.util.find_spec(name)
        for node in ast.walk(ast.parse(Path(spec.origin).read_bytes())):
            targets = ([node.module] if isinstance(node, ast.ImportFrom) and not node.level
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            missing += [f"{name}, line {node.lineno}: {t}" for t in targets
                        if t and t.split(".")[0] in PROJECT_PACKAGES and importlib.util.find_spec(t) is None]
    assert not missing, "\n  ".join(missing)


# ----------------------------------------------------------------------
# the extractor itself: catches what it should, ignores what it should
# ----------------------------------------------------------------------
def _notebook(tmp_path, *cells):
    nb = {"cells": [{"cell_type": "code", "source": src, "metadata": {}, "outputs": [],
                     "execution_count": None} for src in cells]
                   + [{"cell_type": "markdown", "source": "from shared.gone import x", "metadata": {}}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "probe.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_extractor_reports_missing_modules_and_names(tmp_path):
    path = _notebook(
        tmp_path,
        "from shared.stage_runner import Stage3Block\n",          # module that no longer exists
        "from shared.stage_io import StageOutput, NoSuchThing\n",  # one good name, one missing
        "import shared.stage_io as sio\n",
    )
    imports = project_imports(path)
    assert [(i.cell, i.module, i.name) for i in imports] == [
        (0, "shared.stage_runner", "Stage3Block"),
        (1, "shared.stage_io", "StageOutput"),
        (1, "shared.stage_io", "NoSuchThing"),
        (2, "shared.stage_io", None),
    ]
    problems = unresolved_imports(imports)
    assert len(problems) == 2
    assert "module 'shared.stage_runner' not found" in problems[0]
    assert "has no 'NoSuchThing'" in problems[1]


def test_extractor_ignores_magics_third_party_markdown_and_guarded_fallbacks(tmp_path):
    path = _notebook(
        tmp_path,
        "%pip install -q pytest\n!ls\nimport numpy as np\nfrom sklearn.base import clone\n",
        "try:\n    from shared.stage_runner import Stage3Block\n"
        "except ImportError:\n    from shared.stage3 import Stage3Block\n",
        "if True:\n    %time x = 1\n",
    )
    imports = project_imports(path)
    assert [(i.module, i.name) for i in imports] == [("shared.stage3", "Stage3Block")]
    assert unresolved_imports(imports) == []


def test_extractor_reports_unparseable_cells(tmp_path):
    with pytest.raises(SyntaxError, match="cell 0 does not parse"):
        project_imports(_notebook(tmp_path, "def broken(:\n    pass\n"))
