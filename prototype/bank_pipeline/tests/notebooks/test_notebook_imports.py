"""
Notebook import contract: every project import in the active research
notebooks still resolves. Notebook cells are parsed, never executed.

This is the safety net for Commit 4: deleting or renaming a module that a
notebook still imports fails here instead of mid-run in a clean kernel.
"""

from __future__ import annotations

import json

import pytest

from tests.support.notebooks import (
    ACTIVE_NOTEBOOKS, BANK_ROOT, project_imports, unresolved_imports,
)


@pytest.mark.parametrize("notebook", ACTIVE_NOTEBOOKS)
def test_active_notebook_project_imports_resolve(notebook):
    path = BANK_ROOT / notebook
    assert path.exists(), f"active notebook missing: {notebook}"
    imports = project_imports(path)
    assert imports, f"{notebook}: no project imports found (extractor broken?)"
    problems = unresolved_imports(imports)
    assert not problems, f"{notebook}:\n  " + "\n  ".join(problems)


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
