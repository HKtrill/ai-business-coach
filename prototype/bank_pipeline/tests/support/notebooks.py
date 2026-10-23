"""
Static notebook inspection: read code cells as source, never execute them.

``project_imports`` lists the imports of this repository's own packages that a
notebook makes; ``unresolved_imports`` reports the ones that no longer resolve
(a deleted module, or a name removed from a module). Commit 4 cleanup relies on
this to prove that removing code does not break an active notebook.

Deliberately lenient where notebooks legitimately differ from modules:
IPython magics and shell lines (``%pip``, ``!ls``) are ignored, third-party
imports are out of scope, and imports inside ``try`` blocks that catch an
import failure are treated as optional fallbacks.

Public API
----------
PROJECT_PACKAGES      top-level packages that belong to this repository
ACTIVE_NOTEBOOKS      the research notebooks, relative to bank_pipeline
NotebookImport        one import statement found in a notebook
project_imports(path) imports of PROJECT_PACKAGES, in cell order
unresolved_imports(imports)  human-readable problems, empty when all resolve
read_before_bound(path)      names a code cell reads that only a later cell binds
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import builtins
import json
import re
import symtable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BANK_ROOT = Path(__file__).resolve().parents[2]

PROJECT_PACKAGES = ("shared", "data", "glass_pipeline", "blackbox_pipeline", "feature_research")

ACTIVE_NOTEBOOKS = (
    "glass_pipeline/glass_cascade.ipynb",
    "blackbox_pipeline/blackbox_cascade.ipynb",
    "feature_research/exploratory_feature_research.ipynb",
)

_IPYTHON_LINE = re.compile(r"^(\s*)[%!]")
_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


@dataclass(frozen=True)
class NotebookImport:
    """``from module import name`` (or ``import module`` when name is None)."""

    cell: int
    line: int
    module: str
    name: Optional[str] = None

    def __str__(self) -> str:
        what = f"from {self.module} import {self.name}" if self.name else f"import {self.module}"
        return f"cell {self.cell}, line {self.line}: {what}"


def _python_source(cell_source: str) -> str:
    """Replace IPython-only lines with ``pass`` so line numbers stay put."""
    return "\n".join(
        _IPYTHON_LINE.sub(r"\1pass  # ", line) if _IPYTHON_LINE.match(line) else line
        for line in cell_source.splitlines()
    )


def _guarded_lines(tree: ast.AST) -> set:
    """Line numbers of statements inside ``try`` blocks that catch import failures."""
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = set()
        for handler in node.handlers:
            if handler.type is None:
                caught.add("BaseException")
            for t in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]):
                if isinstance(t, ast.Name):
                    caught.add(t.id)
        if caught & _IMPORT_ERRORS:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if hasattr(sub, "lineno"):
                        guarded.add(sub.lineno)
    return guarded


def project_imports(path) -> List[NotebookImport]:
    """Imports of PROJECT_PACKAGES made by the notebook's code cells, in order."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    found: List[NotebookImport] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(_python_source(source))
        except SyntaxError as exc:
            raise SyntaxError(f"{Path(path).name}: cell {i} does not parse: {exc}") from exc
        guarded = _guarded_lines(tree)
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) in guarded:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in PROJECT_PACKAGES:
                        found.append(NotebookImport(i, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] in PROJECT_PACKAGES:
                    for alias in node.names:
                        found.append(NotebookImport(i, node.lineno, node.module,
                                                    None if alias.name == "*" else alias.name))
    return sorted(found, key=lambda x: (x.cell, x.line))


def unresolved_imports(imports: List[NotebookImport]) -> List[str]:
    """One message per import that does not resolve; empty when all do."""
    problems: List[str] = []
    for imp in imports:
        try:
            if importlib.util.find_spec(imp.module) is None:
                problems.append(f"{imp} -> module {imp.module!r} not found")
                continue
            if imp.name is None:
                continue
            module = importlib.import_module(imp.module)
            if hasattr(module, imp.name):
                continue
            is_package = hasattr(module, "__path__")
            if not (is_package and importlib.util.find_spec(f"{imp.module}.{imp.name}")):
                problems.append(f"{imp} -> {imp.module!r} has no {imp.name!r}")
        except Exception as exc:  # noqa: BLE001 — a broken import is a finding
            problems.append(f"{imp} -> {type(exc).__name__}: {exc}")
    return problems


_IPYTHON_NAMES = {"display", "get_ipython", "In", "Out", "exit", "quit"}
_COMPREHENSIONS = {"listcomp", "setcomp", "dictcomp", "genexpr"}


def _nested_scopes(table) -> list:
    """Child scopes whose reads count as the cell's: comprehensions (separate
    scopes until Python 3.12 inlined them) and module annotations (deferred to
    an ``__annotate__`` scope since 3.14; Python 3.13 and earlier evaluate them
    when the cell runs, so they are reads on every version)."""
    return [c for c in table.get_children()
            if c.get_name() in _COMPREHENSIONS or c.get_type() == "annotation"]


def _reads_and_binds(source: str) -> tuple:
    """
    Module-level names a cell reads and binds, the same on every Python version:
    comprehension and annotation reads included, function bodies not (they run
    when called). Only names the cell spells are reads, never a symbol the
    compiler adds (3.14's ``__conditional_annotations__`` for any module
    annotation); a name free in a comprehension is bound by an enclosing
    comprehension, not the module (3.11).
    """
    code = _python_source(source)
    table = symtable.symtable(code, "<cell>", "exec")
    binds = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
    binds |= {c.get_name() for c in table.get_children()
              if c.get_type() in ("function", "class") and c.get_name() not in _COMPREHENSIONS}
    reads = {s.get_name() for s in table.get_symbols() if s.is_referenced()}
    todo = _nested_scopes(table)
    while todo:
        scope = todo.pop()
        reads |= {s.get_name() for s in scope.get_symbols()
                  if s.is_referenced() and s.is_global() and not s.is_assigned()}
        todo += _nested_scopes(scope)
    spelled = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
    return reads & spelled, binds


def read_before_bound(path) -> List[tuple]:
    """
    ``(cell, name, first binding cell or None)`` for every module-level name a
    code cell reads before any earlier (or the same) cell binds it — a
    top-to-bottom run in a clean kernel would raise ``NameError`` there.
    Conditional bindings count as bindings; guarding them is checked separately.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [(i, "".join(c.get("source", []))) for i, c in enumerate(nb.get("cells", []))
             if c.get("cell_type") == "code"]
    parsed = [(i, *_reads_and_binds(src)) for i, src in cells]
    first: dict = {}
    for i, _, binds in parsed:
        for name in binds:
            first.setdefault(name, i)
    known = set(dir(builtins)) | _IPYTHON_NAMES
    return [(i, name, first.get(name)) for i, reads, binds in parsed
            for name in sorted(reads - binds - known) if first.get(name) is None or first[name] > i]
