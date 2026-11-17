"""
shared.notebook_switches
========================
Cell-level switches for the research notebooks (PR 39, execution contract
section 5.3: the legacy Stage 2 cells stay behind an explicit switch that is
off by default).

    RUN_LEGACY_STAGE2 = False
    register_switch("legacy_stage2", "RUN_LEGACY_STAGE2")

registers the cell magic ``%%legacy_stage2``. A cell whose first line is that
magic runs only while the notebook variable named by the switch is ``True``.
While it is ``False`` the cell prints one line naming the cell and the switch
and runs nothing. The cell body below the magic line is kept exactly as it
was, so the switched-on pathway is the historical code, unchanged.

Fail closed: a switch that is not exactly ``True`` or ``False`` is refused,
and a switched-on body that raises re-raises in the cell (a notebook run stops
there, as it would have without the switch).
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["SwitchError", "register_switch", "cell_label"]

_HEADER = re.compile(r"^#\s*CELL\s+([0-9]+[A-Z]?)\s*:\s*(.*)$", re.M)


class SwitchError(RuntimeError):
    """A switch is undefined or not a boolean, or no IPython shell is running."""


def cell_label(cell: str) -> str:
    """``"CELL 13: ..."`` from a cell's ``# CELL <n>: <title>`` header, or ``"this cell"``."""
    m = _HEADER.search(cell or "")
    return f"CELL {m.group(1)}: {m.group(2).strip()}" if m else "this cell"


def register_switch(magic: str, flag: str, *, shell=None) -> None:
    """
    Register the cell magic ``%%<magic>``, gated by the notebook variable ``flag``.

    ``shell`` defaults to the running IPython shell. The flag is read when each
    gated cell runs, so the switch cell sets it once for the whole notebook.
    """
    if shell is None:
        try:
            from IPython import get_ipython
        except ImportError as exc:                              # pragma: no cover
            raise SwitchError("register_switch needs IPython (a notebook kernel)") from exc
        shell = get_ipython()
    if shell is None:
        raise SwitchError("register_switch needs a running IPython shell (a notebook kernel)")

    def gated(line: str, cell: str) -> Optional[object]:
        if line.strip():
            raise SwitchError(f"%%{magic} takes no arguments (got {line.strip()!r})")
        if flag not in shell.user_ns:
            raise SwitchError(f"%%{magic}: {flag} is not defined; run the cell that sets it")
        value = shell.user_ns[flag]
        if value is not True and value is not False:
            raise SwitchError(f"%%{magic}: {flag} must be True or False, got {value!r}")
        if value is False:
            print(f"skipped {cell_label(cell)} ({flag} is False)")
            return None
        result = shell.run_cell(cell, store_history=False)
        result.raise_error()
        return None

    gated.__name__ = magic
    gated.__doc__ = f"Run the cell only while {flag} is True (shared.notebook_switches)."
    shell.register_magic_function(gated, magic_kind="cell", magic_name=magic)
