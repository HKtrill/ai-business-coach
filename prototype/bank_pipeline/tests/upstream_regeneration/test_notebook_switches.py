"""
PR 39 Step 4A: the legacy Stage 2 switch (shared.notebook_switches), run in a
real IPython shell.

Off (the default): a gated cell runs nothing and says so. On: the body runs as
an ordinary cell, unchanged, and an error in it stops the run. A switch that is
undefined or not a boolean is refused, and the magic takes no arguments.
"""

from __future__ import annotations

import pytest

IPython = pytest.importorskip("IPython")

from shared.notebook_switches import SwitchError, cell_label, register_switch  # noqa: E402

BODY = "# ====\n# CELL 13: GLASS ROUTER — FIT\n# ====\nFITTED = FITTED + 1\n"


@pytest.fixture
def shell():
    from IPython.core.interactiveshell import InteractiveShell
    sh = InteractiveShell.instance()
    sh.user_ns.pop("RUN_LEGACY_STAGE2", None)
    sh.user_ns["FITTED"] = 0
    register_switch("legacy_stage2", "RUN_LEGACY_STAGE2", shell=sh)
    yield sh
    sh.user_ns.pop("RUN_LEGACY_STAGE2", None)


def test_off_runs_nothing_and_names_the_cell(shell, capsys):
    shell.user_ns["RUN_LEGACY_STAGE2"] = False
    shell.run_cell_magic("legacy_stage2", "", BODY)
    assert shell.user_ns["FITTED"] == 0
    assert "skipped CELL 13: GLASS ROUTER — FIT (RUN_LEGACY_STAGE2 is False)" in capsys.readouterr().out


def test_on_runs_the_unchanged_body_and_an_error_stops_the_run(shell):
    shell.user_ns["RUN_LEGACY_STAGE2"] = True
    shell.run_cell_magic("legacy_stage2", "", BODY)
    assert shell.user_ns["FITTED"] == 1
    with pytest.raises(ZeroDivisionError):
        shell.run_cell_magic("legacy_stage2", "", "# CELL 14: X\n1 / 0\n")


@pytest.mark.parametrize("value", [None, 0, 1, "False", "no"])
def test_a_switch_that_is_not_exactly_a_boolean_is_refused(shell, value):
    shell.user_ns["RUN_LEGACY_STAGE2"] = value
    with pytest.raises(SwitchError, match="must be True or False"):
        shell.run_cell_magic("legacy_stage2", "", BODY)
    assert shell.user_ns["FITTED"] == 0


def test_an_undefined_switch_or_an_argument_is_refused(shell):
    with pytest.raises(SwitchError, match="RUN_LEGACY_STAGE2 is not defined"):
        shell.run_cell_magic("legacy_stage2", "", BODY)
    shell.user_ns["RUN_LEGACY_STAGE2"] = True
    with pytest.raises(SwitchError, match="takes no arguments"):
        shell.run_cell_magic("legacy_stage2", "on", BODY)
    assert shell.user_ns["FITTED"] == 0


def test_without_a_shell_registration_is_refused(monkeypatch):
    import IPython as ip
    monkeypatch.setattr(ip, "get_ipython", lambda: None)
    with pytest.raises(SwitchError, match="running IPython shell"):
        register_switch("legacy_stage2", "RUN_LEGACY_STAGE2")


def test_cell_label_reads_the_cell_header():
    assert cell_label(BODY) == "CELL 13: GLASS ROUTER — FIT"
    assert cell_label("x = 1") == "this cell"
