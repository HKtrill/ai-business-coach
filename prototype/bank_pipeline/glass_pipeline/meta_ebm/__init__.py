"""
glass_pipeline.meta_ebm
=======================
Historical import path of the GLASS Arbiter (PR 36-38 "Meta-EBM"); the code
lives in ``glass_pipeline.glass_arbiter`` (PR 39, decision D5).

Kept so that historical callers and artifacts resolve unchanged: frozen PR 37
and PR 38 tooling imports ``glass_pipeline.meta_ebm.<module>``, and anything
pickled under that path unpickles. Each submodule here replaces itself, on
import, with the ``glass_pipeline.glass_arbiter`` module of the same name (one
object, not a copy), so no code is duplicated and types compare equal. New
code imports ``glass_pipeline.glass_arbiter``.
"""

from glass_pipeline.glass_arbiter import *            # noqa: F401,F403
from glass_pipeline.glass_arbiter import __all__      # noqa: F401
