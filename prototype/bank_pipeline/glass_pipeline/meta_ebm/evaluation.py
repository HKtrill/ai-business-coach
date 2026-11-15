"""Historical import path of ``glass_pipeline.glass_arbiter.evaluation`` (PR 39, D5).

Importing it yields that module itself (not a copy), so frozen PR 37-38 code
and pickles that name ``glass_pipeline.meta_ebm.evaluation`` resolve unchanged.
"""

import sys

from glass_pipeline.glass_arbiter import evaluation as _module

sys.modules[__name__] = _module
