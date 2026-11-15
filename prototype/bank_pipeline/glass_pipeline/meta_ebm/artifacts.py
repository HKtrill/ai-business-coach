"""Historical import path of ``glass_pipeline.glass_arbiter.artifacts`` (PR 39, D5).

Importing it yields that module itself (not a copy), so frozen PR 37-38 code
and pickles that name ``glass_pipeline.meta_ebm.artifacts`` resolve unchanged.
"""

import sys

from glass_pipeline.glass_arbiter import artifacts as _module

sys.modules[__name__] = _module
