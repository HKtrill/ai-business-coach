"""Historical import path of ``glass_pipeline.glass_arbiter.router_evidence`` (PR 39, D5).

Importing it yields that module itself (not a copy), so frozen PR 37-38 code
and pickles that name ``glass_pipeline.meta_ebm.router_evidence`` resolve unchanged.
"""

import sys

from glass_pipeline.glass_arbiter import router_evidence as _module

sys.modules[__name__] = _module
