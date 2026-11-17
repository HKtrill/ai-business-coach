"""
glass_pipeline.glass_arbiter.artifacts
======================================
Identity and persistence of the Stage 4 GLASS Arbiter artifact.

PR 39 (execution contract decision D5) renamed the arbiter, formerly
"Meta-EBM" (``glass_pipeline.meta_ebm``). Only names changed: the payload,
the algorithm and every number are those of schema ``meta_ebm/2``.

    current (PR 39)                               historical (PR 36-38)
    schema    glass_arbiter/1                     meta_ebm/2
    file      models/glass_arbiter/               models/meta_ebm/
              glass_arbiter_<timestamp>.joblib    meta_ebm_<timestamp>.joblib
    upstream  stage4_inputs["upstream"]["arbiter"] = "glass_arbiter" / "meta_ebm"

Historical artifacts are read as they were saved, never rewritten or
renamed: every reader accepts ``READABLE_SCHEMAS`` and takes the arbiter id an
artifact records from its own schema (``SCHEMA_ARBITER``).
"""

import os
from datetime import datetime

import joblib

#: The arbiter id new artifacts record in ``stage4_inputs["upstream"]``.
ARBITER = "glass_arbiter"
ARTIFACT_SCHEMA = "glass_arbiter/1"
ARTIFACT_DIR = "./models/glass_arbiter"
ARTIFACT_PREFIX = "glass_arbiter_"

#: PR 36-38 identifiers, readable through PR 39 and beyond.
HISTORICAL_ARBITER = "meta_ebm"
HISTORICAL_ARTIFACT_SCHEMA = "meta_ebm/2"
HISTORICAL_ARTIFACT_DIR = "./models/meta_ebm"
HISTORICAL_ARTIFACT_PREFIX = "meta_ebm_"

#: schema -> the upstream arbiter id an artifact of that schema records.
SCHEMA_ARBITER = {ARTIFACT_SCHEMA: ARBITER, HISTORICAL_ARTIFACT_SCHEMA: HISTORICAL_ARBITER}
READABLE_SCHEMAS = tuple(SCHEMA_ARBITER)
#: File name patterns of both generations (current first).
ARTIFACT_PATTERNS = (f"{ARTIFACT_PREFIX}*.joblib", f"{HISTORICAL_ARTIFACT_PREFIX}*.joblib")


def save_glass_arbiter(artifact, base_path=ARTIFACT_DIR):
    """Write ``artifact`` as ``<base_path>/glass_arbiter_<timestamp>.joblib``."""
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(base_path, f"{ARTIFACT_PREFIX}{ts}.joblib")
    joblib.dump(artifact, path)
    return path


#: PR 36-38 name, kept as an alias through PR 39 (D5). It writes the current
#: schema and file name.
save_meta_ebm = save_glass_arbiter
