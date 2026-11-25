"""
PR 42 Commit 2: the frozen references held by local, Git-ignored files, verified
against the accepted PR 39 artifacts - opt-in, like every realdata test:

    pytest -m realdata tests/realdata/test_pr42_frozen_references.py

1. The four Stage 1-3 artifacts with a PR 38 boundary extractor (GLASS LR, both
   routers, GLASS EBM) pass it at the SHA-256 the PR 39 Commit 4 manifest pins,
   against the training side ``draw_training_side`` returns; the two black-box
   models, which have no extractor, are not opened. This is what the Feature
   Research Notebook's Cell 11 does.
2. Through the extractors only, the fields they expose carry the frozen
   contracts: the LR's Stage 1 feature names, and the EBM's 13 features,
   ``per_fold`` feature fit scope, shared 5-fold plan, per-fold calibration
   gates and the recorded block fingerprint.
3. The accepted Stage 3 block cache is the recorded file, by size and SHA-256.

Nothing here unpickles a model or the block cache outside the extractors: the
EBM's stored interaction pairs and the cache's stored implementation identity
are not exposed by them, and the frozen references state those from the
production code instead (``frozen_references.static_references``).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from feature_research import frozen_references as FR
from feature_research.training_side import draw_training_side
from tests.support.notebooks import BANK_ROOT

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def rows():
    train = draw_training_side(verbose=False)
    return {r["role"]: r for r in FR.accepted_artifact_views(train.train_reference)}


def test_the_accepted_artifacts_pass_their_boundary_extractors(rows):
    status = {role: r["status"] for role, r in rows.items()}
    assert status == {"stage1.lr": "passed", "stage2.glass_router": "passed",
                      "stage2.rf_router": "passed", "stage3.ebm": "passed",
                      "stage1.mlp": "not read", "stage3.xgb": "not read"}, json.dumps(rows)[:4000]
    for role, r in rows.items():
        if r["status"] == "passed":
            assert r["checks"] == r["checks_passed"] > 0, role


@pytest.mark.parametrize("role, checks", [
    ("stage1.lr", ("feature_names = shared.stage1 LR_FEATURES",)),
    ("stage3.ebm", ("features = shared.stage3 EBM_FEATURES", "feature_fit_scope is 'per_fold'",
                    "oof fold_id = fold_plan fold_id",
                    "per-fold calibration gates recorded (PR 36 F22)")),
])
def test_the_extractor_exposed_fields_carry_the_frozen_contracts(rows, role, checks):
    assert rows[role]["status"] == "passed", rows[role]
    passed = set(rows[role]["check_names"])
    assert set(checks) <= passed, sorted(set(checks) - passed)
    if role == "stage3.ebm":
        assert rows[role]["block_fingerprint"]["match"], rows[role]["block_fingerprint"]


def test_the_accepted_block_cache_file_is_the_recorded_one():
    manifest, _ = FR.read_record("c4_manifest")
    (entry,) = [f for f in manifest["inventory"]["files"] if f["path"] == FR.BLOCK_CACHE]
    path = BANK_ROOT / FR.BLOCK_CACHE
    assert path.stat().st_size == entry["bytes"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == entry["sha256"]
