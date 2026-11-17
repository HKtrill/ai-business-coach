"""
stage4_regeneration
===================
PR 39 Commit 5, Step 5A: the four Stage 4 arbiters, fitted on the accepted
Commit 4 upstream (execution contract 1.0.0, sections 5.1 item 5, 5.4, 6.5
and 7.1-7.5; gates A5.1-A5.3). Run from ``prototype/bank_pipeline``, in this
order, with one run id throughout:

    python -m stage4_regeneration preflight --run-id c5_<YYYYMMDD_HHMMSS> --sums <SHA256SUMS.txt>
    python -m stage4_regeneration execute   --run-id <id> --notebook glass    --out-dir <outside the repo>
    python -m stage4_regeneration execute   --run-id <id> --notebook blackbox --out-dir <outside the repo>
    python -m stage4_regeneration finalize  --run-id <id> --sums <SHA256SUMS.txt>

``preflight`` (A5.1, section 7.5 before)
    Re-verifies the Commit 4 acceptance of ``c4_20261115_204125`` and its
    artifact manifest (``upstream_regeneration.acceptance.check``): the six
    Stage 1-3 roles, each on disk with its pinned size and SHA-256. Then runs
    the section 7.5 checks before anything is written: the snapshot, the
    baseline CLIs, ``report --check``, the PR 38 run's structural acceptance,
    a SHA-256 inventory of the model folders, and ``git status``.
``execute`` (section 5.4, D4)
    One fresh-kernel execution of a notebook's Stage 4 copy
    (``stage4_copy``). The copy rebuilds the split with the notebook's own
    cells, binds Stages 1-3 by pin, and runs the Stage 4 fitting cells
    unchanged: GLASS Cell 19, then black-box Cells 35-37. Stages 1-3 are not
    retrained. The GLASS cell reads the accepted manifest. Its run then writes
    the run's pins file, which holds the manifest's six roles verbatim and the
    new GLASS Arbiter (``stage4.glass_arbiter``, read by black-box Cell 35).
    The black-box cells read that pins file, and their run writes the PR 39
    arbiter manifest (section 7.1), the file ``$BANK_ARTIFACT_MANIFEST``
    names. Each execution may add only its registered arbiter versions.
``finalize`` (A5.1-A5.3, section 7.5 after)
    Checks each arbiter is unchanged and that every Stage 1-3 file it
    recorded consuming is the manifest's pin. Runs the real-data audit
    (``tests/realdata/test_saved_artifacts.py``) under the arbiter manifest:
    identity, evidence, upstream reload, fresh reload, serving, and both 2x2
    columns with the upstream held fixed. Its JUnit report is kept. Then runs
    the section 7.5 checks after the run and writes the run record, which
    holds every arbiter's path, size and SHA-256 and the section 7.3
    identities.

Records go to ``research_logs/pr39/<run_id>/`` and are written once, each
with a ``.sha256`` sidecar. A step whose checks fail writes
``failed/<step>_<utc>.json`` (Git-ignored, never read by code) and stops.
Nothing here writes inside ``research_logs/pr38``, ``models/pr38``, the
baselines, the Commit 4 records or the committed notebooks. Step 5B (the
registered comparisons, A5.4; the abstention accounting; the baseline record,
A5.5) reads the four arbiters this step pins and fits nothing.
"""

__all__ = ["stage4_copy", "runner"]
