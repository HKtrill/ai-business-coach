"""
phase1_reproduction
===================
PR 39 Commit 6, Step 6B: one full fresh-kernel execution of each notebook on
the final code, as a reproduction, and the Phase 1 freeze (execution contract
1.0.0, sections 5.4, 7.3-7.5 and 12; gates A6.1-A6.5; D4, D7, D8). Run from
``prototype/bank_pipeline``, one run id throughout, without committing until
``freeze`` has passed:

    python -m phase1_reproduction preflight --run-id c6_<YYYYMMDD_HHMMSS> --sums <SHA256SUMS.txt>
    (the realdata, fast, slow and notebook suites, --junitxml into research_logs/pr39/<run>/)
    python -m phase1_reproduction execute   --run-id <id> --notebook glass    --out-dir <outside the repo>
    python -m phase1_reproduction execute   --run-id <id> --notebook blackbox --out-dir <outside the repo>
    python -m phase1_reproduction compare   --run-id <id> --sums <SHA256SUMS.txt>
    python -m phase1_reproduction freeze    --run-id <id> --statements <statements.json>

``preflight``
    The accepted baseline re-verified: the Commit 4 acceptance and manifest
    (``acceptance.check``) and the Commit 5 crossover baseline, every result
    recomputed (``crossover.check``). The notebooks and production code are
    the ones Step 5A executed (its code identity re-derived). Section 7.5
    before anything is written, the ``models/`` inventory as Step 5A left it,
    ``git status``, the Git-tracked files the notebooks write, both full
    copies, and the environment against the accepted runs'.
``execute`` (section 5.4, D4)
    The committed notebook, every cell, in a new kernel (the Commit 4 vehicle,
    ``vehicle.execute``); the executed copy outside the repository. GLASS runs
    first, with no pins file. Its Stage 1 LR, EBM and GLASS Arbiter, and the
    two Stage 2 pins, form the run's pins file, which the black-box notebook
    reads (Cells 14A, 33 and 35). The executed copy must show every code cell
    run once, in order. The model folders may change only by the notebook's
    registered new versions, sidecars and mutable aliases: every accepted
    version stays as it is. The tracked files a notebook rewrites (GLASS
    Cell 20's figure, black-box Cell 14A's D7 CSVs) are recorded before and
    after and copied out; nothing is restored or deleted.
``compare`` (section 7.4, A6.3, A6.4)
    Each reproduced artifact against its accepted counterpart
    (``compare``), the registered statements, the 2x2 recomputed on the
    reproduced arbiters, the reproduced artifacts' Stage 1-4 contract checks,
    the identities, and section 7.5 after the run. ``reproduction.json``
    records the evaluation and its class, whatever the class.
``freeze`` (A6.1, A6.2, A6.5)
    The saved suite reports, the A6.2 route (reproduced, or the environmental
    exception with the owner's recorded disposition), the tracked outputs'
    disposition, and ``phase1_freeze.json``: the accepted baseline, both
    executions, the limitations (with the correction of the Commit 5 record's
    third limitation) and the Phase 2 handoff, with the owner's statements
    verbatim.

The reproduction's artifacts are new immutable versions; they never replace
the accepted ones (section 5.4). Records are written once with ``.sha256``
sidecars; a failed step writes ``failed/<step>_<utc>.json`` and stops.
"""

__all__ = ["compare", "runner"]
