"""
upstream_regeneration
=====================
PR 39 Commit 4, Step 4A: Stage 1-3 regeneration readiness and execution
(execution contract 1.0.0, sections 5.1-5.4, 7.1-7.5; gates A4.1-A4.6).
Run from ``prototype/bank_pipeline``, in this order, one run id throughout:

    python -m upstream_regeneration preflight    --run-id c4_<YYYYMMDD_HHMMSS> --sums <SHA256SUMS.txt>
    python -m upstream_regeneration stage2-reuse --run-id <id>
    python -m upstream_regeneration execute      --run-id <id> --notebook glass    --out-dir <outside the repo>
    python -m upstream_regeneration execute      --run-id <id> --notebook blackbox --out-dir <outside the repo>
    python -m upstream_regeneration finalize     --run-id <id> --sums <SHA256SUMS.txt>

Two registered continuations keep a run whose evidence is intact instead of
re-executing it, each written once, naming the failed attempt by SHA-256 and
recording both code identities: ``recover-glass`` (``recovery``: a completed
GLASS execution refused for an output since registered) and
``finalize-transition`` (``transition``: the A4.4 split-scheme comparison
corrected after a failed ``finalize``, which then rechecks A4.3-A4.6 in full).

``preflight`` (A4.1)
    The section 7.5 checks before anything is written: the snapshot, the
    baseline CLIs, ``report --check`` and the structural acceptance of the
    PR 38 run, a SHA-256 inventory of ``models/`` and
    ``blackbox_pipeline/models/``, ``git status``; the run id, the record
    folder and the output layout.
``stage2-reuse`` (A4.2)
    The section 5.2 decision on the final code: the four regeneration
    triggers and the four reuse checks. ``reuse`` pins the accepted run;
    anything else is ``regenerate`` and stops the run (``shared.pinned_artifacts``).
``execute`` (section 5.4)
    One fresh-kernel execution of a notebook, Stage 4 cells excluded by a
    truncated copy whose remaining cell sources are byte-identical and hashed,
    the executed copy written outside the repository. The inventory before and
    after must show only that notebook's registered outputs and mutable
    aliases. After the GLASS notebook the run's pins file is written; the
    black-box notebook reads the GLASS LR and EBM from it.
``finalize`` (A4.3-A4.6)
    Stage 1 and Stage 3 checks for both arms, ``build_stage4_inputs`` for all
    four arbiter-upstream combinations (inputs only: nothing is fitted), the
    fresh-interpreter reload, the section 7.5 checks after the run, and the
    versioned run record with every artifact's path, size and SHA-256.

Records go to ``research_logs/pr39/<run_id>/`` and are written once. A step
whose checks fail writes ``failed/<step>_<utc>.json`` (Git-ignored, never read
by code) and stops; nothing here marks an artifact accepted, which is Step 4B
(A4.7). Nothing here fits a Stage 4 model, reads a Stage 4 result, or writes
inside ``research_logs/pr38``, ``models/pr38`` or the baselines.
"""

__all__ = ["records", "inventory", "vehicle", "checks", "runner", "recovery",
           "transition"]
