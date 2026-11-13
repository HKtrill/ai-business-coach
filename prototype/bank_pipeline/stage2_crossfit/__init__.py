"""
stage2_crossfit
===============
The PR 38 corrected Stage 2 pathway: the runner that fits both arms under the
frozen PR 37 specification, and the safeguards that must pass before the
five-fold primary reruns (PR 38 roadmap, Commit 2).

Modules
-------
evidence
    Fit-time evidence of one outer fold of each arm (G4, G5, R1, R2, R5,
    D-C11), the selection records the frozen protocol classifies, and the
    JSON form the evidence is saved in.
runner
    Train-side data and identity checks, the strict primary and S1
    configurations (G1, G2, G9), the fit-only preflight (G3), the outer
    cross-fit and full-train fits, and versioned artifact writing (G5, G10).
acceptance
    Commit 3A after the primary: the G7 reproduction of one outer fold and
    structural acceptance V1-V8 of the candidate artifacts; the Step 3B entry
    point (``require_accepted``). No outer-OOF metric is computed.
complementarity
    Commit 3, Step 3B: the comparative analysis of an accepted run (protocol
    sections 3-12), gated by ``acceptance.require_accepted``; one write-once,
    versioned package of tables and provenance per run.
sensitivity
    Commit 3, Step 3C: the G12 sensitivity S1 (GLASS ``full``) for an accepted
    primary, its mandatory G6 Pass 1 invariance (S1 is accepted only if it
    passes), and the S1 package, reported apart from the primary comparison.
phase0
    Read-only historical integrity checks and the PR 38 output root (Phase 0).
g13
    The label-derived feature-engineering audit (G13).
inner_features
    The G13 correction, opt-in: inner-fold features without the inner fold's
    labels (``fold_label_excluded``), specified by ``records/``.

Run ``python -m stage2_crossfit --help`` from ``prototype/bank_pipeline``. The package
imports both arms, so it lives outside ``shared`` (which imports neither).
"""
