"""
stage2_binning
==============
PR 39 Commit 3, Step 3A: the NSD and ECI binning revalidation of the frozen
execution contract (``documentation/methodology/pr39_execution_contract.md``
1.0.0, section 6.2; decision D6; gates A3.1-A3.3).

evidence
    Pure functions: per-bin statistics with Wilson intervals, the registered
    triggers T1-T4, the retain-or-revise outcome and the E-B2 summary.
runner
    The evidence run (training rows only, outer-fold engineer refits; the
    accepted PR 38 run's fit-time records for E-B2) and its write-once record,
    plus ``check``, which recomputes the record.

Run ``python -m stage2_binning --help`` from ``prototype/bank_pipeline``.
It reads both arms' PR 38 artifacts, so it lives outside ``shared``.
"""
