"""
stage1_calibration
==================
PR 39 Commit 2, Step 2A: the Stage 1 calibration evidence (E-C) and the
registered decision rule C1-C5 of the frozen execution contract
(``documentation/methodology/pr39_execution_contract.md`` 1.0.0, section 6.1).

evidence
    Pure functions of training-side out-of-fold vectors: the metrics, the
    registered paired bootstrap, the Stage 4 input diagnostic and the rule.
runner
    The evidence run (both arms' own Stage 1 fits, training rows only) and its
    write-once record, plus ``check``, which recomputes the record.

Run ``python -m stage1_calibration --help`` from ``prototype/bank_pipeline``.
The package imports both arms, so it lives outside ``shared``.
"""
