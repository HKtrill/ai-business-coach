# PR 39 research logs

The PR 39 records, under the frozen execution contract
([`pr39_execution_contract.md`](../../documentation/methodology/pr39_execution_contract.md),
section 7). One folder per run, named `c<commit>_<YYYYMMDD_HHMMSS>`, written once: an
existing record is never replaced, and a new attempt gets a new run id.

| Step | Record in `<run_id>/` | Written and checked by |
|---|---|---|
| Commit 2, Step 2A: Stage 1 calibration evidence E-C and the C1–C5 decision | `stage1_calibration_evidence.json` | `python -m stage1_calibration run` / `check --run-id <run_id>` |
| Commit 2, Step 2A: the owner's approval of that decision, pinning the evidence by SHA-256 | `stage1_calibration_decision.json` | `python -m stage1_calibration decide --run-id <run_id>`; re-verified by `check` |
| Commit 2, Step 2B: the Stage 1 rename's baseline re-check and its code-identity note (contract §7.5) | `stage2_baseline_check.txt` | `python research_logs/baselines/stage2_baseline.py --check`, output saved verbatim |
| Commit 2, Step 2B: test evidence for gates A2.4–A2.5 | `step2b_pytest.xml` | `pytest --junitxml` over the focused suites and the real-data rename and freeze tests |
| Commit 3, Step 3A: NSD/ECI binning evidence E-B and E-B2, triggers T1–T4, the retain-or-revise decision and the Stage 2 binning identity (gates A3.1–A3.3) | `stage2_binning_evidence.json` | `python -m stage2_binning run` / `check --run-id <run_id>` |

The Step 2B records go in their own run folder; the Step 2A folder is not reopened.

Row-level vectors stay local and Git-ignored under `models/pr39/<run_id>/`. The record
pins them by SHA-256 and size, and `check` recomputes every number in the record from them.
The Step 3A record keeps no row-level file: its `check` recomputes it from the training
rows and the accepted PR 38 run.

`commit3_readiness.md`, beside this index, is the Commit 3 readiness record (gate A3.5):
written with Commit 3, not by a run.

`.gitattributes` turns off line-ending conversion inside run folders, so the records'
bytes, and the hashes that pin them, are the same on every checkout.
