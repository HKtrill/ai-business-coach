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
| Commit 4, Step 4A: §7.5 checks before any write, the `models/` inventory, the run id and output layout (A4.1) | `preflight.json` | `python -m upstream_regeneration preflight` |
| Commit 4, Step 4A: the §5.2 Stage 2 decision, reuse or regenerate (A4.2) | `stage2_reuse.json` | `python -m upstream_regeneration stage2-reuse` |
| Commit 4, Step 4A: each notebook's fresh-kernel execution, Stage 4 excluded (§5.4): truncated-copy and executed-copy hashes, the inventory change | `execution_glass.json`, `execution_blackbox.json` | `python -m upstream_regeneration execute` |
| Commit 4, Step 4A: the files the black-box notebook read from the GLASS run (`$BANK_UPSTREAM_PINS`) | `upstream_pins.json` | written by `execute --notebook glass` (or `recover-glass`) |
| Commit 4, Step 4A: a completed GLASS execution refused only for an output since registered, re-verified from its notebook copies, artifacts and inventory, in place of `execution_glass.json`; names the failed record by SHA-256 and records the run's and the recovered code identities | `execution_glass_recovery.json` | `python -m upstream_regeneration recover-glass` |
| Commit 4, Step 4A: the owner-authorized code transition after a failed `finalize` (the A4.4 Stage 3 comparison across the two registered split-fingerprint schemes): names the failed attempt by SHA-256 without reading it, records both code identities and every changed file, and verifies the run's records, model folders and pins are unchanged and the corrected A4.4 passes | `finalize_transition.json` | `python -m upstream_regeneration finalize-transition` |
| Commit 4, Step 4A: Stage 1–3 checks, Stage 4 input builds, §7.5 after the run, every artifact's path, size and SHA-256 (A4.3–A4.6); not an acceptance (that is Step 4B, A4.7) | `run_record.json` | `python -m upstream_regeneration finalize` |
| Commit 4, Step 4B: the PR 39 upstream artifact manifest (§7.1): the six Stage 1–3 roles with path, size, SHA-256, producer and consumers, and every file the run added or changed; a valid `pr39.upstream_pins/1` file, read by Commit 5 through `pinned_path(role, pins_file=…)` (A5.1) | `artifact_manifest.json` | written by `python -m upstream_regeneration.acceptance accept` |
| Commit 4, Step 4B: the explicit Stage 1–3 acceptance decision (A4.7): every record and link re-verified, the code chain re-derived, A4.2–A4.6 recomputed and compared with their records, the D7 disposition, the owner's decision and the §7.1 and §7.2 run-specific dispositions; `run_record.json` keeps `accepted: false` | `upstream_acceptance.json` | `python -m upstream_regeneration.acceptance accept` / `check --run-id <run_id>` |

The Step 2B records go in their own run folder; the Step 2A folder is not reopened.

Row-level vectors stay local and Git-ignored under `models/pr39/<run_id>/`. The record
pins them by SHA-256 and size, and `check` recomputes every number in the record from them.
The Step 3A record keeps no row-level file: its `check` recomputes it from the training
rows and the accepted PR 38 run.

`commit3_readiness.md`, beside this index, is the Commit 3 readiness record (gate A3.5):
written with Commit 3, not by a run.

Every Commit 4 record has a `.sha256` sidecar, and each step reads the earlier ones only
if their sidecars match. A step whose checks fail writes `<run_id>/failed/<step>_<utc>.json`
instead, which `.gitignore` keeps local; the run then stops and a new attempt gets a new
run id. The two registered continuations, `recover-glass` and `finalize-transition`, are
the exceptions: each is written once for its registered cause and code change, and
`finalize` checks that the failed attempts they name are unchanged.

`.gitattributes` turns off line-ending conversion inside run folders, so the records'
bytes, and the hashes that pin them, are the same on every checkout.
