# PR 38 research logs

Stage 2 corrected cross-fit and router complementarity. One folder per run.

**Start here: [`c3_20261112_153333/report_v1/README.md`](c3_20261112_153333/report_v1/README.md)**,
the research report of the accepted Commit 3 run: GLASS vs RF registered outcomes with
intervals and D-C1 statements, routing, complementarity, matched budget, pre/post correction,
contract compliance and the S1 sensitivity, with an index of its eleven tables.

| Run | What it holds |
|---|---|
| `c3_20261112_153333` | **Authoritative.** Commit 3: the accepted corrected five-fold primary (G13 pathway `fold_label_excluded`), its complementarity analysis (Step 3B) and the S1 sensitivity (Step 3C). |
| `s2a_20261111_155321` | Commit 2, Step 2A: Phase 0, configuration and the original (`outer_engineer`) G13 audit; the authorization of `c3_20261112_153333` pins its `g13_fold0.json`. |
| `s2b_20261111_200651` | Commit 2, Step 2B: test report. |

## What is committed

| Where | What | Git |
|---|---|---|
| `<run>/report_v1/` | The research report: eleven CSV tables (one row per source row, each naming its source), three JSON summaries, its README, a byte-exact copy of each v1 package manifest, and its MANIFEST. | committed |
| `<run>/*.json`, `<run>/pytest_fast.xml` | Records the acceptance and S1 gates re-verify by path and SHA-256: Phase 0, configuration, G3, G13, G8 test report, authorization, primary and S1 attempts and manifests, G7, owner dispositions (G10/F31), structural acceptance V1-V8, S1 acceptance. | committed, byte-exact |
| `<run>/complementarity_v1/`, `<run>/s1_sensitivity_v1/` | The v1 packages Step 3B / 3C wrote: the working sources `report` migrates. | local; optional after the migration |
| `<run>/*_failed_*.json`, stop records | Failed or stopped attempts; never read by code. | local |
| `models/pr38/<run>/` | Model artifacts and run evidence. | local (repository `.gitignore`) |

`report_v1/MANIFEST.json` keeps two kinds of checksum apart: the historical SHA-256 of every
v1 package table (`historical_v1_sources`, as the generators wrote them) and the SHA-256 of
every report file (`outputs`).

## Making and checking a report

From `prototype/bank_pipeline`:

    python -m stage2_crossfit report --run-id <run_id>          # migration: writes report_v1/ once
    python -m stage2_crossfit report --run-id <run_id> --check  # verification, without the packages

The **migration** reads the run's v1 packages and reproduces every package table byte for
byte, from the report's own rows or (joins, a duplicate, a restated record) by the
generators' own code, before it writes anything. It deletes nothing.

The **verification** reads no v1 package. It checks every report file against the report's
MANIFEST and the run's accepted records against the ones it lists, reproduces every v1 table
from the report (and the frozen pre-correction baseline) against its historical SHA-256, and
rebuilds the report byte for byte.

Once the migration has succeeded, the v1 package folders can be removed. The packages' own
`--check` (`complementarity`, `s1-report`) needs their tables, so it runs only while they
exist.

For a later run, the packages need not be written into the run folder at all:

    python -m stage2_crossfit complementarity --run-id <run_id> --sums <SUMS> --packages-dir <tmp>
    python -m stage2_crossfit s1-report       --run-id <run_id> --sums <SUMS> --packages-dir <tmp>
    python -m stage2_crossfit report          --run-id <run_id> --packages-dir <tmp>

then `<tmp>` can be removed.

## Byte-exact records

`.gitattributes` turns off line-ending conversion for every file inside a run folder: other
records and the report pin these files by SHA-256, and `core.autocrlf` would change their
bytes.
