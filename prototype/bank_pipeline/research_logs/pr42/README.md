# PR 42 research logs

Feature research on the training side
([`research_roadmap.md`](../../documentation/research_roadmap.md), PR 42). One folder per run, named
`c<commit>_<YYYYMMDD_HHMMSS>`, written once: an existing record is never replaced, and a new attempt gets a
new run id. The conventions are the PR 39 ones ([`../pr39/README.md`](../pr39/README.md)).

| Commit | Record in `<run_id>/` | Written by |
|---|---|---|
| Commit 2: training-side identities, stage feature populations, the per-feature inventory, the DAG, registry / call-site / fit-protocol / DAG checks, the label-permutation probe, the frozen references, the accepted artifacts read through the PR 38 boundary extractors, the duplicated implementations (pairs, callers, parity on training rows) | `feature_inventory.json` + `.sha256` | `feature_research/feature_research.ipynb`, Cells 1–13 |
| Commit 3, Step 3A: the time index (source order verified, periods, contact days), how the target-derived encodings are built, every fitting path, fold-isolation label flips and call sites, the accepted fits' fold scopes, the black-box MLP's early-stopping split against the Stage 1 encoders, the attribution of every encoded value to fit-row labels by contact timing (mechanisms a, b and c per path, held-out own labels measured, reconstruction coverage expected versus completed), threshold and constant provenance, the findings, and this run's inventory compared with the Commit 2 record | `encoding_audit.json` + `.sha256`, beside the same run's `feature_inventory.json` | the same notebook, Cells 14–21 |
| Commit 3, Step 3B: the registration (rules, cold start, windows, comparisons) and its digest, the accepted Stage 1 settings read, the past-only states verified against the production encoder, availability and fallbacks per rule and scope, the invariance flips, the predictive runs (protocol table, paired DeLong comparisons pooled and per segment, the 2 × 2 encoding × split factorial for LR and MLP, per-fold ROC-AUCs, every MLP fit's stopping epoch and split check), the lag-consistent chronological arm and its regression, the contract decisions and the proposed candidate contract, the Stage 2–3 extension, the PR 49 protocol, the Stage 2 prior-shift projection, the interim finding with its interpretation, and the accepted Step 3A run (unavailable if its directory is absent; otherwise both records and sidecars required and their integrity checked; exact reproduction only in the matching environment), checked before the record is written | `temporal_validity.json` + `.sha256`, beside the same run's two records | the same notebook, Cells 22–30 |

[`feature_inventory.md`](feature_inventory.md), beside this index, is the Commit 2 review document: the
inventory and frozen references as tables, generated from `feature_research.feature_inventory` and kept in step
with it by `tests/feature_research/test_pr42_feature_inventory.py`. [`encoding_audit.md`](encoding_audit.md) is
the Step 3A review document: the findings by mechanism, feature family and fitting path (with denominators), the
MLP early-stopping trace, their limits and where each piece of evidence is.
[`temporal_validity.md`](temporal_validity.md) is the Step 3B review document: what each past-only encoding knows
when a row is scored, availability, the predictive comparisons (encoding versus split, MLP early stopping), the
contract decisions, the PR 49 protocol and the interim finding. Its comparisons are differences between whole
protocols; it says what each does and does not isolate.

Since Commit 3 a run writes several records into its folder: the first creates the folder, which must not exist,
and each name is written once (`research_records.write_run_record`).

## What every record holds

| Key | Content |
|---|---|
| `run` | run id, commit, the notebook, UTC start and finish |
| `seeds` | every seed the run used, by name |
| `code` | Git HEAD, `uncommitted_tracked_changes`, the LF SHA-256 of each file in `research_records.CODE_FILES`, and the whole-tree digest of `upstream_regeneration.records.code_identity` |
| `environment` | Python, platform and package versions |
| `content_sha256` | SHA-256 of the content sections only (no run id, times, code or environment): a run that reproduces another has the same value |

`feature_research.research_records.read_run_record(run_id, name)` reads a record only if its `.sha256` sidecar
and its `content_sha256` both match. No record holds a test row, a test label, a test prediction or a test
metric: the notebook never binds them (G7).

`.gitattributes` turns off line-ending conversion inside run folders, so the records' bytes, and the hashes that
pin them, are the same on every checkout.
