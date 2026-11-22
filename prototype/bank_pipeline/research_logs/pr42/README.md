# PR 42 research logs

Feature research on the training side
([`research_roadmap.md`](../../documentation/research_roadmap.md), PR 42). One folder per run, named
`c<commit>_<YYYYMMDD_HHMMSS>`, written once: an existing record is never replaced, and a new attempt gets a
new run id. The conventions are the PR 39 ones ([`../pr39/README.md`](../pr39/README.md)).

| Commit | Record in `<run_id>/` | Written by |
|---|---|---|
| Commit 2: training-side identities, stage feature populations, the per-feature inventory, the DAG, registry / call-site / fit-protocol / DAG checks, the label-permutation probe, the frozen references, the accepted artifacts read through the PR 38 boundary extractors, the duplicated implementations (pairs, callers, parity on training rows) | `feature_inventory.json` + `.sha256` | `feature_research/feature_research.ipynb`, Cells 1–13 |

[`feature_inventory.md`](feature_inventory.md), beside this index, is the Commit 2 review document: the
inventory and frozen references as tables, generated from `feature_research.feature_inventory` and kept in step
with it by `tests/feature_research/test_pr42_feature_inventory.py`.

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
