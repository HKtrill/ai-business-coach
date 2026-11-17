# PR 39 — Commit 3 readiness record (gate A3.5)

Execution contract 1.0.0, §5.2, §5.4, §6.6 and §10. Written with Commit 3,
before any Commit 4 work. It determines the §5.2 outcome; it does not accept
artifact reuse, which only the Commit 4A checks can do.

## Decisions resolved

| Decision | Resolution | Where |
|---|---|---|
| D1 | Stage 1 calibration: `sigmoid` in both arms (C1–C5) | Commit 2A, run `c2_20261114_184433` |
| D2 | (a) Stage 1 only: three features renamed, Stage 1 feature contract 2.0.0, LR / MLP artifact version 3.0; historical 2.0 artifacts read as saved; Stage 2 and 3 names unchanged | Commit 2B |
| D3 | Reuse of the accepted PR 38 Stage 2 run `c3_20261112_153333` stays eligible (below); accepted only by the Commit 4A checks | Commit 4A |
| D4 | The vehicle of §5.4 and the notebooks' Stage 2 source of §5.3, as approved | Implemented in Commit 4A |
| D5 | GLASS Arbiter names (§6.6): `glass_pipeline.glass_arbiter`, schema `glass_arbiter/1`, `models/glass_arbiter/`, `stage4_crossover/glass_arbiter_on_blackbox/`, audit keys `glass_arbiter` / `glass_arbiter_on_blackbox`; every PR 36–38 identifier readable as saved | Commit 3B |
| D6 | NSD and ECI: no trigger fired; thresholds retained; Stage 2 binning identity `9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376` | Commit 3A, run `c3_20261114_231604` |

## §5.2 outcome: no regeneration trigger

| Trigger | After Commit 3 |
|---|---|
| An NSD/ECI revision is adopted | No: both retained (3A) |
| A Stage 2 feature definition, name or binning identity differs from the accepted run's | No: Stage 2 names unchanged (D2 (a)); `BINNING_STRATEGY` and `RF_FEATURES_BINARY` unchanged since `ff22505` (PR 34), so the accepted run's bins have the frozen identity |
| A frozen Stage 2 record changes version | No: the 3A record finds the contract, protocol, profile and pathway identities equal to the accepted run's; `inner_feature_isolation_v1.json` unchanged |
| A Stage 4 reader no longer accepts the accepted artifacts unchanged | No: the Stage 4 adapters read the accepted GLASS Router bundle and RF Router unchanged; Stage 1 readers accept version 2.0 (2B); GLASS Arbiter readers accept `meta_ebm/2` (3B) |

**Outcome:** the reuse path of §5.2. Commit 4A must still show, on the final
code: `acceptance.require_accepted("c3_20261112_153333")`, `report --check`,
the V7 Stage 4 interface checks for both upstreams (train and test side), and
the artifacts' SHA-256 equal to `primary_MANIFEST.json`. Any failure means
regeneration under §5.2.

## Vehicle (fixed, §5.4)

Commit 4A: one fresh-kernel `nbconvert --execute` of each notebook, Stage 4
cells excluded, Stage 2 loaded by pinned path and SHA-256 (§5.3). Commit 5A:
Stage 4 from the PR 39 artifact manifest in a fresh kernel. Commit 6B: one full
reproduction per notebook.

## For Commit 4: producers and consumers that change or need pinning

- **Stage 1:** GLASS Cell 8 and black-box Cells 7–14 write LR / MLP version 3.0
  (contract 2.0.0). Black-box Cell 14A reads the newest `lr_calibrated_*`
  (F9: pin it); `shared.stage_io.assert_comparable` refuses an old-name /
  new-name pair.
- **Stage 4:** GLASS Cell 19 writes `models/glass_arbiter/glass_arbiter_*`
  (`glass_arbiter/1`); black-box Cell 35 finds it with
  `find_glass_arbiter_artifact` (F9: pin it); Cell 36 passes
  `glass_arbiter_path=`; Cell 37 writes `stage4_crossover/glass_arbiter_on_blackbox/`.
- **Other F9 reads:** black-box Cells 18 (`glass_router_*`) and 33 (`ebm_stage3_*`).
- **Audit:** a PR 39 `$BANK_ARTIFACT_MANIFEST` uses the `artifact_audit.ROOTS`
  keys; the PR 38 manifest and `phase0.ARBITER_KEYS` keep the historical keys.
  Without a manifest, the audit selects the PR 39 folders once any PR 39 GLASS
  Arbiter file exists.
- **Baseline-pinned code:** `stage2_baseline.py --check` reports the two
  deliberate changes recorded in `DELIBERATE_PIN_CHANGES`
  (`shared/stage2/diagnostics/boundary.py`, 2B; `tests/support/artifact_audit.py`,
  3B). The G10 pin change in Commit 4A edits `artifact_audit.py` again and
  updates its entry.
