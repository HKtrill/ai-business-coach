"""
PR 37 Commit 2 (Step 2E): the Pre-Correction Stage 2 Baseline built from the
archived October 23 artifacts - opt-in, like every realdata test, and it needs
the preserved snapshot's SHA256SUMS (the artifacts are read from that snapshot,
never from the live models/ folders; tests.support.archived_artifacts):

    $env:PR37_BASELINE_SHA256SUMS = "...\\pr36_regen_2026-10-23\\SHA256SUMS.txt"
    pytest -m realdata tests/realdata/test_router_baseline_artifacts.py

Builds the package twice in memory (never writes) through the validated loader
and checks: byte-stable generation; every source attested against the snapshot;
the validated identities; the exported tables and the report reproduce the
validated Step 2A-2D anchors; the match_glass_oof provenance; and, once the
package has been generated into the repository, that the frozen files are
reproduced from the sources its MANIFEST.json records (the --check path).
Expectations only: the implementation never uses these numbers.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import baseline as b
from tests.support.notebooks import BANK_ROOT
from tests.support.archived_artifacts import (
    HistoricalReal,
    baseline_arbiters,
    collect_archived_baseline_sources,
)
from tests.support.router_artifacts import SUMS_ENV

pytestmark = pytest.mark.realdata

P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN
#: Identities (handoff / baseline PROVENANCE.json; Commit 1).
FINGERPRINTS = {"router_split": "fc3e73a5ef91aea4", "train_label": "b984d2a50527",
                "stage_io": "d6cddadc3439", "fold": "8aea082fcaf371b7"}
N_TRAIN, N_POSITIVES, ROWS_PER_FOLD, SNAPSHOT_ENTRIES = 32_950, 3_712, [6_590] * 5, 37
#: Step 2A.
STATES = {"glass": {P1: 11_489, P2: 8_468, AB: 12_993},
          "blackbox": {P1: 13_074, P2: 9_427, AB: 10_449}}
COVERAGE = {"glass": 19_957, "blackbox": 22_501}
#: Step 2B: union flags routed to Pass 1 (rows, actual positives).
PASS1_UNION = {"glass": (644, 82), "blackbox": (14, 5)}
REFERENCE_FLAGS = {"lr": 7_727, "ebm": 6_113, "union": 8_001, "intersection": 5_839}
EBM_NESTED_THRESHOLDS = (0.56, 0.56, 0.54, 0.49, 0.49)
#: Step 2C.
UNIQUE_VS_UNION = {"glass": (131, 1_862), "blackbox": (117, 1_669)}
COMMON = {"common_rows": 16_189, "common_positives": 2_931, "glass_pass2_outside_common": 74,
          "rf_pass2_outside_common": 813}
MATCHED_TP_LEAD = {("glass", "lr_matched"): -80, ("glass", "ebm_matched"): -116,
                   ("blackbox", "lr_matched"): 0, ("blackbox", "ebm_matched"): -28}
LR_ONLY_TP, EBM_ONLY_TP = 170, 56
#: Step 2D.
CROSS = {"cross_state_agreement": (21_829, 32_950), "cross_routed_agreement": (15_225, 16_112),
         "cross_one_abstains_rate": (10_234, 32_950), "cross_opposite_rate": (887, 32_950),
         "cross_both_abstain_rate": (6_604, 32_950)}
G1_R2 = {"by_fold": (200, 92, 189, 82, 250), "n": 813, "positives": 94}
G2_R1 = {"by_fold": (74, 0, 0, 0, 0), "n": 74, "positives": 8}
CROSS_ROUTER_UNION = {"glass_unique_tp": 131, "rf_unique_tp": 117, "shared_tp": 74,
                      "glass_only_tp": 57, "rf_only_tp": 43, "shared_fp": 1_002,
                      "glass_only_fp": 860, "rf_only_fp": 667}
#: match_glass_oof: the recorded RF constraints equal these GLASS pooled OOF values.
CONSTRAINTS = {"min_precision_not_subscribe": (10_976, 11_489),
               "max_subscriber_leakage_rate": (513, 3_712),
               "max_subscriber_leakage_absolute": (513, None),
               "min_precision_subscribe": (2_337, 8_468)}


@pytest.fixture(scope="module")
def run():
    sums = os.environ.get(SUMS_ENV)
    assert sums, (f"set {SUMS_ENV} to the preserved snapshot's SHA256SUMS.txt: the baseline is "
                  "built only from attested sources")
    real = HistoricalReal(sums_path=sums)
    sources = collect_archived_baseline_sources(real)
    code = b.code_identity(BANK_ROOT)
    first, second = b.build_baseline(sources, code), b.build_baseline(sources, code)
    return {"sources": sources, "code": code, "pkg": first, "again": second, "sums": sums,
            "archive": real.archive.root}


def _csv(pkg, name) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(pkg.files[b.TABLES[name][0]]), dtype=str,
                       keep_default_na=False)


def _ratio(pkg, section, metric, *, fold="all", **keys):
    xs = _csv(pkg, "ratios")
    m = (xs["section"] == section) & (xs["metric"] == metric) & (xs["fold"] == str(fold))
    for k in ("arm", "reference", "population", "group", "score"):
        m &= xs[k] == keys.get(k, "")
    row = xs[m]
    assert len(row) == 1, (section, metric, keys)
    r = row.iloc[0]
    return int(r["numerator"]), (None if r["denominator"] == "" else int(r["denominator"]))


def test_generation_is_byte_stable(run):
    assert dict(run["pkg"].files) == dict(run["again"].files)
    m = run["pkg"].manifest
    assert (m["baseline_id"], m["designation"], m["label"]) == (
        b.BASELINE_ID, "pre_correction", "Pre-Correction Stage 2 Baseline")
    assert all(c["passed"] for c in m["validation"]["consistency"])
    assert m["validation"]["trace_checks"]["passed"] == m["validation"]["trace_checks"]["total"]
    assert m["validation"]["rf_pass1_score_equivalence"] == "verified_from_artifacts"


def test_sources_are_the_attested_artifacts(run):
    m = run["pkg"].manifest
    by = {s["role"]: s for s in m["sources"]}
    assert list(by) == list(b.SOURCE_ROLES)
    for role in (b.ROLE_LR, b.ROLE_GLASS, b.ROLE_EBM, b.ROLE_RF, b.ROLE_RF_RESULT):
        assert by[role]["snapshot_attested"] is True, role
        assert by[role]["fields_read"] and by[role]["extractor_checks"]["passed"] == \
            by[role]["extractor_checks"]["total"]
    for role in (b.ROLE_ARBITER_GLASS, b.ROLE_ARBITER_RF):
        assert by[role]["snapshot_attested"] in (True, None), role
    for s in m["sources"]:
        assert not Path(s["path"]).is_absolute() and (run["archive"] / s["path"]).is_file()
        for f in s["fields_read"] or ():
            assert not any(t in f.lower() for t in d.FORBIDDEN_FIELD_TOKENS), f
    assert m["snapshot"]["entries"] == SNAPSHOT_ENTRIES
    assert m["snapshot"]["sha256"] == d.file_sha256(run["sums"])


def test_identities_and_population(run):
    m = run["pkg"].manifest
    ids = m["identities"]
    assert ids["router_split_fingerprint"]["value"] == FINGERPRINTS["router_split"]
    assert ids["train_label_fingerprint"]["value"] == FINGERPRINTS["train_label"]
    assert (ids["stage_io_split_fingerprint"]["value"],
            ids["stage_io_split_fingerprint"]["status"]) == (FINGERPRINTS["stage_io"], "attested")
    assert ids["fold_plan"]["fingerprint"] == FINGERPRINTS["fold"]
    assert (m["population"]["training_rows"], m["population"]["positives"],
            m["population"]["rows_per_fold"]) == (N_TRAIN, N_POSITIVES, ROWS_PER_FOLD)
    up = m["upstream_operating_points"]
    assert tuple(up["ebm"]["fold_thresholds"].values()) == EBM_NESTED_THRESHOLDS
    assert {k: v["flags"] for k, v in up.items()} == REFERENCE_FLAGS


def test_step_2a_states_and_coverage(run):
    st = _csv(run["pkg"], "states")
    st = st[st["fold"] == "all"]
    for arm, counts in STATES.items():
        got = {r["state"]: int(r["n"]) for _, r in st[st["arm"] == arm].iterrows()}
        assert got == counts, arm
        assert _ratio(run["pkg"], "2A.routing_rates", "coverage", arm=arm) == (COVERAGE[arm],
                                                                             N_TRAIN)


def test_step_2b_and_2c_anchors(run):
    pkg = run["pkg"]
    for arm, (n, pos) in PASS1_UNION.items():
        assert _ratio(pkg, "2B.pass1_suppression", "intercepted_flags", arm=arm,
                      reference="union")[0] == n
        assert _ratio(pkg, "2B.pass1_suppression", "suppressed_tp", arm=arm,
                      reference="union")[0] == pos
    for arm, (tp, fp) in UNIQUE_VS_UNION.items():
        for metric, want in (("unique_tp", tp), ("unique_fp", fp)):
            assert _ratio(pkg, "2C.pass2_frozen", metric, arm=arm, reference="union",
                          population="all_rows")[0] == want
    c = _csv(pkg, "common_population")
    c = c[c["fold"] == "all"].iloc[0]
    assert {k: int(c[k]) for k in COMMON} == COMMON
    for (arm, ref), lead in MATCHED_TP_LEAD.items():
        tp = _ratio(pkg, "2C.pass2_matched", "pass2_tp", arm=arm, reference=ref,
                    population="common_eligible")[0]
        rt = _ratio(pkg, "2C.pass2_matched", "reference_tp", arm=arm, reference=ref,
                    population="common_eligible")[0]
        assert tp - rt == lead, (arm, ref)
    md = _csv(pkg, "model_disagreement")
    fr = md[(md["setting"] == "frozen") & (md["fold"] == "all")].iloc[0]
    assert (int(fr["lr_only_tp"]), int(fr["ebm_only_tp"])) == (LR_ONLY_TP, EBM_ONLY_TP)


def test_step_2d_anchors(run):
    pkg = run["pkg"]
    for metric, want in CROSS.items():
        assert _ratio(pkg, "2D.crosstab_summary", metric) == want, metric
    xt = _csv(pkg, "crosstab")
    for cell, anchor in (((P1, P2), G1_R2), ((P2, P1), G2_R1)):
        sub = xt[(xt["glass_state"] == cell[0]) & (xt["rf_state"] == cell[1])]
        assert tuple(int(sub[sub["fold"] == str(k)]["n"].iloc[0]) for k in range(5)) == \
            anchor["by_fold"]
        pooled = sub[sub["fold"] == "all"].iloc[0]
        assert (int(pooled["n"]), int(pooled["positives"])) == (anchor["n"], anchor["positives"])
    p = _csv(pkg, "pass2_cross_router")
    p = p[(p["reference"] == "union") & (p["fold"] == "all")].iloc[0]
    assert {k: int(p[k]) for k in CROSS_ROUTER_UNION} == CROSS_ROUTER_UNION
    sig = _csv(pkg, "signal_availability").set_index(["arm", "signal"])
    assert sig.loc[("both", "analysis: GLASS Pass 1 vs Pass 2 candidate / signal overlap"),
                   "status"] == "not_verifiable"
    assert sig.loc[("blackbox", "Pass 2 outer score"), "status"] == "not_verifiable"


def test_match_glass_oof_constraints_equal_the_glass_realised_values(run):
    cp = run["pkg"].manifest["rf_constraint_provenance"]
    assert cp["constraint_source"] == "match_glass_oof"
    by = {c["constraint"]: c for c in cp["derivation_check"]}
    for key, (num, den) in CONSTRAINTS.items():
        c = by[key]
        assert (c["glass_numerator"], c["glass_denominator"], c["status"]) == (num, den, "equal"), key


def test_report_carries_the_validated_step_2d_summary(run):
    text = run["pkg"].files[b.REPORT].decode("utf-8")
    for needle in ("Coverage: GLASS 19,957/32,950 = 60.57%; RF 22,501/32,950 = 68.29%.",
                   "Same three-state decision: 21,829/32,950 = 66.25%.",
                   "Agreement when both route (routed only): 15,225/16,112 = 94.49%.",
                   "Exactly one router abstains: 10,234/32,950 = 31.06%.",
                   "Opposite routing decisions: 887/32,950",
                   "GLASS Pass 1 / RF Pass 2: 813 rows, 94 actual positives.",
                   "GLASS Pass 2 / RF Pass 1: 74 rows, 8 actual positives.",
                   "GLASS 644 rows (82 actual positives); RF 14 rows (5 actual positives).",
                   "Pass 2 positives the union misses: GLASS 131, RF 117; of these 74 are found "
                   "by both routers, 57 by GLASS only and 43 by RF only",
                   "**pre_correction**", "remain unproven"):
        assert needle in text, needle


def test_the_frozen_package_is_reproduced_from_its_recorded_sources(run):
    target = Path(BANK_ROOT) / b.BASELINES_DIR / b.BASELINE_ID
    if not (target / b.MANIFEST).is_file():
        pytest.skip("the baseline has not been generated yet (stage2_baseline.py --generate)")
    assert b.verify_baseline(target).ok
    frozen = json.loads((target / b.MANIFEST).read_bytes())
    pinned = HistoricalReal(sums_path=run["sums"], arbiters=baseline_arbiters(frozen))
    sources = collect_archived_baseline_sources(pinned)
    cmp = b.compare_baseline(b.build_baseline(sources, run["code"]), target)
    assert cmp.reproduced, cmp.lines()


def test_no_artifact_file_was_changed(run):
    for a in run["sources"].artifacts:
        assert d.file_sha256(run["archive"] / a.path) == a.sha256, a.role
