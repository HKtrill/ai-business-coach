"""
PR 39, Commit 3, Step 3A on the real training split: E-B and T1-T4.

The out-of-fold NSD / ECI bin counts on GLOBAL_SPLIT's training rows (shared
outer folds ``8aea082fcaf371b7``), the in-sample counts beside them, and the
outcome: no trigger fires for either family, so both are retained. E-B2 needs
the accepted PR 38 artifacts and is covered by ``python -m stage2_binning``.

Marked ``realdata`` (it reads ``data/db``), so it runs only under
``pytest -m realdata``. Test labels are never read.
"""

from __future__ import annotations

import pytest

from shared.stage2.binning import binning_identity
from stage2_binning import evidence as E
from stage2_binning import runner as R
from tests.stage2_binning.test_binning_evidence import FROZEN_BINNING_SHA256

pytestmark = pytest.mark.realdata

#: (support, positives) pooled over the five held-out folds, and in-sample.
OOF = {"nsd_cold": (20364, 1001), "nsd_warm": (8221, 926), "nsd_elevated": (1421, 398),
       "nsd_hot": (2944, 1387), "eci_cold": (8606, 503), "eci_mid": (8721, 720),
       "eci_warm": (11549, 1341), "eci_hot": (4074, 1148)}
IN_SAMPLE = {"nsd_cold": (22120, 1090), "nsd_warm": (6839, 863), "nsd_elevated": (1326, 467),
             "nsd_hot": (2665, 1292), "eci_cold": (7392, 433), "eci_mid": (9078, 714),
             "eci_warm": (12491, 1461), "eci_hot": (3989, 1104)}
FIT_LABEL_FINGERPRINTS = ["85c06667c5ec", "530be40451f7", "24e4da6a264c", "add85f9ee20c",
                          "3e17d1f98a63"]


@pytest.fixture(scope="module")
def real():
    from stage2_crossfit.runner import load_train_side
    ts = load_train_side()
    assert ts.fold_fingerprint == "8aea082fcaf371b7"
    return ts, R.eb_evidence(ts)


def test_out_of_fold_and_in_sample_counts(real):
    _, eb = real
    for fam in E.FAMILIES:
        for kind, want in (("oof", OOF), ("in_sample", IN_SAMPLE)):
            got = {b: (v["pooled"]["support"], v["pooled"]["positives"])
                   for b, v in eb["families"][fam][kind]["bins"].items()}
            assert got == {b: want[b] for b in E.family_bins(fam)}
    assert [f["fit_label_fingerprint"] for f in eb["fold_fits"]] == FIT_LABEL_FINGERPRINTS
    assert all(f["heldout_rows_in_fit"] == 0 for f in eb["fold_fits"])


def test_the_outer_fold_fits_are_the_accepted_runs(real):
    """Fold 0's engineer is fitted on exactly the rows the accepted run's fold 0 fitted on."""
    import json

    from stage2_crossfit import phase0
    _, eb = real
    g3 = json.loads((phase0.RECORD_ROOT / R.PR38_RUN_ID / "g3_preflight_fold0.json")
                    .read_text(encoding="utf-8"))
    for arm in ("glass", "blackbox"):
        assert g3["arms"][arm]["train_label_fingerprint"] == eb["fold_fits"][0]["fit_label_fingerprint"]
        assert g3["arms"][arm]["n_train"] == eb["fold_fits"][0]["fit_rows"]


def test_no_trigger_fires_and_both_families_are_retained(real):
    _, eb = real
    trig = {fam: E.triggers(fam, eb["families"][fam]["oof"]) for fam in E.FAMILIES}
    assert {fam: [k for k in ("T1", "T2", "T3", "T4") if t[k]["fired"]]
            for fam, t in trig.items()} == {"NSD": [], "ECI": []}
    assert {fam: d["outcome"] for fam, d in E.decide(trig).items()} == \
        {"NSD": "retain", "ECI": "retain"}
    assert binning_identity()["sha256"] == FROZEN_BINNING_SHA256
