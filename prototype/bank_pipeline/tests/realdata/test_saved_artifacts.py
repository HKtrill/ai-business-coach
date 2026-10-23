"""
Real-artifact audit (PR 36 Commit 4, P5): the artifacts the two cascade
notebooks saved, read back and checked — run after regeneration (P6) with

    pytest -m realdata

and nothing else selects these tests (tests/conftest.py). Enabled, a missing
database, a missing arbiter or a missing upstream file FAILS with its path;
nothing is skipped, trained, refitted, backfilled or deleted.

Roots are the four Stage 4 arbiters of the 2x2 — the newest in each producer
folder, or the files named in ``$BANK_ARTIFACT_MANIFEST`` — and every Stage
1-3 file is the one an arbiter recorded consuming (tests.support.artifact_audit
says where each is saved and what each check reuses).

    upstream     Meta-EBM                Meta-XGB
    GLASS        meta_ebm (native)       meta_xgb_on_glass
    black-box    meta_ebm_on_blackbox    meta_xgb (native)
"""

from __future__ import annotations

import pytest

from tests.support import artifact_audit as audit
from tests.support.notebooks import BANK_ROOT

pytestmark = pytest.mark.realdata

KEYS = tuple(audit.ROOTS)
NATIVE_KEY = {"glass": "meta_ebm", "blackbox": "meta_xgb"}


class Saved:
    """Each step once per session; a step's failure is re-raised by every test that needs it."""

    def __init__(self):
        self._done: dict = {}

    def _once(self, name, step):
        if name not in self._done:
            try:
                self._done[name] = (True, step())
            except audit.AuditError as exc:
                self._done[name] = (False, str(exc))
        ok, value = self._done[name]
        if not ok:
            raise audit.AuditError(value)
        return value

    def split(self):
        return self._once("split", audit.real_global_split)

    def roots(self):
        return self._once("roots", lambda: audit.select_roots(BANK_ROOT, audit.manifest_from_env()))

    def arbiter(self, key):
        return self._once(("arbiter", key),
                          lambda: audit.load_arbiter(key, self.roots()[key], self.split()))

    def upstream(self, key):
        """(Stage4Inputs, {stage: path})."""
        return self._once(("upstream", key),
                          lambda: audit.reload_upstream(key, self.arbiter(key), self.split()))


@pytest.fixture(scope="module")
def saved():
    return Saved()


def test_the_four_arbiters_of_the_2x2_are_saved(saved):
    assert set(saved.roots()) == set(KEYS)


@pytest.mark.parametrize("key", KEYS)
def test_each_arbiter_is_its_declared_cell_with_complete_evidence(saved, key):
    audit.check_arbiter_identity(key, saved.arbiter(key))


@pytest.mark.parametrize("key", KEYS)
def test_each_arbiters_upstream_files_pass_the_contract_and_are_what_it_consumed(saved, key):
    inputs, paths = saved.upstream(key)
    assert set(paths) == set(audit.STAGES) and inputs.upstream["arm"] == audit.ROOTS[key].upstream


def test_every_audited_file_reloads_in_a_clean_interpreter_as_recorded(saved):
    expected = {}
    for key in KEYS:
        expected.update(audit.expected_reload(key, saved.roots()[key], saved.arbiter(key),
                                              saved.upstream(key)[1]))
    audit.check_fresh_reload(expected)


@pytest.mark.parametrize("upstream", sorted(NATIVE_KEY))
def test_each_stage3_model_serves_its_saved_test_probabilities(saved, upstream):
    from shared.stage3.feature_engineering import EBM_FEATURES     # both arms' feature contract
    path = saved.upstream(NATIVE_KEY[upstream])[1]["stage3"]
    audit.check_stage3_serving(path, saved.split(), upstream, expected_features=list(EBM_FEATURES))


@pytest.mark.parametrize("key", [k for k in KEYS if audit.ROOTS[k].arbiter == "meta_xgb"])
def test_each_meta_xgb_serves_its_saved_test_probabilities(saved, key):
    audit.check_meta_xgb_serving(key, saved.arbiter(key), saved.upstream(key)[0])


def test_the_2x2_holds_each_upstream_fixed_and_keeps_the_arms_apart(saved):
    arts = {key: saved.arbiter(key) for key in KEYS}
    for key in KEYS:
        saved.upstream(key)                     # the files each records are the ones on disk
    audit.check_crossover(arts, saved.roots(), saved.split())


def test_no_xgboost_artifact_a_notebook_could_pick_needs_removed_code():
    bad = audit.unloadable_xgb_artifacts()
    assert not bad, ("files in blackbox_pipeline/models/xgb that black-box Cell 31 could load "
                     "(XGB_PARAM_SOURCE='artifact' takes the newest):\n  "
                     + "\n  ".join(f"{p}: {why}" for p, why in bad))
