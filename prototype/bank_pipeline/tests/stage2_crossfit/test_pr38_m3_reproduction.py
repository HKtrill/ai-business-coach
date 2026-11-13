"""
PR 38 M3: an unproven GLASS ILP incumbent is documented only by an identical
rerun (``runner.document_unproven``; ``evidence.selection_records`` applies the
frozen ``glass_record_reproduction``).

A scripted GLASS whose Pass 2 is a time-limited incumbent with a CBC log gap:

1. G3 (``fit_fold`` + ``preflight_checks``): an identical rerun documents it
   (``valid_unproven_documented``, accepted, G3 passes); the original report and
   its solver diagnostics are untouched; the status survives a JSON round trip.
2. A rerun that selects different rules, a rerun that raises, or a missing
   gap leaves ``valid_unproven_undocumented``: G3 fails.
3. Proven selections are never rerun.
4. The cross-fit and the full-train fit document the same way; a
   non-reproducing fold stops the cross-fit; the G7 reproduction carries the
   same M3 record, so a documented candidate still reproduces.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import itertools
import json

import pytest

import shared.stage2.evaluation_protocol as ep
import stage2_crossfit.evidence as E
import stage2_crossfit.runner as R
from tests.support import stage2_crossfit_world as W
from tests.support.rf_router_world import serial_forests

GAP = {"status": "Optimal", "sol_status": 2, "state": "feasible_unproven",
       "solver_log": {"bound": 81.0, "gap_abs": 21.0, "gap_rel": 0.35, "objective": 60.0,
                      "result": "Stopped on time limit", "source": "cbc_log"}}


def _time_limited_class(*, reproducible=True, gap=True):
    """Pass 2 ends as a time-limited incumbent; ``reproducible=False``: each fit picks other ids."""
    base, calls = W.wired_glass_class(), itertools.count()

    class Sel(W._Sel):
        def _pass(self, scope, X, y):
            rule = super()._pass(scope, X, y)
            if scope != "pass2":
                return rule
            if not reproducible:
                rule = dataclasses.replace(rule, rule_id=10 + next(calls))
            rep = self.selection_report_[W.P2]
            rep.update(lexicographic_guarantee="UNPROVEN", selected_rule_ids=[rule.rule_id])
            if gap:
                rep["phase1"] = json.loads(json.dumps(GAP))
            return rule

    class TimeLimited(base):
        def _new_selector(self):
            return Sel(self)
    return TimeLimited


@pytest.fixture(scope="module")
def ts():
    return W.synthetic_train_side()


@contextlib.contextmanager
def _wired(cls):
    mp = pytest.MonkeyPatch()
    W.wired(mp, cls)
    try:
        with serial_forests(), contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        mp.undo()


def _g3(ts, cls):
    settings = W.make_settings()
    with _wired(cls):
        res = R.fit_fold(ts, settings, 0, arms=(E.ARM_GLASS,))
    return res["arms"][E.ARM_GLASS], R.preflight_checks(res, protocol=settings.protocol), settings


def _pass2(ev, settings):
    return next(r.outcome for r in E.selection_records(ev, settings.contract) if r.key == "pass2")


def test_an_identical_rerun_documents_the_incumbent_and_g3_passes(ts):
    ev, g3, settings = _g3(ts, _time_limited_class())
    assert ev["fit_status"]["by_key"]["pass2"] == ep.FIT_UNPROVEN_DOCUMENTED
    assert ev["fit_status"]["accepted"] and g3["arms"]["glass"]["passed"] and g3["passed"]
    rep = ev["selection_reports"]["pass2"]                       # the original, untouched
    assert rep["lexicographic_guarantee"] == "UNPROVEN" and rep["phase1"] == GAP
    m3 = ev[E.M3_KEY]
    assert m3["error"] is None and list(m3["selections"]) == ["pass2"]
    assert set(m3["selections"]["pass2"]["rerun"]) == set(E.M3_FIELDS)
    o = _pass2(ev, settings)
    assert o.solver["reproduced"] is True and o.solver["gap"] == 0.35
    assert o.solver["gap_source"] == "cbc_log" and o.documentation_complete
    disk = json.loads(json.dumps(ev))                            # as the G3 record is read back
    assert E.fold_status(disk, settings.contract, settings.protocol) == ev["fit_status"]


@pytest.mark.parametrize("cls, reproduced", [
    (lambda: _time_limited_class(reproducible=False), False),
    (lambda: _time_limited_class(gap=False), True),
])
def test_without_reproduction_or_gap_the_status_stays_rejected(ts, cls, reproduced):
    ev, g3, settings = _g3(ts, cls())
    assert ev["fit_status"]["by_key"]["pass2"] == ep.FIT_UNPROVEN_UNDOCUMENTED
    assert not ev["fit_status"]["accepted"] and not g3["arms"]["glass"]["passed"]
    assert g3["arms"]["glass"]["feasible"]
    o = _pass2(ev, settings)
    assert o.solver["reproduced"] is reproduced and not o.documentation_complete


def test_a_rerun_that_raises_is_recorded_and_documents_nothing(ts):
    ev, _, settings = _g3(ts, _time_limited_class())
    ev = {k: v for k, v in ev.items() if k not in (E.M3_KEY, "fit_status")}

    def refit():
        raise RuntimeError("solver unavailable")
    R.document_unproven(ev, refit)
    assert ev[E.M3_KEY]["error"] == "RuntimeError: solver unavailable"
    assert ev[E.M3_KEY]["selections"] == {"pass2": {"rerun": None}}
    status = E.fold_status(ev, settings.contract, settings.protocol)
    assert status["by_key"]["pass2"] == ep.FIT_UNPROVEN_UNDOCUMENTED and not status["accepted"]


def test_proven_selections_are_never_rerun(ts):
    ev, g3, _ = _g3(ts, W.wired_glass_class())
    assert E.M3_KEY not in ev and g3["arms"]["glass"]["passed"]
    assert E.unproven_selections(ev) == []
    calls = []
    R.document_unproven(dict(ev), lambda: calls.append(1))
    assert calls == []


def test_the_cross_fit_and_the_full_train_fit_document_alike(ts):
    settings = W.make_settings()
    with _wired(_time_limited_class()):
        result = R.run_crossfit(ts, settings, arms=(E.ARM_GLASS,))
        R.fit_full_train(ts, settings, result, arms=(E.ARM_GLASS,))
    evs = [*result.evidence[E.ARM_GLASS], result.full_train_evidence[E.ARM_GLASS]]
    assert len(evs) == 6
    assert all(ev["fit_status"]["by_key"]["pass2"] == ep.FIT_UNPROVEN_DOCUMENTED
               and ev["fit_status"]["accepted"] and E.M3_KEY in ev for ev in evs)
    with _wired(_time_limited_class(reproducible=False)):
        result = R.RunResult(settings.role)
        with pytest.raises(R.FoldNotAccepted, match="glass fold 0"):
            R.run_crossfit(ts, settings, arms=(E.ARM_GLASS,), result=result)
    ev = result.evidence[E.ARM_GLASS][0]
    assert ev["fit_status"]["status"] == ep.FIT_UNPROVEN_UNDOCUMENTED


def test_g7_reproduces_a_documented_candidate(tmp_path, monkeypatch):
    from tests.stage2_crossfit.test_pr38_acceptance import _build
    cls = _time_limited_class()
    monkeypatch.setattr(W, "wired_glass_class", lambda script=None: cls)
    frozen_ids = R.primary_settings().checks["identities"]
    settings = dataclasses.replace(W.make_settings(role=R.ROLE_PRIMARY),
                                   checks={"identities": frozen_ids})
    w = _build(tmp_path, settings, "rm3", R.ROLE_PRIMARY)
    fold0 = w["cand"].fold_evidence(E.ARM_GLASS, 0)
    assert fold0["fit_status"]["status"] == ep.FIT_UNPROVEN_DOCUMENTED and E.M3_KEY in fold0
    glass = w["g7"]["arms"]["glass"]
    assert glass["evidence"]["identical_within_tolerance"] and glass["passed"]
