"""
Arm adapters for the shared Stage 2 contract (PR 37 Commit 4, Step 4B).

Contract
1. The GLASS adapter maps every schema-2 obligation onto the selected-union
   fields, from the contract alone, with the lexicographic "rows" Pass 2
   objective and the contract's provenance. It requires the F17 primary
   settings and refuses, never overwrites, a conflicting base value. A config
   that claims shared compliance through per-rule gates alone is refused.
2. The RF adapter keeps the arm's own settings (hyperparameters included)
   and replaces only its constraints, which come from the contract; it
   requires the F17 primary settings. Both arms carry identical obligations.
3. Isolation: every selector the GLASS pipeline builds, the inner-fold Pass 1
   selectors included, carries the obligations; Pass 2 is selected on exactly
   the OOF remainder; inner selections never see their held-out rows.
4. Fit outcomes follow the Commit 4 taxonomy: proven ILP -> primary; an
   unproven incumbent is primary only with its gap and a recorded
   deterministic reproduction; greedy -> review; infeasible, no solution and
   empty unions stop. A feasible claim the independent check contradicts is
   refused.
5. Historical defaults are unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from blackbox_pipeline.models.rf_router import OperatingConstraints
from blackbox_pipeline.models.rf_router.config import RFRouterConfig
from blackbox_pipeline.models.rf_router.contract import (
    RF_F17_PRIMARY, check_rf_config_follows_contract, rf_config_from_contract,
    rf_constraints_from_contract_v2,
)
from glass_pipeline.glass_router.contract import (
    GLASS_F17_PRIMARY, UNION_OBLIGATION_FIELDS, check_glass_config_follows_contract,
    glass_config_from_contract, glass_fit_outcome, glass_pass_report, glass_record_reproduction,
    glass_union_counts, glass_union_obligations_from_contract,
)
from glass_pipeline.glass_router.core.config import GlassRouterConfig
from glass_pipeline.glass_router.ilp_rule_selector.ilp_rule_selector import (
    ILPRuleSelector, SelectorInfeasibleError,
)
from shared.stage2.contract import (
    ContractViolation, SharedOperatingContract, constraint_provenance, load_operating_contract,
    working_profile_path,
)
from shared.stage2.obligations import (
    DISPOSITION_PRIMARY, DISPOSITION_PRIMARY_IF_DOCUMENTED, DISPOSITION_REQUIRES_REVIEW, DISPOSITION_STOP,
    OUTCOME_FEASIBLE_GREEDY, OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN,
    OUTCOME_INFEASIBLE, OUTCOME_INSUFFICIENT, OUTCOME_NO_SOLUTION, PassCounts, check_pass1_fit,
)
from tests.stage2.conftest import legacy_config, new_router, selected_rule
from tests.stage2.test_ilp_rule_selector_union import (
    P1, P2, FakeRule, make_selector, no_solution, real_but_time_limited, rng, script_solver,
)
from tests.stage2.test_stage2_operating_contract_v2 import _v2
from tests.support.synthetic import irregular_index

BANK_ROOT = Path(__file__).resolve().parents[2]
BASELINE = BANK_ROOT / "research_logs" / "baselines" / "pr37_stage2_pre_correction_v1"


def contract():
    return load_operating_contract(working_profile_path())


def f17_base(**overrides) -> GlassRouterConfig:
    """The legacy profile with the F17 primary settings and no union obligation."""
    return legacy_config(**{**dict(GLASS_F17_PRIMARY), "max_union_leakage_rate_pass1": None,
                            **overrides})


def rf_base(**overrides) -> RFRouterConfig:
    manifest = json.loads((BASELINE / "MANIFEST.json").read_text(encoding="utf-8"))
    historical = manifest["rf_constraint_provenance"]["recorded_constraints"]
    kwargs = dict(mode="two_pass", constraints=OperatingConstraints.from_dict(historical),
                  eligibility="nested_t1", eligibility_folds=5, n_oof_folds=5, random_state=42,
                  remainder_min_size=500, rf_params={"n_estimators": 650, "max_depth": 6,
                                                     "min_samples_leaf": 12,
                                                     "max_features": 0.39989815740885837,
                                                     "minority_weight": 1.3001084911026792,
                                                     "class_weight": "balanced"},
                  verbose=False)
    kwargs.update(overrides)
    return RFRouterConfig(**kwargs)


def contract_selector(c=None, **overrides) -> ILPRuleSelector:
    fields = glass_union_obligations_from_contract(c or contract())
    return make_selector(**{**fields, **overrides})


# ============================================================================
# 1. the GLASS adapter
# ============================================================================
def test_glass_union_obligations_come_from_the_contract_alone():
    c = contract()
    assert glass_union_obligations_from_contract(c) == {
        "max_union_leakage_rate_pass1": 0.15, "min_union_npv_pass1": 0.95,
        "min_union_support_pass1": 125, "min_union_precision_pass2": 0.30,
        "min_union_support_pass2": 75, "pass2_union_objective": "rows",
    }
    assert tuple(glass_union_obligations_from_contract(c)) == UNION_OBLIGATION_FIELDS


def test_the_glass_adapter_refuses_a_count_cap_and_a_schema_1_contract():
    with pytest.raises(ContractViolation, match="no union leakage-count constraint"):
        glass_union_obligations_from_contract(_v2(max_pass1_leakage_count=500))
    v1 = SharedOperatingContract(contract_id="synthetic", min_pass1_npv=0.9,
                                 max_pass1_leakage_rate=0.2, max_pass1_leakage_count=None,
                                 min_pass2_precision=0.4, provenance="unit-test fixture")
    with pytest.raises(ContractViolation, match="schema-2 contract"):
        glass_union_obligations_from_contract(v1)


def test_glass_config_from_contract_sets_obligations_and_provenance_only():
    c, base = contract(), f17_base()
    cfg = glass_config_from_contract(base, c)
    for name, value in glass_union_obligations_from_contract(c).items():
        assert getattr(cfg, name) == value
    assert cfg.contract_provenance == constraint_provenance(c)
    changed = {f.name for f in dataclasses.fields(GlassRouterConfig)
               if getattr(cfg, f.name) != getattr(base, f.name)}
    assert changed == set(UNION_OBLIGATION_FIELDS) | {"contract_provenance"}
    assert base.contract_provenance is None and base.min_union_npv_pass1 is None   # untouched
    check_glass_config_follows_contract(cfg, c)


def test_glass_config_from_contract_requires_the_f17_primary_settings():
    with pytest.raises(ContractViolation) as err:
        glass_config_from_contract(legacy_config(max_union_leakage_rate_pass1=None), contract())
    for name in ("pass2_population", "inner_feature_ranker"):
        assert name in str(err.value)
    for name, value in (("remainder_min_size", 400), ("pass1_oof_folds", 3), ("random_state", 7)):
        with pytest.raises(ContractViolation, match=f"{name}={value}"):
            glass_config_from_contract(f17_base(**{name: value}), contract())


def test_a_conflicting_base_value_is_refused_not_overwritten():
    with pytest.raises(ContractViolation, match="max_union_leakage_rate_pass1=0.14"):
        glass_config_from_contract(f17_base(max_union_leakage_rate_pass1=0.14), contract())
    with pytest.raises(ContractViolation, match="min_union_support_pass2=60"):
        glass_config_from_contract(f17_base(min_union_support_pass2=60), contract())
    same = glass_config_from_contract(f17_base(max_union_leakage_rate_pass1=0.15), contract())
    assert same.max_union_leakage_rate_pass1 == 0.15
    built = glass_config_from_contract(f17_base(), contract())
    with pytest.raises(ContractViolation, match="already records a contract_provenance"):
        glass_config_from_contract(built, contract())


def test_per_rule_gates_alone_cannot_claim_shared_compliance():
    c = contract()
    with pytest.raises(ContractViolation, match="contract_provenance is missing"):
        check_glass_config_follows_contract(
            dataclasses.replace(f17_base(), **glass_union_obligations_from_contract(c)), c)
    with pytest.raises(ValueError, match="per-rule gates alone"):
        f17_base(contract_provenance=constraint_provenance(c))
    with pytest.raises(ValueError, match="constraint_source 'shared_contract'"):
        dataclasses.replace(glass_config_from_contract(f17_base(), c),
                            contract_provenance={"constraint_source": "match_glass_oof"})


def test_check_glass_config_refuses_drift_and_foreign_provenance():
    c = contract()
    cfg = glass_config_from_contract(f17_base(), c)
    with pytest.raises(ContractViolation, match="min_union_precision_pass2=0.25"):
        check_glass_config_follows_contract(
            dataclasses.replace(cfg, min_union_precision_pass2=0.25), c)
    with pytest.raises(ContractViolation, match="pass1_oof_folds"):
        check_glass_config_follows_contract(dataclasses.replace(cfg, pass1_oof_folds=3), c)
    other = _v2(min_pass2_precision=0.35)
    with pytest.raises(ContractViolation):
        check_glass_config_follows_contract(
            dataclasses.replace(cfg, min_union_precision_pass2=0.35), other)   # foreign fingerprint


def test_glass_config_validation_of_the_new_fields():
    with pytest.raises(ValueError, match="min_union_npv_pass1"):
        f17_base(min_union_npv_pass1=1.2)
    with pytest.raises(ValueError, match="needs min_union_precision_pass2"):
        f17_base(pass2_union_objective="rows")
    with pytest.raises(ValueError, match="min_union_support_pass1"):
        f17_base(min_union_support_pass1=0)


# ============================================================================
# 2. the RF adapter and cross-arm equivalence
# ============================================================================
def test_rf_config_from_contract_replaces_only_the_constraints():
    c, base = contract(), rf_base()
    cfg = rf_config_from_contract(base, c)
    assert cfg.constraints == rf_constraints_from_contract_v2(c)
    a, b = base.to_dict(), cfg.to_dict()
    assert {k for k in a if a[k] != b[k]} == {"constraints"}
    assert cfg.rf_params == base.rf_params                       # the arm's own settings
    assert base.constraints != cfg.constraints                     # base untouched
    check_rf_config_follows_contract(cfg, c)
    check_rf_config_follows_contract(cfg, c, extra=constraint_provenance(c))


def test_rf_config_from_contract_requires_the_f17_primary_settings():
    with pytest.raises(ContractViolation, match="eligibility='global_t1'"):
        rf_config_from_contract(rf_base(eligibility="global_t1"), contract())
    with pytest.raises(ContractViolation, match="n_oof_folds=10"):
        rf_config_from_contract(rf_base(n_oof_folds=10), contract())
    with pytest.raises(ContractViolation, match="expected an RFRouterConfig"):
        rf_config_from_contract(f17_base(), contract())
    assert dict(RF_F17_PRIMARY)["eligibility"] == "nested_t1"


def test_check_rf_config_refuses_historical_or_drifted_constraints():
    c = contract()
    with pytest.raises(ContractViolation, match="do not follow the shared contract"):
        check_rf_config_follows_contract(rf_base(), c)                # historical constraints
    cfg = rf_config_from_contract(rf_base(), c)
    drifted = dataclasses.replace(cfg.constraints, min_band_support_pass1=100)
    with pytest.raises(ContractViolation, match="min_band_support_pass1"):
        check_rf_config_follows_contract(dataclasses.replace(cfg, constraints=drifted), c)


def test_both_arms_carry_identical_obligations():
    c = contract()
    g = glass_config_from_contract(f17_base(), c)
    rf = rf_config_from_contract(rf_base(), c).constraints
    pairs = [
        (g.min_union_npv_pass1, rf.min_precision_not_subscribe, c.min_pass1_npv),
        (g.max_union_leakage_rate_pass1, rf.max_subscriber_leakage_rate, c.max_pass1_leakage_rate),
        (g.min_union_precision_pass2, rf.min_precision_subscribe, c.min_pass2_precision),
        (g.min_union_support_pass1, rf.min_band_support_pass1, c.min_pass1_support),
        (g.min_union_support_pass2, rf.min_band_support_pass2, c.min_pass2_support),
    ]
    for glass_value, rf_value, contract_value in pairs:
        assert glass_value == rf_value == contract_value
    assert g.contract_provenance == constraint_provenance(c)


# ============================================================================
# 3. pipeline wiring and isolation
# ============================================================================
def test_every_new_selector_carries_the_contract_obligations():
    c = contract()
    router = new_router(glass_config_from_contract(f17_base(), c))
    for sel in (router.ilp_selector, router._new_selector()):
        assert (sel.min_union_npv_pass1, sel.min_union_support_pass1,
                sel.min_union_precision_pass2, sel.min_union_support_pass2,
                sel.pass2_union_objective, sel.max_union_leakage_rate_pass1) == (
            0.95, 125, 0.30, 75, "rows", 0.15)
        assert sel.ilp_builder.record_solver_log is True
    hist = new_router(legacy_config())._new_selector()
    assert (hist.min_union_npv_pass1, hist.min_union_precision_pass2,
            hist.pass2_union_objective, hist.ilp_builder.record_solver_log) == (
        None, None, "positives", False)


N_ROWS, INNER = 2400, 5


class _Ranker:
    def __init__(self, X):
        self.rows = frozenset(X.index)
        self.feature_importances_ = np.linspace(1.0, 0.5, X.shape[1])
        self.random_state = 42


def _data(seed=0):
    rng_ = np.random.default_rng(seed)
    index = irregular_index(N_ROWS, seed=seed)
    y = pd.Series((rng_.random(N_ROWS) < 0.2).astype(int), index=index)
    X = pd.DataFrame({"bucket": rng_.integers(0, 4, N_ROWS), "flag": rng_.integers(0, 2, N_ROWS)},
                     index=index)
    return X, y


def _wired_router(fail_inner_fold=None):
    """Real config and real selectors; rule search, evaluation and solving are recorded stubs."""
    router = new_router(glass_config_from_contract(f17_base(), contract()))
    rec = SimpleNamespace(calls=[], selectors=[])
    router.feature_ranker_factory = lambda X, y: _Ranker(X)

    def wrap(sel):
        rec.selectors.append(sel)

        def select_rules(evaluated_rules, y_val, X_val, segment_builder, passes):
            pass1 = passes == ("pass1",)
            rec.calls.append(SimpleNamespace(selector=sel, rows=frozenset(X_val.index),
                                             positives=int(np.asarray(y_val).sum()),
                                             passes=passes))
            key = P1 if pass1 else P2
            if pass1 and fail_inner_fold is not None and len(rec.calls) == fail_inner_fold + 2:
                sel.selection_report_ = {key: {"source": "infeasible", "stage": "quality gates"}}
                raise SelectorInfeasibleError("inner fold infeasible", proven=True)
            sel.selection_report_ = {key: {"source": "ilp", "lexicographic_guarantee": "proven",
                                           "call": len(rec.calls)}}
            if pass1:
                return {"pass1_rules": [selected_rule(len(rec.calls), [("bucket", 3)], "pass1",
                                                      0.97)], "pass2_rules": []}
            return {"pass1_rules": [],
                    "pass2_rules": [selected_rule(99, [("flag", 1)], "pass2", 0.4)]}

        sel.select_rules = select_rules
        return sel

    real_new_selector = router._new_selector
    router._new_selector = lambda: wrap(real_new_selector())
    wrap(router.ilp_selector)

    class Generator:
        def __init__(self, recall_scale=1.0, rf_model=None):
            pass

        def generate_candidates(self, X, y):
            from glass_pipeline.glass_router.core.rule import NOT_SUBSCRIBE, SUBSCRIBE
            return [SimpleNamespace(predicted_class=NOT_SUBSCRIBE),
                    SimpleNamespace(predicted_class=SUBSCRIBE)]

    class Evaluator:
        def evaluate_candidates(self, candidates, X, y):
            return [SimpleNamespace(recall=0.5, coverage=0.5) for _ in candidates]

    router.rule_generator, router.rule_evaluator = Generator(), Evaluator()
    router._new_generator = Generator
    router._new_evaluator = Evaluator
    return router, rec


@pytest.fixture(scope="module")
def wired():
    X, y = _data()
    router, rec = _wired_router()
    with contextlib.redirect_stdout(io.StringIO()):
        router.fit(X, y)
    return SimpleNamespace(X=X, y=y, router=router, rec=rec)


def test_every_selection_runs_under_the_contract(wired):
    rec = wired.rec
    assert len(rec.calls) == 1 + INNER + 1                       # full Pass 1, folds, Pass 2
    for call in rec.calls:
        sel = call.selector
        assert (sel.min_union_npv_pass1, sel.min_union_precision_pass2,
                sel.pass2_union_objective) == (0.95, 0.30, "rows")
    assert [c.passes for c in rec.calls] == [("pass1",)] * (1 + INNER) + [("pass2",)]
    assert len({id(c.selector) for c in rec.calls[1:]}) == INNER + 1   # a fresh selector each


def test_inner_selections_never_see_their_held_out_rows(wired):
    X, y, rec = wired.X, wired.y, wired.rec
    assert rec.calls[0].rows == frozenset(X.index)
    skf = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=42)
    for call, (tr, va) in zip(rec.calls[1:1 + INNER], skf.split(np.zeros(len(y)), y.to_numpy())):
        assert call.rows == frozenset(X.index[tr])
        assert not call.rows & frozenset(X.index[va])


def test_pass2_is_selected_on_exactly_the_oof_remainder(wired):
    X, y, rec, rep = wired.X, wired.y, wired.rec, wired.router.fit_report_
    remainder = frozenset(X.index[rep["remainder_mask"]])
    last = rec.calls[-1]
    assert last.rows == remainder and len(remainder) == rep["remainder_rows"] >= 500
    assert last.positives == rep["remainder_positives"] == int(y[list(remainder)].sum())
    routed = frozenset(X.index[X["bucket"].to_numpy() == 3])
    assert not remainder & routed


def test_selection_reports_cover_every_pass_and_inner_fold(wired):
    reports = wired.router.selection_reports()
    assert reports["pass1"]["call"] == 1
    assert [r["call"] for r in reports["inner_pass1"]] == list(range(2, 2 + INNER))
    assert reports["pass2"]["call"] == 2 + INNER
    reports["pass1"]["call"] = -1                                 # copies, not live state
    assert wired.router.selection_reports()["pass1"]["call"] == 1
    assert "selection_reports" not in wired.router.fit_report_   # fit_report_ unchanged


def test_an_infeasible_inner_fold_stops_the_fit_and_stays_visible():
    X, y = _data()
    router, rec = _wired_router(fail_inner_fold=2)
    with pytest.raises(SelectorInfeasibleError, match="inner fold infeasible"), \
            contextlib.redirect_stdout(io.StringIO()):
        router.fit(X, y)
    assert not router.is_fitted and router.pass2_rules == []
    reports = router.selection_reports()
    assert len(reports["inner_pass1"]) == 3
    assert reports["inner_pass1"][-1] == {"source": "infeasible", "stage": "quality gates"}
    assert reports["pass2"] is None


def test_full_population_reports_come_from_the_main_selector():
    router = new_router(legacy_config())
    router.ilp_selector.selection_report_ = {P1: {"source": "ilp"}, P2: {"source": "greedy_fallback"}}
    assert router.selection_reports() == {"pass1": {"source": "ilp"},
                                          "pass2": {"source": "greedy_fallback"},
                                          "inner_pass1": []}


# ============================================================================
# 4. fit outcomes
# ============================================================================
def _pass1_world():
    """1,000 rows, 100 positives: A routes 300 rows with 5 positives."""
    n = 1000
    y = np.zeros(n, int)
    y[0:5] = 1
    y[900:995] = 1
    A = FakeRule("A", rng(0, 299), n, precision=0.98)
    B = FakeRule("B", rng(300, 349), n, precision=0.97)
    return n, y, [A, B]


def _run_contract_pass1(sel, rules, y):
    with contextlib.redirect_stdout(io.StringIO()):
        return sel._optimize_pass(
            candidates=rules, min_rules=1, max_rules=2, pass_name=P1,
            scoring_mode="precision_first", y_val=y, min_novelty_ratio=0.2,
            max_union_leakage_rate=sel.max_union_leakage_rate_pass1,
            min_union_ratio=sel.min_union_npv_pass1, min_union_support=sel.min_union_support_pass1)


def test_a_proven_ilp_selection_is_primary_and_its_counts_agree():
    c = contract()
    n, y, rules = _pass1_world()
    sel = contract_selector(c)
    chosen = _run_contract_pass1(sel, rules, y)
    rep = sel.selection_report_[P1]
    report = glass_pass_report("pass1", chosen, y, c)
    counts = glass_union_counts(chosen, y)
    assert (counts.routed_rows, counts.routed_rows - counts.routed_positives) == (
        rep["union_rows"], rep["union_good"])
    assert report == check_pass1_fit(counts, c) and report.all_satisfied
    out = glass_fit_outcome("pass1", rep, report)
    assert out.outcome == OUTCOME_FEASIBLE_OPTIMAL and out.disposition == DISPOSITION_PRIMARY
    assert out.primary_eligible
    assert out.solver["gap"] == 0.0 and out.solver["gap_source"] == "cbc_log"
    assert out.solver["selected_rule_ids"] == tuple(rep["selected_rule_ids"])


def test_an_unproven_incumbent_needs_its_gap_and_a_reproduction():
    c = contract()
    n, y, rules = _pass1_world()
    sel = contract_selector(c)
    script_solver(sel, [real_but_time_limited(sel.ilp_builder), "real"])
    chosen = _run_contract_pass1(sel, rules, y)
    rep = sel.selection_report_[P1]
    report = glass_pass_report("pass1", chosen, y, c)
    out = glass_fit_outcome("pass1", rep, report)
    assert out.outcome == OUTCOME_FEASIBLE_UNPROVEN
    assert out.disposition == DISPOSITION_PRIMARY_IF_DOCUMENTED
    assert out.solver["gap"] is not None and out.solver["reproduced"] is None
    assert not out.documentation_complete and not out.primary_eligible
    rerun = contract_selector(c)
    script_solver(rerun, [real_but_time_limited(rerun.ilp_builder), "real"])
    _run_contract_pass1(rerun, rules, y)
    same = glass_record_reproduction(out, rep, rerun.selection_report_[P1])
    assert same.solver["reproduced"] is True and same.primary_eligible
    differs = glass_record_reproduction(out, rep, {**rerun.selection_report_[P1],
                                                   "selected_rule_ids": ["other"]})
    assert differs.solver["reproduced"] is False and not differs.primary_eligible


def test_an_unproven_incumbent_without_a_solver_log_has_no_gap():
    rep = {"source": "ilp", "lexicographic_guarantee": "UNPROVEN", "selected_rule_ids": ["A"],
           "phase1": {"status": "Optimal", "sol_status": 2, "state": "feasible_unproven",
                      "solver_log": None}}
    c = contract()
    n, y, rules = _pass1_world()
    out = glass_fit_outcome("pass1", rep, glass_pass_report("pass1", rules[:1], y, c))
    assert out.solver["gap"] is None and out.solver["gap_available"] is False
    reproduced = glass_record_reproduction(out, rep, dict(rep))
    assert reproduced.solver["reproduced"] is True and not reproduced.primary_eligible


def test_a_greedy_fallback_requires_review():
    c = contract()
    n, y, rules = _pass1_world()
    sel = contract_selector(c)
    script_solver(sel, [no_solution(sel.ilp_builder)])
    chosen = _run_contract_pass1(sel, rules, y)
    rep = sel.selection_report_[P1]
    out = glass_fit_outcome("pass1", rep, glass_pass_report("pass1", chosen, y, c))
    assert out.outcome == OUTCOME_FEASIBLE_GREEDY and out.disposition == DISPOSITION_REQUIRES_REVIEW
    assert not out.primary_eligible


def test_infeasible_and_unsolved_selections_stop():
    c = contract()
    n, y, rules = _pass1_world()
    sel = contract_selector(c, min_union_support_pass1=400)        # 350 rows at most
    with pytest.raises(SelectorInfeasibleError) as err:
        with contextlib.redirect_stdout(io.StringIO()):
            sel._optimize_pass(candidates=rules, min_rules=1, max_rules=2, pass_name=P1,
                               scoring_mode="precision_first", y_val=y, min_novelty_ratio=0.2,
                               min_union_ratio=0.95, min_union_support=400)
    assert err.value.proven is True
    proven = glass_fit_outcome("pass1", sel.selection_report_[P1], None)
    assert proven.outcome == OUTCOME_INFEASIBLE and proven.disposition == DISPOSITION_STOP

    sel = contract_selector(c, min_union_support_pass1=400)
    script_solver(sel, [no_solution(sel.ilp_builder)])
    with pytest.raises(SelectorInfeasibleError) as err:
        with contextlib.redirect_stdout(io.StringIO()):
            sel._optimize_pass(candidates=rules, min_rules=1, max_rules=2, pass_name=P1,
                               scoring_mode="precision_first", y_val=y, min_novelty_ratio=0.2,
                               min_union_ratio=0.95, min_union_support=400)
    assert err.value.proven is False
    unsolved = glass_fit_outcome("pass1", sel.selection_report_[P1], None)
    assert unsolved.outcome == OUTCOME_NO_SOLUTION and unsolved.disposition == DISPOSITION_STOP

    gated = glass_fit_outcome("pass1", {"source": "infeasible", "stage": "quality gates"}, None)
    assert gated.outcome == OUTCOME_INFEASIBLE
    empty = glass_fit_outcome("pass2", {"source": "empty (min_rules = 0)"}, None)
    assert empty.outcome == OUTCOME_INSUFFICIENT and empty.disposition == DISPOSITION_STOP
    with pytest.raises(ContractViolation, match="unrecognised"):
        glass_fit_outcome("pass1", {"source": "mystery"}, None)


def test_a_feasible_claim_the_independent_check_contradicts_is_refused():
    c = contract()
    n, y, rules = _pass1_world()
    y[300:320] = 1                                   # B now has NPV 30/50 < 0.95
    hist = make_selector()                           # per-rule gates only, no union obligation
    with contextlib.redirect_stdout(io.StringIO()):
        chosen = hist._optimize_pass(candidates=rules, min_rules=2, max_rules=2, pass_name=P1,
                                     scoring_mode="precision_first", y_val=y,
                                     min_novelty_ratio=0.2)
    report = glass_pass_report("pass1", chosen, y, c)
    assert not report.all_satisfied
    with pytest.raises(ContractViolation, match="contradicted"):
        glass_fit_outcome("pass1", hist.selection_report_[P1], report)
    with pytest.raises(ContractViolation, match="scope"):
        glass_pass_report("pass3", chosen, y, c)


# ============================================================================
# 5. historical defaults
# ============================================================================
def test_glass_config_defaults_for_the_new_fields_are_off():
    defaults = {f.name: f.default for f in dataclasses.fields(GlassRouterConfig)}
    assert {k: defaults[k] for k in ("min_union_npv_pass1", "min_union_support_pass1",
                                     "min_union_precision_pass2", "min_union_support_pass2",
                                     "pass2_union_objective", "contract_provenance")} == {
        "min_union_npv_pass1": None, "min_union_support_pass1": None,
        "min_union_precision_pass2": None, "min_union_support_pass2": None,
        "pass2_union_objective": "positives", "contract_provenance": None}
    assert defaults["max_union_leakage_rate_pass1"] is None
    assert defaults["pass2_population"] == "full" and defaults["inner_feature_ranker"] == "outer_rf"


def test_the_selector_time_limit_is_unchanged():
    assert ILPRuleSelector.TIME_LIMIT_PER_SOLVE == 300
