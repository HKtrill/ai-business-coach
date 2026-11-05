"""
The pre-registered PR 38 evaluation protocol (PR 37 Commit 4, Step 4C).

Contract
1. The record loads, validates and has a pinned fingerprint. It binds exactly
   the working contract, carries the F17 primary settings of both adapters,
   the baseline's population identity, upstream operating points and matched-
   rate settings, and refuses drift from any of them. A working protocol may
   not govern PR 38.
2. Identity: a trace is checked against the pinned rows, positives, per-fold
   counts, seeded fold plan and fingerprints; paired routings are aligned by
   row identity and refuse different rows, order, labels or folds.
3. Metrics: denominators are the registry's; pooled values come from summed
   counts, never averaged ratios; undefined stays undefined; class-conditional
   counts reconcile.
4. Uncertainty: the stratified paired bootstrap is deterministic, keeps
   (fold, label) stratum sizes, scores both routings on the same resample and
   reports undefined replicates instead of dropping them.
5. Fit status needs every required selection of a fold exactly once (F17
   keys, the Step 4B selection reports); the fold ledger accepts a fold only
   through a FoldFit that re-classifies from its complete selection records,
   never from a status string; the ledger and pooling never drop a fold
   silently; shortfall classification never relabels an observed violation
   compliant.
6. F31 R2 trigger at 1/20; structural acceptance separate from performance;
   evidence tiers and the (defense-in-depth) input guard; ablation adoption
   gates on a verified upstream reference, with C6 passing only when the
   reported bootstrap run reproduces (a malformed run fails C6, it never
   raises) and the Pass 2 budget held pooled and counted per fold.
8. Upstream identity is per row: artifact SHA-256, recomputed decisions, EBM
   fold thresholds and row-aligned fingerprints catch changes that keep every
   count; reference flags are bound to the pairing's rows.
7. Matched budgets reuse the Step 2C selection: label-free and deterministic
   under row permutation.
Synthetic inputs only: nothing is fitted or read from an artifact.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import shared.stage2.evaluation_protocol as ep
from shared.stage2 import diagnostics as d
from shared.stage2.contract import load_operating_contract, working_profile_path
from shared.stage2.diagnostics import contribution as contrib
from shared.stage2.diagnostics.operating_points import matched_rate_cut
from shared.stage2.obligations import (
    OUTCOME_FEASIBLE_GREEDY, OUTCOME_FEASIBLE_OPTIMAL, OUTCOME_FEASIBLE_UNPROVEN,
    OUTCOME_INFEASIBLE, OUTCOME_INSUFFICIENT, OUTCOME_NO_SOLUTION, SATISFIED, UNDEFINED_STATUS,
    VIOLATED, PassCounts, RoutingCounts, check_pass1_fit, classify_fit_outcome, observe_routing,
)
from tests.stage2.test_stage2_operating_contract_v2 import _v2
from tests.support import diagnostics_world as dw

BANK_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = BANK_ROOT / "research_logs/baselines/pr37_stage2_pre_correction_v1/MANIFEST.json"
#: Pinned identity of the Step 4C working protocol record (changes with any edit).
WORKING_PROTOCOL_FINGERPRINT = "161c6c589696b859"
P1, P2, AB = d.PASS_1_NOT_SUBSCRIBE, d.PASS_2_SUBSCRIBE, d.ABSTAIN


@pytest.fixture(scope="module")
def protocol():
    return ep.load_evaluation_protocol(ep.working_protocol_path())


def _frozen_record(protocol, **changes):
    """``protocol`` as a frozen record: the Step 4D freeze fields (status, version, no
    pending approvals, pinned fingerprints, GLASS reports required, the frozen record's
    arm-profile bindings and freeze decisions, a synthetic confirmed approver), the rest
    unchanged."""
    import copy
    rec = _record(protocol, **changes)
    frozen = json.loads(ep.frozen_protocol_path().read_text(encoding="utf-8"))
    rec.update(status="frozen", protocol_version="1.0.0", pending_approvals=[],
               arm_profiles=copy.deepcopy(frozen["arm_profiles"]),
               freeze=copy.deepcopy(frozen["freeze"]))
    rec["fit_status"]["glass_selection_reports"] = "required"
    rec["freeze"]["approved_by"] = "unit-test approver"      # a confirmed approval (synthetic)
    for name, v in rec["upstream"]["flag_fingerprints"].items():
        rec["upstream"]["flag_fingerprints"][name] = v or "0" * 16
    return rec


@pytest.fixture(scope="module")
def frozen(protocol):
    return ep.EvaluationProtocol.from_dict(_frozen_record(protocol))


@pytest.fixture(scope="module")
def contract():
    return load_operating_contract(working_profile_path())


@pytest.fixture(scope="module")
def world():
    w = dw.world()
    ref, glass, rf, lr, ebm = dw.views(w)
    return dict(ref=ref, glass=glass, rf=rf, lr=lr, ebm=ebm,
                trace=d.build_router_trace(glass, rf, ref))


def _record(protocol, **changes):
    rec = protocol.to_dict()
    for path, value in changes.items():
        node = rec
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return rec


def _errors(protocol, **changes):
    return ep.validate_protocol_record(_record(protocol, **changes))


def _paired(states_a, states_b, y, fold, kind="cross_arm", names=("glass", "blackbox")):
    n = len(y)
    return ep.PairedRouting.build(kind, pd.Index([f"r{i}" for i in range(n)]), np.asarray(y),
                                  np.asarray(fold), {names[0]: states_a, names[1]: states_b})


def _five_fold_world(seed=0, n_per=200):
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(5), n_per)
    y = (rng.random(len(fold)) < 0.15).astype(int)
    a = np.where(rng.random(len(y)) < 0.4, P1, np.where(rng.random(len(y)) < 0.3, P2, AB))
    b = a.copy()
    flip = rng.random(len(y)) < 0.1
    b[flip] = np.where(a[flip] == P1, AB, P1)
    return a.astype(object), b.astype(object), y, fold


# ============================================================================
# 1. the record
# ============================================================================
def test_the_working_protocol_loads_with_its_pinned_fingerprint(protocol):
    assert (protocol.protocol_id, protocol.version, protocol.status) == (
        "pr38-stage2-evaluation", "0.1.0-working", "working")
    assert protocol.fingerprint() == WORKING_PROTOCOL_FINGERPRINT
    again = ep.load_evaluation_protocol(ep.working_protocol_path(),
                                        expected_fingerprint=WORKING_PROTOCOL_FINGERPRINT)
    assert again.fingerprint() == protocol.fingerprint()
    with pytest.raises(ep.ProtocolError, match="fingerprint"):
        ep.load_evaluation_protocol(ep.working_protocol_path(), expected_fingerprint="0" * 16)
    with pytest.raises(ep.ProtocolError, match="status"):
        ep.load_evaluation_protocol(ep.working_protocol_path(), require_status="frozen")


def test_a_frozen_protocol_must_pin_the_upstream_fingerprints(protocol):
    rec = _record(protocol, status="frozen", pending_approvals=[])
    errors = ep.validate_protocol_record(rec)
    assert any("must pin upstream.flag_fingerprints.lr.in_stage_f2_cv" in e for e in errors)
    assert any("must pin upstream.flag_fingerprints.ebm.nested_f2" in e for e in errors)
    assert ep.validate_protocol_record(_frozen_record(protocol)) == []


def test_a_working_protocol_may_not_govern_pr38(protocol, frozen):
    with pytest.raises(ep.ProtocolError, match="only a frozen protocol"):
        ep.require_frozen(protocol)
    assert ep.require_frozen(frozen) is frozen
    assert frozen.fingerprint() != protocol.fingerprint()


def test_the_record_is_read_only(protocol):
    with pytest.raises(TypeError):
        protocol.record["status"] = "frozen"
    with pytest.raises(TypeError):
        protocol["population"]["rows"] = 1


@pytest.mark.parametrize("changes, match", [
    (dict(status="draft"), "status"),
    (dict(schema_version="x"), "schema_version"),
    (dict(contract__contract_fingerprint="abc"), "16 lowercase hex"),
    (dict(population__fold_rows=[6590] * 4), "one value per outer fold"),
    (dict(population__fold_positives=[742, 742, 742, 743, 744]), "sum to"),
    (dict(population__train_label_fingerprint="xyz"), "12 lowercase hex"),
    (dict(hierarchy__secondary_diagnostics=["made_up_metric"]), "unknown diagnostic metric"),
    (dict(upstream__operating_points=["lr.stage4_recall70"]), "primary operating points"),
    (dict(matched_budget__tie_break="row order"), "TIE_BREAK_RULE"),
    (dict(matched_budget__scores={"lr": "raw_model_score", "ebm": "raw_model_score"}),
     "MATCH_SCORES"),
    (dict(matched_budget__population="all_rows"), "common_eligible"),
    (dict(uncertainty__strata=["outer_fold"]), "strata"),
    (dict(uncertainty__replicates=200), "replicates"),
    (dict(shortfall__materiality__multiplier=0), "positive"),
    (dict(f31__trigger="1/10"), "1/20"),
    (dict(evidence_tiers__sources={"bank_marketing_test_split": "tier3"}), "previously_exposed"),
    (dict(fit_status__accepted=["valid_optimal"]), "partition"),
    (dict(acceptance__structural=["V1"]), "every validity gate"),
    (dict(inputs__prohibited_tokens=["test"]), "prohibited_tokens"),
    (dict(status="frozen"), "pending approvals"),
    (dict(uncertainty__procedure_id="my_bootstrap/2"), "procedure_id"),
    (dict(upstream__artifacts__lr__sha256="61a6b144"), "64 lowercase hex"),
    ({"upstream__flag_fingerprints__ebm.nested_f2": "xyz"}, "16 lowercase hex or null"),
    (dict(upstream__flag_fingerprint_method="md5"), "flag_fingerprint_method"),
    (dict(upstream__ebm_fold_thresholds=[0.56]), "one threshold per outer fold"),
])
def test_invalid_records_are_refused(protocol, changes, match):
    errors = _errors(protocol, **changes)
    assert any(match in e for e in errors), errors
    with pytest.raises(ep.ProtocolError):
        ep.EvaluationProtocol.from_dict(_record(protocol, **changes))


def test_malformed_sections_are_reported_not_raised(protocol):
    assert any("JSON objects" in e for e in _errors(protocol, population=[1, 2]))
    assert any("malformed" in e for e in _errors(protocol, hierarchy__validity_gates=[1]))
    assert ep.validate_protocol_record([]) == ["the protocol record must be a JSON object"]


def test_missing_unknown_and_duplicate_keys_are_refused(protocol, tmp_path):
    rec = protocol.to_dict()
    del rec["shortfall"]
    assert any("missing keys" in e for e in ep.validate_protocol_record(rec))
    rec = {**protocol.to_dict(), "extra": 1}
    assert any("unknown keys" in e for e in ep.validate_protocol_record(rec))
    text = json.dumps(protocol.to_dict())
    path = tmp_path / "dup.json"
    path.write_text(text[:-1] + ', "status": "frozen"}', encoding="utf-8")
    with pytest.raises(ep.ProtocolError, match="duplicate key"):
        ep.load_evaluation_protocol(path)


def test_the_protocol_binds_exactly_the_working_contract(protocol, contract):
    ep.check_contract_binding(protocol, contract)
    with pytest.raises(ep.ProtocolError, match="not the contract the protocol binds"):
        ep.check_contract_binding(protocol, _v2(min_pass2_precision=0.35))


def test_the_record_matches_the_f17_primary_settings_of_both_adapters(protocol):
    from blackbox_pipeline.models.rf_router.contract import RF_F17_PRIMARY
    from glass_pipeline.glass_router.contract import GLASS_F17_PRIMARY
    assert dict(protocol["f17_primary"]["glass"]) == dict(GLASS_F17_PRIMARY)
    assert dict(protocol["f17_primary"]["rf"]) == dict(RF_F17_PRIMARY)


def test_the_record_matches_the_frozen_baseline_manifest(protocol):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p, pop, ids = protocol["population"], m["population"], m["identities"]
    assert (p["rows"], p["positives"], p["outer_folds"]) == (
        pop["training_rows"], pop["positives"], pop["outer_folds"])
    assert list(p["fold_rows"]) == pop["rows_per_fold"]
    assert list(p["fold_positives"]) == pop["positives_per_fold"]
    assert p["router_split_fingerprint"] == ids["router_split_fingerprint"]["value"]
    assert p["train_label_fingerprint"] == ids["train_label_fingerprint"]["value"]
    assert p["fold_fingerprint"] == ids["fold_plan"]["fingerprint"]
    assert p["fold_seed"] == ids["fold_plan"]["seed"]
    assert p["stage_io_split_fingerprint"] == ids["stage_io_split_fingerprint"]["value"]
    import hashlib
    lf = MANIFEST.read_bytes().replace(b"\r\n", b"\n")       # verify_baseline's file is LF
    assert p["baseline_manifest_sha256"] == hashlib.sha256(lf).hexdigest()
    up = m["upstream_operating_points"]
    flags = protocol["upstream"]["expected_flags"]
    assert (flags["lr.in_stage_f2_cv"], flags["ebm.nested_f2"], flags["or"], flags["and"]) == (
        up["lr"]["flags"], up["ebm"]["flags"], up["union"]["flags"], up["intersection"]["flags"])
    assert protocol["upstream"]["lr_threshold"] == up["lr"]["threshold"]
    assert list(protocol["upstream"]["ebm_fold_thresholds"]) == [
        up["ebm"]["fold_thresholds"][str(k)] for k in range(5)]
    src = {x["role"]: x for x in m["sources"]}
    for model, role in (("lr", "stage1.lr"), ("ebm", "stage3.ebm")):
        art = protocol["upstream"]["artifacts"][model]
        assert (art["role"], art["path"], art["sha256"]) == (role, src[role]["path"],
                                                               src[role]["sha256"])
    assert protocol["upstream"]["lr_threshold_field"] == up["lr"]["threshold_record"]["field"]
    assert protocol["upstream"]["ebm_decision_field"] == up["ebm"]["field"]
    mr = m["analysis_settings"]["matched_rate"]
    assert protocol["matched_budget"]["tie_break"] == mr["tie_break"] == contrib.TIE_BREAK_RULE
    assert dict(protocol["matched_budget"]["scores"]) == mr["scores"]


def test_every_named_metric_has_an_explicit_definition(protocol):
    h = protocol["hierarchy"]
    names = [o["metric"] for o in h["primary_outcomes"]] + list(h["secondary_diagnostics"]) + \
        list(h["descriptive_totals"])
    for name in names:
        assert (name in d.METRIC_DEFINITIONS or name in ep.PROTOCOL_METRICS
                or name in contrib.CONTRIBUTION_LABELS), name
    assert not set(ep.PROTOCOL_METRICS) & set(d.METRIC_DEFINITIONS)   # never redefined
    assert [o["id"] for o in h["primary_outcomes"]][:2] == ["O1", "O2"]
    assert [o["metric"] for o in h["primary_outcomes"]][:2] == ["pass1_rate", "pass2_rate"]


def test_the_registered_procedure_identities_are_the_implemented_ones(protocol):
    assert protocol["uncertainty"]["procedure_id"] == ep.BOOTSTRAP_PROCEDURE_ID
    assert protocol["upstream"]["flag_fingerprint_method"] == ep.FLAG_FINGERPRINT_METHOD


def test_the_protocol_module_is_arm_neutral():
    tree = ast.parse(Path(ep.__file__).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in mods if m.startswith(("glass_pipeline", "blackbox_pipeline"))}


# ============================================================================
# 2. identity and alignment
# ============================================================================
def _synthetic_identity(trace):
    y, fold = np.asarray(trace.y), np.asarray(trace.fold_id)
    return ep.PopulationIdentity(
        trace.n, int(y.sum()), 5, dw.SEED, tuple(int((fold == k).sum()) for k in range(5)),
        tuple(int(y[fold == k].sum()) for k in range(5)), trace.router_split_fingerprint,
        trace.train_label_fingerprint, trace.views["glass"].metadata["oof_protocol"][
            "fold_fingerprint"])


def test_a_trace_matching_its_identity_passes(world):
    report = ep.check_population_identity(world["trace"], _synthetic_identity(world["trace"]))
    assert report.passed, report.failures()
    assert report.require() is report
    assert len(report.to_frame()) == len(report.checks) >= 10


@pytest.mark.parametrize("field, value, check", [
    ("rows", 2301, "rows"),
    ("positives", 1, "positives"),
    ("fold_rows", (460, 460, 460, 460, 461), "rows per fold"),
    ("fold_seed", 7, "fold plan"),
    ("router_split_fingerprint", "0" * 16, "router split fingerprint"),
    ("train_label_fingerprint", "0" * 12, "train label fingerprint"),
    ("fold_fingerprint", "0" * 16, "fold fingerprint"),
])
def test_any_identity_difference_fails_the_check(world, field, value, check):
    ident = dataclasses.replace(_synthetic_identity(world["trace"]), **{field: value})
    report = ep.check_population_identity(world["trace"], ident)
    assert not report.passed and any(check in f for f in report.failures())
    with pytest.raises(ep.ProtocolError, match="identity check failed"):
        report.require()


def test_the_pinned_identity_is_the_protocol_population(protocol):
    ident = ep.PopulationIdentity.from_protocol(protocol)
    assert (ident.rows, ident.positives, ident.fold_fingerprint) == (32950, 3712,
                                                                     "8aea082fcaf371b7")


def _pinned_upstream(directory, world, protocol):
    """LR / EBM views extracted from artifact FILES, and a protocol pinning those files."""
    import joblib
    w = dw.world()
    paths = {"lr": directory / "lr.joblib", "ebm": directory / "ebm.joblib"}
    joblib.dump(w["lr"], paths["lr"])
    joblib.dump(w["ebm"], paths["ebm"])
    lr = d.extract_lr_view(paths["lr"], world["ref"])
    ebm = d.extract_ebm_view(paths["ebm"], world["ref"])
    refs = d.primary_reference_flags(lr, ebm)
    t = world["trace"]
    pinned = _record(
        protocol,
        upstream__artifacts={m: {"role": r, "path": str(paths[m]), "sha256": d.file_sha256(
            paths[m])} for m, r in (("lr", "stage1.lr"), ("ebm", "stage3.ebm"))},
        upstream__expected_flags={"lr.in_stage_f2_cv": refs["lr"].n_flagged,
                                  "ebm.nested_f2": refs["ebm"].n_flagged,
                                  "or": refs["union"].n_flagged,
                                  "and": refs["intersection"].n_flagged},
        upstream__ebm_fold_thresholds=list(dw.EBM_NESTED),
        upstream__flag_fingerprints={
            "lr.in_stage_f2_cv": ep.flag_fingerprint(t.index, refs["lr"].flags),
            "ebm.nested_f2": ep.flag_fingerprint(t.index, refs["ebm"].flags)})
    return dict(lr=lr, ebm=ebm, refs=refs, paths=paths, trace=t, w=w,
                sources=ep.UpstreamSources(refs, lr, ebm),
                protocol=ep.EvaluationProtocol.from_dict(pinned))


@pytest.fixture()
def upstream(world, protocol, tmp_path):
    return _pinned_upstream(tmp_path, world, protocol)


def _upstream(u, *, refs=None, lr=None, ebm=None, protocol=None):
    return ep.check_upstream_identity(refs or u["refs"], protocol or u["protocol"],
                                      lr_view=lr or u["lr"], ebm_view=ebm or u["ebm"],
                                      population=u["trace"])


def _count_preserving_swap(refs):
    """Two rows outside the EBM flags, one LR-flagged and one not: swapping their LR
    flags keeps the LR, EBM, union and intersection counts."""
    lr, ebm = np.asarray(refs["lr"].flags), np.asarray(refs["ebm"].flags)
    a = int(np.flatnonzero(lr & ~ebm)[0])
    b = int(np.flatnonzero(~lr & ~ebm)[0])
    return a, b


def test_frozen_upstream_points_pass_every_layer(upstream):
    report = _upstream(upstream)
    assert report.passed and report.complete, report.failures() + report.pending()
    assert report.require(complete=True) is report
    names = [c[0] for c in report.checks]
    for needed in ("lr artifact SHA-256 at extraction", "ebm artifact SHA-256 now",
                   "ebm.nested_f2 fold thresholds",
                   "ebm.nested_f2 per-row thresholds are their fold's",
                   "lr.in_stage_f2_cv flags equal the registered operating point, row by row",
                   "lr.in_stage_f2_cv row-aligned flag fingerprint"):
        assert needed in names
    assert report.values["lr.in_stage_f2_cv"] == upstream["protocol"]["upstream"][
        "flag_fingerprints"]["lr.in_stage_f2_cv"]


def test_supplied_flags_moved_between_rows_are_caught_with_all_counts_kept(upstream):
    refs = upstream["refs"]
    a, b = _count_preserving_swap(refs)
    lr = np.array(refs["lr"].flags, copy=True)
    lr[[a, b]] = lr[[b, a]]
    union = lr | np.asarray(refs["ebm"].flags)
    inter = lr & np.asarray(refs["ebm"].flags)
    forged = {**refs, "lr": dataclasses.replace(refs["lr"], flags=lr),
              "union": dataclasses.replace(refs["union"], flags=union),
              "intersection": dataclasses.replace(refs["intersection"], flags=inter)}
    for key in ("lr", "union", "intersection"):
        assert forged[key].n_flagged == refs[key].n_flagged          # every count kept
    report = _upstream(upstream, refs=forged)
    failed = "\n".join(report.failures())
    assert not report.passed
    assert "lr.in_stage_f2_cv flags equal the registered operating point" in failed
    assert "flag count" not in failed


def test_a_tampered_view_is_caught_by_the_row_fingerprint(upstream):
    """Scores swapped inside the extracted view: the recomputed flags agree with
    the supplied ones and every count is kept, but the row fingerprint moves."""
    lr = upstream["lr"]
    a, b = _count_preserving_swap(upstream["refs"])
    stream = lr.scores["calibrated_probability"]
    vals = np.array(stream.values, copy=True)
    vals[[a, b]] = vals[[b, a]]
    tampered = dataclasses.replace(lr, scores={"calibrated_probability": dataclasses.replace(
        stream, values=vals)})
    refs = d.primary_reference_flags(tampered, upstream["ebm"])
    assert refs["lr"].n_flagged == upstream["refs"]["lr"].n_flagged
    report = _upstream(upstream, refs=refs, lr=tampered)
    failed = report.failures()
    assert failed == [f for f in failed if "row-aligned flag fingerprint" in f] and len(failed) == 1


def test_a_changed_artifact_file_is_caught_by_its_sha256(upstream):
    import joblib
    w = upstream["w"]
    art = dict(w["lr"]) if isinstance(w["lr"], dict) else w["lr"]
    joblib.dump(art, upstream["paths"]["lr"], compress=3)            # same content, new bytes
    report = _upstream(upstream)
    failed = "\n".join(report.failures())
    assert "lr artifact SHA-256 now" in failed
    assert "lr artifact SHA-256 at extraction" not in failed         # it was right when opened


def test_ebm_fold_threshold_provenance_is_checked(upstream):
    other = _record(upstream["protocol"], upstream__ebm_fold_thresholds=[0.5] * 5)
    report = _upstream(upstream, protocol=ep.EvaluationProtocol.from_dict(other))
    assert any("ebm.nested_f2 fold thresholds" in f for f in report.failures())
    ebm = upstream["ebm"]
    sd = ebm.stored_decisions["ebm.nested_f2"]
    per_row = np.array(sd.per_row_threshold, copy=True)
    per_row[0] = per_row[0] + 0.01
    moved = dataclasses.replace(ebm, stored_decisions={"ebm.nested_f2": dataclasses.replace(
        sd, per_row_threshold=per_row)})
    report = _upstream(upstream, ebm=moved)
    assert any("per-row thresholds" in f for f in report.failures())


def test_views_without_an_artifact_file_or_on_other_rows_fail(upstream, world):
    report = _upstream(upstream, lr=world["lr"], refs=d.primary_reference_flags(
        world["lr"], upstream["ebm"]))
    failed = "\n".join(report.failures())
    assert "lr artifact path recorded" in failed and "lr artifact SHA-256 at extraction" in failed
    lr = upstream["lr"]
    shifted = dataclasses.replace(lr, index=lr.index[::-1])
    report = _upstream(upstream, lr=shifted)
    assert any("lr rows and labels" in f for f in report.failures())


def test_unpinned_fingerprints_are_pending_not_passed(upstream):
    rec = _record(upstream["protocol"], upstream__flag_fingerprints={
        "lr.in_stage_f2_cv": None, "ebm.nested_f2": None})
    report = _upstream(upstream, protocol=ep.EvaluationProtocol.from_dict(rec))
    assert report.passed and not report.complete and len(report.pending()) == 2
    with pytest.raises(ep.ProtocolError, match="incomplete"):
        report.require(complete=True)


def test_flag_fingerprints_are_row_aligned():
    idx = pd.Index(["a", "b", "c", "d"])
    f = np.array([1, 0, 1, 0], bool)
    assert ep.flag_fingerprint(idx, f) == ep.flag_fingerprint(list(idx), f.astype(int))
    assert ep.flag_fingerprint(idx, f) != ep.flag_fingerprint(idx, f[[1, 0, 2, 3]])
    assert ep.flag_fingerprint(idx, f) != ep.flag_fingerprint(idx[[1, 0, 2, 3]], f)
    with pytest.raises(ep.ProtocolError):
        ep.flag_fingerprint(idx, f[:3])


def test_pre_post_pairs_only_identical_rows_labels_and_folds(world):
    g = world["glass"]
    same = ep.PairedRouting.pre_post(g, g)
    assert same.kind == "pre_post" and same.names == ("pre", "post") and same.changed_rows() == 0
    reordered = dataclasses.replace(g, index=g.index[::-1])
    with pytest.raises(ep.ProtocolError, match="row identities differ"):
        ep.PairedRouting.pre_post(g, reordered)
    y = np.array(g.y, copy=True)
    y[0] = 1 - y[0]
    with pytest.raises(ep.ProtocolError, match="labels differ"):
        ep.PairedRouting.pre_post(g, dataclasses.replace(g, y=y))
    fold = np.array(g.fold_id, copy=True)
    fold[[0, 1]] = fold[[1, 0]] if fold[0] != fold[1] else (fold[0] + 1) % 5
    with pytest.raises(ep.ProtocolError, match="fold ids differ"):
        ep.PairedRouting.pre_post(g, dataclasses.replace(g, fold_id=fold))
    with pytest.raises(ep.ProtocolError, match="different arms"):
        ep.PairedRouting.pre_post(g, world["rf"])


def test_paired_routings_refuse_malformed_inputs():
    a, b, y, fold = _five_fold_world()
    with pytest.raises(ep.ProtocolError, match="unique"):
        ep.PairedRouting.build("cross_arm", pd.Index(["r"] * len(y)), y, fold, {"a": a, "b": b})
    bad = a.copy()
    bad[0] = "uncertain"                            # a stored token, not a router state
    with pytest.raises(ep.ProtocolError, match="unknown states"):
        _paired(bad, b, y, fold)
    with pytest.raises(ep.ProtocolError, match="binary"):
        _paired(a, b, np.where(y == 1, 2, 0), fold)
    with pytest.raises(ep.ProtocolError, match="exactly two"):
        ep.PairedRouting.build("cross_arm", pd.Index(range(len(y))), y, fold, {"a": a})
    with pytest.raises(ep.ProtocolError, match="kind"):
        _paired(a, b, y, fold, kind="matched")


def test_the_cross_arm_pair_comes_from_the_validated_trace(world):
    t = world["trace"]
    paired = ep.PairedRouting.from_trace(t)
    assert paired.names == ("glass", "blackbox") and paired.index.equals(t.index)
    for arm in paired.names:
        c = paired.counts(arm)
        view = t.views[arm]
        assert (c.pass1_rows, c.pass2_rows, c.abstain_rows) == (
            int(view.pass1.sum()), int(view.pass2.sum()), int(view.abstain.sum()))
    tab = paired.transitions()
    assert int(tab["rows"].sum()) == t.n and int(tab["positives"].sum()) == t.n_positives
    assert int(tab.loc[tab["changed"], "rows"].sum()) == paired.changed_rows()


# ============================================================================
# 3. metrics, denominators, pooling
# ============================================================================
def test_denominators_are_the_registry_definitions():
    c = RoutingCounts(rows=100, positives=20, pass1_rows=40, pass1_positives=2, pass2_rows=10,
                      pass2_positives=4)
    r = ep.protocol_ratios(c)
    expect = {"pass1_npv": (38, 40), "pass1_leakage_share": (2, 20), "pass2_precision": (4, 10),
              "pass1_rate": (40, 100), "pass2_rate": (10, 100), "coverage": (50, 100),
              "abstain_rate": (50, 100), "pass2_remainder_rate": (10, 60),
              "pass1_negative_recall": (38, 80), "pass2_recall": (4, 20),
              "pass2_recall_eligible": (4, 18), "pass2_negative_routing_rate": (6, 80),
              "abstain_positive_rate": (14, 50), "abstain_negative_share": (36, 80)}
    for name, (num, den) in expect.items():
        assert (r[name].numerator, r[name].denominator) == (num, den), name


def test_pooled_values_sum_counts_and_are_not_averaged_ratios():
    fold = np.repeat([0, 1], [10, 100])
    y = np.zeros(110, int)
    a = np.array([AB] * 110, dtype=object)
    a[:10] = P2
    y[:5] = 1                                       # fold 0: 5/10 precision
    a[10:110] = P2
    y[10:20] = 1                                    # fold 1: 10/100 precision
    p = _paired(a, a.copy(), y, fold)
    tab = ep.metric_table(p, "glass", ("pass2_precision",))
    pooled = tab[tab["fold"].map(lambda f: f == "all")].iloc[0]
    assert (pooled["numerator"], pooled["denominator"]) == (15, 110)
    assert pooled["value"] == pytest.approx(15 / 110) != pytest.approx((0.5 + 0.1) / 2)


def test_undefined_denominators_stay_undefined_and_are_counted():
    a, b, y, fold = _five_fold_world()
    a = np.where((fold == 2) & (a == P2), AB, a).astype(object)   # no Pass 2 rows in fold 2
    p = _paired(a, b, y, fold)
    tab = ep.metric_table(p, "glass", ("pass2_precision",))
    f2 = tab[tab["fold"].map(lambda f: f == 2)].iloc[0]
    assert f2["status"] == "undefined_empty_denominator" and pd.isna(f2["value"])
    assert (f2["numerator"], f2["denominator"]) == (0, 0)
    disp = ep.fold_dispersion(tab).iloc[0]
    assert (disp["folds_defined"], disp["folds_undefined"]) == (4, 1)
    assert disp["min"] <= disp["pooled"] <= disp["max"]
    obs = observe_routing(p.counts("glass", 2), load_operating_contract(working_profile_path()))
    assert obs.result("pass2_precision_floor").status == UNDEFINED_STATUS


def test_class_conditional_counts_reconcile():
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    cc = ep.class_conditional_counts(p, "glass")
    assert int(cc["positives"].sum()) == int(y.sum())
    assert int(cc["negatives"].sum()) == int((y == 0).sum())
    assert cc["share_of_positives"].sum() == pytest.approx(1.0)
    p1 = cc[cc["state"] == P1].iloc[0]
    assert p1["positives"] == int(((a == P1) & (y == 1)).sum())


# ============================================================================
# 4. the stratified paired bootstrap
# ============================================================================
def test_bootstrap_settings_come_from_the_protocol(protocol):
    assert ep.bootstrap_settings(protocol) == {"replicates": 2000, "seed": 42, "level": 0.95}


def test_the_bootstrap_is_deterministic_and_seeded():
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    kw = dict(metrics=["pass1_rate", "pass1_npv"], replicates=300, level=0.95)
    one = ep.stratified_paired_bootstrap(p, seed=42, **kw)
    two = ep.stratified_paired_bootstrap(p, seed=42, **kw)
    other = ep.stratified_paired_bootstrap(p, seed=43, **kw)
    pd.testing.assert_frame_equal(one, two)
    assert not one.equals(other)


def test_point_values_are_the_exact_counts():
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    boot = ep.stratified_paired_bootstrap(p, ["pass1_npv", "pass2_precision", "coverage"],
                                          replicates=200, seed=1, level=0.95)
    for name in ("glass", "blackbox"):
        r = ep.protocol_ratios(p.counts(name))
        for m in ("pass1_npv", "pass2_precision", "coverage"):
            assert boot.set_index("metric").loc[m, name] == pytest.approx(r[m].value)


def test_strata_sizes_are_kept_in_every_replicate():
    """Pass 1 = all of fold 0: its share is 1/5 of rows and its positives' share of all
    positives is fixed, in every replicate, only if (fold, label) strata keep their size."""
    a, b, y, fold = _five_fold_world()
    a = np.where(fold == 0, P1, AB).astype(object)
    p = _paired(a, b, y, fold)
    boot = ep.stratified_paired_bootstrap(p, ["pass1_rate", "pass1_leakage_share"],
                                          replicates=200, seed=3, level=0.95).set_index("metric")
    share = y[fold == 0].sum() / y.sum()
    assert boot.loc["pass1_rate", "glass_low"] == boot.loc["pass1_rate", "glass_high"] == 0.2
    assert boot.loc["pass1_leakage_share", "glass_low"] == pytest.approx(share)
    assert boot.loc["pass1_leakage_share", "glass_high"] == pytest.approx(share)


def test_both_routings_are_scored_on_the_same_resample():
    a, b, y, fold = _five_fold_world()
    same = ep.stratified_paired_bootstrap(_paired(a, a.copy(), y, fold), ["pass1_npv"],
                                          replicates=300, seed=5, level=0.95).iloc[0]
    assert same["difference_low"] == same["difference_high"] == same["difference"] == 0.0
    near = ep.stratified_paired_bootstrap(_paired(a, b, y, fold), ["pass1_rate"],
                                          replicates=300, seed=5, level=0.95).iloc[0]
    width_arm = near["glass_high"] - near["glass_low"]
    width_diff = near["difference_high"] - near["difference_low"]
    assert width_diff < width_arm


def test_undefined_replicates_are_reported_not_dropped():
    a, b, y, fold = _five_fold_world()
    a = np.where(a == P2, AB, a).astype(object)
    a[np.flatnonzero(fold == 0)[0]] = P2              # one Pass 2 row in the whole routing
    boot = ep.stratified_paired_bootstrap(_paired(a, b, y, fold), ["pass2_precision"],
                                          replicates=400, seed=2, level=0.95).iloc[0]
    assert boot["undefined_replicates"] > 0
    assert boot["interval_status"] == "undefined_empty_denominator"
    assert boot["glass_low"] is None and boot["difference_low"] is None
    assert boot["blackbox_low"] is not None


def _reference(p, flags, name="union"):
    return ep.ReferenceFlags.build(name, p.index, p.y, flags, "unit-test reference")


def test_contribution_metrics_need_the_frozen_reference():
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    with pytest.raises(ep.ProtocolError, match="reference flags"):
        ep.stratified_paired_bootstrap(p, ["unique_tp"], replicates=100, seed=0, level=0.95)
    ref = np.zeros(len(y), bool)
    ref[::3] = True
    with pytest.raises(ep.ProtocolError, match="not a bare array"):
        ep.stratified_paired_bootstrap(p, ["unique_tp"], replicates=100, seed=0, level=0.95,
                                       reference=ref)
    boot = ep.stratified_paired_bootstrap(p, ["unique_tp", "pass1_fp_suppression"],
                                          replicates=100, seed=0, level=0.95,
                                          reference=_reference(p, ref)).set_index("metric")
    assert boot.loc["unique_tp", "glass"] == int(((a == P2) & (y == 1) & ~ref).sum())
    assert boot.loc["unique_tp", "kind"] == "count"
    with pytest.raises(ep.ProtocolError, match="unknown bootstrap metrics"):
        ep.stratified_paired_bootstrap(p, ["auc"], replicates=100, seed=0, level=0.95)


def test_a_reordered_or_relabelled_reference_is_refused():
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    ref = np.zeros(len(y), bool)
    ref[::3] = True
    perm = np.random.default_rng(0).permutation(len(y))
    reordered = ep.ReferenceFlags.build("union", p.index[perm], p.y[perm], ref[perm], "x")
    with pytest.raises(ep.ProtocolError, match="never aligned by position"):
        ep.stratified_paired_bootstrap(p, ["unique_tp"], replicates=50, seed=0, level=0.95,
                                       reference=reordered)
    relabelled = ep.ReferenceFlags.build("union", p.index, 1 - p.y, ref, "x")
    with pytest.raises(ep.ProtocolError, match="labels differ"):
        ep.stratified_paired_bootstrap(p, ["unique_tp"], replicates=50, seed=0, level=0.95,
                                       reference=relabelled)
    with pytest.raises(ep.ProtocolError, match="UpstreamFlags"):
        ep.ReferenceFlags.from_upstream(ref)


def test_reference_metrics_equal_the_diagnostics_registry(world):
    """Point values with the frozen union equal router_upstream_metrics on the same trace."""
    t = world["trace"]
    union = d.primary_reference_flags(world["lr"], world["ebm"])["union"]
    p = ep.PairedRouting.from_trace(t)
    boot = ep.stratified_paired_bootstrap(
        p, ["unique_tp", "unique_fp", "pass1_fp_suppression", "pass1_tp_cost"], replicates=50,
        seed=0, level=0.95, reference=ep.ReferenceFlags.from_upstream(union)).set_index("metric")
    for arm in p.names:
        reg = d.router_upstream_metrics(t.views[arm], union)
        assert boot.loc["unique_tp", arm] == reg["pass2_unique_positives_n"]
        assert boot.loc["unique_fp", arm] == reg["pass2_new_false_positive_rate"].numerator
        assert boot.loc["pass1_fp_suppression", arm] == pytest.approx(
            reg["pass1_fp_suppression"].value)
        assert boot.loc["pass1_tp_cost", arm] == pytest.approx(reg["pass1_tp_cost"].value)


def test_wilson_intervals_are_descriptive_and_undefined_at_zero():
    from feature_research.metrics import wilson_ci
    for num, den in ((95, 100), (3, 75), (0, 125), (125, 125)):
        lo, hi = ep.wilson_interval(num, den)
        assert (lo, hi) == pytest.approx(wilson_ci(num, den))
    assert ep.wilson_interval(0, 0) is None
    with pytest.raises(ep.ProtocolError):
        ep.wilson_interval(5, 4)


# ============================================================================
# 5. fit status, ledger, pooling, shortfalls
# ============================================================================
def _fit(contract, outcome, *, ok=True, solver=None):
    if ok:
        counts = PassCounts(population_rows=1000, population_positives=100, routed_rows=300,
                            routed_positives=5)
    else:
        counts = PassCounts(population_rows=1000, population_positives=100, routed_rows=60,
                            routed_positives=20)
    report = check_pass1_fit(counts, contract)
    return classify_fit_outcome(outcome, scope="pass1", reason="unit", report=report,
                                solver=solver)


def _fit2(contract, outcome, *, ok=True, solver=None):
    from shared.stage2.obligations import check_pass2_fit
    counts = (PassCounts(population_rows=1000, population_positives=150, routed_rows=200,
                         routed_positives=80) if ok else
              PassCounts(population_rows=1000, population_positives=150, routed_rows=40,
                         routed_positives=4))
    return classify_fit_outcome(outcome, scope="pass2", reason="unit",
                                report=check_pass2_fit(counts, contract), solver=solver)


def _records(contract, arm, protocol, **override):
    """A complete record set for one outer fold; ``override``: key -> FitOutcome."""
    out = []
    for key in ep.required_selection_keys(arm, protocol):
        outcome = override.get(key) or (_fit2(contract, OUTCOME_FEASIBLE_OPTIMAL)
                                        if key == "pass2"
                                        else _fit(contract, OUTCOME_FEASIBLE_OPTIMAL))
        out.append(ep.SelectionRecord(key, outcome))
    return out


def _status(records, protocol, arm="glass", **kw):
    return ep.classify_fold_fit(records, arm=arm, protocol=protocol, **kw).status


def _fold_fit(contract, protocol, arm="glass", status="valid_optimal"):
    """A FoldFit with ``status``, classified from a complete record set."""
    override, reviews = {}, None
    if status == "valid_unproven_documented":
        override["pass1"] = _fit(contract, OUTCOME_FEASIBLE_UNPROVEN, solver={
            "status": "Optimal", "gap": 0.02, "reproduced": True})
    elif status == "fallback_accepted_after_review":
        override["pass2"] = _fit2(contract, OUTCOME_FEASIBLE_GREEDY)
        reviews = {"pass2": {"reviewer": "PJ", "rationale": "objective gap 2 rows"}}
    fit = ep.classify_fold_fit(_records(contract, arm, protocol, **override), arm=arm,
                               protocol=protocol, reviews=reviews)
    assert fit.status == status
    return fit


def test_required_selection_keys_follow_the_f17_settings(protocol):
    assert ep.required_selection_keys("glass", protocol) == (
        "pass1", "pass2", "inner_pass1[0]", "inner_pass1[1]", "inner_pass1[2]",
        "inner_pass1[3]", "inner_pass1[4]")
    assert ep.required_selection_keys("blackbox", protocol) == (
        "pass1", "pass2", "eligibility_t1[0]", "eligibility_t1[1]", "eligibility_t1[2]",
        "eligibility_t1[3]", "eligibility_t1[4]")
    rec = _record(protocol, f17_primary__glass__pass1_oof_folds=3)
    assert len(ep.required_selection_keys("glass", ep.EvaluationProtocol.from_dict(rec))) == 5
    with pytest.raises(ep.ProtocolError, match="arm must be"):
        ep.required_selection_keys("rf", protocol)


def test_fold_fit_status_follows_the_fit_outcomes(contract, protocol):
    assert _status(_records(contract, "glass", protocol), protocol) == "valid_optimal"
    assert _status(_records(contract, "blackbox", protocol), protocol, arm="blackbox") == \
        "valid_optimal"
    unproven = _fit(contract, OUTCOME_FEASIBLE_UNPROVEN, solver={"status": "Optimal", "gap": 0.02})
    assert _status(_records(contract, "glass", protocol, **{"inner_pass1[2]": unproven}),
                   protocol) == "valid_unproven_undocumented"
    documented = _fit(contract, OUTCOME_FEASIBLE_UNPROVEN,
                      solver={"status": "Optimal", "gap": 0.02, "reproduced": True})
    assert _status(_records(contract, "glass", protocol, pass1=documented), protocol) == \
        "valid_unproven_documented"
    greedy = _fit2(contract, OUTCOME_FEASIBLE_GREEDY)
    recs = _records(contract, "glass", protocol, pass2=greedy)
    assert _status(recs, protocol) == "fallback_unreviewed"
    assert _status(recs, protocol, reviews={"pass2": {"reviewer": "x", "rationale": ""}}) == \
        "fallback_unreviewed"
    assert _status(recs, protocol, reviews={"pass1": {"reviewer": "PJ", "rationale": "ok"}}) == \
        "fallback_unreviewed"                                    # the review is per selection
    assert _status(recs, protocol, reviews={
        "pass2": {"reviewer": "PJ", "rationale": "objective gap 2 rows; comparable"}}) == \
        "fallback_accepted_after_review"
    for outcome, status in ((OUTCOME_INFEASIBLE, "proven_infeasible"),
                            (OUTCOME_NO_SOLUTION, "no_feasible_solution"),
                            (OUTCOME_INSUFFICIENT, "insufficient_population_or_support")):
        stop = _fit(contract, outcome, ok=False)
        assert _status(_records(contract, "glass", protocol, **{"inner_pass1[4]": stop},
                                pass1=documented), protocol) == status
    forged = dataclasses.replace(_fit(contract, OUTCOME_FEASIBLE_OPTIMAL), report=None)
    assert _status(_records(contract, "glass", protocol, pass1=forged), protocol) == \
        "fit_time_invalid"
    worst = _records(contract, "glass", protocol,
                     pass1=_fit(contract, OUTCOME_NO_SOLUTION, ok=False),
                     **{"inner_pass1[0]": _fit(contract, OUTCOME_INFEASIBLE, ok=False)})
    fit = ep.classify_fold_fit(worst, arm="glass", protocol=protocol)
    assert fit.status == "proven_infeasible" and fit.by_key["pass1"] == "no_feasible_solution"


@pytest.mark.parametrize("arm", ["glass", "blackbox"])
def test_an_accepted_status_cannot_come_from_a_subset_of_the_fits(contract, protocol, arm):
    full = _records(contract, arm, protocol)
    for drop in range(len(full)):
        subset = full[:drop] + full[drop + 1:]
        fit = ep.classify_fold_fit(subset, arm=arm, protocol=protocol)
        assert fit.status == "fit_time_invalid"
        assert fit.completeness.missing == (full[drop].key,)
    assert ep.classify_fold_fit([], arm=arm, protocol=protocol).status == "fit_time_invalid"
    only_outer = [r for r in full if r.key in ("pass1", "pass2")]
    assert ep.classify_fold_fit(only_outer, arm=arm, protocol=protocol).completeness.missing


def test_duplicated_unexpected_and_unsupported_records_are_refused(contract, protocol):
    full = _records(contract, "glass", protocol)
    dup = ep.check_selection_completeness(full + [full[3]], arm="glass", protocol=protocol)
    assert dup.duplicated == ("inner_pass1[1]",) and not dup.complete
    extra = full + [ep.SelectionRecord("inner_pass1[5]", full[2].outcome),
                    ep.SelectionRecord("eligibility_t1[0]", full[2].outcome),
                    ep.SelectionRecord("pass3", full[0].outcome)]
    unexp = ep.check_selection_completeness(extra, arm="glass", protocol=protocol)
    assert set(unexp.unexpected) == {"inner_pass1[5]", "eligibility_t1[0]", "pass3"}
    wrong_scope = [ep.SelectionRecord(r.key, full[0].outcome if r.key == "pass2" else r.outcome)
                   for r in full]
    bad = ep.check_selection_completeness(wrong_scope, arm="glass", protocol=protocol)
    assert any("pass2: outcome scope 'pass1'" in u for u in bad.unsupported)
    not_outcome = full[:-1] + [ep.SelectionRecord(full[-1].key, {"source": "ilp"})]
    assert any("not a FitOutcome" in u for u in ep.check_selection_completeness(
        not_outcome, arm="glass", protocol=protocol).unsupported)
    assert any("not a SelectionRecord" in u for u in ep.check_selection_completeness(
        full + [full[0].outcome], arm="glass", protocol=protocol).unsupported)
    for records in (full + [full[3]], extra, wrong_scope):
        assert ep.classify_fold_fit(records, arm="glass", protocol=protocol).status == \
            "fit_time_invalid"


def test_records_must_match_the_glass_selection_reports(contract, protocol):
    full = _records(contract, "glass", protocol)
    proven = {"source": "ilp", "lexicographic_guarantee": "proven"}
    reports = {"pass1": dict(proven), "pass2": dict(proven),
               "inner_pass1": [dict(proven) for _ in range(5)]}
    ok = ep.classify_fold_fit(full, arm="glass", protocol=protocol, selection_reports=reports)
    assert ok.status == "valid_optimal" and ok.completeness.complete
    short = {**reports, "inner_pass1": reports["inner_pass1"][:4]}   # one inner fold never ran
    fit = ep.classify_fold_fit(full, arm="glass", protocol=protocol, selection_reports=short)
    assert fit.status == "fit_time_invalid"
    assert "inner_pass1[4]" in fit.completeness.missing
    assert any("no selection report" in u for u in fit.completeness.unsupported)
    with pytest.raises(ep.ProtocolError, match="selection_reports"):
        ep.classify_fold_fit(full, arm="glass", protocol=protocol, selection_reports={"pass1": 1})
    with pytest.raises(ep.ProtocolError, match="GLASS pipeline"):
        ep.check_selection_completeness(_records(contract, "blackbox", protocol), arm="blackbox",
                                        protocol=protocol, selection_reports=reports)


def test_the_ledger_takes_fold_fits_of_its_own_arm(contract, protocol):
    fit = ep.classify_fold_fit(_records(contract, "glass", protocol), arm="glass",
                               protocol=protocol)
    led = ep.FoldLedger.build({"glass": {k: fit for k in range(5)}}, protocol)
    assert led.complete(["glass"])
    with pytest.raises(ep.ProtocolError, match="is for arm 'glass'"):
        ep.FoldLedger.build({"blackbox": {0: fit}}, protocol)


def test_the_ledger_counts_attempted_completed_and_accepted_folds(contract, protocol):
    optimal = _fold_fit(contract, protocol)
    status = {"glass": {0: optimal, 1: optimal, 2: "no_feasible_solution",
                        3: _fold_fit(contract, protocol, status="valid_unproven_documented")},
              "blackbox": {k: _fold_fit(contract, protocol, "blackbox") for k in range(5)}}
    led = ep.FoldLedger.build(status, protocol)
    s = led.summary().set_index("arm")
    assert (s.loc["glass", "attempted"], s.loc["glass", "reported"],
            s.loc["glass", "fit_completed"], s.loc["glass", "accepted"]) == (5, 4, 3, 3)
    assert s.loc["glass", "missing"] == [4]
    assert s.loc["glass", "not_accepted"] == {2: "no_feasible_solution"}
    assert s.loc["glass", "status_only"] == [2] and s.loc["blackbox", "status_only"] == []
    assert led.complete(["blackbox"]) and not led.complete(["glass", "blackbox"])
    with pytest.raises(ep.ProtocolError) as err:
        led.require_complete(["glass", "blackbox"])
    assert "glass fold 2: no_feasible_solution" in str(err.value)
    assert "glass fold 4: no fit record" in str(err.value)
    with pytest.raises(ep.ProtocolError, match="not attempted"):
        ep.FoldLedger.build({"glass": {7: optimal}}, protocol)
    with pytest.raises(ep.ProtocolError, match="unknown fit status"):
        ep.FoldLedger.build({"glass": {0: "fine"}}, protocol)


def test_a_status_string_never_makes_a_fold_accepted(contract, protocol):
    for status in ep.ACCEPTED_FIT_STATUSES:
        with pytest.raises(ep.ProtocolError, match="needs the fold's FoldFit"):
            ep.FoldLedger.build({"glass": {0: status}}, protocol)
    led = ep.FoldLedger.build({"glass": {0: "fallback_unreviewed", 1: "fit_time_invalid"}},
                              protocol)
    assert led.accepted_folds("glass") == ()
    assert led.summary().loc[0, "status_only"] == [0, 1]
    attempted = tuple(range(5))
    with pytest.raises(ep.ProtocolError, match="no FoldFit with complete selection records"):
        ep.FoldLedger(attempted, {"glass": {0: "valid_optimal"}})        # constructed directly
    fit = _fold_fit(contract, protocol)
    with pytest.raises(ep.ProtocolError, match="no FoldFit with complete selection records"):
        ep.FoldLedger(attempted, {"glass": {0: "valid_unproven_documented"}},
                      {"glass": {0: fit}})                     # a FoldFit of another status
    with pytest.raises(ep.ProtocolError, match="no FoldFit with complete selection records"):
        ep.FoldLedger(attempted, {"blackbox": {0: "valid_optimal"}}, {"blackbox": {0: fit}})


def test_an_accepted_fold_fit_needs_its_complete_records_and_must_reproduce(contract, protocol):
    full = _records(contract, "glass", protocol)
    incomplete = ep.classify_fold_fit(full[:-1], arm="glass", protocol=protocol)
    assert incomplete.status == "fit_time_invalid" and incomplete.records == tuple(full[:-1])
    with pytest.raises(ep.ProtocolError, match="needs complete selection records"):
        dataclasses.replace(incomplete, status="valid_optimal")
    ok = ep.classify_fold_fit(full, arm="glass", protocol=protocol)
    with pytest.raises(ep.ProtocolError, match="not exactly the required selections"):
        dataclasses.replace(ok, records=tuple(full[:-1]))
    with pytest.raises(ep.ProtocolError, match="no fit status"):
        dataclasses.replace(ok, by_key=MappingProxyType({}))
    with pytest.raises(ep.ProtocolError, match="most severe"):
        dataclasses.replace(ok, status="valid_unproven_documented")
    with pytest.raises(ep.ProtocolError, match="needs complete selection records"):
        ep.FoldFit("glass", "valid_optimal", ok.completeness, ok.by_key)     # no records
    greedy = _records(contract, "glass", protocol, pass2=_fit2(contract, OUTCOME_FEASIBLE_GREEDY))
    unreviewed = ep.classify_fold_fit(greedy, arm="glass", protocol=protocol)
    relabelled = {**unreviewed.by_key, "pass2": "fallback_accepted_after_review"}
    forged = dataclasses.replace(unreviewed, status="fallback_accepted_after_review",
                                 by_key=MappingProxyType(relabelled))
    with pytest.raises(ep.ProtocolError, match="does not reproduce"):
        ep.FoldLedger.build({"glass": {0: forged}}, protocol)            # relabelled, consistent
    reviewed = _fold_fit(contract, protocol, status="fallback_accepted_after_review")
    assert ep.FoldLedger.build({"glass": {0: reviewed}}, protocol).accepted_folds("glass") == (0,)
    with pytest.raises(ep.ProtocolError, match="does not reproduce"):
        ep.FoldLedger.build({"glass": {0: dataclasses.replace(
            reviewed, reviews=MappingProxyType({}))}}, protocol)         # the review removed
    three = ep.EvaluationProtocol.from_dict(_record(protocol,
                                                    f17_primary__glass__pass1_oof_folds=3))
    fit3 = ep.classify_fold_fit(_records(contract, "glass", three), arm="glass", protocol=three)
    assert fit3.status == "valid_optimal"
    with pytest.raises(ep.ProtocolError, match="does not reproduce"):
        ep.FoldLedger.build({"glass": {0: fit3}}, protocol)              # other F17 settings
    with pytest.raises(ep.ProtocolError, match="cannot be re-classified"):
        ep.FoldLedger.build({"glass": {0: dataclasses.replace(
            ok, selection_reports={"pass1": 1})}}, protocol)


def test_pooling_never_drops_a_fold_silently(protocol, contract):
    a, b, y, fold = _five_fold_world()
    p = _paired(a, b, y, fold)
    fit = _fold_fit(contract, protocol)
    full = ep.FoldLedger.build({"glass": {k: fit for k in range(5)}}, protocol)
    obs = ep.pooled_observation(p, "glass", full, contract)
    assert obs["primary_eligible"] and obs["pool_label"] == "all attempted folds"
    assert obs["pooled"].counts == RoutingCounts.from_decisions(
        np.where(a == P1, "pass1", np.where(a == P2, "pass2", "uncertain")), y)
    part = ep.FoldLedger.build({"glass": {**{k: fit for k in range(5)},
                                          3: "fallback_unreviewed"}}, protocol)
    with pytest.raises(ep.ProtocolError, match="4 of 5 folds accepted"):
        ep.pooled_observation(p, "glass", part, contract)
    partial = ep.pooled_observation(p, "glass", part, contract, allow_partial=True)
    assert not partial["primary_eligible"]
    assert partial["pool_label"] == "accepted folds only (4 of 5)"
    assert set(partial["per_fold"]) == {0, 1, 2, 3, 4}       # every fold still reported
    assert partial["pooled"].counts.rows == 800 and partial["fold_status"][3] == \
        "fallback_unreviewed"


def _result(contract, p1_rows, p1_pos, positives=200, rows=2000):
    counts = RoutingCounts(rows=rows, positives=positives, pass1_rows=p1_rows,
                           pass1_positives=p1_pos, pass2_rows=0, pass2_positives=0)
    return observe_routing(counts, contract)


def test_shortfall_classification_keeps_every_violation_visible(protocol, contract):
    sat = _result(contract, 100, 5).result("pass1_npv_floor")          # 95/100: exactly 0.95
    assert sat.status == SATISFIED and sat.shortfall == 0
    o = ep.classify_observation(sat, fit_status="valid_optimal", protocol=protocol)
    assert (o.classification, o.material, o.low_support) == ("valid_fit_observed_satisfied",
                                                             None, False)
    small = _result(contract, 100, 6).result("pass1_npv_floor")        # 0.94: 1 pp short
    o = ep.classify_observation(small, fit_status="valid_optimal", protocol=protocol)
    assert o.observed_status == VIOLATED and o.classification == "valid_fit_outer_oof_shortfall"
    assert o.tolerance == pytest.approx(2 * np.sqrt(0.95 * 0.05 / 100)) and o.material is False
    assert o.materiality_rule_status == "proposed"
    big = _result(contract, 100, 12).result("pass1_npv_floor")         # 0.88
    assert ep.classify_observation(big, fit_status="valid_optimal", protocol=protocol).material
    undefined = _result(contract, 0, 0).result("pass1_npv_floor")
    o = ep.classify_observation(undefined, fit_status="valid_optimal", protocol=protocol)
    assert (o.observed_status, o.classification) == (UNDEFINED_STATUS, "undefined_evaluation")
    low = _result(contract, 20, 2).result("pass1_npv_floor")
    o = ep.classify_observation(low, fit_status="valid_optimal", protocol=protocol)
    assert o.low_support and o.observed_status == VIOLATED
    o = ep.classify_observation(big, fit_status="no_feasible_solution", protocol=protocol)
    assert o.classification == "no_feasible_solution" and o.observed_status == VIOLATED
    with pytest.raises(ep.ProtocolError, match="unknown fit status"):
        ep.classify_observation(big, fit_status="ok", protocol=protocol)


def test_materiality_tolerance_is_undefined_for_an_empty_denominator():
    assert ep.materiality_tolerance("19/20", 0, 2) is None
    assert ep.materiality_tolerance("3/10", 75, 2) == pytest.approx(2 * np.sqrt(0.21 / 75))


# ============================================================================
# 6. F31, acceptance, evidence, inputs, adoption
# ============================================================================
def test_the_f31_strip_trigger_is_one_twentieth(protocol):
    tab = ep.f31_strip_review({0: (4, 80), 1: (3, 80), 2: (0, 0), 3: (0, 50)}, protocol)
    tab = tab.set_index("fold")
    assert tab.loc[0, "status"] == "triggered" and tab.loc[0, "review_required"]
    assert tab.loc[1, "status"] == "below_trigger" and not tab.loc[1, "review_required"]
    assert tab.loc[2, "status"] == "undefined_empty_band" and tab.loc[2, "review_required"]
    assert tab.loc[3, "share"] == 0.0
    with pytest.raises(ep.ProtocolError, match="invalid strip"):
        ep.f31_strip_review({0: (5, 4)}, protocol)


def _gates(protocol, **override):
    out = []
    for g in protocol["acceptance"]["structural"]:
        status = override.get(g, "pass")
        out.append(ep.GateResult(g, status, "evidence recorded" if status == "pass" else ""))
    return out


def test_structural_acceptance_needs_every_gate_and_a_frozen_protocol(protocol, frozen):
    with pytest.raises(ep.ProtocolError, match="only a frozen protocol"):
        ep.structural_acceptance(_gates(protocol), protocol)
    ok = ep.structural_acceptance(_gates(frozen), frozen)
    assert ok.accepted and not (ok.failed or ok.not_run or ok.missing)
    assert "not evidence of better predictive performance" in ok.note
    assert not ep.structural_acceptance(_gates(frozen, V4="not_run"), frozen).accepted
    assert ep.structural_acceptance(_gates(frozen, V6="fail"), frozen).failed == ("V6",)
    assert ep.structural_acceptance(_gates(frozen)[1:], frozen).missing == ("V1",)
    with pytest.raises(ep.ProtocolError, match="needs recorded evidence"):
        ep.GateResult("V1", "pass", " ")
    with pytest.raises(ep.ProtocolError, match="not a structural gate"):
        ep.structural_acceptance([ep.GateResult("O1", "pass", "x")], frozen)
    with pytest.raises(ep.ProtocolError, match="reported twice"):
        ep.structural_acceptance(_gates(frozen) + _gates(frozen)[:1], frozen)


def test_structural_acceptance_takes_no_performance_input():
    params = list(inspect.signature(ep.structural_acceptance).parameters)
    assert params == ["results", "protocol", "allow_working"]


def test_comparative_report_completeness_is_separate(frozen):
    assert ep.comparative_report_gaps(frozen["acceptance"]["comparative_sections"], frozen) == []
    gaps = ep.comparative_report_gaps(["primary_outcomes"], frozen)
    assert "pre_post" in gaps and "uncertainty" in gaps and "primary_outcomes" not in gaps


def test_evidence_tiers_never_promote_the_exposed_test_split(protocol):
    assert ep.evidence_tier("bank_marketing_test_split", protocol) == "previously_exposed"
    assert ep.evidence_tier("pr37_stage2_pre_correction_v1", protocol) == "tier1"
    assert ep.evidence_tier("pr38_corrected_outer_oof", protocol) == "tier2"
    assert protocol["evidence_tiers"]["tier3_available"] is False
    with pytest.raises(ep.ProtocolError, match="unregistered evidence source"):
        ep.evidence_tier("new_cohort", protocol)


@pytest.mark.parametrize("name", ["test_proba", "y_test", "holdout_labels", "oracle_threshold",
                                  "metrics.Test.precision"])
def test_test_derived_inputs_are_refused(name, protocol):
    with pytest.raises(ep.ProtocolError, match="test-derived inputs are prohibited"):
        ep.check_inputs(["train_proba_oof", name], protocol)


def test_names_merely_containing_the_letters_pass(protocol):
    ep.check_inputs(["attestation", "contest_flag", "train_outputs_oof", "fold_id"], protocol)


def _ablation_world(up, seed=7, gain=(2, 2, 2, 2, 2), extra_pass2=0, move_pass1=0,
                    reallocate=()):
    """Primary vs ablation on the synthetic world's rows, against its verified LR-or-EBM
    union U. The primary routes one unique positive per fold to Pass 2. The ablation
    swaps ``gain[k]`` U-flagged negatives in Pass 2 of fold k for abstained positives U
    misses (a negative gain does the reverse); the Pass 2 row count is unchanged.
    ``reallocate``: (from_fold, to_fold, n) moves n U-flagged positives' worth of Pass 2
    budget between folds, keeping the pooled budget and every unique positive."""
    t = up["trace"]
    rng = np.random.default_rng(seed)
    idx, y, fold = t.index, np.asarray(t.y).astype(int), np.asarray(t.fold_id).astype(int)
    n = len(y)
    ref = np.asarray(up["refs"]["union"].flags, bool)
    unique = (y == 1) & ~ref
    primary = np.full(n, AB, dtype=object)
    primary[(y == 0) & (rng.random(n) < 0.35)] = P1
    for k in range(5):
        primary[np.flatnonzero((fold == k) & unique)[:1]] = P2
        cand = np.flatnonzero((fold == k) & (primary == AB) & ~unique)
        primary[rng.choice(cand, 59, replace=False)] = P2
    ablation = primary.copy()
    for k in range(5):
        g = abs(gain[k])
        if gain[k] >= 0:            # swap flagged negatives in Pass 2 for unflagged positives
            drop = np.flatnonzero((fold == k) & (primary == P2) & ref & (y == 0))[:g]
            add = np.flatnonzero((fold == k) & (primary == AB) & unique)[:g]
        else:                       # lose unique positives for flagged negatives
            drop = np.flatnonzero((fold == k) & (primary == P2) & unique)[:g]
            add = np.flatnonzero((fold == k) & (primary == AB) & ref & (y == 0))[:g]
        assert len(drop) == len(add) == g
        ablation[drop], ablation[add] = AB, P2
    for src, dst, m in reallocate:
        drop = np.flatnonzero((fold == src) & (ablation == P2) & ref & (y == 1))[:m]
        add = np.flatnonzero((fold == dst) & (ablation == AB) & ref & (y == 1))[:m]
        assert len(drop) == len(add) == m
        ablation[drop], ablation[add] = AB, P2
    if extra_pass2:
        ablation[np.flatnonzero((ablation == AB) & ~unique)[:extra_pass2]] = P2
    if move_pass1:
        ablation[np.flatnonzero(primary == P1)[:move_pass1]] = AB
    return ep.PairedRouting.build("ablation", idx, y, fold,
                                  {"ablation": ablation, "primary": primary})


@pytest.fixture(scope="module")
def adopt_up(world, protocol, tmp_path_factory):
    """Upstream files and a protocol pinning them, shared by the adoption tests."""
    return _pinned_upstream(tmp_path_factory.mktemp("adoption_upstream"), world, protocol)


@pytest.fixture(scope="module")
def afrozen(adopt_up):
    return ep.EvaluationProtocol.from_dict(_frozen_record(adopt_up["protocol"]))


def _registered(up, protocol, **world_kw):
    """A paired ablation world, its verified reference and the registered run."""
    paired = _ablation_world(up, **world_kw)
    reference = ep.verify_reference(up["sources"], protocol, population=paired,
                                    complete=protocol.status == "frozen").flags
    run = ep.run_registered_bootstrap(paired, ep.ADOPTION_METRICS, protocol, reference=reference)
    return dict(paired=paired, reference=reference, run=run, upstream=up["sources"])


@pytest.fixture(scope="module")
def ablation(adopt_up, afrozen):
    return _registered(adopt_up, afrozen)


def _evidence(run, **changes):
    base = dict(sources=("pr38_corrected_outer_oof",), uncertainty=run,
                ablation_changes_pass1=False, pass1_rules_identical_every_fold=True,
                fit_compliance_every_fold=True, new_material_violation_folds=0,
                worse_fit_status_folds=0, runtime_ratio=1.8, complexity_documented=True,
                rf_symmetry="asymmetry_documented")
    base.update(changes)
    return ep.AdoptionEvidence(**base)


def _adopt(protocol, a, evidence=None, **kw):
    kw.setdefault("upstream", a["upstream"])
    return ep.evaluate_adoption(evidence or _evidence(a["run"]), protocol, paired=a["paired"],
                                **kw)


def test_adoption_requires_a_frozen_protocol_and_every_criterion(protocol, afrozen, ablation,
                                                                 monkeypatch):
    with pytest.raises(ep.ProtocolError, match="only a frozen protocol"):
        _adopt(protocol, ablation)
    ok = _adopt(afrozen, ablation)
    assert ok.decision == "adopt_candidate" and set(ok.criteria.values()) == {"pass"}, ok.criteria
    assert ok.reasons["C6_uncertainty"].startswith("reproduced exactly")
    assert ok.evidence_label.startswith("exploratory")
    assert not ok.official                       # a synthetic frozen record is never official
    assert ok.note.startswith("not an official decision: the protocol is not the pinned frozen")
    monkeypatch.setattr(ep, "FROZEN_PROTOCOL_FINGERPRINT", afrozen.fingerprint())
    pinned = _adopt(afrozen, ablation)                     # as if it were the pinned record
    assert pinned.official and pinned.criteria == ok.criteria
    assert "later, versioned protocol" in pinned.note and "not an official" not in pinned.note


def test_the_reference_is_the_verified_upstream_union(afrozen, ablation, adopt_up):
    ok = _adopt(afrozen, ablation)
    union = adopt_up["refs"]["union"]
    assert ok.provenance["reference"] == "or"
    assert ok.provenance["reference_fingerprint"] == ep.flag_fingerprint(union.index, union.flags)
    assert ok.provenance["upstream_identity"] == "verified, complete"
    assert ok.provenance["protocol_fingerprint"] == afrozen.fingerprint()
    verified = ep.verify_reference(adopt_up["sources"], afrozen, population=ablation["paired"])
    assert verified.complete and verified.flags.fingerprint == ablation["reference"].fingerprint
    rec = _frozen_record(adopt_up["protocol"], complementarity__reference="and")
    inter = ep.verify_reference(adopt_up["sources"], ep.EvaluationProtocol.from_dict(rec),
                                population=ablation["paired"]).flags
    assert np.array_equal(inter.flags, adopt_up["refs"]["intersection"].flags)
    assert ep.validate_protocol_record(_record(adopt_up["protocol"],
                                               complementarity__reference="lr"))


def test_adoption_needs_verified_upstream_reference_provenance(afrozen, ablation, adopt_up):
    for bad in (None, ablation["reference"], adopt_up["refs"]):
        with pytest.raises(ep.ProtocolError, match="needs UpstreamSources"):
            _adopt(afrozen, ablation, upstream=bad)
    refs = adopt_up["refs"]
    a, b = _count_preserving_swap(refs)
    flags = np.array(refs["union"].flags, copy=True)
    flags[[a, b]] = flags[[b, a]]            # union count kept; rows moved
    moved = ep.UpstreamSources({**refs, "union": dataclasses.replace(refs["union"], flags=flags)},
                               adopt_up["lr"], adopt_up["ebm"])
    with pytest.raises(ep.ProtocolError, match="(?s)not verified.*or flags equal the registered"):
        _adopt(afrozen, ablation, upstream=moved)
    other = ep.UpstreamSources(refs, adopt_up["ebm"], adopt_up["ebm"])       # wrong view
    with pytest.raises(ep.ProtocolError, match="not verified"):
        _adopt(afrozen, ablation, upstream=other)
    wrong_fp = ep.EvaluationProtocol.from_dict(_frozen_record(
        adopt_up["protocol"], upstream__flag_fingerprints={"lr.in_stage_f2_cv": "0" * 16,
                                                           "ebm.nested_f2": "0" * 16}))
    with pytest.raises(ep.ProtocolError, match="row-aligned flag fingerprint"):
        _adopt(wrong_fp, ablation)


def test_a_changed_upstream_artifact_blocks_adoption(upstream):
    frozen = ep.EvaluationProtocol.from_dict(_frozen_record(upstream["protocol"]))
    a = _registered(upstream, frozen)
    assert _adopt(frozen, a).decision == "adopt_candidate"
    import joblib
    joblib.dump(upstream["w"]["lr"], upstream["paths"]["lr"], compress=3)   # new bytes
    with pytest.raises(ep.ProtocolError, match="lr artifact SHA-256 now"):
        _adopt(frozen, a)


def test_a_working_protocol_gives_no_official_decision(adopt_up):
    working = ep.EvaluationProtocol.from_dict(_record(
        adopt_up["protocol"], upstream__flag_fingerprints={"lr.in_stage_f2_cv": None,
                                                           "ebm.nested_f2": None}))
    a = _registered(adopt_up, working)
    with pytest.raises(ep.ProtocolError, match="only a frozen protocol"):
        _adopt(working, a)
    out = _adopt(working, a, allow_working=True)
    assert out.decision == "adopt_candidate" and not out.official
    assert out.note.startswith("not an official decision: the protocol is working and the "
                               "upstream identity is incomplete (D-C10)")
    assert out.provenance["upstream_identity"].startswith("verified; pending:")
    assert out.provenance["protocol_status"] == "working"


def test_the_registered_run_records_its_procedure_and_settings(afrozen, ablation):
    run = ablation["run"]
    prov = run.provenance
    assert prov["procedure_id"] == ep.BOOTSTRAP_PROCEDURE_ID
    assert (prov["replicates"], prov["seed"], prov["level"], tuple(prov["strata"])) == (
        2000, 42, 0.95, ("outer_fold", "label"))
    assert prov["protocol_fingerprint"] == afrozen.fingerprint()
    assert prov["data_digest"] == ablation["paired"].digest()
    assert run.verify() and set(run.fold_points["unique_tp"]) == {0, 1, 2, 3, 4}
    again = ep.run_registered_bootstrap(ablation["paired"], ep.ADOPTION_METRICS, afrozen,
                                        reference=ablation["reference"])
    assert again == run                                      # deterministic, same digest


def _forge(run, *, rows=None, provenance=None, redigest=True):
    rows = rows if rows is not None else run.rows
    prov = provenance if provenance is not None else run.provenance
    digest = ep.BootstrapRun.compute_digest(rows, run.fold_points, prov) if redigest else run.digest
    return ep.BootstrapRun(rows, run.fold_points, prov, digest)


def _edited_rows(run, metric, **values):
    return tuple(ep._freeze({**dict(r), **values}) if r["metric"] == metric else r
                 for r in run.rows)


def test_c6_fails_without_demonstrable_use_of_the_registered_procedure(afrozen, ablation):
    run = ablation["run"]
    cases = {
        "missing": (None, "not_evaluable"),
        "not a run": (run.table(), "fail"),
        "edited after it was produced": (
            _forge(run, rows=_edited_rows(run, "unique_tp", difference_low=99.0),
                   redigest=False), "fail"),
        "re-digested but not reproducible": (
            _forge(run, rows=_edited_rows(run, "unique_tp", difference_low=99.0)), "fail"),
        "unsupported procedure": (
            _forge(run, provenance=ep._freeze({**ep._thaw(run.provenance),
                                               "procedure_id": "my_bootstrap/1"})), "fail"),
        "other seed": (_forge(run, provenance=ep._freeze({**ep._thaw(run.provenance),
                                                          "seed": 7})), "fail"),
    }
    for label, (reported, verdict) in cases.items():
        out = _adopt(afrozen, ablation, _evidence(reported))
        assert out.criteria["C6_uncertainty"] == verdict, label
        assert out.decision == "retain_primary", label


def _malformed_runs(run):
    prov, points = ep._thaw(run.provenance), ep._thaw(run.fold_points)
    last = dict(run.rows[-1])
    return {
        "no provenance": (dataclasses.replace(run, provenance=None), "provenance is not"),
        "provenance without its data digest": (dataclasses.replace(run, provenance=ep._freeze(
            {k: v for k, v in prov.items() if k != "data_digest"})), "lacks ['data_digest']"),
        "rows not a sequence": (dataclasses.replace(run, rows=None), "rows are not"),
        "a row that is not a mapping": (dataclasses.replace(
            run, rows=run.rows[:-1] + ("pass1_negative_recall",)), "rows are not"),
        "a row without its metric": (dataclasses.replace(run, rows=run.rows[:-1] + (ep._freeze(
            {k: v for k, v in last.items() if k != "metric"}),)), "rows are not"),
        "duplicated rows": (dataclasses.replace(run, rows=run.rows + run.rows[:1]),
                            "duplicated rows"),
        "a listed metric without its row": (dataclasses.replace(run, rows=run.rows[1:]),
                                            "metrics without rows"),
        "no per-fold points": (dataclasses.replace(run, fold_points=None), "per-fold points"),
        "per-fold points without a metric": (dataclasses.replace(run, fold_points=ep._freeze(
            {k: v for k, v in points.items() if k != "pass2_rows"})), "per-fold points"),
        "a per-fold point that is not a mapping": (dataclasses.replace(run, fold_points=ep._freeze(
            {**points, "unique_tp": {0: 1.0}})), "per-fold points"),
        "no digest": (dataclasses.replace(run, digest=None), "digest"),
        "a truncated digest": (dataclasses.replace(run, digest=run.digest[:16]), "digest"),
        "metrics that are not a list": (_forge(run, provenance=ep._freeze(
            {**prov, "metrics": 5})), "could not be checked"),
        "names that are not a list": (_forge(run, provenance=ep._freeze(
            {**prov, "names": 5})), "could not be checked"),
    }


def test_a_malformed_or_incomplete_run_fails_c6_and_never_raises(afrozen, ablation):
    for label, (reported, why) in _malformed_runs(ablation["run"]).items():
        out = _adopt(afrozen, ablation, _evidence(reported))
        assert out.criteria["C6_uncertainty"] == "fail", label
        assert why in out.reasons["C6_uncertainty"], (label, out.reasons["C6_uncertainty"])
        assert out.decision == "retain_primary", label
    assert _adopt(afrozen, ablation).criteria["C6_uncertainty"] == "pass"   # the intact run


def test_c6_fails_for_a_run_under_other_settings_data_or_metrics(afrozen, ablation, adopt_up):
    lighter = ep.EvaluationProtocol.from_dict(_frozen_record(adopt_up["protocol"],
                                                             uncertainty__replicates=1000))
    other_settings = ep.run_registered_bootstrap(ablation["paired"], ep.ADOPTION_METRICS,
                                                 lighter, reference=ablation["reference"])
    other_data = _registered(adopt_up, afrozen, seed=8, gain=(1, 1, 1, 1, 1))["run"]
    fewer = ep.run_registered_bootstrap(ablation["paired"], ["unique_tp", "coverage"], afrozen,
                                        reference=ablation["reference"])
    other_reference = ep.run_registered_bootstrap(
        ablation["paired"], ep.ADOPTION_METRICS, afrozen,
        reference=ep.ReferenceFlags.from_upstream(adopt_up["refs"]["intersection"]))
    for run in (other_settings, other_data, fewer, other_reference):
        out = _adopt(afrozen, ablation, _evidence(run))
        assert out.criteria["C6_uncertainty"] == "fail" and out.decision == "retain_primary"
    assert "replicates" in _adopt(afrozen, ablation, _evidence(other_settings)).reasons[
        "C6_uncertainty"]
    assert "reference flags" in _adopt(afrozen, ablation, _evidence(other_reference)).reasons[
        "C6_uncertainty"]


def test_adoption_needs_the_paired_ablation_data(afrozen, ablation):
    cross = _paired(*_five_fold_world()[:2], *_five_fold_world()[2:])
    with pytest.raises(ep.ProtocolError, match="paired ablation routing"):
        _adopt(afrozen, {**ablation, "paired": cross})
    with pytest.raises(TypeError):
        _adopt(afrozen, ablation, reference=ablation["reference"])   # never taken from the caller
    with pytest.raises(ep.ProtocolError, match="names its routings"):
        ep.PairedRouting.build("ablation", ablation["paired"].index, ablation["paired"].y,
                               ablation["paired"].fold_id,
                               {"primary": ablation["paired"].states["primary"],
                                "ablation": ablation["paired"].states["ablation"]})


@pytest.mark.parametrize("world_kw, evidence_kw, criterion", [
    (dict(extra_pass2=5), {}, "C1_unique_tp_gain"),                    # flag budget inflated
    (dict(gain=(1, 0, 0, 0, 0)), {}, "C1_unique_tp_gain"),             # gain too small
    (dict(gain=(3, 3, 3, -1, -1)), {}, "C5_fold_reproducibility"),
    (dict(move_pass1=3), {}, "C2_pass1_leakage"),                      # Pass 1 not identical
    ({}, dict(pass1_rules_identical_every_fold=False), "C2_pass1_leakage"),
    ({}, dict(new_material_violation_folds=1), "C3_obligations"),
    ({}, dict(runtime_ratio=3.5), "C7_solver"),
    ({}, dict(worse_fit_status_folds=1), "C7_solver"),
    ({}, dict(rf_symmetry=None), "C9_complexity"),
])
def test_any_failed_or_missing_criterion_keeps_the_primary(afrozen, adopt_up, world_kw,
                                                           evidence_kw, criterion):
    a = _registered(adopt_up, afrozen, **world_kw)
    out = _adopt(afrozen, a, _evidence(a["run"], **evidence_kw))
    assert out.decision == "retain_primary"
    assert out.criteria[criterion] in ("fail", "not_evaluable"), out.criteria
    assert out.criteria["C6_uncertainty"] == "pass"


_C5_PER_FOLD = ("complementarity__criteria__C5_fold_reproducibility__"
                "fold_counts_only_without_pass2_budget_inflation")


def test_the_pass2_budget_is_held_pooled_and_counted_per_fold(afrozen, adopt_up):
    """Pooled non-inflation guards the pooled C1 claim; a fold whose own Pass 2 budget
    grew does not count toward C5, so reallocation between folds cannot be masked."""
    one = _registered(adopt_up, afrozen, reallocate=((1, 0, 5),))
    p = one["paired"]
    assert p.counts("ablation").pass2_rows == p.counts("primary").pass2_rows
    out = _adopt(afrozen, one)
    assert out.decision == "adopt_candidate", out.criteria
    assert "budget inflated in folds [0]" in out.reasons["C5_fold_reproducibility"]
    assert out.reasons["C1_unique_tp_gain"] == "pooled Pass 2 rows: ablation 300 vs primary 300"
    two = _registered(adopt_up, afrozen, reallocate=((1, 0, 5), (3, 2, 5)))
    out = _adopt(afrozen, two)
    assert out.criteria["C1_unique_tp_gain"] == "pass"             # the pooled budget is kept
    assert out.criteria["C5_fold_reproducibility"] == "fail" and out.decision == "retain_primary"
    assert "budget inflated in folds [0, 2]" in out.reasons["C5_fold_reproducibility"]
    pooled_only = ep.EvaluationProtocol.from_dict(_frozen_record(adopt_up["protocol"],
                                                                 **{_C5_PER_FOLD: False}))
    masked = _registered(adopt_up, pooled_only, reallocate=((1, 0, 5), (3, 2, 5)))
    assert _adopt(pooled_only, masked).criteria["C5_fold_reproducibility"] == "pass"
    more = _registered(adopt_up, afrozen, extra_pass2=1)
    assert _adopt(afrozen, more).criteria["C1_unique_tp_gain"] == "fail"   # pooled budget grew
    assert ep.validate_protocol_record(_record(adopt_up["protocol"], **{_C5_PER_FOLD: "yes"}))


_EDGE = dict(unique_tp_difference=10.0, unique_tp_difference_low=0.5, primary_unique_tp=200.0,
             ablation_pass2_rows=900.0, primary_pass2_rows=900.0, coverage_difference_low=-0.01,
             leakage_difference_high=0.005, pass2_negative_routing_difference_high=0.005,
             pass1_negative_recall_difference_low=-0.01, folds_with_nonnegative_unique_tp_gain=4,
             folds_with_nonnegative_unique_tp_gain_without_budget_inflation=4,
             pass1_routing_identical=True)


def test_adoption_margins_are_inclusive_at_their_boundaries(afrozen, ablation):
    cr = afrozen["complementarity"]["criteria"]
    e = _evidence(ablation["run"], runtime_ratio=3)
    assert set(ep._adoption_criteria(_EDGE, e, cr, "pass").values()) == {"pass"}
    moved = ep._adoption_criteria(_EDGE, _evidence(ablation["run"], ablation_changes_pass1=True,
                                                   runtime_ratio=3), cr, "pass")
    assert moved["C2_pass1_leakage"] == "pass"
    beyond = {"unique_tp_difference": 9.99, "unique_tp_difference_low": 0.0,
              "ablation_pass2_rows": 901.0, "coverage_difference_low": -0.0101,
              "pass2_negative_routing_difference_high": 0.0051,
              "pass1_negative_recall_difference_low": -0.0101,
              "folds_with_nonnegative_unique_tp_gain_without_budget_inflation": 3}
    for key, value in beyond.items():
        out = ep._adoption_criteria({**_EDGE, key: value}, e, cr, "pass")
        assert "fail" in out.values(), key
    leak = ep._adoption_criteria({**_EDGE, "leakage_difference_high": 0.0051}, _evidence(
        ablation["run"], ablation_changes_pass1=True, runtime_ratio=3), cr, "pass")
    assert leak["C2_pass1_leakage"] == "fail"


def test_test_derived_evidence_may_not_inform_adoption(afrozen, ablation):
    with pytest.raises(ep.ProtocolError, match="bank_marketing_test_split"):
        _adopt(afrozen, ablation, _evidence(ablation["run"],
                                            sources=("bank_marketing_test_split",)))
    rec = afrozen.to_dict()
    rec["evidence_tiers"]["sources"]["pr35_constraint_results"] = "previously_exposed"
    exposed = ep.EvaluationProtocol.from_dict(rec)
    with pytest.raises(ep.ProtocolError, match="previously_exposed: it may not inform adoption"):
        _adopt(exposed, ablation, _evidence(ablation["run"], sources=("pr35_constraint_results",)))
    with pytest.raises(ep.ProtocolError, match="unregistered"):
        _adopt(afrozen, ablation, _evidence(ablation["run"], sources=("ablation_notes",)))


# ============================================================================
# 7. matched budgets reuse the Step 2C selection
# ============================================================================
def test_matched_selection_takes_no_labels():
    assert "y" not in inspect.signature(matched_rate_cut).parameters
    assert list(inspect.signature(contrib._top_k).parameters) == ["score", "pop", "k", "key"]


def test_matched_ties_are_broken_by_row_identity_not_row_order():
    index = pd.Index([f"row{i}" for i in range(40)])
    score = np.repeat([0.9, 0.5, 0.1], [5, 20, 15])            # K = 12 cuts inside the 0.5 tie
    pop = np.ones(40, bool)
    chosen = contrib._top_k(score, pop, 12, contrib._tie_key(index))
    assert int(chosen.sum()) == 12 and chosen[:5].all()
    perm = np.random.default_rng(0).permutation(40)
    again = contrib._top_k(score[perm], pop[perm], 12, contrib._tie_key(index[perm]))
    assert set(index[perm][again]) == set(index[chosen])
    cut = matched_rate_cut(score, target_count=12, ranking="highest")
    assert not cut.exact and cut.strict_count == 5 and cut.inclusive_count == 25
    assert (np.asarray(cut.strict) <= chosen).all() and (chosen <= np.asarray(cut.inclusive)).all()


def test_matched_reference_selection_is_label_free_on_the_synthetic_trace(world):
    t, lr = world["trace"], world["lr"]
    sel = d.matched_reference_selection(t, "glass", lr)
    flags = np.asarray(sel.flags)
    common = np.asarray(d.common_eligible_mask(t))
    for k in range(t.n_folds):
        in_fold = np.asarray(t.fold_id) == k
        assert int((flags & in_fold).sum()) == int((np.asarray(t.views["glass"].pass2) & common
                                                    & in_fold).sum())
    assert not (flags & ~common).any()
    again = d.matched_reference_selection(t, "glass", lr)
    assert np.array_equal(np.asarray(again.flags), flags)
