"""
PR 37 Commit 1 (Step 1C): the train-only boundary of shared.stage2.diagnostics.

Every artifact is passed with its test-side fields trapped
(``diagnostics_world.trapped``): a read of any of them fails the test. Each
mandatory check is shown to fail loudly on the mutation it guards against.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from shared.stage2 import diagnostics as d
from shared.stage2.diagnostics import boundary
from shared.stage_io import fold_assignment, label_fingerprint
from tests.support import diagnostics_world as dw
from tests.support.mutations import permute_labels


@pytest.fixture
def w():
    return dw.world()


@pytest.fixture
def ref(w):
    return dw.reference(w)


def glass_view(w, ref):
    return d.extract_router_view(dw.trapped(w["glass"]), ref, expected_arm="glass")


def rf_view(w, ref):
    return d.extract_router_view(dw.trapped(w["rf"]), ref, expected_arm="blackbox")


# ======================================================================
# train reference
# ======================================================================
class NoYTest(dict):
    """GLOBAL_SPLIT whose y_test raises when read."""

    def __getitem__(self, key):
        if key == "y_test":
            raise dw.ForbiddenRead("read GLOBAL_SPLIT['y_test']")
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def status_of(view) -> dict:
    frame = view.provenance_frame()
    return dict(zip(frame["name"], frame["status"]))


def test_reference_from_global_split_never_reads_y_test(w):
    ref = d.train_reference_from_global_split(NoYTest(w["GS"]))
    assert ref.n_train == len(w["idx_tr"]) and ref.index_train.equals(w["idx_tr"])
    np.testing.assert_array_equal(ref.y_train, w["y"])


def test_reference_identities(w, ref):
    from blackbox_pipeline.models.rf_router import split_fingerprint
    assert ref.router_split_fingerprint == split_fingerprint(w["GS"]["X_train"], w["GS"]["X_test"])
    assert ref.train_label_fingerprint == label_fingerprint(w["idx_tr"], w["y"])
    assert ref.stage_io_split_fingerprint == w["fp_io"] and ref.stage_io_attestation == dw.ATTESTATION
    bare = dw.reference(w, attested=False)
    assert bare.stage_io_split_fingerprint is None and bare.stage_io_attestation is None


@pytest.mark.parametrize("kw, match", [
    ({"attested_stage_io_split_fingerprint": "D6CDDADC3439", "attestation_source": "x"}, "12 lowercase"),
    ({"attested_stage_io_split_fingerprint": "d6cddadc3439"}, "attestation_source"),
])
def test_reference_attestation_is_explicit(w, kw, match):
    with pytest.raises(d.RouterDiagnosticError, match=match):
        d.build_train_reference(index_train=w["idx_tr"], y_train=w["GS"]["y_train"],
                                index_test=w["idx_te"], **kw)


@pytest.mark.parametrize("case, match", [
    ("overlap", "overlap"), ("labels", "binary"), ("length", "labels for"),
    ("series_index", "y_train.index"),
])
def test_reference_refuses_malformed_inputs(w, case, match):
    idx_tr, idx_te, y = w["idx_tr"], w["idx_te"], w["GS"]["y_train"]
    if case == "overlap":
        idx_te = idx_te.append(idx_tr[:1])
    elif case == "labels":
        y = y.replace(1, 2)
    elif case == "length":
        y = y.iloc[1:].to_numpy()
    else:
        y = y.reset_index(drop=True)
    with pytest.raises(d.RouterDiagnosticError, match=match):
        d.build_train_reference(index_train=idx_tr, y_train=y, index_test=idx_te)


# ======================================================================
# the allowlist
# ======================================================================
class Untouchable:
    def __getattr__(self, name):
        raise AssertionError("touched")

    def __contains__(self, key):
        raise AssertionError("touched")


@pytest.mark.parametrize("path", [
    ("test_proba",), ("extra", "test_decisions"), ("threshold", "test_oracle"),
    ("refit", "train_proba"), ("fold_mean",), ("metrics_test",), ("glass_router_data",),
    ("oof_protocol", "test_outputs"), ("holdout",), ("metrics",),
])
def test_test_side_paths_are_refused_before_anything_is_read(path):
    with pytest.raises(d.RouterDiagnosticError, match="not train-side"):
        boundary._take(Untouchable(), path)


def test_extractors_read_only_allowlisted_fields(w, ref):
    _, glass, rf, lr, ebm = dw.views(w, ref)          # every test-side field trapped
    for v in (glass, rf, lr, ebm):
        for path in v.source["fields_read"]:
            assert not any(t in path.lower() for t in d.FORBIDDEN_FIELD_TOKENS), path


# ======================================================================
# routers: happy path
# ======================================================================
def test_glass_view(w, ref):
    v = glass_view(w, ref)
    assert (v.arm, v.model_family, v.router_mode) == ("glass", "glass_router", None)
    assert dict(v.pass2_population) == {"cross_fit": "full", "full_train": "full",
                                        "normalised": "full"}
    np.testing.assert_array_equal(v.source_decision, w["glass"]["train_decisions"])
    np.testing.assert_array_equal(v.router_score.values, w["glass"]["train_proba"][:, 1])
    assert v.router_score.space == "router_transform" and v.fold_thresholds is None
    status = status_of(v)
    assert status["stage_io_split_fingerprint"] == d.ATTESTED
    assert status["router_transform"] == status["train_label_fingerprint.full_train"] == d.VALIDATED
    assert status["fold_rule_sets"] == d.NOT_RECORDED
    assert v.metadata["n_rules"] == {"Pass 1": 2, "Pass 2": 1}


def test_rf_view(w, ref):
    v = rf_view(w, ref)
    assert (v.arm, v.model_family, v.router_mode) == ("blackbox", "rf_router", "two_pass")
    assert v.pass2_population["normalised"] == "oof_remainder"
    assert dict(v.metadata["rf_params"]) == dw.RF_PARAMS
    assert v.metadata["constraint_source"] == "match_glass_oof"
    assert sorted(v.fold_thresholds) == list(range(5))
    for row in w["rf"]["extra"]["oof_fold_summary"]:
        assert v.fold_thresholds[row["fold"]]["t1"] == row["t1"]


def test_unattested_reference_marks_stage_io_not_recorded(w):
    v = glass_view(w, dw.reference(w, attested=False))
    status = status_of(v)
    assert status["stage_io_split_fingerprint"] == d.NOT_RECORDED


@pytest.mark.parametrize("key, arm", [("glass", "blackbox"), ("rf", "glass")])
def test_router_layout_must_match_the_expected_arm(w, ref, key, arm):
    with pytest.raises(d.RouterDiagnosticError, match="format, not the"):
        d.extract_router_view(dw.trapped(w[key]), ref, expected_arm=arm)


# ======================================================================
# routers: every mandatory check fails loudly
# ======================================================================
def _block(art):
    return art["extra"] if "extra" in art else art


def _set_decision(art, i, token):
    f = _block(art)
    f["train_decisions"] = np.asarray(f["train_decisions"], dtype=object).copy()
    f["train_decisions"][i] = token


def _first(art, token):
    return int(np.flatnonzero(np.asarray(_block(art)["train_decisions"]) == token)[0])


def _full_rate_transform(art, w):
    f = _block(art)
    b = float(w["y"].mean())
    p = dw.transform(np.asarray(f["train_decisions"]), np.asarray(f["train_confidence"]), b)
    f["train_proba"] = np.column_stack([1 - p, p])


def _reorder(art, w):
    f = _block(art)
    key = "train_idx" if "train_idx" in f else "index_train"
    idx = list(f[key])
    idx[0], idx[1] = idx[1], idx[0]
    f[key] = pd.Index(idx)


ROUTER_MUTATIONS = {
    "token_abstain": (lambda a, w: _set_decision(a, _first(a, "uncertain"), "abstain"),
                      "decisions are exactly"),
    "token_none": (lambda a, w: _set_decision(a, 0, None), "decisions are exactly"),
    "token_case": (lambda a, w: _set_decision(a, _first(a, "pass1"), "Pass1"),
                   "decisions are exactly"),
    "pred": (lambda a, w: _block(a)["train_pred"].__setitem__(_first(a, "pass1"), 1), "train_pred"),
    "abstain_conf": (lambda a, w: _block(a)["train_confidence"].__setitem__(
        _first(a, "uncertain"), 1e-9), "0.0 on abstained"),
    "conf_range": (lambda a, w: _block(a)["train_confidence"].__setitem__(
        _first(a, "pass2"), 1.2), "confidence finite"),
    "transform_1e-9": (lambda a, w: _block(a)["train_proba"].__setitem__((0, 1),
                       _block(a)["train_proba"][0, 1] + 1e-9), "route transform"),
    "transform_full_rate": (_full_rate_transform, "route transform"),
    "not_oof": (lambda a, w: _block(a).__setitem__("train_outputs_oof", "yes"), "train_outputs_oof"),
    "index_reset": (lambda a, w: _block(a).__setitem__(
        "train_idx" if "train_idx" in _block(a) else "index_train", pd.RangeIndex(len(w["y"]))),
        "row index"),
    "index_order": (_reorder, "row index"),
    "fold_seed": (lambda a, w: _block(a).__setitem__(
        "train_fold_id", fold_assignment(w["y"], cv_folds=5, random_state=7).to_numpy()),
        "reproduce"),
    "fold_fp": (lambda a, w: _block(a)["oof_protocol"].__setitem__("fold_fingerprint", "0" * 16),
                "fold fingerprint"),
    "n_folds": (lambda a, w: _block(a)["oof_protocol"].__setitem__("n_folds", 10), "declared n_folds"),
    "split_fp": (lambda a, w: a.__setitem__("split_fingerprint", "0" * 16), "router split"),
    "cross_fit_labels": (lambda a, w: _block(a)["oof_protocol"].__setitem__(
        "train_label_fingerprint",
        label_fingerprint(w["idx_tr"], permute_labels(w["GS"]["y_train"], seed=3))),
        "cross-fit label fingerprint"),
    "full_train_labels": (lambda a, w: a.__setitem__("train_label_fingerprint", "abc"),
                          "full-train label fingerprint"),
    "base_rate": (lambda a, w: a.__setitem__("training_base_rate", float(w["y"].mean()) + 1e-6),
                  "training_base_rate"),
    "features": (lambda a, w: a.__setitem__("features" if "features" in a else "feature_names",
                                            ["a", "b"]), "RF_FEATURES_BINARY"),
}
GLASS_ONLY = {
    "pop_mismatch": (lambda a, w: a["fit_report"].__setitem__("pass2_population", "oof_remainder"),
                     "Pass 2 population"),
    "pop_unknown": (lambda a, w: (a["fit_report"].__setitem__("pass2_population", "custom"),
                                  a["oof_protocol"].__setitem__("glass_pass2_population", "custom"),
                                  a["config_full"].__setitem__("pass2_population", "custom")),
                    "Pass 2 population"),
    "pop_config_full": (lambda a, w: a["config_full"].__setitem__("pass2_population", "oof_remainder"),
                        "config_full"),
}
RF_ONLY = {
    "pop_normalised_spelling": (lambda a, w: a["extra"]["oof_protocol"].__setitem__(
        "pass2_population", "oof_remainder"), "exact declaration"),
    "mode": (lambda a, w: a.__setitem__("mode", "single_rf"), "two_pass"),
    "fold_summary_missing": (lambda a, w: a["extra"].__setitem__(
        "oof_fold_summary", a["extra"]["oof_fold_summary"][:4]), "oof_fold_summary"),
    "fold_summary_nan": (lambda a, w: a["extra"]["oof_fold_summary"][2].__setitem__("t1", np.nan),
                         "oof_fold_summary"),
    "protocol_split_fp": (lambda a, w: a["extra"]["oof_protocol"].__setitem__(
        "split_fingerprint", "1" * 16), "oof_protocol split"),
}


@pytest.mark.parametrize("key, arm, name", [
    *[("glass", "glass", n) for n in [*ROUTER_MUTATIONS, *GLASS_ONLY]],
    *[("rf", "blackbox", n) for n in [*ROUTER_MUTATIONS, *RF_ONLY]],
])
def test_router_mutation_fails_loudly(w, ref, key, arm, name):
    mutate, match = {**ROUTER_MUTATIONS, **GLASS_ONLY, **RF_ONLY}[name]
    mutate(w[key], w)
    with pytest.raises(d.BoundaryCheckError, match=match) as exc:
        d.extract_router_view(dw.trapped(w[key]), ref, expected_arm=arm)
    assert isinstance(exc.value.report, pd.DataFrame) and not exc.value.report["passed"].all()


def test_all_failures_are_reported_together(w, ref):
    ROUTER_MUTATIONS["fold_fp"][0](w["glass"], w)
    ROUTER_MUTATIONS["base_rate"][0](w["glass"], w)
    with pytest.raises(d.BoundaryCheckError) as exc:
        glass_view(w, ref)
    assert "2 mandatory check(s) failed" in str(exc.value)


@pytest.mark.parametrize("path", [("train_label_fingerprint",), ("fit_report",),
                                  ("oof_protocol", "fold_fingerprint")])
def test_missing_required_field_is_refused_never_filled_in(w, ref, path):
    target = w["glass"]
    for k in path[:-1]:
        target = target[k]
    del target[path[-1]]
    with pytest.raises(d.RouterDiagnosticError, match="missing required"):
        glass_view(w, ref)


def test_missing_fields_are_listed_together(w, ref):
    del w["rf"]["extra"]["oof_fold_summary"], w["rf"]["training_base_rate"]
    with pytest.raises(d.RouterDiagnosticError) as exc:
        rf_view(w, ref)
    assert "oof_fold_summary" in str(exc.value) and "training_base_rate" in str(exc.value)


# ======================================================================
# files and SHA-256 attestation
# ======================================================================
def test_sha256_attestation(w, ref, tmp_path):
    p = tmp_path / "glass_router.joblib"
    joblib.dump(w["glass"], p)
    sha = d.file_sha256(p)
    v = d.extract_router_view(p, ref, expected_arm="glass", expected_sha256=sha)
    status = status_of(v)
    assert status["file_sha256"] == d.ATTESTED and v.source["path"] == str(p)
    with pytest.raises(d.RouterDiagnosticError, match="not the preserved"):
        d.extract_router_view(p, ref, expected_arm="glass", expected_sha256="0" * 64)
    with pytest.raises(d.RouterDiagnosticError, match="needs a file path"):
        d.extract_router_view(w["glass"], ref, expected_arm="glass", expected_sha256=sha)


# ======================================================================
# Stage 1 LR
# ======================================================================
def lr_view(w, ref):
    return d.extract_lr_view(dw.trapped(w["lr"]), ref)


def _so(w):
    return w["lr"]["stage_output"]


def test_lr_view(w, ref):
    v = lr_view(w, ref)
    assert v.model == "lr" and set(v.scores) == {"calibrated_probability"}
    assert set(v.thresholds) == {"lr.in_stage_f2_cv"}
    rec = v.thresholds["lr.in_stage_f2_cv"]
    assert rec.value == 0.10 and rec.space == "calibrated_probability"
    assert rec.selection == d.schema.SELECTION_GLOBAL_TRAIN_LABELS
    status = status_of(v)
    assert status["stage_io_split_fingerprint"] == d.ATTESTED
    bare = d.extract_lr_view(dw.trapped(w["lr"]), dw.reference(w, attested=False))
    assert status_of(bare)["stage_io_split_fingerprint"] == d.RECORDED_ONLY


def test_lr_other_threshold_source_registers_no_operating_point(w, ref):
    _so(w)["threshold_source"] = "nested_per_fold"
    v = lr_view(w, ref)
    assert dict(v.thresholds) == {}
    with pytest.raises(d.UnsupportedDiagnostic):
        d.resolve_operating_point(v, "lr.in_stage_f2_cv")


LR_MUTATIONS = {
    "source": (lambda so, w: so.__setitem__("threshold_source", "test_oracle"), "THRESHOLD_SOURCES"),
    "provenance": (lambda so, w: so.__setitem__("oof_provenance", "refittable (cross_val_predict)"),
                   "nested"),
    "labels": (lambda so, w: so.__setitem__("y_train", permute_labels(so["y_train"], seed=1)),
               "carried training labels"),
    "features": (lambda so, w: so.__setitem__("feature_names", ["f1", "f2"]), "LR_FEATURES"),
    "seed": (lambda so, w: so.__setitem__("config", {"random_state": 7}), "reproduce"),
    "split": (lambda so, w: so.__setitem__("split_fingerprint", "0" * 12), "attested value"),
    "arm": (lambda so, w: so.__setitem__("arm", "blackbox"), "arm is"),
    "index": (lambda so, w: so.__setitem__("train_proba_oof",
                                           so["train_proba_oof"].reset_index(drop=True)),
              "train_proba_oof index"),
    "threshold": (lambda so, w: so.__setitem__("threshold", 1.0), "threshold finite"),
}


@pytest.mark.parametrize("name", list(LR_MUTATIONS))
def test_lr_mutation_fails_loudly(w, ref, name):
    mutate, match = LR_MUTATIONS[name]
    mutate(_so(w), w)
    with pytest.raises(d.BoundaryCheckError, match=match):
        lr_view(w, ref)


def test_lr_without_stage_output_is_refused(w, ref):
    with pytest.raises(d.RouterDiagnosticError, match="StageOutput"):
        d.extract_lr_view({"model": None}, ref)


# ======================================================================
# Stage 3 EBM
# ======================================================================
def ebm_view(w, ref):
    return d.extract_ebm_view(dw.trapped_stage3(w["ebm"]), ref)


def test_ebm_view(w, ref):
    v = ebm_view(w, ref)
    assert set(v.scores) == {"raw_model_score", "calibrated_probability"}
    assert set(v.thresholds) == {"ebm.global_f2_raw", "ebm.global_f2_calibrated"}
    assert v.thresholds["ebm.global_f2_raw"].space == "raw_model_score"
    assert v.thresholds["ebm.global_f2_calibrated"].space == "calibrated_probability"
    sd = v.stored_decisions["ebm.nested_f2"]
    assert sd.by_fold == dw.EBM_NESTED and sd.selection == d.schema.SELECTION_FOLD_NESTED
    status = status_of(v)
    assert status["split_fingerprint"] == d.ATTESTED and status["nested_decisions"] == d.VALIDATED


def test_ebm_router_scheme_split_fingerprint_is_validated(w, ref):
    w["ebm"].split_fingerprint = w["router_fp"]
    status = status_of(ebm_view(w, ref))
    assert status["split_fingerprint"] == d.VALIDATED


def test_ebm_without_nested_decisions_registers_no_nested_point(w, ref):
    w["ebm"].threshold["oof_nested"]["drives_train_decisions"] = False
    v = ebm_view(w, ref)
    assert dict(v.stored_decisions) == {}
    with pytest.raises(d.UnsupportedDiagnostic, match="drives_train_decisions"):
        d.resolve_operating_point(v, "ebm.nested_f2")


def _flip_decision(a, w):
    a.oof["decision"] = np.asarray(a.oof["decision"]).copy()
    a.oof["decision"][0] = 1 - a.oof["decision"][0]


def _vary_threshold(a, w):
    a.oof["threshold_per_row"] = np.asarray(a.oof["threshold_per_row"]).copy()
    a.oof["threshold_per_row"][0] += 1e-9


EBM_MUTATIONS = {
    "decision": (_flip_decision, "stored decision"),
    "per_row": (_vary_threshold, "constant per fold"),
    "by_fold": (lambda a, w: a.threshold["oof_nested"].__setitem__("by_fold", [0.5] * 5),
                "constant per fold"),
    "family": (lambda a, w: setattr(a, "model_family", "xgboost"), "model_family"),
    "arm": (lambda a, w: a.config.__setitem__("arm", "blackbox"), "config arm"),
    "scope": (lambda a, w: setattr(a, "feature_fit_scope", "global"), "per_fold"),
    "features": (lambda a, w: setattr(a, "features", ["f1"]), "EBM_FEATURES"),
    "labels": (lambda a, w: setattr(a, "y_train", list(1 - np.asarray(a.y_train))),
               "carried training labels"),
    "fold_plan": (lambda a, w: a.fold_plan.__setitem__("random_state", 7), "reproduce"),
    "gates": (lambda a, w: a.calibration.__setitem__("fold_applied", [True] * 4), "gates"),
    "split": (lambda a, w: setattr(a, "split_fingerprint", "0" * 12), "attested value"),
    "raw_nan": (lambda a, w: a.oof["proba"].__setitem__(3, np.nan), "raw OOF"),
}


@pytest.mark.parametrize("name", list(EBM_MUTATIONS))
def test_ebm_mutation_fails_loudly(w, ref, name):
    mutate, match = EBM_MUTATIONS[name]
    mutate(w["ebm"], w)
    with pytest.raises(d.BoundaryCheckError, match=match):
        ebm_view(w, ref)


def test_ebm_dict_artifact_is_refused(w, ref):
    with pytest.raises(d.RouterDiagnosticError, match="Stage3Artifact"):
        d.extract_ebm_view({"oof": {}}, ref)


# ======================================================================
# Stage 4 recall cuts
# ======================================================================
def stage4(w, ref, **kw):
    _, _, _, lr, ebm = dw.views(w, ref)
    return d.extract_stage4_recall_thresholds(dw.trapped(w["meta"]), ref, lr_view=lr,
                                              ebm_view=ebm, **kw)


def test_stage4_recall_cuts_recompute(w, ref):
    s4 = stage4(w, ref)
    assert set(s4.records) == {"lr.stage4_recall70", "ebm.stage4_recall70"}
    assert all(r.space == "calibrated_probability" for r in s4.records.values())
    assert s4.provenance[0].status == d.VALIDATED


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m["thresholds"].__setitem__("lr", m["thresholds"]["lr"] + 1e-4), "lr cut recomputes"),
    (lambda m: m["thresholds"].__setitem__("target_recall", 0.8), "target_recall"),
    (lambda m: m["thresholds"]["sources"].__setitem__("ebm", "manual"), "ebm cut source"),
    (lambda m: m.__setitem__("split_fingerprint", "0" * 12), "attested"),
])
def test_stage4_mutation_fails_loudly(w, ref, mutate, match):
    mutate(w["meta"])
    with pytest.raises(d.BoundaryCheckError, match=match):
        stage4(w, ref)


def test_stage4_without_recall_function_is_recorded_only(w, ref, monkeypatch):
    monkeypatch.setattr(boundary, "_default_recall_fn", lambda: None)
    assert stage4(w, ref).provenance[0].status == d.RECORDED_ONLY


# ======================================================================
# RF Pass 1 candidate
# ======================================================================
def test_rf_candidate_never_reads_the_engineered_test_labels(w, ref):
    cand = d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
    assert cand.random_state == 42 and dict(cand.params) == dw.RF_PARAMS
    assert not any("glass_router_data" in p or "test" in p for p in cand.source["fields_read"])


@pytest.mark.parametrize("mutate, match", [
    (lambda r, w: r.__setitem__("oof_index", pd.RangeIndex(len(w["y"]))), "oof_index"),
    (lambda r, w: r["oof_protocol"].__setitem__("fold_fingerprint", "0" * 16), "fold fingerprint"),
    (lambda r, w: r["oof_protocol"].__setitem__("train_label_fingerprint", "0" * 12), "label"),
    (lambda r, w: r.__setitem__("split_fingerprint", "0" * 16), "split fingerprint"),
    (lambda r, w: r.__setitem__("train_outputs_oof", False), "train_outputs_oof"),
])
def test_rf_candidate_mutation_fails_loudly(w, ref, mutate, match):
    mutate(w["rf_result"], w)
    with pytest.raises(d.BoundaryCheckError, match=match):
        d.extract_rf_pass1_candidate(dw.trapped(w["rf_result"]), ref)
