"""
PR 37 Commit 5: the PR 38 handoff.

1. ``check_glass_primary_ranker`` closes the one gap the Step 5A audit found.
   ``check_glass_primary_config`` covers the ``GlassRouterConfig``; the ranker's
   RF hyperparameters reach ``crossfit_stage2`` and the full-train ranker factory
   as separate arguments, so before Commit 5 a re-tuned or hand-edited value
   (GLASS notebook Cell 12 re-tunes the RF) went unchecked on the strict pathway.
2. The handoff record names the identities the code pins, and the gates and
   requirements PR 38 owes. (The roadmap is maintained by hand; no test pins it.)

Version-controlled records and synthetic values only; nothing is fitted or read
from an artifact.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import shared.stage2.contract as contract_mod
import shared.stage2.evaluation_protocol as ep
from blackbox_pipeline.models.rf_router import profile as rf_profile
from glass_pipeline.glass_router import profile as glass_profile
from shared.stage2.contract import ContractViolation

BANK_ROOT = Path(__file__).resolve().parents[2]
HANDOFF = BANK_ROOT / "documentation/methodology/pr38_handoff.md"
MANIFEST = BANK_ROOT / "research_logs/baselines/pr37_stage2_pre_correction_v1/MANIFEST.json"

def _primary_ranker() -> dict:
    return glass_profile.ranker_settings()


# ======================================================================
# 1. check_glass_primary_ranker
# ======================================================================
def test_the_frozen_ranker_settings_pass_whatever_their_order_or_scalar_types():
    ranker = _primary_ranker()
    glass_profile.check_glass_primary_ranker(**ranker)
    params = ranker["rf_params"]
    reordered = dict(reversed(list(params.items())))
    as_numpy = {k: (np.int64(v) if isinstance(v, int) else np.float64(v)
                    if isinstance(v, float) else v) for k, v in params.items()}
    for given in (reordered, as_numpy):
        glass_profile.check_glass_primary_ranker(glass_rf_model="fold_rf", rf_params=given)


def _changed(**edits):
    params = dict(_primary_ranker()["rf_params"])
    for key, value in edits.items():
        if value is _DROP:
            params.pop(key)
        else:
            params[key] = value
    return params


_DROP = object()


@pytest.mark.parametrize("glass_rf_model, rf_params, match", [
    ("none", None, "glass_rf_model"),
    (np.array(["fold_rf"]), None, "glass_rf_model"),
    ("fold_rf", {"n_estimators": 651}, r"differ \['n_estimators'\]"),          # re-tuned
    ("fold_rf", {"max_features": 0.4}, r"differ \['max_features'\]"),          # rounded
    ("fold_rf", {"minority_weight": 1.0}, r"differ \['minority_weight'\]"),
    ("fold_rf", {"class_weight": "balanced_subsample"}, r"differ \['class_weight'\]"),
    ("fold_rf", {"class_weight": None}, r"differ \['class_weight'\]"),
    ("fold_rf", {"max_depth": _DROP}, r"missing \['max_depth'\]"),
    ("fold_rf", {"random_state": 42}, r"unknown \['random_state'\]"),         # an extra key
    ("fold_rf", {"n_estimators": np.array([650])}, r"differ \['n_estimators'\]"),
])
def test_ranker_settings_that_are_not_the_frozen_ones_are_refused(glass_rf_model, rf_params, match):
    params = _changed(**rf_params) if rf_params is not None else _primary_ranker()["rf_params"]
    with pytest.raises(ContractViolation, match=match):
        glass_profile.check_glass_primary_ranker(glass_rf_model=glass_rf_model, rf_params=params)


#: NumPy arrays of every shape, the zero-dimensional one included, for each kind of setting.
ARRAYS = {
    "n_estimators": [np.array(650), np.array([650]), np.array([[650]]), np.array(650.0)],
    "max_features": [np.array(0.39989815740885837), np.array([0.39989815740885837])],
    "class_weight": [np.array("balanced"), np.array(["balanced"])],
}


@pytest.mark.parametrize("key, array", [(k, a) for k, arrays in ARRAYS.items() for a in arrays],
                         ids=lambda v: f"shape{v.shape}" if isinstance(v, np.ndarray) else str(v))
def test_numpy_arrays_of_any_shape_are_refused(key, array):
    want = _primary_ranker()["rf_params"][key]
    assert not glass_profile._same(array, want)
    with pytest.raises(ContractViolation, match=rf"differ \['{key}'\]"):
        glass_profile.check_glass_primary_ranker(glass_rf_model="fold_rf",
                                                 rf_params=_changed(**{key: array}))


@pytest.mark.parametrize("model", [np.array("fold_rf"), np.array(["fold_rf"])],
                         ids=["0-d", "1-d"])
def test_an_array_never_stands_for_the_ranker_model(model):
    with pytest.raises(ContractViolation, match="glass_rf_model"):
        glass_profile.check_glass_primary_ranker(glass_rf_model=model,
                                                 rf_params=_primary_ranker()["rf_params"])


def test_a_boolean_never_stands_for_a_number_or_the_reverse():
    # True == 1 and False == 0 in Python and NumPy; neither may pass for the other.
    same = glass_profile._same
    for got, want in [(np.bool_(True), 1), (True, 1), (np.bool_(False), 0), (False, 0.0),
                      (np.bool_(True), 1.0), (1, True), (np.int64(1), np.bool_(True)),
                      (np.int64(0), False)]:
        assert not same(got, want), (got, want)
    assert same(np.bool_(True), True) and same(True, np.bool_(True))


def test_a_numpy_boolean_is_refused_where_a_frozen_value_is_1(monkeypatch):
    # The frozen values hold no 1, so a ranker whose min_samples_leaf is 1 shows the
    # substitution is refused for its type, not for a different value.
    ranker = _primary_ranker()
    ranker["rf_params"] = {**ranker["rf_params"], "min_samples_leaf": 1}
    monkeypatch.setattr(glass_profile, "ranker_settings", lambda: ranker)
    for flag in (np.bool_(True), True):
        with pytest.raises(ContractViolation, match=r"differ \['min_samples_leaf'\]"):
            glass_profile.check_glass_primary_ranker(
                glass_rf_model="fold_rf",
                rf_params={**ranker["rf_params"], "min_samples_leaf": flag})
    glass_profile.check_glass_primary_ranker(
        glass_rf_model="fold_rf",
        rf_params={**ranker["rf_params"], "min_samples_leaf": np.int64(1)})


@pytest.mark.parametrize("int_type", [np.int64, np.int32, np.int16, np.uint16, np.uint64])
@pytest.mark.parametrize("float_type", [np.float64, float])
def test_equivalent_numpy_numeric_scalars_are_accepted(int_type, float_type):
    params = {k: (int_type(v) if isinstance(v, int) else float_type(v)
                  if isinstance(v, float) else np.str_(v) if isinstance(v, str) else v)
              for k, v in _primary_ranker()["rf_params"].items()}
    glass_profile.check_glass_primary_ranker(glass_rf_model=np.str_("fold_rf"), rf_params=params)


@pytest.mark.parametrize("key, value", [
    ("n_estimators", 650.0), ("n_estimators", np.float64(650.0)), ("max_depth", np.float32(6)),
])
def test_an_integer_setting_takes_no_float_and_a_float_setting_no_integer(key, value):
    with pytest.raises(ContractViolation, match=rf"differ \['{key}'\]"):
        glass_profile.check_glass_primary_ranker(glass_rf_model="fold_rf",
                                                 rf_params=_changed(**{key: value}))
    assert not glass_profile._same(1, 1.0) and not glass_profile._same(1.0, 1)
    assert not glass_profile._same(np.float64(2.0), 2)


@pytest.mark.parametrize("rf_params", [
    lambda: list(_primary_ranker()["rf_params"].items()),
    lambda: RandomForestClassifier(n_estimators=650),
])
def test_anything_but_a_mapping_of_hyperparameters_is_refused(rf_params):
    with pytest.raises(ContractViolation, match="mapping of hyperparameters"):
        glass_profile.check_glass_primary_ranker(glass_rf_model="fold_rf", rf_params=rf_params())


def test_the_ranker_check_takes_no_record_and_is_keyword_only():
    sig = inspect.signature(glass_profile.check_glass_primary_ranker)
    assert list(sig.parameters) == ["glass_rf_model", "rf_params"]
    assert all(p.kind is p.KEYWORD_ONLY and p.default is p.empty for p in sig.parameters.values())
    assert "check_glass_primary_ranker" in glass_profile.__all__


# ======================================================================
# 2. the handoff record
# ======================================================================
def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_the_handoff_record_names_the_identities_the_code_pins():
    text = _text(HANDOFF)
    pinned = [contract_mod.FROZEN_PROFILE_FINGERPRINT, ep.FROZEN_PROTOCOL_FINGERPRINT,
              glass_profile.PRIMARY_PROFILE_FINGERPRINT, rf_profile.PRIMARY_PROFILE_FINGERPRINT]
    pinned += list(ep.load_frozen_protocol()["upstream"]["flag_fingerprints"].values())
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = manifest["identities"]
    pinned += [ids["router_split_fingerprint"]["value"], ids["train_label_fingerprint"]["value"],
               ids["stage_io_split_fingerprint"]["value"], ids["fold_plan"]["fingerprint"]]
    missing = [v for v in pinned if v not in text]
    for source in manifest["sources"]:
        if f"`{source['path']}` | `{source['sha256'][:8]}…`" not in text:
            missing.append(source["role"])
    assert missing == []
    assert len(manifest["sources"]) == 7


def test_the_handoff_record_covers_every_gate_and_requirement():
    text = _text(HANDOFF)
    owed = [f"G{i}" for i in range(1, 14)] + [f"R{i}" for i in range(1, 9)]
    missing = [g for g in owed if not re.search(rf"\b{g}\b", text)]
    assert missing == []
    for span in ("V1–V8", "O1–O4", "C1–C9", "D-C6", "D-C7", "D-C11", "M3", "M4", "S1"):
        assert span in text, span
