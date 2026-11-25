"""PR 43 registration and pure policy arithmetic. Inputs are aligned evaluation rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_research import research_records as records
from shared.metrics import _check_aligned

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "research_logs/pr43"
SOURCE = "research_logs/pr43/stage4_policy_v1.json"
NOTEBOOK = "feature_research/training_research.ipynb"
POLICY_SHA256 = "42f457df53baa0b52739f99097ae7da2d9d1bc32794351c8ef5dd8b7328051ba"


def load_policy():
    policy = json.loads((ROOT / SOURCE).read_text(encoding="utf-8"))
    if records.content_digest({"policy": policy}) != POLICY_SHA256:
        raise ValueError("Policy digest mismatch: preserve v1 and register an amended version")
    return policy


def risk_curve(y, decision, confidence, cost):
    """All attainable, nonempty retained sets in ascending order of coverage."""
    _check_aligned(y, decision, confidence)
    y, d, c = np.asarray(y), np.asarray(decision), np.asarray(confidence, dtype=float)
    if y.ndim != 1 or not y.size or not (y.shape == d.shape == c.shape):
        raise ValueError("Expected three aligned, nonempty vectors")
    if not (np.isin(y, [0, 1]).all() and np.isin(d, [0, 1]).all()):
        raise ValueError("Expected binary labels and pre-deferral decisions")
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("Expected finite nonnegative confidence")
    order = np.argsort(-c, kind="stable")
    y, d, c = y[order], d[order], c[order]
    ends = np.r_[np.flatnonzero(c[1:] != c[:-1]), len(c) - 1]
    n, kept = len(y), ends + 1
    pos = np.cumsum(y == 1)[ends]
    neg = kept - pos
    p = int(np.count_nonzero(y))
    fp = np.cumsum((d == 1) & (y == 0))[ends]
    fn = np.cumsum((d == 0) & (y == 1))[ends]
    mistakes = fp * cost["fp"] + fn * cost["fn"]
    cut = c[ends].copy()
    cut[-1] = 0.0  # Retain-all remains retain-all when deployed.
    wrong_deferred = np.count_nonzero(d != y) - fp - fn
    return pd.DataFrame({
        "cut": cut, "kept": kept, "coverage": kept / n, "deferred": n - kept,
        "fp": fp, "fn": fn, "tp": pos - fn, "tn": neg - fp,
        "positive_kept": pos, "negative_kept": neg,
        "positive_retention": pos / p if p else np.nan,
        "negative_retention": neg / (n - p) if n != p else np.nan,
        "deferred_positive": p - pos, "deferred_negative": n - p - neg,
        "correct_deferred": n - kept - wrong_deferred, "wrong_deferred": wrong_deferred,
        "effective_recall": (pos - fn) / p if p else np.nan, "retained_prevalence": pos / kept,
        "risk": mistakes / kept, "loss": (mistakes + cost["defer"] * (n - kept)) / n,
    })


def operating_cut(curve, policy):
    """Inner predictions only. None explicitly denotes infeasibility."""
    eligible = curve.loc[
        (curve.coverage >= policy["min_total"])
        & (curve[["positive_retention", "negative_retention"]] >= policy["min_each_class"]).all(axis=1)
        & (curve[["positive_kept", "negative_kept"]] >= policy["min_retained_each_class"]).all(axis=1)
    ]
    if eligible.empty:
        return None
    return float(eligible.sort_values(["loss", "kept", "cut"], ascending=[True, False, True]).iloc[0].cut)


def matched_risks(curves, target):
    """One common row count; a missing group boundary makes the contrast unavailable."""
    sizes = {int(frame.kept.iloc[-1]) for frame in curves.values()}
    if len(sizes) != 1 or not 0 < target <= 1:
        raise ValueError("Expected common population size and target in (0, 1]")
    from fractions import Fraction
    from math import ceil
    count = ceil(Fraction(str(target)) * sizes.pop())
    risks, brackets = {}, {}
    for arm, frame in curves.items():
        counts = frame.kept.to_numpy()
        i = counts.searchsorted(count)
        if counts[i] == count:
            risks[arm] = float(frame.risk.iloc[i])
        else:
            brackets[arm] = (int(counts[i - 1]) if i else 0, int(counts[i]))
    return count, None if brackets else risks, brackets


def partial_aurc(curve, floor):
    coverage = curve.coverage.to_numpy()
    widths = np.diff(np.maximum(np.r_[0, coverage], floor))
    return float(widths @ curve.risk.to_numpy() / (1 - floor))


def loss_verdict(lower, upper, mean, fold_sd, policy):
    """Classify the primary loss interval; other registered safeguards still apply."""
    if not np.isfinite([lower, upper, mean, fold_sd]).all():
        return "inconclusive"
    if lower > upper or fold_sd < 0:
        raise ValueError("Invalid interval or fold SD")
    margin = policy["comparability"]
    if upper < -policy["improvement"] and -mean > fold_sd:
        return "improved"
    if lower >= -margin and upper <= margin:
        return "comparable"
    if lower > margin:
        return "worse"
    return "inconclusive"


def register(root=LOGS):
    policy = load_policy()
    run_id = policy["input_run"]
    frozen = records.read_run_record(run_id, "frozen_inputs", root=root)
    if frozen["run"]["run_id"] != run_id or frozen["run"]["notebook"] != NOTEBOOK:
        raise ValueError("Unexpected Step 2A record identity")
    for key, expected in policy["population"].items():
        if frozen["population"].get(key) != expected:
            raise ValueError(f"Step 2A population mismatch: {key}")
    if set(frozen["upstreams"]) != {"stage1.lr", "stage1.mlp", "stage2.glass_router",
                                    "stage2.rf_router", "stage3.ebm", "stage3.xgb"}:
        raise ValueError("Incomplete Step 2A upstream inventory")
    if set(frozen["historical_reference"]) != {"glass_arbiter", "glass_arbiter_on_blackbox",
                                              "meta_xgb", "meta_xgb_on_glass"}:
        raise ValueError("Incomplete Step 2A crossover inventory")
    run = records.start_run(commit=2, seeds={"paired_bootstrap": policy["uncertainty"]["seed"]},
                            notebook=NOTEBOOK, root=root,
                            code_files=(SOURCE, NOTEBOOK, "feature_research/stage4_policy.py",
                                        "feature_research/research_records.py"))
    source = Path(root) / run_id / "frozen_inputs.json"
    # read_run_record has already verified this sidecar and canonical content.
    return records.write_run_record(run, "stage4_registration", {
        "policy": {"id": policy["id"], "version": policy["version"], "path": SOURCE,
                   "canonical_sha256": POLICY_SHA256},
        "frozen_inputs": {"run_id": run_id, "content_sha256": frozen["content_sha256"],
                          "file_sha256": source.with_suffix(".json.sha256").read_text().strip()},
    })


if __name__ == "__main__":
    print(register())
