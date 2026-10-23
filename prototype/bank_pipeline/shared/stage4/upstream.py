"""
shared.stage4.upstream
======================
Which arm's Stage 1-3 artifacts a Stage 4 arbiter consumes (PR 36 audit F29).

The research design is a 2x2: each arbiter (GLASS Meta-EBM, black-box
Meta-XGB) on each arm's upstream, with the upstream arm held fixed inside one
run. An arbiter reads its own arm by default; the crossover is declared
explicitly with ``upstream="glass"`` or ``upstream="blackbox"``:

    upstream    Stage 1                     Stage 2                    Stage 3
    glass       StageOutput arm 'glass'     GLASS Router bundle        Stage3Artifact 'ebm' / 'glass'
                (GLASS LR)                  (top-level layout)         (GLASS EBM)
    blackbox    StageOutput arm 'blackbox'  RF Router artifact         Stage3Artifact 'xgboost' / 'blackbox'
                (MLP)                       ('extra' layout)           (XGBoost)

Every artifact must belong to the declared arm: a mixed upstream, an
undeclared crossover and an artifact labelled 'unset' (PR 36 F21) are
refused. The verified combination is recorded with the Stage 4 inputs
(``Stage4Inputs.upstream``): arm, arbiter, whether it is a crossover, and per
stage the slot, family, arm, probability space, Pass 2 population (routers),
label evidence and a content fingerprint of exactly the outputs consumed —
so two arbiters can be shown to have read the same upstream
(``blackbox_pipeline.models.meta_xgb.comparison``).

Arbiter slot names (``lr``/``glass``/``ebm``, ``mlp``/``rf_router``/``xgb``)
stay the arbiter's own in a crossover; the record says what filled them.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from .contract import SEC_SEM, Stage4InputError, StageStream

__all__ = ["UPSTREAM_ARMS", "UPSTREAM", "resolve_upstream", "upstream_checks",
           "stream_fingerprint", "upstream_record", "same_upstream"]

UPSTREAM_ARMS = ("glass", "blackbox")

#: Per upstream arm and stage: the identity every artifact must declare, the
#: label its stream carries and the requirement as a check name.
UPSTREAM = {
    "glass": {
        "stage1": {"want": {"arm": "glass"}, "label": "GLASS LR",
                   "check": "GLASS-arm Stage 1 output (StageOutput.arm = 'glass')"},
        "stage2": {"want": {"model_family": "glass_router"}, "label": "GLASS Router",
                   "check": "GLASS Router bundle (top-level Stage 4 layout)"},
        "stage3": {"want": {"model_family": "ebm", "arm": "glass"}, "label": "GLASS EBM",
                   "check": "GLASS EBM artifact (model_family 'ebm', arm 'glass')"},
    },
    "blackbox": {
        "stage1": {"want": {"arm": "blackbox"}, "label": "MLP",
                   "check": "black-box Stage 1 output (StageOutput.arm = 'blackbox')"},
        "stage2": {"want": {"model_family": "rf_router"}, "label": "RF Router (two-pass)",
                   "check": "RF Router artifact ('extra' Stage 4 layout)"},
        "stage3": {"want": {"model_family": "xgboost", "arm": "blackbox"}, "label": "XGBoost",
                   "check": "black-box XGBoost artifact (model_family 'xgboost', arm 'blackbox')"},
    },
}


def resolve_upstream(upstream, arbiter_arm: str) -> str:
    """``upstream`` (None: the arbiter's own arm), validated."""
    arm = arbiter_arm if upstream is None else upstream
    if arm not in UPSTREAM_ARMS:
        raise Stage4InputError(
            f"upstream must be one of {UPSTREAM_ARMS} — the arm every Stage 1-3 artifact "
            f"comes from (PR 36 F29); got {upstream!r}")
    return arm


def upstream_checks(streams, upstream: str) -> list:
    """(section, stage, name, fn, detail) per stream: it belongs to ``upstream``."""
    other = [a for a in UPSTREAM_ARMS if a != upstream][0]
    checks = []
    for s in streams:
        spec = UPSTREAM[upstream][s.stage]
        got = {k: s.notes.get(k) for k in spec["want"]}
        hint = ""
        if "unset" in got.values():
            hint = " (an 'unset' label is PR 36 F21: regenerate Stage 3)"
        elif got == UPSTREAM[other][s.stage]["want"]:
            hint = (f" — a {other} artifact: an arbiter reads another arm's upstream only when "
                    f"declared (upstream='{other}', PR 36 F29)")
        checks.append((SEC_SEM, s.stage, f"{s.name}: {spec['check']}",
                       lambda got=got, want=spec["want"]: got == want,
                       ", ".join(f"{k}={v!r}" for k, v in got.items()) + hint))
    return checks


def stream_fingerprint(s: StageStream) -> str:
    """
    16-hex identity of exactly the outputs a stream hands Stage 4: train and
    test probabilities on the rows where they are defined, router decisions
    and train fold ids. Independent of how an arbiter fills undefined router
    rows (Meta-EBM: 0.5, Meta-XGB: NaN), so both arbiters reading one artifact
    agree.
    """
    h = hashlib.sha256()
    for side in ("train", "test"):
        p = np.asarray(getattr(s, f"{side}_proba"), dtype=float)
        defined = getattr(s, f"{side}_defined")
        if defined is not None and len(defined) == len(p):
            p = np.where(np.asarray(defined, dtype=bool), p, np.nan)
        h.update(side.encode())
        h.update(np.ascontiguousarray(p, dtype="<f8").tobytes())
        dec = getattr(s, f"{side}_decisions")
        if dec is not None:
            h.update(",".join(map(str, dec)).encode())
    if s.train_fold_id is not None:
        h.update(np.ascontiguousarray(s.train_fold_id, dtype="<i8").tobytes())
    return h.hexdigest()[:16]


def upstream_record(streams, upstream: str, arbiter: str, arbiter_arm: str) -> dict:
    """What an arbiter consumed, for its artifact."""
    stages: dict = {}
    for s in streams:
        stages[s.stage] = {
            "slot": s.name, "model": s.model,
            "model_family": s.notes.get("model_family"), "arm": s.notes.get("arm"),
            "artifact_path": s.artifact_path, "probability_space": s.probability_space,
            "label_evidence": s.notes.get("label_evidence"),
            "content_fingerprint": stream_fingerprint(s),
        }
        if s.stage == "stage2":
            stages[s.stage]["pass2_population"] = s.notes.get("pass2_population")
            stages[s.stage]["pass2_population_evidence"] = s.notes.get("pass2_population_evidence")
    return {"arm": upstream, "arbiter": arbiter, "arbiter_arm": arbiter_arm,
            "crossover": upstream != arbiter_arm,
            "combination": f"{upstream} upstream -> {arbiter}", "stages": stages}


def same_upstream(a: Optional[dict], b: Optional[dict]) -> Optional[str]:
    """None when both records describe identical upstream outputs; else why not."""
    if not a or not b:
        return "an artifact records no upstream (written before PR 36 F29)"
    if a.get("arm") != b.get("arm"):
        return f"upstream arms differ ({a.get('arm')} vs {b.get('arm')})"
    for stage in ("stage1", "stage2", "stage3"):
        fa = (a.get("stages") or {}).get(stage, {}).get("content_fingerprint")
        fb = (b.get("stages") or {}).get(stage, {}).get("content_fingerprint")
        if fa is None or fa != fb:
            return f"{stage} outputs differ (content fingerprint {fa} vs {fb})"
    return None
