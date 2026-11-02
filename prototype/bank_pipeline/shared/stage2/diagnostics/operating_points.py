"""
shared.stage2.diagnostics.operating_points
==========================================
Named upstream operating points, resolved to per-row flags, and label-free
rate-matched cuts.

Registry
--------
Every operating point is a named entry of ``OPERATING_POINTS``; there is no
API that takes a bare float. Each names its model, score space, the stored
source of its threshold or decisions, how that value was selected, and the
disclosure that must accompany any number computed with it.

    name                      role       space                    source
    lr.in_stage_f2_cv         primary    calibrated_probability   LR StageOutput.threshold
    ebm.nested_f2             primary    raw_model_score          EBM stored oof.decision
    ebm.global_f2_raw         secondary  raw_model_score          EBM threshold.oof
    ebm.global_f2_calibrated  secondary  calibrated_probability   EBM threshold.oof_calibrated_space
    lr.stage4_recall70        secondary  calibrated_probability   Meta-EBM thresholds.lr
    ebm.stage4_recall70       secondary  calibrated_probability   Meta-EBM thresholds.ebm

A flag is ``score >= threshold`` on the stored float, in the threshold's own
space (raw and calibrated EBM values are never mixed). ``ebm.nested_f2`` uses
the stored decisions, which the boundary verified against the per-fold nested
thresholds.

Rate-matched cuts
-----------------
``matched_rate_cut`` selects the ``k`` highest (or lowest) scores of a
population, where ``k`` is a router's realised count. Labels are never used.
When the k-th value is tied with values outside the top k there is no
label-free way to pick exactly k rows, so BOTH achievable cuts are returned:
``strict`` (the tied block excluded, fewer than k rows) and ``inclusive`` (the
tied block included, more than k rows). There is no default nearest cut and no
arbitrary tie breaking (Step 1C decision 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import (
    SCORE_SPACES,
    SELECTION_FOLD_NESTED,
    SELECTION_GLOBAL_TRAIN_LABELS,
    SELECTION_RATE_MATCHED,
    RouterDiagnosticError,
    ScoreStream,
    Stage4Thresholds,
    UnsupportedDiagnostic,
    UpstreamOOFView,
    _frozen,
)

__all__ = [
    "OperatingPointSpec", "OPERATING_POINTS", "PRIMARY_OPERATING_POINTS",
    "SECONDARY_OPERATING_POINTS", "UpstreamFlags", "resolve_operating_point", "combine_flags",
    "MatchedCut", "matched_rate_cut", "RANKINGS",
]

RANKINGS = ("highest", "lowest")
_POPULATION_ALL = "all training rows (outer out-of-fold scores)"
_IN_SELECTION = ("In-selection: the threshold was chosen on the labels of the same training "
                 "rows it is applied to; rates computed with it are optimistic.")


@dataclass(frozen=True)
class OperatingPointSpec:
    name: str
    model: str                # "lr" | "ebm"
    role: str                 # "primary" | "secondary"
    score_space: str
    rule: str
    source: str               # "threshold" | "stored_decision" | "stage4"
    operator: str
    selection: str
    population: str
    disclosure: str

    def __post_init__(self) -> None:
        if self.score_space not in SCORE_SPACES:
            raise RouterDiagnosticError(f"{self.name}: unknown space {self.score_space!r}")


def _spec(name, model, role, space, rule, source, selection, disclosure) -> OperatingPointSpec:
    return OperatingPointSpec(name, model, role, space, rule, source, ">=", selection,
                              _POPULATION_ALL, disclosure)


OPERATING_POINTS: Mapping[str, OperatingPointSpec] = MappingProxyType({s.name: s for s in (
    _spec("lr.in_stage_f2_cv", "lr", "primary", "calibrated_probability",
          "calibrated OOF probability >= StageOutput.threshold", "threshold",
          SELECTION_GLOBAL_TRAIN_LABELS,
          "Stage 1 LR operating point (threshold_source in_stage_f2_cv, 0.10 in the Oct 23 "
          "artifact). " + _IN_SELECTION),
    _spec("ebm.nested_f2", "ebm", "primary", "raw_model_score",
          "stored oof.decision: raw OOF score >= the fold's nested F2 threshold",
          "stored_decision", SELECTION_FOLD_NESTED,
          "Stage 3 EBM fold-nested operating point: each fold's raw-space threshold was "
          "selected on inner OOF scores of that fold's training rows."),
    _spec("ebm.global_f2_raw", "ebm", "secondary", "raw_model_score",
          "raw OOF score >= threshold.oof.threshold", "threshold", SELECTION_GLOBAL_TRAIN_LABELS,
          "Stage 3 EBM global raw-space F2 threshold (0.53 in the Oct 23 artifact). "
          + _IN_SELECTION),
    _spec("ebm.global_f2_calibrated", "ebm", "secondary", "calibrated_probability",
          "calibrated OOF probability >= threshold.oof_calibrated_space.threshold", "threshold",
          SELECTION_GLOBAL_TRAIN_LABELS,
          "Stage 3 EBM global calibrated-space F2 threshold (0.11 in the Oct 23 artifact). "
          + _IN_SELECTION),
    _spec("lr.stage4_recall70", "lr", "secondary", "calibrated_probability",
          "calibrated OOF probability >= Meta-EBM thresholds.lr", "stage4",
          SELECTION_GLOBAL_TRAIN_LABELS,
          "Stage 4's own recall-0.70 LR cut, not a Stage 1 operating point. " + _IN_SELECTION),
    _spec("ebm.stage4_recall70", "ebm", "secondary", "calibrated_probability",
          "calibrated OOF probability >= Meta-EBM thresholds.ebm", "stage4",
          SELECTION_GLOBAL_TRAIN_LABELS,
          "Stage 4's own recall-0.70 EBM cut (calibrated space), not a Stage 3 operating "
          "point. " + _IN_SELECTION),
)})
PRIMARY_OPERATING_POINTS = tuple(k for k, s in OPERATING_POINTS.items() if s.role == "primary")
SECONDARY_OPERATING_POINTS = tuple(k for k, s in OPERATING_POINTS.items() if s.role == "secondary")


@dataclass(frozen=True)
class UpstreamFlags:
    """Per-row upstream flags (``True`` = the upstream model predicts SUBSCRIBE)."""

    name: str
    model: str
    index: pd.Index
    y: np.ndarray
    fold_id: np.ndarray
    flags: np.ndarray
    score_space: Optional[str]
    threshold: Optional[float]
    field: str
    selection: str
    disclosure: str
    components: Tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.flags)

    @property
    def n_flagged(self) -> int:
        return int(self.flags.sum())


def resolve_operating_point(view: UpstreamOOFView, name: str, *,
                            stage4: Optional[Stage4Thresholds] = None) -> UpstreamFlags:
    """Flags of ``view`` at the registered operating point ``name``."""
    if not isinstance(name, str) or name not in OPERATING_POINTS:
        raise RouterDiagnosticError(
            f"unknown operating point {name!r}; registered: {sorted(OPERATING_POINTS)}")
    if not isinstance(view, UpstreamOOFView):
        raise RouterDiagnosticError("resolve_operating_point needs an UpstreamOOFView")
    spec = OPERATING_POINTS[name]
    if view.model != spec.model:
        raise RouterDiagnosticError(f"{name} is a {spec.model} operating point, view is {view.model}")
    if spec.source == "stored_decision":
        sd = view.stored_decisions.get(name)
        if sd is None:
            raise UnsupportedDiagnostic(
                f"{name}: the artifact stores no decisions for it (for the EBM: "
                "threshold.oof_nested.drives_train_decisions was False)")
        if sd.space != spec.score_space:
            raise RouterDiagnosticError(f"{name}: stored decisions in {sd.space}, spec {spec.score_space}")
        flags, thr, fld = np.asarray(sd.values) == 1, None, sd.field
    else:
        if spec.source == "stage4":
            if stage4 is None:
                raise UnsupportedDiagnostic(f"{name}: pass stage4=extract_stage4_recall_thresholds(...)")
            rec = stage4.records.get(name)
        else:
            rec = view.thresholds.get(name)
        if rec is None:
            raise UnsupportedDiagnostic(f"{name}: no stored threshold registered in this view")
        stream: Optional[ScoreStream] = view.scores.get(spec.score_space)
        if stream is None:
            raise UnsupportedDiagnostic(f"{name}: the view has no {spec.score_space} score")
        if rec.space != spec.score_space or stream.space != spec.score_space:
            raise RouterDiagnosticError(
                f"{name}: threshold space {rec.space} / score space {stream.space} != "
                f"{spec.score_space}")
        flags, thr, fld = np.asarray(stream.values) >= rec.value, float(rec.value), rec.field
    return UpstreamFlags(name=name, model=spec.model, index=view.index, y=view.y,
                         fold_id=view.fold_id, flags=_frozen(flags, bool),
                         score_space=spec.score_space, threshold=thr, field=fld,
                         selection=spec.selection, disclosure=spec.disclosure, components=(name,))


def combine_flags(a: UpstreamFlags, b: UpstreamFlags, *, how: str) -> UpstreamFlags:
    """``how="and"``: both flag; ``"or"``: either flags. Same rows and labels required."""
    if how not in ("and", "or"):
        raise RouterDiagnosticError(f"how must be 'and' or 'or', got {how!r}")
    if not (a.index.equals(b.index) and np.array_equal(a.y, b.y)):
        raise RouterDiagnosticError("combine_flags: the two flag sets cover different rows / labels")
    flags = (a.flags & b.flags) if how == "and" else (a.flags | b.flags)
    return UpstreamFlags(
        name=f"({a.name} {how} {b.name})", model=f"{a.model}+{b.model}", index=a.index, y=a.y,
        fold_id=a.fold_id, flags=_frozen(flags, bool), score_space=None, threshold=None,
        field=f"{a.field} {how} {b.field}",
        selection="; ".join(dict.fromkeys((a.selection, b.selection))),
        disclosure=" | ".join(dict.fromkeys((a.disclosure, b.disclosure))),
        components=a.components + b.components)


# ======================================================================
# Rate-matched cuts
# ======================================================================
@dataclass(frozen=True)
class MatchedCut:
    """
    Label-free cuts selecting the ``target_count`` highest / lowest scores.
    ``exact``: one cut selects exactly ``target_count`` rows and ``strict`` and
    ``inclusive`` are that same set. Otherwise ``strict`` has fewer rows and
    ``inclusive`` more; the tied block at the boundary decides which.
    """

    ranking: str
    target_count: int
    population_size: int
    cut_value: Optional[float]
    strict: np.ndarray
    inclusive: np.ndarray
    exact: bool
    tie_block: int
    by_fold: Optional[Mapping[int, Mapping[str, object]]] = None
    selection: str = SELECTION_RATE_MATCHED

    @property
    def strict_count(self) -> int:
        return int(self.strict.sum())

    @property
    def inclusive_count(self) -> int:
        return int(self.inclusive.sum())


def _one_cut(s: np.ndarray, pop: np.ndarray, k: int, ranking: str):
    n_pop = int(pop.sum())
    if k < 0 or k > n_pop:
        raise RouterDiagnosticError(f"target_count {k} outside 0..{n_pop} (population size)")
    empty = np.zeros(len(s), dtype=bool)
    if k == 0:
        return None, empty, empty.copy(), True, 0
    vals = np.sort(s[pop])
    c = float(vals[::-1][k - 1] if ranking == "highest" else vals[k - 1])
    if ranking == "highest":
        inclusive, strict = pop & (s >= c), pop & (s > c)
    else:
        inclusive, strict = pop & (s <= c), pop & (s < c)
    tie = int((pop & (s == c)).sum())
    if int(inclusive.sum()) == k:
        return c, inclusive.copy(), inclusive, True, tie
    return c, strict, inclusive, False, tie


def matched_rate_cut(score, *, target_count, population_mask=None, ranking: str = "highest",
                     fold_id=None) -> MatchedCut:
    """
    ``score``: a ScoreStream or 1-D array. ``target_count``: an int (pooled), or
    {fold: int} with ``fold_id`` (one cut per fold, combined).
    ``population_mask``: rows eligible for selection (default: all).
    """
    if ranking not in RANKINGS:
        raise RouterDiagnosticError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    s = np.asarray(score.values if isinstance(score, ScoreStream) else score, dtype=float)
    if s.ndim != 1:
        raise RouterDiagnosticError("score must be 1-D")
    pop = (np.ones(len(s), dtype=bool) if population_mask is None
           else np.asarray(population_mask, dtype=bool))
    if pop.shape != s.shape:
        raise RouterDiagnosticError("population_mask length != score length")
    if np.isnan(s[pop]).any():
        raise RouterDiagnosticError("score has NaN inside the population")
    if fold_id is None:
        if isinstance(target_count, Mapping):
            raise RouterDiagnosticError("per-fold target counts need fold_id")
        c, strict, inclusive, exact, tie = _one_cut(s, pop, int(target_count), ranking)
        return MatchedCut(ranking, int(target_count), int(pop.sum()), c, _frozen(strict, bool),
                          _frozen(inclusive, bool), exact, tie)
    fold = np.asarray(fold_id)
    if fold.shape != s.shape or not isinstance(target_count, Mapping):
        raise RouterDiagnosticError("per-fold cuts need fold_id of score length and {fold: count}")
    folds = sorted(int(k) for k in np.unique(fold))
    if sorted(int(k) for k in target_count) != folds:
        raise RouterDiagnosticError(f"target counts for folds {sorted(target_count)} != {folds}")
    strict, inclusive = np.zeros(len(s), bool), np.zeros(len(s), bool)
    detail, exact_all, tie_all = {}, True, 0
    for k in folds:
        c, st, inc, exact, tie = _one_cut(s, pop & (fold == k), int(target_count[k]), ranking)
        strict |= st
        inclusive |= inc
        exact_all &= exact
        tie_all += tie
        detail[k] = MappingProxyType({"target_count": int(target_count[k]), "cut_value": c,
                                      "exact": exact, "tie_block": tie,
                                      "strict_count": int(st.sum()),
                                      "inclusive_count": int(inc.sum())})
    return MatchedCut(ranking, int(sum(int(v) for v in target_count.values())), int(pop.sum()),
                      None, _frozen(strict, bool), _frozen(inclusive, bool), exact_all, tie_all,
                      MappingProxyType(detail))
