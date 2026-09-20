"""
blackbox_pipeline.models.rf_router.thresholds
==============================================
Operating-point selection.

One rule governs this whole module: **thresholds are solved from training /
out-of-fold data only**. No function here accepts test probabilities or test
labels, and none should ever be given them. That is the black-box counterpart
of GLASS selecting its rules against ``y_val`` inside ``fit`` and never looking
at the holdout.

Routing conventions (strict inequalities, matching ``router.predict``)
----------------------------------------------------------------------
    Pass 1 routes NOT_SUBSCRIBE when ``p <  t1``
    Pass 2 flags  SUBSCRIBE     when ``p >  t2``

Selection rules
---------------
``t1`` — the LARGEST threshold whose band still satisfies the NPV floor and
both subscriber-leakage budgets. Largest means maximum Pass 1 coverage at the
declared error budget, which is the role GLASS Pass 1 plays: route away as much
of the negative mass as the leakage budget allows.

``t2`` — the SMALLEST threshold whose band still satisfies the precision floor
(and the recall floor, when set). Smallest means maximum recall at the declared
precision, which is the role GLASS Pass 2 plays.

Both searches scan every realisable band rather than assuming monotonicity.
Empirical precision is not monotone in the threshold at finite sample size, so
a bisection would silently stop at the first local violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .constraints import OperatingConstraints

__all__ = ["ThresholdResult", "ThresholdPair", "ThresholdSolver"]


# ======================================================================
# Results
# ======================================================================

@dataclass
class ThresholdResult:
    """One solved threshold plus the band statistics that justified it."""

    threshold: float
    feasible: bool
    reason: str
    band_size: int
    band_fraction: float
    band_precision: float          # P(target class | in band)
    positives_in_band: int
    leakage_rate: float            # Pass 1 only; 0.0 for Pass 2
    recall: float                  # Pass 2 only; 0.0 for Pass 1
    pass_name: str
    sweep: Optional[pd.DataFrame] = field(default=None, repr=False)

    def raise_if_infeasible(self, context: str = "") -> None:
        if not self.feasible:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"No feasible {self.pass_name} threshold{where}: {self.reason}\n"
                f"Inspect result.sweep to see which constraint binds, then "
                f"either relax the constraint or accept a smaller band."
            )

    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items() if k != "sweep"
        }


@dataclass
class ThresholdPair:
    """The frozen operating point: everything ``predict`` needs."""

    t1: float
    t2: float
    pass1: ThresholdResult
    pass2: ThresholdResult

    def to_dict(self) -> dict:
        return {
            "t1": float(self.t1),
            "t2": float(self.t2),
            "pass1": self.pass1.to_dict(),
            "pass2": self.pass2.to_dict(),
        }


# ======================================================================
# Solver
# ======================================================================

class ThresholdSolver:
    """
    Solves ``t1`` and ``t2`` against an ``OperatingConstraints``.

    Stateless apart from the constraints; safe to reuse across folds, which is
    what the nested diagnostic does.
    """

    def __init__(self, constraints: OperatingConstraints, verbose: bool = False):
        self.constraints = constraints
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Pass 1
    # ------------------------------------------------------------------
    def solve_pass1(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        keep_sweep: bool = True,
    ) -> ThresholdResult:
        """
        Largest ``t1`` such that ``{p < t1}`` meets the NPV floor and both
        leakage budgets.

        ``proba`` and ``y`` must come from out-of-fold predictions on the
        training population.
        """
        p, y = _prepare(proba, y, "Pass 1")
        c = self.constraints
        n = len(p)
        total_pos = int(y.sum())

        order = np.argsort(p, kind="mergesort")
        p_s, y_s = p[order], y[order]
        cum_pos = np.concatenate([[0], np.cumsum(y_s)])  # cum_pos[k] over first k

        ks = _lower_band_sizes(p_s)
        rows, best_k, best = [], None, None

        for k in ks:
            pos = int(cum_pos[k])
            npv = (k - pos) / k
            leak_rate = pos / total_pos if total_pos else 0.0
            cov = k / n

            why: list[str] = []
            if k < c.min_band_support_pass1:
                why.append("support")
            if npv < c.min_precision_not_subscribe:
                why.append("npv")
            if leak_rate > c.max_subscriber_leakage_rate:
                why.append("leak_rate")
            if pos > c.max_subscriber_leakage_absolute:
                why.append("leak_abs")
            if c.min_pass1_coverage is not None and cov < c.min_pass1_coverage:
                why.append("coverage")
            ok = not why

            rows.append(
                {
                    "band_size": k,
                    "coverage": cov,
                    "npv": npv,
                    "subscribers_routed": pos,
                    "leakage_rate": leak_rate,
                    "threshold": _upper_edge(p_s, k),
                    "feasible": ok,
                    "binding": ",".join(why),
                }
            )
            if ok:
                best_k = k  # ks ascending -> last feasible is the largest band

        sweep = pd.DataFrame(rows) if keep_sweep else None

        if best_k is None:
            return ThresholdResult(
                threshold=float("nan"),
                feasible=False,
                reason=(
                    "no band satisfies the NPV floor and leakage budget "
                    f"(npv>={c.min_precision_not_subscribe:.3f}, "
                    f"leak<={c.max_subscriber_leakage_rate:.3f} and "
                    f"<={c.max_subscriber_leakage_absolute} subscribers, "
                    f"support>={c.min_band_support_pass1})"
                ),
                band_size=0,
                band_fraction=0.0,
                band_precision=0.0,
                positives_in_band=0,
                leakage_rate=0.0,
                recall=0.0,
                pass_name="Pass 1",
                sweep=sweep,
            )

        k = best_k
        pos = int(cum_pos[k])
        best = ThresholdResult(
            threshold=float(_upper_edge(p_s, k)),
            feasible=True,
            reason="ok",
            band_size=int(k),
            band_fraction=float(k / n),
            band_precision=float((k - pos) / k),
            positives_in_band=pos,
            leakage_rate=float(pos / total_pos) if total_pos else 0.0,
            recall=0.0,
            pass_name="Pass 1",
            sweep=sweep,
        )
        if self.verbose:
            print(
                f"   t1 = {best.threshold:.6f} | routes {best.band_size:,} "
                f"({best.band_fraction:.1%}) | NPV {best.band_precision:.4f} | "
                f"leaks {best.positives_in_band} subs "
                f"({best.leakage_rate:.2%})"
            )
        return best

    # ------------------------------------------------------------------
    # Pass 2
    # ------------------------------------------------------------------
    def solve_pass2(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        n_positives_global: Optional[int] = None,
        keep_sweep: bool = True,
    ) -> ThresholdResult:
        """
        Smallest ``t2`` such that ``{p > t2}`` meets the precision floor (and
        the recall floor, when set).

        Parameters
        ----------
        proba, y
            Out-of-fold predictions and labels for the population Pass 2
            operates on. In the two-pass cascade that is the REMAINDER, not the
            full training set.
        n_positives_global
            Total positives in the full training population. Required when
            ``constraints.recall_denominator == "global"`` — which is the
            GLASS-comparable setting, because GLASS Pass 2 rules are scored by
            ``RuleEvaluator`` against every positive in the evaluation set, not
            only those the router left unrouted.
        """
        p, y = _prepare(proba, y, "Pass 2")
        c = self.constraints
        n = len(p)

        if c.recall_denominator == "global":
            if n_positives_global is None:
                raise ValueError(
                    "recall_denominator='global' requires n_positives_global "
                    "(total positives in the full training population)."
                )
            denom = int(n_positives_global)
        else:
            denom = int(y.sum())

        order = np.argsort(p, kind="mergesort")
        p_s, y_s = p[order], y[order]
        # suffix_pos[m] = positives among the m highest-scoring rows
        suffix_pos = np.concatenate([[0], np.cumsum(y_s[::-1])])

        ms = _upper_band_sizes(p_s)
        rows, best_m = [], None

        for m in ms:
            pos = int(suffix_pos[m])
            prec = pos / m
            rec = pos / denom if denom else 0.0

            why: list[str] = []
            if m < c.min_band_support_pass2:
                why.append("support")
            if prec < c.min_precision_subscribe:
                why.append("precision")
            if c.min_recall_subscribe is not None and rec < c.min_recall_subscribe:
                why.append("recall")
            ok = not why

            rows.append(
                {
                    "band_size": m,
                    "band_fraction": m / n,
                    "precision": prec,
                    "positives_in_band": pos,
                    "recall": rec,
                    "threshold": _lower_edge(p_s, m),
                    "feasible": ok,
                    "binding": ",".join(why),
                }
            )
            if ok:
                best_m = m  # ms ascending -> last feasible is the largest band

        sweep = pd.DataFrame(rows) if keep_sweep else None

        if best_m is None:
            recall_txt = (
                "" if c.min_recall_subscribe is None
                else f", recall>={c.min_recall_subscribe:.3f}"
            )
            return ThresholdResult(
                threshold=float("nan"),
                feasible=False,
                reason=(
                    "no band satisfies the precision floor "
                    f"(precision>={c.min_precision_subscribe:.3f}{recall_txt}, "
                    f"support>={c.min_band_support_pass2})"
                ),
                band_size=0,
                band_fraction=0.0,
                band_precision=0.0,
                positives_in_band=0,
                leakage_rate=0.0,
                recall=0.0,
                pass_name="Pass 2",
                sweep=sweep,
            )

        m = best_m
        pos = int(suffix_pos[m])
        best = ThresholdResult(
            threshold=float(_lower_edge(p_s, m)),
            feasible=True,
            reason="ok",
            band_size=int(m),
            band_fraction=float(m / n),
            band_precision=float(pos / m),
            positives_in_band=pos,
            leakage_rate=0.0,
            recall=float(pos / denom) if denom else 0.0,
            pass_name="Pass 2",
            sweep=sweep,
        )
        if self.verbose:
            print(
                f"   t2 = {best.threshold:.6f} | flags {best.band_size:,} "
                f"({best.band_fraction:.1%} of its population) | "
                f"precision {best.band_precision:.4f} | "
                f"recall {best.recall:.4f} ({c.recall_denominator} denominator)"
            )
        return best

    # ------------------------------------------------------------------
    def make_pair(
        self, pass1: ThresholdResult, pass2: ThresholdResult
    ) -> ThresholdPair:
        """
        Combine two solved thresholds, enforcing band ordering.

        ``t1 > t2`` means the two bands overlap and the abstain region has
        vanished. That is a real finding about the constraint set being jointly
        slack, so it is raised rather than clipped — clipping would silently
        change the operating point the experiment is reporting.

        Ordering is only meaningful when both thresholds live on the SAME score.
        In the two-pass cascade they do not: ``t1`` cuts the Pass 1 score and
        ``t2`` cuts a different forest's score over a different population. The
        cascade is structurally incapable of overlapping bands, so
        ``RFRouter`` skips this check and ``SingleRFRouter`` applies it.
        """
        pass1.raise_if_infeasible()
        pass2.raise_if_infeasible()

        if self.constraints.require_band_ordering and pass1.threshold > pass2.threshold:
            raise ValueError(
                f"Band ordering violated: t1={pass1.threshold:.6f} > "
                f"t2={pass2.threshold:.6f}. The NOT_SUBSCRIBE and SUBSCRIBE "
                "bands overlap, so there is no abstain region. Both constraint "
                "sets are jointly slack — tighten a floor or report this as the "
                "ablation's finding rather than forcing a band."
            )

        return ThresholdPair(
            t1=float(pass1.threshold),
            t2=float(pass2.threshold),
            pass1=pass1,
            pass2=pass2,
        )


# ======================================================================
# Band enumeration helpers
# ======================================================================

def _lower_band_sizes(p_sorted: np.ndarray) -> np.ndarray:
    """
    Sizes ``k`` for which ``{p < t}`` is exactly the k lowest-scoring rows.

    Only boundaries between DISTINCT probability values are realisable: if
    ``p_sorted[k-1] == p_sorted[k]`` no threshold can split that tie, and
    pretending otherwise would report a band the model cannot actually produce.
    """
    n = len(p_sorted)
    if n == 0:
        return np.empty(0, dtype=int)
    distinct = np.flatnonzero(p_sorted[:-1] < p_sorted[1:]) + 1
    return np.concatenate([distinct, [n]]).astype(int)


def _upper_band_sizes(p_sorted: np.ndarray) -> np.ndarray:
    """Sizes ``m`` for which ``{p > t}`` is exactly the m highest-scoring rows."""
    n = len(p_sorted)
    if n == 0:
        return np.empty(0, dtype=int)
    distinct = n - (np.flatnonzero(p_sorted[:-1] < p_sorted[1:]) + 1)
    return np.sort(np.concatenate([distinct, [n]])).astype(int)


def _upper_edge(p_sorted: np.ndarray, k: int) -> float:
    """Threshold ``t`` with ``{p < t}`` == the k lowest rows."""
    n = len(p_sorted)
    if k >= n:
        return float(np.nextafter(p_sorted[-1], np.inf))
    return float(p_sorted[k])


def _lower_edge(p_sorted: np.ndarray, m: int) -> float:
    """Threshold ``t`` with ``{p > t}`` == the m highest rows."""
    n = len(p_sorted)
    if m >= n:
        return float(np.nextafter(p_sorted[0], -np.inf))
    return float(p_sorted[n - m - 1])


def _prepare(proba, y, label: str):
    p = np.asarray(proba, dtype=float)
    y_arr = np.asarray(y)

    if len(p) != len(y_arr):
        raise ValueError(f"{label}: proba/y length mismatch {len(p)} vs {len(y_arr)}")
    if len(p) == 0:
        raise ValueError(f"{label}: empty population")
    if np.isnan(p).any():
        raise ValueError(
            f"{label}: proba contains NaN. Pass only the SCORED rows "
            "(OOFResult.scored_proba()), not the full padded array."
        )
    if not np.isin(y_arr, (0, 1)).all():
        raise ValueError(f"{label}: y must be binary 0/1")

    return p, y_arr.astype(int)
