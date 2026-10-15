"""
blackbox_pipeline.models.rf_router.tuning.diagnostics
======================================================
Gate 0: is there anything for Pass 2 tuning to optimise?

``ThresholdSolver.solve_pass2`` always offers the WHOLE population as a
candidate band, and keeps the largest feasible band. So whenever the remainder's
own base rate already clears the precision floor (and the support and recall
floors), the whole remainder is chosen: ``t2`` falls below every score and Pass 2
flags every remainder row — whatever its hyperparameters.

That condition depends only on the remainder's LABELS and the constraints, never
on Pass 2's scores. ``diagnose_pass2_degeneracy`` therefore decides it exactly
and for free, before a single trial is spent. When it holds, every Pass 2
objective is flat in the routing sense and Pass 2 tuning cannot change a single
decision; the specialization harness raises ``DegenerateObjectiveError`` rather
than burn a budget reporting noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..constraints import OperatingConstraints
from ..reporting import binding_after, pass_agreement

__all__ = [
    "DegenerateObjectiveError",
    "Pass2Degeneracy",
    "diagnose_pass2_degeneracy",
    "diagnose_router",
]

#: Floor-minus-base-rate margin under which Pass 2 is flagged near-degenerate.
NEAR_DEGENERATE_MARGIN = 0.02


class DegenerateObjectiveError(ValueError):
    """Pass 2 tuning requested on a remainder where routing cannot change."""


@dataclass(frozen=True)
class Pass2Degeneracy:
    n_rows: int
    n_positives: int
    base_rate: float
    precision_floor: float
    support_floor: int
    recall_floor: Optional[float]
    recall_whole_remainder: float
    degenerate: bool

    @property
    def margin(self) -> float:
        """Precision floor minus remainder base rate. <= 0 means degenerate."""
        return float(self.precision_floor - self.base_rate)

    @property
    def verdict(self) -> str:
        if self.degenerate:
            return ("DEGENERATE — the whole remainder clears every Pass 2 constraint "
                    f"(base rate {self.base_rate:.4f} >= floor {self.precision_floor:.4f}), "
                    "so t2 falls below every score and Pass 2 flags all of it whatever "
                    "its hyperparameters. Pass 2 tuning cannot change routing here.")
        if self.margin < NEAR_DEGENERATE_MARGIN:
            return ("NEAR-DEGENERATE — the precision floor sits within "
                    f"{NEAR_DEGENERATE_MARGIN:.0%} of the remainder base rate; Pass 2's "
                    "band will be most of the remainder and its objective nearly flat.")
        return ("OK — the precision floor is above the remainder base rate; t2 is a "
                "genuine top-tail cut and Pass 2's ranking decides which rows clear it.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(margin=self.margin, verdict=self.verdict)
        return d


def diagnose_pass2_degeneracy(
    y_remainder,
    constraints: OperatingConstraints,
    n_positives_global: int,
) -> Pass2Degeneracy:
    """
    Exact test of whether ``solve_pass2`` would pick the whole remainder.

    Mirrors the solver's feasibility checks for the band of size ``n``: support,
    precision floor, and (when set) the recall floor under the constraints'
    recall denominator.
    """
    y = np.asarray(y_remainder).astype(int)
    n, pos = len(y), int(y.sum())
    if n == 0:
        raise ValueError("empty remainder")
    base = pos / n
    denom = int(n_positives_global) if constraints.recall_denominator == "global" else pos
    recall_whole = pos / denom if denom else 0.0

    feasible = (
        n >= constraints.min_band_support_pass2
        and base >= constraints.min_precision_subscribe
        and (constraints.min_recall_subscribe is None
             or recall_whole >= constraints.min_recall_subscribe)
    )
    return Pass2Degeneracy(
        n_rows=n, n_positives=pos, base_rate=float(base),
        precision_floor=float(constraints.min_precision_subscribe),
        support_floor=int(constraints.min_band_support_pass2),
        recall_floor=constraints.min_recall_subscribe,
        recall_whole_remainder=float(recall_whole),
        degenerate=bool(feasible),
    )


def diagnose_router(router) -> dict:
    """
    Gate 0 read-out from a FITTED two-pass router. No fitting, no test data.

    Reports where each pass's operating point sits, which constraint stopped each
    band from growing, whether Pass 2 is degenerate, and how much Pass 2's
    ranking differs from Pass 1's on the remainder.
    """
    th = router.thresholds_
    rem = np.asarray(router.remainder_mask_, dtype=bool)
    y = np.asarray(router._y_train_).astype(int)
    c = router.config.constraints
    deg = diagnose_pass2_degeneracy(y[rem], c, int(y.sum()))
    agree = pass_agreement(router)
    return {
        "pass1 coverage (band fraction)": float(th.pass1.band_fraction),
        "pass1 NPV (OOF)": float(th.pass1.band_precision),
        "pass1 leakage rate (OOF)": float(th.pass1.leakage_rate),
        "pass1 binding next": binding_after(th.pass1.sweep, th.pass1.band_size),
        "remainder fraction": float(rem.mean()),
        "remainder base rate": deg.base_rate,
        "pass2 precision floor": deg.precision_floor,
        "pass2 floor − base rate": deg.margin,
        "pass2 band fraction (of remainder)": float(th.pass2.band_fraction),
        "pass2 recall (OOF)": float(th.pass2.recall),
        "pass2 binding next": binding_after(th.pass2.sweep, th.pass2.band_size),
        "pass2 degenerate": deg.degenerate,
        "spearman(p1, p2) on remainder": agree["spearman(p1, p2) on remainder"],
        "pass2 agreement with p1 top-k": agree["agreement"],
        "pass2 overlap with p1 top-k": agree["overlap with p1 top-k"],
        "agreement status": agree["status"],
        "verdict": deg.verdict,
    }
