"""
The objects a solve produces: one threshold, or a frozen pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

__all__ = ["ThresholdResult", "ThresholdPair"]


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

    @classmethod
    def infeasible(
        cls,
        pass_name: str,
        reason: str,
        sweep: Optional[pd.DataFrame] = None,
    ) -> "ThresholdResult":
        """No band satisfied the constraints. ``sweep`` shows which one binds."""
        return cls(
            threshold=float("nan"),
            feasible=False,
            reason=reason,
            band_size=0,
            band_fraction=0.0,
            band_precision=0.0,
            positives_in_band=0,
            leakage_rate=0.0,
            recall=0.0,
            pass_name=pass_name,
            sweep=sweep,
        )

    def raise_if_infeasible(self, context: str = "") -> None:
        if not self.feasible:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"No feasible {self.pass_name} threshold{where}: {self.reason}\n"
                f"Inspect result.sweep to see which constraint binds, then "
                f"either relax the constraint or accept a smaller band."
            )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "sweep"}


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