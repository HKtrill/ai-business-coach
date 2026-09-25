"""
blackbox_pipeline.models.rf_router.constraints
===============================================
Stage 2 operating constraints, represented as data.

These are the *set-level* counterparts of the GLASS Stage 2 quality gates.

GLASS applies its gates PER RULE (``QualityGateFilter`` tests each candidate
rule in isolation; nothing in the ILP constrains the union of selected rules).
A threshold on a continuous score has no per-rule analogue — a band is a single
set — so every constraint here is evaluated on the routed SET.

Consequence for the comparison: GLASS's realised set-level leakage can exceed
``max_subscriber_leakage_rate`` by a multiple of the number of selected Pass 1
rules, while the RF router is held to the number exactly. When reporting, treat
the GLASS per-rule caps as *configuration* and compare only the measured
set-level numbers produced by ``evaluation.router_metrics``.

Nothing in this module touches data. It is pure configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Optional

__all__ = ["OperatingConstraints"]


@dataclass(frozen=True)
class OperatingConstraints:
    """
    Set-level operating requirements for the two routing decisions.

    Pass 1 (route NOT_SUBSCRIBE)
    ----------------------------
    min_precision_not_subscribe
        Floor on ``P(y = 0 | routed by Pass 1)``. This is the route's NPV and is
        the direct counterpart of GLASS ``min_precision_not_subscribe``
        (``RuleMetrics.compute_precision`` with ``predicted_class = 0``).
    max_subscriber_leakage_rate
        Cap on ``subscribers_routed / total_subscribers``. Same denominator as
        ``RuleMetrics.compute_subscriber_leakage`` — all positives in the
        evaluation population, not just those in the band.
    max_subscriber_leakage_absolute
        Cap on the absolute count of subscribers routed away.
    min_pass1_coverage
        Optional floor on the fraction of the population Pass 1 must claim.
        GLASS has no set-level equivalent; leave ``None`` for a fair baseline
        and use it only when deliberately matching a GLASS operating point.

    Pass 2 (detect SUBSCRIBE)
    -------------------------
    min_precision_subscribe
        Floor on ``P(y = 1 | flagged by Pass 2)``.
    min_recall_subscribe
        Optional floor on Pass 2 recall. See ``recall_denominator``.
    recall_denominator
        ``"global"``  — denominator is all positives in the training population.
                        Matches GLASS, whose Pass 2 rules are scored by
                        ``RuleEvaluator`` against the whole evaluation set.
        ``"remainder"`` — denominator is positives surviving Pass 1. Useful as a
                        diagnostic; NOT the GLASS-comparable number.

    Shared
    ------
    min_band_support_pass1 / min_band_support_pass2
        Minimum number of samples a band must contain for its threshold to be
        considered feasible. Guards against a threshold parked in the extreme
        tail where empirical precision is estimated from a handful of rows.
        Loosely analogous to GLASS ``min_support_pass1`` / ``min_support_pass2``,
        though GLASS applies those per rule during beam search.
    require_band_ordering
        When True (default) a solved pair must satisfy ``t1 <= t2``. When the
        constraints are jointly slack the bands would otherwise overlap and the
        abstain region would vanish; that is a finding to report, not a state to
        force, so the solver raises instead of clipping.
    """

    # ---- Pass 1: route NOT_SUBSCRIBE -------------------------------------
    min_precision_not_subscribe: float
    max_subscriber_leakage_rate: float
    max_subscriber_leakage_absolute: int
    min_pass1_coverage: Optional[float] = None

    # ---- Pass 2: detect SUBSCRIBE ----------------------------------------
    min_precision_subscribe: float = 0.0
    min_recall_subscribe: Optional[float] = None
    recall_denominator: str = "global"

    # ---- Shared ----------------------------------------------------------
    min_band_support_pass1: int = 100
    min_band_support_pass2: int = 50
    require_band_ordering: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []

        for name in (
            "min_precision_not_subscribe",
            "max_subscriber_leakage_rate",
            "min_precision_subscribe",
        ):
            val = getattr(self, name)
            if val is None or not 0.0 <= float(val) <= 1.0:
                errors.append(f"{name} must be in [0, 1], got {val!r}")

        for name in ("min_pass1_coverage", "min_recall_subscribe"):
            val = getattr(self, name)
            if val is not None and not 0.0 <= float(val) <= 1.0:
                errors.append(f"{name} must be in [0, 1] or None, got {val!r}")

        if int(self.max_subscriber_leakage_absolute) < 0:
            errors.append("max_subscriber_leakage_absolute must be >= 0")

        if self.recall_denominator not in ("global", "remainder"):
            errors.append(
                f"recall_denominator must be 'global' or 'remainder', "
                f"got {self.recall_denominator!r}"
            )

        for name in ("min_band_support_pass1", "min_band_support_pass2"):
            if int(getattr(self, name)) < 1:
                errors.append(f"{name} must be >= 1")

        if errors:
            raise ValueError(
                "OperatingConstraints validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Construction from the GLASS config
    # ------------------------------------------------------------------
    @classmethod
    def from_glass_config(
        cls,
        glass_config: Any,
        *,
        min_pass1_coverage: Optional[float] = None,
        use_recall_floor: bool = True,
        recall_denominator: str = "global",
        min_band_support_pass1: Optional[int] = None,
        min_band_support_pass2: Optional[int] = None,
    ) -> "OperatingConstraints":
        """
        Lift the *declared* GLASS operating point into set-level constraints.

        Reads only the binding GLASS floors and budgets. Deliberately ignored:

        * precision CEILINGS (``max_precision_not_subscribe`` /
          ``max_precision_subscribe``) — anti-degeneracy devices that reject
          suspiciously pure small-support rules. A band has no such failure mode.
        * ``max_recall_subscribe`` — same reasoning.
        * rule-count bounds, novelty ratios, diversity / feature-reuse caps,
          beam width, ``max_complexity``, depth-2 pruning — symbolic search
          scaffolding with no black-box counterpart.

        ``min_support_pass1`` / ``min_support_pass2`` are borrowed as default
        band-support floors because they are the closest thing GLASS has to
        "don't trust a tiny cell", even though GLASS applies them per rule.

        Parameters
        ----------
        glass_config
            A ``GlassRouterConfig`` (or anything exposing the same attributes).
        use_recall_floor
            When False, ``min_recall_subscribe`` is dropped. Use this for the
            unconstrained ablation, where we want the RF's natural operating
            point before pushing it toward GLASS.
        """
        def _get(name: str, default=None):
            return getattr(glass_config, name, default)

        recall_floor = _get("min_recall_subscribe") if use_recall_floor else None

        return cls(
            min_precision_not_subscribe=float(_get("min_precision_not_subscribe")),
            max_subscriber_leakage_rate=float(_get("max_subscriber_leakage_rate")),
            max_subscriber_leakage_absolute=int(_get("max_subscriber_leakage_absolute")),
            min_pass1_coverage=min_pass1_coverage,
            min_precision_subscribe=float(_get("min_precision_subscribe")),
            min_recall_subscribe=None if recall_floor is None else float(recall_floor),
            recall_denominator=recall_denominator,
            min_band_support_pass1=int(
                min_band_support_pass1
                if min_band_support_pass1 is not None
                else _get("min_support_pass1", 100)
            ),
            min_band_support_pass2=int(
                min_band_support_pass2
                if min_band_support_pass2 is not None
                else _get("min_support_pass2", 50)
            ),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OperatingConstraints":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "OperatingConstraints("
            f"npv>={self.min_precision_not_subscribe:.3f}, "
            f"leak<={self.max_subscriber_leakage_rate:.3f}"
            f"/{self.max_subscriber_leakage_absolute}, "
            f"prec1>={self.min_precision_subscribe:.3f}, "
            f"recall>={self.min_recall_subscribe})"
        )
