"""
blackbox_pipeline.models.rf_router.router

Orchestration and the Stage 2 output contract.

Two routers live here:

``SingleRFRouter``
    The ablation. One forest, one score, two thresholds.
``RFRouter``
    The GLASS counterpart. Pass 1 routes NOT_SUBSCRIBE; the remainder, defined
    from OOF Pass 1 decisions, trains the Pass 2 forest; the rest abstains.

Both expose exactly the contract ``GLASSBRWPipeline`` exposes, so
``evaluation.router_metrics.evaluate_router`` consumes either arm unchanged::

    predict(X)       -> (preds, conf, decisions)
                        preds     in {0, 1, -1}                         int
                        conf      in [0, 1]                             float
                        decisions in {"pass1", "pass2", "uncertain"}    object
    predict_proba(X) -> (n, 2) array [P(NOT_SUBSCRIBE), P(SUBSCRIBE)]

Layout
------
``decisions``
    Decision codes and labels — the contract's vocabulary.
``split``
    Global-split guards: fingerprint, disjointness, unpacking.
``base``
    Shared validation, the cascade cut, ``predict_proba``, reporting. The
    train/test protocol is documented there.
``single``
    ``SingleRFRouter`` (ablation).
``two_pass``
    ``RFRouter`` (final). The residual-dependence note is documented there.

Notes
-----
Evaluation is deliberately NOT re-exported here. ``router`` is orchestration and
``evaluation.router_metrics`` measures its output, so importing the latter from
this package points the dependency backwards — and because ``router_metrics``
imports the decision codes from ``decisions``, doing so would make the cycle
real. Import the metrics from where they live::

    from blackbox_pipeline.models.rf_router.evaluation import evaluate_router
"""

from __future__ import annotations

from .decisions import ABSTAIN, NOT_SUBSCRIBE, SUBSCRIBE
from .split import assert_disjoint_split, split_fingerprint, unpack_global_split
from .base import _BaseRouter, _FitReport  # noqa: F401  (re-exported for tests)
from .single import SingleRFRouter
from .two_pass import RFRouter

__all__ = [
    "RFRouter",
    "SingleRFRouter",
    "SUBSCRIBE",
    "NOT_SUBSCRIBE",
    "ABSTAIN",
    "split_fingerprint",
    "assert_disjoint_split",
    "unpack_global_split",
]
