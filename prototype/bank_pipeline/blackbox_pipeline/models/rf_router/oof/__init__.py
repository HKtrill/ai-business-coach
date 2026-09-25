"""
blackbox_pipeline.models.rf_router.oof
======================================
Leak-free out-of-fold probability generation.

Every threshold in this router is selected from the arrays this package
produces, which makes it the most leakage-sensitive code here. The invariants
are asserted at construction rather than inspected later; they are listed on
``scorer.OOFScorer`` and enforced by ``results.OOFResult.assert_valid``.

Why OOF at all
--------------
``train_rf_stage`` refits on the full ``X_train``, and in-sample
``predict_proba`` on a forest of that depth is close to separating. Thresholds
read off those numbers would place ``t1`` and ``t2`` almost on top of each
other, produce a near-zero abstain band on train, and collapse on test. OOF
probabilities are what the model would have said about a row it had not seen,
which is the right basis for choosing an operating point.

This is not an advantage over GLASS. GLASS selects its rules against
``X_val``/``y_val``, which ``GlassRouterPipeline.fit`` defaults to the training
data when no validation split is passed. Using OOF here gives the RF a *less*
optimistic view of its own training data than GLASS grants itself. If the GLASS
arm is later re-run with a genuine validation split, revisit this note; the
comparison stays honest either way.

Layout
------
``folds``    row-alignment guard and positional fold construction
``results``  ``OOFResult`` and its structural invariants
``scorer``   ``OOFScorer`` — the single- and subset-population OOF
``nested``   the fully nested cascade diagnostic
"""

from __future__ import annotations

from .folds import check_aligned, make_folds
from .nested import NestedCascadeResult, nested_cascade_oof
from .results import OOFResult
from .scorer import OOFScorer

__all__ = [
    "OOFResult",
    "OOFScorer",
    "make_folds",
    "nested_cascade_oof",
    "NestedCascadeResult",
]

# Back-compat: ``_check_aligned`` was module-private in the old oof.py.
# Safe to delete once nothing imports it by the old name.
_check_aligned = check_aligned