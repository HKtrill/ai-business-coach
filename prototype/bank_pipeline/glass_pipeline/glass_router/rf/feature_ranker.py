"""
glass_pipeline.glass_router.rf.feature_ranker
=============================================
Fold-nested RF feature rankers for GLASS rule generation (PR 37 Commit 3, F17).

GLASS orders its symbolic rule search by an RF's ``feature_importances_``
(``RuleGenerator._get_feature_order``). In ``crossfit_stage2`` that RF is the
outer fold's RF, fitted on every outer-training row. Under
``pass2_population="oof_remainder"`` the same RF historically also ordered each
inner-fold Pass 1 search, so the ranker had seen the inner held-out rows those
searches later route.

``make_rf_feature_ranker_factory`` returns a factory that fits a fresh RF with
the SAME recipe as the cross-fit's fold RF (``_build_pipe`` plus
``_make_sample_weights``, the tuned params and seed) on exactly the rows it is
given. ``GlassRouterPipeline`` calls it once per inner fold when
``config.inner_feature_ranker == "fold_nested_rf"``. The RF stays GLASS's own:
it is fitted inside the GLASS arm and never shared with the comparison arm.
Only the training rows change. The rule generator and the ILP are untouched.

Scope (PR 37 F17, limitation L1): each inner ranker is never TRAINED on the
rows its fold's rules route. The isolation is router-level and conditional on
the engineered features, which the cross-fit builds once per outer fold from
the whole outer-training partition and its labels. Mandatory PR 38 gate G13
audits whether a row's own label reaches those features.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from .rf_training import _build_pipe, _make_sample_weights

__all__ = ["make_rf_feature_ranker_factory"]


def make_rf_feature_ranker_factory(
    rf_params: Dict[str, Any],
    random_state: int,
    *,
    n_jobs: int = -1,
    build_pipe: Optional[Callable[..., Any]] = None,
) -> Callable[[pd.DataFrame, pd.Series], Any]:
    """
    ``(X, y) -> fitted RandomForestClassifier``, the fold-RF recipe on (X, y) only.

    Parameters
    ----------
    rf_params
        ``rf_result.params``, the same dict ``crossfit_stage2`` fits each fold RF with.
    random_state
        The cross-fit seed. Every ranker uses it, so a ranker is a deterministic
        function of its training rows.
    n_jobs
        Passed to ``_build_pipe``. Does not change the fitted forest.
    build_pipe
        Test hook; defaults to ``rf_training._build_pipe``.
    """
    builder = build_pipe or _build_pipe
    minority_weight = rf_params.get("minority_weight", 1.0)

    def fit_ranker(X: pd.DataFrame, y: pd.Series):
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("feature ranker: y must be a Series index-aligned with X")
        pipe = builder(rf_params, random_state, n_jobs=n_jobs)
        pipe.fit(X, y, clf__sample_weight=_make_sample_weights(y, minority_weight))
        return pipe.named_steps["clf"]

    fit_ranker.recipe = {
        "builder": "glass_pipeline.glass_router.rf.rf_training._build_pipe",
        "sample_weights": "rf_training._make_sample_weights(minority_weight)",
        "random_state": int(random_state),
        "rf_params": dict(rf_params),
    }
    return fit_ranker
