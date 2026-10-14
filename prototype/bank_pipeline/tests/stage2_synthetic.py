"""
Synthetic Stage 2 fixtures shared by the PR 35 tests.

Binary features (the router refuses anything else), a shuffled non-RangeIndex
(so positional/label mix-ups fail loudly), and two constraint sets on the same
data:

* ``CONSTRAINTS_OK``         — Pass 2 precision floor ABOVE the remainder base
                               rate, so ``t2`` is a genuine top-tail cut.
* ``CONSTRAINTS_DEGENERATE`` — floor BELOW the remainder base rate, so the
                               whole remainder is the largest feasible band and
                               ``t2`` lands below every score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox_pipeline.models.rf_router import OperatingConstraints, RFRouterConfig

ROUTER_SEED = 7
N_FOLDS = 3

SMALL_PARAMS = {
    "n_estimators": 25,
    "max_depth": None,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "minority_weight": 1.0,
    "class_weight": "balanced",
}

ALT_PARAMS = {  # deliberately different capacity -> a different remainder
    "n_estimators": 25,
    "max_depth": 2,
    "min_samples_leaf": 40,
    "max_features": 0.5,
    "minority_weight": 2.0,
    "class_weight": "balanced",
}

_BASE = dict(
    min_precision_not_subscribe=0.86,
    max_subscriber_leakage_rate=0.30,
    max_subscriber_leakage_absolute=10_000,
    min_band_support_pass1=50,
    min_band_support_pass2=30,
)
CONSTRAINTS_OK = OperatingConstraints(min_precision_subscribe=0.50, **_BASE)
CONSTRAINTS_DEGENERATE = OperatingConstraints(min_precision_subscribe=0.30, **_BASE)


def make_data(n: int = 1500, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.Index(rng.permutation(np.arange(10_000, 10_000 + n)), name="row")
    X = pd.DataFrame(
        rng.integers(0, 2, size=(n, p)),
        columns=[f"b{i}" for i in range(p)],
        index=idx,
    )
    logit = (-2.2 + 1.6 * X.b0 + 1.1 * X.b1 - 0.9 * X.b2
             + 0.8 * X.b3 * X.b4 + 0.5 * X.b5)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), index=idx)
    return X, y


def router_config(constraints=CONSTRAINTS_OK, mode="two_pass", **overrides):
    kw = dict(
        mode=mode,
        constraints=constraints,
        rf_params=SMALL_PARAMS,
        n_oof_folds=N_FOLDS,
        random_state=ROUTER_SEED,
        remainder_min_size=100,
        verbose=False,
    )
    kw.update(overrides)
    return RFRouterConfig(**kw)
