"""
PR 39, Commit 2, Step 2B (gate A2.4) on the real training split.

The renamed Stage 1 features take exactly the values the pre-rename code
produced, on GLOBAL_SPLIT: the full-train train and test transforms and every
per-fold design the Stage 1 stages build (10 outer folds, and the 10 inner
folds of outer fold 0; encoders and scaler refit per fold, audit F3/F19).

The anchors were recorded by running the code at ``fb233f9`` + the two Step 2A
patches (names ``cellular_crisis`` / ``euribor3m_local_rate`` /
``dow_month_encoded``) on this split. They are [sum, weighted sum] of each
matrix with the weights ((7 i + 13 j) mod 17) - 8, so each is sensitive to row
and column order; ``tests/stage1/test_stage1_feature_contract.py`` uses the
same summary on a synthetic split. Test labels are never read.

Marked ``realdata`` (it reads ``data/db``), so it runs only under
``pytest -m realdata``. It reads no saved artifact, which is why it sits here
rather than under ``tests/realdata`` (those modules read the PR 37 archive).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from shared.stage1.crossfit import build_fold_designs, inner_fold_designs
from shared.stage1.feature_engineering import LR_FEATURES, LRFeatureEngineer

pytestmark = pytest.mark.realdata

#: The pinned PR 38 population (router split fingerprint of GLOBAL_SPLIT).
ROUTER_SPLIT_FINGERPRINT = "fc3e73a5ef91aea4"
SHAPES = [[32950, 3], [8238, 3]]
FULL_TRAIN = [[16862.264391700246, 218.13685307818753],          # train
              [4200.068589992874, 353.29590033814645]]           # test
FIRST_COLUMN_POSITIVES = [9735, 2433]
#: [X_train sum, X_train weighted, X_val sum, X_val weighted] per fold (scaled).
OUTER = [
    [-3.183231456205249e-12, 2334.1243091069236, -83.69702240670374, -68.58601256208489],
    [7.560174708487466e-12, 635.4799266786329, -70.16855088718017, -500.3294331802647],
    [2.6147972675971687e-12, 724.8461008165211, 356.1829193389438, 290.53944032308266],
    [-6.963318810448982e-12, 1222.4384356778887, -91.00309148908836, -508.8772939925646],
    [-9.606537787476555e-12, -533.2000868621083, 136.66761082201174, 175.3167506354353],
    [1.0857093002414331e-11, 1544.9185433799153, 54.3673831228379, 514.0646100211027],
    [-5.7411853049416095e-12, 1035.5180540760152, -265.14474782906325, 339.6614987952687],
    [5.5706550483591855e-12, 237.2939646292922, 152.317217313475, -430.77833879850175],
    [1.2306600183364935e-11, -1212.4383066442174, -153.5389128351304, 366.5546763260696],
    [3.581135388230905e-12, 989.0368299403221, -54.83163663622096, -427.9742184128843],
]
INNER_OF_OUTER_0 = [
    [6.934897101018578e-12, -1303.886924776691, 11.242549907577192, 391.99199343803184],
    [5.400124791776761e-13, -1630.9783716480522, -218.8191812499972, -469.390010440864],
    [-4.689582056016661e-12, -376.0070481584287, -42.66494508440246, 780.1057946028516],
    [-1.1937117960769683e-12, -283.12121736481714, -95.8858595396525, 88.40413074807262],
    [-8.981260180007666e-12, 1375.2761458745167, 7.9526984634885345, 229.15426589115586],
    [-8.526512829121202e-14, -1406.612416206849, -17.17823391793121, -426.09303553237174],
    [8.697043085703626e-12, -626.3920286681833, 14.638346572279048, 922.5058073838259],
    [-5.115907697472721e-12, 488.91206158974154, 231.52661699182937, -798.4475811259923],
    [-4.718003765447065e-12, 1240.9588002843752, -28.044528362865208, 484.7501763957708],
    [-1.1368683772161603e-13, -413.7066567772863, 104.27617764332557, 13.590578804991736],
]


def _anchor(M) -> list:
    M = np.asarray(M, dtype=float)
    i, j = np.indices(M.shape)
    return [float(M.sum()), float((M * (((i * 7 + j * 13) % 17) - 8.0)).sum())]


def _close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-9, atol=1e-6)


@pytest.fixture(scope="module")
def real():
    from stage2_crossfit.runner import load_train_side
    ts = load_train_side()
    assert ts.router_split_fingerprint == ROUTER_SPLIT_FINGERPRINT
    X_raw, y = ts.X_train, ts.y_train.astype(int)
    with contextlib.redirect_stdout(io.StringIO()):
        eng = LRFeatureEngineer()
        tr = eng.fit_transform(X_raw, y)
        te = eng.transform(ts.X_test)
        outer = build_fold_designs(tr, y, cv_folds=10, random_state=42, X_raw=X_raw)
        inner = inner_fold_designs(outer[0], tr, y, cv_folds=10, random_state=42, X_raw=X_raw)
    return tr, te, outer, inner


def test_full_train_features_equal_the_pre_rename_values(real):
    tr, te, _, _ = real
    assert list(tr.columns) == list(te.columns) == LR_FEATURES
    assert [list(tr.shape), list(te.shape)] == SHAPES
    _close([_anchor(tr), _anchor(te)], FULL_TRAIN)
    assert [int(tr["cellular_downturn"].sum()), int(te["cellular_downturn"].sum())] == \
        FIRST_COLUMN_POSITIVES


def test_per_fold_designs_equal_the_pre_rename_values(real):
    _, _, outer, inner = real
    _close([_anchor(f.X_train) + _anchor(f.X_val) for f in outer], OUTER)
    _close([_anchor(f.X_train) + _anchor(f.X_val) for f in inner], INNER_OF_OUTER_0)
