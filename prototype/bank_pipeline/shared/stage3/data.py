"""
shared.stage3.data
==================
The exact data / fold / feature contract Stage 3 sees.

* ``FoldPlan``               frozen fold assignment (``shared.stage_io.fold_assignment``)
* ``BalancedWeights``        per-row class-balanced sample weights
* ``Stage3FeatureContract``  column-set / order agreement with the EBM block
* ``FeatureCleaner``         inf -> NaN -> train-median imputation
* ``PassthroughPipeline``    identity pipeline for pre-engineered frames
* ``FoldFrames``             one fold's model-ready frames
* ``Stage3Block``            everything both arms consume, fingerprinted
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from shared.stage_io import fold_assignment


@dataclass(frozen=True)
class FoldPlan:
    """A frozen fold assignment over ``range(n)``."""

    fold_id: np.ndarray
    n_splits: int
    random_state: int
    stratified: bool
    purpose: str = "tuning+oof"

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        y,
        n_splits: int,
        random_state: int,
        stratify: bool = True,
        purpose: str = "tuning+oof",
    ) -> "FoldPlan":
        y_arr = np.asarray(y)
        n = len(y_arr)
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_splits > n:
            raise ValueError(
                f"n_splits={n_splits} exceeds the population size {n}"
            )
        if stratify:
            counts = np.bincount(y_arr.astype(int))
            smallest = int(counts[counts > 0].min())
            if smallest < n_splits:
                raise ValueError(
                    f"Stratified {n_splits}-fold needs at least {n_splits} "
                    f"samples in every class; the smallest has {smallest}."
                )

        if not stratify:
            raise ValueError(
                "Stage 3 folds come from shared.stage_io.fold_assignment, "
                "which is stratified (as in Stages 1–2). stratify=False is "
                "not supported."
            )
        # One fold splitter for the whole cascade: the same function Stages
        # 1–2 use, so fold ids are identical whenever n_splits and seed match.
        fold_id = fold_assignment(
            y_arr, cv_folds=n_splits, random_state=random_state
        ).to_numpy(dtype=int)

        plan = cls(
            fold_id=fold_id, n_splits=n_splits, random_state=random_state,
            stratified=bool(stratify), purpose=purpose,
        )
        plan.assert_valid()
        return plan

    # ------------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.fold_id)

    def splits(self) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Positional ``(train_idx, val_idx)`` pairs, derived from fold_id."""
        return list(self)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for k in range(self.n_splits):
            val = np.flatnonzero(self.fold_id == k)
            train = np.flatnonzero(self.fold_id != k)
            yield train, val

    def holdout(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.fold_id == k)

    # ------------------------------------------------------------------
    def assert_valid(self) -> None:
        if (self.fold_id < 0).any():
            raise AssertionError(
                f"{int((self.fold_id < 0).sum())} rows carry no fold assignment."
            )
        counts = np.bincount(self.fold_id, minlength=self.n_splits)
        if len(counts) != self.n_splits or (counts == 0).any():
            raise AssertionError(
                f"Expected {self.n_splits} non-empty folds, got {counts.tolist()}"
            )
        if int(counts.sum()) != self.n_rows:
            raise AssertionError("Folds do not partition the population.")

    def assert_matches(self, n_rows: int, context: str = "") -> None:
        """Guard against a fold plan being reused against the wrong frame."""
        if n_rows != self.n_rows:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"FoldPlan covers {self.n_rows} rows but was handed "
                f"{n_rows}{where}. The plan is built from the training split "
                "and must not be reused across populations."
            )

    # ------------------------------------------------------------------
    def class_balance(self, y) -> list[dict]:
        """Per-fold positive rate — printed at fit time, stored in the artifact."""
        y_arr = np.asarray(y).astype(int)
        rows = []
        for k, (train, val) in enumerate(self):
            rows.append({
                "fold": k,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "val_positive_rate": float(y_arr[val].mean()),
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "stratified": self.stratified,
            "purpose": self.purpose,
            "n_rows": self.n_rows,
            "fold_id": self.fold_id.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldPlan":
        return cls(
            fold_id=np.asarray(d["fold_id"], dtype=int),
            n_splits=int(d["n_splits"]),
            random_state=int(d["random_state"]),
            stratified=bool(d["stratified"]),
            purpose=d.get("purpose", "tuning+oof"),
        )


@dataclass
class BalancedWeights:
    """Per-row sample weights, computed once and sliced positionally."""

    vector: np.ndarray
    strategy: str
    positive_weight: float
    negative_weight: float
    ratio: float
    n_rows: int

    # ------------------------------------------------------------------
    @classmethod
    def balanced(cls, y, strategy: str = "balanced") -> "BalancedWeights":
        y_arr = np.asarray(y).astype(int)
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        if y_arr.min() == y_arr.max():
            raise ValueError("y is single-class; cannot balance")

        vector = np.asarray(compute_sample_weight(strategy, y_arr), dtype=float)
        pos = float(vector[y_arr == 1].mean())
        neg = float(vector[y_arr == 0].mean())
        return cls(
            vector=vector,
            strategy=strategy,
            positive_weight=pos,
            negative_weight=neg,
            ratio=float(pos / neg) if neg else float("nan"),
            n_rows=len(y_arr),
        )

    # ------------------------------------------------------------------
    def for_rows(self, idx: np.ndarray) -> np.ndarray:
        """Slice by positional index — the EBM's ``sample_weights[tr_idx]``."""
        return self.vector[np.asarray(idx, dtype=int)]

    def assert_matches(self, n_rows: int, context: str = "") -> None:
        if n_rows != self.n_rows:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"Weights cover {self.n_rows} rows but were handed "
                f"{n_rows}{where}."
            )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "ratio": self.ratio,
            "n_rows": self.n_rows,
        }

    def describe(self) -> str:
        return (
            f"balanced sample weights: class 0 → {self.negative_weight:.4f}, "
            f"class 1 → {self.positive_weight:.4f} "
            f"(positive class upweighted {self.ratio:.1f}×)"
        )


class Stage3FeatureContract:
    """
    Asserts that the Stage 3 input matches the EBM's feature block.

    Parameters
    ----------
    expected_features
        The column list the EBM consumed — pass
        ``glass_pipeline.ebm.feature_engineering.EBM_FEATURES``. When ``None``
        the contract only checks train/test agreement, which is weaker; the
        stage warns in that case.
    """

    def __init__(self, expected_features: Optional[list[str]] = None):
        self.expected_features = (
            None if expected_features is None else list(expected_features)
        )

    # ------------------------------------------------------------------
    def validate(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
        """
        Return the agreed column order, or raise explaining the mismatch.

        Order is ``expected_features`` when set, else ``X_train``'s order.
        Raises on non-DataFrame input, duplicate columns, train/test set
        mismatch, or any missing / extra column vs. the contract.
        """
        for name, frame in (("X_train", X_train), ("X_test", X_test)):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"{name} must be a DataFrame, got {type(frame).__name__}"
                )

        train_cols, test_cols = list(X_train.columns), list(X_test.columns)

        for name, cols in (("X_train", train_cols), ("X_test", test_cols)):
            dupes = sorted({c for c in cols if cols.count(c) > 1})
            if dupes:
                raise ValueError(f"{name} has duplicate columns: {dupes}")

        if set(train_cols) != set(test_cols):
            only_train = sorted(set(train_cols) - set(test_cols))
            only_test = sorted(set(test_cols) - set(train_cols))
            raise ValueError(
                "Stage 3 train/test columns differ.\n"
                f"  only in train: {only_train}\n"
                f"  only in test : {only_test}"
            )

        if self.expected_features is None:
            return train_cols

        expected = self.expected_features
        missing = [c for c in expected if c not in train_cols]
        extra = [c for c in train_cols if c not in expected]
        if missing or extra:
            raise ValueError(
                "Stage 3 input does not match the EBM feature contract — the "
                "two arms would not be seeing the same information.\n"
                f"  expected {len(expected)} columns from EBM_FEATURES\n"
                f"  missing  : {missing}\n"
                f"  unexpected: {extra}\n"
                "Re-run the GLASS engineering DAG (drop_leaky_features -> "
                "engineer_ebm_features -> select_ebm_features) and pass its "
                "output, or update config.expected_features deliberately."
            )
        return list(expected)

    # ------------------------------------------------------------------
    def align(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Validate, then return both frames in the contracted column order."""
        order = self.validate(X_train, X_test)
        return X_train[order], X_test[order], order


@dataclass
class FeatureCleaner:
    """
    inf -> NaN -> median impute, with medians learned on the training split.

    Used by both arms (PR 33, audit item 6). Replaces the old guard inside
    ``tune_ebm``, which cleaned only the tuning copy of ``X_train`` and never
    ``X_test``. Medians are fitted on the rows a model trains on (the full
    train split, or the fold's training rows) and reused on every frame that
    model scores; the fitted cleaner is saved in the artifact.

    Assumes numeric columns. Raises if inf remains (e.g. ``clean_inf=False``)
    or if NaN remains after imputation (e.g. a column all-NaN in train).
    ``report_`` lists only columns with NaN in TRAIN.
    """

    clean_inf: bool = True
    impute_missing: bool = True
    medians_: Optional[pd.Series] = field(default=None, repr=False)
    columns_: Optional[list[str]] = field(default=None, repr=False)
    report_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "FeatureCleaner":
        frame = self._replace_inf(X)
        self.columns_ = list(frame.columns)
        self.medians_ = frame.median(numeric_only=False)
        n_inf = self._count_inf(X)
        nan_cols = frame.columns[frame.isna().any()].tolist()
        self.report_ = {
            "n_inf_replaced": n_inf,
            "columns_imputed": nan_cols,
            "medians": {c: float(self.medians_[c]) for c in nan_cols},
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise ValueError("Call fit() first")
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Frame is missing fitted columns: {missing}")

        frame = self._replace_inf(X[self.columns_])
        if self.impute_missing and frame.isna().any().any():
            frame = frame.fillna(self.medians_)

        if self._count_inf(frame):
            raise AssertionError("Infinity values remain after cleaning")
        if self.impute_missing and frame.isna().any().any():
            raise AssertionError("NaN values remain after cleaning")
        return frame

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.clean_inf:
            return X.copy()
        return X.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _count_inf(X: pd.DataFrame) -> int:
        """Count ±inf across numeric columns."""
        numeric = X.select_dtypes(include=["number"])
        if numeric.empty:
            return 0
        return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


class PassthroughPipeline:
    """Identity 'pipeline' for frames that are already engineered."""

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


@dataclass
class FoldFrames:
    """One fold's model-ready frames (engineered + cleaned)."""

    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_tr: pd.DataFrame
    y_tr: pd.Series
    w_tr: np.ndarray
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: Optional[pd.DataFrame] = None
    fitted_on_n_rows: int = 0


def _hash_frame(h, frame: pd.DataFrame) -> None:
    h.update("|".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())


@dataclass
class Stage3Block:
    """Everything both Stage 3 arms consume. Build with ``Stage3Block.build``."""

    features: list[str]
    feature_fit_scope: str
    X_train: pd.DataFrame           # full-train pipeline, cleaned
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Any                   # fitted on all of train (serving)
    cleaner: FeatureCleaner         # fitted on all of train (serving)
    weights: BalancedWeights
    tune_plan: FoldPlan
    report_plan: FoldPlan
    tune_folds: list[FoldFrames]
    report_folds: list[FoldFrames]
    split_fingerprint: Optional[str] = None
    block_fingerprint: str = ""
    settings: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def index_train(self) -> pd.Index:
        return self.X_train.index

    @property
    def index_test(self) -> pd.Index:
        return self.X_test.index

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        X_train_raw: pd.DataFrame,
        y_train,
        X_test_raw: pd.DataFrame,
        y_test,
        pipeline_factory: Callable[[], Any],
        *,
        feature_fit_scope: str = "per_fold",
        n_tune_folds: int = 5,
        n_eval_folds: int = 10,
        random_state: int = 42,
        stratify: bool = True,
        class_weight: str = "balanced",
        clean_inf: bool = True,
        impute_missing: bool = True,
        expected_features: Optional[list[str]] = None,
        split_fingerprint: Optional[str] = None,
        verbose: bool = True,
    ) -> "Stage3Block":
        say = print if verbose else (lambda *a, **k: None)
        if feature_fit_scope not in ("per_fold", "global"):
            raise ValueError(f"unknown feature_fit_scope {feature_fit_scope!r}")

        y_tr = _as_label_series(y_train, X_train_raw, "y_train")
        y_te = _as_label_series(y_test, X_test_raw, "y_test")
        contract = Stage3FeatureContract(expected_features)

        # ---- full-train fit: refit model + test split ------------------
        say(f"  Stage 3 block: fitting feature pipeline on all "
            f"{len(X_train_raw):,} training rows")
        pipe = pipeline_factory()
        Xtr = pipe.fit_transform(X_train_raw.copy(), y_tr)
        Xte = pipe.transform(X_test_raw.copy())
        _assert_index(Xtr, X_train_raw, "full-train fit_transform")
        _assert_index(Xte, X_test_raw, "full-train transform(test)")
        Xtr, Xte, order = contract.align(Xtr, Xte)
        cleaner = FeatureCleaner(clean_inf, impute_missing)
        Xtr = cleaner.fit_transform(Xtr)
        Xte = cleaner.transform(Xte)

        # ---- weights + folds (depend only on y, n and the seed) --------
        weights = BalancedWeights.balanced(y_tr, class_weight)
        tune_plan = FoldPlan.build(
            y_tr, n_tune_folds, random_state, stratify,
            purpose="tuning+oof+calibration+threshold",
        )
        report_plan = FoldPlan.build(
            y_tr, n_eval_folds, random_state, stratify, purpose="reporting",
        )

        # ---- per-fold frames --------------------------------------------
        def fold_frames(plan: FoldPlan, with_test: bool) -> list[FoldFrames]:
            out = []
            for k, (tr, va) in enumerate(plan):
                if feature_fit_scope == "per_fold":
                    p = pipeline_factory()
                    Xa_raw = X_train_raw.iloc[tr]
                    Xb_raw = X_train_raw.iloc[va]
                    Xa = p.fit_transform(Xa_raw.copy(), y_tr.iloc[tr])
                    Xb = p.transform(Xb_raw.copy())
                    _assert_index(Xa, Xa_raw, f"fold {k} fit_transform")
                    _assert_index(Xb, Xb_raw, f"fold {k} transform(val)")
                    Xt = None
                    if with_test:
                        Xt = p.transform(X_test_raw.copy())
                        _assert_index(Xt, X_test_raw, f"fold {k} transform(test)")
                    Xa, Xb, fold_order = contract.align(Xa, Xb)
                    if set(fold_order) != set(order):
                        raise ValueError(
                            f"fold {k}: pipeline produced columns "
                            f"{sorted(set(fold_order) ^ set(order))} that "
                            "differ from the full-train fit"
                        )
                    Xa, Xb = Xa[order], Xb[order]
                    c = FeatureCleaner(clean_inf, impute_missing)
                    Xa = c.fit_transform(Xa)
                    Xb = c.transform(Xb)
                    if Xt is not None:
                        Xt = c.transform(Xt[order])
                else:
                    Xa, Xb = Xtr.iloc[tr], Xtr.iloc[va]
                    Xt = Xte if with_test else None
                out.append(FoldFrames(
                    k=k, train_idx=tr, val_idx=va,
                    X_tr=Xa, y_tr=y_tr.iloc[tr], w_tr=weights.for_rows(tr),
                    X_val=Xb, y_val=y_tr.iloc[va], X_test=Xt,
                    fitted_on_n_rows=int(len(tr)),
                ))
            return out

        say(f"  Stage 3 block: feature_fit_scope={feature_fit_scope!r} — "
            + ("refitting the feature pipeline inside each of "
               f"{n_tune_folds} tuning + {n_eval_folds} reporting folds"
               if feature_fit_scope == "per_fold"
               else "slicing the full-train features (LEAKY — comparison only)"))
        tune_folds = fold_frames(tune_plan, with_test=True)
        report_folds = fold_frames(report_plan, with_test=False)

        settings = {
            "feature_fit_scope": feature_fit_scope,
            "n_tune_folds": int(n_tune_folds),
            "n_eval_folds": int(n_eval_folds),
            "random_state": int(random_state),
            "stratify": bool(stratify),
            "class_weight": class_weight,
            "clean_inf": bool(clean_inf),
            "impute_missing": bool(impute_missing),
            "pipeline": type(pipe).__module__ + "." + type(pipe).__qualname__,
        }
        block = cls(
            features=list(order), feature_fit_scope=feature_fit_scope,
            X_train=Xtr, X_test=Xte, y_train=y_tr, y_test=y_te,
            pipeline=pipe, cleaner=cleaner, weights=weights,
            tune_plan=tune_plan, report_plan=report_plan,
            tune_folds=tune_folds, report_folds=report_folds,
            split_fingerprint=split_fingerprint, settings=settings,
        )
        block.block_fingerprint = block.compute_fingerprint()
        say(f"  Stage 3 block: {len(order)} features, train {Xtr.shape}, "
            f"test {Xte.shape}, fingerprint {block.block_fingerprint[:16]}…")
        return block

    # ------------------------------------------------------------------
    @classmethod
    def from_global_split(
        cls,
        GLOBAL_SPLIT: dict,
        pipeline_factory: Callable[[], Any],
        config,
        split_fingerprint: Optional[str] = None,
    ) -> "Stage3Block":
        """Build with the fold / weight / hygiene settings of a Stage3Config."""
        missing = [k for k in ("X_train", "X_test", "y_train", "y_test")
                   if k not in GLOBAL_SPLIT]
        if missing:
            raise KeyError(f"GLOBAL_SPLIT missing keys: {missing}")
        return cls.build(
            GLOBAL_SPLIT["X_train"], GLOBAL_SPLIT["y_train"],
            GLOBAL_SPLIT["X_test"], GLOBAL_SPLIT["y_test"],
            pipeline_factory,
            feature_fit_scope=config.feature_fit_scope,
            n_tune_folds=config.n_tune_folds,
            n_eval_folds=config.n_eval_folds,
            random_state=config.random_state,
            stratify=config.stratify,
            class_weight=config.class_weight,
            clean_inf=config.clean_inf,
            impute_missing=config.impute_missing,
            expected_features=config.expected_features,
            split_fingerprint=split_fingerprint,
            verbose=config.verbose,
        )

    # ------------------------------------------------------------------
    def compute_fingerprint(self) -> str:
        """SHA-256 over every frame, label vector, fold assignment and weight."""
        h = hashlib.sha256()
        h.update(self.feature_fit_scope.encode())
        _hash_frame(h, self.X_train)
        _hash_frame(h, self.X_test)
        h.update(np.asarray(self.y_train, dtype=np.int64).tobytes())
        h.update(np.asarray(self.y_test, dtype=np.int64).tobytes())
        h.update(self.weights.vector.tobytes())
        h.update(self.tune_plan.fold_id.astype(np.int64).tobytes())
        h.update(self.report_plan.fold_id.astype(np.int64).tobytes())
        for f in self.tune_folds + self.report_folds:
            _hash_frame(h, f.X_tr)
            _hash_frame(h, f.X_val)
            if f.X_test is not None:
                _hash_frame(h, f.X_test)
        return h.hexdigest()

    def check_config(self, config) -> None:
        """Raise if a Stage3Config disagrees with how this block was built."""
        pairs = {
            "feature_fit_scope": config.feature_fit_scope,
            "n_tune_folds": config.n_tune_folds,
            "n_eval_folds": config.n_eval_folds,
            "random_state": config.random_state,
            "stratify": config.stratify,
            "class_weight": config.class_weight,
            "clean_inf": config.clean_inf,
            "impute_missing": config.impute_missing,
        }
        bad = {k: (self.settings.get(k), v) for k, v in pairs.items()
               if self.settings.get(k) != v}
        if config.expected_features is not None and \
                list(config.expected_features) != list(self.features):
            bad["expected_features"] = (self.features, config.expected_features)
        if bad:
            raise ValueError(
                "Stage3Block was built with different settings from the "
                "config:\n" + "\n".join(
                    f"  {k}: block={b!r} config={c!r}" for k, (b, c) in bad.items())
            )

    def describe(self) -> dict:
        return {
            **self.settings,
            "features": list(self.features),
            "n_train": int(len(self.X_train)),
            "n_test": int(len(self.X_test)),
            "split_fingerprint": self.split_fingerprint,
            "block_fingerprint": self.block_fingerprint,
        }


def _as_label_series(y, X: pd.DataFrame, name: str) -> pd.Series:
    if isinstance(y, pd.Series):
        if len(y) != len(X) or not y.index.equals(X.index):
            raise ValueError(
                f"{name}.index differs from its feature frame. Reindex "
                f"before calling ({name} = {name}.loc[X.index])."
            )
        out = y
    else:
        arr = np.asarray(y)
        if len(arr) != len(X):
            raise ValueError(f"{name} length {len(arr)} != {len(X)} rows")
        out = pd.Series(arr, index=X.index)
    if not np.isin(np.asarray(out), (0, 1)).all():
        raise ValueError(f"{name} must be binary 0/1")
    return out.astype(int)


def _assert_index(out: pd.DataFrame, ref: pd.DataFrame, where: str) -> None:
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"{where}: pipeline must return a DataFrame")
    if not out.index.equals(ref.index):
        raise ValueError(
            f"{where}: feature pipeline changed the row index/order. Stage 3 "
            "requires index-preserving transforms."
        )
