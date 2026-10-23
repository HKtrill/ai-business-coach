"""
BankPreprocessor: raw Bank Marketing rows -> the numeric frame both cascades use.

Uses small synthetic raw frames (and a temporary SQLite file where the
database path matters); the research dataset is never read or written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.bank_database import build_database
from data.preprocessing import BankPreprocessor
from tests.support import mutations as mut
from tests.support.synthetic import BANK_SCHEMA, TARGET, irregular_index

JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown"]
LEAKY = ["duration", "pdays", "poutcome"]


def raw_bank(n: int = 60, seed: int = 0, jobs=JOBS) -> pd.DataFrame:
    """Raw rows with the CSV's columns and vocabularies, on an irregular index."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "job": rng.choice(jobs, n),
        "marital": rng.choice(["divorced", "married", "single", "unknown"], n),
        "education": rng.choice(["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
                                 "professional.course", "university.degree", "unknown"], n),
        "default": rng.choice(["no", "yes", "unknown"], n),
        "housing": rng.choice(["no", "yes", "unknown"], n),
        "loan": rng.choice(["no", "yes", "unknown"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], n),
        "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "duration": rng.integers(0, 1000, n),
        "campaign": rng.integers(1, 10, n),
        "pdays": rng.choice([999, 3, 6], n),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
        "emp.var.rate": rng.uniform(-3.4, 1.4, n).round(1),
        "cons.price.idx": rng.uniform(92.2, 94.8, n).round(3),
        "cons.conf.idx": rng.uniform(-50.8, -26.9, n).round(1),
        "euribor3m": rng.uniform(0.6, 5.0, n).round(3),
        "nr.employed": rng.uniform(4963.6, 5228.1, n).round(1),
        "y": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
    }, index=irregular_index(n, seed))


def preprocess(df, **kwargs):
    return BankPreprocessor(verbose=False, **kwargs).fit_transform(df)


# ----------------------------------------------------------------------
# output contract
# ----------------------------------------------------------------------
def test_output_schema_order_and_dtypes():
    out = preprocess(raw_bank())
    assert list(out.columns) == [*BANK_SCHEMA, TARGET]
    assert out.drop(columns=TARGET).dtypes.astype(str).to_dict() == BANK_SCHEMA
    assert out[TARGET].dtype == "int8"
    assert out.notna().all().all()


def test_leaky_features_are_dropped_by_default():
    raw = raw_bank()
    assert not set(LEAKY) & set(preprocess(raw).columns)
    kept = preprocess(raw, drop_leaky=False)
    assert set(LEAKY) <= set(kept.columns)
    assert BankPreprocessor.LEAKY_FEATURES == LEAKY


def test_rows_keep_their_index_and_order():
    raw = raw_bank()
    out = preprocess(raw)
    assert out.index.equals(raw.index)
    np.testing.assert_array_equal(out["age"].to_numpy(), raw["age"].to_numpy())


def test_preprocessing_is_deterministic():
    raw = raw_bank()
    pd.testing.assert_frame_equal(preprocess(raw), preprocess(raw))


def test_encodings_match_the_documented_mappings():
    raw = raw_bank(n=2)
    raw.loc[raw.index[0], ["job", "marital", "education", "default", "housing", "loan",
                           "contact", "month", "day_of_week", "y"]] = [
        "student", "single", "unknown", "no", "yes", "unknown", "telephone", "may", "fri", "yes"]
    raw.loc[raw.index[1], "job"] = "admin."
    row = preprocess(raw).iloc[0]
    expected = {"job": 1, "marital": 2, "education": -1, "default": 0, "housing": 1,
                "loan": -1, "contact": 1, "month": 5, "day_of_week": 4, "y": 1}
    assert {k: int(row[k]) for k in expected} == expected   # job: sorted {admin., student}


@pytest.mark.parametrize("column, value, message", [
    ("month", "smarch", "Unmapped month"),
    ("marital", "complicated", "Unmapped marital values"),
    ("y", "maybe", "Unexpected values in target"),
])
def test_unmapped_categories_are_rejected(column, value, message):
    raw = raw_bank()
    raw.loc[raw.index[3], column] = value
    with pytest.raises(AssertionError, match=message):
        preprocess(raw)


def test_transform_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        BankPreprocessor(verbose=False).transform(raw_bank())


def test_missing_database_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_database"):
        BankPreprocessor(verbose=False, db_path=tmp_path / "absent.db").fit()


def test_database_and_dataframe_entry_points_agree(tmp_path):
    """fit_transform() on raw_bank equals fit_transform(df) on the same rows."""
    raw = raw_bank()
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, sep=";", index=False)
    db = build_database(csv_path=csv, db_path=tmp_path / "bank.db")
    from_db = BankPreprocessor(verbose=False, db_path=db).fit_transform()
    from_df = preprocess(raw)
    assert isinstance(from_db.index, pd.RangeIndex)        # CSV row position
    pd.testing.assert_frame_equal(from_db, from_df.reset_index(drop=True))


# ----------------------------------------------------------------------
# F5: where the job vocabulary comes from
# ----------------------------------------------------------------------
def test_job_vocabulary_comes_from_the_rows_fit_reads_f5():
    train = raw_bank(jobs=JOBS[:6], seed=1)
    test = raw_bank(n=20, jobs=JOBS[6:], seed=2)
    pre = BankPreprocessor(verbose=False).fit(train)
    assert pre.job_map == {job: i for i, job in enumerate(sorted(set(train["job"])))}
    assert (pre.transform(test)["job"] == -1).all()        # unseen jobs -> -1


def test_job_vocabulary_ignores_labels_and_row_order_f5():
    raw = raw_bank()
    base = BankPreprocessor(verbose=False).fit(raw).job_map
    shuffled = mut.shuffle_rows(raw)
    shuffled["y"] = shuffled["y"].iloc[::-1].to_numpy()
    assert BankPreprocessor(verbose=False).fit(shuffled).job_map == base


def test_fit_without_dataframe_learns_from_the_whole_table_f5(tmp_path):
    """The notebooks' path: fit_transform() before the split sees every row."""
    train = raw_bank(jobs=JOBS[:6], seed=1)
    test = raw_bank(n=20, jobs=JOBS[6:], seed=2)
    csv = tmp_path / "raw.csv"
    pd.concat([train, test]).to_csv(csv, sep=";", index=False)
    db = build_database(csv_path=csv, db_path=tmp_path / "bank.db")
    vocab = BankPreprocessor(verbose=False, db_path=db).fit().job_map
    assert set(vocab) == set(train["job"]) | set(test["job"])
