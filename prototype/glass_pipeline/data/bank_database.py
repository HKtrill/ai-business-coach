"""
Bank Marketing SQLite Database
==============================
Builds the file-backed SQLite database and loads the raw CSV into raw_bank.
The database is a generated artifact and can be rebuilt from the CSV at any time.

Usage
-----
    from data.preprocessing import build_database
    build_database()               # builds only if raw_bank is missing
    build_database(rebuild=True)   # drops and reloads raw_bank

Author: Glass Pipeline Team
"""

from __future__ import annotations

import csv
import sqlite3
from contextlib import closing
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent
_SQL_DIR  = _DATA_DIR / "preprocessing" / "sql"

DEFAULT_CSV_PATH = _DATA_DIR / "raw" / "bank-additional-full.csv"
DEFAULT_DB_PATH  = _DATA_DIR / "db" / "bank_marketing.db"
RAW_TABLE        = "raw_bank"


def raw_bank_exists(db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    """Return True if the database file exists and contains raw_bank."""
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (RAW_TABLE,),
        ).fetchone()
    return row is not None


def build_database(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    rebuild: bool = False,
) -> Path:
    """
    Create the SQLite database and load the raw CSV into raw_bank.

    Parameters
    ----------
    csv_path : path to bank-additional-full.csv (semicolon-separated)
    db_path  : path of the SQLite database file to create or update
    rebuild  : drop and reload raw_bank even if it already exists

    Returns
    -------
    Path to the database file.
    """
    csv_path = Path(csv_path)
    db_path  = Path(db_path)

    if not rebuild and raw_bank_exists(db_path):
        return db_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_sql = (_SQL_DIR / "create_raw_bank.sql").read_text(encoding="utf-8")

    with open(csv_path, newline="", encoding="utf-8") as f, \
         closing(sqlite3.connect(db_path, isolation_level=None)) as conn:
        reader  = csv.reader(f, delimiter=";")
        header  = next(reader)
        columns = ", ".join(f'"{c}"' for c in header)
        params  = ", ".join("?" for _ in header)
        insert_sql = f"INSERT INTO {RAW_TABLE} ({columns}) VALUES ({params})"

        # Drop, create, and load in one transaction so a failed load
        # never leaves a partial raw_bank behind.
        conn.execute("BEGIN")
        try:
            conn.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
            conn.execute(create_sql)
            conn.executemany(insert_sql, (row for row in reader if row))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    return db_path