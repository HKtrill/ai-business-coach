-- Schema for raw_bank: the original Bank Marketing CSV, stored as-is.
-- Column names and order match the CSV header. Values are not encoded.
-- Numeric columns use INTEGER/REAL affinity; categorical columns stay TEXT.
CREATE TABLE raw_bank (
    age              INTEGER,
    job              TEXT,
    marital          TEXT,
    education        TEXT,
    "default"        TEXT,
    housing          TEXT,
    loan             TEXT,
    contact          TEXT,
    month            TEXT,
    day_of_week      TEXT,
    duration         INTEGER,
    campaign         INTEGER,
    pdays            INTEGER,
    previous         INTEGER,
    poutcome         TEXT,
    "emp.var.rate"   REAL,
    "cons.price.idx" REAL,
    "cons.conf.idx"  REAL,
    euribor3m        REAL,
    "nr.employed"    REAL,
    y                TEXT
);