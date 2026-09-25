-- Encode Bank Marketing data read from raw_bank.
-- Input tables:
--   raw_bank (raw columns, see create_raw_bank.sql)
--   job_map(job, code)  -- temp table built from the fitted vocabulary
-- Output columns keep the raw column order. Numeric columns pass through.
-- Unmapped categorical values yield NULL and are rejected in Python.
-- Unseen or null job values map to -1.

SELECT
    r.age AS age,

    -- Job (training vocabulary, unseen -> -1)
    COALESCE(j.code, -1) AS job,

    -- Marital (unknown -> -1)
    CASE r.marital
        WHEN 'divorced' THEN 0
        WHEN 'married'  THEN 1
        WHEN 'single'   THEN 2
        WHEN 'unknown'  THEN -1
    END AS marital,

    -- Education (ordinal 0-6, unknown -> -1)
    CASE r.education
        WHEN 'illiterate'          THEN 0
        WHEN 'basic.4y'            THEN 1
        WHEN 'basic.6y'            THEN 2
        WHEN 'basic.9y'            THEN 3
        WHEN 'high.school'         THEN 4
        WHEN 'professional.course' THEN 5
        WHEN 'university.degree'   THEN 6
        WHEN 'unknown'             THEN -1
    END AS education,

    -- Binary (no/yes/unknown -> 0/1/-1)
    CASE r."default"
        WHEN 'no'      THEN 0
        WHEN 'yes'     THEN 1
        WHEN 'unknown' THEN -1
    END AS "default",

    CASE r.housing
        WHEN 'no'      THEN 0
        WHEN 'yes'     THEN 1
        WHEN 'unknown' THEN -1
    END AS housing,

    CASE r.loan
        WHEN 'no'      THEN 0
        WHEN 'yes'     THEN 1
        WHEN 'unknown' THEN -1
    END AS loan,

    -- Contact
    CASE r.contact
        WHEN 'cellular'  THEN 0
        WHEN 'telephone' THEN 1
    END AS contact,

    -- Month (1-12)
    CASE r.month
        WHEN 'jan' THEN 1
        WHEN 'feb' THEN 2
        WHEN 'mar' THEN 3
        WHEN 'apr' THEN 4
        WHEN 'may' THEN 5
        WHEN 'jun' THEN 6
        WHEN 'jul' THEN 7
        WHEN 'aug' THEN 8
        WHEN 'sep' THEN 9
        WHEN 'oct' THEN 10
        WHEN 'nov' THEN 11
        WHEN 'dec' THEN 12
    END AS month,

    -- Day of week (0-4)
    CASE r.day_of_week
        WHEN 'mon' THEN 0
        WHEN 'tue' THEN 1
        WHEN 'wed' THEN 2
        WHEN 'thu' THEN 3
        WHEN 'fri' THEN 4
    END AS day_of_week,

    -- Campaign features (unchanged)
    r.duration AS duration,
    r.campaign AS campaign,
    r.pdays    AS pdays,
    r.previous AS previous,

    -- Poutcome
    CASE r.poutcome
        WHEN 'nonexistent' THEN 0  -- never contacted before
        WHEN 'failure'     THEN 1  -- previous campaign failed
        WHEN 'success'     THEN 2  -- previous campaign succeeded
    END AS poutcome,

    -- Economic features (unchanged)
    r."emp.var.rate"   AS "emp.var.rate",
    r."cons.price.idx" AS "cons.price.idx",
    r."cons.conf.idx"  AS "cons.conf.idx",
    r.euribor3m        AS euribor3m,
    r."nr.employed"    AS "nr.employed",

    -- Target (yes/no -> 1/0)
    CASE r.y
        WHEN 'yes' THEN 1
        WHEN 'no'  THEN 0
    END AS y

FROM raw_bank AS r
LEFT JOIN job_map AS j
    ON j.job = r.job
ORDER BY r.rowid;