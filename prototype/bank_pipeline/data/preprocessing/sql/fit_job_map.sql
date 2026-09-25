-- Learn the job vocabulary from raw_bank.
-- Returns distinct non-null job categories in stable sorted order;
-- Python enumerates them into the job -> integer mapping.

SELECT DISTINCT job
FROM raw_bank
WHERE job IS NOT NULL
ORDER BY job;