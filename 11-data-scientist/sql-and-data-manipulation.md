# SQL and Data Manipulation

---

## 1. Window Functions

Window functions compute a value for each row based on a related set of rows without collapsing the result (unlike GROUP BY).

### Syntax
```sql
function_name(...) OVER (
    PARTITION BY col1, col2   -- reset window per group
    ORDER BY col3             -- ordering within partition
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- frame
)
```

### Ranking Functions
```sql
ROW_NUMBER()    -- unique rank; no ties (arbitrary tiebreak by ORDER BY)
RANK()          -- ties get same rank; next rank skips (1,1,3)
DENSE_RANK()    -- ties get same rank; next rank does not skip (1,1,2)
NTILE(n)        -- divide rows into n equal buckets; returns bucket number
```

### Lag/Lead
```sql
LAG(col, n, default)   -- value of col n rows before current row
LEAD(col, n, default)  -- value of col n rows after current row

-- Compute day-over-day change
SELECT date, revenue,
       revenue - LAG(revenue, 1) OVER (ORDER BY date) AS delta
FROM daily_revenue;
```

### First/Last Value
```sql
FIRST_VALUE(col) OVER (PARTITION BY user_id ORDER BY event_time)
LAST_VALUE(col)  OVER (PARTITION BY user_id ORDER BY event_time
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
-- LAST_VALUE requires explicit frame to include all rows; default frame is RANGE UNBOUNDED PRECEDING TO CURRENT ROW
```

### Running Aggregations
```sql
-- Running total
SUM(revenue) OVER (PARTITION BY user_id ORDER BY date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

-- 7-day moving average
AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- Cumulative percentile rank
PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY ltv)  -- [0,1]
CUME_DIST()    OVER (PARTITION BY cohort ORDER BY ltv)  -- (0,1]
```

### Frame Clause Options
- `ROWS BETWEEN`: physical rows; `RANGE BETWEEN`: logical range (all rows with same ORDER BY value)
- `UNBOUNDED PRECEDING`: all rows before current
- `CURRENT ROW`
- `UNBOUNDED FOLLOWING`: all rows after current

---

## 2. Aggregations

```sql
-- Basic
SELECT region, COUNT(*), SUM(revenue), AVG(revenue), MAX(revenue)
FROM orders
GROUP BY region
HAVING SUM(revenue) > 10000;

-- ROLLUP: sub-totals + grand total
SELECT region, product, SUM(revenue)
FROM orders
GROUP BY ROLLUP(region, product);
-- Produces: (region, product), (region, NULL), (NULL, NULL)

-- CUBE: all combinations of grouping sets
GROUP BY CUBE(region, product);
-- Produces: (region, product), (region, NULL), (NULL, product), (NULL, NULL)

-- GROUPING SETS: explicit combinations
GROUP BY GROUPING SETS ((region, product), (region), ());
```

- `GROUPING(col)`: returns 1 if the NULL is from ROLLUP/CUBE aggregation (not actual NULL data)
- Use `HAVING` for filtering on aggregated values; `WHERE` filters before aggregation

---

## 3. Joins

### Types
```sql
INNER JOIN   -- only matching rows from both tables
LEFT JOIN    -- all rows from left; NULLs for unmatched right
RIGHT JOIN   -- all rows from right; NULLs for unmatched left
FULL OUTER   -- all rows from both; NULLs where no match
CROSS JOIN   -- Cartesian product; every combination
SELF JOIN    -- join table to itself (use aliases)
```

### When to Use Which
- `INNER`: when you only want complete matches (e.g., users who placed orders)
- `LEFT`: when you want all from the primary entity even without matches (e.g., all users, with order counts defaulting to NULL/0)
- `FULL OUTER`: reconciliation queries, finding discrepancies between two datasets
- `CROSS JOIN`: generating date spines, all combinations of two dimension tables
- `SELF JOIN`: hierarchical queries (manager-employee), session detection, comparing rows in same table

### NULL Behavior in Joins
- Rows where the join key is NULL do not match any row — even NULL = NULL is false in join conditions
- Use `COALESCE` or `IS NOT DISTINCT FROM` if you need NULL-to-NULL matching
- LEFT JOIN preserves NULLs in right table columns for unmatched rows

```sql
-- Find users with no orders (anti-join pattern)
SELECT u.user_id
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.user_id IS NULL;
```

---

## 4. CTEs vs Subqueries

```sql
-- CTE
WITH cohort AS (
    SELECT user_id, MIN(DATE_TRUNC('week', created_at)) AS cohort_week
    FROM users
    GROUP BY user_id
),
activity AS (
    SELECT user_id, DATE_TRUNC('week', event_time) AS activity_week
    FROM events
)
SELECT c.cohort_week, a.activity_week,
       COUNT(DISTINCT a.user_id) AS retained_users
FROM cohort c
JOIN activity a ON c.user_id = a.user_id
GROUP BY 1, 2;
```

- CTEs improve readability; can be referenced multiple times in the same query
- In most engines (PostgreSQL, BigQuery, Snowflake), CTEs are inlined by default (not materialized)
- Force materialization with `MATERIALIZE` hint (PostgreSQL 12+) or temp tables when CTE is referenced many times

### Recursive CTEs
```sql
-- Traverse hierarchy (org chart, BOM)
WITH RECURSIVE org AS (
    SELECT employee_id, manager_id, name, 1 AS level
    FROM employees
    WHERE manager_id IS NULL  -- root
    UNION ALL
    SELECT e.employee_id, e.manager_id, e.name, o.level + 1
    FROM employees e
    JOIN org o ON e.manager_id = o.employee_id
)
SELECT * FROM org ORDER BY level;
```

---

## 5. Date and Time

```sql
DATE_TRUNC('week', timestamp_col)   -- floor to week start
DATE_TRUNC('month', timestamp_col)  -- floor to month start

DATE_DIFF('day', start_date, end_date)   -- BigQuery syntax
DATEDIFF(day, start_date, end_date)      -- Snowflake/SQL Server syntax
end_date - start_date                    -- PostgreSQL returns interval

EXTRACT(DOW FROM timestamp_col)   -- day of week (0=Sunday in PG)
EXTRACT(HOUR FROM timestamp_col)  -- hour

-- Cohort week calculation
DATE_TRUNC('week', MIN(created_at) OVER (PARTITION BY user_id)) AS cohort_week

-- Convert epoch to timestamp
TO_TIMESTAMP(epoch_col)  -- PostgreSQL
TIMESTAMP_SECONDS(epoch_col)  -- BigQuery
```

### Time Zone Handling
- Always store timestamps in UTC; convert at query time using `AT TIME ZONE`
- `TIMESTAMP WITH TIME ZONE` vs `TIMESTAMP WITHOUT TIME ZONE` (PostgreSQL) — know the difference
- Aggregating by "day" in user's local time requires converting before `DATE_TRUNC`

---

## 6. NULL Handling

```sql
COALESCE(col, 0)          -- return first non-NULL; replace NULLs in aggregations
NULLIF(col, 0)            -- return NULL if col = 0; prevents division by zero
COALESCE(revenue, 0) / NULLIF(impressions, 0)  -- safe division

-- NULL in aggregations
COUNT(*)          -- counts all rows including NULLs
COUNT(col)        -- counts non-NULL values only
SUM(col)          -- ignores NULLs (not treated as 0)
AVG(col)          -- average of non-NULL values; NULLs excluded from denominator

-- Filtering NULLs
WHERE col IS NULL
WHERE col IS NOT NULL
-- WHERE col = NULL always returns false; never use = NULL
```

---

## 7. Set Operations

```sql
query1 UNION ALL query2   -- all rows including duplicates; faster (no dedup)
query1 UNION query2       -- deduplicated rows; slower (implicit DISTINCT)
query1 INTERSECT query2   -- rows in both queries
query1 EXCEPT query2      -- rows in query1 but not query2 (MINUS in Oracle)
```

- Schemas must match: same number of columns, compatible types
- `UNION ALL` is almost always preferable for performance unless deduplication is required
- `EXCEPT` is an efficient anti-join alternative when the tables share same schema

---

## 8. Performance

### Indexes
- **B-tree**: default; supports equality and range queries; used for ORDER BY, BETWEEN, `<`, `>`
- **Hash**: equality-only; faster for point lookups; not supported by all engines
- Composite indexes: column order matters — leading column must appear in query predicate for index to be used

### Query Analysis
```sql
EXPLAIN SELECT ...             -- show query plan (no execution)
EXPLAIN ANALYZE SELECT ...     -- execute and show actual vs estimated rows, timings (PostgreSQL)
```
- Look for: sequential scans on large tables, nested loop joins on large tables, high row estimates vs actuals

### Optimization Patterns
- **Partition pruning**: `WHERE date >= '2024-01-01'` on a date-partitioned table → scans only relevant partitions; always filter on partition column
- **Avoid `SELECT *`**: fetches all columns; prevents projection pushdown in columnar stores
- **Predicate pushdown**: filter early in CTEs/subqueries to reduce rows processed downstream
- **Filter before join**: reduce join size with WHERE conditions before the join, not after

---

## 9. Funnel Analysis

```sql
-- Step-by-step conversion with window functions
WITH steps AS (
    SELECT user_id,
           MAX(CASE WHEN event = 'page_view' THEN 1 ELSE 0 END)   AS step1,
           MAX(CASE WHEN event = 'add_to_cart' THEN 1 ELSE 0 END) AS step2,
           MAX(CASE WHEN event = 'checkout' THEN 1 ELSE 0 END)    AS step3,
           MAX(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END)    AS step4
    FROM events
    WHERE event_date >= '2024-01-01'
    GROUP BY user_id
)
SELECT
    SUM(step1) AS page_views,
    SUM(step2) AS add_to_carts,
    SUM(step3) AS checkouts,
    SUM(step4) AS purchases,
    ROUND(100.0 * SUM(step2) / NULLIF(SUM(step1), 0), 2) AS step1_to_2_pct,
    ROUND(100.0 * SUM(step3) / NULLIF(SUM(step2), 0), 2) AS step2_to_3_pct,
    ROUND(100.0 * SUM(step4) / NULLIF(SUM(step3), 0), 2) AS step3_to_4_pct
FROM steps;
```

---

## 10. Cohort Retention

```sql
WITH cohorts AS (
    SELECT user_id,
           DATE_TRUNC('week', MIN(created_at)) AS cohort_week
    FROM users
    GROUP BY user_id
),
activity AS (
    SELECT DISTINCT user_id,
           DATE_TRUNC('week', event_time) AS activity_week
    FROM events
),
joined AS (
    SELECT c.cohort_week,
           DATEDIFF('week', c.cohort_week, a.activity_week) AS weeks_since_signup,
           COUNT(DISTINCT a.user_id) AS active_users
    FROM cohorts c
    JOIN activity a ON c.user_id = a.user_id
    GROUP BY 1, 2
),
cohort_sizes AS (
    SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
    FROM cohorts
    GROUP BY cohort_week
)
SELECT j.cohort_week,
       j.weeks_since_signup,
       j.active_users,
       cs.cohort_size,
       ROUND(100.0 * j.active_users / cs.cohort_size, 1) AS retention_pct
FROM joined j
JOIN cohort_sizes cs ON j.cohort_week = cs.cohort_week
ORDER BY 1, 2;
```

---

## 11. Sessionization

```sql
-- Gap-based session boundary: new session if gap > 30 minutes
WITH gaps AS (
    SELECT user_id, event_time,
           LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event_time
    FROM events
),
boundaries AS (
    SELECT user_id, event_time,
           CASE WHEN prev_event_time IS NULL
                     OR DATEDIFF('minute', prev_event_time, event_time) > 30
                THEN 1 ELSE 0 END AS is_session_start
    FROM gaps
),
sessions AS (
    SELECT user_id, event_time,
           SUM(is_session_start) OVER (PARTITION BY user_id ORDER BY event_time
                                       ROWS UNBOUNDED PRECEDING) AS session_id
    FROM boundaries
)
SELECT user_id, session_id,
       MIN(event_time) AS session_start,
       MAX(event_time) AS session_end,
       COUNT(*) AS events_in_session
FROM sessions
GROUP BY user_id, session_id;
```

---

## 12. Deduplication

```sql
-- Keep latest record per user (most common dedup pattern)
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
    FROM users_raw
)
SELECT * FROM ranked WHERE rn = 1;

-- Delete duplicates in-place (PostgreSQL)
DELETE FROM users_raw
WHERE id NOT IN (
    SELECT MIN(id) FROM users_raw GROUP BY user_id
);
```

---

## 13. Metrics Queries

```sql
-- DAU / WAU / MAU
SELECT DATE_TRUNC('day', event_time)   AS date, COUNT(DISTINCT user_id) AS dau FROM events GROUP BY 1;
SELECT DATE_TRUNC('week', event_time)  AS week, COUNT(DISTINCT user_id) AS wau FROM events GROUP BY 1;
SELECT DATE_TRUNC('month', event_time) AS month, COUNT(DISTINCT user_id) AS mau FROM events GROUP BY 1;

-- Rolling 7-day actives
SELECT date,
       COUNT(DISTINCT user_id) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_dau
FROM (SELECT DISTINCT DATE(event_time) AS date, user_id FROM events) d;

-- Revenue per user
SELECT DATE_TRUNC('month', order_date) AS month,
       SUM(revenue) / COUNT(DISTINCT user_id) AS arpu
FROM orders
GROUP BY 1;

-- Percentile latency (p50, p95, p99)
SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99
FROM request_logs
WHERE request_date = CURRENT_DATE;

-- Stickiness: DAU/MAU ratio
WITH dau AS (
    SELECT DATE(event_time) AS dt, COUNT(DISTINCT user_id) AS daily_users
    FROM events GROUP BY 1
),
mau AS (
    SELECT DATE_TRUNC('month', event_time) AS month, COUNT(DISTINCT user_id) AS monthly_users
    FROM events GROUP BY 1
)
SELECT d.dt,
       d.daily_users,
       m.monthly_users,
       ROUND(1.0 * d.daily_users / m.monthly_users, 4) AS stickiness
FROM dau d
JOIN mau m ON DATE_TRUNC('month', d.dt) = m.month;
```
