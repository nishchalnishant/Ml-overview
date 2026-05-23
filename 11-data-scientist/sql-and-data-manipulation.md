---
module: Data Scientist
topic: Sql And Data Manipulation
subtopic: ""
status: unread
tags: [datascientist, ml, sql-and-data-manipulation]
---
# SQL and Data Manipulation

---

## 1. Window Functions

### ROW_NUMBER

**The problem ROW_NUMBER solves**: you have duplicate rows per user — multiple event records, multiple login timestamps, multiple rows from CDC replication — and you want exactly one row per user: the most recent one. A `GROUP BY` won't work because you want the full row, not an aggregated summary.

**The core insight**: you need to number rows within each group by some ordering criterion, then filter to number = 1. `ROW_NUMBER()` does this — it assigns a unique integer to each row within a `PARTITION` (your group), ordered by whatever column you specify.

**The pattern**:
```sql
SELECT * FROM (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
  FROM events
) t
WHERE rn = 1;
```
`PARTITION BY user_id` restarts numbering for each user. `ORDER BY created_at DESC` means the most recent row gets `rn = 1`. The outer `WHERE rn = 1` keeps only that row.

**What breaks**: if two rows for the same user have identical `created_at`, the tiebreak is arbitrary — the engine picks whichever it processes first. You'll get exactly one row (which is usually what you want for dedup), but you can't predict which one. If you want both tied rows, use `RANK()` instead.

---

### RANK and DENSE_RANK

**The problem they solve**: you want to find the top-N products by revenue per category, but two products might have identical revenue and you want to treat them equally — not arbitrarily discard one.

**The core insight**: `RANK()` assigns the same number to tied rows, then skips numbers for the gap (1, 1, 3). `DENSE_RANK()` also assigns the same number to ties but does not skip (1, 1, 2). Choose based on whether the gaps matter to your downstream logic.

**The pattern**:
```sql
-- Top 3 products per category, ties included
SELECT * FROM (
  SELECT category, product, revenue,
         DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS dr
  FROM products
) t
WHERE dr <= 3;
```
With `DENSE_RANK`, if two products share rank 1, both appear and the next product gets rank 2. With `RANK`, the next product would get rank 3.

**What breaks**: `WHERE rank <= 3` with `RANK()` can return fewer than 3 distinct revenue levels if there are gaps. With `DENSE_RANK()`, `<= 3` always means "top 3 distinct revenue levels." Pick based on what "top 3" means in your domain.

---

### NTILE

**The problem NTILE solves**: you have a user base and want to segment them into performance quartiles for a marketing email — "top 25%", "next 25%", etc. You don't know the revenue thresholds in advance; you want the data to set them dynamically.

**The core insight**: `NTILE(n)` divides rows into `n` equal-sized buckets and returns the bucket number. You don't define the boundaries — the engine infers them from the data distribution.

**The pattern**:
```sql
SELECT user_id, revenue,
       NTILE(4) OVER (ORDER BY revenue DESC) AS quartile
FROM user_revenue;
-- quartile=1: top 25%, quartile=4: bottom 25%
```

**What breaks**: if the row count is not divisible by `n`, some buckets get one extra row. The engine fills the first buckets before the last ones, so bucket 1 may have one more row than bucket 4. This is usually fine, but don't assume perfectly equal bucket sizes when making claims about "exactly 25%."

---

### LAG and LEAD

**The problem they solve**: you have a daily revenue table and you need day-over-day change. To compute that you need the current row's value and the previous row's value simultaneously — but SQL normally only gives you one row at a time. A self-join works but is expensive and verbose.

**The core insight**: `LAG(col, n, default)` reaches back `n` rows in the partition's order and returns that value. `LEAD` reaches forward. Both operate within the current row's context, so you avoid a join entirely.

**The pattern**:
```sql
-- Day-over-day revenue change
SELECT date, revenue,
       revenue - LAG(revenue, 1) OVER (ORDER BY date) AS delta,
       ROUND(100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY date))
             / NULLIF(LAG(revenue, 1) OVER (ORDER BY date), 0), 2) AS pct_change
FROM daily_revenue
ORDER BY date;
```

The `NULLIF(..., 0)` prevents division by zero when yesterday's revenue was zero.

**What breaks**: at the first row of each partition, `LAG` has no preceding row and returns `NULL` (or the default you supply as the third argument). Forgetting to handle that `NULL` in downstream arithmetic will silently produce `NULL` results for the first row of every partition.

---

### FIRST_VALUE and LAST_VALUE

**The problem they solve**: you want the first event a user ever performed (their signup channel), carried on every subsequent row, so you can group or filter without a join back to a users table.

**The core insight**: `FIRST_VALUE(col)` returns the value from the first row in the window frame. `LAST_VALUE(col)` returns the value from the last row.

**The pattern**:
```sql
-- Carry first-touch channel on every event row
SELECT user_id, event_time, channel,
       FIRST_VALUE(channel) OVER (
         PARTITION BY user_id ORDER BY event_time
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS first_touch_channel
FROM events;
```

**What breaks**: `LAST_VALUE` has a silent gotcha. The default window frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which means "last value up to and including the current row" — not the last value in the partition. For most rows this returns the current row itself, not what you expect. You must specify:
```sql
LAST_VALUE(channel) OVER (
  PARTITION BY user_id ORDER BY event_time
  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
```
That frame includes all rows in the partition, so `LAST_VALUE` returns the actual final value.

---

### Running Aggregations and the Frame Clause

**The problem they solve**: you need a cumulative total to date, or a 7-day rolling average, without writing a correlated subquery that re-scans the table for every row.

**The core insight**: window aggregation functions (`SUM`, `AVG`, `COUNT`, `MIN`, `MAX`) accept an `OVER` clause with a frame that defines which rows to include in each row's computation. You slide this frame across the ordered partition.

**The pattern**:
```sql
-- Cumulative revenue per user
SELECT user_id, order_date, revenue,
       SUM(revenue) OVER (
         PARTITION BY user_id
         ORDER BY order_date
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_revenue
FROM orders;

-- 7-day rolling average of daily revenue
SELECT date, revenue,
       AVG(revenue) OVER (
         ORDER BY date
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7d_avg
FROM daily_revenue;
```

Frame options:
- `ROWS BETWEEN`: counts physical rows — `6 PRECEDING` means the 6 rows immediately above in the ordered result
- `RANGE BETWEEN`: includes all rows whose `ORDER BY` value falls within the range — if two rows share the same date, both are included or excluded together
- `UNBOUNDED PRECEDING`: all rows from the start of the partition
- `UNBOUNDED FOLLOWING`: all rows to the end of the partition

**What breaks**: `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` counts exactly 7 rows, but those 7 rows might not represent 7 calendar days if the table has missing dates. If you have gaps in your date series, use a date spine (see the metrics section) and fill missing dates before applying the rolling window.

---

### PERCENT_RANK and CUME_DIST

**The problem they solve**: you want to know where a user's LTV falls within their cohort — not just their rank number but their percentile position.

**The core insight**: `PERCENT_RANK()` returns `(rank - 1) / (total_rows - 1)`, ranging from 0 to 1. `CUME_DIST()` returns the fraction of rows with a value ≤ the current row's value, ranging from 1/n to 1. Use `PERCENT_RANK` for "what fraction of users scored below this user"; use `CUME_DIST` when you want the cumulative distribution directly.

**The pattern**:
```sql
SELECT user_id, ltv,
       PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY ltv) AS pct_rank,
       CUME_DIST()    OVER (PARTITION BY cohort ORDER BY ltv) AS cume_dist
FROM user_ltv;
```

**What breaks**: `PERCENT_RANK` returns 0 for the lowest-ranked row (not a true percentile in the statistical sense). For a proper percentile calculation of a specific quantile, use `PERCENTILE_CONT` instead (covered in the metrics section).

---

## 2. Aggregations

### Basic GROUP BY

**The problem**: you have millions of order rows and you need one summary row per region. Scanning all rows to produce region-level totals is unavoidable, but you need a way to declare "collapse rows with the same region into one result row."

**The core insight**: `GROUP BY` collapses all rows with the same key values into one output row. Every column in the `SELECT` must either be in the `GROUP BY` or wrapped in an aggregate function.

**The pattern**:
```sql
SELECT region,
       COUNT(*)           AS total_orders,
       COUNT(DISTINCT user_id) AS unique_buyers,
       SUM(revenue)       AS total_revenue,
       AVG(revenue)       AS avg_order_value,
       MAX(revenue)       AS largest_order
FROM orders
GROUP BY region
HAVING SUM(revenue) > 10000;
```

`WHERE` filters rows before grouping. `HAVING` filters after — it can reference aggregate expressions. Putting a condition in `WHERE` instead of `HAVING` when it involves an aggregate is a syntax error.

**What breaks**: `COUNT(col)` counts non-NULL values; `COUNT(*)` counts all rows including rows with NULLs in that column. These can differ significantly if the column is sparse.

---

### ROLLUP

**The problem**: the business wants a report with revenue by (region, product), subtotals by region, and a grand total — all in one result set. Running three separate queries and UNIONing them is tedious and slow.

**The core insight**: `ROLLUP` extends `GROUP BY` to automatically produce subtotals. `GROUP BY ROLLUP(a, b)` produces groupings: `(a, b)`, `(a)`, and `()` — each level rolling up one dimension at a time from right to left.

**The pattern**:
```sql
SELECT region,
       product,
       SUM(revenue) AS revenue
FROM orders
GROUP BY ROLLUP(region, product)
ORDER BY region NULLS LAST, product NULLS LAST;
```

Result includes: per-product-per-region rows, per-region subtotals (product = NULL), and a grand total (both NULL).

**What breaks**: the NULLs in rollup output are ambiguous — a NULL in the `region` column could mean "grand total" or it could be real data where the region is unknown. Use `GROUPING(region)` — it returns 1 if the NULL comes from ROLLUP aggregation, 0 if it's genuine data:
```sql
SELECT
  CASE WHEN GROUPING(region) = 1 THEN 'ALL' ELSE region END AS region,
  CASE WHEN GROUPING(product) = 1 THEN 'ALL' ELSE product END AS product,
  SUM(revenue)
FROM orders
GROUP BY ROLLUP(region, product);
```

---

### CUBE

**The problem**: you want all possible subtotals across three dimensions — you can't predict which dimension combination a stakeholder will drill into, so you pre-compute every combination.

**The core insight**: `CUBE(a, b, c)` produces every possible subset of the grouping columns: `(a,b,c)`, `(a,b)`, `(a,c)`, `(b,c)`, `(a)`, `(b)`, `(c)`, `()`. For n dimensions, that's 2^n groupings.

**The pattern**:
```sql
SELECT region, product, channel,
       SUM(revenue)
FROM orders
GROUP BY CUBE(region, product, channel);
```

**What breaks**: `CUBE` on high-cardinality dimensions explodes result set size. Three dimensions each with 100 distinct values produces up to 100^3 + 100^2 + 100^1 + 1 rows of output. Use it only on low-cardinality categorical dimensions.

---

### GROUPING SETS

**The problem**: you need exactly these combinations: `(region, product)` and `(channel)` — not all combinations, not a hierarchy. `ROLLUP` and `CUBE` can't express arbitrary combinations.

**The core insight**: `GROUPING SETS` lets you explicitly enumerate the grouping combinations you want, producing a UNION of those GROUP BY queries in a single scan.

**The pattern**:
```sql
SELECT region, product, channel,
       SUM(revenue)
FROM orders
GROUP BY GROUPING SETS (
  (region, product),
  (channel),
  ()              -- grand total
);
```

**What breaks**: each column not part of a given grouping set appears as NULL in that grouping's rows. Same GROUPING() trick applies to distinguish real NULLs from aggregation NULLs.

---

## 3. Joins

### INNER JOIN

**The problem**: you have a `users` table and an `orders` table. You want rows that show user name alongside their order amount. Rows only make sense if both sides exist.

**The core insight**: `INNER JOIN` returns only rows where the join condition is satisfied in both tables. Rows with no match on either side are discarded.

**The pattern**:
```sql
SELECT u.name, o.amount, o.order_date
FROM users u
INNER JOIN orders o ON u.user_id = o.user_id
WHERE o.order_date >= '2024-01-01';
```

**What breaks**: if `user_id` in orders contains values absent from users (orphaned rows), those orders are silently dropped. Whether that's correct depends on your data contract — in a referential integrity violation scenario you'd lose rows unexpectedly.

---

### LEFT JOIN

**The problem**: you want all users, and for each user their total order count — including users who have never ordered. An INNER JOIN would drop users with no orders, giving you a biased count.

**The core insight**: `LEFT JOIN` keeps every row from the left table. For left-table rows with no matching right-table row, all right-table columns are `NULL`.

**The pattern**:
```sql
SELECT u.user_id, u.name,
       COUNT(o.order_id) AS order_count,
       COALESCE(SUM(o.amount), 0) AS total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.name;
```

`COUNT(o.order_id)` returns 0 for users with no orders because `COUNT(col)` ignores NULLs. `COALESCE(..., 0)` handles the NULL total.

**What breaks**: filtering on a right-table column in `WHERE` after a `LEFT JOIN` implicitly converts it to an `INNER JOIN`, because `WHERE col = value` is false when col is NULL. Move such filters into the `ON` clause:
```sql
-- Wrong: loses users with no recent orders
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_date >= '2024-01-01'

-- Correct: keeps all users; only considers recent orders in the join
LEFT JOIN orders o ON u.user_id = o.user_id AND o.order_date >= '2024-01-01'
```

---

### FULL OUTER JOIN

**The problem**: you have two systems that should have the same set of user IDs but you suspect divergence. You want to find IDs present in one system but missing from the other.

**The core insight**: `FULL OUTER JOIN` keeps all rows from both tables. Where the join condition matches, both sides fill in. Where it doesn't, the unmatched side has NULLs.

**The pattern**:
```sql
SELECT
  COALESCE(a.user_id, b.user_id) AS user_id,
  a.system_a_value,
  b.system_b_value
FROM system_a a
FULL OUTER JOIN system_b b ON a.user_id = b.user_id
WHERE a.user_id IS NULL OR b.user_id IS NULL;
-- Rows where one side is NULL = discrepancy
```

**What breaks**: `FULL OUTER JOIN` is rarely supported efficiently in distributed query engines. In BigQuery or Spark, it forces a shuffle of both tables. For large-scale reconciliation, consider using `UNION ALL` with explicit source flags and grouping instead.

---

### CROSS JOIN

**The problem**: you want to compute metrics for every combination of date and user segment, but not all combinations exist in your events table. You need to generate all possible pairs first, then left-join actual data.

**The core insight**: `CROSS JOIN` returns the Cartesian product — every row from the left table paired with every row from the right table. This is how you build a complete "spine" of combinations.

**The pattern**:
```sql
-- Generate a date spine × segment grid, then fill with actuals
WITH dates AS (
  SELECT generate_series('2024-01-01'::date, '2024-12-31'::date, '1 day')::date AS dt
),
segments AS (SELECT DISTINCT segment FROM users)
SELECT d.dt, s.segment,
       COALESCE(COUNT(DISTINCT e.user_id), 0) AS dau
FROM dates d
CROSS JOIN segments s
LEFT JOIN events e ON DATE(e.event_time) = d.dt
  AND e.segment = s.segment
GROUP BY d.dt, s.segment;
```

**What breaks**: a `CROSS JOIN` on two large tables is catastrophic. 1M rows × 1M rows = 1 trillion rows. Always ensure at least one side is small (a list of dates, a list of categories, a small lookup table).

---

### SELF JOIN

**The problem**: your employees table has a `manager_id` column pointing to another row in the same table. You want to show each employee alongside their manager's name.

**The core insight**: a self join treats one table as if it were two by giving it two aliases. The join condition links rows within the same table.

**The pattern**:
```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```

`LEFT JOIN` ensures top-level employees (where `manager_id IS NULL`) still appear, with `manager = NULL`.

**What breaks**: for deeply nested hierarchies (org charts, bill-of-materials), self joins are limited to one level. Use a recursive CTE for arbitrary depth.

---

### NULL behavior in joins

**The problem that surprises people**: you have rows where the join key is NULL and you expect them to match other NULL-key rows. They don't.

**The core insight**: SQL's three-valued logic means `NULL = NULL` evaluates to `UNKNOWN`, not `TRUE`. A join condition must be `TRUE` to produce a match. Two rows with `NULL` join keys never match each other.

**The pattern** (anti-join — finding users with no orders):
```sql
SELECT u.user_id
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.user_id IS NULL;
```

If you genuinely need `NULL` to match `NULL`, use `IS NOT DISTINCT FROM`:
```sql
JOIN table2 ON table1.col IS NOT DISTINCT FROM table2.col
```

---

## 4. CTEs vs Subqueries

### Why CTEs exist

**The problem**: you have a four-step query where each step depends on the previous one. Written as nested subqueries, by the fourth nesting level the query is unreadable and you can't test intermediate steps independently.

**The core insight**: a CTE (`WITH` clause) names an intermediate result so you can reference it by name instead of nesting. Readability is the primary benefit — in most engines the optimizer inlines CTEs anyway, so there's no automatic performance difference.

**The pattern**:
```sql
WITH active_users AS (
  SELECT DISTINCT user_id
  FROM events
  WHERE event_time >= CURRENT_DATE - INTERVAL '30 days'
),
user_revenue AS (
  SELECT user_id, SUM(amount) AS ltv
  FROM orders
  GROUP BY user_id
)
SELECT a.user_id, COALESCE(r.ltv, 0) AS ltv
FROM active_users a
LEFT JOIN user_revenue r ON a.user_id = r.user_id;
```

**What breaks**: in PostgreSQL pre-12, CTEs were always materialized (executed once and cached), acting as an optimization fence. From PostgreSQL 12+, CTEs are inlined by default. In Snowflake and BigQuery, CTEs are generally inlined. This matters when you have a CTE referenced many times: materialization can help (one execution) or hurt (cached result misses stats-based optimizations). Force materialization in PostgreSQL 12+ with `WITH cte AS MATERIALIZED (...)`.

---

### Recursive CTEs

**The problem**: you have an org chart stored as `(employee_id, manager_id)`. You want to find all reports under a given VP — not just direct reports, but the full subtree. This requires traversing an arbitrary depth without knowing the depth in advance.

**The core insight**: a recursive CTE defines a base case (the root) and a recursive step (extend one level deeper), and the engine repeats the recursive step until no new rows are produced.

**The pattern**:
```sql
WITH RECURSIVE org_tree AS (
  -- Base case: start from the target VP
  SELECT employee_id, manager_id, name, 1 AS depth
  FROM employees
  WHERE employee_id = 42  -- the VP's ID

  UNION ALL

  -- Recursive step: find direct reports of already-found employees
  SELECT e.employee_id, e.manager_id, e.name, o.depth + 1
  FROM employees e
  JOIN org_tree o ON e.manager_id = o.employee_id
)
SELECT * FROM org_tree ORDER BY depth, name;
```

**What breaks**: if the data has cycles (A reports to B, B reports to A), the recursive CTE loops forever or until it hits the engine's recursion limit. Add a depth guard (`WHERE depth < 50`) or, in PostgreSQL 14+, use `CYCLE employee_id SET is_cycle USING path` to detect cycles automatically.

---

## 5. Date and Time Operations

### Truncating and bucketing dates

**The problem**: you have event timestamps at millisecond precision, but you need to count users per week for a retention chart. You need to map each timestamp to its containing week.

**The core insight**: `DATE_TRUNC` floors a timestamp to the start of the specified time unit (year, month, week, day, hour, etc.). All timestamps in the same week collapse to the same Monday (or Sunday, depending on locale).

**The pattern**:
```sql
SELECT DATE_TRUNC('week', event_time) AS week_start,
       COUNT(DISTINCT user_id) AS wau
FROM events
GROUP BY 1
ORDER BY 1;
```

Dialect differences:
```sql
DATE_TRUNC('month', ts)     -- PostgreSQL, BigQuery, Snowflake
DATE_FORMAT(ts, '%Y-%m-01') -- MySQL
TRUNC(ts, 'MM')             -- Oracle
```

**What breaks**: `DATE_TRUNC('week', ...)` in PostgreSQL floors to Monday. If your business week starts Sunday, you need to shift: `DATE_TRUNC('week', ts + INTERVAL '1 day') - INTERVAL '1 day'`. In BigQuery, `DATE_TRUNC(date, WEEK(SUNDAY))` handles this directly.

---

### Date arithmetic

**The problem**: you want to find users who signed up but never placed an order within their first 7 days. You need to compare two timestamps and produce a day count.

**The pattern**:
```sql
-- PostgreSQL: subtraction returns interval
SELECT user_id,
       (first_order_date - signup_date) AS days_to_first_order
FROM user_funnel
WHERE first_order_date - signup_date <= INTERVAL '7 days';

-- BigQuery
DATE_DIFF(first_order_date, signup_date, DAY)

-- Snowflake / SQL Server
DATEDIFF(day, signup_date, first_order_date)
```

**What breaks**: mixing `DATE` and `TIMESTAMP` types in subtraction. In PostgreSQL, `TIMESTAMP - DATE` requires explicit casting. Always be explicit about types:
```sql
EXTRACT(EPOCH FROM (ts1 - ts2)) / 86400.0  -- seconds difference → fractional days
```

---

### Extracting date parts

**The problem**: you want to see if revenue differs by day of week or hour of day — to detect weekend drops or off-peak hours.

**The pattern**:
```sql
SELECT EXTRACT(DOW FROM event_time) AS day_of_week,  -- 0=Sunday in PostgreSQL
       EXTRACT(HOUR FROM event_time) AS hour_of_day,
       COUNT(*) AS events
FROM events
GROUP BY 1, 2
ORDER BY 1, 2;
```

**What breaks**: `EXTRACT(DOW ...)` returns 0 for Sunday in PostgreSQL but 1 in some other systems (where 1=Monday). Always test your engine's convention before building visualizations.

---

### Time zone handling

**The problem**: your event timestamps are stored in UTC but users are in different time zones. A "daily" active user who is in UTC-8 but logs in at 11pm UTC is in the previous calendar day for them — your DAU count is wrong if you don't account for this.

**The core insight**: always store timestamps in UTC. Convert to the user's local time zone at query time using `AT TIME ZONE`, then truncate to the desired unit.

**The pattern**:
```sql
-- Convert UTC to US/Pacific before computing the day
SELECT DATE_TRUNC('day',
         event_time AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles'
       ) AS local_day,
       COUNT(DISTINCT user_id) AS dau
FROM events
GROUP BY 1;
```

**What breaks**: `TIMESTAMP WITH TIME ZONE` (PostgreSQL) stores the UTC offset alongside the value and converts correctly. `TIMESTAMP WITHOUT TIME ZONE` stores nothing — `AT TIME ZONE` interprets it as being in the session time zone. Mixing these types silently produces wrong results. Enforce UTC storage at the application layer and always use `TIMESTAMPTZ`.

---

## 6. NULL Handling

### Why NULLs are dangerous

**The problem**: you have a revenue column where some rows are NULL (the transaction failed before an amount was recorded). You want the average revenue per order. `AVG(revenue)` gives a different answer than `SUM(revenue) / COUNT(*)` — which is correct?

**The core insight**: `NULL` means "unknown." Aggregate functions (`SUM`, `AVG`, `MIN`, `MAX`) ignore NULLs entirely. `COUNT(col)` counts non-NULLs; `COUNT(*)` counts all rows. Whether ignoring NULLs is correct depends on what NULL means in your domain.

**The pattern**:
```sql
-- These are different if revenue has NULLs
AVG(revenue)                   -- ignores NULLs: avg of known amounts
SUM(revenue) / COUNT(*)        -- treats NULLs as 0 in the denominator

-- Safe division avoiding zero denominator
COALESCE(revenue, 0) / NULLIF(impressions, 0)
--  NULLIF(x, 0) → returns NULL when x=0, preventing division by zero
--  Any arithmetic with NULL returns NULL, which COALESCE can then handle
```

**What breaks**: `WHERE col = NULL` is never true — `NULL = NULL` evaluates to `UNKNOWN`. Always use `IS NULL` or `IS NOT NULL`. Similarly, `CASE WHEN col = NULL THEN ...` never triggers; write `CASE WHEN col IS NULL THEN ...`.

---

### COALESCE and NULLIF

**The problem**: you're building a display name from `preferred_name`, falling back to `first_name`, falling back to email. You need the first non-NULL value in a sequence.

**The pattern**:
```sql
SELECT COALESCE(preferred_name, first_name, email) AS display_name
FROM users;
```

`NULLIF(a, b)` returns NULL when `a = b`, otherwise returns `a`. Use it to convert sentinel values to NULL so aggregations skip them:
```sql
-- Treat 0-duration sessions as missing data
AVG(NULLIF(session_duration_seconds, 0))
```

**What breaks**: `COALESCE` evaluates arguments left to right and stops at the first non-NULL. In most engines this is lazy evaluation, but if arguments have side effects or are expensive subqueries, order matters for both correctness and performance.

---

## 7. Set Operations

### UNION ALL vs UNION

**The problem**: you want to combine yesterday's events with today's events from two separate partitioned tables into one result set. You need all rows, including duplicates.

**The core insight**: `UNION ALL` appends the results of two queries without deduplication. `UNION` (without `ALL`) performs an implicit `DISTINCT` across the combined result, which requires a sort or hash operation over all rows — expensive and usually unnecessary.

**The pattern**:
```sql
-- Combine two daily partitions efficiently
SELECT user_id, event_type, event_time FROM events_20240101
UNION ALL
SELECT user_id, event_type, event_time FROM events_20240102;
```

Both queries must have the same number of columns with compatible types. Column names come from the first query.

**What breaks**: using `UNION` when you mean `UNION ALL` is a performance mistake that also changes results — it silently removes legitimate duplicate events. Only use `UNION` (deduplicating) when deduplication is actually the goal.

---

### INTERSECT and EXCEPT

**The problem**: you want users who were active in both January and February (intersection), or users active in January who were not active in February (difference). You could express both with joins, but set operations are often more readable.

**The pattern**:
```sql
-- Users active in both months (intersection)
SELECT user_id FROM events WHERE DATE_TRUNC('month', event_time) = '2024-01-01'
INTERSECT
SELECT user_id FROM events WHERE DATE_TRUNC('month', event_time) = '2024-02-01';

-- Users active in January but not February (except / anti-join)
SELECT user_id FROM events WHERE DATE_TRUNC('month', event_time) = '2024-01-01'
EXCEPT
SELECT user_id FROM events WHERE DATE_TRUNC('month', event_time) = '2024-02-01';
```

**What breaks**: `INTERSECT` and `EXCEPT` perform implicit deduplication — if the same `user_id` appears multiple times in the January set, the result contains it only once. If you need counts, use joins or CTEs with `COUNT`.

---

## 8. Performance

### Why indexes matter

**The problem**: your query on a 500M-row events table takes 4 minutes. The `WHERE` clause filters by `user_id`. Without an index, the engine reads every row (sequential scan) to find the matching ones.

**The core insight**: a B-tree index on `user_id` lets the engine jump directly to the matching rows in O(log n) time instead of O(n). The tradeoff: indexes consume disk space and slow down writes (the index must be updated on every insert/update/delete).

**The pattern**:
```sql
-- Create index for point lookup and range queries
CREATE INDEX idx_events_user_id ON events (user_id);

-- Composite index: column order matters
-- This index speeds up queries filtering by (user_id) or (user_id, event_time)
-- It does NOT speed up queries filtering only by event_time
CREATE INDEX idx_events_user_time ON events (user_id, event_time DESC);
```

**What breaks**: an index on `user_id` is useless if your query wraps the column in a function: `WHERE DATE(event_time) = '2024-01-01'` cannot use an index on `event_time`. Write it as `WHERE event_time >= '2024-01-01' AND event_time < '2024-01-02'` instead, or create a functional index on `DATE(event_time)`.

---

### EXPLAIN and reading query plans

**The problem**: you've added an index but the query is still slow. You need to understand what the engine is actually doing — whether it's using the index, how many rows it estimates, and where the time is going.

**The core insight**: `EXPLAIN` shows the query execution plan without running the query. `EXPLAIN ANALYZE` runs the query and shows both estimated and actual row counts and timing. The gap between estimated and actual rows is the primary signal for optimizer problems.

**The pattern**:
```sql
EXPLAIN ANALYZE
SELECT u.user_id, COUNT(o.order_id)
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE u.signup_date >= '2024-01-01'
GROUP BY u.user_id;
```

What to look for:
- `Seq Scan` on a large table with a restrictive filter → missing index
- `Nested Loop` where one side has many rows → should be a Hash Join or Merge Join
- `rows=1000000 actual rows=3` → the optimizer's estimate is wildly wrong (stale statistics; run `ANALYZE table`)
- High `actual time` on a node that appears early in the plan → bottleneck is there, not at the top

**What breaks**: `EXPLAIN` estimates are based on table statistics collected by the autovacuum process. After a large bulk load, statistics may be stale and the optimizer makes bad choices. Run `ANALYZE events;` to refresh statistics immediately after bulk loads.

---

### Optimization patterns

**Partition pruning**:
```sql
-- For a date-partitioned table, always filter on the partition column
-- This tells the engine to skip entire partitions
WHERE event_date >= '2024-01-01' AND event_date < '2024-02-01'
-- Without this filter: full table scan across all partitions
```

**Predicate pushdown** — filter early to reduce rows before expensive operations:
```sql
-- Good: filter happens before the join
WITH recent_events AS (
  SELECT * FROM events WHERE event_time >= CURRENT_DATE - INTERVAL '7 days'
)
SELECT u.name, COUNT(e.event_id)
FROM users u
JOIN recent_events e ON u.user_id = e.user_id
GROUP BY u.name;
```

**Avoid SELECT \***: in columnar stores (BigQuery, Snowflake, Redshift), each column is stored separately. `SELECT *` reads all columns from disk; selecting only needed columns can reduce I/O by 90%+.

---

## 9. Funnel Analysis

**The problem**: users go through a sequence of steps (page view → add to cart → checkout → purchase). You want to know how many users reach each step and the conversion rate between steps. The challenge: events are in a long-format table (one row per event), but you need one row per user with a flag for each step.

**The core insight**: pivot from event rows to user rows using conditional aggregation. `MAX(CASE WHEN event = 'X' THEN 1 ELSE 0 END)` per user is 1 if the user ever did that event. Then sum those flags across all users.

**The pattern**:
```sql
WITH user_steps AS (
  SELECT user_id,
         MAX(CASE WHEN event = 'page_view'    THEN 1 ELSE 0 END) AS step1,
         MAX(CASE WHEN event = 'add_to_cart'  THEN 1 ELSE 0 END) AS step2,
         MAX(CASE WHEN event = 'checkout'     THEN 1 ELSE 0 END) AS step3,
         MAX(CASE WHEN event = 'purchase'     THEN 1 ELSE 0 END) AS step4
  FROM events
  WHERE event_time >= '2024-01-01'
  GROUP BY user_id
)
SELECT
  SUM(step1)                                              AS reached_step1,
  SUM(step2)                                              AS reached_step2,
  SUM(step3)                                              AS reached_step3,
  SUM(step4)                                              AS reached_step4,
  ROUND(100.0 * SUM(step2) / NULLIF(SUM(step1), 0), 2)  AS step1_to_2_pct,
  ROUND(100.0 * SUM(step3) / NULLIF(SUM(step2), 0), 2)  AS step2_to_3_pct,
  ROUND(100.0 * SUM(step4) / NULLIF(SUM(step3), 0), 2)  AS step3_to_4_pct
FROM user_steps;
```

**What breaks**: this "ever did the step" logic doesn't enforce ordering — a user who purchased before viewing the page would still count as completing step 1. For strict ordering, add `MIN(CASE WHEN event = 'X' THEN event_time END)` per step and then filter users where `step2_time > step1_time`, etc.

---

## 10. Cohort Retention

**The problem**: you acquired 10,000 users in week 1. By week 4, how many of them are still active? How does that compare to users acquired in week 2? This is cohort retention — tracking a group of users over time from a common starting point.

**The core insight**: every user has a cohort (their first-activity week). For each (cohort, subsequent week) pair, count how many cohort members were active. Divide by cohort size to get retention rate.

**The pattern**:
```sql
WITH cohorts AS (
  -- Each user's cohort = the week they first appeared
  SELECT user_id,
         DATE_TRUNC('week', MIN(event_time)) AS cohort_week
  FROM events
  GROUP BY user_id
),
activity AS (
  -- All (user, week) pairs where the user was active
  SELECT DISTINCT user_id,
         DATE_TRUNC('week', event_time) AS activity_week
  FROM events
),
retention_raw AS (
  SELECT
    c.cohort_week,
    DATEDIFF('week', c.cohort_week, a.activity_week) AS weeks_since_start,
    COUNT(DISTINCT a.user_id) AS retained_users
  FROM cohorts c
  JOIN activity a ON c.user_id = a.user_id
  GROUP BY 1, 2
),
cohort_sizes AS (
  SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
  FROM cohorts
  GROUP BY 1
)
SELECT
  r.cohort_week,
  r.weeks_since_start,
  r.retained_users,
  cs.cohort_size,
  ROUND(100.0 * r.retained_users / cs.cohort_size, 1) AS retention_pct
FROM retention_raw r
JOIN cohort_sizes cs ON r.cohort_week = cs.cohort_week
ORDER BY r.cohort_week, r.weeks_since_start;
```

The result is a triangle-shaped matrix: cohort weeks as rows, `weeks_since_start` as columns, retention % as values. Week 0 is always 100% (every user was active in their signup week).

**What breaks**: using `JOIN` instead of `LEFT JOIN` in `retention_raw` means weeks with zero retention don't appear as rows (they simply don't exist in the join result). If you need explicit zeros in the matrix, use `CROSS JOIN` between all (cohort_week, weeks_since_start) combinations, then `LEFT JOIN` the actual activity.

---

## 11. Sessionization

**The problem**: you have a stream of user events with timestamps. There's no "session start" or "session end" event — just a sequence of actions. You want to group consecutive events into sessions, where a gap of more than 30 minutes between events defines a session boundary.

**The core insight**: three steps — (1) use `LAG` to find the time gap between each event and the previous one, (2) flag each event as a session start if the gap exceeds the threshold (or if it's the first event), (3) use a running `SUM` of session-start flags to assign a monotonically increasing session ID within each user.

**The pattern**:
```sql
WITH gaps AS (
  SELECT
    user_id,
    event_time,
    event_type,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event_time
  FROM events
),
boundaries AS (
  SELECT
    user_id,
    event_time,
    event_type,
    CASE
      WHEN prev_event_time IS NULL THEN 1         -- first event ever for this user
      WHEN DATEDIFF('minute', prev_event_time, event_time) > 30 THEN 1
      ELSE 0
    END AS is_session_start
  FROM gaps
),
sessions AS (
  SELECT
    user_id,
    event_time,
    event_type,
    SUM(is_session_start) OVER (
      PARTITION BY user_id
      ORDER BY event_time
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_id
  FROM boundaries
)
-- Summarize sessions
SELECT
  user_id,
  session_id,
  MIN(event_time)  AS session_start,
  MAX(event_time)  AS session_end,
  DATEDIFF('minute', MIN(event_time), MAX(event_time)) AS duration_minutes,
  COUNT(*)         AS events_in_session
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

**What breaks**: a session with only one event has `session_start = session_end` and `duration_minutes = 0`. If you want to estimate single-event session duration, you either assume a fixed duration (e.g., 30 seconds) or exclude them from duration statistics. Also, the 30-minute threshold is a convention — adjust it to your product's typical engagement pattern.

---

## 12. Deduplication

**The problem**: your raw events table has duplicates from a retry mechanism or a double-fire in the client SDK. You want to keep exactly one copy of each event, preferring the most complete row (by `updated_at`).

**The core insight**: use `ROW_NUMBER()` partitioned by the logical key, ordered by the column that determines which copy to keep. Then filter to row number 1.

**The pattern** (read query — no mutation):
```sql
WITH ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY user_id, event_type, DATE(event_time)
           ORDER BY updated_at DESC
         ) AS rn
  FROM events_raw
)
SELECT * FROM ranked WHERE rn = 1;
```

**Pattern** (in-place deletion in PostgreSQL):
```sql
-- Keep the row with the smallest ctid (physical row ID) per logical duplicate
DELETE FROM events_raw
WHERE id NOT IN (
  SELECT MIN(id)
  FROM events_raw
  GROUP BY user_id, event_type, DATE(event_time)
);
```

**What breaks**: `ROW_NUMBER` with `ORDER BY updated_at DESC` breaks ties arbitrarily when two duplicate rows have the same `updated_at`. If you truly can't distinguish which is "better," add a secondary sort (e.g., `ORDER BY updated_at DESC, id DESC`) to at least make the choice deterministic. Non-deterministic deduplication produces different results on repeated runs, which breaks reproducibility.

---

## 13. Metrics Queries

### DAU / WAU / MAU

**The problem**: the most basic product health metrics. How many distinct users were active on a given day, week, or month?

**The pattern**:
```sql
-- DAU
SELECT DATE(event_time) AS dt,
       COUNT(DISTINCT user_id) AS dau
FROM events
GROUP BY 1
ORDER BY 1;

-- WAU
SELECT DATE_TRUNC('week', event_time) AS week_start,
       COUNT(DISTINCT user_id) AS wau
FROM events
GROUP BY 1
ORDER BY 1;

-- MAU
SELECT DATE_TRUNC('month', event_time) AS month_start,
       COUNT(DISTINCT user_id) AS mau
FROM events
GROUP BY 1
ORDER BY 1;
```

**What breaks**: `COUNT(DISTINCT user_id)` over a window function is not supported in most engines — you can't write `COUNT(DISTINCT user_id) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`. For rolling distinct counts, you need an approximation (`HLL`/`HyperLogLog`) or a subquery approach (see rolling actives below).

---

### Rolling actives (7-day and 28-day)

**The problem**: MAU aggregated to monthly boundaries is a coarse metric — it jumps discontinuously at month boundaries. Rolling 28-day actives give a smoother, continuously updated signal: "users active in the last 28 days as of today."

**The core insight**: for each calendar date, count distinct users who had any event in the 27 preceding days including that date. This requires joining each date to a window of events, not a window function (because window functions can't apply `COUNT DISTINCT` over a frame).

**The pattern**:
```sql
WITH daily_users AS (
  -- One row per (user, day) — deduplicate within day
  SELECT DISTINCT user_id,
         DATE(event_time) AS activity_date
  FROM events
),
date_spine AS (
  SELECT generate_series(
    '2024-01-01'::date,
    '2024-12-31'::date,
    '1 day'
  )::date AS dt
)
SELECT d.dt,
       COUNT(DISTINCT u.user_id) AS rolling_28d_actives
FROM date_spine d
JOIN daily_users u
  ON u.activity_date BETWEEN d.dt - INTERVAL '27 days' AND d.dt
GROUP BY d.dt
ORDER BY d.dt;
```

**What breaks**: this query joins every date to up to 28 days of user activity — on a large events table this is expensive. In production, pre-aggregate to a `(user_id, activity_date)` distinct table first (as `daily_users` does here), then apply the rolling join on the smaller pre-aggregated table.

---

### Stickiness (DAU/MAU)

**The problem**: absolute MAU growth can mask declining engagement — if users are just visiting once a month, growth looks good but the product isn't sticky. DAU/MAU ratio normalizes engagement: 0.2 means users visit an average of 20% of days in the month.

**The pattern**:
```sql
WITH dau AS (
  SELECT DATE(event_time) AS dt,
         COUNT(DISTINCT user_id) AS daily_users
  FROM events
  GROUP BY 1
),
mau AS (
  SELECT DATE_TRUNC('month', event_time) AS month_start,
         COUNT(DISTINCT user_id) AS monthly_users
  FROM events
  GROUP BY 1
)
SELECT d.dt,
       d.daily_users,
       m.monthly_users,
       ROUND(1.0 * d.daily_users / NULLIF(m.monthly_users, 0), 4) AS stickiness_ratio
FROM dau d
JOIN mau m ON DATE_TRUNC('month', d.dt) = m.month_start
ORDER BY d.dt;
```

---

### Revenue metrics (ARPU, ARPPU)

**The problem**: two products both have $1M monthly revenue. One has 1M users (low ARPU, broad), the other has 10K users (high ARPU, niche). Average revenue per user reveals the difference.

**The pattern**:
```sql
SELECT DATE_TRUNC('month', order_date) AS month_start,
       COUNT(DISTINCT user_id)         AS paying_users,
       SUM(revenue)                    AS total_revenue,
       -- ARPU: total revenue / all active users (not just payers)
       SUM(revenue) / NULLIF(COUNT(DISTINCT user_id), 0) AS arppu,
       -- True ARPU requires joining to the full active user count
       SUM(revenue)                    AS numerator_for_arpu
FROM orders
WHERE status = 'completed'
GROUP BY 1
ORDER BY 1;
```

---

### Percentile latency (p50, p95, p99)

**The problem**: average latency is misleading — a fast median with a slow tail looks fine in averages but users at the p99 are having a terrible experience. You need percentile distributions to understand the tail.

**The core insight**: `PERCENTILE_CONT(p)` computes the p-th percentile using linear interpolation between adjacent values. `PERCENTILE_DISC(p)` returns the nearest actual value. For latency SLOs you typically want `PERCENTILE_CONT`.

**The pattern**:
```sql
SELECT
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms) AS p90_ms,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms,
  MAX(latency_ms)                                           AS max_ms
FROM request_logs
WHERE request_date = CURRENT_DATE;
```

To compute percentiles per endpoint:
```sql
SELECT endpoint,
       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
FROM request_logs
WHERE request_date = CURRENT_DATE
GROUP BY endpoint
ORDER BY p99_ms DESC;
```

**What breaks**: `PERCENTILE_CONT` is an ordered-set aggregate, not a window function — you can't write it with an `OVER` clause to compute a running percentile. For approximate percentiles at scale, use `APPROX_QUANTILES` (BigQuery) or `APPROX_PERCENTILE` (Snowflake), which use probabilistic data structures and run much faster on billions of rows.

---

### Funnel + cohort combined: LTV by acquisition channel

**The problem**: you want to know which acquisition channel produces users with the highest lifetime value, to decide where to increase ad spend.

**The pattern**:
```sql
WITH first_touch AS (
  -- Each user's acquisition channel = their first event's utm_source
  SELECT user_id,
         FIRST_VALUE(utm_source) OVER (
           PARTITION BY user_id
           ORDER BY event_time
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS channel
  FROM events
  QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) = 1
),
user_revenue AS (
  SELECT user_id,
         SUM(amount) AS ltv
  FROM orders
  WHERE status = 'completed'
  GROUP BY user_id
)
SELECT
  f.channel,
  COUNT(DISTINCT f.user_id)       AS users,
  COUNT(DISTINCT r.user_id)       AS paying_users,
  ROUND(100.0 * COUNT(DISTINCT r.user_id)
        / NULLIF(COUNT(DISTINCT f.user_id), 0), 1) AS conversion_pct,
  ROUND(AVG(r.ltv), 2)           AS avg_ltv_paying,
  ROUND(SUM(r.ltv)
        / NULLIF(COUNT(DISTINCT f.user_id), 0), 2) AS avg_ltv_all_users
FROM first_touch f
LEFT JOIN user_revenue r ON f.user_id = r.user_id
GROUP BY f.channel
ORDER BY avg_ltv_all_users DESC;
```

`QUALIFY` (Snowflake/BigQuery syntax) is a post-window filter equivalent to wrapping in a subquery and filtering on the row number — it avoids an extra CTE level.

**What breaks**: `avg_ltv_paying` and `avg_ltv_all_users` tell different stories. A channel with high `avg_ltv_paying` but low conversion is not necessarily better than a channel with moderate LTV and high conversion. Always report both and compute expected value (`avg_ltv_all_users`) for budget allocation decisions.

## Flashcards

**ROWS BETWEEN: counts physical rows?** #flashcard
6 PRECEDING means the 6 rows immediately above in the ordered result

**RANGE BETWEEN: includes all rows whose ORDER BY value falls within the range?** #flashcard
if two rows share the same date, both are included or excluded together

**UNBOUNDED PRECEDING?** #flashcard
all rows from the start of the partition

**UNBOUNDED FOLLOWING?** #flashcard
all rows to the end of the partition

**Seq Scan on a large table with a restrictive filter → missing index?** #flashcard
Seq Scan on a large table with a restrictive filter → missing index

**Nested Loop where one side has many rows → should be a Hash Join or Merge Join?** #flashcard
Nested Loop where one side has many rows → should be a Hash Join or Merge Join

**rows=1000000 actual rows=3 → the optimizer's estimate is wildly wrong (stale statistics; run ANALYZE table)?** #flashcard
rows=1000000 actual rows=3 → the optimizer's estimate is wildly wrong (stale statistics; run ANALYZE table)

**High actual time on a node that appears early in the plan → bottleneck is there, not at the top?** #flashcard
High actual time on a node that appears early in the plan → bottleneck is there, not at the top
