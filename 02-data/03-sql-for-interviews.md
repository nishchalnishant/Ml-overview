---
module: Data
topic: SQL
subtopic: ""
status: unread
tags: [sql, joins, window-functions, aggregation, query-optimization, interview]
prerequisites: []
---
# SQL for Interviews: A Working Tutorial

> This is written for someone who knows *what* SQL is but hasn't spent real time writing queries. Every section builds on the last, uses one running example, and explains *why* the construct exists before showing the syntax. Read it in order the first time.

## The example database

Everything below uses this schema. Picture three tables:

**`employees`**
| id | name | dept_id | manager_id | salary | hire_date |
|---|---|---|---|---|---|
| 1 | Alice | 10 | NULL | 120000 | 2019-03-01 |
| 2 | Bob | 10 | 1 | 95000 | 2020-06-15 |
| 3 | Carol | 20 | 1 | 110000 | 2021-01-10 |
| 4 | Dave | 20 | 3 | 90000 | 2022-04-20 |
| 5 | Erin | 20 | 3 | 90000 | 2023-02-01 |

**`departments`**
| id | name |
|---|---|
| 10 | Engineering |
| 20 | Sales |
| 30 | Marketing |

**`orders`** (Sales made by employees, some employees have none)
| id | employee_id | order_date | amount |
|---|---|---|---|
| 1 | 3 | 2024-01-05 | 500 |
| 2 | 3 | 2024-01-06 | 700 |
| 3 | 3 | 2024-02-01 | 300 |
| 4 | 4 | 2024-01-15 | 200 |

Notice on purpose: Marketing (dept 30) has no employees, Bob and Alice have no orders, Dave and Erin both earn exactly 90000. These are the details interview questions are built around, so keep them in mind as we go.

---

## 1. SELECT, FROM, WHERE — reading one table

The absolute base case:

```sql
SELECT name, salary
FROM employees
WHERE salary > 100000;
```

Read this left to right in your head as "get me name and salary from employees, but only rows where salary is over 100000." That returns Alice (120000) and Carol (110000).

The important thing to internalize now, because it explains almost every confusing SQL behavior later: **the database does not execute your query in the order you typed it.** It actually runs roughly:

1. `FROM` — figure out which table(s) we're reading
2. `WHERE` — throw out rows that don't match
3. `SELECT` — compute the columns you asked for
4. `ORDER BY` — sort what's left

That's why this fails:

```sql
SELECT salary * 1.1 AS new_salary
FROM employees
WHERE new_salary > 100000;   -- ERROR: new_salary doesn't exist yet
```

`WHERE` runs *before* `SELECT` computes `new_salary`, so the alias doesn't exist yet as far as `WHERE` is concerned. You'd have to repeat the expression: `WHERE salary * 1.1 > 100000`. Keep this execution-order idea in your back pocket — we'll come back to it precisely in section 6 once you've seen `GROUP BY` and window functions, because it explains why certain things are illegal in SQL that "should" work.

`ORDER BY` and `LIMIT` come last:

```sql
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 2;
```

This sorts everyone by salary descending, then keeps only the top 2 rows: Alice, Carol.

---

## 2. JOIN — combining tables

Real questions almost never involve a single table, because real schemas normalize data across tables (an employee's department is a separate table, not a repeated string). So joining is the single most load-bearing skill in SQL interviews.

### Why joins exist

`employees.dept_id` is just a number (10, 20, 30). It's meaningless without `departments`, which maps that number to a name. A join is how you stitch related rows from two tables back together based on a shared key.

### INNER JOIN — "only where both sides match"

```sql
SELECT e.name, d.name AS dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.id;
```

Walk through what happens mechanically: for every employee row, the database looks for a department row where `d.id` equals that employee's `dept_id`, and glues them together into one wide row. `e` and `d` are just aliases (nicknames) so you don't have to type `employees`/`departments` everywhere — this matters a lot once you self-join a table against itself (section 2c).

Result: 5 rows, one per employee, each now carrying its department name. Marketing never appears — nobody works there, and plain `JOIN` (shorthand for `INNER JOIN`) only keeps rows where **both** sides have a match.

### LEFT JOIN — "keep everything on the left, even without a match"

Now suppose you want a report of *every department*, including ones with zero employees. Flip the direction:

```sql
SELECT d.name AS dept_name, e.name AS employee_name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id;
```

`LEFT JOIN` says: keep every row from `departments` (the "left" table, the one named first) no matter what. If there's no matching employee, fill the employee columns with NULL instead of dropping the row. Now Marketing shows up once, with `employee_name = NULL`.

**This is also how you answer "find X with no Y" questions** — an extremely common interview pattern:

```sql
-- Departments with no employees
SELECT d.name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
WHERE e.id IS NULL;
```

Think through why this works: the `LEFT JOIN` keeps all departments and fills NULLs where there's no employee match. Then `WHERE e.id IS NULL` keeps only the rows where that fill-in happened — i.e., departments that truly had no match. This pattern is called an **anti-join** and it's worth memorizing the shape, not just this instance of it.

**The gotcha that trips almost everyone up once:** if you put a condition on the right-hand table in `WHERE` instead of `ON`, you silently destroy the "keep everything" behavior:

```sql
-- BUG: this quietly becomes an INNER JOIN
SELECT d.name, e.name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id
WHERE e.salary > 50000;
```

Here's why: Marketing's employee columns are all NULL (that's the whole point of LEFT JOIN). Then `WHERE e.salary > 50000` evaluates `NULL > 50000`, which is `NULL` (not true, not false — see section 8), so that row gets filtered out. You've silently lost your "no match" rows, defeating the purpose of using LEFT JOIN at all. **Fix: put table-specific filters in the `ON` clause, not `WHERE`, when you need to preserve unmatched rows:**

```sql
SELECT d.name, e.name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.id AND e.salary > 50000;
```

Now the salary filter only affects *which employee rows attach*, not whether the department row survives.

### RIGHT JOIN and FULL OUTER JOIN

`RIGHT JOIN` is just `LEFT JOIN` with the tables' roles reversed — keep everything on the right, fill NULLs on the left. In practice almost nobody uses `RIGHT JOIN`; people just rewrite it as a `LEFT JOIN` by swapping the table order, because it reads more naturally. Know it exists for when an interviewer asks.

`FULL OUTER JOIN` keeps everything from both sides, filling NULLs on whichever side is missing. Use case: reconciling two lists where either side might have orphans (e.g., "which customers have orders with no matching customer record, and which customer records have no orders" — both directions at once).

### Self-join — a table joined to itself

The `manager_id` column in `employees` points to another row in the *same* table. To find "who's each person's manager," you join `employees` to itself using two different aliases:

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;
```

Read it as: treat one copy of the table as "the employee" (`e`) and another copy as "the manager" (`m`), then match each employee's `manager_id` to the manager copy's `id`. This returns Bob→Alice, Carol→Alice, Dave→Carol, Erin→Carol. Alice has no manager (`manager_id` is NULL), so she's dropped by this INNER JOIN — if you wanted her included with `manager = NULL`, you'd use `LEFT JOIN` instead.

A classic interview variant — "find employees who earn more than their manager":

```sql
SELECT e.name
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

Nobody in our data qualifies, but trace it mentally: for each employee, we now have their manager's salary sitting right there in the same joined row (`m.salary`), so a plain comparison in `WHERE` does the rest. This is the general lesson of self-joins: they let you compare a row to a *related* row in the same table without writing a loop.

### The join gotcha that produces silently wrong numbers: fan-out

Suppose you join `employees` to `orders` to compute total salary paid, but do it carelessly:

```sql
SELECT e.name, e.salary, o.amount
FROM employees e
JOIN orders o ON e.id = o.employee_id;
```

Carol has 3 orders. That means Carol's row appears 3 times in the result — once per matching order — each time repeating her full salary. If you then did `SUM(e.salary)` over this joined result without grouping carefully, you'd triple-count Carol's salary. This is called **fan-out**, and it's the single most common source of "my aggregate numbers are wrong but the query runs fine" bugs. The fix is almost always: aggregate one side down to one-row-per-key *before* joining, or be very deliberate about what you're summing after the join (sum `o.amount`, which is fine since amount genuinely repeats once per order — but never sum `e.salary` in that same joined result).

---

## 3. GROUP BY and aggregates — collapsing rows into summaries

A join widens rows (more columns, more rows sometimes). `GROUP BY` does the opposite: it collapses many rows down into one row per group, alongside a computed summary.

```sql
SELECT dept_id, AVG(salary) AS avg_salary, COUNT(*) AS headcount
FROM employees
GROUP BY dept_id;
```

Think of this as: bucket every employee row by `dept_id`, then for each bucket compute one average and one count. Result: dept 10 → avg 107500, count 2; dept 20 → avg (110000+90000+90000)/3 = 96666.67, count 3.

**The rule that governs `GROUP BY`:** every column in `SELECT` must either be (a) in the `GROUP BY` list, or (b) wrapped in an aggregate function (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`). The reason is conceptual, not arbitrary: once you've collapsed 3 Sales rows into 1 summary row, there's no single answer to "what was *the* name" for that row — you have to either group by name too (which would un-collapse it) or aggregate it somehow (`MIN(name)`, `STRING_AGG(name, ', ')`). SQL enforces this so you can't accidentally ask a question that has no well-defined answer.

**`COUNT(*)` vs `COUNT(column)` — a real interview gotcha.** `COUNT(*)` counts rows, full stop. `COUNT(column)` counts only the rows where that specific column is non-NULL. If you ran `COUNT(manager_id)` per department, Alice (NULL manager) wouldn't be counted, while `COUNT(*)` would include her. This distinction shows up constantly when computing rates like "% of users who did X" — you want `COUNT(x_column)`, not `COUNT(*)`, if some rows never did X and store NULL for it.

### HAVING — filtering *groups*, not rows

Say you want departments with more than 2 employees:

```sql
SELECT dept_id, COUNT(*) AS headcount
FROM employees
GROUP BY dept_id
HAVING COUNT(*) > 2;
```

Why not `WHERE COUNT(*) > 2`? Because of the execution order from section 1: `WHERE` runs *before* grouping happens, so at that point there is no `COUNT(*)` yet to filter on — individual rows haven't been bucketed into groups. `HAVING` exists specifically as the filter that runs *after* grouping, when aggregates finally exist. Rule of thumb: `WHERE` filters raw rows before they're grouped; `HAVING` filters the group summaries after.

This distinction is also how you find duplicates — one of the most common "gotcha" interview questions:

```sql
SELECT email, COUNT(*)
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

Read it as: group all rows sharing the same email, and only keep the groups where that count exceeds 1 — i.e., emails that appear more than once.

---

## 4. Window functions — per-row calculations that still see neighboring rows

This is the concept most people with SQL-adjacent-but-not-hands-on experience haven't really internalized, and it's the single highest-leverage thing to learn well for interviews, because a huge fraction of "hard" SQL questions are window-function questions in disguise.

### The problem window functions solve

`GROUP BY` collapses rows — you lose the individual employee once you've computed a department average. But what if you want *both*: each employee's own row, *and* some calculation relative to their group (their rank within the department, a running total, the previous row's value)? `GROUP BY` can't do this because it destroys the per-row detail. That's exactly the gap window functions fill.

### The anatomy of a window function

```sql
SELECT name, dept_id, salary,
       AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
```

Walk through this piece by piece:
- `AVG(salary)` — the calculation, same as a normal aggregate.
- `OVER (...)` — this is what makes it a *window* function instead of a normal aggregate. It tells the database "compute this, but don't collapse the rows — attach the result to every row instead."
- `PARTITION BY dept_id` — defines the "window" of rows the calculation considers for each row: group by department, the same way `GROUP BY` would, but without merging rows together.

Result: all 5 employee rows still exist individually, but Alice and Bob both show `dept_avg = 107500`, and Carol/Dave/Erin all show `dept_avg = 96666.67`. Each row got the average *of its own group*, attached to itself, without losing any rows. That's the entire concept — everything else about window functions is variations on this.

### Ranking within a group

A very common question: "find the top-paid employee in each department."

```sql
SELECT name, dept_id, salary,
       RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
FROM employees;
```

New piece: `ORDER BY salary DESC` *inside* the `OVER(...)`. This tells the window function what order to consider rows in *within each partition* — here, highest salary first, per department. `RANK()` then assigns 1 to the highest earner in each department, 2 to the next, and so on, restarting at 1 for each new department.

Trace it: dept 10 sorted by salary desc is Alice (120000) then Bob (95000) → ranks 1, 2. Dept 20 sorted is Carol (110000), then Dave and Erin tied at 90000 → Carol gets rank 1, and here's the interesting part — Dave and Erin tie.

**Three ranking functions differ only in how they handle that tie**, and this is a favorite interview question because it's easy to get wrong from memory:

- `ROW_NUMBER()` — ignores the tie entirely, arbitrarily assigns 2 and 3. Always produces unique numbers 1,2,3,4...
- `RANK()` — gives both tied rows rank 2, then *skips* to rank 4 for the next row (as if two rows occupied "2" and "3"). Sequence: 1, 2, 2, 4.
- `DENSE_RANK()` — gives both tied rows rank 2, but the *next distinct* value continues at 3, no gap. Sequence: 1, 2, 2, 3.

Which one you want depends on the question. "Give me exactly the top 3 salaries, and if there's a 3-way tie for 3rd place, include all of them" → `DENSE_RANK()`. "Give me exactly 3 rows no matter what" → `ROW_NUMBER()`.

### Filtering on a window function — why you need a subquery

Now, "find the top-paid employee *per department*" — you'd want to keep only `rnk = 1`. But you can't write:

```sql
SELECT name, dept_id, salary,
       RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
FROM employees
WHERE rnk = 1;   -- ERROR
```

This goes back to execution order (section 1), now with window functions added to the picture. The full order is roughly:

```
FROM/JOIN → WHERE → GROUP BY → HAVING → window functions → SELECT → ORDER BY → LIMIT
```

Window functions compute *after* `WHERE` has already run. So `WHERE` cannot see `rnk` — it doesn't exist yet at that stage. The fix is to compute the window function in an inner query, then filter in an outer query wrapped around it:

```sql
SELECT * FROM (
  SELECT name, dept_id, salary,
         RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
  FROM employees
) ranked
WHERE rnk = 1;
```

The inner query runs first, in its entirety, producing a temporary result set that already has `rnk` as a real column. *Then* the outer `WHERE rnk = 1` filters that — which is legal, because by that point `rnk` genuinely exists as data. **This subquery-wrapping pattern is the single most reusable trick in SQL interviews** — "top N per group" is just this with `rnk <= n` instead of `= 1`.

### Running totals and comparisons to previous rows

Two more building blocks, both extremely common:

**Running total** — replace `AVG` with `SUM`, drop `PARTITION BY`, keep `ORDER BY`:

```sql
SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders
WHERE employee_id = 3;
```

With `ORDER BY` but no frame specified, the default window is "from the very first row up to the current row" — so each row's `running_total` is the sum of itself and every row before it in date order. Row 1 (Jan 5, 500): running total 500. Row 2 (Jan 6, 700): running total 1200. Row 3 (Feb 1, 300): running total 1500. This is a cumulative sum, computed with no loop, no subquery.

**Comparing to the previous/next row** — `LAG` and `LEAD`:

```sql
SELECT order_date, amount,
       LAG(amount) OVER (ORDER BY order_date) AS prev_amount,
       amount - LAG(amount) OVER (ORDER BY order_date) AS change
FROM orders
WHERE employee_id = 3;
```

`LAG(amount)` reaches "one row back" (in the `ORDER BY` sequence) and pulls that row's amount into the current row, so you can do row-over-row math directly — no self-join required. `LEAD` is the same idea but reaches one row *forward*. This is exactly how you compute month-over-month growth: partition by whatever entity you're comparing within, order by the time period, and subtract `LAG`.

---

## 5. Subqueries and CTEs — building queries out of smaller queries

You've already seen a subquery (the "wrap it and filter outside" pattern above). Let's name the pieces properly.

A **subquery** is a query nested inside another query. There are a few flavors:

- **Scalar subquery** — returns exactly one value, usable anywhere a single value would go:
  ```sql
  SELECT name, salary
  FROM employees
  WHERE salary > (SELECT AVG(salary) FROM employees);
  ```
  The inner query runs once, computes one number (the company-wide average), and the outer query uses it like a constant.

- **Correlated subquery** — references a column from the *outer* query, so it can't be computed once up front; it re-runs conceptually for every outer row:
  ```sql
  SELECT name, dept_id, salary
  FROM employees e
  WHERE salary > (
    SELECT AVG(salary) FROM employees e2 WHERE e2.dept_id = e.dept_id
  );
  ```
  For each employee `e`, the inner query recomputes the average salary *of that employee's own department* (`e2.dept_id = e.dept_id` ties it back to the outer row) and compares. This answers "who earns above their department's average" — Alice easily could be answered with a window function instead (`AVG(salary) OVER (PARTITION BY dept_id)`), and in practice, window functions usually replace correlated subqueries because they're clearer and the database can often run them faster. Recognize both forms; prefer window functions when writing your own answer.

### CTEs — naming your subqueries

A **CTE** (Common Table Expression, written with `WITH`) is a named, temporary result set you define before your main query, mainly for readability:

```sql
WITH dept_avg AS (
  SELECT dept_id, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY dept_id
)
SELECT e.name, e.salary, d.avg_salary
FROM employees e
JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_salary;
```

Nothing here is a new *capability* over subqueries — you could nest this as `FROM employees e JOIN (SELECT ...) d`. The difference is purely readability: you give the intermediate result a name (`dept_avg`) and refer to it like a table, which is far easier to read once you're nesting more than one level. In interviews, defaulting to CTEs over deeply nested subqueries is generally rewarded because it shows you write maintainable SQL, not just correct SQL.

### Recursive CTEs — walking a hierarchy

The `employees.manager_id` chain is a tree (Alice → Bob/Carol → Dave/Erin). To get "everyone under Alice, at every level," you need a query that can walk an *unknown* number of levels deep — a normal join can't do that because you'd have to hardcode the number of self-joins. A recursive CTE solves this:

```sql
WITH RECURSIVE org AS (
  -- base case: start at the top (no manager)
  SELECT id, name, manager_id, 1 AS level
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- recursive step: find employees whose manager is already in `org`
  SELECT e.id, e.name, e.manager_id, o.level + 1
  FROM employees e
  JOIN org o ON e.manager_id = o.id
)
SELECT * FROM org ORDER BY level;
```

Trace the mechanics: the "base case" seeds the recursion with Alice (`level = 1`, no manager). The "recursive step" then joins `employees` back against `org` itself — the set being built — matching any employee whose `manager_id` is someone already found. The database repeats the recursive step, feeding each round's new rows back in, until a round finds nothing new to add, then stops. Round 1 finds Bob and Carol (level 2, managed by Alice). Round 2 finds Dave and Erin (level 3, managed by Carol). Round 3 finds nothing new, so it stops. This pattern — org charts, category trees, bill-of-materials — is the standard use case for recursive CTEs.

---

## 6. Execution order, precisely

Now that you've seen `WHERE`, `GROUP BY`, `HAVING`, window functions, and `ORDER BY` in action, here's the full picture in one place:

```
1. FROM / JOIN     — assemble the rows, combine tables
2. WHERE           — discard rows that don't match (no aggregates/window results exist yet)
3. GROUP BY        — bucket remaining rows into groups
4. HAVING          — discard groups that don't match (aggregates exist now)
5. window functions — compute per-row values over the (grouped or ungrouped) result
6. SELECT          — compute final output columns/aliases
7. DISTINCT        — remove duplicate output rows
8. ORDER BY        — sort (aliases from SELECT are usable here)
9. LIMIT / OFFSET  — trim to the requested rows
```

Every "why can't I do X" confusion in SQL — can't filter a window function in `WHERE`, can't use a `SELECT` alias in `WHERE`, can use it in `ORDER BY`, `HAVING` sees aggregates but `WHERE` doesn't — is a direct consequence of this order. When you get stuck on why a query won't run, walk through this list and ask "does the thing I'm referencing exist yet at this step?"

---

## 7. NULL — the thing that silently breaks queries

NULL means "unknown/absent," not zero or empty string, and it behaves unlike any other value:

- `salary > NULL` is not `false` — it's `NULL` (unknown). You genuinely cannot know if an unknown salary is greater than something.
- `NULL = NULL` is also `NULL`, not `true`. Two unknowns aren't provably equal.
- Because of this, `WHERE col = NULL` never matches anything, ever — you must write `WHERE col IS NULL`.
- Aggregates quietly skip NULLs: `AVG(salary)` ignores rows where salary is NULL rather than treating them as 0. This matters — if you actually wanted NULL treated as 0 for the average, you'd write `AVG(COALESCE(salary, 0))`.

Two small functions you'll use constantly:
- `COALESCE(a, b, c)` — returns the first non-NULL value in the list. Common use: `COALESCE(nickname, name)` to fall back to a default.
- `NULLIF(a, b)` — returns NULL if `a` equals `b`, otherwise returns `a`. Common use: guarding against divide-by-zero, `total / NULLIF(count, 0)`, so a zero denominator produces NULL instead of crashing.

This is why the fan-out/LEFT JOIN gotchas from section 2 happen — they're all really NULL-comparison gotchas wearing a join costume.

---

## 8. Putting it together: worked interview questions

**"Find the 2nd highest salary, without assuming there are no ties."**

Wrong instinct: `ORDER BY salary DESC LIMIT 1 OFFSET 1` — this silently breaks if there's a tie for 1st place, because both rows share "1st," and offsetting by 1 skips one of them, landing you on what's actually still the highest distinct value tied, not the true second-distinct value. Instead, rank by *distinct* values:

```sql
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

Or, using what you now know about window functions and the "wrap and filter" pattern:

```sql
SELECT salary FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
  FROM employees
) t
WHERE rnk = 2;
```

`DENSE_RANK()` is the right choice here specifically because it doesn't skip numbers on ties — "2nd highest distinct value" is exactly what `rnk = 2` means under `DENSE_RANK`.

**"Find employees with no orders."** — the anti-join pattern from section 2:

```sql
SELECT e.name
FROM employees e
LEFT JOIN orders o ON e.id = o.employee_id
WHERE o.id IS NULL;
```

**"For each employee with orders, show a running total of their order amounts over time."** — window function with `PARTITION BY` (per employee) and `ORDER BY` (over time), from section 4:

```sql
SELECT e.name, o.order_date, o.amount,
       SUM(o.amount) OVER (PARTITION BY e.id ORDER BY o.order_date) AS running_total
FROM employees e
JOIN orders o ON e.id = o.employee_id;
```

**"Find consecutive days a user was active"** (gaps and islands — a step up in difficulty, common at senior levels). The trick: if you subtract a per-user, date-ordered `ROW_NUMBER()` from the date itself, consecutive dates all land on the *same* result, because both the date and the row number increase by exactly 1 each consecutive day — the difference stays constant:

```sql
WITH numbered AS (
  SELECT user_id, login_date,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM logins
),
grouped AS (
  SELECT user_id, login_date,
         login_date - rn AS island_id   -- constant within a consecutive run
  FROM numbered
)
SELECT user_id, MIN(login_date) AS streak_start, MAX(login_date) AS streak_end, COUNT(*) AS streak_len
FROM grouped
GROUP BY user_id, island_id;
```

Don't try to memorize this cold — instead, be able to re-derive it: you need something that's constant across a consecutive run and changes when the run breaks. `date - row_number` is the standard trick that produces exactly that, because a broken streak makes the row number "fall behind" the date by a different, new amount.

---

## 9. DDL, DML, and constraints — modifying structure and data

Everything above was reading data (`SELECT`). A smaller but still-tested part of interviews covers changing data and schema.

**DDL (schema)**: `CREATE TABLE`, `ALTER TABLE`, `DROP TABLE`, `TRUNCATE TABLE`. A frequent conceptual question is the difference between the three ways to empty/remove a table:
- `DELETE FROM employees WHERE ...` — removes matching rows, one at a time, logged so it's roll-back-able, and fires any triggers. Can be partial (has a `WHERE`).
- `TRUNCATE TABLE employees` — removes *all* rows at once, minimal logging, resets auto-increment counters, cannot be filtered. Much faster than `DELETE` on large tables, but all-or-nothing.
- `DROP TABLE employees` — removes the rows *and* the table structure itself. The table no longer exists at all afterward.

**DML (data)**: `INSERT`, `UPDATE`, `DELETE`, and "upsert" (insert-or-update) — `INSERT ... ON CONFLICT DO UPDATE` in Postgres, `ON DUPLICATE KEY UPDATE` in MySQL. Upsert exists because "insert this row, but if a row with this key already exists, update it instead" is an extremely common real-world need (e.g., syncing data) that would otherwise take a manual check-then-insert-or-update dance with a race condition.

**Constraints** are rules the database enforces on every write, so bad data can never get in regardless of which application wrote it:
- `PRIMARY KEY` — uniquely identifies each row; implies both "unique" and "not null." `employees.id` is a primary key.
- `FOREIGN KEY` — a column that must match a primary key in another table. `employees.dept_id` referencing `departments.id` is a foreign key; it stops you from inserting an employee with a department that doesn't exist.
- `UNIQUE` — no duplicates allowed, but (unlike primary key) can allow NULLs and a table can have several unique constraints.
- `NOT NULL`, `CHECK (condition)`, `DEFAULT value` — narrower per-column rules.

Interviewers sometimes ask what happens when you delete a department that still has employees pointing at it via foreign key — that's controlled by `ON DELETE CASCADE` (delete the employees too), `ON DELETE SET NULL` (set their `dept_id` to NULL), or `ON DELETE RESTRICT` (refuse the delete entirely, the safe default).

---

## 10. Transactions — grouping writes so they succeed or fail together

Imagine transferring money: subtract from account A, add to account B. If the process crashes between those two statements, money vanishes. A **transaction** groups statements so they commit together or not at all:

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

If anything goes wrong before `COMMIT`, you run `ROLLBACK` instead, and the database behaves as if neither `UPDATE` ever happened. This all-or-nothing guarantee is the "A" (**Atomicity**) in **ACID**, the four properties databases promise for transactions: Atomicity (all-or-nothing), Consistency (never leaves data in a state that violates constraints), Isolation (concurrent transactions don't see each other's half-finished work), Durability (once committed, survives a crash).

**Isolation levels** control exactly how much of *other* transactions' in-progress work you can see, trading correctness guarantees for speed — stricter isolation means more locking/waiting. You don't need to memorize implementation details, but you should be able to say what each level allows: `Read Committed` (the common default) prevents you from ever seeing another transaction's *uncommitted* changes, but if you query the same row twice within your own transaction, it might have changed in between (because someone else committed a change to it). `Repeatable Read` fixes that — a row you've already read stays the same for the rest of your transaction. `Serializable` is the strictest: the database behaves as if every transaction ran one at a time, with no overlap at all.

---

## 11. String, date, and conditional expressions

These come up constantly inside otherwise-standard queries, so it's worth being fluent rather than looking them up mid-interview:

- **`CASE WHEN ... THEN ... ELSE ... END`** is an if/else you can drop into a `SELECT`. It's the standard way to bucket continuous data or turn categories into flags:
  ```sql
  SELECT name,
         CASE WHEN salary >= 100000 THEN 'senior' ELSE 'standard' END AS pay_band
  FROM employees;
  ```
- Combine `CASE WHEN` with `SUM` to **pivot rows into columns** — turn one row per (user, event_type) into one row per user with a column per event type:
  ```sql
  SELECT employee_id,
         SUM(CASE WHEN amount > 400 THEN 1 ELSE 0 END) AS big_orders,
         SUM(CASE WHEN amount <= 400 THEN 1 ELSE 0 END) AS small_orders
  FROM orders
  GROUP BY employee_id;
  ```
  The trick: `CASE WHEN` turns each row into a 1 or 0 depending on a condition, and then `SUM` inside a `GROUP BY` adds those up — effectively a conditional count, computed per group.
- Date functions vary by dialect but the concepts are universal: truncating a timestamp down to a coarser bucket (`DATE_TRUNC('month', order_date)` in Postgres), extracting a piece (`EXTRACT(YEAR FROM order_date)`), and computing differences between two dates. If you only remember one thing, remember that "group orders by month" almost always means "group by a truncated/extracted version of the date column," not the raw timestamp — grouping by the raw timestamp would produce a separate group for every single distinct second.
- `COALESCE`/`NULLIF` were covered in section 7 — they're conditional expressions too, just NULL-specific ones.

---

## 12. A note on dialects

Everything above is portable across Postgres, MySQL, SQL Server, etc. — it's the common core. A few syntax spots differ, and interviewers mostly just want to see you're aware differences exist, not that you have them memorized:

| Concept | Postgres | MySQL | SQL Server |
|---|---|---|---|
| Limit rows | `LIMIT n OFFSET m` | `LIMIT n OFFSET m` | `TOP n` / `OFFSET...FETCH` |
| String concatenation | `\|\|` | `CONCAT(a, b)` | `+` |
| Upsert | `ON CONFLICT DO UPDATE` | `ON DUPLICATE KEY UPDATE` | `MERGE` |
| Auto-increment PK | `GENERATED ALWAYS AS IDENTITY` | `AUTO_INCREMENT` | `IDENTITY` |

If asked "what dialect should I assume," default to ANSI-standard/Postgres-flavored syntax (what this document uses) unless told otherwise — it's the closest to the portable standard and what most take-home platforms (LeetCode, HackerRank, etc.) default to as well.

---

## 13. Schema design concepts (sometimes asked alongside querying)

Two ideas worth having a one-paragraph answer ready for:

**Normalization** is the discipline of splitting data so each fact lives in exactly one place. Our `employees`/`departments` split is normalized — a department's name lives once, in `departments`, not copied onto every employee row. The benefit: update a department's name once, and every employee automatically reflects it, with no risk of two employee rows disagreeing about what department 20 is called. The cost: you now need a `JOIN` to see an employee's department name instead of just reading a column. Formal levels (1NF, 2NF, 3NF) all build on this same idea with increasingly strict rules about what's allowed to depend on what.

**OLTP vs OLAP** is the practical reason real systems sometimes *deliberately* break normalization. OLTP (transactional) systems — think a live app's production database — favor normalized schemas because they're optimized for many small, fast reads/writes of individual records. OLAP (analytical) systems — a data warehouse used for reporting — favor **denormalized** schemas (wide tables, redundant copies of dimension data) because they're optimized for scanning and aggregating millions of rows at once, and joins across many normalized tables would be slow at that scale. Recognizing which one a question is implicitly about tells you whether joins-and-normalization or denormalized-wide-table thinking is the right frame.

---

## Practice checklist

Work through these from memory, not by re-reading — that's the actual test of whether this has sunk in:

- [ ] Explain execution order and why a `SELECT` alias can't be used in `WHERE` but can in `ORDER BY`
- [ ] Write a `LEFT JOIN ... WHERE right.col IS NULL` anti-join from scratch
- [ ] Explain the fan-out bug and why aggregating after a one-to-many join can silently inflate a sum
- [ ] Write a 2nd-highest-value query that correctly handles ties, and explain why the naive `LIMIT 1 OFFSET 1` is wrong
- [ ] Write a top-N-per-group query using a window function wrapped in a subquery
- [ ] Explain, without a table in front of you, how `ROW_NUMBER`, `RANK`, and `DENSE_RANK` differ on a tie
- [ ] Write a running total and a month-over-month delta using window functions
- [ ] Re-derive the gaps-and-islands trick (`date - row_number`) rather than recalling it verbatim
- [ ] Write a recursive CTE for a manager hierarchy
- [ ] Explain why `AVG()` silently ignores NULLs and when that's the wrong behavior
- [ ] Explain `DELETE` vs `TRUNCATE` vs `DROP`
- [ ] Explain ACID and what isolation levels trade off against each other
- [ ] Write a `CASE WHEN` pivot query
- [ ] Give a one-paragraph answer for normalization's cost/benefit and OLTP vs OLAP
