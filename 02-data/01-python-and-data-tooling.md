---
module: Foundations
topic: Python and Data Tooling
subtopic: ""
status: unread
tags: [foundations, python, numpy, pandas, polars, scipy, vectorization, performance]
---
# Python and Data Tooling

**For:** Engineers who know ML theory but need fluency in the libraries that actually move data — NumPy, Pandas, Polars, SciPy — plus the vectorization and memory habits that separate a script that runs in 2 seconds from one that runs in 2 minutes.
**Use:** A working reference, not a tutorial. Code blocks are runnable; read the "why it matters" lines for the parts interviewers actually probe.

---

## 1. NumPy: The Substrate Everything Else Sits On

Every array library in the Python ML stack (PyTorch tensors, Pandas columns, JAX arrays) borrows NumPy's mental model: a contiguous block of typed memory plus a shape/stride descriptor.

### 1.1 Arrays vs. Python Lists
```python
import numpy as np

a = np.array([1, 2, 3], dtype=np.float32)
a.shape, a.dtype, a.strides   # (3,), float32, (4,)
```
- A Python list stores pointers to boxed objects (each `int` has overhead ~28 bytes). A NumPy array stores raw values contiguously — no per-element overhead, no pointer chasing.
- **Why it matters:** this is *why* NumPy is fast. Operations run in C over contiguous memory instead of looping in the Python interpreter.

### 1.2 Vectorization
```python
# Slow: Python-level loop, ~100x slower for large N
out = [x**2 for x in range(1_000_000)]

# Fast: vectorized, single C loop under the hood
x = np.arange(1_000_000)
out = x**2
```
**Vectorization** means expressing a computation as array-level operations instead of explicit Python loops, so the loop happens in compiled C/Fortran (BLAS/LAPACK) rather than the interpreter. Every `for` loop over array elements in Python is a code smell in numerical code.

### 1.3 Broadcasting
Broadcasting lets NumPy operate on arrays of different shapes without copying data, by "stretching" the smaller array's dimensions virtually.

**Rules** (compare shapes right-to-left):
1. If dimensions differ in rank, pad the smaller shape with 1s on the left.
2. Two dimensions are compatible if they're equal, or one of them is 1.
3. Size-1 dimensions are stretched to match, with no actual memory copy.

```python
a = np.ones((3, 4))       # shape (3, 4)
b = np.array([1, 2, 3, 4])  # shape (4,) -> broadcasts to (1, 4) -> (3, 4)
a + b                      # works, no copy of b made

c = np.ones((3, 1))
a + c                      # (3,4) + (3,1) -> (3,4), c's column is stretched
```
**Why it matters:** broadcasting is how you normalize a batch of images by per-channel mean/std, add a bias vector to every row of a matrix, or compute pairwise distances — all without explicit loops. Get the shapes wrong and you'll either get a silent bug (accidental broadcast to the wrong axis) or a `ValueError: operands could not be broadcast together`.

### 1.4 Views vs. Copies
```python
a = np.arange(10)
b = a[2:5]        # VIEW — shares memory with a
b[0] = 999
a[2]               # 999 — a was mutated!

c = a[[2, 3, 4]]   # COPY — fancy indexing always copies
c[0] = -1
a[2]               # unchanged
```
- **Slicing** (`a[2:5]`) returns a view — no data copied, mutations propagate back.
- **Fancy indexing** (`a[[2,3,4]]`, `a[mask]`) and most arithmetic always return copies.
- **Why it matters:** the #1 source of "why did my original array change?!" bugs in data pipelines. When in doubt, `.copy()` explicitly.

### 1.5 Memory Layout and Performance
```python
a = np.zeros((1000, 1000), order='C')  # row-major (NumPy default)
a.sum(axis=0)   # sums down columns — strides across rows, worse cache locality
a.sum(axis=1)   # sums across rows — contiguous memory, better cache locality
```
- NumPy arrays are **row-major (C order)** by default: elements in a row are contiguous in memory.
- Operating along the last axis (contiguous) is faster than along the first axis (strided) for large arrays — this is a cache-locality effect, not an algorithmic one.
- `np.ascontiguousarray()` forces a layout when you're about to hand data to a C extension (e.g. certain BLAS calls) that requires it.

### 1.6 dtype Discipline
```python
np.array([1, 2, 3]).dtype        # int64 on most platforms
np.array([1.0, 2, 3]).dtype      # float64
np.array([1, 2, 3], dtype=np.float32).nbytes  # 12 bytes vs 24 for float64
```
- Default float is `float64` (8 bytes/element). Deep learning almost always wants `float32` (or `float16`/`bfloat16` on GPU) — half the memory, often 2x the throughput.
- Mixing `int` and `float` silently upcasts to `float`; mixing `float32` and `float64` silently upcasts to `float64` — a common way to accidentally double your memory footprint in a data pipeline.

---

## 2. Pandas: Tabular Data

### 2.1 Series and DataFrame
A `DataFrame` is a dict of `Series` (columns) sharing an index; a `Series` is a NumPy array plus an index. Every Pandas performance question reduces to: "is this operation vectorized (fast, C-level) or row-wise (slow, Python-level)?"

```python
import pandas as pd

df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "revenue": [10.5, 0.0, 42.1],
})
```

### 2.2 Vectorized Ops vs. `apply` vs. `iterrows`
```python
# Fastest: vectorized (operates on underlying NumPy arrays in C)
df["revenue_log"] = np.log1p(df["revenue"])

# Slower: apply — still a Python-level loop under the hood, ~10-50x slower
df["revenue_log"] = df["revenue"].apply(np.log1p)

# Slowest: iterrows — boxes every row into a Series object, ~100x+ slower
for idx, row in df.iterrows():
    df.loc[idx, "revenue_log"] = np.log1p(row["revenue"])
```
**Why it matters:** `iterrows()` in a data pipeline is almost always a bug waiting to be reported as "why is this job so slow." Reach for vectorized ops → `.apply()` (last resort for row-wise logic that can't vectorize) → never `iterrows()`.

### 2.3 GroupBy: Split-Apply-Combine
```python
df.groupby("user_id")["revenue"].sum()                    # split, apply sum, combine
df.groupby("user_id").agg(total=("revenue", "sum"),
                           n_orders=("revenue", "count"))   # multiple aggregations
df.groupby("user_id")["revenue"].transform("mean")          # same-shape output, broadcasts group stat back to every row
```
- `.agg()` reduces each group to one row. `.transform()` returns a result the same length as the input — useful for "subtract the group mean from every row" style feature engineering without a manual join.
- **Complexity:** groupby-aggregate is roughly O(n) with a hash-based grouping step; avoid `.groupby().apply(custom_python_fn)` on large data — it's the DataFrame equivalent of `iterrows()`.

### 2.4 Merges and Joins
```python
pd.merge(orders, users, on="user_id", how="left")   # SQL-style join
```
| `how` | Behavior |
|---|---|
| `inner` | Only matching keys in both |
| `left` / `right` | All rows from one side, matched rows from the other, NaN for no match |
| `outer` | All rows from both, NaN where unmatched |

**Common bug:** a many-to-many join on a key with duplicates silently explodes row count (Cartesian product on the duplicated keys). Always check `len(result)` against expectations after a join.

### 2.5 Memory Optimization
```python
df.memory_usage(deep=True)                 # per-column memory, deep=True includes object overhead
df["category_col"] = df["category_col"].astype("category")  # dedups repeated strings
df["small_int"] = df["small_int"].astype("int8")             # downcast if range allows
pd.read_csv("f.csv", dtype={"id": "int32"}, usecols=["id", "value"])  # avoid loading unneeded columns at full width
```
- `object` dtype (generic Python strings) is the biggest memory hog — converting low-cardinality string columns to `category` can cut memory 10x+.
- Downcast numeric types (`int64` → `int32`/`int16`, `float64` → `float32`) when the value range allows.
- Read only needed columns (`usecols`) and set dtypes at read time rather than casting after — avoids materializing the oversized version first.

### 2.6 Chained Indexing and `SettingWithCopyWarning`
```python
# BAD: chained indexing — ambiguous whether df[...] returns a view or copy
df[df.revenue > 0]["flag"] = 1     # may silently not modify df at all

# GOOD: single indexing operation
df.loc[df.revenue > 0, "flag"] = 1
```
**Why it matters:** this is one of the most commonly misunderstood Pandas warnings. `df[condition]` may return a copy; assigning into that copy's column does nothing to the original. Always use `.loc[row_selector, col_selector] = value` for conditional assignment.

---

## 3. Polars: Fast DataFrames for the Modern ML Stack

Polars is a DataFrame library written in Rust with a **lazy, query-optimizing execution engine**. The API looks Pandas-like, but execution is closer to a SQL query planner: automatic multi-threading, Apache Arrow memory layout, and predicate/projection pushdown.

**Eager vs. lazy** is the core distinction interviewers probe: `pl.read_csv` runs immediately like Pandas. `pl.scan_csv(...).filter(...).select(...).collect()` instead builds a query plan first — Polars can push filters into the file scan and skip unused columns before reading anything, which is why it beats Pandas on large aggregations (often 5-10x on multi-GB groupbys).

```python
result = (
    pl.scan_csv("data.csv")           # LazyFrame, nothing read yet
    .filter(pl.col("revenue") > 0)    # predicate pushdown → skips rows at read time
    .select(["segment", "revenue"])   # projection pushdown → skips unused columns
    .group_by("segment")
    .agg(pl.col("revenue").sum())
    .collect()                        # NOW executes the optimized query
)
```

For data that doesn't fit in RAM, `collect(streaming=True)` processes in chunks at constant memory instead of materializing the full result.

### When to Use Which

| Situation | Use |
|---|---|
| Prototyping, < 1GB data | **Pandas** — simpler API, more tutorials |
| Production ETL, multi-GB data | **Polars** — faster, memory-efficient |
| Scikit-learn / XGBoost pipeline | **Pandas** — `.to_pandas()` from Polars if needed |
| PyTorch DataLoader | **NumPy arrays** or **PyTorch tensors** |
| SQL-like heavy aggregations | **Polars** — query optimizer wins big |
| Data doesn't fit in RAM | **Polars streaming** or **Dask** |

---

## 4. SciPy: The Numerical Toolkit Above NumPy

SciPy builds domain-specific numerical routines on top of NumPy arrays.

```python
from scipy import stats, optimize, sparse, linalg

# Statistics: hypothesis tests, distributions
stats.ttest_ind(group_a, group_b)               # two-sample t-test
stats.pearsonr(x, y)                             # correlation + p-value

# Optimization: general-purpose minimizers (used under the hood by many sklearn estimators)
optimize.minimize(loss_fn, x0=initial_guess, method="L-BFGS-B")

# Sparse matrices: critical for high-cardinality one-hot / TF-IDF features
X_sparse = sparse.csr_matrix(dense_array)        # compressed sparse row — fast row slicing
X_sparse = sparse.csc_matrix(dense_array)        # compressed sparse column — fast column slicing, fast for matrix-vector products in some solvers

# Linear algebra beyond NumPy's basics
linalg.svd(A)                                    # more solver options than np.linalg.svd
```
**Why sparse matrices matter:** a one-hot encoded categorical with 100K categories over 1M rows is a 100K x 1M dense matrix (800 GB in float64) — but as a sparse matrix (only non-zero entries stored) it's a few MB. Any feature matrix from text (TF-IDF, bag-of-words) or high-cardinality categoricals should be sparse, and most sklearn linear models accept sparse input directly.

---

## 5. Matplotlib and Seaborn: Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib: low-level, full control, the substrate Seaborn sits on
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y)
ax.set(xlabel="epoch", ylabel="loss", title="Training curve")

# Seaborn: high-level, statistical plots, operates directly on DataFrames
sns.histplot(data=df, x="revenue", hue="segment", kde=True)
sns.pairplot(df[["feat1", "feat2", "feat3", "target"]], hue="target")
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")   # correlation matrix — first EDA plot to make
```
**When to use which:** Seaborn for fast statistical exploration during EDA (distributions, correlations, categorical breakdowns) — it takes a DataFrame and a column name instead of raw arrays. Matplotlib for anything Seaborn doesn't have a recipe for, or when you need pixel-level control (custom annotations, multi-panel figures for a paper/report).

---

## 6. Performance and Memory: General Principles

| Principle | Example |
|---|---|
| Vectorize before you loop | Replace `for` loops over rows/elements with array/column ops |
| Avoid intermediate copies | Chained `.assign()` / boolean masks create temporary full-size copies — filter early to shrink the working set |
| Use the right dtype | `float32` over `float64` when precision allows; `category` over `object` for low-cardinality strings |
| Profile before optimizing | `%timeit` in Jupyter, `line_profiler`, or `df.info(memory_usage="deep")` — don't guess where the bottleneck is |
| Prefer built-in reductions | `df.sum()`, `np.mean()` are implemented in C; a Python `sum(x for x in ...)` is not |
| Chunk what doesn't fit in memory | `pd.read_csv(..., chunksize=100_000)` or switch to Polars lazy / Dask for out-of-core processing |

```python
# Quick profiling in a notebook
%timeit df["revenue"].apply(np.log1p)   # vectorized apply
%timeit np.log1p(df["revenue"].values)  # drop to raw NumPy array — often fastest

import sys
sys.getsizeof(df) / 1e6                  # rough object size in MB (use df.memory_usage(deep=True).sum() for accurate DataFrame size)
```

---

## Interview Questions

1. **Why is `df.apply(func, axis=1)` slow, and what's the vectorized alternative?**
   `apply(axis=1)` calls `func` once per row in a Python-level loop, boxing each row into a `Series` — no C-level speedup. Prefer column-wise vectorized ops (`df["a"] + df["b"]`), or `np.where`/`np.select` for conditional logic, reserving `.apply()` for logic that genuinely can't be vectorized.

2. **Explain NumPy broadcasting rules and give an example where they silently produce a wrong (but non-crashing) result.**
   See §1.3. Silent-bug example: adding a `(3,)` array to a `(3,1)` array broadcasts to `(3,3)` instead of the intended elementwise `(3,)` — shapes are broadcast-compatible but not what the author meant, and no error is raised.

3. **What's the difference between a view and a copy in NumPy, and why does it matter for a data pipeline?**
   See §1.4. Views share memory with the parent array (mutations propagate); copies don't. Mutating a view you thought was independent is a classic silent-corruption bug in multi-stage pipelines.

4. **Why would you convert a Pandas string column to `category` dtype?**
   Low-cardinality string columns store the same string many times as separate Python objects; `category` stores each unique value once and represents rows as integer codes — often a 10x+ memory reduction and faster groupby/comparison operations.

5. **When would you reach for Polars over Pandas?**
   Data at the edge of or beyond single-machine memory, heavy multi-threaded groupby/join workloads, or pipelines where lazy evaluation + predicate/projection pushdown avoids reading unnecessary data. Pandas otherwise wins on ecosystem maturity — most ML libraries expect Pandas/NumPy input directly.

6. **Why store a one-hot-encoded feature matrix as a sparse matrix instead of dense?**
   High-cardinality categoricals or bag-of-words/TF-IDF text features are mostly zeros; a sparse format (CSR/CSC) stores only non-zero entries, turning an infeasible dense memory footprint into a tractable one, and most sklearn linear estimators accept sparse input natively.

---

## Where to Next

- **Data cleaning, EDA, and pipelines** → [04-data-processing-and-eda.md](02-data-processing-and-eda.md)
- **Math foundations these operations implement** → [02-math-and-theory-foundations.md](../01-math-foundations/01-math-and-theory-foundations.md)
- **Classical algorithms that consume this data** → [02-classical-ml/](../02-classical-ml/)
