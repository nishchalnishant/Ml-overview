---
module: Classical Ml
topic: Data Preprocessing Snappy
subtopic: ""
status: unread
tags: [classicalml, ml, data-preprocessing-snappy]
---
# Data preprocessing & engineering (deep-dive)

**Cold open:** Models eat **what you feed them**. In production, the “bug” is often **train ≠ serve**, **leakage**, or **drift** — same class of pain as deploying config that only worked on your laptop.

**Pipeline analogy (Azure-friendly):** Raw data is **untrusted input**. Preprocessing is **validation + transformation stages** — `fit` on train is **defining the release policy**; `transform` on test/prod is **applying** it. Never **fit** on the test set unless you want fake confidence.

---

## 1. Data leakage — the silent prod killer

### Q: Why does leakage wreck “amazing” offline metrics?

**Direct answer:** Information from the **future**, the **label**, or the **test fold** sneaks into training. The model **cheats** on the exam; production has no answer key → metrics **collapse**.

**Workflow (memorize this order):**
1. **Split first** — `train` / `val` / `test` before transforms that peek at distribution.
2. **Fit on train only** — scalers, imputers, encoders learn stats from **training** rows.
3. **Transform everywhere** — apply the **same** fitted params to val/test/prod.

**Mini pop quiz:** *You scale using mean/std from **all** data before split. What broke?* → Information from test leaked into training via global statistics.

---

## 2. Missing data — delete, fill, or flag?

### Q: When is deletion OK vs. imputation?

| Move | Technique | Use when | Risk |
| :--- | :--- | :--- | :--- |
| **Drop** | Rows/columns | Missingness is huge or MNAR mess | Biased sample if missingness carries signal |
| **Impute** | Mean / median / mode | Low–moderate missing, MCAR/MAR-ish | Mean lies on skewed columns — prefer **median** |
| **Fancy** | MICE, k-NN impute | Features correlate | Cost + complexity |
| **Indicator** | “Was missing?” flag | Missingness **predicts** the target | More dimensions to manage |

**Styling analogy:** Deleting rows is **throwing out** damaged samples — fast, but maybe you threw out the **statement piece** (signal).

---

## 3. Scaling — standardize vs. normalize

### Q: Standardization vs. normalization?

- **Z-score (standardize):** Zero mean, unit variance — friends with **SVM, logistic regression, PCA** (Gaussian-ish assumptions / distance fairness).
- **Min–max [0,1]:** Bounded range — often seen with **neural nets** and some distance models where scale matters to bounded activations.

**Outliers — IQR quick fix:**
- $\text{IQR} = Q3 - Q1$
- Fences: $[Q1 - 1.5 \times \text{IQR},\; Q3 + 1.5 \times \text{IQR}]$
- **Winsorize / cap** often beats blind delete — keeps sample size, dulls extremes.

---

## 4. High-cardinality categories (10k cities, anyone?)

### Q: Strategies without blowing up dimensionality?

1. **One-hot** — fine when categories are **few**; explodes when cardinality is huge.
2. **Target encoding** — category → mean target; **danger:** overfit. Use **CV within train**, smoothing, regularization.
3. **Hashing** — fixed-size buckets; collisions trade off noise vs. memory — common in **streaming / online** setups.

**Thought experiment:** *Target encoding on full train before CV — what happens?* → The model **inhales** label info per category; **leakage-style** optimism. Always encode **inside** CV folds.

---

## 5. Drift — when to retrain (or fix the pipe)

### Q: How do you know it’s time to refresh?

**Direct answer:** Watch **covariate shift** — $P(X)$ changes (inputs drift) even if the **old** $P(y|X)$ story might too (**concept drift**). Offline model ages; **monitoring** tells you when.

**Detection (common):** PSI, K-S tests, compare **recent** vs. **training** distributions on key features.

**Action ladder:** Retrain → fix data **upstream** → change **features** → change **objective** — same triage as “is it code, config, or infra?”

---

> **Code path:** For sklearn `Pipeline` + `ColumnTransformer` patterns, see [ML coding patterns](../ml-interview-notes/coding.md) — one artifact, reproducible stages, fewer “works on my machine” ghosts.

## Flashcards

**Z-score (standardize): Zero mean, unit variance?** #flashcard
friends with SVM, logistic regression, PCA (Gaussian-ish assumptions / distance fairness).

**Min–max [0,1]: Bounded range?** #flashcard
often seen with neural nets and some distance models where scale matters to bounded activations.

**$\text{IQR} = Q3 - Q1$?** #flashcard
$\text{IQR} = Q3 - Q1$

**Fences?** #flashcard
$[Q1 - 1.5 \times \text{IQR},\; Q3 + 1.5 \times \text{IQR}]$

**Winsorize / cap often beats blind delete?** #flashcard
keeps sample size, dulls extremes.
