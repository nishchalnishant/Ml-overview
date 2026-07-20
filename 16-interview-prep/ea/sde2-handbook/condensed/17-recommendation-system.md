# Interview 17 — Large-Scale Collaborative Filtering Recommender (Condensed)

Build "Recommended Games for You" for EA Play (millions of users, thousands of games) using only implicit signals (installs + playtime), no explicit ratings.

## Core Architecture
```
Playtime logs (S3) → user-item implicit matrix
   → PySpark ALS (implicit, batch, nightly)
   → user embeddings → Redis (KV lookup)
   → game embeddings → in-memory NumPy matrix in FastAPI (5K games is small)
   → API: fetch user vec → dot product vs all games → filter owned → top 10
```
- **Model choice:** Implicit ALS (Matrix Factorization), not SVD — SVD assumes missing=0, ALS uses confidence weighting ($c_{ui}=1+\alpha \cdot playtime$) so it correctly handles unobserved vs "played but didn't like."
- **Why no vector DB:** 5,000 items × 64 dims = trivial in-memory matmul (<1ms); FAISS/Milvus would be over-engineering at this scale.
- **Training:** PySpark `ALS(implicitPrefs=True)` distributes the ~100B-cell matrix across cluster.

## Talking Points That Signal Seniority
- Proactively names this as an **implicit feedback** problem and picks Implicit ALS over standard SVD without prompting.
- States explicitly *why* in-memory dot product is correct here and *when* it'd break (states the ANN crossover point, e.g. 50K+ items).
- Flags the **item cold-start problem** unprompted and proposes a metadata-based warm-start (hybrid/LightFM/Two-Tower) as the long-term fix, plus a UI-rule override as the short-term fix.
- Calls out **popularity bias** ("Harry Potter problem") and proposes log-penalizing popular items.
- Proposes a graceful-degradation path (static "Top 10 Trending" cache) if Redis is down — treats availability as a first-class concern.
- Mentions an **online re-ranking layer** (e.g., XGBoost on top-50 ALS candidates) to capture session context between nightly retrains.
- Proposes an **explore/exploit slot** in the top-10 to generate unbiased telemetry and help new items escape cold start.
- Calculates Big-O of serving step unprompted ($O(M \cdot D)$) and ties it to why BLAS/NumPy is fast enough.

## Top 3 Tradeoffs
- **ALS vs Two-Tower:** ALS is cheap, convergence-guaranteed, no side-features; Two-Tower handles non-linear signals and metadata (age, genre) but costs much more to train/serve.
- **Implicit vs explicit feedback:** Explicit is clean but <1% coverage; implicit covers 100% of users but needs confidence weighting since high playtime ≠ guaranteed liking.
- **In-memory matmul vs ANN index:** Fine at 5K items; becomes a CPU bottleneck at ~50K+ items, forcing a move to FAISS/Redis Vector Search.

## Biggest Pitfall
Proposing to compute recommendations by joining the 20M-row table live per API request (or not recognizing this needs embeddings/matrix factorization at all) — this is the single fastest path from Hire to No Hire.
