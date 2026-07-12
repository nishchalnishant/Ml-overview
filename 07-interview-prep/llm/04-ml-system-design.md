---
module: Interview Prep
topic: Llm
subtopic: Ml System Design
status: unread
tags: [interviewprep, ml, llm-ml-system-design]
---
# ML System Design — Interview Playbook

---

## 1. The Universal Framework

Most ML system design failures come from one of two mistakes: picking a model before clarifying requirements, or skipping monitoring and rollback. Use this sequence:

```
Step 1: Clarify the product goal
    "What user action are we optimizing? Click, purchase, dwell time, safety?"
    "Is this ranking, classification, generation, or anomaly detection?"
    "What does failure look like? What's the cost of FP vs FN?"

Step 2: Define metrics (both layers)
    Online:  CTR, revenue/query, D7 retention, P99 latency, abuse rate
    Offline: AUC-ROC, NDCG@10, precision@K, perplexity, calibration error
    "What does success look like in a business dashboard 30 days post-launch?"

Step 3: State constraints explicitly
    Latency SLA, QPS, memory budget, cost, regulatory, team size

Step 4: Data and labels
    Volume, freshness needed, label quality, class balance
    Where does the ground truth come from? Delayed? Biased?
    Training-serving skew risks?

Step 5: Baseline
    The simplest thing that could work: popularity ranking, LR, BM25
    This becomes your control arm in the A/B test

Step 6: Architecture
    Justified by constraints + data, not by what's popular
    Two-stage if item corpus > 1M: retrieval → ranking

Step 7: Feature pipeline
    Real-time vs batch features; where are transforms fit?
    Fit on training data only — never on val/test (leakage)

Step 8: Serving
    Batch (offline pre-compute) vs real-time (sub-200ms)
    Caching strategy for expensive lookups

Step 9: Evaluation
    A/B test design: metric, power calculation, duration
    Novelty effect mitigation (users click new things, not better things)

Step 10: Monitoring
    PSI on input features; prediction score distribution
    Business KPI drop threshold for auto-rollback
```

---

## 2. Batch vs Real-Time Serving

### Why this choice matters

Getting it wrong costs you either way:
- **Real-time when batch would do:** wasted complexity building a low-latency pipeline for decisions that could be pre-computed.
- **Batch when real-time needed:** e.g. a fraud system scoring yesterday's transactions — every fraud during the gap goes undetected.

| Dimension | Batch | Real-Time |
| :--- | :--- | :--- |
| Latency | Minutes to hours | < 100–200ms |
| Freshness | Stale (hourly/daily) | Current context |
| Scale | Millions of items/hour | Moderate QPS |
| Model size | No constraint | Usually < 1GB |
| Infrastructure | Spark, Airflow, BigQuery | REST/gRPC, Redis, feature store |
| Use cases | Weekly churn scores, email campaigns | Fraud, autocomplete, ad ranking |

**Hybrid pattern** (most production systems): batch pre-compute heavy features offline, real-time assemble them at query time. Example: recommendation system pre-computes user and item embeddings nightly (batch), assembles and ranks at request time (real-time).

---

## 3. Two-Stage Retrieval + Ranking

### Why not rank everything directly

Running a full neural ranker against 10M items per query is infeasible: at 10ms per inference, that's 27 hours per request. Two-stage retrieval exists because heavy models can't brute-force large corpora.

```
All items (millions)
        │
        ▼
  [Stage 1: Retrieval]
  Recall-focused: return most of the relevant items
  Fast approximate similarity (ANN, BM25, collaborative filtering)
  → Top 100–1000 candidates (< 5ms)
        │
        ▼
  [Stage 2: Ranking]
  Precision-focused: rank the candidates well
  Heavy model with full feature set (DCN, DNN, cross-features)
  → Top 10–20 results (< 50ms)
        │
        ▼
  [Stage 3: Re-ranking / Business Rules]
  Diversity, freshness, sponsored slots, legal constraints
```

**Stage 1 options and their failure modes:**

| Method | Type | Best for | Failure mode |
| :--- | :--- | :--- | :--- |
| BM25 | Lexical | Keyword-heavy search | Misses semantic matches ("car" ≠ "automobile") |
| Two-tower + FAISS | Semantic | Content recommendation | Expensive index rebuilds; cold start |
| Matrix factorization | Collaborative | User-item interaction | No cold start handling; stale without refresh |
| Hybrid | Both | Production systems | More complexity to maintain |

**ANN index choice:** FAISS with IVF (Inverted File Index) + HNSW. IVF partitions the space into Voronoi cells; HNSW builds a hierarchical navigable small-world graph. Trade-off: recall vs latency vs memory.

---

## 4. Recommendation System Design

### Problem
Recommend items from a catalog of 10M products to 100M users in < 50ms.

### Why cold start handling is required

Matrix factorization needs at least one interaction to produce a latent vector. New items have none, so they never surface, never get interactions, and stay "new" forever — a feedback loop that kills recommendation quality in growing catalogs.

**Feature categories and their source:**

| Category | Examples | Freshness |
| :--- | :--- | :--- |
| User long-term | Purchase history, followed categories, demographic | Daily |
| User session | Last 5 clicked items, session duration, current cart | Real-time |
| Item static | Category, price, description embedding | On ingestion |
| Item dynamic | View count (1h/24h/7d), CTR, inventory status | Hourly/real-time |
| Contextual | Time of day, day of week, device, location | Real-time |
| Cross features | User × item: past interactions, affinity score | Batch + real-time |

**Model choices by maturity:**

- **Matrix Factorization (ALS):** $\hat{r}_{ui} = u_i^T v_j$ — fast retrieval, no cold start
- **Two-tower:** separate user and item encoders, dot product at serving — scalable, handles cold start via content features
- **DCN (Deep & Cross Network):** explicit feature crosses + deep network — strong ranking model
- **SASRec / BST:** transformer over user's interaction sequence — best for session-aware recommendations

**Cold start handling:**

| Scenario | Solution | Why |
| :--- | :--- | :--- |
| New user | Popularity-based; onboarding quiz | No history → fall back to prior |
| New item | Content-based features (description, image embeddings) | Item metadata available before interactions |
| New user + new item | Contextual bandits (UCB, Thompson sampling) | Explore to collect data while balancing exploitation |

**Feedback loop problem:** recommending popular items makes them more popular → more interaction data → more confidently recommended → even more popular. Self-reinforcing bias. Mitigate with:
- Exploration: $\epsilon$-greedy (random 5% of slots), UCB, Thompson sampling
- Inverse propensity scoring: weight training examples by the probability that they were shown
- Diversity constraints at re-ranking: cap same-category items

**Evaluation metrics:**

| Metric | Formula | What it measures |
| :--- | :--- | :--- |
| Precision@K | $|\text{relevant} \cap \text{top-K}| / K$ | Quality of top-K |
| Recall@K | $|\text{relevant} \cap \text{top-K}| / |\text{relevant}|$ | Coverage of relevant items |
| NDCG@K | $\sum_{i=1}^K \text{rel}_i / \log_2(i+1) / \text{IDCG}$ | Ranking quality with position discount |
| Hit Rate@K | $P(\text{at least one relevant item in top-K})$ | Simple, interpretable |

---

## 5. Fraud Detection System Design

### Why this architecture

Static rule engines get learned and gamed by fraudsters within days. Weekly retraining is too slow — models get gamed within 24 hours. Velocity features must be computed in real time or they're useless. The architecture below addresses each of these:

```
Transaction event
        │
        ├── Velocity feature computation (Redis: txns/unique merchants in last 1h/24h/7d)
        │   (Redis because these require sub-millisecond reads of rolling counters)
        │
        ├── Rule engine (hard blocks: known bad IPs, impossible geography, banned cards)
        │   (Rules first: model is slower, rules catch known-bad instantly)
        │
        ├── ML model scoring (gradient boosted tree or neural net, < 10ms)
        │         └── Output: fraud probability [0, 1]
        │
        └── Decision engine (cost-aware thresholds)
              ├── score < 0.3  → Auto-approve
              ├── 0.3 ≤ score < 0.7 → 3DS challenge (user authentication)
              └── score ≥ 0.7 → Auto-block
```

**Why gradient boosted tree, not deep learning, for scoring:**
- < 10ms latency required — GBT inference is microseconds; neural net is milliseconds
- Tabular velocity features: GBT handles mixed types natively, no embedding needed
- Interpretability: regulators require explainable decisions; SHAP on GBT is fast and reliable

**Threshold selection — the cost matrix approach:**
$$\text{Expected cost} = FP \times C_{FP} + FN \times C_{FN}$$

Where $C_{FP}$ = cost of blocking a legitimate transaction (customer churn, support call ≈ $5–50), $C_{FN}$ = cost of approving fraud (chargeback + penalty ≈ $100–500). Optimal threshold minimizes expected cost, not F1 or accuracy.

**The survivorship bias problem in retraining:** blocked transactions never get labels — you never know if they would have been fraudulent. Training only on approved transactions gives a biased sample. Fix: inject a small percentage of random approvals at threshold (with monitoring) to collect counterfactual labels. Weight these in retraining by inverse approval probability (propensity score).

**Key distinctions from standard classification:**
- **Adversarial:** fraudsters adapt. Retraining cadence matters more than model accuracy.
- **Imbalanced:** 0.1–1% positive rate. Default threshold 0.5 is wrong.
- **Temporal:** time-since-last-transaction and velocity features are the most predictive. Missing these = large accuracy loss.

---

## 6. Search Ranking Design

### Why pairwise/listwise losses exist

Pointwise models score each document independently and can't optimize for the final ranked list — e.g. scoring A=0.8, B=0.75 when B should rank first. Pairwise and listwise losses exist because ranking metrics (NDCG) can't be optimized directly with simple regression.

**Pipeline:**

```
Query → Query understanding → Retrieval (Stage 1) → Ranking (Stage 2) → Re-ranking
```

**Query understanding:**
- Spell correction → tokenization → intent classification (navigational / informational / transactional) → entity extraction

**Stage 2 ranking model choices:**

| Approach | Loss | What it captures |
| :--- | :--- | :--- |
| Pointwise | MSE/BCE on relevance label | Absolute relevance score |
| Pairwise | RankNet: $-\sigma(s_i - s_j)$ for preferred pairs | Relative ordering |
| Listwise | LambdaRank: gradient of NDCG | Full ranked list quality |

LambdaMART (LightGBM implementation) is the industry workhorse for search ranking:

```python
import lightgbm as lgb

dataset = lgb.Dataset(X_train, label=y_train, group=query_group_sizes)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "num_leaves": 64,
    "learning_rate": 0.1,
    "min_data_in_leaf": 50,
}
model = lgb.train(params, dataset, num_boost_round=500,
                  valid_sets=[val_dataset], callbacks=[lgb.early_stopping(50)])
```

**Personalization:** blend relevance and personalization scores:
$$s_{\text{final}} = \alpha \cdot s_{\text{relevance}} + (1-\alpha) \cdot s_{\text{user\_affinity}}$$

**Evaluation:**
- NDCG@K: position-discounted cumulative gain — standard metric
- MRR: $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank of first relevant doc}}$ — good for navigational queries
- A/B on CTR: the ground truth

---

## 7. LLM Serving Design

### Problem
Serve a 70B LLM at 100 QPS with P90 time-to-first-token < 2s.

### Why KV caching is essential

Without it, every generated token requires recomputing attention keys/values for the entire prefix — for a 1000-token prompt, that's 1000 recomputations per generated token. KV caching computes prefix keys/values once and reuses them, making LLM serving practical.

```
User request
        │
        ▼
Load balancer (least-loaded replica; sticky routing for KV cache locality)
        │
        ▼
Serving engine (vLLM / TGI)
  ├── Prefill: compute KV cache for the prompt (parallelizable, batched)
  ├── Decode: generate tokens one at a time (sequential, memory-bound)
  ├── PagedAttention: KV cache blocks allocated on demand (no fragmentation)
  ├── Continuous batching: new requests join the batch mid-generation
  └── Tensor parallel: model split across GPUs for large models
        │
        ▼
Response streaming (Server-Sent Events / WebSocket)
```

**Capacity planning:**
- 70B model in BF16: 140GB VRAM → minimum 2× H100 80GB with tensor parallelism
- KV cache per request: $2 \times L \times H_{kv} \times d_h \times T$ bytes per token
  - LLaMA 3 70B (80 layers, 8 GQA heads, 128 head dim): ~160MB per 1K tokens
- At 100 QPS and 500 token avg response: need KV cache capacity for ~50 concurrent active requests

**Latency anatomy:**

| Component | Typical range | Primary bottleneck | Optimization |
| :--- | :--- | :--- | :--- |
| TTFT (time-to-first-token) | 500ms–3s | Prefill compute | Prefill batching, quantization, speculative decoding |
| TPOT (time per output token) | 20–80ms | KV cache memory bandwidth | GQA (fewer KV heads), quantization, larger batch |
| Total latency (500 tokens) | 12–42s | TPOT dominates | Speculative decoding (2–3× speedup) |

**Quantization tradeoffs:**
- INT8: ~2% quality drop, 2× memory reduction, faster kernels
- INT4 (GPTQ/AWQ): ~5% quality drop, 4× memory reduction — enables 70B on 2× A100 40GB
- FP8: near-zero quality drop, 2× reduction, requires Hopper+ GPU

---

## 8. Monitoring Design

Every deployed ML system needs three layers of monitoring:

### Layer 1: Infrastructure health (standard SRE)
| Signal | Alert threshold |
| :--- | :--- |
| P99 latency | > SLA threshold |
| Error rate (5xx) | > 0.1% |
| Throughput | < expected QPS |

### Layer 2: Model health
| Signal | Metric | Alert threshold |
| :--- | :--- | :--- |
| Input distribution | PSI per feature | > 0.25 on critical features |
| Prediction distribution | Score histogram shift | > 2σ from 7-day baseline |
| Calibration | $P(y=1 | \text{score} = 0.7)$ | Deviation > 10% |

**PSI (Population Stability Index):**
$$\text{PSI} = \sum_i (A_i - E_i) \ln\frac{A_i}{E_i}$$

$A_i$ = actual proportion in bucket $i$, $E_i$ = expected (baseline). PSI < 0.1 = stable, 0.1–0.25 = some shift, > 0.25 = significant drift.

### Layer 3: Business KPI
| Metric | Alert condition |
| :--- | :--- |
| CTR, conversion rate | > 5% relative drop vs control |
| Revenue/query | Any significant drop |
| Abuse/safety metric | Any significant increase |

### Retraining triggers
- **Scheduled:** weekly regardless of drift (handles slow concept drift)
- **Triggered:** PSI > 0.25 on critical features, or business KPI drop > threshold
- **Continuous:** online learning for high-velocity signals (ads, fraud) — requires careful validation to avoid training on noise

### Rollback plan
Define before launch, not after:
```
1. Automatic: if [business metric] drops > X% in 24h → trigger alert
2. Shadow mode: new model runs but doesn't serve; compare to live model offline
3. Canary: 1% → 10% → 50% → 100% with metric checks at each stage
4. Rollback: one-command deployment of previous artifact; test quarterly that it works
```
