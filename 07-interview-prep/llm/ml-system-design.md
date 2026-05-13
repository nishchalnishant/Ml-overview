# ML System Design — Interview Playbook

---

## 1. Universal Framework (Use for Every Problem)

Never start with "I'll use a neural network." Start with requirements:

```
1. Product goal          → What user behavior is being optimized?
2. Success metric        → Online (CTR, revenue, latency) and offline (AUC, NDCG)
3. Constraints           → Latency SLA, QPS, cost, regulatory
4. Data                  → Labels, features, freshness, volume
5. Baseline              → Simplest thing that could work (heuristic or LR)
6. Model architecture    → Justified by constraints, not by trend
7. Training pipeline     → Feature engineering, training cadence, retraining triggers
8. Serving               → Batch vs real-time, hardware, caching
9. Evaluation            → A/B test design, holdout, backtesting
10. Monitoring           → Drift detection, alerting, rollback
```

---

## 2. Batch vs Real-Time Serving

| Dimension | Batch | Real-Time |
| :--- | :--- | :--- |
| **Latency** | Minutes to hours | < 100ms |
| **Throughput** | Very high | Moderate |
| **Model size** | No constraint | Usually < 1GB |
| **Freshness** | Stale (hourly/daily) | Current |
| **Infrastructure** | Spark, Airflow, BigQuery | REST/gRPC server, Redis |
| **Use cases** | Email campaigns, weekly reports | Fraud detection, ad ranking |

**When to use batch:** when decision freshness isn't critical and you need to score millions of items (recommendation pre-computation, churn prediction sent via email).

**When to use real-time:** when the decision depends on the current context (fraud at transaction time, autocomplete as user types).

---

## 3. Two-Stage Retrieval + Ranking

Used in search, recommendations, and ads because you can't run a heavy model on the entire item corpus at query time.

```
All items (millions)
        │
        ▼
  [Stage 1: Retrieval]
  Fast approximate similarity search
  (ANN with FAISS, BM25, collaborative filtering)
  → Return top 100–1000 candidates
        │
        ▼
  [Stage 2: Ranking]
  Heavy model with full features
  (DCN, DIN, two-tower + cross features)
  → Return top 10–20 results
        │
        ▼
  [Stage 3: Re-ranking / Business Logic]
  Diversity, freshness constraints, sponsorship
```

**Stage 1 options:**

| Method | Type | Best for |
| :--- | :--- | :--- |
| BM25 | Lexical | Text search, keyword matching |
| Two-tower embedding + FAISS | Semantic | Content-based recommendation |
| Matrix factorization | Collaborative | User-item interaction patterns |
| Hybrid | Both | Most production systems |

---

## 4. Recommendation System Design

**Problem:** Recommend items from a catalog of 10M products to 100M users in < 50ms.

**Feature categories:**

| Category | Examples |
| :--- | :--- |
| User features | Age, location, device, purchase history, session behavior |
| Item features | Category, price, freshness, embeddings from description |
| Context features | Time of day, day of week, current session |
| Interaction features | User × item: previous views, dwell time, rating |

**Model choices:**

- **Matrix Factorization (ALS):** $\hat{r}_{ui} = u_i^T v_j$ — fast, no cold start handling
- **Two-tower neural:** separate encoders for user and item, dot product at serving — scales to large catalogs
- **DCN (Deep & Cross Network):** explicit feature crosses + deep network — strong ranking model
- **Sequential models (GRU4Rec, SASRec):** model session context — better freshness

**Cold start handling:**

| Scenario | Solution |
| :--- | :--- |
| New user | Popularity-based recommendations; onboarding flow |
| New item | Content-based features (description, category); metadata embeddings |
| New user + new item | Cross-domain transfer, contextual bandits |

**Evaluation:**

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| Precision@K | $\frac{\|\text{relevant} \cap \text{top-}K\|}{K}$ | Are top-K results good? |
| Recall@K | $\frac{\|\text{relevant} \cap \text{top-}K\|}{\|\text{relevant}\|}$ | Are all relevant items found? |
| NDCG@K | $\sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i+1)} / \text{IDCG}$ | Weighted by position |
| Hit Rate@K | $P(\text{relevant item in top-}K)$ | Simple, popular metric |

**Feedback loops:** recommending popular items makes them more popular. Mitigate with: exploration (epsilon-greedy or UCB), diversity constraints, inverse propensity scoring in training.

---

## 5. Fraud Detection Design

**Problem:** score transactions for fraud in < 200ms with $10^5$ QPS.

**Key distinctions from standard classification:**

- **Adversarial:** fraudsters adapt to the model — retraining cadence matters
- **Imbalanced:** 0.1–1% positive rate — threshold tuning and cost-sensitive metrics critical
- **Temporal:** time-based features matter (velocity, time since last transaction)
- **Business cost:** false positive costs customer trust; false negative costs money — need threshold per cost

**Feature types:**

| Type | Examples |
| :--- | :--- |
| Transaction | Amount, merchant category, location |
| Velocity | Txns in last 1h/24h/7d, unique merchants in 1h |
| User history | Avg amount, typical merchants, account age |
| Device | IP reputation, device fingerprint, new device flag |

**Architecture:**

```
Real-time transaction
        │
        ├── Feature extraction (Redis for velocity features)
        │
        ├── Rule engine (hard blocks: known bad IPs, impossible geography)
        │
        ├── ML model scoring (gradient boosted tree or neural net)
        │         └── Output: fraud probability
        │
        └── Decision engine
              ├── Score < 0.3 → Auto-approve
              ├── 0.3 ≤ Score < 0.7 → 3DS challenge
              └── Score ≥ 0.7 → Auto-block
```

**Threshold selection:** use precision-recall curve with business cost:

$$\text{Expected cost} = FP \times \text{cost}_{FP} + FN \times \text{cost}_{FN}$$

Set threshold to minimize expected cost.

---

## 6. Search Ranking

**Problem:** rank 1000 retrieved documents for a query in < 50ms.

**Query understanding:** spelling correction → tokenization → intent classification (navigational/informational/transactional) → entity extraction.

**Ranking model choices:**

| Approach | Description | Loss |
| :--- | :--- | :--- |
| **Pointwise** | Score each doc independently | MSE or BCE on relevance label |
| **Pairwise** | Compare pairs of docs | RankNet loss: $-\sigma(s_i - s_j)$ for preferred pair |
| **Listwise** | Score entire ranked list | LambdaRank, LambdaMART |

**LambdaMART (LightGBM to rank):**
```python
import lightgbm as lgb

dataset = lgb.Dataset(X_train, y_train, group=query_group_sizes)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "num_leaves": 64,
    "learning_rate": 0.1,
}
model = lgb.train(params, dataset, num_boost_round=500)
```

**Personalization:** blend general relevance score with user affinity:
$$s_{\text{final}} = \alpha \cdot s_{\text{relevance}} + (1-\alpha) \cdot s_{\text{personalization}}$$

**Evaluation:**
- **NDCG@K:** standard ranking metric, position-discounted
- **MRR (Mean Reciprocal Rank):** $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank of first relevant result}}$
- **CTR on A/B test:** ground truth is whether users click

---

## 7. LLM System Design

**Problem:** serve a 70B LLM at 100 QPS with P90 latency < 2s.

**Key components:**

```
User request
        │
        ▼
Load balancer (route to least-loaded replica)
        │
        ▼
Serving engine (vLLM / TGI)
  ├── KV cache (PagedAttention — dynamic allocation)
  ├── Continuous batching (no fixed batch size)
  ├── Speculative decoding (optional, 2–3× speedup)
  └── Tensor parallel across GPUs
        │
        ▼
Response streaming (SSE / WebSocket)
```

**Capacity planning:**
- 70B model in BF16 = 140GB VRAM
- 2× H100 80GB or 4× A100 80GB with tensor parallelism
- KV cache memory per request: $2 \times L \times H \times d_h \times T$ bytes
  - LLaMA 3 70B (80 layers, 8 KV heads, 128 head dim): ≈ 160MB per 1k tokens

**Latency components:**

| Component | Typical | Optimization |
| :--- | :--- | :--- |
| TTFT (time-to-first-token) | 500ms–2s | Prefill batching, quantization |
| TPOT (time per output token) | 20–50ms | KV cache, speculative decoding |
| Throughput | 100–500 tok/s | Continuous batching, larger batch |

---

## 8. Monitoring Checklist

Every deployed ML system needs:

| Signal | What to monitor | Alert threshold |
| :--- | :--- | :--- |
| **Data drift** | KL divergence on input features | > 0.1 on critical features |
| **Prediction drift** | Score distribution shift | > 2σ from baseline |
| **Label quality** | Ground truth arrival delay | If delayed, recalibrate |
| **Latency** | P50, P90, P99 | P99 > SLA threshold |
| **Error rate** | 5xx, timeouts | > 0.1% |
| **Business metric** | CTR, revenue, conversion | > 5% drop triggers incident |

**Drift detection with PSI (Population Stability Index):**
$$\text{PSI} = \sum_i (A_i - E_i) \ln\frac{A_i}{E_i}$$

Where $A_i$ = actual proportion in bucket $i$, $E_i$ = expected (baseline). PSI < 0.1 = stable, 0.1–0.25 = some shift, > 0.25 = significant drift.

**Retraining triggers:**
- Scheduled: weekly/monthly regardless of drift
- Triggered: when PSI > threshold or business metric drops
- Continuous: online learning for high-velocity signals (ads, fraud)

> [!TIP]
> **Interview structure:** ML system design = requirements first (don't assume a model is needed) → two-stage retrieval+ranking for scale → data and feature pipeline → serving constraints (batch vs real-time) → evaluation (offline metrics + online A/B) → monitoring (drift + latency + business KPI). Mentioning cold start, feedback loops, and rollback path signals production experience.
