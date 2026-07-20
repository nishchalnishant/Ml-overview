---
module: System Design
topic: System Design
subtopic: End To End Recommendation System
status: unread
tags: [productionml, ml, system-design-end-to-end-recom]
---
# End-to-End Recommendation System Design

**Scale:** 100M users, 10M items, <100ms P99 latency, 1M QPS peak

---

## 1. Problem Framing

### Clarifying Questions (ask first in any interview)
- **Explicit vs implicit feedback?** (ratings vs clicks/watch-time)
- **Real-time vs batch?** (feed refresh on scroll vs daily email)
- **Cold start exposure?** (new user, new item SLA)
- **Business objective?** (engagement, revenue, diversity, safety)
- **Feedback loop risk?** (filter bubbles, popularity bias)

### Metric Hierarchy
| Layer | Metric | Formula |
|---|---|---|
| Retrieval | Recall@K | \|relevant ∩ top-K\| / \|relevant\| |
| Ranking | nDCG@K | DCG@K / IDCG@K where DCG = Σ (2^rel_i − 1)/log₂(i+1) |
| Business | CTR, Watch-time, Revenue | online A/B |
| Long-term | Retention D7/D30 | survival analysis |

---

## 2. System Architecture Overview

```
User Request
     │
     ▼
┌─────────────┐    ┌──────────────────────────────────────────────┐
│  API Layer  │───▶│          Recommendation Service              │
│  (50ms SLA) │    │                                              │
└─────────────┘    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                   │  │Retrieval │─▶│  Ranking │─▶│  Re-rank │  │
                   │  │(ANN, 1ms)│  │(ML, 30ms)│  │(rules/ML)│  │
                   │  └──────────┘  └──────────┘  └──────────┘  │
                   └──────────────────────────────────────────────┘
                          │                │
                    ┌─────┘           ┌────┘
                    ▼                 ▼
             Vector DB          Feature Store
           (FAISS/Milvus)      (Redis + offline)
```

**Two-stage pipeline** is universal at scale: retrieve hundreds cheaply, rank with expensive model, re-rank for business constraints.

---

## 3. Stage 1: Candidate Retrieval

### Two-Tower Model

**Architecture:**
```
User features ──► User Tower ──► u ∈ ℝᵈ ─┐
                                           ├──► score = u · v
Item features ──► Item Tower ──► v ∈ ℝᵈ ─┘
```

**Retrieval score:**
$$s(u, i) = \frac{\mathbf{u}^T \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$$

**Training loss (in-batch softmax / sampled softmax):**
$$\mathcal{L} = -\log \frac{\exp(s(u, i^+)/\tau)}{\exp(s(u, i^+)/\tau) + \sum_{j \in \mathcal{N}} \exp(s(u, i_j^-)/\tau)}$$

where τ is temperature (typically 0.05–0.1), N is batch negatives. **Correction for popularity bias in negatives:**
$$\tilde{s}(u, i) = s(u, i) - \log p(i)$$

where p(i) is item sampling probability (frequency correction).

```python
class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim=256):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
    
    def forward(self, user_feats, item_feats):
        u = F.normalize(self.user_tower(user_feats), dim=-1)
        v = F.normalize(self.item_tower(item_feats), dim=-1)
        return u, v
    
    def in_batch_loss(self, u, v, tau=0.07):
        # u, v: [B, D] — diagonal = positives
        logits = torch.matmul(u, v.T) / tau  # [B, B]
        labels = torch.arange(len(u), device=u.device)
        return F.cross_entropy(logits, labels)
```

### ANN Index
- **Offline:** embed all 10M items → build FAISS `IndexIVFPQ` (IVF: 4096 cells, PQ: M=64 subspaces, 8 bits)
- **Query time:** probe 128 cells → ~1M distance computations, ~2ms on CPU
- **Alternatives:** ScaNN (Google), Milvus (distributed), Pinecone (managed)

**Index refresh:** nightly full rebuild OR streaming upserts via `IndexIVFFlat` shadow index.

| Index type | Build | QPS | Recall@100 | Memory |
|---|---|---|---|---|
| Flat (exact) | O(N) | 5K | 100% | 10M × 256 × 4B = 10GB |
| IVF-PQ | O(N log N) | 50K | ~95% | ~500MB |
| HNSW | O(N log N) | 200K | ~98% | ~2GB |

---

## 4. Stage 2: Ranking Model

### Features
| Category | Examples |
|---|---|
| User | age, location, recent 50 interactions, session context |
| Item | category, age, popularity P7/P30, embeddings |
| Cross | user-item co-occurrence count, category affinity |
| Context | time-of-day, device, referrer |

### Model: Deep & Cross Network v2 (DCN-v2)

```
Input → [Embedding concat] → Cross Network + Deep Network → Concatenate → Output
```

**Cross layer:**
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (W_l \mathbf{x}_l + b_l) + \mathbf{x}_l$$

**BPR (Bayesian Personalized Ranking) loss** for implicit feedback:
$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$$

where i is positive (clicked), j is negative (not interacted).

**Binary cross-entropy** for click prediction:
$$\mathcal{L} = -\frac{1}{N}\sum_{i} [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

**Calibration:** use Platt scaling or isotonic regression. Predicted CTR must match observed rate for correct business math.

```python
class DCNv2(nn.Module):
    def __init__(self, input_dim, cross_layers=3, deep_hidden=256):
        super().__init__()
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(cross_layers)
        ])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, deep_hidden), nn.ReLU(),
            nn.Linear(deep_hidden, deep_hidden), nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + deep_hidden, 1)
    
    def forward(self, x):
        x0 = x
        xl = x
        for W, b in zip(self.cross_weights, self.cross_biases):
            xl = x0 * (xl @ W + b) + xl  # cross layer
        deep_out = self.deep(x)
        return torch.sigmoid(self.output(torch.cat([xl, deep_out], dim=-1)))
```

---

## 5. Stage 3: Re-ranking

Apply **business constraints** after ML ranking:
- **Diversity:** Maximal Marginal Relevance (MMR): rerank greedily penalizing similarity to already-selected items
- **Freshness:** boost items < 24h old by factor λ
- **Safety:** hard-remove policy-violating items
- **Exploration:** ε-greedy or UCB slot (1 in 20 positions is exploration)

$$\text{MMR}(i) = \lambda \cdot \text{score}(i) - (1-\lambda) \cdot \max_{j \in S} \text{sim}(i, j)$$

---

## 6. Cold Start

### New User (zero history)
1. **Onboarding flow:** collect 3–5 explicit preferences
2. **Demographic proxy:** age/region/device → population prior
3. **Hot start:** serve global trending + category buckets until N≥20 interactions
4. **Cascade:** switch to personalized retrieval at interaction threshold

### New Item (zero interactions)
1. **Content-based embedding:** title + description → text encoder → item tower
2. **Category/tag injection:** route to relevant user segments
3. **Explore-then-exploit:** dedicated explore bucket (top-5% impressions guaranteed)
4. **Warm-up window:** track CTR for 48h before entering main ranker

| Strategy | Latency impact | Coverage | Notes |
|---|---|---|---|
| Popularity fallback | 0 | 100% | no personalization |
| Content embedding | +2ms | 100% | cold start default |
| Collaborative (N≥20) | +10ms | partial | personalized |
| Full two-tower | +20ms | established users | best quality |

---

## 7. Feature Store

```
Raw events (Kafka) ──► Flink job ──► Online store (Redis, <1ms)
                                         │
Batch features (Spark) ──────────────────┘ (merged at query time)
                   └──► Offline store (Parquet/Hive) ──► Training
```

**Point-in-time correctness:** join features using event timestamp, not current time, to avoid label leakage. All offline features must have a `feature_ts` column.

```sql
-- Point-in-time correct feature join
SELECT e.user_id, e.item_id, e.label,
       f.user_30d_clicks, f.user_ctr_category
FROM events e
JOIN user_features f
  ON e.user_id = f.user_id
  AND f.feature_ts = (
    SELECT MAX(feature_ts) FROM user_features
    WHERE user_id = e.user_id AND feature_ts <= e.event_ts
  )
```

---

## 8. Offline Evaluation

**Do not use random train/test split** — use temporal split.

| Metric | Formula | Use |
|---|---|---|
| Recall@K | above | retrieval quality |
| nDCG@K | above | ranking quality |
| MRR | 1/rank of first relevant | sparse feedback |
| AUC-ROC | AUROC | ranking discrimination |
| Calibration (ECE) | Σ\|acc(B) − conf(B)\| × \|B\|/N | CTR prediction |

**Offline → Online gap is real.** Always A/B test before claiming win.

---

## 9. A/B Testing & Experimentation

**Minimum detectable effect (MDE):**
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$$

For CTR = 2%, target δ = 0.1pp, σ² ≈ p(1-p) ≈ 0.02:
$$n \approx \frac{(1.96 + 0.84)^2 \times 2 \times 0.02}{(0.001)^2} \approx 313{,}600 \text{ users/arm}$$

**CUPED variance reduction:**
$$Y_{cuped} = Y - \theta (X - \bar{X}), \quad \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

Use pre-experiment metric X (e.g., last week's CTR) to reduce variance 20–40%.

**SRM detection** (Sample Ratio Mismatch):
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

Run before reading any metric. SRM p < 0.01 → invalidate experiment, diagnose assignment.

**Interleaving** for faster ranking comparison:
- Team-draft interleaving: rank A and B independently, interleave top-k alternating
- Outcome: which list's items got more clicks? 10× more sensitive than standard A/B for ranking

---

## 10. Monitoring & Production Health

### Key Metrics to Track
| Signal | Alert threshold | Action |
|---|---|---|
| Retrieval recall drop >5% | P90 over 1h | retrain / refresh index |
| Ranking AUC drop >1% | rolling 24h | check feature drift |
| Item coverage drop | >20% items unretrieved | cold start regression |
| Click-through rate | ±10% week-over-week | investigate freshness / ranker |
| P99 latency | >100ms | scale ANN replicas or cache |

### Feedback Loop Detection
```python
# Exposure bias: check if item rank correlates with training label rate
def check_position_bias(df):
    # df: {position, label}
    return df.groupby('position')['label'].mean()  # should flatten after debias

# Popularity bias: check if top-1% items dominate impressions
def check_popularity_concentration(df):
    item_counts = df['item_id'].value_counts(normalize=True)
    return item_counts.head(100).sum()  # should be < 0.5 for healthy diversity
```

### Training Pipeline
```
Daily batch:
  1. Pull last 7d interactions (Spark)
  2. Feature join (point-in-time correct)
  3. Negative sampling (100:1 ratio, popularity-corrected)
  4. Train two-tower (2 epochs, early stopping on val loss)
  5. Build ANN index → shadow deploy → shadow eval
  6. Gate: offline recall@100 ≥ current - 0.5% → promote
  7. Online: 10% canary → 50% → 100%
```

---

## Trade-offs Summary

| Decision | Option A | Option B | Choose when |
|---|---|---|---|
| Retrieval | Two-tower (fast, scalable) | GraphSAGE (better cold start) | A for scale, B for sparse graphs |
| Ranking features | Real-time (Redis, 1ms) | Batch (Hive, stale) | Real-time for session signals |
| Negative sampling | In-batch | Hard negatives | Hard negatives for better ranker |
| Index | FAISS IVF-PQ (approx) | ScaNN (exact+compress) | FAISS for open-source, ScaNN at Google |
| Re-ranking | Rule-based | Learned MMR | Rules for safety-critical, learned for diversity |
| Experimentation | A/B (slower) | Interleaving (faster) | A/B for final calls, interleaving for ranker iteration |

---

## 5+ Canonical Interview Q&As

**Q: Why two-tower instead of one model that takes both user and item as input?**  
A: Two-tower decouples user and item computation. Item embeddings are precomputed offline, so query-time cost is O(1) ANN lookup, not O(N) forward passes. The trade-off: you lose explicit feature interaction (user × item cross-features). DLRM/DCN-V2 adds interactions back at ranking stage, where candidate set is small enough (~500 items) to afford it.

**Q: How do you handle position bias in training data?**  
A: Items shown at position 1 have 3–5× higher CTR regardless of quality (position bias). Solutions: (1) Inverse propensity scoring — weight each sample by 1/P(position) to debias, (2) position-aware features — feed position as input and zero it out at inference, (3) pairwise training (BPR) which only uses relative ordering.

**Q: Your retrieval recall dropped from 82% to 74% overnight. Walk me through diagnosis.**  
A: Check (1) item index freshness — did the ANN rebuild fail? (2) feature distribution for user tower — new feature pipeline issue? (3) model version — was a bad model deployed? (4) data issue — popularity distribution shift in items? Start with index health, then feature drift (PSI), then model logs. See diagnosis flowchart in [../../13-production-ml/10-production-diagnosis-flowchart.md](../../13-production-ml/10-production-diagnosis-flowchart.md).

**Q: How would you design for 10× scale from 100M to 1B users?**  
A: (1) Shard user embedding store by user_id mod N, (2) distribute ANN index across shards (each shard searches subset, merge top-K), (3) split retrieval into geographic clusters, (4) serve user tower at edge for low-latency embedding, (5) async re-ranking to decouple from API latency. Key bottleneck shifts from model inference to embedding store I/O.

**Q: nDCG vs MRR — when do you use which?**  
A: MRR focuses on rank of first relevant item — good for navigational queries (user wants exactly one thing). nDCG measures graded relevance across all K positions — better for feed/recommendation where multiple relevant items exist. nDCG@10 is standard for recommendation; MRR is standard for search.

**Q: You have 10M items but 90% have <10 interactions. How do you prevent the long tail from hurting your model?**  
A: (1) Content features prevent zero-shot cold start — item tower uses text/image embeddings regardless of interaction history, (2) popularity-corrected negative sampling prevents head items from dominating gradient signal, (3) exposure floor — guarantee minimum impressions to tail items via explore bucket, (4) cluster-level fallback — items with sparse interactions fall back to cluster embeddings from similar items.

**Q: Should your retrieval model and ranking model share embeddings?**  
A: Generally no. Retrieval optimizes for recall (do I get good candidates?), ranking optimizes for precision (which candidate is best?). Different objectives → different representation optima. Exception: parameter-efficient scenarios where a shared backbone is fine-tuned with separate heads — acceptable when model size is constrained.

## Flashcards

**Why does two-tower retrieval normalize user and item embeddings before taking the dot product?** #flashcard
Normalizing to unit vectors turns the dot product into pure cosine similarity, so retrieval ranks items by direction (semantic alignment) rather than magnitude; without normalization, items with large-norm embeddings (e.g. popular items with more training signal) would score higher regardless of true relevance.

**Why does two-tower training correct in-batch negative scores by subtracting log p(i)?** #flashcard
In-batch sampled negatives are drawn proportional to item frequency in the batch, so popular items appear as negatives far more often than rare ones purely by chance; subtracting log p(i) (the item's sampling probability) removes this frequency bias so the model isn't penalized for scoring popular items highly just because they were oversampled as negatives.

**Why is a two-stage retrieval→ranking pipeline used instead of scoring all 10M items with the full ranking model?** #flashcard
The ranking model uses expensive cross-features and deep interactions that cost too much to run on 10M items within a 100ms budget; ANN retrieval narrows the candidate set to a few hundred in ~2ms so the expensive model only needs to score a small, pre-filtered set.

**Why does IVF-PQ trade exact recall for memory and speed compared to a flat index?** #flashcard
A flat index computes exact distances against all 10M vectors (10GB, 100% recall but only ~5K QPS); IVF-PQ instead clusters vectors into cells and compresses each vector into a compact product-quantized code, so search only probes a subset of cells against compressed codes — dropping to ~500MB and ~95% recall but reaching 50K QPS, which is the right trade for retrieval where ranking downstream can absorb small recall loss.

**Why is BPR (pairwise) loss often preferred over pointwise binary cross-entropy for implicit feedback ranking?** #flashcard
Implicit feedback (clicks, not ratings) only tells you a user preferred one item over ones they didn't click, not an absolute label like "0.3 relevant"; BPR directly optimizes the relative order of a clicked item versus an unclicked one, which matches the actual implicit signal, whereas pointwise BCE forces an artificial absolute click/no-click target that's noisier for ranking quality.

**Why must offline feature joins use point-in-time correctness (feature_ts ≤ event_ts) instead of joining to the current feature value?** #flashcard
If training features are joined using today's feature value for a historical event, the model sees information that wasn't actually available at prediction time (e.g. a user's click count that includes clicks that happened after the training example's timestamp) — this label leakage makes offline metrics look great while the model fails online because that future information doesn't exist at serving time.

**Why does the offline evaluation pipeline require a temporal train/test split instead of a random split?** #flashcard
A random split lets the model train on interactions that happened after some test examples, again leaking future information the model wouldn't have at real serving time; a temporal split (train on the past, test on the future) is the only split that mirrors how the model will actually be used in production.

**Why does interleaving detect ranking quality differences with far fewer users than a standard A/B test?** #flashcard
Standard A/B compares aggregate metrics between two disjoint populations, so the signal is diluted by between-user variance; interleaving shows both rankers' results to the same user in one merged list and measures which ranker's items got clicked more, using each user as their own control — removing between-user variance is what gives it roughly 10x the sensitivity per user.

**Why is CUPED variance reduction applied using a pre-experiment metric rather than just increasing sample size?** #flashcard
Increasing sample size to hit a target MDE can require impractically large populations (300K+ users/arm for a 0.1pp CTR lift); CUPED instead subtracts out the portion of the outcome metric predictable from a user's pre-experiment behavior (e.g. last week's CTR), shrinking residual variance by 20-40% and letting the same MDE be detected with a smaller sample.

**Why must Sample Ratio Mismatch be checked before reading any other experiment metric?** #flashcard
If the actual traffic split between arms deviates from the intended split (e.g. 48/52 instead of 50/50), it signals a broken randomization or logging pipeline — any metric difference measured under a broken assignment mechanism is confounded and cannot be trusted, so SRM must be cleared first or the whole experiment's conclusions are invalid.
