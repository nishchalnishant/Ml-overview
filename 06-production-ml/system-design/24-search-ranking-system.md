---
module: Production Ml
topic: System Design
subtopic: Search Ranking System
status: unread
tags: [productionml, ml, system-design-search-ranking-s]
---
# Search Ranking System Design

End-to-end search ranking: query understanding, retrieval, learning-to-rank. Canonical for Google/Meta/Amazon interviews.

**Scale:** 1M QPS, 10B documents, <200ms P99 latency.

---

## 1. System Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  Query Understanding│ NLP: spell correction, query expansion,
│                  │   intent classification, entity extraction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Retrieval       │ BM25 + Dense (two-tower ANN)
│  (Recall Stage)  │ Target: top-1000 candidates, <50ms
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Ranking         │ LambdaMART / LightGBM
│  (Precision Stage│ Target: top-10 results, <50ms
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Re-ranking &    │ Diversity (MMR), personalization,
│  Post-processing │ freshness boost, policy filters
└────────┬─────────┘
         │
         ▼
       Results
```

---

## 2. Query Understanding

```python
class QueryProcessor:
    def __init__(self):
        self.spell_corrector = SymSpell(max_dictionary_edit_distance=2)
        self.ner_model = load_ner_model()
        self.intent_classifier = load_intent_classifier()

    def process(self, raw_query: str) -> dict:
        corrected = self.spell_corrector.lookup_compound(raw_query, max_edit_distance=2)
        query = corrected[0].term if corrected else raw_query
        tokens = query.lower().split()
        entities = self.ner_model(query)
        intent = self.intent_classifier(query)  # informational / navigational / transactional
        synonyms = self.expand_query(tokens)
        return {
            "original": raw_query, "corrected": query, "tokens": tokens,
            "entities": entities, "intent": intent,
            "expanded_tokens": tokens + synonyms,
        }

    def expand_query(self, tokens: list[str]) -> list[str]:
        """Add synonyms/related terms via word2vec neighbors."""
        expansions = []
        for token in tokens:
            neighbors = self.word2vec.most_similar(token, topn=3)
            expansions.extend([w for w, score in neighbors if score > 0.7])
        return expansions
```

---

## 3. Retrieval: BM25 + Dense

Two-stage retrieval funnels billions of documents to hundreds of candidates.

### BM25 (Lexical)

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d) \cdot (k_1 + 1)}{tf(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

- $\text{IDF}(t) = \log\frac{N - df(t) + 0.5}{df(t) + 0.5}$
- $k_1 \in [1.2, 2.0]$ — term frequency saturation
- $b = 0.75$ — length normalization
- $avgdl$ — average document length

**BM25 fails on:** vocabulary mismatch ("cardiovascular disease" vs "heart attack"), paraphrase ("cheap flights" vs "affordable airfare"), and semantic ambiguity ("Apple earnings" — tech vs orchard).

### Dense Retrieval (Two-Tower)

Query and document encode into the same vector space; relevance = cosine similarity.

```python
class TwoTowerRetriever(nn.Module):
    def __init__(self, encoder_dim=768, projection_dim=128):
        super().__init__()
        self.query_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.doc_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.query_proj = nn.Linear(encoder_dim, projection_dim)
        self.doc_proj = nn.Linear(encoder_dim, projection_dim)

    def encode_query(self, input_ids, attention_mask):
        cls = self.query_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return F.normalize(self.query_proj(cls), dim=-1)

    def encode_doc(self, input_ids, attention_mask):
        cls = self.doc_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return F.normalize(self.doc_proj(cls), dim=-1)

# In-batch negatives: for N (query, doc) pairs, each query's negatives
# are the N-1 other documents in the batch.
def biencoder_loss(query_embeddings, doc_embeddings, temperature=0.05):
    sim = torch.mm(query_embeddings, doc_embeddings.T) / temperature
    labels = torch.arange(len(query_embeddings), device=sim.device)
    return F.cross_entropy(sim, labels)
```

**ANN indexing with FAISS:**
```python
import faiss

d = 128
index = faiss.IndexFlatIP(d)  # exact inner product (cosine for normalized vectors)

# Billion-scale: HNSW (~10ms/100M docs) or IVF (faster, less accurate)
index = faiss.IndexHNSWFlat(d, 32)
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1024)
index.train(doc_embeddings)
index.add(doc_embeddings)

k = 1000
distances, indices = index.search(query_embedding, k)
```

### Hybrid Retrieval (BM25 + Dense Fusion)

Combine via Reciprocal Rank Fusion (RRF):

$$\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$$

k=60 smooths the contribution of high-rank results.

```python
def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    scores = defaultdict(float)
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

## 4. Learning to Rank (LTR)

LTR trains a model to rank candidates using click/engagement data as supervision.

| Paradigm | Training signal | Loss | Example models |
|---|---|---|---|
| Pointwise | Relevance label per doc | MSE / BCE | Linear regression |
| Pairwise | Which doc is better? | RankNet loss | RankBoost, LambdaRank |
| Listwise | Optimize ranking metric (NDCG) | LambdaLoss | LambdaMART |

### nDCG

$$\text{DCG}_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}, \quad \text{nDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}$$

IDCG is the DCG of the ideal ranking. Position 1 weight = 1.0, position 2 = 0.63, position 10 = 0.29 — top 3 positions dominate ranking quality.

### LambdaMART

Optimizes nDCG via "lambda" gradients: virtual gradients representing how much swapping two documents would change nDCG.

$$\lambda_{ij} = \frac{\partial C}{\partial s_i} = -\frac{|\Delta \text{nDCG}_{ij}|}{1 + e^{s_i - s_j}}$$

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train, group=query_groups_train)
val_data = lgb.Dataset(X_val, label=y_val, group=query_groups_val)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "lambdarank_truncation_level": 10,
    "num_leaves": 255,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
}

model = lgb.train(params, train_data, valid_sets=[val_data],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
```

### LTR Features

- **Query:** length, intent, entity type, language, historical CTR for similar queries
- **Document:** BM25 score, PageRank, freshness, domain authority, content quality
- **Query-document:** BM25 (multi-field), semantic similarity, embedding cosine, exact match, TF-IDF, query coverage
- **User-document (personalization):** historical clicks on domain, content-type preference, geo match

---

## 5. Feedback Loops and Position Bias

**Position bias:** users click position 1 more regardless of relevance. Training on raw clicks biases the model toward already-high-ranked documents.

**Inverse Propensity Scoring (IPS):**
$$\text{unbiased click} = \frac{\text{click}}{P(\text{click} | \text{position})}$$

Estimate propensity via randomized experiments (swap test) or counterfactual estimation.

```python
def ips_correction(clicks, positions, propensity_scores):
    weights = 1.0 / propensity_scores[positions]
    return clicks * weights
```

---

## 6. Serving Architecture and Latency Budget

| Stage | Target | Method |
|---|---|---|
| Query processing | 5ms | In-memory, fast NLP |
| BM25 retrieval | 20ms | Inverted index (Elasticsearch) |
| Dense ANN retrieval | 30ms | FAISS HNSW on GPU |
| Hybrid fusion | 5ms | RRF in memory |
| LTR scoring (top-1000) | 50ms | LightGBM, pre-loaded |
| Re-ranking / filters | 10ms | In-memory |
| Total | ~120ms | 80ms headroom for variance |

---

## Canonical Interview Q&As

**Q: Why is nDCG better than precision@k for search evaluation?**
A: Precision@k treats all positions equally — a hit at position 1 counts the same as at position 10. nDCG discounts logarithmically by position (weight 1.0, 0.63, ... 0.29 at position 10), matching the fact that users rarely scroll past position 3. It also uses graded relevance (a 3/3 doc counts more than a 1/3 doc), unlike binary precision@k. This makes nDCG more sensitive to top-rank improvements — a better proxy for CTR and engagement.

**Q: How do you handle vocabulary mismatch between queries and documents?**
A: (1) Dense retrieval embeds queries and documents in the same semantic space, so "heart attack" and "myocardial infarction" land close together; (2) query expansion adds synonyms before BM25 matching; (3) hybrid retrieval combines BM25 (exact/rare-term matches) with dense retrieval (paraphrase/semantic matches) via RRF — this combination consistently beats either alone; (4) learned sparse retrieval (e.g. SPLADE) expands query/doc representations in vocabulary space, combining BM25 efficiency with dense semantics. Hybrid + RRF is the most robust baseline in practice.

**Q: How do you train an LTR model with only click logs?**
A: Clicks are noisy relevance signals due to position/selection bias. Pipeline: (1) collect impression logs with position, doc, query, click; (2) correct position bias via propensity scores (randomized re-ranking A/B test or EM estimation); (3) build pairwise examples — (query, clicked_doc, skipped_doc) where skipped_doc was shown above the click (stronger negative signal); (4) train LambdaMART on propensity-weighted pairwise preferences; (5) supplement with editorial relevance labels for tail queries where clicks are sparse; (6) evaluate on held-out queries with editorial labels to avoid circular evaluation.

## Flashcards

**$\text{IDF}(t) = \log\frac{N - df(t) + 0.5}{df(t) + 0.5}$?** #flashcard
inverse document frequency

**$k_1 \in [1.2, 2.0]$?** #flashcard
term frequency saturation (prevents very frequent terms from dominating)

**$b = 0.75$?** #flashcard
length normalization factor

**$avgdl$?** #flashcard
average document length

**Vocabulary mismatch?** #flashcard
query "cardiovascular disease" doesn't match document "heart attack"

**Paraphrase?** #flashcard
"cheap flights" vs "affordable airfare"

**Semantic?** #flashcard
"Apple earnings" might prefer tech documents over orchard documents depending on context

**Position 1 weight?** #flashcard
$1/\log_2(2) = 1.0$

**Position 2 weight?** #flashcard
$1/\log_2(3) = 0.63$

**Position 10 weight?** #flashcard
$1/\log_2(11) = 0.29$
