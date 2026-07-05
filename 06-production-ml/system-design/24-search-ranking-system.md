---
module: Production Ml
topic: System Design
subtopic: Search Ranking System
status: unread
tags: [productionml, ml, system-design-search-ranking-s]
---
# Search Ranking System Design

End-to-end search ranking system covering query understanding, retrieval, and learning-to-rank. Canonical system design for Google/Meta/Amazon interviews.

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

### Preprocessing Pipeline

```python
class QueryProcessor:
    def __init__(self):
        self.spell_corrector = SymSpell(max_dictionary_edit_distance=2)
        self.ner_model = load_ner_model()
        self.intent_classifier = load_intent_classifier()
    
    def process(self, raw_query: str) -> dict:
        # Spell correction
        corrected = self.spell_corrector.lookup_compound(raw_query, max_edit_distance=2)
        query = corrected[0].term if corrected else raw_query
        
        # Tokenize and normalize
        tokens = query.lower().split()
        
        # Entity extraction
        entities = self.ner_model(query)
        
        # Intent classification: informational / navigational / transactional
        intent = self.intent_classifier(query)
        
        # Query expansion
        synonyms = self.expand_query(tokens)
        
        return {
            "original": raw_query,
            "corrected": query,
            "tokens": tokens,
            "entities": entities,
            "intent": intent,
            "expanded_tokens": tokens + synonyms,
        }
    
    def expand_query(self, tokens: list[str]) -> list[str]:
        """Add synonyms and related terms."""
        expansions = []
        for token in tokens:
            word_vec_neighbors = self.word2vec.most_similar(token, topn=3)
            expansions.extend([w for w, score in word_vec_neighbors if score > 0.7])
        return expansions
```

---

## 3. Retrieval: BM25 + Dense

**Two-stage retrieval funnels billions of documents to hundreds of candidates.**

### BM25 (Lexical)

BM25 scores document relevance based on term frequency with saturation and length normalization:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d) \cdot (k_1 + 1)}{tf(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

where:
- $\text{IDF}(t) = \log\frac{N - df(t) + 0.5}{df(t) + 0.5}$ — inverse document frequency
- $k_1 \in [1.2, 2.0]$ — term frequency saturation (prevents very frequent terms from dominating)
- $b = 0.75$ — length normalization factor
- $avgdl$ — average document length

**BM25 fails on:**
- Vocabulary mismatch: query "cardiovascular disease" doesn't match document "heart attack"
- Paraphrase: "cheap flights" vs "affordable airfare"
- Semantic: "Apple earnings" might prefer tech documents over orchard documents depending on context

### Dense Retrieval (Two-Tower)

Both query and document are encoded into the same vector space; relevance = cosine similarity.

```python
class TwoTowerRetriever(nn.Module):
    def __init__(self, encoder_dim=768, projection_dim=128):
        super().__init__()
        # Shared backbone or separate encoders
        self.query_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.doc_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # Project to smaller space for efficient ANN
        self.query_proj = nn.Linear(encoder_dim, projection_dim)
        self.doc_proj = nn.Linear(encoder_dim, projection_dim)
    
    def encode_query(self, input_ids, attention_mask):
        output = self.query_encoder(input_ids, attention_mask).last_hidden_state
        cls = output[:, 0, :]  # [CLS] token
        return F.normalize(self.query_proj(cls), dim=-1)
    
    def encode_doc(self, input_ids, attention_mask):
        output = self.doc_encoder(input_ids, attention_mask).last_hidden_state
        cls = output[:, 0, :]
        return F.normalize(self.doc_proj(cls), dim=-1)

# In-batch negative training
def biencoder_loss(query_embeddings, doc_embeddings, temperature=0.05):
    """
    In-batch negatives: for batch of N (query, doc) pairs,
    each query's negatives are the N-1 other documents in the batch.
    """
    # Similarity matrix: [N, N]
    sim = torch.mm(query_embeddings, doc_embeddings.T) / temperature
    # Diagonal = positive pairs, off-diagonal = negatives
    labels = torch.arange(len(query_embeddings), device=sim.device)
    return F.cross_entropy(sim, labels)
```

**ANN Indexing with FAISS:**
```python
import faiss

d = 128  # embedding dimension
index = faiss.IndexFlatIP(d)  # Inner product (equivalent to cosine for normalized vectors)

# For billion-scale: use HNSW or IVF
index = faiss.IndexHNSWFlat(d, 32)   # HNSW: ~10ms for 100M docs
# or
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1024)  # IVF: faster but less accurate
index.train(doc_embeddings)

# Index all documents offline
index.add(doc_embeddings)

# Query
k = 1000
distances, indices = index.search(query_embedding, k)
```

### Hybrid Retrieval (BM25 + Dense Fusion)

Combine BM25 and dense scores via Reciprocal Rank Fusion (RRF):

$$\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$$

where k=60 is a constant that smooths the contribution of high-rank results.

```python
def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    """Combine ranked lists using RRF."""
    scores = defaultdict(float)
    
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

## 4. Learning to Rank (LTR)

LTR trains a model to rank candidates optimally, using click/engagement data as supervision.

### Three LTR Paradigms

| Paradigm | Training signal | Loss | Example models |
|---|---|---|---|
| Pointwise | Relevance label per doc | MSE / BCE | Linear regression |
| Pairwise | Which doc is better? | RankNet loss | RankBoost, LambdaRank |
| Listwise | Optimize ranking metric (NDCG) | LambdaLoss | LambdaMART |

### nDCG: The Key Ranking Metric

$$\text{DCG}_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{nDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}$$

where IDCG is the DCG of the ideal (perfect) ranking.

- Position 1 weight: $1/\log_2(2) = 1.0$
- Position 2 weight: $1/\log_2(3) = 0.63$
- Position 10 weight: $1/\log_2(11) = 0.29$

Lower positions contribute much less — the top 3 positions dominate ranking quality.

### LambdaMART

LambdaMART optimizes nDCG by computing "lambda" gradients — virtual gradients that represent how much swapping two documents would change nDCG.

$$\lambda_{ij} = \frac{\partial C}{\partial s_i} = -\frac{|\Delta \text{nDCG}_{ij}|}{1 + e^{s_i - s_j}}$$

where $|\Delta \text{nDCG}_{ij}|$ is the nDCG change from swapping documents i and j.

```python
import lightgbm as lgb

# LambdaMART with LightGBM
train_data = lgb.Dataset(
    X_train, 
    label=y_train,           # relevance labels (0-3 scale)
    group=query_groups_train  # number of docs per query for ranking loss
)
val_data = lgb.Dataset(
    X_val,
    label=y_val,
    group=query_groups_val
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "lambdarank_truncation_level": 10,  # optimize top-10 nDCG
    "num_leaves": 255,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)
```

### LTR Features

**Query features:** query length, intent, entity type, language, historical CTR for similar queries

**Document features:** BM25 score, PageRank, freshness, domain authority, content quality score

**Query-document features:** BM25 (multiple fields), semantic similarity, embedding cosine, exact match fields, TF-IDF, query coverage of document

**User-document features (personalization):** historical clicks on this domain, user preference for content type, geographic match

---

## 5. Dealing with Feedback Loops and Position Bias

**Position bias:** users click position 1 more than position 2, regardless of relevance. Training on clicks without correction biases the model toward already-ranked-high documents.

**Inverse Propensity Scoring (IPS):**
$$\text{unbiased click} = \frac{\text{click}}{P(\text{click} | \text{position})}$$

Estimate propensity by randomized experiments (swap test) or counterfactual estimation.

```python
def ips_correction(clicks, positions, propensity_scores):
    """Inverse Propensity Scoring for unbiased relevance estimation."""
    # propensity_scores[pos] = P(click | position) for relevant doc
    weights = 1.0 / propensity_scores[positions]
    unbiased_relevance = clicks * weights
    return unbiased_relevance
```

---

## 6. Serving Architecture and Latency Budget

**200ms end-to-end budget breakdown:**

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

**Q: Walk me through why nDCG is better than precision@k for search evaluation.**  
A: Precision@k treats all positions equally — a correct result at position 1 counts the same as one at position 10. nDCG uses a logarithmic position discount: position 1 has weight 1.0, position 2 has weight 0.63, position 10 has weight 0.29. This matches user behavior (eye-tracking studies show users rarely scroll past position 3). nDCG also incorporates graded relevance — a document rated highly relevant (3/3) contributes more than a marginally relevant one (1/3), whereas precision@k treats binary relevance. For comparing systems, nDCG is more sensitive to improvements in the top ranks where most user attention is, making it a better proxy for business metrics like CTR and engagement.

**Q: How do you handle vocabulary mismatch between queries and documents?**  
A: Multiple complementary approaches: (1) Dense retrieval (bi-encoder) embeds queries and documents in the same semantic space — "heart attack" and "myocardial infarction" get similar embeddings; (2) Query expansion — automatically add synonyms and related terms to the query before BM25 matching; (3) Hybrid retrieval — combine BM25 (handles exact matches and rare terms well) with dense retrieval (handles paraphrase and semantic equivalence) via RRF; the combination consistently outperforms either alone. (4) SPLADE or learned sparse retrieval — learn to expand document and query representations in the vocabulary space, combining BM25's efficiency with dense retrieval's semantic understanding. In practice, hybrid retrieval with RRF is the most robust baseline.

**Q: How do you train a learning-to-rank model when you only have click logs?**  
A: Clicks are a noisy proxy for relevance due to position bias, selection bias, and temporal effects. Pipeline: (1) Collect impression logs with position, document, query, and whether it was clicked; (2) Apply position bias correction — estimate propensity scores by randomization (A/B test where you randomly re-rank results to break position/click correlation) or EM-based estimation; (3) Create pairwise training examples: (query, clicked_doc, skipped_doc) pairs where the skipped_doc was shown above the clicked_doc (skipped docs at lower positions are stronger negatives); (4) Train LambdaMART on these pairwise preferences with propensity-weighted loss; (5) Supplement with human relevance judgments (editorial labels) for tail queries where clicks are sparse; (6) Evaluate on held-out queries with editorial labels to avoid circular evaluation.

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
