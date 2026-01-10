# ML System Design: Comprehensive Case Studies

This file provides end-to-end walkthroughs for the most common ML System Design interview questions.

---

## 1. Case Study: Recommendation System (e.g., Netflix/YouTube)

### 🗺️ High-Level Architecture
1. **Candidate Generation (Retrieval):** Million items $\rightarrow$ ~1000 items. (High recall).
2. **Ranking (Scoring):** 1000 items $\rightarrow$ Top 20 items. (High precision).
3. **Re-ranking/Calibration:** Diversity, fresh items, removing duplicates.

### 📐 Technical Deep-Dive
- **Embeddings:** Two-Tower Network (User tower $U(x)$, Item tower $V(y)$). The retrieval is a Dot Product $U(x) \cdot V(y)$.
- **ANN Search:** Use **Faiss** or **HNSW** for sub-millisecond similarity search.
- **Handling Feature Crosses:** Use **DCN v2** or **DeepFM** in the Ranking stage to learn non-linear interactions (e.g., `UserAge` x `MovieGenre`).

---

## 2. Case Study: Fraud Detection (Real-Time)

### 🗺️ High-Level Architecture
1. **Event Stream:** Transaction events $\rightarrow$ Feature injection $\rightarrow$ Model.
2. **Inference:** Fast decisioning (<50ms).
3. **Loop:** Alerting $\rightarrow$ Human reviewer $\rightarrow$ Labels $\rightarrow$ Retraining.

### 📐 Technical Deep-Dive
- **Model:** **XGBoost** or **LightGBM**. Fraud data is tabular and highly imbalanced. Boosting handles imbalance well via scale_pos_weight.
- **Features:** "Amount of transactions in the last 10 minutes" (Velocity features). Requires a **Streaming Feature Store** (e.g., Flink/Tecton).
- **Metric:** Focus on **Precision-Recall AUC**. Maximizing Recall (catching fraud) while keeping False Positives (blocking real users) low.

---

## 3. Case Study: Large Scale Search (e.g., E-commerce)

### 🗺️ High-Level Architecture
1. **Query Understanding:** Spell check, Entity extraction, Query expansion.
2. **Retrieval:** BM25 (Keyword) + Semantic Search (Dense vectors).
3. **Ranking:** Learning to Rank (LTR).

### 📐 Technical Deep-Dive
- **Bi-Encoders (Retrieval):** Map query and products to the same space. Fast but ignores query-product nuances.
- **Cross-Encoders (Reranking):** Takes top 100 results and passes (Query, Product) into a Transformer. Very slow, used only for final re-rank.
- **Hybrid Search:** Combine scores from Keyword (BM25) and Semantic using **Reciprocal Rank Fusion (RRF)**.

---

## 4. Case Study: Smart Reply / Auto-Complete (NLP)

### 🗺️ High-Level Architecture
1. **Language Model:** Pre-trained Transformer (e.g., T5 or tiny GPT).
2. **Constrained Search:** Beam Search or Top-P sampling prefix restricted by prefix tree (Trie).
3. **Latency:** Tightly optimized via **Inference Quantization** (Int8) and **KV-Caching**.

---

## ❓ Critical Discussion Points for Seniors (L5)

**1. Personalization vs. Privacy:**
- How do you handle GDPR? (Data anonymization, federated learning).
- Is PII (Personally Identifiable Information) leaking into embeddings?

**2. Handling Skew & Bias:**
- **Positional Bias:** Users click items at the top simply because they are at the top. 
- **Fix:** Include "Position" as a feature during training, but set it to "1" or a fixed value during serving.

**3. Online vs. Offline Metrics:**
- Why does "Offline AUC" go up but "Online CTR" go down?
- **Root Cause:** Usually **Selection Bias** or **Data Leakage**. Your model learned a signal that only exists in historical logs but not in live real-time (e.g., future data leaked into features).
