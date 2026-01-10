# ML System Design: Model-to-Use-Case Mapping

A curated catalog of ML models mapped to real-world system design scenarios. Use this to justify your architectural choices during interviews.

---

## 🏗️ Core Architecture Mapping

### 1. Recommendation & Ranking
*Systems: YouTube Home, Netflix Discovery, Amazon "Users also bought"*

- **Candidate Generation (Retrieval):**
  - **Two-Tower Networks:** Efficient for large-scale retrieval (User/Item embeddings).
  - **Matrix Factorization (SVD++):** Classic, great for latent feature discovery.
- **Ranking (Scoring):**
  - **Wide & Deep:** Balances memorization (Wide) and generalization (Deep).
  - **DeepFM:** Better for learning high-order feature interactions without feature engineering.
  - **DCN (Deep & Cross Network):** Explicitly applies cross-features at each layer.

### 2. Search & Retrieval
*Systems: Google Search, E-commerce Search, Chatbot Knowledge Retrieval*

- **Traditional:** **BM25** (Better than TF-IDF for term frequency saturation).
- **Dense Retrieval:** **Bi-Encoders** (Sentence-BERT) for fast ANN search.
- **Re-ranking:** **Cross-Encoders** (BERT-Ranker) for high precision at the cost of latency.
- **Advanced:** **ColBERT v2** (Token-level late interaction) for state-of-the-art retrieval.

### 3. Fraud & Anomaly Detection
*Systems: Credit Card Fraud, Ad-Click Fraud, System Intrusion*

- **Supervised:** **XGBoost / CatBoost** (Handles tabular data, missing values, and imbalanced classes natively).
- **Unsupervised:** **Isolation Forest** (Great for detecting outliers in high dimensions) or **Autoencoders** (Detects anomalies via high reconstruction error).
- **Relational:** **Graph Neural Networks (GNNs)** to detect "fraud rings" or money laundering patterns.

### 4. Content Moderation & Safety
*Systems: Social Media Filtering (NSFW), Toxic Comment Detection*

- **Text:** **RoBERTa / DeBERTa** (Industry standards for sequence classification).
- **Visual:** **CLIP** (Zero-shot classification for new categories) or **YOLOv10** (Real-time detection).
- **Sequential:** **SlowFast Networks** for detecting violence or dynamic actions in video.

---

## ⚡ How to Justify Your Choice

In an interview, don't just state the model. Use these lenses:

1. **Latency vs. Accuracy:** 
   - *"We use a Logistic Regression baseline for the live ad-serving tier because it has <5ms latency, even though a Transformer might be 2% more accurate but take 200ms."*
2. **Data Type:**
   - *"Since our data is primarily tabular with 40% missing values, I'll use CatBoost which handles categorical features and missingness natively."*
3. **Training Recency:**
   - *"For a news-ranking system where trends change hourly, I'd prefer a simpler model (Linear/LR) that can be re-trained frequently or updated online."*
4. **Explainability:**
   - *"For credit scoring (regulated), I'll use a globally interpretable model like a Decision Tree or use SHAP values with a Random Forest."*

---

## 📚 Specialized Use-Cases

| **Use Case** | **Recommended Model** | **Why?** |
|--------------|-----------------------|-----------|
| **Demand Forecasting** | **TFT (Temporal Fusion Transformer)** | Handles multiple time-series, static metadata, and provides uncertainty bounds. |
| **OCR / Doc Parsing** | **LayoutLMv3** | Modern "Multimodal" transformer that sees both text and spatial position. |
| **Zero-Shot Image Tagging** | **CLIP (OpenAI)** | Learned on 400M image-text pairs; no labeling needed for new tags. |
| **Speech-to-Text** | **Whisper (OpenAI)** | Robust to noise and multiple languages out of the box. |

---

## 👉 Interview Tip: The "Two-Stage" Pattern
Almost every large-scale ML system follows the **Retrieval → Ranking** pattern.
1. **Retrieval (Fast):** Filter 100M items to 500 using ANN (Faiss) and simple embeddings.
2. **Ranking (Precise):** Rank those 500 items using a heavy, multi-feature Deep Learning model.
