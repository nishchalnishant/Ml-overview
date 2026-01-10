# Machine Learning Interview Strategy & System Design

A guide for mid-to-senior (L4/L5) ML Engineer roles, focusing on system design, project deep-dives, and technical strategy.

---

## 🗺️ The Interview Landscape (Engineer 2/L4)

At the **Engineer 2 / Senior** level, interviews shift from "What is X?" to "Why X and not Y?". 

### The Standard Loop:
1. **Recruiter Screen:** 30 min high-level background.
2. **Coding / DSA (1-2 Rounds):** LeetCode Medium/Hard. Focus on strings, graphs, and dynamic programming.
3. **ML Coding:** Implementing an algorithm (e.g., K-Means, LR) or data maniupulation (Pandas/Numpy).
4. **ML System Design (The "Closer"):** 45-60 min of leader-led design discussion.
5. **ML Theory Depth:** Probabilistic "what-if" scenarios.
6. **Behavioral / Project Deep Dive:** Deep dive into 1-2 major projects using the **STAR** method.

---

## 🏗️ ML System Design Framework

This is the most critical round. Use this 5-step framework to lead the discussion.

### 1. Problem Scoping & Requirements
- **Goal:** What is the business metric? (e.g., Click-Through Rate (CTR), Revenue, Latency).
- **Constraints:** Max latency (e.g., <100ms), throughput (QPS), scale (millions of users).
- **Type:** Is this search, recommendation, ranking, or classification?

### 2. Data & Feature Engineering
- **Sources:** Logs, user profile, item metadata.
- **Labeling:** Explicit (ratings) vs. Implicit (clicks, watch time). Handling delay in labels.
- **Features:** Categorical (one-hot vs. embeddings), Numerical (scaling), Text (BERT/Word2Vec), Temporal (recency).
- **Storage:** Feature store considerations.

### 3. Modeling
- **Baseline:** Start simple (e.g., Logistic Regression or XGBoost).
- **Advanced:** Deep Learning (e.g., DeepFM for ranking, Transformers).
- **Trade-offs:** Model size vs. Accuracy vs. Inference time.

### 4. Evaluation (Offline & Online)
- **Offline:** AUC, Log-loss, Recall@K, Precision@K, F1.
- **Online:** A/B Testing, Interleaving.
- **Slicing:** Check performance for specific user segments or item categories.

### 5. Deployment & Post-Production
- **Serving:** Batch inference vs. Real-time API.
- **Optimization:** Quantization, Pruning, Distillation.
- **Monitoring:** Data Drift (K-S test), Model Drift, Latency, Error rates.

---

## 📝 Case Study: Ranking YouTube Videos

**Interviewer:** *"Design a system to rank videos on the YouTube home page."*

1. **Clarify:** 
   - Multi-stage ranking? Yes, Candidate Generation → Ranking.
   - Objective? Maximize long-term watch time.
2. **Candidate Generation:** 
   - Filter 1B videos to ~500.
   - Use Two-Tower Networks (User Tower, Video Tower) with Dot Product similarity.
3. **Ranking (The "Heavy" Model):**
   - Goal: Predict the probability of watch time $> X$.
   - Features: User history, Video embeddings, Context (device, time).
   - Label: Continuous watch time or binary "watched > 30s".
4. **Online Serving:**
   - Use ANN (Approximate Nearest Neighbors) like Faiss for candidate retrieval.
   - Re-rank the top 500 using the heavy DNN.
5. **Cold Start:** 
   - For new videos, use content-based features (title, tags) before they get user interactions.

---

## 💡 Pro-Tips for the Deep Dive
- **Explain the "Why":** Why did you choose Adam over SGD in your project? (e.g., faster convergence, handles noisy gradients).
- **Acknowledge Trade-offs:** "We used XGBoost because it was interpretable and handled our tabular data well, even though a MLP might have slightly higher accuracy."
- **Focus on Impact:** Always mention the % improvement in business metrics, not just ML metrics.

---

## 📚 Study Checklist
- [ ] **LeetCode:** ~100-150 Mediums, focus on the "Top 75".
- [ ] **System Design:** Read "Designing Data-Intensive Applications" (DDIA) and "Machine Learning System Design" by Chip Huyen.
- [ ] **Projects:** Be ready to talk for 20 minutes about your best work, including failures and pivots.
