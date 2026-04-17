# Practical Machine Learning Scenarios

This hub contains situational "war stories" and troubleshooting scenarios common in senior ML interviews. These test your ability to diagnose production failures and design robust systems beyond just code.

---

# Q1: Your model has 95% training accuracy but 60% in production. How do you diagnose?

## 1. 🔹 Direct Answer
This is likely caused by **Train-Serve Skew** or **Data Leakage**. Diagnostic steps include:
1. Identifying features used in training that are unavailable at inference.
2. Checking for **Data Drift** (distribution shift between training and live data).
3. Verifying preprocessing consistency (e.g., using training-time statistics for scaling live data).

## 2. 🔹 Intuition
"Studying the test answers before the exam." If the model sees information it shouldn't, or if the exam covers a completely different subject than the study guide, it will fail in the real world.

## 3. 🔹 Practical Perspective
- **Leakage Check:** "Was a click?" or "Transaction timestamp" often leak the future into the past.
- **Drift Check:** Use metrics like **PSI (Population Stability Index)** or **KL-Divergence** to compare feature distributions.

## 4. 🔹 Difficulty Tag: 🟢 Easy

---

# Q2: Fraud accounts for 0.01% of your data. How do you build a useful model?

## 1. 🔹 Direct Answer
Do NOT use Accuracy. Instead:
1. **Metrics:** Use Precision-Recall AUC, F1-Score, or Cost-Weighted metrics.
2. **Sampling:** Use SMOTE (oversampling) or strategic undersampling.
3. **Algorithm:** Use cost-sensitive learning (penalize FN more than FP).
4. **Logic:** Adjust decision thresholds based on the business cost of fraud vs. friction.

## 2. 🔹 Intuition
If you have 10,000 people and only 1 is a thief, a model that says "everyone is a saint" is 99.99% accurate but 0% useful. You need a model that is "sensitive" to the rare 1.

## 3. 🔹 Code Snippet
```python
# Weighted loss in XGBoost
model = XGBClassifier(scale_pos_weight=99) # 99:1 imbalance
```

## 4. 🔹 Difficulty Tag: 🟡 Medium

---

# Q3: RAG-based LLM is hallucinating facts. How do you stabilize it?

## 1. 🔹 Direct Answer
Ground the model using:
1. **Prompt Constraints:** "Answer ONLY using provided context. If unsure, say 'I don't know'."
2. **Citations:** Force the model to provide source IDs for every claim.
3. **Retrieval Tuning:** Improve chunking, use rerankers (Cohere/BGE), and increase Top-K.
4. **Verification:** Use an "Evaluator" model to check if the response is supported by the context.

## 2. 🔹 Intuition
Don't ask the model to "know" everything—ask it to "summarize the open book." If it starts telling stories not in the book, it's failed its task.

## 3. 🔹 Practical Perspective
- **Citation Check:** "Check if [Source 1] actually contains the sentence 'X'."
- **Hybrid Search:** Use BM25 + Dense Embeddings to ensure exact keyword matching isn't missed.

## 4. 🔹 Difficulty Tag: 🔴 Hard

---

# Q4: Model latency is too high for a real-time recommendation system. Solutions?

## 1. 🔹 Direct Answer
Optimize the stack via:
1. **Model Side:** Quantization (INT8), Pruning, or Distillation (Student/Teacher).
2. **Inference Side:** KV-Caching (for LLMs), TensorRT/ONNX Runtime.
3. **Architecture:** Two-stage retrieval (Retrieve 1k candidates with simple ANN -> Rank top 10 with complex model).

## 2. 🔹 Intuition
You can't do a full background check on every person in the stadium to find a friend. You shout their name (Retrieval) and then look closely at the few people who turn around (Ranking).

## 3. 🔹 Difficulty Tag: 🟡 Medium

---

# 📚 Quick-Fire Scenarios Library

| Scenario | Diagnosis/Action |
| :--- | :--- |
| **Dead ReLU** | Use Leaky ReLU; use lower Learning Rate. |
| **Feedback Loop** | Approve a small % of "rejected" users (Exploration). |
| **Cold Start** | Use Content-Based Filtering or global popularity. |
| **Concept Drift** | People stop buying at Price X. Retrain on fresh data. |
| **Vanishing Gradient** | Add Residual Connections; use Batch Norm. |
| **Multi-Collinearity** | Check VIF; use Ridge/Lasso regularization. |

---

## 🔹 One-line Revision
Practical ML is about identifying the gap between mathematical theory and production messy-ness (drift, skew, latency).
