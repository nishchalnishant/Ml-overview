# PART 6: EVALUATION DECISION FRAMEWORK

## Goal
To teach candidates how to select the right metrics for each problem type, and how to connect offline metrics to real-world business KPIs.

## Mental Model
**"Optimize offline metrics, validate with online metrics, and measure business impact."**
A model can have high AUC but zero business value. A senior engineer understands why the gap exists and how to close it.

---

## 6.1 The Evaluation Pyramid

```text
[Business KPIs]           ← Revenue, Engagement, Retention
       ↑
[Online Metrics]          ← CTR, Conversion Rate, Avg Session Time (A/B Test)
       ↑
[Offline Metrics]         ← AUC, NDCG, F1, BLEU
       ↑
[Engineering Metrics]     ← Latency P50/P95/P99, Throughput, Error Rate
```

**Rule:** Optimize the lowest layer, but validate at the highest layer. Offline metrics that don't correlate with online metrics are the wrong metrics.

---

## 6.2 Classification Metrics

### Decision Tree
```text
Is the dataset balanced?
├── YES → Accuracy is acceptable.
└── NO (Imbalanced) →
    ├── Do you care more about catching True Positives? → Recall (Sensitivity).
    ├── Do you care more about Precision of alerts? → Precision.
    ├── Balanced tradeoff → F1 Score (harmonic mean of P and R).
    ├── Need to tune threshold? → ROC-AUC (threshold-independent).
    └── Severely imbalanced? → PR-AUC (Precision-Recall Curve), better for rare events.
```

| Metric | Formula | Use Case |
| :--- | :--- | :--- |
| **Accuracy** | TP+TN / Total | Balanced datasets only |
| **Precision** | TP / (TP + FP) | Spam filter (cost of false alarm is high) |
| **Recall** | TP / (TP + FN) | Cancer detection (cost of missing is high) |
| **F1** | 2 * P * R / (P + R) | Imbalanced, balanced tradeoff |
| **ROC-AUC** | Area under ROC curve | Model comparison, threshold-invariant |
| **PR-AUC** | Area under PR curve | Heavily imbalanced, rare positive events |
| **Log Loss** | -log(p(y)) | Probabilistic calibration required |

### Game AI Example
**Toxicity Detection:** Maximize Recall first (catch all toxic messages), then tune Precision threshold to reduce false positives based on user complaint rates.

---

## 6.3 Regression Metrics

| Metric | Use When | Sensitivity |
| :--- | :--- | :--- |
| **MSE** | Penalizing large errors is critical | Very sensitive to outliers |
| **RMSE** | Same as MSE, interpretable units | Very sensitive to outliers |
| **MAE** | Robust to outliers required | Robust |
| **MAPE** | Percentage error matters (forecasting) | Fails when actuals are near 0 |
| **R²** | Proportion of variance explained | Business-friendly |
| **Huber Loss** | Balance between MSE and MAE | Semi-robust |

---

## 6.4 Ranking Metrics

| Metric | Definition | Use When |
| :--- | :--- | :--- |
| **Precision@K** | Fraction of top-K results that are relevant | Simple search quality |
| **Recall@K** | Fraction of relevant items in top-K | Information retrieval |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Position-aware ranking (search, recs) |
| **MRR** | Mean Reciprocal Rank | Q&A, first-correct-answer matters |
| **MAP** | Mean Average Precision | Multi-document retrieval |

**NDCG vs MAP:** NDCG allows graded relevance (e.g., 0, 1, 2, 3), while MAP uses binary relevance. Use NDCG for recommendation and search where graded relevance exists.

---

## 6.5 LLM / RAG Metrics

### Automatic Metrics
| Metric | Measures | Tool |
| :--- | :--- | :--- |
| **ROUGE-L** | Recall-oriented overlap for summarization | HuggingFace |
| **BLEU** | Precision-oriented overlap for translation | NLTK |
| **BERTScore** | Semantic similarity (embedding-based) | BERTScore |
| **Perplexity** | How surprised the model is by a text | Native |
| **Faithfulness** | Does the answer stick to the context (RAG)? | RAGAS |
| **Answer Relevance** | Is the answer relevant to the question? | RAGAS |
| **Context Precision** | Is the retrieved context relevant? | RAGAS |
| **Context Recall** | Was all relevant context retrieved? | RAGAS |

### Human Evaluation (G-Eval Framework)
For LLMs, automatic metrics often fail. Use human evaluation on:
1. **Coherence:** Is the output logically structured?
2. **Fluency:** Is the grammar correct?
3. **Relevance:** Does the answer address the question?
4. **Faithfulness:** Is the answer grounded in the source documents (no hallucinations)?

---

## 6.6 Agents Metrics

| Metric | Definition |
| :--- | :--- |
| **Task Completion Rate** | % of tasks the agent fully completed correctly |
| **Steps to Completion** | How many tool calls/steps were needed (efficiency) |
| **Hallucination Rate** | % of tool calls with incorrect arguments |
| **Cost per Task** | Total token cost to complete one task |
| **Retry Rate** | How often the agent needed to self-correct |

---

## 6.7 Offline vs. Online Evaluation

### When to Use Each

| Method | Description | When to Use |
| :--- | :--- | :--- |
| **Offline Eval** | Evaluate on a held-out test set before deployment. | Always. First gate before going live. |
| **Shadow Testing** | Deploy the new model alongside the old one; new model processes requests but responses are not shown to users. | To validate real-traffic behavior without user impact. |
| **A/B Testing** | Randomly split traffic; Group A gets model A, Group B gets model B. | Proving business impact (CTR, conversion). |
| **Canary Release** | Send 1–5% of traffic to new model. Gradually ramp up. | Low-risk rollout, catching issues early. |
| **Multi-armed Bandit** | Dynamically allocate traffic to the better-performing model. | Faster than A/B tests; adaptive testing. |
| **Human Evaluation** | Human raters score model outputs. | LLMs, complex generation tasks, final validation. |

### A/B Test Design Checklist
- [ ] Define a primary metric (conversion rate) and secondary metrics (session duration).
- [ ] Define the minimum detectable effect (MDE) before running.
- [ ] Calculate required sample size (power analysis) to achieve statistical significance.
- [ ] Run for a full business cycle (e.g., 1–2 weeks) to capture weekend effects.
- [ ] Check for novelty effects (users behave differently just because something is new).

---

## Engineering Checklist

- [ ] Is my offline metric correlated with the business KPI I care about?
- [ ] Have I evaluated on a stratified test set (not a random split of time series)?
- [ ] Am I measuring calibration (is a 90% confidence prediction correct 90% of the time)?
- [ ] For LLMs, am I using both automatic and human evaluation?
- [ ] For ranking systems, am I using position-aware metrics (NDCG) not just Precision@K?

## Real-world Examples

- **Netflix:** Offline metric is NDCG (ranking quality). Online A/B metric is "Did the user watch >2 minutes of the recommended content?"
- **Fraud Detection:** Offline: PR-AUC. Online: $$$ fraudulent charges that got through.

## Interview Follow-up Questions & Best Answers

**Q: "Your model has 95% accuracy but the product team says it's failing. What happened?"**
*Best Answer:* "Almost certainly a class imbalance issue. If 95% of examples are negative, a model that always predicts 'negative' achieves 95% accuracy. I would look at precision, recall, and the confusion matrix. I would also check if the business impact metric (e.g., false negatives causing churn) is being captured. The model might be high accuracy but low recall on the minority class that actually matters to the business."
