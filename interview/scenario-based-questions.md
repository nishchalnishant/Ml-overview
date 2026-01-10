# Scenario-Based Questions: 25+ Practical Cases

---

## 🚨 Debugging & Troubleshooting

### Scenario 1: Model accuracy dropped 10% overnight.
**How do you investigate?**
1. Check data pipeline for schema changes, missing values, new nulls.
2. Compare feature distributions (PSI, K-S test) vs baseline.
3. Check for deployment/config changes.
4. Slice by segments—is drop uniform or localized?
5. Check label distribution if available.

### Scenario 2: Training loss decreases but validation loss increases.
**What's happening?**
> **Overfitting**. Model memorizes training data. Fix: More data, regularization, dropout, early stopping, simpler model.

### Scenario 3: Both training and validation accuracy are low.
**What's happening?**
> **Underfitting**. Model too simple. Fix: More features, more complex model, train longer, less regularization.

### Scenario 4: Model works great in testing but fails in production.
**What went wrong?**
1. **Data leakage**: Future information in training features.
2. **Train-serve skew**: Different preprocessing in training vs serving.
3. **Distribution shift**: Production data different from training.
4. Check for missing features or incorrect feature computation.

### Scenario 5: Your model has 99% accuracy but is useless.
**Why?**
> **Class imbalance**. If 99% of data is negative, predicting all negative gives 99% accuracy but catches no positives. Use F1, Precision, Recall, PR-AUC instead.

---

## ⚖️ Trade-offs & Decisions

### Scenario 6: Choose between 90% accurate fast model vs 92% accurate slow model.
**How do you decide?**
1. Quantify business impact of 2% accuracy difference.
2. Real-time system? Speed matters more.
3. Batch system? Accuracy matters more.
4. Consider: Can you distill the slow model?
5. Hybrid: Fast model for most cases, slow model for edge cases.

### Scenario 7: Stakeholders want the model deployed tomorrow. You think it's not ready.
**What do you do?**
1. Present data: error analysis, edge cases, risk scenarios.
2. Propose limited rollout (1% traffic) with monitoring.
3. Ensure robust alerting and rollback plan.
4. Document concerns in writing.

### Scenario 8: You have limited labeling budget. How do you prioritize?
**Approach:**
1. **Active Learning**: Label points where model is most uncertain.
2. **Cluster-based**: Sample from each cluster to ensure diversity.
3. **Error analysis**: Label examples similar to current errors.
4. **Stratified**: Ensure all classes represented.

### Scenario 9: Your training data is 3 years old. Is it still valid?
**Consider:**
1. Has the domain changed? (Concept drift)
2. Are user behaviors different now?
3. Are there new features/entities?
4. Solution: Add recent data, monitor for drift, retrain regularly.

### Scenario 10: Client wants explainability for a black-box model.
**Options:**
1. **SHAP values**: Local and global explanations.
2. **LIME**: Local interpretable model-agnostic explanations.
3. **Permutation importance**: Global feature importance.
4. **Surrogate model**: Train interpretable model on black-box predictions.
5. **Partial Dependence Plots**: Show feature effects.

---

## 🏗️ System Design Scenarios

### Scenario 11: Design a fraud detection system.
**Key points:**
1. **Latency**: Real-time (<50ms).
2. **Model**: XGBoost/LightGBM for tabular data.
3. **Features**: Velocity features (transactions in last 10 min).
4. **Metrics**: Focus on Recall (catch fraud) with acceptable Precision.
5. **Feedback loop**: Human review → labels → retrain.

### Scenario 12: Design a recommendation system for an e-commerce site.
**Key points:**
1. **Two-stage**: Candidate generation (fast) → Ranking (accurate).
2. **Embeddings**: User and item embeddings via Two-Tower.
3. **ANN search**: Faiss/HNSW for retrieval.
4. **Ranking features**: User history, item attributes, context.
5. **Cold start**: Content-based for new items/users.

### Scenario 13: Design a content moderation system.
**Key points:**
1. **Multi-modal**: Text (BERT), Image (CLIP), Video (SlowFast).
2. **Rules layer**: Hashlist for known bad content, regex for keywords.
3. **ML layer**: Classification models for nuanced cases.
4. **Human review**: Queue flagged content for human decision.
5. **Latency**: Must be real-time to block before publish.

### Scenario 14: Design a search ranking system.
**Key points:**
1. **Query understanding**: Spell check, entity extraction.
2. **Retrieval**: BM25 + Dense embeddings (Bi-encoder).
3. **Ranking**: Cross-encoder or LTR (Learning to Rank).
4. **Personalization**: User history, preferences.
5. **Evaluation**: NDCG, MRR.

### Scenario 15: Design a real-time bidding system for ads.
**Key points:**
1. **Latency**: Must respond in ~10ms.
2. **Model**: Light model (logistic regression, small GBM).
3. **Features**: User demographics, context, ad metadata.
4. **Bid calculation**: Predicted CTR × Value per click.
5. **Calibration**: Ensure predicted probabilities are accurate.

---

## 🎯 ML-Specific Scenarios

### Scenario 16: Your model is great on average but terrible for a minority group.
**What do you do?**
1. Identify: Slice metrics by group.
2. Check if training data is representative.
3. Collect more data for underrepresented groups.
4. Use fairness constraints (demographic parity, equalized odds).
5. Report disaggregated metrics to stakeholders.

### Scenario 17: You need to deploy a model to mobile devices.
**Considerations:**
1. **Model size**: Quantization (INT8), pruning.
2. **Framework**: TensorFlow Lite, ONNX, Core ML.
3. **Latency**: Optimize for on-device inference.
4. **Updates**: How to push new models to devices?

### Scenario 18: Your recommendation model increases CTR but users are less satisfied.
**Root cause:**
1. Optimizing for clicks, not satisfaction.
2. Clickbait: Users click but regret.
3. **Fix**: Optimize for engagement quality (watch time, not clicks). Add negative signals (quick bounce, hide, report).

### Scenario 19: You have a cold-start problem for new users.
**Solutions:**
1. Ask for explicit preferences during onboarding.
2. Use content-based features (item attributes).
3. Show popular items initially.
4. Multi-armed bandit for exploration.
5. Transfer from similar users via demographic features.

### Scenario 20: Your model's predictions are well-calibrated in training but poorly calibrated in production.
**Causes:**
1. Distribution shift.
2. Different population in production.
3. **Fix**: Recalibrate on recent production data. Use Platt scaling or isotonic regression.

---

## 🧠 Behavioral & Soft Skills

### Scenario 21: You disagree with your tech lead on the modeling approach.
**How do you handle?**
1. Present data-driven arguments.
2. Propose an experiment to test both approaches.
3. Understand their perspective—they may have context you lack.
4. Disagree and commit if decision is made.

### Scenario 22: Non-technical stakeholder doesn't trust the model.
**Build confidence:**
1. Show specific examples of correct predictions.
2. Run shadow mode: Compare model vs current process.
3. Quantify error rate vs human error rate.
4. Explain with analogies, not jargon.

### Scenario 23: Your model was deployed and caused a PR incident.
**Response:**
1. Immediate: Rollback to previous model.
2. Investigate: What input caused the failure?
3. Document: Post-mortem with root cause.
4. Prevent: Add guardrails, edge-case testing, monitoring.

### Scenario 24: You inherit a model with no documentation.
**What do you do?**
1. Trace data sources and feature engineering.
2. Run test predictions to understand behavior.
3. Check version control for development history.
4. Talk to original authors if available.
5. Document as you learn.

### Scenario 25: You're asked to build a model with very little data.
**Options:**
1. **Transfer learning**: Pre-trained models, fine-tune on small data.
2. **Data augmentation**: Create synthetic variations.
3. **Few-shot learning**: Use prompting or prototypical networks.
4. **Active learning**: Label most informative samples.
5. **Simpler models**: Less prone to overfit on small data.
