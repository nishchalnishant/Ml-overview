# Practical & Scenario-Based ML Interview Questions

These "What would you do if..." questions test your practical engineering judgment.

---

## 🚨 Debugging & Problem Solving

### Scenario 1: Your model's accuracy dropped 10% overnight.
**How do you investigate?**

1. **Check Data Pipeline**: Was there a data source change? Schema drift? Missing values appearing?
2. **Compare Distributions**: Use PSI or K-S test to compare yesterday's input distribution to today's.
3. **Check Labels**: If you have labels, is the label distribution different?
4. **Check Infrastructure**: Any deployment changes? Model version rollback?
5. **Segment Analysis**: Is the drop uniform or concentrated in a specific user segment/region?

**Answer Framework:**
> "First, I'd check if the data pipeline is healthy—are there new nulls or schema changes? Then I'd compare feature distributions against a baseline window. If data looks fine, I'd check for code/config changes in the deployment. Finally, I'd slice performance by segments to isolate where the drop is happening."

---

### Scenario 2: You have 1 million users but only 100 fraud cases.
**How do you build a fraud detection model?**

1. **Don't use Accuracy**: Use Precision, Recall, F1, or **PR-AUC**.
2. **Resampling**: SMOTE (oversample minority), Undersampling (downsample majority), or Hybrid.
3. **Class Weights**: Penalize misclassifying fraud more heavily (`class_weight='balanced'`).
4. **Anomaly Detection**: Treat fraud as an anomaly using Isolation Forest or Autoencoders.
5. **Two-Stage**: Use rules/heuristics to flag candidates, then a model for precision.

**Answer Framework:**
> "I'd frame this as an anomaly detection problem due to extreme imbalance. I'd use PR-AUC as my metric, apply SMOTE or class weighting, and consider an autoencoder approach where a high reconstruction error flags anomalies."

---

### Scenario 3: Stakeholders want explainability for a Black Box model.
**How do you provide it?**

1. **Global Interpretability**: Feature Importance (Permutation Importance, SHAP summary plots).
2. **Local Interpretability**: LIME or SHAP values for a single prediction.
3. **Surrogate Models**: Train a simple Decision Tree on the complex model's predictions.
4. **Partial Dependence Plots (PDPs)**: Show the marginal effect of one feature on prediction.

**Answer Framework:**
> "I'd use SHAP values for both global and local explanations. For stakeholders, I'd provide a summary plot showing which features matter most across the population, and for individual cases, a SHAP force plot explaining why *this specific* prediction was made."

---

## 🏗️ System Design Scenarios

### Scenario 4: Design a real-time content moderation system for a social media platform.
**Requirements:** Flag harmful content (hate speech, violence) before it's published. Latency < 200ms.

**Architecture:**
1. **Input**: Text/Image/Video from user post.
2. **Fast Path (Rules)**: Regex for known banned keywords. Hashlist for known harmful images (PhotoDNA).
3. **ML Path (Parallel)**: 
   - Text: Distilled BERT classifier.
   - Image: CLIP or fine-tuned ResNet.
4. **Decision**: If any path flags, route to human review queue.
5. **Feedback Loop**: Human review labels feed back into training data.

**Key Trade-offs:**
- **Latency vs. Accuracy**: Use smaller, distilled models.
- **Recall vs. Precision**: Err on the side of recall (flag more, humans review) to protect users.

---

### Scenario 5: You built a recommendation model. Click-through rate is up, but user satisfaction is down. Why?

**Root Causes:**
1. **Clickbait Optimization**: Model learned to recommend sensational content that gets clicks but disappoints.
2. **Engagement ≠ Satisfaction**: A user might click out of anger or confusion.
3. **Short-term vs. Long-term**: CTR is a short-term metric; user retention is long-term.

**Solution:**
1. **Change the Metric**: Optimize for **Watch Time**, **Completion Rate**, or **Explicit Rating**.
2. **Add Negative Signals**: Factor in regret signals like "Watch < 5 seconds then close", "Hide post", "Report".
3. **Multi-Objective Optimization**: Balance CTR with a satisfaction proxy.

---

## 💡 Behavioral / Soft Skill Scenarios

### Scenario 6: Your manager wants to deploy a model you think isn't ready.
**How do you handle it?**

> "I'd present data-driven concerns: show the error analysis, highlight risky edge cases, and propose a limited rollout (e.g., 1% of traffic) with monitoring. I'd also propose a timeline for addressing the concerns. If pushed to deploy anyway, I'd ensure robust alerting and a quick rollback plan."

---

### Scenario 7: A non-technical stakeholder doesn't trust the ML model.
**How do you build confidence?**

1. **Explain with Analogies**: "The model is like a very fast intern who's studied 1 million past cases."
2. **Show Examples**: Walk through specific predictions the model got right.
3. **Shadow Mode**: Run the model in parallel with human decisions for a week and compare.
4. **Quantify Errors**: Show the model's error rate vs. the current process's error rate.

---

## 🧠 Edge Case & Tricky Questions

### Scenario 8: Your training and test accuracy are both 95%, but production accuracy is 60%.
**What happened?**

1. **Data Leakage**: A feature that exists in training (e.g., future data) doesn't exist in production.
2. **Train-Serve Skew**: Different preprocessing logic between training and serving.
3. **Distribution Shift**: Production data comes from a different population than training data.
4. **Feedback Loops**: The model's past predictions are influencing future data (e.g., a loan model changing who applies for loans).

**Answer Framework:**
> "This is a classic train-serve skew symptom. I'd first check for data leakage by examining feature importance for suspiciously powerful features. Then I'd compare feature distributions between training data and live serving logs. Finally, I'd audit the preprocessing code to ensure it's identical."

---

### Scenario 9: You need to choose between a 90% accurate model that's fast vs. an 92% accurate model that's 10x slower.
**What do you pick?**

**It Depends:**
- **Real-time (Search, Fraud)**: Speed is critical. Pick the faster model. The 2% difference is usually not worth the latency hit.
- **Batch (Loan approval, Medical)**: Accuracy is critical. Pick the slower model. Process offline.
- **Hybrid**: Use the fast model for initial filtering, the slow model for top candidates.

**Answer Framework:**
> "I'd quantify the business impact of the 2% accuracy difference. If we're saving $10M in fraud, 2% is $200k—likely worth the slower model. If it's a recommendation system where speed is part of the user experience, the faster model is better. I'd also explore distillation to get the best of both worlds."

---

### Scenario 10: Your model is great on average, but terrible for a minority group.
**What do you do?**

1. **Identify**: Slice performance by demographic/segment.
2. **Diagnose**: Is there less training data for this group? Are features less informative for them?
3. **Mitigate**:
   - Collect more representative data.
   - Use stratified sampling.
   - Apply fairness constraints (demographic parity, equalized odds).
   - Use separate calibration for each group.
4. **Document**: Report fairness metrics alongside overall metrics.

**Answer Framework:**
> "This is a fairness issue. I'd first ensure we have enough data from that group. If not, I'd collect more. Then I'd audit features for potential proxies for the protected attribute. Finally, I'd apply post-processing calibration or in-processing fairness constraints and report disaggregated metrics to stakeholders."
