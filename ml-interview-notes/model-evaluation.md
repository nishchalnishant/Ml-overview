# Model Evaluation

Strong interview answers on evaluation always connect the metric to the decision being made. The goal is not to list formulas; it is to show that you know what a metric hides, when it fails, and how it maps to product impact.

---

# Q1: What are precision, recall, F1 score, and accuracy?

**Interview-ready answer**

Accuracy measures the fraction of total predictions that are correct. Precision measures how reliable the positive predictions are, while recall measures how much of the actual positive class you capture. F1 score is the harmonic mean of precision and recall, so it is useful when you want a single number that penalizes imbalance between them. In interviews, the key point is that these metrics answer different business questions and are not interchangeable.

**Good framing**

- Precision matters when false positives are expensive.
- Recall matters when false negatives are expensive.
- Accuracy is only safe when classes are balanced and error costs are similar.

---

# Q2: What is the confusion matrix, and how do you interpret it?

**Interview-ready answer**

A confusion matrix breaks predictions into counts by actual class versus predicted class. For binary classification that gives you true positives, false positives, true negatives, and false negatives. Its value is that it shows where the model is wrong, not just how often it is wrong. That makes it one of the most useful tools for threshold tuning and slice-based error analysis.

**Good nuance**

For multiclass problems, the off-diagonal cells often reveal systematic confusions that aggregate metrics hide.

---

# Q3: What are common evaluation metrics for Classification?

**Interview-ready answer**

The common metrics fall into three families. Threshold-based metrics include accuracy, precision, recall, F1, and balanced accuracy. Ranking metrics include ROC-AUC and PR-AUC. Probability-quality metrics include log loss, Brier score, and calibration error. A strong answer explains that the correct family depends on whether you care about hard decisions, relative ordering, or calibrated probabilities.

---

# Q4: When would you use accuracy vs other metrics?

**Interview-ready answer**

I would use accuracy when the class distribution is reasonably balanced and the costs of false positives and false negatives are similar. Otherwise, accuracy can be deeply misleading. For fraud, medical diagnosis, moderation, or any rare-event problem, I would usually move to precision, recall, F1, PR-AUC, balanced accuracy, or a business-specific cost metric. In interviews, it helps to say that accuracy is often a reporting metric, not the optimization target.

---

# Q5: When would you use log loss vs accuracy?

**Interview-ready answer**

Use log loss when the quality of predicted probabilities matters, not just whether the final class label is right. Log loss heavily penalizes confident wrong predictions, so it is valuable for ranking systems, risk models, bidding systems, and any pipeline that makes downstream decisions from probabilities. Accuracy ignores confidence completely, so two models with the same accuracy can be very different in usefulness if one is much better calibrated.

---

# Q6: What metrics would you use for a multi-class classification problem?

**Interview-ready answer**

For multiclass problems I would usually report a confusion matrix, accuracy, and macro, weighted, or micro F1 depending on what matters. Macro metrics are best when rare classes matter equally, weighted metrics reflect class frequency, and micro metrics summarize overall behavior across all examples. If the model outputs probabilities, I would also consider multiclass log loss and calibration checks.

---

# Q7: How do you handle class imbalance in classification metrics?

**Interview-ready answer**

I switch to metrics that reflect minority-class performance and tune thresholds intentionally. That often means PR-AUC, recall at a given precision, F1, balanced accuracy, or MCC instead of raw accuracy. I also make sure evaluation happens on a distribution that matches deployment, because resampling the validation or test set can produce the wrong operational threshold.

---

# Q8: What is the ROC curve? What is AUC?

**Interview-ready answer**

The ROC curve shows the tradeoff between true positive rate and false positive rate across all thresholds. AUC summarizes that curve and can be interpreted as the probability that the model ranks a random positive above a random negative. Its strength is threshold independence, but in very imbalanced problems it can hide poor precision, so I usually look at PR-AUC as well.

---

# Q9: How do you handle imbalanced datasets?

**Interview-ready answer**

I would treat imbalance as a problem of data, objective, thresholding, and evaluation together. Options include class weighting, focal loss, resampling, better negative sampling, threshold adjustment, and richer features for the minority class. The most important point is that you should choose the intervention based on the operational goal, such as maximizing recall at a fixed false-positive budget, rather than assuming that oversampling is always the answer.

---

# Q10: What are common evaluation metrics for Regression?

**Interview-ready answer**

For regression, the standard metrics are MAE, MSE, RMSE, and sometimes R-squared. MAE is easier to interpret and more robust to outliers. MSE and RMSE penalize large errors more heavily. I choose based on the business cost of error: if large misses are especially harmful, RMSE or MSE may be more appropriate; if robustness and interpretability matter, MAE is often better.

---

# Q11: What's the difference between MAE, MSE, and RMSE?

**Interview-ready answer**

MAE averages absolute errors, so every unit of error contributes linearly. MSE averages squared errors, which means large mistakes are penalized disproportionately. RMSE is just the square root of MSE, so it preserves that larger-error penalty while bringing the metric back to the original target units. In interviews, say that the choice depends on how much you want to punish large misses and how interpretable the metric needs to be.

---

# Q12: How do you choose the right evaluation metric for a given problem?

**Interview-ready answer**

I start from the decision the model supports, not from the model type alone. The right metric depends on class balance, error asymmetry, ranking needs, calibration needs, and whether the final user cares about aggregate performance or performance on specific slices. A strong answer here is: "I choose the metric that best reflects the product cost function, then I pair it with diagnostic metrics so I can see what the headline number is hiding."

---

# Q13: How do you compare the performance of different models?

**Interview-ready answer**

I compare models under the same data split, same feature availability, same preprocessing, and the same evaluation metric set. Then I look beyond a single score: calibration, stability across seeds, subgroup performance, latency, memory, and operational complexity all matter. If the performance gap is small, I usually prefer the simpler model unless the complex one clearly wins where the business cares most.

---

# Q14: Explain cross-validation and its importance.

**Interview-ready answer**

Cross-validation estimates model performance more robustly by repeatedly training and validating across different splits of the data. Its main value is reducing dependence on a single lucky or unlucky split, especially when data is limited. In interviews, the important nuance is that the split strategy must match the data: stratified folds for class balance, group-aware folds for repeated entities, and time-based folds for temporal problems.

---

# Q15: What is Hyperparameter Tuning?

**Interview-ready answer**

Hyperparameter tuning is the process of selecting configuration values that are not learned directly from the training data, such as learning rate, tree depth, regularization strength, or batch size. The point is not to search blindly; it is to find a configuration that generalizes well under a valid evaluation setup. Strong candidates mention validation discipline, reproducibility, and the danger of overfitting to the validation set itself.

---

# Q16: How do you evaluate unsupervised learning models?

**Interview-ready answer**

Unsupervised evaluation depends on the task because there is often no ground truth label. For clustering, I might use cohesion and separation metrics such as silhouette score, Davies-Bouldin, or Calinski-Harabasz, but I would not stop there. The strongest evaluation is often downstream usefulness: whether the clusters are actionable, stable, and meaningful to the business or to a later supervised model.

---

# Q17: How do you evaluate a clustering algorithm?

**Interview-ready answer**

I evaluate clustering from three angles: internal structure, external agreement if labels exist, and practical usefulness. Internal metrics look at compactness and separation. External metrics compare against known labels using measures like adjusted Rand index or NMI when labels are available. Practical evaluation checks whether the clusters are interpretable, stable under perturbation, and useful for segmentation, retrieval, or downstream prediction.

---

# Q18: What metrics would you use for a recommendation system?

**Interview-ready answer**

For recommendation, offline metrics usually focus on ranking quality and coverage: precision@k, recall@k, MAP, NDCG, hit rate, diversity, and coverage are common. But offline ranking metrics are not enough by themselves because recommendation systems interact with users and create feedback loops. So in production I would also care about online metrics such as CTR, conversion, watch time, retention, novelty, and long-term user satisfaction.

---

# Q19: What is A/B testing in the context of ML?

**Interview-ready answer**

A/B testing is the process of comparing a new model or system against a control in a live environment by randomly assigning traffic and measuring outcome differences. It matters because many offline wins do not survive contact with real users, delayed feedback, or feedback loops. In an interview, the strongest answer mentions experiment design, guardrail metrics, statistical power, ramp strategy, and the fact that you often need both offline validation and online testing to trust a change.
