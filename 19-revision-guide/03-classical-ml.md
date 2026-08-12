---
module: Classical ML
topic: Revision Guide
subtopic: ""
status: unread
tags: [revision, classical-ml, supervised-learning, unsupervised-learning, ensembles, dimensionality-reduction, time-series, hyperparameter-optimization, anomaly-detection]
prerequisites: [03-classical-ml]
---
# 03 Classical ML — Revision Guide

Single-page revision for the entire `03-classical-ml/` folder: supervised/unsupervised algorithms, ensembles, dimensionality reduction, time series, hyperparameter optimization, ML fundamentals, anomaly detection, and when classical ML beats deep learning. Use the outline to reconstruct the folder's structure from memory, then self-test with the graded Q&A below.

## Structure Overview

```markdown
- **03-classical-ml**
  - **Supervised Learning**
    - Linear Regression
    - Logistic Regression
    - SVM + kernel trick
    - Decision Trees
    - KNN
    - Naive Bayes
    - Regularization L1/L2/ElasticNet
    - XGBoost/LightGBM/CatBoost compare
    - Algorithm selection framework
  - **Unsupervised Learning**
    - K-Means + K-Means++
    - DBSCAN core/border/noise
    - GMM / EM E-step M-step
    - Hierarchical clustering linkages
    - Clustering eval: silhouette/CH/DB/ARI/NMI
  - **Ensemble Methods**
    - Why ensembles work: bias-variance
    - Bagging + OOB
    - Boosting: AdaBoost
    - Gradient Boosting
    - Voting hard/soft
    - Stacking OOF meta-learner
    - Blending
    - Snapshot ensembles
    - Multi-seed ensemble
    - Calibration before ensembling
  - **Dimensionality Reduction**
    - Curse of dimensionality
    - Manifold hypothesis
    - PCA eigendecomposition/SVD
    - Kernel PCA
    - ICA
    - LDA Fisher criterion
    - t-SNE crowding problem
    - UMAP
    - NMF parts-based
    - Autoencoders / VAE
  - **Time Series Analysis**
    - Stationarity + ADF test
    - ACF/PACF order identification
    - ARIMA/SARIMA
    - Exponential Smoothing Holt-Winters
    - Prophet trend+seasonality+holidays
    - Walk-forward validation
    - Tree-based lag features
    - DL for TS: TCN/PatchTST/N-BEATS
    - Foundation models: Chronos/TimesFM
    - TS anomaly detection
  - **Hyperparameter Optimization**
    - Grid Search
    - Random Search log-uniform
    - Bayesian Opt: surrogate model
    - Acquisition: EI/UCB/Thompson
    - Optuna TPE
    - Hyperband/ASHA successive halving
    - BOHB
    - Population-Based Training
    - Nested CV for HPO
  - **ML Fundamentals**
    - Bias-variance decomposition
    - Overfitting/underfitting
    - Loss vs metric
    - Epoch/batch/iteration
    - Classification vs regression
    - L1 vs L2 geometry
    - Dropout co-adaptation
    - Embeddings
    - Cross-entropy/softmax gradient
    - Cross-validation
    - Precision/recall/F1
    - Curse of dimensionality
    - Policy vs value-based RL
  - **Anomaly Detection**
    - Isolation Forest path-length score
    - Local Outlier Factor
    - One-Class SVM
    - Elliptic Envelope
    - Autoencoder reconstruction error
    - Contamination parameter pitfalls
    - Production: Precision@k, alert suppression
  - **When Classical ML Wins**
    - Tabular data dominance
    - Sample efficiency by data size
    - Interpretability: SHAP/simulatability
    - Latency-critical inference
    - Class imbalance handling
    - Native calibration
    - Feature engineering advantage
    - Decision framework
```

## Questions

### Easy

1. What impurity measure does a decision tree minimize at each split, and what are the two common choices?
2. What is the bias-variance tradeoff, and how does increasing decision tree depth move you along it?
3. Why does L1 regularization tend to produce exactly-zero coefficients while L2 does not?
4. What is the kernel trick in SVM, and why does it avoid computing an explicit high-dimensional mapping?
5. What distinguishes a core point, a border point, and a noise point in DBSCAN?
6. What does K-Means++ initialization do differently from plain random initialization, and why does it help?
7. In Gaussian Mixture Models, what happens in the E-step versus the M-step?
8. What is Out-of-Bag (OOB) error in a Random Forest, and why is it "free" validation?
9. What is the difference between bagging and boosting in terms of what each one reduces (bias vs variance)?
10. What does PCA maximize, and what matrix is eigendecomposed (or SVD'd) to find the principal components?
11. Why is a raw time series usually differenced before fitting an ARIMA model?
12. What do ACF and PACF plots help you identify when choosing ARIMA order?
13. What is the difference between Grid Search and Random Search, and why does Random Search often win in high dimensions?
14. In Bayesian Optimization, what role does the surrogate model play, and name one acquisition function.
15. How does Isolation Forest use path length to score a point as anomalous?
16. What is the intuition behind Local Outlier Factor's "local" density comparison?
17. Define epoch, batch size, and iteration, and how they relate to each other for a fixed dataset size.
18. Why is cross-entropy loss preferred over MSE for classification with softmax outputs?
19. What does the "curse of dimensionality" mean for distance-based methods like KNN?
20. Why does XGBoost/tabular gradient boosting typically outperform deep learning on small-to-medium tabular datasets?
21. What is the difference between a filter, wrapper, and embedded feature selection method (name one example of each)?
22. Why is logistic regression considered "natively calibrated" while SVM and Random Forest are not?
23. What is the role of `min_samples_leaf` / `max_depth` in controlling a single decision tree's overfitting?

### Medium

1. Random Forest and Gradient Boosting are both tree ensembles — walk through why one primarily fights variance and the other primarily fights bias, and connect this to why RF is easier to parallelize.
2. You're picking between SVM and logistic regression for a 2,000-sample, 5,000-feature text classification task. Justify your choice using sample size, dimensionality, and the need for calibrated probabilities.
3. Compare how XGBoost, LightGBM, and CatBoost each grow trees (level-wise vs leaf-wise vs symmetric) and explain the practical consequence of leaf-wise growth on small datasets.
4. K-Means assumes spherical, similarly-sized, similarly-dense clusters. Explain why GMM is a strict generalization of K-Means, and describe a data scenario where GMM succeeds and K-Means fails.
5. Explain why naive stacking (fitting the meta-learner on the same data the base models were trained on) leaks information, and how out-of-fold predictions fix this.
6. Walk through why PCA components are ordered by variance, and describe a case where the direction of maximum variance is NOT the direction most predictive of the target — what would you use instead (name the supervised alternative)?
7. t-SNE and UMAP both produce 2D embeddings for visualization — explain the "crowding problem" t-SNE solves with the Student-t distribution, and why UMAP scales better while also supporting new-point transforms.
8. You add exogenous regressors and want to forecast 10,000 SKUs at once — why would you prefer a tree-based lag-feature approach over ARIMA/SARIMA here, and what do you lose by switching?
9. Explain the mechanics of Hyperband/ASHA successive halving with a concrete bracket example (e.g., η=3) — what problem does this solve relative to plain random search with a fixed budget per trial?
10. Why does Population-Based Training (PBT) combine ideas from both evolutionary search and standard sequential training, and in what setting does it outperform Bayesian optimization?
11. Derive (conceptually) why the gradient of cross-entropy loss with softmax simplifies to (p − y), and explain why this matters practically for optimization stability.
12. A model reports 5% of transactions as anomalous using Isolation Forest with contamination=0.05 — explain precisely what this parameter controls and why blindly trusting it can silently misdiagnose a fraud problem.
13. Compare Platt scaling and isotonic regression for calibrating an SVM's decision scores — under what calibration-set size would you pick each?
14. Why is dropout described as training an implicit ensemble of 2^n subnetworks, and how does this explain why dropout reduces co-adaptation?
15. Explain why standard K-Fold cross-validation is invalid for time series and what walk-forward validation (with an added gap) fixes.
16. When would you pick Prophet over ARIMA, and when would ARIMA's explicit AR/MA structure be the better choice despite Prophet's ease of use?
17. Compare TreeSHAP and permutation importance as global feature-importance methods — which is more sensitive to correlated features and why?
18. Explain why Naive Bayes remains competitive on high-dimensional sparse text data despite its conditional-independence assumption being almost always false.
19. Describe the "when classical ML wins" decision framework: for each of tabular structure, small-n, latency <5ms, and regulatory interpretability, name the classical method you'd reach for and why deep learning underperforms there.
20. A GMM and a K-Means model are fit to the same dataset with the same k. Under what covariance structure assumption does GMM become mathematically equivalent to K-Means?
21. Explain the mechanism by which SMOTE generates synthetic minority samples, and describe a scenario where this interpolation actively hurts the decision boundary.
22. Why does using MDI (Gini-based) feature importance from a Random Forest bias results toward high-cardinality features, and what are two alternatives that avoid this bias?

### Hard

1. Derive why the L1 penalty's diamond-shaped constraint region induces sparsity while L2's spherical region does not — reason geometrically about where the loss contour is likely to first intersect each constraint boundary, and explain why this argument breaks down (or weakens) in very high dimensions with correlated features.
2. You're training an XGBoost model with early stopping on a validation set, and CV accuracy looks great, but production performance collapses after three weeks. Walk through a ranked differential diagnosis distinguishing target leakage, temporal leakage, covariate shift, and concept drift — and describe one diagnostic you'd run for each before concluding which it is.
3. An Isolation Forest anomaly detector that worked well for six months suddenly stops catching known fraud patterns. Rank, in order of likelihood, the following causes and justify the ranking: (a) the training data has become contaminated with undetected fraud over time, (b) adversarial actors have adapted to evade the model, (c) feature distributions have drifted, (d) a new fraud typology emerged outside the training distribution.
4. Explain precisely why t-SNE's cross-run/cross-perplexity instability makes it invalid as a general-purpose preprocessing step before a downstream supervised model, even though the 2D plots look informative — connect this to the fact that t-SNE has no principled out-of-sample transform.
5. Random Forest feature importance (MDI) and permutation importance can disagree sharply when two features are highly correlated. Explain the mechanism behind each failure mode (MDI's cardinality bias vs permutation importance's "protection" effect) and describe a concrete remedy for diagnosing true feature importance under strong correlation.
6. You must choose between Bayesian Optimization (Optuna/TPE) and ASHA/Hyperband for tuning a deep model where a single full training run takes 8 hours but a well-correlated proxy metric is available after 1 hour. Justify which method (or hybrid, e.g. BOHB) you'd pick and explain exactly what assumption about the loss curve's early-vs-final-ranking correlation must hold for successive halving to not prematurely kill eventual winners.
7. Derive the bias-variance decomposition E[(y − f̂(x))²] = Bias² + Variance + σ² conceptually from first principles, and use it to explain why ensembling identical high-variance base learners (bagging) reduces the variance term but does nothing to the bias term — what property of the base learner errors (correlation between them) determines how much variance reduction bagging actually achieves?
8. In stacking, why must the meta-learner be trained on out-of-fold predictions rather than in-sample predictions, and what specific failure mode occurs if you skip this — walk through what happens to a highly overfit base model's OOF vs in-sample predictions and how that distorts the meta-learner's learned weights.
9. A production tabular model needs to support both a strict <5ms p99 latency SLA and a legal requirement for exact, reproducible explanations per prediction (adverse action notices). Compare logistic regression + coefficient attribution against XGBoost + TreeSHAP on both axes, and explain why KernelSHAP would fail the latency requirement even though it would satisfy the explanation requirement.
10. Explain why Kernel PCA's O(n²)/O(n³) Gram matrix cost makes it impractical at scale, and describe what breaks (both computationally and statistically) if you naively subsample the data to fit within budget, then project the full dataset using that subsample's kernel basis.
11. Walk through the full mechanics of why UMAP can preserve more global structure than t-SNE despite both being built on similarity-graph optimization — what specifically differs in their loss functions (KL divergence vs cross-entropy on fuzzy simplicial sets) that produces this effect?
12. A time series foundation model (e.g., Chronos) is proposed to replace a portfolio of per-SKU SARIMA models for a retailer with strong promotional and holiday effects. Identify two structural risks of a zero-shot foundation model in this setting that a fitted SARIMAX with exogenous regressors would not have, and describe one hybrid approach that mitigates them.
13. Explain why calibration and discrimination (ranking ability, e.g. AUC) are orthogonal properties of a classifier — construct a scenario (conceptually) where a model has excellent AUC but terrible calibration, and explain why Platt scaling would fix calibration without changing the AUC at all.
14. In gradient boosting, explain precisely why fitting each new tree to the negative gradient of the loss (rather than the raw residual) generalizes the method beyond squared-error loss, and what changes when you swap in a Huber or quantile (pinball) loss — specifically, what is being fit at each boosting round in each case?
15. Discuss what breaks in DBSCAN when a dataset contains two true clusters of very different densities, and why the standard fix (a single global `eps`) cannot solve this — describe conceptually how HDBSCAN's hierarchy-of-thresholds approach avoids the problem.
