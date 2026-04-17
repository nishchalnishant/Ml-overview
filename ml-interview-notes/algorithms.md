# Algorithms

These are the answers you would actually say in an interview: definition first, then the mechanism, then the tradeoff.

---

# Q1: How does a Decision Tree algorithm work?

**Interview-ready answer**

A decision tree recursively splits the feature space into regions that are increasingly pure with respect to the target. At each node, it chooses the feature and threshold that most reduce impurity, such as Gini or entropy for classification and variance or MSE for regression. The final leaves store a prediction, such as the majority class or the average target value. The key strength is interpretability and the ability to model non-linear interactions without much preprocessing.

**Tradeoff to mention**

Single trees are easy to overfit because they have high variance, which is why ensembles like random forests and boosting usually outperform them.

---

# Q2: Explain how Decision Trees make splits and handle categorical features.

**Interview-ready answer**

Decision trees search for the split that gives the largest reduction in impurity. For numeric features that usually means testing thresholds. For categorical features, the tree needs a way to partition category values into child nodes. Different libraries handle this differently: some rely on one-hot encoding, while others support native category handling or ordered target statistics. The important interview point is that categorical handling is not just preprocessing; it affects both performance and leakage risk.

**Good nuance**

- High-cardinality categories can explode under one-hot encoding.
- Native categorical support, as in CatBoost, can be much better behaved.
- Missing values may be routed using learned default directions or surrogate splits.

---

# Q3: How does Random Forest work? How does it improve over Decision Trees? How does it reduce variance?

**Interview-ready answer**

Random forest trains many decision trees on bootstrap samples of the data and adds randomness in the feature selection at each split. Each individual tree is high variance, but when you average many decorrelated trees, the overall prediction becomes much more stable. That is why random forests usually outperform a single deep tree: the averaging reduces variance without increasing bias too much.

**What strong candidates add**

- Out-of-bag samples provide a built-in validation estimate.
- Random feature selection matters because averaging highly correlated trees gives less variance reduction.
- Random forests are robust tabular baselines but can become less interpretable and heavier at inference time.

---

# Q4: Explain Ensemble Methods. Why are they powerful?

**Interview-ready answer**

Ensemble methods combine multiple models so the final predictor is better than any single one. They work because different models make different errors. Bagging mainly reduces variance, boosting mainly reduces bias, and stacking learns how to combine complementary model strengths. In interviews, the best way to sound strong here is to connect ensembles to the bias-variance tradeoff rather than just naming techniques.

---

# Q5: What is the difference between bagging and boosting?

**Interview-ready answer**

Bagging trains models independently, usually on resampled versions of the data, and then averages them. That makes it mainly a variance-reduction technique. Boosting trains models sequentially, where each new learner focuses on correcting the errors of the current ensemble. That makes it mainly a bias-reduction technique. Bagging is usually more stable and easier to parallelize, while boosting is often more accurate but more sensitive to hyperparameters and noise.

---

# Q6: What is Gradient Boosting? How does XGBoost work?

**Interview-ready answer**

Gradient boosting builds an additive model stage by stage, where each new tree is fit to the negative gradient of the loss with respect to the current predictions. You can think of it as functional gradient descent in tree space. XGBoost is an optimized implementation of gradient boosting that adds regularization, efficient split finding, sparse-aware handling, missing-value support, and strong engineering for speed and scalability. That is why it became a dominant baseline for structured tabular data.

**Good nuance**

- Lower learning rate with more trees often generalizes better.
- Early stopping is usually essential.
- XGBoost is strong because it combines statistical performance with production-grade optimization.

---

# Q7: What are the key hyperparameters for XGBoost?

**Interview-ready answer**

The most important hyperparameters are learning rate, number of trees, max depth, min child weight, subsample, column subsample, and regularization terms. Learning rate and number of trees interact strongly: smaller learning rates usually require more boosting rounds. Depth and min child weight control complexity, while row and column subsampling help reduce overfitting. In practice, I tune these together with early stopping rather than in isolation.

---

# Q8: Explain Gradient Boosting and its advantages over Random Forests.

**Interview-ready answer**

Gradient boosting often outperforms random forests when the signal is complex and you are willing to tune carefully, because it corrects residual errors stage by stage instead of simply averaging independent trees. That makes it better at reducing bias. Random forests are usually more robust, easier to parallelize, and less sensitive to hyperparameters. So the practical answer is that boosting often wins on leaderboard-style accuracy, while random forests are excellent low-maintenance baselines.

---

# Q9: Explain how Logistic Regression differs from Linear Regression.

**Interview-ready answer**

Linear regression predicts a continuous value, while logistic regression models the probability of a class, usually through the log-odds. Linear regression assumes a roughly linear relationship between features and a continuous target and is typically trained with squared error. Logistic regression applies a sigmoid to a linear score and is usually trained with log loss. So despite the name, logistic regression is a classification model whose output is probabilistic rather than continuous.

---

# Q10: How does logistic regression work?

**Interview-ready answer**

Logistic regression computes a linear score `w^T x + b` and passes it through the sigmoid function to produce a probability between 0 and 1. The model is then trained by maximizing likelihood, which is equivalent to minimizing cross-entropy loss. The decision boundary is linear in feature space, but the output is non-linear in probability space. A strong interview answer should also mention that the coefficients are interpretable in terms of log-odds changes.

**Common pitfall**

Do not say logistic regression assumes the target is linear. The linearity is in the log-odds.

---

# Q11: Explain R-squared and adjusted R-squared.

**Interview-ready answer**

R-squared measures the fraction of variance in the target explained by the model relative to a mean baseline. It is useful as a summary of fit for regression, but it tends to increase as you add more predictors, even if they are not truly helpful. Adjusted R-squared corrects for that by penalizing unnecessary features. That is why adjusted R-squared is better for comparing linear models with different numbers of predictors.

**Good nuance**

Neither metric tells you whether the model is unbiased, well-calibrated, or suitable for out-of-sample prediction.

---

# Q12: How do you check for multicollinearity in regression models?

**Interview-ready answer**

I look for highly correlated predictors, unstable coefficients, inflated standard errors, and large variance inflation factors. Multicollinearity does not necessarily hurt prediction, but it makes coefficient estimates less stable and harder to interpret. If interpretability matters, I might drop redundant variables, combine them, regularize with ridge regression, or use PCA-like dimensionality reduction.

---

# Q13: How does K-Nearest Neighbors (KNN) work?

**Interview-ready answer**

KNN is a non-parametric method that predicts using the labels or values of the closest training points under a chosen distance metric. For classification, it typically uses majority vote; for regression, it averages nearby targets. Its simplicity is its main advantage, but it pushes complexity to inference time because prediction requires searching the training set. It is also very sensitive to feature scaling, irrelevant dimensions, and the choice of `k`.

---

# Q14: Explain K-Means Clustering. How does it work? Limitations?

**Interview-ready answer**

K-means partitions data into `k` clusters by alternating between assigning each point to the nearest centroid and recomputing centroids from the assigned points. It is simple, fast, and often useful as a baseline clustering method. But it assumes compact, roughly spherical clusters and depends heavily on the distance metric and initialization. It also requires `k` in advance and is sensitive to scaling and outliers.

**Good nuance**

Mention k-means++ initialization and the fact that the objective is minimizing within-cluster squared distance.

---

# Q15: Explain Support Vector Machines (SVM). What is the kernel trick?

**Interview-ready answer**

An SVM tries to find the decision boundary that maximizes the margin between classes. The idea is that a larger margin often leads to better generalization. For non-linear problems, the kernel trick lets the model operate as if the data were mapped into a higher-dimensional space without explicitly computing that mapping. This makes it possible to learn non-linear boundaries while still solving the optimization problem in terms of dot products.

**Tradeoff**

SVMs can be very strong on medium-sized structured problems, but they become expensive at large scale and are less convenient than modern boosted trees or deep models for many production settings.

---

# Q16: What is the decision boundary in classifiers?

**Interview-ready answer**

The decision boundary is the surface in feature space where the classifier changes its predicted class. In a linear model it is a hyperplane. In a non-linear model it can be much more complex. The key point is that understanding the decision boundary helps explain model capacity: simpler models draw smoother, more constrained boundaries, while complex models can fit intricate boundaries and therefore risk overfitting.

---

# Q17: Explain Naive Bayes.

**Interview-ready answer**

Naive Bayes applies Bayes' theorem and makes the simplifying assumption that features are conditionally independent given the class. That assumption is often false, but the model still works surprisingly well, especially in high-dimensional sparse settings like text classification. Its strengths are speed, simplicity, and low data requirements; its weakness is that the independence assumption can limit performance when feature interactions matter.

---

# Q18: What is Dimensionality Reduction?

**Interview-ready answer**

Dimensionality reduction means representing data using fewer variables while trying to preserve the important structure. The motivation can be better visualization, reduced noise, faster training, lower storage cost, or mitigation of the curse of dimensionality. There are two broad approaches: feature selection, which keeps some original variables, and feature extraction, which builds a lower-dimensional representation such as PCA components or learned embeddings.

---

# Q19: Explain PCA (Principal Component Analysis). How does it work? When would you use it?

**Interview-ready answer**

PCA finds orthogonal directions that capture the maximum variance in the data. Operationally, after centering the data, it computes principal components from the covariance structure or equivalently from the SVD of the data matrix. The first component explains as much variance as possible, the second explains the most remaining variance subject to orthogonality, and so on. I would use PCA when I want compression, decorrelation, denoising, or visualization, especially for numeric features with redundancy.

**Good nuance**

PCA is unsupervised, so the directions of highest variance are not always the directions most useful for prediction.

---

# Q20: Explain Gradient Descent and its variants.

**Interview-ready answer**

Gradient descent updates parameters using the negative gradient of the loss. The main variants differ in how they estimate and scale that gradient. Batch gradient descent uses the whole dataset, SGD uses one example or a mini-batch, momentum smooths updates across steps, and adaptive methods like Adagrad, RMSprop, and Adam adjust the step size per parameter. The best interview answer explains why the variants exist: to improve speed, stability, and behavior under noisy or ill-conditioned optimization.

---

# Q21: What is the ROC-AUC curve, and how is it interpreted?

**Interview-ready answer**

The ROC curve plots true positive rate against false positive rate across all classification thresholds. AUC summarizes that curve as the probability that the model ranks a randomly chosen positive example above a randomly chosen negative example. It is useful because it measures ranking quality independent of a specific threshold. But in heavily imbalanced settings, ROC-AUC can look deceptively good, so I would often pair it with PR-AUC and threshold-specific business metrics.
