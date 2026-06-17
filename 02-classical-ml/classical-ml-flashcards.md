---
module: Classical ML
topic: Flashcards
status: unread
tags: [classical-ml, flashcards, active-recall]
---
# Classical ML Flashcards

Use these for active recall and spaced repetition (e.g., Anki, Obsidian).

## Supervised Learning & Algorithm Selection

- **Gini Importance?**: Total reduction of Gini impurity provided by a feature across all trees. Fast but biased toward high-cardinality features. #flashcard
- **Permutation Importance?**: Model score drop when a feature's values are randomly shuffled. Slower but more reliable and model-agnostic. #flashcard
- **SHAP values?**: Game-theoretically fair attribution. Best choice when explanation quality matters. #flashcard
- **Default choice: CSV, unknown shape, 1 hour?**: Logistic/linear baseline → gradient boosting with CV. #flashcard
- **Default choice: Text, fast inference?**: Naive Bayes or logistic with TF-IDF. #flashcard
- **Default choice: High-dim, small N, clear margin?**: SVM. #flashcard
- **Default choice: Interpretability required?**: Single decision tree or logistic regression. #flashcard
- **Default choice: Best accuracy on tabular?**: XGBoost / LightGBM. #flashcard

## Unsupervised Learning & Clustering

- **DBSCAN: ε (eps)?**: Neighborhood radius. #flashcard
- **DBSCAN: MinPts (min_samples)?**: Minimum points to form a dense region. #flashcard
- **DBSCAN: Core point?**: A point with at least MinPts points within ε. #flashcard
- **DBSCAN: Border point?**: Within ε of a core point but fewer than MinPts neighbors. #flashcard
- **DBSCAN: Noise?**: Neither core nor border. #flashcard
- **GMM: E-step?**: Compute soft assignments $r_{ik} = P(z_k | x_i)$. #flashcard
- **GMM: M-step?**: Update $\mu_k$, $\Sigma_k$, $\pi_k$ using weighted MLE. #flashcard
- **K-Means: Elbow method?**: Plot inertia (WCSS) vs K → pick the "elbow" (diminishing returns). #flashcard
- **K-Means++ initialization**: Spreads centroids during initialization to reduce bad local minima. #flashcard
- **Silhouette Coefficient?**: Range $[-1, 1]$. Measures how much closer each point is to its own cluster than to the nearest other cluster. High = dense, well-separated clusters. #flashcard
- **Calinski-Harabasz Index?**: Ratio of between-cluster to within-cluster dispersion. Higher is better. #flashcard
- **Davies-Bouldin Index?**: Average similarity between each cluster and its most similar one. Lower is better. #flashcard
- **Isolation Forest?**: Randomly isolates points; anomalies = short path lengths; works well in high-D. #flashcard

## Data Preprocessing & Leakage

- **Target leakage?**: A feature encodes the answer. Example: `loan_approved = 1` as a feature when predicting default. #flashcard
- **Train/test contamination?**: Fitting a scaler or encoder on the full dataset before splitting. #flashcard
- **Temporal leakage?**: Using a feature computed from events that happened after the prediction timestamp. #flashcard
- **MCAR (Missing Completely At Random)?**: Missingness is independent of any variable. Dropping is safe. #flashcard
- **MAR (Missing At Random)?**: Missingness depends on other observed variables, not the missing value itself. Conditional imputation is appropriate. #flashcard
- **MNAR (Missing Not At Random)**: Missingness depends on the value that is missing. The missingness is itself a signal. #flashcard
- **Covariate shift**: $P(X)$ changes. The input distribution shifts, but the relationship $P(y|X)$ may be stable. #flashcard
- **Label shift**: $P(y)$ changes. The prior probability of each class changes. #flashcard
- **Concept drift**: $P(y|X)$ changes. The underlying relationship the model learned no longer holds. #flashcard
- **Population Stability Index (PSI) thresholds?**: < 0.1 = stable. 0.1–0.25 = moderate shift. > 0.25 = major shift. #flashcard
- **Z-score (standardization)**: Zero mean, unit variance. Best for distance-based algorithms (SVM, PCA, Logistic Regression). #flashcard
- **Min–max [0,1] normalization**: Bounded range. Best for neural nets. #flashcard
- **Winsorize / cap**: Dulls extremes by capping at IQR fences. Keeps sample size, often better than blind deletion. #flashcard
