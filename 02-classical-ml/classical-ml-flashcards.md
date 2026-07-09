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

## When Classical ML Wins & Interpretation

- **Simulatability?**: A human steps through the model manually. A decision tree with depth ≤ 5 satisfies this. No neural network does. #flashcard
- **Local explanations?**: Explain one prediction. Logistic regression coefficients × feature values give exact local attribution. TreeSHAP gives exact local attribution for gradient-boosted trees in milliseconds. #flashcard
- **Global feature importance?**: Which features drive the model across the dataset. MDI importance from random forests (biased toward high-cardinality features) or permutation importance and SHAP summary plots (unbiased) work directly. #flashcard
- **Explaining a single prediction to a customer?**: SHAP waterfall or LIME. #flashcard
- **Debugging model generally?**: Permutation importance + PDP. #flashcard
- **Regulatory audit (GDPR "right to explanation")?**: SHAP. #flashcard
- **Real-time feature importance in serving?**: pre-computed SHAP on representative samples. #flashcard
- **TreeSHAP?**: exact, fast (milliseconds per sample), tree models only. #flashcard
- **KernelSHAP?**: approximate, slow (seconds per sample), any model. #flashcard
- **Shapley value subset $S$, $v(S)$?**: model prediction using only features in $S$; the combinatorial fraction is the probability that $S$ appears in a random feature ordering. #flashcard
- **LIME limitations?**: not additive (subset explanations don't combine predictably), sensitive to hyperparameters (kernel width σ, n_samples), local linearity assumption may not hold, different runs can give different explanations. #flashcard
- **Correlated features and permutation importance?**: when two features carry the same info, shuffling one doesn't hurt because the other still provides it — both appear low importance. Check by computing importance on the residuals after removing the correlated feature. #flashcard
- **Feature importance measures predictive importance, not causal importance?**: correct — don't over-interpret it causally. #flashcard
- **Attention weights as explanations?**: intermediate computations, not causal attributions; the same attention pattern can be produced with different weight matrices. Gradient-based methods (Integrated Gradients, SHAP for transformers) are more principled. #flashcard

## Hyperparameter Optimization

- **Expected Improvement (EI)?**: Expected gain over the current best, under the surrogate's uncertainty. Standard choice. #flashcard
- **Upper Confidence Bound (UCB)?**: $\mu(\lambda) + \kappa \cdot \sigma(\lambda)$. Optimistic under uncertainty. $\kappa$ controls exploration. #flashcard
- **Thompson Sampling?**: Sample a function from the surrogate's posterior, maximize it. Simple and parallelizable. #flashcard
- **Grid and random search are trivially parallel?**: no communication needed between workers. #flashcard
- **Bayesian optimization has a sequential bottleneck?**: each new configuration depends on all previous results. Use asynchronous variants or batch acquisition. #flashcard

## Ensemble Methods

- **Bagging vs boosting vs stacking?**: Bagging reduces variance (averages out noise, parallel); boosting reduces bias (focuses on hard examples, sequential); stacking can reduce both by learning how to combine models optimally. #flashcard
- **voting='hard' vs 'soft'?**: hard = majority class vote; soft = average probabilities (requires calibrated models, almost always better). #flashcard
- **Random Forest vs bagging?**: Random Forest adds random feature selection on top of bagging — key to diversity. #flashcard
- **OOB score?**: provides free validation without a held-out set. #flashcard
- **Stacking must use OOF predictions?**: fitting on base model predictions trained on the same data leaks labels. #flashcard
- **Meta-learner in stacking?**: usually kept simple (logistic regression) — a complex meta-learner overfits. #flashcard
- **In practice?**: XGBoost/LightGBM alone often matches stacking on tabular data. Stacking matters most in competitions. #flashcard

## Dimensionality Reduction

- **When to use PCA?**: preprocessing before algorithms sensitive to dimensionality (k-NN, SVM, k-means), consolidating collinear features, visualization (first 2-3 PCs), compression. Avoid when features are categorical, non-linear structure dominates, or components must stay interpretable. #flashcard
- **Kernel PCA — RBF**: $k(x, y) = \exp(-\gamma \|x - y\|^2)$, captures local non-linear structure; $\gamma$ controls scale. #flashcard
- **Kernel PCA — Polynomial**: $k(x, y) = (x \cdot y + c)^p$, captures interactions up to degree $p$. #flashcard
- **Kernel PCA — Sigmoid**: $k(x, y) = \tanh(\alpha x \cdot y + c)$. #flashcard
- **t-SNE perplexity, low (5-15) vs high (30-50)?**: low = very local structure, may fragment clusters; high = broader, more global-like neighborhoods. Typical default: 30-50. #flashcard
- **t-SNE pitfalls?**: cluster distance ≠ semantic distance; cluster size ≠ cluster density; always check multiple seeds/perplexities; no out-of-sample extension, so don't use for downstream ML tasks. #flashcard
- **UMAP n_neighbors, small (2-5) vs large (50-200)?**: small = very local, may fragment global structure; large = global structure preserved, local detail lost. #flashcard
- **UMAP min_dist, small (0.0-0.1) vs large (0.5-1.0)?**: small = tight clusters, good for visualization; large = loose, spread-out embedding, good for exploring global topology. #flashcard

## Imbalanced Data (Focal Loss)

- **Focal loss $\gamma = 0$?**: reduces to weighted cross-entropy. #flashcard
- **Focal loss $\gamma = 2$ (typical)?**: easy examples downweighted by ~100x. #flashcard

## Evaluation Metrics

- **Metric choice by task**: Classification → F1/PR-AUC/MCC (accuracy fails on imbalance), ROC-AUC is threshold-independent but optimistic when positives are rare; Calibration → ECE/Brier; Regression → MAE (robust to outliers) vs MSE (penalizes large errors) vs MAPE (breaks near zero) vs R² (not enough alone); Ranking → NDCG (graded relevance + position discount), MAP/MRR (binary relevance); Generative → human win rate is ground truth, BLEU/FID are gameable proxies. #flashcard
- **Offline vs online metrics?**: Offline metrics are proxies; proxy-outcome correlation drifts. Always set guardrail metrics before launching experiments. #flashcard
- **Statistical testing for model comparison?**: Always run a significance test — McNemar for classifiers, paired t-test or Wilcoxon for regression. Correct for multiple comparisons. #flashcard
- **MCC (Matthews Correlation Coefficient)**: range $[-1, 1]$, uses all four confusion-matrix cells, works for any class balance. Harder to explain to stakeholders than F1 but more informative. A model predicting all negatives on a 99/1 split gets MCC = 0, not 0.99. #flashcard
- **Brier score**: range $[0, 1]$, lower is better. Decomposes as $\text{Brier} = \text{Calibration} + \text{Resolution} - \text{Uncertainty}$ (Murphy decomposition). A proper scoring rule, maximized in expectation only when $\hat{p}_i = P(y_i=1)$. Skill score: $\text{BSS} = 1 - \text{Brier}/\text{Brier}_{\text{ref}}$. #flashcard
- **MAE vs MSE vs Huber**: MAE = median regression, outliers shouldn't dominate; MSE = mean regression, outliers are real signal; Huber = production systems with occasional bad labels/anomalous inputs. #flashcard
- **$R^2$**: fraction of variance explained; 1 = perfect, 0 = matches constant baseline, can be negative. Adding features always increases $R^2$ — adjusted $R^2$ penalizes for number of predictors $p$. High $R^2$ doesn't guarantee a good model (Anscombe's quartet: identical $R^2$, wildly different distributions). #flashcard
- **Pinball loss**: minimizing it at quantile $\tau$ produces the $\tau$-th conditional quantile; $\tau=0.5$ recovers MAE. Use for prediction intervals (fit $\tau=0.1$ and $\tau=0.9$ for an 80% interval). A proper scoring rule for quantiles, unlike symmetric losses. #flashcard
- **NDCG**: range $[0,1]$, 1 = perfect ordering. Undefined if no relevant documents exist (IDCG = 0) — handle by treating as 0 or excluding. #flashcard
- **MAP (ranking)**: AP rewards finding relevant documents early; MAP averages AP across queries. Sensitive to missing relevant items, unlike NDCG with fixed $k$. #flashcard
- **MRR**: only cares about the first relevant hit (rank of first relevant result), ignores subsequent ones. Use when there's one "right" answer (QA, spelling correction, entity lookup). #flashcard

## Calibration & Uncertainty

- **Reliability diagram, above vs below diagonal?**: above = underconfident (true positive rate exceeds predicted probability, model is more right than it thinks); below = overconfident (model is more wrong than it thinks). An S-shaped curve (above for low $p$, below for high $p$) is the classic random forest pattern. #flashcard
- **Calibration binning strategy**: `uniform` = equal-width bins (can be sparse at extremes); `quantile` = equal-count bins (better for skewed score distributions). #flashcard
- **Temperature scaling $T$**: $T>1$ softens the distribution (less confident); $T<1$ sharpens it (more confident, rarely needed); $T=1$ unchanged. #flashcard
- **Aleatoric uncertainty?**: Variance of the label given the input, $\text{Var}[Y|X=x]$. From noise in the data generating process — cannot be reduced by collecting more data. #flashcard
- **Epistemic uncertainty?**: Uncertainty about the model parameters given finite training data. Decreases with more data in the uncertain region. #flashcard

## Time Series (Classical & Foundation Models)

- **Differencing (d=1)?**: $\Delta X_t = X_t - X_{t-1}$; apply ARMA(2,1) to $\Delta X_t$ for an ARIMA(2,1,1)-style model. #flashcard
- **Prophet decomposition**: $g(t)$ = trend (piecewise linear or logistic growth), $s(t)$ = seasonality (Fourier series), $h(t)$ = holiday effects, $\epsilon_t$ = Gaussian noise. #flashcard
- **PatchTST**: patch size 16 with no overlap, channel independence (each variate processed separately with shared weights), linear complexity in sequence length for long-horizon forecasting. #flashcard
- **Chronos**: tokenizes time series values (quantization into bins), uses T5 language model architecture, trained on 100K+ time series datasets for zero-shot forecasting without task-specific fine-tuning. #flashcard
- **TimeGPT**: universal forecasting model, any frequency/any prediction length, patch-based tokenization (like PatchTST), handles multivariate series with mixed frequencies. #flashcard
- **TimesFM**: 200M parameter foundation model, trained on 100B time points (Google Trends, Wikipedia, etc.), competitive zero-shot performance on standard benchmarks. #flashcard
