# Classical ML — Rapid Comparison Cheat Sheet

Exhaustive, comparison-first review of every model/algorithm/technique in this folder. Scan a section, compare rows, then check the per-entry pick-criteria before an interview answer.

---

## Regression

### Linear Regression
- **What it is**: Fits $y = w^Tx + b$ minimizing squared error; closed-form (normal equation) or gradient descent.
- **Pros**: Fast to train/predict; fully interpretable coefficients; well-calibrated residual assumptions enable confidence intervals; no hyperparameters to tune in the base form.
- **Cons**: Assumes linear relationship and homoscedastic, independent errors; sensitive to outliers and multicollinearity; underfits non-linear signal.
- **Pick over alternatives when**: Relationship is genuinely linear, you need interpretable coefficients, or you need a fast interpretable baseline before trying trees/boosting.
- **Key hyperparameters**: none in OLS form; add `alpha` if regularized (see Regularization section).

### Logistic Regression
- **What it is**: Linear model for classification; applies sigmoid to $w^Tx+b$ and optimizes log-loss (a proper scoring rule).
- **Pros**: Natively calibrated probabilities (no post-hoc calibration needed); coefficients are exact log-odds — fully interpretable; fast at both train and inference (microseconds); trivial to regularize.
- **Cons**: Linear decision boundary only (no interaction terms unless engineered); degrades if log-odds aren't linear in features; sensitive to multicollinearity.
- **Pick over alternatives when**: Need calibrated probabilities out of the box, need a legally defensible/simulatable model (GDPR/FCRA), n < 1,000, or latency SLA < 5ms.
- **Key hyperparameters**: `C` (inverse regularization strength), `penalty` (l1/l2/elasticnet), `class_weight`.

---

## SVM (Support Vector Machines)

- **What it is**: Finds the maximum-margin hyperplane separating classes; kernel trick projects into higher-dimensional space for non-linear boundaries.
- **Pros**: Effective in high-dimensional spaces; kernel flexibility (linear/poly/RBF) captures non-linear boundaries; robust to overfitting when margin is well regularized (especially with few features relative to samples).
- **Cons**: Doesn't scale well beyond ~10⁴–10⁵ samples (kernel matrix is O(n²)); no native probability output — raw scores need Platt scaling; sensitive to feature scaling; hyperparameter tuning (C, gamma) is finicky.
- **Pick over alternatives when**: Small-to-medium, high-dimensional data (e.g., text with TF-IDF, genomics) where clear margin separation exists; when a non-linear boundary is needed but tree-based non-linearity is undesirable.
- **Key hyperparameters**: `C` (margin softness vs regularization), `kernel` (linear/rbf/poly), `gamma` (RBF kernel width — controls locality).

| Kernel | Best for | Avoid when |
|---|---|---|
| Linear | High-dim sparse data (text) | Non-linear boundary needed |
| RBF | Generic non-linear boundary, moderate dims | Very high dims / very large n |
| Poly | Explicit interaction/degree structure known | Degree > 3 (numerically unstable) |

---

## Tree-Based Models & Ensembles

### Decision Tree
- **What it is**: Recursively splits feature space on the split that most reduces impurity (Gini or entropy for classification, variance for regression).
- **Pros**: Fully interpretable at shallow depth (simulatable — a human can trace the rules); no feature scaling needed; captures non-linear interactions natively.
- **Cons**: High variance — small data changes produce very different trees; prone to overfitting if not pruned/depth-limited; axis-aligned splits only (struggles with diagonal boundaries).
- **Pick over alternatives when**: You need a fully simulatable rule-based model (depth ≤ 5) for regulatory/explanatory purposes, or as a quick baseline before ensembling.
- **Key hyperparameters**: `max_depth`, `min_samples_leaf`, `criterion` (gini/entropy).

### Random Forest (Bagging)
- **What it is**: Bags many deep decision trees on bootstrap samples with random feature subsets per split; averages (regression) or votes (classification).
- **Pros**: Reduces variance of individual trees dramatically; robust to overfitting; parallelizable (trees are independent); built-in OOB validation; minimal tuning needed to get decent performance; handles missing/mixed data well.
- **Cons**: Loses full interpretability of a single tree; MDI feature importance is biased toward high-cardinality features (use permutation importance/SHAP instead); larger memory footprint; probabilities need calibration (S-shaped miscalibration from vote averaging).
- **Pick over XGBoost/GBM when**: You want a fast, low-tuning-effort strong baseline; the data is noisy and you want variance reduction over bias reduction; training must be parallelized trivially; you need OOB estimates without a separate validation split.
- **Key hyperparameters**: `n_estimators`, `max_depth`/`max_features`, `min_samples_leaf`.

### AdaBoost
- **What it is**: Sequentially trains weak learners (stumps), reweighting misclassified samples each round; final prediction is a weighted vote using each model's error-derived weight.
- **Pros**: Simple, few hyperparameters; strong bias reduction; theoretically well-grounded exponential-loss framework.
- **Cons**: Sensitive to noisy data and outliers (they get reweighted up repeatedly); generally outperformed by gradient boosting in practice; sequential — can't parallelize across estimators.
- **Pick over Gradient Boosting when**: Rarely in practice today — mostly of historical/interview-theory interest; simple weak-learner boosting demo, low-noise data.
- **Key hyperparameters**: `n_estimators`, `learning_rate`.

### Gradient Boosting (generic GBM)
- **What it is**: Sequentially fits new trees to the negative gradient (pseudo-residuals) of the loss from previous trees; additive model built stage-wise.
- **Pros**: Strong bias reduction, typically best-in-class accuracy on tabular data; flexible loss functions.
- **Cons**: Sequential training (slower than RF, less parallelizable); more hyperparameters to tune; more prone to overfitting without careful `learning_rate`/early stopping; sensitive to outliers in gradient-based loss.
- **Pick over Random Forest when**: You have time/compute to tune, need to squeeze out maximum accuracy, and can properly validate to avoid overfitting.
- **Key hyperparameters**: `n_estimators`, `learning_rate`, `max_depth`.

### XGBoost / LightGBM / CatBoost
- **What it is**: Production-grade, regularized, highly optimized gradient boosting implementations; differ in tree-growth strategy and categorical handling.
- **Pros**: State-of-the-art tabular accuracy (dominates 90% of tabular benchmarks over deep learning per Grinsztajn et al. 2022); built-in regularization (L1/L2 on leaf weights); TreeSHAP gives exact, fast explanations; `scale_pos_weight` handles imbalance natively; compiles to native code (treelite) for microsecond-to-low-millisecond inference.
- **Cons**: More hyperparameters than RF; can overfit without early stopping/regularization tuning; categorical features need encoding in XGBoost (native in LightGBM/CatBoost); less interpretable than a single tree or linear model without SHAP.
- **Pick XGBoost when**: Need mature ecosystem, GPU support, fine-grained regularization control.
- **Pick LightGBM when**: Very large datasets, need fastest training via leaf-wise growth + histogram binning.
- **Pick CatBoost when**: Many high-cardinality categorical features, want minimal preprocessing (ordered target statistics avoid target leakage).
- **Key hyperparameters**: `learning_rate`, `max_depth`/`num_leaves`, `n_estimators` + early stopping, `subsample`/`colsample_bytree`, `reg_alpha`/`reg_lambda`.

**Comparison table — Tree-based/Ensembles**

| Algorithm | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| Decision Tree | Simulatable rules, quick baseline | Need best accuracy | Very low | Very high (shallow) |
| Random Forest | Low-tuning strong baseline, parallel training | Need absolute best accuracy, need small model size | Medium (parallel) | Medium (needs importance/SHAP) |
| AdaBoost | Simple boosting demo, low-noise data | Noisy/outlier-heavy data | Medium (sequential) | Low-medium |
| Gradient Boosting | High accuracy, flexible loss | Limited tuning time | High (sequential) | Low (needs SHAP) |
| XGBoost/LightGBM/CatBoost | Best tabular accuracy, imbalance, categorical data | Need instant-parallel training or ultra-low latency vs LR | High (sequential, but optimized) | Low (needs TreeSHAP) |

### Ensemble Meta-Techniques (combining any of the above)
- **Bagging** (e.g., Random Forest): trains models on bootstrap resamples in parallel, averages — reduces **variance**. Pick when base learner is high-variance/low-bias (deep trees).
- **Boosting** (AdaBoost/GBM/XGBoost): trains sequentially on residuals/reweighted errors — reduces **bias**. Pick when base learner is high-bias/low-variance (shallow trees/stumps).
- **Voting Ensemble**: combines predictions of several *different* model types (hard = majority vote, soft = averaged probabilities). Pick when you have a few diverse strong models and want a quick accuracy bump with no extra training complexity.
- **Stacking**: trains a meta-learner on out-of-fold (OOF) predictions of base models (via `cross_val_predict` to avoid leakage). Pick when base models are diverse and you can afford the extra meta-model training/complexity for maximum accuracy (e.g., competitions).
- **Blending**: like stacking but meta-learner trained on a single held-out split instead of OOF — simpler, less leakage-safe, less data-efficient. Pick when you want a faster/simpler version of stacking and can spare a holdout set.
- **Snapshot Ensembles**: save model checkpoints at different points of a cyclical-LR training run, ensemble those snapshots — no extra training cost. Pick when training a single deep model is expensive and you want ensemble diversity for free.
- **Multi-Seed Ensemble**: same model/config, multiple random seeds, average. Pick when you want variance reduction from randomness alone with minimal implementation complexity.
- **Calibration Before Ensembling**: apply `CalibratedClassifierCV` per base model before combining probabilities — necessary whenever base models have different, unmatched miscalibration patterns (e.g., averaging SVM + RF probabilities directly is meaningless without calibrating both first).

---

## Instance-Based & Probabilistic Models

### K-Nearest Neighbors (KNN)
- **What it is**: Non-parametric; predicts by majority vote/average of the k closest training points under a distance metric.
- **Pros**: No training phase (lazy learner); naturally captures complex non-linear boundaries; simple, intuitive; easily supports weighted voting by distance.
- **Cons**: Curse of dimensionality — distance becomes meaningless in high dims; O(n) inference cost per query (slow at scale without approximate search structures); sensitive to feature scaling and irrelevant features; memory-heavy (stores full training set).
- **Pick over parametric models when**: Decision boundary is highly irregular/local, dataset is small-to-medium, dimensionality is low, and inference latency isn't critical.
- **Key hyperparameters**: `k` (bias-variance tradeoff — small k = low bias/high variance), `distance metric` (Euclidean/Manhattan/Minkowski), `weights` (uniform vs distance-weighted).

### Naive Bayes (Gaussian / Multinomial / Bernoulli)
- **What it is**: Applies Bayes' theorem assuming conditional feature independence given the class; variant chosen by feature distribution assumption (Gaussian for continuous, Multinomial for counts, Bernoulli for binary).
- **Pros**: Extremely fast to train (closed-form counting/statistics, no iterative optimization); works well with very high-dimensional sparse data (text/spam); needs little training data; naturally probabilistic.
- **Cons**: Independence assumption is almost always violated in practice, hurting probability calibration (though ranking/classification often still works); Multinomial NB needs count/frequency features, not raw continuous data.
- **Pick over Logistic Regression when**: Very high-dimensional sparse text data, need extremely fast training/inference, or as a quick baseline before heavier models.
- **Key hyperparameters**: `alpha` (Laplace/Lidstone smoothing for Multinomial/Bernoulli).

**Comparison table — Instance-based / Probabilistic**

| Algorithm | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| KNN | Small/medium data, irregular local boundary | High-dim data, low-latency serving | None (lazy) | Medium (case-based) |
| Naive Bayes | High-dim sparse text/counts, fast baseline | Strong feature correlations, need calibrated probs | Very low | High (per-feature likelihoods) |

---

## Regularization (Linear/Logistic Models)

### L2 (Ridge)
- **What it is**: Adds $\lambda\sum w_j^2$ penalty; shrinks all coefficients toward zero but never exactly to zero.
- **Pros**: Handles multicollinearity well (distributes weight across correlated features); has closed-form solution; stable/low variance.
- **Cons**: Doesn't perform feature selection — keeps all features nonzero, hurting interpretability with many irrelevant features.
- **Pick over L1 when**: Features are correlated and you want to keep/shrink all of them together (grouped shrinkage) rather than arbitrarily picking one.

### L1 (Lasso)
- **What it is**: Adds $\lambda\sum|w_j|$ penalty; geometry of the constraint region drives many coefficients to exactly zero — performs embedded feature selection.
- **Pros**: Automatic feature selection; produces sparse, more interpretable models.
- **Cons**: Arbitrarily picks one feature from a correlated group (unstable selection across resamples/seeds); underselects when the true signal is spread across many small-effect features (assumes sparsity).
- **Pick over Ridge when**: You want automatic feature selection / sparse models and suspect only a subset of features truly matter.

### Elastic Net
- **What it is**: Convex combination of L1 and L2 penalties.
- **Pros**: Selects features like Lasso but handles correlated-feature groups better (selects them together rather than arbitrarily picking one).
- **Cons**: Two hyperparameters to tune (`alpha` for penalty mix, `lambda`/`C` for strength) instead of one.
- **Pick over pure Lasso/Ridge when**: Many correlated features exist AND you still want sparsity — best of both worlds.

**Comparison table — Regularization**

| Method | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| Ridge (L2) | Multicollinearity, keep all features | Need feature selection | Low (closed-form) | Medium (all coefs nonzero) |
| Lasso (L1) | Sparse model, feature selection | Highly correlated features (unstable picks) | Low | High (sparse) |
| Elastic Net | Correlated features + need sparsity | Simplicity/fewer hyperparameters preferred | Low-medium | Medium-high |

---

## Clustering

### K-Means (+ K-Means++)
- **What it is**: Partitions data into k clusters by iteratively assigning points to nearest centroid and recomputing centroids (Lloyd's algorithm); K-Means++ smartly seeds initial centroids to speed convergence and avoid bad local optima.
- **Pros**: Fast, scales to large n (linear per iteration); simple to understand and implement; works well when clusters are roughly spherical/convex and similar size.
- **Cons**: Must specify k in advance; assumes spherical, equally-sized clusters — fails on elongated/irregular shapes; sensitive to initialization (mitigated by K-Means++) and outliers; only handles numeric features natively.
- **Pick over DBSCAN when**: Clusters are roughly spherical/convex, you know (or can estimate via elbow/silhouette) the number of clusters, and speed on large n matters.
- **Key hyperparameters**: `k` (number of clusters — choose via elbow method or silhouette score), `init` (k-means++ vs random), `n_init`.

### DBSCAN
- **What it is**: Density-based clustering; groups points with enough neighbors within `eps` (core points), attaches border points, labels sparse points as noise/outliers.
- **Pros**: Finds arbitrarily shaped clusters; automatically detects number of clusters; naturally identifies outliers/noise as a byproduct.
- **Cons**: Struggles with varying-density clusters (single global `eps` doesn't fit all regions); sensitive to `eps`/`min_samples` choice; doesn't scale as well as K-Means on very large n without spatial indexing.
- **Pick over K-Means when**: Clusters have irregular/non-convex shapes, the number of clusters is unknown, or you need explicit outlier/noise detection as part of clustering.
- **Key hyperparameters**: `eps` (neighborhood radius), `min_samples` (density threshold for core points).

### Gaussian Mixture Model (GMM)
- **What it is**: Probabilistic clustering assuming data is generated from a mixture of k Gaussians; fit via Expectation-Maximization (EM), assigning soft (probabilistic) cluster membership.
- **Pros**: Soft assignments give uncertainty per point; can model elliptical (not just spherical) clusters via full covariance matrices; principled probabilistic framework (can compute likelihoods, use BIC/AIC for model selection).
- **Cons**: Assumes Gaussian-shaped clusters — fails badly on non-Gaussian structure; EM can converge to local optima depending on init; more parameters to estimate than K-Means (covariance matrices), slower.
- **Pick over K-Means when**: You need soft/probabilistic cluster assignments, clusters are elliptical rather than spherical, or you want a generative/likelihood-based model for anomaly scoring.
- **Key hyperparameters**: `n_components` (k), `covariance_type` (full/tied/diag/spherical).

### Hierarchical (Agglomerative) Clustering
- **What it is**: Builds a tree (dendrogram) of nested clusters by iteratively merging the closest pair of clusters/points, using a linkage criterion.
- **Pros**: No need to pre-specify k (cut dendrogram at desired level); dendrogram gives full hierarchy of cluster relationships, useful for exploratory analysis; deterministic (no random init).
- **Cons**: O(n²) or worse — doesn't scale to large n; once a merge happens it can't be undone (greedy); choice of linkage strongly affects results.
- **Pick over K-Means/DBSCAN when**: You want the full cluster hierarchy/dendrogram for exploratory analysis, dataset is small-to-medium, or the natural number of clusters is genuinely unclear and you want to inspect multiple cuts.
- **Key hyperparameters**: `linkage` (Single = chains/elongated clusters, sensitive to noise; Complete = compact, tighter clusters; Average = balance between the two; Ward's = minimizes within-cluster variance, most similar to K-Means), `n_clusters` or `distance_threshold`.

**Comparison table — Clustering**

| Algorithm | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| K-Means | Spherical clusters, large n, speed | Irregular shapes, unknown k, outliers | Low (linear/iter) | Medium (centroids) |
| DBSCAN | Irregular shapes, outlier detection, unknown k | Varying density clusters, very large n | Medium | Medium |
| GMM | Soft assignments, elliptical clusters, probabilistic scoring | Non-Gaussian structure | Medium (EM iterations) | Medium (probabilistic) |
| Hierarchical | Full hierarchy/dendrogram, exploratory, small n | Large n (O(n²)+) | High (O(n²)) | High (dendrogram) |

### Clustering Evaluation
- **Without ground truth (internal)**: Silhouette Score (cohesion vs separation, range -1 to 1), Calinski-Harabasz (between/within cluster variance ratio, higher better), Davies-Bouldin (avg similarity between clusters, lower better).
- **With ground truth (external)**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) — compare cluster assignments to true labels.

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
- **What it is**: Linear, unsupervised projection onto orthogonal directions of maximum variance, computed via eigendecomposition of the covariance matrix (or SVD directly on data).
- **Pros**: Fast, deterministic, closed-form (SVD); preserves global variance structure; decorrelates features; whitening option normalizes component variance; highly scalable.
- **Cons**: Only captures linear structure; components are linear combinations — not interpretable as original features; sensitive to feature scaling (must standardize first); assumes high variance = high importance, which isn't always true.
- **Pick over Kernel PCA/t-SNE/UMAP when**: You need a fast, linear, easily-invertible reduction for preprocessing (e.g., before clustering/regression), or need exact reconstruction error control.
- **Key hyperparameters**: `n_components`, `whiten` (bool).

### Kernel PCA
- **What it is**: Applies the kernel trick to PCA — implicitly projects data into a higher-dimensional space via a kernel function, then performs linear PCA there, capturing non-linear structure.
- **Pros**: Captures non-linear manifold structure that linear PCA misses.
- **Cons**: O(n²) or O(n³) kernel matrix computation — doesn't scale well; choice of kernel and its hyperparameters adds tuning burden; harder to interpret than PCA.
- **Pick over PCA when**: Data has known non-linear structure (e.g., concentric circles/manifolds) that a linear projection can't separate.
- **Key hyperparameters**: `kernel` (rbf/poly), `gamma`.

### ICA (Independent Component Analysis / FastICA)
- **What it is**: Unmixes signals into statistically independent (not just uncorrelated) components — classic use: blind source separation (e.g., cocktail party problem).
- **Pros**: Finds statistically independent, non-Gaussian sources — useful when the true generative signals are additive mixtures (audio, EEG/artifact removal).
- **Cons**: Assumes non-Gaussian independent sources (fails if sources are Gaussian); components have no inherent ordering (unlike PCA's variance-ranked components); sensitive to preprocessing/whitening.
- **Pick over PCA when**: The goal is unmixing statistically independent source signals rather than maximizing variance capture (e.g., separating audio sources, artifact removal in signals).
- **Key hyperparameters**: `n_components`, `algorithm` (parallel/deflation).

### LDA (Linear Discriminant Analysis)
- **What it is**: Supervised linear projection maximizing between-class variance relative to within-class variance (Fisher's criterion) — up to `n_classes - 1` components.
- **Pros**: Uses label information, so projection is explicitly optimized for class separability; also usable directly as a classifier.
- **Cons**: Assumes classes are Gaussian with shared covariance (violated → use QDA or non-linear method); limited to `n_classes - 1` dimensions; linear boundary only.
- **Pick over PCA when**: You have labels and the goal is maximizing class separability for a downstream classification task, not just unsupervised variance capture.
- **Key hyperparameters**: `n_components` (≤ n_classes - 1), `solver`.

### t-SNE
- **What it is**: Non-linear, non-parametric embedding for visualization; preserves local neighborhood structure by matching pairwise similarity distributions in high-D and low-D space (KL divergence minimization).
- **Pros**: Excellent for visualizing high-dimensional clusters in 2D/3D; captures local non-linear structure very well.
- **Cons**: Does not preserve global structure/distances (cluster sizes and inter-cluster distances are not meaningful); "crowding problem" needs the heavy-tailed t-distribution to fix; computationally expensive (no efficient out-of-sample transform); results vary significantly with `perplexity` and random seed; should not be used as a general-purpose preprocessing step before another model.
- **Pick over UMAP when**: Established, well-understood tool is preferred; primarily for exploratory visualization where compute cost is not a concern (rarely picked over UMAP today given UMAP's speed advantage).
- **Key hyperparameters**: `perplexity` (effective neighborhood size, typically 5-50), `n_iter`, `learning_rate`.

### UMAP
- **What it is**: Non-linear embedding based on manifold learning / fuzzy topological structure; faster and better global-structure preservation than t-SNE.
- **Pros**: Much faster than t-SNE, scales to larger datasets; better preserves some global structure; supports transforming new/unseen points (unlike vanilla t-SNE); can be used as general-purpose preprocessing (not just visualization).
- **Cons**: Still primarily a visualization/heuristic tool — inter-cluster distances aren't fully reliable; sensitive to `n_neighbors`/`min_dist` choices; less mathematically "clean" theoretical guarantees than PCA.
- **Pick over t-SNE when**: You need speed on larger datasets, need to embed new/unseen points after fitting, or want a bit more global structure preserved.
- **Key hyperparameters**: `n_neighbors` (local vs global structure tradeoff), `min_dist` (how tightly points can be packed), `n_components`.

### NMF (Non-negative Matrix Factorization)
- **What it is**: Factorizes a non-negative matrix into two non-negative low-rank matrices via multiplicative update rules; components are additive, not subtractive.
- **Pros**: Produces interpretable, additive parts-based components (e.g., topics in text, parts in images) since no negative values cancel each other out.
- **Cons**: Requires non-negative input data; non-convex optimization (local optima, initialization-sensitive); no unique solution (rotation ambiguity).
- **Pick over PCA when**: Data is inherently non-negative (word counts, pixel intensities, ratings) and you want additive, parts-based interpretable components (e.g., topic modeling).
- **Key hyperparameters**: `n_components`, `init` (nndsvd/random), `solver`.

### Autoencoders / VAE
- **What it is**: Neural network trained to reconstruct its input through a bottleneck layer; VAE additionally imposes a probabilistic latent structure (regularized to match a prior, e.g., standard normal).
- **Pros**: Learns arbitrary non-linear reductions (more expressive than PCA); VAE gives a smooth, generative, samplable latent space; scales to very large/complex data (images) where linear methods fail.
- **Cons**: Needs substantially more data and compute to train well than PCA/UMAP; harder to interpret latent dimensions; requires careful architecture/hyperparameter tuning; risk of posterior collapse in VAE.
- **Pick over PCA/UMAP when**: Non-linear structure is complex (images, high-dim sensor data), you have enough data to train a network, or you need a generative model that can sample new data, not just reduce dimensions.
- **Key hyperparameters**: `latent_dim`, `beta` (VAE KL-term weight, controls disentanglement vs reconstruction fidelity), architecture depth/width.

**Comparison table — Dimensionality Reduction**

| Algorithm | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| PCA | Fast linear reduction, preprocessing, decorrelation | Non-linear structure | Low (SVD) | Medium (linear combos) |
| Kernel PCA | Known non-linear manifold structure | Large n (O(n²)+) | Medium-high | Low |
| ICA | Blind source separation, independent signals | Gaussian sources | Low-medium | Medium |
| LDA | Supervised class-separability projection | Non-Gaussian/unequal-covariance classes | Low | High (class-discriminative) |
| t-SNE | 2D/3D visualization, local structure | General preprocessing, need global distances, need new-point transform | High | Low (visualization only) |
| UMAP | Faster visualization, some preprocessing use, new-point transform | Need exact global distance preservation | Medium | Low |
| NMF | Additive parts-based components (text/images) | Data has negative values | Medium (non-convex) | High (parts-based) |
| Autoencoder/VAE | Complex non-linear structure, generative sampling | Small data, need fast/simple reduction | High (training a NN) | Low |

---

## Feature Selection

### Filter Methods (Variance Threshold, Correlation Removal, Mutual Information, Chi-Squared, ANOVA F-test)
- **What it is**: Score/rank features independently of any model, using statistical properties (variance, correlation, dependence with target).
- **Pros**: Very fast — no model training required; model-agnostic; good first-pass pruning before expensive methods.
- **Cons**: Blind to feature interactions (a feature useless alone but powerful in combination is discarded); linear tests (correlation, ANOVA F) miss non-linear dependence; each test has narrow applicability (chi-squared needs non-negative discrete-ish data, ANOVA assumes linear class-mean separation).
- **Pick over wrapper/embedded methods when**: You have very high dimensionality (thousands of features) and need cheap, fast first-pass pruning before any model training.
- **Key hyperparameters**: threshold/percentile/k (how many features to keep).

### Wrapper Methods (RFE, Sequential Feature Selection)
- **What it is**: Use actual model performance (via CV) to evaluate feature subsets — RFE removes the least important feature iteratively; sequential selection greedily adds/removes features based on CV score.
- **Pros**: Accounts for the actual model/learning algorithm being used; RFECV auto-selects optimal feature count; can capture some interaction effects that filters miss.
- **Cons**: Expensive — O(n_features) or more model retrainings; greedy search isn't guaranteed globally optimal; impractical with slow models or very large feature sets.
- **Pick over filter methods when**: Feature count is manageable (tens to low hundreds), model is fast to train, and you want the subset tuned specifically to that model's performance.
- **Key hyperparameters**: `n_features_to_select`, `direction` (forward/backward for SFS), `cv`.

### Embedded Methods (Lasso/L1, Tree-based importance, SelectFromModel, SHAP-based selection)
- **What it is**: Selection happens as a byproduct of model training itself — no separate selection pass needed (Lasso zeroes out coefficients; trees rank by impurity decrease).
- **Pros**: One training pass does both fitting and selection — most efficient; Lasso gives a principled sparse model; tree importance/SHAP integrates naturally into existing ensemble workflows.
- **Cons**: Lasso arbitrarily picks among correlated features; MDI tree importance is biased toward high-cardinality features (use permutation importance or SHAP instead); SHAP splits credit across correlated features, understating their joint importance.
- **Pick over filter/wrapper methods when**: You're already training a Lasso/tree/boosting model and want selection "for free," or need interaction-aware importance (SHAP) without a separate wrapper search.
- **Key hyperparameters**: `alpha`/`lambda` (Lasso sparsity), `threshold` (SelectFromModel).

**Comparison table — Feature Selection**

| Method family | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| Filter | High-dim data, fast first-pass pruning | Feature interactions matter | Very low | High |
| Wrapper (RFE/SFS) | Small-medium feature count, model-specific tuning | Slow model, huge feature count | High (many retrains) | Medium |
| Embedded (Lasso/tree/SHAP) | Efficient selection during training | Correlated features (unstable/split credit) | Low (piggybacks on training) | Medium-high |

---

## Imbalanced Data Techniques

### Resampling: Random Oversampling / Undersampling
- **What it is**: Duplicate minority samples or drop majority samples to balance class ratio before training.
- **Pros**: Simple, fast, model-agnostic.
- **Cons**: Oversampling duplicates cause overfitting to exact points; undersampling discards potentially useful majority data.
- **Pick over SMOTE when**: Dataset is large (>100k) so undersampling doesn't lose much signal, or you need something trivially simple.

### SMOTE / ADASYN
- **What it is**: Generate synthetic minority samples by interpolating between real minority neighbors (SMOTE uniformly; ADASYN focuses more synthetic density near the boundary where the model struggles).
- **Pros**: More diverse minority coverage than duplication; ADASYN specifically improves hard/boundary-region learning.
- **Cons**: Assumes minority class is convex/interpolatable — fails with multimodal minority clusters or categorical features (need SMOTE-NC); ADASYN can amplify noisy boundary outliers.
- **Pick over cost-sensitive learning when**: Dataset is small (<10k), and synthetic diversity is more valuable than simply reweighting the loss.

### Tomek Links / SMOTEENN / SMOTETomek
- **What it is**: Boundary-cleaning undersampling — remove majority-class members of Tomek link pairs, optionally combined with prior SMOTE oversampling.
- **Pros**: Produces a cleaner decision boundary; combination methods (SMOTEENN/SMOTETomek) balance generation and cleaning.
- **Cons**: Tomek Links alone gives minimal undersampling; combination methods can shrink very small datasets too much during cleaning.
- **Pick over plain SMOTE when**: Classes overlap significantly and boundary noise needs explicit cleanup.

### Cost-Sensitive Learning (class_weight='balanced', scale_pos_weight)
- **What it is**: Reweight the loss function so minority-class errors cost more, without touching the data itself.
- **Pros**: No data modification/synthetic artifacts; composes cleanly with resampling; simple one-line change in sklearn/XGBoost.
- **Cons**: Doesn't change output probability calibration space — probabilities remain in the reweighted-loss space and may need post-hoc calibration.
- **Pick over resampling when**: You want zero data modification, need interpretability preserved exactly, or are using a model/framework where weighted loss is trivial to apply (XGBoost `scale_pos_weight`).

### Threshold Moving
- **What it is**: Keep the trained model as-is; adjust the decision threshold (not 0.5) post-hoc based on the precision-recall curve or asymmetric cost matrix.
- **Pros**: Zero extra training cost; directly optimizes the metric/cost function you actually care about.
- **Cons**: Threshold tuned on one validation set may not generalize under distribution drift; must not tune on the test set (leakage).
- **Pick over resampling/reweighting when**: You already have well-ranked probability scores and just need to pick the right operating point — cheapest possible fix.

### Ensemble-level: BalancedRandomForest / EasyEnsemble
- **What it is**: Push balancing into the ensemble mechanism itself — each tree/base learner in the ensemble trains on a class-balanced bootstrap or undersampled subset.
- **Pros**: Uses many different majority subsets (EasyEnsemble) instead of discarding majority data outright; integrates rebalancing into model training.
- **Cons**: Changes the effective class prior seen during training vs deployment — causes miscalibration; needs post-hoc calibration on the true distribution.
- **Pick over data-level resampling when**: Using tree ensembles specifically and want rebalancing built into the bagging mechanism rather than as a preprocessing step.

### Focal Loss / Label Smoothing (deep learning-adjacent)
- **What it is**: Focal loss down-weights easy/confident examples dynamically via a `(1-p_t)^γ` modulating factor; label smoothing softens hard 0/1 targets to reduce overconfidence.
- **Pros**: Focal loss lets hard examples dominate gradient regardless of class; label smoothing improves calibration.
- **Cons**: Focal loss adds 2 more hyperparameters (`alpha`, `gamma`) and its tabular-classification benefit is inconsistent (designed for dense object detection); label smoothing alone doesn't fix imbalance, just overconfidence.
- **Pick over class weighting when**: Working with deep learning models (not tabular/tree-based), especially dense prediction tasks; try class weighting first for tabular problems.

**Comparison table — Imbalanced Data**

| Technique | Best for | Avoid when | Training cost | Interpretability impact |
|---|---|---|---|---|
| Random over/undersampling | Quick baseline, large datasets (undersample) | Small minority class (undersample loses too much) | Low | None |
| SMOTE / ADASYN | Small datasets, need synthetic diversity | Multimodal minority class, categorical features | Low-medium | None |
| SMOTEENN / SMOTETomek | Overlapping classes, boundary noise | Very small datasets (over-cleans) | Medium | None |
| class_weight / scale_pos_weight | No data modification, XGBoost/sklearn native | Need calibrated probabilities without extra step | None (built-in) | None |
| Threshold moving | Already-trained model, cheapest fix | Threshold must generalize under drift | None | None |
| BalancedRandomForest/EasyEnsemble | Tree ensembles, want balancing baked into bagging | Calibration is critical without post-hoc fix | Medium | None |
| Focal loss / label smoothing | Deep learning, dense prediction tasks | Simple tabular problems (try class weights first) | Low (just a loss change) | None |

**Evaluation metrics for imbalance**: Precision/Recall/F1, F-beta (weight recall vs precision asymmetrically), PR-AUC/Average Precision (preferred over ROC-AUC — ROC-AUC is inflated by the large true-negative pool), MCC (accounts for all 4 confusion matrix quadrants, robust to imbalance).

---

## Calibration

### Platt Scaling
- **What it is**: Fit a logistic regression on top of a model's raw scores (on a held-out calibration set) to map them to calibrated probabilities.
- **Pros**: Works with small calibration sets (<1k examples); simple parametric fit (2 params: A, B).
- **Cons**: Assumes miscalibration is monotone and roughly sigmoid-shaped — wrong if the true miscalibration curve is more complex.
- **Pick over Isotonic Regression when**: Calibration set is small (<500-1k examples) or miscalibration is known to be simple/monotone-sigmoid shaped.

### Isotonic Regression
- **What it is**: Non-parametric, monotone step-function fit (via Pool Adjacent Violators) from raw scores to calibrated probabilities.
- **Pros**: No assumed functional form — handles complex, non-linear miscalibration patterns.
- **Cons**: Overfits with small calibration sets (<500 examples) — memorizes local bin frequencies.
- **Pick over Platt Scaling when**: You have a larger calibration set (>1k) and suspect complex/non-monotone-sigmoid miscalibration.

### Temperature Scaling
- **What it is**: Divide neural network logits by a single learned scalar T before softmax; doesn't change argmax/predictions, only softens/sharpens confidence.
- **Pros**: Single parameter — very low overfitting risk; preserves accuracy exactly; standard fix for the well-documented overconfidence of modern deep nets.
- **Cons**: One scalar can't fix per-class miscalibration (if class A is over- and class B is under-confident, one T can't fix both) — needs vector/matrix scaling for that.
- **Pick over Platt/Isotonic when**: Calibrating a neural network specifically (not tree/SVM/tabular models) — this is the standard, minimal-parameter approach for deep nets.

**Comparison table — Calibration**

| Method | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| Platt Scaling | Small calibration set, SVM/XGBoost scores | Complex non-sigmoid miscalibration | Low | High (2 params) |
| Isotonic Regression | Large calibration set, complex miscalibration | Small calibration set (<500) | Low-medium | Medium (step function) |
| Temperature Scaling | Neural network overconfidence | Per-class miscalibration differs | Very low (1 param) | High (1 param) |

**Model calibration defaults** (from this folder): Logistic Regression — natively well-calibrated (optimizes log-loss). Random Forest — S-shaped miscalibration from vote averaging. SVM — severely miscalibrated (optimizes margin, not likelihood) — needs Platt/Isotonic. XGBoost — moderately overconfident. Deep Networks — overconfident (needs temperature scaling).

**Uncertainty quantification**: Aleatoric (irreducible, data-noise — quantified via output entropy) vs Epistemic (reducible, model-uncertainty — quantified via variance across MC Dropout passes or Deep Ensembles). MC Dropout is a cheap single-model approximation; Deep Ensembles (5 independently trained models) give better-quality uncertainty at 5x training/inference cost. Conformal Prediction gives a distribution-free coverage guarantee ($P(Y \in C(X)) \ge 1-\alpha$) regardless of model — pick when you need a provable guarantee rather than just a well-calibrated probability.

---

## Model Interpretation

### SHAP (TreeSHAP / KernelSHAP)
- **What it is**: Game-theoretic (Shapley value) attribution of each feature's contribution to a prediction; TreeSHAP computes it exactly in polynomial time for tree models, KernelSHAP approximates it for any model via weighted linear regression over feature coalitions.
- **Pros**: Theoretically grounded (efficiency/symmetry/dummy/linearity axioms); consistent — same model always gives same explanation; valid to aggregate locally-computed values into global importance; TreeSHAP is fast (milliseconds).
- **Cons**: KernelSHAP is slow (seconds per sample) for non-tree models; correlated features split SHAP credit, understating joint importance.
- **Pick over LIME when**: You need consistency/reproducibility, want valid global aggregation from local explanations, or are explaining a tree-based model (TreeSHAP is fast and exact).
- **Key hyperparameters**: n/a for TreeSHAP; `nsamples` for KernelSHAP (approximation quality vs speed).

### LIME
- **What it is**: Explains one prediction by fitting a local weighted linear regression on perturbed samples around that input.
- **Pros**: Model-agnostic; conceptually simple; fast approximate local explanation.
- **Cons**: Not additive/consistent across runs (stochastic sampling); local linearity assumption breaks for highly non-linear boundaries; global aggregation of LIME coefficients isn't valid.
- **Pick over SHAP when**: Need a very fast rough local explanation and don't need consistency/global-aggregation guarantees.

### Permutation Importance
- **What it is**: Shuffle one feature at a time on validation data, measure the performance drop — model-agnostic, global.
- **Pros**: Model-agnostic; unbiased with respect to feature cardinality (unlike MDI).
- **Cons**: Correlated features protect each other (shuffling one doesn't hurt because the other compensates) — understates joint importance; measures predictive, not causal, importance.
- **Pick over MDI (Gini) importance when**: You need an unbiased importance measure not skewed toward high-cardinality features.

### PDP vs ALE
- **What it is**: PDP shows a feature's marginal effect by averaging predictions over all other features at fixed values of the target feature; ALE fixes PDP's correlated-feature extrapolation problem by accumulating local differences within narrow bins instead.
- **Pros**: Both give intuitive marginal-effect visualizations; ALE is safe under feature correlation.
- **Cons**: PDP creates unrealistic/extrapolated feature combinations when features are correlated (e.g., age=90 with student-level income).
- **Pick ALE over PDP when**: Features are correlated — always the safer default in real tabular data.

**Comparison table — Model Interpretation**

| Method | Best for | Avoid when | Speed | Fidelity |
|---|---|---|---|---|
| SHAP (Tree) | Tree models, regulatory/legal explanations | Non-tree models (use KernelSHAP, slower) | Fast | Exact (for trees) |
| SHAP (Kernel) | Any model, need theoretical grounding | Need real-time explanation at scale | Slow | Approximate |
| LIME | Fast rough local explanation, any model | Need consistency or valid global aggregation | Moderate | Approximate |
| Permutation Importance | Unbiased global importance, any model | Highly correlated features present | Moderate | Good |
| PDP | Marginal effect visualization, uncorrelated features | Correlated features (use ALE) | Moderate | Marginal (can mislead) |
| ALE | Marginal effect with correlated features | n/a — generally safe default | Moderate | Better than PDP |

---

## Time Series Analysis

### ARIMA / SARIMA
- **What it is**: Autoregressive Integrated Moving Average — models a (differenced, for stationarity) series as a linear function of its own past values (AR) and past forecast errors (MA); SARIMA adds seasonal AR/MA/differencing terms.
- **Pros**: Interpretable coefficients; strong theoretical grounding; works well with limited data; ACF/PACF give a principled way to identify order.
- **Cons**: Requires (weak) stationarity — must difference/transform first; univariate by default; assumes linear dependence; doesn't easily incorporate exogenous regressors without extension (SARIMAX).
- **Pick over Prophet/XGBoost when**: Series is stationary (or easily made so), univariate, short horizon, and you need interpretable AR/MA coefficients with limited data.
- **Key hyperparameters**: `p,d,q` (AR order, differencing order, MA order) via ACF/PACF/auto_arima; `P,D,Q,s` for seasonal terms.

### Exponential Smoothing (SES / Holt / Holt-Winters)
- **What it is**: Forecasts via exponentially decaying weighted average of past observations; Holt adds a trend component, Holt-Winters adds seasonality on top.
- **Pros**: Simple, fast, robust with limited data; explicit trend/seasonality decomposition is interpretable.
- **Cons**: Limited expressiveness vs ARIMA/ML models for complex patterns; still fundamentally linear/additive-or-multiplicative structure.
- **Pick over ARIMA when**: Series has clear, simple trend/seasonality patterns and you want a fast, easy-to-tune model without stationarity testing/differencing.

### Prophet
- **What it is**: Additive decomposition model (trend + seasonality via Fourier series + holiday effects), designed for business time series.
- **Pros**: Handles missing data, outliers, multiple seasonalities, and holidays out of the box; easy to tune with interpretable components (changepoint flexibility, seasonality mode); robust to irregular sampling.
- **Cons**: Doesn't capture autocorrelation directly like ARIMA; can overfit with too many changepoints; less accurate than ML/DL methods when the series has complex non-additive interactions.
- **Pick over ARIMA when**: Strong multi-seasonality (holiday/weekly/annual) and irregular sampling, and non-expert users need tunable, robust components without formal stationarity analysis.

### Tree-based (XGBoost/LightGBM with lag features)
- **What it is**: Reframe forecasting as supervised regression using lag/rolling features as inputs.
- **Pros**: Handles many simultaneous series efficiently; naturally incorporates external regressors (price, promotions); captures complex non-linear patterns.
- **Cons**: Needs careful lag/rolling feature engineering; no native handling of trend/seasonality — must be feature-engineered; can extrapolate poorly beyond training value ranges.
- **Pick over ARIMA/Prophet when**: Forecasting many series simultaneously (e.g., 10k SKUs) with strong external regressors and sufficient history.

### Deep Learning for Time Series (TCN, PatchTST, N-BEATS, Foundation Models)
- **What it is**: TCN uses causal dilated convolutions; PatchTST/Informer/Autoformer are Transformer variants for long-horizon/multivariate forecasting; N-BEATS is an interpretable pure-MLP basis-expansion model; Chronos/Moirai/TimesFM are pretrained foundation models for zero-shot forecasting.
- **Pros**: Captures long-range, multivariate, cross-series dependencies; foundation models give zero-shot forecasting with no historical data (cold start).
- **Cons**: Needs substantially more data/compute to justify over classical methods; less interpretable; overkill for short univariate series.
- **Pick over classical methods when**: Very long horizon, multivariate with complex cross-series dependencies, cold-start with no historical data (use a foundation model zero-shot), or irregular time series (Chronos/Moirai).

**Comparison table — Time Series**

| Method | Best for | Avoid when | Training cost | Interpretability |
|---|---|---|---|---|
| ARIMA/SARIMA | Stationary univariate, short horizon, interpretable coefficients | Complex non-linear/multivariate patterns | Low-medium | High |
| Exponential Smoothing | Simple trend/seasonality, fast baseline | Complex/non-additive patterns | Low | High |
| Prophet | Strong seasonality/holidays, irregular sampling, business series | Need autocorrelation modeling, minimal changepoints needed | Low-medium | High |
| Tree-based (XGBoost+lags) | Many series at once, external regressors, non-linear patterns | Little history, no engineered lag features | Medium | Medium (needs SHAP) |
| Deep Learning / Foundation Models | Long horizon, multivariate, cold-start zero-shot | Short series, limited data/compute, need interpretability | High | Low |

**Key evaluation practice**: never random-split (use walk-forward / `TimeSeriesSplit`); metrics — MAE (robust to outliers), RMSE (penalizes large errors), MAPE (percentage, easy to communicate but breaks near zero), MASE (scale-free vs naive forecast, MASE<1 beats naive).

---

## When Classical ML Wins (Decision Framework)

- **Tabular data**: Trees/GBMs dominate 90% of tabular benchmarks over deep learning (Grinsztajn et al. 2022) — no spatial/sequential structure for DL inductive biases to exploit.
- **Small data (n < 10k)**: Logistic Regression/SVM (<1k), Random Forest/XGBoost with regularization (1k-10k) — fewer parameters fit the data budget; DL overfits.
- **Interpretability/regulatory requirements (GDPR/FCRA/EU AI Act)**: Logistic Regression (exact coefficients) or XGBoost + TreeSHAP (exact, fast, defensible) — neural net explanations (KernelSHAP/Integrated Gradients) are approximate and less legally defensible.
- **Latency-critical (<5ms single-prediction)**: Logistic Regression (1-5μs) or compiled decision tree (5-20μs) — neural nets need GPU batching to compete.
- **Calibrated probabilities required**: Logistic Regression is natively calibrated (optimizes log-loss) — eliminates the calibration step other models need.
- **Class imbalance**: Classical ML composes multiple independent levers (resampling + cost-sensitive weights + threshold moving) vs DL's primary lever (focal loss).
- **Domain expertise available**: Feature engineering (ratios, rolling stats, domain-derived features) lets classical models find relationships a DL model would need huge data to discover from raw inputs.
- **When DL wins instead**: images/text/audio (spatial/sequential inductive bias + pretrained backbones), n > 500k (scaling laws), no domain expertise in a novel domain (representation learning from raw data), a pretrained backbone already exists for the domain (transfer learning inherits sample efficiency).
