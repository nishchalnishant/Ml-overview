---
module: Classical Ml
topic: Interview Questions
subtopic: ""
status: unread
tags: [classicalml, ml, interview-questions]
---

# Classical ML — Interview Questions

**For:** SDE-2 / AI Engineer interviews — calibrated to what's actually asked Round 1 and beyond.
**Difficulty guide:**
- **Easy** → Round 1 basics: definitions, intuition, "explain like you're talking to an engineer." Expect these cold in a first technical round.
- **Medium** → Round 2 depth: connecting concepts, debugging scenarios, trade-off reasoning, applied system design.
- **Hard** → Staff/Research depth: full derivations, theoretical proofs, senior judgment questions.

---

## Easy

> Round 1 fundamentals for Classical ML. Know these cold before any interview.

### Q: How does a decision tree choose a split?
At each node the tree evaluates candidate (feature, threshold) splits and picks the one that maximizes **impurity reduction** (information gain). Two impurity measures: **Gini** = 1 − Σp²_k (faster, no log, sklearn default), **Entropy** = −Σp_k log₂p_k (slightly more sensitive to class balance changes). Both are 0 for a pure node and maximal at uniform distribution. In practice they pick nearly identical splits — the bigger levers for controlling overfitting are `max_depth`, `min_samples_leaf`, and `min_samples_split`.

### Q: What is the difference between supervised and unsupervised learning?
Supervised learning trains on labeled examples (x_i, y_i) to learn a mapping f: X → Y — classification predicts discrete labels, regression predicts continuous values. Unsupervised learning works with unlabeled data X only, discovering structure — clustering (grouping similar points), dimensionality reduction (compressing while preserving structure), density estimation. Evaluation differs: supervised models use ground-truth labels (accuracy, RMSE); unsupervised models need internal metrics (silhouette score, reconstruction error) or external labels never used during training.

### Q: Explain the difference between L1 and L2 regularization. Why does L1 induce sparsity?
Both add a penalty term to the loss to discourage large weights. **L2 (Ridge)**: penalty = λΣw², shrinks weights smoothly toward zero but never to exactly zero. **L1 (Lasso)**: penalty = λΣ|w|, has a non-differentiable corner at zero; geometrically the diamond-shaped L1 constraint region has corners on the coordinate axes, so the loss contour is much more likely to first touch the constraint at a corner (a coordinate = 0) → exact sparsity. Bayesian view: L2 = Gaussian prior on weights, L1 = Laplace prior (sharp peak at zero). Use L1 for feature selection/interpretability; L2 when most features are weakly relevant; ElasticNet when features are correlated.

### Q: What is the bias-variance tradeoff, and how does KNN illustrate it?
Bias is error from overly simplistic assumptions (underfitting); variance is sensitivity to which training examples the model happened to see (overfitting). Total error = Bias² + Variance + Irreducible Noise. In KNN: k=1 has near-zero bias but very high variance (jagged, noise-memorizing decision boundary). As k increases, the boundary smooths (bias increases, variance decreases). At k=n the model always predicts the majority class (maximum bias, zero variance). Extra: KNN also degrades badly in high dimensions because all points become roughly equidistant (curse of dimensionality).

### Q: What are the three broad families of feature selection methods?
1. **Filter**: score features independent of any model (variance threshold, correlation, mutual information, chi-squared) — fast but ignores feature interactions.
2. **Wrapper** (e.g., RFE): use the actual model's performance to greedily add/remove features — captures interactions but computationally expensive.
3. **Embedded**: selection built into training itself (L1/Lasso zeroing coefficients, tree-based feature importances, `SelectFromModel`) — good middle ground, nearly as informative as wrapper but only one model fit.

Practical: use filter to remove obvious junk (near-zero variance, highly correlated duplicates), then run wrapper or embedded on the reduced set.

### Q: What is the curse of dimensionality, and why does it break distance-based algorithms?
As dimensionality grows, data becomes exponentially sparser for fixed sample size. A key consequence: the ratio (max distance − min distance) / min distance → 0 as dimensions increase — nearest and farthest points become almost equidistant, so "nearest neighbor" loses meaning. This breaks KNN, K-Means, and any kernel method relying on meaningful distances. Mitigations: dimensionality reduction (PCA, UMAP) before applying distance-based methods, using cosine similarity for high-dimensional sparse data, or switching to tree-based models that are invariant to feature scale.

### Q: What is one-hot encoding and what problem does plain integer label encoding create?
One-hot encoding represents a categorical with k categories as k binary columns. Plain integer encoding (red=0, green=1, blue=2) imposes a **false ordinal relationship** — a linear model treats "blue" as literally further from "red" than "green" is, which is wrong for nominal (unordered) categories. One-hot avoids this by treating each category as an independent dimension. Trade-off: one-hot explodes dimensionality for high-cardinality categoricals → use target encoding, hashing, or embeddings for hundreds/thousands of levels.

### Q: What is k-fold cross-validation, and why is it preferred over a single train/test split?
K-fold CV splits data into k equally sized folds, trains on k−1 and validates on the remaining, repeating k times so every fold serves as validation once, then averages scores. A single split's score depends heavily on which rows happened to land in each set (high variance, especially with small datasets). K-fold averages over k different validation sets → lower-variance, more reliable estimate. Also uses data more efficiently — every row is used for both training and validation. Common choices: k=5 or k=10.

### Q: Define precision, recall, and F1. When do you prioritize each?
- **Precision** = TP/(TP+FP): of everything flagged positive, how much was actually positive?
- **Recall** = TP/(TP+FN): of everything actually positive, how much did we catch?
- **F1** = 2·P·R/(P+R): harmonic mean, penalizes a large gap between the two.

Prioritize **precision** when false positives are costly (spam filter blocking legitimate email). Prioritize **recall** when false negatives are costly (cancer screening — missing a real case is far worse than a false alarm). Use F1 when you want a single balanced number.

### Q: What is class imbalance, and what are the two broad families of techniques to address it?
Class imbalance: one class vastly outnumbers another (e.g., 1% fraud vs 99% legitimate), causing models trained on overall error to favor the majority class. Two broad families:
1. **Data-level**: resampling — oversample minority (SMOTE, ADASYN) or undersample majority (random removal, Tomek Links).
2. **Algorithm-level**: change how the model weights errors — `class_weight='balanced'`, cost-sensitive loss, focal loss.

A third, often-overlooked lever: **threshold moving** — leave training unchanged, move the classification threshold post-hoc to match the desired precision/recall tradeoff.

### Q: Compare MAE, MSE, and Huber loss for regression.
- **MAE** = (1/n)Σ|y−ŷ|: robust to outliers (linear penalty), but non-differentiable at zero and constant gradient magnitude slows convergence near the optimum.
- **MSE** = (1/n)Σ(y−ŷ)²: smooth and differentiable everywhere, but highly sensitive to outliers (errors are squared).
- **Huber loss**: quadratic for |error| < δ, linear for |error| ≥ δ — combines MSE's smooth gradient near zero with MAE's outlier robustness for large residuals.

Use MSE as the default; switch to Huber when the target has occasional outliers you don't want to chase.

### Q: What is a confusion matrix?
A 2×2 table cross-tabulating predicted vs actual class: **TP** (predicted positive, actually positive), **FP** (false alarm / Type I error), **TN** (predicted negative, actually negative), **FN** (missed case / Type II error). Nearly every classification metric derives from these four counts. Generalizes to a k×k matrix for multi-class — diagonal = correct predictions, off-diagonal shows which classes get confused with which.

---

## Medium

> Round 2 depth — applied scenarios, trade-offs, and "how would you debug this?" questions.

### Q: When would you choose SVM over logistic regression, and vice versa?
SVM maximizes margin and only cares about support vectors near the boundary — robust in high-dimensional, low-sample regimes (e.g., text with TF-IDF) and effective with the kernel trick for non-linear boundaries. Logistic regression gives calibrated probabilities directly, scales better to large datasets (convex, easy SGD), and is more interpretable via coefficients. Use SVM in limited-data, high-dimension settings; use logistic regression when you need probability estimates, fast training at scale, or interpretability. With a linear kernel both often perform similarly — SVM's edge shows up mainly with RBF/poly kernels on non-linear data.

### Q: Explain the kernel trick and why it avoids computing high-dimensional feature maps explicitly.
The kernel trick replaces the dot product φ(x_i)ᵀφ(x_j) in the dual SVM with a kernel function K(x_i, x_j) that computes the same value without materializing φ. For RBF: K(x_i, x_j) = exp(−γ‖x_i − x_j‖²) implicitly corresponds to an infinite-dimensional feature space. Since the SVM dual objective and decision function only ever need inner products, you get the benefit of a non-linear, possibly infinite-dimensional mapping at the cost of an O(n²) Gram matrix instead of the (potentially infeasible) explicit transform. Gotcha: kernel methods scale poorly to large n because of the n×n Gram matrix — why SVMs fall out of favor for big datasets in favor of tree ensembles or neural nets.

### Q: Compare Random Forest and Gradient Boosting — when would you pick one over the other?
Random Forest: builds many deep trees independently on bootstrap samples + random feature subsets, averages them. Reduces variance. Naturally parallelizable, robust to noisy hyperparameters, hard to overfit by adding more trees. Gradient Boosting: builds shallow trees sequentially, each fitting the residual/negative gradient of the current ensemble. Reduces bias. Sequential (slower to train), more sensitive to hyperparameters (learning rate, depth, n_estimators), can overfit without early stopping. Rule of thumb: Random Forest for a fast, low-maintenance baseline; GBM (XGBoost/LightGBM/CatBoost) when squeezing maximum accuracy on tabular data. GBMs typically win Kaggle-style competitions.

### Q: XGBoost, LightGBM, and CatBoost all implement gradient boosting — what actually differs?
- **Tree growth**: XGBoost grows level-wise (breadth-first), LightGBM grows leaf-wise (best-first — faster convergence but more prone to overfitting on small data), CatBoost uses symmetric/oblivious trees (same split across an entire level — regularizing and fast at inference).
- **Categorical handling**: XGBoost needs manual encoding, LightGBM supports categoricals natively, CatBoost uses ordered target statistics avoiding target leakage without manual encoding.
- **Gradient computation**: all three use second-order (Newton) boosting with gradient + Hessian.
- **Practical guidance**: LightGBM is fastest on large datasets; CatBoost often wins with heavy categorical features; XGBoost is the most portable/battle-tested default.

### Q: Your model gets 95% accuracy on a fraud dataset — is it good? Debug it.
First check the base rate: if fraud is 2% of transactions, predicting "not fraud" always gets 98% accuracy. 95% could be *worse* than doing nothing. Steps: (1) check class distribution; (2) replace accuracy with PR-AUC, F1, recall at a fixed precision, or MCC; (3) inspect the confusion matrix — likely near-zero recall on fraud; (4) confirm the model isn't always predicting one class; (5) address imbalance via class weights, SMOTE, or threshold moving; (6) validate with stratified CV. Classic "your metric is lying to you" trap — fix the metric before touching the model.

### Q: List the major types of data leakage and how to catch each one.
1. **Target leakage**: a feature encodes information only available after/because of the outcome (e.g., "days since cancellation" for churn). Catch by checking whether a feature would exist at prediction time, and auditing suspiciously perfect predictors.
2. **Train/test contamination**: fitting preprocessors (scalers, encoders) on full data before splitting. Fix by fitting only on the train fold inside a Pipeline.
3. **Temporal leakage**: using future information to predict the past (random-split CV on time series). Fix with TimeSeriesSplit / walk-forward validation.
4. **ID/proxy leakage**: a near-identifier correlates with the target by construction (e.g., row order). Audit feature-target correlations for implausibly high values. Signal: "too good to be true" validation performance that collapses in production.

### Q: What is K-Means' core assumption, and when does it fail?
K-Means implicitly assumes clusters are (1) convex/spherical, (2) similar size, (3) similar density, (4) separable via Euclidean distance. It fails on: elongated or non-convex clusters (use DBSCAN or spectral clustering), clusters of very different sizes/densities (larger cluster's points get stolen), and when true k is unknown. GMM generalizes K-Means by allowing elliptical clusters with soft assignment — K-Means is the special case of GMM with equal, spherical, isotropic covariances and hard assignment.

### Q: How does DBSCAN define clusters, and what are its strengths vs K-Means?
DBSCAN classifies points as **core** (≥ MinPts neighbors within eps), **border** (within eps of a core point but not itself core), or **noise** (neither). Clusters form by connecting density-reachable core points and absorbing their border points. Strengths: no need to specify k upfront, discovers arbitrarily shaped clusters, naturally identifies outliers as noise. Weaknesses: struggles with clusters of varying density (one global eps can't fit both dense and sparse clusters), sensitive to eps/MinPts choice, degrades in high dimensions. HDBSCAN addresses varying density by building a hierarchy over multiple density thresholds.

### Q: How does target encoding work, and why is naive target encoding dangerous?
Target encoding replaces a categorical value with the mean of the target for that category (μ_k). Naive implementation computed on the full training set: each row "sees" its own label contributing to its encoded value → severe overfitting, especially for rare categories. Fixes: (1) **out-of-fold computation** — encode each fold using only the other k−1 folds so no row contributes to its own encoding; (2) **additive smoothing toward the global mean**: μ̂_k = (n_k · ȳ_k + m · ȳ) / (n_k + m), where m controls shrinkage for low-count categories; (3) add small random noise. Naive target encoding is a classic leaderboard-vs-production gap in Kaggle workflows.

### Q: What is Population Stability Index (PSI) and how do you use it to detect drift?
PSI compares a feature's distribution between a baseline (training) and a current period by binning both into the same buckets: PSI = Σ (p_current − p_baseline) · ln(p_current / p_baseline). Thresholds: PSI < 0.1 = stable, 0.1–0.25 = moderate shift (investigate), > 0.25 = significant shift (likely requires retraining). PSI detects **covariate shift** (change in P(X)) but says nothing about **concept drift** (change in P(Y|X)) — for that you need label monitoring or a proxy like tracking live performance on labeled cohorts as labels arrive.

### Q: Why is standard K-Fold CV invalid for time series? What's the correct alternative?
Standard K-Fold shuffles randomly — some training folds will contain data from *after* the validation fold, leaking future information and inflating validation performance. Fix: `TimeSeriesSplit` (expanding window — train on all data up to time t, validate on the next window, then increase t). Add a `gap` parameter between train and validation to simulate realistic prediction lag and avoid leakage from features that need time to settle (e.g., a 7-day rolling average computed too close to the validation boundary).

### Q: Why can't accuracy evaluate imbalanced classification? What should you use instead?
With a 99:1 class ratio, a trivial always-predict-majority classifier scores 99% accuracy with zero recall on the class that matters. Better metrics: **PR-AUC** (most informative for rare positive class — precision's denominator is sensitive to false positives in absolute terms, unlike ROC-AUC); **F1/F-beta** (harmonic mean of precision/recall, F-beta lets you weight recall vs precision by domain cost); **MCC** (single balanced summary robust to imbalance, uses all four confusion-matrix cells). Always inspect the full confusion matrix alongside any single-number metric.

### Q: What is SMOTE and what is its main failure mode?
SMOTE (Synthetic Minority Oversampling Technique) generates synthetic minority-class samples by picking a minority point, finding its k nearest minority neighbors, and creating a new point along the line segment between them: x_new = x_i + δ·(x_nn − x_i). Better than naive duplication because it creates genuinely new points. **Failure mode**: SMOTE interpolates blindly without considering class boundaries — if minority points are near or inside majority-class regions, SMOTE can generate synthetic points that land inside the majority class, blurring the decision boundary and hurting precision. Fix: Borderline-SMOTE (oversamples only near-boundary minority points), ADASYN (weights generation toward harder-to-learn points), or combine with a cleaning step like Tomek Links (SMOTETomek).

### Q: Compare ARIMA, Prophet, and gradient-boosted trees for time series forecasting.
- **ARIMA/SARIMA**: well-suited for a single, short, well-behaved series with clear autocorrelation structure. Interpretable (AR/I/MA orders map to specific behaviors, diagnosed via ACF/PACF), but requires manual stationarity handling, doesn't scale to thousands of series or incorporate rich exogenous features.
- **Prophet**: designed for business/human-scale time series with strong seasonality (daily/weekly/yearly) and holiday effects. Robust to missing data and outliers, additive decomposition, little tuning needed. Good default for business dashboards.
- **GBM (XGBoost/LightGBM)**: reframes forecasting as supervised regression using lag features, rolling statistics, calendar features, and exogenous variables. Shines when you have rich covariates, many related series at once (global models), or complex non-linear interactions. Needs careful walk-forward validation and feature engineering; doesn't extrapolate trends as gracefully as explicit trend models.

---

## Hard

> Staff-level and Research Engineer depth — derivations, theoretical proofs, senior judgment.

### Q: Derive the gradient of the logistic regression loss function.
The loss is binary cross-entropy: L = −(1/n)Σ[y_i log σ(wᵀx_i) + (1−y_i) log(1−σ(wᵀx_i))]. Since σ'(z) = σ(z)(1−σ(z)), the chain rule collapses cleanly and the gradient w.r.t. w is:

**∇_w L = (1/n) Σ (σ(wᵀx_i) − y_i) x_i**

Same form as linear regression's gradient (prediction error × feature), which is why logistic regression trains just as easily with plain gradient descent. Gotcha: why not MSE? MSE on top of sigmoid is non-convex and gives **vanishing gradients when predictions are confidently wrong** (sigmoid saturates → small gradient exactly when you need large correction). Cross-entropy is convex in w and penalizes confident wrong predictions much more sharply.

### Q: Prove that Lloyd's algorithm (standard K-Means) converges.
K-Means minimizes inertia J = Σ‖x_i − μ_{c_i}‖². Each iteration: (1) **Assignment step** — fixing centroids, assign each point to its nearest centroid → J can only decrease or stay the same (each point moves to a weakly closer centroid). (2) **Update step** — fixing assignments, set each centroid to the mean of its cluster → the mean minimizes within-cluster sum of squares for that fixed partition, again only decreasing or maintaining J. Since J is monotonically non-increasing and bounded below by 0, and there are finitely many possible partitions of n points into k clusters, the algorithm must converge in finite steps — to a **local, not necessarily global, minimum**. This is why K-Means++ (probabilistic seeding proportional to squared distance from existing centroids) and multiple random restarts (n_init) are used in practice.

### Q: Derive PCA from first principles as a variance-maximization problem.
Goal: find a unit direction w (‖w‖=1) maximizing variance of projected data wᵀx. For centered data, projected variance = wᵀΣw where Σ is the covariance matrix. Maximize wᵀΣw subject to wᵀw=1 using Lagrange multiplier: L = wᵀΣw − λ(wᵀw − 1). Gradient = 0 → **Σw = λw** — w must be an eigenvector of Σ, and the variance captured equals its eigenvalue λ. So the first principal component is the eigenvector with the largest eigenvalue; subsequent components are the next-largest eigenvectors, subject to orthogonality. Connection to SVD: X = USVᵀ → Σ = XᵀX/n = VS²Vᵀ/n, so right singular vectors V are the principal components. Explained variance ratio for component i = λ_i / Σλ_j.

### Q: Derive the AdaBoost model weight and explain it intuitively.
After fitting weak learner m with weighted error rate ε_m, AdaBoost assigns voting weight: **α_m = (1/2) ln((1−ε_m)/ε_m)**. Derived by minimizing the exponential loss Σe^{−y_i F(x_i)} greedily one weak learner at a time. Intuition: ε_m < 0.5 (better than random) → α_m > 0, grows large as ε_m → 0 (near-perfect learner gets a huge vote). ε_m = 0.5 (random) → α_m = 0 (no contribution). ε_m > 0.5 (worse than random) → α_m < 0 (predictions are inverted and still used). After computing α_m, sample weights are updated to upweight misclassified points so the next learner focuses on hard examples. Gradient Boosting generalizes this to arbitrary differentiable losses by fitting each new learner to the negative gradient (pseudo-residuals): F_m(x) = F_{m-1}(x) + η·h_m(x).

### Q: Walk through the t-SNE algorithm end to end, including why it uses a Student-t distribution in the low-dimensional space.
(1) In high-dimensional space, compute pairwise similarities as conditional probabilities p_{j|i} using a Gaussian kernel centered at each point, with bandwidth tuned per-point via binary search to match a target `perplexity`; symmetrize to get p_{ij}. (2) In the low-dimensional embedding (random init), compute similarities q_{ij} using a **Student-t distribution with 1 degree of freedom** (heavier tails than Gaussian). (3) Minimize KL divergence Σ p_{ij} log(p_{ij}/q_{ij}) via gradient descent. The heavy-tailed Student-t solves the **"crowding problem"**: in high dimensions there's more room to have many equidistant moderate-distance neighbors than in 2D — if you used a Gaussian in the low-dim space, points would be crushed together. Heavier tails let moderately dissimilar points be placed farther apart in 2D without huge probability penalty, giving t-SNE its well-separated clusters. Gotcha: t-SNE cluster sizes, distances *between* clusters, and even sometimes cluster count are **not reliably meaningful** — only local neighborhood structure is preserved. Never read anything into inter-cluster distances or use t-SNE output as input to a downstream model.

### Q: What are Shapley values, and what four axioms make them a "fair" attribution method?
Shapley values from cooperative game theory: the attribution for feature i is its **average marginal contribution** across all possible orderings/coalitions of features: φ_i = Σ_{S⊆F\{i}} [|S|!(|F|−|S|−1)!/|F|!] · [f(S∪{i}) − f(S)]. Four axioms: **(1) Efficiency** — attributions sum exactly to (prediction − baseline); **(2) Symmetry** — two features with identical contributions to every coalition get identical attribution; **(3) Dummy/Null player** — a feature that never changes the prediction in any coalition gets zero attribution; **(4) Linearity** — attributions for a linear combination of models equal the same combination of individual attributions. SHAP is the **unique** method satisfying all four simultaneously — why it's preferred in regulated/high-stakes settings. Practical: exact Shapley computation is exponential in features; TreeSHAP exploits tree structure to compute exact values in O(TLD²) (trees × leaves × depth²).

### Q: Compare SHAP vs tree-based Gini/MDI importance for feature selection — which is more reliable?
**MDI (mean decrease in impurity)**: sums impurity reduction a feature contributes across all tree splits. Biased toward **high-cardinality/continuous features** (more possible split points → more likely selected by chance) and computed on training data only (can reflect overfitting). **SHAP**: decomposes each individual prediction via Shapley values, satisfying consistency (a feature's importance can't decrease if its marginal contribution increases in a modified model) — MDI doesn't guarantee this. Ranking by mean absolute SHAP value is more reliable than MDI for feature selection, especially when features have very different cardinalities. Trade-off: TreeSHAP is still more expensive than reading off MDI for free during training. **Permutation importance** is a cheaper model-agnostic alternative that also avoids MDI's cardinality bias by measuring performance drop on a held-out set when a feature is shuffled.

### Q: A regulator asks why your model denied a loan applicant. Which interpretability technique do you use and how do you present the answer?
Use **SHAP local explanation** (force plot or waterfall plot) for that specific prediction: it decomposes the model's output into per-feature contributions relative to a baseline (average prediction), showing exactly which features pushed toward denial and which toward approval, summing exactly to the model's actual output (efficiency axiom). This is critical for GDPR's "right to explanation" or ECOA (fair lending laws) — they require an **individualized** reason, not a global feature importance ranking. Global tools (permutation importance, PDPs) answer "what matters on average," not "why this person." Practically: also verify the explanation doesn't rely on a legally protected or proxy attribute (e.g., zip code as race proxy), and validate it's monotonic against domain expectations (higher income should never decrease approval probability). This is where adding monotonic constraints in GBM training is often done proactively.

### Q: Your model has 0.92 ROC-AUC but the product team says predictions "feel wrong." Diagnostic process?
High discrimination (AUC) doesn't guarantee good calibration, actionable thresholds, or subgroup fairness. Steps: (1) **Plot a reliability diagram + compute ECE/Brier score** — check if predicted probabilities are systematically over/under-confident; (2) **check the decision threshold** — 0.5 is often wrong for the actual business cost structure, revisit via precision-recall tradeoff or a cost matrix; (3) **slice by subgroup/segment** — aggregate 0.92 AUC can hide a subgroup near-random performance (Simpson's paradox-style aggregation); (4) **check for distribution shift** (PSI) — the model may have degraded since training even though the historical AUC unchanged; (5) **clarify what "feels wrong" means** — miscalibrated probabilities, wrong ranking on high-value customers specifically, or a UX/threshold issue rather than a model quality problem. A single aggregate discrimination metric is necessary but not sufficient for a good production model.

### Q: How do you correctly compare two models' performance using a statistical test instead of eyeballing metric differences?
For classification, **McNemar's test**: build a 2×2 contingency table of agreement/disagreement between the two models on the **same** test examples (both correct, both wrong, A right/B wrong, A wrong/B right), and test whether the discordant cells are significantly different. This directly tests whether one model makes fewer errors on the same data — correct, since both models were tested on identical, non-independent examples. For comparing across multiple models/metrics simultaneously, apply **multiple hypothesis correction**: Bonferroni (divide α by number of tests, conservative), Holm's step-down (uniformly more powerful than Bonferroni), or Benjamini-Hochberg (controls FDR rather than FWER, preferred for large ablation studies). Gotcha: running many comparisons and reporting only significant ones without correction is p-hacking — with k tests at α=0.05, probability of at least one false positive ≈ 1−(0.95)^k.

### Q: Explain the difference between trend-stationarity and difference-stationarity, and why it matters for modeling.
A **trend-stationary** series can be made stationary by subtracting a deterministic trend (y_t = α + βt + ε_t, where ε_t is stationary noise) — shocks are transient and the series reverts to the trend line over time. A **difference-stationary** series (has a "unit root") requires differencing (y_t − y_{t−1}) to become stationary — shocks have a **permanent effect** on the level. The ADF test's null hypothesis is specifically "series has a unit root" (difference-stationary); failing to reject → difference rather than detrend. Modeling consequence: incorrectly detrending a difference-stationary series leaves residual non-stationarity; incorrectly differencing a trend-stationary series ("over-differencing") introduces artificial negative autocorrelation and inflates variance — a classic time-series gotcha for candidates who difference reflexively without testing which type of non-stationarity is present.

---

## Summary Table — Quick Reference

| Concept | Key fact |
|---|---|
| Decision tree split | Maximize information gain (Gini/Entropy) |
| L1 sparsity | Diamond constraint hits coordinate-axis corners |
| Bias-Variance | Error = Bias² + Variance + σ² |
| Cross-validation | k-fold averages over k validation sets; TimeSeriesSplit for time series |
| SMOTE | Synthetic interpolation along minority-class line segments |
| Precision/Recall | TP/(TP+FP) / TP/(TP+FN); F1 = harmonic mean |
| PR-AUC vs ROC-AUC | PR-AUC more informative for rare positive class |
| Kernel trick | K(x_i,x_j) computes φ(x_i)ᵀφ(x_j) without materializing φ |
| PSI thresholds | <0.1 stable, 0.1–0.25 monitor, >0.25 retrain |
| SHAP axioms | Efficiency, Symmetry, Dummy player, Linearity |
| t-SNE Student-t | Solves crowding problem — heavier tails allow farther embedding |
| AdaBoost weight | α_m = (1/2)ln((1−ε_m)/ε_m) |
| PCA | Eigenvectors of covariance matrix, eigenvalues = variance captured |
