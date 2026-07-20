---
module: Classical ML
topic: Ml
subtopic: Algorithms
status: unread
tags: [interviewprep, ml, ml-algorithms, interview-framing]
---
# ML Algorithms

---

## 1. Decision Trees

**What the interviewer is testing**: can you explain a model by the problem it solves rather than just how it works mechanically? Most candidates describe the algorithm. The interesting question is "why would you ever split a node, and when would you stop?"

**The reasoning structure**: a decision tree solves one specific problem — how do you make predictions when data has complex, non-linear structure but you need something explainable? The answer is to recursively ask yes/no questions that partition the feature space into regions that are progressively more pure. "Pure" means the region mostly contains one class (classification) or has low variance in the target (regression). Every split is just choosing which question best reduces impurity in the current region.

The reason trees overfit is the same reason they are flexible: you can keep splitting until every leaf contains exactly one training point. That is perfect memorization, not learning. The cure is to stop splitting early or prune after the fact.

**The pattern in action**: "I have 100 customers — 60 churn, 40 do not. My first split on `monthly_spend < $50` gives left node: 55 churn out of 60 low-spenders, right node: 5 churn out of 40 high-spenders. Both children are far purer than the root. That is the entire logic: keep finding questions that reduce impurity until the regions are pure enough or the tree is too deep."

**Common traps**:
- confusing the splitting criterion (Gini vs entropy) with the stopping criterion (max_depth, min_samples_leaf) — these solve different problems; the splitting criterion chooses which question to ask, the stopping criterion decides when to stop asking
- assuming trees need feature scaling — they do not, because splits depend only on rank order of feature values, not magnitude
- using a single tree in production without acknowledging its instability — the same dataset with one row changed can produce a completely different tree structure

**Formulas to have cold**:

$$\text{Gini}(t) = 1 - \sum_{c=1}^{C} p_c^2$$

$$H(t) = -\sum_{c=1}^{C} p_c \log_2 p_c$$

$$\text{Information Gain} = H(t) - \sum_{k \in \{\text{left, right}\}} \frac{|t_k|}{|t|} H(t_k)$$

Gini is faster to compute (no log). Both choose the same split over 95% of the time. sklearn defaults to Gini for classification, MSE for regression.

---

## 2. Random Forest

**What the interviewer is testing**: do you understand ensemble methods as a principled solution to a specific problem, or did you just memorize "trains many trees and votes"? The useful question is "what specific failure mode of decision trees does Random Forest fix, and why does the fix work?"

**The reasoning structure**: a single decision tree has high variance — retrain it on a slightly different sample and you get a completely different tree. High variance is the signature of a model that is memorizing rather than generalizing. The fix for variance is averaging: the average of many noisy estimates is less noisy than any single estimate.

But if you train many trees on the same data, they will make the same mistakes and averaging them changes nothing. The key insight is that the trees must be decorrelated. You achieve this in two ways: bootstrap sampling (each tree sees a different ~63% of the data) and random feature subsets (at each node, only consider $\sqrt{p}$ of the $p$ features). Now trees disagree in useful ways, and averaging them reduces variance without systematically increasing bias.

**The pattern in action**: "My single decision tree gets 90% training accuracy and 74% validation accuracy — classic high variance. I switch to a Random Forest with 200 trees. Training accuracy drops to ~85% (individual trees can no longer fully memorize) but validation accuracy rises to 83%. The averaging smoothed out memorized noise. The small increase in bias is worth the large reduction in variance."

**Common traps**:
- thinking more trees always helps proportionally — variance reduction has sharply diminishing returns after ~100–200 trees; compute cost keeps growing but accuracy improvement stops
- not recognizing that Random Forest still overfits on noisy datasets, just less so than single trees
- forgetting that `max_features` controls the decorrelation-accuracy tradeoff: smaller values produce more decorrelated trees (lower variance, higher individual tree bias); the default `sqrt(p)` is usually a good balance

**Key parameters**:
- `n_estimators`: more is better up to ~200; after that, gains are marginal
- `max_features`: `sqrt(p)` for classification, `p/3` for regression
- `max_depth` or `min_samples_leaf`: limits individual tree complexity

Out-of-bag error: each tree was trained on ~63% of data. The remaining ~37% is a free validation set for that tree. Average OOB error across all trees gives a reliable validation estimate without holding out data separately.

---

## 3. Bagging vs Boosting

**What the interviewer is testing**: can you distinguish the two ensemble strategies at the level of what error component each one targets? The answer that separates candidates is not "bagging averages, boosting sequences" — it is understanding which component of the bias-variance decomposition each one reduces.

**The reasoning structure**: every model's expected error decomposes into bias and variance. Bagging and boosting attack different components.

Bagging is the answer to "my model is too sensitive to which training examples I happened to get." Train the same model type on many bootstrapped samples and average. Averaging reduces variance because errors that are random (different on each bootstrap) cancel out. Errors that are systematic (the model is consistently wrong in some direction) do not cancel — bagging cannot reduce bias.

Boosting is the answer to "my model is too weak — it is systematically wrong in predictable ways." Train a sequence of weak models where each one focuses on the examples the previous models got wrong. The ensemble's effective bias decreases because each new model corrects a specific systematic error of the current ensemble.

**The pattern in action**: "My logistic regression has training accuracy 72% and validation accuracy 70% — both low. This is a bias problem. Bagging 100 logistic regressions will not help — the average of 100 weak models is still a weak model. Boosting weak learners sequentially is the right move: each weak learner corrects the residuals of the ensemble so far, and the combined bias shrinks."

**Common traps**:
- applying bagging to a high-bias model hoping for improvement — it will not come; bagging only reduces variance
- applying boosting to a high-variance problem without regularization — boosting can dramatically overfit noisy data because it keeps trying to "correct" mistakes that are actually noise, eventually memorizing the noise
- confusing boosting with stacking — boosting trains sequentially on residuals from the same base learner type; stacking trains a meta-model on the outputs of heterogeneous base models

---

## 4. Gradient Boosting and XGBoost

**What the interviewer is testing**: do you understand gradient boosting as gradient descent in function space, not just as "trees that fix each other's mistakes"? The deeper question is "why fit to residuals specifically?"

**The reasoning structure**: gradient boosting is gradient descent, but instead of updating parameters, you are updating the prediction function itself. At each step, ask: "in which direction should I move my current predictions to decrease the loss?" That direction is the negative gradient of the loss with respect to the current predictions. For MSE loss, the negative gradient at each point is the residual $(y_i - \hat{y}_i)$. So "fit the next tree to residuals" is not an ad hoc trick — it is gradient descent applied directly to the prediction function.

XGBoost extends this by using the second-order Taylor expansion of the loss. The second-order term (Hessian) gives curvature information, allowing XGBoost to weight mistakes more accurately and find better splits. It also adds explicit regularization on leaf weights and the number of leaves.

**The pattern in action**: "Start with a constant prediction (the mean). Compute residuals. Fit a shallow tree to those residuals. The new ensemble prediction is: constant + learning_rate × tree_prediction. Residuals shrink. Repeat. Each tree explains a bit more of the remaining error. The sequence of trees is a sequence of gradient descent steps in function space."

**Common traps**:
- tuning only `n_estimators` while ignoring `learning_rate` and `max_depth` — these three interact: smaller learning rate requires more trees; deeper trees risk overfitting each gradient step
- using the default learning rate (0.3 in XGBoost) in production — it is too large for most problems; 0.01–0.1 with early stopping is more robust
- not understanding that XGBoost's `lambda` and `alpha` regularize the leaf weights, not feature weights — this is different from L1/L2 in linear models

**XGBoost internals**:

Objective at step $m$: $F_m(x) = F_{m-1}(x) + \alpha h_m(x)$ where $h_m$ is fit to pseudo-residuals:
$$r_i = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

Split gain criterion (what XGBoost maximizes at each node):
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

where $G$ = sum of first-order gradients, $H$ = sum of second-order gradients, $\lambda$ = L2 regularization, $\gamma$ = minimum gain threshold.

---

## 5. Random Forest vs XGBoost

**What the interviewer is testing**: can you reason about algorithm selection based on properties of the problem, not just say "XGBoost usually wins"?

**The reasoning structure**: these two algorithms differ on a fundamental axis — robustness versus power. Random Forest builds independent trees in parallel and is resistant to overfitting on clean data because the averaging is inherently conservative. XGBoost builds trees sequentially, each correcting the last, and can model more complex patterns — but because it actively minimizes residuals, it will eventually model noise if you let it. The additional power comes with additional tuning responsibility.

**The pattern in action**:
- Noisy data, quick baseline needed: Random Forest. Robust by construction, works with minimal tuning.
- Structured data competition with time to tune: XGBoost/LightGBM. The additional power justifies the tuning cost.
- Production system with limited monitoring: Random Forest is safer. Badly tuned XGBoost overfits the training distribution more tightly and can silently degrade on distribution shift.

**Common traps**:
- assuming XGBoost always wins — on small, noisy datasets, a well-configured Random Forest often outperforms XGBoost
- treating these as fundamentally different algorithm families — both are tree ensembles; the difference is sequential vs parallel training, not model family
- forgetting that XGBoost is more sensitive to outliers than Random Forest — XGBoost fits residuals, so a single extreme residual can dominate a tree; Random Forest's averaging is more resilient

---

## 6. Logistic Regression

**What the interviewer is testing**: do you understand why a linear model with a sigmoid output is appropriate for binary classification, or do you just know the formula? The key insight is what "linear" means in logistic regression.

**The reasoning structure**: a probability cannot be negative or greater than 1. A linear combination of features can be anything from $-\infty$ to $+\infty$. You cannot directly model probability as a linear function. The solution: model the log-odds (logit) as a linear function. Log-odds can range over all reals. Then map back to probability via the sigmoid.

This means logistic regression is linear in log-odds space, not in probability space. The decision boundary (where predicted probability crosses 0.5) is a linear surface in feature space. If the true boundary is non-linear, logistic regression will be systematically wrong regardless of how much data you give it — this is a bias problem, not a variance problem, and cannot be fixed by adding more data.

**The pattern in action**: "I have a fraud model. Fraud probability increases with transaction amount, but not linearly — it jumps sharply above $1000. If I add a binary feature `is_high_value = (amount > 1000)`, I am providing the non-linearity that logistic regression needs. Without this feature, the model will plateau no matter how much data I add, because the true boundary is not linear in raw feature space."

**Common traps**:
- using logistic regression when the decision boundary is non-linear and failing to add appropriate feature engineering — the model will plateau at a performance ceiling that cannot be broken by adding data
- forgetting that sklearn's `C` is the inverse of regularization strength — low C means strong regularization, high C means weak regularization; this convention trips people up constantly
- confusing well-calibrated with accurate — logistic regression produces naturally calibrated probabilities; calibration and accuracy are independent properties

**Formulas**:
$$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^Tx+b)}}$$

$$\log \frac{P(y=1|x)}{P(y=0|x)} = w^T x + b \quad \text{(linear in log-odds)}$$

Loss: Binary cross-entropy. Gradient: $\frac{1}{n} X^T (\hat{p} - y)$

---

## 7. Linear vs Logistic Regression

**What the interviewer is testing**: whether you understand the choice of output activation as a consequence of the output type, not as an arbitrary design decision.

**The reasoning structure**: the fundamental question is "what does the output represent?" For a continuous target (price, temperature, count), you want unbounded real outputs — linear regression's $w^Tx + b$ is exactly right. For a binary class, you want a probability between 0 and 1 — you need to squash the linear output through the sigmoid function.

Both are linear models in the sense that the decision boundary is linear in feature space. Logistic regression's boundary is where $w^Tx + b = 0$, which is a hyperplane. The sigmoid just maps the two sides of that hyperplane to different probability ranges.

**Common traps**:
- treating logistic regression as "linear regression with a threshold" — the sigmoid is applied before any threshold, not used as the threshold itself; the threshold is 0.5 on the sigmoid output, which corresponds to 0 on the linear output
- forgetting that logistic regression is linear in log-odds, not in probability — when explaining why the model struggles on a dataset, the distinction between non-linear in probability vs non-linear in log-odds matters
- using MSE as the loss for logistic regression — MSE with a sigmoid output creates a non-convex loss surface with poor gradients near saturation; cross-entropy is convex and has well-behaved gradients everywhere

---

## 8. SVM and the Kernel Trick

**What the interviewer is testing**: whether you understand SVMs as solving a specific geometric problem (maximum margin), and the kernel trick as a computational shortcut rather than just "going to higher dimensions."

**The reasoning structure**: most linear classifiers find some boundary that separates the classes. SVMs ask a more specific question: among all separating boundaries, which one has the largest margin? The margin is the distance from the boundary to the nearest training points on each side. Maximum margin gives you the boundary that is least sensitive to noise near the decision surface — a principled geometric criterion for choosing among infinitely many valid separators.

The kernel trick addresses the problem that real data is often not linearly separable. You want to map data to a higher-dimensional space where it becomes separable. But explicitly computing that mapping can be expensive or infinite-dimensional. The key observation: the SVM's decision function only depends on dot products between data points. If you can compute $K(x_i, x_j)$ — a function that equals the dot product in the high-dimensional space without explicitly going there — you get the separation benefit at the cost of just computing pairwise kernel values.

**The pattern in action**: "Text documents as TF-IDF vectors are high-dimensional but sparse. SVM with a linear kernel works well here — the data is already in a space where linear separation is feasible. For image classification with pixel features, an RBF kernel implicitly maps to a space where non-linear spatial patterns become linearly separable."

**Common traps**:
- confusing the $C$ parameter with regularization in the usual sense — high C means the model is penalized heavily for misclassifications, so it finds a narrow margin that closely fits training data (high variance); low C is more permissive and produces a wider margin (higher bias)
- not recognizing that SVMs scale as $O(n^2)$ to $O(n^3)$ in training examples — impractical for large datasets
- forgetting that only support vectors (points at the margin boundary) determine the decision function — the rest of the training data is irrelevant once training is complete

**Formulas**:
Soft-margin primal: $\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i$

Kernel: $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ — computed without explicitly computing $\phi$.

| Kernel | When to use |
| :--- | :--- |
| Linear | High-dimensional, sparse (text) |
| RBF | General-purpose non-linear default |
| Polynomial | When polynomial relationships are expected |

---

## 9. K-Nearest Neighbors

**What the interviewer is testing**: whether you understand KNN as a non-parametric method and what that means for its failure modes at scale.

**The reasoning structure**: KNN has no training phase — it memorizes the entire training set and makes predictions by looking up neighbors at inference time. This is simultaneously its biggest strength and biggest weakness. No training means you can always add new data without retraining. But inference cost scales with dataset size, and in high dimensions, "nearest" becomes meaningless because all pairwise distances converge (curse of dimensionality).

**The pattern in action**: "I have a recommendation system where users are represented as vectors. KNN works fine at 10,000 users. At 10 million users, inference latency is unacceptable, and in 512-dimensional embedding space the distance metric has become unreliable — all users are approximately equidistant from any query user. I need approximate nearest neighbor search (FAISS, HNSW) or a parametric model that compresses the distance computation into parameters."

**Common traps**:
- not normalizing features before KNN — a feature with range 0–1000 and a feature with range 0–1 will cause the first feature to dominate all distance computations, effectively making KNN use only that feature as its decision rule
- choosing K without thinking about the bias-variance axis — small K = low bias, high variance (sensitive to noise); large K = high bias, low variance (blurs true decision boundaries). K is a hyperparameter that should be chosen by validation, not set to a default
- treating KNN as a production model when it is really a baseline and a sanity check

---

## 10. K-Means

**What the interviewer is testing**: do you understand K-Means as solving a specific optimization problem with known failure modes, or do you just know it "groups similar points together"?

**The reasoning structure**: K-Means minimizes within-cluster sum of squared distances (WCSS). The algorithm alternates between two steps: assign points to their nearest centroid, then recompute centroids as the mean of assigned points. This is coordinate descent on the WCSS objective — each step reduces the objective monotonically, so it converges, but not necessarily to the global minimum.

The failure modes all trace back to one implicit assumption: K-Means assumes clusters are spherical, similar in size, and similar in density. When clusters are elongated, different in size, or have complex shapes, K-Means will cut them in geometrically wrong ways because it is minimizing Euclidean distance from a centroid.

**The pattern in action**: "I run K-Means with K=3 on customer data. The silhouette score is 0.3 — low, meaning points are not much closer to their own cluster than to the nearest other cluster. I visualize and see one elongated cluster that spans a wide range. K-Means is forced to cut this elongated structure with a perpendicular bisector between centroids. DBSCAN or a Gaussian Mixture Model would respect the actual shape."

**Common traps**:
- using the elbow method mechanically without checking if an elbow actually exists — the elbow often does not exist cleanly, and forcing an interpretation produces meaningless clusters
- forgetting that K-Means results depend strongly on initialization — always run multiple times with different seeds and keep the best WCSS; K-Means++ initialization starts with spread-out centroids and helps significantly
- not recognizing that K-Means is sensitive to outliers — one extreme point can pull a centroid far from the cluster's true center, distorting all cluster assignments

**Formulas**:
$$\min_{\{C_k\}} \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$

Silhouette: $s = \frac{b - a}{\max(a, b)}$, where $a$ = mean intra-cluster distance, $b$ = mean nearest-cluster distance. Range [-1, 1]; higher is better.

---

## 11. Naive Bayes

**What the interviewer is testing**: whether you understand why an algorithm with an obviously wrong assumption (conditional independence of features) can still work well in practice, and when it fails.

**The reasoning structure**: Naive Bayes applies Bayes' theorem and then makes the "naive" assumption that features are conditionally independent given the class. This is almost never true — word frequencies in text are correlated, pixels in images are correlated. But it does not matter for classification. You do not need a well-calibrated joint probability to make correct class predictions. As long as the relative ordering of class probabilities is correct, you get the right classification. The naive assumption simplifies the computation from exponential to linear in the number of features — and for sparse data like text, this is surprisingly effective.

**The pattern in action**: "I classify customer support tickets into 50 categories. Naive Bayes trains in milliseconds on 100,000 examples with a 10,000-word vocabulary. The independence assumption is wrong — 'account' and 'locked' co-occur frequently. But the classifier correctly routes tickets because the combination of multiple related words overwhelmingly indicates one category, even if the probability magnitude is wrong."

**Common traps**:
- using Naive Bayes when you need well-calibrated probabilities — the raw outputs are often poorly calibrated because the independence assumption inflates confidence in the most likely class
- not applying Laplace smoothing in Multinomial Naive Bayes — one unseen word in a test document can zero out the entire class probability without smoothing
- using Gaussian Naive Bayes on features with heavy-tailed or multi-modal distributions without transformation

**Formula**:
$$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{j=1}^d \log P(x_j \mid y) \right]$$

---

## 12. PCA

**What the interviewer is testing**: whether you understand PCA as solving a specific variance-preservation problem, and its limitation as an unsupervised method that ignores the prediction target.

**The reasoning structure**: PCA asks "if I have to represent my data in fewer dimensions, which directions should I keep?" Its answer: keep the directions of maximum variance. This makes sense if you believe variance corresponds to signal. It is wrong when high-variance directions correspond to noise, or when low-variance directions are highly predictive of the target.

That last point is crucial and often missed: PCA is unsupervised. It does not know what you are trying to predict. Directions of maximum variance are not necessarily the most predictive directions. If the most predictive feature has low variance (a rare binary flag, a small but highly informative signal), PCA will discard it.

**The pattern in action**: "I have 200 features and run PCA. The first 20 components capture 95% of the variance, so I use those. My model performance drops compared to using all features. Why? Some of the remaining variance — the 5% spread across 180 components — contained a highly predictive signal that just had low variance. Explained variance is the wrong criterion for supervised learning; validation performance is the right one."

**Common traps**:
- not standardizing features before PCA — features with larger numerical ranges will dominate the principal components regardless of their actual information content
- treating explained variance as the sole criterion for choosing the number of components — for supervised learning, choose components based on downstream model performance on a validation set
- confusing PCA with feature selection — PCA creates new features as linear combinations of the originals; the components are not interpretable as individual original features

**Steps**: center data → compute covariance matrix $\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$ → eigendecompose → project onto top $k$ eigenvectors.

Explained variance ratio for component $k$: $\frac{\lambda_k}{\sum_j \lambda_j}$

---

## 13. Dimensionality Reduction: The Broader Picture

**What the interviewer is testing**: whether you can articulate why dimensionality reduction is needed beyond "too many features," and what different methods trade off.

**The reasoning structure**: the curse of dimensionality is the real problem. In high dimensions, any two points are roughly the same distance apart. This breaks distance-based reasoning (KNN, SVMs with RBF kernel), makes clustering meaningless (every point is equidistant from every centroid), and requires exponentially more data to densely cover the feature space. Dimensionality reduction is not just about computation — it is about making the geometry of the data meaningful again.

Different methods make different tradeoffs: PCA preserves global linear structure. t-SNE preserves local neighborhood structure and is excellent for visualization but distorts global distances — it is not suitable as input to a downstream model. UMAP balances local and global structure and is faster than t-SNE. Autoencoders learn non-linear compressions.

**Common traps**:
- using t-SNE embeddings as features for a downstream model — t-SNE is stochastic, does not preserve distances faithfully, produces different results each run, and is designed purely for visualization
- applying dimensionality reduction before the train/test split — the reduction transform (e.g., PCA) must be fit on training data only; fitting on the full dataset leaks test set distribution information into the transform

---

## 14. Algorithm Selection: The Decision Framework

**What the interviewer is testing**: can you choose the right tool based on problem properties rather than defaulting to whatever is trendy?

**The reasoning structure**: algorithm selection is a constraint-satisfaction problem. The constraints are: data type, data size, latency requirements at inference, interpretability requirements, and available compute. The question to ask is "what assumptions does this algorithm make, and does my data satisfy them?"

| Problem type | Strong first choice | Why |
| :--- | :--- | :--- |
| Tabular, structured, mixed types | XGBoost / LightGBM | Handles mixed types, missing values, non-linearity natively |
| Tabular, need interpretability | Logistic regression + feature engineering | Linear, coefficients are meaningful |
| Images | Pretrained CNN / ViT fine-tuned | Representation learning vastly outperforms hand features |
| Text classification | Fine-tuned language model | Pretrained context representations |
| Anomaly detection, no labels | Isolation Forest, DBSCAN, Autoencoder | Unsupervised, not dependent on labeled anomaly examples |
| Small dataset, arbitrary structure | KNN or SVM with RBF | Non-parametric or kernel-based, flexible |

**Common traps**:
- jumping to a neural network for tabular data — on most structured tabular datasets, gradient boosted trees outperform neural networks with substantially less tuning effort
- not starting with a simple baseline — a logistic regression or a single decision tree establishes the performance floor for simple models and tells you whether more complex models are buying anything; skipping this step makes it impossible to know whether added complexity is worth the cost
