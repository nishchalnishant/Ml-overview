# Algorithms

---

# Q1: How does a Decision Tree algorithm work?

## 1. 🔹 Direct Answer
A **decision tree** recursively **splits** the feature space to minimize **impurity** (Gini, entropy, MSE for regression)—**greedy**, **axis-aligned** partitions. Leaves predict **majority class** or **mean** value.

## 2. 🔹 Intuition
Series of **if-else** rules on features—interpretable flowchart.

## 3. 🔹 Deep Dive
- **CART**: binary splits; cost **O(n d log n)** per level typical implementations.
- **Stopping**: max depth, min samples leaf, min impurity decrease.

## 4. 🔹 Practical Perspective
- **Pros**: nonlinear, mixed types, little preprocessing.
- **Cons**: **high variance**, overfits—use **RF/GBM**.

## 5. 🔹 Code Snippet
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Gini vs entropy? **A:** Similar splits; Gini slightly faster.

## 7. 🔹 Common Mistakes
Unpruned trees on noisy data—memorize training set.

## 8. 🔹 Comparison / Connections
Random Forest, boosting, rule lists.

## 9. 🔹 One-line Revision
Decision trees greedily split on features to reduce impurity—interpretable but unstable alone.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: Explain how Decision Trees make splits and handle categorical features.

## 1. 🔹 Direct Answer
**Splits** search thresholds on ordered features or **groupings** of categories minimizing impurity (**CART** often **one-vs-rest** or **binary partition** of category subsets). **High-cardinality** categoricals: **ordering** by target mean, **native** handling in LightGBM/CatBoost.

## 2. 🔹 Intuition
For categories, algorithm tries to find **best partition** of levels into two children—not always all splits exhaustive for large K.

## 3. 🔹 Deep Dive
- **One-hot** explosion: tree ensembles may use **optimal split** on grouped categories.
- **Missing** values: **surrogate** splits or learn default direction (XGBoost).

## 4. 🔹 Practical Perspective
**CatBoost** targets ordered categoricals; **label encode** carefully for sklearn CART.

## 5. 🔹 Code Snippet
```text
impurity(parent) - weighted_sum(impurity(children)) maximized at each split
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Cardinality 10k? **A:** Target statistics, hashing, or native CatBoost.

## 7. 🔹 Common Mistakes
Assuming sklearn one-hot is required—tree libraries differ.

## 8. 🔹 Comparison / Connections
RuleFit, optimal splitting algorithms.

## 9. 🔹 One-line Revision
Trees optimize impurity reduction per split; categoricals via partitions or native algorithms—watch cardinality.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q3: How does Random Forest work? How does it improve over Decision Trees? How does it reduce variance?

## 1. 🔹 Direct Answer
**Random Forest** trains many **deep trees** on **bootstrap** samples with **random feature subset** per split (**bagging** + **decorrelation**). **Averages** predictions—**reduces variance** via **ensemble** without large bias increase.

## 2. 🔹 Intuition
Many noisy voters → **stable** average; random features make trees **less correlated**.

## 3. 🔹 Deep Dive
- **OOB** error estimate from out-of-bag samples.
- **Variance** reduction ~1/B for uncorrelated estimators (idealized).

## 4. 🔹 Practical Perspective
**n_estimators**, **max_features**, **max_depth** key knobs; **parallel** training.

## 5. 🔹 Code Snippet
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs boosting? **A:** RF parallel, lower risk of overfit sequence; boosting often higher accuracy with tuning.

## 7. 🔹 Common Mistakes
Thinking RF never overfits—very deep trees + noise can still overfit.

## 8. 🔹 Comparison / Connections
Extra Trees, bagging, gradient boosting.

## 9. 🔹 One-line Revision
RF bagging + feature randomness averages many high-variance trees to stabilize predictions.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: Explain Ensemble Methods. Why are they powerful?

## 1. 🔹 Direct Answer
**Ensembles** combine multiple models (**bagging**, **boosting**, **stacking**) to **reduce variance** (bagging), **bias** (boosting), or **blend strengths** (stacking). **Diversity** + **aggregation** beats single model if errors are **uncorrelated**.

## 2. 🔹 Intuition
Wisdom of crowds—if models err differently, average/stack **cancels** noise.

## 3. 🔹 Deep Dive
- **Bias-variance**: bagging ↓ variance; boosting ↓ bias.
- **Diversity** via different data (bagging), algorithms, or features.

## 4. 🔹 Practical Perspective
Kaggle winners almost always **ensembles**—watch **latency** in production.

## 5. 🔹 Code Snippet
```python
from sklearn.ensemble import VotingClassifier
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Stacking? **A:** Meta-learner on base preds—risk of overfit, use CV.

## 7. 🔹 Common Mistakes
Ensembling **identical** models—no gain.

## 8. 🔹 Comparison / Connections
Random forest, XGBoost, neural ensembles.

## 9. 🔹 One-line Revision
Ensembles combine diverse models to reduce error—mechanism depends on bagging vs boosting vs stacking.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: What is the difference between bagging and boosting?

## 1. 🔹 Direct Answer
**Bagging**: **parallel** train on bootstrap samples, **average**—**reduces variance** (RF). **Boosting**: **sequential** models correct **residuals** of previous—**reduces bias**, can **overfit** if not regularized (XGBoost, LightGBM).

## 2. 🔹 Intuition
Bagging = independent voters; boosting = students fixing each other’s mistakes in order.

## 3. 🔹 Deep Dive
- Bagging: **OOB** error; boosting: **learning rate**, **shrinkage**.

## 4. 🔹 Practical Perspective
Boosting often **stronger** on structured data competitions; **tune** depth, eta, rounds.

## 5. 🔹 Code Snippet
```text
bagging: f = (1/M) Σ f_m ; boosting: f = Σ η_m h_m
```

## 6. 🔹 Interview Follow-ups
1. **Q:** AdaBoost vs GBDT? **A:** AdaBoost reweights points; GBDT fits negative gradients.

## 7. 🔹 Common Mistakes
Saying bagging reduces bias primarily—it targets variance.

## 8. 🔹 Comparison / Connections
Stacking, cascading.

## 9. 🔹 One-line Revision
Bagging averages parallel high-variance models; boosting fits sequential weak learners to reduce bias.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What is Gradient Boosting? How does XGBoost work?

## 1. 🔹 Direct Answer
**Gradient boosting** adds trees that predict **negative gradient** of loss (pseudo-residuals). **XGBoost** adds **regularization** (L1/L2 on weights), **approximate** split finding, **column** subsampling, **missing** value handling, **parallel** block structure—**fast**, **strong** default on tabular data.

## 2. 🔹 Intuition
Each new tree fixes **what previous ensemble still gets wrong**—functional gradient descent in tree space.

## 3. 🔹 Deep Dive
- **Second-order** (Hessian) approximation in XGBoost for many losses.
- **Shrinkage** (learning rate) × **n_estimators**.

## 4. 🔹 Practical Perspective
Tune: **max_depth**, **min_child_weight**, **subsample**, **colsample_bytree**, **eta**, **early stopping**.

## 5. 🔹 Code Snippet
```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs LightGBM? **A:** Leaf-wise growth, faster on large data—watch overfitting.

## 7. 🔹 Common Mistakes
No early stopping—overfits with too many trees.

## 8. 🔹 Comparison / Connections
CatBoost (ordered boosting), NGBoost.

## 9. 🔹 One-line Revision
Gradient boosting fits trees to negative gradients with regularization; XGBoost optimizes speed and generalization for tabular data.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q7: What are the key hyperparameters for XGBoost?

## 1. 🔹 Direct Answer
**n_estimators** + **learning_rate** (shrinkage), **max_depth**, **min_child_weight**, **gamma** (min split gain), **subsample**, **colsample_bytree**, **reg_alpha/lambda**, **early_stopping_rounds**. Interaction: **more trees** + **lower** LR often generalizes better.

## 2. 🔹 Intuition
Depth controls **capacity**; sampling adds **randomness** like RF; regularization fights **overfitting**.

## 3. 🔹 Deep Dive
**scale_pos_weight** for imbalance; **max_delta_step** for logistic instability.

## 4. 🔹 Practical Perspective
Start **coarse** grid; use **early stopping** on validation.

## 5. 🔹 Code Snippet
```python
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Tree method hist? **A:** Approximate quantile sketches—faster.

## 7. 🔹 Common Mistakes
Only tuning depth while ignoring **learning rate** and **estimators**.

## 8. 🔹 Comparison / Connections
LightGBM params, CatBoost depth.

## 9. 🔹 One-line Revision
Key XGB knobs: depth, min_child_weight, sampling, regularization, LR×trees with early stopping.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: Explain Gradient Boosting and its advantages over Random Forests.

## 1. 🔹 Direct Answer
**Boosting** **sequentially** reduces **bias** by fitting residuals; often **higher accuracy** on structured data with tuning. **RF** **bagging** reduces **variance**, **parallel**, **robust** default, less tuning—often **worse peak** accuracy than tuned GBM.

## 2. 🔹 Intuition
Boosting **targets errors** aggressively—can overfit; RF **smooths** via averaging.

## 3. 🔹 Deep Dive
- Boosting: **lower bias** risk but needs **regularization** and **early stopping**.
- RF: **OOB** score, **feature importance** (biased toward high card—prefer SHAP).

## 4. 🔹 Practical Perspective
Use **GBM** when squeezing **tabular** performance; **RF** for **baseline** and **interpretability** speed.

## 5. 🔹 Code Snippet
```text
RF: Var(avg) ≈ ρσ² + (1-ρ)σ²/B ; Boosting: additive bias reduction
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When prefer RF? **A:** Small data noise, need parallel train, less tuning time.

## 7. 🔹 Common Mistakes
Claiming one always dominates—dataset and tuning dependent.

## 8. 🔹 Comparison / Connections
AdaBoost, neural nets on tabular (recent).

## 9. 🔹 One-line Revision
Boosting chases residuals for higher accuracy with care; RF averages trees for robust lower-variance baselines.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: Explain how Logistic Regression differs from Linear Regression.

## 1. 🔹 Direct Answer
**Linear regression** predicts **continuous** **y** with linear model + **MSE** loss. **Logistic regression** predicts **class probabilities** with **linear score + sigmoid** and **cross-entropy**—still **linear decision boundary** in **x**.

## 2. 🔹 Intuition
Same linear structure; different **output** and **loss** for classification vs regression.

## 3. 🔹 Deep Dive
Logistic = **GLM** with logit link; **MLE** Bernoulli.

## 4. 🔹 Practical Perspective
Logistic gives **calibrated** probabilities with **L2** often good baseline.

## 5. 🔹 Code Snippet
```python
from sklearn.linear_model import LogisticRegression, LinearRegression
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multinomial? **A:** Softmax + CE—still linear in x.

## 7. 🔹 Common Mistakes
Using linear regression on 0/1 labels as if continuous.

## 8. 🔹 Comparison / Connections
Probit, linear SVM.

## 9. 🔹 One-line Revision
Linear regression is for continuous targets; logistic regression is linear classifier with log loss.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q10: How does logistic regression work?

## 1. 🔹 Direct Answer
Model **p(y=1|x) = σ(wᵀx + b)**. Fit **w** by minimizing **cross-entropy**—convex, **iterative** (LBFGS, SGD) or **IRLS** for exact Newton steps.

## 2. 🔹 Intuition
Linear **log-odds** model; coefficients are **log-OR** per feature holding others fixed (interpretable).

## 3. 🔹 Deep Dive
**L1** gives **sparse** features; **L2** for correlated features.

## 4. 🔹 Practical Perspective
**Multicollinearity** inflates variance—**regularize**; **calibration** often good.

## 5. 🔹 Code Snippet
```python
LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Imbalance? **A:** `class_weight='balanced'` or scale_pos_weight analog.

## 7. 🔹 Common Mistakes
Perfect separation → coefficients explode—**regularize**.

## 8. 🔹 Comparison / Connections
GLMs, maximum entropy.

## 9. 🔹 One-line Revision
Logistic regression fits linear logit with convex cross-entropy—strong calibrated linear baseline.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: Explain R-squared and adjusted R-squared.

## 1. 🔹 Direct Answer
**R²** = 1 − SS_res/SS_tot—fraction of variance **explained** vs mean baseline. **Adjusted R²** penalizes **extra predictors**: \(1 - (1-R²)(n-1)/(n-p-1)\)—rises only if new term beats **expected** noise.

## 2. 🔹 Intuition
Plain R² **never decreases** when adding features—adjusted punishes **complexity**.

## 3. 🔹 Deep Dive
Can be **negative** if model worse than mean on small samples.

## 4. 🔹 Practical Perspective
Use adjusted for **comparing** models with different **p**; not for causality.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Nonlinear models? **A:** R² still defined but not “variance explained” same way.

## 7. 🔹 Common Mistakes
Claiming high R² implies causal relationship.

## 8. 🔹 Comparison / Connections
Pearson correlation squared (simple linear), AIC/BIC.

## 9. 🔹 One-line Revision
R² measures explained variance; adjusted R² penalizes spurious predictors.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q12: How do you check for multicollinearity in regression models?

## 1. 🔹 Direct Answer
**VIF** (variance inflation factor) per feature—**VIF > 5–10** suggests problematic collinearity. **Correlation matrix**, **condition number** of **XᵀX**. **Remedies**: **drop** redundant, **combine**, **ridge** regression, **PCA**.

## 2. 🔹 Intuition
Collinear features make **coefficients unstable**—small data changes flip signs.

## 3. 🔹 Deep Dive
**VIF_j** = 1/(1 − R²_j) where R²_j is from regressing feature j on others.

## 4. 🔹 Practical Perspective
**Interpretation** suffers even if prediction OK—watch **standard errors**.

## 5. 🔹 Code Snippet
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Trees affected? **A:** Less for prediction; **importance** biased.

## 7. 🔹 Common Mistakes
Removing features solely by correlation without domain.

## 8. 🔹 Comparison / Connections
Ridge, elastic net, identifiability.

## 9. 🔹 One-line Revision
Use VIF and domain knowledge; regularize or reduce dimension when predictors are redundant.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: How does K-Nearest Neighbors (KNN) work?

## 1. 🔹 Direct Answer
Classify (or regress) by **majority vote** or **average** of **k** closest training points by distance metric—**non-parametric**, **instance-based**.

## 2. 🔹 Intuition
Decision boundary adapts **locally**—smooth when k large, jagged when k small.

## 3. 🔹 Deep Dive
**Complexity**: naive query O(nd); **approximate** NN for scale.

## 4. 🔹 Practical Perspective
**Scale features**; **choose k** by CV; **curse of dimensionality**.

## 5. 🔹 Code Snippet
```python
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(n_neighbors=15, metric="minkowski", p=2)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Weighted? **A:** `weights='distance'`—closer neighbors matter more.

## 7. 🔹 Common Mistakes
k=1 on noisy data—overfits.

## 8. 🔹 Comparison / Connections
Parzen windows, kernel density.

## 9. 🔹 One-line Revision
KNN votes over k nearest neighbors—scale features and tune k; watch high-dimensional distance concentration.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: Explain K-Means Clustering. How does it work? Limitations?

## 1. 🔹 Direct Answer
Partition **k** centroids; alternate **assign** points to nearest centroid, **update** centroid as mean—minimize within-cluster SS. **Limitations**: needs **k**, **spherical** clusters, **sensitive** to init/outliers, **local minima**—use **k-means++**, **multiple restarts**.

## 2. 🔹 Intuition
Voronoi cells around centers; like compressing data to k templates.

## 3. 🔹 Deep Dive
**NP-hard** globally; Lloyd’s heuristic converges to **local** optimum.

## 4. 🔹 Practical Perspective
**Mini-batch** k-means for scale; **GMM** if elliptical clusters.

## 5. 🔹 Code Snippet
```python
from sklearn.cluster import KMeans
KMeans(n_clusters=8, n_init="auto", init="k-means++")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Elbow? **A:** Heuristic for k—domain validation better.

## 7. 🔹 Common Mistakes
Choosing k only by silhouette without business interpretability.

## 8. 🔹 Comparison / Connections
Hierarchical clustering, spectral clustering.

## 9. 🔹 One-line Revision
k-means alternates assignment and centroid updates—fast but assumes spherical clusters and known k.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: Explain Support Vector Machines (SVM). What is the kernel trick?

## 1. 🔹 Direct Answer
**SVM** finds **maximum-margin** hyperplane separating classes (soft margin allows slack). **Kernel trick** replaces **xᵀx’** with **k(x,x’)=φ(x)ᵀφ(x’)** implicitly in high (even infinite) **φ** space—**nonlinear** boundaries without explicit **φ**.

## 2. 🔹 Intuition
Margin maximization improves **generalization**; kernels **bend** the space cheaply.

## 3. 🔹 Deep Dive
- **RBF** kernel: \( \exp(-\gamma \|x-x'\|^2) \).
- **Support vectors** are critical points on margin.

## 4. 🔹 Practical Perspective
**Scale** features; **O(n²–n³)** training—**not** for millions of points; **LinearSVM** for big sparse text.

## 5. 🔹 Code Snippet
```python
from sklearn.svm import SVC
SVC(kernel="rbf", C=1.0, gamma="scale")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** C vs margin? **A:** Small C → wider margin, more misclassification allowed.

## 7. 🔹 Common Mistakes
Using RBF on huge data without approximation.

## 8. 🔹 Comparison / Connections
Kernel ridge, representer theorem.

## 9. 🔹 One-line Revision
SVM maximizes margin; kernels implicitly map to rich feature spaces for nonlinear separation.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q16: What is the decision boundary in classifiers?

## 1. 🔹 Direct Answer
The **decision boundary** is the **surface** in feature space where **classifier switches** prediction (e.g., **wᵀx + b = 0** for linear). **Shape** reflects model: **linear**, **piecewise** (trees), **smooth** (kernels).

## 2. 🔹 Intuition
It’s the **border** between regions “class A” vs “class B.”

## 3. 🔹 Deep Dive
For calibrated probabilities, boundary often at **p=0.5**—can shift with **cost-sensitive** threshold.

## 4. 🔹 Practical Perspective
Visualize **2D** projections; **high-d** boundaries hard—trust **metrics**.

## 5. 🔹 Code Snippet
```text
logistic: w·x + b = 0 ; RBF SVM: implicit nonlinear surface
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Multiclass? **A:** One-vs-rest boundaries or softmax single surface in score space.

## 7. 🔹 Common Mistakes
Thinking neural nets always have smooth boundaries—ReLU nets are piecewise linear.

## 8. 🔹 Comparison / Connections
VC dimension, margin theory.

## 9. 🔹 One-line Revision
Decision boundary separates predicted classes—geometry encodes model family and threshold.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: Explain Naive Bayes.

## 1. 🔹 Direct Answer
**Naive Bayes** applies **Bayes’ rule** with **conditional independence** of features given class—**fast**, **closed-form** for discrete/Gaussian variants. Predicts **argmax_y P(y) Π P(x_i|y)**.

## 2. 🔹 Intuition
“Naive” because features rarely independent—but often **works** for text as **baseline**.

## 3. 🔹 Deep Dive
- **Multinomial** NB for word counts; **Gaussian** for continuous.
- **Laplace smoothing** avoids zero probabilities.

## 4. 🔹 Practical Perspective
**Log** space for numerical stability; strong when **little data**, **high-d** sparse text.

## 5. 🔹 Code Snippet
```python
from sklearn.naive_bayes import MultinomialNB
MultinomialNB(alpha=1.0)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Violation of independence? **A:** Still can work; dependencies hurt calibration.

## 7. 🔹 Common Mistakes
No smoothing—zero probabilities kill product.

## 8. 🔹 Comparison / Connections
Logistic regression, generative vs discriminative.

## 9. 🔹 One-line Revision
Naive Bayes is fast generative classifier assuming feature independence given class—great text baseline.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: What is Dimensionality Reduction?

## 1. 🔹 Direct Answer
**Dimensionality reduction** maps **high-dimensional** data to **fewer** dimensions—**compress**, **denoise**, **visualize**, **speed** training. **Linear** (PCA, SVD) or **nonlinear** (t-SNE, UMAP, autoencoders).

## 2. 🔹 Intuition
Most variance often lives on **low** subspace—drop weak directions.

## 3. 🔹 Deep Dive
**Curse of dimensionality**: distances less meaningful in high-d—reduction can help **generalization**.

## 4. 🔹 Practical Perspective
**t-SNE/UMAP** for viz only—**distances** distorted; **PCA** for **preprocessing** pipeline.

## 5. 🔹 Code Snippet
```python
from sklearn.decomposition import PCA
PCA(n_components=50).fit_transform(X)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Supervised reduction? **A:** LDA (maximize class separation).

## 7. 🔹 Common Mistakes
Fitting PCA on full dataset before CV—**leakage**.

## 8. 🔹 Comparison / Connections
Feature selection, manifold learning.

## 9. 🔹 One-line Revision
Dimensionality reduction finds compact representations—supervised vs unsupervised goals differ.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: Explain PCA (Principal Component Analysis). How does it work? When would you use it?

## 1. 🔹 Direct Answer
**PCA** finds **orthogonal** directions (**principal components**) maximizing **variance** of projected data—**eigenvectors** of **covariance** matrix; **eigenvalues** = variance explained. Use for **compression**, **denoising**, **collinearity** reduction, **visualization** (first 2–3 PCs).

## 2. 🔹 Intuition
Rotate axes to where data **spreads** most—drop axes with little spread.

## 3. 🔹 Deep Dive
**SVD** on **centered X** numerically stable; choose **k** by **scree** plot or **95%** variance.

## 4. 🔹 Practical Perspective
**Scale** features first; **interpretability** of PCs mixed—**loadings** help.

## 5. 🔹 Code Snippet
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # variance ratio
Z = pca.fit_transform(X)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** PCA for classification? **A:** Unsupervised—may not align with class labels (use LDA).

## 7. 🔹 Common Mistakes
Not centering data before PCA.

## 8. 🔹 Comparison / Connections
Random projection, kernel PCA.

## 9. 🔹 One-line Revision
PCA is orthogonal projection maximizing variance—use for denoising and dimension reduction, not labels.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q20: Explain Gradient Descent and its variants.

## 1. 🔹 Direct Answer
**GD** variants differ by **how much data** per step and **momentum**: **batch** (full), **mini-batch SGD**, **SGD+momentum**, **Nesterov**, **AdaGrad/RMSprop/Adam** (adaptive per-parameter). Trade-off: **noise**, **speed**, **memory**.

## 2. 🔹 Intuition
Same descent idea—different **step rules** and **gradient estimates**.

## 3. 🔹 Deep Dive
**Adam** combines momentum + RMSprop-like scaling; **second-order** (L-BFGS) for small problems.

## 4. 🔹 Practical Perspective
Deep nets: **AdamW** + **cosine** schedule common; **SGD+momentum** can generalize better with tuning.

## 5. 🔹 Code Snippet
```python
torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)
torch.optim.AdamW(params, lr=3e-4)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Newton method? **A:** Uses Hessian—accurate but O(d²) per step—impractical for large nets.

## 7. 🔹 Common Mistakes
Using default Adam for all problems without LR tuning.

## 8. 🔹 Comparison / Connections
See optimization.md for individual optimizers.

## 9. 🔹 One-line Revision
Gradient descent family trades full vs stochastic gradients and adaptive vs fixed learning rates—match optimizer to problem scale.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: What is the ROC-AUC curve, and how is it interpreted?

## 1. 🔹 Direct Answer
**ROC** plots **TPR vs FPR** across thresholds; **AUC** is area under curve = **P**(random positive scores &gt; random negative). **0.5** = random; **1.0** = perfect ranker. **Threshold-free** summary of **ranking** ability.

## 2. 🔹 Intuition
Higher AUC = better **separation** of score distributions between classes.

## 3. 🔹 Deep Dive
**Equivalent** to Mann-Whitney U statistic; **insensitive** to class **prevalence** in a specific sense vs PR.

## 4. 🔹 Practical Perspective
Imbalanced positives: also report **PR-AUC**; pick **operating point** on ROC for business costs.

## 5. 🔹 Code Snippet
```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_score)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Class imbalance? **A:** ROC can look optimistic—check PR.

## 7. 🔹 Common Mistakes
Using with **misaligned** scores (not comparable across models without calibration).

## 8. 🔹 Comparison / Connections
PR curve, lift, C-statistic.

## 9. 🔹 One-line Revision
ROC-AUC measures ranking quality across thresholds—pair with PR and calibration for imbalanced problems.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
