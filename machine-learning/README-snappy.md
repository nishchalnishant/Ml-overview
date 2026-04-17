# Machine Learning foundations

**Who this is for:** You already think in systems — pipelines, contracts, failure modes. Classical ML is the same game: pick a **representation** (features), pick a **hypothesis class** (algorithm family), and know **what breaks** when the world changes.

**Azure / DevOps bridge:** *Training* ≈ your build that produces an artifact. *Cross-validation* ≈ running the build against multiple “environments” (folds). *Regularization* ≈ guardrails so the artifact doesn’t overfit to **staging** (training) and flop in **prod** (test/real world).

---

## 1. Algorithm map (where things live)

Knowing the **math tribe** helps you defend choices in a design review — same as picking SQL vs. a stream processor for the job.

### Summary table

| Category | Math vibe | Examples | Shines when… |
| :--- | :--- | :--- | :--- |
| **Linear** | Hyperplanes, smooth boundaries | Linear / logistic regression, SVM, PCA | You want interpretability or strong baselines in high dimensions. |
| **Tree / partition** | Axis-aligned cuts in feature space | Trees, Random Forest, XGBoost | Tabular data, interactions, non-linear-ish without hand-crafted features. |
| **Probabilistic** | Distributions + Bayes | Naive Bayes, GMM, HMM | Uncertainty, text, sequences. |
| **Metric / geometry** | Distances, neighborhoods | k-NN, K-Means, DBSCAN | Small/medium data, clustering, “who’s near whom?” |

**Mini pop quiz:** *Which family is naturally **scale-sensitive** unless you normalize?* → Distance-heavy methods (k-NN, K-Means, SVM with RBF).

---

## 2. Linear vs. trees (the culture clash)

### Q: How do linear models differ from tree-based ones?

**Direct answer:** Linear models draw **one smooth boundary** (hyperplane in feature space) by minimizing a global loss. Trees **partition** space into boxes with axis-aligned splits — flexible, local, non-parametric in spirit.

**Runway analogy:** A linear model is a **single clean silhouette** — one line through the look. A tree is **layering** — jacket on/off, belt high/low — piecewise rules that can carve weird shapes.

### Linear regression (the baseline everyone should respect)

**Goal:** Predict $\hat{y} = w^T x + b$.

**MSE:** $J(w, b) = \frac{1}{n} \sum (y - \hat{y})^2$

**Assumptions (interviewers love these):**
1. **Linearity** — relationship is roughly linear in inputs (or you engineered that).
2. **Homoscedasticity** — error spread doesn’t fan out wildly with $x$.
3. **No nasty multicollinearity** — features aren’t redundant clones of each other.

---

## 3. Ensembles — why one tree rarely wins the trophy

### Q: Why are ensembles usually stronger than a single deep tree?

**Direct answer:** Deep trees **overfit** (high variance). Ensembles average or correct errors.
- **Bagging (Random Forest):** Parallel trees on row/feature bootstraps → **variance down** (voting / averaging).
- **Boosting (XGBoost, etc.):** Sequential trees fixing **residuals** → **bias down**, but watch overfitting and tuning.

**MI angle:** One bowler on a bad day loses the match; a **balanced attack** (ensemble) survives noise.

---

## 4. Bias–variance (the trade you can’t outsource)

### Q: Decompose expected error.

**Foundation:** $\text{Total Error} \approx \text{Bias}^2 + \text{Variance} + \sigma^2$ (irreducible noise).

- **High bias** — underfitting: model too simple (linear on a curved world).
- **High variance** — overfitting: model chases noise (deep unpruned tree).

**Quick thought experiment:** *Your offline metrics are great but prod is bad.* List **three** non-model causes before you blame the algorithm. → Leakage, train/serve skew, drift, wrong metric, bad monitoring…

---

## 5. Regularization (L1 / L2 / elastic net)

- **L1 (Lasso):** Pushes some weights to **zero** → sparse models, implicit feature selection.
- **L2 (Ridge):** Shrinks weights smoothly → stability, especially with correlated features.
- **Elastic net:** When you want **both** stories in one recipe.

**DevOps parallel:** Regularization is **rate limiting** for parameters — stops any single feature from screaming over the rest of the system.

---

> **Where to sprint tonight:** [AI & ML revision guide](../AI_ML_REVISION_GUIDE.md) for the cheat sheet. [Math derivations hub](../ml-interview-notes/math-derivations.md) when the whiteboard comes out.
