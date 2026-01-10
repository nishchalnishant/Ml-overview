# Machine Learning Model Revision Guide (Interview Cheat Sheet)

A high-density technical summary of core ML algorithms, designed for quick revision before interviews.

---

## 1. Regression Models

| **Algorithm** | **Type** | **Loss Function** | **Complexity** | **Interview Insight** |
|---------------|----------|-------------------|----------------|-----------------------|
| **Linear Reg** | Parametric | MSE (L2) / MAE (L1) | O(n·p²) | Assumes linearity, homoscedasticity, independence. |
| **Polynomial** | Parametric | MSE | O(n·p^d) | High degree = Overfitting. Use regularization. |
| **Ridge (L2)** | Regularized | MSE + λ||w||₂² | O(n·p²) | Shrinks coeffs; good for multicollinearity. |
| **Lasso (L1)** | Regularized | MSE + λ||w||₁ | O(n·p²) | Coeffs can become 0; perform feature selection. |
| **Elastic Net** | Regularized | MSE + α·L1 + β·L2 | O(n·p²) | Balanced approach; good when features are correlated. |

**Key Metric:** $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ (Proportion of variance explained).

---

## 2. Classification Models

| **Algorithm** | **Link Function** | **Loss Function** | **Decision Boundary** | **Interview Insight** |
|---------------|-------------------|-------------------|------------------------|-----------------------|
| **Logistic Reg** | Sigmoid / Softmax | Binary/Cross-Entropy | Linear | Outputs probabilities. No closed-form solution (use GD). |
| **Naive Bayes** | Probabilistic | N/A (MAP) | Non-linear (usually) | Assumes feature independence. Robust to noise. |
| **KNN** | Lazy | N/A | Non-linear/Wiggly | sensitive to outliers and scale. O(n·p) for inference. |
| **SVM** | Hinge | Hinge + λ||w||₂² | Hyperplane | Max margin separator. Uses Kernel Trick for non-linear. |

**Key Concept:** **Naive Bayes Assumptions.** It assumes that given the class, all features are independent. While rarely true (hence "naive"), it works remarkably well for text classification and high-dimensional sparse data.

---

## 3. Tree & Ensemble Methods

### Comparison Table

| **Method** | **Technique** | **Main Goal** | **Base Learners** | **Bias/Var Influence** |
|------------|---------------|---------------|-------------------|------------------------|
| **Decision Tree**| Splitting | Interpretability | Single | High Variance (overfits) |
| **Random Forest**| Bagging | Reduce Variance | Independent Deep | Reduces Variance |
| **XGBoost/GBM** | Boosting | Reduce Bias | Sequential Shallow | Reduces Bias & Variance |
| **Stacking** | Meta-model | Max Accuracy | Diverse Models | Generalization |

**Decision Tree Splitting Criteria:**
- **Gini Impurity:** $G = 1 - \sum p_i^2$ (Default for CART).
- **Entropy (Info Gain):** $H = -\sum p_i \log p_i$.

**Random Forest Magic:**
1. **Bootstrapping:** Random sampling with replacement.
2. **Feature Subsampling:** Only a random subset of features is considered at each split (reduces tree correlation).

---

## 4. Support Vector Machines (SVM)

**The Hard-Margin Objective:**
$$\min \frac{1}{2} ||w||^2 \quad \text{s.t.} \quad y_i(w^Tx_i + b) \ge 1$$

**The Kernel Trick:**
Allows SVM to find a non-linear boundary by mapping data to a higher dimension.
- **RBF (Gaussian) Kernel:** $K(x, y) = \exp(-\gamma ||x-y||^2)$.
- **Interview Question:** *"What happens if $\gamma$ is too high?"* → **Overfitting** (sharp peaks around support vectors).

---

## 5. Unsupervised Learning

### Clustering & Dim Reduction

| **Technique** | **Goal** | **Key Metric** | **When to Use** |
|---------------|----------|----------------|-----------------|
| **K-Means** | Centroid grouping | Inertia (WCSS) | Circular, equal-sized clusters. |
| **DBSCAN** | Density grouping | Silhouette | Noise detection, non-spherical shapes. |
| **PCA** | Variance preservation | Eigenvalues | Compression, visualizing correlations. |
| **t-SNE** | Local structure | KL Divergence | High-quality 2D/3D visualization only. |

---

## 6. Deep Learning Fundamentals

- **Activation Functions:** 
  - **ReLU:** $f(x) = \max(0, x)$ (Avoids vanishing gradient, but has "Dead ReLU" problem).
  - **Leaky ReLU:** $f(x) = \max(0.01x, x)$ (Fixes Dead ReLU).
  - **Sigmoid:** [0, 1] range; suffers from vanishing gradients in deep nets.
- **Backpropagation:** Chains derivatives from the loss back to the weights using the **Chain Rule**.
- **Optimizers:** 
  - **SGD:** Simple, noisy.
  - **Adam:** Adaptive learning rates per parameter (Momentum + RMSProp).

---

## 7. Fast-Fire Interview Q&A

**1. "L1 vs L2 Regularization?"**
> L1 (Lasso) produces sparse weights (feature selection). L2 (Ridge) produces small weights (reduces multicollinearity).

**2. "Generative vs Discriminative Models?"**
> **Generative** (NB, GMM, GANs) learns $P(X, Y)$ and can generate data. **Discriminative** (Logistic, SVM, Boosting) learns $P(Y|X)$ and focuses on the boundary.

**3. "Parametric vs Non-parametric?"**
> **Parametric** (Linear, Logistic) assumes a fixed functional form. **Non-parametric** (KNN, Trees) grows with the complexity of the data.

**4. "The Bias-Variance Trade-off?"**
> Bias is error from simple assumptions (underfitting). Variance is error from sensitivity to noise (overfitting). Goal: Minimize **Total Error = Bias² + Variance + Irreducible Noise**.
