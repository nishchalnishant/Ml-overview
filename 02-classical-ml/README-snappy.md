---
module: Classical Ml
topic: Readme Snappy
subtopic: ""
status: unread
tags: [classicalml, ml, readme-snappy]
---
# Classical ML — Quick Reference

## Algorithm Cheat Sheet

| Algorithm | When to use | Key hyperparameter | Watch out for |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Continuous target, linear-ish signal, need interpretability | Regularization strength (λ) | Assumes linearity, homoscedasticity, no multicollinearity |
| **Logistic Regression** | Binary/multi-class baseline, calibrated probabilities needed | C (inverse regularization) | Log-loss not MSE; fails on non-linear boundaries |
| **SVM** | High-dim data, clear margin, kernel for non-linear shapes | C (margin), kernel type, γ | Scale-sensitive; slow on large N |
| **Decision Tree** | Interpretable rules, interaction features | max_depth, min_samples_leaf | Overfits deeply; unstable to small data changes |
| **Random Forest** | Tabular data, robust baseline, parallel training | n_estimators, max_features | Slower inference; less interpretable than single tree |
| **Gradient Boosting** | Best tabular accuracy, complex interactions | learning_rate, n_estimators, max_depth | Overfits if unconstrained; needs careful tuning |
| **Naive Bayes** | Text classification, tiny data, fast baseline | var_smoothing (GNB) | Independence assumption often wrong |
| **k-NN** | Small datasets, non-parametric, no training cost | k, distance metric | Scale-sensitive; O(N) predict cost; fails in high-D |
| **K-Means** | Cluster round-ish blobs, customer segmentation | K, n_init | Must choose K; outlier-sensitive; assumes spherical clusters |
| **PCA** | Dimensionality reduction, decorrelate features, visualization | n_components | Linear only; loses interpretability |

---

## Core Concepts (30-second versions)

**Bias–Variance:** Total Error ≈ Bias² + Variance + irreducible noise. High bias = underfitting (too simple). High variance = overfitting (chases noise). Fix bias: add complexity. Fix variance: regularize or ensemble.

**Regularization:**
- L1 (Lasso) → sparsity, implicit feature selection
- L2 (Ridge) → smooth weight shrinkage, handles correlated features
- Elastic net → both

**Ensembles:**
- Bagging (Random Forest): parallel trees → reduces variance
- Boosting (XGBoost/LightGBM): sequential residual correction → reduces bias

**Scale sensitivity:** Distance-based methods (k-NN, K-Means, SVM with RBF) require feature normalization. Tree methods do not.

---

## Quick-access links

- [Supervised learning cheat sheet](supervised-learning-snappy.md)
- [Unsupervised learning cheat sheet](unsupervised-learning-snappy.md)
- [Deep dive: supervised](supervised-learning.md)
- [Deep dive: unsupervised](unsupervised-learning.md)
- [AI & ML revision guide](../01-foundations/AI_ML_REVISION_GUIDE.md)

## Flashcards

**L1 (Lasso) → sparsity, implicit feature selection?** #flashcard
L1 (Lasso) → sparsity, implicit feature selection

**L2 (Ridge) → smooth weight shrinkage, handles correlated features?** #flashcard
L2 (Ridge) → smooth weight shrinkage, handles correlated features

**Elastic net → both?** #flashcard
Elastic net → both

**Bagging (Random Forest)?** #flashcard
parallel trees → reduces variance

**Boosting (XGBoost/LightGBM)?** #flashcard
sequential residual correction → reduces bias

**[Supervised learning cheat sheet](supervised-learning-snappy.md)?** #flashcard
[Supervised learning cheat sheet](supervised-learning-snappy.md)

**[Unsupervised learning cheat sheet](unsupervised-learning-snappy.md)?** #flashcard
[Unsupervised learning cheat sheet](unsupervised-learning-snappy.md)

**[Deep dive?** #flashcard
supervised](supervised-learning.md)

**[Deep dive?** #flashcard
unsupervised](unsupervised-learning.md)

**[AI & ML revision guide](../01-foundations/AI_ML_REVISION_GUIDE.md)?** #flashcard
[AI & ML revision guide](../01-foundations/AI_ML_REVISION_GUIDE.md)
