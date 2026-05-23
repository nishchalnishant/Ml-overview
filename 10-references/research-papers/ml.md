---
module: References
topic: Research Papers
subtopic: Ml
status: unread
tags: [references, ml, research-papers-ml]
---
# Classical ML & Statistics — Key Papers

A curated list of foundational and interview-relevant papers in classical machine learning.

---

## Foundational Theory

| Paper | Year | Why It Matters |
|---|---|---|
| [A Training Algorithm for Optimal Margin Classifiers (Boser, Guyon, Vapnik)](https://dl.acm.org/doi/10.1145/130385.130401) | 1992 | Original SVM paper — kernel trick, max-margin intuition |
| [Random Forests (Breiman)](https://link.springer.com/article/10.1023/A:1010933404324) | 2001 | Ensemble via bagging + feature subsampling |
| [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin)](https://arxiv.org/abs/1603.02754) | 2016 | Regularized boosting, second-order gradients, cache-aware — still wins tabular benchmarks |
| [Greedy Function Approximation: A Gradient Boosting Machine (Friedman)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-a-gradient-boosting-machine/10.1214/aos/1013203451.full) | 2001 | Theoretical foundation for gradient boosting |
| [The Strength of Weak Learnability (Schapire)](https://link.springer.com/article/10.1007/BF00116037) | 1990 | Boosting origins — converting weak learners to strong |

---

## Dimensionality Reduction & Representation

| Paper | Year | Why It Matters |
|---|---|---|
| [Visualizing Data using t-SNE (van der Maaten & Hinton)](https://www.jmlr.org/papers/v9/vandermaaten08a.html) | 2008 | The standard dimensionality reduction for visualization |
| [UMAP: Uniform Manifold Approximation and Projection (McInnes et al.)](https://arxiv.org/abs/1802.03426) | 2018 | Faster, more structure-preserving than t-SNE |
| [Principal Component Analysis (Hotelling)](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) | 1933 | The ur-dimensionality reduction method |

---

## Probabilistic & Bayesian ML

| Paper | Year | Why It Matters |
|---|---|---|
| [Gaussian Processes for Machine Learning (Rasmussen & Williams)](https://gaussianprocess.org/gpml/) | 2006 | Principled uncertainty quantification — interviewed at DeepMind, Jane Street |
| [Variational Inference: A Review for Statisticians (Blei et al.)](https://arxiv.org/abs/1601.00670) | 2017 | Modern variational methods foundation |
| [A Practical Guide to Support Vector Classification (Hsu et al.)](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) | 2003 | Practical SVM tuning — still cited |

---

## Clustering & Unsupervised Learning

| Paper | Year | Why It Matters |
|---|---|---|
| [A Density-Based Algorithm for Discovering Clusters (DBSCAN — Ester et al.)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) | 1996 | Noise-robust clustering without specifying k |
| [Silhouette analysis for cluster validation (Rousseeuw)](https://www.sciencedirect.com/science/article/pii/0377042787901257) | 1987 | Standard cluster quality metric |

---

## Calibration & Reliability

| Paper | Year | Why It Matters |
|---|---|---|
| [Predicting Good Probabilities With Supervised Learning (Niculescu-Mizil & Caruana)](https://dl.acm.org/doi/10.1145/1102351.1102430) | 2005 | When and why to calibrate model outputs |
| [On Calibration of Modern Neural Networks (Guo et al.)](https://arxiv.org/abs/1706.04599) | 2017 | Modern NNs are overconfident — temperature scaling fix |

---

## Feature Selection & Regularization

| Paper | Year | Why It Matters |
|---|---|---|
| [Regression Shrinkage and Selection via the Lasso (Tibshirani)](https://www.jstor.org/stable/2346178) | 1996 | L1 regularization → sparse models |
| [Elastic Net Regularization (Zou & Hastie)](https://www.jstor.org/stable/3647580) | 2005 | L1 + L2 for correlated features |
| [Regularization and Variable Selection via the Elastic Net](https://www.jstor.org/stable/3647580) | 2005 | Combined L1/L2 for grouping effect |

---

## Imbalanced Learning

| Paper | Year | Why It Matters |
|---|---|---|
| [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al.)](https://arxiv.org/abs/1106.1813) | 2002 | Standard oversampling for class imbalance |

---

## Key Interview Takeaways

**"Which paper do you find most influential in classical ML?"**

Strong answers: XGBoost (Chen & Guestrin) — because it explains why regularized boosting beats unregularized, and why second-order gradients matter for speed and accuracy. The algorithmic clarity is exceptional.

**"What's the difference between bagging and boosting?"**

Breiman's Random Forests paper = bagging (parallel, variance reduction). Friedman's gradient boosting = sequential, bias reduction.

**"When would you not use gradient boosting?"**

High-dimensional sparse data (text) — linear models or deep learning win. When you need calibrated probabilities out of the box — GBMs need calibration. When training time matters more than accuracy — simple models first.
