---
module: Classical ML
topic: Classical Machine Learning
subtopic: ""
status: unread
tags: [classicalml, ml, classical-machine-learning]
---
# Classical Machine Learning

Welcome to the Classical Machine Learning library. This track provides a deep-dive, first-principles exploration of traditional ML algorithms, from their mathematical derivation to their practical implementation.

## Routing — which file do I want?

Several files here cover **the same algorithms on purpose**. They are different study modalities, not duplicates, and they are used at different moments. Read this table before assuming something is redundant.

| If you want to… | Go to | Format |
| :--- | :--- | :--- |
| Understand *why* an algorithm works, from the math up | [01-supervised-learning.md](01-supervised-learning.md) | Problem → intuition → mechanics → what breaks |
| Rehearse *answering out loud* under interview pressure | [03-algorithms.md](03-algorithms.md) | What the interviewer is testing → traps → formulas cold |
| Cram the decision tables the night before | [_comparison.md](_comparison.md) | Dense side-by-side comparison |
| Drill active recall | [_flashcards.md](_flashcards.md) | Q → A |
| Work a tiered question ladder end to end | [_interview-questions.md](_interview-questions.md) | Easy → Medium → Hard |

**Do not merge these.** Collapsing the deep-dive into the interview framing loses the derivations; collapsing the other way loses the traps. Both are load-bearing.

## Core Modules

| File | What it covers |
| :--- | :--- |
| [supervised-learning.md](01-supervised-learning.md) | **Deep dive.** Linear/Logistic Regression, SVM, Decision Trees, Ensembles (RF/XGBoost), KNN, Naive Bayes |
| [algorithms.md](03-algorithms.md) | **Interview framing** of the same algorithms + K-Means, PCA, and a selection framework |
| [unsupervised-learning.md](02-unsupervised-learning.md) | K-Means, DBSCAN, GMM, Hierarchical Clustering, PCA/t-SNE/UMAP |
| [data-preprocessing.md](../02-data/03-data-preprocessing.md) | Data Leakage, Missing Data, Scaling, Categorical Encoding, Concept Drift |
| [classical-ml-flashcards.md](_flashcards.md) | Active recall flashcards for all classical ML concepts |

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [when-classical-ml-wins.md](07-when-classical-ml-wins.md) | When to reach for classical ML instead of deep learning |
| [cross-validation.md](../04-evaluation/03-cross-validation.md) | K-fold, stratified, time-series splits, leakage pitfalls |
| [hyperparameter-optimization.md](08-hyperparameter-optimization.md) | Grid/random search, Bayesian optimization, Optuna |
| [ensemble-methods.md](04-ensemble-methods.md) | Bagging, boosting, stacking — Random Forest, XGBoost, LightGBM |
| [feature-selection.md](../02-data/05-feature-selection.md) | Filter/wrapper/embedded methods, mutual information, regularization |
| [dimensionality-reduction.md](05-dimensionality-reduction.md) | PCA, t-SNE, UMAP, autoencoders for compression |
| [imbalanced-data.md](../02-data/06-imbalanced-data.md) | SMOTE, class weighting, resampling, cost-sensitive learning |
| [ml-evaluation-metrics.md](../04-evaluation/01-ml-evaluation-metrics.md) | Precision/recall, ROC-AUC, calibration, ranking metrics |
| [model-interpretation.md](../04-evaluation/05-model-interpretation.md) | SHAP, LIME, feature importance, partial dependence |
| [calibration-and-uncertainty.md](../04-evaluation/04-calibration-and-uncertainty.md) | Platt scaling, isotonic regression, uncertainty quantification |
| [time-series-analysis.md](06-time-series-analysis.md) | ARIMA, exponential smoothing, classical forecasting methods |
| [anomaly-detection.md](10-anomaly-detection.md) | Isolation Forest, LOF, One-Class SVM, autoencoders — when labels are scarce |
| [interviewquestions.md](../16-interview-prep/study-plans/interviewquestions.md) | Interview-style Q&A across classical ML topics |

