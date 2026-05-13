# Week 2 (Days 8-21): Algorithms

**Goal:** Go whiteboard-ready on the core ML algorithms. Know the math behind each one, not just the API call. This is the longest week because it covers the widest ground — supervised, unsupervised, deep learning, and evaluation.

---

## What This Week Covers

| Days   | Topic                              | Key Concepts                                                    |
|--------|------------------------------------|-----------------------------------------------------------------|
| 8-10   | Supervised Learning                | Linear/logistic regression, SVM hinge loss, decision trees     |
| 11-12  | Unsupervised Learning              | K-means, DBSCAN, PCA, t-SNE, autoencoders                      |
| 13-14  | Neural Networks                    | Backprop chain rule, activation functions, optimizers           |
| 15-16  | Model Evaluation & Metrics         | ROC vs. PRC, F1, calibration curves, cross-validation          |
| 17-18  | Hyperparameter Tuning              | Grid search, random search, Bayesian optimization, early stop   |
| 19-21  | Specialized Techniques (NLP & CV)  | Transformers, BERT vs. GPT, CNNs vs. ViT, attention mechanism  |

---

## Focus Areas

- **SVM:** Understand the geometric intuition and the hinge loss derivation. Know how the kernel trick extends it.
- **Decision Trees:** Gini impurity vs. entropy — when does it matter? How does pruning work?
- **Backpropagation:** Be able to trace gradients through a small network by hand.
- **Adam vs. SGD:** Know why momentum helps and when Adam can fail to generalize.
- **ROC vs. Precision-Recall:** ROC is for balanced classes; PRC is for imbalanced. Know this cold.
- **Self-Attention:** Understand Q, K, V matrices and why the scale factor sqrt(d_k) exists.

---

## Daily Study Pattern

1. Read the linked material for that day's topic.
2. Derive or sketch the core math on paper (loss function, gradient, decision boundary).
3. Explain the algorithm in one paragraph as if to a junior engineer.

---

## Linked Resources

- [Supervised Learning](../02-classical-ml/supervised-learning.md)
- [Unsupervised Learning](../02-classical-ml/unsupervised-learning.md)
- [Deep Learning Overview](../03-deep-learning/README.md)
- [Algorithms & Theory (interview notes)](../07-interview-prep/ml/algorithms.md)
- [Model Evaluation & Metrics](../07-interview-prep/ml/model-evaluation.md)

---

## End-of-Week Check

- Can you derive the SVM objective and explain what the support vectors are?
- Can you explain the attention mechanism in a Transformer from scratch?
- Can you choose the right evaluation metric given a class-imbalanced dataset and justify it?
- Can you describe what happens during backprop when you add a skip connection?
