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
4. Run the code snippet for that day and vary one hyperparameter to observe the effect.

---

## Linked Resources

- [Supervised Learning](../02-classical-ml/supervised-learning.md)
- [Unsupervised Learning](../02-classical-ml/unsupervised-learning.md)
- [Deep Learning Overview](../03-deep-learning/README.md)
- [Algorithms & Theory (interview notes)](../07-interview-prep/ml/algorithms.md)
- [Model Evaluation & Metrics](../07-interview-prep/ml/model-evaluation.md)
- Day files in this folder: day-8-9, day-10-11, day-12-14, day-15-16, day-17-18, day-19-21

---

## Projects for This Week

**Days 8-11 Project: Algorithm Comparison Notebook**
- Dataset: sklearn's `make_classification` with n_samples=5000, n_informative=10, class_weight imbalanced 10:1
- Task: Train k-NN, SVM, Random Forest, and XGBoost; compare ROC-AUC and PR-AUC
- Document: For each model, which hyperparameter had the biggest effect? Why does ROC-AUC look good while PR-AUC is low?
- Stretch: Add K-Means clustering as a preprocessing step; does it help any model?

**Days 12-14 Project: Neural Network from Scratch**
- Implement a 2-layer MLP using only NumPy (no PyTorch/TensorFlow) on the MNIST dataset
- Verify your gradient computation is correct by running a gradient check (finite differences)
- Then replicate using PyTorch; compare the training curves
- Deliverable: A plot showing training loss and validation accuracy over 20 epochs for both implementations

**Days 15-18 Project: Evaluation & Tuning Pipeline**
- Take any model from your Day 8-11 project
- Implement nested cross-validation: inner loop for hyperparameter tuning (Optuna), outer loop for unbiased error estimation
- Report: the difference between the inner-loop best CV score and the outer-loop estimate — this gap is your optimism bias
- Experiment: Does early stopping on the neural network from Day 12 result in a lower outer-loop error?

**Days 19-21 Project: Text + Image Classifier**
- NLP: Fine-tune a DistilBERT model on any short text classification dataset (e.g., SST-2 sentiment)
- CV: Fine-tune a ResNet-18 on CIFAR-10 with and without augmentation; compare validation accuracy
- Analysis: Plot the training curves; where does the model start overfitting? Does augmentation change when overfitting begins?

---

## Milestone Checkpoints

**After Days 8-11:** Can you explain why k-NN fails in high dimensions without using the phrase "curse of dimensionality"? (Force yourself to describe the geometric mechanism.) Can you explain why XGBoost is prone to overfitting on noisy labels?

**After Days 12-14:** Can you trace the gradient of a cross-entropy loss backward through one attention layer by hand? Can you explain why dropout should be off during inference and what happens if it is left on?

**After Days 15-18:** Given a model with 5-fold CV scores of [0.95, 0.60, 0.91, 0.89, 0.92], what is your first action before averaging? Can you explain what nested CV prevents that standard CV does not?

**After Days 19-21:** Can you explain the $\sqrt{d_k}$ scaling in attention without looking it up? Can you name one failure mode of data augmentation that would hurt performance on a specific task?

---

## End-of-Week Check

- Can you derive the SVM objective and explain what the support vectors are geometrically?
- Can you explain the attention mechanism in a Transformer from scratch, including why $\sqrt{d_k}$ scaling exists?
- Can you choose the right evaluation metric given a class-imbalanced dataset and justify it with the ROC vs. PR curve distinction?
- Can you describe what happens during backprop when you add a skip connection (ResNet-style)?
- Can you explain why Bayesian optimization outperforms random search only in the small-budget regime?
