# Top ML Interview Questions: 50+ Questions (All Levels)

---

## 🟢 Level 1: Foundations (Entry-Level / Junior)

### Machine Learning Basics

**1. What is Machine Learning?**
> ML is a subset of AI where systems learn patterns from data to make predictions or decisions without being explicitly programmed.

**2. What's the difference between Supervised, Unsupervised, and Reinforcement Learning?**
> **Supervised**: Learns from labeled data (X→y). **Unsupervised**: Finds structure in unlabeled data. **Reinforcement**: Learns from environment feedback (rewards).

**3. What is the Bias-Variance Trade-off?**
> Bias = error from simplifying assumptions (underfitting). Variance = error from sensitivity to noise (overfitting). Goal: minimize total error.

**4. What is Overfitting? How do you prevent it?**
> Model learns noise in training data. Prevent with: regularization, more data, cross-validation, early stopping, dropout.

**5. What is Underfitting?**
> Model is too simple to capture patterns. Fix with: more features, less regularization, more complex model.

**6. Why do we split data into Train/Validation/Test?**
> Train: learn patterns. Validation: tune hyperparameters. Test: final unbiased evaluation.

**7. What is Cross-Validation?**
> Split data into k folds, train on k-1, validate on 1, repeat k times. Gives robust performance estimate.

**8. What is a Hyperparameter vs a Parameter?**
> **Parameter**: learned from data (weights). **Hyperparameter**: set before training (learning rate, k in KNN).

**9. What is Feature Scaling? Why is it important?**
> Normalize features to similar ranges. Important for gradient-based algorithms and distance-based methods.

**10. What is One-Hot Encoding?**
> Convert categorical variables to binary columns. One column per category.

### Regression

**11. What is Linear Regression?**
> Predicts continuous output as a linear combination of features: $y = w^Tx + b$.

**12. What are the assumptions of Linear Regression?**
> Linearity, independence, homoscedasticity (constant variance), normality of errors.

**13. What is R-squared?**
> Proportion of variance explained: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$. Higher is better (max 1).

**14. What is the difference between MAE and MSE?**
> **MAE**: Mean Absolute Error. Robust to outliers. **MSE**: Mean Squared Error. Penalizes large errors more.

**15. What is Polynomial Regression?**
> Fits a polynomial curve by adding polynomial features ($x^2, x^3$, etc.) to linear regression.

### Classification

**16. What is Logistic Regression?**
> Classification algorithm using sigmoid function to output probabilities. Decision boundary is linear.

**17. Why is it called "Regression" if it's for classification?**
> It "regresses" to a probability value, then thresholds for classification.

**18. What is the Sigmoid function?**
> $\sigma(z) = \frac{1}{1 + e^{-z}}$. Maps any real number to (0, 1).

**19. What is Cross-Entropy Loss?**
> $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$. Penalizes confident wrong predictions heavily.

**20. What is a Confusion Matrix?**
> Table showing TP, TN, FP, FN. Used to calculate precision, recall, F1.

**21. What is Precision?**
> $\frac{TP}{TP + FP}$. Of all predicted positive, how many were actually positive?

**22. What is Recall (Sensitivity)?**
> $\frac{TP}{TP + FN}$. Of all actual positives, how many did we catch?

**23. What is F1-Score?**
> Harmonic mean of Precision and Recall: $\frac{2 \cdot P \cdot R}{P + R}$. Balances both.

**24. When do you use ROC-AUC vs PR-AUC?**
> **ROC-AUC**: Balanced datasets. **PR-AUC**: Imbalanced datasets (focuses on positive class).

**25. What is a Decision Boundary?**
> The surface that separates different classes in feature space.

---

## 🟡 Level 2: Intermediate (Mid-Level / L4)

### Model Comparison & Selection

**26. L1 vs L2 Regularization?**
> **L1 (Lasso)**: Adds $\sum|w|$. Creates sparse weights (feature selection). **L2 (Ridge)**: Adds $\sum w^2$. Shrinks weights smoothly.

**27. When to use Random Forest vs XGBoost?**
> **RF**: Robust baseline, less tuning. **XGBoost**: Higher accuracy, needs tuning, prone to overfit.

**28. What is Bagging vs Boosting?**
> **Bagging**: Parallel independent models, reduces variance. **Boosting**: Sequential models correcting errors, reduces bias.

**29. What is the Kernel Trick in SVM?**
> Implicitly maps data to higher dimensions without computing the transformation. Enables non-linear boundaries.

**30. How does Naive Bayes work?**
> Applies Bayes' theorem assuming features are independent given the class. Works well for text.

**31. What are Decision Tree splitting criteria?**
> **Gini**: $1 - \sum p_i^2$. **Entropy**: $-\sum p_i \log p_i$. Lower impurity = better split.

**32. How does Random Forest reduce variance?**
> Bagging + random feature selection at each split decorrelates trees.

**33. What is Gradient Boosting?**
> Sequentially adds trees that predict the residual errors of previous trees.

**34. What is the difference between Hard and Soft Voting?**
> **Hard**: Majority class wins. **Soft**: Average probabilities, pick highest.

**35. What is Stacking?**
> Use predictions of multiple models as features for a meta-model.

### Deep Learning Basics

**36. What is Backpropagation?**
> Algorithm to compute gradients using chain rule, propagating error from output to input.

**37. What is Gradient Descent?**
> Iteratively update weights in direction of negative gradient: $w = w - \alpha \nabla L$.

**38. SGD vs Adam optimizer?**
> **SGD**: Simple, noisy. **Adam**: Adaptive learning rates per parameter, faster convergence.

**39. What is Batch Normalization?**
> Normalizes layer inputs to zero mean, unit variance. Speeds up training, allows higher LR.

**40. What causes Vanishing Gradients?**
> Sigmoid/Tanh derivatives become very small in deep nets. Fix: ReLU, ResNets, BatchNorm.

**41. What is Dropout?**
> Randomly zeroes neurons during training with probability p. Regularization technique.

**42. What are Skip Connections?**
> $y = F(x) + x$. Allows gradients to flow directly, enabling very deep networks.

**43. What is the difference between Epoch, Batch, and Iteration?**
> **Epoch**: One pass through entire dataset. **Batch**: Subset of data. **Iteration**: One weight update.

**44. Why ReLU over Sigmoid?**
> ReLU: No vanishing gradient (for positive), faster to compute. Sigmoid: Vanishing gradient, expensive.

**45. What is Early Stopping?**
> Stop training when validation loss stops improving. Prevents overfitting.

### Evaluation & Metrics

**46. How do you handle class imbalance?**
> Resampling (SMOTE), class weights, different metrics (F1, PR-AUC), anomaly detection.

**47. What is Stratified K-Fold?**
> Ensures each fold has same class distribution as original data.

**48. What is Log-Loss?**
> Cross-entropy loss. Measures how well probabilities match true labels.

**49. What is AUC-ROC?**
> Area under ROC curve. Measures ability to distinguish classes across all thresholds.

**50. What is the Matthews Correlation Coefficient?**
> Balanced metric for binary classification: $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$.

---

## 🔴 Level 3: Advanced (Senior / L5+)

### Architecture & Theory

**51. Explain the Attention mechanism mathematically.**
> $Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V$. Q=Query, K=Key, V=Value. Scaled dot-product.

**52. Why divide by $\sqrt{d_k}$ in attention?**
> Prevents dot products from growing too large, which would saturate softmax and kill gradients.

**53. What is Multi-Head Attention?**
> Run attention h times with different projections, concatenate. Captures different relationships.

**54. Encoder vs Decoder in Transformers?**
> **Encoder**: Bidirectional attention, sees all tokens. **Decoder**: Causal attention, sees only past.

**55. BERT vs GPT?**
> **BERT**: Encoder-only, bidirectional, for understanding. **GPT**: Decoder-only, autoregressive, for generation.

**56. What is the Reparameterization Trick?**
> $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim N(0,1)$. Makes sampling differentiable in VAEs.

**57. How do GANs train?**
> Minimax game: Generator creates fakes, Discriminator distinguishes real/fake. Both improve.

**58. What is Mode Collapse in GANs?**
> Generator produces limited variety of outputs. Fix: Wasserstein GAN, spectral normalization.

**59. What is KL-Divergence?**
> $D_{KL}(P||Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$. Measures how one distribution differs from another.

**60. What is Perplexity?**
> $PPL = e^{H(p,q)}$ where H is cross-entropy. Lower = model is less "confused".

### Production & MLOps

**61. What is Data Drift?**
> Input distribution changes over time. Detect with PSI, K-S test.

**62. What is Concept Drift?**
> Relationship between features and target changes. Harder to detect.

**63. What is Train-Serve Skew?**
> Difference between training and serving data/logic. Causes production failures.

**64. How do you handle A/B testing for ML?**
> Split traffic, monitor business metrics, statistical significance testing, gradual rollout.

**65. What is a Feature Store?**
> Centralized repository for storing, versioning, and serving features consistently.

**66. Batch vs Real-time Inference?**
> **Batch**: Periodic, high throughput. **Real-time**: On-demand, low latency.

**67. How do you monitor model performance in production?**
> Track prediction distributions, latency, error rates, business metrics, feature drift.

**68. What is Model Quantization?**
> Reduce precision (FP32 → INT8/FP16) to speed up inference and reduce memory.

**69. What is Knowledge Distillation?**
> Train a small "student" model to mimic a large "teacher" model's outputs.

**70. What is Canary Deployment?**
> Release new model to small percentage of traffic, monitor, then gradually expand.

### Advanced Scenarios

**71. How do you debug a model that works in training but fails in production?**
> Check data leakage, train-serve skew, feature drift, preprocessing differences.

**72. How do you handle fairness in ML?**
> Audit for bias, use fairness metrics (demographic parity, equalized odds), debias data/model.

**73. What is Positional Bias in recommendations?**
> Users click top items regardless of relevance. Include position as feature in training.

**74. How do you handle cold-start in recommendations?**
> Content-based features, popular items, ask user preferences, multi-armed bandits.

**75. What is the "Exploitation vs Exploration" trade-off?**
> **Exploit**: Use current best. **Explore**: Try new options. Balance with epsilon-greedy, UCB, Thompson Sampling.
