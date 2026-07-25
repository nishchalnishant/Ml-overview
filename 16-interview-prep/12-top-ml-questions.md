---
module: Interview Prep
topic: Ml
subtopic: Top Ml Interview Questions
status: unread
tags: [interviewprep, ml, ml-top-ml-interview-questions]
---
# Top ML Interview Questions — Reference Answers

Concept + intuition + tradeoff for each question.

Use this as a fast recall drill. If you need the full mechanism, jump to the canonical deep dives in:
- [Classical ML](../03-classical-ml/README.md)
- [Deep Learning Core](../05-deep-learning-core/README.md)
- [LLMs](../10-llms/README.md)
- [Production ML](../13-production-ml/README.md)
- [System Design](../15-system-design/README.md)

Answer pattern to practice:
- **What it is**
- **Why it works**
- **Where it fails**
- **What changes the choice**

---

## Easy

#### Q: What is Machine Learning?
Learning a function $f: X \to Y$ from data $(x_i, y_i)$ rather than hand-coding rules.
**Types:**
- **Supervised:** labeled $(x, y)$ pairs — classification, regression
- **Unsupervised:** unlabeled $x$ — clustering, density estimation, representation learning
- **Reinforcement:** agent learns by interacting with environment, maximizing cumulative reward
**Senior angle:** say what signal is being optimized, what the labels represent in production, and what failure mode appears if the labels are delayed, noisy, or misaligned with the real objective.

#### Q: Explain the Train / Validation / Test Split.
- **Train**: Learn parameters. Touched every training run.
- **Validation**: Tune hyperparameters, select model. Touched during development.
- **Test**: Final unbiased evaluation. Touched once — at the end.
**Common mistakes:** tuning hyperparameters on the test set (optimistic bias), using future data in train for time series (leakage).
For time series: always use temporal splits. Random splits allow future data to leak into training.
**Senior angle:** explain which split mirrors the eventual deployment regime, and say what leakage looks like in the real system, not just in the notebook.

#### Q: What is the difference between a Hyperparameter and a Parameter?
- **Parameters**: Learned from Data (gradient descent). Examples: Weights, biases. Tuning method: Optimization.
- **Hyperparameters**: Set before training. Examples: Learning rate, depth, regularization. Tuning method: Grid search, random search, Bayesian optimization.
**Senior angle:** note which knobs are most expensive to tune in your system and which ones are just search-space metadata versus real model state.

#### Q: What is Feature Scaling, and which algorithms need it?
- **KNN**: Yes (Distance-based — unscaled features dominate)
- **SVM**: Yes (Kernel distances, margin computation)
- **Logistic regression**: Yes (Gradient magnitudes depend on feature scale)
- **Neural nets**: Yes (Exploding/vanishing gradients)
- **Decision trees / XGBoost**: No (Split-based — only feature rank matters)
**Senior angle:** mention scaling as a training stability issue, an optimization issue, and sometimes a serving-contract issue if upstream feature ranges drift.

#### Q: Explain Precision vs Recall and F1 score.
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2PR / (P + R)
- **High precision needed:** spam filter (don't block legitimate email)
- **High recall needed:** cancer screening (don't miss cases)
- **F1:** balanced. PR-AUC better than ROC-AUC for rare positive classes.
**Senior angle:** tie the metric to the cost of false positives and false negatives, then say how that cost changes the threshold.

#### Q: What is the Bias-Variance Tradeoff?
Total expected error = Bias^2 + Variance + Irreducible Noise.
- **High bias (underfitting):** model too simple, misses patterns. Fix: more capacity, better features.
- **High variance (overfitting):** model too complex, memorizes noise. Fix: regularization, more data.
- **Goal:** minimize total error, not just training error.
**Senior angle:** give one example where more data helps variance but not bias, and one where feature work reduces bias faster than changing the model.

---

## Medium

#### Q: How do you diagnose Overfitting vs Underfitting from loss curves?
- **Both high**: Underfitting (model too simple or features too weak).
- **Train low, val high**: Overfitting (memorizing training data).
- **Both low, close**: Generalizing well.
**Senior angle:** say what you would check next if the curves look "good" but offline and online metrics still disagree.

#### Q: Explain Logistic Regression, its loss, and regularization.
Predicts probabilities using a sigmoid function: $P(y=1 \mid x) = \sigma(w^T x + b)$.
**Loss:** binary cross-entropy.
**Decision boundary:** linear in feature space.
**Regularization:** C in sklearn = inverse lambda. Low C = more regularization. L1 → sparse coefficients. L2 → all coefficients non-zero.
**Senior angle:** call out calibration, interpretability, and why logistic regression is still a strong baseline even when the final model is more complex.

#### Q: Compare Bagging vs Boosting.
- **Bagging**: Trains independent parallel models. Reduces variance. Examples: Random Forest. Lower overfitting risk. Fast (parallelizable).
- **Boosting**: Trains sequentially, each fixes previous errors. Reduces bias. Examples: XGBoost, LightGBM, AdaBoost. Higher overfitting risk if too many rounds. Slower (sequential dependency).
**Senior angle:** explain which one you choose when the model is too noisy versus too biased, and mention parallelism as an operational constraint.

#### Q: How does Backpropagation work?
Chain rule applied layer-by-layer from loss to parameters.
Forward pass: compute and cache activations. Backward pass: compute gradients in reverse using cached activations.
Activations must be cached because computing the gradient of a weight requires the activation from the previous layer.
**Senior angle:** mention that the real implementation issue is not the math but memory, numerical stability, and debugging broken gradient flow.

#### Q: Why do we need BatchNorm?
Reduces internal covariate shift (distribution of inputs to each layer stays stable).
Allows higher learning rates because gradients stay well-scaled.
Acts as a regularizer due to noise from batch statistics.
LayerNorm is used instead of BatchNorm in Transformers because LayerNorm normalizes over features per sample, while BatchNorm normalizes over the batch dimension, which is problematic for variable-length sequences.
**Senior angle:** say when BatchNorm becomes a liability, especially at batch size 1 or in serving paths where train/inference stats diverge.

#### Q: Why is ReLU the most common activation function?
- Non-saturating for positive values → no vanishing gradient problem.
- Computationally cheap (threshold operation).
- Sparse activations (half the neurons off) → implicit regularization.
- **Dying ReLU problem:** if a neuron always receives negative input, gradient is always 0. Fix: LeakyReLU, ELU, or careful initialization.
**Senior angle:** mention that the answer is not "ReLU is best everywhere," but "it is a strong default because it is cheap, stable, and easy to optimize."

#### Q: Compare Transformers to LSTMs.
- **LSTM**: Sequential $O(T)$ depth. Long-range memory degrades with distance. Cannot parallelize across time. Complexity: $O(T \cdot d^2)$. Used today in embedded systems, short sequences.
- **Transformer**: Parallel $O(1)$ depth. Full attention to all positions. Fully parallelizable training. Complexity: $O(T^2 \cdot d)$ — quadratic in sequence. Used everywhere at scale.
**Senior angle:** connect the architecture choice to latency, memory, and whether the task needs true long-range dependency modeling or just local context.

#### Q: How do you handle Class Imbalance?
- `class_weight="balanced"`: Always try first — free.
- **SMOTE (synthetic oversampling)**: Tabular data with low minority count.
- **Undersampling**: Large dataset where majority is very abundant.
- **Focal loss**: Deep learning — down-weights easy examples.
- **Threshold tuning**: When you have a probability score — use cost matrix.
**Senior angle:** call out that class imbalance is often a thresholding and metrics problem as much as a modeling problem.

#### Q: What are the different types of Model Drift?
- **Feature drift**: $P(X)$ changes. Detect via KS test, PSI on features.
- **Concept drift**: $P(Y \mid X)$ changes. Detect by monitoring prediction accuracy on labeled slice.
- **Label drift**: $P(Y)$ changes. Detect by monitoring score distribution.
**Senior angle:** say which drift can be detected immediately versus which needs delayed labels, and how that affects alerting.

---

## Hard

#### Q: Explain the Candidate Generation vs Ranking (Two-Stage Pattern) in recommendations.
Running a heavy model on 10M items per query is too expensive.
- **Stage 1**: Fast retrieval (ANN, BM25, collaborative filtering) reduces millions of items to ~1000 candidates. Tools: FAISS, Elasticsearch, two-tower embeddings.
- **Stage 2**: Heavy ranking (neural net with full features) ranks candidates to top 10–20 results. Tools: LightGBM LambdaMART, DCN, DIN.

#### Q: Why use Cross-Validation instead of a single Train/Val split?
A single train/val split can overfit to the particular split. Cross-validation averages over multiple splits to give a robust estimate of out-of-sample performance.
- **Stratified K-Fold**: preserves class ratio in each fold for classification.
- **Time-series CV**: cannot use random folds — use a rolling window forward (e.g. `TimeSeriesSplit`) to prevent data leakage.

#### Q: Compare different variants of Gradient Descent.
- **Batch GD**: Full dataset. Stable, exact gradient, but slow per step and needs full data in memory.
- **SGD**: 1 sample. Fast updates, escapes local minima, but noisy and high variance.
- **Mini-batch SGD**: 32–512 samples. Best of both — GPU-efficient. Requires tuning batch size.
- **Adam**: Adaptive LR, fast convergence, but can generalize worse than SGD+momentum on some tasks.

#### Q: How do different Regularization Techniques work?
- **L1 (Lasso)**: Sparse weights — some exactly 0. Adds absolute value penalty.
- **L2 (Ridge)**: Small weights — none exactly 0. Adds squared penalty.
- **Dropout**: Random deactivation at training, forcing the network to not rely on any single neuron.
- **Early stopping**: Stop training before validation loss increases.
- **Data augmentation**: Increase effective dataset size via domain-specific transforms.

#### Q: What are some key numbers an ML engineer should know?
- Chinchilla optimal tokens/param: ~20
- GPT-3 params: 175B
- LLaMA 3 training tokens: 15T
- Standard attention complexity: $O(n^2 d)$
- LoRA trainable params: ~0.06% of base model
- BF16 memory: 2 bytes/param
- 70B model BF16 VRAM: ~140GB
