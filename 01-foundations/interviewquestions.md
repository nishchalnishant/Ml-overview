---
module: Foundations
topic: Interview Questions
subtopic: ""
status: unread
tags: [foundations, interview-prep, ml, math, information-theory, eda, python, glossary, revision]
---
# Foundations — Interview Questions & Answers

**For:** SDE-2 / AI Engineer interviews — calibrated to what's actually asked Round 1 and beyond.
**Difficulty guide:**
- **Easy** → Round 1 basics: definitions, intuition, "explain like you're talking to an engineer." These are the questions you'll get cold from a recruiter screen or first technical interview.
- **Medium** → Round 2 depth: connecting concepts, debugging, applying theory to a scenario, trade-off reasoning.

---

## Easy

> Round 1 fundamentals. If you were asked "basic ML questions" in your first round, this is exactly the bucket. Know all of these cold.

### Q: What is the difference between supervised, unsupervised, and reinforcement learning?
Supervised learning trains on labeled examples (input, correct output) and learns a mapping from one to the other — classification and regression are the canonical tasks. Unsupervised learning works with unlabeled data and looks for structure the data itself contains — clustering (grouping similar points), dimensionality reduction (finding a lower-dimensional representation that preserves structure), and density estimation are typical tasks; there's no "correct answer" to check against. Reinforcement learning trains an agent that takes actions in an environment and learns from a reward signal received over time, rather than from a fixed labeled dataset — it must balance exploring unknown actions against exploiting known-good ones, and the "label" for any given action is often delayed and depends on the agent's own future behavior.

### Q: What is the bias-variance tradeoff?
Bias is systematic error from wrong assumptions — the model is consistently wrong the same way. Variance is instability — the model's predictions swing wildly depending on which training examples it happened to see. Formally: `Expected Error = Bias² + Variance + σ²` (irreducible noise). In practice: both training and validation error high → high bias (underfitting), fix by adding capacity or features. Training error low but validation error high → high variance (overfitting), fix with more data, regularization, or a simpler model.

### Q: What is overfitting and underfitting? How do you detect them?
Overfitting: the model memorizes training data including noise, so it performs well on train but poorly on val/test. Underfitting: the model is too simple to capture the pattern, so it performs poorly on both train and val. Detect by comparing training loss vs. validation loss — if the gap is large, you're overfitting. If both are high and similar, you're underfitting. Learning curves (plotting error vs. training set size) make this visual: a persistent large gap at high N = overfitting; curves that converge but converge high = underfitting.

### Q: What is the difference between a parameter and a hyperparameter?
A **parameter** is a value the model learns from data during training — weights and biases of a neural network, coefficients of a linear regression. A **hyperparameter** is set *before* training and controls the learning process — learning rate, number of trees, regularization strength, number of layers. Hyperparameters are chosen via validation (grid search, random search, Bayesian optimization) because you can't gradient descend on most of them directly.

### Q: What is the purpose of a train/validation/test split?
- **Train**: learn model parameters
- **Validation**: tune hyperparameters, select models — used repeatedly during development
- **Test**: final unbiased evaluation — touched exactly once, at the end

Common mistakes: tuning hyperparameters on the test set (optimistic bias), using future data in train for time series (leakage).

### Q: What is precision and recall? When would you prioritize one over the other?
Precision = TP/(TP+FP) — of everything I flagged positive, how much was actually positive? Recall = TP/(TP+FN) — of everything that was actually positive, how much did I catch? Prioritize **precision** when false positives are costly (spam filter that blocks legitimate email). Prioritize **recall** when false negatives are costly (cancer screening where missing a case is worse than a false alarm). F1 = 2·P·R/(P+R) is the harmonic mean — use when you want a single balanced number.

### Q: Why can't you just use accuracy for a class-imbalanced problem?
If fraud is 1-in-1000, a model that always predicts "not fraud" gets 99.9% accuracy while catching zero fraud. Accuracy is meaningless here. Instead look at the confusion matrix, precision, recall, F1, or PR-AUC. For imbalanced problems PR-AUC is more informative than ROC-AUC because the large true-negative pool inflates the false-positive rate's denominator making ROC look deceptively good.

### Q: Why don't tree-based models need feature scaling, but linear models and neural networks do?
A decision tree splits on thresholds — any monotonic transformation of a feature preserves the ordering, so it can't change which splits are chosen. Trees are scale-invariant by construction. Linear models compute a weighted sum `w₁x₁ + w₂x₂ + ...` — if x₁ ∈ [0, 100000] and x₂ ∈ [0, 1], the optimizer has to find a tiny w₁, making the loss surface badly conditioned and slowing or destabilizing optimization. Neural networks have the same issue at every layer. Rule: distance-based and gradient-based models (KNN, SVM, logistic regression, neural nets, PCA) need scaling; split-based models (Decision Tree, Random Forest, XGBoost, Naive Bayes) don't. `StandardScaler` (mean 0, std 1) is the default choice; `MinMaxScaler` (bounded [0,1]) is preferred when bounds are physically meaningful, e.g. image pixels or Sigmoid/Tanh activations that saturate outside a bounded range.

### Q: What is regularization, and what's the difference between L1 and L2?
Regularization adds a penalty term to the loss that discourages large weights — this reduces overfitting by simplifying the model. **L2 (Ridge)**: penalty = `λΣw²`, produces small but non-zero weights (shrinkage). **L1 (Lasso)**: penalty = `λΣ|w|`, produces sparse weights where many are exactly zero (feature selection). Geometrically: L2 constrains weights to a sphere (no corners → smooth shrinkage), L1 constrains to a diamond (corners on coordinate axes → sparsity). Probabilistically: L2 = Gaussian prior on weights; L1 = Laplace prior.

### Q: What is cross-validation and why is it better than a single train/val split?
A single train/val split can be lucky or unlucky depending on which examples happened to land in which set. K-fold cross-validation splits data into K folds, trains on K-1 and validates on 1, rotates K times, and averages — giving a more reliable estimate of generalization performance. Stratified K-fold preserves class ratios in each fold. For time series, always use `TimeSeriesSplit` (walk-forward) because random folds allow future data to leak into training.

### Q: What's the difference between classification and regression?
**Classification**: predict a discrete label (spam/not-spam, cat/dog/bird). Loss: cross-entropy. Output: class probabilities (softmax). **Regression**: predict a continuous value (house price, temperature). Loss: MSE or MAE. Output: a real number. The core difference is the output type and the appropriate loss function.

### Q: What is gradient descent?
An iterative optimization algorithm that updates model parameters by repeatedly moving in the direction of the negative gradient of the loss — i.e., the direction that most decreases the loss. Update rule: `θ ← θ - α·∇L(θ)`, where α is the learning rate. Variants: **Batch GD** (full dataset per step, stable but slow), **SGD** (one sample per step, noisy but fast), **Mini-batch SGD** (32–512 samples, GPU-efficient sweet spot), **Adam** (adaptive per-parameter learning rates, fast convergence, current default for neural nets).

### Q: What is the difference between precision and recall, and what is F1?
- **Precision** = TP / (TP + FP): out of all positives we predicted, how many were actually positive?
- **Recall** = TP / (TP + FN): out of all actual positives, how many did we catch?
- **F1** = 2 · Precision · Recall / (Precision + Recall): harmonic mean, balances both. Use F1 when neither false positives nor false negatives are dramatically more costly than the other.

### Q: What is a confusion matrix?
A 2×2 (for binary) table showing **True Positives (TP)**, **False Positives (FP)**, **True Negatives (TN)**, **False Negatives (FN)**. It's the starting point for every classification evaluation — accuracy, precision, recall, F1, and specificity are all derived from it. Always look at the confusion matrix first, before quoting a single number.

### Q: What is p-value? What does it NOT tell you?
A p-value is the probability of observing a result at least as extreme as the one observed, *assuming the null hypothesis is true*. A p-value of 0.03 does NOT mean "there's a 3% chance the null hypothesis is true" — that's a common and serious misinterpretation. It also doesn't measure effect size or guarantee practical significance. A huge sample can produce a tiny, meaningless effect with a very small p-value. Correct interpretation: "if there were truly no effect, we'd see data this extreme only 3% of the time by chance."

### Q: What is the difference between bagging and boosting?
**Bagging** (e.g., Random Forest): trains many models *independently in parallel* on bootstrap resamples and averages predictions → reduces variance. Hard to overfit. **Boosting** (e.g., XGBoost, LightGBM): trains models *sequentially*, each one correcting residual errors of the current ensemble → reduces bias, higher peak accuracy but sensitive to hyperparameters and can overfit noisy labels if not regularized. Rule of thumb: Random Forest when you want something that works with default settings; XGBoost when you have time to tune and need extra performance.

---

## Medium

> Applied/connected concepts. Round 2 depth — expect "why", "how would you debug", and "trade-off between X and Y" questions.

### Q: Why must you fit a StandardScaler only on the training set, and what goes wrong if you don't?
`StandardScaler` computes mean and std from the data and uses them to rescale. If you fit it on the *entire* dataset before splitting, the mean and std used to transform training data are influenced by validation/test values — information that shouldn't be available at training time has leaked into preprocessing. Validation scores become overly optimistic because the model was trained on data implicitly informed by the examples it's later evaluated on. Fix: split first, `scaler.fit(X_train)` only, then transform both splits. Wrapping in a scikit-learn `Pipeline` makes this structurally impossible to get wrong since `cross_val_score` refits every step from scratch on each fold's training data.

### Q: You're told a classifier gets 99.9% accuracy on a fraud dataset. Why might that number be meaningless?
If fraud is 1-in-1000, predicting "not fraud" for everything already gives 99.9% accuracy — a completely useless model. First move: look at the confusion matrix. Did we catch any actual fraud (recall)? Of what we flagged, how much was real fraud (precision)? Use PR-AUC or F1 as the primary metric. Consider resampling (SMOTE, undersampling), class weights in the loss, or threshold tuning based on the cost asymmetry between false positives and false negatives.

### Q: Explain vanishing and exploding gradients — why they happen and what the fixes are.
Backpropagation computes gradients via repeated chain-rule multiplication across layers. If each factor is consistently < 1 (e.g., sigmoid max gradient = 0.25), the product shrinks exponentially with depth — early layers stop learning (vanishing). If factors are consistently > 1 (common in deep RNNs), the product explodes — NaN losses or wildly oscillating weights (exploding). **Fixes for vanishing**: ReLU/GELU activations (gradient = 1 in the active region), residual/skip connections, BatchNorm/LayerNorm, He initialization. **Fix for exploding**: gradient clipping (cap the gradient norm), lower learning rate, normalization.

### Q: When would you use Random Forest vs. XGBoost?
Random Forest is bagging — independent parallel trees averaged together, hard to badly overfit, minimal tuning needed. Good robust baseline. XGBoost is boosting — sequential trees each correcting residuals, higher peak accuracy but more tuning required (learning rate, tree depth, rounds, regularization). Reach for Random Forest when you want something that just works; XGBoost when you have time to tune for maximum performance or when you're iterating in a Kaggle-style competition.

### Q: What is LoRA and why does it work for fine-tuning large models?
Fine-tuning a large model updates every weight, requiring gradients and optimizer state for each — often 4-8x the model size in memory. LoRA's insight: the *update* ΔW during fine-tuning tends to have low intrinsic rank. So instead of learning a full-rank ΔW, LoRA freezes the pretrained W and learns two small matrices A (d×r) and B (r×d) such that ΔW ≈ A·B, with r << d. Only A and B are trained — roughly 10,000x fewer trainable parameters in typical configurations, while matching full fine-tuning quality on many tasks. At inference, A·B can be merged back into W with zero added latency.

### Q: A model has been in production 6 months and is getting worse, but no code changed. What do you check?
This is the classic silent-degradation scenario. Distinguish two failure types: **Data drift** (P(X) changed — e.g., new user segment, seasonal shift) detected via PSI or KL divergence on feature distributions. **Concept drift** (P(Y|X) changed — the relationship between features and target shifted, e.g., customer behavior changed after a competitor launched) only detectable via lagged ground-truth accuracy. Practically: pull up dashboards for prediction score distribution over time, feature distributions over time, and rolling accuracy on labeled cohorts. Fix: retrain on recent data; if it's a genuine regime change, revisit feature engineering.

### Q: How would you design an A/B test for a new recommendation model?
1. Randomize by *users* (not sessions) to avoid contamination from one user seeing both models. 2. Run a power analysis upfront to set sample size and duration before starting — this prevents peeking and early stopping that inflates false-positive rates. 3. Monitor both proxy metrics (NDCG, CTR) and business metrics (revenue, retention) — a model can win on NDCG while losing on revenue. 4. Watch guardrail metrics (latency, error rate, unsubscribe rate) for unintended harm. Only decide after the pre-registered sample size is reached.

### Q: What is data leakage? Walk through a leakage debugging checklist.
Leakage is when information about the target "leaks" into training features in a way that doesn't exist at prediction time — causing a model that looks great offline but fails in production. Debugging checklist: 1) Was preprocessing (scaler, encoder) fit on the full dataset including validation? 2) Does any feature contain information downstream of the target (e.g., "number of support calls" for churn prediction — calls happen *because* of impending churn)? 3) Was a chronological process randomly shuffled, letting the model see future data? 4) Does the same entity appear in both train and val? 5) Was target encoding computed using the row's own label? "Suspiciously good" is always the first hypothesis before crediting a smart architecture.

### Q: What is MLE and MAP estimation? How do they connect to common ML losses?
**MLE** (Maximum Likelihood Estimation): find parameters θ that maximize P(data|θ). Under Gaussian noise assumption → minimizing MSE is identical to MLE. Under Bernoulli/Categorical assumption → minimizing cross-entropy is identical to MLE. **MAP** (Maximum A Posteriori): MLE + a prior: θ_MAP = argmax [log P(data|θ) + log P(θ)]. Gaussian prior on θ → the log-prior = -‖θ‖² → L2 regularization. Laplace prior → log-prior = -‖θ‖₁ → L1 regularization. This is the probabilistic reason regularization works: it's encoding a prior belief that weights should be small.

### Q: Explain the difference between normalization and standardization. When would each be the wrong choice?
**Min-max normalization**: (x − min)/(max − min) → scales to [0,1]. Shape-preserving but highly sensitive to outliers (a single extreme value compresses everything else near zero). **Standardization (z-score)**: (x − μ)/σ → mean 0, std 1. More robust to moderate outliers. Wrong choice for normalization: a feature with extreme outliers (e.g., income) — everything gets compressed to a tiny range near zero. Wrong choice for standardization: image pixels with known hard bounds [0, 255] — min-max is natural here since the bounds are physically meaningful.

### Q: What is cross-entropy loss, and why should you use it for classification instead of MSE?
Cross-entropy = -Σ p log q, derived directly from MLE under a Bernoulli/Categorical likelihood assumption. For classification with a softmax output, the gradient w.r.t. logits = (predicted probability − true label) — a clean, well-scaled signal that stays strong even when the model is confidently wrong. MSE with a sigmoid/softmax saturates at the extremes: a confidently-wrong prediction (output ≈ 0.001 for the true class) produces a *small* MSE gradient exactly when you need the strongest correction. Cross-entropy trains classifiers faster and more reliably.

### Q: What is the condition number of a matrix and why should a DL practitioner care?
κ(A) = σ_max / σ_min (ratio of largest to smallest singular value). It measures how sensitive the solution of Ax=b is to perturbations. A large condition number (ill-conditioned) means small floating-point errors get massively amplified. In deep learning, a poorly-conditioned loss landscape (Hessian with huge curvature ratio) causes gradient descent to oscillate in steep directions while crawling in flat ones — this is why adaptive optimizers like Adam help, and why L2 regularization can be described as "improving the condition number" by raising the smallest eigenvalue away from zero.

### Q: What is an outlier — when should you keep it vs. remove it?
Deciding what to do requires domain judgment about *why* the value is extreme. A **data error** (age = 200, negative transaction that should be positive, sensor glitch) should be corrected or dropped — it's corrupted information, not signal. A **genuine extreme value** (a $10M wire transfer in fraud detection, a viral post's engagement count) is often exactly the signal the model needs — automatically removing it silently guts the model's ability to detect the rare, high-value cases the system is built for. Check: does this value represent something that *could* plausibly occur in the real world, or does it violate a hard physical/logical constraint?

### Q: Why is `df[df.revenue > 0]["flag"] = 1` wrong in Pandas?
This is chained indexing: the first `[]` returns either a view or copy depending on internal Pandas heuristics. If it returns a copy, the second `["flag"] = 1` modifies a temporary object that's immediately discarded — `df` is silently left unchanged, no error raised (only a `SettingWithCopyWarning`). Correct form: `df.loc[df.revenue > 0, "flag"] = 1` — a single indexing call that Pandas guarantees operates on the original DataFrame.

---

## Summary Table — Formula Quick Reference

| Concept | Formula |
|---|---|
| Bias-Variance Decomposition | `E[(y-ŷ)²] = Bias² + Variance + σ²` |
| L1 / L2 Regularization | `Loss + α·Σ\|w\|` / `Loss + α·Σw²` |
| Bayes' Theorem | `P(θ\|D) = P(D\|θ)P(θ) / P(D)` |
| MLE | `argmax_θ Σ log P(x_i\|θ)` |
| MAP | `argmax_θ [Σ log P(x_i\|θ) + log P(θ)]` |
| Shannon Entropy | `H(X) = -Σ p(x) log p(x)` |
| Cross-Entropy | `H(P,Q) = -Σ p log q` |
| KL Divergence | `D_KL(P‖Q) = Σ p log(p/q)`, and `H(P,Q) = H(P) + D_KL(P‖Q)` |
| Mutual Information | `I(X;Y) = H(X) - H(X\|Y) = D_KL(p(x,y)‖p(x)p(y))` |
| Scaled Dot-Product Attention | `softmax(QKᵀ/√d_k) V` |
| Condition Number | `κ(A) = σ_max / σ_min` |
| Normal Equation | `θ* = (XᵀX)⁻¹Xᵀy` |
| Precision / Recall | `TP/(TP+FP)` / `TP/(TP+FN)` |
| F1 Score | `2·P·R / (P+R)` |
| Perplexity | `exp(H(P,Q))` |

---

## Where to Next

- [01-ai-ml-systems-and-application.md](01-ai-ml-systems-and-application.md) — full systems guide
- [02-math-and-theory-foundations.md](02-math-and-theory-foundations.md) — derivations and proofs
- [03-python-and-data-tooling.md](03-python-and-data-tooling.md) — library mechanics
- [04-data-processing-and-eda.md](04-data-processing-and-eda.md) — workflow detail
- [05-information-theory.md](05-information-theory.md) — full derivations
- [06-ml-glossary.md](06-ml-glossary.md) — complete term-by-term reference
- [flashcards.md](flashcards.md) — active recall companion
