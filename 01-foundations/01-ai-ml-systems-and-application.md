---
module: Foundations
topic: Applied AI and ML Systems
status: unread
tags: [foundations, ml, ai-systems, revision]
---
# Applied AI & ML Systems Guide

**For:** Software engineers and practitioners who already build and ship software and need a practical, systems-oriented guide to Machine Learning. 
**Use:** Skim in 20 minutes, deep-read in an hour. Every section starts with the problem — if you understand the problem, the concept follows.

---

## 0. One Mental Model: ML Is Just a Different Kind of Pipeline

**The problem:** Most software developers already understand build pipelines, artifact versioning, staged rollouts, and production monitoring. ML looks foreign because the vocabulary is different — but the structure is almost identical.

| You already do | In ML land |
|---|---|
| CI pipeline | Training + eval + tests on data and code |
| Artifact / versioned package | Model weights + config + preprocessor |
| Staging → prod rollout | Registry stage → canary / A-B → full traffic |
| Health probes | Latency, errors, and accuracy / drift |
| Config + secrets | Hyperparameters, feature flags, LLM API keys |

**The rhythm:** *Train (build) → evaluate (sign-off against metrics) → deploy (serve) → monitor (latency + accuracy + drift) → rollback or retrain.*

The only genuinely new part is that the "artifact" has behavior that degrades over time even without code changes — because the world changes and the model's training data does not.

---

## 1. What Is AI and the Core Paradigms

**The problem:** Computers execute exact instructions. But most useful tasks — recognizing a face, translating a sentence — cannot be fully specified as rules.
**The core insight:** Instead of writing rules, write a *learning procedure* that infers rules from examples. 

### Core Paradigms
- **Statistical Machine Learning:** Learn explicit decision boundaries (linear models, decision trees, SVMs) from labeled data. Expressive enough to fit patterns, constrained enough to avoid memorizing noise.
- **Deep Learning:** Stack parameterized transformations (layers). Instead of hand-crafting features, the hierarchy of features emerges from the data via backpropagation. Requires enormous data and compute.
- **Generative AI:** Model the data distribution $P(x)$ or $P(x|c)$. Can sample new data (images, text, audio). E.g., GANs, VAEs, Diffusion, Autoregressive models (GPT).
- **Reinforcement Learning:** Learn to choose actions that maximize cumulative future reward through environment interaction. Supervised against sparse reward signals rather than exact labels.

---

## 2. Failure Modes First

When debugging a model, there are exactly five failure modes worth knowing.

**1. Overfitting (High Variance)**
*The problem:* Model memorizes the training set. Training error is low; validation error is high.
*Fixes:* More training data (best fix), Regularization (L1/L2, dropout), simpler architecture, early stopping, data augmentation.

**2. Underfitting (High Bias)**
*The problem:* Model is too simple to capture the true pattern. Both training and validation errors are high.
*Fixes:* More complex model, better feature engineering, reduce regularization, boosting.

**3. Class Imbalance**
*The problem:* 99% of examples belong to one class. A model that always predicts the majority class gets 99% accuracy but is useless.
*Fixes:*
- **Metrics:** Stop using accuracy. Use PR-AUC, F1, or MCC. Check the confusion matrix.
- **Resampling:** SMOTE (oversample minority), undersample majority.
- **Algorithmic:** Class weights in the loss function, focal loss, threshold adjustment.

**4. Vanishing Gradients**
*The problem:* Backprop multiplies gradients via the chain rule. If each layer's gradient is < 1, the product over deep layers shrinks exponentially to zero.
*Fixes:* ReLU activations, residual connections (skip paths), Batch/Layer Normalization, careful initialization (He).

**5. Exploding Gradients**
*The problem:* Gradients > 1 grow exponentially. Parameter updates destroy representations, loss diverges (common in RNNs).
*Fixes:* Gradient clipping, lower learning rate, batch normalization, LSTM gating.

---

## 3. Machine Learning Fundamentals

### Bias-Variance Trade-off
`Expected Error = Bias² + Variance + σ² (irreducible noise)`
- **Bias²:** Systematic error from wrong structural assumptions.
- **Variance:** Sensitivity to training data fluctuations.
Increasing complexity decreases bias but increases variance. Ensemble methods like Random Forests can reduce variance without increasing bias.

### Regularization
Penalize complexity from within the loss function to prevent overfitting to noise.
- **L1 (Lasso):** `Loss + α·Σ|w|`. Pushes weights exactly to zero (implicit feature selection).
- **L2 (Ridge):** `Loss + α·Σw²`. Shrinks weights smoothly toward zero. Stable with correlated features.
- **Elastic Net:** Combines both.

### Train/Validation/Test Split
- **Training (60-80%):** Model learns from this.
- **Validation (10-20%):** Used to tune hyperparameters. Looking at it consumes its independence.
- **Test (10-20%):** Touched exactly once. Provides an honest generalization estimate.
*Always stratify for imbalanced classes. Always split chronologically for time series.*

### Cross-Validation (K-Fold)
Instead of one split, make K non-overlapping splits. Average the scores. Provides a lower-variance estimate of generalization on small datasets.

### Ensemble Methods
Combine models to cancel out errors (only works if models are independent).
- **Bagging (Bootstrap Aggregating):** Train parallel models on bootstrap samples. Averages predictions. Reduces variance. *Ex: Random Forest.*
- **Boosting:** Train sequentially. Each model corrects the current ensemble's errors. Reduces bias. *Ex: XGBoost, LightGBM.*

---

## 4. Algorithms and Metrics

### Algorithm Selection
- **Regression:** Linear Regression → Random Forest → XGBoost.
- **Classification:** Logistic Regression → Random Forest → XGBoost.
- **Need Interpretability?** Linear/Logistic Regression, Decision Trees.
- **High-Dimensional?** Lasso, Ridge, Random Forest.
- **Fastest large tabular?** LightGBM.

### Unsupervised
- **Clustering:** K-Means (spherical, needs K), DBSCAN (arbitrary shapes, outlier detection), GMM (probabilistic).
- **Dimensionality Reduction:** PCA (global linear variance), t-SNE (local non-linear visualization), UMAP (faster, preserves global better).

### Classification Metrics
- **Precision:** `TP / (TP + FP)`. Of predicted positives, what fraction are real? Maximize when false alarms are costly.
- **Recall:** `TP / (TP + FN)`. Of all real positives, what fraction did we catch? Maximize when misses are costly.
- **F1 Score:** Harmonic mean of Precision and Recall. Punishes extreme imbalances.
- **ROC-AUC:** Threshold-invariant summary for balanced classes.
- **PR-AUC:** Threshold-invariant summary for imbalanced classes.

---

## 5. Deep Learning Basics

### Why Layers?
A single layer of linear transformations is just a linear transformation. Inserting non-linear **Activation Functions** (ReLU, GELU) makes the composition genuinely non-linear. 

### Optimizers
- **SGD + Momentum:** Accumulates gradient history (velocity) to escape local minima and smooth noise.
- **Adam:** Adaptive per-parameter rates. Divides update by RMS of recent gradients. Common default LR: `3e-4`.
- **AdamW:** Decouples weight decay from adaptive scaling. Preferred for Transformers.

### Normalization
- **Batch Normalization:** Normalizes over the batch dimension. Best for CNNs.
- **Layer Normalization:** Normalizes over the feature dimension. Best for Transformers and small batches.

---

## 6. Architectures

### The Transformer
**The problem:** RNNs process sequentially ($O(N)$ steps), preventing parallelization. 
**The insight:** Replace recurrence with parallel self-attention.
- **Attention:** `softmax(QKᵀ / √d_k) V`. Q (Query), K (Key), V (Value). 
- **Scale factor (`√d_k`):** Prevents large dot products in high dimensions from collapsing softmax gradients to zero.
- **Bottleneck:** $O(N^2)$ memory/compute in sequence length. Solved via Flash Attention.

### RAG (Retrieval-Augmented Generation)
**The problem:** LLMs hallucinate and lack private/recent data.
**The insight:** Retrieve relevant documents from an external DB at inference and inject them into the context. Grounds answers in evidence rather than parametric memory.

### LoRA (Low-Rank Adaptation)
**The problem:** Fine-tuning a 70B model requires hundreds of GBs of optimizer states.
**The insight:** Weight updates $\Delta W$ have low rank. Instead of updating $W$, add $\Delta W = A \cdot B$ where $A$ and $B$ are tiny matrices. Reduces trainable parameters by ~10,000×.

### RLHF & DPO
**The problem:** Cross-entropy loss trains for "statistically likely text", not "helpful/aligned text".
- **RLHF:** Train a reward model from human preferences. Use PPO to optimize the LLM policy against it.
- **DPO:** Derives the optimal policy analytically from preference data, bypassing the separate reward model and PPO instability.

### Agents
**The insight:** Loop: perceive state → plan action → call tool → observe result → update state. An orchestration pipeline with an LLM as the control plane.

---

## 7. System Design & Production

### Distributed Training
- **Data Parallelism (DP):** Split batch across GPUs. Each GPU has full model.
- **Tensor Parallelism (TP):** Split weight matrices across GPUs.
- **Pipeline Parallelism (PP):** Split model layers across GPUs (layers 1-12 on GPU1, etc.).

### Inference Optimization
- **Quantization:** Reduce weights to FP16, INT8, or INT4.
- **KV-Cache:** Cache past token keys/values during autoregressive generation.
- **Speculative Decoding:** Small draft model generates tokens; large model verifies. 2-4x speedup.
- **PagedAttention (vLLM):** Manages KV cache like OS virtual memory, eliminating fragmentation.

### Model Monitoring and Drift
Models degrade silently. They return 200 OK, but predictions no longer reflect reality.
- **Data Drift:** Input feature distributions change. Detect via PSI or KL divergence.
- **Concept Drift:** Relationship between features and target changes. Detect via lagged accuracy/precision. 
- **Checklist:** Monitor prediction score distributions, input feature distributions, latency, error rates.

### A/B Testing
Offline evaluation does not guarantee online business outcomes. 
1. Randomize users between control (old) and treatment (new).
2. Monitor model metrics AND business metrics. 
3. Pre-register sample size. **Do not peek early.**

---

## 8. Interview Framework

When an interviewer asks a technical question, use this structure:
> "The direct answer is ___. The intuition is ___. In production, the trade-off is usually ___."

Worked examples of this pattern (bias-variance, imbalanced data, model selection, attention) are in [interviewquestions.md](interviewquestions.md).

---

## 9. Deep-Dive Links

- [LLM Fundamentals](../05-llms/interview-notes/01-llm-fundamentals.md)
- [AI System Design](../05-llms/interview-notes/07-ai-system-design.md)
- [Math Derivations](../07-interview-prep/ml/18-math-derivations.md)
- [MLOps](../06-production-ml/01-mlops.md)
- [Causal Inference & A/B Testing](../08-emerging-topics/experimentation-and-causal-inference/README.md)
