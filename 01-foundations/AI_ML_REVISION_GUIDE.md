---
module: Foundations
topic: Ai Ml Revision Guide
subtopic: ""
status: unread
tags: [foundations, ml, ai-ml-revision-guide]
---
# AI & ML Revision Guide

**For:** Someone who already builds and ships software and does not need the textbook warm-up.
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

## 1. Failure Modes First

When the interviewer says "debug this model," there are exactly five failure modes worth knowing. Each has a distinct signature and a distinct fix.

---

**Overfitting (High Variance)**

*The problem:* The model memorizes the training set — it has too many degrees of freedom and uses them to fit noise. Training error is low; validation error is high. The gap is the overfit.

*Core insight:* The model has learned patterns that exist only in this particular sample of training data, not in the underlying distribution.

*Fixes:* More training data (the most reliable fix). Regularization (L1/L2, dropout, weight decay). Simpler architecture. Early stopping. Data augmentation.

*What breaks these fixes:* More data is usually unavailable. Regularization introduces a new hyperparameter. Early stopping requires a validation set, which itself costs training data.

---

**Underfitting (High Bias)**

*The problem:* The model is too simple to capture the true pattern. Both training error and validation error are high — the model is consistently wrong in the same way across all data.

*Core insight:* The hypothesis class (set of functions the model can represent) does not contain the true function.

*Fixes:* More complex model. More features or better feature engineering. Reduce regularization. Train longer. Boosting (sequentially correct residual errors).

---

**Class Imbalance**

*The problem:* 99% of examples belong to one class. A model that always predicts the majority class achieves 99% accuracy but is useless — it has learned only that the majority class is common.

*Core insight:* The standard loss function treats all examples equally. Misclassifying 100 minority examples costs the same as misclassifying 1 majority example. The loss gradient is dominated by the majority class.

*Fixes:* Use PR-AUC, F1, or MCC instead of accuracy. Class weights in the loss function (mathematically equivalent to oversampling). SMOTE for oversampling the minority class. Threshold adjustment.

*Quick thought experiment:* Fraud detection — 1 in 10,000 transactions is fraud. A model with 99.99% accuracy might catch zero fraudulent transactions. Always check the confusion matrix.

---

**Vanishing Gradients**

*The problem:* Backpropagation multiplies gradients through every layer via the chain rule. If each layer's gradient is < 1, the product over 100 layers approaches zero exponentially. Early layers receive near-zero signal and stop learning.

*Core insight:* Products of numbers less than 1 shrink exponentially. Deep networks using saturating activations (sigmoid, tanh) reliably produce vanishing gradients.

*Fixes:* ReLU activations (gradient is exactly 1 in the positive domain). Residual connections (skip paths bypass layers, providing gradient highways through addition). Batch/Layer Normalization. Careful weight initialization (Xavier, He).

---

**Exploding Gradients**

*The problem:* The reverse problem — products of numbers > 1 grow exponentially. Gradients become huge, parameter updates destroy previously learned representations, loss diverges.

*Core insight:* RNNs processing long sequences reliably produce exploding gradients because the same weight matrix is applied repeatedly.

*Fixes:* Gradient clipping (cap the gradient norm before applying the update). Lower learning rate. Batch normalization. LSTM/GRU gating (designed to control gradient flow).

---

## 2. The Core Concepts

### Bias-Variance Trade-off

**The problem:** You need to diagnose whether a model is failing because of wrong assumptions or because of sensitivity to the training sample. The fixes are opposite — you can't apply both simultaneously without a principled framework.

**The core insight:** Every model's prediction error decomposes into three terms:

```
E[(y - ŷ)²] = Bias² + Variance + σ²
```

- **Bias²:** Systematic error from wrong structural assumptions. The model is consistently wrong in the same direction.
- **Variance:** Sensitivity to training data fluctuations. The model's predictions vary across different training sets.
- **σ²:** Irreducible noise in the data. Cannot be fixed.

**The mechanics:** Increasing model complexity (more features, deeper trees, more parameters) decreases bias but increases variance. Decreasing complexity does the reverse. The minimum total error sits at the sweet spot.

**What breaks:** Ensemble methods (bagging) can reduce variance without increasing bias — they are not strictly on the trade-off curve. This is why Random Forest and XGBoost dominate tabular data competition.

---

### Gradient Descent

**The problem:** You have a loss function L(θ) with millions of parameters. You can't enumerate all values of θ to find the minimum.

**The core insight:** The gradient ∇L(θ) tells you which direction increases L most steeply. Go the opposite direction.

**The mechanics:** `θ ← θ - α·∇L(θ)`. The learning rate α controls step size.

- Too large: overshoot minima, training diverges.
- Too small: impractically slow, gets stuck in flat regions.

**What breaks:** Non-convex landscapes (local minima, saddle points). Vanishing/exploding gradients in deep networks. Extreme sensitivity to learning rate — it's the most important hyperparameter.

---

### Regularization

**The problem:** A model with too many parameters relative to training examples will fit noise. You need to penalize complexity from within the loss function.

**The core insight:** Large weights encode strong, specific patterns. Penalizing large weights biases the model toward simpler explanations less likely to be coincidental noise.

**L1 (Lasso):** `Loss + α·Σ|w|` — The absolute value penalty has a corner at zero; optimization pushes weights exactly to zero. Implicit feature selection. Choose when you suspect many features are irrelevant.

**L2 (Ridge):** `Loss + α·Σw²` — Smooth gradient; weights shrink toward zero but never reach it. Stable with correlated features. Choose when most features likely contribute.

**What breaks:** Regularization introduces α, a new hyperparameter to tune. Too high → underfitting. Assumes all weights should be small — wrong if a few features genuinely have large true effects.

---

### Cross-Validation

**The problem:** A single train/validation split gives a performance estimate that depends heavily on which examples ended up in each set. With limited data, this variance is high enough to mislead model selection.

**The core insight:** Instead of one split, make K non-overlapping splits. Each example acts as validation data exactly once. Average the K scores — a lower-variance estimate of true generalization.

**What breaks:** K times more expensive. For hyperparameter tuning, use nested CV — the outer loop estimates performance, the inner selects hyperparameters. Using the same CV loop for both produces optimistic bias.

**Time series:** Always use walk-forward (TimeSeriesSplit). Never shuffle time series — future data would leak into training.

---

### Ensemble Methods

**The problem:** A single model makes a single type of error. Can combining multiple models cancel out errors?

**The core insight:** Errors cancel only when models are sufficiently independent — trained on different data subsets or using different algorithms.

**Bagging:** Train in parallel on bootstrap samples. Average predictions. Reduces variance. *Example: Random Forest.*

**Boosting:** Train sequentially. Each model corrects the errors of the current ensemble. Reduces bias. *Examples: XGBoost, LightGBM, AdaBoost.*

**What breaks:** Boosting overfits noisy data — it aggressively chases every misclassification, including mislabeled examples. Bagging helps variance, not bias; if the base learner is systematically wrong, averaging many copies is still wrong.

---

## 3. Architectures

### Transformer

**The problem:** RNNs process sequences one step at a time. Step t+1 cannot begin until step t completes. This prevents parallelization and makes training long sequences slow. Information from early positions must pass through many sequential states to reach the output.

**The core insight:** Replace sequential recurrence with parallel self-attention. Let every position attend directly to every other position simultaneously. The entire sequence is processed in one parallel operation.

**The mechanics:**

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Q (Query), K (Key), V (Value) are linear projections of the input. The dot product QKᵀ measures pairwise relevance. Dividing by `√d_k` prevents large dot products in high dimensions from collapsing softmax into near-zero gradients. The output is a weighted sum of Value vectors.

**What breaks:** O(N²) memory and compute in sequence length. Requires positional encodings — attention is permutation-invariant without them. No inductive sequence bias — needs large data to learn sequence structure.

---

### RAG (Retrieval-Augmented Generation)

**The problem:** LLMs are frozen at training time. They can't answer questions about private documents, recent events, or proprietary data. And they hallucinate — generating plausible-sounding but false statements.

**The core insight:** Separate memory from computation. Retrieve relevant documents from an external knowledge base at inference time, inject them into the context. The LLM grounds its answer in the retrieved evidence rather than purely in its parametric memory.

**Components:** Embedding model + Vector Database + LLM.

**RAG vs. Fine-tuning:** RAG for updating knowledge and grounding answers in sources. Fine-tuning for changing behavior, tone, or task format.

**What breaks:** Retrieval quality is the bottleneck — the LLM can't improve on bad context. Long retrieved documents may exceed the context window or cause "lost in the middle" effects (models ignore content in the middle of very long contexts).

---

### LoRA (Low-Rank Adaptation)

**The problem:** Full fine-tuning of a 70B-parameter model requires updating 70B parameters — hundreds of GB of optimizer states. Inaccessible to most practitioners.

**The core insight:** Weight updates during fine-tuning have low rank. Instead of updating W, add `ΔW = AB` where A ∈ R^(d×r) and B ∈ R^(r×k) with r << d, k. Only the small matrices A and B are trained. Reduces trainable parameters by ~10,000×.

*DevOps parallel:* Patching a service without forking the entire monolith.

---

### RLHF / DPO

**The problem:** A language model trained to minimize cross-entropy loss generates statistically likely text, not helpful or aligned text. "High probability under the training distribution" and "helpful, honest, and harmless" are not the same objective.

**RLHF mechanics:** Train a reward model from human preference data. Use PPO to optimize the language model's policy against the reward model.

**What breaks:** Reward hacking — the policy learns to score high on the reward model without actually being helpful. PPO is unstable and sensitive to reward model quality.

**DPO (Direct Preference Optimization):** The optimal RLHF policy can be derived analytically from the preference data, allowing direct optimization on preference pairs without a separate reward model or PPO. Simpler and more stable.

---

### Agents

**The problem:** A single model call can answer a question but cannot execute a multi-step plan, call external APIs, or update a knowledge base. Real-world tasks require sequences of actions with feedback.

**The core insight:** Loop: perceive state → plan action → call a tool → observe result → update state. The LLM provides the planning; tools provide the actions.

*DevOps parallel:* An orchestration pipeline with an LLM as the control plane. Same questions apply: what are the failure modes, what are the guardrails, how do you handle partial failures?

---

## 4. Math You Can Whiteboard

**Backpropagation:**
The gradient of the loss with respect to any weight is computed via the chain rule — multiply partial derivatives backward through the computation graph. Each gradient is computed once. *Intuition: post-incident blame propagation, but useful.*

**Attention scaling:**
`softmax(QKᵀ / √d_k) V`. Why `√d_k`? Dot products grow in variance with dimension (sum of d independent random variables has variance proportional to d). Dividing by `√d_k` normalizes variance to 1, keeping softmax inputs in a range where gradients are non-zero.

*Mini pop quiz:* What happens without the `√d_k` scaling? Dot products become large → softmax saturates → near-one-hot distribution → gradients near zero → training stalls.

**Sigmoid derivative:**
`σ(z)(1-σ(z))` has maximum 0.25. Deep stacks of sigmoids multiply gradients by ≤ 0.25 at each layer — after 10 layers, gradients are ≤ (0.25)^10 ≈ 10^-6. This is the vanishing gradient problem made concrete.

**Adam update:**
`m = β₁m + (1-β₁)g` (first moment, like momentum)
`v = β₂v + (1-β₂)g²` (second moment, like variance)
`θ ← θ - α · m̂ / √v̂` (bias-corrected update)

Not exotic — just robust per-parameter scaling. Common starting LR: `3e-4`.

**Cross-entropy + softmax gradient:**
The gradient of cross-entropy loss with respect to the logits (before softmax) simplifies to `ŷ - y` — the predicted probability minus the true label. This is the "linear error" intuition: the gradient is proportional to how wrong the probability is.

---

## 5. System Design and Infra

### Distributed Training

**The problem:** A model that doesn't fit in one GPU's memory, or a training run that would take months on a single GPU.

**3D Parallelism:**
- **Data Parallelism (DP):** Split the batch across GPUs. Each GPU has a full model copy. Gradients are synchronized after each step. Scales linearly with GPUs; requires the full model to fit on one GPU.
- **Tensor Parallelism (TP):** Split individual weight matrices across GPUs. Requires fast inter-GPU communication (NVLink). Used when a single layer is too large for one GPU.
- **Pipeline Parallelism (PP):** Split model layers across GPUs — layers 1-12 on GPU 1, layers 13-24 on GPU 2, etc. Efficient for very deep models; introduces "pipeline bubbles" (idle time waiting for the previous stage).

Combine all three when one strategy is insufficient.

---

### Inference Optimization

**The problem:** Serving a large model at production latency and throughput requirements.

**Quantization:** Reduce weights from FP32 to FP16, INT8, or INT4. Halves or quarters memory; accelerates compute. Small accuracy cost.

**KV-cache:** During autoregressive generation, past token key and value vectors don't change. Cache them to avoid recomputing. Essential for any reasonable inference throughput.

**Speculative Decoding:** Small draft model generates K tokens in parallel; large model verifies all K in one forward pass. Accepts tokens at the large model's distribution; achieves 2–4× speedup with no quality degradation.

**Flash Attention:** Restructures attention computation to stay within fast SRAM — avoids materializing the O(N²) attention matrix in slow HBM. Enables longer sequences; also faster due to memory hierarchy effects.

**PagedAttention (vLLM):** Manages KV cache as OS-style virtual memory pages. Eliminates fragmentation; enables 24× higher throughput for concurrent requests of variable length.

---

### Evaluation for LLMs

**The problem:** LLM outputs are non-deterministic and open-ended. You can't use a simple loss or accuracy metric.

- **G-Eval / LLM-as-judge:** Use a strong LLM (e.g., GPT-4) to evaluate outputs on dimensions like coherence, relevance, and harmlessness. Scalable but inherits the evaluator's biases.
- **RAGAS:** Framework for evaluating RAG pipelines on faithfulness, answer relevancy, context precision, and context recall.
- **Benchmarks (MMLU, HumanEval, etc.):** Standardized test sets. Interpret carefully — contamination (training data overlap with benchmark) can inflate scores.

---

## 6. Topics That Separate Good from Great

### Reinforcement Learning

**The problem:** Supervised learning requires labeled examples. For game playing, robotics, and dialogue, there are no "correct" outputs — only sparse rewards from the environment after sequences of actions.

**Core framework — MDP:** (S, A, P, R, γ) — States, Actions, Transition probabilities, Reward function, Discount factor. The Markov Property: next state depends only on current state and action, not full history.

**Q-Learning:** Learn Q(s, a) — expected cumulative return from taking action a in state s. Bellman update: `Q(s,a) ← r + γ · max_a' Q(s', a')`. DQN adds a neural network approximator, experience replay (random sampling of past transitions to break temporal correlations), and a target network (slow-moving Q-network copy to stabilize training targets).

**Policy Gradients:** Directly optimize the policy by reinforcing actions that led to high returns. PPO clips the policy update ratio to prevent destructively large updates. The algorithm behind RLHF.

**RLHF pipeline:** Supervised fine-tuning (SFT) → Reward model from human preferences → PPO to maximize reward → Aligned LLM.

---

### Recommender Systems

**The problem:** For each of millions of users, retrieve and rank the best items from a catalog of millions — in milliseconds.

**Two-Tower Model:** Encode user and item independently into the same embedding space. At inference: compute user embedding once; retrieve top-k items by approximate nearest-neighbor search. Separate query and index encoders scale to the retrieval problem.

**Collaborative Filtering:** Recommendations based on user-item interaction patterns — users who liked A also liked B. Matrix factorization finds latent factors explaining the interaction matrix.

**Cold Start:** New users and items have no interaction history. Fallback: content-based features (use the item's attributes), popularity-based (recommend globally popular items), or explicit preference elicitation (ask).

**Evaluation:** NDCG@K, Precision@K, Recall@K. Not accuracy — order matters. Being correct at rank 1 is worth more than being correct at rank 10.

---

### Interpretability

**The problem:** A model says "deny this loan." The regulatory framework requires an explanation. The black-box model provides none.

**SHAP (SHapley values):** Fair attribution of each feature's contribution to a specific prediction, based on cooperative game theory. Satisfies efficiency (attributions sum to the prediction). TreeSHAP is exact and fast for tree-based models.

**LIME:** Fit a local linear model around a single prediction on perturbed samples. Simpler than SHAP; unstable across runs.

**Grad-CAM:** For CNNs, highlight which image regions drove a prediction. Gradient of the class score with respect to the last convolutional layer's feature maps.

**Interview trap:** Attention ≠ explanation. High attention weight on a token does not mean that token caused the output — attention weights are not causal attribution.

---

### Causal Inference

**The problem:** Your model found a strong correlation between feature A and outcome B. Your stakeholder wants to intervene on A to change B. But if A is correlated with B only because both are caused by a confounder C, the intervention fails.

**The core distinction:** Correlation tells you what varies together. Causation tells you what happens when you intervene. Different tools.

**When you can randomize:** A/B test. Randomization breaks confounding — any difference in outcomes must be caused by the treatment assignment.

**When you can't randomize:**
- **DiD (Difference-in-Differences):** Compare treated vs. control groups before and after treatment. Controls for time-invariant confounders.
- **Propensity Score Matching:** Match treated and untreated units on observed covariates.
- **Instrumental Variables:** Use a variable that affects treatment but has no direct effect on the outcome.

**Peeking problem:** Stopping an A/B test early when it looks significant inflates the false positive rate. Pre-register your sample size and significance threshold before launching.

---

### Emerging Trends (2024–2025)

**Test-Time Scaling:** More inference compute → better answers. Chain-of-thought reasoning as a search process over intermediate steps. OpenAI o1, DeepSeek-R1 demonstrate performance scaling with inference compute budget. The implication: the quality ceiling is no longer just a function of training compute.

**Mamba / SSMs:** Linear-time sequence models with input-dependent state selection. Competitive with Transformers on long sequences without O(N²) attention cost. Still maturing; Transformers dominate in practice.

**MoE at scale:** Sparse activation — route each token to a subset of expert sub-networks. Large parameter count at a fraction of active compute. Mixtral 8×7B demonstrated competitive quality at much lower serving cost.

**Synthetic data:** Phi-series models showed quality > quantity. Self-play and rejection sampling can generate training signal better than web-crawled data for specific capabilities. Implications for data-limited domains.

**Long context (1M+ tokens):** Ring Attention distributes the attention computation across devices. YaRN scales positional encodings to support longer sequences. "Lost in the middle" is a real failure mode — models tend to ignore content in the middle of very long contexts even when the context fits.

---

## 7. Production Deployment

### Model Serving

**Batch inference:** Process large datasets offline. No strict latency requirement. Higher throughput, more complex models. *Examples: overnight churn scoring, weekly recommendation refresh.*

**Real-time inference:** Low latency (single-digit to hundreds of milliseconds). Model optimization is not optional. *Examples: search ranking, fraud detection, content moderation.*

**Serving infrastructure options:**
1. REST API (FastAPI) — simple, wide tooling support
2. gRPC — lower latency, better for high-throughput internal services
3. Cloud managed endpoints (AWS SageMaker, GCP Vertex AI, Azure ML)
4. Edge deployment (TensorFlow Lite, ONNX) for on-device inference

---

### Monitoring and Drift

**The problem:** Models degrade silently. No code changes, no alarms — just predictions that slowly stop reflecting reality. Unlike a broken API endpoint, a drifting model returns 200 OK on every request.

**Data drift:** Input feature distributions change over time. Detection: Population Stability Index (PSI), KL divergence between reference and current distributions. Fix: retrain on recent data.

**Concept drift:** The relationship between features and target changes over time. Detection: track model performance metrics directly (requires labeled ground truth, which lags). Fix: retrain, potentially with new features.

**Monitoring checklist:**
- Model accuracy / precision / recall (lagged by label availability)
- Prediction score distribution (can detect drift without labels)
- Input feature distributions
- Inference latency and throughput
- Error rates

---

### A/B Testing

**The problem:** Offline evaluation on a test set does not guarantee that a better test-set model produces better business outcomes in production. You need production evidence.

**The core insight:** Randomize users between old and new models. Any difference in outcomes is caused by the model, not confounders.

**Setup:**
1. Pre-register sample size and significance threshold.
2. Split traffic (90% control / 10% treatment is common to limit exposure).
3. Monitor both model metrics (accuracy, latency) and business metrics (conversion, engagement, revenue).
4. Do not peek early — stopping when the p-value first drops below 0.05 inflates the false positive rate.
5. Gradual rollout if treatment wins.

---

### MLOps: The Full Pipeline

**Trigger** (schedule or data landing event)
→ **Validate** data (schema checks, distribution drift checks, volume checks)
→ **Train** in a containerized job (Azure ML, Vertex AI, SageMaker, or raw Kubernetes)
→ **Evaluate** vs. threshold (don't deploy automatically if metrics regress)
→ **Register** in model registry with metadata (data version, hyperparameters, metrics)
→ **Deploy** to managed endpoint or AKS
→ **Monitor** (latency, accuracy, feature drift, error rates)
→ **Alert** → rollback to previous version or trigger retraining

*Same story as always: gates, artifacts, observability.*

---

## 8. Interview Framework

When an interviewer asks a technical question, use this structure:

> "The direct answer is ___. The intuition is ___. In production, the trade-off is usually ___."

This signals three things: you know the concept, you understand it, and you've thought about it at scale.

---

**Common questions and what they're actually testing:**

**"Explain bias-variance trade-off with an example."**
They want: the decomposition formula, a concrete illustration of both failure modes, and the fixes.
*Example:* "Linear regression on quadratic data has high bias — it's consistently wrong in the same direction. A 20-degree polynomial on 50 training points has high variance — it fits training noise and fails on new data. The trade-off is that reducing one typically increases the other. Ensembles like Random Forest can reduce both simultaneously, which is why they dominate tabular benchmarks."

**"How would you handle imbalanced data?"**
They want: metric selection first (accuracy is misleading), then data techniques, then algorithmic techniques.
*Example:* "First, switch from accuracy to F1 or PR-AUC. Then either class weights in the loss or SMOTE for the minority class. The choice depends on whether false positives or false negatives are more costly — that determines whether you optimize precision, recall, or their harmonic mean."

**"When would you use Random Forest vs. XGBoost?"**
They want: Random Forest as a low-effort robust baseline; XGBoost when you need maximum performance and are willing to tune.
*Example:* "Random Forest is my first pass — minimal tuning, robust to outliers, rarely overfits badly. XGBoost is for when I need every percentage point of performance and have time to tune learning rate, depth, and regularization. LightGBM for very large datasets where XGBoost training time is prohibitive."

**"L1 vs. L2 regularization?"**
They want: L1 produces sparsity (feature selection); L2 shrinks weights smoothly; when to use each.
*Example:* "L1 pushes weights exactly to zero — use it when you suspect many features are irrelevant and want the model to self-select. L2 shrinks weights toward zero smoothly — use it when most features contribute and you want stability with correlated features. Elastic net when both matter."

**"Why ReLU over sigmoid in hidden layers?"**
They want: vanishing gradient explanation; ReLU's gradient = 1 in positive domain; dead ReLU as the failure mode.
*Example:* "Sigmoid's gradient saturates at the extremes and maxes out at 0.25. Stack 50 layers and the gradient is at most (0.25)^50 ≈ 10^-30. ReLU's gradient is exactly 1 in the positive domain — no attenuation. The failure mode is dead neurons: if a neuron always receives negative input, its gradient is always zero and it never recovers. Leaky ReLU and He initialization address this."

**"Your model has 99% accuracy but stakeholders are unhappy. What's wrong?"**
They want: class imbalance framing; confusion matrix inspection; correct metric selection.
*Example:* "Almost certainly class imbalance. If 99% of examples are negative, always predicting negative achieves 99% accuracy with zero predictive value. I'd look at the confusion matrix — specifically the true positive rate for the minority class. Then switch to F1 or PR-AUC and retrain with class weights or resampling."

**"Explain how attention works."**
They want: the QKV formulation; the scaling; why it replaces RNNs.
*Example:* "Each input position is projected into a Query, Key, and Value. The dot product of Query with every Key gives a relevance score for each position. Softmax converts these to weights; the output is a weighted sum of Values. Dividing by √d_k prevents large dot products in high dimensions from collapsing the softmax distribution. Unlike RNNs, the whole sequence is processed in parallel — this is what made training large language models feasible."

---

## 9. Deep-Dive Links

- [LLM Fundamentals](../05-llms/interview-notes/llm-fundamentals.md)
- [AI System Design](../05-llms/interview-notes/ai-system-design.md)
- [Math Derivations](../07-interview-prep/ml/math-derivations.md)
- [MLOps](../06-production-ml/mlops.md)
- [Reinforcement Learning](../04-specialized-domains/reinforcement-learning/README.md)
- [Recommender Systems](../04-specialized-domains/recommender-systems/README.md)
- [Interpretability & XAI](../08-emerging-topics/interpretability-and-xai/README.md)
- [Causal Inference & A/B Testing](../08-emerging-topics/experimentation-and-causal-inference/README.md)
- [Emerging Trends](../08-emerging-topics/emerging-trends/README.md)
- [Graph Neural Networks](../04-specialized-domains/graph-neural-networks/README.md)

## Flashcards

**Bias²?** #flashcard
Systematic error from wrong structural assumptions. The model is consistently wrong in the same direction.

**Variance?** #flashcard
Sensitivity to training data fluctuations. The model's predictions vary across different training sets.

**σ²?** #flashcard
Irreducible noise in the data. Cannot be fixed.

**Too large?** #flashcard
overshoot minima, training diverges.

**Too small?** #flashcard
impractically slow, gets stuck in flat regions.

**Data Parallelism (DP)?** #flashcard
Split the batch across GPUs. Each GPU has a full model copy. Gradients are synchronized after each step. Scales linearly with GPUs; requires the full model to fit on one GPU.

**Tensor Parallelism (TP)?** #flashcard
Split individual weight matrices across GPUs. Requires fast inter-GPU communication (NVLink). Used when a single layer is too large for one GPU.

**Pipeline Parallelism (PP): Split model layers across GPUs?** #flashcard
layers 1-12 on GPU 1, layers 13-24 on GPU 2, etc. Efficient for very deep models; introduces "pipeline bubbles" (idle time waiting for the previous stage).

**G-Eval / LLM-as-judge?** #flashcard
Use a strong LLM (e.g., GPT-4) to evaluate outputs on dimensions like coherence, relevance, and harmlessness. Scalable but inherits the evaluator's biases.

**RAGAS?** #flashcard
Framework for evaluating RAG pipelines on faithfulness, answer relevancy, context precision, and context recall.

**Benchmarks (MMLU, HumanEval, etc.): Standardized test sets. Interpret carefully?** #flashcard
contamination (training data overlap with benchmark) can inflate scores.

**DiD (Difference-in-Differences)?** #flashcard
Compare treated vs. control groups before and after treatment. Controls for time-invariant confounders.

**Propensity Score Matching?** #flashcard
Match treated and untreated units on observed covariates.

**Instrumental Variables?** #flashcard
Use a variable that affects treatment but has no direct effect on the outcome.

**Model accuracy / precision / recall (lagged by label availability)?** #flashcard
Model accuracy / precision / recall (lagged by label availability)

**Prediction score distribution (can detect drift without labels)?** #flashcard
Prediction score distribution (can detect drift without labels)

**Input feature distributions?** #flashcard
Input feature distributions

**Inference latency and throughput?** #flashcard
Inference latency and throughput

**Error rates?** #flashcard
Error rates

**[LLM Fundamentals](../05-llms/interview-notes/llm-fundamentals.md)?** #flashcard
[LLM Fundamentals](../05-llms/interview-notes/llm-fundamentals.md)

**[AI System Design](../05-llms/interview-notes/ai-system-design.md)?** #flashcard
[AI System Design](../05-llms/interview-notes/ai-system-design.md)

**[Math Derivations](../07-interview-prep/ml/math-derivations.md)?** #flashcard
[Math Derivations](../07-interview-prep/ml/math-derivations.md)

**[MLOps](../06-production-ml/mlops.md)?** #flashcard
[MLOps](../06-production-ml/mlops.md)

**[Reinforcement Learning](../04-specialized-domains/reinforcement-learning/README.md)?** #flashcard
[Reinforcement Learning](../04-specialized-domains/reinforcement-learning/README.md)

**[Recommender Systems](../04-specialized-domains/recommender-systems/README.md)?** #flashcard
[Recommender Systems](../04-specialized-domains/recommender-systems/README.md)

**[Interpretability & XAI](../08-emerging-topics/interpretability-and-xai/README.md)?** #flashcard
[Interpretability & XAI](../08-emerging-topics/interpretability-and-xai/README.md)

**[Causal Inference & A/B Testing](../08-emerging-topics/experimentation-and-causal-inference/README.md)?** #flashcard
[Causal Inference & A/B Testing](../08-emerging-topics/experimentation-and-causal-inference/README.md)

**[Emerging Trends](../08-emerging-topics/emerging-trends/README.md)?** #flashcard
[Emerging Trends](../08-emerging-topics/emerging-trends/README.md)

**[Graph Neural Networks](../04-specialized-domains/graph-neural-networks/README.md)?** #flashcard
[Graph Neural Networks](../04-specialized-domains/graph-neural-networks/README.md)
