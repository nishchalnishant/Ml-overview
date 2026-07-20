---
module: Interview Prep
topic: Ml
subtopic: Ml Revision
status: unread
tags: [interviewprep, ml, ml-ml-revision]
---
# Quick Revision Cheat Sheet

Last-minute reference. Every entry includes the formula, one-line insight, and the key distinction from its nearest alternative.

---

## 1. Regression Metrics

| Metric | Formula | Insight | When to use |
| :--- | :--- | :--- | :--- |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Linear penalty — outliers don't dominate | When outliers exist and you don't want them overweighted |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Quadratic penalty — outliers dominate | When large errors are disproportionately costly |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target — interpretable | Default reporting metric |
| R² | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Fraction of variance explained | Model comparison on the same dataset |

Normal equation: $\hat{\theta} = (X^TX)^{-1}X^Ty$. Cost: $O(d^3)$ for the matrix inversion. Use gradient descent when $d > 10{,}000$ or the matrix is near-singular.

---

## 2. Classification Metrics

$$\text{Precision} = \frac{TP}{TP+FP} \quad \text{Recall} = \frac{TP}{TP+FN} \quad F1 = \frac{2PR}{P+R}$$

**Precision:** of everything we called positive, what fraction actually was?
**Recall:** of everything that actually was positive, what fraction did we catch?

**When each matters more:**
- Spam filter: FP (blocking legitimate email) is costly → prioritize precision
- Cancer screening: FN (missing a case) is costly → prioritize recall
- Fraud: FN usually costs more (undetected fraud) → recall-oriented

**ROC-AUC vs PR-AUC:**
- ROC-AUC: measures separation across all thresholds. Optimistic on imbalanced data because TN is plentiful and easy — FPR = FP/(FP+TN) looks artificially small.
- PR-AUC: does not involve TN. Better choice when the positive class is rare (fraud, disease). A random classifier gets PR-AUC ≈ class prevalence; ROC-AUC ≈ 0.5 regardless of prevalence.

**Imbalance trap:** 99.5% accuracy when fraud rate = 0.5% means the model predicts all-negative. Always check F1 or PR-AUC.

---

## 3. Trees and Ensembles

**Split criteria:**

Gini impurity: $G = 1 - \sum_k p_k^2$ (cheaper to compute)

Entropy: $H = -\sum_k p_k \log_2 p_k$ (information-theoretic)

Both select the feature and threshold that maximally reduces impurity after the split.

| Method | Mechanism | Reduces | Parallelizable |
| :--- | :--- | :--- | :--- |
| Random Forest | Bootstrap samples + random feature subsets | Variance | Yes |
| Gradient Boosting | Sequential: each tree fits residuals of prior ensemble | Bias | No |
| XGBoost | GBM + L2 tree regularization + second-order Taylor | Bias | Partial |

**Boosting learning rate:** lower `learning_rate` → more trees needed for same fit → better generalization (regularization effect), but slower training. Never increase `learning_rate` without also increasing `n_estimators`.

**Why trees don't need feature scaling:** tree splits are based on rank (which side of a threshold), not magnitude. Multiplying a feature by 1000 changes nothing — the split threshold changes by the same factor. KNN, SVM, and neural nets do need scaling (distance or gradient magnitude depends on scale).

---

## 4. Neural Net Fundamentals

**Activation functions and their roles:**

| Activation | Formula | Gradient | Use case |
| :--- | :--- | :--- | :--- |
| ReLU | $\max(0, x)$ | $\mathbb{1}[x>0]$ | Default hidden layers — no saturation for $x > 0$ |
| Leaky ReLU | $\max(0.01x, x)$ | $\mathbb{1}[x>0] + 0.01 \cdot \mathbb{1}[x\leq 0]$ | Avoids dying ReLU problem |
| GELU | $x \cdot \Phi(x)$ | Smooth approximation | Transformers — smoother than ReLU |
| Sigmoid | $1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ | Binary output only |
| Softmax | $e^{x_i}/\sum_j e^{x_j}$ | Jacobian form | Multiclass output only |

**Gradient problems and fixes:**

| Problem | Mechanism | Fix |
| :--- | :--- | :--- |
| Vanishing gradient | Deep sigmoids: product of $\sigma' < 0.25$ → 0 | ReLU, residual connections, LayerNorm |
| Exploding gradient | Large weight matrices: product grows exponentially | Gradient clipping: `clip_grad_norm_(params, 1.0)` |
| Dying ReLU | Negative pre-activation → zero gradient forever | Leaky ReLU, lower learning rate, He initialization |

**Weight initialization — why it matters:**
At initialization, you want the variance of activations to stay approximately constant through layers. If variance shrinks, gradients vanish; if it grows, gradients explode.

- Xavier / Glorot (sigmoid, tanh): $\sigma = \sqrt{2/(n_{in}+n_{out})}$ — balances forward and backward variance
- He (ReLU): $\sigma = \sqrt{2/n_{in}}$ — accounts for ReLU zeroing ~half of activations

---

## 5. Regularization

| Method | Effect | Formula | Key interaction |
| :--- | :--- | :--- | :--- |
| L1 (Lasso) | Sparsity: weights pushed exactly to 0 | $\lambda\sum|w_i|$ | MAP with Laplace prior |
| L2 (Ridge / weight decay) | All weights small, none exactly 0 | $\lambda\sum w_i^2$ | MAP with Gaussian prior |
| Dropout | Neurons can't co-adapt — each must work alone | Keep probability $p$; scale by $1/p$ at test | Don't combine with BatchNorm without care |
| BatchNorm | Normalizes layer inputs | $\hat{x} = (x-\mu_B)/\sigma_B \cdot \gamma + \beta$ | Acts as regularizer via batch noise |
| Early stopping | Stops before val loss rises | Monitor val loss, patience = N epochs | Free — should always be on |

**Bias-variance decomposition:**
$$\text{MSE} = \underbrace{\text{Bias}^2}_{\text{underfitting}} + \underbrace{\text{Variance}}_{\text{overfitting}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

Regularization trades bias for variance: adding L2 increases bias (shrinks toward zero) but reduces variance (less sensitive to training noise). The right tradeoff depends on whether you have more data or a more complex model than you need.

---

## 6. Optimization

**SGD with momentum:**
$$v_t = \beta v_{t-1} + (1-\beta) g_t, \quad \theta_t = \theta_{t-1} - \alpha v_t$$

**Adam update:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(second moment)}$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t) \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. For LLM pretraining: $\beta_2=0.95$ (faster adaptation to gradient changes).

**AdamW vs Adam:** AdamW applies weight decay directly to parameters ($\theta \mathrel{*}= (1 - \alpha\lambda)$) before the gradient update. Adam applies it as $g_t \mathrel{+}= \lambda\theta_t$, which interacts with the adaptive scaling in an unintended way. AdamW is the correct implementation for neural networks.

**Learning rate schedules:**
- Warmup: linear ramp from 0 for first 1–5% of steps. Reason: at random initialization, gradient directions are chaotic — a high LR causes divergence. Warmup lets the network find a good basin first.
- Cosine decay: $\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi t/T))$. Smooth decay — model fine-tunes as LR decreases.

---

## 7. Normalization Methods

| Method | Normalized over | Why | Best for |
| :--- | :--- | :--- | :--- |
| BatchNorm | Batch dimension (per feature) | Stabilizes layer inputs; enables higher LR | CNNs, vision models |
| LayerNorm | Feature dimension (per sample) | Sequence-length agnostic; works at batch size 1 | Transformers, NLP |
| GroupNorm | Groups of channels (per sample) | Between BN and LN — stable at small batches | Small-batch vision |
| RMSNorm | Feature dimension, no mean subtraction | Simpler, faster than LayerNorm | LLaMA, modern LLMs |

**BatchNorm failure conditions:**
- Small batch ($B < 8$): noisy mean/variance estimates corrupt normalization
- Variable-length sequences: different samples have different valid positions; batch statistics are meaningless
- Inference with batch size 1: uses running statistics from training — fine if distribution hasn't shifted

---

## 8. Unsupervised Learning

**K-Means:** minimizes $\sum_k \sum_{x \in C_k} \|x - \mu_k\|^2$ via alternating E-step (assign) and M-step (update centroids). Convergence is guaranteed but to a local minimum — use K-Means++ initialization and multiple restarts.

**Choosing k:** elbow method (plot inertia vs k, look for the inflection) or silhouette score ($s \in [-1, 1]$, higher is better, measures how well-separated each cluster is).

**PCA:** projects data onto the top eigenvectors of the covariance matrix — the directions of maximum variance:
$$\Sigma = \frac{1}{n} X^T X, \quad \Sigma v_i = \lambda_i v_i$$

Variance explained by $k$ components: $\sum_{i=1}^k \lambda_i / \sum_i \lambda_i$. PCA is linear — it cannot capture non-linear manifolds. For non-linear structure: UMAP (better than t-SNE: faster, more consistent, can be used for downstream tasks).

**t-SNE vs UMAP:**
- t-SNE: preserves local neighborhoods; not reproducible unless seeded; no inverse transform; only for visualization
- UMAP: preserves both local and global structure better; faster; reproducible; can produce embeddings for downstream use

---

## 9. MLOps Fundamentals

| Stage | Core purpose | Primary failure mode |
| :--- | :--- | :--- |
| Feature store | Single source of truth for features; prevents train/serve skew | Features computed differently in training vs serving |
| CI/CD pipeline | Automated testing, training, and deployment | Missing model validation gates — shipping a regressed model |
| Model monitoring | Detect degradation before users notice | Monitoring the wrong metrics (latency but not accuracy) |
| A/B testing | Measure real business impact with statistical rigor | Peeking early (inflates Type I error) |

**Drift types and responses:**
- **Data drift** ($P(X)$ changes): feature distributions shift — model applies to different inputs than it trained on. Detect with KS test or PSI per feature.
- **Concept drift** ($P(Y \mid X)$ changes): the relationship between inputs and labels changes (e.g., fraud patterns evolve). Detect by monitoring accuracy on a recent labeled slice.
- **Label drift** ($P(Y)$ changes): class balance shifts (seasonal effects on click rates). Detect by monitoring score distribution.

PSI thresholds: $< 0.1$ = stable, $0.1$–$0.25$ = investigate, $> 0.25$ = retrain.

---

## 10. LLM Quick Reference

| Concept | Value / Formula | Why it matters |
| :--- | :--- | :--- |
| Attention complexity | $O(n^2 d)$ | Long context is expensive — drives Flash Attention, sparse attention |
| KV cache memory | $2 \times L \times H \times d_h \times T \times B$ bytes | Main memory constraint at serving time |
| Chinchilla optimal | ~20 tokens per parameter | Undertrained models waste compute on parameters |
| LoRA trainable % | ~0.06–0.1% of parameters | Fine-tunes efficiently without touching base weights |
| DPO beta | 0.01–0.5 (KL penalty strength) | Controls how far the policy can deviate from the reference model |
| Perplexity | $\exp\left(-\frac{1}{T}\sum_t \log P(x_t \mid x_{<t})\right)$ | Lower = more confident at each step; GPT-4 ≈ 8 on WikiText |
| BF16 vs FP16 | BF16 has same exponent range as FP32 — no overflow | FP16 overflows on LLM activations; BF16 doesn't |
| 70B BF16 VRAM | ~140GB | Need 2× H100 80GB (tensor parallel) |

---

## 11. Fast Comparison Index

**L1 vs L2:** L1 produces sparse weights (exactly 0) via the non-smooth penalty at zero. L2 shrinks all weights proportionally but never to exactly zero. Probabilistically: L1 = Laplace prior, L2 = Gaussian prior on MAP.

**Generative vs discriminative:** generative models $P(X,Y)$ (Naive Bayes, VAE, GAN, diffusion). Discriminative models $P(Y|X)$ (logistic regression, SVM, fine-tuned BERT). Discriminative models are typically more accurate for classification; generative models can sample new data.

**BERT vs GPT:** BERT bidirectional → cannot generate text (sees future tokens during training, has no causal mask). GPT causal → cannot see the full context at each position. Choose by task: understanding/classification → BERT; generation/completion → GPT.

**Bagging vs boosting:** Bagging = parallel, independent models, each on a bootstrap sample → reduces variance. Boosting = sequential, each model corrects the previous one's residuals → reduces bias. Random Forest = bagging. XGBoost = boosting. Boosting overfits more readily (more hyperparameters to tune).

**BatchNorm vs LayerNorm:** BatchNorm normalizes over the batch dimension (statistics depend on batch size and composition). LayerNorm normalizes over the feature dimension per individual sample (statistics depend only on that sample). LayerNorm is the default for anything sequential or variable-length.
