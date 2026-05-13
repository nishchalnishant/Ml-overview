# Quick Revision Cheat Sheet

Last-minute reference. Formulas and numbers you need when the interviewer says "let's move quickly."

---

## 1. Regression

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Robust to outliers |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors more |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target |
| R² | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | 1.0 = perfect, 0 = mean baseline |

Linear regression normal equation: $\hat{\theta} = (X^TX)^{-1}X^Ty$. Use gradient descent when $n$ or $d$ is large ($O(d^3)$ for matrix inversion).

---

## 2. Classification

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F1 = \frac{2 \cdot P \cdot R}{P + R}$$

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

**ROC-AUC vs PR-AUC:**
- ROC-AUC: how well model separates classes. Unaffected by class ratio. Optimistic on imbalanced data.
- PR-AUC: precision-recall tradeoff. Better for imbalanced data (rare positive class).

**Imbalanced data:** "99.5% accuracy" when fraud = 0.5% of data means the model predicts all-negative. Check F1/PR-AUC.

**Logistic regression output:**
$$P(y=1 \mid x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx+b)}}$$
$$\mathcal{L} = -\frac{1}{n}\sum [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

---

## 3. Trees and Ensembles

**Decision tree split (Gini impurity):**
$$G = 1 - \sum_{k} p_k^2$$

**Information gain (entropy):**
$$H = -\sum_k p_k \log_2 p_k$$

| Method | Key idea | Reduces | Parallel? |
| :--- | :--- | :--- | :--- |
| Random Forest | Bagging + feature subsampling | Variance | Yes |
| Gradient Boosting | Sequential residual fitting | Bias | No |
| XGBoost | GBM + L2 regularization + approx splits | Bias | Partial |

**Boosting learning rate rule:** lower `learning_rate` → need more trees → better generalization but slower.

---

## 4. Neural Nets

**Activation functions:**

| Activation | Formula | Use case |
| :--- | :--- | :--- |
| ReLU | $\max(0, x)$ | Default hidden layers |
| GELU | $x \cdot \Phi(x)$ | Transformers |
| Sigmoid | $1/(1+e^{-x})$ | Binary output |
| Softmax | $e^{x_i}/\sum_j e^{x_j}$ | Multiclass output |
| Tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | LSTM gates |

**Gradient issues and fixes:**

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| Vanishing gradient | Deep sigmoids/tanh | ReLU, skip connections, better init |
| Exploding gradient | Large weight matrices | Gradient clipping (`max_norm=1.0`) |
| Dead neurons | ReLU never activates | Leaky ReLU, good init, lower LR |

**Weight initialization:**
- Xavier (sigmoid/tanh): $\sigma = \sqrt{2/(n_{in}+n_{out})}$
- He (ReLU): $\sigma = \sqrt{2/n_{in}}$

---

## 5. Regularization

| Method | Effect | Formula |
| :--- | :--- | :--- |
| L1 (Lasso) | Sparsity — zeroes out weights | $\lambda\sum|w_i|$ |
| L2 (Ridge) | Smooth shrinkage | $\lambda\sum w_i^2$ |
| Dropout | Random deactivation | Keep prob $p$, scale by $1/p$ at test |
| BatchNorm | Normalizes activations | $\hat{x} = (x-\mu)/\sigma \cdot \gamma + \beta$ |

**Bias-variance tradeoff:**
$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$$

- High bias (underfitting): model too simple → increase capacity or reduce regularization
- High variance (overfitting): model too complex → more data, regularization, ensemble

---

## 6. Optimization

**SGD with momentum:**
$$v_t = \beta v_{t-1} + (1-\beta) g_t, \quad \theta_t = \theta_{t-1} - \alpha v_t$$

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Typical defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. For LLM pretraining: $\beta_2=0.95$.

**Learning rate schedules:**
- Warmup: linear ramp for first 1-5% of steps (prevents instability at random init)
- Cosine decay: smooth decay to 10% of max LR

---

## 7. Normalization Methods

| Method | Normalized over | Best for |
| :--- | :--- | :--- |
| BatchNorm | Batch dimension | CNNs, vision |
| LayerNorm | Feature dimension | Transformers, NLP |
| GroupNorm | Groups of channels | Small batches |
| RMSNorm | Feature dimension (no mean) | LLaMA — faster than LayerNorm |

---

## 8. Unsupervised Learning

**K-Means:** minimize $\sum_k \sum_{x \in C_k} \|x - \mu_k\|^2$. Initialize with K-Means++ (probability proportional to squared distance). Choose $k$ with elbow method or silhouette score.

**PCA:** eigendecomposition of covariance matrix $\Sigma = \frac{1}{n}X^TX$. Top $k$ eigenvectors = directions of maximum variance. Whitening: normalize by eigenvalue magnitudes.

**t-SNE:** non-linear dimensionality reduction for visualization. Preserves local neighborhood structure. Not for feature engineering (non-deterministic, no inverse transform).

---

## 9. MLOps Quick Recall

| Stage | Purpose | Key concern |
| :--- | :--- | :--- |
| Feature store | Reusable, versioned features | Train-serve skew |
| CI/CD pipeline | Automated testing + deployment | Model validation gates |
| Model monitoring | Post-deployment health | Data drift, concept drift |
| A/B testing | Safe rollout | Statistical significance |

**Data drift:** input distribution $P(X)$ changes. **Concept drift:** relationship $P(Y|X)$ changes. Monitor with KL divergence or PSI on feature distributions.

---

## 10. LLM Quick Reference

| Concept | Key number / formula |
| :--- | :--- |
| Attention complexity | $O(n^2 d)$ in sequence length |
| KV cache memory | $2 \times L \times H \times d_h \times T \times B$ bytes |
| Chinchilla optimal | ~20 tokens per parameter |
| LoRA trainable % | ~0.06–0.1% of parameters |
| DPO beta | 0.01–0.5 (KL penalty strength) |
| RLHF models | 4: policy, reference, reward, value |
| Perplexity | $\exp(-\frac{1}{T}\sum \log P(x_t|x_{<t}))$ |
| BF16 vs FP16 | BF16: same exponent as FP32 — no overflow |

---

## 11. Fast Compare

**L1 vs L2:** L1 → sparsity (some weights exactly 0). L2 → all weights small, none zero.

**Generative vs Discriminative:** generative models $P(X, Y)$ (Naive Bayes, VAE, GAN). Discriminative models $P(Y|X)$ (logistic regression, SVM, BERT fine-tuned).

**BERT vs GPT:** BERT bidirectional → classification/understanding. GPT causal → generation. Can't use BERT for generation (no causal mask).

**BatchNorm vs LayerNorm:** BatchNorm normalizes across batch (unstable at small batch sizes, can't use in Transformers with variable-length sequences). LayerNorm normalizes across features per sample (sequence-length agnostic, Transformer default).

**Bagging vs Boosting:** Bagging = parallel, independent models, reduce variance (Random Forest). Boosting = sequential, each model fixes previous errors, reduce bias (XGBoost).
