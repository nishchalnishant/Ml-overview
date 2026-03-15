# Optimizers

Optimizers update model parameters using gradients from the loss function. Modern deep learning relies on adaptive learning-rate methods and careful scheduling.

---

## SGD (Stochastic Gradient Descent)

**Update rule:**
\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
\]

- **η**: learning rate. **Momentum** adds a velocity term to smooth updates: \(v_{t+1} = \mu v_t + \nabla L\), \(\theta_{t+1} = \theta_t - \eta v_{t+1}\).
- **Advantages:** Simple, well understood, good generalization when tuned.
- **Limitations:** Sensitive to learning rate and scale; no per-parameter adaptation.

---

## Adam (Adaptive Moment Estimation)

Combines momentum with per-parameter adaptive learning rates using first and second moments of gradients:

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]
\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

- **Default:** \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\). Works well across many tasks without heavy tuning.
- **Advantages:** Fast convergence, little tuning.
- **Limitations:** Can underperform SGD with momentum on some vision/NLP benchmarks; weight decay in Adam is not equivalent to L2 (see AdamW).

---

## AdamW

**Decoupled weight decay:** weight decay is applied directly to \(\theta\) instead of through the gradient:

\[
\theta_{t+1} = (1 - \lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

- **λ**: weight decay. This matches L2 regularization behavior and improves generalization in transformers and LLMs.
- **Standard choice** for pretraining and fine-tuning transformer-based models (BERT, GPT, LLaMA, etc.).

---

## Learning rate scheduling

- **Constant:** fixed η.
- **Step decay:** reduce η by a factor every N steps.
- **Cosine annealing:** \(\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1 + \cos(\pi t/T))\); smooth decay to \(\eta_{min}\).
- **Warmup:** linear or linear then decay; avoids instability at the start of training (common in transformers).
- **Warmup + cosine** is typical for large transformer pretraining.

---

## Large-batch training

- Larger batch size → more stable gradients but fewer updates per epoch; can require **higher learning rate** (linear scaling rule: scale η with batch size) and **warmup**.
- **LARS / LAMB:** layer-wise adaptive scaling for very large batches (e.g. LAMB in BERT).

---

## Mixed precision training

- Use **FP16** (or BFloat16) for forward/backward and **FP32** (or master copy) for weight updates to speed up training and reduce memory on GPUs with tensor cores.
- **Loss scaling:** scale loss before backward to avoid underflow in FP16 gradients; unscale before optimizer step.
- **PyTorch:** `torch.cuda.amp` (autocast + GradScaler). Standard in modern LLM and vision training.

---

## Distributed training

- **Data parallel (DP/DDP):** same model on multiple devices; split batch across devices; all-reduce gradients.
- **Model parallel:** split model across devices (pipeline or tensor parallelism) when the model does not fit on one GPU.
- **FSDP (Fully Sharded Data Parallel):** shard parameters, gradients, and optimizer state across ranks; reduces memory per device for large models.

---

## Quick revision

- **SGD:** \(\theta \leftarrow \theta - \eta \nabla L\); add momentum for stability. **Adam:** adaptive η per parameter via \(m_t\), \(v_t\). **AdamW:** Adam + decoupled weight decay; default for transformers.
- **Scheduling:** warmup then cosine (or decay) is common. **Large batch:** scale LR, use warmup. **Mixed precision:** FP16/BF16 + loss scaling. **Distributed:** DDP, FSDP, model parallelism.
