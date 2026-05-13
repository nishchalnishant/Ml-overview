# Optimization for ML Interviews

---

# 1. Gradient Descent

Core update rule — move weights in the direction of steepest descent:

$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

where $\eta$ is the learning rate and $\nabla_\theta L$ is the gradient of the loss.

**Variants by batch size:**

| Variant | Batch size | Gradient quality | Typical use |
| :--- | :--- | :--- | :--- |
| Batch GD | Full dataset | Low variance, exact | Rarely practical |
| SGD | 1 sample | Very noisy | Online learning |
| Mini-batch SGD | 32–512 | Balanced | Deep learning default |

**Convergence for convex $L$:** with step size $\eta \leq 1/L$ (where $L$ is the Lipschitz constant of the gradient), GD converges at rate $O(1/t)$.

---

# 2. Learning Rate

Controls the step size per gradient update. One of the most critical hyperparameters.

**Too high:** $\|\theta_{t+1} - \theta^*\| > \|\theta_t - \theta^*\|$ — training diverges or oscillates.

**Too low:** updates are negligible; convergence takes exponentially more steps.

**Learning rate range test:** sweep LR from small to large over a few hundred steps; choose just before the loss starts rising.

**Typical starting values:**
- Adam/AdamW: $3 \times 10^{-4}$ (Karpathy constant)
- SGD with momentum: $0.01$–$0.1$

---

# 3. Momentum

Accumulates a velocity vector to dampen oscillations and accelerate in consistent directions:

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta L$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

Typical $\beta = 0.9$. Effectively computes an exponentially weighted moving average of past gradients.

**Nesterov momentum** (look-ahead):

$$v_t = \beta v_{t-1} + \eta \nabla_\theta L(\theta_{t-1} - \beta v_{t-1})$$

Nesterov evaluates the gradient at the "predicted" next position, giving slightly faster convergence and better theoretical guarantees.

```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

# 4. Adagrad

Adapts learning rate per parameter based on historical gradient magnitudes:

$$G_t = G_{t-1} + g_t^2 \quad \text{(accumulated squared gradients)}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

- Parameters with large past gradients get smaller effective LR
- Parameters with small past gradients get larger effective LR

**Problem:** $G_t$ grows monotonically — learning rate decays to near-zero and training stalls. Good for sparse features (NLP with bag-of-words), impractical for deep networks.

---

# 5. RMSprop

Fixes Adagrad's decaying LR by using an exponential moving average of squared gradients:

$$v_t = \rho v_{t-1} + (1 - \rho) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$

Typical $\rho = 0.99$, $\epsilon = 10^{-8}$. The moving average forgets old gradient magnitudes, preventing LR collapse.

Still used in RL (A3C, DQN) and RNNs. For most other tasks, Adam is preferred.

---

# 6. Adam

Combines momentum (first moment) with adaptive per-parameter learning rates (second moment):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(uncentered variance)}$$

**Bias correction** (critical at initialization when $m_0 = v_0 = 0$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3\times 10^{-4}$.

Without bias correction, early updates are severely underscaled — $m_1 = (1-\beta_1)g_1$ is $10\times$ smaller than it should be when $\beta_1 = 0.9$.

---

# 7. AdamW

Adam's weight decay is applied through the gradient, which corrupts the adaptive scaling:

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t + \lambda \theta_{t-1}}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{(Adam — wrong)}$$

AdamW decouples weight decay from the adaptive update:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right) \quad \text{(AdamW — correct)}$$

This makes L2 regularization scale-invariant per parameter. Always use AdamW over Adam for modern deep learning.

```python
optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
```

---

# 8. Vanishing and Exploding Gradients

**Vanishing:** gradients shrink exponentially as they propagate backward through layers.

For a network with $L$ layers, the gradient of the loss w.r.t. layer $l$ includes:

$$\frac{\partial L}{\partial \theta^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \prod_{k=l}^{L-1} \frac{\partial a^{(k+1)}}{\partial a^{(k)}}$$

If $\|\partial a^{(k+1)} / \partial a^{(k)}\| < 1$ at each step, the product decays to zero exponentially. Sigmoid saturation causes this: $\sigma'(z) \leq 0.25$.

**Exploding:** product of Jacobians $> 1$ — gradients grow exponentially; weights diverge.

**Fixes:**
- Vanishing: ReLU activations, residual connections, LayerNorm, better initialization
- Exploding: gradient clipping $\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, \tau / \|\mathbf{g}\|_2)$

---

# 9. Learning Rate Scheduling

## Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

Smoothly decays from $\eta_{\max}$ to $\eta_{\min}$ over $T$ steps.

## Warmup + Cosine Decay (LLM standard)

$$\eta_t = \begin{cases} \eta_{\max} \cdot t / T_w & t < T_w \text{ (linear warmup)} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos\frac{\pi(t - T_w)}{T - T_w}) & t \geq T_w \end{cases}$$

Warmup prevents instability when $m_t$ and $v_t$ are unreliable (near zero initialization). Typical $T_w = 1$–$5\%$ of total steps.

## Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

Reduce LR by factor $\gamma$ every $s$ epochs. Simple but abrupt transitions.

## ReduceLROnPlateau

Reduce LR when validation metric stops improving. Patience-based: wait $p$ epochs before decaying.

---

# 10. Hyperparameter Search

## Grid Search

Exhaustively evaluate all combinations in a Cartesian product of hyperparameter values. Cost: $\prod_i |H_i|$ evaluations. Wasteful when most dimensions have low impact.

## Random Search

Sample hyperparameter combinations uniformly from predefined ranges. For $n$ evaluations, covers more of the space in high-impact dimensions than grid search. **Empirically outperforms grid search for the same budget** when most hyperparameters matter less than a few.

## Bayesian Optimization

Models the objective $f(\mathbf{h})$ as a Gaussian Process (GP) or tree-based surrogate. Uses an **acquisition function** to select the next candidate:

- **EI (Expected Improvement):** $\alpha(\mathbf{h}) = \mathbb{E}[\max(f(\mathbf{h}) - f^*, 0)]$
- **UCB:** $\alpha(\mathbf{h}) = \mu(\mathbf{h}) + \kappa \sigma(\mathbf{h})$

Best when each evaluation is expensive (hours of training). Library: `optuna`, `hyperopt`.

## TPE (Tree-structured Parzen Estimator)

Models $p(\mathbf{h} | y < y^*)$ (good region) and $p(\mathbf{h} | y \geq y^*)$ (bad region) separately using kernel density estimates:

$$\alpha(\mathbf{h}) \propto \frac{p(\mathbf{h} | y < y^*)}{p(\mathbf{h} | y \geq y^*)}$$

More scalable than GP-based BO for high-dimensional hyperparameter spaces.

---

# 11. Quantization

Reduces numerical precision to decrease memory and speed up inference:

| Format | Bits | Memory (1B params) | Relative speed |
| :--- | :--- | :--- | :--- |
| FP32 | 32 | 4 GB | 1× |
| FP16/BF16 | 16 | 2 GB | ~2× |
| INT8 | 8 | 1 GB | ~4× |
| INT4 | 4 | 0.5 GB | ~6-8× |

**Post-training quantization (PTQ):** quantize a trained model without retraining. Fast but can lose accuracy.

**Quantization-aware training (QAT):** simulate quantization during training using straight-through estimator for gradients. Recovers most accuracy lost by PTQ.

**Scale factor:** for INT8, $x_{\text{int8}} = \text{round}(x_{\text{fp32}} / s)$ where $s = \max(|x|) / 127$.

---

# 12. Practical Training Debugging Checklist

When training is unstable or not converging:

1. **Loss NaN/Inf** → learning rate too high, or numerical instability in log/exp ops
2. **Loss not decreasing** → LR too low, wrong loss function, data bug, label issue
3. **Train loss decreases, val does not** → overfitting — increase regularization, more data, dropout
4. **Val loss decreases then spikes** → LR too high, use warmup or schedule
5. **Gradient norm exploding** → add gradient clipping (`max_norm=1.0`)
6. **Gradient norm near zero** → vanishing gradients — check activations, initialization

```python
# Log gradient norms during training
total_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
```

---

# 13. Optimizer Comparison

| Optimizer | Adaptive LR | Momentum | Weight decay | Best for |
| :--- | :--- | :--- | :--- | :--- |
| SGD | No | Optional | L2 coupled | Vision (careful tuning) |
| SGD + Nesterov | No | Yes (lookahead) | L2 coupled | ResNets |
| Adagrad | Yes (cumulative) | No | No | Sparse NLP features |
| RMSprop | Yes (EMA) | No | No | RNNs, RL |
| Adam | Yes (EMA) | Yes | Coupled (wrong) | Fast iteration |
| AdamW | Yes (EMA) | Yes | Decoupled | Transformers, LLMs |
