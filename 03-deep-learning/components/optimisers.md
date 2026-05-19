# Optimisers

---

## Gradient Descent

**The problem**: you have a loss $L(\theta)$ and you want to find weights $\theta$ that minimize it. The loss surface is a high-dimensional landscape and you cannot enumerate all possible weight settings. You need a systematic way to move downhill.

**The core insight**: the gradient $\nabla_\theta L$ points in the direction of steepest *ascent*. Moving in the *negative* gradient direction descends the loss.

**The mechanics**:

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

$\eta$ is the learning rate — how far to step in the gradient direction each update.

**What breaks**: choosing the right learning rate is hard. Too large: you overshoot minima and the loss oscillates or diverges. Too small: training is so slow it never converges in a practical time. A fixed learning rate also treats all parameters identically — a parameter whose gradient is consistently large gets the same step size as one whose gradient is consistently tiny, even though they may be at very different scales.

**Batch size variants**:
- **Full-batch gradient descent**: use all data to compute gradient. Low variance, accurate gradient, but prohibitively slow for large datasets.
- **SGD (stochastic gradient descent)**: use one example. Very noisy gradient — updates jump around. Fast, but oscillates.
- **Mini-batch SGD**: use 32–512 examples. Practical default: fast GPU parallelism, noisy enough to escape local minima, stable enough to converge.

---

## SGD with Momentum

**The problem**: plain SGD oscillates in directions where the loss surface has high curvature (e.g., ravines) and moves slowly in consistent directions. The gradient at each step is dominated by local noise.

**The core insight**: accumulate a velocity that averages past gradients. The velocity builds up in consistent directions (gradients that keep pointing the same way), dampens in oscillating directions (gradients that cancel out). Like a ball rolling downhill — it accelerates in the direction of persistent slope and smooths out surface roughness.

**The mechanics**:

$$v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta L$$

$$\theta \leftarrow \theta - \eta v_t$$

Typical $\beta = 0.9$ — each update is 90% previous velocity, 10% new gradient.

**Nesterov momentum**: compute the gradient *after* a preliminary step in the current velocity direction. The gradient is evaluated at where you will be, not where you are — slightly more accurate correction:

$$v_t = \beta v_{t-1} + \eta \nabla_\theta L(\theta - \beta v_{t-1})$$

**What breaks**: momentum can cause overshooting when the loss surface curves sharply. The accumulated velocity carries the optimizer past the minimum. Learning rate must typically be lower with momentum than without.

---

## Adam

**The problem**: SGD with momentum uses one global learning rate $\eta$ for all parameters. Some parameters have sparse gradients (they rarely receive non-zero gradient — e.g., word embeddings for rare words). A global learning rate is simultaneously too large for frequently updated parameters and too small for rarely updated ones.

**The core insight**: maintain a separate effective learning rate for each parameter, adapted based on the history of that parameter's gradients. Parameters with large historical gradients get a smaller effective learning rate; parameters with small or sparse gradients get a larger one. Combine this with momentum.

**The mechanics**:

First moment (momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

Second moment (uncentered variance, tracking gradient magnitudes):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Bias correction (early in training, $m_t$ and $v_t$ are biased toward zero because they are initialized at 0):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update:
$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The denominator $\sqrt{\hat{v}_t}$ scales down the learning rate for parameters with large gradient history and scales it up for parameters with small gradient history.

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3 \times 10^{-4}$ (the "Karpathy constant" — reliable default for Adam).

**What breaks**: Adam's adaptive learning rates can cause poor generalization on some tasks. The optimizer may "memorize" the gradient magnitudes of noisy early training examples, making it slow to change course for novel inputs. On clean vision benchmarks, SGD+momentum with careful scheduling often slightly outgeneralized Adam historically.

---

## AdamW

**The problem**: Adam's weight decay is implemented by adding $\lambda w$ to the gradient before the adaptive scaling. This means the regularization is not applied uniformly — parameters with large gradients get less effective regularization because their denominator $\sqrt{\hat{v}}$ is large. The coupling between weight decay and gradient scaling is unintended.

**The core insight**: apply weight decay directly to the weights after the gradient update, completely decoupled from the adaptive scaling. Weight decay should shrink all weights proportionally, regardless of their gradient history.

**The mechanics**:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

The $\lambda \theta_{t-1}$ term subtracts a fixed fraction of the current weight directly — this is true L2 regularization. The adaptive step handles gradient direction and magnitude separately.

Always use AdamW over Adam for modern deep learning. The decoupled weight decay improves generalization, especially for Transformers where weight norms can grow unboundedly without it.

**What breaks**: if weight decay $\lambda$ is too large, it dominates the gradient update and the model shrinks aggressively — underfitting. The right value depends on the learning rate (they interact), so tune them together. Typical values: $\lambda = 0.01$ to $0.1$.

---

## RMSprop

**The problem**: Adam requires storing two moment vectors. For some applications (RL, online learning), you want the adaptive learning rate behavior with simpler mechanics and faster updates.

**The core insight**: maintain only the running average of squared gradients (second moment), and divide the gradient by its root mean square. No momentum, no bias correction.

**The mechanics**:

$$v_t = \rho v_{t-1} + (1-\rho) g_t^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$

Typical $\rho = 0.99$.

**What breaks**: without momentum, RMSprop is noisier than Adam. Without bias correction, the effective learning rate is artificially large in the first few steps (when $v_t$ is near zero). Adam is almost always preferred — RMSprop is mainly used in RL codebases (A3C, DQN) due to historical precedent.

---

## Lion

**The problem**: Adam stores two moment vectors (first and second moment). For very large models, this doubles or triples the optimizer memory footprint relative to the model parameters themselves. A 70B parameter model at BF16 is ~140GB; Adam optimizer states add another ~280GB. Training requires hundreds of GB of GPU memory just for the optimizer.

**The core insight**: discard the second moment entirely. Use only the sign of the update direction. Every parameter moves by exactly the same magnitude $\eta$ per step — only the direction matters.

**The mechanics**:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(standard momentum, for the next step's update)}$$

$$\theta_t = \theta_{t-1} - \eta \left(\text{sign}(\beta_2 m_{t-1} + (1-\beta_2) g_t) + \lambda \theta_{t-1}\right)$$

The update direction is a sign — either $+\eta$ or $-\eta$ per parameter. Weight decay is decoupled (like AdamW).

**What breaks**: because every parameter moves by the same step size, Lion requires a lower learning rate than AdamW (roughly $3$–$10\times$ smaller). If the learning rate is not adjusted down, Lion updates are effectively larger than AdamW's adaptive steps for parameters with small gradients. Also: the sign update ignores gradient magnitude entirely — a gradient of $10^{-6}$ and a gradient of $10$ produce the same update magnitude. This can cause instability when gradients have high variance.

---

## When SGD Still Wins

**The problem**: Adam converges faster and requires less learning rate tuning. Why use anything else?

**The core insight**: Adam's adaptive learning rates lower the effective learning rate for parameters that have seen large gradients. This helps early in training (escaping bad regions quickly) but can hurt at convergence — the adaptivity causes the optimizer to overshoot or stall in flat regions that require persistent small updates.

SGD with momentum does not adapt per-parameter. It pushes equally in all gradient directions. On well-tuned vision benchmarks (ImageNet, CIFAR) with cosine LR schedules, this lack of adaptivity acts as implicit regularization and sometimes produces better final test accuracy.

Rule: AdamW for Transformers and NLP — always. For CNNs on vision, benchmark AdamW against SGD+momentum+cosine annealing. The gap has narrowed but SGD occasionally still wins.

---

## Learning Rate Scheduling

**The problem**: the right learning rate at the start of training is different from the right learning rate at the end. Early in training, large steps help escape bad initializations. Late in training, large steps cause oscillation around the minimum.

### Warmup

**The problem**: at the very start of training, gradients are noisy and large because the model's parameters are random. If you start with the full learning rate, the first few updates can send weights to bad regions from which recovery is slow.

**The core insight**: start with a very small learning rate and ramp up linearly over the first $T_w$ steps. After warmup, the model is in a more well-behaved region of the loss surface.

### Cosine Annealing

Smoothly decay the learning rate from maximum to minimum following a cosine curve:

$$\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\frac{\pi t}{T}\right)$$

Avoids the sharp transitions of step decay while achieving low final learning rate for fine convergence.

### Warmup + Cosine Decay (LLM standard)

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps       # linear ramp-up
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)     # cosine decay
```

**What breaks**: the warmup duration matters. Too short and the early noise problem persists. Too long and training is slow to get started. Typical warmup: 1–5% of total training steps for LLMs.

---

## Gradient Clipping

**The problem**: in deep networks or recurrent networks, gradients can grow exponentially large due to the chain rule — multiplying many Jacobians can produce values that overflow numerical precision or cause catastrophically large weight updates.

**The core insight**: cap the gradient norm before applying it. If the gradient vector is too large, scale it down uniformly to a maximum norm. This preserves the gradient direction — only the step size is limited.

**The mechanics**:

$$\text{if } \|g\|_2 > \tau: \quad g \leftarrow g \cdot \frac{\tau}{\|g\|_2}$$

Typical $\tau$: 1.0 for Transformers, 5.0 for RNNs. Always apply after `loss.backward()` and before `optimizer.step()`.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What breaks**: value clipping (clipping each gradient element to $[-\tau, \tau]$ independently) changes the gradient direction, not just the magnitude. This can make optimization inconsistent — parameters that should update together (because their gradients point in a coherent direction) get clipped independently and the direction is corrupted. Norm clipping is almost always preferred.

---

## Optimiser Comparison

| Optimiser | Adaptive LR | Momentum | Memory | Best for |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | No | Optional | $1\times$ params | Vision (careful tuning) |
| **Adam** | Yes | Yes | $3\times$ params | Quick prototyping |
| **AdamW** | Yes | Yes (decoupled WD) | $3\times$ params | Transformers, LLMs (default) |
| **RMSprop** | Yes | No | $2\times$ params | RL agents |
| **Lion** | No | Yes (sign update) | $2\times$ params | Memory-constrained large models |
