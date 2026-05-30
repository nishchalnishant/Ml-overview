---
module: Deep Learning
topic: Components
subtopic: Optimisers
status: unread
tags: [deeplearning, ml, components-optimisers]
---
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

## Additional Learning Rate Schedules

### Step Decay

Reduce learning rate by a fixed factor every $k$ epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

Typical: $\gamma = 0.1$ every 30 epochs (ResNet-style). Simple and interpretable but has discontinuous drops — the loss often temporarily increases at each step boundary.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Polynomial Decay

Decay from $\eta_\text{max}$ to $\eta_\text{min}$ following a polynomial curve over $T$ steps:

$$\eta_t = (\eta_\text{max} - \eta_\text{min}) \cdot \left(1 - \frac{t}{T}\right)^p + \eta_\text{min}$$

With $p=1$ this is linear decay; $p=2$ gives a slower initial decay and steeper final decay. Used in BERT, some ViT trainings.

```python
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=T, power=p)
```

### One-Cycle LR

Popularized by fastai. Ramps from a low initial LR up to a maximum, then decays steeply — completing one full cycle over the entire training run:

1. Phase 1 (45% of steps): LR increases from `max_lr/div_factor` to `max_lr`
2. Phase 2 (45% of steps): LR decreases from `max_lr` to `max_lr/div_factor`
3. Phase 3 (10% of steps): LR anneals from `max_lr/div_factor` to `max_lr/(div_factor * final_div_factor)`

Simultaneously cycles momentum in the opposite direction (high → low → high).

**What it does well**: converges fast — often achieves competitive accuracy in $5\times$ fewer epochs than fixed-LR training. The high LR phase acts as regularization (large steps prevent memorizing individual batches). Works particularly well for CNNs on vision tasks.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=10,
    pct_start=0.3,          # fraction of cycle spent increasing LR
    div_factor=25.0,        # initial LR = max_lr / div_factor
    final_div_factor=1e4    # final LR = max_lr / (div_factor * final_div_factor)
)
```

### ReduceLROnPlateau

Monitors a validation metric; reduces LR when the metric stops improving for `patience` epochs:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # 'min' for loss, 'max' for accuracy
    factor=0.1,         # new_lr = old_lr * factor
    patience=10,        # epochs without improvement before reduction
    threshold=1e-4,     # minimum change to count as improvement
    min_lr=1e-7
)

# Must be stepped with validation metric
val_loss = validate(model, val_loader)
scheduler.step(val_loss)
```

**What it does well**: model-agnostic and adaptive — no need to specify a schedule in advance. Appropriate when training duration is uncertain or validation behavior is unpredictable.

**What breaks**: requires frequent validation. With large datasets or expensive validation, the patience+evaluation cost can dominate training time. Also: ReduceLROnPlateau reacts to validation noise — a single bad validation step can trigger premature LR reduction. The `threshold` parameter mitigates this.

---

## Gradient Accumulation

**The problem**: training large models with large effective batch sizes requires large GPU memory. An LLM training run might need a global batch size of 2048 or more for stable optimization, but fitting 2048 sequences at once in GPU memory is impossible.

**The core insight**: accumulate gradients over multiple small forward-backward passes before calling `optimizer.step()`. The optimizer sees gradients from $k$ micro-batches summed together, which is mathematically equivalent to a single pass over all $k$ micro-batches at once.

**The mechanics**: run $k$ forward-backward passes without resetting gradients or stepping the optimizer. Gradients accumulate in `.grad` attributes by default (they sum, not replace). After $k$ steps, divide the accumulated gradients by $k$ (to keep the scale correct) and take one optimizer step.

$$\text{effective batch size} = \text{micro\_batch\_size} \times k$$

```python
optimizer.zero_grad()
accumulation_steps = 8   # effective batch size = micro_batch * 8

for step, (inputs, labels) in enumerate(dataloader):
    # Forward + backward — gradients accumulate
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps   # normalize loss to match single-batch scale
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        # Now gradients represent the average over accumulation_steps batches
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**Why divide the loss**: PyTorch accumulates gradients by summation. If you backward over $k$ batches without dividing, the gradient magnitude is $k\times$ larger than expected — equivalent to multiplying the learning rate by $k$. Dividing `loss / k` before backward keeps gradient magnitudes consistent.

**What breaks**:
- BatchNorm statistics are computed per micro-batch, not across the full accumulated batch. If BatchNorm is present, large $k$ makes per-micro-batch statistics noisy. LayerNorm does not have this problem.
- Gradient checkpointing interacts with accumulation: you need to be careful not to clear activation checkpoints between accumulation steps.
- Mixed precision (AMP) with gradient accumulation requires the GradScaler to be called only on the optimizer step, not on every backward.

---

## Mixed Precision Training (AMP)

**The problem**: full float32 training is memory-inefficient and slow. A 7B model at fp32 requires 28GB just for weights, plus 84GB for Adam optimizer states — 112GB total before accounting for activations or gradients. Modern GPU hardware (Ampere, Hopper) has specialized tensor cores that run fp16/bf16 matrix multiplications 2–4× faster than fp32.

**The core insight**: most forward-pass computation is numerically stable at half precision. Only certain operations (loss accumulation, weight updates, normalizations) require full precision to avoid gradient underflow or weight update instability. Use fp16/bf16 for forward passes and activations; maintain fp32 master weights for the optimizer update.

**PyTorch AMP mechanics**:

1. **Autocast context**: automatically casts operations to fp16 or bf16. Matrix multiplications, convolutions, attention — all run at half precision. Layer norm, softmax, loss — kept at fp32 automatically.
2. **GradScaler** (fp16 only): fp16 has a limited dynamic range ($\approx 10^{-4}$ to $65504$). Gradients can underflow to zero. The scaler multiplies the loss by a large factor before backward, inflating gradients to avoid underflow, then divides them back before the optimizer step. The scale is automatically adjusted: if inf/nan appears (overflow), scale is halved; if no overflow for $N$ steps, scale is doubled.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()   # only needed for fp16, not bf16

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for inputs, labels in dataloader:
    inputs, labels = inputs.cuda(), labels.cuda()

    # Forward pass at fp16/bf16 precision
    with autocast(dtype=torch.bfloat16):    # or torch.float16
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # Backward pass: GradScaler handles fp16 gradient scaling
    scaler.scale(loss).backward()

    # Unscale gradients before clipping (clipping operates on unscaled gradients)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step — GradScaler checks for inf/nan; skips update if found
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**fp16 vs bf16**:

| | fp16 | bf16 |
| :--- | :--- | :--- |
| Mantissa bits | 10 | 7 |
| Exponent bits | 5 | 8 |
| Max value | 65504 | ~3.4 × 10³⁸ |
| Precision | Higher | Lower |
| Range | Narrow (overflow risk) | Same as fp32 (no overflow) |
| GradScaler needed | Yes | No |
| Recommendation | Pre-Ampere GPUs | Ampere+ (A100, H100) |

bf16 has the same exponent range as fp32, so gradient overflow is not a concern — no GradScaler required. This makes bf16 training simpler and equally fast. **For Ampere+ hardware, bf16 is preferred over fp16.**

**Memory savings**:
- Weights: fp32 (4 bytes) → fp16/bf16 (2 bytes) — 2× weight memory reduction
- Activations: same — 2× activation memory reduction
- Optimizer states: kept in fp32 for stability — no reduction
- Net: roughly 1.5× overall memory reduction

**What breaks**:
- Certain operations must remain in fp32: loss accumulation over very long sequences can underflow in fp16; normalizations that depend on squared magnitudes (LayerNorm, BatchNorm) need fp32 precision.
- AMP's autocast context handles most of this automatically, but custom operations may need explicit `dtype` specification.
- With gradient accumulation, the `scaler.step()` and `scaler.update()` calls should only happen at the actual optimizer step, not at every micro-batch backward pass.

---

## Optimiser Comparison

| Optimiser | Adaptive LR | Momentum | Memory | Best for |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | No | Optional | $1\times$ params | Vision (careful tuning) |
| **Adam** | Yes | Yes | $3\times$ params | Quick prototyping |
| **AdamW** | Yes | Yes (decoupled WD) | $3\times$ params | Transformers, LLMs (default) |
| **RMSprop** | Yes | No | $2\times$ params | RL agents |
| **Lion** | No | Yes (sign update) | $2\times$ params | Memory-constrained large models |

---

## Canonical Interview Q&As

**Q: Derive the Adam update rule from first principles and explain why it outperforms SGD with momentum.**  
A: Adam maintains two moment estimates: m_t = β_1·m_{t-1} + (1-β_1)·g_t (1st moment, exponential moving average of gradients) and v_t = β_2·v_{t-1} + (1-β_2)·g_t² (2nd moment, EMA of squared gradients). Both are bias-corrected: m̂_t = m_t/(1-β_1^t), v̂_t = v_t/(1-β_2^t) (early steps are underestimated due to zero initialization). Update: θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε). The key insight: the denominator √v̂_t is a per-parameter learning rate adaptor — parameters with large historical gradients get smaller effective LR, parameters with small gradients get larger effective LR. This makes Adam insensitive to gradient scale, so the global learning rate α doesn't need careful tuning per layer. SGD + momentum only tracks the gradient direction, not the scale — it needs careful per-layer LR tuning or extensive warm-up schedules. In practice, Adam converges faster (10-100× fewer iterations to reach similar loss) and is more robust to initialization.

**Q: What is the difference between Adam and AdamW, and why does L2 regularization work differently in adaptive optimizers?**  
A: In vanilla Adam, adding L2 regularization (weight decay) to the loss means the gradient becomes g_t + λ·θ. This gradient goes through the adaptive scaling: update ∝ (g_t + λ·θ) / √(v_t). For parameters with large gradient variance (large v_t), the regularization effect λ·θ is dampened by the same factor — the effective weight decay is different for each parameter and depends on gradient history. This means Adam doesn't apply uniform L2 regularization. AdamW (Loshchilov & Hutter 2019) decouples weight decay from the gradient: θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε) - α·λ·θ_{t-1}. The weight decay term bypasses the adaptive scaling, applying uniformly to all parameters. In practice, AdamW with λ=0.01-0.1 produces better-regularized models than Adam + L2, especially for LLMs. All modern LLM training uses AdamW.

**Q: Why might you choose a simpler optimizer like SGD over Adam for a specific task?**  
A: SGD with momentum can generalize better than Adam on some tasks — this is the "generalization gap" phenomenon. Adam finds sharp minima quickly; SGD explores more broadly and often finds flatter minima (lower sharpness) that generalize better. Evidence: many CV classification benchmarks (ImageNet ResNets) achieve better test accuracy with SGD + cosine LR than Adam. Intuition: Adam adapts per-parameter LRs to converge fast in the loss landscape it sees during training, which can lead to solutions that fit training data well but are sensitive to distribution shift (sharp minima). SGD is slower but covers more of the loss landscape. Practical rule: for transformer/LLM training → AdamW (faster convergence is critical at scale); for small CNNs with abundant data → try SGD; for production systems where training time is fixed and convergence speed matters → AdamW. Also: SGD requires much less memory (no moment buffers), saving 8 bytes/param vs Adam's 12 bytes/param.

## Flashcards

**Full-batch gradient descent?** #flashcard
use all data to compute gradient. Low variance, accurate gradient, but prohibitively slow for large datasets.

**SGD (stochastic gradient descent): use one example. Very noisy gradient?** #flashcard
updates jump around. Fast, but oscillates.

**Mini-batch SGD?** #flashcard
use 32–512 examples. Practical default: fast GPU parallelism, noisy enough to escape local minima, stable enough to converge.
