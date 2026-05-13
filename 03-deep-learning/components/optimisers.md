# Optimisers

Optimizers are how the model actually moves through the loss landscape.

Same model. Same data. Different optimizer. Very different training experience.

---

# 1. Gradient Descent Family

**Core idea:** compute gradient, move weights downhill:

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

**Variants by batch size:**

| Variant | Batch size | Gradient quality | Speed |
| :--- | :--- | :--- | :--- |
| **Batch GD** | Full dataset | Low variance | Slow |
| **SGD** | 1 sample | Very noisy | Very fast per step |
| **Mini-batch SGD** | 32–512 | Balanced | Practical default |

Mini-batch SGD is the practical default for deep learning: fast computation with GPU parallelism, noisy enough to escape local minima.

---

# 2. SGD with Momentum

**Momentum** accumulates a velocity vector to keep updates moving in useful directions:

$$v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta L$$
$$\theta \leftarrow \theta - \eta v_t$$

Typical $\beta = 0.9$. Think of it as exponentially weighted moving average of past gradients.

**Nesterov momentum** (lookahead variant): compute gradient at the "looked-ahead" position → slightly better convergence:

$$v_t = \beta v_{t-1} + \eta \nabla_\theta L(\theta - \beta v_{t-1})$$

```python
import torch.optim as optim

sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True,
                weight_decay=1e-4)
```

---

# 3. Adam

Adam (Adaptive Moment Estimation) combines momentum with adaptive per-parameter learning rates:

**First moment** (mean, like momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment** (uncentered variance):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction** (critical early in training when $m_t, v_t$ are near zero):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**
$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3\times 10^{-4}$.

The $3\times 10^{-4}$ learning rate is widely cited as the "Karpathy constant" — reliable default for Adam.

---

# 4. AdamW

Adam's weight decay is implemented via the gradient update, which couples it with the adaptive scaling. This means regularization is weaker for parameters with large gradients.

**AdamW decouples** weight decay from the gradient update:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

This is the correct way to do L2 regularization with Adam. **Always use AdamW over Adam** for modern deep learning.

```python
adamw = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999),
                    weight_decay=0.01)
```

---

# 5. RMSprop

RMSprop adapts the learning rate per parameter using a running average of squared gradients:

$$v_t = \rho v_{t-1} + (1-\rho) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$

Typical $\rho = 0.99$, $\epsilon = 10^{-8}$.

**Key difference from Adam:** no first-moment term (no momentum by default), no bias correction.

Used in: RNNs (historically), RL (still common — A3C, DQN). Adam is preferred for most feedforward tasks.

---

# 5a. Lion (EvoLved Sign Momentum)

Memory-efficient alternative to AdamW — uses sign of momentum update rather than gradient magnitude:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(standard momentum)}$$
$$\theta_t = \theta_{t-1} - \eta \left( \text{sign}(\beta_2 m_{t-1} + (1-\beta_2) g_t) + \lambda \theta_{t-1} \right)$$

- Update direction: $\text{sign}(\cdot) \in \{-1, +1\}$ — every parameter moves by exactly the same step size $\eta$
- Weight decay decoupled (like AdamW)
- Memory: stores one momentum vector vs Adam's two (saves memory for very large models)
- Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.99$, lower LR than AdamW (step magnitude is fixed)

```python
# pip install lion-pytorch
from lion_pytorch import Lion

optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

---

# 6. When SGD Still Wins

Despite Adam's popularity, SGD with momentum + careful LR scheduling still wins in some vision settings (e.g., ResNet training on ImageNet with cosine annealing).

Why? Adam can generalize slightly worse because it adapts too aggressively and "memorizes" bad local structure. SGD's noisiness can act as implicit regularization.

**Rule of thumb:**
- **Transformers / LLMs / NLP:** AdamW (always)
- **CNNs / vision:** try AdamW first; consider SGD+momentum if generalization matters more than speed

---

# 6. Learning Rate Scheduling

Learning rate is often more important than the optimizer choice.

## Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

Smoothly decays from max to min. Common in vision and NLP.

## Warmup + Cosine Decay (standard for LLMs)

Start with very small LR, ramp up for first $T_w$ steps, then cosine decay:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

## Step Decay and ReduceLROnPlateau

```python
# Decay LR by factor of 0.1 every 30 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Decay when validation metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

---

# 7. Comparison Table

| Optimizer | Adaptive LR | Momentum | Weight Decay | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | No | Optional | Yes (L2) | Vision (careful tuning) |
| **SGD+Momentum** | No | Yes | Yes | ResNets, CNNs |
| **Adam** | Yes | Yes | Coupled | Quick iteration, NLP |
| **AdamW** | Yes | Yes | Decoupled | Transformers, LLMs (default) |
| **Lion** | No | Yes | Yes | Memory-efficient alternative |

---

# 8. Code Example — Full Training Loop

```python
import torch
import torch.optim as optim

model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
```

---

# 9. Gradient Clipping

Prevents exploding gradients by capping the global gradient norm before the update step.

**Norm clipping (preferred):**

$$\text{if } \|\mathbf{g}\|_2 > \tau: \quad \mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|_2} \mathbf{g}$$

Scales the entire gradient vector uniformly when it exceeds threshold $\tau$. Preserves gradient direction.

**Value clipping:** clip each gradient element to $[-\tau, \tau]$. Changes direction — generally less preferred.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # norm clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  # value clipping
```

Typical max_norm: 1.0 for Transformers/LLMs, 5.0 for RNNs. Always apply after `loss.backward()`, before `optimizer.step()`.

---

The clean answer is not "Adam is always best." It is:

> "AdamW with warmup + cosine decay is the reliable default for Transformers. For CNNs, benchmark against SGD+momentum."
