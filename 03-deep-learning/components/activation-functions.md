# Activation Functions

Activation functions are what stop neural networks from becoming glorified spreadsheet formulas.

Without them, stacking layers would still collapse into something linear and boring.

---

# 1. Why Activations Exist

Activations introduce **non-linearity**.

That is what lets a network learn curves, boundaries, interactions, and richer structure.

Without activation functions, depth would look impressive but behave disappointingly.

**Proof by collapse:** If $f(x) = W_2(W_1 x) = (W_2 W_1)x$, stacking linear layers is equivalent to one linear layer.

---

# 2. Formulas and Properties

## Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Range: $(0, 1)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, max value **0.25**
- Problem: saturates at extremes → gradients vanish in deep nets
- Use: binary output layers only

## Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

- Range: $(-1, 1)$
- Derivative: $1 - \tanh^2(z)$, max value **1.0**
- Zero-centered (advantage over sigmoid in hidden layers)
- Still saturates → still has vanishing gradient issues

## ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

- Range: $[0, \infty)$
- Derivative: 1 if $z > 0$, 0 if $z < 0$ (undefined at 0, set to 0 in practice)
- **Why it dominated:** simple, cheap, avoids saturation for positive inputs

**Dead ReLU problem:** neurons with $z < 0$ always produce 0 gradient → permanently stop learning. Caused by large learning rates or bad initialization.

**Fix:** use small positive learning rate, He initialization, or switch to Leaky ReLU/ELU.

## Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

Typical $\alpha = 0.01$. Prevents dead neurons by keeping a small gradient for negatives.

## ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

Smooth for negative values; output can be negative → zero-centered mean activations → faster convergence than ReLU in some settings.

## GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)$$

where $\Phi$ is the standard Gaussian CDF.

- Smooth, differentiable everywhere
- Non-monotonic for negative inputs (unlike ReLU)
- Default in GPT, BERT, ViT and most modern Transformers

## Swish / SiLU

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Self-gated variant. Similar properties to GELU, slightly cheaper to compute.

## SwiGLU (used in LLaMA, PaLM, Gemini)

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)$$

Two linear projections: one is gated by Swish, element-wise multiplied with the other. The gate controls how much of each dimension flows forward.

Typically used in the FFN sub-layer of Transformers:

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_3) \cdot W_2$$

Reduces FFN hidden dim by factor $2/3$ to compensate for the extra projection while keeping parameter count constant.

## Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- Outputs sum to 1 → probability distribution over classes
- Numerically stable form: subtract max before exponentiating

---

# 3. Comparison Table

| Activation | Range | Zero-centered | Gradient vanishes | Dead neurons | Use case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $(0,1)$ | No | Yes | No | Binary output |
| **Tanh** | $(-1,1)$ | Yes | Yes | No | Rare hidden layers |
| **ReLU** | $[0,\infty)$ | No | No (positive) | Yes | CNN hidden layers |
| **Leaky ReLU** | $(-\infty,\infty)$ | No | No | No | Hidden layers, safe ReLU |
| **ELU** | $(-\alpha,\infty)$ | Near-zero mean | No | No | Hidden layers |
| **GELU** | $\approx(-0.17,\infty)$ | No | No | No | Transformers, LLMs |
| **Swish** | $\approx(-0.28,\infty)$ | No | No | No | EfficientNet, modern CNNs |
| **Softmax** | $(0,1)$ | — | — | — | Multiclass output |

---

# 4. Which Activation to Use Where

## Hidden layers
- **Default:** ReLU (CNNs, simple feedforward)
- **Transformers/LLMs:** GELU or SwiGLU
- **Modern CNNs:** Swish (EfficientNet)
- **When dead neurons are a problem:** Leaky ReLU or ELU

## Output layer

| Task | Activation | Why |
| :--- | :--- | :--- |
| Regression | None (identity) | Unbounded output |
| Binary classification | Sigmoid | Output in $(0,1)$ = probability |
| Multi-class (exclusive) | Softmax | Outputs sum to 1 |
| Multi-label | Sigmoid (per-class) | Each label independent |
| Sequence generation (token) | Softmax over vocabulary | Probability distribution |

---

# 5. Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Using nn.Module activations
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.GELU(),                     # Transformer-style
    nn.Linear(128, 64),
    nn.ReLU(),                     # Standard CNN-style
    nn.Linear(64, 10),
    # No activation here — CrossEntropyLoss expects raw logits
)

# Manual usage
x = torch.randn(32, 128)
out_relu = F.relu(x)
out_gelu = F.gelu(x)
out_sigmoid = torch.sigmoid(x)
out_softmax = F.softmax(x, dim=-1)  # Always specify dim

# Numerically stable log-softmax + NLLLoss = CrossEntropyLoss
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, targets)
# Equivalent to:
loss = F.cross_entropy(logits, targets)
```

---

# 6. Mini Pop Quiz

If multiple classes can all be true simultaneously (multi-label), use **sigmoid** (not softmax).

Sigmoid: each output is independent, not competing.
Softmax: forces outputs to sum to 1 — mutual exclusivity assumption.

Also: using sigmoid in a deep hidden layer is usually a mistake — vanishing gradients will slow training. Use ReLU family instead.

---

# 7. Weight Initialization (activation-dependent)

Bad initialization → dead neurons (ReLU) or vanishing gradients (Sigmoid/Tanh).

## Xavier / Glorot Initialization

Designed for symmetric activations (Sigmoid, Tanh):

$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right], \quad \sigma = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

Maintains variance of activations and gradients through layers.

## He / Kaiming Initialization

Designed for ReLU (accounts for half the neurons being zeroed):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

The factor of 2 compensates for ReLU setting negative activations to 0.

| Activation | Init | Gain |
| :--- | :--- | :--- |
| Sigmoid, Tanh | Xavier | $\sqrt{2/(n_{in}+n_{out})}$ |
| ReLU | He (fan-in) | $\sqrt{2/n_{in}}$ |
| Leaky ReLU $(\alpha)$ | He variant | $\sqrt{2/(1+\alpha^2) \cdot n_{in}}$ |
| GELU / Swish | He (empirically similar to ReLU) | $\sqrt{2/n_{in}}$ |

```python
import torch.nn as nn

# PyTorch applies He init by default for Linear/Conv layers
# Explicit:
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
```
