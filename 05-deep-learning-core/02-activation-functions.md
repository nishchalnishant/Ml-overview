---
module: Deep Learning Core
topic: Components
subtopic: Activation Functions
status: unread
tags: [deeplearning, ml, components-activation-function]
---
# Activation Functions

---

## Sigmoid

**The problem**: a neural network without any non-linearity is just matrix multiplication. No matter how many layers you stack, $W_2(W_1 x) = (W_2 W_1)x$ — the whole thing collapses into a single linear transformation. You need something that bends the function space, not just rotates it.

**The core insight**: squash the output into a bounded range so the neuron behaves like a "soft switch" — values near zero pass little signal, values near one pass a lot.

**The mechanics**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{range } (0,1)$$

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)), \quad \text{max value } 0.25$$

**What breaks**: every layer multiplies the gradient by at most 0.25. In a 10-layer network: $0.25^{10} \approx 10^{-6}$. Early layers receive a gradient of essentially zero and stop learning — the vanishing gradient problem. Additionally, outputs are not zero-centered (always positive), which slows convergence in subsequent layers.

Use sigmoid only in the output layer of a binary classifier. Never in hidden layers of deep networks.

---

## Tanh

**The problem**: sigmoid is not zero-centered — all outputs are positive. When gradients flow back through a sigmoid layer, they are all the same sign, causing zig-zag weight updates and slower convergence.

**The core insight**: rescale and shift sigmoid to be symmetric around zero so positive and negative activations balance out.

**The mechanics**:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \text{range } (-1, 1)$$

$$\tanh'(z) = 1 - \tanh^2(z), \quad \text{max value } 1.0$$

Zero-centered outputs. Strictly preferred over sigmoid in hidden layers when you must use a saturating activation.

**What breaks**: still saturates. For large $|z|$, the derivative $\approx 0$ and gradients vanish through deep stacks. Practically the same problem as sigmoid, just less severe.

---

## ReLU

**The problem**: sigmoid and tanh squash large inputs into a tiny range. In a 10-layer network with sigmoid activations, gradients shrink by at most $0.25$ at each step — by the time the signal reaches the first layer, it has essentially vanished.

**The core insight**: use an activation that does not saturate for positive values. If the gradient does not shrink, it can travel through arbitrarily many layers.

**The mechanics**:

$$\text{ReLU}(z) = \max(0, z)$$

Derivative is exactly 1 for $z > 0$ — no shrinkage. Derivative is exactly 0 for $z < 0$ — that neuron contributes nothing this pass.

**What breaks**: the dying ReLU problem. If a neuron receives a large negative pre-activation (e.g., due to a large negative bias or a bad gradient update), its derivative is 0 and no gradient flows back through it. It cannot update its weights and remains dead permanently. A network with many dead neurons has less capacity and trains poorly.

---

## Leaky ReLU

**The problem**: a dead ReLU cannot recover — its gradient is identically zero for all negative inputs, so no update ever pushes the bias back into the positive regime.

**The core insight**: keep a tiny gradient for negative inputs so a dead neuron can, in principle, receive signal and revive.

**The mechanics**:

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha = 0.01 \text{ typical}$$

Gradient is $\alpha$ for negative inputs instead of zero. Neurons can revive.

**What breaks**: $\alpha$ is a fixed hyperparameter — it does not adapt to the data. Parametric ReLU (PReLU) learns $\alpha$ per neuron, but adds parameters and overfitting risk.

---

## ELU

**The problem**: ReLU activations are non-negative on average — the mean activation is positive, creating a bias shift. Each layer's output has a non-zero mean, which must be compensated by biases downstream and slows convergence.

**The core insight**: let the negative regime produce smoothly negative values so the mean activation over a batch is closer to zero.

**The mechanics**:

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

For $z \ll 0$, ELU saturates at $-\alpha$ (typically $-1$), giving a mean activation near zero. The smooth negative regime avoids the sharp discontinuity at zero.

**What breaks**: the exponential is more expensive to compute than a max operation. Saturates for very negative inputs (though at a constant, not at zero), so neurons that consistently see very negative inputs do not learn much.

---

## GELU

**The problem**: ReLU is a hard gate — a neuron is either fully on or fully off. This discontinuity at zero can cause instability and sharp loss landscapes, and the all-or-nothing behavior may not be ideal when inputs cluster near zero.

**The core insight**: gate the input by the probability it would be kept under a standard Gaussian — a soft, probabilistic masking that smoothly transitions between zero and full pass-through.

**The mechanics**:

$$\text{GELU}(z) = z \cdot \Phi(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)$$

where $\Phi$ is the standard Gaussian CDF. Small positive inputs are passed through at partial strength; large positive inputs are passed through fully; negative inputs are nearly zeroed.

**What breaks**: non-monotonic for small negative values — GELU can produce slightly negative outputs for inputs around $-0.17$. This is a feature (more expressive), but can occasionally cause unexpected behavior in networks that assume monotone activations.

Default in GPT, BERT, ViT, and most modern Transformers.

---

## Swish / SiLU

**The problem**: GELU's Gaussian CDF approximation has a complex formula. Can we get the same smooth, self-gating behavior more cheaply?

**The core insight**: replace the Gaussian CDF with sigmoid — which is already computed cheaply and has a similar shape.

**The mechanics**:

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Self-gated: the input modulates itself. Similar properties to GELU — smooth, non-monotonic near zero — but cheaper to compute.

**What breaks**: still has the non-monotonic property near zero. Slightly more expensive than ReLU due to the sigmoid computation, though cheaper than GELU's full approximation.

Used in EfficientNet family. Under the name SiLU in PyTorch.

---

## SwiGLU

**The problem**: the FFN sub-layer in Transformers uses ReLU between two linear projections. Better activations exist, but to use them with a gating mechanism — which empirically improves quality — requires a third projection, increasing parameter count.

**The core insight**: use two separate linear projections: one goes through Swish (the gate), the other passes through directly. Multiply them element-wise. The gate dynamically controls how much of each dimension flows forward. Reduce the hidden dimension to $2/3$ of the standard $4d$ to keep parameter count equal.

**The mechanics**:

$$\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)$$

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_3) \cdot W_2$$

Three projections ($W_1$, $W_2$, $W_3$), hidden dimension $\frac{8}{3}d$ instead of $4d$, keeping total parameters matched.

**What breaks**: requires the third projection, which adds memory and compute overhead proportional to $d^2$. The parameter-count compensation (reducing hidden dim to $2/3$) means each projection is narrower — slightly less expressiveness per parameter than vanilla FFN, empirically compensated by the gating mechanism.

Used in LLaMA, PaLM, Gemini.

---

## Softmax

**The problem**: for multiclass classification, the network outputs a vector of raw scores (logits). These scores have no natural probabilistic interpretation — they can be negative, arbitrarily large, and do not sum to one.

**The core insight**: exponentiate to make all values positive, then normalize so they sum to one. The resulting values are a proper probability distribution over the classes.

**The mechanics**:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Numerically stable form: subtract the max before exponentiating to prevent overflow. $\text{softmax}(z - \max(z))$ gives identical results.

**What breaks**: softmax assumes mutual exclusivity — increasing one class's probability necessarily decreases the others. For multi-label tasks (multiple classes can be true simultaneously), use sigmoid per output instead. Softmax also amplifies the largest logit exponentially — at low temperatures it collapses to a one-hot vector; at high temperatures it approaches uniform.

---

## Comparison Table

| Activation | Range | Zero-centered | Gradient vanishes | Dead neurons | Use case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $(0,1)$ | No | Yes | No | Binary output only |
| **Tanh** | $(-1,1)$ | Yes | Yes | No | Rare; prefer ReLU family |
| **ReLU** | $[0,\infty)$ | No | No (positive) | Yes | CNN hidden layers |
| **Leaky ReLU** | $(-\infty,\infty)$ | No | No | No | Safe ReLU replacement |
| **ELU** | $(-\alpha,\infty)$ | Near-zero mean | No | No | Hidden layers |
| **GELU** | $\approx(-0.17,\infty)$ | No | No | No | Transformers, LLMs |
| **Swish** | $\approx(-0.28,\infty)$ | No | No | No | EfficientNet, modern CNNs |
| **SwiGLU** | — | — | No | No | Transformer FFN layers |
| **Softmax** | $(0,1)$ | — | — | — | Multiclass output |

---

## Output Layer Activation by Task

| Task | Activation | Why |
| :--- | :--- | :--- |
| Regression | None | Unbounded output |
| Binary classification | Sigmoid | Output in $(0,1)$ = probability |
| Multi-class (exclusive) | Softmax | Outputs sum to 1 |
| Multi-label | Sigmoid (per-class) | Each label independent |
| Token generation | Softmax over vocabulary | Probability distribution |

---

## Canonical Interview Q&As

**Q: Why did GELU replace ReLU in transformers, and what is GELU computing?**
A: GELU (Gaussian Error Linear Unit) is defined as GELU(x) = x·Φ(x), where Φ(x) is the standard normal CDF. Intuitively: the input x is gated by the probability that a standard normal random variable is less than x — inputs far below zero are dropped (near-zero probability), inputs far above zero pass through almost entirely. Unlike ReLU which is a hard gate (0 or 1), GELU is smooth everywhere, including at zero. This matters because: (1) smooth gradients near zero avoid the "dead neuron" problem where ReLU neurons with x<0 get zero gradient permanently; (2) GELU has a small negative region below zero (unlike ReLU's strict non-negativity), giving the network more expressiveness; (3) empirically, BERT, GPT-2, and all subsequent transformers trained with GELU converge better than ReLU. The practical approximation used in code is GELU(x) ≈ 0.5x·(1 + tanh(√(2/π)·(x + 0.044715x³))), which avoids computing the true CDF.

**Q: What is SwiGLU and why do all modern LLMs use it?**
A: SwiGLU (Swish-Gated Linear Unit) replaces the standard FFN block FFN(x) = ReLU(xW_1)W_2 with a gated variant: SwiGLU(x) = (xW ⊙ SiLU(xV))W_2, where SiLU(x) = x·σ(x) (Swish activation). The key change is the multiplicative gate — the linear projection xW is element-wise multiplied by the activated projection SiLU(xV). This gating mechanism lets the network learn to selectively suppress or amplify feature dimensions. Because there are now 3 weight matrices (W, V, W_2) instead of 2, the intermediate hidden dimension is typically scaled to 2/3 of the standard 4d_model to keep total parameter count constant. Empirically, SwiGLU achieves the same validation loss as ReLU FFNs in ~15-20% fewer training steps (Noam Shazeer 2020). All Llama variants, Gemma, Mistral, and PaLM use SwiGLU. Interview answer: "it combines smooth gradients from Swish with the gating mechanism's selective feature control — empirically the best FFN activation at scale."

**Q: Why does the dying ReLU problem occur, and how do LeakyReLU and ELU address it differently?**
A: ReLU(x) = max(0, x). For x < 0, the output is 0 and the gradient is exactly 0. If a neuron's pre-activation is negative for every sample in a batch, it receives zero gradient — the weights don't update, and the neuron is permanently "dead." This happens when: large learning rates push weights such that the neuron always produces negative pre-activations, or poor initialization creates systematic negative outputs. LeakyReLU: f(x) = x if x > 0, else αx (α ≈ 0.01). The small slope for x < 0 ensures gradient is never exactly zero — dead neurons can recover. ELU (Exponential Linear Unit): f(x) = x if x > 0, else α(e^x - 1). ELU has negative saturation (approaches -α asymptotically), which makes the mean activation closer to zero (like batch normalization's effect) — this accelerates learning. ELU also has smooth gradient everywhere including at x=0. Trade-off: LeakyReLU is faster to compute; ELU converges faster but costs an exp() per activation. In practice for transformers: both are superseded by GELU/SwiGLU. For lightweight CNNs where compute matters: LeakyReLU is standard.
