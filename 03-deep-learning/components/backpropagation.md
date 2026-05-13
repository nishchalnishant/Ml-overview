# Backpropagation

Backpropagation is how a neural network learns which weights deserve blame.

Forward pass: make a prediction.
Backward pass: assign responsibility for the error.

---

# 1. The Core Idea

Backpropagation uses the **chain rule** to compute how the final loss changes with respect to each weight.

Instead of differentiating everything from scratch for every parameter, it reuses intermediate derivatives efficiently.

That efficiency is the reason deep learning is practical at all.

---

# 2. Chain Rule — The Math

For a composition $L = f(g(h(x)))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

For a single neuron: $z = wx + b$, $a = \sigma(z)$, $L = \text{loss}(a, y)$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = \frac{\partial L}{\partial a} \cdot \sigma'(z) \cdot x$$

---

# 3. Concrete Numerical Example

Consider a 2-layer network on a single input $x = 1$, target $y = 0$:

**Forward pass:**
- Layer 1: $z_1 = w_1 \cdot x = 0.5 \cdot 1 = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$
- Layer 2: $z_2 = w_2 \cdot a_1 = 0.8 \cdot 0.622 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$
- Loss (MSE): $L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$

**Backward pass:**
- $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 1.244$
- $\frac{\partial \hat{y}}{\partial z_2} = \sigma'(z_2) = 0.622 \cdot (1 - 0.622) \approx 0.235$
- $\frac{\partial L}{\partial w_2} = 1.244 \cdot 0.235 \cdot a_1 = 1.244 \cdot 0.235 \cdot 0.622 \approx 0.182$
- Continue back through $w_1$ using the same chain

---

# 4. Gradient Flow Through Activation Functions

The activation's derivative directly affects gradient magnitude:

| Activation | Derivative | Gradient behavior |
| :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z)(1-\sigma(z))$, max **0.25** | Multiplies gradient by ≤0.25 per layer |
| **Tanh** | $1 - \tanh^2(z)$, max **1.0** | Still shrinks at saturation |
| **ReLU** | 1 if $z>0$, else 0 | Preserves gradient exactly (or kills it) |
| **GELU** | Smooth approximation ≈1 for large $z$ | Well-behaved across range |

In a 10-layer sigmoid network: gradient multiplied by $\leq 0.25^{10} \approx 10^{-6}$ — effectively zero.

---

# 5. Vanishing vs Exploding Gradients

## Vanishing

Gradients shrink exponentially in early layers → those layers barely learn.

**Causes:** sigmoid/tanh activations, many layers, poor initialization.

**Fixes:**
- ReLU-family activations
- Residual connections (gradient highway around layers)
- BatchNorm / LayerNorm
- Better initialization (Xavier, He)

## Exploding

Gradients grow too large → unstable updates, `NaN` loss.

**Causes:** large weights, no normalization, high learning rate.

**Fixes:**
- Gradient clipping: $g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}$ if $\|g\| > \text{max\_norm}$
- Weight regularization
- Careful initialization

---

# 6. Weight Initialization

Bad initialization causes vanishing/exploding gradients from the first forward pass.

**Xavier / Glorot** (for tanh, sigmoid):

$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**He / Kaiming** (for ReLU):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

The factor 2 accounts for ReLU zeroing half the inputs in expectation.

```python
import torch.nn as nn

# PyTorch applies He init by default to Conv and Linear for ReLU networks
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight)  # for tanh/sigmoid
```

---

# 7. Autograd in Practice

In PyTorch, the forward pass builds a **computation graph**. The backward pass traverses it in reverse.

```python
import torch

# Manual backprop demo
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = w * x + b
a = torch.sigmoid(z)
loss = (a - 0.0) ** 2   # target = 0

loss.backward()

print(f"dL/dw = {w.grad.item():.4f}")
print(f"dL/db = {b.grad.item():.4f}")

# Training loop pattern
optimizer.zero_grad()    # clear previous gradients
loss = criterion(model(X), y)
loss.backward()          # compute gradients
optimizer.step()         # update weights
```

**Common mistakes:**
- Forgetting `zero_grad()` → gradients accumulate across batches
- Detaching tensors unnecessarily → breaks the computation graph
- Computing loss outside `backward()` scope → no gradient flows

---

# 8. Whiteboard Answer Structure

1. Define the forward equations ($z = wx + b$, $a = \sigma(z)$, $L = \text{loss}$)
2. State the loss function
3. Apply chain rule backward step by step
4. Explain what the gradient means physically (sensitivity of loss to weight)
5. State the update rule: $w \leftarrow w - \eta \nabla_w L$

That structure sounds clean and confident.

---

# Quick Thought Experiment

If an early layer never learns:
- suspect vanishing gradients (check activation type)
- check initialization (near-zero init → near-zero gradients early on)
- check learning rate (too small → negligible updates)
- check residual connections (add them if architecture allows)
