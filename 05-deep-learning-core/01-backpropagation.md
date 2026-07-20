---
module: Deep Learning Core
topic: Components
subtopic: Backpropagation
status: unread
tags: [deeplearning, ml, components-backpropagation]
---
# Backpropagation

---

## The Credit Assignment Problem

**The problem**: a neural network makes a prediction, you compute a loss, and you want to know how much each of the millions of weights contributed to that loss. Changing weight $w_{ij}$ in layer 3 affects the activations in layer 3, which affects layer 4, which affects layer 5, which affects the loss. To update $w_{ij}$, you need $\partial L / \partial w_{ij}$. Computing this independently for every weight from scratch would be catastrophically expensive.

**The core insight**: the chain rule of calculus lets you reuse intermediate derivatives. If you have already computed how much the loss changes with respect to layer 4's activations, you can use that to compute the gradient for layer 3 cheaply — without starting over. Backpropagation is simply the systematic application of the chain rule, propagating gradients backwards through the computation graph while reusing every intermediate result exactly once.

**The mechanics**: for a composition $L = f(g(h(x)))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

For a single neuron with $z = wx + b$, $a = \sigma(z)$, $L = \text{loss}(a, y)$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\sigma'(z)} \cdot \underbrace{\frac{\partial z}{\partial w}}_{x}$$

The network accumulates $\partial L / \partial a$ from the layer ahead during the backward pass, multiplies by the local derivative $\sigma'(z)$, and multiplies by the input $x$ to get the weight gradient.

**What breaks**: the gradient at each layer is multiplied by the local derivative of the activation. If those derivatives are consistently small (e.g., sigmoid's max derivative is 0.25), gradients shrink exponentially as they travel through layers. In a 10-layer sigmoid network: $0.25^{10} \approx 10^{-6}$. Early layers receive a near-zero gradient and stop learning — the vanishing gradient problem.

---

## Concrete Walkthrough

Single input $x = 1$, target $y = 0$, weights $w_1 = 0.5$, $w_2 = 0.8$, sigmoid activations, MSE loss.

**Forward pass**:
- $z_1 = w_1 x = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$
- $z_2 = w_2 a_1 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$
- $L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$

**Backward pass**:
- $\partial L / \partial \hat{y} = 2(\hat{y} - y) = 1.244$
- $\partial \hat{y} / \partial z_2 = \sigma'(z_2) = 0.622 \cdot 0.378 \approx 0.235$
- $\partial L / \partial w_2 = 1.244 \times 0.235 \times a_1 = 1.244 \times 0.235 \times 0.622 \approx 0.182$
- Continue: $\partial L / \partial a_1 = 1.244 \times 0.235 \times w_2$, then multiply by $\sigma'(z_1)$ and $x$ to get $\partial L / \partial w_1$

The gradient for $w_1$ has already been multiplied by two sigmoid derivatives. This is the vanishing gradient mechanism in action.

---

## Matrix Form of Backpropagation

For a batch of $N$ examples through a linear layer $z = xW + b$, given the upstream gradient $\delta = \partial L / \partial z$:

$$\frac{\partial L}{\partial W} = x^\top \delta \qquad \frac{\partial L}{\partial b} = \mathbf{1}^\top \delta \qquad \frac{\partial L}{\partial x} = \delta W^\top$$

**Why this matters**: every autodiff framework implements exactly these three operations for every linear layer. Backprop through a matmul costs the same as the forward pass — one matrix multiply.

**Activation layers**: for an element-wise activation $a = f(z)$, the vector-Jacobian product is just element-wise multiplication:

$$\delta_\text{prev} = \delta \odot f'(z)$$

---

## Gradient Flow Through Activations

The activation's local derivative is the multiplier at each layer:

| Activation | Derivative | Effect on gradient |
| :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z)(1-\sigma(z))$, max 0.25 | Multiplies gradient by $\leq 0.25$ per layer |
| **Tanh** | $1 - \tanh^2(z)$, max 1.0 | Shrinks at saturation, better than sigmoid |
| **ReLU** | 1 if $z>0$, else 0 | Passes gradient through unchanged, or kills it completely |
| **GELU** | Smooth approximation | Well-behaved; close to 1 for large positive $z$ |

ReLU's gradient being exactly 1 for positive activations is why it enabled training deep networks. The gradient travels from the output all the way to layer 1 without shrinkage — provided neurons are not dead.

---

## Vanishing Gradients

**The problem**: as gradients are multiplied by small local derivatives at each layer, they become exponentially small before reaching early layers. Those layers receive gradient $\approx 0$ and their weights barely update.

**Causes**: sigmoid/tanh activations, many layers, weights initialized too small.

**Fixes**:
- ReLU-family activations: local derivative is 1 for positive activations
- Residual connections: $x_{l+1} = x_l + F(x_l)$ — gradient flows directly through the skip path, bypassing the activation's derivative entirely
- LayerNorm / BatchNorm: prevents activations from drifting into saturation regions
- He initialization: sets weight variance so signal stays at consistent scale at layer initialization

---

## Exploding Gradients

**The problem**: if the local derivatives are $>1$ consistently, gradients grow exponentially. This causes weight updates so large they destabilize training — loss spikes, then diverges to NaN.

**Causes**: large initial weights, no normalization, recurrent networks (where the same weight matrix is multiplied many times over long sequences).

**The mechanics of gradient clipping** (the standard fix):

$$\text{if } \|g\|_2 > \tau: \quad g \leftarrow g \cdot \frac{\tau}{\|g\|_2}$$

Scale the entire gradient vector down so its norm equals $\tau$. Critically, this preserves the *direction* of the gradient — only the magnitude is clipped. Value clipping (clipping each component independently) changes the gradient direction and is generally worse.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Called after loss.backward(), before optimizer.step()
```

---

## Autograd in Practice

PyTorch builds a dynamic computation graph during the forward pass. Each operation records its inputs and its local derivative rule. `loss.backward()` traverses this graph in reverse, calling each node's gradient function and accumulating results.

```python
optimizer.zero_grad()       # clear accumulated gradients from previous step
loss = criterion(model(X), y)
loss.backward()             # compute all gradients via backprop
optimizer.step()            # update weights using computed gradients
```

**What breaks if you do it wrong**:

- Forget `zero_grad()`: gradients accumulate across batches. Updates grow unboundedly. Loss spikes.
- Detach a tensor mid-computation: the computation graph is cut. Gradients do not flow through the detached node. Weights upstream of the detach never update.
- Compute the loss, then modify it outside the graph: the modified loss has no connection to the original graph. `backward()` computes gradients for the wrong thing.

---

## Diagnosing a Layer That Isn't Learning

If an early layer's weights are barely changing:

1. **Check the activation type** — sigmoid/tanh in a 20-layer network will vanish. Switch to ReLU.
2. **Check initialization** — weights initialized near zero produce near-zero activations, which produce near-zero gradients. Use He or Xavier initialization.
3. **Check for dead neurons** — if many ReLU neurons have pre-activation $< 0$ always, their gradients are zero always. Lower the learning rate or use Leaky ReLU.
4. **Check for missing residual connections** — in a very deep network without skip connections, gradients to early layers can vanish even with ReLU. Add skip connections.
5. **Check the learning rate** — too small means numerically nonzero gradients produce negligible weight changes. Too large causes oscillation that prevents convergence.

---

## Interview Angles

### Q: Explain what vanishing gradients mean and why they happen. [Easy]

A: For stacked layers, ∂L/∂x₁ = W_N^T · W_{N-1}^T · ... · W_1^T · ∂L/∂h_N (plus activation derivatives at each step). If the weight matrices/activation derivatives consistently scale the gradient by less than 1, repeated multiplication causes exponential decay — early layers receive a near-zero gradient and stop learning. This was the key obstacle to deep networks before ReLU (doesn't saturate for positive inputs), normalization layers, and residual connections.

**Cross-questions to expect:**
- *"Do exploding gradients have the same cause?"* → Same mechanism, opposite direction: repeated multiplication by factors consistently above 1. Worth saying they're one phenomenon, since the fixes differ — clipping addresses exploding, architecture addresses vanishing.
- *"ReLU has derivative exactly 1 for positive inputs. Does that eliminate vanishing gradients?"* → No. It removes the activation's contribution to shrinkage, but the weight matrices still multiply. A deep plain ReLU net can still vanish if weights are poorly scaled — which is exactly what He initialization exists to prevent.
- *"Why did LSTMs help before ResNets existed?"* → The cell state is an additive path with derivative ≈1, the same trick as a residual connection, applied along time rather than depth. Recognizing them as the same idea is the strong answer.
- *"Can you detect this from training curves alone?"* → Not reliably — a flat loss looks the same for a bad learning rate. You need per-layer gradient norms; the signature is early-layer norms orders of magnitude below late-layer norms.

**Trap:** Saying "sigmoid causes vanishing gradients" and stopping. Sigmoid's max derivative of 0.25 makes it *worse*, but the cause is repeated multiplication of factors below 1 through a deep composition — swapping in ReLU without fixing initialization or adding skip connections does not make a 50-layer network trainable.

---

### Q: How do residual connections solve the vanishing gradient problem? [Medium]

A: ResNet defines each block as F(x) = x + G(x) instead of F(x) = G(x). The gradient at the input is ∂L/∂x = ∂L/∂F · (1 + ∂G/∂x). The "+1" term gives gradients a direct path back through the identity shortcut, regardless of how small ∂G/∂x becomes — even if G contributes zero gradient, signal still flows through unchanged. This is why 152-layer ResNets train from scratch while equally deep plain networks don't. Transformers use the same trick: every attention and FFN block is wrapped in a residual connection.

**Cross-questions to expect:**
- *"If the identity path solves it, why not make every network 1000 layers?"* → Depth still costs compute and memory, and returns diminish — ResNet-1001 barely beats ResNet-152. Skip connections make depth *trainable*, not automatically *better*.
- *"Pre-activation or post-activation residual blocks?"* → He et al. (2016) showed pre-activation (BN → ReLU → conv inside the branch) gives a completely unobstructed identity path, since nothing operates on the skip itself. It trains deeper nets more stably and is the modern default.
- *"What does the network actually learn if the identity is already good?"* → $G(x) \approx 0$. That's the point of the name: it learns the *residual* from identity, and learning a near-zero function is far easier than learning a near-identity one through a stack of nonlinearities.
- *"Do residual connections interact with normalization placement?"* → Yes, directly. Post-norm applies LN to the sum, so the skip path passes through a normalizer and the clean gradient route is degraded; pre-norm keeps the identity clean. That's why deep transformers are pre-norm — see [07-normalization.md](07-normalization.md).

**Trap:** Describing the skip as "adding the input back so information isn't lost." That's the forward-pass intuition and it's incomplete. The load-bearing fact is the derivative: the $(1 + \partial G/\partial x)$ term guarantees a gradient path that survives no matter how small $\partial G/\partial x$ gets.

---

### Q: What is gradient checkpointing and when would you use it? [Medium]

A: Standard backprop caches every intermediate activation from the forward pass for reuse in the backward pass — memory scales with depth × batch size × sequence length. Gradient checkpointing trades compute for memory: only activations at checkpoint boundaries are saved, and the rest are recomputed during the backward pass. This cuts activation memory substantially at the cost of roughly 30% more compute. Use it when fine-tuning large models that don't fit in GPU memory, or training with long sequences/large batches where activation memory dominates.

**Cross-questions to expect:**
- *"Why roughly 30% and not 100% more compute?"* → Only the forward pass is recomputed, and only between checkpoints. Backward is roughly twice the cost of forward, so the full step is ~3 units; adding one extra forward makes ~4. Hence ~33%, not double.
- *"Where do you place the checkpoints?"* → With $\sqrt{n}$ evenly spaced segments over $n$ layers, memory drops to $O(\sqrt{n})$ — the standard result. In practice you checkpoint at transformer block boundaries, which is what `torch.utils.checkpoint` wrappers do by default.
- *"It broke my model's dropout — why?"* → The recomputed forward must reproduce the original one exactly. Any RNG-dependent op (dropout, stochastic depth) will resample unless the RNG state is saved and restored. PyTorch handles this with `preserve_rng_state=True`; getting it wrong makes backward compute gradients for a *different* network than the one that produced the loss.
- *"How does it compare to just reducing batch size?"* → Both cut activation memory, but small batches hurt throughput and can destabilize BatchNorm statistics. Checkpointing keeps the batch size and pays in compute instead — usually the better trade on modern accelerators.

**Trap:** Calling it a memory optimization with no downsides. It is a compute-for-memory trade, and it interacts badly with anything stochastic in the forward pass. If asked how to fit a larger model, checkpointing is one lever among several (mixed precision, ZeRO sharding, offload) — naming only this one suggests a narrow view of the problem.
