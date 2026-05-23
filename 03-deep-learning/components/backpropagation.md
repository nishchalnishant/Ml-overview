---
module: Deep Learning
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

## Canonical Interview Q&As

**Q: Derive the backprop update for a single linear layer and explain what vanishing gradients mean geometrically.**  
A: For layer output h = Wx + b, loss L, the gradient is ∂L/∂W = (∂L/∂h) · xᵀ and ∂L/∂x = Wᵀ · (∂L/∂h). When stacking N layers, ∂L/∂x_1 = W_N^T · W_{N-1}^T · ... · W_1^T · ∂L/∂h_N. Each matrix multiplication can scale the gradient by the dominant singular value of each W. If all singular values < 1, repeated multiplication causes exponential decay — gradient signal at layer 1 is O(σ^N) of the output gradient. Geometrically: the gradient vector collapses toward zero in the directions that the weight matrices shrink. The early layers receive nearly zero gradient, so their weights don't update — the model effectively becomes shallow. This was the key obstacle to deep networks before ReLUs (ReLU doesn't saturate in the positive region), BatchNorm (normalizes activations), and residual connections (bypass the chain of multiplications).

**Q: How do residual connections solve the vanishing gradient problem?**  
A: ResNet defines each block as F(x) = x + G(x) instead of F(x) = G(x). The gradient at the input is: ∂L/∂x = ∂L/∂F · (1 + ∂G/∂x). The "+1" term means gradients always have a direct path back through the identity shortcut, regardless of how small ∂G/∂x becomes. Even if the learned transformation G contributes zero gradient, the gradient flows through unchanged. For a network of N residual blocks, the gradient at the first layer is a sum of 2^N terms (all possible paths through/around each block), rather than a single product of N matrices — exponential growth in paths rather than exponential decay in magnitude. Practically: this is why 152-layer ResNets can be trained from scratch but 152-layer plain networks cannot. In transformers, the same principle applies — each attention and FFN block is wrapped in a residual connection.

**Q: What is gradient checkpointing and when would you use it?**  
A: In standard backprop, all intermediate activations from the forward pass are cached for use in the backward pass — memory scales with model depth × batch size × sequence length. Gradient checkpointing trades compute for memory: only activations at "checkpoint" boundaries are saved; intermediate activations between checkpoints are recomputed during backward pass. For a model with N layers and checkpoints every k layers, memory reduces from O(N) to O(√N) activations (with optimal checkpoint spacing). The tradeoff is ~33% more FLOPs (each layer forward-pass runs ~1.33× on average). Use it when: (1) fine-tuning large models that don't fit in GPU memory, (2) training with very long sequences (activation memory grows with sequence length), (3) training with large batch sizes. For a 70B model fine-tuned on 4K sequences, activations alone can be 50–100GB without checkpointing; with checkpointing, this drops to ~5–10GB.

## Flashcards

**$z_1 = w_1 x = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$?** #flashcard
$z_1 = w_1 x = 0.5$, $a_1 = \sigma(0.5) \approx 0.622$

**$z_2 = w_2 a_1 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$?** #flashcard
$z_2 = w_2 a_1 = 0.498$, $\hat{y} = \sigma(0.498) \approx 0.622$

**$L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$?** #flashcard
$L = (\hat{y} - y)^2 = 0.622^2 \approx 0.387$

**$\partial L / \partial \hat{y} = 2(\hat{y} - y) = 1.244$?** #flashcard
$\partial L / \partial \hat{y} = 2(\hat{y} - y) = 1.244$

**$\partial \hat{y} / \partial z_2 = \sigma'(z_2) = 0.622 \cdot 0.378 \approx 0.235$?** #flashcard
$\partial \hat{y} / \partial z_2 = \sigma'(z_2) = 0.622 \cdot 0.378 \approx 0.235$

**$\partial L / \partial w_2 = 1.244 \times 0.235 \times a_1 = 1.244 \times 0.235 \times 0.622 \approx 0.182$?** #flashcard
$\partial L / \partial w_2 = 1.244 \times 0.235 \times a_1 = 1.244 \times 0.235 \times 0.622 \approx 0.182$

**Continue?** #flashcard
$\partial L / \partial a_1 = 1.244 \times 0.235 \times w_2$, then multiply by $\sigma'(z_1)$ and $x$ to get $\partial L / \partial w_1$

**ReLU-family activations?** #flashcard
local derivative is 1 for positive activations

**Residual connections: $x_{l+1} = x_l + F(x_l)$?** #flashcard
gradient flows directly through the skip path, bypassing the activation's derivative entirely

**LayerNorm / BatchNorm?** #flashcard
prevents activations from drifting into saturation regions

**He initialization?** #flashcard
sets weight variance so signal stays at consistent scale at layer initialization

**Forget zero_grad()?** #flashcard
gradients accumulate across batches. Updates grow unboundedly. Loss spikes.

**Detach a tensor mid-computation?** #flashcard
the computation graph is cut. Gradients do not flow through the detached node. Weights upstream of the detach never update.

**Compute the loss, then modify it outside the graph?** #flashcard
the modified loss has no connection to the original graph. backward() computes gradients for the wrong thing.
