# Backpropagation

**Backpropagation** is the algorithm used to compute the **gradient** of the loss with respect to every weight in a neural network. These gradients are then used by an **optimizer** (e.g. SGD, Adam) to update the weights and minimize the loss.

---

## Why gradients?

- Training minimizes a **loss** L (e.g. cross-entropy, MSE). We need ∂L/∂w for each weight w to take a step in the direction that reduces L.
- **Backprop** computes these gradients efficiently in one backward pass by reusing intermediate values from the forward pass (chain rule).

---

## Chain rule

For a composition of functions, the derivative propagates backward:

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}
\]

In a network, the output of one layer is the input to the next. So we compute **local** derivatives (∂layer_output/∂layer_input and ∂layer_output/∂weights) and multiply them from the loss backward to the weights.

---

## Forward and backward pass

1. **Forward pass**: Compute activations layer by layer; store activations (and any values needed for derivatives) for the backward pass.
2. **Backward pass**: Start from ∂L/∂output; for each layer, compute ∂L/∂inputs and ∂L/∂weights using the stored activations and the chain rule; pass ∂L/∂inputs to the previous layer.

---

## In practice

- **Autograd** (PyTorch, JAX, TensorFlow) does this automatically: you define the forward computation; the framework builds the graph and computes gradients on `.backward()` or similar.
- **Gradient checkpointing**: Trade compute for memory by recomputing some activations in the backward pass instead of storing them.

---

## Quick revision

- **Backprop** = backward pass of gradients using the chain rule so every weight gets ∂L/∂w.
- **Forward**: compute and store activations. **Backward**: compute gradients layer by layer from output to input.
- **Autograd** in modern frameworks implements this automatically.
