# Activation Functions

**Activation functions** introduce **non-linearity** into neural networks. Without them, stacking linear layers would be equivalent to a single linear layer and could not learn complex mappings.

---

## Common activations

| Function | Formula | Use |
|----------|---------|-----|
| **ReLU** | max(0, x) | Default in many feed-forward and CNN layers |
| **GELU** | x Φ(x) (Gaussian CDF) | Often in transformers |
| **Sigmoid** | 1 / (1 + e^{-x}) | Output for binary probability (0–1) |
| **Tanh** | (e^x - e^{-x}) / (e^x + e^{-x}) | Bounded output (-1, 1) |
| **Softmax** | e^{x_i} / Σ_j e^{x_j} | Output for multi-class probabilities |

---

## Properties

- **ReLU**: Sparse (zeros for x ≤ 0); avoids vanishing gradient for x > 0; "dying ReLU" if many units stay at 0.
- **GELU**: Smooth; used in BERT, GPT. **Sigmoid / Tanh**: Can suffer from vanishing gradients at saturation.
- **Softmax**: Normalizes a vector to a probability distribution; used at the output for classification.

---

## Where they appear

- **Hidden layers**: ReLU or GELU after linear layers.
- **Output**: Sigmoid (binary), softmax (multi-class), or linear (regression).
- **Attention**: Softmax over scores to get attention weights.

---

## Quick revision

- **Activations** add non-linearity; ReLU and GELU are standard in hidden layers.
- **Sigmoid** and **softmax** for probabilities; **softmax** for multi-class output.
- Choice affects gradient flow and training stability.
