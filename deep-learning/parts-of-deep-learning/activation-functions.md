# Activation Functions (Deep-Dive)

Activation functions introduce non-linearity into a neural network, allowing it to learn complex, non-linear decision boundaries.

---

# 1. 🔹 The ReLU Family

## Q1: Why did ReLU replace Sigmoid/Tanh in deep networks?

### 🔹 Direct Answer
Sigmoid and Tanh suffer from the **Vanishing Gradient Problem**. For large positive or negative inputs, their derivative is nearly zero, which prevents weights from updated in deep layers. **ReLU (Rectified Linear Unit)** has a constant derivative of 1 for all positive values, ensuring strong signal transmission.

### 🔹 Comparison Table

| Function | Formula | Derivative Range | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $1/(1+e^{-x})$ | [0, 0.25] | Probabilistic output. | Vanishing gradients, non-zero centered. |
| **ReLU** | $\max(0, x)$ | {0, 1} | Fast, no vanishing gradient. | "Dying ReLU" (neurons get stuck at 0). |
| **Leaky ReLU** | $\max(\alpha x, x)$ | {$\alpha$, 1} | Prevents dying neurons. | $\alpha$ is a hyperparameter to tune. |
| **GeLU** | $x \cdot \Phi(x)$ | Continuous | Self-regularizing. | Standard for Transformers (BERT/GPT). |

---

# 2. 🔹 Softmax & Output Layers

## Q2: Softmax vs. Sigmoid for Classification.

### 🔹 Direct Answer
- **Sigmoid:** Used for **Binary Classification** or **Multi-Label Classification** (where multiple classes can be true simultaneously). Each output node is independent.
- **Softmax:** Used for **Multi-Class Classification** (where only one class is true). It ensures that the sum of all output probabilities is exactly 1.

---

# 3. 🔹 Selecting the Right Function

- **Hidden Layers:** Use **ReLU** by default. Use **GELU** for Transformers.
- **Regression:** Use **Identity** (No activation) at the output.
- **Binary Classification:** Use **Sigmoid** at the output.
- **Multi-Class:** Use **Softmax** at the output.

---

> [!TIP]
> **Implementation Note:** In PyTorch, GELU is becoming the standard for virtually all modern transformer-based research. Use `nn.GELU()` in your layer definitions.
