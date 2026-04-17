# Deep Learning Foundations

Welcome to the Deep Learning Foundations library. This track focuses on the fundamental neural components, architectural breakthroughs, and optimization strategies that power modern AI.

---

# 1. 🔹 Neural Network Building Blocks

A deep neural network is built from simple mathematical primitives arranged in complex hierarchies.

### 🔹 Layer Families
- **Fully Connected (Dense):** Learns global patterns.
- **Convolutional (CNN):** Learns spatial hierarchies (edges -> textures -> shapes).
- **Recurrent (RNN/LSTM):** Learns sequential dependencies (Time-series, Text).
- **Attention:** Learns contextual relationships across arbitrary distances.

---

# 2. 🔹 Training & Optimization

## Q1: Why is Gradient Descent not enough for deep networks?

### 🔹 Direct Answer
Vanilla SGD often gets stuck in plateaus or oscillating around ravines. Modern optimizers like **Adam** and **AdamW** combine **Momentum** (to accelerate through flat regions) and **Adaptive Learning Rates** (RMSProp) to handle varying gradients across different parameters.

### 🔹 Core components
- **[Activation Functions](./parts-of-deep-learning/activation-functions.md):** Introducing non-linearity (ReLU, GeLU).
- **[Loss Functions](./parts-of-deep-learning/loss-functions.md):** Defining the objective (Cross-Entropy, MSE).
- **[Backpropagation](./parts-of-deep-learning/backpropagation.md):** The math of signal transmission.

---

# 3. 🔹 Modern Architectures

## Q2: RNNs vs. Transformers.

### 🔹 Direct Answer
RNNs are sequential, meaning they process one token at a time, making them slow and prone to forgetting long-range context. **Transformers** use **Self-Attention** to process all tokens in parallel, making them highly scalable and superior at capturing global context.

---

# 4. 🔹 Transfer Learning & Fine-Tuning

## Q3: When should you freeze the base model?

### 🔹 Direct Answer
1. **Low Data:** Freeze the base and only train the new classification head to prevent overfitting.
2. **Abundant Data:** Fine-tune the entire model (differential learning rates) for maximum adaptation.

---

> [!TIP]
> **Learning Tip:** For high-speed interview prep, visit the [Deep Learning Essentials Hub](../ml-interview-notes/deep-learning.md). For Pytorch-specific implementation, see the [Pytorch Core Library](../pytorch/README.md).
