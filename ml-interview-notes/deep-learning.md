# Deep Learning Fundamentals & Architectures

These notes follow the **Gold Standard** for interview preparation: providing direct answers, geometric intuition, and practical implementation details for Deep Learning concepts.

---

# 1. Neural Network Foundations

## Q1: What are Neural Networks and how do they learn?

### 🔹 Direct Answer
Neural Networks are parameterized function approximators composed of layers of neurons that apply linear transformations followed by non-linear activations. They learn by adjusting their weights via **Gradient Descent** and **Backpropagation** to minimize a predefined loss function.

### 🔹 Intuition
Imagine a complex machine with millions of knobs. To get the machine to play a game, you don't write rules; you just turn the knobs slightly every time it makes a mistake. Over time, the knobs settle into a configuration that "understands" the game. Representation learning is the network's ability to automatically find the "best knobs" (features) instead of us designing them.

### 🔹 Code Snippet (Basic PyTorch Neuron)
```python
import torch.nn as nn
linear_layer = nn.Linear(in_features=10, out_features=5)
# output = input @ weight.T + bias
```

---

# 2. Activation Functions

## Q2: Why are activation functions needed, and how do they differ?

### 🔹 Direct Answer
Activation functions introduce **non-linearity**. Without them, stacking 100 linear layers would still collapse into a single linear transformation ($W_2(W_1X) = (W_2W_1)X = W'X$). They allow the network to approximate any complex function.

### 🔹 Comparison Table

| Activation | Range | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | (0, 1) | Good for probability. | Vanishing Gradients, Not zero-centered. |
| **Tanh** | (-1, 1) | Zero-centered, better gradients. | Still suffers from vanishing gradients. |
| **ReLU** | [0, ∞) | Fast, sparse, prevents vanishing grads. | "Dying ReLU" (neurons getting stuck at 0). |
| **Leaky ReLU** | (-∞, ∞) | Fixes Dying ReLU. | Small gradient for negative values. |
| **Softmax** | (0, 1) | Prob. distribution (sum=1). | Comp. intensive for large vocabularies. |

### 🔹 Deep Dive: The Vanishing Gradient Problem
When using Sigmoid/Tanh, as the input becomes very large or small, the derivative becomes near zero. During backprop, we multiply these tiny gradients through many layers, causing the signal to "vanish" before it reaches the early layers. ReLU solves this because its gradient is exactly 1 for all positive values.

---

# 3. Regularization & Training Dynamics

## Q3: Explain Dropout and Batch Normalization.

### 🔹 Direct Answer
- **Dropout:** A regularization technique that randomly "shuts off" neurons during training to prevent co-adaptation and overfitting.
- **Batch Normalization (BatchNorm):** Normalizes the activations of a layer to have zero mean and unit variance. It accelerates training and provides mild regularization.

### 🔹 Intuition
- **Dropout:** Like a sports team where any player might get injured at any time. Every player must learn to play with everyone else, making the whole team more robust.
- **BatchNorm:** Like adjusting the volume on a microphone. It ensures the signal coming into the next layer is always within a "comfortable" range, preventing the model from dealing with wildly different scales.

### 🔹 Code Snippet (PyTorch)
```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.5)
)
```

---

# 4. Advanced Architectures (CNNs & RNNs)

## Q4: Why are CNNs superior for images compared to MLPs?

### 🔹 Direct Answer
CNNs use **Inductive Biases** that match the nature of images:
1. **Locality:** Pixels near each other are more related.
2. **Translation Invariance:** A cat is still a cat whether it's in the top-left or bottom-right corner.
3. **Parameter Sharing:** The same edge detector filter can be used across the whole image, reducing the parameter count significantly.

## Q5: LSTM vs. GRU vs. Transformer

### 🔹 Direct Answer
- **LSTMs:** Use three gates (input, forget, output) to maintain long-term memory. Good for sequences but slow (sequential).
- **GRUs:** Simplified LSTMs with two gates. Faster but slightly less expressive.
- **Transformers:** Use **Self-Attention** to process the whole sequence in parallel. They dominate today because they scale better with data and compute.

---

# 5. Generative Modeling

## Q6: Diffusion vs. GANs vs. VAEs

### 🔹 Direct Answer
- **VAEs:** Learn a latent distribution; often produce blurry outputs.
- **GANs:** Adversarial setup (Generator vs. Discriminator); produce sharp outputs but are unstable to train (Mode Collapse).
- **Diffusion:** Iteratively denoises a random signal; current state-of-the-art for high-quality, diverse image generation.

### 🔹 High-Yield Comparison Table

| Model | Learning Type | Main Advantage | Main Weakness |
| :--- | :--- | :--- | :--- |
| **VAE** | Explicit Density | Stable training | Blurry results |
| **GAN** | Implicit Density | Sharpest results | Training instability |
| **Diffusion** | Score-based | Best quality/diversity | Slow inference (iterative) |
