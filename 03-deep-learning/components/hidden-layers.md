# Hidden Layers

Hidden layers are where a network stops being a simple input-output calculator and starts building internal representations.

That is the entire point of depth.

---

# 1. What Hidden Layers Do

Each hidden layer transforms the representation it receives.

Early layers often learn:
- simple patterns (edges, frequencies, n-gram statistics)

Deeper layers learn:
- more abstract structure (shapes, syntax, semantics)

That is why deep learning often feels like stacked feature engineering, except the model is doing the engineering for you.

**Universal Approximation Theorem:** a feedforward network with at least one hidden layer and a non-linear activation can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision — given enough width.

The theorem guarantees existence, not learnability. In practice, depth beats raw width.

---

# 2. Width vs Depth

**Width** (neurons per layer): increases the number of functions the layer can represent simultaneously.

**Depth** (number of layers): enables hierarchical composition — complex functions built from simpler ones.

**Practical tradeoffs:**

| Property | Wider | Deeper |
| :--- | :--- | :--- |
| **Expressiveness** | Linear growth | Exponential growth |
| **Optimization** | Easier | Harder (vanishing gradients) |
| **Overfitting risk** | Lower | Higher |
| **Compute per forward pass** | $O(\text{width}^2)$ | $O(\text{depth})$ |
| **Inductive bias** | Less hierarchy | More hierarchy |

**Rule of thumb:** deeper is usually better up to the point where optimization becomes the bottleneck. Use skip connections (ResNet) to push depth further.

---

# 3. Types of Layers

| Layer type | Operation | Common use |
| :--- | :--- | :--- |
| **Dense / Linear** | $y = xW + b$ | Feedforward nets, classifier heads |
| **Convolutional** | Local filter convolutions | Images, sequences (1D conv) |
| **Recurrent (LSTM/GRU)** | Stateful sequential processing | Time series, sequences (pre-Transformer) |
| **Attention** | Dynamic weighted aggregation | Transformers, cross-modal |
| **Embedding** | Integer index → dense vector | Text tokens, categorical IDs |
| **Normalization** | BatchNorm / LayerNorm / GroupNorm | Stabilize training |
| **Pooling** | Spatial aggregation (max/avg) | CNNs, reduce spatial dims |

---

# 4. Skip Connections (Residual Connections)

**Problem:** very deep networks suffer from degradation — training accuracy plateaus even without overfitting.

**ResNet solution:** add the input directly to the output of a block:

$$H(x) = F(x) + x$$

where $F(x)$ is the residual to learn (much easier than learning $H(x)$ from scratch).

**Why it helps:**
- Gradient highway: gradients flow directly through the skip connection, bypassing potentially vanishing paths
- Identity shortcut: if $F(x) \approx 0$, the block approximates identity — easier to learn than zero-mapping
- Enables networks of hundreds of layers

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        return x + self.net(x)   # residual connection
```

---

# 5. Normalization Layers

Normalization stabilizes training by controlling activation distributions.

| Layer | Normalizes over | Best for |
| :--- | :--- | :--- |
| **BatchNorm** | Batch dimension (per feature) | CNNs, large batch training |
| **LayerNorm** | Feature dimension (per sample) | Transformers, variable-length sequences |
| **GroupNorm** | Groups of channels (per sample) | Small batch CNNs |
| **RMSNorm** | RMS of features (no mean shift) | LLaMA, efficient LayerNorm variant |

**RMSNorm** (used in LLaMA, Mistral):

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

Faster than LayerNorm (no mean computation), empirically similar quality.

---

# 6. Practical Sizing Guidelines

**Width:** start with powers of 2 (128, 256, 512, 1024). Wider helps up to a point; diminishing returns set in.

**Depth:** for tabular data, 2–4 layers is usually enough. For vision/NLP, depth is more critical.

**Bottleneck architecture:** compress to smaller dimension then expand. Common in autoencoders and residual nets (1×1 conv in ResNet).

**Output layer:** always size to match the task — $K$ outputs for $K$-class softmax, 1 output for binary/regression.

**General heuristics:**
- Start with a simple 2-3 layer MLP baseline
- Add depth before width when underfitting
- Add regularization when overfitting before reducing capacity
- Check that depth increase is paired with skip connections

---

# 7. Why More Layers Help (and When They Hurt)

Depth lets the model build complex functions out of simpler ones — it can capture interactions, hierarchy, and abstraction.

But more layers also mean:
- harder optimization (vanishing/exploding gradients)
- more overfitting risk
- higher compute and memory

**Fixes for deep network problems:**
- Vanishing gradients → skip connections, LayerNorm, ReLU activations, He initialization
- Overfitting → Dropout, weight decay, data augmentation, early stopping
- Compute → model compression, mixed precision, efficient architectures

Deeper is not automatically better. It is more powerful when paired with the right architecture choices.
