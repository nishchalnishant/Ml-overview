---
module: Deep Learning Core
topic: Components
subtopic: Hidden Layers
status: unread
tags: [deeplearning, ml, components-hidden-layers]
---
# Hidden Layers

---

## Why Depth Exists

**The problem**: a single linear layer can only draw straight-line decision boundaries. Adding more linear layers does not help — $W_2(W_1 x) = (W_2 W_1)x$ collapses into a single linear transformation. For any real problem (images, language, audio), the input-output relationship is highly non-linear and hierarchically structured. A flat linear model cannot represent it.

**The core insight**: pair each layer with a non-linear activation. Now stacked layers can compose non-linear functions. With enough layers and non-linearity, you can build arbitrarily complex functions — and more importantly, you can build them *hierarchically*, where each layer builds on abstractions learned by the layer below.

**The mechanics**: for a layer $l$ with weights $W_l$, biases $b_l$, and activation $\sigma$:

$$h_l = \sigma(W_l h_{l-1} + b_l)$$

Early layers learn simple features (edges, frequencies, character n-grams). Middle layers combine these into patterns (shapes, phonemes, words). Late layers assemble patterns into task-relevant abstractions (objects, syntax, semantics).

**What breaks**: depth without proper architecture support causes vanishing gradients, optimization difficulty, and training instability. Adding layers without adding residual connections is often counterproductive beyond ~10 layers.

**Universal Approximation Theorem**: a single hidden layer with a non-linear activation and sufficient width can approximate any continuous function on a compact domain to arbitrary precision. This establishes that width alone is *sufficient* in theory — but depth is far more *efficient* in practice.

---

## Width vs Depth

**The problem**: given a fixed parameter budget, should you put it into more neurons per layer (wider) or more layers (deeper)?

**The core insight**: depth enables hierarchical composition — complex functions can be built as compositions of simpler ones. A deep narrow network can represent functions that would require exponentially wider shallow networks.

**What breaks** with excessive depth: gradients must travel through more layers. Without residual connections, they vanish. Optimization becomes harder as the loss surface develops more saddle points and flat regions.

**What breaks** with excessive width: diminishing returns set in quickly. Each additional neuron in a layer captures increasingly redundant representations. The parameter count grows as $O(\text{width}^2)$ but expressiveness grows linearly.

| Property | Wider | Deeper |
| :--- | :--- | :--- |
| **Expressiveness** | Linear growth | Exponential growth |
| **Optimization** | Easier | Harder (without skip connections) |
| **Overfitting risk** | Lower (per-param) | Higher |
| **Inductive bias** | Less hierarchy | More hierarchy |

Rule of thumb: add depth before width when underfitting. Pair any depth increase beyond ~5 layers with residual connections.

---

## Layer Types

| Layer | Operation | What it's for |
| :--- | :--- | :--- |
| **Dense / Linear** | $y = xW + b$ | Feedforward nets, classifier heads |
| **Convolutional** | Local filter applied with weight sharing | Images, 1D sequences |
| **Recurrent (LSTM/GRU)** | Stateful sequential computation | Time series (pre-Transformer) |
| **Attention** | Dynamic weighted aggregation over positions | Transformers, cross-modal |
| **Embedding** | Integer index → dense vector | Tokens, categorical IDs |
| **Normalization** | Stabilize activation distributions | Every deep network |
| **Pooling** | Spatial aggregation (max/avg) | Reduce spatial dimensions in CNNs |

---

## Residual Connections

**The problem**: adding more layers should make a network at least as good as a shallower one — the deeper network can just learn identity mappings in the extra layers. In practice, very deep networks perform *worse* than shallower ones on training data. They are harder to optimize — the identity mapping is difficult to learn from scratch with random initialization.

**The core insight**: make the identity mapping trivially easy by adding a skip connection. The layer only needs to learn the *residual* — the deviation from identity — which can be zero (and near-zero is easy to learn from random small initialization).

**The mechanics**:

$$H(x) = F(x) + x$$

The layer learns $F(x) = H(x) - x$ (the residual). If the optimal transformation is near-identity, $F(x) \approx 0$ and learning converges quickly. If the layer should transform significantly, $F(x)$ captures that.

Gradient benefit: the gradient of $H$ with respect to $x$ is $\nabla F + 1$. The "+1" term provides a direct gradient highway — even if $\nabla F$ vanishes, at least gradient magnitude 1 flows through the skip connection.

```python
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

**What breaks**: if the skip connection adds $x$ but the layer outputs have very different scale, the sum is dominated by one term. This is managed by normalization layers and careful initialization (see weight initialization). Also: residual connections require matching dimensions across the skip — if dimensions change, you need a projection.

---

## Normalization Layers

**The problem**: as weights update during training, the distribution of each layer's activations shifts — the mean and variance change. Deeper layers must constantly adapt to shifting input distributions from earlier layers, which slows learning and destabilizes training. This is internal covariate shift.

**The core insight**: after each layer's transformation, explicitly normalize the activations to have controlled statistics. This removes the distributional shift and allows each layer to focus on learning its own function, not compensating for upstream drift.

| Layer | Normalizes over | Best for |
| :--- | :--- | :--- |
| **BatchNorm** | Batch dimension (per feature) | CNNs, large batches |
| **LayerNorm** | Feature dimension (per sample) | Transformers, variable-length sequences |
| **GroupNorm** | Groups of channels (per sample) | Small-batch detection/segmentation |
| **RMSNorm** | RMS of features, no mean shift | Modern LLMs (LLaMA, Gemma) |

**RMSNorm**: simplification of LayerNorm — removes the mean subtraction. Empirically similar quality, faster:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

**What breaks** with BatchNorm: requires a large enough batch for stable statistics. Batch size of 1 or 2 makes the mean/variance estimate noisy and the normalization meaningless. Variable-length sequences are problematic because the same position across a batch may contain padding.

---

## Practical Sizing

**Width**: start with powers of 2 (256, 512, 1024). There is no magic number — scale with dataset size and task complexity. For tabular data, 128–512 is usually sufficient. For vision/NLP, 512–4096.

**Depth**: for tabular tasks, 2–4 layers. For CNNs on images, 10–100+ layers with residual connections (ResNet-50 is 50 layers). For Transformers, 12–96+ blocks.

**Bottleneck architecture**: compress to a smaller dimension, then expand. Used in:
- Autoencoders (compression by design)
- ResNet bottleneck blocks (1×1 convolutions to reduce channels, then 3×3, then expand)
- Transformer FFN (2 linear layers with a 4× width expansion in the middle)

**Output layer**: always match to task — $K$ outputs for $K$-class softmax, 1 output for binary/regression. Never apply activation before the loss function expects raw logits (e.g., `CrossEntropyLoss` in PyTorch applies softmax internally).
