# Normalization Techniques in Deep Learning

Normalization layers stabilize training by controlling the distribution of activations. Different variants suit different architectures and batch sizes.

---

## Why Normalize?

Without normalization:
- Internal covariate shift: distribution of layer inputs shifts as weights update
- Deep networks saturate activations (tanh/sigmoid → zero gradients)
- Training requires careful learning rate tuning and initialization

---

## Batch Normalization (BatchNorm)

Introduced by Ioffe & Szegedy (2015). Normalizes each feature across the **batch** dimension.

For a mini-batch of activations `{x_1, ..., x_m}` for a single feature:

```
μ_B = (1/m) Σ x_i
σ²_B = (1/m) Σ (x_i - μ_B)²
x̂_i = (x_i - μ_B) / √(σ²_B + ε)
y_i = γ x̂_i + β    ← learned scale and shift (γ and β are trainable)
```

```python
# 2D (after Linear layer)
bn = nn.BatchNorm1d(num_features)

# 4D (after Conv2d): normalizes over (N, H, W) for each channel C
bn = nn.BatchNorm2d(num_channels)

# Training vs Inference behavior difference
model.train()   # uses batch statistics for μ, σ²
model.eval()    # uses running mean/var accumulated during training
```

**Key properties:**
- Running statistics (exponential moving average of μ and σ²) used at inference
- `momentum` parameter controls how fast running stats update (default 0.1 in PyTorch)
- Acts as regularizer — slightly reduces need for Dropout

**Problems:**
- Fails with small batch sizes (statistics noisy)
- Cannot be used with sequence batches of variable length (per-element statistics meaningless)
- Non-causal in time series (future statistics leak into past)

---

## Layer Normalization (LayerNorm)

Normalizes across **all features** for each single sample. Batch-independent.

```
For sample x of dimension d:
μ = (1/d) Σ x_j
σ² = (1/d) Σ (x_j - μ)²
x̂ = (x - μ) / √(σ² + ε)
y = γ ⊙ x̂ + β    ← per-feature scale and shift
```

```python
ln = nn.LayerNorm(normalized_shape)   # e.g., embed_dim for Transformers
# normalized_shape = [d] → normalizes last d dimensions
# normalized_shape = [H, W] → normalizes spatial dims for each (N, C) independently
```

**Used in:** Transformers (BERT, GPT, T5), RNNs, ViTs.  
**Works for any batch size, including 1.** Consistent train/inference behavior (no running stats).

---

## Instance Normalization (InstanceNorm)

Normalizes each sample and each channel independently over **spatial dimensions** (H, W). No batch statistics involved.

```python
# 2D: normalizes over (H, W) for each (N, C) independently
inst = nn.InstanceNorm2d(num_channels, affine=True)
```

**Used in:** Style transfer (AdaIN — Adaptive Instance Normalization). Removes style while preserving content structure.

**Limitation:** Ignores batch-wide statistics and cross-channel relationships.

---

## Group Normalization (GroupNorm)

Divides channels into G groups; normalizes over (C/G, H, W) for each sample. Batch-independent.

```python
# G=1 → equivalent to LayerNorm over spatial dims
# G=C → equivalent to InstanceNorm
gn = nn.GroupNorm(num_groups=32, num_channels=256)
```

**Used in:** Object detection (FPN, Mask R-CNN), segmentation, video models where batch size is small (1-4 per GPU).

**Rule of thumb:** 32 channels per group typical. Outperforms BatchNorm when batch size < 16.

---

## Root Mean Square Normalization (RMSNorm)

Simplified LayerNorm: omits mean centering, only scales by RMS.

```
RMS(x) = √((1/d) Σ x_j²)
y = (x / RMS(x)) × γ
```

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale
```

**Used in:** LLaMA, Mistral, Gemma — replacing LayerNorm in modern LLMs. Faster (no mean computation), nearly identical performance.

---

## Adaptive Instance Normalization (AdaIN)

Normalizes content feature statistics to match style feature statistics.

```
AdaIN(x, y) = σ(y) × ((x - μ(x)) / σ(x)) + μ(y)
```

where `y` is the style feature map. Core of fast neural style transfer (Huang & Belongie, 2017).

---

## Weight Normalization

Reparameterize weights: `w = g / ‖v‖ × v`. Separates magnitude `g` from direction `v`.

```python
layer = nn.Linear(...)
torch.nn.utils.weight_norm(layer, name='weight')
```

**Advantage:** No batch statistics; works at step 1 of training. Used in flow models, WaveNet.

---

## Spectral Normalization

Constrains the Lipschitz constant of each layer by dividing weights by their spectral norm (largest singular value).

```python
layer = nn.utils.spectral_norm(nn.Linear(in_features, out_features))
```

**Used in:** GAN discriminators to stabilize training (prevents discriminator from becoming too powerful).

---

## Pre-Norm vs Post-Norm

Refers to where normalization sits relative to the residual connection in Transformers.

**Post-Norm (original Transformer):**
```
x = LayerNorm(x + Sublayer(x))
```
Harder to train deep networks; gradient signal can vanish.

**Pre-Norm (modern standard — GPT-2+, LLaMA):**
```
x = x + Sublayer(LayerNorm(x))
```
More stable gradients; easier to train very deep networks; preferred in practice.

---

## Summary Comparison

| Method | Normalizes over | Batch-independent | Use case |
|--------|----------------|------------------|---------|
| BatchNorm | Batch + spatial | No | CNN training, large batches |
| LayerNorm | Features per sample | Yes | Transformers, NLP, RNNs |
| InstanceNorm | Spatial per sample+channel | Yes | Style transfer |
| GroupNorm | Groups of channels | Yes | Small-batch training (detection) |
| RMSNorm | Features per sample (no mean) | Yes | Modern LLMs (LLaMA, Gemma) |
| WeightNorm | Weight vectors | Yes | Generative models, flow models |
| SpectralNorm | Weight matrices | Yes | GAN discriminators |

---

## Key Interview Points

- BatchNorm has different train/eval behavior — `.eval()` switches to running statistics. Forgetting this is a common bug.
- LayerNorm is batch-independent: works with batch size 1, variable-length sequences, autoregressive generation.
- GroupNorm outperforms BatchNorm for small batches (<16); common in detection/segmentation where images don't fit in large batches.
- RMSNorm = LayerNorm without mean centering — faster and used in all modern LLMs.
- Pre-Norm (normalize before sublayer) stabilizes deep Transformer training; post-Norm (original) is harder to train.
- AdaIN = normalize by content statistics, re-scale/shift by style statistics — key to fast style transfer.
