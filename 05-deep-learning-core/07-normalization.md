---
module: Deep Learning Core
topic: Components
subtopic: Normalization
status: unread
tags: [deeplearning, ml, components-normalization]
---
# Normalization

---

## Internal Covariate Shift

**The problem**: during training, a layer's weights update based on the distribution of its inputs. But the layer below is also updating — so the input distribution shifts with every gradient step. Layer $l$ adapts to distribution $D_t$, but by the next step, the layer below has changed and now produces distribution $D_{t+1}$. Layer $l$ must perpetually chase a moving target. This slows learning and requires lower learning rates to prevent instability.

**The core insight**: after each layer's transformation, explicitly re-normalize the activations to have controlled statistics. This removes the distributional drift, decouples the optimization of different layers, and allows higher learning rates.

**What breaks** without normalization: activations drift into saturation regions (tanh/sigmoid gradients vanish), training requires very careful initialization and learning rate tuning, and deep networks often fail to converge at all.

> **This motivation is historical, not settled.** ICS is the argument from the original BatchNorm paper (Ioffe & Szegedy, 2015) and it's why normalization was invented — but it is not the current explanation for *why it works*. Santurkar et al. (2018) injected distribution shift *after* the normalization layer and found BN still helped, which the ICS story cannot account for; they argue the real mechanism is a smoother, better-conditioned loss landscape that tolerates larger steps. Learn the ICS framing because interviewers reference it, but don't assert it as the accepted cause — see the trap in [Interview Angles](#interview-angles).

---

## Batch Normalization

**The problem**: the most natural "controlled statistics" would be zero mean and unit variance across the feature dimension. But how do you estimate mean and variance? A single example gives noisy statistics. Use the batch.

**The core insight**: for each feature, normalize across all examples in the mini-batch simultaneously. This gives stable statistics (many data points) without needing the full dataset.

**The mechanics**: for a mini-batch $\{x_1, \ldots, x_m\}$, for each feature independently:

$$\mu_B = \frac{1}{m} \sum_i x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_i (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

$\gamma$ and $\beta$ are learned per-feature scale and shift — they let the network undo the normalization if needed. $\epsilon$ prevents division by zero.

At inference, batch statistics are unavailable (or noisy for batch size 1). BatchNorm uses a running mean and variance accumulated during training via exponential moving average.

```python
model.train()  # uses batch statistics
model.eval()   # uses running statistics — CRITICAL to call before inference
```

**What breaks**: with small batch sizes ($< 8$ or so), the per-batch mean and variance estimates are noisy. The "normalization" introduces more noise than it removes. Single-example inference with batch-norm statistics is wrong unless you call `model.eval()` — a common and silent bug. BatchNorm also breaks for variable-length sequences where padding means different batch elements have different numbers of meaningful tokens.

---

## Layer Normalization

**The problem**: BatchNorm requires a batch to compute statistics, but many settings have batch size 1 (autoregressive generation), variable-length sequences (NLP), or small batches where batch statistics are unreliable. You need a normalization that works sample-by-sample.

**The core insight**: normalize across the *feature* dimension for each sample independently. Each sample is self-normalized — no other samples needed.

**The mechanics**: for a single sample $x \in \mathbb{R}^d$:

$$\mu = \frac{1}{d} \sum_j x_j, \quad \sigma^2 = \frac{1}{d} \sum_j (x_j - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat{x} + \beta$$

$\gamma$, $\beta$ are learned per-feature. No dependence on batch size or other samples.

Train/inference behavior is identical — no running statistics needed.

**What breaks**: LayerNorm normalizes over all features of a single sample. If features have very different natural scales or semantics, forcing them to share the same mean and variance destroys structure. This is generally not a problem in Transformer embeddings (all dimensions are treated symmetrically), but can be an issue for heterogeneous feature spaces.

Used in: BERT, GPT, T5, LLaMA, virtually all Transformer architectures.

---

## RMSNorm

**The problem**: LayerNorm computes both the mean and variance. Subtracting the mean is an additional operation. Does the mean subtraction actually matter for training stability?

**The core insight**: empirically, most of LayerNorm's benefit comes from the variance scaling, not the mean centering. If you just divide by the root mean square, you get almost the same quality at lower computational cost.

**The mechanics**:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_j x_j^2}, \quad y = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

No mean subtraction. $\gamma$ is a learned per-feature scale (no $\beta$ shift needed since there is no mean centering to undo).

**What breaks**: if the mean of $x$ is significantly non-zero, RMSNorm does not remove it. The representation retains a mean shift. This is generally tolerable because the next layer's bias and the subsequent operations absorb the mean. But if mean shifts accumulate through many layers, optimization can be slightly less stable than with full LayerNorm.

Used in LLaMA, Mistral, Gemma, and most modern LLMs.

---

## Instance Normalization

**The problem**: in style transfer, you want to separate an image's *content* (spatial structure, edges) from its *style* (color distribution, texture statistics). The style is encoded in the per-channel mean and variance across spatial positions. If you normalize those statistics, you remove the style while preserving content.

**The core insight**: normalize each channel of each image independently over its spatial dimensions. This removes the channel's mean and variance — erasing style information.

**The mechanics**: for input $x \in \mathbb{R}^{N \times C \times H \times W}$, compute $\mu$ and $\sigma^2$ over $(H, W)$ for each $(N, C)$ independently:

$$y_{nchw} = \frac{x_{nchw} - \mu_{nc}}{\sigma_{nc}}$$

**Adaptive Instance Normalization (AdaIN)**: normalize the content feature statistics, then re-scale and shift by the style's statistics:

$$\text{AdaIN}(x_\text{content}, x_\text{style}) = \sigma(x_\text{style}) \cdot \frac{x_\text{content} - \mu(x_\text{content})}{\sigma(x_\text{content})} + \mu(x_\text{style})$$

This transfers style by replacing content statistics with style statistics.

**What breaks**: normalizing over only $H \times W$ gives noisy statistics for small spatial dimensions (e.g., $1 \times 1$ feature maps in a deep network). InstanceNorm is inappropriate for classification tasks where per-sample mean and variance carry discriminative information.

---

## Group Normalization

**The problem**: BatchNorm degrades at small batch sizes. LayerNorm normalizes over all channels simultaneously, which may not be appropriate for CNNs where different channel groups learn different types of features. You want something batch-independent but still respects channel structure.

**The core insight**: normalize within groups of channels, per sample. Larger than InstanceNorm's per-channel normalization (more statistics), but smaller than LayerNorm's cross-channel normalization (preserves some channel structure).

**The mechanics**: divide $C$ channels into $G$ groups. For each sample and each group, normalize over $(C/G, H, W)$:

- $G = 1$: equivalent to LayerNorm (over all channels)
- $G = C$: equivalent to InstanceNorm (one channel per group)
- $G = 32$: typical setting in detection/segmentation networks

**What breaks**: choosing $G$ requires the channel count to be divisible by $G$. Performance depends on group size — too many groups (each with few channels) and statistics become noisy; too few groups and channel-level structure is destroyed.

Used in: object detection (FPN, Mask R-CNN), video models, any setting where batch sizes are small per GPU.

---

## Pre-Norm vs Post-Norm

**The problem**: in Transformers, where does normalization go — before or after the attention/FFN sublayer? The original "Attention is All You Need" paper placed it after the residual addition (Post-Norm). Later models moved it before (Pre-Norm). Why?

**Post-Norm** (original Transformer):

$$x = \text{LayerNorm}(x + \text{Sublayer}(x))$$

The residual is added first; normalization is applied after. The gradient must flow through the normalization layer to reach the residual stream — this adds a scaling step that can destabilize gradients in very deep networks.

**Pre-Norm** (modern default — GPT-2+, LLaMA, most LLMs):

$$x = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Normalization is applied to the input *before* the sublayer. The residual stream $x$ receives the sublayer's output without any normalization on the skip path. The gradient flows directly through the residual addition with gradient magnitude 1 — no normalization scaling in the gradient path of the residual stream.

**What breaks** with Pre-Norm: the final residual stream values are not normalized before the output head. This requires a final LayerNorm before the output linear layer, which all Pre-Norm architectures include. Without this final norm, the logit magnitudes can vary widely across positions.

---

## Weight Normalization

**The problem**: BatchNorm, LayerNorm, and their variants normalize *activations* — but the instability could also be addressed by normalizing the *weights* themselves, decoupling their magnitude from their direction.

**The core insight**: reparameterize each weight vector as a direction $v$ and a magnitude $g$: $w = g \cdot v / \|v\|$. Optimization over $g$ controls magnitude; optimization over $v$ controls direction. No batch statistics needed.

**What breaks**: weight normalization does not remove internal covariate shift — it only removes the scaling degree of freedom from weights. LayerNorm is generally preferred for deep networks.

Used in: WaveNet, normalizing flow models where batch statistics would break the invertibility guarantee.

---

## Spectral Normalization

**The problem**: in GANs, the discriminator can become arbitrarily large in magnitude — its gradients can explode, destabilizing the entire training dynamic. You need the discriminator to be Lipschitz-constrained.

**The core insight**: constrain each layer's largest singular value (spectral norm) to 1. A matrix with spectral norm 1 is 1-Lipschitz. This bounds how much the discriminator can amplify signals.

**The mechanics**: divide each weight matrix by its spectral norm, estimated efficiently via power iteration during each forward pass.

**What breaks**: spectral normalization reduces the expressiveness of the discriminator. A very tight Lipschitz constraint may prevent the discriminator from distinguishing real from fake examples effectively, slowing or stalling GAN training. The constraint is a regularizer — like all regularizers, too much hurts performance.

---

## Normalization Summary

| Method | Normalizes over | Batch-independent | Primary use |
| :--- | :--- | :--- | :--- |
| **BatchNorm** | Batch dim per feature | No | CNNs, large-batch training |
| **LayerNorm** | Features per sample | Yes | Transformers, all NLP |
| **RMSNorm** | Features per sample (no mean) | Yes | Modern LLMs |
| **InstanceNorm** | Spatial dims per sample+channel | Yes | Style transfer |
| **GroupNorm** | Channel groups per sample | Yes | Small-batch detection |
| **WeightNorm** | Weight vector magnitude | Yes | Flow models |
| **SpectralNorm** | Weight matrix spectral norm | Yes | GAN discriminators |

---

## Interview Angles

### Q: Derive batch normalization and explain why it helps training. [Medium]
  
A: For a mini-batch of activations {x_1, ..., x_m}, BN computes: μ_B = (1/m)Σx_i (batch mean); σ²_B = (1/m)Σ(x_i - μ_B)² (batch variance); x̂_i = (x_i - μ_B)/√(σ²_B + ε) (normalize); y_i = γ·x̂_i + β (scale and shift with learnable params). The normalization reduces internal covariate shift — the distribution of each layer's input changes less as earlier layers update, so later layers experience a more stable input distribution. This allows higher learning rates (the normalization prevents activations from exploding), acts as regularization (the batch statistics introduce noise similar to dropout), and makes the model less sensitive to initialization. At inference, use running statistics (exponential moving averages computed during training) rather than batch statistics. Key limitation: BN depends on batch size — with batch=1, the batch statistics are just the sample itself (no normalization). This is why LayerNorm is used for NLP/transformers where sequences have variable length and batch size may be small.

**Cross-questions to expect:**
- *"You said BN reduces internal covariate shift — is that actually why it works?"* → This is the trap question. The ICS explanation is from the original paper and is now contested: Santurkar et al. (2018) showed BN still helps when you *inject* distribution shift after the normalization, and argued the real mechanism is a smoother loss landscape (better-conditioned gradients, so larger steps stay stable). Say what BN does mechanically, then note the explanation is disputed — claiming ICS as settled fact is a dated answer.
- *"Where exactly do you place BN relative to the activation?"* → Original paper puts it before the nonlinearity (conv → BN → ReLU); this is still the common default. After-activation sometimes trains slightly better in practice. Either is defensible if you say why.
- *"What happens to BN at inference with batch size 1?"* → Nothing bad, because inference uses the running averages, not batch statistics. The batch-size problem is a *training* problem. Confusing the two is a common slip.
- *"Why do the learnable γ and β exist if you just normalized?"* → Normalization forces zero mean/unit variance, which may be the wrong distribution for the next layer — γ/β let the network undo it. Setting γ=√σ² and β=μ recovers the identity, so BN can learn to do nothing.

**Trap:** Asserting "BN reduces internal covariate shift" as the accepted explanation. It's the historical motivation, not the current understanding — the loss-smoothing account has better evidence. Interviewers who know the literature use this to separate memorized answers from real understanding.

---

### Q: When would you use Layer Norm vs Batch Norm vs Group Norm? [Medium]
  
A: **Batch Norm**: best for large-batch training on CNNs with fixed-size inputs (ImageNet classification) — statistics are stable with batch size ≥ 32; fails at small batch sizes or variable-length sequences. **Layer Norm**: normalizes across the feature dimension of each individual sample, independent of batch — preferred for NLP/transformers (variable-length sequences, small batches common, autoregressive generation has batch=1) and RNNs. All modern LLMs use LayerNorm or RMSNorm. **Group Norm**: divides channels into G groups, normalizes within each group — intermediate between BN and LN; designed for object detection/segmentation where batch size is typically 1-2 (high-res images, memory-limited); GN outperforms BN at batch size 2-8. **Instance Norm**: normalizes per-sample, per-channel — designed for style transfer where spatial statistics encode style; removes style information from feature maps. **RMSNorm**: LayerNorm without mean subtraction (only scale by RMS) — 10-15% faster than LN with negligible quality loss; used in Llama, Gemma, Mistral.

**Cross-questions to expect:**
- *"Why can't transformers just use BatchNorm?"* → Sequences are variable-length and padded. Batch statistics would be computed partly over padding, and they'd shift with batch composition. Autoregressive decoding also runs at batch=1 per step. LN sidesteps all of it by normalizing within a sample.
- *"What does RMSNorm drop, and why is it safe to drop?"* → Mean subtraction — it keeps only the scaling by RMS. Empirically the re-centering contributes little once the network has learnable scale, so you get the speedup for negligible quality loss.
- *"Pre-norm or post-norm, and why does it matter?"* → Post-norm (original Transformer) needs LR warm-up to train stably at depth; pre-norm (LN inside the residual branch) gives a clean gradient path through the residual stream and trains stably without heavy warm-up. Nearly all modern LLMs are pre-norm — see [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm) above for the residual-stream argument.
- *"Group Norm with G=1 or G=C — what do you get?"* → G=1 is LayerNorm over all channels; G=C is InstanceNorm. GN interpolates between them, which is why it degrades gracefully at small batch.

**Trap:** Reciting the list without connecting it to the constraint that drives the choice. The real axis is *what does the normalization depend on* — BN depends on other samples in the batch, everything else doesn't. Every practical consequence (small batches, variable length, batch=1 decoding) follows from that one fact.

---

### Q: Why does batch normalization act as regularization, and how does this interact with dropout? [Hard]
  
A: BN introduces two sources of randomness: (1) mean and variance are computed over the current mini-batch, not the full dataset — these are noisy estimates, adding stochastic noise to each activation; (2) BN computed on one batch cannot perfectly represent the true distribution, so each sample's normalized value depends on which other samples it's batched with. This noise acts similarly to dropout — it prevents co-adaptation of neurons. When BN and dropout are used together, they can interact poorly: during training, dropout scales activations by 1/p; the scale change is absorbed by BN's normalization (it renormalizes regardless of scale). But at test time, without dropout, the batch statistics see different variance, causing a train/test discrepancy. In practice: for transformers with LayerNorm, dropout after attention/FFN works well because LN doesn't depend on batch statistics. For CNNs with BN, using dropout after BN layers typically hurts — either use one or the other; if both, put dropout before BN.

**Cross-questions to expect:**
- *"If BN already regularizes, why do LLMs still use dropout?"* → LayerNorm has no batch-derived noise — it's deterministic given the sample. So the regularization that comes free with BN doesn't exist in a transformer, and dropout has to supply it.
- *"You said put dropout before BN. What actually goes wrong in the other order?"* → Dropout changes the activation variance at training time but not at test time. BN estimates its running variance from the dropout-perturbed training activations, then at test time sees undropped activations with different variance — the stored statistics are calibrated to the wrong distribution. Li et al. (2019) call this the "variance shift."
- *"Does the BN regularization effect get stronger or weaker as batch size grows?"* → Weaker. Larger batches give less noisy statistics, so the noise-driven regularization fades — part of why very-large-batch training often needs extra explicit regularization.
- *"How does this interact with gradient accumulation?"* → Accumulation doesn't change BN's statistics: BN normalizes per forward pass, so the effective statistical batch stays the micro-batch size, not the accumulated size. People expecting accumulation to fix small-batch BN are surprised by this.

**Trap:** Treating "BN regularizes" as a reason to drop other regularization. The effect is a side effect of noisy batch statistics — it weakens with large batches, vanishes at inference, and disappears entirely if you switch to LayerNorm. It is not a substitute for weight decay or dropout.
