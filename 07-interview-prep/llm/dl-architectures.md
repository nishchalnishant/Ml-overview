# Deep Learning Architectures — Interview Reference

Fast architecture comparison with the key math, trade-offs, and interview answers for each major family.

---

## 1. CNN (Convolutional Neural Network)

**Core operation:** convolution applies a learned kernel $W$ across spatial positions:

$$(f * W)[i, j] = \sum_{m} \sum_{n} f[i+m, j+n] \cdot W[m, n]$$

**Why it works for images:**
- **Locality:** pixels near each other are correlated; local receptive fields exploit this
- **Parameter sharing:** the same kernel detects a feature (edge, texture) anywhere in the image
- **Translation equivariance:** shifting the input shifts the feature map, but doesn't change what features are detected

**Standard building blocks:**

| Block | Operation | Purpose |
| :--- | :--- | :--- |
| Conv2d | $W * x + b$ | Feature extraction |
| BatchNorm | $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | Training stability |
| ReLU | $\max(0, x)$ | Non-linearity |
| MaxPool | $\max$ over window | Spatial downsampling |
| GlobalAvgPool | Mean over $H \times W$ | Aggregate to vector |

**Parameter count:** for a conv layer with $C_{in}$ input channels, $C_{out}$ output channels, and $k \times k$ kernel:
$$\text{params} = C_{out} \times (C_{in} \times k^2 + 1)$$

**Key architectures:**

| Model | Year | Innovation |
| :--- | :--- | :--- |
| AlexNet | 2012 | Deep conv + ReLU + dropout |
| VGG | 2014 | Depth with 3×3 convolutions |
| ResNet | 2015 | Skip connections: $y = F(x) + x$ |
| EfficientNet | 2019 | Compound scaling (depth+width+resolution) |
| ConvNeXt | 2022 | CNN with Transformer-style design choices |

**ResNet skip connection:**
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)  # identity shortcut
```

**Interview answer:** CNNs exploit the spatial structure of images through parameter sharing (the same kernel everywhere) and local connectivity. ResNet's skip connection $y = F(x) + x$ made very deep networks trainable by solving gradient vanishing — the residual formulation makes identity the default, so even if $F(x)$ vanishes, the signal passes through.

---

## 2. RNN / LSTM / GRU

**Problem CNNs can't solve:** sequential dependencies where position $t$ depends on all previous positions.

### Vanilla RNN

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**Vanishing gradient problem:** gradients flow through $\prod_{t=1}^{T} W_{hh}$. If $|W_{hh}| < 1$, this shrinks exponentially; if $|W_{hh}| > 1$, it explodes. Sequences longer than ~20 steps become untrainable.

### LSTM (Long Short-Term Memory)

Introduces a **cell state** $c_t$ — a highway for gradient flow — controlled by gates:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(c_t)$$

The additive update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ prevents gradient vanishing — gradients flow through addition, not multiplication.

### GRU (Gated Recurrent Unit)

Simpler: merges forget+input into one update gate, no separate cell state:

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Comparison:**

| | RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| **Parameters** | Low | High (4× RNN) | Medium (3× RNN) |
| **Long-range memory** | Poor | Good | Good |
| **Training speed** | Fast | Slow | Medium |
| **Parallelizable** | No | No | No |
| **Best for** | Short sequences | Long sequences | Long sequences (efficient) |

**Why Transformers replaced RNNs:** RNNs are sequential — token $t$ cannot be processed before token $t-1$, limiting GPU utilization. Transformers process all positions in parallel via attention.

---

## 3. Transformer

**Core innovation:** replace recurrent processing with self-attention — every position attends to every other position directly.

**Scaled dot-product attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $Q = XW_Q$, $K = XW_K$, $V = XW_V$ — linear projections
- Dividing by $\sqrt{d_k}$ prevents large dot products from saturating softmax
- Output: weighted sum of values, where weights reflect query-key similarity

**Multi-head attention:** run $h$ attention heads in parallel, concatenate:
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Transformer block:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm (modern: norm before, not after)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
```

**Complexity:**

| Component | Compute | Memory |
| :--- | :--- | :--- |
| Self-attention | $O(n^2 d)$ | $O(n^2)$ — the attention matrix |
| Feed-forward | $O(n d^2)$ | $O(n d)$ |
| Full transformer (L layers) | $O(L n^2 d + L n d^2)$ | $O(L n^2)$ |

The $O(n^2)$ attention cost is why long-context models (>32k tokens) need Flash Attention or sparse attention approximations.

**Positional encoding (sinusoidal):**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Modern: RoPE (Rotary Position Embedding) encodes relative positions in the Q/K rotation, generalizes better to unseen lengths.

**Encoder vs decoder:**

| | Encoder | Decoder |
| :--- | :--- | :--- |
| **Attention** | Bidirectional (sees all tokens) | Causal (only past tokens) |
| **Use case** | Classification, embeddings (BERT) | Generation (GPT, LLaMA) |
| **Training** | Masked Language Model | Next token prediction |

---

## 4. GAN (Generative Adversarial Network)

**Two-player game:** generator $G$ tries to fool discriminator $D$; $D$ tries to distinguish real from fake:

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

At Nash equilibrium: $D(x) = 0.5$ everywhere — discriminator can't distinguish real from fake.

**Training instabilities and fixes:**

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| Mode collapse | Generator maps all $z$ to few outputs | Minibatch discrimination, unrolled GANs |
| Vanishing gradient | $D$ saturates early | Non-saturating loss: $\max_G \mathbb{E}[\log D(G(z))]$ |
| Training oscillation | No clear convergence criterion | Wasserstein GAN (Earth-mover distance) |

**WGAN loss:**
$$\mathcal{L}_{\text{WGAN}} = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

where $D$ is 1-Lipschitz (enforced via gradient penalty or weight clipping).

**Current status:** GANs largely replaced by diffusion models for image generation — diffusion is more stable, no adversarial training needed, better diversity.

---

## 5. Autoencoder / VAE

**Autoencoder:** encoder $E$ maps $x \to z$ (bottleneck), decoder $D$ maps $z \to \hat{x}$:

$$\mathcal{L}_{\text{AE}} = \|x - D(E(x))\|^2$$

Applications: compression, denoising, anomaly detection (high reconstruction loss = anomaly).

**VAE (Variational Autoencoder):** encoder outputs distribution parameters $\mu, \sigma$, not a point:

$$q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) \cdot I)$$

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \| p(z))}_{\text{regularization}}$$

The KL term pushes the latent space toward $\mathcal{N}(0, I)$, making interpolation and generation possible.

**Reparameterization trick** (makes sampling differentiable):
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## 6. Diffusion Models

**Forward process:** gradually add Gaussian noise over $T$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

Using the reparameterized form (direct from $x_0$):
$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$.

**Reverse process:** learn to denoise:
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Training objective** (simplified): predict the added noise $\epsilon$:
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

**Why diffusion beat GANs:**
- Stable training — no adversarial game, just regression
- Better diversity — no mode collapse
- Controllable — guidance methods (classifier-free guidance) work reliably
- Cost: slow inference (many denoising steps). Fixed by DDIM (deterministic sampling, 10–50 steps) and flow matching.

---

## 7. Architecture Selection Guide

| Task | Architecture | Why |
| :--- | :--- | :--- |
| Image classification | CNN (ResNet, EfficientNet) or ViT | Spatial inductive bias (CNN) or scale (ViT) |
| Object detection | CNN backbone + detector head (YOLO, DETR) | Spatial features + bounding box regression |
| Text generation | Decoder-only Transformer (GPT) | Causal attention, scales with data |
| Text understanding | Encoder Transformer (BERT) | Bidirectional context |
| Image generation | Diffusion (UNet + noise schedule) | Stable training, high quality |
| Time-series | Transformer or temporal CNN | Depends on sequence length and parallelism needs |
| Graph data | GNN (message passing) | Permutation invariance, topology-awareness |
| Tabular data | Gradient boosting or MLP | Rarely benefits from spatial/sequential inductive biases |
