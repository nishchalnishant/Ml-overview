# Generative Models — Deep Dive

> Covers VAEs, GANs, Diffusion Models (DDPM/DDIM), Latent Diffusion, Flow Matching, and the DiT/FLUX architecture wave. Math-first, production-second.

---

## Table of Contents

1. [Variational Autoencoders (VAEs)](#1-variational-autoencoders-vaes)
2. [Generative Adversarial Networks (GANs)](#2-generative-adversarial-networks-gans)
3. [Diffusion Models — DDPM](#3-diffusion-models--ddpm)
4. [Faster Sampling — DDIM & Schedulers](#4-faster-sampling--ddim--schedulers)
5. [Classifier-Free Guidance (CFG)](#5-classifier-free-guidance-cfg)
6. [Latent Diffusion Models (Stable Diffusion)](#6-latent-diffusion-models-stable-diffusion)
7. [Flow Matching & Rectified Flow](#7-flow-matching--rectified-flow)
8. [Diffusion Transformers (DiT) & FLUX](#8-diffusion-transformers-dit--flux)
9. [Production Considerations](#9-production-considerations)
10. [Interview Questions](#10-interview-questions)

---

## 1. Variational Autoencoders (VAEs)

### Motivation

A standard autoencoder maps `x → z → x̂` through a bottleneck. The latent space `z` is unstructured — there's no guarantee that points sampled from it decode to valid outputs. VAEs impose a probabilistic structure on `z` that enables clean sampling.

### The ELBO Derivation

We want to learn a generative model `p_θ(x) = ∫ p_θ(x|z) p(z) dz`. This integral is intractable. Instead, introduce an encoder `q_φ(z|x)` that approximates the true posterior `p_θ(z|x)`:

```
log p_θ(x) = log ∫ p_θ(x|z) p(z) dz

         = log E_{z~q_φ(z|x)} [ p_θ(x|z) p(z) / q_φ(z|x) ]

         ≥ E_{z~q_φ(z|x)} [ log p_θ(x|z) ] - KL(q_φ(z|x) || p(z))
           \_______________________/   \___________________________/
               Reconstruction term          Regularization term
```

This lower bound is the **Evidence Lower BOund (ELBO)**. Maximizing ELBO jointly over θ (decoder) and φ (encoder):

```
L(θ, φ; x) = E_{z~q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

**KL term with Gaussian prior:** If `q_φ(z|x) = N(μ, σ²I)` and `p(z) = N(0, I)`:

```
KL(N(μ, σ²) || N(0, 1)) = ½ Σ_j (1 + log σ_j² - μ_j² - σ_j²)
```

Analytic — no sampling needed for this term.

**Reparameterization trick:** The reconstruction term requires sampling `z ~ q_φ(z|x)`, which blocks gradient flow. Instead:

```
z = μ_φ(x) + σ_φ(x) ⊙ ε,   ε ~ N(0, I)
```

Now z is a deterministic function of x and ε — gradients flow through μ and σ.

```python
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std         # reparameterization trick

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
```

### β-VAE

Setting β > 1 in the loss `L = recon_loss + β · KL` increases pressure on the KL term, forcing more disentangled representations (each latent dimension encodes an independent factor of variation). β = 4–10 empirically produces more interpretable latents at the cost of reconstruction quality.

---

## 2. Generative Adversarial Networks (GANs)

### The Minimax Game

GAN trains two networks adversarially:
- **Generator G:** maps noise z to data space x = G(z)
- **Discriminator D:** classifies real vs. fake, outputs P(real | input)

```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

At Nash equilibrium: `p_G = p_data` and `D(x) = 0.5` everywhere. In practice, training dynamics are far from this ideal.

### Training Instability

**Vanishing gradients:** When D is too strong, `log(1 - D(G(z))) ≈ log(1) = 0` — G gets no gradient signal. Fix: train G to maximize `log D(G(z))` (non-saturating objective) instead of minimizing `log(1 - D(G(z)))`.

**Mode collapse:** G learns to produce a small subset of modes that fool D, ignoring the rest of the distribution. G output is high quality but low diversity.

**Training instability:** GAN training is notoriously sensitive to hyperparameters, learning rates, architecture choices.

### WGAN: Wasserstein Distance

Replace JS divergence (standard GAN) with Wasserstein-1 distance, which has better gradient properties:

```
W(p_r, p_g) = sup_{||f||_L ≤ 1} E_{x~p_r}[f(x)] - E_{x~p_g}[f(x)]
```

The discriminator (now called **critic**) is constrained to be 1-Lipschitz. Gradient penalty enforcement:

```
L = E[D(x̃)] - E[D(x)] + λ E[(||∇_x̂ D(x̂)||_2 - 1)²]
```

where `x̂` is a random interpolation between real and fake. This gradient penalty (WGAN-GP) is the dominant WGAN variant — more stable training with meaningful loss curves.

### StyleGAN2 Architecture (State-of-Art GAN)

StyleGAN2 (Karras et al., 2020) produces the highest-quality GAN outputs:

- **Mapping network:** z → w (8-layer MLP) — moves from spherical Gaussian to more structured "style space"
- **Synthesis network:** Constant 4×4 → progressive growing via adaptive instance normalization
- **Style injection (AdaIN):** at each layer, apply style vector to control feature statistics
- **Weight demodulation:** normalization that removes pixel-level artifacts

GANs are largely superseded by diffusion models for image generation but remain relevant for video synthesis and real-time generation.

---

## 3. Diffusion Models — DDPM

### Intuition

Diffusion models learn to reverse a gradual noising process. Forward: destroy the image step by step. Reverse: learn to denoise step by step. At inference, start from pure noise and run the learned reverse process.

### Forward Process

Add Gaussian noise at each of T steps. The key property: can jump directly from x_0 to x_t without iterating through all steps (closed-form):

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

where:
  αt = 1 - βt           (noise schedule: β_1, ..., β_T small positive values)
  ᾱ_t = ∏_{s=1}^t αs   (cumulative product)
```

Reparameterized:

```
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,    ε ~ N(0, I)
```

At T steps with appropriate β schedule: `x_T ≈ N(0, I)` — pure noise.

### Reverse Process

The model `ε_θ(x_t, t)` predicts the noise ε added to x_0 to get x_t. Training objective (simplified):

```
L_simple = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]

where x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
```

This is equivalent to score matching (estimating the score ∇_x log p(x_t)) — DDPM noise prediction and score-based diffusion are mathematically unified via the relationship:

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
```

### Noise Schedule

The β schedule controls how quickly noise is added:

```python
def linear_schedule(T: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T: int, s=0.008):
    """Cosine schedule (Nichol & Dhariwal, 2021) — better than linear"""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)
```

The cosine schedule maintains more information in early steps (avoids over-destroying the image too quickly), improving sample quality.

### U-Net Architecture for DDPM

The noise predictor is typically a U-Net with:
- **Downsampling blocks:** conv → attention → residual
- **Bottleneck:** multi-head self-attention
- **Upsampling blocks:** skip connections from encoder + conv → attention → residual
- **Time conditioning:** sinusoidal time embedding → projected → added to each residual block
- **Class conditioning (optional):** class embedding → cross-attention or added to time embedding

```python
class DiffusionUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        # Encoder: 256 → 128 → 64 → 32
        self.down = nn.ModuleList([
            ResBlock(base_channels, base_channels * 2, time_dim),
            ResBlock(base_channels * 2, base_channels * 4, time_dim),
            AttentionBlock(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 8, time_dim),
        ])
        # Bottleneck
        self.mid = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 8, time_dim),
            AttentionBlock(base_channels * 8),
        )
        # Decoder with skip connections
        self.up = nn.ModuleList([
            ResBlock(base_channels * 16, base_channels * 4, time_dim),
            ResBlock(base_channels * 8, base_channels * 2, time_dim),
            ResBlock(base_channels * 4, base_channels, time_dim),
        ])

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skips = []
        for layer in self.down:
            x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)
            skips.append(x)
        x = self.mid(x)
        for layer, skip in zip(self.up, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = layer(x, t_emb)
        return x
```

---

## 4. Faster Sampling — DDIM & Schedulers

### The Problem with DDPM Sampling

DDPM requires T=1000 denoising steps at inference — 1000 forward passes through the U-Net. At 512×512 resolution with a large U-Net, this takes 30+ seconds per image.

### DDIM (Denoising Diffusion Implicit Models)

DDIM (Song et al., 2020) derived a non-Markovian reverse process that produces the same marginal distributions as DDPM but can skip steps:

```
x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t
         + √(1-ᾱ_{t-1} - σ_t²) · ε_θ(x_t, t)
         + σ_t · ε_t

where σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})
```

When η=0: **deterministic DDIM** — same latent → same image (enables interpolation in latent space). Can reduce from 1000 to 20-50 steps with acceptable quality.

### Common Schedulers (2023-2025)

| Scheduler | Steps | Quality | Notes |
|-----------|-------|---------|-------|
| DDPM | 1000 | Highest | Too slow for production |
| DDIM | 20-50 | Good | Deterministic, invertible |
| DPM-Solver++ | 10-20 | Very good | Best quality-speed ratio |
| LCM (Latent Consistency) | 4-8 | Good | Requires LCM fine-tuning |
| SDXL-Turbo / ADD | 1-4 | Decent | Adversarial diffusion distillation |

**DPM-Solver++** is the current production default — achieves near-DDPM quality in 15-20 steps by solving the diffusion ODE with a high-order numerical method.

---

## 5. Classifier-Free Guidance (CFG)

### Motivation

We want to generate images conditioned on a text prompt y. Classifier guidance uses a separate classifier `p(y|x_t)` to steer the denoising process. Classifier-free guidance eliminates the need for a separate classifier.

### CFG Mechanism

Train the diffusion model with and without conditioning (randomly drop the condition with probability p_uncond during training):

```
ε_θ(x_t, t, y)   # conditional noise prediction
ε_θ(x_t, t, ∅)   # unconditional noise prediction (y = null token)
```

At inference, combine the two predictions:

```
ε_guided = ε_θ(x_t, t, ∅) + w · (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

where w is the **guidance scale** (typically 7.5 for Stable Diffusion).

**Interpretation:** The guided prediction amplifies the difference between conditional and unconditional predictions, pushing the sample toward the conditioning signal.

**Trade-off:**
- Higher w → stronger adherence to prompt, less diversity, potential quality artifacts
- Lower w → more diverse samples, less prompt adherence
- w=1 → no guidance (conditional only)
- w=7.5 → typical production default

**Implementation (during inference only):**

```python
@torch.no_grad()
def p_sample_cfg(model, x_t, t, condition, guidance_scale=7.5):
    # Run model twice: conditional and unconditional
    # Batch them together for efficiency
    x_in = torch.cat([x_t, x_t])
    cond_in = torch.cat([condition, empty_condition])
    t_in = torch.cat([t, t])
    
    noise_pred = model(x_in, t_in, cond_in)
    noise_cond, noise_uncond = noise_pred.chunk(2)
    
    # CFG interpolation
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    return denoise_step(x_t, t, noise_guided)
```

---

## 6. Latent Diffusion Models (Stable Diffusion)

### The Key Insight

Running diffusion in pixel space at 512×512 is expensive: 512×512×3 = 786K-dimensional data. Latent Diffusion (Rombach et al., 2022) runs diffusion in a compressed **latent space**:

```
Encoder: x (512×512×3) → z (64×64×4)   [compression factor: 48×]
Diffusion model: operates on z (64×64×4)
Decoder: z → x̂ (512×512×3)
```

The VAE (encoder/decoder) is pre-trained and frozen — diffusion only trains in the latent space. This gives ~48× speedup in forward passes.

### Architecture

**VAE (KL-regularized):**
- Encoder: convolutional downsampler + attention → 64×64×4 latent
- Decoder: convolutional upsampler → pixel space reconstruction
- KL penalty keeps latents well-behaved (near-Gaussian)

**U-Net with Cross-Attention:**
The latent diffusion U-Net adds cross-attention layers that condition on text:

```
Cross-Attention(Q=z_features, K=text_emb, V=text_emb)
```

Text is encoded via CLIP's text encoder. The U-Net "attends" to text tokens to inject conditioning at each resolution level.

**CLIP Text Encoder:**
CLIP (Contrastive Language-Image Pre-Training) encodes text into a 77×768 sequence of token embeddings (not just a single class embedding). Each spatial feature in the U-Net cross-attends to all 77 text tokens simultaneously.

### SDXL (Stable Diffusion XL)

SDXL (Podell et al., 2023) adds:
- Two CLIP encoders (ViT-L and OpenCLIP ViT-G) — concatenated text embeddings
- Size/crop conditioning (model sees original image resolution as conditioning)
- Refiner model: a second model that upscales/refines the base output at high-noise steps

---

## 7. Flow Matching & Rectified Flow

### Motivation

DDPM defines a noising process using stochastic differential equations. Flow matching is an alternative framework that defines **deterministic** probability flows — straight-line paths from noise to data.

### Rectified Flow (Liu et al., 2022)

Define a simple linear flow from noise distribution `π_0 = N(0,I)` to data distribution `π_1 = p_data`:

```
x_t = (1-t) · x_1 + t · x_0,    t ∈ [0, 1]
```

This is literally a straight line from data (t=0) to noise (t=1). The velocity field at point x_t at time t is:

```
dx/dt = x_0 - x_1   (constant — straight line)
```

Train a neural network `v_θ(x_t, t)` to predict this velocity:

```
L = E_{x_0 ~ p_data, x_1 ~ N(0,I), t ~ Uniform[0,1]} [ ||v_θ(x_t, t) - (x_0 - x_1)||² ]
```

**At inference:** start from `x_1 ~ N(0,I)`, integrate the ODE `dx/dt = v_θ(x_t, t)` backwards from t=1 to t=0. Straight-line flows are easier to integrate — can use fewer function evaluations (steps).

### Advantages over DDPM

- Fewer sampling steps (5-10 vs 20-50 for DDIM)
- More interpretable: interpolation in the flow space is linear
- Better for high-resolution generation (less error accumulation)

---

## 8. Diffusion Transformers (DiT) & FLUX

### DiT (Peebles & Xie, 2022)

Replacing the U-Net backbone with a **Vision Transformer**:
- Patch the latent (8×8 patches of 64×64×4 = 64 patches for 512×512 images)
- Apply standard transformer blocks with conditioning injection
- Condition via **adaptive layer norm (adaLN):** scale and shift LayerNorm based on time + class embeddings

```
adaLN(x, c) = γ(c) · LayerNorm(x) + β(c)
```

where γ, β are linear projections of the conditioning vector c.

**Why DiT over U-Net?**
- Scales more predictably (same scaling laws as language transformers)
- No architectural inductive biases (no convolutions, no skip connections)
- Better at capturing long-range dependencies (global attention vs. limited U-Net receptive field)

DiT-XL/2 (largest variant) outperforms all U-Net diffusion models on ImageNet class-conditional generation at 256×256 and 512×512.

### FLUX Architecture (Black Forest Labs, 2024)

FLUX is the commercial successor to Stable Diffusion by its original creators, using rectified flow + DiT:

**Key innovations:**

**Dual-stream architecture:** Separate transformer streams for image tokens and text tokens that interact via cross-attention, then merge:

```
Image tokens ─┐                    ┌─ Image tokens
               ├── Cross-Attention ─┤
Text tokens  ─┘                    └─ Text tokens
```

This is richer than simple cross-attention (SDXL) because image and text representations evolve jointly.

**Flow Matching:** FLUX uses rectified flow instead of DDPM — enables 8-10 step high-quality sampling vs 20+ for SDXL.

**Guidance distillation:** FLUX-schnell (the fast variant) is distilled to run without CFG at inference — single model pass instead of two, further 2× speedup.

**FLUX model variants:**
| Variant | Params | Steps | Use case |
|---------|--------|-------|----------|
| FLUX.1-dev | 12B | 20-50 | Research, fine-tuning |
| FLUX.1-schnell | 12B | 1-4 | Fast inference, production |
| FLUX.1-pro | 12B | — | Proprietary API only |

---

## 9. Production Considerations

### Inference Speed

**Latency bottlenecks (typical 512×512 SDXL on A100):**
- U-Net forward pass: ~35ms per step
- VAE decode: ~150ms (once, at end)
- At 20 steps: 35×20 + 150 = 850ms end-to-end

**Optimizations:**
1. **Fewer steps:** DPM-Solver++ at 15 steps, LCM at 4-8 steps
2. **Quantization:** U-Net INT8 → ~1.5× speedup, minimal quality loss
3. **xFormers / Flash Attention:** memory-efficient attention in U-Net → 20-40% speedup
4. **Model compilation:** `torch.compile` → 20% speedup on modern GPUs
5. **VAE tiling:** for very large images, tile the VAE decode to avoid OOM
6. **CUDA graph capture:** eliminate Python overhead for fixed-shape inference

### Safety and Content Filtering

Production image generation pipelines require multi-layer safety:

```python
class SafetyPipeline:
    def __init__(self):
        self.prompt_classifier = load_prompt_classifier()
        self.image_classifier = load_nsfw_classifier()

    def generate(self, prompt: str) -> Optional[Image]:
        # Layer 1: Prompt filtering
        if self.prompt_classifier.is_unsafe(prompt):
            return None

        # Layer 2: Generate
        image = diffusion_model.generate(prompt)

        # Layer 3: Output filtering
        if self.image_classifier.is_nsfw(image):
            return None

        return image
```

**Common classifiers:** CLIP-based NSFW classifiers, Stable Diffusion safety checker (ViT-based binary), Llama Guard for prompt classification.

### Fine-Tuning Methods

| Method | VRAM | Data needed | Use case |
|--------|------|-------------|----------|
| DreamBooth | 24GB+ | 3-20 images | Subject personalization |
| Textual Inversion | 8GB | 3-20 images | New concept embedding |
| LoRA | 8-12GB | 100s images | Style transfer, fine-tuning |
| Full fine-tuning | 80GB+ | 1000s+ images | Domain adaptation |

**LoRA for diffusion:** Same rank-decomposition as for LLMs, applied to the U-Net/DiT attention weight matrices. At r=4, reduces trainable params by ~100× vs full fine-tuning.

---

## 10. Interview Questions

**Q: What is the ELBO and why do we maximize it instead of the marginal likelihood?**

The marginal likelihood `p(x) = ∫ p(x|z) p(z) dz` is intractable — the integral is over all possible latent codes. The ELBO (Evidence Lower BOund) is a tractable lower bound derived via Jensen's inequality by introducing an approximate posterior `q(z|x)`. Maximizing ELBO simultaneously trains the encoder (make `q(z|x)` close to the true posterior) and decoder (make reconstruction accurate). The gap between the true log-likelihood and ELBO equals `KL(q(z|x) || p(z|x))` — so maximizing ELBO also minimizes this gap, improving posterior approximation.

---

**Q: Why did diffusion models replace GANs for high-quality image generation?**

Three key advantages:
1. **Training stability:** Diffusion training is stable — the objective is a simple noise prediction MSE. GAN training is a minimax game that frequently diverges, collapses, or oscillates.
2. **Mode coverage:** GANs are prone to mode collapse (generating a subset of the data distribution). Diffusion models learn the full distribution more reliably.
3. **Scaling:** Diffusion models (especially DiT variants) follow the same scaling laws as transformers — more parameters + more compute = better results, predictably. GANs don't scale as cleanly.

The main disadvantage of diffusion: slower inference (multiple denoising steps vs. single GAN forward pass). Flow matching and distillation methods are closing this gap.

---

**Q: What is classifier-free guidance and what does the guidance scale control?**

CFG eliminates the need for a separate classifier for conditional generation. The model is trained to make both conditional predictions `ε(x_t, y)` and unconditional predictions `ε(x_t, ∅)` (conditioning randomly dropped during training). At inference: `ε_guided = ε_uncond + w·(ε_cond - ε_uncond)`. The guidance scale w amplifies the signal that distinguishes the conditioned from unconditioned prediction — pushing samples toward the conditioning signal. Higher w = more prompt-aligned images at the cost of diversity and potential quality artifacts (manifold drift — the sample moves to a high-likelihood-under-conditioning region that may be out-of-distribution for the denoiser).

---

**Q: What is the difference between DDPM and DDIM sampling?**

DDPM defines a Markovian reverse process: `p(x_{t-1}|x_t)` depends only on x_t. This requires iterating through all T steps. DDIM defines a non-Markovian reverse process that still produces the same marginal distributions as DDPM but allows skipping steps — you can go from x_T to x_{T-50} directly. When η=0, DDIM is deterministic: the same latent always produces the same image. This enables latent space interpolation. In practice, 20-50 DDIM steps achieve quality comparable to 1000 DDPM steps.

---

**Q: Why does Stable Diffusion run diffusion in latent space rather than pixel space?**

Pixel space for 512×512×3 images is 786K dimensions — each denoising step operates on a huge tensor. The VAE compresses images to 64×64×4 latents (4096 dimensions) — a 192× reduction in dimensionality. Since diffusion requires hundreds of forward passes per image, this compression gives a corresponding speedup. The VAE is pre-trained to be a near-lossless codec (perceptual quality is preserved), so running diffusion in latent space loses minimal image quality while dramatically reducing compute cost.

---

*Last updated: May 2026 | Coverage: VAE through FLUX/DiT | Focus: math-first, production-second*
