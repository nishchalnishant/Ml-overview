---
module: Emerging Topics
topic: Emerging Trends
subtopic: Diffusion Models
status: complete
tags: [emergingtopics, ml, diffusion, generative-models, ddpm, ddim, stable-diffusion]
---
# Diffusion Models

Diffusion models are the dominant paradigm for image, audio, and video generation as of 2025. They work by learning to reverse a gradual noising process, producing samples of remarkable quality and diversity. This document covers the mathematical foundations (DDPM, score matching), efficient sampling (DDIM), latent diffusion (Stable Diffusion, DALL-E 3), conditional generation (classifier-free guidance), and flow matching — the successor framework used in FLUX and Sora.

---

## 1. The Core Idea

The central insight: **it is easier to learn to remove noise than to learn to generate from scratch.**

Instead of training a model to directly output an image from a random vector, diffusion models:
1. Define a **forward process** that gradually corrupts data x₀ into pure Gaussian noise x_T over T steps
2. Train a neural network to **reverse** this process — given a noisy image at step t, predict what the clean image was

The forward process is fixed (no learned parameters). Only the reverse process is learned. The generative model then runs the learned reverse process starting from pure noise.

```
Forward:  x₀ → x₁ → x₂ → ... → x_T ≈ N(0, I)
Reverse:  x_T → x_{T-1} → ... → x₁ → x₀  (learned)
```

---

## 2. DDPM: Denoising Diffusion Probabilistic Models

**Ho et al. (2020)** formalized the framework that became the standard.

### 2.1 Forward Process

The forward process adds Gaussian noise in T steps using a noise schedule β₁, β₂, ..., β_T (typically β_t ∈ [0.0001, 0.02]):

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) x_{t-1}, β_t I)
```

**Key property**: x_t can be sampled from x₀ in closed form without iterating all steps. Define:
```
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^{t} α_s   (cumulative product)

q(x_t | x₀) = N(x_t; sqrt(ᾱ_t) x₀, (1 - ᾱ_t) I)
```

This allows sampling noisy x_t at any step t directly:
```python
def forward_diffusion_sample(x0, t, noise_schedule):
    """Sample x_t from x_0 in one step."""
    sqrt_alpha_bar = noise_schedule.sqrt_alpha_bar[t]        # sqrt(ᾱ_t)
    sqrt_one_minus_alpha_bar = noise_schedule.sqrt_one_minus_alpha_bar[t]  # sqrt(1 - ᾱ_t)
    epsilon = torch.randn_like(x0)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
    return x_t, epsilon
```

At t = T: ᾱ_T ≈ 0, so x_T ≈ N(0, I). The data is completely destroyed.

### 2.2 Reverse Process

The true reverse q(x_{t-1} | x_t) is intractable (requires marginalizing over all x₀). But conditioned on x₀ it is tractable:

```
q(x_{t-1} | x_t, x₀) = N(x_{t-1}; μ̃_t(x_t, x₀), β̃_t I)

where:
  μ̃_t = (sqrt(ᾱ_{t-1}) β_t x₀ + sqrt(α_t)(1-ᾱ_{t-1}) x_t) / (1 - ᾱ_t)
  β̃_t = (1 - ᾱ_{t-1}) β_t / (1 - ᾱ_t)
```

The learned reverse process approximates this:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### 2.3 Training Objective

Rather than parameterizing the mean μ_θ directly, DDPM predicts the noise ε that was added:

```
L_simple = E_{t, x₀, ε} [ || ε - ε_θ(sqrt(ᾱ_t) x₀ + sqrt(1-ᾱ_t) ε, t) ||² ]
```

This is simply MSE between the true noise and the model's noise prediction.

```python
def ddpm_training_step(model, x0, noise_schedule, T):
    """Single training step for DDPM."""
    batch_size = x0.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, T, (batch_size,)).to(x0.device)
    
    # Sample noise
    epsilon = torch.randn_like(x0)
    
    # Create noisy image
    sqrt_alpha_bar = noise_schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_one_minus = noise_schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * epsilon
    
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    # Simple L2 loss
    loss = F.mse_loss(epsilon_pred, epsilon)
    return loss
```

### 2.4 Sampling (DDPM)

```python
@torch.no_grad()
def ddpm_sample(model, noise_schedule, shape, T=1000, device='cuda'):
    """Generate sample by running the reverse process."""
    x = torch.randn(shape).to(device)
    
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        epsilon_pred = model(x, t_tensor)
        
        # Compute x_{t-1}
        alpha_t = noise_schedule.alpha[t]
        alpha_bar_t = noise_schedule.alpha_bar[t]
        beta_t = noise_schedule.beta[t]
        
        # Predicted x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # Compute mean of reverse process
        coef1 = torch.sqrt(alpha_bar_t - 1) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t / alpha_t) / (1 - alpha_bar_t)  # simplified
        mean = coef1 * x0_pred + coef2 * x
        
        if t > 0:
            noise = torch.randn_like(x)
            beta_tilde = noise_schedule.beta_tilde[t]
            x = mean + torch.sqrt(beta_tilde) * noise
        else:
            x = mean
    
    return x
```

**DDPM weakness**: requires T = 1000 steps for high quality. At 256×256 with a large UNet, each sample takes ~10-30 seconds on a single GPU.

---

## 3. Score Matching and Score-Based Generative Models

**Song & Ermon (2019, 2020)** provide an equivalent but more principled view via score matching.

### 3.1 Score Functions

The **score function** of a distribution p(x) is:
```
s(x) = ∇_x log p(x)
```

This is the gradient of the log density — a vector field pointing toward high-probability regions. If you could evaluate s(x), you could generate samples by following Langevin dynamics:
```
x_{k+1} = x_k + (ε/2) ∇_x log p(x_k) + sqrt(ε) z_k,  z_k ~ N(0,I)
```

### 3.2 Score Matching Objective

Training a score model s_θ(x) ≈ ∇_x log p(x) via explicit score matching requires:
```
L_SM = E_p [ ||s_θ(x) - ∇_x log p(x)||² ]
```
But ∇_x log p(x) is unknown. **Denoising score matching** (Vincent, 2011) circumvents this by training at multiple noise levels:

```
L_DSM = E_{t, x₀, ε} [ || s_θ(x_t, t) - ∇_{x_t} log q(x_t|x₀) ||² ]
        = E_{t, x₀, ε} [ || s_θ(x_t, t) + ε/sqrt(1 - ᾱ_t) ||² ]
```

The connection to DDPM: the noise prediction ε_θ and the score s_θ are related by:
```
s_θ(x_t, t) = -ε_θ(x_t, t) / sqrt(1 - ᾱ_t)
```

Both DDPM and score-based models learn the same underlying function; they just parameterize it differently.

---

## 4. DDIM: Denoising Diffusion Implicit Models

**Song et al. (2020)** — the key insight for fast sampling.

### 4.1 Non-Markovian Forward Process

DDIM defines a broader family of forward processes with the same marginals q(x_t | x₀) as DDPM (so the same model can be used), but allows a non-Markovian reverse process:

```
q_σ(x_{t-1} | x_t, x₀) = N(x_{t-1}; 
    sqrt(ᾱ_{t-1}) x₀ + sqrt(1 - ᾱ_{t-1} - σ_t²) · ε_θ(x_t, t),
    σ_t I)
```

When σ_t = 0 for all t: the process is **deterministic** given x_T and the model. Same initial noise → same output (unlike stochastic DDPM).

### 4.2 DDIM Sampling

Because the process is deterministic and skips the Markov constraint, you can subsample the T=1000 timesteps to use only S < T steps (e.g., S=50):

```python
@torch.no_grad()
def ddim_sample(model, noise_schedule, shape, S=50, eta=0.0, device='cuda'):
    """
    DDIM sampling with S steps instead of T=1000.
    eta=0.0: deterministic DDIM
    eta=1.0: equivalent to DDPM (stochastic)
    """
    T = noise_schedule.T
    # Subsample timesteps
    timesteps = torch.linspace(T - 1, 0, S, dtype=torch.long)
    
    x = torch.randn(shape).to(device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        t_prev = timesteps[i + 1] if i + 1 < S else torch.tensor(0)
        
        alpha_bar_t = noise_schedule.alpha_bar[t]
        alpha_bar_prev = noise_schedule.alpha_bar[t_prev]
        
        # Predict noise
        epsilon_pred = model(x, t_tensor)
        
        # Predict x₀
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
        
        # Compute sigma
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        
        # DDIM update
        direction = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * epsilon_pred
        noise = sigma_t * torch.randn_like(x) if eta > 0 else 0
        
        x = torch.sqrt(alpha_bar_prev) * x0_pred + direction + noise
    
    return x
```

**DDIM speedup**: 50 steps vs 1000 → 20× faster with minimal quality loss. At eta=0, sampling is deterministic: given the same x_T noise seed, you always get the same image. This enables **interpolation in latent space**: linearly interpolate between two noise seeds and the outputs smoothly interpolate.

---

## 5. Latent Diffusion Models (Stable Diffusion)

**Rombach et al. (2022, LDM)** — the architecture behind Stable Diffusion, DALL-E 3, Midjourney v5+.

### 5.1 The Problem with Pixel-Space Diffusion

Running DDPM/DDIM directly in pixel space at 512×512 requires operating on ~786K-dimensional vectors. The UNet is expensive, and most diffusion steps refine perceptually irrelevant high-frequency details.

### 5.2 The LDM Architecture

**Stage 1: Train a Perceptual Autoencoder (VQ-VAE / KL-VAE)**

Compress images from pixel space (H×W×3) to a compact latent space (H/8 × W/8 × C):

```python
# Encoder: image → latent
z = encoder(x)  # z ∈ R^{H/8 × W/8 × 4}  (4× lower spatial, 8× fewer pixels total)

# Decoder: latent → image
x_hat = decoder(z)
```

The autoencoder is trained with perceptual loss (feature-space L2 + LPIPS) and a KL regularization term to keep the latent space approximately Gaussian.

**Stage 2: Train Diffusion Model in Latent Space**

Run the entire DDPM/DDIM process on z rather than x:
```
Forward: z₀ → z_t  (add noise to latent, not pixel)
Reverse: z_T → z₀  (predict denoised latent)
Output:  x₀ = decoder(z₀)
```

The UNet operates on 64×64×4 (for 512×512 output) instead of 512×512×3 — roughly 48× fewer operations per diffusion step.

### 5.3 Conditioning with Cross-Attention

Text conditioning is injected via cross-attention in the UNet. Text prompt → CLIP text encoder → token embeddings → cross-attention keys/values in each UNet block.

```python
class ResBlock(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        # Cross-attention for conditioning
        self.cross_attn = CrossAttention(query_dim=channels, context_dim=context_dim)
    
    def forward(self, x, context=None):
        # context = CLIP text embeddings [B, seq_len, context_dim]
        h = self.norm(x)
        h = self.conv(h)
        
        # Reshape for attention: [B, C, H, W] → [B, H*W, C]
        B, C, H, W = h.shape
        h = h.view(B, C, H * W).transpose(1, 2)
        h = self.cross_attn(h, context=context)
        h = h.transpose(1, 2).view(B, C, H, W)
        
        return x + h
```

**SDXL** improvements (Podell et al. 2023): larger UNet backbone (2.6B parameters), two-stage generation (base 1024×1024 + refiner), improved aesthetic quality.

**Stable Diffusion 3 and FLUX**: replaced UNet with a **DiT (Diffusion Transformer)**, using transformer blocks throughout instead of convolutional UNet. FLUX.1 uses flow matching (Section 7) instead of DDPM objective.

---

## 6. Classifier-Free Guidance (CFG)

**Ho & Salimans (2021)** — the key trick that makes conditional generation high quality.

### 6.1 The Problem

Training a conditional model ε_θ(x_t, t, c) where c is the text embedding works, but the model learns to interpolate between the conditional and unconditional distributions — the conditioning signal is weak.

### 6.2 Classifier Guidance (Original)

Original approach (Dhariwal & Nichol, 2021): train a noisy classifier p_φ(y|x_t) and use its gradients to steer sampling:
```
ε̃(x_t, t, c) = ε_θ(x_t, t) - γ · sqrt(1 - ᾱ_t) ∇_{x_t} log p_φ(y=c|x_t)
```

Requires training a separate noisy classifier. Expensive and constrains conditioning to classifier labels.

### 6.3 Classifier-Free Guidance

Train a single model that conditions on c but randomly drops the conditioning (replaces c with ∅) with probability p_uncond ≈ 0.1 during training:

```python
def training_step_cfg(model, x0, t, condition, p_uncond=0.1):
    # Randomly drop conditioning
    mask = torch.rand(x0.shape[0]) < p_uncond
    cond = torch.where(mask.view(-1,1,1), torch.zeros_like(condition), condition)
    
    # Standard noise prediction loss
    epsilon = torch.randn_like(x0)
    x_t = add_noise(x0, t, epsilon)
    epsilon_pred = model(x_t, t, cond)
    return F.mse_loss(epsilon_pred, epsilon)
```

At sampling time, compute both conditional and unconditional predictions and extrapolate:

```python
def cfg_predict(model, x_t, t, condition, guidance_scale=7.5):
    """
    Classifier-Free Guidance.
    guidance_scale=1.0: pure conditional (same as no CFG)
    guidance_scale=7.5: strong guidance (default for Stable Diffusion)
    guidance_scale=20+: over-guidance, loses diversity, produces artifacts
    """
    # Unconditional prediction (null text)
    eps_uncond = model(x_t, t, null_condition)
    
    # Conditional prediction
    eps_cond = model(x_t, t, condition)
    
    # Linear extrapolation away from unconditional toward conditional
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps_guided
```

Intuitively: move away from the unconditional manifold and toward the conditional manifold at rate `guidance_scale`. Increases sample quality and prompt adherence at the cost of diversity.

**Negative prompting** uses the same CFG formula: replace `eps_uncond` with the prediction conditioned on the negative prompt. "Low quality, blurry" as the negative condition steers away from those attributes.

---

## 7. Flow Matching

**Lipman et al. (2022), Liu et al. (2022)** — the framework used in FLUX, Stable Diffusion 3, and Sora.

### 7.1 Motivation

DDPM's training objective is a special case of score matching at each noise level. Flow matching is a more general framework that:
- Produces straight-line probability flows (vs curved DDPM trajectories)
- Requires fewer NFE (neural function evaluations) at inference
- More principled theoretical foundations via optimal transport

### 7.2 Continuous Normalizing Flows

Instead of a discrete Markov chain, define a continuous-time flow:
```
dx/dt = v_θ(x, t),  t ∈ [0, 1]
x₀ ~ N(0, I),  x₁ ~ p_data
```

The velocity field v_θ transports samples from noise at t=0 to data at t=1. Training: fit v_θ to a target velocity field derived from interpolating noise and data.

### 7.3 Conditional Flow Matching Objective

For a data sample x₁ and noise x₀ ~ N(0,I), define the interpolated path:
```
x_t = (1 - t) x₀ + t x₁   (straight-line interpolation)
```

The target velocity at time t along this path is just:
```
v_target(x_t, t | x₀, x₁) = x₁ - x₀   (constant velocity!)
```

Training objective:
```
L_CFM = E_{t~U[0,1], x₁~p_data, x₀~N(0,I)} [ || v_θ(x_t, t) - (x₁ - x₀) ||² ]
```

```python
def flow_matching_training_step(model, x1, t):
    """Flow matching training step."""
    x0 = torch.randn_like(x1)
    
    # Straight-line interpolation
    x_t = (1 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * x1
    
    # Target velocity (constant along path)
    target_velocity = x1 - x0
    
    # Predict velocity
    v_pred = model(x_t, t)
    
    return F.mse_loss(v_pred, target_velocity)
```

### 7.4 Advantages of Flow Matching

| Property | DDPM | Flow Matching |
|---|---|---|
| Training | Noise prediction loss | Velocity prediction loss |
| Path | Curved (SDE) | Straight line (ODE) |
| NFE at inference | 50-1000 | 20-50 (fewer due to straight paths) |
| Invertibility | Not exact | Exact (ODE is reversible) |
| Encoder | Not available | Can encode data to noise exactly |
| Models using it | SD 1.x, SDXL | SD3, FLUX, Sora, Lumina |

The straight-line paths mean the ODE solver needs fewer steps to stay on the trajectory — directly translating to faster inference.

---

## 8. Architecture: UNet vs DiT

### 8.1 UNet (DDPM Era)

Original diffusion models used a U-Net with residual blocks, skip connections, and attention:
```
Encoder: [Conv → ResBlock → Attn] × n  (downsampling)
Bottleneck: [ResBlock → Attn] × m
Decoder: [ResBlock → Attn → ConvTranspose] × n  (upsampling, skip connections)
```

Timestep t and conditioning c are injected via additive sinusoidal embeddings.

### 8.2 Diffusion Transformer (DiT)

**Peebles & Xie (2022)** — replace UNet with a Vision Transformer:

1. Patchify latent z into patches → linear projection → token sequence
2. Append timestep embedding and conditioning as extra tokens
3. N transformer blocks with full self-attention
4. Unpatchify tokens → latent

```
z ∈ R^{H/8 × W/8 × C}
→ patches of size p×p: (H/8p × W/8p) tokens of dim p²C
→ transformer (N layers, self-attention)
→ unpatch → z
```

**adaLN-Zero**: modulate LayerNorm scale/shift parameters using the conditioning signal (class label, timestep, text) rather than cross-attention. More efficient at incorporating conditioning.

DiT demonstrates the **scaling laws** of diffusion: larger DiT (XL vs B vs L) with same compute budget → better FID. This motivated Sora's 3D DiT.

### 8.3 Sora's 3D DiT

Sora (OpenAI, 2024) extends DiT to video:
- Input: video patches (H/P × W/P × T_frames) → flattened token sequence
- Each patch is a spatiotemporal cube: height × width × time
- Full 3D self-attention across all spatiotemporal patches
- Conditioning: text (via CLIP/T5), duration, resolution, aspect ratio

Key insight: treating video as a sequence of spatiotemporal patches allows variable-length, variable-resolution video generation in a single unified model — no separate temporal or spatial attention stages.

---

## 9. Text-to-Image Model Landscape

| Model | Architecture | Training Objective | Release | Notes |
|---|---|---|---|---|
| DALL-E 2 | CLIP prior + DDPM UNet | Noise prediction | 2022 | Two-stage: image prior + diffusion |
| Stable Diffusion 1.x | LDM + UNet | Noise prediction (DDPM) | 2022 | Open weights, 860M params |
| Midjourney v5 | LDM (proprietary) | Unknown | 2023 | Best aesthetic quality at launch |
| SDXL | LDM + larger UNet | Noise prediction | 2023 | 2.6B params, 2-stage |
| DALL-E 3 | LDM + DiT (?) | Noise prediction | 2023 | Significantly improved text following |
| Stable Diffusion 3 | LDM + DiT | Flow matching | 2024 | MM-DiT: separate streams for image/text |
| FLUX.1 | LDM + DiT | Flow matching (rectified) | 2024 | Best open text-following, 12B params |
| Sora | 3D DiT | Flow matching | 2024 | Video generation, variable duration |
| Imagen 3 | Pixel-space cascade | Score distillation | 2024 | Google, superior text rendering |

---

## 10. Consistency Models and Distillation

**Song et al. (2023)** — train models that produce high-quality samples in 1-4 steps.

### 10.1 The Problem

DDPM/DDIM require 20-1000 NFE. Diffusion models are too slow for real-time applications.

### 10.2 Consistency Models

**Key insight**: along any trajectory {x_t}_{t∈[0,T]} generated by the probability flow ODE, all points map to the same x₀. A **consistency function** f_θ(x_t, t) should satisfy:
```
f_θ(x_t, t) = f_θ(x_s, s) = x₀   for all t, s on the same trajectory
```

Training: enforce self-consistency between adjacent points on the trajectory:
```
L_CM = E [ d(f_θ(x_{t+1}, t+1), f_θ̄(x_t, t)) ]
```
where d is a perceptual distance and θ̄ is an exponential moving average (teacher).

**Consistency distillation**: distill a pre-trained diffusion model into a consistency model. Achieves 1-4 step generation with near-diffusion quality.

### 10.3 Diffusion Distillation Methods

| Method | Steps | Quality | Notes |
|---|---|---|---|
| DDIM | 20-50 | Good | No extra training |
| PNDM | 20 | Good | Higher-order ODE solver |
| DPM-Solver | 10-20 | Very good | Analytical approximation |
| Consistency Models | 1-4 | Good-Very good | Self-consistency training |
| Latent Consistency Models (LCM) | 4 | Good | Consistency distillation in latent space |
| Score Distillation Sampling (SDS) | 1 | OK | Used in text-to-3D (DreamFusion) |
| Progressive Distillation | 4 | Very good | Halve steps iteratively |

---

## 11. Applications Beyond Images

### 11.1 Audio Generation

**AudioLDM / Stable Audio**: run latent diffusion on mel-spectrograms or latent audio representations (EnCodec/DAC codec).

**MusicGen (Meta)**: autoregressive over codec tokens, not diffusion.

**Voicebox (Meta, 2023)**: flow matching for speech synthesis — condition on phoneme sequence and surrounding audio context.

### 11.2 Video Generation

**Stable Video Diffusion**: fine-tune image diffusion model by adding temporal attention layers. Generates 14-25 frames conditioned on a single image.

**Sora**: full 3D DiT with flow matching. Generates coherent long videos (up to 60s) with consistent physics, lighting, and camera motion. OpenAI frames Sora as an implicit world model — it must have learned physical constraints to generate plausible videos.

### 11.3 3D Generation

**DreamFusion (Poole et al., 2022)**: use a 2D diffusion model as a prior to optimize a NeRF. Score Distillation Sampling (SDS) loss:
```
∇_θ L_SDS = E_{t,ε} [ w(t) (ε_φ(x_t; y, t) - ε) ∂x/∂θ ]
```
Backpropagates diffusion score gradients through the 3D representation.

**3D Gaussian Splatting + Diffusion**: more recent approaches generate explicit 3D Gaussians directly via diffusion (GaussianDreamer, etc.).

### 11.4 Scientific Applications

**AlphaFold 3 (Abramson et al., 2024)**: uses a diffusion module on top of learned 3D coordinate representations for protein/ligand structure prediction (see frontier-ai-developments-2025.md Section 7).

**RFDiffusion**: diffusion model for protein backbone generation. Generates novel protein backbones conditioned on functional constraints (binding site, secondary structure). Used for de novo protein design.

---

## 12. DDPM vs GAN vs VAE

| Property | VAE | GAN | Diffusion |
|---|---|---|---|
| Training stability | Stable | Mode collapse common | Stable |
| Sample quality | Blurry | Sharp (when trained) | State of the art |
| Sample diversity | Good | Mode dropping | Excellent |
| Sampling speed | Fast (one pass) | Fast (one pass) | Slow (many steps) |
| Latent interpolation | Smooth | Smooth (W-space) | DDIM gives deterministic codes |
| Likelihood evaluation | Approximate (ELBO) | Not available | ELBO (expensive) |
| Controllability | Limited | GAN inversion | Excellent (CFG, SDEdit) |
| Training compute | Low | Medium | High |

---

## Canonical Interview Q&As

**Q: Derive the DDPM training objective. Why do we predict noise instead of x₀?**

A: The DDPM loss is a variational lower bound on log p(x₀). Expanding the ELBO for T steps and discarding terms that don't depend on θ, we get a weighted sum of terms:
```
L = Σ_t E [ || μ̃_t(x_t, x₀) - μ_θ(x_t, t) ||² ]
```
where μ̃_t is the posterior mean of the reverse process given x₀. Substituting the closed-form expression for μ̃_t and reparameterizing, the model can equivalently be trained to predict x₀ directly OR the noise ε. Ho et al. found empirically that **noise prediction (ε-parameterization) works better** because: (1) the target ε ~ N(0,I) is always well-scaled regardless of t; (2) predicting x₀ directly at high noise levels where ᾱ_t ≈ 0 is ill-conditioned because sqrt(ᾱ_t) ≈ 0 amplifies prediction errors; (3) noise prediction naturally connects to score matching (s_θ = -ε_θ/sqrt(1-ᾱ_t)).

**Q: What is classifier-free guidance and why does it work?**

A: CFG trains a single model on both conditional and unconditional inputs (null condition applied with probability p_uncond ≈ 0.1 during training). At inference, the guided prediction is:
```
ε̃ = ε_uncond + γ (ε_cond - ε_uncond)
```
This can be interpreted as Bayes' theorem: ε̃ implicitly represents sampling from p(x|c)^γ / Z — a sharpened version of the conditional distribution. Higher γ → more peaked distribution → higher fidelity to the condition but lower diversity. The guidance scale γ=7.5 is a hyperparameter found empirically; γ=1 is the unmodified conditional distribution; γ=0 is unconditional. CFG is preferred over classifier guidance because it requires no separate classifier and works with any conditioning type (text, class, image, etc.).

**Q: What is the relationship between DDPM and score-based models?**

A: They are mathematically equivalent. DDPM's noise prediction ε_θ(x_t, t) relates to the score function by:
```
∇_{x_t} log q(x_t) ≈ -ε_θ(x_t, t) / sqrt(1 - ᾱ_t)
```
Both learn to estimate the gradient of the log density at each noise level. Song's continuous-time SDE framework unifies them: the DDPM forward process is a discretized version of the SDE `dx = f(x,t)dt + g(t)dW`, and the learned reverse process corresponds to the reverse-time SDE. DDIM sampling corresponds to solving the probability flow ODE (no stochastic term), which is a special case of the continuous framework.

**Q: How does flow matching differ from diffusion models, and why is it becoming preferred?**

A: DDPM defines a curved SDE trajectory from noise to data; the learned score function guides denoising along this curved path, requiring many small steps. Flow matching instead defines **straight-line** paths: x_t = (1-t)x₀ + tx₁, and trains the model to predict the constant velocity v = x₁ - x₀ at each point. Advantages: (1) straight paths can be traversed in fewer ODE solver steps (~20 vs ~1000 for DDPM); (2) the training objective is simpler — no noise schedule design; (3) exact invertibility via ODE solvers allows encoding images to noise; (4) empirically superior FID/quality at equal or less compute (FLUX, SD3 both use flow matching and achieve SOTA). The main disadvantage is that naive conditional flow matching has high variance (different pairs (x₀, x₁) can create crossing paths); optimal transport flow matching (Tong et al., 2024) addresses this by pairing x₀ and x₁ to minimize path length.

**Q: Design a text-to-image system at production scale (1000 QPS, P99 < 3s latency).**

A: (1) **Model**: Latent diffusion with SDXL or FLUX — pixel-space models are too slow. Use 20-step DDIM or 4-step LCM for latency. (2) **Inference stack**: batch requests dynamically (batch size 4-8). Use PyTorch compile + flash attention for UNet/DiT. Mixed precision (fp16/bf16). On A100 80GB: SDXL in ~500ms at batch 8 = ~16 images/sec/GPU. For 1000 QPS: 1000/16 ≈ 63 GPUs needed. (3) **Safety**: run NSFW classifier on generated outputs before serving. Use prompt content moderation to block harmful inputs. (4) **Serving**: async generation queue (Redis/Kafka), GPU workers pull from queue, push results to object store (S3). Client polls or uses websockets. (5) **Cost optimization**: spot instances for batch workloads, on-demand for interactive. Cache embeddings for repeated prompts. Use smaller SDXL base only for high-latency/low-cost tier.
