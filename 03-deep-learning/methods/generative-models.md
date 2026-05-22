# Generative Models

> Covers VAEs, GANs, Diffusion Models (DDPM/DDIM), Latent Diffusion, Flow Matching, and the DiT/FLUX architecture wave.

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

**The problem:** A standard autoencoder encodes `x → z → x̂`. It reconstructs well, but the latent space `z` is unstructured — random points in it decode into garbage. You can't sample from the autoencoder. The fundamental question: how do you train an encoder–decoder pair such that sampling a random `z` produces a valid output?

**The core insight:** Force the encoder to map each input to a *distribution* over z (specifically, a Gaussian), not a single point. Then sample z from that distribution to decode. The distribution is regularized toward a known prior `N(0, I)`. If the regularization succeeds, any sample from `N(0, I)` decodes to a valid output.

**The mechanics:** We want to maximize `log p_θ(x) = log ∫ p_θ(x|z) p(z) dz`, but this integral is intractable. Introduce encoder `q_φ(z|x)` as an approximate posterior. Via Jensen's inequality:

```
log p_θ(x) ≥ E_{z~q_φ(z|x)} [log p_θ(x|z)] − KL(q_φ(z|x) ∥ p(z))
              ────────────────────────────────   ──────────────────────
                   reconstruction term              regularization term
```

This lower bound is the **ELBO**. Maximize it jointly over encoder (φ) and decoder (θ).

With Gaussian encoder `q_φ(z|x) = N(μ, σ²I)` and prior `p(z) = N(0, I)`, the KL is analytic:

```
KL = ½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)
```

Sampling `z ~ q_φ(z|x)` blocks gradients. The **reparameterization trick** restores them:

```
z = μ_φ(x) + σ_φ(x) ⊙ ε,   ε ~ N(0, I)
```

z is now a deterministic function of (x, ε); gradients flow through μ and σ.

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl
```

**What breaks:**
- **Posterior collapse:** the encoder ignores the input and collapses to the prior; the decoder then ignores z entirely. Occurs when the decoder is too powerful or KL weight β is too high.
- **Blurry reconstructions:** the reconstruction term under a Gaussian decoder is MSE, which averages over modes — sharp details are penalized unless the KL pressure forces the latent to carry them.
- **β-VAE trade-off:** setting β > 1 increases disentanglement (each z dimension encodes one independent factor) at the cost of reconstruction fidelity. There is no free lunch between the two terms.

---

## 2. Generative Adversarial Networks (GANs)

**The problem:** If you want a network to generate realistic images, what's the loss function? MSE against real images produces blurry averages. You need a signal that says "this sample looks unrealistic" — but realism has no closed-form definition.

**The core insight:** Train a second network (the discriminator) to be the judge of realism. The generator improves by fooling the discriminator; the discriminator improves by detecting fakes. This adversarial game, at equilibrium, drives the generator to produce samples indistinguishable from real data.

**The mechanics:**

```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 − D(G(z)))]
```

D maximizes the probability of correctly labeling real (x) vs fake (G(z)). G minimizes the probability of D detecting its outputs. At Nash equilibrium, `p_G = p_data` and `D(x) = 0.5` everywhere.

**What breaks:**

- **Vanishing gradients:** when D is too strong, `log(1 − D(G(z))) ≈ 0` — G receives no gradient. Fix: train G to maximize `log D(G(z))` (non-saturating objective) rather than minimizing `log(1 − D(G(z)))`.
- **Mode collapse:** G learns a small subset of high-scoring outputs and ignores the rest of the data distribution. High quality but zero diversity.
- **Training instability:** the minimax game has no unique convergence path; it oscillates or diverges under most hyperparameter settings.
- **No density estimate:** you can sample from G but cannot evaluate `p(x)` for a given x.

### WGAN: replacing the divergence

The original GAN objective minimizes JS divergence, which saturates when `p_G` and `p_data` have disjoint supports (the common case early in training). Replace it with Wasserstein-1 (Earth Mover's distance), which has useful gradients even when supports don't overlap:

```
W(p_r, p_g) = sup_{‖f‖_L ≤ 1} E_{x~p_r}[f(x)] − E_{x~p_g}[f(x)]
```

The critic f must be 1-Lipschitz. WGAN-GP enforces this with a gradient penalty:

```
L = E[D(x̃)] − E[D(x)] + λ E[(‖∇_x̂ D(x̂)‖_2 − 1)²]
```

where x̂ is a random linear interpolation between real and fake. The loss curve now has a meaningful magnitude — a practical diagnostic for training progress.

### StyleGAN2 architecture

StyleGAN2 (Karras et al., 2020) achieves state-of-the-art GAN fidelity via:
- **Mapping network:** z → w (8-layer MLP); moves from spherical Gaussian to a more disentangled style space.
- **AdaIN style injection:** at each synthesis layer, apply affine transforms of w to control feature statistics.
- **Weight demodulation:** replaces instance normalization to remove characteristic droplet artifacts.

GANs are largely superseded by diffusion models for image generation but remain relevant where single-forward-pass inference speed is required (video, real-time).

---

## 3. Diffusion Models — DDPM

**The problem:** Both VAEs and GANs approximate the data distribution indirectly — VAEs through a variational bound, GANs through an adversarial game. Neither produces a stable, principled training objective that scales cleanly. Is there a way to train a generative model with a simple, stable loss?

**The core insight:** Take a data sample and *gradually* add noise until it becomes pure Gaussian noise. This forward process is fixed and analytically tractable. Then train a network to *reverse* each small denoising step. Because each reverse step is a small local move, the neural network only needs to predict what noise was added — a regression problem with MSE loss.

**The mechanics (DDPM, Ho et al. 2020):**

Forward process — add Gaussian noise over T steps. Key closed-form: you can jump directly from x_0 to any x_t:

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1−ᾱ_t)I)

αt = 1 − βt,   ᾱ_t = ∏_{s=1}^t αs

x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ N(0, I)
```

The β schedule is chosen so that x_T ≈ N(0, I).

Train ε_θ(x_t, t) to predict the noise ε added to x_0 to produce x_t:

```
L_simple = E_{t, x_0, ε} [‖ε − ε_θ(x_t, t)‖²]
```

This is equivalent to score matching: the noise predictor is proportional to the score `∇_x log p(x_t)`:

```
s_θ(x_t, t) = −ε_θ(x_t, t) / √(1−ᾱ_t)
```

The noise schedule matters: a cosine schedule (Nichol & Dhariwal 2021) avoids over-destroying information in early steps, improving sample quality over the linear schedule.

**What breaks:**

- **Slow inference:** the reverse process requires T=1000 sequential network evaluations per sample — hundreds of forward passes for one image.
- **No explicit latent structure:** unlike VAEs, there is no interpretable compressed representation; the "latent" is the full-resolution noisy image at step T.
- **Exposure bias:** the network is trained with ground-truth x_t but at inference uses its own predictions as inputs — accumulated errors compound over 1000 steps.

---

## 4. Faster Sampling — DDIM & Schedulers

**The problem:** DDPM inference requires T=1000 sequential denoising steps, each a full U-Net forward pass. At 512×512 resolution this takes 30+ seconds per image. The Markovian reverse process is the bottleneck — each step depends on the previous one, so you can't parallelize.

**The core insight:** The DDPM training objective doesn't require the reverse process to be Markovian. You can derive a *non-Markovian* reverse process that shares the same marginal distributions as DDPM, allowing arbitrary step skipping, without retraining the model.

**The mechanics (DDIM, Song et al. 2020):**

```
x_{t-1} = √ᾱ_{t-1} · (x_t − √(1−ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t
         + √(1−ᾱ_{t-1} − σ_t²) · ε_θ(x_t, t)
         + σ_t · ε_t

σ_t = η · √((1−ᾱ_{t-1})/(1−ᾱ_t)) · √(1−ᾱ_t/ᾱ_{t-1})
```

When η=0: **deterministic DDIM** — the mapping from latent to image is a deterministic ODE. Same initial noise always produces the same image. Can reduce from T=1000 to 20–50 steps with acceptable quality. Enables latent space interpolation.

**What breaks:**

- **Quality degrades sharply below ~20 DDIM steps** because the ODE approximation accumulates error with large step sizes. Higher-order ODE solvers (DPM-Solver++) recover some quality.
- **Determinism is a double edge:** reproducibility is useful, but all diversity comes from the initial noise — the model cannot inject stochasticity mid-trajectory to recover from early errors.

### Common schedulers

| Scheduler | Steps | Notes |
|-----------|-------|-------|
| DDPM | 1000 | Maximum quality, impractical for production |
| DDIM | 20–50 | Deterministic, invertible, good quality |
| DPM-Solver++ | 10–20 | Best quality/speed ratio in practice |
| LCM | 4–8 | Requires LCM distillation fine-tuning |
| SDXL-Turbo / ADD | 1–4 | Adversarial diffusion distillation |

---

## 5. Classifier-Free Guidance (CFG)

**The problem:** You want to generate images conditioned on a text prompt. One approach uses a separate classifier `p(y|x_t)` to steer the denoising process. But training a robust classifier on noisy images at every noise level is costly, and the classifier may not generalize to all prompt types.

**The core insight:** The same diffusion model can serve as its own classifier-free steering mechanism. Train it jointly on conditional and unconditional generation (randomly drop the conditioning during training). At inference, amplify the difference between conditional and unconditional predictions to push samples toward the conditioning signal — no external classifier needed.

**The mechanics:**

Training: with probability p_uncond, replace the condition y with a null token ∅. The model learns both `ε_θ(x_t, t, y)` and `ε_θ(x_t, t, ∅)` from a single set of weights.

Inference:

```
ε_guided = ε_θ(x_t, t, ∅) + w · (ε_θ(x_t, t, y) − ε_θ(x_t, t, ∅))
```

w is the guidance scale (typically 7.5 for Stable Diffusion). Geometrically: take the unconditional prediction and add w times the "direction toward the conditioning signal."

```python
@torch.no_grad()
def p_sample_cfg(model, x_t, t, condition, guidance_scale=7.5):
    x_in = torch.cat([x_t, x_t])
    cond_in = torch.cat([condition, empty_condition])
    noise_pred = model(x_in, torch.cat([t, t]), cond_in)
    noise_cond, noise_uncond = noise_pred.chunk(2)
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    return denoise_step(x_t, t, noise_guided)
```

**What breaks:**

- **Manifold drift:** high guidance scale pushes samples toward regions of high conditional likelihood that may be out-of-distribution for the denoiser — oversaturated colors, distorted anatomy.
- **Diversity collapse:** at w >> 1, the denoising process converges to a few high-probability modes of p(x|y), sacrificing sample diversity.
- **Inference cost:** requires two forward passes per step (conditional + unconditional). FLUX-schnell solves this via guidance distillation — a single unconditional model trained to match CFG output.

---

## 6. Latent Diffusion Models (Stable Diffusion)

**The problem:** Running diffusion directly in pixel space at 512×512×3 means every denoising step operates on a 786K-dimensional tensor. With 1000 steps and a large U-Net, this is computationally prohibitive for training and inference. Can we get the benefits of diffusion without the pixel-space cost?

**The core insight:** Most of the perceptual content of an image lives in a much lower-dimensional space. Train a VAE to compress images into a small latent space; then run diffusion in *that* space. The VAE is a fixed, pre-trained codec — diffusion only ever sees compact latents.

**The mechanics:**

```
VAE encoder:  x (512×512×3) → z (64×64×4)   [compression: ~48×]
Diffusion:    operates on z
VAE decoder:  z → x̂ (512×512×3)
```

The U-Net denoiser adds **cross-attention layers** that condition on text:

```
Cross-Attention(Q=z_features, K=text_emb, V=text_emb)
```

Text is encoded by a frozen CLIP text encoder into a 77×768 sequence. At each resolution level, spatial features attend to all 77 text tokens simultaneously.

**What breaks:**

- **VAE bottleneck quality:** any detail the VAE discards during encoding cannot be recovered by diffusion. The VAE's reconstruction quality sets a hard ceiling on output fidelity.
- **KL regularization vs. reconstruction:** the KL penalty needed to make latents well-behaved for diffusion also forces the VAE to lose some high-frequency detail. SDXL uses a larger channel count (4→16) to partially mitigate this.
- **Cross-attention alignment:** CLIP text embeddings may not capture all semantic detail in a long or complex prompt. SDXL concatenates two CLIP encoders (ViT-L and OpenCLIP ViT-G) to increase text representational capacity.

---

## 7. Flow Matching & Rectified Flow

**The problem:** DDPM defines curved paths through noise space — the reverse process follows a nonlinear SDE. Integrating a curved path requires many small steps to stay on the manifold. Can we define a generative process with straighter paths that require fewer integration steps?

**The core insight:** Define the flow from noise to data as a *straight line*. A straight-line path has constant velocity, so the ODE integrator never accumulates curvature error — you can take very few steps and still land close to the data manifold.

**The mechanics (Rectified Flow, Liu et al. 2022):**

For each training sample, pair a data point x_1 ~ p_data with a noise sample x_0 ~ N(0, I). The straight-line interpolation is:

```
x_t = (1−t) · x_1 + t · x_0,   t ∈ [0, 1]
```

The velocity at every point along this line is constant:

```
dx/dt = x_0 − x_1
```

Train v_θ(x_t, t) to predict this velocity:

```
L = E_{x_0~N(0,I), x_1~p_data, t~U[0,1]} [‖v_θ(x_t, t) − (x_0 − x_1)‖²]
```

Inference: start from x_0 ~ N(0, I), integrate forward from t=1 to t=0 using an ODE solver. Straight paths mean 5–10 steps are typically sufficient.

**What breaks:**

- **Straight paths in expectation, not per-sample:** the optimal transport pairing of noise and data is not straight lines between arbitrary pairs. If you pair x_0 and x_1 randomly, the learned velocity field averages over many crossing trajectories — it becomes curved in practice. Reflow (iterating the procedure) straightens the paths further.
- **ODE solver sensitivity:** with so few steps, step size matters more than in diffusion; an inappropriate step size or solver can produce visible artifacts.

---

## 8. Diffusion Transformers (DiT) & FLUX

**The problem:** U-Nets are the default backbone for diffusion models, but U-Nets have fixed spatial inductive biases (skip connections, hierarchical structure). They don't scale as predictably as transformers, and their receptive fields at high resolutions are limited. Can the scaling laws from language transformers transfer to image generation?

**The core insight:** Patch the latent image into tokens, then apply a plain transformer. Condition on time and class via adaptive layer norm (adaLN), which modulates features without requiring extra cross-attention. The transformer's global self-attention handles long-range spatial dependencies that U-Nets miss.

**The mechanics (DiT, Peebles & Xie 2022):**

```
Latent z (64×64×4) → patch tokens (8×8 patches → 64 tokens)
→ L transformer blocks with adaLN conditioning
→ linear head → predicted noise

adaLN(x, c) = γ(c) · LayerNorm(x) + β(c)
```

γ and β are linear projections of a conditioning vector c (concatenation of time embedding + class embedding). This injects conditioning at every layer without additional attention heads.

**What breaks:**

- **No spatial inductive bias:** unlike U-Nets, transformers have no built-in notion that nearby pixels are related. This is eventually learned from data, but requires more training to converge on small datasets.
- **Quadratic attention cost:** global self-attention over 64 tokens is fine at 512×512, but scaling to 1024×1024 with smaller patches makes the attention prohibitive. Efficient attention variants are required.

### FLUX (Black Forest Labs, 2024)

FLUX combines rectified flow with a DiT-based architecture. Two key innovations over vanilla DiT:

**Dual-stream architecture:** instead of a single token sequence mixing image and text tokens, FLUX maintains separate transformer streams for image and text that interact via cross-attention and then merge into single-stream blocks. Image and text representations evolve jointly:

```
Image tokens ──┐                    ┌── Image tokens
                ├── Cross-Attention ─┤
Text tokens  ──┘                    └── Text tokens
```

This is richer than Stable Diffusion's cross-attention (which uses frozen text embeddings as keys/values) because text features update in response to image features.

**Guidance distillation:** FLUX-schnell is distilled to produce CFG-quality output in a single forward pass — no double evaluation at inference.

| Variant | Params | Steps | Use case |
|---------|--------|-------|----------|
| FLUX.1-dev | 12B | 20–50 | Research, fine-tuning |
| FLUX.1-schnell | 12B | 1–4 | Production inference |
| FLUX.1-pro | 12B | — | Proprietary API |

---

## 9. Production Considerations

### Inference latency

Bottleneck breakdown for 512×512 SDXL on an A100:
- U-Net forward pass: ~35 ms/step
- VAE decode: ~150 ms (once at the end)
- At 20 steps: 35×20 + 150 = **850 ms** end-to-end

Key optimizations:
1. **Fewer steps:** DPM-Solver++ at 15 steps; LCM at 4–8 steps.
2. **Flash Attention / xFormers:** 20–40% memory reduction + speedup.
3. **INT8 quantization of U-Net:** ~1.5× throughput, minimal quality loss.
4. **`torch.compile`:** ~20% speedup on modern GPUs with static shapes.
5. **VAE tiling:** for large images, tile the decode to stay within VRAM.

### Fine-tuning methods

| Method | VRAM | Data needed | Use case |
|--------|------|-------------|----------|
| DreamBooth | 24 GB+ | 3–20 images | Subject personalization |
| Textual Inversion | 8 GB | 3–20 images | New concept as a token |
| LoRA | 8–12 GB | 100s of images | Style transfer, domain fine-tuning |
| Full fine-tuning | 80 GB+ | 1000s of images | Domain adaptation |

LoRA applies rank decomposition `W ≈ W_0 + BA` to U-Net/DiT attention matrices. At rank r=4, trainable parameters are reduced ~100× vs full fine-tuning.

---

## 10. Interview Questions

**Q: What is the ELBO and why maximize it instead of the true marginal likelihood?**

The marginal `p(x) = ∫ p(x|z) p(z) dz` is intractable — summing over all latent codes requires exponential computation. The ELBO is a tractable lower bound derived by introducing an approximate posterior `q(z|x)` and applying Jensen's inequality. Maximizing ELBO simultaneously trains the encoder to approximate the true posterior and the decoder to reconstruct the input. The gap between log p(x) and ELBO equals `KL(q(z|x) ∥ p(z|x))` — so maximizing ELBO also tightens the bound by pushing q closer to the true posterior.

---

**Q: Why did diffusion models replace GANs for high-quality image generation?**

Three reasons: (1) **Training stability** — diffusion loss is a simple MSE noise prediction objective; GAN training is a minimax game that frequently diverges or collapses. (2) **Mode coverage** — diffusion models learn the full data distribution; GANs are prone to mode collapse. (3) **Predictable scaling** — DiT variants follow transformer scaling laws; more parameters + more compute yields proportionally better results. The main disadvantage of diffusion is slower inference; flow matching and distillation are closing this gap.

---

**Q: What is classifier-free guidance and what does the guidance scale control?**

CFG trains a single model to produce both conditional predictions `ε(x_t, y)` and unconditional predictions `ε(x_t, ∅)` by randomly dropping the conditioning during training. At inference: `ε_guided = ε_uncond + w · (ε_cond − ε_uncond)`. The guidance scale w amplifies the component of the noise prediction that points toward the conditioning signal. Higher w = stronger prompt adherence, lower diversity, and potential manifold drift (the sample is pushed to a high-conditional-likelihood region that may be out-of-distribution for the denoiser — causing oversaturation, distorted anatomy, etc.).

---

**Q: What is the difference between DDPM and DDIM sampling?**

DDPM defines a Markovian reverse process: `p(x_{t-1}|x_t)` depends only on x_t, requiring all T steps. DDIM defines a non-Markovian reverse process that produces the same marginals as DDPM but allows skipping steps — you can step from x_T directly to x_{T-50}. When the stochasticity parameter η=0, DDIM becomes a deterministic ODE: the same initial noise always produces the same image, enabling latent interpolation. In practice, 20–50 DDIM steps match 1000-step DDPM quality.

---

**Q: Why does Stable Diffusion operate in latent space rather than pixel space?**

Diffusion requires hundreds of sequential forward passes per image. Running each pass on a 512×512×3 pixel tensor (786K dimensions) is prohibitive. A pre-trained VAE compresses images to 64×64×4 latents (4K dimensions) — a ~192× reduction. Since the VAE is fixed and acts as a near-lossless codec, diffusion in latent space loses minimal perceptual quality while dramatically reducing compute cost per step.

---

*Last updated: May 2026 | Coverage: VAE through FLUX/DiT*

---

## Canonical Interview Q&As

**Q: Explain the VAE ELBO derivation — what are you actually maximizing?**  
A: We want to maximize log p(x) = log ∫ p(x|z)p(z)dz, which is intractable. The VAE introduces an approximate posterior q(z|x) and derives a lower bound (ELBO) via Jensen's inequality: log p(x) ≥ E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z)). The first term is the reconstruction loss — how well can we reconstruct x from the latent code. The second term is the KL divergence between the learned posterior and the prior — a regularizer that prevents the encoder from mapping every input to a unique non-overlapping point in latent space (which would make it an autoencoder, not a generative model). By making the posterior close to N(0,I), samples from N(0,I) during generation will land in regions the decoder has seen. The reparameterization trick z = μ + σ·ε, ε~N(0,1) moves the randomness out of z, making the sampling step differentiable. The trade-off parameter β (β-VAE) upweights the KL term, producing more disentangled but lower-quality reconstructions.

**Q: What is the fundamental training objective of diffusion models and how is it derived from the VLB?**  
A: Diffusion models define a forward process q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t·I) that gradually adds noise. The reverse process p_θ(x_{t-1}|x_t) is parameterized by a neural network. Training maximizes the VLB (variational lower bound on log p(x)) which, after algebraic simplification (Ho et al. 2020), reduces to: L = E_{t,x_0,ε}[||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²], where ε~N(0,I) and ᾱ_t = Π_{s=1}^t(1-β_s). In words: sample a clean image x_0, sample a noise level t uniformly, add noise to get x_t, and train the network to predict the noise ε. This noise prediction objective is simpler and more stable than predicting x_0 directly. At inference, starting from pure noise x_T~N(0,I), iteratively denoise with the learned reverse process. DDIM (denoising diffusion implicit models) enables ~10-50× faster sampling by solving the deterministic ODE corresponding to the diffusion process, allowing large step sizes.

**Q: Compare GANs and diffusion models — when would you choose each for an image generation task?**  
A: **GANs**: fast inference (single forward pass, ~10ms for 512×512), excellent at sharp textures, but training is unstable (mode collapse, gradient vanishing from the discriminator), requires careful tuning, and has limited diversity. Good for: real-time generation, style transfer, image-to-image translation (pix2pix, CycleGAN), super-resolution. **Diffusion models**: slow inference (50-1000 denoising steps, 1-10s), exceptional diversity and quality (FID on ImageNet < 2 vs GAN FID ~3-5), stable training on large diverse datasets, support for classifier-free guidance enabling text conditioning. Good for: high-quality text-to-image (Stable Diffusion, DALL-E 3), editing workflows, research on diverse generation. **The trend**: diffusion models have largely replaced GANs for high-quality image generation because GANs at scale require prohibitive discriminator architecture engineering. GANs remain competitive for real-time applications (video game texture generation, real-time video). Key metric for comparison: FID (Fréchet Inception Distance) measures distribution distance between generated and real images — lower is better; a good model scores < 5 on ImageNet 256×256.
