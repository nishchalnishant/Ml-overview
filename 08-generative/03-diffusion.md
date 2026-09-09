---
module: Generative Models
topic: Diffusion
subtopic: DDPM / DDIM
status: unread
tags: [generative, diffusion, ddpm, ddim, latent-diffusion, score-matching, flow-matching, classifier-free-guidance, dit, interview-framing]
prerequisites: [probability, neural-networks, gan, autoencoders]
---
# Diffusion Models

### What the interviewer is actually testing
Whether you understand the closed-form forward process and why it enables efficient training. The competency: explain why "predict the noise" produces stable training that GANs could never achieve, and what DDIM does differently from DDPM.

### The reasoning structure

The key insight that makes diffusion models tractable: the forward process (adding Gaussian noise over $T$ steps) has a closed form. You don't need to simulate 1000 steps to get the noisified version of $x_0$ at timestep $t$ — you can compute it in one shot:

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\right), \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

This is what enables training at scale: for any batch element, sample a random timestep $t$, corrupt $x_0$ by the right amount in one operation, and train the denoiser.

### The pattern in action

**Simplified training objective (Ho et al.):** predict the noise that was added:
$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\; t)\|^2\right]$$

The network $\epsilon_\theta$ takes a noisy image and a timestep, and outputs the predicted noise. This is regression — not a minimax game, not a discriminator to balance, not an intractable likelihood. It trains stably with standard optimization. Mode collapse has no mechanism to occur: each training step is an independent regression on a random noise level.

**Why diffusion beat GANs structurally:**
1. Stable training — regression loss, no adversarial dynamics
2. Better sample diversity — no mode collapse mechanism
3. Classifier-free guidance works reliably — scale guidance weight to trade diversity for fidelity
4. Better coverage of complex distributions at large scale

**Inference DDPM vs DDIM:**
DDPM: stochastic reverse process, requires all $T$ steps (typically 1000).
DDIM: reframes the reverse process as an ODE (deterministic). The same trained model; a different sampling scheme. Allows 10-50 steps with minimal quality loss because the ODE solver can take larger steps.

**The current state of the speed gap:**
DDIM: 20-50 steps. Consistency models: 1-4 steps. Latent diffusion (Stable Diffusion): denoises in a compressed latent space (64×64 instead of 512×512), reducing per-step cost by ~64×. The single-step advantage of GANs has been largely closed.

### Common traps

**Trap: saying diffusion models are slow without qualification.**
The 1000-step requirement was DDPM-specific. DDIM showed the trained model works with far fewer steps. The framing to use in an interview: "Inference speed was the limitation, and it's been addressed through ODE-based sampling (DDIM), latent space diffusion, and consistency models. The remaining tradeoff is one-step generation quality vs multi-step generation quality — not whether multi-step is required at all."

### The unifying view: score matching and flow matching

Three framings describe the same object; interviewers use them interchangeably.

**Denoising = score estimation.** The score is the gradient of the log-density, $\nabla_x \log p(x)$ — it points toward higher-probability regions. The noise-prediction network is a rescaled score estimator:

$$\nabla_{x_t} \log p(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

So "predict the noise" and "learn the score" are the same training objective. This matters because it explains *why* sampling works: the reverse process is Langevin-style ascent on a learned score field. Song & Ermon's score-based SDE framing showed DDPM and score matching are discretizations of one continuous-time SDE, and that every SDE has a corresponding **probability-flow ODE** with identical marginals — which is exactly what DDIM solves.

**Flow matching** (the framing behind Stable Diffusion 3 and most recent systems) generalizes further: instead of a noising process, pick *any* path between noise and data and regress the velocity field along it. Rectified flow takes the straight-line path $x_t = (1-t)x_0 + t\epsilon$, and the training target becomes the constant velocity $\epsilon - x_0$. Straight paths are the point — a straight trajectory is exactly what an ODE solver integrates accurately in few steps, so few-step sampling comes from the *path design* rather than from a post-hoc distillation.

**Why the field moved this way:** DDPM's variance schedule is one arbitrary choice of path. Flow matching makes the path a design variable, and straighter paths mean fewer function evaluations.

### The likelihood connection

Diffusion is a hierarchical VAE with a fixed, non-learned encoder. The forward noising process is the encoder; the latents $x_1 \dots x_T$ have the same dimensionality as the data; the denoiser is the decoder. The training loss is the ELBO, and $\mathcal{L}_\text{simple}$ is that ELBO with the per-timestep weighting dropped — a reweighting that trades likelihood for sample quality.

This is a common interview thread: *"is diffusion a latent variable model?"* Yes — and the reason it avoids the VAE's blurriness is that the encoder is fixed, so there is no posterior-collapse mechanism and no encoder/decoder co-adaptation.

### Conditioning mechanics

Worth knowing concretely, because production questions land here:

- **Classifier guidance** required a separate classifier trained on noisy images. Effectively dead.
- **Classifier-free guidance (CFG)** trains one model that sees the condition dropped ~10% of the time, then extrapolates at inference: $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w\,(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$. Costs **two forward passes per step** — the single biggest inference-cost line item, and the first thing to interrogate in a latency budget.
- **Cross-attention** injects text embeddings at multiple U-Net (or DiT) depths — the mechanism through which prompts actually reach the image.
- **ControlNet / adapters** add spatial conditioning (pose, depth, edges) by cloning encoder blocks with zero-initialized connections, leaving base weights frozen. Structurally the same trick as LoRA's zero-init: the model starts as an exact copy of the original and learns a delta.

### Architecture: U-Net → DiT

Early diffusion used a convolutional U-Net with attention at low resolutions. Current systems use **Diffusion Transformers (DiT)** — patchify the latent, run a plain transformer, condition via adaLN-zero. The motivation is scaling behavior: DiT follows clean scaling laws where U-Nets plateau, and it lets diffusion inherit the entire transformer infrastructure stack (FlashAttention, sequence parallelism, established scaling recipes). Sora and SD3 are both DiT-based.

---

## Interview Angles

### Q: Why is the closed-form forward process the thing that makes diffusion trainable? [Medium]

Because it decouples training cost from $T$. To train at timestep $t$ you would otherwise have
to simulate $t$ sequential noising steps. The closed form
$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ lets you jump
straight there in one operation, so each training step is O(1) regardless of $T$.

**Cross-questions to expect**
- *"Does the reverse process have a closed form too?"* — No. That asymmetry is the whole
  reason inference is slow while training is cheap.
- *"Why Gaussian noise specifically?"* — Gaussians are closed under convolution, which is
  exactly what gives the closed form.

**Trap:** Explaining the closed form as an inference optimization. It is a *training*
optimization; inference still walks the steps.

### Q: DDPM vs. DDIM — same model or different? [Medium]

Same trained model, different sampling scheme. DDIM reframes the reverse process as a
deterministic ODE, so a solver can take larger steps — 20–50 instead of 1000, with minimal
quality loss.

**Cross-questions to expect**
- *"So you can swap the sampler after training?"* — Yes, and that is why it landed so fast.
- *"What does determinism buy you beyond speed?"* — A reproducible noise→image map, which
  makes latent interpolation and editing coherent.

**Trap:** Claiming DDIM retrains or distills the model. It does neither.

### Q: Your text-to-image model produces beautiful images that ignore half the prompt. [Hard]

Most likely the classifier-free guidance weight is too low. Guidance scales the difference
between conditional and unconditional predictions; at low weight the model drifts toward the
unconditional distribution — plausible images, weak prompt adherence.

**Cross-questions to expect**
- *"So raise it?"* — Up to a point. Too high oversaturates and collapses diversity. It is a
  fidelity/diversity dial, not a quality dial.
- *"What if raising it doesn't fix adherence?"* — Then it's the text encoder or
  cross-attention conditioning, not the sampler. Check whether the failure is compositional
  (counting, spatial relations) — those are known encoder-side failures guidance cannot fix.

**Trap:** Jumping to "fine-tune on more data." Guidance weight is an inference-time parameter.
Test it before spending a training run.

### Q: Explain diffusion as score matching. [Hard]

The noise-prediction net is a rescaled estimate of $\nabla_x \log p(x_t)$, the score. Sampling
is then gradient ascent on a learned log-density, with noise injected to keep it from collapsing
onto modes. DDPM and score-based SDEs are two discretizations of the same continuous-time process.

**Cross-questions to expect**
- *"Why does the score avoid the normalizing constant?"* — $\nabla_x \log p(x)$ kills any
  term without $x$ in it, and $Z$ is constant in $x$. That is the whole appeal: you learn the
  shape of the density without ever computing $Z$.
- *"Where does DDIM fit?"* — It is the probability-flow ODE of that SDE: same marginals, no
  injected noise, hence deterministic and few-step.

**Trap:** Treating score matching as a different model family. It is a different derivation of
the same trained network.

### Q: Your text-to-image service is too slow. Walk through the cost. [Hard]

Per image: (steps) × (2 forward passes, if CFG is on) × (per-pass cost). Attack in that order.

1. **CFG is a 2× multiplier** — guidance distillation folds it into one pass. Usually the
   largest single win, and often overlooked.
2. **Step count** — flow-matching/rectified-flow models and distilled samplers (LCM, Turbo)
   reach 1-8 steps. Quality degrades in fine texture and text rendering first.
3. **Latent resolution** — SD's VAE gives an 8× spatial downsample, so ~64× fewer pixels per step.
4. **Per-pass cost** — quantization, FlashAttention, batching. Diffusion batches well because
   every image in a batch runs identical step counts, unlike autoregressive decoding.

**Cross-questions to expect**
- *"Which do you try first?"* — Whichever is free. Sampler and step count are config changes;
  distillation is a training run.
- *"How do you know quality held?"* — Not FID alone at low step counts. Human preference or a
  paired-prompt eval; FID is insensitive to prompt adherence.

**Trap:** Jumping straight to a smaller model. The step-count and CFG multipliers are usually
worth more and cost nothing to test.

---

## Connections

- [02-gans.md](02-gans.md) — what diffusion displaced, and the one regime where it didn't
- [01-autoencoders.md](01-autoencoders.md) — latent diffusion denoises in an autoencoder's latent space
- [../06-architectures/04-dl-architectures.md](../06-architectures/04-dl-architectures.md) — architecture selection framework
