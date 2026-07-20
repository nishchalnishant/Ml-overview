---
module: Generative Models
topic: Diffusion
subtopic: DDPM / DDIM
status: unread
tags: [generative, diffusion, ddpm, ddim, latent-diffusion, interview-framing]
prerequisites: [probability, neural-networks, gan]
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

---

## Connections

- [02-gans.md](02-gans.md) — what diffusion displaced, and the one regime where it didn't
- [01-autoencoders.md](01-autoencoders.md) — latent diffusion denoises in an autoencoder's latent space
- [../06-architectures/04-dl-architectures.md](../06-architectures/04-dl-architectures.md) — architecture selection framework
