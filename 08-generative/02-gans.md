---
module: Generative Models
topic: GAN
subtopic: Adversarial Training
status: unread
tags: [generative, gan, wgan, adversarial-training, interview-framing]
prerequisites: [probability, neural-networks, optimization]
---
# Generative Adversarial Networks

### What the interviewer is actually testing
Whether you understand adversarial training dynamics — specifically the instabilities and why they are structurally hard to fix. The competency: framing the min-max game, diagnosing each failure mode, and articulating why diffusion models overcame GAN limitations.

### The reasoning structure

Direct likelihood maximization — $\max \log p_\theta(x)$ — is intractable for complex distributions like images: you can't evaluate $p_\theta(x)$ for a neural generator in closed form. GANs sidestep this entirely by replacing the likelihood criterion with an adversarial one: does a discriminator network think the generated sample is real?

The minimax game:
$$\min_G \max_D \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

At Nash equilibrium: $D(x) = 0.5$ everywhere — discriminator cannot distinguish real from generated. Getting there reliably is the unsolved problem.

### The pattern in action

**Three structural failure modes:**

| Failure | Mechanism | What it looks like | Fix |
| :--- | :--- | :--- | :--- |
| Mode collapse | Generator converges to a small set of outputs that fool the discriminator — diversity disappears | All generated faces look similar | Minibatch discrimination, unrolled GANs, diversity regularization |
| Vanishing gradient | When $D$ is too accurate, $\log(1 - D(G(z))) \approx 0$ — generator gradient approaches zero | Generator stops improving despite discriminator being perfect | Non-saturating loss: maximize $\log D(G(z))$ instead of minimizing $\log(1-D(G(z)))$ |
| Training oscillation | Generator and discriminator cycle — neither converges | Loss curves oscillate without settling | Wasserstein GAN with Lipschitz constraint |

**WGAN:** replaces the classifier discriminator with a critic that measures Earth-mover distance:
$$\mathcal{L}_\text{WGAN} = \mathbb{E}_{x \sim p_\text{data}}[D(x)] - \mathbb{E}_z[D(G(z))]$$

The critic must be 1-Lipschitz (enforced via gradient penalty: $\mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$). This provides meaningful gradients even when the generator and data distributions don't overlap at all — solving the vanishing gradient problem structurally.

### Common traps

**Trap: not knowing why GANs lost to diffusion models.**
The adversarial training instabilities — mode collapse, vanishing gradients, oscillation — were never fully solved despite a decade of effort. The fundamental issue is that the training objective is a minimax game, which has no guarantee of convergence to the optimal solution. Diffusion models train by simple regression (predict the added noise), have no adversarial dynamics, and produce better sample diversity with no mode collapse mechanism. The one remaining advantage of GANs: single-step inference. Diffusion requires 20–1000 denoising steps; a GAN requires one forward pass. This matters for real-time generation.

---

## Interview Angles

### Q: Why does a GAN need two networks when a VAE needs one? [Easy]

Because a GAN has no tractable likelihood to maximize. A VAE optimizes a bound on
$\log p_\theta(x)$ directly. A GAN cannot evaluate $p_\theta(x)$ at all, so it replaces the
likelihood criterion with a *learned* one — the discriminator is the loss function, and it
is trained alongside the generator because a fixed loss function would be too weak.

**Cross-questions to expect**
- *"So the discriminator is the loss function?"* — Yes, and that is the source of the
  instability: the loss surface moves while you descend it.
- *"What happens if the discriminator is perfect from step one?"* — The generator gradient
  vanishes; see the non-saturating loss below.

**Trap:** Saying the discriminator "classifies real vs. fake" and stopping. That's the
mechanism, not the purpose. The purpose is to supply a gradient signal where no likelihood exists.

### Q: Mode collapse — diagnose it and fix it. [Medium]

The generator finds a small set of outputs that reliably fool the current discriminator and
stops exploring. Diversity collapses while the loss looks fine.

**Cross-questions to expect**
- *"Why doesn't the discriminator just learn to reject those?"* — It does, and the generator
  hops to a different small set. That's the oscillation failure, not a fix.
- *"How would you detect this in production?"* — Not from the loss curves. Track sample
  diversity directly.

**Trap:** Proposing "train the discriminator less" as the fix. That trades mode collapse for
vanishing gradients. WGAN-GP addresses the cause structurally.

### Q: You are asked to ship real-time avatar generation at 60fps. GAN or diffusion? [Hard]

GAN — this is the one regime where GANs still win. A GAN is a single forward pass. Diffusion
needs 20–1000 denoising steps, and even consistency models at 1–4 steps carry a larger network.

**Cross-questions to expect**
- *"What do you give up?"* — Training stability and sample diversity. Budget engineering time
  for the instabilities, not just the training run.
- *"Would you reconsider if the frame budget were 200ms?"* — Yes. Latent diffusion plus a
  distilled sampler becomes viable, and the quality/diversity gap is worth it.

**Trap:** Reciting "diffusion beat GANs" as a universal fact. It's true for offline image
synthesis, which is not every problem. The interviewer picked the latency constraint deliberately.

---

## Connections

- [03-diffusion.md](03-diffusion.md) — what replaced GANs for most image synthesis, and why
- [01-autoencoders.md](01-autoencoders.md) — the likelihood-based alternative to adversarial training
- [../06-architectures/04-dl-architectures.md](../06-architectures/04-dl-architectures.md) — architecture selection framework
