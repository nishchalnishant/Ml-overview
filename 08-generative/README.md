---
module: Generative Models
topic: Overview
subtopic: ""
status: unread
tags: [vae, gan, diffusion, index]
prerequisites: [probability, neural-networks]
---
# Generative Models

Models that learn a distribution well enough to sample from it, rather than only to discriminate.

| File | Covers |
| :--- | :--- |
| [01-autoencoders.md](01-autoencoders.md) | Undercomplete, denoising, sparse, VAE, reparameterization, KL collapse, β-VAE |
| [02-gans.md](02-gans.md) | Minimax objective, mode collapse, vanishing gradients, oscillation, WGAN-GP |
| [03-diffusion.md](03-diffusion.md) | Closed-form forward process, noise prediction, DDPM vs. DDIM, latent diffusion, consistency models |

## Routing — this folder vs. 06-architectures

[`../06-architectures/04-dl-architectures.md`](../06-architectures/04-dl-architectures.md) §4–6
covers the same three model families in **interview-framing** modality: what the interviewer is
testing → reasoning structure → traps. The files here are the **deep-dive**: derivations and
mechanics. Both are load-bearing; §4 and §6 now point here for depth, while §5 keeps its framing
alongside `01-autoencoders.md`.

**Interviewers probe** the reparameterization trick (why sampling must be moved out of the
gradient path) and why diffusion largely displaced GANs for image synthesis — and, if they are
good, where it did not.

**Known gap:** normalizing flows and autoregressive generative models are not yet covered.
