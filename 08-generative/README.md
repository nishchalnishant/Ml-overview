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

**What lives here:** autoencoders, VAE, GAN, diffusion models, normalizing flows.

**Status — known gap.** Only `01-autoencoders.md` is in place. Substantive coverage of GANs (incl. WGAN-GP), VAEs (ELBO, reparameterization), and diffusion (DDPM/DDIM, latent diffusion, consistency models) currently lives in `../06-architectures/04-dl-architectures.md` §4–6 and should be moved here — roughly 4k words. It is misfiled, not missing.

**Interviewers probe** the reparameterization trick (why sampling must be moved out of the gradient path) and why diffusion largely displaced GANs for image synthesis.
