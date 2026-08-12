---
module: Generative Models
topic: Revision Guide
subtopic: "08-generative"
status: unread
tags: [vae, gan, diffusion, flows, autoregressive, revision]
prerequisites: [probability, neural-networks]
---
# 08 Generative Models — Revision Guide

Fast-revision pass over `08-generative/`: one outline of every major concept, then graded Q&A
(Easy/Medium/Hard) spanning autoencoders, GANs, diffusion, flows/autoregressive, and frontier
generative models. Use the outline to locate a gap, then drill it with the matching questions.

```markdown
- **08 Generative Models**
  - **1. Autoencoders**
    - Undercomplete AE<br/>bottleneck compression
    - Denoising AE<br/>corrupt then reconstruct
    - Sparse AE<br/>L1 penalty on latent
    - VAE<br/>encode to distribution
    - ELBO<br/>recon - KL
    - Reparameterization trick<br/>z = mu + eps*sigma
    - Posterior/KL collapse
    - beta-VAE<br/>disentanglement trade-off
    - AE use cases<br/>anomaly/denoise/interp
  - **2. GANs**
    - Minimax objective<br/>min_G max_D
      - Nash equilibrium<br/>D(x)=0.5
    - Mode collapse
    - Vanishing gradient<br/>D too strong
      - Non-saturating loss
    - Training oscillation
    - WGAN<br/>Earth-mover distance
    - WGAN-GP<br/>Lipschitz constraint
    - GAN vs diffusion<br/>1-step advantage
  - **3. Diffusion Models**
    - Forward process<br/>closed-form q(xt|x0)
    - Noise-prediction objective<br/>L_simple
    - Why training is stable<br/>regression not adversarial
    - DDPM<br/>stochastic, T steps
    - DDIM<br/>deterministic ODE reframe
      - Classifier-free guidance<br/>fidelity/diversity dial
    - Latent diffusion<br/>compressed latent space
    - Consistency models intro<br/>1-4 steps
  - **4. Flows & Autoregressive**
    - Exact likelihood motivation
    - Normalizing flows<br/>invertible map
    - Change-of-variables formula
    - Jacobian determinant cost<br/>O(d^3) problem
    - Coupling layers<br/>RealNVP
    - Triangular Jacobian<br/>cheap det
    - Flow weakness<br/>dimension-preserving, no compression
    - Flow anomaly pathology<br/>CIFAR vs SVHN
    - Autoregressive models<br/>chain rule factorization
    - Conditionals p(xi given x<i)
    - PixelCNN<br/>masked convolutions
    - Train parallel / sample sequential
    - LLMs as autoregressive<br/>next-token prediction
  - **5. Frontier Generative Models**
    - Flow matching<br/>regress velocity field
      - SD3 / FLUX<br/>flow matching + DiT
    - Probability path<br/>straight-line interpolation
    - Rectified flow
    - Reflow<br/>straightening trajectories
    - Consistency models<br/>trajectory to origin map
      - Consistency distillation<br/>teacher-based
      - Consistency training<br/>no teacher
      - Progressive distillation<br/>halving steps
    - DiT backbone<br/>transformer over patches
    - adaLN conditioning
    - Video generation<br/>diffusion over time
      - Autoregressive-over-time<br/>chunked video
    - Temporal consistency<br/>flicker problem
    - Temporal attention
```

## Questions

### Easy

1. What problem does an undercomplete autoencoder's bottleneck solve, and what breaks if the bottleneck isn't tight enough?
2. How does a denoising autoencoder differ from a plain undercomplete autoencoder in what it's trained to do?
3. What does the L1 penalty in a sparse autoencoder actually do to the latent code?
4. Why does a standard (non-variational) autoencoder produce "holes" in latent space when you sample randomly?
5. What two things does a VAE's encoder output, instead of a single point?
6. Write out the two terms of the ELBO in words — what does each one push the model to do?
7. Why can't you backpropagate through the sampling step $z \sim \mathcal{N}(\mu,\sigma^2)$ directly?
8. What does the reparameterization trick actually change about how $z$ is computed?
9. Why are VAE samples typically blurrier than GAN samples?
10. When would you prefer an autoencoder over PCA for dimensionality reduction?
11. What are the two networks in a GAN, and what does each one optimize?
12. State the GAN minimax objective. What does the Nash equilibrium look like in terms of $D(x)$?
13. What is mode collapse, in one sentence?
14. What replaces DDPM's noise-prediction objective in flow matching?
15. What is the closed-form forward process in diffusion, and why is it "closed form"?
16. What does $\epsilon_\theta$ predict in the DDPM training objective?
17. What's the practical difference between DDPM and DDIM sampling?
18. Why do normalizing flows need an invertible mapping $f_\theta$?
19. What does the chain rule factorization $p_\theta(x) = \prod_i p_\theta(x_i|x_{<i})$ buy an autoregressive model?
20. Why is LLM next-token prediction described as autoregressive generative modeling?
21. Name the one regime where GANs still beat diffusion models today.
22. What is classifier-free guidance a dial between?
23. What's the key architectural difference between a U-Net and a DiT backbone?

### Medium

1. Posterior collapse: describe what the encoder, KL term, and decoder each look like when it happens, and give two different root causes with two different fixes.
2. Why does increasing $\beta$ in a β-VAE trade off against reconstruction quality — walk through the mechanism.
3. Explain why the reparameterization trick works for Gaussians but breaks down for discrete/categorical latents.
4. A GAN's discriminator becomes near-perfect early in training. Why does the generator's gradient vanish, and what's the standard fix?
5. Why does WGAN's critic need to be 1-Lipschitz, and how is that enforced in WGAN-GP?
6. Compare mode collapse and training oscillation — why does "just train the discriminator less" trade one for the other?
7. Why is the closed-form forward process specifically a *training*-time optimization rather than an inference-time one?
8. DDIM and DDPM use the same trained model. What exactly changes between them, and what does determinism buy you beyond speed?
9. Explain why flows are dimension-preserving by construction, and why that's the structural reason they lost the image-synthesis race to diffusion.
10. Why is training parallel but sampling sequential for autoregressive models — is this a Transformer-specific property?
11. In RealNVP's coupling layer, why is the Jacobian triangular, and why does that make the determinant cheap?
12. Why do $s$ and $t$ in a coupling layer never need to be inverted, even though they're arbitrary neural networks?
13. Compare the failure mode of VAE blur vs. GAN mode collapse vs. flow anomaly-misassignment — are they the same underlying issue?
14. Mechanically, what's different between DDPM's training target and flow matching's training target?
15. Why does a straight-line probability path (rectified flow) allow far fewer sampling steps than DDPM's implied ODE trajectory?
16. What is "reflow" in rectified flow, and why does retraining on the model's own generated pairs straighten trajectories?
17. Explain the difference between consistency distillation and consistency training — what does each need or not need?
18. Why is progressive distillation conceptually different from a single consistency-model objective, even though both compress step count?
19. Why do production text-to-image systems pair rectified flow with a DiT backbone specifically (not just coincidentally)?
20. What makes video diffusion harder than image diffusion beyond "one more dimension" — name the two engineering problems it introduces.
21. Your text-to-image model produces plausible images that ignore half the prompt — what's the most likely first thing to check, and why?
22. Why would you pick a normalizing flow over a VAE if you need an actual likelihood value rather than samples?

### Hard

1. Derive the ELBO from $\log p_\theta(x)$ using Jensen's inequality / the KL divergence between $q_\phi(z|x)$ and the true posterior. Where exactly does the "lower bound" gap come from?
2. Given the closed-form Gaussian KL term for a diagonal-covariance VAE, derive $D_{KL} = -\frac12\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$ from the general Gaussian KL formula.
3. You're asked to ship real-time avatar generation at 60fps — GAN or diffusion, and what do you give up? Now the frame budget relaxes to 200ms — does your answer change, and why?
4. A flow-based anomaly detector flags normal production inputs as anomalous and misses real anomalies, even though the model is fully converged. Diagnose the root cause and propose an alternative that avoids it.
5. Explain why deep generative models (flows in particular) can assign *higher* likelihood to simpler/smoother out-of-distribution inputs than to genuine training-distribution data. Is this a fitting failure?
6. Is flow matching a different model family from diffusion, or a generalization of it? Justify using the probability-path framing, and explain what DDPM's forward process looks like as a special case.
7. Derive/justify why a consistency model's identity-consistency constraint at $t=0$ ($f_\theta(x_0,0)=x_0$) is necessary for the self-consistency loss to be a well-posed regression target rather than degenerate.
8. Why does a consistency model trained via consistency distillation inherit biases from its teacher's ODE solver, while consistency training does not — and what does that cost in convergence speed and quality?
9. Explain the "blind-spot problem" in PixelCNN's causal masked convolutions and why gated PixelCNN's two-stack design fixes it.
10. Both diffusion models and autoregressive models are "exact" in different senses (autoregressive: exact likelihood; diffusion: bound). Reconcile why diffusion still outperforms exact-likelihood flows on image quality despite optimizing a looser objective.
11. What breaks in the reparameterization trick if you use a full-covariance Gaussian posterior instead of diagonal — what's the added computational cost and why do most VAEs avoid it?
12. In WGAN-GP, why does penalizing the gradient norm toward 1 (rather than clipping weights, as in the original WGAN) better preserve the critic's capacity to produce meaningful gradients?
13. You need exact likelihoods AND high sample fidelity AND fast sampling simultaneously — walk through why no single family in this folder gives you all three, and what you'd actually ship.
14. A video diffusion model produces temporally flickering output despite strong per-frame quality. Diagnose which architectural piece is likely missing and why chunked/autoregressive-over-time generation only partially fixes it.
15. Explain why DiT's adaptive layer norm conditioning is a meaningfully different mechanism from U-Net's feature-concatenation conditioning, and why that difference matters for scaling behavior.
16. Rectified flow's "reflow" procedure is claimed to make trajectories non-crossing. Why does non-crossing matter specifically for the accuracy of 1-step sampling, and what would crossing trajectories cause a 1-step model to produce?
17. If you had unlimited compute, would you still expect a diffusion model trained with the DDPM objective to beat a normalizing flow of equal parameter count on image sample quality? Justify using the compression/bottleneck argument from `04-flows-and-autoregressive.md`.
18. A team claims their new consistency model "is just a distilled diffusion model." Under what training procedure would that claim be true, under what procedure would it be false, and why does the distinction matter for downstream sampling-time behavior?
19. Explain structurally why GAN training has no convergence guarantee (minimax game framing), and why diffusion's regression objective sidesteps that guarantee issue entirely rather than just mitigating it.
20. Construct a case where β-VAE's disentanglement claim (independent latent dims ↔ independent generative factors) fails even at high β — what property of the true generative factors would cause this?
