---
module: Generative Models
topic: Frontier Generative Models
subtopic: Flow Matching / Consistency Models / Video Diffusion
status: unread
tags: [generative, flow-matching, rectified-flow, consistency-models, distillation, diffusion-transformer, video-generation]
prerequisites: [probability, neural-networks, diffusion]
---
# Frontier Generative Models

### What the interviewer is actually testing

Whether you've kept up past DDPM/DDIM — that you know diffusion's training objective and
sampling cost have both moved since 2020, and that today's production image/video systems
(SD3, FLUX, Sora-class models) are not "DDPM with a bigger U-Net." The competency: explain flow
matching as a distinct training objective from noise-prediction, explain how consistency models
get to 1-4 steps, and place both in the systems that actually ship.

---

## 1. Flow matching / rectified flow

### The reasoning structure

DDPM defines a *stochastic* forward process — a fixed Markov chain of Gaussian noise additions —
and trains a network to predict the noise at each step. The reverse process is stochastic (DDPM)
or reframed as an ODE post-hoc (DDIM), but the training objective was never phrased in terms of
that ODE directly.

Flow matching flips the design order: start from the ODE. Define a *probability path* $p_t$ that
continuously interpolates between the data distribution and a simple noise distribution, and
directly regress a velocity field $v_\theta$ that generates that path:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right], \qquad x_t = (1-t)x_0 + t\, x_1$$

where $x_0$ is a data sample and $x_1$ is noise. This is **rectified flow / conditional flow
matching** in its simplest form: the target velocity is just the straight line from data to
noise — even simpler than DDPM's noise-prediction target, since there's no $\bar\alpha_t$
schedule to bake in.

### The pattern in action

DDPM's forward process is a random walk — the marginal $x_0 \to x_t$ is Gaussian, but the
*implied ODE trajectory* a sample follows during reverse sampling is curved, because the score
function changes direction as noise decreases. Curved trajectories need many small steps to
integrate accurately; that's why DDIM still wants 20-50 steps rather than 1.

Rectified flow trains directly on straight-line interpolation paths, which are exactly
integrable with far fewer steps than a curved one. "Reflow" — resampling the model's own
generated pairs $(x_0, x_1)$ and retraining — straightens paths further, since a flow model
trained on its own outputs learns non-crossing trajectories, which is what one-step accuracy
requires.

**Contrast with 03-diffusion.md directly:**

| | DDPM | Flow matching / rectified flow |
| :--- | :--- | :--- |
| Forward process | Fixed stochastic noising schedule | Any interpolation path (often straight-line) |
| Training target | Predict added noise $\epsilon$ | Predict velocity $x_1 - x_0$ |
| Underlying object | Score function (implicitly) | Velocity field (explicitly) |
| Sampling trajectory | Curved (even under DDIM's ODE reframing) | Straight by construction, straighter after reflow |
| Steps to good quality | 20-50 (DDIM) | 1-8, sometimes 1 after reflow/distillation |

**Trap: thinking flow matching is a different model family from diffusion.** It isn't — DDPM,
score-based SDEs, and flow matching are different parameterizations of the same idea (learn to
reverse a data→noise process). Flow matching is best understood as a *generalization*: DDPM's
forward process is one particular curved, variance-preserving probability path; flow matching
lets you pick the path, and picking a straight one is what buys the speed.

---

## 2. Consistency models and distillation

### The problem being solved

Diffusion sampling — even DDIM at 20-50 steps — means 20-50 sequential network evaluations per
sample. That's the dominant cost of serving image/video generation at scale. The goal: collapse
sampling to 1-4 steps without retraining from scratch as a GAN (which reintroduces adversarial
instability) or discarding the diffusion model's coverage/diversity advantages.

### The core trick

Every diffusion (or flow-matching) model implicitly defines an ODE trajectory from noise
$x_T$ back to a clean sample $x_0$. A consistency model $f_\theta$ is trained so that **any
point on that same trajectory maps to the same origin**:

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') = x_0 \quad \text{for any } t, t' \text{ on the same ODE trajectory}$$

If you can learn that function, sampling is one call: draw noise, apply $f_\theta$, done. Because
$f_\theta$ is also defined to be identity-consistent at $t=0$ ($f_\theta(x_0, 0) = x_0$), it's a
well-posed regression target, not a generative game.

**Two ways to get there:**

- **Consistency distillation.** Start from an already-trained diffusion/flow model. Use it (with
  a numerical ODE solver) to generate adjacent points on trajectories, and train $f_\theta$ to
  agree on nearby $(x_t, x_{t'})$ pairs. A teacher model defines the trajectories; the student
  learns to shortcut them.
- **Consistency training.** Skip the teacher. Use the same self-consistency loss but estimate the
  trajectory locally from the forward-noising process directly. Slower to converge and
  historically lower quality than distillation, but removes the dependency on a good teacher.

**Progressive distillation** is the earlier, related idea worth contrasting: repeatedly train a
student to do in $N/2$ steps what the teacher does in $N$, then repeat, halving each generation.
It reaches few-step models through a sequence of distillation rounds rather than one
consistency objective — more training runs, conceptually simpler, and often used as one stage in
a longer pipeline (distill to few-step, then apply a consistency-style objective on top).

### The quality/speed tradeoff

1-step consistency models are fast but visibly behind multi-step diffusion on fine detail and
diversity; 2-4 step consistency or distilled-rectified-flow models close most of that gap.
Production systems generally don't ship pure 1-step — they use a handful of steps (LCM-style
latent consistency models at 4-8 steps, or few-step distilled rectified flow) as the sweet spot
where the speed win is large and the quality loss is close to imperceptible.

**Trap: saying consistency models "are" diffusion models.** They're trained *from* or *alongside*
a diffusion/flow process but the sampling-time function is not doing the same thing — it's a
direct trajectory-shortcut regressor, not an iterative denoiser. Calling it "diffusion with fewer
steps" undersells that the training objective is structurally different (self-consistency across
a trajectory vs. per-step noise/velocity regression).

---

## 3. Where production systems actually sit

SD3, FLUX, and Sora-class video models combine two choices that are each independently covered
elsewhere in this repo and worth keeping separate in an interview answer:

- **Training objective:** rectified flow / flow matching (§1 above), not DDPM noise-prediction.
  This is a modeling choice about the target and the probability path.
- **Backbone architecture:** a Diffusion Transformer (DiT) instead of a convolutional U-Net —
  patchify the (usually latent-space) input into tokens and process with a standard transformer
  block stack, conditioning via adaptive layer norm rather than U-Net-style feature concatenation.
  This is an architecture choice, decoupled from the objective; see
  [../06-architectures/04-dl-architectures.md](../06-architectures/04-dl-architectures.md) for
  the backbone-selection framing. DiT scales the way transformers scale elsewhere in this repo —
  predictably with parameters and compute — which is a large part of why it displaced the U-Net
  at frontier scale.

These two choices pair up for practical, not logically necessary, reasons: rectified flow's
straighter trajectories reduce the number of forward passes needed, and DiT's scaling behavior
makes each pass more worth paying for at high resolution. Distillation (§2) is then typically
applied on top of the trained flow-matching + DiT model to get few-step inference for
interactive products.

---

## 4. Video generation as diffusion over time

Video generation is conceptually "diffusion/flow matching, plus a temporal axis" — but that
addition is where most of the real engineering difficulty lives.

- **Temporal consistency.** Denoising each frame independently produces flicker — objects
  subtly change identity frame to frame because each frame's denoising trajectory is an
  independent sample. Production video models add temporal attention/convolution layers so the
  model attends across frames, not just within one, making the denoising process jointly
  consistent across the clip rather than per-frame.
- **Compute cost.** A video is a 3D tensor (time × height × width); attention cost over
  space-time tokens grows sharply with clip length and resolution. This is why video models lean
  even harder on latent-space compression (spatial *and* temporal compression in the encoder)
  and on few-step sampling (§2) — the per-step cost is already much higher than image generation,
  so cutting step count matters proportionally more.
- **Autoregressive-over-time hybrids.** Some systems generate video in chunks — diffuse a short
  window, condition the next window on it — trading long-range temporal consistency for bounded
  memory/compute, a sequential structure layered on top of a diffusion/flow generator per chunk.

**Trap: treating video generation as "just diffusion with an extra dimension."** The training
objective (flow matching or noise prediction) does generalize directly. What doesn't come for
free is consistency across that new axis — it requires architectural additions (temporal
attention) and, usually, an explicit compute strategy (chunking, temporal compression), not just
a reshaped tensor.

---

## Interview Angles

### Q: How is flow matching different from DDPM, mechanically? [Medium]

DDPM fixes a stochastic Gaussian noising schedule and trains the network to predict the noise
added at a sampled timestep. Flow matching instead defines a probability path (often the
straight-line interpolation between data and noise) and trains the network to predict the
velocity field that generates that path. Both are regression objectives with no adversarial
component, but flow matching's straight paths are what make few-step sampling accurate — DDPM's
implied ODE trajectory is curved even when DDIM samples from it.

**Cross-questions to expect**
- *"Is flow matching a strict improvement, or a different model?"* — Generalization, not a
  separate family: DDPM's forward process is one specific (curved) probability path; flow
  matching lets you choose the path.
- *"Why does a straight path matter for sampling speed?"* — Straight lines are exactly
  integrable in fewer steps; curved trajectories accumulate discretization error faster per step,
  so they need smaller steps and thus more of them.

**Trap:** Describing flow matching as "faster diffusion" without being able to say what's
actually different in the loss. The speed is a downstream consequence of the path choice, not an
independent trick bolted on.

### Q: How do consistency models get to 1-4 step sampling, and what do you give up? [Hard]

They train a function that maps any point on a diffusion/flow ODE trajectory directly to the
trajectory's origin (the clean sample), using either consistency distillation (from a pretrained
teacher, via a solver-generated trajectory) or consistency training (self-consistency loss with
no teacher). Because the function is trained to be trajectory-consistent, one evaluation
approximates what previously took an iterative solver. The cost is quality/diversity at the very
low step counts — 1-step models visibly lag multi-step diffusion on fine detail — which is why
production systems typically land at 4-8 steps rather than 1.

**Cross-questions to expect**
- *"Distillation implies a teacher — does consistency training need one?"* — No, that's the
  distinction: consistency training estimates local trajectory structure from the forward process
  directly, at the cost of slower convergence and historically weaker quality than distillation.
- *"How is this different from progressive distillation?"* — Progressive distillation halves step
  count through repeated distillation rounds (teacher at $N$ steps → student at $N/2$ → repeat).
  Consistency models use a single self-consistency objective targeting arbitrary points on one
  trajectory, not a sequence of halving rounds.

**Trap:** Calling a consistency model "a distilled diffusion model" unconditionally — consistency
*training* explicitly has no teacher. Only consistency *distillation* fits that description.

---

## Connections

- [03-diffusion.md](03-diffusion.md) — the DDPM/DDIM objective and ODE reframing that flow
  matching generalizes and consistency models distill
- [04-flows-and-autoregressive.md](04-flows-and-autoregressive.md) — normalizing flows'
  invertibility/change-of-variables machinery, a different (exact-likelihood) use of "flow"
  terminology than rectified flow here — don't conflate the two
- [02-gans.md](02-gans.md) — the single-step generative alternative that consistency
  models/distilled flow matching now largely match on speed without adversarial instability
- [../06-architectures/04-dl-architectures.md](../06-architectures/04-dl-architectures.md) —
  backbone selection (DiT vs. U-Net vs. transformer), decoupled from the objective covered here
