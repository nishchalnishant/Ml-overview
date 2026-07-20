---
module: Generative Models
topic: Flows and Autoregressive
subtopic: Exact Likelihood Models
status: unread
tags: [generative, normalizing-flows, autoregressive, realnvp, pixelcnn, exact-likelihood]
prerequisites: [probability, neural-networks, autoencoders]
---
# Normalizing Flows and Autoregressive Models

**The problem**: VAEs optimize a *lower bound* on the likelihood. GANs abandon likelihood
entirely. Diffusion models optimize a bound too. If you actually need $\log p_\theta(x)$ — for
anomaly detection, compression, or model comparison — none of the three will give it to you.

**The core insight**: two model families give exact likelihood, by two different tricks. Flows
make the model *invertible* so the change-of-variables formula applies. Autoregressive models
*factor* the joint distribution into a product of conditionals, each of which is a tractable
1-D density. Both trade architectural freedom for exactness.

---

## 1. Normalizing Flows

### The mechanics

Start from a simple base density $z \sim p_Z$ (usually $\mathcal{N}(0,I)$) and push it through
an invertible map $x = f_\theta(z)$. The change-of-variables formula gives the density exactly:

$$\log p_X(x) = \log p_Z(f_\theta^{-1}(x)) + \log\left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|$$

This is an equality, not a bound. Train by maximizing it directly.

The cost is visible in the formula: you need $f_\theta$ invertible, *and* you need its Jacobian
determinant cheap. A general $d \times d$ determinant is $O(d^3)$ — fatal. Every flow
architecture is a scheme for making that determinant cheap.

### Coupling layers (RealNVP)

Split the input in half. Leave one half untouched; transform the other half using parameters
computed *from* the untouched half:

$$y_{1:k} = x_{1:k}, \qquad y_{k+1:d} = x_{k+1:d} \odot \exp(s(x_{1:k})) + t(x_{1:k})$$

The Jacobian is triangular, so its determinant is just the product of the diagonal —
$\sum_i s(x_{1:k})_i$. Linear time. And inversion is trivial algebra: subtract $t$, divide by
$\exp(s)$. Crucially $s$ and $t$ can be arbitrary neural networks — they never need inverting,
because they only ever see the half that passes through unchanged.

Stack many coupling layers, permuting which half is frozen each time, and expressiveness
accumulates.

```python
def coupling_forward(x, s_net, t_net, k):
    x1, x2 = x[:, :k], x[:, k:]
    s, t = s_net(x1), t_net(x1)
    y2 = x2 * torch.exp(s) + t
    return torch.cat([x1, y2], dim=1), s.sum(dim=1)   # output, log|det J|

def coupling_inverse(y, s_net, t_net, k):
    y1, y2 = y[:, :k], y[:, k:]
    s, t = s_net(y1), t_net(y1)                        # same nets, no inversion needed
    x2 = (y2 - t) * torch.exp(-s)
    return torch.cat([y1, x2], dim=1)
```

### When it breaks

Flows are **dimension-preserving by construction** — latent and data space have the same
dimensionality. There is no bottleneck, so no compression, and high-resolution images demand
enormous models. This is the structural reason flows lost the image-synthesis race to diffusion
despite the theoretical advantage of exact likelihood.

The notorious empirical failure: flows trained on CIFAR-10 assign *higher* likelihood to
SVHN images than to CIFAR-10 itself. Exact likelihood turns out not to mean "good anomaly
detector" — likelihood is dominated by low-level statistics like local smoothness.

---

## 2. Autoregressive Models

### The mechanics

Factor the joint exactly with the chain rule — no approximation involved:

$$p_\theta(x) = \prod_{i=1}^{d} p_\theta(x_i \mid x_{<i})$$

Each conditional is a small tractable distribution (a softmax over pixel intensities, a
categorical over tokens). The product is the exact joint likelihood.

**Training is parallel; sampling is sequential.** With masking, every conditional is evaluated
in a single forward pass over the whole input — teacher forcing. But sampling requires $d$
sequential passes, because $x_i$ cannot be drawn until $x_{<i}$ exists. For a 256×256 RGB image
that is ~196k sequential steps.

PixelCNN enforces the ordering with masked convolutions; the masking must be causal in raster
order, which is what creates the well-known blind-spot problem that gated PixelCNN fixed with
two separate stacks.

### The connection you should make in an interview

**Every LLM is an autoregressive generative model.** Next-token prediction *is* the chain-rule
factorization above, with a Transformer computing the conditionals and causal masking enforcing
the ordering. The reason training parallelizes and inference doesn't is exactly the reason
described here — it's not a Transformer quirk.

---

## Comparison

| Family | Likelihood | Sampling | Latent | Main weakness |
| :--- | :--- | :--- | :--- | :--- |
| VAE | Lower bound (ELBO) | 1 pass | Compressed, smooth | Blurry samples |
| GAN | None | 1 pass | Unstructured | Training instability, mode collapse |
| Diffusion | Bound | 20–1000 passes | Same dim as data | Sampling cost |
| **Normalizing flow** | **Exact** | 1 pass | Same dim as data | No compression; huge models |
| **Autoregressive** | **Exact** | $d$ passes | None | Sampling is $O(d)$ sequential |

---

## Interview Angles

### Q: Why would you pick a flow over a VAE? [Easy]

When you need the actual likelihood value, not a bound — density estimation, compression, or
comparing how well two models fit the same data. A VAE's ELBO is a lower bound with an unknown
gap, so ELBO differences between models are not trustworthy comparisons.

**Cross-questions to expect**
- *"When would the bound be good enough?"* — Whenever you only want samples or representations.
  Which is most of the time, and is why VAEs are more common.
- *"What do you pay for exactness?"* — Invertibility forces equal dimensionality: no bottleneck,
  no compression, much larger models.

**Trap:** Saying flows give "better samples." They generally don't. They give better *likelihoods*.

### Q: Training an autoregressive model is parallel but sampling is sequential. Why the asymmetry? [Medium]

Training has the ground-truth prefix available, so all conditionals $p(x_i \mid x_{<i})$ can be
evaluated simultaneously with causal masking — one forward pass. Sampling has no ground truth:
$x_i$ must be drawn before $x_{i+1}$ can be conditioned on it. The dependency is inherent to the
factorization, not to the architecture.

**Cross-questions to expect**
- *"Does this apply to LLMs?"* — Yes, identically. This is why LLM inference is memory-bandwidth
  bound while training is compute-bound.
- *"Any way around it?"* — Speculative decoding, and non-autoregressive/diffusion text models.
  Both trade exactness or quality for parallelism.

**Trap:** Attributing the asymmetry to the Transformer. It's the chain-rule factorization; an
LSTM has the same property.

### Q: Your flow-based anomaly detector flags normal production inputs as anomalies, and misses real ones. [Hard]

Likely the known likelihood-misassignment pathology: deep generative models assign high
likelihood to inputs that are *simpler* than the training data — smoother, lower-variance,
more compressible — regardless of whether they are semantically in-distribution. A flow trained
on complex images will happily assign higher likelihood to near-constant ones.

**Cross-questions to expect**
- *"So the model is undertrained?"* — No. This reproduces with perfectly converged models. It
  is a property of likelihood in high dimensions, not a fitting failure.
- *"What would you use instead?"* — Likelihood ratios against a background model, typicality
  tests rather than raw density, or an approach that doesn't route through likelihood at all —
  Isolation Forest or a reconstruction-error method.

**Trap:** Adding more training data or a bigger flow. The failure is in using likelihood as an
anomaly score, so scaling the likelihood model does not address it.

---

## Connections

- [01-autoencoders.md](01-autoencoders.md) — the bound-based alternative, and where blurriness comes from
- [02-gans.md](02-gans.md) — the likelihood-free alternative
- [03-diffusion.md](03-diffusion.md) — what beat flows on image synthesis despite optimizing only a bound
- [../03-classical-ml/10-anomaly-detection.md](../03-classical-ml/10-anomaly-detection.md) — isolation-based scoring, which sidesteps the likelihood pathology above
- [../10-llms/01-training-process.md](../10-llms/01-training-process.md) — autoregressive factorization at LLM scale
