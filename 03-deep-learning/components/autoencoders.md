# Autoencoders

---

## Undercomplete Autoencoder

**The problem**: you have a dataset of high-dimensional inputs (images, sensor readings, documents) and no labels. You want to learn which features are actually informative — not by hand, but by letting the model discover structure on its own. Supervised learning is off the table.

**The core insight**: force the network to compress the input through a narrow bottleneck, then reconstruct the original. To reconstruct accurately, the bottleneck must capture the most important structure. The bottleneck representation is what you care about — the compression has done your feature engineering.

**The mechanics**: an encoder $f_\phi$ maps input $x \in \mathbb{R}^D$ to a latent code $z \in \mathbb{R}^d$ where $d \ll D$. A decoder $g_\theta$ reconstructs from $z$:

$$L = \|x - g_\theta(f_\phi(x))\|^2 \quad \text{(MSE for continuous data)}$$
$$L = -\sum_i x_i \log \hat{x}_i \quad \text{(BCE for binary/probability data)}$$

**What breaks**: if the bottleneck is not tight enough — or if the network has enough capacity to route information around the bottleneck — it learns a near-identity mapping. No meaningful compression occurs. You must ensure $d$ is small enough that true compression is required.

---

## Denoising Autoencoder

**The problem**: an undercomplete autoencoder is trained to reconstruct its input exactly — but "reconstruct exactly" can be gamed by memorizing training examples rather than learning generalizable structure. It is also extremely sensitive to input noise at test time.

**The core insight**: corrupt the input deliberately before passing it to the encoder, then train the decoder to reconstruct the clean original. To do this, the encoder must learn what the underlying clean signal looks like — it cannot just memorize the corrupted version.

**The mechanics**:

$$\tilde{x} = \text{corrupt}(x) \quad \text{(add Gaussian noise, or randomly mask values)}$$

$$L = \|x - g_\theta(f_\phi(\tilde{x}))\|^2$$

The encoder sees corrupted $\tilde{x}$ but must produce a latent $z$ from which the decoder can reconstruct clean $x$. Forces the model to learn robust, noise-invariant representations.

**What breaks**: the corruption distribution must match the kinds of corruption seen at test time. If you add Gaussian noise during training but encounter salt-and-pepper noise at deployment, the model may not generalize. BERT's masked language modeling is structurally a denoising autoencoder: mask some tokens (corruption), predict the original tokens (reconstruction).

---

## Sparse Autoencoder

**The problem**: undercomplete autoencoders compress by reducing dimensionality. But sometimes you want a large latent space where only a small number of dimensions are active for any given input — a distributed, sparse code. This makes the learned features more interpretable (each feature fires for specific inputs) and prevents feature entanglement.

**The core insight**: allow a wide latent space, but penalize having many non-zero activations simultaneously. Each input should activate only a few features; different inputs activate different subsets.

**The mechanics**:

$$L = \|x - \hat{x}\|^2 + \lambda \|z\|_1$$

The $L_1$ penalty on the latent code drives most activations to zero for any given input. Increasing $\lambda$ increases sparsity.

**What breaks**: if $\lambda$ is too large, the model achieves sparsity by zeroing everything — reconstruction quality collapses. If too small, the sparsity constraint is ineffective. Tuning $\lambda$ requires monitoring both reconstruction error and the average activation fraction.

Sparse autoencoders are used in interpretability research (including Anthropic's work on understanding LLM activations) because sparse, high-dimensional codes are easier to map to human-interpretable concepts.

---

## Variational Autoencoder (VAE)

**The problem**: a standard autoencoder maps each input to a single point in latent space. If you sample a random point $z$ from that space and decode it, you likely get garbage — the decoder has never been trained to handle points that aren't the encoded outputs of real training examples. The latent space has holes.

**The core insight**: instead of mapping each input to a point, map it to a *distribution* — specifically a Gaussian. Train the decoder to reconstruct from samples drawn from this distribution. To prevent the encoder from learning narrow, non-overlapping distributions (which would defeat the purpose), penalize distributions that deviate from a standard Gaussian prior. Now the latent space is densely covered, and any sampled point can be decoded into something meaningful.

**The mechanics**: the encoder outputs distribution parameters:

$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

The decoder models the generative process:

$$p_\theta(x|z) \quad \text{(e.g., Gaussian or Bernoulli)}$$

The training objective is the ELBO (Evidence Lower BOund):

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization toward } \mathcal{N}(0,I)}$$

For Gaussian encoder and prior, the KL term has a closed form:

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**What breaks**: sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is not differentiable — you cannot backpropagate through a random node.

---

## Reparameterization Trick

**The problem**: to train the VAE with gradient descent, you need gradients to flow through the sampling step $z \sim q_\phi(z|x)$. But sampling is stochastic — there is no derivative with respect to $\mu$ or $\sigma$.

**The core insight**: separate the randomness from the learned parameters. Sample the noise externally from a fixed distribution; the learned parameters only *transform* the noise deterministically.

**The mechanics**:

$$z = \mu + \epsilon \odot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now $z$ is a deterministic function of $\mu$, $\sigma$, and a constant-distribution noise variable $\epsilon$. Gradients flow through $\mu$ and $\sigma$ normally; $\epsilon$ is treated as a constant input with no gradient.

**What breaks**: the reparameterization trick works cleanly for Gaussian and a few other exponential family distributions. For discrete latent variables (e.g., categorical), it breaks down — you need the Gumbel-Softmax approximation or other specialized estimators.

---

## KL Collapse and β-VAE

**The problem**: during training, the model discovers it can minimize the reconstruction loss by ignoring the latent variable entirely — the encoder outputs a standard Gaussian regardless of input, the KL term is zero, and the decoder learns a fixed mean output. The latent space is not used. This is called posterior collapse or KL collapse.

**The core insight**: the KL term and reconstruction term are in tension. To use the latent space, reconstruction pressure must outweigh the pressure toward the prior. You can tilt this balance explicitly.

**The mechanics**: β-VAE multiplies the KL term by $\beta > 1$:

$$\mathcal{L} = \mathbb{E}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))$$

Higher $\beta$ forces the latent space to be smoother and more disentangled — latent dimensions tend to correspond to independent generative factors (pose, color, shape). Lower $\beta$ allows richer, more compressed representations at the cost of disentanglement.

**What breaks**: increasing $\beta$ reduces reconstruction quality. There is a fundamental trade-off between reconstruction fidelity and latent space structure. VAEs also produce blurrier reconstructions than GANs for the same reason — the reconstruction loss averages over multiple plausible reconstructions.

---

## Use Cases

| Application | How autoencoders are used |
| :--- | :--- |
| **Anomaly detection** | Train on normal data; flag inputs with high reconstruction error |
| **Denoising** | DAE trained on clean/noisy pairs; used in speech and image restoration |
| **Image generation** | VAE decoder samples novel examples from latent prior $\mathcal{N}(0,I)$ |
| **Representation learning** | Encoder pre-trained unsupervised; fine-tuned on downstream labeled task |
| **Dimensionality reduction** | Encoder replaces PCA for non-linear manifold structure |
| **Interpolation** | VAE's smooth latent space enables semantically meaningful blends between inputs |
| **Interpretability** | Sparse autoencoders extract monosemantic features from dense neural activations |

---

## Comparison with Other Generative Models

| Model | Training | Sample quality | Latent control | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **VAE** | ELBO | Blurry (images) | Smooth interpolation | Fast |
| **GAN** | Adversarial min-max | Sharp | Less stable | Fast |
| **Diffusion** | Denoising score matching | Best | Moderate | Slow |
| **Flow** | Exact likelihood | Good | Exact inversion | Medium |

VAEs generate blurrier images because the MSE/BCE reconstruction loss averages over multiple plausible pixel configurations. Their value is not in image generation per se — it is in the structured, continuous, well-behaved latent space that makes anomaly detection, interpolation, and representation learning reliable.
