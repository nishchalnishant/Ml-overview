# Generative Models: Creating from Data

Generative models learn the underlying probability distribution of the training data $(P(x))$ to generate new, similar samples.

---

##  Core Architectures

### 1. Variational Autoencoders (VAEs)
- **Mechanism:** Maps input to a latent space (mean and variance) and samples from a Gaussian distribution to reconstruct the data.
- **Likelihood:** Optimizes the Evidence Lower Bound (ELBO).
- **Pros:** Smooth latent space, great for interpolation.
- **Cons:** Often produces blurry images.

### 2. Generative Adversarial Networks (GANs)
- **Mechanism:** Two networks—a **Generator** (creates fake data) and a **Discriminator** (tries to spot fakes)—compete in a zero-sum game.
- **Loss:** Minimax loss function.
- **Pros:** Produces sharp, realistic images.
- **Cons:** Unstable training, "Mode Collapse" (generating the same sample repeatedly).

### 3. Diffusion Models
- **Mechanism:** Gradually adds noise to data until it's pure Gaussian noise, then learns to reverse the process (denoising).
- **SOTA:** Powers DALL-E 3, Midjourney, and Stable Diffusion.
- **Pros:** Stable training, high-quality diverse samples.
- **Cons:** Slow inference (requires many denoising steps).

### 4. Autoregressive Models (LLMs)
- **Mechanism:** Predicts the next token/pixel based on all previous ones.
- **Examples:** GPT-4, Llama 3, PixelCNN.
- **Pros:** Excellent for discrete data (text).

---

## 🔬 Discriminative vs. Generative

| Feature | **Discriminative** | **Generative** |
|---------|-------------------|----------------|
| **Goal** | Learn $P(y|x)$ (Boundary) | Learn $P(x)$ or $P(x,y)$ (Distribution) |
| **Output** | Label / Class | New Data Sample |
| **Example** | SVM, Random Forest | GAN, VAE, Diffusion |

---

##  Interview Questions

**1. "What is Mode Collapse in GANs and how to fix it?"**
> Mode collapse is when the generator produces a very limited set of outputs that "fool" the discriminator but lack diversity. Fixes: Unrolled GANs, Wasserstein GAN (WGAN), or Mini-batch discrimination.

**2. "Why are Diffusion models preferred over GANs now?"**
> Diffusion models provide much better sample diversity, more stable training (no adversarial game), and state-of-the-art image quality, despite being slower at inference.

**3. "Explain the Reparameterization Trick in VAEs."**
> Backpropagation cannot go through a random sampling step. Instead of sampling $z \sim N(\mu, \sigma)$, we sample $\epsilon \sim N(0, 1)$ and compute $z = \mu + \sigma \cdot \epsilon$. This makes the sampling step differentiable with respect to $\mu$ and $\sigma$.
