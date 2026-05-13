# Autoencoders

Autoencoders are models that learn to compress data and then reconstruct it.

That sounds simple. It is also a very useful way to learn representations.

---

# 1. Core Idea

An autoencoder has two main parts:

- **Encoder:** $z = f_\phi(x)$ — compresses input $x \in \mathbb{R}^D$ into latent code $z \in \mathbb{R}^d$ where $d \ll D$
- **Decoder:** $\hat{x} = g_\theta(z)$ — reconstructs from latent code

The model is trained to minimize reconstruction loss:

$$L = \| x - g_\theta(f_\phi(x)) \|^2 \quad \text{(MSE)} \quad \text{or} \quad -\sum x_i \log \hat{x}_i \quad \text{(BCE for binary data)}$$

---

# 2. Why They Matter

Autoencoders are useful for:
- **Compression:** learn compact, meaningful representations
- **Denoising:** reconstruct clean data from noisy inputs
- **Anomaly detection:** high reconstruction error = anomaly
- **Representation learning:** pre-train encoder for downstream tasks
- **Generative modeling** (VAE): sample from latent space to generate new data

---

# 3. Main Variants

## Undercomplete Autoencoder

Bottleneck dimension $d < D$ forces compression. The encoder must preserve the most informative structure.

**Limitation:** can learn trivial copies via identity function if capacity is too high and no other constraint is added.

## Denoising Autoencoder (DAE)

Input is deliberately corrupted ($\tilde{x} = x + \epsilon$, or random masking), target is the clean original:

$$L = \| x - g_\theta(f_\phi(\tilde{x})) \|^2$$

Forces the encoder to learn robust representations that ignore noise. Related to masked language modeling (BERT).

## Sparse Autoencoder

Adds sparsity penalty to latent activations:

$$L = \| x - \hat{x} \|^2 + \lambda \| z \|_1$$

Results in distributed, sparse codes. Only a few neurons activate per input. Used in neural network interpretability research (Anthropic's sparse autoencoder work on understanding LLMs).

## Variational Autoencoder (VAE)

---

# 4. Variational Autoencoder (VAE)

The VAE replaces the deterministic bottleneck with a **probabilistic latent space**.

**Encoder outputs distribution parameters**, not a point:

$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

**Decoder maps samples back to data space:**

$$p_\theta(x|z) \quad \text{(generative model)}$$

### ELBO Loss (Evidence Lower BOund)

The VAE maximizes the ELBO = reconstruction - KL divergence:

$$\mathcal{L}(\phi, \theta; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

- **Reconstruction term:** $\log p_\theta(x|z)$ — how well the decoder recovers $x$
- **KL term:** penalizes deviation from the prior $p(z) = \mathcal{N}(0, I)$ — keeps latent space smooth and continuous

For Gaussian encoder and prior, KL has closed form:

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### Reparameterization Trick

Sampling $z \sim q_\phi(z|x)$ is not differentiable. Solution:

$$z = \mu + \epsilon \odot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

This separates the stochastic part ($\epsilon$) from the learned parameters, allowing gradients to flow through $\mu$ and $\sigma$.

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std             # reparameterization trick
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### β-VAE

Increases the weight on the KL term ($\beta > 1$) to encourage **disentangled** representations — latent dimensions correspond to independent generative factors (e.g., pose, color, shape become separate dimensions).

---

# 5. Comparison with Other Generative Models

| Model | Training objective | Samples quality | Latent control | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **VAE** | ELBO (reconstruction + KL) | Blurry (for images) | Smooth interpolation | Fast |
| **GAN** | Adversarial (min-max game) | Sharp | Less stable | Fast |
| **Diffusion** | Denoising score matching | Best quality | Moderate | Slow |
| **Flow** | Exact likelihood | Good | Exact inversion | Medium |

VAEs generate blurrier images than GANs because the reconstruction loss averages over modes. But their **structured, continuous latent space** makes them great for anomaly detection, interpolation, and representation learning.

---

# 6. Latent Space Properties

**Interpolation:** because the prior is Gaussian and the KL term enforces smooth latent space, interpolating between two encodings produces semantically meaningful intermediate representations:

```python
z1, _ = model.encode(x1)
z2, _ = model.encode(x2)
for alpha in torch.linspace(0, 1, steps=10):
    z_interp = alpha * z1 + (1 - alpha) * z2
    x_interp = model.decode(z_interp)
```

**Sampling novel data:**
```python
z_sample = torch.randn(batch_size, latent_dim)   # sample from prior N(0, I)
x_generated = model.decode(z_sample)
```

**Anomaly detection:** encode a test sample; high reconstruction error or large $|z|$ (far from prior) signals an anomaly.

---

# 7. Use Cases

| Application | How autoencoder is used |
| :--- | :--- |
| **Anomaly detection** | Train on normal data; flag high reconstruction error |
| **Denoising** | DAE trained on clean/noisy pairs |
| **Image generation** | VAE decoder samples from latent prior |
| **Feature learning** | Encoder pre-trained unsupervised; fine-tuned on downstream task |
| **Dimensionality reduction** | Encoder replaces PCA for non-linear structure |
| **Drug discovery** | VAE over molecular graphs for latent-space optimization |
