# Autoencoders

Autoencoders are unsupervised models that learn to reconstruct their input through a **bottleneck** representation. They are used for dimensionality reduction, denoising, and representation learning.

---

## Architecture

- **Encoder:** Maps input \(x\) to a latent code \(z = f(x)\) (often lower-dimensional). **Decoder:** Maps \(z\) to reconstruction \(\hat{x} = g(z)\). **Loss:** Reconstruction error, e.g. MSE \(\|x - \hat{x}\|^2\) or cross-entropy for binary/categorical inputs.
- The **bottleneck** (smaller dimension of \(z\)) forces the network to learn a compressed, informative representation.

---

## Variants

- **Undercomplete:** Bottleneck dimension smaller than input; learns compression. **Denoising:** Input is corrupted (e.g. noise); model learns to reconstruct clean input, encouraging robust features.
- **Variational (VAE):** Encoder outputs mean and variance of a Gaussian; sample \(z\); decoder reconstructs. Loss = reconstruction + KL divergence between latent distribution and standard normal. Enables generative sampling.
- **Sparse autoencoders:** Encourage sparse activations (e.g. L1 on codes) for interpretability or compact codes.

---

## Applications

- **Dimensionality reduction:** Use encoder output as features for downstream tasks (alternative to PCA). **Denoising:** Preprocess noisy images or signals. **Anomaly detection:** High reconstruction error on outliers. **Generative models:** VAE for sampling; also a building block in some diffusion and latent-space methods.

---

## Quick revision

- **Autoencoder:** encode → bottleneck \(z\) → decode; minimize reconstruction error. **Denoising** and **VAE** (with KL loss) are common variants. Used for compression, denoising, and representation learning.
