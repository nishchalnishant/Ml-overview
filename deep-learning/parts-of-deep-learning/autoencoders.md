# Autoencoders

Autoencoders are models that learn to compress data and then reconstruct it.

That sounds simple.
It is also a very useful way to learn representations.

---

# 1. Core Idea

An autoencoder has two main parts:

- encoder
- decoder

The encoder compresses the input into a latent representation.
The decoder reconstructs the input from that latent space.

So the model is forced to learn what information is essential.

---

# 2. Why They Matter

Autoencoders are useful for:

- compression
- denoising
- anomaly detection
- representation learning

They are not the default answer for everything, but they are conceptually important and still very useful in the right settings.

---

# 3. Main Variants

## Undercomplete Autoencoder

Uses a smaller latent space so compression is forced.

## Denoising Autoencoder

Learns to reconstruct clean input from noisy input.

## Sparse Autoencoder

Adds sparsity pressure to latent activations.

## Variational Autoencoder

Learns a probabilistic latent space for smoother generation and interpolation.

That last one is the most interview-relevant generative extension.
