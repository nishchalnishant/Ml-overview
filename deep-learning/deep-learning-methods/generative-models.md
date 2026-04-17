# Generative Models

Generative models try to learn what data looks like well enough to create new examples.

That is why they feel magical.

It is also why they can become very expensive, unstable, or gloriously overhyped if you are not careful.

---

# 1. VAE

Variational Autoencoders learn a probabilistic latent space.

Why they matter:

- generation
- interpolation
- structured latent representations

Tradeoff:

- smoother latent space
- often blurrier outputs than more aggressive generative methods

---

# 2. GAN

GANs use:

- a generator
- a discriminator

The generator tries to fool the discriminator.

The discriminator tries to spot fake samples.

Why GANs became famous:

- sharp image generation

Why they became infamous:

- unstable training
- mode collapse

Very glamorous.
Very temperamental.

---

# 3. Diffusion Models

Diffusion models learn to reverse a gradual noise process.

They became dominant in image generation because they often produce:

- strong quality
- stable training
- very flexible generation behavior

Tradeoff:

- slower sampling than some alternatives

---

# 4. Autoregressive Models

These generate one token or piece at a time.

Very common in:

- language generation
- code models
- some multimodal settings

They are strong because likelihood-based training is clean and scalable.
