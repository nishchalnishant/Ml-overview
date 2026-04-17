# Deep Learning Architectures

This file is for fast architecture comparison.

The goal is not to memorize every paper title.
It is to know:

- what each architecture is good at
- what problem it was invented to solve
- what tradeoff it introduces

---

## 1. CNN

Best for:

- images
- spatial hierarchy

Why:

- locality
- parameter sharing

Short answer:

CNNs work well for images because they match the structure of visual data.

---

## 2. RNN / LSTM / GRU

Best for:

- sequential data

Why:

- they carry state across time

Weakness:

- slow
- vanishing gradients
- long-range memory pain

---

## 3. Transformer

Best for:

- long-range context
- parallel training
- scale

Why it won:

- attention handles relationships directly
- removes recurrent bottleneck

Weakness:

- expensive at long context lengths

---

## 4. GAN

Great for:

- sharp generation

Pain points:

- unstable training
- mode collapse

---

## 5. Autoencoder / VAE

Good for:

- compression
- denoising
- latent representation learning

VAE adds:

- probabilistic latent space

Which helps with smooth generation and interpolation.
