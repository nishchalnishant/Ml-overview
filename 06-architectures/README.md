---
module: Architectures
topic: Overview
subtopic: ""
status: unread
tags: [cnn, rnn, transformers, index]
prerequisites: [backpropagation, linear-algebra]
---
# Architectures

Model *architectures*, as distinct from the training *components* in `05-deep-learning-core/`. Components answer "how does it learn"; architectures answer "what shape is it".

**What lives here:** MLP, CNN, RNN/LSTM/GRU, attention, transformers, GNN, SSM/Mamba.

**Split completed.** `04-dl-architectures.md` §4 (GAN) and §6 (Diffusion) now point at
[`../08-generative/02-gans.md`](../08-generative/02-gans.md) and
[`../08-generative/03-diffusion.md`](../08-generative/03-diffusion.md) for depth, keeping a
selection summary here. §5 (Autoencoder/VAE) **stayed** — it is the interview-framing modality
of [`../08-generative/01-autoencoders.md`](../08-generative/01-autoencoders.md), not a duplicate
of it, and moving it would have collided with existing deep-dive content.
