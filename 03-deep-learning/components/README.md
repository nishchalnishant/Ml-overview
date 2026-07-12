---
module: Deep Learning
topic: Parts of Deep Learning
subtopic: ""
status: unread
tags: [deeplearning, ml, parts-of-deep-learning]
---
# Parts of Deep Learning

This folder is the "open the hood" section.

If the methods folder shows you the cars, this one shows you:

- the engine
- the gearbox
- the brakes
- the things that explode if you ignore them

These are the core concepts every serious deep-learning conversation keeps coming back to.

---

## Files in This Folder

| File | What it covers |
| :--- | :--- |
| [activation-functions.md](02-activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — purpose, failure modes, when to use which |
| [attention.md](10-attention.md) | Scaled dot-product attention, multi-head attention, why it replaced RNNs |
| [autoencoders.md](12-autoencoders.md) | Encoder-decoder structure, latent space, VAEs, reconstruction loss |
| [backpropagation.md](01-backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients, gradient clipping |
| [hidden-layers.md](04-hidden-layers.md) | Layer types, depth vs width, capacity tradeoffs |
| [loss-functions.md](05-loss-functions.md) | MSE, cross-entropy, focal loss, contrastive, triplet — when each applies |
| [model-compression.md](16-model-compression.md) | Compression landscape overview: quantization, distillation, pruning, low-rank factorization, speculative decoding, combination pipelines — read this first |
| [normalization.md](07-normalization.md) | BatchNorm, LayerNorm, RMSNorm — why and where each is used |
| [optimisers.md](06-optimisers.md) | SGD, momentum, Adam, AdaGrad, learning rate schedules, warmup |
| [quantization-pruning-detailed.md](17-quantization-pruning-detailed.md) | Math and code deep-dive on quantization/pruning internals (GPTQ, AWQ, QAT, IMP) — read after 16 for implementation depth |
| [regularization.md](08-regularization.md) | Dropout, batch norm, layer norm, weight decay, early stopping |
| [rnn-lstm-gru.md](09-rnn-lstm-gru.md) | Recurrent architectures, gating, vanishing gradients, when they still matter |
| [llm-serving.md](18-llm-serving.md) | LLM inference serving, batching, KV cache, continuous batching |
| [transformers.md](11-transformers.md) | Full Transformer architecture: embeddings, attention, FFN, positional encoding |
| [weight-initialization.md](03-weight-initialization.md) | Xavier, He, and why bad init breaks deep nets |

---

## Start With These

If you can explain these five cleanly, you can survive most deep-learning interviews:

1. [backpropagation.md](01-backpropagation.md)
2. [activation-functions.md](02-activation-functions.md)
3. [optimisers.md](06-optimisers.md)
4. [regularization.md](08-regularization.md)
5. [attention.md](10-attention.md)

Then add [transformers.md](11-transformers.md) and you're prepared for the architecture questions.

---

## Back to top

[Deep Learning README](../README.md)
