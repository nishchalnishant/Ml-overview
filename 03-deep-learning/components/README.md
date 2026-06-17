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
| [activation-functions.md](activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — purpose, failure modes, when to use which |
| [attention.md](attention.md) | Scaled dot-product attention, multi-head attention, why it replaced RNNs |
| [autoencoders.md](autoencoders.md) | Encoder-decoder structure, latent space, VAEs, reconstruction loss |
| [backpropagation.md](backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients, gradient clipping |
| [distributed-training-and-parallelism.md](distributed-training-and-parallelism.md) | Data/tensor/pipeline parallelism, ZeRO, sharding strategies |
| [hidden-layers.md](hidden-layers.md) | Layer types, depth vs width, capacity tradeoffs |
| [instruction-tuning-and-alignment.md](instruction-tuning-and-alignment.md) | SFT, RLHF, DPO — turning a base model into an assistant |
| [loss-functions.md](loss-functions.md) | MSE, cross-entropy, focal loss, contrastive, triplet — when each applies |
| [model-compression.md](model-compression.md) | Pruning, quantization, knowledge distillation, deployment tradeoffs |
| [normalization.md](normalization.md) | BatchNorm, LayerNorm, RMSNorm — why and where each is used |
| [optimisers.md](optimisers.md) | SGD, momentum, Adam, AdaGrad, learning rate schedules, warmup |
| [quantization-pruning-detailed.md](quantization-pruning-detailed.md) | INT8/INT4 quantization, structured/unstructured pruning, accuracy tradeoffs |
| [regularization.md](regularization.md) | Dropout, batch norm, layer norm, weight decay, early stopping |
| [rnn-lstm-gru.md](rnn-lstm-gru.md) | Recurrent architectures, gating, vanishing gradients, when they still matter |
| [scaling-laws-and-chinchilla.md](scaling-laws-and-chinchilla.md) | Compute-optimal scaling, Chinchilla, parameters vs data tradeoffs |
| [transformers.md](transformers.md) | Full Transformer architecture: embeddings, attention, FFN, positional encoding |
| [weight-initialization.md](weight-initialization.md) | Xavier, He, and why bad init breaks deep nets |

---

## Start With These

If you can explain these five cleanly, you can survive most deep-learning interviews:

1. [backpropagation.md](backpropagation.md)
2. [activation-functions.md](activation-functions.md)
3. [optimisers.md](optimisers.md)
4. [regularization.md](regularization.md)
5. [attention.md](attention.md)

Then add [transformers.md](transformers.md) and you're prepared for the architecture questions.

---

## Back to top

[Deep Learning README](../README.md)
