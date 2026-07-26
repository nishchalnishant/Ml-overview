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

**What lives here:** MLP, CNN, RNN/LSTM/GRU, attention, transformers, GNN. SSM/Mamba is covered as
part of [`01-rnn-lstm-gru.md`](01-rnn-lstm-gru.md)
(the RNN-successor framing), not as a standalone file here.

| File | What it covers |
| :--- | :--- |
| [01-rnn-lstm-gru.md](01-rnn-lstm-gru.md) | Vanilla RNN, LSTM, GRU — gating, BPTT, vanishing gradients |
| [02-attention.md](02-attention.md) | Attention mechanics, KV cache, MQA/GQA |
| [03-transformers.md](03-transformers.md) | Full transformer architecture, positional encoding, architecture milestones |
| [04-dl-architectures.md](04-dl-architectures.md) | CNN/RNN/Transformer/GAN/Autoencoder/Diffusion — interview-framing tour + selection framework |
| [05-gnn.md](05-gnn.md) | Message passing, GCN, GraphSAGE, GAT, GIN, over-smoothing |

**Split completed.** `04-dl-architectures.md` §4 (GAN) and §6 (Diffusion) now point at
[`../08-generative/02-gans.md`](../08-generative/02-gans.md) and
[`../08-generative/03-diffusion.md`](../08-generative/03-diffusion.md) for depth, keeping a
selection summary here. §5 (Autoencoder/VAE) **stayed** — it is the interview-framing modality
of [`../08-generative/01-autoencoders.md`](../08-generative/01-autoencoders.md), not a duplicate
of it, and moving it would have collided with existing deep-dive content.
