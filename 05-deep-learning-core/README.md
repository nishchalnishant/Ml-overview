---
module: Deep Learning Core
topic: Deep Learning
subtopic: ""
status: unread
tags: [deeplearning, ml, deep-learning]
---
# Deep Learning

This folder is the deep-learning version of a good Azure DevOps project:

- clear building blocks
- reusable components
- strong deployment instincts
- fewer mysterious things happening in production

If you already come from **Azure + DevOps**, here is the easiest bridge:

- **model architecture** = service design
- **training loop** = build pipeline
- **validation** = release gates
- **inference** = deployed runtime
- **monitoring** = observability plus business outcomes

Deep learning is just software delivery with:

- tensors
- gradients
- GPUs
- and a little more emotional volatility

---

## Navigation

> Full per-file table for components lives in [components/README.md](README.md) — that is the source of truth; this table is a curated subset. If a file is missing from here, check the sub-folder README first.

| File / Folder | What it covers |
| :--- | :--- |
| [components/](README.md) | Core building blocks: activations, loss, backprop, optimizers, normalization, attention, transformers, RNN/LSTM/GRU, distributed training, quantization |
| [components/02-activation-functions.md](02-activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — when and why |
| [components/10-attention.md](../06-architectures/02-attention.md) | Scaled dot-product attention, multi-head attention |
| [components/12-autoencoders.md](../08-generative/01-autoencoders.md) | Encoder-decoder structure, latent space, VAEs |
| [components/01-backpropagation.md](01-backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients |
| [components/07-normalization.md](07-normalization.md) | BatchNorm, LayerNorm, RMSNorm |
| [components/09-rnn-lstm-gru.md](../06-architectures/01-rnn-lstm-gru.md) | Recurrent architectures, gating, when they still matter |
| [components/04-hidden-layers.md](04-hidden-layers.md) | Layer types, depth vs width tradeoffs |
| [components/05-loss-functions.md](05-loss-functions.md) | MSE, cross-entropy, focal loss, contrastive |
| [components/16-model-compression.md](../12-systems-and-scale/01-model-compression.md) | Compression overview: pruning, quantization, distillation — start here |
| [components/17-quantization-pruning-detailed.md](../12-systems-and-scale/02-quantization-pruning.md) | Math/code deep-dive on quantization and pruning internals — read after 16 |
| [components/06-optimisers.md](06-optimisers.md) | SGD, Adam, AdaGrad, learning rate schedules |
| [components/08-regularization.md](08-regularization.md) | Dropout, batch norm, weight decay, early stopping |
| [components/03-weight-initialization.md](03-weight-initialization.md) | Xavier, He, why bad init breaks deep nets |
| [components/11-transformers.md](../06-architectures/03-transformers.md) | Transformer architecture end to end, incl. MoE FFN |
| [components/18-llm-serving.md](../12-systems-and-scale/03-llm-serving.md) | LLM inference serving, batching, KV cache |
| [methods/](methods/) | Application areas: NLP, CV, time series |
| [methods/03-computer-vision.md](../07-domains/04-computer-vision.md) | CNNs, ViTs, object detection |
| [methods/01-nlp-fundamentals.md](../07-domains/01-nlp-fundamentals.md) | Bag-of-Words through Transformers |
| [methods/09-time-series.md](../07-domains/06-time-series.md) | Forecasting through the deep-learning lens |
| [transfer-learning.md](10-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [pytorch-fundamentals.md](09-pytorch-fundamentals.md) | Dense practical reference: tensors, autograd, training loop, debugging |
| [deep-learning-cheatsheet.md](_cheatsheet.md) | 10-Minute Revision Card / Cheatsheet |
| [deep-learning-flashcards.md](_flashcards.md) | Consolidated flashcards for the module |

---

## Start Here

If you want the highest-value path first:

1. [components/02-activation-functions.md](02-activation-functions.md)
2. [components/01-backpropagation.md](01-backpropagation.md)
3. [components/06-optimisers.md](06-optimisers.md)
4. [components/08-regularization.md](08-regularization.md)
5. [components/10-attention.md](../06-architectures/02-attention.md)
6. [components/11-transformers.md](../06-architectures/03-transformers.md)
7. [pytorch-fundamentals.md](09-pytorch-fundamentals.md)

That gives you the fundamentals before the fancier methods.

---

## Quick Thought Experiment

If a deep model performs beautifully in training but fails in deployment, what do you inspect first?

- the architecture?
- the optimizer?
- the data path?
- the serving mismatch?

If your DevOps brain answered:

> "data path and serving mismatch first"

excellent.

That instinct will save you a lot of drama.
