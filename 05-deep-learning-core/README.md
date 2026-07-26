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

Building blocks that every architecture is assembled from — activations, loss, backprop, optimizers, normalization, regularization, PyTorch mechanics, transfer learning. Attention, RNN/LSTM/GRU, and Transformers live in [06-architectures/](../06-architectures/); application areas (NLP, CV, time series) live in [07-domains/](../07-domains/).

| File | What it covers |
| :--- | :--- |
| [01-backpropagation.md](01-backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients |
| [02-activation-functions.md](02-activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — when and why |
| [03-weight-initialization.md](03-weight-initialization.md) | Xavier, He, why bad init breaks deep nets |
| [04-hidden-layers.md](04-hidden-layers.md) | Layer types, depth vs width tradeoffs, residual connections |
| [05-loss-functions.md](05-loss-functions.md) | MSE, cross-entropy, focal loss, contrastive |
| [06-optimisers.md](06-optimisers.md) | SGD, Adam, AdaGrad, learning rate schedules |
| [07-normalization.md](07-normalization.md) | BatchNorm, LayerNorm, RMSNorm |
| [08-regularization.md](08-regularization.md) | Dropout, weight decay, early stopping |
| [09-pytorch-fundamentals.md](09-pytorch-fundamentals.md) | Dense practical reference: tensors, autograd, training loop, debugging |
| [10-transfer-learning.md](10-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [11-deep-learning-fundamentals.md](11-deep-learning-fundamentals.md) | Synthesis/overview tying the above together |

**Elsewhere but closely related:**

| File | What it covers |
| :--- | :--- |
| [../06-architectures/01-rnn-lstm-gru.md](../06-architectures/01-rnn-lstm-gru.md) | Recurrent architectures, gating, when they still matter |
| [../06-architectures/02-attention.md](../06-architectures/02-attention.md) | Scaled dot-product attention, multi-head attention |
| [../06-architectures/03-transformers.md](../06-architectures/03-transformers.md) | Transformer architecture end to end, incl. MoE FFN |
| [../08-generative/01-autoencoders.md](../08-generative/01-autoencoders.md) | Encoder-decoder structure, latent space, VAEs |
| [../12-systems-and-scale/01-model-compression.md](../12-systems-and-scale/01-model-compression.md) | Compression overview: pruning, quantization, distillation — start here |
| [../12-systems-and-scale/02-quantization-pruning.md](../12-systems-and-scale/02-quantization-pruning.md) | Math/code deep-dive on quantization and pruning internals |
| [../12-systems-and-scale/03-llm-serving.md](../12-systems-and-scale/03-llm-serving.md) | LLM inference serving, batching, KV cache |
| [../07-domains/04-computer-vision.md](../07-domains/04-computer-vision.md) | CNNs, ViTs, object detection |
| [../07-domains/01-nlp-fundamentals.md](../07-domains/01-nlp-fundamentals.md) | Bag-of-Words through Transformers |
| [../07-domains/06-time-series.md](../07-domains/06-time-series.md) | Forecasting through the deep-learning lens |

**Modalities:** [_cheatsheet.md](_cheatsheet.md) · [_flashcards.md](_flashcards.md) · [_comparison.md](_comparison.md) · [_interview-questions.md](_interview-questions.md)

---

## Start Here

If you want the highest-value path first:

1. [02-activation-functions.md](02-activation-functions.md)
2. [01-backpropagation.md](01-backpropagation.md)
3. [06-optimisers.md](06-optimisers.md)
4. [08-regularization.md](08-regularization.md)
5. [../06-architectures/02-attention.md](../06-architectures/02-attention.md)
6. [../06-architectures/03-transformers.md](../06-architectures/03-transformers.md)
7. [09-pytorch-fundamentals.md](09-pytorch-fundamentals.md)

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
