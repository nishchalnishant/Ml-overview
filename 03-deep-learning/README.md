---
module: Deep Learning
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

| File / Folder | What it covers |
| :--- | :--- |
| [components/](components/README.md) | Core building blocks: activations, loss, backprop, optimizers, attention, regularization, transformers |
| [components/activation-functions.md](components/activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — when and why |
| [components/attention.md](components/attention.md) | Scaled dot-product attention, multi-head attention |
| [components/autoencoders.md](components/autoencoders.md) | Encoder-decoder structure, latent space, VAEs |
| [components/backpropagation.md](components/backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients |
| [components/hidden-layers.md](components/hidden-layers.md) | Layer types, depth vs width tradeoffs |
| [components/loss-functions.md](components/loss-functions.md) | MSE, cross-entropy, focal loss, contrastive |
| [components/model-compression.md](components/model-compression.md) | Pruning, quantization, distillation |
| [components/optimisers.md](components/optimisers.md) | SGD, Adam, AdaGrad, learning rate schedules |
| [components/regularization.md](components/regularization.md) | Dropout, batch norm, weight decay, early stopping |
| [components/transformers.md](components/transformers.md) | Transformer architecture end to end |
| [methods/](methods/README.md) | Application areas: NLP, CV, generative models, time series |
| [methods/computer-vision.md](methods/computer-vision.md) | CNNs, ViTs, detection, segmentation, OCR |
| [methods/generative-models.md](methods/generative-models.md) | VAEs, GANs, diffusion, autoregressive generation |
| [methods/nlp-fundamentals.md](methods/nlp-fundamentals.md) | Bag-of-Words through Transformers |
| [methods/time-series.md](methods/time-series.md) | Forecasting through the deep-learning lens |
| [pytorch-fundamentals.md](pytorch-fundamentals.md) | Dense practical reference: tensors, autograd, training loop, debugging |
| [deep-learning-cheatsheet.md](deep-learning-cheatsheet.md) | 10-Minute Revision Card / Cheatsheet |
| [deep-learning-flashcards.md](deep-learning-flashcards.md) | Consolidated flashcards for the module |
| [mcp.md](mcp.md) | Tooling and protocol notes for model-connected workflows |

---

## Start Here

If you want the highest-value path first:

1. [components/activation-functions.md](components/activation-functions.md)
2. [components/backpropagation.md](components/backpropagation.md)
3. [components/optimisers.md](components/optimisers.md)
4. [components/regularization.md](components/regularization.md)
5. [components/attention.md](components/attention.md)
6. [components/transformers.md](components/transformers.md)
7. [pytorch-fundamentals.md](pytorch-fundamentals.md)

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
