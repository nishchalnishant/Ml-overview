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

> Full per-file tables live in [components/README.md](components/README.md) and [methods/README.md](methods/README.md) — those are the source of truth; this table is a curated subset. If a file is missing from here, check the sub-folder README first.

| File / Folder | What it covers |
| :--- | :--- |
| [components/](components/README.md) | Core building blocks: activations, loss, backprop, optimizers, normalization, attention, transformers, RNN/LSTM/GRU, scaling laws, distributed training, instruction tuning, quantization |
| [components/activation-functions.md](components/activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — when and why |
| [components/attention.md](components/attention.md) | Scaled dot-product attention, multi-head attention |
| [components/autoencoders.md](components/autoencoders.md) | Encoder-decoder structure, latent space, VAEs |
| [components/backpropagation.md](components/backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients |
| [components/normalization.md](components/normalization.md) | BatchNorm, LayerNorm, RMSNorm |
| [components/rnn-lstm-gru.md](components/rnn-lstm-gru.md) | Recurrent architectures, gating, when they still matter |
| [components/hidden-layers.md](components/hidden-layers.md) | Layer types, depth vs width tradeoffs |
| [components/loss-functions.md](components/loss-functions.md) | MSE, cross-entropy, focal loss, contrastive |
| [components/model-compression.md](components/model-compression.md) | Pruning, quantization, distillation |
| [components/quantization-pruning-detailed.md](components/quantization-pruning-detailed.md) | INT8/INT4 quantization, structured/unstructured pruning |
| [components/optimisers.md](components/optimisers.md) | SGD, Adam, AdaGrad, learning rate schedules |
| [components/regularization.md](components/regularization.md) | Dropout, batch norm, weight decay, early stopping |
| [components/weight-initialization.md](components/weight-initialization.md) | Xavier, He, why bad init breaks deep nets |
| [components/transformers.md](components/transformers.md) | Transformer architecture end to end, incl. MoE FFN |
| [components/scaling-laws-and-chinchilla.md](components/scaling-laws-and-chinchilla.md) | Compute-optimal scaling, Chinchilla |
| [components/distributed-training-and-parallelism.md](components/distributed-training-and-parallelism.md) | Data/tensor/pipeline parallelism, ZeRO |
| [components/instruction-tuning-and-alignment.md](components/instruction-tuning-and-alignment.md) | SFT, RLHF, DPO |
| [methods/](methods/README.md) | Application areas: NLP, CV, generative models, time series, video, 3D, graphs |
| [methods/computer-vision.md](methods/computer-vision.md) | CNNs, ViTs, object detection |
| [methods/segmentation.md](methods/segmentation.md) | Semantic/instance/panoptic segmentation, pose estimation |
| [methods/open-vocabulary-detection-and-tracking.md](methods/open-vocabulary-detection-and-tracking.md) | Grounding DINO, SAM, multi-object tracking, OCR |
| [methods/generative-models.md](methods/generative-models.md) | VAEs, GANs, diffusion, autoregressive generation |
| [methods/nlp-fundamentals.md](methods/nlp-fundamentals.md) | Bag-of-Words through Transformers |
| [methods/nlp-advanced.md](methods/nlp-advanced.md) | Pretraining objectives, transfer learning |
| [methods/time-series.md](methods/time-series.md) | Forecasting through the deep-learning lens |
| [methods/video-understanding.md](methods/video-understanding.md) | Action recognition, video transformers |
| [methods/3d-vision.md](methods/3d-vision.md) | Point clouds, NeRF, 3D reconstruction |
| [methods/dynamic-graphs.md](methods/dynamic-graphs.md) | Temporal GNNs, graph generation |
| [methods/metric-learning.md](methods/metric-learning.md) | Contrastive/triplet losses, retrieval embeddings |
| [transfer-learning.md](transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
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
