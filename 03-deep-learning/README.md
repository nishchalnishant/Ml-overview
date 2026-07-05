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
| [components/02-activation-functions.md](components/02-activation-functions.md) | ReLU, sigmoid, tanh, GELU, swish — when and why |
| [components/10-attention.md](components/10-attention.md) | Scaled dot-product attention, multi-head attention |
| [components/12-autoencoders.md](components/12-autoencoders.md) | Encoder-decoder structure, latent space, VAEs |
| [components/01-backpropagation.md](components/01-backpropagation.md) | Chain rule, gradient flow, vanishing/exploding gradients |
| [components/07-normalization.md](components/07-normalization.md) | BatchNorm, LayerNorm, RMSNorm |
| [components/09-rnn-lstm-gru.md](components/09-rnn-lstm-gru.md) | Recurrent architectures, gating, when they still matter |
| [components/04-hidden-layers.md](components/04-hidden-layers.md) | Layer types, depth vs width tradeoffs |
| [components/05-loss-functions.md](components/05-loss-functions.md) | MSE, cross-entropy, focal loss, contrastive |
| [components/16-model-compression.md](components/16-model-compression.md) | Pruning, quantization, distillation |
| [components/17-quantization-pruning-detailed.md](components/17-quantization-pruning-detailed.md) | INT8/INT4 quantization, structured/unstructured pruning |
| [components/06-optimisers.md](components/06-optimisers.md) | SGD, Adam, AdaGrad, learning rate schedules |
| [components/08-regularization.md](components/08-regularization.md) | Dropout, batch norm, weight decay, early stopping |
| [components/03-weight-initialization.md](components/03-weight-initialization.md) | Xavier, He, why bad init breaks deep nets |
| [components/11-transformers.md](components/11-transformers.md) | Transformer architecture end to end, incl. MoE FFN |
| [components/14-scaling-laws-and-chinchilla.md](components/14-scaling-laws-and-chinchilla.md) | Compute-optimal scaling, Chinchilla |
| [components/13-distributed-training-and-parallelism.md](components/13-distributed-training-and-parallelism.md) | Data/tensor/pipeline parallelism, ZeRO |
| [components/15-instruction-tuning-and-alignment.md](components/15-instruction-tuning-and-alignment.md) | SFT, RLHF, DPO |
| [methods/](methods/README.md) | Application areas: NLP, CV, generative models, time series, video, 3D, graphs |
| [methods/03-computer-vision.md](methods/03-computer-vision.md) | CNNs, ViTs, object detection |
| [methods/04-segmentation.md](methods/04-segmentation.md) | Semantic/instance/panoptic segmentation, pose estimation |
| [methods/05-open-vocabulary-detection-and-tracking.md](methods/05-open-vocabulary-detection-and-tracking.md) | Grounding DINO, SAM, multi-object tracking, OCR |
| [methods/08-generative-models.md](methods/08-generative-models.md) | VAEs, GANs, diffusion, autoregressive generation |
| [methods/01-nlp-fundamentals.md](methods/01-nlp-fundamentals.md) | Bag-of-Words through Transformers |
| [methods/02-nlp-advanced.md](methods/02-nlp-advanced.md) | Pretraining objectives, transfer learning |
| [methods/09-time-series.md](methods/09-time-series.md) | Forecasting through the deep-learning lens |
| [methods/07-video-understanding.md](methods/07-video-understanding.md) | Action recognition, video transformers |
| [methods/06-3d-vision.md](methods/06-3d-vision.md) | Point clouds, NeRF, 3D reconstruction |
| [methods/11-dynamic-graphs.md](methods/11-dynamic-graphs.md) | Temporal GNNs, graph generation |
| [methods/10-metric-learning.md](methods/10-metric-learning.md) | Contrastive/triplet losses, retrieval embeddings |
| [transfer-learning.md](02-transfer-learning.md) | Fine-tuning, DANN, few-shot, MAML, zero-shot |
| [pytorch-fundamentals.md](01-pytorch-fundamentals.md) | Dense practical reference: tensors, autograd, training loop, debugging |
| [deep-learning-cheatsheet.md](deep-learning-cheatsheet.md) | 10-Minute Revision Card / Cheatsheet |
| [deep-learning-flashcards.md](deep-learning-flashcards.md) | Consolidated flashcards for the module |
| [mcp.md](mcp.md) | Tooling and protocol notes for model-connected workflows |

---

## Start Here

If you want the highest-value path first:

1. [components/02-activation-functions.md](components/02-activation-functions.md)
2. [components/01-backpropagation.md](components/01-backpropagation.md)
3. [components/06-optimisers.md](components/06-optimisers.md)
4. [components/08-regularization.md](components/08-regularization.md)
5. [components/10-attention.md](components/10-attention.md)
6. [components/11-transformers.md](components/11-transformers.md)
7. [pytorch-fundamentals.md](01-pytorch-fundamentals.md)

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
