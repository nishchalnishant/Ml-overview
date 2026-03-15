# Deep Learning

Deep learning uses neural networks with many layers to learn hierarchical representations from data. This section covers the **parts** of deep learning (how networks are built and trained) and **methods** (how they are applied to vision, language, time series, and generation).

---

## Parts of deep learning

Core building blocks and training machinery:

- **[Backpropagation](parts-of-deep-learning/backpropagation.md)** — How gradients flow backward through the network to update weights.
- **[Hidden layers](parts-of-deep-learning/hidden-layers.md)** — Intermediate representations between input and output.
- **[Activation functions](parts-of-deep-learning/activation-functions.md)** — Non-linearities (ReLU, GELU, softmax, etc.) that enable learning complex functions.
- **[Loss functions](parts-of-deep-learning/loss-functions.md)** — Objectives to minimize (cross-entropy, MSE, etc.).
- **[Optimizers](parts-of-deep-learning/optimizers.md)** — SGD, Adam, and variants that update parameters from gradients.
- **[Regularization](parts-of-deep-learning/regularization.md)** — Dropout, weight decay, and other techniques to reduce overfitting.
- **[Attention](parts-of-deep-learning/attention.md)** — Mechanism to focus on relevant parts of the input; core of transformers.
- **[Transformers](parts-of-deep-learning/transformers.md)** — Encoder/decoder and decoder-only architectures underlying LLMs.
- **[Autoencoders](parts-of-deep-learning/autoencoders.md)** — Unsupervised models that learn compact representations.

---

## Deep learning methods

Application areas:

- **[Computer vision](deep-learning-methods/computer-vision.md)** — CNNs, ViT, detection, segmentation.
- **[NLP](deep-learning-methods/nlp.md)** — From RNNs to transformers and LLMs.
- **[Time series](deep-learning-methods/time-series.md)** — Forecasting and representation learning.
- **[Generative models](deep-learning-methods/generative-models.md)** — VAEs, GANs, diffusion, autoregressive LLMs.

---

## From classical ML to modern AI

Deep learning extends classical ML with:

- **Representation learning**: the network learns features from raw or weakly labeled data.
- **Scale**: large models (transformers) and large data enable emergent capabilities (reasoning, tool use, instruction following).
- **Unified architectures**: the same transformer backbone is used for text, vision, and multimodal tasks.

For a path from fundamentals to modern AI systems (LLMs, RAG, agents), see [MODERN_AI_ENGINEER_ROADMAP.md](../MODERN_AI_ENGINEER_ROADMAP.md).
