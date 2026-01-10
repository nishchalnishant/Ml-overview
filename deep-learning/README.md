# Deep Learning

## Overview

Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data. This directory contains comprehensive notes on deep learning concepts, architectures, and applications.

---

## Table of Contents

1. **[Core Components](parts-of-deep-learning/)** - Neural network building blocks
   - [Layers](parts-of-deep-learning/) - Dense, Conv, RNN, Attention
   - [Activation Functions](parts-of-deep-learning/) - ReLU, GELU, Sigmoid, Softmax
   - [Loss Functions](parts-of-deep-learning/) - MSE, Cross-Entropy, Hinge
   - [Optimizers](parts-of-deep-learning/) - SGD, Adam, AdamW
   
2. **[Methods & Applications](deep-learning-methods/)** - Modern architectures
   - [Computer Vision](deep-learning-methods/computer-vision.md) - CNNs, ResNet, ViT
   - [Natural Language Processing](deep-learning-methods/) - Transformers, BERT, GPT
   - [Generative Models](deep-learning-methods/generative-models.md) - GANs, VAEs, Diffusion

3. **[Advanced Topics](mcp.md)** - Production and scaling

---

## When to Use Deep Learning

### Choose Deep Learning When:
✅ **Large Dataset:** 100K+ samples available  
✅ **Unstructured Data:** Images, text, audio, video  
✅ **Complex Patterns:** Non-linear relationships  
✅ **Performance Priority:** Accuracy over interpretability  
✅ **Computational Resources:** Access to GPUs/TPUs  

### Choose Traditional ML When:
❌ **Small Dataset:** < 10K samples  
❌ **Tabular Data:** Structured features  
❌ **Need Interpretability:** Medical, financial domains  
❌ **Limited Resources:** CPU-only environment  
❌ **Quick Iteration:** Rapid prototyping needed  

---

## Quick Comparison: Traditional ML vs Deep Learning

| **Aspect** | **Traditional ML** | **Deep Learning** |
|-----------|-------------------|-------------------|
| **Data Size** | 1K-100K samples | 100K-1M+ samples |
| **Feature Engineering** | Manual (critical) | Automatic (learned) |
| **Training Time** | Minutes to hours | Hours to days |
| **Hardware** | CPU sufficient | GPU/TPU required |
| **Interpretability** | High | Low (black box) |
| **Performance Ceiling** | Plateaus with more data | Improves with scale |
| **Best For** | Tabular, structured | Images, text, audio |

---

## Modern Deep Learning Timeline

```
1998: LeNet (CNNs for digit recognition)
2012: AlexNet (Deep learning breakthrough on ImageNet)
2014: VGG, GoogLeNet (Very deep networks)
2015: ResNet (Skip connections, 152 layers)
2017: Transformer (Attention is All You Need)
2018: BERT (Bidirectional language understanding)
2018: GPT (Autoregressive language generation)
2020: GPT-3 (175B parameters, few-shot learning)
2020: Vision Transformer (Transformers for images)
2021: DALL-E (Text-to-image generation)
2022: Stable Diffusion (Open-source generation)
2023: GPT-4, LLaMA (Multimodal, efficient LLMs)
```

---

## Core Architecture Families

### 1. Convolutional Neural Networks (CNNs)
**Purpose:** Extract spatial features from images

**Key Components:**
- **Conv Layers:** Learn local patterns (edges, textures, shapes)
- **Pooling:** Downsample feature maps (Max, Average)
- **Fully Connected:** Final classification/regression

**Modern Architectures:**
- **ResNet:** Skip connections solve vanishing gradients
- **EfficientNet:** Compound scaling (depth + width + resolution)
- **Vision Transformer (ViT):** Transformers replacing convolutions

**Applications:**
- Image classification
- Object detection (YOLO, Faster R-CNN)
- Semantic segmentation (U-Net, Mask R-CNN)
- Face recognition

---

### 2. Recurrent Neural Networks (RNNs)
**Purpose:** Process sequential data (time series, text)

**Variants:**
- **Vanilla RNN:** Simple recurrence (vanishing gradient issues)
- **LSTM:** Long Short-Term Memory (gates to control information flow)
- **GRU:** Gated Recurrent Unit (simplified LSTM)

**Limitations:**
- Sequential processing (slow, can't parallelize)
- Difficulty capturing long-range dependencies
- **Replaced by:** Transformers for most NLP tasks

**Still Used For:**
- Time series forecasting
- Audio processing
- Video analysis

---

### 3. Transformers
**Purpose:** Capture long-range dependencies via self-attention

**Key Innovation:**
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**Advantages:**
- Fully parallelizable (faster training than RNNs)
- Captures global dependencies
- Scales extremely well with data and compute

**Architecture Types:**

| **Type** | **Example** | **Use Case** |
|---------|-----------|-------------|
| **Encoder-only** | BERT, RoBERTa | Text understanding, classification |
| **Decoder-only** | GPT, LLaMA | Text generation, completion |
| **Encoder-Decoder** | T5, BART | Translation, summarization |

**Modern Applications:**
- **NLP:** GPT-4, BERT, T5
- **Vision:** ViT, CLIP, Swin Transformer
- **Multimodal:** CLIP, Flamingo, GPT-4V

---

### 4. Generative Models
**Purpose:** Create new data samples

**Architecture Types:**

| **Model** | **How It Works** | **Best For** |
|----------|-----------------|-------------|
| **GAN** | Generator vs Discriminator (adversarial) | Image generation, style transfer |  
| **VAE** | Encoder-decoder with latent space | Smooth interpolation, data augmentation |
| **Diffusion** | Iterative denoising process | High-quality image generation (DALL-E, Midjourney) |
| **Autoregressive** | Predict next token sequentially | Text generation (GPT), Image generation (PixelCNN) |

**Applications:**
- Image generation (Stable Diffusion, Midjourney)
- Text generation (GPT, Claude)
- Code generation (GitHub Copilot)
- Drug discovery
- Data augmentation

---

## Training Best Practices

### Preventing Overfitting
1. **Regularization:**
   - Dropout (0.2-0.5)
   - Weight decay / L2 regularization
   - Batch normalization
   - Early stopping

2. **Data Augmentation:**
   - **Images:** Rotation, flipping, cropping, color jitter
   - **Text:** Back-translation, synonym replacement
   - **MixUp/CutMix:** Mix training examples

3. **Architecture Choices:**
   - Smaller networks
   - Reduce layer depth/width
   - Use pretrained models (transfer learning)

### Optimization Techniques
1. **Learning Rate Strategies:**
   - Warm-up: Start low, gradually increase
   - Decay: Reduce over time (step, exponential, cosine)
   - Cyclical: Oscillate between bounds

2. **Gradient Management:**
   - Gradient clipping (prevent exploding gradients)
   - Gradient accumulation (simulate larger batches)
   - Mixed precision training (FP16 + FP32)

3. **Advanced Optimizers:**
   - Adam/AdamW (default choice)
   - SGD + Momentum (better generalization for fine-tuning)
   - LAMB, RAdam (large batch training)

---

## Transfer Learning & Fine-Tuning

**Why Transfer Learning?**
- Leverage pretrained models (trained on millions of images/text)
- Requires less data (can work with 100s of samples)
- Faster training (hours vs days)
- Better performance (pretrained features)

**Strategy:**
```
1. Load pretrained model (e.g., ResNet-50, BERT-base)
2. Freeze base layers
3. Replace/add task-specific head
4. Train head only (few epochs)
5. Optionally: Unfreeze top layers, fine-tune end-to-end
```

**Learning Rate Rules:**
- **Frozen base:** Regular LR (1e-3)
- **Fine-tuning:** Very low LR (1e-5 to 1e-6)

**Popular Pretrained Models:**
- **Vision:** ResNet, EfficientNet, ViT, CLIP
- **NLP:** BERT, RoBERTa, T5, GPT-2
- **Multimodal:** CLIP, ALIGN

---

## Hardware & Scaling

### Training Considerations

| **Aspect** | **Small Scale** | **Large Scale** |
|-----------|----------------|----------------|
| **Hardware** | Single GPU (RTX 3090, A100) | Multi-GPU cluster, TPU pods |
| **Batch Size** | 16-128 | 1024-4096+ |
| **Dataset Size** | 10K-1M samples | 10M-1B+ samples |
| **Training Time** | Hours to days | Days to weeks |
| **Techniques** | Standard backprop, data augmentation | Distributed training, gradient accumulation, mixed precision |

### Distributed Training Strategies
1. **Data Parallelism:** Split batch across GPUs
2. **Model Parallelism:** Split model across GPUs (for very large models)
3. **Pipeline Parallelism:** Split layers across GPUs
4. **ZeRO (Zero Redundancy Optimizer):** Optimize memory usage

### Inference Optimization
1. **Quantization:** INT8/INT4 (reduce precision)
2. **Pruning:** Remove unnecessary weights
3. **Knowledge Distillation:** Train smaller student model
4. **Model Compression:** ONNX, TensorRT, TFLite

---

## Common Interview Topics

### Conceptual Questions

**1. "Explain vanishing/exploding gradients"**
> Vanishing: Gradients become very small in early layers → no learning. Caused by deep networks with sigmoid/tanh. Fix: ReLU, batch norm, ResNet.
> Exploding: Gradients become very large → unstable training. Caused by deep RNNs. Fix: Gradient clipping, lower learning rate.

**2. "Why batch normalization helps?"**
> Normalizes layer inputs, reducing internal covariate shift. Allows higher learning rates, faster convergence, regularization effect. Used in CNNs.

**3. "Transformer vs RNN?"**
> Transformer: Parallel processing, better long-range, scalable. RNN: Sequential, slower, vanishing gradients. Transformers dominate NLP now.

**4. "How does attention work?"**
> Learns to focus on relevant parts of input. Query-Key-Value mechanism: Attention(Q,K,V) = softmax(QK^T/√d)V

**5. "Transfer learning vs training from scratch?"**
> Transfer: Use pretrained weights, faster, less data needed, better for small datasets. Scratch: More data needed, longer training, full control.

### Practical Questions

**6. "Model overfits on training data. What to do?"**
> 1. Get more data 2. Regularization (dropout, weight decay) 3. Data augmentation 4. Reduce model size 5. Early stopping 6. Batch normalization

**7. "How to choose batch size?"**
> Larger batches: Faster training, better GPU utilization, but need more memory and may hurt generalization. Start with largest that fits in memory (32, 64, 128). Use gradient accumulation if needed.

**8. "CNN vs Vision Transformer?"**
> CNN: Better for small-medium datasets, inductive bias (locality), less data needed. ViT: Better for large datasets (>1M images), more scalable, SOTA performance.

---

## Modern Tools & Frameworks

### Deep Learning Frameworks
- **PyTorch:** Research-friendly, dynamic graphs, most popular
- **TensorFlow/Keras:** Production-ready, static graphs, Google ecosystem
- **JAX:** High-performance, functional, research (Google)

### Pretrained Model Hubs
- **Hugging Face:** NLP models (BERT, GPT, T5)
- **timm (PyTorch Image Models):** Vision models (ResNet, EfficientNet, ViT)
- **TensorFlow Hub:** Google's model repository

### Training Tools
- **Weights & Biases:** Experiment tracking
- **MLflow:** Model versioning, tracking
- **TensorBoard:** Visualization
- **DeepSpeed:** Distributed training optimization
- **Lightning:** PyTorch wrapper for cleaner code

---

## Learning Path

### Beginner
1. Understand neural network basics (forward/backward prop)
2. Implement simple MLP from scratch
3. Learn one framework (PyTorch recommended)
4. Build CNN for image classification (MNIST, CIFAR-10)
5. Understand training process (loss, optimization, regularization)

### Intermediate
1. Transfer learning with pretrained models
2. Build RNN/LSTM for sequence tasks
3. Understand attention mechanism
4. Implement simple Transformer
5. Work with real datasets (ImageNet, COCO)
6. Learn hyperparameter tuning

### Advanced
1. Read research papers (Attention Is All You Need, ResNet, BERT)
2. Implement SOTA architectures from scratch
3. Distributed training on multiple GPUs
4. Fine-tune large language models
5. Contribute to open-source models
6. Deploy models to production

---

## Further Reading

- **Books:**
  - Deep Learning (Goodfellow, Bengio, Courville)
  - Dive into Deep Learning (d2l.ai)
  - PyTorch Deep Learning Course (Fast.ai)

- **Courses:**
  - CS231n: CNNs for Visual Recognition (Stanford)
  - CS224n: NLP with Deep Learning (Stanford)
  - Fast.ai Practical Deep Learning

- **Papers:**
  - ImageNet Classification with Deep CNNs (AlexNet)
  - Deep Residual Learning (ResNet)
  - Attention Is All You Need (Transformer)
  - BERT, GPT, T5

---

**Note:** For more detailed topic coverage, see subdirectories:
- `parts-of-deep-learning/` - Detailed component explanations
- `deep-learning-methods/` - Architecture-specific guides
- `mcp.md` - Advanced production topics
