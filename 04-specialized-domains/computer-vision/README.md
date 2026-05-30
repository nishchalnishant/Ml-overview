---
module: Specialized Domains
topic: Computer Vision
subtopic: ""
status: unread
tags: [computer-vision, cnn, object-detection, segmentation, vit, specialized-domains]
---
# Computer Vision

---

## Table of Contents

1. [Why Computer Vision is Hard](#1-why-computer-vision-is-hard)
2. [Convolutional Neural Networks](#2-convolutional-neural-networks)
3. [CNN Architectures](#3-cnn-architectures)
4. [Object Detection](#4-object-detection)
5. [Image Segmentation](#5-image-segmentation)
6. [Vision Transformers (ViT)](#6-vision-transformers-vit)
7. [Self-Supervised Learning in Vision](#7-self-supervised-learning-in-vision)
8. [Generative Models for Vision](#8-generative-models-for-vision)
9. [Common Interview Questions](#9-common-interview-questions)

---

## 1. Why Computer Vision is Hard

**The problem:** A pixel is just a number. An image is a grid of numbers. The semantic meaning of an image — "a dog sitting on a rug" — is not localized in any single pixel; it emerges from the spatial arrangement of pixels across the entire image. A classifier must be invariant to translations, rotations, lighting changes, viewpoint shifts, occlusions, and intra-class variations, while remaining sensitive to meaningful differences between classes.

**The core insight:** Three properties of images make convolutional architectures the right prior:

1. **Local structure:** Pixels are more correlated with nearby pixels than distant ones. Edges, textures, and shapes are local patterns.
2. **Translation equivariance:** A cat in the top-left corner and a cat in the bottom-right corner should activate the same detectors. The detector should move with the pattern, not be tied to a fixed spatial position.
3. **Compositionality:** Complex visual concepts are composed of simpler ones — objects are composed of parts; parts are composed of textures; textures are composed of edges.

**What MLPs miss:** An MLP with an image flattened to a vector treats every pixel independently. A pixel at position (10, 10) and one at (11, 10) have no special relationship — the model must learn from scratch that adjacent pixels are correlated. CNNs build in this spatial prior via weight sharing and local receptive fields.

---

## 2. Convolutional Neural Networks

### The Convolution Operation

**The problem:** You need to detect a pattern (e.g., a horizontal edge) regardless of where it appears in the image. You could use one neuron per location per filter, but this requires O(H × W × C) parameters per filter and does not share the insight that the same filter should fire at all locations.

**The core insight:** Use a small filter (kernel) of learned weights, and slide it across the image. At each position, compute the dot product between the filter and the local image patch. The result is a feature map showing where that filter's pattern activates strongly.

**Convolution:**
```
output[i, j] = Σ_h Σ_w filter[h, w] × input[i+h, j+w]
```

For a 3×3 filter applied to a 32×32 image: instead of 32×32×9 = 9216 parameters per filter, you have 9 parameters shared across all positions. This is the **parameter sharing** principle.

**Why this works:** The same edge detector that fires on a horizontal edge in the top-left will fire on the same edge anywhere else in the image. Translation equivariance is built in by construction.

### Key Components

**Padding:**
- `VALID`: no padding — output shrinks by (kernel_size - 1) in each dimension
- `SAME`: pad with zeros to preserve spatial dimensions — output is same size as input

**Stride:** Step size of the filter slide. Stride 2 halves the spatial dimensions.

**Depth:** Multiple filters in one layer, each learning a different pattern. Output has C_out channels, one per filter.

```python
import torch
import torch.nn as nn

# A basic conv layer: 32 filters, 3×3 kernel, padding=1 to preserve size
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
# Input: (batch, 3, 224, 224) → Output: (batch, 32, 224, 224)
```

### Pooling

**Max pooling:** Take the maximum value in each local window. Provides spatial invariance — a pattern detected anywhere in the window activates the pooling unit. Most common: 2×2 max pool with stride 2 (halves resolution).

**Average pooling:** Take the mean. Used in global average pooling (GAP) — average the entire feature map to a single number per channel, collapsing spatial dimensions entirely.

**Global average pooling:** Replaces fully-connected layers at the end of a CNN. If the last conv output is (batch, C, H, W), GAP produces (batch, C) — one number per channel, corresponding to "how much of this feature is present anywhere in the image." Dramatically reduces parameters and acts as a regularizer.

### Receptive Field

After k conv layers with kernel size 3 (no pooling), each unit in layer k sees a (2k+1) × (2k+1) region of the input. With pooling, the effective receptive field grows faster. Deep CNNs build hierarchical representations: early layers detect edges and textures; middle layers detect parts (eyes, wheels); late layers detect objects.

### BatchNorm + ReLU

**The standard conv block:**
```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

BatchNorm normalizes activations across the batch, reducing internal covariate shift and enabling higher learning rates. ReLU introduces nonlinearity without vanishing gradients for positive activations.

---

## 3. CNN Architectures

### AlexNet (2012) — The Breakthrough

AlexNet won ImageNet by a ~10% margin over the next-best method — proof that deep CNNs work at scale.

**Key innovations:**
- ReLU instead of tanh/sigmoid — faster training, no vanishing gradient for positive activations
- Dropout (p=0.5) in fully-connected layers — prevents overfitting
- Data augmentation — random crops and horizontal flips at training time
- GPU training — two GPUs trained in parallel

**Architecture:** 5 conv layers + 3 FC layers. 60M parameters.

**What broke the field:** Prior to AlexNet, hand-crafted features (SIFT, HOG) dominated. AlexNet showed learned features surpass hand-crafted ones at scale.

### VGG (2014) — Depth Through Simplicity

**Key insight:** Replace large filters (11×11, 7×7) with stacks of 3×3 filters. Two 3×3 layers have the same receptive field as one 5×5 layer but fewer parameters and an extra nonlinearity.

VGG-16: 16 weight layers, all 3×3 conv, max pooling to reduce resolution. 138M parameters — overparameterized but highly regular and easy to understand.

**Legacy:** VGG-16/19 feature extractors are still used as perceptual loss networks (compare features, not pixels) in style transfer and super-resolution.

### ResNet (2015) — Residual Connections

**The problem:** As networks get deeper, gradients vanish during backpropagation. Adding more layers makes training harder, not easier — performance degrades even on the training set (not just overfitting).

**The core insight:** If a deeper network should learn the same function as a shallower one, the extra layers need to learn the identity function. Identity is hard to learn directly but trivially achievable via a shortcut connection: let the layers learn a *residual* F(x) = H(x) - x rather than the full mapping H(x).

**Residual block:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)  # skip connection
```

**Why this works:** The gradient ∂L/∂x = ∂L/∂(x + F(x)) = 1 + ∂L/∂F(x). The "1" ensures the gradient signal is never fully blocked, even in networks with 100+ layers.

**Variants:** ResNet-18/34 (basic blocks), ResNet-50/101/152 (bottleneck blocks — 1×1 → 3×3 → 1×1 for efficiency).

### EfficientNet (2019) — Compound Scaling

**The insight:** Width, depth, and resolution all affect performance. Scale all three jointly with a fixed compound coefficient rather than scaling one at a time.

**Neural Architecture Search (NAS) finds the base network (EfficientNet-B0); compound scaling rule scales up to B1–B7.**

EfficientNet-B7 matched or exceeded ResNet-152 at ~10× fewer parameters on ImageNet.

### Architecture Comparison

| Model | Year | Top-1 (ImageNet) | Parameters | Key innovation |
|---|---|---|---|---|
| AlexNet | 2012 | 63.3% | 60M | Deep CNNs work; ReLU |
| VGG-16 | 2014 | 74.4% | 138M | All 3×3 convs; depth |
| ResNet-50 | 2015 | 76.1% | 25M | Residual connections |
| EfficientNet-B0 | 2019 | 77.3% | 5.3M | Compound scaling |
| ViT-B/16 | 2020 | 81.8% | 86M | Pure attention on patches |

---

## 4. Object Detection

**The problem:** Classification answers "what is in the image?" Detection answers "what is where?" — return a class label and a bounding box for every object instance. Multiple objects of different classes can overlap.

### Two-Stage Detectors

**R-CNN family:** Generate region proposals first, then classify each proposal.

**Faster R-CNN (2015):** The dominant two-stage architecture.

1. **Backbone:** ResNet extracts feature maps from the full image. Output: C × H' × W' feature tensor.
2. **Region Proposal Network (RPN):** A small network sliding over the feature map. At each location, predicts k anchor boxes (fixed shapes) and for each: objectness score (object vs. background) + bounding box regression offsets.
3. **RoI Pooling:** For each proposal, extract a fixed-size (7×7) feature from the shared backbone features using bilinear interpolation. This allows proposals of different sizes to produce same-size features.
4. **Detection Head:** FC layers classify each RoI and refine the bounding box.

**Key advantage:** Backbone features are computed once, shared across all proposals. Fast inference (5 fps).

**Anchor boxes:** Pre-defined bounding box shapes (aspect ratios × scales). The model predicts *offsets* from anchors rather than absolute coordinates — makes regression easier.

```
Anchor regression:
tx = (x_gt - x_a) / w_a
ty = (y_gt - y_a) / h_a
tw = log(w_gt / w_a)
th = log(h_gt / h_a)
```

### One-Stage Detectors

**YOLO (You Only Look Once):** Divide image into an S×S grid. Each cell predicts B bounding boxes + confidence scores + C class probabilities in a single forward pass. No separate proposal stage.

**YOLOv8 (2023) architecture overview:**
```
Image → Backbone (CSPNet/DarkNet) → Neck (FPN + PAN) → Head (anchor-free)
```

**Feature Pyramid Network (FPN):** Combine features at multiple scales — large feature maps for small objects (high resolution, small receptive field), small feature maps for large objects.

```
P5: 8×8  (large objects)
P4: 16×16
P3: 32×32 (small objects)

Top-down pathway: upsample P5 + lateral connection from backbone at same scale
```

**One-stage vs. two-stage tradeoff:**
| | Two-stage (Faster RCNN) | One-stage (YOLO) |
|---|---|---|
| Speed | ~5 fps | 30–200 fps |
| Accuracy | Higher (especially small objects) | Slightly lower |
| Architecture | Complex | Simpler |
| Use case | Medical imaging, precision tasks | Real-time applications |

### Loss Functions

**Classification loss:** Cross-entropy (or focal loss to handle class imbalance — background is ~99% of anchors).

**Bounding box regression loss:** Smooth L1 (Huber loss) or IoU loss.

**Focal loss (RetinaNet):** Down-weights easy negatives so the model focuses on hard examples:
```
FL(p_t) = -(1 - p_t)^γ log(p_t)
```
When p_t is high (easy, well-classified), (1-p_t)^γ ≈ 0 — contribution to loss is nearly zero.

### Intersection over Union (IoU)

```
IoU = |A ∩ B| / |A ∪ B|
```

Used for: (1) matching ground-truth boxes to anchor boxes during training (IoU > 0.5 → positive anchor), (2) NMS to suppress duplicate detections, (3) evaluation metric (mAP at IoU=0.5).

**Non-Maximum Suppression (NMS):** After detection, many overlapping boxes predict the same object. NMS: sort by confidence, keep the highest-confidence box, discard any box with IoU > threshold against it, repeat.

---

## 5. Image Segmentation

**The problem:** Classification gives one label per image; detection gives bounding boxes. Segmentation gives a label for every pixel — answering "what class does each pixel belong to?"

### Semantic Segmentation

Every pixel gets a class label. No distinction between instances — two cats → all pixels labeled "cat."

**U-Net (2015) — the foundational architecture:**

```
Encoder (contracting path): Conv → Pool × 4 levels
Decoder (expanding path): Upsample + skip connection from encoder × 4 levels
Output: H × W × num_classes
```

**Skip connections:** The decoder at each level receives features from the corresponding encoder level via skip connections. This provides high-resolution spatial detail (where things are) combined with semantic information from the bottleneck (what things are).

```python
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))

# In decoder:
def decode(x, skip):
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = torch.cat([x, skip], dim=1)  # concatenate skip features
    return self.conv_block(x)
```

**DeepLab (dilated convolutions):** Instead of downsampling aggressively, use dilated (atrous) convolutions — the kernel samples with gaps, expanding the receptive field without reducing resolution.

A dilated conv with dilation rate d: `output[i] = Σ filter[k] × input[i + k×d]`

Atrous Spatial Pyramid Pooling (ASPP): apply dilated convolutions at multiple dilation rates in parallel to capture multi-scale context.

### Instance Segmentation

Each object instance gets a separate mask. Two cats → "cat 1 mask" and "cat 2 mask."

**Mask R-CNN (2017):** Extends Faster R-CNN with a mask prediction branch. For each detected RoI, the mask head predicts a per-class binary mask (28×28 resolution).

**Key addition over Faster R-CNN:** RoIAlign — bilinear interpolation instead of quantized RoI pooling. Eliminates misalignment errors that are small for bounding boxes but critical for pixel-level masks.

### Panoptic Segmentation

Combines semantic (background "stuff": sky, road) and instance (foreground "things": cars, people) segmentation into one unified prediction.

**Output:** Every pixel labeled with both a class and an instance ID (for "things") or just a class (for "stuff").

---

## 6. Vision Transformers (ViT)

**The problem:** CNNs have strong inductive biases (local receptive fields, translation equivariance) that help on small datasets but may limit performance when trained at scale. Can transformers — which make fewer assumptions — match or beat CNNs given enough data?

**The core insight (Dosovitskiy et al., 2020):** Divide the image into non-overlapping patches (16×16 pixels). Flatten each patch into a vector, linearly project it. Add a learnable CLS token. Add positional embeddings. Feed the sequence of patch embeddings into a standard transformer encoder. Classify via the CLS token output.

```
Image (224×224×3) → 196 patches of 16×16×3 = 768 dims each
+ 1 CLS token
→ sequence of 197 tokens
→ transformer encoder (12 layers, 12 heads for ViT-B/16)
→ CLS output → MLP → class label
```

**Why this works:** Self-attention is permutation-equivariant and can model long-range dependencies in a single layer. The model learns which patches attend to which — discovering spatial relationships from data, not from inductive bias.

**What breaks:**
- **Data hunger:** ViT requires large datasets (JFT-300M) to outperform CNNs. On ImageNet alone (1.2M images), ResNets are competitive. CNNs' spatial inductive bias is helpful when data is limited.
- **Quadratic attention cost:** Standard attention is O(n²) in sequence length. 196 patches → manageable; high-resolution tasks (medical imaging, detection) require efficient attention variants.

### Positional Embeddings

Without positional information, the transformer cannot distinguish a patch in the top-left from one in the bottom-right. ViT learns 1D positional embeddings for the 196+1 positions. Relative positional biases and 2D positional encodings improve performance.

### Data Augmentation and Training for ViT

ViT requires heavy regularization and augmentation:
- **RandAugment:** random sequences of photometric augmentations
- **Mixup:** blend two images and their labels linearly
- **CutMix:** replace a rectangular crop with one from another image
- **Label smoothing:** prevent overconfident predictions
- **Stochastic depth (DropPath):** randomly drop entire residual branches

### DeiT (Data-Efficient Image Transformers)

**Insight:** Train ViT on ImageNet only (no JFT) using knowledge distillation from a CNN teacher. Adds a **distillation token** alongside the CLS token; trained to match the teacher's hard label predictions.

This makes ViT competitive with CNNs on ImageNet-scale data without massive pre-training datasets.

### DINO and MAE (Self-Supervised ViT Pre-Training)

**DINO (Self-DIstillation with NO labels):** Student and teacher ViTs. Teacher is an exponential moving average of the student. Student is trained to predict teacher's representations for augmented views of the same image. No labels required.

**MAE (Masked Autoencoder):** Mask 75% of patches randomly. Train an encoder (processes only visible patches) + decoder to reconstruct the masked patches from pixel values. The high masking ratio forces the encoder to learn rich semantic representations rather than interpolating local textures.

### Hybrid Architectures

**Swin Transformer (2021):** Combines CNN's hierarchical feature maps with transformer attention. Key innovations:
- **Window-based attention:** Restrict attention to local windows (7×7 patches) — O(n) complexity instead of O(n²)
- **Shifted window:** Alternate between regular and shifted window partitions to allow cross-window information flow
- **Hierarchical features:** Patch merging reduces resolution and increases channel depth, creating a pyramid like ResNets — compatible with FPN for detection

Swin achieved state-of-the-art on detection (COCO) and segmentation (ADE20K) while being ViT-based.

---

## 7. Self-Supervised Learning in Vision

**The problem:** Labeled image datasets are expensive and domain-specific. ImageNet labels were collected via Amazon Mechanical Turk over years. Medical imaging labels require expert radiologists. Can we learn rich visual representations without any labels?

**The core insight:** Formulate a pretext task that requires understanding visual structure to solve, but whose labels come from the data itself. After pretraining, fine-tune on downstream tasks with few labels.

### Contrastive Learning (SimCLR, MoCo)

**The idea:** Different augmented views of the same image should have similar representations. Different images should have dissimilar representations. Learn a representation space where this is true.

**SimCLR (2020):**
1. Take an image x; generate two augmented views x_i and x_j (random crop, color jitter, grayscale, blur)
2. Encode both with the same CNN: h_i = f(x_i), h_j = f(x_j)
3. Project to lower-dimensional space: z_i = g(h_i), z_j = g(h_j)
4. Minimize NT-Xent loss: pull z_i and z_j together, push against all other images in the batch

```
NT-Xent loss for pair (i, j):
l(i, j) = -log[ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
```

**What breaks:** SimCLR needs very large batch sizes (4096+) to have enough negatives. Each image needs many negatives to avoid false dismissals.

**MoCo (Momentum Contrast):** Maintains a queue of negative keys from past batches. Uses a momentum encoder (slowly updated copy of the main encoder) to encode keys. Separates batch size from the number of negatives.

### CLIP (Contrastive Language-Image Pre-Training)

**The idea:** Train image encoder and text encoder jointly to align matching image-caption pairs and push apart non-matching pairs. No explicit class labels — supervision comes from image-caption pairs scraped from the web.

```
For N (image, text) pairs in a batch:
- Compute image embeddings: I_1, ..., I_N
- Compute text embeddings: T_1, ..., T_N
- Maximize cosine similarity of (I_i, T_i) pairs
- Minimize cosine similarity of (I_i, T_j) pairs where i≠j
```

**What this buys:** Zero-shot classification — classify images by comparing their embedding to text embeddings of class descriptions ("a photo of a dog"). No task-specific fine-tuning required. CLIP generalizes across domains (natural images, medical scans, satellite images) without any domain-specific training.

---

## 8. Generative Models for Vision

### GANs

**Generator G:** Maps random noise z to an image. **Discriminator D:** Distinguishes real from generated images. They play a min-max game.

**Training instability:** Mode collapse (G always generates the same image), vanishing gradients when D is too strong. Fixes: Wasserstein GAN (WGAN), spectral normalization, progressive growing (ProGAN, StyleGAN).

**StyleGAN2:** State-of-the-art unconditional image generation. Separates coarse (pose, shape) and fine (texture, color) style via an AdaIN-based style injection at each layer.

### Diffusion Models

**Forward process:** Progressively add Gaussian noise over T steps: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε.

**Reverse process:** Train a U-Net to predict the noise ε given the noisy image x_t and timestep t:
```
L_simple = E_{t, x_0, ε}[ || ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t) ||² ]
```

**Classifier-free guidance (Ho & Salimans, 2022):** Train the model both conditioned and unconditionally. At inference, extrapolate away from the unconditional prediction toward the conditional prediction. Controls the tradeoff between image quality and diversity.

```
ε_guided = ε_uncond + w × (ε_cond - ε_uncond)
```

High guidance weight w → more faithful to the conditioning signal; lower diversity. Low w → more diverse; less faithful.

**Stable Diffusion:** Latent diffusion model (LDM). Run the diffusion process in the latent space of a VAE, not in pixel space. Reduces computation by ~8× while preserving quality. Enables high-resolution generation on consumer hardware.

---

## 9. Common Interview Questions

### Q1: Why do residual connections help train very deep networks?

Without skip connections, gradients in backpropagation must flow through every layer multiplication. In a 100-layer network, even slightly sub-1.0 multiplications compound to near-zero gradients in early layers — those layers receive no learning signal. The residual connection adds a direct path ∂L/∂x = 1 + ∂L/∂F(x). Even if ∂L/∂F(x) vanishes, the gradient "1" ensures the signal always passes through. This allows training networks with 50–150+ layers.

---

### Q2: What is the difference between semantic, instance, and panoptic segmentation?

**Semantic:** Every pixel gets a class label. Two people → both labeled "person." No distinction between instances.

**Instance:** Every object instance gets a separate mask. Two people → "person 1 mask" and "person 2 mask." Background pixels unlabeled.

**Panoptic:** Unifies both. "Things" (countable objects like people, cars) get instance masks. "Stuff" (amorphous regions like sky, grass) gets class labels only. Every pixel is labeled.

---

### Q3: What is the receptive field of a stack of 3×3 convolutions? Why does this matter?

A single 3×3 conv: each output unit sees a 3×3 region. Two stacked 3×3 convs: each unit sees a 5×5 region (2×(3-1)+1 = 5). k stacked 3×3 convs: (2k+1)×(2k+1) receptive field. Three 3×3 convs have the same receptive field (7×7) as one 7×7 conv but use 3×27 = 81 parameters vs. 49 parameters, plus two extra nonlinearities.

This matters because the receptive field determines what context each unit can see. Object detection requires large receptive fields; local texture detection requires small ones. FPNs and dilated convolutions manage receptive field size explicitly.

---

### Q4: What is IoU and how is it used in detection?

IoU = intersection area / union area of two bounding boxes. Used in three ways:
1. **Anchor matching:** During training, assign anchors with IoU > 0.5 to ground truth as positive; IoU < 0.4 as negative.
2. **NMS:** Suppress duplicate detections — after sorting by confidence, remove any box with IoU > 0.5 against the kept box.
3. **Evaluation:** mAP@0.5 — a detection counts as correct only if IoU with ground truth > 0.5. mAP@[0.5:0.95] averages over IoU thresholds from 0.5 to 0.95.

---

### Q5: How does attention in ViT differ from convolution in CNNs?

**CNN convolution:** Fixed local receptive field (3×3 or 5×5). Same filter at every spatial position (weight sharing). Translation equivariant by construction. Receptive field grows linearly with depth.

**ViT self-attention:** Computes relationships between all pairs of patches in a single layer. Receptive field is the entire image from the first layer. No spatial locality inductive bias — the model learns from data which patches to attend to. Global context at every layer, not just at the top.

**Key difference:** CNNs impose spatial locality as an inductive prior (useful with small data). ViT learns spatial relationships from data (useful at scale). In practice, hybrid architectures like Swin Transformer use local attention windows, combining both.

---

### Q6: What is Non-Maximum Suppression and why is it necessary?

Object detectors typically produce many overlapping bounding box predictions for the same object. NMS removes redundant predictions:

1. Sort all predictions by confidence score (descending).
2. Keep the highest-confidence prediction.
3. Remove any remaining prediction with IoU > threshold against the kept box.
4. Repeat from step 2 with the remaining predictions.

**Why needed:** With 8,000+ anchor boxes per image (Faster RCNN), many fire on the same object. Without NMS, each object would have dozens of duplicate detections in the output.

**What breaks:** NMS treats nearby high-confidence predictions as duplicates. When two objects of the same class are heavily overlapping (e.g., two people standing close), NMS may suppress the less-confident one even though it's a different object. Soft-NMS (decay rather than remove) partially addresses this.

---

### Q7: Why does CLIP enable zero-shot classification?

CLIP trains image and text encoders jointly to maximize cosine similarity between matching image-text pairs. After training, both encoders map to the same embedding space — images and their text descriptions have nearby embeddings.

Zero-shot classification: for each class, construct a text prompt ("a photo of a {class}"). Embed the image and all class prompts. Predict the class whose text embedding is most similar to the image embedding.

No class-specific training needed — CLIP has already learned what "cat," "dog," "X-ray," and "satellite image" look like from 400M image-text pairs. The text encoder effectively programs the classifier at inference time.

---

## Quick Reference: Architecture Comparison

| Task | Model | Key idea |
|---|---|---|
| Image classification | ResNet | Residual connections; deep but trainable |
| Image classification | EfficientNet | Compound scaling |
| Image classification | ViT | Patch embeddings + transformer |
| Object detection (accurate) | Faster R-CNN + FPN | Two-stage; shared backbone |
| Object detection (fast) | YOLOv8 | Single-stage; anchor-free |
| Semantic segmentation | DeepLabV3+ | Dilated convs + ASPP |
| Instance segmentation | Mask R-CNN | Faster RCNN + mask head |
| Panoptic segmentation | Mask2Former | Unified transformer head |
| Self-supervised | DINO / MAE | Knowledge distillation / masked patches |
| Vision-language | CLIP | Contrastive image-text alignment |
| Image generation | Stable Diffusion | Latent diffusion in VAE space |

---

## Key Papers

| Paper | Year | Contribution |
|---|---|---|
| Krizhevsky et al. | 2012 | AlexNet — deep CNNs on ImageNet |
| Simonyan & Zisserman | 2014 | VGGNet — depth through 3×3 convs |
| He et al. | 2015 | ResNet — residual connections |
| Ren et al. | 2015 | Faster R-CNN — RPN for fast detection |
| He et al. | 2017 | Mask R-CNN — instance segmentation |
| Tan & Le | 2019 | EfficientNet — compound scaling |
| Dosovitskiy et al. | 2020 | ViT — transformers for image classification |
| Liu et al. | 2021 | Swin Transformer — hierarchical shifted-window attention |
| He et al. | 2022 | MAE — masked autoencoder self-supervised ViT |
| Rombach et al. | 2022 | Latent Diffusion Models / Stable Diffusion |
| Radford et al. | 2021 | CLIP — contrastive language-image pretraining |
