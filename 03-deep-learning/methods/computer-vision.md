# Computer Vision — Comprehensive Reference

---

## Table of Contents

1. CNN Building Blocks
2. Architecture Evolution
3. Skip Connections and Residual Learning
4. Object Detection (R-CNN → YOLO → DETR)
5. Semantic and Instance Segmentation
6. Vision Transformers (ViT, DeiT, Swin)
7. CLIP and Vision-Language Models (BLIP-2, LLaVA)
8. Image Generation: GANs, VAEs, Diffusion, Score Matching
9. Self-Supervised Learning (SimCLR, DINO, MAE)
10. Data Augmentation Strategies
11. Evaluation Metrics
12. Production Considerations
13. Common Interview Questions

---

## 1. CNN Building Blocks

### 1.1 Convolution Operation

A convolutional layer slides a filter (kernel) of size `K x K` across an input feature map, computing a dot product at each position:

```
output[i, j] = sum_{m=0}^{K-1} sum_{n=0}^{K-1} input[i+m, j+n] * filter[m, n] + bias
```

**Key properties:**
- **Local connectivity**: each output neuron is connected to a `K x K` patch of the input, not the full image
- **Weight sharing**: the same filter is applied everywhere — this is why a cat detector works regardless of where the cat appears
- **Translation equivariance**: if the input shifts, the output shifts by the same amount (not invariant — pooling adds that)

**Output spatial dimensions:**
```
H_out = floor((H_in + 2*pad - K) / stride) + 1
W_out = floor((W_in + 2*pad - K) / stride) + 1
```

```python
import torch
import torch.nn as nn

# Single conv layer: 3 input channels, 64 output channels, 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                 stride=1, padding=1)  # padding=1 preserves spatial size with 3x3 kernel

# Input: (batch=8, channels=3, H=224, W=224)
x = torch.randn(8, 3, 224, 224)
out = conv(x)   # (8, 64, 224, 224)
```

**Parameter count** for a conv layer: `K * K * C_in * C_out + C_out (bias)`

For a 3x3 conv, 64->128 channels: `3*3*64*128 + 128 = 73,856` parameters. Compare to a fully connected layer on the same input: `(H*W*64) * (H*W*128)` — millions of parameters.

### 1.2 Activation Functions

**ReLU**: `f(x) = max(0, x)`. Computationally cheap, helps with vanishing gradient. Dead neurons (always output 0) are a concern.

**Leaky ReLU**: `f(x) = max(alpha*x, x)` with small `alpha` (e.g., 0.01). Fixes dead neuron problem.

**GELU**: `f(x) = x * Phi(x)` where `Phi` is the standard normal CDF. Smoother than ReLU; used in modern architectures (ViT, EfficientNet).

### 1.3 Pooling

**Max pooling**: take maximum in each pool region. Most common. Provides local translation invariance.

**Average pooling**: take mean. Smoother, used in later layers.

**Global Average Pooling (GAP)**: reduce each feature map to a single value. Used before the classifier head. Dramatically reduces parameters and acts as regularization compared to flattening.

```python
# Replace flatten + large FC with GAP + small FC
gap = nn.AdaptiveAvgPool2d(1)  # output: (batch, channels, 1, 1)
x = gap(feature_maps).flatten(1)  # (batch, channels)
logits = nn.Linear(channels, num_classes)(x)
```

### 1.4 Receptive Field

The **receptive field** of a neuron is the region of the input image that influences its output. Deep networks have large receptive fields despite local connectivity:

- Layer 1 with 3x3 kernel: RF = 3x3
- Layer 2 with 3x3 kernel: RF = 5x5 (adds 1 on each side)
- Layer L with 3x3 kernels: RF = 2L+1 (linear growth)

With **dilation**: RF grows exponentially. With **strides**: effectively compresses the feature map.

**Effective receptive field** (empirical finding): neurons in practice behave as if their effective RF is much smaller than the theoretical RF, roughly Gaussian-distributed. This is why very large kernels are often not more effective than stacked small ones.

Two stacked 3x3 conv layers have the same RF as one 5x5 layer but fewer parameters (`2*(3*3*C^2)` vs `5*5*C^2`) and an extra nonlinearity.

### 1.5 Feature Maps

Each channel in a conv layer learns to detect a different visual feature. Early layers detect edges and colors; middle layers detect textures and shapes; later layers detect semantic concepts (eyes, wheels, fur).

Visualization via **activation maximization** or **Grad-CAM** confirms this hierarchy:

```python
# Grad-CAM: visualize which input regions drive a particular class prediction
import torch
import torch.nn.functional as F

def grad_cam(model, x, target_class):
    model.eval()
    x.requires_grad_(True)

    # Forward pass, capture last conv layer activations
    activations = {}
    def hook(module, input, output):
        activations['feat'] = output

    handle = model.layer4.register_forward_hook(hook)
    out = model(x)
    handle.remove()

    # Backward pass for target class
    model.zero_grad()
    out[0, target_class].backward()

    # Weight activations by gradient
    grads = activations['feat'].grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations['feat']).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
    return cam.squeeze()
```

### 1.6 Depthwise Separable Convolution

Standard 3x3 conv on C channels: `3*3*C*C` multiplications per spatial location.

**Depthwise separable** (MobileNet): split into:
1. Depthwise conv: one filter per channel, `3*3*C` multiplications
2. Pointwise (1x1) conv: `C*C'` multiplications

Total: `3*3*C + C*C'` vs `3*3*C*C'` — approximately 8-9x fewer operations for large C.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch)   # depthwise
        self.pw = nn.Conv2d(in_ch, out_ch, 1)          # pointwise
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x
```

### 1.7 Batch Normalization

Normalizes activations within a mini-batch, then applies learned scale and shift:

```
y = gamma * (x - mu_B) / sqrt(sigma_B^2 + eps) + beta
```

Benefits:
- Reduces internal covariate shift (distributions of activations stay stable across layers)
- Acts as regularizer (slight noise from batch statistics)
- Allows higher learning rates
- Reduces sensitivity to weight initialization

```python
# BatchNorm goes after conv, before activation
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU()
)
```

At inference, use running mean and variance (accumulated during training), not batch statistics.

---

## 2. Architecture Evolution

### 2.1 LeNet-5 (1998)

The first practical deep convolutional network for digit recognition.

Structure: `Conv -> Pool -> Conv -> Pool -> FC -> FC -> output`

Key: demonstrated that learning hierarchical features via convolution beats hand-crafted features. Limited to grayscale 32x32 inputs.

### 2.2 AlexNet (2012)

Winner of ImageNet 2012 — reduced error from ~26% to ~15%. Marks the beginning of the deep learning era in vision.

Innovations:
- **ReLU activations** (vs tanh): faster training, less vanishing gradient
- **Dropout** (0.5 in FC layers): regularization
- **Data augmentation**: random crops, horizontal flips, color jittering
- **GPU training**: split across two GTX 580s
- **Local Response Normalization (LRN)**: lateral inhibition (later superseded by BatchNorm)

Architecture: 5 conv layers, 3 FC layers. 60M parameters.

### 2.3 VGG (2014)

Key insight: **depth matters more than kernel size**. Replace large filters (7x7, 11x11) with stacked 3x3 filters.

Two 3x3 convs: same receptive field as one 5x5 but fewer parameters and one more nonlinearity.
Three 3x3 convs: same RF as one 7x7.

VGG-16 (16 weight layers): uniform architecture, easy to understand, still used as backbone.

Weakness: 138M parameters, most in FC layers — very memory-heavy.

### 2.4 Inception (GoogLeNet, 2014)

**Inception module**: apply multiple filter sizes in parallel, concatenate outputs.

```
Input -> [1x1 conv | 3x3 conv | 5x5 conv | 3x3 max pool] -> concatenate
```

Uses 1x1 convolutions before 3x3/5x5 to reduce channel dimension (bottleneck) — dramatically reduces parameters. 4M parameters vs VGG's 138M.

**Inception V3**: factorize n×n into 1×n followed by n×1 (asymmetric convolution). Reduces parameters further.

### 2.5 ResNet (2015)

See Section 3 for deep coverage. Key metric: first model to surpass human-level performance on ImageNet (top-5 error 3.57% vs human 5.1%).

Enabled training of 50-, 101-, 152-layer networks that were previously impossible to train.

### 2.6 DenseNet (2017)

Every layer receives feature maps from **all preceding layers** and passes its own features to all subsequent layers.

```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

Where `[...]` is channel-wise concatenation.

Benefits:
- Strong gradient flow to all layers
- Feature reuse: each layer can use low-level features from early layers
- Fewer parameters than ResNet (no need to re-learn low-level features)
- Built-in regularization from feature reuse

Transition layers (1x1 conv + 2x2 avg pool) reduce dimension between dense blocks.

### 2.7 EfficientNet (2019)

**Neural Architecture Search (NAS)** finds a baseline architecture, then **compound scaling** scales it uniformly in three dimensions:

```
depth:   d = alpha^phi
width:   w = beta^phi
resolution: r = gamma^phi
constraint: alpha * beta^2 * gamma^2 ~= 2
```

`phi` is a compound scaling coefficient. Doubling FLOPs is budgeted optimally across all three dimensions.

EfficientNet-B7 achieves 84.3% top-1 accuracy on ImageNet with 8x fewer parameters than the best prior models.

### 2.8 ConvNeXt (2022)

"A ConvNet for the 2020s" — modernizes ResNet-50 by adopting design choices from ViT:
- Larger kernels (7x7 depthwise conv)
- Inverted bottleneck (wide middle, narrow ends)
- GELU activation
- LayerNorm instead of BatchNorm
- Fewer activation/normalization layers

Result: matches or exceeds ViT-based models at the same compute budget, with simpler training (no need for large-scale pretraining).

```python
# Torchvision provides ConvNeXt
import torchvision.models as models

model = models.convnext_base(pretrained=True)
# Replace classifier head for fine-tuning
model.classifier[2] = nn.Linear(1024, num_classes)
```

### 2.9 Architecture Selection Guide

| Scenario | Recommended Backbone |
|---|---|
| Edge / mobile deployment | MobileNetV3, EfficientNet-B0 |
| Moderate compute, high accuracy | ResNet-50, EfficientNet-B4 |
| Maximum accuracy, large compute | EfficientNet-B7, ConvNeXt-XL, ViT-L |
| Fine-tuning from ImageNet | ResNet-50 (stable, well-understood) |
| Detection backbone | ResNet + FPN, EfficientDet |

---

## 3. Skip Connections and Residual Learning

### 3.1 The Degradation Problem

As networks get deeper, training error increases — not from overfitting, but from optimization difficulty. A 56-layer plain network performs worse than a 20-layer one on CIFAR-10.

Hypothesis: it should always be possible to match a shallower network by setting extra layers to identity. But gradient-based optimization fails to find this solution in deep plain networks.

### 3.2 Residual Block

ResNet introduces the **residual (skip) connection**:

```
y = F(x, {W_i}) + x
```

Instead of learning `H(x)` directly, the layers learn the residual `F(x) = H(x) - x`. The identity `x` is added back.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)   # skip connection
        return out
```

**Highway analogy**: skip connections create a "gradient superhighway" — gradients flow directly from loss to early layers without passing through all intermediate transformations. Multiplying partial derivatives through 100 layers no longer degrades the gradient.

Formally, the gradient through a skip connection is:
```
dL/dx = dL/dy * (dF/dx + I)
```

The identity term `I` means gradients never vanish even if `dF/dx` is small.

### 3.3 Bottleneck Block (ResNet-50+)

For deeper networks, use a bottleneck to reduce computation:

```
1x1 conv (reduce channels: 256 -> 64)
3x3 conv (bottleneck: 64 -> 64)
1x1 conv (expand channels: 64 -> 256)
```

Parameters: `1*1*256*64 + 3*3*64*64 + 1*1*64*256 = 69,632` vs `3*3*256*256*2 = 1,179,648` for two 3x3 convs.

```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)
```

### 3.4 Why Residual Learning Works

Three complementary explanations:

1. **Gradient flow**: direct paths from loss to early layers prevent vanishing gradients
2. **Ensemble interpretation**: ResNets can be seen as an ensemble of many shallow paths (paths of different lengths from input to output all exist due to skip connections)
3. **Loss landscape smoothing**: residual connections make the loss surface smoother and easier to optimize (Li et al., 2018)

---

## 4. Object Detection

### 4.1 Problem Definition

Given an image, produce a set of bounding boxes `(x1, y1, x2, y2)`, class labels, and confidence scores for each detected object.

Two families: **two-stage** (propose then classify) and **one-stage** (predict directly).

### 4.2 Two-Stage Detectors: R-CNN Family

**R-CNN (2014)**:
1. Extract ~2000 region proposals using Selective Search
2. Warp each region to fixed size, run CNN to extract features
3. Classify with SVM, refine bounding box with linear regression

Problem: ~50s per image — too slow. Each region processed separately.

**Fast R-CNN (2015)**:
1. Run CNN on full image once to get feature map
2. Project region proposals onto feature map
3. **ROI Pooling**: extract fixed-size feature for each region
4. Shared FC layers classify and regress boxes simultaneously

Speed: ~0.3s per image. Bottleneck: Selective Search (CPU-bound).

**Faster R-CNN (2016)**:
Replace Selective Search with **Region Proposal Network (RPN)** — a small network sliding over feature map, proposing boxes at each location.

```
Feature Map (from backbone)
      |
      +---> RPN ---> proposals (objectness score + box)
      |
      +---> ROI Pooling (using proposals) ---> classification + regression
```

RPN shares convolutions with detection network — trained end-to-end.

Speed: ~0.2s per image on GPU. Sets the two-stage paradigm.

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Faster R-CNN with ResNet-50 + FPN backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = [torch.randn(3, 800, 600)]  # list of images
with torch.no_grad():
    predictions = model(image)
# predictions[0]: dict with 'boxes', 'labels', 'scores'
```

**Feature Pyramid Network (FPN)**: builds a multi-scale feature pyramid with lateral connections. Detects small objects at high-resolution early features and large objects at low-resolution late features — both with rich semantic information.

```
C2 (stride 4)   -------> P2 (high res, low semantic)
C3 (stride 8)   -------> P3
C4 (stride 16)  -------> P4
C5 (stride 32)  -------> P5 (low res, high semantic)
```

### 4.3 Anchor Boxes

**Anchors**: predefined boxes at each spatial location on the feature map, spanning multiple scales and aspect ratios (e.g., scales [0.5, 1, 2], ratios [1:1, 1:2, 2:1]).

The network predicts **offsets** from anchors to ground truth boxes, rather than absolute coordinates:
```
tx = (gx - ax) / aw
ty = (gy - ay) / ah
tw = log(gw / aw)
th = log(gh / ah)
```

Anchors make training stable by giving the model a reference point.

**Anchor-free detectors** (CenterNet, FCOS): instead of anchors, predict object center and box extent directly, avoiding anchor hyperparameter tuning.

### 4.4 IoU (Intersection over Union)

```
IoU = Area(Intersection) / Area(Union)
```

Used to:
- Match predicted boxes to ground truth during training (IoU > 0.5 = positive)
- Filter predictions in NMS (Non-Maximum Suppression)
- Compute mAP metric

```python
def compute_iou(box1, box2):
    # box format: (x1, y1, x2, y2)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)
```

### 4.5 Non-Maximum Suppression (NMS)

Removes redundant overlapping boxes for the same object:

1. Sort boxes by confidence score (descending)
2. Take the highest-scoring box, add to output
3. Remove all boxes with IoU > threshold (e.g., 0.5) with selected box
4. Repeat from step 2 until no boxes remain

**Soft-NMS**: instead of hard removal, decay scores of overlapping boxes by a factor (Gaussian or linear). Helps when objects are legitimately occluded.

### 4.6 One-Stage Detectors

**YOLO v1 (2016)**: divide image into SxS grid. Each cell predicts B boxes and C class probabilities directly. Single forward pass: 45 FPS.

**YOLO v3**: multi-scale prediction at 3 different grid scales using FPN-style feature pyramid. Darknet-53 backbone. 320x320: 22ms inference.

**YOLO v5/v8**: further refinements — anchor clustering, mosaic augmentation, task-aligned head. YOLO v8 is anchor-free, using a decoupled head (separate classification and regression branches).

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano variant, fastest
results = model('image.jpg', conf=0.25, iou=0.45)

for r in results:
    print(r.boxes.xyxy)   # bounding boxes
    print(r.boxes.cls)    # class indices
    print(r.boxes.conf)   # confidence scores
```

**SSD (Single Shot MultiBox Detector, 2016)**: anchor-based predictions at multiple feature map scales simultaneously. Faster than Faster R-CNN, less accurate.

**RetinaNet + Focal Loss (2017)**:

Class imbalance problem: most anchors are background (easy negatives). They dominate the loss and prevent learning from hard examples.

**Focal Loss** down-weights easy examples:
```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `p_t`: model probability for correct class
- `(1 - p_t)^gamma`: focusing factor — high when `p_t` is high (easy), low when `p_t` is low (hard)
- `gamma=2` is standard; effectively reduces easy examples' contribution by 100x

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

### 4.7 DETR: Transformer-Based Detection, No NMS

DETR (Detection Transformer, Facebook AI 2020) reformulated object detection as a direct set prediction problem, eliminating anchors and NMS entirely.

Architecture:
1. CNN backbone (ResNet-50) extracts a feature map.
2. Flatten spatial dimensions → sequence of image features, add positional encodings.
3. Transformer encoder processes the full sequence with global self-attention.
4. Transformer decoder takes N fixed **object queries** (learnable embeddings, e.g., N=100) and attends to encoder output via cross-attention.
5. Each query independently produces a predicted class + bounding box.

Training uses the **Hungarian algorithm** to find the optimal bipartite matching between the N predictions and the ground truth objects (padded with "no-object"). The matched pairs are then supervised with class cross-entropy + L1 + GIoU box loss.

The object queries act like detectives: each one specializes in finding one candidate object anywhere in the scene, attending to the globally-contextualized image features to decide if and where an object exists.

Benefits:
- No hand-designed anchors or NMS post-processing.
- End-to-end differentiable pipeline.
- Global context: every query can attend to any part of the image from the very first layer.

Limitations:
- Slow training convergence: ~500 epochs to match Faster R-CNN (~12 epochs).
- Poor at small objects in the original version.
- Quadratic attention cost over high-resolution feature maps.

**Deformable DETR (2021)**: fixes convergence and small-object detection by replacing full attention with deformable attention — each query attends to a small set of learned sparse spatial locations rather than all positions. Converges in ~50 epochs with better small-object mAP.

**DINO-Det (2022)**: combines deformable attention, improved query initialization, and contrastive denoising training. State-of-the-art end-to-end detector on COCO.

```python
# Conceptual DETR forward pass
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_queries=100, num_classes=91):
        super().__init__()
        self.backbone = backbone                          # CNN feature extractor
        self.input_proj = nn.Conv2d(2048, 256, 1)       # reduce channels
        self.transformer = transformer                   # encoder + decoder
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 for no-object
        self.bbox_embed  = MLP(256, 256, 4, 3)          # predict (cx, cy, w, h)
        self.query_embed = nn.Embedding(num_queries, 256)

    def forward(self, x):
        feat = self.input_proj(self.backbone(x))         # (B, 256, H/32, W/32)
        src = feat.flatten(2).permute(2, 0, 1)          # (HW/1024, B, 256)
        hs = self.transformer(src, self.query_embed.weight)  # (B, N_queries, 256)
        outputs_class = self.class_embed(hs)             # (B, N, num_classes+1)
        outputs_coord = self.bbox_embed(hs).sigmoid()   # (B, N, 4) in [0,1]
        return outputs_class, outputs_coord
```

### 4.8 Two-Stage vs One-Stage vs Transformer Comparison

| | Two-Stage | One-Stage | DETR-family |
|---|---|---|---|
| Example | Faster R-CNN | YOLO v8, RetinaNet | Deformable DETR, DINO-Det |
| Speed | Slower (~0.1-1s) | Faster (5-50ms) | Moderate (50-200ms) |
| Accuracy | High | Slightly lower, closing gap | SOTA (COCO) |
| Anchors | Yes | Yes (or anchor-free) | No |
| NMS | Yes | Yes | No |
| Architecture | Separate proposal + detection | Unified single pass | End-to-end transformer |
| Use case | High-accuracy, offline | Real-time, edge | Research, SOTA pipelines |

---

## 5. Semantic and Instance Segmentation

### 5.1 Fully Convolutional Network (FCN, 2015)

Replace FC layers in classification CNNs with conv layers to produce a spatial output map.

**Upsampling**: feature maps are coarser than input due to pooling. FCN uses transposed convolutions (also called deconvolution) to upscale back to input resolution.

Skip connections from earlier layers restore fine spatial detail:
```
Prediction = upsample(pool5) + upsample(pool4) + pool3
```

Output: per-pixel class prediction (H x W x num_classes).

### 5.2 U-Net (2015)

**Encoder-decoder** with **skip connections** at every resolution level:

```
Input (572x572)
  -> Encoder (conv + pool, repeat)
     -> Bottleneck (deepest features)
  -> Decoder (upsample + concat skip connection + conv, repeat)
     -> Output (H x W x num_classes)
```

Each decoder layer receives: upsampled features from below + high-resolution features from corresponding encoder layer (concatenated, not added).

Originally designed for biomedical image segmentation with limited data. The skip connections preserve fine detail (cell boundaries, vessel edges) that the bottleneck would otherwise lose.

```python
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.bottleneck = UNetBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)  # 512 + 512 skip
        # ... similar for dec3, dec2, dec1
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        # ... similar upsampling steps
        return self.final(d1)
```

### 5.3 DeepLab Series

**DeepLabV3+** uses:
- **Atrous (dilated) convolution**: increase receptive field without losing resolution. Dilation rate `r` inserts `r-1` zeros between filter elements.
- **ASPP (Atrous Spatial Pyramid Pooling)**: apply atrous convolutions at multiple rates in parallel (like Inception but with dilations), capturing context at multiple scales.
- **Encoder-decoder**: encode with ResNet or Xception, decode with ASPP + simple upsampling

```python
# Using torchvision's DeepLabV3+
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True)
model.eval()

with torch.no_grad():
    output = model(image_tensor)
# output['out']: (1, num_classes, H, W)
seg_mask = output['out'].argmax(dim=1)
```

### 5.4 Mask R-CNN (2017) — Instance Segmentation

Extends Faster R-CNN with a third parallel head that predicts a binary mask for each detected object.

```
Faster R-CNN backbone + FPN
      |
      +---> RPN -> proposals
      |
      +---> ROIAlign -> [Classification head]
                     -> [Box regression head]
                     -> [Mask head: FCN outputting K binary masks]
```

**ROIAlign** (vs ROI Pooling): uses bilinear interpolation instead of quantization when projecting proposals onto feature maps. Eliminates spatial misalignment, critical for mask quality.

Each instance gets its own mask — this is what distinguishes instance from semantic segmentation. Two cats in an image get two separate masks.

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

with torch.no_grad():
    preds = model(image)

for i, mask in enumerate(preds[0]['masks']):
    # mask: (1, H, W) with float values
    binary_mask = (mask.squeeze() > 0.5).numpy()
```

---

## 6. Vision Transformers (ViT)

### 6.1 Core Architecture

**ViT (2020)**: treat an image as a sequence of patches.

1. Split image (H x W x C) into `N = HW/P^2` patches, each of size `P x P x C`
2. Linearly project each patch to dimension `D` (patch embedding)
3. Add learnable positional encoding
4. Prepend a learnable `[CLS]` token
5. Pass through L layers of standard Transformer encoder
6. Use `[CLS]` token's output for classification

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, d_model=768, depth=12, heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model*4,
                                       dropout=0.1, batch_first=True),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        # Reshape to patches
        x = x.reshape(B, C, H//P, P, W//P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*P*P)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])  # CLS token
```

### 6.2 Positional Encoding

Unlike CNNs, Transformers have no built-in spatial bias. Positional encoding injects location information.

**Learned absolute**: a learnable embedding per patch position. Simple, works well for fixed input size.

**2D sinusoidal**: extend 1D sinusoidal PE to 2D by separate row and column encodings. Generalizes to different image sizes.

**Relative positional bias** (Swin, T5): instead of adding position to token embeddings, add a learnable bias to the attention logits based on relative position between tokens. More flexible, supports variable-size images.

### 6.3 ViT Scaling Properties

ViT underperforms CNNs when trained on ImageNet alone (1.2M images). It lacks the inductive biases of CNNs (local connectivity, translation equivariance). With sufficient pretraining data, it surpasses CNNs:

- ViT-B/16 trained on ImageNet-21K (14M images): 84.2% top-1
- ViT-L/16 trained on JFT-300M (300M images): 87.76% top-1

**DeiT (Data-efficient Image Transformers)**: trains ViT-scale models on ImageNet alone using knowledge distillation from a CNN teacher. Eliminates the need for massive external datasets.

### 6.4 Swin Transformer (2021)

Addresses ViT's two limitations: quadratic attention cost and lack of multi-scale features.

**Window attention**: compute self-attention within local non-overlapping windows of size `M x M`. Complexity: `O(M^2 * N)` linear in image size.

**Shifted window**: alternate between regular and shifted windows to allow cross-window information exchange.

**Hierarchical feature maps**: patch merging layers halve resolution and double channels (like CNN stride), producing multi-scale representations usable by FPN for detection.

Swin Transformer achieves SOTA on detection (COCO) and segmentation (ADE20K) as a backbone.

```python
import torchvision.models as models

# Swin-T as backbone
backbone = models.swin_t(pretrained=True)
```

---

## 7. CLIP and Contrastive Vision-Language Models

### 7.1 CLIP (Contrastive Language-Image Pretraining, OpenAI 2021)

**Training**: 400M (image, text) pairs from the internet. A vision encoder and a text encoder are trained jointly using contrastive loss.

For a batch of N image-text pairs, the objective maximizes similarity of matching pairs and minimizes similarity of non-matching pairs:

```
L = -1/(2N) * [sum_i log(sim(v_i, t_i)/tau) + sum_i log(sim(t_i, v_i)/tau)]
```

Where `sim(v, t) = v^T t / (||v|| ||t||)` (cosine similarity) and `tau` is a learned temperature.

```python
import torch
import clip

model, preprocess = clip.load("ViT-B/32", device="cuda")

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to("cuda")
texts = clip.tokenize(["a cat", "a dog", "a car"]).to("cuda")

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(texts)

    # Normalize and compute similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)  # [0.92, 0.06, 0.02] -> "a cat" with 92% prob
```

### 7.2 Zero-Shot Classification

CLIP enables zero-shot classification: classify images into categories never seen during training.

```python
# No fine-tuning needed for new categories
categories = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# Prompt engineering improves performance
text_prompts = [f"a photo of a {c}" for c in categories]
text_tokens = clip.tokenize(text_prompts).to("cuda")

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
```

CLIP achieves 76.2% zero-shot top-1 accuracy on ImageNet without any ImageNet training examples.

### 7.3 CLIP Applications

- **Zero-shot classification**: any category expressible in language
- **Image retrieval**: embed images and text queries in shared space
- **Grounding** (e.g., OWL-ViT): extend to object detection with text queries
- **CLIP fine-tuning** (CoOp, CLIP-Adapter): adapt to specific domains with few examples
- **Backbone for generation**: CLIP text encoder is used in Stable Diffusion's conditioning

### 7.4 BLIP-2 and LLaVA: Multimodal LLMs

CLIP maps images to text-aligned embeddings but cannot generate language. The next step: plug a vision encoder into a language model.

**BLIP-2 (2023, Salesforce)**

BLIP-2 bridges a frozen image encoder and a frozen LLM using a lightweight trainable bridge called a **Q-Former** (Querying Transformer).

The Q-Former takes a fixed set of N learned query embeddings (e.g., 32) and uses cross-attention over the image encoder's output to extract the most task-relevant visual information. These N visual tokens are then prepended to the LLM's input as a soft visual prompt.

Key design: both the vision encoder and the LLM are frozen. Only the Q-Former (~188M params) is trained. This makes pretraining efficient: you leverage large pretrained components and train only the bridge between them.

Training is staged: first train Q-Former with a frozen image encoder on image-text contrastive + matching + generation objectives. Then align the Q-Former output with the frozen LLM.

**LLaVA (2023, Haotian Liu et al.)**

LLaVA (Large Language and Vision Assistant) is architecturally simpler: a CLIP ViT-L/14 image encoder, a single linear projection layer, and an LLaMA-based instruction-following LLM.

The projection layer maps CLIP's 256 visual tokens directly into the LLM's embedding space. The model is then instruction-tuned on visual question-answering data generated by GPT-4.

Despite the simplicity of the visual projection (no Q-Former), LLaVA performs strongly on visual reasoning benchmarks. LLaVA-1.5 upgraded to MLP projection and more training data. LLaVA-NeXT increased input resolution with dynamic high-res tiling.

```python
# Conceptual LLaVA forward pass
class LLaVA(nn.Module):
    def __init__(self, vision_encoder, projector, llm):
        super().__init__()
        self.vision_encoder = vision_encoder  # CLIP ViT-L/14, frozen
        self.projector = projector             # nn.Linear(1024, 4096) — single layer
        self.llm = llm                         # LLaMA-2-7B

    def forward(self, images, input_ids):
        with torch.no_grad():
            image_features = self.vision_encoder(images)   # (B, 256, 1024)
        visual_tokens = self.projector(image_features)     # (B, 256, 4096)
        # Concatenate visual tokens with text token embeddings, then run LLM
        text_embeddings = self.llm.get_input_embeddings()(input_ids)  # (B, L, 4096)
        inputs = torch.cat([visual_tokens, text_embeddings], dim=1)   # (B, 256+L, 4096)
        return self.llm(inputs_embeds=inputs)
```

These architectures underpin modern vision-language assistants. The pattern is now standard: frozen vision encoder + lightweight bridge + frozen or lightly fine-tuned LLM.

---

## 8. Image Generation: GANs, VAEs, Diffusion

### 8.1 Generative Adversarial Networks (GANs)

Two networks trained adversarially:
- **Generator G**: maps noise `z ~ p(z)` to fake images
- **Discriminator D**: classifies real vs fake

Objective:
```
min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

Key challenges: **mode collapse** (G only generates a few modes), **training instability**, **vanishing gradients** (D too confident, G gets no signal).

Modern variants: StyleGAN2/3 (high-quality face synthesis), BigGAN (class-conditional), CycleGAN (unpaired image-to-image translation).

### 8.2 Variational Autoencoders (VAEs)

Encode input to a **distribution** in latent space, sample, and decode.

```
Encoder: q(z|x) = N(mu(x), sigma^2(x))
Decoder: p(x|z)

ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

The reparameterization trick: `z = mu(x) + sigma(x) * epsilon, epsilon ~ N(0,I)` — allows backpropagation through the sampling step.

VAEs produce blurrier images than GANs but have a structured, smooth latent space useful for interpolation and manipulation.

### 8.3 Diffusion Models

**Forward process**: gradually add Gaussian noise to image over T steps until it is pure noise.

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)
```

**Reverse process**: train a neural network (UNet) to denoise step-by-step.

```
p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma^2_t * I)
```

Training objective (simplified): predict the noise `epsilon` added at each step.

```
L = E_{x_0, t, epsilon}[||epsilon - epsilon_theta(sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon, t)||^2]
```

**DDPM (Ho et al., 2020)**: established the modern diffusion framework. Inference: 1000 steps.

**DDIM**: deterministic sampling, enables 50x speedup (20-50 steps) with similar quality.

**Stable Diffusion**: latent diffusion — compress image to latent space first, run diffusion in latent space. Much faster than pixel-space diffusion.

```python
# Simplified DDPM training step
import torch
import torch.nn.functional as F

def ddpm_train_step(model, x_0, T=1000):
    """model: U-Net that predicts noise; x_0: clean image batch"""
    # Sample random timestep and noise
    t = torch.randint(0, T, (x_0.shape[0],), device=x_0.device)
    epsilon = torch.randn_like(x_0)

    # Precomputed noise schedule: alpha_bar_t = prod_{s=1}^{t}(1 - beta_s)
    alpha_bar = get_alpha_bar(t)  # (B,)
    alpha_bar = alpha_bar.view(-1, 1, 1, 1)

    # Forward process: add noise to get x_t
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon

    # Reverse process: predict the noise
    epsilon_pred = model(x_t, t)
    return F.mse_loss(epsilon_pred, epsilon)
```

**Score-Matching Intuition**

The "score" of a probability distribution p(x) is its gradient with respect to x: `∇_x log p(x)`. This score field points in the direction of increasing probability — toward more likely images.

Denoising diffusion models are score estimators in disguise. The noise prediction network `ε_θ(x_t, t)` is mathematically equivalent to estimating the score of the noisy data distribution at noise level t:

```
score ≈ -ε_θ(x_t, t) / sqrt(1 - alpha_bar_t)
```

Generation is gradient ascent in probability space with injected noise (Langevin dynamics):
```
x_{t-1} = x_t + (step) * score(x_t) + noise
```

This "score-based" view, unified by Song et al. (2021), shows that DDPM, DDIM, and continuous-time (stochastic differential equation) diffusion models are all instances of the same framework — learning to follow probability gradients through the data manifold.

Diffusion models currently produce the highest-quality images and have supplanted GANs for most generation tasks.

---

## 9. Self-Supervised Learning for Vision

### 9.1 Motivation

Labeled ImageNet has 1.2M images. The internet has billions of unlabeled images. Self-supervised learning (SSL) learns representations from unlabeled data.

### 9.2 SimCLR (2020)

**Contrastive learning**: for each image, create two augmented views. Maximize agreement between views of the same image, minimize agreement with all other images in the batch.

```python
# SimCLR loss (NT-Xent)
def simclr_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch_size, dim) — normalized embeddings
    """
    N = z1.shape[0]

    # Concatenate and compute similarity matrix
    z = torch.cat([z1, z2], dim=0)  # (2N, dim)
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Remove self-similarity
    sim.fill_diagonal_(-1e9)

    # Labels: for i in [0,N), positive pair is i+N; for i in [N,2N), positive is i-N
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(z.device)

    return F.cross_entropy(sim, labels)
```

Critical augmentations: random crop + resize, color jitter, grayscale, Gaussian blur.

Weakness: requires large batch sizes (4096-8192) for enough negatives.

### 9.3 MoCo (2020)

**Momentum Contrast**: maintain a large queue of negative keys (encoded by a momentum-updated encoder), decoupling batch size from number of negatives.

```
Online encoder: theta
Momentum encoder: theta_m = m * theta_m + (1-m) * theta  (m=0.999)
```

The momentum encoder provides stable keys without backpropagation. Queue stores embeddings from recent batches.

MoCo v3 drops the queue and uses ViT backbone — simpler and competitive with SimCLR.

### 9.4 DINO (2021)

**Self-Distillation with No Labels**: student and teacher network, where teacher is an EMA of the student (like MoCo). Key insight: ViT trained with DINO produces attention heads that naturally segment objects without any segmentation supervision.

```python
# DINO's self-distillation loss (cross-entropy between softened distributions)
def dino_loss(student_out, teacher_out, temperature_s=0.1, temperature_t=0.04):
    teacher_probs = F.softmax(teacher_out / temperature_t, dim=-1).detach()
    student_log_probs = F.log_softmax(student_out / temperature_s, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()
```

DINO features excel at k-NN classification, dense prediction, and are the backbone of SAM (Segment Anything Model).

### 9.5 MAE (Masked Autoencoders, 2022)

**Masked prediction** inspired by BERT: mask a high fraction (75%) of image patches, reconstruct the masked patches.

Key design choices:
- **High masking ratio** (75%): forces model to learn meaningful representations, not just copy nearby patches
- **Asymmetric encoder-decoder**: encoder only sees visible patches (fast), lightweight decoder reconstructs masked ones
- **Pixel reconstruction**: predict raw pixel values, not discrete tokens

```python
# MAE forward pass sketch
def mae_forward(images, mask_ratio=0.75):
    patches = patchify(images)           # (N, num_patches, patch_dim)
    ids_shuffle = torch.argsort(torch.rand(N, L), dim=1)
    
    # Keep only visible patches for encoder
    keep = int(L * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :keep]
    x_visible = patches.gather(1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
    
    # Encode visible patches
    latent = encoder(x_visible)
    
    # Decode: visible + mask tokens
    mask_tokens = mask_token.expand(N, L - keep, -1)
    x_full = torch.cat([latent, mask_tokens], dim=1)
    # unshuffle
    pred = decoder(x_full)
    
    # Loss only on masked patches
    loss = ((pred - patches) ** 2).mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

MAE ViT-L fine-tuned on ImageNet achieves 85.9% top-1, competitive with supervised ViT-L.

### 9.6 Comparison

| Method | Approach | Key Requirement | Strength |
|---|---|---|---|
| SimCLR | Contrastive | Large batch | Simple, strong |
| MoCo | Contrastive + queue | Moderate batch | Memory efficient |
| DINO | Self-distillation | Multi-crop aug | Emergent segmentation |
| MAE | Masked prediction | Standard batch | ViT-native, fast train |

---

## 10. Data Augmentation Strategies

### 10.1 Basic Augmentations

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Random erasing**: randomly zero out a rectangle of the image, forcing focus on other regions.

**Test-time augmentation (TTA)**: at inference, average predictions over multiple augmented views.

### 10.2 Mixup

Interpolate two training images and their labels:

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

Forces the model to interpolate between classes, acting as strong regularization. Particularly effective for fine-grained classification.

### 10.3 CutMix

Cut a rectangular region from one image and paste it into another, mixing labels proportionally to area:

```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    W, H = x.size(3), x.size(2)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed_x, y, y[index], lam
```

CutMix outperforms Mixup for object detection and segmentation (mixed patches contain meaningful regions, not blended artifacts).

### 10.4 AutoAugment

Learn augmentation policies from data. Defines a search space of (operation, magnitude, probability) triplets (e.g., rotate by 30 degrees with probability 0.5). Uses reinforcement learning or evolution to find the best policy for a given dataset.

**RandAugment** (simplified): randomly sample augmentation operations and magnitudes. Two hyperparameters: N (number of ops) and M (magnitude).

```python
from torchvision.transforms import RandAugment

transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor()
])
```

### 10.5 TrivialAugment and AugMix

**TrivialAugment**: even simpler — uniformly sample one operation, uniformly sample magnitude. Outperforms RandAugment on ImageNet.

**AugMix**: create multiple augmented views, mix them with Dirichlet coefficients, enforce consistency between mixed and original via Jensen-Shannon divergence loss. Improves robustness to distribution shift.

---

## 11. Evaluation Metrics

### 11.1 Classification

**Top-1 accuracy**: fraction where the highest-probability class is correct.
**Top-5 accuracy**: fraction where the correct class is among the top 5 predictions.

### 11.2 mAP (mean Average Precision) for Detection

For each class:
1. Sort detections by confidence (descending)
2. For each detection, check if IoU with ground truth > threshold (e.g., 0.5)
3. Compute precision-recall curve
4. Average Precision (AP) = area under PR curve

mAP = mean of AP over all classes.

**COCO mAP**: average mAP over IoU thresholds 0.50:0.05:0.95 (10 thresholds). More stringent than VOC mAP at IoU=0.50 only.

```
COCO mAP = mean over [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
AP_50:95 = (AP_50 + AP_55 + ... + AP_95) / 10
```

### 11.3 Segmentation Metrics

**IoU (mIoU)**: per-class IoU averaged over classes.
```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = (1/C) * sum_c IoU_c
```

**Pixel accuracy**: fraction of pixels correctly classified (misleading when classes are imbalanced — background dominates).

### 11.4 Generative Metrics

**FID (Frechet Inception Distance)**: measures distance between real and generated image distributions in Inception feature space.

```
FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^(1/2))
```

Lower FID = better. Requires ~10k generated samples for stable estimates.

**Inception Score (IS)**: measures both quality (sharp predictions) and diversity (uniform marginal distribution):
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

Higher IS = better. Insensitive to mode collapse where modes match training data distribution.

**CLIP Score**: cosine similarity between CLIP embeddings of generated images and text prompts. Used for text-to-image model evaluation.

---

## 12. Production Considerations

### 12.1 Model Quantization

Full-precision inference uses FP32 (4 bytes/weight). Quantization reduces to INT8 (1 byte) or INT4 (0.5 byte).

**Post-Training Quantization (PTQ)**: calibrate scales using a small calibration dataset, then quantize.

```python
import torch.quantization

model.eval()
# Prepare: insert observers for calibration
model_prepared = torch.quantization.prepare(model)

# Calibrate on ~100 representative samples
with torch.no_grad():
    for x, _ in calibration_loader:
        model_prepared(x)

# Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)

# Check size reduction
print(f"FP32 model: {get_size_mb(model):.2f} MB")
print(f"INT8 model: {get_size_mb(model_quantized):.2f} MB")
# Typically ~4x size reduction, ~2-4x speedup
```

**Quantization-Aware Training (QAT)**: simulate quantization during training. Better accuracy but requires retraining.

### 12.2 Pruning

Remove weights below a magnitude threshold, making the model sparse.

**Unstructured pruning**: zero out individual weights. Hardware speedup requires specialized sparse kernels (limited support).

**Structured pruning**: remove entire filters or channels. Hardware-friendly — reduces the network's width.

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in conv1 by magnitude
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# After pruning, fine-tune to recover accuracy
```

### 12.3 Knowledge Distillation

Train a small **student** model to mimic a large **teacher** model.

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.7):
    # Soft targets from teacher (high temperature = softer distribution)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * temperature ** 2

    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 12.4 ONNX Export

ONNX (Open Neural Network Exchange) enables deployment across runtimes (TensorRT, OpenVINO, CoreML, ONNX Runtime).

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)

# Validate
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': dummy_input.numpy()})
```

### 12.5 TensorRT for Edge/Server Inference

```python
# After ONNX export, convert to TensorRT engine for maximum GPU throughput
# trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

import tensorrt as trt
import pycuda.driver as cuda

# FP16 inference: typically 2x faster than FP32 on NVIDIA GPUs with minimal accuracy loss
```

### 12.6 Deployment Checklist

| Step | Tool | Notes |
|---|---|---|
| Profile model | `torch.profiler` | Identify bottleneck layers |
| Quantize | PyTorch PTQ / QAT | INT8: 4x smaller, 2-4x faster |
| Prune | `torch.nn.utils.prune` | Structured pruning for hardware speedup |
| Distill | Custom training loop | Student 5-10x smaller |
| Export | ONNX | Cross-runtime compatibility |
| Optimize | TensorRT / CoreML | Platform-specific |
| Validate | Accuracy + latency on target device | Do not skip this |

---

## 13. Common Interview Questions

---

**Q: Why do CNNs work so well for images compared to fully connected networks?**

CNNs exploit three properties of natural images: (1) **locality** — nearby pixels are more correlated than distant ones; CNNs process local patches rather than the full image at once; (2) **weight sharing** — the same features (edges, textures) appear in different locations; one filter detects all instances; (3) **hierarchical composition** — low-level features compose into higher-level ones. A fully connected layer on a 224x224 image has 224*224*3 = 150k inputs per neuron. For a 1000-unit hidden layer, that is 150M parameters in one layer, compared to 64 * 3 * 3 * 3 = 1728 for a single conv layer.

---

**Q: Explain how ResNet's skip connections help training.**

Without skip connections, a 100-layer network backpropagates gradients through 100 multiplicative operations. If each has magnitude < 1 (common with sigmoid or tanh), gradients vanish exponentially. ResNet's skip connection `y = F(x) + x` means the gradient has two paths: through the residual `F(x)` and directly through `x`. The direct path provides a gradient of exactly 1 from the output, regardless of the depth. Additionally, the skip connection makes it easier for the network to learn the identity mapping — the residual only needs to learn small corrections, which is a simpler optimization target than learning the full transformation from scratch.

---

**Q: What is the difference between semantic segmentation and instance segmentation?**

Semantic segmentation assigns a class label to every pixel, but cannot distinguish between different instances of the same class. If two cars overlap, all their pixels get the label "car" with no distinction. Instance segmentation additionally assigns a unique identity to each object instance — each car gets its own mask. Mask R-CNN achieves this by extending Faster R-CNN with a mask prediction head per detected instance. Panoptic segmentation unifies both: semantic labels for "stuff" (sky, road) and instance masks for "things" (cars, people).

---

**Q: How does the Vision Transformer differ from a CNN?**

ViT splits an image into `P x P` patches and processes them as a sequence with Transformer self-attention. Key differences: (1) **No inductive bias** — ViT makes no assumptions about locality or translation equivariance; it must learn these from data; (2) **Global receptive field** — every patch attends to every other patch from layer 1; CNNs build up global receptive field gradually through depth; (3) **Data efficiency** — CNNs are more sample-efficient because their inductive biases match image statistics; ViT needs more data or strong pretraining; (4) **Quadratic complexity** — standard ViT attention is O(N^2) in number of patches; efficient variants (Swin, Linformer) address this.

---

**Q: Explain focal loss and why it was needed.**

In one-stage detection, anchors are generated densely across the image. For a 800x800 image with anchors at multiple scales and ratios, there can be 100,000+ anchors. Fewer than 1000 overlap with ground truth objects — so >99% of anchors are background (easy negatives). With standard cross-entropy, easy negatives contribute the bulk of the loss (they are numerous, even though each contributes a tiny loss). The model gets overwhelmed learning "this is background" rather than learning to detect objects. Focal loss adds `(1-p_t)^gamma` to the loss. For easy examples (`p_t` near 1), this factor approaches 0, down-weighting their contribution by orders of magnitude. Hard examples (`p_t` near 0) retain their full loss weight.

---

**Q: What is CLIP and how does zero-shot classification work?**

CLIP trains a vision encoder and text encoder jointly on 400M (image, text) pairs using contrastive loss — matching pairs should have high cosine similarity, non-matching pairs low similarity. This creates a shared embedding space where visual and textual representations of the same concept are nearby.

Zero-shot classification: for each candidate class, create a text prompt ("a photo of a dog"), encode with the text encoder, and compute cosine similarity to the image's visual embedding. The class with the highest similarity is the prediction. No training examples for the target classes are needed — the model generalizes via the shared embedding space learned from web-scale data.

---

**Q: Compare GANs, VAEs, and Diffusion models for image generation.**

**GANs**: adversarial training produces sharp, high-quality images but suffers from training instability, mode collapse, and difficult evaluation. Very fast sampling (single forward pass). Hard to control without conditioning.

**VAEs**: maximize ELBO, producing a smooth, structured latent space. Training is stable. Images tend to be blurry due to the reconstruction loss encouraging averaging over modes. The continuous latent space enables smooth interpolation.

**Diffusion models**: iterative denoising over T steps (e.g., 1000). Training is stable (score matching). Produce the highest-quality and most diverse images. Slow sampling (T forward passes) though DDIM reduces this to 20-50 steps. State-of-the-art for unconditional and conditional generation (Stable Diffusion, DALL-E 3, Imagen).

Current consensus: diffusion models have replaced GANs for most generation tasks where sampling speed is not critical.

---

**Q: What is the difference between Mixup and CutMix, and when would you prefer one over the other?**

Mixup interpolates entire images: `x_mixed = lam * x_a + (1-lam) * x_b`. The result is a blended, unrealistic image. Labels are soft mixtures of both classes. Forces the model to produce calibrated, confident-but-not-certain predictions and smooths decision boundaries.

CutMix pastes a rectangular region from one image into another. The pasted region is a realistic image patch, not a blend. Labels are proportional to pixel area. Forces the model to use local regions for classification rather than global cues.

**Prefer Mixup**: classification tasks where global features matter, regularization is the main goal.
**Prefer CutMix**: detection/segmentation tasks (CutMix preserves local object semantics), when you want the model to be robust to occlusion.

---

**Q: How would you adapt a pretrained ImageNet model to a medical imaging task with limited data?**

1. **Use pretrained backbone**: despite domain shift, ImageNet features (edges, textures) generalize. Transfer learning is almost always better than random initialization even for medical images.
2. **Progressive unfreezing**: first train only the head (classifier), then unfreeze later backbone layers, then earlier layers. Use a lower learning rate for earlier layers (discriminative fine-tuning / layer-wise LR decay).
3. **Domain-specific augmentation**: for histopathology — stain normalization, random stain augmentation. For X-ray — random rotation ±15 degrees, brightness jitter. Avoid augmentations that destroy diagnostically relevant features (e.g., horizontal flip may not be valid for asymmetric anatomy).
4. **Regularization**: strong dropout (0.5), weight decay, early stopping.
5. **SSL pretraining**: if unlabeled medical data is available, pretrain with SimCLR or MAE on the unlabeled data before fine-tuning with labels.
6. **Smaller architecture**: with very limited data (<1000 images), use a smaller backbone (ResNet-18 vs ResNet-50) to reduce overfitting risk.

---

**Q: What is model quantization and what are the tradeoffs?**

Quantization reduces numerical precision of weights and activations, typically from FP32 to INT8 or INT4. Benefits: 4x smaller model size (INT8), 2-4x faster inference on hardware with INT8 units (NVIDIA Tensor Cores, Apple Neural Engine, Qualcomm Hexagon), lower energy consumption.

Tradeoffs: (1) **Accuracy degradation**: small but non-zero. Usually < 1% top-1 drop for INT8 with calibration. INT4 can lose 1-3%. (2) **Calibration requirement**: PTQ needs a representative calibration dataset to compute quantization scales. (3) **Layer compatibility**: some operations (e.g., LayerNorm, attention softmax) require higher precision — these may remain FP32 in mixed-precision. (4) **Hardware dependency**: INT8 speedup only materializes on hardware with INT8 SIMD or tensor cores; on CPU without INT8 support, quantized models may not be faster.

---
