# Segmentation & Pose Estimation — Comprehensive Reference

> Covers semantic segmentation (FCN → U-Net → DeepLab → SegFormer), instance segmentation (Mask R-CNN → SOLOv2 → Mask2Former), panoptic segmentation, and pose estimation (OpenPose → HRNet → ViTPose). Metrics, math, and interview points included.

---

## Table of Contents

1. [Semantic Segmentation](#1-semantic-segmentation)
2. [Instance Segmentation](#2-instance-segmentation)
3. [Panoptic Segmentation](#3-panoptic-segmentation)
4. [Pose Estimation](#4-pose-estimation)
5. [Key Metrics](#5-key-metrics)
6. [Key Interview Points](#6-key-interview-points)

---

## 1. Semantic Segmentation

**Task definition:** assign a class label to *every pixel* in an image. Output is a `H × W` label map. All pixels belonging to the same class share one label — there is no distinction between individual instances of the same class (both sheep are "sheep").

### 1.1 FCN — Fully Convolutional Network (Long et al., 2015)

The first end-to-end trainable dense prediction network. Key ideas:

- Replace fully connected layers in a classification backbone (e.g., VGG-16) with 1×1 convolutions, preserving spatial dimensions.
- Upsample the coarse prediction map back to the input resolution with **bilinear interpolation** or **transposed convolutions**.
- **Skip connections** fuse fine spatial detail from earlier (higher-resolution) layers with coarse semantic features from deeper layers.

```
Input → [Conv backbone] → 1/32 pred
                    ↑ skip (pool4, 1/16) → FCN-16s
                    ↑ skip (pool3, 1/8)  → FCN-8s
```

FCN-8s outperforms FCN-32s by recovering finer spatial details lost through stride-32 downsampling.

**Limitation:** fixed receptive field; no explicit mechanism to capture long-range context.

---

### 1.2 U-Net (Ronneberger et al., 2015)

Designed for biomedical images but now ubiquitous. Architecture:

- **Encoder (contracting path):** successive `Conv → Conv → MaxPool` blocks halve spatial size while doubling channels.
- **Decoder (expansive path):** `TransposedConv (×2) → Concat(skip) → Conv → Conv` restores spatial size.
- **Skip connections:** concatenate (not add) encoder feature maps directly into the corresponding decoder level — preserves precise localization.
- Trained with cross-entropy loss (or Dice loss for class imbalance).

```
Encoder:  3→64→128→256→512→1024  (spatial: 572→286→143→71→35)
Bottleneck: 1024
Decoder:  1024→512→256→128→64→C  (spatial: 35→71→143→286→572)
```

**Dice Loss** (handles class imbalance better than cross-entropy):

```
Dice = 1 - (2 * |P ∩ G|) / (|P| + |G|)
```

where P = predicted foreground, G = ground truth foreground.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv-BN-ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 64):
        super().__init__()
        b = base
        # Encoder
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(b * 8, b * 16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)   # concat doubles channels
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)   # (B, num_classes, H, W)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """pred: (B, C, H, W) logits; target: (B, H, W) long."""
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
    inter = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()
```

---

### 1.3 DeepLab v3+ (Chen et al., 2018)

Addresses the tension between resolution and receptive field using **atrous (dilated) convolutions**.

**Atrous convolution** with rate `r` inserts `r-1` zeros between filter taps, expanding the receptive field by factor `r` *without* increasing parameters or reducing resolution:

```
y[i] = Σ_k x[i + r·k] · w[k]
```

**ASPP (Atrous Spatial Pyramid Pooling):** applies parallel atrous convolutions with rates {1, 6, 12, 18} plus global average pooling, then concatenates results. Captures multi-scale context in a single forward pass.

**Encoder-Decoder with Xception backbone:**

```
Input
  └─ Xception encoder (stride 16 output, low-level features at stride 4)
       └─ ASPP module → 1×1 conv → 256-ch feature
            └─ 4× bilinear upsample
                 └─ concat(low-level features from stride-4)
                      └─ 3×3 conv → 1×1 head → predictions
                           └─ 4× bilinear upsample → full resolution output
```

**Depthwise separable convolutions** replace standard 3×3 convs in ASPP and decoder:

```
Standard 3×3 conv:        K² · C_in · C_out  parameters
Depthwise separable:      K² · C_in  (depthwise)  +  C_in · C_out  (pointwise)
Speedup ratio ≈ 1/(C_out) + 1/K²  ≈ 8–9× for K=3
```

---

### 1.4 SegFormer (Xie et al., 2021)

Pure-Transformer semantic segmentation that avoids positional encoding limitations.

**Hierarchical Mix Transformer (MiT) encoder:**
- 4 stages at strides {4, 8, 16, 32} — produces multi-scale feature maps like a CNN pyramid.
- **Efficient Self-Attention:** reduce sequence length by reshaping `K, V` with ratio `R` before attention (`R ∈ {64, 16, 4, 1}` across stages) — cost goes from `O(N²)` to `O(N²/R)`.
- **Mix-FFN:** `x + Conv3×3(FFN(x))` — the local 3×3 conv implicitly encodes positional information, removing the need for fixed positional embeddings (enabling arbitrary resolution inference).

**Lightweight MLP decoder:**
- Upsample each stage's features to stride-4 with bilinear interpolation.
- Concatenate → one MLP → segmentation head.
- No cross-attention, no FPN: O(1) decoder complexity.

```
MiT Stage1 (1/4)  ──┐
MiT Stage2 (1/8)  ──┤ upsample all to 1/4 → concat → MLP → head → 4× up → output
MiT Stage3 (1/16) ──┤
MiT Stage4 (1/32) ──┘
```

SegFormer-B5 achieves 84.0 mIoU on ADE20K at ~80M params; SegFormer-B0 achieves 37.4 mIoU at 3.7M params.

---

## 2. Instance Segmentation

**Task definition:** detect and delineate *each individual object instance* with a binary mask. Two objects of the same class (e.g., two people) get separate masks and separate IDs. Output: a set of `(class, score, binary mask)` tuples, not a single label map.

---

### 2.1 Mask R-CNN (He et al., 2017)

Extends Faster R-CNN by adding a **mask head** running in parallel with the box regression and classification heads.

**Pipeline:**
```
Image → Backbone (ResNet + FPN)
      → RPN → top-K region proposals
           → RoIAlign → 7×7 or 14×14 feature crops
                → [Box head: cls + bbox regression]
                → [Mask head: 4× Conv → deconv → sigmoid → 28×28 binary mask per class]
```

**RoIPool vs RoIAlign:**

| | RoIPool | RoIAlign |
|---|---|---|
| Quantization | 2× (proposal → feature, feature → grid) | None — uses bilinear interpolation |
| Effect | Misalignment artifacts hurt mask quality | Pixel-accurate alignment |
| AP improvement | baseline | +2–3 AP on COCO |

RoIAlign computes each sample point as a bilinear interpolation of the 4 nearest feature map cells:

```
feature(x, y) = Σ_{i,j ∈ neighbors} w_ij · F[i, j]
   where w_ij = (1 - |x - i|)(1 - |y - j|)
```

**Mask head:** 4 consecutive 3×3 convs (256 channels) + 2×2 deconv → 28×28 output with `C` channels (one per class). Binary cross-entropy loss applied only to the ground-truth class channel — masks are class-independent during training.

**Multi-task loss:**
```
L = L_cls + L_box + L_mask
L_mask = binary_cross_entropy(pred_mask[gt_class], gt_mask)   # average pixel-wise
```

---

### 2.2 SOLOv2 (Wang et al., 2020)

Eliminates region proposals entirely. Assigns each instance to a grid cell based on its center location and size.

**Key idea:** segment objects by *location* — divide the image into an `S×S` grid. Each cell (i, j) is responsible for the instance whose center falls in that cell.

**Two heads (FPN features as input):**

1. **Category head:** `S×S×C` — predicts class probability for each grid cell.
2. **Mask kernel head:** `S×S×D` — predicts a D-dimensional weight vector per cell.

**Mask generation:** a shared **mask feature map** `(H/4 × W/4 × D)` is produced by the feature network. Each instance mask is the *inner product* of its kernel vector with the mask feature map, then sigmoid:

```
mask_ij = sigmoid(kernel_ij · mask_features)   # (H/4, W/4)
```

**Advantages over Mask R-CNN:**
- No NMS on proposals — uses matrix NMS on final masks (suppresses masks by overlap score).
- Faster inference; better for heavily overlapping objects.
- No RoI operations, fully convolutional.

---

### 2.3 DETR-based Instance Segmentation — Mask2Former (Cheng et al., 2022)

**MaskDETR / Mask2Former** unify detection and segmentation using *masked attention Transformers*. Mask2Former serves as a single architecture for semantic, instance, and panoptic segmentation (see Section 3).

**Transformer decoder with masked attention:**

Standard cross-attention attends over the entire image feature map (quadratic cost for high-res). Mask2Former restricts each query's attention to a predicted foreground region:

```
Attention(Q, K, V) with mask M:
  softmax( (QK^T / sqrt(d)) + M )V
  where M[h, i, j] = 0 if query i should attend to position j, else -inf
```

At each decoder layer, the current mask prediction is used to restrict attention for the *next* layer — iterative refinement.

**Architecture:**

```
Image → backbone + pixel decoder (FPN or MSDeformAttn) → multi-scale feature maps
      → L-layer Transformer decoder (N learnable queries)
           → class predictions + binary mask predictions (via inner product with pixel features)
```

Training with Hungarian matching + per-mask binary cross-entropy + Dice loss.

---

## 3. Panoptic Segmentation

**Task definition:** unify semantic segmentation (dense per-pixel labels) and instance segmentation (per-instance masks) into one output. Every pixel gets a `(class, instance_id)` label.

- **Stuff** classes (amorphous, uncountable): sky, road, grass — handled like semantic segmentation; all pixels share one label, `instance_id = 0`.
- **Things** classes (countable, discrete objects): person, car, dog — each instance gets a unique ID, handled like instance segmentation.

### 3.1 Panoptic FPN (Kirillov et al., 2019)

Adds a lightweight semantic segmentation branch to the Mask R-CNN + FPN framework:

```
Backbone → FPN
         → [Mask R-CNN head: things instances]
         → [Semantic head: stuff + things combined map]
→ Merging: instance masks override semantic predictions for things
```

Trained with combined loss; merging at inference uses heuristics (e.g., prefer high-confidence instances over the semantic prediction).

### 3.2 Mask2Former as Unified Model

Mask2Former achieves state-of-the-art on all three tasks with one architecture by changing only the training data and loss weighting — no architectural changes. Queries learn to predict both stuff regions (single mask per class) and things instances (one mask per instance).

### 3.3 Panoptic Quality (PQ) Metric

PQ decomposes into recognition quality (RQ) and segmentation quality (SQ):

```
PQ = SQ × RQ
   = [Σ_{(p,g)∈TP} IoU(p,g) / |TP|]  ×  [|TP| / (|TP| + ½|FP| + ½|FN|)]
```

- **TP** (matched pairs): IoU > 0.5 between predicted and ground-truth segment.
- Computed separately for stuff and things, then averaged across all categories.
- PQ = 0 if a category has no ground truth and no predictions (not penalized).

**Typical COCO numbers:** Mask2Former achieves ~57 PQ on COCO panoptic; Panoptic FPN ~40 PQ.

---

## 4. Pose Estimation

**Task definition:** detect a set of anatomical **keypoints** (joints) for each person in an image — e.g., 17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. Output: `(x, y, visibility)` per keypoint per person.

Two paradigms:
- **Top-down:** detect people first (bounding boxes) → estimate pose per crop. Accurate; scales linearly with people count.
- **Bottom-up:** detect all keypoints globally → group them into individuals. Fast; fixed inference time regardless of people count.

### 4.1 Heatmap-based Keypoint Representation

Both paradigms typically represent each keypoint as a 2D Gaussian heatmap:

```
H_k(x, y) = exp( -((x - x_k)² + (y - y_k)²) / (2σ²) )
```

where `(x_k, y_k)` is the ground-truth location of keypoint `k`. The network predicts `K` heatmaps; the predicted location is `argmax` of each heatmap.

**Loss:** MSE between predicted and ground-truth heatmaps, ignoring invisible keypoints.

---

### 4.2 OpenPose (Cao et al., 2017) — Bottom-up

Introduces **Part Affinity Fields (PAFs)**: 2D vector fields that encode the orientation and location of limbs connecting pairs of keypoints.

**Two-branch architecture (VGG backbone):**

```
Image → VGG features
      → Branch 1: PAF estimation (2×L vector maps, L = #limbs)
      → Branch 2: Confidence maps (K heatmaps, K = #keypoints)
      → Iterative refinement (multiple stages using both branches)
      → Bipartite matching (Hungarian) using PAF line integrals to group keypoints into skeletons
```

**PAF line integral** for candidate limb (d_a, d_b):

```
E = ∫_{u=0}^{1} L(p(u)) · v(p(u)) du
   where p(u) = (1-u)·d_a + u·d_b,  v = (d_b - d_a)/|d_b - d_a|
```

High E = strong limb connection. This allows grouping in O(K²) without per-person region proposals.

---

### 4.3 HRNet (Sun et al., 2019) — Top-down

Maintains **high-resolution representations throughout** the entire network — unlike most backbones that progressively downsample.

**Architecture:**

```
Stage 1: single high-res branch (1/4 resolution)
Stage 2: add parallel 1/8 branch; exchange features
Stage 3: add parallel 1/16 branch; exchange features
Stage 4: add parallel 1/32 branch; exchange features
→ aggregate all branches (upsample lower-res to 1/4) → heatmap head
```

**Multi-scale fusion (repeated exchange units):**
- Each branch receives summed contributions from all other branches (strided conv to downsample, bilinear upsample to go up).
- The high-res branch never loses spatial detail, making HRNet exceptional for precise localization.

HRNet-W48 achieves 76.3 AP on COCO val2017 (top-down, single-scale).

---

### 4.4 ViTPose (Xu et al., 2022) — Transformer Top-down

Uses a plain ViT backbone with minimal pose-specific design:

```
Person crop → ViT backbone (patch embed → L transformer layers) → feature map (H/16, W/16, D)
           → lightweight decoder (2× deconv or simple upsample)
           → K heatmaps
```

**Key findings:**
- A plain ViT (no hierarchical design, no FPN) outperforms HRNet when pretrained on MAE.
- Larger ViT → consistently better pose AP; ViT-H achieves 80.9 AP on COCO val.
- Decoder is deliberately minimal — performance comes from the pretrained representation, not decoder complexity.
- Supports **partial finetuning**: freeze ViT body, train only decoder and last few layers → 95% of full-finetune performance.

```python
# ViTPose inference using mmpose / transformers ecosystem
from transformers import ViTPoseForPoseEstimation, ViTPoseImageProcessor
from PIL import Image
import torch

processor = ViTPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model     = ViTPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")

image  = Image.open("person.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.heatmaps: (B, K, H/4, W/4) — one heatmap per keypoint
keypoints, scores = processor.post_process_pose_estimation(
    outputs, boxes=[[[0, 0, image.width, image.height]]]
)[0][0]["keypoints"], \
processor.post_process_pose_estimation(
    outputs, boxes=[[[0, 0, image.width, image.height]]]
)[0][0]["scores"]
```

---

## 5. Key Metrics

### 5.1 mIoU — Semantic Segmentation

**Intersection over Union** for class `c`:
```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
      = |pred_c ∩ gt_c| / |pred_c ∪ gt_c|
```

**mean IoU** averages over all `C` classes:
```
mIoU = (1/C) Σ_c IoU_c
```

- Standard benchmark metric for PASCAL VOC and ADE20K.
- Unaffected by true negatives — correct background predictions don't inflate the score.

### 5.2 AP — Instance / Object Detection

**Average Precision** integrates the precision-recall curve:
```
AP = Σ_k (R_k - R_{k-1}) · P_k
```

- A prediction is a TP if its IoU with the matched ground-truth box/mask exceeds a threshold.
- **AP@50**: threshold = 0.5 (lenient). **AP@75**: threshold = 0.75 (strict).
- **AP** (COCO): average of AP at thresholds {0.50, 0.55, ..., 0.95} — 10 thresholds.
- **mAP**: mean AP over all categories.
- Instance segmentation uses **mask IoU** instead of box IoU.

### 5.3 Panoptic Quality (PQ)

Defined in Section 3.3. Reported as `PQ`, `PQ_th` (things), `PQ_st` (stuff).

### 5.4 OKS — Object Keypoint Similarity

Analogous to IoU for keypoints. For a single keypoint `k`:
```
OKS_k = exp( -d_k² / (2 · s² · σ_k²) ) · δ(v_k > 0)
```

- `d_k`: Euclidean distance between predicted and ground-truth keypoint.
- `s`: object scale (square root of bounding box area).
- `σ_k`: per-keypoint constant reflecting annotation noise (e.g., hip σ = 0.107, eye σ = 0.025).
- `v_k`: visibility flag.

**Overall OKS** for a person = mean over keypoints. AP is computed using OKS as the match threshold (analogous to IoU in object detection). COCO reports AP at OKS ∈ {0.50, 0.55, ..., 0.95}.

### 5.5 PCKh — Percentage of Correct Keypoints (head-normalized)

Used on MPII dataset:
```
PCKh@α: keypoint is correct if  d_k < α · h
```

where `h` is the head segment length (distance between head-top and upper-neck keypoints) for that person, and `α = 0.5` is the standard threshold. Reported as the percentage of correctly localized keypoints across all test images.

PCKh is robust to varying person scales because it normalizes by a person-specific measurement rather than a fixed pixel distance.

---

## 6. Key Interview Points

**Semantic vs instance vs panoptic segmentation**
- Semantic: per-pixel class, no instance distinction (two cars = same label).
- Instance: per-object mask + class, no dense coverage (background unlabeled).
- Panoptic: combines both — every pixel labeled with (class, instance_id); background (stuff) gets instance_id = 0.

**Why RoIAlign over RoIPool?**
RoIPool quantizes twice (proposal coordinates → feature map cells, then feature cells → fixed grid). This misalignment can be 0.5 cell — acceptable for detection but significant for 28×28 masks. RoIAlign uses bilinear interpolation at sub-pixel sample points, eliminating quantization and recovering ~2–3 mask AP on COCO.

**How does ASPP capture multi-scale context?**
ASPP applies atrous convolutions at multiple dilation rates in parallel. Each rate gives a different effective receptive field. Concatenating their outputs lets a single pixel's prediction see context at scales ranging from local texture (rate 1) to large regions (rate 18). The global average pooling branch captures image-level context.

**What problem do skip connections solve in U-Net vs FCN?**
FCN adds skip connections via *element-wise addition* of upsampled predictions with shallower feature maps — it adds back spatial information after aggressive downsampling. U-Net uses *concatenation* which is more expressive (doubles the channel count, so the next conv can learn how to merge the two feature sets rather than averaging them). U-Net also uses symmetric encoder-decoder depths with skip at every level.

**Bottom-up vs top-down pose estimation trade-offs**
- Top-down (HRNet, ViTPose): crop each detected person → pose network. Accurate but inference time scales with person count; depends on detector quality.
- Bottom-up (OpenPose): one forward pass for the whole image → fixed inference cost. Harder to achieve high precision for closely overlapping people; PAF grouping adds post-processing complexity.

**Why does ViTPose outperform HRNet despite a simpler decoder?**
The ViT backbone pretrained with MAE learns rich, spatially-aware representations that transfer well to localization. The high-capacity Transformer body compensates for the minimal decoder. HRNet's advantage — maintaining high resolution throughout — is less critical when the encoder already has strong representations that bilinear upsampling can restore.

**SOLOv2 kernel trick — how does it avoid region proposals?**
Instead of cropping features for each proposal and running a mask head, SOLOv2 generates a global mask feature map once. Each grid cell predicts a lightweight D-dim weight vector. The instance mask is the inner product (dot product) of that vector with the global mask feature map — essentially a conditional convolution. This is ~5× faster than Mask R-CNN on equivalent hardware.

**mIoU vs AP — when to use which?**
- mIoU: natural for semantic segmentation where every pixel must be labeled; averages over classes with equal weight.
- AP: natural for detection/instance tasks where the number of predicted instances per image is variable and confidence-ranked; the precision-recall curve captures this.
- Panoptic uses PQ (= SQ × RQ) because it combines both types of segments and needs to penalize both missed instances (FN) and spurious ones (FP).

**Dice loss vs cross-entropy for segmentation**
Cross-entropy is computed per-pixel and averages equally — small foreground regions contribute few pixels, making the loss dominated by background. Dice loss directly optimizes the IoU-like overlap ratio, naturally handling class imbalance. In practice: use cross-entropy + Dice as a combined loss for best convergence.

**SegFormer's trick for resolution-agnostic inference**
Standard ViT uses fixed positional embeddings — if you change the input resolution at test time, the positional embeddings no longer correspond to the right locations. SegFormer removes positional embeddings and replaces them with depth-wise 3×3 convolutions inside the FFN (Mix-FFN). The convolutions provide implicit positional information without fixing the resolution, enabling arbitrary-resolution inference without interpolation.
