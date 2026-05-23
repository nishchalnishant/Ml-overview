---
module: Deep Learning
topic: Methods
subtopic: Segmentation
status: unread
tags: [deeplearning, ml, methods-segmentation]
---
# Segmentation & Pose Estimation

> Covers semantic segmentation (FCN → U-Net → DeepLab → SegFormer), instance segmentation (Mask R-CNN → SOLOv2 → Mask2Former), panoptic segmentation, and pose estimation (OpenPose → HRNet → ViTPose). Metrics and interview points included.

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

**The task:** assign a class label to *every pixel*. Output is an H×W label map. Two pixels of the same class share one label — there is no distinction between individual instances.

---

### 1.1 FCN — Fully Convolutional Network (Long et al., 2015)

**The problem:** classification backbones (VGG, AlexNet) end with fully-connected layers that collapse spatial dimensions to a class score. You can't recover per-pixel predictions from a scalar. How do you adapt a classification network to predict a class at every spatial location?

**The core insight:** replace fully-connected layers with 1×1 convolutions. This keeps spatial dimensions alive throughout the network. The coarse, semantically-rich feature map at the end is then upsampled back to input resolution. Add skip connections from earlier layers to recover the spatial detail lost by stride-32 downsampling.

**The mechanics:**
```
Input → [Conv backbone] → 1/32 prediction
                    ↑ skip (pool4, stride 16) → FCN-16s
                    ↑ skip (pool3, stride 8)  → FCN-8s
```

FCN-8s recovers finer spatial detail than FCN-32s by fusing features from layers before the resolution collapse.

**What breaks:** The receptive field is fixed by the architecture. There is no mechanism to aggregate context from distant pixels — an isolated pixel has no way to know whether the region it sits in is sky or water without looking far enough.

---

### 1.2 U-Net (Ronneberger et al., 2015)

**The problem:** FCN's skip connections add information back after downsampling, but the decoder is shallow. For precise boundary delineation — especially in medical images with fine structures — you need a decoder that is as deep as the encoder, with feature information passed across at every resolution level.

**The core insight:** build a symmetric encoder–decoder where skip connections *concatenate* (not add) encoder feature maps directly into the corresponding decoder level. Concatenation doubles the channel count, giving the decoder the option to selectively use encoder features rather than forcing an average.

**The mechanics:**
```
Encoder: 3→64→128→256→512→1024  (spatial: 572→286→143→71→35)
Bottleneck: 1024
Decoder: 1024→512→256→128→64→C (spatial: 35→71→143→286→572)
         each step: TransposedConv(×2) → Concat(skip) → Conv → Conv
```

Loss: cross-entropy works, but Dice loss handles class imbalance better (e.g., small foreground, large background):

```
Dice = 1 − (2 · |P ∩ G|) / (|P| + |G|)
```

```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=64):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b*2)
        self.enc3 = DoubleConv(b*2, b*4)
        self.enc4 = DoubleConv(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b*8, b*16)
        self.up4 = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = DoubleConv(b*16, b*8)
        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = DoubleConv(b*8, b*4)
        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = DoubleConv(b*4, b*2)
        self.up1 = nn.ConvTranspose2d(b*2, b, 2, stride=2)
        self.dec1 = DoubleConv(b*2, b)
        self.head = nn.Conv2d(b, num_classes, 1)

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
        return self.head(d1)  # (B, num_classes, H, W)

def dice_loss(pred, target, smooth=1.0):
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
    inter = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()
```

**What breaks:** U-Net still has a fixed receptive field set by the depth of the encoder. Faraway context — useful for resolving class ambiguity (is this sky or ocean?) — is not explicitly modeled.

---

### 1.3 DeepLab v3+ (Chen et al., 2018)

**The problem:** to capture large-scale context, convolution networks must be deep (many pooling steps). But each pooling step throws away spatial resolution. You want both: a wide receptive field (context) and a fine-grained spatial map (precise boundaries). These are in direct tension.

**The core insight:** use *atrous (dilated) convolutions* — insert gaps between filter taps to expand the receptive field without striding. At dilation rate r, each filter tap covers a region r times larger than a standard convolution, but the output has the same spatial size. Run multiple dilation rates in parallel (ASPP) to capture context at multiple scales simultaneously.

**The mechanics:**

Atrous convolution with rate r:
```
y[i] = Σ_k x[i + r·k] · w[k]
```

ASPP: apply parallel atrous convolutions at rates {1, 6, 12, 18} plus a global average pooling branch, then concatenate all outputs.

Encoder–Decoder flow:
```
Input
  └─ Xception encoder (stride-16 output; stride-4 low-level features)
       └─ ASPP → 1×1 conv → 256-ch features
            └─ 4× bilinear upsample
                 └─ concat(low-level stride-4 features)
                      └─ 3×3 conv → 1×1 head → 4× upsample → output
```

Depthwise separable convolutions replace standard 3×3 convs throughout, reducing FLOP count by ~8–9×.

**What breaks:** ASPP rates are fixed hyperparameters — if objects appear at scales outside the chosen rate range, they are missed. The global pooling branch partially compensates, but ASPP is fundamentally a fixed-scale multi-resolution probe.

---

### 1.4 SegFormer (Xie et al., 2021)

**The problem:** transformers require positional embeddings to encode spatial location. Fixed positional embeddings mean the model can only handle the training resolution — if you want to run inference on a different image size, you must interpolate embeddings, which degrades accuracy.

**The core insight:** replace explicit positional embeddings with depth-wise 3×3 convolutions inside each feed-forward block. The local convolution implicitly encodes relative position through its receptive field, without fixing an absolute position to each token. The model becomes resolution-agnostic by construction.

**The mechanics:**

Hierarchical Mix Transformer (MiT) encoder with 4 stages at strides {4, 8, 16, 32}:
- **Efficient self-attention:** reshape K and V by reduction ratio R before computing attention, reducing cost from O(N²) to O(N²/R). R decreases across stages: {64, 16, 4, 1}.
- **Mix-FFN:** `x + Conv_{3×3}(FFN(x))` — the 3×3 conv carries implicit positional signal.

Lightweight all-MLP decoder:
```
MiT Stage1 (1/4)  ──┐
MiT Stage2 (1/8)  ──┤ upsample all to 1/4 → concat → MLP → head → 4× upsample
MiT Stage3 (1/16) ──┤
MiT Stage4 (1/32) ──┘
```

**What breaks:** the implicit positional encoding from 3×3 convolutions is weaker than explicit encodings for images where precise absolute location matters (e.g., top-of-frame vs. bottom-of-frame class statistics). SegFormer trades positional precision for resolution flexibility — acceptable for most segmentation benchmarks.

---

## 2. Instance Segmentation

**The task:** detect and delineate *each individual object instance* with a binary mask. Two objects of the same class get separate masks and IDs. Output: a set of (class, score, binary mask) tuples.

---

### 2.1 Mask R-CNN (He et al., 2017)

**The problem:** object detection networks localize boxes, but boxes are coarse — all pixels inside the box get the label regardless of object shape. How do you get pixel-accurate masks without redesigning the detection pipeline?

**The core insight:** add a third head to Faster R-CNN that predicts a binary mask in parallel with classification and box regression. The key challenge is alignment — RoIPool quantizes region coordinates to feature map cell boundaries, introducing misalignment that is tolerable for box prediction but fatal for pixel-accurate masks. Replace RoIPool with bilinear-interpolation-based RoIAlign.

**The mechanics:**
```
Image → Backbone (ResNet + FPN)
      → RPN → top-K region proposals
           → RoIAlign → 7×7 or 14×14 feature crops
                → [Box head: cls + bbox regression]
                → [Mask head: 4× Conv → deconv → sigmoid → 28×28 mask per class]
```

RoIAlign computes each output cell as a bilinear interpolation of the 4 nearest feature map cells:
```
feature(x, y) = Σ_{i,j} (1−|x−i|)(1−|y−j|) · F[i, j]
```
This eliminates the two-stage quantization of RoIPool, recovering +2–3 mask AP on COCO.

Multi-task loss:
```
L = L_cls + L_box + L_mask
L_mask = BCE(pred_mask[gt_class], gt_mask)  — only the ground-truth class channel
```

**What breaks:** inference time scales linearly with the number of proposals. Each proposal runs through the mask head independently. Dense scenes (crowds, retail shelves) are slow and memory-intensive. The two-stage design (region proposals then mask prediction) also limits the minimum achievable latency.

---

### 2.2 SOLOv2 (Wang et al., 2020)

**The problem:** Mask R-CNN's pipeline depends on region proposals — a serial bottleneck. Proposals have fixed-shape RoI crops, which are expensive, and the whole framework doesn't scale to very dense object scenes.

**The core insight:** avoid proposals entirely. Assign each instance to a grid cell based on where its center falls in the image. Each grid cell predicts a lightweight weight vector; instance masks are produced by taking the inner product of that weight vector with a shared global mask feature map. This turns mask generation into a conditional convolution — one matrix multiply per instance, no proposals needed.

**The mechanics:**

Two heads operating on FPN features:
1. **Category head:** S×S×C — class probability per grid cell.
2. **Mask kernel head:** S×S×D — D-dimensional weight vector per grid cell.

Mask generation:
```
mask_ij = sigmoid(kernel_ij · mask_features)   # (H/4, W/4)
```

mask_features is a single (H/4 × W/4 × D) tensor produced by the feature network — computed once, shared across all instances.

**What breaks:** instances whose centers fall in the same grid cell can't both be detected at the same scale — cells can only hold one instance. This limits performance on very small, densely packed objects of the same class at the same scale. NMS is replaced with matrix NMS (suppression via mask overlap score), which can occasionally over-suppress overlapping instances.

---

### 2.3 Mask2Former (Cheng et al., 2022)

**The problem:** standard cross-attention in a transformer decoder attends over the *entire* image feature map for each query. At high resolution, this is quadratic in sequence length and computationally prohibitive. Additionally, early in training, queries have no focus — they attend to irrelevant regions — making convergence slow.

**The core insight:** restrict each query's attention to its *predicted foreground region* at the previous decoder layer. Early layers can only attend where they currently predict a mask; later layers iteratively refine. This masked attention dramatically reduces the attention footprint and makes the attention meaningful from the first iteration.

**The mechanics:**
```
Attention(Q, K, V) with mask M:
  softmax((QKᵀ / √d) + M) V
  where M[h, i, j] = 0 if query i should attend position j, else −∞
```

Architecture:
```
Image → backbone + pixel decoder (FPN or MSDeformAttn) → multi-scale features
      → L-layer Transformer decoder (N learnable queries)
           → class predictions + binary masks (inner product with pixel features)
```

Trained with Hungarian matching + per-mask BCE + Dice loss. Same architecture serves semantic, instance, and panoptic segmentation — only training data and loss weighting differ.

**What breaks:** the iterative refinement requires L decoder layers to converge (typically 9 layers). This is slower per-image than a single-pass method like SOLOv2. The Hungarian matching during training is also expensive for large N.

---

## 3. Panoptic Segmentation

**The task:** assign every pixel a (class, instance_id) label. *Things* classes (countable objects: person, car) get unique instance IDs. *Stuff* classes (amorphous regions: sky, road) get instance_id = 0. Every pixel is labeled — no background.

---

### 3.1 Panoptic FPN (Kirillov et al., 2019)

**The problem:** things and stuff require fundamentally different handling — things need instance-level masks, stuff needs dense class coverage. Running two separate networks is expensive and misses shared representations.

**The core insight:** share an FPN backbone between a Mask R-CNN head (things) and a lightweight semantic segmentation head (stuff). At inference, merge the two outputs: instance masks take priority over the semantic prediction in their region.

**What breaks:** the merging heuristic is brittle — confidence thresholds for when instances should override stuff predictions require tuning per-dataset. The two heads don't share information about each other's outputs during training.

---

### 3.2 Mask2Former as a Unified Model

Mask2Former achieves state-of-the-art on all three segmentation tasks (semantic, instance, panoptic) with a single architecture by changing only the training data and loss weighting. Queries learn to predict both stuff regions (one mask per class) and things instances (one mask per instance). This eliminates the merging heuristic: every output is a (class, mask) pair; panoptic output is the union of all predictions.

---

### 3.3 Panoptic Quality (PQ) Metric

PQ decomposes recognition quality (RQ) from segmentation quality (SQ):

```
PQ = SQ × RQ

SQ = Σ_{(p,g)∈TP} IoU(p,g) / |TP|        (average IoU of matched pairs)
RQ = |TP| / (|TP| + ½|FP| + ½|FN|)        (F1 score over segments)
```

A predicted segment is a TP if its IoU with a ground-truth segment exceeds 0.5. Computed separately for things and stuff, then averaged across all categories. Mask2Former achieves ~57 PQ on COCO; Panoptic FPN ~40 PQ.

---

## 4. Pose Estimation

**The task:** detect anatomical keypoints (joints) for each person in an image. COCO defines 17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. Output: (x, y, visibility) per keypoint per person.

Two paradigms:
- **Top-down:** detect bounding boxes first → crop each person → run pose network. Accurate; inference time scales linearly with person count.
- **Bottom-up:** detect all keypoints globally → group into individuals. Fixed inference time regardless of crowd size.

---

### 4.1 Heatmap-based Keypoint Representation

**The problem:** predicting raw (x, y) coordinates directly is sensitive to small localization errors — a 1-pixel shift in a dense scene is a large regression target. How do you make the prediction task easier?

**The core insight:** predict a 2D Gaussian heatmap centered at each keypoint location instead of coordinates directly. The network predicts a dense probability field — a "soft" target that is forgiving of small spatial errors and easy to supervise with MSE. The predicted location is the argmax of the heatmap.

```
H_k(x, y) = exp(−((x−x_k)² + (y−y_k)²) / (2σ²))
```

Loss: MSE between predicted and ground-truth heatmaps, ignoring invisible keypoints.

---

### 4.2 OpenPose (Cao et al., 2017) — Bottom-up

**The problem:** in a bottom-up system you detect all keypoints globally, but you don't know which arm belongs to which person. Grouping keypoints into skeletons naively requires checking all pairwise combinations across all detected points — quadratic complexity that fails in dense crowds.

**The core insight:** predict *Part Affinity Fields (PAFs)* — 2D vector fields that encode the orientation and location of limbs. A limb between two keypoints produces a vector field pointing from one joint to the other along the limb's path. Grouping then becomes: integrate the PAF along a candidate connection; high integral = strong limb = same person.

**The mechanics:**

Two-branch network on VGG features:
```
Image → VGG features
      → Branch 1: PAF estimation (2L vector maps, L = number of limbs)
      → Branch 2: Confidence maps (K heatmaps, K = number of keypoints)
      → Iterative refinement (multiple stages, both branches as input)
      → Bipartite matching via PAF line integrals → skeletons
```

PAF line integral for candidate limb (d_a → d_b):
```
E = ∫_0^1 L(p(u)) · v(p(u)) du
   p(u) = (1−u)·d_a + u·d_b,   v = (d_b − d_a)/|d_b − d_a|
```

High E means the limb vector field strongly supports this connection.

**What breaks:** PAF integration is an approximation; for highly overlapping people with similar poses, the line integral can produce incorrect assignments. Inference time grows with the number of candidates.

---

### 4.3 HRNet (Sun et al., 2019) — Top-down

**The problem:** most CNN backbones progressively downsample the feature map, losing spatial resolution in exchange for semantic richness. For keypoint localization — where being off by 2–3 pixels matters — this resolution loss hurts even after upsampling in the decoder.

**The core insight:** maintain a high-resolution representation throughout the entire network by running parallel branches at multiple resolutions that constantly exchange information. The high-resolution branch never compresses spatial detail; lower-resolution branches provide semantic context. Both are continuously fused.

**The mechanics:**
```
Stage 1: single high-res branch (1/4 resolution)
Stage 2: add parallel 1/8 branch; exchange features
Stage 3: add parallel 1/16 branch; exchange features
Stage 4: add parallel 1/32 branch; exchange features
→ aggregate all branches (upsample lower-res to 1/4) → heatmap head
```

Multi-scale fusion: each branch receives summed contributions from all other branches (strided conv to downsample, bilinear upsample to go up). The high-res branch never sees a resolution drop.

**What breaks:** the parallel-branch design is memory-intensive. HRNet-W48 uses ~65 GB of GPU memory during training at high batch sizes. The architecture is also not easily accelerated by standard efficiency tools because the multi-scale branch structure does not map neatly to block-sparse operations.

---

### 4.4 ViTPose (Xu et al., 2022) — Transformer Top-down

**The problem:** HRNet's high-resolution advantage relies on architectural inductive biases. As ViT backbones (pre-trained with MAE) became available with increasingly rich spatial representations, the question became: does a plain ViT with a minimal decoder beat an architecture-heavy model like HRNet?

**The core insight:** a large ViT pre-trained with MAE learns spatially rich representations that transfer effectively to localization. The high capacity of the transformer encoder compensates for the lack of architectural localization biases. You don't need a complex decoder — bilinear upsampling from ViT features is sufficient.

**The mechanics:**
```
Person crop → ViT backbone (patch embed → L transformer layers) → feature map (H/16, W/16, D)
           → lightweight decoder (2× deconv or bilinear upsample)
           → K heatmaps
```

ViT-H achieves 80.9 AP on COCO val. Larger backbone → consistently better AP; decoder complexity has minimal impact.

**What breaks:** a plain ViT at patch size 16 on a 256×192 crop yields 16×12 = 192 tokens — fine for global attention. But ViT is not hierarchical: the feature map is at fixed 1/16 resolution throughout, which limits the model's ability to capture fine spatial details. Deconv decoders partially compensate but can't fully recover sub-patch spatial precision.

---

## 5. Key Metrics

### 5.1 mIoU — Semantic Segmentation

Intersection over Union for class c:
```
IoU_c = TP_c / (TP_c + FP_c + FN_c) = |pred_c ∩ gt_c| / |pred_c ∪ gt_c|
```

Mean IoU averages over all C classes:
```
mIoU = (1/C) Σ_c IoU_c
```

True negatives (correctly predicted background) don't inflate the score — mIoU is insensitive to easy negatives.

### 5.2 AP — Instance / Object Detection

Average Precision integrates the precision-recall curve:
```
AP = Σ_k (R_k − R_{k-1}) · P_k
```

- A prediction is TP if its mask/box IoU with the matched ground-truth exceeds a threshold.
- **COCO AP:** average of AP at IoU thresholds {0.50, 0.55, ..., 0.95} — 10 thresholds.
- **mAP:** mean AP over all categories.

### 5.3 Panoptic Quality (PQ)

Defined in Section 3.3. Reported as PQ (all), PQ_th (things), PQ_st (stuff).

### 5.4 OKS — Object Keypoint Similarity

Analogous to IoU for keypoints:
```
OKS_k = exp(−d_k² / (2·s²·σ_k²)) · δ(v_k > 0)
```

- d_k: Euclidean distance between predicted and ground-truth keypoint.
- s: object scale (√bounding box area).
- σ_k: per-keypoint constant reflecting annotation noise (hip σ=0.107, eye σ=0.025).
- v_k: visibility flag.

Overall OKS for a person = mean over keypoints. COCO reports AP at OKS ∈ {0.50, 0.55, ..., 0.95}.

### 5.5 PCKh — Percentage of Correct Keypoints (head-normalized)

Used on MPII:
```
PCKh@α: keypoint is correct if  d_k < α · h
```

h = head segment length (head-top to upper-neck distance); α=0.5 is standard. Normalizing by a person-specific measurement makes PCKh robust to varying person scale.

---

## 6. Key Interview Points

**Semantic vs instance vs panoptic segmentation**
Semantic: per-pixel class, no instance distinction. Instance: per-object mask + class, no dense coverage (background unlabeled). Panoptic: every pixel labeled with (class, instance_id); stuff gets instance_id = 0.

**Why RoIAlign over RoIPool?**
RoIPool quantizes twice — proposal coordinates to feature map cells, then cells to a fixed grid. This misalignment can be half a cell, tolerable for detection but catastrophic for 28×28 masks. RoIAlign uses bilinear interpolation at continuous sample points, eliminating quantization and recovering +2–3 mask AP on COCO.

**How does ASPP capture multi-scale context?**
ASPP applies atrous convolutions at multiple dilation rates in parallel. Each rate gives a different effective receptive field. Concatenating the outputs lets a single pixel's prediction draw on context at scales from local texture (rate 1) to large regions (rate 18). The global average pooling branch adds image-level context.

**What problem do skip connections solve in U-Net vs FCN?**
FCN adds skip connections via element-wise addition — adds spatial information back after aggressive downsampling. U-Net uses concatenation — more expressive (doubles channel count, so the next conv can learn how to merge the two feature sets). U-Net also has a symmetric decoder as deep as the encoder, with skip connections at every resolution level.

**Bottom-up vs top-down pose estimation trade-offs**
Top-down (HRNet, ViTPose): crop each detected person → dedicated pose network. Accurate; scales linearly with person count; depends on detector quality. Bottom-up (OpenPose): one forward pass → fixed inference cost regardless of crowd size. Harder to achieve high precision for closely overlapping people; PAF grouping adds post-processing cost.

**Why does ViTPose outperform HRNet despite a simpler decoder?**
The ViT backbone pre-trained with MAE learns rich spatially-aware representations. The high-capacity transformer encoder compensates for the minimal decoder. HRNet's architectural advantage — maintaining high resolution throughout — becomes less critical when the encoder already captures the needed spatial precision.

**SOLOv2 kernel trick — how does it avoid region proposals?**
SOLOv2 generates a global mask feature map once. Each grid cell predicts a D-dimensional weight vector. The instance mask is the inner product of that vector with the global mask feature map — a conditional convolution. This is ~5× faster than Mask R-CNN on equivalent hardware.

**mIoU vs AP — when to use which?**
mIoU: natural for semantic segmentation where every pixel is labeled; averages over classes equally. AP: natural for detection/instance tasks where predictions are confidence-ranked and the number per image varies. Panoptic uses PQ (= SQ × RQ) because it must penalize both missed instances (FN) and spurious ones (FP), combining precision and recall into one metric.

**Dice loss vs cross-entropy for segmentation**
Cross-entropy averages equally over pixels — small foreground regions contribute few pixels, so loss is dominated by background. Dice loss directly optimizes the IoU-like overlap ratio, naturally handling class imbalance. In practice: combine cross-entropy + Dice for best convergence.

**SegFormer's trick for resolution-agnostic inference**
Standard ViT uses fixed positional embeddings — changing input resolution at test time misaligns them. SegFormer removes positional embeddings and adds depth-wise 3×3 convolutions inside the FFN (Mix-FFN). The convolutions provide implicit positional signal without fixing a resolution, enabling arbitrary-size inference without interpolation artifacts.

## Flashcards

**Efficient self-attention?** #flashcard
reshape K and V by reduction ratio R before computing attention, reducing cost from O(N²) to O(N²/R). R decreases across stages: {64, 16, 4, 1}.

**Mix-FFN: x + Conv_{3×3}(FFN(x))?** #flashcard
the 3×3 conv carries implicit positional signal.

**Top-down?** #flashcard
detect bounding boxes first → crop each person → run pose network. Accurate; inference time scales linearly with person count.

**Bottom-up?** #flashcard
detect all keypoints globally → group into individuals. Fixed inference time regardless of crowd size.

**A prediction is TP if its mask/box IoU with the matched ground-truth exceeds a threshold.?** #flashcard
A prediction is TP if its mask/box IoU with the matched ground-truth exceeds a threshold.

**COCO AP: average of AP at IoU thresholds {0.50, 0.55, ..., 0.95}?** #flashcard
10 thresholds.

**mAP?** #flashcard
mean AP over all categories.

**d_k?** #flashcard
Euclidean distance between predicted and ground-truth keypoint.

**s?** #flashcard
object scale (√bounding box area).

**σ_k?** #flashcard
per-keypoint constant reflecting annotation noise (hip σ=0.107, eye σ=0.025).

**v_k?** #flashcard
visibility flag.
