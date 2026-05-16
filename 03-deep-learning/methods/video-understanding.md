# Video Understanding

Video understanding extends image recognition to the temporal dimension. The model must understand not just what objects are present, but how they move, interact, and change over time.

---

## Core Challenges

- **Temporal modeling:** Which frames matter, and how do they relate?
- **Computational cost:** Videos are ~25-60 fps × minutes; full processing is prohibitive
- **Long-range dependencies:** Action recognition may need context from seconds or minutes apart
- **Motion vs appearance:** Some actions look identical as stills but differ in motion (opening vs closing a door)

---

## Input Representations

### Raw RGB Frames
Stack T consecutive frames: `(T, H, W, 3)` → treat as spatial-temporal volume.

### Optical Flow
Encodes per-pixel motion between adjacent frames. Classic two-stream approach uses both RGB (appearance) and optical flow (motion) streams.

```python
import cv2

def compute_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    return flow  # (H, W, 2): horizontal and vertical motion

# Dense optical flow is expensive — compute during preprocessing, cache
```

**TVL-1** and **PWC-Net** (deep learning-based) produce higher quality flow.

---

## Architectures

### 1. Two-Stream Networks (Simonyan & Zisserman, 2014)

- **Spatial stream:** Single RGB frame → CNN → appearance features
- **Temporal stream:** Stack of optical flow frames → CNN → motion features
- **Fusion:** Average or concatenate final predictions

```
RGB frame ──────────────→ CNN → Spatial Softmax ──┐
                                                    → Fusion → Action class
Optical Flow stack ──────→ CNN → Temporal Softmax ─┘
```

**Strength:** Separates appearance and motion. Simple, interpretable.  
**Weakness:** Optical flow is expensive to compute; late fusion loses cross-stream interactions.

---

### 2. 3D CNNs (C3D, I3D)

Replace 2D spatial convolutions with 3D spatio-temporal convolutions `(K_t × K_h × K_w)`.

**C3D (Tran et al., 2015):** All 3×3×3 convolutions on 16-frame clips.

**I3D (Carreira & Zisserman, 2017):** "Inflated" 2D convolutions from Inception. Inflate 2D filters `(K_h × K_w)` → 3D filters `(K_t × K_h × K_w)`. Initialize from ImageNet-pretrained 2D weights by repeating and rescaling.

```python
# 3D convolution example
conv3d = nn.Conv3d(in_channels=3, out_channels=64, 
                   kernel_size=(3, 3, 3), 
                   padding=(1, 1, 1))
# Input: (B, C, T, H, W)
```

**I3D + Two-Stream:** Best known combination; still a strong baseline.

---

### 3. SlowFast Networks (Feichtenhofer et al., 2019)

Two pathways at different temporal resolutions:

| Pathway | Frame rate | Channels | Captures |
|---------|-----------|----------|---------|
| **Slow** | Low (e.g., 4 fps) | Many | Spatial semantics |
| **Fast** | High (e.g., 32 fps) | Few (~1/8 slow) | Fine temporal motion |

Lateral connections from Fast → Slow at multiple stages.

**Intuition:** Human visual system has P-cells (spatial detail) and M-cells (motion/temporal).

---

### 4. Video Transformers

Apply the Transformer architecture to video by tokenizing spatio-temporal patches.

#### TimeSformer (Bertasius et al., 2021)

Decomposed space-time attention:
1. **Temporal attention:** Each spatial location attends across time
2. **Spatial attention:** Each time step attends across space

Reduces O(T²H²W²) full attention to O(T² + H²W²).

```python
# Patch tokenization for video
# Video: (B, C, T, H, W)
# Divide into patches of size (t, p, p)
# Each patch → embedding token
# Full sequence: (B, T//t * H//p * W//p, embed_dim)
```

#### Video MAE (He et al., 2022)

Masked autoencoder for video: mask ~90% of spatio-temporal patches and reconstruct them. High masking ratio works because video is temporally redundant.

Pre-training on unlabeled video, then fine-tune for downstream tasks (action recognition, temporal localization).

#### VideoMAE V2 / InternVideo2

Scale to 1B parameters; best performance on Kinetics-400/600, Something-Something.

---

### 5. MViT (Multiscale Vision Transformers)

Hierarchical transformer with pooling attention — pools keys/values progressively, reducing resolution while increasing channel depth. Efficient alternative to full space-time attention.

---

## Key Tasks

### Action Recognition

Classify the action in a video clip.

**Benchmarks:** Kinetics-400/600/700, UCF-101, HMDB-51, ActivityNet, Something-Something (tests temporal reasoning, not just appearance)

```python
from torchvision.models.video import r3d_18
import torch

model = r3d_18(pretrained=True)
model.eval()

# Input: (batch, channels, time, height, width)
video_clip = torch.randn(1, 3, 16, 112, 112)
logits = model(video_clip)
```

### Temporal Action Localization

Detect when actions start and end in a long untrimmed video. Output: `[(t_start, t_end, class, confidence), ...]`

Approaches:
- **Proposal + classification** (BSN, BMN): generate proposals, then classify
- **One-stage** (AFSD, ActionFormer): directly predict start/end/class
- **DETR-style** (RTD-Net): end-to-end with action queries

### Temporal Action Segmentation

Dense frame-by-frame labeling (e.g., cooking step detection).

Methods: MS-TCN (Multi-Scale Temporal Convolutional Network), ASRF, ASFormer.

### Video Question Answering

Answer natural language questions about video content.

Models: MERLOT, All-in-One, Video-LLaMA, VideoChat.

### Video Captioning & Generation

- Captioning: describe video in text (dense video captioning for long videos)
- Generation: text-to-video (Sora, VideoLDM, CogVideoX)

---

## Optical Flow in Detail

### Classical Methods

- **Lucas-Kanade:** Sparse (feature-point-based), fast, assumes small motion
- **Horn-Schunck:** Dense, global smoothness constraint
- **Farneback:** Dense polynomial expansion, good balance of speed/quality

### Deep Learning Flow Estimation

| Model | Approach | Notes |
|-------|---------|-------|
| FlowNet | Supervised | First DL optical flow |
| SPyNet | Spatial pyramid | Lightweight |
| PWC-Net | Cost volume + warping | Good accuracy/speed |
| RAFT | Iterative refinement | Near-SOTA quality |

```python
# RAFT-based flow (simplified)
# flow = raft_model(img1, img2)
# flow shape: (B, 2, H, W)  — u and v components
```

---

## Efficiency Considerations

**Clip sampling:** Don't process entire videos. Sample T-frame clips (dense or sparse sampling).

**Sparse sampling (TSN):** Sample 3–8 segments; one frame per segment. Captures global video structure cheaply.

```python
def sparse_sample(video_frames, n_segments=8):
    segment_size = len(video_frames) // n_segments
    sampled = []
    for i in range(n_segments):
        idx = i * segment_size + np.random.randint(segment_size)
        sampled.append(video_frames[min(idx, len(video_frames)-1)])
    return sampled
```

**Temporal stride:** Use stride > 1 when reading frames (e.g., every other frame).

**Knowledge distillation:** Distill 3D CNN teacher into efficient 2D CNN student for edge deployment.

---

## Datasets

| Dataset | Task | Size | Notes |
|---------|------|------|-------|
| Kinetics-400/700 | Action recognition | 400–700 classes, 650K clips | Primary benchmark |
| Something-Something | Temporal reasoning | 174 classes, 220K clips | Requires temporal understanding |
| UCF-101 | Action recognition | 101 classes, 13K clips | Older, smaller |
| AVA | Spatio-temporal action | 80 atomic actions, 430 clips | Detects WHO is doing WHAT |
| ActivityNet | Temporal localization | 200 classes, 20K videos | Long videos |
| EPIC-Kitchens | Egocentric | 97 actions, 11M frames | First-person cooking |

---

## Key Interview Points

- Two-stream networks separate appearance (RGB) and motion (optical flow) — strong baseline for action recognition.
- I3D "inflates" ImageNet-pretrained 2D filters into 3D; leverages pre-training without 3D pre-training data.
- SlowFast captures fine motion with Fast pathway and semantics with Slow pathway at modest cost.
- Video MAE masks ~90% of patches — high because adjacent frames are temporally redundant.
- Something-Something requires temporal understanding (optical flow helps); Kinetics mostly tests appearance (single frame often enough).
- For temporal localization, ActionFormer (transformer on proposals) is the current standard.
