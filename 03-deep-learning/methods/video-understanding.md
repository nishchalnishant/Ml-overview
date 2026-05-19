# Video Understanding

> Covers two-stream networks, 3D CNNs, SlowFast, video transformers, and key tasks (action recognition, temporal localization, segmentation, VQA).

---

## Table of Contents

1. [Core Challenges](#1-core-challenges)
2. [Input Representations](#2-input-representations)
3. [Two-Stream Networks](#3-two-stream-networks)
4. [3D CNNs (C3D, I3D)](#4-3d-cnns-c3d-i3d)
5. [SlowFast Networks](#5-slowfast-networks)
6. [Video Transformers](#6-video-transformers)
7. [Key Tasks](#7-key-tasks)
8. [Optical Flow in Detail](#8-optical-flow-in-detail)
9. [Efficiency Considerations](#9-efficiency-considerations)
10. [Datasets](#10-datasets)
11. [Key Interview Points](#11-key-interview-points)

---

## 1. Core Challenges

**The problem:** image recognition networks classify a single frame. But video is not a collection of independent frames — temporal structure is the signal. An image model watching a video is blind to whether a door is opening or closing, whether a person is walking toward or away, whether a fall is beginning or ending.

Four compounding difficulties:

- **Temporal modeling:** which frames matter, and how do they relate to each other?
- **Computational cost:** videos run at 25–60 fps over minutes. Processing every frame at full resolution with a deep network is prohibitive.
- **Long-range dependencies:** some actions (cooking a meal, playing a sport) require context from seconds or minutes apart — well beyond the receptive field of any practical 3D CNN.
- **Motion vs. appearance confusion:** some actions look identical as still images but differ only in motion direction or speed. A model that ignores temporal order cannot distinguish them.

These challenges drive every architectural choice in the field.

---

## 2. Input Representations

### Raw RGB Frames

Stack T consecutive frames: (T, H, W, 3). Treat as a spatio-temporal volume. Captures appearance; motion must be learned implicitly.

### Optical Flow

Encodes per-pixel motion between adjacent frames — a 2D velocity field (u, v) per pixel. Separates motion from appearance, making it explicit.

```python
import cv2

def compute_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow  = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
    return flow  # (H, W, 2): horizontal and vertical motion
```

Flow is expensive to compute on the fly; precompute and cache for training. Deep flow estimators (RAFT, PWC-Net) produce higher quality than classical Farneback but add inference cost.

---

## 3. Two-Stream Networks

**The problem:** a single CNN applied to an RGB frame captures appearance. But it has no explicit signal for *how things are moving*. Separately, if you only look at optical flow, you lose the appearance context needed to understand what is moving. Both signals are necessary; neither alone is sufficient.

**The core insight:** run two separate CNNs in parallel — one on RGB (appearance stream), one on stacked optical flow frames (motion stream) — and fuse their predictions. Appearance answers "what objects are present"; motion answers "how are they moving." Late fusion combines both signals.

**The mechanics:**

```
RGB frame ──────────────→ CNN → Spatial softmax ──┐
                                                    → Fusion → action class
Optical flow stack ──────→ CNN → Temporal softmax ─┘
```

The temporal stream sees a stack of T optical flow frames (2T channels — u and v components for each frame). Fusion is typically weighted averaging or a small MLP on concatenated scores.

**What breaks:**

- **Optical flow is expensive:** dense flow computation adds significant preprocessing time and storage. It's feasible offline but not for real-time pipelines.
- **Late fusion misses cross-stream interactions:** the two streams are trained independently and only communicate at the prediction level. An arm moving in a way that only makes sense in context of the adjacent body part cannot be captured.
- **Flow estimation errors propagate:** inaccurate flow (from motion blur, rapid scene changes, occlusion) produces noisy motion features that hurt the temporal stream.

---

## 4. 3D CNNs (C3D, I3D)

**The problem:** two-stream networks decouple appearance and motion into separate networks and fuse late. This is crude — the interaction between spatial features and temporal dynamics (e.g., detecting the contact moment in a ball kick) requires joint reasoning over space and time in the same representation.

**The core insight:** extend 2D convolutions to 3D by adding a temporal dimension. A (K_t × K_h × K_w) filter learns to detect spatio-temporal patterns jointly — edges moving in a particular direction, the expanding silhouette of a jumping person. You get a single unified spatio-temporal representation.

**The mechanics:**

C3D (Tran et al., 2015): All 3×3×3 convolutions on 16-frame clips. Trained from scratch on Sports-1M.

I3D (Carreira & Zisserman, 2017): **Inflated** Inception 2D filters → 3D filters by repeating weights along the temporal axis and dividing by K_t to preserve activation magnitudes:

```
2D filter (K_h × K_w) → 3D filter (K_t × K_h × K_w)
   weights: w_{h,w} → w_{h,w} / K_t   (repeated K_t times)
```

This allows weight initialization from ImageNet-pretrained 2D networks, sidestepping the need for large-scale 3D pre-training data.

```python
conv3d = nn.Conv3d(in_channels=3, out_channels=64,
                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
# Input: (B, C, T, H, W)
```

I3D + Two-Stream (3D RGB + 3D flow) is still a strong baseline.

**What breaks:**

- **Memory and compute:** 3D convolutions are ~K_t × more expensive than 2D. A 3×3×3 conv is 3× more operations per layer than a 3×3 conv, and memory scales with temporal depth T.
- **Pre-training data scarcity:** 3D representations require video pre-training to be useful. Unlike 2D models with ImageNet, 3D models can't leverage the vast image pre-training ecosystem as directly.
- **Short temporal range:** C3D and I3D process 16-frame clips. Long-range dependencies (minutes apart) require separate architectural mechanisms.

---

## 5. SlowFast Networks

**The problem:** not all temporal signals are equal. Coarse semantic content (what objects are in the scene, what class of action is happening) changes slowly. Fine motion details (subtle hand movements, fast ball trajectories) require high temporal resolution. A single stream sampled at one frame rate must trade off between the two.

**The core insight:** the primate visual cortex uses two cell types: P-cells (slow, fine-grained spatial detail) and M-cells (fast, coarse motion sensitivity). Run two pathways at *different temporal resolutions* — Slow pathway at low frame rate with many channels for semantic richness, Fast pathway at high frame rate with few channels for motion detail. Connect them with lateral connections.

**The mechanics:**

| Pathway | Frame rate | Channels | Captures |
|---------|-----------|----------|---------|
| Slow | Low (e.g., 4 fps) | Many (C) | Spatial semantics |
| Fast | High (e.g., 32 fps) | Few (~C/8) | Fine temporal motion |

Lateral connections from Fast → Slow at multiple spatial stages fuse the motion signal into the semantic pathway.

**What breaks:**

- **Two-pathway cost:** the Fast pathway runs at 8× the frame rate but with 1/8 the channels. The total FLOP cost is comparable to a single-pathway model, but the two-stage design adds engineering complexity.
- **Fixed ratio:** the 4:1 or 8:1 frame-rate ratio is a hyperparameter that may not be optimal across all action categories.

---

## 6. Video Transformers

**The problem:** 3D CNNs and SlowFast have limited effective temporal receptive fields — they process short clips (16–64 frames). Actions that span many seconds, or that require understanding narrative causality, cannot be captured in a single clip-level feature vector.

**The core insight:** transformers have global attention — every token can directly attend to every other token regardless of distance. Apply the transformer to video by tokenizing space-time patches. Each token corresponds to a spatio-temporal cube; attention over all tokens gives the model access to the full temporal extent of the clip in a single layer.

---

### TimeSformer (Bertasius et al., 2021)

**The problem:** full space-time attention over a video clip is O(T²H²W²) — prohibitive even for short clips.

**The core insight:** factorize attention: first attend across time at each spatial location, then attend across space at each time step. This reduces cost from O(T²H²W²) to O(T²·HW + T·H²W²) — manageable for typical clip sizes.

**The mechanics:**

```
Tokens: (B, T×H×W, D)

Temporal attention: each spatial location i attends to all T time steps at location i
Spatial attention:  each time step t attends to all HW spatial locations at time t
```

**What breaks:** factorized attention can miss interactions between temporal and spatial patterns that co-occur at different locations and times. Full joint attention would capture these but is too expensive.

---

### Video MAE (He et al., 2022)

**The problem:** labeled video data is scarce. Collecting and annotating hours of action video for every new domain is expensive. Can you learn a powerful video representation without labels?

**The core insight:** apply masked autoencoding to video — mask ~90% of spatio-temporal patches and train the model to reconstruct them from the remaining visible 10%. The high masking ratio is possible because video is *temporally redundant*: adjacent frames are nearly identical, so the model cannot simply copy visible patches to fill in masked ones. It must learn genuine temporal structure to reconstruct coherently.

**The mechanics:**

Mask 90% of spatio-temporal patch tokens → ViT encoder on visible 10% → lightweight decoder reconstructs masked patches in pixel space → photometric loss on masked patches only.

Pre-train on unlabeled video → fine-tune for downstream tasks (action recognition, temporal localization).

**What breaks:** the pre-training task is reconstruction, which may over-emphasize low-level texture and under-emphasize high-level temporal semantics. Fine-tuning is required; zero-shot generalization to new tasks is limited compared to language-supervised models.

---

### MViT (Multiscale Vision Transformers)

**The problem:** vision transformers run all layers at the same resolution, with the same number of tokens throughout. This is wasteful — early layers should capture fine-grained local features (many tokens, small receptive fields); later layers should capture global semantics (fewer tokens, large receptive fields).

**The core insight:** hierarchically pool keys and values as the network deepens, progressively reducing the number of tokens while increasing channel width. This mimics the multi-scale structure of CNNs but within a transformer.

**What breaks:** pooling attention introduces information loss. Unlike convolutions which have learned weights, the pooling is typically average or max — a fixed, unlearned downsampling.

---

## 7. Key Tasks

### Action Recognition

Classify the dominant action in a short video clip.

**Benchmarks:** Kinetics-400/600/700 (clips labeled with 400–700 action classes), Something-Something (174 classes, specifically designed to require temporal understanding — "moving object to the left" looks identical as a still to "moving object to the right").

```python
from torchvision.models.video import r3d_18

model = r3d_18(pretrained=True)
model.eval()
video_clip = torch.randn(1, 3, 16, 112, 112)  # (B, C, T, H, W)
logits = model(video_clip)
```

Key finding: on Kinetics, single-frame accuracy is ~85% of full-clip accuracy — appearance alone is nearly sufficient. On Something-Something, single-frame accuracy drops dramatically — temporal understanding is required.

---

### Temporal Action Localization

Detect *when* actions start and end in a long untrimmed video. Output: `[(t_start, t_end, class, confidence), ...]`.

**The problem:** unlike image detection where objects have clear spatial extent, temporal boundaries are ambiguous — actions start and end gradually. Proposals must cover a wide range of temporal scales.

Approaches:
- **Proposal + classification** (BSN, BMN): generate temporal proposals of varying durations, then classify each.
- **One-stage** (AFSD, ActionFormer): directly predict start/end/class from dense temporal features.
- **DETR-style** (RTD-Net): end-to-end with action queries that decode to (start, end, class) tuples.

ActionFormer (transformer on temporal proposal features) is the current standard.

---

### Temporal Action Segmentation

Dense frame-by-frame labeling — assign an action class to every frame. Used in instructional video understanding (cooking step detection, procedure recognition).

The challenge: adjacent frames share the same label for long stretches (over-segmentation is heavily penalized), but transitions between steps happen rapidly.

Methods: MS-TCN (Multi-Scale Temporal Convolutional Network — dilated TCN at multiple scales), ASRF, ASFormer.

---

### Video Question Answering

Answer natural language questions about video content: "What does the person do after opening the fridge?" Requires joint reasoning over visual events and language.

Models: Video-LLaMA, VideoChat, InternVideo2.

**What breaks:** long videos require either summarization (discarding information) or efficient attention (still quadratic in tokens). Current models struggle with videos longer than ~1 minute.

---

## 8. Optical Flow in Detail

### Classical Methods

| Method | Type | Notes |
|--------|------|-------|
| Lucas-Kanade | Sparse (feature points) | Fast, assumes small motion |
| Horn-Schunck | Dense, global | Global smoothness constraint |
| Farneback | Dense, polynomial | Good speed/quality balance |

### Deep Learning Flow Estimation

| Model | Approach | Notes |
|-------|---------|-------|
| FlowNet | Supervised CNN | First DL optical flow |
| SPyNet | Spatial pyramid | Lightweight |
| PWC-Net | Cost volume + warping | Good accuracy/speed |
| RAFT | Iterative refinement | Near-SOTA quality |

RAFT builds an all-pairs correlation volume between two frames and iteratively refines the flow field. Flow quality directly impacts temporal stream performance in two-stream models.

**What breaks:** all optical flow methods assume *brightness constancy* (a pixel's intensity doesn't change as it moves). This fails under illumination changes, specular reflections, and fast motion blur. RAFT partially mitigates this via iterative refinement but cannot recover from fundamental brightness constancy violations.

---

## 9. Efficiency Considerations

**The problem:** full video processing (every frame, full resolution, deep network) is computationally prohibitive at scale. How do you preserve accuracy while drastically reducing computation?

### Clip Sampling

Don't process entire videos. Sample T-frame clips.

**Dense sampling:** T consecutive frames from a short temporal window — captures fine motion.

**Sparse sampling (TSN):** sample S segments; one frame per segment. Captures global structure at low cost.

```python
def sparse_sample(video_frames, n_segments=8):
    segment_size = len(video_frames) // n_segments
    sampled = []
    for i in range(n_segments):
        idx = i * segment_size + np.random.randint(segment_size)
        sampled.append(video_frames[min(idx, len(video_frames)-1)])
    return sampled
```

**What breaks:** sparse sampling misses fast actions (e.g., ball contact, brief gestures) that happen within a single segment and require consecutive frames. Dense sampling requires more compute. The right sampling strategy is action-duration-dependent.

### Other Efficiency Techniques

- **Temporal stride > 1:** read every other frame — halves temporal resolution and compute.
- **Knowledge distillation:** distill a 3D CNN teacher into an efficient 2D CNN student for edge deployment.
- **Efficient attention:** MViT pooling attention, TimeSformer factorized attention — reduce O(T²H²W²) attention to manageable cost.

---

## 10. Datasets

| Dataset | Task | Size | Notes |
|---------|------|------|-------|
| Kinetics-400/700 | Action recognition | 400–700 classes, 650K clips | Primary benchmark |
| Something-Something | Temporal reasoning | 174 classes, 220K clips | Requires temporal understanding |
| UCF-101 | Action recognition | 101 classes, 13K clips | Older, smaller, appearance-dominated |
| AVA | Spatio-temporal action | 80 atomic actions, 430 clips | Detects WHO is doing WHAT, where |
| ActivityNet | Temporal localization | 200 classes, 20K videos | Long untrimmed videos |
| EPIC-Kitchens | Egocentric actions | 97 actions, 11M frames | First-person cooking |

---

## 11. Key Interview Points

**Two-stream networks separate appearance and motion — why does this help?**
A single CNN on RGB must learn to extract both appearance and motion signals from raw pixels. Pre-computed optical flow hands the motion signal directly to the temporal stream, making that task easier. The two streams capture complementary signals; late fusion combines both.

**How does I3D leverage ImageNet pre-training despite being 3D?**
I3D "inflates" 2D Inception filters to 3D by repeating the 2D weights K_t times along the temporal axis and dividing by K_t to preserve activation magnitude. This allows the model to start from ImageNet-pretrained weights rather than random initialization, making training on video data more data-efficient.

**Why does SlowFast have a Fast pathway with far fewer channels than the Slow pathway?**
The Fast pathway samples high frame rates to capture fine motion detail, but this detail is mostly low-level (flow, edge direction). Low-level motion features require few channels; semantic features require many. The channel asymmetry (C vs. C/8) means the Fast pathway adds modest compute while providing a high-frame-rate motion signal.

**Video MAE masks ~90% of patches — why is such a high masking ratio used?**
Video is temporally redundant — adjacent frames are nearly identical. At low masking rates (~75% as in image MAE), the model can trivially reconstruct masked patches by copying from nearby visible patches in adjacent frames. The high masking rate forces the model to reason about temporal structure and appearance rather than doing local interpolation.

**Something-Something requires temporal understanding; Kinetics mostly tests appearance — why?**
Kinetics clips are labeled with fine-grained action classes (e.g., "kite surfing") that are visually distinctive in a single frame. A frame-level model already knows what's happening. Something-Something explicitly tests *relational* actions ("moving object to the left/right," "pretending to do X") that require observing motion direction and causal structure across frames.

**For temporal localization, why is ActionFormer the current standard?**
ActionFormer applies a transformer to temporal features of an untrimmed video, where each token represents a short temporal window. Global attention allows long-range context for action boundary detection. It predicts start/end directly (one-stage), avoiding the two-stage proposal-then-classify pipeline and its associated latency.
