---
module: Deep Learning
topic: Methods
subtopic: Open-Vocabulary Detection, Segmentation, Tracking, and OCR
status: unread
tags: [deeplearning, ml, methods, sam, grounding-dino, tracking, ocr]
---
# Open-Vocabulary Detection, Promptable Segmentation, Tracking, and OCR

> Builds on [computer-vision.md](computer-vision.md) §4 (Object Detection) and [segmentation.md](segmentation.md) §1-2. Those cover *closed-vocabulary* detectors/segmenters trained on a fixed label set (COCO's 80 classes). This file covers the newer generation: models that detect/segment **arbitrary categories described in natural language**, and the video-level problem of tracking objects across frames.

## Table of Contents

1. [Why Closed-Vocabulary Detection Isn't Enough](#1-why-closed-vocabulary-detection-isnt-enough)
2. [Grounding DINO — Open-Vocabulary Detection](#2-grounding-dino--open-vocabulary-detection)
3. [SAM — Segment Anything](#3-sam--segment-anything)
4. [Grounded SAM — Combining the Two](#4-grounded-sam--combining-the-two)
5. [Multi-Object Tracking](#5-multi-object-tracking)
6. [OCR — Scene Text Detection and Recognition](#6-ocr--scene-text-detection-and-recognition)
7. [Canonical Interview Q&As](#canonical-interview-qas)

---

## 1. Why Closed-Vocabulary Detection Isn't Enough

**The problem:** YOLO, Faster R-CNN, DETR (see [computer-vision.md](computer-vision.md) §4) are trained to detect a fixed set of classes baked in at training time — COCO's 80 categories, for instance. Adding a new class ("detect forklifts") means collecting labeled data and retraining. Production systems increasingly need to detect *whatever a user describes in text*, without retraining.

**The core insight:** replace the fixed classification head with a language-conditioned scoring function. Instead of "is this box a member of class $c \in \{1..80\}$," ask "does this box match this free-text phrase?" This turns detection into a vision-language grounding problem, solvable by models pretrained on image-text pairs at web scale (CLIP-style contrastive pretraining — see [computer-vision.md](computer-vision.md) and [05-llms/applications/multimodal.md](../../05-llms/applications/multimodal.md)).

## 2. Grounding DINO — Open-Vocabulary Detection

**The problem:** given an image and a free-text prompt ("red backpack," "person wearing a hat"), output bounding boxes for every matching instance — including categories never seen during training.

**The core insight:** fuse a DETR-style detection transformer with a text encoder, and perform cross-modal attention between image features and text-token features at multiple stages of the backbone (not just at the final classification layer). This lets text conditioning shape *which regions the detector even proposes*, not just how proposed boxes get labeled.

**The mechanics:**
```
Image → Vision backbone (Swin) ──┐
                                   ├─→ Cross-modal (image↔text) feature fusion → Language-guided query selection → Cross-modal decoder → boxes + phrase grounding scores
Text prompt → Text backbone (BERT) ──┘
```
- **Feature fusion** happens early: image and text features attend to each other before object queries are formed, so text can steer region proposals (unlike CLIP-based post-hoc filtering, which only re-ranks a fixed detector's outputs).
- **Language-guided query selection**: initial object queries are selected based on similarity to the text embedding, focusing the decoder's limited query budget on regions plausibly relevant to the prompt.
- **Output**: each predicted box comes with a similarity score against each token in the phrase, giving word-level grounding ("which word does this box correspond to") for free — useful for prompts describing multiple objects at once.

**What breaks**: open-vocabulary grounding is only as good as the underlying vision-language alignment — ambiguous or compositional prompts ("the second person from the left") routinely fail, since the model has no explicit spatial-reasoning module, only learned correlations between phrases and regions. Long-tail or highly technical categories (specific machine part names) perform far worse than common nouns, mirroring CLIP's own long-tail weaknesses.

## 3. SAM — Segment Anything

**The problem:** instance segmentation models (Mask R-CNN, Mask2Former — see [segmentation.md](segmentation.md) §2) require training on a fixed label taxonomy and produce one mask set per image, with no way to interactively select "just this one object" at inference time.

**The core insight:** decouple mask *quality* from mask *category*. Train a model to produce a valid segmentation mask for **any** prompt (a point, a box, or rough mask) without ever needing to know what the object is called — segmentation becomes a promptable, class-agnostic geometry problem, trained on a massive (11M image, 1.1B mask) dataset built via a model-in-the-loop annotation pipeline (SA-1B).

**The mechanics:**
```
Image ──→ Image Encoder (ViT-H, heavy, run once per image) ──→ Image embedding
Prompt (point / box / mask) ──→ Prompt Encoder (lightweight) ──┐
                                                                  ├─→ Mask Decoder (lightweight, fast) ──→ mask(s) + IoU confidence
Image embedding ─────────────────────────────────────────────────┘
```
- **Amortized cost split**: the expensive ViT image encoder runs once per image; the lightweight prompt encoder + mask decoder run per-prompt in ~50ms, enabling interactive use (click, get a mask, click again, get a refined mask) without re-encoding the image.
- **Ambiguity-aware output**: a single point prompt on, say, a person's shirt is ambiguous — does the user want the shirt, the person, or the whole scene? SAM outputs 3 mask candidates (whole/part/subpart) with confidence scores rather than forcing one answer.
- **Class-agnostic**: SAM outputs *a* mask, never a label. It answers "what is the boundary of the thing at this point," not "what is this thing" — that's why it composes naturally with a detector like Grounding DINO (§4) which supplies the "what."

**What breaks**: SAM has no semantic understanding — it cannot be prompted with text natively (SAM 2 added limited video-tracking prompts, still not open-vocabulary text). It also struggles with fine structures (thin wires, hair) and heavily overlapping/occluded instances, since its training signal is purely geometric mask quality, not semantic disambiguation.

## 4. Grounded SAM — Combining the Two

**The core insight:** Grounding DINO answers "where is the thing described by this text, as a box" (§2); SAM answers "given this box/point, what's the precise mask" (§3). Chaining them gives open-vocabulary *instance segmentation* — text in, precise pixel masks out — without training either model further:

```
Text prompt → Grounding DINO → boxes (+ labels) → SAM (box-prompted) → pixel-accurate masks
```
This pattern (specialist model composition rather than one model doing everything) is common in modern CV pipelines and mirrors the agentic "tool composition" pattern seen elsewhere in this repo ([agentic-ai-systems.md](../../08-emerging-topics/emerging-trends/agentic-ai-systems.md)) — each model is a callable tool with a narrow, well-defined interface (text→boxes, box→mask).

## 5. Multi-Object Tracking

**The problem:** detection gives per-frame boxes; video applications (surveillance, autonomous driving, sports analytics) need consistent object *identities* across frames — track ID 7 must refer to the same physical object in frame 1 and frame 300, even through brief occlusion.

**The core insight:** most production trackers follow **tracking-by-detection**: run a detector independently on every frame, then solve a data-association problem to link detections across frames into tracks. The hard part isn't detecting objects — it's the assignment problem under occlusion, missed detections, and visually similar objects.

**The mechanics — SORT / DeepSORT lineage:**
1. **Predict**: a Kalman filter predicts each existing track's next-frame bounding box from its motion history (constant-velocity assumption).
2. **Associate**: match new detections to predicted tracks by IoU (SORT) or IoU + appearance embedding similarity (DeepSORT — adds a small re-identification CNN so tracks survive occlusion by "recognizing" the object visually, not just by position).
3. **Assignment**: solved via the Hungarian algorithm on the cost matrix (1 − IoU, or a weighted combination of motion + appearance cost) — a bipartite matching problem, not a greedy nearest-match.
4. **Track lifecycle**: unmatched detections spawn new tracks (tentative, confirmed only after N consecutive matches to suppress false positives); unmatched tracks are kept alive for a grace window (to survive brief occlusion) before being deleted.

**ByteTrack's key fix**: earlier trackers discarded low-confidence detections before association, silently dropping partially-occluded objects (which produce weak detection scores). ByteTrack associates **all** detection boxes, high- and low-confidence, in two cascaded matching passes — high-confidence boxes matched first, then low-confidence boxes matched only against tracks still unmatched. This single change substantially reduced ID switches under occlusion without changing the detector.

**What breaks**: appearance-based re-identification fails when objects are visually near-identical (a crowd of people in the same uniform) or when lighting/pose changes drastically between the last-seen and re-appearing frames. Pure motion-based association (SORT) fails during long occlusions or non-linear motion (sudden direction changes) since the constant-velocity Kalman prediction drifts.

## 6. OCR — Scene Text Detection and Recognition

**The problem:** extracting text from natural images (street signs, product labels, documents) is a two-stage problem distinct from clean, scanned-document OCR — text can be small, rotated, curved, occluded, or stylized, and must first be *located* before it can be *read*.

**The core insight:** decompose into **detection** (where is text, as a box or polygon — since scene text is often non-rectangular) and **recognition** (what does the localized crop say, as a sequence-prediction problem over characters).

**The mechanics:**
- **Text detection**: adapts general object detection to text's extreme aspect ratios and curved layouts. DBNet (Differentiable Binarization) predicts a per-pixel probability map plus a learned adaptive threshold, then extracts polygon boundaries — differentiable binarization allows the thresholding step to be trained end-to-end rather than treated as fixed post-processing.
- **Text recognition**: a CRNN (CNN feature extractor + BiLSTM sequence encoder) predicts a character sequence per cropped region, trained with **CTC loss** (Connectionist Temporal Classification) — critical because the recognizer never has ground-truth per-character alignment, only the target string; CTC marginalizes over all possible alignments between predicted per-timestep character distributions and the target sequence, using a blank token to handle repeated/absent characters.
- **Modern unified approaches**: transformer-based end-to-end models (e.g., TrOCR) replace the CNN+BiLSTM+CTC pipeline with a ViT encoder + autoregressive transformer decoder, treating recognition as image-to-text generation directly — trading the CTC alignment-marginalization trick for standard cross-entropy sequence generation, at the cost of slower (non-parallel) autoregressive decoding per crop.

**What breaks**: detection fails on extreme perspective distortion or very small/dense text (receipts, dense tables); recognition degrades sharply on rare fonts, handwriting, or non-Latin scripts underrepresented in training data. Production OCR systems typically layer a language model or dictionary-based post-correction step on top of raw recognition output to fix single-character errors using linguistic priors.

---

## Canonical Interview Q&As

**Q: How does Grounding DINO achieve open-vocabulary detection, and why can't you just use CLIP with a normal detector?**
A: A naive approach — run a class-agnostic region proposal network, then classify each region by CLIP similarity to the text prompt — is limited because the region proposals themselves are never informed by the text; if the proposal network was trained on a fixed label set, it will systematically under-propose boxes for object types outside that distribution, and CLIP can only re-rank what's already proposed. Grounding DINO instead fuses image and text features *before* query/box generation (cross-modal feature fusion + language-guided query selection), so the text prompt actively shapes which regions the model even considers, not just how they're scored afterward. This end-to-end conditioning is why Grounding DINO generalizes to novel categories better than post-hoc CLIP filtering of a closed-vocabulary detector's proposals.

**Q: Why does SAM output multiple masks per prompt instead of one, and why is the image encoder decoupled from the prompt encoder?**
A: Multiple masks address prompt ambiguity — a single point on an object is inherently underspecified (a click on a shirt could mean "the shirt," "the person," or "the person's whole outfit"), so SAM predicts 3 mask hypotheses at different granularities with confidence scores rather than committing to one interpretation the user didn't ask for. The image/prompt encoder split is a latency optimization for interactive use: the heavy ViT image encoder runs once per image (hundreds of ms), producing a reusable embedding; the lightweight prompt encoder and mask decoder then run per-click in ~50ms, so a user can click multiple times to refine a selection without re-encoding the whole image each time — this amortized-cost design is what makes SAM usable as an interactive annotation tool rather than a batch-only model.

**Q: Walk through what happens when a tracked object is occluded for several frames in a DeepSORT-style tracker, and what ByteTrack changes about this.**
A: In classic SORT, when a track's detection disappears (occlusion), the Kalman filter keeps predicting the box forward using constant-velocity motion, and the track is kept "alive" for a grace period (e.g., 30 frames) waiting for a re-matching detection; if the object reappears with a similar position/motion, IoU-based matching recovers it, but if the occlusion is long or the object's motion was non-linear, position drift makes the reappearing detection fall outside the IoU-based gating and a new track ID is spawned (an "ID switch"). DeepSORT mitigates this using an appearance embedding (a small re-ID CNN) so re-matching can also succeed via visual similarity even after position drift makes IoU fail. ByteTrack's contribution is orthogonal to appearance modeling: it observed that partially-occluded objects produce genuine but low-confidence detections that earlier pipelines discarded before association — by running a second association pass that matches these low-confidence boxes against still-unmatched tracks (rather than discarding them upfront), ByteTrack recovers tracks through partial occlusion using detections that would otherwise have been thrown away, reducing ID switches without any change to the underlying detector or appearance model.

---

## Where to Next

- **Prerequisite: closed-vocabulary detection** → [computer-vision.md](computer-vision.md) §4
- **Prerequisite: instance segmentation architectures** → [segmentation.md](segmentation.md) §2
- **Vision-language pretraining (CLIP) underlying open-vocabulary grounding** → [computer-vision.md](computer-vision.md), [05-llms/applications/multimodal.md](../../05-llms/applications/multimodal.md)
- **Video-level temporal modeling beyond tracking** → [video-understanding.md](video-understanding.md)
