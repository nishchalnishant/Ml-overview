# Computer Vision

---

# Q1: What is computer vision, and why is it important?

## 1. 🔹 Direct Answer
**Computer vision (CV)** is building algorithms that **infer** useful information from **images or video** (pixels): classification, detection, segmentation, tracking, 3D, etc. It matters for automation, accessibility, robotics, medical imaging, and safety systems.

## 2. 🔹 Intuition
Humans parse scenes effortlessly; CV **approximates** that pipeline with **representations** learned from data.

## 3. 🔹 Deep Dive
- Classical: filters, HOG, SIFT; modern: **CNNs**, **ViTs**, multimodal models.
- Pipeline: **acquire → preprocess → model → post-process** (NMS, calibration).

## 4. 🔹 Practical Perspective
- Use: quality inspection, OCR, autonomous systems, AR.
- Hard: **domain shift**, **lighting**, **occlusion**, **latency** on edge devices.

## 5. 🔹 Code Snippet
```python
import torch.nn as nn
model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** CV vs NLP? **A:** Grid structure vs sequence; inductive biases (conv locality vs attention).
2. **Q:** Real-time? **A:** TensorRT, quantization, smaller backbones.

## 7. 🔹 Common Mistakes
Treating ImageNet accuracy as transfer to medical/industrial without domain adaptation.

## 8. 🔹 Comparison / Connections
Multimodal LLMs, sensor fusion, graphics.

## 9. 🔹 One-line Revision
CV maps pixels to semantics via learned representations—watch data shift and deployment constraints.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: What is image segmentation, and what are its applications?

## 1. 🔹 Direct Answer
**Segmentation** assigns a **label to each pixel** (semantic: class per pixel; instance: separate objects). Apps: medical imaging, autonomous driving, matting, robotics grasping.

## 2. 🔹 Intuition
Classification says “what”; segmentation says “**where** exactly,” at pixel level.

## 3. 🔹 Deep Dive
- Architectures: **U-Net**, **DeepLab**, **Mask R-CNN** (instance).
- Loss: **cross-entropy**, **Dice**, **IoU**-based; handle class imbalance.

## 4. 🔹 Practical Perspective
- Annotation is expensive—weak labels, semi-supervised, synthetic data.
- Metrics: **mIoU**, boundary F-score.

## 5. 🔹 Code Snippet
```python
# IoU for binary masks
def iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return inter / union if union else 1.0
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Instance vs semantic? **A:** Semantic merges same class; instance separates individuals.

## 7. 🔹 Common Mistakes
Reporting pixel accuracy on imbalanced backgrounds—misleading.

## 8. 🔹 Comparison / Connections
Object detection (boxes), panoptic segmentation.

## 9. 🔹 One-line Revision
Segmentation is dense labeling—U-Net-style encoder-decoders and IoU-style metrics dominate practice.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What is object detection, and how does it differ from image classification?

## 1. 🔹 Direct Answer
**Classification**: one label for whole image. **Detection**: **where** (bounding boxes) + **what** (class per box), often multiple objects. Requires **localization** + **classification**.

## 2. 🔹 Intuition
Classification answers “cat or dog?”; detection answers “**how many** cats, **where**?”

## 3. 🔹 Deep Dive
- Families: **two-stage** (Faster R-CNN), **one-stage** (YOLO, SSD), **transformers** (DETR).
- Training: anchor boxes or anchor-free; losses combine **cls + reg** (IoU/GIoU).

## 4. 🔹 Practical Perspective
- Use detection when **counting** or **interaction** matters; else classification is simpler.
- Metrics: **mAP** at IoU thresholds.

## 5. 🔹 Code Snippet
```text
mAP: mean AP over classes; AP = area under precision-recall curve
```

## 6. 🔹 Interview Follow-ups
1. **Q:** NMS? **A:** Suppress overlapping boxes by score—tune IoU threshold.

## 7. 🔹 Common Mistakes
Confusing **top-1 accuracy** with **detection mAP**.

## 8. 🔹 Comparison / Connections
Segmentation (masks), tracking (temporal association).

## 9. 🔹 One-line Revision
Detection = multi-object localization + classification; optimize mAP, not single-label accuracy.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What are the steps to build an image recognition system?

## 1. 🔹 Direct Answer
**Define** task/metrics → **data** (collect, label, split by time/entity) → **baseline** (simple model) → **iterate** (architecture, aug, regularization) → **validate** (calibration, slices) → **deploy** (latency, monitoring, drift).

## 2. 🔹 Intuition
Ship a **vertical slice** early; don’t optimize ImageNet on day one.

## 3. 🔹 Deep Dive
- **Data**: train/val/test leakage (same person in both = bad).
- **Augmentation**: flips, color jitter—match deployment distortions.
- **Monitoring**: input resolution, exposure shift.

## 4. 🔹 Practical Perspective
- Edge: quantize, prune; cloud: larger models OK.
- **Failure modes**: OOD inputs, adversarial patches (brief).

## 5. 🔹 Code Snippet
```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Active learning? **A:** Label uncertain/hard examples first.

## 7. 🔹 Common Mistakes
Random split for time-series imagery (leakage).

## 8. 🔹 Comparison / Connections
MLOps, experiment tracking, A/B testing.

## 9. 🔹 One-line Revision
End-to-end: problem definition, clean splits, strong baseline, iterate with deployment-aware metrics.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: What are the challenges in real-time object tracking?

## 1. 🔹 Direct Answer
**Latency** budget, **ID switches** (occlusion), **scale** variation, **motion blur**, **similar-looking** objects, **camera jitter**, and **association** across frames (SORT/DeepSORT, trackers).

## 2. 🔹 Intuition
Detection per frame is heavy; tracking adds **temporal** reasoning and **identity** stability.

## 3. 🔹 Deep Dive
- **Detection + association** vs end-to-end trackers.
- Metrics: **MOTA**, **IDF1**, FPS.
- Trade-off: accuracy vs **edge** compute.

## 4. 🔹 Practical Perspective
Sports analytics, drones, retail—often **reduce resolution** or **ROI** crop first.

## 5. 🔹 Code Snippet
```text
Kalman filter for motion prior + Hungarian matching for association (classic pipeline)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Re-ID embeddings? **A:** Appearance features help after occlusion.

## 7. 🔹 Common Mistakes
Ignoring **camera calibration** and **frame sync** in multi-cam setups.

## 8. 🔹 Comparison / Connections
Video understanding, multi-object tracking, SLAM.

## 9. 🔹 One-line Revision
Real-time tracking juggles speed, association under occlusion, and stable IDs—measure MOTA/IDF1 at target FPS.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q6: What is feature extraction in computer vision?

## 1. 🔹 Direct Answer
**Feature extraction** maps raw pixels to **compact, informative** vectors (HOG, SIFT, or **CNN/ViT embeddings**) that make downstream tasks (classification, retrieval) easier.

## 2. 🔹 Intuition
Raw pixels are high-dimensional and redundant; features capture **edges, textures, semantics**.

## 3. 🔹 Deep Dive
- **Hand-crafted**: gradients, keypoints—interpretable, limited.
- **Learned**: intermediate layer activations or **CLS** token; often **fine-tuned** or frozen backbone.

## 4. 🔹 Practical Perspective
- **Retrieval**: L2-normalized embeddings + ANN index.
- **Transfer**: ImageNet-pretrained backbone as default starting point.

## 5. 🔹 Code Snippet
```python
with torch.no_grad():
    feats = backbone(batch)["pooler_output"]  # e.g., ViT/ResNet head
    feats = torch.nn.functional.normalize(feats, dim=1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which layer? **A:** Earlier = texture; later = semantics—task-dependent.

## 7. 🔹 Common Mistakes
Using features from **wrong preprocessing** (mean/std mismatch).

## 8. 🔹 Comparison / Connections
Self-supervised learning (SimCLR, DINO), metric learning.

## 9. 🔹 One-line Revision
Features are compressed representations—hand-crafted or learned; match preprocessing and domain.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is OCR, and what are its main applications?

## 1. 🔹 Direct Answer
**OCR (Optical Character Recognition)** converts **images of text** into machine-readable text. Uses: document digitization, receipts, license plates, assistive tech, search indexing.

## 2. 🔹 Intuition
Pipeline: **detect text regions** → **recognize** characters/words; modern end-to-end models handle layout.

## 3. 🔹 Deep Dive
- Challenges: **fonts**, **skew**, **noise**, **languages**, **handwriting**.
- Metrics: **CER/WER** (character/word error rate).

## 4. 🔹 Practical Perspective
- Preprocess: deskew, binarize when helpful.
- **Layout**: tables need structure-aware models (not plain line OCR).

## 5. 🔹 Code Snippet
```text
# Typical stack: detect (DB/EAST) + CRNN/Transformer recognizer
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Handwriting? **A:** Harder; more diverse data; seq2seq with attention.

## 7. 🔹 Common Mistakes
Evaluating on clean screenshots only—fails in the wild.

## 8. 🔹 Comparison / Connections
Document AI, multimodal LLMs with image input.

## 9. 🔹 One-line Revision
OCR is detect-then-recognize (or end-to-end); optimize CER/WER on realistic noise and layouts.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: How does CNN differ from traditional neural networks in computer vision?

## 1. 🔹 Direct Answer
**CNNs** use **local connectivity**, **weight sharing** (filters), and **pooling**—exploiting **spatial structure**. Fully-connected nets on raw pixels don’t scale and ignore **translation equivariance** (approximately).

## 2. 🔹 Intuition
A CNN slides the same small pattern detector across the image—like spotting edges everywhere efficiently.

## 3. 🔹 Deep Dive
- **Convolution**: **(H−k+2p)/s + 1** output size; parameters **independent** of input size (per layer).
- **Hierarchical features**: edges → textures → parts → objects.

## 4. 🔹 Practical Perspective
- Data efficiency vs MLP on images.
- Modern: **ViTs** replace conv with patches + attention—still common to hybridize.

## 5. 🔹 Code Snippet
```python
import torch.nn as nn
conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Translation invariance vs equivariance? **A:** Conv is equivariant; pooling adds local invariance.

## 7. 🔹 Common Mistakes
Saying CNNs are “invariant” to translation without nuance.

## 8. 🔹 Comparison / Connections
ViT, locality-sensitive hashing, spatial pyramid pooling.

## 9. 🔹 One-line Revision
CNNs encode spatial locality and hierarchy via shared filters—parameter-efficient vs dense nets on images.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: What is data augmentation, and what techniques are commonly used?

## 1. 🔹 Direct Answer
**Augmentation** applies **label-preserving transforms** to training images to **reduce overfitting** and improve **generalization**. Common: flips, crops, rotation, color jitter, blur, CutOut/MixUp, **RandAugment**.

## 2. 🔹 Intuition
Synthetic diversity teaches invariance the test set may need—without new labels.

## 3. 🔹 Deep Dive
- **Match deployment**: medical may forbid flips if orientation matters.
- **Strong aug** + regularization can hurt small data—tune carefully.
- **MixUp/CutMix**: interpolate labels—soft targets.

## 4. 🔹 Practical Perspective
- **Test-time augmentation (TTA)**: average predictions—latency cost.
- **AutoAugment**: search policies—expensive offline.

## 5. 🔹 Code Snippet
```python
from torchvision import transforms as T
train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ColorJitter(0.2,0.2,0.2,0.1)])
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Aug hurts? **A:** Over-distortion breaks label semantics (digit 6 vs 9).

## 7. 🔹 Common Mistakes
Using augmentations that **violate** physics/domain constraints.

## 8. 🔹 Comparison / Connections
Regularization, domain randomization in sim2real.

## 9. 🔹 One-line Revision
Augment with domain-appropriate transforms to simulate test variability and curb overfitting.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What are some popular deep learning frameworks for computer vision?

## 1. 🔹 Direct Answer
**PyTorch** (dynamic graphs, research + prod), **TensorFlow/Keras**, **JAX/Flax** (research, XLA), **ONNX Runtime** / **TensorRT** for **deployment**. Higher-level: **timm**, **torchvision**, **OpenCV** for I/O.

## 2. 🔹 Intuition
Pick framework by **team**, **deployment target**, and **ecosystem** (pretrained models, quantization tools).

## 3. 🔹 Deep Dive
- **torch.compile**, **TF Serving**, **Triton** inference server.
- Mobile: **CoreML**, **TFLite**, **NCNN**.

## 4. 🔹 Practical Perspective
- Research: PyTorch dominant in many labs.
- Google stack: TensorFlow + GCP integrations.

## 5. 🔹 Code Snippet
```python
import timm
m = timm.create_model("resnet50", pretrained=True, num_classes=1000)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** PyTorch vs TF? **A:** Dynamic vs historical static graph; both mature—team fit matters.

## 7. 🔹 Common Mistakes
Choosing framework without considering **serving** and **quantization** path.

## 8. 🔹 Comparison / Connections
MLOps, model registry, CI for ML.

## 9. 🔹 One-line Revision
PyTorch/TF for training; ONNX/TensorRT/Triton for optimized inference—choose for ecosystem and deployment.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q11: How can Transformers be used for computer vision tasks?

## 1. 🔹 Direct Answer
**ViT** splits images into **patches**, embeds them as tokens, runs **Transformer** encoder—**global** self-attention models long-range deps. Variants: **Swin**, **DeiT**, **DETR** for detection; **Segmenter** for segmentation.

## 2. 🔹 Intuition
Same attention machinery as NLP, but on **patch sequences**—less inductive bias than CNNs, often needs **more data** or **strong aug**.

## 3. 🔹 Deep Dive
- **Complexity**: O(n²) in patches—use windowed/local attention (Swin) for efficiency.
- **Pretrain** on large image data or multimodal (CLIP).

## 4. 🔹 Practical Perspective
- Great when you have **scale**; CNNs still strong on small data / edge.
- **Hybrid** CNN stem + Transformer body is common.

## 5. 🔹 Code Snippet
```text
patch_embed(x) -> [B, N, D] -> TransformerEncoder -> cls or dense prediction
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Inductive bias? **A:** CNN: locality; ViT learns it from data—data-hungry.

## 7. 🔹 Common Mistakes
Claiming ViT always beats ResNet on small datasets.

## 8. 🔹 Comparison / Connections
CNNs, multimodal LLMs with vision encoder (CLIP, Flamingo).

## 9. 🔹 One-line Revision
Vision Transformers tokenize patches and apply self-attention—powerful at scale; use efficient/local variants for speed.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
