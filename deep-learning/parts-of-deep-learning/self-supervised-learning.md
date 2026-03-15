# Self-Supervised Learning

Self-supervised learning (SSL) uses unlabeled data by defining a **pretext task** that produces supervisory signal from the data itself. Post-2020, **contrastive learning** and **masked prediction** dominate vision and language representation learning.

---

## Contrastive learning

**Idea:** Learn representations so that **positive pairs** (e.g. two augmentations of the same image) are close in embedding space and **negatives** (different images) are far. Loss: **InfoNCE** or **NT-Xent** (normalized temperature-scaled cross-entropy).

**SimCLR:** For each image, create two augmented views; encode with shared backbone; project with MLP; contrastive loss over batch (other views in batch are negatives). **No momentum encoder** in v1; strong augmentation (crop, color, blur) is critical.

**BYOL:** Two branches (online + target); target is exponential moving average of online. Predict target from online; no explicit negatives. **Avoids collapse** via asymmetry (target not updated by gradient).

**DINO:** Self-distillation with no labels; student and teacher (EMA) share same architecture; cross-entropy between student and teacher softmax over batch. Produces semantically meaningful features and attention maps.

---

## Masked prediction

**Masked autoencoders (MAE):** Randomly mask a high fraction of image patches (e.g. 75%); encoder sees only visible patches; decoder reconstructs masked patches from encoder output + mask tokens. **Loss:** MSE on normalized pixel values of masked patches. **Efficiency:** Encoder runs on small subset of patches; fast and scalable. Dominant for vision SSL in many benchmarks.

**BERT-style MLM:** In NLP, mask tokens in input; predict masked tokens. Standard for encoder-only language models.

---

## Representation learning

- SSL pretraining produces **general-purpose representations** that transfer to downstream tasks via linear probe or fine-tuning. **Evaluation:** Linear evaluation on ImageNet (freeze backbone, train linear classifier); fine-tuning; retrieval.
- **Trend:** Scale data and model; combine with supervision (semi-supervised or multitask) for best downstream performance.

---

## Quick revision

- **Contrastive:** SimCLR (augmentations, InfoNCE), BYOL (no negatives, EMA target), DINO (self-distillation). **Masked:** MAE (mask patches, reconstruct); BERT (mask tokens). SSL learns transferable representations without labels.
