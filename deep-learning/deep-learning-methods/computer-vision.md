# Computer Vision

Computer vision uses deep learning for image and video understanding. Post-2020, **Vision Transformers (ViT)** and **hybrid** architectures complement **CNNs**; models are pretrained at scale and fine-tuned or used as backbones for detection and segmentation.

---

## CNNs (Convolutional Neural Networks)

- **Convolution:** Slide filters (kernels) over the image; each filter detects local patterns (edges, textures). **Pooling:** Downsample (e.g. max pool) to reduce spatial size and add invariance.
- **Stack:** Conv → activation (ReLU) → pool, repeated; then fully connected layers for classification. **Representative:** ResNet (skip connections), EfficientNet (compound scaling).
- **Use:** Image classification, backbone for detection and segmentation.

---

## Vision Transformers (ViT)

- **Patch embedding:** Split image into patches (e.g. 16×16); linearly embed each patch; add positional embedding. **Transformer:** Standard encoder (self-attention + FFN) over patch tokens; [CLS] or average pooling for image-level representation.
- **Pretraining:** Often with **supervised** labels (ImageNet) or **self-supervised** (e.g. MAE: mask patches, reconstruct pixels). ViT scales well with data and model size.
- **Hybrid:** CNN backbone + transformer (e.g. some detection/segmentation models); combine local feature hierarchy with global attention.

---

## Detection and segmentation

- **Object detection:** Localize and classify objects (bounding boxes). Methods: two-stage (R-CNN family), one-stage (YOLO, RetinaNet). **Backbone:** ResNet, ViT, or hybrid; **head:** box regression + classification.
- **Segmentation:** Pixel-level labels. **Semantic:** Class per pixel. **Instance:** Separate instances of same class. **Architectures:** U-Net, Mask R-CNN, vision transformers with decoder.

---

## Multimodal vision–language

- **CLIP:** Contrastive pretraining on image–text pairs; shared embedding space for zero-shot image classification and retrieval. See [Multimodal AI](multimodal-ai.md).
- **Vision–language models (VLMs):** Combine vision encoder + LLM; train for captioning, VQA, or instruction following (e.g. LLaVA, GPT-4V). Used for document understanding, charts, and agents.

---

## Quick revision

- **CNN:** Convolutions + pooling; ResNet, EfficientNet. **ViT:** Patches → transformer encoder; scales with data. **Detection:** Bounding boxes (YOLO, R-CNN). **Segmentation:** Pixel-level (U-Net, Mask R-CNN). **VLMs:** Vision encoder + LLM for image understanding and generation.
