# Computer Vision — Key Papers

The papers that built the field, and the ones that still get asked about in interviews.

---

## CNN Foundations

| Paper | Year | Why It Matters |
|---|---|---|
| [ImageNet Classification with Deep CNNs — AlexNet (Krizhevsky et al.)](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 2012 | The paper that started the deep learning era — dropout, ReLU, GPU training |
| [Very Deep Convolutional Networks — VGGNet (Simonyan & Zisserman)](https://arxiv.org/abs/1409.1556) | 2014 | Simplicity wins: deep stacks of 3×3 convolutions |
| [Going Deeper with Convolutions — GoogLeNet/Inception (Szegedy et al.)](https://arxiv.org/abs/1409.4842) | 2014 | Inception modules — parallel multi-scale convolutions |
| [Deep Residual Learning for Image Recognition — ResNet (He et al.)](https://arxiv.org/abs/1512.03385) | 2015 | Skip connections solve vanishing gradients — enables 100s of layers |
| [Densely Connected Convolutional Networks — DenseNet (Huang et al.)](https://arxiv.org/abs/1608.06993) | 2016 | Every layer connects to all subsequent layers — maximum feature reuse |
| [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le)](https://arxiv.org/abs/1905.11946) | 2019 | Compound scaling of depth/width/resolution — NAS-derived baseline |

---

## Object Detection

| Paper | Year | Why It Matters |
|---|---|---|
| [Rich Feature Hierarchies for Accurate Object Detection — R-CNN (Girshick et al.)](https://arxiv.org/abs/1311.2524) | 2013 | Region proposals + CNN features — the two-stage paradigm |
| [Fast R-CNN (Girshick)](https://arxiv.org/abs/1504.08083) | 2015 | ROI pooling — share convolutional computation |
| [Faster R-CNN (Ren et al.)](https://arxiv.org/abs/1506.01497) | 2015 | Region Proposal Network — end-to-end trainable detection |
| [You Only Look Once — YOLOv1 (Redmon et al.)](https://arxiv.org/abs/1506.02640) | 2015 | Single-pass detection — speed over two-stage accuracy |
| [Feature Pyramid Networks — FPN (Lin et al.)](https://arxiv.org/abs/1612.03144) | 2016 | Multi-scale feature hierarchy for detection |
| [Focal Loss for Dense Object Detection — RetinaNet (Lin et al.)](https://arxiv.org/abs/1708.02002) | 2017 | Focal loss solves class imbalance in one-stage detectors |
| [End-to-End Object Detection with Transformers — DETR (Carion et al.)](https://arxiv.org/abs/2005.12872) | 2020 | Transformer for detection — no NMS, no anchors |

---

## Segmentation

| Paper | Year | Why It Matters |
|---|---|---|
| [Fully Convolutional Networks for Semantic Segmentation — FCN (Long et al.)](https://arxiv.org/abs/1411.4038) | 2014 | Replace FC layers with convolutions — any-size input segmentation |
| [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) | 2015 | Encoder-decoder with skip connections — still dominant for biomedical |
| [Mask R-CNN (He et al.)](https://arxiv.org/abs/1703.06870) | 2017 | Extends Faster R-CNN with pixel mask prediction |
| [Segment Anything Model — SAM (Kirillov et al., Meta)](https://arxiv.org/abs/2304.02643) | 2023 | Foundation model for segmentation — prompt-driven, zero-shot |

---

## Vision Transformers

| Paper | Year | Why It Matters |
|---|---|---|
| [An Image is Worth 16x16 Words — ViT (Dosovitskiy et al.)](https://arxiv.org/abs/2010.11929) | 2020 | Transformer on image patches — competitive with CNNs at scale |
| [Training data-efficient image transformers — DeiT (Touvron et al.)](https://arxiv.org/abs/2012.12877) | 2020 | ViT without giant pretraining datasets — knowledge distillation |
| [Swin Transformer (Liu et al.)](https://arxiv.org/abs/2103.14030) | 2021 | Hierarchical ViT with shifted windows — better for dense prediction |

---

## Contrastive & Self-Supervised Learning

| Paper | Year | Why It Matters |
|---|---|---|
| [Learning Transferable Visual Models from Natural Language Supervision — CLIP (Radford et al.)](https://arxiv.org/abs/2103.00020) | 2021 | Vision-language alignment via contrastive pretraining — zero-shot classification |
| [A Simple Framework for Contrastive Learning — SimCLR (Chen et al.)](https://arxiv.org/abs/2002.05709) | 2020 | Augmentation-based contrastive self-supervised learning |
| [Momentum Contrast for Unsupervised Visual Representation Learning — MoCo (He et al.)](https://arxiv.org/abs/1911.05722) | 2019 | Dictionary-as-queue for contrastive learning |
| [Emerging Properties in Self-Supervised Vision Transformers — DINO (Caron et al.)](https://arxiv.org/abs/2104.14294) | 2021 | Self-supervised ViT — attention maps reveal semantic segmentation for free |
| [Masked Autoencoders Are Scalable Vision Learners — MAE (He et al.)](https://arxiv.org/abs/2111.06377) | 2021 | BERT-style masking for images — 75% masking, reconstruct pixels |

---

## Generative Vision Models

| Paper | Year | Why It Matters |
|---|---|---|
| [Generative Adversarial Networks — GAN (Goodfellow et al.)](https://arxiv.org/abs/1406.2661) | 2014 | The original GAN — generator vs discriminator game |
| [Progressive Growing of GANs (Karras et al.)](https://arxiv.org/abs/1710.10196) | 2017 | High-res image synthesis by growing resolution progressively |
| [A Style-Based Generator Architecture — StyleGAN (Karras et al.)](https://arxiv.org/abs/1812.04948) | 2018 | Disentangled latent space — controllable generation |
| [Denoising Diffusion Probabilistic Models — DDPM (Ho et al.)](https://arxiv.org/abs/2006.11239) | 2020 | The paper that made diffusion models mainstream |
| [High-Resolution Image Synthesis with Latent Diffusion Models — Stable Diffusion (Rombach et al.)](https://arxiv.org/abs/2112.10752) | 2022 | Diffusion in latent space — 10× cheaper than pixel-space |

---

## Key Interview Takeaways

**"Explain the intuition behind skip connections in ResNet."**

Without them, gradients shrink as they pass through 100+ layers (vanishing gradient). Skip connections create a "gradient highway" — the identity shortcut means the gradient can bypass layers entirely, so deep networks effectively learn residuals (corrections) on top of identity mappings. This is like fixing a bug on top of working code rather than rewriting from scratch.

**"When would you use ViT over a CNN?"**

At scale (large data, large model), ViT outperforms CNNs because attention can model long-range dependencies that convolutions miss. CNNs are better with limited data (inductive bias of locality/translation-equivariance helps). In practice: fine-tune ViT from a strong pretrained checkpoint (ImageNet-21k, CLIP) rather than training from scratch.

**"What does CLIP enable that wasn't possible before?"**

Zero-shot classification without task-specific labels — just describe the categories in text. Also enables image retrieval with text queries, multimodal embeddings for retrieval, and is the visual backbone for GPT-4V, LLaVA, and most modern vision-language models.

**"Why does MAE use 75% masking ratio while BERT uses 15%?"**

Images are spatially redundant — you can reconstruct most patches from context, so 15% masking is too easy. 75% masking forces the model to learn meaningful representations rather than just interpolating nearby pixels.
