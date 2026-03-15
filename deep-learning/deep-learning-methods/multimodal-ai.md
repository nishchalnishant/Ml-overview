# Multimodal AI

Multimodal models combine multiple modalities (e.g. image, text, audio, video) in a single architecture. Post-2020, **vision–language models (VLMs)** and **contrastive image–text pretraining** are central to products and research.

---

## Vision–language models

- **Goal:** Joint understanding and generation over images and text (captioning, VQA, document understanding, instruction following).
- **Architecture:** **Vision encoder** (CNN or ViT) turns image into patch or image tokens; **projection** maps them into the **LLM token space**; **LLM** (decoder-only transformer) attends to both text and vision tokens and generates text. Training: image–text pairs or instruction data with images.
- **Examples:** LLaVA, GPT-4V, Gemini, Claude (multimodal); used for charts, OCR, reasoning over images, and agent interfaces.

---

## CLIP-style training

- **CLIP (Contrastive Language–Image Pre-training):** Image encoder and text encoder produce embeddings; training objective: maximize similarity of matched image–text pairs and minimize similarity of non-matched pairs (InfoNCE / contrastive loss over batch).
- **Result:** Shared embedding space; **zero-shot** image classification by matching image to text labels; also retrieval (image→text, text→image). **Extensions:** ALIGN, Florence; scaling data and model improves robustness and capability.

---

## Multimodal transformers

- **Single sequence:** Concatenate image tokens (from patch embedding) and text tokens; one transformer processes both with self-attention. **Cross-attention:** Encoder for one modality, decoder attends to encoder output (e.g. image encoder + text decoder for captioning).
- **Position encoding:** Separate or unified positional embeddings for image and text; some models use 2D positional encoding for patches.

---

## Image–text alignment

- **Contrastive:** CLIP-style; align embeddings of paired vs unpaired image–text. **Generative:** Model generates text given image (captioning, VQA); alignment via cross-entropy or next-token loss. **Combined:** Pretrain with contrastive then fine-tune with generative (e.g. BLIP-2).

---

## Audio–text and video–language

- **Audio–text:** Similar to CLIP but with audio encoder (e.g. Wav2Vec, HuBERT) and text; contrastive or captioning. **Speech:** Whisper (encoder–decoder for ASR/translation); multimodal if combined with text LLMs.
- **Video–language:** Encode video as sequence of frame embeddings (or 3D conv / video transformer); concatenate with text and use transformer for QA, captioning, or retrieval. **Examples:** VideoCLIP, Flamingo, video-LLaMA.

---

## Quick revision

- **VLMs:** Vision encoder + LLM; captioning, VQA, instruction following. **CLIP:** Contrastive image–text pretraining; shared embedding, zero-shot. **Multimodal transformers:** Single or cross-attention over image and text tokens. **Audio/video:** Same principles with modality-specific encoders.
