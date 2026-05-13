# Multimodal AI — Deep Dive

> Architecture, training, deployment, and interview preparation for vision-language, audio, and video models.

---

## Table of Contents

1. [Multimodal Landscape](#1-multimodal-landscape)
2. [Vision Encoders](#2-vision-encoders)
3. [CLIP — Contrastive Pre-training](#3-clip--contrastive-pre-training)
4. [Fusion Architectures](#4-fusion-architectures)
5. [Native Multimodal LLMs](#5-native-multimodal-llms)
6. [Audio Integration](#6-audio-integration)
7. [Video Models](#7-video-models)
8. [Production Deployment](#8-production-deployment)
9. [Interview Questions](#9-interview-questions)

---

## 1. Multimodal Landscape

### Three Generations of VLMs

**Generation 1 — Frozen fusion (2022):** Freeze a pretrained vision encoder and LLM; train only a small projection layer. BLIP-2, LLaVA-1. Cheap to train, but limited cross-modal integration depth.

**Generation 2 — Co-trained fusion (2023-2024):** Train vision encoder jointly with LLM, or with larger bridging components. InstructBLIP, LLaVA-1.5, InternVL2. Better visual reasoning.

**Generation 3 — Native multimodal (2024-2025):** Process all modalities as a unified token stream from the first layer. Gemini, GPT-4o, Llama 4. Deepest integration, most expensive to train from scratch.

---

### Modality Coverage by Production Model

| Model | Text | Image | Audio | Video | Docs/PDF |
|-------|------|-------|-------|-------|----------|
| GPT-4o | ✅ | ✅ | ✅ (native) | ✅ | ✅ |
| Claude 3.7 | ✅ | ✅ | ❌ | ❌ | ✅ |
| Gemini 2.0 Flash | ✅ | ✅ | ✅ | ✅ | ✅ |
| Llama 4 Scout | ✅ | ✅ | ❌ | ❌ | ✅ |
| Qwen2.5-VL | ✅ | ✅ | ❌ | ✅ | ✅ |

---

## 2. Vision Encoders

### ViT — Vision Transformer (Dosovitskiy et al., 2020)

Split image into non-overlapping patches, flatten, project to embedding vectors, prepend [CLS] token, apply standard transformer:

```
Input image: H × W × C
Patch size: P × P
Number of patches: N = (H/P) × (W/P)

Patch embeddings: E ∈ R^{N × d_model}
Position embeddings: added (learned or sinusoidal)
CLS token: [cls; E] → transformer → CLS output = image representation
```

**Standard configurations:**
| Variant | Layers | d_model | Heads | Patch size | Params |
|---------|--------|---------|-------|-----------|--------|
| ViT-B/16 | 12 | 768 | 12 | 16×16 | 86M |
| ViT-L/14 | 24 | 1024 | 16 | 14×14 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 14×14 | 632M |

**Why ViT replaced CNNs for multimodal:** ViT produces a sequence of patch embeddings — same format as token embeddings in an LLM. Bridging becomes a projection rather than a modality translation. CNNs produce spatial feature maps that require more complex adapters.

### DINOv2 (Meta, 2023)

Self-supervised ViT pre-training using self-distillation with no labels. Teacher model (EMA of student) generates pseudo-labels; student matches them.

**Why it matters for VLMs:** DINOv2 features are more spatially aligned than CLIP features — they capture "where" things are, not just "what" they are. Better for dense tasks (OCR, object localization) in multimodal models.

```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = AutoModel.from_pretrained("facebook/dinov2-large")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# patch_embeddings: [1, num_patches, 1024] — one embedding per 14×14 patch
patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # exclude CLS
cls_embedding = outputs.last_hidden_state[:, 0, :]       # CLS token
```

---

## 3. CLIP — Contrastive Pre-training

### Architecture

Two separate encoders: image encoder (ViT or ResNet) and text encoder (Transformer). Trained jointly to align paired (image, text) embeddings:

```
Image I → ImageEncoder → i_emb ∈ R^d
Text T  → TextEncoder  → t_emb ∈ R^d

Both normalized to unit sphere
```

### Contrastive Loss (InfoNCE)

For a batch of N (image, text) pairs, maximize cosine similarity of matching pairs and minimize for all N²-N non-matching pairs:

```
L = -1/N Σ_i log exp(sim(i_i, t_i)/τ) / Σ_j exp(sim(i_i, t_j)/τ)

where:
  sim(a, b) = a · b / (||a|| ||b||)  # cosine similarity
  τ = temperature parameter (learned)
```

Each image should match its text more than any other text in the batch — and vice versa. With N=32768 (large batch), this is effectively a 32768-class classification.

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)   # [N, d]
    text_embeds  = F.normalize(text_embeds, dim=-1)    # [N, d]
    
    # Cosine similarity matrix [N, N]
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    
    # Labels: diagonal (i-th image matches i-th text)
    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    
    # Symmetric loss: image→text and text→image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

### Zero-Shot Classification with CLIP

Encode class names as text ("a photo of a [cat/dog/car]") and compare image embedding to all class embeddings:

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

classes = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=-1)  # [1, num_classes]
```

**CLIP limitation:** A global CLS embedding loses spatial information — can't localize objects within the image or answer "what is in the top-left corner?"

---

## 4. Fusion Architectures

### 4a. Simple Linear Projection (LLaVA-1 style)

The simplest approach: project ViT patch embeddings directly into LLM token embedding space:

```
image (H×W) → ViT → patch_embeds [N_patches, d_vision]
                   → Linear(d_vision → d_llm)
                   → Concat with text tokens
                   → LLM
```

Training: keep ViT and LLM frozen; train only the projection layer on image-text pairs. Fast and surprisingly effective.

**LLaVA-1.5 improvement:** Replace linear projection with a 2-layer MLP — slightly more capacity to translate between vision and language representations.

```python
class LLaVAProjector(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )
    
    def forward(self, vision_features):
        # vision_features: [B, N_patches, vision_dim]
        return self.proj(vision_features)  # [B, N_patches, text_dim]
```

### 4b. Q-Former Bridge (BLIP-2 style)

A learnable bottleneck of K query tokens that "ask" the image for relevant features:

```
Architecture:
  - K learnable query tokens Q ∈ R^{K×d}
  - Image features F from frozen ViT
  - Cross-attention: Q attends to F
  - Self-attention: Q tokens interact
  - Output: K compressed image tokens → LLM

Q-Former pre-training objectives:
  1. Image-Text Matching (ITM): binary real/fake pair prediction
  2. Image-Text Contrastive (ITC): CLIP-style alignment
  3. Image-grounded Text Generation (ITG): generate text conditioned on Q
```

**Trade-off:** Q-Former has higher training cost (need to pre-train the Q-Former) but produces a more compact image representation (K=32 tokens vs N=256 raw patches).

### 4c. Perceiver Resampler (Flamingo style)

Similar to Q-Former but simpler — learnable query tokens attend to image features via cross-attention, producing a fixed-size output:

```
N_patches image tokens (variable) → Perceiver → K output tokens (fixed)
```

Flamingo interleaves these image token blocks with text tokens in the LLM, using cross-attention gating to control image influence at each layer.

### 4d. Early Fusion (Llama 4, Gemini)

Image patches and text tokens enter the transformer together from layer 1. No separate projection or bridging module — the transformer itself learns to integrate modalities.

```
Text tokens:   [T1, T2, ... , Tn]
Image patches: [I1, I2, ... , Im]
Combined:      [T1, T2, I1, I2, T3, ... , Im, Tn]  → Transformer
```

Requires training from scratch with multimodal data — can't reuse a pretrained text-only backbone. Enables richer cross-modal reasoning at the cost of training complexity.

---

## 5. Native Multimodal LLMs

### GPT-4o — Unified Token Stream

GPT-4o processes text, image, and audio as a unified token stream with a single autoregressive model. Audio is encoded directly from raw waveform (not text transcripts) — this enables prosody, tone, and timing understanding.

**Practical implications:**
- Can understand emotional tone of speech, not just words
- Real-time audio-in, audio-out with ~320ms latency (vs 2-3 second pipeline)
- Vision understanding integrated at all layers (not bolted on at input)

### Gemini 2.5 — Architecture Notes

Gemini processes multiple modalities through dedicated tokenizers that map each modality into the LLM's token space:
- Text: SentencePiece BPE tokenizer
- Images: Compressed patch embeddings from a ViT-based encoder
- Audio: Log-mel spectrogram patches
- Video: Sparse frame sampling + temporal encoding

The 1M token context allows processing ~8-11 hours of audio, ~3000 pages, or ~45 minutes of video.

### Qwen2.5-VL — Dynamic Resolution

A key challenge in VLMs: how to handle images of vastly different resolutions without either padding to a fixed size (wasting compute on padding tokens) or resizing (losing fine-grained detail for high-res images like document scans).

Qwen2.5-VL uses **dynamic resolution** and **native resolution** processing:

```python
# Dynamic: partition image into patches based on its actual resolution
# High-res image (1920×1080): more patches, more tokens
# Low-res image (224×224):   fewer patches, fewer tokens

# NaViT-style: pack variable-length image sequences into batches efficiently
def pack_images(images: list[Image]) -> dict:
    all_patches = []
    patch_counts = []
    for img in images:
        patches = extract_patches(img, patch_size=14)  # variable count
        all_patches.append(patches)
        patch_counts.append(len(patches))
    return {"patches": torch.cat(all_patches), "counts": patch_counts}
```

**M-RoPE (Multi-dimensional RoPE):** Apply separate RoPE axes for x-position, y-position, and temporal position (for video). This gives the model spatial awareness of where each patch is in the image.

---

## 6. Audio Integration

### Whisper — Weakly Supervised ASR

Whisper (Radford et al., 2022) is an encoder-decoder model trained on 680K hours of internet audio with automatically generated transcripts:

**Architecture:**
- Input: log-mel spectrogram (80-channel, 25ms windows, 10ms hop)
- Encoder: ViT-style on 2D spectrogram (sequence of frames)
- Decoder: autoregressive text generation
- Multitask: transcription, translation, language detection, timestamp prediction

**Why it generalizes well:** The 680K training hours cover 96 languages and diverse acoustic conditions. The model is robust to background noise, accents, and recording quality by virtue of the diversity of internet audio.

```python
import whisper

model = whisper.load_model("large-v3")
result = model.transcribe("audio.mp3", language="en", word_timestamps=True)
print(result["text"])

# Word-level timestamps for downstream alignment
for segment in result["segments"]:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

### Neural Audio Codecs (EnCodec / SoundStream)

To enable LLMs to generate audio, discretize audio into tokens:

```
Audio waveform → Neural Encoder → Continuous embeddings
                              → Residual Vector Quantization (RVQ)
                              → Discrete codebook tokens
                              → LLM processes/generates tokens
                              → Neural Decoder → Audio waveform
```

**RVQ (Residual Vector Quantization):** Quantize the audio embedding in stages, with each stage quantizing the residual from the previous:

```
Stage 1: quantize embedding → code_1, residual_1
Stage 2: quantize residual_1 → code_2, residual_2
...
Stage N: quantize residual_{N-1} → code_N
```

Result: each audio frame is represented as N codebook indices. EnCodec at 24kHz uses 8 codebooks of size 1024 → 3-4 bits per sample.

**Applications:** MusicGen (Meta), AudioPaLM, GPT-4o audio — all build on discrete audio tokens that LLMs can process autoregressively.

---

## 7. Video Models

### The Video Challenge

Video is simply high-dimensional image sequences, but the compute cost is prohibitive:
- 1 hour of 30fps 1080p video = 108,000 frames
- Each frame at 256 patches = 27.6M visual tokens
- Full attention: O((27.6M)²) — intractable

Three strategies for tractable video understanding:

### Strategy 1 — Sparse Frame Sampling

Sample 1-4 frames per second from the video and process independently:

```python
def sample_frames(video_path: str, fps: float = 1.0, max_frames: int = 32):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += frame_interval
    
    return frames
```

**Limitation:** Misses fast motion, temporal dynamics. Good for "what's in this video" but not "what happened."

### Strategy 2 — Factorized Spatiotemporal Attention

Apply spatial attention within each frame and temporal attention across frames separately:

```
TimeSformer architecture:
  Frame t: spatial attention across all patches of frame t
  Patch p: temporal attention across all frames for patch p
  
  Cost: O(N_patches × T²) + O(T × N_patches²)
        vs O((N_patches × T)²) for full attention
```

**ViViT (Video Vision Transformer)** uses a similar approach with factorized encoders for spatial and temporal dimensions.

### Strategy 3 — Video Diffusion (Sora / DiT on Spacetime Patches)

Sora (OpenAI, 2024) treats videos as sequences of **spacetime patches** — 3D patches (height × width × time) rather than 2D spatial patches:

```
Video: T frames × H × W
Spacetime patch: t_patch × h_patch × w_patch
Number of patches: (T/t_patch) × (H/h_patch) × (W/w_patch)
```

These spacetime patches are tokenized and fed into a DiT (Diffusion Transformer) for generation. This handles variable duration, frame rate, and resolution naturally — just different numbers of patches.

**The consistency achievement:** Sora generates coherent physics (objects don't teleport, lighting is consistent across frames) by modeling the full spacetime patch sequence jointly. This is harder for frame-by-frame methods.

---

## 8. Production Deployment

### Latency Breakdown for a VLM API

Typical latency for "image + question → answer" request (GPT-4o scale):

```
Image preprocessing (resize, normalize):      ~10ms
Image encoding (ViT forward pass):            ~50ms
Token concatenation (image + text tokens):    ~5ms
LLM prefill (process all input tokens):       ~200ms (depends on length)
LLM decode (generate response tokens):        ~500ms (for 100 token response)
─────────────────────────────────────────────────────
Total:                                        ~765ms
```

**Main optimization targets:**
1. **Image resolution reduction:** 448px vs 1024px for input — 5× fewer patches, faster encoding and prefill
2. **Image caching:** Cache ViT features for duplicate or similar images (useful for document QA)
3. **Speculative decoding:** Draft tokens for the generation phase
4. **Batch multiple images:** Process batch of similar-size images together in ViT

### Token Budget Management

Multimodal inputs consume LLM context tokens:

| Image Size | Tokens (ViT-L/14) | Tokens (compressed) |
|-----------|-------------------|---------------------|
| 224×224 | 256 patches | 64-256 |
| 512×512 | 1369 patches | 256-512 |
| 1024×1024 | 5476 patches | 512-1024 |

High-resolution images can consume thousands of tokens — with 128K context, a 1024px image leaves ~127K tokens for text but costs significantly more per request.

**Resolution routing:** Route low-complexity queries (simple VQA) to lower resolution (cheaper); route OCR/document analysis to higher resolution (necessary).

### Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Hallucinated objects | Model "fills in" what's plausible from training | Ground with OCR/detection pre-pass |
| Spatial confusion ("top-left vs bottom-right") | Positional RoPE may be weak for vision | M-RoPE, spatial coordinates in prompt |
| Counting errors | Attention doesn't explicitly count | Extract bounding boxes first |
| Fine text illegible | Low input resolution | Route to higher-res pipeline |
| Multi-image confusion | Long context, images interleaved | Explicit image tags and references |

---

## 9. Interview Questions

**Q: What is the trade-off between late fusion (LLaVA-style) and early fusion (Gemini-style) for multimodal models?**

Late fusion (frozen encoders + projection layer) is cheap to train — you reuse pretrained checkpoints and only train a small adapter. The downside is shallow integration: the LLM only sees image features as additional context tokens; the model never jointly learns vision-language representations from layer 1. Early fusion (unified token stream from layer 1) allows deeper cross-modal integration — the model can learn representations that are inherently multimodal. But it requires training from scratch on large multimodal datasets; you can't bootstrap from a pretrained text model.

For production systems with limited compute budgets, late fusion is the practical starting point. Native multimodal (early fusion) is the research direction for frontier capability.

---

**Q: How does CLIP enable zero-shot classification?**

CLIP is trained to maximize the cosine similarity between matched (image, text) pairs and minimize it for mismatched pairs using InfoNCE loss with a large batch. At inference, you convert class labels into text prompts ("a photo of a cat"), encode all class prompts with the text encoder, and encode the image with the image encoder. The class whose text embedding has highest cosine similarity with the image embedding is the predicted class — no task-specific training required. The key is that CLIP's embedding space maps semantically similar images and texts to nearby points, so "a photo of a cat" (text) and an actual cat photo (image) end up near each other.

---

**Q: Why is audio handled differently in GPT-4o vs. a pipeline approach (Whisper + text LLM)?**

Pipeline approach: Whisper transcribes audio → text → LLM processes text. This loses all non-lexical audio information: tone, emotion, speaking pace, background noise, laughter. The LLM has no access to anything beyond the words.

GPT-4o processes audio natively as audio tokens (from an audio encoder, not a transcript). The model sees the actual acoustic signal, enabling:
- Emotion and sentiment understanding from prosody
- Humor recognition (timing matters)
- Background context (is the speaker in a car? whispering?)
- Real-time conversation flow without transcript delay

The trade-off: native audio requires training on paired audio-language data; the pipeline approach works with separate ASR + language models (modular, easier to maintain).

---

**Q: What is the lost-in-the-middle problem and how does it affect multimodal models?**

Liu et al. showed LLMs have a U-shaped recall curve over long contexts — information at the beginning and end is recalled better than information in the middle. For multimodal models with long contexts (e.g., multiple images + long text), this means images placed in the middle of the context may be underutilized.

Practical implications: place the most important image(s) at the beginning or end of the context window. For document VQA with many pages, rather than concatenating all page images linearly, consider retrieval to identify the most relevant pages and place them at prominent positions.

---

*Last updated: May 2026 | Coverage: CLIP through Sora, production deployment | Focus: architecture depth + interview readiness*
