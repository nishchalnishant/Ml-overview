---
module: Llms
topic: Applications
subtopic: Multimodal
status: unread
tags: [llms, ml, applications-multimodal]
---
# Multimodal AI

---

## The Core Problem

**The problem:** a language model processes tokens — discrete symbols from a fixed vocabulary. Images, audio, and video are continuous signals in completely different representational spaces. An image patch is not a word; a spectrogram frame is not a sentence. The model has no native way to process these signals, and a pipeline that converts them to text first (OCR for images, ASR for audio) loses all non-lexical information: spatial layout, tone, emotion, timing, visual relationships.

**The core insight:** the solution is to turn non-text modalities into sequences of vectors in the same embedding space the language model already operates in. The approach has evolved through three generations: freeze the language model and learn only an adapter (cheap but shallow), co-train vision and language together (better integration), and process all modalities as a unified token stream from the first layer (deepest integration, highest training cost).

---

## Vision Encoders

### ViT (Vision Transformer)

**The problem:** attention operates on sequences of vectors. An image is a 2D grid of pixel values. To process an image with a transformer, it must be converted into a sequence of vectors — and the conversion should preserve semantically meaningful structure.

**The core insight:** divide the image into non-overlapping patches, flatten each patch into a vector, and project it to the model's embedding dimension. This turns an image into a sequence of patch embeddings in exactly the same format as text token embeddings, making it compatible with any transformer.

**The mechanics:**
```
Input image: H × W × C
Patch size: P × P → N = (H/P) × (W/P) patches

Each patch → flatten to vector of size P²·C
           → linear projection to d_model
           → add position embedding

Prepend [CLS] token → standard transformer → CLS output = image representation
```

Common configurations: ViT-B/16 (12 layers, 768d, 16×16 patches, 86M params), ViT-L/14 (24 layers, 1024d, 14×14 patches, 307M params).

**What breaks:** ViT produces a global CLS embedding and a sequence of patch embeddings. The CLS embedding loses spatial information — it cannot answer "what is in the top-left corner?" The patch embeddings retain spatial structure, but their positional encoding is learned at a fixed resolution and degrades at other resolutions unless explicitly handled.

**DINOv2 (Meta, 2023):** self-supervised ViT trained via self-distillation. Key property for multimodal models: DINOv2 features are more spatially aligned than CLIP features — they encode where things are, not just what they are. Better for dense tasks like OCR, object localization, and document understanding inside a VLM.

---

### CLIP — Contrastive Image-Language Pretraining

**The problem:** a ViT trained on ImageNet classification knows about ImageNet labels. Multimodal models need an image encoder that understands images in terms of natural language concepts — the same semantic space as text. Classification-trained encoders are too narrow; they lack the open-vocabulary alignment that text-image reasoning requires.

**The core insight:** train two encoders — one for images, one for text — jointly using contrastive loss on matched (image, text) pairs from the internet. Pull matched pairs together in embedding space; push mismatched pairs apart. Both encoders learn to map semantically related content to nearby vectors, regardless of whether the similarity was expressed as an image or as text.

**The mechanics (InfoNCE loss):**

For a batch of N (image, text) pairs, each image should be more similar to its matching text than to all other N-1 texts in the batch, and vice versa:
```
L = -1/N Σᵢ log exp(sim(iᵢ, tᵢ)/τ) / Σⱼ exp(sim(iᵢ, tⱼ)/τ)
```
where sim is cosine similarity and τ is a learned temperature. With batch size N=32768, this is effectively a 32768-class classification at each step — a strong training signal.

Zero-shot classification: encode class names as text ("a photo of a cat"), encode the image, find the class with highest cosine similarity. No task-specific training required.

**What breaks:** CLIP's global CLS embedding cannot localize within an image. A CLIP embedding for "a photo with a cat in the bottom-left corner" and "a photo with a cat in the top-right corner" may be nearly identical — spatial detail is averaged out. This limits CLIP for tasks requiring spatial precision (counting, reading text, detecting objects at specific locations).

---

## Fusion Architectures

**The problem:** even with a strong image encoder, how does the image signal enter the language model? The answer determines the depth of cross-modal integration and the cost of training.

### Linear Projection (LLaVA-1 style)

**The core insight:** the simplest possible bridge. Project ViT patch embeddings linearly into the LLM's token embedding dimension, then concatenate them with text tokens. The LLM sees image patches and text tokens in the same sequence — no architectural change required.

**The mechanics:**
```
image → ViT → patch_embeds [N_patches, d_vision]
            → Linear(d_vision → d_llm)
            → concat with text tokens
            → LLM
```
Training: freeze ViT and LLM; train only the projection layer on image-text pairs. LLaVA-1.5 replaces the linear projection with a 2-layer MLP for more expressive translation between vision and language representations.

**What breaks:** the LLM receives one token per image patch. At 14×14 patches for a 224×224 image, that is 256 extra tokens. For larger images (512×512), it is 1369 extra tokens — a significant context window cost.

---

### Q-Former Bridge (BLIP-2 style)

**The problem:** passing all N patch embeddings to the LLM is expensive and may include visual detail the LLM cannot use. Can the relevant visual information be compressed?

**The core insight:** use K learnable query tokens that "query" the image for relevant information via cross-attention, reducing N variable-length patch embeddings to K fixed-length output tokens. The Q-Former learns to select what visual information matters for language generation.

**The mechanics:**
```
K learnable query tokens Q ∈ R^{K×d}
Image features F from frozen ViT
Cross-attention: Q attends to F
Self-attention: Q tokens interact
Output: K compressed image tokens → LLM
```
K is typically 32 — far fewer than the 256+ raw patch embeddings. This reduces the LLM's context cost by 8× at the price of training the Q-Former.

**What breaks:** the Q-Former adds training complexity: it must be pretrained on image-text matching, contrastive, and generation objectives before being plugged into the LLM. The compression is lossy — fine-grained spatial detail may not survive the bottleneck.

---

### Perceiver Resampler (Flamingo style)

**The core insight:** similar to Q-Former but simpler. Learnable query tokens attend to image features via cross-attention, producing fixed-size output. Flamingo interleaves these image token blocks with text in the LLM, using cross-attention gating layers to control image influence at each transformer layer rather than only at the input.

**What breaks:** interleaved cross-attention at every layer requires architectural modification of the LLM — Flamingo cannot be built on top of a frozen pretrained LLM without adding new cross-attention modules and training them.

---

### Early Fusion (Gemini, Llama 4)

**The problem:** all of the above are "late fusion" — image and text representations are computed separately and combined at the LLM's input. The LLM never sees raw image information; it only sees already-encoded representations. Deeper cross-modal integration requires joint processing from the first layer.

**The core insight:** treat image patches and text tokens as entries in a single unified token stream, processed by the same transformer from layer 1. The transformer itself learns to integrate modalities — no separate encoder or bridging module needed.

**The mechanics:**
```
Text tokens:    [T1, T2, ..., Tn]
Image patches:  [I1, I2, ..., Im]
Combined:       [T1, T2, I1, I2, T3, ..., Im, Tn] → Transformer
```

**What breaks:** requires training from scratch on multimodal data. Cannot reuse a pretrained text-only backbone without significant modification. This is the research frontier for frontier-class models (GPT-4o, Gemini 2.x, Llama 4); not practical for teams with limited compute budgets.

---

## Native Multimodal LLMs

**The problem:** pipeline approaches (ASR + text LLM for audio; OCR + text LLM for documents) lose information at each modality boundary. Speech has tone, emotion, and timing that transcription discards. Images have spatial relationships that caption summaries discard. Video has temporal dynamics that individual frames discard.

**The core insight:** process each modality as tokens directly, without a pipeline step that reduces it to text first. The model has access to the full information in the original signal.

**GPT-4o:** unified token stream for text, image, and audio. Audio is encoded from raw waveform — not a transcript — enabling tone, emotion, and timing understanding. Real-time audio-in, audio-out with ~320ms latency vs 2–3 seconds for a Whisper + text LLM pipeline.

**Gemini:** dedicated tokenizers per modality (SentencePiece for text, ViT-based patches for images, log-mel spectrogram patches for audio, sparse frame sampling + temporal encoding for video). The 1M token context window enables processing ~8–11 hours of audio, ~3000 pages, or ~45 minutes of video.

**Qwen2.5-VL — dynamic resolution:** padding images to a fixed resolution wastes compute on padding tokens; resizing loses fine-grained detail (important for document OCR). Dynamic resolution processes images at their native resolution by extracting a variable number of patches proportional to image size. M-RoPE (Multi-dimensional RoPE) applies separate positional encoding axes for x-position, y-position, and temporal position, giving the model explicit spatial awareness.

**What breaks:** native multimodal training requires paired multimodal data at scale — audio-text pairs, image-text pairs, video-text pairs. These are harder to obtain and verify than text-only data. The model can still hallucinate visual content it did not see, because it generates text autoregressively conditioned on image features that may not fully constrain the output.

---

## Audio Integration

**The problem:** audio is a 1D continuous signal sampled at 16–44kHz. A 10-second clip is 160,000–440,000 samples — far too many to process token by token. The challenge is converting audio into a compact discrete representation that a language model can process autoregressively.

### Whisper — Weakly Supervised ASR

**The core insight:** convert audio to a log-mel spectrogram (a 2D time-frequency representation), then treat it as an image — process with a ViT-style encoder, generate text with an autoregressive decoder.

**The mechanics:**
- Input: log-mel spectrogram (80-channel, 25ms windows, 10ms hop)
- Encoder: convolutional layers + transformer on spectrogram frames
- Decoder: autoregressive text generation
- Training: 680K hours of internet audio with automatically generated transcripts

Trained on 96 languages and diverse acoustic conditions. Robust to background noise and accents by virtue of training data diversity.

**What breaks:** Whisper produces a transcript. All non-lexical information — tone, emotion, prosody, background context — is discarded. The LLM downstream sees only words, not the acoustic signal.

### Neural Audio Codecs (EnCodec / SoundStream)

**The problem:** to enable LLMs to generate audio (not just transcribe it), audio must be discretized into tokens the model can predict autoregressively.

**The core insight:** train a neural encoder-decoder (audio codec) that compresses audio into sequences of discrete codebook indices. The encoder produces continuous embeddings; Residual Vector Quantization (RVQ) maps them to discrete codes; the decoder reconstructs audio from those codes.

**The mechanics (RVQ):**
```
Stage 1: quantize embedding → code_1, residual_1
Stage 2: quantize residual_1 → code_2, residual_2
...
Stage N: quantize residual_{N-1} → code_N
```
Each audio frame becomes N codebook indices. EnCodec at 24kHz uses 8 codebooks of size 1024 — 3–4 bits per sample. An LLM can then predict these discrete codes autoregressively to generate audio. MusicGen (Meta), AudioPaLM, and GPT-4o's audio generation all build on this idea.

**What breaks:** the codec introduces lossy compression. Low-bitrate codecs (few codebooks) lose detail in high-frequency content. RVQ codes are not as interpretable as text tokens — the LLM must learn their semantics from scratch.

---

## Video Models

**The problem:** full video attention is intractable. One hour of 30fps 1080p video is 108,000 frames; at 256 patches per frame, that is 27.6 million visual tokens per hour. Full self-attention over 27.6M tokens is impossible on current hardware.

### Sparse Frame Sampling

**The core insight:** for understanding video content ("what happens in this clip"), most semantic information is carried by a small number of representative frames, not every frame. Sample 1–4 frames per second and process them independently or as a sequence.

**What breaks:** misses fast motion, action transitions, and temporal dynamics. Appropriate for "what is in this video" but not for "what sequence of events occurred" or "how does this object move."

### Factorized Spatiotemporal Attention

**The core insight:** separate spatial attention (within each frame) from temporal attention (across frames for the same patch). Running them independently costs O(N_patches × T²) + O(T × N_patches²) instead of O((N_patches × T)²) — a substantial reduction for long videos.

**The mechanics (TimeSformer / ViViT):**
```
For frame t: spatial attention across all N_patches of frame t
For patch p: temporal attention across all T frames for patch p
```

**What breaks:** factorization approximates joint spatiotemporal attention. Interactions that are simultaneously spatial and temporal (e.g., a fast-moving object crossing frame boundaries) may not be captured as well as with full joint attention.

### Spacetime Patches (Sora / DiT approach)

**The core insight:** extend the 2D ViT patch to 3D: a spacetime patch covers height × width × time simultaneously. A video is then a sequence of spacetime patches, processed by a Diffusion Transformer (DiT). Variable duration, frame rate, and resolution map to different numbers of patches.

**The mechanics:**
```
Video: T frames × H × W
Spacetime patch: t_patch × h_patch × w_patch
Patches: (T/t_patch) × (H/h_patch) × (W/w_patch)
```

Modeling the full spacetime patch sequence jointly enables coherent temporal consistency — objects do not teleport, lighting is consistent across frames — because the model attends to all patches across space and time simultaneously.

**What breaks:** compute scales with the number of spacetime patches, which grows with resolution and duration. Sora-scale video generation requires orders of magnitude more compute than image generation.

---

## Production Deployment

**The problem:** a VLM API request has a different latency profile than a text-only LLM request. Image encoding and the increased context length from image tokens add overhead at every stage.

**The core insight:** multimodal latency has two components that text-only systems do not: image preprocessing + encoding time, and the increased prefill cost from image tokens. Optimizing these requires different levers than optimizing decode speed.

**Typical latency breakdown for a VLM request:**
```
Image preprocessing (resize, normalize):      ~10ms
Image encoding (ViT forward pass):            ~50ms
Token concatenation (image + text tokens):    ~5ms
LLM prefill (all input tokens):               ~200ms
LLM decode (100-token response):              ~500ms
```

Optimization targets: reduce image resolution (448px vs 1024px gives 5× fewer patches), cache ViT features for repeated images (document QA), route low-complexity queries to lower resolution.

**Token budget:** at ViT-L/14, a 1024×1024 image produces 5476 patches — thousands of LLM context tokens. For document analysis with multiple pages, token budget is the primary constraint. Resolution routing (send simple VQA to low-res; send OCR/document tasks to high-res) manages this tradeoff.

**Common failure modes:**

| Failure | Cause | Mitigation |
|:---|:---|:---|
| Hallucinated objects | Model "fills in" plausible content | Ground with OCR/detection pre-pass |
| Spatial confusion | Positional encoding weak for vision | M-RoPE, explicit spatial coordinates in prompt |
| Counting errors | Attention does not count explicitly | Extract bounding boxes first |
| Fine text illegible | Low input resolution | Route to higher-resolution pipeline |
| Lost-in-the-middle | U-shaped attention over long context | Place most relevant images first or last |

*Related: [Context Window Extension](context-window-extension.md) | [Hallucination Mitigation](hallucination-mitigation.md) | [Inference Optimization](inference-optimization.md)*

## Flashcards

**Input?** #flashcard
log-mel spectrogram (80-channel, 25ms windows, 10ms hop)

**Encoder?** #flashcard
convolutional layers + transformer on spectrogram frames

**Decoder?** #flashcard
autoregressive text generation

**Training?** #flashcard
680K hours of internet audio with automatically generated transcripts
