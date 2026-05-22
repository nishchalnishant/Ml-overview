# Multimodal Architectures

How frontier models unify vision, audio, video, and language into a single representational space — and why this requires fundamentally rethinking tokenization, pretraining, and the attention mechanism.

---

## 1. Core Concept & Intuition

The central insight: intelligence is grounded in multiple sensory modalities simultaneously. A human who reads "the glass shattered" has immediate access to sound (shattering), visual memory (broken shards), tactile sensation (sharp), and semantic meaning — all from three words. Current language-only models have the semantic meaning but none of the grounding.

**What older paradigms couldn't do:**

Bolt-on multimodality (CLIP + LLM) treats modalities as separate pipelines joined by a projection layer. The fundamental limitation: the projection must bridge two representation spaces that were independently optimized. The LLM never "sees" an image during pretraining — it sees text descriptions of images. The gap between "a red apple" (text) and the actual pixel representation of a red apple cannot be fully closed by a linear projection trained after the fact.

Native multimodality trains all modalities jointly from token 1. The model develops representations that are inherently grounded — "red" in the attention layers means the same thing whether it came from a pixel or a character.

```
Bolt-on pipeline:
  Image → ViT → [CLS token] → Linear(768→4096) → "pretend text" → LLM
  (ViT and LLM were never co-trained; projection is a patch over a fundamental mismatch)

Native pipeline:
  [img_patch_1, img_patch_2, ..., text_tok_1, text_tok_2, ...] → Transformer
  (same self-attention across all tokens; no modality boundary in the attention graph)
```

---

## 2. Architecture & Mathematics

### 2.1 Visual Tokenization

**ViT tokenization (standard):** Divide the image into P×P patches (typically 16×16 or 14×14 pixels). Project each patch to d_model dimensions via a linear layer. Prepend [CLS] token.

```
Image H×W×3 → reshape to (H/P)×(W/P) patches → each patch: P²×3 → Linear(P²·3, d_model)
Number of visual tokens: N_v = (H/P) × (W/P)
For 224×224 image with P=16: N_v = 14×14 = 196 tokens
For 1024×1024 image with P=16: N_v = 64×64 = 4096 tokens
```

**Resolution problem:** High-resolution images → thousands of visual tokens → quadratic attention cost. Solutions:

1. **Dynamic resolution (LLaVA-UHD, InternVL):** Tile the image into sub-images at native resolution, process each tile separately, then concatenate with a global downsampled view.

2. **Perceiver Resampler (Flamingo):** Cross-attention from a fixed set of Q learned query vectors to all N_v image tokens. Outputs exactly Q tokens regardless of image resolution.
   ```
   Q_learned: [Q × d_model] (fixed, Q=64 or 256)
   K, V: [N_v × d_model] (from ViT encoder)
   Output: Cross-Attention(Q_learned, K, V) → [Q × d_model]
   ```
   Reduces visual tokens from O(N_v) to O(Q) at the cost of some information compression.

3. **Tokenizer unification (Chameleon, Llama 4):** Discretize image patches into codebook indices via VQ-VAE. Images become sequences of integer tokens — identical format to text. The model treats them identically in self-attention.

### 2.2 VQ-VAE for Image Tokenization

Vector Quantization maps continuous patch embeddings to discrete codebook entries:

```
Encoder:  z_e = Encoder(x)       # continuous embedding
Quantize: z_q = e_k  where k = argmin_j ||z_e - e_j||₂   # nearest codebook vector
Decoder:  x̂ = Decoder(z_q)

Loss: L = ||x - x̂||² + ||sg[z_e] - z_q||² + β·||z_e - sg[z_q]||²
     reconstruction   codebook update         commitment loss
```

`sg` = stop gradient. The commitment loss (β typically 0.25) prevents the encoder from growing its outputs unboundedly to chase codebook vectors. The codebook is updated via exponential moving average of the encoder outputs that map to each entry.

**Why discrete tokens for images:** Enables unified autoregressive modeling. The next-token prediction loss `L = -Σ log P(x_t | x_{<t})` works identically over text and image tokens. No special image-specific loss required.

### 2.3 Cross-Modal Attention (Flamingo Architecture)

Flamingo freezes a large language model and inserts trainable cross-attention layers that condition language generation on visual features:

```
For each transformer layer l:
  if l is a cross-attention layer:
    h_l = h_{l-1} + Attention(Q=h_{l-1}·W_Q, K=v·W_K, V=v·W_V)
    where v = visual features from Perceiver Resampler
  else:
    h_l = standard_transformer_layer(h_{l-1})
```

Only the cross-attention layers and Perceiver Resampler are trained; the LLM weights are frozen. This is efficient (few trainable parameters) but limits how deeply language and vision representations integrate.

### 2.4 Contrastive Pretraining (CLIP)

CLIP trains a vision encoder and text encoder such that matching image-text pairs have high cosine similarity:

```
For a batch of N (image, text) pairs:
  v_i = VisionEncoder(image_i) / ||VisionEncoder(image_i)||
  t_i = TextEncoder(text_i) / ||TextEncoder(text_i)||
  
  Similarity matrix: S = [N × N] where S_{ij} = v_i · t_j / τ
  (τ = temperature, typically 0.07)
  
  L_CLIP = -1/(2N) · Σ_i [log(softmax(S_i·)[i]) + log(softmax(S·i)[i])]
  (symmetric cross-entropy: each image-text pair is positive; all others in batch are negatives)
```

**Why CLIP representations are powerful:** The diagonal of S contains the true pairs. With batch size N=32768 (as in original CLIP), there are 32767 negatives per positive. The model must learn very fine-grained discriminative features to correctly rank the true pair. This forces visual representations to align with semantic language structure.

**CLIP's limitation for generation:** CLIP representations are discriminative (good for retrieval, zero-shot classification). They compress images into global [CLS] embeddings that discard spatial structure. Generative VLMs need patch-level representations to answer spatial questions ("what is in the top-left corner?").

### 2.5 LLaVA Architecture (Standard Production VLM)

```
Image → CLIP ViT → [N_v × d_visual] → MLP projection → [N_v × d_model]
Text → Tokenizer → [N_t × d_model]
[visual_tokens, text_tokens] → LLM → autoregressive text output

Training:
  Stage 1: Train only MLP projection, freeze ViT and LLM
           Loss: L = -Σ log P(text_t | visual_tokens, text_{<t})
  Stage 2: Train MLP + LLM; freeze ViT
           Same loss on richer instruction-following data
```

The two-stage approach prevents catastrophic forgetting of LLM capabilities during visual adaptation.

### 2.6 iRoPE (Llama 4's Interleaved RoPE)

Rotary Position Encoding (RoPE) embeds 1D sequence position into attention via rotation:

```
RoPE: q_m = q · R_m  where R_m rotates by angle m·θ_d for each head dimension d
Attention score: (q_m)ᵀ k_n = qᵀ R_{m-n} k  (only relative position m-n matters)
```

For 2D images, naively flattening patches into 1D sequence loses spatial structure. iRoPE alternates:

- **RoPE layers:** Apply standard 1D RoPE; image patches get sequential positions. These layers handle local sequential relationships.
- **Non-RoPE (NoPE) layers:** No positional encoding at all. Attention is position-agnostic. Image patches can attend to each other based purely on content, regardless of where they appear in the sequence.

NoPE layers implicitly learn 2D spatial attention because the content of a patch (edges, colors) is informative about spatial relationships — without needing explicit position encoding.

---

### 2.7 Audio as a Modality

**Whisper-style processing:** Raw audio waveform → mel spectrogram (frequency-time representation) → 2D image → ViT-style patch encoding → transformer.

```
Audio 30s at 16kHz → 480,000 samples
→ Mel spectrogram: 80 mel bins × 3000 time frames
→ 2D array treated as image patches
→ Two 1D Conv layers to subsample to 1500 tokens
→ Transformer encoder
```

**Native audio (GPT-4o style):** Audio codecs (EnCodec, SoundStream) discretize the waveform directly into residual vector quantization (RVQ) codes — similar to VQ-VAE but with multiple codebooks in sequence (residual quantization):

```
RVQ(x): 
  q_1 = nearest codebook entry for x
  r_1 = x - q_1  (residual)
  q_2 = nearest codebook entry for r_1
  r_2 = r_1 - q_2
  ...repeat for K codebooks...
  x ≈ q_1 + q_2 + ... + q_K
```

With K=8 codebooks at 75 Hz, 1 second of audio = 600 discrete tokens. These interleave with text tokens in a unified transformer — enabling simultaneous audio understanding and generation without any modality-specific preprocessing.

---

### 2.8 Video Generation: Diffusion Transformers (DiT) and Sora

**Core insight of video generation:** Video = 3D tensor (T × H × W × C). Temporal consistency requires modeling correlations across the time dimension, not just spatial.

**Diffusion model objective:**
```
Forward process: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε,  ε ~ N(0, I)
Reverse process: x_{t-1} = μ_θ(x_t, t) + σ_t·z
Training loss: L = E[||ε - ε_θ(x_t, t)||²]
```

The noise prediction network ε_θ predicts the noise added to create x_t from x_0.

**Latent Diffusion:** Run diffusion in the latent space of a VAE, not pixel space. 8× spatial compression (e.g., 512×512 → 64×64 latents) makes diffusion computationally tractable.

**Patchification for Video (Sora):**
```
Video T×H×W×C → spatial-temporal patches of size (t, p, p)
Number of patches: (T/t) × (H/p) × (W/p)
Each patch → linear projection → d_model token
All patches → Transformer (DiT) with full 3D attention
```

**3D attention** allows every spatial-temporal patch to attend to every other — capturing that a moving object at frame t must be consistent with its position at frame t+1. This is computationally expensive (O(N²) where N = all patches across all frames) but produces temporally coherent video.

**Sora's key contributions:**
1. Train on videos of any resolution, aspect ratio, and duration by treating all as sequences of spacetime patches.
2. Use a transformer (not a U-Net) as the denoising backbone — DiT scales better with compute.
3. Condition on text embeddings from T5/CLIP for semantic control.

**World Model framing:** If you train a video generation model on enough diverse video of physical systems, it must implicitly learn a model of physics (gravity, collision, causality) to generate plausible videos. Sora and similar models show emergent physical reasoning — objects cast consistent shadows, liquids flow correctly — suggesting video generation learns an implicit world model, not just visual statistics.

---

## 3. Trade-offs & System Design Implications

### Visual Token Count vs. Quality

| Resolution | Patch Size | Visual Tokens | Context Used | Notes |
|---|---|---|---|---|
| 224×224 | 16 | 196 | ~1.5K ctx | Low res; standard CLIP pretraining |
| 448×448 | 14 | 1024 | ~8K ctx | Good for detail |
| 1024×1024 | 14 | 5329 | ~40K ctx | High res; needs long context model |
| Dynamic tiling (4 tiles) | 14 | 4×1024 | ~32K ctx | Best quality for documents/charts |

Rule: for OCR, charts, or fine-grained visual questions — use high resolution with dynamic tiling. For general image Q&A — 336×336 is sufficient.

### Frozen vs. Trainable Components

| Component | Frozen | Trainable | Trade-off |
|---|---|---|---|
| ViT encoder | Most production VLMs | Domain-specific tuning | Frozen: avoids catastrophic forgetting; trainable: better domain adaptation |
| Projection layer | Never | Always | The bridge must be trained |
| LLM | Flamingo, frozen-LLM VLMs | LLaVA, InstructBLIP | Frozen: preserves text ability; trainable: better multimodal reasoning |

### When to Choose Native Multimodal vs Bolt-On

**Use bolt-on (CLIP + LLM) when:**
- Deploying quickly with an existing LLM
- Images are simple (product photos, charts with clear structure)
- Budget is limited — no pretraining from scratch

**Use native multimodal when:**
- Tasks require reasoning that combines modalities mid-answer ("the diagram on the left shows X, which contradicts the equation on the right")
- Audio or video is required — bolt-on doesn't generalize beyond vision
- Production scale where quality differences matter

### Latency Model for VLM Inference

```
Total latency = visual encoding time + projection time + LLM prefill time + LLM decode time

Visual encoding: O(N_v²) for ViT attention over patches
LLM prefill: O((N_v + N_text)² · L) for L layers
LLM decode: O((N_v + N_text + N_gen) · d · L) per generated token

At 1024 visual tokens + 256 text tokens + 512 generated tokens:
- Prefill dominates: 1280² tokens = 1.6M attention ops
- Each decode step must attend to 1280 key-value pairs + growing cache
```

**Optimization:** Cache visual KV entries (they don't change during generation). All decode steps reuse the same visual KV — only text KV grows. Visual KV caching is a significant throughput improvement for VLMs.

---

## 4. Canonical Interview Q&As

**Q1: Why does CLIP produce representations that work well for zero-shot classification but poorly for fine-grained spatial tasks like "count the objects in the left quadrant"?**

CLIP's contrastive objective trains a global image-text similarity. The image is compressed into a single [CLS] embedding — a 768- or 1024-dimensional vector that captures the overall semantic content but discards spatial arrangement. "A dog on the left and a cat on the right" and "a cat on the left and a dog on the right" may have identical or very similar [CLS] embeddings because the presence of a dog and a cat is the dominant signal. For zero-shot classification, global semantics is sufficient — "this is a German Shepherd" doesn't require spatial decomposition. For spatial tasks, you need patch-level representations (individual tokens per patch) and attention between patches that preserves position. This is why LLaVA uses the intermediate patch token representations from CLIP ViT, not the [CLS] token — the 196 patch tokens retain spatial structure. For truly high-resolution spatial tasks, you need either higher patch resolution (more tokens) or dynamic tiling (multiple crops processed independently then combined).

**Q2: Compare the training objectives of CLIP, BLIP-2, and LLaVA. What does each optimize for, and what downstream capability does each training objective produce?**

CLIP: contrastive loss on (image, text) pairs from web data. Optimizes global image-text alignment. Produces: strong visual features, zero-shot classification, image-text retrieval. Weakness: no generation ability; features are discriminative not generative.

BLIP-2: three losses simultaneously — (1) image-text contrastive (like CLIP), (2) image-text matching (binary: does this image match this text?), (3) image-conditioned text generation (autoregressive LM loss). Uses a Q-Former (cross-attention bottleneck) to mediate between a frozen ViT and a frozen LLM. Produces: rich multimodal representations with both discriminative and generative capabilities. Weakness: frozen components limit integrated reasoning.

LLaVA: purely supervised fine-tuning on instruction-following multimodal data with an autoregressive language modeling loss `L = -Σ log P(answer_t | image_tokens, question_tokens, answer_{<t})`. Simple, no contrastive term. Produces: instruction-following VLM that can answer arbitrary questions about images. The quality is determined almost entirely by the quality and diversity of the instruction-tuning dataset. Weakness: the ViT is frozen, so visual features are from CLIP pretraining — no visual specialization to the target task distribution.

**Q3: What is the mathematical relationship between the number of video frames, patch size, and the computational cost of a 3D attention video diffusion transformer? How does Sora mitigate this?**

For a video with T frames, each frame H×W, using spatial patches of size p×p and temporal patches of size t frames:

```
N_patches = (T/t) × (H/p) × (W/p)
3D attention: O(N_patches²) = O((T/t)² × (H/p)² × (W/p)²)
```

For a 30-frame 512×512 video with p=16, t=1: N = 30 × 32 × 32 = 30,720 patches; attention = O(943M) — already expensive. For a 120-frame 1080×1080 video: N = 120 × 67 × 67 ≈ 539K patches; attention = O(290B) — completely intractable.

Sora mitigates via: (1) Latent space — run diffusion in compressed VAE latent space (8× spatial compression → reduces H/p ratio); (2) Temporal patch size t>1 — using t=4 reduces the temporal axis by 4×; (3) Training at variable resolution — smaller training examples reduce average N; (4) At extreme durations, factorized attention: spatial attention within each frame separately, then temporal attention across frames at the same spatial position. This reduces O(N²) to O(T·H²W²/p⁴ + H²W²T²/t²) — still cubic in dimensions but factored.

**Q4: Design a production VLM serving system for a document understanding product (PDF → structured data extraction). What are the key architectural and infrastructure decisions?**

**Model choice:** Use a high-resolution VLM with dynamic tiling (InternVL2 or LLaVA-UHD style). Documents require 1024px+ resolution to resolve small text and table cell boundaries. At 4 tiles × 1024 visual tokens = ~4096 visual tokens per page — budget for 128K context LLM for multi-page documents.

**Infrastructure:**
1. **Preprocessing pipeline:** Convert PDF pages to images at 150-300 DPI (rasterize), apply dynamic tiling to identify the optimal grid (2×2, 2×3, etc.) based on aspect ratio.
2. **Visual KV caching:** For multi-turn interactions on the same document, cache the visual KV entries from all pages. A 20-page document processed once; queries are cheap (only text KV added per query).
3. **Batching:** Group requests with similar page counts. Visual encoding (ViT forward pass) is embarrassingly parallel — batch multiple pages simultaneously. LLM prefill is less parallelizable (batch elements have different visual token counts).
4. **Output parsing:** Structured extraction (JSON schema output) — use constrained decoding to enforce schema validity, eliminating post-processing failures.
5. **Quality monitoring:** Track extraction confidence via token-level log probabilities. Low-confidence outputs → flag for human review.

**Latency targets:** Document ingestion (visual encoding for all pages): 2-5s. Per-query extraction: 1-3s. High-resolution visual encoding dominates; use FlashAttention-2 and bf16 for ViT to reduce this.

**Q5: Explain the "world model" hypothesis for video generation models. What evidence supports it and what evidence contradicts it?**

The hypothesis: a model trained to predict the next frame of video must implicitly represent the laws governing the physical world — gravity, collision physics, object permanence, causal chains. The model cannot generate plausible video of a thrown ball without learning parabolic trajectories; cannot generate liquid without learning fluid dynamics. This implicit knowledge constitutes a "world model."

**Evidence for:** Sora generates physically consistent videos — shadows move with objects, liquids spread realistically, rigid objects collide correctly without explicit physics simulation. These behaviors were not programmed; they emerged from video prediction. Genie (Google DeepMind, 2024) trained purely on platformer game videos generates interactive environments — demonstrating that video models learn controllable physics-governed state transitions.

**Evidence against:** Sora fails on specific physical edge cases — it cannot reliably conserve matter (objects appear/disappear mid-video), violates rigid body physics in unusual configurations, and struggles with precise counting over time. These failures suggest the model learned visual statistics that correlate with physics rather than explicit causal physical models. Crucially: the "world model" is implicit in the weights and not directly queryable — you cannot ask a video diffusion model "what is the mass of this object?" Unlike a symbolic physics engine, the model has no explicit representation of physical quantities. Whether this constitutes a "world model" or "very good visual pattern matching" is a definitional question; practically, the model's physics understanding is reliable for common scenarios and brittle for edge cases.
