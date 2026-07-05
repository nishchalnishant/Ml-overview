---
module: Llms
topic: Interview Notes
subtopic: Multi Modal Ai
status: unread
tags: [llms, ml, interview-notes-multi-modal-ai]
---

> _Full study notes. For the quick-recall version, see [multi-modal-ai-snappy.md](multi-modal-ai-snappy.md)._

# Multi-modal AI

---

## The Scenario That Drives Every Topic Here

You have a vision-language model (VLM) for a product catalog assistant. Users upload images and ask questions. The model generates confident, fluent answers — but frequently describes details that aren't in the image. When you check, the model is ignoring the visual tokens and answering from text priors alone.

Why? Because the model's language modeling loss dominates. Unless you specifically train for and enforce visual grounding, the model learns that plausible text answers score well regardless of image content.

Every technique in this file — CLIP, cross-attention fusion, VQA grounding, multimodal RAG, visual entailment checks — exists to close the gap between "has access to an image" and "actually uses the image."

---

## Q1: What are multi-modal AI models, and how do they process different types of data?

### The Problem
A single-modality model can't reason about a question that requires integrating two types of evidence — for example, "does this chart support the claim in this paragraph?" Text-only models have no access to visual structure; image-only models can't parse the claim. You need a model that treats both as first-class inputs.

### The Core Insight
Each modality has its own structure (pixel grids, waveforms, token sequences). You can't naively concatenate them. You need modality-specific encoders that map each input into a common representational space — then reason over the result.

### The Mechanics
```
Input → modality-specific encoder → aligned embedding space → fusion/reasoning → output
```

Pipeline:
- **Preprocess**: resize/crop images, tokenize text, extract log-mel spectrograms for audio
- **Encode**: vision encoder (ViT or CNN) for images; audio encoder (transformer on mel features) for speech; text tokenizer + encoder for language
- **Align**: contrastive training (CLIP-style) maps matching pairs close in embedding space; or cross-attention feeds visual tokens directly into a language decoder
- **Fuse**: early fusion (merge tokens before reasoning) or late fusion (combine decisions after separate processing)
- **Output head**: classification, captioning, generative decoding, or retrieval ranking

```python
img_feat = vision_encoder(image)         # [d]
txt_feat = text_encoder(tokenize(text))  # [d]
score = cosine_similarity(img_feat, txt_feat)
```

### What Breaks
- **OCR/ASR errors propagate**: bad extraction → bad embeddings → bad answers; measure each stage separately
- **Modality imbalance**: if one modality is noisy or absent, the model silently falls back to the other
- **Alignment failures**: embeddings trained on general data may not align for domain-specific content (medical images, specialized diagrams)
- **Cross-modal hallucination**: model generates answers consistent with text priors, not actual image content

### What the Interviewer Is Testing
Whether you understand that multimodal is an engineering problem (alignment, grounding, evaluation per modality), not just a model-selection problem.

### Common Traps
- Saying "just concatenate pixels and tokens" without explaining why encoders + alignment are required
- Treating multimodal systems as black boxes and skipping per-stage evaluation
- Assuming the model uses all modalities equally — in practice it learns shortcuts

---

## Q2: How do vision-language models process images?

### The Problem
LLMs operate on token sequences. Images are 2D pixel grids. You need a way to convert an image into something that fits into the LLM's input sequence without losing spatial structure.

### The Core Insight
Divide the image into fixed-size patches, project each patch into the LLM's embedding dimension, and treat the result as a sequence of "visual tokens." The LLM then attends over both text and visual tokens using its standard self-attention or cross-attention mechanism.

### The Mechanics
```
Image → ViT patches → [N, d_patch] → linear projection → [N, d_model] → LLM input
```

Common architectures:
- **Contrastive alignment (CLIP-style)**: image embedding and text embedding are trained to be close for matching pairs; no generation, only similarity
- **Generative VLM (cross-attention)**: visual tokens are fed via cross-attention into an autoregressive decoder that generates text
- **Token prepending (LLaVA-style)**: visual tokens are prepended to the text token sequence and processed jointly

Training signals:
- Contrastive loss for alignment (image-text matching)
- Language modeling loss for generation tasks (captioning, VQA)
- Instruction tuning with multimodal examples

```python
visual_tokens = vit(image)           # [N, d_patch]
visual_tokens = proj(visual_tokens)  # [N, d_model] — match LLM hidden size
output = llm.generate(input_ids, cross_attention_to=visual_tokens)
```

### What Breaks
- **Small patch size**: captures more detail but increases sequence length and quadratic attention cost
- **OCR-heavy images**: models trained on natural images fail on text-dense documents; need specialized training or explicit OCR preprocessing
- **Visual hallucination**: model generates plausible captions without grounding in actual patch content; measure with visual entailment checks
- **Truncation of visual tokens**: if context window fills up, visual tokens may be dropped silently

### What the Interviewer Is Testing
Whether you can explain the patch/projection pipeline and identify where grounding breaks down vs where architecture works correctly.

### Common Traps
- Describing "the model sees the image" without explaining the projection step
- Ignoring that vision encoder and LLM have different embedding dimensions requiring projection
- Assuming that visual tokens are always attended to — in practice, truncation and positional encoding choices matter

---

## Q3: How does CLIP work, and why is it important for multi-modal AI?

### The Problem
You want to retrieve images using a text query (or vice versa) without manually labeling every image with tags. You need a shared embedding space where "a photo of a dog" is close to an image of a dog, even if you never explicitly trained on that exact phrase.

### The Core Insight
Train two encoders (image and text) contrastively on a massive dataset of image-text pairs. Maximize similarity for matching pairs, minimize it for mismatched pairs. The result is a shared embedding space that transfers to new tasks zero-shot.

### The Mechanics
InfoNCE (contrastive) loss over a batch of B pairs:

```
z_img = normalize(f_img(image))    # [B, d]
z_txt = normalize(f_txt(text))     # [B, d]
logits = z_img @ z_txt.T           # [B, B]
loss = cross_entropy(logits, target=range(B)) + cross_entropy(logits.T, target=range(B))
```

The diagonal of `logits` should be high (correct pairs); off-diagonals should be low (mismatched pairs). With large batch sizes (CLIP uses ~32k), there are many hard negatives per step.

Why it matters:
- Zero-shot classification: compare image embedding to text prompt embeddings ("a photo of a {class}") — no fine-tuning needed
- Cross-modal retrieval: text query → nearest image embeddings
- Backbone for generative VLMs: CLIP's vision encoder provides strong visual features

### What Breaks
- **Prompt sensitivity**: "a photo of a dog" vs "dog" can give meaningfully different similarity scores
- **Fine-grained attributes**: CLIP struggles with counting, precise spatial relationships, and rare attributes not well-covered in training data
- **Distribution shift**: similarity scores are not calibrated; high CLIP score ≠ correct answer for downstream tasks
- **Retrieval ≠ grounded generation**: CLIP can retrieve relevant images but cannot explain or reason about them

### What the Interviewer Is Testing
Understanding of contrastive learning mechanics and the distinction between retrieval capability (CLIP) and generative reasoning capability (VLMs built on CLIP).

### Common Traps
- Treating CLIP similarity scores as ground truth without calibration
- Using CLIP for generation tasks where you need cross-attention and a language decoder
- Ignoring that the text encoder consumes prompts — prompt engineering matters for CLIP just as for LLMs

---

## Q4: What are the key architectures for multi-modal models?

### The Problem
Different multimodal tasks have different requirements: retrieval is fast and index-based, generation requires attending over evidence, and fine-grained reasoning needs tight cross-modal interaction. No single architecture is optimal for all.

### The Core Insight
The key design axis is where modalities meet: before similarity scoring (early alignment), inside the decoder's attention (cross-attention generation), or at the decision level (late fusion). Choose based on task type and latency constraints.

### The Mechanics

**Dual encoder (CLIP-style)**:
- Compute `z_img` and `z_txt` independently; compare for retrieval
- O(1) retrieval against precomputed index
- Cannot produce grounded text explanations

**Generative cross-attention VLM**:
- Visual tokens fed via cross-attention into LLM decoder
- Can produce detailed, grounded text
- Higher cost per query; cannot precompute image representation at query time

**Early fusion**:
- Merge modality tokens early in the network
- Stronger cross-modal interaction
- Higher compute; risk of overfitting when one modality is noisy

**Late fusion**:
- Encode modalities separately; combine scores or embeddings at decision level
- More robust and modular
- Weaker cross-modal interaction for complex reasoning

**Adapter/Q-Former (BLIP-2 style)**:
- Small learned adapter bridges the vision encoder and LLM
- Compresses N visual tokens into K query tokens (K << N)
- Reduces LLM context cost while preserving visual information

```python
# dual encoder — retrieval
score = cosine(encode_img(img), encode_txt(txt))

# generative cross-attention — grounded generation
visual_tokens = vision_encode(img)       # [N, d]
answer = llm.generate(text_prompt, cross_attn=visual_tokens)
```

### What Breaks
- Using a dual encoder when the task requires generative grounded reasoning (retrieval finds the right image, but can't explain it)
- Using cross-attention generation for high-throughput retrieval (too slow per query)
- Early fusion when one modality has high noise rate (corrupts all representations)

### What the Interviewer Is Testing
Whether you can match architecture to task requirements rather than defaulting to "use the latest model."

### Common Traps
- Conflating retrieval performance (CLIP score) with generation performance (answer quality)
- Not knowing that Q-Former/adapter approaches exist to reduce visual token cost
- Describing architecture choices without mentioning latency or cost consequences

---

## Q5: How does image generation work with diffusion models?

### The Problem
Generative models like GANs collapse modes and are hard to train stably. You want a model that generates diverse, high-quality images conditioned on text, with stable training and controllable outputs.

### The Core Insight
Learn to reverse a noise process. The forward process gradually destroys an image into pure noise. The model learns the reverse: given a partially noisy image at timestep t and a text condition, predict the noise to remove. At generation time, start from pure noise and iteratively denoise to produce an image.

### The Mechanics
Forward process (adds noise):
```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps,   eps ~ N(0, I)
```

Reverse process (learned denoising):
```
Train eps_theta(x_t, t, cond) to predict eps
```

Sampling: iterate from t=T down to t=0, applying the predicted noise removal at each step.

**Latent diffusion (Stable Diffusion)**: encode image to latent space with a VAE encoder; denoise in the latent space (much cheaper); decode with VAE decoder. The U-Net uses cross-attention on text embeddings to condition each denoising step.

**Classifier-free guidance (CFG)**: mix conditional and unconditional predictions to amplify prompt adherence:
```
eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
```
Higher CFG scale → stronger prompt adherence, less diversity.

```python
z = randn(latent_shape)
for t in timesteps:
    eps = unet(z, t, cond=text_emb)
    z = denoise_step(z, eps, t)
img = vae_decoder(z)
```

### What Breaks
- **CFG too high**: images follow prompt rigidly but look stylistically stiff and lose diversity
- **Too few steps**: faster but lower quality; need to calibrate quality vs latency on your prompt distribution
- **Prompt ambiguity**: vague prompts produce inconsistent results; precise attribute specification matters
- **Compute cost**: video generation multiplies this by number of frames

### What the Interviewer Is Testing
Understanding of the denoising objective and the role of CFG — distinguishing diffusion from other generative approaches.

### Common Traps
- Comparing diffusion outputs without controlling for steps, CFG scale, and resolution
- Describing diffusion as "just adding and removing noise" without explaining the learned prediction objective
- Not knowing what latent diffusion is or why it matters for compute efficiency

---

## Q6: What is text-to-speech (TTS), and what models are used for it?

### The Problem
You need to convert text to natural-sounding audio — but the mapping from text to sound is not unique (prosody, rhythm, speaker voice) and depends on context the text alone doesn't specify.

### The Core Insight
Decompose the problem: predict intermediate acoustic features (mel-spectrogram) from text, then synthesize the waveform from those features. This allows separate optimization of linguistic accuracy and audio quality.

### The Mechanics
Three-stage pipeline:
1. **Text normalization**: expand numbers, abbreviations, punctuation into speakable form
2. **Acoustic model**: text tokens → mel-spectrogram (or similar acoustic features); transformer or RNN; conditioned on speaker embedding for multi-speaker
3. **Vocoder**: mel-spectrogram → waveform; neural vocoders (HiFi-GAN variants) produce high-quality, real-time-capable audio

For streaming TTS: generate mel chunks progressively and feed to vocoder incrementally to reduce time-to-first-audio.

```python
tokens = text_tokenize(text)
mel = acoustic_model(tokens, speaker_id=spk)
wav = vocoder(mel)
```

### What Breaks
- **Text normalization failures**: "1.5 million" read as "one point five million" vs "one and a half million" depending on normalization
- **Prosody**: flat or unnatural stress because the acoustic model lacks prosody annotation in training data
- **Vocoder mismatch**: acoustic model and vocoder trained on different audio distributions produce artifacts
- **Voice cloning misuse**: speaker embeddings enable impersonation; requires consent and policy controls

### What the Interviewer Is Testing
Whether you understand the full pipeline (not just "the model generates audio") and the failure modes specific to audio generation.

### Common Traps
- Skipping text normalization as an afterthought — it's where most production artifacts originate
- Treating voice cloning as a pure technical feature without mentioning consent and policy requirements
- Not knowing what a vocoder does or why it's separate from the acoustic model

---

## Q7: How does speech-to-text (Whisper) work?

### The Problem
Microphones capture raw audio signals. You need to convert them to text — accurately, across languages, speaker accents, and noise conditions — without requiring domain-specific fine-tuning for each deployment.

### The Core Insight
Treat ASR as a sequence-to-sequence problem: encode the audio as mel spectrograms and decode into text tokens autoregressively. Train on massive multilingual audio-text pairs to build robust priors across conditions.

### The Mechanics
```
Audio waveform → 30s chunks → log-mel spectrogram [80 channels × 3000 timesteps]
    → Transformer encoder → audio embeddings
    → Transformer decoder (cross-attention to audio) → text tokens
```

Key properties:
- Multilingual: single model handles 99+ languages
- Robustness: trained on diverse noisy, accented audio
- Optional timestamp decoding: aligns words to time positions

Production considerations:
- **VAD (Voice Activity Detection)**: detect speech segments before sending to Whisper; don't send silence
- **Chunking**: Whisper processes 30s windows; overlap chunks to avoid word boundary artifacts
- **Diarization**: speaker separation requires a separate model (Whisper identifies what was said, not who said it)

```python
mel = audio_to_mel(audio)        # [80, 3000]
tokens = whisper_decode(mel)     # auto-regressive
text = detokenize(tokens)
```

### What Breaks
- **Fixed chunk size**: word boundaries at chunk edges can produce garbled transcriptions; use overlap with deduplication
- **No diarization**: "we have two speakers" requires a separate speaker diarization step
- **Proper nouns and domain terms**: low WER on general speech but higher on technical terms, names, and specialized vocabulary
- **Real-time**: Whisper is not inherently streaming; requires chunked architecture for low-latency transcription

### What the Interviewer Is Testing
Whether you understand the model architecture, its production constraints (chunking, VAD), and what it doesn't do (diarization, real-time streaming natively).

### Common Traps
- Treating Whisper as a drop-in real-time ASR solution without addressing chunk latency
- Conflating transcription accuracy with diarization capability
- Not mentioning VAD as a required preprocessing step in production

---

## Q8: What is multi-modal RAG, and how does it differ from text-only RAG?

### The Problem
Your document corpus contains images, charts, tables, and embedded figures. Text-only RAG misses these entirely or converts them to low-quality descriptions. Users ask questions whose answers are in a chart, not in surrounding text.

### The Core Insight
Multi-modal RAG extends the retrieval index to include non-text evidence: image embeddings, audio transcripts, video segment embeddings, layout-aware document embeddings. Retrieval becomes cross-modal: a text query can retrieve an image; an image query can retrieve text. Grounding requires citing the actual evidence modality, not just chunks.

### The Mechanics
Text-only RAG pipeline additions for multimodal:

**Evidence representation**:
- Images: vision encoder embeddings + optional caption/OCR
- Audio: ASR transcript embeddings + segment timestamps
- Video: frame/segment embeddings + ASR transcript
- Documents: layout-aware embeddings with bounding box metadata

**Retrieval**:
- Shared embedding space (CLIP-aligned) or separate indices per modality with merge step
- Metadata filters: ACL, modality type, timestamps, speaker
- Reranking with cross-modal cross-encoder

**Context packaging**:
```python
query_emb = embed_text(query)                           # or embed_image if query is image
hits = vector_index.search(query_emb, top_k=10, filters=acl_filters)
ctx = format_multimodal_context(hits)                   # captions, OCR snippets, timestamps, evidence IDs
answer = llm.generate(f"Use ONLY this evidence:\n{ctx}\n\nQuestion: {query}")
assert answer_cites_evidence_ids(answer, [h.id for h in hits])
```

**Faithfulness check**: claims in the answer must be entailed by retrieved evidence, including visual evidence.

### What Breaks
- **OCR/ASR errors**: bad extraction → irrelevant or wrong embeddings → retrieval misses relevant evidence
- **Missing provenance**: answer cites "the document" but not which page/frame/timestamp — unverifiable
- **Prompt injection via embedded text**: image OCR may contain adversarial instructions; treat extracted text as untrusted
- **Modality blindness**: ACL enforcement for images must be in the retrieval backend, not the prompt layer

### What the Interviewer Is Testing
Whether you understand that multimodal RAG is not just "give the model images" — it requires per-stage pipeline design, multimodal faithfulness evaluation, and provenance tracking.

### Common Traps
- Treating multimodal evidence as plain text without preserving modality IDs and evidence references
- Ignoring that OCR/ASR quality directly limits retrieval recall
- Not considering prompt injection from embedded text in images

---

## Q9: How do you build a system that processes both images and text?

### The Problem
You need to ship a product assistant that answers questions using both uploaded images and text documents. The system must be reliable, secure (ACL-aware), and verifiable (grounded answers with citations).

### The Core Insight
Don't force raw modalities together. Convert each to compatible representations first, then apply the same RAG pattern: retrieve evidence, package it, generate with grounding constraints, verify faithfulness.

### The Mechanics
```
Retrieval pipeline:
  query → embed (text or image) → ANN search against multimodal index (ACL filtered)
  → rerank → format evidence with modality IDs

Generation pipeline:
  prompt = f"Use ONLY the evidence below. Cite evidence IDs for each claim.\n{evidence}\n\n{query}"
  response = llm.generate(prompt, vision_tokens=visual_evidence)
  if not faithfulness_check(response, evidence):
      return "I can't confirm that from the available evidence."
```

Security layers:
- **OCR/ASR**: treat extracted text from images and audio as untrusted input (prompt injection vector)
- **ACL**: enforce in vector DB retrieval filters, not in the prompt
- **Output moderation**: run classifier on generated text for harmful content

Evaluation:
- Retrieval recall@k per modality (images vs text vs video segments)
- Answer faithfulness against multimodal evidence
- Citation accuracy (does the answer cite the right evidence IDs?)
- Safety: moderation pass rate on outputs

```python
img_tok = vision_encode(image)
hits = ann.search(embed_query(query), filters={"user_acl": user.acl_groups})
evidence = format_evidence(hits)
resp = llm.generate(messages=[{"role":"user", "content": query}],
                    system=f"Use only:\n{evidence}",
                    vision_tokens=img_tok)
```

### What Breaks
- Model ignores image tokens and generates from text priors (see Q23)
- Different image sizes cause resolution or aspect ratio issues; use consistent preprocessing with tiling for large images
- Mixed-modality evidence mixes high-confidence and low-confidence signals without visibility

### What the Interviewer Is Testing
System design thinking: preprocessing, retrieval, grounding, security, and evaluation as a complete pipeline — not just "call a VLM API."

### Common Traps
- Skipping ACL enforcement or placing it in the prompt (wrong layer)
- Not mentioning prompt injection from OCR-extracted text
- Treating "call VLM with image" as sufficient without grounding or faithfulness verification

---

## Q10: What are multi-modal embeddings, and how are they used for cross-modal search?

### The Problem
A user uploads an image and asks "find similar products." Your search index contains product descriptions in text. You need to retrieve text descriptions that match the image semantically — without any text query.

### The Core Insight
If you train encoders for different modalities with a shared contrastive objective, matching pairs end up close in the same embedding space. Then cross-modal retrieval is just nearest-neighbor search in that space: encode the query modality, search against the indexed target modality.

### The Mechanics
Training: contrastive loss on matched pairs (image-text, audio-text, image-audio)

```python
# at indexing time
for item in corpus:
    z = encode(item)        # vision encoder, text encoder, or audio encoder
    index.add(z, metadata={"id": item.id, "modality": item.type, "acl": item.acl})

# at query time
q = encode(query)           # encode with the appropriate modality encoder
hits = index.search(q, top_k=20, filters={"acl": user.groups})
hits = rerank_cross_encoder(query, hits)[:5]   # optional reranking for precision
```

Reranking is needed when ANN recall is good (right items exist in top-20) but relevance ranking needs a more expensive cross-modal cross-encoder for top-5 precision.

ACL enforcement must be in the index filter, not in the prompt. If a user shouldn't see an image, the embedding must not be returned — downstream filtering is too late.

### What Breaks
- **Domain mismatch**: CLIP trained on natural photos fails on medical images, technical diagrams, or specialized product photography
- **Similarity ≠ accuracy**: high cosine similarity doesn't mean the image matches the semantic intent; evaluate on downstream task metrics, not just similarity scores
- **OCR/transcript errors**: if text descriptions are built from OCR or ASR, errors propagate into the index

### What the Interviewer Is Testing
Understanding of how contrastive training creates a shared space, and the engineering requirements (ACL in backend, reranking, evaluation).

### Common Traps
- Claiming CLIP embeddings work universally without discussing domain adaptation needs
- Placing ACL filtering in the application layer or prompt instead of retrieval backend
- Not distinguishing ANN recall from reranking precision

---

## Q11: How do you evaluate multi-modal AI systems?

### The Problem
End-to-end metrics (e.g., "does the answer sound correct?") conflate four distinct failure modes: bad extraction (OCR/ASR), bad retrieval (wrong evidence returned), bad generation (hallucinated claims), and bad grounding (right evidence ignored). You need to measure each stage independently.

### The Core Insight
Multimodal evaluation is pipeline-aware. Measure extraction quality first. If extraction is broken, everything downstream is broken. Then retrieval. Then generation faithfulness against the retrieved evidence.

### The Mechanics
**Stage 1 — Modality extraction quality**:
- OCR character/word error rate on test documents
- ASR word error rate on test audio
- Frame detection quality for video (if applicable)

**Stage 2 — Retrieval quality** (if using RAG):
- Recall@k and mAP by modality slice
- Per-modality recall: are image evidence units retrieved when needed?

**Stage 3 — End-task quality**:
- VQA: exact match / F1 / human preference on answer quality
- Captioning: BLEU/ROUGE as rough proxies (unreliable alone); supplement with human judgments
- Faithfulness: NLI check that answer claims are entailed by retrieved evidence
- Citation accuracy: do cited evidence IDs actually support the claims?

**Stage 4 — Safety**:
- Moderation label pass rate on extracted text + generated outputs
- Adversarial test cases: text embedded in images, audio overlays

```python
extract_ok = ocr_wer < thr and asr_wer < thr
retrieval_recall = recall_at_k(pred_hits, gold_ids)
faithful = all(evidence_entails_claim(c, evidence_ids) for c in extract_claims(answer))
```

### What Breaks
- **Evaluating only end-task**: a 60% VQA score masks whether the problem is in OCR, retrieval, or generation
- **Ignoring temporal evaluation for video**: accuracy per frame may look fine while temporal consistency is poor
- **Human eval without stratification**: aggregate human ratings hide per-modality and per-difficulty failures

### What the Interviewer Is Testing
Whether you build evaluation as a staged pipeline rather than a single black-box metric.

### Common Traps
- Reporting only captioning BLEU without faithfulness check (BLEU measures n-gram overlap, not factual accuracy)
- Not having separate eval sets for each extraction modality
- Skipping adversarial test cases for prompt injection via embedded text

---

## Q12: What are the challenges of real-time multi-modal AI processing?

### The Problem
A live video moderation system needs to classify and optionally generate descriptions of video frames in near-real-time. The preprocessing (frame extraction, ASR, OCR) plus inference easily exceeds the available latency budget for each frame.

### The Core Insight
Real-time multimodal is a resource allocation problem under strict latency constraints. You can't run full pipeline on every frame. The design must be streaming-native with backpressure, adaptive compute, and per-stream state.

### The Mechanics
**Streaming pipeline**:
- VAD-gated audio chunks: only process segments with detected speech
- Adaptive frame sampling: motion-based or attention-based, not uniform fixed-rate
- Partial ASR updates: streaming decoder emits incremental transcripts

**Adaptive compute cascade**:
- Fast first-pass (small safety classifier, no generation) → route to slow path only when needed
- Cache: image/segment embeddings for repeated content; partial ASR for ongoing stream

**Backpressure and budgets**:
- Per-stream token budget: max frames per window, max transcript tokens for cross-attention
- Drop or downsample frames under load; emit confidence-aware partial results

**Temporal state**:
- Maintain per-stream context (speaker turns, prior segment summaries) for grounding consistency

```python
for segment in stream_windows():
    asr_partial = asr.update(segment.audio)
    if segment.vad_speech or confidence_low(asr_partial):
        fast_scores = fast_multimodal_checks(segment, asr_partial)
        if fast_scores.unsafe:
            block_or_delay(segment)
        elif fast_scores.uncertain:
            enqueue_slow_path(segment)
```

### What Breaks
- **Fixed-rate full pipeline**: latency exceeds budget on resource spike; requires adaptive downsampling
- **No backpressure**: unbounded queue builds up under load; drop or degrade gracefully
- **Inconsistent temporal context**: if prior segment state is lost, model loses coherence
- **Incomplete transcript used for decision**: abstain or use visual-only fallback when transcript is partial

### What the Interviewer Is Testing
Whether you think about real-time processing as a systems problem (budgets, backpressure, graceful degradation) rather than a model accuracy problem.

### Common Traps
- Running full multimodal generation per frame with no downsampling
- Not mentioning VAD as a prerequisite for efficient audio processing
- Treating latency as only the model inference time, ignoring preprocessing

---

## Q13: How do you handle video understanding with AI?

### The Problem
A video is too long to process as a single input. Passing every frame to a VLM is prohibitively expensive, and uniform frame sampling misses key events that happen in short bursts.

### The Core Insight
Video understanding is retrieval over temporal segments, not end-to-end attention over all frames. Sample events intelligently, encode each segment independently, maintain temporal ordering, and retrieve the relevant window for each query.

### The Mechanics
```
Video → segment (motion-based or uniform) → encode each segment
    → index with timestamps
    → at query time: retrieve relevant segments → generate grounded answer with timestamp citations
```

Pipeline:
1. **Segment selection**: motion-based, scene-change-based, or fixed windows with overlap
2. **Encode**: 3D conv or transformer over sampled frames per segment; produce `[T_segments, d]`
3. **Auxiliary signals**: ASR transcript for audio; OCR for on-screen text, subtitles
4. **Temporal fusion**: attention over segment embeddings; maintain memory of earlier segments for "what happened before X" queries
5. **Grounding**: answers reference segment timestamps

```python
segments = sample_video_segments(video, strategy="motion")
vid_embs = video_encoder(segments)                  # [T_segments, d]
text = asr.transcribe(video_audio)
answer = multimodal_llm.generate(
    question,
    evidence={"video_segments": vid_embs, "transcript": text},
    require_timestamp_citations=True
)
```

### What Breaks
- **Key event sampling failure**: motion-based sampling can miss slow-moving but important changes; evaluate with temporal QA sets
- **Long videos**: temporal attention over hundreds of segments exceeds context; use hierarchical segment summarization
- **No temporal grounding**: answers don't cite segment timestamps → unverifiable

### What the Interviewer Is Testing
That you approach video as a retrieval-over-segments problem with temporal grounding, not as "feed all frames to VLM."

### Common Traps
- Summarizing entire videos without segmentation, losing temporal specificity
- Not mentioning that ASR/OCR add valuable non-visual signals
- Not requiring timestamp citations in the output

---

## Q14: What is visual question answering (VQA)?

### The Problem
A user provides an image and asks a natural-language question. The model must use the image as evidence to generate the answer — not just answer from text priors about typical image contents.

### The Core Insight
VQA is grounded generation: the answer must be supported by actual visual evidence from the image, not by the model's statistical priors about what images typically contain.

### The Mechanics
Inputs: image (visual tokens via ViT) + question (text tokens)
Output: answer text

Two approaches:
- **Classification-head VQA**: map visual + text features to a fixed answer vocabulary; fast but limited to seen answer types
- **Generative VLM**: cross-attend over visual tokens while generating answer text; flexible but needs faithfulness control

Evaluation:
- Exact match / F1 for factual questions
- Human preference for open-ended descriptions
- **Visual entailment check**: does the answer correspond to what's actually in the image?

```python
answer = vlm.generate(image=image, question=question,
                      prompt="Answer based only on what is visible in the image.")
if not visual_entailment(answer, image):
    answer = "I can't confirm that from the image."
```

### What Breaks
- **Language prior dominance**: model answers "what color is the banana?" with "yellow" regardless of image content
- **Ambiguous questions**: "is there a dog?" with partial image view requires abstention, not a confident guess
- **Spatial reasoning**: VLMs trained on natural images often fail on counting, left/right, above/below reasoning

### What the Interviewer Is Testing
Whether you understand the grounding problem in VQA — not just the architecture — and how to enforce visual evidence use.

### Common Traps
- Not knowing that language priors cause answers that ignore the image
- Skipping visual entailment check (treating plausible text output as correct output)
- Not mentioning the spatial reasoning limitations of current VLMs

---

## Q15: What is document understanding, and how do models parse documents with layouts?

### The Problem
Enterprise documents (invoices, contracts, forms) contain structured information in tables, headers, and multi-column layouts. Treating them as plain text destroys spatial structure — the reading order is wrong, table cells are misassociated, field values are incorrectly attributed.

### The Core Insight
Documents have two information channels: the text content and the spatial layout. Layout-aware models encode both simultaneously — each OCR token gets spatial embeddings (bounding box coordinates) in addition to text embeddings, allowing the model to learn layout-specific reasoning.

### The Mechanics
```
Document image
    → OCR: extract text tokens with bounding boxes [token, x, y, w, h]
    → Layout transformer: encode with spatial positional embeddings + text embeddings
    → Optional: image backbone for non-text elements (signatures, stamps, diagrams)
    → Extraction head: predict field values with citations to region IDs

Output: {"invoice_total": "€1,234.56", "evidence_region": "page_1_table_row_3_col_2"}
```

Table handling requires specialized logic: detect table regions, extract cells, maintain row/column relationships.

```python
tokens, boxes = ocr.extract_with_boxes(document_image)   # [(token, x, y, w, h), ...]
doc_repr = layout_transformer(tokens=tokens, boxes=boxes)
fields = extractor_llm.generate(doc_repr, output_schema=schema, require_provenance=True)
```

### What Breaks
- **OCR quality is the bottleneck**: poor OCR propagates incorrect tokens into all downstream steps; measure OCR WER separately
- **Multi-column reading order**: standard OCR may read across columns incorrectly; need layout-aware reading order reconstruction
- **Low-confidence fields without abstention**: model extracts a value when it should say "not found in document"

### What the Interviewer Is Testing
Whether you understand that document understanding requires layout awareness, not just language understanding — and that OCR quality gates extraction quality.

### Common Traps
- Treating documents as plain text (dropping bounding boxes) and wondering why extraction is wrong
- Evaluating only field value accuracy without checking provenance correctness
- Not adding abstention for low-confidence or missing fields

---

## Q16: How do you fine-tune a vision-language model?

### The Problem
A general-purpose VLM trained on natural images performs poorly on your domain (medical imaging, product catalogs, engineering diagrams) because it lacks domain-specific visual vocabulary and doesn't know your output format requirements.

### The Core Insight
Fine-tune efficiently by freezing what generalizes (the vision encoder, most language model weights) and adapting what's domain-specific (cross-attention adapters, task-specific instruction examples). The limiting factor is usually training data quality and grounding supervision, not model size.

### The Mechanics
**Data requirements**:
- Domain image-text pairs for alignment (contrastive loss)
- Task-specific instruction examples: image + question → target answer with evidence citations
- Hard negatives: examples where the correct answer contradicts text-only priors
- Adversarial examples: prompt injection via OCR/image text

**Training strategy**:
1. Freeze vision encoder (or fine-tune last few layers if domain is very different)
2. Add domain adapters / LoRA to cross-attention and MLP layers
3. Train with: contrastive loss (alignment) + language modeling loss (generation)
4. Include "abstention" examples: images where the question can't be answered from visual evidence

**Evaluation**:
- Retrieval recall if using RAG
- VQA accuracy + faithfulness to visual evidence
- Structured output validity (JSON schema compliance)
- Hallucination rate: how often does the model claim to see things not in the image?

```python
for batch in multimodal_batches:
    loss = (
        contrastive_loss(batch.img_emb, batch.txt_emb)
        + lm_loss(batch.img, batch.target_output)
    )
    loss.backward()
    optimizer.step()
```

### What Breaks
- **Small datasets + unfrozen encoder**: overfits quickly; PEFT (LoRA/adapters) is safer for small domain datasets
- **No evidence-grounded examples**: model learns to generate plausible outputs without actually using visual evidence
- **Missing abstention training**: model over-confidently answers unanswerable visual questions

### What the Interviewer Is Testing
That you know fine-tuning is not just running train.py — it requires data quality, grounding supervision, and targeted parameter adaptation.

### Common Traps
- Fine-tuning without evidence/provenance supervision
- Freezing nothing (full fine-tune with 500 examples → severe overfitting)
- Not including adversarial or abstention examples in training data

---

## Q17: What are the latency and cost considerations for multi-modal AI in production?

### The Problem
Your multimodal pipeline is slow. Users complain. You profile and discover that model inference is only 30% of latency — the other 70% is image preprocessing, OCR, and ASR.

### The Core Insight
Multimodal latency is dominated by preprocessing (OCR, ASR, frame extraction) and encoding (vision encoder FLOPs), not just LLM token generation. Optimization must target the full pipeline, not just generation.

### The Mechanics
**Cost drivers by component** (typical order):
1. Video frame extraction + 3D encoding (highest cost per request for video)
2. OCR/ASR (non-trivial, especially at scale)
3. Vision encoder pass (ViT on images)
4. LLM cross-attention + generation (dominant for text, but mitigated by shorter visual tokens)

**Latency mitigation**:
- **Embedding cache**: hash image/document → cache vision encoder output; reuse across queries on the same image
- **Visual token compression**: Q-Former or pooling reduces N visual tokens to K < N; cuts cross-attention cost
- **Cascade routing**: fast classifier (cheap) → route to expensive VLM only when needed
- **Streaming**: emit partial ASR/captions progressively; don't wait for full video
- **Per-stream budgets**: max frames per window, max transcript tokens

```python
emb = cache.get(img_id) or (vision_encode(image), cache.set(img_id, ...))
hits = ann.search(embed_text(query), top_k=20, filters=acl)
if confidence(hits) > 0.8:
    return synthesize_from_hits(hits)              # retrieval-only path, no generation
return llm_cross_attn_generate(image, query)       # full VLM path
```

### What Breaks
- **Ignoring preprocessing**: reporting only LLM latency misses the actual bottleneck
- **Uncached vision encoding**: re-encoding the same product images per query wastes compute
- **No fast path**: all queries go through full VLM generation even when retrieval-only is sufficient

### What the Interviewer Is Testing
Whether you understand where compute actually goes in multimodal pipelines, and that optimization requires stage-level profiling.

### Common Traps
- Treating latency as only the LLM generation time
- Not knowing about visual token compression (Q-Former, pooling) as a cost-reduction mechanism
- Proposing a "use smaller model" solution without stage-level profiling to find the actual bottleneck

---

## Q18: How do you handle multi-modal content moderation?

### The Problem
Your platform accepts user-uploaded content: images, videos, audio messages, and text. A bad actor uploads a screenshot with harmful text embedded in the image as an overlay, bypassing the text classifier. Your moderation pipeline only checks the user-entered text fields.

### The Core Insight
Multi-modal moderation must extract and classify all modalities — including text hidden inside images via overlays, subtitles, or OCR-accessible content. The attack surface is the union of all modalities, not just the explicit text fields.

### The Mechanics
```
Input → extract all modalities
    → OCR from images/documents (treat extracted text as untrusted)
    → ASR from audio
    → frame sampling + visual classifiers for video
    → fuse safety signals
    → policy action (allow / label / block / human review)
```

Safety pipeline per modality:
- Text (user-entered): standard text classifier
- OCR text from images: text classifier, but flagged as "extracted from image" for audit
- Image frames: visual safety classifier (nudity, violence, IP violations)
- Audio transcript: text classifier on ASR output
- Final fusion: weighted combination of per-modality scores with calibrated thresholds

```python
ocr_text = ocr(image)
frames = sample_frames(video, strategy="uniform")
asr_text = asr.transcribe(audio) if audio else None

scores = fuse_safety_signals(
    text_cls(ocr_text),
    vision_cls(frames),
    text_cls(asr_text) if asr_text else None
)

if scores["unsafe"] > BLOCK_THR:
    return {"action": "block"}
elif scores["unsafe"] > REVIEW_THR:
    return {"action": "human_review", "evidence": scores}
return {"action": "allow"}
```

Borderline cases go to human review with all evidence IDs attached for auditing.

### What Breaks
- **Moderating only explicit text fields**: misses text-in-image attacks
- **No per-modality audit trail**: can't explain why content was blocked
- **Over-blocking**: visual classifiers have high false positive rates on culturally-specific content; requires localized calibration and appeal paths
- **Moderation of generated content**: if VLM generates text from visual content, that generated text also needs moderation

### What the Interviewer Is Testing
Whether you understand that the attack surface in multimodal systems includes all modalities, not just the visible text.

### Common Traps
- Moderating only user-entered text while ignoring image/video overlays
- Not maintaining evidence IDs for audit and appeals
- Treating moderation as a single binary classifier rather than a per-modality pipeline with fusion

---

## Q19: What is text-to-video generation, and what are current approaches?

### The Problem
You want to generate a short video clip from a text prompt. Unlike image generation, video must be temporally coherent: objects can't teleport between frames, motion must be physically plausible, and lighting/style must stay consistent.

### The Core Insight
Video generation extends latent diffusion from 2D spatial to 3D spatiotemporal: denoise across both spatial dimensions and time simultaneously, conditioned on text. Temporal coherence is the central technical challenge.

### The Mechanics
**Video diffusion (latent)**:
```
Encode T frames → latent tensor [T, H', W', C']
Add noise across time → x_T
Train 3D U-Net to predict noise: eps_theta(x_t, t, cond=text_emb)
Denoise iteratively to produce [T, H', W', C'] → decode each frame

z = randn([T, H_lat, W_lat, C])
for t in timesteps:
    z = video_unet_denoise(z, t, cond=text_emb)
video = vae_decoder(z)  # applied per frame
```

**Temporal consistency mechanisms**:
- Temporal attention layers: frames attend to neighboring frames
- Motion conditioning: explicit optical flow or camera motion embeddings
- Hierarchical generation: low-res keyframes first → interpolate + upscale

**Long video strategies**: generate short windows with overlap, stitch with temporal constraints.

### What Breaks
- **Temporal flickering**: per-frame inconsistencies if temporal attention is insufficient
- **Compute cost**: T× more expensive than image generation; 30 frames at 512×512 is a major memory and compute challenge
- **Prompt ambiguity**: text prompts can't fully specify camera motion, lighting changes, or object trajectories
- **Content safety**: generated video requires provenance/watermarking and moderation

### What the Interviewer Is Testing
Whether you understand the extension from image diffusion to video (temporal coherence is the new challenge) and the current limitations.

### Common Traps
- Evaluating only per-frame quality and ignoring temporal consistency
- Not knowing what latent video diffusion is
- Treating text-to-video as a solved problem rather than an active research area with significant compute and quality constraints

---

## Q20: Explain multi-modal fusion: early fusion vs. late fusion.

### The Problem
You have two modalities — image and text — and need to combine them for a classification task. Should you mix their representations early (let the model learn cross-modal interactions) or late (combine scores from independently trained models)?

### The Core Insight
The choice depends on whether the task requires tight cross-modal interaction (early fusion) or whether modalities can be processed independently and combined at decision level (late fusion). Early fusion captures interactions but increases compute and fragility; late fusion is modular and robust but may miss fine-grained cross-modal patterns.

### The Mechanics
**Early fusion** (feature/token level):
```
x = concatenate(img_tokens, text_tokens)    # or interleave
out = transformer(x)                        # joint attention from first layer
```
- Cross-modal attention from the start
- Risk: if one modality is corrupted, it corrupts the joint representation

**Late fusion** (decision level):
```python
img_score = img_model(image)     # [B, num_classes]
txt_score = txt_model(text)      # [B, num_classes]
final = w1 * img_score + w2 * txt_score   # learned or fixed weights
```
- Independent per-modality quality; swap one without retraining the other
- Can't capture cases where meaning requires jointly attending to both

**Cross-attention (middle ground)**:
```python
# Text decoder attends to image tokens — tighter integration than late fusion,
# but image encoder is still independent
output = decoder(text_tokens, cross_attn_source=visual_tokens)
```

**Rule**: use early fusion / cross-attention for tasks requiring fine-grained joint reasoning (VQA, captioning); use late fusion for retrieval, classification, or when robustness to modality noise is critical.

### What Breaks
- **Early fusion + noisy modality**: one bad image corrupts the entire representation
- **Late fusion + tightly coupled task**: loses cross-modal interactions needed for correct answers
- **Wrong fusion for task**: a retrieval model (late fusion) used for VQA will miss image-grounded details

### What the Interviewer Is Testing
Whether you can explain the trade-off and match the fusion strategy to the task type.

### Common Traps
- Choosing one approach without justifying based on task type
- Not knowing that cross-attention is the practical middle ground (most production VLMs)
- Claiming early fusion is always better without acknowledging the noise and compute trade-offs

---

## Q21: Your VLM generates factually incorrect image descriptions. How do you fix it?

### The Problem
Your image description pipeline for a product catalog is generating descriptions that include attributes not present in the image — wrong colors, fabricated dimensions, non-existent text on the label. The outputs sound plausible but are factually wrong.

### The Core Insight
The model is generating from language priors, not from visual evidence. The fix requires forcing the model to ground claims in visual evidence and verifying that generated claims are actually supported by what's in the image.

### The Mechanics
**Without retraining (runtime controls)**:
1. Prompt constraint: "Describe only what is visually confirmed in the image. If uncertain about any detail, say 'unclear.'"
2. Claim extraction + visual entailment check:
```python
draft = vlm.generate(image, prompt="Describe facts only from this image.")
claims = extract_claims(draft)
if not all(verify_visual_entailment(c, image) for c in claims):
    draft = "I cannot confirm all details from the image."
return draft
```
3. Multi-candidate reranking: generate N candidates, pick the one with highest faithfulness score

**With retraining**:
- Add hard negative training examples: prompts where correct answer contradicts text priors
- Include region-level supervision: train model to cite bounding box regions for each claim
- Include abstention examples: images where "I can't confirm X from this image" is the correct answer

### What Breaks
- **Prompt-only fix without verification**: model still generates plausible-sounding unchecked claims
- **Visual entailment check is expensive**: running a separate VQA model per claim adds latency; balance rigor against cost
- **Feedback loop on corrections**: if you just log incorrect outputs without adding them to the eval set, the problem recurs

### What the Interviewer Is Testing
Whether you understand that hallucination in VLMs is a grounding problem requiring verification, not just a prompting problem.

### Common Traps
- Fixing with only prompt rewording and calling it done
- Not adding a faithfulness verification step
- Not building regression tests from observed hallucination cases

---

## Q22: Your VLM answers single-image questions well but fails on multi-page documents. How do you fix it?

### The Problem
Your document assistant works on individual pages but fails when the answer spans multiple pages or requires synthesizing information across sections. The model doesn't know which page to look at.

### The Core Insight
Multi-page document QA is a retrieval problem, not a "give the model all pages" problem. You need to: segment documents into evidence units, index each unit, retrieve the relevant pages for each query, and generate with page-level citations.

### The Mechanics
```python
# Preprocessing (at index time)
pages = split_doc_into_pages(doc)
for page in pages:
    layout_tokens, boxes = ocr_with_layout(page.image)
    page_emb = layout_embed(layout_tokens, boxes)
    index.add(page_emb, metadata={"doc_id": doc.id, "page": page.num, "acl": doc.acl})

# Query time
hits = retrieve_pages(query, index, top_k=3)
evidence = format_pages_with_ids(hits)   # "Page 3: [text...], Page 7: [text...]"
answer = vlm.generate(
    evidence=evidence,
    question=query,
    require_citations=True      # "According to page 3..."
)
assert citations_are_valid(answer, hit_page_ids)
```

### What Breaks
- **Retrieval recall failure**: the right page wasn't retrieved; check retrieval recall@k on a gold labeled set
- **OCR reading order error on multi-column layouts**: breaks text extraction and therefore retrieval
- **Model ignores retrieved pages**: enforce citation requirement and validate citation IDs

### What the Interviewer Is Testing
Understanding that multi-page document QA is a RAG problem with layout-aware indexing, not a "longer context window" problem.

### Common Traps
- Trying to fit the entire document into context ("just use a 128k context window")
- Running the VLM on the whole document as a single image without page evidence structure
- Not requiring page citations in the output

---

## Q23: Your multimodal LLM ignores the image and generates from text alone. How do you fix it?

### The Problem
A product assistant is supposed to describe what's in uploaded images, but analysis shows the model's outputs are statistically identical whether an image is provided or not. The model is ignoring visual tokens.

### The Core Insight
If the model can get high reward from text priors alone, it will. Visual token utilization requires explicit enforcement: both in evaluation (measure whether citations to visual evidence appear) and in architecture (ensure visual tokens are not truncated from context).

### The Mechanics
**Diagnostic steps**:
1. Measure: run the same questions with and without image; if outputs are statistically similar, visual tokens are ignored
2. Check architecture: are visual tokens actually fed into cross-attention, or are they being truncated due to context length limits?
3. Check training data: is there a strong text-only signal that makes image redundant?

**Runtime fixes**:
```python
resp = vlm.generate(image=image, question=q,
                    require_image_citations=True)
if not citation_supports_answer(resp, image):
    resp = "I need clearer image input to answer this accurately."
return resp
```

**Architectural fixes**:
- Reduce text context length to ensure visual tokens are not crowded out
- Use prefix position for visual tokens (before text), not suffix
- Check that cross-attention layers are actually reading visual tokens (attention visualization)

**Training fixes**:
- Hard negative examples: questions where image contradicts the text-prior answer
- "Image-required" examples: questions that can only be answered from the image
- Penalize answers that claim visual content not grounded in image tokens

### What Breaks
- Prompt-only fixes (adding "describe the image") don't work if the model has learned to ignore visual tokens architecturally
- Attention inspection is necessary but not sufficient — model may attend to visual tokens but apply them with near-zero weight

### What the Interviewer Is Testing
Whether you diagnose the problem at the right level (architecture/training) rather than defaulting to prompt engineering.

### Common Traps
- Only changing the prompt without verifying visual token utilization
- Not knowing that context truncation can silently drop visual tokens
- Reporting that "the model uses the image" based on anecdotal good cases without systematic evaluation

---

## Q24: Your diffusion model ignores precise control requirements in text prompts. How do you improve controllability?

### The Problem
Users ask the model to generate "a red car on the left side of the image with a blue sky background" but get results where the car is centered, the sky is gray, and the color is wrong. Text prompts alone are insufficient for precise spatial or attribute constraints.

### The Core Insight
Text prompts are too ambiguous for precise spatial and attribute control. Controllable generation requires explicit structured control signals — masks, depth maps, edge maps, bounding boxes, reference images — in addition to text. These machine-readable constraints are unambiguous in a way that natural language isn't.

### The Mechanics
**Prompt-level improvements** (cheapest):
- Use structured attribute lists: "object: red car; position: left; background: blue sky; no text, no watermark"
- Negative prompts: "no grey, no center-position car"
- Higher CFG scale to strengthen prompt adherence (may reduce diversity)

**Control signal conditioning** (stronger):
```python
control = {
    "depth_map": depth_estimator(reference_image),
    "edge_map": edge_detector(reference_image),
    "bounding_boxes": parse_layout_from_prompt(prompt)
}
img = diffusion_sample(cond=text_emb, negative_prompt=neg, control=control, cfg_scale=8.0)
```

**Evaluation**:
- Measure attribute compliance: color detection, object position detection, style label classification on generated outputs
- Not just "does it look good" — measure whether specified attributes appear

**Fine-tuning for control**: train on examples with structured conditioning + ground truth compliance labels.

### What Breaks
- **CFG too high**: strong adherence but artifacts and reduced diversity
- **Conflicting constraints**: "vintage style" + "photorealistic" may conflict; model needs to resolve priority
- **Evaluation gap**: qualitative review misses systematic attribute failures; need automated attribute measurement

### What the Interviewer Is Testing
Whether you distinguish between what text prompts can express vs what structured control signals can enforce, and how to measure compliance.

### Common Traps
- Over-relying on natural language wording for precise constraints
- Not knowing about ControlNet-style conditioning as a standard technique
- Evaluating controllability qualitatively without automated attribute measurement

---

## Q25: Your diffusion model generates sharp but repetitive images. How do you balance quality vs. diversity?

### The Problem
High guidance scale + high steps produces sharp images that all look the same. Users want variety while maintaining quality.

### The Core Insight
High classifier-free guidance amplifies the most likely mode, suppressing diversity. The quality-diversity trade-off is a hyperparameter trade-off with CFG scale as the primary knob — plus a multi-candidate strategy to sample across modes.

### The Mechanics
**CFG scale**:
- High (≥10): strong prompt adherence, low diversity
- Low (2–5): more variety, potentially weaker fidelity
- Sweep on your distribution to find the Pareto frontier

**Multi-candidate generation + selection**:
```python
imgs = []
for seed in seeds[:8]:
    imgs.append(diffusion_sample(seed=seed, cfg_scale=cfg_scale))

# Pick by: quality × diversity trade-off
best = pick_by_quality_diversity(
    imgs,
    quality_model=aesthetic_scorer,    # or CLIP similarity to prompt
    diversity_metric=embedding_distance
)
```

**Diversity metric**: average pairwise embedding distance in the candidate set — higher is more diverse.

**Trade-off objective**: maximize `quality + λ × diversity` where λ is tuned based on use case (creative ideation: high λ; brand compliance: low λ).

### What Breaks
- Lowering CFG without checking attribute compliance on constrained prompts
- Not evaluating on the full prompt distribution — some prompts are more sensitive to CFG changes
- Diversity as measured by embedding distance may not match user-perceived diversity

### What the Interviewer Is Testing
Whether you understand CFG as the quality-diversity lever and can propose a principled evaluation approach.

### Common Traps
- Claiming you can achieve high diversity and high adherence simultaneously without acknowledging the trade-off
- Not knowing that multiple seeds + candidate selection is standard practice for diversity

---

## Q26: Your diffusion model is too slow. How do you speed up sampling?

### The Problem
Each image takes 4+ seconds to generate. Users expect results in under 1 second. You can't simply buy more GPUs — the sampling loop itself is too many sequential steps.

### The Core Insight
Most latency in diffusion is the sequential denoising loop. Each step depends on the previous. You can't parallelize steps easily, but you can reduce the number of steps while preserving quality using smarter samplers and distillation.

### The Mechanics
**Step reduction with better samplers**:
- DDIM: deterministic, high-quality with 20-50 steps (vs 1000 for DDPM)
- DPM-Solver++: 10-20 steps with similar quality
- These are ODE solvers that take larger, smarter steps through the noise schedule

**Latent diffusion** (already standard in Stable Diffusion):
- Denoise in compressed latent space → decode once with VAE
- ~4-8× cheaper than pixel-space diffusion

**Distillation to fewer steps**:
- Consistency models: train model to map any point on the trajectory to the final image — single step
- Progressive distillation: iteratively distill a N-step model into N/2-step

**Caching**:
- Cache text embeddings (constant per prompt)
- Cache vision encoder outputs for reference images

**Evaluation gate**:
- Before deploying faster sampler, measure quality degradation on representative prompts at each step count
- Choose the Pareto-optimal point (minimal steps where quality degradation is acceptable)

```python
img = diffusion_sample(
    steps=20,
    sampler="dpm_solver_pp",
    latent_mode=True,
    text_emb=cache.get(prompt_hash) or encode_text(prompt)
)
```

### What Breaks
- Fewer steps with wrong sampler choice: artifacts and quality degradation
- Distillation on wrong data: distilled model fails on out-of-distribution prompts
- Not evaluating quality regression: shipping faster sampler without checking outputs

### What the Interviewer Is Testing
Whether you know the actual techniques (DDIM, DPM-Solver, distillation) and understand that speed optimization requires quality evaluation.

### Common Traps
- Suggesting "reduce steps" without knowing which sampler to use
- Not knowing that latent diffusion is the baseline for most modern systems
- Not proposing an evaluation methodology to validate the speed-quality trade-off

---

## Reference: Key Multi-modal Patterns

| Problem | Core Mechanism | Where It Breaks |
|---|---|---|
| Image-text alignment | Contrastive loss (InfoNCE) on matched pairs | Domain mismatch; fine-grained attributes |
| Visual token integration | ViT patch embedding → projection → cross-attention | Context truncation; visual tokens ignored |
| Cross-modal retrieval | Shared embedding space + ANN index | ACL must be in backend, not prompt |
| Grounded generation | Require visual evidence citations + faithfulness check | Language priors dominate without enforcement |
| Document understanding | OCR + spatial embeddings + layout transformer | OCR quality gates all downstream steps |
| Real-time processing | Streaming + VAD + adaptive compute cascade | Fixed-rate full pipeline exceeds latency budget |
| Image generation controllability | Structured control signals (ControlNet) + CFG | Text prompts too ambiguous for precise constraints |
| Speed vs quality | Fewer steps + better samplers (DPM-Solver) | Quality regression without evaluation gate |
