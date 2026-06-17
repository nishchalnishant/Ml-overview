---
module: Llms
topic: Interview Notes
subtopic: Multi Modal Ai Snappy
status: unread
tags: [llms, ml, interview-notes-multi-modal-ai]
---

> _Quick-recall companion. For the full deep-dive, see [multi-modal-ai.md](multi-modal-ai.md)._

# Multi-modal AI — text, images, audio (without the hype)

Multi-modal systems are just the same engineering story with more inputs: **bigger artifacts, higher cost, nastier safety**.

**One-line:** Multi-modal = shared representations + modality-specific encoders/decoders + careful routing.

---

# Q1: What are multi-modal AI models, and how do they process different types of data?
- **Direct answer:** Use encoders per modality and a fusion mechanism to reason/generate.

---

# Q2: How do vision-language models process images?
- **Direct answer:** Vision encoder → tokens/patch embeddings → cross-attend with text.

---

# Q3: How does CLIP work, and why is it important?
- **Direct answer:** Contrastive training aligns image/text embeddings for cross-modal search.
- **Azure/DevOps bridge:** treat inputs as artifacts; validate, version, and monitor costs per request.

---

# Q4: Key architectures for multi-modal models?
- **Direct answer:** Encoder-decoder, cross-attention, unified token models, mixture-of-experts.

---

# Q5: How does image generation work with diffusion models?
- **Direct answer:** Denoise from noise with a learned score model; classifier-free guidance for control.
- **Mini prompt:** what’s the first knob to turn for speed? → reduce steps / batch / cache.

---

# Q6: What is TTS?
- **Direct answer:** Text → acoustic tokens/waveform via neural vocoders.

---

# Q7: How does Whisper-style STT work?
- **Direct answer:** Audio → encoder → decoder predicts text tokens.

---

# Q8: What is multi-modal RAG?
- **Direct answer:** Retrieve text + images (and maybe OCR/layout) as evidence, not just text.
- **Azure/DevOps bridge:** treat inputs as artifacts; validate, version, and monitor costs per request.

---

# Q9: Build a system that processes images + text?
- **Direct answer:** Ingest → OCR/layout → embed → retrieve → VLM → validate output.
- **Azure/DevOps bridge:** treat inputs as artifacts; validate, version, and monitor costs per request.

---

# Q10: Multi-modal embeddings for cross-modal search?
- **Direct answer:** Shared embedding space enables ‘text query → image results’ and vice versa.

---

# Q11: Evaluate multi-modal systems?
- **Direct answer:** Task metrics + human review; check hallucinations and grounding.

---

# Q12: Challenges of real-time multi-modal processing?
- **Direct answer:** Compute, bandwidth, batching, latency; async pipelines help.

---

# Q13: Video understanding?
- **Direct answer:** Sample frames + temporal modeling; retrieval of key segments.

---

# Q14: What is VQA?
- **Direct answer:** Answer questions about images using visual + textual context.

---

# Q15: Document understanding with layouts?
- **Direct answer:** OCR + layout tokens; treat page structure as features.

---

# Q16: Fine-tune a vision-language model?
- **Direct answer:** PEFT/LoRA on cross-attention and adapters; curate high-quality pairs.

---

# Q17: Latency/cost considerations?
- **Direct answer:** Images/video are expensive; compress inputs, cache, route by complexity.
- **Azure/DevOps bridge:** treat inputs as artifacts; validate, version, and monitor costs per request.

---

# Q18: Multi-modal content moderation?
- **Direct answer:** Detect unsafe content in images/video/audio + text; multi-stage filters.
- **Azure/DevOps bridge:** treat inputs as artifacts; validate, version, and monitor costs per request.

---

# Q19: Text-to-video generation?
- **Direct answer:** Diffusion/transformer hybrids; expensive and currently noisy in control.

---

# Q20: Early vs late fusion?
- **Direct answer:** Early: fuse features early; Late: separate models combine decisions.

---

# Q21: Bad image descriptions (hallucinations) — fix?
- **Direct answer:** Force grounding (regions/OCR), lower temperature, add verification/citations.

---

# Q22: VLM fails on multi-page docs — fix?
- **Direct answer:** Page-wise processing + layout-aware retrieval + hierarchical summarization.

---

# Q23: Model ignores image — fix?
- **Direct answer:** Ensure image tokens present; better prompts; model choice; test with counterfactuals.

---

# Q24: Diffusion ignores precise control — improve?
- **Direct answer:** ControlNet, adapters, reference images, better conditioning, negative prompts.
- **Mini prompt:** what’s the first knob to turn for speed? → reduce steps / batch / cache.

---

# Q25: Sharp but repetitive images — balance?
- **Direct answer:** Tune guidance/temperature, diversity controls, better sampling schedules.

---

# Q26: Sampling too slow — speed up?
- **Direct answer:** Fewer steps, distillation, better schedulers, faster kernels, smaller model.
- **Mini prompt:** what’s the first knob to turn for speed? → reduce steps / batch / cache.
