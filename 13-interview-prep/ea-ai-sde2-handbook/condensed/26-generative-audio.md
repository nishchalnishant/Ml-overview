# Interview 26 — Generative Audio: Dynamic NPC Voice Cloning (Condensed)

Design a TTS/voice-cloning pipeline to voice 500 NPCs across 1M+ dialogue lines generated dynamically by an LLM, matching original actors' voices, emotionally expressive, fast enough for live gameplay.

## Core Architecture
```
LLM Dialogue Service → (streamed text + emotion tag)
   → TTS Inference Service (GPU cluster)
       1. Text normalization (numbers/jargon expansion)
       2. Acoustic model (VITS/XTTS) conditioned on speaker + emotion embedding
       3. Vocoder (HiFi-GAN) → 24kHz waveform chunks
   → streamed audio buffers → Game Client plays seamlessly
```
- **VITS** (end-to-end acoustic+vocoder) or **XTTS** (zero-shot cloning from 3s clip) — avoids Tacotron2+WaveGlow's extra latency hop.
- **Speaker embeddings (d-vectors)** via ECAPA-TDNN — one multi-speaker model serves 500 NPCs, no per-NPC training.
- **Sentence-level chunking** of LLM output stream — synthesize on punctuation boundaries, not full paragraphs, to hit SLA.
- **Emotion/style tokens** or Global Style Token embeddings condition prosody.
- **CDN/S3 audio caching** keyed on hash(text+speaker) to avoid re-hitting GPU for repeated lines.

## Talking Points That Signal Seniority
- Proactively says streaming synthesis (sentence-chunked) is required — batch-processing whole paragraphs fails the SLA.
- Flags chunk-boundary phase discontinuity ("popping") and proposes cross-fade / passing prior chunk's trailing waveform context into next chunk.
- Proposes text-hash caching (CDN-backed) as the primary cost lever at 10M-player scale, not just bigger GPUs.
- Names MOS and MCD as the real audio-quality metrics — explicitly rejects loss/MSE as inadequate for perceptual quality.
- Raises deepfake/abuse risk unprompted: signed API tokens from game server + pre-TTS toxicity/moderation filter on LLM text.
- Mentions phoneme-based normalization for game-specific jargon/fantasy names to prevent mispronunciation hallucinations.
- Suggests TensorRT compilation for the model to hit latency budget on GPU.
- Proposes speaker-embedding interpolation for stylized voices (e.g., robot NPC) instead of cheap post-processing filters.

## Top 3 Tradeoffs
- **Sentence vs word chunking** — sentence chunking gives natural prosody but ~500ms added latency; word chunking hits ~50ms but sounds robotic/stilted.
- **Zero-shot cloning (XTTS) vs fine-tuning (VITS per actor)** — zero-shot is instant and scalable from 3s of audio; fine-tuning needs 5hrs audio + 2 days GPU but yields glitch-free AAA-grade quality.
- **Grapheme vs phoneme input** — raw text is simpler/modern but mispronounces invented names; phoneme dictionaries add engineering overhead but give control needed for fantasy jargon.

## Biggest Pitfall
Proposing to just call an off-the-shelf API (e.g., ElevenLabs) instead of engineering the internal pipeline — ignores latency control, cost at EA's scale, and voice-rights/data ownership, and is an instant drop toward No Hire.
