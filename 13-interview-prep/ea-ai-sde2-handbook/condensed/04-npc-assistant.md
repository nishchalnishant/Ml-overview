# Interview 04 — LLM-Powered NPC Dialogue Assistant (Condensed)

**Problem:** Design the backend "brain" service for a real-time voice-driven NPC in a game (e.g. Mass Effect). Player speaks → STT → your service returns streaming text → TTS → NPC voice. Must stay in-character, avoid hallucinating lore, resist jailbreaks, and hit tight voice-to-voice latency.

---

## Core Architecture
```
Player Mic → STT → [NPC Dialogue Service] → streaming text chunks → TTS → Player
                         │
   1. Input guardrail (jailbreak/toxicity regex or small classifier)
   2. Context assembly: Redis conversation history + NPC persona + RAG lore lookup
   3. Prompt build (persona + history + lore + new msg)
   4. LLM generation via vLLM (streaming, PagedAttention, continuous batching)
   5. Sentence-boundary chunking (buffer until ./!/?) before handoff to TTS
   6. Output guardrail (streaming toxicity filter)
```
- **Key technique:** vLLM for serving — PagedAttention enables high-concurrency batching per GPU, critical for cost/latency at scale.
- **Sentence-level streaming**, not token/word-level — word-by-word makes TTS intonation sound robotic; must buffer to punctuation.
- **RAG for lore**, scoped by persona/namespace, not global — prevents an in-universe farmer NPC from "knowing" boss strategies.

---

## Talking Points That Signal Seniority
- Proactively say synchronous (wait-for-full-response) architecture is a non-starter — must stream, or latency blows past 1.5s to 5-10s.
- Mention PagedAttention / continuous batching by name as the reason vLLM beats a naive HF `transformers` pipeline for concurrent GPU serving.
- Call out sentence-boundary buffering explicitly as the fix for robotic TTS pacing — word-level streaming is a trap.
- Propose scoping RAG retrieval by NPC persona/namespace (metadata filtering) so a farmer can't accidentally surface boss-fight lore.
- Bring up quantization (AWQ/GPTQ INT4) unprompted as the cost lever — 16GB→5GB VRAM, cheaper GPUs or bigger batches.
- Mention prefix/prompt caching in vLLM to skip re-computing the static system prompt's KV cache on every request.
- Flag that Redis conversation history needs summarization/truncation, or TTFT silently degrades as sessions grow.
- Propose a fast async "Llama-Guard"-style output classifier that can cut the TTS stream mid-response if toxicity slips through.

---

## Top 3 Tradeoffs
- **Self-hosted Llama 3 8B vs GPT-4 API:** open weights = no variable cost, no network round-trip latency, but you own GPU infra and it's less capable out-of-the-box.
- **Sentence-level vs word-level TTS streaming:** sentence buffering adds ~300ms upfront latency but sounds natural; word-level is faster to start but robotic/unnatural intonation.
- **System prompt vs LoRA fine-tuning for persona:** system prompts iterate fast but burn context tokens every request; LoRA costs engineering effort upfront but is cheaper per-token and sticks to character better.

---

## Biggest Pitfall
Proposing a synchronous "wait for full LLM output → then TTS → then play audio" pipeline — it blows the 1.5s latency budget by 3-6x and shows the candidate didn't internalize the core constraint of the problem.
