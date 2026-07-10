# Interview 04 — LLM-Powered NPC Dialogue Assistant (Condensed)

**Problem:** Design the backend "brain" service for a real-time voice-driven NPC in a game (e.g. Mass Effect). Player speaks → STT → your service returns streaming text → TTS → NPC voice. Must stay in-character, avoid hallucinating lore, resist jailbreaks, and hit tight voice-to-voice latency.

---

## Clarifying Questions to Ask
- End-to-end latency budget? → **< 1.5s** voice-to-voice, faster is better.
- Are we building STT/TTS too? → **No**, separate services; you own text-in → text-out "Brain".
- How strict on lore/hallucination? → **Very strict** — NPC must not invent lore, must stay in character if it doesn't know something.
- Jailbreak/toxicity tolerance? → **Zero** — must not break character or generate hate speech even under provocation.
- Self-hosted vs API model? → **Open-weight model** (Llama 3 8B / Mistral) self-hosted on k8s, for latency + cost control.

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

## Toughest Follow-ups

**Q: Farmer NPC gets asked about the level-5 boss — standard RAG might retrieve and leak the strategy guide. How do you prevent this?**
> Use metadata-filtered/namespaced RAG — the farmer persona only has retrieval access to "Farmer Lore"/"Local Town Lore" namespaces. Query against boss-fight lore returns nothing, and the system prompt instructs the model to act confused/in-character rather than answer from general knowledge.

**Q: Finance wants a 50% cost cut without downgrading the model. Options?**
> Three levers: (1) INT4 quantization (AWQ/GPTQ) cuts VRAM ~70%, enabling cheaper GPUs (T4/L4 instead of A100/A10G); (2) tune vLLM continuous batching to maximize GPU utilization and lower cost-per-token; (3) enable prefix/prompt-caching so the static system prompt's KV cache isn't recomputed every request.

**Q: A player jailbreaks the NPC live on stream. Need an immediate fix without retraining. What do you do?**
> Deploy an emergency output filter (exact-match blocklist / fast regex) in front of TTS, plus a targeted input guardrail blocking the specific phrases used — both pushed via dynamic config (Redis/LaunchDarkly) so it's live in seconds without a redeploy.

---

## Biggest Pitfall
Proposing a synchronous "wait for full LLM output → then TTS → then play audio" pipeline — it blows the 1.5s latency budget by 3-6x and shows the candidate didn't internalize the core constraint of the problem.
