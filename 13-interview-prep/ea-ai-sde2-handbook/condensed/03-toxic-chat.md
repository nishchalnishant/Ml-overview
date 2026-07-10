# Interview 03 — Toxic Chat Moderation (Cheat Sheet)

Real-time microservice to detect/block toxic messages (hate speech, harassment) in in-game chat across multiple titles before they reach other players. Core tension: sub-50ms latency at 20k QPS vs. accuracy needed to catch obfuscated/nuanced toxicity.

## Clarifying Questions to Ask
- Latency SLA? → p99 < 50ms (else laggy/timeout).
- Peak QPS? → ~20,000 globally.
- FP vs FN preference? → Heavily minimize false positives (players hate being wrongly blocked).
- Languages? → English only for v1, architecture must be extensible.
- Need conversation context or single-message eval? → Single string only, for latency.
- How to handle game-specific slang (e.g. "kill" in a shooter vs. FIFA)? → `game_id` is passed in request, use per-game thresholds.

## Core Architecture
```
Game Client → API Gateway → Toxicity Service (FastAPI)
   1. Regex/blocklist + Redis cache for common phrases (O(1), skip ML)
   2. Lightweight ML classifier (<10ms)
   3. Per-game threshold lookup → is_toxic decision
   → async log to Kafka (audit + retraining), fail-open on error
```
- **Model choice: DistilBERT via ONNX Runtime** (or fastText if throughput-bound) — small transformer gets context/obfuscation understanding that regex misses, ONNX gives 2-5x CPU inference speedup over raw PyTorch.
- CPU serving, not GPU — batch size 1 requests favor low single-request latency over GPU's batching throughput advantage.
- Cache layer for high-frequency benign strings ("gg", "lol") to avoid hitting the model at all.
- Kafka sampling instead of full logging — bandwidth doesn't scale linearly with QPS otherwise.

## Talking Points That Signal Seniority
- Immediately names latency (not accuracy) as the binding constraint given 20k QPS + 50ms SLA.
- Proposes hybrid funnel (regex/cache → cheap model → threshold) instead of routing every message through a transformer.
- Justifies CPU-over-GPU explicitly by batch-size-1 reasoning, not just cost.
- Raises **fail-open vs fail-closed** as a product decision, not just an engineering detail.
- Flags that per-game thresholds imply per-game calibration/monitoring, not a single global cutoff.
- Volunteers text normalization (unicode stripping, homoglyph mapping) as needed even before being asked about obfuscation.
- Proposes sampling strategy for Kafka logging (100% toxic, 100% borderline, 1% benign) to control bandwidth while preserving retraining signal.
- Mentions shadow-mode deployment for safely rolling out a new model version.

## Top 3 Tradeoffs
- **DistilBERT vs fastText**: DistilBERT understands context/obfuscation better but is ~20x slower/costlier — choose fastText first if QPS/cost dominates.
- **CPU vs GPU serving**: GPU wins on batched throughput; CPU wins on single-request latency and is simpler to autoscale in K8s — matters because requests arrive one at a time, not in batches.
- **Fail open vs fail closed**: Fail-open keeps the game usable if the model service dies but lets toxicity through; fail-closed protects users but breaks chat entirely on an outage — this is a product/trust-and-safety call, not just ops.

## Toughest Follow-ups
**Q: Model has max_length=128 tokens, but messages can be 10,000 chars. What now?**
A: Don't truncate blindly — toxicity may be at the end. Chunk into ~100-token windows, batch-run inference across chunks, flag the whole message if any chunk exceeds threshold. Also rate-limit/penalize abnormally long messages since they're likely spam.

**Q: Players bypass the model using zero-width spaces inside slurs. Fix it?**
A: Add a normalization step before tokenization: strip non-printable/invisible unicode, and apply homoglyph mapping (Cyrillic "а" → Latin "a", "@" → "a") so obfuscated variants collapse to the canonical word before the model or regex ever sees them.

**Q: Logging every request to Kafka at 20k QPS saturates bandwidth. Reduce without losing retraining signal?**
A: Sample by confidence band — log 100% of flagged-toxic and 100% of borderline (0.4-0.6) messages since they're rare/high-value for active learning, but only ~1% of clearly benign (<0.1) messages.

## Biggest Pitfall
Optimizing purely for accuracy (reaching for GPT-4/large transformers) without first anchoring on the 50ms/20k-QPS constraint — a candidate who doesn't pivot to lightweight models/hybrid filtering when given the latency SLA reads as not understanding the core constraint of the problem.
