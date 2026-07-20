# Interview 03 — Toxic Chat Moderation (Cheat Sheet)

Real-time microservice to detect/block toxic messages (hate speech, harassment) in in-game chat across multiple titles before they reach other players. Core tension: sub-50ms latency at 20k QPS vs. accuracy needed to catch obfuscated/nuanced toxicity.

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

## Biggest Pitfall
Optimizing purely for accuracy (reaching for GPT-4/large transformers) without first anchoring on the 50ms/20k-QPS constraint — a candidate who doesn't pivot to lightweight models/hybrid filtering when given the latency SLA reads as not understanding the core constraint of the problem.
