# Interview 09 — In-Game Store Recommendation Engine (Cheat Sheet)

Build the "For You" carousel (top 6 of 5,000 cosmetic items) for 50M players to maximize purchase conversion. Core challenge: item/user cold start + real-time inventory filtering on top of heavy offline ML.

## Core Architecture
```
Nightly batch: train Two-Tower model → score 50M users × 5K items (FAISS/matrix mult)
   → push Top 100 item IDs/user to Redis
Real-time API: fetch Top 100 from Redis → fetch owned inventory (Redis set)
   → set-difference filter → apply business rules → return Top 6
```
- **Two-Tower / Dual-Encoder** (User tower + Item tower, contrastive loss) — chosen over pure Matrix Factorization because item metadata (rarity, color, hero) solves item cold-start; MF can't recommend an item with no purchase history.
- Redis is the batch→real-time handoff layer (O(1) lookups, TTL for staleness safety).
- Retrieval (Two-Tower, Top 100) → Ranking (heavier model, Top 6) is a two-stage pipeline, industry standard even at only 5K items.

## Talking Points That Signal Seniority
- Proactively distinguishes Retrieval vs Ranking stages, doesn't just say "one model."
- Solves item cold-start via content/metadata embeddings, not just "collect more data."
- Raises position/presentation bias unprompted — trains with slot-position as a feature, fixes it to a constant at inference.
- Mentions diversity re-ranking (MMR) so the 6 slots aren't all near-duplicate skins.
- Flags popularity bias (rich-get-richer) and proposes a penalty/exploration mechanism (bandits) for long-tail items.
- Designs graceful degradation: if inventory service times out, serve infinitely-purchasable consumables instead of failing the request.
- Sets Redis TTL (~48h) so a failed nightly pipeline degrades to stale-but-safe recommendations, not an outage.
- Uses shadow testing (log new model's output, serve old model) instead of a risky direct A/B cutover.

## Top 3 Tradeoffs
- **Batch vs real-time inference**: batch is cheap and 1ms via Redis but up to 24h stale; real-time neural inference captures intent instantly but costs heavily in GPU/latency.
- **Matrix Factorization vs Two-Tower**: MF is simpler, less data engineering, but fails completely on item cold-start; Two-Tower uses metadata to place new items in vector space immediately.
- **Implicit vs explicit feedback**: explicit (ratings) is high-quality but rare; implicit (purchase = positive, impression-without-click = negative) is noisy but abundant — used in practice.

## Biggest Pitfall
Proposing to filter owned items via a per-request SQL JOIN across 50M rows (or otherwise skipping real-time inventory filtering / owned-item exclusion) — this is the single fastest drop from Hire to No Hire, since selling owned cosmetics is a severe, visible production bug.
