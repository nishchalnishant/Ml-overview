# Interview 09 — In-Game Store Recommendation Engine (Cheat Sheet)

Build the "For You" carousel (top 6 of 5,000 cosmetic items) for 50M players to maximize purchase conversion. Core challenge: item/user cold start + real-time inventory filtering on top of heavy offline ML.

## Clarifying Questions to Ask
- Can we recommend items the player already owns? → No, must filter owned inventory.
- Real-time or batch scoring? → Batch nightly ML, but real-time filtering (bought item vanishes instantly).
- Do we have negative signals (impressions, not just purchases)? → Yes, impression logs exist.
- How do we handle brand-new items with zero history? → Weekly new skins, no data — candidate must propose a fix.
- Scale? → 50M active players, 5,000 items.

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

## Toughest Follow-ups
**Q: How do you handle a brand-new player (user cold start)?**
No purchase history for the User Tower, so fall back to onboarding features (platform, region, time of day) or global popularity. Better: prompt the user to pick 3 favorite characters at first login and use that as the seed input to generate an initial embedding.

**Q: How does inventory filtering handle bundles (e.g., weapon+character skin bundle where user owns one component)?**
Simple set-difference (`item not in owned`) breaks here. The serving API must decompose the bundle into component IDs: filter the bundle only if all components are owned; if partially owned, keep it but flag for dynamic pricing. The ML model just ranks the bundle ID — the API owns the entitlement logic.

**Q: Slot 1 gets 5x the clicks of Slot 6 regardless of item — how does this bias affect training?**
Classic position/presentation bias — the model conflates slot position with item quality. Fix: include UI position as a training feature so the model learns intrinsic item value separately from slot value, then set position to a constant at inference time to isolate true relevance.

## Biggest Pitfall
Proposing to filter owned items via a per-request SQL JOIN across 50M rows (or otherwise skipping real-time inventory filtering / owned-item exclusion) — this is the single fastest drop from Hire to No Hire, since selling owned cosmetics is a severe, visible production bug.
