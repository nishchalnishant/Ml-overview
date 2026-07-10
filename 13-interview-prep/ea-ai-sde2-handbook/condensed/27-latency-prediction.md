# Interview 27 — Multiplayer Latency Compensation (Time-Series Prediction)

Design an ML model for **client-side prediction** in a multiplayer shooter: predict an enemy's position 100ms ahead so shots feel accurate despite 20-200ms network ping, beating naive linear interpolation/dead reckoning.

## Clarifying Questions to Ask
- Compute budget per inference? → **< 0.1ms/enemy** (10 enemies on screen = can't burn 5ms/frame).
- Why ML over `pos = velocity * time`? → Players strafe/jump/turn abruptly; ML can learn non-linear human movement patterns.
- Do we get the enemy's controller inputs? → No, only X,Y,Z delayed by ping.
- Is ping fixed at 100ms? → No, fluctuates — model must condition on variable lookahead.
- What happens when the server's true position arrives and disagrees? → Needs smooth correction, not a teleport/snap.
- Is this client-authoritative for hit detection? → No — purely visual; server stays authoritative.

## Core Architecture
```
Enemy packets (X,Y,Z @ ping) → Feature Buffer (last 10 frames, deltas + velocity)
     → Shallow GRU (1 layer, 32 units), input [1,10,6] → predicted (dX,dY,dZ)
     → Physics engine clamp (prevent wall clipping) → EMA smoothing → Animation
```
- **Model choice: GRU over LSTM/Transformer** — fewer gates than LSTM (faster), and attention is O(N²) overkill for a 10-frame window.
- Train on delta coordinates, not absolute (avoids activation blowup, keeps values normalized).
- Export to ONNX, run via C++ runtime on main thread; batch all enemies into one matrix call for SIMD.
- LOD scaling: only run ML for enemies within ~50m; use cheap linear math beyond that.

## Talking Points That Signal Seniority
- Proactively separates **visual prediction (client)** from **hit registration (server-side rewind/lag compensation)** — never let ML dictate authoritative hits.
- Flags the **dataset imbalance problem**: most frames are players standing still, so naive MSE training collapses to predicting zero velocity; proposes downsampling/custom loss.
- Insists ML output be **clamped by the physics engine** to stop enemies clipping through walls ("ghosting/overshoot").
- Uses **delta/relative coordinates**, not absolute, and explains why (activation explosion, cache/precision).
- Proposes **batching enemies into one inference call** for SIMD efficiency instead of N separate calls.
- Suggests **LOD-based inference** — skip ML for distant/small-on-screen enemies to save CPU.
- Mentions conditioning the model on a variable **target_delta_t** feature since ping isn't fixed.
- Raises **quantization/pruning (INT8, sparse weights)** for constrained platforms (Switch/mobile).

## Top 3 Tradeoffs
- **GRU vs TCN:** GRU is sequential (slower per-step) but smaller memory footprint; TCN parallelizes via 1D convolutions and is often faster on modern CPUs.
- **Delta vs absolute coordinates:** deltas keep inputs small/stable for the network but can accumulate drift/rounding error without periodic resync to server truth.
- **Kalman filter vs learned model:** Kalman is near-instant and needs no training data, optimal for linear motion, but breaks on sudden human direction changes that a GRU can learn.

## Toughest Follow-ups
**Q: Player's screen shows a perfect hit on the ML-predicted position, but server says the enemy actually turned left — award the hit?**
Server is always authoritative; use **server-side rewind** — rewind the enemy's hitbox to the shooter's timestamp and check collision there. The ML model is purely cosmetic for smooth rendering and never influences hit registration, or you open the door to cheating exploits.

**Q: To fix wall-clipping you want to feed "distance to nearest wall" into the GRU — what's the cost?**
That requires running physics raycasts for every player every frame during data collection, which tanks server performance. Better: keep the model purely kinematic (X,Y,Z) and let the client physics engine handle collision/clamping after prediction.

**Q: Millions of players — train personalized models per player?**
Infeasible to serve millions of models. Use one **global model + per-player embeddings** (e.g., 16-dim "playstyle" vector) fed into the shared GRU, giving personalization without per-user weight storage.

## Biggest Pitfall
Treating the ML prediction as authoritative for hit detection (client-authoritative combat) instead of purely visual — this is a No-Hire-level cheating/security exposure, not just a modeling mistake.
