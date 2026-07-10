# Anti-Cheat: Client-side Memory Anomaly Detection — Cheat Sheet

EAAC's kernel driver collects mouse/memory telemetry to catch aimbots/wallhacks that mimic human behavior. Design an ML system that runs **on the game client** (not just server) under brutal CPU/RAM limits, with a safe escalation path so false positives don't instant-ban innocent players.

## Clarifying Questions to Ask
- Compute budget for the client model? → **<1% CPU, <50MB RAM** — game needs the resources.
- What features does the driver actually collect? → Mouse (x,y,t) trajectories, `ReadProcessMemory` call frequency from other processes, driver signatures.
- Do we ban instantly off the client model? → **No** — client only triggers; uploads telemetry to a heavier cloud model for the actual ban decision.
- How do we handle legit high-skill/high-DPI players? → Must normalize features by sensitivity/DPI/resolution or pros get flagged.
- Can we trust the OS-level signal at all? → No — DMA hardware cheats bypass the OS/kernel driver entirely.

## Core Architecture
```
Kernel Driver (mouse x,y,t + memory reads)
   → Local Feature Extraction (C++): jerk, acceleration, angle variance
   → Local ML Inference (LightGBM, every 10s, <0.1% CPU)
   → if P(cheat) > 0.95: encrypt 30s buffer → upload
   → Cloud: heavy ensemble/NN reviews → delayed ban wave
```
- **Model choice: LightGBM/shallow trees**, not a CNN/LSTM/Transformer — trees run in microseconds, ~1MB footprint, no GPU needed on client.
- **Key feature insight**: humans move mice in curved, accelerating paths (Fitts's Law); aimbots snap in straight lines with unnatural jerk — jerk & angle-variance are the signal.
- Client is a tripwire only; ground truth ban decision happens server-side on richer data.
- Model file shipped encrypted (AES-256), key delivered at runtime handshake — client-side model is an attack surface.
- Model updates pushed via lightweight OTA `.txt` file swap, not a full game patch.

## Talking Points That Signal Seniority
- Proactively reject deep learning on-client — name the CPU/RAM budget constraint before being told the numbers.
- Bring up Fitts's Law / physics-based features unprompted (jerk, curvature, acceleration) as the aimbot signal.
- Say false positives (banning innocents) are worse than false negatives — drives a "flag, don't ban" architecture.
- Propose **ban waves** (delayed bans) explicitly to break the cheat dev's feedback loop on which trigger fired.
- Mention model file encryption + integrity heartbeat as a tamper-resistance mechanism, not just an afterthought.
- Propose honeypots (invisible bots) as a near-zero-false-positive wallhack detector.
- Flag that DMA hardware cheats defeat any OS/kernel-level signal, so server-side behavioral analysis is the fallback of last resort.
- Note that streaming raw 60Hz telemetry for all players doesn't scale — only upload on local trigger (~99.9% traffic reduction).

## Top 3 Tradeoffs
- **LightGBM vs LSTM/CNN on-device**: trees give near-zero perf cost; neural nets are more accurate on nuanced behavior but cost ~30MB RAM and CPU cycles the game can't spare — accuracy traded for zero frame-rate impact.
- **Instant ban vs ban waves**: instant bans let cheat devs reverse-engineer exactly which feature tripped detection; delayed waves obfuscate the trigger at the cost of a window where cheaters keep playing.
- **Edge-only vs server-only detection**: edge catches cheap/local cheats cheaply but can be blinded by DMA hardware bypassing the OS; server-only behavioral analysis is bypass-proof but needs constant telemetry and heavier compute.

## Toughest Follow-Ups
**Q: Cheat dev deletes the model file on launch so it never runs — how do you stop this?**
A: Server expects a signed heartbeat every 60s containing a hash of the model file. Missing/invalid heartbeat → treat as anti-cheat auth failure and kick/ban, independent of whether any cheat behavior was observed.

**Q: A DMA PCIe card reads memory physically, bypassing the OS/kernel driver entirely — how does the pipeline catch it?**
A: Give up on OS-level signals for this case; the cheat still has to act in-game. Shift to server-side behavioral analysis — track crosshair placement vs. actual enemy positions (e.g., perfect tracking through walls). Behavior is anomalous regardless of what hardware read the memory.

**Q: LSTMs are slow/sequential and suffer vanishing gradients — what would you use for edge time-series instead?**
A: Temporal Convolutional Networks (dilated 1D convolutions) — parallelizable unlike LSTMs, no vanishing gradient issue, and compile efficiently to ONNX/TFLite for edge deployment.

## Biggest Pitfall
Proposing a heavy on-device model (Transformer/CNN/LSTM) that ignores the extreme client CPU/RAM budget, or treating false positives as an acceptable cost — either one signals no real-world game-client experience and is a No-Hire.
