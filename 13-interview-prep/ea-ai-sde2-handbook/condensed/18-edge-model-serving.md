# Interview 18 — Model Serving on Edge Devices (Cheat Sheet)

Deploy a PyTorch swipe-classification model (pass/shoot/skill) directly onto iOS/Android game clients to kill the ~150ms cloud round-trip and hit real-time gameplay latency.

## Clarifying Questions to Ask
- Model size/architecture? → ResNet-style 1D-CNN, 120MB FP32.
- Target devices? → Must hit 60 FPS on mid-tier (iPhone 11, Galaxy S20), not just flagship.
- Update cadence? → Gameplay physics retuned every 2 weeks; needs OTA, not App Store release.
- Is model theft a concern? → Yes, proprietary game logic; needs obfuscation/encryption.
- Is a fallback needed for old/unsupported devices? (not in source but worth raising) → heuristic fallback, no network call.

## Core Architecture
```
PyTorch (FP32, 120MB) → PTQ/QAT → INT8 (~30MB) → ONNX export
        → iOS: coremltools → .mlmodelc (runs on ANE)
        → Android: ONNX Runtime Mobile (NNAPI, fallback XNNPACK CPU)
   → Encrypted binary on CDN (CloudFront/S3) → client OTA fetch + hash verify
   → Loaded on boot, memory-mapped, inference on background thread (<5ms)
```
- Key technique: Post-Training Quantization FP32→INT8 (4x size cut, big NPU speedup); QAT as fallback if accuracy degrades.
- ONNX as universal IR so one export path feeds both CoreML (iOS/ANE) and ONNX Runtime Mobile (Android/NNAPI).

## Talking Points That Signal Seniority
- Distinguishes training framework (PyTorch) from execution graph formats (ONNX/CoreML) explicitly.
- Proactively proposes QAT as the fix when PTQ accuracy drop is unacceptable, not just "quantize harder."
- Raises zero-copy memory management between C++ game engine and ONNX Runtime tensors (avoids memcpy on hot path).
- Runs inference on a background thread, not the render/UI thread — flags 16ms/frame budget at 60fps.
- Proposes encrypted OTA pipeline (AES-256, SHA-256 hash verify before rename) to update models without App Store review.
- Mentions execution-provider fallback chain (NNAPI → XNNPACK CPU) for Android device fragmentation.
- Suggests knowledge distillation (large teacher → tiny student) if even INT8 is still too large for OTA budget.
- Flags model-IP theft risk and proposes obfuscated ops/custom runtime build, while conceding on-device binaries are fundamentally extractable — pushes truly sensitive logic server-side.

## Top 3 Tradeoffs
- Edge vs Cloud ML: edge = zero latency + zero inference cost but exposes IP and is harder to patch; cloud is easy to update/protect but adds network latency, ruled out here by the 150ms requirement.
- ONNX Runtime vs PyTorch Mobile: ONNX Runtime gives smaller binaries and native CoreML/NNAPI hooks; PyTorch Mobile keeps one ecosystem but forces shipping libtorch, bloating the app.
- PTQ vs QAT: PTQ is a one-line script but can cost 2-5% accuracy; QAT preserves accuracy by simulating INT8 math during training at the cost of training-loop complexity.

## Toughest Follow-ups
**Q: PTQ dropped accuracy from 95% to 70% — still need 4x smaller, what now?**
Outlier activations are getting crushed by the INT8 scale range. Switch to QAT so the model learns to compensate for rounding during backprop. If that's insufficient, use mixed precision — keep the first conv layer and final softmax in FP32, quantize only the dense hidden layers.

**Q: Hackers jailbreak a device and reverse-engineer the extracted model into a cheat bot — how do you protect it?**
Accept that any on-device binary is ultimately extractable. Mitigate: store encrypted on disk, decrypt only into volatile RAM at runtime, use custom/renamed ONNX operators to defeat graph viewers like Netron. Structurally, keep genuinely sensitive logic (cheat detection) server-side — only ship UX-improving models to the edge.

**Q: How is memory managed when the C++ engine passes a multi-dim array into ONNX — are we copying, and why does it matter?**
A naive path copies the engine's buffer into a new ONNX tensor, wasting CPU/RAM on every frame at 60fps. Use zero-copy: allocate the tensor via the ONNX Runtime allocator and hand the engine a raw pointer to write into directly, eliminating the memcpy.

## Biggest Pitfall
Treating this as "just export the PyTorch model and load it with `torch.load()` in Swift/Kotlin" — missing that mobile requires a real conversion/quantization pipeline (ONNX/CoreML/TFLite) and NPU-vs-CPU execution awareness is an instant No-Hire signal.
