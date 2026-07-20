# Interview 18 — Model Serving on Edge Devices (Cheat Sheet)

Deploy a PyTorch swipe-classification model (pass/shoot/skill) directly onto iOS/Android game clients to kill the ~150ms cloud round-trip and hit real-time gameplay latency.

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

## Biggest Pitfall
Treating this as "just export the PyTorch model and load it with `torch.load()` in Swift/Kotlin" — missing that mobile requires a real conversion/quantization pipeline (ONNX/CoreML/TFLite) and NPU-vs-CPU execution awareness is an instant No-Hire signal.
