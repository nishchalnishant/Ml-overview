# Interview 18 — Model Serving on Edge Devices
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on the Mobile team (e.g., EA Sports FC Mobile). We have a deep neural network trained to classify the player's swipe trajectory on the screen to determine if they are passing, shooting, or doing a skill move. 

Currently, the model runs on AWS. The latency (network round-trip) is ~150ms, which is too slow for real-time gameplay. Your task is to **design a system to deploy and run this PyTorch model directly on the player's iOS and Android devices.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Model size & Architecture (Is it a 5MB CNN or a 2GB Transformer?)
- Device constraints (Targeting iPhone 15 Pro only, or 5-year-old Androids?)
- Framework translation (How does PyTorch run on mobile?)
- OTA (Over-The-Air) updates (How do we update the model without forcing an App Store update?)
- Privacy/Security (Can players steal the model binary?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the model size and architecture?"**
   → *Answer: It's a ResNet-based 1D-CNN (time-series). The PyTorch weights are currently 120MB in FP32.*

2. **"What are the target devices?"**
   → *Answer: Must run at 60 FPS on mid-tier devices (e.g., iPhone 11, Samsung Galaxy S20).*

3. **"Do we need to update the model frequently?"**
   → *Answer: Yes. We tune the gameplay physics every 2 weeks. The model needs to update without requiring a full App Store release.*

4. **"Is it a problem if a competitor extracts our model file from the game client?"**
   → *Answer: Yes, the game logic is proprietary. We need some basic obfuscation.*

---

## Part 4 — Expected Assumptions

- **Conversion:** PyTorch -> ONNX -> CoreML (iOS) / TFLite (Android). Or using ExecuTorch / PyTorch Mobile.
- **Optimization:** 120MB is too large for mobile RAM budgets. Must use Quantization (INT8) or Pruning.
- **Delivery:** OTA update pipeline using AWS S3 and presigned URLs.

---

## Part 5 — High-Level Solution

```
  [Offline ML Pipeline]
  PyTorch Model (120MB, FP32)
       │
       ▼
  Post-Training Quantization (INT8) ➔ Shrinks to ~30MB
       │
       ▼
  Export to ONNX (Universal Format)
       │
       ▼
  [Platform Compilation]
  iOS ➔ coremltools ➔ .mlmodelc
  Android ➔ ONNX Runtime Mobile / TFLite
       │
       ▼
  [OTA CDN (AWS CloudFront / S3)]
  Stores encrypted model binaries.
  
       =========================================================

  [Game Client (iOS/Android)]
  1. Boot up ➔ Check CDN for new Model Version.
  2. Download & Decrypt model to local disk.
  3. Load model into NPU (Neural Processing Unit) or GPU via CoreML / NNAPI.
  4. Gameplay: Capture Swipe ➔ Inference (< 5ms) ➔ Action.
```

**Core ML Component:** Translating a server-side PyTorch graph into an optimized edge format (ONNX/CoreML) and quantizing it to INT8 without losing accuracy.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Quantization
- Converting 32-bit floats to 8-bit integers reduces the model size by exactly 4x (120MB -> 30MB) and massively speeds up inference on mobile processors (especially NPUs).
- Use Post-Training Quantization (PTQ). If accuracy drops too much, switch to Quantization-Aware Training (QAT).

### Step 2: ONNX Export
- Trace the PyTorch model with a dummy input tensor matching the expected swipe shape (e.g., `batch=1, channels=3, sequence_length=60`).
- Export to `.onnx`.

### Step 3: Platform Execution
- **iOS:** Use Apple's `coremltools` to convert ONNX to CoreML format. CoreML automatically targets the Apple Neural Engine (ANE) for zero-battery-drain inference.
- **Android:** Use `ONNX Runtime` C++ API integrated via JNI, utilizing the NNAPI execution provider for hardware acceleration.

---

## Part 7 — Complete Python Code

```python
"""
edge_exporter.py - Quantize and Export PyTorch Model for Mobile
"""
import logging
import torch
import torch.nn as nn
import torch.quantization
import onnx
import coremltools as ct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dummy Model (1D CNN for Swipe Time-Series)
# ---------------------------------------------------------------------------
class SwipeClassifier(nn.Module):
    def __init__(self):
        super(SwipeClassifier, self).__init__()
        # Input: [Batch, Channels(x,y,pressure), TimeSteps(60)]
        self.conv = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 60, 3) # 3 actions: Pass, Shoot, Skill

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------
def quantize_model(model: nn.Module) -> nn.Module:
    """Applies Post-Training Dynamic Quantization (FP32 -> INT8)"""
    logger.info("Quantizing model to INT8...")
    
    # Dynamic quantization targets Linear and RNN layers.
    # For Conv layers, Static Quantization is better, but requires calibration data.
    # We use dynamic here for simplicity.
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def export_to_onnx(model: nn.Module, save_path: str):
    """Exports the PyTorch graph to ONNX."""
    logger.info(f"Exporting ONNX to {save_path}...")
    model.eval()
    
    # Dummy input representing one 60-frame swipe
    dummy_input = torch.randn(1, 3, 60)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14, # Stable opset for mobile
        do_constant_folding=True,
        input_names=['swipe_input'],
        output_names=['action_logits'],
        dynamic_axes={'swipe_input': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
    )

def export_to_coreml(onnx_path: str, save_path: str):
    """Converts ONNX to Apple CoreML for ANE (Neural Engine) execution."""
    logger.info(f"Converting ONNX to CoreML: {save_path}...")
    
    onnx_model = onnx.load(onnx_path)
    
    # coremltools translates the ONNX graph to an MLProgram
    mlmodel = ct.converters.onnx.convert(
        onnx_model,
        minimum_ios_deployment_target='15'
    )
    mlmodel.save(save_path)

if __name__ == "__main__":
    model = SwipeClassifier()
    
    # In reality, you'd load trained weights here
    # model.load_state_dict(torch.load("model_fp32.pth"))
    
    # 1. Quantize
    q_model = quantize_model(model)
    
    # 2. Export ONNX (for Android / Windows)
    export_to_onnx(model, "swipe_model.onnx") # Note: ONNX export of PyTorch qint8 is tricky, often better to export FP32 and quantize via ONNX Runtime tools.
    
    # 3. Export CoreML (for iOS)
    # export_to_coreml("swipe_model.onnx", "SwipeModel.mlpackage")
```

---

## Part 8 — Deployment

### OTA (Over-The-Air) Updates
- The build pipeline outputs `swipe_model_v2.onnx` and encrypts it (AES-256).
- Uploads to AWS CloudFront.
- Game Client API calls `/v1/meta/config` on boot. It returns `{"swipe_model_version": "v2", "url": "https://cdn.ea..."}`.
- If the local version is `v1`, the client downloads the file, decrypts it in memory, and writes it to the app's secure sandbox directory.

---

## Part 9 — Unit Testing

```python
import torch
import onnxruntime as ort
import numpy as np

def test_onnx_inference():
    # Verify the ONNX export produces identical results to PyTorch
    
    # 1. PyTorch Baseline
    pt_model = SwipeClassifier()
    pt_model.eval()
    dummy_input = torch.randn(1, 3, 60)
    pt_out = pt_model(dummy_input).detach().numpy()
    
    # 2. Export ONNX
    torch.onnx.export(pt_model, dummy_input, "test.onnx")
    
    # 3. ONNX Runtime Inference
    session = ort.InferenceSession("test.onnx")
    ort_out = session.run(None, {"swipe_input": dummy_input.numpy()})[0]
    
    # Assert they are mathematically identical within float tolerance
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-03, atol=1e-05)
```

---

## Part 10 — Integration Testing

- **On-Device Benchmarking (AWS Device Farm):**
  - Run the ONNX model on real hardware (iPhone 11, Samsung S20).
  - Assert that `p99_latency_ms < 5ms`.
  - Assert that memory consumption (RAM spike during inference) is `< 50MB`. (If it spikes to 500MB, the OS will kill the game client).

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Device Fragmentation** | Android has thousands of hardware profiles. Some have NPUs, some don't. ONNX Runtime handles this gracefully via "Execution Providers". It attempts to use NNAPI (Hardware). If NNAPI fails or crashes, it falls back to XNNPACK (Optimized CPU). |
| **Model Size Limits** | If 30MB (INT8) is still too large for the OTA budget, implement **Knowledge Distillation**. Train a massive 500MB Teacher model on AWS, and use it to train a tiny 5MB Student model (MobileNet) to mimic its outputs. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Edge vs Cloud ML | Edge gives zero latency (perfect for gameplay physics) and zero AWS compute costs. Cloud is vastly easier to update, handles massive models, and hides intellectual property. |
| ONNX Runtime vs PyTorch Mobile | PyTorch Mobile keeps everything in the PyTorch ecosystem, but binary size is huge (you have to ship the libtorch C++ library in the app). ONNX Runtime is heavily optimized by Microsoft for mobile, has smaller binaries, and connects easily to CoreML/NNAPI. |
| Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT) | PTQ is a 1-line script but can cause accuracy to drop by 2-5%. QAT requires modifying the PyTorch training loop to simulate INT8 math during training. It maintains 99% accuracy but adds engineering complexity. |

---

## Part 13 — Alternative Approaches

1. **Server-Side Fallback:** If the edge model crashes or the device is too old, disable the feature, or fall back to a simple hardcoded heuristic (e.g., standard swipe math) rather than making a network call.
2. **Federated Learning:** Keep the model on device, but allow the device to train/fine-tune the model slightly based on the specific player's swipe habits. Send the weight diffs (not the raw data) back to EA to improve the global model.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| OOM (Out of Memory) Crash | The OS kills the game because the model loaded into RAM alongside heavy 3D graphics. | Memory map (mmap) the `.onnx` file. Instead of loading the whole 30MB file into RAM, the OS pages it into memory only when needed. |
| Corrupt OTA Download | The user is on bad WiFi, the model file downloads half-way, and crashes on load. | Always download to a `.tmp` file. Verify the SHA-256 hash against the server. Only rename to `.onnx` if the hash matches. |
| Float16 vs Float32 iOS bug | CoreML converts FP32 to FP16 automatically. Sometimes this causes NaN (Not a Number) explosions in deep networks. | Test the exported CoreML model specifically on Apple Silicon hardware before deploying. |

---

## Part 15 — Debugging

**Symptom:** The model works perfectly on iOS, but on Android, the predictions are complete garbage (random actions).

**Debugging steps:**
1. Is it a hardware execution bug? Force ONNX Runtime to use the CPU provider instead of NNAPI. If it works on CPU, it's a driver bug on that specific Android phone.
2. Is it a preprocessing bug? iOS might be passing the swipe data as `Float32`, but the Java/JNI layer on Android is mistakenly passing `Float64` or un-normalized pixels (0-255 instead of 0-1).
3. **Fix:** Ensure the C++ binding layer that feeds the tensor enforces strict data typing and normalization identical to the Python training script.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `edge_inference_latency_ms` | > 16ms (1 frame at 60fps) → The model is stuttering the game thread. |
| `ota_download_success_rate` | < 95% → CDN or hash verification issue. |
| `model_fallback_rate` | > 5% → Devices are failing to initialize the NPU. |

---

## Part 17 — Production Improvements

1. **Async Inference:** Do not run the ONNX session on the main UI/Render thread. If inference takes 8ms, the game will drop frames. Pass the swipe buffer to a background C++ thread, run inference, and return the callback to the main thread.
2. **A/B Testing:** Send `model_v1` to 50% of devices and `model_v2` to 50%. Track the "Goal Scoring Rate" via standard game telemetry to see if `v2` is actually interpreting swipes better.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"Quantizing to INT8 ruined our accuracy. It dropped from 95% to 70%. We can't use it, but we still need the model to be 4x smaller. What specific PyTorch techniques do you use to fix this?"**
2. **"Hackers jailbreak their iPhone, find your `swipe_model.mlmodelc`, and reverse engineer it to build an auto-aim cheat bot. How do you protect the model architecture?"**
3. **"The C++ Game Engine needs to pass a multi-dimensional array to the ONNX model. Describe how memory is managed here—are we copying memory from the game engine into ONNX Runtime, and why is that a problem?"**

---

## Part 19 — Ideal Answers

**Q1 (Accuracy drop in PTQ):**
> "An aggressive accuracy drop means certain activation layers have extreme outliers (e.g., massive values that get crushed when scaling to -128 to 127). First, I would switch to **Quantization-Aware Training (QAT)** to let the model learn the rounding errors during backprop. If that fails, I would use **Mixed Precision**: keep the sensitive layers (like the first Conv layer and the final Softmax) in FP32, and only quantize the dense hidden layers to INT8."

**Q2 (Security / IP Protection):**
> "Once a binary is on a user's device, it is fundamentally vulnerable. However, we can heavily obfuscate it. 
> 1. Store the model encrypted on disk and only decrypt it into volatile RAM at runtime. 
> 2. Use a custom ONNX Runtime build with proprietary operator names (so tools like Netron can't render the graph visually). 
> 3. Ultimately, we must move critical security logic (cheat detection) server-side, and only put UX-improving models (like swipe recognition) on the edge."

**Q3 (Memory Management / Zero-Copy):**
> "If the Game Engine allocates a buffer for the swipe, and we copy it into an ONNX tensor, we waste CPU cycles and RAM. We should use **Zero-Copy Memory Mapping**. We allocate the tensor memory *using the ONNX Runtime allocator*, and pass a raw pointer to the Game Engine. The engine writes the swipe data directly into the ONNX memory block, avoiding the `memcpy` overhead entirely."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the difference between PyTorch (Training framework) and ONNX/CoreML (Execution graphs).
- Clearly articulates how Quantization works (FP32 -> INT8) and knows the fallback (QAT).
- Discusses Zero-Copy C++ memory management and thread blocking (Q3).
- Proposes an encrypted OTA pipeline to bypass App Store limits.

### Hire
- Successfully designs the export pipeline.
- Mentions ONNX Runtime or TFLite.
- Understands that a 120MB model is too big for edge inference.

### Lean Hire
- Suggests exporting a Flask API to run in a Docker container on the phone. (Technically possible, but horrific for battery and memory).
- Needs prompting to understand how to shrink the model.

### Lean No Hire
- Thinks they can just use `torch.load()` inside the iOS Swift app.
- Doesn't understand the difference between GPU, CPU, and NPU execution on mobile.

### No Hire
- Fails to grasp edge vs cloud computing.
- Cannot write the PyTorch export code.
