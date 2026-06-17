---
module: Deep Learning
topic: Components
subtopic: Quantization Pruning Detailed
status: unread
tags: [deeplearning, ml, components-quantization-prunin]
---
# Quantization and Pruning

How to make large models smaller and faster without significant quality loss. Critical for LLM deployment interviews.

---

## 1. Why Quantization

**Llama-3 70B in fp16:** 140 GB — needs 2× A100 80GB.  
**In int4:** 35 GB — fits on a single A100.  
**Inference speedup:** 1.5–2× from reduced memory bandwidth.

**The memory-bandwidth bottleneck:** LLM decode is bandwidth-bound (load all weights per token). Halving weight size halves the time to load them → proportional speedup.

---

## 2. Quantization Fundamentals

### Uniform Quantization

Map floating-point values to integers in range [−2^(b-1), 2^(b-1)−1]:

$$X_{int} = \text{round}\left(\frac{X_{fp} - z}{s}\right)$$
$$X_{dequant} = X_{int} \cdot s + z$$

where s = scale factor, z = zero point (for asymmetric quantization).

**Scale computation (per-tensor, symmetric):**
$$s = \frac{\max(|X|)}{2^{b-1} - 1}$$

**Scale computation (per-tensor, asymmetric):**
$$s = \frac{\max(X) - \min(X)}{2^b - 1}, \quad z = -\text{round}\left(\frac{\min(X)}{s}\right)$$

**Quantization error:** $\epsilon = X_{fp} - X_{dequant}$. Bounded by $|\epsilon| \leq s/2$ (rounding error).

```python
def quantize_tensor(x: torch.Tensor, bits: int = 8, symmetric: bool = True):
    qmax = 2 ** (bits - 1) - 1
    
    if symmetric:
        scale = x.abs().max() / qmax
        zero_point = 0
        x_int = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    else:
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (2 ** bits - 1)
        zero_point = (-x_min / scale).round().clamp(0, 2**bits - 1).to(torch.int8)
        x_int = (torch.round(x / scale) + zero_point).clamp(0, 2**bits - 1).to(torch.uint8)
    
    return x_int, scale, zero_point

def dequantize(x_int, scale, zero_point):
    return (x_int.float() - zero_point) * scale
```

### Granularity Options

| Granularity | Scale per | Accuracy | Overhead |
|---|---|---|---|
| Per-tensor | Entire weight matrix | Lowest | Minimal |
| Per-channel (row) | Each output neuron row | Good | Small |
| Per-group (g=128) | Every 128 consecutive elements | Best | Small |
| Per-element | Each weight | Highest | Huge (no compression) |

**Per-group quantization** (g=128) is the sweet spot — used by GPTQ, AWQ, QLoRA.

---

## 3. Post-Training Quantization (PTQ)

Quantize a trained model without retraining.

### Naive Round-to-Nearest (RTN)

Simply round weights to nearest quantized value. Fast, but significant quality loss at 4-bit.

### GPTQ (Frantar et al. 2022)

**Key insight:** Use second-order information (Hessian) to minimize quantization error layer-by-layer.

**Objective:** For each weight matrix W, find quantized W_q minimizing:
$$\min_{W_q} ||WX - W_q X||_F^2$$

where X is calibration data activations.

**Algorithm (column-by-column):**
1. Compute Hessian H = 2XX^T (second-order sensitivity)
2. For each column j: quantize w_j to nearest integer, compute error
3. Update remaining columns to compensate: $w_j \leftarrow w_j - \frac{e_j}{[H^{-1}]_{jj}} H^{-1}_{:,j}$
4. Repeat for next column

**Intuition:** When you quantize one weight and introduce error e_j, you can optimally adjust all remaining weights to partially compensate using the Hessian (second derivative = sensitivity to each weight).

```python
def gptq_quantize_layer(W, X, bits=4, blocksize=128):
    """
    W: [out_features, in_features] weight matrix
    X: [in_features, n_calibration_samples] calibration activations
    """
    d = W.shape[1]
    H = 2 * X @ X.T  # Hessian approximation
    
    # Add damping for numerical stability
    H += 0.01 * H.trace() / d * torch.eye(d, device=W.device)
    H_inv = torch.linalg.inv(H)
    
    W_q = W.clone()
    
    for col in range(0, d, blocksize):
        block_end = min(col + blocksize, d)
        
        for j in range(col, block_end):
            w = W_q[:, j]
            # Quantize column j
            scale = w.abs().max() / (2 ** (bits-1) - 1)
            w_q = torch.round(w / scale).clamp(-(2**(bits-1)), 2**(bits-1)-1) * scale
            
            # Compute error
            error = w - w_q
            W_q[:, j] = w_q
            
            # Update remaining columns in block to compensate
            if j + 1 < block_end:
                W_q[:, j+1:block_end] -= (
                    error.unsqueeze(1) @ 
                    H_inv[j, j+1:block_end].unsqueeze(0) / H_inv[j, j]
                )
    
    return W_q
```

### AWQ (Lin et al. 2023)

**Observation:** Not all weights are equally important. A small fraction (salient) weights have large activations — quantizing these causes disproportionate error.

**AWQ idea:** Scale up salient weights before quantization (so they get finer resolution), scale down activations to compensate:

$$\text{Linear}(x; W) = \text{Linear}(x \cdot \mathbf{s}^{-1}; W \cdot \text{diag}(\mathbf{s}))$$

where s is a per-channel scale chosen to minimize quantization error.

**Finding optimal scale:**

$$s_j^* = \arg\min_s ||W_j \cdot s_j \cdot X_j / s_j - W_q(\cdot) \cdot X_j||$$

AWQ searches for s using calibration data, minimizing quantization error on salient channels.

**Advantage over GPTQ:** No layer-by-layer Hessian inversion — faster to apply. Similar quality. Both widely used.

| Method | Quality | Speed to quantize | Inference overhead |
|---|---|---|---|
| RTN | Worst | Fastest (<1min) | None |
| GPTQ | Good | Moderate (~1h for 70B) | None |
| AWQ | Good | Fast (~20min for 70B) | None |
| SmoothQuant | Good (W8A8) | Fast | None |

---

## 4. Quantization-Aware Training (QAT)

Simulate quantization during training using **straight-through estimator (STE)** for gradients.

**Forward pass:** Apply quantize → dequantize (lossy round-trip).  
**Backward pass:** Pass gradients through as if quantization didn't exist (STE):

$$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial W_q}$$

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bits):
        qmax = 2 ** (bits - 1) - 1
        x_int = torch.round(x / scale).clamp(-qmax, qmax)
        return x_int * scale  # dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # STE: pass grad through unchanged

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
    
    def forward(self, x):
        scale = self.linear.weight.abs().max() / (2**(self.bits-1) - 1)
        w_fake_quant = FakeQuantize.apply(self.linear.weight, scale, self.bits)
        return F.linear(x, w_fake_quant, self.linear.bias)
```

**QAT vs PTQ:**
| | PTQ | QAT |
|---|---|---|
| Training required | No | Yes (fine-tuning) |
| Quality at 8-bit | ~same as fp16 | Same as fp16 |
| Quality at 4-bit | ~1–2% degradation | <0.5% degradation |
| Time | Minutes | Hours–days |
| Use case | Quick deployment | Production, latency-critical |

---

## 5. QLoRA

**Problem:** Fine-tuning a quantized model — gradients need to flow through quantized weights.

**QLoRA (Dettmers et al. 2023):** Combine 4-bit NF4 quantization with LoRA adapters.

**NF4 (NormalFloat4):** Quantization scheme optimized for normally distributed weights:
- 4-bit integers mapped to quantiles of normal distribution (not uniform)
- Better preserves weight distribution than uniform int4

**Architecture:**
```
W_frozen (4-bit NF4, ~35GB for 70B) — NOT updated
     +
A·B (LoRA adapters, bf16, ~0.5% of params) — updated

Forward: h = W_frozen · x + (B·A) · x
Backward: gradients only through B·A
```

**Double quantization:** Quantize the quantization constants themselves (scale factors stored in fp32 → quantize to fp8), saving additional ~0.5 bytes/parameter.

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Add LoRA adapters on top
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 33,554,432 || all params: 33,029,857,280 || trainable%: 0.1016
```

**Memory for 70B fine-tuning with QLoRA:**
- Base model (4-bit): ~35GB
- Gradients + optimizer (only LoRA params): ~1GB
- Activations: ~6GB
- **Total: ~42GB — fits on single A100 80GB**

---

## 6. Quantization Formats Comparison

| Format | Bits | Size/param | Quality vs fp16 | Hardware support |
|---|---|---|---|---|
| fp32 | 32 | 4 bytes | Baseline | All |
| bf16 | 16 | 2 bytes | ~Same | Ampere+ (A100) |
| fp8 (e4m3) | 8 | 1 byte | ~Same | Hopper (H100) |
| int8 (W8A8) | 8 | 1 byte | <0.1% loss | All with CUDA |
| int4 (GPTQ/AWQ) | 4 | 0.5 bytes | ~0.5–1% loss | With custom kernels |
| NF4 (QLoRA) | 4 | 0.5 bytes | ~0.5% loss | bitsandbytes |
| int2 | 2 | 0.25 bytes | ~3–5% loss | Research |

**H100 fp8 training:** Hopper GPUs natively support fp8 matrix multiply — 2× throughput over bf16 with hardware-accelerated scaling. Used by DeepSeek-V3 training.

---

## 7. Pruning

### Magnitude Pruning

Remove weights with smallest absolute value:
$$\text{Prune if } |w_{ij}| < \tau$$

**Structured vs unstructured:**
- **Unstructured:** remove individual weights → sparse matrix → hard to accelerate (no hardware support for arbitrary sparsity)
- **Structured:** remove entire neurons/heads/layers → dense smaller model → hardware-friendly

### Iterative Magnitude Pruning (IMP)

1. Train full model to convergence
2. Prune p% of smallest weights
3. Reset remaining weights to initial values ("lottery ticket")
4. Retrain
5. Repeat

**Lottery Ticket Hypothesis (Frankle & Carlin 2019):** Dense networks contain sparse subnetworks ("winning tickets") that can train to full accuracy in isolation.

### Movement Pruning (for fine-tuning)

Instead of pruning by magnitude, prune by gradient × weight (movement from zero):
$$\text{score}_{ij} = w_{ij} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}$$

Weights moving away from zero are important; weights staying near zero are prunable.

### Structured Pruning of Attention Heads

```python
def prune_attention_heads(model, importance_scores, prune_ratio=0.3):
    """Remove least important attention heads."""
    n_to_prune = int(len(importance_scores) * prune_ratio)
    heads_to_prune = sorted(importance_scores, key=lambda x: x[1])[:n_to_prune]
    
    # Group by layer
    layers_to_prune = defaultdict(list)
    for (layer_idx, head_idx), _ in heads_to_prune:
        layers_to_prune[layer_idx].append(head_idx)
    
    model.prune_heads(layers_to_prune)
    return model
```

---

## 8. Knowledge Distillation

Train a small "student" model to mimic a large "teacher" model.

**Distillation loss:**
$$\mathcal{L}_{distill} = \alpha \mathcal{L}_{CE}(y, \hat{y}_{student}) + (1-\alpha) \mathcal{L}_{KL}(\text{softmax}(z_T/T), \text{softmax}(z_S/T))$$

where T = temperature (softens probability distributions), z = logits.

**Why temperature works:** At T=1, softmax is peaked — wrong classes get near-zero probability. At T=4, wrong classes get meaningful probability — this "dark knowledge" teaches the student relative class relationships.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # Hard label loss
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Soft label loss (KL divergence)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    return alpha * ce_loss + (1 - alpha) * kl_loss
```

---

## Canonical Interview Q&As

**Q: Why is per-group quantization better than per-tensor for LLM weights?**  
A: LLM weight matrices have different value ranges in different subsets — some rows have large magnitudes, others are small. Per-tensor quantization uses a single scale for the entire matrix: large values set the scale, forcing small values to round to zero or a few discrete levels. Per-group (g=128) uses a separate scale for every 128 consecutive weights, so each group's range is fully utilized. The tradeoff: per-group stores (d_model/128) extra scale factors per matrix — ~0.5% overhead for d_model=4096. Worth it: per-group quantization reduces quantization error 2–5× vs per-tensor at 4-bit.

**Q: How does GPTQ differ from RTN and why does it perform better at 4-bit?**  
A: RTN rounds each weight independently to the nearest integer — simple but ignores the interaction between weights. GPTQ treats quantization as an optimization problem: given calibration data activations X, minimize ||WX − W_q X||. It processes columns left-to-right, and when a column is quantized (introducing error), the remaining columns are adjusted via the inverse Hessian to partially compensate. This error compensation means GPTQ distributes quantization error across weights, rather than letting each weight's error accumulate independently. Result: ~1–2% less perplexity degradation than RTN at 4-bit for 70B models.

**Q: Why does QLoRA work — how can you fine-tune through a quantized model?**  
A: The key insight is that QLoRA doesn't update the quantized weights. The 4-bit weights are frozen. Training only updates the LoRA adapter matrices A and B (full precision bf16). During forward pass: output = W_4bit·x + B·A·x. Gradients flow through B·A (full precision, differentiable), not through W_4bit (discrete, non-differentiable). The LoRA adapters learn the task-specific corrections while the quantized base model provides the pretrained representations. The only approximation is that W_4bit introduces noise vs the full-precision W — this slightly hurts fine-tuning compared to full fine-tuning, but preserves 99.9% of the base model's capability.

**Q: What's the quality vs memory trade-off when choosing between fp16, int8, and int4 for serving a 70B model?**  
A: fp16: 140GB, highest quality, needs 2× A100s. Int8 (LLM.int8): 70GB, <0.1% quality loss, fits on 1× A100 with ~10% inference overhead. Int4 (GPTQ/AWQ): 35GB, ~0.5–1% quality loss on most benchmarks, fits on 1× A100 with faster inference (lower bandwidth needed). In practice for production: int4 is the standard for deployment when GPU cost matters. For latency-critical paths: fp8 on H100 gives near-fp16 quality with 2× throughput using hardware-native fp8 matrix multiply.
