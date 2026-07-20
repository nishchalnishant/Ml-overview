---
module: LLMs
topic: Fine Tuning At Scale
subtopic: ""
status: unread
tags: [llms, ml, fine-tuning-at-scale]
---
# Fine-tuning at Scale

How to fine-tune LLMs efficiently — from LoRA math to QLoRA to full fine-tuning with ZeRO. Essential for Staff/L6 interviews at companies running training workloads.

---

## 1. Memory Budget for Fine-tuning

Before choosing a method, understand the memory footprint.

**Per-parameter memory (fp32 Adam):**
- Model weights (bf16): 2 bytes
- Gradients (bf16): 2 bytes
- Optimizer states (fp32 master weights + momentum + variance): 12 bytes
- Activations (depends on batch/seq size)
- **Total: ~16 bytes/param** (before activations)

**Concrete examples:**

| Model | Parameters | Weights | + Grads | + Adam | Min GPUs (80GB A100) |
|---|---|---|---|---|---|
| Llama-3 8B | 8B | 16 GB | 32 GB | 128 GB | 2 (ZeRO-3) |
| Llama-3 70B | 70B | 140 GB | 280 GB | 1120 GB | 15 (ZeRO-3) |
| Llama-3 405B | 405B | 810 GB | 1.6 TB | 6.5 TB | ~82 (ZeRO-3) |

**With LoRA (rank=16, only 0.1% of params trainable):**
- 8B model: ~20 GB total (weights frozen in bf16 + small LoRA grads/optimizer)
- Fits on a single 24GB GPU

---

## 2. LoRA: Low-Rank Adaptation

### Mathematical Foundation

Full fine-tuning updates all W ∈ ℝ^{d×k}. LoRA constrains updates to a low-rank decomposition:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

**Initialization:** A initialized with Gaussian random, B initialized to zero → $\Delta W = 0$ at start, so training begins from pretrained weights.

**Scaling:** LoRA adds a scaling factor to control the magnitude of the adaptation:

$$W' = W_0 + \frac{\alpha}{r} BA$$

where $\alpha$ is a hyperparameter (typically set equal to $r$, so $\alpha/r = 1$). Higher $\alpha/r$ → stronger adaptation signal.

**Parameter count reduction:**
```
Original W: d × k parameters
LoRA: (d + k) × r parameters

Llama-3 8B attention Q projection: 4096 × 4096 = 16.8M params
LoRA rank=16: (4096 + 4096) × 16 = 131K params  →  0.78% of original
```

### LoRA Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        
        d, k = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, k) * (1.0 / rank**0.5))
        self.lora_B = nn.Parameter(torch.zeros(d, rank))
        
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
    
    def forward(self, x):
        base_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + (self.alpha / self.rank) * lora_out
    
    def merge(self):
        """Merge LoRA weights back into original for inference (no overhead)."""
        delta_W = (self.alpha / self.rank) * self.lora_B @ self.lora_A
        self.original.weight.data += delta_W
        return self.original

def apply_lora_to_model(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"]):
    """Apply LoRA to target projection layers."""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank, alpha))
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")
    return model
```

### PEFT Library (Production Usage)

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # rank
    lora_alpha=32,                 # scaling (alpha/r = 2)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # all attention + FFN
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 8,030,261,248 || trainable%: 1.04%
```

---

## 3. LoRA Rank Selection

**Rank = expressiveness vs efficiency trade-off.**

| Rank | Trainable params | Use case |
|---|---|---|
| 1–4 | Very few | Simple style transfer, format adaptation |
| 8–16 | ~0.5–1% | General instruction following (most common) |
| 32–64 | ~2–4% | Complex domain adaptation |
| 128–256 | ~8–16% | Near-full fine-tuning expressiveness |
| Full | 100% | When data is abundant and budget allows |

**Intrinsic dimensionality argument:** pre-trained LLMs have low intrinsic dimensionality for task adaptation — the effective dimension of fine-tuning updates is often < 100. Rank 16 is usually sufficient.

**Rule of thumb:** Start with r=16. If validation loss plateaus early and training loss is much lower, increase rank. If training is unstable, decrease rank and/or learning rate.

---

## 4. QLoRA: Fine-tuning with Quantized Base Model

**Key insight:** freeze the base model in 4-bit (NF4) quantization — saves memory for the large frozen model — while training LoRA adapters in bf16. Double quantization further compresses the quantization constants.

### Memory Analysis

```
Llama-3 70B with QLoRA:
  Base model (NF4, 4-bit): 70B × 0.5 bytes = 35 GB
  Double quantization savings: ~2.5 GB saved
  LoRA adapters (bf16, r=16): ~3 GB
  Optimizer states (Adam, only LoRA params): ~6 GB
  Activations (seq=512, batch=4): ~5 GB
  Total: ~40 GB → fits in 1 A100 80GB
```

Compare to full fine-tuning: 70B × 16 bytes = 1.12 TB minimum.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — optimal for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in nf4
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training (handles gradient issues with quantized weights)
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)
```

### NF4 (NormalFloat4)

NF4 is designed for weights that follow a normal distribution (which pre-trained LLM weights do, by induction/training dynamics). It places quantization levels non-uniformly:

$$q_i = Q_N^{-1}\left(\frac{i}{2^k - 1}\right) \quad \text{where } Q_N \text{ is the normal CDF}$$

This makes quantization levels denser near the mean (where most weight values are) rather than uniformly spaced, minimizing quantization error for normally distributed weights.

---

## 5. Post-Training Quantization: GPTQ, AWQ, GGUF

QLoRA (Section 4) quantizes for *training*. Post-training quantization (PTQ) quantizes an already-trained model for *inference*, making large models deployable on smaller hardware.

### GPTQ

**Core idea:** minimize the per-layer weight quantization error using second-order (Hessian) information. For each layer, instead of rounding every weight independently, GPTQ quantizes one column at a time and compensates the remaining columns for the error introduced.

**Algorithm (one layer):**
```
For each column j in W:
    q_j = round(w_j / scale)          # quantize column j
    error = w_j - q_j * scale         # compute introduced error
    W[:, j+1:] -= error * H⁻¹[:, j+1:] * H[j, j+1:]  # propagate correction
```
where H = X^T X is the activation Hessian (computed from a calibration set of ~128 sequences).

**Properties:**
- One-shot: does not require gradient computation after the initial calibration pass
- Achieves good quality at INT4 (W4A16): for 7B models, ~2-3% perplexity increase vs fp16
- 4× memory reduction: 70B → 35 GB fp16 → ~8.5 GB INT4
- No inference speedup on CUDA cores (weights dequantized to fp16 before matmul), but significant VRAM savings enable larger batch sizes

```python
# Using AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,    # quantize in groups of 128 weights; finer = better quality
    damp_percent=0.01, # Hessian damping for numerical stability
)
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(calibration_dataset)
model.save_quantized("./model-gptq-4bit")
```

**Key hyperparameter:** `group_size`. With `group_size=128`, each group of 128 weights shares one scale and one zero-point. Smaller groups → better quality (less quantization error per group) but more overhead (one scale per 128 weights = 0.4% overhead at INT4).

---

### AWQ (Activation-aware Weight Quantization)

**Problem with GPTQ:** it treats all weight channels equally. But some channels are significantly more important than others — specifically, channels that multiply large activation values, because the output error = weight error × activation value. A 1% error in a channel with activation magnitude 10 causes 10× more output error than the same weight error in a channel with activation magnitude 1.

**Core idea:** identify "salient" weight channels (those paired with large activations via calibration), and scale those channels up before quantization to protect them — effectively using fewer quantization bits for less important channels and more precision for important ones. Applied via a per-channel scaling factor before quantization.

**Algorithm:**
```
1. Run calibration data → compute per-channel activation statistics: s_c = mean(|X_c|)
2. For each weight channel c: scale w_c by s_c^α (α ≈ 0.5 found by grid search)
3. Quantize the scaled weights (salient channels are now more "spread out" in the quantization range)
4. Compensate: divide activations by s_c at inference (fuse into LayerNorm)
```

**Properties vs GPTQ:**
- No Hessian computation: uses only activation statistics (faster calibration)
- Typically better quality than GPTQ at INT4, especially for reasoning tasks
- Hardware-friendly: the channel scaling can be fused into preceding LayerNorm, adding zero inference overhead
- AWQ is the default in many production serving stacks (vLLM natively supports AWQ)

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("./model-awq-4bit")
```

---

### GGUF (GPT-Generated Unified Format)

**Context:** GGUF is not a quantization algorithm — it is a binary file format designed for portable, efficient CPU/GPU inference via **llama.cpp**. It packages model weights, tokenizer, and model metadata in a single file with memory-mapped access.

**Quantization schemes in GGUF:** uses k-quant formats that quantize blocks of weights together with per-block scales:

| Format | Bits/weight | Size (7B) | Quality vs fp16 | Use case |
|---|---|---|---|---|
| Q2_K | ~2.6 | ~2.7 GB | Large degradation | Extreme memory constraint |
| Q3_K_M | ~3.4 | ~3.5 GB | Moderate degradation | Minimum viable |
| Q4_K_M | ~4.5 | ~4.8 GB | Small degradation | **Best size/quality balance** |
| Q5_K_M | ~5.7 | ~5.7 GB | Minimal degradation | High quality, fits 8GB RAM |
| Q6_K | ~6.6 | ~6.1 GB | Near-lossless | When quality is critical |
| Q8_0 | ~8.5 | ~7.7 GB | Effectively lossless | Maximum quality, limited RAM |

**K-quant mechanics (Q4_K_M):** weights are organized into super-blocks of 256 weights; each super-block contains 8 sub-blocks of 32 weights with individual scales. The "M" (medium) variant uses 4-bit quantization for most layers but 6-bit for specific sensitive layers (embedding, output, some attention layers), balancing size and quality.

**Memory-mapped access:** GGUF weights are read directly from disk pages into GPU/CPU memory on demand — the OS page cache handles repeated access. This enables loading a 4-bit 70B model (≈35 GB) incrementally on a machine with 24 GB RAM if you're willing to accept slower inference (disk reads on cache miss).

**CPU inference:** llama.cpp can run GGUF models entirely on CPU using BLAS-optimized matrix operations. For 7B Q4_K_M on a modern CPU (Apple M2 Pro), ~10-15 tokens/second is achievable — viable for local development without a GPU.

```bash
# Create GGUF from HuggingFace model
python convert_hf_to_gguf.py ./llama-3-8b --outtype f16 --outfile llama3-8b-f16.gguf
./llama-quantize llama3-8b-f16.gguf llama3-8b-q4_k_m.gguf Q4_K_M

# Serve via llama.cpp
./llama-server -m llama3-8b-q4_k_m.gguf --ctx-size 8192 --n-gpu-layers 33
```

---

### Comparison: GPTQ vs AWQ vs GGUF

| Dimension | GPTQ | AWQ | GGUF (Q4_K_M) |
|---|---|---|---|
| Algorithm | Hessian-guided column-wise | Activation-aware channel scaling | Block-wise k-quantization |
| Calibration | Required (128 samples, ~10 min) | Required (128 samples, ~5 min) | None |
| Hardware target | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) | CPU, Apple Silicon, any GPU |
| Inference framework | AutoGPTQ, vLLM, TGI | vLLM (native), TGI | llama.cpp, Ollama |
| Quality at INT4 | Good (PPL +2-4%) | Better (PPL +1-3%) | Good (PPL +2-4%) |
| Production use | Older standard; AWQ preferred | Current GPU production standard | Local inference, CPU serving |

**Rule of thumb:** for GPU serving in production, use AWQ INT4. For local CPU/Mac inference, use GGUF Q4_K_M. For fine-tuning on a quantized base, use NF4 via QLoRA (Section 4).

---

## 6. Full Fine-tuning with ZeRO

When LoRA is insufficient (e.g., catastrophic forgetting on diverse tasks, or need to update all layers), use full fine-tuning with ZeRO Stage 3.

```python
# DeepSpeed config for full fine-tuning of 70B model
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "offload_optimizer": {
            "device": "cpu",       # offload optimizer states to CPU
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",       # offload params to CPU (ZeRO-Infinity)
            "pin_memory": True
        }
    },
    "bf16": {"enabled": True},
    "gradient_checkpointing": True,
    "train_batch_size": 128,
    "gradient_accumulation_steps": 16,
}
```

**GPU budget for full fine-tuning:**

| Model | ZeRO Stage | GPUs (80GB A100) | Notes |
|---|---|---|---|
| 7B | ZeRO-2 | 1 | Fits with GC |
| 13B | ZeRO-2 | 2 | |
| 70B | ZeRO-3 | 8 | With CPU offload |
| 70B | ZeRO-3 | 16 | Without CPU offload |
| 405B | ZeRO-3 + TP=8 | 128 | |

---

## 7. Gradient Checkpointing for Fine-tuning

At long sequences (4K+), activation memory dominates. Gradient checkpointing trades compute for memory.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,      # ~33% more FLOPs, ~10x less activation memory
    gradient_checkpointing_kwargs={"use_reentrant": False},  # better for PEFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,    # effective batch = 4 × 8 × n_gpus
    max_seq_length=4096,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,
)
```

---

## 8. Fine-tuning Hyperparameters

| Hyperparameter | LoRA | QLoRA | Full FT | Notes |
|---|---|---|---|---|
| Learning rate | 1e-4 to 3e-4 | 2e-4 | 1e-5 to 5e-5 | Lower for full FT |
| Batch size | 16–128 | 16–32 | 128–512 | Larger = more stable |
| Epochs | 1–3 | 1–3 | 1–2 | LLMs overfit quickly |
| Warmup | 3–5% of steps | 3% | 5–10% | |
| LR schedule | cosine | cosine | cosine or linear | |
| Rank (LoRA) | 16 | 16 | N/A | |
| Alpha (LoRA) | 32 (=2×r) | 32 | N/A | |

**Learning rate rule of thumb:** LoRA adapters need higher LR than full fine-tuning because they start from zero and must converge in fewer steps. Full fine-tuning needs lower LR to avoid catastrophic forgetting.

---

## 9. Prefix Tuning

**Core idea**: instead of modifying model weights, prepend a set of *learned soft tokens* to the key and value sequences at every transformer layer. The model's parameters stay frozen; only the prefix embeddings are trained.

**Why "soft" tokens**: the prefix is not actual text — it's a trainable tensor $P_i \in \mathbb{R}^{l \times d_{model}}$ for layer $i$, where $l$ is the prefix length (typically 10–100). These virtual tokens condition the model's attention at every layer, functioning as trainable "task context" that reshapes what each layer attends to.

**How it differs from prompt tuning**: prompt tuning only prepends soft tokens to the input embedding layer. Prefix tuning prepends them to the K and V matrices at *every* layer — more expressive but uses more memory per layer.

**Mathematical form**: for a transformer layer with original KV sequences $K, V \in \mathbb{R}^{T \times d}$, prefix tuning inserts:

$$K' = [P^K_i; K], \quad V' = [P^V_i; V]$$

The attention is now computed over the extended sequence: queries attend to both the prefix tokens and the original context tokens. The prefix tokens consume $l$ positions of context window per layer.

```python
from peft import PrefixTuningConfig, get_peft_model, TaskType

prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,      # prefix length l
    prefix_projection=True,     # use an MLP to project prefix embeddings (more stable)
    encoder_hidden_size=512,    # hidden size of the prefix MLP reparameterization
)

model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()
# trainable params: ~180,000 || all params: 6,738,415,616 || trainable%: ~0.003%
```

**Reparameterization trick**: directly optimizing raw prefix embeddings is unstable (they lie in a high-dimensional space with no natural structure). In practice, the prefix is generated by a small MLP: $P_i = \text{MLP}_\phi(P_{\text{raw}})$, and only $\phi$ is trained. After training, the MLP is discarded and only the resulting $P_i$ is stored.

**Memory profile**:
- Trainable parameters: 0.01–0.1% of total (vs LoRA's ~1%)
- Context window cost: each prefix of length $l$ consumes $l$ KV positions permanently — a 100-token prefix reduces usable context by 100 tokens at every layer
- No inference overhead if prefix is precomputed and cached in KV format

**When prefix tuning outperforms LoRA**:
- Very few training examples (< 1K): prefix tuning is more regularized
- Inference-time task switching: precompute prefix KVs for each task and swap them without touching model weights
- Multi-tenant serving: store one base model, load per-task prefix KVs on demand

**When LoRA is preferred**:
- Larger datasets (> 10K examples): LoRA can learn more expressive task-specific transformations
- Long-context workloads: prefix tokens consume fixed context budget regardless of actual input length
- Interpretability: LoRA weight deltas can be analyzed via SVD; prefix soft tokens are not interpretable

---

## 10. Adapter Layers

**Core idea**: insert small bottleneck feed-forward modules between the sublayers of each transformer block. The base model is frozen; only the adapter parameters are trained.

**Architecture**: each adapter is a two-layer MLP with a bottleneck:

$$\text{Adapter}(h) = h + W_{\text{up}} \cdot \sigma(W_{\text{down}} \cdot h)$$

where $W_{\text{down}} \in \mathbb{R}^{d \times r}$ projects down to bottleneck dimension $r$, and $W_{\text{up}} \in \mathbb{R}^{r \times d}$ projects back up. The residual connection ensures the adapter starts as an identity (if $W_{\text{up}}$ and $W_{\text{down}}$ are initialized near zero).

Adapters are typically inserted in two positions per block:
1. After the self-attention sublayer (before the residual add)
2. After the FFN sublayer (before the residual add)

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Near-zero initialization → starts as identity
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(x))))
```

**Parameter count**: for each transformer block with $d_{model}=4096$ and bottleneck $r=64$:
- $W_{\text{down}}$: 4096 × 64 = 262K params
- $W_{\text{up}}$: 64 × 4096 = 262K params  
- Two adapters per block × 32 blocks = ~33M additional params for a 7B model (0.5%)

**Adapter vs LoRA — what actually differs**:

| Property | Adapter | LoRA |
|---|---|---|
| Parameter efficiency | ~0.5–2% | ~0.5–1% |
| Inference overhead | Yes — sequential MLP per layer | No — merge into W at inference |
| Parallelism | Sequential; adds latency | Parallel with base computation |
| Placement | After sublayers (residual path) | In weight matrices (parallel path) |
| Expressiveness per param | Higher (nonlinear via activation) | Lower (linear, rank-limited) |

**Why LoRA has largely displaced adapters**: LoRA adapters add zero inference latency (merge $BA$ into $W_0$ before serving), while bottleneck adapters add a sequential MLP computation to every forward pass. For production serving, this matters at scale. Adapters retain an advantage when you need nonlinear task-specific transformations or when merging weights into the base model is undesirable (multi-tenant scenarios with many tasks).

**PEFT library (Houlsby-style adapters)**:

```python
from peft import AdaptionPromptConfig  # Note: bottleneck adapters use custom configs
# Most practical: use Hugging Face adapters integration or adapters library

# Alternative: direct integration with adapters library
# pip install adapters
import adapters
from adapters import AdapterConfig

adapter_config = AdapterConfig(
    mh_adapter=True,       # add adapter after multi-head attention
    output_adapter=True,   # add adapter after FFN
    reduction_factor=16,   # d_model / reduction_factor = bottleneck size
    non_linearity="relu",
)
```

---

## 11. LoRA Variants

| Variant | Key idea | Benefit |
|---|---|---|
| LoRA | Low-rank decomposition | Memory efficient |
| LoRA+ | Different LRs for A and B matrices | Faster convergence |
| DoRA | Decompose into magnitude + direction | Closer to full FT quality |
| AdaLoRA | Adaptive rank allocation via SVD | Auto-tunes rank per layer |
| VeRA | Share A and B across layers, learn scale vectors | Extreme param efficiency |
| LoRAMoE | Multiple LoRA adapters as MoE | Multi-task without interference |

---

## PEFT Method Summary

| Method | Trainable % | Inference overhead | Expressiveness | Best for |
|---|---|---|---|---|
| Full fine-tuning | 100% | None | Highest | Large data, compute available |
| LoRA (r=16) | ~1% | None (merge at inference) | High | General instruction tuning |
| QLoRA (r=16, NF4) | ~1% | None | High | Consumer GPU, 70B+ models |
| Prefix tuning | 0.01–0.1% | None (cached KV) | Medium | Few-shot, task switching |
| Adapter layers | 0.5–2% | Sequential MLP | Medium-high | Multi-task, non-linear tasks |
| Prompt tuning | 0.001% | None | Low | Very few-shot |

---

## Canonical Interview Q&As

**Q: Why does LoRA work well despite only updating 1% of parameters?**  
A: The key insight is that fine-tuning updates lie in a low intrinsic dimensionality subspace. Empirically, when you SVD the delta-W from full fine-tuning, most of the signal is captured by the top few singular vectors. Pre-trained LLMs already encode rich representations — fine-tuning mostly adjusts how those representations map to task-specific outputs, which requires only a small directional change in weight space. The rank-r decomposition $\Delta W = BA$ constrains updates to lie in an r-dimensional subspace, and r=16 is usually sufficient to capture this intrinsic dimensionality. Evidence: LoRA fine-tuned models on instruction following reach 95%+ of full fine-tuning performance at <1% of the trainable parameters.

**Q: What's the memory breakdown for QLoRA fine-tuning a 70B model?**  
A: Base model frozen in NF4 (4-bit): ~35 GB. LoRA adapters in bf16 (r=16): ~3 GB. Adam optimizer states only for LoRA parameters: ~6 GB. Activations for a seq=512, batch=4: ~5 GB. Total ~49 GB — fits on a single A100 80GB. Compare to full bf16 fine-tuning: weights 140 GB + gradients 140 GB + optimizer 840 GB = 1.12 TB. QLoRA achieves ~95% of full fine-tuning quality at 2.3% of the memory cost by combining NF4 quantization for the base model with bf16 precision LoRA adapters that never need to backprop through the quantized weights.

**Q: How do you decide between LoRA and full fine-tuning?**  
A: Four factors: (1) Data scale — less than 100K examples generally doesn't justify full fine-tuning; LoRA is less prone to overfitting; (2) Task shift — if the task is far from pretraining distribution (e.g., medical domain with specialized terminology and reasoning), full fine-tuning learns better representations; if it's just format/instruction following, LoRA is sufficient; (3) Compute budget — LoRA enables fine-tuning 70B models on a single GPU; full fine-tuning needs 16+ A100s; (4) Deployment — LoRA adapters can be hot-swapped for different tasks on the same base model (100MB adapter vs 140GB weights), crucial for multi-tenant serving. In practice: start with LoRA, evaluate quality gap vs full fine-tuning on held-out test set, only invest in full fine-tuning if the gap exceeds 3–5%.

## Flashcards

**What is the per-parameter memory footprint for full fine-tuning with fp32 Adam?** #flashcard
~16 bytes/param before activations: 2 bytes weights (bf16) + 2 bytes gradients (bf16) + 12 bytes optimizer states (fp32 master weights + momentum + variance).

**How much memory does LoRA fine-tuning need for an 8B model, and why?** #flashcard
~20 GB total — fits on a single 24GB GPU. The base weights stay frozen in bf16 while only the small LoRA adapter's gradients and optimizer states are trainable (~0.1% of params).
