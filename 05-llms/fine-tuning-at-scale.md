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

## 5. Full Fine-tuning with ZeRO

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

## 6. Gradient Checkpointing for Fine-tuning

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

## 7. Fine-tuning Hyperparameters

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

## 8. LoRA Variants

| Variant | Key idea | Benefit |
|---|---|---|
| LoRA | Low-rank decomposition | Memory efficient |
| LoRA+ | Different LRs for A and B matrices | Faster convergence |
| DoRA | Decompose into magnitude + direction | Closer to full FT quality |
| AdaLoRA | Adaptive rank allocation via SVD | Auto-tunes rank per layer |
| VeRA | Share A and B across layers, learn scale vectors | Extreme param efficiency |
| LoRAMoE | Multiple LoRA adapters as MoE | Multi-task without interference |

---

## Canonical Interview Q&As

**Q: Why does LoRA work well despite only updating 1% of parameters?**  
A: The key insight is that fine-tuning updates lie in a low intrinsic dimensionality subspace. Empirically, when you SVD the delta-W from full fine-tuning, most of the signal is captured by the top few singular vectors. Pre-trained LLMs already encode rich representations — fine-tuning mostly adjusts how those representations map to task-specific outputs, which requires only a small directional change in weight space. The rank-r decomposition $\Delta W = BA$ constrains updates to lie in an r-dimensional subspace, and r=16 is usually sufficient to capture this intrinsic dimensionality. Evidence: LoRA fine-tuned models on instruction following reach 95%+ of full fine-tuning performance at <1% of the trainable parameters.

**Q: What's the memory breakdown for QLoRA fine-tuning a 70B model?**  
A: Base model frozen in NF4 (4-bit): ~35 GB. LoRA adapters in bf16 (r=16): ~3 GB. Adam optimizer states only for LoRA parameters: ~6 GB. Activations for a seq=512, batch=4: ~5 GB. Total ~49 GB — fits on a single A100 80GB. Compare to full bf16 fine-tuning: weights 140 GB + gradients 140 GB + optimizer 840 GB = 1.12 TB. QLoRA achieves ~95% of full fine-tuning quality at 2.3% of the memory cost by combining NF4 quantization for the base model with bf16 precision LoRA adapters that never need to backprop through the quantized weights.

**Q: How do you decide between LoRA and full fine-tuning?**  
A: Four factors: (1) Data scale — less than 100K examples generally doesn't justify full fine-tuning; LoRA is less prone to overfitting; (2) Task shift — if the task is far from pretraining distribution (e.g., medical domain with specialized terminology and reasoning), full fine-tuning learns better representations; if it's just format/instruction following, LoRA is sufficient; (3) Compute budget — LoRA enables fine-tuning 70B models on a single GPU; full fine-tuning needs 16+ A100s; (4) Deployment — LoRA adapters can be hot-swapped for different tasks on the same base model (100MB adapter vs 140GB weights), crucial for multi-tenant serving. In practice: start with LoRA, evaluate quality gap vs full fine-tuning on held-out test set, only invest in full fine-tuning if the gap exceeds 3–5%.
