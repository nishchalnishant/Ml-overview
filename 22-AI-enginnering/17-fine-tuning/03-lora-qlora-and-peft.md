---
module: AI Engineering
topic: Fine-Tuning
subtopic: LoRA, QLoRA, and PEFT
status: unread
source: original (addition — not in the course)
tags: [fine-tuning, lora, qlora, peft, sft, hyperparameters, trl, unsloth]
---
# LoRA, QLoRA, and PEFT

```mermaid
flowchart LR
    X[Input x] --> W[Frozen weight W<br/>d × k]
    X --> A[A: r × k<br/>trainable]
    A --> B[B: d × r<br/>trainable, init 0]
    W --> SUM((+))
    B -->|× α / r| SUM
    SUM --> H[Output h = Wx + (α/r)·BAx]
```

Full fine-tuning updates every weight. For a 7B model that means ~112 GB of GPU memory for weights, gradients, and Adam states. **Parameter-efficient fine-tuning (PEFT)** freezes the base model and trains a small number of extra parameters. LoRA is the default: it usually reaches close to full-fine-tuning quality on task adaptation at a fraction of the memory, and it produces adapters of a few dozen MB that can be swapped at serving time.

## 1. How LoRA Works
For a weight matrix W (d × k), LoRA learns a low-rank update ΔW = B·A, with B (d × r) and A (r × k), r ≪ min(d, k).
- Trainable params per matrix: r(d + k) instead of d·k. For d = k = 4096 and r = 16: 131k vs. 16.8M (≈ 0.8%).
- B is initialized to zero, so training starts from the base model's behavior.
- Scaling α/r controls update strength.
- After training, ΔW can be **merged** into W (no inference overhead) or kept separate (hot-swappable adapters).

## 2. Memory Comparison (7–8B model, rough)
| Method | Weights | Trainable | GPU memory | Hardware |
|---|---|---|---|---|
| Full FT (bf16 + Adam) | 16 GB | 100% | ~110–130 GB | Multi-GPU (A100/H100 × 2–4) |
| LoRA (bf16 base) | 16 GB | ~0.1–1% | ~20–30 GB | 1× A100 40GB / L40S / 4090 (tight) |
| **QLoRA** (4-bit NF4 base) | ~5 GB | ~0.1–1% | ~10–16 GB | 1× 24 GB consumer GPU |

**QLoRA** quantizes the frozen base to 4-bit NormalFloat, uses double quantization and paged optimizers, and trains LoRA adapters in bf16. Quality is close to LoRA at much lower memory; training is somewhat slower.

## 3. PEFT Family
| Method | Idea | Note |
|---|---|---|
| **LoRA** | Low-rank update on chosen matrices | Default choice |
| QLoRA | LoRA on 4-bit base | Single-GPU fine-tuning |
| DoRA | Splits weights into magnitude and direction; LoRA on direction | Small gains over LoRA at the same rank |
| rsLoRA | Scale by α/√r | More stable at high rank |
| LoRA+ | Different learning rates for A and B | Faster convergence |
| Prefix / prompt tuning | Learn virtual tokens | Fewer params, usually weaker |
| (IA)³ | Learn rescaling vectors | Very few params |
| Full FT | Everything | Best for large shifts; most expensive |

## 4. Hyperparameters That Matter
| Parameter | Typical start | Notes |
|---|---|---|
| `r` (rank) | 16 (8–64) | Higher for harder or bigger shifts; returns diminish |
| `lora_alpha` | 16–32 (often = r or 2r) | Effective scale α/r |
| `target_modules` | All linear: q,k,v,o,gate,up,down | Attention-only is weaker |
| `lora_dropout` | 0–0.05 | |
| Learning rate | 1e-4 – 2e-4 (LoRA); 1e-5 – 2e-5 (full) | LoRA tolerates ~10× higher LR |
| Epochs | 1–3 | More → overfitting; watch val loss |
| Effective batch size | 16–64 | Use gradient accumulation |
| Max sequence length | Fit the data's p99 | Use packing for short examples |
| LR schedule | Cosine with ~3–5% warmup | |

## 5. Training with TRL
```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

base = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16),  # QLoRA
)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

trainer = SFTTrainer(
    model=model, peft_config=peft_cfg,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    args=SFTConfig(output_dir="out", num_train_epochs=2, learning_rate=2e-4,
                   per_device_train_batch_size=4, gradient_accumulation_steps=8,
                   lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
                   completion_only_loss=True, packing=True, eval_strategy="steps", eval_steps=100),
)
trainer.train()
trainer.save_model("out/adapter")   # ~tens of MB
```
Faster alternatives: **Unsloth** (2× speed, lower memory), **Axolotl** (YAML configs), **torchtune**, **LLaMA-Factory**.

## 6. Monitoring a Run
- **Train loss falls, val loss rises** → overfitting: fewer epochs, more data, lower LR.
- **Loss spikes / NaN** → LR too high, bad examples, fp16 overflow (use bf16).
- **Loss barely moves** → LR too low, wrong target modules, labels masked incorrectly.
- Loss is a proxy. **Evaluate checkpoints on the task eval** and a general benchmark (catastrophic forgetting) and the safety suite.

## 7. Multi-GPU and Larger Models
- **DDP** when the model fits on one GPU; **FSDP** or **DeepSpeed ZeRO-3** to shard weights, grads, and optimizer state when it doesn't.
- 70B QLoRA fits on 1–2× 80 GB GPUs; full FT of 70B needs a multi-node cluster.
- Details: repo [Fine-tuning at scale](../../10-llms/06-fine-tuning-at-scale.md).

## Takeaways
- LoRA trains a low-rank update BA on frozen weights: ~0.1–1% of parameters, small swappable adapters, mergeable for zero overhead.
- QLoRA adds a 4-bit base so 7–8B models fine-tune on a single 24 GB GPU.
- Start with r=16, alpha 16–32, all linear layers, LR ~2e-4, 1–3 epochs, completion-only loss.
- Judge checkpoints by task, general, and safety evals, not by loss.

## References
- Hu et al., *LoRA* (2021) · Dettmers et al., *QLoRA* (2023) · Liu et al., *DoRA* (2024) · Biderman et al., *LoRA Learns Less and Forgets Less* (2024)
- Hugging Face PEFT and TRL docs · Unsloth docs · Sebastian Raschka, *Practical Tips for Finetuning LLMs Using LoRA*

## Related
- [Preparing fine-tuning data](02-preparing-fine-tuning-data.md) · [Preference tuning](04-preference-tuning-dpo-and-rl.md) · [Distillation and serving](05-distillation-and-serving-fine-tuned-models.md)
- Repo: [Fine-tuning at scale](../../10-llms/06-fine-tuning-at-scale.md) · [Model merging](../../11-llm-applications/08-model-merging.md)
