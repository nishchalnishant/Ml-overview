# LLM Training: From Raw Text to Deployed Model

Training a modern LLM involves three stages with fundamentally different objectives, data types, and engineering constraints. This file covers pretraining mechanics, stability techniques, instruction tuning, and alignment — with the numerical specifics that matter in practice.

---

## 1. Pretraining

### Objective

Autoregressive language modeling: predict the next token given all previous tokens.

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

This seemingly simple objective forces the model to learn syntax, semantics, world knowledge, reasoning patterns, and even code — because all of these are required to predict language accurately.

### Data Scale

| Model | Tokens | Approx. unique web pages |
| :--- | :--- | :--- |
| GPT-3 | 300B | 45TB compressed |
| LLaMA 2 | 2T | ~300B unique pages |
| LLaMA 3 | 15T | Deduplicated from 30T raw |

### Batch Size Scaling

Larger batches → more stable gradients but require careful learning rate scaling:

$$\text{lr}_{\text{large batch}} = \text{lr}_{\text{base}} \times \sqrt{B / B_{\text{base}}}$$

(Linear scaling rule: lr × batch size. Square root rule is empirically safer for very large batches.)

GPT-3 used batch size of 3.2M tokens (reached after warmup from 32k). LLaMA 3 70B: 4M tokens.

### Learning Rate Schedule

Standard: linear warmup → cosine decay

```python
import math

def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor

# Typical hyperparameters for 7B model:
# max_lr = 3e-4, min_lr = 3e-5, warmup_steps = 2000, total_steps = ~500k
```

### Optimizer: AdamW

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}$$

The $\lambda \theta_{t-1}$ term is weight decay (L2 regularization decoupled from adaptive scaling). Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$.

**Why $\beta_2 = 0.95$ not $0.999$?** Faster decay of second moment estimates allows larger effective learning rates early in training when loss is volatile.

---

## 2. Training Stability

### Gradient Clipping

Clips gradient norm to prevent parameter explosions:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Without clipping, a single bad batch can destroy weeks of training. With clipping, loss spikes are contained.

**Spike detection:** monitor the norm of gradients per step. A sudden 10× spike indicates instability; some training runs reduce LR at this point and resume.

### Mixed Precision (BF16)

BF16 has the same exponent range as FP32 (8 bits) but fewer mantissa bits (7 vs 23). This prevents overflow while reducing memory:

| Format | Memory | Overflow risk | Notes |
| :--- | :--- | :--- | :--- |
| FP32 | 4 bytes | Low | Training default (old) |
| FP16 | 2 bytes | High for large activations | Requires loss scaling |
| BF16 | 2 bytes | Same as FP32 | A100/H100 training default |

**Typical mixed-precision setup:**
- Weights: BF16
- Activations: BF16
- Gradient accumulation: FP32
- Master weights (optimizer states): FP32

### Activation Checkpointing

Trading compute for memory: recompute activations during backward pass instead of storing them.

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward_with_checkpoint(self, x):
        return checkpoint(self.forward, x, use_reentrant=False)
```

Reduces activation memory by ~$\sqrt{L}$ (where $L$ = number of layers) at the cost of ~33% more compute.

### ZeRO (Zero Redundancy Optimizer)

Distributed training across GPUs normally replicates optimizer state, gradients, and parameters on every device. ZeRO eliminates redundancy:

| Stage | What is sharded | Memory per GPU |
| :--- | :--- | :--- |
| ZeRO-1 | Optimizer states | ~1/N |
| ZeRO-2 | Optimizer states + gradients | ~1/N |
| ZeRO-3 | Optimizer states + gradients + parameters | ~1/N |

With ZeRO-3, a 70B model can be trained on 8× A100 80GB GPUs with batch size > 1.

```python
# DeepSpeed ZeRO config
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # CPU offload for larger models
        "offload_param": {"device": "cpu"},
    },
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
}
```

### Training Loss Shape

A healthy pretraining run shows:
1. Rapid initial loss drop (first ~1B tokens): learning basic language patterns
2. Smooth continued decay following a power law
3. No significant loss spikes (or rapid recovery if they occur)

Red flags: persistent loss spike that doesn't recover (restart from checkpoint), increasing validation loss while training loss decreases (data distribution shift), NaN loss (overflow or learning rate too high).

---

## 3. Supervised Fine-Tuning (SFT)

Converts a base model (next-token predictor) into an instruction-following assistant.

### Training Objective

Only compute loss on the response tokens, not the system/user prompt:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# Completion-only collator: mask loss on prompt tokens
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|start_header_id|>assistant<|end_header_id|>",
    tokenizer=tokenizer
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,          # much lower than pretraining
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
    )
)
```

### Data Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain gradient descent in one paragraph."},
    {"role": "assistant", "content": "Gradient descent is an optimization algorithm..."}
  ]
}
```

### SFT Data Quality Guidelines

- **1,000 high-quality > 100,000 noisy** (LIMA paper: 1,000 examples ≈ RLHF-aligned model)
- Include diverse tasks: reasoning, code, summarization, QA, creative writing
- Response quality matters more than coverage — reject mediocre synthetic responses
- Maintain train/val split on user intent categories, not random splits

---

## 4. RLHF (Reinforcement Learning from Human Feedback)

### Stage 1: Collect Preference Data

Human annotators rank model outputs for the same prompt:

```
Prompt: "Explain quantum entanglement to a 10-year-old."
Response A: [technical jargon, hard to follow]
Response B: [clear analogy, age-appropriate]

Human ranking: B > A
```

### Stage 2: Train Reward Model

A reward model $r_\phi(x, y)$ is an LLM with a regression head:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

where $y_w$ is the preferred response ("winner") and $y_l$ is the dispreferred response ("loser").

```python
from transformers import AutoModelForSequenceClassification

# Reward model = LLM backbone + linear head outputting scalar
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    num_labels=1,     # single scalar reward
)
```

### Stage 3: PPO Fine-Tuning

Update the policy (SFT model) to maximize reward while staying close to the SFT model:

$$\mathcal{L}_{\text{PPO}} = r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(y \mid x) \| \pi_{\text{SFT}}(y \mid x))$$

The KL penalty $\beta$ prevents **reward hacking**: exploiting the reward model with degenerate outputs (repetition, specific token patterns) that score high but are not actually good.

**Operational complexity:** PPO requires four models simultaneously:
1. Policy model $\pi_\theta$ (being updated)
2. Reference model $\pi_{\text{SFT}}$ (frozen, for KL)
3. Reward model $r_\phi$ (frozen)
4. Value model (critic, for advantage estimation)

---

## 5. DPO (Direct Preference Optimization)

DPO achieves the same alignment goal as RLHF with significantly less engineering complexity.

### Derivation

The optimal RLHF policy has a closed-form solution:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)}{Z(x)}$$

Solving for $r(x,y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting into the preference learning objective and noting $Z(x)$ cancels between winner and loser:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**Plain English:** increase the log-probability of preferred responses *relative to the reference model*, decrease the log-probability of rejected responses relative to reference.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                       # KL penalty strength
    loss_type="sigmoid",            # original DPO loss
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,             # very low — alignment data is small
    num_train_epochs=1,
    bf16=True,
    max_prompt_length=512,
    max_length=1024,
)

# Data format: {"prompt": "...", "chosen": "...", "rejected": "..."}
dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=sft_model,            # frozen reference
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

### DPO Failure Modes

- **Chosen/rejected length correlation:** DPO can learn to make chosen responses longer rather than better — use length-normalized DPO variants
- **Reference model drift:** if policy drifts far from reference, the DPO loss provides weak gradient signal — monitor KL divergence during training
- **Preference data quality:** contradictory or ambiguous preferences degrade alignment — curate carefully

---

## 6. Post-Training Hyperparameter Reference

| Hyperparameter | SFT | DPO | Notes |
| :--- | :--- | :--- | :--- |
| Learning rate | 1e-5 to 2e-5 | 1e-7 to 5e-7 | DPO 10-100× lower than SFT |
| Batch size | 128-512 tokens/step | 64-256 tokens/step | — |
| Epochs | 2-5 | 1-2 | More epochs → forgetting base model |
| Warmup | 3% of total steps | 3% of total steps | — |
| Beta (DPO) | N/A | 0.01-0.5 | Higher = stay closer to reference |
| Max grad norm | 1.0 | 1.0 | — |

---

## 7. Continual Pre-Training

To adapt a base model to a new domain without forgetting general capabilities:

```python
# Mix domain-specific data with general data to prevent catastrophic forgetting
# Typical ratio: 80% domain-specific, 20% general web text

dataset = concatenate_datasets([
    domain_data.select(range(int(0.8 * total_steps))),
    general_data.select(range(int(0.2 * total_steps))),
]).shuffle()

# Use a lower learning rate than initial pretraining
# lr = 1e-5 to 1e-4 (vs. 3e-4 for initial pretraining of 7B)
```

**Catastrophic forgetting:** the model overwrites general knowledge weights with domain-specific ones. Mitigation: lower LR, data mixing, Elastic Weight Consolidation (EWC), or LoRA (which preserves base weights entirely).

> [!TIP]
> **Interview structure:** LLM training = pretraining (world knowledge, next-token prediction, billions of tokens) → SFT (instruction format, thousands of high-quality examples) → alignment (RLHF for maximum quality but complex, DPO for simpler preference learning). The key insight: pretraining gives capability, SFT gives format, alignment gives values. Skipping any stage leaves the model worse in a predictable way.
