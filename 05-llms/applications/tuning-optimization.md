# Tuning and Optimization

The right question before any LLM adaptation: **what problem are you actually solving?**

- Knowledge gap → RAG
- Behavior / style / format → Fine-tuning
- Latency / cost → Inference optimization
- Alignment / safety → RLHF / DPO

Choosing the wrong lever wastes compute and often makes things worse.

---

## 1. The Decision Tree

```
Is the model's behavior correct but knowledge is stale/private?
    → RAG (no training needed)

Is the model's knowledge fine but behavior, tone, or format is wrong?
    → Fine-tuning (SFT or PEFT)

Is the model's alignment (helpfulness, safety, refusals) wrong?
    → RLHF or DPO

Is the model too slow or expensive to serve?
    → Quantization, distillation, pruning

Is the prompt just poorly designed?
    → Fix the prompt first — it's free
```

---

## 2. Prompting vs RAG vs Fine-Tuning

| Approach | Best for | Cost | Latency | Freshness |
| :--- | :--- | :--- | :--- | :--- |
| **Prompting** | Format, reasoning style, persona | Near zero | None | N/A |
| **RAG** | Current/private facts, citations | Medium (infra) | +50–300ms | Real-time |
| **SFT** | Consistent task behavior, domain tone | High (GPU) | None at serve | Snapshot in time |
| **PEFT (LoRA)** | Same as SFT but cheaper | Medium | None at serve | Snapshot |
| **RLHF/DPO** | Alignment, preference following | Very high | None at serve | Snapshot |

---

## 3. Supervised Fine-Tuning (SFT)

SFT trains the model on (prompt, response) pairs using standard cross-entropy language modeling loss.

**Data format:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a customer support agent for Acme Corp."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, go to Settings > Account > Reset Password. You'll receive an email within 2 minutes."}
  ]
}
```

**Data quality >> data quantity.** 1,000 high-quality examples often outperform 100,000 noisy ones (see Phi models, LIMA paper).

**Training recipe:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

training_args = TrainingArguments(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=2048,
)
trainer.train()
```

---

## 4. LoRA (Low-Rank Adaptation)

### The Idea

Instead of updating all $d \times d$ weight matrices ($d$ can be 4096+), learn a low-rank decomposition:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

Typically $r \in \{4, 8, 16, 64\}$. With $r=8$ and $d=4096$: update $2 \times 4096 \times 8 = 65536$ parameters instead of $4096^2 = 16.7M$. That's a **256× reduction** in trainable parameters for that layer.

At initialization: $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ — so $\Delta W = 0$ initially, preserving the pretrained model.

The scaling factor $\alpha/r$ controls the update magnitude.

### Which Layers to Apply LoRA

Typically: query, key, value, output projections in attention. Sometimes FFN layers too.

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,           # scaling = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 6,553,600 || all params: 6,744,547,328 || trainable%: 0.097
```

### QLoRA

Combines 4-bit quantization of the base model with LoRA training:

1. Load base model in 4-bit NF4 quantization (bitsandbytes)
2. Apply LoRA adapters (in full precision BF16) on top
3. Compute forward pass in 4-bit, gradients in BF16

This enables fine-tuning a 70B model on a single 48GB A100.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
# Then apply LoRA config as above
```

---

## 5. RLHF (Reinforcement Learning from Human Feedback)

### Three-Stage Pipeline

**Stage 1 — Supervised Fine-Tuning (SFT):**
Fine-tune on high-quality demonstration data to get a reasonably capable base.

**Stage 2 — Reward Model (RM):**
Collect preference data: human annotators rank model responses.
Train a reward model $r_\phi(x, y)$ (LLM with a regression head) to predict human preference:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

where $y_w$ is the preferred response and $y_l$ is the less preferred one.

**Stage 3 — PPO (Proximal Policy Optimization):**
Fine-tune the SFT model to maximize reward, with a KL penalty to prevent diverging from the SFT model:

$$\mathcal{L}_{PPO} = r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta(y|x) \| \pi_{SFT}(y|x))$$

The KL penalty prevents reward hacking (exploiting the reward model without actually being better).

**Operational complexity:** requires running 3 models simultaneously during PPO (policy, reference, reward model). Memory and coordination overhead is significant.

---

## 6. DPO (Direct Preference Optimization)

DPO achieves the same alignment goal as RLHF without a separate reward model or RL loop.

### Key Insight

The optimal RLHF policy has a closed-form relationship to the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Substituting this into the preference learning objective and cancelling the partition function $Z(x)$ gives the DPO loss:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

In plain English: increase the probability of preferred responses relative to the reference model, decrease the probability of rejected ones.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                    # KL penalty strength
    loss_type="sigmoid",         # original DPO
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,          # much lower than SFT
    num_train_epochs=1,
    bf16=True,
)

# Data format: {"prompt": "...", "chosen": "...", "rejected": "..."}
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,         # frozen copy of SFT model
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

**DPO vs RLHF:**

| Aspect | RLHF (PPO) | DPO |
| :--- | :--- | :--- |
| **Reward model** | Explicit, trained separately | Implicit |
| **Online sampling** | Yes (generates during training) | No (offline) |
| **Memory** | 3 models loaded | 2 models loaded |
| **Stability** | Tricky (RL variance) | More stable (supervised) |
| **Quality ceiling** | Potentially higher (online) | Competitive in practice |

---

## 7. Quantization

Quantization reduces numerical precision to shrink model size and increase inference speed.

### Data Types

| Format | Bits | Range | Notes |
| :--- | :--- | :--- | :--- |
| **FP32** | 32 | Full | Training default |
| **BF16** | 16 | Same exponent range as FP32 | Training and inference default on A100/H100 |
| **FP16** | 16 | Smaller range | Can overflow for large activations |
| **INT8** | 8 | -128 to 127 | ~2× inference speedup, minor quality loss |
| **INT4 / NF4** | 4 | Limited | ~4× memory reduction; used in QLoRA |
| **GPTQ** | 3–4 | N/A | Post-training quantization with calibration data |

### Post-Training Quantization (PTQ)

No additional training required. Apply after standard training:

```python
from transformers import AutoModelForCausalLM
import torch

# Load in 8-bit (bitsandbytes LLM.int8())
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
```

### GPTQ (Post-Training Quantization with Calibration)

More accurate than naive INT4 because it optimizes quantization error layer by layer using calibration data:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,       # quantize in groups of 128
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
examples = [tokenizer(text, return_tensors="pt") for text in calibration_texts]
model.quantize(examples)
model.save_quantized("./quantized-model")
```

### AWQ (Activation-aware Weight Quantization)

Identifies salient weights (those with large activations) and protects them from aggressive quantization. Often better quality than GPTQ at INT4.

---

## 8. Knowledge Distillation

Transfer knowledge from a large **teacher** model to a smaller **student** model.

**Objective:**

$$\mathcal{L}_{KD} = (1-\alpha) \mathcal{L}_{CE}(y, \hat{y}_s) + \alpha \mathcal{L}_{KL}(p_s^T \| p_t^T)$$

where $p^T = \text{softmax}(\text{logits}/T)$ with temperature $T > 1$ (softer distributions reveal more information about similar classes).

**Sequence-level distillation:** train the student on sequences generated by the teacher (data augmentation style). Used to create DistilGPT, DistilBERT.

**Online distillation (speculative decoding):** small draft model generates candidate tokens; large model verifies in parallel. Achieves 2–4× speedup with identical output distribution.

```python
# Speculative decoding
from transformers import AutoModelForCausalLM, AutoTokenizer

draft_model = AutoModelForCausalLM.from_pretrained("gpt2")      # small, fast
target_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")  # large, accurate

# HuggingFace handles speculative decoding via `assistant_model`
output = target_model.generate(
    inputs,
    assistant_model=draft_model,
    max_new_tokens=200,
)
```

---

## 9. Evaluating Fine-Tuned Models

Never ship a tuned model without comparing against the base on a held-out evaluation set.

| What changed | Eval method | Metric |
| :--- | :--- | :--- |
| **Format/style** | LLM-as-judge (GPT-4) on format compliance | Rate 1–5 or binary |
| **Task accuracy** | Task-specific benchmark | F1, EM, ROUGE |
| **Alignment** | Safety eval, red-teaming, refusal rate | Human eval |
| **General capability** | MMLU, HellaSwag, TruthfulQA | Accuracy |
| **Regression** | Eval on old training distribution | Perplexity, task accuracy |

**LLM-as-judge pattern:**

```python
def llm_judge(question, reference_answer, model_answer):
    prompt = f"""
    Question: {question}
    Reference answer: {reference_answer}
    Model answer: {model_answer}

    Rate the model answer from 1–5 on:
    - Correctness (does it answer the question accurately?)
    - Helpfulness (is it useful to the user?)
    - Conciseness (is it appropriately brief?)

    Output JSON: {{"correctness": X, "helpfulness": X, "conciseness": X, "reasoning": "..."}}
    """
    return json.loads(llm.complete(prompt))
```

---

## 10. Production Versioning

A model artifact is like a software release — it needs version control and a rollback path.

**Checklist before deploying a tuned model:**
- [ ] Base model version pinned
- [ ] Training data snapshot versioned
- [ ] LoRA config / hyperparameters logged (MLflow, W&B)
- [ ] Evaluation metrics vs baseline documented
- [ ] Safety checks passed (red-team, content filter audit)
- [ ] Inference latency benchmarked
- [ ] Rollback to previous artifact tested

> [!TIP]
> **Interview structure:** "When would you fine-tune vs RAG?" → Diagnose the problem first (knowledge vs behavior). Fine-tune when you need **consistent behavioral change** that persists across requests. RAG when facts need to be fresh or cited. If format is the only issue, fix the prompt first — it's free.
