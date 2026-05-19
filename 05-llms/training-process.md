# LLM Training: From Raw Text to Deployed Model

---

## 1. Pretraining: What Problem Does Next-Token Prediction Actually Solve?

**The problem**: you want a model with broad linguistic and world knowledge — something that understands syntax, semantics, factual relationships, reasoning chains, and code. But "broad knowledge" is not a training objective. You need a single, scalable loss function that forces the model to acquire all of this implicitly.

**The core insight**: next-token prediction over enough text forces the model to learn everything needed to make good predictions. To predict "the mitochondria is the ___", the model needs biology. To predict the next line of Python code, it needs syntax and semantics. The objective is simple; the breadth of knowledge required to minimize it is enormous.

**The mechanics**:

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

Nothing in this loss explicitly rewards factual accuracy, logical consistency, or helpfulness — those emerge because they help predict the next token in text written by humans who care about those things.

**What breaks**: next-token prediction is agnostic to whether the model is *right* or *helpful*. A model that confidently predicts plausible-sounding false statements gets rewarded if that's what appears in the training data. Pretraining instills capability and knowledge; it does not instill values. That requires subsequent alignment training.

---

## 2. Data Scale: Why You Need Trillions of Tokens

**The problem**: language has enormous combinatorial depth. A 7B-parameter model trained on too little data memorizes the training distribution rather than learning generalizable patterns. More parameters demand more data to avoid underfitting.

| Model | Tokens | Notes |
| :--- | :--- | :--- |
| GPT-3 | 300B | Underfit relative to model size |
| LLaMA 2 | 2T | Overtrained for inference efficiency |
| LLaMA 3 | 15T | Deduplicated from ~30T raw tokens |

**Why the tokens-per-parameter ratio matters**: the Chinchilla result (Section 4 of scaling-and-data.md) showed that GPT-3 was severely undertrained. LLaMA 3's 15T/70B ≈ 214 tokens per parameter deliberately overtrain relative to compute-optimal because the resulting model is smaller and cheaper to serve. Inference cost scales with model size at every request; training cost is one-time.

---

## 3. The Optimizer: Why Adam, and Why AdamW Specifically

**The problem**: vanilla gradient descent applies the same learning rate to every parameter. But in a transformer, different parameters have vastly different gradient scales — the embedding matrix for rare tokens gets sparse gradient updates, while the attention projection matrices get dense ones. A single learning rate is too large for some parameters and too small for others.

**The core insight**: maintain a running estimate of the gradient mean (first moment) and variance (second moment) per parameter. Scale each parameter's update by the inverse square root of its gradient variance — parameters with small, consistent gradients get larger effective updates; parameters with noisy, large gradients get smaller ones.

**The mechanics — Adam**:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment, momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment, adaptive scale)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \quad \text{(bias correction for early steps)}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**The AdamW fix for weight decay**: standard Adam applies L2 regularization by adding $\lambda \theta$ to the gradient $g_t$, which then gets scaled by the adaptive term $\hat{v}_t$. This means weight decay is weaker for parameters with large gradient variance — not the desired behavior. AdamW decouples weight decay: it applies the decay *after* the adaptive update, giving every parameter the same regularization strength regardless of gradient scale:

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$.

**Why $\beta_2 = 0.95$ not $0.999$?** At $\beta_2 = 0.999$, the second moment estimate updates very slowly — it takes ~1000 steps to forget past gradient information. Early in training, when the loss is volatile, this produces stale variance estimates that cause the effective learning rate to be poorly calibrated. $\beta_2 = 0.95$ makes the estimate forget faster, responding better to the large gradient changes that happen early in training.

**What breaks**: Adam's adaptive scaling has an unexpected interaction with learning rate schedules and warmup — see the stability file. Also, Adam maintains two momentum tensors per parameter, tripling optimizer state memory versus SGD. At 70B parameters in FP32: ~800 GB of optimizer state alone.

---

## 4. Learning Rate Schedule: Why Not Just a Constant Rate

**The problem**: a constant learning rate is a poor fit for the loss landscape's changing geometry over training. Early in training, the model is far from any good solution and gradients are noisy — large steps help, but also risk overshooting. Late in training, the model is near a local minimum and large steps cause it to bounce around rather than converge.

**The core insight**: use a large learning rate in the middle of training and small rates at both ends — warming up from essentially zero at the start, and decaying back down to a small value at the end.

**The mechanics — cosine decay with linear warmup**:

```python
import math

def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor

# Typical for a 7B model:
# max_lr = 3e-4, min_lr = 3e-5 (10%), warmup_steps = 2000, total_steps ~500k
```

**Why warmup is not optional**: Adam's second moment estimate $v_t$ starts at zero. The bias correction $1/(1-\beta_2^t)$ partially compensates, but for the first ~100 steps the effective learning rate is wildly amplified. Starting with a small LR prevents the random early gradients from causing destructive parameter updates before the optimizer statistics stabilize.

**What breaks**: without warmup, the first few hundred steps can push parameters so far from initialization that the model never recovers a good training trajectory — particularly for large models where the loss landscape is sharper.

---

## 5. Batch Size and Gradient Accumulation

**The problem**: training on one sequence at a time gives noisy gradient estimates — any single document might be an outlier. Larger batches average out the noise and give more reliable gradient direction. But GPU memory limits batch size.

**The core insight**: accumulate gradients over multiple forward passes before updating weights. The model sees a mini-batch effectively as large as (single-GPU batch) × (accumulation steps) × (number of GPUs).

**Batch size scaling with learning rate**: when you scale the batch size, you should scale the learning rate proportionally (linear scaling rule) or by square root (safer for very large batches):

$$\text{lr}_{\text{new}} = \text{lr}_{\text{base}} \times \sqrt{B_{\text{new}} / B_{\text{base}}}$$

GPT-3 used batch sizes up to 3.2M tokens (reached by growing the batch through training). LLaMA 3 70B: 4M tokens per batch.

**What breaks**: the gradient noise reduction from large batches plateaus — doubling the batch size halves gradient noise, but the benefit shrinks. Very large batches can also harm generalization by finding sharper minima.

---

## 6. Supervised Fine-Tuning (SFT): Teaching Format, Not Capability

**The problem**: a base pretrained model is a next-token predictor. Given "What is the capital of France?", it might complete "...asked by geography students worldwide" because that's a plausible continuation of text it saw. It has no instilled notion of "this is a question I should answer."

**The core insight**: a small number of high-quality (instruction, response) pairs teaches the model the format of being an assistant. The capability to answer the question already exists from pretraining; SFT teaches when and how to deploy it.

**The mechanics — train only on the response, not the prompt**:

The key implementation detail is loss masking: compute the cross-entropy loss only on assistant response tokens, not on the system prompt or user message. Training on the prompt would teach the model to predict instructions, which is not the goal.

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# Mask loss on everything before the assistant response token
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
        learning_rate=2e-5,       # ~15× lower than pretraining
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
    )
)
```

**Data format**:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain gradient descent in one paragraph."},
    {"role": "assistant", "content": "Gradient descent is an optimization algorithm..."}
  ]
}
```

**Data quality dominates quantity**: the LIMA paper showed that 1,000 carefully selected examples produced a model comparable to RLHF-aligned models trained on much more data. Diversity of task types and high response quality matter far more than scale. Mediocre synthetic responses actively hurt performance.

**What breaks**: SFT on a small, narrow dataset causes catastrophic forgetting of general capabilities. The learning rate must be much lower than pretraining to avoid overwriting base model weights. SFT also doesn't teach the model *not* to say harmful things — it only teaches format. Alignment requires the next stage.

---

## 7. RLHF: Why Cross-Entropy Loss Can't Teach Values

**The problem**: SFT trains on the "correct" response to each prompt. But "correct" is defined by whoever wrote the training data. For subjective properties — helpfulness, honesty, harmlessness — there is no single correct response. And the cross-entropy objective doesn't distinguish between two plausible responses where one is safe and one is harmful. You need a way to incorporate human preference signals directly.

**The core insight**: learn a reward function from human comparisons, then use reinforcement learning to optimize the policy (the LLM) to maximize that reward while staying close to the SFT model.

### Stage 1: Collect Preference Data

Human annotators see the same prompt with two model responses and indicate which is better:

```
Prompt: "How do I lose weight quickly?"
Response A: "Here are some healthy strategies..."
Response B: "Skip all meals and run five miles daily..."

Annotator preference: A > B
```

### Stage 2: Train a Reward Model

Fit a regression model on these comparisons. The reward model takes (prompt, response) and outputs a scalar score. Training loss is the Bradley-Terry pairwise preference model:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

This pushes $r_\phi(x, y_w) > r_\phi(x, y_l)$ for every preferred-rejected pair without needing absolute reward values.

### Stage 3: PPO Fine-Tuning

Update the policy to maximize reward while staying close to the SFT model. The KL penalty prevents reward hacking:

$$\mathcal{L}_{\text{PPO}} = r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(y \mid x) \| \pi_{\text{SFT}}(y \mid x))$$

Without the KL term, the model quickly discovers ways to achieve high reward model scores through degenerate behaviors — repetition of favored phrases, excessive length, sycophantic agreement — that the reward model wasn't specifically trained to penalize.

**Operational complexity**: PPO requires four models simultaneously in memory:
1. Policy $\pi_\theta$ (being updated)
2. Reference $\pi_{\text{SFT}}$ (frozen, for KL computation)
3. Reward model $r_\phi$ (frozen)
4. Value model (critic, for advantage estimation)

At 70B scale this is essentially infeasible on a single machine and requires very careful distributed coordination.

---

## 8. DPO: Cutting the Reward Model Out Entirely

**The problem**: RLHF's complexity is not just inconvenient — training four models simultaneously introduces many new failure modes (reward model overfitting, KL explosion, PPO instability). For many practitioners, the reward model is the weakest link: it's trained on far less data than the policy and may not generalize.

**The core insight**: the optimal RLHF policy has a closed-form expression in terms of the reference model and the reward function. You can invert this relationship to express the reward function in terms of the policy, substitute into the preference learning objective, and train the policy *directly* on preferences without ever building a reward model.

**The derivation**: the optimal policy for the RLHF objective is:

$$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp(r(x, y) / \beta)$$

Solving for the implicit reward:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting into the pairwise preference loss, $Z(x)$ cancels between winner and loser, leaving:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

In plain language: increase the probability of preferred responses *relative to the reference model*, decrease the probability of rejected responses *relative to the reference model*.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                       # KL penalty strength; higher = stay closer to reference
    loss_type="sigmoid",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,             # very low — preference data is small
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
```

**What breaks**:

- **Length correlation**: DPO can learn to make chosen responses longer instead of better — reward model annotators often preferred longer responses, so the dataset implicitly encodes a length bias. Length-normalized DPO variants address this.
- **Gradient degradation**: if the policy drifts far from the reference model, the DPO loss provides weak gradient signal (the log-probability ratio saturates). Monitor KL divergence during training.
- **Preference data quality**: contradictory preferences or ambiguous annotations degrade alignment. DPO is more sensitive to data quality than PPO because it has no explicit regularization mechanism beyond the $\beta$ term.

---

## 9. Continual Pretraining: Domain Adaptation Without Forgetting

**The problem**: a base model trained on general web text has broad but shallow knowledge in specialized domains (medicine, law, finance). Fine-tuning on domain-specific data can improve domain performance but causes catastrophic forgetting — the model overwrites general weights with domain-specific information and degrades on tasks it previously handled.

**The core insight**: mix domain-specific data with general data during continued pretraining. The mixing ratio controls the tradeoff between specialization and general capability retention.

```python
# Typical ratio: 80% domain-specific, 20% general
dataset = concatenate_datasets([
    domain_data.select(range(int(0.8 * total_steps))),
    general_data.select(range(int(0.2 * total_steps))),
]).shuffle()

# Much lower LR than initial pretraining — base weights are already well-trained
# lr ~ 1e-5 to 1e-4 (vs 3e-4 for initial pretraining of a 7B model)
```

**What breaks**: even with mixing, some forgetting is inevitable. LoRA (low-rank adaptation) is an alternative: freeze base model weights entirely and train small rank-decomposition matrices alongside them. This eliminates forgetting by construction but limits how much the model can adapt.

---

## 10. Post-Training Hyperparameter Reference

| Hyperparameter | SFT | DPO | Notes |
| :--- | :--- | :--- | :--- |
| Learning rate | 1e-5 – 2e-5 | 1e-7 – 5e-7 | DPO 10–100× lower than SFT |
| Batch size | 128–512 tokens | 64–256 tokens | — |
| Epochs | 2–5 | 1–2 | More epochs → catastrophic forgetting |
| Warmup | 3% of steps | 3% of steps | — |
| Beta (DPO) | N/A | 0.01–0.5 | Higher = stay closer to reference |
| Max grad norm | 1.0 | 1.0 | — |
