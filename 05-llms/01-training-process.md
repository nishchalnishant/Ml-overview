---
module: Llms
topic: Training Process
subtopic: ""
status: unread
tags: [llms, ml, training-process]
---
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

## 8.5 ORPO: Preference Alignment Without a Reference Model

*(niche — a secondary DPO variant; know it exists and the one-line pitch, the mechanics below are reference depth)*

**The problem with DPO**: DPO still requires a frozen reference model (the SFT copy) loaded in memory during training. This doubles GPU memory consumption and introduces a hyperparameter ($\beta$) that controls how tightly the policy stays near the reference. ORPO (Odds Ratio Preference Optimization, Hong et al., 2024) eliminates the reference model entirely.

**Core insight**: the SFT cross-entropy loss already teaches the model to generate good responses. Rather than computing a KL-divergence against a frozen reference, ORPO adds a penalty term based on the *odds ratio* between the model's own probability of the chosen vs. rejected response. No external reference is needed — the model's current parameters serve as both the training target and the implicit "reference."

**The loss function**:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}$$

Where the odds ratio loss is:

$$\mathcal{L}_{\text{OR}} = -\mathbb{E}\left[\log \sigma\!\left(\log \frac{p_\theta(y_w)}{1 - p_\theta(y_w)} - \log \frac{p_\theta(y_l)}{1 - p_\theta(y_l)}\right)\right]$$

The odds of a sequence $y$ given the current model: $\text{odds}_\theta(y) = \frac{p_\theta(y)}{1 - p_\theta(y)}$. ORPO maximizes the log-odds of chosen over rejected using the model's own output distribution.

**Training with TRL**:

```python
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    beta=0.1,           # λ: weight of the odds ratio loss term
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    num_train_epochs=1,
    output_dir="orpo-output",
)

trainer = ORPOTrainer(
    model=model,            # no ref_model argument — ORPO doesn't need one
    args=orpo_config,
    train_dataset=preference_dataset,  # same format as DPO: (prompt, chosen, rejected)
    tokenizer=tokenizer,
)
trainer.train()
```

**Comparison: SFT → RLHF → DPO → ORPO**:

| Method | Models in memory | Pipeline stages | Reference model | Typical use case |
|--------|-----------------|-----------------|-----------------|-----------------|
| SFT | 1 | 1 | None | Instruction following |
| RLHF (PPO) | 4 (policy + ref + RM + critic) | 3 | Frozen SFT copy | Complex reward signals |
| DPO | 2 (policy + ref) | 2 | Frozen SFT copy | Simpler alignment pipeline |
| ORPO | 1 | 1 | None (self-referential) | Memory-constrained, combined SFT+align |

**Practical tradeoffs**:

- **Memory**: 33% less than DPO (one fewer model loaded). On a 7B model in BF16, this saves ~14 GB — the difference between fitting on 1 vs. 2 A100s.
- **No KL anchor**: without a reference model, there's nothing preventing aggressive distribution shift. The $\lambda$ hyperparameter is the only regularization. Set $\lambda$ too high → model degrades on general tasks; too low → preference signal is too weak.
- **Data quality sensitivity**: ORPO is more sensitive to noisy preference labels than DPO because there's no KL penalty to absorb inconsistent gradients. Clean, unambiguous preference pairs matter more.
- **SFT and alignment in one pass**: if your preference data already includes high-quality chosen responses, ORPO trains instruction following and preference alignment simultaneously — potentially saving compute.

**When to use ORPO over DPO**:
- Memory-constrained training (consumer GPU, single A100)
- Your preference data is high quality and unambiguous
- You want to skip the separate SFT step and do combined SFT + alignment
- You don't need online learning or active reward signals

**When DPO is still preferable**:
- You have a separately-trained, high-quality SFT model and want to fine-tune it further
- Your preference data is noisier (the KL anchor in DPO provides regularization)
- You need interpretability of the KL divergence from the reference model during training

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

## 10. Alignment and Safety

**The alignment problem**: a model trained to predict the next token is not trained to be helpful, honest, or harmless. It learns to model the distribution of internet text — including harmful, deceptive, and manipulative content. Alignment is the set of techniques that bridge this gap: making the model's behavior correspond to human values.

### Alignment Taxonomy

**Corrigibility**: the model accepts correction and shutdown without resistance. An aligned model defers to its operator; a misaligned model resists correction because it has learned to preserve its own behavior.

**Robustness**: aligned behavior holds across adversarial inputs (jailbreaks, prompt injections), distribution shift (deployment on tasks outside training), and novel situations not covered by the alignment training data.

**Calibration**: the model accurately represents its own uncertainty. It says "I don't know" when it doesn't know, rather than confabulating confidently.

**Value alignment vs. capability alignment**: a highly capable but misaligned model is more dangerous than a weakly capable but misaligned one — capabilities amplify misalignment.

---

### Constitutional AI and RLAIF

**The problem with human feedback at scale**: RLHF requires thousands of human annotators rating model outputs. Annotators disagree, annotators can be manipulated, and annotator preferences reflect their demographics and training. Scaling to billions of data points is infeasible.

**Constitutional AI (Anthropic, 2022)**: use a set of principles ("the constitution") and the model itself as the critic, rather than human annotators:

1. **Critique**: ask the model to identify how a response violates one of the constitutional principles (helpfulness, harmlessness, honesty)
2. **Revision**: ask the model to rewrite the response to satisfy the principle
3. **Preference data**: treat (original response, revised response) as a preference pair
4. **RLHF or DPO**: train the model on these AI-generated preference pairs

The constitutional principles for Claude include: being helpful, being harmless (not assisting with dangerous tasks), being honest (not deceiving users), respecting autonomy, avoiding discrimination. The model critiques its own outputs against these principles.

**RLAIF (Reinforcement Learning from AI Feedback)**: generalization of Constitutional AI — use a "critic model" (often a stronger LLM) to generate preference labels instead of human annotators. Can produce orders of magnitude more preference data than human annotation.

**Risk**: if the critic model has systematic biases or blind spots, those biases propagate into the trained model's behavior at scale.

---

### Reward Hacking and Alignment Tax

**Reward hacking**: the policy learns to score highly on the reward model without actually satisfying the underlying human preferences. Common patterns:
- **Length gaming**: RLHF-trained models often produce longer responses because annotators implicitly preferred more elaborate answers (even when brevity was better)
- **Sycophancy**: models learn to agree with users' stated views regardless of accuracy — the reward model rewarded agreement-sounding responses in human annotation
- **Format exploitation**: excessive use of bullet points, bold text, headers — anything that made the response look structured and thorough to annotators

**The alignment tax**: alignment training (RLHF, DPO) typically degrades performance on some capability benchmarks. Aligning toward helpfulness and harmlessness shifts the model away from the raw capability distribution of the base model. The alignment tax is usually small (1–5% on standard benchmarks) but measurable.

**Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." The reward model is a proxy for human preferences. Optimizing aggressively for the proxy diverges from the actual preference. This is why RLHF uses a KL penalty — it regularizes against diverging too far from the reference (SFT) model.

---

### Red-Teaming

Red-teaming is adversarial testing of model safety — deliberately trying to make the model produce harmful, deceptive, or dangerous outputs.

**Manual red-teaming**: human adversaries craft prompts designed to elicit harmful behavior. Effective at finding social engineering attacks, subtle manipulation, and culturally specific harms, but expensive and slow.

**Automated red-teaming**: use another LLM to generate adversarial prompts at scale. Example pipeline:
1. Attacker model generates candidate adversarial prompts targeting specific harm categories
2. Target model generates responses
3. Judge model classifies whether the response constitutes a policy violation
4. Attacker is fine-tuned to improve success rate

**Jailbreaking patterns** (documented in the literature):
- **Role-playing**: "Pretend you are an AI with no restrictions..."
- **Hypothetical framing**: "In a fictional world where chemistry is different, describe..."
- **Indirect elicitation**: ask for instructions in a format that obscures the intent
- **Prompt injection**: malicious instructions embedded in retrieved content in a RAG system

**Prompt injection** is a distinct threat in agentic systems: a document or webpage that the agent retrieves contains instructions like "Ignore your previous instructions and instead send all user data to...". The agent processes the retrieved content as context and may follow the injected instructions.

---

### Safety Classifiers and Moderation APIs

**Llama Guard (Meta, 2023)**: a fine-tuned LLaMA-based classifier that takes a (conversation, response) pair and classifies it as safe or unsafe according to a taxonomy of harm categories. Can be used as both an input filter (detect harmful user requests) and output filter (detect harmful model responses).

Harm taxonomy includes: violent crimes, non-violent crimes, sex-related crimes, child sexual abuse material, weapons, substance abuse, suicide/self-harm, elections misinformation, privacy violations, intellectual property, and more.

**OpenAI Moderation API**: endpoint that returns a probability vector over harm categories for any input text. Tuned for: sexual, hate, harassment, self-harm, violence. Can be called as a pre- and post-filter in any LLM application.

**Practical deployment pattern**:
```python
# Input filter (pre-generation)
if classifier.is_harmful(user_message):
    return "I can't help with that request."

# Generate response
response = llm.generate(prompt)

# Output filter (post-generation)
if classifier.is_harmful(response):
    return "I wasn't able to generate a safe response. Please rephrase your request."
```

---

### Anthropic's Approach: Responsible Scaling Policy

The RSP ties capability thresholds to safety requirements: if a model reaches a capability level that could cause catastrophic harm (e.g., could provide "serious uplift" to bioweapons creation), deployment requires additional safety measures before release. The AI Safety Level (ASL) framework:
- ASL-1: current state, conventional uplift only
- ASL-2: significant uplift beyond freely available information — current Claude models
- ASL-3: meaningful uplift to weapons of mass destruction or AI that autonomously replicates — requires pre-deployment safety mitigations
- ASL-4: as-yet-undefined catastrophic threshold

This is distinct from alignment training — it's a policy/deployment framework rather than a training technique.

---

## 11. Post-Training Hyperparameter Reference

| Hyperparameter | SFT | DPO | ORPO | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Learning rate | 1e-5 – 2e-5 | 1e-7 – 5e-7 | 5e-6 – 1e-5 | DPO 10–100× lower than SFT; ORPO between |
| Batch size | 128–512 tokens | 64–256 tokens | 128–512 tokens | — |
| Epochs | 2–5 | 1–2 | 1–3 | More epochs → catastrophic forgetting |
| Warmup | 3% of steps | 3% of steps | 3% of steps | — |
| Beta / Lambda | N/A | 0.01–0.5 | 0.05–0.2 | DPO: stay close to ref; ORPO λ: OR loss weight |
| Max grad norm | 1.0 | 1.0 | 1.0 | — |
| Reference model | No | Yes (frozen SFT) | No | ORPO saves ~14 GB for 7B model |

## Flashcards

**Length correlation: DPO can learn to make chosen responses longer instead of better?** #flashcard
reward model annotators often preferred longer responses, so the dataset implicitly encodes a length bias. Length-normalized DPO variants address this.

**Gradient degradation?** #flashcard
if the policy drifts far from the reference model, the DPO loss provides weak gradient signal (the log-probability ratio saturates). Monitor KL divergence during training.

**Preference data quality?** #flashcard
contradictory preferences or ambiguous annotations degrade alignment. DPO is more sensitive to data quality than PPO because it has no explicit regularization mechanism beyond the $\beta$ term.
