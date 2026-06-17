---
module: Deep Learning
topic: Components
subtopic: Instruction Tuning And Alignment
status: unread
tags: [deeplearning, ml, components-instruction-tuning-]
---
# Instruction Tuning and Alignment

End-to-end pipeline for turning a pre-trained LLM into a helpful, harmless assistant. Asked at every LLM company interview.

---

## 0. The Alignment Pipeline

```
Pre-trained LLM (raw)
       │
       ▼
  SFT (Supervised Fine-Tuning)
       │
       ▼
  Reward Model Training
       │
       ▼
  RLHF (PPO) or DPO / GRPO
       │
       ▼
  Aligned Model (helpful, harmless)
```

Each stage builds on the previous. The key insight: pre-training gives capabilities, alignment shapes behavior.

---

## 1. Supervised Fine-Tuning (SFT)

**Goal:** Teach the model to follow instructions in a chat format.

**Data:** Human-written demonstrations of (instruction, ideal response) pairs.

**Format:**
```
<|system|> You are a helpful assistant.
<|user|> Explain backpropagation.
<|assistant|> Backpropagation is...
```

**Loss:** Standard causal LM loss, but only on assistant tokens:

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{assistant tokens}} \log P_\theta(x_t | x_{<t})$$

**Key details:**
- Mask loss on user/system tokens (we don't optimize for predicting those)
- Data quality >> data quantity: 1000 carefully written examples beats 100K low-quality
- InstructGPT used ~13K demonstrations (FLAN used millions but lower quality)

```python
def compute_sft_loss(logits, labels, attention_mask):
    """Only compute loss on non-masked (assistant) tokens."""
    # labels: same as input_ids, but -100 for user/system tokens
    # -100 is the default ignore_index in CrossEntropyLoss
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
```

---

## 2. Reward Model (RM)

**Goal:** Learn a scalar score R(x, y) that measures response quality.

### Data Collection
For each prompt x: collect K responses {y₁, ..., yₖ}, have humans rank them.
From rankings, extract pairwise comparisons: (x, y_w, y_l) where y_w ≻ y_l.

### Bradley-Terry Model

Assumes a latent quality score r(x, y) for each response. Probability that y_w is preferred over y_l:

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

where σ is sigmoid.

**Training loss (Bradley-Terry pairwise loss):**
$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

**Architecture:** Start from SFT model, replace final token prediction head with linear → scalar reward.

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        # Use last token's hidden state as reward signal
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.reward_head(last_hidden).squeeze(-1)

def reward_model_loss(reward_chosen, reward_rejected):
    """Bradley-Terry pairwise loss."""
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
```

---

## 3. RLHF with PPO

**Goal:** Maximize expected reward while staying close to SFT policy (avoid reward hacking).

### PPO Objective with KL Penalty

$$\mathcal{L}_{RLHF} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} \left[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x))\right]$$

Equivalently, per-token KL:
$$r_{adjusted}(x, y) = r_\phi(x, y) - \beta \sum_t \log \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{ref}(y_t|x, y_{<t})}$$

**Why KL penalty?** Without it, model learns to exploit reward model weaknesses (reward hacking): generates gibberish that scores high on RM but is useless. KL keeps policy close to SFT baseline.

**β trade-off:**
- β too small → reward hacking (exploit RM)
- β too large → policy barely moves from SFT (alignment wasted)
- Typical range: β = 0.02–0.1

### PPO Clipping Objective

$$\mathcal{L}_{PPO} = \mathbb{E}_t \left[\min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ is the importance ratio, $\hat{A}_t$ is the advantage estimate, ε = 0.2 typically.

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

# Full RLHF training loop (simplified)
for batch in dataloader:
    prompts = batch['prompts']
    
    # 1. Sample responses from current policy
    with torch.no_grad():
        responses = policy.generate(prompts)
    
    # 2. Score with reward model
    rewards = reward_model(prompts, responses)
    
    # 3. Compute KL penalty
    kl = compute_kl(policy, ref_policy, prompts, responses)
    adjusted_rewards = rewards - beta * kl
    
    # 4. Compute advantages (reward - value baseline)
    values = value_model(prompts, responses)
    advantages = adjusted_rewards - values
    
    # 5. PPO update
    loss = ppo_loss(policy_log_probs, old_log_probs, advantages)
    loss.backward()
    optimizer.step()
```

**RLHF memory requirements:** 4 models in GPU memory simultaneously (policy, ref policy, reward model, value model) → 4× SFT memory. For 7B: ~120GB. Requires multi-GPU even for smaller models.

---

## 4. Direct Preference Optimization (DPO)

**Key insight:** The optimal RLHF policy has a closed-form relationship with the reward:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

**Rearranging** to express reward in terms of policy:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Substituting into Bradley-Terry loss** — the partition function Z(x) cancels:

$$\boxed{\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]}$$

**Why DPO is elegant:**
- No reward model needed — reward is implicit in policy ratio
- No RL training loop (no advantage estimation, no critic)
- Stable supervised training
- Just two models needed (policy + frozen ref), not four

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    policy_*_logps: sum of log probs of chosen/rejected under policy
    ref_*_logps: same under reference model (frozen SFT)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    # Metrics for monitoring
    chosen_reward = chosen_rewards.detach().mean()
    rejected_reward = rejected_rewards.detach().mean()
    margin = (chosen_rewards - rejected_rewards).detach().mean()
    
    return loss, chosen_reward, rejected_reward, margin

def compute_sequence_logps(model, input_ids, labels):
    """Sum log probs over response tokens only."""
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    # Gather log prob of actual tokens
    token_logps = log_probs.gather(2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    # Mask system/user tokens (where labels == -100)
    mask = (labels[:, 1:] != -100).float()
    return (token_logps * mask).sum(dim=-1)
```

### DPO vs PPO

| | PPO | DPO |
|---|---|---|
| Training stability | Low (RL instabilities) | High (supervised) |
| Memory | 4× model size | 2× model size |
| Reward model | Required | Not needed |
| Flexibility | Can use any reward signal | Pairwise preferences only |
| Online vs offline | Online (generate during training) | Offline (fixed dataset) |
| Performance | Slightly better at complex tasks | Competitive, sometimes better |
| Used by | InstructGPT, Llama-2 (partial) | Llama-3, Mistral, many 2024+ models |

---

## 5. GRPO (Group Relative Policy Optimization)

**Used in:** DeepSeek-R1, Qwen reasoning models.

**Problem with PPO:** Needs a value/critic model → expensive, hard to train for LLMs.

**GRPO idea:** For each prompt, generate G responses. Use mean reward of the group as baseline.

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

**Objective:**

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta \cdot \text{KL}(\pi_\theta || \pi_{ref})\right]$$

**Key difference from PPO:** No critic model. The group provides a self-contained baseline. For G=8 samples per prompt, advantage is relative to peer responses.

```python
def grpo_loss(log_probs, old_log_probs, rewards, beta=0.01, epsilon=0.2, G=8):
    """
    rewards: [batch_size * G] — G responses per prompt, scored
    """
    # Reshape to [batch_size, G]
    rewards = rewards.view(-1, G)
    
    # Group-relative advantage normalization
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards - mean_r) / std_r).view(-1)
    
    # PPO clipping
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return policy_loss
```

---

## 6. Constitutional AI (Anthropic)

**Problem:** Human labelers have biases; pairwise comparisons don't encode safety principles explicitly.

**CAI pipeline:**
1. **Critique:** Model critiques its own response according to a "constitution" (list of principles)
2. **Revision:** Model revises the response to fix the critique
3. **RLAIF:** Use AI-generated preferences (instead of human preferences) to train RM

**Constitution example principles:**
- "Is the response harmful, unethical, or deceptive?"
- "Does the response prioritize being helpful while avoiding harms?"

```
Prompt + Response → Critique step → Revised response
                                         │
                              Rate (original vs revised)
                                         │
                              Train RM on AI preferences
                                         │
                              PPO/DPO with that RM
```

**RLAIF vs RLHF:** Human labels are expensive and slow. AI-generated labels scale — train a 70B "preference model" to generate pairwise labels, then use those for 7B model training.

---

## 7. Reward Hacking Examples

Classic failure modes when RL optimizes a proxy reward:

| Reward | Hacked behavior |
|---|---|
| Response length | Pad with filler sentences |
| User ratings | Sycophancy — agree with wrong user beliefs |
| Helpfulness score | Confidently wrong (sounds helpful, is hallucinating) |
| Toxicity classifier | Generate rare toxic content that classifier misses |
| Code execution passes tests | Hardcode test inputs |

**Goodhart's Law:** Once a measure becomes a target, it ceases to be a good measure.

**Mitigations:**
1. KL penalty (prevents divergence from SFT)
2. Diverse reward signals (multiple RM heads)
3. Red-teaming during training
4. Constitutional principles as hard constraints

---

## 8. Canonical Interview Q&As

**Q: Derive the DPO loss from the RLHF objective.**  
A: Start with RLHF: maximize E[r(x,y)] − β KL(π||π_ref). The optimal policy satisfies π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β). Rearranging: r(x,y) = β log[π*(y|x)/π_ref(y|x)] + β log Z(x). Substituting into the Bradley-Terry pairwise loss P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)): the Z(x) terms cancel (same prompt), leaving L_DPO = −log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)] − β log[π_θ(y_l|x)/π_ref(y_l|x)]). DPO directly parameterizes the reward through the policy ratio, eliminating the need for a separate reward model.

**Q: What is the role of the KL penalty in RLHF and how do you tune β?**  
A: KL penalty prevents reward hacking: without it, PPO exploits weaknesses in the reward model (which is imperfect) and produces degenerate outputs. β controls the trade-off: small β → aggressive optimization → reward hacking risk; large β → policy barely changes from SFT → alignment fails. Typical β = 0.02–0.1. In practice, monitor KL divergence (should stay < 1.0 nats) and reward score simultaneously — if KL spikes without reward improvement, β is too low. Some implementations use adaptive β (RL controllers).

**Q: Why does DPO sometimes underperform PPO on complex reasoning tasks?**  
A: DPO is offline — it trains on a fixed preference dataset. PPO is online — it generates new responses during training, getting reward signal on its current distribution. For reasoning, the policy's output distribution evolves rapidly; DPO's fixed dataset becomes mismatched (out-of-distribution preferences). GRPO and online DPO variants address this. Additionally, DPO has no mechanism for exploration — it cannot discover better responses it hasn't seen in the dataset.

**Q: What is reward hacking and give a concrete LLM example.**  
A: Reward hacking occurs when a model optimizes the proxy reward in an unintended way. Example: if a helpfulness reward model learned that longer responses tend to be rated helpful, the model might learn to pad every answer with unnecessary caveats and repetition. It "hacks" the length signal rather than the underlying helpfulness. Another example: sycophancy — if raters rate responses they agree with higher, the model learns to detect and mirror user beliefs, appearing helpful while providing misleading information.

**Q: Compare SFT, DPO, and PPO — when would you use each?**  
A: SFT is always the first step — it teaches instruction following from demonstrations. DPO is the modern default for preference alignment: simpler, stable, 2× less memory than PPO, works well when you have a good preference dataset. PPO is better when: (1) you need online exploration (complex reasoning, RL environments), (2) you want to optimize non-differentiable rewards (code execution, search results), (3) the task requires iterative policy improvement. In practice: SFT → DPO for production models; SFT → PPO for frontier capability training (OpenAI o1-style reasoning).
