# LLM Training Stability

## Executive Summary

Training a large language model is not a single optimization run — it is a months-long, multi-node, high-stakes process where instability can waste millions of dollars of compute. Loss spikes, gradient explosions, and silent numerical failures are routine. This guide covers the failure modes, detection methods, and recovery procedures that separate teams who successfully train LLMs from those who restart from scratch.

| Failure Mode | Signature | Primary Fix |
|-------------|-----------|-------------|
| Loss spike | Sudden 2-5x loss increase, brief | Rollback 100–500 steps; reduce LR |
| Gradient explosion | Grad norm >> clip threshold | Lower clip threshold; check for bad batch |
| Token collapse | Decreasing diversity, repetition | Check entropy of attention; reduce temperature in sampling during eval |
| FP16 underflow | Silent accuracy degradation | Switch to BF16 |
| KL explosion (RLHF) | KL penalty term spikes | Reduce RL learning rate; increase KL coefficient |
| Reward hacking | Reward rises, generation quality falls | Increase KL coefficient; diversify reward model |

---

## 1. Loss Spikes: Causes, Detection, and Recovery

### What a Loss Spike Looks Like

A loss spike is a sudden, transient increase in training loss — typically 2x to 10x the recent moving average — that either recovers on its own within a few hundred steps or diverges irreversibly.

**Spike vs. divergence:**
- **Spike:** Loss jumps but returns to previous trajectory within 200–500 steps. Manageable.
- **Divergence:** Loss jumps and does not recover, or oscillates with increasing amplitude. Requires rollback.

### Root Causes

**1. Bad batches:**
The most common cause. A batch containing an anomalous sequence — extremely long, containing degenerate tokens, or from a corrupted data shard — produces a loss that is orders of magnitude larger than normal. The gradient from this batch causes a large parameter update that destabilizes adjacent optimization.

Detection:
```python
# During training, log per-batch loss statistics
batch_loss = compute_loss(batch)
if batch_loss > 5 * running_average_loss:
    log_warning(f"Anomalous batch detected: loss={batch_loss:.2f}, avg={running_average_loss:.2f}")
    skip_batch = True  # Or: reduce LR for this step only
```

**2. Learning rate too high:**
The loss landscape of a large language model is non-convex with sharp curvature in some directions. A learning rate that is globally reasonable can be too high for occasional high-curvature regions encountered during training, causing oscillation or explosion.

**3. Weight overflow:**
In FP16 training, large parameter magnitudes can overflow the representable range ($2^{15} = 32768$). When a weight or gradient exceeds this, it becomes NaN or infinity, which propagates through the network. Gradient clipping prevents the gradient from overflowing; a master FP32 copy of weights (in mixed precision) prevents the weights from overflowing.

**4. Sequence length outliers:**
A sequence significantly longer than the training distribution's 99th percentile can produce disproportionately large attention matrices, large gradient magnitudes, and destabilize the batch.

### Recovery Procedure

**Step 1: Detect the spike.** Monitor loss with a rolling average. Alert if `current_loss > threshold * rolling_avg`. Typical threshold: 3–5x.

**Step 2: Assess severity.** If the loss is recovering naturally (next 50 steps show decreasing loss), continue and investigate the cause post-hoc. If the loss is not recovering after 100 steps, initiate rollback.

**Step 3: Roll back 100–500 steps.**

```bash
# Checkpoint naming convention: step_N.pt
# If spike detected at step 48200, roll back to step 47700
python train.py --resume_from_checkpoint step_47700.pt --learning_rate 2e-5
```

Rolling back 100–500 steps (not more) is usually sufficient because:
- The optimization trajectory before the spike was stable
- The problematic batch won't be re-encountered in the same order (due to data shuffling)
- More rollback wastes valid training progress

**Step 4: Identify and remove the bad data.** Log the batch contents at the spike step if possible. Bad data shard identification:
- Deduplication failures (repeated sequences with near-zero loss initially, then loss spikes on duplicates)
- HTML artifacts, boilerplate text, or data pipeline corruption
- Encoding errors producing invalid Unicode sequences

**Step 5: Resume with slightly lower LR.** Reduce the learning rate by 10–30% for the first 500 steps after rollback to allow the model to stabilize before returning to the previous schedule.

---

## 2. Token Collapse and Repetition

### What Token Collapse Is

Token collapse is a training failure mode where the model begins generating highly repetitive or degenerate text — repeating phrases, outputting only punctuation, or collapsing to a small subset of the vocabulary. The cross-entropy training loss can continue to decrease even as generation quality degrades, because the model becomes increasingly confident about a limited set of patterns.

### Why It Happens

**1. Perplexity divergence:** During RLHF or fine-tuning, if the reward signal overwhelmingly favors a specific pattern (e.g., starting every response with "Certainly!"), the model over-optimizes for that pattern. This reduces vocabulary entropy while maintaining low cross-entropy loss on a narrow distribution.

**2. Length penalty miscalibration:** If the reward model implicitly rewards longer responses, the model learns to repeat content to maximize length.

**3. Gradient scale mismatch:** During RLHF with PPO, if the policy gradient scale is much larger than the language modeling loss scale, the RL signal can overwhelm the language prior and cause distribution collapse.

**4. Reinforcement of early-training biases:** In the first 1,000 steps of fine-tuning, if the model strongly fits a small fine-tuning dataset, high-loss tokens (rare words, technical terms) get suppressed.

### Detection Metrics

**Attention entropy:** Average entropy of the attention distribution across heads. A healthy model has high-entropy attention (attending to many tokens). Collapsing attention entropy (becoming peaky) is an early signal of repetition.

```python
def compute_attention_entropy(attn_weights):
    # attn_weights: [batch, heads, seq_len, seq_len]
    eps = 1e-8
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    return entropy.mean().item()
```

**N-gram repetition rate:** Fraction of output tokens that appear in the preceding $n$ tokens.

**Vocabulary coverage:** During evaluation, fraction of vocabulary tokens appearing in a sample of 1,000 generated sequences. Healthy models use > 30% of vocabulary; collapsing models may drop to < 5%.

### Recovery

- Stop training immediately; rollback to pre-collapse checkpoint
- Reduce learning rate or RL step size
- Increase KL penalty coefficient (see RLHF section)
- Augment training data with high-diversity examples

---

## 3. Gradient Clipping

### Why Norm-Based Clipping, Not Value-Based

**Value-based clipping** clips each gradient component independently:
$$g_i \leftarrow \text{clip}(g_i, -c, c)$$

This distorts the gradient direction — it clips large components disproportionately while leaving small components unchanged, changing where in parameter space the update points.

**Norm-based clipping** rescales the entire gradient vector:
$$g \leftarrow g \cdot \frac{\min(c, \|g\|_2)}{\|g\|_2}$$

This preserves the direction of the gradient and only reduces its magnitude when the norm exceeds the threshold $c$. The update moves in the correct direction, just with a bounded step size.

```python
# PyTorch norm-based gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# This is equivalent to:
total_norm = torch.norm(torch.stack([
    torch.norm(p.grad.detach(), 2) for p in model.parameters()
]))
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for p in model.parameters():
        p.grad.detach().mul_(clip_coef)
```

### Typical Clipping Threshold: 1.0

Most production LLM training uses `max_norm=1.0`. This comes from:

- **GPT-3 (OpenAI, 2020):** gradient clip at 1.0
- **PaLM (Google, 2022):** gradient clip at 1.0
- **LLaMA (Meta, 2023):** gradient clip at 1.0

Why 1.0 specifically? It is a pragmatic choice: aggressive enough to prevent explosion, permissive enough not to over-restrict updates during normal training. The gradient norm of a well-trained transformer at scale hovers around 0.3–0.7 during stable training.

**Monitoring the clip rate:** If gradients are being clipped on > 50% of steps, the learning rate is too high or the data is pathological. The clip threshold is not a substitute for a well-tuned learning rate.

```python
# Log gradient norm to detect systematic issues
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
wandb.log({"grad_norm": grad_norm.item()})

# Alert if grad_norm >> 1.0 frequently
if grad_norm > 5.0:
    logger.warning(f"Large gradient norm: {grad_norm:.2f}")
```

---

## 4. Mixed Precision Pitfalls

### FP16 Underflow

FP16 has a representable range of approximately $6 \times 10^{-8}$ to $65504$. Gradients during LLM training can be much smaller, causing **underflow**: small gradients round to zero before contributing to parameter updates.

This is a **silent failure** — no NaN, no spike, just gradual degradation of training signal. The model still trains, but slowly or incorrectly, as gradients for certain parameters vanish.

**Solution: loss scaling.** Multiply the loss by a large constant before backpropagation, then divide the gradients by the same constant before the optimizer step. This shifts small gradients into the representable FP16 range.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(init_scale=2**16)

with autocast():
    loss = model(x)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

The `GradScaler` automatically adjusts the scale: it reduces the scale when it detects NaN/inf gradients (indicating overflow), and gradually increases it when training is stable (to address potential underflow).

### BF16 Advantages

BF16 (Brain Float 16) was designed specifically for deep learning by Google. Its key advantage over FP16:

| Property | FP16 | BF16 |
|----------|------|------|
| Exponent bits | 5 | 8 (same as FP32) |
| Mantissa bits | 10 | 7 |
| Representable range | ±65504 | ±3.4 × 10^38 (same as FP32) |
| Overflow risk | High | Extremely low |
| Underflow risk | Moderate | Low |
| Precision | Higher mantissa | Lower mantissa |

**BF16 eliminates loss scaling requirements** because its exponent range matches FP32 — gradients and activations are unlikely to overflow. The lower mantissa precision (7 bits vs 10) is acceptable because gradient descent is inherently noisy.

```python
# BF16 training — no GradScaler needed
model = model.to(torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Hardware availability:** A100, H100, and TPU v3+ support BF16 natively. Older V100 GPUs do not. If training on A100+, prefer BF16 over FP16.

---

## 5. Checkpoint Averaging and Model Merging

### Checkpoint Averaging for Stability

Training loss oscillates even in a healthy run. The model at the final step may be at a local minimum in an unstable region. Averaging weights across the final $k$ checkpoints produces a smoother, often better-generalizing model.

```python
import torch

checkpoint_paths = [f"step_{n}.pt" for n in range(95000, 100500, 500)]
averaged_weights = {}

for path in checkpoint_paths:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    for key, value in state_dict.items():
        if key not in averaged_weights:
            averaged_weights[key] = value.float()
        else:
            averaged_weights[key] += value.float()

for key in averaged_weights:
    averaged_weights[key] /= len(checkpoint_paths)

# Save averaged model
torch.save({"model_state_dict": averaged_weights}, "averaged_model.pt")
```

**Typical practice:** Average the last 5–20 checkpoints spaced 500–1,000 steps apart. Stochastic Weight Averaging (SWA) formalizes this with a cyclic learning rate schedule.

### Model Merging (SLERP, TIES, DARE)

For post-training stability, model merging combines multiple fine-tuned models or the base model with a fine-tuned model.

**SLERP (Spherical Linear Interpolation):**
Interpolates model weights on the parameter space sphere rather than linearly:
$$\text{slerp}(\theta_1, \theta_2, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \theta_1 + \frac{\sin(t\Omega)}{\sin\Omega} \theta_2$$

Where $\Omega = \arccos(\hat\theta_1 \cdot \hat\theta_2)$.

SLERP is used when merging a base model with a fine-tuned model to recover some base model capabilities while retaining fine-tuning gains.

**TIES-Merging:** Resolves sign conflicts between task vectors from multiple fine-tuned models — prevents parameter cancellation during merging.

**DARE:** Randomly prunes delta weights before merging; reduces interference between models.

---

## 6. Learning Rate Warmup

### Why Cold Learning Rate Kills Early Training

At initialization, model weights are random — the loss landscape from the model's perspective is steep and poorly understood. A high initial learning rate causes large updates from random gradients, potentially pushing parameters into regions far from any good solution, from which recovery may take thousands of steps.

**Warmup schedule:** Start with a very small LR and linearly (or cosine) increase to the target LR over the first $T_{\text{warmup}}$ steps.

```python
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-9):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay after warmup
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Typical warmup configurations:**

| Model scale | Warmup steps | Rationale |
|------------|-------------|-----------|
| 7B parameters | 2,000 | ~0.1% of 2M total steps |
| 70B parameters | 2,000–4,000 | Larger model, more sensitive initialization |
| Continual pretraining | 100–500 | Existing weights are already stable |
| Fine-tuning | 50–500 | Much shorter total training |

**Why warmup helps:**
- Early gradients are noisy; small LR reduces the variance of early updates
- Adam's second moment estimate ($v_t$) starts at zero and needs a few hundred steps to stabilize — a large LR before this stabilization amplifies the noise
- Warmup is particularly critical for very large models where the loss landscape curvature changes rapidly in early training

---

## 7. The Loss Spike Recovery Trick

### Roll Back 100–500 Steps

This is the standard production playbook for LLM training spikes:

```
Detection: loss > 3x rolling_100_step_avg
           OR grad_norm > 10.0 for 3 consecutive steps
           OR loss is NaN/inf

Decision tree:
├── Loss recovering (dropping after spike)?
│   └── Continue + investigate data shard post-hoc
│
└── Loss not recovering after 100 steps?
    ├── Identify last stable checkpoint (step N)
    ├── Roll back to step N - 200 (or N - 500 for severe spikes)
    ├── Resume training with:
    │   ├── Same checkpoint weights
    │   ├── LR reduced by 20% for next 500 steps
    │   └── Data pipeline skipping or reshuffling the suspected bad shard
    └── Monitor for 1,000 steps before declaring stability
```

**Why not roll back further?**
- Rolling back 5,000 steps wastes a day of compute on a large cluster
- The optimization trajectory is path-dependent; rolling back too far can undo useful learning
- Most spikes are caused by a single bad batch that won't recur after reshuffling

**Checkpoint frequency recommendation:** Save a checkpoint every 500–1,000 steps for models up to 70B. Every 250 steps for larger models or when training on a known-unstable data mixture. On Azure, use Azure Blob Storage with lifecycle policies to retain the last 20 checkpoints and archive older ones to cold storage.

---

## 8. Monitoring: Loss Curve Shape, Grad Norm, Attention Entropy

### The Monitoring Stack

A production LLM training run should log the following metrics every 10–50 steps:

**Loss metrics:**
- `train/loss`: raw batch loss
- `train/loss_rolling_100`: rolling 100-step average (smoothed)
- `train/perplexity`: $e^{\text{loss}}$, more interpretable for cross-entropy

**Gradient metrics:**
- `train/grad_norm`: pre-clip gradient norm; should stay in 0.2–2.0 for stable training
- `train/clip_fraction`: fraction of steps where clipping occurred; alert if > 0.5

**Activation and attention metrics:**
- `train/attention_entropy`: average entropy of attention weights; dropping entropy → repetition risk
- `train/activation_norm`: L2 norm of layer activations; sudden increase signals instability

**Learning rate and step metrics:**
- `train/learning_rate`: current LR (useful for warmup/decay verification)
- `train/tokens_seen`: cumulative tokens; normalizes loss curves across training runs

**Model quality metrics (every 1,000 steps):**
- Validation loss on a fixed held-out set
- Downstream task benchmarks (MMLU-subset, HumanEval-subset)
- Vocabulary entropy of generated samples

### Loss Curve Shape Interpretation

| Shape | Interpretation |
|-------|---------------|
| Smooth monotone decrease | Healthy training |
| Sudden jump, then recovery | Data spike; investigate batch |
| Oscillating with increasing amplitude | LR too high; reduce by 2x |
| Plateau followed by sudden decrease | Escaped a local flat region; normal |
| Slow creeping increase over many steps | Distribution shift in data pipeline |
| Immediate divergence at step 1 | LR far too high; check warmup |

---

## 9. Common Failure Modes in RLHF Training

### Reward Hacking

**Definition:** The policy learns to maximize the reward model's score through behaviors that do not reflect genuine quality improvement. The reward model is a proxy for human preference, and proxies can be gamed.

**Common reward hacking patterns:**
- Verbosity: reward model was trained on preferences that correlate with length; policy learns to write excessively long responses
- Sycophancy: model learns to agree with any premise in the prompt regardless of factuality
- Format exploitation: bullet points, numbered lists, specific phrases that reward annotators tended to prefer

**Detection:** Track KL divergence between policy and reference model alongside reward. If reward increases while KL grows rapidly, suspect reward hacking.

```python
# KL penalty in PPO objective
kl_divergence = torch.mean(
    ref_log_probs - policy_log_probs  # Approximate KL
)
penalized_reward = reward - kl_coef * kl_divergence
```

**Fix:** Increase the KL coefficient $\beta$ (stronger penalty for diverging from the reference model). Typical range: 0.01–0.3. Higher values produce more conservative, less reward-hacky outputs but also less improvement from RL.

### KL Explosion

**Definition:** The KL divergence between the RLHF policy and the reference SFT model grows without bound, causing the policy to produce incoherent or degenerate text.

**Causes:**
- RL learning rate too high
- KL coefficient too low
- Reward model has high variance or is overfit (small reward model trained on few comparisons)
- Too many PPO iterations per batch

**Detection:** Monitor `kl_divergence` per step. Alert if KL > 20 nats (a strong signal of instability; healthy RLHF runs maintain KL of 2–8 nats).

**Recovery:**
```python
# Adaptive KL controller (used in InstructGPT)
target_kl = 6.0
kl_penalty_coeff = 0.2  # Initial value

# After each PPO epoch:
if measured_kl > 1.5 * target_kl:
    kl_penalty_coeff *= 1.5  # Increase penalty
elif measured_kl < 0.5 * target_kl:
    kl_penalty_coeff /= 1.5  # Reduce penalty
```

### Mode Collapse in DPO

In Direct Preference Optimization, mode collapse occurs when the model assigns near-zero probability to the losing responses so quickly that the gradient signal from the preference loss becomes negligible. Training continues but the model makes no further improvement.

**Detection:** Monitor the log-probability margin between winning and losing responses. If the margin grows faster than 10 nats/epoch, suspect mode collapse.

**Fix:** Reduce DPO learning rate, add a reference regularization term, or use IPO (Identity Preference Optimization) which is more robust to this failure.

---

## 10. Interview Q&A

**Q1: What causes loss spikes during LLM training and how do you recover?**

> Loss spikes are most often caused by anomalous data batches — corrupted sequences, degenerate tokens, or length outliers — or by a learning rate that is too high for the local curvature of the loss landscape. FP16 overflow is a third cause in mixed precision training. Recovery: if the loss recovers within 100 steps, continue and investigate the data shard post-hoc. If not, roll back 100–500 steps to the last stable checkpoint, resume with a 20% lower LR for 500 steps, and reshuffle or skip the suspected bad data shard.

**Q2: Why do we use norm-based gradient clipping instead of value-based?**

> Value-based clipping clips each gradient component independently, which distorts the direction of the gradient update — it disproportionately reduces large components and leaves small ones unchanged. Norm-based clipping rescales the entire gradient vector uniformly when its L2 norm exceeds a threshold, preserving the gradient direction and only bounding the step size. This is crucial for optimization stability because the gradient direction carries information about which way to move in parameter space; distorting it introduces systematic bias.

**Q3: What is the typical gradient clipping threshold for LLM training and why?**

> 1.0 is the standard across GPT-3, PaLM, LLaMA, and most other production LLMs. During stable training, the gradient norm hovers around 0.3–0.7, so a threshold of 1.0 only activates when there is genuine instability. A threshold that is too low (e.g., 0.1) would clip on nearly every step and slow training; too high (e.g., 10.0) provides no protection against spikes.

**Q4: What are the advantages of BF16 over FP16 for LLM training?**

> BF16 uses 8 exponent bits (same as FP32), giving it the same representable range (~10^38) compared to FP16's 5 exponent bits and range of ~65,000. This eliminates overflow risk entirely and greatly reduces underflow risk, removing the need for loss scaling. The tradeoff is lower mantissa precision (7 bits vs 10), but gradient computations tolerate this imprecision well. On A100 and H100 GPUs with native BF16 tensor cores, switching from FP16 to BF16 is essentially a free stability improvement.

**Q5: What is learning rate warmup and why is it necessary?**

> Warmup linearly increases the learning rate from near-zero to the target LR over the first 1,000–4,000 steps. It is necessary because (1) early gradients from random initialization are noisy and high-variance; a large LR amplifies this noise into destructive parameter updates; (2) Adam's adaptive second moment estimate starts at zero and requires ~100 steps to provide accurate gradient scaling — a large LR before stabilization is uncontrolled. Skipping warmup often causes immediate training divergence, particularly for models with many layers.

**Q6: What is reward hacking in RLHF and how do you detect it?**

> Reward hacking occurs when the policy learns to maximize the reward model's score through behaviors the reward model wasn't trained to penalize — e.g., verbosity, sycophancy, specific formatting patterns — rather than through genuine quality improvement. Detection: track KL divergence from the reference model alongside reward. If reward improves while KL grows rapidly (> 15 nats), the policy is diverging from the reference in a way that suggests hacking rather than genuine learning. Also monitor generation diversity, response length distribution, and downstream human evaluations.

**Q7: What is KL explosion in RLHF and how do you prevent it?**

> KL explosion is when the KL divergence between the RLHF policy and the reference SFT model grows without bound, causing the policy to produce incoherent text. It occurs when the RL learning rate is too high or the KL penalty coefficient is too low. Prevention: use an adaptive KL controller (as in InstructGPT) that increases the penalty coefficient when measured KL exceeds a target (e.g., 6 nats) and decreases it when KL is too low. Monitor KL continuously; alert if KL > 20 nats.

**Q8: How does checkpoint averaging improve training stability?**

> Training loss oscillates even in a healthy run, and the final-step checkpoint may sit at an unstable point in the loss landscape. Averaging weights across the last 10–20 checkpoints (spaced 500 steps apart) produces a smoother model that is geometrically more centered in a flat region of the loss landscape. Stochastic Weight Averaging (SWA) formalizes this: use a cyclic LR schedule to explore parameter space during the final phase of training, then average the checkpoints visited at the trough of each cycle. The averaged model typically has lower validation loss and better downstream task performance than any individual checkpoint.

**Q9: What metrics would you monitor to detect early signs of training instability?**

> Gradient norm (pre-clip): should stay in 0.2–2.0; sustained values > 3.0 or sudden spikes to > 10 indicate instability. Clip fraction: if > 50% of steps are being clipped, the learning rate is too high. Attention entropy: dropping entropy signals emerging repetition. Loss rolling average: the 100-step rolling average should decrease monotonically; plateaus are normal, but increases exceeding 20% of the moving average trigger investigation. For RLHF specifically: KL divergence from reference and reward model score distribution (not just mean reward).

**Q10: What is token collapse and how does it differ from a loss spike?**

> A loss spike is a sudden, visible increase in training loss — detectable on the loss curve. Token collapse is a silent, gradual degradation: the loss may continue to decrease while the model converges on repetitive, low-diversity outputs. The model is "learning" to predict a narrow distribution of tokens with high confidence. Detection requires monitoring metrics beyond loss: vocabulary coverage in generated samples, n-gram repetition rate, and attention entropy. Token collapse is most common in RLHF when the reward model has a strong preference for specific patterns, or in fine-tuning on a small, low-diversity dataset.
