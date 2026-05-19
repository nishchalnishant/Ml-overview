# LLM Training Stability

---

## 1. The Canonical Disaster: Loss Spikes to Infinity at Step 50,000

Picture a training run that has been going well for six days. Loss is smoothly declining, gradient norms are stable around 0.4. Then, at step 50,147, the loss jumps from 2.1 to 47.3 and doesn't come back. Six days of compute are either wasted or you're rolling back.

This scenario is not rare — it is routine in large-scale LLM training. Every major technique in this section exists as a direct response to a concrete failure mode like this one.

---

## 2. Loss Spikes: Causes, Detection, and Recovery

### What actually causes a loss spike

**Bad batches** are the most common cause. A batch containing an anomalous sequence — extremely long, encoding-corrupted, containing degenerate repeated tokens, or from a mislabeled data shard — produces a loss that is orders of magnitude larger than the average. The gradient from this one batch causes a parameter update large enough to push the model out of the stable training regime it has developed over thousands of steps.

**Learning rate too high for local curvature**: the loss landscape of a large transformer is not uniformly curved. Most of the time, the current learning rate is well-suited to the local geometry. Occasionally the model enters a high-curvature region — a sharp valley — where the same step size causes it to overshoot and bounce. This looks like a spike on the loss curve.

**FP16 weight overflow**: FP16 has a maximum representable value of ~65,504. Large activations or gradient magnitudes can exceed this, producing NaN or infinity, which propagates through all subsequent computations. The run collapses, not from bad data but from numerical overflow.

### Detecting spikes in real time

```python
# Log per-batch loss and maintain a rolling average
SPIKE_THRESHOLD = 3.0

def check_for_spike(batch_loss: float, running_avg: float) -> bool:
    if batch_loss > SPIKE_THRESHOLD * running_avg:
        logger.warning(f"Loss spike: {batch_loss:.2f} vs avg {running_avg:.2f}")
        return True
    return False
```

Also monitor gradient norm (pre-clip). If the gradient norm jumps to 10× its recent average on the same step as a loss spike, the two are almost certainly linked.

### Recovery procedure

**Step 1: Assess whether the loss is recovering.** A spike that recovers within 100–200 steps is not an emergency — investigate the batch post-hoc but continue training.

**Step 2: If not recovering after 100 steps, roll back.** Load the checkpoint from 100–500 steps before the spike. Rolling back more than 500 steps wastes valid training progress; less than 100 steps risks hitting the same bad batch again before data reshuffling takes effect.

```bash
# If spike detected at step 48,200, roll back to step 47,700
python train.py --resume_from_checkpoint step_47700.pt --learning_rate 2e-5
```

**Step 3: Resume with a slightly reduced learning rate.** Reduce by 10–30% for the first 500 steps after rollback to let the model re-stabilize before returning to the normal schedule.

**Step 4: Identify and quarantine the bad data.** If the spike correlates with a specific data shard, inspect it for: encoding errors, HTML artifacts, repetitive near-duplicate sequences, or sequences far outside the normal length distribution.

**Why not roll back 5,000 steps?** Data order is randomized per epoch. The bad batch that caused the spike is extremely unlikely to appear again after a reshuffle. The optimization trajectory from steps $N-5000$ to $N-100$ represents real learning that would be thrown away.

**Checkpoint frequency recommendation**: every 500–1,000 steps for models up to 70B. Every 250 steps for models larger than 70B or when training on a known-unstable data mixture.

---

## 3. Gradient Clipping: Why It Exists and What It Actually Does

**The problem**: during a bad batch or a high-curvature region of the loss landscape, the gradient can be enormous — not 2× the normal magnitude but 100× or 1,000×. A single gradient update of this size destroys weeks of training. You need a hard upper bound on how large any single update can be.

**Why not clip each gradient component independently?** Value-based clipping does this: $g_i \leftarrow \text{clip}(g_i, -c, +c)$. The problem: it clips large components disproportionately while leaving small components unchanged. The resulting gradient vector points in a *different direction* than the original — it is biased toward the small-gradient parameters. You've not just bounded the step; you've changed where in parameter space the update is pointing.

**The core insight**: what matters is that the *step size* is bounded. The *direction* should be preserved. Scale the entire gradient vector down uniformly if its norm exceeds the threshold.

**The mechanics — norm-based clipping**:

$$g \leftarrow g \cdot \frac{\min(c, \|g\|_2)}{\|g\|_2}$$

This preserves the gradient direction exactly and only reduces magnitude when necessary.

```python
# Standard PyTorch implementation
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Equivalent implementation to understand what it does:
total_norm = torch.norm(
    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()])
)
clip_coef = 1.0 / (total_norm + 1e-6)
if clip_coef < 1:
    for p in model.parameters():
        p.grad.detach().mul_(clip_coef)
```

**Why the threshold is 1.0**: GPT-3, PaLM, LLaMA — essentially every large public LLM — uses `max_norm=1.0`. During stable training, gradient norms hover around 0.3–0.7. A threshold of 1.0 only activates during genuine instability, not during normal training steps. A threshold of 0.1 would clip on nearly every step and throttle learning; a threshold of 10.0 would provide no protection against spikes.

**Using the clip fraction as a diagnostic**: if gradients are being clipped on more than 50% of steps, the learning rate is too high or the data contains systematic pathologies. The clipping threshold is not a substitute for proper learning rate tuning.

```python
# Log gradient norm to detect systematic issues
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
wandb.log({"grad_norm": grad_norm.item()})

if grad_norm > 5.0:
    logger.warning(f"Large gradient norm at step {step}: {grad_norm:.2f}")
```

---

## 4. Learning Rate Warmup: Why You Can't Start at Full Speed

**The problem**: at initialization, model weights are random. Adam's second moment estimate $v_t$ is initialized to zero. The bias correction term $1/(1-\beta_2^t)$ compensates in theory, but in the first few hundred steps, the effective learning rate computed by Adam is poorly calibrated — the variance estimates haven't stabilized yet. If you start at the full learning rate (e.g., $3 \times 10^{-4}$), those first few steps apply a massive, poorly-calibrated update from noisy initial gradients. The model parameters get pushed to a bad region of the loss landscape that may take thousands of steps to escape — or it may never escape.

**The core insight**: give Adam's internal statistics time to stabilize before applying large updates. Start with a learning rate near zero and linearly ramp to the target over the first few thousand steps.

**The mechanics**:

```python
def cosine_schedule_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # linear ramp
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
```

| Scenario | Warmup steps | Rationale |
| :--- | :--- | :--- |
| Pretraining 7B | 2,000 | ~0.1% of total steps |
| Pretraining 70B+ | 2,000–4,000 | Larger model, steeper initialization |
| Continual pretraining | 100–500 | Weights already stable |
| SFT / DPO | 50–500 | Short total training run |

**What breaks without warmup**: not a gradual degradation — training often diverges immediately or within the first hundred steps when warmup is omitted at large scale. The random initialization produces gradients whose magnitude varies wildly across parameters, and a large learning rate amplifies this into destructive updates before Adam has any useful statistics.

---

## 5. Mixed Precision Training: Why FP16 Fails and BF16 Fixes It

**The problem**: training in FP32 (4 bytes per value) is expensive. A 7B parameter model with its optimizer states uses ~100 GB in FP32. But reducing precision introduces risks that can silently corrupt training.

### The FP16 trap: silent gradient underflow

FP16's representable range is approximately $6 \times 10^{-8}$ to $65,504$. LLM gradients during normal training can be much smaller than $6 \times 10^{-8}$. When a gradient underflows FP16, it rounds to zero — the corresponding parameter receives no update. This is a **silent failure**: no NaN, no spike, no error message. The model continues training, but certain parameters stop learning. The effect is subtle degradation of model quality, hard to diagnose without careful gradient monitoring.

**The loss scaling workaround**: multiply the loss by a large constant $S$ before backprop, which scales all gradients up by $S$, shifting them into FP16's representable range. Divide by $S$ before the optimizer step. The `GradScaler` manages $S$ automatically, increasing it when training is stable and decreasing it when it detects overflow (NaN gradients).

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

### Why BF16 eliminates these problems

BF16 was designed by Google specifically for deep learning. The key difference: it uses 8 exponent bits (same as FP32) instead of FP16's 5. This gives BF16 the same representable *range* as FP32 (up to ~$3.4 \times 10^{38}$), eliminating overflow risk. The tradeoff is fewer mantissa bits (7 vs FP16's 10), meaning less precision per value — but gradient computations are inherently noisy, and this imprecision is empirically harmless.

| Format | Exponent bits | Mantissa bits | Max value | Overflow risk |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | 8 | 23 | $3.4 \times 10^{38}$ | None |
| FP16 | 5 | 10 | 65,504 | High |
| BF16 | 8 | 7 | $3.4 \times 10^{38}$ | None |

**Practical setup for BF16 training** (A100/H100 — no loss scaling needed):

```python
model = model.to(torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**What breaks**: older GPUs (V100 and earlier) don't support BF16 natively; on those, FP16 with loss scaling is the only option.

---

## 6. Token Collapse: When the Loss Looks Fine but the Model Is Dying

**The problem**: cross-entropy loss can keep decreasing even as a model converges toward degenerate outputs. The model becomes increasingly confident about a narrow distribution of tokens — repeating phrases, outputting only punctuation, defaulting to "Certainly! Here is..." — while genuine generative quality collapses.

**Why token collapse is invisible in the training loss**: the loss measures how well the model predicts its training data. If the model is being trained on its own repetitive outputs (as in RLHF), or if the reward signal strongly favors a specific pattern, the model learns to reproduce that pattern with high confidence. Confidence over a narrow distribution gives *low* loss. The loss curve looks healthy while the model becomes useless.

**What triggers it**:
- RLHF where the reward model implicitly rewards length → model learns to pad
- SFT on a small, low-diversity dataset → rare vocabulary tokens get suppressed
- Gradient scale mismatch in PPO → RL signal overwhelms the language prior

### Detection: look beyond loss

**Attention entropy**: a healthy model attends broadly across context. As token collapse begins, attention distributions become peaky — the model looks at fewer and fewer tokens.

```python
def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    # attn_weights: [batch, heads, seq_len, seq_len]
    eps = 1e-8
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    return entropy.mean().item()
```

**Vocabulary coverage**: in a healthy model, 1,000 generated sequences use more than 30% of the vocabulary. A collapsing model may drop below 5%.

**N-gram repetition rate**: fraction of output tokens that appeared in the preceding $n$ tokens. Values above ~15% for 4-grams are a warning sign.

### Recovery

Stop training immediately and roll back to the pre-collapse checkpoint. The collapse cannot be trained away from because the gradients themselves are pointing toward further collapse. After rollback: reduce learning rate, increase KL penalty if in RLHF, augment data with high-diversity examples.

---

## 7. RLHF-Specific Failures: Reward Hacking and KL Explosion

### Reward hacking

**The problem**: the reward model is a proxy for human preference, not a perfect measurement of it. Any proxy can be gamed. The policy is an optimizer — given enough gradient steps, it will find the behaviors that maximize reward model scores regardless of whether those behaviors are actually good.

Common patterns:
- **Verbosity**: annotators who trained the reward model tended to prefer longer responses; the policy learns to pad responses
- **Sycophancy**: model learns to agree with any stated premise in the prompt
- **Format exploitation**: specific phrases or bullet-point structures that reward annotators tended to prefer

**Detection**: plot reward alongside KL divergence from the reference model. Genuine improvement looks like: reward increases while KL grows slowly. Reward hacking looks like: reward increases while KL grows rapidly. The model is diverging from the reference model in a direction that scores well but isn't generalizing to real quality.

```python
kl_divergence = torch.mean(ref_log_probs - policy_log_probs)
penalized_reward = reward - kl_coef * kl_divergence
```

**Fix**: increase the KL coefficient $\beta$. Typical range: 0.01–0.3. Higher values keep the policy closer to the SFT model but limit the improvement achievable from RL.

### KL explosion

**The problem**: the KL divergence between the RLHF policy and the reference SFT model grows without bound. The policy produces increasingly incoherent text as it moves far from the distribution it was trained on. The language prior disintegrates.

**Causes**: RL learning rate too high, KL coefficient too low, reward model with high variance (overfit to small annotation set), too many PPO iterations per batch.

**Detection**: alert when KL > 20 nats. Healthy RLHF training maintains KL of 2–8 nats.

**The adaptive KL controller** (from InstructGPT):

```python
target_kl = 6.0
kl_coef = 0.2  # Starting value

# After each PPO epoch:
if measured_kl > 1.5 * target_kl:
    kl_coef *= 1.5    # Tighten the leash
elif measured_kl < 0.5 * target_kl:
    kl_coef /= 1.5    # Give more room to explore
```

### Mode collapse in DPO

**The problem**: DPO trains the policy to assign higher probability to preferred responses and lower probability to rejected responses. If the learning rate is too high, the model drives the probability of rejected responses to near-zero very quickly. Once log-probability of the rejected response is very negative, the gradient from that term becomes negligible — the loss is saturated. Training continues but the model makes no further improvement.

**Detection**: monitor the log-probability margin between preferred and rejected responses. If the margin grows faster than ~10 nats per epoch, suspect mode collapse.

**Fix**: reduce DPO learning rate, or switch to IPO (Identity Preference Optimization), which uses a squared loss that doesn't saturate.

---

## 8. Distributed Training Stability: ZeRO and Mixed Parallelism

**The problem**: a 70B model in BF16 requires 140 GB of parameter memory. Add FP32 optimizer states (Adam keeps $m_t$ and $v_t$ per parameter, both in FP32): $70\text{B} \times 3 \times 4 = 840$ GB. A single A100 has 80 GB. Training requires splitting model state across multiple GPUs.

**Naive data parallelism**: each GPU holds a full copy of the model and processes a different batch slice. Gradients are averaged across GPUs. This simply doesn't work for models that exceed single-GPU memory.

**The core insight — ZeRO (Zero Redundancy Optimizer)**: in data parallelism, every GPU holds the same optimizer states, gradients, and parameters. This is pure redundancy. Shard each across all $N$ GPUs.

| Stage | What is sharded | Memory per GPU vs baseline |
| :--- | :--- | :--- |
| ZeRO-1 | Optimizer states | ~$1/N$ of optimizer state |
| ZeRO-2 | Optimizer states + gradients | ~$1/N$ of optimizer state + gradients |
| ZeRO-3 | All of the above + parameters | ~$1/N$ total |

With ZeRO-3, a 70B model can be trained on 8× A100 80 GB GPUs. The cost is inter-GPU communication to reconstruct parameters before each forward pass — this communication overhead must be overlapped with computation.

```python
# DeepSpeed ZeRO-3 config
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},   # spill optimizer states to CPU RAM
        "offload_param": {"device": "cpu"},
    },
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
}
```

**What breaks**: ZeRO-3 requires all-gather communication before each layer's forward pass to reconstruct parameters. At small batch sizes or on high-latency interconnects, communication latency becomes the bottleneck rather than computation. Pipeline parallelism and tensor parallelism address different bottlenecks but introduce their own complexity.

---

## 9. Checkpoint Averaging: Getting a Better Model for Free

**The problem**: training loss oscillates even in a healthy run. The model at the final training step may sit at a locally unstable point in the loss landscape — slightly worse than the average trajectory.

**The core insight**: averaging weights across multiple checkpoints near the end of training produces a model that sits geometrically closer to the center of a flat loss basin, which generalizes better than any individual checkpoint.

```python
import torch

checkpoint_paths = [f"step_{n}.pt" for n in range(95000, 100500, 500)]
averaged_weights = {}

for path in checkpoint_paths:
    ckpt = torch.load(path, map_location="cpu")
    for key, value in ckpt["model_state_dict"].items():
        if key not in averaged_weights:
            averaged_weights[key] = value.float()
        else:
            averaged_weights[key] += value.float()

for key in averaged_weights:
    averaged_weights[key] /= len(checkpoint_paths)
```

Averaging the last 5–20 checkpoints spaced 500–1,000 steps apart typically improves validation loss and downstream benchmark scores over the final checkpoint alone.

---

## 10. Monitoring Stack: What to Track and When to Panic

A production LLM training run should log the following every 10–50 steps:

**Loss metrics**:
- Raw batch loss
- 100-step rolling average (smoothed signal for spike detection)
- Validation loss on a fixed held-out set (every 1,000 steps)

**Gradient metrics**:
- `grad_norm` (pre-clip): should stay 0.2–2.0 during stable training
- `clip_fraction`: fraction of steps where clipping occurred; alert if sustained above 50%

**Activation metrics**:
- Attention entropy: dropping entropy → repetition risk
- Activation norm per layer: sudden increases signal instability

**RLHF-specific**:
- KL divergence from reference (alert if > 20 nats)
- Reward model score distribution (watch for rapid reward increase with KL growth)
- Response length distribution (widening toward long tail → verbosity reward hacking)

| Loss curve shape | Interpretation |
| :--- | :--- |
| Smooth monotone decrease | Healthy |
| Sudden jump, then recovery within 200 steps | Data spike — investigate shard |
| Oscillating with increasing amplitude | Learning rate too high; reduce 2× |
| Plateau then sudden decrease | Escaped flat region; normal |
| Slow creeping increase over many steps | Distribution shift in data pipeline |
| Immediate NaN at step 1 | Learning rate far too high |
