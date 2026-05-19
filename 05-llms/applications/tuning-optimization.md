# Tuning and Optimization

---

## Diagnosing the Problem First

**The problem:** a model behaves incorrectly. The temptation is to reach for the most powerful-sounding tool — fine-tuning, RLHF — when a cheaper intervention would have fixed the problem. Each intervention has a cost; using the wrong one wastes compute and often makes things worse.

**The core insight:** the correct intervention depends on what kind of wrong the model is. Knowledge wrong (stale facts, private data) → RAG. Behavior wrong (wrong tone, wrong format, wrong task execution) → fine-tuning. Alignment wrong (too harmful, too refusal-happy, sycophantic) → RLHF or DPO. Model too slow or large → quantization, distillation. Prompt is just bad → fix the prompt first, it is free.

**The decision:**
```
Is the problem stale or private knowledge?
    → RAG (no training cost)

Is the problem behavioral (tone, format, task execution)?
    → Fine-tuning (SFT or PEFT)

Is the problem alignment (helpfulness, safety, sycophancy)?
    → RLHF or DPO

Is the problem latency or cost?
    → Quantization, distillation, pruning

Is the prompt just poorly designed?
    → Fix the prompt first
```

| Approach | Best for | Cost | Freshness |
|:---|:---|:---|:---|
| Prompting | Format, reasoning style, persona | Near zero | N/A |
| RAG | Current/private facts, citations | Medium (infra) | Real-time |
| SFT | Consistent task behavior, domain tone | High (GPU) | Snapshot |
| PEFT (LoRA) | Same as SFT but cheaper | Medium | Snapshot |
| RLHF/DPO | Alignment, preference following | Very high | Snapshot |

---

## Supervised Fine-Tuning (SFT)

**The problem:** a base pretrained model knows a lot, but it does not follow instructions reliably, does not adopt a specific persona, and does not consistently format outputs for a particular task. Cross-entropy pretraining on internet text does not teach instruction-following — the training signal is "predict the next token", not "follow the instruction in the prompt."

**The core insight:** show the model examples of the exact behavior you want — (system prompt, user message, correct assistant response) triplets — and train it to maximize the probability of the correct assistant response. This is identical to pretraining except the training data is high-quality demonstrations rather than raw web text. Data quality dominates data quantity: 1,000 carefully curated examples often outperform 100,000 noisy ones.

**The mechanics:** standard causal language modeling loss, computed only on the assistant's response tokens (not the prompt).

Data format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a customer support agent for Acme Corp."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Go to Settings > Account > Reset Password. You'll receive an email within 2 minutes."}
  ]
}
```

Training recipe: 2–5 epochs, learning rate 1e-5 to 2e-5, cosine schedule with 3% warmup, batch size 32–128, bfloat16 precision.

**What breaks:** SFT can cause catastrophic forgetting — optimizing for task-specific behavior on a narrow dataset degrades performance on tasks outside that distribution. Too many epochs on a small dataset causes overfitting — the model memorizes examples rather than generalizing. SFT changes behavior but not knowledge; the model still hallucinates facts it did not know before fine-tuning.

---

## LoRA (Low-Rank Adaptation)

**The problem:** full fine-tuning updates all model parameters. For a 7B model, that means updating 7 billion parameters, storing full gradients (7B × 4 bytes = 28GB), and storing optimizer states (Adam: 2× more = 56GB). For a 70B model, this requires a multi-GPU cluster just for optimizer state.

**The core insight:** the weight updates learned during fine-tuning lie in a low-dimensional subspace — most of the useful adaptation can be expressed as a low-rank matrix. Instead of updating the full d×d weight matrix directly, learn a small low-rank decomposition of the update and add it to the frozen base weights.

**The mechanics:**
```
W' = W₀ + ΔW = W₀ + B·A

where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d, k)
```

At initialization: A is random Gaussian, B = 0 — so ΔW = 0, preserving the pretrained model. Only B and A are trained; W₀ is frozen.

With r=8 and d=4096: trainable parameters = 2 × 4096 × 8 = 65,536 instead of 4096² = 16.7M — a 256× reduction for that layer. Apply LoRA to the query, key, value, and output projection matrices in each attention block.

The scaling factor α/r controls update magnitude. Typical: r ∈ {4, 8, 16, 64}, α = 2r.

**QLoRA:** load the base model in 4-bit NF4 quantization (bitsandbytes), apply LoRA adapters in full bfloat16. Forward pass in 4-bit, gradients and optimizer states in bfloat16. Enables fine-tuning a 70B model on a single 48GB A100 by quantizing base weights (35GB → ~9GB) while keeping adapter weights in full precision.

**What breaks:** LoRA reduces trainable parameters but not activation memory — the forward pass still runs at the base model's full activation size. For models where activation memory dominates (long sequences, large batches), LoRA's memory savings are smaller than expected. Low rank r may be insufficient for tasks requiring large behavioral changes. The adapter must be merged back into the base weights for efficient inference.

---

## RLHF (Reinforcement Learning from Human Feedback)

**The problem:** cross-entropy training on human demonstrations teaches the model to imitate correct behavior. But it does not teach the model to prefer true statements over plausible-sounding false ones, or helpful responses over sycophantic ones, when both appear in training data. The training objective is "match the distribution", not "optimize for quality."

**The core insight:** explicitly optimize for what humans actually prefer. Collect human rankings of model responses. Train a reward model to predict human preference. Fine-tune the LLM to maximize the reward model's score — with a KL divergence penalty to prevent the model from exploiting the reward model by generating bizarre high-scoring outputs.

**The mechanics — three stages:**

Stage 1 — SFT: fine-tune on demonstration data to produce a capable starting point.

Stage 2 — Reward model: collect preference data (human annotators rank response A vs B). Train a reward model rᵩ(x, y) — an LLM with a scalar regression head — using the Bradley-Terry loss:
```
L_RM = -E[(x, y_w, y_l)] [log σ(rᵩ(x, y_w) - rᵩ(x, y_l))]
```
where y_w is the preferred response and y_l is the rejected one.

Stage 3 — PPO: fine-tune the SFT model to maximize reward, with KL penalty:
```
L_PPO = rᵩ(x, y) - β · KL(π_θ(y|x) || π_SFT(y|x))
```

**What breaks:** reward hacking — the model learns to generate outputs that score highly without actually being better. The reward model can be fooled by confident, fluent responses. PPO requires running three models simultaneously (policy, reference, reward model), consuming 3× the memory. Training is unstable and requires careful hyperparameter tuning. Human annotation at scale is expensive and slow.

---

## DPO (Direct Preference Optimization)

**The problem:** RLHF's three-stage pipeline is operationally complex: it requires a separate reward model, the PPO loop, careful KL penalty tuning, and running three models simultaneously during training. The reward hacking risk is inherent to the explicit reward model formulation.

**The core insight:** the optimal RLHF policy has a closed-form relationship to the reward. Substituting this closed form into the preference objective and eliminating the partition function yields a loss that directly trains the policy on preference pairs — no explicit reward model, no RL loop.

**The mechanics (DPO loss):**
```
L_DPO = -E[(x, y_w, y_l)] [log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

In plain terms: increase the probability of preferred responses relative to the reference model; decrease the probability of rejected ones. The β parameter controls how aggressively to deviate from the reference.

DPO requires: (a) a reference model π_ref (frozen copy of the SFT model), (b) preference pairs (prompt, chosen, rejected), and standard supervised training — no RL loop.

**DPO vs RLHF:**

| Aspect | RLHF (PPO) | DPO |
|:---|:---|:---|
| Reward model | Explicit, trained separately | Implicit |
| Online sampling | Yes (generates during training) | No (offline pairs) |
| Memory | 3 models loaded | 2 models loaded |
| Stability | Tricky (RL variance) | More stable (supervised) |
| Quality ceiling | Potentially higher (online) | Competitive in practice |

**What breaks:** DPO is offline — it learns from fixed preference pairs and cannot adaptively collect new data in regions where the policy is currently poor. Online RLHF (PPO) can self-improve; DPO cannot without new data collection. DPO can "forget" capabilities not represented in the preference pairs. The β hyperparameter requires careful tuning: too high and the model barely deviates from the reference; too low and it drifts to reward-hacking outputs.

---

## Quantization

**The problem:** a 70B parameter model in float32 weighs 280GB. In bfloat16, 140GB. Loading 140GB of weights per inference at ~2TB/s HBM bandwidth sets a hard floor on token generation speed regardless of compute. Smaller precision means smaller model, faster memory transfer, and potentially faster compute — but accuracy must not degrade unacceptably.

**The core insight:** most model weights and activations do not need 16-bit precision for inference. The information required for accurate next-token prediction can often be maintained with 4–8 bits, especially with calibration-aware quantization that carefully maps the weight distribution to the reduced precision range.

**The mechanics — four main approaches:**

W4A16 (weight-only 4-bit): store weights as INT4, dequantize to fp16 before matrix multiply. Memory footprint drops 4×. No compute speedup from INT4 tensor cores because dequantization happens before compute.

W8A8 (weight + activation 8-bit): store and compute in INT8. Requires hardware with INT8 tensor cores (A100+). Provides real compute speedup (~2×) in addition to memory savings.

GPTQ: post-training quantization that minimizes weight error layer by layer using the Hessian of the loss with calibration data. Better quality than naive rounding at INT4, especially for arithmetic reasoning.

AWQ: identifies "salient" weights — those whose magnitudes are amplified by large activations — and protects them from aggressive quantization. Often better than GPTQ at INT4 for tasks requiring factual precision.

SmoothQuant: activations have more extreme outliers than weights, making them harder to quantize. SmoothQuant migrates quantization difficulty from activations to weights via a per-channel smoothing factor, enabling accurate W8A8 quantization.

**What breaks:** quantization degrades non-uniformly across tasks. Arithmetic reasoning, code generation, and factual recall are more sensitive to precision loss than creative generation or summarization. Always evaluate on the target task — perplexity improvements do not guarantee task-specific quality.

---

## Knowledge Distillation

**The problem:** a smaller model trained from scratch on the same data as a large model will be worse. But a smaller model trained to mimic the output distribution of a large model can learn much more from each training example, because the large model's soft probability distribution over the vocabulary encodes its uncertainty and learned relationships between similar concepts.

**The core insight:** a large teacher model, when predicting the next token, assigns non-zero probabilities to many plausible alternatives — not just the single correct token. These soft labels convey information about which tokens are semantically related, which concepts are similar, and where the model is uncertain. Training a student on these soft distributions transfers more knowledge than training on hard labels alone.

**The mechanics:**
```
L_KD = (1-α) · L_CE(y, ŷ_student) + α · L_KL(p_student^T || p_teacher^T)

where p^T = softmax(logits / T), T > 1 (temperature > 1 produces softer distributions)
```

Higher temperature T flattens both distributions, making the KL term more informative about small probability differences between similar tokens.

Sequence-level distillation: train the student on sequences generated by the teacher (not just the teacher's logits). Used to create DistilGPT, DistilBERT.

**What breaks:** the student is bounded by the teacher's quality — distillation cannot exceed the teacher's capabilities. For tasks where the teacher hallucinates, the student learns to hallucinate in the same ways.

---

## Evaluating Fine-Tuned Models

**The problem:** a model that improves on the fine-tuning task may degrade on other tasks. Evaluating only on the target task misses regression. Evaluating only on general benchmarks misses task improvement.

**The core insight:** evaluate on the target task for improvement, on a representative general benchmark for regression, and on a safety eval for alignment. A model that is better at coding but worse at general reasoning or more prone to harmful outputs is not an improvement.

| What changed | Evaluation method | Metric |
|:---|:---|:---|
| Format / style | LLM-as-judge on format compliance | Binary or 1–5 scale |
| Task accuracy | Task-specific benchmark | F1, Exact Match, ROUGE |
| Alignment | Safety eval, refusal rate, red-team | Human evaluation |
| General capability | MMLU, HellaSwag, TruthfulQA | Accuracy |
| Regression | Previous task distribution | Perplexity, task accuracy |

**LLM-as-judge:** for open-ended generation where ground truth is hard to define, use a strong model (GPT-4) to rate responses on correctness, helpfulness, and format. Reliable for relative comparisons; unreliable for absolute calibration.

**What breaks:** LLM judges share biases with the judged model. GPT-4 may prefer responses in GPT-4's style. Self-evaluation produces inflated scores. Always validate LLM judge ratings against human judgments on a sample of cases.

---

## Production Versioning

**The problem:** a fine-tuned model is a software artifact. Without versioning, rollback is impossible when a model degrades in production, and attribution of improvements is impossible across experiments.

**Checklist before deploying:**
- Base model version pinned
- Training data snapshot versioned (hash or dataset version)
- LoRA config / hyperparameters logged (MLflow, Weights & Biases)
- Evaluation metrics vs. baseline documented
- Safety checks passed (red-team, content filter audit)
- Inference latency benchmarked
- Rollback to previous artifact tested

*Related: [Hallucination Mitigation](hallucination-mitigation.md) | [Inference Optimization](inference-optimization.md) | [RAG](rag.md)*
