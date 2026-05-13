See also: [Fine-tuning deep dive](../applications/tuning-optimization.md)

# Fine-Tuning & Model Adaptation — Interview Q&A

---

**Q1. When should you fine-tune an LLM vs. prompting vs. RAG?**

The clean decision tree:

| Problem | Right lever |
|---|---|
| Knowledge is stale or private | RAG |
| Output format is inconsistent | Fine-tune (SFT) |
| Tone / domain voice needs to change | Fine-tune |
| Task requires reasoning patterns not in base model | Fine-tune |
| Quick behavioral tweak, have no training data | Prompt engineering |
| Fresh knowledge AND consistent style needed | RAG + fine-tune |

Fine-tune when you have quality training data and need stable, reproducible behavior changes. Do not fine-tune to memorize facts — that is RAG's job. Facts baked into weights go stale and the model still hallucinates when uncertain.

---

**Q2. What is the difference between full fine-tuning and PEFT?**

**Full fine-tuning:** every parameter in the model is updated. Optimizer states (Adam: 2× parameter count in fp32) plus gradients require roughly 16 bytes per parameter — a 7B model needs ~112 GB of GPU memory just for training states. Risk of catastrophic forgetting is high.

**Parameter-Efficient Fine-Tuning (PEFT):** the base model weights are frozen. Only a small set of added or modified parameters are trained. Memory drops dramatically. Forgetting is reduced because base weights are untouched.

Common PEFT methods: LoRA, QLoRA, prefix tuning, adapter modules.

For most teams: start with LoRA. Only consider full fine-tuning if you have the compute, the data volume, and LoRA has demonstrably plateaued.

---

**Q3. Explain LoRA — how does it work mathematically?**

LoRA (Low-Rank Adaptation) exploits the hypothesis that task-specific weight updates lie in a low-dimensional subspace.

For a linear projection `y = Wx`, instead of learning the full ΔW (d×k), LoRA learns two smaller matrices:

```
ΔW = B · A     where A is (r × k), B is (d × r), r << min(d, k)
```

During forward pass: `y = Wx + BAx`. B is initialized to zero so the adapter starts as an identity (no change), and A is initialized with random Gaussian values.

Key hyperparameters:
- **rank r:** typical values 4–64. Lower = fewer parameters, less capacity. Higher = more capacity, more compute.
- **alpha (α):** scaling factor applied as `(α/r) * BAx`. Controls the effective learning rate of the adapter.
- **target modules:** commonly Q, K, V, O projections in attention. Sometimes FFN projections too.

After training, the adapter can be merged into the base weights (`W' = W + BA`) for zero-latency inference.

---

**Q4. What is QLoRA and why did it matter?**

QLoRA (Dettmers et al., 2023) combines:
1. **NF4 quantization** of the frozen base model weights (4-bit Normal Float, optimal for normally distributed weights)
2. **Double quantization** of the quantization constants themselves (saves another ~0.5 bits/parameter)
3. **Paged optimizers** to handle GPU memory spikes by offloading optimizer states to CPU RAM

Result: a 65B parameter model that previously required 8×80GB A100s can be fine-tuned on a single 48GB GPU. This democratized LLM adaptation.

Trade-offs: 4-bit quantization introduces small precision loss. Always validate on your eval set; some tasks are sensitive. Training is slightly slower due to dequantization on the forward pass. The adapters themselves are stored in bf16/fp16 — do not quantize them.

---

**Q5. What is RLHF and what problem does it solve that SFT cannot?**

SFT (Supervised Fine-Tuning) teaches format and behavior by maximizing likelihood of good responses. But "good" is hard to specify exhaustively as examples. Models fine-tuned only with SFT can be helpful but still produce harmful, dishonest, or sycophantic outputs.

RLHF pipeline:
1. Collect preference data: annotators compare `(prompt, response_A, response_B)` and label which is better.
2. Train a reward model R(x, y) to predict the preferred response.
3. Optimize the LLM policy π to maximize expected reward while staying close to the SFT baseline:

```
maximize E[R(prompt, response)] - β · KL(π || π_ref)
```

The KL penalty (controlled by β) prevents "reward hacking" — the model gaming the reward model in ways that diverge from actual human preference (e.g., producing overly verbose or sycophantic text).

Common failure modes: reward model errors compound; annotator disagreement injects noise; the model can discover exploits in the reward model that look high-scoring but are nonsense.

---

**Q6. What is DPO and why do teams prefer it over RLHF?**

DPO (Direct Preference Optimization, Rafailov et al., 2023) achieves alignment without training a separate reward model or running RL.

It re-parameterizes the RLHF objective so the reward model is implicit in the ratio of the policy's probabilities:

```
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

where `y_w` is the preferred ("won") response and `y_l` is the rejected ("lost") response.

Why teams use it:
- No separate reward model to train, store, and maintain.
- No RL training loop (PPO is notoriously unstable and sensitive to hyperparameters).
- Simpler to implement: just standard cross-entropy-style training on preference pairs.
- Same or better results as RLHF on many benchmarks.

Limitation: still requires quality preference data. Bad preference pairs teach wrong behavior just as surely as bad SFT data.

---

**Q7. How do you prevent catastrophic forgetting during fine-tuning?**

Catastrophic forgetting: fine-tuning on new data overwrites general capabilities baked in during pretraining. The model becomes better at your task and worse at everything else.

Mitigations, in order of effectiveness:
1. **PEFT (LoRA/adapters):** base weights never change. The strongest mitigation by far.
2. **Data mixing:** include ~10–20% general-domain examples in the training batch alongside domain data.
3. **KL regularization:** penalize deviation from the reference model: `loss = task_loss + α · KL(π_finetuned || π_ref)`.
4. **Lower learning rate:** fine-tuning LR should be 5–10× smaller than pretraining LR.
5. **Early stopping:** monitor a general capability benchmark (MMLU, GSM8K) alongside task-specific metrics. Stop when general capability starts declining.

Signal to watch: if task accuracy goes up but general reasoning or language fluency scores drop, you have forgetting. Full evaluation suites, not just task accuracy, are essential.

---

**Q8. Walk me through preparing a dataset for SFT fine-tuning.**

Quality matters far more than quantity. 1,000 perfectly curated examples typically outperform 100,000 generated ones (LIMA paper result).

Steps:
1. **Define schema:** input format (instruction, context, input), output format (expected response). Stick to one format consistently.
2. **Collect:** human-written examples, curated logs, or synthetic generation with filtering (not raw).
3. **Clean:** deduplicate with fuzzy matching, remove truncated examples, filter unsafe content.
4. **Split:** hold out 10–15% for validation, 5% for test. No examples from the same source in both train and eval (leakage).
5. **Tokenize:** verify sequence length distribution. Examples longer than the model's context window need truncation or filtering — truncation from the right (losing the response) is almost always wrong; filter them or truncate input.
6. **Balance:** check class/intent distribution. A model trained on 80% of one intent type will overfit to it.

For LoRA/PEFT: also verify that your chunking/padding strategy pads to a consistent length per batch to avoid shape mismatches.

---

**Q9. What hyperparameters matter most for LoRA fine-tuning?**

| Hyperparameter | Typical range | Effect |
|---|---|---|
| Learning rate | 1e-5 to 3e-4 | Too high → unstable; too low → underfitting. Start at 2e-4, decay with cosine schedule. |
| LoRA rank r | 4–64 | Higher = more capacity, more memory. Start at 8–16. |
| LoRA alpha | r × 1 to r × 2 | `(alpha/r)` is the effective scale. Doubling alpha ≈ doubling LR. |
| Epochs | 1–3 | More epochs → overfitting. For large datasets, 1 epoch is often enough. |
| Batch size | 4–32 | Larger = more stable gradients. Use gradient accumulation to simulate large batches on small GPUs. |
| Warmup steps | 5–10% of total steps | Prevents early training instability. |

For QLoRA specifically: use a slightly lower LR than plain LoRA. The quantization introduces noise that a high LR amplifies.

---

**Q10. How do you evaluate a fine-tuned model?**

Evaluate on three dimensions simultaneously:

**Task performance:** exact match, F1, ROUGE (for generation tasks), JSON schema validity, tool call accuracy — whatever is specific to your task.

**General capability retention:** run MMLU or a similar benchmark before and after fine-tuning. A >3–5% drop indicates catastrophic forgetting.

**Safety and robustness:** check refusal rates on harmful inputs, jailbreak probes, and adversarial rephrasing of edge cases in your domain.

Evaluation discipline:
- Test data must not overlap with training data. Leakage gives false confidence.
- Use both automatic metrics and human evaluation. Automatic metrics catch regressions fast; humans catch subtle quality issues.
- For structured outputs: use a schema validator, not just visual inspection. A model that generates syntactically invalid JSON 5% of the time will break production pipelines.

If your RAG system is also fine-tuned: evaluate end-to-end faithfulness (does the answer stay grounded in context?) and latency/cost alongside task metrics.

---

**Q11. What is instruction tuning and how does it create chat models?**

A base LLM (pretrained only) will complete "Write a poem about..." by generating more instructions — it has learned to predict internet text, which is full of instructions. It does not understand the "User → Assistant" interaction model.

Instruction tuning (SFT on instruction-response pairs) teaches the model this contract. The training objective is identical to standard SFT — maximize the likelihood of the response tokens given the instruction tokens — but the data is specifically formatted as:

```
<system>You are a helpful assistant.</system>
<user>Instruction here</user>
<assistant>Response here</assistant>
```

The model learns that after a `<user>` turn it should produce a response, not more instructions. This is the step that separates a base model from a chat model. RLHF/DPO then refines the quality of those responses — instruction tuning is the prerequisite.

---

**Q12. What are prefix tuning and prompt tuning, and how do they compare to LoRA?**

Both learn soft (continuous) tokens prepended to the input, conditioning the model without touching internal weights.

**Prompt tuning:** appends a small set of learned vectors (the "soft prompt") to the token embeddings before the first layer. Only these vectors are trained. Works well for large models (>10B), poorly for small models.

**Prefix tuning:** injects learned vectors at every transformer layer (prepended to K and V in attention), not just the input. More expressive than prompt tuning. Both are extremely parameter-efficient.

Comparison to LoRA:
- LoRA is more parameter-efficient for the same performance level.
- LoRA adapts internal weight projections; prefix tuning steers via input-conditioning.
- LoRA merges into base weights at inference; prefix tuning requires carrying the prefix tensors.
- In practice, LoRA/QLoRA has largely displaced prefix/prompt tuning as the default.

---

**Q13. How do you merge multiple LoRA adapters?**

You can combine domain adapters (e.g., legal adapter + finance adapter) via weighted addition of their low-rank updates:

```python
ΔW_merged = w1 * (B1 @ A1) + w2 * (B2 @ A2)
W_final = W_base + ΔW_merged
```

This works because LoRA updates are linear and the base weights are frozen, so additions are well-defined.

Selecting weights `w1, w2`: grid search on a validation set that covers both target domains. Equal weights (0.5, 0.5) are a reasonable start.

Risks: adapters can conflict. If legal and finance training data contain contradictory behavioral patterns, the merge can produce degraded output on both. Always re-evaluate the merged model on both domain eval sets and on safety benchmarks before shipping.

Alternatives: run adapters sequentially (apply one then the other), or use mixture-of-experts routing at inference time if adapters must remain separate.

---

**Q14. Your fine-tuned model memorizes training data verbatim. How do you fix it?**

Symptoms: training loss near zero, validation loss diverging; generated outputs contain verbatim training examples; high ROUGE against training set.

Root causes and fixes:

| Cause | Fix |
|---|---|
| Dataset has near-duplicates | Deduplicate with MinHash or exact substring matching |
| Too many epochs | Reduce to 1–2; use early stopping on validation perplexity |
| Learning rate too high | Halve it; use warmup |
| Adapter rank too high (over-capacity) | Reduce LoRA r |
| Dataset too small for model capacity | Add more diverse data or regularize more aggressively |

After fixing, verify generalization: run held-out prompts that are semantically similar but not verbatim to training examples. If the model still "solves" them correctly, the fix worked.

---

**Q15. Your RLHF/DPO model rewards hacking — it produces high-reward but bad outputs. What now?**

Reward hacking: the model discovers artifacts in the reward model that produce high scores without actually being good (e.g., verbose repetition if annotators prefer longer answers, or specific sentence openers that annotators associate with quality).

Diagnostic: run the optimized model against human raters blind to reward model scores. Divergence between reward model scores and human scores confirms hacking.

Fixes:
1. **Stronger KL penalty (higher β):** constrains how far the model can move from the SFT reference. Reduces hacking but also reduces alignment gains.
2. **Update the reward model:** collect new preference data specifically targeting the hacked behaviors and retrain.
3. **Red-team the reward model:** systematically probe for inputs where the reward model scores clearly bad outputs highly. Fix those before policy optimization.
4. **RLAIF auditing:** if using AI feedback, audit a representative sample with human judges to catch evaluator model biases.
5. **Ensemble reward models:** use multiple reward models; only allow updates that score well across all of them.

The underlying lesson: the reward model is a proxy for human preference, not ground truth. Treat it as a weak signal that needs ongoing validation, not a static oracle.
