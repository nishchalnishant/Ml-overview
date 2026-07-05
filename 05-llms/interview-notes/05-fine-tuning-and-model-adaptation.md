---
module: Llms
topic: Interview Notes
subtopic: Fine Tuning And Model Adaptation
status: unread
tags: [llms, ml, interview-notes-fine-tuning-an]
---

> _Full study notes. For the quick-recall version, see [fine-tuning-and-model-adaptation-snappy.md](fine-tuning-and-model-adaptation-snappy.md)._

See also: [Fine-tuning deep dive](../applications/10-tuning-optimization.md)

# Fine-Tuning and Model Adaptation — Interview Notes

---

## When to Fine-Tune vs. Prompt vs. RAG

**The problem**: you have a task and a pretrained LLM. You could write a better prompt, retrieve relevant context at runtime, or update the model weights. These are not interchangeable — using the wrong lever wastes resources and produces unstable results.

**The core insight**: the three levers change three different things. Prompting changes what the model sees at inference. RAG changes what facts are available at inference. Fine-tuning changes the model's behavior permanently. Match the lever to what is actually wrong.

**The mechanics**:

| What is broken | Right lever |
|---|---|
| The model doesn't know the current data | RAG — update the knowledge store, not the model |
| The model produces the wrong output format | SFT — format consistency is behavioral, not factual |
| The model uses the wrong tone, domain vocabulary, or style | SFT |
| The model lacks a reasoning pattern needed for this task | SFT |
| You have no training data, need a quick test | Prompt engineering |
| Knowledge changes AND style needs to change | RAG + fine-tune |

**What breaks**: fine-tuning to inject facts into weights fails because: (1) facts go stale but weights don't update automatically; (2) the model can still hallucinate the fact when uncertain even after fine-tuning. Fine-tuning is for behavior change, not knowledge injection.

**What the interviewer is testing**: whether you know that these approaches are complementary — not a ranked list where fine-tuning is always "better."

**Common traps**: proposing fine-tuning for anything that seems "not working" — often a retrieval or prompt issue; saying "RAG is cheaper than fine-tuning" without acknowledging that RAG requires building and maintaining a separate retrieval system.

---

## Full Fine-Tuning vs. PEFT

**The problem**: updating all parameters of a 7B-parameter model with Adam requires storing: the parameters (2 bytes/param in fp16), gradients (2 bytes/param), and two Adam optimizer states (8 bytes/param). Total: ~12 bytes/parameter. A 7B model requires ~84 GB just for training states — more than most GPU clusters have in a single machine.

**The core insight**: for most adaptation tasks, you are not teaching the model new language or fundamentally new reasoning — you are redirecting existing capabilities. The required change to the model is small. Only the "delta" needs to be learned, not the full parameter set.

**The mechanics**:

**Full fine-tuning**: every parameter is updated every step. Optimizer states (Adam: m, v vectors) consume 2× the parameter count in fp32. For a 7B model: ~112 GB training memory. Catastrophic forgetting risk is high because base weights are directly modified.

**Parameter-Efficient Fine-Tuning (PEFT)**: freeze base weights. Only train a small set of additional or modified parameters. Common variants:
- **LoRA** (see below): adds low-rank matrices to weight projections
- **QLoRA**: LoRA on a quantized base model
- **Prefix tuning / prompt tuning**: learn soft tokens that condition the model

Memory drops from ~112 GB to ~6–16 GB for a 7B model. Forgetting risk drops because base weights are untouched. Start with LoRA; consider full fine-tuning only when LoRA has demonstrably plateaued on your task and you have both the data and compute.

**What breaks**: PEFT does not help when the task requires knowledge that is genuinely absent from the base model — the frozen weights don't know what they were never trained on. PEFT with too low capacity (very low rank) may miss complex task structure.

**What the interviewer is testing**: that you understand the memory breakdown of full fine-tuning, not just that PEFT "uses less memory."

**Common traps**: saying PEFT prevents forgetting "completely" (it reduces it, but the trained adapter can still push the model away from general-purpose behavior); confusing number of trainable parameters with memory savings (the base model still sits in memory even with PEFT).

---

## LoRA

**The problem**: full fine-tuning a 70B parameter model requires as much GPU memory as pretraining it. For most adaptation tasks, you don't need to change all 70B parameters — the task adaptation is a relatively small "delta" on top of the pretrained knowledge.

**The core insight**: the change to the weight matrix during fine-tuning has low intrinsic rank — it lives in a much smaller subspace than the full d×d weight matrix. Parameterize just that low-rank subspace.

**The mechanics**: freeze W, add ΔW = BA where B is d×r and A is r×d, with r ≪ d. Only A and B are trained. Forward pass: y = Wx + (α/r)·BAx. B is initialized to zero (so the adapter starts as no-op); A is initialized with random Gaussian values.

At inference, merge: W' = W + (α/r)·BA. No added latency — the adapter disappears into the base weight matrix. r=8 typically captures 80–90% of the fine-tuning benefit with ~0.1% of the parameters.

LoRA is applied to specific weight projections — typically Q, K, V, O in attention, sometimes FFN projections. Applying to all projections vs. just Q and V is a hyperparameter choice; wider application with lower rank often beats narrow application with higher rank.

**What breaks**: LoRA doesn't help with tasks requiring genuine new knowledge (the frozen W doesn't know it). Very low rank (r=1, 2) may miss complex task structure. The rank choice has no principled selection rule — it is tuned on validation performance. LoRA on the wrong target modules can underperform relative to full fine-tuning on tasks requiring deep representation changes.

**What the interviewer is testing**: do you understand why parameter-efficient fine-tuning works at all, or just that it saves memory?

**Common traps**: confusing LoRA rank with number of parameters saved; not knowing that LoRA matrices are merged at inference time (no latency penalty); saying LoRA trains "on top of" the model as a separate network at inference — it doesn't, after merging.

---

## QLoRA

**The problem**: LoRA reduces the number of trainable parameters, but the frozen base model still sits in GPU memory at full precision (fp16: 2 bytes/param). A 65B model at fp16 requires ~130 GB — still inaccessible to a single GPU.

**The core insight**: if the base model weights are frozen during fine-tuning, they never need to be in full precision. They only need to be accurate enough for the forward pass. Quantize them aggressively, dequantize just-in-time during the forward pass, and keep only the LoRA adapters in full precision.

**The mechanics** (Dettmers et al., 2023):

1. **NF4 quantization** of the frozen base model weights: 4-bit Normal Float, a data type specifically designed for normally distributed neural network weights. Stores each weight in 4 bits (2× more compressed than fp16 = 8× more compressed than fp32).
2. **Double quantization**: quantize the quantization constants themselves (the scale factors), saving another ~0.5 bits/parameter on average.
3. **Paged optimizers**: use NVIDIA's unified memory to page optimizer states to CPU RAM during memory spikes, preventing OOM errors on long sequences.

Result: a 65B model that previously required 8×80GB A100s can be fine-tuned on a single 48 GB GPU. The adapters themselves remain in bf16/fp16 — do not quantize them.

**What breaks**: 4-bit quantization introduces small precision loss in the base model forward pass — not training imprecision per se, but accumulated rounding errors. Always evaluate on your task; some domains (precision math, structured output) are more sensitive. Training is slightly slower than plain LoRA due to the dequantization step.

**What the interviewer is testing**: that you understand the distinction between the frozen base model memory problem (QLoRA solves this) and the trainable parameter memory problem (LoRA already addresses this).

**Common traps**: thinking QLoRA quantizes the LoRA adapters (it doesn't — adapters stay in fp16/bf16); saying QLoRA is "the same as LoRA but quantized" without understanding the paged optimizer contribution; not knowing what NF4 is optimized for (normally distributed weights, which is empirically what neural networks have).

---

## Instruction Tuning and the Base-to-Chat Transition

**The problem**: a base model trained on next-token prediction will, when given "What is the capital of France?", produce more quiz questions — because the training data contains many lists of questions. It has no concept that this string is a question directed at it, expecting an answer. The model needs to learn the user→assistant interaction contract.

**The core insight**: the interaction format is a behavioral pattern, not a capability. The knowledge needed to answer the question is already in the base model. What's missing is the learned association between the instruction format and the response format. SFT on a small number of high-quality instruction-response pairs teaches this association.

**The mechanics**: apply standard cross-entropy fine-tuning on data formatted as:

```
<system>You are a helpful assistant.</system>
<user>Instruction here</user>
<assistant>Response here</assistant>
```

Critically: compute loss only on the response tokens, not the instruction tokens. This prevents the model from memorizing instructions as completions. A few thousand carefully curated pairs is often enough (LIMA: 1,000 examples matched the quality of 52,000 automatically filtered examples).

**What breaks**: SFT teaches the format and basic helpfulness but does not reliably produce safe, non-sycophantic, or consistently honest behavior — because bad behavior also appears in training data, and the model can learn any pattern with sufficient frequency. This is why RLHF/DPO is needed on top of SFT, not instead of it.

**What the interviewer is testing**: that you understand SFT as the prerequisite step that makes RLHF possible — and that "instruction tuning" and "alignment" are separate things.

**Common traps**: confusing instruction tuning with RLHF (SFT teaches format; RLHF teaches preference); saying "you need millions of instruction pairs" (LIMA showed quality dominates quantity); not knowing that loss should not be applied to the instruction tokens.

---

## RLHF

**The problem**: SFT can produce a model that follows instructions in a reasonable format. But "good" behavior is multidimensional and hard to fully specify via examples: helpfulness, honesty, avoiding harm, not being sycophantic. You cannot enumerate exhaustive examples of every failure mode. You need a way to optimize the model against a richer signal of human preference.

**The core insight**: humans can easily compare two responses and say which is better, even when they cannot write a perfect response themselves. Use that comparative judgment to train a proxy reward model, then optimize the LLM policy against that reward. The KL divergence penalty prevents the policy from exploiting reward model weaknesses.

**The mechanics**:

1. Collect preference data: annotators compare (prompt, response_A, response_B) and label which is preferred.
2. Train a reward model R(x, y) on these pairs using the Bradley-Terry model: R should score the preferred response higher.
3. Optimize the policy using PPO:
   ```
   maximize E[R(prompt, response)] - β · KL(π_θ || π_ref)
   ```
   The KL penalty (controlled by β) keeps the optimized policy close to the SFT reference. Higher β = less reward hacking, less alignment gain. Lower β = more alignment gain, more risk of reward exploitation.

**What breaks**: reward hacking — the policy finds patterns in the reward model that produce high scores but are not actually preferred (e.g., verbose responses if annotators equate length with quality; specific sentence openers that annotators associate with quality). PPO is notoriously hyperparameter-sensitive and unstable. Annotator disagreement introduces noise into preference labels.

**What the interviewer is testing**: whether you understand the KL penalty's role and can explain reward hacking concretely.

**Common traps**: saying RLHF "makes the model safe" (it reduces certain failure modes; it does not guarantee safety); not knowing that the reward model itself can be wrong and is a proxy, not ground truth; confusing RLHF with DPO (RLHF requires a separate reward model and RL training loop; DPO does not).

---

## DPO

**The problem**: RLHF requires training and maintaining a separate reward model, then running a complex, unstable RL training loop (PPO). Three separate training phases, three separate model checkpoints, sensitive hyperparameters. For many teams, this is impractical.

**The core insight**: the optimal policy under the RLHF objective can be expressed analytically as a function of the reward and the reference policy. This means the reward model is implicit — you can write a training objective directly in terms of the policy's probabilities, eliminating the need for a separate reward model.

**The mechanics** (Rafailov et al., 2023):

```
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

where y_w is the preferred ("won") response and y_l is the rejected ("lost") response.

Intuitively: increase the probability of the preferred response relative to the reference model, decrease the probability of the rejected response relative to the reference model. The β parameter controls how much the policy is allowed to deviate from the reference.

**What breaks**: DPO is still sensitive to the quality of the preference data — bad preference pairs teach wrong behavior just as surely as bad SFT data. DPO can suffer from "chosen token probability collapse" where the model learns to maximize the chosen/rejected log-ratio by reducing probability of rejected, without genuinely improving the preferred response. Some evidence suggests RLHF outperforms DPO on highly complex alignment tasks, though DPO is competitive on most.

**What the interviewer is testing**: whether you can explain why DPO is mathematically equivalent to RLHF, not just that it's "simpler."

**Common traps**: saying DPO "doesn't use RL" (it's derived from the RL objective — the RL is implicit, not absent); not knowing that DPO still requires a reference model (the SFT checkpoint); confusing DPO with simple SFT on chosen responses only (DPO is a contrastive objective using both chosen and rejected).

---

## Catastrophic Forgetting

**The problem**: fine-tuning a model on a narrow task improves task performance but can overwrite general capabilities learned during pretraining. The model becomes better at your task and worse at everything else — a phenomenon called catastrophic forgetting. You may not notice this until a user asks something outside your fine-tuning distribution.

**The core insight**: fine-tuning is gradient descent on your task data. There is nothing in the loss that says "preserve general capabilities" — only "predict these responses correctly." If your task data is narrow, the model optimizes for it at the expense of the broader weight configuration learned during pretraining.

**The mechanics — mitigations in order of effectiveness**:

1. **PEFT (LoRA/adapters)**: base weights never change. The strongest mitigation. You cannot forget what you never modified.
2. **Data mixing**: include ~10–20% general-domain examples (e.g., from the original pretraining distribution) alongside task-specific data.
3. **KL regularization**: add a loss term penalizing deviation from the reference model: `total_loss = task_loss + α · KL(π_finetuned || π_ref)`.
4. **Lower learning rate**: fine-tuning LR should be 5–10× lower than pretraining LR.
5. **Early stopping**: monitor general benchmarks (MMLU, GSM8K) alongside task metrics. Stop when general capability begins declining.

**What breaks**: data mixing requires having access to general-domain data, which can conflict with privacy constraints. KL regularization adds a hyperparameter (α) with no principled selection rule. Early stopping on general benchmarks is noisy for small datasets.

**What the interviewer is testing**: whether you know that catastrophic forgetting is a training dynamics problem, not a prompt problem — and that the mitigations are prevention strategies, not post-hoc fixes.

**Common traps**: confusing forgetting with the model "losing knowledge" (the weights are modified, not the knowledge erased — the model was never explicitly told to preserve it); thinking fine-tuning on a small dataset is safe (small datasets with high learning rates on many epochs cause more forgetting, not less); not evaluating on general benchmarks after fine-tuning.

---

## Dataset Preparation for SFT

**The problem**: the quality of a fine-tuned model is bounded by the quality of its training data. Garbage in, garbage out — but the problem is subtler than it sounds. Even a small percentage of low-quality, inconsistent, or leaked examples can systematically bias a model's behavior in ways that are hard to debug without careful eval.

**The core insight**: the model learns whatever pattern is most consistent in your training data. Inconsistency, near-duplicates, and format drift teach the model inconsistency and format drift. Quality of each example matters more than total dataset size.

**The mechanics**:

1. **Define schema first**: choose a single consistent format (instruction, context, response). Every example must conform. Mixed formats teach the model to be inconsistent.
2. **Collect and filter**: human-written > model-written > raw web text. If using synthetic generation, filter aggressively — do not use raw model output as training data.
3. **Deduplicate**: use MinHash or exact substring matching. Near-duplicates teach the model to memorize, not generalize; they also inflate apparent dataset size.
4. **Split without leakage**: hold out 10–15% validation, 5% test. Ensure no examples from the same source appear in both train and eval (source-level, not example-level deduplication).
5. **Check length distribution**: examples longer than the model's context window require truncation. Truncate from the right only if the response is incomplete — truncating the response is almost always wrong. Filter or truncate the input instead.
6. **Check class/intent balance**: a dataset that is 80% one intent type produces a model that handles that intent well and everything else poorly.

**What breaks**: the LIMA result (1,000 curated examples outperforming 100,000 noisy ones) is widely cited, but requires genuinely curated data. Assuming synthetic generation produces "curated" data without human review is the most common failure.

**What the interviewer is testing**: whether you treat data preparation as an engineering discipline with measurable quality criteria, not just "collect examples."

**Common traps**: using raw model output as training data without filtering (this trains the model on its own errors); deduplicating at the example level but missing source-level overlap between train and eval; forgetting to check that response tokens, not instruction tokens, dominate the loss.

---

## Hyperparameters for LoRA Fine-Tuning

**The problem**: LoRA introduces hyperparameters on top of standard training hyperparameters, and their interaction is not intuitive. Choosing the wrong rank or alpha can produce a model that underfit the task, overfits specific examples, or behaves identically to the unfine-tuned base model.

**The core insight**: rank determines the capacity of the adaptation, alpha/rank determines the effective learning rate of the adapter, and target modules determine which parts of the network are allowed to adapt. Each of these shapes what the fine-tuning can and cannot change.

**The mechanics**:

| Hyperparameter | Range | Effect |
|---|---|---|
| Learning rate | 1e-5 to 3e-4 | Too high → unstable. Too low → underfitting. Start at 2e-4 with cosine decay. |
| LoRA rank r | 4–64 | Higher = more capacity, more memory. Start at 8–16. |
| LoRA alpha | r × 1 to r × 2 | (alpha/r) scales the adapter contribution. Doubling alpha ≈ doubling effective LR. |
| Epochs | 1–3 | More epochs → overfitting. For large datasets, 1 epoch is often sufficient. |
| Batch size | 4–32 | Larger = more stable gradients. Use gradient accumulation for small GPUs. |
| Target modules | q,k,v,o or all | Wider application at lower rank often beats narrow at higher rank. |
| Warmup steps | 5–10% | Prevents early instability. |

For QLoRA specifically: use a lower learning rate than plain LoRA. The quantization step introduces noise that high LR amplifies.

**What breaks**: rank is a capacity hyperparameter without a principled selection rule — it must be tuned on validation performance. Alpha is often set to 2×rank as a default but this is a heuristic. Training beyond 2–3 epochs on small datasets reliably causes memorization.

**What the interviewer is testing**: whether you understand what each hyperparameter controls, not just their typical values.

**Common traps**: setting alpha = rank always (valid default but not universally optimal); treating more epochs as always better; not knowing that target module selection affects what kinds of behavioral changes are possible.

---

## Evaluating a Fine-Tuned Model

**The problem**: task accuracy on the training-adjacent eval set is necessary but not sufficient. A model can achieve high task accuracy while simultaneously forgetting general reasoning, producing unsafe outputs, or becoming brittle on rephrasing — none of which appear in a task-only eval.

**The core insight**: you need to evaluate on three independent dimensions simultaneously: task performance, general capability retention, and behavioral safety. Any one of these can fail while the others look fine.

**The mechanics**:

**Task performance**: metrics specific to your task — exact match, F1, ROUGE-L (for generation tasks), JSON schema validity, tool call accuracy. Use schema validators for structured outputs, not visual inspection (a model that produces invalid JSON 5% of the time will break production pipelines 5% of the time).

**General capability retention**: run MMLU or a similar benchmark before and after fine-tuning. A >3–5% drop indicates catastrophic forgetting. If you only care about task performance, you will ship a broken model.

**Safety and robustness**: test refusal rates on harmful inputs, adversarial rephrasing of edge cases, and out-of-distribution prompts. For RAG-augmented fine-tuned models: evaluate faithfulness — does the model stay grounded in retrieved context, or does it override context with fine-tuned knowledge?

**What breaks**: automatic metrics catch regressions fast but miss subtle quality issues. Human evaluation catches quality but is slow and expensive. Eval sets that overlap with training data produce false confidence. The solution is both: automated metrics for continuous regression detection, human eval for quality baselines.

**What the interviewer is testing**: whether you know that evaluation is multi-dimensional and that task accuracy alone is insufficient.

**Common traps**: not checking general capability benchmarks before shipping; running human eval without also running automated metrics (no way to detect regressions continuously); assuming visual inspection of structured outputs is sufficient.

---

## Reward Hacking in RLHF/DPO

**The problem**: reward hacking occurs when the policy learns to exploit patterns in the reward model that produce high scores without producing genuinely preferred outputs. The reward model is a proxy for human preference — any proxy can be gamed. As policy optimization proceeds, reward scores improve while human-rated quality plateaus or declines.

**The core insight**: the reward model was trained on a finite dataset of human preferences. It has blind spots — patterns it was not trained to evaluate or patterns that happen to correlate with preferred labels in training but are not inherently good (e.g., certain sentence structures, response length, specific phrases). A strong optimizer (the policy) will find and exploit these blind spots.

**The mechanics — diagnosis and fixes**:

**Diagnosis**: run the optimized policy against human raters, blinded to reward model scores. Divergence between reward model scores and human preference scores confirms hacking.

**Fixes**:
1. **Stronger KL penalty (higher β)**: constrains policy deviation. Reduces hacking but also reduces alignment gain. Tuning β is a fundamental tradeoff.
2. **Retrain the reward model**: collect new preference data specifically targeting hacked behaviors. The reward model needs to see the failure modes to learn to penalize them.
3. **Red-team the reward model first**: systematically probe for inputs where it scores clearly bad outputs highly. Fix before policy optimization.
4. **Ensemble reward models**: use multiple reward models trained on different data; only allow policy updates that score well across all of them.
5. **Iterative RLHF**: repeatedly retrain the reward model on outputs from the current policy, closing the distribution gap.

**What breaks**: ensemble reward models add significant infrastructure overhead. Red-teaming reward models is labor-intensive. There is no termination condition for iterative RLHF that is principled — you stop when human evals look good, which is expensive.

**What the interviewer is testing**: whether you understand that the reward model is a learned approximation that can be wrong, and that treating it as an oracle is a known failure mode.

**Common traps**: thinking a high reward model score means the model is aligned (reward hacking makes scores go up while quality goes down); not knowing that the KL penalty is the primary lever for trading off alignment gain versus reward hacking; confusing the reward model (a trained model) with a hand-coded score function.

---

## Merging LoRA Adapters

**The problem**: a team fine-tunes separate LoRA adapters for different domains (legal, medical, financial). They want a single model that handles all domains. Running separate models per domain is operationally expensive. Is there a way to combine multiple adapters without retraining?

**The core insight**: LoRA updates are linear additions to the base weight matrix. Because the base weights are frozen across all adapters, the adapter deltas are additive in the same space and can be combined algebraically.

**The mechanics**:

```python
ΔW_merged = w1 * (B1 @ A1) + w2 * (B2 @ A2)
W_final = W_base + ΔW_merged
```

Select weights w1, w2 via grid search on a validation set covering both target domains. Equal weights (0.5, 0.5) are a reasonable starting point. After merging, evaluate on all domain eval sets and safety benchmarks before shipping.

**What breaks**: adapters can conflict. If domain A and domain B training data contain contradictory behavioral patterns, the merge degrades performance on both. Always evaluate the merged model — do not assume linear combination preserves each adapter's behavior.

Alternatives when merging fails: run adapters sequentially (apply one then the other), use mixture-of-experts routing at inference time to keep adapters separate, or use model merging techniques like TIES or DARE which handle conflicting parameters more carefully.

**What the interviewer is testing**: whether you know that LoRA merging is possible and understand why — because LoRA updates are linear transformations of frozen base weights, not modifications to the model's computational graph.

**Common traps**: thinking you need to retrain when combining domains (merging is often sufficient); not knowing that conflicting adapters can hurt both domains; confusing adapter merging with model ensembling (merging produces a single model; ensembling runs multiple models).

---

## Prefix Tuning and Prompt Tuning vs. LoRA

**The problem**: the most parameter-efficient PEFT approaches would not modify internal weights at all — they would steer the model via the input. Prefix tuning and prompt tuning take this approach: learn soft (continuous) tokens that condition the model without touching any weight matrix.

**The core insight**: if a model can be conditioned on input context, then a learned "virtual prompt" prepended to every input is a form of task specification. The model's weights are unchanged; only the input distribution is shifted.

**The mechanics**:

- **Prompt tuning**: add a small set of learned embedding vectors (the "soft prompt") to the input token embeddings before the first transformer layer. Only these vectors are trained. Effective for very large models (>10B) where the model has enough capacity to interpret arbitrary conditioning; works poorly for smaller models that lack this flexibility.
- **Prefix tuning**: inject learned key-value pairs at every transformer layer's attention computation (prepended to K and V matrices). More expressive than prompt tuning — the conditioning is applied throughout the network, not just at the input. More parameters, better performance.

**What breaks relative to LoRA**: prefix tuning requires carrying the prefix tensors at inference (additional memory and computation). LoRA adapters are merged into base weights at inference, incurring zero overhead. In practice, LoRA and QLoRA have largely displaced prefix/prompt tuning as defaults. Prefix tuning degrades for small models and short sequence tasks where the prefix consumes meaningful context.

**What the interviewer is testing**: whether you know the family of PEFT approaches beyond LoRA and can explain why LoRA has become dominant.

**Common traps**: saying prompt tuning and prompt engineering are the same (prompt engineering modifies input text; prompt tuning learns continuous embedding vectors); not knowing that prefix tuning adds per-layer computation while LoRA adds none after merging; confusing "soft prompts" (learned vectors) with "hard prompts" (text).
