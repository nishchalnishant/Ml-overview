# Q1: What is fine-tuning, and when should you fine-tune an LLM?

## 1. 🔹 Direct Answer
Fine-tuning adapts an LLM’s parameters to new behavior or domain knowledge using task-specific training data. You fine-tune when you need consistent **instruction-following/format**, domain reasoning patterns, or reduced refusal/error rates, and when knowledge is relatively stable or can be encoded as behavior rather than facts.

## 2. 🔹 Intuition
Prompting is “asking.” Fine-tuning is “teaching the model how to behave” so it doesn’t need extremely long prompts.

## 3. 🔹 Deep Dive
### Typical pipeline
1. Curate training examples (instruction → ideal response).
2. Choose SFT or PEFT method.
3. Train with a learning-rate schedule and proper regularization.
4. Evaluate on held-out tasks; consider alignment training if needed.

### When it works best
- Stable label/behavior requirements
- Enough high-quality examples
- Need for deterministic structure (JSON/forms)

## 4. 🔹 Practical Perspective
- **Use when:** you have strong example datasets and want stable behavior improvements.
- **Avoid when:** knowledge changes constantly (prefer RAG) or you lack labeled data.
- **Trade-offs:** training cost, risk of forgetting (catastrophic forgetting), and possible brittleness.

## 5. 🔹 Code Snippet
```python
# High-level SFT pseudocode (framework-specific)
for batch in dataloader:
    logits = model(batch["input_ids"])
    loss = cross_entropy(logits, batch["labels"])
    loss.backward(); opt.step()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Fine-tune vs RAG?  
   **A:** RAG updates facts via retrieval; fine-tuning changes behavior/format and can teach domain reasoning.
2. **Q:** Do you always need RLHF?  
   **A:** Not always; often SFT + RAG + guardrails is enough.

## 7. 🔹 Common Mistakes
- Fine-tuning to memorize facts that should be retrieved.

## 8. 🔹 Comparison / Connections
- Connects to **parametric vs non-parametric memory** (fine-tuning vs RAG).

## 9. 🔹 One-line Revision
Fine-tuning teaches the model behavior from examples; use it when behavior must be consistent and data is stable.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q2: Explain the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT).

## 1. 🔹 Direct Answer
Full fine-tuning updates **all** model parameters, which is expensive and risks forgetting. PEFT updates only a small subset (adapters/low-rank matrices), reducing compute/VRAM and often improving safety and iteration speed.

## 2. 🔹 Intuition
Full fine-tune = rewrite the whole book. PEFT = write a margin note.

## 3. 🔹 Deep Dive
### Full fine-tuning
- Optimize θ for new loss: `min_θ L(θ)`.
- Costs: large optimizer states, gradients for all weights.

### PEFT
- Freeze base model θ₀.
- Learn extra parameters Δθ (small).
- Effective model: `f(x; θ₀ + Δθ)`.

## 4. 🔹 Practical Perspective
- **Use PEFT when:** you need fast iteration, limited hardware, or multiple domain variants.
- **Trade-offs:** may have less capacity than full fine-tune; sometimes use “hybrid” (PEFT + light full-tune).

## 5. 🔹 Code Snippet
```python
for p in model.base.parameters():
    p.requires_grad = False
for p in model.adapters.parameters():
    p.requires_grad = True
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which is safer?  
   **A:** PEFT often reduces catastrophic forgetting by leaving base weights frozen.

## 7. 🔹 Common Mistakes
- Freezing incorrectly (adapters don’t receive gradients).

## 8. 🔹 Comparison / Connections
- Connects to **regularization** and **transfer learning**.

## 9. 🔹 One-line Revision
Full fine-tuning updates all weights; PEFT updates small add-on parameters for cheaper, safer adaptation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What is LoRA (Low-Rank Adaptation), and how does it work?

## 1. 🔹 Direct Answer
LoRA adapts a pretrained model by adding low-rank matrices to weight updates. Instead of learning a full ΔW, it learns `A` and `B` such that `ΔW ≈ A·B`, reducing trainable parameters.

## 2. 🔹 Intuition
You don’t need to rewrite the entire weight matrix—learn a low-rank correction.

## 3. 🔹 Deep Dive
### In linear layers
- Original: `y = W x`.
- LoRA: `W' = W + (BA)` (rank r).
- Output: `y = (W + BA) x = Wx + B(Ax)`.
### Training
- Freeze W.
- Train A and B.
### Hyperparameters
- rank `r`, scaling `α`, dropout on LoRA updates.

## 4. 🔹 Practical Perspective
- **Use when:** you want domain adaptation with limited compute.
- **Avoid when:** you need maximum capacity and can afford full fine-tune.

## 5. 🔹 Code Snippet
```python
# Conceptual LoRA update
W_eff = W + (B @ A)  # A:(r, in), B:(out, r)
y = W_eff @ x
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why low-rank works?  
   **A:** Many task adaptations lie in a subspace of parameter changes.
2. **Q:** Where do you apply LoRA?  
   **A:** Often to attention projection matrices (Q, K, V, O) and sometimes FFN projections.

## 7. 🔹 Common Mistakes
- Setting rank too low → underfitting.

## 8. 🔹 Comparison / Connections
- Connects to **matrix factorization** and PEFT.

## 9. 🔹 One-line Revision
LoRA learns a low-rank weight update by adding BA adapters to frozen weights.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What is QLoRA, and how does it enable fine-tuning on consumer hardware?

## 1. 🔹 Direct Answer
QLoRA combines **quantization** of the base model weights (often 4-bit) with LoRA adapters trained on top. This drastically reduces VRAM usage, enabling fine-tuning on smaller GPUs while keeping quality close to higher-precision training.

## 2. 🔹 Intuition
Freeze the base model in compressed form, then train small adapters on top.

## 3. 🔹 Deep Dive
### Key components
- Quantize base weights to 4-bit.
- Keep LoRA adapters in higher precision.
- Use “dequantization on the fly” for forward passes.
### Benefits
- Much lower memory footprint for optimizer/training states.

## 4. 🔹 Practical Perspective
- **Use when:** you need PEFT on limited VRAM.
- **Avoid when:** quantization harms your task—always validate with eval set.
- **Trade-offs:** possible quality drop vs FP16/BF16 full-precision, plus tuning complexity.

## 5. 🔹 Code Snippet
```python
# Pseudocode
model = load_4bit(model_name)
model = attach_lora_adapters(model, r=8)
train(model)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why not quantize adapters too?  
   **A:** Adapters often need higher precision to learn effectively.

## 7. 🔹 Common Mistakes
- Training with incompatible learning rates for QLoRA (leads to instability).

## 8. 🔹 Comparison / Connections
- Connects to **quantization** and PEFT.

## 9. 🔹 One-line Revision
QLoRA is LoRA trained on a quantized (4-bit) base model to fit into consumer hardware.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Explain Prefix Tuning and Prompt Tuning. How are they different from LoRA?

## 1. 🔹 Direct Answer
Prefix/prompt tuning learns **soft tokens** or prompts added to the input sequence, steering the model without changing its internal weights. LoRA changes internal projections by learning low-rank weight updates.

## 2. 🔹 Intuition
Prefix tuning = “write a custom preface” (learned tokens). LoRA = “modify how the model’s internal layers compute.”

## 3. 🔹 Deep Dive
### Prompt/prefix tuning
- Add learned vectors `P` before the actual tokens (prefix).
- The model attends over these learned vectors and uses them as conditioning.
### Differences from LoRA
- **Trainable parameters location:** input-conditioned prefix vs internal weight adapters.
- **Capacity/expressiveness:** depends on prefix length and training setup.

## 4. 🔹 Practical Perspective
- **Use when:** you want extremely cheap adaptation and fast domain switching.
- **Avoid when:** you need strong performance gains requiring internal adaptation (LoRA often better).

## 5. 🔹 Code Snippet
```python
# Conceptual: prepend learned prefix embeddings
emb = torch.cat([prefix_vectors, token_embeddings], dim=1)
logits = model(emb)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which is more common in practice?  
   **A:** LoRA/QLoRA in many LLM fine-tuning stacks due to strong quality/cost trade-offs.

## 7. 🔹 Common Mistakes
- Confusing prefix tuning with “hard prompt engineering” (soft tokens are learned, not human-written).

## 8. 🔹 Comparison / Connections
- Connects to **prompt engineering** and PEFT.

## 9. 🔹 One-line Revision
Prefix/prompt tuning steers via learned soft tokens; LoRA steers via low-rank internal weight updates.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What is adapter-based fine-tuning?

## 1. 🔹 Direct Answer
Adapter-based fine-tuning inserts small neural modules (“adapters”) inside the transformer layers. Only adapter parameters are trained while the base model stays frozen.

## 2. 🔹 Intuition
Add small “plug-in components” into each layer that can learn task-specific transformations.

## 3. 🔹 Deep Dive
- Insert adapters after attention or FFN blocks.
- Adapter typically: down-project → nonlinearity → up-project (bottleneck).
- Train adapters + maybe layer norms; freeze base.

## 4. 🔹 Practical Perspective
- **Use when:** you want efficient domain adaptation across many tasks.
- **Trade-offs:** memory grows with adapter count; may be slower inference if adapters aren’t fused.

## 5. 🔹 Code Snippet
```python
# Conceptual adapter block
z = W_up(gelu(W_down(h)))
h = h + z
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How does it compare to LoRA?  
   **A:** LoRA modifies linear projections with low-rank updates; adapters add extra modules.

## 7. 🔹 Common Mistakes
- Training too many parameters and losing PEFT advantages.

## 8. 🔹 Comparison / Connections
- Connects to PEFT families.

## 9. 🔹 One-line Revision
Adapter tuning learns small bottleneck modules inserted into transformer layers, freezing the base model.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is RLHF (Reinforcement Learning from Human Feedback), and how is it used to align LLMs?

## 1. 🔹 Direct Answer
RLHF aligns LLM behavior to human preferences by training a **reward model** from human-labeled comparisons, then optimizing the LLM policy to maximize reward (often with PPO) while staying close to a reference model.

## 2. 🔹 Intuition
Humans grade outputs; the system learns “what humans like,” then updates the model to do that more.

## 3. 🔹 Deep Dive
### Pipeline
1. Collect preference data: `(prompt, response_A, response_B)` with “A preferred.”
2. Train reward model R(x, y) to predict which response is preferred.
3. Optimize policy π (the LLM) using RL to maximize expected reward with a KL penalty to avoid drifting:
   - maximize `E[R(prompt, response)] - β * KL(π || π_ref)`

## 4. 🔹 Practical Perspective
- **Use when:** you need behavioral alignment (helpfulness/harmlessness) beyond SFT.
- **Avoid when:** reward model is unreliable or expensive to maintain—consider DPO or other preference optimization.
- **Trade-offs:** RL training is costly and can suffer from reward hacking.

## 5. 🔹 Code Snippet
```python
# Pseudocode objective
loss = -E[reward_model(prompt, sampled_response)] + beta * KL_div(pi, pi_ref)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why KL penalty?  
   **A:** Prevents over-optimizing the reward model and preserves base capabilities.
2. **Q:** Biggest RLHF risks?  
   **A:** Reward hacking and alignment tax.

## 7. 🔹 Common Mistakes
- Treating reward as truth rather than a proxy for human preference.

## 8. 🔹 Comparison / Connections
- Connects to **DPO** and **alignment**.

## 9. 🔹 One-line Revision
RLHF is reward-model learning from preferences plus RL optimization with a KL anchor to align LLM behavior.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q8: What is instruction tuning, and why is it important for chat models?

## 1. 🔹 Direct Answer
Instruction tuning fine-tunes an LLM on pairs of **(instruction, response)** so it learns to follow user intent and produce helpful, formatted outputs in a conversational style.

## 2. 🔹 Intuition
It teaches the model the job: “Given instructions, respond like an assistant.”

## 3. 🔹 Deep Dive
- Objective: maximize next-token likelihood of the desired response given the instruction (SFT on instruction-response).
- Provides the “chat behavior” baseline before alignment (RLHF/DPO).

## 4. 🔹 Practical Perspective
- **Use when:** you build chat/completion systems and need consistent assistant behavior.
- **Avoid when:** you have very small instruction datasets without careful curation.

## 5. 🔹 Code Snippet
```python
# SFT on (instruction, response)
loss = cross_entropy(model(inst_tokens), response_tokens)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How is it different from generic fine-tuning?  
   **A:** It’s specifically aligned to instruction-following and conversational patterns.

## 7. 🔹 Common Mistakes
- Fine-tuning on low-quality or inconsistent instruction formats.

## 8. 🔹 Comparison / Connections
- Connects to SFT and alignment training pipeline stages.

## 9. 🔹 One-line Revision
Instruction tuning teaches an LLM to reliably follow user instructions in chat-like behavior.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: How do you prepare a dataset for fine-tuning an LLM?

## 1. 🔹 Direct Answer
Prepare by curating high-quality instruction-response examples, ensuring consistent formatting, removing duplicates/noise, and building a coverage plan across intents. For alignment, create preference pairs or structured feedback.

## 2. 🔹 Intuition
Garbage in → garbage out. Fine-tuning is only as good as the examples you teach.

## 3. 🔹 Deep Dive
### Data steps
1. Define task schema (input fields, output format).
2. Collect data: human-written, curated logs, synthetic generation with filtering.
3. Clean: deduplicate, normalize, remove unsafe content if required.
4. Split: train/validation/test with no leakage.
5. For LoRA/PEFT: ensure sequence length and truncation strategies are consistent.

## 4. 🔹 Practical Perspective
- **Use when:** you need measurable improvements and predictable behavior.
- **Avoid when:** datasets are unbalanced (model overfits to frequent intents).

## 5. 🔹 Code Snippet
```python
# Basic filtering concept
dataset = [ex for ex in dataset if ex["response"].strip() and len(ex["input"])<max_len]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How do you handle safety?  
   **A:** Use refusal examples, safe completions, and filtered preference data.

## 7. 🔹 Common Mistakes
- Training on overlapping train/test sets.

## 8. 🔹 Comparison / Connections
- Connects to **evaluation splits** and **data governance**.

## 9. 🔹 One-line Revision
Dataset prep requires schema consistency, high-quality curated examples, strong cleaning, and correct train/val/test splits.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: What is catastrophic forgetting, and how do you prevent it during fine-tuning?

## 1. 🔹 Direct Answer
Catastrophic forgetting is when fine-tuning causes a model to lose previously learned capabilities. Prevent it by mixing in general-domain data, using PEFT methods (often safer), regularizing updates, and early stopping with validation.

## 2. 🔹 Intuition
Learning a new skill can accidentally overwrite older knowledge.

## 3. 🔹 Deep Dive
### Mitigations
- **Replay/mixing:** include a portion of original/pretraining-like data during fine-tuning.
- **PEFT:** freeze base weights (LoRA/adapter) reduces forgetting.
- **Regularization:** KL penalty to reference model, smaller learning rate.
- **Validation + early stopping:** stop before degradation.

## 4. 🔹 Practical Perspective
- **Use when:** domain fine-tuning degrades general reasoning/format skills.
- **Avoid when:** you can’t afford adding general data; then rely more on PEFT + KL constraints.

## 5. 🔹 Code Snippet
```python
# Pseudocode: KL regularization
loss = task_loss + alpha * KL(pi_finetuned || pi_reference)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to detect?  
   **A:** Evaluate on general benchmarks + domain benchmarks.

## 7. 🔹 Common Mistakes
- Using too large learning rate or too many epochs.

## 8. 🔹 Comparison / Connections
- Connects to **regularization** and **bias-variance** trade-offs.

## 9. 🔹 One-line Revision
Prevent catastrophic forgetting by using PEFT, mixing general data, regularizing to a reference model, and early stopping.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q11: When should you choose fine-tuning over RAG over prompt engineering?

## 1. 🔹 Direct Answer
Choose prompt engineering for quick behavioral shaping with no training data. Choose RAG when knowledge must be retrieved (fresh/internal facts) and citations matter. Choose fine-tuning when you need stable improvements in behavior/format and you have quality training data.

## 2. 🔹 Intuition
Prompting is configuration. RAG is external memory. Fine-tuning is internal behavior change.

## 3. 🔹 Deep Dive
### Decision criteria
- Knowledge changes? → RAG.
- Behavior/format consistency needed? → Fine-tune.
- No/limited data? → Prompt engineering (and RAG).

## 4. 🔹 Practical Perspective
- **Use when:** you must reduce prompt length or improve reliability systematically.
- **Trade-offs:** fine-tune can be expensive and can degrade generalization.

## 5. 🔹 Code Snippet
```python
# Typical hybrid
answer = llm.generate(rag_prompt(query, retrieved_docs), system="Output strict JSON.")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** If RAG is accurate but formatting is wrong?  
   **A:** Fine-tune for structured output behavior; keep RAG for knowledge.

## 7. 🔹 Common Mistakes
- Overusing fine-tuning for volatile facts.

## 8. 🔹 Comparison / Connections
- Connects to **parametric vs non-parametric memory**.

## 9. 🔹 One-line Revision
Prompt if behavior tweaks are enough, RAG for knowledge, fine-tune for stable behavioral improvements with good data.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q12: How do you evaluate a fine-tuned model's performance?

## 1. 🔹 Direct Answer
Evaluate with: (1) offline benchmark tests on task-specific and general capabilities, (2) structured output correctness, and (3) robustness and safety checks. For RAG+fine-tune systems, evaluate end-to-end faithfulness and latency/cost.

## 2. 🔹 Intuition
You verify both “does it do the task” and “did it keep the skills it had.”

## 3. 🔹 Deep Dive
### Evaluation suite
- Task accuracy / exact match / F1 (as applicable).
- Format validation (JSON schema, tool call correctness).
- Safety refusal rate, jailbreak robustness.
- General capability drift (reasoning, language fluency).

## 4. 🔹 Practical Perspective
- **Use when:** you have representative test sets.
- **Avoid when:** test data overlaps training set.

## 5. 🔹 Code Snippet
```python
def is_valid_json(text):
    try:
        json.loads(text); return True
    except: return False
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Human eval vs automatic?  
   **A:** Use both; automatic for iteration, human for nuanced quality and safety.

## 7. 🔹 Common Mistakes
- Only evaluating on training-like examples.

## 8. 🔹 Comparison / Connections
- Connects to ML evaluation best practices.

## 9. 🔹 One-line Revision
Evaluate with task metrics, format correctness, general capability retention, and safety/robustness tests.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: What is synthetic data generation, and how do you use it for fine-tuning?

## 1. 🔹 Direct Answer
Synthetic data generation uses models (often the base LLM or specialized generation pipelines) to create training examples when human-labeled data is scarce. You then filter, deduplicate, and validate synthetic outputs before using them to fine-tune.

## 2. 🔹 Intuition
You create practice questions automatically, but you must ensure they’re correct.

## 3. 🔹 Deep Dive
### Workflow
1. Generate candidates using prompts/specs.
2. Filter with automatic validators (schema checks, unit tests, constraint checks).
3. (Optional) human review for a subset.
4. Include synthetic data with weighting so it doesn’t dominate.

## 4. 🔹 Practical Perspective
- **Use when:** data is limited or you need coverage of edge cases.
- **Avoid when:** generator is untrustworthy; otherwise it teaches wrong behavior.
- **Trade-offs:** cost vs coverage.

## 5. 🔹 Code Snippet
```python
synthetic = [ex for ex in generated if schema_validator(ex["response"])]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to prevent reinforcing model biases?  
   **A:** Use diverse prompts, adversarial generation, and evaluate for bias regressions.

## 7. 🔹 Common Mistakes
- Training directly on unfiltered synthetic hallucinations.

## 8. 🔹 Comparison / Connections
- Connects to **self-training** and data quality governance.

## 9. 🔹 One-line Revision
Synthetic data can help when filtered and validated so it expands coverage without teaching errors.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: What are the key hyperparameters for fine-tuning (learning rate, epochs, batch size, LoRA rank)?

## 1. 🔹 Direct Answer
Key hyperparameters include learning rate (controls update magnitude), epochs (how many passes), batch size (affects gradient noise and stability), and LoRA rank `r` (capacity of low-rank adapters). You tune them with validation to avoid overfitting/forgetting.

## 2. 🔹 Intuition
Learning rate is “step size,” epochs is “how long you train,” batch size is “stability,” and rank is “how much flexibility adapters have.”

## 3. 🔹 Deep Dive
### Typical guidelines
- Use smaller LR with PEFT; schedule with warmup + decay.
- Early stop when validation stops improving.
- LoRA rank too low underfits; too high overfits/raises cost.

## 4. 🔹 Practical Perspective
- **Use when:** you have eval loops.
- **Avoid when:** you can’t validate frequently—then hyperparameter tuning is risky.

## 5. 🔹 Code Snippet
```python
cfg = {"lr": 2e-5, "epochs": 2, "batch_size": 8, "lora_r": 8}
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to pick LR for QLoRA?  
   **A:** Start smaller and validate; quantization changes stability.

## 7. 🔹 Common Mistakes
- Training too many epochs leads to forgetting and overfitting.

## 8. 🔹 Comparison / Connections
- Connects to optimization and regularization.

## 9. 🔹 One-line Revision
Tune LR/epochs/batch and LoRA rank with validation to balance learning capacity vs overfitting/forgetting.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: How do you fine-tune a model for a specific domain (legal, medical, finance)?

## 1. 🔹 Direct Answer
Fine-tune by curating domain-relevant instruction data, including citations/grounding targets when needed, using PEFT (LoRA/QLoRA) to reduce risk/cost, and validating against domain benchmarks and safety constraints (hallucination and compliance).

## 2. 🔹 Intuition
You’re teaching the model domain language and response standards—not just facts.

## 3. 🔹 Deep Dive
### Steps
1. Gather: domain Q&A, templates, exemplars, refusal examples.
2. Preprocess: normalize terms, ensure output formats.
3. Train: SFT/PEFT; possibly follow with preference tuning (RLHF/DPO) for safety.
4. Evaluate: factuality/grounding and policy compliance.

## 4. 🔹 Practical Perspective
- **Use when:** you need consistent domain formatting and legal-style responses.
- **Avoid when:** the domain requires strict factual grounding that changes—prefer RAG + fine-tune behavior.

## 5. 🔹 Code Snippet
```python
system = "Answer in legal memo style. If evidence missing, say 'not found'."
prompt = system + "\n" + build_domain_prompt(query)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How do you reduce harmful medical advice?  
   **A:** Add safety refusals, verify with tools, and restrict unsafe content generation.

## 7. 🔹 Common Mistakes
- Fine-tuning without safety examples, leading to confident unsafe outputs.

## 8. 🔹 Comparison / Connections
- Connects to RAG vs fine-tuning and AI safety.

## 9. 🔹 One-line Revision
Domain fine-tuning is careful data curation + PEFT + strict evaluation and safety checks.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q16: What is continual pre-training, and when would you use it?

## 1. 🔹 Direct Answer
Continual pre-training continues self-supervised training on fresh domain text to improve general language modeling and domain representations. Use it when you want domain understanding improvements and have lots of unlabeled domain data.

## 2. 🔹 Intuition
It’s like re-reading textbooks in your target subject before learning new tasks.

## 3. 🔹 Deep Dive
- Objective: next-token prediction (or masked LM depending on model).
- Continue training on domain corpus.
- Then SFT/PEFT for instructions.

## 4. 🔹 Practical Perspective
- **Use when:** abundant unlabeled domain text exists and you need better representations.
- **Avoid when:** you only have small labeled data and mainly need behavior formatting (SFT/PEFT is enough).

## 5. 🔹 Code Snippet
```python
for batch in domain_text_batches:
    loss = lm_loss(model(batch["input_ids"]), batch["labels"])
    loss.backward(); opt.step()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to avoid catastrophic drift?  
   **A:** Use smaller LR, mix in general corpus, and validate frequently.

## 7. 🔹 Common Mistakes
- Continual pretraining without safety alignment.

## 8. 🔹 Comparison / Connections
- Connects to representation learning and catastrophic forgetting.

## 9. 🔹 One-line Revision
Continual pre-training improves domain representations using self-supervised learning on unlabeled domain text.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: How do you merge multiple LoRA adapters?

## 1. 🔹 Direct Answer
Merge LoRA adapters by combining their low-rank updates (weighted sum of their BA matrices) or by applying one adapter sequentially depending on implementation. After merging, validate on eval tasks because merges can interact.

## 2. 🔹 Intuition
Adapters are different “margin notes.” Merging is blending their corrections.

## 3. 🔹 Deep Dive
### Typical method
- If each adapter produces `ΔW_i = B_i A_i`, merged update:
  - `ΔW_merge = Σ_i w_i * ΔW_i`
- Apply to base weights (or keep as merged adapter).

## 4. 🔹 Practical Perspective
- **Use when:** you have multiple domain adapters (legal + finance) and want a unified model.
- **Avoid when:** adapters conflict strongly; merging can harm behavior.

## 5. 🔹 Code Snippet
```python
DeltaW = w1*(B1@A1) + w2*(B2@A2)
W_merged = W_base + DeltaW
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How do you pick weights `w_i`?  
   **A:** Grid search on validation metrics for each adapter’s target domain.

## 7. 🔹 Common Mistakes
- Merging without validating both safety and task behavior.

## 8. 🔹 Comparison / Connections
- Connects to ensemble/mixture of policies at adapter level.

## 9. 🔹 One-line Revision
Merge LoRA adapters by weighted combination of low-rank updates and validate because adapters can conflict.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: What is the difference between SFT (Supervised Fine-Tuning) and alignment training?

## 1. 🔹 Direct Answer
SFT trains the model to follow instructions using labeled examples (maximize likelihood of target responses). Alignment training further shapes the model toward human preferences (helpfulness/harmlessness) using preference data and methods like RLHF or DPO.

## 2. 🔹 Intuition
SFT teaches “how to respond.” Alignment teaches “what kind of responses humans prefer.”

## 3. 🔹 Deep Dive
- **SFT objective:** standard supervised learning on instruction-response pairs.
- **Alignment objective:** optimize for preferences; reward models or direct preference optimization.

## 4. 🔹 Practical Perspective
- **Use when:** you want safer and more helpful behavior beyond matching labeled outputs.
- **Trade-offs:** alignment can introduce an “alignment tax” (capability drop on hard tasks).

## 5. 🔹 Code Snippet
```python
# SFT: token-level cross entropy
loss_sft = CE(logits, target_tokens)

# Alignment (conceptual): preference loss/reward optimization
loss_align = -log_prob_preferred + log_prob_dispreferred  # depends on method
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Is DPO an alignment method?  
   **A:** Yes—preference optimization without explicit RL.

## 7. 🔹 Common Mistakes
- Treating SFT labels as equivalent to preference alignment signals.

## 8. 🔹 Comparison / Connections
- Connects to RLHF vs DPO.

## 9. 🔹 One-line Revision
SFT learns to mimic target responses; alignment trains the model toward human preferences and safety.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: What is RLAIF (RL from AI Feedback), and how does it differ from RLHF?

## 1. 🔹 Direct Answer
RLAIF uses feedback generated by an AI model instead of human annotators to create preference/reward signals. RLHF uses human-labeled preferences. RLAIF can scale faster but may embed model biases or reward model errors.

## 2. 🔹 Intuition
Humans are expensive; AI graders are cheap. RLAIF uses AI to grade to reduce human workload.

## 3. 🔹 Deep Dive
- Create preference pairs using an evaluator LLM (with instructions and constraints).
- Train reward model or apply preference optimization.
- Optimize policy with RL or preference methods.

## 4. 🔹 Practical Perspective
- **Use when:** you need large preference datasets with limited human labeling.
- **Avoid when:** the evaluator LLM is biased or can be exploited (reward hacking risk increases).
- **Trade-offs:** scale vs correctness of feedback.

## 5. 🔹 Code Snippet
```python
# Pseudocode
pref = evaluator_llm.compare(prompt, resp_a, resp_b)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to validate RLAIF?  
   **A:** Human auditing on a representative subset + red teaming.

## 7. 🔹 Common Mistakes
- Assuming AI feedback is “ground truth.”

## 8. 🔹 Comparison / Connections
- Connects to alignment data quality and safety.

## 9. 🔹 One-line Revision
RLAIF replaces human preference labels with AI-generated feedback to scale alignment, at the risk of AI bias.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q20: What is knowledge distillation for fine-tuning, and what are the legal considerations?

## 1. 🔹 Direct Answer
Knowledge distillation trains a smaller student model to match a larger teacher’s outputs (soft targets like logits or generated responses). Legally, you must have rights to training data/outputs and ensure compliance with copyright/terms of use.

## 2. 🔹 Intuition
Teach the smaller model to imitate a smarter teacher.

## 3. 🔹 Deep Dive
### Distillation methods
- **Logit matching:** minimize KL divergence between teacher and student distributions.
- **Response imitation:** train on teacher-generated outputs.
### Objective
`L = KL(p_teacher || p_student)` (over tokens), often with temperature scaling.

## 4. 🔹 Practical Perspective
- **Use when:** you need smaller, faster models for cost/latency.
- **Avoid when:** teacher outputs encode problematic content; filter and validate.

## 5. 🔹 Code Snippet
```python
# Pseudocode distillation loss
loss = kl_div(student_logits_T, teacher_logits_T)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Is distillation different from SFT?  
   **A:** Yes—targets come from a teacher’s soft outputs rather than human labels.

## 7. 🔹 Common Mistakes
- Training on teacher outputs without filtering unsafe or copyrighted content.

## 8. 🔹 Comparison / Connections
- Connects to model efficiency and compliance.

## 9. 🔹 One-line Revision
Distillation transfers teacher knowledge to a smaller student; ensure data/output licensing and compliance.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: Your fine-tuned LLM produces factually wrong outputs due to training data quality issues. How do you fix it?

## 1. 🔹 Direct Answer
Fix by improving data quality: remove incorrect/low-quality examples, add retrieval grounding (RAG) for facts, and re-train with curated evidence and verification constraints. Use evaluation focusing on factuality/entailment.

## 2. 🔹 Intuition
You taught it the wrong answers; the model confidently repeats them.

## 3. 🔹 Deep Dive
### Fix workflow
1. Identify failure clusters: domains, prompt types, incorrect entity extraction.
2. Audit dataset: provenance, label correctness, and contradictory examples.
3. Clean or relabel.
4. Add evidence-based training: include citations or “only if supported” examples.
5. Re-train (prefer PEFT).

## 4. 🔹 Practical Perspective
- **Use when:** wrong facts systematically correlate with bad training data.
- **Avoid when:** facts must always be current—then RAG is the right architecture.

## 5. 🔹 Code Snippet
```python
# Data cleaning concept
dataset = [ex for ex in dataset if verify_fact_with_source(ex)]
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How do you verify facts at scale?  
   **A:** Use retrieval+entailment models or external ground truth where available.

## 7. 🔹 Common Mistakes
- Overfitting the model by only training on the most frequent incorrect pattern.

## 8. 🔹 Comparison / Connections
- Connects to factuality evaluation and RAG vs fine-tuning.

## 9. 🔹 One-line Revision
Wrong facts come from data—clean/relabel and use retrieval grounding for factual reliability.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q22: You must choose between LoRA and full fine-tuning for a domain-specific assistant. How do you decide?

## 1. 🔹 Direct Answer
Choose LoRA/PEFT when you need efficiency, limited hardware, fast iteration, and want to reduce forgetting risk. Choose full fine-tuning when you need maximum capacity, can afford compute, and have strong curated datasets to avoid degradation.

## 2. 🔹 Intuition
LoRA is a “cheap, safe adapter,” full fine-tuning is a “full rewrite.”

## 3. 🔹 Deep Dive
### Decision criteria
- Data size/quality
- Hardware budget
- Risk tolerance (forgetting/safety)
- Time-to-iterate

## 4. 🔹 Practical Perspective
- **Use when:** domain behavior can be expressed via adapter changes → LoRA.
- **Avoid when:** you need large representational changes and have enough data/compute → full FT.

## 5. 🔹 Code Snippet
```python
# Heuristic: start with LoRA baseline + evaluate
model = attach_lora(model, r=8); train(model); evaluate()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** What’s a good experimental plan?  
   **A:** LoRA baseline first, then increase rank or move to full FT if metrics plateau.

## 7. 🔹 Common Mistakes
- Choosing full FT without validating whether adapters suffice.

## 8. 🔹 Comparison / Connections
- Connects to optimization budget and regularization.

## 9. 🔹 One-line Revision
Pick LoRA for efficient safer adaptation; pick full fine-tuning only when you truly need capacity and can afford cost/risk.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q23: Your fine-tuned model memorized training data verbatim instead of learning patterns. How do you fix it overfitting?

## 1. 🔹 Direct Answer
Prevent/mitigate overfitting by improving dataset diversity, reducing epochs/learning rate, using stronger regularization (dropout/weight decay), and adding early stopping based on validation loss/metrics. For PEFT, reduce LoRA rank or adapter capacity if needed.

## 2. 🔹 Intuition
The model learned the exact examples, not the underlying rule—so it repeats them.

## 3. 🔹 Deep Dive
### Symptoms
- Training accuracy high; validation poor.
- High similarity to training strings; “verbatim memorization.”
### Fixes
- Deduplicate near-duplicates in dataset.
- Use data augmentation or more varied prompts.
- Early stopping + smaller LR.
- Reduce adapter rank (LoRA) or dropout.

## 4. 🔹 Practical Perspective
- **Use when:** you see exact duplication patterns.
- **Trade-offs:** too much regularization can underfit.

## 5. 🔹 Code Snippet
```python
# Early stopping concept
if val_metric_improves is False for patience_steps:
    stop_training()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** How to detect memorization?  
   **A:** Run similarity checks between generated outputs and training examples.

## 7. 🔹 Common Mistakes
- Only lowering loss without checking factual generalization.

## 8. 🔹 Comparison / Connections
- Connects to classical **overfitting** and regularization.

## 9. 🔹 One-line Revision
Overfitting/memorization is fixed with data de-duplication, regularization, smaller learning rates, fewer epochs, and adapter capacity tuning.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q24: Your fine-tuned LLM forgot its general capabilities after domain-specific fine-tuning. How do you fix catastrophic forgetting?

## 1. 🔹 Direct Answer
Fix catastrophic forgetting by mixing general-domain examples during training, using lower learning rates, applying KL regularization to a reference model, and preferring PEFT methods that preserve base weights.

## 2. 🔹 Intuition
You focused too much on the new domain and washed out general skills.

## 3. 🔹 Deep Dive
### Fix plan
1. Diagnose forgetting via general eval suite.
2. Adjust training:
   - add replay/general data
   - reduce LR and epochs
   - add KL-to-reference penalty
3. Validate after each adjustment.

## 4. 🔹 Practical Perspective
- **Use when:** general reasoning or fluency drops.
- **Trade-offs:** replay adds data/compute but improves retention.

## 5. 🔹 Code Snippet
```python
loss = task_loss + alpha * KL(pi_finetuned || pi_reference)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Is this solved by more data always?  
   **A:** Not always—data mix and optimization schedule matter.

## 7. 🔹 Common Mistakes
- Overtraining domain data without validation checkpoints.

## 8. 🔹 Comparison / Connections
- Connects to regularization and bias-variance trade-offs.

## 9. 🔹 One-line Revision
Catastrophic forgetting is mitigated by replay/mixing general data, lower LR, and KL regularization to preserve base behavior.

## 10. 🔹 Difficulty Tag
🔴 Hard

---

# Q25: Your RLHF preference data has low annotator agreement. How do you ensure data quality?

## 1. 🔹 Direct Answer
Low annotator agreement means the preference signal is noisy. Improve quality with clearer annotation guidelines, training annotators with calibration tasks, collecting multiple annotations per pair and aggregating (e.g., majority vote or Bradley–Terry), and auditing disagreements with expert review.

## 2. 🔹 Intuition
If graders disagree, you’re teaching the reward model conflicting rules.

## 3. 🔹 Deep Dive
### Quality controls
- Calibration: measure annotator reliability.
- Agreement-aware labeling: require consensus for strong preferences.
- Use probabilistic preference models (e.g., Bradley–Terry) rather than only hard labels.
- Stratify disagreement by prompt type and iterate guidelines.

## 4. 🔹 Practical Perspective
- **Use when:** preference datasets are expensive and noisy.
- **Avoid when:** you proceed without any audit; the reward model may learn spurious signals.

## 5. 🔹 Code Snippet
```python
# Bradley–Terry style aggregation (conceptual)
p = sigmoid(score_diff)
loss = - (y*log(p) + (1-y)*log(1-p))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** What do you do with ambiguous pairs?  
   **A:** Downweight or exclude based on confidence.

## 7. 🔹 Common Mistakes
- Treating low agreement as “just noise” instead of investigating guidelines.

## 8. 🔹 Comparison / Connections
- Connects to dataset quality and reward model reliability.

## 9. 🔹 One-line Revision
High-quality RLHF data requires clearer guidelines, calibration, multi-annotator aggregation, and disagreement audits.

## 10. 🔹 Difficulty Tag
🔴 Hard

