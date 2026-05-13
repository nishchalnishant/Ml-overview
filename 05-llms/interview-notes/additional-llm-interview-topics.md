# Additional LLM interview topics

Extra **questions and answers** that show up often in **machine learning / GenAI** interviews—aligned with the same template as the rest of this folder.

---

# Q1: What are scaling laws, and what does “Chinchilla-optimal” training mean?

## 1. 🔹 Direct Answer
**Scaling laws** describe how **loss** improves predictably as you scale **model size (N)**, **dataset size (D)**, and **compute (C)**—often power-law relationships in the pretraining regime. **Chinchilla** showed many models were **undertrained**: for a fixed compute budget, **smaller models on more data** often beat **larger models on less data**—**compute-optimal** pairs balance **N** and **D** (roughly **D ∝ N** in the studied regime), not “largest N possible on fixed tokens.”

## 2. 🔹 Intuition
Throwing parameters at the problem without **enough tokens** leaves **capacity unused**; data is not free either—**match** model size to **how much quality text you can train on**.

## 3. 🔹 Deep Dive
- Empirical fits: loss vs scale; **IsoFLOP** curves compare runs at same **FLOPs**.
- **Inference** prefers smaller models if quality adequate—**training** scaling ≠ **serving** scaling.

## 4. 🔹 Practical Perspective
Interview: connect to **budget**, **data pipeline quality**, and **serving** latency—not only parameter count.

## 5. 🔹 Code Snippet
```text
FLOPs ~ 6ND (decoder-only rough order-of-magnitude storytelling)
```

## 6. 🔹 Interview Follow-ups
1. Q: Why do companies still train huge models?  
   A: **Capability ceiling**, **downstream** fine-tunes, **ecosystem**—Chinchilla is about **pretrain efficiency**, not every product constraint.
2. Q: Does scaling fix alignment?  
   A: **Partially**—does not replace **RLHF/DPO** and **evals**.

## 7. 🔹 Common Mistakes
Quoting scaling laws as **universal** outside pretrain distribution or **data-limited** regimes.

## 8. 🔹 Comparison / Connections
Kaplan vs Chinchilla trade-offs, data quality vs quantity.

## 9. 🔹 One-line Revision
Scaling laws link loss to N, D, C; Chinchilla argues many LLMs were compute-suboptimal—balance model size and training tokens.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q2: Explain RLHF, PPO, and DPO in interview terms—what problem does each solve?

## 1. 🔹 Direct Answer
**RLHF** (**Reinforcement Learning from Human Feedback**) aligns a model to **human preferences** by training a **reward model** from comparisons, then optimizing the policy (often with **PPO**) to **increase reward** while staying near the reference model (**KL** penalty). **PPO** is a **stable policy-gradient** algorithm used in that inner loop. **DPO** (**Direct Preference Optimization**) **skips explicit reward modeling** and optimizes preferences **directly** from pairwise data with a **classification-style** loss on policy ratios—**simpler** and **popular** for alignment fine-tunes.

## 2. 🔹 Intuition
RLHF: learn “what humans like,” then **steer** generation. DPO: **directly** learn “prefer A over B” without a separate reward **network** if assumptions hold.

## 3. 🔹 Deep Dive
- RLHF stack: **SFT** → **reward model** → **PPO** (or variants).
- **DPO** reparameterizes the optimal policy under Bradley-Terry-style preferences.

## 4. 🔹 Practical Perspective
**DPO** often **easier** to ship in research/org settings; **RLHF+PPO** still common at **large** scale with **infrastructure** for reward models and **online** data.

## 5. 🔹 Code Snippet
```text
RLHF: max E[r(x,y)] - β KL(π || π_ref)
DPO: implicit reward from preference pairs → direct loss on log π / log π_ref
```

## 6. 🔹 Interview Follow-ups
1. Q: Why KL to reference?  
   A: **Prevent** **mode collapse** and **forgetting** base capabilities / **toxic** exploitation.
2. Q: ORPO / IPO?  
   A: Other **preference** objectives—know they exist as **iterations** on DPO.

## 7. 🔹 Common Mistakes
Saying “DPO replaces RLHF everywhere”—**trade-offs** and **product** constraints differ.

## 8. 🔹 Comparison / Connections
Constitutional AI, RLAIF (AI feedback), KTO.

## 9. 🔹 One-line Revision
RLHF uses reward model + often PPO; DPO aligns from preferences without separate reward model—both encode “prefer this output” under constraints.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q3: What is a Mixture of Experts (MoE), and why use it in LLMs?

## 1. 🔹 Direct Answer
**MoE** layers replace dense FFN blocks with **multiple expert** networks and a **router** that picks **one or few** experts per token (**top-k**). **Total parameters** grow, but **activated** parameters per token stay **smaller**—**higher capacity** with **similar** **per-token** **compute** (routing overhead aside). Used to **scale** models without **linear** FFN cost explosion.

## 2. 🔹 Intuition
Like a **hospital**: many specialists, but each patient sees **only** the relevant few.

## 3. 🔹 Deep Dive
Challenges: **load balancing** (experts **idle** or **overloaded**), **routing** instability, **training** at scale. **Auxiliary** load-balancing losses help.

## 4. 🔹 Practical Perspective
**Serving**: expert **parallelism**, **all-to-all** comms—**harder** ops than dense models.

## 5. 🔹 Code Snippet
```text
y = sum_{i in topk} gate_i * Expert_i(x)
```

## 6. 🔹 Interview Follow-ups
1. Q: MoE vs wide dense?  
   A: MoE **specializes** subspaces; **routing** can **hurt** if **data** not diverse enough.

## 7. 🔹 Common Mistakes
Equating **total params** with **inference FLOPs**—MoE decouples partially.

## 8. 🔹 Comparison / Connections
Conditional computation, Switch Transformer, expert parallelism.

## 9. 🔹 One-line Revision
MoE scales model capacity by activating sparse subsets of experts per token—watch routing and load balancing.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q4: How do you extend or extrapolate context length beyond training (RoPE scaling, YaRN, etc.)?

## 1. 🔹 Direct Answer
**Rotary (RoPE)** encodes position by **rotating** Q/K in subspaces; **trained** on length **L** may **fail** at **longer** lengths. **Mitigations**: **position interpolation** (squeeze positions into trained range), **NTK-aware** scaling, **YaRN** (blending **attention** and **interpolation** factors), **long fine-tunes** on long data. Goal: stable **perplexity** and **retrieval** quality at **longer** context.

## 2. 🔹 Intuition
The model learned **“where we are”** in a **band** of positions—**stretch** or **rescale** the dial carefully so attention **does not explode**.

## 3. 🔹 Deep Dive
**ALiBi** uses **distance bias** instead of RoPE—different **extrapolation** story.

## 4. 🔹 Practical Perspective
**KV cache** memory grows **linearly** with context—**systems** limit often hits before **math** limit.

## 5. 🔹 Code Snippet
```text
RoPE theta scaled: base * factor for longer context finetunes
```

## 6. 🔹 Interview Follow-ups
1. Q: “Lost in the middle”?  
   A: **U-shaped** attention quality in long contexts—**RAG** chunking and **reranking** help.

## 7. 🔹 Common Mistakes
Claiming **any** model **generalizes** to **10×** context **without** **eval** or **finetune**.

## 8. 🔹 Comparison / Connections
Sliding window attention, Ring Attention (distributed long context).

## 9. 🔹 One-line Revision
Context extension combines positional encoding fixes (RoPE scaling/YaRN), long data finetuning, and KV memory engineering.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q5: What causes LLM hallucinations, and how do you reduce them in production?

## 1. 🔹 Direct Answer
**Hallucinations** are **confident** outputs **unsupported** by **ground truth** or **retrieved** evidence—causes include **training** objective (next-token **imitation**), **lack of grounding**, **exposure** to **wrong** patterns, **prompt** ambiguity, and **overlong** context **noise**. **Mitigations**: **RAG** with **citations**, **retrieval** **rerankers**, **constrained** generation, **lower temperature** for factuality-sensitive tasks, **tool use** (calculator, DB), **refusal** when **uncertain**, **evals** (faithfulness, **NLI** checks), **human** review for **high-stakes**.

## 2. 🔹 Intuition
The model **predicts plausible text**, not **truth**—unless **anchored** to **evidence** or **tools**.

## 3. 🔹 Deep Dive
**SFT/RLHF** can reduce some **toxic** hallucinations but **not** eliminate **fact** errors without **grounding**.

## 4. 🔹 Practical Perspective
**Product**: show **sources**, **confidence** disclaimers, **feedback** loops to **fix** bad answers.

## 5. 🔹 Code Snippet
```python
# sketch: answer only from retrieved passages
system = "Answer ONLY using the provided passages. If unknown, say you don't know."
```

## 6. 🔹 Interview Follow-ups
1. Q: Hallucination vs creativity?  
   A: Same mechanism—**task** defines whether **fabrication** is **failure** or **feature** (fiction).

## 7. 🔹 Common Mistakes
**Only** prompt-tuning without **retrieval** or **eval** on **domain** facts.

## 8. 🔹 Comparison / Connections
Calibration, factuality benchmarks (TruthfulQA-style).

## 9. 🔹 One-line Revision
Hallucination is confident unsupported text—mitigate with RAG, tools, constraints, and measurement—not prompts alone.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What is structured output (JSON mode, grammars, constrained decoding)?

## 1. 🔹 Direct Answer
**Structured output** forces the LLM to emit **valid** **syntax** (JSON, SQL, tool **arguments**) via **prompting**, **grammar-masked** decoding (**GBNF**, **outlines**), **constrained** **logit** processors, or API **modes** that **restrict** tokens step-by-step. **Reduces** parse failures and **downstream** **errors** vs **free-form** text.

## 2. 🔹 Intuition
Instead of “please output JSON,” **block** illegal tokens at **generation** time—**guarantee** **validity** (for a given grammar).

## 3. 🔹 Deep Dive
**Function calling** schemas map to **JSON** **arguments**; **compiler**-like **FSMs** for **regex**/CFG.

## 4. 🔹 Practical Perspective
**Latency** cost for **mask** computation—**trade** vs **reliability**.

## 5. 🔹 Code Snippet
```python
# Conceptual: logits[position, :] = -inf for illegal next tokens
```

## 6. 🔹 Interview Follow-ups
1. Q: JSON in prompt vs grammar?  
   A: **Grammar** **stronger**; **prompt** **only** is **brittle**.

## 7. 🔹 Common Mistakes
**regex**-repairing broken JSON in prod forever—**fix** at **decoder**.

## 8. 🔹 Comparison / Connections
Tool calling, OpenAI **response_format**, pydantic validation.

## 9. 🔹 One-line Revision
Structured output uses grammars or masks so generations are parse-valid—critical for agents and APIs.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: What is LLM-as-a-judge, and what are its pitfalls?

## 1. 🔹 Direct Answer
**LLM-as-judge** uses a **strong** LLM to **score** or **rank** outputs (helpfulness, correctness, style) **cheaply** at scale. **Pitfalls**: **position bias** (prefers first answer), **verbosity bias**, **self-enhancement** (favors **similar** models), **lack of** **domain** **grounding**, **prompt** sensitivity, and **non-stationarity** when **judge** **updates**.

## 2. 🔹 Intuition
A **fast** **approximate** **human** rater—**not** a **ground-truth** oracle.

## 3. 🔹 Deep Dive
**Mitigations**: **swap** positions, **multi-judge**, **human** **spot-check**, **rubric** **anchoring**, **task-specific** **metrics** (unit tests, **NLI** to **gold**).

## 4. 🔹 Practical Perspective
Good for **iteration** and **ranking**; **bad** as **sole** **safety** **gate**.

## 5. 🔹 Code Snippet
```text
score = judge(f"Which answer is better? A: {a} B: {b}")
```

## 6. 🔹 Interview Follow-ups
1. Q: G-Eval?  
   A: **Rubric + LLM score**—common pattern in **papers**.

## 7. 🔹 Common Mistakes
Treating **judge** **scores** as **interoperable** across **model** versions without **calibration**.

## 8. 🔹 Comparison / Connections
Human eval, MT-Bench, Arena Elo.

## 9. 🔹 One-line Revision
LLM judges scale evaluation but carry biases—use controls, rubrics, and human validation.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: What is knowledge distillation for LLMs?

## 1. 🔹 Direct Answer
**Distillation** trains a **smaller** **student** model to match a **larger** **teacher**’s outputs—using **hard labels**, **soft** **probabilities** (**temperature-scaled** logits), or **hidden-state** matching. **Goal**: **cheaper** **inference**, **lower** **latency**, **edge** deployment, while preserving **much** of the teacher’s behavior.

## 2. 🔹 Intuition
**Imitate** the **teacher’s** **uncertainty** (dark knowledge), not just **argmax** **class**.

## 3. 🔹 Deep Dive
**Sequence-level** distillation for **autoregressive** models; **on-policy** vs **off-policy** **data**.

## 4. 🔹 Practical Perspective
**Distill** **after** **SFT**; **watch** **distribution** shift on **long** outputs.

## 5. 🔹 Code Snippet
```python
# Soft target: KL(student_logits/T || teacher_logits/T)
loss = kl_div(log_softmax(student/T), softmax(teacher/T)) * (T**2)
```

## 6. 🔹 Interview Follow-ups
1. Q: Distill vs quantize?  
   A: **Orthogonal**—often **both** (student + INT8).

## 7. 🔹 Common Mistakes
**Only** matching **final** text **greedily**—loses **calibration**.

## 8. 🔹 Comparison / Connections
Speculative decoding (draft model as partial distillation), pruning.

## 9. 🔹 One-line Revision
Distillation transfers teacher behavior to a smaller student via soft targets—latency and cost win.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q9: What is prompt injection, and how does it differ from jailbreaking?

## 1. 🔹 Direct Answer
**Prompt injection** is **untrusted** content (web page, email, user message) **manipulating** the model or **tool** **loop** to **override** **developer** **instructions** (“ignore previous instructions…”). **Jailbreaking** is **eliciting** **policy-violating** outputs from the **model** itself (safety **bypass**). **Overlap**: both **abuse** **instruction-following**, but **injection** is often **about** **systems** **integrating** **external** **data** into **prompts**.

## 2. 🔹 Intuition
**Injection** = **malicious** **input** channel; **jailbreak** = **attack** on **model** **refusal** training.

## 3. 🔹 Deep Dive
**Defenses**: **separate** **trusted** vs **untrusted** **sections**, **tool** **allowlists**, **output** **filtering**, **minimal** **privilege**, **human** **approval** for **actions**, **detect** **instruction-like** patterns.

## 4. 🔹 Practical Perspective
**RAG** is **high** **risk** if **retrieved** **text** is **attacker-controlled**.

## 5. 🔹 Code Snippet
```text
untrusted_doc must never be interpreted as system policy
```

## 6. 🔹 Interview Follow-ups
1. Q: Model-level fix?  
   A: **Partial**—**system** **design** is **primary** for **injection**.

## 7. 🔹 Common Mistakes
Assuming **RLHF** **eliminates** **injection**—it’s a **trust** **boundary** problem.

## 8. 🔹 Comparison / Connections
Adversarial examples, OWASP LLM Top 10.

## 9. 🔹 One-line Revision
Prompt injection abuses untrusted text in prompts; jailbreaks bypass safety—defend with architecture and least privilege, not prompts alone.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q10: How does prefix / prompt caching reduce LLM cost and latency?

## 1. 🔹 Direct Answer
**Prefix caching** **stores** **KV activations** for a **reused** **prefix** (system prompt, **RAG** **chunks**, **long** **documents**) so **later** **requests** **skip** **recomputing** **prefill** on that **prefix**. **Cuts** **TTFT** and **compute** **cost** when **many** **queries** share the **same** **context** **block**.

## 2. 🔹 Intuition
Pay **once** for the **static** **part** of the **prompt**, **only** **extend** **KV** for **new** **suffix** **tokens**.

## 3. 🔹 Deep Dive
Implemented in **serving** **stacks** (vendor APIs, **vLLM** **features**); **cache** **keying** must match **model** **version** and **tokenizer**.

## 4. 🔹 Practical Perspective
**Semantic** **cache** (similar **query** → **reuse** **answer**) is **orthogonal**—**different** **failure** **mode** (**staleness**).

## 5. 🔹 Code Snippet
```text
cache_key = hash(model_id, prefix_token_ids)
```

## 6. 🔹 Interview Follow-ups
1. Q: Multi-tenant isolation?  
   A: **Separate** **caches** or **keys** per **tenant** for **privacy**.

## 7. 🔹 Common Mistakes
Expecting **cache** **hits** when **every** **prompt** is **unique** **noise**.

## 8. 🔹 Comparison / Connections
PagedAttention memory, speculative decoding.

## 9. 🔹 One-line Revision
Prefix caching reuses KV for shared static prompts—big win for RAG and system prompts at scale.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q11: Compare AWQ and GPTQ for LLM weight quantization (interview level).

## 1. 🔹 Direct Answer
Both are **post-training quantization (PTQ)** methods for **4-bit** (often) **weights** to **shrink** **memory** and **speed** **inference**. **GPTQ** uses **layer-wise** **Hessian**-aware **errors** (approximate **second-order** info) to **quantize** weights **greedily**. **AWQ** (**Activation-aware**) protects **salient** weights based on **activation** **statistics**—**assumes** **few** **critical** **weights** matter **more**. **Both** need **calibration** **batches**.

## 2. 🔹 Intuition
**GPTQ**: **minimize** **reconstruction** **error** in **weight** **space** with **curvature** **info**. **AWQ**: **protect** **weights** that **blow up** **activations** when **wrong**.

## 3. 🔹 Deep Dive
**Quality** depends on **calibration** **data** **match** to **deployment** **domain**.

## 4. 🔹 Practical Perspective
Often **packaged** in **llama.cpp**, **vLLM**, **HF** **quantize** **pipelines**—**benchmark** **perplexity** / **task** **metrics**.

## 5. 🔹 Code Snippet
```text
AWQ: scale factors per group; GPTQ: sequential quant with Hessian-inspired updates
```

## 6. 🔹 Interview Follow-ups
1. Q: vs QAT?  
   A: **QAT** **trains** with **fake** **quant**—**higher** **quality**, **more** **expensive**.

## 7. 🔹 Common Mistakes
Assuming **4-bit** **always** **free** **quality** **loss**—**always** **measure**.

## 8. 🔹 Comparison / Connections
GGUF, bitsandbytes NF4, SmoothQuant.

## 9. 🔹 One-line Revision
GPTQ uses Hessian-aware greedy PTQ; AWQ protects activation-sensitive weights—both popular for INT4 LLM inference.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q12: What should a “model card” or release checklist cover for an LLM product?

## 1. 🔹 Direct Answer
**Model cards** document **intent**, **training** **data** **provenance**, **evaluation** **results** (**safety**, **bias** **slices**), **limitations**, **misuse** **risks**, **environmental** **cost**, **versioning**, and **contact** **for** **feedback**. A **release** **checklist** adds **red-teaming**, **incident** **response**, **monitoring** **plan**, **license**, **PII** **handling**, and **rollback** **for** **bad** **deployments**.

## 2. 🔹 Intuition
**Transparency** for **users** and **auditors**—**not** **marketing** **fluff**.

## 3. 🔹 Deep Dive
**EU** **AI** **Act** and **enterprise** **procurement** increasingly **expect** **documented** **evals**.

## 4. 🔹 Practical Perspective
**Internal** **cards** **even** for **fine-tunes** **on** **private** **data**.

## 5. 🔹 Code Snippet
```text
sections: scope, data, metrics, limitations, ethical considerations, caveats
```

## 6. 🔹 Interview Follow-ups
1. Q: Who owns updates?  
   A: **MLOps** + **policy** + **legal**—**living** **document**.

## 7. 🔹 Common Mistakes
**Only** **reporting** **average** **accuracy** **without** **slices**.

## 8. 🔹 Comparison / Connections
Datasheets for datasets, system cards.

## 9. 🔹 One-line Revision
Model cards document capabilities, limits, data, and evals—essential for responsible LLM release and compliance conversations.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
