# Large Language Model

---

# Q1: What is a Large Language Model (LLM), and how does it work?

## 1. 🔹 Direct Answer
An **LLM** is a **large** **Transformer** (usually **decoder-only** or encoder-decoder) trained on **massive text** with **next-token prediction** (and variants). It models **P(token | context)**—**autoregressive** generation by sampling from learned distribution.

## 2. 🔹 Intuition
**Compress** internet-scale statistics into weights; **generate** by repeatedly asking “what word comes next?”

## 3. 🔹 Deep Dive
**Scale** (params, data, compute) improves **emergent** capabilities; **alignment** (RLHF) steers behavior.

## 4. 🔹 Practical Perspective
**APIs** vs **self-host** trade-offs; **eval** on **task** metrics not just perplexity.

## 5. 🔹 Code Snippet
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Emergent abilities? **A:** Sharp transitions with scale—debated mechanism.

## 7. 🔹 Common Mistakes
Treating LLM outputs as **reliable facts** without retrieval/verification.

## 8. 🔹 Comparison / Connections
SLMs, RAG, tool use.

## 9. 🔹 One-line Revision
LLMs are large autoregressive Transformers modeling token sequences—scale and alignment define behavior.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: What are Transformer Models and how do they work?

## 1. 🔹 Direct Answer
**Transformer** is **attention-only** architecture (no recurrence): **self-attention** lets each position **attend** to all others in one layer; **stacked** blocks + **FFN** + **residuals** + **norm**. **Encoder** (bidirectional), **decoder** (causal), or **encoder-decoder**.

## 2. 🔹 Intuition
**Relate** all words to all words in parallel—**mixing** information globally each layer.

## 3. 🔹 Deep Dive
**Scaled dot-product attention**; **multi-head** for multiple subspaces.

## 4. 🔹 Practical Perspective
**Dominant** in NLP/CV/modalities; **quadratic** cost in sequence length.

## 5. 🔹 Code Snippet
```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs RNN? **A:** Parallelization and long-range deps in O(1) layers vs O(T) sequential steps.

## 7. 🔹 Common Mistakes
Ignoring **O(L²)** memory for long contexts.

## 8. 🔹 Comparison / Connections
“Attention is all you need”, ViT, diffusion transformers.

## 9. 🔹 One-line Revision
Transformers stack self-attention and FFN blocks to mix token information in parallel—scalable with compute.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: What are the key components of a Transformer model?

## 1. 🔹 Direct Answer
**Multi-head self-attention**, **position encodings**, **feed-forward** sublayers (per-token MLP), **residual** connections, **layer normalization** (often **Pre-LN**), **causal mask** (decoder), **output** projection to vocab logits.

## 2. 🔹 Intuition
Attention = **routing** information; FFN = **process** per token after mixing.

## 3. 🔹 Deep Dive
**Optional**: cross-attention in encoder-decoder; **MoE** replaces dense FFN in some LLMs.

## 4. 🔹 Practical Perspective
**KV-cache** for fast autoregressive inference.

## 5. 🔹 Code Snippet
```text
x = x + Attn(LN(x)); x = x + FFN(LN(x))  # pre-LN variant
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Post-LN vs Pre-LN? **A:** Pre-LN often trains deeper models more stably.

## 7. 🔹 Common Mistakes
Forgetting **positional** info—without it permutation invariant.

## 8. 🔹 Comparison / Connections
Sub-layer ordering variants, SwiGLU FFN.

## 9. 🔹 One-line Revision
Core Transformer block: attention + FFN + residuals + norm + positional info.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q4: What is self-attention, and how does it work in Transformers?

## 1. 🔹 Direct Answer
**Self-attention** computes for each token a **weighted sum** of **values** from **all** tokens, weights from **compatibility** of **queries** with **keys**: **softmax(QKᵀ/√d)V**. Each position **aggregates** context-dependent mix.

## 2. 🔹 Intuition
Every word **asks** others “how relevant are you to me?”

## 3. 🔹 Deep Dive
**√d_k** scaling prevents softmax saturation; **multi-head** = parallel attention subspaces.

## 4. 🔹 Practical Perspective
**FlashAttention** reduces memory IO—critical for long contexts.

## 5. 🔹 Code Snippet
```python
scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
attn = torch.softmax(scores, dim=-1) @ V
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Linear attention? **A:** Kernel approximations to softmax—trade fidelity.

## 7. 🔹 Common Mistakes
Confusing self-attention with **cross-attention** (two sequences).

## 8. 🔹 Comparison / Connections
Graph attention, RNN hidden state mixing.

## 9. 🔹 One-line Revision
Self-attention mixes tokens via QKV softmax-weighted values—O(L²) per layer without recurrence.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: How does attention help capture long-range dependencies?

## 1. 🔹 Direct Answer
Each layer can **directly connect** any pair of positions in **one** attention hop (subject to depth stacking)—**path length O(1)** per layer vs RNN **O(T)** sequential steps. **Stacking** layers builds **multi-hop** reasoning.

## 2. 🔹 Intuition
No need to **pass** information step-by-step through time—**jump** links.

## 3. 🔹 Deep Dive
**Gradient paths** through residual + attention can be shorter than RNN BPTT.

## 4. 🔹 Practical Perspective
Still **limited** by **context window** length and **training** signal—not infinite magic.

## 5. 🔹 Code Snippet
```text
receptive field grows with layers; single layer connects all positions
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Lost in middle? **A:** Empirical degradation for facts in long context—RAG helps.

## 7. 🔹 Common Mistakes
Equating “can attend” with “will reliably retrieve” without **capacity** limits.

## 8. 🔹 Comparison / Connections
LSTM gating, dilated convolutions.

## 9. 🔹 One-line Revision
Attention provides direct pairwise connectivity per layer—eases long-range deps vs chained RNN steps.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q6: What does each Transformer block learn?

## 1. 🔹 Direct Answer
**Not** one crisp human role per layer—**emergent** **hierarchical** features: lower layers often **syntax/local** patterns; mid **semantic**; upper **task-specific** / **world knowledge** traces—**highly** **model** and **data** dependent. **Probing** and **mechanistic** interpretability study this.

## 2. 🔹 Intuition
Like CNNs: **rough** progression from local to global—**but** attention is dynamic.

## 3. 🔹 Deep Dive
**Residual** stream carries **linear** subspace hypotheses (logit lens line of work).

## 4. 🔹 Practical Perspective
Avoid **strong** universal claims—cite **specific** papers/findings cautiously.

## 5. 🔹 Code Snippet
```text
N/A qualitative
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Circuit discovery? **A:** Attributing heads to induction/copying—active research.

## 7. 🔹 Common Mistakes
“Layer 7 does only syntax”—oversimplified.

## 8. 🔹 Comparison / Connections
Representation learning, probing classifiers.

## 9. 🔹 One-line Revision
Transformer layers build increasingly abstract representations—exact roles are empirical, not fixed.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q7: What is pre-training vs fine-tuning in LLMs?

## 1. 🔹 Direct Answer
**Pre-training**: **self-supervised** next-token (or masked) LM on **broad** corpora—**general** language modeling. **Fine-tuning**: **supervised** or **preference** data to **specialize** tasks, styles, safety—**smaller** LR, fewer steps; **PEFT** (LoRA) updates **subsets**.

## 2. 🔹 Intuition
Pretrain learns **language + world priors**; fine-tune **steers** behavior.

## 3. 🔹 Deep Dive
**Continued pretraining** for domain; **instruction** tuning; **RLHF** as alignment stage.

## 4. 🔹 Practical Perspective
**Catastrophic forgetting** risk—**KL** to reference model in RLHF mitigates.

## 5. 🔹 Code Snippet
```python
# HF Trainer fine-tuning sketch
trainer = Trainer(model=model, train_dataset=ds, args=args)
trainer.train()
```

## 6. 🔹 Interview Follow-ups
1. **Q:** LoRA? **A:** Low-rank adapters on attention weights—cheap fine-tune.

## 7. 🔹 Common Mistakes
Fine-tuning on **tiny** data without regularization—overfits.

## 8. 🔹 Comparison / Connections
Prompting as **soft** adaptation without weight updates.

## 9. 🔹 One-line Revision
Pretrain broad LM; fine-tune or align on task-specific or preference data—often with efficient adapters.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: What are some challenges in training LLMs?

## 1. 🔹 Direct Answer
**Compute/data** scale, **stability** (loss spikes, **numerics**), **distributed** training complexity, **hallucination**, **bias/toxicity**, **evaluation**, **alignment**, **long context** memory, **environmental** cost, **data** rights and **quality**.

## 2. 🔹 Intuition
Scaling laws help but **don’t** solve alignment or truthfulness automatically.

## 3. 🔹 Deep Dive
**Mixed precision**, **gradient checkpointing**, **MoE** routing stability.

## 4. 🔹 Practical Perspective
**Data filtering** for toxicity and PII; **reproducibility** hard at scale.

## 5. 🔹 Code Snippet
```text
3D parallel: data + tensor + pipeline parallelism
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Data contamination? **A:** Benchmark leakage from web—need decontamination.

## 7. 🔹 Common Mistakes
Attributing all failures to “need more parameters” vs data/process.

## 8. 🔹 Comparison / Connections
Scaling laws, Chinchilla optimal compute.

## 9. 🔹 One-line Revision
LLM training challenges span systems scale, data quality, safety, eval, and alignment—not just loss curves.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q9: What is zero-shot learning in the context of LLMs?

## 1. 🔹 Direct Answer
**Zero-shot**: model performs task **without** task-specific fine-tune examples—only **natural language instruction** in prompt. Enabled by **broad pretraining** + **instruction tuning** / **RLHF**.

## 2. 🔹 Intuition
“Follow English instructions” transfers to **new** tasks if similar patterns existed in training.

## 3. 🔹 Deep Dive
**Not** classical zero-shot vision sense exactly—**in-context** learning blurs with **few-shot**.

## 4. 🔹 Practical Perspective
**Reliability** varies—**eval** on **held-out** task suite (MMLU, etc.).

## 5. 🔹 Code Snippet
```text
prompt = "Classify sentiment: ... Answer with positive/negative."
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs few-shot? **A:** Few-shot adds exemplars in context—often boosts accuracy.

## 7. 🔹 Common Mistakes
Expecting robust zero-shot on **niche** structured tasks.

## 8. 🔹 Comparison / Connections
In-context learning, meta-learning framing.

## 9. 🔹 One-line Revision
Zero-shot LLM use relies on instructions alone—strong with scale and alignment, not infallible.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: How do you handle bias and fairness in LLMs?

## 1. 🔹 Direct Answer
**Data** curation/deduplication, **filtering** toxic content, **RLHF/DPO** with **fairness** criteria, **red-teaming**, **eval** by **demographic** slices, **system** prompts and **refusal** policies, **RAG** with **trusted** sources—**no** silver bullet.

## 2. 🔹 Intuition
Web text encodes **biases**—mitigation is **iterative** measurement + training + product guardrails.

## 3. 🔹 Deep Dive
**Representation** harms vs **allocative** harms; **tension** between **helpfulness** and **refusal**.

## 4. 🔹 Practical Perspective
**User-specific** controls; **monitoring** in deployment.

## 5. 🔹 Code Snippet
```text
eval: fairness slices on BBQ, Winogender-style probes
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Debias embeddings? **A:** Hard—often hurts performance; prefer holistic alignment.

## 7. 🔹 Common Mistakes
Claiming “debias once” works—distribution shift breaks guarantees.

## 8. 🔹 Comparison / Connections
Responsible AI, constitutional AI.

## 9. 🔹 One-line Revision
Bias mitigation combines data, alignment training, eval slices, and runtime policies—continuous process.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q11: What are some real-world applications of LLMs in business and tech?

## 1. 🔹 Direct Answer
**Assistants**, **coding** copilots, **customer support**, **search** augmentation (**RAG**), **document** Q&A, **content** drafting (with review), **translation**, **summarization**, **data** extraction—**always** with **human** oversight for **high-stakes**.

## 2. 🔹 Intuition
Anything needing **language** + **scale**—pair with **tools** for **facts** and **actions**.

## 3. 🔹 Deep Dive
**Enterprise** concerns: **PII**, **IP**, **latency**, **cost** per token.

## 4. 🔹 Practical Perspective
**Vertical** solutions beat generic chat for ROI.

## 5. 🔹 Code Snippet
```text
architecture: LLM + retriever + APIs + policy layer
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When not LLM? **A:** Strict deterministic logic, tiny models suffice, regulated latency.

## 7. 🔹 Common Mistakes
LLM for **math** without tool use—hallucination risk.

## 8. 🔹 Comparison / Connections
Agents, workflow automation.

## 9. 🔹 One-line Revision
LLMs power language-heavy products—usually as copilots with retrieval, tools, and governance.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: How does the Transformer architecture improve LLM performance over RNNs?

## 1. 🔹 Direct Answer
**Parallelism** across sequence for training; **shorter** paths for **long-range** dependencies; **scalable** with **data/compute** (scaling laws). RNNs **sequential** and **vanishing gradient** issues for long deps—Transformers **dominate** at scale.

## 2. 🔹 Intuition
RNN reads like a tape; Transformer **jumps** everywhere each layer.

## 3. 🔹 Deep Dive
**Self-attention** content-based routing vs fixed recurrence.

## 4. 🔹 Practical Perspective
**Inference** still costly—**KV cache**, **speculative decoding**.

## 5. 🔹 Code Snippet
```text
train: parallel matmuls ; RNN: sequential steps per layer
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When RNN still? **A:** Extreme memory/latency constraints, streaming on tiny devices—niche.

## 7. 🔹 Common Mistakes
Saying RNNs always worse on **tiny** data—sometimes simpler models suffice.

## 8. 🔹 Comparison / Connections
SSMs (Mamba) for long seq efficiency.

## 9. 🔹 One-line Revision
Transformers train faster in parallel and model long-range deps more directly—key to large-scale LLM success.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: Explain the attention mechanism in LLMs.

## 1. 🔹 Direct Answer
**Attention** computes **compatibility scores** between **query** (current focus) and **keys** (all positions), **softmax** to weights, **sum values**—**dynamic**, **input-dependent** routing. **Multi-head** runs multiple **parallel** attentions.

## 2. 🔹 Intuition
Soft **lookup** over the sequence weighted by relevance.

## 3. 🔹 Deep Dive
**Decoder** uses **causal mask** to prevent seeing future tokens.

## 4. 🔹 Practical Perspective
**FlashAttention** kernels fuse operations for speed/memory.

## 5. 🔹 Code Snippet
```text
weights = softmax(QK^T / sqrt(d) + mask); out = weights V
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why softmax? **A:** Nonnegative weights summing to 1—differentiable routing; alternatives exist.

## 7. 🔹 Common Mistakes
Skipping **scale** factor importance.

## 8. 🔹 Comparison / Connections
Linear attention, kernel attention.

## 9. 🔹 One-line Revision
Attention = softmax-normalized QK scores mixing V vectors—core mixing operation of Transformers.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: What are multi-head attention mechanisms? Why use multiple attention heads?

## 1. 🔹 Direct Answer
**Multi-head attention** runs **H** independent attention ops in parallel on **projected** **Q,K,V** subspaces then **concatenate** and **project**—allows attending to **different** relationship types (**syntax vs coreference**) **simultaneously**.

## 2. 🔹 Intuition
One head might track **subject-verb**, another **anaphora**—**specialization**.

## 3. 🔹 Deep Dive
Each head has **lower** dim **d_h = d_model / H** to keep total params similar.

## 4. 🔹 Practical Perspective
**Head pruning** research for efficiency.

## 5. 🔹 Code Snippet
```text
MultiHead(Q,K,V) = Concat(head_1..head_H) W_O
```

## 6. 🔹 Interview Follow-ups
1. **Q:** GQA/MQA? **A:** Share K/V across heads to reduce KV cache—used in inference-optimized LLMs.

## 7. 🔹 Common Mistakes
Thinking heads are **named** roles by design—emergent.

## 8. 🔹 Comparison / Connections
Ensemble within layer.

## 9. 🔹 One-line Revision
Multi-head attention runs parallel attention in subspaces then merges—increases representational capacity.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: Explain Query (Q), Key (K), and Value (V) in attention.

## 1. 🔹 Direct Answer
**Q,K,V** are **linear projections** of input embeddings: **compatibility** **score_ij = q_i · k_j** determines how much **value v_j** contributes to output position **i**. **Interpretation**: **query** “what I’m looking for,” **key** “what I offer as label,” **value** “content I provide if selected.”

## 2. 🔹 Intuition
Like **database** retrieval: query matches keys; fetch **values**.

## 3. 🔹 Deep Dive
**Learned** projections—**not** hand-designed roles; **rotary** (RoPE) encodes position in Q,K.

## 4. 🔹 Practical Perspective
**Grouped-query** shares K/V heads to save memory.

## 5. 🔹 Code Snippet
```python
Q, K, V = x @ W_q, x @ W_k, x @ W_v
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Cross-attention? **A:** Q from decoder, K,V from encoder—seq2seq.

## 7. 🔹 Common Mistakes
Literal database analogy too far—keys/values are **learned** features.

## 8. 🔹 Comparison / Connections
Memories, retrieval-augmented models.

## 9. 🔹 One-line Revision
Q and K score relevance; V supplies content aggregated into output—learned linear maps of hidden states.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q16: Tokenization in Large Language Models (LLMs).

## 1. 🔹 Direct Answer
**Tokenization** maps **text → token IDs** via **subword** algorithms (**BPE**, **WordPiece**, **SentencePiece**) to balance **vocab size**, **OOV** handling, and **efficiency**. **Imperfect**—**spaces**, **numbers**, **typos** split oddly; affects **performance**.

## 2. 🔹 Intuition
LLMs see **tokens**, not characters—**compression** of text stream.

## 3. 🔹 Deep Dive
**BPE** merges frequent pairs; **SentencePiece** is **unigram** language model based.

## 4. 🔹 Practical Perspective
**Same tokenizer** for train/infer; **count tokens** for pricing/latency.

## 5. 🔹 Code Snippet
```python
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
ids = tok.encode("hello world", add_special_tokens=True)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Adversarial tokens? **A:** Unicode tricks—input sanitation.

## 7. 🔹 Common Mistakes
Ignoring **max length** truncation effects on meaning.

## 8. 🔹 Comparison / Connections
Byte-level BPE for robustness.

## 9. 🔹 One-line Revision
Subword tokenization defines the model’s input alphabet—impacts length, robustness, and multilingual behavior.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: What is subword tokenization?

## 1. 🔹 Direct Answer
**Subword** splits words into **smaller** units (prefixes, suffixes, frequent fragments)—**open vocabulary** with **finite** vocab. Reduces **OOV** vs word-level, **more efficient** than char-level for long texts.

## 2. 🔹 Intuition
“**un**” + “**happiness**” style pieces—reuse fragments across words.

## 3. 🔹 Deep Dive
**BPE** merge rules from corpus statistics.

## 4. 🔹 Practical Perspective
**Trade-off**: very aggressive splitting → **longer** sequences.

## 5. 🔹 Code Snippet
```text
"unhappiness" -> ["un", "happiness"] or finer depending on vocab
```

## 6. 🔹 Interview Follow-ups
1. **Q:** vs word-level? **A:** Word OOV; char too long sequences.

## 7. 🔹 Common Mistakes
Assuming tokenizer is **language agnostic**—English biases exist.

## 8. 🔹 Comparison / Connections
Morphologically rich languages need good subword handling.

## 9. 🔹 One-line Revision
Subword tokenization merges stats-driven pieces—balances vocab size and coverage.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q18: What is BPE (Byte Pair Encoding) in LLMs?

## 1. 🔹 Direct Answer
**BPE** starts with **bytes/chars** or symbols and **iteratively merges** the **most frequent adjacent pair** in corpus until **vocab** budget—creates **subword** tokens. **GPT-2/3**, many LLMs use BPE variants.

## 2. 🔹 Intuition
**Compress** text by building a **dictionary** of common chunks.

## 3. 🔹 Deep Dive
**Training** tokenizer on corpus; **merges** stored as **rules** for deterministic encode.

## 4. 🔹 Practical Perspective
**tiktoken** fast library for OpenAI token counting.

## 5. 🔹 Code Snippet
```text
merge rules: ('t','h')->'th', ...
```

## 6. 🔹 Interview Follow-ups
1. **Q:** WordPiece difference? **A:** Often likelihood-based merges + ## continuations in BERT.

## 7. 🔹 Common Mistakes
Thinking BPE is **byte-level** always—often hybrid with byte fallback.

## 8. 🔹 Comparison / Connections
SentencePiece unigram model.

## 9. 🔹 One-line Revision
BPE builds subword vocab via iterative frequency merges—standard LLM tokenizer approach.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: What is positional embedding in LLMs?

## 1. 🔹 Direct Answer
Attention is **permutation invariant** without position info—**positional embeddings** inject **order**: **sinusoidal** (original), **learned** absolute, **relative** biases, **RoPE** (rotate Q,K by position)—**critical** for language.

## 2. 🔹 Intuition
Tell model **where** in sequence each token sits.

## 3. 🔹 Deep Dive
**RoPE** prevalent in modern LLMs—**extrapolation** challenges beyond train length—**scaling** methods (NTK, YaRN).

## 4. 🔹 Practical Perspective
**ALiBi** alternative with distance bias in attention scores.

## 5. 🔹 Code Snippet
```text
x = token_embed + pos_embed  # absolute case
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Length extrapolation? **A:** Hard—position methods differ in generalization.

## 7. 🔹 Common Mistakes
Forgetting **relative** vs **absolute** trade-offs.

## 8. 🔹 Comparison / Connections
Sinusoidal vs RoPE vs ALiBi.

## 9. 🔹 One-line Revision
Positional embeddings encode token order—RoPE/ALiBi common in modern LLMs with extrapolation nuances.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q20: What is temperature in the context of LLMs?

## 1. 🔹 Direct Answer
**Temperature T** scales logits **before softmax**: **p ∝ exp(z/T)**. **T>1** → **softer** distribution (**more random**); **T<1** → **sharper** (**more greedy**). **T→0** approaches **argmax**.

## 2. 🔹 Intuition
Heat shakes up probabilities—**creativity** vs **determinism** knob.

## 3. 🔹 Deep Dive
**Sampling** uses T with **top-k/top-p** often.

## 4. 🔹 Practical Perspective
**Calibration** not same as temperature for **truthfulness**—often **separate** tuning.

## 5. 🔹 Code Snippet
```python
logits = logits / temperature
probs = torch.softmax(logits, dim=-1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Temperature scaling for calibration? **A:** Post-hoc **T** on logits to match confidence—different use of name.

## 7. 🔹 Common Mistakes
High T for tasks needing **faithfulness**—increases hallucination risk.

## 8. 🔹 Comparison / Connections
Top-p, min-p sampling.

## 9. 🔹 One-line Revision
Sampling temperature sharpens or flattens the next-token distribution—controls randomness vs determinism.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: What is causal masking?

## 1. 🔹 Direct Answer
In **decoder** self-attention, **causal mask** sets **scores to −∞** for **j > i** so position **i** **cannot attend** to **future** tokens—ensures **autoregressive** **valid** **factorization** **P(x)** = ∏_i P(x_i | x_1, …, x_{i−1}).

## 2. 🔹 Intuition
Prevent **cheating** by seeing the future during **training** of generative model.

## 3. 🔹 Deep Dive
Implemented as **additive** mask to logits pre-softmax.

## 4. 🔹 Practical Perspective
**Bidirectional** encoder (BERT) **no** causal mask—uses **MLM** instead.

## 5. 🔹 Code Snippet
```python
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Prefix LM? **A:** Hybrid attention regions—prefix bidirectional sometimes.

## 7. 🔹 Common Mistakes
Applying causal mask in **encoder** improperly for BERT-style.

## 8. 🔹 Comparison / Connections
KV-cache assumes causal generation order.

## 9. 🔹 One-line Revision
Causal masking blocks attention to future positions—defines autoregressive LM training.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q22: What are skip connections?

## 1. 🔹 Direct Answer
**Skip connections** (**residuals**) add **input x** to **sublayer output**: **y = x + F(x)**—ease **gradient flow**, enable **very deep** networks, **identity** baseline if **F** learns zero.

## 2. 🔹 Intuition
Let each layer **refine** rather than **recompute** full representation from scratch.

## 3. 🔹 Deep Dive
**Pre-LN** vs **Post-LN** ordering with residuals.

## 4. 🔹 Practical Perspective
Standard in **Transformers**, **ResNets**.

## 5. 🔹 Code Snippet
```python
x = x + sublayer(x)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Highway networks? **A:** Gated residuals—less common now.

## 7. 🔹 Common Mistakes
Confusing with **dense** connections (DenseNet).

## 8. 🔹 Comparison / Connections
RNN skip connections, wavelet residual.

## 9. 🔹 One-line Revision
Residual skip connections add sublayer outputs to inputs—stabilize deep training.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q23: What is normalization?

## 1. 🔹 Direct Answer
In LLMs, **LayerNorm** normalizes **per token** across features: **(x−μ)/σ** with **learned** **γ,β**—**stabilizes** activations and **helps** optimization. **RMSNorm** variant drops mean centering—used in LLaMA etc.

## 2. 🔹 Intuition
Keep **scale** of vectors controlled layer to layer.

## 3. 🔹 Deep Dive
**Pre-LN** places norm **before** sublayer—common in GPT-style.

## 4. 🔹 Practical Perspective
**BatchNorm** less common in Transformers (sequence/batch issues).

## 5. 🔹 Code Snippet
```python
import torch.nn as nn
nn.LayerNorm(d_model)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Why not BatchNorm in Transformers? **A:** Variable seq, small batch issues—LayerNorm per token stable.

## 7. 🔹 Common Mistakes
Confusing **weight decay** with **normalization**.

## 8. 🔹 Comparison / Connections
BatchNorm, RMSNorm.

## 9. 🔹 One-line Revision
LayerNorm/RMSNorm stabilizes Transformer activations per token—pre-LN common for deep stacks.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q24: What is dropout, and how is it applied in LLMs?

## 1. 🔹 Direct Answer
**Dropout** randomly **zeros** activations during training—**regularization**. In LLMs applied to **FFN**, sometimes **attention** (varies); **disabled** at inference. Some models use **stochastic depth** variants.

## 2. 🔹 Intuition
Prevent **co-adaptation**—forces **redundant** robust paths.

## 3. 🔹 Deep Dive
**Attention dropout** on **attention weights** (not always used in large LMs—depends).

## 4. 🔹 Practical Perspective
**Small** models benefit more; **large** may use **lower** dropout rates.

## 5. 🔹 Code Snippet
```python
nn.Dropout(p=0.1)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Dropout on residual? **A:** Usually after sublayer before adding—architecture dependent.

## 7. 🔹 Common Mistakes
Leaving dropout **on** during eval.

## 8. 🔹 Comparison / Connections
DropPath, L2 reg.

## 9. 🔹 One-line Revision
Dropout regularizes Transformers by stochastic unit removal during training—toggle off in inference.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q25: Why does Attention use Softmax?

## 1. 🔹 Direct Answer
**Softmax** turns **compatibility scores** into a **convex combination** over positions—**nonnegative weights summing to 1**—interpretable **attention distribution**, **differentiable**, **stable** with **log-sum-exp**. Alternatives exist (**sparsemax**, **sigmoid** attention) but softmax is standard **smooth** routing.

## 2. 🔹 Intuition
Convert scores to **probabilities** of “how much to read” from each position.

## 3. 🔹 Deep Dive
**Attention sink** phenomenon in softmax—some heads attend strongly to specific tokens.

## 4. 🔹 Practical Perspective
**FlashAttention** computes softmax in tiles fused with matmul.

## 5. 🔹 Code Snippet
```text
α = softmax(QK^T / sqrt(d))
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Linear attention kernels? **A:** Replace softmax with feature maps for O(L)—approximate.

## 7. 🔹 Common Mistakes
Claiming softmax is **only** possible choice.

## 8. 🔹 Comparison / Connections
Gumbel-softmax for discrete relaxations.

## 9. 🔹 One-line Revision
Softmax turns attention scores into a differentiable probability mix over values—standard smooth routing.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q26: What does a vector database (Vector DB) store for LLM usage?

## 1. 🔹 Direct Answer
**Embeddings** (dense vectors) of **documents/chunks** + **metadata** (ids, source, ACLs) for **similarity search** (**ANN**) to support **RAG**—retrieve **relevant context** given **query embedding**. Not raw text alone—**indexes** like **HNSW**, **IVF**.

## 2. 🔹 Intuition
**Semantic** search: “find passages **like** this question.”

## 3. 🔹 Deep Dive
**Hybrid** search combines **BM25** + vectors; **rerankers** refine top-k.

## 4. 🔹 Practical Perspective
**Freshness**, **deletion**, **multi-tenant** isolation—ops concerns.

## 5. 🔹 Code Snippet
```text
store: (id, embedding, metadata) ; query: nearest_neighbors(q_emb, k)
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Which embedding model? **A:** Same model for index/query or aligned spaces.

## 7. 🔹 Common Mistakes
Expecting vector DB to be **truth**—still need **grounding** and **citation**.

## 8. 🔹 Comparison / Connections
Elasticsearch dense_vector, FAISS, Pinecone.

## 9. 🔹 One-line Revision
Vector DBs store embeddings + metadata for fast ANN retrieval powering RAG.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q27: How do you improve inference speed in production LLM deployments?

## 1. 🔹 Direct Answer
**Quantization** (INT8/INT4), **FlashAttention**, **KV-cache** + **paged** attention (**vLLM**), **continuous batching**, **speculative decoding**, **smaller** models / **distillation**, **tensor parallel**, **better hardware**, **caching** repeated prefixes, **trim** prompts, **early exit** (rare).

## 2. 🔹 Intuition
Attack **memory bandwidth** and **autoregressive** serial steps.

## 3. 🔹 Deep Dive
**Batching** improves throughput; **latency** still bound by **decode** steps.

## 4. 🔹 Practical Perspective
**TTFT** vs **tokens/sec** SLOs differ—optimize accordingly.

## 5. 🔹 Code Snippet
```text
TensorRT-LLM, vLLM, TGI stacks
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Speculative decoding? **A:** Draft small model proposes tokens; big model verifies.

## 7. 🔹 Common Mistakes
Only measuring **prefill** not **decode** latency.

## 8. 🔹 Comparison / Connections
See ai infrastructure notes in repo.

## 9. 🔹 One-line Revision
Speed up LLM inference with quant, KV paging, batching, speculative decoding, and serving-optimized runtimes.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q28: What is the context window in LLM?

## 1. 🔹 Direct Answer
**Context window** is **maximum tokens** the model can **attend** to in one forward pass—**working memory** limit. Longer windows enable **more** document context but **cost** **O(L²)** attention (naive) and **more** **KV memory**.

## 2. 🔹 Intuition
Like RAM size—beyond it, must **truncate**, **summarize**, or **retrieve**.

## 3. 🔹 Deep Dive
**Position extrapolation** methods extend beyond train length (**RoPE scaling**).

## 4. 🔹 Practical Perspective
**RAG** for effectively infinite knowledge without infinite window.

## 5. 🔹 Code Snippet
```python
assert len(token_ids) <= model.config.max_position_embeddings
```

## 6. 🔹 Interview Follow-ups
1. **Q:** Long context models? **A:** 100k+ with approx attention / ring attention—systems heavy.

## 7. 🔹 Common Mistakes
Assuming model **uses** all context equally—**lost in middle** effect.

## 8. 🔹 Comparison / Connections
Sliding window, recurrent memory.

## 9. 🔹 One-line Revision
Context window caps tokens per forward pass—drives memory, cost, and design of RAG vs long-context finetuning.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q29: Explain Prompting, Retrieval-Augmented Generation (RAG), and Fine-Tuning.

## 1. 🔹 Direct Answer
**Prompting**: steer **frozen** model with **instructions/examples** in context—**fast**, **no** weight updates. **RAG**: **retrieve** relevant docs, **condition** generation—**updates knowledge** without retraining, **grounding**. **Fine-tuning**: **update weights** on domain/task data—**strongest behavioral** specialization, **costly**.

## 2. 🔹 Intuition
Prompt = quick steering; RAG = **open book** exam; fine-tune = **study** new material into weights.

## 3. 🔹 Deep Dive
Trade-offs: **latency** (RAG retrieval), **staleness** (FT still needs refresh), **hallucination** risk without RAG.

## 4. 🔹 Practical Perspective
Often **combine**: FT for **tone/format**, RAG for **facts**.

## 5. 🔹 Code Snippet
```text
RAG: query -> embed -> retrieve(k) -> concat -> LLM
```

## 6. 🔹 Interview Follow-ups
1. **Q:** When FT over RAG? **A:** Style, tool formats, private patterns not in retriever index.

## 7. 🔹 Common Mistakes
FT on **facts** that change daily—RAG/update pipeline better.

## 8. 🔹 Comparison / Connections
Tool use, memory, distillation.

## 9. 🔹 One-line Revision
Prompt for zero-shot control; RAG for dynamic knowledge; fine-tune for durable task-specific behavior—often hybridize.

## 10. 🔹 Difficulty Tag
🟡 Medium

---
