# LLM fundamentals — the “DevOps brain” edition

You already ship systems. So we’ll talk about LLMs like systems: **interfaces, failure modes, cost, latency, and guardrails**.

**How to use this file**
- **Direct answer**: what you say in 10 seconds.
- **Azure/DevOps bridge**: the mapping you already trust.
- **Analogy**: fashion / Mumbai Indians / classic romance + Gulzar-style context.
- **Mini prompt**: quick interactive break.

> **North star:** An LLM is a **probabilistic next-token service**. Production GenAI is mostly making that service behave.

---

# Q1: What are foundation models, and how have they changed AI engineering?
- **Direct answer:** A **foundation model** is a huge self-supervised model trained on broad data that you **adapt** (prompt/RAG/fine-tune) for many tasks.
- **Azure/DevOps bridge:** It’s “build on Azure” vs “build your own datacenter.” You spend less time on core modeling and more on **integration + policy + observability**.
- **Analogy:** Capsule wardrobe—same staples, different styling.
- **Mini pop quiz:** Name a non-text foundation model. → CLIP, Stable Diffusion, Wav2Vec.

---

# Q2: What is a Large Language Model (LLM), and how does it work?
- **Direct answer:** An LLM generates text by predicting the **next token** repeatedly (autoregressive decoding), typically using Transformers.
- **Mechanics (one breath):** prompt → tokenize → Transformer → logits → softmax → decode → append token → repeat.
- **Mini prompt:** `temperature=0` gives you what vibe? → deterministic / strict.

---

# Q3: What are Transformer Models and how do they work?
- **Direct answer:** Transformers use **self-attention** to connect tokens directly, enabling parallel training and long-range context.
- **Ghazal hook:** Like a Gulzar line—meaning depends on what came before and what surrounds it. Attention is “who is this word in conversation with?”
- **Trade-off:** naive attention has ugly scaling with long context.

---

# Q4: What are the key components of a Transformer model?
- **Direct answer:** embeddings + position signal + attention + FFN + residuals + LayerNorm.
- **DevOps bridge:** residual paths are the “safe rollback route” for gradients.

---

# Q5: What is tokenisation in LLMs?
- **Direct answer:** Tokenization slices text into tokens (often subwords) and maps them to IDs.
- **Practical truths:** **1 token ≠ 1 word**; tokenizers can create a **token tax** for some languages.

---

# Q6: Explain BPE (Byte Pair Encoding).
- **Direct answer:** BPE iteratively merges frequent adjacent symbols until a fixed vocabulary is built (often byte-level).
- **Fashion analogy:** threads → panels → outfit.

---

# Q7: Explain WordPiece and SentencePiece.
- **Direct answer:** WordPiece uses likelihood-based merges; SentencePiece is language-agnostic and treats spaces as characters.
- **Mini prompt:** Which is friendlier to languages without spaces? → SentencePiece.

---

# Q8: What is positional encoding, and why is it needed in Transformers?
- **Direct answer:** Attention doesn’t encode order by itself, so we inject position info (absolute or relative-ish like RoPE/ALiBi).

---

# Q9: What is causal masking?
- **Direct answer:** A training-time mask that prevents decoder models from attending to future tokens.
- **DevOps bridge:** blocks leakage at the mechanism level (like forbidding prod secrets in PR pipelines).

---

# Q10: What is self-attention, and how does it work in Transformers?
- **Direct answer:**

\[
Attention(Q,K,V)=softmax\left(rac{QK^T}{\sqrt{d_k}}
ight)V
\]

- **Why \(\sqrt{d_k}\)?** Keeps dot products from blowing up so gradients don’t vanish.

---

# Q11: Explain Query (Q), Key (K), and Value (V) in attention.
- **Direct answer:** Q asks, K advertises, V carries payload. Dot(Q,K) gives weights; weights mix V.

---

# Q12: What are multi-head attention mechanisms? Why use multiple attention heads?
- **Direct answer:** Multiple heads learn different relationship patterns in parallel.
- **MI analogy:** multiple fielders for different shots.

---

# Q13: What is the context window in LLMs, and why does it matter?
- **Direct answer:** Max tokens (input + output) per call; outside the window = forgotten.
- **DevOps bridge:** context window ≈ RAM; fine-tuning ≈ installed software; RAG ≈ read-through cache.

---

# Q14: What is temperature in the context of LLMs?
- **Direct answer:** Temperature rescales logits before softmax:

\[
p_i=rac{e^{x_i/T}}{\sum_j e^{x_j/T}}
\]

- Low T = deterministic; high T = diverse + more hallucination risk.

---

# Q15: Explain Top-p (nucleus) sampling and Top-k sampling.
- **Direct answer:** Top-k keeps the k most likely tokens; Top-p keeps the smallest set with cumulative probability ≥ p.

---

# Q16: What are logits, and how are they used in text generation?
- **Direct answer:** Logits are raw scores per vocab token; softmax turns them into probabilities; decoding chooses/samples a token.

---

# Q17: What are skip connections (residual connections) in Transformers?
- **Direct answer:** Add input back to sublayer output; improves stability and gradient flow.

---

# Q18: Open-source vs closed-source LLMs? When choose which?
- **Direct answer:** Closed = strongest, fastest adoption; Open = control/customization/on-prem, but you own ops.
- **Azure lens:** closed feels like managed PaaS; open feels like AKS.

---

# Q19: Encoder-only vs decoder-only vs encoder-decoder Transformer architectures?
- **Direct answer:** Encoder-only for understanding/embeddings; decoder-only for generation; encoder-decoder for seq2seq transforms.

---

# Q20: What is KV cache, and how does it speed up inference?
- **Direct answer:** Cache past K/V tensors so decoding doesn’t recompute history every token.
- **Infra reality:** it’s a latency win and a VRAM bill.

---

# Q21: Autoregressive vs masked language modeling.
- **Direct answer:** CLM predicts next token from past; MLM fills blanks using both sides.

---

# Q22: What is model distillation?
- **Direct answer:** Train a smaller student to mimic a larger teacher (often via soft targets/logits).

---

# Q23: What is Mixture of Experts (MoE)?
- **Direct answer:** Route tokens to a subset of expert networks; huge capacity with sparse compute.

---

# Q24: Dense vs sparse models?
- **Direct answer:** Dense runs all params each step; sparse conditionally activates subsets.

---

# Q25: What is Flash Attention?
- **Direct answer:** GPU-kernel optimization that reduces attention memory IO (tiling + fused ops) without changing math.

---

# Q26: What is Grouped-Query Attention (GQA)?
- **Direct answer:** Share K/V across groups of query heads to shrink KV cache with little quality loss.

---

# Q27: How does Rotary Position Embedding (RoPE) work?
- **Direct answer:** Rotate Q/K vectors by position so dot products encode relative distance; helps length extrapolation.

---

# Q28: LLM ignores instructions. How do you enforce structured output?
- **Direct answer:** Combine system constraints + few-shot + constrained decoding/structured outputs (schema-driven).

---

# Q29: Context window limit on long documents. What do you do?
- **Direct answer:** Use RAG, map-reduce summarization, and/or agentic search tools.

---

# Q30: Make it say “I don’t know.”
- **Direct answer:** Explicit escape hatch + grounding (retrieval) + low temperature + optional confidence gates.

---

# Q31: Code generation keeps adding explanations. What do you do?
- **Direct answer:** Tighten format contract (“only code”), few-shot, stop sequences/structured output, lower temperature.

---

# Q32: Explain the “Lost in the Middle” phenomenon.
- **Direct answer:** In long prompts, models use start/end info better than middle; recall can be U-shaped.

---

# Q33: What is quantization?
- **Direct answer:** Reduce weight/activation precision (FP16 → INT8/INT4) to cut memory and bandwidth.

---

# Q34: How do you choose between 8-bit, 4-bit, and 2-bit quantization?
- **Direct answer:** 8-bit ≈ near-lossless; 4-bit ≈ best local/cheap; 2-bit often collapses quality.

---

# Q35: AWQ vs GPTQ?
- **Direct answer:** GPTQ compensates quantization error layer-wise; AWQ protects salient weights via activation profiling.

---

# Q36: Explain PEFT.
- **Direct answer:** Fine-tune by training a tiny subset (adapters/LoRA/prefix) while freezing the base.

---

# Q37: How does LoRA work?
- **Direct answer:** Learn low-rank matrices \(A,B\) so \(\Delta Wpprox BA\), added to frozen weights.

---

# Q38: What is QLoRA?
- **Direct answer:** Quantize base model (e.g., 4-bit NF4) during tuning while training LoRA adapters in 16-bit.

---

# Q39: What is RLHF?
- **Direct answer:** Alignment using preference feedback (reward signal) + optimization (historically PPO).

---

# Q40: What is Constitutional AI?
- **Direct answer:** A rulebook-driven alignment approach using critique/rewrite against a “constitution,” reducing human labeling.

---

# Q41: What is instruction tuning?
- **Direct answer:** Supervised tuning on instruction→response pairs to make base models follow commands reliably.

---

# Q42: What is DPO?
- **Direct answer:** Preference alignment via a stable supervised-style objective on (chosen, rejected) pairs.

---

# Q43: What is PPO?
- **Direct answer:** RL algorithm with clipped updates to prevent destructive policy jumps.

---

# Q44: Why use reward models in RLHF?
- **Direct answer:** Humans can’t label at scale; reward models provide scalable scalar feedback.

---

# Q45: Name 3 metrics used to evaluate LLMs.
- **Direct answer:** Perplexity, task benchmarks (MMLU/HumanEval), judge/human A/B.

---

# Q46: What is perplexity?
- **Direct answer:** How surprised the model is on held-out text:

\[
PPL(X)=\exp\left(-rac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})
ight)
\]

---

# Q47: Why is ROUGE or BLEU often insufficient for evaluating LLMs?
- **Direct answer:** n-gram overlap misses semantic equivalence; use semantic/functional/judge-based evals.
