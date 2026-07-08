---
module: LLMs
topic: Interview Questions
subtopic: ""
status: unread
tags: [llms, interview-questions, revision]
---

# LLM — Interview Questions

**For:** SDE-2 / AI Engineer interviews — calibrated to what's actually asked Round 1 and beyond.
**Difficulty guide:**
- **Easy** → Round 1 basics: definitions, intuition, tokenization basics, RAG pipeline, prompting techniques. Know these cold.
- **Medium** → Round 2 depth: applied debugging, serving optimizations (PagedAttention, continuous batching), RAG evaluation, RLHF/SFT trade-offs, agent architectures.
- **Hard** → Staff/Research depth: derivations, RoPE extrapolation, distributed serving (MLA, KV Cache math), DPO math, multi-billion vector search.

---

## Easy

> Round 1 LLM fundamentals. Basic prompting, tokenization concepts, and high-level RAG.

### Q: What problem does next-token prediction actually solve, and why is it a sufficient training objective for building capable models?
Next-token prediction (`maximize P(x_t | x_1...x_{t-1})`) forces the model to build an internal representation of syntax, facts, reasoning patterns, and world knowledge, because accurately predicting the next token in arbitrary text requires all of these. Predicting the next word in "The capital of France is ___" requires factual knowledge; predicting the next line of a proof requires logical consistency. The objective is trivial to specify and infinitely scalable. The gotcha: the objective optimizes for *plausibility*, not truth. A model minimizing cross-entropy loss will happily produce a fluent, statistically likely, and factually wrong statement — this is the root cause of hallucination.

### Q: Walk through the BPE algorithm and explain why byte-level BPE avoids OOV entirely.
BPE starts with a vocabulary of individual characters (or bytes) and iteratively merges the most frequent adjacent pair into a new token, repeating until the target vocab size is reached. The final vocabulary is a mix of full words, subwords, and single characters. Byte-level BPE operates on the raw UTF-8 byte sequence (256 possible byte values) rather than Unicode characters. Since *every* possible string can be represented as some sequence of bytes, and every byte value is in the base vocabulary, there is no out-of-vocabulary (OOV) token ever, even for emoji or malformed text.

### Q: WordPiece vs BPE — what's the actual algorithmic difference?
Both are greedy subword merge algorithms. BPE merges the pair with the **highest raw frequency** in the corpus. WordPiece merges the pair that **maximizes the likelihood of the training corpus** under a unigram language model (i.e., pairs that co-occur more than chance would predict). Practically, WordPiece (BERT) marks non-initial subword pieces with a `##` prefix so the boundary information is preserved for detokenization; BPE (GPT) instead marks token boundaries via a leading-space convention baked into the token itself.

### Q: Why do LLMs struggle with character-level tasks like counting letters in a word or reversing a string?
Because tokenization abstracts away the character level before the model ever sees the input — the model operates on token embeddings, not characters. A word like "strawberry" may be tokenized into 2-3 subword pieces, and the model has no direct access to the individual character sequence inside a token. Giving the model a character-spaced version of the word ("s t r a w b e r r y") measurably improves accuracy on these tasks.

### Q: What is a context window, and what determines its maximum size for a given transformer model?
The context window is the maximum number of tokens (prompt + generation combined) a model can process in a single forward pass. Its ceiling is set primarily by what the model was trained on: positional encodings are only well-calibrated up to the sequence lengths seen during training, and self-attention's compute/memory cost grows quadratically. Simply feeding more tokens than the model was trained for produces degraded, incoherent output.

### Q: What is the difference between few-shot and zero-shot prompting, and why might few-shot examples sometimes hurt performance?
Zero-shot prompting gives the model only an instruction and the actual input. Few-shot prompting prepends a handful of example input-output pairs before the real query, letting the model infer the task pattern by analogy. Few-shot can hurt when: (1) the examples are unrepresentative or biased (e.g., all positive-sentiment), causing the model to overfit to a spurious pattern; (2) the examples consume context budget better spent on the actual input; (3) for modern chat-tuned LLMs, poorly chosen examples can conflict with the model's more general zero-shot instruction-following behavior.

### Q: What is the "lost in the middle" phenomenon?
LLMs recall information placed at the very beginning or very end of a long context substantially better than information placed in the middle, producing a U-shaped accuracy curve. Practical mitigation: for RAG, place the most critical retrieved chunks at the beginning or end of the context, not buried in the middle.

### Q: Walk through a full RAG pipeline from document ingestion to final answer, naming the failure mode at each stage.
(1) **Chunking** — failure mode: chunks too large dilute embedding relevance, chunks too small lose necessary context. (2) **Embedding** — failure mode: embedding model mismatch with domain. (3) **Indexing/storage** in a vector DB — failure mode: stale index causing retrieval of outdated content. (4) **Query embedding + retrieval** — failure mode: vocabulary mismatch problem (short queries embed poorly relative to prose chunks). (5) **Reranking** — failure mode: relevant-but-not-top-ranked chunks buried below context window. (6) **Context assembly** — failure mode: lost-in-the-middle or context window overflow. (7) **Generation** — failure mode: the model ignores retrieved context or hallucinated ungrounded info.

### Q: How does RAGAS evaluate a RAG pipeline?
RAGAS decomposes RAG quality into independent metrics: **Faithfulness** (does the generated answer's content logically follow from retrieved context); **Answer Relevance** (does the generated answer actually address the user's question); **Context Precision** (fraction of retrieved chunks that were actually relevant); **Context Recall** (fraction of necessary information that was retrieved).

### Q: What is the ReAct pattern, and why does interleaving reasoning and acting outperform pure chain-of-thought or pure tool-calling alone?
ReAct (Reasoning + Acting) interleaves "Thought" steps (reasoning in natural language) with "Action" steps (tool invocation) and "Observation" steps (tool result fed back). Pure chain-of-thought is fully "open-loop" with no way to verify assumptions. Pure tool-calling lacks a deliberate planning step to interpret results. ReAct lets the model course-correct, grounding the reasoning chain in real, verifiable information at each step.

### Q: What is the Model Context Protocol (MCP)?
MCP standardizes the interface between an LLM application (client) and any external tool/data source (server). Instead of M×N bespoke integrations for M agent frameworks to talk to N tools, MCP provides a standard protocol so any compatible client can consume any compatible server's capabilities (resources, tools, prompts).

### Q: Why does "think step by step" (zero-shot CoT) improve accuracy on reasoning tasks?
The phrase cues the model to produce intermediate reasoning tokens before committing to a final answer. Since autoregressive generation is sequential, forcing the model to "show its work" gives it extra sequential computation steps to arrive at a harder answer than it could produce in a single forward pass's worth of implicit computation.

### Q: What distinguishes LLMOps from traditional MLOps?
(1) **Non-deterministic, open-ended outputs** requiring LLM-as-judge or rubric-based evaluation instead of exact-match ground truth. (2) **Prompt as a first-class artifact** requiring independent versioning and testing. (3) **Externally-hosted base models** that can change silently, requiring regression testing. (4) **Cost as a core metric** since token-based pricing scales with prompt/output length.

### Q: What is structured output prompting and why is JSON-mode/function-calling more reliable than instructing "please respond in JSON"?
Natural language instructions to "respond in JSON" can still produce malformed output. Constrained decoding (JSON mode, function-calling) modifies the *decoding process itself*: at each generation step, the set of allowed next tokens is restricted to only those consistent with a valid continuation of the target schema, making malformed output structurally impossible.

### Q: Why is perplexity a poor metric for comparing instruction-tuned chat models?
Perplexity measures how well a model predicts a specific reference text. For chat models, many phrasings can be equally good answers, and higher perplexity just means the model used a different valid phrasing. It doesn't capture instruction-following fidelity, safety, or task success.

---

## Medium

> Round 2 depth — applied debugging, design trade-offs, serving optimizations, and "how would you build this?" questions.

### Q: What is PagedAttention and what specific inefficiency does it solve?
Naive serving pre-allocates a contiguous KV cache buffer sized to the maximum possible sequence length for every incoming request, wasting memory on unused tokens. PagedAttention borrows OS virtual-memory paging: the KV cache is divided into fixed-size blocks (e.g., 16 tokens), allocated lazily as generation proceeds. Benefits: no upfront over-allocation, no fragmentation, and copy-on-write sharing for beam search or shared prefixes. Yields massive throughput increases in frameworks like vLLM.

### Q: Explain continuous (iteration-level) batching and why it outperforms static batching.
Static batching waits until *all* sequences in a batch finish before admitting new ones, wasting GPU compute on padded sequences. Continuous batching makes the admission decision at every decoding step: as soon as any sequence finishes, its slot is immediately freed and a new request is admitted, keeping the GPU's batch dimension continuously full of active work.

### Q: What is prefix caching and when does it provide the largest wins?
Prefix caching stores the KV cache for a shared prompt prefix across multiple requests, avoiding prefill recomputation. It wins big when many requests share a long identical prefix (system prompt, RAG context). Gotcha: it only helps for exact token-for-token matches up to the point of divergence, so prompt engineering means putting *stable* content first and *variable* content last.

### Q: State the Chinchilla scaling law and compute the compute-optimal model size for a 1e24 FLOPs budget.
Chinchilla found that parameter count and token count should be scaled equally for compute-optimal training. Rule of thumb: **~20 training tokens per parameter**. For C = 1e24 (and C ≈ 6ND): `6N(20N) = 1e24` → `120N² = 1e24` → N ≈ 91B parameters, D ≈ 1.8T tokens. Note: this is *training* compute-optimal, not deployment-optimal (Llama 3 was trained far past Chinchilla-optimal to make inference cheaper).

### Q: Explain test-time compute scaling (e.g., OpenAI o1) and its trade-offs.
Test-time compute scaling spends more FLOPs *per query at inference* (via chain-of-thought, best-of-N, or tree search) rather than during training. Trade-off: pretraining compute is a sunk cost amortized over all queries; test-time compute is a recurring marginal cost per query. It's best for high-value, hard reasoning queries (math, coding).

### Q: Why do modern LLMs use RMSNorm and Pre-LN instead of LayerNorm and Post-LN?
RMSNorm drops LayerNorm's mean-centering step, normalizing only by root-mean-square. It's cheaper and performs equivalently. Pre-LN normalizes *before* the sublayer, avoiding the vanishing gradient problem seen in deep networks with Post-LN (where the residual stream magnitude compounds layer over layer).

### Q: What breaks with sliding window attention (SWA) in long contexts?
SWA restricts attention to the nearest W tokens. Theoretically, across L layers, the receptive field grows to L×W. However, information doesn't propagate losslessly layer-to-layer — facts can be diluted or overwritten before reaching final layers, leading to "lost in the middle" failures on long-context needle tasks despite mathematical receptive field coverage.

### Q: Compare MHA, MQA, and GQA for serving efficiency.
- **MHA**: Separate K/V per head (highest memory, highest quality).
- **MQA**: 1 shared K/V head across all query heads (lowest memory, noticeably lower quality).
- **GQA**: K/V heads shared across groups of query heads (e.g., 8 KV heads for 64 Q heads). Massive memory reduction vs MHA with near-MHA quality. Default in Llama 3, Mistral.

### Q: How does KV cache quantization differ from weight quantization?
KV cache memory scales linearly with sequence length × batch size and can dwarf weight memory. Quantizing K and V tensors to int8/int4 on the fly relieves this. Unlike weights, K/V distributions vary per token/layer and have outlier activations, making naive per-tensor quantization riskier; production schemes use dynamic per-channel or per-token scaling.

### Q: Derive LoRA's parameter count savings for a 4096×4096 weight matrix with rank r=8.
Full fine-tuning: 4096 × 4096 = 16.7M parameters.
LoRA: replaces update with ΔW = BA. B is 4096×8, A is 8×4096. 32K + 32K = 65K parameters.
Reduction: 16.7M / 65K = **256× reduction**. At inference, it folds into the base weights with zero latency overhead.

### Q: Explain QLoRA's memory savings and the role of NF4.
QLoRA trains full-precision LoRA adapters on top of a **frozen, 4-bit-quantized** base model. NF4 (NormalFloat4) places quantization bins at the quantiles of a standard normal distribution, matching the empirical near-Gaussian distribution of neural net weights to minimize error compared to uniform int4. Allows fine-tuning 70B models on a single consumer GPU.

### Q: GPTQ vs AWQ vs GGUF — when would you pick each?
- **GPTQ**: Layer-by-layer quantization with Hessian-based error correction. Strong 4-bit accuracy, slower to quantize.
- **AWQ**: Protects a small fraction of salient weight channels based on activation magnitudes. Faster to quantize, comparable accuracy.
- **GGUF**: A file format designed for CPU/Apple Silicon inference (llama.cpp) supporting multiple quantization schemes, not an algorithm itself.

### Q: Why can't SFT alone teach human preferences, and what does RLHF solve?
SFT imitates specific labeled responses (absolute targets). Human preferences are often *comparative* (e.g., "is longer better?"). RLHF decouples this: (1) collect pairwise rankings; (2) train a reward model to predict preferences; (3) optimize the LLM via RL (PPO) to maximize reward. This generalizes the ranking signal across the output space.

### Q: What is reward hacking in RLHF, and how do you detect it?
Reward hacking is when the policy maximizes the reward model's score without satisfying true human preference (e.g., padding responses with verbosity because the reward model learned a spurious length bias). Detect by monitoring output length trends, re-running human eval on held-out samples, and monitoring KL divergence from the reference policy.

### Q: How do you detect and mitigate catastrophic forgetting when fine-tuning?
Detect by evaluating on general benchmarks (MMLU) before/after fine-tuning. PEFT (LoRA) structurally prevents true forgetting because base weights are frozen. For full fine-tuning, use experience replay (mixing 5-15% pretraining data) and a low learning rate.

### Q: Plan-and-Execute vs ReAct for agents?
ReAct plans one step at a time, highly adaptive but expensive/opaque. Plan-and-Execute generates a full multi-step plan upfront, then executes it. Use Plan-and-Execute for knowable sequences where you want human reviewability of the plan before execution. Use ReAct for exploratory tasks where outcomes genuinely change the next step.

### Q: What is prompt injection in an agent, and why is it dangerous?
Prompt injection is when untrusted content (a webpage the agent browsed) contains text crafted as instructions. In a plain chatbot, it just changes text output. In an agent with tools (email, API), it can cause unauthorized real-world actions (data exfiltration). Defense: least privilege, human-in-the-loop for high-consequence actions, and strict separation of instructions vs data.

### Q: Your RAG chatbot is hallucinating despite correct retrieval. Debugging checklist?
1. Verify the chunk is actually relevant, not just topically related.
2. Check context assembly (was the chunk truncated?).
3. Check prompt instructions (is it explicitly told to answer *only* from context?).
4. Test in isolation: feed only the chunk and question. If it still hallucinates, generation is at fault.
5. Use RAGAS Faithfulness scoring to quantify.

### Q: Design an evaluation plan for a domain-specific LLM application (e.g., medical).
Standard benchmarks (MMLU) fail to test production tasks or safety-critical failure modes. Plan: (1) Build a domain-specific golden test set of real representative examples. (2) Define task-specific metrics (factual consistency, omission rate). (3) Safety-specific adversarial test set for tail-risk edge cases. (4) Human-in-the-loop periodic audit.

### Q: How does HNSW (Vector DB) work?
HNSW builds a multi-layer graph. Top layers are sparse for fast coarse navigation; bottom layers are dense for precise nearest-neighbor finding. It achieves logarithmic search speed at the cost of being *approximate*. Tuning `ef_search` trades off speed vs recall.

### Q: Dense vs sparse embeddings for retrieval?
Dense (BGE, OpenAI) captures semantic meaning/synonyms well. Sparse (BM25, SPLADE) excels at exact keyword matching (SKUs, rare terms). Hybrid search runs both and fuses rankings (Reciprocal Rank Fusion) to capture both semantics and precision.

### Q: What is embedding drift and how do you catch it?
Drift happens when corpus terminology evolves away from the embedding model's training distribution, or the model version was silently upgraded. Catch it by monitoring retrieval-quality proxy metrics on a fixed eval query set over time, and tracking embedding model version checksums.

---

## Hard

> Staff-level and Research Engineer depth — derivations, distributed training internals, quantization theory, transformer scaling.

### Q: Derive why scaled dot-product attention divides by √d_k.
If Q and K entries are i.i.d. with mean 0, variance 1, the dot product over d_k terms has variance `d_k`. As d_k grows, the raw dot-product magnitude grows as `√d_k`, pushing pre-softmax logits to large magnitudes. Softmax saturates for large inputs, causing vanishing gradients. Dividing by `√d_k` renormalizes the variance back to 1, keeping gradients informative.

### Q: In a top-2 MoE with 8 experts (Mixtral 8×7B), why is inference compute closer to a 12B dense model rather than 56B?
Total parameters (47B) dictate VRAM footprint. But *active* compute per token only touches the router plus the 2 selected experts. Since each expert is roughly the size of the FFN in a 7B dense model, and only 2 of 8 are used, the active FFN compute is 25% of a full 8×7B model, plus the shared dense components (attention). MoE buys capacity scaling without the full compute cost per token.

### Q: Kaplan (2020) vs. Chinchilla (2022) scaling laws — what did Kaplan get wrong?
Kaplan found model size mattered much more than data size. The error: Kaplan used a fixed learning rate schedule not re-tuned per model size, biasing the fit toward parameters. Chinchilla used properly cosine-annealed LR schedules for every config and found the true optimum is equal scaling of parameters and tokens.

### Q: Are "emergent abilities" of LLMs a real phase transition?
Substantially a measurement artifact. Abilities that look like sudden jumps on discontinuous metrics (exact-match accuracy) often appear as smooth, predictable improvements on continuous metrics (token log-likelihood or edit distance).

### Q: Derive the total KV cache memory for serving LLaMA-3 8B at batch size 32, sequence length 8192, in BF16.
LLaMA 3 8B: 32 layers, GQA with 8 KV heads, head dim 128, BF16 (2 bytes).
`2 (K,V) × 32 (layers) × 8 (heads) × 128 (dim) × 8192 (seq) × 32 (batch) × 2 (bytes)`.
`2 × 32 × 8 × 128 = 65,536` bytes per token.
`65,536 × 8192 = 536 MB` per sequence.
`536 MB × 32 = ~17.1 GB` total KV cache memory.

### Q: What is Multi-Head Latent Attention (MLA) in DeepSeek V2/V3?
MLA compresses K and V into a shared low-rank latent vector before caching, instead of caching separate per-head tensors. At attention time, it up-projects back to per-head K and V. The up-projection folds into the query/output projections, adding zero FLOPs. Yields 5–13× KV cache reduction beyond GQA, enabling massive effective batch sizes.

### Q: Explain the expert-collapse failure mode in MoE and the fix.
A MoE router tends to converge to routing most tokens to a few experts (rich-get-richer), wasting the rest of the capacity. Fix: an auxiliary load-balancing loss `L_balance = α · N · Σ f_i P_i`, where `f_i` is the actual fraction of tokens and `P_i` is the mean router probability. This nudges routing toward uniformity.

### Q: Why does RoPE fail to extrapolate beyond its training length, and how does linear interpolation fix it?
RoPE rotates Q/K by an angle proportional to absolute position `m`. Positions beyond training length `L_train` produce unseen rotation angles, causing attention scores to degrade. Linear interpolation rescales the position index `m → m/s` (where `s = L_target/L_train`), compressing the target range back into the trained angular range. Quality holds up to ~2× extension before degrading.

### Q: NTK-aware scaling vs YaRN for RoPE extrapolation?
Linear interpolation uniformly compresses all frequencies, hurting high-frequency (short-range) resolution. **NTK-aware scaling** scales the RoPE base frequency, stretching low-frequency dimensions much more than high-frequency ones, preserving short-range precision. **YaRN** applies non-uniform per-dimension scaling plus a softmax temperature correction, achieving reliable 16-32× context extension.

### Q: How does StreamingLLM enable "infinite" generation with a fixed-size KV cache?
It observed "attention sinks" — early tokens get disproportionate attention mass regardless of semantic content because softmax needs to dump excess attention somewhere. Naively evicting early tokens from a sliding window destabilizes generation. StreamingLLM always keeps the first ~4 "sink tokens" permanently in the KV cache, preventing perplexity spikes during unbounded streaming generation.

### Q: DPO vs PPO-based RLHF — what's the mathematical trick?
DPO eliminates the reward model and RL loop entirely. The trick: the optimal policy under the RLHF objective has a closed-form relationship to the reward function. Substituting this into the Bradley-Terry preference model allows rewriting the objective directly in terms of the policy's log-probabilities. It trains directly on `(chosen, rejected)` pairs via supervised gradient descent.

### Q: What's the difference between reward hacking and KL explosion in RLHF?
Reward hacking is exploiting the proxy reward model without improving true quality. KL explosion is a mechanical failure where the policy drifts arbitrarily far from the SFT reference distribution (because the KL penalty is too weak), entering regions where the reward model is uncalibrated, causing generation collapse.

### Q: Distinguish alignment tax, reward hacking, and mode collapse.
**Alignment tax**: accepted capability drop (e.g., in reasoning) caused by safety alignment pulling behavior toward a cautious distribution. **Reward hacking**: objective misalignment (optimizing proxy over true). **Mode collapse (in DPO)**: converging to a narrow, low-diversity output distribution because DPO lacks an exploration mechanism (entropy bonus) and over-optimizes toward narrow response patterns that reliably beat rejected examples.

### Q: How do you scale vector search to billions of vectors?
When a full HNSW graph exceeds single-machine RAM: (1) **Sharding**: partition the index across multiple machines and merge top-k results. (2) **Quantization**: product quantization splits and quantizes sub-vectors, shrinking memory 10×+. (3) **DiskANN**: keeps a compact navigational structure in RAM while bulk vector data lives on fast SSDs.
